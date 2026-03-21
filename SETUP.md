# Echo — Setup Guide

## Project Structure

```
echo/
├── app/         React Native (Expo) — iPhone app
├── backend/     FastAPI — REST API server
└── ml/          Custom CNN training pipeline + FAISS index builder
```

---

## 1. Get API Keys (all free)

### Chromaprint — fpcalc (audio fingerprinting, open source)
Install the `fpcalc` CLI so the backend can generate audio fingerprints:
- **Linux:** `sudo apt install libchromaprint-tools`
- **macOS:** `brew install chromaprint`
- **Windows:** download from https://acoustid.org/chromaprint

### AcoustID (song lookup, free & open source)
1. Sign up at https://acoustid.org/login
2. Register an application → copy the API key

### Spotify Web API (display metadata only — not used for ML features)
1. Go to https://developer.spotify.com/dashboard
2. Create an app → copy Client ID and Client Secret

---

## 2. Backend Setup

```bash
cd echo/backend

# Create virtual env
python -m venv venv
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env with your AcoustID + Spotify keys

# Start the API server (requires FAISS index built in step 3)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## 3. ML Pipeline — Train the Model & Build the FAISS Index

The pipeline trains a custom CNN (CnnMusicEncoder) on mel spectrograms from FMA
audio files, exports it to ONNX, and builds a FAISS similarity index.
No pretrained models. No Spotify features.

### Step 1: Download FMA audio + metadata

Download from https://github.com/mdeff/fma:
- Audio:    `fma_small.zip`  (~8 GB, 8k tracks) or `fma_medium.zip` (~22 GB)
- Metadata: `fma_metadata.zip`

Extract audio to `ml/data/fma_small/` and metadata to `ml/data/fma_metadata/`.

### Step 2: Install ML dependencies

```bash
cd echo

# PyTorch with CUDA (RTX 3080)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Remaining ML deps
pip install -r ml/requirements.txt
```

### Step 3: Preprocess audio → mel spectrograms

```bash
python ml/data/preprocess.py \
    --audio-dir  ml/data/fma_small \
    --output-dir ml/data/spectrograms \
    --duration   10.0 \
    --sr         22050
```

Output: `ml/data/spectrograms/<track_id>.npy` — one (128, T) float32 array per track.

### Step 4: Train CnnMusicEncoder

```bash
python ml/training/train.py \
    --spec-dir   ml/data/spectrograms \
    --tracks-csv ml/data/fma_metadata/tracks.csv \
    --output     ml/checkpoints \
    --epochs     60 \
    --batch      64
```

Outputs:
- `ml/checkpoints/best_model.pt`  — best checkpoint (lowest val triplet loss)
- `ml/checkpoints/train_meta.json` — crop_frames + embedding_dim for inference

Training logs stream to stdout and TensorBoard (`ml/checkpoints/tb_logs/`).

### Step 5: Export to ONNX

```bash
python ml/inference/export_onnx.py \
    --checkpoint ml/checkpoints/best_model.pt \
    --output     ml/inference/music_encoder.onnx
```

The ONNX model accepts a dynamic T axis so any clip length can be passed at inference time.

### Step 6: Build FAISS similarity index

```bash
python ml/inference/build_index.py \
    --spec-dir   ml/data/spectrograms \
    --tracks-csv ml/data/fma_metadata/tracks.csv \
    --onnx-model ml/inference/music_encoder.onnx \
    --output-dir ml/inference \
    --batch      64
```

Outputs:
- `ml/inference/music_index.faiss`   — vector similarity index
- `ml/inference/track_metadata.json` — song info for results
- `ml/inference/embeddings.npy`      — (N, 512) embeddings for debugging

---

## 4. iOS App Setup

```bash
cd echo/app

# Install dependencies
npm install

# Install Expo CLI globally (if not already)
npm install -g expo-cli

# Start development server
npx expo start
```

### Test on iPhone 16 Plus
1. Install **Expo Go** from the App Store on your iPhone.
2. Make sure your PC and iPhone are on the **same Wi-Fi network**.
3. Scan the QR code shown in the terminal with the iPhone camera.
4. Update `BASE_URL` in `app/src/services/api.ts` to your PC's local IP:
   ```ts
   const BASE_URL = 'http://192.168.x.x:8000';
   ```
   Find your IP: `ipconfig` → look for "IPv4 Address" under your Wi-Fi adapter.

---

## 5. App Workflows

### Tap to Identify (mic recording)
```
User taps record → mic captures ~10s of ambient audio
        ↓
POST /api/identify
  ├── AcoustID fingerprint → "Song X by Artist Y" (for display)
  └── librosa mel spec → CnnMusicEncoder ONNX → 512-dim embedding
                                    ↓
                             FAISS similarity search
                                    ↓
Returns: identified track (Spotify metadata) + similar song recommendations
```

### Search by Name (text search)
```
User types a song or artist name
        ↓
GET /api/search?q=...
  → Spotify search → list of matching tracks displayed
        ↓
User taps a track
        ↓
GET /api/recommendations/{track_id}
  → Download Spotify 30s preview MP3 (free, no extra auth)
  → librosa mel spectrogram → CnnMusicEncoder ONNX → 512-dim embedding
  → FAISS similarity search
        ↓
Returns: similar song recommendations based on the actual sound of the track

Note: ~20% of Spotify tracks have no preview URL (licensing restrictions).
For those, a 404 is returned — the user can pick a different track.
```

---

## 6. ML Architecture

```
Raw audio (mic clip or Spotify 30s preview)
         ↓
  librosa mel spectrogram  (128 bins, 10s, 22 050 Hz)
         ↓
  CnnMusicEncoder (custom CNN, trained on FMA with TripletLoss)
    Conv blocks: 1 → 32 → 64 → 128 → 256 → 512 channels
    AdaptiveAvgPool2d → Linear projection
         ↓
  512-dimensional embedding (L2-normalised, unit sphere)
         ↓
  FAISS IndexFlatIP — cosine similarity search
         ↓
  Top-k similar tracks from FMA index
```

**Encoder:** CnnMusicEncoder — custom CNN trained from scratch on FMA genre labels.
**Loss:** TripletLoss with online hard mining (same genre = closer, diff genre = farther).
**Index:** FAISS inner-product search (= cosine similarity on L2-normalised vectors).
**Inference format:** ONNX — no PyTorch dependency in the backend.
**Speed:** ~50ms encode on CPU; <5ms FAISS search.
