# Echo — Setup & Architecture Guide

## Project Structure

| Path | Component | Role | 
| ----- | ----- | ----- | 
| `app/` | React Native (Expo) | iPhone app (`HomeScreen`, `ResultsScreen`, `SongCard`) | 
| `backend/` | FastAPI | REST API server, ONNX + FAISS inference, self-generated audio metrics, and genre/language matching | 
| `ml/` | PyTorch + ONNX + FAISS | Self-supervised `CnnMusicEncoder` training, raw audio harvesting, and FAISS index builder | 

## 1. Self-Generated Audio Metrics & ML Architecture

Echo computes all musical, emotional, and similarity metrics directly from the raw audio signal without relying on external pre-computed features:

1. **Fixed-Floor Spectrogram Scaling (`ml/data/preprocess.py` & `backend/app/services/recommender.py`)**:

   * Audio is converted to 128-bin mel spectrograms (`sr=22050`, `n_fft=2048`, `hop_length=512`) using a fixed `top_db = 80.0` floor mapped from `[-80 dB, 0 dB]` to `[0.0, 1.0]`.

   * Unlike per-clip min-max normalization, this preserves true loudness contrast and dynamic range across both full-length tracks and 30-second previews.

2. **`CnnMusicEncoder` + 16-Feature `EmotionFeatureExtractor` (`ml/models/music_encoder.py`)**:

   * **Acoustic Branch**: 5 `ConvBlock` layers followed by both mean and temporal standard deviation pooling (`1024-d`) to capture both timbral texture and temporal modulation.

   * **Emotion & Rhythm Branch (`16` differentiable features** $\rightarrow$ **`64-d` L2-normalized)**:

     * *Energy & Dynamics*: `rms_mean`, `rms_std`, and robust 90th-to-10th percentile `dynamic_range` (via `torch.topk`).

     * *Brightness & Timbre*: Normalized `centroid_mean`, `centroid_std`, `rolloff_mean`, `contrast_mean`, `contrast_std`, `hf_ratio`, and `bass_ratio`.

     * *Adrenaline & Envelope*: `flux_mean`, `flux_std`, and length-normalized `energy_entropy` (divided by $\ln(T)$ so it stays invariant across variable clip lengths).

     * *Rhythm, Beat & Tempo*: `onset_density`, multi-lag autocorrelation `pulse_clarity`, and `tempo_fast_ratio` (fast 125–195 BPM vs. slow 70–115 BPM pulse energy).

3. **Self-Generated Per-Track Analytics (`ml/inference/audio_features.py` & `backend/app/services/recommender.py`)**:

   * **BPM (`bpm`)**: Estimated via onset-strength autocorrelation with perceptual tempo weighting (`65–185 BPM`) and parabolic sub-frame interpolation.

   * **Musical Key & Mode (`key`)**: Estimated by projecting mel frequencies onto the 12 semitone pitch classes (`C` through `B`) and correlating against Krumhansl-Schmuckler Major and Minor key profiles (e.g., `"A Minor"`, `"C# Major"`).

   * **Energy (`energy`)**: Computed in `[0, 1]` from RMS loudness, spectral flux intensity, beat pulse clarity, and tempo.

   * **Valence / Mood (`valence`)**: Computed in `[0, 1]` from Major/Minor harmonic mode polarity, spectral centroid brightness, tempo bounce, and pulse clarity.

   * **Vibe Tags (`vibes`)**: Automatically derived from the track's computed acoustic and emotional dimensions (`"adrenaline"`, `"energetic"`, `"driving"`, `"soothing"`, `"dark"`, `"uplifting"`, `"melodic"`, `"bass-heavy"`, etc.).

4. **Composite Similarity Scoring & Strict Genre + Language Matching (`backend/app/api/routes.py`)**:

   * **Genre & Language Gate**: Detects the source song's language and canonical genre family using Deezer album genres, Last.fm artist tags, Unicode script detection (Devanagari, Gurmukhi, Tamil, Telugu, Hangul, Kana, Arabic, Cyrillic), and Deezer's `/artist/{id}/related` scene graph—avoiding statistical `langdetect` false positives on short English titles.

   * **Composite Score**: Ranks genre- and language-matched candidates using:

     * `65%` `CnnMusicEncoder` multi-crop neural cosine similarity,

     * `15%` self-computed `(energy, valence)` proximity,

     * `10%` self-computed `bpm` tempo compatibility (with double-time / half-time octave equivalence),

     * `10%` self-computed `key` compatibility on the Circle of Fifths.

## 2. Training & Indexing Workflows

### Option A: Harvest Raw 30s Previews of Recognizable Mainstream Songs (Recommended)

To train and build your FAISS index on real commercial songs while still generating 100% of the spectrograms, embeddings, and audio metrics yourself:

```
# 1. Harvest raw 30s preview audio & extract fixed-floor .npy spectrograms
python ml/data/harvest_deezer_previews.py \
    --output-specs ml/data/deezer_spectrograms \
    --output-json  ml/data/deezer_catalog.json \
    --tracks-per-query 50

# 2. Train CnnMusicEncoder with emotion-preserving contrastive augmentations
python ml/training/train.py \
    --spec-dir ml/data/deezer_spectrograms \
    --output   ml/checkpoints \
    --epochs   60 \
    --batch    256

# 3. Export trained encoder to ONNX
python ml/inference/export_onnx.py \
    --checkpoint ml/checkpoints/best_model.pt \
    --output     ml/inference/music_encoder.onnx

# 4. Build FAISS index with multi-crop pooling & self-generated audio metrics
python ml/inference/build_index.py \
    --spec-dir     ml/data/deezer_spectrograms \
    --catalog-json ml/data/deezer_catalog.json \
    --onnx-model   ml/inference/music_encoder.onnx \
    --output-dir   ml/inference

```

### Option B: Train & Index on FMA Audio

```
# 1. Preprocess FMA audio to fixed-floor mel spectrograms
python ml/data/preprocess.py \
    --audio-dir  ml/data/fma_large/fma_large \
    --output-dir ml/data/spectrograms \
    --sr         22050

# 2. Train CnnMusicEncoder
python ml/training/train.py \
    --spec-dir ml/data/spectrograms \
    --output   ml/checkpoints \
    --epochs   60 \
    --batch    256

# 3. Export to ONNX
python ml/inference/export_onnx.py \
    --checkpoint ml/checkpoints/best_model.pt \
    --output     ml/inference/music_encoder.onnx

# 4. Build FAISS index (always pass --tracks-csv so titles/artists/genres are populated)
python ml/inference/build_index.py \
    --spec-dir   ml/data/spectrograms \
    --tracks-csv ml/data/fma_metadata/fma_metadata/tracks.csv \
    --onnx-model ml/inference/music_encoder.onnx \
    --output-dir ml/inference

```

## 3. Running the Backend & Mobile App

```
# Start FastAPI backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Start Expo iOS/Android app
cd ../app
npm install
npx expo start

```