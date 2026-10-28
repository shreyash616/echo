# Echo — Setup & Architecture Guide

## Project Structure

| Path | Component | Role | 
| ----- | ----- | ----- | 
| `app/` | React Native (Expo) | iPhone app (`HomeScreen`, `ResultsScreen`, `SongCard`) | 
| `backend/` | FastAPI | REST API server, CLAP + FAISS inference, self-generated audio metrics, and feel-space matching | 
| `ml/` | PyTorch + FAISS | Official LAION CLAP embeddings, DSP audio harvesting, and FAISS index builder | 

## 1. Self-Generated Audio Metrics & ML Architecture

Echo computes all musical, emotional, and similarity metrics directly from the raw audio signal without relying on external pre-computed features:

1. **Decoded Audio Processing (`backend/app/audio/decode.py` & `features.py`)**:
   * Previews are decoded directly to 22,050 Hz mono float32 arrays.
   * DSP extractors compute continuous tempo posterior, kick-class, grit, intimacy, restraint, darkness, and groove directly from the audio envelope and spectrograms.

2. **Official LAION CLAP (`backend/app/services/recommender.py`)**:
   * Official LAION CLAP `630k-audioset-fusion-best` model running with 10s consecutive window pooling, producing 512-d L2-normalized acoustic representations.

3. **Feel Scoring Pipeline (`backend/app/services/recommender.py`)**:
   * **No Hard Gates / Soft Language Prior**: Echo has no hard genre, artist, or language gates. Recommendations are retrieved via candidate pool union (CLAP top 500 + Feel top 600) and reranked using the feel descriptor table (Section 12.6 of `FEEL_ALGORITHM.md`), with a soft empirical language affinity prior (`language_affinity.json`) and supply-scaled cross-language budget via MMR.
   * **Multi-Axis Feel Scoring**: Ranks candidate pool using continuous audio descriptors:
     * Official LAION CLAP acoustic embedding cosine similarity
     * Anchor vibes profile similarity
     * Felt-tempo distribution overlap with octave invariance (Section 11.5.1)
     * Groove and kick-class compatibility
     * Darkness, grit, vocal presence, intimacy, and restraint matching
     * Standardized whole-lyric semantic theme similarity

## 2. Indexing Workflow

The catalog is built from decoded 30s previews described by DSP feature extractors and embedded with official LAION CLAP:

```bash
# Build FAISS index, feel matrix, and metadata bundles
py ml/inference/build_index_v2.py
```

## 3. Running the Backend & Mobile App

```bash
# Start FastAPI backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Start Expo iOS/Android app
cd ../app
npm install
npx expo start
```
