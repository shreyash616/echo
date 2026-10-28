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

4. **Feel Scoring Pipeline (`backend/app/services/recommender.py`)**:

   * **No Hard Gates / Soft Language Prior**: Echo has no hard genre, artist, or language gates. Recommendations are retrieved via candidate pool union (CLAP top 500 + Feel top 600) and reranked using the feel descriptor table (Section 12.6 of `FEEL_ALGORITHM.md`), with a soft empirical language affinity prior (`language_affinity.json`) and supply-scaled cross-language budget via MMR.

   * **Multi-Axis Feel Scoring**: Ranks candidate pool using continuous audio descriptors:
     * Official LAION CLAP acoustic embedding cosine similarity
     * Anchor vibes profile similarity
     * Felt-tempo distribution overlap with octave invariance
     * Groove and kick-class compatibility
     * Darkness, grit, vocal presence, intimacy, and restraint matching
     * Standardized whole-lyric semantic theme similarity

## 2. Indexing Workflow

The catalog is built from decoded 30s previews described by DSP feature extractors and embedded with official LAION CLAP:

```bash
# Build FAISS index, feel matrix, and metadata bundles
python ml/inference/build_index_v2.py
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