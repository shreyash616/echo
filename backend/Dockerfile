FROM python:3.13-slim

# System deps: fpcalc (Chromaprint) for AcoustID fingerprinting, ffmpeg for MP3 decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libchromaprint-tools \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps laion_clap==1.1.7

# Copy backend source
COPY . .

# ML artifact paths — populated at startup by startup.sh
RUN mkdir -p /app/ml/inference
ENV FAISS_INDEX_PATH=/app/ml/inference/music_index.faiss
ENV TRACK_METADATA_PATH=/app/ml/inference/track_metadata.json

EXPOSE 7860

RUN chmod +x /app/startup.sh
CMD ["/app/startup.sh"]
