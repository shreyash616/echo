#!/usr/bin/env bash
# batch_preprocess.sh
#
# Extracts fma_large.zip one subfolder at a time (~600 MB each), converts to
# float16 mel spectrograms, then deletes the mp3s before moving to the next
# batch. This keeps peak disk usage at ~1 GB of mp3s at a time rather than
# requiring the full 93 GB extracted upfront.
#
# Disk layout:
#   Source zip:      D:/Downloads/fma_large.zip      (94 GB, stays in place)
#   Temp extraction: D:/Downloads/fma_tmp/            (~600 MB at a time, deleted after each batch)
#   Spectrograms:    D:/fma_spectrograms/             (~67 GB total when done, float16)
#
# Usage (from the echo/ project root):
#   bash ml/data/batch_preprocess.sh
#
# Resume-safe: already-processed .npy files are skipped automatically.

set -euo pipefail

ZIP="/d/Downloads/fma_large.zip"
TMP="/d/Downloads/fma_tmp"
SPEC_DIR="/d/fma_spectrograms"
WORKERS=${WORKERS:-8}   # override with: WORKERS=4 bash ml/data/batch_preprocess.sh

mkdir -p "$SPEC_DIR"
mkdir -p "$TMP"

echo "=== Echo FMA batch preprocessor ==="
echo "  zip:           $ZIP"
echo "  tmp:           $TMP"
echo "  spectrograms:  $SPEC_DIR"
echo "  workers:       $WORKERS"
echo ""

# Get the list of numbered subfolders inside the zip (000, 001, ..., 155)
FOLDERS=$(unzip -l "$ZIP" | grep -oP 'fma_large/\d{3}(?=/)' | sort -u)
TOTAL=$(echo "$FOLDERS" | wc -l)
COUNT=0

for folder in $FOLDERS; do
    COUNT=$((COUNT + 1))
    echo "── [$COUNT/$TOTAL] $folder ──────────────────────────────────"

    # How many tracks in this folder are already processed?
    folder_num=$(basename "$folder")
    already=$(ls "$SPEC_DIR"/${folder_num}*.npy 2>/dev/null | wc -l || echo 0)
    expected=$(unzip -l "$ZIP" "$folder/*.mp3" 2>/dev/null | grep -c '\.mp3' || echo 0)

    if [ "$already" -ge "$expected" ] && [ "$expected" -gt 0 ]; then
        echo "  Already done ($already/$expected tracks). Skipping."
        continue
    fi

    # Extract this subfolder only
    echo "  Extracting $folder ($expected tracks)..."
    unzip -q "$ZIP" "$folder/*" -d "$TMP"

    # Preprocess to float16 spectrograms
    echo "  Preprocessing..."
    python ml/data/preprocess.py \
        --audio-dir  "$TMP/$folder" \
        --output-dir "$SPEC_DIR" \
        --workers    "$WORKERS" \
        --fp16

    # Delete the extracted mp3s to free space for the next batch
    rm -rf "$TMP/$folder"
    echo "  Done. Freed $(du -sh "$TMP" 2>/dev/null | cut -f1) temp space."

    # Show progress
    total_specs=$(ls "$SPEC_DIR"/*.npy 2>/dev/null | wc -l || echo 0)
    spec_gb=$(du -sh "$SPEC_DIR" 2>/dev/null | cut -f1 || echo "?")
    d_free=$(df -h /d/ | awk 'NR==2{print $4}')
    echo "  Spectrograms so far: $total_specs  |  Size: $spec_gb  |  D: free: $d_free"
    echo ""
done

rmdir "$TMP" 2>/dev/null || true

echo "=== Preprocessing complete ==="
total_specs=$(ls "$SPEC_DIR"/*.npy 2>/dev/null | wc -l || echo 0)
spec_gb=$(du -sh "$SPEC_DIR" 2>/dev/null | cut -f1)
echo "  Total spectrograms: $total_specs"
echo "  Total size:         $spec_gb"
echo ""
echo "Next step:"
echo "  python ml/training/train.py --spec-dir $SPEC_DIR --output ml/checkpoints --batch 256"
