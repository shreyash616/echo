"""
Export the trained CnnMusicEncoder checkpoint to ONNX.

The ONNX model accepts variable-length mel spectrograms (dynamic T axis) so
the backend can pass any clip length without resampling to a fixed size.

Usage:
    python ml/inference/export_onnx.py \
        --checkpoint ml/checkpoints/best_model.pt \
        --output     ml/inference/music_encoder.onnx \
        --crop-frames 256
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.models.music_encoder import CnnMusicEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True, help="Path to best_model.pt")
    p.add_argument("--output",        default="ml/inference/music_encoder.onnx")
    p.add_argument("--embedding-dim", type=int, default=512)
    p.add_argument("--crop-frames",   type=int, default=256,
                   help="T used for the dummy input (must match training crop)")
    args = p.parse_args()

    device = torch.device("cpu")
    ckpt   = torch.load(args.checkpoint, map_location=device)

    # Prefer architecture dims stored inside the checkpoint over CLI defaults
    saved = ckpt.get("args", {})
    embedding_dim = saved.get("embedding_dim", args.embedding_dim)
    emotion_dim   = saved.get("emotion_dim", 64)

    model = CnnMusicEncoder(embedding_dim=embedding_dim, emotion_dim=emotion_dim)
    model.load_state_dict(ckpt["model"])
    model.eval()

    logger.info("Loaded checkpoint from %s", args.checkpoint)
    logger.info(
        "Architecture: embedding_dim=%d  emotion_dim=%d",
        embedding_dim, emotion_dim,
    )
    logger.info("Parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    # Dummy input: (batch=1, channels=1, mels=128, time=crop_frames)
    dummy = torch.randn(1, 1, 128, args.crop_frames)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["mel_spectrogram"],
        output_names=["embedding"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch_size", 3: "time_frames"},
            "embedding":       {0: "batch_size"},
        },
    )
    logger.info("ONNX model exported to %s", output_path)

    # Verify
    import onnxruntime as ort

    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    out  = sess.run(None, {"mel_spectrogram": dummy.numpy()})
    emb  = out[0]
    norm = float(np.linalg.norm(emb[0]))
    logger.info(
        "Verification: output shape=%s  L2-norm=%.4f (should be ~1.0)",
        emb.shape, norm,
    )
    assert abs(norm - 1.0) < 0.01, f"Embedding not normalised! norm={norm}"
    logger.info("Export successful.")


if __name__ == "__main__":
    main()
