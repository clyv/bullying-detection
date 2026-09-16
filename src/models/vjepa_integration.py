"""Optional RGB appearance stream: frozen V-JEPA 2 features + a small probe.

The skeleton pipeline throws away appearance entirely. That is the point (privacy),
but the pose-anomaly literature is consistent that pose alone underperforms on hard
real-world footage, and recommends combining it with a second modality. This module
is that second modality, kept strictly optional: nothing in the skeleton path imports
it, and ``transformers`` is not in requirements.txt.

Design notes (the corrections that matter):

* V-JEPA 2's predictor reconstructs masked patches of *the same* clip. Comparing
  ``predictor(clip_t)`` against ``encoder(clip_{t+1})`` is therefore NOT a world-model
  "surprise" signal — it mostly measures how much the scene changed between two
  windows, and fires on doors opening and lights flickering. We do not do that.
  We use the encoder as a frozen feature extractor and train a supervised probe,
  which is how V-JEPA 2 is actually evaluated.
* Clips are read in windows, never by decoding a whole video into RAM.
* The probe outputs logits that go through the same calibration path as the ST-GCN
  (src/evaluation/calibrate.py), so fusion happens between comparable probabilities
  rather than between a logit and an arbitrarily rescaled distance.

Install extras before use:
    pip install transformers torchcodec

Usage:
    python -m src.models.vjepa_integration --self-test
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import nn

DEFAULT_MODEL = "facebook/vjepa2-vitl-fpc64-256"


class AttentiveProbe(nn.Module):
    """Small supervised head over frozen V-JEPA 2 patch embeddings.

    Self-attention pools the spatio-temporal token sequence (V-JEPA 2 emits one
    token per tubelet, not a single CLS vector), then an MLP classifies.
    """

    def __init__(self, embedding_dim: int, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=8, batch_first=True
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, num_tokens, dim) -> (B, num_classes)."""
        attended, _ = self.attention(tokens, tokens, tokens)
        pooled = self.norm(attended.mean(dim=1))
        return self.classifier(pooled)


def sample_window_indices(num_frames: int, window: int) -> np.ndarray:
    """Evenly sample ``window`` frame indices from a clip of ``num_frames``.

    Shorter clips repeat their last frame (matching the edge-padding the skeleton
    loader uses), so a 40-frame clip and a 400-frame clip both produce ``window``
    indices without either being silently truncated to its first seconds.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if num_frames >= window:
        return np.linspace(0, num_frames - 1, window).round().astype(int)
    idx = np.arange(window)
    return np.clip(idx, 0, num_frames - 1)


class VJEPAFeatureExtractor:
    """Frozen V-JEPA 2 encoder. Turns a window of RGB frames into patch embeddings.

    Lazily imports ``transformers`` so importing this module (e.g. during test
    collection) never requires the optional dependency.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        from transformers import AutoModel, AutoVideoProcessor

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # bfloat16 on CUDA only — CPU bf16 matmul support is patchy and slow.
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.processor = AutoVideoProcessor.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(
            model_name, dtype=self.dtype, attn_implementation="sdpa"
        ).to(self.device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.embedding_dim = self.encoder.config.hidden_size

    @torch.no_grad()
    def encode(self, frames) -> torch.Tensor:
        """``frames`` (T, H, W, 3) uint8 RGB -> (1, num_tokens, embedding_dim) float32.

        ``skip_predictor`` avoids running the predictor head we have no use for.
        Only the pixel tensor is cast to the encoder dtype; the processor's other
        outputs (masks, integer position ids) must keep their own dtypes.
        """
        inputs = self.processor(frames, return_tensors="pt")
        inputs = {
            key: (
                value.to(self.device, dtype=self.dtype)
                if torch.is_floating_point(value)
                else value.to(self.device)
            )
            for key, value in inputs.items()
        }
        outputs = self.encoder(**inputs, skip_predictor=True)
        return outputs.last_hidden_state.float()


def iter_video_windows(video_path: str, window: int = 16, stride: int = 16):
    """Yield (start_frame, frames) windows from a video without loading it all.

    Keeps at most ``window`` frames in memory at a time, so an hour of CCTV costs
    the same as a ten-second clip.
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    buffer: list[np.ndarray] = []
    start = 0
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            index += 1
            if len(buffer) == window:
                yield start, np.stack(buffer)
                keep = window - stride
                buffer = buffer[stride:] if keep > 0 else []
                start = index - len(buffer)
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--self-test", action="store_true", help="probe shapes, no download")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if args.self_test:
        # Exercises the probe and the sampling helper without touching the network.
        probe = AttentiveProbe(embedding_dim=1024, num_classes=2)
        tokens = torch.randn(2, 128, 1024)
        logits = probe(tokens)
        print(f"probe output: {tuple(logits.shape)} (expected (2, 2))")
        print(f"window indices from 40 frames: {sample_window_indices(40, 16)}")
        return

    extractor = VJEPAFeatureExtractor(args.model)
    print(f"loaded {args.model}: embedding_dim={extractor.embedding_dim}")


if __name__ == "__main__":
    main()
