"""Convert Bullying10K DVS event streams (.npy) into accumulated pseudo-frames.

Bullying10K records each segment with a DVS346 event camera (346 x 260).
Events are stored as .npy — either a structured array with t/x/y/p fields or
a plain (N, 4) array in that column order, with timestamps in microseconds.

Accumulating events over fixed time windows yields image-like frames that
can be fed to the same pose extractor used for the RGB sources
(src/preprocessing/pose_extraction.py), keeping every dataset in one
unified 2D-skeleton representation.

Usage:
    python -m src.preprocessing.dvs_to_frames --input data/bullying10k --output outputs/bullying10k_frames --png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SENSOR_WIDTH = 346
SENSOR_HEIGHT = 260


def load_events(path: Path) -> np.ndarray:
    """Load one segment and return events as a float64 (N, 4) array [t, x, y, p]."""
    raw = np.load(path, allow_pickle=True)
    if raw.dtype.names:
        names = raw.dtype.names
        cols = []
        for candidates in (("t", "timestamp", "ts"), ("x",), ("y",), ("p", "pol", "polarity")):
            field = next((c for c in candidates if c in names), None)
            if field is None:
                raise ValueError(f"{path}: no {candidates[0]!r} field among {names}")
            cols.append(np.asarray(raw[field], dtype=np.float64))
        return np.stack(cols, axis=1)
    events = np.asarray(raw, dtype=np.float64)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"{path}: expected (N, 4) events, got {events.shape}")
    return events


def accumulate(
    events: np.ndarray,
    window_ms: float = 33.0,
    height: int = SENSOR_HEIGHT,
    width: int = SENSOR_WIDTH,
) -> np.ndarray:
    """Bin events into per-window polarity histograms of shape (T, 2, H, W).

    Channel 0 counts ON (positive) events, channel 1 OFF (negative).
    """
    if len(events) == 0:
        return np.zeros((0, 2, height, width), dtype=np.float32)
    t = events[:, 0] - events[:, 0].min()
    frame_idx = (t / (window_ms * 1000.0)).astype(np.int64)  # timestamps are in µs
    x = np.clip(events[:, 1].astype(np.int64), 0, width - 1)
    y = np.clip(events[:, 2].astype(np.int64), 0, height - 1)
    neg = (events[:, 3] <= 0).astype(np.int64)  # OFF events → channel 1

    frames = np.zeros((int(frame_idx.max()) + 1, 2, height, width), dtype=np.float32)
    np.add.at(frames, (frame_idx, neg, y, x), 1.0)
    return frames


def render(frames: np.ndarray) -> np.ndarray:
    """Render polarity histograms as (T, H, W, 3) uint8 RGB images (ON → red, OFF → blue)."""
    images = np.zeros((frames.shape[0], frames.shape[2], frames.shape[3], 3), dtype=np.uint8)
    if frames.size == 0 or not (frames > 0).any():
        return images
    scale = max(float(np.percentile(frames[frames > 0], 99)), 1.0)
    images[..., 0] = np.clip(frames[:, 0] / scale * 255.0, 0, 255)
    images[..., 2] = np.clip(frames[:, 1] / scale * 255.0, 0, 255)
    return images


def convert_file(src: Path, out_dir: Path, window_ms: float, save_png: bool) -> Path:
    images = render(accumulate(load_events(src), window_ms))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src.stem}.npy"
    np.save(out_path, images)
    if save_png:
        import cv2

        png_dir = out_dir / src.stem
        png_dir.mkdir(exist_ok=True)
        for i, img in enumerate(images):
            cv2.imwrite(str(png_dir / f"{i:05d}.png"), img[..., ::-1])  # RGB → BGR
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True, help=".npy event file or directory tree")
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--window-ms", type=float, default=33.0, help="accumulation window (default ≈30 fps)")
    parser.add_argument("--png", action="store_true", help="also dump per-frame PNGs for pose extraction / inspection")
    args = parser.parse_args()

    files = [args.input] if args.input.is_file() else sorted(args.input.rglob("*.npy"))
    if not files:
        raise SystemExit(f"no .npy event files under {args.input}")
    for src in files:
        rel = src.parent.relative_to(args.input) if args.input.is_dir() else Path()
        out = convert_file(src, args.output / rel, args.window_ms, args.png)
        print(f"{src} -> {out}")


if __name__ == "__main__":
    main()
