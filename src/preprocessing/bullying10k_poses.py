"""Convert Bullying10K's provided pose labels into unified 2D skeleton sequences.

Bullying10K ships COCO-format pose annotations alongside its DVS event
segments. Using those directly is the simpler of the two Phase 2 routes (the
other being events -> pseudo-frames via dvs_to_frames.py -> pose_extraction.py).
Both land in the same unified .npz format the model trains on.

Output is one .npz per clip, matching ntu_skeleton.py / pose_extraction.py:
    keypoints (T, M, 17, 2) float32 — COCO-17 order, pixel coordinates
    scores    (T, M, 17)    float32 — per-joint confidence (0 where absent)
    label     int           — Bullying10K action index (see BULLYING10K_CLASSES)
    label_name str
    aggressive bool          — True for the six physical-aggression actions

The COCO keypoint convention is 17 joints of (x, y, v) where v==0 means the
joint is unlabelled. Bullying10K's exact on-disk layout (per-clip .npy/.npz of
shape (T, M, 17, 3) or a flattened (T, M, 51)) should be confirmed against the
downloaded release; the core conversion lives in ``coco_to_unified`` and is
format-agnostic.

Usage:
    python -m src.preprocessing.bullying10k_poses --input data/bullying10k --output outputs/b10k_poses
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Canonical Bullying10K action order: six aggressive, then four non-violent.
BULLYING10K_CLASSES = [
    "slapping",
    "punching",
    "kicking",
    "strangling",
    "hair_grab",
    "pushing",
    "walking",
    "greeting",
    "handshake",
    "finger_guessing",
]
AGGRESSIVE = set(BULLYING10K_CLASSES[:6])

# Keyword -> class index, with a few aliases for naming variation in the release.
_LABEL_ALIASES = {
    "slap": "slapping",
    "punch": "punching",
    "kick": "kicking",
    "strangle": "strangling",
    "strangling": "strangling",
    "hair": "hair_grab",
    "hairgrab": "hair_grab",
    "push": "pushing",
    "walk": "walking",
    "greet": "greeting",
    "shake": "handshake",
    "handshake": "handshake",
    "finger": "finger_guessing",
    "guess": "finger_guessing",
}
CLASS_TO_IDX = {name: i for i, name in enumerate(BULLYING10K_CLASSES)}


def label_for(name: str) -> int | None:
    """Map a clip path/name to a Bullying10K class index, or None if unknown."""
    stem = name.lower()
    for keyword, canonical in _LABEL_ALIASES.items():
        if keyword in stem:
            return CLASS_TO_IDX[canonical]
    for canonical in BULLYING10K_CLASSES:
        if canonical in stem:
            return CLASS_TO_IDX[canonical]
    return None


def coco_to_unified(keypoints: np.ndarray, max_persons: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """COCO keypoints -> (T, M, 17, 2) coords + (T, M, 17) scores.

    Accepts (T, M, 17, 3) or flattened (T, M, 51). The third value per joint is
    the COCO visibility/confidence; joints with v==0 get a zero score and zeroed
    coordinates. Person slots are padded/truncated to ``max_persons``.
    """
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 51:
        arr = arr.reshape(arr.shape[0], arr.shape[1], 17, 3)
    if arr.ndim != 4 or arr.shape[2:] != (17, 3):
        raise ValueError(f"expected (T, M, 17, 3) or (T, M, 51) COCO keypoints, got {arr.shape}")

    T, M = arr.shape[:2]
    keypoints_out = np.zeros((T, max_persons, 17, 2), dtype=np.float32)
    scores_out = np.zeros((T, max_persons, 17), dtype=np.float32)
    m = min(M, max_persons)
    coords = arr[:, :m, :, :2]
    vis = arr[:, :m, :, 2]
    visible = vis > 0
    keypoints_out[:, :m] = coords * visible[..., None]
    # COCO v is {0,1,2}; map to a [0,1] confidence (1 and 2 -> ~0.5 and 1.0).
    scores_out[:, :m] = np.clip(vis / 2.0, 0.0, 1.0)
    return keypoints_out, scores_out


def load_pose_file(path: Path) -> np.ndarray:
    """Load a per-clip COCO pose annotation (.npy or .npz with a 'keypoints' key)."""
    if path.suffix == ".npz":
        with np.load(path) as data:
            key = "keypoints" if "keypoints" in data else data.files[0]
            return data[key]
    return np.load(path, allow_pickle=False)


def convert_file(src: Path, out_dir: Path, max_persons: int = 2) -> Path:
    keypoints, scores = coco_to_unified(load_pose_file(src), max_persons)
    label = label_for(str(src))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src.stem}.npz"
    fields = dict(keypoints=keypoints, scores=scores, source=str(src))
    if label is not None:
        fields["label"] = label
        fields["label_name"] = BULLYING10K_CLASSES[label]
        fields["aggressive"] = BULLYING10K_CLASSES[label] in AGGRESSIVE
    np.savez_compressed(out_path, **fields)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="pose .npy/.npz file or directory tree"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="output directory for unified .npz files"
    )
    parser.add_argument("--max-persons", type=int, default=2)
    args = parser.parse_args()

    if args.input.is_file():
        files = [args.input]
    else:
        files = sorted(p for p in args.input.rglob("*") if p.suffix in (".npy", ".npz"))
    if not files:
        raise SystemExit(f"no .npy/.npz pose files under {args.input}")

    converted = 0
    for src in files:
        rel = src.parent.relative_to(args.input) if args.input.is_dir() else Path()
        out = convert_file(src, args.output / rel, args.max_persons)
        print(f"{src} -> {out}")
        converted += 1
    print(f"converted {converted} files")


if __name__ == "__main__":
    main()
