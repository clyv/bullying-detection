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

The real Bullying10K release ships a single COCO-style ``train_keypoints.json``
(``images`` + ``annotations`` + ``categories``) covering every clip, with
Halpe-26 keypoints (78 values/person) whose first 17 joints are COCO-17 in the
same order, and the action class encoded in each image ``file_name`` path. When
``--input`` is (or contains) such a JSON, it is parsed directly; otherwise the
per-clip array path (``coco_to_unified`` on (T, M, 17, 3) / (T, M, 51) .npy/.npz)
is used. Both land in the same unified .npz format.

Note: the JSON is large (~1.7 GB / millions of annotations); parsing loads it
into memory once. Use ``--limit N`` to convert only the first N clips for a
quick check before processing everything.

Usage:
    python -m src.preprocessing.bullying10k_poses --input data/bullying10k --output outputs/b10k_poses
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_HALPE_TO_COCO = 17  # first 17 Halpe-26 joints are COCO-17 in the same order

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


def keypoints_to_coco17(flat) -> np.ndarray:
    """Flat keypoint list (Halpe-26 = 78 vals, or COCO-17 = 51) -> (17, 3) COCO array."""
    arr = np.asarray(flat, dtype=np.float32).reshape(-1, 3)
    return arr[:N_HALPE_TO_COCO]


def _clip_and_frame(file_name: str) -> tuple[str, int]:
    """'punching/.../dvSave-XXXX/12.png' -> ('punching/.../dvSave-XXXX', 12)."""
    parts = file_name.replace("\\", "/").split("/")
    return "/".join(parts[:-1]), int(Path(parts[-1]).stem)


def convert_keypoints_json(json_path: Path, out_dir: Path, max_persons: int = 2, limit=None) -> int:
    """Parse a Bullying10K COCO ``*_keypoints.json`` into one unified .npz per clip.

    Frames are grouped by their clip directory and ordered by frame index; up to
    ``max_persons`` people per frame are kept by detection ``score``. ``limit``
    caps the number of clips converted (for a quick check).
    """
    with open(json_path) as f:
        data = json.load(f)

    id_to_loc = {img["id"]: _clip_and_frame(img["file_name"]) for img in data["images"]}

    # clip -> {frame_index: [(score, (17,3) keypoints), ...]}
    clips: dict[str, dict[int, list]] = {}
    for ann in data["annotations"]:
        loc = id_to_loc.get(ann["image_id"])
        if loc is None:
            continue
        clip, frame = loc
        kp = keypoints_to_coco17(ann["keypoints"])
        clips.setdefault(clip, {}).setdefault(frame, []).append((float(ann.get("score", 1.0)), kp))
    del data, id_to_loc

    out_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    for clip in sorted(clips):
        if limit is not None and converted >= limit:
            break
        frames = clips[clip]
        order = sorted(frames)
        seq = np.zeros((len(order), max_persons, 17, 3), dtype=np.float32)
        for ti, fr in enumerate(order):
            people = sorted(frames[fr], key=lambda sp: sp[0], reverse=True)[:max_persons]
            for pi, (_score, kp) in enumerate(people):
                seq[ti, pi] = kp

        keypoints, scores = coco_to_unified(seq, max_persons)
        label = label_for(clip)
        out_path = out_dir / (clip.replace("/", "__") + ".npz")
        fields = dict(keypoints=keypoints, scores=scores, source=clip)
        if label is not None:
            fields["label"] = label
            fields["label_name"] = BULLYING10K_CLASSES[label]
            fields["aggressive"] = BULLYING10K_CLASSES[label] in AGGRESSIVE
        np.savez_compressed(out_path, **fields)
        print(f"{clip} ({len(order)} frames) -> {out_path}")
        converted += 1
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="*_keypoints.json, or a dir/file of .npy/.npz poses",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="output directory for unified .npz files"
    )
    parser.add_argument("--max-persons", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="convert only the first N clips")
    args = parser.parse_args()

    # JSON route: --input is a *_keypoints.json, or a directory containing one.
    json_files = []
    if args.input.is_file() and args.input.suffix == ".json":
        json_files = [args.input]
    elif args.input.is_dir():
        json_files = sorted(args.input.glob("*keypoints*.json"))
    if json_files:
        total = sum(
            convert_keypoints_json(jp, args.output, args.max_persons, args.limit)
            for jp in json_files
        )
        print(f"converted {total} clips from {len(json_files)} json file(s)")
        return

    # Per-clip array route.
    if args.input.is_file():
        files = [args.input]
    else:
        files = sorted(p for p in args.input.rglob("*") if p.suffix in (".npy", ".npz"))
    if not files:
        raise SystemExit(f"no *_keypoints.json or .npy/.npz pose files under {args.input}")

    converted = 0
    for src in files:
        rel = src.parent.relative_to(args.input) if args.input.is_dir() else Path()
        out = convert_file(src, args.output / rel, args.max_persons)
        print(f"{src} -> {out}")
        converted += 1
    print(f"converted {converted} files")


if __name__ == "__main__":
    main()
