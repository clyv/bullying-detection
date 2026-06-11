"""Parse NTU RGB+D 120 .skeleton files into unified 2D skeleton sequences.

Each .skeleton file is a plain-text dump of Kinect v2 body tracking:
frame count, then per frame the body count, one body-info line per body,
the joint count (25), and one line per joint:

    x y z depthX depthY colorX colorY orientW orientX orientY orientZ trackingState

Two routes to 2D (CCTV-style) coordinates:
- "color" (default): use the colorX/colorY columns — Kinect's own projection
  of each joint into the 1920x1080 RGB frame.
- "project": pinhole-project the 3D camera-space joints with approximate
  Kinect v2 colour intrinsics (basis for virtual-view augmentation later).

Output is one .npz per file, matching pose_extraction.py:
    keypoints (T, M, 17, 2) float32 — COCO-17 order, pixel coordinates
    scores    (T, M, 17)    float32 — 1.0 where tracked, 0.0 where absent

Usage:
    python -m src.preprocessing.ntu_skeleton --input data/ntu/skeletons --output outputs/ntu_poses --classes relevant
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

N_KINECT_JOINTS = 25

# Kinect v2 joint index feeding each COCO-17 slot. Kinect has no eye/ear
# joints, so COCO slots 1-4 reuse the head joint and get their score zeroed.
KINECT25_TO_COCO17 = np.array([3, 3, 3, 3, 3, 4, 8, 5, 9, 6, 10, 12, 16, 13, 17, 14, 18])
COCO_FACE_SLOTS = [1, 2, 3, 4]

# NTU action ids relevant to aggression detection (mutual actions unless noted)
AGGRESSIVE = {50, 51, 52, 106, 107, 108, 109, 111}
#   A50 punch/slap, A51 kicking, A52 pushing, A106 hit with object,
#   A107 wield knife, A108 knock over, A109 grab stuff, A111 step on foot
SUBTLE = {54, 93, 116, 117}
#   A54 point finger, A93 shake fist (single-person), A116 follow, A117 whisper
NEUTRAL = {53, 55, 58, 112, 118, 119}
#   A53 pat on back, A55 hugging, A58 handshake, A112 high-five,
#   A118 exchange things, A119 support somebody
RELEVANT = AGGRESSIVE | SUBTLE | NEUTRAL


def parse_name(stem: str) -> dict[str, int]:
    """'S001C002P003R002A050' → setup/camera/performer/replication/action ids."""
    keys = ("setup", "camera", "performer", "replication", "action")
    return {key: int(stem[i + 1 : i + 4]) for key, i in zip(keys, range(0, 20, 4))}


def parse_skeleton_file(path: Path) -> list[list[dict]]:
    """Return per-frame lists of bodies, each {'body_id': str, 'joints': (25, 12) array}."""
    tokens = iter(Path(path).read_text().split())
    frames = []
    for _ in range(int(next(tokens))):
        bodies = []
        for _ in range(int(next(tokens))):
            body_id = next(tokens)
            for _ in range(9):  # clipedEdges, hand confidences/states, isRestricted, lean x/y, trackingState
                next(tokens)
            n_joints = int(next(tokens))
            joints = np.array(
                [float(next(tokens)) for _ in range(n_joints * 12)], dtype=np.float64
            ).reshape(n_joints, 12)
            bodies.append({"body_id": body_id, "joints": joints})
        frames.append(bodies)
    return frames


def _motion(track: np.ndarray) -> float:
    """Total 3D positional variance of a body track — ghost skeletons score near zero."""
    pts = track[:, :, :3]
    pts = pts[~np.isnan(pts).any(axis=(1, 2))]
    return float(pts.var()) if len(pts) else 0.0


def to_sequence(
    frames: list[list[dict]],
    max_bodies: int = 2,
    mode: str = "color",
    fx: float = 1100.0,  # approximate Kinect v2 colour intrinsics; "color" mode
    fy: float = 1100.0,  # uses the projection baked into the file instead
    cx: float = 960.0,
    cy: float = 540.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack parsed bodies into (T, M, 17, 2) keypoints and (T, M, 17) scores.

    Kinect occasionally tracks spurious "ghost" bodies; the max_bodies kept
    are those with the highest joint-position variance over the clip.
    """
    n_frames = len(frames)
    tracks: dict[str, np.ndarray] = {}
    for t, bodies in enumerate(frames):
        for body in bodies:
            track = tracks.setdefault(
                body["body_id"], np.full((n_frames, N_KINECT_JOINTS, 12), np.nan)
            )
            track[t] = body["joints"]
    ranked = sorted(tracks.values(), key=_motion, reverse=True)[:max_bodies]

    keypoints = np.zeros((n_frames, max_bodies, 17, 2), dtype=np.float32)
    scores = np.zeros((n_frames, max_bodies, 17), dtype=np.float32)
    for m, joints in enumerate(ranked):
        if mode == "color":
            xy = joints[:, :, 5:7]
        else:
            xyz = joints[:, :, 0:3]
            z = np.where(np.abs(xyz[:, :, 2]) < 1e-6, np.nan, xyz[:, :, 2])
            u = fx * xyz[:, :, 0] / z + cx
            v = cy - fy * xyz[:, :, 1] / z  # camera y points up, image v points down
            xy = np.stack([u, v], axis=-1)
        coco_xy = xy[:, KINECT25_TO_COCO17]
        valid = ~np.isnan(coco_xy).any(axis=-1)
        scores[:, m] = valid.astype(np.float32)
        scores[:, m, COCO_FACE_SLOTS] = 0.0  # no real eye/ear joints on Kinect
        keypoints[:, m] = np.nan_to_num(coco_xy)
    return keypoints, scores


def convert_file(src: Path, out_dir: Path, mode: str) -> Path:
    keypoints, scores = to_sequence(parse_skeleton_file(src), mode=mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src.stem}.npz"
    np.savez_compressed(
        out_path,
        keypoints=keypoints,
        scores=scores,
        action=parse_name(src.stem)["action"],
        source=str(src),
        frame_size=(1920, 1080),
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True, help=".skeleton file or directory")
    parser.add_argument("--output", type=Path, required=True, help="output directory for .npz files")
    parser.add_argument("--mode", choices=("color", "project"), default="color")
    parser.add_argument(
        "--classes",
        choices=("relevant", "aggressive", "all"),
        default="relevant",
        help="which NTU action classes to convert",
    )
    args = parser.parse_args()

    keep = {"relevant": RELEVANT, "aggressive": AGGRESSIVE, "all": None}[args.classes]
    files = [args.input] if args.input.is_file() else sorted(args.input.rglob("*.skeleton"))
    if not files:
        raise SystemExit(f"no .skeleton files under {args.input}")
    converted = 0
    for src in files:
        if keep is not None and parse_name(src.stem)["action"] not in keep:
            continue
        print(f"{src} -> {convert_file(src, args.output, args.mode)}")
        converted += 1
    print(f"converted {converted}/{len(files)} files")


if __name__ == "__main__":
    main()
