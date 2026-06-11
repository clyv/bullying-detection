"""Extract unified 2D pose sequences from RGB sources with YOLO-Pose.

Works on UT-Interaction / RWF-2000 videos, directories of frames (e.g. the
pseudo-frames produced by dvs_to_frames.py --png), and eventually CCTV
footage — using the same extractor everywhere keeps the skeleton domain
consistent across training and deployment.

Output is one .npz per clip, matching ntu_skeleton.py:
    keypoints (T, M, 17, 2) float32 — COCO-17 order, pixel coordinates
    scores    (T, M, 17)    float32 — per-joint confidence (0 where absent)

Person slots are kept temporally consistent with greedy centroid matching,
which is adequate for fixed-camera two-person clips; swap in a real tracker
(ultralytics .track / BoT-SORT) if it proves too brittle on busy scenes.

Usage:
    python -m src.preprocessing.pose_extraction --input data/ut_interaction --output outputs/ut_poses
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def assign_slots(detections: list[np.ndarray], last_centroids: np.ndarray) -> list[int]:
    """Greedily map detections (≤ M of them) to person slots by centroid distance."""
    n_slots = len(last_centroids)
    centroids = [det.mean(axis=0) for det in detections]
    pairs = sorted(
        (float(np.linalg.norm(c - last_centroids[s])), i, s)
        for i, c in enumerate(centroids)
        for s in range(n_slots)
        if not np.isnan(last_centroids[s]).any()
    )
    result = [-1] * len(detections)
    taken: set[int] = set()
    for _, i, s in pairs:
        if result[i] == -1 and s not in taken:
            result[i] = s
            taken.add(s)
    free = (s for s in range(n_slots) if s not in taken)
    return [s if s != -1 else next(free) for s in result]


def extract_clip(
    model,
    source: Path,
    max_persons: int = 2,
    conf: float = 0.25,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run YOLO-Pose over a video or frame directory → (T, M, 17, 2) + (T, M, 17)."""
    keypoints_seq: list[np.ndarray] = []
    scores_seq: list[np.ndarray] = []
    last_centroids = np.full((max_persons, 2), np.nan)

    for result in model.predict(source=str(source), stream=True, conf=conf, device=device, verbose=False):
        kp = np.zeros((max_persons, 17, 2), dtype=np.float32)
        sc = np.zeros((max_persons, 17), dtype=np.float32)
        if result.keypoints is not None and len(result.keypoints):
            xy = result.keypoints.xy.cpu().numpy()  # (n, 17, 2)
            kp_conf = result.keypoints.conf
            kp_conf = kp_conf.cpu().numpy() if kp_conf is not None else np.ones(xy.shape[:2], np.float32)
            box_conf = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(xy))
            top = np.argsort(box_conf)[::-1][:max_persons]
            dets_xy = [xy[i] for i in top]
            dets_sc = [kp_conf[i] for i in top]
            for det_xy, det_sc, slot in zip(dets_xy, dets_sc, assign_slots(dets_xy, last_centroids)):
                kp[slot] = det_xy
                sc[slot] = det_sc
                visible = det_sc > 0.1
                last_centroids[slot] = det_xy[visible].mean(axis=0) if visible.any() else det_xy.mean(axis=0)
        keypoints_seq.append(kp)
        scores_seq.append(sc)

    if not keypoints_seq:
        return (
            np.zeros((0, max_persons, 17, 2), dtype=np.float32),
            np.zeros((0, max_persons, 17), dtype=np.float32),
        )
    return np.stack(keypoints_seq), np.stack(scores_seq)


def find_sources(input_path: Path) -> list[Path]:
    """A video file, a tree of videos, a directory of frames, or a directory of frame directories."""
    if input_path.is_file():
        return [input_path]
    videos = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if videos:
        return videos
    if any(p.suffix.lower() in IMAGE_EXTS for p in input_path.iterdir()):
        return [input_path]  # one clip stored as frames
    return sorted(
        d for d in input_path.iterdir()
        if d.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in d.iterdir())
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True, help="video file, directory of videos, or directory of frames")
    parser.add_argument("--output", type=Path, required=True, help="output directory for .npz files")
    parser.add_argument("--model", default="yolov8m-pose.pt", help="any ultralytics *-pose checkpoint")
    parser.add_argument("--max-persons", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    parser.add_argument("--device", default=None, help='e.g. "0" for first GPU, "cpu"')
    args = parser.parse_args()

    sources = find_sources(args.input)
    if not sources:
        raise SystemExit(f"nothing to process under {args.input}")

    from ultralytics import YOLO  # deferred: slow import, pulls in torch

    model = YOLO(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    for src in sources:
        keypoints, scores = extract_clip(model, src, args.max_persons, args.conf, args.device)
        out_path = args.output / f"{src.stem}.npz"
        np.savez_compressed(out_path, keypoints=keypoints, scores=scores, source=str(src), model=args.model)
        print(f"{src} -> {out_path}  ({len(keypoints)} frames)")


if __name__ == "__main__":
    main()
