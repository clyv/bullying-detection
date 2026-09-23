"""Multi-person pose extraction with stable track ids.

Separate from src/preprocessing/pose_extraction.py, which stays exactly as it is:
that one feeds the assault detector two people per clip, and changing it would
change every cached pose in the project. This one is for early warning, where
encirclement, outnumbering and convergence are invisible with two skeletons and
meaningless without identities that persist across frames.

Writes the same .npz contract as the rest of the project, plus track ids:

    keypoints  (T, M, 17, 2) float32
    scores     (T, M, 17)    float32
    track_ids  (T, M)        int32, -1 where the slot is empty
    fps        ()            float32, the *processed* frame rate

Usage:
    python -m src.early_warning.tracked_pose --video hall.mp4 --output hall_tracked.npz
"""

from __future__ import annotations

import argparse

import numpy as np

DEFAULT_TRACKER = "bytetrack.yaml"  # botsort.yaml adds camera-motion compensation


class SlotAssigner:
    """Maps tracker ids to fixed array slots, reusing slots only after a grace period.

    A tracker id is an arbitrary integer that grows forever; the feature code
    needs a small fixed M where slot m is the same person over time. Slots are
    released only after `grace_frames` of absence, so a brief occlusion keeps the
    person in place instead of shuffling everyone along.
    """

    def __init__(self, max_people: int = 12, grace_frames: int = 15):
        self.max_people = int(max_people)
        self.grace_frames = int(grace_frames)
        self.slot_of: dict[int, int] = {}
        self.last_seen: dict[int, int] = {}

    def assign(self, track_ids, frame_index: int) -> dict[int, int]:
        """Return {track_id: slot} for the ids visible in this frame."""
        for tid in track_ids:
            if tid in self.slot_of:
                self.last_seen[tid] = frame_index
        free = set(range(self.max_people)) - {
            slot
            for tid, slot in self.slot_of.items()
            if frame_index - self.last_seen.get(tid, frame_index) <= self.grace_frames
        }
        for tid in track_ids:
            if tid in self.slot_of:
                continue
            if not free:
                continue  # more people than slots: the extras are dropped this frame
            slot = min(free)
            free.discard(slot)
            # Evict whoever held this slot beyond the grace period.
            for other, other_slot in list(self.slot_of.items()):
                if other_slot == slot and other != tid:
                    del self.slot_of[other]
                    self.last_seen.pop(other, None)
            self.slot_of[tid] = slot
            self.last_seen[tid] = frame_index
        return {tid: self.slot_of[tid] for tid in track_ids if tid in self.slot_of}

    def switches(self) -> int:
        """How many distinct tracker ids have been seen (identity churn proxy)."""
        return len(self.last_seen)


def extract(
    video: str,
    weights: str = "yolov8m-pose.pt",
    max_people: int = 12,
    tracker: str = DEFAULT_TRACKER,
    process_fps: float = 10.0,
    conf: float = 0.25,
) -> dict:
    """Run tracked pose estimation over a video and return the .npz arrays.

    Frames are subsampled to `process_fps`: social signals do not need 30 fps,
    and the cost is linear in frames processed.
    """
    import os

    import cv2
    from ultralytics import YOLO

    # OpenCV returns the same "not opened" for a missing path and an undecodable
    # file, which sends people hunting for a codec problem they don't have.
    if not os.path.exists(video):
        raise FileNotFoundError(f"no such video file: {video} (cwd {os.getcwd()})")

    model = YOLO(weights)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError(f"found {video} but OpenCV could not decode it — unsupported codec?")
    src_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / max(process_fps, 1e-6))))

    slots = SlotAssigner(max_people)
    kp_frames, sc_frames, id_frames = [], [], []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step:
                index += 1
                continue
            result = model.track(frame, persist=True, tracker=tracker, conf=conf, verbose=False)[0]
            kp = np.zeros((max_people, 17, 2), dtype=np.float32)
            sc = np.zeros((max_people, 17), dtype=np.float32)
            ids = np.full(max_people, -1, dtype=np.int32)
            if result.boxes is not None and result.boxes.id is not None:
                track_ids = [int(t) for t in result.boxes.id.cpu().numpy()]
                mapping = slots.assign(track_ids, len(kp_frames))
                points = result.keypoints.xy.cpu().numpy()
                confs = result.keypoints.conf
                confs = (
                    np.ones(points.shape[:2], dtype=np.float32)
                    if confs is None
                    else confs.cpu().numpy()
                )
                for row, tid in enumerate(track_ids):
                    slot = mapping.get(tid)
                    if slot is None:
                        continue
                    kp[slot], sc[slot], ids[slot] = points[row], confs[row], tid
            kp_frames.append(kp)
            sc_frames.append(sc)
            id_frames.append(ids)
            index += 1
    finally:
        capture.release()

    return dict(
        keypoints=np.stack(kp_frames) if kp_frames else np.zeros((0, max_people, 17, 2), "float32"),
        scores=np.stack(sc_frames) if sc_frames else np.zeros((0, max_people, 17), "float32"),
        track_ids=np.stack(id_frames) if id_frames else np.zeros((0, max_people), "int32"),
        fps=np.float32(src_fps / step),
        source_fps=np.float32(src_fps),
        n_tracks=np.int32(slots.switches()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default="yolov8m-pose.pt")
    parser.add_argument("--max-people", type=int, default=12)
    parser.add_argument("--tracker", default=DEFAULT_TRACKER, help="bytetrack.yaml | botsort.yaml")
    parser.add_argument("--process-fps", type=float, default=10.0)
    args = parser.parse_args()

    data = extract(args.video, args.weights, args.max_people, args.tracker, args.process_fps)
    np.savez_compressed(args.output, **data)
    print(
        f"{args.video}: {len(data['keypoints'])} frames at {float(data['fps']):.1f} fps, "
        f"{int(data['n_tracks'])} distinct track ids -> {args.output}"
    )


if __name__ == "__main__":
    main()
