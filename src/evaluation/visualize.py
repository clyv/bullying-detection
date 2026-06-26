"""Operator-facing visualization of aggression detection on a video.

Turns the windowed model scores (see localize.py) into two human-reviewable
outputs for a continuous clip:

1. An **annotated video** — the original footage with skeletons overlaid, a red
   border + banner on frames the model flags as aggressive, and a live timeline
   bar along the bottom.
2. A **timeline image** — a single plot of the per-frame aggression probability
   over the whole clip, with the flagged incident spans shaded.

The detector is temporal (a 64-frame window), so there is no true per-frame
score; instead every frame's score is the average of all windows covering it
(frame_scores_from_windows), which also smooths the curve. This is a
presentation layer over the same model — it makes results legible, not more
accurate. Every flag is for human review, never an autonomous judgement.

Usage:
    python -m src.evaluation.visualize --video clip.mp4 \\
        --checkpoint outputs/checkpoints/phase4_unified/stgcn_best.pt --output outputs/review
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from src.evaluation.localize import find_incidents

# COCO-17 skeleton edges for drawing (index pairs).
COCO_EDGES = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
]
PERSON_COLORS = [(0, 200, 255), (255, 200, 0)]  # BGR, one per person slot


def frame_scores_from_windows(window_probs, starts, window, num_frames):
    """Average the windowed aggression probabilities down to one score per frame.

    Each frame is covered by every window that spans it; its score is their mean,
    which both maps window scores to frames and smooths the curve.
    """
    total = np.zeros(num_frames, dtype=np.float64)
    count = np.zeros(num_frames, dtype=np.float64)
    for prob, start in zip(window_probs, starts):
        end = min(start + window, num_frames)
        total[start:end] += prob
        count[start:end] += 1
    count[count == 0] = 1.0
    return total / count


def draw_skeletons(frame, kp, sc, vis_thresh=0.1):
    """Draw each person's COCO skeleton on a BGR frame (in place)."""
    import cv2

    for person in range(kp.shape[0]):
        color = PERSON_COLORS[person % len(PERSON_COLORS)]
        joints, conf = kp[person], sc[person]
        for a, b in COCO_EDGES:
            if conf[a] > vis_thresh and conf[b] > vis_thresh:
                pa = (int(joints[a, 0]), int(joints[a, 1]))
                pb = (int(joints[b, 0]), int(joints[b, 1]))
                cv2.line(frame, pa, pb, color, 2)
        for j in range(joints.shape[0]):
            if conf[j] > vis_thresh:
                cv2.circle(frame, (int(joints[j, 0]), int(joints[j, 1])), 3, color, -1)


def draw_status(frame, agg, threshold, idx, num_frames, frame_scores):
    """Overlay the flag banner, score, and a timeline bar (in place)."""
    import cv2

    h, w = frame.shape[:2]
    flagged = agg >= threshold
    if flagged:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
    banner = (0, 0, 255) if flagged else (60, 60, 60)
    label = (
        f"AGGRESSION p={agg:.2f}  [FLAGGED - for human review]"
        if flagged
        else f"monitoring  p={agg:.2f}"
    )
    cv2.rectangle(frame, (0, 0), (w, 34), banner, -1)
    cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Timeline strip along the bottom: per-frame score as bar height, threshold
    # line, and a marker for the current position.
    bar_h, y0 = 40, h - 40
    cv2.rectangle(frame, (0, y0), (w, h), (30, 30, 30), -1)
    if num_frames > 1:
        for x in range(w):
            fi = int(x / w * num_frames)
            val = frame_scores[min(fi, num_frames - 1)]
            col = (0, 0, 255) if val >= threshold else (0, 180, 0)
            cv2.line(frame, (x, h), (x, h - int(val * bar_h)), col, 1)
        ty = h - int(threshold * bar_h)
        cv2.line(frame, (0, ty), (w, ty), (200, 200, 200), 1)
        mx = int(idx / num_frames * w)
        cv2.line(frame, (mx, y0), (mx, h), (255, 255, 255), 2)


def render_video(video_path, keypoints, scores, frame_scores, out_path, threshold=0.5, fps=None):
    """Write an annotated copy of ``video_path`` with skeletons + flags overlaid."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
    n = min(len(frame_scores), len(keypoints))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i < n:
            draw_skeletons(frame, keypoints[i], scores[i])
            draw_status(frame, float(frame_scores[i]), threshold, i, n, frame_scores)
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()
    return out_path


def save_timeline(frame_scores, threshold, out_path, fps=30.0):
    """Save a PNG of the per-frame aggression probability with incidents shaded."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(frame_scores)) / fps
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, frame_scores, color="#c0392b", lw=1.5, label="aggression probability")
    ax.axhline(threshold, color="gray", ls="--", lw=1, label=f"threshold={threshold}")
    ax.fill_between(
        t, 0, 1, where=np.asarray(frame_scores) >= threshold, color="#c0392b", alpha=0.2
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("P(aggressive)")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_title("Aggression timeline (flag for human review)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run(
    video,
    checkpoint,
    config_path="configs/unified.yaml",
    output_dir="outputs/review",
    poses=None,
    fps=None,
):
    import torch
    import yaml

    from src.evaluation.localize import score_stream
    from src.models.stgcn import STGCNBaseline

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    loc = cfg.get("localization", {})
    window, stride = loc.get("window", 64), loc.get("stride", 16)
    threshold, max_gap = loc.get("threshold", 0.5), loc.get("max_gap", 0)
    normalize = cfg["data"].get("normalize", False)

    # Poses: reuse a precomputed .npz, else run YOLO-Pose on the video.
    if poses and os.path.exists(poses):
        with np.load(poses) as d:
            keypoints, scores = d["keypoints"], d["scores"]
    else:
        from ultralytics import YOLO

        from src.preprocessing.pose_extraction import extract_clip

        keypoints, scores = extract_clip(YOLO("yolov8m-pose.pt"), video, cfg["data"]["max_persons"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STGCNBaseline(
        in_channels=cfg["model"]["in_channels"],
        num_classes=2,
        num_persons=cfg["data"]["max_persons"],
        graph_strategy="spatial",
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)

    probs, starts = score_stream(
        model, keypoints, scores, device, window, stride, normalize=normalize
    )
    frame_scores = frame_scores_from_windows(probs, starts, window, len(keypoints))
    incidents = find_incidents(frame_scores, range(len(frame_scores)), 1, threshold, max_gap)

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(str(video)))[0]
    out_fps = fps or 30.0
    timeline = save_timeline(
        frame_scores, threshold, os.path.join(output_dir, f"{stem}_timeline.png"), out_fps
    )
    annotated = render_video(
        video,
        keypoints,
        scores,
        frame_scores,
        os.path.join(output_dir, f"{stem}_annotated.mp4"),
        threshold,
        fps,
    )

    print(f"{len(incidents)} incident(s) flagged for review:")
    for start, end, score in incidents:
        print(
            f"  frames {start:>5}-{end:<5} ({start / out_fps:6.2f}s-{end / out_fps:6.2f}s)  peak={score:.2f}"
        )
    print(f"annotated video -> {annotated}")
    print(f"timeline image  -> {timeline}")
    return incidents


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video", required=True, help="source video to annotate")
    parser.add_argument("--checkpoint", required=True, help="trained binary (unified) checkpoint")
    parser.add_argument("--config", default="configs/unified.yaml")
    parser.add_argument("--output", default="outputs/review", help="output directory")
    parser.add_argument(
        "--poses", default=None, help="optional precomputed pose .npz (else runs YOLO)"
    )
    parser.add_argument("--fps", type=float, default=None, help="output fps (defaults to source)")
    args = parser.parse_args()
    run(args.video, args.checkpoint, args.config, args.output, args.poses, args.fps)


if __name__ == "__main__":
    main()
