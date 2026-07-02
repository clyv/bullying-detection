"""Operator-facing visualization of aggression detection on a video.

Turns the model's windowed scores (see localize.py) into two human-reviewable
outputs for a continuous clip:

1. An **annotated video** — the original footage with skeletons overlaid, a red
   border + banner on frames the model flags as aggressive, and a live timeline
   bar along the bottom.
2. A **timeline image** — a plot of the per-frame aggression probability over
   the whole clip, with the flagged incident spans shaded.

Crowd scenes (more than 2 tracked people) are scored pair-wise: every visible
person is paired with their nearest neighbour and the two-person model scans
each pair; a frame's score is the *most aggressive interacting pair* anywhere
in frame, and that pair is the one highlighted. A complementary **crowd
pressure** signal (how tightly people cluster) catches sudden crowd
convergence — e.g. a ring forming around a victim — that pose-level scoring
misses once the victim is occluded.

Honesty notes: this layer makes results legible, not more accurate; the model
is still trained on adult two-person data. ``--mark`` spans are operator
annotations, rendered in orange as MANUAL MARK — they are never presented as
model detections. Every flag is for human review, never an autonomous
judgement.

Usage:
    python -m src.evaluation.visualize --video clip.mp4 \\
        --checkpoint outputs/checkpoints/phase4_unified/stgcn_best.pt --output outputs/review
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

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
PERSON_COLORS = [(0, 200, 255), (255, 200, 0)]  # BGR, one per highlighted person
CROWD_COLOR = (160, 160, 160)  # faint gray for non-highlighted people


# --------------------------------------------------------------------------
# Pure signal helpers (numpy only, unit-tested)
# --------------------------------------------------------------------------


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


def frame_best_pairs(window_probs, starts, window, best_pairs, num_frames):
    """Per frame, the (i, j) pair from the highest-scoring window covering it."""
    best_prob = np.full(num_frames, -1.0)
    result: list[tuple[int, int] | None] = [None] * num_frames
    for prob, start, pair in zip(window_probs, starts, best_pairs):
        end = min(start + window, num_frames)
        for f in range(start, end):
            if prob > best_prob[f]:
                best_prob[f] = prob
                result[f] = pair
    return result


def normalize_signal(values):
    """Min-max scale a 1-D signal to [0, 1]; constant signals become zeros."""
    values = np.asarray(values, dtype=np.float64)
    span = values.max() - values.min() if len(values) else 0.0
    if span <= 0:
        return np.zeros_like(values)
    return (values - values.min()) / span


def crowd_pressure_scores(keypoints, scores):
    """Per-frame crowd compactness in [0, 1): high when people cluster tightly.

    For each frame, every visible person's closeness to their nearest neighbour
    is scale / (scale + distance) (scale = median skeleton height, so the metric
    is camera-resolution invariant); the frame's pressure is the mean closeness.
    A ring of people converging on one spot drives this toward 1.
    """
    T, n_persons = scores.shape[:2]
    pressure = np.zeros(T, dtype=np.float64)
    for t in range(T):
        cents, heights = [], []
        for m in range(n_persons):
            vis = scores[t, m] > 0
            if not vis.any():
                continue
            pts = keypoints[t, m][vis]
            cents.append(pts.mean(axis=0))
            heights.append(max(float(pts[:, 1].max() - pts[:, 1].min()), 1.0))
        if len(cents) < 2:
            continue
        cents_arr = np.stack(cents)
        scale = float(np.median(heights))
        closeness = []
        for i in range(len(cents_arr)):
            d = np.linalg.norm(cents_arr - cents_arr[i], axis=1)
            d[i] = np.inf
            closeness.append(scale / (scale + float(d.min())))
        pressure[t] = float(np.mean(closeness))
    return pressure


def moving_average(values, width):
    """Box-filter smoothing; width clamped to >= 1."""
    width = max(1, int(width))
    return np.convolve(np.asarray(values, dtype=np.float64), np.ones(width) / width, mode="same")


def motion_energy_scores(keypoints, scores, min_joints=3):
    """Per-person height-normalized joint speed (heuristic activity signal).

    Returns (frame_energy (T,), per_person (T, M)) where per_person[t, m] is the
    mean joint displacement of person m between frames t-1 and t, divided by
    their skeleton height (so small far-away people register like near ones).
    frame_energy is the max over persons — "the most agitated person in frame".

    This is NOT the trained classifier: frames flagged from this signal are
    labelled HIGH ACTIVITY for review, never AGGRESSION.
    """
    T, n_persons = scores.shape[:2]
    per_person = np.zeros((T, n_persons))
    for t in range(1, T):
        for m in range(n_persons):
            vis = (scores[t, m] > 0) & (scores[t - 1, m] > 0)
            if vis.sum() < min_joints:
                continue
            ys = keypoints[t, m][scores[t, m] > 0][:, 1]
            height = max(float(ys.max() - ys.min()), 8.0)
            disp = np.linalg.norm(keypoints[t, m][vis] - keypoints[t - 1, m][vis], axis=1)
            per_person[t, m] = float(disp.mean()) / height
    return per_person.max(axis=1), per_person


def activity_pairs(keypoints, scores, per_person_motion, smooth_width):
    """Per frame: (most agitated person, their nearest visible neighbour) or None."""
    T, n_persons = scores.shape[:2]
    smoothed = np.stack(
        [moving_average(per_person_motion[:, m], smooth_width) for m in range(n_persons)], axis=1
    )
    result: list[tuple[int, int] | None] = [None] * T
    for t in range(T):
        order = np.argsort(smoothed[t])[::-1]
        agitated = int(order[0])
        if smoothed[t, agitated] <= 0:
            continue
        vis_a = scores[t, agitated] > 0
        if not vis_a.any():
            continue
        ca = keypoints[t, agitated][vis_a].mean(axis=0)
        best, best_d = None, None
        for m in range(n_persons):
            if m == agitated:
                continue
            vis = scores[t, m] > 0
            if not vis.any():
                continue
            d = float(np.linalg.norm(keypoints[t, m][vis].mean(axis=0) - ca))
            if best_d is None or d < best_d:
                best, best_d = m, d
        if best is not None:
            result[t] = (min(agitated, best), max(agitated, best))
    return result


def pressure_spikes(pressure, baseline_frames):
    """Rise of crowd pressure above its recent baseline (sudden gatherings).

    School corridors are always somewhat crowded, so absolute density would
    flag constantly; a spike over the trailing-median baseline isolates the
    *convergence* moment instead.
    """
    pressure = np.asarray(pressure, dtype=np.float64)
    spikes = np.zeros_like(pressure)
    for i in range(len(pressure)):
        lo = max(0, i - baseline_frames)
        baseline = float(np.median(pressure[lo : i + 1]))
        spikes[i] = max(0.0, pressure[i] - baseline)
    return spikes


def keep_top_incident_scores(frame_scores, incidents, threshold, top_n):
    """Keep only the ``top_n`` strongest incidents; suppress the rest below threshold.

    Returns (adjusted_scores, kept_incidents). Useful when an operator wants the
    review queue limited to the most severe events in a clip.
    """
    frame_scores = np.asarray(frame_scores, dtype=np.float64).copy()
    ranked = sorted(incidents, key=lambda inc: inc[2], reverse=True)
    kept = sorted(ranked[:top_n], key=lambda inc: inc[0])
    for start, end, _score in ranked[top_n:]:
        frame_scores[start:end] = np.minimum(frame_scores[start:end], threshold * 0.9)
    return frame_scores, kept


def earliest_incident_before(incidents, before_frame, pad, num_frames):
    """The earliest incident starting before ``before_frame``, padded by ``pad``.

    Used to surface the likely *triggering* event preceding a crowd-convergence
    peak (the beating usually happens just before the crowd forms). Returns a
    one-element list [(start, end, score)] or [] if none qualifies.
    """
    candidates = [inc for inc in incidents if inc[0] < before_frame]
    if not candidates:
        return []
    start, end, score = min(candidates, key=lambda inc: inc[0])
    return [(max(0, start - pad), min(num_frames, end + pad), score)]


def add_forced_incidents(frame_scores, incidents, forced, threshold):
    """Merge operator-supplied manual marks into the incident list.

    ``forced`` is a list of (start, end, score) spans an operator wants flagged
    regardless of the model (e.g. known ground truth for a review session).
    Their scores are lifted just above ``threshold`` so they render as flagged,
    but they are labelled MANUAL MARK in the video — never as model detections.
    """
    boosted = np.asarray(frame_scores, dtype=np.float64).copy()
    merged = list(incidents)
    for start, end, score in forced:
        peak = round(max(float(score), threshold + 0.01), 6)
        boosted[start:end] = np.maximum(boosted[start:end], peak)
        merged.append((start, end, peak))
    return boosted, sorted(merged, key=lambda inc: inc[0])


def output_stem(video_path, run_name=None, unique=True):
    """Output file stem for a run: clip name + run name or a timestamp.

    Unique-by-default so repeated runs (e.g. threshold tuning) never overwrite
    each other's outputs.
    """
    stem = os.path.splitext(os.path.basename(str(video_path)))[0]
    if run_name:
        return f"{stem}_{run_name}"
    if unique:
        return f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return stem


# --------------------------------------------------------------------------
# Rendering (cv2 / matplotlib, imported lazily)
# --------------------------------------------------------------------------


def draw_skeletons(frame, kp, sc, vis_thresh=0.1, color=None, thickness=2):
    """Draw COCO skeletons on a BGR frame (in place).

    ``color`` fixes one colour for all people (used for the faint crowd);
    otherwise each person slot gets its own colour.
    """
    import cv2

    for person in range(kp.shape[0]):
        col = color if color is not None else PERSON_COLORS[person % len(PERSON_COLORS)]
        joints, conf = kp[person], sc[person]
        for a, b in COCO_EDGES:
            if conf[a] > vis_thresh and conf[b] > vis_thresh:
                pa = (int(joints[a, 0]), int(joints[a, 1]))
                pb = (int(joints[b, 0]), int(joints[b, 1]))
                cv2.line(frame, pa, pb, col, thickness)
        for j in range(joints.shape[0]):
            if conf[j] > vis_thresh:
                cv2.circle(frame, (int(joints[j, 0]), int(joints[j, 1])), thickness + 1, col, -1)


def draw_status(frame, agg, threshold, idx, num_frames, frame_scores, manual=False, activity=False):
    """Overlay the flag banner, score, and a timeline bar (in place).

    Three flag types, honestly distinguished: model detections (AGGRESSION),
    heuristic motion flags (HIGH ACTIVITY), operator marks (MANUAL MARK).
    """
    import cv2

    h, w = frame.shape[:2]
    flagged = agg >= threshold
    if flagged:
        border = (0, 165, 255) if manual else (0, 80, 255) if activity else (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border, 8)
    if manual and flagged:
        banner, label = (0, 165, 255), f"MANUAL MARK p={agg:.2f}  [operator annotation]"
    elif activity and flagged:
        banner, label = (0, 80, 255), f"HIGH ACTIVITY p={agg:.2f}  [FLAGGED - for human review]"
    elif flagged:
        banner, label = (0, 0, 255), f"AGGRESSION p={agg:.2f}  [FLAGGED - for human review]"
    else:
        banner, label = (60, 60, 60), f"monitoring  p={agg:.2f}"
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


def render_video(
    video_path,
    keypoints,
    scores,
    frame_scores,
    out_path,
    threshold=0.5,
    fps=None,
    highlight_pairs=None,
    manual_mask=None,
    activity_mask=None,
):
    """Write an annotated copy of ``video_path``.

    With ``highlight_pairs`` (crowd mode) everyone is drawn faintly and only the
    highest-scoring interacting pair is drawn in colour.
    """
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
            if highlight_pairs is None:
                draw_skeletons(frame, keypoints[i], scores[i])
            else:
                draw_skeletons(frame, keypoints[i], scores[i], color=CROWD_COLOR, thickness=1)
                pair = highlight_pairs[i]
                if pair is not None:
                    sel = list(pair)
                    draw_skeletons(frame, keypoints[i][sel], scores[i][sel], thickness=3)
            manual = bool(manual_mask[i]) if manual_mask is not None else False
            activity = bool(activity_mask[i]) if activity_mask is not None else False
            draw_status(
                frame, float(frame_scores[i]), threshold, i, n, frame_scores, manual, activity
            )
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()
    return out_path


def save_timeline(frame_scores, threshold, out_path, fps=30.0, activity=None):
    """Save a PNG of the per-frame aggression probability with incidents shaded."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(frame_scores)) / fps
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, frame_scores, color="#c0392b", lw=1.5, label="combined score")
    if activity is not None:
        ax.plot(
            t, activity, color="#e67e22", lw=1.0, ls=":", alpha=0.9, label="activity (heuristic)"
        )
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


# --------------------------------------------------------------------------
# End-to-end driver
# --------------------------------------------------------------------------


def _parse_marks(marks, fps, num_frames):
    """'start-end' second spans -> (start_frame, end_frame, 0.0) tuples."""
    forced = []
    for span in marks or []:
        lo, _, hi = span.partition("-")
        s = max(0, int(float(lo) * fps))
        e = min(num_frames, int(float(hi) * fps))
        if e > s:
            forced.append((s, e, 0.0))
    return forced


def run(
    video,
    checkpoint,
    config_path="configs/unified.yaml",
    output_dir="outputs/review",
    poses=None,
    fps=None,
    max_persons=None,
    top_incidents=None,
    marks=None,
    run_name=None,
    unique=True,
):
    import cv2
    import torch
    import yaml

    from src.evaluation.localize import score_stream, score_stream_pairs
    from src.models.stgcn import STGCNBaseline

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    loc = cfg.get("localization", {})
    window, stride = loc.get("window", 64), loc.get("stride", 16)
    threshold, max_gap = loc.get("threshold", 0.5), loc.get("max_gap", 0)
    normalize = cfg["data"].get("normalize", False)
    crowd_cfg = cfg.get("crowd", {})
    extract_persons = max_persons or crowd_cfg.get("max_persons", 16)
    pressure_weight = crowd_cfg.get("pressure_weight", 1.5)
    baseline_seconds = crowd_cfg.get("baseline_seconds", 10)
    max_pairs = crowd_cfg.get("max_pairs", 16)

    src_fps = fps
    if src_fps is None:
        cap = cv2.VideoCapture(str(video))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

    # Poses: reuse a precomputed .npz, else run YOLO-Pose on the video.
    if poses and os.path.exists(poses):
        with np.load(poses) as d:
            keypoints, scores = d["keypoints"], d["scores"]
    else:
        from ultralytics import YOLO

        from src.preprocessing.pose_extraction import extract_clip

        keypoints, scores = extract_clip(YOLO("yolov8m-pose.pt"), video, extract_persons)

    num_frames = len(keypoints)
    n_persons = keypoints.shape[1] if num_frames else 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STGCNBaseline(
        in_channels=cfg["model"]["in_channels"],
        num_classes=2,
        num_persons=2,  # the classifier always scores two-person interactions
        graph_strategy="spatial",
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)

    motion_floor = crowd_cfg.get("motion_floor", 0.5)
    motion_saturate = crowd_cfg.get("motion_saturate", 2.0)
    motion_weight = crowd_cfg.get("motion_weight", 0.9)
    smooth_seconds = crowd_cfg.get("smooth_seconds", 0.5)

    pressure = None
    activity_curve = None
    activity_mask = None
    highlight = None
    if n_persons > 2:
        # Crowd mode: scan interacting pairs; a frame's model score is its worst pair.
        probs, starts, best_pairs = score_stream_pairs(
            model,
            keypoints,
            scores,
            device,
            window,
            stride,
            normalize=normalize,
            max_pairs=max_pairs,
        )
        model_scores = frame_scores_from_windows(probs, starts, window, num_frames)
        model_pairs = frame_best_pairs(probs, starts, window, best_pairs, num_frames)

        # Heuristic activity signal: the most agitated person in frame, in
        # height-normalized units, ramped between motion_floor and motion_saturate
        # (calibrated on 30fps CCTV). Flags from this path say HIGH ACTIVITY.
        smooth_w = int(smooth_seconds * src_fps)
        frame_motion, per_person_motion = motion_energy_scores(keypoints, scores)
        motion_smooth = moving_average(frame_motion, smooth_w)
        span = max(motion_saturate - motion_floor, 1e-6)
        activity_curve = np.clip((motion_smooth - motion_floor) / span, 0.0, 1.0) * motion_weight
        # Crowd-convergence spikes also count as activity (ring forming around
        # a victim the pose model can't see into).
        pressure = crowd_pressure_scores(keypoints, scores)
        spikes = pressure_spikes(pressure, int(baseline_seconds * src_fps))
        activity_curve = np.maximum(activity_curve, np.clip(spikes * pressure_weight, 0.0, 0.99))

        activity_mask = activity_curve > model_scores
        frame_scores = np.maximum(model_scores, activity_curve)
        motion_hl = activity_pairs(keypoints, scores, per_person_motion, smooth_w)
        highlight = [
            motion_hl[f] if activity_mask[f] and motion_hl[f] is not None else model_pairs[f]
            for f in range(num_frames)
        ]
    else:
        probs, starts = score_stream(
            model, keypoints, scores, device, window, stride, normalize=normalize
        )
        frame_scores = frame_scores_from_windows(probs, starts, window, num_frames)

    incidents = find_incidents(frame_scores, range(num_frames), 1, threshold, max_gap)

    if top_incidents:
        frame_scores, incidents = keep_top_incident_scores(
            frame_scores, incidents, threshold, top_incidents
        )

    # Point the reviewer at the likely trigger preceding the biggest gathering.
    if pressure is not None and len(incidents):
        spikes_arr = pressure_spikes(pressure, int(baseline_seconds * src_fps))
        if spikes_arr.max() > 0:
            peak = int(spikes_arr.argmax())
            trigger = earliest_incident_before(incidents, peak, int(src_fps), num_frames)
            if trigger:
                s, e, sc = trigger[0]
                print(
                    f"[note] crowd convergence peaks at {peak / src_fps:.1f}s; "
                    f"likely triggering incident {s / src_fps:.1f}s-{e / src_fps:.1f}s (p={sc:.2f})"
                )

    manual_mask = None
    forced = _parse_marks(marks, src_fps, num_frames)
    if forced:
        frame_scores, incidents = add_forced_incidents(frame_scores, incidents, forced, threshold)
        manual_mask = np.zeros(num_frames, dtype=bool)
        for s, e, _ in forced:
            manual_mask[s:e] = True

    os.makedirs(output_dir, exist_ok=True)
    stem = output_stem(video, run_name, unique)
    timeline = save_timeline(
        frame_scores,
        threshold,
        os.path.join(output_dir, f"{stem}_timeline.png"),
        src_fps,
        activity=activity_curve,
    )
    annotated = render_video(
        video,
        keypoints,
        scores,
        frame_scores,
        os.path.join(output_dir, f"{stem}_annotated.mp4"),
        threshold,
        fps,
        highlight_pairs=highlight,
        manual_mask=manual_mask,
        activity_mask=activity_mask,
    )

    forced_spans = {(s, e) for s, e, _ in forced}
    print(f"{len(incidents)} incident(s) flagged for review:")
    for start, end, score in incidents:
        tag = "  [manual mark]" if (start, end) in forced_spans else ""
        print(
            f"  frames {start:>5}-{end:<5} "
            f"({start / src_fps:6.2f}s-{end / src_fps:6.2f}s)  peak={score:.2f}{tag}"
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
    parser.add_argument(
        "--max-persons", type=int, default=None, help="people to track (>2 enables crowd mode)"
    )
    parser.add_argument(
        "--top-incidents", type=int, default=None, help="keep only the N strongest incidents"
    )
    parser.add_argument(
        "--mark",
        action="append",
        default=None,
        help='manual review span in seconds, e.g. "65-80" (repeatable; rendered as MANUAL MARK)',
    )
    parser.add_argument("--run-name", default=None, help="suffix for output filenames")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="reuse the plain clip name instead of a unique timestamped name",
    )
    args = parser.parse_args()
    run(
        args.video,
        args.checkpoint,
        args.config,
        args.output,
        args.poses,
        args.fps,
        args.max_persons,
        args.top_incidents,
        args.mark,
        args.run_name,
        unique=not args.overwrite,
    )


if __name__ == "__main__":
    main()
