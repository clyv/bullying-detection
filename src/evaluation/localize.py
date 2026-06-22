"""Phase 5 (stretch) — temporal localization of aggression in a continuous stream.

Single-clip classification (Phases 1-4) answers "is this clip aggressive?".
Deployment instead needs "*when* in this stream does an incident occur?". This
module slides a fixed window over a long pose sequence (the untrimmed .npz that
pose_extraction.py produces from a continuous video), scores each window with
the trained binary aggressive-vs-neutral model, and merges consecutive
above-threshold windows into incident intervals — flagging spans for human
review, never making autonomous accusations.

The windowing, merging, and temporal-IoU logic is plain Python/numpy so it is
unit-testable without a GPU; only score_stream needs torch + a checkpoint.

Usage:
    python -m src.evaluation.localize --stream outputs/cctv_poses/clip.npz \\
        --checkpoint outputs/checkpoints/stgcn_baseline_epoch_40.pt --config configs/unified.yaml
"""

from __future__ import annotations

import argparse

import numpy as np

from src.datasets.taxonomy import AGGRESSIVE


def window_starts(num_frames: int, window: int, stride: int) -> list[int]:
    """Start frames for sliding windows; always includes a final flush-right window."""
    if num_frames <= window:
        return [0]
    starts = list(range(0, num_frames - window + 1, stride))
    if starts[-1] != num_frames - window:
        starts.append(num_frames - window)
    return starts


def find_incidents(
    scores, starts, window: int, threshold: float = 0.5, max_gap: int = 0
) -> list[tuple[int, int, float]]:
    """Merge above-threshold windows into (start_frame, end_frame, peak_score) intervals.

    Adjacent or overlapping aggressive windows (within ``max_gap`` frames) are
    merged into one incident; the peak window score is kept.
    """
    incidents: list[list[float]] = []
    for score, start in zip(scores, starts):
        if score < threshold:
            continue
        s, e = start, start + window
        if incidents and s <= incidents[-1][1] + max_gap:
            incidents[-1][1] = max(incidents[-1][1], e)
            incidents[-1][2] = max(incidents[-1][2], float(score))
        else:
            incidents.append([s, e, float(score)])
    return [(int(s), int(e), sc) for s, e, sc in incidents]


def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Intersection-over-union of two [start, end) frame intervals."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def detection_matches(
    predicted: list[tuple], ground_truth: list[tuple], iou_threshold: float = 0.5
) -> int:
    """Count ground-truth incidents matched by some prediction at >= iou_threshold."""
    matched = 0
    for gt in ground_truth:
        if any(temporal_iou(pred[:2], gt[:2]) >= iou_threshold for pred in predicted):
            matched += 1
    return matched


def score_stream(model, keypoints, scores, device, window=64, stride=16, batch_size=32):
    """Per-window aggression probability over a (T, M, 17, 2)/(T, M, 17) stream."""
    import torch

    from src.datasets.unified_loader import features_to_tensor

    starts = window_starts(len(keypoints), window, stride)
    tensors = [
        features_to_tensor(keypoints[s : s + window], scores[s : s + window], window)
        for s in starts
    ]
    probs: list[float] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i : i + batch_size]).to(device)
            softmax = torch.softmax(model(batch), dim=1)
            probs.extend(softmax[:, AGGRESSIVE].cpu().numpy().tolist())
    return np.array(probs), starts


def localize_stream(
    model, keypoints, scores, device, *, window=64, stride=16, threshold=0.5, max_gap=0
):
    """Return incident intervals (start_frame, end_frame, peak_score) for a stream."""
    probs, starts = score_stream(model, keypoints, scores, device, window, stride)
    return find_incidents(probs, starts, window, threshold, max_gap)


def run(stream_path, checkpoint, config_path="configs/unified.yaml", fps=30.0):
    import torch
    import yaml

    from src.models.stgcn import STGCNBaseline

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    loc = cfg.get("localization", {})
    window = loc.get("window", 64)
    stride = loc.get("stride", 16)
    threshold = loc.get("threshold", 0.5)
    max_gap = loc.get("max_gap", 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STGCNBaseline(
        in_channels=cfg["model"]["in_channels"],
        num_classes=2,
        num_persons=cfg["data"]["max_persons"],
        graph_strategy="spatial",
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)

    with np.load(stream_path) as data:
        incidents = localize_stream(
            model,
            data["keypoints"],
            data["scores"],
            device,
            window=window,
            stride=stride,
            threshold=threshold,
            max_gap=max_gap,
        )

    print(f"{stream_path}: {len(incidents)} incident(s) flagged for review")
    for start, end, score in incidents:
        print(
            f"  frames {start:>5}-{end:<5} "
            f"({start / fps:6.2f}s - {end / fps:6.2f}s)  aggression={score:.2f}"
        )
    return incidents


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stream", required=True, help="pose .npz for one continuous stream")
    parser.add_argument("--checkpoint", required=True, help="trained binary model checkpoint")
    parser.add_argument("--config", default="configs/unified.yaml")
    parser.add_argument("--fps", type=float, default=30.0, help="for frame -> timestamp reporting")
    args = parser.parse_args()
    run(args.stream, args.checkpoint, args.config, args.fps)


if __name__ == "__main__":
    main()
