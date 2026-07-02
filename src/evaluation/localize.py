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


def score_stream(
    model, keypoints, scores, device, window=64, stride=16, batch_size=32, normalize=False
):
    """Per-window aggression probability over a (T, M, 17, 2)/(T, M, 17) stream.

    ``normalize`` must match how the checkpoint was trained (the unified model
    uses skeleton normalization), otherwise the inputs are at the wrong scale.
    """
    import torch

    from src.datasets.unified_loader import features_to_tensor

    starts = window_starts(len(keypoints), window, stride)
    tensors = [
        features_to_tensor(keypoints[s : s + window], scores[s : s + window], window, normalize)
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


def candidate_pairs(kp_w, sc_w, max_pairs=16, distance_scale=2.5, min_visibility=0.3):
    """Interacting-person pairs within one window of a crowd scene.

    The classifier was trained on two-person interactions, so a crowded frame is
    scanned as a set of pairs: every sufficiently visible person is paired with
    their nearest neighbour, kept only if the two are within ``distance_scale``
    body-heights of each other (people across the room aren't interacting).
    Returns up to ``max_pairs`` (i, j) slot pairs, closest first.
    """
    n_persons = sc_w.shape[1]
    present, cents, sizes = [], {}, {}
    for m in range(n_persons):
        vis = sc_w[:, m] > 0  # (T, 17)
        if vis.any(axis=1).mean() < min_visibility:
            continue
        pts = kp_w[:, m][vis]
        if len(pts) == 0:
            continue
        present.append(m)
        cents[m] = pts.mean(axis=0)
        sizes[m] = max(float(pts[:, 1].max() - pts[:, 1].min()), 1.0)
    if len(present) < 2:
        return []

    scale = float(np.median([sizes[m] for m in present]))
    pairs: dict[tuple[int, int], float] = {}
    for m in present:
        dists = [(float(np.linalg.norm(cents[m] - cents[n])), n) for n in present if n != m]
        d, nearest = min(dists)
        if d <= distance_scale * scale:
            key = (min(m, nearest), max(m, nearest))
            pairs[key] = min(pairs.get(key, d), d)
    return sorted(pairs, key=pairs.get)[:max_pairs]


def score_stream_pairs(
    model, keypoints, scores, device, window=64, stride=16, normalize=False, max_pairs=16
):
    """Crowd-aware scoring: per window, the max aggression over interacting pairs.

    Returns (probs, starts, best_pairs) where best_pairs[k] is the (i, j) slot
    pair responsible for window k's score (None if nobody was interacting).
    """
    import torch

    from src.datasets.unified_loader import features_to_tensor

    starts = window_starts(len(keypoints), window, stride)
    probs: list[float] = []
    best_pairs: list[tuple[int, int] | None] = []
    model.eval()
    with torch.no_grad():
        for s in starts:
            kp_w, sc_w = keypoints[s : s + window], scores[s : s + window]
            pairs = candidate_pairs(kp_w, sc_w, max_pairs=max_pairs)
            if not pairs:
                probs.append(0.0)
                best_pairs.append(None)
                continue
            tensors = [
                features_to_tensor(kp_w[:, [i, j]], sc_w[:, [i, j]], window, normalize)
                for i, j in pairs
            ]
            batch = torch.stack(tensors).to(device)
            p = torch.softmax(model(batch), dim=1)[:, AGGRESSIVE].cpu().numpy()
            k = int(p.argmax())
            probs.append(float(p[k]))
            best_pairs.append(pairs[k])
    return np.array(probs), starts, best_pairs


def localize_stream(
    model,
    keypoints,
    scores,
    device,
    *,
    window=64,
    stride=16,
    threshold=0.5,
    max_gap=0,
    normalize=False,
    crowd_aware=False,
    max_pairs=16,
):
    """Return incident intervals (start_frame, end_frame, peak_score) for a stream."""
    if crowd_aware:
        probs, starts, _ = score_stream_pairs(
            model,
            keypoints,
            scores,
            device,
            window,
            stride,
            normalize=normalize,
            max_pairs=max_pairs,
        )
    else:
        probs, starts = score_stream(
            model, keypoints, scores, device, window, stride, normalize=normalize
        )
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
    crowd_aware = loc.get("crowd_aware", True)
    max_pairs = loc.get("max_pairs", 24)
    normalize = cfg["data"].get("normalize", False)  # must match training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STGCNBaseline(
        in_channels=cfg["model"]["in_channels"],
        num_classes=2,
        num_persons=cfg["data"]["max_persons"],
        graph_strategy="spatial",
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
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
            normalize=normalize,
            crowd_aware=crowd_aware,
            max_pairs=max_pairs,
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
