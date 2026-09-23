"""Leakage-safe window datasets for escalation anticipation.

Sliding windows from one incident overlap almost completely, so a random split
over windows puts near-copies of the same seconds on both sides of the
evaluation boundary and reports a number that means nothing. Every split here
is therefore made over *videos* (or datasets), never windows — the same
discipline as the leave-one-dataset-out protocol on the detection side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.early_warning.anticipation_labels import (
    VideoEvents,
    labels_to_arrays,
    make_window_labels,
)
from src.early_warning.social_features import window_stats


@dataclass
class WindowSet:
    """Windows pooled across videos, with the arrays needed to split and train."""

    x: np.ndarray  # (n, F) window features
    feature_names: list[str]
    video_id: np.ndarray  # (n,) str
    dataset: np.ndarray  # (n,) str
    event: np.ndarray  # (n,) 1 if an onset falls inside the horizon
    bin: np.ndarray  # (n,) onset bin, or n_bins when censored
    tau_s: np.ndarray  # (n,) seconds to onset (inf if none)
    in_assault: np.ndarray  # (n,) bool
    phase: np.ndarray  # (n,) annotated phase or -1
    weight: np.ndarray  # (n,) 0 inside assaults and the onset-jitter band
    end_frame: np.ndarray  # (n,)

    def __len__(self) -> int:
        return len(self.x)

    def trainable(self) -> np.ndarray:
        """Mask of windows that may be used for hazard supervision."""
        return (~self.in_assault) & (self.weight > 0)

    def targets(self, horizons_s) -> dict[float, np.ndarray]:
        """Binary "onset within h seconds" target per horizon."""
        return {
            float(h): ((self.event == 1) & (self.tau_s <= h)).astype(np.int64) for h in horizons_s
        }


def video_windows(
    events: VideoEvents,
    signals: dict,
    fps: float,
    windows_s=(2.0, 5.0, 10.0),
    **label_kwargs,
) -> tuple[np.ndarray, list[str], dict]:
    """Window features and labels for one video.

    `signals` comes from social_features.compute(...).signals and is indexed by
    the same frame numbering as `events`.
    """
    stats, names = window_stats(signals, fps, windows_s)
    labels = make_window_labels(events, **label_kwargs)
    n_frames = len(next(iter(signals.values())))
    labels = [x for x in labels if x.end_frame < n_frames]
    arrays = labels_to_arrays(labels)
    return stats[arrays["end_frame"]], names, arrays


def build(
    items: list[tuple[VideoEvents, dict]],
    fps: float,
    windows_s=(2.0, 5.0, 10.0),
    **label_kwargs,
) -> WindowSet:
    """Pool windows from many videos into one WindowSet."""
    xs, names = [], []
    cols: dict[str, list] = {
        k: []
        for k in (
            "video_id",
            "dataset",
            "event",
            "bin",
            "tau_s",
            "in_assault",
            "phase",
            "weight",
            "end_frame",
        )
    }
    for events, signals in items:
        x, names, arrays = video_windows(events, signals, fps, windows_s, **label_kwargs)
        if len(x) == 0:
            continue
        xs.append(x)
        cols["video_id"].append(np.full(len(x), events.video_id, dtype=object))
        cols["dataset"].append(np.full(len(x), events.dataset, dtype=object))
        for key in ("event", "bin", "tau_s", "in_assault", "phase", "weight", "end_frame"):
            cols[key].append(arrays[key])
    if not xs:
        raise ValueError("no windows produced; check fps, min_context_s and clip lengths")
    return WindowSet(
        x=np.concatenate(xs),
        feature_names=names,
        **{k: np.concatenate(v) for k, v in cols.items()},
    )


def group_split(
    ws: WindowSet, val_frac: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Train / validation masks split by video, never by window."""
    videos = np.unique(ws.video_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(videos)
    n_val = max(1, int(round(len(videos) * val_frac)))
    val_videos = set(videos[:n_val].tolist())
    is_val = np.array([v in val_videos for v in ws.video_id])
    return ~is_val, is_val


def leave_one_dataset_out(ws: WindowSet):
    """Yield (held-out dataset name, train mask, test mask) for each corpus."""
    for name in sorted(set(ws.dataset.tolist())):
        test = ws.dataset == name
        yield name, ~test, test


def balance_negatives(
    ws: WindowSet, mask: np.ndarray, neg_per_pos: float = 3.0, seed: int = 42
) -> np.ndarray:
    """Subsample censored windows so positives are not drowned out.

    Returns a mask over the same index space. Only applied to training, never to
    evaluation, where the true negative rate is exactly what the false-alarm
    budget needs to measure.
    """
    usable = mask & ws.trainable()
    pos = np.flatnonzero(usable & (ws.event == 1))
    neg = np.flatnonzero(usable & (ws.event == 0))
    if pos.size == 0 or neg.size <= pos.size * neg_per_pos:
        return usable
    rng = np.random.default_rng(seed)
    keep = rng.choice(neg, size=int(pos.size * neg_per_pos), replace=False)
    out = np.zeros(len(ws), dtype=bool)
    out[pos] = True
    out[keep] = True
    return out
