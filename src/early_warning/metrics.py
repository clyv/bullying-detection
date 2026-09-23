"""Anticipation metrics, following traffic-accident-anticipation practice.

Scores are per-frame values (for example P(onset within 5 s)) over whole
videos. An alarm only counts as anticipation if it fires BEFORE the assault
onset; alarms at or after the onset are detection, with lead time 0.

Always report AP next to lead time: a model that fires on everything gets
long lead times and poor precision.

Headline metric (design doc, section 6): anticipation_at_budget, the share of
assaults flagged at least min_lead_s before onset at the threshold that keeps
false alarms on normal footage within an agreed budget.
"""

from __future__ import annotations

import numpy as np


def first_alarm_frame(scores, threshold: float, end: int | None = None) -> int | None:
    """First frame (before `end`) whose score reaches the threshold, or None."""
    s = np.asarray(scores)[:end]
    idx = np.flatnonzero(s >= threshold)
    return int(idx[0]) if idx.size else None


def lead_times(pos_scores, pos_onsets, threshold: float, fps: float) -> np.ndarray:
    """Seconds between the first pre-onset alarm and the onset, per positive video."""
    leads = []
    for s, onset in zip(pos_scores, pos_onsets):
        f = first_alarm_frame(s, threshold, end=onset)
        leads.append((onset - f) / fps if f is not None else 0.0)
    return np.asarray(leads, dtype=np.float64)


def anticipation_curve(
    pos_scores, pos_onsets, neg_scores, fps: float, thresholds=None
) -> list[dict]:
    """Video-level precision, recall and mean lead time (TTA) per threshold.

    pos_scores: per-frame scores for videos containing an assault
    pos_onsets: first assault frame of each positive video
    neg_scores: per-frame scores for videos without an assault
    """
    thresholds = np.linspace(0.05, 0.95, 19) if thresholds is None else thresholds
    rows = []
    for p in thresholds:
        leads = lead_times(pos_scores, pos_onsets, p, fps)
        tp = int((leads > 0).sum())
        fp = sum(first_alarm_frame(s, p) is not None for s in neg_scores)
        rows.append(
            {
                "threshold": float(p),
                "precision": tp / (tp + fp) if tp + fp else 1.0,
                "recall": tp / max(len(leads), 1),
                "tta": float(leads[leads > 0].mean()) if tp else 0.0,
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    """AP (step integration of the video-level PR curve), mTTA (mean over
    thresholds) and TTA at 80% recall (highest threshold still reaching 80%)."""
    ordered = sorted(rows, key=lambda r: (r["recall"], -r["threshold"]))
    ap, prev = 0.0, 0.0
    for r in ordered:
        ap += (r["recall"] - prev) * r["precision"]
        prev = r["recall"]
    hit = [r for r in rows if r["recall"] >= 0.8]
    return {
        "ap": float(ap),
        "mtta": float(np.mean([r["tta"] for r in rows])) if rows else 0.0,
        "tta_at_r80": max(hit, key=lambda r: r["threshold"])["tta"] if hit else 0.0,
    }


def alarm_episodes(
    scores, threshold: float, fps: float, min_gap_s: float = 10.0
) -> list[tuple[int, int]]:
    """Runs of frames above threshold, merging runs separated by < min_gap_s.

    One incident should cost one alarm, not one per frame.
    """
    above = np.asarray(scores) >= threshold
    if not above.any():
        return []
    edges = np.diff(np.concatenate([[0], above.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    episodes = [[int(starts[0]), int(ends[0])]]
    for s, e in zip(starts[1:], ends[1:]):
        if (s - episodes[-1][1]) / fps < min_gap_s:
            episodes[-1][1] = int(e)
        else:
            episodes.append([int(s), int(e)])
    return [(s, e) for s, e in episodes]


def false_alarms_per_hour(
    neg_scores, threshold: float, fps: float, min_gap_s: float = 10.0
) -> float:
    """Alarm episodes per hour of footage that contains no assault."""
    n_alarms = sum(len(alarm_episodes(s, threshold, fps, min_gap_s)) for s in neg_scores)
    hours = sum(len(s) for s in neg_scores) / fps / 3600.0
    return n_alarms / hours if hours > 0 else 0.0


def threshold_for_budget(neg_scores, fps: float, budget_per_hour: float, grid=None) -> float:
    """Lowest threshold whose false alarms per hour on normal footage stay within budget."""
    grid = np.linspace(0.05, 0.99, 95) if grid is None else np.sort(np.asarray(grid))
    for p in grid:
        if false_alarms_per_hour(neg_scores, float(p), fps) <= budget_per_hour:
            return float(p)
    return 1.0


def anticipation_at_budget(
    pos_scores,
    pos_onsets,
    neg_scores,
    fps: float,
    budget_per_hour: float,
    min_lead_s: float = 1.0,
    grid=None,
) -> dict:
    """Share of assaults flagged at least min_lead_s early, at the alarm-budget threshold."""
    thr = threshold_for_budget(neg_scores, fps, budget_per_hour, grid)
    leads = lead_times(pos_scores, pos_onsets, thr, fps)
    hit = leads >= min_lead_s
    return {
        "threshold": thr,
        "recall": float(hit.mean()) if leads.size else 0.0,
        "median_lead_s": float(np.median(leads[hit])) if hit.any() else 0.0,
        "false_alarms_per_hour": false_alarms_per_hour(neg_scores, thr, fps),
    }


def lead_time_distribution(leads) -> dict:
    """Percentiles of the lead-time distribution.

    Report this rather than a single mean: the design promises a distribution,
    because some assaults have no visible build-up and will always score zero.
    """
    leads = np.asarray([x for x in leads if x > 0], dtype=np.float64)
    if leads.size == 0:
        return {"n": 0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0}
    return {
        "n": int(leads.size),
        "p25": float(np.percentile(leads, 25)),
        "median": float(np.median(leads)),
        "p75": float(np.percentile(leads, 75)),
        "max": float(leads.max()),
    }
