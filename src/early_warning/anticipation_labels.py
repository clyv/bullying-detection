"""Time-to-onset labels for escalation anticipation.

The task is framed like traffic-accident anticipation: at each window end t,
"will an assault start within the next H seconds, and when?" Only the assault
onset time is needed, which is objective and already annotated in UBI-Fights,
NTU-CCTV-Fights and UT-Interaction. Precursor / build-up phase intervals are
optional and come from our own annotation pass (design doc, section 5).

Label rules per window end t, with tau = seconds to the next assault onset:
    inside an assault         -> in_assault=True, no hazard supervision
    0 < tau <= onset_jitter_s -> event, weight 0: annotators disagree about the
                                 exact first blow, and catching the first blow
                                 early must not be credited as anticipation
    tau <= horizon_s          -> event in bin ceil(tau / bin_s) - 1
    otherwise / no onset      -> right-censored at the horizon, i.e. a true
                                 negative for "no assault within H seconds"

Censoring matters: a quiet video is not evidence that nothing will ever happen,
only that nothing happened while the camera was running. Treating those windows
as plain negatives is how a model learns that build-ups are safe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# UT-Interaction class ids as used in the segmented filenames
# (0 handshake, 1 hug, 2 kick, 3 point, 4 punch, 5 push).
# Verify against the labels spreadsheet before trusting this mapping.
UT_ASSAULT = {2, 4, 5}
UT_FRIENDLY = {0, 1, 3}

CALM, PRECURSOR, BUILDUP, ASSAULT, UNKNOWN = 0, 1, 2, 3, -1
PHASE_NAMES = ("calm", "precursor", "build-up", "assault")


@dataclass
class VideoEvents:
    video_id: str
    dataset: str
    fps: float
    n_frames: int
    assault: list[tuple[int, int]]  # inclusive frame intervals
    friendly: list[tuple[int, int]] = field(default_factory=list)  # non-assault interactions
    phases: list[tuple[int, int, int]] = field(default_factory=list)  # (start, end, phase id)
    phase_annotated: bool = False


@dataclass
class WindowLabel:
    end_frame: int
    event: int  # 1 if an assault starts within the horizon
    bin: int  # onset bin if event, else n_bins (censored)
    tau_s: float  # seconds to the next onset (inf if none)
    in_assault: bool
    phase: int  # CALM / PRECURSOR / BUILDUP / ASSAULT, or UNKNOWN if not annotated
    weight: float


def _phase_at(ev: VideoEvents, t: int, in_assault: bool) -> int:
    if in_assault:
        return ASSAULT
    if not ev.phase_annotated:
        return UNKNOWN
    for start, end, phase in ev.phases:
        if start <= t <= end:
            return phase
    return CALM


def make_window_labels(
    ev: VideoEvents,
    horizon_s: float = 10.0,
    bin_s: float = 1.0,
    stride_s: float = 0.5,
    onset_jitter_s: float = 0.5,
    min_context_s: float = 2.0,
) -> list[WindowLabel]:
    """One label per window end, every `stride_s`, after `min_context_s` of history."""
    n_bins = round(horizon_s / bin_s)
    stride = max(1, round(stride_s * ev.fps))
    start = round(min_context_s * ev.fps)
    onsets = np.array(sorted(s for s, _ in ev.assault), dtype=np.int64)
    labels = []
    for t in range(start, ev.n_frames, stride):
        in_assault = any(s <= t <= e for s, e in ev.assault)
        later = onsets[onsets > t]
        tau = float((later[0] - t) / ev.fps) if later.size else float("inf")
        phase = _phase_at(ev, t, in_assault)
        if in_assault:
            labels.append(WindowLabel(t, 0, n_bins, tau, True, phase, 0.0))
            continue
        weight = 0.0 if tau <= onset_jitter_s else 1.0
        if tau <= horizon_s:
            b = min(n_bins - 1, max(0, int(np.ceil(tau / bin_s)) - 1))
            labels.append(WindowLabel(t, 1, b, tau, False, phase, weight))
        else:
            labels.append(WindowLabel(t, 0, n_bins, tau, False, phase, weight))
    return labels


def labels_to_arrays(labels: list[WindowLabel]) -> dict[str, np.ndarray]:
    """Stack window labels into arrays for a training dataset."""
    return {
        "end_frame": np.array([x.end_frame for x in labels], dtype=np.int64),
        "event": np.array([x.event for x in labels], dtype=np.int64),
        "bin": np.array([x.bin for x in labels], dtype=np.int64),
        "tau_s": np.array([x.tau_s for x in labels], dtype=np.float64),
        "in_assault": np.array([x.in_assault for x in labels], dtype=bool),
        "phase": np.array([x.phase for x in labels], dtype=np.int64),
        "weight": np.array([x.weight for x in labels], dtype=np.float32),
    }


def within_horizon_target(labels: list[WindowLabel], horizon_s: float) -> np.ndarray:
    """Binary target "an assault starts within `horizon_s`", for the Stage 0 baseline."""
    return np.array(
        [1 if (x.event == 1 and x.tau_s <= horizon_s) else 0 for x in labels], dtype=np.int64
    )


# ---------------------------------------------------------------- loaders


def intervals_from_mask(mask) -> list[tuple[int, int]]:
    """Inclusive (start, end) runs of True in a per-frame boolean mask."""
    m = np.asarray(mask, dtype=bool).ravel().astype(np.int8)
    edges = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def load_ubi_fights(annotation_dir: str | Path, fps: float = 30.0) -> list[VideoEvents]:
    """One CSV of per-frame 0/1 labels per video.

    Verify this against your copy of the release and adjust the parsing if the
    layout differs (header row, one row per frame vs one line of values, etc.).
    Videos with no positive frames become pure negatives, which is what the
    false-alarm-per-hour metric needs.
    """
    events = []
    for path in sorted(Path(annotation_dir).glob("*.csv")):
        mask = np.loadtxt(path, delimiter=",", ndmin=1).ravel() > 0.5
        events.append(
            VideoEvents(path.stem, "ubi_fights", fps, int(mask.size), intervals_from_mask(mask))
        )
    return events


def load_phase_annotation(path: str | Path) -> VideoEvents:
    """Read one annotation JSON in the schema from the design doc (section 5)."""
    d = json.loads(Path(path).read_text())
    return VideoEvents(
        video_id=d["video_id"],
        dataset=d["dataset"],
        fps=float(d["fps"]),
        n_frames=int(d["n_frames"]),
        assault=[tuple(x) for x in d.get("assault", [])],
        friendly=[tuple(x) for x in d.get("friendly", [])],
        phases=[tuple(x) for x in d.get("phases", [])],
        phase_annotated=bool(d.get("phase_annotated", False)),
    )
