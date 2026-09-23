"""Baselines for escalation hazard, from "no training at all" upwards.

Three rungs, in the order you should try them:

1. ``geometry_hazard`` - coarse pair geometry only (how close, how fast closing).
   Deliberately crude, because the one controlled study of pre-event separability
   (Ganesh 2026) found that no pose representation beat coarse bounding-box
   geometry. Anything richer has to beat this before it earns its place.
2. ``heuristic_hazard`` - the interpretable social signals combined by hand,
   with a self-exciting term so a shove sharply raises short-horizon hazard
   (borrowed from the Hawkes-process aggression-forecasting literature). Needs
   no labels, so it runs on day one and gives the annotation pass something to
   argue with.
3. ``WindowHazard`` - gradient boosting on trailing-window statistics, the first
   trained model. scikit-learn, already a dependency; no new install.

None of these outputs is calibrated. They are scores in [0, 1] whose threshold
must be set from a false-alarm budget on real footage (metrics.threshold_for_budget),
not read as probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _squash(x, lo: float, hi: float) -> np.ndarray:
    """Linear ramp from lo to hi, clipped to [0, 1]."""
    return np.clip((np.asarray(x, dtype=np.float64) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def self_exciting(impulse, fps: float, half_life_s: float = 3.0) -> np.ndarray:
    """Decaying trace of an impulse signal: e_t = max(x_t, e_{t-1} * decay).

    A shove does not stop mattering the instant it ends. This is the
    self-exciting idea from Hawkes-process onset forecasting, reduced to its
    simplest causal form.
    """
    x = np.nan_to_num(np.asarray(impulse, dtype=np.float64))
    decay = 0.5 ** (1.0 / max(half_life_s * fps, 1e-6))
    out = np.zeros_like(x)
    running = 0.0
    for i, v in enumerate(x):
        running = max(v, running * decay)
        out[i] = running
    return out


def geometry_hazard(signals: dict, fps: float) -> np.ndarray:
    """Baseline 1: closeness and closing speed only. No pose, no group structure."""
    close = 1.0 - _squash(signals["min_pair_dist"], 0.5, 2.5)  # 1 when almost touching
    closing = _squash(signals["max_closing"], 0.1, 1.0)
    return np.clip(0.7 * close + 0.5 * closing, 0.0, 1.0)


@dataclass
class HeuristicWeights:
    """Starting points. Every number here is a hypothesis, not a measurement."""

    # precursor: hostile posture between two specific people
    facing: float = 0.45
    proximity: float = 0.30
    gesture: float = 0.25
    advance: float = 0.20
    # build-up: spatial or physical pressure on a target
    enclosure: float = 0.70
    outnumber: float = 0.35
    cornered: float = 0.65
    contact: float = 0.40
    converging: float = 0.25
    # self-exciting contact term
    shove: float = 0.80
    shove_half_life_s: float = 3.0


@dataclass
class HeuristicHazard:
    """Untrained hazard from the social signals. Returns scores, not probabilities."""

    fps: float = 10.0
    w: HeuristicWeights = field(default_factory=HeuristicWeights)

    def precursor(self, s: dict) -> np.ndarray:
        """Two people squared up at close range, gesturing or advancing."""
        near = 1.0 - _squash(s["min_pair_dist"], 0.8, 2.0)
        facing = _squash(s["max_mutual_facing"], 0.3, 0.8)
        gesture = _squash(s["gesture_close_still"], 0.2, 1.2)
        advance = _squash(s["max_advance"], 0.2, 1.0)
        arm = _squash(s["arm_raise_close"], 0.5, 1.0)
        w = self.w
        # Facing AND proximity gate the rest: gesturing alone is just conversation.
        core = np.minimum(facing, near)
        return np.clip(
            (w.facing + w.proximity) * core
            + w.gesture * gesture * core
            + w.advance * advance * core
            + 0.15 * arm * core,
            0.0,
            1.0,
        )

    def buildup(self, s: dict) -> np.ndarray:
        """Pressure on a target: enclosed, outnumbered, cornered, shoved."""
        w = self.w
        enclosure = _squash(s["enclosure"], 0.25, 0.7)
        outnumber = _squash(s["outnumbering"], 1.5, 3.0)
        cornered = _squash(s["cornered"], 0.2, 0.7)
        contact = _squash(s["contact"], 0.5, 1.0)
        converging = _squash(s["converging"], 1.0, 3.0)
        shove = self_exciting(_squash(s["shove"], 0.2, 1.0), self.fps, w.shove_half_life_s)
        return np.clip(
            w.enclosure * enclosure
            + w.outnumber * outnumber
            + w.cornered * cornered
            + w.contact * contact
            + w.converging * converging
            + w.shove * shove,
            0.0,
            1.0,
        )

    def __call__(self, signals: dict) -> dict:
        """All four series the escalation policy needs."""
        pre = self.precursor(signals)
        build = self.buildup(signals)
        shove = self_exciting(
            _squash(signals["shove"], 0.2, 1.0), self.fps, self.w.shove_half_life_s
        )
        # Short horizon leans on physical pressure and recent contact; the longer
        # horizon accepts posture alone, because that is all there is early on.
        hazard_5s = np.clip(0.85 * build + 0.35 * pre + 0.30 * shove, 0.0, 1.0)
        hazard_10s = np.clip(0.60 * build + 0.70 * pre, 0.0, 1.0)
        return {
            "precursor": pre,
            "buildup": build,
            "hazard_5s": hazard_5s,
            "hazard_10s": np.maximum(hazard_10s, hazard_5s),  # 10 s includes the next 5
        }


class WindowHazard:
    """Gradient boosting on trailing-window statistics, one model per horizon.

    Fitted on window features from social_features.window_stats and the binary
    target "an assault starts within `horizon_s`". Windows inside an assault and
    zero-weight windows (the onset-jitter band) must be dropped before fitting.
    """

    def __init__(self, horizons_s=(2.0, 5.0, 10.0), **kwargs):
        self.horizons_s = tuple(horizons_s)
        # update(), not dict(**defaults, **kwargs): the caller must be able to
        # override a default rather than collide with it.
        self.kwargs = {"max_depth": 3, "learning_rate": 0.06, "max_iter": 300}
        self.kwargs.update(kwargs)
        self.models: dict[float, object] = {}
        self.feature_names: list[str] = []

    def fit(
        self,
        x: np.ndarray,
        targets: dict[float, np.ndarray],
        feature_names=None,
        sample_weight=None,
    ):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.feature_names = list(feature_names or [])
        for h in self.horizons_s:
            y = np.asarray(targets[h])
            model = HistGradientBoostingClassifier(**self.kwargs)
            if len(np.unique(y)) < 2:
                # A fold with no positives would raise; keep an explicit stub so
                # the leave-one-dataset-out loop reports it instead of crashing.
                self.models[h] = None
                continue
            model.fit(x, y, sample_weight=sample_weight)
            self.models[h] = model
        return self

    def predict(self, x: np.ndarray) -> dict[float, np.ndarray]:
        out = {}
        for h, model in self.models.items():
            out[h] = (
                np.zeros(len(x))
                if model is None
                else model.predict_proba(x)[:, 1].astype(np.float64)
            )
        return out
