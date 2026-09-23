"""Tiered alert policy on top of the escalation model's outputs.

Tiers: CALM -> WATCH (highlight the camera, no notification) -> WARN (notify
the nearest staff member to go and supervise) -> INCIDENT (assault in
progress). A higher tier's signal must persist for enter_dwell_s before it
takes effect; lower exit thresholds plus exit_dwell_s give hysteresis so
alerts don't flicker, and tiers step down one at a time.

Two deliberate design choices:

* **The tiers are defined by what a human is asked to do**, following the
  operator-defined scale in the train-aggression literature (medium = increase
  attention, high = act). WATCH never notifies anyone. Most confrontations never
  become violent, so a precursor must not be an alarm.
* **A quality gate.** Tiny or occluded skeletons produce confident-looking
  nonsense. Below `min_quality` the anticipation tiers are capped at WATCH, so
  bad geometry can raise attention but never summon staff. This is an explicit
  abstention, the same idea as the conformal abstention on the detection side.

The default thresholds are starting points only. Set them per camera from
the false-alarm budget on that camera's normal footage
(metrics.threshold_for_budget), after a shadow-mode period with no alerts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Tier(IntEnum):
    CALM = 0
    WATCH = 1
    WARN = 2
    INCIDENT = 3


@dataclass
class PolicyConfig:
    watch_enter: float = 0.35
    watch_exit: float = 0.25
    warn_enter: float = 0.55
    warn_exit: float = 0.40
    incident_enter: float = 0.80
    incident_exit: float = 0.50
    enter_dwell_s: float = 1.0
    exit_dwell_s: float = 4.0
    cooldown_s: float = 30.0  # minimum gap between WARN notifications per camera
    min_quality: float = 0.35  # below this, anticipation tiers are capped at WATCH
    # The assault detector has its own abstention (pose-quality gate, conformal
    # threshold), so by default a confirmed assault is still allowed through on
    # poor geometry. Set True to make the quality gate cap INCIDENT too.
    quality_caps_incident: bool = False


@dataclass
class Signals:
    t_s: float
    hazard_5s: float
    hazard_10s: float
    p_assault: float
    p_precursor: float = 0.0
    p_buildup: float = 0.0
    quality: float = 1.0  # 0..1, see social_features.observation_quality
    reasons: dict = field(default_factory=dict)  # top contributing signals, shown to staff


@dataclass
class Alert:
    t_s: float
    tier: Tier
    previous: Tier
    notify: bool
    reasons: dict
    quality: float = 1.0
    gated: bool = False  # True when the quality gate held the tier down


class EscalationPolicy:
    """One instance per camera. Call update() once per model output."""

    def __init__(self, cfg: PolicyConfig | None = None):
        self.cfg = cfg or PolicyConfig()
        self.tier = Tier.CALM
        self._above_since: float | None = None
        self._below_since: float | None = None
        self._last_notify = float("-inf")
        self.gated = False

    def _level(self, s: Signals) -> Tier:
        """Highest tier whose enter condition currently holds."""
        c = self.cfg
        if s.p_assault >= c.incident_enter:
            return Tier.INCIDENT
        if max(s.hazard_5s, s.p_buildup) >= c.warn_enter:
            return Tier.WARN
        if max(s.hazard_10s, s.p_precursor) >= c.watch_enter:
            return Tier.WATCH
        return Tier.CALM

    def _cap(self, level: Tier, s: Signals) -> tuple[Tier, bool]:
        """Apply the observation-quality gate. Returns (level, was_gated)."""
        c = self.cfg
        if s.quality >= c.min_quality or level <= Tier.WATCH:
            return level, False
        if level == Tier.INCIDENT and not c.quality_caps_incident:
            return level, False
        return Tier.WATCH, True

    def _holds(self, s: Signals, tier: Tier) -> bool:
        """Whether the current tier's (lower) exit threshold is still exceeded."""
        c = self.cfg
        if tier == Tier.INCIDENT:
            return s.p_assault >= c.incident_exit
        if tier == Tier.WARN:
            return max(s.hazard_5s, s.p_buildup) >= c.warn_exit
        if tier == Tier.WATCH:
            return max(s.hazard_10s, s.p_precursor) >= c.watch_exit
        return True

    def update(self, s: Signals) -> Alert | None:
        """Advance the state machine; returns an Alert only when the tier changes."""
        c, prev = self.cfg, self.tier
        eps = 1e-9

        level, self.gated = self._cap(self._level(s), s)
        if level > self.tier:
            if self._above_since is None:
                self._above_since = s.t_s
            if s.t_s - self._above_since >= c.enter_dwell_s - eps:
                self.tier, self._above_since = level, None
        else:
            self._above_since = None

        if self.tier == prev and self.tier > Tier.CALM and not self._holds(s, self.tier):
            if self._below_since is None:
                self._below_since = s.t_s
            if s.t_s - self._below_since >= c.exit_dwell_s - eps:
                self.tier, self._below_since = Tier(self.tier - 1), None
        else:
            self._below_since = None

        if self.tier == prev:
            return None
        rising = self.tier > prev
        notify = (
            rising
            and self.tier >= Tier.WARN
            and (self.tier == Tier.INCIDENT or s.t_s - self._last_notify >= c.cooldown_s)
        )
        if notify:
            self._last_notify = s.t_s
        return Alert(s.t_s, self.tier, prev, notify, s.reasons, s.quality, self.gated)


def top_reasons(signals: dict, index: int, k: int = 3) -> dict:
    """The k strongest named signals at one timestep, for the alert text.

    Staff act on "three people facing one person who backed away and stopped",
    not on a probability. Reasons are observable behaviour, never inferred
    emotion or intent.
    """
    # (phrase, scale) per signal. Ranking on raw values would be meaningless:
    # largest_group is a count of people and would always outrank enclosure,
    # which is a 0-1 score. Each value is divided by a typical "this is notable"
    # level before the comparison.
    readable = {
        "enclosure": ("target enclosed by others", 0.7),
        "outnumbering": ("outnumbered at close range", 3.0),
        "cornered": ("backed away, then stopped", 0.7),
        "shove": ("contact followed by recoil", 1.0),
        "contact": ("physical contact", 1.0),
        "max_closing": ("rapid interpersonal closure", 1.0),
        "max_mutual_facing": ("squared up face to face", 0.8),
        "max_advance": ("one person advancing", 1.0),
        "gesture_close_still": ("hard gesturing at close range", 1.2),
        "arm_raise_close": ("raised arm near another person", 1.0),
        "converging": ("others converging on the group", 3.0),
        "ring_still": ("still, inward-facing ring", 1.0),
        "largest_group": ("group forming", 6.0),
    }
    scored = []
    for name, series in signals.items():
        if name not in readable:
            continue
        value = float(series[index])
        if value > 0:
            scored.append((value / readable[name][1], value, name))
    scored.sort(reverse=True)
    return {readable[name][0]: round(value, 3) for _, value, name in scored[:k]}
