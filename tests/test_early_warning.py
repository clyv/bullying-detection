"""Tests for the escalation early-warning module, using synthetic scenes.

The synthetic skeletons are deliberately simple; they check that each signal
responds to the situation it is meant to capture, not that it is calibrated.
"""

import numpy as np
import pytest

from src.early_warning import metrics
from src.early_warning.anticipation_labels import VideoEvents, make_window_labels
from src.early_warning.escalation_policy import EscalationPolicy, PolicyConfig, Signals, Tier
from src.early_warning.social_features import (
    EDGE_KEYS,
    PERSON_KEYS,
    SCENE_KEYS,
    TARGET_KEYS,
    FeatureConfig,
    compute,
    to_model_inputs,
)

H = 200.0  # body height in pixels for synthetic people


def skeleton(fx: float, fy: float, facing: tuple[float, float]):
    """COCO-17 keypoints for a standing person with feet at (fx, fy).

    facing is an image-plane direction; +y means facing the camera.
    """
    lx, fz = facing
    kp = np.zeros((17, 2), np.float32)
    sc = np.ones(17, np.float32)
    spread = 0.1 * H * abs(fz) + 0.02 * H
    sgn = 1.0 if fz > 0 else -1.0  # facing the camera puts the left shoulder on the image right

    def pair(i_left: int, i_right: int, y: float, s: float) -> None:
        kp[i_left] = (fx + sgn * s, y)
        kp[i_right] = (fx - sgn * s, y)

    pair(5, 6, fy - 0.80 * H, spread)
    pair(7, 8, fy - 0.65 * H, spread + 0.02 * H)
    pair(9, 10, fy - 0.52 * H, spread + 0.02 * H)
    pair(11, 12, fy - 0.50 * H, 0.7 * spread)
    pair(13, 14, fy - 0.25 * H, 0.6 * spread)
    pair(15, 16, fy, 0.5 * spread)
    kp[0] = (fx + 0.05 * H * lx, fy - 0.92 * H)
    kp[1:5] = kp[0]
    if fz < -0.3:  # back to the camera: the face is not visible
        sc[:5] = 0.0
    return kp, sc


def scene(frames):
    """frames: per-timestep lists of (fx, fy, facing) -> keypoints, scores."""
    T, M = len(frames), max(len(f) for f in frames)
    kp = np.zeros((T, M, 17, 2), np.float32)
    sc = np.zeros((T, M, 17), np.float32)
    for t, people in enumerate(frames):
        for m, (fx, fy, facing) in enumerate(people):
            kp[t, m], sc[t, m] = skeleton(fx, fy, facing)
    return kp, sc


def test_encircled_target_beats_conversation_ring():
    cx, cy, r = 500.0, 500.0, 0.6 * H
    ring = [
        (cx - r, cy, (1, 0)),
        (cx + r, cy, (-1, 0)),
        (cx, cy - r, (0, 1)),
        (cx, cy + r, (0, -1)),
    ]
    target = [(cx, cy, (0, 1))]
    cfg = FeatureConfig(fps=10)
    encircled = compute(*scene([ring + target] * 20), cfg)
    conversation = compute(*scene([ring] * 20), cfg)
    assert encircled.signals["enclosure"][-1] > 0.6
    assert encircled.signals["outnumbering"][-1] == 4
    assert conversation.signals["enclosure"][-1] < 0.35


def test_cornering_retreat_then_stall():
    fps = 10
    step = 0.8 * H / fps  # 0.8 body heights per second
    ax, tx = 300.0, 300.0 + H
    frames = []
    for t in range(31):
        if t > 0:
            ax += step  # aggressor keeps advancing
            if t <= 20:
                tx += step  # target backs away, then hits the wall at t = 2 s
        frames.append([(ax, 500.0, (1, 0)), (tx, 500.0, (-1, 0))])
    cornered = compute(*scene(frames), FeatureConfig(fps=fps)).target["cornered"][:, 1]
    assert cornered[10] < 0.1  # both still moving: no cornering yet
    assert cornered[30] > 0.5  # target stalled while the aggressor keeps closing


def test_model_input_shapes():
    frames = [[(300.0, 500.0, (1, 0)), (450.0, 500.0, (-1, 0)), (380.0, 380.0, (0, 1))]] * 12
    x = to_model_inputs(compute(*scene(frames), FeatureConfig(fps=10)))
    assert x["person"].shape == (12, 3, len(PERSON_KEYS) + len(TARGET_KEYS))
    assert x["edge"].shape == (12, 3, 3, len(EDGE_KEYS))
    assert x["scene"].shape == (12, len(SCENE_KEYS))
    assert np.isfinite(x["person"]).all() and np.isfinite(x["edge"]).all()


def test_window_labels_bins_and_censoring():
    ev = VideoEvents("v", "test", fps=10.0, n_frames=200, assault=[(100, 140)])
    labels = {
        lab.end_frame: lab
        for lab in make_window_labels(
            ev, horizon_s=10, bin_s=1, stride_s=0.1, onset_jitter_s=0.5, min_context_s=0
        )
    }
    assert labels[95].event == 1 and labels[95].bin == 0 and labels[95].weight == 0.0  # jitter band
    assert labels[85].event == 1 and labels[85].bin == 1 and labels[85].weight == 1.0  # 1.5 s ahead
    assert labels[0].event == 1 and labels[0].bin == 9  # exactly 10 s ahead
    assert labels[120].in_assault and labels[120].weight == 0.0
    assert labels[150].event == 0 and labels[150].bin == 10  # after the fight: censored

    quiet = VideoEvents("n", "test", fps=10.0, n_frames=100, assault=[])
    assert all(lab.event == 0 for lab in make_window_labels(quiet))


def test_anticipation_metrics_reward_early_alarms():
    fps, onset = 10.0, 100
    ramp = np.clip((np.arange(150) - 70) / 30.0, 0, 1)  # rises over the 3 s before onset
    flat = np.full(3000, 0.1)  # five minutes of calm footage
    row = metrics.anticipation_curve([ramp], [onset], [flat], fps, thresholds=[0.5])[0]
    assert row["recall"] == 1.0 and row["precision"] == 1.0
    assert row["tta"] == pytest.approx(1.5, abs=0.11)
    assert metrics.false_alarms_per_hour([flat], 0.5, fps) == 0.0

    res = metrics.anticipation_at_budget([ramp], [onset], [flat], fps, budget_per_hour=0.15)
    assert res["recall"] == 1.0 and res["median_lead_s"] > 2.0
    assert res["false_alarms_per_hour"] <= 0.15


def test_alarm_after_onset_is_detection_not_anticipation():
    # A score that only rises once the fight is under way must earn zero lead time.
    fps, onset = 10.0, 100
    late = np.zeros(150)
    late[onset:] = 1.0
    assert metrics.lead_times([late], [onset], 0.5, fps)[0] == 0.0


def test_policy_escalates_with_dwell_and_hysteresis():
    pol = EscalationPolicy(PolicyConfig(enter_dwell_s=1.0, exit_dwell_s=2.0))
    clock = {"t": 0.0}

    def run(h10: float, h5: float, pa: float, seconds: float) -> None:
        for _ in range(int(seconds * 10)):
            clock["t"] += 0.1
            pol.update(Signals(clock["t"], hazard_5s=h5, hazard_10s=h10, p_assault=pa))

    run(0.1, 0.1, 0.0, 2)
    assert pol.tier == Tier.CALM
    run(0.5, 0.3, 0.0, 2)
    assert pol.tier == Tier.WATCH
    run(0.8, 0.7, 0.0, 2)
    assert pol.tier == Tier.WARN
    run(0.9, 0.9, 0.95, 2)
    assert pol.tier == Tier.INCIDENT
    run(0.1, 0.1, 0.0, 10)
    assert pol.tier == Tier.CALM


def test_policy_ignores_brief_spikes():
    pol = EscalationPolicy(PolicyConfig(enter_dwell_s=1.0))
    for i in range(5):  # 0.5 s spike, shorter than the dwell time
        pol.update(Signals(0.1 * (i + 1), hazard_5s=0.9, hazard_10s=0.9, p_assault=0.0))
    for i in range(5, 30):
        pol.update(Signals(0.1 * (i + 1), hazard_5s=0.1, hazard_10s=0.1, p_assault=0.0))
    assert pol.tier == Tier.CALM


def test_escalation_net_shapes_and_loss():
    torch = pytest.importorskip("torch")
    from src.early_warning.hazard_model import EscalationNet, escalation_loss, within_horizon_prob

    B, T, N, K = 2, 6, 5, 10
    P, E, S = len(PERSON_KEYS) + len(TARGET_KEYS), len(EDGE_KEYS), len(SCENE_KEYS)
    net = EscalationNet(P, E, S, n_bins=K)
    mask = torch.ones(B, T, N, dtype=torch.bool)
    mask[:, :, -1] = False  # one empty slot
    mask[0, 0] = False  # a frame with nobody in it
    out = net(torch.randn(B, T, N, P), torch.randn(B, T, N, N, E), torch.randn(B, T, S), mask)
    assert out["hazard_logits"].shape == (B, T, K)
    assert out["role_logits"].shape == (B, T, N, 2)
    batch = {
        "event": torch.randint(0, 2, (B, T)),
        "bin": torch.randint(0, K + 1, (B, T)),
        "weight": torch.ones(B, T),
        "in_assault": torch.zeros(B, T, dtype=torch.bool),
        "phase": torch.full((B, T), -1),
        "valid": torch.ones(B, T, dtype=torch.bool),
    }
    loss = escalation_loss(out, batch)["total"]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    p5 = within_horizon_prob(out["hazard_logits"], 5)
    assert ((p5 >= 0) & (p5 <= 1)).all()
