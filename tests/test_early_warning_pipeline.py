"""Tests for the pieces that turn the signals into a running system:
slot assignment, the untrained hazards, leakage-safe windows, and the replay CLI.
"""

import numpy as np

from src.early_warning import windows
from src.early_warning.anticipation_labels import VideoEvents
from src.early_warning.baseline import (
    HeuristicHazard,
    WindowHazard,
    geometry_hazard,
    self_exciting,
)
from src.early_warning.escalation_policy import EscalationPolicy, PolicyConfig, Signals, Tier
from src.early_warning.replay import score_timeline, summarise
from src.early_warning.social_features import FeatureConfig, compute, observation_quality
from src.early_warning.tracked_pose import SlotAssigner
from tests.test_early_warning import scene

FPS = 10.0


def _calm(n=40):
    """Two people standing well apart, not facing each other."""
    return [[(200.0, 500.0, (1, 0)), (900.0, 500.0, (1, 0))]] * n


def _closing(n=40):
    """One person walking straight into another's space, both facing."""
    frames = []
    for t in range(n):
        frames.append([(250.0 + 12.0 * t, 500.0, (1, 0)), (800.0, 500.0, (-1, 0))])
    return frames


# ------------------------------------------------------------------ tracking


def test_slot_assigner_keeps_a_person_in_one_slot():
    slots = SlotAssigner(max_people=4, grace_frames=5)
    first = slots.assign([7, 9], 0)
    for frame in range(1, 10):
        again = slots.assign([7, 9], frame)
        assert again == first  # the same people never shuffle between slots


def test_slot_assigner_survives_a_brief_gap_then_reuses_the_slot():
    slots = SlotAssigner(max_people=2, grace_frames=5)
    mapping = slots.assign([1, 2], 0)
    slot_of_1 = mapping[1]
    slots.assign([2], 1)  # person 1 briefly occluded
    assert slots.assign([1, 2], 3)[1] == slot_of_1  # back within the grace period

    for frame in range(4, 30):  # now person 1 leaves for good
        slots.assign([2], frame)
    assert slots.assign([2, 5], 30)[5] == slot_of_1  # the freed slot is reused


def test_slot_assigner_respects_the_person_limit():
    slots = SlotAssigner(max_people=2)
    mapping = slots.assign([1, 2, 3, 4], 0)
    assert len(set(mapping.values())) <= 2


# ------------------------------------------------------------------ hazards


def test_self_exciting_decays_after_an_impulse():
    x = np.zeros(50)
    x[10] = 1.0
    trace = self_exciting(x, fps=10.0, half_life_s=1.0)
    assert trace[10] == 1.0
    assert np.isclose(trace[20], 0.5, atol=0.05)  # one half-life later
    assert 0.0 < trace[40] < trace[20] < trace[10]  # decays, never quite to zero


def test_heuristic_hazard_rises_on_an_approach_and_stays_low_when_calm():
    calm = compute(*scene(_calm()), FeatureConfig(fps=FPS)).signals
    closing = compute(*scene(_closing()), FeatureConfig(fps=FPS)).signals
    hazard = HeuristicHazard(fps=FPS)
    assert hazard(closing)["hazard_10s"].max() > hazard(calm)["hazard_10s"].max() + 0.2
    assert hazard(calm)["hazard_10s"].max() < 0.35  # below the WATCH threshold


def test_hazard_10s_is_never_below_hazard_5s():
    # "within 10 s" contains "within 5 s"; an inversion would be incoherent.
    signals = compute(*scene(_closing()), FeatureConfig(fps=FPS)).signals
    out = HeuristicHazard(fps=FPS)(signals)
    assert (out["hazard_10s"] >= out["hazard_5s"] - 1e-9).all()


def test_geometry_baseline_responds_to_closing_distance():
    calm = compute(*scene(_calm()), FeatureConfig(fps=FPS)).signals
    closing = compute(*scene(_closing()), FeatureConfig(fps=FPS)).signals
    assert geometry_hazard(closing, FPS).max() > geometry_hazard(calm, FPS).max()


# ------------------------------------------------------------- quality gate


def test_quality_gate_caps_anticipation_at_watch():
    pol = EscalationPolicy(PolicyConfig(enter_dwell_s=0.5, min_quality=0.35))
    for i in range(30):  # a strong build-up signal, but unusable geometry
        pol.update(
            Signals(0.1 * (i + 1), hazard_5s=0.9, hazard_10s=0.9, p_assault=0.0, quality=0.1)
        )
    assert pol.tier == Tier.WATCH  # never summons staff on nonsense


def test_quality_gate_lets_a_confirmed_assault_through():
    # The assault detector has its own abstention, so it is not gated by default.
    pol = EscalationPolicy(PolicyConfig(enter_dwell_s=0.5, min_quality=0.35))
    for i in range(30):
        pol.update(
            Signals(0.1 * (i + 1), hazard_5s=0.9, hazard_10s=0.9, p_assault=0.95, quality=0.1)
        )
    assert pol.tier == Tier.INCIDENT


def test_observation_quality_falls_for_tiny_skeletons():
    frames = [[(300.0, 500.0, (1, 0)), (420.0, 500.0, (-1, 0))]] * 12
    kp, sc = scene(frames)
    good = observation_quality(compute(kp, sc, FeatureConfig(fps=FPS)).geometry, min_height_px=90)
    small = observation_quality(
        compute(kp * 0.15, sc, FeatureConfig(fps=FPS)).geometry, min_height_px=90
    )
    assert good.max() > small.max()


# ---------------------------------------------------------------- windows


def _windowset():
    fight = compute(*scene(_closing(60)), FeatureConfig(fps=FPS)).signals
    calm = compute(*scene(_calm(60)), FeatureConfig(fps=FPS)).signals
    items = [
        (VideoEvents("fight_a", "ds_a", FPS, 60, [(50, 59)]), fight),
        (VideoEvents("calm_a", "ds_a", FPS, 60, []), calm),
        (VideoEvents("fight_b", "ds_b", FPS, 60, [(45, 59)]), fight),
        (VideoEvents("calm_b", "ds_b", FPS, 60, []), calm),
    ]
    return windows.build(items, FPS, stride_s=0.2, min_context_s=0.5)


def test_windows_carry_labels_and_provenance():
    ws = _windowset()
    assert len(ws) > 0
    assert ws.x.shape[1] == len(ws.feature_names)
    assert set(ws.dataset.tolist()) == {"ds_a", "ds_b"}
    assert ws.event.sum() > 0  # the fight videos produce pre-onset positives
    assert (ws.weight[ws.in_assault] == 0).all()  # no supervision inside an assault


def test_split_never_puts_one_video_on_both_sides():
    ws = _windowset()
    train, val = windows.group_split(ws, val_frac=0.5, seed=0)
    assert not set(ws.video_id[train]) & set(ws.video_id[val])


def test_leave_one_dataset_out_holds_out_whole_corpora():
    ws = _windowset()
    folds = list(windows.leave_one_dataset_out(ws))
    assert {name for name, _, _ in folds} == {"ds_a", "ds_b"}
    for name, train, test in folds:
        assert set(ws.dataset[test]) == {name}
        assert name not in set(ws.dataset[train])


def test_balance_negatives_keeps_every_positive():
    ws = _windowset()
    mask = np.ones(len(ws), dtype=bool)
    balanced = windows.balance_negatives(ws, mask, neg_per_pos=1.0, seed=0)
    positives = ws.trainable() & (ws.event == 1)
    assert (balanced & positives).sum() == positives.sum()
    assert balanced.sum() <= mask.sum()


def test_window_hazard_trains_and_predicts():
    ws = _windowset()
    use = ws.trainable()
    model = WindowHazard(horizons_s=(5.0,), max_iter=30).fit(
        ws.x[use], {5.0: ws.targets([5.0])[5.0][use]}, ws.feature_names
    )
    pred = model.predict(ws.x[use])[5.0]
    assert pred.shape == (use.sum(),)
    assert ((pred >= 0) & (pred <= 1)).all()


# ----------------------------------------------------------------- replay


def test_replay_timeline_runs_end_to_end():
    bundle = compute(*scene(_closing(60)), FeatureConfig(fps=FPS))
    quality = observation_quality(bundle.geometry)
    hazard = HeuristicHazard(fps=FPS)(bundle.signals)
    timeline = score_timeline(bundle.signals, quality, FPS, hazard)
    assert len(timeline) == 60
    assert {r["tier"] for r in timeline} <= set(Tier.__members__)
    summary = summarise(timeline)
    assert summary["steps"] == 60
    assert summary["duration_s"] > 0


def test_reasons_rank_by_significance_not_raw_magnitude():
    # largest_group is a head-count and enclosure is a 0-1 score. Ranking on raw
    # values would report "group forming" for a textbook encirclement.
    from src.early_warning.escalation_policy import top_reasons

    signals = {
        "largest_group": np.array([5.0]),
        "enclosure": np.array([0.68]),
        "outnumbering": np.array([4.0]),
    }
    assert "target enclosed by others" in top_reasons(signals, 0, k=2)


def test_replay_stays_calm_on_calm_footage():
    bundle = compute(*scene(_calm(60)), FeatureConfig(fps=FPS))
    hazard = HeuristicHazard(fps=FPS)(bundle.signals)
    timeline = score_timeline(bundle.signals, observation_quality(bundle.geometry), FPS, hazard)
    assert all(r["tier"] == "CALM" for r in timeline)
    assert summarise(timeline)["notifications"] == 0
