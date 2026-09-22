import numpy as np

from src.evaluation.baselines import (
    clip_motion_energy,
    fit_energy_threshold,
    threshold_accuracy,
)


def _reference_energy(kp, sc, min_joints=4):
    """Straight-loop definition the vectorized version must reproduce exactly."""
    per_frame = []
    for t in range(1, kp.shape[0]):
        best = 0.0
        for m in range(kp.shape[1]):
            vis = (sc[t, m] > 0) & (sc[t - 1, m] > 0)
            if vis.sum() < min_joints:
                continue
            ys = kp[t, m][sc[t, m] > 0][:, 1]
            height = max(float(ys.max() - ys.min()), 8.0)
            speed = np.linalg.norm(kp[t, m][vis] - kp[t - 1, m][vis], axis=1).mean() / height
            best = max(best, float(speed))
        per_frame.append(best)
    return float(np.median(per_frame))


def _walk(t=30, step=2.0, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    body = rng.uniform(0, 100, size=(1, 2, 17, 2))
    drift = rng.normal(0, step, size=(t, 2, 17, 2)).cumsum(axis=0)
    return ((body + drift) * scale).astype("float32"), np.ones((t, 2, 17), dtype="float32")


def test_vectorized_energy_matches_the_loop_definition():
    rng = np.random.default_rng(1)
    for seed in range(5):
        kp, sc = _walk(seed=seed)
        sc = sc * (rng.random(sc.shape) > 0.3)  # knock out ~30% of joints
        # float32 keypoints summed in a different order: agree to float32 precision.
        assert np.isclose(clip_motion_energy(kp, sc), _reference_energy(kp, sc), rtol=1e-6)


def test_motionless_clip_has_zero_energy():
    kp = np.tile(np.random.default_rng(2).uniform(0, 100, (1, 2, 17, 2)), (20, 1, 1, 1))
    assert clip_motion_energy(kp.astype("float32"), np.ones((20, 2, 17), "float32")) == 0.0


def test_faster_motion_scores_higher():
    slow, sc = _walk(step=1.0)
    fast, _ = _walk(step=5.0)
    assert clip_motion_energy(fast, sc) > clip_motion_energy(slow, sc)


def test_energy_is_camera_distance_invariant():
    # The same movement filmed at twice the pixel scale is the same movement.
    near, sc = _walk(scale=2.0)
    far, _ = _walk(scale=1.0)
    assert abs(clip_motion_energy(near, sc) - clip_motion_energy(far, sc)) < 1e-6


def test_energy_handles_degenerate_clips():
    assert clip_motion_energy(np.zeros((1, 2, 17, 2)), np.ones((1, 2, 17))) == 0.0
    assert clip_motion_energy(np.zeros((10, 2, 17, 2)), np.zeros((10, 2, 17))) == 0.0


def test_threshold_separates_separable_data_perfectly():
    energies = [0.01, 0.02, 0.03, 0.10, 0.20, 0.30]
    labels = [0, 0, 0, 1, 1, 1]
    t = fit_energy_threshold(energies, labels)
    assert threshold_accuracy(energies, labels, t) == 1.0


def test_threshold_never_splits_inside_a_run_of_ties():
    # Four motionless clips (two of each class) at exactly 0.0: no threshold can put
    # some of them on each side, so the reported accuracy must be one that a real
    # threshold achieves.
    energies = [0.0, 0.0, 0.0, 0.0, 0.5, 0.6]
    labels = [0, 0, 1, 1, 1, 1]
    t = fit_energy_threshold(energies, labels)
    achieved = threshold_accuracy(energies, labels, t)
    best_possible = max(threshold_accuracy(energies, labels, c) for c in (-1.0, 0.0, 0.55, 0.7))
    assert achieved == best_possible


def test_threshold_calls_everything_aggressive_when_that_is_best():
    t = fit_energy_threshold([0.1, 0.2, 0.3], [1, 1, 1])
    assert t == -np.inf
    assert threshold_accuracy([0.1, 0.2, 0.3], [1, 1, 1], t) == 1.0


def test_threshold_handles_empty_input():
    assert threshold_accuracy([], [], 0.5) == 0.0
    assert fit_energy_threshold([], []) == 0.0
