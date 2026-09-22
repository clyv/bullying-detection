import numpy as np

from src.datasets.augment import (
    COCO_FLIP_PAIRS,
    AugmentConfig,
    apply_geometric,
    apply_structural,
)


def _clip(t=20, m=2, v=17):
    rng = np.random.default_rng(0)
    kp = rng.normal(200, 40, size=(t, m, v, 2)).astype("float32")
    scores = np.ones((t, m, v), dtype="float32")
    return kp, scores


def _only(**kwargs):
    """An AugmentConfig with everything off except the named transforms."""
    off = {
        "joint_dropout": 0.0,
        "person_dropout": 0.0,
        "coord_noise": 0.0,
        "flip_prob": 0.0,
        "person_swap_prob": 0.0,
        "temporal_crop": 0.0,
        "scale_jitter": 0.0,
        "rotate_degrees": 0.0,
        "shear": 0.0,
    }
    off.update(kwargs)
    return AugmentConfig(**off)


def test_joint_dropout_zeroes_both_coords_and_scores():
    kp, scores = _clip()
    rng = np.random.default_rng(1)
    out_kp, out_scores = apply_structural(kp, scores, rng, _only(joint_dropout=0.5))
    dropped = out_scores == 0
    assert dropped.any(), "expected some joints to be dropped at p=0.5"
    # A dropped joint must not leave stale coordinates behind — normalize_skeleton
    # keys off the score mask, so a nonzero coord with a zero score is a landmine.
    assert (out_kp[dropped] == 0).all()


def test_joint_dropout_off_is_a_no_op():
    kp, scores = _clip()
    out_kp, out_scores = apply_structural(kp, scores, np.random.default_rng(1), _only())
    assert np.array_equal(out_kp, kp) and np.array_equal(out_scores, scores)


def test_structural_does_not_mutate_the_caller_arrays():
    # kp/scores come straight out of an .npz; mutating them would corrupt the cache
    # view for every later epoch.
    kp, scores = _clip()
    kp_before, scores_before = kp.copy(), scores.copy()
    apply_structural(kp, scores, np.random.default_rng(2), _only(joint_dropout=0.9))
    assert np.array_equal(kp, kp_before) and np.array_equal(scores, scores_before)


def test_flip_swaps_left_and_right_joints():
    kp, scores = _clip()
    scores[:, :, 5] = 0.11  # tag left shoulder
    scores[:, :, 6] = 0.99  # tag right shoulder
    _out_kp, out_scores = apply_structural(
        kp, scores, np.random.default_rng(3), _only(flip_prob=1.0)
    )
    # After a mirror, the joint sitting at index 5 must be the former right shoulder.
    assert np.allclose(out_scores[:, :, 5], 0.99)
    assert np.allclose(out_scores[:, :, 6], 0.11)


def test_flip_preserves_vertical_coordinates():
    kp, scores = _clip()
    out_kp, _ = apply_structural(kp, scores, np.random.default_rng(4), _only(flip_prob=1.0))
    # A horizontal mirror must leave y untouched (modulo the L/R index swap).
    assert np.allclose(np.sort(out_kp[..., 1], axis=2), np.sort(kp[..., 1], axis=2))


def test_flip_pairs_cover_every_lateral_joint():
    flat = [j for pair in COCO_FLIP_PAIRS for j in pair]
    assert len(flat) == len(set(flat))  # no joint listed twice
    assert set(flat) == set(range(1, 17))  # everything except the nose


def test_person_swap_preserves_the_set_of_skeletons():
    kp, scores = _clip()
    kp[:, 0] = 1.0
    kp[:, 1] = 2.0
    out_kp, _ = apply_structural(kp, scores, np.random.default_rng(5), _only(person_swap_prob=1.0))
    # Aggression is symmetric under relabelling who is person 0, so both skeletons
    # must survive — only their order may change.
    people = {float(out_kp[0, i, 0, 0]) for i in range(2)}
    assert people == {1.0, 2.0}


def test_temporal_crop_shortens_the_clip():
    kp, scores = _clip(t=100)
    out_kp, out_scores = apply_structural(
        kp, scores, np.random.default_rng(6), _only(temporal_crop=0.3)
    )
    assert out_kp.shape[0] < 100
    assert out_kp.shape[0] == out_scores.shape[0]
    assert out_kp.shape[1:] == kp.shape[1:]


def test_geometric_keeps_missing_joints_at_the_origin():
    kp = np.ones((4, 2, 17, 2), dtype="float32")
    kp[:, :, 3] = 0.0  # joint 3 already zeroed by normalization
    out = apply_geometric(kp, np.random.default_rng(7), _only(rotate_degrees=30.0, shear=0.2))
    assert (out[:, :, 3] == 0).all()


def test_rotation_preserves_distances():
    kp = np.random.default_rng(8).normal(size=(3, 2, 17, 2)).astype("float32")
    out = apply_geometric(kp, np.random.default_rng(9), _only(rotate_degrees=45.0))
    before = np.linalg.norm(kp[0, 0, 5] - kp[0, 0, 6])
    after = np.linalg.norm(out[0, 0, 5] - out[0, 0, 6])
    assert abs(before - after) < 1e-4


def test_scale_jitter_changes_scale_but_not_shape():
    kp = np.random.default_rng(10).normal(size=(3, 2, 17, 2)).astype("float32")
    out = apply_geometric(kp, np.random.default_rng(11), _only(scale_jitter=0.5))
    ratios = np.linalg.norm(out.reshape(-1, 2), axis=1) / (
        np.linalg.norm(kp.reshape(-1, 2), axis=1) + 1e-9
    )
    assert np.allclose(ratios, ratios[0], atol=1e-4)  # one uniform scale factor


def test_extent_is_the_skeleton_size_not_the_frame_position():
    # Regression: the extent once mixed axes (max of all coords minus min of all), so
    # an 80px person sitting at x~1500, y~300 measured ~1200px and got 15x the
    # intended coordinate noise. Size must not depend on where in frame they stand.
    from src.datasets.augment import _extent

    rng = np.random.default_rng(12)
    person = rng.uniform(0, 80, size=(10, 1, 17, 2)).astype("float32")
    scores = np.ones((10, 1, 17), dtype="float32")
    near_origin = _extent(person, scores)
    off_diagonal = person + np.array([1500.0, 300.0], dtype="float32")
    assert abs(_extent(off_diagonal, scores) - near_origin) < 1e-3
    assert near_origin <= 80.0


def test_from_dict_disables_on_empty_and_ignores_unknown_keys():
    assert AugmentConfig.from_dict(None) is None
    assert AugmentConfig.from_dict({}) is None
    cfg = AugmentConfig.from_dict({"joint_dropout": 0.4, "nonsense": 1})
    assert cfg.joint_dropout == 0.4
