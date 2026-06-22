import numpy as np

from src.datasets.taxonomy import (
    AGGRESSIVE,
    NEUTRAL,
    binary_label,
    ut_interaction_aggressive,
)


def test_ut_interaction_aggressive():
    assert ut_interaction_aggressive("seq_kick_01") == AGGRESSIVE
    assert ut_interaction_aggressive("seq_punch_02") == AGGRESSIVE
    assert ut_interaction_aggressive("seq_push_03") == AGGRESSIVE
    assert ut_interaction_aggressive("seq_handshake_04") == NEUTRAL
    assert ut_interaction_aggressive("seq_hug_05") == NEUTRAL
    assert ut_interaction_aggressive("seq_point_06") == NEUTRAL
    assert ut_interaction_aggressive("mystery") is None


def test_binary_label_prefers_stored_flag():
    assert binary_label({"aggressive": np.array(True)}, "x", "bullying10k") == AGGRESSIVE
    assert binary_label({"aggressive": np.array(False)}, "x", "ntu") == NEUTRAL


def test_binary_label_falls_back_to_ut_filename():
    assert binary_label({}, "clip_kick_1.npz", "ut_interaction") == AGGRESSIVE
    assert binary_label({}, "clip_hug_1.npz", "ut_interaction") == NEUTRAL


def test_binary_label_unknown_returns_none():
    assert binary_label({}, "no_class_here.npz", "ut_interaction") is None
