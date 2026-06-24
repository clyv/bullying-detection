import numpy as np

from src.datasets.taxonomy import (
    AGGRESSIVE,
    NEUTRAL,
    binary_label,
    ut_interaction_aggressive,
)


def test_ut_interaction_aggressive_keywords():
    assert ut_interaction_aggressive("seq_kick_01") == AGGRESSIVE
    assert ut_interaction_aggressive("seq_punch_02") == AGGRESSIVE
    assert ut_interaction_aggressive("seq_push_03") == AGGRESSIVE
    assert ut_interaction_aggressive("seq_handshake_04") == NEUTRAL
    assert ut_interaction_aggressive("seq_hug_05") == NEUTRAL
    assert ut_interaction_aggressive("mystery") is None


def test_ut_interaction_aggressive_numeric_filenames():
    # real UT naming "<seq>_<pair>_<class>" with trailing class id 0-5
    assert ut_interaction_aggressive("0_11_2.npz") == AGGRESSIVE  # kick
    assert ut_interaction_aggressive("0_11_4") == AGGRESSIVE  # punch
    assert ut_interaction_aggressive("3_7_5") == AGGRESSIVE  # push
    assert ut_interaction_aggressive("0_11_0.npz") == NEUTRAL  # handshake
    assert ut_interaction_aggressive("1_2_3") == NEUTRAL  # point
    assert ut_interaction_aggressive("seq_99") is None  # class id 99 unknown


def test_binary_label_prefers_stored_flag():
    assert binary_label({"aggressive": np.array(True)}, "x", "bullying10k") == AGGRESSIVE
    assert binary_label({"aggressive": np.array(False)}, "x", "ntu") == NEUTRAL


def test_binary_label_falls_back_to_ut_filename():
    assert binary_label({}, "clip_kick_1.npz", "ut_interaction") == AGGRESSIVE
    assert binary_label({}, "clip_hug_1.npz", "ut_interaction") == NEUTRAL


def test_binary_label_unknown_returns_none():
    assert binary_label({}, "no_class_here.npz", "ut_interaction") is None
