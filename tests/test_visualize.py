import numpy as np

from src.evaluation.localize import find_incidents
from src.evaluation.visualize import (
    add_forced_incidents,
    crowd_pressure_scores,
    earliest_incident_before,
    frame_scores_from_windows,
    keep_top_incident_scores,
    normalize_signal,
    output_stem,
)


def test_frame_scores_single_window_broadcasts():
    fs = frame_scores_from_windows([0.8], starts=[0], window=4, num_frames=4)
    assert fs.shape == (4,)
    assert np.allclose(fs, 0.8)


def test_frame_scores_averages_overlapping_windows():
    # windows [0,4) score 1.0 and [2,6) score 0.0 overlap on frames 2,3
    fs = frame_scores_from_windows([1.0, 0.0], starts=[0, 2], window=4, num_frames=6)
    assert np.allclose(fs[:2], 1.0)  # only first window
    assert np.allclose(fs[2:4], 0.5)  # averaged overlap
    assert np.allclose(fs[4:], 0.0)  # only second window


def test_frame_scores_uncovered_frames_are_zero():
    fs = frame_scores_from_windows([1.0], starts=[0], window=2, num_frames=5)
    assert np.allclose(fs[:2], 1.0)
    assert np.allclose(fs[2:], 0.0)  # no window covers these -> 0, not div-by-zero


def test_per_frame_incidents_via_find_incidents():
    # a per-frame curve thresholded into incidents (window=1 trick)
    fs = np.array([0.1, 0.9, 0.95, 0.2, 0.8, 0.85])
    incidents = find_incidents(fs, range(len(fs)), window=1, threshold=0.5, max_gap=0)
    assert incidents == [(1, 3, 0.95), (4, 6, 0.85)]


def test_crowd_pressure_prefers_compact_clusters():
    keypoints = np.zeros((2, 4, 17, 2), dtype="float32")
    scores = np.ones((2, 4, 17), dtype="float32")
    keypoints[:, :, :, 1] += np.linspace(0, 40, 17)
    keypoints[0, 0, :, :] += [10, 10]
    keypoints[0, 1, :, :] += [200, 10]
    keypoints[0, 2, :, :] += [350, 10]
    keypoints[0, 3, :, :] += [450, 10]
    keypoints[1, 0, :, :] += [100, 100]
    keypoints[1, 1, :, :] += [115, 106]
    keypoints[1, 2, :, :] += [128, 112]
    keypoints[1, 3, :, :] += [260, 100]

    pressure = crowd_pressure_scores(keypoints, scores)

    assert pressure[1] > pressure[0]


def test_keep_top_incident_scores_suppresses_weaker_regions():
    frame_scores = np.array([0.8, 0.8, 0.1, 0.9, 0.9, 0.9, 0.1, 0.7], dtype="float32")
    incidents = [(0, 2, 0.8), (3, 6, 0.9), (7, 8, 0.7)]

    kept_scores, kept_incidents = keep_top_incident_scores(
        frame_scores, incidents, threshold=0.5, top_n=1
    )

    assert kept_incidents == [(3, 6, 0.9)]
    assert kept_scores[4] == frame_scores[4]
    assert kept_scores[0] < 0.5


def test_output_stem_supports_unique_run_names():
    assert output_stem("some/path/clip.mp4", run_name="trial_a") == "clip_trial_a"
    assert output_stem("some/path/clip.mp4", unique=False) == "clip"
    assert output_stem("some/path/clip.mp4").startswith("clip_20")


def test_normalize_signal_scales_nonconstant_values():
    values = normalize_signal(np.array([2.0, 4.0, 6.0]))

    assert values.tolist() == [0.0, 0.5, 1.0]


def test_earliest_incident_before_returns_padded_first_match():
    incidents = [(50, 60, 0.7), (100, 120, 0.9)]

    assert earliest_incident_before(incidents, before_frame=90, pad=10, num_frames=200) == [
        (40, 70, 0.7)
    ]


def test_add_forced_incidents_boosts_scores_above_threshold():
    frame_scores = np.zeros(10, dtype="float32")
    boosted, incidents = add_forced_incidents(frame_scores, [], [(2, 5, 0.2)], threshold=0.6)

    assert incidents == [(2, 5, 0.61)]
    assert boosted[3] > 0.6


def test_motion_energy_moving_person_scores_higher():
    from src.evaluation.visualize import motion_energy_scores

    T, M = 10, 2
    kp = np.zeros((T, M, 17, 2), dtype="float32")
    sc = np.ones((T, M, 17), dtype="float32")
    kp[:, :, :, 1] = np.linspace(0, 50, 17)  # 50px tall skeletons
    kp[:, 1, :, 0] = np.arange(T)[:, None] * 5.0  # person 1 moves 5px/frame

    frame_e, per_person = motion_energy_scores(kp, sc)
    assert per_person[5, 1] > per_person[5, 0]  # mover beats the statue
    assert np.isclose(frame_e[5], per_person[5, 1])  # frame takes the max


def test_motion_energy_is_height_normalized():
    from src.evaluation.visualize import motion_energy_scores

    T = 6
    kp_small = np.zeros((T, 1, 17, 2), dtype="float32")
    kp_small[:, 0, :, 1] = np.linspace(0, 50, 17)
    kp_small[:, 0, :, 0] = np.arange(T)[:, None] * 5.0  # 5px/frame at 50px tall
    kp_big = kp_small * 4.0  # same relative motion at 200px tall
    sc = np.ones((T, 1, 17), dtype="float32")

    _, small = motion_energy_scores(kp_small, sc)
    _, big = motion_energy_scores(kp_big, sc)
    assert np.allclose(small[1:], big[1:], atol=1e-5)
