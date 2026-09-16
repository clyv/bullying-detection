import numpy as np

from src.evaluation.calibrate import (
    abstention_report,
    conformal_threshold,
    expected_calibration_error,
    fit_temperature,
    prediction_sets,
    softmax,
)


def test_softmax_rows_sum_to_one_and_survive_large_logits():
    logits = np.array([[1000.0, -1000.0], [0.0, 0.0]])
    probs = softmax(logits)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert not np.isnan(probs).any()
    assert np.allclose(probs[1], [0.5, 0.5])


def test_temperature_flattens_saturated_logits():
    # The failure this module exists for: hugely separated logits (P ~= 1.000) on a
    # model that is only ~70% right. The fitted temperature must be well above 1.
    rng = np.random.default_rng(0)
    targets = rng.integers(0, 2, size=400)
    logits = np.zeros((400, 2))
    correct = rng.random(400) < 0.7
    for i, (t, ok) in enumerate(zip(targets, correct)):
        winner = t if ok else 1 - t
        logits[i, winner] = 12.0
        logits[i, 1 - winner] = -12.0
    temperature = fit_temperature(logits, targets)
    assert temperature > 2.0
    # And it must actually improve calibration.
    before = expected_calibration_error(softmax(logits), targets)
    after = expected_calibration_error(softmax(logits, temperature), targets)
    assert after < before


def test_temperature_falls_back_to_identity_on_degenerate_input():
    # Each of these makes NLL minimal as T -> 0 or infinity, so an unguarded LBFGS
    # returns inf/NaN — which then poisons every probability downstream.
    rng = np.random.default_rng(4)
    tiny_logits = rng.normal(size=(4, 2))
    assert fit_temperature(tiny_logits, np.array([0, 1, 0, 1])) == 1.0  # too few samples

    single_class = rng.normal(size=(40, 2))
    assert fit_temperature(single_class, np.zeros(40, dtype=int)) == 1.0  # one class only

    # Perfectly separable and far from the boundary: T wants to run to zero.
    separable = np.array([[30.0, -30.0]] * 20 + [[-30.0, 30.0]] * 20)
    targets = np.array([0] * 20 + [1] * 20)
    temperature = fit_temperature(separable, targets)
    assert np.isfinite(temperature) and temperature > 0


def test_temperature_is_always_finite_and_bounded():
    rng = np.random.default_rng(5)
    for scale in (0.001, 1.0, 100.0, 10000.0):
        targets = rng.integers(0, 2, size=64)
        logits = rng.normal(size=(64, 2)) * scale
        temperature = fit_temperature(logits, targets)
        assert np.isfinite(temperature)
        assert 0.05 <= temperature <= 10.0


def test_temperature_scaling_never_changes_predictions():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(50, 2)) * 5
    assert np.array_equal(softmax(logits).argmax(axis=1), softmax(logits, 3.7).argmax(axis=1))


def test_ece_is_zero_for_a_perfectly_calibrated_confident_model():
    targets = np.array([0, 1, 0, 1])
    logits = np.array([[50.0, -50.0], [-50.0, 50.0], [50.0, -50.0], [-50.0, 50.0]])
    assert expected_calibration_error(softmax(logits), targets) < 1e-6


def test_ece_is_large_for_a_confidently_wrong_model():
    targets = np.array([0, 0, 0, 0])
    logits = np.array([[-50.0, 50.0]] * 4)  # 100% confident, 100% wrong
    assert expected_calibration_error(softmax(logits), targets) > 0.9


def test_ece_handles_empty_input():
    assert expected_calibration_error(np.zeros((0, 2)), np.zeros(0)) == 0.0


def test_conformal_threshold_achieves_requested_coverage():
    rng = np.random.default_rng(2)
    n = 2000
    targets = rng.integers(0, 2, size=n)
    # A noisy but honest model: true-class probability drawn around 0.75.
    true_p = np.clip(rng.normal(0.75, 0.15, size=n), 0.01, 0.99)
    probs = np.zeros((n, 2))
    probs[np.arange(n), targets] = true_p
    probs[np.arange(n), 1 - targets] = 1 - true_p

    cal, test = slice(0, 1000), slice(1000, n)
    q = conformal_threshold(probs[cal], targets[cal], alpha=0.1)
    report = abstention_report(probs[test], targets[test], q)
    # Split-conformal guarantees >= 1 - alpha coverage; allow finite-sample slack.
    assert report["coverage"] >= 0.87


def test_lower_alpha_means_lower_threshold_and_more_coverage():
    rng = np.random.default_rng(3)
    targets = rng.integers(0, 2, size=500)
    true_p = np.clip(rng.normal(0.7, 0.2, size=500), 0.01, 0.99)
    probs = np.zeros((500, 2))
    probs[np.arange(500), targets] = true_p
    probs[np.arange(500), 1 - targets] = 1 - true_p
    strict = conformal_threshold(probs, targets, alpha=0.01)
    loose = conformal_threshold(probs, targets, alpha=0.2)
    assert strict <= loose


def test_prediction_sets_admit_both_classes_when_unsure():
    probs = np.array([[0.5, 0.5], [0.95, 0.05]])
    sets = prediction_sets(probs, threshold=0.4)
    assert sets[0].sum() == 2  # abstain: both classes admitted
    assert sets[1].sum() == 1  # decide


def test_abstention_report_counts_uncertain_windows_as_abstentions():
    probs = np.array([[0.5, 0.5], [0.99, 0.01], [0.98, 0.02]])
    targets = np.array([0, 0, 0])
    report = abstention_report(probs, targets, threshold=0.4)
    assert report["n"] == 3
    assert abs(report["abstention_rate"] - 1 / 3) < 1e-9
    assert report["selective_accuracy"] == 1.0  # right on both it answered


def test_abstention_report_handles_empty_input():
    report = abstention_report(np.zeros((0, 2)), np.zeros(0), 0.5)
    assert report["n"] == 0
