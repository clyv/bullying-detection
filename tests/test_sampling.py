import numpy as np
import torch

from src.evaluation.cross_dataset import per_dataset_report
from src.training.sampling import (
    corpus_shares,
    dataset_balance_weights,
    effective_class_counts,
    make_sampler,
)

# The real pool's proportions, scaled down: two big lab corpora, two small CCTV ones.
SOURCES = ["ntu"] * 170 + ["b10k"] * 100 + ["ubi"] * 17 + ["surv"] * 3


def test_power_zero_is_the_natural_mix():
    weights = dataset_balance_weights(SOURCES, 0.0)
    assert np.allclose(weights, 1 / len(SOURCES))


def test_power_one_gives_every_corpus_an_equal_share():
    shares = corpus_shares(SOURCES, dataset_balance_weights(SOURCES, 1.0))
    assert all(abs(s - 0.25) < 1e-9 for s in shares.values())


def test_power_half_gives_sqrt_proportional_shares():
    shares = corpus_shares(SOURCES, dataset_balance_weights(SOURCES, 0.5))
    counts = {"ntu": 170, "b10k": 100, "ubi": 17, "surv": 3}
    root_total = sum(np.sqrt(c) for c in counts.values())
    for name, count in counts.items():
        assert abs(shares[name] - np.sqrt(count) / root_total) < 1e-9
    # And the small CCTV corpora actually gain ground over the natural mix.
    assert shares["ubi"] + shares["surv"] > (17 + 3) / len(SOURCES)


def test_weights_are_a_distribution():
    for power in (0.0, 0.5, 1.0):
        weights = dataset_balance_weights(SOURCES, power)
        assert abs(weights.sum() - 1.0) < 1e-9 and (weights > 0).all()


def test_effective_class_counts_follow_the_sampling_weights():
    # Corpus A is all neutral, corpus B all aggressive, B is 9x smaller. Balanced
    # sampling makes the model *see* them 50/50, and the loss weights must say so.
    sources = ["a"] * 90 + ["b"] * 10
    labels = [0] * 90 + [1] * 10
    natural = effective_class_counts(labels, dataset_balance_weights(sources, 0.0))
    balanced = effective_class_counts(labels, dataset_balance_weights(sources, 1.0))
    assert np.allclose(natural, [90, 10])
    assert np.allclose(balanced, [50, 50])


def test_sampler_draws_one_epoch_in_proportion():
    weights = dataset_balance_weights(SOURCES, 1.0)
    drawn = list(make_sampler(weights, num_samples=20000, seed=0))
    assert len(drawn) == 20000
    picked = np.array(SOURCES)[drawn]
    surv_share = (picked == "surv").mean()
    assert abs(surv_share - 0.25) < 0.02  # 3 of 290 clips, yet a quarter of the draws


def test_sampler_is_reproducible_with_a_seed():
    weights = dataset_balance_weights(SOURCES, 0.5)
    first = list(make_sampler(weights, seed=7))
    second = list(make_sampler(weights, seed=7))
    assert first == second


def test_per_dataset_report_breaks_out_every_source():
    preds = np.array([1, 1, 0, 0, 1, 0])
    targets = np.array([1, 1, 0, 1, 0, 0])
    sources = ["b10k", "b10k", "b10k", "ubi", "ubi", "ubi"]
    report = per_dataset_report(preds, targets, sources)
    assert "b10k" in report and "ubi" in report
    assert "100.0%" in report  # b10k: all three right
    assert "33.3%" in report  # ubi: one of three right


def test_per_dataset_report_handles_a_single_class_source():
    report = per_dataset_report(np.array([0, 0]), np.array([0, 0]), ["x", "x"])
    assert "-" in report  # no aggressive clips: recall undefined, not a crash


def test_pool_energies_caches_and_invalidates_on_change(tmp_path):
    from types import SimpleNamespace

    from src.evaluation.cross_dataset import pool_energies

    clip = tmp_path / "clip.npz"
    rng = np.random.default_rng(0)
    kp = rng.uniform(0, 100, (10, 2, 17, 2)).astype("float32")
    np.savez(clip, keypoints=kp, scores=np.ones((10, 2, 17), "float32"))
    ds = SimpleNamespace(samples=[(str(clip), "x", 0)])
    cache = tmp_path / "cache.json"

    first = pool_energies(ds, str(cache))
    assert cache.exists()
    assert pool_energies(ds, str(cache)) == first  # served from cache

    # A re-extracted clip (new mtime) must be recomputed, not served stale.
    import os
    import time

    np.savez(clip, keypoints=kp * 3.0 + 50.0, scores=np.ones((10, 2, 17), "float32"))
    later = time.time() + 5
    os.utime(clip, (later, later))
    pool_energies(ds, str(cache))
    # Energy is scale-invariant, so the value can't tell a recompute from a stale hit;
    # a second cache entry (new mtime key) can.
    import json

    assert len(json.loads(cache.read_text())) == 2


def test_fit_returns_the_best_validation_weights(tmp_path):
    # The printed test number must describe the checkpoint on disk.
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.agcn import AGCN
    from src.training.train import fit

    torch.manual_seed(0)
    data = TensorDataset(torch.randn(8, 3, 16, 17, 2), torch.tensor([0, 1] * 4))
    loader = DataLoader(data, batch_size=4)
    best = tmp_path / "best.pt"
    model = fit(
        AGCN(3, 2, base_channels=8),
        loader,
        loader,
        epochs=3,
        lr=1e-2,
        weight_decay=0.0,
        device=torch.device("cpu"),
        best_path=str(best),
    )
    saved = torch.load(best, weights_only=False)["model_state_dict"]
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, saved[name]), name
