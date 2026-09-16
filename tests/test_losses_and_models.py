import numpy as np
import torch

from src.evaluation.ensemble import fuse
from src.models.agcn import AGCN, AdaptiveGraphConv, MultiScaleTCN
from src.models.factory import build_model, checkpoint_name
from src.training.losses import (
    FocalLoss,
    class_weights_from_counts,
    mixup_batch,
    soft_target_cross_entropy,
)


def _config(name="agcn", num_classes=2):
    return {
        "model": {"name": name, "num_classes": num_classes, "in_channels": 3, "dropout": 0.1},
        "data": {"max_persons": 2},
    }


# --- losses -----------------------------------------------------------------


def test_focal_with_gamma_zero_equals_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(8, 2)
    targets = torch.randint(0, 2, (8,))
    focal = FocalLoss(gamma=0.0)(logits, targets)
    ce = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.allclose(focal, ce, atol=1e-6)


def test_focal_downweights_easy_examples():
    # One easy example (right, p~0.98) and one hard (wrong, p~0.02). Kept off the
    # saturation rail so the ratio below is well-defined rather than 0/0.
    logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
    targets = torch.tensor([0, 0])
    per_sample_ce = FocalLoss(gamma=0.0, reduction="none")(logits, targets)
    per_sample_focal = FocalLoss(gamma=2.0, reduction="none")(logits, targets)
    easy_shrink = float(per_sample_focal[0] / per_sample_ce[0])
    hard_shrink = float(per_sample_focal[1] / per_sample_ce[1])
    assert easy_shrink < 0.01  # the easy one is suppressed almost entirely
    assert hard_shrink > 0.9  # the hard one is essentially untouched
    assert easy_shrink < hard_shrink


def test_focal_loss_is_finite_on_saturated_logits():
    logits = torch.tensor([[200.0, -200.0], [-200.0, 200.0]])
    targets = torch.tensor([0, 1])
    assert torch.isfinite(FocalLoss(gamma=2.0)(logits, targets))


def test_class_weights_are_inverse_frequency_and_mean_one():
    weights = class_weights_from_counts([900, 100])
    assert weights[1] > weights[0]  # rarer class weighted higher
    assert abs(float(weights.mean()) - 1.0) < 1e-5


def test_class_weights_cap_protects_against_a_near_empty_class():
    weights = class_weights_from_counts([100000, 1], cap=10.0)
    assert float(weights.max()) <= 10.0 / float(weights.mean()) + 1e-4
    assert torch.isfinite(weights).all()


def test_class_weights_survive_a_zero_count():
    assert torch.isfinite(class_weights_from_counts([50, 0])).all()


def test_mixup_returns_soft_targets_that_stay_a_distribution():
    torch.manual_seed(0)
    tensors = torch.randn(6, 3, 16, 17, 2)
    targets = torch.randint(0, 2, (6,))
    mixed, soft = mixup_batch(tensors, targets, num_classes=2, alpha=0.4)
    assert mixed.shape == tensors.shape
    assert torch.allclose(soft.sum(dim=1), torch.ones(6), atol=1e-5)
    assert (soft >= 0).all()


def test_mixup_with_alpha_zero_is_a_passthrough():
    tensors = torch.randn(4, 3, 8, 17, 2)
    targets = torch.tensor([0, 1, 1, 0])
    mixed, soft = mixup_batch(tensors, targets, num_classes=2, alpha=0.0)
    assert torch.equal(mixed, tensors)
    assert torch.equal(soft.argmax(dim=1), targets)


def test_soft_target_cross_entropy_matches_hard_ce_on_one_hot():
    torch.manual_seed(1)
    logits = torch.randn(5, 3)
    targets = torch.randint(0, 3, (5,))
    one_hot = torch.nn.functional.one_hot(targets, 3).float()
    assert torch.allclose(
        soft_target_cross_entropy(logits, one_hot),
        torch.nn.functional.cross_entropy(logits, targets),
        atol=1e-6,
    )


# --- model ------------------------------------------------------------------


def test_agcn_forward_shape():
    model = AGCN(in_channels=3, num_classes=2)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 17, 2))
    assert out.shape == (2, 2)


def test_agcn_handles_odd_frame_counts_and_single_person():
    model = AGCN(in_channels=3, num_classes=2, num_persons=1)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 37, 17, 1))
    assert out.shape == (1, 2)


def test_agcn_is_trainable_end_to_end():
    model = AGCN(in_channels=3, num_classes=2)
    out = model(torch.randn(2, 3, 32, 17, 2))
    loss = FocalLoss(gamma=2.0)(out, torch.tensor([0, 1]))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)


def test_adaptive_term_starts_inert():
    # alpha is initialised at zero so training begins as a plain (fixed + learned)
    # graph conv and the data-dependent topology fades in rather than destabilising
    # the first few hundred steps.
    A = torch.rand(3, 17, 17)
    layer = AdaptiveGraphConv(3, 16, A)
    assert float(layer.alpha.detach().abs().sum()) == 0.0
    layer.eval()
    x = torch.randn(2, 3, 8, 17)
    with torch.no_grad():
        adaptive = layer(x)
        layer.adaptive = False
        static = layer(x)
    assert adaptive.shape == (2, 16, 8, 17)
    # With alpha == 0 the data-dependent branch contributes nothing, so switching
    # it off must not change the output at all.
    assert torch.allclose(adaptive, static, atol=1e-5)


def test_adaptive_graph_learns_edges_the_skeleton_lacks():
    # The whole point: B is free to grow connections the anatomy does not have —
    # e.g. one person's fist to the other's head.
    A = torch.zeros(3, 17, 17)
    layer = AdaptiveGraphConv(3, 8, A)
    assert layer.B.requires_grad
    assert not layer.A_fixed.requires_grad


def test_multiscale_tcn_branch_channels_sum_exactly():
    # 64 is not divisible by 4 branches when a remainder appears; the bottleneck
    # branch must absorb it or the concat silently changes the channel count.
    for channels in (16, 32, 64, 66, 130):
        tcn = MultiScaleTCN(channels, channels)
        out = tcn(torch.randn(1, channels, 16, 17))
        assert out.shape[1] == channels, channels


def test_multiscale_tcn_halves_time_on_stride_two():
    tcn = MultiScaleTCN(32, 32, stride=2)
    out = tcn(torch.randn(1, 32, 64, 17))
    assert out.shape[2] == 32


# --- factory ----------------------------------------------------------------


def test_factory_builds_both_architectures():
    for name in ("stgcn", "agcn"):
        model = build_model(_config(name))
        model.eval()
        with torch.no_grad():
            assert model(torch.randn(1, 3, 32, 17, 2)).shape == (1, 2)


def test_factory_rejects_unknown_model():
    try:
        build_model(_config("transformer"))
    except ValueError as err:
        assert "transformer" in str(err)
    else:
        raise AssertionError("expected ValueError for an unknown model name")


def test_checkpoint_names_are_unique_per_model_and_stream():
    names = {
        checkpoint_name(_config(model), stream)
        for model in ("stgcn", "agcn")
        for stream in ("joint", "bone", "joint_motion", "bone_motion")
    }
    assert len(names) == 8  # nothing overwrites anything else
    assert checkpoint_name(_config("stgcn"), "joint") == "stgcn_best.pt"  # back-compat


# --- ensemble ---------------------------------------------------------------


def test_fuse_averages_and_keeps_a_distribution():
    a = np.array([[0.9, 0.1], [0.2, 0.8]])
    b = np.array([[0.5, 0.5], [0.4, 0.6]])
    fused = fuse([a, b])
    assert np.allclose(fused, [[0.7, 0.3], [0.3, 0.7]])
    assert np.allclose(fused.sum(axis=1), 1.0)


def test_fuse_respects_weights():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    assert np.allclose(fuse([a, b], weights=[3, 1]), [[0.75, 0.25]])


def test_fuse_rejects_empty():
    try:
        fuse([])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when fusing nothing")
