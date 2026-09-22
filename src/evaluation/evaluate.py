"""Evaluate a trained ST-GCN baseline checkpoint on the unified pose cache.

Loads the same config the training run used, restores a checkpoint produced by
src/training/train.py, and reports accuracy plus a per-class confusion matrix
and precision/recall — the missing half of the Phase 1 baseline.

By default it evaluates the **held-out test split** (the same seeded split
train.py reserves and never trains on) and uses the **best-validation**
checkpoint, so the reported number is an honest generalization estimate rather
than training-set memorization. Pass --split all to score the whole cache.

Metrics are computed with plain numpy so this module stays dependency-light
and unit-testable without a GPU.

Usage:
    python -m src.evaluation.evaluate --config configs/bullying10k.yaml          # test split, best ckpt
    python -m src.evaluation.evaluate --config configs/bullying10k.yaml --split all
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

# Class index -> name, derived from the loader's keyword map (UT-Interaction).
from src.datasets.unified_loader import CLASS_KEYWORDS

IDX_TO_CLASS = {idx: name for name, idx in CLASS_KEYWORDS.items()}


def confusion_matrix(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(np.asarray(targets), np.asarray(preds)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    targets = np.asarray(targets)
    if len(targets) == 0:
        return 0.0
    return float((np.asarray(preds) == targets).mean())


def per_class_precision_recall(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precision and recall per class from a confusion matrix (0 where undefined)."""
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    actual = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    return precision, recall


def format_report(cm: np.ndarray, acc: float, class_names: list[str] | None = None) -> str:
    precision, recall = per_class_precision_recall(cm)
    lines = [f"Accuracy: {acc * 100:.2f}%  (n={int(cm.sum())})", "", "Per-class:"]
    for idx in range(cm.shape[0]):
        if class_names and idx < len(class_names):
            name = class_names[idx]
        else:
            name = IDX_TO_CLASS.get(idx, f"class_{idx}")
        lines.append(
            f"  {idx} {name:<10} support={int(cm[idx].sum()):<4} "
            f"precision={precision[idx]:.2f} recall={recall[idx]:.2f}"
        )
    lines += ["", "Confusion matrix (rows=true, cols=pred):", str(cm)]
    return "\n".join(lines)


def latest_checkpoint(checkpoint_dir: str) -> str | None:
    files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    return max(files, key=os.path.getmtime) if files else None


def default_checkpoint(checkpoint_dir: str, preferred: str | None = None) -> str | None:
    """Prefer an exact per-(model, stream) name, then the same architecture, then any.

    The exact-name lookup matters once four streams train into one experiment
    directory: "newest *best*.pt" would otherwise hand back whichever stream
    finished last. The architecture-scoped fallback matters because handing an
    ``agcn`` config a leftover ``stgcn`` checkpoint fails deep inside
    ``load_state_dict`` with an unreadable shape error.
    """
    if preferred:
        exact = os.path.join(checkpoint_dir, preferred)
        if os.path.exists(exact):
            return exact
        architecture = preferred.split("_")[0]
        same_arch = glob.glob(os.path.join(checkpoint_dir, f"{architecture}*best*.pt"))
        if same_arch:
            return max(same_arch, key=os.path.getmtime)
    best = glob.glob(os.path.join(checkpoint_dir, "*best*.pt"))
    if best:
        return max(best, key=os.path.getmtime)
    return latest_checkpoint(checkpoint_dir)


def collect_logits(model, loader, device):
    """Run inference and return (logits, targets) as numpy arrays.

    Returns logits rather than predictions so calibration and conformal abstention
    can work from the raw scores; argmax is recoverable, the reverse is not.
    """
    import torch

    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for tensors, labels in loader:
            outputs = model(tensors.to(device))
            logits.append(outputs.cpu().numpy())
            targets.append(labels.numpy())
    if not logits:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)
    return np.concatenate(logits), np.concatenate(targets)


def evaluate(
    config_path: str = "configs/baseline.yaml",
    checkpoint: str | None = None,
    split: str = "test",
    device: str = "auto",
    stream: str = "joint",
    alpha: float = 0.1,
) -> dict | None:
    """Run inference and return {'accuracy', 'confusion_matrix', ...} (None if nothing to do).

    ``split`` is one of "test" (default, held-out), "val", "train", or "all".
    ``device`` is "auto" (cuda if available else cpu), "cpu", or "cuda".

    Besides accuracy, this fits a temperature on the **validation** split and reports
    ECE before/after plus a conformal abstention summary at level ``alpha``. Those
    three numbers are what tell you whether a confident-looking score means anything.
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    import torch
    from torch.utils.data import DataLoader, Subset

    from src.datasets.unified_loader import UnifiedSkeletonDataset, split_indices
    from src.evaluation.calibrate import (
        abstention_report,
        conformal_threshold,
        expected_calibration_error,
        fit_temperature,
        format_calibration,
        softmax,
    )
    from src.models.factory import checkpoint_name, load_for_inference

    pose_cache = config["data"]["pose_cache"]
    num_classes = config["model"]["num_classes"]

    dataset = UnifiedSkeletonDataset(
        pose_cache,
        config["data"]["num_frames"],
        config["data"].get("normalize", False),
        config["data"]["max_persons"],
        stream=stream,
    )
    if len(dataset) == 0:
        print(f"[warning] No .npz files in {pose_cache}; nothing to evaluate.")
        return None

    seed = config["training"].get("seed", 42)
    val_frac = config["data"].get("val_frac", 0.15)
    test_frac = config["data"].get("test_frac", 0.15)
    train_idx, val_idx, test_idx = split_indices(len(dataset), seed, val_frac, test_frac)
    if split == "all":
        subset = dataset
    else:
        subset = Subset(dataset, {"train": train_idx, "val": val_idx, "test": test_idx}[split])

    # Checkpoints are namespaced per experiment (set by train.py).
    experiment = config.get("experiment", "default")
    checkpoint = checkpoint or default_checkpoint(
        os.path.join("outputs/checkpoints", experiment), checkpoint_name(config, stream)
    )
    if checkpoint is None or not os.path.exists(checkpoint):
        print(
            f"[error] No checkpoint for experiment '{experiment}'. "
            "Train this config first (python -m src.training.train --config ...)."
        )
        return None

    torch_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    model, normalize = load_for_inference(checkpoint, config, torch_device, num_classes)
    # Every Subset below references this one dataset, so this re-points them all at
    # the normalization the checkpoint was trained with.
    dataset.normalize = normalize
    device = torch_device
    print(f"Loaded checkpoint: {checkpoint}  (device={device}, stream={stream})")
    print(f"Evaluating on '{split}' split: n={len(subset)}")

    batch_size = config["training"]["batch_size"]
    logits, targets = collect_logits(model, DataLoader(subset, batch_size=batch_size), device)

    preds = logits.argmax(axis=1)
    cm = confusion_matrix(preds, targets, num_classes)
    acc = accuracy(preds, targets)
    print(format_report(cm, acc, config["model"].get("class_names")))

    result = {"accuracy": acc, "confusion_matrix": cm, "logits": logits, "targets": targets}

    # Calibration is fitted on validation and *applied* to whatever split we scored,
    # so the temperature never sees the data it is judged on.
    if split != "val" and len(val_idx) > 0:
        val_logits, val_targets = collect_logits(
            model, DataLoader(Subset(dataset, val_idx), batch_size=batch_size), device
        )
        temperature = fit_temperature(val_logits, val_targets)
        before = expected_calibration_error(softmax(logits), targets)
        after = expected_calibration_error(softmax(logits, temperature), targets)
        threshold = conformal_threshold(softmax(val_logits, temperature), val_targets, alpha)
        report = abstention_report(softmax(logits, temperature), targets, threshold)
        print()
        print(format_calibration(temperature, before, after, report))
        print(f"  conformal q      {threshold:.4f}  (alpha={alpha})")
        result.update(
            temperature=temperature, ece=after, conformal_threshold=threshold, abstention=report
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="defaults to the best (then newest) in outputs/checkpoints",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("test", "val", "train", "all"),
        help="which split to score",
    )
    parser.add_argument("--device", default="auto", help='"auto", "cpu", or "cuda"')
    parser.add_argument(
        "--stream", default="joint", choices=("joint", "bone", "joint_motion", "bone_motion")
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="conformal miscoverage level (default 0.1)"
    )
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.split, args.device, args.stream, args.alpha)


if __name__ == "__main__":
    main()
