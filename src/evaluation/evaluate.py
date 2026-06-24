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


def default_checkpoint(checkpoint_dir: str) -> str | None:
    """Prefer a *best* checkpoint (best validation accuracy) over the newest epoch."""
    best = glob.glob(os.path.join(checkpoint_dir, "*best*.pt"))
    if best:
        return max(best, key=os.path.getmtime)
    return latest_checkpoint(checkpoint_dir)


def evaluate(
    config_path: str = "configs/baseline.yaml",
    checkpoint: str | None = None,
    split: str = "test",
    device: str = "auto",
) -> dict | None:
    """Run inference and return {'accuracy', 'confusion_matrix'} (or None if nothing to do).

    ``split`` is one of "test" (default, held-out), "val", "train", or "all".
    ``device`` is "auto" (cuda if available else cpu), "cpu", or "cuda".
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    import torch
    from torch.utils.data import DataLoader, Subset

    from src.datasets.unified_loader import UnifiedSkeletonDataset, split_indices
    from src.models.stgcn import STGCNBaseline

    pose_cache = config["data"]["pose_cache"]
    num_classes = config["model"]["num_classes"]

    dataset = UnifiedSkeletonDataset(
        pose_cache, config["data"]["num_frames"], config["data"].get("normalize", False)
    )
    if len(dataset) == 0:
        print(f"[warning] No .npz files in {pose_cache}; nothing to evaluate.")
        return None

    if split == "all":
        subset = dataset
    else:
        seed = config["training"].get("seed", 42)
        val_frac = config["data"].get("val_frac", 0.15)
        test_frac = config["data"].get("test_frac", 0.15)
        train_idx, val_idx, test_idx = split_indices(len(dataset), seed, val_frac, test_frac)
        subset = Subset(dataset, {"train": train_idx, "val": val_idx, "test": test_idx}[split])

    # Checkpoints are namespaced per experiment (set by train.py).
    experiment = config.get("experiment", "default")
    checkpoint = checkpoint or default_checkpoint(os.path.join("outputs/checkpoints", experiment))
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
    model = STGCNBaseline(
        in_channels=config["model"]["in_channels"],
        num_classes=num_classes,
        num_persons=config["data"]["max_persons"],
        graph_strategy="spatial",
    ).to(torch_device)
    state = torch.load(checkpoint, map_location=torch_device, weights_only=False)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()
    device = torch_device
    print(f"Loaded checkpoint: {checkpoint}  (device={device})")
    print(f"Evaluating on '{split}' split: n={len(subset)}")

    loader = DataLoader(subset, batch_size=config["training"]["batch_size"], shuffle=False)
    preds, targets = [], []
    with torch.no_grad():
        for tensors, labels in loader:
            outputs = model(tensors.to(device))
            preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            targets.extend(labels.numpy().tolist())

    cm = confusion_matrix(np.array(preds), np.array(targets), num_classes)
    acc = accuracy(np.array(preds), np.array(targets))
    print(format_report(cm, acc, config["model"].get("class_names")))
    return {"accuracy": acc, "confusion_matrix": cm}


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
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.split, args.device)


if __name__ == "__main__":
    main()
