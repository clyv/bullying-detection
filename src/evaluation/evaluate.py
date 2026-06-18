"""Evaluate a trained ST-GCN baseline checkpoint on the unified pose cache.

Loads the same configs/baseline.yaml the training run used, restores a
checkpoint produced by src/training/train.py, runs inference over the pose
cache, and reports overall accuracy plus a per-class confusion matrix and
precision/recall — the missing half of the Phase 1 baseline (train + evaluate).

Metrics are computed with plain numpy so this module stays dependency-light
and unit-testable without a GPU.

Usage:
    python -m src.evaluation.evaluate                     # newest checkpoint
    python -m src.evaluation.evaluate --checkpoint outputs/checkpoints/stgcn_baseline_epoch_80.pt
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


def evaluate(
    config_path: str = "configs/baseline.yaml", checkpoint: str | None = None
) -> dict | None:
    """Run inference and return {'accuracy', 'confusion_matrix'} (or None if nothing to do)."""
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    import torch
    from torch.utils.data import DataLoader

    from src.datasets.unified_loader import UnifiedSkeletonDataset
    from src.models.stgcn import STGCNBaseline

    pose_cache = config["data"]["pose_cache"]
    num_classes = config["model"]["num_classes"]

    dataset = UnifiedSkeletonDataset(pose_cache, target_frames=config["data"]["num_frames"])
    if len(dataset) == 0:
        print(f"[warning] No .npz files in {pose_cache}; nothing to evaluate.")
        return None

    checkpoint = checkpoint or latest_checkpoint("outputs/checkpoints")
    if checkpoint is None or not os.path.exists(checkpoint):
        print("[error] No checkpoint found. Train a model first (python -m src.training.train).")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STGCNBaseline(
        in_channels=config["model"]["in_channels"],
        num_classes=num_classes,
        num_persons=config["data"]["max_persons"],
        graph_strategy="spatial",
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}  (device={device})")

    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False)
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
        "--checkpoint", default=None, help="defaults to newest in outputs/checkpoints"
    )
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
