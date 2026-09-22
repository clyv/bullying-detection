"""Phase 4 — unified aggressive-vs-neutral model across all datasets.

Pools every configured corpus under the binary label space
(src/datasets/taxonomy.py) and runs the two Phase 4 analyses:

1. Pooled evaluation — train on a mix of all datasets, report the
   aggressive-vs-neutral confusion matrix on a held-out split.
2. Cross-dataset generalisation / per-dataset ablation — leave-one-dataset-out:
   train on the others, test on the held-out dataset.

**Read (2), not (1).** Pooled accuracy on a random split is an upper bound inflated
by corpus identity: when five corpora with distinct capture rigs are mixed and then
split at random, the cheapest route to a high score is to recognise which corpus a
clip came from and apply that corpus's class prior. Leave-one-dataset-out is the
number that estimates whether the model learned aggression.

Usage:
    python -m src.evaluation.cross_dataset --config configs/unified.yaml
    python -m src.evaluation.cross_dataset --config configs/unified.yaml --stream bone
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.datasets.taxonomy import BINARY_NAMES
from src.evaluation.evaluate import accuracy, confusion_matrix, format_report


def predict(model, loader, device):
    """Return (preds, targets) numpy arrays over a loader."""
    import torch

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for tensors, labels in loader:
            outputs = model(tensors.to(device))
            preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            targets.extend(labels.numpy().tolist())
    return np.array(preds), np.array(targets)


def evaluate_model(model, loader, device, num_classes=2):
    preds, targets = predict(model, loader, device)
    cm = confusion_matrix(preds, targets, num_classes)
    return accuracy(preds, targets), cm


def _train_binary(train_ds, val_ds, cfg, device, best_path=None, class_counts=None, stream="joint"):
    from torch.utils.data import DataLoader

    from src.models.factory import build_model, checkpoint_meta
    from src.training.train import build_criterion, fit

    model = build_model(cfg, num_classes=2).to(device)
    batch_size = cfg["training"]["batch_size"]
    fit(
        model,
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        device=device,
        best_path=best_path,
        criterion=build_criterion(cfg, class_counts, device),
        num_classes=2,
        mixup_alpha=cfg["training"].get("mixup_alpha", 0.0),
        clip_grad=cfg["training"].get("clip_grad", 1.0),
        warmup_epochs=cfg["training"].get("warmup_epochs", 0),
        checkpoint_meta=checkpoint_meta(cfg, stream),
    )
    return model


def specs_from_config(cfg):
    return [(d["name"], d["cache"]) for d in cfg["data"]["datasets"]]


def _counts(dataset, indices=None, num_classes=2):
    """Per-class counts over a dataset (or a subset of its indices)."""
    labels = np.array([s[2] for s in dataset.samples])
    if indices is not None:
        labels = labels[np.asarray(indices)]
    return np.bincount(labels, minlength=num_classes).tolist()


def _build_pool(cfg, specs, stream):
    """Construct the pooled dataset for ``specs`` with the configured options."""
    from src.datasets.unified_loader import MultiDatasetSkeletonDataset

    return MultiDatasetSkeletonDataset(
        specs,
        cfg["data"]["num_frames"],
        cfg["data"].get("normalize", False),
        cfg["data"]["max_persons"],
        stream=stream,
    )


def pooled_evaluation(cfg, device, stream="joint"):
    """Train on a seeded 70/15/15 split, evaluate on the held-out test slice."""
    from torch.utils.data import DataLoader, Subset

    from src.datasets.augment import AugmentConfig
    from src.datasets.unified_loader import split_indices
    from src.models.factory import checkpoint_name

    ds = _build_pool(cfg, specs_from_config(cfg), stream)
    if len(ds) < 3:
        print("[warning] pooled dataset has <3 samples; skipping pooled evaluation.")
        return None

    seed = cfg["training"].get("seed", 42)
    train_idx, val_idx, test_idx = split_indices(
        len(ds), seed, cfg["data"].get("val_frac", 0.15), cfg["data"].get("test_frac", 0.15)
    )
    augment = AugmentConfig.from_dict(cfg.get("augment"))
    train_source = ds.with_augment(augment) if augment else ds

    experiment = cfg.get("experiment", "phase4_unified")
    ckpt_dir = os.path.join("outputs/checkpoints", experiment)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, checkpoint_name(cfg, stream))

    model = _train_binary(
        Subset(train_source, train_idx),
        Subset(ds, val_idx),
        cfg,
        device,
        best_path=best_path,
        class_counts=_counts(ds, train_idx),
        stream=stream,
    )
    acc, cm = evaluate_model(
        model,
        DataLoader(Subset(ds, test_idx), batch_size=cfg["training"]["batch_size"]),
        device,
    )
    print("\n=== Pooled aggressive-vs-neutral evaluation (held-out test split) ===")
    print(format_report(cm, acc, BINARY_NAMES))
    print(f"[checkpoint] pooled model saved to {best_path}")
    print(
        "[note] This number is inflated by corpus identity — read the "
        "leave-one-dataset-out table below as the generalisation estimate."
    )
    return acc, cm


def leave_one_out(cfg, device, resume_index=0, ckpt_path=None, stream="joint"):
    """Train on every dataset but one, test on the one held out. The real scoreboard."""
    from torch.utils.data import DataLoader, Subset

    from src.datasets.augment import AugmentConfig
    from src.datasets.unified_loader import split_indices

    specs = specs_from_config(cfg)
    augment = AugmentConfig.from_dict(cfg.get("augment"))
    seed = cfg["training"].get("seed", 42)
    results = {}
    print("\n=== Leave-one-dataset-out cross-dataset generalisation ===")

    for idx, held in enumerate(specs):
        if idx < resume_index:
            print(f"[resume] skipping {held[0]} (already completed)")
            continue

        train_specs = [s for s in specs if s != held]
        train_ds = _build_pool(cfg, train_specs, stream)
        test_ds = _build_pool(cfg, [held], stream)

        if len(train_ds) < 3 or len(test_ds) == 0:
            print(f"[skip] {held[0]}: empty train or test split")
            continue

        # Carve a validation slice out of the *training* corpora. Selecting the best
        # checkpoint on the training data itself (the previous behaviour) just picked
        # the most overfit epoch, which is the opposite of what this study measures.
        inner_train, inner_val, _ = split_indices(len(train_ds), seed, val_frac=0.1, test_frac=0.0)
        train_source = train_ds.with_augment(augment) if augment else train_ds

        model = _train_binary(
            Subset(train_source, inner_train),
            Subset(train_ds, inner_val),
            cfg,
            device,
            class_counts=_counts(train_ds, inner_train),
            stream=stream,
        )
        acc, cm = evaluate_model(
            model, DataLoader(test_ds, batch_size=cfg["training"]["batch_size"]), device
        )

        results[held[0]] = (acc, cm)
        print(f"\n-- tested on held-out: {held[0]} (n={len(test_ds)}) --")
        print(format_report(cm, acc, BINARY_NAMES))

        if ckpt_path is not None:
            with open(ckpt_path, "w") as f:
                json.dump({"last_index": idx + 1, "stream": stream}, f)

    if results:
        print("\n=== Leave-one-dataset-out summary ===")
        for name, (acc, cm) in results.items():
            print(f"  {name:<16} {acc * 100:6.2f}%   (n={int(cm.sum())})")
        mean = float(np.mean([a for a, _ in results.values()]))
        print(f"  {'MEAN':<16} {mean * 100:6.2f}%   <- headline generalisation number")
    return results


def run(config_path="configs/unified.yaml", resume=False, stream="joint", skip_pooled=False):
    import torch
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["training"].get("seed", 42))
    print(f"Using execution device: {device}  (stream={stream})")

    experiment = cfg.get("experiment", "phase4_unified")
    ckpt_dir = os.path.join("outputs/checkpoints", experiment)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"progress_{stream}.json")

    if resume and os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            resume_index = json.load(f).get("last_index", 0)
        print(f"[resume] starting from dataset index {resume_index}")
    else:
        resume_index = 0

    if not skip_pooled:
        pooled_evaluation(cfg, device, stream)

    leave_one_out(cfg, device, resume_index, ckpt_path, stream)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/unified.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stream", default="joint", choices=("joint", "bone", "joint_motion", "bone_motion")
    )
    parser.add_argument(
        "--skip-pooled", action="store_true", help="run only the leave-one-dataset-out study"
    )
    args = parser.parse_args()
    run(args.config, args.resume, args.stream, args.skip_pooled)


if __name__ == "__main__":
    main()
