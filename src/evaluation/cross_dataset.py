"""Phase 4 — unified aggressive-vs-neutral model across all datasets.

Pools UT-Interaction, Bullying10K, and NTU under the binary label space
(src/datasets/taxonomy.py) and runs the three Phase 4 analyses:

1. Pooled evaluation — train on a mix of all datasets, report the
   aggressive-vs-neutral confusion matrix on a held-out split.
2. Cross-dataset generalisation / per-dataset ablation — leave-one-dataset-out:
   train on the others, test on the held-out dataset. Shows how well aggression
   cues transfer across camera types and how much each dataset contributes.

Usage:
    python -m src.evaluation.cross_dataset --config configs/unified.yaml
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

def _train_binary(train_ds, val_ds, cfg, device, best_path=None):
    from torch.utils.data import DataLoader

    from src.models.stgcn import STGCNBaseline
    from src.training.train import fit

    model = STGCNBaseline(
        in_channels=cfg["model"]["in_channels"],
        num_classes=2,
        num_persons=cfg["data"]["max_persons"],
        graph_strategy="spatial",
    ).to(device)
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
    )
    return model

def specs_from_config(cfg):
    return [(d["name"], d["cache"]) for d in cfg["data"]["datasets"]]

def pooled_evaluation(cfg, device):
    """Train on a pooled 80% split, evaluate the binary confusion on the held-out 20%."""
    from torch.utils.data import DataLoader, random_split

    from src.datasets.unified_loader import MultiDatasetSkeletonDataset

    normalize = cfg["data"].get("normalize", False)
    max_persons = cfg["data"]["max_persons"]
    ds = MultiDatasetSkeletonDataset(
        specs_from_config(cfg), cfg["data"]["num_frames"], normalize, max_persons
    )
    if len(ds) < 2:
        print("[warning] pooled dataset has <2 samples; skipping pooled evaluation.")
        return None
    val_size = max(1, int(len(ds) * 0.2))
    train_ds, val_ds = random_split(ds, [len(ds) - val_size, val_size])
    # Persist the pooled model so localize.py can run on new footage.
    experiment = cfg.get("experiment", "phase4_unified")
    ckpt_dir = os.path.join("outputs/checkpoints", experiment)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "stgcn_best.pt")
    model = _train_binary(train_ds, val_ds, cfg, device, best_path=best_path)
    acc, cm = evaluate_model(
        model, DataLoader(val_ds, batch_size=cfg["training"]["batch_size"]), device
    )
    print("\n=== Pooled aggressive-vs-neutral evaluation ===")
    print(format_report(cm, acc, BINARY_NAMES))
    print(f"[checkpoint] pooled model saved to {best_path}")
    return acc, cm

def leave_one_out(cfg, device, resume_index=0, ckpt_path=None):
    from torch.utils.data import DataLoader
    from src.datasets.unified_loader import MultiDatasetSkeletonDataset
    import json

    specs = specs_from_config(cfg)
    num_frames = cfg["data"]["num_frames"]
    normalize = cfg["data"].get("normalize", False)
    max_persons = cfg["data"]["max_persons"]
    results = {}
    print("\n=== Leave-one-dataset-out cross-dataset generalisation ===")

    for idx, held in enumerate(specs):

        # Resume support: skip completed datasets
        if idx < resume_index:
            print(f"[resume] skipping {held[0]} (already completed)")
            continue

        train_specs = [s for s in specs if s != held]
        train_ds = MultiDatasetSkeletonDataset(train_specs, num_frames, normalize, max_persons)
        test_ds = MultiDatasetSkeletonDataset([held], num_frames, normalize, max_persons)

        if len(train_ds) == 0 or len(test_ds) == 0:
            print(f"[skip] {held[0]}: empty train or test split")
            continue

        model = _train_binary(train_ds, train_ds, cfg, device)
        acc, cm = evaluate_model(
            model, DataLoader(test_ds, batch_size=cfg["training"]["batch_size"]), device
        )

        results[held[0]] = (acc, cm)
        print(f"\n-- tested on held-out: {held[0]} (n={len(test_ds)}) --")
        print(format_report(cm, acc, BINARY_NAMES))

        # Save progress for resume
        if ckpt_path is not None:
            with open(ckpt_path, "w") as f:
                json.dump({"last_index": idx + 1}, f)

    return results

def run(config_path="configs/unified.yaml", resume=False):
    import torch
    import yaml
    import json
    import os

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")

    # Set up checkpoint path
    experiment = cfg.get("experiment", "phase4_unified")
    ckpt_dir = os.path.join("outputs/checkpoints", experiment)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "progress.json")

    # Load resume index if requested
    if resume and os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            ckpt = json.load(f)
        resume_index = ckpt.get("last_index", 0)
        print(f"[resume] starting from dataset index {resume_index}")
    else:
        resume_index = 0

    # Run pooled eval (no resume needed here)
    pooled_evaluation(cfg, device)

    # Pass resume_index and ckpt_path into leave_one_out
    leave_one_out(cfg, device, resume_index, ckpt_path)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/unified.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.config, resume=args.resume)

if __name__ == "__main__":
    main()
