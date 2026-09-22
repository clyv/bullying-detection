import argparse
import math
import os

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from src.datasets.augment import AugmentConfig
from src.datasets.unified_loader import UnifiedSkeletonDataset, split_indices
from src.models.factory import build_model, checkpoint_meta, checkpoint_name
from src.training.losses import (
    FocalLoss,
    class_weights_from_counts,
    mixup_batch,
    soft_target_cross_entropy,
)


def build_criterion(config, class_counts=None, device=None):
    """Loss from config: focal by default, plain CE when ``gamma`` is 0.

    Focal loss is the default because this project's failure mode was saturated
    confidence, not low accuracy — see src/training/losses.py.
    """
    training = config.get("training", {})
    gamma = float(training.get("focal_gamma", 2.0))
    smoothing = float(training.get("label_smoothing", 0.0))
    weight = None
    if training.get("class_balanced", True) and class_counts is not None:
        weight = class_weights_from_counts(class_counts, device=device)
    return FocalLoss(gamma=gamma, weight=weight, label_smoothing=smoothing).to(device)


def dataset_labels(dataset, indices=None):
    """Label array for a dataset (or a subset of it), without decoding tensors.

    MultiDatasetSkeletonDataset already knows every label from its scan; the
    single-cache loader has to read them, but only the ``label`` field.
    """
    if hasattr(dataset, "samples"):
        labels = np.array([s[2] for s in dataset.samples])
    elif hasattr(dataset, "labels"):
        labels = np.asarray(dataset.labels())
    else:
        return None
    return labels if indices is None else labels[np.asarray(indices)]


def make_scheduler(optimizer, epochs, warmup_epochs=0):
    """Linear warmup then cosine decay.

    Warmup matters for the adaptive backbone: its data-dependent adjacency term is
    unstable in the first few hundred steps when the learning rate starts at full.
    """
    warmup_epochs = max(0, min(int(warmup_epochs), max(epochs - 1, 0)))

    def lr_lambda(epoch):
        if warmup_epochs and epoch < warmup_epochs:
            return (epoch + 1) / (warmup_epochs + 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def fit(
    model,
    train_loader,
    val_loader,
    *,
    epochs,
    lr,
    weight_decay,
    device,
    checkpoint_dir=None,
    best_path=None,
    criterion=None,
    num_classes=2,
    mixup_alpha=0.0,
    clip_grad=1.0,
    warmup_epochs=0,
    checkpoint_meta=None,
):
    """Train ``model`` in place and return it. Shared by train.py and cross-dataset eval.

    If ``best_path`` is given, the checkpoint with the highest validation accuracy
    seen so far is (re)saved there each time it improves — so evaluation can use
    the best-generalizing weights rather than the (overfit) final epoch.

    ``checkpoint_meta`` (see models.factory.checkpoint_meta) is written into every
    saved file so inference tools can rebuild the right architecture and feed it
    the normalization it was trained on, whatever the config says by then.
    """
    meta = dict(checkpoint_meta or {})
    criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, epochs, warmup_epochs)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    best_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = correct = total = 0
        for tensors, labels in train_loader:
            tensors, labels = tensors.to(device), labels.to(device)
            optimizer.zero_grad()
            if mixup_alpha > 0:
                mixed, soft = mixup_batch(tensors, labels, num_classes, mixup_alpha)
                outputs = model(mixed)
                loss = soft_target_cross_entropy(outputs, soft)
            else:
                outputs = model(tensors)
                loss = criterion(outputs, labels)
            loss.backward()
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            running_loss += loss.item() * tensors.size(0)
            total += labels.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
        scheduler.step()
        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1) * 100

        v_loss, v_acc = _evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch [{epoch:02d}/{epochs}] | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% "
            f"| Val Loss: {v_loss:.4f} Acc: {v_acc:.2f}%"
        )

        if best_path and v_acc > best_acc:
            best_acc = v_acc
            torch.save(
                {
                    **meta,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": v_acc,
                },
                best_path,
            )

        if checkpoint_dir and (epoch % 10 == 0 or epoch == epochs):
            path = os.path.join(checkpoint_dir, f"stgcn_baseline_epoch_{epoch}.pt")
            torch.save(
                {
                    **meta,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                },
                path,
            )
            print(f"[checkpoint] saved to {path}")
    if best_path and best_acc >= 0:
        print(f"[checkpoint] best val acc {best_acc:.2f}% -> {best_path}")
    return model


def _evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    with torch.no_grad():
        for tensors, labels in loader:
            tensors, labels = tensors.to(device), labels.to(device)
            outputs = model(tensors)
            loss_sum += criterion(outputs, labels).item() * tensors.size(0)
            total += labels.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
    if total == 0:
        return 0.0, 0.0
    return loss_sum / total, correct / total * 100


def resolve_device(name="auto"):
    """'auto' -> cuda if available else cpu; otherwise the named device."""
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_model(config_path="configs/baseline.yaml", device="auto", stream="joint"):
    """Train the configured model on a single-dataset pose cache from a YAML config."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file missing at {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pose_cache = config["data"]["pose_cache"]
    num_frames = config["data"]["num_frames"]
    batch_size = config["training"]["batch_size"]
    seed = config["training"].get("seed", 42)
    val_frac = config["data"].get("val_frac", 0.15)
    test_frac = config["data"].get("test_frac", 0.15)
    num_classes = config["model"]["num_classes"]
    # Checkpoints are namespaced per experiment so datasets don't overwrite each other.
    experiment = config.get("experiment", "default")
    checkpoint_dir = os.path.join("outputs/checkpoints", experiment)

    device = resolve_device(device)
    torch.manual_seed(seed)
    print(f"Using execution device: {device}")
    print(f"Loading unified dataset from cache: {pose_cache}  (stream={stream})")
    normalize = config["data"].get("normalize", False)
    full_dataset = UnifiedSkeletonDataset(
        pose_cache, num_frames, normalize, config["data"]["max_persons"], stream=stream
    )
    if len(full_dataset) == 0:
        print(
            f"[warning] No .npz files found in {pose_cache}. "
            "Run pose extraction first (see README), then re-run training."
        )
        return

    # Seeded split shared with evaluate.py; the test slice is never seen here.
    train_idx, val_idx, test_idx = split_indices(len(full_dataset), seed, val_frac, test_frac)
    print(
        f"Split (seed={seed}): train={len(train_idx)} val={len(val_idx)} "
        f"test={len(test_idx)} (test held out for evaluation)"
    )

    # Augmentation applies to the training subset only — a clone sharing the scan.
    augment = AugmentConfig.from_dict(config.get("augment"))
    train_source = full_dataset.with_augment(augment) if augment else full_dataset
    if augment:
        print(f"Augmentation: {augment}")
    train_loader = DataLoader(Subset(train_source, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False)

    model = build_model(config).to(device)
    counts = _label_counts(full_dataset, train_idx, num_classes)
    criterion = build_criterion(config, counts, device)
    print(f"Model: {config['model'].get('name', 'stgcn')} | class counts (train): {counts}")

    print(f"Starting training loop ({config['training']['epochs']} epochs)...")
    fit(
        model,
        train_loader,
        val_loader,
        epochs=config["training"]["epochs"],
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        device=device,
        checkpoint_dir=checkpoint_dir,
        best_path=os.path.join(checkpoint_dir, checkpoint_name(config, stream)),
        criterion=criterion,
        num_classes=num_classes,
        mixup_alpha=config["training"].get("mixup_alpha", 0.0),
        clip_grad=config["training"].get("clip_grad", 1.0),
        warmup_epochs=config["training"].get("warmup_epochs", 0),
        checkpoint_meta=checkpoint_meta(config, stream),
    )
    print(f"Training loop completed. Checkpoints in {checkpoint_dir}")


def _label_counts(dataset, indices, num_classes):
    """Per-class sample counts on the training indices (None-safe)."""
    labels = dataset_labels(dataset, indices)
    if labels is None:
        labels = np.array([int(dataset[i][1]) for i in indices])
    return np.bincount(labels, minlength=num_classes).tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the skeleton action classifier.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--device", default="auto", help='"auto", "cpu", or "cuda"')
    parser.add_argument(
        "--stream",
        default="joint",
        choices=("joint", "bone", "joint_motion", "bone_motion"),
        help="input representation; train all four and ensemble for the best result",
    )
    args = parser.parse_args()
    train_model(args.config, args.device, args.stream)
