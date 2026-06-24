import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset

from src.datasets.unified_loader import UnifiedSkeletonDataset, split_indices
from src.models.stgcn import STGCNBaseline


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
):
    """Train ``model`` in place and return it. Shared by train.py and cross-dataset eval.

    If ``best_path`` is given, the checkpoint with the highest validation accuracy
    seen so far is (re)saved there each time it improves — so evaluation can use
    the best-generalizing weights rather than the (overfit) final epoch.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    best_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = correct = total = 0
        for tensors, labels in train_loader:
            tensors, labels = tensors.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(tensors)
            loss = criterion(outputs, labels)
            loss.backward()
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
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_acc": v_acc},
                best_path,
            )

        if checkpoint_dir and (epoch % 10 == 0 or epoch == epochs):
            path = os.path.join(checkpoint_dir, f"stgcn_baseline_epoch_{epoch}.pt")
            torch.save(
                {
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


def train_model(config_path="configs/baseline.yaml"):
    """Train the ST-GCN baseline on a single-dataset pose cache from a YAML config."""
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")
    print(f"Loading unified dataset from cache: {pose_cache}")
    full_dataset = UnifiedSkeletonDataset(data_dir=pose_cache, target_frames=num_frames)
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
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False)

    model = STGCNBaseline(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
        num_persons=config["data"]["max_persons"],
        graph_strategy="spatial",
    ).to(device)

    print(f"Starting training loop ({config['training']['epochs']} epochs)...")
    fit(
        model,
        train_loader,
        val_loader,
        epochs=config["training"]["epochs"],
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        device=device,
        checkpoint_dir="outputs/checkpoints",
        best_path="outputs/checkpoints/stgcn_best.pt",
    )
    print("Training loop completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ST-GCN baseline.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    train_model(parser.parse_args().config)
