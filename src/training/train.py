import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.datasets.unified_loader import UnifiedSkeletonDataset
from src.models.stgcn import STGCNBaseline


def train_model():
    # 1. Load your intact configs/baseline.yaml layout
    config_path = "configs/baseline.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file missing at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Extract configuration values dynamically
    pose_cache = config["data"]["pose_cache"]
    num_frames = config["data"]["num_frames"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]
    num_classes = config["model"]["num_classes"]
    in_channels = config["model"]["in_channels"]
    num_persons = config["data"]["max_persons"]

    # Checkpoint configuration handling
    checkpoint_dir = "outputs/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Device setup (Targets your Blackwell RTX 5060 using the cu128 build)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")

    # 3. Initialize full preprocessed dataset
    print(f"Loading unified dataset from cache: {pose_cache}")
    full_dataset = UnifiedSkeletonDataset(data_dir=pose_cache, target_frames=num_frames)

    if len(full_dataset) == 0:
        print(
            f"[warning] No .npz files found in {pose_cache}. "
            "Run pose extraction first (see README), then re-run training."
        )
        return

    # Train / Validation Split (80% / 20%)
    val_size = max(1, int(len(full_dataset) * 0.2))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 4. Model Instantiation (channels = X, Y, score; persons per the config)
    model = STGCNBaseline(
        in_channels=in_channels,
        num_classes=num_classes,
        num_persons=num_persons,
        graph_strategy="spatial",
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine Annealing Learning Rate Scheduler matching your config
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Training Loop Engine
    print(f"Starting Phase 1 baseline training loop ({epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for tensors, labels in train_loader:
            tensors, labels = tensors.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(tensors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * tensors.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100

        # Run validation pass every epoch
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for tensors, labels in val_loader:
                tensors, labels = tensors.to(device), labels.to(device)
                outputs = model(tensors)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * tensors.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        v_loss = val_loss / val_total
        v_acc = (val_correct / val_total) * 100

        print(
            f"Epoch [{epoch:02d}/{epochs}] | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | Val Loss: {v_loss:.4f} Acc: {v_acc:.2f}%"
        )

        # Save checkpoint weights periodically or on final step
        if epoch % 10 == 0 or epoch == epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f"stgcn_baseline_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                },
                checkpoint_path,
            )
            print(f"[checkpoint] saved to {checkpoint_path}")

    print("Phase 1 training loop completed.")


if __name__ == "__main__":
    train_model()
