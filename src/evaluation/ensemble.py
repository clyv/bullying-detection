"""Late fusion over the four skeleton streams.

Train one model per stream (joint / bone / joint_motion / bone_motion), then sum
their calibrated softmax scores. This is the standard multi-stream recipe and it is
worth a couple of points on any benchmark — but it is worth more here, because bone
and motion streams are translation-invariant and largely scale-invariant, so they
disagree with the joint stream exactly where the joint stream is being fooled by
absolute pixel geometry.

Fusing **calibrated** probabilities rather than raw logits matters: an uncalibrated
stream with saturated scores would dominate the sum regardless of how good it is.
Each stream therefore gets its own temperature, fitted on validation.

Usage:
    python -m src.evaluation.ensemble --config configs/unified.yaml
    python -m src.evaluation.ensemble --config configs/bullying10k.yaml --split test
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from src.datasets.streams import STREAMS
from src.evaluation.calibrate import (
    abstention_report,
    conformal_threshold,
    expected_calibration_error,
    fit_temperature,
    softmax,
)
from src.evaluation.evaluate import (
    accuracy,
    collect_logits,
    confusion_matrix,
    format_report,
)


def fuse(stream_probs, weights=None):
    """Weighted mean of per-stream probability arrays -> (n, num_classes)."""
    if not stream_probs:
        raise ValueError("no streams to fuse")
    stacked = np.stack(stream_probs)  # (S, n, C)
    if weights is None:
        return stacked.mean(axis=0)
    w = np.asarray(weights, dtype=np.float64).reshape(-1, 1, 1)
    return (stacked * w).sum(axis=0) / w.sum()


def stream_checkpoints(checkpoint_dir, config, streams=STREAMS):
    """Map stream -> checkpoint path, skipping streams that were never trained."""
    from src.models.factory import checkpoint_name

    found = {}
    for stream in streams:
        path = os.path.join(checkpoint_dir, checkpoint_name(config, stream))
        if os.path.exists(path):
            found[stream] = path
    return found


def run(config_path="configs/unified.yaml", split="test", device="auto", alpha=0.1):
    """Score every trained stream, fuse them, and report the ensemble."""
    import torch
    import yaml
    from torch.utils.data import DataLoader, Subset

    from src.datasets.unified_loader import (
        MultiDatasetSkeletonDataset,
        UnifiedSkeletonDataset,
        split_indices,
    )
    from src.models.factory import load_for_inference

    with open(config_path) as f:
        config = yaml.safe_load(f)

    experiment = config.get("experiment", "default")
    checkpoint_dir = os.path.join("outputs/checkpoints", experiment)
    available = stream_checkpoints(checkpoint_dir, config)
    if not available:
        print(f"[error] no stream checkpoints in {checkpoint_dir}.")
        print("        Train them first, e.g.:")
        for stream in STREAMS:
            print(
                f"          python -m src.training.train --config {config_path} --stream {stream}"
            )
        return None
    print(f"Found {len(available)} stream(s): {', '.join(available)}")

    torch_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    data_cfg = config["data"]
    batch_size = config["training"]["batch_size"]
    seed = config["training"].get("seed", 42)

    def make_dataset(stream):
        """Same corpus, different input representation."""
        if "pose_cache" in data_cfg:
            return UnifiedSkeletonDataset(
                data_cfg["pose_cache"],
                data_cfg["num_frames"],
                data_cfg.get("normalize", False),
                data_cfg["max_persons"],
                stream=stream,
            )
        specs = [(d["name"], d["cache"]) for d in data_cfg["datasets"]]
        return MultiDatasetSkeletonDataset(
            specs,
            data_cfg["num_frames"],
            data_cfg.get("normalize", False),
            data_cfg["max_persons"],
            stream=stream,
        )

    base = make_dataset("joint")
    if len(base) == 0:
        print("[warning] dataset is empty; nothing to evaluate.")
        return None
    train_idx, val_idx, test_idx = split_indices(
        len(base), seed, data_cfg.get("val_frac", 0.15), data_cfg.get("test_frac", 0.15)
    )
    index = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
    num_classes = config["model"]["num_classes"]

    test_probs, val_probs, targets, val_targets = [], [], None, None
    for stream, path in available.items():
        dataset = base.with_stream(stream)
        # Per stream: each checkpoint may have been trained under a different
        # normalization, and the clone lets each one get its own.
        model, dataset.normalize = load_for_inference(path, config, torch_device, num_classes)

        logits, tgt = collect_logits(
            model, DataLoader(Subset(dataset, index), batch_size=batch_size), torch_device
        )
        v_logits, v_tgt = collect_logits(
            model, DataLoader(Subset(dataset, val_idx), batch_size=batch_size), torch_device
        )
        temperature = fit_temperature(v_logits, v_tgt)
        acc = accuracy(logits.argmax(axis=1), tgt)
        print(
            f"  {stream:<13} acc={acc * 100:5.2f}%  T={temperature:.3f}  ({os.path.basename(path)})"
        )

        test_probs.append(softmax(logits, temperature))
        val_probs.append(softmax(v_logits, temperature))
        targets, val_targets = tgt, v_tgt

    fused = fuse(test_probs)
    fused_val = fuse(val_probs)
    preds = fused.argmax(axis=1)
    cm = confusion_matrix(preds, targets, num_classes)
    acc = accuracy(preds, targets)

    print(f"\n=== Ensemble of {len(test_probs)} stream(s) on '{split}' ===")
    print(format_report(cm, acc, config["model"].get("class_names")))

    # Each stream was calibrated before fusion, so there is no second temperature to
    # fit here; ECE on the fused distribution is what says whether that held up.
    threshold = conformal_threshold(fused_val, val_targets, alpha)
    report = abstention_report(fused, targets, threshold)
    print()
    print(f"Ensemble ECE       {expected_calibration_error(fused, targets):.4f}")
    print(f"  coverage         {report['coverage'] * 100:.1f}%")
    print(f"  abstention rate  {report['abstention_rate'] * 100:.1f}%")
    print(f"  selective acc.   {report['selective_accuracy'] * 100:.2f}%")
    print(f"  conformal q      {threshold:.4f}  (alpha={alpha})")
    return {"accuracy": acc, "confusion_matrix": cm, "streams": list(available)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/unified.yaml")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    run(args.config, args.split, args.device, args.alpha)


if __name__ == "__main__":
    main()
