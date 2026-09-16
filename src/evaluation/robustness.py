"""Degradation benchmark: accuracy as a function of pose-quality corruption.

"It doesn't work on real CCTV" is an anecdote. This turns it into a curve with an
x-axis, by taking the held-out split and corrupting it along one axis at a time:

* **joint_dropout** — fraction of keypoints marked invisible. This is what actually
  happens at range: the extractor returns fewer joints, not worse ones.
* **coord_noise** — gaussian jitter in units of skeleton extent (low-resolution
  subjects have proportionally noisier keypoints).
* **person_dropout** — one participant lost entirely to occlusion.
* **scale_jitter** — residual scale error that per-clip normalization did not remove.

The output tells you which corruption costs the most accuracy, which is the one
worth engineering against. Run it before and after a change to show the change did
something on the axis that matters, not just on clean lab clips.

Usage:
    python -m src.evaluation.robustness --config configs/unified.yaml
    python -m src.evaluation.robustness --config configs/bullying10k.yaml --stream bone
"""

from __future__ import annotations

import argparse
import os

from src.datasets.augment import AugmentConfig
from src.evaluation.evaluate import accuracy, collect_logits

# Sweep levels per corruption axis. 0.0 is always first so every curve starts from
# the clean baseline measured on the same samples.
SWEEPS = {
    "joint_dropout": (0.0, 0.1, 0.2, 0.3, 0.5, 0.7),
    "coord_noise": (0.0, 0.02, 0.05, 0.1, 0.2, 0.4),
    "person_dropout": (0.0, 0.1, 0.25, 0.5),
    "scale_jitter": (0.0, 0.1, 0.25, 0.5, 0.75),
}


def corruption_config(axis: str, level: float) -> AugmentConfig | None:
    """An AugmentConfig with every transform off except ``axis``.

    Returns None at level 0 so the clean baseline goes through the ordinary
    un-augmented path rather than a no-op augmentation path.
    """
    if level <= 0:
        return None
    if axis not in SWEEPS:
        raise ValueError(f"unknown corruption axis {axis!r}; expected one of {tuple(SWEEPS)}")
    zeros = {
        "joint_dropout": 0.0,
        "person_dropout": 0.0,
        "coord_noise": 0.0,
        "flip_prob": 0.0,
        "person_swap_prob": 0.0,
        "temporal_crop": 0.0,
        "scale_jitter": 0.0,
        "rotate_degrees": 0.0,
        "shear": 0.0,
    }
    zeros[axis] = level
    return AugmentConfig(**zeros)


def format_sweep(axis, levels, accuracies, baseline):
    """One corruption axis as an aligned table with the drop from clean."""
    lines = [f"  {axis}:"]
    for level, acc in zip(levels, accuracies):
        delta = (acc - baseline) * 100
        bar = "#" * round(acc * 40)
        lines.append(f"    {level:<5.2f}  {acc * 100:6.2f}%  {delta:+6.2f}  {bar}")
    return "\n".join(lines)


def run(config_path="configs/unified.yaml", split="test", device="auto", stream="joint", seed=1234):
    """Sweep every corruption axis on the held-out split and print the curves."""
    import torch
    import yaml
    from torch.utils.data import DataLoader, Subset

    from src.datasets.unified_loader import (
        MultiDatasetSkeletonDataset,
        UnifiedSkeletonDataset,
        split_indices,
    )
    from src.evaluation.evaluate import default_checkpoint
    from src.models.factory import build_model, checkpoint_name

    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    if "pose_cache" in data_cfg:
        base = UnifiedSkeletonDataset(
            data_cfg["pose_cache"],
            data_cfg["num_frames"],
            data_cfg.get("normalize", False),
            data_cfg["max_persons"],
            stream=stream,
        )
    else:
        specs = [(d["name"], d["cache"]) for d in data_cfg["datasets"]]
        base = MultiDatasetSkeletonDataset(
            specs,
            data_cfg["num_frames"],
            data_cfg.get("normalize", False),
            data_cfg["max_persons"],
            stream=stream,
        )
    if len(base) == 0:
        print("[warning] dataset is empty; nothing to evaluate.")
        return None

    experiment = config.get("experiment", "default")
    ckpt = default_checkpoint(
        os.path.join("outputs/checkpoints", experiment), checkpoint_name(config, stream)
    )
    if ckpt is None or not os.path.exists(ckpt):
        print(f"[error] no checkpoint for experiment '{experiment}'; train it first.")
        return None

    torch_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    model = build_model(config).to(torch_device)
    state = torch.load(ckpt, map_location=torch_device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))

    train_idx, val_idx, test_idx = split_indices(
        len(base),
        config["training"].get("seed", 42),
        data_cfg.get("val_frac", 0.15),
        data_cfg.get("test_frac", 0.15),
    )
    index = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
    batch_size = config["training"]["batch_size"]

    print(f"Checkpoint: {ckpt}")
    print(f"Degradation sweep on '{split}' split (n={len(index)}, stream={stream})\n")

    results = {}
    baseline = None
    for axis, levels in SWEEPS.items():
        accuracies = []
        for level in levels:
            cfg = corruption_config(axis, level)
            # Seeded so the same corruption is applied on every re-run: the sweep is
            # a measurement, and a measurement that moves when nothing changed is
            # not one.
            dataset = base.with_augment(cfg, seed=seed) if cfg else base
            logits, targets = collect_logits(
                model, DataLoader(Subset(dataset, index), batch_size=batch_size), torch_device
            )
            acc = accuracy(logits.argmax(axis=1), targets)
            accuracies.append(acc)
            if baseline is None:
                baseline = acc
        results[axis] = (levels, accuracies)
        print(format_sweep(axis, levels, accuracies, baseline))
        print()

    worst = min(((axis, vals[1][-1]) for axis, vals in results.items()), key=lambda pair: pair[1])
    print(f"Clean baseline: {baseline * 100:.2f}%")
    print(f"Most damaging axis at max level: {worst[0]} -> {worst[1] * 100:.2f}%")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/unified.yaml")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--stream", default="joint", choices=("joint", "bone", "joint_motion", "bone_motion")
    )
    parser.add_argument("--seed", type=int, default=1234, help="corruption seed (reproducibility)")
    args = parser.parse_args()
    run(args.config, args.split, args.device, args.stream, args.seed)


if __name__ == "__main__":
    main()
