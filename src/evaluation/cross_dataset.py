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


def _train_binary(
    train_ds,
    val_ds,
    cfg,
    device,
    best_path=None,
    labels=None,
    sources=None,
    stream="joint",
):
    """Train one binary model. ``labels``/``sources`` align with ``train_ds``'s items.

    With ``training.dataset_balance`` > 0, batches are drawn by corpus-balanced
    weights (src/training/sampling.py) and the class-balanced loss is computed from
    the class mix those weights actually produce.
    """
    from torch.utils.data import DataLoader

    from src.models.factory import build_model, checkpoint_meta
    from src.training.sampling import (
        corpus_shares,
        dataset_balance_weights,
        effective_class_counts,
        make_sampler,
    )
    from src.training.train import build_criterion, fit

    model = build_model(cfg, num_classes=2).to(device)
    batch_size = cfg["training"]["batch_size"]
    power = float(cfg["training"].get("dataset_balance", 0.0))

    class_counts = None
    if labels is not None:
        class_counts = np.bincount(np.asarray(labels), minlength=2).tolist()
    if power > 0 and sources is not None:
        weights = dataset_balance_weights(sources, power)
        if labels is not None:
            class_counts = effective_class_counts(labels, weights)
        shares = ", ".join(f"{n} {s:.0%}" for n, s in corpus_shares(sources, weights).items())
        print(f"[sampling] dataset_balance={power}: {shares}")
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=make_sampler(weights, seed=cfg["training"].get("seed", 42)),
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    fit(
        model,
        train_loader,
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


def _labels_and_sources(dataset, indices):
    """Binary labels and source corpora for ``indices`` into a pooled dataset."""
    picked = [dataset.samples[int(i)] for i in indices]
    return [s[2] for s in picked], [s[1] for s in picked]


def per_dataset_report(preds, targets, sources):
    """Accuracy and per-class recall broken down by source corpus.

    The pooled number is a weighted average dominated by whichever corpora are
    largest. One run printed 88% overall while scoring 95.8% on Bullying10K, 87.1%
    on NTU — and 55-62% on the real-CCTV datasets it had trained on. This table is
    what stops a headline number from hiding that.
    """
    preds, targets, sources = np.asarray(preds), np.asarray(targets), np.asarray(sources)
    lines = [f"  {'source':<16} {'acc':>7} {'agg-recall':>11} {'neu-recall':>11} {'n':>7}"]
    for name in sorted(set(sources.tolist())):
        mask = sources == name
        aggressive, neutral = mask & (targets == 1), mask & (targets == 0)

        def pct(values):
            return f"{values.mean() * 100:6.1f}%" if len(values) else "     - "

        lines.append(
            f"  {name:<16} {pct(preds[mask] == targets[mask])} "
            f"{pct(preds[aggressive] == 1):>11} {pct(preds[neutral] == 0):>11} "
            f"{int(mask.sum()):>7}"
        )
    return "\n".join(lines)


ENERGY_CACHE = os.path.join("outputs", "cache", "motion_energy.json")


def pool_energies(dataset, cache_path=ENERGY_CACHE):
    """Motion energy for every clip in a pooled dataset, keyed by file path.

    Cached on disk, keyed by path and modification time: reading all ~29k clips
    took 23 minutes on a laptop on battery, which is not worth repeating on every
    leave-one-dataset-out run when the clips haven't changed.
    """
    from src.evaluation.baselines import clip_motion_energy

    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    energies, computed = {}, 0
    for path, _, _ in dataset.samples:
        key = f"{path}|{os.path.getmtime(path):.0f}"
        if key not in cache:
            with np.load(path) as data:
                cache[key] = clip_motion_energy(data["keypoints"], data["scores"])
            computed += 1
        energies[path] = cache[key]
    if computed and cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f)
    print(f"[baseline] motion energy: {computed} computed, {len(energies) - computed} cached")
    return energies


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

    labels, sources = _labels_and_sources(ds, train_idx)
    # fit() restores the best-validation weights, so what is tested below is the
    # same model that was saved to best_path.
    model = _train_binary(
        Subset(train_source, train_idx),
        Subset(ds, val_idx),
        cfg,
        device,
        best_path=best_path,
        labels=labels,
        sources=sources,
        stream=stream,
    )
    preds, targets = predict(
        model,
        DataLoader(Subset(ds, test_idx), batch_size=cfg["training"]["batch_size"]),
        device,
    )
    cm = confusion_matrix(preds, targets, 2)
    acc = accuracy(preds, targets)
    print("\n=== Pooled aggressive-vs-neutral evaluation (held-out test split) ===")
    print(format_report(cm, acc, BINARY_NAMES))
    print("\nBy source dataset:")
    print(per_dataset_report(preds, targets, _labels_and_sources(ds, test_idx)[1]))
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
    from src.evaluation.baselines import fit_energy_threshold, threshold_accuracy

    specs = specs_from_config(cfg)
    augment = AugmentConfig.from_dict(cfg.get("augment"))
    seed = cfg["training"].get("seed", 42)
    results = {}
    baseline = {}
    print("\n=== Leave-one-dataset-out cross-dataset generalisation ===")
    # One pass over every clip for the motion-energy baseline (seconds per thousand
    # clips); each fold then fits its threshold on the corpora it trains on.
    print("[baseline] measuring motion energy for every clip ...")
    energies = pool_energies(_build_pool(cfg, specs, stream))

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

        labels, sources = _labels_and_sources(train_ds, inner_train)
        model = _train_binary(
            Subset(train_source, inner_train),
            Subset(train_ds, inner_val),
            cfg,
            device,
            labels=labels,
            sources=sources,
            stream=stream,
        )
        acc, cm = evaluate_model(
            model, DataLoader(test_ds, batch_size=cfg["training"]["batch_size"]), device
        )

        # The same held-out corpus, scored by "faster than t means aggressive" with
        # t fitted on the training corpora only.
        threshold = fit_energy_threshold(
            [energies[p] for p, _, _ in train_ds.samples], [y for _, _, y in train_ds.samples]
        )
        baseline[held[0]] = threshold_accuracy(
            [energies[p] for p, _, _ in test_ds.samples],
            [y for _, _, y in test_ds.samples],
            threshold,
        )

        results[held[0]] = (acc, cm)
        print(f"\n-- tested on held-out: {held[0]} (n={len(test_ds)}) --")
        print(format_report(cm, acc, BINARY_NAMES))
        print(f"motion-energy baseline on the same clips: {baseline[held[0]] * 100:.2f}%")

        if ckpt_path is not None:
            with open(ckpt_path, "w") as f:
                json.dump({"last_index": idx + 1, "stream": stream}, f)

    if results:
        print("\n=== Leave-one-dataset-out summary ===")
        print(f"  {'held out':<16} {'model':>8} {'energy':>8} {'gain':>7}   n")
        for name, (acc, cm) in results.items():
            gain = (acc - baseline[name]) * 100
            print(
                f"  {name:<16} {acc * 100:7.2f}% {baseline[name] * 100:7.2f}% "
                f"{gain:+6.2f}   {int(cm.sum())}"
            )
        mean = float(np.mean([a for a, _ in results.values()]))
        base_mean = float(np.mean([baseline[n] for n in results]))
        print(
            f"  {'MEAN':<16} {mean * 100:7.2f}% {base_mean * 100:7.2f}% "
            f"{(mean - base_mean) * 100:+6.2f}   <- headline generalisation number"
        )
        print(
            "  'energy' = one threshold on movement speed. The gain column is how much"
            " the model knows about aggression beyond 'someone is moving fast'."
        )
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
