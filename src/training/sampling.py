"""Dataset-balanced sampling for the pooled corpus.

The pool is 27k lab clips (NTU, Bullying10K) against ~2k real-CCTV clips (UBI-Fights,
fight-surv). Sampled naturally, CCTV is under 7% of every batch, and the pooled
model reached 95.8% on Bullying10K and 87.1% on NTU while scoring 55-62% on the CCTV
datasets it had trained on. The lab corpora dominate the gradient.

``dataset_balance`` is a power ``p`` on each corpus's size: a clip's sampling
weight is ``n_dataset ** -p``, so a corpus's share of every epoch is proportional to
``n_dataset ** (1 - p)``.

* ``p = 0`` — the natural mix (current behaviour).
* ``p = 0.5`` — shares proportional to sqrt(n): CCTV rises from ~7% to ~20%.
* ``p = 1`` — every corpus equal. Maximal, but each of UT-Interaction's ~80 training
  clips is then drawn ~50 times an epoch, which invites memorising them.
"""

from __future__ import annotations

import numpy as np


def dataset_balance_weights(sources, power=0.0):
    """Per-sample sampling weights (summing to 1) from each sample's source corpus."""
    sources = np.asarray(sources)
    if len(sources) == 0:
        return np.zeros(0)
    names, counts = np.unique(sources, return_counts=True)
    per_corpus = dict(zip(names, counts.astype(np.float64) ** -float(power)))
    weights = np.array([per_corpus[s] for s in sources], dtype=np.float64)
    return weights / weights.sum()


def effective_class_counts(labels, weights, num_classes=2):
    """Expected class counts per epoch under weighted sampling, scaled to ``len(labels)``.

    Class-balanced loss weights must come from what the model actually *sees*.
    Rebalancing corpora shifts the class mix — UBI is 40% aggressive, Bullying10K 60% —
    so weights computed from the raw counts would correct for a distribution that
    no longer exists.
    """
    labels = np.asarray(labels)
    weights = np.asarray(weights, dtype=np.float64)
    mass = np.bincount(labels, weights=weights, minlength=num_classes)
    return (mass / max(mass.sum(), 1e-12) * len(labels)).tolist()


def make_sampler(weights, num_samples=None, seed=None):
    """WeightedRandomSampler drawing ``num_samples`` (default: one epoch's worth)."""
    import torch
    from torch.utils.data import WeightedRandomSampler

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(num_samples if num_samples is not None else len(weights)),
        replacement=True,
        generator=generator,
    )


def corpus_shares(sources, weights):
    """{corpus: share of each epoch} — for logging what balancing actually did."""
    sources = np.asarray(sources)
    weights = np.asarray(weights, dtype=np.float64)
    return {str(n): float(weights[sources == n].sum()) for n in np.unique(sources)}
