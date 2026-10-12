"""Post-hoc logit adjustment (class priors) and temperature scaling."""
from __future__ import annotations

import glob
import os
from typing import Sequence, Tuple

import numpy as np


def logit_adjustment_from_dir(train_dir: str, class_names: Sequence[str]) -> np.ndarray:
    """Per-class offset ``-log(prior)`` where prior = class frequency in the training split."""
    counts = {cls: len(glob.glob(os.path.join(train_dir, cls, "*"))) for cls in class_names}
    total = sum(counts.values())
    priors = {cls: counts[cls] / total for cls in class_names}
    return np.array([-np.log(max(priors[cls], 1e-8)) for cls in class_names], dtype=np.float32)


def collect_logits(model, ds, logit_adjustment: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Stack logits (optionally + ``logit_adjustment``) and labels over a dataset."""
    logits_list, labels_list = [], []
    for images, labels in ds:
        logits, _ = model(images, training=False)
        logits = logits.numpy()
        if logit_adjustment is not None:
            logits = logits + logit_adjustment
        logits_list.append(logits)
        labels_list.append(labels.numpy())
    return np.vstack(logits_list), np.concatenate(labels_list)


def nll(T: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood of ``softmax(logits / T)``."""
    if T <= 1e-6:
        return np.inf
    s = logits / T
    m = s.max(axis=1, keepdims=True)
    exp = np.exp(s - m)
    p = exp / exp.sum(axis=1, keepdims=True)
    return float((-np.log(p[np.arange(len(labels)), labels] + 1e-12)).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Two-stage grid search of the temperature that minimises validation NLL.

    Coarse grid: 50 values in [0.1, 5.0]; fine grid: 50 values in
    [max(0.1, best - 0.5), best + 0.5].
    """
    grid = np.linspace(0.1, 5.0, 50)
    best = grid[int(np.argmin([nll(t, logits, labels) for t in grid]))]
    fine = np.linspace(max(0.1, best - 0.5), best + 0.5, 50)
    return float(fine[int(np.argmin([nll(t, logits, labels) for t in fine]))])


def calibrated_probs(logits: np.ndarray, logit_adjustment: np.ndarray, T: float) -> np.ndarray:
    """``softmax((logits + logit_adjustment) / T)`` computed in NumPy."""
    z = (logits + logit_adjustment) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
