"""Test-set evaluation: accuracy, classification report, confusion matrix, ECE."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import tensorflow as tf

from .calibration import calibrated_probs


def evaluate_test(model, test_ds, class_names: Sequence[str]) -> Dict:
    """Arg-max of the raw logits on the test split (as in the notebook)."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    y_true, y_pred = [], []
    for images, labels in test_ds:
        logits, _ = model(images, training=False)
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        y_pred.extend(list(preds.numpy()))
        y_true.extend(list(labels.numpy()))
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=list(class_names), digits=4)
    cm = confusion_matrix(y_true, y_pred)
    print(f"Test Acc: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(report)
    return {"accuracy": acc, "report": report, "confusion_matrix": cm,
            "y_true": np.array(y_true), "y_pred": np.array(y_pred)}


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], path: str | None = None):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=100)
        plt.close(fig)
    return fig


def reliability(model, test_ds, logit_adjustment: np.ndarray, T: float, n_bins: int = 10) -> Dict:
    """Expected Calibration Error of the calibrated probabilities on the test split.

    Probabilities are ``softmax((logits + logit_adjustment) / T)``. Confidence =
    max probability, binned into ``n_bins`` equal-width bins on [0, 1];
    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|.
    """
    ytrue, maxprob, predlab = [], [], []
    for images, labels in test_ds:
        logits, _ = model(images, training=False)
        probs = calibrated_probs(logits.numpy(), logit_adjustment, T)
        maxprob.extend(list(probs.max(axis=1)))
        predlab.extend(list(probs.argmax(axis=1)))
        ytrue.extend(list(labels.numpy()))
    ytrue, maxprob, predlab = np.array(ytrue), np.array(maxprob), np.array(predlab)
    correct = predlab == ytrue

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(maxprob, bins) - 1
    bin_tot, bin_cor, bin_conf = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    for p, c, b in zip(maxprob, correct, idx):
        if 0 <= b < n_bins:
            bin_tot[b] += 1
            bin_cor[b] += 1 if c else 0
            bin_conf[b] += p
    bin_acc, bin_avg = np.zeros(n_bins), np.zeros(n_bins)
    for i in range(n_bins):
        if bin_tot[i] > 0:
            bin_acc[i] = bin_cor[i] / bin_tot[i]
            bin_avg[i] = bin_conf[i] / bin_tot[i]
        else:
            bin_acc[i] = 0
            bin_avg[i] = (i + 0.5) / n_bins
    ece = float(np.sum((bin_tot / len(ytrue)) * np.abs(bin_acc - bin_avg)))
    print("ECE:", round(ece, 4))
    return {"ece": ece, "bin_acc": bin_acc, "bin_avg": bin_avg, "bin_tot": bin_tot}


def plot_reliability_diagram(bin_acc: np.ndarray, bin_avg: np.ndarray, path: str | None = None):
    import matplotlib.pyplot as plt

    n_bins = len(bin_acc)
    fig = plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--")
    centers = np.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins)
    plt.bar(centers, bin_acc, width=0.9 / n_bins, align="center", alpha=0.7, label="accuracy/bin")
    plt.scatter(centers, bin_avg, label="avg confidence")
    plt.xlabel("confidence")
    plt.ylabel("accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    if path:
        fig.savefig(path, dpi=100)
        plt.close(fig)
    return fig
