"""Custom training loop: cross-entropy + prototype (center) loss with EMA prototypes."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


def eval_epoch(model: tf.keras.Model, ds, ce_loss_fn) -> Tuple[float, float]:
    """Mean cross-entropy and accuracy of ``model`` on ``ds`` (inference mode)."""
    tloss, corr, n = 0.0, 0, 0
    for images, labels in ds:
        labels = tf.cast(labels, tf.int32)
        logits, _ = model(images, training=False)
        loss = ce_loss_fn(labels, logits)
        tloss += float(loss) * images.shape[0]
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        corr += int(tf.reduce_sum(tf.cast(preds == labels, tf.int32)))
        n += images.shape[0]
    return tloss / n, corr / n


def train(model: tf.keras.Model, prototypes: tf.Variable, train_ds, val_ds,
          epochs: int = 10, lambda_center: float = 0.5, proto_momentum: float = 0.1,
          learning_rate: float = 1e-3) -> Dict[str, List[float]]:
    """Train ``model`` and update ``prototypes`` in place.

    Per mini-batch:
      loss = CE(labels, logits) + lambda_center * mean_i ||e_i - c_{y_i}||^2
    followed by an EMA update of every class present in the batch:
      c_k <- (1 - m) * c_k + m * mean(e_i | y_i = k)

    Note: the reported training loss therefore includes the center-loss term,
    which is why it is much larger than the validation (cross-entropy only) loss.
    """
    ce_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for e in range(1, epochs + 1):
        print(f"\nEpoch {e}/{epochs}")
        tloss, corr, n = 0.0, 0, 0
        for images, labels in train_ds:
            labels = tf.cast(labels, tf.int32)
            with tf.GradientTape() as tape:
                logits, embeds = model(images, training=True)
                ce = ce_loss_fn(labels, logits)
                centers = tf.gather(prototypes, labels)
                c_loss = tf.reduce_mean(tf.reduce_sum(tf.square(embeds - centers), axis=1))
                loss = ce + lambda_center * c_loss
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # EMA prototype update for the classes present in this batch
            en, ln = embeds.numpy(), labels.numpy()
            for cls_idx in np.unique(ln):
                mean = en[ln == cls_idx].mean(axis=0)
                prototypes[cls_idx].assign((1 - proto_momentum) * prototypes[cls_idx] + proto_momentum * mean)

            tloss += float(loss) * images.shape[0]
            preds = tf.argmax(logits, axis=1, output_type=tf.int32)
            corr += int(tf.reduce_sum(tf.cast(preds == labels, tf.int32)))
            n += images.shape[0]

        tr_loss, tr_acc = tloss / n, corr / n
        va_loss, va_acc = eval_epoch(model, val_ds, ce_loss_fn)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(f" - train_loss: {tr_loss:.4f} - train_acc: {tr_acc:.4f}"
              f" - val_loss: {va_loss:.4f} - val_acc: {va_acc:.4f}")
    return history


def plot_training_curves(history: Dict[str, List[float]], path: str | None = None):
    """Loss / accuracy curves for train and validation."""
    import matplotlib.pyplot as plt

    ep = range(1, len(history["train_loss"]) + 1)
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ep, history["train_loss"], label="train")
    plt.plot(ep, history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(ep, history["train_acc"], label="train")
    plt.plot(ep, history["val_acc"], label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=100)
        plt.close(fig)
    return fig
