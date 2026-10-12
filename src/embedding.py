"""Embedding analysis: UMAP projection and nearest-to-prototype images."""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np
import tensorflow as tf


def extract_embeddings(model, train_dir: str, img_size: int = 224,
                       batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Embeddings, labels and file paths of every training image (no shuffling)."""
    from tensorflow.keras.preprocessing import image_dataset_from_directory

    ds = image_dataset_from_directory(train_dir, labels="inferred", label_mode="int",
                                      image_size=(img_size, img_size), batch_size=batch_size,
                                      shuffle=False)
    file_paths = np.array(ds.file_paths)
    class_names = ds.class_names
    embs, lbls = [], []
    for images, labels in ds:
        _, emb = model(images, training=False)
        embs.append(emb.numpy())
        lbls.append(labels.numpy())
    return np.vstack(embs), np.concatenate(lbls), file_paths, class_names


def plot_umap(embeddings: np.ndarray, labels: np.ndarray, class_names: Sequence[str],
              path: str | None = None, random_state: int = 42):
    """2-D UMAP of the 128-d embeddings, coloured by class."""
    import matplotlib.pyplot as plt
    import umap

    reducer = umap.UMAP(n_components=2, random_state=random_state)
    emb2d = reducer.fit_transform(embeddings)
    fig = plt.figure(figsize=(8, 6))
    for i, cls in enumerate(class_names):
        idx = labels == i
        plt.scatter(emb2d[idx, 0], emb2d[idx, 1], s=10, label=cls)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("UMAP of Embeddings (Train)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    if path:
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    return fig


def nearest_to_prototype(embeddings: np.ndarray, labels: np.ndarray, file_paths: np.ndarray,
                         prototypes: tf.Variable, class_names: Sequence[str]) -> Dict[str, str]:
    """For each class, the training image whose embedding is closest (L2) to its prototype."""
    out: Dict[str, str] = {}
    for i, cls in enumerate(class_names):
        idxs = np.where(labels == i)[0]
        if idxs.size == 0:
            continue
        proto = prototypes[i].numpy()
        d = np.sum((embeddings[idxs] - proto) ** 2, axis=1)
        out[cls] = str(file_paths[idxs[np.argmin(d)]])
    return out


def plot_prototype_images(prototype_image_paths: Dict[str, str], path: str | None = None):
    import matplotlib.pyplot as plt

    cols = 3
    rows = math.ceil(len(prototype_image_paths) / cols)
    fig = plt.figure(figsize=(10, 10))
    for k, (cls, p) in enumerate(prototype_image_paths.items()):
        im = tf.keras.utils.load_img(p, target_size=(160, 160))
        plt.subplot(rows, cols, k + 1)
        plt.imshow(im)
        plt.title(f"{cls} prototype")
        plt.axis("off")
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=100)
        plt.close(fig)
    return fig
