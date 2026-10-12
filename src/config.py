"""Hyper-parameters and paths (mirrors the first cell of the notebook)."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import List

KAGGLE_DATASET = "anirudhg15/mammals-classification"


@dataclass
class Config:
    # Target classes (only these folders are taken from the Kaggle dataset)
    classes: List[str] = field(default_factory=lambda: [
        "bear", "cat", "dog", "elephant", "goat", "horse", "lion", "tiger", "wolf"])

    # Data cleaning
    min_side: int = 160          # drop images whose shorter side < 160 px
    blur_thr: float = 60.0       # drop images whose Laplacian variance < 60
    phash_dup_thr: int = 6       # pHash Hamming distance <= 6 -> near-duplicate

    # Input pipeline / model
    img_size: int = 224
    batch_size: int = 32
    embed_dim: int = 128

    # Training
    lambda_center: float = 0.5   # weight of the prototype (center) loss
    proto_momentum: float = 0.1  # EMA momentum of the class prototypes
    epochs: int = 10
    learning_rate: float = 1e-3
    seed: int = 1337

    # Split ratios: 10 % test, then 22.22 % of the rest as validation (~70/20/10)
    test_size: float = 0.1
    val_size: float = 0.2222

    # Working directories (the notebook used /content/... on Colab)
    work_dir: str = "work"

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.work_dir, "dataset_raw")

    @property
    def clean_dir(self) -> str:
        return os.path.join(self.work_dir, "dataset_clean")

    @property
    def split_dir(self) -> str:
        return os.path.join(self.work_dir, "subset_clean")

    @property
    def export_dir(self) -> str:
        return os.path.join(self.work_dir, "export")

    @property
    def figure_dir(self) -> str:
        return os.path.join(self.work_dir, "figures")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow RNGs exactly like the notebook."""
    import numpy as np
    import tensorflow as tf

    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
