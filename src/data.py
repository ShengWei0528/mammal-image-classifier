"""Dataset download, cleaning, stratified split and tf.data pipeline."""
from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from .config import KAGGLE_DATASET

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------- #
# 1. Download (KaggleHub)
# --------------------------------------------------------------------------- #
def download_dataset(raw_dir: str, classes: Sequence[str]) -> Dict[str, int]:
    """Download the Kaggle dataset and copy images of the target classes.

    An image belongs to class ``c`` when its parent folder is named ``c``.
    Returns the number of copied images per class.
    """
    import kagglehub

    print(f"Downloading {KAGGLE_DATASET} via KaggleHub")
    try:
        path = kagglehub.dataset_download(KAGGLE_DATASET)
    except Exception:
        kagglehub.login()
        path = kagglehub.dataset_download(KAGGLE_DATASET)
    print("Local path:", path)

    os.makedirs(raw_dir, exist_ok=True)
    moved = 0
    for p in glob.glob(os.path.join(path, "**", "*.*"), recursive=True):
        if not p.lower().endswith(IMAGE_EXTS):
            continue
        cls = Path(p).parent.name
        if cls not in classes:
            continue
        dst = os.path.join(raw_dir, cls)
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(p, os.path.join(dst, Path(p).name))
        moved += 1
    print("Copied images:", moved)
    counts = {c: len(glob.glob(os.path.join(raw_dir, c, "*"))) for c in classes}
    print("Counts:", counts)
    return counts


# --------------------------------------------------------------------------- #
# 2. Cleaning: pHash near-duplicates, minimum side, Laplacian blur
# --------------------------------------------------------------------------- #
def laplacian_variance(pil_img: Image.Image) -> float:
    """Variance of the Laplacian of the grey-scale image (sharpness score)."""
    import cv2

    return cv2.Laplacian(np.array(pil_img.convert("L")), cv2.CV_64F).var()


def find_near_duplicates(paths: Sequence[str], thr: int | None) -> set:
    """Return the set of paths flagged as near-duplicates within one class.

    Every pair (i, j) with i < j is compared; if the pHash Hamming distance is
    <= ``thr`` the *later* image j is marked as a duplicate (as in the notebook).
    """
    import imagehash

    hashes: List[Tuple[str, "imagehash.ImageHash"]] = []
    for p in paths:
        try:
            hashes.append((p, imagehash.phash(Image.open(p).convert("RGB"))))
        except Exception:
            pass
    dup = set()
    if thr is not None:
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                if (hashes[i][1] - hashes[j][1]) <= thr:
                    dup.add(hashes[j][0])
    return dup


def clean_dataset(raw_dir: str, clean_dir: str, classes: Sequence[str],
                  min_side: int, blur_thr: float, phash_dup_thr: int | None) -> Tuple[int, int]:
    """Copy images that pass all filters from ``raw_dir`` to ``clean_dir``.

    Filters (applied per class): pHash near-duplicate removal, shorter side
    >= ``min_side`` and Laplacian variance >= ``blur_thr``.
    Returns ``(kept, total)``.
    """
    kept, total = 0, 0
    for cls in classes:
        srcs = sorted(glob.glob(os.path.join(raw_dir, cls, "*")))
        if not srcs:
            continue
        dup = find_near_duplicates(srcs, phash_dup_thr)
        dst = os.path.join(clean_dir, cls)
        os.makedirs(dst, exist_ok=True)
        for p in srcs:
            total += 1
            if p in dup:
                continue
            try:
                im = Image.open(p).convert("RGB")
                w, h = im.size
                if min(w, h) < min_side:
                    continue
                if laplacian_variance(im) < blur_thr:
                    continue
                im.save(os.path.join(dst, Path(p).name), quality=95)
                kept += 1
            except Exception:
                pass
    print(f"Clean kept {kept}/{total}")
    return kept, total


# --------------------------------------------------------------------------- #
# 3. Stratified train / val / test split
# --------------------------------------------------------------------------- #
def stratified_split(clean_dir: str, split_dir: str, classes: Sequence[str], seed: int,
                     test_size: float = 0.1, val_size: float = 0.2222):
    """Split cleaned images into ``split_dir/{train,val,test}/<class>/``.

    First ``test_size`` of all images go to the test set, then ``val_size`` of
    the remainder to validation (stratified by class both times).
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    rows = []
    for cls in classes:
        for p in glob.glob(os.path.join(clean_dir, cls, "*")):
            rows.append((cls, p))
    df = pd.DataFrame(rows, columns=["class", "path"])
    if len(df) == 0:
        raise RuntimeError("No images after cleaning")

    train_val, test = train_test_split(df, test_size=test_size, stratify=df["class"], random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size, stratify=train_val["class"], random_state=seed)

    def dump(split: str, sub) -> None:
        for cls, path in sub.values:
            dst = os.path.join(split_dir, split, cls, Path(path).name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(path, dst)

    dump("train", train)
    dump("val", val)
    dump("test", test)
    print("Split:", len(train), len(val), len(test))
    return train, val, test


# --------------------------------------------------------------------------- #
# 4. tf.data pipeline
# --------------------------------------------------------------------------- #
def make_datasets(split_dir: str, img_size: int, batch_size: int, seed: int):
    """Return ``(train_ds, val_ds, test_ds, class_names)``.

    Images are resized to ``img_size`` x ``img_size``; pixel scaling happens
    inside the model (``Rescaling`` layer). Only the training split is shuffled.
    """
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image_dataset_from_directory

    def load(split: str, shuffle: bool):
        return image_dataset_from_directory(
            os.path.join(split_dir, split), labels="inferred", label_mode="int",
            image_size=(img_size, img_size), batch_size=batch_size, shuffle=shuffle, seed=seed)

    train = load("train", True)
    val = load("val", False)
    test = load("test", False)
    names = train.class_names
    autotune = tf.data.AUTOTUNE
    return (train.cache().prefetch(autotune), val.cache().prefetch(autotune),
            test.cache().prefetch(autotune), names)
