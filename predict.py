"""Calibrated Top-3 prediction for one or more images using the exported model.

    python predict.py path/to/image.jpg [more.jpg ...] --export-dir work/export
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+")
    p.add_argument("--export-dir", default=os.path.join("work", "export"))
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    import tensorflow as tf

    from src.inference import predict_topk

    model = tf.keras.models.load_model(os.path.join(args.export_dir, "mammals_classifier.keras"))
    with open(os.path.join(args.export_dir, "calibration.json")) as f:
        calib = json.load(f)
    adj = np.array(calib["logit_adjustment"], dtype=np.float32)

    for path in args.images:
        print(f"\nFile: {path}")
        top = predict_topk(model, path, calib["class_names"], adj, calib["temperature"], k=args.top_k)
        for i, (cls, prob) in enumerate(top, 1):
            print(f"Top{i}: {cls} - {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
