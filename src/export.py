"""Export the trained model as Keras + float16 TFLite, with labels and calibration values."""
from __future__ import annotations

import json
import os
import zipfile
from typing import Sequence

import numpy as np
import tensorflow as tf


def export_artifacts(model: tf.keras.Model, class_names: Sequence[str], export_dir: str,
                     logit_adjustment: np.ndarray | None = None,
                     temperature: float | None = None) -> str:
    """Write ``mammals_classifier.keras``, ``mammals_classifier.tflite`` (float16
    post-training quantisation), ``labels.txt`` and ``release_artifacts.zip``.

    ``calibration.json`` (logit adjustment + temperature) is an addition of this
    refactor so that :mod:`predict` can reproduce the notebook's calibrated
    demo probabilities without re-running calibration.
    """
    os.makedirs(export_dir, exist_ok=True)
    keras_path = os.path.join(export_dir, "mammals_classifier.keras")
    tflite_path = os.path.join(export_dir, "mammals_classifier.tflite")
    labels_path = os.path.join(export_dir, "labels.txt")
    calib_path = os.path.join(export_dir, "calibration.json")
    zip_path = os.path.join(export_dir, "release_artifacts.zip")

    model.save(keras_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    with open(tflite_path, "wb") as f:
        f.write(converter.convert())
    with open(labels_path, "w") as f:
        for c in class_names:
            f.write(c + "\n")

    files = [keras_path, tflite_path, labels_path]
    if logit_adjustment is not None and temperature is not None:
        with open(calib_path, "w") as f:
            json.dump({"class_names": list(class_names),
                       "logit_adjustment": [float(v) for v in logit_adjustment],
                       "temperature": float(temperature)}, f, indent=2)
        files.append(calib_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=os.path.basename(p))
    print("Exported", zip_path)
    return zip_path
