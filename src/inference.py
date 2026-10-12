"""Single-image inference with calibrated Top-k probabilities (notebook demo cell)."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image


def predict_topk(model, image_path: str, class_names: Sequence[str], logit_adjustment: np.ndarray,
                 temperature: float, img_size: int = 224, k: int = 3) -> List[Tuple[str, float]]:
    """Resize to ``img_size``, run the model and return the Top-k classes with
    probabilities ``softmax((logits + logit_adjustment) / T)``."""
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    arr = np.array(img).astype(np.float32)[None]
    logits, _ = model.predict(arr, verbose=0)
    la = (logits + logit_adjustment) / temperature
    probs = tf.nn.softmax(la, axis=1).numpy()[0]
    top = probs.argsort()[-k:][::-1]
    return [(class_names[i], float(probs[i])) for i in top]
