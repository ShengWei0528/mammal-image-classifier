"""Nine-class mammal image classifier (MobileNetV2 + prototype center loss + calibration).

Refactored from ``notebooks/mammals_full_pipeline.ipynb``.
"""

__all__ = ["config", "data", "model", "training", "calibration", "evaluation",
           "embedding", "export", "inference"]
