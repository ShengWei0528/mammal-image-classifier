"""MobileNetV2 backbone + 128-d embedding + linear classifier, and class prototypes."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers


def build_model(num_classes: int, img_size: int = 224, embed_dim: int = 128) -> Model:
    """Two-output model: ``[logits, embedding]``.

    * ImageNet-pre-trained MobileNetV2 without top, frozen (``trainable=False``)
    * inputs in [0, 255] are mapped to [-1, 1] by a ``Rescaling`` layer
    * GlobalAveragePooling2D -> Dense(embed_dim) -> BatchNormalization = embedding
    * Dense(num_classes) on the embedding = logits (no softmax)
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    embedding = layers.Dense(embed_dim, activation=None, name="embedding")(x)
    embedding = layers.BatchNormalization()(embedding)
    logits = layers.Dense(num_classes, activation=None, name="logits")(embedding)
    return Model(inputs, [logits, embedding])


def make_prototypes(num_classes: int, embed_dim: int = 128) -> tf.Variable:
    """Non-trainable ``(K, embed_dim)`` matrix of class centres, initialised to zero."""
    return tf.Variable(tf.zeros((num_classes, embed_dim), dtype=tf.float32), trainable=False)
