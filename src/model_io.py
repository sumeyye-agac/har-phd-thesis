"""
Model saving and loading helpers for TF and TFLite models.
"""

import os
import tensorflow as tf
from tensorflow.keras.models import load_model

from models.attention_layers import ChannelAttention, SpatialAttention
from src.registry import TF_DIR, TFLITE_DIR


# -------------------------------------------------------------------------
# TF Models
# -------------------------------------------------------------------------

def save_model_tf(model, name):
    """
    Save a TensorFlow model into models_saved/tf/.
    """
    path = os.path.join(TF_DIR, name)
    model.save(path)
    print(f"[Saved TF model] {path}")
    return path


def load_model_tf(name):
    """
    Load a TensorFlow model from models_saved/tf/.
    Automatically handles attention layers.
    """
    path = os.path.join(TF_DIR, name)

    custom_objects = {
        "ChannelAttention": ChannelAttention,
        "SpatialAttention": SpatialAttention,
    }

    model = load_model(path, compile=False, custom_objects=custom_objects)
    print(f"[Loaded TF model] {path}")
    return model


# -------------------------------------------------------------------------
# TFLite Models
# -------------------------------------------------------------------------

def save_model_tflite(model_bytes, name):
    """
    Save a TFLite model (bytes) into models_saved/tflite/.
    """
    path = os.path.join(TFLITE_DIR, name)
    with open(path, "wb") as f:
        f.write(model_bytes)
    print(f"[Saved TFLite model] {path}")
    return path


def load_interpreter(name):
    """
    Load a TFLite interpreter from models_saved/tflite/.
    """
    path = os.path.join(TFLITE_DIR, name)
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    print(f"[Loaded TFLite interpreter] {path}")
    return interpreter
