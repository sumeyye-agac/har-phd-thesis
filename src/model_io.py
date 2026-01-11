"""
Model saving and loading helpers for TF and TFLite models.
"""

import os
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model

from models.attention_layers import ChannelAttention, SpatialAttention
from src.registry import TF_DIR, TFLITE_DIR

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# TF Models
# -------------------------------------------------------------------------

def save_model_tf(model, name):
    """
    Save a TensorFlow model into models_saved/tf/.
    """
    path = os.path.join(TF_DIR, name)
    model.save(path)
    logger.info(f"Saved TF model: {path}")
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
    logger.info(f"Loaded TF model: {path}")
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
    logger.info(f"Saved TFLite model: {path}")
    return path


def load_interpreter(name):
    """
    Load a TFLite interpreter from models_saved/tflite/.
    Attempts to use Flex delegate for models with LSTM/Select TF ops.
    Falls back to regular interpreter if Flex delegate is not available.
    """
    path = os.path.join(TFLITE_DIR, name)
    
    # Try with Flex delegate first (for LSTM models)
    try:
        flex_delegate = tf.lite.experimental.load_delegate('libtensorflowlite_flex.so')
        interpreter = tf.lite.Interpreter(
            model_path=path,
            experimental_delegates=[flex_delegate]
        )
        interpreter.allocate_tensors()
        logger.info(f"Loaded TFLite interpreter with Flex delegate: {path}")
        return interpreter
    except (ValueError, OSError, RuntimeError):
        # Flex delegate not available, try regular interpreter
        pass
    
    # Try regular interpreter
    try:
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        logger.info(f"Loaded TFLite interpreter: {path}")
        return interpreter
    except RuntimeError as e:
        if "Select TensorFlow op" in str(e) or "Flex" in str(e):
            # Model requires Flex ops but delegate not available
            raise RuntimeError(
                f"Model requires Flex ops (LSTM layers) but Flex delegate is not available. "
                f"TFLite models with LSTM cannot be evaluated without Flex delegate. "
                f"This is expected on macOS. Models are still created successfully."
            ) from e
        raise
