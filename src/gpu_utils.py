"""
GPU environment setup utilities.

Provides helper functions for configuring TensorFlow GPU settings
and setting up deterministic operations.
"""

import os
import tensorflow as tf
from typing import Optional


def setup_gpu_environment(
    enable_memory_growth: bool = True,
    set_seed: bool = True,
    set_floatx: bool = False,
    seed_value: Optional[int] = None
):
    """
    Setup GPU environment with memory growth and seeds.
    
    This function should be called BEFORE importing TensorFlow
    or at the very beginning of the script.
    
    Args:
        enable_memory_growth: Enable GPU memory growth (default: True)
        set_seed: Set TensorFlow random seed (default: True)
        set_floatx: Set floatx to 'float32' (default: False)
        seed_value: Random seed value (if None, uses config)
    """
    if enable_memory_growth:
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    if set_seed and seed_value is not None:
        tf.random.set_seed(seed_value)
    
    if set_floatx:
        tf.keras.backend.set_floatx('float32')


def configure_gpu_memory_growth():
    """
    Configure GPU memory growth after TensorFlow is imported.
    
    This should be called after TensorFlow import, typically in main().
    """
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"Warning: Could not set GPU memory growth: {e}")

