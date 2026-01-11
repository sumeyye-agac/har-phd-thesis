"""
GPU environment setup utilities.

Provides helper functions for configuring TensorFlow GPU settings
and setting up deterministic operations.
"""

import os
import tensorflow as tf
from typing import Optional
from logging import Logger


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
        # This function doesn't have logger, so keep print for backward compatibility
        # But prefer using setup_gpu_and_log_device() which has logger support
        import warnings
        warnings.warn(f"Could not set GPU memory growth: {e}")


def setup_gpu_and_log_device(logger: Optional[Logger] = None):
    """
    Setup GPU memory growth and log device information.
    
    This is a convenience function that combines GPU setup and device logging.
    Should be called after TensorFlow import, typically at the start of main().
    
    Args:
        logger: Logger instance (if None, uses print statements)
    
    Returns:
        str: Active device name (e.g., '/GPU:0' or '/CPU:0')
    """
    # Setup GPU memory growth
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if logger:
                logger.info(f"GPU memory growth enabled: {len(gpus)} GPU(s)")
        else:
            if logger:
                logger.warning("No GPU found, using CPU")
    except Exception as e:
        if logger:
            logger.error(f"Could not set GPU memory growth: {e}")
    
    # Log device information
    devices = tf.config.list_physical_devices()
    if logger:
        logger.debug(f"TensorFlow devices: {devices}")
    
    test_tensor = tf.constant([1.0])
    active_device = test_tensor.device
    if logger:
        logger.info(f"Active device: {active_device}")
    
    return active_device

