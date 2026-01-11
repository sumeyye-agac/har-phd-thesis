"""
Memory management utilities.

Provides helper functions for aggressive memory cleanup
during grid search experiments.
"""

import gc
import logging
import tensorflow as tf
from typing import Optional

logger = logging.getLogger(__name__)


def cleanup_memory(*objects_to_delete):
    """
    Aggressive memory cleanup.
    
    Deletes specified objects, clears TensorFlow session,
    and runs garbage collection.
    
    Args:
        *objects_to_delete: Variable number of objects to delete
    """
    for obj in objects_to_delete:
        if obj is not None:
            del obj
    tf.keras.backend.clear_session()
    gc.collect()


def periodic_cleanup(experiment_count: int, interval: int = 20, verbose: bool = False):
    """
    Periodic cleanup helper for grid search loops.
    
    Args:
        experiment_count: Current experiment number
        interval: Cleanup interval (default: 20)
        verbose: Print cleanup message if True
    
    Returns:
        True if cleanup was performed, False otherwise
    """
    if experiment_count % interval == 0:
        if verbose:
            logger.debug("Memory cleanup: Clearing session...")
        tf.keras.backend.clear_session()
        gc.collect()
        if verbose:
            logger.debug("Memory cleanup: Done")
        return True
    return False


def cleanup_on_error(*objects_to_delete):
    """
    Cleanup memory on error, setting objects to None.
    
    Args:
        *objects_to_delete: Variable number of objects to clean up
    """
    for obj in objects_to_delete:
        if obj is not None:
            del obj
    tf.keras.backend.clear_session()
    gc.collect()

