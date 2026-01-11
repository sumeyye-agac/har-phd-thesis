"""
Training utilities for baseline model training scripts.

Provides helper functions to reduce code duplication across training scripts.
"""

import json
import os
from typing import Callable, Optional, Tuple
import numpy as np
import tensorflow as tf
from logging import Logger

from src.registry import exists_tf, tf_path, tf_name
from src.model_io import load_model_tf
from src.training_tf import train_model
from src.evaluation_tf import evaluate_tf_model
from src.data_opportunity import load_and_prepare_opportunity_data
from src.config import (
    DEFAULT_EPOCHS, DEFAULT_PATIENCE, DEFAULT_BATCH_SIZE,
    USE_GPU_TRAIN, USE_GPU_EVALUATE
)


def train_or_evaluate_baseline_model(
    variant: str,
    logger: Logger,
    build_model_func: Callable,
    build_model_kwargs: dict,
    chapter_title: str,
) -> Optional[tf.keras.Model]:
    """
    Generic function to train or evaluate baseline models (OM, LM, MM).
    
    This function handles the common pattern:
    1. Check if model exists -> load and evaluate
    2. If not exists -> train, save, load best, and evaluate
    
    Args:
        variant: Model variant name ("OM", "LM", "MM")
        logger: Logger instance
        build_model_func: Function to build the model (e.g., build_deepconvlstm_om, build_deepconvlstm)
        build_model_kwargs: Keyword arguments to pass to build_model_func (input_shape and num_classes will be added automatically)
        chapter_title: Title for logging (e.g., "Chapter 4: Training Original Model (OM)")
    
    Returns:
        Trained/evaluated model, or None if evaluation-only
    """
    logger.info(f"=== {chapter_title} ===")
    
    # If already trained, skip training
    if exists_tf(variant):
        logger.info(f"{variant} exists -> loading and evaluating: {tf_path(variant)}")
        model = load_model_tf(tf_name(variant))
        _, _, _, _, X_test, y_test, _, _ = load_and_prepare_opportunity_data(logger)

        acc, f1, prec, rec, cm = evaluate_tf_model(model, X_test, y_test, use_gpu=USE_GPU_EVALUATE)

        logger.info(f"=== {variant} MODEL TEST RESULTS (LOADED) ===")
        logger.info(f"Accuracy:  {acc:.4f}")
        logger.info(f"F1 Score:  {f1:.4f}")
        logger.info(f"Precision: {prec:.4f}")
        logger.info(f"Recall:    {rec:.4f}")
        logger.debug(f"Confusion Matrix:\n{cm}")
        return model

    # Otherwise train from scratch
    logger.info(f"{variant} not found -> training from scratch.")
    X_train, y_train, X_val, y_val, X_test, y_test, input_shape, num_classes = load_and_prepare_opportunity_data(logger)
    logger.info(f"Input shape: {input_shape}, Number of classes: {num_classes}")

    # Build model - add input_shape and num_classes to kwargs
    model_kwargs = {**build_model_kwargs, 'input_shape': input_shape, 'num_classes': num_classes}
    model = build_model_func(**model_kwargs)
    logger.debug("Model architecture:")
    model.summary()

    # Train
    logger.info(f"Starting training - Epochs: {DEFAULT_EPOCHS}, Batch size: {DEFAULT_BATCH_SIZE}, Patience: {DEFAULT_PATIENCE}")
    history, training_info = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        save_path=tf_path(variant),
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        patience=DEFAULT_PATIENCE,
        use_gpu=USE_GPU_TRAIN,
    )
    logger.info(f"Training completed - Best epoch: {training_info.get('best_epoch')}, Total epochs: {training_info.get('total_epochs')}")

    # Save training info to JSON file (for later retrieval in evaluation scripts)
    training_info_path = tf_path(variant).replace('.h5', '_training_info.json')
    try:
        with open(training_info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        logger.debug(f"Training info saved to: {training_info_path}")
    except Exception as e:
        logger.warning(f"Could not save training info to {training_info_path}: {e}")

    # Load best saved model and evaluate
    logger.info("Loading best saved model for evaluation")
    model = load_model_tf(tf_name(variant))
    acc, f1, prec, rec, cm = evaluate_tf_model(model, X_test, y_test, use_gpu=USE_GPU_EVALUATE)

    logger.info(f"=== {variant} MODEL TEST RESULTS (TRAINED) ===")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.debug(f"Confusion Matrix:\n{cm}")
    
    return model
