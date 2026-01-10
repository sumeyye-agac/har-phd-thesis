"""
Evaluation utilities for TensorFlow models.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from src.config import USE_GPU_EVALUATE


def evaluate_tf_model(model, X, y, use_gpu=None):
    """
    Evaluate a TF model.
    
    Args:
        model: TensorFlow model
        X: Input data
        y: True labels
        use_gpu: If None, uses config.USE_GPU_EVALUATE. If True/False, overrides config.
    
    Returns: accuracy, f1, precision, recall, confusion_matrix
    """
    if use_gpu is None:
        use_gpu = USE_GPU_EVALUATE
    
    # Use GPU context manager if requested
    if use_gpu:
        device_context = tf.device('/GPU:0')
    else:
        device_context = tf.device('/CPU:0')
    
    with device_context:
        preds = []
        for sample in X:
            sample = np.expand_dims(sample, axis=0)
            logits = model.predict(sample, verbose=0)
            preds.append(np.argmax(logits))

        preds = np.array(preds)

        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="macro")
        prec = precision_score(y, preds, average="macro")
        rec = recall_score(y, preds, average="macro")
        cm = confusion_matrix(y, preds)

        return acc, f1, prec, rec, cm