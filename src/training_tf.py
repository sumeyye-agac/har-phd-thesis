"""
Training and evaluation utilities for TensorFlow models.
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

from src.config import (
    DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE, 
    DEFAULT_EPOCHS, DEFAULT_PATIENCE, DEFAULT_VERBOSE
)


class TrainingInfoCallback(tf.keras.callbacks.Callback):
    """Track training information: best epoch, total epochs, learning rate."""
    def __init__(self):
        super().__init__()
        self.best_epoch = 0
        self.total_epochs = 0
        self.best_val_acc = float('-inf')
        self.learning_rate = None
    
    def on_train_begin(self, logs=None):
        # Get learning rate from optimizer
        if hasattr(self.model.optimizer, 'learning_rate'):
            lr = self.model.optimizer.learning_rate
            if hasattr(lr, 'numpy'):
                self.learning_rate = float(lr.numpy())
            else:
                self.learning_rate = float(lr)
    
    def on_epoch_end(self, epoch, logs=None):
        self.total_epochs = epoch + 1
        logs = logs or {}
        val_acc = logs.get('val_accuracy')
        if val_acc is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch + 1  # 1-indexed


def train_model(model, X_train, y_train, X_val, y_val, save_path=None,
                batch_size=DEFAULT_BATCH_SIZE, epochs=DEFAULT_EPOCHS, 
                patience=DEFAULT_PATIENCE, verbose=DEFAULT_VERBOSE,
                use_gpu=True):
    """
    Train a TF model with early stopping.
    
    Args:
        use_gpu: If True, use GPU context manager (tf.device('/GPU:0')).
                 If False, use CPU or default device.
    
    Returns:
        history: Training history
        training_info: Dict with 'total_epochs', 'best_epoch', 'learning_rate'
    """
    # Clone model to avoid issues with repeated training (Metal GPU compatibility)
    model_clone = tf.keras.models.clone_model(model)
    model_clone.set_weights(model.get_weights())
    
    # Use GPU context manager if requested
    if use_gpu:
        device_context = tf.device('/GPU:0')
    else:
        device_context = tf.device('/CPU:0')
    
    # Create training info callback
    training_info_cb = TrainingInfoCallback()
    
    with device_context:
        model_clone.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=DEFAULT_LEARNING_RATE, beta_1=0.9),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
        )

        callbacks = [
            EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True, verbose=2),
            training_info_cb,  # Add training info callback
        ]

        if save_path:
            callbacks.append(ModelCheckpoint(save_path, save_best_only=True, monitor="val_accuracy"))

        history = model_clone.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        if save_path and not os.path.exists(save_path):
            model_clone.save(save_path)
    
    # Update original model weights
    model.set_weights(model_clone.get_weights())
    
    # Prepare training info
    training_info = {
        'total_epochs': training_info_cb.total_epochs,
        'best_epoch': training_info_cb.best_epoch,
        'learning_rate': training_info_cb.learning_rate or DEFAULT_LEARNING_RATE,
    }
    
    return history, training_info