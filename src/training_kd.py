"""
Training utility for Knowledge Distillation (KD).
Wraps Distiller into a clean, simple API.
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.losses import SparseCategoricalCrossentropy, KLDivergence
from tensorflow.keras.metrics import SparseCategoricalAccuracy

from src.distillation import Distiller
from src.config import (
    DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE,
    KD_DEFAULT_EPOCHS, KD_DEFAULT_PATIENCE, KD_DEFAULT_VERBOSE,
    USE_GPU_TRAIN_KD,
)


class StudentModelCheckpoint(Callback):
    """Custom callback to save student model (works with subclassed Distiller)."""
    def __init__(self, filepath, monitor='val_accuracy', save_best_only=True, mode='max'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if self.mode == 'min':
            is_best = current < self.best
        else:
            is_best = current > self.best
            
        if is_best:
            self.best = current
            if self.save_best_only:
                self.model.student.save(self.filepath)
                print(f"\n[Saved best student model] {self.filepath} (val_accuracy={current:.4f})")


class TrainingInfoCallback(Callback):
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


def train_kd(
    student,
    teacher,
    architecture,
    X_train,
    y_train,
    X_val,
    y_val,
    alpha=0.1,
    temperature=1,
    beta=0.0,
    attention_list=None,
    attention_layer=None,
    batch_size=DEFAULT_BATCH_SIZE,
    epochs=KD_DEFAULT_EPOCHS,
    patience=KD_DEFAULT_PATIENCE,
    verbose=KD_DEFAULT_VERBOSE,
    save_path=None,
    use_gpu=None,
):
    """Train a knowledge-distilled student model.
    
    Args:
        use_gpu: If None, uses config.USE_GPU_TRAIN_KD. If True/False, overrides config.
    
    Returns:
        student_model: Trained student model
        training_info: Dict with 'total_epochs', 'best_epoch', 'learning_rate'
    """
    # Use config if not specified
    if use_gpu is None:
        use_gpu = USE_GPU_TRAIN_KD
    
    # Determine device based on use_gpu flag
    device = '/GPU:0' if use_gpu else '/CPU:0'
    
    # Clone student model to avoid issues with repeated training
    with tf.device(device):
        student_clone = tf.keras.models.clone_model(student)
        student_clone.set_weights(student.get_weights())
    
    # Create training info callback
    training_info_cb = TrainingInfoCallback()
    
    # Build and compile distiller (with explicit device context)
    with tf.device(device):
        distiller = Distiller(student=student_clone, teacher=teacher, architecture=architecture)
        distiller.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE, beta_1=0.9),
            metrics=[SparseCategoricalAccuracy(name="accuracy")],
            student_loss_fn=SparseCategoricalCrossentropy(from_logits=True),
            distillation_loss_fn=KLDivergence(),
            alpha=alpha,
            temperature=temperature,
            beta=beta,
            attention_list=attention_list or [],
            attention_layer=attention_layer,
        )

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True, verbose=2),
        training_info_cb,
    ]
    if save_path:
        callbacks.append(StudentModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True))

    # Training should also happen on the specified device
    with tf.device(device):
        distiller.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
        )

    if save_path and not os.path.exists(save_path):
        distiller.student.save(save_path)
    
    # Prepare training info
    training_info = {
        'total_epochs': training_info_cb.total_epochs,
        'best_epoch': training_info_cb.best_epoch,
        'learning_rate': training_info_cb.learning_rate or DEFAULT_LEARNING_RATE,
    }

    return distiller.student, training_info