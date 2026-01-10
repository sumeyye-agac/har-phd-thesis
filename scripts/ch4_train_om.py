"""
Chapter 4 – Train/Evaluate Original DeepConvLSTM (OM) model on Opportunity dataset.

Idempotent behavior:
- If models_saved/tf/OM.h5 exists -> load and evaluate (no training).
- Else -> train, save, then load best and evaluate.
"""

import numpy as np
import tensorflow as tf
from models.base_deepconvlstm import build_deepconvlstm_om
from src.data_opportunity import load_opportunity_splits
from src.training_tf import train_model
from src.evaluation_tf import evaluate_tf_model
from src.model_io import load_model_tf
from src.registry import exists_tf, tf_path, tf_name
from src.config import (
    DEFAULT_EPOCHS, DEFAULT_PATIENCE, DEFAULT_BATCH_SIZE,
    USE_GPU_TRAIN, USE_GPU_EVALUATE
)

def main():
    # Enable GPU memory growth
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled: {len(gpus)} GPU(s)")
        else:
            print("No GPU found, using CPU")
    except Exception as e:
        print(f"Could not set GPU memory growth: {e}")
    
    # Check available devices and active device
    print(f"TensorFlow devices: {tf.config.list_physical_devices()}")
    test_tensor = tf.constant([1.0])
    print(f"Active device: {test_tensor.device}")
    print()
    
    # If already trained, skip training
    if exists_tf("OM"):
        print("[Chapter 4] OM exists -> loading and evaluating:", tf_path("OM"))
        model = load_model_tf(tf_name("OM"))
        X_train, y_train, X_val, y_val, X_test, y_test = load_opportunity_splits()

        X_test = np.expand_dims(X_test, -1)
        acc, f1, prec, rec, cm = evaluate_tf_model(model, X_test, y_test, use_gpu=USE_GPU_EVALUATE)

        print("=== OM MODEL TEST RESULTS (LOADED) ===")
        print(f"Accuracy:  {acc:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print("Confusion Matrix:")
        print(cm)
        return

    # Otherwise train from scratch
    print("[Chapter 4] OM not found -> training from scratch.")
    X_train, y_train, X_val, y_val, X_test, y_test = load_opportunity_splits()

    # Add channel dimension
    X_train = np.expand_dims(X_train, -1)
    X_val   = np.expand_dims(X_val, -1)
    X_test  = np.expand_dims(X_test, -1)

    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))

    model = build_deepconvlstm_om(input_shape, num_classes)
    model.summary()

    history, training_info = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        save_path=tf_path("OM"),
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        patience=DEFAULT_PATIENCE,
        use_gpu=USE_GPU_TRAIN,
    )

    # Load best saved model and evaluate
    model = load_model_tf(tf_name("OM"))
    acc, f1, prec, rec, cm = evaluate_tf_model(model, X_test, y_test, use_gpu=USE_GPU_EVALUATE)

    print("=== OM MODEL TEST RESULTS (TRAINED) ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()