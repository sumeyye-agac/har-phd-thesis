"""
Chapter 5 – Train/Evaluate Lightweight (LM) DeepConvLSTM model on Opportunity dataset.

Idempotent:
- If models_saved/tf/LM.h5 exists -> load and evaluate.
- Else -> train, save, load best and evaluate.
"""

import numpy as np
import tensorflow as tf
from src.data_opportunity import load_opportunity_splits
from src.training_tf import train_model
from src.evaluation_tf import evaluate_tf_model
from src.model_io import load_model_tf
from src.registry import exists_tf, tf_path, tf_name
from models.base_deepconvlstm import build_deepconvlstm
from src.config import (
    MODEL_SEED, DEFAULT_EPOCHS, DEFAULT_PATIENCE, DEFAULT_BATCH_SIZE,
    USE_GPU_TRAIN, USE_GPU_EVALUATE,
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
    
    if exists_tf("LM"):
        print("[Chapter 5] LM exists -> loading and evaluating:", tf_path("LM"))
        model = load_model_tf(tf_name("LM"))
        _, _, _, _, X_test, y_test = load_opportunity_splits()

        X_test = np.expand_dims(X_test, -1)
        acc, f1, prec, rec, cm = evaluate_tf_model(model, X_test, y_test, use_gpu=USE_GPU_EVALUATE)

        print("=== LM MODEL TEST RESULTS (LOADED) ===")
        print(f"Accuracy:  {acc:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print("Confusion Matrix:")
        print(cm)
        return

    print("[Chapter 5] LM not found -> training from scratch.")
    X_train, y_train, X_val, y_val, X_test, y_test = load_opportunity_splits()

    X_train = np.expand_dims(X_train, -1)
    X_val   = np.expand_dims(X_val, -1)
    X_test  = np.expand_dims(X_test, -1)

    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))

    model = build_deepconvlstm(
        variant="LM",
        input_shape=input_shape,
        num_classes=num_classes,
        seed=MODEL_SEED,
    )
    model.summary()

    history, training_info = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        save_path=tf_path("LM"),
        epochs=DEFAULT_EPOCHS,  
        batch_size=DEFAULT_BATCH_SIZE,
        patience=DEFAULT_PATIENCE,
        use_gpu=USE_GPU_TRAIN,
    )

    model = load_model_tf(tf_name("LM"))
    acc, f1, prec, rec, cm = evaluate_tf_model(model, X_test, y_test, use_gpu=USE_GPU_EVALUATE)

    print("=== LM MODEL TEST RESULTS (TRAINED) ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()