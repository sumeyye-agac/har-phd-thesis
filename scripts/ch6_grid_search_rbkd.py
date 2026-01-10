"""
Chapter 6 — LM RB-KD Grid Search

Grid search over Knowledge Distillation hyperparameters:
- Temperature (τ): [1, 2, 4, 6, 8, 10, 15] (7 values)
- Alpha (α): [0.1, 0.2, ..., 0.9] (9 values)

Total: 7 × 9 = 63 models

Teacher: OM (Original Model)
Student: LM (Lightweight Model)

All models are saved and results are written to results/grid_search_rbkd.csv
"""

import os
import numpy as np

# Set GPU memory growth BEFORE importing TensorFlow
from src.gpu_utils import setup_gpu_environment
setup_gpu_environment(enable_memory_growth=True)

import tensorflow as tf
from itertools import product

from src.data_opportunity import load_opportunity_splits
from src.grid_search_utils import set_seeds, save_grid_search_result, get_active_device
from src.model_naming import generate_kd_model_name
from src.model_io import load_model_tf
from src.training_kd import train_kd
from src.evaluation_tf import evaluate_tf_model
from models.base_deepconvlstm import build_deepconvlstm
from src.registry import TF_DIR, exists_tf, tf_name
from src.config import (
    MODEL_SEED, TRAINING_TF_SEED, TRAINING_NP_SEED,
    KD_DEFAULT_EPOCHS, KD_DEFAULT_PATIENCE, DEFAULT_BATCH_SIZE,
    KD_GRID_TEMPERATURES, KD_GRID_ALPHAS,
    KD_BASELINE_BETA, DEFAULT_VERBOSE,
    USE_GPU_TRAIN_KD, USE_GPU_EVALUATE,
)
from src.utils_resources import get_model_size as get_model_size_kb
from src.cli_parser import parse_args
from src.memory_utils import cleanup_memory, cleanup_on_error, periodic_cleanup


# Parser configuration
PARSER_CONFIG = {
    'attention_params': [],
    'list_params': ['temperatures', 'alphas'],
    'valid_values': {
        'temperatures': [1, 2, 4, 6, 8, 10, 15],
        'alphas': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    },
    'help_text': """
LM RB-KD Grid Search - Usage

Hyperparameter Lists (format: value1,value2,...):
  temperatures=1,2,4,6,8,10,15  Temperature values (default: 1,2,4,6,8,10,15)
                                  Valid values: 1, 2, 4, 6, 8, 10, 15
  
  alphas=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9  Alpha values (default: 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
                                  Valid values: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

Examples:
  # Run with all defaults (7×9=63 models)
  python ch6_grid_search_rbkd.py
  
  # Run with only temperature 1, 2, 4 and alpha 0.1, 0.2
  python ch6_grid_search_rbkd.py temperatures=1,2,4 alphas=0.1,0.2
  
  # Run with only temperature 1
  python ch6_grid_search_rbkd.py temperatures=1 alphas=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9

Note: All parameters are case-insensitive. Invalid parameters or values will stop execution.
""",
    'script_name': 'ch6_grid_search_rbkd.py'
}


def main():
    # Parse command line arguments
    args = parse_args(PARSER_CONFIG)
    
    # Get hyperparameter lists (default: from config)
    temperatures = args.get('temperatures', KD_GRID_TEMPERATURES)
    alphas = args.get('alphas', KD_GRID_ALPHAS)
    
    print("=== Chapter 6: LM RB-KD Grid Search ===")
    
    total_experiments = len(temperatures) * len(alphas)
    
    print(f"Total experiments: {total_experiments} ({len(temperatures)} temperatures × {len(alphas)} alphas)")
    print(f"Configuration: temperatures={temperatures}, alphas={alphas}")
    print()
    
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
    
    # Check teacher model
    if not exists_tf("OM"):
        raise FileNotFoundError(
            "Teacher model OM not found. Run Chapter 4 first to create: models_saved/tf/OM.h5"
        )
    
    # Load data
    print("Loading Opportunity dataset splits...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_opportunity_splits()
    
    # Add channel dimension
    X_train = np.expand_dims(X_train, -1)
    X_val = np.expand_dims(X_val, -1)
    X_test = np.expand_dims(X_test, -1)
    
    num_classes = len(np.unique(y_train))
    input_shape = X_train.shape[1:]
    
    print(f"Data shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"Number of classes: {num_classes}")
    print()
    
    # Load teacher (loaded once, reused for all experiments)
    print("Loading teacher model (OM)...")
    teacher = load_model_tf(tf_name("OM"))
    print("✓ Teacher model loaded")
    print()
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    csv_path = "results/grid_search_rbkd.csv"
    
    print(f"Starting grid search: {total_experiments} experiments")
    print(f"Results will be saved to: {csv_path}")
    print()
    
    experiment_count = 0
    
    for temperature, alpha in product(temperatures, alphas):
        experiment_count += 1
        
        model_name = generate_kd_model_name(
            variant="LM",
            kd_type="RB-KD",
            temperature=temperature,
            alpha=alpha,
        )
        
        print(f"  [{experiment_count}/{total_experiments}] {model_name}...", end=" ", flush=True)
        
        model_path = os.path.join(TF_DIR, f"{model_name}.h5")
        
        # Check if already trained
        if os.path.exists(model_path):
            print("SKIPPED (already exists)")
            # Still evaluate and save to CSV
            try:
                # Load best model and evaluate
                student_model = load_model_tf(f"{model_name}.h5")

                # Evaluate on all sets
                train_acc, train_f1, train_prec, train_rec, train_cm = evaluate_tf_model(student_model, X_train, y_train, use_gpu=USE_GPU_EVALUATE)
                val_acc, val_f1, val_prec, val_rec, val_cm = evaluate_tf_model(student_model, X_val, y_val, use_gpu=USE_GPU_EVALUATE)
                test_acc, test_f1, test_prec, test_rec, test_cm = evaluate_tf_model(student_model, X_test, y_test, use_gpu=USE_GPU_EVALUATE)

                # Save to CSV (training_info not available for skipped models)
                save_grid_search_result(
                    csv_path=csv_path,
                    model_name=model_name,
                    hyperparams={
                        'temperature': temperature,
                        'alpha': alpha,
                    },
                    val_accuracy=val_acc,
                    # Val metrics
                    val_f1=val_f1,
                    val_precision=val_prec,
                    val_recall=val_rec,
                    val_confusion_matrix=val_cm,
                    # Test metrics
                    test_accuracy=test_acc,
                    test_f1=test_f1,
                    test_precision=test_prec,
                    test_recall=test_rec,
                    test_confusion_matrix=test_cm,
                    # Train metrics
                    train_accuracy=train_acc,
                    train_f1=train_f1,
                    train_precision=train_prec,
                    train_recall=train_rec,
                    train_confusion_matrix=train_cm,
                    # Model info
                    model_size_kb=get_model_size_kb(model_path),
                    parameter_count=student_model.count_params(),
                    device=get_active_device(),
                    # Training info (not available for skipped models)
                    total_epochs=None,
                    best_epoch=None,
                    learning_rate=None,
                    training_time_seconds=None,
                    final_train_loss=None,
                    final_val_loss=None,
                    # Training hyperparameters
                    batch_size=DEFAULT_BATCH_SIZE,
                    max_epochs=KD_DEFAULT_EPOCHS,
                    patience=KD_DEFAULT_PATIENCE,
                )
                
                # Clear memory
                del student_model
                cleanup_memory()
                
                continue
            except (OSError, IOError, ValueError) as e:
                print(f"⚠ Corrupted model file detected, will retrain: {e}")
                os.remove(model_path)
                # Fall through to training below
        
        # Set seeds
        set_seeds()
        tf.keras.backend.set_floatx("float32")
        
        # Build student
        student = build_deepconvlstm(
            variant="LM",
            input_shape=input_shape,
            num_classes=num_classes,
            seed=MODEL_SEED,
        )
        
        # Train KD
        try:
            # Additional aggressive cleanup BEFORE training
            cleanup_memory()
            
            # Small delay to let GPU memory settle (optional but might help)
            import time
            time.sleep(0.1)

            student_model, training_info = train_kd(
                student=student,
                teacher=teacher,
                architecture="deepconvlstm",
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                alpha=alpha,
                temperature=temperature,
                beta=KD_BASELINE_BETA,
                attention_list=[],
                attention_layer=None,
                batch_size=DEFAULT_BATCH_SIZE,
                epochs=KD_DEFAULT_EPOCHS,
                patience=KD_DEFAULT_PATIENCE,
                verbose=DEFAULT_VERBOSE,
                save_path=model_path,
                use_gpu=None,  # Use config.USE_GPU_TRAIN_KD
            )
            
            # Load best model and evaluate
            student_model = load_model_tf(f"{model_name}.h5")
            
            # Evaluate on all sets
            train_acc, train_f1, train_prec, train_rec, train_cm = evaluate_tf_model(student_model, X_train, y_train, use_gpu=USE_GPU_EVALUATE)
            val_acc, val_f1, val_prec, val_rec, val_cm = evaluate_tf_model(student_model, X_val, y_val, use_gpu=USE_GPU_EVALUATE)
            test_acc, test_f1, test_prec, test_rec, test_cm = evaluate_tf_model(student_model, X_test, y_test, use_gpu=USE_GPU_EVALUATE)
            
            # Save to CSV
            save_grid_search_result(
                csv_path=csv_path,
                model_name=model_name,
                hyperparams={
                    'temperature': temperature,
                    'alpha': alpha,
                },
                val_accuracy=val_acc,
                # Val metrics
                val_f1=val_f1,
                val_precision=val_prec,
                val_recall=val_rec,
                val_confusion_matrix=val_cm,
                # Test metrics
                test_accuracy=test_acc,
                test_f1=test_f1,
                test_precision=test_prec,
                test_recall=test_rec,
                test_confusion_matrix=test_cm,
                # Train metrics
                train_accuracy=train_acc,
                train_f1=train_f1,
                train_precision=train_prec,
                train_recall=train_rec,
                train_confusion_matrix=train_cm,
                # Model info
                model_size_kb=get_model_size_kb(model_path),
                parameter_count=student_model.count_params(),
                device=get_active_device(),
                # Training info
                total_epochs=training_info.get('total_epochs'),
                best_epoch=training_info.get('best_epoch'),
                learning_rate=training_info.get('learning_rate'),
                training_time_seconds=training_info.get('training_time_seconds'),
                final_train_loss=training_info.get('final_train_loss'),
                final_val_loss=training_info.get('final_val_loss'),
                # Training hyperparameters
                batch_size=training_info.get('batch_size') or DEFAULT_BATCH_SIZE,
                max_epochs=training_info.get('max_epochs') or KD_DEFAULT_EPOCHS,
                patience=training_info.get('patience') or KD_DEFAULT_PATIENCE,
            )
            
            print(f"✓ Val Acc: {val_acc:.4f}")
            
            # Clear memory after each experiment (keep teacher in memory)
            del student
            del student_model
            cleanup_memory()
            
            # Periodic cleanup every 20 experiments
            if experiment_count % 20 == 0:
                print(f"\n  [Memory cleanup] Clearing session...", end=" ", flush=True)
                cleanup_memory()
                print("Done")
                
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            # Clear memory even on error (keep teacher in memory)
            if 'student' in locals():
                del student
            if 'student_model' in locals():
                del student_model
            cleanup_memory()
            continue
    
    # Clear teacher at the end
    del teacher
    tf.keras.backend.clear_session()
    gc.collect()
    
    print(f"\n=== Grid Search Complete ===")
    print(f"Results saved to: {csv_path}")
    print(f"Models saved to: {TF_DIR}")


if __name__ == "__main__":
    main()