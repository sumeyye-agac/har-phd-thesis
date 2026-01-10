"""
Grid search utilities for hyperparameter tuning.

Provides common functions for running grid searches, training models,
and tracking results.
"""

import os
import csv
import time
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any, Optional, Callable
from datetime import datetime

from src.config import (
    MODEL_SEED, TRAINING_TF_SEED, TRAINING_NP_SEED,
    DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_PATIENCE,
    KD_DEFAULT_EPOCHS, KD_DEFAULT_PATIENCE,
    USE_GPU_TRAIN,
)


# Standard CSV fieldnames for all results files
STANDARD_CSV_FIELDNAMES = [
    # Basic info
    'model_name',
    'timestamp',
    'device',  # GPU/CPU
    'variant',  # OM, LM, MM
    'has_attention',  # True/False
    'has_kd',  # True/False
    'kd_type',  # RB-KD, RB-KD-Att, RAB-KD-Att, N/A
    
    # Hyperparameters
    'hyperparam_temperature',
    'hyperparam_alpha',
    'hyperparam_attention_type',
    'hyperparam_reduction_ratio',
    'hyperparam_kernel_size',
    'hyperparam_layer_positions',
    'hyperparam_teacher_model',
    
    # Training hyperparameters
    'batch_size',
    'max_epochs',
    'patience',
    
    # Training info
    'learning_rate',
    'best_epoch',
    'total_epochs',
    'training_time_seconds',
    'final_train_loss',
    'final_val_loss',
    
    # Train metrics
    'train_accuracy',
    'train_f1',
    'train_precision',
    'train_recall',
    'train_confusion_matrix',
    
    # Validation metrics
    'val_accuracy',
    'val_f1',
    'val_precision',
    'val_recall',
    'val_confusion_matrix',
    
    # Test metrics
    'test_accuracy',
    'test_f1',
    'test_precision',
    'test_recall',
    'test_confusion_matrix',
    
    # Model info
    'model_size_kb',
    'model_size_gzipped_kb',  # Gzipped model size
    'parameter_count',
    'mac_operations_tf',      # MAC operations for TF models
    'mac_operations_tflite',   # MAC operations for TFLite models
    'flops',                   # FLOPs (for TF models)
]


def set_seeds():
    """Set all random seeds for reproducibility."""
    tf.random.set_seed(TRAINING_TF_SEED)
    np.random.seed(TRAINING_NP_SEED)
    tf.keras.backend.set_floatx("float32")


def get_active_device() -> str:
    """Get the active device (GPU or CPU) that TensorFlow is using."""
    test_tensor = tf.constant([1.0])
    device_str = str(test_tensor.device)
    if 'GPU' in device_str:
        return 'GPU'
    else:
        return 'CPU'


def confusion_matrix_to_string(cm):
    """Convert confusion matrix (numpy array) to string representation for CSV."""
    if isinstance(cm, np.ndarray):
        return str(cm.tolist()).replace('\n', ' ')
    return str(cm).replace('\n', ' ')


def extract_model_metadata(model_name: str) -> Dict[str, Any]:
    """
    Extract model metadata from model name.
    
    Returns:
        Dictionary with variant, has_attention, has_kd, kd_type
    """
    metadata = {
        'variant': 'Unknown',
        'has_attention': False,
        'has_kd': False,
        'kd_type': 'N/A',
    }
    
    # Variant detection
    if 'LM' in model_name:
        metadata['variant'] = 'LM'
    elif 'MM' in model_name:
        metadata['variant'] = 'MM'
    elif 'OM' in model_name:
        metadata['variant'] = 'OM'
    
    # Attention detection
    if 'Att' in model_name:
        metadata['has_attention'] = True
    
    # Knowledge Distillation detection
    if 'KD' in model_name or 'BKD' in model_name:
        metadata['has_kd'] = True
        if 'RAB-KD-Att' in model_name or 'RABKD-Att' in model_name or 'RA-BKD-Att' in model_name:
            metadata['kd_type'] = 'RAB-KD-Att'
        elif 'RB-KD-Att' in model_name or 'RBKD-Att' in model_name:
            metadata['kd_type'] = 'RB-KD-Att'
        elif 'RB-KD' in model_name or 'RBKD' in model_name:
            metadata['kd_type'] = 'RB-KD'
        elif 'RAB-KD' in model_name or 'RABKD' in model_name or 'RA-BKD' in model_name:
            metadata['kd_type'] = 'RAB-KD'
    
    return metadata


def save_grid_search_result(
    csv_path: str,
    model_name: str,
    hyperparams: Dict[str, Any],
    val_accuracy: float,
    # Test metrics
    test_accuracy: Optional[float] = None,
    test_f1: Optional[float] = None,
    test_precision: Optional[float] = None,
    test_recall: Optional[float] = None,
    test_confusion_matrix: Optional[np.ndarray] = None,
    # Train metrics
    train_accuracy: Optional[float] = None,
    train_f1: Optional[float] = None,
    train_precision: Optional[float] = None,
    train_recall: Optional[float] = None,
    train_confusion_matrix: Optional[np.ndarray] = None,
    # Val metrics (additional)
    val_f1: Optional[float] = None,
    val_precision: Optional[float] = None,
    val_recall: Optional[float] = None,
    val_confusion_matrix: Optional[np.ndarray] = None,
    # Model info
    model_size_kb: Optional[float] = None,
    model_size_gzipped_kb: Optional[float] = None,
    parameter_count: Optional[int] = None,
    mac_operations_tf: Optional[int] = None,
    mac_operations_tflite: Optional[int] = None,
    flops: Optional[int] = None,
    # Training info
    timestamp: Optional[str] = None,
    device: Optional[str] = None,
    total_epochs: Optional[int] = None,
    best_epoch: Optional[int] = None,
    learning_rate: Optional[float] = None,
    training_time_seconds: Optional[float] = None,
    final_train_loss: Optional[float] = None,
    final_val_loss: Optional[float] = None,
    # Training hyperparameters
    batch_size: Optional[int] = None,
    max_epochs: Optional[int] = None,
    patience: Optional[int] = None,
    # Model metadata (auto-detected if not provided)
    variant: Optional[str] = None,
    has_attention: Optional[bool] = None,
    has_kd: Optional[bool] = None,
    kd_type: Optional[str] = None,
):
    """
    Save a single grid search result to CSV and sort by val_accuracy.
    Uses STANDARD_CSV_FIELDNAMES for consistency.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%d %m %Y, %H:%M:%S")
    
    if device is None:
        device = get_active_device()
    
    # Auto-detect model metadata if not provided
    if variant is None or has_attention is None or has_kd is None or kd_type is None:
        metadata = extract_model_metadata(model_name)
        if variant is None:
            variant = metadata['variant']
        if has_attention is None:
            has_attention = metadata['has_attention']
        if has_kd is None:
            has_kd = metadata['has_kd']
        if kd_type is None:
            kd_type = metadata['kd_type']
    
    # Prepare row with all standard fields
    row = {
        'model_name': model_name,
        'timestamp': timestamp,
        'device': device,
        'variant': variant,
        'has_attention': str(has_attention),
        'has_kd': str(has_kd),
        'kd_type': kd_type,
        'val_accuracy': f"{val_accuracy:.6f}",
    }
    
    # Add hyperparameters
    row.update({f'hyperparam_{k}': str(v) if v is not None else None for k, v in hyperparams.items()})
    
    # Add training hyperparameters
    if batch_size is not None:
        row['batch_size'] = str(batch_size)
    if max_epochs is not None:
        row['max_epochs'] = str(max_epochs)
    if patience is not None:
        row['patience'] = str(patience)
    
    # Add training info
    if total_epochs is not None:
        row['total_epochs'] = str(total_epochs)
    if best_epoch is not None:
        row['best_epoch'] = str(best_epoch)
    if learning_rate is not None:
        row['learning_rate'] = f"{learning_rate:.6f}"
    if training_time_seconds is not None:
        row['training_time_seconds'] = f"{training_time_seconds:.2f}"
    if final_train_loss is not None:
        row['final_train_loss'] = f"{final_train_loss:.6f}"
    if final_val_loss is not None:
        row['final_val_loss'] = f"{final_val_loss:.6f}"
    
    # Add train metrics
    if train_accuracy is not None:
        row['train_accuracy'] = f"{train_accuracy:.6f}"
    if train_f1 is not None:
        row['train_f1'] = f"{train_f1:.6f}"
    if train_precision is not None:
        row['train_precision'] = f"{train_precision:.6f}"
    if train_recall is not None:
        row['train_recall'] = f"{train_recall:.6f}"
    if train_confusion_matrix is not None:
        row['train_confusion_matrix'] = confusion_matrix_to_string(train_confusion_matrix)
    
    # Add validation metrics
    if val_f1 is not None:
        row['val_f1'] = f"{val_f1:.6f}"
    if val_precision is not None:
        row['val_precision'] = f"{val_precision:.6f}"
    if val_recall is not None:
        row['val_recall'] = f"{val_recall:.6f}"
    if val_confusion_matrix is not None:
        row['val_confusion_matrix'] = confusion_matrix_to_string(val_confusion_matrix)
    
    # Add test metrics
    if test_accuracy is not None:
        row['test_accuracy'] = f"{test_accuracy:.6f}"
    if test_f1 is not None:
        row['test_f1'] = f"{test_f1:.6f}"
    if test_precision is not None:
        row['test_precision'] = f"{test_precision:.6f}"
    if test_recall is not None:
        row['test_recall'] = f"{test_recall:.6f}"
    if test_confusion_matrix is not None:
        row['test_confusion_matrix'] = confusion_matrix_to_string(test_confusion_matrix)
    
    # Add model info
    if model_size_kb is not None:
        row['model_size_kb'] = f"{model_size_kb:.2f}"
    if model_size_gzipped_kb is not None:
        row['model_size_gzipped_kb'] = f"{model_size_gzipped_kb:.2f}"
    if parameter_count is not None:
        row['parameter_count'] = str(parameter_count)
    if mac_operations_tf is not None:
        row['mac_operations_tf'] = str(mac_operations_tf)
    if mac_operations_tflite is not None:
        row['mac_operations_tflite'] = str(mac_operations_tflite)
    if flops is not None:
        row['flops'] = str(flops)
    
    # Read existing CSV if it exists
    file_exists = os.path.exists(csv_path)
    rows = []

    if file_exists:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            rows = list(reader)
        
        # Merge fieldnames: use STANDARD_CSV_FIELDNAMES as base, add any extra from existing
        new_fieldnames = set(row.keys())
        standard_set = set(STANDARD_CSV_FIELDNAMES)
        existing_fieldnames_set = set(existing_fieldnames)
        
        # Start with STANDARD_CSV_FIELDNAMES, add any extras from existing/new
        fieldnames = list(STANDARD_CSV_FIELDNAMES)
        for field in existing_fieldnames:
            if field not in standard_set:
                fieldnames.append(field)
        for field in new_fieldnames:
            if field not in standard_set and field not in fieldnames:
                fieldnames.append(field)
    else:
        fieldnames = list(STANDARD_CSV_FIELDNAMES)
        # Add any extra fields from row
        for field in row.keys():
            if field not in fieldnames:
                fieldnames.append(field)

    # Ensure all existing rows have all fieldnames (fill missing with None)
    for existing_row in rows:
        for field in fieldnames:
            if field not in existing_row:
                existing_row[field] = None

    # Ensure new row has all fieldnames (fill missing with None)
    for field in fieldnames:
        if field not in row:
            row[field] = None

    # Add new row
    rows.append(row)
    
    # Sort by val_accuracy (descending)
    rows.sort(key=lambda x: float(x.get('val_accuracy', 0) or 0), reverse=True)
    
    # Write sorted rows back to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_best_model_from_csv(csv_path: str, metric: str = 'val_accuracy') -> Optional[Dict]:
    """
    Find the best model from a grid search CSV based on validation accuracy.
    """
    if not os.path.exists(csv_path):
        return None
    
    # CSV is already sorted, so first row is best
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        first_row = next(reader, None)
        return first_row


def train_and_evaluate_model(
    model_builder: Callable,
    model_name: str,
    save_path: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    patience: int = DEFAULT_PATIENCE,
    verbose: int = 0,
    use_gpu: bool = None,
) -> Tuple[float, Optional[Dict[str, float]], Dict[str, Any], np.ndarray, Optional[Dict[str, float]], Optional[np.ndarray]]:
    """
    Train a model and return validation accuracy and optional test metrics.
    
    Args:
        model_builder: Function that builds the model (no arguments)
        model_name: Name of the model (for logging)
        save_path: Path to save the trained model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data (optional)
        epochs: Number of epochs
        batch_size: Batch size
        patience: Early stopping patience
        verbose: Verbosity level
        use_gpu: If None, uses config.USE_GPU_TRAIN. If True/False, overrides config.
    
    Returns:
        Tuple of (val_accuracy, test_metrics_dict or None, training_info, val_confusion_matrix, train_metrics_dict or None, train_confusion_matrix or None)
    """
    from src.training_tf import train_model
    from src.evaluation_tf import evaluate_tf_model
    from src.model_io import load_model_tf
    
    # Use config if not specified
    if use_gpu is None:
        use_gpu = USE_GPU_TRAIN
    
    # Set seeds
    set_seeds()
    
    # Build model
    model = model_builder()
    
    # Train with timing
    train_start = time.time()
    history, training_info = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        save_path=save_path,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        verbose=verbose,
        use_gpu=use_gpu,
    )
    training_time = time.time() - train_start
    
    # Add training time to training_info
    training_info['training_time_seconds'] = training_time
    
    # Add final losses from history
    if history and len(history.history) > 0:
        if 'loss' in history.history and len(history.history['loss']) > 0:
            training_info['final_train_loss'] = float(history.history['loss'][-1])
        if 'val_loss' in history.history and len(history.history['val_loss']) > 0:
            training_info['final_val_loss'] = float(history.history['val_loss'][-1])
    
    # Add training hyperparameters to training_info
    training_info['batch_size'] = batch_size
    training_info['max_epochs'] = epochs
    training_info['patience'] = patience
    
    # Load best model and evaluate
    model = load_model_tf(os.path.basename(save_path))
    
    # Train metrics
    train_acc, train_f1, train_prec, train_rec, train_cm = evaluate_tf_model(model, X_train, y_train, use_gpu=use_gpu)
    train_metrics = {
        'accuracy': train_acc,
        'f1': train_f1,
        'precision': train_prec,
        'recall': train_rec,
        'confusion_matrix': train_cm,
    }
    
    # Validation metrics
    val_acc, val_f1, val_prec, val_rec, val_cm = evaluate_tf_model(model, X_val, y_val, use_gpu=use_gpu)
    
    # Test metrics (if provided)
    test_metrics = None
    if X_test is not None and y_test is not None:
        test_acc, test_f1, test_prec, test_rec, test_cm = evaluate_tf_model(model, X_test, y_test, use_gpu=use_gpu)
        test_metrics = {
            'accuracy': test_acc,
            'f1': test_f1,
            'precision': test_prec,
            'recall': test_rec,
            'confusion_matrix': test_cm,
        }
    
    # Return val_accuracy, test_metrics, training_info, val_confusion_matrix, train_metrics, train_confusion_matrix
    # Also include val_f1, val_precision, val_recall in training_info for convenience
    training_info['val_f1'] = val_f1
    training_info['val_precision'] = val_prec
    training_info['val_recall'] = val_rec
    
    return val_acc, test_metrics, training_info, val_cm, train_metrics, train_cm


def run_single_experiment(
    variant: str,
    attention_type: str,
    reduction_ratio: Optional[int] = None,
    kernel_size: Optional[int] = None,
    layer_positions: Optional[List[int]] = None,
    input_shape: Optional[Tuple] = None,
    num_classes: Optional[int] = None,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    csv_path: Optional[str] = None,
    experiment_num: Optional[int] = None,
    total_experiments: Optional[int] = None,
):
    """
    Run a single grid search experiment for attention models.
    
    This function handles model training, evaluation, and CSV saving
    for a single attention configuration.
    
    Args:
        variant: Model variant ('OM', 'LM', 'MM')
        attention_type: 'CH_ATT', 'SP_ATT', or 'CBAM'
        reduction_ratio: Channel attention reduction ratio
        kernel_size: Spatial attention kernel size
        layer_positions: List of layer positions (1-indexed)
        input_shape: Model input shape
        num_classes: Number of classes
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        csv_path: Path to CSV results file
        experiment_num: Current experiment number
        total_experiments: Total number of experiments
    """
    from src.model_naming import generate_attention_model_name
    from src.model_io import load_model_tf
    from src.evaluation_tf import evaluate_tf_model
    from src.registry import TF_DIR
    from models.att_models import build_deepconvlstm_att
    from src.utils_resources import get_model_size as get_model_size_kb
    from src.attention_utils import create_attention_lists
    from src.memory_utils import cleanup_memory
    from src.config import (
        MODEL_SEED, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_PATIENCE,
        DEFAULT_VERBOSE, USE_GPU_EVALUATE,
    )
    
    model_name = generate_attention_model_name(
        variant=variant,
        attention_type=attention_type,
        reduction_ratio=reduction_ratio,
        kernel_size=kernel_size,
        layer_positions=layer_positions,
    )
    
    print(f"  [{experiment_num}/{total_experiments}] {model_name}...", end=" ", flush=True)
    
    model_path = os.path.join(TF_DIR, f"{model_name}.h5")
    
    # Check if already trained
    if os.path.exists(model_path):
        print("SKIPPED (already exists)")
        # Still evaluate and save to CSV
        try:
            model = load_model_tf(f"{model_name}.h5")
            val_acc, _, _, _, val_cm = evaluate_tf_model(model, X_val, y_val, use_gpu=USE_GPU_EVALUATE)
            test_acc, test_f1, test_prec, test_rec, test_cm = evaluate_tf_model(model, X_test, y_test, use_gpu=USE_GPU_EVALUATE)
            
            save_grid_search_result(
                csv_path=csv_path,
                model_name=model_name,
                hyperparams={
                    'attention_type': attention_type,
                    'reduction_ratio': reduction_ratio if reduction_ratio else None,
                    'kernel_size': kernel_size if kernel_size else None,
                    'layer_positions': '-'.join(map(str, layer_positions)),
                },
                val_accuracy=val_acc,
                test_accuracy=test_acc,
                test_f1=test_f1,
                test_precision=test_prec,
                test_recall=test_rec,
                model_size_kb=get_model_size_kb(model_path),
                parameter_count=model.count_params(),
                device=get_active_device(),
                val_confusion_matrix=val_cm,
                test_confusion_matrix=test_cm,
            )
            
            # Clear memory
            cleanup_memory(model)
            
            return
        except (OSError, IOError, ValueError) as e:
            print(f"⚠ Corrupted model file detected, will retrain: {e}")
            os.remove(model_path)
            # Fall through to training below
    
    # Create attention lists
    channelatt_list, spatialatt_list = create_attention_lists(
        attention_type=attention_type,
        reduction_ratio=reduction_ratio,
        kernel_size=kernel_size,
        layer_positions=layer_positions,
    )
    
    # Model builder function
    def model_builder():
        set_seeds()
        return build_deepconvlstm_att(
            version=variant,
            input_shape=input_shape,
            num_classes=num_classes,
            channelatt_list=channelatt_list,
            spatialatt_list=spatialatt_list,
            seed=MODEL_SEED,
        )
    
    # Train and evaluate
    try:
        val_acc, test_metrics, training_info, val_cm, train_metrics, train_cm = train_and_evaluate_model(
            model_builder=model_builder,
            model_name=model_name,
            save_path=model_path,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            epochs=DEFAULT_EPOCHS,
            batch_size=DEFAULT_BATCH_SIZE,
            patience=DEFAULT_PATIENCE,
            verbose=DEFAULT_VERBOSE,
        )
        
        # Get model for parameter count
        model = model_builder()
        
        # Get test confusion matrix
        test_cm = test_metrics.get('confusion_matrix') if test_metrics else None
        
        # Save to CSV
        save_grid_search_result(
            csv_path=csv_path,
            model_name=model_name,
            hyperparams={
                'attention_type': attention_type,
                'reduction_ratio': reduction_ratio if reduction_ratio else None,
                'kernel_size': kernel_size if kernel_size else None,
                'layer_positions': '-'.join(map(str, layer_positions)),
            },
            val_accuracy=val_acc,
            # Val metrics
            val_f1=training_info.get('val_f1'),
            val_precision=training_info.get('val_precision'),
            val_recall=training_info.get('val_recall'),
            val_confusion_matrix=val_cm,
            # Test metrics
            test_accuracy=test_metrics['accuracy'] if test_metrics else None,
            test_f1=test_metrics['f1'] if test_metrics else None,
            test_precision=test_metrics['precision'] if test_metrics else None,
            test_recall=test_metrics['recall'] if test_metrics else None,
            test_confusion_matrix=test_cm,
            # Train metrics
            train_accuracy=train_metrics['accuracy'] if train_metrics else None,
            train_f1=train_metrics['f1'] if train_metrics else None,
            train_precision=train_metrics['precision'] if train_metrics else None,
            train_recall=train_metrics['recall'] if train_metrics else None,
            train_confusion_matrix=train_cm,
            # Model info
            model_size_kb=get_model_size_kb(model_path),
            parameter_count=model.count_params(),
            device=get_active_device(),
            # Training info
            total_epochs=training_info['total_epochs'],
            best_epoch=training_info['best_epoch'],
            learning_rate=training_info['learning_rate'],
            training_time_seconds=training_info.get('training_time_seconds'),
            final_train_loss=training_info.get('final_train_loss'),
            final_val_loss=training_info.get('final_val_loss'),
            # Training hyperparameters
            batch_size=training_info.get('batch_size'),
            max_epochs=training_info.get('max_epochs'),
            patience=training_info.get('patience'),
        )
        
        print(f"✓ Val Acc: {val_acc:.4f}")
        
        # Clear memory
        cleanup_memory(model)
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        cleanup_memory()