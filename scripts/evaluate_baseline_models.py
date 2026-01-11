"""
Evaluate baseline models (no attention) and save to CSV.

Baseline models (without attention):
- OM (Original Model)
- LM (Lightweight Model)
- MM (Mid-size Model)

Note: Attention models are evaluated via grid search and best models
are selected separately. This CSV contains only non-attention baseline models.

Results saved to: results/baseline_models.csv
"""

import os
import csv
import numpy as np
import tensorflow as tf
from datetime import datetime

from src.data_opportunity import load_opportunity_splits
from src.evaluation_tf import evaluate_tf_model
from src.model_io import load_model_tf
from src.registry import exists_tf, tf_name, tf_path
from src.utils_resources import get_model_size as get_model_size_kb
from src.grid_search_utils import STANDARD_CSV_FIELDNAMES, get_active_device, confusion_matrix_to_string, extract_model_metadata, get_training_info_from_grid_search_csvs
from src.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)


def evaluate_baseline_model(model_key: str, csv_path: str):
    """Evaluate a single baseline model and save to CSV using STANDARD_CSV_FIELDNAMES."""
    if not exists_tf(model_key):
        logger.warning(f"{model_key}: Model not found, skipping")
        return
    
    logger.info(f"Evaluating {model_key}...")
    
    # Load model
    model = load_model_tf(tf_name(model_key))
    model_path = tf_path(model_key)
    
    # Load and prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, _, _ = load_and_prepare_opportunity_data(logger)
    
    # Evaluate - get confusion matrices
    train_acc, train_f1, train_prec, train_rec, train_cm = evaluate_tf_model(model, X_train, y_train)
    val_acc, val_f1, val_prec, val_rec, val_cm = evaluate_tf_model(model, X_val, y_val)
    test_acc, test_f1, test_prec, test_rec, test_cm = evaluate_tf_model(model, X_test, y_test)
    
    # Save to CSV using STANDARD_CSV_FIELDNAMES
    timestamp = datetime.now().strftime("%d-%m-%Y, %H:%M:%S")
    device = get_active_device()
    
    # Extract metadata using centralized function
    metadata = extract_model_metadata(model_key)
    
    # Get training info from grid search CSV files (best_epoch, total_epochs)
    # Note: Baseline models (OM, LM, MM) may not be in grid search CSVs,
    # but we check anyway in case they were trained via grid search
    training_info = get_training_info_from_grid_search_csvs(model_key)
    
    # Get model size for efficiency calculations
    model_size_kb = get_model_size_kb(model_path)
    model_size_mb = model_size_kb / 1024.0
    
    # Calculate efficiency metrics
    if model_size_mb > 0:
        accuracy_per_mb = f"{test_acc / model_size_mb:.6f}"
        f1_per_mb = f"{test_f1 / model_size_mb:.6f}"
    else:
        accuracy_per_mb = 'N/A'
        f1_per_mb = 'N/A'
    
    row = {
        'model_name': model_key,
        'timestamp': timestamp,
        'device': device,
        'variant': metadata['variant'],
        'has_attention': str(metadata['has_attention']),
        'has_kd': str(metadata['has_kd']),
        'kd_type': metadata['kd_type'] or 'N/A',
        'attention_type': metadata.get('attention_type', 'N/A'),
        'compression_type': metadata.get('compression_type', 'N/A'),
        'compression_sparsity': metadata.get('compression_sparsity', 'N/A'),
        'quantization_type': metadata.get('quantization_type', 'N/A'),
        'best_epoch': training_info.get('best_epoch') or None,
        'total_epochs': training_info.get('total_epochs') or None,
        'model_size_kb': f"{model_size_kb:.2f}",
        'parameter_count': str(model.count_params()),
        'parameter_count_trainable': str(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
        'parameter_count_non_trainable': str(model.count_params() - sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
        'accuracy_per_mb': accuracy_per_mb,
        'f1_per_mb': f1_per_mb,
        # Train metrics
        'train_accuracy': f"{train_acc:.6f}",
        'train_f1': f"{train_f1:.6f}",
        'train_precision': f"{train_prec:.6f}",
        'train_recall': f"{train_rec:.6f}",
        'train_confusion_matrix': confusion_matrix_to_string(train_cm),
        # Validation metrics
        'val_accuracy': f"{val_acc:.6f}",
        'val_f1': f"{val_f1:.6f}",
        'val_precision': f"{val_prec:.6f}",
        'val_recall': f"{val_rec:.6f}",
        'val_confusion_matrix': confusion_matrix_to_string(val_cm),
        # Test metrics
        'test_accuracy': f"{test_acc:.6f}",
        'test_f1': f"{test_f1:.6f}",
        'test_precision': f"{test_prec:.6f}",
        'test_recall': f"{test_rec:.6f}",
        'test_confusion_matrix': confusion_matrix_to_string(test_cm),
    }
    
    file_exists = os.path.exists(csv_path)
    
    # Use STANDARD_CSV_FIELDNAMES as base (already includes model_type)
    fieldnames = list(STANDARD_CSV_FIELDNAMES)
    
    # Read existing rows if file exists
    rows = []
    if file_exists:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            rows = list(reader)
            
            # Merge fieldnames: add any extra fields from existing CSV
            for field in existing_fieldnames:
                if field not in fieldnames:
                    fieldnames.append(field)
            
            # Fill missing fields in existing rows
            for existing_row in rows:
                for field in fieldnames:
                    if field not in existing_row:
                        existing_row[field] = None
    
    # Fill missing fields in new row
    for field in fieldnames:
        if field not in row:
            row[field] = None
    
    rows.append(row)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"✓ (Val Acc: {val_acc:.4f})")


def main():
    logger.info("=== Baseline Models Evaluation ===")
    logger.info("Evaluating baseline models (no attention)")
    
    # Baseline models (without attention)
    baseline_models = [
        "OM",   # Original Model
        "LM",   # Lightweight Model
        "MM",   # Mid-size Model
    ]
    
    os.makedirs("results", exist_ok=True)
    csv_path = "results/baseline_models.csv"
    
    logger.info(f"Evaluating {len(baseline_models)} baseline models...")
    logger.info(f"Results will be saved to: {csv_path}")
    
    for model_key in baseline_models:
        evaluate_baseline_model(model_key, csv_path)
    
    logger.info("=== Done ===")
    logger.info(f"Results saved to: {csv_path}")
    logger.info("Note: Attention models are evaluated via grid search.")
    logger.info("Best attention models are selected via scripts/select_best_models.py")


if __name__ == "__main__":
    main()