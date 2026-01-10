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
from datetime import datetime

from src.data_opportunity import load_opportunity_splits
from src.evaluation_tf import evaluate_tf_model
from src.model_io import load_model_tf
from src.registry import exists_tf, tf_name, tf_path
from src.utils_resources import get_model_size as get_model_size_kb
from src.grid_search_utils import STANDARD_CSV_FIELDNAMES, get_active_device, confusion_matrix_to_string



def evaluate_baseline_model(model_key: str, csv_path: str):
    """Evaluate a single baseline model and save to CSV using STANDARD_CSV_FIELDNAMES."""
    if not exists_tf(model_key):
        print(f"  ⚠ {model_key}: Model not found, skipping")
        return
    
    print(f"  Evaluating {model_key}...", end=" ", flush=True)
    
    # Load model
    model = load_model_tf(tf_name(model_key))
    model_path = tf_path(model_key)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_opportunity_splits()
    X_train = np.expand_dims(X_train, -1)
    X_val = np.expand_dims(X_val, -1)
    X_test = np.expand_dims(X_test, -1)
    
    # Evaluate - get confusion matrices
    train_acc, train_f1, train_prec, train_rec, train_cm = evaluate_tf_model(model, X_train, y_train)
    val_acc, val_f1, val_prec, val_rec, val_cm = evaluate_tf_model(model, X_val, y_val)
    test_acc, test_f1, test_prec, test_rec, test_cm = evaluate_tf_model(model, X_test, y_test)
    
    # Save to CSV using STANDARD_CSV_FIELDNAMES
    timestamp = datetime.now().strftime("%d %m %Y, %H:%M:%S")
    device = get_active_device()
    
    # Determine variant from model_key
    variant = model_key  # OM, LM, or MM
    
    row = {
        'model_name': model_key,
        'timestamp': timestamp,
        'device': device,
        'variant': variant,
        'has_attention': 'False',
        'has_kd': 'False',
        'kd_type': 'N/A',
        'model_size_kb': f"{get_model_size_kb(model_path):.2f}",
        'parameter_count': str(model.count_params()),
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
    
    # Use STANDARD_CSV_FIELDNAMES as base
    fieldnames = list(STANDARD_CSV_FIELDNAMES)
    
    # Add 'model_type' if not in standard (for baseline models)
    if 'model_type' not in fieldnames:
        fieldnames.append('model_type')
    
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
    
    print(f"✓ (Val Acc: {val_acc:.4f})")


def main():
    print("=== Baseline Models Evaluation ===")
    print("Evaluating baseline models (no attention)")
    print()
    
    # Baseline models (without attention)
    baseline_models = [
        "OM",   # Original Model
        "LM",   # Lightweight Model
        "MM",   # Mid-size Model
    ]
    
    os.makedirs("results", exist_ok=True)
    csv_path = "results/baseline_models.csv"
    
    print(f"Evaluating {len(baseline_models)} baseline models...")
    print(f"Results will be saved to: {csv_path}")
    print()
    
    for model_key in baseline_models:
        evaluate_baseline_model(model_key, csv_path)
    
    print(f"\n=== Done ===")
    print(f"Results saved to: {csv_path}")
    print("\nNote: Attention models are evaluated via grid search.")
    print("Best attention models are selected via scripts/select_best_models.py")


if __name__ == "__main__":
    main()