"""
Evaluate all models and save results to CSV.

Evaluates:
- All TF models (9 models)
- All TFLite models (5 models)

Metrics per model:
- Train/Val/Test: Accuracy, F1, Precision, Recall
- Per-class metrics (Precision, Recall, F1 for each class)
- Confusion Matrix (as string)
- Model Size (KB and MB)
- Parameter Count (Total, Trainable, Non-trainable)
- Inference Time / Latency (average per sample)
- Evaluation Time (total time to evaluate model)
- Model Architecture Info (variant, attention, etc.)
- Timestamp
"""

import os
import csv
import time
import io
import numpy as np
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    confusion_matrix, classification_report
)

from src.data_opportunity import load_opportunity_splits
from src.evaluation_tf import evaluate_tf_model
from src.evaluation_tflite import evaluate_tflite_model
from src.model_io import load_model_tf, load_interpreter
from src.registry import (
    TFModelKey, TFLiteModelKey,
    TF_MODEL_FILES, TFLITE_MODEL_FILES,
    exists_tf, exists_tflite,
    tf_path, tflite_path,
    tf_name, tflite_name,
)
from src.utils_resources import (
    get_model_size as get_model_size_kb, 
    get_tflite_params,
    compute_tf_mac_operations,
    compute_tflite_mac_operations,
    get_flops,
    get_gzipped_model_size
)
from src.grid_search_utils import get_active_device, STANDARD_CSV_FIELDNAMES, confusion_matrix_to_string

def get_model_params(model):
    """Get total, trainable, and non-trainable parameter counts for TF model."""
    try:
        total = model.count_params()
        trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable = total - trainable
        return int(total), int(trainable), int(non_trainable)
    except Exception:
        return None, None, None


def get_per_class_metrics(y_true, y_pred, num_classes):
    """Calculate per-class precision, recall, and F1 scores."""
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Convert to string representation
    return {
        'per_class_precision': str(per_class_precision.tolist()),
        'per_class_recall': str(per_class_recall.tolist()),
        'per_class_f1': str(per_class_f1.tolist()),
    }


def get_model_architecture_info(model_name):
    """Extract architecture information from model name."""
    info = {
        'variant': 'Unknown',
        'has_attention': False,
        'has_kd': False,
        'kd_type': None,
    }
    
    # Variant detection
    if 'LM' in model_name:
        info['variant'] = 'LM'
    elif 'MM' in model_name:
        info['variant'] = 'MM'
    elif 'OM' in model_name:
        info['variant'] = 'OM'
    
    # Attention detection
    if 'Att' in model_name:
        info['has_attention'] = True
    
    # Knowledge Distillation detection
    if 'KD' in model_name or 'BKD' in model_name:
        info['has_kd'] = True
        if 'RAB-KD-Att' in model_name or 'RABKD-Att' in model_name or 'RA-BKD-Att' in model_name:
            info['kd_type'] = 'RAB-KD-Att'
        elif 'RB-KD-Att' in model_name or 'RBKD-Att' in model_name:
            info['kd_type'] = 'RB-KD-Att'
        elif 'RB-KD' in model_name or 'RBKD' in model_name:
            info['kd_type'] = 'RB-KD'
        elif 'RAB-KD' in model_name or 'RABKD' in model_name or 'RA-BKD' in model_name:
            info['kd_type'] = 'RAB-KD'
    
    return info

def get_model_summary_string(model):
    """Get model.summary() output as a string."""
    try:
        from io import StringIO
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        model.summary(print_fn=lambda x: sys.stdout.write(x + '\n'))
        summary_str = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Clean up: replace newlines with spaces, limit length
        summary_str = summary_str.replace('\n', ' | ').strip()
        # Limit to reasonable length for CSV (e.g., 5000 chars)
        if len(summary_str) > 5000:
            summary_str = summary_str[:5000] + "... (truncated)"
        
        return summary_str
    except Exception as e:
        return f"Error getting summary: {str(e)}"

def measure_inference_time_tf(model, X_sample, num_runs=100):
    """Measure average inference time for TF model."""
    # Warmup
    for _ in range(10):
        _ = model.predict(X_sample, verbose=0)
    
    # Actual timing
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = model.predict(X_sample, verbose=0)
        times.append(time.time() - start)
    
    return np.mean(times) * 1000  # Convert to milliseconds


def measure_inference_time_tflite(interpreter, X_sample, num_runs=100):
    """Measure average inference time for TFLite model."""
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_index, X_sample.astype(np.float32))
        interpreter.invoke()
        _ = interpreter.get_tensor(output_index)
    
    # Actual timing
    times = []
    for _ in range(num_runs):
        start = time.time()
        interpreter.set_tensor(input_index, X_sample.astype(np.float32))
        interpreter.invoke()
        _ = interpreter.get_tensor(output_index)
        times.append(time.time() - start)
    
    return np.mean(times) * 1000  # Convert to milliseconds


def evaluate_tf_model_full(model, X_train, y_train, X_val, y_val, X_test, y_test, num_classes):
    """Evaluate TF model on train, val, and test sets with all metrics."""
    results = {}
    
    # Get model architecture summary
    model_architecture = get_model_summary_string(model)
    results['model_architecture'] = model_architecture
    
    # Test set
    eval_start = time.time()
    acc_test, f1_test, prec_test, rec_test, cm_test = evaluate_tf_model(model, X_test, y_test)
    eval_time_test = time.time() - eval_start
    
    # Get predictions for per-class metrics
    preds_test = []
    for sample in X_test:
        sample = np.expand_dims(sample, axis=0)
        logits = model.predict(sample, verbose=0)
        preds_test.append(np.argmax(logits))
    preds_test = np.array(preds_test)
    
    per_class_test = get_per_class_metrics(y_test, preds_test, num_classes)
    
    results['test_accuracy'] = acc_test
    results['test_f1'] = f1_test
    results['test_precision'] = prec_test
    results['test_recall'] = rec_test
    results['test_confusion_matrix'] = confusion_matrix_to_string(cm_test)
    results['test_eval_time_seconds'] = eval_time_test
    results.update({f'test_{k}': v for k, v in per_class_test.items()})
    
    # Validation set
    eval_start = time.time()
    acc_val, f1_val, prec_val, rec_val, cm_val = evaluate_tf_model(model, X_val, y_val)
    eval_time_val = time.time() - eval_start
    
    preds_val = []
    for sample in X_val:
        sample = np.expand_dims(sample, axis=0)
        logits = model.predict(sample, verbose=0)
        preds_val.append(np.argmax(logits))
    preds_val = np.array(preds_val)
    
    per_class_val = get_per_class_metrics(y_val, preds_val, num_classes)
    
    results['val_accuracy'] = acc_val
    results['val_f1'] = f1_val
    results['val_precision'] = prec_val
    results['val_recall'] = rec_val
    results['val_confusion_matrix'] = confusion_matrix_to_string(cm_val)
    results['val_eval_time_seconds'] = eval_time_val
    results.update({f'val_{k}': v for k, v in per_class_val.items()})
    
    # Train set
    eval_start = time.time()
    acc_train, f1_train, prec_train, rec_train, cm_train = evaluate_tf_model(model, X_train, y_train)
    eval_time_train = time.time() - eval_start
    
    preds_train = []
    for sample in X_train:
        sample = np.expand_dims(sample, axis=0)
        logits = model.predict(sample, verbose=0)
        preds_train.append(np.argmax(logits))
    preds_train = np.array(preds_train)
    
    per_class_train = get_per_class_metrics(y_train, preds_train, num_classes)
    
    results['train_accuracy'] = acc_train
    results['train_f1'] = f1_train
    results['train_precision'] = prec_train
    results['train_recall'] = rec_train
    results['train_confusion_matrix'] = confusion_matrix_to_string(cm_train)
    results['train_eval_time_seconds'] = eval_time_train
    results.update({f'train_{k}': v for k, v in per_class_train.items()})
    
    # Measure inference time (on a single test sample)
    sample = np.expand_dims(X_test[0], axis=0)
    inference_time_ms = measure_inference_time_tf(model, sample)
    results['inference_time_ms'] = inference_time_ms
    
    return results


def evaluate_tflite_model_full(interpreter, X_train, y_train, X_val, y_val, X_test, y_test, num_classes):
    """Evaluate TFLite model on train, val, and test sets with all metrics."""
    results = {}
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    # Test set
    eval_start = time.time()
    acc_test, f1_test, prec_test, rec_test, cm_test = evaluate_tflite_model(interpreter, X_test, y_test)
    eval_time_test = time.time() - eval_start
    
    # Get predictions for per-class metrics
    preds_test = []
    for sample in X_test:
        sample = np.expand_dims(sample, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        preds_test.append(np.argmax(output[0]))
    preds_test = np.array(preds_test)
    
    per_class_test = get_per_class_metrics(y_test, preds_test, num_classes)
    
    results['test_accuracy'] = acc_test
    results['test_f1'] = f1_test
    results['test_precision'] = prec_test
    results['test_recall'] = rec_test
    results['test_confusion_matrix'] = confusion_matrix_to_string(cm_test)
    results['test_eval_time_seconds'] = eval_time_test
    results.update({f'test_{k}': v for k, v in per_class_test.items()})
    
    # Validation set
    eval_start = time.time()
    acc_val, f1_val, prec_val, rec_val, cm_val = evaluate_tflite_model(interpreter, X_val, y_val)
    eval_time_val = time.time() - eval_start
    
    preds_val = []
    for sample in X_val:
        sample = np.expand_dims(sample, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        preds_val.append(np.argmax(output[0]))
    preds_val = np.array(preds_val)
    
    per_class_val = get_per_class_metrics(y_val, preds_val, num_classes)
    
    results['val_accuracy'] = acc_val
    results['val_f1'] = f1_val
    results['val_precision'] = prec_val
    results['val_recall'] = rec_val
    results['val_confusion_matrix'] = confusion_matrix_to_string(cm_val)
    results['val_eval_time_seconds'] = eval_time_val
    results.update({f'val_{k}': v for k, v in per_class_val.items()})
    
    # Train set
    eval_start = time.time()
    acc_train, f1_train, prec_train, rec_train, cm_train = evaluate_tflite_model(interpreter, X_train, y_train)
    eval_time_train = time.time() - eval_start
    
    preds_train = []
    for sample in X_train:
        sample = np.expand_dims(sample, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        preds_train.append(np.argmax(output[0]))
    preds_train = np.array(preds_train)
    
    per_class_train = get_per_class_metrics(y_train, preds_train, num_classes)
    
    results['train_accuracy'] = acc_train
    results['train_f1'] = f1_train
    results['train_precision'] = prec_train
    results['train_recall'] = rec_train
    results['train_confusion_matrix'] = confusion_matrix_to_string(cm_train)
    results['train_eval_time_seconds'] = eval_time_train
    results.update({f'train_{k}': v for k, v in per_class_train.items()})
    
    # Measure inference time (on a single test sample)
    sample = np.expand_dims(X_test[0], axis=0).astype(np.float32)
    inference_time_ms = measure_inference_time_tflite(interpreter, sample)
    results['inference_time_ms'] = inference_time_ms
    
    return results


def calculate_summary_statistics(csv_rows):
    """Calculate summary statistics from all evaluated models."""
    if not csv_rows:
        return None
    
    summary = {
        'model_name': 'SUMMARY',
        'model_type': 'STATISTICS',
        'timestamp': datetime.now().strftime("%d %m %Y, %H:%M:%S"),
    }
    
    # Metrics to summarize
    metrics = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall',
               'val_accuracy', 'val_f1', 'val_precision', 'val_recall',
               'train_accuracy', 'train_f1', 'train_precision', 'train_recall',
               'model_size_kb', 'parameter_count', 'inference_time_ms']
    
    for metric in metrics:
        values = []
        for row in csv_rows:
            val = row.get(metric)
            if val and val != 'N/A':
                try:
                    # Handle string numbers
                    if isinstance(val, str):
                        val = float(val.replace(',', ''))
                    else:
                        val = float(val)
                    values.append(val)
                except (ValueError, TypeError):
                    pass
        
        if values:
            summary[f'{metric}_mean'] = f"{np.mean(values):.4f}"
            summary[f'{metric}_min'] = f"{np.min(values):.4f}"
            summary[f'{metric}_max'] = f"{np.max(values):.4f}"
            summary[f'{metric}_std'] = f"{np.std(values):.4f}"
    
    # Find best models
    test_acc_values = []
    for row in csv_rows:
        val = row.get('test_accuracy')
        if val and val != 'N/A':
            try:
                if isinstance(val, str):
                    val = float(val.replace(',', ''))
                else:
                    val = float(val)
                test_acc_values.append((val, row.get('model_name', 'Unknown')))
            except (ValueError, TypeError):
                pass
    
    if test_acc_values:
        best_model = max(test_acc_values, key=lambda x: x[0])
        summary['best_model_test_accuracy'] = f"{best_model[1]} ({best_model[0]:.4f})"
    
    return summary


def main():
    print("=== Evaluating All Models ===")
    print()
    
    # Load data
    print("Loading Opportunity dataset splits...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_opportunity_splits()
    num_classes = len(np.unique(y_train))
    print(f"Dataset loaded: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples")
    print(f"Number of classes: {num_classes}")
    print()
    
    # Add channel dimension for TF models
    X_train_tf = np.expand_dims(X_train, -1)
    X_val_tf = np.expand_dims(X_val, -1)
    X_test_tf = np.expand_dims(X_test, -1)
    
    # Keep original shape for TFLite (will add dimension in evaluation if needed)
    X_train_tflite = X_train
    X_val_tflite = X_val
    X_test_tflite = X_test
    
    # Get timestamp
    timestamp = datetime.now().strftime("%d %m %Y, %H:%M:%S")
    
    # Prepare CSV rows
    csv_rows = []
    
    # ============================================
    # Evaluate TF Models
    # ============================================
    print("[1/2] Evaluating TF Models...")
    tf_models = list(TF_MODEL_FILES.keys())
    
    for model_key in tqdm(tf_models, desc="TF Models"):
        model_name = model_key
        
        if not exists_tf(model_key):
            print(f"  ⚠ {model_name}: SKIPPED (model not found)")
            continue
        
        try:
            # Load model
            model = load_model_tf(tf_name(model_key))
            
            # Get model info
            model_path = tf_path(model_key)
            model_size_kb = get_model_size_kb(model_path)
            try:
                model_size_gzipped_kb = get_gzipped_model_size(model_path)
            except Exception as e:
                print(f"    Warning: Could not calculate gzipped model size for {model_name}: {e}")
                model_size_gzipped_kb = None
            param_total, param_trainable, param_non_trainable = get_model_params(model)
            
            # Calculate MAC operations and FLOPs for TF models
            try:
                mac_ops_tf = compute_tf_mac_operations(model)
            except Exception as e:
                print(f"    Warning: Could not calculate MAC operations for {model_name}: {e}")
                mac_ops_tf = None
            
            try:
                # Get input shape from model and create TensorSpec for FLOPs calculation
                # get_flops expects TensorSpec objects, not Tensor objects
                if isinstance(model.input, list):
                    # Multiple inputs
                    model_inputs = [inp for inp in model.input]
                else:
                    # Single input
                    model_inputs = [model.input]
                
                # Convert to TensorSpec if needed
                tensor_specs = []
                for inp in model_inputs:
                    if hasattr(inp, 'shape') and hasattr(inp, 'dtype'):
                        # It's already a TensorSpec or Tensor, create TensorSpec
                        batch_size = 1
                        shape = inp.shape[1:] if len(inp.shape) > 1 else inp.shape
                        tensor_specs.append(tf.TensorSpec([batch_size] + list(shape), dtype=inp.dtype))
                    else:
                        # Fallback: use model's input shape
                        input_shape = model.input_shape
                        if isinstance(input_shape, list):
                            input_shape = input_shape[0]
                        batch_size = 1
                        tensor_specs.append(tf.TensorSpec([batch_size] + list(input_shape[1:]), dtype=tf.float32))
                
                flops = get_flops(model, tensor_specs)
            except Exception as e:
                print(f"    Warning: Could not calculate FLOPs for {model_name}: {e}")
                import traceback
                traceback.print_exc()
                flops = None
            
            # Get architecture info
            arch_info = get_model_architecture_info(model_name)
            
            # Evaluate
            results = evaluate_tf_model_full(
                model, X_train_tf, y_train, X_val_tf, y_val, X_test_tf, y_test, num_classes
            )
            
            # Add to CSV row
            row = {
                'model_name': model_name,
                'model_type': 'TF',
                'device': get_active_device(),
                'variant': arch_info['variant'],
                'has_attention': str(arch_info['has_attention']),
                'has_kd': str(arch_info['has_kd']),
                'kd_type': arch_info['kd_type'] or 'N/A',
                'model_architecture': results.get('model_architecture', 'N/A'),
                'timestamp': timestamp,
                'model_size_kb': f"{model_size_kb:.2f}" if model_size_kb else "N/A",
                'model_size_gzipped_kb': f"{model_size_gzipped_kb:.2f}" if model_size_gzipped_kb is not None else "N/A",
                'parameter_count': param_total if param_total else "N/A",
                'parameter_count_trainable': param_trainable if param_trainable else "N/A",
                'parameter_count_non_trainable': param_non_trainable if param_non_trainable else "N/A",
                'mac_operations_tf': str(mac_ops_tf) if mac_ops_tf is not None else "N/A",
                'flops': str(flops) if flops is not None else "N/A",
                **results
            }
            csv_rows.append(row)
            print(f"  ✓ {model_name}: Test Acc={results['test_accuracy']:.4f}, F1={results['test_f1']:.4f}, Inference={results['inference_time_ms']:.2f}ms")
            
        except Exception as e:
            print(f"  ✗ {model_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ============================================
    # Evaluate TFLite Models
    # ============================================
    print("\n[2/2] Evaluating TFLite Models...")
    tflite_models = list(TFLITE_MODEL_FILES.keys())
    
    for model_key in tqdm(tflite_models, desc="TFLite Models"):
        model_name = model_key
        
        if not exists_tflite(model_key):
            print(f"  ⚠ {model_name}: SKIPPED (model not found)")
            continue
        
        try:
            # Load interpreter
            interpreter = load_interpreter(tflite_name(model_key))
            
            # Get model info
            model_path = tflite_path(model_key)
            model_size_kb = get_model_size_kb(model_path)
            try:
                model_size_gzipped_kb = get_gzipped_model_size(model_path)
            except Exception as e:
                print(f"    Warning: Could not calculate gzipped model size for {model_name}: {e}")
                model_size_gzipped_kb = None
            param_total, param_trainable, param_non_trainable = get_tflite_params(interpreter)
            
            # Calculate MAC operations for TFLite models
            try:
                mac_ops_tflite = compute_tflite_mac_operations(interpreter)
            except Exception as e:
                print(f"    Warning: Could not calculate MAC operations for {model_name}: {e}")
                import traceback
                traceback.print_exc()
                mac_ops_tflite = None
            
            # Get architecture info
            arch_info = get_model_architecture_info(model_name)
            
            # Add channel dimension for TFLite evaluation
            X_train_tfl = np.expand_dims(X_train_tflite, -1)
            X_val_tfl = np.expand_dims(X_val_tflite, -1)
            X_test_tfl = np.expand_dims(X_test_tflite, -1)
            
            # Evaluate
            results = evaluate_tflite_model_full(
                interpreter, X_train_tfl, y_train, X_val_tfl, y_val, X_test_tfl, y_test, num_classes
            )
            
            # Add to CSV row
            row = {
                'model_name': model_name,
                'model_type': 'TFLite',
                'device': get_active_device(),
                'variant': arch_info['variant'],
                'has_attention': str(arch_info['has_attention']),
                'has_kd': str(arch_info['has_kd']),
                'kd_type': arch_info['kd_type'] or 'N/A',
                'timestamp': timestamp,
                'model_size_kb': f"{model_size_kb:.2f}" if model_size_kb else "N/A",
                'model_size_gzipped_kb': f"{model_size_gzipped_kb:.2f}" if model_size_gzipped_kb is not None else "N/A",
                'parameter_count': param_total if param_total else "N/A",
                'parameter_count_trainable': param_trainable if param_trainable else "N/A",
                'parameter_count_non_trainable': param_non_trainable if param_non_trainable else "N/A",
                'mac_operations_tflite': str(mac_ops_tflite) if mac_ops_tflite is not None else "N/A",
                **results
            }
            csv_rows.append(row)
            print(f"  ✓ {model_name}: Test Acc={results['test_accuracy']:.4f}, F1={results['test_f1']:.4f}, Inference={results['inference_time_ms']:.2f}ms")
            
        except Exception as e:
            print(f"  ✗ {model_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ============================================
    # Calculate Summary Statistics
    # ============================================
    print("\n[3/3] Calculating Summary Statistics...")
    summary = calculate_summary_statistics(csv_rows)
    if summary:
        csv_rows.append(summary)
        print("  ✓ Summary statistics calculated")
    
    # ============================================
    # Save to CSV
    # ============================================
    if not csv_rows:
        print("\n⚠ No results to save!")
        return
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    csv_path = "results/evaluation_results.csv"
    
    # Use STANDARD_CSV_FIELDNAMES as base
    fieldnames = list(STANDARD_CSV_FIELDNAMES)
    
    # Add evaluation-specific fields
    additional_fields = [
        'model_type',  # TF or TFLite
        'parameter_count_trainable',
        'parameter_count_non_trainable',
        'inference_time_ms',
        'train_per_class_precision',
        'train_per_class_recall',
        'train_per_class_f1',
        'train_eval_time_seconds',
        'val_per_class_precision',
        'val_per_class_recall',
        'val_per_class_f1',
        'val_eval_time_seconds',
        'test_per_class_precision',
        'test_per_class_recall',
        'test_per_class_f1',
        'test_eval_time_seconds',
        'model_architecture',
    ]
    
    # Add additional fields that are not in STANDARD_CSV_FIELDNAMES
    for field in additional_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    
    # Add summary columns if summary exists
    if summary:
        summary_cols = [col for col in summary.keys() if col not in fieldnames]
        fieldnames.extend(summary_cols)
    
    # Write CSV in append mode
    file_exists = os.path.exists(csv_path)
    
    # Filter out statistics rows, keep only model rows
    model_rows = [row for row in csv_rows if row.get('model_type') != 'STATISTICS']
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerows(model_rows)
    
    # Save summary statistics to separate file
    if summary:
        summary_path = "results/evaluation_summary.csv"
        summary_fieldnames = [col for col in summary.keys() if col.startswith(('test_', 'val_', 'train_', 'model_', 'parameter_', 'inference_')) or col in ['model_name', 'model_type', 'timestamp', 'best_model_test_accuracy']]
        
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerow(summary)
    
    print(f"\n{'='*60}")
    print(f"✓ Results saved to: {csv_path}")
    print(f"✓ Total models evaluated: {len(model_rows)}")
    print(f"✓ Mode: APPEND (new rows added on each run, tracked by timestamp)")
    if summary:
        print(f"✓ Summary statistics saved to: results/evaluation_summary.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()