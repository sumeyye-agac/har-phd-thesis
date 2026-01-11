"""
Evaluate all models and save results to CSV.

Evaluates:
- All TF models (9 models)
- All TFLite models (5 models)

Metrics per model:
- Train/Val/Test: Accuracy, F1, Precision, Recall
- Confusion Matrix (as string) for Train/Val/Test
- Model Size (KB and gzipped KB)
- Parameter Count (Total, Trainable, Non-trainable)
- Inference Time / Latency (average per sample in ms)
- Model Architecture Info (variant, attention, KD type, etc.)
- Timestamp

Results saved to: results/baseline_results.csv
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
    accuracy_score, confusion_matrix, classification_report
)

from src.data_opportunity import load_opportunity_splits, load_and_prepare_opportunity_data
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
from src.grid_search_utils import get_active_device, STANDARD_CSV_FIELDNAMES, confusion_matrix_to_string, get_training_info_from_grid_search_csvs
from src.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)

def get_model_params(model):
    """Get total, trainable, and non-trainable parameter counts for TF model."""
    try:
        total = model.count_params()
        trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable = total - trainable
        return int(total), int(trainable), int(non_trainable)
    except Exception:
        return None, None, None


def get_model_architecture_info(model_name):
    """Extract architecture information from model name."""
    info = {
        'variant': 'Unknown',
        'has_attention': False,
        'has_kd': False,
        'kd_type': None,
        'attention_type': 'N/A',
        'compression_type': 'N/A',
        'compression_sparsity': 'N/A',
        'quantization_type': 'N/A',
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
        # Attention type detection
        if 'CBAM' in model_name or 'cbam' in model_name.lower():
            info['attention_type'] = 'CBAM'
        elif 'CH_ATT' in model_name or 'CHATT' in model_name or 'ch_att' in model_name.lower():
            info['attention_type'] = 'CH_ATT'
        elif 'SP_ATT' in model_name or 'SPATT' in model_name or 'sp_att' in model_name.lower():
            info['attention_type'] = 'SP_ATT'
        else:
            info['attention_type'] = 'Unknown'
    
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
    
    # Compression type detection (TFLite models)
    if '-Lite' in model_name or model_name.endswith('Lite.tflite'):
        info['compression_type'] = 'Lite'
        info['quantization_type'] = 'None'
    elif '-DRQ' in model_name or model_name.endswith('DRQ.tflite'):
        info['compression_type'] = 'DRQ'
        info['quantization_type'] = 'INT8'
    elif '-FQ' in model_name or model_name.endswith('FQ.tflite'):
        info['compression_type'] = 'FQ'
        info['quantization_type'] = 'Float16'
    elif '-CP' in model_name or model_name.endswith('CP.tflite'):
        info['compression_type'] = 'CP'
        info['compression_sparsity'] = '0.5'  # 50% sparsity
        info['quantization_type'] = 'None'
    elif '-PDP' in model_name or model_name.endswith('PDP.tflite'):
        info['compression_type'] = 'PDP'
        info['compression_sparsity'] = '0.8'  # 80% sparsity
        info['quantization_type'] = 'None'
    
    return info

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
    
    # Test set
    acc_test, f1_test, prec_test, rec_test, cm_test = evaluate_tf_model(model, X_test, y_test)
    
    results['test_accuracy'] = acc_test
    results['test_f1'] = f1_test
    results['test_precision'] = prec_test
    results['test_recall'] = rec_test
    results['test_confusion_matrix'] = confusion_matrix_to_string(cm_test)
    
    # Validation set
    acc_val, f1_val, prec_val, rec_val, cm_val = evaluate_tf_model(model, X_val, y_val)
    
    results['val_accuracy'] = acc_val
    results['val_f1'] = f1_val
    results['val_precision'] = prec_val
    results['val_recall'] = rec_val
    results['val_confusion_matrix'] = confusion_matrix_to_string(cm_val)
    
    # Train set
    acc_train, f1_train, prec_train, rec_train, cm_train = evaluate_tf_model(model, X_train, y_train)
    
    results['train_accuracy'] = acc_train
    results['train_f1'] = f1_train
    results['train_precision'] = prec_train
    results['train_recall'] = rec_train
    results['train_confusion_matrix'] = confusion_matrix_to_string(cm_train)
    
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
    acc_test, f1_test, prec_test, rec_test, cm_test = evaluate_tflite_model(interpreter, X_test, y_test)
    
    results['test_accuracy'] = acc_test
    results['test_f1'] = f1_test
    results['test_precision'] = prec_test
    results['test_recall'] = rec_test
    results['test_confusion_matrix'] = confusion_matrix_to_string(cm_test)
    
    # Validation set
    acc_val, f1_val, prec_val, rec_val, cm_val = evaluate_tflite_model(interpreter, X_val, y_val)
    
    results['val_accuracy'] = acc_val
    results['val_f1'] = f1_val
    results['val_precision'] = prec_val
    results['val_recall'] = rec_val
    results['val_confusion_matrix'] = confusion_matrix_to_string(cm_val)
    
    # Train set
    acc_train, f1_train, prec_train, rec_train, cm_train = evaluate_tflite_model(interpreter, X_train, y_train)
    
    results['train_accuracy'] = acc_train
    results['train_f1'] = f1_train
    results['train_precision'] = prec_train
    results['train_recall'] = rec_train
    results['train_confusion_matrix'] = confusion_matrix_to_string(cm_train)
    
    # Measure inference time (on a single test sample)
    sample = np.expand_dims(X_test[0], axis=0).astype(np.float32)
    inference_time_ms = measure_inference_time_tflite(interpreter, sample)
    results['inference_time_ms'] = inference_time_ms
    
    return results


def main():
    logger.info("=== Evaluating All Models ===")
    
    # Load data
    logger.info("Loading Opportunity dataset splits...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_opportunity_splits()
    num_classes = len(np.unique(y_train))
    logger.info(f"Dataset loaded: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples")
    logger.info(f"Number of classes: {num_classes}")
    
    # Add channel dimension for TF models
    X_train_tf = np.expand_dims(X_train, -1)
    X_val_tf = np.expand_dims(X_val, -1)
    X_test_tf = np.expand_dims(X_test, -1)
    
    # Keep original shape for TFLite (will add dimension in evaluation if needed)
    X_train_tflite = X_train
    X_val_tflite = X_val
    X_test_tflite = X_test
    
    # Get timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y, %H:%M:%S")
    
    # Prepare CSV rows
    csv_rows = []
    
    # ============================================
    # Evaluate TF Models
    # ============================================
    logger.info("[1/2] Evaluating TF Models...")
    tf_models = list(TF_MODEL_FILES.keys())
    
    for model_key in tqdm(tf_models, desc="TF Models"):
        model_name = model_key
        
        if not exists_tf(model_key):
            logger.warning(f"{model_name}: SKIPPED (model not found)")
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
                logger.warning(f"Could not calculate gzipped model size for {model_name}: {e}")
                model_size_gzipped_kb = None
            param_total, param_trainable, param_non_trainable = get_model_params(model)
            
            # Calculate MAC operations and FLOPs for TF models
            try:
                mac_ops_tf = compute_tf_mac_operations(model)
            except Exception as e:
                logger.warning(f"Could not calculate MAC operations for {model_name}: {e}")
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
                logger.warning(f"Could not calculate FLOPs for {model_name}: {e}")
                import traceback
                traceback.print_exc()
                flops = None
            
            # Get architecture info
            arch_info = get_model_architecture_info(model_name)
            
            # Get training info from grid search CSV files (best_epoch, total_epochs)
            training_info = get_training_info_from_grid_search_csvs(model_name)
            
            # Evaluate
            results = evaluate_tf_model_full(
                model, X_train_tf, y_train, X_val_tf, y_val, X_test_tf, y_test, num_classes
            )
            
            # Calculate efficiency metrics (accuracy and F1 per MB)
            model_size_mb = model_size_kb / 1024.0 if model_size_kb else None
            accuracy_per_mb = results['test_accuracy'] / model_size_mb if model_size_mb and model_size_mb > 0 else "N/A"
            f1_per_mb = results['test_f1'] / model_size_mb if model_size_mb and model_size_mb > 0 else "N/A"
            
            # Add to CSV row
            row = {
                'model_name': model_name,
                'model_type': 'TF',
                'device': get_active_device(),
                'variant': arch_info['variant'],
                'has_attention': str(arch_info['has_attention']),
                'has_kd': str(arch_info['has_kd']),
                'kd_type': arch_info['kd_type'] or 'N/A',
                'attention_type': arch_info['attention_type'],
                'compression_type': arch_info['compression_type'],
                'compression_sparsity': arch_info['compression_sparsity'],
                'quantization_type': arch_info['quantization_type'],
                'timestamp': timestamp,
                'best_epoch': training_info.get('best_epoch') or None,
                'total_epochs': training_info.get('total_epochs') or None,
                'model_size_kb': f"{model_size_kb:.2f}" if model_size_kb else "N/A",
                'model_size_gzipped_kb': f"{model_size_gzipped_kb:.2f}" if model_size_gzipped_kb is not None else "N/A",
                'parameter_count': param_total if param_total else "N/A",
                'parameter_count_trainable': param_trainable if param_trainable else "N/A",
                'parameter_count_non_trainable': param_non_trainable if param_non_trainable else "N/A",
                'mac_operations_tf': str(mac_ops_tf) if mac_ops_tf is not None else "N/A",
                'flops': str(flops) if flops is not None else "N/A",
                'accuracy_per_mb': f"{accuracy_per_mb:.6f}" if accuracy_per_mb != "N/A" else "N/A",
                'f1_per_mb': f"{f1_per_mb:.6f}" if f1_per_mb != "N/A" else "N/A",
                **results
            }
            csv_rows.append(row)
            logger.info(f"{model_name}: Test Acc={results['test_accuracy']:.4f}, F1={results['test_f1']:.4f}, Inference={results['inference_time_ms']:.2f}ms")
            
        except Exception as e:
            logger.error(f"{model_name}: ERROR - {e}", exc_info=True)
            continue
    
    # ============================================
    # Evaluate TFLite Models
    # ============================================
    logger.info("[2/2] Evaluating TFLite Models...")
    tflite_models = list(TFLITE_MODEL_FILES.keys())
    
    for model_key in tqdm(tflite_models, desc="TFLite Models"):
        model_name = model_key
        
        if not exists_tflite(model_key):
            logger.warning(f"{model_name}: SKIPPED (model not found)")
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
                logger.warning(f"Could not calculate gzipped model size for {model_name}: {e}")
                model_size_gzipped_kb = None
            param_total, param_trainable, param_non_trainable = get_tflite_params(interpreter)
            
            # Calculate MAC operations for TFLite models
            try:
                mac_ops_tflite = compute_tflite_mac_operations(interpreter)
            except Exception as e:
                logger.warning(f"Could not calculate MAC operations for {model_name}: {e}")
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
            
            # Calculate efficiency metrics (accuracy and F1 per MB)
            model_size_mb = model_size_kb / 1024.0 if model_size_kb else None
            accuracy_per_mb = results['test_accuracy'] / model_size_mb if model_size_mb and model_size_mb > 0 else "N/A"
            f1_per_mb = results['test_f1'] / model_size_mb if model_size_mb and model_size_mb > 0 else "N/A"
            
            # Add to CSV row
            row = {
                'model_name': model_name,
                'model_type': 'TFLite',
                'device': get_active_device(),
                'variant': arch_info['variant'],
                'has_attention': str(arch_info['has_attention']),
                'has_kd': str(arch_info['has_kd']),
                'kd_type': arch_info['kd_type'] or 'N/A',
                'attention_type': arch_info['attention_type'],
                'compression_type': arch_info['compression_type'],
                'compression_sparsity': arch_info['compression_sparsity'],
                'quantization_type': arch_info['quantization_type'],
                'timestamp': timestamp,
                'model_size_kb': f"{model_size_kb:.2f}" if model_size_kb else "N/A",
                'model_size_gzipped_kb': f"{model_size_gzipped_kb:.2f}" if model_size_gzipped_kb is not None else "N/A",
                'parameter_count': param_total if param_total else "N/A",
                'parameter_count_trainable': param_trainable if param_trainable else "N/A",
                'parameter_count_non_trainable': param_non_trainable if param_non_trainable else "N/A",
                'mac_operations_tflite': str(mac_ops_tflite) if mac_ops_tflite is not None else "N/A",
                'accuracy_per_mb': f"{accuracy_per_mb:.6f}" if accuracy_per_mb != "N/A" else "N/A",
                'f1_per_mb': f"{f1_per_mb:.6f}" if f1_per_mb != "N/A" else "N/A",
                **results
            }
            csv_rows.append(row)
            logger.info(f"{model_name}: Test Acc={results['test_accuracy']:.4f}, F1={results['test_f1']:.4f}, Inference={results['inference_time_ms']:.2f}ms")
            
        except Exception as e:
            logger.error(f"{model_name}: ERROR - {e}", exc_info=True)
            continue
    
    # ============================================
    # Save to CSV
    # ============================================
    if not csv_rows:
        logger.warning("No results to save!")
        return
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    csv_path = "results/baseline_results.csv"
    
    # Use STANDARD_CSV_FIELDNAMES as base (already includes model_type and inference_time_ms)
    fieldnames = list(STANDARD_CSV_FIELDNAMES)
    
    # Write CSV in append mode
    file_exists = os.path.exists(csv_path)
    
    # All rows are model rows (no statistics rows)
    model_rows = csv_rows
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerows(model_rows)
    
    logger.info(f"{'='*60}")
    logger.info(f"✓ Results saved to: {csv_path}")
    logger.info(f"✓ Total models evaluated: {len(model_rows)}")
    logger.info(f"✓ Mode: APPEND (new rows added on each run, tracked by timestamp)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()