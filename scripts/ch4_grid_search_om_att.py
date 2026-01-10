"""
Chapter 4 — OM-Att Grid Search

Grid search over attention hyperparameters:
- Channel Attention: reduction_ratio [2, 4, 8] × 4 layer positions = 12 models
- Spatial Attention: kernel_size [3, 5, 7] × 4 layer positions = 12 models
- CBAM: reduction_ratio [2, 4, 8] × kernel_size [3, 5, 7] × 4 layer positions = 36 models

Total: 60 models

All models are saved and results are written to results/grid_search_om_att.csv
"""

import os
import numpy as np
import tensorflow as tf
from itertools import product, combinations

from src.data_opportunity import load_opportunity_splits
from src.grid_search_utils import (
    set_seeds, train_and_evaluate_model, save_grid_search_result, get_active_device,
)
from src.model_naming import generate_attention_model_name
from src.model_io import save_model_tf, load_model_tf
from src.evaluation_tf import evaluate_tf_model
from src.registry import TF_DIR
from models.att_models import build_deepconvlstm_att
from src.config import (
    MODEL_SEED, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_PATIENCE,
    ATT_GRID_CHANNEL_RATIOS, ATT_GRID_SPATIAL_KERNELS, ATT_GRID_LAYER_POSITIONS,
    DEFAULT_VERBOSE, USE_GPU_EVALUATE,
)
from src.utils_resources import get_model_size as get_model_size_kb
from src.cli_parser import parse_args
from src.attention_utils import create_attention_lists
from src.memory_utils import cleanup_memory, periodic_cleanup


# Parser configuration
PARSER_CONFIG = {
    'attention_params': ['ch_att', 'sp_att', 'cbam'],
    'list_params': ['channel_ratios', 'spatial_kernels', 'layer_positions'],
    'valid_values': {
        'channel_ratios': [2, 4, 8],
        'spatial_kernels': [3, 5, 7],
        'layer_positions': [1, 2, 3, 4]
    },
    'help_text': """
OM-Att Grid Search - Usage

Attention Type Parameters (boolean: true/false):
  ch_att=true|false     Run Channel Attention experiments (default: true)
  sp_att=true|false     Run Spatial Attention experiments (default: true)
  cbam=true|false       Run CBAM experiments (default: true)

Hyperparameter Lists (format: value1,value2,...):
  channel_ratios=2,4,8       Channel attention reduction ratios (default: 2,4,8)
                             Valid values: 2, 4, 8
  
  spatial_kernels=3,5,7      Spatial attention kernel sizes (default: 3,5,7)
                             Valid values: 3, 5, 7
  
  layer_positions=1,2,3,4     Layer positions to apply attention (default: 1,2,3,4)
                             Valid values: 1, 2, 3, 4

Examples:
  # Run only ch_att with all defaults
  python ch4_grid_search_om_att.py ch_att=true sp_att=false cbam=false
  
  # Run ch_att with only ratio 2 and 4, on layers 1 and 2
  python ch4_grid_search_om_att.py ch_att=true sp_att=false cbam=false channel_ratios=2,4 layer_positions=1,2
  
  # Run all (default)
  python ch4_grid_search_om_att.py
  
  # Case-insensitive (all work the same)
  python ch4_grid_search_om_att.py CH_ATT=true ch_att=false Ch_Att=true

Note: All parameters are case-insensitive. Invalid parameters or values will stop execution.
""",
    'script_name': 'ch4_grid_search_om_att.py'
}


def main():
    # Parse command line arguments
    args = parse_args(PARSER_CONFIG)
    
    # Get attention type flags (default: all true)
    run_ch_att = args.get('ch_att', True)
    run_sp_att = args.get('sp_att', True)
    run_cbam = args.get('cbam', True)
    
    # Get hyperparameter lists (default: from config)
    channel_ratios = args.get('channel_ratios', ATT_GRID_CHANNEL_RATIOS)
    spatial_kernels = args.get('spatial_kernels', ATT_GRID_SPATIAL_KERNELS)
    layer_positions_options = args.get('layer_positions', ATT_GRID_LAYER_POSITIONS)
    
    print("=== Chapter 4: OM-Att Grid Search ===")
    
    # Calculate total experiments based on selected attention types
    ch_att_count = len(channel_ratios) * len(layer_positions_options) if run_ch_att else 0
    sp_att_count = len(spatial_kernels) * len(layer_positions_options) if run_sp_att else 0
    cbam_count = len(channel_ratios) * len(spatial_kernels) * len(layer_positions_options) if run_cbam else 0
    total_experiments = ch_att_count + sp_att_count + cbam_count
    
    print(f"Total experiments: {total_experiments} ({ch_att_count} CH + {sp_att_count} SP + {cbam_count} CBAM)")
    print(f"Configuration: ch_att={run_ch_att}, sp_att={run_sp_att}, cbam={run_cbam}")
    print(f"  channel_ratios={channel_ratios}, spatial_kernels={spatial_kernels}, layer_positions={layer_positions_options}")
    print()

    # Check available devices and active device
    print(f"TensorFlow devices: {tf.config.list_physical_devices()}")
    test_tensor = tf.constant([1.0])
    print(f"Active device: {test_tensor.device}")
    print()
    
    # Load data
    print("Loading Opportunity dataset splits...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_opportunity_splits()
    
    # Add channel dimension
    X_train = np.expand_dims(X_train, -1)
    X_val = np.expand_dims(X_val, -1)
    X_test = np.expand_dims(X_test, -1)
    
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))
    
    print(f"Data shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"Number of classes: {num_classes}")
    print()
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    csv_path = "results/grid_search_om_att.csv"
    
    print(f"Starting grid search: {total_experiments} experiments")
    print(f"Results will be saved to: {csv_path}")
    print()
    
    experiment_count = 0
    
    # ============================================
    # Channel Attention Grid Search
    # ============================================
    if run_ch_att:
        ch_att_experiments = len(channel_ratios) * len(layer_positions_options)
        print(f"[1/3] Channel Attention Grid Search ({ch_att_experiments} experiments)")
        for ratio, layer_pos in product(channel_ratios, layer_positions_options):
            layers = [layer_pos]
            experiment_count += 1
            model_name = generate_attention_model_name(
                variant="OM",
                attention_type="CH_ATT",
                reduction_ratio=ratio,
                layer_positions=layers,
            )
            
            print(f"  [{experiment_count}/{total_experiments}] {model_name}...", end=" ", flush=True)
            
            # Check if already trained
            model_path = os.path.join(TF_DIR, f"{model_name}.h5")
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
                            'attention_type': 'CH_ATT',
                            'reduction_ratio': ratio,
                            'layer_positions': '-'.join(map(str, layers)),
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
                    
                    continue
                except (OSError, IOError, ValueError) as e:
                    print(f"⚠ Corrupted model file detected, will retrain: {e}")
                    os.remove(model_path)
                    # Fall through to training below
            
            # Create attention lists
            channelatt_list, spatialatt_list = create_attention_lists(
                attention_type="CH_ATT",
                reduction_ratio=ratio,
                layer_positions=layers,
            )
            
            # Model builder function
            def model_builder():
                set_seeds()
                return build_deepconvlstm_att(
                    version="OM",
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
                            
                # Get test confusion matrix
                test_cm = test_metrics.get('confusion_matrix') if test_metrics else None

                # Save to CSV
                save_grid_search_result(
                    csv_path=csv_path,
                    model_name=model_name,
                    hyperparams={
                        'attention_type': 'CH_ATT',
                        'reduction_ratio': ratio,
                        'layer_positions': '-'.join(map(str, layers)),
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
                    parameter_count=model_builder().count_params(),
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
                cleanup_memory()

                # Periodic cleanup every 20 experiments
                periodic_cleanup(experiment_count, interval=20, verbose=True)
                
            except Exception as e:
                print(f"✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                cleanup_memory()
                continue
    else:
        print("[1/3] Channel Attention Grid Search - SKIPPED (ch_att=false)")
    
    # ============================================
    # Spatial Attention Grid Search
    # ============================================
    if run_sp_att:
        sp_att_experiments = len(spatial_kernels) * len(layer_positions_options)
        print(f"\n[2/3] Spatial Attention Grid Search ({sp_att_experiments} experiments)")
        for kernel, layer_pos in product(spatial_kernels, layer_positions_options):
            layers = [layer_pos]
            experiment_count += 1
            model_name = generate_attention_model_name(
                variant="OM",
                attention_type="SP_ATT",
                kernel_size=kernel,
                layer_positions=layers,
            )
            
            print(f"  [{experiment_count}/{total_experiments}] {model_name}...", end=" ", flush=True)
            
            # Check if already trained
            model_path = os.path.join(TF_DIR, f"{model_name}.h5")
            if os.path.exists(model_path):
                print("SKIPPED (already exists)")
                continue
            
            # Create attention lists
            channelatt_list, spatialatt_list = create_attention_lists(
                attention_type="SP_ATT",
                kernel_size=kernel,
                layer_positions=layers,
            )
            
            # Model builder function
            def model_builder():
                set_seeds()
                return build_deepconvlstm_att(
                    version="OM",
                    input_shape=input_shape,
                    num_classes=num_classes,
                    channelatt_list=channelatt_list,
                    spatialatt_list=spatialatt_list,
                    seed=MODEL_SEED,
                )
            
            # Train and evaluate
            try:
                val_acc, test_metrics, training_info, val_cm = train_and_evaluate_model(
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
                
                # Get test confusion matrix
                test_cm = test_metrics.get('confusion_matrix') if test_metrics else None
                
                # Save to CSV
                save_grid_search_result(
                    csv_path=csv_path,
                    model_name=model_name,
                    hyperparams={
                        'attention_type': 'SP_ATT',
                        'kernel_size': kernel,
                        'layer_positions': '-'.join(map(str, layers)),
                    },
                    val_accuracy=val_acc,
                    test_accuracy=test_metrics['accuracy'] if test_metrics else None,
                    test_f1=test_metrics['f1'] if test_metrics else None,
                    test_precision=test_metrics['precision'] if test_metrics else None,
                    test_recall=test_metrics['recall'] if test_metrics else None,
                    model_size_kb=get_model_size_kb(model_path),
                    parameter_count=model_builder().count_params(),
                    device=get_active_device(),
                    total_epochs=training_info['total_epochs'],
                    best_epoch=training_info['best_epoch'],
                    learning_rate=training_info['learning_rate'],
                    val_confusion_matrix=val_cm,
                    test_confusion_matrix=test_cm,
                )
                
                print(f"✓ Val Acc: {val_acc:.4f}")
                
                # Clear memory
                cleanup_memory()

                # Periodic cleanup every 20 experiments
                periodic_cleanup(experiment_count, interval=20, verbose=True)
                
            except Exception as e:
                print(f"✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                cleanup_memory()
                continue
    else:
        print("\n[2/3] Spatial Attention Grid Search - SKIPPED (sp_att=false)")
    
    # ============================================
    # CBAM Grid Search
    # ============================================
    if run_cbam:
        cbam_experiments = len(channel_ratios) * len(spatial_kernels) * len(layer_positions_options)
        print(f"\n[3/3] CBAM Grid Search ({cbam_experiments} experiments)")
        for ratio, kernel, layer_pos in product(channel_ratios, spatial_kernels, layer_positions_options):
            layers = [layer_pos]
            experiment_count += 1
            model_name = generate_attention_model_name(
                variant="OM",
                attention_type="CBAM",
                reduction_ratio=ratio,
                kernel_size=kernel,
                layer_positions=layers,
            )
            
            print(f"  [{experiment_count}/{total_experiments}] {model_name}...", end=" ", flush=True)
            
            # Check if already trained
            model_path = os.path.join(TF_DIR, f"{model_name}.h5")
            if os.path.exists(model_path):
                print("SKIPPED (already exists)")
                continue
            
            # Create attention lists
            channelatt_list, spatialatt_list = create_attention_lists(
                attention_type="CBAM",
                reduction_ratio=ratio,
                kernel_size=kernel,
                layer_positions=layers,
            )
            
            # Model builder function
            def model_builder():
                set_seeds()
                return build_deepconvlstm_att(
                    version="OM",
                    input_shape=input_shape,
                    num_classes=num_classes,
                    channelatt_list=channelatt_list,
                    spatialatt_list=spatialatt_list,
                    seed=MODEL_SEED,
                )
            
            # Train and evaluate
            try:
                val_acc, test_metrics, training_info, val_cm = train_and_evaluate_model(
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
                
                # Get test confusion matrix
                test_cm = test_metrics.get('confusion_matrix') if test_metrics else None
                
                # Save to CSV
                save_grid_search_result(
                    csv_path=csv_path,
                    model_name=model_name,
                    hyperparams={
                        'attention_type': 'CBAM',
                        'reduction_ratio': ratio,
                        'kernel_size': kernel,
                        'layer_positions': '-'.join(map(str, layers)),
                    },
                    val_accuracy=val_acc,
                    test_accuracy=test_metrics['accuracy'] if test_metrics else None,
                    test_f1=test_metrics['f1'] if test_metrics else None,
                    test_precision=test_metrics['precision'] if test_metrics else None,
                    test_recall=test_metrics['recall'] if test_metrics else None,
                    model_size_kb=get_model_size_kb(model_path),
                    parameter_count=model_builder().count_params(),
                    device=get_active_device(),
                    total_epochs=training_info['total_epochs'],
                    best_epoch=training_info['best_epoch'],
                    learning_rate=training_info['learning_rate'],
                    val_confusion_matrix=val_cm,
                    test_confusion_matrix=test_cm,
                )
                
                print(f"✓ Val Acc: {val_acc:.4f}")
                
                # Clear memory
                cleanup_memory()

                # Periodic cleanup every 20 experiments
                periodic_cleanup(experiment_count, interval=20, verbose=True)
                
            except Exception as e:
                print(f"✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                cleanup_memory()
                continue
    else:
        print("\n[3/3] CBAM Grid Search - SKIPPED (cbam=false)")
    
    # Final cleanup
    cleanup_memory()
    
    print(f"\n=== Grid Search Complete ===")
    print(f"Results saved to: {csv_path}")
    print(f"Models saved to: {TF_DIR}")


if __name__ == "__main__":
    main()