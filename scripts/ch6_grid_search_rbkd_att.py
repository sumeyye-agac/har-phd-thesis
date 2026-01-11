"""
Chapter 6 — LM RB-KD-Att Grid Search

Grid search over:
- KD hyperparameters: τ = [1, 2, 4, 6, 8, 10, 15] × α = [0.1, 0.2, ..., 0.9] = 63 combinations
- Attention hyperparameters: 60 combinations (12 CH + 12 SP + 36 CBAM)

Total: 63 × 60 = 3,780 models

Teacher: OM-Att (with matching attention config)
Student: LM-Att (with same attention config)

All models are saved and results are written to results/grid_search_rbkd_att.csv
"""

import os
import time
import numpy as np

# Set GPU memory growth BEFORE importing TensorFlow
from src.gpu_utils import setup_gpu_environment
from src.config import TRAINING_TF_SEED
setup_gpu_environment(enable_memory_growth=True, set_seed=True, set_floatx=True, seed_value=TRAINING_TF_SEED)

import tensorflow as tf
from itertools import product

from src.data_opportunity import load_opportunity_splits, load_and_prepare_opportunity_data
from src.grid_search_utils import set_seeds, save_grid_search_result, get_active_device
from src.model_naming import generate_attention_model_name, generate_kd_model_name
from src.model_io import load_model_tf
from src.training_kd import train_kd
from src.evaluation_tf import evaluate_tf_model
from src.registry import TF_DIR, exists_tf
from models.att_models import build_deepconvlstm_att
from src.config import (
    MODEL_SEED, TRAINING_TF_SEED, TRAINING_NP_SEED,
    KD_DEFAULT_EPOCHS, KD_DEFAULT_PATIENCE, DEFAULT_BATCH_SIZE,
    KD_GRID_TEMPERATURES, KD_GRID_ALPHAS,
    ATT_GRID_CHANNEL_RATIOS, ATT_GRID_SPATIAL_KERNELS, ATT_GRID_LAYER_POSITIONS,
    KD_BASELINE_BETA_ATT, DEFAULT_VERBOSE,
    USE_GPU_TRAIN_KD, USE_GPU_EVALUATE,
)
from src.utils_resources import get_model_size as get_model_size_kb
from src.cli_parser import parse_args
from src.attention_utils import create_attention_lists, get_attention_config_string
from src.memory_utils import cleanup_memory, cleanup_on_error
from src.logger import setup_logger
from src.gpu_utils import setup_gpu_and_log_device

# Setup logger
logger = setup_logger(__name__)

# Parser configuration
PARSER_CONFIG = {
    'attention_params': ['ch_att', 'sp_att', 'cbam'],
    'list_params': ['temperatures', 'alphas', 'channel_ratios', 'spatial_kernels', 'layer_positions'],
    'valid_values': {
        'temperatures': [1, 2, 4, 6, 8, 10, 15],
        'alphas': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'channel_ratios': [2, 4, 8],
        'spatial_kernels': [3, 5, 7],
        'layer_positions': [1, 2, 3, 4]
    },
    'help_text': """
LM RB-KD-Att Grid Search - Usage

Attention Type Parameters (boolean: true/false):
  ch_att=true|false     Run Channel Attention experiments (default: true)
  sp_att=true|false     Run Spatial Attention experiments (default: true)
  cbam=true|false       Run CBAM experiments (default: true)

Hyperparameter Lists (format: value1,value2,...):
  temperatures=1,2,4,6,8,10,15  Temperature values (default: 1,2,4,6,8,10,15)
                                  Valid values: 1, 2, 4, 6, 8, 10, 15
  
  alphas=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9  Alpha values (default: 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
                                  Valid values: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
  
  channel_ratios=2,4,8         Channel attention reduction ratios (default: 2,4,8)
                                  Valid values: 2, 4, 8
  
  spatial_kernels=3,5,7        Spatial attention kernel sizes (default: 3,5,7)
                                  Valid values: 3, 5, 7
  
  layer_positions=1,2,3,4      Layer positions to apply attention (default: 1,2,3,4)
                                  Valid values: 1, 2, 3, 4

Examples:
  # Run with all defaults (63 KD × 60 Attention = 3,780 models)
  python ch6_grid_search_rbkd_att.py
  
  # Run only CH_ATT with temperature 1,2 and alpha 0.1,0.2
  python ch6_grid_search_rbkd_att.py ch_att=true sp_att=false cbam=false temperatures=1,2 alphas=0.1,0.2
  
  # Run with custom attention parameters
  python ch6_grid_search_rbkd_att.py channel_ratios=2,4 layer_positions=1,2

Note: All parameters are case-insensitive. Invalid parameters or values will stop execution.
""",
    'script_name': 'ch6_grid_search_rbkd_att.py'
}


def main():
    # Parse command line arguments
    args = parse_args(PARSER_CONFIG)
    
    # Get attention type flags (default: all true)
    run_ch_att = args.get('ch_att', True)
    run_sp_att = args.get('sp_att', True)
    run_cbam = args.get('cbam', True)
    
    # Get hyperparameter lists (default: from config)
    temperatures = args.get('temperatures', KD_GRID_TEMPERATURES)
    alphas = args.get('alphas', KD_GRID_ALPHAS)
    channel_ratios = args.get('channel_ratios', ATT_GRID_CHANNEL_RATIOS)
    spatial_kernels = args.get('spatial_kernels', ATT_GRID_SPATIAL_KERNELS)
    layer_positions_options = args.get('layer_positions', ATT_GRID_LAYER_POSITIONS)
    
    logger.info("=== Chapter 6: LM RB-KD-Att Grid Search ===")
    
    # Generate all attention configs based on selected attention types
    attention_configs = []
    
    # Channel Attention
    if run_ch_att:
        for ratio, layer_pos in product(channel_ratios, layer_positions_options):
            layers = [layer_pos]
            attention_configs.append({
                'type': 'CH_ATT',
                'reduction_ratio': ratio,
                'kernel_size': None,
                'layer_positions': layers,
            })
    
    # Spatial Attention
    if run_sp_att:
        for kernel, layer_pos in product(spatial_kernels, layer_positions_options):
            layers = [layer_pos]
            attention_configs.append({
                'type': 'SP_ATT',
                'reduction_ratio': None,
                'kernel_size': kernel,
                'layer_positions': layers,
            })
    
    # CBAM
    if run_cbam:
        for ratio, kernel, layer_pos in product(channel_ratios, spatial_kernels, layer_positions_options):
            layers = [layer_pos]
            attention_configs.append({
                'type': 'CBAM',
                'reduction_ratio': ratio,
                'kernel_size': kernel,
                'layer_positions': layers,
            })
    
    ch_att_count = len(channel_ratios) * len(layer_positions_options) if run_ch_att else 0
    sp_att_count = len(spatial_kernels) * len(layer_positions_options) if run_sp_att else 0
    cbam_count = len(channel_ratios) * len(spatial_kernels) * len(layer_positions_options) if run_cbam else 0
    attention_count = ch_att_count + sp_att_count + cbam_count
    total_experiments = len(temperatures) * len(alphas) * len(attention_configs)
    
    logger.info(f"Total experiments: {total_experiments} ({len(temperatures)} temperatures × {len(alphas)} alphas × {attention_count} attention configs)")
    logger.info(f"Configuration: ch_att={run_ch_att}, sp_att={run_sp_att}, cbam={run_cbam}")
    logger.info(f"  temperatures={temperatures}, alphas={alphas}")
    logger.info(f"  channel_ratios={channel_ratios}, spatial_kernels={spatial_kernels}, layer_positions={layer_positions_options}")
    
    # Setup GPU and log device info
    setup_gpu_and_log_device(logger)
    
    # Load and prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, input_shape, num_classes = load_and_prepare_opportunity_data(logger)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    csv_path = "results/grid_search_rbkd_att.csv"
    
    logger.info(f"Starting grid search: {total_experiments} experiments")
    logger.info(f"Results will be saved to: {csv_path}")
    
    experiment_count = 0
    
    for temp, alpha, att_config in product(temperatures, alphas, attention_configs):
        experiment_count += 1
        
        # Get attention config string
        att_config_str = get_attention_config_string(
            attention_type=att_config['type'],
            reduction_ratio=att_config['reduction_ratio'],
            kernel_size=att_config['kernel_size'],
            layer_positions=att_config['layer_positions'],
        )
        
        # Generate model name
        model_name = generate_kd_model_name(
            variant="LM",
            kd_type="RB-KD-Att",
            temperature=temp,
            alpha=alpha,
            attention_config=att_config_str,
        )
        
        if experiment_count % 100 == 0:
            logger.info(f"Progress: {experiment_count}/{total_experiments} experiments completed...")
            # Aggressive cleanup every 100 experiments
            cleanup_memory()
        
        logger.info(f"[{experiment_count}/{total_experiments}] {model_name}...")
        
        model_path = os.path.join(TF_DIR, f"{model_name}.h5")
        
        # Check if already trained
        if os.path.exists(model_path):
            logger.info("SKIPPED")
            # Periodic cleanup even for skipped models
            if experiment_count % 50 == 0:
                cleanup_memory()
            continue
        
        # Get teacher model name (OM-Att with same attention config)
        teacher_name = generate_attention_model_name(
            variant="OM",
            attention_type=att_config['type'],
            reduction_ratio=att_config['reduction_ratio'],
            kernel_size=att_config['kernel_size'],
            layer_positions=att_config['layer_positions'],
        )
        
        teacher_path = os.path.join(TF_DIR, f"{teacher_name}.h5")
        
        if not os.path.exists(teacher_path):
            logger.warning(f"SKIPPED (teacher {teacher_name} not found)")
            continue
        
        # Cleanup at start of each iteration
        cleanup_memory()

        # Load teacher model
        teacher = None
        try:
            # Determine device based on config
            device = '/GPU:0' if USE_GPU_TRAIN_KD else '/CPU:0'
            
            with tf.device(device):
                teacher = load_model_tf(f"{teacher_name}.h5")
            
            # Remove warm-up (causes Metal GPU issues, not needed for CPU)
            # Aggressive cleanup after teacher load
            cleanup_memory()
            time.sleep(0.1)  # Shorter delay for CPU
            
        except Exception as e:
            logger.error(f"ERROR loading teacher: {e}", exc_info=True)
            cleanup_memory()
            continue

        # Create attention lists from config
        channelatt_list, spatialatt_list = create_attention_lists(
            attention_type=att_config['type'],
            reduction_ratio=att_config['reduction_ratio'],
            kernel_size=att_config['kernel_size'],
            layer_positions=att_config['layer_positions'],
        )

        # Build student
        student = None
        try:
            # Cleanup before building student
            cleanup_memory()
            time.sleep(0.1)  # Shorter delay for CPU
            
            # Determine device based on config
            device = '/GPU:0' if USE_GPU_TRAIN_KD else '/CPU:0'
            
            # Build student with explicit device context
            with tf.device(device):
                student = build_deepconvlstm_att(
                    version="LM",
                    input_shape=input_shape,
                    num_classes=num_classes,
                    channelatt_list=channelatt_list,
                    spatialatt_list=spatialatt_list,
                    seed=MODEL_SEED,
                )
            
            cleanup_memory()
            time.sleep(0.05)  # Shorter delay for CPU
            
        except Exception as e:
            logger.error(f"ERROR building student: {e}", exc_info=True)
            cleanup_on_error(teacher)
            teacher = None
            time.sleep(0.1)  # Shorter delay on error
            continue

        # Determine attention_list and attention_layer for KD
        attention_list = []
        attention_layer = None
        if att_config['type'] == 'CH_ATT':
            attention_list = ['CH_ATT']
            attention_layer = att_config['layer_positions'][0] if att_config['layer_positions'] else 1
        elif att_config['type'] == 'SP_ATT':
            attention_list = ['SP_ATT']
            attention_layer = att_config['layer_positions'][0] if att_config['layer_positions'] else 1
        elif att_config['type'] == 'CBAM':
            attention_list = ['CBAM']
            attention_layer = att_config['layer_positions'][0] if att_config['layer_positions'] else 1

        # Train KD
        student_model = None
        try:
            # Aggressive cleanup before training
            cleanup_memory()
            time.sleep(0.1)

            student_model, training_info = train_kd(
                student=student,
                teacher=teacher,
                architecture="LM",
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                alpha=alpha,
                temperature=temp,
                beta=KD_BASELINE_BETA_ATT,
                batch_size=DEFAULT_BATCH_SIZE,
                epochs=KD_DEFAULT_EPOCHS,
                patience=KD_DEFAULT_PATIENCE,
                verbose=DEFAULT_VERBOSE,
                save_path=model_path,
                use_gpu=USE_GPU_TRAIN_KD,  # Explicitly pass config value
            )
            
            # Immediate cleanup after training
            cleanup_on_error(teacher, student)
            teacher = None
            student = None
            time.sleep(0.05)
            
            # Load best model and evaluate
            student_model = load_model_tf(f"{model_name}.h5")
            
            # Cleanup before evaluation
            cleanup_memory()
            
            # Evaluate on all sets
            train_acc, train_f1, train_prec, train_rec, train_cm = evaluate_tf_model(student_model, X_train, y_train, use_gpu=USE_GPU_EVALUATE)
            val_acc, val_f1, val_prec, val_rec, val_cm = evaluate_tf_model(student_model, X_val, y_val, use_gpu=USE_GPU_EVALUATE)
            test_acc, test_f1, test_prec, test_rec, test_cm = evaluate_tf_model(student_model, X_test, y_test, use_gpu=USE_GPU_EVALUATE)

            # Calculate parameter counts
            from src.grid_search_utils import get_model_param_counts
            param_total, param_trainable, param_non_trainable = get_model_param_counts(student_model)

            # Cleanup after evaluation
            cleanup_memory()
            
            # Save to CSV
            save_grid_search_result(
                csv_path=csv_path,
                model_name=model_name,
                hyperparams={
                    'temperature': temp,
                    'alpha': alpha,
                    'attention_type': att_config['type'],
                    'reduction_ratio': att_config['reduction_ratio'],
                    'kernel_size': att_config['kernel_size'],
                    'layer_positions': '-'.join(map(str, att_config['layer_positions'])),
                    'teacher_model': teacher_name,
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
                parameter_count=param_total,
                parameter_count_trainable=param_trainable,
                parameter_count_non_trainable=param_non_trainable,
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
            
            logger.info(f"✓ Val Acc: {val_acc:.4f}")
            
            # Aggressive cleanup after CSV save
            cleanup_on_error(student_model)
            student_model = None
            time.sleep(0.05)
            
        except Exception as e:
            logger.error(f"ERROR: {e}", exc_info=True)
            # Aggressive cleanup on error
            cleanup_on_error(teacher, student, student_model)
            teacher = None
            student = None
            student_model = None
            time.sleep(0.1)
            continue
    
    # Final cleanup (outside the for loop)
    cleanup_memory()
    
    logger.info("=== Grid Search Complete ===")
    logger.info(f"Results saved to: {csv_path}")
    logger.info(f"Models saved to: {TF_DIR}")


if __name__ == "__main__":
    main()