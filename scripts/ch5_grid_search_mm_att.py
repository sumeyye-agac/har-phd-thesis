"""
Chapter 5 — MM-Att Grid Search

Grid search over attention hyperparameters:
- Channel Attention: reduction_ratio [2, 4, 8] × 4 layer positions = 12 models
- Spatial Attention: kernel_size [3, 5, 7] × 4 layer positions = 12 models
- CBAM: reduction_ratio [2, 4, 8] × kernel_size [3, 5, 7] × 4 layer positions = 36 models

Total: 60 models

All models are saved and results are written to results/grid_search_mm_att.csv
"""

import os
import numpy as np
import tensorflow as tf
from itertools import product

from src.data_opportunity import load_opportunity_splits, load_and_prepare_opportunity_data
from src.grid_search_utils import (
    set_seeds, save_grid_search_result, get_active_device, run_single_experiment,
)
from src.registry import TF_DIR
from src.config import (
    ATT_GRID_CHANNEL_RATIOS, ATT_GRID_SPATIAL_KERNELS, ATT_GRID_LAYER_POSITIONS,
)
from src.cli_parser import parse_args
from src.memory_utils import periodic_cleanup, cleanup_memory
from src.logger import setup_logger
from src.gpu_utils import setup_gpu_and_log_device

# Setup logger
logger = setup_logger(__name__)

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
MM-Att Grid Search - Usage

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
  python ch5_grid_search_mm_att.py ch_att=true sp_att=false cbam=false
  
  # Run ch_att with only ratio 2 and 4, on layers 1 and 2
  python ch5_grid_search_mm_att.py ch_att=true sp_att=false cbam=false channel_ratios=2,4 layer_positions=1,2
  
  # Run all (default)
  python ch5_grid_search_mm_att.py
  
  # Case-insensitive (all work the same)
  python ch5_grid_search_mm_att.py CH_ATT=true ch_att=false Ch_Att=true

Note: All parameters are case-insensitive. Invalid parameters or values will stop execution.
""",
    'script_name': 'ch5_grid_search_mm_att.py'
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
    
    logger.info("=== Chapter 5: MM-Att Grid Search ===")
    
    # Calculate total experiments based on selected attention types
    ch_att_count = len(channel_ratios) * len(layer_positions_options) if run_ch_att else 0
    sp_att_count = len(spatial_kernels) * len(layer_positions_options) if run_sp_att else 0
    cbam_count = len(channel_ratios) * len(spatial_kernels) * len(layer_positions_options) if run_cbam else 0
    total_experiments = ch_att_count + sp_att_count + cbam_count
    
    logger.info(f"Total experiments: {total_experiments} ({ch_att_count} CH + {sp_att_count} SP + {cbam_count} CBAM)")
    logger.info(f"Configuration: ch_att={run_ch_att}, sp_att={run_sp_att}, cbam={run_cbam}")
    logger.info(f"  channel_ratios={channel_ratios}, spatial_kernels={spatial_kernels}, layer_positions={layer_positions_options}")

    # Setup GPU and log device info
    setup_gpu_and_log_device(logger)
    
    # Load and prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, input_shape, num_classes = load_and_prepare_opportunity_data(logger)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    csv_path = "results/grid_search_mm_att.csv"
    
    logger.info(f"Starting grid search: {total_experiments} experiments")
    logger.info(f"Results will be saved to: {csv_path}")
    
    experiment_count = 0
    
    # Channel Attention Grid Search
    if run_ch_att:
        ch_att_experiments = len(channel_ratios) * len(layer_positions_options)
        logger.info(f"[1/3] Channel Attention Grid Search ({ch_att_experiments} experiments)")
        for ratio, layer_pos in product(channel_ratios, layer_positions_options):
            layers = [layer_pos]
            experiment_count += 1
            run_single_experiment(
            variant="MM",
            attention_type="CH_ATT",
            reduction_ratio=ratio,
            layer_positions=layers,
            input_shape=input_shape,
            num_classes=num_classes,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            csv_path=csv_path,
            experiment_num=experiment_count,
            total_experiments=total_experiments,
            logger=logger,
        )

        # Periodic cleanup every 20 experiments
        periodic_cleanup(experiment_count, interval=20, verbose=True)
    else:
        logger.info("[1/3] Channel Attention Grid Search - SKIPPED (ch_att=false)")
    
    # Spatial Attention Grid Search
    if run_sp_att:
        sp_att_experiments = len(spatial_kernels) * len(layer_positions_options)
        logger.info(f"[2/3] Spatial Attention Grid Search ({sp_att_experiments} experiments)")
        for kernel, layer_pos in product(spatial_kernels, layer_positions_options):
            layers = [layer_pos]
            experiment_count += 1
            run_single_experiment(
            variant="MM",
            attention_type="SP_ATT",
            kernel_size=kernel,
            layer_positions=layers,
            input_shape=input_shape,
            num_classes=num_classes,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            csv_path=csv_path,
            experiment_num=experiment_count,
            total_experiments=total_experiments,
            logger=logger,
        )

        # Periodic cleanup every 20 experiments
        periodic_cleanup(experiment_count, interval=20, verbose=True)
    else:
        logger.info("[2/3] Spatial Attention Grid Search - SKIPPED (sp_att=false)")
    
    # CBAM Grid Search
    if run_cbam:
        cbam_experiments = len(channel_ratios) * len(spatial_kernels) * len(layer_positions_options)
        logger.info(f"[3/3] CBAM Grid Search ({cbam_experiments} experiments)")
        for ratio, kernel, layer_pos in product(channel_ratios, spatial_kernels, layer_positions_options):
            layers = [layer_pos]
            experiment_count += 1
            run_single_experiment(
            variant="MM",
            attention_type="CBAM",
            reduction_ratio=ratio,
            kernel_size=kernel,
            layer_positions=layers,
            input_shape=input_shape,
            num_classes=num_classes,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            csv_path=csv_path,
            experiment_num=experiment_count,
            total_experiments=total_experiments,
            logger=logger,
        )

        # Periodic cleanup every 20 experiments
        periodic_cleanup(experiment_count, interval=20, verbose=True)
    else:
        logger.info("[3/3] CBAM Grid Search - SKIPPED (cbam=false)")
    
    # Final cleanup
    cleanup_memory()
    
    logger.info("=== Grid Search Complete ===")
    logger.info(f"Results saved to: {csv_path}")
    logger.info(f"Models saved to: {TF_DIR}")


if __name__ == "__main__":
    main()