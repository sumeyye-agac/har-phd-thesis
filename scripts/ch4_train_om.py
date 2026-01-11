"""
Chapter 4 – Train/Evaluate Original DeepConvLSTM (OM) model on Opportunity dataset.

Idempotent behavior:
- If models_saved/tf/OM.h5 exists -> load and evaluate (no training).
- Else -> train, save, then load best and evaluate.
"""

import numpy as np
import tensorflow as tf
from models.base_deepconvlstm import build_deepconvlstm_om
from src.logger import setup_logger
from src.gpu_utils import setup_gpu_and_log_device
from src.training_utils import train_or_evaluate_baseline_model

# Setup logger
logger = setup_logger(__name__)

def main():
    # Setup GPU and log device info
    setup_gpu_and_log_device(logger)
    
    # Train or evaluate OM model
    train_or_evaluate_baseline_model(
        variant="OM",
        logger=logger,
        build_model_func=build_deepconvlstm_om,
        build_model_kwargs={},  # input_shape and num_classes will be added automatically
        chapter_title="Chapter 4: Training Original Model (OM)",
    )


if __name__ == "__main__":
    main()