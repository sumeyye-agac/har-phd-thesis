"""
Chapter 5 – Train/Evaluate Lightweight (LM) DeepConvLSTM model on Opportunity dataset.

Idempotent:
- If models_saved/tf/LM.h5 exists -> load and evaluate.
- Else -> train, save, load best and evaluate.
"""

import numpy as np
import tensorflow as tf
from models.base_deepconvlstm import build_deepconvlstm
from src.logger import setup_logger
from src.gpu_utils import setup_gpu_and_log_device
from src.training_utils import train_or_evaluate_baseline_model
from src.config import MODEL_SEED

# Setup logger
logger = setup_logger(__name__)

def main():
    # Setup GPU and log device info
    setup_gpu_and_log_device(logger)
    
    # Train or evaluate LM model
    train_or_evaluate_baseline_model(
        variant="LM",
        logger=logger,
        build_model_func=build_deepconvlstm,
        build_model_kwargs={'variant': 'LM', 'seed': MODEL_SEED},
        chapter_title="Chapter 5: Training Lightweight Model (LM)",
    )


if __name__ == "__main__":
    main()