"""
Central configuration for reproducibility.

All seeds and random states are defined here to ensure consistency
across all experiments and grid searches.
"""

# Data split random state (used in data_opportunity.py)
DATA_RANDOM_STATE = 42

# Model initialization seed (used in model building)
MODEL_SEED = 42

# Training random seeds (TensorFlow and NumPy)
TRAINING_TF_SEED = 42
TRAINING_NP_SEED = 42

# Training hyperparameters
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 300
DEFAULT_PATIENCE = 10
DEFAULT_LEARNING_RATE = 0.001

# KD training hyperparameters
KD_DEFAULT_EPOCHS = 300
KD_DEFAULT_PATIENCE = 10

# GPU/CPU Device Configuration
# Set to True to use GPU for training, False for CPU
USE_GPU_TRAIN = True  # For all training (OM, LM, MM)
USE_GPU_TRAIN_KD = False  # For knowledge distillation training (separate due to Metal GPU issues)

# Set to False to use CPU for evaluation (useful for speed tests, memory profiling)
USE_GPU_EVALUATE = False

# Grid Search Hyperparameters

# Attention Grid Search (Chapter 4 & 5)
ATT_GRID_CHANNEL_RATIOS = [2, 4, 8]
ATT_GRID_SPATIAL_KERNELS = [3, 5, 7]
ATT_GRID_LAYER_POSITIONS = [1, 2, 3, 4]

# Knowledge Distillation Grid Search (Chapter 6)
KD_GRID_TEMPERATURES = [1, 2, 4, 6, 8, 10, 15]
KD_GRID_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# KD Baseline Hyperparameters
KD_BASELINE_BETA = 0.0
KD_BASELINE_BETA_ATT = 1.0

# Training Verbosity
DEFAULT_VERBOSE = 0
KD_DEFAULT_VERBOSE = 2

# Compression Hyperparameters (Chapter 7)
COMPRESSION_CP_SPARSITY = 0.5  # Constant Pruning sparsity (50%)
COMPRESSION_PDP_SPARSITY = 0.8  # Polynomial Decay Pruning sparsity (80%)