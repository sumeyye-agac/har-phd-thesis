# Boosting Lightweight Human Activity Recognition on Edge Devices With Knowledge Distillation and Attention Mechanisms

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)](https://www.tensorflow.org/)

> **PhD Thesis Implementation**  
> Author: **Sumeyye Agac**  
> Advisor: **Assoc. Prof. Ozlem Durmaz Incel**  
> University: **Bogazici University, Istanbul, Turkey**  
> Program: **PhD Program in Computer Engineering**

---

## TL;DR

This repository provides a chapter-by-chapter reference implementation of a PhD thesis on improving lightweight human activity recognition models for edge devices using attention mechanisms, knowledge distillation, and model compression.

**Current Implementation:**
- ✅ DeepConvLSTM architecture (OM, LM, MM variants)
- ✅ Opportunity dataset

**Future Extensions:**
- 🔄 SqueezeNet architecture integration
- 🔄 Additional datasets (WISDM, SENSORS, etc.)

---

## Requirements

- Python 3.11+
- TensorFlow 2.13.0
- See `requirements.txt` for full dependencies

---

## Thesis Chapter Mapping

- Chapter 4 – Attention Models  
- Chapter 5 – Lightweight Models + Attention  
- Chapter 6 – Knowledge Distillation + Attention  
- Chapter 7 – Model Compression

---

## Abstract (Short Version)

Sensor-based Human Activity Recognition (HAR) has become an essential component of ubiquitous and mobile computing applications. However, deploying deep learning models on edge devices requires architectures that are both accurate and computationally efficient.

This dissertation improves lightweight HAR models using four complementary techniques:

1. **Attention mechanisms** (Channel, Spatial, Channel–Spatial)  
2. **Lightweight deep architectures** (LM, MM, OM variants)  
3. **Knowledge distillation** (RB-KD, RAB-KD-Att, RB-KD-Att)  
4. **Model compression** (quantization and pruning)

The combined use of attention and knowledge distillation significantly improves the performance of lightweight models, while compression enables deployment on resource-constrained hardware with minimal accuracy degradation.

---

## Model Naming Conventions

Models follow a consistent naming pattern: `{Variant}-Att-{Type}-{Params}-{Layers}` for attention models, `{Variant}-{KD-Type}-T{Temp}-A{Alpha}-{AttConfig}` for KD models, and `OM-{CompressionType}` for compression models. See `src/model_naming.py` for details.

---

## Repository Structure
```
./
├── data/
│   ├── raw/                         # Opportunity dataset (automatically downloaded)
│   └── processed/                   # Canonical train/validation/test split
│
├── models/                          # Model definitions
│   ├── __init__.py
│   ├── base_deepconvlstm.py         # DeepConvLSTM base (OM/MM/LM variants)
│   ├── att_models.py                # Attention-based models
│   └── attention_layers.py          # Channel & Spatial Attention layers
│
├── src/                             # Shared utilities
│   ├── config.py                    # Centralized configuration (seeds, hyperparameters)
│   ├── cli_parser.py                # Command-line argument parser for grid search scripts
│   ├── attention_utils.py           # Attention mechanism utilities
│   ├── memory_utils.py              # Memory management utilities
│   ├── gpu_utils.py                 # GPU environment setup utilities
│   ├── data_opportunity.py          # Data loading and preprocessing
│   ├── model_io.py                  # Model save/load utilities
│   ├── model_naming.py              # Model naming conventions
│   ├── registry.py                  # Model registry (canonical names/paths)
│   ├── training_tf.py               # TensorFlow training utilities
│   ├── training_kd.py               # Knowledge Distillation training
│   ├── evaluation_tf.py             # TF model evaluation
│   ├── evaluation_tflite.py         # TFLite model evaluation
│   ├── grid_search_utils.py         # Grid search helpers & CSV management
│   ├── distillation.py              # Distiller Keras Model
│   └── utils_resources.py           # Resource metrics (FLOPs, MACs, model size)
│
├── scripts/                         # Runnable scripts (Chapter 4-7)
│   ├── ch4_train_om.py              # Chapter 4: Train baseline OM
│   ├── ch4_grid_search_om_att.py    # Chapter 4: OM-Att grid search
│   ├── ch5_train_lm.py              # Chapter 5: Train baseline LM
│   ├── ch5_train_mm.py              # Chapter 5: Train baseline MM
│   ├── ch5_grid_search_lm_att.py    # Chapter 5: LM-Att grid search
│   ├── ch5_grid_search_mm_att.py    # Chapter 5: MM-Att grid search
│   ├── ch6_grid_search_rbkd.py      # Chapter 6: LM-RB-KD grid search
│   ├── ch6_grid_search_rbkd_att.py  # Chapter 6: LM-RB-KD-Att grid search
│   ├── ch6_grid_search_rabkd_att.py # Chapter 6: LM-RAB-KD-Att grid search
│   ├── ch7_compress_om.py           # Chapter 7: OM compression (TFLite, pruning)
│   ├── select_best_models.py        # Select best models from grid searches
│   ├── evaluate_baseline_models.py  # Evaluate baseline models (OM, LM, MM)
│   └── evaluate_all.py              # Evaluate all trained models (TF & TFLite)
│
├── models_saved/                    # Trained models
│   ├── tf/                          # TensorFlow models (.h5)
│   └── tflite/                      # TFLite models (.tflite)
│
├── results/                         # Evaluation results (CSV files)
│
├── requirements.txt                 # Python dependencies
├── run_all.sh                       # Automated execution script
└── README.md                        # Project documentation
```

---

## Running the Project Locally

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd har-phd-thesis
   ```
2. **Create virtual environment (recommended)**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the full pipeline**
   ```bash
   chmod +x run_all.sh
   ./run_all.sh
   ```

**Note:** The Opportunity dataset will be automatically downloaded to `data/raw/` on first run.

---

## Running Individual Chapters

All grid search scripts support command-line parameters. Use `--help` for details. Default behavior runs full grid searches with all hyperparameter combinations from `src/config.py`.

**Parameter Format:**
- Boolean: `ch_att=true`, `sp_att=false`
- Lists: `channel_ratios=2,4,8`, `temperatures=1,2,4` (no spaces around commas)

### Chapter 4: Attention Models

```bash
# Train baseline OM
python scripts/ch4_train_om.py

# OM-Att grid search (default: 60 models - 12 CH + 12 SP + 36 CBAM)
python scripts/ch4_grid_search_om_att.py

# Custom: Only Channel Attention with specific ratios
python scripts/ch4_grid_search_om_att.py ch_att=true sp_att=false cbam=false \
    channel_ratios=2,4 layer_positions=1,2
```

### Chapter 5: Lightweight + Attention

```bash
# Train baseline models
python scripts/ch5_train_lm.py
python scripts/ch5_train_mm.py

# LM-Att grid search (default: 60 models)
python scripts/ch5_grid_search_lm_att.py

# MM-Att grid search (default: 60 models)
python scripts/ch5_grid_search_mm_att.py
```

### Chapter 6: Knowledge Distillation

```bash
# RB-KD grid search (default: 63 models - 7 temps × 9 alphas)
python scripts/ch6_grid_search_rbkd.py

# RB-KD-Att grid search (default: 3,780 models)
python scripts/ch6_grid_search_rbkd_att.py

# RAB-KD-Att grid search (default: 3,780 models)
# Note: Requires pre-trained OM-Att models from Chapter 4
python scripts/ch6_grid_search_rabkd_att.py
```

### Chapter 7: Model Compression

```bash
# Compress OM model (TFLite, quantization, pruning)
python scripts/ch7_compress_om.py
```

### Evaluation Scripts

```bash
python scripts/evaluate_baseline_models.py  # Baseline models
python scripts/evaluate_all.py              # All models (TF & TFLite)
python scripts/select_best_models.py        # Best model selection
```

---

## Re-running Experiments

All scripts are idempotent. To force a clean run:
```bash
# Delete all trained models
rm -rf models_saved/tf/*
rm -rf models_saved/tflite/*

# Delete processed data (will be regenerated)
rm data/processed/opportunity_splits.pkl
```

---

## Results

All evaluation results are saved to CSV files in the `results/` directory:

- `baseline_models.csv` - Baseline model evaluations
- `grid_search_om_att.csv` - OM-Att grid search results
- `grid_search_lm_att.csv` - LM-Att grid search results
- `grid_search_mm_att.csv` - MM-Att grid search results
- `grid_search_rbkd.csv` - RB-KD grid search results
- `grid_search_rbkd_att.csv` - RB-KD-Att grid search results
- `grid_search_rabkd_att.csv` - RAB-KD-Att grid search results
- `best_models_summary.csv` - Best model selections
- `evaluation_all.csv` - Complete evaluation of all models

---

## Citation

If you use this repository, models, or experimental setup, please cite the PhD thesis:

**APA Style:**  
Agac, S. *Boosting Lightweight Human Activity Recognition on Edge Devices With Knowledge Distillation and Attention Mechanisms* (PhD Dissertation). Bogazici University.

```bibtex
@phdthesis{agac_har,  
  title       = {Boosting Lightweight Human Activity Recognition on Edge Devices With Knowledge Distillation and Attention Mechanisms},  
  author      = {Agac, Sumeyye},  
  school      = {Bogazici University},  
  address     = {Istanbul, Turkey},  
  advisor     = {Ozlem Durmaz Incel},  
  program     = {PhD Program in Computer Engineering}  
}
```

