#!/usr/bin/env bash
# run_all.sh - Automated grid search execution

set -euo pipefail

# Set PYTHONPATH to include project root
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Log file
LOG_FILE="run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== HAR PhD Thesis (Opportunity) — End-to-End Run ==="
echo "Log file: $LOG_FILE"
echo "Started at: $(date)"
echo

# ============================================
# Baseline Models (No Attention)
# ============================================
echo "[Baseline] Training baseline models (OM, LM, MM)..."
python scripts/ch4_train_om.py || { echo "⚠ Error in ch4_train_om.py"; exit 1; }
python scripts/ch5_train_lm.py || { echo "⚠ Error in ch5_train_lm.py"; exit 1; }
python scripts/ch5_train_mm.py || { echo "⚠ Error in ch5_train_mm.py"; exit 1; }
echo

# ============================================
# Grid Search - Chapter 4 & 5
# ============================================
echo "[Grid Search Ch4&5] Attention hyperparameter grid search..."
echo "  Note: This will train 180 models (60 OM-Att + 60 LM-Att + 60 MM-Att)"
echo "  Auto-running grid search..."
python scripts/ch4_grid_search_om_att.py || { echo "⚠ Error in ch4_grid_search_om_att.py"; exit 1; }
python scripts/ch5_grid_search_lm_att.py || { echo "⚠ Error in ch5_grid_search_lm_att.py"; exit 1; }
python scripts/ch5_grid_search_mm_att.py || { echo "⚠ Error in ch5_grid_search_mm_att.py"; exit 1; }
echo

echo "[Best Models Ch4&5] Selecting best attention models..."
python scripts/select_best_models.py || { echo "⚠ Error in select_best_models.py"; exit 1; }
echo

# ============================================
# Grid Search - Chapter 6
# ============================================
echo "[Grid Search Ch6] Knowledge Distillation hyperparameter grid search..."
echo "  Note: This will train 7,623 models (63 RB-KD + 3,780 RB-KD-Att + 3,780 RAB-KD-Att)"
echo "  Auto-running grid search..."
python scripts/ch6_grid_search_rbkd.py || { echo "⚠ Error in ch6_grid_search_rbkd.py"; exit 1; }
python scripts/ch6_grid_search_rbkd_att.py || { echo "⚠ Error in ch6_grid_search_rbkd_att.py"; exit 1; }
python scripts/ch6_grid_search_rabkd_att.py || { echo "⚠ Error in ch6_grid_search_rabkd_att.py"; exit 1; }
echo

echo "[Best Models Ch6] Selecting best KD models..."
python scripts/select_best_models.py || { echo "⚠ Error in select_best_models.py"; exit 1; }
echo

# ============================================
# Evaluation
# ============================================
echo "[Evaluation] Evaluating all models..."
python scripts/evaluate_baseline_models.py || { echo "⚠ Error in evaluate_baseline_models.py"; exit 1; }
python scripts/evaluate_all.py || { echo "⚠ Error in evaluate_all.py"; exit 1; }
echo

# ============================================
# Compression
# ============================================
echo "[Compression] Chapter 7 — Compression (OM only)"
python scripts/ch7_compress_om.py || { echo "⚠ Error in ch7_compress_om.py"; exit 1; }
echo

echo "=== DONE ==="
echo "Completed at: $(date)"
echo "TF models:     models_saved/tf/"
echo "TFLite models: models_saved/tflite/"
echo "Results:       results/"
echo "Log file:      $LOG_FILE"