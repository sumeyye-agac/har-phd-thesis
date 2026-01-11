#!/bin/bash

# Custom script execution - modify this file to run your desired experiments
# Works both locally and in Google Colab

# Set PYTHONPATH (works in both local and Colab)
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

# Activate virtual environment if it exists (local only, skip in Colab)
if [ -f "har-phd-thesis-env/bin/activate" ]; then
    source har-phd-thesis-env/bin/activate
    echo "✓ Virtual environment activated"
fi

echo "=== Starting custom script execution ==="
echo ""

# Chapter 4: Train OM baseline
echo ">>> [1/12] Training OM baseline..."
python scripts/ch4_train_om.py
if [ $? -ne 0 ]; then
    echo "ERROR: ch4_train_om.py failed"
    exit 1
fi
echo ""

# Chapter 4: OM-Att (1 model)
echo ">>> [2/12] Running OM-Att grid search (1 model)..."
python scripts/ch4_grid_search_om_att.py ch_att=true sp_att=false cbam=false channel_ratios=4 layer_positions=1
if [ $? -ne 0 ]; then
    echo "ERROR: ch4_grid_search_om_att.py failed"
    exit 1
fi
echo ""

# Chapter 5: LM-Att (1 model)
echo ">>> [3/12] Running LM-Att grid search (1 model)..."
python scripts/ch5_grid_search_lm_att.py ch_att=true sp_att=false cbam=false channel_ratios=4 layer_positions=1
if [ $? -ne 0 ]; then
    echo "ERROR: ch5_grid_search_lm_att.py failed"
    exit 1
fi
echo ""

# Chapter 5: MM-Att (1 model)
echo ">>> [4/12] Running MM-Att grid search (1 model)..."
python scripts/ch5_grid_search_mm_att.py ch_att=true sp_att=false cbam=false channel_ratios=4 layer_positions=1
if [ $? -ne 0 ]; then
    echo "ERROR: ch5_grid_search_mm_att.py failed"
    exit 1
fi
echo ""

# Chapter 5: Train LM baseline
echo ">>> [5/12] Training LM baseline..."
python scripts/ch5_train_lm.py
if [ $? -ne 0 ]; then
    echo "ERROR: ch5_train_lm.py failed"
    exit 1
fi
echo ""

# Chapter 5: Train MM baseline
echo ">>> [6/12] Training MM baseline..."
python scripts/ch5_train_mm.py
if [ $? -ne 0 ]; then
    echo "ERROR: ch5_train_mm.py failed"
    exit 1
fi
echo ""

# Chapter 6: RB-KD (1 model)
echo ">>> [7/12] Running RB-KD grid search (1 model)..."
python scripts/ch6_grid_search_rbkd.py temperatures=4 alphas=0.5
if [ $? -ne 0 ]; then
    echo "ERROR: ch6_grid_search_rbkd.py failed"
    exit 1
fi
echo ""

# Chapter 6: RB-KD-Att (1 model)
echo ">>> [8/12] Running RB-KD-Att grid search (1 model)..."
python scripts/ch6_grid_search_rbkd_att.py ch_att=true sp_att=false cbam=false temperatures=4 alphas=0.5 channel_ratios=4 layer_positions=1
if [ $? -ne 0 ]; then
    echo "ERROR: ch6_grid_search_rbkd_att.py failed"
    exit 1
fi
echo ""

# Chapter 6: RAB-KD-Att (1 model)
echo ">>> [9/12] Running RAB-KD-Att grid search (1 model)..."
python scripts/ch6_grid_search_rabkd_att.py ch_att=true sp_att=false cbam=false temperatures=4 alphas=0.5 channel_ratios=4 layer_positions=1
if [ $? -ne 0 ]; then
    echo "ERROR: ch6_grid_search_rabkd_att.py failed"
    exit 1
fi
echo ""

# Select best models from all grid searches (Ch4, Ch5, Ch6)
echo ">>> [10/12] Selecting best models from all grid searches..."
python scripts/select_best_models.py
if [ $? -ne 0 ]; then
    echo "ERROR: select_best_models.py failed"
    exit 1
fi
echo ""

# Chapter 7: Compress OM
echo ">>> [11/12] Compressing OM model..."
python scripts/ch7_compress_om.py
if [ $? -ne 0 ]; then
    echo "ERROR: ch7_compress_om.py failed"
    exit 1
fi
echo ""

# Evaluation
echo ">>> [12/12] Evaluating all models..."
python scripts/evaluate_all.py
if [ $? -ne 0 ]; then
    echo "ERROR: evaluate_all.py failed"
    exit 1
fi
echo ""

echo "=== All scripts completed successfully! ==="
