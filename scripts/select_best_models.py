"""
Select best models from grid search results.

Reads grid search CSV files and selects the best model for each variant
based on validation accuracy. Copies best models to canonical names and
saves summary to results/best_models.csv.
"""

import os
import shutil
import csv
from typing import Dict, Optional

from src.grid_search_utils import find_best_model_from_csv, STANDARD_CSV_FIELDNAMES, get_active_device
from src.registry import TF_DIR
from src.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)


def copy_best_model(source_name: str, dest_name: str):
    """Copy a model file to a new name."""
    source_path = os.path.join(TF_DIR, f"{source_name}.h5")
    dest_path = os.path.join(TF_DIR, f"{dest_name}.h5")
    
    if not os.path.exists(source_path):
        logger.warning(f"Source model not found: {source_path}")
        return False
    
    shutil.copy2(source_path, dest_path)
    logger.info(f"Copied: {source_name}.h5 → {dest_name}.h5")
    return True


def save_best_models_summary(csv_path: str, best_models: Dict[str, Dict]):
    """Save best models summary to CSV using STANDARD_CSV_FIELDNAMES order."""
    # Use STANDARD_CSV_FIELDNAMES as base, but replace 'model_name' with 'best_model_name'
    # Keep the same order as STANDARD_CSV_FIELDNAMES for consistency
    fieldnames = []
    for field in STANDARD_CSV_FIELDNAMES:
        if field == 'model_name':
            fieldnames.append('best_model_name')  # Replace model_name with best_model_name
        else:
            fieldnames.append(field)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for variant, model_info in best_models.items():
            if model_info:
                # Get device from model_info or current device
                device = model_info.get('device') or get_active_device()
                
                # Build row following STANDARD_CSV_FIELDNAMES order
                row = {}
                for field in STANDARD_CSV_FIELDNAMES:
                    if field == 'model_name':
                        # Use best_model_name instead
                        row['best_model_name'] = model_info.get('model_name', '')
                    else:
                        # Copy value from model_info if available
                        value = model_info.get(field)
                        if value is not None and value != '':
                            row[field] = value
                        else:
                            row[field] = None
                
                # Ensure variant is set (from the loop key)
                row['variant'] = variant
                
                # Ensure device is set
                if not row.get('device'):
                    row['device'] = device
                
                # Fill missing fields with None (shouldn't happen, but safety check)
                for field in fieldnames:
                    if field not in row:
                        row[field] = None
                
                writer.writerow(row)


def main():
    logger.info("=== Best Model Selection ===")
    
    # Grid search CSV files - Chapter 4 & 5
    grid_search_files_ch45 = {
        'OM-Att': 'results/grid_search_om_att.csv',
        'LM-Att': 'results/grid_search_lm_att.csv',
        'MM-Att': 'results/grid_search_mm_att.csv',
    }
    
    # Grid search CSV files - Chapter 6
    grid_search_files_ch6 = {
        'LM-RB-KD': 'results/grid_search_rbkd.csv',
        'LM-RB-KD-Att': 'results/grid_search_rbkd_att.csv',
        'LM-RAB-KD-Att': 'results/grid_search_rabkd_att.csv',
    }
    
    # Best model names - Chapter 4 & 5
    best_model_names_ch45 = {
        'OM-Att': 'Best-OM-Att',
        'LM-Att': 'Best-LM-Att',
        'MM-Att': 'Best-MM-Att',
    }
    
    # Best model names - Chapter 6
    best_model_names_ch6 = {
        'LM-RB-KD': 'Best-LM-RB-KD',
        'LM-RB-KD-Att': 'Best-LM-RB-KD-Att',
        'LM-RAB-KD-Att': 'Best-LM-RAB-KD-Att',
    }
    
    # Combine all
    grid_search_files = {**grid_search_files_ch45, **grid_search_files_ch6}
    best_model_names = {**best_model_names_ch45, **best_model_names_ch6}
    
    best_models = {}
    
    # Find best models
    logger.info("Finding best models from grid search results...")
    logger.info("[Chapter 4 & 5 - Attention Models]")
    for variant, csv_path in grid_search_files_ch45.items():
        logger.info(f"[{variant}]")
        
        if not os.path.exists(csv_path):
            logger.warning(f"Grid search CSV not found: {csv_path}")
            logger.warning("→ Run grid search script first")
            best_models[variant] = None
            continue
        
        best_model = find_best_model_from_csv(csv_path, metric='val_accuracy')
        
        if not best_model:
            logger.warning(f"No models found in CSV: {csv_path}")
            best_models[variant] = None
            continue
        
        model_name = best_model.get('model_name', '')
        val_acc = best_model.get('val_accuracy', 'N/A')
        
        logger.info(f"Best model: {model_name}")
        logger.info(f"Validation Accuracy: {val_acc}")
        
        # Copy to canonical name
        best_name = best_model_names[variant]
        if copy_best_model(model_name, best_name):
            best_models[variant] = best_model
        else:
            best_models[variant] = None
    
    logger.info("[Chapter 6 - Knowledge Distillation Models]")
    for variant, csv_path in grid_search_files_ch6.items():
        logger.info(f"[{variant}]")
        
        if not os.path.exists(csv_path):
            logger.warning(f"Grid search CSV not found: {csv_path}")
            logger.warning("→ Run grid search script first")
            best_models[variant] = None
            continue
        
        best_model = find_best_model_from_csv(csv_path, metric='val_accuracy')
        
        if not best_model:
            logger.warning(f"No models found in CSV: {csv_path}")
            best_models[variant] = None
            continue
        
        model_name = best_model.get('model_name', '')
        val_acc = best_model.get('val_accuracy', 'N/A')
        
        logger.info(f"Best model: {model_name}")
        logger.info(f"Validation Accuracy: {val_acc}")
        
        # Copy to canonical name
        best_name = best_model_names[variant]
        if copy_best_model(model_name, best_name):
            best_models[variant] = best_model
        else:
            best_models[variant] = None
    
    # Save summary
    os.makedirs("results", exist_ok=True)
    summary_path = "results/best_models.csv"
    
    logger.info("=== Saving Summary ===")
    save_best_models_summary(summary_path, best_models)
    logger.info(f"Summary saved to: {summary_path}")
    
    # Print summary
    logger.info("=== Best Models Summary ===")
    logger.info("Chapter 4 & 5:")
    for variant in grid_search_files_ch45.keys():
        model_info = best_models.get(variant)
        if model_info:
            logger.info(f"  {variant}: {model_info.get('model_name', 'N/A')} (Val Acc: {model_info.get('val_accuracy', 'N/A')})")
        else:
            logger.warning(f"  {variant}: Not found")
    
    logger.info("Chapter 6:")
    for variant in grid_search_files_ch6.keys():
        model_info = best_models.get(variant)
        if model_info:
            logger.info(f"  {variant}: {model_info.get('model_name', 'N/A')} (Val Acc: {model_info.get('val_accuracy', 'N/A')})")
        else:
            logger.warning(f"  {variant}: Not found")
    
    logger.info("=== Done ===")
    logger.info("Best models are saved as:")
    for variant, best_name in best_model_names.items():
        if best_models.get(variant):
            logger.info(f"  {best_name}.h5")

if __name__ == "__main__":
    main()