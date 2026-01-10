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


def copy_best_model(source_name: str, dest_name: str):
    """Copy a model file to a new name."""
    source_path = os.path.join(TF_DIR, f"{source_name}.h5")
    dest_path = os.path.join(TF_DIR, f"{dest_name}.h5")
    
    if not os.path.exists(source_path):
        print(f"  ⚠ Warning: Source model not found: {source_path}")
        return False
    
    shutil.copy2(source_path, dest_path)
    print(f"  ✓ Copied: {source_name}.h5 → {dest_name}.h5")
    return True


def save_best_models_summary(csv_path: str, best_models: Dict[str, Dict]):
    """Save best models summary to CSV using STANDARD_CSV_FIELDNAMES."""
    # Use STANDARD_CSV_FIELDNAMES as base, but keep 'variant' and 'best_model_name' for clarity
    fieldnames = ['variant', 'best_model_name'] + [f for f in STANDARD_CSV_FIELDNAMES if f not in ['variant', 'model_name']]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for variant, model_info in best_models.items():
            if model_info:
                # Get device from model_info or current device
                device = model_info.get('device') or get_active_device()
                
                # Start with variant and best_model_name
                row = {
                    'variant': variant,
                    'best_model_name': model_info.get('model_name', ''),
                }
                
                # Copy all standard fields from model_info
                for field in STANDARD_CSV_FIELDNAMES:
                    if field != 'model_name':  # Skip model_name, we use best_model_name instead
                        value = model_info.get(field)
                        if value is not None and value != '':
                            row[field] = value
                
                # Ensure device is set
                if 'device' not in row or not row['device']:
                    row['device'] = device
                
                # Fill missing fields with None
                for field in fieldnames:
                    if field not in row:
                        row[field] = None
                
                writer.writerow(row)


def main():
    print("=== Best Model Selection ===")
    print()
    
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
    print("Finding best models from grid search results...")
    print("\n[Chapter 4 & 5 - Attention Models]")
    for variant, csv_path in grid_search_files_ch45.items():
        print(f"\n[{variant}]")
        
        if not os.path.exists(csv_path):
            print(f"  ⚠ Grid search CSV not found: {csv_path}")
            print(f"  → Run grid search script first")
            best_models[variant] = None
            continue
        
        best_model = find_best_model_from_csv(csv_path, metric='val_accuracy')
        
        if not best_model:
            print(f"  ⚠ No models found in CSV: {csv_path}")
            best_models[variant] = None
            continue
        
        model_name = best_model.get('model_name', '')
        val_acc = best_model.get('val_accuracy', 'N/A')
        
        print(f"  Best model: {model_name}")
        print(f"  Validation Accuracy: {val_acc}")
        
        # Copy to canonical name
        best_name = best_model_names[variant]
        if copy_best_model(model_name, best_name):
            best_models[variant] = best_model
        else:
            best_models[variant] = None
    
    print("\n[Chapter 6 - Knowledge Distillation Models]")
    for variant, csv_path in grid_search_files_ch6.items():
        print(f"\n[{variant}]")
        
        if not os.path.exists(csv_path):
            print(f"  ⚠ Grid search CSV not found: {csv_path}")
            print(f"  → Run grid search script first")
            best_models[variant] = None
            continue
        
        best_model = find_best_model_from_csv(csv_path, metric='val_accuracy')
        
        if not best_model:
            print(f"  ⚠ No models found in CSV: {csv_path}")
            best_models[variant] = None
            continue
        
        model_name = best_model.get('model_name', '')
        val_acc = best_model.get('val_accuracy', 'N/A')
        
        print(f"  Best model: {model_name}")
        print(f"  Validation Accuracy: {val_acc}")
        
        # Copy to canonical name
        best_name = best_model_names[variant]
        if copy_best_model(model_name, best_name):
            best_models[variant] = best_model
        else:
            best_models[variant] = None
    
    # Save summary
    os.makedirs("results", exist_ok=True)
    summary_path = "results/best_models.csv"
    
    print(f"\n=== Saving Summary ===")
    save_best_models_summary(summary_path, best_models)
    print(f"Summary saved to: {summary_path}")
    
    # Print summary
    print(f"\n=== Best Models Summary ===")
    print("\nChapter 4 & 5:")
    for variant in grid_search_files_ch45.keys():
        model_info = best_models.get(variant)
        if model_info:
            print(f"  {variant}: {model_info.get('model_name', 'N/A')} (Val Acc: {model_info.get('val_accuracy', 'N/A')})")
        else:
            print(f"  {variant}: Not found")
    
    print("\nChapter 6:")
    for variant in grid_search_files_ch6.keys():
        model_info = best_models.get(variant)
        if model_info:
            print(f"  {variant}: {model_info.get('model_name', 'N/A')} (Val Acc: {model_info.get('val_accuracy', 'N/A')})")
        else:
            print(f"  {variant}: Not found")
    
    print(f"\n=== Done ===")
    print("Best models are saved as:")
    for variant, best_name in best_model_names.items():
        if best_models.get(variant):
            print(f"  {best_name}.h5")

if __name__ == "__main__":
    main()