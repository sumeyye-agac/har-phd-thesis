"""
Chapter 7 — Compression (TFLite) for OM only.

Outputs (models_saved/tflite):
- OM-Lite.tflite : plain TFLite (no optimization)
- OM-DRQ.tflite  : Dynamic Range Quantization (Optimize.DEFAULT, no repset)
- OM-FQ.tflite   : Float16 quantization
- OM-CP.tflite   : "Constant pruning" approximation (global magnitude pruning @ 50% sparsity) + TFLite
- OM-PDP.tflite  : "Polynomial decay pruning" approximation (global magnitude pruning @ 80% sparsity) + TFLite

Note:
- This repo does not include tensorflow_model_optimization (tfmot). Therefore pruning here is implemented
  as deterministic global magnitude pruning (zeroing smallest-magnitude weights) before conversion.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from src.config import MODEL_SEED, TRAINING_NP_SEED
from src.model_io import load_model_tf, save_model_tflite
from src.registry import (
    exists_tf, exists_tflite,
    tf_name,
    tflite_name,
)
from src.config import MODEL_SEED, TRAINING_NP_SEED, COMPRESSION_CP_SPARSITY, COMPRESSION_PDP_SPARSITY
from src.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)

# -----------------------------
# Helpers
# -----------------------------

def convert_lite(model: tf.keras.Model) -> bytes:
    """Convert to TFLite with SELECT_TF_OPS for LSTM support."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    return converter.convert()

def convert_drq(model: tf.keras.Model) -> bytes:
    """Convert to TFLite with DRQ and SELECT_TF_OPS for LSTM support."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # DRQ when no representative dataset provided
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    return converter.convert()

def convert_fq_float16(model: tf.keras.Model) -> bytes:
    """Convert to TFLite with Float16 quantization and SELECT_TF_OPS for LSTM support."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    return converter.convert()

def global_magnitude_prune_inplace(model: tf.keras.Model, sparsity: float, seed: int = TRAINING_NP_SEED) -> tf.keras.Model:
    """
    Deterministic global magnitude pruning:
    - Collect all float weights across layers
    - Zero-out the smallest |w| to reach desired sparsity
    - Writes back into model weights (in-place)

    sparsity: fraction of weights to set to zero, e.g., 0.5 means 50% zeros.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError("sparsity must be in [0, 1).")

    rng = np.random.default_rng(seed)

    # Gather all float weights into a single vector (excluding biases is NOT required; keep everything float)
    weight_refs = []  # (layer, weight_index_in_layer, original_shape, dtype)
    all_weights = []

    for layer in model.layers:
        wts = layer.get_weights()
        if not wts:
            continue

        new_wts = []
        touched = False

        for i, w in enumerate(wts):
            if np.issubdtype(w.dtype, np.floating):
                weight_refs.append((layer, i, w.shape, w.dtype))
                all_weights.append(w.reshape(-1))
                touched = True
            new_wts.append(w)

        # no set here; we will set later with updated arrays

    if not all_weights:
        # Nothing to prune
        return model

    flat = np.concatenate(all_weights, axis=0)
    n = flat.size
    k = int(np.floor(sparsity * n))
    if k <= 0:
        return model

    # Threshold for smallest magnitudes
    mags = np.abs(flat)
    # Use partial sort for threshold; to be deterministic with ties, add tiny noise
    noise = rng.normal(loc=0.0, scale=1e-12, size=mags.shape)
    mags_noisy = mags + noise
    thresh = np.partition(mags_noisy, k - 1)[k - 1]

    # Mask: keep weights whose magnitude > thresh; prune the rest
    keep = mags_noisy > thresh
    pruned = flat.copy()
    pruned[~keep] = 0.0

    # Write back pruned vector into each layer weight
    cursor = 0
    for layer in model.layers:
        wts = layer.get_weights()
        if not wts:
            continue

        updated = []
        for w in wts:
            if np.issubdtype(w.dtype, np.floating):
                size = w.size
                chunk = pruned[cursor:cursor + size].reshape(w.shape).astype(w.dtype, copy=False)
                updated.append(chunk)
                cursor += size
            else:
                updated.append(w)

        layer.set_weights(updated)

    return model

# -----------------------------
# Main
# -----------------------------

def main():
    logger.info("=== Chapter 7: Model Compression (TFLite) ===")
    
    # Safety: OM must exist
    if not exists_tf("OM"):
        error_msg = "OM.h5 not found. Run Chapter 4 first to create models_saved/tf/OM.h5"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info("Loading OM model...")
    om = load_model_tf(tf_name("OM"))

    # 1) Lite
    if exists_tflite("OM-Lite"):
        logger.info(f"OM-Lite exists -> skipping: {tflite_name('OM-Lite')}")
    else:
        logger.info("Converting OM-Lite (plain TFLite)...")
        tflite_bytes = convert_lite(om)
        save_model_tflite(tflite_bytes, tflite_name("OM-Lite"))
        logger.info("✓ OM-Lite converted successfully")

    # 2) DRQ
    if exists_tflite("OM-DRQ"):
        logger.info(f"OM-DRQ exists -> skipping: {tflite_name('OM-DRQ')}")
    else:
        logger.info("Converting OM-DRQ (Dynamic Range Quantization)...")
        tflite_bytes = convert_drq(om)
        save_model_tflite(tflite_bytes, tflite_name("OM-DRQ"))
        logger.info("✓ OM-DRQ converted successfully")

    # 3) FQ (float16)
    if exists_tflite("OM-FQ"):
        logger.info(f"OM-FQ exists -> skipping: {tflite_name('OM-FQ')}")
    else:
        logger.info("Converting OM-FQ (Float16 quantization)...")
        tflite_bytes = convert_fq_float16(om)
        save_model_tflite(tflite_bytes, tflite_name("OM-FQ"))
        logger.info("✓ OM-FQ converted successfully")

    # 4) CP (approx. constant pruning @ 50%)
    if exists_tflite("OM-CP"):
        logger.info(f"OM-CP exists -> skipping: {tflite_name('OM-CP')}")
    else:
        logger.info(f"Converting OM-CP (Constant Pruning @ {COMPRESSION_CP_SPARSITY*100:.0f}%)...")
        om_cp = load_model_tf(tf_name("OM"))  # Load fresh copy
        global_magnitude_prune_inplace(om_cp, sparsity=COMPRESSION_CP_SPARSITY, seed=TRAINING_NP_SEED)
        tflite_bytes = convert_lite(om_cp)
        save_model_tflite(tflite_bytes, tflite_name("OM-CP"))
        logger.info("✓ OM-CP converted successfully")

    # 5) PDP (approx. polynomial decay pruning @ 80%)
    if exists_tflite("OM-PDP"):
        logger.info(f"OM-PDP exists -> skipping: {tflite_name('OM-PDP')}")
    else:
        logger.info(f"Converting OM-PDP (Polynomial Decay Pruning @ {COMPRESSION_PDP_SPARSITY*100:.0f}%)...")
        om_pdp = load_model_tf(tf_name("OM"))  # Load fresh copy
        global_magnitude_prune_inplace(om_pdp, sparsity=COMPRESSION_PDP_SPARSITY, seed=TRAINING_NP_SEED)
        tflite_bytes = convert_lite(om_pdp)        
        save_model_tflite(tflite_bytes, tflite_name("OM-PDP"))
        logger.info("✓ OM-PDP converted successfully")

    logger.info("All compression models completed!")


if __name__ == "__main__":
    main()
