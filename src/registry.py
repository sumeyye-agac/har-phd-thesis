# src/registry.py
"""
Single source of truth for:
  - Canonical model filenames
  - Canonical on-disk locations (models_saved/tf and models_saved/tflite)
  - Lightweight helpers (exists/path)

All training/evaluation scripts and notebooks should import model names/paths
from here instead of hard-coding strings.
"""

from __future__ import annotations

import os
from typing import Dict, Final, Literal

# Canonical directories (repo-relative)
TF_DIR: Final[str] = os.path.join("models_saved", "tf")
TFLITE_DIR: Final[str] = os.path.join("models_saved", "tflite")

# Ensure dirs exist when imported (safe + convenient)
os.makedirs(TF_DIR, exist_ok=True)
os.makedirs(TFLITE_DIR, exist_ok=True)

# ---------------------------
# Canonical model keys
# ---------------------------

TFModelKey = Literal[
    "OM",
    "OM-Att",
    "LM",
    "MM",
    "LM-Att",
    "MM-Att",
    "LM-RB-KD",
    "LM-RAB-KD",
    "LM-RB-KD-Att",
]

TFLiteModelKey = Literal[
    "OM-Lite",
    "OM-DRQ",
    "OM-FQ",
    "OM-CP",
    "OM-PDP",
]

# ---------------------------
# Filenames
# ---------------------------

TF_MODEL_FILES: Final[Dict[str, str]] = {
    "OM": "OM.h5",
    "OM-Att": "OM-Att.h5",
    "LM": "LM.h5",
    "MM": "MM.h5",
    "LM-Att": "LM-Att.h5",
    "MM-Att": "MM-Att.h5",
    "LM-RB-KD": "LM-RB-KD.h5",
    "LM-RAB-KD": "LM-RAB-KD.h5",
    "LM-RB-KD-Att": "LM-RB-KD-Att.h5",
}

TFLITE_MODEL_FILES: Final[Dict[str, str]] = {
    "OM-Lite": "OM-Lite.tflite",
    "OM-DRQ": "OM-DRQ.tflite",
    "OM-FQ": "OM-FQ.tflite",
    "OM-CP": "OM-CP.tflite",
    "OM-PDP": "OM-PDP.tflite",
}

# ---------------------------
# Path helpers
# ---------------------------

def tf_name(key: TFModelKey) -> str:
    return TF_MODEL_FILES[key]

def tflite_name(key: TFLiteModelKey) -> str:
    return TFLITE_MODEL_FILES[key]

def tf_path(key: TFModelKey) -> str:
    return os.path.join(TF_DIR, tf_name(key))

def tflite_path(key: TFLiteModelKey) -> str:
    return os.path.join(TFLITE_DIR, tflite_name(key))

def exists_tf(key: TFModelKey) -> bool:
    return os.path.exists(tf_path(key))

def exists_tflite(key: TFLiteModelKey) -> bool:
    return os.path.exists(tflite_path(key))
