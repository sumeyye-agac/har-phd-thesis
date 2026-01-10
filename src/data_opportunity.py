"""
Opportunity Dataset Preprocessing

Based on:
https://github.com/AniMahajan20/DeepConvLSTM-NNFL
Dataset:
https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition
"""

import os
import zipfile
from io import BytesIO
from collections import Counter

import numpy as np
import pickle as cp
from pandas import Series
from sklearn.model_selection import train_test_split

from src.config import DATA_RANDOM_STATE

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

RAW_ZIP_PATH = "data/raw/OpportunityUCIDataset.zip"
PROCESSED_RAW_PATH = "data/raw/oppChallenge_gestures.data"
PROCESSED_SPLIT_PATH = "data/processed/opportunity_splits.pkl"

# -----------------------------------------------------------------------------
# Constants from Opportunity challenge preprocessing
# -----------------------------------------------------------------------------

NB_SENSOR_CHANNELS = 113

OPPORTUNITY_DATA_FILES = [
    "OpportunityUCIDataset/dataset/S1-Drill.dat",
    "OpportunityUCIDataset/dataset/S1-ADL1.dat",
    "OpportunityUCIDataset/dataset/S1-ADL2.dat",
    "OpportunityUCIDataset/dataset/S1-ADL3.dat",
    "OpportunityUCIDataset/dataset/S1-ADL4.dat",
    "OpportunityUCIDataset/dataset/S1-ADL5.dat",
]

# From original DeepConvLSTM preprocessing code
MAX_THRESHOLDS = [
    3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
    3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
    3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
    3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
    3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
    3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
    3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
    3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
    3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
    250, 25, 200, 5000, 5000, 5000, 5000, 5000, 5000,
    10000, 10000, 10000, 10000, 10000, 10000, 250, 250, 25,
    200, 5000, 5000, 5000, 5000, 5000, 5000, 10000, 10000,
    10000, 10000, 10000, 10000, 250
]

MIN_THRESHOLDS = [
    -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000,
    -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000,
    -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000,
    -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000,
    -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000,
    -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000,
    -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000,
    -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000,
    -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000,
    -250, -100, -200, -5000, -5000, -5000, -5000, -5000, -5000,
    -10000, -10000, -10000, -10000, -10000, -10000, -250, -250, -100,
    -200, -5000, -5000, -5000, -5000, -5000, -5000, -10000, -10000,
    -10000, -10000, -10000, -10000, -250
]


# -----------------------------------------------------------------------------
# Preprocessing helpers
# -----------------------------------------------------------------------------

def select_columns(data):
    """Remove quaternion and object channels exactly like original pipeline."""
    delete = np.concatenate([
        np.arange(46, 50),
        np.arange(59, 63),
        np.arange(72, 76),
        np.arange(85, 89),
        np.arange(98, 102),
        np.arange(134, 243),
        np.arange(244, 249)
    ])
    return np.delete(data, delete, axis=1)


def norm(data, max_list, min_list):
    max_list = np.array(max_list)
    min_list = np.array(min_list)
    diff = max_list - min_list

    for i in range(data.shape[1]):
        data[:, i] = (data[:, i] - min_list[i]) / diff[i]

    data[data > 1] = 0.99
    data[data < 0] = 0.00
    return data


def check_data(dataset):
    """Download Opportunity ZIP if missing."""
    data_dir, data_file = os.path.split(dataset)

    if (not os.path.isfile(dataset)) and data_file == "OpportunityUCIDataset.zip":
        import urllib.request
        os.makedirs(data_dir, exist_ok=True)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip"
        print("Downloading Opportunity dataset...")
        urllib.request.urlretrieve(url, dataset)
        print("Download complete.")

    return data_dir


def process_dataset_file(raw):
    """Process one .dat file → sensor selection, interpolation, normalization, remapping."""
    data = select_columns(raw)

    data_x = data[:, 1:114]
    data_y = data[:, 115]

    mapping = {
        406516: 1, 406517: 2, 404516: 3, 404517: 4,
        406520: 5, 404520: 6, 406505: 7, 404505: 8,
        406519: 9, 404519: 10, 406511: 11, 404511: 12,
        406508: 13, 404508: 14, 408512: 15, 407521: 16,
        405506: 17,
    }

    for old, new in mapping.items():
        data_y[data_y == old] = new

    data_x = np.array([Series(col).interpolate() for col in data_x.T]).T
    data_x[np.isnan(data_x)] = 0

    data_x = norm(data_x, MAX_THRESHOLDS, MIN_THRESHOLDS)
    return data_x, data_y.astype(np.uint8)


# -----------------------------------------------------------------------------
# Step 1: Generate raw continuous dataset (oppChallenge_gestures.data)
# -----------------------------------------------------------------------------

def generate_raw_gestures(zip_path, out_name):
    data_dir = check_data(zip_path)

    X = np.empty((0, NB_SENSOR_CHANNELS))
    Y = np.empty((0))

    zf = zipfile.ZipFile(zip_path)
    print("Processing .dat files...")

    for fname in OPPORTUNITY_DATA_FILES:
        try:
            raw = np.loadtxt(BytesIO(zf.read(fname)))
            x, y = process_dataset_file(raw)
            X = np.vstack([X, x])
            Y = np.concatenate([Y, y])
        except KeyError:
            print(f"Missing: {fname}")

    out_path = os.path.join(data_dir, out_name)
    with open(out_path, "wb") as f:
        cp.dump([X, Y], f)

    print("Saved:", out_path)
    return out_path


def load_raw_gestures(path):
    """Load raw gesture data from pickle file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw gestures file not found: {path}")
    with open(path, "rb") as f:
        X, Y = cp.load(f)
    return X.astype(np.float32), Y.astype(np.uint8)


# -----------------------------------------------------------------------------
# Step 2: Windowing + Splitting
# -----------------------------------------------------------------------------

def sliding_window(data_x, data_y, window=30, step=30):
    X, Y = [], []
    start = 0

    while start + window <= len(data_x):
        X.append(data_x[start:start + window])
        Y.append(data_y[start])
        start += step

    return np.array(X), np.array(Y)


def normalize_splits(train, val, test):
    mean = np.mean(train, axis=(0, 1))
    std = np.std(train, axis=(0, 1)) + 1e-8
    return (train - mean) / std, (val - mean) / std, (test - mean) / std


# -----------------------------------------------------------------------------
# Step 3: Full preprocessing pipeline
# -----------------------------------------------------------------------------

def prepare_opportunity_splits():
    """
    Prepare Opportunity splits with a single, consistent label contract:

    - Remove NULL class (label 0) completely.
    - Remap labels to start at 0 (original gesture labels 1..17 -> 0..16).
    - Windowing: window=30, step=30.
    - Save one canonical pickle used by all chapters.
    """
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    if not os.path.exists(PROCESSED_RAW_PATH):
        generate_raw_gestures(RAW_ZIP_PATH, "oppChallenge_gestures.data")

    X_raw, Y_raw = load_raw_gestures(PROCESSED_RAW_PATH)

    # Windowing
    X, Y = sliding_window(X_raw, Y_raw, 30, 30)

    # ---------------------------------------------------------
    # IMPORTANT: enforce label contract
    #   - drop NULL class (0)
    #   - shift labels to 0..C-1
    # ---------------------------------------------------------
    mask = (Y != 0)
    X = X[mask]
    Y = Y[mask].astype(np.int64) - 1  # 1..17 -> 0..16

    # Deterministic random splits (example pipeline)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=0.15,
        random_state=DATA_RANDOM_STATE,
        shuffle=True,
        stratify=Y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,
        random_state=DATA_RANDOM_STATE,
        shuffle=True,
        stratify=y_train,
    )

    # Normalize using train stats only
    X_train, X_val, X_test = normalize_splits(X_train, X_val, X_test)

    data = {
        "X_train": X_train.astype(np.float32), "y_train": y_train.astype(np.int64),
        "X_val":   X_val.astype(np.float32),   "y_val":   y_val.astype(np.int64),
        "X_test":  X_test.astype(np.float32),  "y_test":  y_test.astype(np.int64),
        "meta": {
            "window": 30,
            "step": 30,
            "nb_channels": NB_SENSOR_CHANNELS,
            "label_contract": "drop_null(0), shift(1..17->0..16)",
            "random_state": DATA_RANDOM_STATE,
            "test_size": 0.15,
            "val_size_from_train": 0.15,
        }
    }

    with open(PROCESSED_SPLIT_PATH, "wb") as f:
        cp.dump(data, f)

    print("Saved splitted dataset →", PROCESSED_SPLIT_PATH)
    print("Shapes:",
          "X_train", data["X_train"].shape, "y_train", data["y_train"].shape,
          "| X_val", data["X_val"].shape, "y_val", data["y_val"].shape,
          "| X_test", data["X_test"].shape, "y_test", data["y_test"].shape)
    print("Classes:", int(np.max(data["y_train"])) + 1)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def load_opportunity_splits():
    if not os.path.exists(PROCESSED_SPLIT_PATH):
        print("Splits missing — generating now...")
        prepare_opportunity_splits()

    with open(PROCESSED_SPLIT_PATH, "rb") as f:
        data = cp.load(f)

    return (
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        data["X_test"], data["y_test"],
    )
