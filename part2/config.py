"""
config.py – Project-wide constants and paths.
"""
from pathlib import Path
import numpy as np

# -- Reproducibility -----------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

# -- Paths ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "dataset.csv"
PROCESSED_DIR = DATA_DIR / "processed"

# -- Target --------------------------------------------------------------------
TARGET_COLUMN = "Price"

# -- Features ------------------------------------------------------------------
# Numeric features (sau khi clean)
NUMERIC_FEATURES = [
    "Screen_Size",
    "RAM",
]

# Categorical features
CATEGORICAL_FEATURES = [
    "Brand",
    "GPU_Type",
    "Condition",
]

# -- Lambda grid cho Ridge / Lasso ---------------------------------------------
LAMBDAS = np.logspace(-4, 4, 50)
