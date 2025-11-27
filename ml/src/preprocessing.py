"""
Preprocessing utilities for MedAI-Cardiac
-----------------------------------------

This module handles:
- Feature grouping (numeric / categorical)
- Fitting the StandardScaler on training data
- Saving and loading scalers
- Preprocessing new input samples for evaluation/inference

All transformations MUST be identical between training, validation,
testing, and inference. Only the training step is allowed to fit
new scaler parameters; all other steps only transform data using
the stored scalers.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATH HANDLING
# ============================================================

# ml/src/preprocessing.py → ml/
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SCALERS_PATH = os.path.join(BASE_DIR, "scalers.pkl")


# ============================================================
# FEATURE GROUPS
# ============================================================

NUMERIC_FEATURES = ["age", "resting_bp", "chol", "max_hr", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "fasting_glucose", "rest_ecg", "exercise_angina", "st_slope"]


# ============================================================
# SCALER UTILITIES
# ============================================================

def load_scalers():
    """
    Loads previously saved scalers from disk.
    Returns:
        dict or None
    """
    if os.path.exists(SCALERS_PATH):
        return joblib.load(SCALERS_PATH)
    return None


def save_scalers(scalers: dict):
    """
    Saves fitted scaler dictionary to disk.
    """
    joblib.dump(scalers, SCALERS_PATH)


# ============================================================
# TRAINING PREPROCESSING (FITTING)
# ============================================================

def fit_preprocess(df: pd.DataFrame):
    """
    Fits preprocessing steps on training data:
      - Fits StandardScaler on numeric features
      - Leaves categorical features unchanged (kept numeric already)
      - Saves scalers to disk
      - Returns processed feature matrix and scaler dictionary

    Args:
        df (pd.DataFrame): training data without labels

    Returns:
        X_processed (np.ndarray)
        scalers (dict)
    """

    scalers = {}

    # ---- 1. Numeric Features ----
    df_num = df[NUMERIC_FEATURES]
    num_scaler = StandardScaler()
    df_num_scaled = num_scaler.fit_transform(df_num)
    scalers["numeric"] = num_scaler

    # ---- 2. Categorical Features (already numeric) ----
    df_cat = df[CATEGORICAL_FEATURES].astype(float)

    # ---- 3. Concatenate numeric + categorical ----
    df_processed = np.concatenate(
        [df_num_scaled, df_cat.to_numpy()],
        axis=1
    )

    # Save scalers for evaluation/inference
    save_scalers(scalers)

    return df_processed, scalers


# ============================================================
# INFERENCE / EVALUATION PREPROCESSING (NO FITTING)
# ============================================================

def preprocess_input(df: pd.DataFrame):
    """
    Applies preprocessing to new input samples:
      - Loads stored StandardScaler
      - Transforms numeric features
      - Concatenates numeric + categorical

    Args:
        df (pd.DataFrame): input sample(s)

    Returns:
        np.ndarray: processed feature matrix
    """
    scalers = load_scalers()
    if scalers is None:
        raise ValueError(
            "Scalers not found! Run train.py first to generate them."
        )

    # Validate columns
    required_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: '{col}'")

    # ---- Numeric ----
    df_num = df[NUMERIC_FEATURES]
    df_num_scaled = scalers["numeric"].transform(df_num)

    # ---- Categorical ----
    df_cat = df[CATEGORICAL_FEATURES].astype(float)

    # ---- Combine ----
    result = np.concatenate(
        [df_num_scaled, df_cat.to_numpy()],
        axis=1
    )

    return result