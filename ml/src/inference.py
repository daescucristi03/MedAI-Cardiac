"""
Inference Engine for MedAI-Cardiac
----------------------------------

This module handles real-time inference for the trained cardiac
risk prediction model. It loads the trained classifier and the
stored preprocessing scalers and produces a unified diagnostic output.

Pipeline:
1. Validate that model and scalers exist.
2. Preprocess user-provided input using stored scalers.
3. Run probability prediction using the neural network.
4. Generate:
    - binary prediction (0/1)
    - risk probability
    - risk category (low / medium / high)
    - heuristic multi-diagnostic estimations
    - explanation of top contributing features

This inference module is used both locally (CLI test) and by the
FastAPI backend in production.
"""

import os
import joblib
import pandas as pd
from preprocessing import preprocess_input


# ============================================================
# PATH HANDLING
# ============================================================

# ml/src/inference.py → ml/ → project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
SCALERS_PATH = os.path.join(BASE_DIR, "ml", "scalers.pkl")


# ============================================================
# MODEL & SCALERS VALIDATION
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}. Please run train.py first."
    )

if not os.path.exists(SCALERS_PATH):
    raise FileNotFoundError(
        f"Scalers not found at: {SCALERS_PATH}. Please run train.py first."
    )

# Load model once at import time (faster inference)
model = joblib.load(MODEL_PATH)


# ============================================================
# RISK CATEGORY
# ============================================================

def risk_level(prob: float) -> str:
    """
    Converts a probability into a categorical cardiovascular
    risk level.

    Args:
        prob (float): Predicted probability (0–1)

    Returns:
        str: "low", "medium", or "high"
    """
    if prob < 0.33:
        return "low"
    elif prob < 0.66:
        return "medium"
    return "high"


# ============================================================
# SECONDARY DIAGNOSTIC HEURISTICS
# ============================================================

def multi_diagnostic(prob: float, patient: dict) -> dict:
    """
    Generates heuristic secondary cardiac diagnostics based on
    the predicted probability and selected clinical indicators.

    NOTE:
    These computations are heuristic and should NOT be used
    for medical decision-making. They are suitable for academic
    demonstration purposes.

    Args:
        prob (float): primary probability
        patient (dict): raw patient input

    Returns:
        dict: scores for CHD, CAD, MI, ischemia
    """
    oldpeak = patient["oldpeak"]
    chol = patient["chol"]
    angina = patient["exercise_angina"]

    def clamp(x):
        return max(0, min(1, x))

    chd = clamp(prob * (0.4 + oldpeak / 6 + chol / 400))
    cad = clamp(prob * (chol / 300))
    mi = clamp(prob * (angina + oldpeak / 3))
    ischemia = clamp(prob * (oldpeak / 3))

    return {
        "chd": float(chd),
        "cad": float(cad),
        "mi": float(mi),
        "ischemia": float(ischemia)
    }


# ============================================================
# EXPLANATION HEURISTICS
# ============================================================

def top_factors(patient: dict) -> list:
    """
    Returns the top three contributing factors based on
    raw clinical values.

    This is NOT feature importance — simply a heuristic
    ranking based on magnitude.

    Args:
        patient (dict): raw patient dictionary

    Returns:
        list[str]: ordered feature names
    """
    scores = {
        "chol": patient["chol"],
        "oldpeak": patient["oldpeak"],
        "resting_bp": patient["resting_bp"]
    }

    return sorted(scores, key=scores.get, reverse=True)[:3]


# ============================================================
# UNIFIED PREDICTION FUNCTION
# ============================================================

def predict_single(input_dict: dict) -> dict:
    """
    Performs a full cardiac risk prediction for a single patient.

    Workflow:
      - Convert input to DataFrame
      - Preprocess using stored scalers
      - Predict probability with model
      - Create binary classification
      - Generate heuristic diagnostics
      - Produce human-readable risk category
      - Provide simple interpretation (top factors)

    Args:
        input_dict (dict): raw patient feature dictionary

    Returns:
        dict: structured prediction output
    """

    df = pd.DataFrame([input_dict])
    X = preprocess_input(df)
    prob = model.predict_proba(X)[0][1]

    binary_prediction = int(prob >= 0.5)
    diagnostics = multi_diagnostic(prob, input_dict)
    category = risk_level(prob)

    explanation = {"top_factors": top_factors(input_dict)}

    return {
        "binary_prediction": binary_prediction,
        "risk_score": float(prob),
        "risk_level": category,
        "diagnostics": diagnostics,
        "explanation": explanation
    }


# ============================================================
# LOCAL TEST (RUN DIRECTLY FROM TERMINAL)
# ============================================================

if __name__ == "__main__":
    sample = {
        "age": 65,
        "sex": 0,
        "resting_bp": 120,
        "chol": 170,
        "fasting_glucose": 0,
        "rest_ecg": 0,
        "max_hr": 185,
        "exercise_angina": 0,
        "oldpeak": 0.0,
        "st_slope": 2
    }

    print("=== Local Inference Test (Unified Output) ===")
    print(predict_single(sample))