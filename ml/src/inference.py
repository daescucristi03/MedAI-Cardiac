import os
import joblib
import pandas as pd
from preprocessing import preprocess_input

# PATH HANDLING
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
SCALERS_PATH = os.path.join(BASE_DIR, "ml", "scalers.pkl")

# VALIDATE MODEL AND SCALERS
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelul nu exista: {MODEL_PATH}. Rulează train.py.")

if not os.path.exists(SCALERS_PATH):
    raise FileNotFoundError(f"Scalers nu exista: {SCALERS_PATH}. Ruleaza train.py.")

# Load Model
model = joblib.load(MODEL_PATH)

# UTILS
def risk_level(prob):
    """Returneaza low / medium / high risk."""
    if prob < 0.33:
        return "low"
    elif prob < 0.66:
        return "medium"
    else:
        return "high"


def multi_diagnostic(prob, patient):
    """
    Creeaza diagnostice multiple bazate pe prob + factorii clinici.
    Aceste inferente sunt euristice, foarte bune pentru un proiect universitar.
    """

    oldpeak = patient["oldpeak"]
    chol = patient["chol"]
    angina = patient["exercise_angina"]

    # Clamp values to normal range
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


def top_factors(patient):
    """Simplu: primele 3 riscuri potentiale."""
    scores = {
        "chol": patient["chol"],
        "oldpeak": patient["oldpeak"],
        "resting_bp": patient["resting_bp"]
    }
    return sorted(scores, key=scores.get, reverse=True)[:3]


# UNIFIED PREDICTION
def predict_single(input_dict: dict):
    """
    Input unic → Output unificat.
    Returneaza:
    - predictie binara
    - scor risc
    - nivel risc
    - diagnostice multiple
    - explanation
    """

    df = pd.DataFrame([input_dict])

    X = preprocess_input(df)
    prob = model.predict_proba(X)[0][1]
    binary = int(prob >= 0.5)

    multi = multi_diagnostic(prob, input_dict)
    risk_cat = risk_level(prob)

    explanation = {
        "top_factors": top_factors(input_dict)
    }

    return {
        "binary_prediction": binary,
        "risk_score": float(prob),
        "risk_level": risk_cat,
        "diagnostics": multi,
        "explanation": explanation
    }


# LOCAL TEST
if __name__ == "__main__":
    sample = {
        "age": 35,
        "sex": 0,
        "resting_bp": 115,
        "chol": 170,
        "fasting_glucose": 0,
        "rest_ecg": 0,
        "max_hr": 185,
        "exercise_angina": 0,
        "oldpeak": 0.0,
        "st_slope": 2
    }

    print("=== Test Local Inference (Unified Output) ===")
    print(predict_single(sample))