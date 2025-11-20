import pandas as pd
from preprocessing import preprocess_input
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report

# Path catre radacina proiectului
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
TEST_PATH = os.path.join(BASE_DIR, "data", "splits", "test.csv")

def evaluate():
    print("Incarc test.csv de la:", TEST_PATH)
    df = pd.read_csv(TEST_PATH)

    y_true = df["heart_disease"]
    X = preprocess_input(df.drop("heart_disease", axis=1))

    print("=== DEBUG ===")
    print("Columns in df:", df.drop("heart_disease", axis=1).columns)
    print("Shape of df:", df.drop("heart_disease", axis=1).shape)
    print("X returned:", X)
    print("X type:", type(X))
    if X is None:
        print("ERROR: preprocess_input returned None")
        exit()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Modelul nu exista! Ruleaza train.py.")

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X)

    print("\n=== Rezultate Evaluare ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    evaluate()