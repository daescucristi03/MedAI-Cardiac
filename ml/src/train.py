import pandas as pd
from preprocessing import fit_preprocess
from model_def import create_model
import joblib
import os

# PROJECT ROOT
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
TRAIN_PATH = os.path.join(BASE_DIR, "data", "splits", "train.csv")

def train():
    print("Citesc datasetul din:", TRAIN_PATH)

    df = pd.read_csv(TRAIN_PATH)

    y = df["heart_disease"]
    X, scalers = fit_preprocess(df.drop("heart_disease", axis=1))

    model = create_model()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print("Model antrenat si salvat la:", MODEL_PATH)

if __name__ == "__main__":
    train()