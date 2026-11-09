"""Predicción de una o varias transacciones usando artefactos entrenados."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_artifacts():
    model_file, scaler_file = ROOT / "models" / "best_model.pkl", ROOT / "models" / "scaler.pkl"
    if not model_file.exists() or not scaler_file.exists() or model_file.stat().st_size == 0 or scaler_file.stat().st_size == 0:
        raise FileNotFoundError("No hay un modelo entrenado. Selecciona un CSV y pulsa 'Entrenar modelo'.")
    return joblib.load(model_file), joblib.load(scaler_file)


def predict_transaction(values: dict) -> dict:
    artifacts, scaler = load_artifacts()
    features = artifacts["features"]
    missing = [name for name in features if name not in values]
    if missing:
        raise ValueError("Faltan columnas: " + ", ".join(missing))
    row = pd.DataFrame([[float(values[name]) for name in features]], columns=features)
    model = artifacts["model"]
    probability = float(model.predict_proba(scaler.transform(row))[0, 1])
    return {"prediction": int(probability >= .5), "probability": probability}
