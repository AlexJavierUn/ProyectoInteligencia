"""Entrenamiento comparable y persistencia versionada de los modelos."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.svm import LinearSVC

from .preprocessing import TARGET, load_dataset, split_scale_and_resample

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"

MODEL_SPECS = {
    "Regresión logística": ("logistic_regression", "Regresión Logística"),
    "SVM": ("svm_linear", "SVM lineal calibrado"),
}


def _make_metrics(model, X_test, y_test, rows, fraud_count, features, train_before, train_after, model_name, run_id):
    predictions = model.predict(X_test).astype(int)
    probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
    return {
        "run_id": run_id, "model_type": model_name,
        "rows": int(rows), "fraud_count": int(fraud_count), "test_size": int(len(y_test)),
        "features": list(features),
        "train_class_counts_before_smote": {str(k): int(v) for k, v in train_before.value_counts().sort_index().items()},
        "train_class_counts_after_smote": {str(k): int(v) for k, v in train_after.value_counts().sort_index().items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y_test, probabilities)) if probabilities is not None else None,
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if probabilities is not None and y_test.nunique() == 2 else None,
        "confusion_matrix": matrix,
        "report": classification_report(y_test, predictions, zero_division=0, output_dict=True),
    }


def train_models(csv_path: str | Path) -> dict:
    """Entrena ambos modelos con una misma partición, scaler, SMOTE y test intacto."""
    df = load_dataset(csv_path)
    X_train, X_test, y_train, y_test, scaler, features, y_train_original, _ = split_scale_and_resample(df)
    if y_train_original.nunique() < 2:
        raise ValueError("Se requieren ambas clases en entrenamiento.")

    run_id = uuid.uuid4().hex
    trained_at = datetime.now(timezone.utc).isoformat()
    classifiers = {
        "Regresión logística": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        "SVM": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=42, max_iter=5000), cv=3
        ),
    }
    results = {}
    registry = {"run_id": run_id, "trained_at_utc": trained_at, "models": {}}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, classifier in classifiers.items():
        key, display_name = MODEL_SPECS[model_name]
        classifier.fit(X_train, y_train)
        metrics = _make_metrics(classifier, X_test, y_test, len(df), int(df[TARGET].sum()), features,
                                y_train_original, y_train, display_name, run_id)
        model_path = MODELS_DIR / f"{key}_model.pkl"
        scaler_path = MODELS_DIR / f"{key}_scaler.pkl"
        joblib.dump({"model": classifier, "features": features, "model_type": display_name,
                     "run_id": run_id, "trained_at_utc": trained_at}, model_path)
        joblib.dump({"scaler": scaler, "run_id": run_id, "features": features}, scaler_path)
        registry["models"][display_name] = {"key": key, "model_path": model_path.name,
                                                   "scaler_path": scaler_path.name, "metrics": metrics}
        results[display_name] = metrics

    registry["configuration"] = {
        "target": TARGET, "test_size": .2, "random_state": 42,
        "scaler": "StandardScaler ajustado únicamente con entrenamiento",
        "resampling": "SMOTE únicamente sobre entrenamiento escalado",
        "model_parameters": {
            "Regresión logística": {"class_weight": "balanced", "max_iter": 2000, "random_state": 42},
            "SVM lineal calibrado": {"estimator": "LinearSVC", "class_weight": "balanced", "max_iter": 5000, "calibration_cv": 3},
        },
    }
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    # Mantener compatibilidad con predictor.py y la app Tkinter existentes.
    legacy_name = "SVM" if "SVM" in classifiers else next(iter(classifiers))
    key, display_name = MODEL_SPECS[legacy_name]
    legacy_artifact = joblib.load(MODELS_DIR / f"{key}_model.pkl")
    legacy_scaler = joblib.load(MODELS_DIR / f"{key}_scaler.pkl")["scaler"]
    joblib.dump(legacy_artifact, MODELS_DIR / "best_model.pkl")
    joblib.dump(legacy_scaler, MODELS_DIR / "scaler.pkl")
    METRICS_PATH.write_text(json.dumps(results[display_name], indent=2, ensure_ascii=False), encoding="utf-8")
    return registry


def train_model(csv_path: str | Path, model_type: str = "Regresión logística") -> dict:
    """Compatibilidad con la interfaz Tkinter; reutiliza el flujo compartido."""
    df = load_dataset(csv_path)
    X_train, X_test, y_train, y_test, scaler, features, y_train_original, _ = split_scale_and_resample(df)
    if model_type == "SVM":
        classifier = CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=42, max_iter=5000), cv=3)
        display_name = MODEL_SPECS["SVM"][1]
    else:
        classifier = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
        display_name = MODEL_SPECS["Regresión logística"][1]
    classifier.fit(X_train, y_train)
    run_id = uuid.uuid4().hex
    metrics = _make_metrics(classifier, X_test, y_test, len(df), int(df[TARGET].sum()), features,
                            y_train_original, y_train, display_name, run_id)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": classifier, "features": features, "model_type": display_name, "run_id": run_id}, MODELS_DIR / "best_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics
