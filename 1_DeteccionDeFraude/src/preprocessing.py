"""Carga y preparación segura de datos de transacciones."""
from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

TARGET = "Class"


def load_dataset(path: str | Path) -> pd.DataFrame:
    # Pandas puede renombrar encabezados repetidos silenciosamente (p. ej. V1.1).
    with open(path, "r", encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), [])
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise ValueError("El CSV contiene nombres de columna duplicados: " + ", ".join(duplicates))
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError("El CSV debe incluir la columna objetivo 'Class' (0=legítima, 1=fraudulenta).")
    if df.empty:
        raise ValueError("El archivo CSV está vacío.")
    df = df.drop_duplicates().dropna(subset=[TARGET])
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")
    if not np.isfinite(df[TARGET].to_numpy(dtype=float)).all() or not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("La columna 'Class' solo puede contener 0 y 1.")
    df[TARGET] = df[TARGET].astype(int)
    feature_columns = [column for column in df.columns if column != TARGET]
    if not feature_columns:
        raise ValueError("El CSV debe incluir al menos una característica además de 'Class'.")
    try:
        features = df[feature_columns].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Todas las características deben ser numéricas.") from exc
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Las características contienen valores vacíos o infinitos.")
    df[feature_columns] = features
    return df


def split_scale_and_resample(df: pd.DataFrame, target: str = TARGET, test_size: float = .2, random_state: int = 42):
    """Divide primero. Ajusta escalador y SMOTE solo sobre el bloque de entrenamiento."""
    X = df.drop(columns=[target]).apply(pd.to_numeric, errors="raise")
    y = df[target].astype(int)
    if y.nunique() < 2:
        raise ValueError("Se necesitan ejemplos de ambas clases (0 y 1) para entrenar.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if y_train.value_counts().min() < 2:
        raise ValueError("Se requieren al menos dos ejemplos de cada clase en entrenamiento para SMOTE.")
    k_neighbors = min(5, int(y_train.value_counts().min()) - 1)
    sampler = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_train_balanced, y_train_balanced = sampler.fit_resample(X_train_scaled, y_train)
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test, scaler, list(X.columns), y_train, sampler
