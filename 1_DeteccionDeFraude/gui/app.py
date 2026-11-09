"""FraudGuard AI: aplicación web Streamlit para demostración académica."""
from __future__ import annotations

import io
import json
import sys
import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.dataset import download_kaggle_dataset
from src.preprocessing import TARGET, load_dataset
from src.train import train_models

DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
MODEL_PATH = ROOT / "models" / "best_model.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"
REGISTRY_PATH = ROOT / "models" / "model_registry.json"
NAVY, BLUE, GREEN, RED = "#10243A", "#2878D0", "#16856B", "#C64747"

st.set_page_config(page_title="FraudGuard AI", page_icon="🛡", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
[data-testid="stAppViewContainer"] .main{color:#172b42}
[data-testid="stAppViewContainer"] .main p,[data-testid="stAppViewContainer"] .main label,[data-testid="stAppViewContainer"] .main [data-testid="stMarkdownContainer"]{color:#24384f}
.stApp{background:#f3f6fa;color:#172b42}.block-container{padding-top:1.6rem;max-width:1500px}
[data-testid="stSidebar"]{background:#10243a}[data-testid="stSidebar"] *{color:#f2f6fb!important}
[data-testid="stSidebar"] [data-testid="stAlert"] *{color:#24384f!important}
.hero{background:linear-gradient(115deg,#10243a,#1d4263);padding:1.5rem 1.8rem;border-radius:16px;color:white;margin-bottom:1rem}
.hero h1{color:white;margin:0;font-size:2rem}.hero p{color:#d8e6f3;margin:.35rem 0 0}
.kpi{background:white;border:1px solid #e4eaf1;border-radius:14px;padding:1rem 1.1rem;box-shadow:0 3px 12px #1734510b;min-height:105px}
.kpi-label{color:#64758a;font-size:.84rem}.kpi-value{color:#142a42;font-size:1.55rem;font-weight:700;margin-top:.3rem}
.stButton>button[kind="primary"]{background:#2878d0;border-color:#2878d0}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def read_data(path: str, modified: float) -> pd.DataFrame:
    return load_dataset(path)


@st.cache_resource(show_spinner=False)
def read_artifacts(model_path: str, scaler_path: str, model_mtime: float, scaler_mtime: float):
    return joblib.load(model_path), joblib.load(scaler_path)


@st.cache_data(show_spinner=False)
def read_metrics(path: str, modified: float):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_registry(path: str, modified: float):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def card(label, value, detail=""):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-label">{detail}</div></div>', unsafe_allow_html=True)


def get_dataset():
    if not DATA_PATH.exists():
        return None
    return read_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)


def get_models():
    models = {}
    if REGISTRY_PATH.exists():
        try:
            registry = read_registry(str(REGISTRY_PATH), REGISTRY_PATH.stat().st_mtime)
            for name, info in registry.get("models", {}).items():
                model_path = ROOT / "models" / info["model_path"]
                scaler_path = ROOT / "models" / info["scaler_path"]
                if not model_path.exists() or not scaler_path.exists():
                    raise FileNotFoundError(f"Falta el artefacto registrado de {name}.")
                artifact, scaler_bundle = read_artifacts(str(model_path), str(scaler_path), model_path.stat().st_mtime, scaler_path.stat().st_mtime)
                scaler = scaler_bundle.get("scaler") if isinstance(scaler_bundle, dict) else scaler_bundle
                if artifact.get("run_id") != registry.get("run_id") or (isinstance(scaler_bundle, dict) and scaler_bundle.get("run_id") != registry.get("run_id")):
                    raise ValueError(f"El modelo y escalador de {name} pertenecen a ejecuciones diferentes.")
                if isinstance(scaler_bundle, dict) and scaler_bundle.get("features") != artifact.get("features"):
                    raise ValueError(f"El escalador de {name} no coincide con el orden de características del modelo.")
                models[name] = (artifact, scaler)
            return models
        except Exception as exc:
            st.error(f"No se pudo validar el registro de modelos: {exc}")
            return {}
    if MODEL_PATH.exists() and SCALER_PATH.exists() and MODEL_PATH.stat().st_size and SCALER_PATH.stat().st_size:
        try:
            artifacts, scaler = read_artifacts(str(MODEL_PATH), str(SCALER_PATH), MODEL_PATH.stat().st_mtime, SCALER_PATH.stat().st_mtime)
            models[artifacts.get("model_type", "Modelo guardado")] = (artifacts, scaler)
        except Exception as exc:
            st.error(f"No se pudieron cargar los artefactos: {exc}")
    return models


def predict_frame(frame: pd.DataFrame, choice: str, models, progress_callback=None):
    duplicates = sorted(frame.columns[frame.columns.duplicated()].unique().tolist())
    if duplicates:
        raise ValueError("El CSV contiene columnas duplicadas: " + ", ".join(duplicates))
    artifacts, scaler = models[choice]
    features = artifacts["features"]
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise ValueError("Faltan características requeridas: " + ", ".join(missing))
    X = frame.loc[:, features].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Las características contienen valores vacíos o no finitos.")
    model = artifacts["model"]
    predictions, probabilities = [], []
    chunk_size = 10000
    for start in range(0, len(X), chunk_size):
        batch_x = scaler.transform(X.iloc[start:start + chunk_size])
        predictions.append(model.predict(batch_x).astype(int))
        if hasattr(model, "predict_proba"):
            probabilities.append(model.predict_proba(batch_x)[:, 1])
        if progress_callback:
            progress_callback(min(start + chunk_size, len(X)) / max(1, len(X)))
    prediction = np.concatenate(predictions) if predictions else np.array([], dtype=int)
    probability = np.concatenate(probabilities) if probabilities else None
    return prediction, probability


def evaluation_scores(df, artifacts, scaler):
    """Reproduce la partición estratificada original para recuperar scores reales."""
    X = df.drop(columns=[TARGET]).loc[:, artifacts["features"]]
    y = df[TARGET].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    X_test_scaled = scaler.transform(X_test)
    model = artifacts["model"]
    if not hasattr(model, "predict_proba"):
        return None
    return y_test, model.predict_proba(X_test_scaled)[:, 1]


def section_header(title, text):
    st.subheader(title)
    st.caption(text)


with st.sidebar:
    st.markdown("## 🛡 FraudGuard AI")
    st.caption("Centro de monitoreo antifraude")
    page = st.radio("NAVEGACIÓN", ["Dashboard", "Evaluación", "Probar transacción", "Análisis masivo", "Información"], label_visibility="visible")
    st.divider()
    if DATA_PATH.exists():
        st.success("Dataset disponible")
    else:
        st.warning("Dataset no cargado")
    st.caption("Prototipo académico · v1")

st.markdown('<div class="hero"><h1>FraudGuard AI</h1><p>Sistema inteligente de clasificación de transacciones financieras</p></div>', unsafe_allow_html=True)
df = get_dataset()
models = get_models()
metrics = read_metrics(str(METRICS_PATH), METRICS_PATH.stat().st_mtime) if METRICS_PATH.exists() else None
registry = read_registry(str(REGISTRY_PATH), REGISTRY_PATH.stat().st_mtime) if REGISTRY_PATH.exists() else None

if page == "Dashboard":
    section_header("Centro de monitoreo antifraude", "Resumen del dataset, distribución de clases y estado de los artefactos cargados.")
    if df is None:
        st.info("Carga el dataset de Kaggle para habilitar el dashboard.")
        if st.button("Descargar dataset Kaggle", type="primary"):
            try:
                with st.spinner("Descargando dataset..."):
                    download_kaggle_dataset()
                st.cache_data.clear(); st.rerun()
            except Exception as exc:
                st.error(f"No fue posible descargarlo: {exc}")
    else:
        legitimate = int((df[TARGET] == 0).sum()); fraud = int((df[TARGET] == 1).sum())
        cols = st.columns(5)
        for col, label, value in zip(cols, ["Transacciones", "Legítimas", "Fraudulentas", "Porcentaje fraude", "Modelo cargado"], [f"{len(df):,}", f"{legitimate:,}", f"{fraud:,}", f"{fraud/len(df):.2%}", next(iter(models), "No disponible")]):
            with col: card(label, value)
        st.write("")
        left, right = st.columns(2)
        with left:
            counts = df[TARGET].value_counts().rename(index={0: "Legítima", 1: "Fraudulenta"}).rename_axis("Clase").reset_index(name="Transacciones")
            fig = px.bar(counts, x="Clase", y="Transacciones", color="Clase", color_discrete_map={"Legítima": GREEN, "Fraudulenta": RED}, title="Distribución de transacciones")
            fig.update_layout(showlegend=False, paper_bgcolor="white", plot_bgcolor="white"); st.plotly_chart(fig, width="stretch")
        with right:
            fig = px.histogram(df, x="Amount", color=TARGET, nbins=60, barmode="overlay", opacity=.75, color_discrete_map={0: BLUE, 1: RED}, title="Distribución de montos", log_y=True)
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", legend_title="Clase (0 legítima, 1 fraude)"); st.plotly_chart(fig, width="stretch")
        if "Time" in df.columns:
            fig = px.histogram(df, x="Time", color=TARGET, nbins=60, barmode="overlay", opacity=.65, color_discrete_map={0: BLUE, 1: RED}, title="Transacciones según tiempo")
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white"); st.plotly_chart(fig, width="stretch")
        with st.expander("Estado del modelo"):
            if models:
                st.write("Modelo disponible:", ", ".join(models.keys()))
                if registry:
                    st.write("Ejecución compartida:", registry.get("run_id", "—"))
                    st.write("Entrenados (UTC):", registry.get("trained_at_utc", "—"))
                if metrics:
                    st.write(f"Última evaluación registrada: {metrics.get('model_type','desconocido')}")
                    if not registry: st.write("Fecha de entrenamiento: no está guardada en los artefactos.")
            else: st.warning("No se encontraron modelos entrenados.")

elif page == "Evaluación":
    section_header("Evaluación de modelos", "Resultados guardados y visualización sobre el conjunto de prueba original, sin SMOTE.")
    stored_metrics = {name: info["metrics"] for name, info in registry.get("models", {}).items()} if registry else ({metrics.get("model_type", "Modelo guardado"): metrics} if metrics else {})
    if df is not None and len(stored_metrics) < 2:
        st.info("Para comparar ambos algoritmos de forma justa, la acción siguiente los entrenará con la misma partición, StandardScaler, SMOTE y conjunto de prueba.")
        if st.button("Entrenar y guardar ambos modelos", type="primary"):
            try:
                with st.status("Entrenando Regresión Logística y SVM…", expanded=True) as status:
                    st.write("Dividiendo datos, ajustando el escalador y aplicando SMOTE solo al entrenamiento.")
                    train_models(DATA_PATH)
                    status.update(label="Ambos modelos fueron entrenados y guardados.", state="complete")
                st.cache_resource.clear(); st.cache_data.clear(); st.rerun()
            except Exception as exc:
                st.error(f"Falló el entrenamiento comparativo: {exc}")
    if stored_metrics:
        if len(stored_metrics) > 1:
            comparison = pd.DataFrame([{"Modelo": name, **{key: values.get(key) for key in ("accuracy", "precision", "recall", "f1", "average_precision", "roc_auc")}} for name, values in stored_metrics.items()])
            st.markdown("#### Comparación con el mismo conjunto de prueba")
            st.dataframe(comparison, hide_index=True, width="stretch")
            ranked = sorted(stored_metrics.items(), key=lambda item: (item[1].get("recall", 0), item[1].get("precision", 0), item[1].get("f1", 0), item[1].get("average_precision", 0)), reverse=True)
            best_name, best_values = ranked[0]
            st.info(f"Con el criterio explícito de priorizar primero Recall y, en empate, Precision, el mejor resultado de esta ejecución es **{best_name}** (Recall {best_values.get('recall', 0):.3f}, Precision {best_values.get('precision', 0):.3f}). Esto describe esta partición y no garantiza superioridad con otros datos.")
            plot_data = comparison.melt(id_vars="Modelo", var_name="Métrica", value_name="Valor")
            fig = px.bar(plot_data, x="Métrica", y="Valor", color="Modelo", barmode="group", title="Comparación de métricas (priorizar Recall, Precision, F1 y AP)")
            fig.update_layout(yaxis_range=[0, 1], paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig, width="stretch")
            model_choice = st.selectbox("Ver matriz y curvas del modelo", list(stored_metrics))
            metrics = stored_metrics[model_choice]
            name = model_choice
        else:
            name, metrics = next(iter(stored_metrics.items()))
            st.info(f"Modelo con evaluación almacenada: **{name}**. Entrena ambos para mostrar una comparación experimental.")
        cols = st.columns(4)
        for col, label, key in zip(cols, ["Precision", "Recall", "F1-score", "Average Precision"], ["precision", "recall", "f1", "average_precision"]):
            with col: card(label, f"{metrics[key]:.3f}")
        report = metrics.get("report", {})
        accuracy_value = metrics.get("accuracy", report.get("accuracy"))
        if accuracy_value is not None:
            card("Accuracy", f"{accuracy_value:.3f}", "No es suficiente por sí sola con clases desbalanceadas")
        cm = np.array(metrics["confusion_matrix"])
        tn, fp, fn, tp = cm.ravel()
        l, r = st.columns([1, 1.4])
        with l:
            fig = px.imshow(cm, text_auto=True, x=["Pred. legítima", "Pred. fraude"], y=["Real legítima", "Real fraude"], color_continuous_scale=[[0, "#eaf1f8"],[1, BLUE]], aspect="auto", title="Matriz de confusión")
            fig.update_layout(coloraxis_showscale=False, paper_bgcolor="white"); st.plotly_chart(fig, width="stretch")
            st.caption(f"TN {tn:,} · FP {fp:,} · FN {fn:,} · TP {tp:,}")
        with r:
            st.markdown("**Lectura de los resultados**")
            st.write(f"El modelo detectó {tp} de {tp+fn} fraudes reales (recall {metrics['recall']:.1%}).")
            st.write(f"De {tp+fp} alertas de fraude, {tp} fueron correctas (precision {metrics['precision']:.1%}).")
            st.write(f"Hubo {fp:,} falsos positivos (legítimas marcadas como fraude) y {fn:,} falsos negativos (fraudes no detectados).")
            st.caption("Recall: fracción de fraudes encontrados. Precision: fiabilidad de las alertas. F1: equilibrio de ambas. Average Precision: resumen del área bajo Precision‑Recall.")
        if df is not None and models:
            model_name = name if name in models else next(iter(models))
            try:
                scored = evaluation_scores(df, models[model_name][0], models[model_name][1])
                if scored:
                    y_true, y_score = scored
                    p, rec, _ = precision_recall_curve(y_true, y_score)
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = go.Figure(go.Scatter(x=rec, y=p, mode="lines", line={"color": BLUE, "width": 3}, name=model_name))
                        fig.update_layout(title=f"Curva Precision–Recall · AP {average_precision_score(y_true,y_score):.3f}", xaxis_title="Recall", yaxis_title="Precision", paper_bgcolor="white", plot_bgcolor="white"); st.plotly_chart(fig, width="stretch")
                    with c2:
                        auc = roc_auc_score(y_true, y_score)
                        fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", line={"color": GREEN, "width": 3}, name=f"AUC {auc:.3f}"))
                        fig.add_shape(type="line", x0=0,y0=0,x1=1,y1=1,line={"dash":"dash","color":"#9aa9b8"})
                        fig.update_layout(title=f"Curva ROC · AUC {auc:.3f}", xaxis_title="Falsos positivos", yaxis_title="Verdaderos positivos", paper_bgcolor="white", plot_bgcolor="white"); st.plotly_chart(fig, width="stretch")
            except Exception as exc:
                st.warning(f"No se pudieron reconstruir las curvas con los artefactos disponibles: {exc}")
    else:
        st.info("Todavía no hay métricas guardadas. Carga el dataset y entrena ambos modelos desde esta sección.")

elif page == "Probar transacción":
    section_header("Análisis individual", "Prueba con valores manuales o selecciona una fila real del dataset para una demostración.")
    if not models:
        st.warning("No hay modelos entrenados disponibles.")
    else:
        model_name = st.selectbox("Modelo", list(models.keys()))
        artifacts, scaler = models[model_name]
        features = artifacts["features"]
        options = ["Seleccionar ejemplo del dataset", "Introducción manual"] if df is not None else ["Introducción manual"]
        mode = st.radio("Origen de la transacción", options, horizontal=True)
        if mode == "Seleccionar ejemplo del dataset" and df is not None:
            idx = st.selectbox("Fila del dataset", range(min(1000, len(df))), format_func=lambda i: f"Registro {i+1} · monto ${df.iloc[i].get('Amount', 0):.2f}")
            row = df.iloc[[idx]]
            values = row.loc[:, features].copy()
            actual = int(row[TARGET].iloc[0])
        else:
            st.info("V1–V28 no son datos como país, tienda o tipo de compra. Son 28 componentes matemáticos creados con PCA a partir de datos anonimizados. Cada componente combina patrones de varias variables originales; por eso no tiene una descripción cotidiana independiente.")
            st.caption("Para una demostración más realista, selecciona una transacción del dataset. Si introduces valores manualmente, usa valores de una fila compatible con este dataset; dejar todo en cero solo sirve como ejemplo técnico.")
            with st.form("manual_prediction"):
                values_dict = {}
                groups = [
                    ("Datos directos", [c for c in features if c in ("Time", "Amount")]),
                    ("Componentes PCA · V1–V14", [c for c in features if c.startswith("V")][:14]),
                    ("Componentes PCA · V15–V28", [c for c in features if c.startswith("V")][14:]),
                ]
                for group_name, group in groups:
                    with st.expander(group_name, expanded=(group_name == "Datos directos")):
                        if group_name.startswith("Componentes PCA"):
                            st.caption("Son valores transformados y anonimizados. El número identifica el componente, no una categoría bancaria.")
                        cols = st.columns(2)
                        for j, col in enumerate(group):
                            if col.startswith("V"):
                                label = f"{col} · componente PCA {int(col[1:])}"
                                help_text = "Variable transformada mediante PCA. No tiene una interpretación individual directa; usa un valor de una transacción del dataset."
                            else:
                                label = "Tiempo desde la primera operación" if col == "Time" else "Monto de la transacción"
                                help_text = "Tiempo registrado en el dataset." if col == "Time" else "Importe original de la transacción."
                            with cols[j % 2]:
                                values_dict[col] = st.number_input(label, value=0.0, format="%.6f", key=f"manual_{col}", help=help_text)
                submitted = st.form_submit_button("Clasificar transacción", type="primary")
            values = pd.DataFrame([values_dict], columns=features)
            actual = None
        if mode == "Seleccionar ejemplo del dataset":
            submitted = st.button("Clasificar transacción", type="primary")
        if submitted:
            try:
                pred, prob = predict_frame(values, model_name, models)
                if pred[0] == 1:
                    st.error("Posible transacción fraudulenta", icon="🚨")
                else: st.success("Transacción clasificada como legítima", icon="✅")
                st.write(f"Modelo utilizado: **{model_name}**")
                if prob is not None: st.metric("Probabilidad estimada de fraude", f"{prob[0]:.1%}")
                if actual is not None:
                    st.caption(f"Etiqueta real del dataset: {'Fraudulenta' if actual else 'Legítima'} · mostrada por separado de la predicción.")
            except Exception as exc: st.error(f"No se pudo predecir: {exc}")
    st.warning("Prototipo académico; no utilizar sus resultados para tomar decisiones financieras reales.")

elif page == "Análisis masivo":
    section_header("Análisis masivo mediante CSV", "Carga transacciones con las variables de entrada requeridas; se conservarán las columnas originales en la exportación.")
    if not models: st.warning("No hay modelos entrenados disponibles.")
    else:
        model_name = st.selectbox("Modelo para análisis", list(models.keys()))
        upload = st.file_uploader("Archivo CSV", type=["csv"])
        if upload:
            try:
                raw = upload.getvalue()
                try:
                    header = next(csv.reader(io.StringIO(raw.decode("utf-8-sig"))), [])
                except UnicodeDecodeError as exc:
                    raise ValueError("El CSV debe estar codificado en UTF-8.") from exc
                duplicates = sorted({name for name in header if header.count(name) > 1})
                if duplicates: raise ValueError("El CSV contiene nombres de columna duplicados: " + ", ".join(duplicates))
                batch = pd.read_csv(io.BytesIO(raw))
                if batch.empty: raise ValueError("El CSV está vacío.")
                artifacts, _ = models[model_name]
                missing = [c for c in artifacts["features"] if c not in batch.columns]
                if missing: raise ValueError("Faltan columnas: " + ", ".join(missing))
                if st.button("Analizar archivo", type="primary"):
                    progress = st.progress(0, text="Preparando transacciones…")
                    try:
                        pred, prob = predict_frame(batch, model_name, models, lambda value: progress.progress(value, text=f"Clasificando… {value:.0%}"))
                    finally:
                        progress.empty()
                    result = batch.copy()
                    result["Prediction"] = pred
                    result["Predicted_Label"] = np.where(pred == 1, "Fraudulenta", "Legítima")
                    result["Model"] = model_name
                    if TARGET in result.columns:
                        real = pd.to_numeric(result[TARGET], errors="raise")
                        if real.isna().any() or not set(real.unique()).issubset({0, 1}):
                            raise ValueError("La columna Class debe contener solo valores 0 y 1, sin vacíos.")
                        real = real.astype(int)
                        result["Real_Label"] = np.where(real == 1, "Fraudulenta", "Legítima")
                    if prob is not None: result["Fraud_Probability"] = prob
                    counts = pd.Series(pred).value_counts()
                    c1,c2,c3=st.columns(3)
                    with c1: card("Analizadas", f"{len(result):,}")
                    with c2: card("Legítimas", f"{counts.get(0,0):,}")
                    with c3: card("Posible fraude", f"{counts.get(1,0):,}")
                    st.dataframe(result, width="stretch", height=420)
                    if TARGET in result.columns:
                        st.caption("Evaluación frente a Class del archivo (no se usó como característica): "
                                   f"accuracy {accuracy_score(real, pred):.3f} · recall fraude {recall_score(real, pred, zero_division=0):.3f} · F1 fraude {f1_score(real, pred, zero_division=0):.3f}")
                    st.download_button("Descargar resultados CSV", result.to_csv(index=False).encode("utf-8-sig"), "fraudguard_results.csv", "text/csv", type="primary")
            except Exception as exc: st.error(f"Archivo no válido: {exc}")

else:
    section_header("Acerca de FraudGuard AI", "Proyecto académico de Inteligencia de Negocios para clasificación binaria de transacciones.")
    st.markdown("""
    ### Propósito
    FraudGuard AI estima si una transacción es legítima o posiblemente fraudulenta a partir de datos históricos. Es un prototipo académico y no una herramienta de decisión financiera.

    ### Datos y método
    Se utiliza el dataset **Credit Card Fraud Detection** (`mlg-ulb/creditcardfraud`) de Kaggle. `Class` representa la etiqueta: 0 legítima y 1 fraude. Las variables V1–V28 son componentes transformados por PCA.

    La preparación divide primero entrenamiento y prueba. `StandardScaler` aprende sus parámetros solo con entrenamiento; SMOTE balancea solo ese entrenamiento. La prueba mantiene su distribución original y nunca se sobremuestrea.

    ### Modelos y métricas
    El proyecto implementa Regresión Logística y SVM lineal calibrado. Precision mide qué proporción de alertas fue correcta; Recall mide cuántos fraudes reales se detectaron; F1 equilibra ambas; Average Precision resume la curva Precision–Recall. Los falsos positivos son legítimas marcadas como fraude y los falsos negativos son fraudes no detectados.

    ### Tecnologías
    Python · Streamlit · Pandas · NumPy · Scikit-learn · Imbalanced-learn · Plotly · Joblib
    """)

st.divider()
st.caption("FraudGuard AI · Herramienta académica de apoyo a la demostración de Machine Learning.")

