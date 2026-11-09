"""Interfaz web para clasificar imágenes de gatos y perros."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "resnet50.keras"
REPORTS_DIR = PROJECT_DIR / "reports"
HISTORY_PATH = REPORTS_DIR / "entrenamiento.png"
MATRIX_PATH = REPORTS_DIR / "matriz_confusion.png"
METRICS_PATH = REPORTS_DIR / "metricas.json"
IMAGE_SIZE = (224, 224)
CLASSES = ("Gato", "Perro")

st.set_page_config(
    page_title="Huellas | Perros y gatos",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#172820; --muted:#64736a; --green:#315c45; --lime:#d9efbd; --paper:#f7f8f3; }
    .stApp { background:var(--paper); color:var(--ink); }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
    h1,h2,h3 { font-family:'Manrope', sans-serif; letter-spacing:-.035em; color:var(--ink); }
    [data-testid="stSidebar"] { background:#edf1e8; border-right:1px solid #dfe5db; }
    [data-testid="stSidebar"] h2 { font-size:1rem; }
    .hero { padding:2.2rem 2.4rem; border-radius:26px; color:#fff; background:linear-gradient(118deg,#234b37 0%,#376c4d 58%,#648c58 100%); position:relative; overflow:hidden; margin:.3rem 0 1.6rem; }
    .hero:after { content:'🐾'; position:absolute; right:5%; top:-25px; font-size:9rem; opacity:.12; transform:rotate(-18deg); }
    .hero-kicker { text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; color:#d9efbd; }
    .hero h1 { color:white; font-size:clamp(2rem,4vw,3.2rem); margin:.45rem 0 .55rem; }
    .hero p { max-width:690px; color:#e7efe6; font-size:1.03rem; margin:0; }
    .section-label { color:#66816d; text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:700; margin-bottom:.25rem; }
    .soft-card { background:#fff; border:1px solid #e4e9e0; padding:1.1rem 1.25rem; border-radius:18px; box-shadow:0 8px 24px rgba(36,57,43,.045); }
    .result-card { background:#fff; border:1px solid #e3e9df; border-radius:20px; padding:1.1rem; box-shadow:0 8px 24px rgba(36,57,43,.05); height:100%; }
    .class-pill { display:inline-block; padding:.35rem .7rem; border-radius:99px; background:#eaf4df; color:#315c45; font-weight:700; font-size:.82rem; }
    .footer { color:#718077; font-size:.82rem; border-top:1px solid #e1e7df; padding-top:1rem; margin-top:2.5rem; }
    div[data-testid="stMetric"] { background:white; border:1px solid #e3e9df; border-radius:16px; padding:1rem; }
    div[data-testid="stMetricLabel"] { color:#66756b; }
    .stButton>button { border-radius:12px; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)


def preparar_imagen(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """Normaliza orientación y prepara una imagen para ResNet50."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    resized = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    batch = np.asarray(resized, dtype=np.float32)[None, ...]
    return image, preprocess_input(batch)


@st.cache_resource(show_spinner="Cargando ResNet50…")
def cargar_modelo(model_path: str, modified_at: float):
    # modified_at invalida la caché después de reemplazar el archivo del modelo.
    del modified_at
    return keras.models.load_model(model_path, compile=False)


def cargar_metricas() -> dict:
    if not METRICS_PATH.is_file():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def renderizar_resultados(files, model) -> None:
    if not files:
        st.markdown(
            '<div class="soft-card">📷 <b>Sube una foto para empezar.</b> '
            'Puedes seleccionar varias imágenes y compararlas en una sola pasada.</div>',
            unsafe_allow_html=True,
        )
        return

    valid_images: list[tuple[str, Image.Image, np.ndarray]] = []
    for uploaded in files:
        try:
            with Image.open(uploaded) as image:
                display_image, batch = preparar_imagen(image)
            valid_images.append((uploaded.name, display_image, batch))
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
            st.warning(f"No pude leer **{uploaded.name}**: {exc}")

    if not valid_images:
        return

    input_batch = np.concatenate([item[2] for item in valid_images], axis=0)
    probabilities = model.predict(input_batch, verbose=0).reshape(-1)
    columns = st.columns(2, gap="large")

    for index, ((filename, image, _), dog_probability) in enumerate(zip(valid_images, probabilities)):
        dog_probability = float(np.clip(dog_probability, 0.0, 1.0))
        class_probabilities = (1.0 - dog_probability, dog_probability)
        predicted_index = int(np.argmax(class_probabilities))
        confidence = class_probabilities[predicted_index]
        predicted_class = CLASSES[predicted_index]

        with columns[index % 2]:
            st.image(image, use_container_width=True)
            st.markdown(f'<span class="class-pill">{predicted_class}</span>', unsafe_allow_html=True)
            st.markdown(f"**{filename}**")
            st.metric("Confianza del modelo", f"{confidence:.1%}")
            if confidence < 0.70:
                st.info("La predicción tiene confianza moderada. Prueba con una foto más clara y centrada.")
            for label, probability in zip(CLASSES, class_probabilities):
                st.progress(float(probability), text=f"{label}: {probability:.1%}")


def renderizar_evaluacion(metrics: dict) -> None:
    st.markdown('<div class="section-label">Rendimiento</div>', unsafe_allow_html=True)
    st.header("Evaluación del modelo")
    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Exactitud", f"{metrics.get('accuracy', 0):.1%}")
        m2.metric("Precisión", f"{metrics.get('precision', 0):.1%}")
        m3.metric("Recall", f"{metrics.get('recall', 0):.1%}")
        m4.metric("F1", f"{metrics.get('f1_score', 0):.1%}")
        if metrics.get("test_images"):
            st.caption(f"Evaluado con {metrics['test_images']:,} imágenes de prueba.")

    chart_col, matrix_col = st.columns(2, gap="large")
    with chart_col:
        st.markdown("#### Curvas de aprendizaje")
        if HISTORY_PATH.is_file():
            st.image(str(HISTORY_PATH), use_container_width=True)
        else:
            st.info("El gráfico aparecerá después de completar el entrenamiento.")
    with matrix_col:
        st.markdown("#### Matriz de confusión")
        if MATRIX_PATH.is_file():
            st.image(str(MATRIX_PATH), use_container_width=True)
        else:
            st.info("La matriz aparecerá después de completar la evaluación.")

    with st.expander("¿Qué significan estas métricas?"):
        st.write(
            "**Exactitud** mide cuántas predicciones fueron correctas. **Precisión** indica "
            "qué proporción de las predicciones positivas fue correcta. **Recall** mide "
            "cuántos casos positivos encontró el modelo y **F1** resume precisión y recall. "
            "La matriz cruza la clase real con la predicción."
        )


st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">Visión avanzada · Transfer Learning</div>
      <h1>¿Quién aparece en la foto?</h1>
      <p>Un clasificador visual que reutiliza ResNet50, una red convolucional preentrenada,
      para reconocer gatos y perros en fotografías reales.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

metrics = cargar_metricas()
with st.sidebar:
    st.markdown("## 🐾 Huellas")
    st.caption("Clasificación de imágenes con aprendizaje por transferencia")
    st.markdown("---")
    st.markdown("### Sobre el modelo")
    st.write("**Arquitectura** · ResNet50")
    st.write("**Técnica** · Transfer Learning")
    st.write("**Clases** · Gato y perro")
    if metrics.get("training_date"):
        st.caption(f"Último entrenamiento: {metrics['training_date']}")
    st.markdown("---")
    if MODEL_PATH.is_file():
        st.success("Modelo listo para clasificar")
    else:
        st.warning("Falta entrenar el modelo")
        st.code('& ".\\.venv\\Scripts\\python.exe" ".\\entrenar.py"', language="powershell")
    st.caption("Las imágenes se procesan durante esta sesión.")

tab_classifier, tab_evaluation = st.tabs(["📷 Clasificador", "📊 Evaluación"])

with tab_classifier:
    st.markdown('<div class="section-label">Clasificador de imágenes</div>', unsafe_allow_html=True)
    st.header("Analiza tus fotografías")
    st.write("Carga una o varias imágenes. El modelo devuelve la clase estimada y su nivel de confianza.")
    uploads = st.file_uploader(
        "Selecciona fotos de perros o gatos",
        type=("jpg", "jpeg", "png", "webp"),
        accept_multiple_files=True,
        help="Formatos admitidos: JPG, PNG y WEBP.",
    )

    if MODEL_PATH.is_file():
        try:
            model = cargar_modelo(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
            renderizar_resultados(uploads, model)
        except Exception as exc:
            st.error("No se pudo cargar el modelo. Vuelve a entrenarlo y revisa el mensaje técnico.")
            with st.expander("Detalle del error"):
                st.exception(exc)
    else:
        st.info("Todavía no hay un modelo guardado. Ejecuta este comando desde la carpeta del proyecto:")
        st.code('& ".\\.venv\\Scripts\\python.exe" ".\\entrenar.py"', language="powershell")

with tab_evaluation:
    renderizar_evaluacion(metrics)

st.markdown(
    '<div class="footer">Proyecto académico de Visión Avanzada · TensorFlow/Keras · ResNet50</div>',
    unsafe_allow_html=True,
)
