import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras
from pathlib import Path

CLASES = ["Camiseta", "Pantalón", "Suéter", "Vestido", "Abrigo",
          "Sandalia", "Camisa", "Zapatilla", "Bolso", "Bota"]
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Clasificador de prendas",
    page_icon="👕",
    layout="wide",
)

st.markdown("""
<style>
    .block-container {max-width: 1100px; padding-top: 2rem;}
    .hero {
        padding: 1.5rem 1.8rem; border-radius: 18px;
        background: linear-gradient(120deg, #eef4ff, #f4efff);
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; color: #172554;}
    .hero p {margin: .55rem 0 0; color: #475569;}
    [data-testid="stFileUploader"] {
        padding: .6rem; border: 1px dashed #94a3b8; border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo(nombre):
    ruta = BASE_DIR / "models" / f"{nombre}.keras"
    return keras.models.load_model(ruta)

def preparar_imagen(img):
    img = ImageOps.grayscale(img).resize((28, 28))
    arr = np.array(img).astype("float32")
    # Fashion MNIST: prenda clara sobre fondo oscuro. Si el fondo es claro, invertir.
    borde = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if borde.mean() > 127:
        arr = 255 - arr
    arr = arr / 255.0
    return arr.reshape(1, 28, 28, 1)

st.markdown("""
<div class="hero">
  <h1>👕 Clasificador de prendas</h1>
  <p>Explora cómo una red neuronal reconoce las categorías de Fashion MNIST.</p>
</div>
""", unsafe_allow_html=True)

with st.container(border=True):
    st.subheader("Prueba una imagen")
    st.caption("El modelo fue entrenado con Fashion MNIST: imágenes de 28 × 28 píxeles, en escala de grises y con la prenda centrada. Las fotos cotidianas pueden diferir de ese formato.")
    control1, control2 = st.columns([1, 2])
    with control1:
        opcion = st.radio("Arquitectura", ["CNN", "MLP"], horizontal=True)
    with control2:
        archivo = st.file_uploader("Selecciona una imagen", type=["png", "jpg", "jpeg"])

if archivo:
    try:
        img = ImageOps.exif_transpose(Image.open(archivo)).convert("RGB")
        col1, col2 = st.columns(2)
        col1.image(img, caption="Imagen original", use_container_width=True)

        entrada = preparar_imagen(img)
        col2.image(
            entrada.reshape(28, 28),
            caption="Vista en escala de grises (28 × 28)",
            clamp=True,
            width=220,
        )

        if st.button("Clasificar imagen", type="primary", use_container_width=True):
            ruta_modelo = BASE_DIR / "models" / f"{opcion.lower()}.keras"
            if not ruta_modelo.exists():
                st.error("No encuentro el modelo entrenado. Ejecuta `python entrenar.py` primero.")
            else:
                modelo = cargar_modelo(opcion.lower())
                probs = modelo.predict(entrada, verbose=0)[0]
                mejores = np.argsort(probs)[::-1][:3]
                indice = int(mejores[0])
                st.success(f"Predicción principal: **{CLASES[indice]}**")
                st.caption("Puntajes del modelo (no son una garantía de acierto)")
                resultados = pd.DataFrame({
                    "Categoría": [CLASES[i] for i in mejores],
                    "Puntaje": [float(probs[i]) for i in mejores],
                }).set_index("Categoría")
                st.bar_chart(resultados)
    except (OSError, ValueError) as error:
        st.error(f"No pude procesar esa imagen: {error}")

st.divider()
st.header("📊 Cómo aprendió el modelo")
st.caption("Compara el desempeño durante el entrenamiento y los errores por categoría en el conjunto de prueba.")

tab_cnn, tab_mlp = st.tabs(["CNN", "MLP"])
for tab, nombre in [(tab_cnn, "cnn"), (tab_mlp, "mlp")]:
    with tab:
        metricas_path = BASE_DIR / "reports" / "metricas.json"
        if metricas_path.exists():
            try:
                metricas = json.loads(metricas_path.read_text(encoding="utf-8"))
                if nombre in metricas:
                    st.metric(
                        "Accuracy en prueba",
                        f"{metricas[nombre]['accuracy']:.1%}",
                    )
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                st.info("No se pudieron leer las métricas guardadas.")
        ruta_accuracy = BASE_DIR / "reports" / f"accuracy_{nombre}.png"
        ruta_matriz = BASE_DIR / "reports" / f"matriz_{nombre}.png"
        if ruta_accuracy.exists() and ruta_matriz.exists():
            c1, c2 = st.columns(2)
            c1.image(str(ruta_accuracy), caption="Accuracy por época")
            c2.image(str(ruta_matriz), caption="Matriz de confusión")
        else:
            st.info("Ejecuta `python entrenar.py` para generar los reportes.")
