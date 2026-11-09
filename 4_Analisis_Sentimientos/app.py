"""Interfaz Streamlit para clasificar reseñas de productos."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras

from utils import limpiar


BASE = Path(__file__).resolve().parent
RUTA_MODELO = BASE / "models" / "lstm.keras"
RUTA_REPORTES = BASE / "reports"
CLASES = ["Negativo", "Neutral", "Positivo"]
ICONOS = {"Negativo": "😠", "Neutral": "😐", "Positivo": "😊"}

st.set_page_config(page_title="Sentimiento en reseñas", page_icon="💬", layout="wide")


@st.cache_resource
def cargar_modelo():
    return keras.models.load_model(RUTA_MODELO)


def predecir(textos):
    textos_limpios = tf.constant([limpiar(texto) for texto in textos], dtype=tf.string)
    return cargar_modelo().predict(textos_limpios, verbose=0)


st.title("💬 Análisis de sentimiento en reseñas")
st.write(
    "Clasifica reseñas de productos como **negativas, neutrales o positivas**. "
    "El modelo usa embeddings y una red neuronal recurrente BiLSTM entrenada en español."
)

if not RUTA_MODELO.exists():
    st.error("Todavía no hay un modelo entrenado. Ejecuta `python entrenar.py` y vuelve a iniciar la aplicación.")
    st.stop()

tab_individual, tab_masivo = st.tabs(["Una reseña", "Carga masiva (CSV)"])

with tab_individual:
    texto = st.text_area(
        "Escribe una reseña",
        height=130,
        placeholder="Ej.: El producto llegó rápido y funciona perfectamente.",
    )
    if st.button("Analizar", type="primary"):
        if not texto.strip():
            st.warning("Escribe una reseña primero.")
        else:
            probabilidades = predecir([texto])[0]
            indice = int(np.argmax(probabilidades))
            etiqueta = CLASES[indice]
            st.success(f"{ICONOS[etiqueta]} **{etiqueta}** ({probabilidades[indice] * 100:.1f}% de confianza)")
            st.caption("La confianza es la probabilidad estimada por el modelo, no una garantía de acierto.")
            st.bar_chart(pd.DataFrame({"Probabilidad (%)": probabilidades * 100}, index=CLASES))

with tab_masivo:
    archivo = st.file_uploader("Sube un CSV con reseñas", type=["csv"])
    if archivo is not None:
        try:
            df = pd.read_csv(archivo)
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            st.error(f"No se pudo leer el archivo CSV: {error}")
            df = pd.DataFrame()

        if not df.empty:
            columnas_texto = [col for col in df.columns if df[col].dtype == "object"]
            if not columnas_texto:
                st.error("El CSV debe incluir al menos una columna de texto.")
            else:
                columna = st.selectbox("Columna con el texto de la reseña", columnas_texto)
                if st.button("Analizar todas"):
                    reseñas = df[columna].fillna("").astype(str).tolist()
                    probabilidades = predecir(reseñas)
                    indices = np.argmax(probabilidades, axis=1)
                    df_resultado = df.copy()
                    df_resultado["Sentimiento"] = [CLASES[i] for i in indices]
                    df_resultado["Confianza (%)"] = (probabilidades.max(axis=1) * 100).round(1)
                    st.dataframe(df_resultado, use_container_width=True)
                    st.bar_chart(df_resultado["Sentimiento"].value_counts())
                    st.download_button(
                        "Descargar resultados",
                        df_resultado.to_csv(index=False).encode("utf-8-sig"),
                        "resultados_sentimiento.csv",
                        "text/csv",
                    )

st.divider()
st.subheader("📊 Evaluación del modelo")
grafico_accuracy = RUTA_REPORTES / "accuracy.png"
grafico_matriz = RUTA_REPORTES / "matriz.png"
ruta_metricas = RUTA_REPORTES / "metricas.json"
if grafico_accuracy.exists() and grafico_matriz.exists():
    c1, c2 = st.columns(2)
    c1.image(str(grafico_accuracy), caption="Exactitud por época")
    c2.image(str(grafico_matriz), caption="Matriz de confusión en el conjunto de prueba")
    if ruta_metricas.exists():
        metricas = pd.read_json(ruta_metricas).T
        columnas = [col for col in ("precision", "recall", "f1-score", "support") if col in metricas]
        st.caption("Métricas calculadas sobre reseñas de prueba reservadas para evaluación.")
        st.dataframe(metricas.loc[[*CLASES, "macro avg", "weighted avg"], columnas].round(3),
                     use_container_width=True)
else:
    st.info("Los reportes se generan al entrenar el modelo con `python entrenar.py`.")
