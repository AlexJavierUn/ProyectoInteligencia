"""Entrena y evalúa una red BiLSTM para reseñas de productos en español."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

from utils import limpiar


BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
REPORTS = BASE / "reports"
CLASES = ["Negativo", "Neutral", "Positivo"]
NUM_CLASES = len(CLASES)
MAX_VOCAB = 20_000
LONGITUD = 100
N_TRAIN = 30_000
EPOCAS = 5
SEMILLA = 42

# Carga los JSONL directamente para no depender del script remoto del dataset.
URL_BASE = "https://huggingface.co/datasets/mteb/amazon_reviews_multi/resolve/main/es"
ARCHIVOS = {
    split: f"{URL_BASE}/{split}.jsonl"
    for split in ("train", "validation", "test")
}


def preparar_split(dataset, cantidad=None):
    """Devuelve textos y clases; las etiquetas originales son estrellas 0–4."""
    dataset = dataset.shuffle(seed=SEMILLA)
    if cantidad is not None:
        dataset = dataset.select(range(min(cantidad, len(dataset))))

    textos = [limpiar(texto or "") for texto in dataset["text"]]
    # label 0–1: 1–2 estrellas; label 2: 3 estrellas; label 3–4: 4–5 estrellas.
    etiquetas = [0 if valor <= 1 else 1 if valor == 2 else 2
                 for valor in dataset["label"]]
    return textos, np.asarray(etiquetas, dtype=np.int32)


def guardar_graficos(historial, y_real, y_pred):
    REPORTS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(historial.history["accuracy"], label="Entrenamiento")
    ax.plot(historial.history["val_accuracy"], label="Validación")
    ax.set(title="Exactitud por época — BiLSTM", xlabel="Época", ylabel="Exactitud")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / "accuracy.png", dpi=150)
    plt.close(fig)

    matriz = confusion_matrix(y_real, y_pred, labels=range(NUM_CLASES))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(matriz, cmap="Blues")
    ax.set_xticks(range(NUM_CLASES), labels=CLASES, rotation=20, ha="right")
    ax.set_yticks(range(NUM_CLASES), labels=CLASES)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    ax.set_title("Matriz de confusión")
    for fila in range(NUM_CLASES):
        for columna in range(NUM_CLASES):
            ax.text(columna, fila, str(matriz[fila, columna]),
                    ha="center", va="center",
                    color="white" if matriz[fila, columna] > matriz.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(REPORTS / "matriz.png", dpi=150)
    plt.close(fig)

    informe = classification_report(
        y_real, y_pred, labels=range(NUM_CLASES), target_names=CLASES,
        output_dict=True, zero_division=0,
    )
    with (REPORTS / "metricas.json").open("w", encoding="utf-8") as archivo:
        json.dump(informe, archivo, ensure_ascii=False, indent=2)
    return informe


def main():
    np.random.seed(SEMILLA)
    tf.random.set_seed(SEMILLA)

    datos = load_dataset("json", data_files=ARCHIVOS)
    x_train, y_train = preparar_split(datos["train"], N_TRAIN)
    x_val, y_val = preparar_split(datos["validation"])
    x_test, y_test = preparar_split(datos["test"])
    # TensorFlow/Keras en Windows no acepta directamente arrays NumPy con dtype <U.
    x_train = tf.constant(x_train, dtype=tf.string)
    x_val = tf.constant(x_val, dtype=tf.string)
    x_test = tf.constant(x_test, dtype=tf.string)
    print(f"Reseñas: {len(x_train)} entrenamiento | {len(x_val)} validación | {len(x_test)} prueba")

    for nombre, etiquetas in (("entrenamiento", y_train), ("validación", y_val), ("prueba", y_test)):
        conteos = np.bincount(etiquetas, minlength=NUM_CLASES)
        print(f"Clases en {nombre}: " + ", ".join(
            f"{CLASES[i]}={conteos[i]}" for i in range(NUM_CLASES)
        ))

    vectorizador = layers.TextVectorization(
        max_tokens=MAX_VOCAB,
        output_mode="int",
        output_sequence_length=LONGITUD,
        name="vectorizador",
    )
    vectorizador.adapt(x_train)

    entrada = keras.Input(shape=(), dtype="string", name="resena")
    tokens = vectorizador(entrada)
    x = layers.Embedding(MAX_VOCAB, 64, mask_zero=True, name="embeddings")(tokens)
    x = layers.Bidirectional(layers.LSTM(64), name="bilstm")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    salida = layers.Dense(NUM_CLASES, activation="softmax", name="sentimiento")(x)
    modelo = keras.Model(entrada, salida, name="clasificador_resenas_bilstm")
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    modelo.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True,
                                      monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.5, monitor="val_loss"),
    ]
    historial = modelo.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCAS,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )

    perdida, exactitud = modelo.evaluate(x_test, y_test, verbose=0)
    predicciones = np.argmax(modelo.predict(x_test, verbose=0), axis=1)
    informe = guardar_graficos(historial, y_test, predicciones)
    MODELS.mkdir(parents=True, exist_ok=True)
    modelo.save(MODELS / "lstm.keras")

    print(f"Exactitud en prueba: {exactitud:.4f} | Pérdida: {perdida:.4f}")
    print(f"F1 macro en prueba: {informe['macro avg']['f1-score']:.4f}")
    print(classification_report(
        y_test, predicciones, labels=range(NUM_CLASES), target_names=CLASES,
        zero_division=0,
    ))
    print(f"Modelo y reportes guardados en {BASE}")


if __name__ == "__main__":
    main()
