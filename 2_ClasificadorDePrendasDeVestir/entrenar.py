from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # para guardar imágenes sin abrir ventanas
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

CLASES = ["Camiseta", "Pantalón", "Suéter", "Vestido", "Abrigo",
          "Sandalia", "Camisa", "Zapatilla", "Bolso", "Bota"]
BASE_DIR = Path(__file__).resolve().parent

# 1. Cargar y preparar datos
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32")[..., None] / 255.0
x_test = x_test.astype("float32")[..., None] / 255.0

# 2. Definir los modelos
def crear_mlp():
    return keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

def crear_cnn():
    return keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        # Aumentos pequeños ayudan a que la CNN tolere ligeros cambios de posición.
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomRotation(0.04),
        layers.Conv2D(32, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(10, activation="softmax"),
    ])

# 3. Funciones para los gráficos del informe
def guardar_accuracy(historial, nombre):
    plt.figure(figsize=(6, 4))
    plt.plot(historial["accuracy"], label="Entrenamiento")
    plt.plot(historial["val_accuracy"], label="Validación")
    plt.title(f"Accuracy por época - {nombre.upper()}")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "reports" / f"accuracy_{nombre}.png", dpi=150)
    plt.close()

def guardar_matriz(y_real, y_pred, nombre):
    cm = tf.math.confusion_matrix(y_real, y_pred, num_classes=10).numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASES, rotation=45, ha="right")
    ax.set_yticklabels(CLASES)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión - {nombre.upper()}")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8)
    fig.tight_layout()
    fig.savefig(BASE_DIR / "reports" / f"matriz_{nombre}.png", dpi=150)
    plt.close(fig)

# 4. Entrenar, evaluar y guardar
(BASE_DIR / "models").mkdir(exist_ok=True)
(BASE_DIR / "reports").mkdir(exist_ok=True)
metricas_path = BASE_DIR / "reports" / "metricas.json"
metricas = {}

for nombre, crear in [("mlp", crear_mlp), ("cnn", crear_cnn)]:
    print(f"\n=== Entrenando {nombre.upper()} ===")
    modelo = crear()
    modelo.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, mode="max",
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2,
            min_lr=1e-5, verbose=1,
        ),
    ]
    h = modelo.fit(
        x_train, y_train, epochs=25, batch_size=128,
        validation_split=0.1, callbacks=callbacks, verbose=2,
    )

    loss, acc = modelo.evaluate(x_test, y_test, verbose=0)
    print(f">>> Accuracy en prueba ({nombre.upper()}): {acc:.4f}")
    metricas[nombre] = {"accuracy": float(acc), "loss": float(loss)}
    metricas_path.write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    y_pred = np.argmax(modelo.predict(x_test, verbose=0), axis=1)
    guardar_accuracy(h.history, nombre)
    guardar_matriz(y_test, y_pred, nombre)

    modelo.save(BASE_DIR / "models" / f"{nombre}.keras")
