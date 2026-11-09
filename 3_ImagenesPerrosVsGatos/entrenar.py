"""Entrena y evalúa un clasificador gato/perro con ResNet50 preentrenada."""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.image_classification import cats_vs_dogs as cats_vs_dogs_dataset
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
MODEL_PATH = MODELS_DIR / "resnet50.keras"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ("Gato", "Perro")  # cats_vs_dogs: 0 = cat, 1 = dog


def aplicar_compatibilidad_rutas_windows() -> None:
    """Evita que TFDS genere ZIP en memoria con separadores Windows incompatibles."""
    if os.name != "nt":
        return

    def generar_ejemplos_con_rutas_posix(self, archive):
        skipped = 0
        for filename, file_object in archive:
            # ZIP almacena rutas con '/', aunque el entrenamiento se ejecute en Windows.
            normalized_name = filename.replace("\\", "/")
            match = cats_vs_dogs_dataset._NAME_RE.match(normalized_name)
            if not match:
                continue
            label = match.group(1).lower()
            if tf.compat.as_bytes("JFIF") not in file_object.peek(10):
                skipped += 1
                continue

            image_data = file_object.read()
            image_tensor = tf.image.decode_image(image_data)
            recoded_image = tf.io.encode_jpeg(image_tensor)
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as image_zip:
                image_zip.writestr(normalized_name, recoded_image.numpy())
            buffer.seek(0)
            image_file = zipfile.ZipFile(buffer).open(normalized_name)
            yield normalized_name, {
                "image": image_file,
                "image/filename": normalized_name,
                "label": label,
            }

        expected_corrupt = cats_vs_dogs_dataset._NUM_CORRUPT_IMAGES
        if skipped != expected_corrupt:
            raise ValueError(f"Se esperaban {expected_corrupt} imágenes corruptas y se encontraron {skipped}.")

    # TFDS 4.9.x normaliza rutas con os.path.normpath, que en Windows produce '\\'.
    # ZipFile guarda esos nombres con '/', por lo que el .open() posterior falla.
    cats_vs_dogs_dataset.CatsVsDogs._generate_examples = generar_ejemplos_con_rutas_posix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un clasificador de perros y gatos con ResNet50 y Transfer Learning."
    )
    parser.add_argument("--epochs", type=int, default=5, help="Épocas para entrenar la nueva cabeza.")
    parser.add_argument(
        "--fine-tune-epochs", type=int, default=2,
        help="Épocas de ajuste fino; usa 0 para omitir esta fase.",
    )
    parser.add_argument(
        "--fine-tune-layers", type=int, default=30,
        help="Capas finales de ResNet50 que se descongelan durante el ajuste fino.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Imágenes por lote.")
    parser.add_argument(
        "--max-train", type=int, default=6000,
        help="Máximo de imágenes de entrenamiento; usa 0 para utilizar toda la partición.",
    )
    parser.add_argument(
        "--max-eval", type=int, default=1500,
        help="Máximo de imágenes para validación y prueba; usa 0 para usar todas.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad.")
    args = parser.parse_args()
    if args.epochs < 1 or args.fine_tune_epochs < 0 or args.batch_size < 1:
        parser.error("epochs y batch-size deben ser positivos; fine-tune-epochs no puede ser negativo.")
    if args.fine_tune_layers < 0 or args.max_train < 0 or args.max_eval < 0:
        parser.error("fine-tune-layers, max-train y max-eval no pueden ser negativos.")
    return args


def preparar_ejemplo(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
    image = preprocess_input(image)
    return image, tf.cast(label, tf.float32)


def construir_datasets(args: argparse.Namespace):
    print("Cargando cats_vs_dogs (la primera descarga puede tardar)...")
    aplicar_compatibilidad_rutas_windows()
    train_ds, val_ds, test_ds = tfds.load(
        "cats_vs_dogs",
        split=["train[:70%]", "train[70%:85%]", "train[85%:]"],
        as_supervised=True,
        shuffle_files=True,
        read_config=tfds.ReadConfig(shuffle_seed=args.seed),
    )

    train_ds = train_ds.shuffle(4096, seed=args.seed, reshuffle_each_iteration=True)
    if args.max_train:
        train_ds = train_ds.take(args.max_train)
    if args.max_eval:
        val_ds = val_ds.shuffle(4096, seed=args.seed, reshuffle_each_iteration=False).take(args.max_eval)
        test_ds = test_ds.shuffle(4096, seed=args.seed + 1, reshuffle_each_iteration=False).take(args.max_eval)

    options = tf.data.Options()
    options.experimental_deterministic = True

    def preparar(dataset, training: bool):
        dataset = dataset.map(preparar_ejemplo, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            dataset = dataset.shuffle(1024, seed=args.seed, reshuffle_each_iteration=True)
        return dataset.with_options(options).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    return preparar(train_ds, True), preparar(val_ds, False), preparar(test_ds, False)


def construir_modelo() -> tuple[keras.Model, keras.Model]:
    backbone = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
    )
    backbone.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="imagen_rgb_preprocesada")
    x = layers.RandomFlip("horizontal", name="aumento_volteo")(inputs)
    x = layers.RandomRotation(0.08, name="aumento_rotacion")(x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="pooling_global")(x)
    x = layers.BatchNormalization(name="normalizacion_cabeza")(x)
    x = layers.Dropout(0.35, name="regularizacion")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="probabilidad_perro")(x)
    model = keras.Model(inputs, outputs, name="clasificador_gatos_perros_resnet50")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model, backbone


def configurar_ajuste_fino(backbone: keras.Model, fine_tune_layers: int) -> int:
    if fine_tune_layers == 0:
        return 0
    backbone.trainable = True
    fine_tune_layers = min(fine_tune_layers, len(backbone.layers))
    first_trainable = len(backbone.layers) - fine_tune_layers
    trainable_count = 0
    for index, layer in enumerate(backbone.layers):
        # Batch Normalization conserva sus estadísticas preentrenadas durante el ajuste fino.
        layer.trainable = index >= first_trainable and not isinstance(layer, layers.BatchNormalization)
        trainable_count += int(layer.trainable)
    return trainable_count


def callbacks_entrenamiento() -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.TerminateOnNaN(),
    ]


def combinar_historiales(histories: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    combined: dict[str, list[float]] = {}
    for history in histories:
        for name, values in history.items():
            combined.setdefault(name, []).extend(float(value) for value in values)
    return combined


def guardar_curvas(history: dict[str, list[float]], head_epochs: int) -> None:
    epochs = np.arange(1, len(history.get("loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#f7f8f3")
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].plot(epochs, history.get("loss", []), marker="o", label="Entrenamiento", color="#315c45")
    axes[0].plot(epochs, history.get("val_loss", []), marker="o", label="Validación", color="#e39b55")
    axes[0].set_title("Pérdida", loc="left", fontweight="bold")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[1].plot(epochs, history.get("accuracy", []), marker="o", label="Entrenamiento", color="#315c45")
    axes[1].plot(epochs, history.get("val_accuracy", []), marker="o", label="Validación", color="#e39b55")
    axes[1].set_title("Exactitud", loc="left", fontweight="bold")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        if 0 < head_epochs < len(epochs):
            ax.axvline(head_epochs + 0.5, color="#758379", linestyle="--", alpha=0.7)
        ax.legend(frameon=False)
    fig.suptitle("Aprendizaje de ResNet50", x=0.06, ha="left", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "entrenamiento.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def guardar_matriz(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=np.int64)
    np.add.at(matrix, (y_true.astype(int), y_pred.astype(int)), 1)
    row_totals = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    row_percent = matrix / row_totals

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    fig.patch.set_facecolor("#f7f8f3")
    ax.set_facecolor("#ffffff")
    ax.imshow(row_percent, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)
    ax.set_xlabel("Predicción", labelpad=10)
    ax.set_ylabel("Clase real", labelpad=10)
    ax.set_title("Matriz de confusión", loc="left", fontweight="bold", pad=16)
    for row in range(2):
        for column in range(2):
            ax.text(
                column, row,
                f"{matrix[row, column]:,}\n{row_percent[row, column]:.1%}",
                ha="center", va="center", fontsize=13,
                color="white" if row_percent[row, column] > 0.55 else "#172820",
                fontweight="bold",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "matriz_confusion.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return matrix


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = construir_datasets(args)
    model, backbone = construir_modelo()
    print("\nFase 1/2: entrenamiento de la cabeza nueva; ResNet50 permanece congelada.")
    head_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks_entrenamiento(),
        verbose=2,
    )
    histories = [head_history.history]
    head_epoch_count = len(head_history.history.get("loss", []))

    fine_tuned_count = 0
    if args.fine_tune_epochs:
        fine_tuned_count = configurar_ajuste_fino(backbone, args.fine_tune_layers)
    if args.fine_tune_epochs and fine_tuned_count:
        print(f"\nFase 2/2: ajuste fino de {fine_tuned_count} capas de ResNet50.")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ],
        )
        fine_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks_entrenamiento(),
            verbose=2,
        )
        histories.append(fine_history.history)

    print("\nEvaluando con el conjunto de prueba...")
    evaluation = model.evaluate(test_ds, return_dict=True, verbose=0)
    y_true = np.concatenate([labels.numpy().astype(int) for _, labels in test_ds])
    dog_probabilities = model.predict(test_ds, verbose=0).reshape(-1)
    y_pred = (dog_probabilities >= 0.5).astype(int)
    confusion = guardar_matriz(y_true, y_pred)
    history = combinar_historiales(histories)
    guardar_curvas(history, head_epoch_count)

    true_negative, false_positive = (int(confusion[0, 0]), int(confusion[0, 1]))
    false_negative, true_positive = (int(confusion[1, 0]), int(confusion[1, 1]))
    total = int(confusion.sum())
    accuracy = (true_positive + true_negative) / max(total, 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-12)

    model.save(MODEL_PATH)
    report = {
        "architecture": "ResNet50",
        "pretrained_weights": "ImageNet",
        "transfer_learning": True,
        "classes": list(CLASS_NAMES),
        "training_date": datetime.now().astimezone().isoformat(timespec="minutes"),
        "test_images": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "auc": float(evaluation.get("auc", 0.0)),
        "test_loss": float(evaluation.get("loss", 0.0)),
        "confusion_matrix": confusion.tolist(),
        "config": {
            "epochs_head_requested": args.epochs,
            "epochs_fine_tune_requested": args.fine_tune_epochs,
            "fine_tune_layers": fine_tuned_count,
            "batch_size": args.batch_size,
            "max_train": args.max_train or "all",
            "max_eval_per_split": args.max_eval or "all",
            "seed": args.seed,
        },
    }
    (REPORTS_DIR / "metricas.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nEntrenamiento finalizado.")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Exactitud de prueba: {accuracy:.1%} | Precisión: {precision:.1%} | Recall: {recall:.1%} | F1: {f1_score:.1%}")
    print(f"Reportes: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
