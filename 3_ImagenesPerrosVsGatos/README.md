# Clasificador de perros y gatos con Transfer Learning

Proyecto de Visión Avanzada que clasifica fotografías reales como **gato** o **perro**. Reutiliza **ResNet50 preentrenada con ImageNet** y TensorFlow/Keras; entrena una nueva cabeza de clasificación y, opcionalmente, ajusta las últimas capas de la red.

## Requisitos

- Python compatible con la versión de TensorFlow que instales.
- Conexión a internet durante la primera ejecución para descargar `cats_vs_dogs` y los pesos preentrenados de ResNet50.
- Espacio libre suficiente para los datos, las dependencias y el modelo.

## Preparar el entorno en Windows / PowerShell

Ejecuta estos comandos desde la carpeta del proyecto:

```powershell
py -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r ".\requirements.txt"
```

Si ya existe `.venv`, no hace falta crearla otra vez.

## Entrenar y abrir la aplicación

```powershell
& ".\.venv\Scripts\python.exe" ".\entrenar.py"
& ".\.venv\Scripts\python.exe" -m streamlit run ".\app.py"
```

Para una corrida rápida de demostración, usando menos imágenes y sin ajuste fino:

```powershell
& ".\.venv\Scripts\python.exe" ".\entrenar.py" --epochs 2 --fine-tune-epochs 0 --max-train 1000 --max-eval 300
```

En la aplicación, carga archivos JPG, PNG o WEBP. Se pueden seleccionar varias imágenes y el modelo las procesa en un lote.

## Diseño del experimento

1. **Datos:** `cats_vs_dogs` de TensorFlow Datasets se divide en 70% entrenamiento, 15% validación y 15% prueba.
2. **Preparación:** cada imagen se redimensiona a 224 × 224 y se normaliza con `preprocess_input` de ResNet50.
3. **Transfer Learning:** la ResNet50 con pesos ImageNet comienza congelada. Se entrena una cabeza con pooling global, normalización, dropout y una neurona sigmoid.
4. **Aumento de datos:** volteo horizontal y rotación aleatoria se aplican durante el entrenamiento.
5. **Ajuste fino:** las últimas capas convolucionales se descongelan con una tasa de aprendizaje menor; las capas Batch Normalization conservan sus estadísticas preentrenadas.
6. **Evaluación:** se calcula exactitud, precisión, recall, F1 y AUC sobre el conjunto de prueba.

La etiqueta 0 corresponde a gato y la etiqueta 1 a perro. La salida sigmoid representa la puntuación estimada para perro.

## Archivos generados

- `models/resnet50.keras`: modelo listo para inferencia.
- `reports/entrenamiento.png`: pérdida y exactitud de entrenamiento y validación.
- `reports/matriz_confusion.png`: aciertos y errores por clase, con proporción por fila.
- `reports/metricas.json`: métricas del conjunto de prueba y configuración usada.

Los resultados dependen de las imágenes, el hardware, las épocas y los límites configurados. La puntuación de una foto no sustituye una evaluación con un conjunto de datos independiente.

En Windows, el entrenamiento normaliza los nombres internos del ZIP a `/` antes de preparar el conjunto. Esto evita el `KeyError` de rutas al que puede llegar TFDS durante la creación de su caché.
