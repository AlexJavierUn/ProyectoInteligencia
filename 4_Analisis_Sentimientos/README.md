# Análisis de sentimiento en reseñas de productos

Proyecto de Procesamiento de Lenguaje Natural que asigna una de tres clases a una reseña de producto escrita en español: **negativa, neutral o positiva**.

## Relación con el objetivo

- **Entrada:** texto de una reseña de producto.
- **Salida:** una clase de sentimiento y la probabilidad estimada para cada clase.
- **Arquitectura:** capa de embeddings entrenables, red recurrente **BiLSTM**, dropout y capa softmax de tres salidas.
- **Herramientas:** TensorFlow/Keras para vectorización y red neuronal; spaCy para tokenización en español; NLTK para filtrar palabras vacías conservando negaciones; Streamlit para la interfaz; Hugging Face Datasets para obtener las reseñas.
- **Interfaz:** análisis individual y análisis masivo desde CSV.

## Datos y etiquetas

El entrenamiento utiliza la partición española del conjunto **Multilingual Amazon Reviews Corpus** publicado en Hugging Face como `mteb/amazon_reviews_multi`. El corpus incluye reseñas en español separadas en entrenamiento, validación y prueba. Cada reseña tiene una etiqueta de estrellas codificada de 0 a 4; la conversión usada por el proyecto es:

| Etiqueta original | Estrellas | Clase del proyecto |
| ---: | ---: | --- |
| 0–1 | 1–2 | Negativo |
| 2 | 3 | Neutral |
| 3–4 | 4–5 | Positivo |

Se toman 30 000 ejemplos del conjunto de entrenamiento para reducir el tiempo de ejecución. Validación y prueba se mantienen aparte; el conjunto de prueba se usa para calcular el informe de métricas y la matriz de confusión. El corpus está equilibrado por estrellas, pero las tres clases finales no quedan en proporción idéntica (negativo y positivo agrupan dos niveles cada uno).

## Preparación del texto

`utils.py` aplica minúsculas, elimina enlaces y caracteres que no son letras, tokeniza con el modelo lingüístico vacío de spaCy para español y filtra palabras vacías con NLTK. No descarga un modelo lingüístico de spaCy. Conserva negaciones y matices como `no`, `nunca`, `sin`, `pero`, `muy`, `más` y `menos`, porque pueden cambiar el sentido de una reseña. Se usa la misma función en entrenamiento y predicción.

## Instalación y ejecución

Se recomienda Python 3.10 o 3.11 y un entorno virtual.

```bash
python -m venv .venv
```

En Windows:

```powershell
.venv\Scripts\Activate.ps1
```

En macOS o Linux:

```bash
source .venv/bin/activate
```

Instala las dependencias y entrena el modelo. El entrenamiento descarga el corpus y NLTK puede descargar la lista de palabras vacías en español la primera vez.

```bash
python -m pip install -r requirements.txt
python entrenar.py
```

Para iniciar la aplicación:

```bash
streamlit run app.py
```

El primer entrenamiento puede tardar varios minutos y requiere conexión a internet. Para un modelo más preciso, aumenta `N_TRAIN` y `EPOCAS` en `entrenar.py`; esto también aumenta el uso de memoria y el tiempo de entrenamiento.

## Archivos generados

- `models/lstm.keras`: modelo listo para cargar desde Streamlit. Incluye la capa de vectorización y puede recibir texto crudo.
- `reports/accuracy.png`: exactitud de entrenamiento y validación por época.
- `reports/matriz.png`: matriz de confusión sobre el conjunto de prueba.
- `reports/metricas.json`: precisión, recall y F1 por clase, además de promedios macro y ponderado.

## Estructura

```text
app.py          Interfaz web para predicción individual y por CSV
entrenar.py     Descarga de datos, entrenamiento, evaluación y exportación
utils.py        Limpieza compartida de reseñas
requirements.txt
models/         Modelo entrenado
reports/        Gráficos e informe de evaluación
```

## Alcance y limitaciones

El modelo aprende de reseñas de Amazon y de sus estrellas convertidas a tres clases; no equivale a una anotación manual independiente del texto. Una reseña de tres estrellas se toma como neutral, aunque su redacción pueda expresar una opinión positiva o negativa. Las predicciones son estimaciones y pueden fallar con ironía, sarcasmo, faltas ortográficas o expresiones regionales. Los resultados sirven como proyecto académico y no deben presentarse como evaluación perfecta de cualquier tienda o dominio.

El corpus tiene condiciones de uso académico. Revisa la [ficha del conjunto de datos](https://huggingface.co/datasets/mteb/amazon_reviews_multi) y sus condiciones antes de redistribuir los datos o usar el modelo fuera de este proyecto académico.
