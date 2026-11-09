# Proyecto final — guía para ejecutar en Windows

Este repositorio contiene cuatro ejercicios independientes. Puedes instalar todas sus dependencias en un solo entorno virtual desde la carpeta principal. Se recomienda Python 3.11 y conexión a internet durante la instalación.

## 1. Crear un entorno virtual compartido

Abre PowerShell en esta carpeta (la que contiene `1_DeteccionDeFraude`, `2_ClasificadorDePrendasDeVestir`, `3_ImagenesPerrosVsGatos` y `4_Analisis_Sentimientos`). Comprueba Python:

```powershell
py -3.11 --version
```

Crea el entorno en la carpeta principal e instala los requisitos de los cuatro ejercicios en una sola operación:

```powershell
py -3.11 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r ".\requirements-all.txt"
& ".\.venv\Scripts\python.exe" -m pip check
```

`requirements-all.txt` reúne los cuatro archivos de requisitos. Probé la resolución conjunta de dependencias con Python 3.11; `pip check` permite confirmar que la instalación de tu computadora quedó consistente.

Si ya existe una `.venv` en la carpeta principal, no la recrees: ejecuta desde la actualización de `pip` en adelante. Si esa `.venv` ya tenía paquetes mezclados o `pip check` informa conflictos, renómbrala o elimínala y créala nuevamente con los comandos anteriores.

## 2. Ejecutar un ejercicio

Usa los comandos desde la carpeta principal, siempre con el Python de la `.venv` compartida.

### Ejercicio 1 — Detección de fraude

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run ".\1_DeteccionDeFraude\gui\app.py"
```

El CSV y los modelos ya están incluidos. Si faltara el CSV, la aplicación intenta descargarlo y podría requerir acceso a Kaggle. No hace falta volver a entrenar para la demostración normal.

### Ejercicio 2 — Clasificador de prendas

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run ".\2_ClasificadorDePrendasDeVestir\app.py"
```

Los modelos MLP y CNN y sus reportes están incluidos. Para volver a entrenar (descarga Fashion-MNIST si aún no está en caché):

```powershell
& ".\.venv\Scripts\python.exe" ".\2_ClasificadorDePrendasDeVestir\entrenar.py"
```

### Ejercicio 3 — Clasificador de perros y gatos

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run ".\3_ImagenesPerrosVsGatos\app.py"
```

El modelo ResNet50 y sus reportes están incluidos. Para volver a entrenar, se necesita internet para descargar datos y/o pesos de ResNet50; el entrenamiento completo puede tardar. Para una demostración rápida:

```powershell
& ".\.venv\Scripts\python.exe" ".\3_ImagenesPerrosVsGatos\entrenar.py" --epochs 2 --fine-tune-epochs 0 --max-train 1000 --max-eval 300
```

### Ejercicio 4 — Análisis de sentimientos

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run ".\4_Analisis_Sentimientos\app.py"
```

El modelo y los reportes están incluidos. Para volver a entrenar, hace falta internet para descargar reseñas y posiblemente recursos de NLTK:

```powershell
& ".\.venv\Scripts\python.exe" ".\4_Analisis_Sentimientos\entrenar.py"
```

Streamlit muestra en la terminal la dirección local de la app (normalmente `http://localhost:8501`). Para detenerla, pulsa `Ctrl+C`.

## 3. Compartir el proyecto en `.rar`

El entorno virtual no se debe incluir en el `.rar`: cada computadora necesita el suyo y puede crearse con los pasos anteriores. En WinRAR, excluye todas las carpetas llamadas `.venv` (en este proyecto hay entornos locales dentro de algunas carpetas de ejercicios). También conserva las carpetas `models`, `reports` y los datos del ejercicio 1 (`1_DeteccionDeFraude/data/raw/creditcard.csv`). Incluye `README.md` y `requirements-all.txt` para que tu compañero tenga esta guía y el instalador conjunto.

No borres tus entornos locales si todavía los necesitas; basta con excluirlos al crear el archivo comprimido. Si prefieres, puedes usar el entorno compartido de la carpeta principal para lanzar las cuatro apps.

## Si clonas una copia de código desde GitHub

El repositorio de código puede omitir `models`, `reports`, los entornos virtuales y los datasets grandes. En ese caso, antes de usar las predicciones, genera los modelos desde la carpeta principal:

```powershell
& ".\.venv\Scripts\python.exe" ".\2_ClasificadorDePrendasDeVestir\entrenar.py"
& ".\.venv\Scripts\python.exe" ".\3_ImagenesPerrosVsGatos\entrenar.py"
& ".\.venv\Scripts\python.exe" ".\4_Analisis_Sentimientos\entrenar.py"
```

Estos entrenamientos descargan datos y pueden tardar; se requiere internet. Para el ejercicio 1, la aplicación intenta obtener el CSV si falta y ofrece el entrenamiento desde su interfaz.
