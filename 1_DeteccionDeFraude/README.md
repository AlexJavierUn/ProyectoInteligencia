# FraudGuard AI

Dashboard académico de detección de fraude con tarjetas, desarrollado con Streamlit y conectado con el dataset y los artefactos de Machine Learning del proyecto.

## Ejecutar en Windows

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run gui\app.py
```

Si aún no tienes el entorno virtual, créalo primero con `python -m venv .venv`. La aplicación abre en el navegador local.

## Secciones

- **Dashboard:** conteos reales, porcentajes y distribuciones de clase, monto y tiempo.
- **Evaluación:** métricas guardadas, matriz de confusión y curvas Precision–Recall/ROC calculadas sobre la partición de prueba original.
- **Probar transacción:** prueba manual o selección de una fila real del dataset; la etiqueta real se muestra separada de la predicción.
- **Análisis masivo:** CSV de transacciones, predicciones, tabla y descarga de resultados.
- **Información:** método, datos y métricas del proyecto.

Tkinter se conserva como respaldo en `gui/app_tkinter.py`.

## Datos, modelos y SMOTE

Dataset: Kaggle `mlg-ulb/creditcardfraud`, esperado en `data/raw/creditcard.csv`. El botón del dashboard descarga el CSV mediante KaggleHub si hace falta.

La aplicación carga los artefactos existentes de `models/` sin entrenar al abrir. El entrenamiento original divide primero, ajusta StandardScaler solo con entrenamiento y aplica SMOTE únicamente a ese bloque. La evaluación usa prueba sin sobremuestrear. El conjunto actual contiene artefactos de SVM lineal calibrado; la aplicación no muestra métricas inventadas de un modelo que aún no tenga evaluación guardada.

Las variables V1–V28 son componentes transformados por PCA, por lo que el módulo individual ofrece selección de transacciones reales para facilitar demostraciones. El proyecto es académico y no debe usarse para decisiones financieras reales.
