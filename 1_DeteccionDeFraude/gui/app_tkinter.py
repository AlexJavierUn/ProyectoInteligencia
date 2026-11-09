"""Aplicación gráfica de escritorio para entrenar y probar el detector."""
from __future__ import annotations

import json
import sys
import threading
import queue
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.preprocessing import TARGET, load_dataset
from src.train import train_model
from src.predictor import predict_transaction
from src.dataset import download_kaggle_dataset


class FraudApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Detección de fraude con tarjetas")
        self.geometry("900x680")
        self.minsize(640, 480)
        self.configure(bg="#f3f6fb")
        self.csv_path = None
        self.df = None
        self.entries = {}
        self.training_queue = queue.Queue()
        self.training_active = False
        self._style()
        self._build()

    def _style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#f3f6fb")
        style.configure("Card.TFrame", background="white")
        style.configure("TLabel", background="#f3f6fb", foreground="#25324b", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 21, "bold"), foreground="#14213d")
        style.configure("Sub.TLabel", font=("Segoe UI", 10), foreground="#65738b")
        style.configure("CardTitle.TLabel", background="white", font=("Segoe UI", 13, "bold"), foreground="#14213d")
        style.configure("TButton", font=("Segoe UI", 10), padding=8)
        style.configure("Accent.TButton", background="#315efb", foreground="white", font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#214bd8")])

    def _build(self):
        outer = ttk.Frame(self, padding=12); outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Detector de transacciones fraudulentas", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Clasificación binaria · Regresión logística o SVM", style="Sub.TLabel").pack(anchor="w", pady=(2, 8))
        cards = ttk.Frame(outer); cards.pack(fill="x")
        self.dataset_label = ttk.Label(cards, text="Dataset\nNo seleccionado", style="CardTitle.TLabel", padding=14)
        self.dataset_label.pack(side="left", fill="x", expand=True, padx=(0, 8), ipady=8)
        self.summary_label = ttk.Label(cards, text="Registros\n—", style="CardTitle.TLabel", padding=14)
        self.summary_label.pack(side="left", fill="x", expand=True, padx=8, ipady=8)
        self.fraud_label = ttk.Label(cards, text="Fraudes\n—", style="CardTitle.TLabel", padding=14)
        self.fraud_label.pack(side="left", fill="x", expand=True, padx=(8, 0), ipady=8)
        action = ttk.Frame(outer); action.pack(fill="x", pady=8)
        ttk.Button(action, text="Seleccionar CSV", command=self.choose_csv).pack(side="left", padx=(0, 4))
        ttk.Button(action, text="Descargar Kaggle", command=self.download_dataset).pack(side="left", padx=4)
        self.model_var = tk.StringVar(value="Regresión logística")
        ttk.Label(action, text="Algoritmo:").pack(side="left", padx=(8, 4))
        ttk.Combobox(action, textvariable=self.model_var, values=["Regresión logística", "SVM"], state="readonly", width=18).pack(side="left")
        self.train_btn = ttk.Button(action, text="Entrenar y evaluar", style="Accent.TButton", command=self.train)
        self.train_btn.pack(side="left", padx=12)
        self.progress = ttk.Progressbar(action, mode="indeterminate", length=130)
        self.status_label = ttk.Label(action, text="")
        self.tabs = ttk.Notebook(outer); self.tabs.pack(fill="both", expand=True, pady=(2, 6))
        left = ttk.Frame(self.tabs, style="Card.TFrame", padding=12)
        self.tabs.add(left, text="Evaluación")
        ttk.Label(left, text="Métricas del modelo", style="CardTitle.TLabel").pack(anchor="w")
        self.metrics = tk.Text(left, height=13, wrap="word", font=("Consolas", 10), bg="#f8faff", relief="flat", padx=12, pady=10)
        self.metrics.pack(fill="both", expand=True, pady=(8, 0)); self.metrics.insert("1.0", "Descarga el dataset Kaggle o selecciona un CSV con la columna Class.\n\n0 = transacción legítima\n1 = transacción fraudulenta\n\nEl flujo divide los datos primero; escalador y SMOTE se ajustan solo con entrenamiento.")
        right = ttk.Frame(self.tabs, style="Card.TFrame", padding=12)
        self.tabs.add(right, text="Probar transacción")
        # La evaluación es la vista principal; el formulario se abre bajo demanda.
        self.tabs.select(left)
        ttk.Label(right, text="Probar transacción", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(right, text="Carga el CSV para generar los campos de entrada.", style="Sub.TLabel").pack(anchor="w", pady=(4, 6))
        self.form = ttk.Frame(right, style="Card.TFrame"); self.form.pack(fill="both", expand=True)
        self.form_canvas = tk.Canvas(self.form, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.form, orient="vertical", command=self.form_canvas.yview)
        self.fields_frame = ttk.Frame(self.form, style="Card.TFrame")
        self.fields_frame.bind("<Configure>", lambda e: self.form_canvas.configure(scrollregion=self.form_canvas.bbox("all")))
        self.form_canvas.create_window((0, 0), window=self.fields_frame, anchor="nw")
        self.form_canvas.configure(yscrollcommand=scrollbar.set)
        self.form_canvas.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
        self.predict_btn = ttk.Button(right, text="Analizar transacción", style="Accent.TButton", command=self.predict, state="disabled")
        self.predict_btn.pack(fill="x", pady=(12, 6))
        self.prediction_label = ttk.Label(right, text="", style="CardTitle.TLabel"); self.prediction_label.pack(anchor="w")
        ttk.Label(outer, text="Uso académico · No usar como decisión financiera real.", style="Sub.TLabel").pack(anchor="w", pady=(2, 0))
        self.after(100, self._poll_training)

    def choose_csv(self):
        path = filedialog.askopenfilename(title="Selecciona dataset CSV", filetypes=[("Archivos CSV", "*.csv")])
        if not path: return
        try:
            self._load_dataset(path)
        except Exception as exc: messagebox.showerror("No se pudo cargar el CSV", str(exc))

    def download_dataset(self):
        try:
            self.config(cursor="watch"); self.update_idletasks()
            path = download_kaggle_dataset()
            self.config(cursor="")
            self._load_dataset(path)
            messagebox.showinfo("Dataset descargado", f"Dataset Kaggle guardado en:\n{path}")
        except Exception as exc:
            self.config(cursor="")
            messagebox.showerror("No se pudo descargar el dataset", str(exc))

    def _load_dataset(self, path):
        df = load_dataset(path)
        self.csv_path, self.df = str(path), df
        self.dataset_label.configure(text=f"Dataset\n{Path(path).name}")
        self.summary_label.configure(text=f"Registros\n{len(df):,}")
        self.fraud_label.configure(text=f"Fraudes\n{int(df[TARGET].sum()):,} ({df[TARGET].mean():.2%})")
        self._make_fields(df)
        self.metrics.delete("1.0", "end"); self.metrics.insert("1.0", f"Dataset cargado: {len(df):,} filas y {len(df.columns)-1} variables.\nFraudes: {int(df[TARGET].sum()):,} de {len(df):,}.\n\nAl entrenar, el dataset se divide antes de aplicar SMOTE, que sobremuestrea únicamente el bloque de entrenamiento.")

    def _make_fields(self, df):
        for widget in self.fields_frame.winfo_children(): widget.destroy()
        self.entries.clear()
        for i, column in enumerate(c for c in df.columns if c != TARGET):
            ttk.Label(self.fields_frame, text=column, style="Sub.TLabel").grid(row=i, column=0, sticky="w", padx=(0, 8), pady=3)
            entry = ttk.Entry(self.fields_frame, width=19)
            sample = pd.to_numeric(df[column], errors="coerce").median()
            entry.insert(0, str(float(sample) if pd.notna(sample) else 0.0))
            entry.grid(row=i, column=1, sticky="ew", pady=3)
            self.entries[column] = entry
        self.fields_frame.columnconfigure(1, weight=1)
        self.predict_btn.configure(state="normal" if (ROOT / "models" / "best_model.pkl").stat().st_size > 0 else "disabled")
        self.tabs.select(0)

    def train(self):
        if self.training_active:
            return
        if not self.csv_path:
            messagebox.showinfo("Dataset requerido", "Selecciona primero un archivo CSV."); return
        self.training_active = True
        self.train_btn.configure(state="disabled")
        self.status_label.configure(text="Entrenando… la interfaz seguirá disponible")
        self.status_label.pack(side="left", padx=8)
        self.progress.pack(side="left", padx=4)
        self.progress.start(12)
        csv_path, model_type = self.csv_path, self.model_var.get()
        threading.Thread(target=self._train_worker, args=(csv_path, model_type), daemon=True).start()

    def _train_worker(self, csv_path, model_type):
        try:
            self.training_queue.put((True, train_model(csv_path, model_type)))
        except Exception as exc:
            self.training_queue.put((False, str(exc)))

    def _poll_training(self):
        try:
            success, result = self.training_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_training)
            return
        self.training_active = False
        self.progress.stop()
        self.progress.pack_forget()
        self.status_label.pack_forget()
        self.train_btn.configure(state="normal")
        if success:
            matrix = result["confusion_matrix"]
            self.metrics.delete("1.0", "end")
            self.metrics.insert("1.0", "Evaluación en conjunto de prueba\n" + "─" * 32 +
                f"\nFilas: {result['rows']:,} · Fraudes: {result['fraud_count']:,} · Prueba: {result['test_size']:,}" +
                f"\n\nPrecisión (fraude): {result['precision']:.3f}\nRecall (fraude): {result['recall']:.3f}\nF1 (fraude): {result['f1']:.3f}\nAverage precision: {result['average_precision']:.3f}" +
                f"\n\nMatriz de confusión\nTN={matrix[0][0]}  FP={matrix[0][1]}\nFN={matrix[1][0]}  TP={matrix[1][1]}" +
                "\n\nSMOTE se aplicó solo a entrenamiento. Modelo y escalador guardados en models/.")
            self.predict_btn.configure(state="normal")
            self.tabs.select(0)
            messagebox.showinfo("Entrenamiento completado", "Modelo entrenado y métricas guardadas en la carpeta models.")
        else:
            messagebox.showerror("Error al entrenar", result)
        self.after(100, self._poll_training)

    def predict(self):
        try:
            result = predict_transaction({key: value.get() for key, value in self.entries.items()})
            if result["prediction"]:
                self.prediction_label.configure(text=f"⚠ Posible fraude · probabilidad {result['probability']:.1%}", foreground="#b42318")
            else:
                self.prediction_label.configure(text=f"✓ Probablemente legítima · riesgo {result['probability']:.1%}", foreground="#067647")
        except Exception as exc: messagebox.showerror("Error de predicción", str(exc))


if __name__ == "__main__":
    FraudApp().mainloop()
