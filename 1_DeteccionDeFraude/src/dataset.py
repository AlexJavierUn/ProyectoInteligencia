"""Descarga y localización del dataset público de Kaggle."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATASET_ID = "mlg-ulb/creditcardfraud"


def download_kaggle_dataset() -> Path:
    """Descarga vía kagglehub y copia creditcard.csv a data/raw."""
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Falta kagglehub. Instala dependencias con: pip install -r requirements.txt") from exc

    downloaded = Path(kagglehub.dataset_download(DATASET_ID))
    candidates = list(downloaded.rglob("creditcard.csv"))
    if not candidates:
        candidates = list(downloaded.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Kaggle descargó el dataset, pero no se encontró CSV en {downloaded}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    destination = RAW_DIR / "creditcard.csv"
    if candidates[0].resolve() != destination.resolve():
        shutil.copy2(candidates[0], destination)
    return destination
