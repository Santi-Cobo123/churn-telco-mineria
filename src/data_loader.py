"""Fase 2 - Carga del conjunto de datos."""
import pandas as pd

from src.config import DATA_RAW


def load_raw(path=DATA_RAW) -> pd.DataFrame:
    """Carga el CSV original del dataset Telco Customer Churn."""
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}.\n"
            "Descarga el dataset desde "
            "https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            "y colócalo en data/raw/."
        )
    return pd.read_csv(path)
