"""Configuración global del proyecto: rutas, semilla y nombres de columnas."""
from pathlib import Path

# Rutas
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATA_PROCESSED = ROOT / "data" / "processed" / "telco_procesado.csv"
FIGURES = ROOT / "results" / "figures"
METRICS = ROOT / "results" / "metrics"

# Reproducibilidad
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Variables
ID_COL = "customerID"
TARGET = "Churn"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies",
]

for _p in (FIGURES, METRICS, DATA_PROCESSED.parent):
    _p.mkdir(parents=True, exist_ok=True)
