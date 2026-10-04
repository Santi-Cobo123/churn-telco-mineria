"""Fase 3 - Limpieza, transformación y generación de variables."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ID_COL, RANDOM_STATE, SERVICE_COLS, TARGET, TEST_SIZE


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige tipos, maneja vacíos y elimina duplicados."""
    df = df.copy()
    # TotalCharges viene como texto y tiene espacios en blanco en clientes nuevos (tenure = 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(),
                                       errors="coerce")
    # Un cliente con 0 meses aún no ha pagado nada: imputación lógica con 0
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    # SeniorCitizen viene como 0/1; se pasa a Yes/No para tratarla como categórica
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"])
    df = df.drop_duplicates(subset=ID_COL)
    return df


def generar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Ingeniería de características."""
    df = df.copy()
    servicios_activos = df[SERVICE_COLS].apply(
        lambda c: c.isin(["Yes", "DSL", "Fiber optic"])).sum(axis=1)
    df["NumServicios"] = servicios_activos
    df["CargoPromedioMensual"] = np.where(df["tenure"] > 0,
                                          df["TotalCharges"] / df["tenure"].replace(0, np.nan),
                                          df["MonthlyCharges"])
    df["CargoPromedioMensual"] = df["CargoPromedioMensual"].fillna(df["MonthlyCharges"])
    df["GrupoAntiguedad"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                                   labels=["0-12m", "13-24m", "25-48m", "49-72m"]).astype(str)
    df["ContratoLargo"] = (df["Contract"] != "Month-to-month").map({True: "Yes", False: "No"})
    df["PagoAutomatico"] = df["PaymentMethod"].str.contains("automatic").map(
        {True: "Yes", False: "No"})
    return df


def separar_xy(df: pd.DataFrame):
    X = df.drop(columns=[ID_COL, TARGET])
    y = (df[TARGET] == "Yes").astype(int)
    return X, y


def construir_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    """Imputación + escalado para numéricas, imputación + one-hot para categóricas."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_pipe = Pipeline([("imputar", SimpleImputer(strategy="median")),
                         ("escalar", StandardScaler())])
    cat_pipe = Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                                  sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def dividir(X, y):
    """Partición estratificada entrenamiento/prueba."""
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
