"""Fase 2 - Análisis exploratorio de datos (EDA)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURES, METRICS, TARGET

sns.set_theme(style="whitegrid", palette="Set2")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)


def resumen(df: pd.DataFrame) -> pd.DataFrame:
    """Tipos, nulos, valores únicos y estadísticos de cada variable."""
    info = pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "nulos": df.isna().sum(),
        "vacios_texto": df.apply(lambda c: (c.astype(str).str.strip() == "").sum()),
        "unicos": df.nunique(),
    })
    info.to_csv(METRICS / "eda_resumen_variables.csv")
    df.describe(include="all").T.to_csv(METRICS / "eda_estadisticos.csv")
    return info


def plot_target(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[TARGET].value_counts()
    ax.bar(counts.index, counts.values, color=["#66c2a5", "#fc8d62"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_title("Distribución de la variable objetivo (Churn)")
    ax.set_ylabel("Clientes")
    _save(fig, "01_distribucion_churn.png")


def plot_numericas(df, cols):
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    for ax, col in zip(axes, cols):
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, ax=ax, element="step")
        ax.set_title(f"{col} según Churn")
    _save(fig, "02_histogramas_numericas.png")


def plot_categoricas(df, cols):
    n = len(cols)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    for ax, col in zip(axes.flat, cols):
        tasa = (df.groupby(col)[TARGET].apply(lambda s: (s == "Yes").mean())
                .sort_values(ascending=False))
        ax.bar(tasa.index.astype(str), tasa.values, color="#8da0cb")
        ax.set_title(f"Tasa de abandono por {col}")
        ax.set_ylabel("% Churn")
        ax.tick_params(axis="x", rotation=20)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    _save(fig, "03_tasa_churn_categoricas.png")


def plot_correlacion(df_num):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df_num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Matriz de correlación (variables numéricas)")
    _save(fig, "04_matriz_correlacion.png")


def run_eda(df, numeric_cols):
    """Ejecuta el EDA completo y guarda tablas y figuras en results/."""
    info = resumen(df)
    plot_target(df)
    plot_numericas(df, numeric_cols)
    plot_categoricas(df, ["Contract", "PaymentMethod", "InternetService", "TechSupport"])
    num = df[numeric_cols].copy()
    num["Churn_bin"] = (df[TARGET] == "Yes").astype(int)
    plot_correlacion(num)
    return info
