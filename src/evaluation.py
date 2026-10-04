"""Fase 5 - Evaluación y validación de modelos."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import CV_FOLDS, FIGURES, METRICS, RANDOM_STATE

SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "f1": "f1", "roc_auc": "roc_auc"}


def validacion_cruzada(pipelines: dict, X, y) -> pd.DataFrame:
    """Validación cruzada estratificada de k pliegues sobre el conjunto de entrenamiento."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    filas = []
    for nombre, pipe in pipelines.items():
        res = cross_validate(clone(pipe), X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        fila = {"modelo": nombre}
        for m in SCORING:
            fila[f"{m}_media"] = res[f"test_{m}"].mean()
            fila[f"{m}_std"] = res[f"test_{m}"].std()
        filas.append(fila)
    tabla = pd.DataFrame(filas).sort_values("f1_media", ascending=False).round(4)
    tabla.to_csv(METRICS / "validacion_cruzada.csv", index=False)
    return tabla


def evaluar_prueba(pipelines_ajustados: dict, X_test, y_test) -> pd.DataFrame:
    """Métricas finales en el conjunto de prueba, matrices de confusión y curvas ROC."""
    filas = []
    n = len(pipelines_ajustados)
    fig_cm, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    reportes = []
    for ax, (nombre, pipe) in zip(np.atleast_1d(axes), pipelines_ajustados.items()):
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        filas.append({
            "modelo": nombre,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
        })
        ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["No", "Sí"],
                                                cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(nombre)
        ax.grid(False)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Valor real")
        RocCurveDisplay.from_predictions(y_test, proba, name=nombre, ax=ax_roc)
        reportes.append(f"=== {nombre} ===\n"
                        f"{classification_report(y_test, pred, target_names=['No churn', 'Churn'])}")
    fig_cm.suptitle("Matrices de confusión (conjunto de prueba)")
    fig_cm.tight_layout()
    fig_cm.savefig(FIGURES / "07_matrices_confusion.png", dpi=150)
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_title("Curvas ROC (conjunto de prueba)")
    ax_roc.set_xlabel("Tasa de falsos positivos")
    ax_roc.set_ylabel("Tasa de verdaderos positivos (recall)")
    fig_roc.tight_layout()
    fig_roc.savefig(FIGURES / "08_curvas_roc.png", dpi=150)
    plt.close("all")

    (METRICS / "reporte_clasificacion.txt").write_text("\n".join(reportes), encoding="utf-8")
    tabla = pd.DataFrame(filas).sort_values("f1", ascending=False).round(4)
    tabla.to_csv(METRICS / "metricas_prueba.csv", index=False)
    return tabla


def importancia_variables(pipe, X_test, y_test, nombre, top=15) -> pd.DataFrame:
    """Importancia por permutación (agnóstica al modelo) medida con el recall."""
    r = permutation_importance(pipe, X_test, y_test, scoring="recall", n_repeats=10,
                               random_state=RANDOM_STATE, n_jobs=-1)
    imp = (pd.DataFrame({"variable": X_test.columns, "importancia": r.importances_mean,
                         "std": r.importances_std})
           .sort_values("importancia", ascending=False))
    imp.to_csv(METRICS / "importancia_variables.csv", index=False)

    datos = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(datos["variable"], datos["importancia"], xerr=datos["std"], color="#66c2a5")
    ax.set_title(f"Importancia por permutación – {nombre}")
    ax.set_xlabel("Caída media del recall")
    fig.tight_layout()
    fig.savefig(FIGURES / "09_importancia_variables.png", dpi=150)
    plt.close(fig)
    return imp
