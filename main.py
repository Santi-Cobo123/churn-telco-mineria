"""Ejecuta el proceso completo de minería de datos de principio a fin.

Uso:
    python main.py            # k elegido automáticamente por silueta
    python main.py --k 4      # fija el número de segmentos
"""
import argparse

from sklearn.base import clone

from src.clustering import elegir_k, segmentar
from src.config import DATA_PROCESSED, NUMERIC_COLS
from src.data_loader import load_raw
from src.eda import run_eda
from src.evaluation import evaluar_prueba, importancia_variables, validacion_cruzada
from src.models import HAS_SMOTE, HAS_XGB, construir_pipeline, definir_modelos
from src.preprocessing import (construir_preprocesador, dividir, generar_variables,
                               limpiar, separar_xy)


def main(k=None):
    print("\n[1/5] Cargando datos...")
    df_raw = load_raw()
    print(f"      {df_raw.shape[0]} filas x {df_raw.shape[1]} columnas")

    print("[2/5] Limpieza y análisis exploratorio...")
    df = limpiar(df_raw)
    run_eda(df, NUMERIC_COLS)

    print("[3/5] Generación de variables...")
    df = generar_variables(df)
    df.to_csv(DATA_PROCESSED, index=False)

    print("[4/5] Modelado")
    print("      a) Segmentación K-Means")
    tabla_k = elegir_k(df)
    k = k or int(tabla_k.loc[tabla_k["silueta"].idxmax(), "k"])
    _, perfil = segmentar(df, k)
    print(f"         k = {k}\n{perfil.to_string()}\n")

    print("      b) Modelos predictivos")
    print(f"         SMOTE: {'sí' if HAS_SMOTE else 'no (se usa class_weight)'} | "
          f"XGBoost: {'sí' if HAS_XGB else 'no (se usa HistGradientBoosting)'}")
    X, y = separar_xy(df)
    X_train, X_test, y_train, y_test = dividir(X, y)
    prep = construir_preprocesador(X_train)
    pipelines = {n: construir_pipeline(clone(prep), m)
                 for n, m in definir_modelos(y_train).items()}

    print("[5/5] Evaluación")
    cv = validacion_cruzada(pipelines, X_train, y_train)
    print("      Validación cruzada (entrenamiento):")
    print(cv[["modelo", "f1_media", "recall_media", "roc_auc_media"]].to_string(index=False))

    ajustados = {n: p.fit(X_train, y_train) for n, p in pipelines.items()}
    test = evaluar_prueba(ajustados, X_test, y_test)
    print("\n      Conjunto de prueba:")
    print(test.to_string(index=False))

    mejor = test.iloc[0]["modelo"]
    imp = importancia_variables(ajustados[mejor], X_test, y_test, mejor)
    print(f"\n      Variables más influyentes ({mejor}):")
    print(imp.head(8)[["variable", "importancia"]].to_string(index=False))
    print("\nResultados guardados en results/figures y results/metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None, help="número de segmentos K-Means")
    main(parser.parse_args().k)
