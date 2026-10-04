# Predicción de abandono de clientes (Churn) – Minería de Datos

Práctica Experimental 1 de la asignatura **Minería de Datos**: aplicación del proceso completo de data mining (definición del problema, datos, preprocesamiento, modelado y evaluación) sobre un caso de estudio de telecomunicaciones.

## Problema

Una empresa de telecomunicaciones pierde ingresos por clientes que cancelan su servicio. El objetivo es **identificar a los clientes con alto riesgo de abandono** para dirigir campañas de retención, segmentar a los clientes según su comportamiento y determinar qué factores influyen más en el abandono.

## Dataset

[Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7.043 clientes, 21 variables (demográficas, servicios contratados, contrato, facturación y variable objetivo `Churn`).

> El CSV no se incluye en el repositorio. Descárgalo y guárdalo como
> `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Estructura del repositorio

```
churn-telco-mineria/
├── data/
│   ├── raw/                 # dataset original (no versionado)
│   └── processed/           # dataset limpio con variables nuevas
├── notebooks/
│   ├── 01_definicion_problema.ipynb
│   ├── 02_datos_eda.ipynb
│   ├── 03_preprocesamiento.ipynb
│   ├── 04_modelado.ipynb
│   └── 05_evaluacion.ipynb
├── src/
│   ├── config.py            # rutas, semilla y parámetros globales
│   ├── data_loader.py       # carga de datos
│   ├── eda.py               # análisis exploratorio y gráficos
│   ├── preprocessing.py     # limpieza, ingeniería de variables, pipeline de transformación
│   ├── clustering.py        # segmentación K-Means
│   ├── models.py            # definición de modelos predictivos
│   └── evaluation.py        # validación cruzada, métricas, ROC, importancia de variables
├── results/
│   ├── figures/             # gráficos generados
│   └── metrics/             # tablas de métricas (CSV)
├── main.py                  # ejecuta todo el proceso de principio a fin
├── requirements.txt
└── README.md
```

## Instalación y ejecución

```bash
git clone https://github.com/<tu-usuario>/churn-telco-mineria.git
cd churn-telco-mineria
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Ejecutar el proceso completo:

```bash
python main.py          # k de K-Means elegido por coeficiente de silueta
python main.py --k 4    # fijar el número de segmentos
```

O bien recorrer las fases paso a paso con los notebooks (`jupyter notebook notebooks/`).

## Metodología

| Fase | Técnicas |
|---|---|
| 1. Problema | Objetivo de retención, objetivos específicos, alcance y criterio de éxito |
| 2. Datos | Estadísticos descriptivos, distribución de la variable objetivo, histogramas, tasas de abandono por categoría, correlaciones |
| 3. Preprocesamiento | Conversión de `TotalCharges`, imputación, one-hot encoding, escalado, variables nuevas (`NumServicios`, `CargoPromedioMensual`, `GrupoAntiguedad`, `ContratoLargo`, `PagoAutomatico`), balanceo con SMOTE |
| 4. Modelado | **Descriptivo:** K-Means (codo + silueta, proyección PCA). **Predictivo:** Regresión Logística, Árbol de Decisión, Random Forest, XGBoost |
| 5. Evaluación | Partición 80/20 estratificada, validación cruzada de 5 pliegues, accuracy, precisión, recall, F1, AUC-ROC, matriz de confusión, importancia por permutación |

Todo el preprocesamiento y el balanceo se encapsulan en un `Pipeline`, de modo que se ajustan solo con los datos de entrenamiento de cada pliegue y se evita la fuga de información.

**Métrica principal: recall de la clase Churn**, porque no detectar a un cliente que se va a ir cuesta más que contactar a uno que no lo haría.

> `xgboost` e `imbalanced-learn` son opcionales: si no están instalados, el código usa
> `HistGradientBoostingClassifier` y ponderación de clases (`class_weight="balanced"`).

## Resultados

Tras ejecutar `main.py`, los resultados quedan en `results/`:

| Archivo | Contenido |
|---|---|
| `metrics/validacion_cruzada.csv` | Media y desviación de cada métrica por modelo (5 pliegues) |
| `metrics/metricas_prueba.csv` | Métricas finales en el conjunto de prueba |
| `metrics/kmeans_perfil_segmentos.csv` | Perfil y tasa de abandono de cada segmento |
| `metrics/importancia_variables.csv` | Importancia por permutación del mejor modelo |
| `figures/01`–`09_*.png` | Gráficos de EDA, clustering y evaluación |

<!-- Completar con los resultados reales, por ejemplo:
| Modelo | Accuracy | Precisión | Recall | F1 | AUC |
|---|---|---|---|---|---|
| ... | | | | | |
-->

## Autores

- Nombre Apellido – correo@ejemplo.com

## Licencia

MIT. Ver [LICENSE](LICENSE).
