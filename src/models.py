"""Fase 4b - Modelos predictivos.

El desbalance de clases se trata con SMOTE (si imbalanced-learn está instalado)
o, en su defecto, con ponderación de clases (class_weight="balanced").
"""
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


def definir_modelos(y_train=None):
    rs = RANDOM_STATE
    # Con SMOTE las clases quedan balanceadas, así que no se ponderan
    cw = None if HAS_SMOTE else "balanced"
    modelos = {
        "Regresión Logística": LogisticRegression(max_iter=2000, class_weight=cw),
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                                    class_weight=cw, random_state=rs),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10,
                                                min_samples_leaf=5, class_weight=cw,
                                                n_jobs=-1, random_state=rs),
    }
    if HAS_XGB:
        spw = 1.0
        if not HAS_SMOTE and y_train is not None:
            spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        modelos["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                           subsample=0.9, colsample_bytree=0.8,
                                           scale_pos_weight=spw, eval_metric="logloss",
                                           random_state=rs)
    else:
        modelos["Gradient Boosting"] = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=300, class_weight=cw, random_state=rs)
    return modelos


def construir_pipeline(preprocesador, modelo):
    """Une preprocesamiento, balanceo (opcional) y clasificador en un solo pipeline.

    Así el preprocesamiento y SMOTE se ajustan solo con los datos de entrenamiento
    de cada pliegue, evitando fuga de información (data leakage).
    """
    if HAS_SMOTE:
        return ImbPipeline([("prep", preprocesador),
                            ("smote", SMOTE(random_state=RANDOM_STATE)),
                            ("modelo", modelo)])
    return Pipeline([("prep", preprocesador), ("modelo", modelo)])
