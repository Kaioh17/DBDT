from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


@dataclass
class BaselineSpec:
    name: str
    estimator: object


def get_baseline_models(random_state: int = 42) -> Dict[str, object]:
    models: Dict[str, object] = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
        ]),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                learning_rate_init=1e-3,
                max_iter=100,
                early_stopping=True,
                random_state=random_state,
            )),
        ]),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=random_state,
            n_jobs=-1,
        )
    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    return models


def get_scores(estimator: object, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return np.asarray(proba).reshape(-1)
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X)).reshape(-1)
    preds = estimator.predict(X)
    return np.asarray(preds).reshape(-1)
