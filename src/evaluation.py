## DARSH ##
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

try:
    from hmeasure import h_score
except Exception:  # pragma: no cover
    h_score = None

from .baselines import get_scores


@dataclass
class FoldResult:
    auc: float
    h_measure: float
    f1: float
    precision: float
    recall: float


@dataclass
class CVResult:
    name: str
    folds: List[FoldResult]

    def summary(self) -> Dict[str, float]:
        keys = ["auc", "h_measure", "f1", "precision", "recall"]
        out = {"model": self.name}
        for key in keys:
            vals = np.array([getattr(f, key) for f in self.folds], dtype=float)
            out[f"{key}_mean"] = float(np.nanmean(vals))
            out[f"{key}_std"] = float(np.nanstd(vals))
        return out


class TorchModelAdapter:
    def __init__(self, fit_fn: Callable, score_fn: Callable, pred_fn: Optional[Callable] = None):
        self._fit_fn = fit_fn
        self._score_fn = score_fn
        self._pred_fn = pred_fn

    def fit(self, X, y):
        self._fit_fn(X, y)
        return self

    def decision_function(self, X):
        return self._score_fn(X)

    def predict(self, X):
        if self._pred_fn is not None:
            return self._pred_fn(X)
        scores = self.decision_function(X)
        return (np.asarray(scores) >= 0).astype(int)


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: Optional[float] = None) -> FoldResult:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).reshape(-1)
    # y_true should be 0/1 here
    if threshold is None:
        threshold = 0.5 if ((scores >= 0).all() and (scores <= 1).all()) else 0.0
    y_pred = (scores >= threshold).astype(int)

    auc_val = roc_auc_score(y_true, scores)
    hm = float(h_score(y_true, scores)) if h_score is not None else np.nan
    return FoldResult(
        auc=auc_val,
        h_measure=hm,
        f1=f1_score(y_true, y_pred, zero_division=0),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
    )


def run_stratified_10fold_cv(models: Dict[str, object], X, y, random_state: int = 42) -> Tuple[pd.DataFrame, Dict[str, CVResult]]:
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = np.asarray(y)
    # map {-1,+1} => {1,0} for fraud-positive reporting
    y_bin = np.where(y_arr == -1, 1, 0)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    results: Dict[str, CVResult] = {name: CVResult(name, []) for name in models}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_arr, y_bin), start=1):
        X_tr, X_te = X_arr[tr_idx], X_arr[te_idx]
        y_tr, y_te = y_bin[tr_idx], y_bin[te_idx]

        for name, model in models.items():
            est = clone(model)
            est.fit(X_tr, y_tr)
            scores = get_scores(est, X_te)
            results[name].folds.append(compute_metrics(y_te, scores))

    summary_df = pd.DataFrame([cv.summary() for cv in results.values()]).sort_values("auc_mean", ascending=False)
    return summary_df, results
