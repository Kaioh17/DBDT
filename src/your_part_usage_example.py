"""Minimal usage example for Darsh's part."""

from __future__ import annotations

import numpy as np
import torch

from src.baselines import get_baseline_models
from src.dbdt import DBDT_SGD
from src.evaluation import compute_metrics, run_stratified_10fold_cv
from src.pdsca import DBDTComTrainer
from src.preprocessing import (
    load_data,
    test_train_split,
    train_valid_split,
    interquatile_range,
    apply_smote,
    torch_cast,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    X, y = load_data()
    X_train, X_test, y_train, y_test = test_train_split(X, y)
    X_train, X_valid, y_train, y_valid = train_valid_split(X_train, y_train)
    X_train, y_train = interquatile_range(X_train, y_train)
    X_train, y_train = apply_smote(X_train, y_train)

    # baselines 
    baseline_models = get_baseline_models(random_state=42)
    baseline_summary, _ = run_stratified_10fold_cv(baseline_models, X_train, y_train)
    print("\nBaseline CV summary:\n", baseline_summary)

    # DBDT-Com
    Xtr_t, Xte_t, ytr_t, yte_t = torch_cast(X_train, X_test, y_train, y_test, device)

    model = DBDT_SGD(
        T=3,
        input_dim=Xtr_t.shape[1],
        depth=3,
        hidden_dim=32,
        lr=1e-3,
        device=device,
    )

    trainer = DBDTComTrainer(
        model=model,
        eta1=1e-3,
        batch_size=256,
        device=device,
    )

    trainer.fit(Xtr_t, ytr_t, epochs=20)

   
    # Model learns fraud=-1, legit=+1.
    # So raw scores are LOWER for fraud.
    raw_scores = trainer.predict_scores(Xte_t)

    # Flip sign so HIGHER score means fraud/positive class
    fraud_scores = -raw_scores

    # Convert to probabilities for H-measure / metrics
    test_scores = 1 / (1 + np.exp(-fraud_scores))

    y_test_bin = np.where(yte_t.detach().cpu().numpy() == -1, 1, 0)

    print("\nDBDT-Com test metrics:\n", compute_metrics(y_test_bin, test_scores))


if __name__ == "__main__":
    main()