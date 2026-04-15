## Efficient Fraud Detection Using Deep Boosting Decision Trees (DBDT)

This repository contains a from-scratch implementation of key components from:

**Biao Xu, Yao Wang, Xiuwu Liao, Kaidong Wang**  
*Efficient Fraud Detection Using Deep Boosting Decision Trees* (2023)

The implementation focuses on fraud detection with highly imbalanced data using:
- Soft Decision Trees (SDT)
- Deep Boosting Decision Trees with SGD-style optimization (DBDT-SGD)
- Imbalance handling through preprocessing and evaluation strategy

---

## Project Goal

Reproduce and study the DBDT method for credit card fraud detection while preserving:
- tree-like interpretability,
- deep representation capacity,
- robust evaluation on imbalanced data.

---

## Dataset

- **Name:** Credit Card Fraud Detection
- **Source:** [Kaggle - mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Access in code:** loaded with `kagglehub` in `src/preprocessing.py`

---

## Repository Structure

- `main.ipynb`  
  Main experimental notebook (data prep, training, threshold tuning, evaluation, plots).

- `src/preprocessing.py`  
  Data loading, train/validation/test splitting, scaling, IQR filtering, SMOTE, plotting, and metrics helpers.

- `src/sdt.py`  
  Soft Decision Tree implementation:
  - inner-node MLP routing,
  - soft/hard forward passes,
  - path probability computation,
  - Xavier initialization.

- `src/dbdt.py`  
  DBDT-SGD implementation:
  - ensemble of SDTs,
  - exponential-loss residual fitting,
  - local + global objective accumulation,
  - regularization terms.

---

## Environment Setup

Python 3.10+ is recommended.

Install required dependencies:

```bash
pip install torch numpy scikit-learn imbalanced-learn matplotlib tqdm kagglehub hmeasure
```

If running on GPU, install a CUDA-compatible PyTorch build matching your system.

---

## How to Run

1. Open `main.ipynb`.
2. Run cells top to bottom.
3. Confirm dataset loads successfully from Kaggle.
4. Train `DBDT_SGD` and evaluate with configured metrics/plots.

---

## Practical Compute Note (Important)

The full dataset is highly imbalanced, and applying SMOTE can greatly increase training size and compute time.

To keep experiments feasible and reproducible, this project supports using a **stratified subset** of the data before training:
- preserve class ratio with stratified sampling,
- apply SMOTE only on training data,
- keep validation/test distribution realistic,
- report sizes before and after SMOTE.

This is an intentional experimental constraint and should be documented in report/presentation methodology.

---

## Current Pipeline Summary

1. Load data and convert labels to `{-1, +1}`
2. Train/test split (stratified)
3. Train/validation split (stratified)
4. Standardize selected numeric features
5. Remove outliers with IQR (training set)
6. Apply SMOTE (training set only)
7. Train SDT/DBDT-SGD
8. Tune decision threshold on validation set
9. Evaluate on test set with metrics and plots

---

## Metrics Used

The notebook reports multiple metrics suitable for imbalanced classification:
- AUC
- F1-score
- Precision
- Recall
- H-measure
- Confusion matrix
- ROC curve

---

## Reproducibility Tips

- Fix random seeds for:
  - sampling,
  - splits,
  - SMOTE,
  - PyTorch initialization.
- Log dataset size at each stage:
  - full/subset,
  - train/val/test,
  - post-IQR,
  - post-SMOTE.
- Record training hyperparameters (`T`, depth, hidden size, learning rate, epochs, batch size).

---

## Acknowledgment

This work is an educational implementation for a course project based on the referenced paper and public dataset.