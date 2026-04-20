# DARSH
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


# -------------------------
# SIMPLE OPTIMIZER
# -------------------------
class PDSCAOptimizer:
    def __init__(self, model_score_fn, params, eta1=1e-3, device=None):
        self.model_score_fn = model_score_fn
        self.params = params
        self.eta1 = eta1
        self.device = device if device else torch.device("cpu")

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self, X, y):
        self.zero_grad()

        scores = self.model_score_fn(X)

        # simple loss (stable)
        loss = torch.mean((scores - y) ** 2)

        loss.backward()

        for p in self.params:
            if p.grad is not None:
                p.data -= self.eta1 * p.grad

        return {"loss": float(loss.detach().cpu())}


# -------------------------
# TRAINER
# -------------------------
class DBDTComTrainer:
    def __init__(self, model, eta1=1e-3, batch_size=256, device=None):
        self.model = model
        self.batch_size = batch_size
        self.device = device if device else torch.device("cpu")

 
        params = []
        for tree in self.model.trees:
            params += list(tree.parameters())

        self.optimizer = PDSCAOptimizer(
            model_score_fn=self.score_samples,
            params=params,
            eta1=eta1,
            device=self.device,
        )

 
    def score_samples(self, X):
        H = torch.zeros(X.shape[0], device=self.device)

        for tree in self.model.trees:
            ht, _, _, _ = tree.forward(X)
            H += ht

        return H

    def fit(self, X, y, epochs=5):
        X = X.to(self.device)
        y = y.to(self.device)

        n = X.shape[0]
        epoch_bar = tqdm(range(epochs), desc="Training DBDT")

        for epoch in epoch_bar:
            for i in range(0, n, self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]

                stats = self.optimizer.step(X_batch, y_batch)

            # print(f"[PDSCA] epoch={epoch+1} loss={stats['loss']:.4f}")
        epoch_bar.set_postfix({"loss": f"{stats['loss']:.4f}"})
        
        return self

    def predict_scores(self, X):
        
        with torch.no_grad():
            return self.score_samples(X.to(self.device)).cpu().numpy()

    def predict(self, X):
        scores = self.predict_scores(X)
        return np.where(scores >= 0, 1, -1)