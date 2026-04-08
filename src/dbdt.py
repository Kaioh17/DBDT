import numpy as np
import torch
import torch.nn as nn 
from .sdt import SDT
"""
PSEUDO CODE - Algorithm 1 DBDT-SGD

Input: Training Set, number of SDTs: T, depth: d, 
       layer number c, λ1, λ2, nEpochs

1: for t = 1,...,T do
2:   Construct SDT ht(x;Θt) with depth d and layer 
     number c, initialize Θt using Xavier init
3: end for

4: for i = 1,...,nEpochs do
5:   Let h0(x)=0, LExp=0, H(x)=0
6:   for t = 1,...,T do
7:     Compute ht(xi;Θt) for all (xi,yi) ∈ T
8:     Compute residuals ri = yi·exp(-yi·H(xi))
9:     Update H(x) = H(x) + ht(xi;Θt)
10:    Compute Lt = Σ[ht(xi) - ri]²
11:    Compute Ct, Ωt
12:    Update LExp = LExp + (Lt + Ct + Ωt)
13:  end for
14:  Update Θ={Θt} w.r.t. LExp using SGD
15: end for

Return: H(x) = Σ ht(x)
"""
class DBDT_SGD:
    def __init__(self, T, input_dim, depth, hidden_dim, lr):
        self.T = T
        self.trees = [SDT(input_dim, depth, hidden_dim) for _ in range(T)] #line 1-3
        all_params = []
        for tree in self.trees:
            all_params += list(tree.parameters())
            
        
        self.optimizer = torch.optim.SGD(all_params, lr=lr)
        self.lamda1 = 0.1 # Ct
        self.lamda2 = 0.005 # Ωt
        
        
    def fit(self, X, y, epochs, batch_size = 128):
        n = X.shape[0]
        for i in range(epochs):
            perm = torch.randperm(n,  device=X.device)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                X_batch = X[idx]
                y_batch = y[idx]
                l_exp = 0; H = torch.zeros(X_batch.shape[0], device=X.device, dtype=X_batch.dtype)
                for tree in self.trees:
                    ht_out, _, node_outputs, node_reach = tree.forward(X_batch)
                    residuals = y_batch * torch.exp(-y_batch * H) #rᵢ = yᵢ · exp(-yᵢ · H(xᵢ))
                    H = H + ht_out # running some accross trees. 
                    Lt = torch.nn.functional.mse_loss(ht_out, residuals, reduction='sum') # local loss (Lt = Σ[ht(xi) - ri]²)
                    l_exp = l_exp + Lt + self._compute_regularizations(tree, node_outputs, node_reach)
                # line 14 - Joint update of all tree params at once
                self.optimizer.zero_grad()
                l_exp.backward()
                self.optimizer.step()
        return H
    def _compute_regularizations(self, tree, node_outputs, node_reach_probs):
        """helper function to get the sum of the main regularizers in the paper (Ct + Ωt)
            Ct: controls regularization by encouraging balanced splits at each inner node
            alph_i [is the wighted average routing the probability at node i]
            
            Ωt:Controls overfitting by penalizing large weights in inner node MLPs.
        """
        # path_probs shape: (batch_size, num_leaves)
        # for each inner node i, sum the path probs of leaves under it
        denom = node_reach_probs.sum(dim=0) + 1e-8
        alpha_i = (node_reach_probs * node_outputs).sum(dim=0) / denom
        Ct = -self.lamda1 * (2 ** -tree.depth) *  (0.5 * torch.log(alpha_i +1e-8) + 
                                                   0.5 * torch.log(1- alpha_i + 1e-8)
                                                   ).sum()
        Omega_t = 0
        for node in tree.inner_nodes:
            for param in node.parameters():
                Omega_t = Omega_t + param.norm(2) ** 2
        Omega_t = self.lamda2 * Omega_t 
        
        return Ct + Omega_t
        
    def predict(self, X):
        with torch.no_grad():
            H = torch.zeros(X.shape[0])
            for tree in self.trees:
                ht_out, _, _, _ = tree.forward(X)
                H = H + ht_out
            return torch.sign(H)
        