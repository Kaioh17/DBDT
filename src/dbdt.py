## MUBARAQ ##

import numpy as np
import torch
import torch.nn as nn 
from .sdt import SDT
from tqdm import tqdm
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
    """
    Paper-aligned implementation of DBDT-SGD.
        - Xavier init handled in SDT
        - full-batch epoch updates (Algorithm 1 style)
        - no exp clipping, no grad clipping
    Args:
        T (int): number of trees
        input_dim (int): input dimension
        depth (int): depth of the trees
        hidden_dim (int): hidden dimension
        lr (float): learning rate
        device (torch.device): device to use
        grad_clip_norm (float, optional): gradient clip norm. Defaults to 1.0.
        exp_clip (float, optional): exp clip. Defaults to 20.0.
    """
    def __init__(self, T, input_dim, depth, hidden_dim, lr, device, grad_clip_norm=1.0, exp_clip=20.0):
        """T: number of trees"""
        self.T = T
        self.device = device
        self.exp_clip = exp_clip
        self.grad_clip_norm = grad_clip_norm
        self.trees = [SDT(input_dim, depth, hidden_dim).to(device) for _ in range(T)] #line 1-3
        all_params = []
        for tree in self.trees:
            all_params += list(tree.parameters())
        self._all_params = all_params

        self.optimizer = torch.optim.SGD(all_params, lr=lr)
        self.lamda1 = 0.1 # Ct
        self.lamda2 = 0.005 # Ωt
        
        
    def fit(self, X, y, epochs = 200, batch_size = 128):
        """Built to mirror the paper's Algorithm 1. But with some optimizations. For example: 
        - batching and shuffling the data.
        - computing the regularizations in a more efficient way.

        Args:
            X (torch.Tensor): feature samples
            y (torch.Tensor): predicted samples
            epochs (int, optinal): number iterations. Defaults 200
            batch_size (int, optional): Defaults to 128.

        Returns:
            self
        """
        X = X.to(self.device)
        y = y.to(self.device)
        n = X.shape[0]
        epoch_bar = tqdm(range(epochs), desc="Training DBDT")
        for i in epoch_bar:
            perm = torch.randperm(n,  device=X.device)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                X_batch = X[idx]
                y_batch = y[idx]
                l_exp = 0; H = torch.zeros(X_batch.shape[0], device=X.device, dtype=X_batch.dtype).to(self.device)
                for tree in self.trees:
                    ht_out, _, node_outputs, node_reach = tree.forward_soft(X_batch)
                    # Clamp exponent so exp(-y*H) cannot overflow (common with lr that is too large).
                    exp_arg = torch.clamp(-y_batch * H, -self.exp_clip, self.exp_clip)
                    residuals = y_batch * torch.exp(exp_arg)  # rᵢ = yᵢ · exp(-yᵢ · H(xᵢ))
                    H = H + ht_out # running some accross trees. 
                    Lt = torch.nn.functional.mse_loss(ht_out, residuals, reduction='sum') # local loss (Lt = Σ[ht(xi) - ri]²)
                    l_exp = l_exp + Lt + self._compute_regularizations(tree, node_outputs, node_reach)
                # line 14 - Joint update of all tree params at once
                self.optimizer.zero_grad()
                l_exp.backward()
                if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._all_params, self.grad_clip_norm)
                self.optimizer.step()
                
                epoch_bar.set_postfix({'loss': f'{l_exp.item():.4f}'})
        return self
    def _compute_regularizations(self, tree, node_outputs, node_reach_probs):
        """helper function to get the sum of the main regularizers in the paper (Ct + Ωt)
            Ct: controls regularization by encouraging balanced splits at each inner node
            alpha_i [is the wighted average routing the probability at node i]
            
            Ωt:Controls overfitting by penalizing large weights in inner node MLPs.
        """
         # alpha_i = sum_x pi_i(x)*d_i(x) / sum_x pi_i(x)
        denom = node_reach_probs.sum(dim=0) + 1e-8
        alpha_i = (node_reach_probs * node_outputs).sum(dim=0) / denom
        Ct = -self.lamda1 * (2 ** -tree.depth) *  (0.5 * torch.log(alpha_i +1e-8) + 
                                                   0.5 * torch.log(1- alpha_i + 1e-8)
                                                   ).sum()
                                                
        Omega_t = torch.tensor(0.0, device=node_outputs.device, dtype=node_outputs.dtype)
        # Omega_t = torch.tensor(0.0, device=self.device, dtype=self.device.dtype)
        for node in tree.inner_nodes:
            for param in node.parameters():
                Omega_t = Omega_t + param.norm(2).pow(2)
        Omega_t = self.lamda2 * Omega_t 
        
        return Ct + Omega_t
        

    def predict_score(self, X, hard=False): # will try to optimize for better scores
        """
        Predict the score of the input X.
        Args:
            X (torch.Tensor): The input tensor.
            hard (bool): Whether to use the hard or soft prediction. [use hard for testing]
        Returns:
            torch.Tensor: The predicted score.
        """
        X = X.to(self.device)
        with torch.no_grad():
            H = torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)

            for tree in self.trees:
                if hard:
                    ht_out, _, _, _ = tree.forward_hard(X)
                else:
                    ht_out, _, _, _ = tree.forward_soft(X)
                H = H + ht_out
            
            return H
    def predict(self, X, hard=False): #as implemented in the paper
        """
        Predict the class of the input X.
        Args:
            X (torch.Tensor): The input tensor.
            hard (bool): Whether to use the hard or soft prediction. [use hard for testing]
        Returns:
            torch.Tensor: The predicted class.
        """
        return torch.sign(self.predict_score(X, hard=hard))