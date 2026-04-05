from tensorflow import keras
import numpy as np
import torch
import torch.nn as nn 
"""
3.1.1. Generating Inner Nodes
In SDT, each inner node is designed to be a multilayer perceptron whose input is the
sample features and output the probability of selecting the right sub-path, and a sigmoid
activation function σ(x) = (1 + exp(−x))−1 is used to bring to nonlinearity and enhance
the representation of the SDT. Thus, the input dimension of the inner node network is the
feature dimension and the output dimension is 1. Taking the threshold as 0.5, we then
compare the probability of selecting the right sub-path pi with the threshold. When the
probability of the right sub-path pi > 0.5, the right path is selected, and when pi < 0.5,
the left sub-path is selected. In our paper, we indicate I`
i and Ir
i respectively as whether the
node selects the left path or right path. I`
i and Ir
i satisfy I`
i + Ir
i = 1, and can be defined as"""

class SDT(nn.Module):
    def __init__(self, input_dim, depth, hidden_dim, num_classes=2):
        super(SDT, self).__init__()
        
        self.depth = depth
        self.num_inner_nodes = 2 ** depth -1
        self.num_leaves = 2 ** depth
        self.num_classes = num_classes
        
        self.inner_nodes = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                        nn.ReLU(),
                                                        nn.Linear(hidden_dim, 1), 
                                                        nn.Sigmoid()
                                                        ) for _ in range(self.num_inner_nodes)])
        # since the leaf nodes are just learned parameters we write does using `nn.parameters`
        self.leaf_nodes = nn.Parameter(
            torch.randn(self.num_leaves, 1)
            )
    def forward(self, x):
        batch_size = x.size(0)
        
        # all innser node aoutputs
        node_probs = [node(x) for node in self.inner_nodes]
        
        #computig path probability for each leaf
        
        path_probs = torch.ones(batch_size, self.num_leaves)
        
        for leaf_idx in range(self.num_leaves):
            node_idx = 0 # so we start root
            cur_leaf = leaf_idx
            
            for depth in range(self.depth):
                p = node_probs[node_idx].squeeze(1) 

                if cur_leaf % 2 == 1: # right child
                    path_probs[:, leaf_idx] *= p
                else:  # left child
                    path_probs[:, leaf_idx] *= (1-p)
                
                #to move down the tree
                cur_leaf = cur_leaf // 2
                node_idx = 2 * node_idx + 1 # to descend
            
        # weighted sum over leaves
        h = torch.matmul(path_probs, self.leaf_nodes)
        
        return h.squeeze(1)
                    
    def predict(self, x):
        with torch.no_grad():  # no gradients needed for inference
            h = self.forward(x)
            return torch.sign(h)  # maps to {-1, +1}