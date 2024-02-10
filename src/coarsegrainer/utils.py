
# function to produce a linear backbone with a given number of nodes
import numpy as np
import torch

def line_graph_A(n, k=1):
    A = np.eye(n,k=k) + np.eye(n,k=-k)
    return torch.tensor(A, dtype=torch.float32)


