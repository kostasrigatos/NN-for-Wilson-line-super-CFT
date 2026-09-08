import torch
import numpy as np
from blocks import f_delta

# Step 1: pre-compute the quadrature nodes and weights
quadrature_degree = 32
_nodes_std, _weights_std = np.polynomial.legendre.leggauss(quadrature_degree)

# Step 2: affine map to target the [0, 0.5] interval
_nodes_mapped = (_nodes_std + 1) / 4
_weights_mapped = _weights_std / 4

# Step 3: convert to high-precision tensors
nodes = torch.from_numpy(_nodes_mapped).to(torch.float64)
weights = torch.from_numpy(_weights_mapped).to(torch.float64)

# Step 4: pre-compute the fixed mathematical terms at the locations of the nodes
_term1 = (nodes - 1 - nodes**2) / (nodes**2)
_term2 = (1 - 2 * nodes) / (nodes * (1.0 - nodes))
_term3 = (2*nodes - 1) / (nodes**2)
combined_weights1 = _term1 * _term2 * weights
combined_weights2 = _term3 * weights

def int1(Delta: torch.Tensor) -> torch.Tensor:

    f_evaluated = f_delta(Delta, nodes)
    weighted_sum = torch.sum(-f_evaluated * combined_weights1, dim = -1)

    return weighted_sum

def int2(Delta: torch.Tensor) -> torch.Tensor:

    f_evaluated = f_delta(Delta, nodes)
    weighted_sum = torch.sum(f_evaluated * combined_weights2, dim = -1)

    return weighted_sum
