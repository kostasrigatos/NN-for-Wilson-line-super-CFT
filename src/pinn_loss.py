import torch
import torch.nn as nn
from design_matrix import build_design_matrix, build_target_vector

def build_physics_cache(g_values: list[float]) -> tuple[dict[float, torch.Tensor], dict[float, torch.Tensor]]:
    r"""
         Pre-computes and caches PyTorch tensor representations of the design matrices and target vectors.

         Parameters
         ----------
         g_values: list[float]
            A list of values of the coupling constant, g, where the matrices of the system are evaluated.

         Returns
         -------
         design_matrix_cache: dict[float, torch.Tensor]
             A dictionary mapping each value of the coupling constant, g, to its 2D design matrix.
         target_vector_cache: dict[float, torch.Tensor]
             A dictionary mapping each value of the coupling constant, g, to its 1D target vector.
     """
    design_matrix_cache = {}
    target_vector_cache = {}

    for g in g_values:
        design_matrix_cache[g] = torch.tensor(build_design_matrix(g), dtype=torch.float64)
        target_vector_cache[g] = torch.tensor(build_target_vector(g), dtype=torch.float64)

    return design_matrix_cache, target_vector_cache

def crossing_loss(model: nn.Module, g_values: list[float], design_matrix_cache: dict[float, torch.Tensor], target_vector_cache: dict[float, torch.Tensor]) -> torch.Tensor:
    r"""
         Calculates the total residual sum of squares loss across the cached physics targets.

         Parameters
         ----------
         model: nn.Module
            The neural network model.
         g_values: list[float]
            A list of values of the coupling constant, g, where the matrices of the system are evaluated.
         design_matrix_cache: dict[float, torch.Tensor]
            A dictionary mapping each value of the coupling constant, g, to its 2D design matrix.
         target_vector_cache: dict[float, torch.Tensor]
            A dictionary mapping each value of the coupling constant, g, to its 1D target vector.

         Returns
         -------
         torch.Tensor
             A scalar 64-bit floating-point PyTorch tensor that represents the total crossing residual loss.
     """
    g_tensor = torch.tensor(g_values, dtype=torch.float64).unsqueeze(1)
    predictions = model(g_tensor)

    total_loss = torch.tensor(0.0, dtype=torch.float64)

    for idx, g in enumerate(g_values):
        prediction_g = predictions[idx]
        X_g = design_matrix_cache[g]
        y_g = target_vector_cache[g]
        residual = torch.matmul(X_g, prediction_g) - y_g

        total_loss += torch.sum(residual ** 2)

    return total_loss

def smoothness_penalty(model: nn.Module, g_values: list[float]) -> torch.Tensor:
    r"""
         Computes the continuous second-order derivatives smoothness regularisation using automatic differentiation.

         Parameters
         ----------
         model: nn.Module
            The neural network model.
         g_values: list[float]
            A list of values of the coupling constant, g, where the matrices of the system are evaluated.

         Returns
         -------
         torch.Tensor
             A scalar PyTorch tensor with the accumulated sum of squares of the second-order derivatives.
     """
    g_tensor = torch.tensor(g_values, dtype=torch.float64).unsqueeze(1).requires_grad_(True)
    c_pred = model(g_tensor)
    dc_dg = torch.autograd.grad(c_pred, g_tensor, grad_outputs=torch.ones_like(c_pred), create_graph=True)[0]
    dc2_dg2 = torch.autograd.grad(dc_dg, g_tensor, grad_outputs=torch.ones_like(dc_dg), create_graph=True)[0]

    penalty = torch.sum(dc2_dg2 ** 2)

    return penalty