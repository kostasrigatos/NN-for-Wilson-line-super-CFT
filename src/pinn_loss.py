import torch
from design_matrix import build_design_matrix, build_target_vector

def build_physics_cache(g_values):
    design_matrix_cache = {}
    target_vector_cache = {}

    for g in g_values:
        design_matrix_cache[g] = torch.tensor(build_design_matrix(g), dtype=torch.float64)
        target_vector_cache[g] = torch.tensor(build_target_vector(g), dtype=torch.float64)

    return design_matrix_cache, target_vector_cache

def crossing_loss(model, g_values, design_matrix_cache, target_vector_cache):
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

def smoothness_penalty(model, g_values):
    g_tensor = torch.tensor(g_values, dtype=torch.float64).unsqueeze(1).requires_grad_(True)
    c_pred = model(g_tensor)
    dc_dg = torch.autograd.grad(c_pred, g_tensor, grad_outputs=torch.ones_like(c_pred), create_graph=True)[0]
    dc2_dg2 = torch.autograd.grad(dc_dg, g_tensor, grad_outputs=torch.ones_like(dc_dg), create_graph=True)[0]

    penalty = torch.sum(dc2_dg2 ** 2)

    return penalty