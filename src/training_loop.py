import torch
from pinn_loss import build_physics_cache, crossing_loss, smoothness_penalty
from pinn_model import WilsonNetwork
from spectrum_data import available_g_values
from pathlib import Path

def train_and_save(lam, g_values, design_matrix_cache, target_vector_cache, n_epochs = 30000):

    torch.manual_seed(42)
    model = WilsonNetwork().double()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        if lam > 0:
            loss = crossing_loss(model, g_values, design_matrix_cache, target_vector_cache) + lam * smoothness_penalty(
                model, g_values)
        else:
            loss = crossing_loss(model, g_values, design_matrix_cache, target_vector_cache)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"For epoch {epoch} the loss is {loss.item()}")

    save_path = Path(__file__).parent.parent / "models" / f"trained_model_lam_{lam}.pt"
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    g_values = available_g_values[1:]
    design_matrix_cache, target_vector_cache = build_physics_cache(g_values)

    for lam in [0, 0.001, 0.01, 0.1]:
        train_and_save(lam, g_values, design_matrix_cache, target_vector_cache)