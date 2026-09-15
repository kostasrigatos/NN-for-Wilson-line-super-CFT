import torch
from ope_bounds_data import ope_bounds_at_g
from ope_bounds_data import available_g_ope_values
from pinn_model import WilsonNetwork
from pathlib import Path

def pinn_violation_score(model, test_g_values):

    total = 0.0

    for g in test_g_values:
        g_tensor = torch.tensor([[g]], dtype=torch.float64)
        fitted_C = model(g_tensor)[0]

        for state in [1, 2 ,3]:
            lower, upper = ope_bounds_at_g(state, g)
            fitted = fitted_C[state - 1]

            violation = max(0, lower - fitted) + max(0, fitted - upper)
            total += violation

    return total

if __name__ == "__main__":
    test_g_values = available_g_ope_values[::10]
    for lam in [0, 0.001, 0.01, 0.1]:
        model = WilsonNetwork().double()
        model.load_state_dict(torch.load(Path(__file__).parent.parent / "models" / f"trained_model_lam_{lam}.pt"))
        model.eval()
        score = pinn_violation_score(model, test_g_values)
        print(f"For lam={lam}, the violation score is equal to {score}")