import torch
import torch.nn as nn
from ope_bounds_data import ope_bounds_at_g
from ope_bounds_data import available_g_ope_values
from pinn_model import WilsonNetwork
from pathlib import Path

def pinn_violation_score(model: nn.Module, test_g_values: list[float]) -> float:
    r"""
        Computes the violation score for the neural network.

        Parameters
        ----------
        model: nn.Module
            The trained neural network model.
        test_g_values: list[float]
            The list of out-of-sample values of the coupling constant used to check boundary violation.

        Returns
        -------
        float
            The accumulated violation score across all evaluated states and coupling constant values.
            A returned score of 0.0 suggests that the predictions are in perfect agreement with the known bounds.

        Notes
        -----
        It evaluates the model using a 64-bit float 2D tensor of shape (1, 1) for each coordinate. Implicitly it is
        expected that the model's layers match this precision; calling .double() on the model before passing it in, e.g.
        as it happens in the __main__ block below, is necessary for consistency.
    """
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