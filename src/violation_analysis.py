import pandas as pd
import numpy as np
from convex_baseline import reconstruct_C_squared
from ope_bounds_data import ope_bounds_at_g
from coupling import G_MIN

def violation_by_state(theta: np.ndarray, test_g_values: list[float], knots: np.ndarray, degree: int) -> pd.DataFrame:
    r"""
        Computes the individual metrics for physical boundary violation grouped by the state and the value of the
        coupling constant.

        Parameters
        ----------
        theta: np.ndarray
            A 2D array of shape (10, n_basis) that contains the optimised non-negative B-spline coefficients.

        test_g_values: list[float]
            A list of values of the coupling constant, g, where the validation bounds are evaluated.

        knots: np.ndarray
            A 1D array that contains the knot sequence including the endpoints.

        degree: int
            The polynomial degree of the B-spline function.

        Returns
        -------
        pd.DataFrame
            A DataFrame that contains the metrics for each evaluation point
            a) g: float. The value of the coupling constant.
            b) state: int. The identifier of the physical state.
            c) violation: float. Absolute distance from the bounds.
            d) in_domain: bool. True if the coupling constant satisfies g >= G_MIN

        Notes
        -----
        This function loops through a verification set of values for the coupling constant, evaluates the B-spline
        reconstruction array against analytical lower and upper limits, and logs the absolute magnitude of any out-of-bounds
        error.
    """
    rows = []

    for g in test_g_values:
        fitted_C = reconstruct_C_squared(theta, g, knots, degree)

        for state in [1, 2, 3]:
            lower, upper = ope_bounds_at_g(state, g)
            fitted = fitted_C[state - 1]

            violation = max(0, lower - fitted) + max(0, fitted - upper)
            rows.append(
                {'g':g, 'state':state, 'violation':violation, 'in_domain': g >= G_MIN}
            )

    return pd.DataFrame(rows)

def ratio_to_truth(theta: np.ndarray, g: float, knots: np.ndarray, degree: int) -> pd.DataFrame:
    r"""
        Computes the ratio of the reconstructed curve relative to the midpoint of the physical bounds.

        Parameters
        ----------
        theta: np.ndarray
            A 2D array of shape (10, n_basis) that contains the optimised non-negative B-spline coefficients.

        g: float
            The value of the coupling constant.

        knots: np.ndarray
            A 1D array that contains the knot sequence including the endpoints.

        degree: int
            The polynomial degree of the B-spline function.

        Returns
        -------
        pd.DataFrame
            A DataFrame that contains
            a) g: float. The value of the coupling constant.
            b) state: int. The identifier of the physical state.
            c) ratio: float. The value computed by dividing by the bounding window midpoint.
            d) width: float. The absolute width of the allowed window; upper - lower

        Notes
        -----
        This function establishes a normalised scaling score for a specific value of the coupling constant.
        This is done by treating the exact centre of the lower and upper analytical limits as a benchmark.
        It quantifies, also, the absolute width of the bounding window.
    """
    rows = []

    fitted_C = reconstruct_C_squared(theta, g, knots, degree)

    for state in [1, 2, 3]:
        lower, upper = ope_bounds_at_g(state, g)
        true = (upper + lower) / 2
        ratio = fitted_C[state - 1] / true

        bound_width = upper - lower

        rows.append(
            {'g':g, 'state':state, 'ratio': ratio, 'width': bound_width}
        )

    return pd.DataFrame(rows)