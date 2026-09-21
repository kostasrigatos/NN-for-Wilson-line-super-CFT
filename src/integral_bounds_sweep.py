import cvxpy as cp
import pandas as pd
import numpy as np
from integral_constraints import build_stacked_integral_target, build_stacked_integral_block
from spline_basis import build_smoothness_penalty
from stack_crossing import build_stacked_design_matrix, build_stacked_target_vector
from spectrum_data import available_g_values
from ope_bounds_data import available_g_ope_values
from coupling import G_MIN
from spline_basis import build_knot_vector
from violation_analysis import violation_by_state, ratio_to_truth


def solve_from_cached_with_integrals(A_cross: np.ndarray, b_cross: np.ndarray, A_int: np.ndarray, b_int: np.ndarray, D: np.ndarray, w_int: float, lam: float) -> tuple[np.ndarray, str]:
    r"""
        Solves the multi-objective, non-negative B-spline regression problem using cached matrices.

        Parameters
        ----------
        A_cross: np.ndarray
            A 2D design matrix for the data-driven crossing objectives.

        b_cross: np.ndarray
            A 1D target data vector for the crossing objectives.

        A_int: np.ndarray
            A 2D integral design matrix evaluated above the threshold g >= G_MIN.

        b_int: np.ndarray
            A 1D target data vector for the asymptotic integral physics metrics.

        D: np.ndarray
            A 2D discrete second-difference smoothness penalty matrix of shape (n_basis - 2, n_basis).

        w_int: float
            The regularisation hyperparameter weighing factor for the integral term.

        lam: float
            The regularisation hyperparameter for the smoothness penalty.

        Returns
        -------
        theta: np.ndarray
            A 2D array of shape (10, n_basis) that contains the optimised non-negative B-spline coefficients.

        status: str
            The convergence status of the CVXPY solver.

        Notes
        -----
        This function isolates the optimisation core from the matrix generation. It takes the cached design operators
        and data array in order to formulate and solve the constrained convex problem by minimising the crossing errors,
        integral constraints and smoothness penalty.
    """
    n_basis = D.shape[1]

    theta = cp.Variable((10, n_basis))
    theta_flat = cp.reshape(theta, (10 * n_basis,), order='C')

    crossing_term = cp.sum_squares(A_cross @ theta_flat - b_cross)
    integral_term = cp.sum_squares(A_int @ theta_flat - b_int)
    smoothness_term = cp.sum_squares(D @ theta.T)

    objective = cp.Minimize(crossing_term + w_int * integral_term + lam * smoothness_term)
    problem = cp.Problem(objective, [theta >= 0])
    problem.solve(solver=cp.CLARABEL)

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Solver status: {problem.status}")

    return theta.value, problem.status

def sweep_lam_and_w_against_bounds(g_values: list[float], test_g_values: list[float], knots: np.ndarray, degree: int, lam_values: list[float], w_int_values: list[float], probe_g: float = 2.045) -> pd.DataFrame:
    r"""
        Executes a 2D hyperparameter grid sweep to evaluate bound violations and target the ratios of the states.

        Parameters
        ----------
        g_values: list[float]
            A list of values for the coupling constant used for training.

        test_g_values: list[float]
            A list of values of the coupling constant, g, where the validation bounds are evaluated.

        knots: np.ndarray
            A 1D array that contains the knot sequence including the endpoints.

        degree: int
            The polynomial degree of the B-spline function.

        lam_values: list[float]
            A list of values of the hyperparameter for the smoothness penalty.

        w_int_values: list[float]
            A list of values of the hyperparameter for the integral term.

        probe_g: float
            The specific value of the coupling constant at which to evaluate the scaling ratios for the 3 physical states.
            Defaults to the value 2.045 which is the value of the coupling at which the clearest resolution of the
            near-degenerate states was observed during the earlier exploratory sweeps. It is not an arbitrary choice.

        Returns
        -------
        pd.DataFrame
            A DataFrame that contains the results for every parameter. The DataFrame contains one row per (lam, w_int)
            combination; len(lam_values) * len(w_int_values) in total.
            a) lam: float. The evaluated smoothness penalty weight.
            b) w_int: float. The evaluated integral term weight.
            c) status: str. The convergence status of the CVXPY solver.
            d) aggregate_violation: float. The summed distance of predictions falling outside the valid OPE bounds.
            e) ratio_state1: float. The target reconstruction accuracy ratio for physical state 1 at probe_g.
            f) ratio_state2: float. The target reconstruction accuracy ratio for physical state 2 at probe_g.
            g) ratio_state3: float. The target reconstruction accuracy ratio for physical state 3 at probe_g.

        Notes
        -----
        This function pre-assembles the global regression and physics constraint tensors, and then maps over the cross-product
        of smoothness scaling constants and integral weight factors. For each mode in the grid, it scores violations of
        out-of-sample physical bounds and checks the reconstruction accuracy relative to the midpoint benchmark at a
        specific checkpoint value of the coupling constant. Both metrics are recorded because they can disagree in general.
        Some earlier exploratory sweeps found the aggregate violation staying flat across a wide range of values for w_int,
        while the individual per-state ratios moved from over- to under-estimating the true value across the same range.
        This resulted in crossing the correct answer somewhere in between. Relying only on aggregate_violation would have
        missed that crossing point.
    """
    n_basis = len(knots) - degree - 1
    D = build_smoothness_penalty(n_basis)

    A_cross = build_stacked_design_matrix(g_values, knots, degree)
    b_cross = build_stacked_target_vector(g_values)

    int_g_values = [g for g in g_values if g >= G_MIN]
    A_int = build_stacked_integral_block(int_g_values, knots, degree)
    b_int = build_stacked_integral_target(int_g_values)

    rows = []

    for lam in lam_values:
        for w_int in w_int_values:
            theta, status = solve_from_cached_with_integrals(A_cross, b_cross, A_int, b_int, D, w_int, lam)
            aggregate_violation = violation_by_state(theta, test_g_values, knots, degree)['violation'].sum()
            probe = ratio_to_truth(theta, probe_g, knots, degree)
            ratio_state1 = probe[probe['state'] == 1]['ratio'].values[0]
            ratio_state2 = probe[probe['state'] == 2]['ratio'].values[0]
            ratio_state3 = probe[probe['state'] == 3]['ratio'].values[0]

            rows.append(
                {'lam':lam, 'w_int': w_int, 'status': status, 'aggregate_violation': aggregate_violation,
                 'ratio_state1': ratio_state1, 'ratio_state2': ratio_state2, 'ratio_state3': ratio_state3}
            )

    return pd.DataFrame(rows)

if __name__ == '__main__':
    test_g_values = [g for g in available_g_ope_values if g >= G_MIN]
    g_values = available_g_values[1:]
    degree = 3
    knots = build_knot_vector(g_values[0], g_values[-1], 16, degree)

    result = sweep_lam_and_w_against_bounds(g_values, test_g_values, knots, degree, lam_values=np.logspace(-3.5, -2.5, 5).tolist(),
                                            w_int_values=np.logspace(6, 7, 6).tolist())
    print(result)