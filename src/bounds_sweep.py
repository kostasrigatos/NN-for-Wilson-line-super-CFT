import numpy as np
from convex_baseline import reconstruct_C_squared, solve_convex_problem
from ope_bounds_data import ope_bounds_at_g
from ope_bounds_data import available_g_ope_values
from spline_basis import build_knot_vector, build_smoothness_penalty
from spectrum_data import available_g_values
from cross_validation import build_theta_block_cache, build_target_vector_cache, solve_from_cached

def violation_score(theta: np.ndarray, test_g_values: list[float], knots: np.ndarray, degree: int) -> float:
    r"""
        Calculates the total physical constraint violation score across a set of test parameters of the coupling constant.

        Parameters
        ----------
        theta: np.ndarray
            A 2D array of shape (10, n_basis) that contains the optimal B-spline coefficients.

        test_g_values: list[float]
            The list of the values for the coupling constant used to evaluate the validation bounds.

        knots: np.ndarray
            A 1D array that contains the knot sequence including the endpoints.

        degree: int
            The polynomial degree of the B-spline function.

        Returns
        -------
        float
           The total accumulated violation score across all states and parameters.
           A score of 0.0 corresponds to perfect alignment with no violations.

        Notes
        -----
        This function evaluates how well a fitted set of B-spline coefficients (`theta`) satisfies the rigorous
        upper and lower OPE bounds across multiple test coupling points. Any fitted value falling outside its valid
        analytical window incurs a linear penalty.
    """
    total = 0.0

    for g in test_g_values:
        fitted_C = reconstruct_C_squared(theta, g, knots, degree)

        for state in [1, 2, 3]:
            lower, upper = ope_bounds_at_g(state, g)
            fitted = fitted_C[state - 1]

            violation = max(0, lower - fitted) + max(0, fitted - upper)
            total = total + violation

    return total

def sweep_lam_against_bounds(g_values: list[float], test_g_values: list[float], knots: np.ndarray, degree: int, lam_candidates: list[float]) -> tuple[float, np.ndarray]:
    r"""
        Performs a sweep of candidate hyperparameters over lambda to find the model that violates the fewest physical bounds.

        Parameters
        ----------
        g_values: np.ndarray
            A 1D array of values of the coupling constant, g, where the matrices of the system are evaluated.

        test_g_values: list[float]
            The list of the values for the coupling constant used to evaluate the validation bounds.

        knots: np.ndarray
            A 1D array that contains the knot sequence including the endpoints.

        degree: int
            The polynomial degree of the B-spline function.

        lam_candidates: list[float]
            A list of candidate regularisation parameters to sweep over.

        Returns
        -------
        best_lam: float
           The hyperparameter candidate that minimized the boundary violation score.
        best_theta: np.ndarray
            The optimal B-spline coefficient array that corresponds to the best_lam value.

        Notes
        -----
        This function evaluates how well a fitted set of B-spline coefficients (`theta`) satisfies the rigorous
        upper and lower OPE bounds across multiple test coupling points. Any fitted value falling outside its valid
        analytical window incurs a linear penalty.
    """
    n_basis = len(knots) - degree - 1
    D = build_smoothness_penalty(n_basis)
    theta_cache = build_theta_block_cache(g_values, knots, degree)
    target_vector_cache = build_target_vector_cache(g_values)

    scores = {}
    thetas = {}
    for lam in lam_candidates:
        theta_fitted = solve_from_cached(g_values, theta_cache, target_vector_cache, D, lam)
        scores[lam] = violation_score(theta_fitted, test_g_values, knots, degree)
        thetas[lam] = theta_fitted

    best_lam = min(scores, key=scores.get)
    best_theta = thetas[best_lam]

    return (best_lam, best_theta)


if __name__ == '__main__':
    g_values_full = available_g_values[1:]
    test_g_values = available_g_ope_values[::10]
    degree = 3
    knots = build_knot_vector(g_values_full[0], g_values_full[-1], 16, 3)
    lam_candidates = np.logspace(-3, 4, 15)
    theta_bad = solve_convex_problem(g_values_full, knots, degree, lam = 1000.0)
    theta_good = solve_convex_problem(g_values_full, knots, degree, lam = 0.01)
    print(violation_score(theta_bad, available_g_ope_values[::20], knots, degree))
    print(violation_score(theta_good, available_g_ope_values[::20], knots, degree))

    best_lam, best_theta = sweep_lam_against_bounds(g_values_full, test_g_values, knots, degree, lam_candidates)
    print(f"The best lam (by considering bounds violation) is: {best_lam}")
    print(f"The shape of the best theta is: {best_theta.shape}")
