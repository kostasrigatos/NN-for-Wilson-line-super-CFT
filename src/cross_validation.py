import numpy as np
import cvxpy as cp

from spline_basis import build_smoothness_penalty
from stack_crossing import build_theta_block
from design_matrix import build_target_vector

def build_theta_block_cache(g_values: list[float], knots: np.ndarray, degree: int) -> dict[float, np.ndarray]:
    r"""
         Pre-computes and caches the design matrix and theta blocks for a sequence of values of the coupling constant, g.

         Parameters
         ----------
         g_values: list[float]
            The list of values for the coupling constant to sweep over.
        knots: np.ndarray
            A 1D array that contains the B-spline knot sequence.
        degree: int
            The degree of the polynomial of the B-spline basis.

         Returns
         -------
         dict[float, np.ndarray]
           A dictionary that contains the cached theta block (the horizontally stacked outer-product matrix from
           build_theta_block) and the corresponding value of the coupling constant, g.
     """
    theta_block_cached = {}
    for g in g_values:
        theta_block_cached[g] = build_theta_block(g, knots, degree)
    return theta_block_cached

def build_target_vector_cache(g_values: list[float]) -> dict[float, np.ndarray]:
    r"""
         Pre-computes and cache the target vectors for a sequence of values of the coupling constant, g.

         Parameters
         ----------
         g_values: list[float]
            The list of values for the coupling constant to sweep over.

         Returns
         -------
         dict[float, np.ndarray]
             A dictionary with the cached target vector (b = - h(x_i, g)) for the corresponding value of the coupling
             constant, g.
     """
    target_vector_cached = {}
    for g in g_values:
        target_vector_cached[g] = build_target_vector(g)
    return target_vector_cached

def build_folds(g_values: list[float], n_folds: int) -> list[list[float]]:
    r"""
         Splits a list of values of the coupling constant, g, into a number n_folds of interleaved validation folds.

         Parameters
         ----------
         g_values: list[float]
            The list of values for the coupling constant to sweep over.

         n_folds: int
            The total number of validation folds.

         Returns
         -------
         list[list[float]]
             A nested list containing n_folds sublists, where each sublist holds the parameter elements that were
             assigned to that particular validation fold.
     """
    result = []
    for _ in range(n_folds):
        result.append([])
    for idx, g in enumerate(g_values):
        fold_idx = idx % n_folds
        result[fold_idx].append(g)

    return result

def solve_from_cached(g_subset: list[float], theta_cache: dict[float, np.ndarray], target_vector_cache: dict[float, np.ndarray], D: np.ndarray, lam: float) -> np.ndarray:
    r"""
         Solves the regularised B-spline optimisation problem for a subset of the available values for the coupling constant.

         Parameters
         ----------
         g_subset: list[float]
            The list of values for the coupling constant to sweep over.

         theta_cache: dict[float, np.ndarray]
            Cached 2D theta blocks mapped to their corresponding coupling constant values.

         target_vector_cache: dict[float, np.ndarray]
            Cached 1D target vectors mapped to their corresponding coupling constant values.

         D: np.ndarray
             The smoothness penalty matrix.

         lam: float
             The regularisation parameter scaling the penalty term.

         Returns
         -------
         np.ndarray
             A 2D array of shape (10, n_basis) that contains the optimised non-negative coefficients.

         Notes
         -----
         It raises a RuntimeError if the CVXPY solver fails to find a global solution.
     """
    n_basis = D.shape[1]

    A_stacked = np.vstack([theta_cache[g] for g in g_subset])
    b_stacked = np.concatenate([target_vector_cache[g] for g in g_subset])

    theta = cp.Variable((10, n_basis))

    residual_term = cp.sum_squares(A_stacked @ cp.reshape(theta, (10 * n_basis,), order = 'C') - b_stacked)
    smoothness_term = cp.sum_squares(D @ theta.T)

    objective = cp.Minimize(residual_term + lam * smoothness_term)

    constraints = [theta >= 0]

    problem = cp.Problem(objective, constraints)

    problem.solve()

    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f"Optimization failed with the status of the CVPXY solver: {problem.status}")

    return theta.value

def cv_score_for_lam(lam: float, folds: list[list[float]], theta_cache: dict[float, np.ndarray], target_vector_cache: dict[float, np.ndarray], D: np.ndarray) -> float:
    r"""
         Calculates the mean CV out-of-sample error for a specific value of lambda.

         Parameters
         ----------
         lam: float
             The regularisation parameter scaling the penalty term.

         folds: list[list[float]]
             The CV dataset partitioning layout.

         theta_cache: dict[float, np.ndarray]
            Cached 2D theta blocks mapped to their corresponding coupling constant values.

         target_vector_cache: dict[float, np.ndarray]
            Cached 1D target vectors mapped to their corresponding coupling constant values.

         D: np.ndarray
             The smoothness penalty matrix.

         Returns
         -------
         float
             The mean out-of-sample crossing-equation residual norm across all folds, for this value of lam.
     """
    resid_norms = []
    n_folds = len(folds)
    all_g_values = [g for fold in folds for g in fold]

    for idx in range(n_folds):
        cv_set = folds[idx]
        training_set = [g for g in all_g_values if g not in folds[idx]]
        theta_fitted = solve_from_cached(training_set, theta_cache, target_vector_cache, D, lam)

        for g in cv_set:
            residual = (theta_cache[g] @ theta_fitted.flatten()) - target_vector_cache[g]
            res_norm = np.linalg.norm(residual)
            resid_norms.append(res_norm)

    lam_cv_score = float(np.mean(resid_norms))

    return lam_cv_score

def run_cross_validation(g_values: list[float], knots: np.ndarray, degree: int, lam_candidates: list[float], n_folds: int) -> tuple[float, np.ndarray]:
    r"""
         Performs a complete K-fold CV loop to identify the optimal value of the hyperparameter.

         Parameters
         ----------
         g_values: list[float]
            The list of values for the coupling constant.

         knots: np.ndarray
             A 1D array that contains the knot sequence including the endpoints.

         degree: int
            The polynomial degree of the B-spline function.

         lam_candidates: list[float]
             A list of values for the hyperparameter to sweep over and score.

         n_folds: int
            The total number of validation folds.

         Returns
         -------
         best_lam : float
            The best hyperparameter from the list of candidates that minimised the CV score.
         best_theta_fitted : np.ndarray
            The optimal 2D B-spline coefficient array computed over the full data using `best_lam`.
     """
    n_basis = len(knots) - degree - 1
    D = build_smoothness_penalty(n_basis)

    theta_cache = build_theta_block_cache(g_values, knots, degree)
    target_vector_cache = build_target_vector_cache(g_values)
    folds = build_folds(g_values, n_folds)

    scores = {}
    for lam in lam_candidates:
        scores[lam] = cv_score_for_lam(lam, folds, theta_cache, target_vector_cache, D)

    best_lam = min(scores, key = scores.get)
    best_theta_fitted = solve_from_cached(g_values, theta_cache, target_vector_cache, D, best_lam)

    return (best_lam, best_theta_fitted)

