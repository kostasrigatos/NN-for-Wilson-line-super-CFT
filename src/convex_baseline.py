import cvxpy as cp
import numpy as np
from stack_crossing import build_stacked_design_matrix, build_stacked_target_vector
from spline_basis import build_smoothness_penalty, build_basis_matrix

def solve_convex_problem(g_values: np.ndarray, knots: np.ndarray, degree: int, lam: float) -> np.ndarray:
    r"""
         Sets up and solves a regularised non-negative B-spline problem using the CVXPY solver.

         Parameters
         ----------
         g_values: np.ndarray
            A 1D array of values of the coupling constant, g, where the matrices of the system are evaluated.

        knots: np.ndarray
            A 1D array that contains the knot sequence including the endpoints.

        degree: int
            The polynomial degree of the B-spline function.

        lam: float
            The regularization parameter that controls the strength of the smoothness penalty.
            Larger values correspond to smoother spline profiles.

         Returns
         -------
         np.ndarray
             A 2D array of shape (10, n_basis) that contains the optimised non-negative line coefficients for each one
             of the 10 states.

         Notes
         -----
         It raises a RuntimeError if the CVXPY solver fails to find a global solution.
         Other than the CVXPY dependence, the function also depends on:
         1. build_stacked_target_vector
         2. build_stacked_design_matrix
         3. build_smoothness_penalty
     """
    n_basis = len(knots) - degree - 1

    A_stacked = build_stacked_design_matrix(g_values, knots, degree)
    b_stacked = build_stacked_target_vector(g_values)
    D = build_smoothness_penalty(n_basis)

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

def reconstruct_C_squared(theta: np.ndarray, g: float, knots: np.ndarray, degree: int) -> np.ndarray:
    r"""
         Reconstructs the 10 values for the squared 3-point coupling coefficients for a given value of the coupling
         constant, g, using the fitted theta matrix.

         Parameters
         ----------
         theta: np.ndarray
            The fitted weights matrix of shape (10, n_basis).

        g: float
            The coupling constant.

        knots: np.ndarray
            A 1D array that contains the knot sequence including the endpoints.

        degree: int
            The polynomial degree of the B-spline function.

         Returns
         -------
         np.ndarray
             An array of 10 values for the squared 3-point coupling coefficients at the given value for g.
     """
    basis_row = build_basis_matrix(np.array([g]), knots, degree)[0]
    c_squared_values = theta @ basis_row
    return c_squared_values

