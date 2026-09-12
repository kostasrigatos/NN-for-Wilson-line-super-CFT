import numpy as np
from scipy.interpolate import BSpline

def build_knot_vector(g_min: float, g_max: float, n_interior_knots: int, degree: int) -> np.ndarray:
    r"""
        Constructs a B-spline knot vector with geometrically spaced interior knots.

        Parameters
        ----------
        g_min: float
            The minimum value of the coupling constant.

        g_max: float
            The maximum value of the coupling constant.

        n_interior_knots: int
            The number of unique interior knot positions to generate.

        degree: int
            The polynomial degree of the B-spline.

        Returns
        -------
        np.ndarray
             A 1D array representing the complete and sorted B-spline knot vector.
    """
    pool_points = np.geomspace(g_min, g_max, num=n_interior_knots + 2)
    pool_points = pool_points[1:-1]
    left_pad = np.repeat(g_min, degree + 1)
    right_pad = np.repeat(g_max, degree + 1)
    stacked = np.concatenate([left_pad, pool_points, right_pad])

    return stacked


def build_basis_matrix(g_values: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    r"""
        Constructs a B-spline basis matrix given a set of parameters.


        Parameters
        ----------
        g_values: np.ndarray
            A 1D array of values of the coupling constant, g, where the spline basis functions are evaluated.

        knots: np.ndarray
            A 1D array that contains the knot sequence including the endpoints.

        degree: int
            The polynomial degree of the B-spline function.

        Returns
        -------
        np.ndarray
             A 2D standard dense array representing B-spline design matrix.

        Notes
        -----
        The function depends on scipy.interpolate.BSpline.design_matrix that outputs a sparse matrix.
        The method .toarray() is called explicitly to return a standard dense NumPy array.
    """
    design_mat = BSpline.design_matrix(g_values, knots, degree).toarray()

    return design_mat

def build_smoothness_penalty(n_basis: int) -> np.ndarray:
    r"""
        Constructs a discrete second-difference penalty matrix for the spline basis.


        Parameters
        ----------
        n_basis: int
            The total number of B-spline basis functions; this is equal to the number of the spline coefficients.

        Returns
        -------
        np.ndarray
             A 2D array shaped (n_basis -2, n_basis) that represents the second-difference operator. For the cases
             n_basis < 3 it returns a matrix (0, n_basis).

        Notes
        -----
        This matrix acts as a regularisation operator that penalises rapid changes in the spline coefficients.
        When multiplied by a coefficient, c, it computes the second-order differences, c_{i+2} - 2 c_{i+1} + c_i.
    """
    if n_basis < 3:
        return np.empty((0, n_basis))

    ident_matrix = np.eye(n_basis)
    first_diff_op = np.diff(ident_matrix, axis=0)
    second_diff_op = np.diff(first_diff_op, axis=0)

    return second_diff_op
