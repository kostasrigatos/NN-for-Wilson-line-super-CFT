import numpy as np
from scipy.optimize import nnls
from design_matrix import build_target_vector, build_design_matrix

def sltn_nnls(g: float) -> tuple[np.ndarray, float]:
    r"""
        Solves a Non-Negative Least Squares (NNLS) problem for a given value of the coupling constant, g.

        Parameters
        ----------
        g: float
            The coupling constant.

        Returns
        -------
        sltn_vector: np.ndarray
           The solution vector with the non-negative coefficients.
        residual_norm: float
            The residual norm of the fit ||A \cdot sltn_vector -b||_2.
            Namely, how well the NNLS solution satisfies the crossing equation.
        Notes
        -----
        The function depends on build_target_vector to construct the target vector -b- and
        build_design_matrix to construct the design matrix -A.
    """
    b = build_target_vector(g)
    A = build_design_matrix(g)
    sltn_vector, residual_norm = nnls(A, b)

    return sltn_vector, residual_norm