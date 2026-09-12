import numpy as np

from design_matrix import build_design_matrix
from spline_basis import build_basis_matrix
from design_matrix import build_target_vector

def build_theta_block(g: float, knots: np.ndarray, degree: int) -> np.ndarray:
    r"""
        Constructs a block matrix of outer products between a design matrix and B-spline basis.

        Parameters
        ----------
        g: float
            The value of the coupling constant.

        knots: np.ndarray
            A 1D array that contains the B-spline knot sequence.

        degree: int
            The degree of the polynomial of the B-spline basis.

        Returns
        -------
        np.ndarray
             A 2D array representing the horizontally stacked outer product matrices for all states.

        Notes
        -----
        The function relies on the following external components:
        - build_design_matrix: generates a matrix where columns correspond to states.
        - build_basis_matrix: generates the B-spline evaluation matrix.

        The build_basis_matrix(np.array([g]), knots, degree)[0] selection extracts the 1D basis vector that corresponds
        to the input value of the coupling constant, g.
    """
    A = build_design_matrix(g)
    B = build_basis_matrix(np.array([g]), knots, degree)[0]

    lst_outer_prod = []

    for state in range(10):
        states_outer_basis = np.outer(A[:,state], B)
        lst_outer_prod.append(states_outer_basis)

    stacked_lst_outer_prod = np.hstack(lst_outer_prod)

    return stacked_lst_outer_prod

def build_stacked_target_vector(g_values: list[float]) -> np.ndarray:
    r"""
        Concatenates target vectors evaluated across a sequence of values of the coupling constant, g.

        Parameters
        ----------
        g_values: list[float]
            A list of parameters at which to evaluate the target vector.

        Returns
        -------
        np.ndarray
             A 1D array containing all the individual target vectors joined end to end.

        Notes
        -----
        The function depends on build_target_vector.
        Must be called with the identical g_values list, in the same order,
        as the paired stacked-design-matrix/target_vector function, otherwise row k of one will no longer correspond
        to entry k of the other.
    """
    the_list = []
    for g_value in g_values:
        the_list.append(build_target_vector(g_value))

    stacked_list = np.concatenate(the_list)

    return stacked_list

def build_stacked_design_matrix(g_values: list[float], knots: np.ndarray, degree: int) -> np.ndarray:
    r"""
        A vertical stacked version of theta block matrices evaluated across a sequence of values of the coupling constant, g.

        Parameters
        ----------
        g_values: list[float]
            A list of parameters at which to evaluate the design matrix.
        knots: np.ndarray
            A 1D array that contains the B-spline knot sequence.
        degree: int
            The degree of the polynomial of the B-spline basis.

        Returns
        -------
        np.ndarray
             A 2D array that contains the vertically stacked theta blocks.

        Notes
        -----
        The function depends on build_theta_block.
        Must be called with the identical g_values list, in the same order,
        as the paired stacked-design-matrix/target_vector function, otherwise row k of one will no longer correspond
        to entry k of the other.
    """
    the_list = []
    for g_value in g_values:
        the_list.append(build_theta_block(g_value, knots, degree))

    stacked_list = np.vstack(the_list)

    return stacked_list
