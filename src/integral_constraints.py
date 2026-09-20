import torch
import numpy as np
from spectrum_data import spectrum_at_g
from constraints import int1, int2
from coupling import RHS1, RHS2, G_MIN
from spline_basis import build_basis_matrix

def build_integral_design(g: float) -> np.ndarray:
    r"""
        Constructs an integral design matrix by evaluating the integral constraints over the spectrum.

        Parameters
        ----------
        g: float
            The value of the coupling constant.

        Returns
        -------
        np.ndarray
             A 2D array where each column corresponds to a state's integral constraints.
    """
    values_for_Delta_aux = spectrum_at_g(g)
    values_for_Delta = torch.tensor(values_for_Delta_aux, dtype=torch.float64)

    cols= []
    for delta_value in values_for_Delta:
        col = torch.stack([int1(delta_value), int2(delta_value)])
        cols.append(col)

    A = torch.stack(cols, dim = 1)
    A = A.numpy()

    return A

def build_integral_target(g: float) -> np.ndarray:
    r"""
        Constructs the RHS of the integral constraints for a given value of the coupling constant, g.
        More specifically, the negative RHS of the integral constraints.

        Parameters
        ----------
        g: float
            The value of the coupling constant.

        Returns
        -------
        np.ndarray
             A 1D array representing evaluations of the integral constraints.
    """
    b = np.array([-float(RHS1(g)), - float(RHS2(g))])
    return b

def build_integral_block(g: float, knots: np.ndarray, degree: int) -> np.ndarray:
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
             A 2D array representing the horizontally stacked integral blocks for all 10 states.
    """
    A_int = build_integral_design(g)
    B = build_basis_matrix(np.array([g]), knots, degree)[0]

    lst_outer_prod = []

    for state in range(10):
        states_outer_basis = np.outer(A_int[:, state], B)
        lst_outer_prod.append(states_outer_basis)

    stacked_lst_outer_prod = np.hstack(lst_outer_prod)

    return stacked_lst_outer_prod

def _check_g_values_valid(g_values: list[float]) -> None:
    r"""
        Validates that every coupling constant in g_values meets or exceeds G_MIN, raising if any does not.

        Parameters
        ----------
        g_values: list[float]
            The list of coupling constant coordinates that need to be verified.

        Raises
        -------
        ValueError
             If any value inside g_values drops below the globally specified G_MIN bounds.
    """
    for g in g_values:
        if g < G_MIN:
            raise ValueError(f"g = {g} is below the minimum value for g, g_min = {G_MIN}."
                             f"The integral constraints rely on the asymptotic expansion of the curvature function C,"
                             f"which is valid only for g >= {G_MIN}."
                             )

def build_stacked_integral_block(g_values: list[float], knots: np.ndarray, degree: int) -> np.ndarray:
    r"""
        Vertically stacks the integral blocks for a sequence of coupling constant values into a single design matrix in
        theta-space.

        Parameters
        ----------
        g_values: list[float]
            A list of different values for the coupling constant at which to evaluate each integral block matrix.

        knots: np.ndarray
            A 1D array that contains the B-spline knot sequence.

        degree: int
            The degree of the polynomial of the B-spline basis.

        Returns
        -------
        np.ndarray
             A 2D array integral constraint design matrix containing the vertically stacked blocks.
    """
    _check_g_values_valid(g_values)

    the_list = []
    for g_value in g_values:
        the_list.append(build_integral_block(g_value, knots, degree))

    stacked_list = np.vstack(the_list)

    return stacked_list

def build_stacked_integral_target(g_values: list[float]) -> np.ndarray:
    r"""
        Concatenates the integral target vectors for a sequence of coupling constant values into a single stacked
        target vector.

        Parameters
        ----------
        g_values: list[float]
            A list of different values for the coupling constant at which to evaluate each integral block matrix.

        Returns
        -------
        np.ndarray
             A 1D array containing all individual integral target vector joined.
    """
    _check_g_values_valid(g_values)

    the_list = []
    for g_value in g_values:
        the_list.append(build_integral_target(g_value))

    stacked_list = np.concatenate(the_list)

    return stacked_list