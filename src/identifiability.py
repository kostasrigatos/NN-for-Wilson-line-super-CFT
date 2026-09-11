from design_matrix import build_design_matrix
from spectrum_data import spectrum_at_g
import numpy as np

def singular_values_at_g(g: float) -> np.ndarray:
    r"""
        Computes the singular values of the design matrix at a given value of the coupling constant, g.

        Parameters
        ----------
        g: float
            The coupling constant.

        Returns
        -------
        res_return: np.ndarray
           The singular values of the design matrix in descending order of magnitude.

        Notes
        -----
        g = 0 has to be excluded from any call to this function since the results are not physically meaningful.
        The conformal dimension, Δ, of state 1 is exactly 1.0 at g = 0 and lands directly on the pole of f_delta.
        This in turn makes the largest singular value larger by roughly 10^7.
    """
    A = build_design_matrix(g)
    res_return = np.linalg.svd(A, compute_uv=False)
    return res_return

def singular_values_sweep(g_values: list[float]) -> np.ndarray:
    r"""
        Makes a sweep over a list of values of the coupling constant and computes the singular values for each.

        Parameters
        ----------
        g_values: list[float]
            The list of values for the coupling constant to sweep over.

        Returns
        -------
        result: np.ndarray
           A 2D array of stacked singular values.

        Notes
        -----
        The function itself does not guard against the problematic value g = 0.
        The exclusion of g = 0 relies on calling singular_values_at_g within this function.
    """
    singular_values = []
    for g_value in g_values:
        sing_val = singular_values_at_g(g_value)
        singular_values.append(sing_val)
    result = np.stack(singular_values)

    return result

def near_degenerate_clusters(g: float, num_threshold: float) -> list[list[tuple[float, int]]]:
    r"""
        Groups the values of the spectrum into clusters based on near-degeneracy.

        Parameters
        ----------
        g: float
            The coupling constant.
        num_threshold: float
            The max gap between consecutive values such that they are part of the same degenerate cluster.

        Returns
        -------
        clusters: list[list[tuple[float, int]]]
           A list of clusters, where each cluster is a list of tuples.

        Notes
        -----
        num_threshold is not a universal parameter. The value 0.1 was verified to work cleanly near g = 0.
        No single value for num_threshold can cleanly separate "same family" from "different family" across
        the full range of values for the coupling constant, g. Two examples using the same value for num_threshold:
        1. g = 0.01 recovers [[1],[2,3],[4,5,6,7,8,9],[10]]
        2. g = 4 recovers [[1],[2,3],[4,6,5,8],[7],[10,9]]
    """
    spectrum_asc = sorted(zip(spectrum_at_g(g), range(1,11)))
    clusters = [[spectrum_asc[0]]]
    for i in range(len(spectrum_asc) - 1):
        gap = spectrum_asc[i + 1][0] - spectrum_asc[i][0]
        if gap < num_threshold:
            clusters[-1].append(spectrum_asc[i+1])
        else:
            clusters.append([spectrum_asc[i+1]])

    return clusters
