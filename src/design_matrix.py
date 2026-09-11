from pathlib import Path

import importlib.util
spec = importlib.util.spec_from_file_location(
    "bootstop_z_points",
    Path(__file__).parent.parent / 'data' / 'bootstop_z_points.py'
)
bootstop_z_points = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstop_z_points)
z_points = bootstop_z_points.z_points

import torch
import numpy as np
from crossing import h
from blocks import F_delta
from spectrum_data import spectrum_at_g

z_tensor = torch.tensor(z_points, dtype=torch.float64)

def build_target_vector(g: float) -> np.ndarray:
    r"""
        First ingredient of the crossing symmetry equation; the target vector.

        Parameters
        ----------
        g: float
            The coupling constant.

        Returns
        -------
        np.ndarray
           b = - h(x_i, g).

        Notes
        -----
        Cavaglià–Gromov–Julius–Preti (2203.09556), eq. (2.31), page 12 is crossing symmetry equation.
        This bootstrap constraint for a fixed value of the coupling g can be re-arranged into a standard
        linear-algebra shape; A \cdot c = b.
    """
    b = - h(g, z_tensor)
    b = b.numpy()
    return b

def build_design_matrix(g: float) -> np.ndarray:
    r"""
        Second ingredient of the crossing symmetry equation; the design matrix.

        Parameters
        ----------
        g: float
            The coupling constant.

        Returns
        -------
        np.ndarray
           A = (350, 10) matrix with each column being equal to F_delta(delta_value, z_tensor) for the specific value
           of delta_value.

        Notes
        -----
        Cavaglià–Gromov–Julius–Preti (2203.09556), eq. (2.31), page 12 is crossing symmetry equation.
        This bootstrap constraint for a fixed value of the coupling g can be re-arranged into a standard
        linear-algebra shape; A \cdot c = b.
    """
    values_for_Delta_aux = spectrum_at_g(g)
    values_for_Delta = torch.tensor(values_for_Delta_aux, dtype=torch.float64)

    cols= []
    for delta_value in values_for_Delta:
        col = F_delta(delta_value, z_tensor)
        cols.append(col)

    A = torch.stack(cols, dim = 1)
    A = A.numpy()

    return A



