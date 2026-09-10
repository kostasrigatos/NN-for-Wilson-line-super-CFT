import torch
from blocks import F_delta, f_delta
from coupling import F_function

def C2_BPS(g: float) -> float:
    r"""
        The 3-point BPS couplings.

        Parameters
        ----------
        g: float
            The coupling constant.

        Returns
        -------
        float
            C^2_BPS(g) = F(g) - 1.

        Notes
        -----
        Cavaglià–Gromov–Julius–Preti (2203.09556), an inline equation between eq. (2.28) and eq. (2.29), page 11.
    """
    CBPS_squared_expr = F_function(g) - 1
    return CBPS_squared_expr

def f_I(x: torch.Tensor) -> torch.Tensor:
    r"""
        The identity superconformal block.

        Parameters
        ----------
        x: torch.Tensor
            Conformal cross-ratio; x = \frac{x_{12}x_{34}}{x_{13}x_{24}}.

        Returns
        -------
        torch.Tensor
            f_I(x) = x.

        Notes
        -----
        Cavaglià–Gromov–Julius–Preti (2203.09556), eq. (2.26), page 11.
    """
    f_I_expr = x
    return f_I_expr

def f_B2(x: torch.Tensor) -> torch.Tensor:
    r"""
        The B_2-multiplet superconformal block.

        Parameters
        ----------
        x: torch.Tensor
            Conformal cross-ratio; x = \frac{x_{12}x_{34}}{x_{13}x_{24}}.

        Returns
        -------
        torch.Tensor
            f_B2(x) = x - x 2F1(1, 2, 4; x).

        Notes
        -----
        Cavaglià–Gromov–Julius–Preti (2203.09556), eq. (2.27), page 11.
    """
    fB2_expr = f_I(x) - f_delta(torch.tensor(0.0, dtype=torch.float64),x)
    return fB2_expr

def h(g: float, x: torch.Tensor) -> torch.Tensor:
    r"""
        The simple superconformal block.

        Parameters
        ----------
        g: float, x: torch.Tensor
            The coupling constant.
        x: Conformal cross-ratio; x = \frac{x_{12}x_{34}}{x_{13}x_{24}}.

        Returns
        -------
        torch.Tensor
            h(g,x) = \mathcal{G}_{\mathcal{I}}(x) + C^2_{BPS}(g) \mathcal{G}_{\mathcal{B}_2}(x).

        Notes
        -----
        Cavaglià–Gromov–Julius–Preti (2203.09556), eq. (2.33), page 11.
    """
    term1 = x*(1-x) * F_function(g)
    term2 = C2_BPS(g) * F_delta(torch.tensor(0.0, dtype=torch.float64),x)
    simple_block = term1 - term2

    return simple_block