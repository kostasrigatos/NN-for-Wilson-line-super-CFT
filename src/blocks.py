import torch


def hyper2f1(Delta: torch.Tensor, x: torch.Tensor, n_terms = 300) -> torch.Tensor:
    r"""
         The hypergeometric function 2F1 as an infinite series of Pochhammer symbols.

         Parameters
         ----------
         Delta, x: torch.Tensor both
            Delta: Even integer scaling dimension label (free-theory / weak-coupling
            convention -- see module docstring).
            x: Conformal cross-ratio; x = \frac{x_{12}x_{34}}{x_{13}x_{24}}.

         Returns
         -------
         torch.Tensor
             hyper2f1(Δ, x) = \sum^{∞}_{n=0} (a)_n (b_n) / (c)_n x^n / n!.

         Notes
         -----
         This is a standard formula, but as a reference Trenta–Bacciu–Cossu–Ferrero (2404.14909), eq. (5), page 6.
         accuracy degrades sharply as x→1; series converges like 1/n² there
         Validated only for x ≤ 0.85, matching the project's actual sample range — see test_blocks.py.
     """
    Delta = torch.as_tensor(Delta).double()
    x = torch.as_tensor(x).double()

    a, b, c = Delta + 1, Delta + 2, 2 * Delta + 4

    term = torch.ones_like(Delta)
    total = term.clone()

    for n in range(n_terms):
        term  = term * ((a + n) * (b + n)) / ((c + n) * (n + 1)) * x
        total = total + term

    return total

def f_delta(Delta: torch.Tensor, x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    r"""
        The superconformal blocks f_{Δ}(g).

        Parameters
        ----------
        Delta, x: torch.Tensor both
            Delta: Even integer scaling dimension label (free-theory / weak-coupling
            convention -- see module docstring).
        x: Conformal cross-ratio; x = \frac{x_{12}x_{34}}{x_{13}x_{24}}.

        Returns
        -------
        torch.Tensor
            f_{Δ}(Δ, x) = χ^{Δ + 1}/(1 - Δ) 2F1(Δ + 1, Δ + 2, 2Δ + 4; x).

        Notes
        -----
        Cavaglià–Gromov–Julius–Preti (2203.09556), eq. (2.28), page 11.
        We have used a safe-guard for a proper handling of the Δ = 1 pole.
    """
    denominator = 1 - Delta
    sign = torch.where(denominator >=0, 1.0, -1.0)
    safe_denominator = denominator + sign * eps

    block_expr = (x**(Delta + 1)) / safe_denominator * hyper2f1(Delta, x)

    return block_expr

def F_delta(Delta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""
        The crossed superconformal blocks F_{Δ}(g).

        Parameters
        ----------
        Delta, x: torch.Tensor both
            Delta: Even integer scaling dimension label (free-theory / weak-coupling
            convention -- see module docstring).
        x: Conformal cross-ratio; x = \frac{x_{12}x_{34}}{x_{13}x_{24}}.

        Returns
        -------
        torch.Tensor
            F_{Δ}(Δ, x) = x^2 f_{Δ}(Δ, 1 - x) + (1 - x)^2 f_{Δ}(Δ, x).

        Notes
        -----
        Cavaglià–Gromov–Julius–Preti (2203.09556), eq. (2.32), page 12.
    """
    crossed_block_expr = x**2 * f_delta(Delta, 1 - x) + (1 - x)**2 * f_delta(Delta, x)

    return crossed_block_expr

