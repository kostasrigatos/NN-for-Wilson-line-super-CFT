import math
from scipy import special

G_MIN = 3  # curvature_C's eq.(2.14) truncation: all 5 consecutive-term
           # ratios drop below 0.1 by g=3 (worst case t2/t1 ≈ 0.075).
           # See notebooks/figures.ipynb for the derivation and plot.
           # This means u_max = 0.026 and λ_min = 1422

def B_function(g: float) -> float:
    r"""
        The Bremsstrahlung function B(g).

        Parameters
        ----------
        g: float
            The coupling constant.

        Returns
        -------
        float
            B(g) = (g/π) (I_2(4πg)/I_1(4πg)).

        Notes
        -----
        Cavaglià–Gromov–Julius–Preti (2203.09556), eq. (2.11), page 7.
        We use ive (exponentially-scaled Bessel) rather than iv to avoid overflow for g ≳ 56;
        mathematically identical since the exp(-x) cancels in the ratio — see tests/test_coupling.py
    """
    B_function_expr = (g / math.pi) * (special.ive(2, 4 * math.pi * g))/(special.ive(1, 4 * math.pi * g))
    return B_function_expr

def F_function(g: float) -> float:
    r"""
        The constant F(g).

        Parameters
        ----------
        g: float
            The coupling constant.

        Returns
        -------
        float
            F(g) = 1 + C^2_{BPS}(g).

        Notes
        -----
        Cavaglià–Gromov–Julius–Preti (2203.09556), eq. (2.29), page 11.
        We use ive (exponentially-scaled Bessel) rather than iv to avoid overflow for g ≳ 56;
        mathematically identical since the exp(-x) cancels in the ratio — see tests/test_coupling.py
    """
    num   = 3 * special.ive(1, 4 * g * math.pi) * ((2 * math.pi**2 * g**2 + 1)*special.ive(1, 4 * g * math.pi) - 2 * g * math.pi * special.ive(0, 4 * g * math.pi))
    denom = 2 * g**2 * math.pi ** 2 * (special.ive(2, 4 * g * math.pi)) ** 2
    F_function_expr = num / denom
    return F_function_expr

def F_function_cross_check(g: float) -> float:
    r"""
        The constant F(g).

        Parameters
        ----------
        g: float
            The coupling constant.

        Returns
        -------
        float
            F(g) = (3(g^2 - B(g))) / (π^2 (B(g))^2).

        Notes
        -----
        Cavaglià–Gromov–Julius–Preti (2203.09556), eq. (2.30), page 11.
        Algebraically independent form via B(g), used to validate F_function
    """
    F_function_cross_check_expr = 3 * (g**2 - B_function(g)) / (math.pi**2 * B_function(g)**2)
    return F_function_cross_check_expr

def curvature_C(g: float) -> float:
    r"""
        The  perturbative expansion of the curvature function, C(g) at strong coupling truncated at O(1/g^4).

        Parameters
        ----------
        g: float
            The coupling constant.

        Returns
        -------
        float
            C(g) = long-complicated expression.

        Notes
        -----
        Cavaglià–Gromov–Julius–Preti (2203.09556), eq. (2.14), page 8.
        Valid for g ≥ G_MIN (see that constant's comment for the derivation), and also
        see notebooks/figures.ipynb for the derivation and plot..
    """
    term1 = ((2 * math.pi**2 -3)*g) / (6 * math.pi**3)
    term2 = (-24 * special.zeta(3) + 5 - 4 * math.pi**2) / (32 * math.pi**4)
    term3 = (11 + 2 * math.pi**2) / (256 * math.pi**5 * g)
    term4 = (96 * special.zeta(3) + 75 + 8 * math.pi**2) / (4096 * math.pi**6 * g**2)
    term5 = (3*(408*special.zeta(3) - 240 * special.zeta(5) + 213 + 14 * math.pi**2)) / (65536 * math.pi**7 * g**3)
    term6 = (3*(315*special.zeta(3) - 240 * special.zeta(5) + 149 + 6 * math.pi**2)) / (65536 * math.pi**8 * g**4)
    curvature_C_expr = term1 + term2 + term3 + term4 + term5 + term6
    return curvature_C_expr

def RHS1(g: float) -> float:
    r"""
        Explicit function of the coupling constant.

        Parameters
        ----------
        g: float
            The coupling constant.

        Returns
        -------
        float
            RHS_1(g) = (B(g) - 3 C(g))/(8 (B(g))^2) + (7 log(2) - 41/8)(F(g)-1) + log(2).

        Notes
        -----
        Cavaglià–Gromov–Julius–Preti (2203.09556), eq. (4.36), page 27.
    """
    term1 = (B_function(g) - 3 * curvature_C(g)) / (8 * B_function(g)**2)
    term2 = (7*math.log(2) - 41/8) * (F_function(g) - 1)
    right_hand_expression = term1 + term2 + math.log(2)
    return right_hand_expression

def RHS2(g: float) -> float:
    r"""
        Explicit function of the coupling constant.

        Parameters
        ----------
        g: float
            The coupling constant.

        Returns
        -------
        float
            RHS_2(g) = (1 - F(g))/6 + (2 - F(g)) log(2) + 1 - C(g)/(4 (B(g))^2).

        Notes
        -----
        Cavaglià–Gromov–Julius–Preti (2203.09556), eq. (4.37), page 27.
    """
    term1 = (1 - F_function(g)) / 6
    term2 = (2 - F_function(g)) * math.log(2)
    term3 = 1 - (curvature_C(g)) / (4 * B_function(g)**2)
    right_hand_expression = term1 + term2 + term3
    return right_hand_expression

if __name__ == "__main__":
    print(f"For g = 3.0, the RHS1 is equal to {RHS1(3.0):.5f} and the RHS2 is equal to {RHS2(3.0):.5f}")
    print(f"For g = 3.5, the RHS1 is equal to {RHS1(3.5):.5f} and the RHS2 is equal to {RHS2(3.5):.5f}")
    print(f"For g = 4.0, the RHS1 is equal to {RHS1(4.0):.5f} and the RHS2 is equal to {RHS2(4.0):.5f}")