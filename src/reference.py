"""
Index / labelling convention
-----------------------------
Operator labels (Delta_1, Delta_2, ... or "phi^2", "the Delta^0=6 quartet")
are NOT stable across the literature this module draws from, because the
spectrum has genuine level crossings in g (Cavaglia et al., Fig. 3).

This module uses the free theory / weak-coupling labelling convention:
operators are identified by their free theory dimension Delta^(0) = J
(so "J=1" is the unique dimension-1 state, "J=2" is the two-state
degeneracy, etc.), NOT by an at-strong-coupling numeric ordering.

Known cross-references:
- phi^2 (Ferrero-Meneghelli: 2103.10440, 2312.12550, 2312.12551) == the unique J=1 state == "Delta_1" in
  Cavaglia et al.'s 2107.08510 NUMERIC (strong-coupling-ordered) convention.
- The "C4^2+C5^2+C6^2+C8^2" quartet (project brief, MultiSTOP-derived
  numbering - 2404.14909) == the J=6 degeneracy (4 states) == Cavaglia et al.'s
  "C4^2+C5^2+C6^2+C7^2" (their numeric labels, NOT J values).
Do not mix numeric-order labels and J-labels without translating.
"""
from scipy import special

def j_squared(Delta: int) -> int:
    r"""
        Casimir eigenvalue for operators with in the representation ω = {∆, 0, [0, 0]} of osp(4^{*}|4).

        Parameters
        ----------
        Delta : int
            Even integer scaling dimension label (free-theory / weak-coupling
            convention -- see module docstring).

        Returns
        -------
        int
            Δ (Δ + 3).

        Notes
        -----
        Ferrero-Meneghelli (2103.10440), eq. (C2), page 11.
    """
    j_sq_expr = Delta * (Delta + 3)
    return j_sq_expr

def harmonic_number(n: int, m: int) -> float:
    r"""
        Definition of harmonic number H^{(m)}_{n}

        Parameters
        ----------
        n, m : int

        Returns
        -------
        float
            H^{(m)}_{n} = sum_{k=1}^{n} 1 / k^m.

        Notes
        -----
        Ferrero-Meneghelli (2103.10440), eq. (C4), page 11.
    """
    total_sum = 0.0
    for k in range(1, n + 1):
        total_sum += 1 / (k ** m)
    return total_sum

def anal_continuation_harmonic_number(n: float) -> float:
    r"""
        Definition of harmonic number H^{(2)}_{n}

        Parameters
        ----------
        n: float

        Returns
        -------
        float
            H^{(2)}_{n} = ζ(2) - ψ^{(1)}(n+1).

        Notes
        -----
        This is the analytic continuation of H^{(2)}_{n} from eq.(C4) using the trigamma identity.
    """
    h2_n = special.zeta(2) - special.polygamma(1, n+1)
    return h2_n

def s_minus_2_sum(Delta: int) -> float:
    r"""
        Alternating harmonic sum S_{-2}(Δ), integer args.

        Parameters
        ----------
        Delta : int
            Even integer scaling dimension label (free-theory / weak-coupling
            convention -- see module docstring).

        Returns
        -------
        float
            S_{-2}(Δ) = sum_{n=1}^{Δ} (-1)^n / n^2.

        Notes
        Reference implementation, used only in tests
    """
    total_sum = 0.0
    for n in range(1, Delta + 1):
        total_sum += ((-1) ** n) / (n ** 2)
    return total_sum

def s_minus_2(Delta: int) -> float:
    r"""
        Alternating harmonic sum S_{-2}(Δ), continued to half-integer args.

        Parameters
        ----------
        Delta : int
            Even integer scaling dimension label (free-theory / weak-coupling
            convention -- see module docstring).

        Returns
        -------
        float
            S_{-2}(Δ) = sum_{n=1}^{Δ} (-1)^n / n^2.

        Notes
        -----
        Ferrero-Meneghelli (2103.10440), eq. (C3), page 11. The literal formula involves
        H^{(2)}_{Delta/2} and H^{(2)}_{(Δ-1)/2}, and the second of these has
        a half-integer argument -- so it is evaluated via the trigamma
        continuation H_n^{(2)} = ζ(2) - ψ^{(1)}(n+1), not a finite sum.
        Validated against the literal alternating sum in
        tests/test_reference.py for Δ = 2, 4.
    """

    assert Delta % 2 == 0, "s_minus_2 assumes even Delta (the (-1)^Delta factor is dropped)"

    upper = anal_continuation_harmonic_number(Delta / 2)
    lower = anal_continuation_harmonic_number((Delta - 1) / 2)
    s_minus_2_expr = (1/4) * (upper - lower) - (1/2) * special.zeta(2)
    return s_minus_2_expr

def gamma1(Delta: int) -> float:
    r"""
        Tree-level anomalous dimension <γ^{(1)}_{Delta}>.

        Parameters
        ----------
        Delta : int
            Even integer scaling dimension label (free-theory / weak-coupling
            convention -- see module docstring).

        Returns
        -------
        float
            <γ^{(1)}_Δ> = - (1/2) j^2_Δ.

        Notes
        -----
        Ferrero-Meneghelli (2103.10440), eq. (C1), page 11.
    """
    gamma1_expr = - (1/2) * j_squared(Delta)
    return gamma1_expr

def gamma2(Delta: int) -> float:
    r"""
        One-loop anomalous dimension <γ^{(2)}_Δ>.

        Parameters
        ----------
        Delta : int
            Even integer scaling dimension label (free-theory / weak-coupling
            convention -- see module docstring).

        Returns
        -------
        float
            <γ^{(2)}_Δ> = (1/8) j^2_Δ ((Δ-1)(4Δ^2 + 11Δ + 8) / (j^2_Δ + 2) + 4 H_Δ).

        Notes
        -----
        Ferrero-Meneghelli (2103.10440), eq. (C1), page 11.
    """
    numerator = (Delta - 1) * (4 * Delta ** 2 + 11 * Delta + 8)
    denominator = j_squared(Delta) + 2
    gamma2_expr = (1/8) * j_squared(Delta) * (numerator / denominator + 4 * harmonic_number(Delta, 1))
    return gamma2_expr

def gamma3(Delta: int) -> float:
    r"""
        Two-loop anomalous dimension <γ^{(3)}_Δ>.

        Parameters
        ----------
        Delta : int
            Even integer scaling dimension label (free-theory / weak-coupling
            convention -- see module docstring).

        Returns
        -------
        float
            <γ^{(3)}_Δ> = long, complicated expression with 5 terms.

        Notes
        -----
        Ferrero-Meneghelli (2103.10440), eq. (C1), page 11.

        KNOWN ISSUE: gamma3(2) currently gives -755/16, but eq. (1) of
        2103.10440 requires -305/16.
    """
    term1 = (j_squared(Delta) / 4) * harmonic_number(Delta, 2)
    term2 = (j_squared(Delta) / 2) * (harmonic_number(Delta, 1) ** 2)
    term3 = (1/4) * j_squared(Delta) * (j_squared(Delta) - 2) * s_minus_2(Delta)
    num_term4 = (Delta - 1) * (2 * Delta **4 + 12 * Delta ** 3 + 24 * Delta ** 2 + 21 * Delta + 12)
    denom_term4 = 2 * (j_squared(Delta) + 2)
    term4 = (num_term4 / denom_term4) * harmonic_number(Delta, 1)
    num_term5 = Delta * (5 * Delta ** 5 + 23 * Delta ** 4 + 17 * Delta ** 3 - 45 * Delta ** 2 - 71 * Delta - 21)
    denom_term5 = 8 * (j_squared(Delta) + 2)
    term5 =  num_term5 / denom_term5
    gamma3_expr = term1 - term2 + term3 - term4 - term5
    return gamma3_expr

def delta_phi2(u: float) -> float:
    r"""
        Scaling dimension of the lightest non-protected scalar operator.

        Parameters
        ----------
        u : float
            u = 1 / \sqrt{λ}.

        Returns
        -------
        float
            <Δ_{φ^2}> = long, complicated expression with 5 terms.

        Notes
        -----
        Ferrero-Meneghelli (2103.10440), eq. (1), page 1.
    """
    term1 = 2
    term2 = 5
    term3 = 295/24
    term4 = 305/16
    term5 = (351845/13824 - 75/2 * special.zeta(3))
    delta_phi2_expr = term1 - term2 * u + term3 * u ** 2 - term4 * u ** 3 + term5 * u ** 4
    return  delta_phi2_expr

def mu_phi2_squared(u: float) -> float:
    r"""
        Squared OPE coefficient of the lightest non-protected scalar operator.

        Parameters
        ----------
        u : float
            u = 1 / \sqrt{λ}.

        Returns
        -------
        float
            <μ^2_{φ^2}> = long, complicated expression with 5 terms.

        Notes
        -----
        Ferrero-Meneghelli (2103.10440), eq. (30), page 5.
    """
    term1 = 2 / 5
    term2 = 43 / 30
    term3 = 5 / 6
    term4 = (11195 / 1728 + 4 * special.zeta(3))
    term5 = (1705 / 96 + 1613 / 24 * special.zeta(3))
    mu_phi2_squared_expr = term1 - term2 * u + term3 * u ** 2 + term4 * u ** 3 - term5 * u ** 4
    return mu_phi2_squared_expr

def mu_D2_squared(u: float) -> float:
    r"""
            Perturbative expansion of squared OPE coefficient for the exchange of the short operator.

            Parameters
            ----------
            u : float
                u = 1 / \sqrt{λ}.

            Returns
            -------
            float
                <μ^2_{D_2}> = 2 - 3 / \sqrt{λ} + 45 / (8 λ^{3/2}) + 45 / (4 λ^2).

            Notes
            -----
            Ferrero-Meneghelli (2103.10440), eq. (A.23), page 8.
        """
    term1 = 2
    term2 = 3
    term3 = 45/8
    term4 = 45/4
    mu_D2_squared_expr = term1 - term2 * u + term3 * u ** 3 + term4 * u ** 4
    return mu_D2_squared_expr

