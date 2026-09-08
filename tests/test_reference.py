import pytest
from reference import j_squared, harmonic_number, anal_continuation_harmonic_number, s_minus_2_sum, s_minus_2, gamma1, gamma2, gamma3

def test_j_squared():
    assert j_squared(2) == 10
    assert j_squared(4) == 28

def test_harmonic_number():
    assert harmonic_number(4, 1) == pytest.approx(2.083333)
    assert harmonic_number(3, 2) == pytest.approx(1.361111)

def test_anal_continuation_matches_finite_sum_at_integers():
    for n in [0, 1, 2, 3, 4, 5]:
        assert anal_continuation_harmonic_number(n) == pytest.approx(harmonic_number(n, 2))

def test_s_minus_2_expressions():
    assert s_minus_2(2) == pytest.approx(s_minus_2_sum(2))
    assert s_minus_2(4) == pytest.approx(s_minus_2_sum(4))

def test_s_minus_2():
    with pytest.raises(AssertionError):
        s_minus_2(3)

def test_gamma_expressions():
    assert gamma1(2) == pytest.approx(-5)
    assert gamma2(2) == pytest.approx(295/24)
    assert gamma3(2) == pytest.approx(- 305/16)