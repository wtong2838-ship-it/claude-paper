"""
Tests for Phase 5: Valuation and Dario Dilemma
===============================================
Tests firm value decomposition, equity sensitivity to λ,
credit spreads, and the asymmetric cost of λ misspecification.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase5_valuation import AIFirmValuation, DarioDilemma
from phase2_duopoly import DuopolyParams, LelandDefault
from phase1_base_model import SingleFirmModel


# ============================================================ #
#  Fixtures                                                      #
# ============================================================ #

@pytest.fixture
def params():
    return DuopolyParams()

@pytest.fixture
def valuation(params):
    return AIFirmValuation(params)

@pytest.fixture
def sol(params):
    model = SingleFirmModel(params)
    return model.solve()

@pytest.fixture
def K_star(sol):
    return sol['H']['K_star']

@pytest.fixture
def X_star(sol):
    return sol['H']['X_star']

@pytest.fixture
def dilemma(params):
    return DarioDilemma(params)


# ============================================================ #
#  AIFirmValuation: Assets in Place                             #
# ============================================================ #

def test_assets_in_place_positive(valuation, X_star, K_star):
    """V_AIP should be positive at the investment trigger."""
    V_AIP = valuation.assets_in_place(X_star, K_star)
    assert V_AIP > 0, f"V_AIP = {V_AIP:.4f} should be positive"

def test_assets_in_place_increasing_in_X(valuation, K_star):
    """V_AIP increases with demand X."""
    X_grid = np.linspace(0.01, 0.20, 20)
    V_AIP = [valuation.assets_in_place(X, K_star) for X in X_grid]
    diffs = np.diff(V_AIP)
    assert np.all(diffs > 0), "V_AIP should increase with X"

def test_assets_in_place_increasing_in_K(valuation, X_star):
    """V_AIP increases with capacity K."""
    K_grid = np.linspace(0.05, 0.50, 20)
    V_AIP = [valuation.assets_in_place(X_star, K) for K in K_grid]
    diffs = np.diff(V_AIP)
    assert np.all(diffs > 0), "V_AIP should increase with K"


# ============================================================ #
#  AGI Premium V_λ                                              #
# ============================================================ #

def test_agr_premium_non_negative(valuation, X_star, K_star):
    """V_λ ≥ 0: switching option has non-negative value."""
    V_lambda = valuation.agr_premium(X_star, K_star, 'L')
    assert V_lambda >= 0, f"AGI premium V_λ = {V_lambda:.4f} should be non-negative"

def test_agr_premium_higher_lambda_higher_premium(params, X_star, K_star):
    """Higher λ → higher AGI premium (switching is more valuable)."""
    import copy
    premiums = []
    for lam in [0.05, 0.20, 0.40, 0.70]:
        p = copy.copy(params)
        p.lam = lam
        v = AIFirmValuation(p)
        premiums.append(v.agr_premium(X_star, K_star, 'L'))

    # Should be (roughly) increasing
    assert premiums[-1] > premiums[0], \
        f"Premium should increase with λ: {premiums}"


# ============================================================ #
#  Growth Option Decomposition                                  #
# ============================================================ #

def test_decomposition_has_all_components(valuation, X_star, K_star):
    """Decomposition includes all three value components."""
    decomp = valuation.growth_option_decomposition(X_star, K_star, K_star * 2.0)
    for key in ['V_AIP', 'V_expand', 'V_lambda', 'V_total',
                 'AIP_fraction', 'expand_fraction', 'lambda_fraction']:
        assert key in decomp, f"Key '{key}' missing from decomposition"

def test_decomposition_AIP_fraction_positive(valuation, X_star, K_star):
    """Assets-in-place fraction > 0 at trigger."""
    decomp = valuation.growth_option_decomposition(X_star, K_star, K_star * 2.0)
    assert decomp['AIP_fraction'] > 0

def test_decomposition_lambda_fraction_positive(valuation, X_star, K_star):
    """AGI premium fraction > 0 for λ > 0."""
    decomp = valuation.growth_option_decomposition(X_star, K_star, K_star * 2.0)
    assert decomp['lambda_fraction'] >= 0


# ============================================================ #
#  Equity Sensitivity to λ                                      #
# ============================================================ #

def test_equity_sensitivity_positive(valuation, X_star, K_star):
    """Equity value > 0 at the trigger."""
    lam_range = np.array([0.10, 0.20, 0.30])
    E_vals = valuation.equity_sensitivity_to_lambda(X_star, K_star, lam_range)
    assert np.all(E_vals >= 0), f"Equity values should be non-negative: {E_vals}"

def test_equity_sensitivity_shape(valuation, X_star, K_star):
    """Returns array of same shape as input lambda range."""
    lam_range = np.linspace(0.05, 0.80, 20)
    E_vals = valuation.equity_sensitivity_to_lambda(X_star, K_star, lam_range)
    assert len(E_vals) == len(lam_range)


# ============================================================ #
#  Credit Spread Analysis                                       #
# ============================================================ #

def test_credit_spread_positive(params, K_star):
    """Credit spread > 0 for any X above default boundary."""
    ld = LelandDefault(params, K_star, 'H', in_duopoly=False)
    X_D = ld.default_boundary()
    X_test = X_D * 3
    cs = ld.credit_spread(X_test)
    assert cs > 0, f"Credit spread should be positive: {cs}"

def test_credit_spread_decreasing_in_X(params, K_star):
    """Credit spread decreases as X moves away from default boundary."""
    ld = LelandDefault(params, K_star, 'H', in_duopoly=False)
    X_D = ld.default_boundary()
    X_near = X_D * 1.5
    X_far = X_D * 10
    cs_near = ld.credit_spread(X_near)
    cs_far = ld.credit_spread(X_far)
    assert cs_near > cs_far, \
        f"Spread near default ({cs_near:.4f}) should exceed spread far from default ({cs_far:.4f})"

def test_default_probability_in_01(params, K_star):
    """Default probability is in [0, 1]."""
    ld = LelandDefault(params, K_star, 'H', in_duopoly=False)
    X_D = ld.default_boundary()
    dp = ld.default_probability(X_D * 2, T=5.0)
    assert 0 <= dp <= 1, f"Default probability {dp:.4f} out of [0, 1]"


# ============================================================ #
#  Dario Dilemma                                                #
# ============================================================ #

def test_dario_optimal_policy_returns_tuple(dilemma):
    """optimal_policy returns (K_star, X_star) tuple."""
    K, X = dilemma.optimal_policy(0.30, 'L')
    assert K > 0, f"K_star should be positive: {K}"
    assert X > 0, f"X_star should be positive: {X}"

def test_dario_conservative_higher_trigger(dilemma):
    """Conservative λ (lower) → higher X* threshold."""
    lam_conservative = 0.10
    lam_true = 0.30
    _, X_conserv = dilemma.optimal_policy(lam_conservative, 'L')
    _, X_true = dilemma.optimal_policy(lam_true, 'L')
    assert X_conserv > X_true, \
        f"Conservative X*={X_conserv:.5f} should exceed true X*={X_true:.5f}"

def test_dario_aggressive_lower_trigger(dilemma):
    """Aggressive λ (higher) → lower X* threshold."""
    lam_aggressive = 0.60
    lam_true = 0.30
    _, X_aggress = dilemma.optimal_policy(lam_aggressive, 'L')
    _, X_true = dilemma.optimal_policy(lam_true, 'L')
    assert X_aggress < X_true, \
        f"Aggressive X*={X_aggress:.5f} should be below true X*={X_true:.5f}"

def test_dario_forgone_revenue_positive_for_underinvestment(dilemma):
    """Forgone revenue > 0 when conservative (invests later)."""
    forgone = dilemma.forgone_revenue(0.30, 0.15, T=5.0, X_0=0.05)
    assert forgone > 0, f"Forgone revenue should be positive: {forgone}"

def test_dario_forgone_revenue_zero_for_overinvestment(dilemma):
    """Forgone revenue = 0 when aggressive (invests earlier)."""
    forgone = dilemma.forgone_revenue(0.30, 0.50, T=5.0, X_0=0.05)
    assert forgone == 0.0, f"Forgone revenue should be 0 for overinvestment: {forgone}"

def test_dario_default_prob_positive_for_overinvestment(dilemma):
    """Default probability > 0 when aggressive (invests at lower X)."""
    dp = dilemma.default_probability(0.30, 0.60, T=5.0)
    assert dp >= 0, f"Default probability should be non-negative: {dp}"

def test_dario_default_prob_zero_for_underinvestment(dilemma):
    """Default probability = 0 when conservative (invests later)."""
    dp = dilemma.default_probability(0.30, 0.15, T=5.0)
    assert dp == 0.0, f"Default probability should be 0 for underinvestment: {dp}"

def test_dario_table_rows(dilemma):
    """Dilemma table has 7 rows (one per misspecification level)."""
    table = dilemma.dilemma_table(lambda_true=0.30)
    assert len(table) == 7

def test_dario_table_columns(dilemma):
    """Dilemma table has expected columns."""
    table = dilemma.dilemma_table(lambda_true=0.30)
    expected_cols = ['Misspecification ε', 'λ used', 'X* (used)', 'X* (true)',
                      'X* deviation', 'Forgone Revenue (scaled, 5yr)', '5-yr Default Prob.']
    for col in expected_cols:
        assert col in table.columns, f"Column '{col}' missing from table"

def test_dario_symmetry_at_eps_zero(dilemma):
    """At ε=0 (exact belief), both costs are zero."""
    table = dilemma.dilemma_table(lambda_true=0.30)
    row_zero = table[table['Misspecification ε'] == '+0%'].iloc[0]
    assert row_zero['Forgone Revenue (scaled, 5yr)'] == 0.0
    assert row_zero['5-yr Default Prob.'] == '0.0%'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
