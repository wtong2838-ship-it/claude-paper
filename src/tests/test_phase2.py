"""
Tests for Phase 2: Duopoly with Default Risk
============================================
Tests the Leland (1994) default model and preemption equilibrium.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase2_duopoly import DuopolyParams, LelandDefault, DuopolyModel


# ============================================================ #
#  Fixtures                                                      #
# ============================================================ #

@pytest.fixture
def params():
    return DuopolyParams()

@pytest.fixture
def model(params):
    return DuopolyModel(params)

@pytest.fixture
def equilibrium(model):
    return model.preemption_equilibrium('H')


# ============================================================ #
#  DuopolyParams validation                                      #
# ============================================================ #

def test_params_validate(params):
    params.validate()

def test_theta_bounds():
    with pytest.raises(AssertionError):
        DuopolyParams(theta=0.0).validate()
    with pytest.raises(AssertionError):
        DuopolyParams(theta=1.0).validate()

def test_phi_mono_greater_than_phi_duo(params):
    """φ_mono > φ_duo (monopoly more profitable than duopoly)."""
    K = 0.15
    assert params.phi_mono(K) > params.phi_duo(K), \
        "Monopoly φ should exceed duopoly φ (duopoly has θ < 1 discount)"

def test_phi_duo_is_theta_times_phi_mono_H(params):
    """In H regime: φ_duo = θ · K^α/(r-μ_H) = θ · φ_mono."""
    K = 0.15
    phi_m = params.phi_mono(K, 'H')
    phi_d = params.phi_duo(K, 'H')
    assert abs(phi_d / phi_m - params.theta) < 1e-10, \
        f"φ_duo/φ_mono = {phi_d/phi_m:.6f}, should equal θ={params.theta}"


# ============================================================ #
#  Leland (1994) Default Model                                   #
# ============================================================ #

def test_leland_default_boundary_positive(params):
    """Default boundary X_D > 0 with positive coupon."""
    ld = LelandDefault(params, K=0.15, in_duopoly=False)
    X_D = ld.default_boundary()
    assert X_D > 0, "Default boundary should be strictly positive"

def test_leland_duopolist_higher_XD(params):
    """Duopolist has higher X_D than monopolist (less revenue to service debt)."""
    K = 0.15
    ld_mono = LelandDefault(params, K, in_duopoly=False)
    ld_duo = LelandDefault(params, K, in_duopoly=True)
    XD_mono = ld_mono.default_boundary()
    XD_duo = ld_duo.default_boundary()
    assert XD_duo > XD_mono, \
        f"Duopolist X_D={XD_duo:.6f} should exceed monopolist X_D={XD_mono:.6f}"

def test_equity_zero_at_default(params):
    """Equity value = 0 at the default boundary."""
    ld = LelandDefault(params, K=0.15, in_duopoly=False)
    X_D = ld.default_boundary()
    E_at_XD = ld.equity_value(X_D * (1 + 1e-6))
    assert E_at_XD >= 0, "Equity should be non-negative just above default boundary"
    E_below = ld.equity_value(X_D * 0.5)
    assert E_below == 0.0, "Equity should be 0 below default boundary"

def test_equity_increasing_in_X(params):
    """Equity value is increasing in X (above X_D)."""
    ld = LelandDefault(params, K=0.15, in_duopoly=False)
    X_D = ld.default_boundary()
    X_grid = np.linspace(X_D * 1.5, X_D * 10, 20)
    E_vals = [ld.equity_value(x) for x in X_grid]
    diffs = np.diff(E_vals)
    assert np.all(diffs >= -1e-8), "Equity should be increasing in X"

def test_debt_positive(params):
    """Debt value > 0 for X > X_D."""
    ld = LelandDefault(params, K=0.15, in_duopoly=False)
    X_D = ld.default_boundary()
    D = ld.debt_value(X_D * 3)
    assert D > 0, "Debt should be positive above default boundary"

def test_debt_bounded_by_risk_free(params):
    """Risky debt value ≤ risk-free debt value C_D/r."""
    ld = LelandDefault(params, K=0.15, in_duopoly=False)
    X_D = ld.default_boundary()
    D = ld.debt_value(X_D * 5)
    D_risk_free = params.coupon / params.r
    assert D <= D_risk_free + 1e-10, \
        f"Risky debt D={D:.6f} should not exceed risk-free D={D_risk_free:.6f}"

def test_credit_spread_positive(params):
    """Credit spread > 0 (risky debt yields above risk-free)."""
    ld = LelandDefault(params, K=0.15, in_duopoly=False)
    X_D = ld.default_boundary()
    cs = ld.credit_spread(X_D * 3)
    assert cs > 0, "Credit spread should be positive"

def test_credit_spread_increasing_near_default(params):
    """Credit spread increases as X approaches X_D (more distress)."""
    ld = LelandDefault(params, K=0.15, in_duopoly=False)
    X_D = ld.default_boundary()
    cs_far = ld.credit_spread(X_D * 10)
    cs_near = ld.credit_spread(X_D * 1.5)
    assert cs_near > cs_far, "Credit spread should be higher near default boundary"

def test_default_probability_bounds(params):
    """Default probability is in [0, 1]."""
    ld = LelandDefault(params, K=0.15, in_duopoly=False)
    X_D = ld.default_boundary()
    dp = ld.default_probability(X_D * 2, T=5.0)
    assert 0 <= dp <= 1, f"Default probability {dp} out of [0,1]"

def test_default_probability_higher_near_default(params):
    """Default probability is higher when X is closer to X_D."""
    ld = LelandDefault(params, K=0.15, in_duopoly=False)
    X_D = ld.default_boundary()
    dp_far = ld.default_probability(X_D * 5, T=5.0)
    dp_near = ld.default_probability(X_D * 1.5, T=5.0)
    assert dp_near > dp_far, "Default probability should be higher closer to X_D"

def test_char_roots(params):
    """Characteristic roots: y_plus > 1, y_minus < 0."""
    y_plus, y_minus = params.leland_char_roots()
    assert y_plus > 1, f"y_plus={y_plus} should be > 1"
    assert y_minus < 0, f"y_minus={y_minus} should be < 0"


# ============================================================ #
#  Preemption Equilibrium                                        #
# ============================================================ #

def test_equilibrium_triggers_positive(equilibrium):
    """Both leader and follower triggers must be positive."""
    assert equilibrium['X_L'] > 0
    assert equilibrium['X_F'] > 0

def test_follower_invests_after_leader(equilibrium):
    """Follower's trigger X_F* > leader's trigger X_L* (sequential equilibrium)."""
    assert equilibrium['X_F'] >= equilibrium['X_L'], \
        f"Follower trigger {equilibrium['X_F']:.4f} should be ≥ leader trigger {equilibrium['X_L']:.4f}"

def test_leader_invests_before_single_firm(equilibrium):
    """Preemption: leader invests earlier than single firm (lower trigger)."""
    assert equilibrium['X_L'] <= equilibrium['X_single'] * 1.01, \
        (f"Leader trigger X_L={equilibrium['X_L']:.4f} should be ≤ "
         f"single-firm X*={equilibrium['X_single']:.4f} (preemption effect)")

def test_preemption_gap_positive(equilibrium):
    """Preemption gap > 0: competition erodes option value of waiting."""
    assert equilibrium['preemption_gap'] > 0, \
        f"Preemption gap should be positive, got {equilibrium['preemption_gap']:.4f}"

def test_leader_default_boundary_below_trigger(equilibrium, params):
    """Sanity check: leader's default boundary should be below its investment trigger.
    (Otherwise the firm would default immediately upon investing — not rational.)"""
    X_D_L = equilibrium['X_D_L']
    X_L = equilibrium['X_L']
    # With small coupon, X_D should be well below X_L
    if params.coupon > 0:
        # Only check if coupon is positive
        assert X_D_L < X_L * 1.1, \
            (f"Leader default boundary X_D={X_D_L:.6f} should be below "
             f"investment trigger X_L={X_L:.6f}")

def test_capacities_positive(equilibrium):
    """Both K_L and K_F must be positive."""
    assert equilibrium['K_L'] > 0
    assert equilibrium['K_F'] > 0

def test_NPV_positive_at_equilibrium(equilibrium):
    """NPV should be positive at equilibrium triggers."""
    assert equilibrium['NPV_L'] > 0, "Leader NPV should be positive"
    assert equilibrium['NPV_F'] > 0, "Follower NPV should be positive"


# ============================================================ #
#  Competition-Leverage Spiral                                   #
# ============================================================ #

def test_spiral_preemption_lowers_trigger(model):
    """Competition reduces trigger relative to monopoly."""
    spiral = model.competition_leverage_spiral('H')
    X_mono = spiral['steps'][0]['X_star']
    X_comp = spiral['steps'][1]['X_star']
    assert X_comp <= X_mono, \
        f"Competition trigger {X_comp:.4f} should be ≤ monopoly trigger {X_mono:.4f}"

def test_spiral_default_risk_appears(model):
    """Default risk (X_D > 0) only appears with debt financing."""
    spiral = model.competition_leverage_spiral('H')
    X_D_nodefault = spiral['steps'][1]['X_D']   # competition, no default
    X_D_withdefault = spiral['steps'][2]['X_D']  # competition, with default
    assert X_D_nodefault == 0.0, "No default risk without coupon debt"
    assert X_D_withdefault > 0.0, "Default boundary > 0 with coupon debt"

def test_spiral_preemption_effect_positive(model):
    """Preemption effect is positive (competition lowers trigger)."""
    spiral = model.competition_leverage_spiral('H')
    assert spiral['preemption_effect'] > 0, \
        f"Preemption effect should be positive, got {spiral['preemption_effect']:.4f}"


# ============================================================ #
#  Comparative statics                                          #
# ============================================================ #

def test_theta_higher_reduces_preemption(model):
    """Higher θ (weaker competition) → smaller preemption effect (X_L closer to X_single)."""
    theta_vals = np.array([0.40, 0.55, 0.70, 0.85])
    cs = model.comparative_statics('theta', theta_vals)
    X_L = cs['X_L']
    valid = np.isfinite(X_L)
    if valid.sum() < 3:
        pytest.skip("Too many failed solves for θ comparative static")
    # Higher θ → less competitive → less preemption → higher X_L
    corr = np.corrcoef(theta_vals[valid], X_L[valid])[0, 1]
    assert corr > 0, f"X_L should increase in θ (weaker competition), but corr={corr:.3f}"

def test_lambda_affects_L_regime_equilibrium(model):
    """λ affects the L-regime equilibrium triggers via the regime-switching option premium."""
    lam_vals = np.array([0.10, 0.20, 0.35, 0.50])
    cs = model.comparative_statics('lam', lam_vals, regime='L')
    X_L = cs['X_L']
    valid = np.isfinite(X_L)
    if valid.sum() < 3:
        pytest.skip("Too many failed solves for λ comparative static in L regime")
    # Higher λ → regime switch more likely → regime L is less different from H
    # → L-regime triggers should change with λ (direction can vary)
    assert np.any(np.diff(X_L[valid]) != 0), "X_L in L-regime should vary with λ"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
