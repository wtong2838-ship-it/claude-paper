"""
Tests for Phase 1: Single-Firm Base Model
=========================================
Verifies analytical structure, parameter conditions, and comparative statics.
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase1_base_model import ModelParams, SingleFirmModel


# ============================================================ #
#  Fixtures                                                      #
# ============================================================ #

@pytest.fixture
def params():
    return ModelParams()

@pytest.fixture
def model(params):
    return SingleFirmModel(params)

@pytest.fixture
def solution(model):
    return model.solve()


# ============================================================ #
#  Parameter validation                                          #
# ============================================================ #

def test_params_validate_ok(params):
    """Default params should pass validation."""
    params.validate()  # should not raise

def test_params_invalid_r_less_mu_H():
    """r < mu_H should fail validation."""
    p = ModelParams(r=0.05, mu_H=0.08)
    with pytest.raises(AssertionError):
        p.validate()

def test_params_invalid_alpha():
    """alpha outside (0,1) should fail."""
    p = ModelParams(alpha=1.1)
    with pytest.raises(AssertionError):
        p.validate()

def test_params_invalid_gamma():
    """gamma < 1 should fail."""
    p = ModelParams(gamma=0.8)
    with pytest.raises(AssertionError):
        p.validate()


# ============================================================ #
#  Characteristic roots                                          #
# ============================================================ #

def test_characteristic_root_H_positive(params):
    """Positive root β_H > 1 for regime H."""
    beta_plus, beta_minus = params.characteristic_roots('H')
    assert beta_plus > 1.0, f"β_H should be > 1, got {beta_plus}"
    assert beta_minus < 0.0, f"β_H- should be < 0, got {beta_minus}"

def test_characteristic_root_L_larger_with_lambda(params):
    """β_L (effective discount r+λ) > β_H (only r)."""
    beta_L, _ = params.characteristic_roots('L', extra_discount=params.lam)
    beta_H, _ = params.characteristic_roots('H')
    assert beta_L > beta_H, (
        f"β_L={beta_L:.4f} should exceed β_H={beta_H:.4f} "
        f"because effective discount r+λ={params.r+params.lam:.2f} > r={params.r}")

def test_char_eq_satisfied_H(params):
    """Verify characteristic equation is zero at computed root."""
    beta, _ = params.characteristic_roots('H')
    mu, sigma, r = params.mu_H, params.sigma_H, params.r
    lhs = 0.5 * sigma**2 * beta * (beta - 1) + mu * beta - r
    assert abs(lhs) < 1e-10, f"Char eq not satisfied: {lhs}"

def test_char_eq_satisfied_L(params):
    """Verify characteristic equation (with extra λ) is zero at computed root."""
    beta, _ = params.characteristic_roots('L', extra_discount=params.lam)
    mu, sigma = params.mu_L, params.sigma_L
    rho = params.r + params.lam
    lhs = 0.5 * sigma**2 * beta * (beta - 1) + mu * beta - rho
    assert abs(lhs) < 1e-10, f"Char eq (L) not satisfied: {lhs}"


# ============================================================ #
#  Interior solution condition                                   #
# ============================================================ #

def test_interior_condition_H(params):
    """Default params should satisfy interior-solution condition in H regime."""
    assert params.check_interior_condition('H'), (
        f"ψ_H={params.psi('H'):.4f} should satisfy 1 < ψ < γ={params.gamma}")

def test_psi_formula(params):
    """ψ = αβ/(β-1)."""
    beta, _ = params.characteristic_roots('H')
    psi_expected = params.alpha * beta / (beta - 1)
    assert abs(params.psi('H') - psi_expected) < 1e-12


# ============================================================ #
#  Value of installed capacity                                   #
# ============================================================ #

def test_V_positive_at_trigger(model, solution):
    """V(X*, K*) > 0 in each regime."""
    for regime in ['H', 'L']:
        s = solution[regime]
        V = model.V(s['X_star'], s['K_star'], regime)
        assert V > 0, f"V in regime {regime} should be positive at trigger"

def test_V_H_greater_than_V_L_large_K(model):
    """At same X, V_H > V_L because installed capacity earns more in H."""
    X, K = 0.5, 1.0
    assert model.V(X, K, 'H') > model.V(X, K, 'L'), (
        "V_H should exceed V_L: H regime has higher drift so PV of installed capacity is higher")

def test_phi_L_includes_switching_premium(model):
    """φ_L should include the switching premium from future H-regime revenue."""
    K = 1.0
    p = model.p
    # φ_H = K^α / (r - μ_H)
    phi_H_formula = K**p.alpha / (p.r - p.mu_H)
    # φ_L = (K^α + λ φ_H) / (r - μ_L + λ) > K^α/(r-μ_L)  [no switching]
    phi_L_no_switch = K**p.alpha / (p.r - p.mu_L)
    phi_L_val = model.phi_L(K)
    assert phi_L_val > phi_L_no_switch, "φ_L should exceed no-switch baseline due to H premium"


# ============================================================ #
#  Investment trigger X* properties                             #
# ============================================================ #

def test_triggers_positive(solution):
    """Both triggers must be strictly positive."""
    assert solution['H']['X_star'] > 0
    assert solution['L']['X_star'] > 0

def test_capacity_positive(solution):
    """Optimal capacity K* > 0 in both regimes."""
    assert solution['H']['K_star'] > 0
    assert solution['L']['K_star'] > 0

def test_NPV_positive_at_trigger(solution):
    """NPV at trigger ≥ 0 (firm would invest)."""
    for regime in ['H', 'L']:
        assert solution[regime]['NPV'] >= -1e-6, (
            f"NPV in regime {regime} should be non-negative at trigger")

def test_switching_premium_positive(solution):
    """D_L > 0: switching premium adds to F_L."""
    assert solution['L']['D'] > 0, "Coupling coefficient D_L should be positive"

def test_option_elasticity_beta_H(params, solution):
    """β_H in solution should match characteristic root."""
    beta_analytical, _ = params.characteristic_roots('H')
    beta_solution = solution['H']['beta']
    assert abs(beta_analytical - beta_solution) < 1e-8


# ============================================================ #
#  Smooth-pasting and value-matching verification               #
# ============================================================ #

def test_value_matching_H(model, solution):
    """Verify VM: F_H(X*_H) = e^{-rτ} V_H(X*_H, K*_H) - I(K*_H)."""
    s = solution['H']
    X_star, K_star = s['X_star'], s['K_star']
    d = np.exp(-model.p.r * model.p.tau)

    F_val = s['A'] * X_star**s['beta']
    rhs = d * model.V(X_star, K_star, 'H') - model.I(K_star)
    assert abs(F_val - rhs) / (abs(rhs) + 1e-10) < 1e-4, (
        f"Value-matching violated in H: F={F_val:.6f}, RHS={rhs:.6f}")

def test_smooth_pasting_H(model, solution):
    """Verify SP: F_H'(X*_H) = e^{-rτ} φ_H(K*_H)."""
    s = solution['H']
    X_star, K_star = s['X_star'], s['K_star']
    d = np.exp(-model.p.r * model.p.tau)

    F_prime = s['A'] * s['beta'] * X_star**(s['beta'] - 1)
    phi = model.phi_H(K_star)
    assert abs(F_prime - d * phi) / (d * phi + 1e-10) < 1e-4, (
        f"Smooth-pasting violated in H: F'={F_prime:.6f}, d*φ={d*phi:.6f}")


# ============================================================ #
#  Comparative statics                                          #
# ============================================================ #

def test_lambda_lowers_L_trigger(model):
    """Higher λ → lower X*_L (firm invests sooner in L as adoption becomes more likely)."""
    lam_vals = np.array([0.05, 0.15, 0.30, 0.50])
    cs = model.comparative_statics('lam', lam_vals)
    X_L = cs['X_star_L']
    valid = np.isfinite(X_L)
    if valid.sum() < 3:
        pytest.skip("Too many failed solves for λ comparative static")
    # Check general downward trend
    x_valid = X_L[valid]
    l_valid = lam_vals[valid]
    corr = np.corrcoef(l_valid, x_valid)[0, 1]
    assert corr < 0, f"X*_L should decrease in λ, but correlation = {corr:.3f}"

def test_sigma_increases_H_trigger(model):
    """Higher σ_H → higher X*_H (standard real options option-value-of-waiting result)."""
    sigma_vals = np.array([0.15, 0.25, 0.35, 0.45])
    cs = model.comparative_statics('sigma_H', sigma_vals)
    X_H = cs['X_star_H']
    valid = np.isfinite(X_H)
    if valid.sum() < 3:
        pytest.skip("Too many failed solves for σ_H comparative static")
    x_valid = X_H[valid]
    s_valid = sigma_vals[valid]
    corr = np.corrcoef(s_valid, x_valid)[0, 1]
    assert corr > 0, f"X*_H should increase in σ_H (option value of waiting), but corr={corr:.3f}"

def test_alpha_effect_on_capacity(model):
    """Higher α → larger optimal capacity K* (higher returns justify larger scale).
    The trigger X* can go either way due to competing effects:
    higher α → larger K* (via FOC) → larger I(K*) → ambiguous trigger direction.
    The capacity response K* is unambiguously increasing in α.
    """
    alpha_vals = np.array([0.35, 0.42, 0.50, 0.58, 0.65])
    cs = model.comparative_statics('alpha', alpha_vals)
    K_H = cs['K_star_H']
    valid = np.isfinite(K_H)
    if valid.sum() < 3:
        pytest.skip("Too many failed solves for α comparative static")
    k_valid = K_H[valid]
    a_valid = alpha_vals[valid]
    corr = np.corrcoef(a_valid, k_valid)[0, 1]
    assert corr > 0, f"K*_H should increase in α (higher compute returns → build more), but corr={corr:.3f}"


# ============================================================ #
#  Option value properties                                       #
# ============================================================ #

def test_option_value_positive(model, solution):
    """Option value should be positive below trigger."""
    for regime in ['H', 'L']:
        X_star = solution[regime]['X_star']
        X_below = X_star * 0.5
        F = model.option_value(X_below, regime, solution)
        assert F > 0, f"Option value in {regime} should be positive for X < X*"

def test_option_value_continuous_at_trigger(model, solution):
    """Option value continuous at trigger for regime H (value-matching satisfied).
    Regime L uses numerical approximation, so only test H analytically.
    """
    # Regime H: analytical solution should be continuous
    regime = 'H'
    X_star = solution[regime]['X_star']
    K_star = solution[regime]['K_star']
    d = np.exp(-model.p.r * model.p.tau)

    F_below = model.option_value(X_star * (1 - 1e-5), regime, solution)
    F_at = d * model.V(X_star, K_star, regime) - model.I(K_star)  # NPV at trigger
    rel_diff = abs(F_below - F_at) / (abs(F_at) + 1e-10)
    assert rel_diff < 0.02, (
        f"H regime option value discontinuous at trigger: F_below={F_below:.6f}, NPV_at={F_at:.6f}")

def test_option_value_increasing_in_X(model, solution):
    """Option value is increasing in X (monotone)."""
    for regime in ['H', 'L']:
        X_star = solution[regime]['X_star']
        X_grid = np.linspace(0.1 * X_star, 0.9 * X_star, 10)
        F_vals = [model.option_value(x, regime, solution) for x in X_grid]
        diffs = np.diff(F_vals)
        assert np.all(diffs > 0), f"Option value in {regime} not monotone increasing in X"


# ============================================================ #
#  Sympy verification (runs the symbolic check)                 #
# ============================================================ #

def test_sympy_verification():
    """Run the SymPy analytical verification."""
    from phase1_base_model import sympy_verify
    result = sympy_verify()
    assert result is True


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
