"""
Tests for Phase 4: Revealed Beliefs Methodology
================================================
Tests the model inversion X*(K, λ) = X_obs for λ.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase4_revealed_beliefs import RevealedBeliefs, STYLIZED_FIRMS
from phase1_base_model import ModelParams


# ============================================================ #
#  Fixtures                                                      #
# ============================================================ #

@pytest.fixture
def params():
    return ModelParams()

@pytest.fixture
def rb(params):
    return RevealedBeliefs(params)


# ============================================================ #
#  Trigger Function X*(K, λ)                                    #
# ============================================================ #

def test_trigger_positive(rb):
    """X*(K, λ) > 0 for valid inputs."""
    for K in [0.10, 0.30, 0.50]:
        for lam in [0.10, 0.30, 0.60]:
            x = rb.trigger_at_lambda(K, lam, 'L')
            assert x > 0, f"X*(K={K}, λ={lam}) = {x} should be positive"

def test_trigger_decreasing_in_lambda(rb):
    """X*(K, λ) is decreasing in λ for L regime."""
    K = 0.20
    lambdas = [0.05, 0.15, 0.30, 0.50, 0.80]
    triggers = [rb.trigger_at_lambda(K, lam, 'L') for lam in lambdas]
    diffs = np.diff(triggers)
    assert np.all(diffs < 0), \
        f"Trigger should decrease monotonically in λ, got diffs: {diffs}"

def test_trigger_increasing_in_K(rb):
    """X*(K, λ) is increasing in K (larger investment needs higher trigger)."""
    lam = 0.20
    K_vals = [0.10, 0.20, 0.40, 0.60]
    triggers = [rb.trigger_at_lambda(K, lam, 'L') for K in K_vals]
    diffs = np.diff(triggers)
    assert np.all(diffs > 0), \
        f"Trigger should increase in K, got diffs: {diffs}"

def test_trigger_H_regime_lambda_independent(rb):
    """X*(K, λ) in H regime is independent of λ (H is absorbing)."""
    K = 0.20
    triggers_H = [rb.trigger_at_lambda(K, lam, 'H') for lam in [0.10, 0.30, 0.60]]
    # All should be equal (within numerical tolerance)
    assert max(triggers_H) - min(triggers_H) < 1e-10, \
        f"H-regime trigger should be λ-independent: {triggers_H}"


# ============================================================ #
#  Point Estimate: _point_estimate                              #
# ============================================================ #

def test_point_estimate_recovers_lambda(rb):
    """Inversion should recover the true λ from model-generated data."""
    for K in [0.10, 0.20, 0.40]:
        for lam_true in [0.10, 0.25, 0.50]:
            # Generate X_obs from model
            X_obs = rb.trigger_at_lambda(K, lam_true, 'L')
            # Recover lambda
            lam_hat = rb._point_estimate(K, X_obs, 'L')
            assert abs(lam_hat - lam_true) < 1e-5, \
                f"Expected λ={lam_true:.3f}, got λ̂={lam_hat:.3f} for K={K}"

def test_point_estimate_bounds(rb):
    """λ̂ should be within [lam_lo, lam_hi]."""
    K, X_obs = 0.20, 0.039
    lam_hat = rb._point_estimate(K, X_obs, 'L', lam_bounds=(0.01, 0.99))
    assert 0.01 <= lam_hat <= 0.99, f"λ̂={lam_hat} out of bounds"

def test_point_estimate_low_X_gives_high_lambda(rb):
    """Low X_obs (invested early) → high revealed λ."""
    K = 0.20
    X_low = rb.trigger_at_lambda(K, 0.70, 'L')   # aggressive
    X_high = rb.trigger_at_lambda(K, 0.10, 'L')  # conservative
    lam_from_low = rb._point_estimate(K, X_low, 'L')
    lam_from_high = rb._point_estimate(K, X_high, 'L')
    assert lam_from_low > lam_from_high, \
        f"Low X_obs should give high λ̂: {lam_from_low:.3f} vs {lam_from_high:.3f}"


# ============================================================ #
#  Full infer_lambda                                            #
# ============================================================ #

def test_infer_lambda_returns_dict(rb):
    """infer_lambda returns a dict with expected keys."""
    result = rb.infer_lambda(0.20, 0.039, 'L')
    for key in ['lambda_hat', 'CI_low', 'CI_high', 'interpretation', 'X_obs', 'K_obs']:
        assert key in result, f"Key '{key}' missing from result"

def test_infer_lambda_hat_in_bounds(rb):
    """λ̂ is between CI_low and CI_high, and both in [0, 1]."""
    result = rb.infer_lambda(0.20, 0.039, 'L')
    assert 0 <= result['CI_low'] <= result['lambda_hat'] <= result['CI_high'] <= 1.0, \
        f"CI ordering violated: {result['CI_low']:.3f} ≤ {result['lambda_hat']:.3f} ≤ {result['CI_high']:.3f}"

def test_infer_lambda_interpretation_string(rb):
    """Interpretation field is a non-empty string."""
    result = rb.infer_lambda(0.20, 0.039, 'L')
    assert isinstance(result['interpretation'], str)
    assert len(result['interpretation']) > 0


# ============================================================ #
#  Cross-Firm Application                                        #
# ============================================================ #

def test_apply_to_all_firms_rows(rb):
    """apply_to_all_firms returns one row per firm."""
    df = rb.apply_to_all_firms()
    assert len(df) == len(STYLIZED_FIRMS)

def test_apply_to_all_firms_lambda_in_range(rb):
    """All revealed λ̂ in [0.01, 0.99]."""
    df = rb.apply_to_all_firms()
    for _, row in df.iterrows():
        lam = row['λ̂ (central)']
        assert 0.01 <= lam <= 0.99, f"λ̂={lam} for {row['Firm']} out of range"

def test_aggressive_firm_higher_lambda(rb):
    """Aggressive startup (Anthropic) has higher λ̂ than conservative incumbent."""
    df = rb.apply_to_all_firms()
    df_indexed = df.set_index('Firm')
    lam_anthropic = df_indexed.loc['Anthropic', 'λ̂ (central)']
    lam_msft = df_indexed.loc['Microsoft Azure AI', 'λ̂ (central)']
    assert lam_anthropic > lam_msft, \
        f"Anthropic λ̂={lam_anthropic:.3f} should exceed MSFT λ̂={lam_msft:.3f}"


# ============================================================ #
#  Time Series                                                  #
# ============================================================ #

def test_lambda_time_series_keys(rb):
    """Time series has entries for each year."""
    ts = rb.lambda_time_series()
    assert set(ts.keys()) == {'2022', '2023', '2024', '2025'}

def test_lambda_time_series_rising(rb):
    """Industry λ̂ should be rising over 2022→2025 (increasing confidence in AI adoption)."""
    ts = rb.lambda_time_series()
    lams = [ts[y]['lambda_hat'] for y in sorted(ts.keys())]
    # Allow some non-monotonicity but trend should be upward
    assert lams[-1] > lams[0], \
        f"λ̂ should rise from 2022 to 2025: {lams}"


# ============================================================ #
#  Sensitivity Analysis                                         #
# ============================================================ #

def test_sensitivity_analysis_baseline_scenario(rb):
    """Baseline scenario is included."""
    df = rb.sensitivity_analysis(0.20, 0.039, 'L')
    scenarios = df['Scenario'].tolist()
    assert 'Baseline' in scenarios

def test_sensitivity_analysis_lambda_range(rb):
    """All sensitivity λ̂ values are in [0.01, 0.99]."""
    df = rb.sensitivity_analysis(0.20, 0.039, 'L')
    for _, row in df.iterrows():
        lam = row['λ̂']
        if not np.isnan(lam):
            assert 0.01 <= lam <= 0.99, f"Sensitivity λ̂={lam} out of range"

def test_sensitivity_high_alpha_higher_lambda(rb):
    """Higher α → more sensitive to revenue → firm can invest at lower X → higher λ̂."""
    K, X_obs = 0.20, 0.039
    p_hi = ModelParams(alpha=rb.base_params.alpha * 1.5)
    p_lo = ModelParams(alpha=rb.base_params.alpha * 0.5)
    lam_hi = RevealedBeliefs(p_hi)._point_estimate(K, X_obs, 'L')
    lam_lo = RevealedBeliefs(p_lo)._point_estimate(K, X_obs, 'L')
    # Higher alpha makes the investment more valuable → lower threshold → higher λ needed
    assert lam_hi > lam_lo, \
        f"High α should give higher λ̂: {lam_hi:.3f} vs {lam_lo:.3f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
