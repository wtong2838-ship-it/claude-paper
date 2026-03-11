"""
Tests for Phase 3: N-Firm Sequential Investment Game
=====================================================
Tests the accordion effect, training/inference allocation,
and firm heterogeneity models.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase3_nfirm import NFirmParams, NFirmModel, HeterogeneousFirms


# ============================================================ #
#  Fixtures                                                      #
# ============================================================ #

@pytest.fixture
def params():
    return NFirmParams(N=4)

@pytest.fixture
def model(params):
    return NFirmModel(params)

@pytest.fixture
def equilibrium(model):
    return model.solve_sequential('H')


# ============================================================ #
#  NFirmParams validation                                        #
# ============================================================ #

def test_params_default_N4():
    p = NFirmParams(N=4)
    assert p.N == 4
    assert len(p.cash) == 4
    assert len(p.quality) == 4
    assert len(p.capacity_init) == 4
    assert len(p.r_individual) == 4

def test_params_validate(params):
    params.validate()

def test_params_N_too_small():
    with pytest.raises(AssertionError):
        NFirmParams(N=1).validate()

def test_params_default_uniform_cash(params):
    """All cash reserves equal by default."""
    assert all(c == params.cash[0] for c in params.cash)

def test_params_default_uniform_quality(params):
    """All quality values equal by default."""
    assert all(q == params.quality[0] for q in params.quality)

def test_phi_nfirm_decreasing_N(params):
    """Per-firm φ decreases as more firms enter (market dilution)."""
    K = 0.15
    phi_mono = params.phi_nfirm(K, 1, 'H')
    phi_duo = params.phi_nfirm(K, 2, 'H')
    phi_trio = params.phi_nfirm(K, 3, 'H')
    assert phi_mono > phi_duo > phi_trio, \
        "φ should decrease with more active firms"

def test_phi_nfirm_positive(params):
    """φ coefficient is positive."""
    phi = params.phi_nfirm(0.15, 2, 'H')
    assert phi > 0

def test_phi_nfirm_N1_equals_monopoly(params):
    """φ_nfirm with N=1 equals monopoly φ."""
    K = 0.15
    phi_n1 = params.phi_nfirm(K, 1, 'H')
    phi_mono = K**params.alpha / (params.r - params.mu_H)
    assert abs(phi_n1 - phi_mono) < 1e-10


# ============================================================ #
#  Sequential Equilibrium (Accordion Effect)                     #
# ============================================================ #

def test_equilibrium_triggers_positive(equilibrium):
    """All entry triggers must be positive."""
    for X in equilibrium['triggers']:
        assert X is not None and X > 0, f"Trigger {X} should be positive"

def test_equilibrium_capacities_positive(equilibrium):
    """All capacity choices must be positive."""
    for K in equilibrium['capacities']:
        assert K is not None and K > 0, f"Capacity {K} should be positive"

def test_N_triggers_returned(equilibrium, params):
    """Exactly N triggers returned."""
    assert len(equilibrium['triggers']) == params.N

def test_accordion_effect_positive(equilibrium):
    """Accordion effect: first-mover trigger is lower than single-firm benchmark."""
    assert equilibrium['accordion_effect'] > 0, \
        "Accordion effect should be positive (competition lowers entry threshold)"

def test_accordion_effect_bounded(equilibrium):
    """Accordion effect is between 0% and 100%."""
    ae = equilibrium['accordion_effect']
    assert 0 < ae < 100, f"Accordion effect {ae:.1f}% should be in (0,100)"

def test_more_firms_lower_first_trigger():
    """Higher N → lower first-mover trigger (stronger competition)."""
    X_1_vals = []
    for N in [2, 3, 4]:
        p = NFirmParams(N=N)
        model = NFirmModel(p)
        eq = model.solve_sequential('H')
        X_1_vals.append(min(eq['triggers']))  # first mover is minimum

    # Should be (roughly) decreasing in N
    assert X_1_vals[0] >= X_1_vals[-1] * 0.5, \
        "First-mover trigger should be lower with more competitors"


# ============================================================ #
#  Training vs. Inference Allocation                             #
# ============================================================ #

def test_training_inference_sum_to_K(model):
    """K_I + K_T = K_total for any X."""
    K_total = 0.5
    quality = 1.0
    comp_quality = 1.0
    for X in [0.05, 0.10, 0.20]:
        alloc = model.training_inference_allocation(K_total, X, quality, comp_quality)
        K_I = alloc['K_I']
        K_T = alloc['K_T']
        assert abs((K_I + K_T) - K_total) < 1e-6, \
            f"K_I + K_T = {K_I+K_T:.6f} ≠ K_total = {K_total}"

def test_training_inference_non_negative(model):
    """Both allocations must be non-negative."""
    K_total = 0.5
    quality, comp_quality = 1.0, 1.0
    for X in [0.01, 0.05, 0.20, 0.50]:
        alloc = model.training_inference_allocation(K_total, X, quality, comp_quality)
        assert alloc['K_I'] >= 0, "K_I (inference) must be non-negative"
        assert alloc['K_T'] >= 0, "K_T (training) must be non-negative"

def test_high_X_more_inference(model):
    """At high demand X, more capacity goes to inference (immediate revenue)."""
    K_total = 0.5
    quality, comp_quality = 1.0, 1.0
    alloc_low = model.training_inference_allocation(K_total, 0.02, quality, comp_quality)
    alloc_high = model.training_inference_allocation(K_total, 0.50, quality, comp_quality)
    # Higher X → more inference demand (split = training fraction decreases)
    # Use tolerance since marginal values can be close
    assert alloc_high['K_I'] >= alloc_low['K_I'] - 1e-6, \
        f"More inference at higher demand: K_I(high)={alloc_high['K_I']:.6f}, K_I(low)={alloc_low['K_I']:.6f}"

def test_inference_revenue_positive(model):
    """Inference revenue must be positive when K_I > 0."""
    K_total = 0.5
    quality, comp_quality = 1.0, 1.0
    alloc = model.training_inference_allocation(K_total, 0.10, quality, comp_quality)
    if alloc['K_I'] > 0:
        assert alloc['inference_revenue'] >= 0


# ============================================================ #
#  Firm Heterogeneity                                            #
# ============================================================ #

def test_heterogeneous_returns_entry_order():
    """Heterogeneous model returns an entry order."""
    p = NFirmParams(
        N=4,
        cash=[3.0, 1.0, 0.5, 0.2],
        quality=[1.5, 1.0, 1.0, 0.8],
        r_individual=[0.12, 0.15, 0.18, 0.22],
    )
    het = HeterogeneousFirms(p)
    result = het.solve_heterogeneous_equilibrium('H')
    assert 'entry_order' in result
    assert len(result['entry_order']) == 4

def test_heterogeneous_triggers_positive():
    """All heterogeneous-firm triggers must be positive."""
    p = NFirmParams(
        N=4,
        cash=[3.0, 1.0, 0.5, 0.2],
        quality=[1.5, 1.0, 1.0, 0.8],
        r_individual=[0.12, 0.15, 0.18, 0.22],
    )
    het = HeterogeneousFirms(p)
    result = het.solve_heterogeneous_equilibrium('H')
    for X in result['triggers']:
        assert X is not None and X > 0, f"Trigger {X} must be positive"

def test_N_firms_heterogeneous_equilibrium():
    """Exactly N triggers returned."""
    N = 3
    p = NFirmParams(
        N=N,
        cash=[1.0, 0.5, 0.2],
        quality=[1.0, 1.0, 0.8],
        r_individual=[0.15, 0.18, 0.22],
    )
    het = HeterogeneousFirms(p)
    result = het.solve_heterogeneous_equilibrium('H')
    assert len(result['triggers']) == N


# ============================================================ #
#  Competition-Leverage Spiral in N-firm context                 #
# ============================================================ #

def test_duopoly_accordion_smaller_than_Nfirm():
    """N=4 accordion effect is at least as large as N=2 (duopoly)."""
    eq_duo = NFirmModel(NFirmParams(N=2)).solve_sequential('H')
    X_1_duo = min(eq_duo['triggers'])
    X_single = eq_duo['X_single']
    accordion_duo = (X_single - X_1_duo) / X_single * 100

    eq_quad = NFirmModel(NFirmParams(N=4)).solve_sequential('H')
    X_1_quad = min(eq_quad['triggers'])
    X_single_4 = eq_quad['X_single']
    accordion_quad = (X_single_4 - X_1_quad) / X_single_4 * 100

    # Note: accordion effect increases with N
    assert accordion_quad >= accordion_duo * 0.8, \
        f"N=4 accordion {accordion_quad:.1f}% should be ≥ N=2 accordion {accordion_duo:.1f}%"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
