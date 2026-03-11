"""
Phase 5: Valuation Analysis
=============================

Compute AI firm value decomposition and quantify the 'Dario dilemma':
the asymmetric costs of underinvestment vs. overinvestment in compute.

Components:
1. Firm value decomposition: assets-in-place + expansion option + regime-switch premium
2. Equity sensitivity to λ (the "AGI premium")
3. Credit spread curves and capital structure analysis
4. Dario dilemma quantification: forgone revenue vs. default probability
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from phase1_base_model import ModelParams, SingleFirmModel
from phase2_duopoly import DuopolyParams, LelandDefault, DuopolyModel


# --------------------------------------------------------------------------- #
#  Firm Value Decomposition                                                     #
# --------------------------------------------------------------------------- #

class AIFirmValuation:
    """
    Value decomposition following Berk, Green & Naik (1999):
      V_total = V_AIP (assets in place) + V_expand (expansion option) + V_λ (AGI premium)

    The AGI premium V_λ is the incremental option value from the regime switch:
      V_λ = V(with switching) - V(no switching, λ=0)

    This premium is what makes AI firm valuations sensitive to timeline beliefs.
    """

    def __init__(self, params: DuopolyParams):
        self.p = params

    def assets_in_place(self, X: float, K: float, regime: str = 'H') -> float:
        """V_AIP = present value of current capacity revenues."""
        p = self.p
        phi = K**p.alpha / (p.r - p.mu_H)   # H regime (once adopted)
        return phi * X - p.delta * K / p.r

    def firm_value_with_switching(self, X: float, K: float, regime: str = 'L') -> float:
        """Total firm value including switching premium."""
        import copy
        p_copy = copy.copy(self.p)
        model = SingleFirmModel(p_copy)
        sol = model.solve()
        d = np.exp(-p_copy.r * p_copy.tau)

        if X >= sol[regime]['X_star']:
            # Invested: value = V(X, K)
            phi_H = K**p_copy.alpha / (p_copy.r - p_copy.mu_H)
            phi_L_num = K**p_copy.alpha + p_copy.lam * phi_H
            phi_L = phi_L_num / (p_copy.r - p_copy.mu_L + p_copy.lam)
            phi = phi_H if regime == 'H' else phi_L
            return phi * X - p_copy.delta * K / p_copy.r
        else:
            return model.option_value(X, regime, sol)

    def agr_premium(self, X: float, K: float, regime: str = 'L') -> float:
        """
        AGI (regime-switch) premium = V(with λ) - V(without λ).
        """
        import copy

        # Value with switching (current λ)
        V_with = self.firm_value_with_switching(X, K, regime)

        # Value without switching (λ → 0)
        p_no_switch = copy.copy(self.p)
        p_no_switch.lam = 0.001   # nearly zero
        model_ns = SingleFirmModel(p_no_switch)
        sol_ns = model_ns.solve()
        d = np.exp(-p_no_switch.r * p_no_switch.tau)

        if X >= sol_ns[regime]['X_star']:
            phi_H = K**p_no_switch.alpha / (p_no_switch.r - p_no_switch.mu_H)
            phi_L = K**p_no_switch.alpha / (p_no_switch.r - p_no_switch.mu_L)
            phi = phi_H if regime == 'H' else phi_L
            V_no = phi * X - p_no_switch.delta * K / p_no_switch.r
        else:
            V_no = model_ns.option_value(X, regime, sol_ns)

        return max(0.0, V_with - V_no)

    def growth_option_decomposition(self, X: float, K: float, K_future: float,
                                     regime: str = 'L') -> dict:
        """
        Decompose total firm value into:
        - V_AIP: assets in place (current installed capacity value)
        - V_expand: option to expand from K to K_future
        - V_λ: AGI premium (incremental value from regime switch)
        """
        # V_AIP: value of current capacity
        V_AIP = self.assets_in_place(X, K, regime)

        # V_total with switching
        V_total = self.firm_value_with_switching(X, K, regime)

        # V_expand ≈ option value with K_future capacity potential
        # Simplified: marginal value of capacity expansion
        V_AIP_future = self.assets_in_place(X, K_future, regime)
        V_expand = max(0.0, V_AIP_future - V_AIP - self.p.c * (K_future**self.p.gamma - K**self.p.gamma))

        # V_λ: AGI premium
        V_lambda = self.agr_premium(X, K, regime)

        # Residual: total - (AIP + expand + lambda)
        V_total_check = V_AIP + V_expand + V_lambda

        return {
            'V_AIP': V_AIP,
            'V_expand': V_expand,
            'V_lambda': V_lambda,
            'V_total': V_total,
            'V_total_check': V_total_check,
            'AIP_fraction': V_AIP / max(V_total, 1e-10),
            'expand_fraction': V_expand / max(V_total, 1e-10),
            'lambda_fraction': V_lambda / max(V_total, 1e-10),
        }

    def equity_sensitivity_to_lambda(self, X: float, K: float,
                                      lambda_range: np.ndarray,
                                      coupon: float = None) -> np.ndarray:
        """
        Equity value as a function of λ (the AGI timeline sensitivity).
        Higher λ → higher equity value for growth-option-intensive firms.
        """
        import copy
        equity_vals = []
        coupon = coupon if coupon is not None else self.p.coupon

        for lam in lambda_range:
            p_new = copy.copy(self.p)
            p_new.lam = lam
            try:
                p_new.validate()
                ld = LelandDefault(p_new, K, 'H', in_duopoly=False)
                E = ld.equity_value(X)
                equity_vals.append(E)
            except Exception:
                equity_vals.append(np.nan)

        return np.array(equity_vals)

    def credit_spread_curve(self, X: float, K: float,
                             coupon_range: np.ndarray,
                             in_duopoly: bool = False) -> np.ndarray:
        """Credit spreads as function of coupon level."""
        import copy
        spreads = []
        for coupon in coupon_range:
            p_new = copy.copy(self.p)
            p_new.coupon = coupon
            ld = LelandDefault(p_new, K, 'H', in_duopoly=in_duopoly)
            cs = ld.credit_spread(X)
            spreads.append(min(cs * 10000, 5000))  # in bps, capped
        return np.array(spreads)


# --------------------------------------------------------------------------- #
#  Dario Dilemma                                                                #
# --------------------------------------------------------------------------- #

class DarioDilemma:
    """
    Quantify the asymmetric cost of misspecifying λ.

    Key insight: K* is independent of λ for this model (∂φ_s/∂K / φ_s = α/K in
    both regimes). The λ misspecification affects the TRIGGER X*, not the scale K*.

    Scenario:
    - True λ = λ_true → optimal trigger X*_true (invest when X hits this level)
    - Conservative: uses λ_conserv < λ_true → X*_conserv > X*_true (waits too long)
    - Aggressive: uses λ_aggress > λ_true → X*_aggress < X*_true (invests too early)

    Costs:
    - Underinvestment: forgone operating profits during delay [X*_true → X*_conserv]
    - Overinvestment: elevated default risk when investing at low X (close to X_D)
    """

    def __init__(self, params: DuopolyParams, coupon_override: float = None):
        import copy
        self.p = copy.copy(params)
        # Use a lower coupon for L-regime analysis so X_D < X*_L
        # (baseline coupon=0.005 gives X_D > X*_L which implies immediate default)
        if coupon_override is not None:
            self.p.coupon = coupon_override
        elif self.p.coupon >= 0.003:
            self.p.coupon = 0.002   # ensure X_D < X*_L for all λ in [0.10, 0.60]

    def optimal_policy(self, lam_used: float, regime: str = 'L') -> tuple:
        """K*, X*(λ_used) — optimal policy for a given λ belief."""
        import copy
        p_new = copy.copy(self.p)
        p_new.lam = lam_used
        try:
            p_new.validate()
            model = SingleFirmModel(p_new)
            sol = model.solve()
            return sol[regime]['K_star'], sol[regime]['X_star']
        except Exception:
            return self.p.c, 1.0  # fallback

    # Keep backward-compatible alias
    def optimal_K(self, lam_used: float, regime: str = 'L') -> tuple:
        return self.optimal_policy(lam_used, regime)

    def forgone_revenue(self, lambda_true: float, lambda_conservative: float,
                         T: float = 5.0, X_0: float = 0.10) -> float:
        """
        Expected forgone revenue from investing too late (conservative λ).

        Conservative firm waits until X*_conserv > X*_true. During [X*_true, X*_conserv]
        the firm is not yet producing. We estimate forgone NPV as:

          Forgone ≈ (X*_conserv - X*_true) · K*^α · φ_H   [≈ value of delay]

        where φ_H = 1/(r - μ_H) is the perpetuity discount factor.
        This is the PV of the wedge in investment timing.
        """
        K_star, X_true = self.optimal_policy(lambda_true)
        _, X_conserv = self.optimal_policy(lambda_conservative)

        if X_conserv <= X_true:
            return 0.0   # not underinvesting (more aggressive)

        # Revenue each period = X · K^α; forgone during [X_true, X_conserv] window
        # Approximate: demand was at X_true when should have invested; delay ≈ Δ periods
        X_mid = 0.5 * (X_true + X_conserv)  # midpoint demand during delay
        K_alpha = K_star**self.p.alpha

        # PV of forgone revenue: K^α · X_mid / (r - μ_H) · (1 - e^{-(r-μ_H)T})
        # where T is the expected delay duration
        rho = self.p.r - self.p.mu_H
        # Expected delay ≈ (X_conserv - X_true) / (μ_H · X_true) years (GBM first-passage estimate)
        mu_eff = max(self.p.mu_L, 0.01)  # use L regime drift (firm still in L)
        expected_delay = (np.log(X_conserv / X_true)) / mu_eff if mu_eff > 0 else 1.0
        expected_delay = min(expected_delay, T)
        pv_factor = (1 - np.exp(-rho * expected_delay)) / rho if rho > 0.001 else expected_delay

        forgone = K_alpha * X_mid * pv_factor
        return forgone

    def default_probability(self, lambda_true: float, lambda_aggressive: float,
                             T: float = 5.0) -> float:
        """
        Default probability when firm invests too early (aggressive λ).

        When firm uses λ_aggressive > λ_true:
        - Invests at X*_aggress < X*_true (invested early, demand may stay low)
        - Post-investment in L regime: X follows GBM with low drift μ_L
        - Default when X falls to X_D (Leland boundary)

        Uses first-passage time probability for GBM hitting X_D from X*_aggress.
        """
        K_star, X_aggress = self.optimal_policy(lambda_aggressive)
        _, X_true = self.optimal_policy(lambda_true)

        if X_aggress >= X_true:
            return 0.0   # not overinvesting (more conservative)

        # Default boundary for this firm after investment
        ld = LelandDefault(self.p, K_star, 'H', in_duopoly=False)
        X_D = ld.default_boundary()

        if X_D <= 0:
            return 0.0

        # Invested at X_aggress (lower than X_true); compute default probability
        if X_aggress <= X_D:
            return 1.0

        return ld.default_probability(X_aggress, T=T)

    def dilemma_table(self, lambda_true: float = 0.30, T: float = 5.0,
                       X_0: float = 0.10) -> pd.DataFrame:
        """
        Compute the full dilemma table for a range of misspecifications.
        """
        misspec_levels = [-0.50, -0.30, -0.15, 0.0, 0.15, 0.30, 0.50]
        rows = []

        K_star, X_true = self.optimal_policy(lambda_true)

        for eps in misspec_levels:
            lam_used = lambda_true * (1 + eps)
            _, X_used = self.optimal_policy(lam_used)

            if eps < 0:  # conservative → invests later
                forgone = self.forgone_revenue(lambda_true, lam_used, T, X_0)
                dp = 0.0
            elif eps > 0:  # aggressive → invests earlier
                forgone = 0.0
                dp = self.default_probability(lambda_true, lam_used, T)
            else:
                forgone = 0.0
                dp = 0.0

            rows.append({
                'Misspecification ε': f'{eps:+.0%}',
                'λ used': round(lam_used, 3),
                'X* (used)': round(X_used, 5),
                'X* (true)': round(X_true, 5),
                'X* deviation': f'{(X_used - X_true)/X_true:+.1%}',
                'Forgone Revenue (scaled, 5yr)': round(forgone * 20, 4),
                '5-yr Default Prob.': f'{dp:.1%}',
            })

        return pd.DataFrame(rows)

    def plot_dilemma(self, lambda_true: float = 0.30, T: float = 5.0,
                      output_path: str = None) -> plt.Figure:
        """
        Two-panel figure showing the Dario dilemma:
        - Left: forgone revenue vs. underinvestment degree
        - Right: default probability vs. overinvestment degree
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel (a): Underinvestment cost
        ax = axes[0]
        eps_vals = np.linspace(-0.70, -0.05, 40)
        lam_conserv = lambda_true * (1 + eps_vals)
        forgone_vals = [self.forgone_revenue(lambda_true, lam, T, 0.10) * 20
                         for lam in lam_conserv]  # in $B

        ax.plot(-eps_vals * 100, forgone_vals, color='#1f77b4', lw=2.5)
        ax.fill_between(-eps_vals * 100, 0, forgone_vals, alpha=0.2, color='#1f77b4')
        ax.axvline(30, ls='--', color='gray', lw=1.5, label='30% underinvestment')
        ax.set_xlabel('Degree of underinvestment (%)', fontsize=11)
        ax.set_ylabel('Forgone revenue ($B over 5 years)', fontsize=11)
        ax.set_title(f'(a) Cost of Underinvestment\n($\\lambda_{{true}}={lambda_true}$)',
                      fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

        # Panel (b): Overinvestment cost
        ax = axes[1]
        eps_vals_pos = np.linspace(0.05, 0.80, 40)
        lam_aggress = lambda_true * (1 + eps_vals_pos)
        dp_vals = [self.default_probability(lambda_true, lam, T) * 100
                    for lam in lam_aggress]

        ax.plot(eps_vals_pos * 100, dp_vals, color='#d62728', lw=2.5)
        ax.fill_between(eps_vals_pos * 100, 0, dp_vals, alpha=0.2, color='#d62728')
        ax.axvline(30, ls='--', color='gray', lw=1.5, label='30% overinvestment')
        ax.axhline(20, ls=':', color='orange', lw=1.5, label='20% threshold')
        ax.set_xlabel('Degree of overinvestment (%)', fontsize=11)
        ax.set_ylabel('5-year default probability (%)', fontsize=11)
        ax.set_title(f'(b) Cost of Overinvestment (Default Risk)\n($\\lambda_{{true}}={lambda_true}$)',
                      fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 100)

        plt.suptitle('The Dario Dilemma: Asymmetric Costs of λ Misspecification',
                      fontsize=12)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {output_path}")
        return fig


# --------------------------------------------------------------------------- #
#  Plotting utilities                                                            #
# --------------------------------------------------------------------------- #

def plot_valuation_decomposition(params: DuopolyParams,
                                  output_path: str = None) -> plt.Figure:
    """
    4-panel valuation analysis figure.
    """
    valuation = AIFirmValuation(params)
    model_sf = SingleFirmModel(params)
    sol = model_sf.solve()

    K_star = sol['H']['K_star']
    X_star = sol['H']['X_star']

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    # Panel (a): Equity sensitivity to λ
    ax = axes[0]
    lambda_range = np.linspace(0.05, 0.80, 40)
    X_vals = [X_star * 0.8, X_star, X_star * 1.5, X_star * 2.5]
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(X_vals)))
    for X_val, color in zip(X_vals, colors):
        eq_vals = valuation.equity_sensitivity_to_lambda(X_val, K_star, lambda_range)
        ax.plot(lambda_range, eq_vals, color=color, lw=2, label=f'$X={X_val:.3f}$')
    ax.set_xlabel('Adoption arrival rate $\\lambda$', fontsize=10)
    ax.set_ylabel('Equity value $E(X)$', fontsize=10)
    ax.set_title('(a) Equity Value vs. $\\lambda$ (AGI Sensitivity)', fontsize=10)
    ax.legend(title='Demand $X$', fontsize=8)
    ax.grid(alpha=0.3)

    # Panel (b): Credit spread curve
    ax = axes[1]
    coupon_range = np.linspace(0.001, 0.05, 50)
    for in_duo, color, label in [(False, '#1f77b4', 'Monopolist'),
                                   (True, '#d62728', 'Duopolist (θ=0.6)')]:
        cs_vals = valuation.credit_spread_curve(X_star, K_star, coupon_range, in_duo)
        ax.plot(coupon_range * 100, cs_vals, color=color, lw=2, label=label)
    ax.set_xlabel('Debt coupon $C_D$ (annual rate, %)', fontsize=10)
    ax.set_ylabel('Credit spread (bps)', fontsize=10)
    ax.set_title('(b) Credit Spread Curve', fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 500)
    ax.grid(alpha=0.3)

    # Panel (c): Value decomposition across X
    ax = axes[2]
    X_grid = np.linspace(X_star * 0.3, X_star * 3.0, 30)
    V_AIP_vals = [valuation.assets_in_place(x, K_star) for x in X_grid]
    V_lambda_vals = [valuation.agr_premium(x, K_star) for x in X_grid]

    ax.stackplot(X_grid, V_AIP_vals, V_lambda_vals,
                  labels=['Assets in place $V_{AIP}$', 'AGI premium $V_\\lambda$'],
                  colors=['#2ca02c', '#ff7f0e'], alpha=0.7)
    ax.axvline(X_star, ls='--', color='black', lw=1.5, label='Investment trigger $X^*$')
    ax.set_xlabel('Demand level $X$', fontsize=10)
    ax.set_ylabel('Firm value', fontsize=10)
    ax.set_title('(c) Firm Value Decomposition', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel (d): AGI premium fraction vs λ
    ax = axes[3]
    lambda_range2 = np.linspace(0.05, 0.80, 30)
    decomp_vals = []
    import copy
    for lam in lambda_range2:
        p_new = copy.copy(params)
        p_new.lam = lam
        try:
            p_new.validate()
            v = AIFirmValuation(p_new)
            decomp = v.growth_option_decomposition(X_star, K_star, K_star * 2.0)
            decomp_vals.append(decomp['lambda_fraction'])
        except Exception:
            decomp_vals.append(np.nan)

    ax.plot(lambda_range2, np.array(decomp_vals) * 100, '#d62728', lw=2.5)
    ax.fill_between(lambda_range2, 0, np.array(decomp_vals) * 100, alpha=0.2, color='#d62728')
    ax.set_xlabel('Adoption arrival rate $\\lambda$', fontsize=10)
    ax.set_ylabel('AGI premium fraction of total value (%)', fontsize=10)
    ax.set_title('(d) Growth Option Intensity vs. $\\lambda$', fontsize=10)
    ax.grid(alpha=0.3)

    plt.suptitle('Phase 5: AI Firm Valuation Analysis', fontsize=13)
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    return fig


# --------------------------------------------------------------------------- #
#  Main                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import os
    os.makedirs('/workspaces/claude-paper/figures', exist_ok=True)

    print("=" * 60)
    print("Phase 5: Valuation and the Dario Dilemma")
    print("=" * 60)

    params = DuopolyParams()
    valuation = AIFirmValuation(params)
    model_sf = SingleFirmModel(params)
    sol = model_sf.solve()

    K_star = sol['H']['K_star']
    X_star = sol['H']['X_star']

    print(f"\nReference point: K*={K_star:.4f}, X*={X_star:.4f}")

    # --- Value decomposition ---
    print("\n--- Firm Value Decomposition (at trigger X*) ---")
    decomp = valuation.growth_option_decomposition(X_star, K_star, K_star * 2.0)
    print(f"  V_AIP:    {decomp['V_AIP']:.4f}  ({decomp['AIP_fraction']:.1%})")
    print(f"  V_expand: {decomp['V_expand']:.4f}  ({decomp['expand_fraction']:.1%})")
    print(f"  V_λ:      {decomp['V_lambda']:.4f}  ({decomp['lambda_fraction']:.1%})")
    print(f"  V_total:  {decomp['V_total']:.4f}")

    # --- Equity sensitivity ---
    print("\n--- Equity Sensitivity to λ (at X*) ---")
    lam_test = [0.10, 0.20, 0.30, 0.40, 0.50]
    E_vals = valuation.equity_sensitivity_to_lambda(X_star, K_star, np.array(lam_test))
    for lam, E in zip(lam_test, E_vals):
        print(f"  λ={lam}: E={E:.4f}")

    # --- Credit spreads ---
    print("\n--- Credit Spread Analysis ---")
    ld = LelandDefault(params, K_star, 'H', False)
    for X_mult in [0.5, 1.0, 1.5, 2.0]:
        X_test = X_star * X_mult
        E = ld.equity_value(X_test)
        D = ld.debt_value(X_test)
        cs = ld.credit_spread(X_test) * 10000
        dp = ld.default_probability(X_test, T=5.0) * 100
        print(f"  X={X_test:.4f}: E={E:.4f}, D={D:.4f}, CS={cs:.1f}bps, DP(5yr)={dp:.1f}%")

    # --- Dario Dilemma ---
    print("\n--- The Dario Dilemma (λ_true = 0.30) ---")
    dilemma = DarioDilemma(params)
    table = dilemma.dilemma_table(lambda_true=0.30, T=5.0)
    print(table.to_string(index=False))

    print("\nKey insights:")
    print("  - 30% underinvestment: forgone revenue ≈ ${:.1f}B over 5 years".format(
        dilemma.forgone_revenue(0.30, 0.21, 5.0, 0.10) * 20))
    print("  - 30% overinvestment: 5yr default probability ≈ {:.1%}".format(
        dilemma.default_probability(0.30, 0.39, 5.0)))

    # --- Figures ---
    print("\nGenerating figures...")
    plot_valuation_decomposition(params,
                                  output_path='/workspaces/claude-paper/figures/phase5_valuation.png')
    dilemma.plot_dilemma(lambda_true=0.30,
                          output_path='/workspaces/claude-paper/figures/phase5_dario_dilemma.png')

    print("\nPhase 5 Valuation complete. ✓")
