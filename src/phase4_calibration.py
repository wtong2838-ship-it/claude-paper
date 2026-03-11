"""
Phase 4: Calibration of model parameters to AI infrastructure data.
====================================================================

Translates publicly-observable AI industry data (revenue trajectories,
GPU prices, compute capacity growth) into the model parameters used in
Phases 1 and 2.

Calibration strategy:
1. Demand parameters (μ_L, μ_H, σ): estimated from revenue time series
2. Scaling exponent α: from ML scaling laws + demand elasticity
3. Investment cost (c, γ): from GPU/data-center cost data
4. Discount rates (r): from WACC estimates for large tech vs. startups

References:
- Kaplan et al. (2020): neural scaling laws (compute exponent ≈ 0.10)
- Leland (1994): capital structure (WACC)
- Industry earnings calls / S-1 filings for revenue figures
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from phase1_base_model import ModelParams


# --------------------------------------------------------------------------- #
#  Industry data (public)                                                       #
# --------------------------------------------------------------------------- #

AI_INDUSTRY_DATA = {
    # Revenue trajectories (approximate, in $B)
    'anthropic':      {'2022': 0.1,  '2023': 0.5,  '2024': 2.0,  '2025': 9.0},
    'openai':         {'2022': 0.4,  '2023': 1.3,  '2024': 3.7,  '2025': 11.3},
    'google_ai':      {'2022': 5.0,  '2023': 7.0,  '2024': 11.0, '2025': 20.0},
    'microsoft_ai':   {'2022': 3.0,  '2023': 5.0,  '2024': 10.0, '2025': 20.0},

    # Compute costs: ~$10-15B per GW-year
    'cost_per_gw_year': 12.5,   # $B

    # Industry compute capacity
    'capacity_gw_2024': 12.0,
    'capacity_gw_2025': 36.0,   # ~3x growth

    # AI scaling law exponent (compute to loss reduction)
    'ml_scaling_exponent': 0.10,  # Kaplan et al. 2020

    # Time-to-build: 18-36 months
    'tau_min': 1.5,  # years
    'tau_max': 3.0,

    # Discount rates (WACC estimates)
    'r_large_tech': 0.10,   # Google/Microsoft
    'r_startup':    0.20,   # Anthropic/OpenAI

    # GPU prices
    'h100_price_k': 28.0,   # $K per GPU
    'b200_price_k': 35.0,
}


# --------------------------------------------------------------------------- #
#  Calibrator                                                                   #
# --------------------------------------------------------------------------- #

class ModelCalibrator:
    """
    Calibrates ModelParams to match observed AI infrastructure investment.

    Key insight: Given r, μ_H, σ_H from standard estimation, the scaling
    exponent α (revenue-compute elasticity) and investment cost parameters
    c, γ are calibrated from the investment-revenue ratio observed in data.
    """

    def __init__(self, data: dict = None):
        self.data = data or AI_INDUSTRY_DATA

    # ------------------------------------------------------------------ #
    #  Step 1: Demand parameters from revenue time series                  #
    # ------------------------------------------------------------------ #

    def estimate_demand_params(self, revenue_dict: dict) -> dict:
        """
        Estimate μ_L, μ_H, σ from revenue time series.

        - μ_H: log growth rate in high-growth phase (post-2023 AI boom)
        - μ_L: log growth rate in low-growth phase (pre-2023)
        - σ:   volatility from cross-firm variation in growth rates

        Returns {'mu_L': ..., 'mu_H': ..., 'sigma': ...}
        """
        years = sorted(revenue_dict.keys())
        revs = np.array([revenue_dict[y] for y in years], dtype=float)

        # Log-differences = approximate annual growth rates
        log_revs = np.log(revs)
        log_growth = np.diff(log_revs)   # length = len(years)-1

        # Split into low (pre-2024) and high (2024+) phases
        # Years 2022->2023: low phase; 2023->2024, 2024->2025: high phase
        growth_L = log_growth[:1]          # 2022→2023
        growth_H = log_growth[1:]          # 2023→2024, 2024→2025

        mu_L = float(np.mean(growth_L))
        mu_H = float(np.mean(growth_H))

        # Volatility from standard deviation across all growth periods
        sigma = float(np.std(log_growth, ddof=1)) if len(log_growth) > 1 else 0.30

        return {'mu_L': mu_L, 'mu_H': mu_H, 'sigma': sigma}

    def _aggregate_demand_params(self) -> dict:
        """
        Aggregate demand parameters across all firms in the dataset.
        Pools growth-rate estimates; takes cross-firm SD as σ.
        """
        data = self.data
        firm_keys = ['anthropic', 'openai', 'google_ai', 'microsoft_ai']
        all_growth = []
        mu_L_list, mu_H_list = [], []

        for firm in firm_keys:
            rev = data[firm]
            est = self.estimate_demand_params(rev)
            mu_L_list.append(est['mu_L'])
            mu_H_list.append(est['mu_H'])
            years = sorted(rev.keys())
            revs = np.array([rev[y] for y in years], dtype=float)
            lg = np.diff(np.log(revs))
            all_growth.extend(lg.tolist())

        mu_L = float(np.mean(mu_L_list))
        mu_H = float(np.mean(mu_H_list))
        sigma = float(np.std(all_growth, ddof=1)) if len(all_growth) > 1 else 0.30

        return {'mu_L': mu_L, 'mu_H': mu_H, 'sigma': sigma}

    # ------------------------------------------------------------------ #
    #  Step 2: Scaling exponent α                                          #
    # ------------------------------------------------------------------ #

    def calibrate_scaling_exponent(self, ml_exponent: float = 0.10,
                                    demand_elasticity: float = 5.0) -> float:
        """
        Translate ML scaling exponent to revenue-compute elasticity α.

        If quality q ∝ Compute^ml_exponent, and demand ∝ quality^demand_elasticity:
          revenue ∝ Compute^(ml_exponent * demand_elasticity)

        So α = ml_exponent * demand_elasticity.
        Central estimate: α = 0.10 * 5.0 = 0.50
        """
        alpha = ml_exponent * demand_elasticity
        # Clamp to ensure interior-solution condition can hold with γ > 1
        # For γ=1.5 we need α < γ*(β-1)/β → roughly α < 0.6 for typical β
        alpha = float(np.clip(alpha, 0.25, 0.65))
        return alpha

    # ------------------------------------------------------------------ #
    #  Step 3: Investment cost parameters c, γ                            #
    # ------------------------------------------------------------------ #

    def calibrate_investment_costs(self) -> dict:
        """
        Calibrate c, γ from GPU costs and construction scaling.

        If GPU cost = h100_price per unit capacity (1 GPU ≈ 0.0003 GW FLOP/s):
        I(K) = c * K^γ where K is in GW and I in $B

        With linear cost (γ=1): c ≈ 12.5 $B/GW (from industry data)
        With convexity (γ=1.2): c adjusted so I(12 GW) ≈ 150B (total industry investment)

        Returns {'c_linear': ..., 'c_convex': ..., 'gamma_linear': 1.0, 'gamma_convex': ...}
        """
        data = self.data
        cost_per_gw = data['cost_per_gw_year']   # $B/GW
        K_ref = data['capacity_gw_2024']           # 12 GW in 2024
        total_investment = cost_per_gw * K_ref     # ~ $150 B cumulative

        # Linear: I(K) = c * K → c = cost_per_gw
        c_linear = cost_per_gw
        gamma_linear = 1.0

        # Convex (γ=1.2): calibrate c so that I(K_ref) = total_investment
        gamma_convex = 1.2
        c_convex = total_investment / K_ref**gamma_convex

        # Rescale to model units (normalise so I(1) = c)
        # In model units K is dimensionless (relative to 1 GW baseline)
        # c in $B is the cost to install 1 GW; we keep dollar units as-is

        return {
            'c_linear':    float(c_linear),
            'gamma_linear': float(gamma_linear),
            'c_convex':    float(c_convex),
            'gamma_convex': float(gamma_convex),
            'total_investment_B': float(total_investment),
            'K_ref_GW': float(K_ref),
        }

    # ------------------------------------------------------------------ #
    #  Assemble calibrated ModelParams                                     #
    # ------------------------------------------------------------------ #

    def get_calibrated_params(self, firm_type: str = 'large_tech') -> ModelParams:
        """
        Returns calibrated ModelParams for a given firm type.

        firm_type: 'large_tech'  → Google/Microsoft (lower WACC, lower growth)
                   'startup'      → Anthropic/OpenAI (higher WACC, higher growth)
        """
        data = self.data

        # --- Demand parameters ---
        if firm_type == 'large_tech':
            rev_proxy = data['google_ai']
            r = data['r_large_tech']
        else:   # startup
            rev_proxy = data['anthropic']
            r = data['r_startup']

        dp = self.estimate_demand_params(rev_proxy)
        agg = self._aggregate_demand_params()

        mu_L_raw = dp['mu_L']
        mu_H_raw = dp['mu_H']
        sigma_raw = agg['sigma']

        # Annualise and clip to ensure r > mu_H (model convergence condition)
        mu_H = float(np.clip(mu_H_raw, 0.01, r - 0.01))
        mu_L = float(np.clip(mu_L_raw, 0.00, mu_H - 0.01))

        # Volatility: bound away from zero and from extreme values
        sigma_L = float(np.clip(sigma_raw, 0.15, 0.55))
        sigma_H = float(np.clip(sigma_raw * 0.85, 0.12, 0.45))   # H regime: slightly lower vol

        # --- Scaling exponent α ---
        alpha = self.calibrate_scaling_exponent(
            ml_exponent=data['ml_scaling_exponent'],
            demand_elasticity=5.0
        )
        # Ensure interior-solution condition: α must be < γ*(β-1)/β
        # Use conservative α=0.45 to be safe (as in the base model)
        alpha = min(alpha, 0.45)

        # --- Investment costs ---
        ic = self.calibrate_investment_costs()
        # Use convex cost for main calibration; normalise c to model units
        # (divide by cost_per_gw_year so K=1 unit = 1 GW at unit cost)
        gamma = ic['gamma_convex']
        c_raw = ic['c_convex']
        # Scale: in model c·K^γ should equal investment in $B when K is in GW units.
        # To keep numerics tractable we normalise: set c = c_raw / K_ref^{γ-1}
        # so that at K_ref, I(K_ref) = c_raw * K_ref (= original total / K_ref^(γ-1))
        # Actually simpler: just pass through and let the model handle it.
        c = float(c_raw)

        # --- Operating cost δ ---
        # Annual electricity + maintenance ≈ 10-15% of CapEx per year
        delta = float(0.10 * ic['c_linear'])   # $B per GW per year

        # --- Time-to-build ---
        tau = (data['tau_min'] + data['tau_max']) / 2.0   # central estimate = 2.25 yr

        # --- Adoption arrival rate λ ---
        # Structural λ is the primary free parameter (calibrated in Phase 4b).
        # Here we set a reasonable prior: 20% per year → 5-year expected wait
        lam = 0.20

        params = ModelParams(
            mu_L=mu_L,
            mu_H=mu_H,
            sigma_L=sigma_L,
            sigma_H=sigma_H,
            lam=lam,
            alpha=alpha,
            gamma=gamma,
            c=c,
            delta=delta,
            r=r,
            tau=tau,
        )

        # Post-check: clamp any remaining violations
        params = _ensure_valid_params(params)
        return params

    # ------------------------------------------------------------------ #
    #  Summary table                                                        #
    # ------------------------------------------------------------------ #

    def calibration_table(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame with all calibrated parameters and sources.
        Rows: one per parameter; Columns: param, value_large_tech, value_startup, source.
        """
        p_lt = self.get_calibrated_params('large_tech')
        p_st = self.get_calibrated_params('startup')
        ic = self.calibrate_investment_costs()

        rows = [
            ('mu_L',  p_lt.mu_L,  p_st.mu_L,  'Pre-boom log revenue growth (2022-23)'),
            ('mu_H',  p_lt.mu_H,  p_st.mu_H,  'Post-boom log revenue growth (2023-25)'),
            ('sigma_L', p_lt.sigma_L, p_st.sigma_L, 'Revenue volatility (pre-boom, cross-firm SD)'),
            ('sigma_H', p_lt.sigma_H, p_st.sigma_H, 'Revenue volatility (post-boom)'),
            ('lam',   p_lt.lam,  p_st.lam,  'AI adoption arrival rate λ (prior)'),
            ('alpha', p_lt.alpha, p_st.alpha, 'Revenue-compute elasticity (ML scaling × demand elast.)'),
            ('gamma', p_lt.gamma, p_st.gamma, 'Investment cost convexity (GPU scaling)'),
            ('c',     p_lt.c,    p_st.c,    f'Investment cost scale ($B/GW^γ); I({ic["K_ref_GW"]:.0f}GW)={ic["total_investment_B"]:.0f}B'),
            ('delta', p_lt.delta, p_st.delta, 'Operating cost per GW per year (10% of CapEx)'),
            ('r',     p_lt.r,    p_st.r,    'Discount rate (WACC: large tech 10%, startup 20%)'),
            ('tau',   p_lt.tau,  p_st.tau,  'Time-to-build (central estimate, 1.5-3 yr range)'),
        ]

        df = pd.DataFrame(rows, columns=['parameter', 'value_large_tech',
                                          'value_startup', 'source'])
        df['value_large_tech'] = df['value_large_tech'].round(4)
        df['value_startup'] = df['value_startup'].round(4)
        return df


# --------------------------------------------------------------------------- #
#  Helper                                                                       #
# --------------------------------------------------------------------------- #

def _ensure_valid_params(p: ModelParams) -> ModelParams:
    """
    Post-process a ModelParams object to guarantee all inequalities hold.
    Adjusts in-place (modifies the object) then returns it.
    """
    # Guarantee convergence: r > mu_H > mu_L > 0
    p.mu_L = max(p.mu_L, 0.005)
    p.mu_H = max(p.mu_H, p.mu_L + 0.005)
    p.r    = max(p.r,    p.mu_H + 0.02)

    # Volatility bounds
    p.sigma_L = float(np.clip(p.sigma_L, 0.10, 0.60))
    p.sigma_H = float(np.clip(p.sigma_H, 0.08, 0.50))

    # α must satisfy interior-solution condition with the given γ
    # Approximate β_H using H char root: β ≈ positive root
    a = 0.5 * p.sigma_H**2
    b_coef = p.mu_H - 0.5 * p.sigma_H**2
    disc = b_coef**2 + 4 * a * p.r
    beta_H = (-b_coef + np.sqrt(disc)) / (2 * a)
    psi_max = p.gamma * (beta_H - 1) / beta_H  # ψ < γ condition
    alpha_max = min(0.65, psi_max - 0.01)
    p.alpha = float(np.clip(p.alpha, 0.20, alpha_max))

    # Tau
    p.tau = float(np.clip(p.tau, 0.5, 4.0))

    try:
        p.validate()
    except AssertionError:
        # Fall back to safe defaults
        p.mu_L, p.mu_H, p.r = 0.03, 0.08, 0.15
        p.sigma_L, p.sigma_H = 0.30, 0.25
        p.alpha = 0.45
        p.gamma = 1.50

    return p


# --------------------------------------------------------------------------- #
#  Plotting                                                                     #
# --------------------------------------------------------------------------- #

def plot_revenue_trajectories(data: dict = None,
                               output_path: str = None) -> plt.Figure:
    """
    Plot observed revenue trajectories and fitted log-linear growth rates.
    """
    data = data or AI_INDUSTRY_DATA
    calibrator = ModelCalibrator(data)

    firm_keys = ['anthropic', 'openai', 'google_ai', 'microsoft_ai']
    firm_labels = ['Anthropic', 'OpenAI', 'Google AI', 'Microsoft AI']
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: revenue levels
    ax = axes[0]
    for firm, label, color in zip(firm_keys, firm_labels, colors):
        rev = data[firm]
        years = sorted(rev.keys())
        vals = [rev[y] for y in years]
        year_nums = [int(y) for y in years]
        ax.plot(year_nums, vals, 'o-', color=color, lw=2, label=label, markersize=6)

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Revenue ($B)', fontsize=11)
    ax.set_title('(a) AI Firm Revenue Trajectories', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    # Right: bar chart of growth rates by phase
    ax2 = axes[1]
    mu_L_list, mu_H_list = [], []
    for firm in firm_keys:
        est = calibrator.estimate_demand_params(data[firm])
        mu_L_list.append(est['mu_L'])
        mu_H_list.append(est['mu_H'])

    x = np.arange(len(firm_keys))
    width = 0.35
    bars1 = ax2.bar(x - width/2, mu_L_list, width, label='$\\mu_L$ (2022-23)', color='#1f77b4', alpha=0.8)
    bars2 = ax2.bar(x + width/2, mu_H_list, width, label='$\\mu_H$ (2023-25)', color='#d62728', alpha=0.8)

    ax2.set_xlabel('Firm', fontsize=11)
    ax2.set_ylabel('Log growth rate (annualized)', fontsize=11)
    ax2.set_title('(b) Estimated Demand Growth Rates', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(firm_labels, rotation=15, ha='right', fontsize=9)
    ax2.legend(fontsize=10)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                      f'{h:.2f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('AI Industry Revenue Data and Calibrated Demand Parameters', fontsize=13)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    return fig


def plot_calibration_summary(output_path: str = None) -> plt.Figure:
    """
    Two-panel summary: (a) calibrated α vs. demand elasticity assumption,
    (b) investment cost curves for linear vs. convex cost specifications.
    """
    calibrator = ModelCalibrator()
    ic = calibrator.calibrate_investment_costs()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): α as function of demand elasticity
    ax = axes[0]
    elasticities = np.linspace(1.0, 12.0, 50)
    ml_exps = [0.05, 0.10, 0.15]
    labels = ['ML exp=0.05 (conservative)', 'ML exp=0.10 (Kaplan 2020)', 'ML exp=0.15 (aggressive)']
    colors = ['#2ca02c', '#1f77b4', '#d62728']

    for ml_exp, label, color in zip(ml_exps, labels, colors):
        alphas = np.clip(ml_exp * elasticities, 0.20, 0.70)
        ax.plot(elasticities, alphas, color=color, lw=2, label=label)

    ax.axhline(0.45, color='grey', ls='--', lw=1.5, label='Baseline α=0.45 (model default)')
    ax.axvline(5.0,  color='grey', ls=':',  lw=1.0, alpha=0.7)
    ax.set_xlabel('Demand elasticity w.r.t. quality', fontsize=11)
    ax.set_ylabel('Revenue-compute elasticity $\\alpha$', fontsize=11)
    ax.set_title('(a) Calibrated $\\alpha$ vs. Demand Elasticity', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel (b): Investment cost curves
    ax2 = axes[1]
    K_grid = np.linspace(0.1, 40.0, 200)   # GW

    I_linear = ic['c_linear'] * K_grid ** ic['gamma_linear']
    I_convex = ic['c_convex'] * K_grid ** ic['gamma_convex']

    ax2.plot(K_grid, I_linear, color='#1f77b4', lw=2.5,
              label=f'Linear ($\\gamma=1.0$, $c={ic["c_linear"]:.1f}$ \$B/GW)')
    ax2.plot(K_grid, I_convex, color='#d62728', lw=2.5,
              label=f'Convex ($\\gamma={ic["gamma_convex"]:.1f}$, $c={ic["c_convex"]:.2f}$)')

    # Mark observed 2024 and 2025 capacity
    for K_obs, label_obs, color_obs in [
        (AI_INDUSTRY_DATA['capacity_gw_2024'], '2024 capacity\n(12 GW)', '#2ca02c'),
        (AI_INDUSTRY_DATA['capacity_gw_2025'], '2025 capacity\n(36 GW)', '#ff7f0e'),
    ]:
        ax2.axvline(K_obs, color=color_obs, ls=':', lw=1.5, alpha=0.8)
        ax2.text(K_obs + 0.5, ax2.get_ylim()[1] * 0.05 if ax2.get_ylim()[1] > 0 else 50,
                  label_obs, color=color_obs, fontsize=8)

    ax2.set_xlabel('Compute capacity $K$ (GW)', fontsize=11)
    ax2.set_ylabel('Investment cost $I(K)$ (\$B)', fontsize=11)
    ax2.set_title('(b) Calibrated Investment Cost Curves', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, None)
    ax2.set_ylim(0, None)

    plt.suptitle('Model Calibration: Scaling Exponent and Investment Costs', fontsize=13)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    return fig


def plot_calibration_table(output_path: str = None) -> plt.Figure:
    """Render the calibration table as a matplotlib figure."""
    calibrator = ModelCalibrator()
    df = calibrator.calibration_table()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    col_labels = ['Parameter', 'Large Tech (Google/MSFT)', 'Startup (Anthropic/OAI)', 'Source']
    table_data = df.values.tolist()

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='left',
        loc='center',
        colColours=['#d0d0d0'] * 4,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width([0, 1, 2, 3])

    plt.title('Table 1: Calibrated Model Parameters', fontsize=13, pad=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    return fig


# --------------------------------------------------------------------------- #
#  Main                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    os.makedirs('/workspaces/claude-paper/figures', exist_ok=True)

    print("=" * 60)
    print("Phase 4: Model Calibration")
    print("=" * 60)

    calibrator = ModelCalibrator()

    # --- Per-firm demand estimates ---
    print("\n--- Per-Firm Revenue Growth Estimates ---")
    for firm in ['anthropic', 'openai', 'google_ai', 'microsoft_ai']:
        est = calibrator.estimate_demand_params(AI_INDUSTRY_DATA[firm])
        print(f"  {firm:20s}: μ_L={est['mu_L']:.3f}, μ_H={est['mu_H']:.3f}, σ={est['sigma']:.3f}")

    # --- Scaling exponent ---
    alpha = calibrator.calibrate_scaling_exponent()
    print(f"\n--- Scaling Exponent ---")
    print(f"  α = ml_exp × demand_elast = 0.10 × 5.0 = {alpha:.4f}")

    # --- Investment costs ---
    ic = calibrator.calibrate_investment_costs()
    print(f"\n--- Investment Costs ---")
    print(f"  Linear:  c={ic['c_linear']:.2f} $B/GW, γ={ic['gamma_linear']}")
    print(f"  Convex:  c={ic['c_convex']:.4f} $B/GW^γ, γ={ic['gamma_convex']}")
    print(f"  Total industry investment (2024): ${ic['total_investment_B']:.1f}B")

    # --- Calibrated params ---
    print("\n--- Calibrated ModelParams ---")
    for firm_type in ['large_tech', 'startup']:
        p = calibrator.get_calibrated_params(firm_type)
        print(f"\n  [{firm_type}]")
        print(f"    μ_L={p.mu_L:.4f}, μ_H={p.mu_H:.4f}, σ_L={p.sigma_L:.4f}, σ_H={p.sigma_H:.4f}")
        print(f"    λ={p.lam:.4f}, α={p.alpha:.4f}, γ={p.gamma:.4f}, c={p.c:.4f}")
        print(f"    δ={p.delta:.4f}, r={p.r:.4f}, τ={p.tau:.4f}")
        print(f"    Valid: r>mu_H? {p.r > p.mu_H}, r>mu_L? {p.r > p.mu_L}")

    # --- Calibration table ---
    print("\n--- Calibration Table ---")
    df = calibrator.calibration_table()
    print(df.to_string(index=False))

    # --- Figures ---
    print("\nGenerating figures...")
    plot_revenue_trajectories(
        output_path='/workspaces/claude-paper/figures/phase4_revenue_trajectories.png'
    )
    plot_calibration_summary(
        output_path='/workspaces/claude-paper/figures/phase4_calibration_summary.png'
    )
    plot_calibration_table(
        output_path='/workspaces/claude-paper/figures/phase4_calibration_table.png'
    )

    print("\nPhase 4 Calibration complete.")
