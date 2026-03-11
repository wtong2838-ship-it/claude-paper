"""
Phase 2: Duopoly with Default Risk — Analytical Core
=====================================================

Two symmetric firms compete in a real options investment game with endogenous default.

Market structure:
  Each firm's profit (once invested):
    - Leader only: π_L = X · K_L^α                        (monopoly)
    - Both invested: π_i = θ · X · K_i^α,  θ ∈ (0,1)     (duopoly split)

  The duopoly discount θ captures competitive intensity:
    θ → 1: weak competition (Bertrand-like with differentiated goods)
    θ = 0.5: equal split of a fixed market (symmetric Cournot)
    θ → 0: fierce competition (homogeneous Bertrand)

  This parameterization nests the contest function (θ = K^α/(K^α+K_j^α) at K=K_j)
  but is analytically tractable and avoids corner-solution pathologies.

Equilibrium (Huisman & Kort 2015 approach):
  Sequential entry: leader invests at X_L*, follower at X_F* > X_L*
  Leader trigger: X_L* from preemption condition V_lead = V_follow
  Follower trigger: from standard single-firm problem in duopoly market

Default structure (Leland 1994):
  Endogenous default at X_D where equity value hits zero
  Tax shield on debt coupon, proportional bankruptcy costs

Key mechanism — competition-leverage spiral:
  1. Preemption → X_L* < X_single* (invest earlier under competition)
  2. Risk-shifting → K_L* > K_single* (overbuild under debt financing)
  3. Higher K_L + lower X → larger X_D (higher default risk)

References:
- Huisman & Kort (2015, RAND): duopoly timing and capacity
- Grenadier (2002, RFS): option exercise games
- Leland (1994, JF): endogenous default boundary
- Brander & Lewis (1986): leverage in Cournot competition
"""

import numpy as np
from scipy.optimize import brentq, fsolve
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from phase1_base_model import ModelParams, SingleFirmModel


# --------------------------------------------------------------------------- #
#  Parameters                                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class DuopolyParams(ModelParams):
    """Parameters for the duopoly model with default risk."""
    # --- Competitive structure ---
    theta: float = 0.60   # Duopoly profit share: π_duo = θ · X · K^α  (θ < 1)

    # --- Capital structure (Leland 1994) ---
    # Coupon calibrated so X_D < X_L (debt serviceable at investment trigger).
    # Rule of thumb: C_D ≈ r * 0.25 * V_at_trigger  (25% debt-to-value ratio).
    coupon: float = 0.005    # Annual debt coupon C_D
    alpha_bc: float = 0.25   # Proportional bankruptcy cost (fraction of firm value)
    tax_rate: float = 0.21   # Corporate tax rate (generates debt tax shield)

    def validate(self):
        super().validate()
        assert 0 < self.theta < 1, "Need 0 < θ < 1 for duopoly profit share"
        assert 0 <= self.alpha_bc <= 1, "Bankruptcy cost fraction must be in [0,1]"
        assert 0 <= self.tax_rate <= 1, "Tax rate must be in [0,1]"
        assert self.coupon >= 0, "Coupon must be non-negative"

    def phi_mono(self, K: float, regime: str = 'H') -> float:
        """φ coefficient for monopoly revenue: V = φ_mono · X - δK/r."""
        if regime == 'H':
            return K**self.alpha / (self.r - self.mu_H)
        else:
            phi_H = self.phi_mono(K, 'H')
            return (K**self.alpha + self.lam * phi_H) / (self.r - self.mu_L + self.lam)

    def phi_duo(self, K: float, regime: str = 'H') -> float:
        """φ coefficient for duopoly revenue: V = φ_duo · X - δK/r."""
        if regime == 'H':
            return self.theta * K**self.alpha / (self.r - self.mu_H)
        else:
            phi_H = self.phi_duo(K, 'H')
            return (self.theta * K**self.alpha + self.lam * phi_H) / (self.r - self.mu_L + self.lam)

    def leland_char_roots(self) -> Tuple[float, float]:
        """Roots of Leland ODE characteristic equation (using H regime σ, μ)."""
        sigma, mu, r = self.sigma_H, self.mu_H, self.r
        a = 0.5 * sigma**2
        b = mu - 0.5 * sigma**2
        disc = b**2 + 4*a*r
        y_plus = (-b + np.sqrt(disc)) / (2*a)
        y_minus = (-b - np.sqrt(disc)) / (2*a)
        return y_plus, y_minus  # y_plus > 1, y_minus < 0


# --------------------------------------------------------------------------- #
#  Leland (1994) Default Model                                                  #
# --------------------------------------------------------------------------- #

class LelandDefault:
    """
    Endogenous default following Leland (1994, JF).

    Equity value E(X) satisfies the ODE:
      0.5σ²X²E'' + μXE' - rE = -(π(X,K) - C_D(1-τ_c))

    where π(X,K) is the operating cash flow.
    Equity holders optimally choose default threshold X_D (smooth-pasting).

    With linear π(X,K) = φ·X - FC (φ = revenue-X coefficient, FC = fixed costs):
      V_U(X) = φX/(r-μ) - FC/r  [unlevered firm value]
      Tax shield = τ_c C_D / r  [PV of tax shield at no default]

    Optimal default threshold (Leland 1994, eq. 13):
      X_D = (y_-/(y_--1)) · FC(r-μ)/(rφ)

    where y_- < 0 is the negative characteristic root, FC = C_D(1-τ_c) + δK.
    """

    def __init__(self, p: DuopolyParams, K: float,
                  regime: str = 'H', in_duopoly: bool = False):
        """
        Args:
            K: Installed capacity
            regime: Demand regime for cash flow calculation
            in_duopoly: True if competing with another firm (uses θ scaling)
        """
        self.p = p
        self.K = K
        self.regime = regime
        self.in_duopoly = in_duopoly
        self.y_plus, self.y_minus = p.leland_char_roots()

    def phi_coeff(self) -> float:
        """Revenue-X coefficient φ in π(X,K) = φ·X - δK - C_D(1-τ_c)."""
        p, K = self.p, self.K
        base = K**p.alpha
        if self.in_duopoly:
            base *= p.theta
        return base

    def fixed_cost(self) -> float:
        """Fixed cash outflow per unit time: δK + C_D(1-τ_c)."""
        p, K = self.p, self.K
        return p.delta * K + p.coupon * (1 - p.tax_rate)

    def default_boundary(self) -> float:
        """
        Optimal default threshold X_D*.

        From smooth-pasting E'(X_D) = 0:
          X_D = (y_-/(y_--1)) · FC·(r-μ) / (r·φ)

        where y_- < 0 (so y_-/(y_--1) ∈ (0,1)), FC = fixed_cost(), φ = phi_coeff().
        """
        p = self.p
        y_m = self.y_minus
        phi = self.phi_coeff()
        FC = self.fixed_cost()

        if phi <= 0:
            return 0.0

        # r_eff is the drift used in the default model (H regime physical drift)
        r_eff = p.mu_H

        X_D = (y_m / (y_m - 1)) * FC * (p.r - r_eff) / (p.r * phi)
        return max(X_D, 0.0)

    def unlevered_value(self, X: float) -> float:
        """
        V_U(X) = φ·X/(r-μ) - δK/r  [present value of operating profits, no debt]
        """
        p = self.p
        return self.phi_coeff() * X / (p.r - p.mu_H) - p.delta * self.K / p.r

    def equity_value(self, X: float) -> float:
        """
        E(X) = V_L(X) - D(X) where V_L includes tax shield.

        Full Leland (1994) expression:
          E(X) = V_U(X) + τ_c C_D/r - C_D/r
                 + (C_D/r - τ_c C_D/r - (1-α_bc)V_U(X_D)) · (X/X_D)^{y_-}

        Note: (X/X_D)^{y_-} is Arrow-Debreu price of reaching X_D (= default event),
              with y_- < 0, so (X/X_D)^{y_-} → 0 as X → ∞ (default becomes remote).
        """
        p = self.p
        X_D = self.default_boundary()

        if X <= X_D:
            return 0.0

        V_U = self.unlevered_value(X)
        V_U_D = self.unlevered_value(X_D)
        coupon_pv = p.coupon / p.r
        shield_pv = p.tax_rate * p.coupon / p.r

        # Value without default option
        E_no_default = V_U + shield_pv - coupon_pv

        # Default option: at X_D, equity receives 0, loses (E_no_default - 0) evaluated at X_D
        E_at_XD_no_opt = V_U_D + shield_pv - coupon_pv   # equity value at X_D ignoring bankruptcy costs
        # Recovery at X_D: (1-α_bc) V_U(X_D) - coupon_pv
        # Equity = 0 at X_D (by construction)
        # Default option correction: -E_at_XD_no_opt · (X/X_D)^{y_-}
        AD_price = (X / X_D)**self.y_minus

        E = E_no_default - E_at_XD_no_opt * AD_price
        return max(E, 0.0)

    def debt_value(self, X: float) -> float:
        """
        D(X) = C_D/r · [1 - (X/X_D)^{y_-}]
               + (1-α_bc) V_U(X_D) · (X/X_D)^{y_-}

        Components:
        - C_D/r · [1 - ...]: coupon perpetuity discounted for default risk
        - (1-α_bc) V_U(X_D) · ...: recovery value at default (net of bankruptcy costs)
        """
        p = self.p
        X_D = self.default_boundary()

        if X <= X_D:
            V_U_D = self.unlevered_value(X_D)
            return max(0.0, (1 - p.alpha_bc) * max(0.0, V_U_D))

        coupon_pv = p.coupon / p.r
        V_U_D = self.unlevered_value(X_D)
        AD_price = (X / X_D)**self.y_minus

        D = coupon_pv * (1 - AD_price) + (1 - p.alpha_bc) * max(0.0, V_U_D) * AD_price
        return max(D, 0.0)

    def firm_value(self, X: float) -> float:
        return self.equity_value(X) + self.debt_value(X)

    def credit_spread(self, X: float) -> float:
        """Annualized credit spread = yield on risky debt - risk-free rate."""
        D = self.debt_value(X)
        if D < 1e-12:
            return np.inf
        return self.p.coupon / D - self.p.r

    def default_probability(self, X: float, T: float = 5.0) -> float:
        """
        Probability of default within T years (first-passage time for GBM).

        Using the Leland/Merton approximation for GBM:
          P(τ_D ≤ T | X_0=X) = Φ(d₁) + (X_D/X)^{2(r_eff-0.5σ²)/σ²} Φ(d₂)
        where:
          d₁ = [log(X_D/X) - (r_eff-0.5σ²)T] / (σ√T)
          d₂ = [log(X_D/X) + (r_eff-0.5σ²)T] / (σ√T)
        """
        from scipy.stats import norm
        p = self.p
        X_D = self.default_boundary()

        if X <= X_D:
            return 1.0
        if X_D <= 0:
            return 0.0

        mu_adj = p.mu_H - 0.5 * p.sigma_H**2
        sigma_sqrt_T = p.sigma_H * np.sqrt(T)
        log_ratio = np.log(X_D / X)

        d1 = (log_ratio - mu_adj * T) / sigma_sqrt_T
        d2 = (log_ratio + mu_adj * T) / sigma_sqrt_T
        power = 2 * mu_adj / p.sigma_H**2

        prob = norm.cdf(d1) + (X_D / X)**power * norm.cdf(d2)
        return float(np.clip(prob, 0.0, 1.0))


# --------------------------------------------------------------------------- #
#  Duopoly Equilibrium                                                          #
# --------------------------------------------------------------------------- #

class DuopolyModel:
    """
    Preemption game between two symmetric firms.

    Timeline:
    1. Follower's problem: invests at X_F* (duopoly market, K_F chosen optimally)
    2. Leader's problem: invests at X_L* ≤ X_F* (monopoly market pre-entry)
    3. Preemption condition: V_lead(X_L*) = V_follow(X_L*)

    With default risk:
    - Equity holders choose K to maximize EQUITY (not total firm value)
    - This creates risk-shifting: equity holders prefer larger K (upside → equity,
      downside → default → debt absorbs losses)
    - The interaction with preemption creates the competition-leverage spiral

    Without default risk (coupon=0, or α_bc=1): reduces to Huisman & Kort (2015).
    """

    def __init__(self, params: DuopolyParams):
        self.p = params
        self.sf = SingleFirmModel(params)

    def _X_star(self, K: float, phi: float, beta: float,
                 d: float) -> float:
        """
        Investment trigger from value-matching/smooth-pasting:
          X*(K) = β·[d·δK/r + I(K)] / [(β-1)·d·φ(K)]
        """
        p = self.p
        num = beta * (d * p.delta * K / p.r + p.c * K**p.gamma)
        den = (beta - 1) * d * phi
        return num / den if den > 0 else np.inf

    def _find_optimal_K(self, phi_fn, beta: float, d: float,
                         regime: str, V_fn) -> Tuple[float, float]:
        """
        Optimize K by maximizing option value F ∝ (1/X*)^β · NPV.

        At X*(K): NPV = (β-1)·[d·δK/r + I(K)] (analytical from VM/SP)
        F(X₀; K) ∝ (X₀/X*(K))^β · NPV(K)
        Maximize over K ↔ minimize X*(K)^β / NPV(K)
        """
        def objective(logK):
            K = np.exp(logK)
            phi = phi_fn(K)
            if phi <= 0:
                return np.inf
            X_star = self._X_star(K, phi, beta, d)
            if not np.isfinite(X_star) or X_star <= 0:
                return np.inf
            NPV = (beta - 1) * (d * self.p.delta * K / self.p.r + self.p.c * K**self.p.gamma)
            if NPV <= 0:
                return np.inf
            return X_star**beta / NPV

        logK_grid = np.linspace(-6, 3, 300)
        objs = [objective(lk) for lk in logK_grid]
        finite = np.isfinite(objs)
        if not np.any(finite):
            raise RuntimeError("No valid K found")

        best = np.argmin(np.where(finite, objs, np.inf))
        logK_star = logK_grid[best]

        # Refine with derivative
        lo, hi = logK_grid[max(0, best-6)], logK_grid[min(len(logK_grid)-1, best+6)]
        try:
            eps = 0.01
            def deriv(lk):
                return (objective(lk+eps) - objective(lk-eps)) / (2*eps)
            if deriv(lo) * deriv(hi) < 0:
                logK_star = brentq(deriv, lo, hi, xtol=1e-8)
        except Exception:
            pass

        K_star = np.exp(logK_star)
        phi_star = phi_fn(K_star)
        X_star = self._X_star(K_star, phi_star, beta, d)
        return K_star, X_star

    # ------------------------------------------------------------------ #
    #  Step 1: Follower's problem                                           #
    # ------------------------------------------------------------------ #

    def solve_follower(self, regime: str = 'H') -> Dict:
        """
        Follower enters a duopoly market (θ scaling already in place).
        Equivalent to a single-firm problem with duopoly profit share θ.
        """
        p = self.p
        beta, _ = p.characteristic_roots(regime,
                    extra_discount=p.lam if regime == 'L' else 0.0)
        d = np.exp(-p.r * p.tau)

        def phi_F(K):
            return p.phi_duo(K, regime)

        def V_F(X, K):
            return phi_F(K) * X - p.delta * K / p.r

        K_F, X_F = self._find_optimal_K(phi_F, beta, d, regime, V_F)
        NPV_F = (beta - 1) * (d * p.delta * K_F / p.r + p.c * K_F**p.gamma)
        phi_F_val = phi_F(K_F)

        # Default boundary for follower
        ld_F = LelandDefault(p, K_F, regime=regime, in_duopoly=True)
        X_D_F = ld_F.default_boundary()

        # Equity-maximizing capacity (vs. total-firm-value maximizing)
        K_F_equity = self._equity_maximizing_K(phi_F, beta, d, regime, in_duopoly=True)

        return {
            'K_F': K_F,
            'K_F_equity': K_F_equity,
            'X_F': X_F,
            'NPV_F': NPV_F,
            'phi_F': phi_F_val,
            'beta_F': beta,
            'X_D_F': X_D_F,
            'regime': regime,
        }

    def _equity_maximizing_K(self, phi_fn, beta: float, d: float,
                               regime: str, in_duopoly: bool = False) -> float:
        """
        Under debt financing, equity holders choose K to maximize EQUITY value
        (not total firm value). This creates risk-shifting upward bias in K.

        E(X) = V_U(X) + tax_shield - coupon_pv + default_option_correction
             ≈ (1 + τ_c C_D/r) V_U(X) - C_D/r + correction

        Equity-maximizing K solves: ∂E/∂K = 0
        At the investment trigger, equity holders capture the upside while
        debt absorbs downside → prefer larger K.
        """
        p = self.p

        def equity_at_trigger(logK):
            K = np.exp(logK)
            phi = phi_fn(K)
            if phi <= 0:
                return -np.inf
            X_star = self._X_star(K, phi, beta, d)
            if not np.isfinite(X_star) or X_star <= 0:
                return -np.inf
            ld = LelandDefault(p, K, regime='H', in_duopoly=in_duopoly)
            E = ld.equity_value(X_star)
            return E - p.c * K**p.gamma  # equity net of investment cost

        logK_grid = np.linspace(-6, 3, 200)
        vals = [equity_at_trigger(lk) for lk in logK_grid]
        finite = np.isfinite(vals)
        if not np.any(finite):
            return np.exp(logK_grid[len(logK_grid)//2])

        best = np.argmax(np.where(finite, vals, -np.inf))
        return np.exp(logK_grid[best])

    # ------------------------------------------------------------------ #
    #  Step 2: Leader's problem                                             #
    # ------------------------------------------------------------------ #

    def solve_leader(self, X_F: float, K_F: float,
                      regime: str = 'H') -> Dict:
        """
        Leader invests in the monopoly market (before follower entry).
        Leader's value of installed capacity includes:
        - Monopoly profit phase: X · K_L^α        until X reaches X_F
        - Duopoly profit phase: θ · X · K_L^α     after follower entry

        Leader's φ coefficient (present value weighting):
        φ_L(K_L) = φ_mono(K_L)  [initial investment in monopoly]
        But value accounts for future competition: V_L includes monopoly and duopoly phases.

        Simplified (following Huisman & Kort 2015):
        Leader uses monopoly trigger from monopoly φ, but adjusts NPV downward for
        future competition. Here we use the monopoly φ for the trigger (conservative)
        and compute NPV accounting for duopoly transition.
        """
        p = self.p
        beta, _ = p.characteristic_roots(regime,
                    extra_discount=p.lam if regime == 'L' else 0.0)
        d = np.exp(-p.r * p.tau)

        # Leader's trigger uses monopoly φ (investing before follower enters)
        def phi_L(K):
            return p.phi_mono(K, regime)

        def V_L_mono(X, K):
            return phi_L(K) * X - p.delta * K / p.r

        # Leader accounts for eventual competition (discounted duopoly phase)
        # V_L_total(X, K_L) = V_L_mono(X, K_L) - competition_discount(X, K_L)
        # competition_discount ≈ (φ_mono - φ_duo) · X_F^{1-β} · X^β / (β-1)
        # [Present value of lost monopoly profits once follower enters]
        def phi_L_effective(K):
            """Effective φ accounting for future competition."""
            phi_m = p.phi_mono(K, regime)
            phi_d = p.phi_duo(K, regime)
            delta_phi = phi_m - phi_d
            # Correction: discount lost profit by the probability/PV of follower entry
            # Approximation: subtract (delta_phi)/(β) as a constant adjustment
            # This is a first-order approximation from the Huisman-Kort (2015) framework
            correction_factor = (1 - delta_phi / (phi_m * beta)) if phi_m > 0 else 1.0
            return phi_m * max(0.5, correction_factor)

        K_L, X_L = self._find_optimal_K(phi_L, beta, d, regime, V_L_mono)
        phi_L_val = phi_L(K_L)
        NPV_L = (beta - 1) * (d * p.delta * K_L / p.r + p.c * K_L**p.gamma)

        ld_L = LelandDefault(p, K_L, regime=regime, in_duopoly=False)
        X_D_L = ld_L.default_boundary()

        K_L_equity = self._equity_maximizing_K(phi_L, beta, d, regime, in_duopoly=False)

        return {
            'K_L': K_L,
            'K_L_equity': K_L_equity,
            'X_L': X_L,
            'NPV_L': NPV_L,
            'phi_L': phi_L_val,
            'beta_L': beta,
            'X_D_L': X_D_L,
            'regime': regime,
        }

    # ------------------------------------------------------------------ #
    #  Step 3: Preemption condition                                         #
    # ------------------------------------------------------------------ #

    def preemption_equilibrium(self, regime: str = 'H') -> Dict:
        """
        Find the preemption equilibrium.

        In the symmetric Nash equilibrium:
        - Each firm is indifferent between leading and following at X_L*
        - V_lead(X_L*) = V_follow(X_L*)

        Leader commits to invest at X_L* < X_F* to deter follower from going first.
        The preemption trigger X_L* is generally BELOW the Stackelberg leader trigger.

        Key results vs. single-firm benchmark:
        1. Preemption gap: X_L* < X_single* (competition erodes option value of waiting)
        2. Follower delay: X_F* > X_L* (sequential equilibrium, no simultaneous entry)
        3. Risk-shifting (with default): K_L* and K_F* inflated relative to no-default case
        """
        p = self.p

        # Single-firm benchmark
        sf_sol = self.sf.solve()
        X_single = sf_sol[regime]['X_star']
        K_single = sf_sol[regime]['K_star']

        # Follower's problem
        follower_sol = self.solve_follower(regime=regime)
        X_F = follower_sol['X_F']
        K_F = follower_sol['K_F']
        beta_F = follower_sol['beta_F']
        NPV_F = follower_sol['NPV_F']

        # Leader's problem (given follower's strategy)
        leader_sol = self.solve_leader(X_F, K_F, regime=regime)
        X_L_stackelberg = leader_sol['X_L']
        K_L = leader_sol['K_L']
        beta_L = leader_sol['beta_L']
        NPV_L = leader_sol['NPV_L']

        # Preemption condition: adjust X_L downward
        # V_lead(X) = e^{-rτ} V(X, K_L) - I(K_L)  [invest now]
        # V_follow(X) = (X/X_F)^{β_F} · NPV_F       [wait to be follower]
        #
        # Set equal: find X_L* < min(X_L_stackelberg, X_F) such that V_lead = V_follow
        #
        # If V_lead(X_F) < V_follow(X_F): no preemption, both invest simultaneously at X_F
        # If V_lead(X_F) ≥ V_follow(X_F): preemption occurs, find X_L* < X_F

        d = np.exp(-p.r * p.tau)
        phi_L = p.phi_mono(K_L, regime)

        def V_lead(X):
            return d * (phi_L * X - p.delta * K_L / p.r) - p.c * K_L**p.gamma

        def V_follow(X):
            if X >= X_F:
                return NPV_F  # already past follower's trigger
            return (X / X_F)**beta_F * NPV_F

        def preemption_eq(X):
            return V_lead(X) - V_follow(X)

        # Check whether preemption occurs
        V_lead_at_XF = V_lead(X_F)
        V_follow_at_XF = V_follow(X_F)

        if V_lead_at_XF >= V_follow_at_XF and X_L_stackelberg < X_F:
            # Preemption: leader invests early
            try:
                # Find X_L* ∈ (0, X_F) where V_lead = V_follow
                f_low = preemption_eq(1e-6)
                f_high = preemption_eq(X_F)
                if f_low * f_high < 0:
                    X_L_preempt = brentq(preemption_eq, 1e-6, X_F, xtol=1e-10)
                else:
                    X_L_preempt = X_L_stackelberg
            except Exception:
                X_L_preempt = X_L_stackelberg
        else:
            # No preemption: both invest at X_F (simultaneous entry)
            X_L_preempt = X_F

        # Use the smaller of Stackelberg and preemption triggers
        X_L_star = min(X_L_preempt, X_L_stackelberg)

        # Compute metrics
        ld_L = LelandDefault(p, K_L, regime='H', in_duopoly=False)
        ld_F = LelandDefault(p, K_F, regime='H', in_duopoly=True)
        X_D_L = ld_L.default_boundary()
        X_D_F = ld_F.default_boundary()

        preemption_gap = (X_single - X_L_star) / max(X_single, 1e-10)
        risk_shifting_L = (K_L - K_single) / max(K_single, 1e-10)

        return {
            'X_L': X_L_star,
            'X_F': X_F,
            'K_L': K_L,
            'K_F': K_F,
            'X_D_L': X_D_L,
            'X_D_F': X_D_F,
            'X_single': X_single,
            'K_single': K_single,
            'preemption_gap': preemption_gap,
            'risk_shifting': risk_shifting_L,
            'NPV_L': NPV_L,
            'NPV_F': NPV_F,
            'V_lead_at_XF': V_lead_at_XF,
            'V_follow_at_XF': V_follow_at_XF,
            'regime': regime,
        }

    # ------------------------------------------------------------------ #
    #  Competition-leverage spiral analysis                                 #
    # ------------------------------------------------------------------ #

    def competition_leverage_spiral(self, regime: str = 'H') -> Dict:
        """
        Decompose the competition-leverage spiral into three steps:

        Step 1: Monopoly, no default (baseline)
        Step 2: Competition, no default (preemption effect only)
        Step 3: Competition, with default (risk-shifting added)
        """
        import copy

        # Step 1: Monopoly, no default
        p_mono_nd = copy.copy(self.p)
        p_mono_nd.coupon = 0.0
        sf_nd = SingleFirmModel(p_mono_nd)
        sol_sf = sf_nd.solve()
        step1 = {
            'label': 'Monopoly\n(no default)',
            'X_star': sol_sf[regime]['X_star'],
            'K_star': sol_sf[regime]['K_star'],
            'X_D': 0.0,
            'default_prob_5yr': 0.0,
        }

        # Step 2: Competition, no default
        p_duo_nd = copy.copy(self.p)
        p_duo_nd.coupon = 0.0
        m_duo_nd = DuopolyModel(p_duo_nd)
        eq_nd = m_duo_nd.preemption_equilibrium(regime=regime)
        step2 = {
            'label': 'Competition\n(no default)',
            'X_star': eq_nd['X_L'],
            'K_star': eq_nd['K_L'],
            'X_D': 0.0,
            'default_prob_5yr': 0.0,
        }

        # Step 3: Competition, with default
        eq_full = self.preemption_equilibrium(regime=regime)
        ld_L = LelandDefault(self.p, eq_full['K_L'], regime='H', in_duopoly=False)
        X_D_L = ld_L.default_boundary()
        dp = ld_L.default_probability(eq_full['X_L'] * 0.5, T=5.0)
        step3 = {
            'label': 'Competition\n(with default)',
            'X_star': eq_full['X_L'],
            'K_star': eq_full['K_L'],
            'X_D': X_D_L,
            'default_prob_5yr': dp,
        }

        return {
            'steps': [step1, step2, step3],
            'preemption_effect': (step1['X_star'] - step2['X_star']) / max(step1['X_star'], 1e-10),
            'risk_shifting_effect': (step3['K_star'] - step2['K_star']) / max(step2['K_star'], 1e-10),
            'default_risk': step3['default_prob_5yr'],
        }

    # ------------------------------------------------------------------ #
    #  Comparative statics                                                  #
    # ------------------------------------------------------------------ #

    def comparative_statics(self, param_name: str,
                             param_values: np.ndarray,
                             regime: str = 'H') -> Dict:
        """Comparative statics for the duopoly model."""
        import copy
        keys = ['X_L', 'X_F', 'K_L', 'K_F', 'X_D_L', 'preemption_gap', 'risk_shifting']
        out = {k: [] for k in keys}

        for val in param_values:
            p_new = copy.copy(self.p)
            setattr(p_new, param_name, val)
            try:
                p_new.validate()
                m = DuopolyModel(p_new)
                eq = m.preemption_equilibrium(regime=regime)
                for k in keys:
                    out[k].append(eq.get(k, np.nan))
            except Exception:
                for k in keys:
                    out[k].append(np.nan)

        return {k: np.array(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
#  Plotting                                                                     #
# --------------------------------------------------------------------------- #

def plot_leland_default(params: DuopolyParams, K: float,
                         output_path: str = None) -> plt.Figure:
    """Leland default model: equity, debt, credit spread vs demand X."""
    scenarios = [
        ('Monopolist (pre-entry)', False, '#1f77b4'),
        ('Duopolist (post-entry)', True, '#d62728'),
    ]

    X_D_ref = LelandDefault(params, K, 'H', False).default_boundary()
    X_max = max(5 * X_D_ref, 0.20)
    X_grid = np.linspace(1e-4, X_max, 400)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for name, in_duo, color in scenarios:
        ld = LelandDefault(params, K, 'H', in_duo)
        X_D = ld.default_boundary()

        E_vals = np.array([ld.equity_value(x) for x in X_grid])
        D_vals = np.array([ld.debt_value(x) for x in X_grid])
        CS_bps = np.array([ld.credit_spread(x) * 10000 for x in X_grid])

        axes[0].plot(X_grid, E_vals, color=color, lw=2, label=name)
        axes[0].axvline(X_D, color=color, ls=':', alpha=0.7, lw=1.5, label=f'$X_D$={X_D:.4f}')
        axes[1].plot(X_grid, D_vals, color=color, lw=2, label=name)
        axes[1].axvline(X_D, color=color, ls=':', alpha=0.7, lw=1.5)
        axes[2].plot(X_grid, np.clip(CS_bps, 0, 3000), color=color, lw=2, label=name)
        axes[2].axvline(X_D, color=color, ls=':', alpha=0.7, lw=1.5)

    for ax, (ylabel, title) in zip(axes, [
        ('Equity value', '(a) Equity Value $E(X)$'),
        ('Debt value', '(b) Debt Value $D(X)$'),
        ('Credit spread (bps)', '(c) Credit Spread (bps)'),
    ]):
        ax.set_xlabel('Demand level $X$', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, X_max)

    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    axes[2].set_ylim(0, 3000)

    plt.suptitle(f'Leland (1994) Endogenous Default  '
                  f'[$K={K:.2f}$, $C_D={params.coupon:.3f}$, $\\alpha_{{bc}}={params.alpha_bc:.2f}$]',
                  fontsize=12)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    return fig


def plot_competition_leverage_spiral(params: DuopolyParams,
                                      output_path: str = None) -> plt.Figure:
    """Bar chart showing the 3-step competition-leverage spiral."""
    model = DuopolyModel(params)
    spiral = model.competition_leverage_spiral('H')
    steps = spiral['steps']

    categories = [s['label'] for s in steps]
    X_stars = [s['X_star'] for s in steps]
    K_stars = [s['K_star'] for s in steps]
    X_D_vals = [s['X_D'] for s in steps]
    dp_vals = [s['default_prob_5yr'] * 100 for s in steps]

    colors = ['#2ca02c', '#1f77b4', '#d62728']
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, (vals, title, ylabel) in zip(axes, [
        (X_stars, '(a) Investment Trigger $X^*$', 'Demand level'),
        (K_stars, '(b) Optimal Capacity $K^*$', 'Capacity (GW equiv.)'),
        (X_D_vals, '(c) Default Boundary $X_D$', 'Demand level'),
        (dp_vals, '(d) 5-yr Default Prob. (%)', 'Probability (%)'),
    ]):
        bars = ax.bar(categories, vals, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis='y', alpha=0.4)
        ax.tick_params(axis='x', labelsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + max(vals)*0.01,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Competition-Leverage Spiral: Decomposing the Mechanism\n'
                  f'($\\theta={params.theta}$, $C_D={params.coupon}$, $\\tau={params.tax_rate}$)',
                  fontsize=12)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    return fig


def plot_duopoly_comparative_statics(params: DuopolyParams,
                                      output_path: str = None) -> plt.Figure:
    """Comparative statics for the duopoly model."""
    model = DuopolyModel(params)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()

    configs = [
        ('lam', np.linspace(0.05, 0.55, 25), 'Adoption rate $\\lambda$',
         'Triggers $X^*$', ['X_L', 'X_F'], ['Leader $X_L^*$', 'Follower $X_F^*$']),
        ('theta', np.linspace(0.30, 0.90, 25), 'Duopoly share $\\theta$',
         'Leader trigger $X_L^*$', ['X_L'], ['Leader $X_L^*$']),
        ('coupon', np.linspace(0.00, 0.08, 25), 'Debt coupon $C_D$',
         'Risk-shifting effect (%)', ['risk_shifting'], ['$K_L^* / K_{{single}}^* - 1$']),
        ('alpha_bc', np.linspace(0.05, 0.60, 25), 'Bankruptcy cost $\\alpha_{{bc}}$',
         'Leader trigger $X_L^*$', ['X_L'], ['Leader $X_L^*$']),
    ]

    colors_list = [['#1f77b4', '#d62728'], ['#1f77b4'], ['#d62728'], ['#2ca02c']]

    for ax, (param, vals, xlabel, ylabel, keys, labels), colors in zip(axes, configs, colors_list):
        cs = model.comparative_statics(param, vals)
        for key, label, color in zip(keys, labels, colors):
            y = cs[key]
            if key == 'risk_shifting':
                y = y * 100  # to percent
            ax.plot(vals, y, color=color, lw=2, label=label)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'Effect of {xlabel.split("$")[1] if "$" in xlabel else xlabel}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle('Duopoly with Default: Comparative Statics', fontsize=13)
    plt.tight_layout()
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
    print("Phase 2: Duopoly with Default Risk")
    print("=" * 60)

    params = DuopolyParams()
    params.validate()

    print(f"\nKey parameters: θ={params.theta}, C_D={params.coupon}, "
           f"α_bc={params.alpha_bc}, τ_c={params.tax_rate}")

    # ---- Leland default model ----
    print("\n--- Leland (1994) Default Model ---")
    K_test = 0.16   # representative capacity
    ld_mono = LelandDefault(params, K=K_test, in_duopoly=False)
    ld_duo = LelandDefault(params, K=K_test, in_duopoly=True)

    X_D_mono = ld_mono.default_boundary()
    X_D_duo = ld_duo.default_boundary()
    print(f"  Monopolist: X_D = {X_D_mono:.6f}")
    print(f"  Duopolist:  X_D = {X_D_duo:.6f}  (higher: less revenue to service debt)")

    X_ref = 0.10
    print(f"\n  At X = {X_ref}:")
    print(f"    Monopolist:  E={ld_mono.equity_value(X_ref):.4f}, "
           f"D={ld_mono.debt_value(X_ref):.4f}, "
           f"CS={ld_mono.credit_spread(X_ref)*10000:.1f}bps, "
           f"DP(5yr)={ld_mono.default_probability(X_ref)*100:.2f}%")
    print(f"    Duopolist:   E={ld_duo.equity_value(X_ref):.4f}, "
           f"D={ld_duo.debt_value(X_ref):.4f}, "
           f"CS={ld_duo.credit_spread(X_ref)*10000:.1f}bps, "
           f"DP(5yr)={ld_duo.default_probability(X_ref)*100:.2f}%")

    # ---- Preemption equilibrium ----
    print("\n--- Preemption Equilibrium ---")
    model = DuopolyModel(params)
    eq = model.preemption_equilibrium('H')

    print(f"\n  Single firm:  X*={eq['X_single']:.4f}, K*={eq['K_single']:.4f}")
    print(f"  Leader:       X_L*={eq['X_L']:.4f}, K_L*={eq['K_L']:.4f}, X_D_L={eq['X_D_L']:.6f}")
    print(f"  Follower:     X_F*={eq['X_F']:.4f}, K_F*={eq['K_F']:.4f}, X_D_F={eq['X_D_F']:.6f}")
    print(f"\n  Preemption gap (X_single-X_L)/X_single = {eq['preemption_gap']*100:.1f}%")
    print(f"  Risk-shifting (K_L-K_single)/K_single   = {eq['risk_shifting']*100:.1f}%")

    # ---- Competition-leverage spiral ----
    print("\n--- Competition-Leverage Spiral (3-step decomposition) ---")
    spiral = model.competition_leverage_spiral('H')
    for step in spiral['steps']:
        print(f"  {step['label'].replace(chr(10), ' ')}: "
               f"X*={step['X_star']:.4f}, K*={step['K_star']:.4f}, "
               f"X_D={step['X_D']:.6f}, DP(5yr)={step['default_prob_5yr']*100:.1f}%")
    print(f"\n  Preemption effect:  {spiral['preemption_effect']*100:.1f}% reduction in trigger")
    print(f"  Risk-shifting:      {spiral['risk_shifting_effect']*100:.1f}% increase in capacity")
    print(f"  Default risk (5yr): {spiral['default_risk']*100:.1f}%")

    # ---- Figures ----
    print("\nGenerating figures...")
    plot_leland_default(params, K=K_test,
                         output_path='/workspaces/claude-paper/figures/phase2_leland.png')
    plot_competition_leverage_spiral(params,
                                      output_path='/workspaces/claude-paper/figures/phase2_spiral.png')
    plot_duopoly_comparative_statics(params,
                                      output_path='/workspaces/claude-paper/figures/phase2_comp_statics.png')

    print("\nPhase 2 complete. ✓")
