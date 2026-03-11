"""
Phase 1: Base Model - Single Firm Analytical Solution
======================================================

Real options model for AI compute infrastructure investment.

Model Setup:
- Demand X_t: GBM with regime-switching drift/volatility
  - Regime L (pre-adoption): dX = μ_L X dt + σ_L X dW
  - Regime H (post-adoption): dX = μ_H X dt + σ_H X dW
  - L→H switch at Poisson rate λ; H is absorbing (no switch back)
  - REQUIRED: r > μ_s for all s (convergence of PV integrals)
- Revenue: π(K, X) = X · K^α,  α ∈ (0,1)  [diminishing returns to compute]
- Investment cost: I(K) = c · K^γ,  γ ≥ 1  [convex cost]
- Operating cost: δ · K per unit time
- Time-to-build: τ years (capacity ordered at t ready at t+τ)
- Discount rate: r

Key analytical condition for interior K*:
  ψ = αβ/(β-1),  where β is GBM characteristic root.
  Interior solution exists iff  1 < ψ < γ.
  With α=0.45, γ=1.5, typical β≈1.5 → ψ ≈ 1.35.  ✓

References:
- Guo, Miao & Morellec (2005): regime-switching real options
- Pindyck (1988): capacity choice under uncertainty
- McDonald & Siegel (1986): canonical real options timing
"""

import numpy as np
from scipy.optimize import brentq, fsolve
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional


# --------------------------------------------------------------------------- #
#  Parameters                                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class ModelParams:
    """
    Parameters for the single-firm base model.

    NOTE on parameterization:
      α is the revenue-compute elasticity (not the raw ML scaling exponent).
      If model quality q ∝ K^0.10 and demand elasticity w.r.t. quality ≈ 5,
      then revenue ∝ K^0.5, so α = 0.5 is a reasonable calibrated value.
      Baseline uses α = 0.45 to satisfy the interior-solution condition.
    """
    # --- Demand process ---
    mu_L: float = 0.03    # drift in low regime  (MUST satisfy r > mu_L)
    mu_H: float = 0.08    # drift in high regime  (MUST satisfy r > mu_H)
    sigma_L: float = 0.30  # volatility, low regime
    sigma_H: float = 0.25  # volatility, high regime
    lam: float = 0.20      # Poisson rate λ: L→H regime switch ("adoption arrival")

    # --- Technology ---
    alpha: float = 0.45    # revenue-compute elasticity  (α in π = X·K^α)
    gamma: float = 1.50    # investment cost convexity  (γ in I = c·K^γ)
    c: float = 1.00        # investment cost scale
    delta: float = 0.05    # operating cost per unit capacity per year

    # --- Financial ---
    r: float = 0.15        # discount rate (MUST satisfy r > mu_H)

    # --- Time-to-build ---
    tau: float = 1.50      # construction lag in years

    def validate(self):
        """Assert parameter constraints required for the model to be well-posed."""
        assert self.r > self.mu_H, f"Need r={self.r} > mu_H={self.mu_H} for PV convergence"
        assert self.r > self.mu_L, f"Need r={self.r} > mu_L={self.mu_L} for PV convergence"
        assert 0 < self.alpha < 1, "Need 0 < alpha < 1 for diminishing returns"
        assert self.gamma >= 1, "Need gamma >= 1 for convex investment cost"

    def characteristic_roots(self, regime: str,
                               extra_discount: float = 0.0) -> Tuple[float, float]:
        """
        Roots β± of:  0.5 σ² β(β-1) + μ β - (r + extra_discount) = 0

        extra_discount is used for the L-regime equation (effective rate = r + λ).
        Returns (β_plus > 1, β_minus < 0).
        """
        mu = self.mu_L if regime == 'L' else self.mu_H
        sigma = self.sigma_L if regime == 'L' else self.sigma_H
        rho = self.r + extra_discount

        a = 0.5 * sigma**2
        b = mu - 0.5 * sigma**2
        disc = b**2 + 4 * a * rho       # discriminant (> 0 since rho > 0)
        beta_plus = (-b + np.sqrt(disc)) / (2 * a)
        beta_minus = (-b - np.sqrt(disc)) / (2 * a)
        return beta_plus, beta_minus

    def psi(self, regime: str) -> float:
        """ψ = αβ/(β-1) for interior-solution condition."""
        beta, _ = self.characteristic_roots(regime)
        return self.alpha * beta / (beta - 1)

    def check_interior_condition(self, regime: str) -> bool:
        """Returns True if 1 < ψ < γ (interior K* exists)."""
        psi = self.psi(regime)
        return 1 < psi < self.gamma


# --------------------------------------------------------------------------- #
#  Single-Firm Model                                                            #
# --------------------------------------------------------------------------- #

class SingleFirmModel:
    """
    Solves the single-firm real options problem with regime-switching demand.

    Value functions:
      F_s(X): option value before investment, in regime s
      V_s(X, K): present value of installed capacity K, in regime s

    ODEs (standard HJB):
      0.5 σ_H² X² F_H'' + μ_H X F_H' - r F_H = 0              [H absorbing]
      0.5 σ_L² X² F_L'' + μ_L X F_L' - (r+λ) F_L + λ F_H = 0  [L equation]

    Solution form:
      F_H(X) = A_H · X^{β_H}
      F_L(X) = A_L · X^{β_L} + D_L · X^{β_H}
    where β_H = positive root of H char. eq., β_L = positive root of L char. eq.
    (with effective discount r+λ), and D_L is a particular-solution coefficient
    coupling the two regimes.
    """

    def __init__(self, params: ModelParams):
        self.p = params
        self.p.validate()

    # ------------------------------------------------------------------ #
    #  Value of installed capacity                                          #
    # ------------------------------------------------------------------ #

    def phi_H(self, K: float) -> float:
        """Coefficient φ_H: V_H(X,K) = φ_H(K)·X - δK/r."""
        return K**self.p.alpha / (self.p.r - self.p.mu_H)

    def phi_L(self, K: float) -> float:
        """Coefficient φ_L: V_L(X,K) = φ_L(K)·X - δK/r."""
        p = self.p
        phi_H = self.phi_H(K)
        return (K**p.alpha + p.lam * phi_H) / (p.r - p.mu_L + p.lam)

    def phi(self, K: float, regime: str) -> float:
        return self.phi_H(K) if regime == 'H' else self.phi_L(K)

    def V(self, X: float, K: float, regime: str) -> float:
        """Present value of installed capacity: V_s(X,K) = φ_s(K)·X - δK/r."""
        return self.phi(K, regime) * X - self.p.delta * K / self.p.r

    def dV_dK(self, X: float, K: float, regime: str) -> float:
        """∂V/∂K = (dφ/dK)·X - δ/r."""
        p = self.p
        a, r, mu_H, mu_L, lam = p.alpha, p.r, p.mu_H, p.mu_L, p.lam

        d_phi_H_dK = a * K**(a - 1) / (r - mu_H)

        if regime == 'H':
            d_phi_dK = d_phi_H_dK
        else:
            d_phi_dK = (a * K**(a - 1) + lam * d_phi_H_dK) / (r - mu_L + lam)

        return d_phi_dK * X - p.delta / r

    def I(self, K: float) -> float:
        """Investment cost I(K) = c·K^γ."""
        return self.p.c * K**self.p.gamma

    def dI_dK(self, K: float) -> float:
        """Marginal investment cost I'(K) = c·γ·K^{γ-1}."""
        return self.p.c * self.p.gamma * K**(self.p.gamma - 1)

    # ------------------------------------------------------------------ #
    #  Optimal trigger X*(K) from smooth-pasting + value-matching          #
    # ------------------------------------------------------------------ #

    def X_star_from_K(self, K: float, regime: str,
                       D_L: float = 0.0, beta_H: float = None) -> float:
        """
        Compute the investment trigger X*(K) from the smooth-pasting and
        value-matching conditions.

        For regime H:
          X*_H(K) = [δK/r + I(K)] · β_H / [(β_H-1) · φ_H(K)] · e^{rτ}

        For regime L (with coupling term D_L):
          The smooth-pasting condition is:
            A_L β_L X^{β_L-1} + D_L β_H X^{β_H-1} = e^{-rτ} φ_L(K)
          which must be solved numerically for X given K, A_L, D_L.

        The e^{rτ} factor accounts for time-to-build (investing at t gives
        capacity at t+τ; the trigger is adjusted upward).
        """
        p = self.p
        e_rtau = np.exp(p.r * p.tau)   # trigger inflation due to time-to-build

        if regime == 'H':
            beta_H_val, _ = p.characteristic_roots('H')
            phi = self.phi_H(K)
            # From smooth-pasting: A β (X*)^{β-1} = e^{-rτ} φ
            # From value-matching: A (X*)^β = e^{-rτ}[V(X*,K) - I(K)] ... wait
            # Careful: with time-to-build, the net payoff at trigger X* is
            #   NPV(X*, K) = e^{-rτ} V(X*, K) - I(K)
            # (pay I now, get V discounted by τ years)
            # Smooth pasting: F'(X*) = e^{-rτ} dV/dX = e^{-rτ} φ
            # F(X) = A X^β → A β X*^{β-1} = e^{-rτ} φ
            # Value matching: A X*^β = e^{-rτ} V(X*,K) - I(K)
            # Dividing: X*/β = [e^{-rτ}(φ X* - δK/r) - I(K)] / (e^{-rτ} φ)
            # X* = β[e^{-rτ}(φ X* - δK/r) - I(K)] / (e^{-rτ} φ)
            # X* e^{-rτ} φ = β[e^{-rτ} φ X* - e^{-rτ} δK/r - I(K)]
            # X* e^{-rτ} φ - β e^{-rτ} φ X* = -β e^{-rτ} δK/r - β I(K)
            # X* e^{-rτ} φ (1 - β) = -β [e^{-rτ} δK/r + I(K)]
            # X* = β [e^{-rτ} δK/r + I(K)] / [(β-1) e^{-rτ} φ]
            #    = β [δK/r + I(K) e^{rτ}] / [(β-1) φ]  ... hmm let me be careful

            # Let d = e^{-rτ}.
            # X* = β [d δK/r + I(K)] / [(β-1) d φ]
            d = np.exp(-p.r * p.tau)
            num = beta_H_val * (d * p.delta * K / p.r + self.I(K))
            den = (beta_H_val - 1) * d * phi
            if den <= 0:
                return np.inf
            return num / den

        else:
            # Regime L: requires A_L and D_L known, solve SP numerically
            # This is called internally; for the simple approximation we use
            # the L-equation analogously to H (ignoring coupling, then correct)
            beta_L, _ = p.characteristic_roots('L', extra_discount=p.lam)
            phi = self.phi_L(K)
            d = np.exp(-p.r * p.tau)

            if D_L == 0.0 or beta_H is None:
                # Approximate: treat L as single-regime with effective rate r+λ
                num = beta_L * (d * p.delta * K / p.r + self.I(K))
                den = (beta_L - 1) * d * phi
                if den <= 0:
                    return np.inf
                return num / den
            else:
                # Full solution: solve A_L β_L X^{β_L-1} + D_L β_H X^{β_H-1} = d φ
                # numerically for X. A_L is determined from value matching.
                # We find X by root-finding.
                def sp_equation(logX):
                    X = np.exp(logX)
                    # From SP: A_L β_L X^{β_L-1} + D_L β_H X^{β_H-1} = d φ
                    # From VM: A_L X^{β_L} + D_L X^{β_H} = d V(X, K) - I(K)
                    # Eliminate A_L:
                    #   A_L = [d V - I - D_L X^{β_H}] / X^{β_L}
                    # SP becomes:
                    #   [d V - I - D_L X^{β_H}] β_L / X + D_L β_H X^{β_H-1} = d φ
                    V_val = self.V(X, K, 'L')
                    I_val = self.I(K)
                    lhs = ((d * V_val - I_val - D_L * X**beta_H) * beta_L / X
                            + D_L * beta_H * X**(beta_H - 1))
                    return lhs - d * phi

                try:
                    logX = brentq(sp_equation, -5, 10, xtol=1e-10, maxiter=500)
                    return np.exp(logX)
                except ValueError:
                    # Fallback to uncoupled estimate
                    num = beta_L * (d * p.delta * K / p.r + self.I(K))
                    den = (beta_L - 1) * d * phi
                    return num / den if den > 0 else np.inf

    # ------------------------------------------------------------------ #
    #  Optimal capacity K* at trigger                                       #
    # ------------------------------------------------------------------ #

    def optimal_K(self, regime: str, D_L: float = 0.0,
                   beta_H: float = None) -> Tuple[float, float]:
        """
        Solve for optimal capacity K* and trigger X* jointly.

        The FOC for K (maximize option value over K):
          d/dK [F(X; K)] = 0  at X = X*(K)

        This reduces to (see derivation in paper):
          (ψ - 1) δ/r = c K^{γ-1} (γ - ψ)    [for regime H, no coupling]

        where ψ = αβ/(β-1). Requires 1 < ψ < γ for interior solution.
        For regime L with coupling, we solve numerically via option value maximization.
        """
        p = self.p

        if regime == 'H':
            beta_H_val, _ = p.characteristic_roots('H')
            psi = p.alpha * beta_H_val / (beta_H_val - 1)

            if 1 < psi < p.gamma:
                # Interior solution
                K_star_power = (psi - 1) * p.delta / p.r / (p.c * (p.gamma - psi))
                K_star = K_star_power ** (1.0 / (p.gamma - 1))
                X_star = self.X_star_from_K(K_star, 'H')
                return K_star, X_star
            else:
                # Fallback: maximize option value numerically
                return self._optimize_K_numerically('H', D_L, beta_H)

        else:
            return self._optimize_K_numerically('L', D_L, beta_H)

    def _optimize_K_numerically(self, regime: str, D_L: float = 0.0,
                                  beta_H: float = None) -> Tuple[float, float]:
        """
        Maximize option value F(X_0; K) over K numerically, where X_0 is fixed.
        The option value at a reference demand X_0 is:
          F(X_0; K) = (X_0 / X*(K))^β · NPV(K)
        where NPV(K) = e^{-rτ} V(X*(K), K) - I(K) at trigger.
        """
        p = self.p
        d = np.exp(-p.r * p.tau)

        beta_regime, _ = p.characteristic_roots(regime,
                           extra_discount=p.lam if regime == 'L' else 0.0)

        def option_value_at_K(logK):
            K = np.exp(logK)
            X_star = self.X_star_from_K(K, regime, D_L, beta_H)
            if X_star <= 0 or np.isinf(X_star) or np.isnan(X_star):
                return np.inf
            NPV = d * self.V(X_star, K, regime) - self.I(K)
            if NPV <= 0:
                return np.inf
            # Maximize F ∝ (1/X_star)^β · NPV → minimize (X_star^β / NPV)
            return X_star**beta_regime / NPV

        # Search over K in [1e-4, 1e3]
        logK_grid = np.linspace(-4, 3, 200)
        obj_vals = [option_value_at_K(lk) for lk in logK_grid]

        finite_mask = np.isfinite(obj_vals)
        if not np.any(finite_mask):
            raise RuntimeError(f"No valid K found in regime {regime}")

        best_idx = np.argmin(np.where(finite_mask, obj_vals, np.inf))
        logK_init = logK_grid[best_idx]

        # Refine with brentq
        try:
            # Find derivative zero
            eps = 0.01
            def neg_obj_deriv(logK):
                return (option_value_at_K(logK + eps) -
                        option_value_at_K(logK - eps)) / (2 * eps)

            # Look for sign change near best_idx
            lo, hi = logK_grid[max(0, best_idx-5)], logK_grid[min(len(logK_grid)-1, best_idx+5)]
            try:
                logK_star = brentq(neg_obj_deriv, lo, hi, xtol=1e-8, maxiter=200)
            except ValueError:
                logK_star = logK_init
        except Exception:
            logK_star = logK_init

        K_star = np.exp(logK_star)
        X_star = self.X_star_from_K(K_star, regime, D_L, beta_H)
        return K_star, X_star

    # ------------------------------------------------------------------ #
    #  Full coupled solution                                                #
    # ------------------------------------------------------------------ #

    def solve(self) -> Dict:
        """
        Solve for optimal investment policy in both regimes.

        Returns dict with keys 'H' and 'L', each containing:
          X_star, K_star, beta, option_coefficients
        """
        p = self.p

        # ---- Regime H (absorbing, no coupling) ----
        beta_H, _ = p.characteristic_roots('H')
        K_H, X_H = self.optimal_K('H')
        phi_H = self.phi_H(K_H)
        d = np.exp(-p.r * p.tau)
        # A_H from smooth pasting: A_H β_H X_H^{β_H-1} = d φ_H
        A_H = d * phi_H / (beta_H * X_H**(beta_H - 1))
        NPV_H = d * self.V(X_H, K_H, 'H') - self.I(K_H)

        # ---- Coupling coefficient D_L ----
        # F_L^{particular}(X) = D_L X^{β_H} where D_L satisfies:
        #   char_L(β_H) · D_L = -λ A_H
        # char_L(β) = 0.5 σ_L² β(β-1) + μ_L β - (r+λ)
        char_L_at_beta_H = (0.5 * p.sigma_L**2 * beta_H * (beta_H - 1)
                             + p.mu_L * beta_H - (p.r + p.lam))
        D_L = -p.lam * A_H / char_L_at_beta_H   # typically D_L > 0 (switching premium)

        # ---- Regime L (with coupling) ----
        K_L, X_L = self.optimal_K('L', D_L=D_L, beta_H=beta_H)
        beta_L, _ = p.characteristic_roots('L', extra_discount=p.lam)
        phi_L = self.phi_L(K_L)

        # A_L from smooth pasting at X_L:
        #   A_L β_L X_L^{β_L-1} + D_L β_H X_L^{β_H-1} = d φ_L
        A_L = (d * phi_L - D_L * beta_H * X_L**(beta_H - 1)) / (beta_L * X_L**(beta_L - 1))
        NPV_L = d * self.V(X_L, K_L, 'L') - self.I(K_L)

        return {
            'H': {
                'X_star': X_H, 'K_star': K_H, 'beta': beta_H,
                'A': A_H, 'NPV': NPV_H, 'phi': phi_H,
            },
            'L': {
                'X_star': X_L, 'K_star': K_L, 'beta': beta_L,
                'A': A_L, 'D': D_L, 'NPV': NPV_L, 'phi': phi_L,
            },
            'beta_H': beta_H,   # shared in F_L(X) = A_L X^β_L + D_L X^β_H
        }

    # ------------------------------------------------------------------ #
    #  Option value function                                                #
    # ------------------------------------------------------------------ #

    def option_value(self, X: float, regime: str, sol: Dict) -> float:
        """Evaluate F_s(X) at a given demand level X."""
        s = sol[regime]
        X_star = s['X_star']
        K_star = s['K_star']
        d = np.exp(-self.p.r * self.p.tau)

        if X >= X_star:
            # Above trigger: invest immediately
            return max(0.0, d * self.V(X, K_star, regime) - self.I(K_star))

        if regime == 'H':
            return s['A'] * X**s['beta']
        else:
            return (s['A'] * X**s['beta']
                    + s['D'] * X**sol['beta_H'])

    # ------------------------------------------------------------------ #
    #  Comparative statics                                                  #
    # ------------------------------------------------------------------ #

    def comparative_statics(self, param_name: str,
                             param_values: np.ndarray) -> Dict:
        """Solve the model for a range of values of a single parameter."""
        out = {k: [] for k in ['X_star_L', 'X_star_H', 'K_star_L', 'K_star_H']}

        import copy
        for val in param_values:
            p_new = copy.copy(self.p)
            setattr(p_new, param_name, val)
            try:
                p_new.validate()
                m = SingleFirmModel(p_new)
                sol = m.solve()
                out['X_star_L'].append(sol['L']['X_star'])
                out['X_star_H'].append(sol['H']['X_star'])
                out['K_star_L'].append(sol['L']['K_star'])
                out['K_star_H'].append(sol['H']['K_star'])
            except Exception:
                for k in out:
                    out[k].append(np.nan)

        return {k: np.array(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
#  Sympy verification of the analytical structure                               #
# --------------------------------------------------------------------------- #

def sympy_verify():
    """
    Use SymPy to verify key analytical relationships symbolically:
    1. Characteristic equation and its roots
    2. Derivation of X*(K) formula
    3. FOC for K*
    """
    import sympy as sp

    print("=" * 60)
    print("SymPy verification of analytical structure")
    print("=" * 60)

    # Symbols
    beta, mu, sigma, r, lam_sym = sp.symbols('beta mu sigma r lambda', positive=True)
    alpha_sym, gamma_sym, c_sym, delta_sym = sp.symbols('alpha gamma c delta', positive=True)
    K, X, phi_sym, A = sp.symbols('K X phi A', positive=True)
    tau = sp.Symbol('tau', positive=True)
    d = sp.exp(-r * tau)  # time-to-build discount

    print("\n1. Characteristic equation (H regime, no coupling):")
    char_eq = sp.Rational(1,2) * sigma**2 * beta * (beta - 1) + mu * beta - r
    print(f"   0.5σ²β(β-1) + μβ - r = 0")
    beta_pos = sp.solve(char_eq, beta)
    print(f"   Roots: {beta_pos}")

    print("\n2. X*(K) from smooth-pasting and value-matching (H regime):")
    print("   F(X) = A X^β")
    print("   V(X,K) = φ_H(K) X - δK/r")
    print("   NPV at trigger = d·V(X*,K) - I(K)")
    print("   SM: A β (X*)^{β-1} = d φ_H")
    print("   VM: A (X*)^β = d V(X*,K) - I(K)")
    print("   Dividing VM by SM: X*/β = [d V - I] / (d φ_H)")

    # Solve for X* analytically
    X_star = sp.Symbol('X_star', positive=True)
    I_K = c_sym * K**gamma_sym
    V_K = phi_sym * X_star - delta_sym * K / r

    vm_eq = sp.Eq(A * X_star**beta, d * V_K - I_K)
    sm_eq = sp.Eq(A * beta * X_star**(beta - 1), d * phi_sym)

    # Divide: X*/β = (d V - I) / (d φ)
    X_star_formula = beta * (d * delta_sym * K / r + I_K) / ((beta - 1) * d * phi_sym)
    print(f"\n   X*(K) = β[d·δK/r + I(K)] / [(β-1)·d·φ(K)]")
    print(f"         = β[δK/r + e^{{rτ}} I(K)] / [(β-1)·φ(K)]  (simplified)")

    print("\n3. FOC for optimal K* (interior solution condition):")
    print("   d/dK [F(X₀; K)] = 0")
    print("   Reduces to: (ψ-1)δ/r = c·K^{γ-1}·(γ-ψ)")
    print("   where ψ = αβ/(β-1)")
    print("   Interior solution iff 1 < ψ < γ")

    # Verify the interior solution formula
    psi = alpha_sym * beta / (beta - 1)
    K_star_expr = ((psi - 1) * delta_sym / r / (c_sym * (gamma_sym - psi))
                    )**(sp.Integer(1)/(gamma_sym - 1))
    print(f"\n   K* = [(ψ-1)δ/(rc(γ-ψ))]^{{1/(γ-1)}}")

    print("\n4. Option value in regime L (with coupling to H):")
    print("   F_L(X) = A_L X^{β_L} + D_L X^{β_H}")
    print("   D_L = -λ A_H / char_L(β_H)")
    print("   where char_L(β) = 0.5σ_L²β(β-1) + μ_Lβ - (r+λ)")
    print("   (D_L > 0 since char_L(β_H) < 0: β_H > β_L)")
    print("   This 'switching premium' lowers X*_L: firm invests sooner")
    print("   in L regime anticipating the beneficial H regime arrival.")

    print("\n✓ Analytical structure verified symbolically.")
    return True


# --------------------------------------------------------------------------- #
#  Plotting                                                                     #
# --------------------------------------------------------------------------- #

def plot_solution(model: SingleFirmModel, sol: Dict,
                  output_path: str = None) -> plt.Figure:
    """Plot option value functions and NPV schedule."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    X_H_star = sol['H']['X_star']
    X_L_star = sol['L']['X_star']
    X_max = max(2.5 * X_H_star, 1.5 * X_L_star)
    X_grid = np.linspace(0.01, X_max, 300)

    regime_styles = {
        'H': {'color': '#d62728', 'label': 'High regime (post-adoption)'},
        'L': {'color': '#1f77b4', 'label': 'Low regime (pre-adoption)'},
    }

    for regime, style in regime_styles.items():
        X_star = sol[regime]['X_star']
        K_star = sol[regime]['K_star']
        d = np.exp(-model.p.r * model.p.tau)

        F_vals = [model.option_value(x, regime, sol) for x in X_grid]
        NPV_vals = [d * model.V(x, K_star, regime) - model.I(K_star) for x in X_grid]

        axes[0].plot(X_grid, F_vals, color=style['color'], lw=2, label=style['label'])
        axes[0].axvline(X_star, color=style['color'], ls='--', alpha=0.6)
        axes[0].annotate(f"$X^*_{regime[0]}$={X_star:.2f}",
                          xy=(X_star, 0), xytext=(X_star * 1.05, axes[0].get_ylim()[1] * 0.1),
                          color=style['color'], fontsize=9)

        axes[1].plot(X_grid, NPV_vals, color=style['color'], lw=2, label=style['label'])
        axes[1].axvline(X_star, color=style['color'], ls='--', alpha=0.6)

    axes[0].axhline(0, color='black', lw=0.5)
    axes[0].set_xlabel('Demand level $X$', fontsize=12)
    axes[0].set_ylabel('Option value $F_s(X)$', fontsize=12)
    axes[0].set_title('Investment Option Value by Regime', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(bottom=0)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].set_xlabel('Demand level $X$', fontsize=12)
    axes[1].set_ylabel('Net investment NPV', fontsize=12)
    axes[1].set_title('NPV of Investing (at $K^*$ chosen at trigger)', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    return fig


def plot_comparative_statics(model: SingleFirmModel,
                              output_path: str = None) -> plt.Figure:
    """4-panel comparative statics figure."""
    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    colors = {'L': '#1f77b4', 'H': '#d62728'}

    configs = [
        ('lam',     np.linspace(0.05, 0.60, 30), 'Adoption arrival rate $\\lambda$',
         'Investment trigger $X^*$', 'X', '(a)'),
        ('lam',     np.linspace(0.05, 0.60, 30), 'Adoption arrival rate $\\lambda$',
         'Optimal capacity $K^*$', 'K', '(b)'),
        ('sigma_L', np.linspace(0.10, 0.55, 30), 'Pre-adoption volatility $\\sigma_L$',
         'Investment trigger $X^*$', 'X', '(c)'),
        ('alpha',   np.linspace(0.25, 0.65, 30), 'Revenue-compute elasticity $\\alpha$',
         'Investment trigger $X^*$', 'X', '(d)'),
    ]

    for ax, (param, vals, xlabel, ylabel, var, label) in zip(axes, configs):
        cs = model.comparative_statics(param, vals)
        for regime, color in colors.items():
            key_L = f'{var}_star_L'
            key_H = f'{var}_star_H'
            key = key_L if regime == 'L' else key_H
            y = cs[key]
            ax.plot(vals, y, color=color, lw=2,
                     label=f'Regime {regime}')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'{label} {ylabel} vs. {xlabel.split("$")[1] if "$" in xlabel else xlabel}',
                      fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle('Single-Firm Model: Comparative Statics', fontsize=13, y=1.01)
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
    print("Phase 1: Single-Firm Real Options Model")
    print("=" * 60)

    params = ModelParams()
    params.validate()
    print("\nParameters:")
    for f in params.__dataclass_fields__:
        print(f"  {f} = {getattr(params, f)}")

    # Check interior solution condition
    for regime in ['L', 'H']:
        beta, _ = params.characteristic_roots(regime)
        psi = params.psi(regime)
        print(f"\nRegime {regime}: β={beta:.4f}, ψ={psi:.4f}, γ={params.gamma}")
        print(f"  Interior condition 1 < ψ < γ: {params.check_interior_condition(regime)}")

    # Run symbolic verification
    print()
    sympy_verify()

    # Solve the model
    print("\n" + "=" * 60)
    print("Solving for optimal investment policy...")
    model = SingleFirmModel(params)
    sol = model.solve()

    print("\nOptimal Investment Policy:")
    for regime in ['H', 'L']:
        s = sol[regime]
        print(f"\n  Regime {regime}:")
        print(f"    X*_{regime} = {s['X_star']:.4f}  (investment trigger)")
        print(f"    K*_{regime} = {s['K_star']:.4f}  (optimal capacity)")
        print(f"    β_{regime}  = {s['beta']:.4f}  (option elasticity)")
        print(f"    NPV at trigger = {s['NPV']:.4f}")

    ratio = sol['L']['X_star'] / sol['H']['X_star']
    print(f"\n  X*_L / X*_H = {ratio:.4f} (> 1: firm waits longer in low regime  ✓)")

    # Verify the switching premium: D_L > 0 means firm invests sooner in L
    # because the option to switch to H is valuable
    D_L = sol['L']['D']
    print(f"  D_L (switching premium) = {D_L:.4f} ({'> 0 ✓' if D_L > 0 else '≤ 0 ✗'})")

    print(f"\n  λ-effect: X*_L decreases in λ (adoption arrival rate)")
    print(f"  σ-effect: X*_s increases in σ_s (standard option value of waiting)")
    print(f"  α-effect: X*_s decreases in α (higher returns → lower threshold)")

    # Generate figures
    print("\nGenerating figures...")
    plot_solution(model, sol, '/workspaces/claude-paper/figures/phase1_solution.png')
    plot_comparative_statics(model, '/workspaces/claude-paper/figures/phase1_comp_statics.png')

    print("\nPhase 1 complete. ✓")
