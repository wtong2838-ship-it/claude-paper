"""
Phase 3: N-Firm Sequential Investment Game with Heterogeneity
=============================================================

Extends the duopoly (Phase 2) to N >= 3 firms with:

1. N-firm Cournot-style sequential investment game
   - Bouis, Huisman & Kort (2009): "accordion effect" — higher N → lower entry triggers
   - Each active firm earns: revenue = X * K_i^alpha / N_active
   - Sequential backward-induction equilibrium: X_1* < X_2* < ... < X_N*

2. Training vs. inference allocation (Akcigit & Kerr 2018 style)
   - Inference: immediate revenue from serving customers
   - Training: quality increment => future competitive advantage
   - FOC equates marginal revenue gain from inference vs. marginal quality gain from training

3. Firm heterogeneity (Bolton, Wang & Yang 2019)
   - Cash reserves W_i: financial constraints raise investment thresholds
   - Existing capacity K_i^0: incumbents have lower marginal benefit of new investment
   - Model quality q_i^0: quality advantage lowers trigger (higher demand share)
   - Cost of capital r_i: financially constrained firms invest later

References:
- Bouis, Huisman & Kort (2009, EJOR): "The Number of Firms and the Market Entry Game"
- Akcigit & Kerr (2018, JPE): "Growth Through Heterogeneous Innovations"
- Bolton, Wang & Yang (2019, JFE): "Optimal Contracting, Corporate Finance, and Valuation"
- Huisman & Kort (2015, RAND): "Strategic Capacity Investment under Uncertainty"
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from phase1_base_model import ModelParams, SingleFirmModel
from phase2_duopoly import DuopolyParams, DuopolyModel, LelandDefault


# --------------------------------------------------------------------------- #
#  Parameters                                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class NFirmParams(DuopolyParams):
    """
    Parameters for the N-firm investment game.

    Inherits all DuopolyParams fields and adds:
    - N: total number of potential entrants
    - cash: list of cash reserves per firm (length N)
    - quality: list of existing model quality per firm (length N)
    - capacity_init: list of existing compute capacity per firm (length N)
    - r_individual: list of individual discount rates (length N); if None, all use r

    Revenue structure (Cournot N-active):
      pi_i(X, K_i) = X * K_i^alpha / N_active

    This generalizes the duopoly theta parameterization:
    - N_active = 1: monopoly  => pi = X * K^alpha
    - N_active = 2: duopoly   => pi = theta * X * K^alpha  (theta = 1/2 here)
    - N_active = N: oligopoly => pi = X * K^alpha / N

    Note: the theta parameter from DuopolyParams is overridden by 1/N_active
    in N-firm calculations.
    """
    N: int = 4

    # Firm heterogeneity — will be set to uniform defaults if not provided
    cash: List[float] = field(default_factory=list)
    quality: List[float] = field(default_factory=list)
    capacity_init: List[float] = field(default_factory=list)
    r_individual: List[float] = field(default_factory=list)

    # Training-inference parameters (Akcigit & Kerr 2018)
    beta_scale: float = 0.30   # quality increment: Δq = beta_scale * log(1 + K_T)
    quality_elasticity: float = 0.50   # demand share sensitivity to quality: s_i ∝ q_i^quality_elasticity

    def __post_init__(self):
        """Initialize heterogeneous firm attributes to uniform defaults."""
        if not self.cash:
            self.cash = [1.0] * self.N
        if not self.quality:
            self.quality = [1.0] * self.N
        if not self.capacity_init:
            self.capacity_init = [0.0] * self.N
        if not self.r_individual:
            self.r_individual = [self.r] * self.N

    def validate(self):
        super().validate()
        assert self.N >= 2, "Need N >= 2 firms"
        assert len(self.cash) == self.N, f"cash list must have length N={self.N}"
        assert len(self.quality) == self.N, f"quality list must have length N={self.N}"
        assert len(self.capacity_init) == self.N, f"capacity_init list must have length N={self.N}"
        assert len(self.r_individual) == self.N, f"r_individual list must have length N={self.N}"
        assert all(c >= 0 for c in self.cash), "Cash reserves must be non-negative"
        assert all(q > 0 for q in self.quality), "Quality must be positive"
        assert all(ri > 0 for ri in self.r_individual), "Individual discount rates must be positive"

    def phi_nfirm(self, K: float, N_active: int, regime: str = 'H') -> float:
        """
        φ coefficient for N-firm Cournot revenue.
        Revenue per firm: pi_i = X * K^alpha / N_active
        => V = phi * X - delta*K/r
        phi = K^alpha / (N_active * (r - mu))
        """
        mu = self.mu_H if regime == 'H' else self.mu_L
        if regime == 'H':
            return K**self.alpha / (N_active * (self.r - self.mu_H))
        else:
            phi_H = K**self.alpha / (N_active * (self.r - self.mu_H))
            return (K**self.alpha / N_active + self.lam * phi_H) / (self.r - self.mu_L + self.lam)


# --------------------------------------------------------------------------- #
#  N-Firm Sequential Investment Model                                           #
# --------------------------------------------------------------------------- #

class NFirmModel:
    """
    N-firm sequential investment game with backward induction.

    Equilibrium structure (Bouis, Huisman & Kort 2009):
    - Solve backward: Nth firm's problem first, then (N-1)th, ..., then 1st
    - Each firm anticipates subsequent entrants' strategies
    - Triggers: X_1* < X_2* < ... < X_N* ("accordion effect")
    - The accordion effect: more firms => lower triggers for earlier entrants

    Revenue formula:
      When k firms are already invested, the (k+1)th entrant earns:
        pi_{k+1}(X, K) = X * K^alpha / (k+1)
      i.e. N_active increases by 1 with each entry.

    Backward induction:
      Step N: Firm N is the last entrant, faces N active firms after entry.
              Standard single-firm problem with phi = K^alpha/(N*(r-mu))
      Step k (k=N-1,...,1): Firm k anticipates firms k+1,...,N entering later.
              Pre-entry: monopoly-like (k firms active including k itself)
              Post-entry of k+1: k+1 firms active
              Trigger adjusted downward by preemption from k+1.
    """

    def __init__(self, params: NFirmParams):
        self.p = params
        self.p.validate()

    def _optimal_K_for_phi(self, phi_fn, beta: float, d: float) -> Tuple[float, float]:
        """
        Find optimal K and corresponding trigger X* by minimizing X*^beta / NPV.
        phi_fn: function K -> phi coefficient
        Returns (K_star, X_star).
        """
        p = self.p

        def objective(logK):
            K = np.exp(logK)
            phi = phi_fn(K)
            if phi <= 0:
                return np.inf
            num = beta * (d * p.delta * K / p.r + p.c * K**p.gamma)
            den = (beta - 1) * d * phi
            if den <= 0:
                return np.inf
            X_star = num / den
            NPV = (beta - 1) * (d * p.delta * K / p.r + p.c * K**p.gamma)
            if NPV <= 0:
                return np.inf
            return X_star**beta / NPV

        logK_grid = np.linspace(-6, 3, 300)
        objs = [objective(lk) for lk in logK_grid]
        finite = np.isfinite(objs)
        if not np.any(finite):
            raise RuntimeError("No valid K found in _optimal_K_for_phi")

        best = np.argmin(np.where(finite, objs, np.inf))
        logK_star = logK_grid[best]

        # Refine
        lo = logK_grid[max(0, best - 6)]
        hi = logK_grid[min(len(logK_grid) - 1, best + 6)]
        try:
            eps = 0.01
            def deriv(lk):
                return (objective(lk + eps) - objective(lk - eps)) / (2 * eps)
            if deriv(lo) * deriv(hi) < 0:
                logK_star = brentq(deriv, lo, hi, xtol=1e-8)
        except Exception:
            pass

        K_star = np.exp(logK_star)
        phi_star = phi_fn(K_star)
        num = beta * (d * p.delta * K_star / p.r + p.c * K_star**p.gamma)
        den = (beta - 1) * d * phi_star
        X_star = num / den if den > 0 else np.inf
        return K_star, X_star

    def _X_trigger_from_K(self, K: float, phi: float, beta: float, d: float) -> float:
        """Compute X*(K) from smooth-pasting: X* = beta*(d*delta*K/r + c*K^gamma)/((beta-1)*d*phi)."""
        p = self.p
        num = beta * (d * p.delta * K / p.r + p.c * K**p.gamma)
        den = (beta - 1) * d * phi
        return num / den if den > 0 else np.inf

    def solve_sequential(self, regime: str = 'H') -> Dict:
        """
        Backward induction for N-firm sequential investment game.

        Returns dict with:
        - 'triggers': list [X_1*, X_2*, ..., X_N*] (ascending order)
        - 'capacities': list [K_1*, K_2*, ..., K_N*]
        - 'N': total firms
        - 'accordion_effect': relative decrease in first-mover trigger vs single-firm
        - 'entry_order': list of firm indices (for heterogeneous case uses ordering)
        """
        p = self.p
        beta, _ = p.characteristic_roots(regime,
                    extra_discount=p.lam if regime == 'L' else 0.0)
        d = np.exp(-p.r * p.tau)
        N = p.N

        triggers = [None] * N
        capacities = [None] * N

        # ---- Step 1: Solve backward from firm N (last entrant) ----
        # Firm N enters with N firms already active (including itself)
        # phi = K^alpha / (N * (r - mu))
        def phi_N(K):
            return p.phi_nfirm(K, N, regime)

        K_N, X_N = self._optimal_K_for_phi(phi_N, beta, d)
        triggers[N - 1] = X_N
        capacities[N - 1] = K_N

        # ---- Step 2: Solve backward for firms k = N-1, ..., 1 ----
        # Firm k is the k-th entrant (k firms active after it enters).
        # After firm k enters: k firms active => pi_k = X * K^alpha / k
        # Before entry of k+1: firm k earns as if k firms active.
        #
        # Preemption condition (Bouis et al. 2009):
        # Firm k accelerates entry to X_k* where V_lead(X_k*) = V_follow(X_k*)
        # V_lead: invest now as k-th firm (k active, earn X * K_k^alpha / k)
        # V_follow: wait to be (k+1)th firm (earn X * K^alpha / (k+1))
        #
        # Leader trigger from V_lead:
        #   X_k^lead = solve optimal K with N_active = k
        # Preemption pushes trigger below X_k^lead to where indifferent.

        for k in range(N - 1, 0, -1):
            # k = number of firms active after this entrant invests (1-indexed)
            # phi as the k-th entrant (N_active = k after entry)
            def phi_k(K, k=k):
                return p.phi_nfirm(K, k, regime)

            K_k, X_k_lead = self._optimal_K_for_phi(phi_k, beta, d)

            # Preemption: compare V_lead(X) vs. V_follow(X) at X_next = triggers[k]
            # V_lead(X) = d * (phi_k(K_k) * X - delta * K_k / r) - c * K_k^gamma
            # V_follow(X) = (X / X_next)^beta * NPV_next  [wait to be (k+1)th entrant]
            X_next = triggers[k]          # trigger of next entrant
            K_next = capacities[k]        # capacity of next entrant
            phi_next = phi_k(K_next)      # phi for k-th entrant at K_next position

            # NPV of (k+1)th entrant at their trigger X_next
            phi_k1 = p.phi_nfirm(K_next, k + 1, regime)
            NPV_next = (beta - 1) * (d * p.delta * K_next / p.r + p.c * K_next**p.gamma)

            phi_k_val = phi_k(K_k)

            def V_lead(X):
                return d * (phi_k_val * X - p.delta * K_k / p.r) - p.c * K_k**p.gamma

            def V_follow(X):
                if X >= X_next:
                    return NPV_next
                return (X / X_next)**beta * NPV_next

            def preemption_eq(X):
                return V_lead(X) - V_follow(X)

            # Check if preemption occurs at X_k_lead
            if X_k_lead < X_next:
                # Candidate preemption trigger
                f_at_lead = preemption_eq(X_k_lead)
                f_at_eps = preemption_eq(1e-8)
                if f_at_eps * f_at_lead < 0:
                    try:
                        X_k_preempt = brentq(preemption_eq, 1e-8, X_k_lead,
                                              xtol=1e-10, maxiter=500)
                    except Exception:
                        X_k_preempt = X_k_lead
                else:
                    X_k_preempt = X_k_lead
            else:
                # No room for preemption (already past next trigger)
                X_k_preempt = X_k_lead

            # Take the minimum (preemption or Stackelberg)
            X_k_star = min(X_k_preempt, X_k_lead)
            X_k_star = max(X_k_star, 1e-8)  # floor

            triggers[k - 1] = X_k_star
            capacities[k - 1] = K_k

        # ---- Compute accordion effect ----
        # Single-firm trigger for comparison
        sf = SingleFirmModel(self.p)
        sf_sol = sf.solve()
        X_single = sf_sol[regime]['X_star']

        accordion_effect = (X_single - triggers[0]) / max(X_single, 1e-10)

        return {
            'triggers': triggers,
            'capacities': capacities,
            'N': N,
            'accordion_effect': accordion_effect,
            'X_single': X_single,
            'regime': regime,
            'beta': beta,
        }

    def training_inference_allocation(self, K: float, X: float,
                                       quality: float,
                                       competitor_quality: float) -> Dict:
        """
        Split total capacity K between training (K_T) and inference (K_I = K - K_T).

        Inference: revenue = X * K_I^alpha / (K_I^alpha + competitor_K_I^alpha)
          => We use the quality-weighted share: s_I = K_I^alpha / (K_I^alpha + comp_alpha)
          Here competitor_K_I is approximated from competitor_quality as K_I_comp^alpha ~ competitor_quality
          Inference revenue = X * K_I^alpha / (K_I^alpha + competitor_quality)

        Training: quality increment Δq = beta_scale * log(1 + K_T)
          Future competitive value of quality: V_quality(Δq) ∝ Δq * X / (r - mu_H)
          This is the incremental PV from improved demand share next period.

        FOC: dRevenue/dK_I = dQuality_Value/dK_T (with K_T = K - K_I)
          alpha * X * K_I^(alpha-1) * competitor_quality / (K_I^alpha + competitor_quality)^2
             = dV_quality/dK_T
          where dV_quality/dK_T = beta_scale / (1 + K_T) * X / (r - mu_H)

        We solve this FOC numerically for K_I in (epsilon, K).
        """
        p = self.p
        q_comp = max(competitor_quality, 1e-6)
        r_mu = p.r - p.mu_H
        if r_mu <= 0:
            r_mu = 0.01

        def marginal_inference(K_I):
            """d(inference revenue)/d(K_I)."""
            if K_I <= 0 or K_I >= K:
                return 0.0
            return (p.alpha * X * K_I**(p.alpha - 1) * q_comp
                    / (K_I**p.alpha + q_comp)**2)

        def marginal_training(K_I):
            """d(quality value)/d(K_T) = d(quality value)/d(K - K_I) evaluated at K_T = K - K_I."""
            K_T = K - K_I
            if K_T <= 0:
                return 0.0
            return p.beta_scale / (1 + K_T) * X / r_mu

        def foc(K_I):
            return marginal_inference(K_I) - marginal_training(K_I)

        # Evaluate at boundaries
        eps = 1e-6 * K
        f_lo = foc(eps)
        f_hi = foc(K - eps)

        K_I_star = K / 2.0  # default: 50/50 split
        if f_lo * f_hi < 0:
            try:
                K_I_star = brentq(foc, eps, K - eps, xtol=1e-10, maxiter=500)
            except Exception:
                pass
        elif f_lo > 0:
            # Marginal inference > marginal training at low K_I => push K_I up
            K_I_star = K * 0.9
        else:
            # Marginal training dominates at low K_I => push K_I down
            K_I_star = K * 0.1

        K_I_star = float(np.clip(K_I_star, eps, K - eps))
        K_T_star = K - K_I_star

        # Compute revenues and quality increment
        inference_revenue = X * K_I_star**p.alpha / (K_I_star**p.alpha + q_comp)
        quality_increment = p.beta_scale * np.log(1 + K_T_star)
        quality_value = quality_increment * X / r_mu

        return {
            'K_I': K_I_star,
            'K_T': K_T_star,
            'split': K_T_star / K,       # training fraction
            'inference_revenue': inference_revenue,
            'quality_increment': quality_increment,
            'quality_value': quality_value,
        }


# --------------------------------------------------------------------------- #
#  Heterogeneous Firms                                                          #
# --------------------------------------------------------------------------- #

class HeterogeneousFirms:
    """
    N firms with heterogeneous characteristics:
    - Cash reserves W_i (Bolton, Wang & Yang 2019): financial constraints
    - Existing capacity K_i^0: incumbent advantage
    - Model quality q_i^0: demand share advantage
    - Individual discount rates r_i: cost of capital differences

    The key mechanism (Bolton, Wang & Yang 2019):
    - Financially constrained firms (low W_i) face higher effective discount rate
      => they invest LATER (higher trigger) due to cost-of-carry on the investment option
    - Cash-rich firms (Google, Microsoft) invest SOONER (lower trigger)

    We model this by adjusting each firm's effective discount rate:
      r_i_eff = r_i + lambda_W * max(0, W_bar - W_i) / W_bar
    where lambda_W is the shadow cost of financial constraints.

    Quality advantage lowers the effective trigger:
      phi_i(K) = q_i^epsilon * K^alpha / (r_eff - mu)
    so higher quality => higher phi => lower X*_i.

    The equilibrium is computed by:
    1. Rank firms by (adjusted) trigger X*_i (ascending)
    2. The firm with lowest X*_i is the leader (first entrant)
    3. Sequential entry in order of ascending trigger
    """

    def __init__(self, params: NFirmParams):
        self.p = params
        self.p.validate()
        self.nfirm = NFirmModel(params)

    def _effective_discount_rate(self, firm_idx: int) -> float:
        """
        Effective discount rate for firm i, adjusted for financial constraints.
        Cash-constrained firms face higher effective r (delayed investment).

        r_eff_i = r_i + financial_penalty
        financial_penalty = max(0, cash_penalty * (1 - W_i / W_bar)) / r
        where W_bar = max(cash) is the cash-rich benchmark.
        """
        p = self.p
        W_i = p.cash[firm_idx]
        r_i = p.r_individual[firm_idx]
        W_bar = max(p.cash)

        if W_bar <= 0:
            return r_i

        # Financial constraint penalty: constrained firms pay up to 50% extra
        # This is a simplified version of Bolton et al. (2019) Proposition 3
        constraint_severity = max(0.0, 1.0 - W_i / W_bar)  # 0 for unconstrained, 1 for no cash
        financial_penalty = 0.5 * r_i * constraint_severity  # up to 50% premium
        return r_i + financial_penalty

    def _firm_phi(self, K: float, firm_idx: int, N_active: int,
                  regime: str = 'H') -> float:
        """
        Quality-adjusted phi coefficient for firm i with N_active competing firms.
        phi_i = q_i^epsilon * K^alpha / (N_active * (r_eff - mu))
        """
        p = self.p
        q_i = p.quality[firm_idx]
        r_eff = self._effective_discount_rate(firm_idx)
        eps_q = p.quality_elasticity

        mu = p.mu_H if regime == 'H' else p.mu_L
        r_mu = max(r_eff - mu, 1e-4)  # prevent division by zero

        if regime == 'H':
            return q_i**eps_q * K**p.alpha / (N_active * r_mu)
        else:
            phi_H = q_i**eps_q * K**p.alpha / (N_active * (r_eff - p.mu_H))
            return (q_i**eps_q * K**p.alpha / N_active + p.lam * phi_H) / (r_eff - p.mu_L + p.lam)

    def _optimal_K_firm(self, firm_idx: int, N_active: int,
                         regime: str = 'H') -> Tuple[float, float]:
        """Find optimal (K, X) for firm i entering as the N_active-th firm."""
        p = self.p
        r_eff = self._effective_discount_rate(firm_idx)
        # Compute beta with firm's effective discount rate
        sigma = p.sigma_H if regime == 'H' else p.sigma_L
        mu = p.mu_H if regime == 'H' else p.mu_L
        rho = r_eff + (p.lam if regime == 'L' else 0.0)
        a = 0.5 * sigma**2
        b = mu - 0.5 * sigma**2
        disc = b**2 + 4 * a * rho
        beta = (-b + np.sqrt(disc)) / (2 * a)

        d = np.exp(-r_eff * p.tau)

        def phi_fn(K):
            return self._firm_phi(K, firm_idx, N_active, regime)

        def objective(logK):
            K = np.exp(logK)
            phi = phi_fn(K)
            if phi <= 0:
                return np.inf
            num = beta * (d * p.delta * K / r_eff + p.c * K**p.gamma)
            den = (beta - 1) * d * phi
            if den <= 0:
                return np.inf
            X_star = num / den
            NPV = (beta - 1) * (d * p.delta * K / r_eff + p.c * K**p.gamma)
            if NPV <= 0:
                return np.inf
            return X_star**beta / NPV

        logK_grid = np.linspace(-6, 3, 200)
        objs = [objective(lk) for lk in logK_grid]
        finite = np.isfinite(objs)
        if not np.any(finite):
            K_star = 0.1
            phi_star = phi_fn(K_star)
            num = beta * (d * p.delta * K_star / r_eff + p.c * K_star**p.gamma)
            den = (beta - 1) * d * phi_star
            return K_star, (num / den if den > 0 else 1.0)

        best = np.argmin(np.where(finite, objs, np.inf))
        logK_star = logK_grid[best]

        lo = logK_grid[max(0, best - 6)]
        hi = logK_grid[min(len(logK_grid) - 1, best + 6)]
        try:
            eps = 0.01
            def deriv(lk):
                return (objective(lk + eps) - objective(lk - eps)) / (2 * eps)
            if deriv(lo) * deriv(hi) < 0:
                logK_star = brentq(deriv, lo, hi, xtol=1e-8)
        except Exception:
            pass

        K_star = np.exp(logK_star)
        phi_star = phi_fn(K_star)
        r_eff_i = self._effective_discount_rate(firm_idx)
        num = beta * (d * p.delta * K_star / r_eff_i + p.c * K_star**p.gamma)
        den = (beta - 1) * d * phi_star
        X_star = num / den if den > 0 else np.inf
        return K_star, X_star

    def solve_heterogeneous_equilibrium(self, regime: str = 'H') -> Dict:
        """
        Solve the heterogeneous N-firm sequential investment game.

        Algorithm:
        1. Compute each firm's standalone optimal trigger X*_i (as if entering alone)
        2. Rank firms by ascending X*_i => entry order
        3. Backward induction accounting for quality-adjusted revenue and individual r_i
        4. Return entry sequence with firm identities

        Returns dict with:
        - 'entry_order': list of firm indices in order of entry
        - 'triggers': list of triggers X_i* in entry order
        - 'capacities': list of capacities K_i* in entry order
        - 'firm_attributes': dict of per-firm attributes (cash, quality, r_eff)
        - 'cash_effect': correlation between cash rank and entry rank
        - 'quality_effect': correlation between quality rank and entry rank (negative)
        """
        p = self.p
        N = p.N

        # Step 1: Compute standalone triggers for each firm
        # "Standalone" = entering as the N-th entrant in a symmetric world
        standalone_K = []
        standalone_X = []
        for i in range(N):
            K_i, X_i = self._optimal_K_firm(i, N_active=N, regime=regime)
            standalone_K.append(K_i)
            standalone_X.append(X_i)

        # Step 2: Entry order = ascending standalone trigger (lower trigger => enters earlier)
        entry_order = np.argsort(standalone_X).tolist()

        # Step 3: Backward induction with heterogeneous firms
        # Process in reverse entry order: last entrant first
        N_active_at_entry = list(range(1, N + 1))  # firm entry_order[k] is the (k+1)-th entrant

        # Solve last entrant (position N-1 in 0-indexed order)
        triggers = [None] * N
        capacities = [None] * N

        last_firm_idx = entry_order[N - 1]
        K_last, X_last = self._optimal_K_firm(last_firm_idx, N_active=N, regime=regime)
        triggers[N - 1] = X_last
        capacities[N - 1] = K_last

        # Backward induction for positions N-2 to 0
        for pos in range(N - 2, -1, -1):
            firm_idx = entry_order[pos]
            n_active = pos + 1   # this firm is the (pos+1)-th entrant

            K_i, X_i_lead = self._optimal_K_firm(firm_idx, N_active=n_active, regime=regime)

            # Preemption condition vs. next entrant
            X_next = triggers[pos + 1]
            K_next = capacities[pos + 1]
            next_firm = entry_order[pos + 1]

            r_eff = self._effective_discount_rate(firm_idx)
            d = np.exp(-r_eff * p.tau)
            beta_vals, _ = p.characteristic_roots(regime,
                             extra_discount=p.lam if regime == 'L' else 0.0)

            phi_i = self._firm_phi(K_i, firm_idx, n_active, regime)
            phi_next = self._firm_phi(K_next, next_firm, n_active + 1, regime)

            NPV_next = (beta_vals - 1) * (d * p.delta * K_next / r_eff + p.c * K_next**p.gamma)
            NPV_next = max(NPV_next, 0.0)

            def V_lead(X):
                return d * (phi_i * X - p.delta * K_i / r_eff) - p.c * K_i**p.gamma

            def V_follow(X):
                if X >= X_next:
                    return NPV_next
                return (X / X_next)**beta_vals * NPV_next

            def preemption_eq(X):
                return V_lead(X) - V_follow(X)

            X_k_star = X_i_lead  # default: no preemption
            if X_i_lead < X_next:
                f_lo = preemption_eq(1e-8)
                f_hi = preemption_eq(X_i_lead)
                if f_lo * f_hi < 0:
                    try:
                        X_k_star = brentq(preemption_eq, 1e-8, X_i_lead,
                                           xtol=1e-10, maxiter=500)
                    except Exception:
                        pass

            X_k_star = max(X_k_star, 1e-8)
            triggers[pos] = X_k_star
            capacities[pos] = K_i

        # Step 4: Compute effects
        cash_ranks = np.argsort(np.argsort(-np.array(p.cash)))  # rank by descending cash
        entry_ranks = np.array([entry_order.index(i) for i in range(N)])

        r_effs = [self._effective_discount_rate(i) for i in range(N)]
        qualities = p.quality

        return {
            'entry_order': entry_order,
            'triggers': triggers,
            'capacities': capacities,
            'firm_attributes': {
                'cash': list(p.cash),
                'quality': list(p.quality),
                'capacity_init': list(p.capacity_init),
                'r_individual': list(p.r_individual),
                'r_effective': r_effs,
            },
            'standalone_triggers': standalone_X,
            'standalone_capacities': standalone_K,
            'regime': regime,
        }


# --------------------------------------------------------------------------- #
#  Plotting                                                                     #
# --------------------------------------------------------------------------- #

def plot_nfirm_timeline(sol: Dict, output_path: str = None) -> plt.Figure:
    """
    Waterfall chart showing the N-firm investment timeline.
    X_1* < X_2* < ... < X_N* with the accordion effect.
    """
    triggers = sol['triggers']
    capacities = sol['capacities']
    N = sol['N']
    X_single = sol['X_single']
    accordion = sol['accordion_effect']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: triggers as horizontal timeline
    ax = axes[0]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, N))

    for i, (X_i, K_i) in enumerate(zip(triggers, capacities)):
        label = f'Firm {i+1}: $X_{i+1}^* = {X_i:.3f}$'
        ax.barh(i, X_i, color=colors[i], edgecolor='white', linewidth=1.5,
                 height=0.7, label=label)
        ax.text(X_i + max(triggers) * 0.01, i, f'{X_i:.3f}', va='center', fontsize=9)

    ax.axvline(X_single, color='#d62728', ls='--', lw=2, label=f'Single-firm $X^*={X_single:.3f}$')
    ax.set_yticks(range(N))
    ax.set_yticklabels([f'Firm {i+1}' for i in range(N)], fontsize=10)
    ax.set_xlabel('Investment trigger $X^*$', fontsize=12)
    ax.set_title(f'N={N}-Firm Investment Timeline\n'
                  f'(Accordion effect: {accordion*100:.1f}% below single-firm)',
                  fontsize=11)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max(triggers) * 1.20)

    # Right panel: capacities and triggers scatter
    ax = axes[1]
    entry_numbers = np.arange(1, N + 1)
    ax.plot(entry_numbers, triggers, 'o-', color='#1f77b4', lw=2.5,
             markersize=8, label='Investment trigger $X_k^*$')
    ax.axhline(X_single, color='#d62728', ls='--', lw=1.5,
                label=f'Single-firm benchmark')

    ax2 = ax.twinx()
    ax2.plot(entry_numbers, capacities, 's--', color='#2ca02c', lw=2,
              markersize=7, label='Optimal capacity $K_k^*$')
    ax2.set_ylabel('Optimal capacity $K_k^*$', color='#2ca02c', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='#2ca02c')

    ax.set_xlabel('Entry order (k)', fontsize=12)
    ax.set_ylabel('Investment trigger $X_k^*$', color='#1f77b4', fontsize=11)
    ax.tick_params(axis='y', labelcolor='#1f77b4')
    ax.set_title('Triggers and Capacities by Entry Order', fontsize=11)
    ax.set_xticks(entry_numbers)
    ax.grid(alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')

    plt.suptitle('N-Firm Sequential Investment Game (Bouis, Huisman & Kort 2009)',
                  fontsize=12, y=1.01)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    return fig


def plot_training_inference(params: NFirmParams, output_path: str = None) -> plt.Figure:
    """
    Training/inference split as function of quality gap and demand X.
    Three-panel figure:
    (a) Training fraction vs. quality gap (competitor_quality / own_quality)
    (b) Training fraction vs. demand X
    (c) 2D heatmap: training fraction over (quality_gap, X)
    """
    model = NFirmModel(params)
    K = 0.5  # fixed total capacity

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel (a): Training fraction vs. quality gap
    ax = axes[0]
    quality_gaps = np.linspace(0.1, 5.0, 50)
    X_vals_panel_a = [0.05, 0.10, 0.20]
    colors_a = ['#1f77b4', '#ff7f0e', '#d62728']

    for X_val, color in zip(X_vals_panel_a, colors_a):
        splits = []
        for qgap in quality_gaps:
            alloc = model.training_inference_allocation(K, X_val, 1.0, qgap)
            splits.append(alloc['split'])
        ax.plot(quality_gaps, splits, color=color, lw=2, label=f'$X={X_val}$')

    ax.set_xlabel('Quality gap (competitor quality / own)', fontsize=11)
    ax.set_ylabel('Training fraction $K_T / K$', fontsize=11)
    ax.set_title('(a) Training Share vs. Quality Gap', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    # Panel (b): Training fraction vs. demand X
    ax = axes[1]
    X_range = np.linspace(0.01, 0.50, 60)
    quality_gaps_panel_b = [0.5, 1.0, 2.0, 4.0]
    colors_b = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(quality_gaps_panel_b)))

    for qgap, color in zip(quality_gaps_panel_b, colors_b):
        splits = []
        for X_val in X_range:
            alloc = model.training_inference_allocation(K, X_val, 1.0, qgap)
            splits.append(alloc['split'])
        ax.plot(X_range, splits, color=color, lw=2, label=f'quality gap={qgap}')

    ax.set_xlabel('Demand level $X$', fontsize=11)
    ax.set_ylabel('Training fraction $K_T / K$', fontsize=11)
    ax.set_title('(b) Training Share vs. Demand $X$', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    # Panel (c): 2D heatmap
    ax = axes[2]
    X_2d = np.linspace(0.02, 0.40, 30)
    qgap_2d = np.linspace(0.2, 4.0, 30)
    XX, QQ = np.meshgrid(X_2d, qgap_2d)
    splits_2d = np.zeros_like(XX)

    for i in range(len(qgap_2d)):
        for j in range(len(X_2d)):
            alloc = model.training_inference_allocation(K, XX[i, j], 1.0, QQ[i, j])
            splits_2d[i, j] = alloc['split']

    im = ax.contourf(XX, QQ, splits_2d, levels=20, cmap='RdYlBu_r')
    fig.colorbar(im, ax=ax, label='Training fraction')
    ax.set_xlabel('Demand level $X$', fontsize=11)
    ax.set_ylabel('Quality gap', fontsize=11)
    ax.set_title('(c) Training Share Heatmap', fontsize=11)
    # Add contour lines
    cs = ax.contour(XX, QQ, splits_2d, levels=5, colors='white', alpha=0.5, linewidths=0.8)
    ax.clabel(cs, fmt='%.2f', fontsize=8)

    plt.suptitle('Training vs. Inference Allocation (Akcigit & Kerr 2018 Style)',
                  fontsize=12, y=1.02)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    return fig


def plot_heterogeneity_effects(params: NFirmParams, output_path: str = None) -> plt.Figure:
    """
    Three-panel figure showing heterogeneity effects on equilibrium:
    (a) Cash reserves vs. investment trigger (cash-rich firms invest earlier)
    (b) Quality vs. investment trigger (higher quality => lower trigger)
    (c) Cost of capital vs. investment trigger (higher r => later investment)
    """
    N = params.N
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ---- Panel (a): Cash effect ----
    ax = axes[0]
    import copy

    n_points = 20
    cash_ratios = np.linspace(0.1, 3.0, n_points)  # cash[0] / cash[1] (firm 0 vs. uniform)
    triggers_cash_rich = []
    triggers_cash_poor = []
    entry_orders_cash = []

    base_cash = [1.0] * N
    for ratio in cash_ratios:
        p_new = copy.deepcopy(params)
        p_new.cash = base_cash.copy()
        p_new.cash[0] = ratio  # vary firm 0's cash
        try:
            hf = HeterogeneousFirms(p_new)
            sol = hf.solve_heterogeneous_equilibrium('H')
            # Trigger of firm 0
            firm0_pos = sol['entry_order'].index(0)
            triggers_cash_rich.append(sol['triggers'][firm0_pos])
            entry_orders_cash.append(firm0_pos)
        except Exception:
            triggers_cash_rich.append(np.nan)
            entry_orders_cash.append(np.nan)

    ax.plot(cash_ratios, triggers_cash_rich, 'o-', color='#1f77b4', lw=2,
             markersize=6, label='Trigger $X^*$ (firm 0)')
    ax.set_xlabel('Cash ratio (firm 0 / others)', fontsize=11)
    ax.set_ylabel('Investment trigger $X^*$', fontsize=11)
    ax.set_title('(a) Cash Reserves Effect\n(higher cash => lower trigger)', fontsize=11)
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(cash_ratios, entry_orders_cash, 's--', color='#ff7f0e', lw=1.5,
              markersize=5, alpha=0.7, label='Entry position')
    ax2.set_ylabel('Entry position (0=first)', color='#ff7f0e', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax2.set_ylim(-0.5, N - 0.5)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    # ---- Panel (b): Quality effect ----
    ax = axes[1]
    quality_vals = np.linspace(0.5, 3.0, n_points)
    triggers_quality = []

    base_quality = [1.0] * N
    for q0 in quality_vals:
        p_new = copy.deepcopy(params)
        p_new.quality = base_quality.copy()
        p_new.quality[0] = q0
        try:
            hf = HeterogeneousFirms(p_new)
            sol = hf.solve_heterogeneous_equilibrium('H')
            firm0_pos = sol['entry_order'].index(0)
            triggers_quality.append(sol['triggers'][firm0_pos])
        except Exception:
            triggers_quality.append(np.nan)

    ax.plot(quality_vals, triggers_quality, 'o-', color='#2ca02c', lw=2,
             markersize=6, label='Trigger $X^*$ (firm 0)')
    ax.set_xlabel('Quality of firm 0 (others = 1)', fontsize=11)
    ax.set_ylabel('Investment trigger $X^*$', fontsize=11)
    ax.set_title('(b) Model Quality Effect\n(higher quality => lower trigger)', fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # ---- Panel (c): Cost of capital effect ----
    ax = axes[2]
    r_vals = np.linspace(0.10, 0.30, n_points)
    triggers_r = []

    base_r = [params.r] * N
    for r0 in r_vals:
        p_new = copy.deepcopy(params)
        p_new.r_individual = base_r.copy()
        p_new.r_individual[0] = r0
        try:
            hf = HeterogeneousFirms(p_new)
            sol = hf.solve_heterogeneous_equilibrium('H')
            firm0_pos = sol['entry_order'].index(0)
            triggers_r.append(sol['triggers'][firm0_pos])
        except Exception:
            triggers_r.append(np.nan)

    ax.plot(r_vals, triggers_r, 'o-', color='#d62728', lw=2,
             markersize=6, label='Trigger $X^*$ (firm 0)')
    ax.set_xlabel('Cost of capital $r$ (firm 0)', fontsize=11)
    ax.set_ylabel('Investment trigger $X^*$', fontsize=11)
    ax.set_title('(c) Cost of Capital Effect\n(higher r => higher trigger)', fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    plt.suptitle('Firm Heterogeneity Effects on Investment Equilibrium\n'
                  '(Bolton, Wang & Yang 2019 Financial Constraints)',
                  fontsize=12, y=1.03)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    return fig


def plot_accordion_effect(params: NFirmParams, N_range: range = None,
                           output_path: str = None) -> plt.Figure:
    """
    Show how the first-mover trigger decreases as N increases (accordion effect).
    """
    import copy

    if N_range is None:
        N_range = range(2, 7)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    all_N = []
    all_first_trigger = []
    all_last_trigger = []
    all_accordion = []

    for N_val in N_range:
        p_new = copy.deepcopy(params)
        p_new.N = N_val
        p_new.cash = [1.0] * N_val
        p_new.quality = [1.0] * N_val
        p_new.capacity_init = [0.0] * N_val
        p_new.r_individual = [params.r] * N_val
        try:
            model = NFirmModel(p_new)
            sol = model.solve_sequential('H')
            all_N.append(N_val)
            all_first_trigger.append(sol['triggers'][0])
            all_last_trigger.append(sol['triggers'][-1])
            all_accordion.append(sol['accordion_effect'])
        except Exception as e:
            print(f"  Warning: N={N_val} failed: {e}")

    # Panel (a): Triggers vs. N
    ax = axes[0]
    ax.plot(all_N, all_first_trigger, 'o-', color='#1f77b4', lw=2.5,
             markersize=9, label='First entrant $X_1^*$')
    ax.plot(all_N, all_last_trigger, 's--', color='#d62728', lw=2,
             markersize=7, label='Last entrant $X_N^*$')
    if all_N:
        ax.axhline(all_first_trigger[0], color='gray', ls=':', alpha=0.7,
                    label=f'Duopoly first trigger (N=2)')

    ax.set_xlabel('Number of firms N', fontsize=12)
    ax.set_ylabel('Investment trigger $X^*$', fontsize=12)
    ax.set_title('Accordion Effect: Trigger vs. N', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xticks(all_N)
    ax.grid(alpha=0.3)

    # Panel (b): Accordion effect magnitude
    ax = axes[1]
    ax.bar(all_N, [a * 100 for a in all_accordion],
            color=plt.cm.Blues(np.linspace(0.4, 0.9, len(all_N))),
            edgecolor='white', linewidth=1.5)
    ax.set_xlabel('Number of firms N', fontsize=12)
    ax.set_ylabel('Accordion effect (% below single-firm)', fontsize=12)
    ax.set_title('First-Mover Preemption Intensity', fontsize=12)
    ax.set_xticks(all_N)
    ax.grid(axis='y', alpha=0.3)
    for i, (n, a) in enumerate(zip(all_N, all_accordion)):
        ax.text(n, a * 100 + 0.5, f'{a*100:.1f}%', ha='center', fontsize=9)

    plt.suptitle('Accordion Effect (Bouis, Huisman & Kort 2009):\n'
                  'More firms => stronger preemption for first entrant',
                  fontsize=12, y=1.02)
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
    print("Phase 3: N-Firm Sequential Investment Game")
    print("=" * 60)

    params = NFirmParams(N=4)
    params.validate()
    print(f"\nN={params.N} firms, theta={params.theta}")
    print(f"beta_scale={params.beta_scale}, quality_elasticity={params.quality_elasticity}")

    # ---- Symmetric N-firm game ----
    print("\n--- Symmetric N=4 Firm Sequential Game ---")
    model = NFirmModel(params)
    sol = model.solve_sequential('H')

    print(f"\nEntry triggers (X_1* < X_2* < ... < X_N*):")
    for i, (X_i, K_i) in enumerate(zip(sol['triggers'], sol['capacities'])):
        print(f"  Firm {i+1}: X*={X_i:.4f}, K*={K_i:.4f}")
    print(f"\nSingle-firm benchmark: X*={sol['X_single']:.4f}")
    print(f"Accordion effect: {sol['accordion_effect']*100:.1f}% reduction in first-mover trigger")

    # ---- Training/inference allocation ----
    print("\n--- Training vs. Inference Allocation ---")
    K_total = 0.5
    test_cases = [
        (0.05, 1.0, 2.0, "Low X, quality advantage"),
        (0.20, 1.0, 0.5, "High X, quality disadvantage"),
        (0.10, 1.0, 1.0, "Medium X, equal quality"),
    ]
    for X, q_own, q_comp, desc in test_cases:
        alloc = model.training_inference_allocation(K_total, X, q_own, q_comp)
        print(f"\n  {desc}: X={X}, q_comp/q_own={q_comp:.1f}")
        print(f"    K_I={alloc['K_I']:.3f}, K_T={alloc['K_T']:.3f}, "
               f"split(train%)={alloc['split']*100:.1f}%")
        print(f"    Inference revenue={alloc['inference_revenue']:.4f}, "
               f"Quality increment={alloc['quality_increment']:.4f}")

    # ---- Heterogeneous firms ----
    print("\n--- Heterogeneous Firms Equilibrium ---")
    params_het = NFirmParams(
        N=4,
        cash=[3.0, 1.0, 0.5, 0.2],          # firm 0 (Google/MSFT) is cash-rich
        quality=[1.5, 1.0, 1.0, 0.8],        # firm 0 has quality advantage
        capacity_init=[0.2, 0.1, 0.0, 0.0],  # firm 0 and 1 have existing capacity
        r_individual=[0.12, 0.15, 0.18, 0.22],  # firm 0 has lower cost of capital
    )
    hf = HeterogeneousFirms(params_het)
    sol_het = hf.solve_heterogeneous_equilibrium('H')

    print(f"\nEntry order: {[f'Firm {i+1}' for i in sol_het['entry_order']]}")
    print(f"Triggers in entry order:")
    for pos, (firm_idx, trigger, cap) in enumerate(
        zip(sol_het['entry_order'], sol_het['triggers'], sol_het['capacities'])
    ):
        cash_i = params_het.cash[firm_idx]
        q_i = params_het.quality[firm_idx]
        r_i = params_het.r_individual[firm_idx]
        r_eff = sol_het['firm_attributes']['r_effective'][firm_idx]
        print(f"  Position {pos+1}: Firm {firm_idx+1} "
               f"(cash={cash_i}, q={q_i}, r={r_i:.2f}, r_eff={r_eff:.2f}) "
               f"=> X*={trigger:.4f}, K*={cap:.4f}")

    # ---- Figures ----
    print("\nGenerating figures...")

    plot_nfirm_timeline(sol,
                         output_path='/workspaces/claude-paper/figures/phase3_timeline.png')

    plot_training_inference(params,
                             output_path='/workspaces/claude-paper/figures/phase3_training_inference.png')

    plot_heterogeneity_effects(params_het,
                                output_path='/workspaces/claude-paper/figures/phase3_heterogeneity.png')

    plot_accordion_effect(params, N_range=range(2, 7),
                           output_path='/workspaces/claude-paper/figures/phase3_accordion.png')

    print("\nPhase 3 complete.")
