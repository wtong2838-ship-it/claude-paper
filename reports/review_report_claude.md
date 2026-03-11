# Review Report: AI Lab Investment

**Reviewer:** claude
**Date:** 2026-03-11

---

## Executive Summary

This paper develops a real options model of AI compute infrastructure investment combining regime-switching demand, oligopoly preemption, Leland (1994) endogenous default, diminishing returns (scaling laws), and training/inference allocation. The theoretical ambition is high and the subject matter timely. The codebase is reasonably well-structured, the 117 tests all pass, and the Phase 1 (single-firm) and Phase 2 (duopoly) implementations are mathematically sound and match the paper's propositions closely.

However, several issues require attention before submission. The most significant is a correctness bug in the Phase 3 N-firm backward-induction solver: the computed entry triggers are not monotonically ordered (trigger[1] = 0.0852 > trigger[2] = 0.0486 for N=4), violating the equilibrium property X₁* < X₂* < X₃* < X₄* that the paper claims to deliver. There is also a default-argument bug in `infer_lambda` (defaults to regime=`'H'`, which produces incorrect results due to H-regime λ-independence), and the Appendix is incomplete — Propositions 3 and 4 have no proofs. The paper is well-motivated and contains genuine contributions, but is not yet ready for submission to a top journal in its current form.

**Code verdict:** Partially correct. Phase 1 and Phase 2 implementations are sound. Phase 3 has a monotonicity bug. Phase 4 has an API default bug. Phase 5 forgone-revenue formula is a rough approximation not matching the stated proposition.

**Paper verdict:** Revise before submission. The core mechanism (competition-leverage spiral) is well-motivated and numerically illustrated, but proofs for two of four propositions are missing, the N-firm section has an implementation issue, and some quantitative claims in the abstract are hard to reconcile with the actual model output.

---

## Part 1: Code Validation

### 1. Mathematical Correctness

**Propositions vs. code — Phase 1 (Proposition 1):**

*Passes.* Proposition 1(a) states $K^*(s) = [(\psi_s-1)\delta/(r \cdot c(\gamma-\psi_s))]^{1/(\gamma-1)}$. The code at `phase1_base_model.py:300` computes exactly this formula. Tested numerically: baseline gives $K^*_H = 0.1654$, $\beta_H = 1.546$, $\psi_H = 1.275 \in (1, 1.5)$ — interior condition satisfied.

*Passes.* Proposition 1(b) states $X^*(s) = \beta_s[e^{-r\tau}\delta K^*/r + I(K^*)] / [(\beta_s-1)e^{-r\tau}\phi_s(K^*)]$. The code at `phase1_base_model.py:225-230` computes `num = beta*(d*delta*K/r + I(K))`, `den = (beta-1)*d*phi`, yielding exactly the proposition formula (where `d = e^{-rτ}`). Value-matching and smooth-pasting verified by `test_value_matching_H` and `test_smooth_pasting_H`, both passing.

*Passes.* The coupling coefficient $D_L = -\lambda A_H / \text{char}_L(\beta_H)$ is implemented at `phase1_base_model.py:394-396`. The denominator `char_L_at_beta_H` is correctly computed as $\frac{1}{2}\sigma_L^2\beta_H(\beta_H-1) + \mu_L\beta_H - (r+\lambda)$. Sign is confirmed positive by test `test_switching_premium_positive`.

*Minor issue.* Proposition 1(c) states "$X^*_L \leq X^*_H$ when $\lambda$ is sufficiently large." The actual baseline gives $X^*_L = 0.0070 \ll X^*_H = 0.0621$, consistent with the claim. However the ordering is reversed from what the text says: the paper at line 319 states "regime comparison… $X^*_L \leq X^*_H$" but the preceding text says "Higher λ lowers $X^*_L$" and "firm waits longer in low regime." The comparison is correct economically — in the low regime the firm has a lower trigger (invests earlier to avoid missing the adoption event) — but the notation in the proposition is confusing: the trigger $X^*_L$ is lower than $X^*_H$, meaning the firm invests earlier at a lower demand level, not later. The phrase "waits longer" at `phase1_base_model.py:690` ("X*_L / X*_H > 1: firm waits longer in low regime") is wrong — the ratio is 0.0070/0.0621 = 0.113 < 1, and the assertion is inverted.

**Propositions vs. code — Phase 2 (Propositions 2–3):**

*Passes.* Proposition 2(a) asserts $X^*_L < X^*_F \leq X^*_{\text{single}}$. Baseline gives $X_L = 0.0158$, $X_F = 0.0710$, $X_{\text{single}} = 0.0621$. Note that $X_F = 0.0710 > X_{\text{single}} = 0.0621$: the follower's trigger exceeds the single-firm benchmark, which is consistent with the follower entering a duopoly market (lower revenue share), requiring a higher demand level to break even. This is economically correct.

*Passes.* Proposition 2(b): preemption gap = $(X_{\text{single}} - X^*_L)/X_{\text{single}} = 74.5\%$. The paper text says "approximately 70%" which is close; the discrepancy is a rounding statement, not a bug.

*Has issues — Proposition 2 proof (Appendix A.2).* The proof asserts "The Stackelberg trigger $X_L^{\text{Stack}} = X_{\text{single}}$ (the single-firm trigger coincides with the Stackelberg leader trigger in this symmetric setup)." This claim is incorrect. In the standard Stackelberg real-options game, the leader earns monopoly profits initially and then duopoly profits after the follower enters. The leader's optimal unconstrained trigger is generally not equal to the single-firm benchmark; it depends on the follower's response function and the competitive externality. Asserting their equality without proof is a gap.

*Cannot verify — Proposition 3 (Competition-Leverage Spiral).* No proof is provided in the Appendix. Parts (a) and (b) are standard risk-shifting arguments (Jensen, 1976; Brander & Lewis, 1986) that could be cited rather than proved, but part (c) involves a specific default probability formula that is stated without derivation. The formula in Proposition 3(c) matches the first-passage time formula implemented at `phase2_duopoly.py:253-281`.

**Propositions vs. code — Phase 4 (Proposition 4, Dario Dilemma):**

*Has issues.* Proposition 4(a) states the revenue gap is "approximately $-\epsilon \cdot \alpha \cdot \text{NPV}_{\text{opt}} \cdot T$ per unit of capacity shortfall." The code at `phase5_valuation.py:219-253` computes forgone revenue as:
```python
X_mid = 0.5 * (X_true + X_conserv)
expected_delay = log(X_conserv / X_true) / mu_eff
forgone = K_alpha * X_mid * pv_factor
```
This is a midpoint-GBM approximation that does not reduce to the proposition formula. The proposition formula is proportional to $\alpha$, but the code computes `K_alpha = K_star**p.alpha` and multiplies by $X_{\text{mid}}$, not $\alpha \cdot \text{NPV}_{\text{opt}} \cdot T$. These formulas are quantitatively different.

*Cannot verify — Proposition 4 proof.* No proof is provided in the Appendix.

**Numerical methods:**

*Passes.* The characteristic root computation in `phase1_base_model.py:91-95` uses the quadratic formula directly and is verified by `test_char_eq_satisfied_H/L` (both pass). Bisection/brentq root-finding in `phase1_base_model.py:266` uses `xtol=1e-10, maxiter=500` — adequate tolerance.

*Passes.* The backward induction in `phase3_nfirm.py:209-325` uses `brentq` with `xtol=1e-10, maxiter=500` for each preemption condition.

**Parameter consistency:**

*Passes.* Table 1 in the paper lists $\mu_L=0.03$, $\mu_H=0.08$, $\sigma_L=0.30$, $\lambda=0.20$, $\alpha=0.45$, $\gamma=1.50$, $r=0.15$, $\tau=1.5$. All match `ModelParams` defaults at `phase1_base_model.py:54-70`. $\sigma_H=0.25$ and $\delta=0.05$ are used in code but absent from Table 1 — these should be added.

**Regime-switching:**

*Passes.* The L-regime HJB is $0 = \frac{1}{2}\sigma_L^2 X^2 F_L'' + \mu_L X F_L' - (r+\lambda)F_L + \lambda F_H$. The code implements this correctly through the characteristic root with `extra_discount=lam` at `phase1_base_model.py:87-96` and the coupling coefficient at `phase1_base_model.py:393-396`. The absorbing H-regime is correctly handled with no coupling back.

---

### 2. Code Quality and Testing

**Test coverage:**

All 117 tests pass (verified with `python -m pytest src/tests/ -q`, runtime ~48s). Coverage is reasonable across all five phases. However, `matplotlib` must be installed separately — it is not in the virtual environment by default, causing all tests to fail on a fresh install until `pip install matplotlib` is run. There is no `requirements.txt`, `pyproject.toml`, or other dependency specification file visible in the repository root.

**Test meaningfulness:**

*Passes.* Tests check economically meaningful properties: characteristic equation roots satisfy their defining equations (`test_char_eq_satisfied_H/L`); smooth-pasting and value-matching are verified (`test_value_matching_H`, `test_smooth_pasting_H`); option value is monotone increasing in $X$; credit spread increases near default; default probability is in $[0,1]$; revealed beliefs inversion recovers true $\lambda$ from model-generated data (`test_point_estimate_recovers_lambda`).

**Missing test — N-firm trigger monotonicity:**

*Has issues.* There is no test verifying that the N-firm equilibrium triggers are monotonically ordered ($X_1^* < X_2^* < \cdots < X_N^*$). Running the model confirms a violation:

```
N=4 Triggers: [0.0156, 0.0852, 0.0486, 0.1704]
Monotone increasing: False
VIOLATION: triggers[1]=0.0852 > triggers[2]=0.0486
```

This is a significant bug. The second entrant's trigger (0.0852) exceeds the third entrant's trigger (0.0486), violating the sequential equilibrium ordering. The backward induction for $k=2$ computes an unconstrained leader trigger (≈ duopoly single-firm benchmark of 0.085) that exceeds $X_3^* = 0.0486$, and then fails to apply the preemption condition correctly. The `test_accordion_effect_positive` test only checks that the first-mover trigger is lower than the single-firm benchmark, but does not check internal monotonicity.

**Test quality issue — trivially true bound:**

`test_accordion_effect_bounded` at `test_phase3.py:105-108` checks `assert 0 < ae < 100`. Since `accordion_effect` is a fraction in $(0,1)$, the upper bound of 100 is trivially satisfied. The test should check `0 < ae < 1`.

**Edge cases:**

*Partially covered.* The λ=0 edge case is tested implicitly via `agr_premium` (which uses λ=0.001 as a near-zero proxy). The N=1 case raises an assertion error as designed. There is no test for $\sigma \to 0$ (deterministic limit) or $\tau \to 0$ (no time-to-build).

**Numerical stability:**

*Passes mostly.* Division-by-zero guards are present throughout (e.g., `if den <= 0: return np.inf`). The `brentq` root-finding has try/except fallbacks. One minor concern: `_optimize_K_numerically` uses a grid of 200 points over $\log K \in [-4, 3]$, then attempts brentq on a neighborhood. The neighborhood width (±5 grid points = ±0.035 in log-space) may miss the true optimum if the objective is multimodal.

**Dead code:**

`phi_L_effective` is defined at `phase2_duopoly.py:483` but never called. The function is referenced only in its own definition line. It should be removed or documented as aspirational.

**Code organization:**

*Passes.* Phases are cleanly separated. Each file is self-contained with a `__main__` block. The inheritance chain `ModelParams → DuopolyParams → NFirmParams` is logical. The use of `dataclass` for parameters is clean.

**Reproducibility:**

*Partially.* Running `python src/phase1_base_model.py` through `python src/phase5_valuation.py` reproduces all figures. However, the paper.qmd embeds figures as static `.png` files for Phases 3–5 (rather than generating them inline), so figures are not regenerated when the paper is compiled. The paper compilation is therefore not fully reproducible from the paper source alone.

---

## Part 2: Paper Review

### 3. Paper Content Review

#### 3a. Structure and Argument

**Motivation:**

*Passes.* The introduction is compelling. The $300B annual capex figure is striking and well-placed. The "six ingredients" framing is clear and provides a useful roadmap. The claim of novelty ("absent from any single prior paper") is reasonable given the combination, though each individual ingredient is present in existing work.

**Literature positioning:**

*Has minor issues.* The literature review is generally adequate. Notable omissions:
- Lambrecht & Perraudin (2003, *Review of Financial Studies*): preemption under incomplete information — directly relevant to the revealed beliefs methodology.
- Pawlina & Kort (2006, *Journal of Economics & Management Strategy*): asymmetric firms in real options games.
- Morellec & Schürhoff (2011, *Journal of Finance*): dynamic investment and financing decisions with strategic uncertainty.
- The Grenadier (2002) citation is for Cournot competition eroding option value, but Grenadier (2002) actually solves a competitive equilibrium in continuous time with all firms investing simultaneously — the preemption/sequential-entry game is closer to Fudenberg & Tirole (1985) and Huisman & Kort (1999).

**Model building:**

*Mostly passes.* Phase 1 → Phase 2 is natural. Phase 2 → Phase 3 drops the Leland default structure: the N-firm model in Phase 3 (`phase3_nfirm.py`) uses a pure option model without endogenous default, so the "competition-leverage spiral" established in Phase 2 is not active in the N-firm analysis. This should be acknowledged explicitly.

**Identification:**

*Has issues.* The revealed beliefs methodology inverts a *single-firm* model to extract λ. But AI labs operate in oligopoly. A firm's observed investment behavior is jointly determined by its own λ belief AND the strategic behavior of rivals. Using the single-firm model for inversion conflates the two. The paper does not formally establish that this identification strategy is valid under competition, nor does it conduct a robustness check using the duopoly model.

Additionally, the stylized firm data (K_M, X_M in `phase4_revealed_beliefs.py:35-63`) appears to be chosen to produce the desired λ̂ ranges rather than calibrated from public data. For example, Anthropic is assigned K_M=0.10, X_M=0.021, which gives λ̂≈0.49, slightly above the paper's stated range of 0.30–0.45. The methodology section should explain how K_M and X_M are derived from actual data (capacity in GW, revenue in $B) via a documented unit-conversion procedure.

**Conclusion:**

*Passes.* Concise and accurate. Does not overclaim.

#### 3b. Writing Quality

**Clarity:**

*Minor issues.* Section 2.1 (Single-Firm Benchmark) contains a confusing statement at line 319: Proposition 1(c) states $X^*_L \leq X^*_H$ "when λ is sufficiently large." But this ordering ($X^*_L < X^*_H$) means the firm invests at a *lower* demand level in the L regime than in H, i.e., preempts the regime switch by investing early in the L state. The subsequent bullet "Higher λ lowers $X^*_L$" is consistent, but the surrounding text says "firm waits longer in the low regime" which contradicts the ordering. This confusion also appears in `phase1_base_model.py:690`: the comment "firm waits longer in low regime ✓" is wrong for baseline parameters.

The footnote-style remark at line 471: "large $\beta_H/(\beta_H-1)$ term arising from the near-unit-root demand process (low $r - \mu_H$)" — this is slightly misleading. The option multiplier $\beta/(\beta-1)$ is large because $\beta_H \approx 1.55$ is close to 1, which is itself a consequence of low $r - \mu_H = 0.07$. The description "near-unit-root" is technically accurate but unconventional.

**Notation:**

*Minor issues.* The paper uses $\psi_s = \alpha\beta_s/(\beta_s-1)$ defined at Proposition 1 but this symbol is not introduced before use. The symbol $\theta$ (duopoly profit share) is introduced in Section 3 without a forward reference; readers of Section 2's Table 1 see it without definition.

Table 1 omits $\sigma_H = 0.25$ and $\delta = 0.05$, both of which are baseline parameters used in the code.

**Length and focus:**

*Minor issue.* The training/inference allocation (Section 3.2) is somewhat disconnected from the main competition-leverage spiral narrative. The section is brief (one paragraph) and the figure is referenced as a static `.png`. It would benefit from either more development or trimming.

**Abstract:**

*Mostly passes.* The abstract quantifies key results: "20–45% annual probability," "$2–4B over five years," "15–25% five-year default probability." These are model-generated numbers from the Dario Dilemma analysis. One concern: the actual code produces λ̂ ≈ 0.49 for "Anthropic-like" stylized firms, which is slightly above the stated "30–45%" range. The abstract should match the model output precisely.

#### 3c. Journal Fit

**Contribution significance:**

The paper's combination of six modeling ingredients is novel, and the revealed beliefs methodology is a genuine contribution. The "Dario dilemma" framing is intuitive and policy-relevant. However, top finance journals (JF, RFS) will require that the welfare and policy implications be more formally developed, and that the calibration be tied to actual firm data rather than stylized representations.

**Methodological rigor:**

The single-firm and duopoly solutions are analytically clean and well-derived. The N-firm extension uses a sound backward-induction approach, but the trigger monotonicity bug undermines confidence in those results. The revealed beliefs methodology is conceptually sound but the identification argument needs strengthening.

**Formatting and conventions:**

The paper uses `callout-note` environments for propositions, which renders well in HTML/PDF but is non-standard for finance journals. Most journals expect the standard LaTeX `\begin{proposition}...\end{proposition}` environment, which the preamble does define (`\newtheorem{proposition}{Proposition}`). The `.qmd` source should be restructured to use these environments for a submission-ready document.

**Recommended target journal:**

*Review of Financial Studies* (RFS) is the best fit. The paper bridges corporate finance (default risk, capital structure) and real options (irreversible investment under uncertainty) — exactly the intersection RFS has published extensively. The *Journal of Finance* is a reasonable second choice. Econometrica would require substantially more mathematical rigor (full proofs for all propositions, welfare analysis with formal optimality characterization). AER would require the AI policy/welfare angle to be the dominant contribution rather than the model mechanics.

---

### 4. Figures

**Figure 1 (Single-firm, `fig1_single_firm.pdf`):**

*Passes.* Generated inline in `paper.qmd:341-413` using actual model computation. Four panels: (a) option value by regime, (b) trigger vs. λ, (c) trigger vs. σ_H, (d) K* vs. α. Panel (d) shows K*_H and K*_L both increasing in α, which is correct per the formula $K^* \propto [(\psi-1)/(\gamma-\psi)]^{1/(\gamma-1)}$ where $\psi = \alpha\beta/(\beta-1)$ increases in α. One issue: the label in panel (b) says "High regime (post-adoption)" X*_H is shown as *flat* vs. λ (as expected from theory, since X*_H is λ-independent), but the legend says both H and L regime — the flat line for H is correct and should be noted in the caption.

**Figure 2 (Duopoly, `fig2_duopoly.pdf`):**

*Passes.* Also inline. Panel (a) shows Leland equity/debt curves with annotated default boundaries. Panel (b) shows the competition-leverage spiral as a bar chart. Panel (b) caption correctly describes the three-step decomposition.

**Figure (N-firm, `phase3_timeline.png`, static):**

*Has issues.* This figure is referenced as a static image and shows "74.8% compression" in the caption. The 74.8% figure is correct (accordion_effect = 0.748 from code). However, the figure presumably shows the four trigger levels, which as documented above are not monotonically ordered. If the figure plots triggers in entry order (0.0156, 0.0852, 0.0486, 0.1704), this will be visually puzzling to readers.

**Figure (Revealed beliefs, `phase4_cross_firm_lambda.png`, static):**

*Has issues.* The figure shows λ̂ ≈ 0.49 for Anthropic (based on code output), but the paper text states 0.30–0.45 and Table 2 shows 0.35–0.45. The discrepancy (code: 0.49, paper text: ≤0.45) should be reconciled.

**Tracing code-figure consistency (3 figures spot-checked):**

1. *Figure 1, Panel (b)*: `paper.qmd:378-385` calls `model_sf.comparative_statics('lam', lam_vals)` and plots `cs_lam['X_star_L']`. This correctly traces through `SingleFirmModel.comparative_statics → solve()` with varying λ. Verified: X*_L decreases monotonically with λ. ✓

2. *Figure 2, Panel (b)*: `paper.qmd:507-565` calls `model_duo.competition_leverage_spiral('H')` and reads `spiral['steps']`. Traces to `DuopolyModel.competition_leverage_spiral → preemption_equilibrium`. ✓

3. *Revealed beliefs figure*: `phase4_revealed_beliefs.py:342-380` (function `plot_cross_firm_lambda`) calls `apply_to_all_firms` → `infer_lambda`. The function is called with `regime='L'` in `apply_to_all_firms`, which is correct. The values plotted correspond to the code output (λ̂ ≈ 0.49 for Anthropic). ✓ (but inconsistent with paper text)

---

### 5. Calibration and Results

**Parameter values:**

*Mostly reasonable.* μ_L=0.03 (pre-2023 AI revenue growth, plausible), μ_H=0.08 (2023–2025 enterprise AI, plausible), σ_L=0.30 (cross-lab variance, plausible), r=0.15 (AI startup WACC, on the high side for hyperscalers like Google/Microsoft but appropriate for Anthropic). The α=0.45 derivation via the chain (scaling law exponent 0.10) × (demand elasticity 4.5) is conceptually reasonable but the demand elasticity estimate (3–5×) has no citation.

*Issue.* The discount rate r=0.15 is used for both high-leverage startups (Anthropic) and large-cap hyperscalers (Google, Microsoft). A more careful treatment would use firm-specific WACCs (Google ≈ 0.08–0.10; Anthropic ≈ 0.15–0.20). Using a single r for all firms in the revealed beliefs analysis conflates λ heterogeneity with WACC heterogeneity.

**Sensitivity:**

*Partially covered.* The code includes `RevealedBeliefs.sensitivity_analysis()` with 8 scenarios, and `infer_lambda` provides CI bands for ±20% α and ±0.5yr τ. The paper does not present a sensitivity table for the Dario dilemma results or for the preemption gap.

**Comparative statics:**

*Passes.* The paper's comparative static claims are consistent with model output: λ↑ lowers X*_L (verified), σ↑ raises X* (verified), α↑ raises K* (verified), θ↓ lowers X_L* (verified by test_theta_higher_reduces_preemption).

**Revealed beliefs results:**

*Has issues.* The model returns λ̂ ≈ 0.49 for Anthropic, 0.41 for OpenAI, 0.16 for Google DeepMind, 0.14 for Microsoft Azure AI. The paper text states 0.30–0.45 for Anthropic-like firms, which is inconsistent with the computed 0.49. The "monotone increase" in revealed λ from 2022 to 2025 (stated in the text) holds for the trajectory data in `lambda_time_series()`, but this trajectory is hypothetical, not sourced from actual annual investment data. This should be clearly labeled as illustrative.

**Growth decomposition:**

*Passes.* The V_AIP / V_expand / V_λ decomposition in `phase5_valuation.py:96-130` is internally consistent. The AGI premium (V_λ) correctly computes the incremental value from having λ > 0 vs. λ ≈ 0, following the Berk, Green & Naik (1999) growth option framework.

---

## Summary of Issues

### Critical Issues

1. **N-firm trigger monotonicity violation** (`phase3_nfirm.py:253-307`): For N=4, triggers[1]=0.0852 > triggers[2]=0.0486, violating the sequential equilibrium ordering X₁* < X₂* < X₃* < X₄*. The backward induction for k=2 incorrectly applies the unconstrained Stackelberg trigger without checking the monotonicity constraint. **Fix**: add an explicit constraint that X_k* ≤ X_{k+1}*, and revisit the preemption condition for the case where V_lead and V_follow do not intersect below X_next. Add `test_triggers_monotone_increasing` to `test_phase3.py`.

2. **`infer_lambda` default regime bug** (`phase4_revealed_beliefs.py:138`): The method signature `def infer_lambda(self, K_obs, X_obs, regime: str = 'H', ...)` defaults to H regime, which produces degenerate results (λ̂ = 0.99) because X*(K, λ) is constant in H. All correct usage specifies `regime='L'`, but the incorrect default is a trap. **Fix**: change default to `regime='L'`. Add a test calling `infer_lambda` without specifying regime and verifying a reasonable result.

3. **Missing Appendix proofs for Propositions 3 and 4**: The Appendix provides A.1 (Proposition 1), A.2 (Proposition 2, with a gap noted above), and A.3 (numerical method). Propositions 3 and 4 have no proofs at all. For submission to JF or RFS, either provide full proofs or explicitly classify these as "results" or "observations" supported numerically.

### Major Issues

4. **N-firm section omits default risk**: Phase 3 drops the Leland endogenous default structure from Phase 2. The paper's central mechanism (competition-leverage spiral) is therefore absent from the N-firm analysis. The paper should either integrate default risk into the N-firm model, or explicitly state and justify the simplification.

5. **Proposition 1(c) / code comment contradiction**: The paper implies firms in the L regime have a higher trigger than H ("waiting longer in low regime"), but the actual model output and economic logic show X*_L < X*_H (firms invest *earlier* in L regime to preempt the adoption event). The comment at `phase1_base_model.py:690` says "X*_L/X*_H > 1: firm waits longer in low regime ✓" but the ratio is 0.113, not >1. **Fix**: correct the comment and clarify Proposition 1(c)'s ordering.

6. **Revealed beliefs: single-firm identification under oligopoly**: The inversion uses a single-firm model but the described firms (Anthropic, OpenAI, Google, Microsoft) are strategic oligopolists. The observed investment behavior is a Nash equilibrium outcome, not a single-firm optimum. The paper needs an identification argument for why single-firm inversion is valid in this setting, or should use the duopoly model for inversion.

7. **Proposition 4 forgone-revenue formula mismatch**: The stated formula ($-\epsilon \cdot \alpha \cdot \text{NPV}_{\text{opt}} \cdot T$) does not match the code's approximation (`phase5_valuation.py:247-253`). Either derive the approximation and explain it, or correct the proposition statement.

8. **Stylized firm calibration transparency**: The K_M, X_M values in `STYLIZED_FIRMS` appear chosen to produce target λ̂ ranges rather than derived from public capacity/revenue data via a documented conversion. The mapping from GW capacity and $B revenue to model units (K_M, X_M) should be explicitly stated.

### Minor Issues

9. **BibTeX duplicates**: `references.bib` contains `bouis2009number` and `bouis2009investment` pointing to the same paper. It also contains `hackbarth2014competition` and `hackbarth2014capital` pointing to the same paper. Both pairs reference identical journal/year/author metadata. **Fix**: remove duplicates, standardize citation keys.

10. **Table 1 missing parameters**: $\sigma_H = 0.25$ and $\delta = 0.05$ are used in the model but absent from Table 1. Add them.

11. **`phi_L_effective` dead code**: Defined at `phase2_duopoly.py:483` but never called. Remove or use.

12. **Weak test bound**: `test_accordion_effect_bounded` checks `0 < ae < 100` but `ae` is a fraction ∈ (0,1), making the upper bound trivially satisfied. **Fix**: check `0 < ae < 1`.

13. **No dependency specification**: `matplotlib`, `scipy`, `pandas`, `sympy` must be installed manually. Add a `requirements.txt` or declare dependencies in `pyproject.toml`.

14. **Static figures in paper**: Figures for Phases 3–5 are static `.png` files, breaking reproducibility from paper source. Prefer inline computation or a `generate_figures.py` script called from the Quarto document.

15. **`guo2005irreversible` citation title**: The bib entry for `@guo2005irreversible` has title "Optimal Stopping Problems for Brownian Motion with Poisson Jumps," which is a different paper from the Guo, Miao & Morellec (2005) regime-switching investment paper typically cited in this context. Verify and correct.

---

## Overall Recommendation

**Revise and resubmit (major revision needed).**

The paper has genuine conceptual contributions — the competition-leverage spiral is well-motivated and illustrated, and the revealed beliefs methodology is novel — but two critical technical issues (N-firm trigger ordering, missing proofs) and one major conceptual issue (single-firm inversion under oligopoly) must be resolved before submission.

**Recommended target journal: Review of Financial Studies (RFS).**

The combination of endogenous default risk, real options, and oligopoly competition is precisely the intersection where RFS has a strong track record (Grenadier 2002 RFS, Hackbarth & Morellec 2014 JF would be peers). The paper's structure and length are appropriate. Once the N-firm bug is fixed and the proofs are added, RFS is the natural first submission. *Journal of Finance* is a strong second choice if the policy/welfare angle is developed further. Econometrica would require a substantially higher bar for mathematical rigor and formal proof.
