# Wilson-line-CFT

A physics-informed neural network (PINN) approach to the 1D defect CFT
bootstrap for the 1/2-BPS Wilson line in 4d N=4 super Yang-Mills at
strong coupling, benchmarked against the numerical conformal bootstrap
and the reinforcement-learning approach of BootSTOP/MultiSTOP.

**Status:** Completed: special functions, coupling functions, conformal
blocks, integral-constraint module, strong-coupling reference data, convex
baselines, PINN investigation (spectrum given), convex baseline extended
with integral constraints. Not yet attempted: the exact (coupling-
independent) Curvature function from eq. (2.12) — derived and verified
in `docs/derivations.md`, not yet implemented numerically — and folding
the integral constraints into the PINN fit — see "What this motivated"
below.

This README has two parts. Part 1 explains the physics motivation and
findings, for readers coming from the bootstrap/hep-th side. Part 2
documents the engineering — architecture, training methodology,
reproducibility — for readers coming from the ML side. Numerical results
are stated once, in Part 1; Part 2 references them rather than repeating
them.

---

# Part 1: Physics

## Identifiability: Why This Problem Is Hard

![Singular value spectrum vs coupling](notebooks/assets/identifiability_svd_sweep.png)

The design matrix's singular values, swept across the coupling g, reveal
*before any fitting is attempted* which operators can be individually
resolved and which cannot. At g→0, six singular values collapse toward
zero — matching exactly the six redundant directions predicted by the known
free-theory degeneracy structure (a pair of states sharing one dimension,
a sextet sharing another: 1 + 5 = 6). As g increases, this conditioning
loosens, but not uniformly: by g=4, the original sextet has reorganized
into a quartet plus two singletons, visible directly in how the smallest
singular values evolve. This figure is the evidence that motivates
everything that follows — any method that fits the crossing equation
independently at each g inherits this ill-conditioning.

## Convex Baseline with Smoothness Regularization

**Status: complete.** This investigation tests a specific question before
any neural network is introduced: *does smoothness-in-g alone, with no
learned representation, resolve the near-degeneracy the identifiability
analysis predicted would be unresolvable pointwise?*

### Background

Ordinary non-negative least squares at a single g cannot distinguish
degenerate states. This investigation asks whether *sharing information
across g*, via a smoothness prior on each state's `C_n²(g)` curve, is
enough to break that degeneracy on its own.

### Method

Each state's OPE-squared coefficient is represented as `C_n²(g) = Σ_k θ_{n,k}
φ_k(g)`, a shared B-spline basis (`src/spline_basis.py`) with coefficients
`θ` common across the entire crossing-equation system (`src/stack_crossing.py`).
The system is solved as a single convex QP (`src/convex_baseline.py`):
minimize the stacked crossing-equation residual plus a second-derivative
smoothness penalty on `θ`, subject to `θ ≥ 0`.

The regularization weight `λ` was chosen two independent ways: 5-fold
cross-validation (`src/cross_validation.py`), and a systematic 15-point
sweep against the known bootstrap bounds (`src/bounds_sweep.py`), using
`data/ope_bounds_strong_coupling.csv`.

### Result

![Fitted C_n² vs known bounds at the best λ](notebooks/assets/convex_baseline_vs_bounds.png)

**Both selection methods independently agree on `λ ≈ 0.01`.** At this
jointly-optimal `λ`, state 1 (non-degenerate) fits the known value closely.
States 2 and 3 do not: the total violation of known bounds is `0.566`, not
zero. A manual sweep across `λ ∈ {1, 30, 100, 1000}` confirms this is not a
matter of having missed the right value — state 2 never converges to its
true value at any `λ` tested, and state 3's best-fitting `λ` is not
consistent across the g-range. State 1's excellent fit degrades once `λ`
is pushed hard enough to meaningfully move state 3, revealing a real
tradeoff rather than a free improvement.

### Honest limitations

- The bounds-sweep selection is informed directly by the same ground truth
  used to evaluate it. Independent cross-validation landing on the same
  `λ` substantially mitigates this concern.
- The B-spline settings (16 log-spaced interior knots, degree 3) were a
  reasonable first choice, never independently tuned against alternatives.

## PINN Investigation (Spectrum Given)

**Status: complete for the experiments described here.** This replaces the
spline's fixed basis with a small neural network, testing whether added
representational flexibility succeeds where the convex baseline did not.
The spectrum Δₙ(g) remains frozen exactly as before; only `C_n²(g)` is
learned. Architecture and training details are in Part 2.

### Pure crossing-equation loss

Trained with no smoothness term — minimizing only the stacked
crossing-equation residual, the same objective the convex baseline
minimizes. Result: violation score `2.75` — worse than the convex
baseline's `0.566`, despite the crossing residual itself converging to a
far smaller value (~0.005) than the spline ever achieved. A network
satisfying crossing more precisely while resolving the actual physics
question less well is consistent with, not contradictory to, the
identifiability finding: crossing constrains the *sum* of near-degenerate
states far more than their individual split, so a more flexible model with
no other bias is free to exploit that slack rather than resolve it.

### Smoothness-regularized loss

An explicit smoothness penalty — the network output's second derivative
with respect to g, computed via automatic differentiation — was added at
four weights: `λ ∈ {0, 0.001, 0.01, 0.1}`. Each was trained across 5
random seeds (0–4) to check whether the ranking found initially held up
under genuinely different initializations, not just different `λ`.

| λ | violation score (mean ± std) |
|---|---|
| 0 | 2.763 ± 0.061 |
| 0.001 | 3.152 ± 0.065 |
| 0.01 | 3.580 ± 0.203 |
| 0.1 | 3.961 ± 0.105 |

**The monotonic trend survives seed variation, with one honest nuance at
the top end.** For `λ=0 → 0.001 → 0.01`, every seed at one weight beats
every seed at the next — the full min-max ranges never overlap, not just
the means. Between `λ=0.01` and `λ=0.1`, the means stay cleanly ordered
and the ±1 std bands still don't overlap, but one `λ=0.01` seed (3.887)
edges past the lowest `λ=0.1` seed (3.821) — the trend holds on average,
without every single seed beating every single seed at the boundary. As
before, the best configuration across every PINN experiment has no
smoothness term at all, and even that configuration remains roughly 5x
worse than the convex baseline's best result.

Separately, locating the loss plateau's escape epoch automatically —
across all 20 runs, using a windowed-average detector robust to late-
training instability — placed it at `17130`–`20603` regardless of `λ`,
consistent with the single-seed observation above. One seed (of five)
showed the escape landing within 200 epochs of the same point across all
four `λ` values, suggesting the timing is substantially set by the
initialization itself — though this held for only one of the five seeds
tested, not universally.

### Honest limitations

- Only tested down to `λ=0.001`; a smaller value was not tried.
- This penalty (continuous second derivative via autograd) is not on a
  directly comparable scale to the spline's (discrete coefficient
  differences), so the precise claim is that *this specific* smoothness
  mechanism did not help this network, not a general claim about
  smoothness across representations.
- Five seeds is enough to confirm the trend is not a single-initialization
  artifact, but not enough to bound the variance precisely — a claim
  about, say, the 95th-percentile violation score at each `λ` would need
  more.

## Comparison to MultiSTOP

[Trenta, Bacciu, Cossu, Ferrero, arXiv:2404.14909](https://arxiv.org/abs/2404.14909)
independently documents the same phenomenon found here: when states are
near-degenerate, their *summed* OPE coefficients are resolved with high
precision, while the *individual* coefficients are not — and the paper
notes this degeneracy is more severe at weak coupling than at strong
coupling, precisely the regime tested in this project. The MultiSTOP
authors explicitly leave individual resolution of such states to future
work requiring additional constraints beyond plain crossing symmetry.

This project's convex baseline and PINN investigation — two structurally
unrelated optimization approaches, developed independently of MultiSTOP's
implementation — arrive at the same conclusion from scratch. Neither
approach here resolves the degeneracy any more than MultiSTOP does; the
value of this result is convergent evidence, from a third and fourth
independent method, that the limitation is structural to plain crossing
symmetry rather than an artifact of any one optimization approach.

## What This Motivated

The rigorous analytic/numerical bootstrap bounds used as ground truth
throughout this project (`data/ope_bounds_strong_coupling.csv`, from
Cavaglià–Gromov–Julius–Preti, arXiv:2203.09556) *do* resolve C₁², C₂², C₃²
individually — but that method combines crossing symmetry with
integrability input beyond what either the convex baseline or the PINN
have used. `src/constraints.py`'s `int1`/`int2` — the integral constraints
from the same bootstrappability framework, built and independently tested
since early in this project but never yet incorporated into a fit — are
linear in `C_n²`, so they could be added to the convex system without
sacrificing convexity, or as an additional loss term for the PINN. Folding
them into the convex baseline is now done — see "Integral Constraints
Extension" below — and resolves the degeneracy to within ~3%. The PINN
extension has not yet been attempted.

## Integral Constraints Extension

**Status: complete for the convex baseline.** This tests the concrete next
step flagged earlier — whether the two integral constraints from the same
bootstrappability framework, linear in `C_n²` and therefore foldable
directly into the existing convex QP, actually resolve the C₂²/C₃²
degeneracy that smoothness alone could not.

### Background

Neither the convex baseline nor the PINN, using only the crossing equation
plus smoothness regularization, could resolve the near-degenerate pair —
confirmed independently by both methods, and matching a limitation
MultiSTOP's own authors documented. The rigorous bootstrap succeeds where
these fail specifically because it combines crossing with *integrability*.
`src/constraints.py`'s `int1`/`int2` — eqs. (4.32)/(4.33) of
Cavaglià–Gromov–Julius–Preti, built and tested since early in this project
but never incorporated into a fit — are exactly that kind of additional,
integrability-derived information: two more linear constraints on `C_n²`,
with right-hand sides (`RHS1`/`RHS2` in `src/coupling.py`) built from the
Bremsstrahlung and Curvature functions.

One real restriction going in: `RHS1`/`RHS2` depend on `curvature_C`'s
strong-coupling asymptotic series, valid only for `g >= G_MIN = 3` — the
opposite end of the coupling range from where the degeneracy is worst.
Whether information from that region could still help at small g, through
the shared spline basis, was an open question rather than an assumption.

### Method

Two additional weighted terms enter the same convex objective:

```
minimize   ‖A_cross θ − b_cross‖²  +  w_int ‖A_int θ − b_int‖²  +  λ ‖D θᵀ‖²
subject to θ ≥ 0
```

Built as a separate module family (`src/integral_constraints.py`,
`src/convex_baseline_with_integrals.py`, `src/violation_analysis.py`,
`src/integral_bounds_sweep.py`) rather than modifying the tested Stage 0.5
code, following the same discipline as `solve_from_cached` earlier in the
project. `w_int = 0` was verified to exactly reproduce the original convex
baseline's objective value, confirming the extension is a strict
generalization.

Evaluation moved from the single aggregate `violation_score` to a
per-state, per-coupling breakdown (`violation_by_state`) plus a direct
fitted-to-true ratio at a probe coupling (`ratio_to_truth`) — the aggregate
metric was found, empirically, to hide exactly the signal that mattered:
it stayed flat across a wide range of `w_int` while the individual C₂²/C₃²
ratios moved from over-estimating to under-estimating the true value
across that same range, crossing through the correct answer in between.

`λ` and `w_int` were searched jointly, not separately, since adding a third
term to the objective gives no reason to assume the pre-integral-constraint
optimal `λ` still applies. A coarse log-spaced grid was followed by a
second, finer grid centered on the coarse optimum, to confirm the result
was a genuine local optimum rather than an artifact of grid resolution.

### Result

**At `λ ≈ 0.001`, `w_int ≈ 4×10⁶`, both C₂² and C₃² land within ~3% of
their known bootstrap values simultaneously** — down from `3.97×` and
`0.47×` respectively with no integral constraints at all. The refined grid
search converged to the same neighborhood the coarse grid found
(`3.6%` → `3.18%` worst-case deviation), confirming a real local optimum.

Two further findings from the sweep, both suggesting a genuine, non-trivial
interaction rather than a single well-placed hyperparameter:

- **`λ = 0` is numerically unstable** — the fit jumps erratically between
  distinct configurations depending on solver initialization, rather than
  converging cleanly. Neither smoothness nor the integral constraints
  suffice alone; the combination is what makes the problem well-posed.
- **`λ = 1.0` is essentially unresponsive to the integral constraints** —
  C₂² stays above `3.9×` across the entire `w_int` range,
  barely moving even at `w_int = 10⁸`, where it finally drops to `3.98×`; still nowhere the real value.
  Too much smoothness weight locks the fit into a configuration the integral term can't meaningfully shift.

Information also propagates *backward* in coupling through the shared
spline basis: the constraints are imposed only at `g ≥ 3`, yet the largest
improvements land at `g ≈ 2.0–2.5`, outside the constrained region,
decaying to negligible effect below `g ≈ 0.5`.

### Honest limitations

- **The `g ≥ G_MIN` restriction is still real.** The exact, coupling-
  independent expression for the Curvature function (eq. 2.12, a double
  contour integral) has been fully derived and verified analytically —
  see `docs/derivations.md` — but not yet implemented numerically. Until
  it is, the constraints cannot be imposed anywhere near where the
  degeneracy is actually worst; the current result works entirely through
  the spline basis's propagation of information from `g ≥ 3` downward.
- **`w_int` selection is informed by the same ground truth used to
  evaluate it** — the same caveat as the original `λ` sweep. It is
  meaningfully weaker here, though: `w_int` showed a wide plateau of
  near-equally-good values above `~10⁶`, rather than a sharp, delicately-
  tuned peak, so the result is not sensitive to precisely which value in
  that range was chosen.
- **Only the convex baseline has been extended this way.** Whether adding
  the integral constraints as an additional PINN loss term succeeds where
  the smoothness-penalty experiment failed — plausible, given the
  reward-shaping difficulty MultiSTOP reported when combining objectives
  in an RL framework, versus straightforward loss addition in a
  differentiable-programming setup — has not been tested.


## References

- Ferrero, Meneghelli, arXiv:2103.10440, arXiv:2312.12550, arXiv:2312.12551
- Cavaglià, Gromov, Julius, Preti, arXiv:2203.09556
- Trenta, Bacciu, Cossu, Ferrero, arXiv:2404.14909

---

# Part 2: Engineering

## Repository structure

    src/          core physics modules: special functions, coupling
                  functions, conformal blocks, the crossing equation,
                  integral constraints, convex baseline, PINN model and
                  training
    data/         data-extraction scripts and the resulting reference CSVs
    models/       trained PINN weights, one file per λ configuration
    tests/        unit tests, one file per src/ module
    notebooks/    supporting figures and derivations

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Convex Baseline: Formulation and Solver Engineering

**Formulation (`src/convex_baseline.py`, extended in
`src/convex_baseline_with_integrals.py`).** Representing `C_n²(g)` as a
B-spline expansion turns the crossing equation — linear in `C_n²` at any
fixed `g` — into a system still linear in the spline coefficients `θ`,
now stacked across every training `g` simultaneously
(`src/stack_crossing.py`). The integral-constraint extension adds two more
linear rows per constrained coupling in exactly the same system, rather
than requiring a different formulation. Non-negativity is enforced as a
hard constraint, `θ ≥ 0`, rather than architecturally as in the PINN — and
because B-spline basis functions are themselves non-negative and sum to
one at every point, `θ ≥ 0` guarantees `C_n²(g) ≥ 0` across the *entire*
continuous curve, not just at the training points.

**Solver selection and numerical conditioning.** `solve_convex_problem`
(the original Stage 0.5 baseline) calls `cvxpy`'s solver with no explicit
choice, which auto-selects `OSQP` for this problem class. Every function
built afterward (`solve_from_cached`, `solve_convex_problem_with_integral_constraints`,
`solve_from_cached_with_integrals`) explicitly requests `CLARABEL` instead,
chosen for better-observed conditioning on the badly-scaled objectives the
integral-constraint sweep produces — `w_int` ranges from `10⁴` to `10⁸`,
a scale spread that stresses generic QP solvers. Both solvers were
confirmed to agree on the underlying optimization: at matched inputs, the
two converge to the same objective value to within `~2×10⁻⁵` relative,
despite the raw coefficient vectors differing by up to `5×10⁻⁵` in the
sextet states specifically — the same near-null-space directions flagged
by the identifiability analysis. Two solvers landing at different points
of an equally-optimal, nearly-flat region is expected behavior for a
near-degenerate convex problem, not a sign of error; it is also why the
project's tests compare achieved objective values rather than raw `θ`
where an exact-generalization claim is being verified, since the latter is
sensitive to noise the former is not.

**Caching for fast hyperparameter search.** As with the PINN's physics
cache, the design matrices, target vectors, and smoothness-penalty
operator depend only on the training `g`-values and spline settings — never
on the regularization weights being searched over. `solve_from_cached` and
`solve_from_cached_with_integrals` isolate the optimization step from this
expensive setup, so cross-validation, the bounds sweep, and the joint
`(λ, w_int)` grid search each build the underlying matrices exactly once
and reuse them across every candidate — the same underlying pattern as
`pinn_loss.py`'s `build_physics_cache`, applied to the convex side of the
project.

**Solver status is checked, not assumed.** Every solve function accepts
`cp.OPTIMAL` or `cp.OPTIMAL_INACCURATE` and rejects anything else with a
raised error, and the status string is returned alongside the fitted
coefficients rather than discarded — callers that need to distinguish a
clean solve from a numerically marginal one can do so, rather than
silently trusting whatever the solver returned.

**Grid search: coarse-to-fine, not single-pass.** The `(λ, w_int)` search
for the integral-constraint weight ran in two stages: a coarse,
log-spaced grid across four orders of magnitude in each parameter,
followed by a second grid at finer resolution centered on the coarse
optimum. The refined search converging to the same neighborhood the
coarse one found (`3.6%` → `3.18%` worst-case deviation) was the check
that the result was a genuine local optimum rather than an artifact of
where the first grid happened to sample.

## Seed-Robustness Infrastructure

**Parallelization.** Five seeds across four `λ` values means 20 independent
training runs. Since each is fully independent — nothing about seed `k`'s
training depends on seed `k-1`'s — they parallelize trivially across CPU
cores via `concurrent.futures.ProcessPoolExecutor`, rather than running
sequentially. Two things matter for getting this right on a CPU-only
PyTorch workload specifically:

- **Thread oversubscription.** PyTorch's CPU backend uses a multi-threaded
  linear algebra library internally by default. Spawning `N` worker
  processes that each *also* try to use every available thread creates
  severe contention — more threads competing than physical cores exist.
  Each worker calls `torch.set_num_threads(1)` immediately on start,
  making parallelism come entirely from process-level concurrency rather
  than fighting itself internally.
- **Per-worker cache rebuilding.** The physics cache (design matrices,
  target vectors) is a large object; passing it from the main process to
  every worker means serializing it repeatedly across process boundaries.
  Since building it costs seconds while training costs minutes, each
  worker rebuilds its own cache independently — trading a small amount of
  redundant, fully-parallel computation for much simpler code with no
  cross-process data sharing.

Measured on a 10-core machine: genuine contention under full 10-way
concurrency costs a real but modest `~1.7×` slowdown on both cache-building
and training, relative to running two jobs at a time. The full 20-job sweep
completed in under 10 minutes, against a naive linear extrapolation that
would have suggested several times that.

**Detecting the loss plateau's escape point reliably.** Training curves
show a long flat plateau followed by a sharp drop (see Part 1). Locating
that transition automatically, across 20 runs, turned out to need real
care. A first attempt — searching for the single largest one-epoch
relative decrease in the loss — was misled by late-training instability:
intermittent loss spikes (visible directly in the training logs), followed
by a one-epoch recovery back down, register as an even larger relative
drop than the genuine plateau-escape itself. The detected "escape" epoch
clustered suspiciously close to whatever boundary the search window
happened to end at, rather than settling at a consistent point — the
signature of a metric tracking search-window artifacts rather than a real
structural feature.

The fix compares windowed averages rather than single epochs — the mean
loss over `W` epochs before a candidate point against the mean over `W`
epochs after it — so a transient spike lasting a few epochs gets diluted
across the window, while a genuine, sustained regime change survives
intact. Verified on a synthetic loss curve containing both a real
transition and an injected spike deliberately made larger than the
transition itself: the windowed metric correctly recovered the genuine
transition where the single-epoch version, as expected, picked the spike.
A window of 50 epochs, sufficient in that synthetic test, still clustered
near the search boundary on the real data — the late-training instability
here evidently spans longer than a brief spike. `W=500` resolved it
completely: detected escape epochs spread naturally across
`17130`–`20603`, closely matching an earlier single-seed observation of
`~13000`–`19000`, with none clustering near either search boundary.


## PINN Architecture and Training

**Model (`src/pinn_model.py`).** A single shared network with 10 output
heads — one MLP taking g in and predicting all 10 states' `C_n²` at once,
rather than 10 separate networks. Three hidden layers of 64 units with
`Tanh` activations, chosen deliberately over `ReLU`: the smoothness
experiment below requires the network's output to be twice differentiable
everywhere, which `ReLU`'s kink at zero would break. The final layer is
passed through `Softplus`, guaranteeing non-negativity by construction —
the network's architectural equivalent of the convex baseline's explicit
`θ ≥ 0` constraint. Model parameters are cast to `float64`
(`.double()`) to match the precision used throughout the rest of the
codebase; `nn.Linear` defaults to `float32`.

**Physics-informed loss (`src/pinn_loss.py`).** The training objective is
the same stacked crossing-equation residual the convex baseline minimizes
— `build_design_matrix(g)` and `build_target_vector(g)` are cached once,
before training starts, since they depend only on the (frozen) spectrum
and never on the network's weights. This caching is what keeps training
fast despite the underlying physics being expensive to evaluate: without
it, every epoch would re-run the same hypergeometric-series computation
already available from the first epoch onward.

**Smoothness penalty via double differentiation.** Unlike the spline's
discrete second-difference penalty on basis coefficients, the PINN's
smoothness term is the network output's literal second derivative with
respect to g, computed directly via automatic differentiation:
`torch.autograd.grad(..., create_graph=True)`, called twice in sequence.
The `create_graph=True` flag is essential — it keeps the *gradient
computation itself* part of the differentiable graph, which is what makes
both the second derivative and the eventual backward pass through the
network's weights possible.

**Training dynamics.** Across every training run (pure crossing-loss and
each smoothness weight), loss curves showed the same qualitative pattern:
a long, nearly-flat stretch, followed by a sharp, sudden drop over a few
hundred epochs, then continued gradual improvement. Observed initially
under a single fixed seed at roughly epoch 13000-19000, the multi-seed
study in Part 1 confirmed this is not an artifact of that one
initialization: across 20 runs (4 λ values × 5 seeds), the automatically-
detected escape point clustered at `17130`–`20603` regardless of λ — see
"Smoothness-regularized loss" above, and "Seed-Robustness Infrastructure"
below for how that escape point was detected reliably.
Later in training, loss showed intermittent spikes of increasing severity
— a signature of the fixed learning rate (`1e-3` throughout, via Adam)
no longer being well-matched to the sharper local landscape near
convergence. A learning-rate scheduler was identified as the natural next
refinement but not implemented, since the resulting comparison (Part 1)
did not depend on resolving this instability.

## Reproducibility

`torch.manual_seed(42)`, set before model creation, is sufficient for
exact reproducibility of every result in this project. The only source of
randomness anywhere in the training pipeline is PyTorch's own weight
initialization; the physics computations (`build_design_matrix`,
`build_target_vector`) are fully deterministic given a fixed g, and no
`numpy`-based randomness is used anywhere in the training code.

The original single-seed λ sweep's trained weights are saved to
`models/trained_model_lam_{λ}.pt`, allowing exact reproduction of that
result without retraining. The multi-seed table's 20 runs were not saved
as checkpoints by design (`save=False` throughout) — reproducing them
means re-running the sweep, which is exact and deterministic given the
same seeds, but does take the original training time. The resulting
summary statistics (violation score, final loss, plateau epoch and
magnitude, for every (λ, seed) pair) are saved to
`models/summaries_seed_sweep_parallel.csv`.

## Data provenance

`data/spectrum_strong_coupling.csv` and `data/ope_bounds_strong_coupling.csv`
are extracted from `Conformal_Data.nb`, an ancillary file attached to
Cavaglià, Gromov, Julius, Preti, arXiv:2203.09556. This notebook is the
authors' own supplementary material and is not redistributed in this
repository.

To regenerate the CSVs from scratch:

1. Download `Conformal_Data.nb` from the "Ancillary files" link on
   https://arxiv.org/abs/2203.09556
2. Place it at `data/Conformal_Data.nb`
3. Run:

   ```bash
   python data/extract_conformal_data.py
   python data/extract_OPE_bounds.py
   ```

The two extraction scripts are otherwise self-contained: `resolve_fractions`
and `extract_numbers` (defined in `extract_conformal_data.py`) are reused
by `extract_OPE_bounds.py`.
