# Wilson-line-CFT

A physics-informed neural network (PINN) approach to the 1D defect CFT
bootstrap for the 1/2-BPS Wilson line in 4d N=4 super Yang-Mills at
strong coupling, benchmarked against the numerical conformal bootstrap
and the reinforcement-learning approach of BootSTOP/MultiSTOP.

**Status:** Completed: special functions, coupling functions,
conformal blocks, integral constraints, strong-coupling reference data, convex baselines.
In progress: the construction of the PINN itself.

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
independently at each g inherits this ill-conditioning, which is exactly
what the convex baseline investigation below tests directly.

## Convex Baseline with Smoothness Regularization

This investigation tests a specific question before
any neural network is introduced: *does smoothness-in-g alone, with no
learned representation, resolve the near-degeneracy the identifiability
analysis predicted would be unresolvable pointwise?*

### Background

The SVD/identifiability analysis (`src/identifiability.py`, see the
corresponding notebook figure) predicted, from the design matrix's
near-null-space structure alone, that states 2 and 3 (a pair degenerate at
g→0) and states 4-9 (a sextet, also degenerate at g→0) would be poorly
resolved by any method that fits the crossing equation independently at each
g. Ordinary non-negative least squares at a single g cannot distinguish
them; this investigation asks whether *sharing information across g*, via a
smoothness prior on each state's `C_n²(g)` curve, is enough to break that
degeneracy on its own.

### Method

Each state's OPE-squared coefficient is represented as `C_n²(g) = Σ_k θ_{n,k}
φ_k(g)`, a shared B-spline basis (`src/spline_basis.py`) with coefficients
`θ` common across the entire crossing-equation system (`src/stack_crossing.py`).
The system is solved as a single convex QP (`src/convex_baseline.py`):
minimize the stacked crossing-equation residual plus a second-derivative
smoothness penalty on `θ`, subject to `θ ≥ 0` (the exact non-negativity the
B-spline basis guarantees translates from coefficients to the fitted curve
itself).

The regularization weight `λ` was chosen two independent ways:

- **5-fold cross-validation** (`src/cross_validation.py`), holding out
  entire g-values and measuring the crossing-equation residual on the
  held-out points.
- **A systematic 15-point sweep against the known bootstrap bounds**
  (`src/bounds_sweep.py`), using `data/ope_bounds_strong_coupling.csv` — the
  Cavaglià–Gromov–Julius–Preti bounds on C₁², C₂², C₃² extracted early in
  the project but not used until this investigation.

### Result

![Fitted C_n² vs known bounds at the best λ](notebooks/assets/convex_baseline_vs_bounds.png)

State 1's fitted curve (red) tracks its known bound (blue band) almost
exactly across the entire range. States 2 and 3 show a striking
mirror-image failure: state 2's fit sits visibly *above* its band once
g exceeds roughly 1, while state 3's sits visibly *below* — consistent
with the model splitting their combined weight incorrectly rather than
simply guessing badly. Notably, the fit isn't uniformly wrong: for g
below about 0.5–0.7, where the bootstrap bounds themselves are still
wide and only weakly constraining, the fitted curves track reasonably
well. The divergence begins specifically once the bounds tighten enough
to actually test the fit — the failure appears exactly where the ground
truth becomes precise enough to reveal it. (The rapid oscillation in the
bounds very close to g=0 for states 2 and 3 is a genuine feature of the
underlying bootstrap data in that regime, not a plotting artifact.)

**Both selection methods independently agree on `λ ≈ 0.01`.**


**Both selection methods independently agree on `λ ≈ 0.01`.**

At this jointly-optimal `λ`, state 1 (non-degenerate) fits the known value
closely — within 1-9% across the tested g-range, and within 1% at several
points. States 2 and 3 do not: at the best available `λ`, the total
violation of known bounds across the tested points is `0.566`, not zero.

A manual sweep across `λ ∈ {1, 30, 100, 1000}` — three orders of magnitude —
shows this is not a matter of having missed the right value:

- **State 2 never converges toward its true value at any `λ` tested**,
  remaining 4-6× too high throughout, and drifting *further* from truth at
  the largest `λ`.
- **State 3 responds to `λ`** — improving through `λ≈100` — but overshoots
  badly by `λ=1000`, and the `λ` that best resolves it is not consistent
  across the g-range.
- **State 1's excellent fit degrades once `λ` is pushed hard enough** to
  meaningfully move state 3, revealing a real tradeoff rather than a free
  improvement.

### Conclusion

Smoothness-in-g alone, in this specific implementation, does not resolve
the C₂²/C₃² degeneracy. This is treated as a genuine result, not a failed
attempt: it directly motivates the PINN as bringing something beyond simple
cross-g smoothness — the ablation this investigation was designed to
perform.

### Honest limitations

- The bounds-sweep selection is informed directly by the same ground truth
  used to evaluate it — a legitimate diagnostic, but not a blind test on its
  own. The fact that independent cross-validation lands on the same `λ`
  substantially mitigates this concern, but it is worth stating plainly
  rather than leaving implicit.
- The B-spline settings (16 log-spaced interior knots, degree 3) were a
  reasonable first choice, never independently tuned or validated against
  alternatives. A different knot count or placement might behave
  differently; this was not explored, for scope reasons.
- Bounds-checking used a strided subset of the available g-grid, not
  exhaustive coverage.

### Next steps

- **PINN: in progress.** The main continuation of this project.
- **Integral constraints (`src/constraints.py`'s `int1`/`int2`) as
  additional linear rows in the same convex system**: not yet attempted.
  Both integrals are linear in `C_n²`, so this extension stays fully convex
  — no network required. Left as a possible future exercise, not a planned
  next step.


## Repository structure

    src/          core physics modules: special functions, coupling
                  functions, conformal blocks, the crossing equation,
                  integral constraints
    data/         data-extraction scripts and the resulting reference CSVs
    tests/        unit tests, one file per src/ module
    notebooks/    supporting figures and derivations

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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

## References

- Ferrero, Meneghelli, arXiv:2103.10440, arXiv:2312.12550, arXiv:2312.12551
- Cavaglià, Gromov, Julius, Preti, arXiv:2203.09556
- Trenta, Bacciu, Cossu, Ferrero, arXiv:2404.1490

