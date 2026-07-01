# JAX prototype — differentiable 2D LCODE (Branch 2)

De-risking prototype for a differentiable / GPU-native 2D solver (see the dev-branch plans;
GitHub issue #16 for GPU). Goal: reimplement the 2D quasistatic solver in **JAX** so it is
differentiable (`jax.grad`), jit-fusible, and GPU-ready — and check it stays accurate vs the
numba reference before committing to a full port.

Not wired into the package; standalone scripts run against the existing numba code + the
unit-test reference data.

```
conda activate lcode-env         # jax installed here (CPU)
PYTHONPATH=$(git rev-parse --show-toplevel) python validate_field_solver.py
```

## Milestones
- [x] **M1 — Field solver** (`field_solver_jax.py`): the hardest, sequential-along-r part
      (Thomas tridiagonal solve + cumulative integrals). Ported with `lax.scan` / `cumsum`.
- [x] **M2 — Deposition** (`deposition_jax.py`): vectorized masked scatter with the quadratic
      (C1) shape, central-cell antisymmetry, boundary clamp. Matches numba to 2.9e-11 (tol 5e-11);
      differentiable w.r.t. particle position (grad vs FD ~2e-9).
- [x] **M3 — Particle push** (`move_jax.py`): vectorized Lorentz push (branches → `jnp.where`)
      + substepping as a **bounded `lax.scan`** (reverse-mode grad does not work through
      `lax.while_loop`). Matches numba mover to 4.5e-13 (inside 1e-12 tol); differentiable
      (grad vs FD ~2e-8).
- [x] **M4 — ξ-layer + full ξ-march** (`march_jax.py`): predictor/corrector `step_dxi` + the
      whole march via `lax.scan`, driven by a precomputed per-layer beam density. On-axis Ez(ξ)
      matches numba to mean 5e-9 (max 4e-5 of peak) over 301 layers; gradient through the whole
      march matches finite differences (~1e-4). (Outer step_dxi substepping omitted — weak-driver
      regime; needs a bounded form for the general case, like the mover in M3.)
### Time dynamics (self-consistent beam evolution)
- [x] **MB1 — Beam deposition** (`beam_jax.py`): bilinear (xi, r) scatter with the per-layer
      `rho_layout` carry. Matches numba `deposit_beam_layer` to 2.8e-14 (bit-exact; matches the
      *recomputed* numba even where the stored file differs by issue #2). Differentiable (grad
      w.r.t. beam charge ~1e-13).
- [ ] MB2 — Beam push (evolve beam particles in the wake fields), validate vs numba `push_beam_layer`.
- [ ] MB3 — Couple beam deposit + push into the ξ-march (one full time step).
- [ ] MB4 — Outer time loop over N steps; validate beam energy evolution vs numba; differentiable.

### Single-time-step milestones
- [x] **M5 — Differentiable optimization demo** (`optimize_demo.py`): end-to-end gradient-based
      **inverse design** — a differentiable rigid Gaussian driver feeds the JAX march; `jax.grad`
      (with `jax.checkpoint` for memory) drives Adam to recover the driver radius that produces a
      target wake (σ=1.24 vs true 1.2, ~3% from σ_start=2.5). This is the Branch-2 payoff. Still
      open (real-implementation phase, not de-risking): **self-consistent beam push** (beam
      evolving in the wake), bounded **outer** step_dxi substepping, GPU run, float32/perf tuning.

## M1 results (2026-07-01, CPU, float64)
- **Accuracy vs numba `compute_fields`** on all 6 reference states: worst **3.9e-12** relative
  (float64 round-off; the field-solver unit test tolerance is 1e-9). Sub-kernels: tridiagonal
  ~1e-12–1e-13, cumtrapz ~1e-17.
- **Differentiable:** `jax.grad` end-to-end (through the tridiagonal solve) matches finite
  differences to ~3e-8 (FD accuracy limit); gradient nonzero across all components.
- **jit:** ~3.4 µs/call for `compute_fields` E_z on n=201.

**Takeaway:** JAX reproduces the sequential-along-r field solver to round-off *and*
differentiates it correctly. The trickiest numerics for Branch 2 are de-risked.

## M2 results (2026-07-01, CPU, float64)
- **Accuracy vs numba `compute_rhoj`** on all 6 states: worst **2.9e-11** relative (inside the
  5e-11 deposition unit-test tolerance; residual is scatter-order FP, not bit-identical).
- **Differentiable:** grad w.r.t. particle position finite & nonzero for all 2000 particles,
  matches finite differences to ~2e-9 for particles interior to a cell. The quadratic C1
  shape keeps deposition smooth — the main differentiability worry is manageable.

## M3 results (2026-07-01, CPU, float64)
- **Accuracy vs numba mover** on all 6 states: worst **4.5e-13** relative — inside the strict
  1e-12 pusher tolerance (the push is per-particle elementwise, so it reproduces to round-off).
- **Differentiable:** grad through the bounded-scan push finite & nonzero, matches finite
  differences to ~2e-8.
- **Key control-flow finding:** reverse-mode `grad` fails through `lax.while_loop`
  (data-dependent trip count) → the adaptive substepping is reformulated as a bounded
  `lax.scan` with masking (N=64 attempts sufficed for the reference states). Deeply-trapped
  particles could need more attempts / a smooth relaxation; revisit in M4/M5.

**Status:** all three physics-numerics kernels (field solve, deposition, push) are ported,
match numba within the respective unit-test tolerances, and are differentiable. The core risk
of Branch 2 is retired; M4/M5 are assembly (predictor-corrector + ξ-scan + beam) rather than
new numerical risk.

## M4 results (2026-07-01, CPU, float64)
- **Full ξ-march** (301 layers) driven by numba's per-layer beam density: on-axis Ez(ξ)
  matches numba to **mean 5e-9, max 4e-5 of peak** (round-off accumulation over the coupled
  march). Gradient d(sum Ez^2)/d(beam amplitude) through the whole `lax.scan` matches finite
  differences to ~1e-4.
- **Harness lesson:** `Simulation.step()` runs a 2-xi-step warmup (JIT compile) that also
  calls step_dxi/deposit with a fake beam; capturing per-layer data must disable warmup, or
  the JAX march is driven by 2 spurious layers (this caused a spurious 3.5% mismatch until
  found). `validate_march.py` disables warmup during capture.

## M5 results (2026-07-01, CPU, float64)
- **End-to-end differentiable inverse design** (`optimize_demo.py`): objective = match a target
  trailing-wake energy by tuning the driver radius σ. `jax.grad` through the full 100-layer
  march (with `jax.checkpoint`) + Adam recovers **σ ≈ 1.24 vs true 1.2** (~3%) from σ_start=2.5,
  monotone loss decrease — verified against a σ-scan. This is the capability that justifies
  Branch 2 (impossible with the numba code).
- **Performance note:** reverse-mode grad through the nested scans is heavy; needs
  `jax.checkpoint` (remat) on the ξ-step and a small bounded `n_attempts` for the mover
  (~0.6 s/grad at 100 layers after a ~4 s compile). GPU + float32 would speed this up a lot.
- **Deferred to a real implementation phase** (not de-risking): self-consistent beam push
  (beam evolving in the wake), bounded outer step_dxi substepping, GPU execution, perf tuning.

## Notes
- `jax_enable_x64` is required to match numba float64 (JAX defaults to float32).
- Not bit-identical to numba (different op order) — Branch 2 validates against a *relaxed*
  tolerance, by design.
