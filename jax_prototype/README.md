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
- [ ] M3 — Particle push (`vmap` over particles; boundary/loss via masks; substepping).
- [ ] M4 — Assemble one ξ-layer, then the full ξ-march via `lax.scan` + `jax.checkpoint`.
- [ ] M5 — Full single time step; validate vs LCODE at a relaxed tolerance; grad demo.

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

## Notes
- `jax_enable_x64` is required to match numba float64 (JAX defaults to float32).
- Not bit-identical to numba (different op order) — Branch 2 validates against a *relaxed*
  tolerance, by design.
