"""
Backends 2 & 3 — JAX LCODE (same code on Mac CPU and GPU), self-consistent evolving beam.

Starts from the numba-captured initial beam (beam_init.npz) so it is directly comparable, then:
  - OUTPUT check: step-1 on-axis wake Ez(xi) vs numba (ez_numba.npy) — identical beam input, so
    this tests the plasma solver directly;
  - evolves n_t-1 steps with the validated `one_time_step` dynamics and compares beam stats;
  - times the evolution (device is whatever JAX sees: CpuDevice on the Mac, CudaDevice on GPU).

Run after run_numba.py (which writes beam_init.npz, ez_numba.npy, stats_numba.npy).
"""
import time, platform
import numpy as np
import jax
import jax.numpy as jnp

from lcode.jax import (one_time_step, march_with_fields, deposit_beam_full,
                       init_substepping, compute_rhoj, cell_volume, FIELD_KEYS)

WW, R, XI, WL, PPC = 10.0, 0.1, 0.1, 100.0, 10
NC = int(WW / R) + 1
NXI = int(WL / XI)
N_T = 10
TIME_STEP, E_SS = 25.0, 2.0

print("=" * 68)
print("host:", platform.node(), "| jax", jax.__version__, "| devices:", jax.devices())
print("=" * 68, flush=True)


def make_plasma():
    n = int(WW / R) * PPC
    rp = (np.arange(n) + 0.5) * (WW / n)
    mp = 2 * np.pi * (rp[1] - rp[0]) * rp
    part = {"r": jnp.asarray(rp), "p_r": jnp.zeros(n), "p_f": jnp.zeros(n),
            "p_z": jnp.zeros(n), "q": jnp.asarray(-mp), "m": jnp.asarray(mp)}
    vol = cell_volume(R, PPC, NC)
    plasma = {"particles": part, "fields": {k: jnp.zeros(NC) for k in FIELD_KEYS},
              "currents": compute_rhoj(part, NC, R, vol, 1.0)}
    p = {"n_xi": NXI, "n_cells": NC, "r_step": R, "xi_step": XI, "vol": vol, "ni": 1.0,
         "max_radius": WW, "corrector_steps": 2, "n_attempts": 4, "checkpoint": False,
         "magnetic_field": 0.0}
    return plasma, p


d = np.load("beam_init.npz")
beam = {k: jnp.asarray(d[k]) for k in d.files}
plasma, p = make_plasma()
print(f"loaded initial beam: {beam['r'].size} particles", flush=True)

# ---- OUTPUT check: step-1 wake Ez(xi) from the identical initial beam
rho_beam = deposit_beam_full(beam["r"], beam["xi"], beam["q_norm"], NXI, NC, R, XI)
hist, _ = march_with_fields(plasma["particles"], plasma["fields"], plasma["currents"], rho_beam, p)
ez_jax = np.asarray(hist["E_z"][:, 0])
ez_numba = np.load("ez_numba.npy")
denom = np.abs(ez_numba).max()
print(f"\n[OUTPUT] step-1 wake Ez(xi) vs numba:  max|dEz|={np.max(np.abs(ez_jax-ez_numba)):.3e}"
      f"   rel={np.max(np.abs(ez_jax-ez_numba))/denom:.3e}", flush=True)

# ---- self-consistent evolution (validated dynamics), matched to numba
dt, steps = init_substepping(beam["p_z"], beam["q_m"], jnp.zeros_like(beam["p_z"]),
                             jnp.zeros_like(beam["p_z"]), TIME_STEP, E_SS)
msub = int(np.asarray(steps).max())
beam["remaining_steps"] = steps


def evolve(beam, n):
    def body(beam, _):
        b, ez, lost = one_time_step(beam, plasma, p, dt, msub)
        return b, None
    beam, _ = jax.lax.scan(body, beam, None, length=n)
    return beam


ev = jax.jit(lambda b: evolve(b, N_T - 1))
final = ev(beam); jax.block_until_ready(final["p_z"])           # warm + evolve
stats_jax = np.array([float(jnp.mean(final["p_z"])), float(jnp.sqrt(jnp.mean(final["r"] ** 2)))])
stats_numba = np.load("stats_numba.npy")
print(f"[OUTPUT] after evolution:  JAX  <p_z>={stats_jax[0]:.3f} rms_r={stats_jax[1]:.4f}")
print(f"                           numba <p_z>={stats_numba[0]:.3f} rms_r={stats_numba[1]:.4f}",
      flush=True)

reps = 3
t = time.perf_counter()
for _ in range(reps):
    jax.block_until_ready(ev(beam)["p_z"])
elapsed = (time.perf_counter() - t) / reps
dev = "GPU" if "cuda" in str(jax.devices()[0]).lower() else "CPU"
print(f"\nJAX ({dev}): {N_T-1} evolving steps in {elapsed:.3f}s -> {elapsed/(N_T-1)*1e3:.1f} ms/step",
      flush=True)
