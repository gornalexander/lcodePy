"""
Backend 3 (parallel) — JAX time-step PIPELINE, evolving beam on the 1000 x 101 grid.

Evolves ONE beam for n_t time steps with the wavefront pipeline (pipelined_beam.py, issue #19):
the n_t consecutive time steps are overlapped as a diagonal wavefront (per-layer beam hand-off),
so every scan tick advances all in-flight steps at once. Compares against the sequential-fused
evolution (must be identical) and the numba stats (stats_numba.npy), and times both.

Runs on Mac CPU and on GPU (same code). Uses the numba-captured initial beam (beam_init.npz).
"""
import time, platform
import numpy as np
import jax
import jax.numpy as jnp

from lcode.jax import init_substepping, compute_rhoj, cell_volume, FIELD_KEYS
from lcode.jax.pipelined_beam import sequential_fused_evolve, pipelined_evolve

WW, R, XI, WL, PPC = 10.0, 0.1, 0.1, 100.0, 10
NC = int(WW / R) + 1
NXI = int(WL / XI)
N_T = 10

print("=" * 68)
print("host:", platform.node(), "| jax", jax.__version__, "| devices:", jax.devices())
DEV = "GPU" if "cuda" in str(jax.devices()[0]).lower() else "CPU"
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


def bucket_beam(d):
    """Bucket the flat numba beam into layer-locked buckets [NXI, C]."""
    xi = np.asarray(d["xi"]); L = np.floor(-xi / XI).astype(int)
    C = int(np.bincount(L, minlength=1).max())
    b = {k: np.zeros((NXI, C)) for k in ("r", "xi", "p_z", "p_r", "M", "q_m", "q_norm")}
    b["r"][:] = 1e-3; b["q_m"][:] = -1.0; b["p_z"][:] = 1.0        # inert padding
    real = np.zeros((NXI, C), bool); slot = np.zeros(NXI, int)
    for i in range(xi.size):
        l = L[i]; s = slot[l]
        for k in b:
            b[k][l, s] = d[k][i]
        real[l, s] = True; slot[l] += 1
    beam = {k: jnp.asarray(v) for k, v in b.items()}
    dt, steps = init_substepping(beam["p_z"], beam["q_m"], jnp.zeros((NXI, C)),
                                 jnp.zeros((NXI, C)), 25.0, 2.0)
    beam["remaining_steps"] = steps; beam["dt"] = dt
    return beam, int(np.asarray(steps).max()), real, C


def timeit(fn, reps, *a):
    o = fn(*a); jax.block_until_ready(o)
    t = time.perf_counter()
    for _ in range(reps):
        o = fn(*a)
    jax.block_until_ready(o)
    return (time.perf_counter() - t) / reps * 1e3


plasma, p = make_plasma()
d = np.load("beam_init.npz")
beam, msub, real, C = bucket_beam(d)
print(f"beam bucketed: {int(real.sum())} particles, {NXI}x{C} buckets, msub={msub}\n", flush=True)

# --- correctness: pipelined == sequential-fused, and vs numba
seq = jax.jit(lambda b: sequential_fused_evolve(plasma, b, msub, p, N_T - 1))
pip = jax.jit(lambda b: pipelined_evolve(plasma, b, msub, p, N_T - 1))
out_s = seq(beam); out_p = pip(beam)
dmax = max(float(jnp.max(jnp.abs(out_s[k] - out_p[k]))) for k in ("r", "p_z", "p_r"))
rms = float(np.sqrt(np.mean(np.asarray(out_p["r"])[real] ** 2)))
pz = float(np.mean(np.asarray(out_p["p_z"])[real]))
sn = np.load("stats_numba.npy")
print(f"[CHECK] pipelined vs sequential-fused: max diff = {dmax:.2e}  (must be ~0)")
print(f"[OUTPUT] after {N_T-1} steps:  pipeline <p_z>={pz:.3f} rms_r={rms:.4f}")
print(f"                               numba    <p_z>={sn[0]:.3f} rms_r={sn[1]:.4f}", flush=True)

# --- performance: sequential-fused vs pipelined
ms_seq = timeit(seq, 2, beam)
ms_pip = timeit(pip, 2, beam)
print(f"\nJAX ({DEV}), evolve 1 beam x {N_T-1} steps on {NXI}x{NC} grid:")
print(f"  sequential : {ms_seq:9.1f} ms  ({ms_seq/(N_T-1):.1f} ms/step)")
print(f"  PIPELINED  : {ms_pip:9.1f} ms  ({ms_pip/(N_T-1):.1f} ms/step)   speedup {ms_seq/ms_pip:.2f}x",
      flush=True)
print("\nDONE.", flush=True)
