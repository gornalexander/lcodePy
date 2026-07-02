"""
The batched-wavefront march must be numerically identical to running the marches
sequentially. The sequential JAX march is itself validated against numba in
test_jax_consistency.py, so wavefront == sequential == numba (transitively).
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from lcode import jax as lj


def _plasma(WW=5.0, R=0.05, XI=0.05, WL=6.0, PPC=10):
    """LCODE 2D plasma layout, reproduced in numpy (no numba)."""
    NC = int(WW / R) + 1
    NXI = int(WL / XI)
    n = int(WW / R) * PPC
    rp = (np.arange(n) + 0.5) * (WW / n)
    mp = 2 * np.pi * (rp[1] - rp[0]) * rp
    part = {"r": jnp.asarray(rp), "p_r": jnp.zeros(n), "p_f": jnp.zeros(n),
            "p_z": jnp.zeros(n), "q": jnp.asarray(-mp), "m": jnp.asarray(mp)}
    vol = lj.cell_volume(R, PPC, NC)
    plasma = {"particles": part,
              "fields": {k: jnp.zeros(NC) for k in lj.FIELD_KEYS},
              "currents": lj.compute_rhoj(part, NC, R, vol, 1.0)}
    p = {"n_xi": NXI, "n_cells": NC, "r_step": R, "xi_step": XI, "vol": vol, "ni": 1.0,
         "max_radius": WW, "corrector_steps": 2, "n_attempts": 4, "checkpoint": False,
         "magnetic_field": 0.0}
    return plasma, p, NC, NXI


def test_wavefront_matches_sequential():
    plasma, p, NC, NXI = _plasma()
    depth = 5
    rng = np.random.default_rng(0)
    # small, smooth beam-density profiles so the solver stays well-behaved
    r = np.arange(NC)
    rho = np.stack([-0.02 * np.exp(-((r - 3) ** 2) / 8.0) * (1 + 0.1 * rng.standard_normal())
                    * np.ones((NXI, 1)) for _ in range(depth)])
    rho_beam_batch = jnp.asarray(rho)

    seq = lj.sequential_marches(rho_beam_batch, plasma, p)
    wav = lj.wavefront_marches(rho_beam_batch, plasma, p)

    for k in seq:
        err = float(jnp.max(jnp.abs(seq[k] - wav[k])))
        assert err < 1e-10, f"{k}: wavefront vs sequential differ by {err:.2e}"


def test_wavefront_depth_one_matches_single_march():
    """depth=1 wavefront must equal a plain march_with_fields."""
    plasma, p, NC, NXI = _plasma()
    rho = jnp.asarray(-0.02 * np.ones((1, NXI, NC)))
    wav = lj.wavefront_marches(rho, plasma, p)
    fh, _ = lj.march_with_fields(plasma["particles"], plasma["fields"],
                                 plasma["currents"], rho[0], p)
    for k in fh:
        err = float(jnp.max(jnp.abs(fh[k] - wav[k][0])))
        assert err < 1e-10, f"{k}: depth-1 wavefront vs march differ by {err:.2e}"
