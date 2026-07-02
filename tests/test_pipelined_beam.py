"""
The single-trajectory time-step pipeline (per-layer beam hand-off) must reproduce the
sequential fused evolution particle-by-particle — i.e. overlapping the time steps as a
diagonal wavefront changes nothing but the execution order.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from lcode import jax as lj
from lcode.jax.pipelined_beam import sequential_fused_evolve, pipelined_evolve


def _plasma(WW=5.0, R=0.1, XI=0.1, WL=1.0, PPC=6):
    NC = int(WW / R) + 1
    NXI = int(WL / XI)
    n = int(WW / R) * PPC
    rp = (np.arange(n) + 0.5) * (WW / n)
    mp = 2 * np.pi * (rp[1] - rp[0]) * rp
    part = {"r": jnp.asarray(rp), "p_r": jnp.zeros(n), "p_f": jnp.zeros(n),
            "p_z": jnp.zeros(n), "q": jnp.asarray(-mp), "m": jnp.asarray(mp)}
    vol = lj.cell_volume(R, PPC, NC)
    plasma = {"particles": part, "fields": {k: jnp.zeros(NC) for k in lj.FIELD_KEYS},
              "currents": lj.compute_rhoj(part, NC, R, vol, 1.0)}
    p = {"n_xi": NXI, "n_cells": NC, "r_step": R, "xi_step": XI, "vol": vol, "ni": 1.0,
         "max_radius": WW, "corrector_steps": 2, "n_attempts": 4, "checkpoint": False,
         "magnetic_field": 0.0}
    return plasma, p, NXI


def _beam(NXI, XI, C=8, seed=0):
    rng = np.random.default_rng(seed)
    r = np.abs(rng.normal(0.0, 1.0, (NXI, C))) + 0.05
    j = np.arange(NXI)[:, None]
    xi = -(j + rng.uniform(0.2, 0.8, (NXI, C))) * XI
    beam = {"r": jnp.asarray(r), "xi": jnp.asarray(xi), "p_z": jnp.full((NXI, C), 1000.0),
            "p_r": jnp.zeros((NXI, C)), "M": jnp.zeros((NXI, C)),
            "q_m": -jnp.ones((NXI, C)), "q_norm": jnp.asarray(np.full((NXI, C), -0.01 / C))}
    dt, steps = lj.init_substepping(beam["p_z"], beam["q_m"], jnp.zeros((NXI, C)),
                                    jnp.zeros((NXI, C)), 25.0, 2.0)
    beam["remaining_steps"] = steps
    beam["dt"] = dt
    return beam, int(np.asarray(steps).max())


@pytest.mark.parametrize("n_steps", [1, 3, 6])
def test_pipeline_matches_sequential(n_steps):
    plasma, p, NXI = _plasma()
    beam, msub = _beam(NXI, p["xi_step"])
    seq = sequential_fused_evolve(plasma, beam, msub, p, n_steps)
    pip = pipelined_evolve(plasma, beam, msub, p, n_steps)
    for k in ("r", "xi", "p_z", "p_r", "M"):
        err = float(jnp.max(jnp.abs(seq[k] - pip[k])))
        assert err < 1e-9, f"{k}: pipeline vs sequential differ by {err:.2e} at n_steps={n_steps}"
