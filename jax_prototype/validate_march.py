"""
Validate the JAX xi-march (M4) against numba, driven by the SAME per-layer beam density.

Runs a weak-driver numba 2D simulation, records (a) the per-layer rho_beam it deposits and
(b) the on-axis Ez it produces after each xi-step. Then runs the JAX march from the identical
initial plasma state with the recorded rho_beam, and compares the on-axis Ez(xi).
Also checks a gradient flows through the whole march.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

import lcode
import lcode.plasma as nplasma
import lcode.plasma.rhoj as nrhoj
import lcode.beam.beam_calculate as nbeamcalc
import lcode.plasma.solver as nsolver
from lcode.plasma.data import Arrays
import deposition_jax as jdep
import march_jax as jmarch

jax.config.update("jax_enable_x64", True)

R_STEP = 0.05
WINDOW_WIDTH = 5
XI_STEP = 0.05
WINDOW_LENGTH = 15
PPC = 10
N_CELLS = int(WINDOW_WIDTH / R_STEP) + 1

CONFIG = {
    "geometry": "2d", "processing-unit-type": "cpu",
    "window-width": WINDOW_WIDTH, "transverse-step": R_STEP,
    "window-length": WINDOW_LENGTH, "xi-step": XI_STEP,
    "time-limit": 1, "time-step": 1,
    "plasma-particles-per-cell": PPC, "ion-model": "background",
    "noise-reductor-enabled": False,
}
BEAM = {"current": -0.005, "particles_in_layer": 400,   # weak -> linear, no outer substepping
        "default": {"angspread": 2e-4, "energy": 1000},
        "driver": {"xishape": "l", "length": 10, "radius": 1.0}}


def run_numba_capture():
    """Run numba once, capturing per-layer rho_beam and post-step on-axis Ez.

    NOTE: Simulation.step() runs a 2-xi-step warmup (to JIT-compile the kernels) that also
    drives step_dxi / deposit_beam_layer with a fake beam. We disable warmup here so the
    captured sequence contains only the real xi-layers (otherwise the JAX march would be
    driven by 2 spurious warmup layers and diverge).
    """
    import lcode.push_solvers.push_solver as nps
    orig_warmup = nps.PusherAndSolver.warmup
    nps.PusherAndSolver.warmup = lambda self, *a, **k: None

    rho_beams, ez_axis = [], []
    orig_dep = nbeamcalc.BeamCalculator2D.deposit_beam_layer
    orig_step = nsolver.CylindricalPlasmaSolver.step_dxi

    def dep(self, layer, xi_i):
        out = orig_dep(self, layer, xi_i)
        rho_beams.append(np.array(out))
        return out

    def step(self, *a, **k):
        p, f, c = orig_step(self, *a, **k)
        ez_axis.append(float(f.E_z[0]))
        return p, f, c

    nbeamcalc.BeamCalculator2D.deposit_beam_layer = dep
    nsolver.CylindricalPlasmaSolver.step_dxi = step
    try:
        sim = lcode.Simulation(config=dict(CONFIG), beam_parameters=BEAM,
                               diagnostics=[], runas_filename="")
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            sim.step()
    finally:
        nbeamcalc.BeamCalculator2D.deposit_beam_layer = orig_dep
        nsolver.CylindricalPlasmaSolver.step_dxi = orig_step
        nps.PusherAndSolver.warmup = orig_warmup
    return np.array(rho_beams), np.array(ez_axis)


def jax_initial_state():
    """Initial plasma state from numba init_plasma_2d, as JAX pytrees."""
    class Cfg:
        _m = {"geometry": "2d", "transverse-step": R_STEP, "window-width": WINDOW_WIDTH,
              "plasma-particles-per-cell": PPC, "ion-model": "background"}
        def get(self, k): return self._m.get(k, None)
        def getfloat(self, k): return float(self._m[k])
        def getint(self, k): return int(self._m[k])
    state = nplasma.init_plasma_2d(lcode.config.config.Config(Cfg._m))
    el = state[1]["electrons"]
    particles = {"r": jnp.asarray(el.r), "p_r": jnp.asarray(el.p_r),
                 "p_f": jnp.asarray(el.p_f), "p_z": jnp.asarray(el.p_z),
                 "q": jnp.asarray(el.q), "m": jnp.asarray(el.m)}
    fields = {k: jnp.zeros(N_CELLS) for k in jmarch.FIELD_KEYS}
    vol = jdep.cell_volume(R_STEP, PPC, N_CELLS)
    currents = jdep.compute_rhoj(particles, N_CELLS, R_STEP, vol, ni=1.0)
    return particles, fields, currents, vol


def main():
    print("Running numba (weak driver) and capturing rho_beam + Ez per layer...")
    rho_beams, ez_nb = run_numba_capture()
    print(f"  captured {rho_beams.shape[0]} layers, n_cells={rho_beams.shape[1]}")

    particles, fields, currents, vol = jax_initial_state()
    p = {"n_cells": N_CELLS, "r_step": R_STEP, "xi_step": XI_STEP, "vol": vol,
         "ni": 1.0, "max_radius": WINDOW_WIDTH, "corrector_steps": 2}

    print("Running JAX march with the same rho_beam...")
    ez_jax, _ = jmarch.march(particles, fields, currents, jnp.asarray(rho_beams), p)
    ez_jax = np.asarray(ez_jax)

    n = min(len(ez_nb), len(ez_jax))
    ez_nb, ez_jax = ez_nb[:n], ez_jax[:n]
    absdiff = np.abs(ez_jax - ez_nb)
    scale = np.max(np.abs(ez_nb))
    print(f"\nOn-axis Ez(xi) match over {n} layers:")
    print(f"  max |Ez_jax - Ez_nb|        = {absdiff.max():.3e}")
    print(f"  relative to peak |Ez|={scale:.3e}: {absdiff.max()/scale:.3e}")
    print(f"  mean abs diff               = {absdiff.mean():.3e}")

    print("\nGradient through the whole march (d sum(Ez^2) / d beam amplitude):")
    rb = jnp.asarray(rho_beams)

    def obj(scale_factor):
        ez, _ = jmarch.march(particles, fields, currents, rb * scale_factor, p)
        return jnp.sum(ez ** 2)

    g = float(jax.grad(obj)(1.0))
    eps = 1e-4
    fd = float((obj(1.0 + eps) - obj(1.0 - eps)) / (2 * eps))
    print(f"  grad={g:.6e}  finite-diff={fd:.6e}  rel={abs(g-fd)/(abs(fd)+1e-30):.2e}")


if __name__ == "__main__":
    main()
