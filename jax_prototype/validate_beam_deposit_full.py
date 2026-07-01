"""
Validate the whole-beam deposition (MB3) against numba.

Runs a numba simulation (warmup disabled), captures the per-layer rho_beam it deposits and
the beam particles, then deposits the whole beam at once with the JAX kernel and compares.
"""
import io
import contextlib
import numpy as np
import jax
import jax.numpy as jnp

import lcode
import lcode.beam.beam_calculate as nbeamcalc
import lcode.push_solvers.push_solver as nps
import beam_jax as jbeam

jax.config.update("jax_enable_x64", True)
R_STEP, WW, XI_STEP, WL, PPC = 0.05, 5, 0.05, 5, 10
N_CELLS = int(WW / R_STEP) + 1
CONFIG = {"geometry": "2d", "processing-unit-type": "cpu", "window-width": WW,
          "transverse-step": R_STEP, "window-length": WL, "xi-step": XI_STEP,
          "time-limit": 1, "time-step": 1, "plasma-particles-per-cell": PPC,
          "ion-model": "background", "noise-reductor-enabled": False}
BEAM = {"current": -0.02, "particles_in_layer": 300,
        "default": {"angspread": 2e-4, "energy": 1000},
        "driver": {"xishape": "l", "length": 3, "radius": 1.0}}


def main():
    nps.PusherAndSolver.warmup = lambda self, *a, **k: None
    rho_beams, PR, PXI, PQ = [], [], [], []
    orig = nbeamcalc.BeamCalculator2D.deposit_beam_layer

    def dep(self, layer, xi_i):
        if layer.size:
            PR.append(np.array(layer.r)); PXI.append(np.array(layer.xi)); PQ.append(np.array(layer.q_norm))
        o = orig(self, layer, xi_i); rho_beams.append(np.array(o)); return o

    nbeamcalc.BeamCalculator2D.deposit_beam_layer = dep
    try:
        sim = lcode.Simulation(config=dict(CONFIG), beam_parameters=BEAM, diagnostics=[], runas_filename="")
        with contextlib.redirect_stdout(io.StringIO()):
            sim.step()
    finally:
        nbeamcalc.BeamCalculator2D.deposit_beam_layer = orig

    rho_nb = np.array(rho_beams)
    n_layers = rho_nb.shape[0]
    r = np.concatenate(PR); xi = np.concatenate(PXI); q = np.concatenate(PQ)
    rho_jx = np.asarray(jbeam.deposit_beam_full(jnp.asarray(r), jnp.asarray(xi), jnp.asarray(q),
                                                n_layers, N_CELLS, R_STEP, XI_STEP))
    d = np.abs(rho_jx - rho_nb).max()
    print(f"whole-beam deposition: {n_layers} layers, {r.size} particles")
    print(f"  max|jax - numba| = {d:.2e}   (peak rho_beam = {np.abs(rho_nb).max():.3f})")


if __name__ == "__main__":
    main()
