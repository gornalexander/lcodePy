"""
Numba reference march driven by a prescribed per-layer beam density.

Mirrors march_jax.march but using the original numba `CylindricalPlasmaSolver.step_dxi`
in a Python loop, so the JAX and numba solvers can be compared apples-to-apples (same
rigid driver, same grid) for accuracy and wall-clock timing, and so a numba
finite-difference optimization can be benchmarked against JAX autodiff.
"""
import numpy as np

import lcode.plasma as nplasma
import lcode.config.config as cfgmod
from lcode.plasma.solver import CylindricalPlasmaSolver


def build_numba_march(config_dict):
    cfg = cfgmod.Config(config_dict)
    solver = CylindricalPlasmaSolver(cfg)
    state = nplasma.init_plasma_2d(cfg)
    fields0, parts0, currents0, const = state[0], state[1], state[2], state[3]
    if not getattr(const, "sorts", None):
        const.sorts = {"electrons": 0}
    n_cells = fields0.E_r.shape[0]
    rbp = np.zeros(n_cells)

    def march(rho_beam_seq):
        # fresh deep copies of the initial state each call
        particles = {"electrons": parts0["electrons"].copy()}
        fields = fields0.copy()
        currents = currents0.copy()
        ez = np.empty(len(rho_beam_seq))
        for k in range(len(rho_beam_seq)):
            particles, fields, currents = solver.step_dxi(
                particles, fields, currents, const, rho_beam_seq[k], rbp)
            ez[k] = fields.E_z[0]
        return ez

    return march, n_cells
