"""
MB3/MB4 demo: self-consistent beam evolution over several time steps, and differentiability.

Uses a real numba-generated beam as the initial condition, then evolves it with the JAX
time loop (deposit -> plasma march -> push, each step). Shows the driver loses energy to the
wake, and that d(final beam energy)/d(initial beam energy) flows through the whole loop.
"""
import io
import contextlib
import numpy as np
import jax
import jax.numpy as jnp

import lcode
import lcode.plasma as nplasma
import lcode.beam.beam_calculate as nbeamcalc
import lcode.push_solvers.push_solver as nps
import deposition_jax as jdep
import beam_jax as jbeam
import march_jax as jmarch
import time_dynamics as td

jax.config.update("jax_enable_x64", True)

R_STEP, WW, XI_STEP, WL, PPC = 0.05, 5, 0.05, 6, 10
N_CELLS = int(WW / R_STEP) + 1
N_XI = int(WL / XI_STEP)
TIME_STEP = 25.0
E_SS = 2.0
CONFIG = {"geometry": "2d", "processing-unit-type": "cpu", "window-width": WW,
          "transverse-step": R_STEP, "window-length": WL, "xi-step": XI_STEP,
          "time-limit": TIME_STEP, "time-step": TIME_STEP, "plasma-particles-per-cell": PPC,
          "ion-model": "background", "noise-reductor-enabled": False}
BEAM = {"current": -0.05, "particles_in_layer": 200,
        "default": {"angspread": 1e-5, "energy": 1000},
        "driver": {"xishape": "l", "length": 3, "radius": 1.0}}


def capture_numba_beam():
    nps.PusherAndSolver.warmup = lambda self, *a, **k: None
    cols = {k: [] for k in ("r", "xi", "p_z", "p_r", "M", "q_m", "q_norm")}
    orig = nbeamcalc.BeamCalculator2D.deposit_beam_layer

    def dep(self, layer, xi_i):
        if layer.size:
            for k in cols:
                cols[k].append(np.array(getattr(layer, k)))
        return orig(self, layer, xi_i)

    nbeamcalc.BeamCalculator2D.deposit_beam_layer = dep
    try:
        sim = lcode.Simulation(config=dict(CONFIG), beam_parameters=BEAM, diagnostics=[], runas_filename="")
        with contextlib.redirect_stdout(io.StringIO()):
            sim.step()
    finally:
        nbeamcalc.BeamCalculator2D.deposit_beam_layer = orig
    return {k: np.concatenate(v) for k, v in cols.items()}


def fresh_plasma():
    cfgd = {"geometry": "2d", "transverse-step": R_STEP, "window-width": WW,
            "plasma-particles-per-cell": PPC, "ion-model": "background"}
    state = nplasma.init_plasma_2d(lcode.config.config.Config(cfgd))
    el = state[1]["electrons"]
    particles = {k: jnp.asarray(getattr(el, k)) for k in ("r", "p_r", "p_f", "p_z", "q", "m")}
    fields = {k: jnp.zeros(N_CELLS) for k in jmarch.FIELD_KEYS}
    vol = jdep.cell_volume(R_STEP, PPC, N_CELLS)
    currents = jdep.compute_rhoj(particles, N_CELLS, R_STEP, vol, ni=1.0)
    return particles, fields, currents, vol


def main():
    print("Capturing a numba-generated beam as the initial condition...")
    b = capture_numba_beam()
    print(f"  {b['r'].size} beam particles;  mean p_z = {b['p_z'].mean():.2f}")

    particles, fields, currents, vol = fresh_plasma()
    PLASMA = {"particles": particles, "fields": fields, "currents": currents}
    P = {"n_xi": N_XI, "n_cells": N_CELLS, "r_step": R_STEP, "xi_step": XI_STEP, "vol": vol,
         "ni": 1.0, "max_radius": WW, "corrector_steps": 2, "n_attempts": 4,
         "checkpoint": True, "magnetic_field": 0.0}

    dt, steps0 = jbeam.init_substepping(jnp.asarray(b["p_z"]), jnp.asarray(b["q_m"]),
                                        jnp.zeros_like(b["p_z"]), jnp.zeros_like(b["p_z"]),
                                        TIME_STEP, E_SS)
    max_sub = int(np.asarray(steps0).max())
    beam = {k: jnp.asarray(b[k]) for k in b}
    beam["remaining_steps"] = steps0

    step_fn = jax.jit(lambda beam: td.one_time_step(beam, PLASMA, P, dt, max_sub))

    print("\nSelf-consistent evolution (driver in its own wake):")
    print(f"{'step':>4} {'mean p_z':>12} {'peak |Ez|':>12} {'lost':>6}")
    energies = []
    for n in range(15):
        beam, ez, lost = step_fn(beam)
        mean_pz = float(jnp.mean(beam["p_z"]))
        energies.append(mean_pz)
        print(f"{n:4d} {mean_pz:12.4f} {float(jnp.max(jnp.abs(ez))):12.4e} {int(jnp.sum(lost)):6d}")

    print(f"\ndriver mean energy change over 15 steps: {energies[-1] - b['p_z'].mean():+.3f}")

    print("\nDifferentiability through the whole time loop:")
    beam0 = {k: jnp.asarray(b[k]) for k in b}; beam0["remaining_steps"] = steps0

    def final_energy(pz_scale):
        beam = dict(beam0); beam["p_z"] = beam0["p_z"] * pz_scale
        for _ in range(5):
            beam, ez, lost = td.one_time_step(beam, PLASMA, P, dt, max_sub)
        return jnp.mean(beam["p_z"])

    g = float(jax.grad(final_energy)(1.0))
    eps = 1e-4
    fd = float((final_energy(1.0 + eps) - final_energy(1.0 - eps)) / (2 * eps))
    print(f"  d(final mean p_z)/d(initial energy scale): grad={g:.4f}  finite-diff={fd:.4f}  "
          f"rel={abs(g-fd)/(abs(fd)+1e-30):.2e}")


if __name__ == "__main__":
    main()
