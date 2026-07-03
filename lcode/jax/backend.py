"""
JAX backend entry point for the 2D `Simulation` (selected with config `backend: jax`).

Evolves the beam self-consistently with the validated differentiable dynamics (deposit_beam_full
-> march_with_fields -> push_beam_history, i.e. `one_time_step`) and returns the final plasma
state in numba's `Arrays` format, so it is a drop-in replacement for the numba solver's
`_plasmastate`. 2D, background ions only (the JAX solver does not model mobile ions yet).
"""
import numpy as np
import jax
import jax.numpy as jnp

from ..plasma.data import Arrays
from . import (deposit_beam_full, march_with_fields, march_with_fields_ions,
               push_beam_history, init_substepping, compute_rhoj, cell_volume, FIELD_KEYS)

jax.config.update("jax_enable_x64", True)


def run_2d(config, beam_array, n_steps):
    """Run `n_steps` time steps in JAX.

    Returns ({'electrons': Arrays}, fields, currents, field_history), where field_history is a
    dict {E_r, E_f, E_z, B_f, B_z} of (n_xi, n_cells) arrays: the wake fields at every xi-layer of
    the last time step (so diagnostics can plot fields vs xi without re-marching).
    """
    WW = config.getfloat('window-width'); R = config.getfloat('transverse-step')
    XI = config.getfloat('xi-step'); WL = config.getfloat('window-length')
    PPC = config.getint('plasma-particles-per-cell')
    E_SS = config.getfloat('beam-substepping-energy'); TS = config.getfloat('time-step')
    Bext = config.getfloat('magnetic-field')
    NC = int(WW / R) + 1
    # numba marches xi_i = 0 .. xi_steps, i.e. window_length/xi_step + 1 layers (an extra front
    # layer at xi=0). March the same count so the JAX plasma state lines up with numba's.
    NXI = int(WL / XI) + 1

    # plasma initial state (init_plasma_2d layout)
    n = int(WW / R) * PPC
    rp = (np.arange(n) + 0.5) * (WW / n)
    mp = 2 * np.pi * (rp[1] - rp[0]) * rp
    part = {"r": jnp.asarray(rp), "p_r": jnp.zeros(n), "p_f": jnp.zeros(n),
            "p_z": jnp.zeros(n), "q": jnp.asarray(-mp), "m": jnp.asarray(mp)}
    vol = cell_volume(R, PPC, NC)

    # Mobile ions (ion-model 'mobile'): a second, heavy, positive species starting at the electron
    # positions at rest, pushed by the same fields (init_plasma mobile branch: q_ion = -q_e,
    # m_ion = ion_mass * |q_e|). With 'background' the ion row is the static density ni.
    ions = None
    if config.get('ion-model') == 'mobile':
        ion_mass = config.getint('ion-mass')
        ions = {"r": jnp.asarray(rp), "p_r": jnp.zeros(n), "p_f": jnp.zeros(n),
                "p_z": jnp.zeros(n), "q": jnp.asarray(mp), "m": jnp.asarray(ion_mass * mp)}

    plasma = {"particles": part, "fields": {k: jnp.zeros(NC) for k in FIELD_KEYS},
              "currents": compute_rhoj(part, NC, R, vol, 1.0, ions=ions)}
    if ions is not None:
        plasma["ions"] = ions
    p = {"n_xi": NXI, "n_cells": NC, "r_step": R, "xi_step": XI, "vol": vol, "ni": 1.0,
         "max_radius": WW, "corrector_steps": config.getint('correctotransverse-steps'),
         "n_attempts": 4, "checkpoint": False, "magnetic_field": Bext}

    # beam (accept a structured numpy array `beam_array[k]` or a BeamParticles object `beam.k`)
    def field(k):
        raw = getattr(beam_array, k) if hasattr(beam_array, k) else beam_array[k]
        return jnp.asarray(np.asarray(raw, dtype=np.float64))
    beam = {k: field(k) for k in ("r", "xi", "p_z", "p_r", "M", "q_m", "q_norm")}
    dt, steps = init_substepping(beam["p_z"], beam["q_m"], jnp.zeros_like(beam["p_z"]),
                                 jnp.zeros_like(beam["p_z"]), TS, E_SS)
    msub = int(np.asarray(steps).max())
    beam["remaining_steps"] = steps

    step_fn = jax.jit(lambda b, dt: _one_step(b, plasma, p, dt, msub))
    pf = ff = cf = iff = history = None
    for _ in range(n_steps):
        beam, (pf, ff, cf, iff), history = step_fn(beam, dt)
    jax.block_until_ready(pf["r"])

    def to_arrays(sp):
        return Arrays(xp=np, age=np.zeros(np.asarray(sp["r"]).size),
                      **{a: np.asarray(sp[a]) for a in ("r", "p_r", "p_f", "p_z", "q", "m")})

    particles = {"electrons": to_arrays(pf)}
    if iff is not None:
        particles["ions"] = to_arrays(iff)
    fields = Arrays(xp=np, **{k: np.asarray(ff[k]) for k in FIELD_KEYS})
    currents = Arrays(xp=np, **{k: np.asarray(cf[k]) for k in ("rho", "j_r", "j_f", "j_z")})
    field_history = {k: np.asarray(history[k]) for k in FIELD_KEYS}   # (n_xi, n_cells) per field
    return particles, fields, currents, field_history


def _one_step(beam, plasma, p, dt, msub):
    """One time step; returns (new_beam, (final electrons, fields, currents, ions), field_history).

    `ions` is None for ion-model 'background', or the final ion species for 'mobile'.
    """
    rho = deposit_beam_full(beam["r"], beam["xi"], beam["q_norm"],
                            p["n_xi"], p["n_cells"], p["r_step"], p["xi_step"])
    if "ions" in plasma:
        hist, (pf, iff, ff, cf) = march_with_fields_ions(
            plasma["particles"], plasma["ions"], plasma["fields"], plasma["currents"], rho, p)
    else:
        hist, (pf, ff, cf) = march_with_fields(plasma["particles"], plasma["fields"],
                                               plasma["currents"], rho, p)
        iff = None
    r, xi, pz, pr, M, lost = push_beam_history(
        beam["r"], beam["xi"], beam["p_z"], beam["p_r"], beam["M"], beam["q_m"],
        beam["remaining_steps"], dt, hist, p["r_step"], p["xi_step"], p["max_radius"],
        p.get("magnetic_field", 0.0), msub)
    new_beam = {**beam, "r": r, "xi": xi, "p_z": pz, "p_r": pr, "M": M}
    return new_beam, (pf, ff, cf, iff), hist
