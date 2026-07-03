"""
JAX backend for the 2D `Simulation` (selected with config `backend: jax`).

Evolves the beam self-consistently with the differentiable dynamics (deposit_beam_full ->
march_with_fields -> push_beam_history) and returns the plasma state in numba's `Arrays` format,
so it is a drop-in replacement for the numba solver's `_plasmastate`. 2D; ion-model 'background'
or 'mobile' (see `march.step_dxi`).

Two entry points share the same solver setup (`_build_solver_state`):
  - `run_2d`      : run n_steps internally (self-contained; used by tests/notebooks);
  - `make_engine` : a per-time-step engine that also returns the evolved beam, so the unified
                    `Simulation.step()` pipeline can hand the beam through a source/drain like numba.
"""
import numpy as np
import jax
import jax.numpy as jnp

from ..plasma.data import Arrays
from ..beam.data import particle_dtype
from . import (deposit_beam_full, march_with_fields, push_beam_history,
               init_substepping, compute_rhoj, cell_volume, FIELD_KEYS)

jax.config.update("jax_enable_x64", True)

_PLASMA_KEYS = ("r", "p_r", "p_f", "p_z", "q", "m")
_BEAM_KEYS = ("r", "xi", "p_z", "p_r", "M", "q_m", "q_norm")


def _species(r, q, m):
    """A plasma species dict (particles start at rest) from positions `r` and charges/masses."""
    r = jnp.asarray(r)
    zero = jnp.zeros_like(r)
    return {"r": r, "p_r": zero, "p_f": zero, "p_z": zero,
            "q": jnp.asarray(q), "m": jnp.asarray(m)}


def _build_solver_state(config):
    """Build the (fresh) plasma state and solver params from config. Returns (plasma, params).

    The plasma is quasistatic — reset at the start of every time step — so this is called once
    and reused: electrons (and mobile ions) at cell centres at rest, plus the `params` dict the
    xi-march reads. `params["_time_step"]` / `["_substep_energy"]` carry beam-substepping config.
    """
    window_width = config.getfloat('window-width')
    window_length = config.getfloat('window-length')
    r_step = config.getfloat('transverse-step')
    xi_step = config.getfloat('xi-step')
    particles_per_cell = config.getint('plasma-particles-per-cell')

    n_cells = int(window_width / r_step) + 1
    # numba marches xi_i = 0 .. xi_steps, i.e. window_length/xi_step + 1 layers (an extra front
    # layer at xi=0). March the same count so the JAX plasma state lines up with numba's.
    n_layers = int(window_length / xi_step) + 1

    # Plasma electrons at cell centres, at rest (init_plasma_2d layout).
    n_plasma = int(window_width / r_step) * particles_per_cell
    r_plasma = (np.arange(n_plasma) + 0.5) * (window_width / n_plasma)
    weight = 2 * np.pi * (r_plasma[1] - r_plasma[0]) * r_plasma   # macroparticle weight = |q_electron|
    electrons = _species(r_plasma, q=-weight, m=weight)
    cell_vol = cell_volume(r_step, particles_per_cell, n_cells)

    # Mobile ions (ion-model 'mobile'): heavy positive species co-located with the electrons.
    ions = None
    if config.get('ion-model') == 'mobile':
        ions = _species(r_plasma, q=weight, m=config.getint('ion-mass') * weight)

    plasma = {"particles": electrons, "fields": {k: jnp.zeros(n_cells) for k in FIELD_KEYS},
              "currents": compute_rhoj(electrons, n_cells, r_step, cell_vol, 1.0, ions=ions)}
    if ions is not None:
        plasma["ions"] = ions

    params = {"n_xi": n_layers, "n_cells": n_cells, "r_step": r_step, "xi_step": xi_step,
              "vol": cell_vol, "ni": 1.0, "max_radius": window_width,
              "corrector_steps": config.getint('correctotransverse-steps'),
              "n_attempts": 4, "checkpoint": False,
              "magnetic_field": config.getfloat('magnetic-field'),
              "_time_step": config.getfloat('time-step'),
              "_substep_energy": config.getfloat('beam-substepping-energy')}
    return plasma, params


def _beam_dict(beam_array):
    """Read a beam (structured numpy array `beam[k]` or a BeamParticles object `beam.k`) to a dict
    of JAX arrays, plus the integer `id` (carried through, defaulting to arange)."""
    def field(key):
        raw = getattr(beam_array, key) if hasattr(beam_array, key) else beam_array[key]
        return jnp.asarray(np.asarray(raw, dtype=np.float64))
    beam = {k: field(k) for k in _BEAM_KEYS}
    if hasattr(beam_array, "id"):
        beam_id = np.asarray(beam_array.id)
    elif hasattr(beam_array, "dtype") and beam_array.dtype.names and "id" in beam_array.dtype.names:
        beam_id = np.asarray(beam_array["id"])
    else:
        beam_id = np.arange(np.asarray(beam["r"]).size)
    return beam, beam_id


def _to_arrays(species):
    return Arrays(xp=np, age=np.zeros(np.asarray(species["r"]).size),
                  **{a: np.asarray(species[a]) for a in _PLASMA_KEYS})


def _finalize(new_beam, electrons_f, fields_f, currents_f, ions_f, history, beam_id):
    """Convert one step's JAX outputs to numba `Arrays` + a structured evolved-beam array."""
    particles = {"electrons": _to_arrays(electrons_f)}
    if ions_f is not None:
        particles["ions"] = _to_arrays(ions_f)
    fields = Arrays(xp=np, **{k: np.asarray(fields_f[k]) for k in FIELD_KEYS})
    currents = Arrays(xp=np, **{k: np.asarray(currents_f[k]) for k in ("rho", "j_r", "j_f", "j_z")})
    field_history = {k: np.asarray(history[k]) for k in FIELD_KEYS}   # (n_xi, n_cells) per field

    evolved = np.zeros(np.asarray(new_beam["r"]).size, dtype=particle_dtype)
    for k in _BEAM_KEYS:
        evolved[k] = np.asarray(new_beam[k])
    evolved["id"] = beam_id
    return particles, fields, currents, field_history, evolved


def make_engine(config):
    """Build the JAX solver once; return `one_step(beam) -> (particles, fields, currents,
    field_history, evolved_beam)` running a single quasistatic time step.

    `beam` is a structured array / BeamParticles; `evolved_beam` is a structured array in
    particle_dtype order. Substepping is (re)initialised each step, matching numba (the beam
    source resets dt=0 every time step). jitted step functions are cached per substep depth.
    """
    plasma, params = _build_solver_state(config)
    time_step, substep_energy = params["_time_step"], params["_substep_energy"]
    step_cache = {}

    def one_step(beam_array):
        beam, beam_id = _beam_dict(beam_array)
        dt, remaining_steps = init_substepping(beam["p_z"], beam["q_m"],
                                               jnp.zeros_like(beam["p_z"]),
                                               jnp.zeros_like(beam["p_z"]), time_step, substep_energy)
        max_substeps = int(np.asarray(remaining_steps).max())
        beam["remaining_steps"] = remaining_steps
        if max_substeps not in step_cache:
            step_cache[max_substeps] = jax.jit(
                lambda b, d: _one_step(b, plasma, params, d, max_substeps))
        new_beam, (electrons_f, fields_f, currents_f, ions_f), history = \
            step_cache[max_substeps](beam, dt)
        jax.block_until_ready(electrons_f["r"])
        return _finalize(new_beam, electrons_f, fields_f, currents_f, ions_f, history, beam_id)

    return one_step


def run_2d(config, beam_array, n_steps):
    """Run `n_steps` time steps in JAX; return (particles, fields, currents, field_history).

    Self-contained (threads the beam internally). `Simulation.step()` uses `make_engine` instead,
    so the beam flows through the shared source/drain pipeline; this stays for tests/notebooks.
    """
    engine = make_engine(config)
    beam = beam_array
    result = None
    for _ in range(n_steps):
        particles, fields, currents, field_history, beam = engine(beam)
        result = (particles, fields, currents, field_history)
    return result


def _one_step(beam, plasma, params, dt, max_substeps):
    """One quasistatic time step: deposit the beam, march the plasma, push the beam in the wake.

    Returns (new_beam, (electrons, fields, currents, ions), field_history), where `ions` is None
    for ion-model 'background' or the final ion species for 'mobile'.
    """
    rho_beam = deposit_beam_full(beam["r"], beam["xi"], beam["q_norm"],
                                 params["n_xi"], params["n_cells"], params["r_step"], params["xi_step"])
    ions = plasma.get("ions")
    history, final = march_with_fields(plasma["particles"], plasma["fields"],
                                       plasma["currents"], rho_beam, params, ions=ions)
    if ions is None:
        (electrons, fields, currents), ions_f = final, None
    else:
        electrons, ions_f, fields, currents = final

    r, xi, p_z, p_r, M, lost = push_beam_history(
        beam["r"], beam["xi"], beam["p_z"], beam["p_r"], beam["M"], beam["q_m"],
        beam["remaining_steps"], dt, history, params["r_step"], params["xi_step"],
        params["max_radius"], params.get("magnetic_field", 0.0), max_substeps)
    new_beam = {**beam, "r": r, "xi": xi, "p_z": p_z, "p_r": p_r, "M": M}
    return new_beam, (electrons, fields, currents, ions_f), history
