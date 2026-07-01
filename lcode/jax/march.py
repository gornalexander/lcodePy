"""
JAX assembly of one xi-layer (predictor-corrector) and the full xi-march (M4).

Combines the validated M1-M3 kernels (field_solver_jax, deposition_jax, move_jax) into:
  - step_dxi: predictor (move + deposit) then `corrector_steps` x (fields + move + deposit),
  - march:    lax.scan over xi-layers, driven by a precomputed rho_beam[xi, r].

The OUTER step_dxi substepping (the coarse xi/10 reduction when charge_move exceeds the
sensitivity) is another data-dependent while_loop; it is omitted here (fine for the
weak-driver validation) and will need a bounded/relaxed form for the general case (M5).
"""
from functools import partial
import jax
import jax.numpy as jnp

from . import field_solver as jfs
from . import deposition as jdep
from . import move as jm

jax.config.update("jax_enable_x64", True)

FIELD_KEYS = ("E_r", "E_f", "E_z", "B_f", "B_z")


def step_dxi(particles, fields, currents, rho_beam, p):
    """One xi-step (no outer substepping). particles/fields/currents are dicts."""
    n, h, xi = p["n_cells"], p["r_step"], p["xi_step"]
    vol, ni, mr = p["vol"], p["ni"], p["max_radius"]

    n_attempts = p.get("n_attempts", jm.N_SUBSTEP_ATTEMPTS)

    def move(f):
        r, pr, pf, pz, q = jm.move_particles(
            f["E_r"], f["E_f"], f["E_z"], f["B_f"], f["B_z"],
            particles["r"], particles["p_r"], particles["p_f"], particles["p_z"],
            particles["q"], particles["m"], h, xi, mr, n_attempts=n_attempts)
        return {"r": r, "p_r": pr, "p_f": pf, "p_z": pz, "q": q, "m": particles["m"]}

    def deposit(pp):
        return jdep.compute_rhoj(pp, n, h, vol, ni)

    # Predictor
    p_new = move(fields)
    c_new = deposit(p_new)

    # Correctors
    fields_new = fields
    for _ in range(p["corrector_steps"]):
        fields_new, favg = jfs.compute_fields(fields_new, fields, rho_beam,
                                              currents, c_new, xi, h)
        # compute_fields returns dicts missing nothing; ensure all field keys present
        p_new = move(favg)
        c_new = deposit(p_new)
    return p_new, fields_new, c_new


def march(init_particles, init_fields, init_currents, rho_beam_seq, p):
    """Run the xi-march. rho_beam_seq: (n_layers, n_cells). Returns Ez-on-axis history.

    Set p["checkpoint"]=True to wrap each xi-step in jax.checkpoint (remat): recompute the
    layer in the backward pass instead of storing its tape, which makes reverse-mode grad
    over long marches feasible in memory.
    """
    def _step(particles, fields, currents, rho_beam):   # p closed over (static)
        return step_dxi(particles, fields, currents, rho_beam, p)
    if p.get("checkpoint", False):
        _step = jax.checkpoint(_step)

    def body(carry, rho_beam):
        particles, fields, currents = carry
        particles, fields, currents = _step(particles, fields, currents, rho_beam)
        return (particles, fields, currents), fields["E_z"][0]

    (pf, ff, cf), ez_axis = jax.lax.scan(
        body, (init_particles, init_fields, init_currents), rho_beam_seq)
    return ez_axis, (pf, ff, cf)


def march_with_fields(init_particles, init_fields, init_currents, rho_beam_seq, p):
    """Like `march`, but returns the full wake-field history F[k] for each xi-layer k
    (a dict of (n_layers, n_cells) arrays), for pushing the beam afterwards (MB3)."""
    def _step(particles, fields, currents, rho_beam):
        return step_dxi(particles, fields, currents, rho_beam, p)
    if p.get("checkpoint", False):
        _step = jax.checkpoint(_step)

    def body(carry, rho_beam):
        particles, fields, currents = carry
        particles, fields, currents = _step(particles, fields, currents, rho_beam)
        return (particles, fields, currents), fields

    (pf, ff, cf), field_hist = jax.lax.scan(
        body, (init_particles, init_fields, init_currents), rho_beam_seq)
    return field_hist, (pf, ff, cf)
