"""
The xi-march: one predictor-corrector plasma layer (`step_dxi`) and the full sweep (`march`).

`step_dxi` advances the plasma by one xi-layer; `march` scans it over all layers, driven by a
precomputed rho_beam[xi, r]. Both take an optional second `ions` species (ion-model 'mobile'):
when given, the ions are pushed by the same fields and deposited into row 1 of the currents;
when None (ion-model 'background') row 1 is the static background density `ni`.

Not yet ported: the OUTER step_dxi substepping (the coarse xi/10 reduction when charge_move
exceeds the sensitivity). It is data-dependent control flow that needs a bounded scan+mask; see
issue #21. Fine for weak/moderate drivers, where numba never triggers it either.
"""
import jax
import jax.numpy as jnp

from . import field_solver as jfs
from . import deposition as jdep
from . import move as jm

jax.config.update("jax_enable_x64", True)

FIELD_KEYS = ("E_r", "E_f", "E_z", "B_f", "B_z")


def _move_species(species, fields, params):
    """Push one plasma species in `fields`; returns the updated species dict (mass unchanged)."""
    r, p_r, p_f, p_z, q = jm.move_particles(
        fields["E_r"], fields["E_f"], fields["E_z"], fields["B_f"], fields["B_z"],
        species["r"], species["p_r"], species["p_f"], species["p_z"],
        species["q"], species["m"], params["r_step"], params["xi_step"], params["max_radius"],
        n_attempts=params.get("n_attempts", jm.N_SUBSTEP_ATTEMPTS))
    return {"r": r, "p_r": p_r, "p_f": p_f, "p_z": p_z, "q": q, "m": species["m"]}


def step_dxi(electrons, fields, currents, rho_beam, params, ions=None):
    """One xi-step (predictor + `corrector_steps` correctors), no outer substepping.

    Returns (electrons, fields, currents) for background, or (electrons, ions, fields, currents)
    when a mobile `ions` species is given. Each corrector re-moves from the ORIGINAL species with
    the freshly averaged field, matching numba's predictor-corrector.
    """
    def deposit(moved_electrons, moved_ions):
        return jdep.compute_rhoj(moved_electrons, params["n_cells"], params["r_step"],
                                 params["vol"], params["ni"], ions=moved_ions)

    def advance(fields_for_move):
        moved_electrons = _move_species(electrons, fields_for_move, params)
        moved_ions = _move_species(ions, fields_for_move, params) if ions is not None else None
        return moved_electrons, moved_ions, deposit(moved_electrons, moved_ions)

    electrons_new, ions_new, currents_new = advance(fields)        # predictor
    fields_new = fields
    for _ in range(params["corrector_steps"]):                     # correctors
        fields_new, fields_avg = jfs.compute_fields(fields_new, fields, rho_beam, currents,
                                                    currents_new, params["xi_step"], params["r_step"])
        electrons_new, ions_new, currents_new = advance(fields_avg)

    if ions is None:
        return electrons_new, fields_new, currents_new
    return electrons_new, ions_new, fields_new, currents_new


def march(electrons, fields, currents, rho_beam_seq, params, ions=None, keep_history=False):
    """Scan `step_dxi` over the xi-layers in `rho_beam_seq` (shape [n_layers, n_cells]).

    Returns (history, final_state):
      - history is the per-layer wake-field dict F[n_layers, n_cells] if `keep_history`, else the
        on-axis E_z[n_layers] (cheap diagnostic);
      - final_state is (electrons, fields, currents), or (electrons, ions, fields, currents) with
        mobile ions -- the plasma state at the last layer.

    Set params["checkpoint"]=True to rematerialise each layer in the backward pass (memory for grad).
    """
    def body(carry, rho_beam):
        electrons, ions, fields, currents = carry
        if ions is None:
            electrons, fields, currents = step_dxi(electrons, fields, currents, rho_beam, params)
        else:
            electrons, ions, fields, currents = step_dxi(
                electrons, fields, currents, rho_beam, params, ions=ions)
        history_entry = fields if keep_history else fields["E_z"][0]
        return (electrons, ions, fields, currents), history_entry

    if params.get("checkpoint", False):
        body = jax.checkpoint(body)

    (electrons, ions, fields, currents), history = jax.lax.scan(
        body, (electrons, ions, fields, currents), rho_beam_seq)
    final_state = (electrons, fields, currents) if ions is None else (electrons, ions, fields, currents)
    return history, final_state


def march_with_fields(electrons, fields, currents, rho_beam_seq, params, ions=None):
    """`march` keeping the full per-layer wake-field history (for pushing the beam afterwards)."""
    return march(electrons, fields, currents, rho_beam_seq, params, ions=ions, keep_history=True)
