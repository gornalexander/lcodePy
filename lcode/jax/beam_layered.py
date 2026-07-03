"""
Layer-locked beam for the JAX time-step pipeline.

The beam is bucketed by xi-layer into a fixed-capacity `[n_layers, C]` grid (each particle stays
in its layer -- valid for relativistic beams, where the per-step xi drift is << dxi; the
cross-layer "fell" path is issue #22).

Key simplification: the plasma march is unchanged -- `deposit_beam_full` on the flat beam is
bit-identical to numba's per-layer deposit, so the wake history is the same. The layer-locked
beam push is then just numba's per-layer pusher `push_beam_layer` applied independently to each
layer's bucket, with the numba offset "push bucket[L] bracketed by fields[L] -> fields[L+1]".
Because the layers are independent (no fell), that push is a single `vmap` over layers -- and the
same per-layer cell is what the time-step wavefront (`wavefront_evolve`) batches across steps.
"""
import numpy as np
import jax
import jax.numpy as jnp

from .beam import deposit_beam_full, deposit_beam_layer, push_beam_layer, init_substepping
from .march import march_with_fields, step_dxi, FIELD_KEYS

jax.config.update("jax_enable_x64", True)

_BUCKET_KEYS = ("r", "xi", "p_z", "p_r", "M", "q_m", "q_norm")


def bucket_beam(beam, n_layers, xi_step, capacity=None):
    """Scatter a flat beam into layer-locked buckets `[n_layers, C]`.

    Returns (bucket, order_info). `bucket` holds `[n_layers, C]` arrays for each field plus a
    boolean `active` mask (padding slots have q_norm=0, depositing/contributing nothing).
    `order_info` = (valid_idx, layer, slot, C, n_flat) lets `unbucket` restore the flat order.
    """
    def col(key):
        return np.asarray(getattr(beam, key) if hasattr(beam, key) else beam[key], dtype=np.float64)
    xi = col("xi")
    n_flat = xi.size
    layer_all = np.floor(-xi / xi_step).astype(np.int64)
    valid = np.where((layer_all >= 0) & (layer_all < n_layers))[0]
    layer = layer_all[valid]
    counts = np.bincount(layer, minlength=n_layers)
    C = int(capacity) if capacity is not None else int(counts.max() if counts.size else 0)
    # slot within layer = order of appearance
    slot = np.zeros(layer.size, dtype=np.int64)
    running = np.zeros(n_layers, dtype=np.int64)
    for i, L in enumerate(layer):
        slot[i] = running[L]; running[L] += 1
    keep = slot < C
    vi, lay, sl = valid[keep], layer[keep], slot[keep]

    bucket = {}
    for key in _BUCKET_KEYS:
        grid = np.zeros((n_layers, C), dtype=np.float64)
        grid[lay, sl] = col(key)[vi]
        bucket[key] = jnp.asarray(grid)
    active = np.zeros((n_layers, C), dtype=bool)
    active[lay, sl] = True
    bucket["active"] = jnp.asarray(active)
    return bucket, (vi, lay, sl, C, n_flat)


def unbucket_beam(bucket, order_info):
    """Restore a flat structured-like dict from layer buckets (inverse of bucket_beam placement)."""
    vi, lay, sl, C, n_flat = order_info
    flat = {}
    for key in _BUCKET_KEYS:
        arr = np.zeros(n_flat, dtype=np.float64)
        arr[vi] = np.asarray(bucket[key])[lay, sl]
        flat[key] = arr
    return flat


def push_buckets(bucket, dt, remaining, field_history, params, max_substeps):
    """Push every layer's bucket by one time step (vmap over layers, independent = layer-locked).

    numba offset: bucket[L] is bracketed by fields[L] (prev) and fields[L+1] (cur), pushed with
    xi_i = L+1. The last layer clamps L+1 -> L. Returns the pushed bucket (same [n_layers, C]).
    """
    n_layers = params["n_xi"]
    r_step, xi_step = params["r_step"], params["xi_step"]
    max_radius, Bext = params["max_radius"], params.get("magnetic_field", 0.0)
    L = jnp.arange(n_layers)
    Lp = jnp.minimum(L + 1, n_layers - 1)
    prev = {k: field_history[k][L] for k in FIELD_KEYS}      # [n_layers, n_cells]
    cur = {k: field_history[k][Lp] for k in FIELD_KEYS}

    def push_one(b_r, b_xi, b_pz, b_pr, b_M, b_qm, rem, d, prevf, curf, xi_i):
        return push_beam_layer(b_r, b_xi, b_pz, b_pr, b_M, b_qm, rem, d, prevf, curf,
                               xi_i, r_step, xi_step, max_radius, Bext, max_substeps)

    r, xi, pz, pr, M, rem, lost = jax.vmap(push_one)(
        bucket["r"], bucket["xi"], bucket["p_z"], bucket["p_r"], bucket["M"], bucket["q_m"],
        remaining, dt, prev, cur, L + 1)
    out = {"r": r, "xi": xi, "p_z": pz, "p_r": pr, "M": M,
           "q_m": bucket["q_m"], "q_norm": bucket["q_norm"], "active": bucket["active"]}
    return out


def layered_time_step(bucket, plasma, params, time_step, substep_energy, ions=None):
    """One quasistatic time step, layer-locked beam. Returns (evolved_bucket, history, final).

    The plasma march is the standard whole-beam march (the per-layer bucket flattened for
    `deposit_beam_full` gives rho identical to numba; padding slots have q_norm=0). The beam is
    pushed per-layer. Substepping is re-initialised for the beam this step (as numba does).
    """
    flat_r = bucket["r"].ravel()
    flat_xi = bucket["xi"].ravel()
    flat_q = bucket["q_norm"].ravel()
    rho_beam = deposit_beam_full(flat_r, flat_xi, flat_q,
                                 params["n_xi"], params["n_cells"], params["r_step"], params["xi_step"])
    history, final = march_with_fields(plasma["particles"], plasma["fields"],
                                       plasma["currents"], rho_beam, params, ions=ions)

    zero = jnp.zeros_like(bucket["p_z"])
    dt, remaining = init_substepping(bucket["p_z"], bucket["q_m"], zero, zero,
                                     time_step, substep_energy)
    remaining = jnp.where(bucket["active"], remaining, 0)
    max_substeps = int(np.asarray(remaining).max())
    evolved_bucket = push_buckets(bucket, dt, remaining, history, params, max_substeps)
    return evolved_bucket, history, final


def sequential_evolve(bucket0, plasma, params, time_step, substep_energy, n_steps, ions=None):
    """Evolve the beam for `n_steps` time steps, step by step (the correctness oracle).

    Returns (evolved_bucket, final_plasma_state). `final_plasma_state` is (electrons, fields,
    currents) or (electrons, ions, fields, currents) with mobile ions -- the last step's plasma.
    """
    bucket = bucket0
    final = None
    for _ in range(n_steps):
        bucket, _history, final = layered_time_step(bucket, plasma, params,
                                                    time_step, substep_energy, ions)
    return bucket, final


def _one_cell(state, bucket_now, xi_i, params, time_step, substep_energy, max_substeps):
    """One (lane, layer) wavefront cell. `state` = (electrons, ions, fields_prev, currents,
    rho_layout, bucket_prev). Deposit bucket_now (= layer j), step plasma, push bucket_prev
    (= layer j-1) bracketed by fields[j-1]->fields[j] (xi_i = j). Returns (new_state, pushed)."""
    electrons, ions, fields_prev, currents, rho_layout, bucket_prev = state
    r_step, xi_step, n_cells = params["r_step"], params["xi_step"], params["n_cells"]
    max_radius, Bext = params["max_radius"], params.get("magnetic_field", 0.0)

    dens, rho_next = deposit_beam_layer(rho_layout, bucket_now["r"], bucket_now["xi"],
                                        bucket_now["q_norm"], xi_i, r_step, xi_step, n_cells)
    res = step_dxi(electrons, fields_prev, currents, dens, params, ions=ions)
    if ions is None:
        electrons_n, fields_now, currents_n = res
        ions_n = None
    else:
        electrons_n, ions_n, fields_now, currents_n = res

    zero = jnp.zeros_like(bucket_prev["p_z"])
    dt, rem = init_substepping(bucket_prev["p_z"], bucket_prev["q_m"], zero, zero,
                              time_step, substep_energy)
    rem = jnp.where(bucket_prev["active"], rem, 0)
    r, xi, pz, pr, M, rem_o, lost = push_beam_layer(
        bucket_prev["r"], bucket_prev["xi"], bucket_prev["p_z"], bucket_prev["p_r"],
        bucket_prev["M"], bucket_prev["q_m"], rem, dt,
        fields_prev, fields_now, xi_i, r_step, xi_step, max_radius, Bext, max_substeps)
    pushed = {"r": r, "xi": xi, "p_z": pz, "p_r": pr, "M": M,
              "q_m": bucket_prev["q_m"], "q_norm": bucket_prev["q_norm"],
              "active": bucket_prev["active"]}
    new_state = (electrons_n, ions_n, fields_now, currents_n, rho_next, bucket_now)
    return new_state, pushed


def wavefront_evolve(bucket0, plasma, params, time_step, substep_energy, n_steps,
                     ions0=None, max_substeps=1, keep_history=False):
    """Evolve the beam for `n_steps` time steps as a diagonal wavefront (pipelined time steps).

    Numerically identical to `sequential_evolve`, but every scan tick advances all in-flight time
    steps together. Schedule D = 2*t + j (lane t at layer j = (D - 2t)): the numba field offset
    (push bucket[j-1] with fields[j-1]->fields[j]) makes the beam hand-off land one diagonal after
    it is produced, so a stride-2 schedule keeps the dependencies causal.

    Returns (evolved_bucket_last_lane, final_plasma_state, history). `history` is None unless
    `keep_history`, in which case it is a dict of `[n_steps, n_layers, n_cells]` wake fields
    (history[t, j] = fields of time step t at xi-layer j), for fields-vs-xi diagnostics.
    """
    P = n_steps
    n_layers, n_cells = params["n_xi"], params["n_cells"]
    C = bucket0["r"].shape[1]
    lanes = jnp.arange(P)
    bc = lambda x: jnp.broadcast_to(x, (P,) + x.shape)

    electrons = {k: bc(v) for k, v in plasma["particles"].items()}
    ions = None if ions0 is None else {k: bc(v) for k, v in ions0.items()}
    fields_prev = {k: jnp.zeros((P, n_cells)) for k in FIELD_KEYS}
    currents = {k: bc(v) for k, v in plasma["currents"].items()}
    rho_layout = jnp.zeros((P, n_cells))
    zero_bucket = {k: jnp.zeros((P, C)) for k in _BUCKET_KEYS}
    zero_bucket["active"] = jnp.zeros((P, C), dtype=bool)
    bucket_prev = zero_bucket
    # beamP[t] = lane t's input buckets; lane 0 = bucket0, higher lanes filled by hand-off.
    beamP = {k: jnp.where((lanes == 0).reshape((P,) + (1,) * bucket0[k].ndim),
                          bc(bucket0[k]), 0 if bucket0[k].dtype != bool else False)
             for k in bucket0}
    # collect each lane's pushed output per layer -> evolved buckets [P, n_layers, C]
    out_beam = {k: jnp.zeros((P, n_layers, C), dtype=bucket0[k].dtype) for k in bucket0}
    # optional per-(time step, layer) wake-field history for diagnostics
    history = {k: jnp.zeros((P, n_layers, n_cells)) for k in FIELD_KEYS} if keep_history else None

    n_diag = 2 * (P - 1) + n_layers   # last cell: t=P-1, j=n_layers-1 -> D=2(P-1)+(n_layers-1)
    state = (electrons, ions, fields_prev, currents, rho_layout, bucket_prev)

    cell = lambda st, bn, xi_i: _one_cell(st, bn, xi_i, params, time_step, substep_energy, max_substeps)
    vcell = jax.vmap(cell)

    def body(carry, D):
        electrons, ions, fields_prev, currents, rho_layout, bucket_prev, beamP, out_beam, history = carry
        state = (electrons, ions, fields_prev, currents, rho_layout, bucket_prev)
        j = D - 2 * lanes                                  # layer each lane works on
        active = (j >= 0) & (j < n_layers)
        jc = jnp.clip(j, 0, n_layers - 1)
        bucket_now = {k: beamP[k][lanes, jc] for k in beamP}          # (P, C) gather

        new_state, pushed = vcell(state, bucket_now, jc)
        (e_n, i_n, f_n, c_n, rl_n, bprev_n) = new_state

        if keep_history:                                   # history[t, j] = fields of step t, layer j
            hm = active.reshape((P,) + (1,) * (f_n["E_z"].ndim - 1))
            history = {k: history[k].at[lanes, jc].set(
                jnp.where(hm, f_n[k], history[k][lanes, jc])) for k in history}

        def sel(new, old):
            a = active.reshape((P,) + (1,) * (new.ndim - 1))
            return jnp.where(a, new, old)

        electrons = {k: sel(e_n[k], electrons[k]) for k in electrons}
        ions = None if ions is None else {k: sel(i_n[k], ions[k]) for k in ions}
        fields_prev = {k: sel(f_n[k], fields_prev[k]) for k in fields_prev}
        currents = {k: sel(c_n[k], currents[k]) for k in currents}
        rho_layout = sel(rl_n, rho_layout)
        bucket_prev = {k: sel(bprev_n[k], bucket_prev[k]) for k in bucket_prev}

        # store lane t's pushed bucket (= its layer j-1) into out_beam[t, j-1], and hand it to the
        # next lane's input beamP[t+1, j-1]. Only where active and j-1 >= 0.
        jm1 = jc - 1
        can = active & (j - 1 >= 0)
        tgt_out = jnp.clip(jm1, 0, n_layers - 1)
        for k in out_beam:
            m = can.reshape((P,) + (1,) * (pushed[k].ndim - 1))
            out_beam[k] = out_beam[k].at[lanes, tgt_out].set(
                jnp.where(m, pushed[k], out_beam[k][lanes, tgt_out]))
        # hand-off to lane t+1 (receiver q = t+1): q gets pushed[q-1] at layer j-1.
        recv = lanes                                        # q
        src = {k: jnp.roll(pushed[k], 1, axis=0) for k in pushed}   # src[q] = pushed[q-1]
        can_recv = jnp.roll(can, 1) & (recv >= 1)
        jm1_recv = jnp.roll(jm1, 1)
        tgt_recv = jnp.clip(jm1_recv, 0, n_layers - 1)
        for k in beamP:
            if k == "active":
                continue
            m = can_recv.reshape((P,) + (1,) * (src[k].ndim - 1))
            beamP[k] = beamP[k].at[recv, tgt_recv].set(
                jnp.where(m, src[k], beamP[k][recv, tgt_recv]))
        m = can_recv.reshape((P, 1))
        beamP["active"] = beamP["active"].at[recv, tgt_recv].set(
            jnp.where(m, src["active"], beamP["active"][recv, tgt_recv]))

        carry = (electrons, ions, fields_prev, currents, rho_layout, bucket_prev,
                 beamP, out_beam, history)
        return carry, None

    carry = (electrons, ions, fields_prev, currents, rho_layout, bucket_prev,
             beamP, out_beam, history)
    carry, _ = jax.lax.scan(body, carry, jnp.arange(n_diag))
    electrons, ions, fields_prev, currents, rho_layout, bucket_prev, beamP, out_beam, history = carry

    # last lane's evolved beam; the final layer (n_layers-1) was never pushed (no j+1), push it now.
    evolved = {k: out_beam[k][P - 1] for k in out_beam}
    last_bucket = {k: beamP[k][P - 1, n_layers - 1] for k in beamP}
    zero = jnp.zeros_like(last_bucket["p_z"])
    dt, rem = init_substepping(last_bucket["p_z"], last_bucket["q_m"], zero, zero,
                              time_step, substep_energy)
    rem = jnp.where(last_bucket["active"], rem, 0)
    lf = {k: fields_prev[k][P - 1] for k in FIELD_KEYS}
    r, xi, pz, pr, M, rem_o, lost = push_beam_layer(
        last_bucket["r"], last_bucket["xi"], last_bucket["p_z"], last_bucket["p_r"],
        last_bucket["M"], last_bucket["q_m"], rem, dt, lf, lf, n_layers,
        params["r_step"], params["xi_step"], params["max_radius"],
        params.get("magnetic_field", 0.0), max_substeps)
    for k, v in (("r", r), ("xi", xi), ("p_z", pz), ("p_r", pr), ("M", M)):
        evolved[k] = evolved[k].at[n_layers - 1].set(v)

    if ions is None:
        final = ({k: electrons[k][P - 1] for k in electrons},
                 {k: fields_prev[k][P - 1] for k in fields_prev},
                 {k: currents[k][P - 1] for k in currents})
    else:
        final = ({k: electrons[k][P - 1] for k in electrons},
                 {k: ions[k][P - 1] for k in ions},
                 {k: fields_prev[k][P - 1] for k in fields_prev},
                 {k: currents[k][P - 1] for k in currents})
    return evolved, final, history
