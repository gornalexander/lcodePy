"""
Single-trajectory time-step pipeline for the JAX backend — the on-device analogue of the
per-layer beam hand-off in `lcode/mpi/beam_io.py`.

`wavefront_marches` (pipelined.py) accelerates *independent* marches. To pipeline the
*consecutive time steps of one beam*, step t+1's input must be handed to it layer-by-layer as
step t produces each wake layer (exactly what `MPIBeamSource/Drain` do across MPI ranks). This
module does that on-device.

Model
-----
The beam is held **bucketed by xi-layer** (`[n_xi, C]`, layer-locked), so each layer's beam is
an independent chain over time steps coupled to the plasma only through the wake at that layer:

    deposit beam_t[j] -> rho_beam_t[j] --step_dxi--> F_t[j] --push--> beam_{t+1}[j].

Cell (t, j) therefore depends only on (t, j-1) [plasma / rho_layout carry within the march] and
(t-1, j) [the beam bucket from the previous step]. On the anti-diagonal d = t + j both parents
lie on d-1, so a scan over diagonals advances all in-flight steps at once — one batched
layer-solve per lane per tick. `pipelined_evolve` runs a block of `n_steps` time steps this way;
`sequential_fused_evolve` is the identical computation done step-by-step (the correctness oracle).

Kernels are the validated `deposit_beam_layer`, `step_dxi`, `push_beam_layer`. This uses the
per-layer (single-wake-layer) pusher and a layer-locked beam, so it is a faithful *variant* of
the whole-history dynamics in dynamics.py; both trace to the same numba kernels.
"""
import jax
import jax.numpy as jnp

from .march import step_dxi
from . import beam as jbeam

jax.config.update("jax_enable_x64", True)

_BEAM_KEYS = ("r", "xi", "p_z", "p_r", "M", "q_m", "q_norm")


def _cell(particles, fields, currents, rho_layout, bucket, j, msub, p):
    """One (time step, layer) cell: deposit bucket j, plasma step, push bucket j.

    Returns (new plasma carry, new rho_layout, pushed bucket). `bucket` is a dict of [C] arrays
    and carries its own per-particle `dt` and `remaining_steps`.
    """
    h, xi_s, nc = p["r_step"], p["xi_step"], p["n_cells"]
    mr, B = p["max_radius"], p.get("magnetic_field", 0.0)
    dens, rho_layout_next = jbeam.deposit_beam_layer(
        rho_layout, bucket["r"], bucket["xi"], bucket["q_norm"], j, h, xi_s, nc)
    p_new, f_new, c_new = step_dxi(particles, fields, currents, dens, p)
    r, xi_, pz, pr, M, _, lost = jbeam.push_beam_layer(
        bucket["r"], bucket["xi"], bucket["p_z"], bucket["p_r"], bucket["M"], bucket["q_m"],
        bucket["remaining_steps"], bucket["dt"], fields, f_new, j + 1, h, xi_s, mr, B, msub)
    out = {"r": r, "xi": xi_, "p_z": pz, "p_r": pr, "M": M,
           "q_m": bucket["q_m"], "q_norm": bucket["q_norm"],
           "remaining_steps": bucket["remaining_steps"], "dt": bucket["dt"]}
    return (p_new, f_new, c_new), rho_layout_next, out


def fused_time_step(plasma0, beam, msub, p):
    """One time step, layer-locked buckets, sequential over layers (the reference march)."""
    n_xi, nc = p["n_xi"], p["n_cells"]

    def body(carry, j):
        (particles, fields, currents), rho_layout = carry
        bucket = {k: beam[k][j] for k in beam}
        newpl, rho_next, out = _cell(particles, fields, currents, rho_layout, bucket, j, msub, p)
        return (newpl, rho_next), out

    init = ((plasma0["particles"], plasma0["fields"], plasma0["currents"]), jnp.zeros(nc))
    _, outs = jax.lax.scan(body, init, jnp.arange(n_xi))
    return outs                                   # dict of [n_xi, C]


def sequential_fused_evolve(plasma0, beam, msub, p, n_steps):
    """Evolve the beam for `n_steps` time steps, one after another (correctness oracle).

    Uses lax.scan over time steps (not a Python unroll) so the compiled graph stays small.
    """
    def body(beam, _):
        return fused_time_step(plasma0, beam, msub, p), None
    beam, _ = jax.lax.scan(body, beam, None, length=n_steps)
    return beam


def pipelined_evolve(plasma0, beam, msub, p, n_steps):
    """Evolve the beam for `n_steps` steps as one diagonal wavefront (batched across steps).

    Numerically identical to `sequential_fused_evolve`, but every scan tick advances all
    in-flight time steps together (one batched layer-solve per lane) — the GPU-friendly shape.
    """
    P = n_steps
    n_xi, nc = p["n_xi"], p["n_cells"]
    n_diag = P + n_xi - 1
    lanes = jnp.arange(P)

    bc = lambda x: jnp.broadcast_to(x, (P,) + x.shape)
    particles = {k: bc(v) for k, v in plasma0["particles"].items()}
    fields = {k: bc(v) for k, v in plasma0["fields"].items()}
    currents = {k: bc(v) for k, v in plasma0["currents"].items()}
    rho_layout = jnp.zeros((P, nc))
    # beam[lane]: lane 0 = initial beam; higher lanes filled by hand-off before first read.
    beamP = {k: bc(beam[k]) for k in beam}          # (P, n_xi, C)

    vcell = jax.vmap(_cell, in_axes=(0, 0, 0, 0, 0, 0, None, None))

    def body(carry, d):
        particles, fields, currents, rho_layout, beamP = carry
        j = d - lanes                               # layer each lane works on
        active = (j >= 0) & (j < n_xi)
        jc = jnp.clip(j, 0, n_xi - 1)
        bucket = {k: beamP[k][lanes, jc] for k in beamP}          # (P, C) gather

        (np_, nf_, nc_), rl_next, out = vcell(
            particles, fields, currents, rho_layout, bucket, jc, msub, p)

        def sel(new, old):
            a = active.reshape((P,) + (1,) * (new.ndim - 1))
            return jnp.where(a, new, old)

        particles = {k: sel(np_[k], particles[k]) for k in particles}
        fields = {k: sel(nf_[k], fields[k]) for k in fields}
        currents = {k: sel(nc_[k], currents[k]) for k in currents}
        rho_layout = sel(rl_next, rho_layout)

        # hand-off: lane p's pushed bucket at layer j -> lane p+1's input bucket at layer j.
        # Indexed by the receiving lane q = p+1: lane q gets out[q-1] at layer jc[q-1] (a single
        # masked scatter per lane, no per-tick concatenate -> far cheaper to compile/run).
        tgt_layer = jnp.roll(jc, 1)                       # layer q receives at
        valid = (lanes >= 1) & jnp.roll(active, 1)        # only if lane q-1 was active
        vmask = valid.reshape((P,) + (1,) * (beamP["r"].ndim - 2))
        newbeam = {}
        for k in beamP:
            src = jnp.roll(out[k], 1, axis=0)             # src[q] = out[q-1]
            existing = beamP[k][lanes, tgt_layer]
            newval = jnp.where(vmask, src, existing)
            newbeam[k] = beamP[k].at[lanes, tgt_layer].set(newval)
        beamP = newbeam
        return (particles, fields, currents, rho_layout, beamP), out

    carry0 = (particles, fields, currents, rho_layout, beamP)
    _, ys = jax.lax.scan(body, carry0, jnp.arange(n_diag))
    # ys[k]: (n_diag, P, C). Final beam = last lane's push at each layer: ys[(P-1)+j, P-1].
    diag_idx = (P - 1) + jnp.arange(n_xi)
    return {k: ys[k][diag_idx, P - 1] for k in ys}
