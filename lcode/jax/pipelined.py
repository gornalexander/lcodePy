"""
Batched-wavefront (pipelined) xi-march for the differentiable JAX backend.

Motivation
----------
A single 2D LCODE run is two nested *sequential* recurrences — the xi-march (layer j needs
layer j-1) wrapped in the time loop (step t+1 needs step t). On a GPU that is latency-bound:
each layer solve is a tiny `lax.scan` on ~one thread, so a single trajectory is ~8x slower
than numba on CPU.

But the coupling is per-*layer*, not per-*step* (see `lcode/mpi/transport.py`, which already
pipelines this across MPI ranks: rank R streams each finished xi-layer to rank R+1, so
consecutive time steps run concurrently as a diagonal wavefront). This module realises the
*same* wavefront on-device: instead of one MPI rank per pipeline stage, we batch `depth`
marches and advance them as a staggered wavefront, so each `lax.scan` tick does `depth`
layer-solves at once — turning the sequential march into wide, GPU-friendly work.

`wavefront_marches` computes `depth` marches (each producing a full field history) in a single
staggered scan of `n_xi + depth - 1` ticks. It is *numerically identical* to running the
`depth` marches sequentially (see tests), but on a GPU its throughput approaches the batched
ceiling because every tick is `depth`-wide.

Which marches fill the batch is a data-routing question:
  - independent (beam, plasma) configs  -> parameter scans / ensembles (works today);
  - causally-coupled consecutive time steps of ONE trajectory -> the time-step pipeline
    (needs per-layer beam hand-off à la `lcode/mpi`; the throughput ceiling is identical and
    is what this engine measures).
"""
from functools import partial
import jax
import jax.numpy as jnp

from . import march as jmarch

jax.config.update("jax_enable_x64", True)


def sequential_marches(rho_beam_batch, plasma0, p):
    """Reference: run `depth` marches one after another (the current sequential approach).

    rho_beam_batch : (depth, n_xi, n_cells)
    plasma0        : {"particles","fields","currents"} initial (unbatched) plasma state
    returns field history dict, each entry (depth, n_xi, n_cells).
    """
    depth = rho_beam_batch.shape[0]
    outs = []
    for i in range(depth):
        fh, _ = jmarch.march_with_fields(
            plasma0["particles"], plasma0["fields"], plasma0["currents"],
            rho_beam_batch[i], p)
        outs.append(fh)
    return {k: jnp.stack([o[k] for o in outs]) for k in outs[0]}


def wavefront_marches(rho_beam_batch, plasma0, p):
    """Compute `depth` marches concurrently as a staggered wavefront (one scan).

    Pipeline p processes layer (tick - p) at each tick, so at any tick up to `depth` distinct
    marches are each on a different layer and are advanced together (batched). Finished /
    not-yet-started pipelines are masked (their carry is frozen) — that masked work is exactly
    the pipeline fill/drain overhead, so this timing is faithful to a real pipeline.

    rho_beam_batch : (depth, n_xi, n_cells)
    returns field history dict, each entry (depth, n_xi, n_cells), identical to
    `sequential_marches` up to floating point.
    """
    depth = rho_beam_batch.shape[0]
    n_xi = p["n_xi"]
    n_ticks = n_xi + depth - 1
    pidx = jnp.arange(depth)

    bc = lambda x: jnp.broadcast_to(x, (depth,) + x.shape)
    particles0 = {k: bc(v) for k, v in plasma0["particles"].items()}
    fields0 = {k: bc(v) for k, v in plasma0["fields"].items()}
    currents0 = {k: bc(v) for k, v in plasma0["currents"].items()}

    vstep = jax.vmap(lambda pp, ff, cc, rb: jmarch.step_dxi(pp, ff, cc, rb, p))

    def body(carry, k):
        particles, fields, currents = carry
        idx = k - pidx                               # layer each pipeline works on
        active = (idx >= 0) & (idx < n_xi)
        rb = rho_beam_batch[pidx, jnp.clip(idx, 0, n_xi - 1)]     # (depth, n_cells)
        np_, nf_, nc_ = vstep(particles, fields, currents, rb)

        def sel(new, old):
            a = active.reshape((depth,) + (1,) * (new.ndim - 1))
            return jnp.where(a, new, old)

        particles = {kk: sel(np_[kk], particles[kk]) for kk in particles}
        fields = {kk: sel(nf_[kk], fields[kk]) for kk in fields}
        currents = {kk: sel(nc_[kk], currents[kk]) for kk in currents}
        return (particles, fields, currents), fields

    _, hist = jax.lax.scan(body, (particles0, fields0, currents0), jnp.arange(n_ticks))

    # hist[key]: (n_ticks, depth, n_cells). Gather F[p, j] = hist[j + p, p].
    J = jnp.arange(n_xi)
    ticks = J[None, :] + pidx[:, None]               # (depth, n_xi)
    prow = pidx[:, None]                             # (depth, 1)
    return {k: hist[k][ticks, prow] for k in hist}    # (depth, n_xi, n_cells)
