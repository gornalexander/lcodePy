"""
Faithful JAX port of numba's per-layer beam orchestration (push_solver.step_dt +
BeamCalculator2D), so the JAX beam evolution reproduces numba particle-by-particle.

numba processes one time step as a xi-march (xi_i = 1..n_xi). At each layer it:
  1. pulls the fresh layer-xi_i beam and DEPOSITS it            (deposit_beam_layer, rho_layout carry)
  2. solves the plasma layer                                    (step_dxi)
  3. PUSHES `beam_to_move` = the particles laid out at xi_i-1 (+ any that fell), using the
     fields at (xi_i-1, xi_i)                                   (push_beam_layer)
  4. splits the pushed particles:
       - stable (remaining_steps==0 and xi>=xi_end)  -> done, go to the next time step
       - moving (remaining_steps>0  or  xi<xi_end)   -> "fell": re-pushed at xi_i+1
       - lost                                        -> removed
So deposit happens at layer xi_i while push happens at xi_i for particles laid out at xi_i-1
(a one-layer offset). This module reproduces exactly that, on fixed-size arrays with masks:
each particle carries a `layout_layer` (= floor(-xi/dxi), where it deposits) and a `push_layer`
(= layout_layer+1 initially, incremented every time it falls).

Uses the already bit-exact kernels deposit_beam_layer / step_dxi / push_beam_layer.
"""
import jax
import jax.numpy as jnp

from .march import step_dxi
from . import beam as jbeam

jax.config.update("jax_enable_x64", True)

_STATE = ("r", "xi", "p_z", "p_r", "M", "q_m", "q_norm")


def numba_time_step(plasma0, beam, p, time_step=25.0, subst_energy=2.0):
    """One quasistatic time step, replicating numba step_dt exactly. Returns evolved beam dict."""
    n_xi, nc = p["n_xi"], p["n_cells"]
    h, xi_s = p["r_step"], p["xi_step"]
    mr, B = p["max_radius"], p.get("magnetic_field", 0.0)
    max_sub = p["max_substeps"]
    N = beam["r"].shape[0]

    # per-time-step init: layout layer, first push layer, fresh substepping (dt reset to 0)
    L = jnp.floor(-beam["xi"] / xi_s).astype(jnp.int64)
    dt, rsteps = jbeam.init_substepping(beam["p_z"], beam["q_m"], jnp.zeros(N),
                                        jnp.zeros(N), time_step, subst_energy)
    st = {k: beam[k] for k in _STATE}
    st.update(dt=dt, remaining_steps=rsteps, push_layer=L + 1,
              lost=jnp.zeros(N, bool), done=jnp.zeros(N, bool))
    # Beam density is fixed during the step (deposited from start-of-step positions), exactly as
    # numba/one_time_step: rho_beam_seq[k] drives plasma layer xi_i = k+1.
    rho_beam = jbeam.deposit_beam_full(beam["r"], beam["xi"], beam["q_norm"], n_xi, nc, h, xi_s)

    def body(carry, xi_i):
        (particles, fields, currents), st = carry

        # (1) plasma layer xi_i, driven by rho_beam[xi_i-1];  prev = fields(xi_i-1), new = fields(xi_i)
        dens = rho_beam[xi_i - 1]
        p_new, f_new, c_new = step_dxi(particles, fields, currents, dens, p)

        # (3) push particles whose push_layer == xi_i (fields prev=fields, cur=f_new)
        active = (st["push_layer"] == xi_i) & (~st["done"])
        eff_steps = jnp.where(active, st["remaining_steps"], 0)
        r2, xi2, pz2, pr2, M2, steps2, lost2 = jbeam.push_beam_layer(
            st["r"], st["xi"], st["p_z"], st["p_r"], st["M"], st["q_m"],
            eff_steps, st["dt"], fields, f_new, xi_i, h, xi_s, mr, B, max_sub)

        sel = lambda new, old: jnp.where(active, new, old)
        r, xi, pz, pr, M = sel(r2, st["r"]), sel(xi2, st["xi"]), sel(pz2, st["p_z"]), sel(pr2, st["p_r"]), sel(M2, st["M"])
        rsteps = jnp.where(active, steps2, st["remaining_steps"])
        lost = st["lost"] | (active & lost2)

        # (4) split: stable -> done ; moving -> fell (push at xi_i+1) ; lost -> done(removed)
        xi_end = xi_i * -xi_s
        finished = active & (~lost2) & (steps2 == 0) & (xi2 >= xi_end)
        done = st["done"] | (active & lost2) | finished
        fell = active & (~lost2) & (~finished)
        push_layer = jnp.where(fell, xi_i + 1, st["push_layer"])

        st = {**st, "r": r, "xi": xi, "p_z": pz, "p_r": pr, "M": M,
              "remaining_steps": rsteps, "lost": lost, "done": done, "push_layer": push_layer}
        return ((p_new, f_new, c_new), st), None

    init = ((plasma0["particles"], plasma0["fields"], plasma0["currents"]), st)
    (_, st), _ = jax.lax.scan(body, init, jnp.arange(1, n_xi + 1))
    out = {k: st[k] for k in _STATE}
    out["lost"] = st["lost"]
    return out


def numba_evolve(plasma0, beam, p, n_steps, time_step=25.0, subst_energy=2.0):
    """Evolve the beam for `n_steps` time steps (sequential), faithful to numba."""
    beam = {k: beam[k] for k in _STATE}
    for _ in range(n_steps):
        out = numba_time_step(plasma0, beam, p, time_step, subst_energy)
        beam = {k: out[k] for k in _STATE}
    return beam
