"""Core of the differentiable optimisation interface: SimContext, Trajectory, Optimizer."""
import numpy as np
import jax
import jax.numpy as jnp

import lcode
import lcode.plasma as nplasma
from lcode import jax as lj

jax.config.update("jax_enable_x64", True)


class Trajectory:
    """Result of evolving a beam for several time steps; differentiable observables."""

    def __init__(self, states, wakes):
        self.states = states          # list of beam dicts (jax arrays)
        self.wakes = wakes            # list of on-axis E_z(xi)
        self.final = states[-1]

    def rms_size(self):               # per-step rms transverse size
        return jnp.stack([jnp.sqrt(jnp.mean(s["r"] ** 2)) for s in self.states])

    def mean_energy(self):            # per-step mean p_z
        return jnp.stack([jnp.mean(s["p_z"]) for s in self.states])

    def energy_spread(self):          # per-step rms energy spread
        return jnp.stack([jnp.std(s["p_z"]) / (jnp.mean(s["p_z"]) + 1e-30) for s in self.states])

    def wake(self, step=-1):
        return self.wakes[step]


def _plasma_context(config):
    R = config["transverse-step"]; WW = config["window-width"]
    XI = config["xi-step"]; WL = config["window-length"]; PPC = config["plasma-particles-per-cell"]
    NC = int(WW / R) + 1; NXI = int(WL / XI)
    cfgd = {"geometry": "2d", "transverse-step": R, "window-width": WW,
            "plasma-particles-per-cell": PPC, "ion-model": "background"}
    el = nplasma.init_plasma_2d(lcode.config.config.Config(cfgd))[1]["electrons"]
    part = {k: jnp.asarray(getattr(el, k)) for k in ("r", "p_r", "p_f", "p_z", "q", "m")}
    vol = lj.cell_volume(R, PPC, NC)
    plasma = {"particles": part,
              "fields": {k: jnp.zeros(NC) for k in lj.FIELD_KEYS},
              "currents": lj.compute_rhoj(part, NC, R, vol, 1.0)}
    P = {"n_xi": NXI, "n_cells": NC, "r_step": R, "xi_step": XI, "vol": vol, "ni": 1.0,
         "max_radius": WW, "corrector_steps": config.get("correctotransverse-steps", 2),
         "n_attempts": config.get("beam-substep-attempts", 4), "checkpoint": True,
         "magnetic_field": config.get("magnetic-field", 0.0)}
    return plasma, P


class OptResult:
    def __init__(self, params, history, opt):
        self.params = params
        self.history = history
        self._opt = opt

    def trajectory(self):
        return self._opt.evolve(self.params)


class Optimizer:
    """Gradient-based design optimisation through the differentiable JAX time loop.

    Parameters
    ----------
    config : dict
        Grid/plasma settings (window-width, transverse-step, window-length, xi-step,
        plasma-particles-per-cell; optional correctotransverse-steps, magnetic-field).
    beam_of : callable
        `params -> beam dict` (differentiable) building the initial beam from design params.
        `params` may be any JAX pytree (e.g. a dict of scalars/arrays).
    objective : callable
        `Trajectory -> scalar` loss to minimise (see `lcode.optimize.objectives`).
    n_steps : int
        Number of time steps to evolve per evaluation.
    """

    def __init__(self, config, beam_of, objective, n_steps=6,
                 time_step=25.0, subst_energy=2.0):
        self.plasma, self.P = _plasma_context(config)
        self.beam_of = beam_of
        self.objective = objective
        self.n_steps = n_steps
        self.time_step = time_step
        self.subst_energy = subst_energy
        self._max_sub = 1

    def _substep(self, beam):
        return lj.init_substepping(beam["p_z"], beam["q_m"],
                                   jnp.zeros_like(beam["p_z"]), jnp.zeros_like(beam["p_z"]),
                                   self.time_step, self.subst_energy)

    def evolve(self, params):
        beam = self.beam_of(params)
        dt, steps = self._substep(beam)
        beam = {**beam, "remaining_steps": steps}
        states, wakes = [], []
        for _ in range(self.n_steps):
            beam, ez, lost = lj.one_time_step(beam, self.plasma, self.P, dt, self._max_sub)
            states.append(beam); wakes.append(ez)
        return Trajectory(states, wakes)

    def loss(self, params):
        return self.objective(self.evolve(params))

    def minimize(self, init_params, iters=40, lr=0.1, verbose=True):
        # fix the bounded-substep length from the initial beam (static for jit)
        _, s0 = self._substep(self.beam_of(init_params))
        self._max_sub = int(np.asarray(s0).max())
        value_and_grad = jax.jit(jax.value_and_grad(self.loss))

        params = init_params
        m = jax.tree_util.tree_map(jnp.zeros_like, params)
        v = jax.tree_util.tree_map(jnp.zeros_like, params)
        b1, b2, eps = 0.9, 0.999, 1e-8
        history = []
        for it in range(1, iters + 1):
            L, g = value_and_grad(params)
            history.append(float(L))
            m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, m, g)
            v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, v, g)
            mh = jax.tree_util.tree_map(lambda m: m / (1 - b1 ** it), m)
            vh = jax.tree_util.tree_map(lambda v: v / (1 - b2 ** it), v)
            params = jax.tree_util.tree_map(
                lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mh, vh)
            if verbose and (it == 1 or it % max(1, iters // 8) == 0):
                print(f"iter {it:3d}: loss = {float(L):.4e}")
        return OptResult(params, history, self)
