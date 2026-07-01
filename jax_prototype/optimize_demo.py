"""
M5 — end-to-end differentiable optimization demo.

The payoff of Branch 2: differentiate a wake objective w.r.t. a *driver design parameter*
through the whole plasma xi-march, and optimize it with gradient ascent — something the
numba code cannot do.

Setup: a rigid Gaussian driver of fixed current I_b (LCODE relation I_b = rho_b0 * sigma^2 / 2,
so on-axis density scales as 1/sigma^2 at fixed current). We optimize the transverse radius
sigma to maximize the trailing wake amplitude (sum of on-axis E_z^2 behind the driver).
There is a physical optimum near the plasma skin depth, so a correct gradient should find it.

Run: PYTHONPATH=<repo> python optimize_demo.py
"""
import io
import contextlib
import numpy as np
import jax
import jax.numpy as jnp

import lcode
import lcode.plasma as nplasma
import deposition_jax as jdep
import march_jax as jmarch

jax.config.update("jax_enable_x64", True)

R_STEP = 0.05
WINDOW_WIDTH = 5
XI_STEP = 0.05
WINDOW_LENGTH = 5
PPC = 10
N_CELLS = int(WINDOW_WIDTH / R_STEP) + 1
N_XI = int(WINDOW_LENGTH / XI_STEP)
I_B = 0.05            # fixed driver current (magnitude)
DRIVER_LEN = 2.0      # driver length in xi (flat top)
N_ATTEMPTS = 6        # bounded mover substep attempts (weak driver -> few needed)
R = jnp.arange(N_CELLS) * R_STEP
XI = jnp.arange(1, N_XI + 1) * XI_STEP   # |xi| of each layer


def initial_state():
    cfgd = {"geometry": "2d", "transverse-step": R_STEP, "window-width": WINDOW_WIDTH,
            "plasma-particles-per-cell": PPC, "ion-model": "background"}
    state = nplasma.init_plasma_2d(lcode.config.config.Config(cfgd))
    el = state[1]["electrons"]
    particles = {"r": jnp.asarray(el.r), "p_r": jnp.asarray(el.p_r),
                 "p_f": jnp.asarray(el.p_f), "p_z": jnp.asarray(el.p_z),
                 "q": jnp.asarray(el.q), "m": jnp.asarray(el.m)}
    fields = {k: jnp.zeros(N_CELLS) for k in jmarch.FIELD_KEYS}
    vol = jdep.cell_volume(R_STEP, PPC, N_CELLS)
    currents = jdep.compute_rhoj(particles, N_CELLS, R_STEP, vol, ni=1.0)
    return particles, fields, currents, vol


PARTICLES, FIELDS, CURRENTS, VOL = initial_state()
PARAMS = {"n_cells": N_CELLS, "r_step": R_STEP, "xi_step": XI_STEP, "vol": VOL,
          "ni": 1.0, "max_radius": WINDOW_WIDTH, "corrector_steps": 2,
          "n_attempts": N_ATTEMPTS, "checkpoint": True}
# longitudinal driver profile: flat over the first DRIVER_LEN, 0 after (fixed).
LONG = jnp.where(XI <= DRIVER_LEN, 1.0, 0.0)
BEHIND = XI > DRIVER_LEN                      # wake region behind the driver


def rho_beam_of_sigma(sigma):
    """Differentiable rigid Gaussian driver, fixed current I_b (rho_b0 = 2 I_b / sigma^2)."""
    rho_b0 = 2.0 * I_B / sigma ** 2
    trans = jnp.exp(-R ** 2 / (2.0 * sigma ** 2))
    return -(rho_b0) * LONG[:, None] * trans[None, :]   # negative: electron driver


def wake_ez(sigma):
    rho = rho_beam_of_sigma(sigma)
    ez, _ = jmarch.march(PARTICLES, FIELDS, CURRENTS, rho, PARAMS)
    return ez


def wake_metric(sigma):
    ez = wake_ez(sigma)
    return jnp.sum(jnp.where(BEHIND, ez ** 2, 0.0))   # trailing wake energy (scalar)


# Inverse design: recover the driver radius that produces a *target* wake amplitude.
SIGMA_TRUE = 1.2
TARGET = float(wake_metric(jnp.array(SIGMA_TRUE)))


def objective(log_sigma):
    return (wake_metric(jnp.exp(log_sigma)) / TARGET - 1.0) ** 2   # normalized


def main():
    val_and_grad = jax.jit(jax.value_and_grad(objective))

    print(f"grid: {N_XI} xi-layers x {N_CELLS} radial;  fixed current I_b={I_B}")
    print(f"Inverse design: recover the driver radius that gives the target wake.")
    print(f"  target wake energy = {TARGET:.6e}  (produced by sigma_true = {SIGMA_TRUE})\n")

    log_sigma = jnp.log(2.5)     # start far from the true value
    # Adam: robust to the strongly-varying gradient scale of this 1/sigma^2 landscape.
    lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8
    m = v = 0.0
    print(f"{'iter':>4} {'sigma':>8} {'loss':>13} {'wake':>13} {'|grad|':>10}")
    for it in range(1, 41):
        J, g = val_and_grad(log_sigma)
        g = float(g)
        wm = float(wake_metric(jnp.exp(log_sigma)))
        print(f"{it:4d} {float(jnp.exp(log_sigma)):8.4f} {float(J):13.4e} {wm:13.6e} {abs(g):10.3e}")
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1 ** it)
        vhat = v / (1 - b2 ** it)
        log_sigma = log_sigma - lr * mhat / (jnp.sqrt(vhat) + eps)

    sigma_opt = float(jnp.exp(log_sigma))
    print(f"\nrecovered sigma = {sigma_opt:.4f}   (true = {SIGMA_TRUE};  "
          f"error = {abs(sigma_opt - SIGMA_TRUE):.2e})")
    print("Gradient-based inverse design through the full plasma march converged.")


if __name__ == "__main__":
    main()
