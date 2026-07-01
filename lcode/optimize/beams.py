"""Differentiable beam builders for optimisation.

`transverse_gaussian` takes a *template* (the longitudinal beam structure — xi, charge,
energy — e.g. from a numba-generated driver) and returns a `beam_of(params)` closure that
regenerates the **transverse** phase space from design parameters (size, divergence) using
fixed random samples (the reparameterisation trick), so the beam is a smooth differentiable
function of the parameters.
"""
import numpy as np
import jax.numpy as jnp


def transverse_gaussian(template, seed=0):
    """Return `beam_of(params)` where params = {'log_sigma_r':..., 'log_alpha':...}.

    The transverse size and divergence are parameterised in **log space** — they are
    positive and typically span orders of magnitude (size ~1, divergence ~1e-4), so
    optimising `log` keeps the two directions comparably scaled and the optimiser
    well-conditioned. Use `size_params(sigma_r, alpha)` to build the log-params from
    physical values.

    template : dict with keys 'xi', 'p_z', 'q_m', 'q_norm' (arrays of equal length).
    """
    n = np.asarray(template["xi"]).shape[0]
    rng = np.random.default_rng(seed)
    rho = jnp.asarray(np.sqrt(-2 * np.log(rng.uniform(1e-9, 1.0, n))))   # Rayleigh (2D)
    nu = jnp.asarray(rng.standard_normal(n))
    XI = jnp.asarray(template["xi"]); PZ = jnp.asarray(template["p_z"])
    QM = jnp.asarray(template["q_m"]); QN = jnp.asarray(template["q_norm"])

    def beam_of(params):
        sr = jnp.exp(params["log_sigma_r"]); al = jnp.exp(params["log_alpha"])
        return {"r": sr * rho, "xi": XI, "p_z": PZ, "p_r": al * PZ * nu,
                "M": jnp.zeros(n), "q_m": QM, "q_norm": QN}
    return beam_of


def size_params(sigma_r, alpha):
    """Build the (log-space) params dict for `transverse_gaussian` from physical values."""
    return {"log_sigma_r": float(np.log(sigma_r)), "log_alpha": float(np.log(alpha))}
