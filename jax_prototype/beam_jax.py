"""
JAX beam deposition for 2D (MB1) — beam macroparticles -> per-layer beam density.

Ports numba `weights.get_layer_weights` / `deposit_particles` +
`BeamCalculator2D.deposit_beam_layer` (bilinear weights in (xi, r) with the per-layer
`rho_layout` carry) to a differentiable JAX scatter.

The deposition is bilinear:
  - in r: split between cells j and j+1 (weights 1-dr, dr),
  - in xi: split between the current layer and the next layer (weights dxi, 1-dxi),
so a particle contributes to `rho_layout` (current) and `next_rho_layout` (carried forward).
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def deposit_beam_layer(rho_layout, r, xi, q_norm, xi_i, r_step, xi_step, n_cells):
    """One beam layer -> (current_density, next_rho_layout). Mirrors numba deposit_beam_layer.

    `rho_layout` is the carry (contributions from the previous layer's `a1_xi` part).
    Returns the normalised density for layer `xi_i` and the carry for `xi_i+1`.
    """
    xi_end = (xi_i + 1) * (-xi_step)
    j = jnp.floor(r / r_step).astype(jnp.int64)
    dxi = -(xi_end - xi) / xi_step
    dr = r / r_step - j
    a0_xi, a1_xi, a0_r, a1_r = dxi, 1.0 - dxi, 1.0 - dr, dr
    a00 = a0_xi * a0_r    # current layer, cell j
    a01 = a0_xi * a1_r    # current layer, cell j+1
    a10 = a1_xi * a0_r    # next layer,    cell j
    a11 = a1_xi * a1_r    # next layer,    cell j+1

    jp = j + 1
    cur = rho_layout.at[j].add(a00 * q_norm).at[jp].add(a01 * q_norm)
    nxt = jnp.zeros(n_cells, dtype=cur.dtype).at[j].add(a10 * q_norm).at[jp].add(a11 * q_norm)

    dens = cur / r_step ** 2
    dens = dens.at[0].multiply(6.0)
    dens = dens.at[1:].divide(jnp.arange(1, n_cells, dtype=dens.dtype))
    return dens, nxt
