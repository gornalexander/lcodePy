"""
JAX reimplementation of the 2D plasma current/charge deposition (M2).

Ports numba `rhoj.deposit_plasma` + `get_rhoj_computer` to a vectorized, differentiable
JAX scatter. The quadratic (C1) shape function makes deposition smooth in the particle
position; the cell index `floor(r/h + 0.5)` is piecewise-constant (zero gradient), which is
the correct PIC behaviour — position gradients flow through the smooth weights, not the bin.

The three per-particle contributions (places -1, 0, +1) reproduce the numba branches:
  - central cell (idx==0): the place=-1 contribution goes to cell 1 with antisymmetric
    j_r/j_f sign (the "deposited antisymmetrically" special case);
  - last cell (idx==grid-1): the place=+1 contribution is clamped to cell grid-2.
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def cell_volume(r_step, particles_per_cell, n_cells):
    """Identical to numba rhoj._cell_volume (a constant; not differentiated)."""
    cpp = 1.0 / particles_per_cell
    vol = 2 * np.pi * r_step ** 2 * np.arange(n_cells)
    vol[0] = np.pi * r_step ** 2 * (13 + 2 * cpp ** 2) / 32
    vol[1] *= (193 + 2 * cpp ** 2) / 192
    vol[-1] = np.pi * r_step ** 2 * (n_cells - 45 / 32)
    return jnp.asarray(vol)


def deposit_plasma(r, p_r, p_f, p_z, q, m, n_cells, r_step):
    """Deposit one particle species onto (rho, j_r, j_f, j_z) grids (row-0, pre-volume)."""
    gamma_mass = jnp.sqrt(m ** 2 + p_r ** 2 + p_f ** 2 + p_z ** 2)
    dq = q / (gamma_mass - p_z)
    drho = dq * gamma_mass
    dj_r = dq * p_r
    dj_f = dq * p_f
    dj_z = dq * p_z

    r_norm = r / r_step
    idx = jnp.floor(r_norm + 0.5).astype(jnp.int64)
    rloc = r_norm - idx

    w_m1 = (rloc - 0.5) ** 2 / 2
    w_0 = 0.75 - rloc ** 2
    w_p1 = (rloc + 0.5) ** 2 / 2

    cell_m1 = jnp.where(idx == 0, 1, idx - 1)
    cell_0 = idx
    cell_p1 = jnp.where(idx == n_cells - 1, n_cells - 2, idx + 1)
    sign_jrf_m1 = jnp.where(idx == 0, -1.0, 1.0)  # antisymmetric central deposition

    cells = jnp.concatenate([cell_m1, cell_0, cell_p1])

    def scatter(vals):
        return jnp.zeros(n_cells, dtype=r.dtype).at[cells].add(vals)

    rho = scatter(jnp.concatenate([drho * w_m1, drho * w_0, drho * w_p1]))
    j_z = scatter(jnp.concatenate([dj_z * w_m1, dj_z * w_0, dj_z * w_p1]))
    j_r = scatter(jnp.concatenate([dj_r * w_m1 * sign_jrf_m1, dj_r * w_0, dj_r * w_p1]))
    j_f = scatter(jnp.concatenate([dj_f * w_m1 * sign_jrf_m1, dj_f * w_0, dj_f * w_p1]))

    # Symmetry / boundary conditions (as in numba deposit_plasma).
    j_r = j_r.at[0].set(0.0).at[n_cells - 1].set(0.0)
    j_f = j_f.at[0].set(0.0).at[n_cells - 1].set(0.0)
    j_z = j_z.at[n_cells - 1].set(0.0)
    return rho, j_r, j_f, j_z


def compute_rhoj(particles, n_cells, r_step, vol, ni):
    """Port of get_rhoj_computer.compute_rhoj for a single (electron) species + background ions.

    `particles` is a dict with r, p_r, p_f, p_z, q, m. Returns dict of (2, n) arrays.
    """
    rho0, j_r0, j_f0, j_z0 = deposit_plasma(
        particles["r"], particles["p_r"], particles["p_f"], particles["p_z"],
        particles["q"], particles["m"], n_cells, r_step)

    zeros = jnp.zeros(n_cells, dtype=rho0.dtype)
    rho = jnp.stack([rho0 / vol, jnp.full(n_cells, ni, dtype=rho0.dtype)])
    j_r = jnp.stack([j_r0 / vol, zeros])
    j_f = jnp.stack([j_f0 / vol, zeros])
    j_z = jnp.stack([j_z0 / vol, zeros])
    return {"rho": rho, "j_r": j_r, "j_f": j_f, "j_z": j_z}
