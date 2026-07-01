"""
JAX reimplementation of the 2D (cylindrical) plasma field solver.

Prototype for Branch 2 (differentiable LCODE). This ports the numba field solver
(`lcode/plasma/fields.py` + `lcode/plasma/ode.py`) to pure JAX so it is:
  - differentiable  (jax.grad works end-to-end),
  - jit-compilable   (fuses into one XLA program),
  - GPU-ready        (runs on device unchanged).

The tricky parts are the sequential-along-r recurrences:
  - cumulative trapezoid integral  -> expressed with jnp.cumsum,
  - tridiagonal Thomas solve       -> expressed with jax.lax.scan (fwd + back sweep).

Validated against the numba reference in validate_field_solver.py.
"""
import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)  # match numba float64


# ------------------------------------------------------------------ integrals
def cumtrapz(y, dx, mode):
    """Cumulative trapezoidal integral with a zero initial value.

    forward:  out[0]   = 0, out[i+1] = out[i]   + dx*(y[i+1]+y[i])/2
    backward: out[-1]  = 0, out[i]   = out[i+1] - dx*(y[i+1]+y[i])/2
    """
    trap = dx * (y[1:] + y[:-1]) / 2.0          # length n-1
    if mode == "forward":
        return jnp.concatenate([jnp.zeros(1, y.dtype), jnp.cumsum(trap)])
    elif mode == "backward":
        suffix = jnp.cumsum(trap[::-1])[::-1]    # suffix[i] = sum_{k>=i} trap[k]
        return jnp.concatenate([-suffix, jnp.zeros(1, y.dtype)])
    raise ValueError(mode)


# ---------------------------------------------------------------- tridiagonal
def tridiagonal_solve(right_part, previous_factor, r_step, boundaries):
    """JAX port of ode.tridiagonal_solve (Thomas algorithm) via lax.scan.

    Mirrors the numba version index-for-index:
        for i in 1..nr-1:  a=1+0.5/i, b=2+pf[i]*h^2+1/i^2, c=1-0.5/i, d=h^2*rp[i]
                           denom=1/(b-c*alpha[i-1]); alpha[i]=a*denom
                           beta[i]=(c*beta[i-1]-d)*denom
        boundary at nr; then back-substitution beta[i]+=beta[i+1]*alpha[i].
    """
    n = right_part.shape[0]
    nr = n - 1
    h2 = r_step * r_step
    i = jnp.arange(n, dtype=right_part.dtype)     # 0..nr

    a = 1.0 + 0.5 / i
    b = 2.0 + previous_factor * h2 + 1.0 / (i * i)
    c = 1.0 - 0.5 / i
    d = h2 * right_part

    # Forward sweep over interior indices 1..nr-1.
    def fwd(carry, idx):
        alpha_prev, beta_prev = carry
        denom = 1.0 / (b[idx] - c[idx] * alpha_prev)
        alpha_i = a[idx] * denom
        beta_i = (c[idx] * beta_prev - d[idx]) * denom
        return (alpha_i, beta_i), (alpha_i, beta_i)

    idxs = jnp.arange(1, nr)                       # 1..nr-1
    (_, _), (alpha_in, beta_in) = lax.scan(fwd, (0.0, 0.0), idxs)

    # alpha/beta over full length; index 0 = 0, indices 1..nr-1 from scan.
    alpha = jnp.concatenate([jnp.zeros(1, right_part.dtype), alpha_in,
                             jnp.zeros(1, right_part.dtype)])
    beta_fwd = jnp.concatenate([jnp.zeros(1, right_part.dtype), beta_in,
                                jnp.zeros(1, right_part.dtype)])

    # Boundary row at nr.
    cb, bb = boundaries[0], boundaries[1]
    d_nr = r_step * right_part[nr]
    beta_nr = (cb * beta_fwd[nr - 1] - d_nr) / (bb - cb * alpha[nr - 1])
    beta_fwd = beta_fwd.at[nr].set(beta_nr)

    # Back-substitution: i = nr-1 .. 0,  beta[i] += beta[i+1]*alpha[i].
    def back(beta_ip1, idx):
        beta_i = beta_fwd[idx] + beta_ip1 * alpha[idx]
        return beta_i, beta_i

    back_idx = jnp.arange(nr - 1, -1, -1)
    _, beta_rev = lax.scan(back, beta_nr, back_idx)
    beta_low = beta_rev[::-1]                       # beta[0..nr-1]
    return jnp.concatenate([beta_low, beta_nr[None]])


def tridiagonal_solve_neumann_like(right_part, previous_factor, r_step):
    nr = right_part.shape[0] - 1
    boundaries = jnp.array([-(nr - 1) / (nr - 0.5), -nr / (nr - 0.5)],
                           dtype=right_part.dtype)
    return tridiagonal_solve(right_part, previous_factor, r_step, boundaries)


def tridiagonal_solve_dirichlet(right_part, previous_factor, r_step):
    boundaries = jnp.array([0.0, -1.0], dtype=right_part.dtype)
    return tridiagonal_solve(right_part, previous_factor, r_step, boundaries)


# --------------------------------------------------------------- field pieces
def compute_e_r(total_rho, j_r, j_r_prev, e_r_prev, previous_factor, r_step, xi_step_p):
    total_rho_deriv_r = (total_rho[2:] - total_rho[:-2]) / 2.0 / r_step
    j_r_deriv_xi = (j_r - j_r_prev) / xi_step_p
    right_part = j_r_deriv_xi - previous_factor * e_r_prev
    right_part = right_part.at[1:-1].add(total_rho_deriv_r)
    right_part = right_part.at[0].set(0.0)
    right_part = right_part.at[-1].set((total_rho[-2] + total_rho[-1]) / 2.0)
    return tridiagonal_solve_neumann_like(right_part, previous_factor, r_step)


def compute_e_phi(j_phi, j_phi_prev, e_phi_prev, previous_factor, r_step, xi_step_p):
    right_part = (j_phi - j_phi_prev) / xi_step_p - previous_factor * e_phi_prev
    right_part = right_part.at[0].set(0.0)
    right_part = right_part.at[-1].set(0.0)
    return tridiagonal_solve_dirichlet(right_part, previous_factor, r_step)


def compute_e_z(j_r, r_step):
    return cumtrapz(j_r, r_step, "backward")


def compute_b_phi(rho, j_z, e_r, r_step):
    n = rho.shape[0]
    tmp = cumtrapz(jnp.arange(n, dtype=rho.dtype) * (rho - j_z), r_step, "forward")
    tmp = tmp.at[1:].divide(jnp.arange(1, n, dtype=rho.dtype))
    return e_r - tmp


def compute_b_z(j_phi, r_step):
    n = j_phi.shape[0]
    b_z = cumtrapz(-j_phi, r_step, "backward")
    magnetic_flux = jnp.sum((2 * jnp.arange(n, dtype=j_phi.dtype) + 1) * b_z)
    return b_z - magnetic_flux / (n - 1) ** 2


_PREV_MULT = 3.0
_PREV_THRESH = 5.0


def compute_e_r_previous_factor(rho, j_r, j_phi, j_z):
    electron_density = rho - 1.0
    active = electron_density <= -0.1
    ed_safe = jnp.where(active, electron_density, -1.0)
    invgamma_sq = 1.0 - (j_r ** 2 + j_phi ** 2 + j_z ** 2) / ed_safe ** 2
    pos = invgamma_sq > 0.0
    ig_safe = jnp.where(pos, invgamma_sq, 1.0)
    val = _PREV_MULT * (1.0 / jnp.sqrt(ig_safe) - 1.0) - _PREV_THRESH
    prev = jnp.where(active & pos, val, 0.0)
    prev = jnp.where(prev < 0.0, 0.0, prev)
    return prev + 1.0


def compute_fields(fields, fields_prev, rho_beam, currents_prev, currents, xi_step_p, r_step):
    """Port of get_field_computer.compute_fields; returns (new_fields, fields_average).

    `fields`/`fields_prev` are dicts with keys E_r,E_f,E_z,B_f,B_z (1d arrays).
    `currents`/`currents_prev` are dicts with keys rho,j_r,j_f,j_z of shape (2, n).
    """
    # Average fields (input `fields` treated immutably).
    favg = {k: (fields[k] + fields_prev[k]) / 2.0 for k in fields}

    previous_factor = compute_e_r_previous_factor(
        currents["rho"][0], currents["j_r"][0], currents["j_f"][0], currents["j_z"][0])

    rho_sum = currents["rho"].sum(0)
    j_r_sum = currents["j_r"].sum(0)
    j_f_sum = currents["j_f"].sum(0)
    j_z_sum = currents["j_z"].sum(0)
    j_r_prev_sum = currents_prev["j_r"].sum(0)
    j_f_prev_sum = currents_prev["j_f"].sum(0)

    total_rho = rho_sum + rho_beam
    E_r = compute_e_r(total_rho, j_r_sum, j_r_prev_sum, favg["E_r"],
                      previous_factor, r_step, xi_step_p)
    ones = jnp.ones_like(favg["E_f"])
    E_f = compute_e_phi(j_f_sum, j_f_prev_sum, favg["E_f"], ones, r_step, xi_step_p)
    E_z = compute_e_z(j_r_sum, r_step)
    B_f = compute_b_phi(rho_sum, j_z_sum, E_r, r_step)
    B_z = compute_b_z(j_f_sum, r_step)

    new_fields = {"E_r": E_r, "E_f": E_f, "E_z": E_z, "B_f": B_f, "B_z": B_z}
    fields_average = {k: (new_fields[k] + fields_prev[k]) / 2.0 for k in new_fields}
    return new_fields, fields_average
