"""
MB3/MB4 — self-consistent time dynamics: the beam evolving in its own wake.

Quasistatic time step (the beam is rigid during the plasma calc):
  1. deposit the whole beam -> rho_beam[n_xi, n_r]           (beam_jax.deposit_beam_full)
  2. plasma xi-march -> wake field history F[n_xi, n_r]      (march_jax.march_with_fields)
  3. push the whole beam over dt in that wake                (push_beam_history below)

`push_beam_history` samples the 2D field history bilinearly at each particle's (xi, r);
this is equivalent to numba's two-adjacent-xi-layer interpolation, but done for the whole
beam at once. (numba's exact per-layer bookkeeping with 'fell' re-processing is an
implementation-phase detail; here the physics — beam drives wake, wake pushes beam — is
captured and fully differentiable.)
"""
import jax
import jax.numpy as jnp

from . import beam as jbeam
from . import march as jmarch

jax.config.update("jax_enable_x64", True)


def _sample_history(x, y, xi, F, r_step, xi_step):
    """Bilinear sample of a field history F[n_xi, n_cells] at (r=hypot(x,y), xi)."""
    n_xi, n = F.shape
    r = jnp.sqrt(x ** 2 + y ** 2)
    xin = -xi / xi_step
    L = jnp.clip(jnp.floor(xin).astype(jnp.int64), 0, n_xi - 2)
    fxi = xin - jnp.floor(xin)
    j = jnp.clip(jnp.floor(r / r_step).astype(jnp.int64), 0, n - 2)
    dr = r / r_step - jnp.floor(r / r_step)
    Lp, jp = L + 1, j + 1
    return ((1 - fxi) * ((1 - dr) * F[L, j] + dr * F[L, jp])
            + fxi * ((1 - dr) * F[Lp, j] + dr * F[Lp, jp]))


def push_beam_history(r, xi, p_z, p_r, M, q_m, remaining_steps, dt, hist,
                      r_step, xi_step, max_radius, magnetic_field, max_substeps):
    """Whole-beam push in the stored wake-field history (differentiable)."""
    lost_boundary = max(0.9 * max_radius, max_radius - 1)
    lb2 = lost_boundary ** 2
    inv_qm2 = (1.0 / q_m) ** 2
    sgn = jnp.sign(q_m)
    has_B = magnetic_field != 0
    steps = remaining_steps.astype(jnp.float64)
    lost = jnp.zeros_like(r, dtype=bool)
    done = jnp.zeros_like(r, dtype=bool)

    def body(s, _):
        r, xi, p_r, p_z, M, steps, lost, done = s
        active = (~done) & (steps > 0)
        x = r; px = p_r; py = M / r; pz = p_z
        gamma = jnp.sqrt(inv_qm2 + px ** 2 + py ** 2 + pz ** 2)
        xh = x + dt / 2 * px / gamma
        yh = dt / 2 * py / gamma
        xih = xi + dt / 2 * pz / gamma - dt / 2
        lost_half = (xh ** 2 + yh ** 2) >= lb2

        ex = _sample_history(xh, yh, xih, hist["E_r"], r_step, xi_step)
        ey = _sample_history(xh, yh, xih, hist["E_f"], r_step, xi_step)
        ez = _sample_history(xh, yh, xih, hist["E_z"], r_step, xi_step)
        by = _sample_history(xh, yh, xih, hist["B_f"], r_step, xi_step)
        bz = _sample_history(xh, yh, xih, hist["B_z"], r_step, xi_step) + magnetic_field
        bx = -ey
        ux, uy, uz = px / gamma, py / gamma, pz / gamma
        cx = uy * bz - uz * by; cy = uz * bx - ux * bz; cz = ux * by - uy * bx
        phx = px + dt / 2 * sgn * (ex + cx)
        phy = py + dt / 2 * sgn * (ey + cy)
        phz = pz + dt / 2 * sgn * (ez + cz)
        gh = jnp.sqrt(inv_qm2 + phx ** 2 + phy ** 2 + phz ** 2)
        xn = x + dt * phx / gh
        yn = dt * phy / gh
        xin = xi + dt * phz / gh - dt
        pxn = 2 * phx - px; pyn = 2 * phy - py; pzn = 2 * phz - pz
        rn = jnp.sqrt(xn ** 2 + yn ** 2)
        p_r_n = (xn * pxn + yn * pyn) / rn
        M_n = jnp.where(has_B, xn * pyn - yn * pxn, M)
        lost_full = (xn ** 2 + yn ** 2) >= lb2

        do_full = active & (~lost_half)
        break_half = active & lost_half
        r = jnp.where(do_full, rn, r)
        xi = jnp.where(do_full, xin, xi)
        p_r = jnp.where(do_full, p_r_n, p_r)
        p_z = jnp.where(do_full, pzn, p_z)
        M = jnp.where(do_full, M_n, M)
        steps = jnp.where(do_full, steps - 1, steps)
        lost = lost | break_half | (do_full & lost_full)
        done = done | break_half | (do_full & lost_full)
        return (r, xi, p_r, p_z, M, steps, lost, done), None

    init = (r, xi, p_r, p_z, M, steps, lost, done)
    (r, xi, p_r, p_z, M, steps, lost, done), _ = jax.lax.scan(body, init, None, length=max_substeps)
    return r, xi, p_z, p_r, M, lost


def one_time_step(beam, plasma, p, dt, max_substeps):
    """One quasistatic time step. `beam` and `plasma` are dicts; returns (new_beam, ez_axis)."""
    n_xi = p["n_xi"]
    rho_beam = jbeam.deposit_beam_full(beam["r"], beam["xi"], beam["q_norm"],
                                       n_xi, p["n_cells"], p["r_step"], p["xi_step"])
    hist, _ = jmarch.march_with_fields(plasma["particles"], plasma["fields"],
                                       plasma["currents"], rho_beam, p)
    ez_axis = hist["E_z"][:, 0]
    r, xi, p_z, p_r, M, lost = push_beam_history(
        beam["r"], beam["xi"], beam["p_z"], beam["p_r"], beam["M"], beam["q_m"],
        beam["remaining_steps"], dt, hist, p["r_step"], p["xi_step"],
        p["max_radius"], p.get("magnetic_field", 0.0), max_substeps)
    new_beam = dict(beam, r=r, xi=xi, p_z=p_z, p_r=p_r, M=M)
    return new_beam, ez_axis, lost
