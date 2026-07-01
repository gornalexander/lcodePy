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


# ----------------------------------------------------------------- beam push (MB2)
def init_substepping(p_z, q_m, dt, remaining_steps, time_step, substepping_energy):
    """Port of numba init_substepping / beam_substepping_step.

    For newly-entered particles (dt==0), choose the largest sub-step dt = 2^-n such that
    2^-n <= sqrt(gamma/E_ss), and remaining_steps = 1/dt. Particles that already have dt
    (came from a previous time step) are left unchanged.
    """
    gamma_mass = jnp.sqrt((1.0 / q_m) ** 2 + p_z ** 2)
    max_dt = jnp.sqrt(gamma_mass / substepping_energy)
    n = jnp.maximum(0.0, jnp.ceil(-jnp.log2(max_dt)))     # dt = 2^-n <= max_dt, dt<=1
    dt_sub = 2.0 ** (-n)
    steps_sub = jnp.round(1.0 / dt_sub)
    mask = dt == 0
    dt_out = jnp.where(mask, dt_sub * time_step, dt)
    steps_out = jnp.where(mask, steps_sub, remaining_steps.astype(dt_out.dtype))
    return dt_out, steps_out



def _interp_field(x, y, xi, v_prev, v_cur, xi_end, r_step, xi_step):
    """Bilinear interpolation of a field between two time levels (prev=k, cur=k+1).

    Mirrors weights.particle_fields/interpolate_particle: a0_xi weights the previous time
    level, a1_xi the current one; a0_r/a1_r split between radial cells j and j+1.
    """
    n = v_prev.shape[0]
    r = jnp.sqrt(x ** 2 + y ** 2)
    j = jnp.clip(jnp.floor(r / r_step).astype(jnp.int64), 0, n - 2)
    jp = j + 1
    dxi = -(xi_end - xi) / xi_step
    dr = r / r_step - j
    a00 = dxi * (1 - dr); a01 = dxi * dr
    a10 = (1 - dxi) * (1 - dr); a11 = (1 - dxi) * dr
    return a00 * v_prev[j] + a01 * v_prev[jp] + a10 * v_cur[j] + a11 * v_cur[jp]


def _particle_fields(x, y, xi, prev, cur, xi_end, r_step, xi_step):
    e_x = _interp_field(x, y, xi, prev["E_r"], cur["E_r"], xi_end, r_step, xi_step)
    e_y = _interp_field(x, y, xi, prev["E_f"], cur["E_f"], xi_end, r_step, xi_step)
    e_z = _interp_field(x, y, xi, prev["E_z"], cur["E_z"], xi_end, r_step, xi_step)
    b_y = _interp_field(x, y, xi, prev["B_f"], cur["B_f"], xi_end, r_step, xi_step)
    b_z = _interp_field(x, y, xi, prev["B_z"], cur["B_z"], xi_end, r_step, xi_step)
    b_x = -e_y   # symmetry
    return (e_x, e_y, e_z), (b_x, b_y, b_z)


def push_beam_layer(r, xi, p_z, p_r, M, q_m, remaining_steps, dt,
                    prev_fields, cur_fields, xi_i, r_step, xi_step,
                    max_radius, magnetic_field, max_substeps):
    """Vectorized, differentiable port of numba `push_particles` (2D beam pusher).

    Each particle takes up to `remaining_steps` sub-steps of size `dt`; the adaptive
    per-particle `while steps>0` loop is a bounded `lax.scan` of length `max_substeps`
    with masking. Returns updated (r, xi, p_z, p_r, M, remaining_steps, lost).
    """
    xi_end = xi_i * -xi_step
    lost_boundary = max(0.9 * max_radius, max_radius - 1)
    lb2 = lost_boundary ** 2
    inv_qm2 = (1.0 / q_m) ** 2
    sgn = jnp.sign(q_m)
    has_B = magnetic_field != 0

    # State carried in CYLINDRICAL form (r, xi, p_r, p_z, M); each sub-step reconstructs the
    # local cartesian frame with y=0 (beam_to_vec), matching numba's per-step round-trip.
    steps = remaining_steps.astype(jnp.float64)
    lost = jnp.zeros_like(r, dtype=bool)
    done = jnp.zeros_like(r, dtype=bool)

    def body(s, _):
        r, xi, p_r, p_z, M, steps, lost, done = s
        active = (~done) & (steps > 0)

        # reconstruct cartesian (x along r, y=0)
        x = r
        px = p_r
        py = M / r
        pz = p_z
        gamma = jnp.sqrt(inv_qm2 + px ** 2 + py ** 2 + pz ** 2)
        xh = x + dt / 2 * px / gamma
        yh = dt / 2 * py / gamma
        xih = xi + dt / 2 * pz / gamma - dt / 2

        not_in = xih < xi_end
        lost_half = (xh ** 2 + yh ** 2) >= lb2

        (ex, ey, ez), (bx, by, bz) = _particle_fields(xh, yh, xih, prev_fields, cur_fields,
                                                       xi_end, r_step, xi_step)
        bz = bz + magnetic_field
        ux, uy, uz = px / gamma, py / gamma, pz / gamma
        cx = uy * bz - uz * by
        cy = uz * bx - ux * bz
        cz = ux * by - uy * bx
        phx = px + dt / 2 * sgn * (ex + cx)
        phy = py + dt / 2 * sgn * (ey + cy)
        phz = pz + dt / 2 * sgn * (ez + cz)
        gh = jnp.sqrt(inv_qm2 + phx ** 2 + phy ** 2 + phz ** 2)
        xn = x + dt * phx / gh
        yn = dt * phy / gh
        xin = xi + dt * phz / gh - dt

        pxn = 2 * phx - px
        pyn = 2 * phy - py
        pzn = 2 * phz - pz
        rn = jnp.sqrt(xn ** 2 + yn ** 2)
        p_r_n = (xn * pxn + yn * pyn) / rn
        M_n = jnp.where(has_B, xn * pyn - yn * pxn, M)
        lost_full = (xn ** 2 + yn ** 2) >= lb2

        do_full = active & (~not_in) & (~lost_half)
        break_notin = active & not_in
        break_half = active & (~not_in) & lost_half

        r = jnp.where(do_full, rn, r)
        xi = jnp.where(do_full, xin, xi)
        p_r = jnp.where(do_full, p_r_n, p_r)
        p_z = jnp.where(do_full, pzn, p_z)
        M = jnp.where(do_full, M_n, M)
        steps = jnp.where(do_full, steps - 1, steps)
        lost = lost | break_half | (do_full & lost_full)
        done = done | break_notin | break_half | (do_full & lost_full)
        return (r, xi, p_r, p_z, M, steps, lost, done), None

    init = (r, xi, p_r, p_z, M, steps, lost, done)
    (r, xi, p_r, p_z, M, steps, lost, done), _ = jax.lax.scan(body, init, None, length=max_substeps)
    return r, xi, p_z, p_r, M, steps.astype(remaining_steps.dtype), lost

