"""
JAX reimplementation of the 2D plasma particle push (M3).

Ports numba `move._move_one_particle` (single Lorentz push, all branches -> jnp.where) and
`_move_particles_with_substepping`. The adaptive substepping (a data-dependent while_loop in
numba) is reformulated as a **bounded lax.scan with masking**, because reverse-mode grad does
not work through lax.while_loop — this is the key control-flow finding for Branch 2.

Noise reductor is omitted (disabled in the reference cases).
Everything is fully vectorized over particles (no python loop, no vmap needed).
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

N_SUBSTEP_ATTEMPTS = 64  # bounded replacement for the adaptive while_loop


def _interpolate_fields(cell_idx, local_coord, E_r, E_f, E_z, B_f, B_z):
    size = E_r.shape[0]
    jc = jnp.clip(cell_idx, 0, size - 1)
    jp = jnp.where(cell_idx == size - 1, cell_idx - 1, cell_idx + 1)
    jm = jnp.where(cell_idx == 0, cell_idx + 1, cell_idx - 1)
    jm = jnp.clip(jm, 0, size - 1)
    jp = jnp.clip(jp, 0, size - 1)

    a_1 = 0.5 * (local_coord - 0.5) ** 2
    a_2 = 0.75 - local_coord ** 2
    a_3 = 0.5 * (local_coord + 0.5) ** 2
    is0 = cell_idx == 0

    def interp(F):
        three = a_1 * F[jm] + a_2 * F[jc] + a_3 * F[jp]
        zero = (a_3 - a_1) * F[jm] + a_2 * F[jc]
        return jnp.where(is0, zero, three)

    e_x = interp(E_r); e_y = interp(E_f); b_y = interp(B_f)
    e_z = a_1 * E_z[jm] + a_2 * E_z[jc] + a_3 * E_z[jp]
    b_z = a_1 * B_z[jm] + a_2 * B_z[jc] + a_3 * B_z[jp]
    return e_x, e_y, e_z, b_y, b_z


def move_one_particle(E_r, E_f, E_z, B_f, B_z, r, p_r, p_f, p_z, q, m,
                      r_step, xi_step_p, max_radius):
    """Vectorized single Lorentz push. Returns (r, p_r, p_f, p_z, q, path)."""
    mr = max_radius
    p_xo, p_yo, p_zo = p_r, p_f, p_z
    gammam = jnp.sqrt(m ** 2 + p_xo ** 2 + p_yo ** 2 + p_zo ** 2)
    lost1 = (gammam == p_z) | ~jnp.isfinite(gammam)

    denom = gammam - p_zo
    dl_t = xi_step_p / jnp.where(denom == 0, 1.0, denom)   # safe; lost1 masks it
    r_pred = r + p_xo * dl_t / 2
    r_k = r_pred

    neg = r_k < 0
    over = r_k > mr
    r_k = jnp.where(neg, -r_k, jnp.where(over, 2 * mr - r_k, r_k))
    p_xo = jnp.where(over, 0.0, p_xo)
    p_yo = jnp.where(over, 0.0, p_yo)
    p_zo = jnp.where(over, 0.0, p_zo)
    dl_t = jnp.where(over, xi_step_p / m, dl_t)
    gammam = jnp.where(over, m, gammam)
    still_out = (r_k < 0) | (r_k > mr)

    cell_idx = jnp.floor(r_k / r_step + 0.5).astype(jnp.int64)
    local = r_k / r_step - cell_idx
    e_x, e_y, e_z, b_y, b_z = _interpolate_fields(cell_idx, local, E_r, E_f, E_z, B_f, B_z)

    d_px = q * dl_t * (gammam * e_x + p_yo * b_z - p_zo * b_y)
    d_py = q * dl_t * (gammam * e_y - p_zo * e_y - p_xo * b_z)
    d_pz = q * dl_t * (gammam * e_z + p_xo * b_y + p_yo * e_y)
    p_x = p_xo + d_px / 2
    p_y = p_yo + d_py / 2
    p_zh = p_zo + d_pz / 2

    gammam2 = jnp.sqrt(m ** 2 + p_x ** 2 + p_y ** 2 + p_zh ** 2)
    lost3 = (gammam2 == p_zh) | ~jnp.isfinite(gammam2)
    denom2 = gammam2 - p_zh
    dl_t2 = xi_step_p / jnp.where(denom2 == 0, 1.0, denom2)

    d_px = q * dl_t2 * (gammam2 * e_x + p_y * b_z - p_zh * b_y)
    d_py = q * dl_t2 * (gammam2 * e_y - p_zh * e_y - p_x * b_z)
    d_pz = q * dl_t2 * (gammam2 * e_z + p_x * b_y + p_y * e_y)
    p_xh = p_xo + d_px / 2
    p_yh = p_yo + d_py / 2
    dx = p_xh * dl_t2
    dy = p_yh * dl_t2
    p_xf = p_xo + d_px
    p_yf = p_yo + d_py
    p_zf = p_zo + d_pz

    x_new = r + dx
    r_new = jnp.sqrt(x_new ** 2 + dy ** 2)
    path = jnp.maximum(jnp.maximum(jnp.abs(r - r_new), jnp.abs(r - r_pred)),
                       jnp.abs(r_new - r_pred))
    nz = r_new != 0
    p_r_new = jnp.where(nz, (p_xf * x_new + p_yf * dy) / jnp.where(nz, r_new, 1.0), p_xf)
    p_f_new = jnp.where(nz, (p_yf * x_new - p_xf * dy) / jnp.where(nz, r_new, 1.0), 0.0)

    neg2 = r_new < 0
    over2 = r_new > mr
    r_new = jnp.where(neg2, -r_new, jnp.where(over2, 2 * mr - r_new, r_new))
    p_r_new = jnp.where(neg2, -p_r_new, jnp.where(over2, 0.0, p_r_new))
    p_f_new = jnp.where(over2, 0.0, p_f_new)
    p_zf = jnp.where(over2, 0.0, p_zf)
    lost4 = (~jnp.isfinite(p_zf) | ~jnp.isfinite(p_r_new) | ~jnp.isfinite(p_f_new)
             | (r_new > mr) | (r_new < 0))

    lost_any = lost1 | still_out | lost3 | lost4
    r_out = jnp.where(lost_any, mr / 2, r_new)
    p_r_out = jnp.where(lost_any, 0.0, p_r_new)
    p_f_out = jnp.where(lost_any, 0.0, p_f_new)
    p_z_out = jnp.where(lost_any, 0.0, p_zf)
    q_out = jnp.where(lost_any, 0.0, q)
    path_out = jnp.where(lost1 | lost3 | lost4, 0.0, path)
    path_out = jnp.where(still_out & ~lost1, jnp.abs(r - r_pred), path_out)
    return r_out, p_r_out, p_f_out, p_z_out, q_out, path_out


def move_particles(E_r, E_f, E_z, B_f, B_z, r0, p_r0, p_f0, p_z0, q0, m,
                   r_step, xi_step_p, max_radius, n_attempts=N_SUBSTEP_ATTEMPTS):
    """Bounded-scan port of _move_particles_with_substepping (differentiable)."""
    fields = (E_r, E_f, E_z, B_f, B_z)

    init = dict(xip=jnp.zeros_like(r0), dxc=jnp.full_like(r0, xi_step_p),
                r=r0, pr=p_r0, pf=p_f0, pz=p_z0, q=q0,
                rs=r0, prs=p_r0, pfs=p_f0, pzs=p_z0, qs=q0)

    def body(s, _):
        active = (s["xip"] <= 0.99999 * xi_step_p) & (s["q"] != 0)
        rn, prn, pfn, pzn, qn, path = move_one_particle(
            *fields, s["r"], s["pr"], s["pf"], s["pz"], s["q"], m,
            r_step, s["dxc"], max_radius)
        halve = active & (path >= r_step) & (s["dxc"] / xi_step_p > 1e-4)
        accept = active & ~halve

        dxc = jnp.where(halve, s["dxc"] / 2, s["dxc"])
        # accept -> take new & save; halve -> restore saved; inactive -> unchanged
        def upd(new, cur, sav):
            return jnp.where(accept, new, jnp.where(halve, sav, cur))
        r = upd(rn, s["r"], s["rs"]); pr = upd(prn, s["pr"], s["prs"])
        pf = upd(pfn, s["pf"], s["pfs"]); pz = upd(pzn, s["pz"], s["pzs"])
        q = upd(qn, s["q"], s["qs"])
        rs = jnp.where(accept, rn, s["rs"]); prs = jnp.where(accept, prn, s["prs"])
        pfs = jnp.where(accept, pfn, s["pfs"]); pzs = jnp.where(accept, pzn, s["pzs"])
        qs = jnp.where(accept, qn, s["qs"])
        xip = jnp.where(accept, s["xip"] + s["dxc"], s["xip"])
        return dict(xip=xip, dxc=dxc, r=r, pr=pr, pf=pf, pz=pz, q=q,
                    rs=rs, prs=prs, pfs=pfs, pzs=pzs, qs=qs), None

    s, _ = jax.lax.scan(body, init, None, length=n_attempts)
    return s["r"], s["pr"], s["pf"], s["pz"], s["q"]
