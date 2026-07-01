"""
Validate the JAX field solver against the numba reference (Branch-2 de-risking).

Checks, on the real unit-test reference states (tests/unit/data/plasma_solver):
  1. Forward match: JAX compute_fields vs numba compute_fields (max abs/rel diff).
  2. Sub-kernel match: JAX tridiagonal_solve / cumtrapz vs numba on random input.
  3. Differentiability: jax.grad of a scalar objective through the whole solver runs,
     and matches a finite-difference check.
  4. A quick jit timing sanity check.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

import lcode.plasma.fields as nfields
import lcode.plasma.ode as node
import field_solver_jax as jfs

jax.config.update("jax_enable_x64", True)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PL_DIR = os.path.join(REPO, "tests", "unit", "data", "plasma_solver")
STATES = ["2D_state_0", "2D_state_1", "2D_state_2",
          "2D_state_Bz0_0", "2D_state_Bz0_1", "2D_state_Bz0_2"]
R_STEP = 0.01


def maxreldiff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b) / (np.abs(b) + 1e-30)), np.max(np.abs(a - b))


def numba_compute_fields(data):
    """Reproduce exactly what test_fields_solver feeds to the numba solver."""
    from lcode.plasma.data import Arrays
    n = data["fields_pred"][0].shape[0]
    fp = data["fields_prev"]
    fields_prev = Arrays(xp=np, E_r=fp[0], E_f=fp[1], E_z=fp[2], B_z=fp[3], B_f=fp[4])
    fields_pred = Arrays(xp=np, E_r=data["fields_pred"][0], E_f=data["fields_pred"][1],
                         E_z=np.zeros(n), B_z=np.zeros(n), B_f=np.zeros(n))
    cp = Arrays(xp=np, rho=np.ones((2, n)), j_r=np.zeros((2, n)),
                j_f=np.zeros((2, n)), j_z=np.zeros((2, n)))
    cpr = cp.copy()
    cpr.rho[0] = data["currents_prev"][0] - 1; cpr.j_r[0] = data["currents_prev"][1]
    cpr.j_f[0] = data["currents_prev"][2];     cpr.j_z[0] = data["currents_prev"][3]
    cp.rho[0] = data["currents_pred"][0] - 1;  cp.j_r[0] = data["currents_pred"][1]
    cp.j_f[0] = data["currents_pred"][2];      cp.j_z[0] = data["currents_pred"][3]

    class Cfg:  # minimal config stub for get_field_computer
        def getfloat(self, k): return R_STEP if k == "transverse-step" else 0.0
    solver = nfields.get_field_computer(Cfg())
    out = solver(fields_pred, fields_prev, data["beam_current"], cpr, cp, R_STEP)[0]
    return {k: getattr(out, k) for k in ("E_r", "E_f", "E_z", "B_f", "B_z")}


def to_jax_inputs(data):
    n = data["fields_pred"][0].shape[0]
    fp = data["fields_prev"]
    fields = {"E_r": data["fields_pred"][0], "E_f": data["fields_pred"][1],
              "E_z": np.zeros(n), "B_z": np.zeros(n), "B_f": np.zeros(n)}
    fields_prev = {"E_r": fp[0], "E_f": fp[1], "E_z": fp[2], "B_z": fp[3], "B_f": fp[4]}
    cpr = {"rho": np.vstack([data["currents_prev"][0] - 1, np.ones(n)]),
           "j_r": np.vstack([data["currents_prev"][1], np.zeros(n)]),
           "j_f": np.vstack([data["currents_prev"][2], np.zeros(n)]),
           "j_z": np.vstack([data["currents_prev"][3], np.zeros(n)])}
    cp = {"rho": np.vstack([data["currents_pred"][0] - 1, np.ones(n)]),
          "j_r": np.vstack([data["currents_pred"][1], np.zeros(n)]),
          "j_f": np.vstack([data["currents_pred"][2], np.zeros(n)]),
          "j_z": np.vstack([data["currents_pred"][3], np.zeros(n)])}
    j = lambda d: {k: jnp.asarray(v) for k, v in d.items()}
    return (j(fields), j(fields_prev), jnp.asarray(data["beam_current"]), j(cpr), j(cp))


def main():
    print("=== 1. sub-kernel checks vs numba (random input) ===")
    rng = np.random.default_rng(0)
    n = 201
    rp = rng.standard_normal(n); pf = np.abs(rng.standard_normal(n)) + 1
    nb_tri = node.tridiagonal_solve_neumann_like(rp.copy(), pf.copy(), R_STEP)
    jx_tri = np.asarray(jfs.tridiagonal_solve_neumann_like(jnp.asarray(rp), jnp.asarray(pf), R_STEP))
    print(f"  tridiagonal_neumann : max_rel={maxreldiff(jx_tri, nb_tri)[0]:.2e}")
    nb_trid = node.tridiagonal_solve_dirichlet(rp.copy(), pf.copy(), R_STEP)
    jx_trid = np.asarray(jfs.tridiagonal_solve_dirichlet(jnp.asarray(rp), jnp.asarray(pf), R_STEP))
    print(f"  tridiagonal_dirichlet: max_rel={maxreldiff(jx_trid, nb_trid)[0]:.2e}")
    y = rng.standard_normal(n); ez = np.zeros(n)
    node.cumtrapz_numba(y, R_STEP, ez, "backward")
    jz = np.asarray(jfs.cumtrapz(jnp.asarray(y), R_STEP, "backward"))
    print(f"  cumtrapz backward   : max_abs={maxreldiff(jz, ez)[1]:.2e}")

    print("\n=== 2. full compute_fields vs numba (reference states) ===")
    worst = 0.0
    for s in STATES:
        data = np.load(os.path.join(PL_DIR, s + ".npz"))
        nb = numba_compute_fields(data)
        jin = to_jax_inputs(data)
        jx, _ = jfs.compute_fields(*jin, xi_step_p=R_STEP, r_step=R_STEP)
        rels = {k: maxreldiff(jx[k], nb[k])[0] for k in nb}
        worst = max(worst, max(rels.values()))
        print(f"  {s:16s}: " + "  ".join(f"{k}={v:.1e}" for k, v in rels.items()))
    print(f"  --> worst rel diff across all states/components: {worst:.2e}")

    print("\n=== 3. differentiability (jax.grad end-to-end) ===")
    data = np.load(os.path.join(PL_DIR, "2D_state_1.npz"))
    jin = to_jax_inputs(data)

    # Objective that genuinely depends on beam_current: it enters E_r via
    # total_rho = rho_sum + rho_beam, then flows through the tridiagonal solve.
    def objective(beam_current):
        f, _ = jfs.compute_fields(jin[0], jin[1], beam_current, jin[3], jin[4],
                                  xi_step_p=R_STEP, r_step=R_STEP)
        return jnp.sum(f["E_r"] ** 2)  # scalar

    bc = jin[2]
    g = np.asarray(jax.grad(objective)(bc))
    i0 = int(np.argmax(np.abs(g)))          # a component with real sensitivity
    eps = 1e-6
    fd = float((objective(bc.at[i0].add(eps)) - objective(bc.at[i0].add(-eps))) / (2 * eps))
    print(f"  grad ok: shape={g.shape}, finite={bool(np.all(np.isfinite(g)))}, "
          f"nonzero={int(np.sum(g != 0))}/{g.size}")
    print(f"  grad[{i0}]={g[i0]:.6e}  vs  finite-diff={fd:.6e}  "
          f"rel={abs(g[i0] - fd) / (abs(fd) + 1e-30):.2e}  (FD-limited)")

    print("\n=== 4. jit sanity ===")
    import time
    fjit = jax.jit(lambda bc: jfs.compute_fields(jin[0], jin[1], bc, jin[3], jin[4],
                                                 xi_step_p=R_STEP, r_step=R_STEP)[0]["E_z"])
    fjit(bc).block_until_ready()  # compile
    t = time.perf_counter()
    for _ in range(100):
        fjit(bc).block_until_ready()
    print(f"  jit compute_fields E_z: {(time.perf_counter()-t)/100*1e6:.1f} us/call")


if __name__ == "__main__":
    main()
