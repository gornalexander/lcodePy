"""
Validate the JAX deposition (M2) against the numba reference.

On the real plasma_solver reference states (particles_prev), compare JAX compute_rhoj to
numba compute_rhoj (tolerance context: the deposition unit test uses rtol=5e-11), and check
differentiability w.r.t. particle coordinates.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

import lcode.plasma.rhoj as nrhoj
from lcode.plasma.data import Arrays
import deposition_jax as jdep

jax.config.update("jax_enable_x64", True)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PL_DIR = os.path.join(REPO, "tests", "unit", "data", "plasma_solver")
STATES = ["2D_state_0", "2D_state_1", "2D_state_2",
          "2D_state_Bz0_0", "2D_state_Bz0_1", "2D_state_Bz0_2"]
R_STEP = 0.01
WINDOW_WIDTH = 2
N_CELLS = int(WINDOW_WIDTH / R_STEP) + 1        # 201
PPC = 10


class Cfg:
    _m = {"transverse-step": R_STEP, "window-width": WINDOW_WIDTH,
          "plasma-particles-per-cell": PPC}
    def getfloat(self, k): return float(self._m[k])
    def getint(self, k): return int(self._m[k])
    def get(self, k): return "background"


def numba_rhoj(pp):
    const = Arrays(xp=np, ni=1); const.sorts = {"electrons": 0}
    p = Arrays(xp=np, r=pp[0], p_r=pp[1], p_f=pp[2], p_z=pp[3], q=pp[4], m=pp[5],
               age=np.zeros_like(pp[0]))
    out = nrhoj.get_rhoj_computer(Cfg())({"electrons": p}, const)
    return {k: getattr(out, k) for k in ("rho", "j_r", "j_f", "j_z")}


def maxrel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b) / (np.abs(b) + 1e-30))


def main():
    vol = jdep.cell_volume(R_STEP, PPC, N_CELLS)

    print("=== JAX deposition vs numba compute_rhoj (reference states) ===")
    worst = 0.0
    for s in STATES:
        pp = np.load(os.path.join(PL_DIR, s + ".npz"))["particles_prev"]
        nb = numba_rhoj(pp)
        p = {k: jnp.asarray(pp[i]) for i, k in enumerate(("r", "p_r", "p_f", "p_z", "q", "m"))}
        jx = jdep.compute_rhoj(p, N_CELLS, R_STEP, vol, ni=1.0)
        # compare the summed (electrons+ions) current, as the unit test does
        rels = {k: maxrel(jx[k].sum(0), nb[k].sum(0)) for k in nb}
        worst = max(worst, max(rels.values()))
        print(f"  {s:16s}: " + "  ".join(f"{k}={v:.1e}" for k, v in rels.items()))
    print(f"  --> worst rel diff: {worst:.2e}   (deposition unit-test tol = 5e-11)")

    print("\n=== differentiability (grad w.r.t. particle r) ===")
    pp = np.load(os.path.join(PL_DIR, "2D_state_1.npz"))["particles_prev"]
    p0 = {k: jnp.asarray(pp[i]) for i, k in enumerate(("r", "p_r", "p_f", "p_z", "q", "m"))}

    def objective(r):
        p = dict(p0); p["r"] = r
        out = jdep.compute_rhoj(p, N_CELLS, R_STEP, vol, ni=1.0)
        return jnp.sum(out["rho"][0] ** 2)  # depends on where/how much each particle deposits

    g = np.asarray(jax.grad(objective)(p0["r"]))
    # finite-diff check on a particle well inside a cell (avoid the floor discontinuity)
    r_np = np.asarray(p0["r"]); rloc = r_np / R_STEP - np.floor(r_np / R_STEP + 0.5)
    interior = np.where(np.abs(rloc) < 0.3)[0]
    i0 = int(interior[np.argmax(np.abs(g[interior]))])
    eps = 1e-8
    fd = float((objective(p0["r"].at[i0].add(eps)) - objective(p0["r"].at[i0].add(-eps))) / (2 * eps))
    print(f"  grad finite={bool(np.all(np.isfinite(g)))}, nonzero={int(np.sum(g != 0))}/{g.size}")
    print(f"  grad[{i0}]={g[i0]:.6e}  finite-diff={fd:.6e}  "
          f"rel={abs(g[i0]-fd)/(abs(fd)+1e-30):.2e}  (FD-limited, particle interior to cell)")


if __name__ == "__main__":
    main()
