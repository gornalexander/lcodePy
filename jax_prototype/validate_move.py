"""
Validate the JAX particle push (M3) against the numba mover.

test_plasma_pusher feeds averaged fields + particles_prev to the mover and compares
(r, p_r, p_f, p_z, q) at rtol=1e-12. We compare the JAX bounded-scan mover to numba on the
same states and check differentiability (grad through the bounded-scan push).
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

import lcode.plasma.move as nmove
from lcode.plasma.data import Arrays
import move_jax as jm

jax.config.update("jax_enable_x64", True)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PL_DIR = os.path.join(REPO, "tests", "unit", "data", "plasma_solver")
STATES = ["2D_state_0", "2D_state_1", "2D_state_2",
          "2D_state_Bz0_0", "2D_state_Bz0_1", "2D_state_Bz0_2"]
R_STEP, WINDOW_WIDTH = 0.01, 2
MAX_R = WINDOW_WIDTH


class Cfg:
    def __init__(self, bz0): self.bz0 = bz0
    def getfloat(self, k):
        return {"transverse-step": R_STEP, "window-width": WINDOW_WIDTH,
                "magnetic-field": 0.5 if self.bz0 else 0.0}[k]


def numba_move(data, bz0):
    const = Arrays(xp=np, ni=1); const.sorts = {"electrons": 0}
    fp, ff = data["fields_prev"], data["fields"]
    avg = Arrays(xp=np, E_r=(fp[0]+ff[0])/2, E_f=(fp[1]+ff[1])/2, E_z=(fp[2]+ff[2])/2,
                 B_z=(fp[3]+ff[3])/2, B_f=(fp[4]+ff[4])/2)
    pp = data["particles_prev"]
    prev = {"electrons": Arrays(xp=np, r=pp[0], p_r=pp[1], p_f=pp[2], p_z=pp[3],
                                q=pp[4], m=pp[5], age=np.zeros_like(pp[0]))}
    out = nmove.get_plasma_particles_mover(Cfg(bz0))(avg, prev, np.zeros_like(fp[0]),
                                                     R_STEP, const)["electrons"]
    return {a: getattr(out, a) for a in ("r", "p_r", "p_f", "p_z", "q")}


def maxrel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b) / (np.abs(b) + 1e-30))


def jax_move(data, bz0):
    fp, ff = data["fields_prev"], data["fields"]
    bz_ext = 0.5 if bz0 else 0.0
    E_r = jnp.asarray((fp[0]+ff[0])/2); E_f = jnp.asarray((fp[1]+ff[1])/2)
    E_z = jnp.asarray((fp[2]+ff[2])/2); B_z = jnp.asarray((fp[3]+ff[3])/2 + bz_ext)
    B_f = jnp.asarray((fp[4]+ff[4])/2)
    pp = data["particles_prev"]
    r, pr, pf, pz, q = jm.move_particles(
        E_r, E_f, E_z, B_f, B_z,
        jnp.asarray(pp[0]), jnp.asarray(pp[1]), jnp.asarray(pp[2]), jnp.asarray(pp[3]),
        jnp.asarray(pp[4]), jnp.asarray(pp[5]), R_STEP, R_STEP, MAX_R)
    return {"r": r, "p_r": pr, "p_f": pf, "p_z": pz, "q": q}


def main():
    print("=== JAX mover vs numba mover (reference states) ===")
    worst = 0.0
    for s in STATES:
        bz0 = "Bz0" in s
        data = np.load(os.path.join(PL_DIR, s + ".npz"))
        nb = numba_move(data, bz0)
        jx = jax_move(data, bz0)
        rels = {k: maxrel(jx[k], nb[k]) for k in nb}
        worst = max(worst, max(rels.values()))
        print(f"  {s:16s}: " + "  ".join(f"{k}={v:.1e}" for k, v in rels.items()))
    print(f"  --> worst rel diff: {worst:.2e}   (pusher unit-test tol = 1e-12)")

    print("\n=== differentiability (grad through bounded-scan push) ===")
    data = np.load(os.path.join(PL_DIR, "2D_state_1.npz"))
    fp, ff = data["fields_prev"], data["fields"]
    E_r = jnp.asarray((fp[0]+ff[0])/2); E_f = jnp.asarray((fp[1]+ff[1])/2)
    E_z = jnp.asarray((fp[2]+ff[2])/2); B_z = jnp.asarray((fp[3]+ff[3])/2); B_f = jnp.asarray((fp[4]+ff[4])/2)
    pp = data["particles_prev"]
    args = [jnp.asarray(pp[i]) for i in range(6)]

    def objective(E_z_field):
        r, pr, pf, pz, q = jm.move_particles(E_r, E_f, E_z_field, B_f, B_z, *args,
                                             R_STEP, R_STEP, MAX_R)
        return jnp.sum(pz ** 2)   # final longitudinal momentum depends on E_z

    g = np.asarray(jax.grad(objective)(E_z))
    i0 = int(np.argmax(np.abs(g)))
    eps = 1e-7
    fd = float((objective(E_z.at[i0].add(eps)) - objective(E_z.at[i0].add(-eps))) / (2 * eps))
    print(f"  grad finite={bool(np.all(np.isfinite(g)))}, nonzero={int(np.sum(g != 0))}/{g.size}")
    print(f"  grad[{i0}]={g[i0]:.6e}  finite-diff={fd:.6e}  "
          f"rel={abs(g[i0]-fd)/(abs(fd)+1e-30):.2e}  (FD-limited)")


if __name__ == "__main__":
    main()
