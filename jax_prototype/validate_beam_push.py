"""
Validate the JAX beam push (MB2) against numba `push_beam_layer`.

Uses the beam_solver reference states (init_layer + init_remaining_steps + init_dt +
prev_pl_fields + pl_fields). Pushes with numba and with the JAX kernel and compares the
updated particle state (r, xi, p_z, p_r, M). Also checks differentiability.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

import lcode
import lcode.beam.beam_calculate as nbeamcalc
import lcode.beam.data as nbeamdata
from lcode.plasma.data import Arrays
import beam_jax as jbeam

jax.config.update("jax_enable_x64", True)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "tests", "unit", "data", "beam_solver")
R_STEP, XI_STEP, WINDOW_WIDTH = 0.01, 0.01, 2
XI_I = 58

EXTRA = {"2D_state_1": {}, "2D_state_2": {"magnetic-field": 5},
         "2D_state_3": {"beam-substepping-energy": 3900}}


def cfg_for(state):
    d = {"geometry": "2d", "processing-unit-type": "cpu", "time-step": 40,
         "transverse-step": R_STEP, "window-width": WINDOW_WIDTH, "xi-step": XI_STEP,
         "plasma-particles-per-cell": 10, "beam-substepping-energy": 2}
    d.update(EXTRA[state])
    return lcode.config.config.Config(d)


def numba_push(state):
    data = np.load(os.path.join(DATA_DIR, state + ".npz"))
    conf = cfg_for(state)
    bl = nbeamdata.BeamParticles(beam_array=data["init_layer"])
    bl.remaining_steps = data["init_remaining_steps"].copy()
    bl.dt = data["init_dt"].copy()
    bc = nbeamcalc.BeamCalculator2D(conf); bc.start_time_step()
    pf, cf = data["prev_pl_fields"], data["pl_fields"]
    prev = Arrays(xp=np, E_r=pf[:, 0], E_f=pf[:, 1], E_z=pf[:, 2], B_f=pf[:, 3], B_z=pf[:, 4])
    cur = Arrays(xp=np, E_r=cf[:, 0], E_f=cf[:, 1], E_z=cf[:, 2], B_f=cf[:, 3], B_z=cf[:, 4])
    bc.move_beam_slice(bl, XI_I, cur, prev)   # pushes in place
    return {"r": np.array(bl.r), "xi": np.array(bl.xi), "p_z": np.array(bl.p_z),
            "p_r": np.array(bl.p_r), "M": np.array(bl.M),
            "remaining_steps": np.array(bl.remaining_steps)}


def jax_push(state):
    data = np.load(os.path.join(DATA_DIR, state + ".npz"))
    il = data["init_layer"]
    pf, cf = data["prev_pl_fields"], data["pl_fields"]
    prev = {"E_r": jnp.asarray(pf[:, 0]), "E_f": jnp.asarray(pf[:, 1]), "E_z": jnp.asarray(pf[:, 2]),
            "B_f": jnp.asarray(pf[:, 3]), "B_z": jnp.asarray(pf[:, 4])}
    cur = {"E_r": jnp.asarray(cf[:, 0]), "E_f": jnp.asarray(cf[:, 1]), "E_z": jnp.asarray(cf[:, 2]),
           "B_f": jnp.asarray(cf[:, 3]), "B_z": jnp.asarray(cf[:, 4])}
    mag = float(EXTRA[state].get("magnetic-field", 0.0))
    e_ss = float(EXTRA[state].get("beam-substepping-energy", 2))
    # replicate move_beam_slice: init_substepping for dt==0 particles first
    dt, steps0 = jbeam.init_substepping(jnp.asarray(il["p_z"]), jnp.asarray(il["q_m"]),
                                        jnp.asarray(data["init_dt"]),
                                        jnp.asarray(data["init_remaining_steps"]), 40.0, e_ss)
    max_sub = int(np.asarray(steps0).max())
    r, xi, pz, pr, M, steps, lost = jbeam.push_beam_layer(
        jnp.asarray(il["r"]), jnp.asarray(il["xi"]), jnp.asarray(il["p_z"]),
        jnp.asarray(il["p_r"]), jnp.asarray(il["M"]), jnp.asarray(il["q_m"]),
        steps0, dt, prev, cur, XI_I, R_STEP, XI_STEP, WINDOW_WIDTH, mag, max_sub)
    return {"r": np.array(r), "xi": np.array(xi), "p_z": np.array(pz),
            "p_r": np.array(pr), "M": np.array(M), "remaining_steps": np.array(steps)}


def main():
    print("=== JAX beam push vs numba push_beam_layer (reference states) ===")
    for s in EXTRA:
        nb = numba_push(s)
        jx = jax_push(s)
        # element-wise (numba pushes in place, no reordering)
        rels = {k: np.max(np.abs(nb[k] - jx[k])) for k in ("r", "xi", "p_z", "p_r", "M", "remaining_steps")}
        print(f"  {s}: " + "  ".join(f"{k}={v:.1e}" for k, v in rels.items()))

    print("\n=== differentiability (grad of final energy w.r.t. wake field) ===")
    data = np.load(os.path.join(DATA_DIR, "2D_state_1.npz"))
    il = data["init_layer"]; cf = data["pl_fields"]; pf = data["prev_pl_fields"]
    prev = {"E_r": jnp.asarray(pf[:, 0]), "E_f": jnp.asarray(pf[:, 1]), "E_z": jnp.asarray(pf[:, 2]),
            "B_f": jnp.asarray(pf[:, 3]), "B_z": jnp.asarray(pf[:, 4])}
    max_sub = int(data["init_remaining_steps"].max())
    base = {"E_r": jnp.asarray(cf[:, 0]), "E_f": jnp.asarray(cf[:, 1]), "E_z": jnp.asarray(cf[:, 2]),
            "B_f": jnp.asarray(cf[:, 3]), "B_z": jnp.asarray(cf[:, 4])}

    def objective(Ez):
        cur = dict(base); cur["E_z"] = Ez
        r, xi, pz, pr, M, steps, lost = jbeam.push_beam_layer(
            jnp.asarray(il["r"]), jnp.asarray(il["xi"]), jnp.asarray(il["p_z"]),
            jnp.asarray(il["p_r"]), jnp.asarray(il["M"]), jnp.asarray(il["q_m"]),
            jnp.asarray(data["init_remaining_steps"]), jnp.asarray(data["init_dt"]),
            prev, cur, XI_I, R_STEP, XI_STEP, WINDOW_WIDTH, 0.0, max_sub)
        return jnp.sum(pz ** 2)     # final beam energy depends on the accelerating field

    g = np.asarray(jax.grad(objective)(base["E_z"]))
    i0 = int(np.argmax(np.abs(g)))
    eps = 1e-6
    fd = float((objective(base["E_z"].at[i0].add(eps)) - objective(base["E_z"].at[i0].add(-eps))) / (2 * eps))
    print(f"  grad finite={bool(np.all(np.isfinite(g)))}, nonzero={int(np.sum(g!=0))}/{g.size}")
    print(f"  grad[{i0}]={g[i0]:.6e}  finite-diff={fd:.6e}  rel={abs(g[i0]-fd)/(abs(fd)+1e-30):.2e}")


if __name__ == "__main__":
    main()
