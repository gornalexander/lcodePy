"""
Validate the JAX beam deposition (MB1) against numba.

Uses the beam_solver reference states: pull the same three beam layers via numba's
MemoryBeamSource2D, deposit them with numba (reference `density`) and with the JAX kernel
(reproducing the per-layer rho_layout carry), and compare. Also check differentiability
w.r.t. the beam charge.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

import lcode
import lcode.beam.beam_io as nbeam_io
import lcode.beam.beam_calculate as nbeamcalc
import beam_jax as jbeam

jax.config.update("jax_enable_x64", True)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "tests", "unit", "data", "beam_solver")
R_STEP, XI_STEP, WINDOW_WIDTH, PPC = 0.01, 0.01, 2, 10
N_CELLS = int(WINDOW_WIDTH / R_STEP) + 1
XI_I0 = 55


def Cfg():
    return lcode.config.config.Config({
        "geometry": "2d", "processing-unit-type": "cpu", "time-step": 40,
        "transverse-step": R_STEP, "window-width": WINDOW_WIDTH, "xi-step": XI_STEP,
        "plasma-particles-per-cell": PPC, "beam-substepping-energy": 2})


def pulled_layers(state):
    """Pull the three beam layers (55,56,57) as (r, xi, q_norm) arrays via numba source."""
    data = np.load(os.path.join(DATA_DIR, state + ".npz"))
    src = nbeam_io.MemoryBeamSource2D(Cfg(), data["init_layer"])
    layers = []
    for i in range(3):
        bl = src.pull(XI_I0 + i)
        layers.append((np.array(bl.r), np.array(bl.xi), np.array(bl.q_norm)))
    return layers, data["density"]


def numba_density(state):
    data = np.load(os.path.join(DATA_DIR, state + ".npz"))
    src = nbeam_io.MemoryBeamSource2D(Cfg(), data["init_layer"])
    bc = nbeamcalc.BeamCalculator2D(Cfg()); bc.start_time_step()
    dens = np.zeros((N_CELLS, 3))
    for i in range(3):
        dens[:, i] = bc.deposit_beam_layer(src.pull(XI_I0 + i), XI_I0 + i)
    return dens


def jax_density(layers):
    rho_layout = jnp.zeros(N_CELLS)
    cols = []
    for i, (r, xi, q) in enumerate(layers):
        dens, rho_layout = jbeam.deposit_beam_layer(
            rho_layout, jnp.asarray(r), jnp.asarray(xi), jnp.asarray(q),
            XI_I0 + i, R_STEP, XI_STEP, N_CELLS)
        cols.append(dens)
    return jnp.stack(cols, axis=1)


def main():
    states = ["2D_state_1", "2D_state_2", "2D_state_3"]
    print("=== JAX beam deposition vs numba (reference beam_solver states) ===")
    worst = 0.0
    for s in states:
        layers, ref = pulled_layers(s)
        nb = numba_density(s)
        jx = np.asarray(jax_density(layers))
        d_nb = np.abs(jx - nb).max()
        d_ref = np.abs(jx - ref).max()
        worst = max(worst, d_nb)
        print(f"  {s}: max|jax-numba|={d_nb:.2e}  max|jax-ref_file|={d_ref:.2e}  peak={np.abs(nb).max():.1f}")
    print(f"  --> worst |jax-numba| = {worst:.2e}")

    print("\n=== differentiability (grad of deposited charge w.r.t. beam q_norm) ===")
    layers, _ = pulled_layers("2D_state_1")
    r, xi, q0 = layers[1]           # a non-empty layer
    q0 = jnp.asarray(q0)

    def objective(q):
        dens, _ = jbeam.deposit_beam_layer(jnp.zeros(N_CELLS), jnp.asarray(r), jnp.asarray(xi),
                                           q, XI_I0 + 1, R_STEP, XI_STEP, N_CELLS)
        return jnp.sum(dens ** 2)

    g = np.asarray(jax.grad(objective)(q0))
    i0 = int(np.argmax(np.abs(g)))
    eps = 1e-6
    fd = float((objective(q0.at[i0].add(eps)) - objective(q0.at[i0].add(-eps))) / (2 * eps))
    print(f"  grad finite={bool(np.all(np.isfinite(g)))}, nonzero={int(np.sum(g!=0))}/{g.size}")
    print(f"  grad[{i0}]={g[i0]:.6e}  finite-diff={fd:.6e}  rel={abs(g[i0]-fd)/(abs(fd)+1e-30):.2e}")


if __name__ == "__main__":
    main()
