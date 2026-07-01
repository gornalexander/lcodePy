"""
Regression tests: the differentiable JAX backend (`lcode.jax`) must stay consistent with the
original numba code.

Two levels:
  1. kernel level  — the JAX field solver reproduces the numba solver on the reference states;
  2. full simulation — an identical driver run through the real `lcode.Simulation` and through
     the JAX time loop agrees particle-by-particle (matched by id) after several time steps.

Skipped automatically if `jax` is not installed.
"""
import io
import contextlib
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import lcode
import lcode.plasma as nplasma
import lcode.beam.beam_calculate as nbeamcalc
import lcode.push_solvers.push_solver as nps
from lcode.plasma.data import Arrays
from lcode import jax as lj

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unit", "data", "plasma_solver")


# --------------------------------------------------------------- kernel level
def test_field_solver_matches_numba():
    """JAX compute_fields reproduces numba get_field_computer on a reference state."""
    import lcode.plasma.fields as nfields
    R = 0.01
    data = np.load(os.path.join(DATA, "2D_state_1.npz"))
    n = data["fields_pred"][0].shape[0]
    fp = data["fields_prev"]
    fields_prev = Arrays(xp=np, E_r=fp[0], E_f=fp[1], E_z=fp[2], B_z=fp[3], B_f=fp[4])
    fields_pred = Arrays(xp=np, E_r=data["fields_pred"][0], E_f=data["fields_pred"][1],
                         E_z=np.zeros(n), B_z=np.zeros(n), B_f=np.zeros(n))
    cp = Arrays(xp=np, rho=np.ones((2, n)), j_r=np.zeros((2, n)), j_f=np.zeros((2, n)), j_z=np.zeros((2, n)))
    cpr = cp.copy()
    cpr.rho[0] = data["currents_prev"][0] - 1; cpr.j_r[0] = data["currents_prev"][1]
    cpr.j_f[0] = data["currents_prev"][2];     cpr.j_z[0] = data["currents_prev"][3]
    cp.rho[0] = data["currents_pred"][0] - 1;  cp.j_r[0] = data["currents_pred"][1]
    cp.j_f[0] = data["currents_pred"][2];      cp.j_z[0] = data["currents_pred"][3]

    class Cfg:
        def getfloat(self, k): return R if k == "transverse-step" else 0.0
    nb = nfields.get_field_computer(Cfg())(fields_pred, fields_prev, data["beam_current"], cpr, cp, R)[0]

    to = lambda d: {k: jnp.asarray(v) for k, v in d.items()}
    fields = {"E_r": data["fields_pred"][0], "E_f": data["fields_pred"][1],
              "E_z": np.zeros(n), "B_z": np.zeros(n), "B_f": np.zeros(n)}
    fprev = {"E_r": fp[0], "E_f": fp[1], "E_z": fp[2], "B_z": fp[3], "B_f": fp[4]}
    cprj = {c: np.vstack([data["currents_prev"][i] - (1 if c == "rho" else 0), np.zeros(n) if c != "rho" else np.ones(n)])
            for i, c in enumerate(("rho", "j_r", "j_f", "j_z"))}
    curj = {c: np.vstack([data["currents_pred"][i] - (1 if c == "rho" else 0), np.zeros(n) if c != "rho" else np.ones(n)])
            for i, c in enumerate(("rho", "j_r", "j_f", "j_z"))}
    jx, _ = lj.compute_fields(to(fields), to(fprev), jnp.asarray(data["beam_current"]),
                              to(cprj), to(curj), xi_step_p=R, r_step=R)
    for comp in ("E_r", "E_f", "E_z", "B_f", "B_z"):
        rel = np.max(np.abs(np.asarray(jx[comp]) - getattr(nb, comp)) / (np.abs(getattr(nb, comp)) + 1e-30))
        assert rel < 1e-8, f"{comp}: rel diff {rel:.2e}"


# ---------------------------------------------------------- full simulation
CFG = dict(geometry="2d", **{"processing-unit-type": "cpu", "window-width": 5,
           "transverse-step": 0.05, "window-length": 5, "xi-step": 0.05,
           "plasma-particles-per-cell": 10, "ion-model": "background",
           "noise-reductor-enabled": False, "time-step": 25})
BEAM = {"current": -0.05, "particles_in_layer": 150,
        "default": {"angspread": 1e-5, "energy": 1000},
        "driver": {"xishape": "l", "length": 3, "radius": 1.0}}
N_STEPS, TIME_STEP = 3, 25.0


def _run_numba(n_steps):
    nps.PusherAndSolver.warmup = lambda self, *a, **k: None
    cols = {k: [] for k in ("r", "xi", "p_z", "p_r", "M", "q_m", "q_norm", "id")}
    orig = nbeamcalc.BeamCalculator2D.deposit_beam_layer
    grab = [True]

    def dep(self, layer, xi_i):
        if layer.size and grab[0]:
            for k in cols: cols[k].append(np.array(getattr(layer, k)))
        return orig(self, layer, xi_i)
    nbeamcalc.BeamCalculator2D.deposit_beam_layer = dep
    cfg1 = dict(CFG); cfg1["time-limit"] = TIME_STEP
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            lcode.Simulation(config=cfg1, beam_parameters=BEAM, diagnostics=[], runas_filename="").step()
    finally:
        nbeamcalc.BeamCalculator2D.deposit_beam_layer = orig
    init = {k: np.concatenate(v) for k, v in cols.items()}

    # full run with a ParticlesDiag to save the beam each step
    from lcode.diagnostics import ParticlesDiag
    import glob, shutil
    if os.path.isdir("diagnostics/_consist"): shutil.rmtree("diagnostics/_consist")
    cfg2 = dict(CFG); cfg2["time-limit"] = TIME_STEP * n_steps
    diag = [ParticlesDiag(save_beam=True, output_period=TIME_STEP, directory_name="_consist")]
    with contextlib.redirect_stdout(io.StringIO()):
        lcode.Simulation(config=cfg2, beam_parameters=BEAM, diagnostics=diag, runas_filename="").step()
    final = np.load(sorted(glob.glob("diagnostics/_consist/beam_*.npz"))[-1])
    shutil.rmtree("diagnostics/_consist", ignore_errors=True)
    return init, final


def _run_jax(init, n_steps):
    R = CFG["transverse-step"]; WW = CFG["window-width"]; XI = CFG["xi-step"]
    PPC = CFG["plasma-particles-per-cell"]; NC = int(WW / R) + 1; NXI = int(CFG["window-length"] / XI)
    cfgd = {"geometry": "2d", "transverse-step": R, "window-width": WW,
            "plasma-particles-per-cell": PPC, "ion-model": "background"}
    el = nplasma.init_plasma_2d(lcode.config.config.Config(cfgd))[1]["electrons"]
    part = {k: jnp.asarray(getattr(el, k)) for k in ("r", "p_r", "p_f", "p_z", "q", "m")}
    vol = lj.cell_volume(R, PPC, NC)
    plasma = {"particles": part, "fields": {k: jnp.zeros(NC) for k in lj.FIELD_KEYS},
              "currents": lj.compute_rhoj(part, NC, R, vol, 1.0)}
    P = {"n_xi": NXI, "n_cells": NC, "r_step": R, "xi_step": XI, "vol": vol, "ni": 1.0,
         "max_radius": WW, "corrector_steps": 2, "n_attempts": 4, "checkpoint": False, "magnetic_field": 0.0}
    dt, steps0 = lj.init_substepping(jnp.asarray(init["p_z"]), jnp.asarray(init["q_m"]),
                                     jnp.zeros_like(init["p_z"]), jnp.zeros_like(init["p_z"]), TIME_STEP, 2.0)
    max_sub = int(np.asarray(steps0).max())
    beam = {k: jnp.asarray(init[k]) for k in init}; beam["remaining_steps"] = steps0
    for _ in range(n_steps):
        beam, ez, lost = lj.one_time_step(beam, plasma, P, dt, max_sub)
    return {k: np.asarray(beam[k]) for k in ("r", "xi", "p_z", "p_r", "id")}


def test_time_dynamics_matches_numba_per_particle():
    """Real numba Simulation vs JAX time loop: per-particle agreement (matched by id)."""
    init, nb = _run_numba(N_STEPS)
    jx = _run_jax(init, N_STEPS)
    nb_id = np.abs(nb["id"]).astype(int); jx_id = np.abs(jx["id"]).astype(int)
    common = np.intersect1d(nb_id, jx_id)
    assert common.size == nb_id.size == jx_id.size, "particle sets differ"
    nb_ix = {i: k for k, i in enumerate(nb_id)}; jx_ix = {i: k for k, i in enumerate(jx_id)}
    tol = {"r": 1e-4, "p_z": 1e-3, "p_r": 1e-3, "xi": 1e-6}
    for f, t in tol.items():
        a = np.array([nb[f][nb_ix[i]] for i in common])
        b = np.array([jx[f][jx_ix[i]] for i in common])
        mx = np.max(np.abs(a - b))
        assert mx < t, f"{f}: max per-particle |delta| = {mx:.2e} exceeds {t:.0e}"
