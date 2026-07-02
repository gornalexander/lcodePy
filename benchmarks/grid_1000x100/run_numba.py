"""
Backend 1 — numba LCODE, self-consistent evolving beam on the 1000 x 101 grid.

Runs the real `lcode.Simulation` (numba, parallel njit kernels) for n_t time steps with an
evolving beam. Captures, so the JAX backends can start from an identical state and be compared:
  - the INITIAL beam particles          -> beam_init.npz
  - the step-1 on-axis wake Ez(xi)      -> ez_numba.npy
  - beam <p_z>, rms r after evolution   -> stats_numba.npy
and times the n_t-step evolution.

Capture is done by monkeypatching the numba beam-deposition and plasma-solver (the same trick
the validation notebooks use) — no changes to lcode itself.
"""
import io, contextlib, time
import numpy as np

import lcode
import lcode.beam.beam_calculate as nbeamcalc
import lcode.push_solvers.push_solver as nps
from lcode.plasma.solver import CylindricalPlasmaSolver

WW, R, XI, WL, PPC = 10.0, 0.1, 0.1, 100.0, 10          # NC=101, NXI=1000
NC = int(WW / R) + 1
NXI = int(WL / XI)
N_T = 10
CONFIG = {"geometry": "2d", "processing-unit-type": "cpu", "window-width": WW,
          "transverse-step": R, "window-length": WL, "xi-step": XI,
          "time-limit": N_T * 25.0, "time-step": 25.0, "plasma-particles-per-cell": PPC,
          "ion-model": "background", "noise-reductor-enabled": False}
BEAM = {"current": -0.05, "particles_in_layer": 200,
        "default": {"angspread": 1e-5, "energy": 1000},
        "driver": {"xishape": "l", "length": 3, "radius": 1.0}}

print(f"grid NC={NC} NXI={NXI}  n_t={N_T}", flush=True)

# ---------------------------------------------------------------- capture hooks
_capture_beam = {"on": False, "cols": None}
_orig_deposit = nbeamcalc.BeamCalculator2D.deposit_beam_layer
def _deposit(self, layer, xi_i):
    if _capture_beam["on"] and layer.size:
        for k in _capture_beam["cols"]:
            _capture_beam["cols"][k].append(np.array(getattr(layer, k)))
    return _orig_deposit(self, layer, xi_i)
nbeamcalc.BeamCalculator2D.deposit_beam_layer = _deposit

_capture_ez = {"on": False, "ez": None, "k": 0}
_orig_stepdxi = CylindricalPlasmaSolver.step_dxi
def _stepdxi(self, particles, fields, currents, const_arrays, rho_beam, rho_beam_prev):
    particles, fields, currents = _orig_stepdxi(self, particles, fields, currents,
                                                const_arrays, rho_beam, rho_beam_prev)
    if _capture_ez["on"] and _capture_ez["k"] < NXI:
        _capture_ez["ez"][_capture_ez["k"]] = fields.E_z[0]
        _capture_ez["k"] += 1
    return particles, fields, currents
CylindricalPlasmaSolver.step_dxi = _stepdxi


def capture_beam_on():
    _capture_beam["cols"] = {k: [] for k in ("r", "xi", "p_z", "p_r", "M", "q_m", "q_norm")}
    _capture_beam["on"] = True
def capture_beam_off():
    _capture_beam["on"] = False
    return {k: np.concatenate(v) for k, v in _capture_beam["cols"].items()}


nps.PusherAndSolver.warmup = lambda self, *a, **k: None       # skip 2-step JIT warmup
sim = lcode.Simulation(config=dict(CONFIG), beam_parameters=BEAM, diagnostics=[], runas_filename="")

# --- step 1: capture initial beam + step-1 wake Ez(xi); also compiles the kernels
# NOTE: sim.step(1) runs exactly ONE time step (sim.step() with no arg runs time-limit/time-step).
_capture_ez["ez"] = np.empty(NXI); _capture_ez["k"] = 0; _capture_ez["on"] = True
capture_beam_on()
with contextlib.redirect_stdout(io.StringIO()):
    sim.step(1)
beam_init = capture_beam_off()
_capture_ez["on"] = False
ez_numba = _capture_ez["ez"].copy()
np.savez("beam_init.npz", **beam_init)
np.save("ez_numba.npy", ez_numba)
print(f"initial beam: {beam_init['r'].size} particles, mean p_z={beam_init['p_z'].mean():.1f}")
print(f"step-1 wake Ez(xi): min={ez_numba.min():.4e} max={ez_numba.max():.4e}", flush=True)

# --- run to n_t steps, capturing the beam at the last step for stats
t0 = time.perf_counter()
for i in range(2, N_T + 1):
    last = (i == N_T)
    if last:
        capture_beam_on()
    with contextlib.redirect_stdout(io.StringIO()):
        sim.step(1)
    if last:
        beam_final = capture_beam_off()
elapsed = time.perf_counter() - t0
stats = np.array([beam_final["p_z"].mean(), np.sqrt(np.mean(beam_final["r"] ** 2))])
np.save("stats_numba.npy", stats)
print(f"after evolution: mean p_z={stats[0]:.3f}  rms r={stats[1]:.4f}")
print(f"\nNUMBA (CPU): {N_T - 1} evolving steps in {elapsed:.3f}s -> {elapsed/(N_T-1)*1e3:.1f} ms/step",
      flush=True)
