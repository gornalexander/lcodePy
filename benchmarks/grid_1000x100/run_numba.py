"""
Backend 1 — numba LCODE, self-consistent evolving beam on the 1000 x 101 grid.

Runs the real `lcode.Simulation` (numba, parallel njit kernels) for n_t time steps with an
evolving beam. Captures, so the JAX backends can start from an identical state and be compared:
  - the INITIAL beam particles          -> beam_init.npz
  - the step-1 on-axis wake Ez(xi)      -> ez_numba.npy
  - beam <p_z>, rms r after evolution   -> stats_numba.npy
and times the n_t-step evolution.

IMPORTANT: the beam only chains across time steps *within a single* `sim.step(N)` call
(`step()` regenerates a fresh beam + rebuilds the transport every call). So we evolve with ONE
`sim.step(n_t)` and delineate the per-step beam by hooking `step_dt`. (Calling `sim.step(1)`
repeatedly would re-inject the initial beam each time -> a rigid, non-evolving driver.)

Capture is done by monkeypatching the numba beam-deposition, plasma-solver and step_dt — no
changes to lcode itself.
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
KEYS = ("id", "r", "xi", "p_z", "p_r", "M", "q_m", "q_norm")
print(f"grid NC={NC} NXI={NXI}  n_t={N_T}", flush=True)

# ---------------------------------------------------------------- capture hooks
_cap = {"cols": None}                                  # per-step beam buffer (None = off)
_orig_dep = nbeamcalc.BeamCalculator2D.deposit_beam_layer
def _dep(self, layer, xi_i):
    if _cap["cols"] is not None and layer.size:
        for k in _cap["cols"]:
            _cap["cols"][k].append(np.array(getattr(layer, k)))
    return _orig_dep(self, layer, xi_i)
nbeamcalc.BeamCalculator2D.deposit_beam_layer = _dep

_ez = {"arr": None, "k": 0, "on": False}
_orig_sdxi = CylindricalPlasmaSolver.step_dxi
def _sdxi(self, particles, fields, currents, ca, rb, rbp):
    particles, fields, currents = _orig_sdxi(self, particles, fields, currents, ca, rb, rbp)
    if _ez["on"] and _ez["k"] < NXI:
        _ez["arr"][_ez["k"]] = fields.E_z[0]; _ez["k"] += 1
    return particles, fields, currents
CylindricalPlasmaSolver.step_dxi = _sdxi

per_step = []                                          # one beam dict per time step
_capture = {"on": True}
_orig_sdt = nps.PusherAndSolver.step_dt
def _sdt(self, *a, **k):
    if not _capture["on"]:
        return _orig_sdt(self, *a, **k)
    _cap["cols"] = {kk: [] for kk in KEYS}
    r = _orig_sdt(self, *a, **k)
    per_step.append({kk: (np.concatenate(v) if v else np.array([])) for kk, v in _cap["cols"].items()})
    _cap["cols"] = None
    return r
nps.PusherAndSolver.step_dt = _sdt
nps.PusherAndSolver.warmup = lambda self, *a, **k: None      # skip built-in JIT warmup

sim = lcode.Simulation(config=dict(CONFIG), beam_parameters=BEAM, diagnostics=[], runas_filename="")

# --- warm up (compile numba kernels); discard captured beam
with contextlib.redirect_stdout(io.StringIO()):
    sim.step(1)
per_step.clear()

# --- ONE chained run of n_t steps; capture per-step beam + step-1 wake Ez(xi)
_ez["arr"] = np.empty(NXI); _ez["k"] = 0; _ez["on"] = True
with contextlib.redirect_stdout(io.StringIO()):
    sim.step(N_T)
_ez["on"] = False
ez_numba = _ez["arr"].copy()
beam_init = {k: per_step[0][k] for k in per_step[0]}          # 0 pushes
beam_final = per_step[N_T - 1]                                # after n_t-1 pushes
np.savez("beam_init.npz", **{k: beam_init[k] for k in ("r", "xi", "p_z", "p_r", "M", "q_m", "q_norm")})
np.save("ez_numba.npy", ez_numba)
stats = np.array([beam_final["p_z"].mean(), np.sqrt(np.mean(beam_final["r"] ** 2))])
np.save("stats_numba.npy", stats)
print(f"initial beam: {beam_init['r'].size} particles, mean p_z={beam_init['p_z'].mean():.1f}")
print(f"step-1 wake Ez(xi): min={ez_numba.min():.4e} max={ez_numba.max():.4e}")
print(f"after {N_T-1} evolving steps: mean p_z={stats[0]:.3f}  rms r={stats[1]:.4f}", flush=True)

# --- timing: another chained n_t-step run, capture off
_capture["on"] = False
t0 = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    sim.step(N_T)
elapsed = time.perf_counter() - t0
print(f"\nNUMBA (CPU): {N_T} chained steps in {elapsed:.3f}s -> {elapsed/N_T*1e3:.1f} ms/step",
      flush=True)
