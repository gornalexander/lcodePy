"""
JAX push-solver: the JAX backend's own time-step driver (no MPI — MPI is numba's cross-process
pipeline; JAX parallelises on-device).

`run` evolves the beam for N time steps entirely on-device: the beam dict is threaded from step
to step without a host round-trip, and the per-layer wake-field history (needed on-device to push
the beam) is only copied to the host when diagnostics ask for it. Substepping is re-initialised
each step, matching numba (numba's beam source resets dt=0 every step).

`Simulation.step()` calls `run` for the JAX backend; the plasma state is exposed on
`_plasmastate` / `_field_history` in numba `Arrays` format, like the numba push-solver.
"""
import numpy as np
import jax

from ..plasma.data import Arrays
from . import FIELD_KEYS, init_substepping
from .backend import _build_solver_state, _beam_dict, _one_step, _to_arrays


class JaxPusherAndSolver2D:
    """JAX time-step driver behind the same public surface as `PusherAndSolver2D`."""

    def __init__(self, config):
        self.config = config
        self.dxi = config.getfloat('xi-step')
        self.time_step = config.getfloat('time-step')
        self.substep_energy = config.getfloat('beam-substepping-energy')
        # 'wavefront' (default: on-device time-step pipeline) or 'sequential' (fallback, and used
        # automatically when diagnostics or the field-history flag are active, since the wavefront
        # keeps no per-step history).
        self.pipeline = config.get('pipeline', 'wavefront')
        self.save_field_history = config.getbool('save-field-history')
        self._plasma, self._params = _build_solver_state(config)
        self._step_cache = {}                # jitted one-step per substep depth
        self._plasmastate = None
        self._field_history = None

    def warmup(self, plasma_state=None, rank: int = 0):
        """No-op: JAX jit-compiles lazily on the first step."""

    def _step_fn(self, max_substeps):
        if max_substeps not in self._step_cache:
            self._step_cache[max_substeps] = jax.jit(
                lambda beam, dt: _one_step(beam, self._plasma, self._params, dt, max_substeps))
        return self._step_cache[max_substeps]

    def run(self, n_steps, beam_array, diagnostics_list, current_time):
        """Evolve `beam_array` for `n_steps` time steps on-device; store the final plasma state."""
        keep_history = bool(diagnostics_list) or self.save_field_history
        if self.pipeline == 'wavefront':
            self._run_wavefront(n_steps, beam_array, diagnostics_list, current_time, keep_history)
            return
        beam, _ = _beam_dict(beam_array)          # jax arrays, stays on-device across steps
        t = current_time
        electrons = fields = currents = ions = None

        for _ in range(n_steps):
            zero = jax.numpy.zeros_like(beam["p_z"])
            dt, remaining_steps = init_substepping(beam["p_z"], beam["q_m"], zero, zero,
                                                   self.time_step, self.substep_energy)
            max_substeps = int(np.asarray(remaining_steps).max())
            beam["remaining_steps"] = remaining_steps
            beam, (electrons, fields, currents, ions), history = \
                self._step_fn(max_substeps)(beam, dt)
            if keep_history:
                history_host = {k: np.asarray(history[k]) for k in FIELD_KEYS}
                self._field_history = history_host
                if diagnostics_list:
                    self._run_diagnostics(diagnostics_list, t, electrons, currents, history_host)
            t += self.time_step

        jax.block_until_ready(electrons["r"])
        particles = {"electrons": _to_arrays(electrons)}
        if ions is not None:
            particles["ions"] = _to_arrays(ions)
        plasma_fields = Arrays(xp=np, **{k: np.asarray(fields[k]) for k in FIELD_KEYS})
        plasma_currents = Arrays(xp=np, **{k: np.asarray(currents[k])
                                           for k in ("rho", "j_r", "j_f", "j_z")})
        self._plasmastate = (particles, plasma_fields, plasma_currents)
        if not keep_history:
            self._field_history = None

    def _run_wavefront(self, n_steps, beam_array, diagnostics_list, current_time, keep_history):
        """Evolve the beam as a diagonal wavefront over time steps (on-device pipeline, #19).

        With `keep_history` the per-(time step, xi-layer) wake fields are returned too, so
        diagnostics and `_field_history` work on the wavefront just like the sequential driver.
        """
        import jax.numpy as jnp
        from . import init_substepping
        from .beam_layered import bucket_beam, wavefront_evolve

        bucket, _ = bucket_beam(beam_array, self._params["n_xi"], self.dxi)
        zero = jnp.zeros_like(bucket["p_z"])
        _, remaining = init_substepping(bucket["p_z"], bucket["q_m"], zero, zero,
                                        self.time_step, self.substep_energy)
        remaining = jnp.where(bucket["active"], remaining, 0)
        max_substeps = int(np.asarray(remaining).max())
        ions = self._plasma.get("ions")

        _evolved, final, history = wavefront_evolve(
            bucket, self._plasma, self._params, self.time_step, self.substep_energy, n_steps,
            ions0=ions, max_substeps=max_substeps, keep_history=keep_history)
        if ions is None:
            electrons, fields, currents = final
            ion_sp = None
        else:
            electrons, ion_sp, fields, currents = final
        particles = {"electrons": _to_arrays(electrons)}
        if ion_sp is not None:
            particles["ions"] = _to_arrays(ion_sp)
        plasma_fields = Arrays(xp=np, **{k: np.asarray(fields[k]) for k in FIELD_KEYS})
        plasma_currents = Arrays(xp=np, **{k: np.asarray(currents[k])
                                           for k in ("rho", "j_r", "j_f", "j_z")})
        self._plasmastate = (particles, plasma_fields, plasma_currents)

        if history is None:
            self._field_history = None
            return
        # history[k]: [n_steps, n_layers, n_cells]. Expose the last step's history and, if any
        # diagnostics are active, replay them per time step (fields exact; final-lane particles).
        history_host = {k: np.asarray(history[k]) for k in FIELD_KEYS}
        self._field_history = {k: history_host[k][-1] for k in FIELD_KEYS}
        if diagnostics_list:
            for step in range(n_steps):
                step_hist = {k: history_host[k][step] for k in FIELD_KEYS}
                self._run_diagnostics(diagnostics_list, current_time + step * self.time_step,
                                      electrons, currents, step_hist)

    def _run_diagnostics(self, diagnostics_list, current_time, electrons, currents, history):
        """Replay per-xi-layer diagnostics from the wake-field history (fields-level).

        NOTE: JAX retains the wake fields at every layer but not the plasma particles per layer,
        so `after_step_dxi` receives the final-layer plasma state; field-vs-xi diagnostics
        (e.g. FXi) are exact, particle-slice diagnostics are not yet supported.
        """
        n_layers = history['E_z'].shape[0]
        particles = _to_arrays(electrons)
        currents_arr = Arrays(xp=np, **{k: np.asarray(currents[k])
                                        for k in ("rho", "j_r", "j_f", "j_z")})
        xi = 0.0
        for xi_i in range(n_layers):
            layer_fields = Arrays(xp=np, **{k: history[k][xi_i] for k in FIELD_KEYS})
            xi = -xi_i * self.dxi
            for diagnostic in diagnostics_list:
                diagnostic.after_step_dxi(current_time, xi, particles, layer_fields,
                                          currents_arr, None)
        last_fields = Arrays(xp=np, **{k: history[k][-1] for k in FIELD_KEYS})
        for diagnostic in diagnostics_list:
            diagnostic.dump(current_time, xi, particles, last_fields, currents_arr, None)
