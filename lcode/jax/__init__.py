"""
Differentiable JAX backend for 2D LCODE.

A pure-JAX reimplementation of the 2D quasistatic solver + self-consistent beam dynamics that
reproduces the numba kernels to machine precision and is end-to-end differentiable (jax.grad),
jit-compilable, and GPU-portable. Intended as the engine for gradient-based design optimisation
(see `lcode.optimize`).

Requires `jax` and `jax_enable_x64` (set on import to match numba float64).

Public API
----------
- `march`, `march_with_fields`, `step_dxi`, `FIELD_KEYS` : the plasma xi-march.
- `compute_rhoj`, `cell_volume`                          : plasma current deposition.
- `move_particles`                                       : plasma particle push.
- `compute_fields`                                       : plasma field solve.
- `deposit_beam_full`, `push_beam_layer`, `push_beam_history`, `init_substepping` : beam side.
- `one_time_step`                                        : one full quasistatic time step.
"""
import jax
jax.config.update("jax_enable_x64", True)

from . import field_solver, deposition, move, march, beam, dynamics, pipelined, pipelined_beam

from .march import march, march_with_fields, step_dxi, FIELD_KEYS
from .pipelined import wavefront_marches, sequential_marches
from .pipelined_beam import fused_time_step, sequential_fused_evolve, pipelined_evolve
from .deposition import compute_rhoj, cell_volume
from .move import move_particles
from .field_solver import compute_fields
from .beam import (deposit_beam_layer, deposit_beam_full, push_beam_layer,
                   init_substepping)
from .dynamics import one_time_step, push_beam_history

__all__ = [
    "field_solver", "deposition", "move", "march", "beam", "dynamics",
    "march", "march_with_fields", "step_dxi", "FIELD_KEYS",
    "compute_rhoj", "cell_volume", "move_particles", "compute_fields",
    "deposit_beam_layer", "deposit_beam_full", "push_beam_layer",
    "init_substepping", "one_time_step", "push_beam_history",
    "pipelined", "wavefront_marches", "sequential_marches",
    "pipelined_beam", "fused_time_step", "sequential_fused_evolve", "pipelined_evolve",
]
