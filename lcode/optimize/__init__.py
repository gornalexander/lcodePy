"""
Differentiable design-optimisation interface for 2D LCODE (JAX backend).

Define a differentiable beam builder `params -> beam`, pick an objective (a function of the
simulated `Trajectory`), and call `.minimize()` — `jax.grad` computes the gradient of the
objective through the whole multi-time-step, self-consistent simulation and Adam optimises it.

Example
-------
    import lcode
    from lcode.optimize import Optimizer, objectives, beams

    config = {'window-width': 5, 'transverse-step': 0.05,
              'window-length': 6, 'xi-step': 0.05, 'plasma-particles-per-cell': 10}
    beam_of = beams.transverse_gaussian(template)          # template: numba-generated driver
    opt = Optimizer(config, beam_of, objectives.constant_size(target=1.4), n_steps=6)
    res = opt.minimize({'sigma_r': 1.0, 'alpha': 5e-4}, iters=40, lr=0.1)
    res.params          # matched design
    res.trajectory()    # re-evaluate the evolution at the optimum

Note: optimisation requires the differentiable JAX backend (`lcode.jax`); the original numba
solver cannot provide gradients (it would need finite differences — see the gradient-cost
benchmark).
"""
from ._core import Optimizer, Trajectory, OptResult
from . import objectives, beams

__all__ = ["Optimizer", "Trajectory", "OptResult", "objectives", "beams"]
