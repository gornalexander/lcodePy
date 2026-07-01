"""A small library of ready-made objectives for `lcode.optimize.Optimizer`.

Each factory returns a callable `Trajectory -> scalar` to minimise.
"""
import jax
import jax.numpy as jnp


def constant_size(target=None):
    """Keep the rms transverse size constant (beam matching).

    If `target` is None, the size at the first step is used as the target (held fixed with
    a stop-gradient so the trivial 'shrink everything' solution is avoided).
    """
    def obj(traj):
        s = traj.rms_size()
        t = target if target is not None else jax.lax.stop_gradient(s[0])
        return jnp.mean((s - t) ** 2)
    return obj


def min_energy_spread():
    """Minimise the final relative energy spread."""
    def obj(traj):
        return traj.energy_spread()[-1]
    return obj


def match_wake(target_ez):
    """Match the final on-axis E_z(xi) to a target profile."""
    tt = jnp.asarray(target_ez)
    def obj(traj):
        return jnp.mean((traj.wake() - tt) ** 2)
    return obj


def max_wake():
    """Maximise the trailing wake amplitude (minimise its negative energy)."""
    def obj(traj):
        return -jnp.mean(traj.wake() ** 2)
    return obj


def target_size(target):
    """Drive the rms size to a target value and hold it there."""
    def obj(traj):
        return jnp.mean((traj.rms_size() - target) ** 2)
    return obj
