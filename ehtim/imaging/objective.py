"""Stokes-I visibility objective with a NumPy/JAX backend switch.

The value kernels are backend-agnostic (they read numpy or jax.numpy from their
inputs). ``vis_objective`` returns a scipy ``fun(x) -> (f, g)`` for the active
backend: jax uses ``jax.grad`` (jitted, GPU-capable), numpy uses the analytic
``chisqgrad_vis``. The image is parameterized as ``imvec = exp(x)``.

Arrays: A (nvis, npix) complex128, x/imvec (npix,) float64, vis/sigma (nvis,).
The nvis axis leads so a later vmap over observations needs no kernel change.
"""
import functools

import numpy as np

from ehtim.backends import array_namespace, get_backend


def chisq_vis(imvec, A, vis, sigma):
    """Reduced visibility chi-squared, sum(|(A@imvec - vis)/sigma|^2) / (2*nvis)."""
    xp = array_namespace(imvec, A)
    return xp.sum(xp.abs((A @ imvec - vis) / sigma) ** 2) / (2 * vis.shape[0])


def loss_vis_log(x, A, vis, sigma):
    """Objective in log-image coordinates: chisq_vis(exp(x)) - 1."""
    xp = array_namespace(x)
    return chisq_vis(xp.exp(x), A, vis, sigma) - 1.0


@functools.cache
def _jitted_value_and_grad():
    """jit(value_and_grad(loss_vis_log)), built once and reused across calls."""
    import jax
    return jax.jit(jax.value_and_grad(loss_vis_log))


def vis_objective(A, vis, sigma, device=None):
    """scipy ``fun(x) -> (value, grad)`` for the active backend.

    On jax, A/vis/sigma are placed on ``device``. value/grad are host float64.
    """
    if get_backend() == "jax":
        import jax
        import jax.numpy as jnp

        A_d = jax.device_put(jnp.asarray(A, jnp.complex128), device)
        vis_d = jax.device_put(jnp.asarray(vis, jnp.complex128), device)
        sigma_d = jax.device_put(jnp.asarray(sigma, jnp.float64), device)
        value_and_grad = _jitted_value_and_grad()

        def fun(x):
            value, grad = value_and_grad(jnp.asarray(x), A_d, vis_d, sigma_d)
            return float(value), np.asarray(grad, dtype=np.float64)

        return fun

    from ehtim.imaging.imager_backend import transform_gradients
    from ehtim.imaging.imager_utils import chisqgrad_vis

    def fun(x):
        value = float(loss_vis_log(x, A, vis, sigma))
        grad = transform_gradients(
            chisqgrad_vis(np.exp(x), A, vis, sigma), x, ["log"], np.array([1]),
        )
        return value, np.asarray(grad, dtype=np.float64)

    return fun
