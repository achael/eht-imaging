"""Direct Stokes-I chi-squared objectives with a NumPy/JAX backend switch.

The value kernels are backend-agnostic (they read numpy or jax.numpy from their
inputs). ``objective(dtype, A, data, sigma)`` returns a scipy ``fun(x) -> (f, g)``
for the active backend: jax uses ``jax.grad`` (jitted, GPU-capable), numpy uses
the analytic ``chisqgrad_*``. The image is parameterized as ``imvec = exp(x)``.

``dtype`` is one of vis, amp, bs, cphase, camp, logcamp (ttype='direct', Stokes I).
``A`` is a single (nvis, npix) matrix (vis/amp) or a tuple of 3 (bs/cphase) / 4
(camp/logcamp) matrices; the namespace is read from ``imvec``, so tuple-A is fine.
"""
import functools

import numpy as np

import ehtim.const_def as ehc
from ehtim.backends import array_namespace, get_backend
from ehtim.imaging.imager_backend import transform_gradients
from ehtim.imaging.imager_utils import (
    chisqgrad_amp,
    chisqgrad_bs,
    chisqgrad_camp,
    chisqgrad_cphase,
    chisqgrad_logcamp,
    chisqgrad_vis,
)


def chisq_vis(imvec, A, vis, sigma):
    """Visibility chi-squared, sum(|(A@imvec - vis)/sigma|^2) / (2*nvis)."""
    xp = array_namespace(imvec)
    return xp.sum(xp.abs((A @ imvec - vis) / sigma) ** 2) / (2 * vis.shape[0])


def chisq_amp(imvec, A, amp, sigma):
    """Amplitude chi-squared. A is a single (nvis, npix) matrix."""
    xp = array_namespace(imvec)
    amp_samples = xp.abs(A @ imvec)
    return xp.sum(xp.abs((amp - amp_samples) / sigma) ** 2) / amp.shape[0]


def chisq_bs(imvec, A, bis, sigma):
    """Bispectrum chi-squared. A is a 3-tuple of (nbs, npix) matrices."""
    xp = array_namespace(imvec)
    bisamples = (A[0] @ imvec) * (A[1] @ imvec) * (A[2] @ imvec)
    return xp.sum(xp.abs((bis - bisamples) / sigma) ** 2) / (2.0 * bis.shape[0])


def chisq_cphase(imvec, A, clphase, sigma):
    """Closure-phase chi-squared (data in degrees). A is a 3-tuple."""
    xp = array_namespace(imvec)
    clphase = clphase * ehc.DEGREE
    sigma = sigma * ehc.DEGREE
    clphase_samples = xp.angle((A[0] @ imvec) * (A[1] @ imvec) * (A[2] @ imvec))
    return (2.0 / clphase.shape[0]) * xp.sum((1.0 - xp.cos(clphase - clphase_samples)) / sigma**2)


def chisq_camp(imvec, A, clamp, sigma):
    """Closure-amplitude chi-squared. A is a 4-tuple."""
    xp = array_namespace(imvec)
    i1, i2, i3, i4 = (A[k] @ imvec for k in range(4))
    clamp_samples = xp.abs((i1 * i2) / (i3 * i4))
    return xp.sum(xp.abs((clamp - clamp_samples) / sigma) ** 2) / clamp.shape[0]


def chisq_logcamp(imvec, A, log_clamp, sigma):
    """Log-closure-amplitude chi-squared. A is a 4-tuple."""
    xp = array_namespace(imvec)
    a1, a2, a3, a4 = (xp.abs(A[k] @ imvec) for k in range(4))
    samples = xp.log(a1) + xp.log(a2) - xp.log(a3) - xp.log(a4)
    return xp.sum(xp.abs((log_clamp - samples) / sigma) ** 2) / log_clamp.shape[0]


_CHISQ = {
    "vis": chisq_vis, "amp": chisq_amp, "bs": chisq_bs,
    "cphase": chisq_cphase, "camp": chisq_camp, "logcamp": chisq_logcamp,
}
_CHISQGRAD = {
    "vis": chisqgrad_vis, "amp": chisqgrad_amp, "bs": chisqgrad_bs,
    "cphase": chisqgrad_cphase, "camp": chisqgrad_camp, "logcamp": chisqgrad_logcamp,
}


def _loss_log(dtype, x, A, data, sigma):
    """chisq[dtype](exp(x), ...) - 1 in log-image coordinates."""
    xp = array_namespace(x)
    return _CHISQ[dtype](xp.exp(x), A, data, sigma) - 1.0


def loss_vis_log(x, A, vis, sigma):
    """Visibility objective in log-image coordinates: chisq_vis(exp(x)) - 1."""
    return _loss_log("vis", x, A, vis, sigma)


@functools.cache
def _jitted_value_and_grad(dtype):
    """Per-dtype jit(value_and_grad(loss)); built once, dtype baked into the closure."""
    import jax
    return jax.jit(jax.value_and_grad(functools.partial(_loss_log, dtype)))


def _device_put_A(A, device):
    """device_put A (single matrix or tuple of matrices) as complex128."""
    import jax
    import jax.numpy as jnp
    if isinstance(A, (tuple, list)):
        return tuple(jax.device_put(jnp.asarray(a, jnp.complex128), device) for a in A)
    return jax.device_put(jnp.asarray(A, jnp.complex128), device)


def objective(dtype, A, data, sigma, device=None):
    """scipy ``fun(x) -> (value, grad)`` for one direct Stokes-I data term.

    On jax the gradient is autodiff (jitted value_and_grad) and A/data/sigma are
    placed on ``device``; on numpy it is the analytic ``chisqgrad_*``. value/grad
    are host float64.
    """
    if dtype not in _CHISQ:
        raise ValueError(f"dtype must be one of {sorted(_CHISQ)}, got {dtype!r}")

    if get_backend() == "jax":
        import jax
        import jax.numpy as jnp

        A_d = _device_put_A(A, device)
        data_d = jax.device_put(jnp.asarray(data), device)
        sigma_d = jax.device_put(jnp.asarray(sigma, jnp.float64), device)
        value_and_grad = _jitted_value_and_grad(dtype)

        def fun(x):
            value, grad = value_and_grad(jnp.asarray(x), A_d, data_d, sigma_d)
            return float(value), np.asarray(grad, dtype=np.float64)

        return fun

    grad_fn = _CHISQGRAD[dtype]

    def fun(x):
        value = float(_loss_log(dtype, x, A, data, sigma))
        grad = transform_gradients(grad_fn(np.exp(x), A, data, sigma), x, ["log"], np.array([1]))
        return value, np.asarray(grad, dtype=np.float64)

    return fun


def vis_objective(A, vis, sigma, device=None):
    """scipy ``fun(x) -> (value, grad)`` for the visibility term."""
    return objective("vis", A, vis, sigma, device)
