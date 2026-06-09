"""NumPy / JAX backend selection for the imaging kernels.

``set_backend("jax")`` runs the kernels on JAX (GPU, autodiff); the default
"numpy" backend is unchanged. Kernels pick their array module from their inputs
(``array_namespace``), so they stay single-sourced and jit-safe. JAX is imported
lazily when selected; install it with the ``[dev]`` (CPU) or ``[gpu]`` (CUDA) extra.
"""
import contextlib
import importlib.util

import numpy as _np

_BACKENDS = ("numpy", "jax")
_active = "numpy"
# Whether jax is installed, resolved once -- find_spec does not import jax, so
# `import ehtim` stays jax-free, and array_namespace avoids a per-call try/except
# on the numpy hot path.
_HAS_JAX = importlib.util.find_spec("jax") is not None


def set_backend(name):
    """Select the active imaging backend, "numpy" or "jax"."""
    global _active
    if name not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {name!r}")
    if name == "jax":
        import jax
        jax.config.update("jax_enable_x64", True)
    _active = name


def get_backend():
    """Return the name of the active imaging backend."""
    return _active


@contextlib.contextmanager
def backend(name):
    """Select a backend for the duration of a ``with`` block, then restore."""
    previous = get_backend()
    set_backend(name)
    try:
        yield
    finally:
        set_backend(previous)


def array_namespace(*arrays):
    """Return jax.numpy if any argument is a JAX array or tracer, else numpy."""
    if not _HAS_JAX:
        return _np
    import jax
    if any(isinstance(a, jax.Array) for a in arrays):
        import jax.numpy as jnp
        return jnp
    return _np
