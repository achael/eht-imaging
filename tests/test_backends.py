"""Tests for the NumPy/JAX backend switch in ehtim.backends."""
import jax.numpy as jnp
import numpy as np
import pytest

from ehtim.backends import array_namespace, backend, get_backend, set_backend

pytestmark = pytest.mark.jax


@pytest.fixture(autouse=True)
def _reset_backend():
    yield
    set_backend("numpy")


def test_default_backend_is_numpy():
    assert get_backend() == "numpy"


def test_set_backend_round_trip():
    set_backend("jax")
    assert get_backend() == "jax"
    set_backend("numpy")
    assert get_backend() == "numpy"


def test_backend_context_manager_restores():
    set_backend("jax")
    with backend("numpy"):
        assert get_backend() == "numpy"
    assert get_backend() == "jax"


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        set_backend("torch")


def test_array_namespace_dispatches_on_input():
    assert array_namespace(np.ones(3)) is np
    assert array_namespace(jnp.ones(3)) is jnp
    assert array_namespace(np.ones(3), jnp.ones(3)) is jnp
