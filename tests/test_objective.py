"""Tests for the direct Stokes-I chisq objectives and their NumPy/JAX gradients.

Each term's JAX gradient is checked three ways (autodiff == analytic == finite
difference) in float64, plus jit correctness, backend parity, and GPU execution.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ehtim.imaging.imager_utils as iu
from ehtim.backends import set_backend
from ehtim.imaging import objective
from ehtim.imaging.imager_backend import transform_gradients
from ehtim.imaging.imager_utils import chisqdata

pytestmark = pytest.mark.jax

DTYPES = ["vis", "amp", "bs", "cphase", "camp", "logcamp"]
RECOVER_DTYPES = ["vis", "bs"]   # only vis recovers structure; bs checks the host-shim drives L-BFGS

KERNEL_ATOL = 1e-12
# Closure-term gradients (bs/cphase/camp/logcamp) span large magnitudes, so
# gradient comparisons use a relative tolerance, not an absolute one alone.
GRAD_RTOL = 1e-9
GRAD_ATOL = 1e-12
PARITY_ATOL = 1e-10
FD_STEP = 1e-6
FD_MEDIAN_TOL = 1e-3
FD_MAX_TOL = 1e-2
N_FD_SAMPLES = 40
NXCORR_FLOOR = 0.95
RNG_SEED = 4
PERTURB = 0.10


def _has_gpu():
    try:
        return len(jax.devices("gpu")) > 0
    except RuntimeError:
        return False


requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="no CUDA GPU available")


def _nxcorr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


@pytest.fixture(autouse=True)
def _reset_backend():
    yield
    set_backend("numpy")


@pytest.fixture(scope="module", params=DTYPES)
def term_data(request, obs_direct, gauss_im):
    # x = log of a 10%-perturbed image, so the gradient is nonzero (it vanishes
    # at the truth, which generated these noise-free observations).
    dtype = request.param
    truth = gauss_im.imvec.astype(np.float64)
    mask = np.ones(truth.size, dtype=bool)
    data, sigma, A = chisqdata(obs_direct, gauss_im, mask, dtype, ttype="direct")
    rng = np.random.default_rng(RNG_SEED)
    imvec = np.clip(truth * (1.0 + PERTURB * rng.standard_normal(truth.size)), 1e-12, None)
    return dtype, np.log(imvec), A, data, sigma, imvec, truth


def test_numpy_kernel_matches_reference(term_data):
    dtype, _, A, data, sigma, imvec, _ = term_data
    got = objective._CHISQ[dtype](imvec, A, data, sigma)
    ref = getattr(iu, "chisq_" + dtype)(imvec, A, data, sigma)
    assert np.allclose(got, ref, atol=KERNEL_ATOL, rtol=0)


def test_three_way_gradient(term_data):
    dtype, x, A, data, sigma, imvec, _ = term_data

    set_backend("jax")
    g_jax = objective.objective(dtype, A, data, sigma)(x)[1]

    set_backend("numpy")
    g_analytic = transform_gradients(
        objective._CHISQGRAD[dtype](imvec, A, data, sigma), x, ["log"], np.array([1]),
    )
    assert np.allclose(g_jax, g_analytic, rtol=GRAD_RTOL, atol=GRAD_ATOL)

    idx = np.argsort(imvec)[-N_FD_SAMPLES:]  # brightest pixels are well-conditioned
    g_fd = np.empty(N_FD_SAMPLES)
    for k, j in enumerate(idx):
        xp, xm = x.copy(), x.copy()
        xp[j] += FD_STEP
        xm[j] -= FD_STEP
        fp = float(objective._loss_log(dtype, xp, A, data, sigma))
        fm = float(objective._loss_log(dtype, xm, A, data, sigma))
        g_fd[k] = (fp - fm) / (2 * FD_STEP)
    frac = np.abs((g_fd - g_analytic[idx]) / (np.abs(g_analytic[idx]) + 1e-100))
    assert np.median(frac) < FD_MEDIAN_TOL
    assert np.max(frac) < FD_MAX_TOL


def test_backend_switch_parity(term_data):
    dtype, x, A, data, sigma, _, _ = term_data
    set_backend("numpy")
    f_np, g_np = objective.objective(dtype, A, data, sigma)(x)
    set_backend("jax")
    f_jx, g_jx = objective.objective(dtype, A, data, sigma)(x)
    assert np.allclose(f_np, f_jx, rtol=GRAD_RTOL, atol=PARITY_ATOL)
    assert np.allclose(g_np, g_jx, rtol=GRAD_RTOL, atol=PARITY_ATOL)


def test_x64_holds_through_jit(term_data):
    dtype, x, A, data, sigma, _, _ = term_data
    assert jnp.ones(1).dtype == jnp.float64
    A_d = objective._device_put_A(A, None)
    value, grad = objective._jitted_value_and_grad(dtype)(
        jnp.asarray(x), A_d, jnp.asarray(data), jnp.asarray(sigma),
    )
    assert value.dtype == jnp.float64
    assert grad.dtype == jnp.float64
    leaves = A_d if isinstance(A_d, tuple) else (A_d,)
    assert all(a.dtype == jnp.complex128 for a in leaves)


def test_jit_matches_eager(term_data):
    dtype, x, A, data, sigma, _, _ = term_data
    loss = functools.partial(objective._loss_log, dtype)
    A_d = objective._device_put_A(A, None)
    xj, dj, sj = jnp.asarray(x), jnp.asarray(data), jnp.asarray(sigma)
    eager = loss(xj, A_d, dj, sj)
    jitted = jax.jit(loss)(xj, A_d, dj, sj)
    assert np.allclose(eager, jitted, rtol=GRAD_RTOL, atol=KERNEL_ATOL)


def test_no_retrace_same_shapes(term_data):
    # also pins tuple-A (bs/cphase/camp/logcamp) as a jax pytree that traces once.
    dtype, x, A, data, sigma, _, _ = term_data
    traces = {"n": 0}
    base = functools.partial(objective._loss_log, dtype)

    def counted(x, A, data, sigma):
        traces["n"] += 1
        return base(x, A, data, sigma)

    jf = jax.jit(counted)
    A_d = objective._device_put_A(A, None)
    xj, dj, sj = jnp.asarray(x), jnp.asarray(data), jnp.asarray(sigma)
    jf(xj, A_d, dj, sj).block_until_ready()
    jf(xj + 1.0, A_d, dj, sj).block_until_ready()
    assert traces["n"] == 1


@pytest.mark.slow
@pytest.mark.parametrize("which", ["numpy", "jax"])
@pytest.mark.parametrize("dtype", RECOVER_DTYPES)
def test_lbfgsb_reduces_objective(obs_direct, gauss_im, dtype, which):
    # Exercises the scipy host-shim end to end (gradient correctness is isolated
    # in test_three_way_gradient). Only vis constrains absolute image structure.
    import scipy.optimize as opt

    truth = gauss_im.imvec.astype(np.float64)
    mask = np.ones(truth.size, dtype=bool)
    data, sigma, A = chisqdata(obs_direct, gauss_im, mask, dtype, ttype="direct")
    set_backend(which)
    fun = objective.objective(dtype, A, data, sigma)
    x0 = np.log(np.full_like(truth, truth.mean()))
    f0 = fun(x0)[0]
    res = opt.minimize(fun, x0, method="L-BFGS-B", jac=True, options={"maxiter": 1000})
    assert res.fun < f0
    if dtype == "vis":   # only vis constrains absolute image structure
        assert _nxcorr(np.exp(res.x), truth) >= NXCORR_FLOOR


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("dtype", ["vis", "bs"])
def test_grad_runs_on_gpu(obs_direct, gauss_im, dtype):
    truth = gauss_im.imvec.astype(np.float64)
    mask = np.ones(truth.size, dtype=bool)
    data, sigma, A = chisqdata(obs_direct, gauss_im, mask, dtype, ttype="direct")
    gpu = jax.devices("gpu")[0]
    x = np.log(np.clip(truth, 1e-12, None))
    _, grad = objective._jitted_value_and_grad(dtype)(
        jax.device_put(jnp.asarray(x), gpu),
        objective._device_put_A(A, gpu),
        jax.device_put(jnp.asarray(data), gpu),
        jax.device_put(jnp.asarray(sigma), gpu),
    )
    assert any(d.platform == "gpu" for d in grad.devices())


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("dtype", ["vis", "bs"])
def test_gpu_cpu_parity(obs_direct, gauss_im, dtype):
    truth = gauss_im.imvec.astype(np.float64)
    mask = np.ones(truth.size, dtype=bool)
    data, sigma, A = chisqdata(obs_direct, gauss_im, mask, dtype, ttype="direct")
    x = np.log(np.clip(truth, 1e-12, None))
    set_backend("jax")
    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    f_c, g_c = objective.objective(dtype, A, data, sigma, device=cpu)(x)
    f_g, g_g = objective.objective(dtype, A, data, sigma, device=gpu)(x)
    assert np.allclose(f_c, f_g, rtol=GRAD_RTOL, atol=PARITY_ATOL)
    assert np.allclose(g_c, g_g, rtol=GRAD_RTOL, atol=PARITY_ATOL)
    assert np.all(np.isfinite(g_g))
