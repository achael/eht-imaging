"""Tests for the Stokes-I visibility objective and its NumPy/JAX gradients.

The JAX gradient is checked three ways (autodiff == analytic == finite difference)
in float64, plus jit correctness, backend parity, and GPU execution.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ehtim.backends import set_backend
from ehtim.imaging import objective
from ehtim.imaging.imager_backend import transform_gradients
from ehtim.imaging.imager_utils import chisq_vis as np_chisq_vis
from ehtim.imaging.imager_utils import chisqdata, chisqgrad_vis

pytestmark = pytest.mark.jax

KERNEL_ATOL = 1e-12
PARITY_ATOL = 1e-10
GRAD_RTOL = 1e-9
GRAD_ATOL = 1e-12
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


@pytest.fixture(scope="module")
def vis_data(obs_direct, gauss_im):
    # x = log of a 10%-perturbed image, so the gradient is nonzero (it vanishes
    # at the truth, which generated these noise-free visibilities).
    truth = gauss_im.imvec.astype(np.float64)
    mask = np.ones(truth.size, dtype=bool)
    vis, sigma, A = chisqdata(obs_direct, gauss_im, mask, "vis", ttype="direct")
    rng = np.random.default_rng(RNG_SEED)
    imvec = np.clip(truth * (1.0 + PERTURB * rng.standard_normal(truth.size)), 1e-12, None)
    return np.log(imvec), A, vis, sigma, imvec, truth


def test_numpy_kernel_matches_reference(vis_data):
    _, A, vis, sigma, imvec, _ = vis_data
    got = objective.chisq_vis(imvec, A, vis, sigma)
    ref = np_chisq_vis(imvec, A, vis, sigma)
    assert np.allclose(got, ref, atol=KERNEL_ATOL, rtol=0)


def test_three_way_gradient(vis_data):
    x, A, vis, sigma, imvec, _ = vis_data

    set_backend("jax")
    g_jax = objective.vis_objective(A, vis, sigma)(x)[1]

    set_backend("numpy")
    g_analytic = transform_gradients(
        chisqgrad_vis(imvec, A, vis, sigma), x, ["log"], np.array([1]),
    )
    assert np.allclose(g_jax, g_analytic, rtol=GRAD_RTOL, atol=GRAD_ATOL)

    idx = np.argsort(imvec)[-N_FD_SAMPLES:]  # brightest pixels are well-conditioned
    g_fd = np.empty(N_FD_SAMPLES)
    for k, j in enumerate(idx):
        xp, xm = x.copy(), x.copy()
        xp[j] += FD_STEP
        xm[j] -= FD_STEP
        fp = float(objective.loss_vis_log(xp, A, vis, sigma))
        fm = float(objective.loss_vis_log(xm, A, vis, sigma))
        g_fd[k] = (fp - fm) / (2 * FD_STEP)
    frac = np.abs((g_fd - g_analytic[idx]) / (np.abs(g_analytic[idx]) + 1e-100))
    assert np.median(frac) < FD_MEDIAN_TOL
    assert np.max(frac) < FD_MAX_TOL


def test_backend_switch_parity(vis_data):
    x, A, vis, sigma, _, _ = vis_data
    set_backend("numpy")
    f_np, g_np = objective.vis_objective(A, vis, sigma)(x)
    set_backend("jax")
    f_jx, g_jx = objective.vis_objective(A, vis, sigma)(x)
    assert np.allclose(f_np, f_jx, atol=PARITY_ATOL, rtol=0)
    assert np.allclose(g_np, g_jx, atol=PARITY_ATOL, rtol=0)


def test_x64_holds_through_jit(vis_data):
    x, A, vis, sigma, _, _ = vis_data
    assert jnp.ones(1).dtype == jnp.float64
    vg = jax.jit(jax.value_and_grad(objective.loss_vis_log))
    value, grad = vg(jnp.asarray(x), jnp.asarray(A), jnp.asarray(vis), jnp.asarray(sigma))
    assert value.dtype == jnp.float64
    assert grad.dtype == jnp.float64
    assert jnp.asarray(A).dtype == jnp.complex128


def test_jit_matches_eager(vis_data):
    x, A, vis, sigma, _, _ = vis_data
    xj, Aj, vj, sj = (jnp.asarray(z) for z in (x, A, vis, sigma))
    eager = objective.loss_vis_log(xj, Aj, vj, sj)
    jitted = jax.jit(objective.loss_vis_log)(xj, Aj, vj, sj)
    assert np.allclose(eager, jitted, atol=KERNEL_ATOL, rtol=0)


def test_no_retrace_same_shapes(vis_data):
    x, A, vis, sigma, _, _ = vis_data
    traces = {"n": 0}

    def counted(x, A, vis, sigma):
        traces["n"] += 1
        return objective.loss_vis_log(x, A, vis, sigma)

    jf = jax.jit(counted)
    xj, Aj, vj, sj = (jnp.asarray(z) for z in (x, A, vis, sigma))
    jf(xj, Aj, vj, sj).block_until_ready()
    jf(xj + 1.0, Aj, vj, sj).block_until_ready()
    assert traces["n"] == 1


@pytest.mark.slow
@pytest.mark.parametrize("which", ["numpy", "jax"])
def test_lbfgsb_recovers_image(vis_data, which):
    import scipy.optimize as opt

    _, A, vis, sigma, _, truth = vis_data
    set_backend(which)
    fun = objective.vis_objective(A, vis, sigma)
    x0 = np.log(np.full_like(truth, truth.mean()))
    f0 = fun(x0)[0]
    res = opt.minimize(fun, x0, method="L-BFGS-B", jac=True, options={"maxiter": 1000})
    assert res.fun < f0
    assert _nxcorr(np.exp(res.x), truth) >= NXCORR_FLOOR


@pytest.mark.gpu
@requires_gpu
def test_grad_runs_on_gpu(vis_data):
    x, A, vis, sigma, _, _ = vis_data
    gpu = jax.devices("gpu")[0]
    vg = jax.jit(jax.value_and_grad(objective.loss_vis_log))
    _, grad = vg(
        jax.device_put(jnp.asarray(x), gpu),
        jax.device_put(jnp.asarray(A), gpu),
        jax.device_put(jnp.asarray(vis), gpu),
        jax.device_put(jnp.asarray(sigma), gpu),
    )
    assert any(d.platform == "gpu" for d in grad.devices())


@pytest.mark.gpu
@requires_gpu
def test_gpu_cpu_parity(vis_data):
    x, A, vis, sigma, _, _ = vis_data
    set_backend("jax")
    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    f_c, g_c = objective.vis_objective(A, vis, sigma, device=cpu)(x)
    f_g, g_g = objective.vis_objective(A, vis, sigma, device=gpu)(x)
    assert np.allclose(f_c, f_g, atol=PARITY_ATOL)
    assert np.allclose(g_c, g_g, atol=PARITY_ATOL)


@pytest.mark.gpu
@requires_gpu
def test_gpu_value_and_grad_finite(vis_data):
    x, A, vis, sigma, _, _ = vis_data
    set_backend("jax")
    f, g = objective.vis_objective(A, vis, sigma, device=jax.devices("gpu")[0])(x)
    assert np.isfinite(f)
    assert np.all(np.isfinite(g))
