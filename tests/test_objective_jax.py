"""JAX parity tests for the full imaging objective (compute_objective).

unpack_imarr and transform_imarr are now functional, so jax.grad(compute_objective)
is the jax objective gradient. These check it equals the hand-written
compute_objective_grad (the gold standard), matches finite differences, and that
the value agrees numpy<->jax. Stokes-I, ttype='direct' (the #291 kernel scope).

The Imager wrappers objfunc/objgrad call compute_objective/compute_objective_grad,
so jax.grad(imgr.objfunc) is exactly the autodiff objective gradient.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ehtim as eh
from ehtim.imaging.imager_backend import make_objective_jax

pytestmark = pytest.mark.jax

# rtol, not atol: XLA reorders the float summation, so terms differ from numpy
# at ~1e-14 relative; a real bug fails by orders of magnitude.
VALUE_RTOL = 1e-9
GRAD_RTOL = 1e-9
# nfft is an approximate transform: jax_finufft's VJP matches the analytic adjoint
# only to the NUFFT accuracy. The residual scales with eps (verified 1e-9 -> ~1e-6,
# 1e-12 -> ~7e-10 max), so the nfft fixture uses eps=1e-12 and this looser-but-tight
# rtol; a real VJP bug is eps-independent and fails by orders of magnitude.
NFFT_EPS = 1e-12
NFFT_GRAD_RTOL = 1e-7
GRAD_ATOL = 1e-12
FD_STEP = 1e-6
FD_MEDIAN_TOL = 1e-3
FD_MAX_TOL = 1e-2
N_FD_SAMPLES = 40
RNG_SEED = 4
PERTURB = 0.10

DATA_TERM = {"amp": 100, "cphase": 100, "logcamp": 50}
REG_TERM = {"simple": 1, "flux": 1}


@pytest.fixture(scope="module", params=["direct", "nfft"])
def imager(request, obs_direct, obs_nfft, gauss_im, gauss_prior):
    ttype = request.param
    obs = obs_direct if ttype == "direct" else obs_nfft
    extra = {} if ttype == "direct" else {"nfft_eps": NFFT_EPS}
    imgr = eh.imager.Imager(
        obs, gauss_prior, prior_im=gauss_prior, flux=gauss_im.total_flux(),
        data_term=DATA_TERM, reg_term=REG_TERM, ttype=ttype, pol="I", maxit=100, **extra,
    )
    imgr.init_imager()
    return imgr


def _grad_rtol(imgr):
    return GRAD_RTOL if imgr._config.ttype == "direct" else NFFT_GRAD_RTOL


@pytest.fixture(scope="module")
def x0(imager):
    # perturb the initial solver vector so the gradient is nonzero
    rng = np.random.default_rng(RNG_SEED)
    base = np.asarray(imager._init_vec, dtype=np.float64)
    return base + PERTURB * rng.standard_normal(base.size)


def test_value_numpy_jax_consistent(imager, x0):
    v_np = float(imager.objfunc(x0))
    v_jx = float(imager.objfunc(jnp.asarray(x0)))
    assert np.allclose(v_np, v_jx, rtol=VALUE_RTOL, atol=GRAD_ATOL)


def test_grad_parity_autodiff_vs_analytic(imager, x0):
    # the gold standard: jax.grad(compute_objective) == compute_objective_grad
    g_analytic = np.asarray(imager.objgrad(x0))
    g_jax = np.asarray(jax.grad(imager.objfunc)(jnp.asarray(x0)))
    assert np.allclose(g_jax, g_analytic, rtol=_grad_rtol(imager), atol=GRAD_ATOL)


def test_grad_finite_difference(imager, x0):
    g_analytic = np.asarray(imager.objgrad(x0))
    idx = np.argsort(np.abs(g_analytic))[-N_FD_SAMPLES:]  # best-conditioned pixels
    g_fd = np.empty(N_FD_SAMPLES)
    for k, j in enumerate(idx):
        xp_, xm_ = x0.copy(), x0.copy()
        xp_[j] += FD_STEP
        xm_[j] -= FD_STEP
        g_fd[k] = (float(imager.objfunc(xp_)) - float(imager.objfunc(xm_))) / (2 * FD_STEP)
    frac = np.abs((g_fd - g_analytic[idx]) / (np.abs(g_analytic[idx]) + 1e-100))
    assert np.median(frac) < FD_MEDIAN_TOL
    assert np.max(frac) < FD_MAX_TOL


def _make_fun(imgr, device=None):
    return make_objective_jax(
        imgr._init_arr, imgr._config, imgr._which_solve, imgr._data_tuples,
        imgr._logfreqratio_list, len(imgr.obslist_next), imgr.dat_term_next,
        imgr.reg_term_next, imgr._prior_arr, imgr.norm_reg, imgr._regparams(),
        imgr._embed_mask, device=device,
    )


def test_factory_matches_analytic(imager, x0):
    value, grad = _make_fun(imager)(x0)
    assert np.allclose(value, float(imager.objfunc(x0)), rtol=VALUE_RTOL, atol=GRAD_ATOL)
    assert np.allclose(grad, np.asarray(imager.objgrad(x0)), rtol=_grad_rtol(imager), atol=GRAD_ATOL)


def test_objective_traces_once_across_x(imager, x0):
    # the factory closes over everything but x, so one jit trace serves every
    # solver step; here we count traces of objfunc (what the factory jits).
    traces = {"n": 0}

    def counted(x):
        traces["n"] += 1
        return imager.objfunc(x)

    f = jax.jit(counted)
    f(jnp.asarray(x0)).block_until_ready()
    f(jnp.asarray(x0 + 0.5)).block_until_ready()
    assert traces["n"] == 1
