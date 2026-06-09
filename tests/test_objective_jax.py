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

pytestmark = pytest.mark.jax

# rtol, not atol: XLA reorders the float summation, so terms differ from numpy
# at ~1e-14 relative; a real bug fails by orders of magnitude.
VALUE_RTOL = 1e-9
GRAD_RTOL = 1e-9
GRAD_ATOL = 1e-12
FD_STEP = 1e-6
FD_MEDIAN_TOL = 1e-3
FD_MAX_TOL = 1e-2
N_FD_SAMPLES = 40
RNG_SEED = 4
PERTURB = 0.10

DATA_TERM = {"amp": 100, "cphase": 100, "logcamp": 50}
REG_TERM = {"simple": 1, "flux": 1}


@pytest.fixture(scope="module")
def imager_I(obs_direct, gauss_im, gauss_prior):
    imgr = eh.imager.Imager(
        obs_direct, gauss_prior, prior_im=gauss_prior, flux=gauss_im.total_flux(),
        data_term=DATA_TERM, reg_term=REG_TERM, ttype="direct", pol="I", maxit=100,
    )
    imgr.init_imager()
    return imgr


@pytest.fixture(scope="module")
def x0(imager_I):
    # perturb the initial solver vector so the gradient is nonzero
    rng = np.random.default_rng(RNG_SEED)
    base = np.asarray(imager_I._init_vec, dtype=np.float64)
    return base + PERTURB * rng.standard_normal(base.size)


def test_value_numpy_jax_consistent(imager_I, x0):
    v_np = float(imager_I.objfunc(x0))
    v_jx = float(imager_I.objfunc(jnp.asarray(x0)))
    assert np.allclose(v_np, v_jx, rtol=VALUE_RTOL, atol=GRAD_ATOL)


def test_grad_parity_autodiff_vs_analytic(imager_I, x0):
    # the gold standard: jax.grad(compute_objective) == compute_objective_grad
    g_analytic = np.asarray(imager_I.objgrad(x0))
    g_jax = np.asarray(jax.grad(imager_I.objfunc)(jnp.asarray(x0)))
    assert np.allclose(g_jax, g_analytic, rtol=GRAD_RTOL, atol=GRAD_ATOL)


def test_grad_finite_difference(imager_I, x0):
    g_analytic = np.asarray(imager_I.objgrad(x0))
    idx = np.argsort(np.abs(g_analytic))[-N_FD_SAMPLES:]  # best-conditioned pixels
    g_fd = np.empty(N_FD_SAMPLES)
    for k, j in enumerate(idx):
        xp_, xm_ = x0.copy(), x0.copy()
        xp_[j] += FD_STEP
        xm_[j] -= FD_STEP
        g_fd[k] = (float(imager_I.objfunc(xp_)) - float(imager_I.objfunc(xm_))) / (2 * FD_STEP)
    frac = np.abs((g_fd - g_analytic[idx]) / (np.abs(g_analytic[idx]) + 1e-100))
    assert np.median(frac) < FD_MEDIAN_TOL
    assert np.max(frac) < FD_MAX_TOL
