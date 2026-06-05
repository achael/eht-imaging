"""JAX-backend tests for the embed-free Stokes-I regularizers.

Each regularizer's JAX gradient is checked three ways (autodiff == analytic ==
finite difference) in float64. The image is parameterized as imvec = exp(x).
The numpy regularizer leaves are tested separately in test_regularizers.py.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ehtim as eh
import ehtim.imaging.imager_utils as iu
from ehtim.imaging import regularizers as rg
from ehtim.imaging.imager_backend import transform_gradients

pytestmark = pytest.mark.jax

RTYPES = ["flux", "simple", "l1", "l1w", "lA", "gs"]

KERNEL_ATOL = 1e-12
GRAD_RTOL = 1e-9
GRAD_ATOL = 1e-12
FD_STEP = 1e-6
FD_MEDIAN_TOL = 1e-3
FD_MAX_TOL = 1e-2
N_FD_SAMPLES = 40
RNG_SEED = 4
PERTURB = 0.10
BEAM_SIZE = 20.0 * eh.RADPERUAS
ALPHA_A = 1.0


@pytest.fixture(scope="module")
def reg_setup(gauss_im):
    truth = gauss_im.imvec.astype(np.float64)
    flux = float(truth.sum())
    nprior = np.full_like(truth, flux / truth.size)
    mask = np.ones(truth.size, dtype=bool)
    rng = np.random.default_rng(RNG_SEED)
    imvec = np.clip(truth * (1.0 + PERTURB * rng.standard_normal(truth.size)), 1e-12, None)
    params = dict(flux=flux, nprior=nprior, norm_reg=True,
                  psize=gauss_im.psize, beam_size=BEAM_SIZE, alpha_A=ALPHA_A)
    return np.log(imvec), imvec, mask, params


@pytest.mark.parametrize("rtype", RTYPES)
def test_numpy_kernel_matches_reference(reg_setup, rtype):
    _, imvec, mask, params = reg_setup
    got = rg._REG[rtype](imvec, **params)
    ref = getattr(iu, "reg_" + rtype)(imvec, mask, **params)
    assert np.allclose(got, ref, atol=KERNEL_ATOL, rtol=0)


@pytest.mark.parametrize("rtype", RTYPES)
def test_three_way_gradient(reg_setup, rtype):
    x, imvec, mask, params = reg_setup

    g_jax = np.asarray(jax.grad(functools.partial(rg.reg_loss_log, rtype))(jnp.asarray(x), **params))
    g_analytic = transform_gradients(
        rg._REGGRAD[rtype](imvec, mask, **params), x, ["log"], np.array([1]),
    )
    assert np.allclose(g_jax, g_analytic, rtol=GRAD_RTOL, atol=GRAD_ATOL)

    idx = np.argsort(imvec)[-N_FD_SAMPLES:]  # brightest pixels are well-conditioned
    g_fd = np.empty(N_FD_SAMPLES)
    for k, j in enumerate(idx):
        xp, xm = x.copy(), x.copy()
        xp[j] += FD_STEP
        xm[j] -= FD_STEP
        fp = float(rg.reg_loss_log(rtype, xp, **params))
        fm = float(rg.reg_loss_log(rtype, xm, **params))
        g_fd[k] = (fp - fm) / (2 * FD_STEP)
    frac = np.abs((g_fd - g_analytic[idx]) / (np.abs(g_analytic[idx]) + 1e-100))
    assert np.median(frac) < FD_MEDIAN_TOL
    assert np.max(frac) < FD_MAX_TOL
