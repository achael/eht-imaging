"""JAX-backend tests for the embed-free Stokes-I regularizers in imager_utils.

Each regularizer's JAX gradient is checked three ways (autodiff == analytic ==
finite difference) in float64, plus numpy/jax value parity. The image is
parameterized as imvec = exp(x). The numpy regularizers are exercised by the rest
of the suite; here we confirm the same imager_utils functions run under jax.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ehtim as eh
import ehtim.imaging.imager_utils as iu
from ehtim.backends import array_namespace
from ehtim.imaging.imager_backend import transform_gradients

pytestmark = pytest.mark.jax

RTYPES = ["flux", "simple", "l1", "l1w", "lA", "gs"]

# rtol, not atol: XLA reorders the float summation vs numpy.
VALUE_RTOL = 1e-9
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


def _reg(rtype, im, **kwargs):
    # the embed-free regularizers take a mask they ignore; pass None
    return getattr(iu, "reg_" + rtype)(im, None, **kwargs)


def _reg_loss_log(rtype, x, **kwargs):
    xp = array_namespace(x)
    return _reg(rtype, xp.exp(x), **kwargs)


@pytest.fixture(scope="module")
def reg_setup(gauss_im):
    truth = gauss_im.imvec.astype(np.float64)
    flux = float(truth.sum())
    nprior = np.full_like(truth, flux / truth.size)
    rng = np.random.default_rng(RNG_SEED)
    imvec = np.clip(truth * (1.0 + PERTURB * rng.standard_normal(truth.size)), 1e-12, None)
    params = dict(flux=flux, nprior=nprior, norm_reg=True,
                  psize=gauss_im.psize, beam_size=BEAM_SIZE, alpha_A=ALPHA_A)
    return np.log(imvec), imvec, params


@pytest.mark.parametrize("rtype", RTYPES)
def test_value_numpy_jax_consistent(reg_setup, rtype):
    _, imvec, params = reg_setup
    got_np = _reg(rtype, imvec, **params)
    got_jx = _reg(rtype, jnp.asarray(imvec), **params)
    assert np.allclose(got_np, np.asarray(got_jx), rtol=VALUE_RTOL, atol=GRAD_ATOL)


@pytest.mark.parametrize("rtype", RTYPES)
def test_three_way_gradient(reg_setup, rtype):
    x, imvec, params = reg_setup
    mask = np.ones(imvec.size, dtype=bool)

    g_jax = np.asarray(jax.grad(functools.partial(_reg_loss_log, rtype))(jnp.asarray(x), **params))
    # transform_gradients: ["log"] is the exp(x) chain rule; which_solve is ignored for
    # single-Stokes (nimage==1), so (1,0,0,0) is the nominal mask
    g_analytic = transform_gradients(
        getattr(iu, "reggrad_" + rtype)(imvec, mask, **params), x, ["log"], np.array((1, 0, 0, 0)),
    )
    assert np.allclose(g_jax, g_analytic, rtol=GRAD_RTOL, atol=GRAD_ATOL)

    idx = np.argsort(imvec)[-N_FD_SAMPLES:]  # brightest pixels are well-conditioned
    g_fd = np.empty(N_FD_SAMPLES)
    for k, j in enumerate(idx):
        xp_, xm_ = x.copy(), x.copy()
        xp_[j] += FD_STEP
        xm_[j] -= FD_STEP
        fp = float(_reg_loss_log(rtype, xp_, **params))
        fm = float(_reg_loss_log(rtype, xm_, **params))
        g_fd[k] = (fp - fm) / (2 * FD_STEP)
    frac = np.abs((g_fd - g_analytic[idx]) / (np.abs(g_analytic[idx]) + 1e-100))
    assert np.median(frac) < FD_MEDIAN_TOL
    assert np.max(frac) < FD_MAX_TOL


# ============================== embed / partial mask ==============================
def test_embed_functional_jax():
    # embed's functional scatter is byte-identical on numpy and differentiable on jax
    mask = np.array([True, False, True, False, True, True])
    imvec = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(iu.embed(imvec, mask), np.asarray(iu.embed(jnp.asarray(imvec), mask)))
    g = jax.grad(lambda v: jnp.sum(iu.embed(v, mask) ** 2))(jnp.asarray(imvec))
    assert np.allclose(np.asarray(g), 2 * imvec)  # grad routes only to on-mask pixels


def test_spatial_reg_partial_mask_parity():
    # reg_tv with a partial mask exercises the embed scatter under jax; clipfloor=0
    # makes the off-mask fill deterministic (no seed needed for numpy<->jax parity).
    rng = np.random.default_rng(0)
    ny, nx = 6, 6
    full = rng.uniform(0.1, 1.0, ny * nx)
    mask = np.ones(ny * nx, dtype=bool)
    mask[rng.choice(ny * nx, 8, replace=False)] = False
    imvec = full[mask]
    kw = dict(xdim=nx, ydim=ny, psize=1.0, flux=float(full.sum()), beam_size=2.0, norm_reg=True)
    assert np.allclose(iu.reg_tv(imvec, mask, **kw),
                       float(iu.reg_tv(jnp.asarray(imvec), mask, **kw)), rtol=1e-12)
    g_jax = np.asarray(jax.grad(lambda v: iu.reg_tv(v, mask, **kw))(jnp.asarray(imvec)))
    assert np.allclose(g_jax, iu.reggrad_tv(imvec, mask, **kw), rtol=1e-8, atol=1e-10)
