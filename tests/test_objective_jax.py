"""JAX parity tests for the full imaging objective (compute_objective).

unpack_imarr and transform_imarr are now functional, so jax.grad(compute_objective)
is the jax objective gradient. These check it equals the hand-written
compute_objective_grad (the gold standard), matches finite differences, and that
the value agrees numpy<->jax. Stokes-I across ttype direct + nfft, all regularizers
(embed-free + spatial), full and partial embed masks.

The Imager wrappers objfunc/objgrad call compute_objective/compute_objective_grad,
so jax.grad(imgr.objfunc) is exactly the autodiff objective gradient.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ehtim as eh
import ehtim.imaging.imager_utils as iu
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
# Pol grad tests validate jax.grad against central FD (the ground truth), NOT the
# analytic compute_objective_grad: the V-pol analytic gradient is wrong (jax and FD
# agree, the analytic disagrees -- tracked bug), and ptv/vtv have a sqrt singularity.
FD_STEP = 1e-6
FD_MEDIAN_TOL = 1e-3
FD_MAX_TOL = 1e-2
N_FD_SAMPLES = 40
RNG_SEED = 4
PERTURB = 0.10

DATA_TERM = {"amp": 100, "cphase": 100, "logcamp": 50}
REG_TERM = {"simple": 1, "flux": 1, "tv": 10}  # tv exercises the spatial-reg path


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


@pytest.fixture(scope="module")
def pol_imager(obs_pol_direct, gauss_im_pol, gauss_prior):
    # IP imaging: Stokes I + linear pol (pvis, m); mcv change-of-variables.
    # Stokes-I reg only here; pol regs are exercised below.
    imgr = eh.imager.Imager(
        obs_pol_direct, gauss_prior, prior_im=gauss_prior, flux=gauss_im_pol.total_flux(),
        data_term={"amp": 100, "pvis": 100, "m": 50},
        reg_term={"simple": 1, "hw": 1, "ptv": 1},  # Stokes-I + linear-pol regs
        ttype="direct", pol="IP", transform=["log", "mcv"], maxit=100,
    )
    imgr.init_imager()
    return imgr


@pytest.fixture(scope="module")
def pol_x0(pol_imager):
    rng = np.random.default_rng(RNG_SEED)
    base = np.asarray(pol_imager._init_vec, dtype=np.float64)
    return base + PERTURB * rng.standard_normal(base.size)


def test_pol_value_numpy_jax_consistent(pol_imager, pol_x0):
    assert np.allclose(float(pol_imager.objfunc(pol_x0)),
                       float(pol_imager.objfunc(jnp.asarray(pol_x0))), rtol=VALUE_RTOL, atol=GRAD_ATOL)


def _assert_pol_grad_matches_fd(imgr, x0):
    # jax.grad vs central FD on the best-conditioned pixels (the ground truth).
    g_jax = np.asarray(jax.grad(imgr.objfunc)(jnp.asarray(x0)))
    idx = np.argsort(np.abs(g_jax))[-N_FD_SAMPLES:]
    g_fd = np.empty(N_FD_SAMPLES)
    for k, j in enumerate(idx):
        xp_, xm_ = x0.copy(), x0.copy()
        xp_[j] += FD_STEP
        xm_[j] -= FD_STEP
        g_fd[k] = (float(imgr.objfunc(xp_)) - float(imgr.objfunc(xm_))) / (2 * FD_STEP)
    frac = np.abs((g_fd - g_jax[idx]) / (np.abs(g_jax[idx]) + 1e-100))
    assert np.median(frac) < FD_MEDIAN_TOL
    assert np.max(frac) < FD_MAX_TOL


def test_pol_grad_finite_difference(pol_imager, pol_x0):
    _assert_pol_grad_matches_fd(pol_imager, pol_x0)


@pytest.fixture(scope="module")
def pol_imager_v(obs_pol_direct, gauss_im_pol, gauss_prior):
    # IV imaging: Stokes I + circular pol (vvis); vcv change-of-variables; V regs.
    imgr = eh.imager.Imager(
        obs_pol_direct, gauss_prior, prior_im=gauss_prior, flux=gauss_im_pol.total_flux(),
        data_term={"amp": 100, "vvis": 100},
        reg_term={"simple": 1, "l1v": 1, "vtv": 1},
        ttype="direct", pol="IV", transform=["log", "vcv"], maxit=100,
    )
    imgr.init_imager()
    return imgr


@pytest.fixture(scope="module")
def pol_v_x0(pol_imager_v):
    rng = np.random.default_rng(RNG_SEED)
    base = np.asarray(pol_imager_v._init_vec, dtype=np.float64)
    return base + PERTURB * rng.standard_normal(base.size)


def test_pol_v_grad_finite_difference(pol_imager_v, pol_v_x0):
    _assert_pol_grad_matches_fd(pol_imager_v, pol_v_x0)


def _nxcorr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


@pytest.mark.slow
def test_make_image_use_jax_recovers_and_matches_numpy(obs_direct, gauss_im, gauss_prior):
    # OO integration: make_image(use_jax=True) runs the jax objective end to end
    # through scipy L-BFGS-B and recovers the source; matches the numpy path.
    def recon(use_jax):
        imgr = eh.imager.Imager(
            obs_direct, gauss_prior, prior_im=gauss_prior, flux=gauss_im.total_flux(),
            data_term={"amp": 100, "cphase": 100, "logcamp": 50},
            reg_term={"simple": 1, "tv": 10}, ttype="direct", pol="I", maxit=200,
        )
        return imgr.make_image_I(niter=1, show_updates=False, use_jax=use_jax).imvec

    truth = gauss_im.imvec
    im_np, im_jx = recon(False), recon(True)
    assert _nxcorr(im_jx, truth) > 0.9    # jax recon recovers the source
    assert _nxcorr(im_jx, im_np) > 0.95   # jax matches numpy (same objective)
