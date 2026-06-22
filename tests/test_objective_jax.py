"""JAX parity tests for the imaging objective (compute_objective).

Each case checks, uniformly: value parity (numpy vs jax), analytic-gradient parity
(jax.grad vs the hand-written objgrad), and finite differences. The union of the
parametrized cases exercises every jaxified function:

  * transforms      : direct + nfft
  * Stokes-I chisqs : vis, amp, bs, cphase, camp, logcamp  (the _diag closures are
                      numpy-only, not jaxified)
  * Stokes-I regs   : simple, flux, l1, l1w, gs, patch, cm, tv, tvlog, tv2, tv2log,
                      compact, compact2, rgauss
  * pol modes       : IP/P (mcv), IV/V (vcv), IQUV/IPV (polcv) -- all full parity
  * pol chisqs      : pvis, m, vvis    pol regs : msimple, hw, ptv, vflux, l1v, l2v, vtv, vtv2
  * multifrequency  : 2-frequency Stokes-I, mf_order=1, l2_alpha + tv_alpha

Pol tests use an asymmetric image: a symmetric source zeros gradient components and
hides bugs (these parity checks caught several mcv/vcv gradient errors).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ehtim as eh
from ehtim.imaging.imager_backend import make_objective_jax
from ehtim.observing.obs_helpers import NFFTInfo, ftmatrix, nufft2_backend

pytestmark = pytest.mark.jax

# rtol, not atol: XLA reorders the float summation, so terms differ from numpy at
# ~1e-14 relative; a real bug fails by orders of magnitude.
VALUE_RTOL = 1e-9
GRAD_RTOL = 1e-9
GRAD_ATOL = 1e-12
# nfft is an approximate transform: jax_finufft's VJP matches the analytic adjoint
# only to the NUFFT accuracy. The residual scales with eps (1e-9 -> ~1e-6, 1e-12 ->
# ~7e-10); the nfft fixtures use eps=1e-12 and this looser-but-tight rtol. A real VJP
# bug is eps-independent and fails by orders of magnitude.
NFFT_EPS = 1e-12
NFFT_GRAD_RTOL = 1e-7
# finite-difference comparison (jax.grad vs central differences on best-conditioned pixels)
FD_STEP = 1e-6
FD_MEDIAN_TOL = 1e-3
FD_MAX_TOL = 1e-2
N_FD_SAMPLES = 30
RNG_SEED = 4
PERTURB = 0.10
# uniform epsilon for the TV-family sqrt (tv/tvlog/ptv/vtv...); the default 0 is
# singular where neighbouring pixels are equal, in Stokes I exactly as in pol.
EPSILON_TV = 1e-10

# --- coverage cases: the union exercises every jaxified chisq + regularizer ---
STOKESI_CASES = [
    ({"vis": 100, "amp": 100, "bs": 100},
     {"simple": 1, "flux": 1, "tv": 10, "l1": 1, "gs": 1, "tv2": 1, "l1w": 1}),
    ({"cphase": 100, "camp": 50, "logcamp": 50},
     {"patch": 1, "cm": 1, "tvlog": 1, "tv2log": 1, "compact": 1, "compact2": 1, "rgauss": 1}),
]
_SI_PARAMS = [("direct", 0), ("direct", 1), ("nfft", 0), ("nfft", 1)]
_SI_IDS = ["direct-A", "direct-B", "nfft-A", "nfft-B"]

# Six pol modes across mcv/vcv/polcv. V can't use log (V is signed) or simple/gs regs;
# IPV aliases IQUV. Union covers all pol chisqs + regs.
POL_CASES = [
    ("IP",   ["log", "mcv"],   {"amp": 100, "pvis": 100, "m": 50}, {"simple": 1, "msimple": 1, "hw": 1, "ptv": 1}),
    ("P",    ["log", "mcv"],   {"pvis": 100, "m": 50}, {"msimple": 1, "hw": 1, "ptv": 1}),
    ("IV",   ["log", "vcv"],   {"amp": 100, "vvis": 100}, {"simple": 1, "vflux": 1, "l1v": 1, "vtv": 1}),
    ("V",    ["vcv"],          {"vvis": 100}, {"vflux": 1, "l1v": 1, "l2v": 1, "vtv": 1, "vtv2": 1}),
    ("IQUV", ["log", "polcv"], {"amp": 100, "pvis": 100, "m": 50, "vvis": 100}, {"simple": 1, "l2v": 1, "vtv2": 1}),
    ("IPV",  ["log", "polcv"], {"amp": 100, "pvis": 100, "m": 50, "vvis": 100}, {"simple": 1, "l2v": 1, "vtv2": 1}),
]
_POL_IDS = [c[0] for c in POL_CASES]

# nfft pol: all six modes, parallel to direct (exercises the _nfft chisqs + gradients).
_POL_NFFT_CASES = POL_CASES
_POL_NFFT_IDS = _POL_IDS


# ============================== shared helpers ==============================
def _x0(imgr):
    # perturb the initial solver vector so the gradient is nonzero
    rng = np.random.default_rng(RNG_SEED)
    base = np.asarray(imgr._init_vec, dtype=np.float64)
    return base + PERTURB * rng.standard_normal(base.size)


def _grad_rtol(imgr):
    return GRAD_RTOL if imgr._config.ttype == "direct" else NFFT_GRAD_RTOL


def _assert_value(imgr, x):
    v_np = float(imgr.objfunc(np.asarray(x)))
    v_jx = float(imgr.objfunc(jnp.asarray(x)))
    assert np.allclose(v_np, v_jx, rtol=VALUE_RTOL, atol=GRAD_ATOL)


def _assert_grad_analytic(imgr, x):
    # jax.grad vs the hand-written analytic, compared in norm. nfft's VJP matches the
    # analytic adjoint only to the NUFFT eps (relative), so a per-component atol would
    # trip on negligible near-zero components; the norm ratio still catches real bugs
    # (the fixed mcv/vcv gradient bugs were ~1e-2 here, far above the 1e-9 threshold).
    g_an = np.asarray(imgr.objgrad(np.asarray(x)))
    g_jx = np.asarray(jax.grad(imgr.objfunc)(jnp.asarray(x)))
    rel = np.linalg.norm(g_jx - g_an) / np.linalg.norm(g_an)
    assert rel < _grad_rtol(imgr), f"||jax - analytic|| / ||analytic|| = {rel:.2e}"


def _assert_grad_fd(imgr, x):
    g_jx = np.asarray(jax.grad(imgr.objfunc)(jnp.asarray(x)))
    idx = np.argsort(np.abs(g_jx))[-N_FD_SAMPLES:]  # best-conditioned pixels
    g_fd = np.empty(N_FD_SAMPLES)
    for k, j in enumerate(idx):
        xp_, xm_ = np.array(x, float), np.array(x, float)
        xp_[j] += FD_STEP
        xm_[j] -= FD_STEP
        g_fd[k] = (float(imgr.objfunc(xp_)) - float(imgr.objfunc(xm_))) / (2 * FD_STEP)
    frac = np.abs((g_fd - g_jx[idx]) / (np.abs(g_jx[idx]) + 1e-100))
    assert np.median(frac) < FD_MEDIAN_TOL
    assert np.max(frac) < FD_MAX_TOL


def _make_fun(imgr, device=None):
    return make_objective_jax(
        imgr._init_arr, imgr._config, imgr._which_solve, imgr._data_tuples,
        imgr._logfreqratio_list, len(imgr.obslist_next), imgr.dat_term_next,
        imgr.reg_term_next, imgr._prior_arr, imgr.norm_reg, imgr._regparams(),
        imgr._embed_mask, device=device,
    )


# ============================== Stokes-I ==============================
@pytest.fixture(scope="module", params=_SI_PARAMS, ids=_SI_IDS)
def imager(request, obs_direct, obs_nfft, gauss_im, gauss_prior):
    ttype, case = request.param
    data_term, reg_term = STOKESI_CASES[case]
    obs = obs_direct if ttype == "direct" else obs_nfft
    extra = {} if ttype == "direct" else {"nfft_eps": NFFT_EPS}
    imgr = eh.imager.Imager(
        obs, gauss_prior, prior_im=gauss_prior, flux=gauss_im.total_flux(),
        data_term=data_term, reg_term=reg_term, ttype=ttype, pol="I",
        maxit=100, epsilon_tv=EPSILON_TV, **extra,
    )
    imgr.init_imager()
    return imgr


@pytest.fixture(scope="module")
def x0(imager):
    return _x0(imager)


def test_value_numpy_jax_consistent(imager, x0):
    _assert_value(imager, x0)


def test_grad_parity_autodiff_vs_analytic(imager, x0):
    _assert_grad_analytic(imager, x0)


def test_grad_finite_difference(imager, x0):
    _assert_grad_fd(imager, x0)


def test_factory_matches_analytic(imager, x0):
    value, grad = _make_fun(imager)(x0)
    assert np.allclose(value, float(imager.objfunc(x0)), rtol=VALUE_RTOL, atol=GRAD_ATOL)
    assert np.allclose(grad, np.asarray(imager.objgrad(x0)), rtol=_grad_rtol(imager), atol=GRAD_ATOL)


def test_objective_traces_once_across_x(imager, x0):
    # the factory closes over everything but x, so one jit trace serves every solver step.
    traces = {"n": 0}

    def counted(x):
        traces["n"] += 1
        return imager.objfunc(x)

    f = jax.jit(counted)
    f(jnp.asarray(x0)).block_until_ready()
    f(jnp.asarray(x0 + 0.5)).block_until_ready()
    assert traces["n"] == 1


# ============================== nufft2_backend equality ==============================
def test_nufft2_backend_numpy_jax_equality():
    # nufft2_backend dispatches finufft (numpy) vs jax_finufft (jax); the two must agree,
    # and both must reproduce the direct DFT to NFFT accuracy. (The chisqs exercise this
    # indirectly; this pins it down on its own.)
    npix, psize = 16, 200 * eh.RADPERUAS / 16
    pulse = eh.observing.pulses.trianglePulse2D
    rng = np.random.default_rng(1)
    uv = rng.uniform(-2e9, 2e9, size=(40, 2))
    nfft_info = NFFTInfo(npix, npix, psize, pulse, 1, 2, uv, eps=1e-12)
    imvec = rng.standard_normal(npix * npix) + 1j * rng.standard_normal(npix * npix)

    s_np = np.asarray(nufft2_backend(imvec, nfft_info))
    s_jx = np.asarray(nufft2_backend(jnp.asarray(imvec), nfft_info))
    assert np.allclose(s_np, s_jx, rtol=1e-9, atol=1e-9)  # numpy == jax

    s_dft = ftmatrix(psize, npix, npix, uv, pulse=pulse) @ imvec
    assert np.allclose(s_np * nfft_info.pulsefac, s_dft, rtol=1e-6, atol=1e-6)  # == DFT


# ============================== Polarization ==============================
# Asymmetric image (off-center, elongated): symmetry would zero gradient components
# and hide the pol-gradient bugs these tests guard.
@pytest.fixture(scope="module")
def pol_asym(eht_array):
    R = eh.RADPERUAS
    im = eh.image.make_empty(32, 200 * R, 17.761, -29.0, rf=230e9)
    im = im.add_gauss(1.0, (60 * R, 30 * R, 0.5, 25 * R, -20 * R))
    im.add_qu(0.10 * im.imarr(), 0.05 * im.imarr())
    im.add_v(0.02 * im.imarr())
    prior = im.blur_circ(35 * R)
    kw = dict(ampcal=True, phasecal=True, add_th_noise=True, seed=42)
    obs_d = im.observe(eht_array, 5, 600, 0, 24, 4e9, ttype="direct", **kw)
    obs_n = im.observe(eht_array, 5, 600, 0, 24, 4e9, ttype="nfft", **kw)
    return im, prior, obs_d, obs_n


@pytest.fixture(scope="module", params=POL_CASES, ids=_POL_IDS)
def pol_imager(request, pol_asym):
    im, prior, obs_d, _ = pol_asym
    mode, transform, data_term, reg_term = request.param
    imgr = eh.imager.Imager(
        obs_d, prior, prior_im=prior, flux=im.total_flux(),
        data_term=data_term, reg_term=reg_term, ttype="direct", pol=mode,
        transform=transform, maxit=100, epsilon_tv=EPSILON_TV,
    )
    imgr.init_imager()
    return imgr


@pytest.fixture(scope="module")
def pol_x0(pol_imager):
    return _x0(pol_imager)


def test_pol_value_numpy_jax_consistent(pol_imager, pol_x0):
    _assert_value(pol_imager, pol_x0)


def test_pol_grad_parity_autodiff_vs_analytic(pol_imager, pol_x0):
    # parallel to Stokes I: jax.grad == analytic, now that the IV (vcv) gradient is fixed
    _assert_grad_analytic(pol_imager, pol_x0)


def test_pol_grad_finite_difference(pol_imager, pol_x0):
    _assert_grad_fd(pol_imager, pol_x0)


@pytest.fixture(scope="module", params=_POL_NFFT_CASES, ids=_POL_NFFT_IDS)
def pol_imager_nfft(request, pol_asym):
    im, prior, _, obs_n = pol_asym
    mode, transform, data_term, reg_term = request.param
    imgr = eh.imager.Imager(
        obs_n, prior, prior_im=prior, flux=im.total_flux(),
        data_term=data_term, reg_term=reg_term, ttype="nfft", pol=mode,
        transform=transform, maxit=100, nfft_eps=NFFT_EPS, epsilon_tv=EPSILON_TV,
    )
    imgr.init_imager()
    return imgr


@pytest.fixture(scope="module")
def pol_nfft_x0(pol_imager_nfft):
    return _x0(pol_imager_nfft)


def test_pol_nfft_value_numpy_jax_consistent(pol_imager_nfft, pol_nfft_x0):
    _assert_value(pol_imager_nfft, pol_nfft_x0)


def test_pol_nfft_grad_parity_autodiff_vs_analytic(pol_imager_nfft, pol_nfft_x0):
    _assert_grad_analytic(pol_imager_nfft, pol_nfft_x0)


def test_pol_nfft_grad_finite_difference(pol_imager_nfft, pol_nfft_x0):
    _assert_grad_fd(pol_imager_nfft, pol_nfft_x0)


# ============================== Multifrequency (Stokes-I) ==============================
# TODO: this uses a 2-frequency synthetic Gaussian with a constant spectral index. Add
# real multifrequency synthetic data (multiple realistic bands) for thorough coverage,
# and extend to multifrequency *polarization* (currently only Stokes-I mf is covered).
@pytest.fixture(scope="module")
def _mf_setup(gauss_im, eht_array):
    im = gauss_im.copy().add_const_mf(1.0, 0)  # alpha=1, beta=0
    prior = im.blur_circ(40 * eh.RADPERUAS)
    obslist = [
        im.get_image_mf(nu).observe(eht_array, 5, 600, 0, 24, 4e9, ampcal=True,
                                     phasecal=True, ttype="direct", add_th_noise=True, seed=42)
        for nu in (220e9, 240e9)
    ]
    return obslist, prior, im


@pytest.fixture(scope="module")
def mf_imager(_mf_setup):
    obslist, prior, im = _mf_setup
    imgr = eh.imager.Imager(
        obslist, prior, prior_im=prior, flux=im.total_flux(),
        data_term={"amp": 100, "cphase": 100},
        reg_term={"simple": 1, "tv": 1, "l2_alpha": 1, "tv_alpha": 1},  # + spectral-index regs
        ttype="direct", pol="I", mf=True, mf_order=1, maxit=100, epsilon_tv=EPSILON_TV,
    )
    imgr.init_imager()
    return imgr


@pytest.fixture(scope="module")
def mf_x0(mf_imager):
    return _x0(mf_imager)


def test_mf_value_numpy_jax_consistent(mf_imager, mf_x0):
    _assert_value(mf_imager, mf_x0)


def test_mf_grad_parity_autodiff_vs_analytic(mf_imager, mf_x0):
    _assert_grad_analytic(mf_imager, mf_x0)


def test_mf_grad_finite_difference(mf_imager, mf_x0):
    _assert_grad_fd(mf_imager, mf_x0)


# ============================== OO integration ==============================
def _nxcorr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


@pytest.mark.slow
def test_make_image_use_jax_recovers_and_matches_numpy(obs_direct, gauss_im, gauss_prior):
    # make_image(use_jax=True) runs the jax objective end to end through scipy
    # L-BFGS-B and recovers the source; it matches the numpy path (same objective).
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
    assert _nxcorr(im_jx, im_np) > 0.95   # jax matches numpy


# ============================== GPU ==============================
def _has_gpu():
    try:
        return len(jax.devices("gpu")) > 0
    except RuntimeError:
        return False


requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="no CUDA GPU available")


@pytest.mark.gpu
@requires_gpu
def test_objective_runs_on_gpu(imager, x0):
    gpu = jax.devices("gpu")[0]
    value, grad = _make_fun(imager, device=gpu)(x0)
    assert np.isfinite(value) and np.all(np.isfinite(grad))


@pytest.mark.gpu
@requires_gpu
def test_objective_gpu_cpu_parity(imager, x0):
    cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    v_c, g_c = _make_fun(imager, device=cpu)(x0)
    v_g, g_g = _make_fun(imager, device=gpu)(x0)
    assert np.allclose(v_c, v_g, rtol=_grad_rtol(imager), atol=1e-9)
    assert np.allclose(g_c, g_g, rtol=_grad_rtol(imager), atol=1e-9)
