"""Canonical finite-difference gradient tests for the numpy imaging backend.

Every analytic gradient -- chi-squared data terms, regularizers, and the pol
change-of-variables transforms -- is checked against central finite differences
of its own scalar value, in pure numpy. The jax-autodiff parity check is the
separate second opinion in test_objective_jax.py; cross-transform value/gradient
consistency (direct vs fast vs nfft) lives in test_chisquared.py.

Structure -- parallel Stokes-I / pol / spectral sections, term lists at the top:

  S1 chi^2 Stokes-I : vis amp logamp bs cphase cphase_diag camp logcamp logcamp_diag
  S2 chi^2 pol      : pvis m vvis
  S3 reg   Stokes-I : every name in imager_backend.REGULARIZERS
  S4 reg   pol      : every name in REGULARIZERS_POL (all four physical slots)
  S5 reg   spectral : every name in REGULARIZERS_SPECTRAL
  S6 transforms     : mcv vcv polcv

Methodology (uniform across every section): central differences (FD_EPS),
compared by max fractional error relative to the per-slot gradient scale -- robust
where the analytic gradient is ~0 (a real missing term gives an O(1) ratio, while
finite-difference noise at a near-zero component does not). Finite differences are
full-grid on a small image so the zero-pad boundary (row 0 / col 0), where the TV
neighbour-roll regularizer gradients live, is always covered. Pol cases run at
v != 0 AND m != 0 at every pixel -- the regime where the mcv/vcv cross-term bugs hid.
"""
import numpy as np
import pytest

import ehtim as eh
from ehtim.imaging.imager_backend import (
    ImagerConfig,
    MfConfig,
    compute_chisq_term,
    compute_chisqdata_term,
    compute_chisqgrad_term,
)
from ehtim.imaging.imager_utils import chisq, chisqdata, chisqgrad

# --- finite-difference harness (one implementation for every section) ---------
FD_EPS = 1e-6
FD_RTOL = 1e-3        # max fractional error vs the gradient scale; correct grads sit far below
NFFT_RTOL = 1e-2      # nfft adds finufft truncation on top of the central-difference floor
ABS_FLOOR = 1e-9      # |grad| below this counts as identically zero (e.g. the vvis EVPA slot)
REL_FLOOR = 1e-3      # ignore components below this fraction of the slot's gradient scale
SEED = 4

# observation parameters (match conftest.py)
TINT_SEC, TADV_SEC, TSTART_HR, TSTOP_HR, BW_HZ = 5, 600, 0, 24, 4e9


def fd_grad(value_fn, x, eps=FD_EPS):
    """Full-grid central-difference gradient of scalar value_fn over array x."""
    x = np.asarray(x, dtype=float)
    g = np.zeros(x.shape)
    for idx in np.ndindex(x.shape):
        xp, xm = x.copy(), x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        g[idx] = (value_fn(xp) - value_fn(xm)) / (2 * eps)
    return g


def assert_grad_close(analytic, fd, rtol=FD_RTOL, label=""):
    """Assert analytic == fd by max fractional error relative to the gradient scale.

    A genuinely-zero gradient (scale below ABS_FLOOR) must have a ~0 analytic too;
    otherwise each component is normalized by max(|analytic|, |fd|, REL_FLOOR*scale)
    so near-zero components do not blow up the ratio while a real missing term does.
    """
    analytic = np.asarray(analytic, dtype=float)
    fd = np.asarray(fd, dtype=float)
    scale = float(np.max(np.abs(fd))) if fd.size else 0.0
    if scale < ABS_FLOOR:
        amax = float(np.max(np.abs(analytic))) if analytic.size else 0.0
        assert amax < ABS_FLOOR, f"{label}: expected ~0 gradient, got max|analytic|={amax:.2e}"
        return
    denom = np.maximum(np.maximum(np.abs(analytic), np.abs(fd)), REL_FLOOR * scale)
    frac = np.abs(analytic - fd) / denom
    assert np.max(frac) < rtol, (
        f"{label}: max frac err {np.max(frac):.2e} (median {np.median(frac):.2e})")


def _rtol(ttype):
    return NFFT_RTOL if ttype == "nfft" else FD_RTOL


def _chisqdata_kwargs(ttype):
    """chisqdata options, explicit so a default flip cannot silently move a tolerance.

    debias=False keeps raw amplitudes: debiasing floors sqrt(amp^2 - sigma^2) to 0 at
    baselines where the source has resolved out, which makes logamp's log|V| singular.
    """
    kw = dict(systematic_noise=0.0, snrcut=0.0, debias=False, weighting="natural",
              maxset=False, cp_uv_min=False, systematic_cphase_noise=0.0)
    if ttype in ("fast", "nfft"):
        kw.update(fft_pad_factor=10, p_rad=12, conv_func="gaussian", order=3)
    return kw


TTYPES = ["direct", "nfft"]


# ============================ S1: chi^2 Stokes-I ==============================
# Per-baseline amplitude/visibility terms run on a null-free single Gaussian: logamp is
# log(|V|), singular wherever a visibility nulls (an asymmetric source interferes to zero
# at some baselines). Closure phases/amps run on an asymmetric image: a symmetric source
# has zero closure phase, making those tests vacuous. Each group uses the image that keeps
# its gradient both well-conditioned and non-trivial.
PERBASELINE_TERMS = ["vis", "amp", "logamp"]
CLOSURE_TERMS = ["bs", "cphase", "cphase_diag", "camp", "logcamp", "logcamp_diag"]
DATATERMS_SI = PERBASELINE_TERMS + CLOSURE_TERMS


def _si_setup(im, eht_array):
    """Normalize a small (8x10) Stokes-I image, observe it once (DFT), jitter the imvec."""
    im.imvec = im.imvec * 2.0 / im.total_flux()
    obs = im.observe(eht_array, TINT_SEC, TADV_SEC, TSTART_HR, TSTOP_HR, BW_HZ,
                     sgrscat=False, ampcal=True, phasecal=True, ttype="direct",
                     add_th_noise=False)
    prior = im.copy()
    rng = np.random.default_rng(SEED)
    imvec = im.imvec * (1.0 + 0.05 * (rng.random(im.imvec.size) - 0.5))
    mask = np.ones(imvec.size, dtype=bool)
    return {"obs": obs, "prior": prior, "imvec": imvec, "mask": mask}


@pytest.fixture(scope="module")
def si_gauss_setup(eht_array):
    """Compact single Gaussian on a fine grid for the per-baseline amplitude terms.

    A compact source keeps |V| well above the noise floor at every EHT baseline, so
    log|V| (logamp) is well-conditioned everywhere; an extended source resolves out and
    its long-baseline amplitudes underflow.
    """
    im = eh.image.make_empty(10, 80 * eh.RADPERUAS, 17.761, -29.0, rf=230e9)
    im = im.add_gauss(2.0, (25 * eh.RADPERUAS, 25 * eh.RADPERUAS, 0, 0, 0))
    return _si_setup(im, eht_array)


@pytest.fixture(scope="module")
def si_asym_setup(eht_array, make_asym_image):
    """Asymmetric image (rect 8x10) -> nonzero closure phases, for the closure terms."""
    return _si_setup(make_asym_image(8, 10), eht_array)


class TestChisqGradientStokesI:
    """Analytic Stokes-I chi^2 gradients match central finite differences."""

    @pytest.mark.parametrize("ttype", TTYPES)
    @pytest.mark.parametrize("dtype", DATATERMS_SI)
    def test_grad_matches_fd(self, request, dtype, ttype):
        fixture = "si_gauss_setup" if dtype in PERBASELINE_TERMS else "si_asym_setup"
        s = request.getfixturevalue(fixture)
        obs, prior, mask, imvec = (s[k] for k in ("obs", "prior", "mask", "imvec"))
        data, sigma, A = chisqdata(obs, prior, mask, dtype, ttype=ttype, **_chisqdata_kwargs(ttype))
        analytic = chisqgrad(imvec, A, data, sigma, dtype, ttype=ttype, mask=mask)
        fd = fd_grad(lambda v: chisq(v, A, data, sigma, dtype, ttype=ttype, mask=mask), imvec)
        assert_grad_close(analytic, fd, rtol=_rtol(ttype), label=f"{dtype} {ttype}")


def test_diag_chisq_nfft_matches_direct(si_asym_setup):
    """Block-diagonal nfft diag chi^2 agrees with the direct-DFT diag chi^2.

    The nfft diagonalized closures apply the per-block decorrelating transforms as
    one block-diagonal matmul; the direct terms loop. Both share the same transforms
    and measured closures, so their chi^2 must agree -- a self-consistent-but-wrong
    restructure error that finite differences alone would not catch.
    """
    obs, prior, mask, imvec = (si_asym_setup[k] for k in ("obs", "prior", "mask", "imvec"))
    for dtype in ("cphase_diag", "logcamp_diag"):
        cdir = chisqdata(obs, prior, mask, dtype, ttype="direct", **_chisqdata_kwargs("direct"))
        cnf = chisqdata(obs, prior, mask, dtype, ttype="nfft", **_chisqdata_kwargs("nfft"))
        chi_dir = chisq(imvec, cdir[2], cdir[0], cdir[1], dtype, ttype="direct", mask=mask)
        chi_nf = chisq(imvec, cnf[2], cnf[0], cnf[1], dtype, ttype="nfft", mask=mask)
        assert abs(chi_dir - chi_nf) <= 1e-2 * abs(chi_dir), f"{dtype}: {chi_dir:.6g} vs {chi_nf:.6g}"


# ============================== S2: chi^2 pol ================================
POL_DATATERMS = ["pvis", "m", "vvis"]


def _pol_config(ttype):
    return ImagerConfig(pol="IP", transforms=[], ttype=ttype, mf=False,
                        mf_config=MfConfig(mf_order=0, mf_order_pol=0, mf_rm=0, mf_cm=0))


@pytest.fixture(scope="module")
def pol_setup(eht_array, make_asym_image):
    """Small (8x10) asymmetric polarized image + a jittered physical imcur [I,rho,phi,psi].

    add_random_pol gives a spatially-varying EVPA and (cmag>0) circular fraction, so
    chi, vfrac, rho and psi all vary; the imcur is clipped to rho in (0,1) and psi away
    from 0 so v != 0 and m != 0 at every pixel.
    """
    im = make_asym_image(8, 10)
    im.imvec = im.imvec * 2.0 / im.total_flux()
    im = im.add_random_pol(0.25, 40 * eh.RADPERUAS, cmag=0.06, ccorr=40 * eh.RADPERUAS, seed=7)
    prior = im.copy()
    obs = im.observe(eht_array, TINT_SEC, TADV_SEC, TSTART_HR, TSTOP_HR, BW_HZ,
                     ampcal=True, phasecal=True, ttype="direct", add_th_noise=False)
    mask = np.ones(im.imvec.size, dtype=bool)
    rng = np.random.default_rng(SEED)
    I = im.imvec
    Q, U, V = im.qvec, im.uvec, im.vvec
    P = np.sqrt(Q**2 + U**2 + V**2)
    n = I.size
    imcur = np.array([
        I * (1.0 + 0.05 * (rng.random(n) - 0.5)),
        np.clip((P / I) * (1.0 + 0.1 * (rng.random(n) - 0.5)), 0.02, 0.95),
        np.arctan2(U, Q) + 0.1 * (rng.random(n) - 0.5),
        np.clip(np.abs(np.arcsin(V / (P + 1e-30))) * (1.0 + 0.1 * (rng.random(n) - 0.5)), 0.02, 1.5),
    ])
    return {"obs": obs, "prior": prior, "mask": mask, "imcur": imcur}


class TestChisqGradientPol:
    """Analytic pol chi^2 gradients match central finite differences in all four slots.

    vvis is independent of the EVPA, so its slot-2 gradient is identically zero and
    assert_grad_close checks that the analytic slot is ~0 too.
    """

    @pytest.mark.parametrize("ttype", TTYPES)
    @pytest.mark.parametrize("dtype", POL_DATATERMS)
    def test_grad_matches_fd(self, pol_setup, dtype, ttype):
        obs, prior, mask, imcur = (pol_setup[k] for k in ("obs", "prior", "mask", "imcur"))
        data, sigma, A = compute_chisqdata_term(obs, prior, mask, dtype, _pol_config(ttype))
        analytic = compute_chisqgrad_term(imcur, dtype, A, data, sigma, ttype=ttype, mask=mask,
                                          pol_solve=np.array([1, 1, 1, 1]))
        fd = fd_grad(lambda im: compute_chisq_term(im, dtype, A, data, sigma, ttype=ttype, mask=mask), imcur)
        for s in range(4):
            assert_grad_close(analytic[s], fd[s], rtol=_rtol(ttype), label=f"{dtype} {ttype} slot{s}")
