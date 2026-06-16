"""Temporary gradient-validation suite for the `main` (v1.3.x) branch.

This is a *minimal, interim* suite whose sole purpose is to validate that the
analytic gradients on `main` are correct before `dev` (which carries the full
test suite) replaces `main`. It mirrors the gradient tests in PR #306
(branch fix/pol-grad-mask -> dev), re-expressed against main's API
(`imager_utils` / `pol_imager_utils` / `imager`), since main lacks the
backend (`imager_backend`) those dev tests call.

Scope (only gradients; everything else is inherited from dev later):
  1. nfft vs direct (dft) agreement for every intensity and pol chi^2 term.
  2. analytic gradients vs finite differences for every intensity / pol chi^2
     term and regularizer.
  3. the polcv / vcv / mcv image transforms correctly modify the gradient,
     checked against finite differences both in isolation (chain rule) and
     end-to-end through the Imager objective.

RUN WITH THE SYSTEM PYTHON (it has the legacy `nfft` backend):
    python3 -m pytest tests/test_gradients_main_tmp.py
The `jax-ehtim` micromamba env lacks legacy `nfft`; the nfft cases skip there.

Delete this file once dev lands on main.
"""

import os

import numpy as np
import pytest

import ehtim as eh
import ehtim.imaging.imager_utils as iu
import ehtim.imaging.pol_imager_utils as pu
from ehtim.imager import transform_gradients, transform_imarr

# ---------------------------------------------------------------------------
# nfft availability (legacy backend; present only in the system env)
# ---------------------------------------------------------------------------
try:
    import nfft as _nfft_mod  # noqa: F401
    HAS_NFFT = True
except Exception:
    HAS_NFFT = False

TTYPES = ["direct"] + (["nfft"] if HAS_NFFT else [])
requires_nfft = pytest.mark.skipif(
    not HAS_NFFT, reason="legacy 'nfft' backend not installed (use system python)"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARRAY_PATH = os.path.join(os.path.dirname(__file__), "..", "arrays", "EHT2017.txt")
TINT, TADV, TSTART, TSTOP, BW = 5, 600, 0, 24, 4e9
RNG_SEED = 4
UA = eh.RADPERUAS

# Intensity chi^2 terms under test (every entry of imager_utils.DATATERMS).
INTENSITY_DATATERMS = ["vis", "bs", "amp", "logamp", "cphase", "camp", "logcamp",
                       "cphase_diag", "logcamp_diag"]

# Intensity regularizers under test (the well-conditioned set + the log-TV
# variants, which share the boundary-masking path the #306 patch touched).
# Omitted from this interim suite (special params / FD-ill-conditioned;
# inherited from dev's full suite later): lA, patch, compact, compact2, rgauss.
INTENSITY_REGS = ["simple", "gs", "l1w", "tv", "tv2", "tvlog", "tv2log"]

POL_DATATERMS = ["pvis", "m", "vvis"]
POL_REGS = list(pu.REGULARIZERS_POL)  # msimple, hw, ptv, l1v, l2v, vtv, vtv2, vflux

# Tolerances (mirrored from the dev #306 suite, calibrated on a 32x48 image)
CHISQ_FD_MEDIAN, CHISQ_FD_MAX = 1e-3, 1e-2
REG_FD_MEDIAN, REG_FD_MAX = 0.05, 0.6
POL_CHISQ_FD_MEDIAN, POL_CHISQ_FD_MAX = 1e-5, 1e-3
POL_REG_FD_MEDIAN, POL_REG_FD_MAX = 1e-4, 1e-2
# nfft-vs-direct: median of per-pixel frac diff + relative L2 norm of the
# whole gradient. (A per-pixel MAX is meaningless here -- it blows up on
# near-zero-gradient pixels where a tiny absolute Fourier diff is a huge
# relative one; the L2 norm is the robust "do these vectors agree" metric.)
NFFT_VALUE_FRAC, NFFT_GRAD_MEDIAN, NFFT_GRAD_REL_L2 = 1e-2, 0.05, 0.05

POL_FD_REL, POL_FD_FLOOR = 1e-6, 1e-9
FD_REL, FD_FLOOR = 1e-8, 1e-12
N_SAMPLES = 100


# ---------------------------------------------------------------------------
# Image helpers / fixtures (self-contained; main has no tests/conftest.py)
# ---------------------------------------------------------------------------
def make_asym_image(xdim, ydim, psize=None):
    """Asymmetric offset double-Gaussian Stokes-I image of size (xdim, ydim).

    Breaks reflection/rotation/x<->y symmetry and pushes flux toward the edges
    so boundary-masking / axis-ordering bugs (which a centered Gaussian hides)
    surface in the chisq / regularizer / gradient FD checks. Blobs are broad so
    the grid is filled (no dead pixels that make TV gradients FD-ill-conditioned
    at epsilon=0). xdim != ydim exercises rectangular-image code paths.
    """
    if psize is None:
        psize = 200 * UA / max(xdim, ydim)
    yy, xx = np.mgrid[0:ydim, 0:xdim]
    x = (xx - xdim / 2.0) * psize
    y = (yy - ydim / 2.0) * psize

    def blob(flux, fmaj, fmin, pa, x0, y0):
        smaj, smin = fmaj / 2.355, fmin / 2.355
        xr = (x - x0) * np.cos(pa) + (y - y0) * np.sin(pa)
        yr = -(x - x0) * np.sin(pa) + (y - y0) * np.cos(pa)
        return flux * np.exp(-(xr**2 / (2 * smaj**2) + yr**2 / (2 * smin**2)))

    image_arr = (blob(0.6, 120 * UA, 78 * UA, 0.5, 16 * UA, -11 * UA)
                 + blob(0.4, 84 * UA, 116 * UA, -0.7, -19 * UA, 17 * UA))
    image_arr /= image_arr.sum()
    return eh.image.Image(image_arr, psize, 17.761, -29.0,
                          polrep="stokes", pol_prim="I", rf=230e9)


@pytest.fixture(scope="module")
def array():
    return eh.array.load_txt(ARRAY_PATH)


@pytest.fixture(scope="module")
def intensity_setup(array):
    """Asymmetric Stokes-I image + noise-free obs + a jittered test image."""
    im = make_asym_image(32, 48)
    im.imvec = im.imvec * 2.0 / im.total_flux()
    obs = im.observe(array, TINT, TADV, TSTART, TSTOP, BW,
                     ampcal=True, phasecal=True, ttype="direct", add_th_noise=False)
    mask = np.ones(im.imvec.size, dtype=bool)
    rng = np.random.default_rng(RNG_SEED)
    test_imvec = im.imvec * (1.0 + 0.1 * (rng.random(im.imvec.size) - 0.5))
    return dict(obs=obs, prior=im.copy(), mask=mask, imvec=test_imvec, im=im)


@pytest.fixture(scope="module")
def pol_setup(array):
    """Asymmetric polarized image (spatially-varying EVPA + circular fraction)
    + noise-free obs + a jittered physical imcur [I, rho, phi=2chi, psi].

    Square grid: add_random_pol -> MakePhaseScreen has a rectangular-image bug
    on main (deferred scattering rect fix), so pol fixtures use a square image.
    The offset/rotated double-Gaussian still breaks the symmetries.
    """
    im = make_asym_image(32, 32)
    im.imvec = im.imvec * 2.0 / im.total_flux()
    im = im.add_random_pol(0.25, 40 * UA, cmag=0.06, ccorr=40 * UA, seed=7)
    obs = im.observe(array, TINT, TADV, TSTART, TSTOP, BW,
                     ampcal=True, phasecal=True, ttype="direct", add_th_noise=False)
    mask = np.ones(im.imvec.size, dtype=bool)

    rng = np.random.default_rng(RNG_SEED)
    I = im.imvec
    Q, U, V = im.qvec, im.uvec, im.vvec
    P = np.sqrt(Q**2 + U**2 + V**2)
    n = I.size
    imcur = np.array([
        I * (1.0 + 0.05 * (rng.random(n) - 0.5)),
        np.clip((P / I) * (1.0 + 0.1 * (rng.random(n) - 0.5)), 0.02, 0.95),
        np.arctan2(U, Q) + 0.1 * (rng.random(n) - 0.5),
        np.clip(np.abs(np.arcsin(V / (P + 1e-30))) * (1.0 + 0.1 * (rng.random(n) - 0.5)),
                0.02, 1.5),
    ])
    return dict(obs=obs, prior=im.copy(), mask=mask, imcur=imcur, im=im)


def _frac_stats(numeric, exact):
    """Median and max of the elementwise fractional difference |num-ex|/|ex|."""
    floor = np.min(np.abs(exact)) * 1e-20 + 1e-100
    frac = np.abs((numeric - exact) / (np.abs(exact) + floor))
    return float(np.median(frac)), float(np.max(frac))


# ===========================================================================
# (2) Intensity chi^2 gradients vs finite differences  [direct + nfft]
# ===========================================================================
def _intensity_chisq_fd(setup, dtype, ttype):
    obs, prior, mask, imvec = setup["obs"], setup["prior"], setup["mask"], setup["imvec"]
    # debias=False: the obs is noise-free, so amplitude debiasing would subtract
    # sigma^2 from clean sub-noise amplitudes and drive log-amp data to NaN.
    data, sigma, A = iu.chisqdata(obs, prior, mask, dtype, ttype=ttype, debias=False)
    grad = iu.chisqgrad(imvec, A, data, sigma, dtype, ttype=ttype, mask=mask)
    y0 = iu.chisq(imvec, A, data, sigma, dtype, ttype=ttype, mask=mask)

    rng = np.random.default_rng(RNG_SEED)
    samp = rng.choice(imvec.size, size=min(N_SAMPLES, imvec.size), replace=False)
    numeric = np.empty(samp.size)
    for i, j in enumerate(samp):
        dx = max(FD_REL * abs(imvec[j]), FD_FLOOR)
        v2 = imvec.copy()
        v2[j] += dx
        numeric[i] = (iu.chisq(v2, A, data, sigma, dtype, ttype=ttype, mask=mask) - y0) / dx
    return _frac_stats(numeric, grad[samp])


class TestIntensityChisqGradFD:
    """Analytic chisqgrad matches forward FD of chisq for every DATATERM."""

    @pytest.mark.parametrize("dtype", INTENSITY_DATATERMS)
    @pytest.mark.parametrize("ttype", TTYPES)
    def test_fd(self, intensity_setup, dtype, ttype):
        med, mx = _intensity_chisq_fd(intensity_setup, dtype, ttype)
        assert med < CHISQ_FD_MEDIAN, f"{dtype}/{ttype}: median frac {med:.2e}"
        assert mx < CHISQ_FD_MAX, f"{dtype}/{ttype}: max frac {mx:.2e}"


# ===========================================================================
# (1) Intensity chi^2: nfft vs direct (value + gradient)
# ===========================================================================
@requires_nfft
class TestIntensityChisqNFFTvsDirect:
    """nfft and direct chi^2 (value + gradient) agree to Fourier accuracy.

    Independent of the FD checks: FD validates each transform against itself,
    so it cannot catch a self-consistent-but-wrong Fourier operator.
    """

    @pytest.mark.parametrize("dtype", INTENSITY_DATATERMS)
    def test_value_and_grad(self, intensity_setup, dtype):
        obs, prior, mask, imvec = (intensity_setup["obs"], intensity_setup["prior"],
                                   intensity_setup["mask"], intensity_setup["imvec"])
        vals, grads = {}, {}
        for tt in ("direct", "nfft"):
            data, sigma, A = iu.chisqdata(obs, prior, mask, dtype, ttype=tt, debias=False)
            vals[tt] = iu.chisq(imvec, A, data, sigma, dtype, ttype=tt, mask=mask)
            grads[tt] = iu.chisqgrad(imvec, A, data, sigma, dtype, ttype=tt, mask=mask)
        vfrac = abs((vals["direct"] - vals["nfft"]) / abs(vals["direct"]))
        assert vfrac < NFFT_VALUE_FRAC, f"{dtype}: value frac {vfrac:.2e}"
        gmed, _ = _frac_stats(grads["nfft"], grads["direct"])
        rel_l2 = np.linalg.norm(grads["nfft"] - grads["direct"]) / np.linalg.norm(grads["direct"])
        assert gmed < NFFT_GRAD_MEDIAN, f"{dtype}: grad median frac {gmed:.2e}"
        assert rel_l2 < NFFT_GRAD_REL_L2, f"{dtype}: grad rel L2 {rel_l2:.2e}"


# ===========================================================================
# (2) Intensity regularizer gradients vs finite differences
# ===========================================================================
def _intensity_reg_fd(setup, rtype):
    im, imvec, mask = setup["im"], setup["imvec"], setup["mask"]
    nprior = np.ones_like(imvec)
    nprior = nprior * np.sum(imvec) / np.sum(nprior)
    flux = float(np.sum(imvec))
    kwargs = dict(beam_size=20 * UA, alpha_A=5000.0, epsilon_tv=0.0, norm_reg=True)

    y0 = iu.regularizer(imvec, nprior, mask, flux, im.xdim, im.ydim, im.psize, rtype, **kwargs)
    grad = iu.regularizergrad(imvec, nprior, mask, flux, im.xdim, im.ydim, im.psize, rtype, **kwargs)

    rng = np.random.default_rng(RNG_SEED)
    samp = rng.choice(imvec.size, size=min(N_SAMPLES, imvec.size), replace=False)
    numeric = np.empty(samp.size)
    for i, j in enumerate(samp):
        dx = max(FD_REL * abs(imvec[j]), FD_FLOOR)
        v2 = imvec.copy()
        v2[j] += dx
        y1 = iu.regularizer(v2, nprior, mask, flux, im.xdim, im.ydim, im.psize, rtype, **kwargs)
        numeric[i] = (y1 - y0) / dx
    return _frac_stats(numeric, grad[samp])


class TestIntensityRegGradFD:
    """Analytic regularizergrad matches forward FD of regularizer."""

    @pytest.mark.parametrize("rtype", INTENSITY_REGS)
    def test_fd(self, intensity_setup, rtype):
        med, mx = _intensity_reg_fd(intensity_setup, rtype)
        assert med < REG_FD_MEDIAN, f"{rtype}: median frac {med:.2e}"
        assert mx < REG_FD_MAX, f"{rtype}: max frac {mx:.2e}"


# ===========================================================================
# (2) Pol chi^2 gradients vs finite differences  [direct + nfft, all 4 slots]
# ===========================================================================
def _pol_data(setup, dtype, ttype):
    data, sigma, A = pu.polchisqdata(setup["obs"], setup["prior"], setup["mask"], dtype, ttype=ttype)
    return A, data, sigma


class TestPolChisqGradFD:
    """polchisqgrad matches central FD of polchisq in all four physical slots
    (driven with pol_solve=[1,1,1,1] so the cross-coupling slots are exercised);
    the vvis EVPA slot (2) must be identically zero -- V is independent of EVPA.
    """

    @pytest.mark.parametrize("dtype", POL_DATATERMS)
    @pytest.mark.parametrize("ttype", TTYPES)
    def test_fd(self, pol_setup, dtype, ttype):
        mask, imcur = pol_setup["mask"], pol_setup["imcur"]
        A, data, sigma = _pol_data(pol_setup, dtype, ttype)
        grad = pu.polchisqgrad(imcur, A, data, sigma, dtype, ttype=ttype, mask=mask,
                               pol_solve=np.array([1, 1, 1, 1]))
        if dtype == "vvis":
            np.testing.assert_array_equal(grad[2], 0.0)

        rng = np.random.default_rng(RNG_SEED)
        n = imcur.shape[1]
        samp = rng.choice(n, size=min(30, n), replace=False)
        frac = []
        for slot in range(4):
            for j in samp:
                dx = max(POL_FD_REL * abs(imcur[slot, j]), POL_FD_FLOOR)
                ip, im_ = imcur.copy(), imcur.copy()
                ip[slot, j] += dx
                im_[slot, j] -= dx
                fd = (pu.polchisq(ip, A, data, sigma, dtype, ttype=ttype, mask=mask)
                      - pu.polchisq(im_, A, data, sigma, dtype, ttype=ttype, mask=mask)) / (2 * dx)
                ex = grad[slot, j]
                frac.append(abs(ex - fd) / max(abs(ex), abs(fd), POL_FD_FLOOR))
        frac = np.array(frac)
        assert np.median(frac) < POL_CHISQ_FD_MEDIAN, f"{dtype}/{ttype}: median {np.median(frac):.2e}"
        assert np.max(frac) < POL_CHISQ_FD_MAX, f"{dtype}/{ttype}: max {np.max(frac):.2e}"


# ===========================================================================
# (1) Pol chi^2: nfft vs direct (value + gradient)
# ===========================================================================
@requires_nfft
class TestPolChisqNFFTvsDirect:
    """nfft and direct pol chi^2 (value + gradient) agree to Fourier accuracy."""

    @pytest.mark.parametrize("dtype", POL_DATATERMS)
    def test_value_and_grad(self, pol_setup, dtype):
        mask, imcur = pol_setup["mask"], pol_setup["imcur"]
        vals, grads = {}, {}
        for tt in ("direct", "nfft"):
            A, data, sigma = _pol_data(pol_setup, dtype, tt)
            vals[tt] = pu.polchisq(imcur, A, data, sigma, dtype, ttype=tt, mask=mask)
            grads[tt] = pu.polchisqgrad(imcur, A, data, sigma, dtype, ttype=tt, mask=mask,
                                        pol_solve=np.array([1, 1, 1, 1]))
        vfrac = abs((vals["direct"] - vals["nfft"]) / abs(vals["direct"]))
        assert vfrac < NFFT_VALUE_FRAC, f"{dtype}: value frac {vfrac:.2e}"
        gmed, _ = _frac_stats(grads["nfft"], grads["direct"])
        rel_l2 = np.linalg.norm(grads["nfft"] - grads["direct"]) / np.linalg.norm(grads["direct"])
        assert gmed < NFFT_GRAD_MEDIAN, f"{dtype}: grad median frac {gmed:.2e}"
        assert rel_l2 < NFFT_GRAD_REL_L2, f"{dtype}: grad rel L2 {rel_l2:.2e}"


# ===========================================================================
# (2) Pol regularizer gradients vs finite differences  [all four slots]
# ===========================================================================
class TestPolRegGradFD:
    """polregularizergrad matches central FD of polregularizer in all four
    physical slots (pol_solve=[1,1,1,1] so the cross-coupling slots are checked);
    reuses the chisq pol image, whose per-pixel jitter keeps the TV denominators
    non-degenerate. Central differences keep the tan(psi)-steep entropy slots
    well-conditioned even at the realistic (large-psi) circular-pol pixels."""

    @pytest.mark.parametrize("rtype", POL_REGS)
    def test_fd(self, pol_setup, rtype):
        im, imarr, mask = pol_setup["im"], pol_setup["imcur"], pol_setup["mask"]
        priorarr = np.zeros_like(imarr)
        flux = im.total_flux()
        kwargs = dict(beam_size=20 * UA, norm_reg=True)
        args = (priorarr, mask, flux, flux, flux, im.xdim, im.ydim, im.psize, rtype)
        grad = pu.polregularizergrad(imarr, *args, pol_solve=np.array([1, 1, 1, 1]), **kwargs)

        rng = np.random.default_rng(RNG_SEED)
        nimage = imarr.shape[1]
        samp = rng.choice(nimage, size=min(N_SAMPLES, nimage), replace=False)
        frac = []
        for slot in range(4):
            for j in samp:
                dx = max(POL_FD_REL * abs(imarr[slot, j]), POL_FD_FLOOR)
                a, b = imarr.copy(), imarr.copy()
                a[slot, j] += dx
                b[slot, j] -= dx
                num = (pu.polregularizer(a, *args, **kwargs)
                       - pu.polregularizer(b, *args, **kwargs)) / (2 * dx)
                ex = grad[slot, j]
                frac.append(abs(num - ex) / max(abs(ex), abs(num), POL_FD_FLOOR))
        frac = np.array(frac)
        assert np.median(frac) < POL_REG_FD_MEDIAN, f"{rtype}: median frac {np.median(frac):.2e}"
        assert np.max(frac) < POL_REG_FD_MAX, f"{rtype}: max frac {np.max(frac):.2e}"
