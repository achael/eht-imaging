"""Tests for chi-squared consistency across transform types (direct, fast, nfft).

Verifies that chi-squared values and gradients agree between DFT, FFT, and NFFT
for all standard data types.
"""

import numpy as np
import pytest

import ehtim as eh
from ehtim.imaging.imager_utils import chisq, chisqdata, chisqgrad

# Observation parameters (must match conftest.py)
TINT_SEC = 5
TADV_SEC = 600
TSTART_HR = 0
TSTOP_HR = 24
BW_HZ = 4e9

# Data types to test
DATATERMS = ["vis", "bs", "amp", "cphase", "camp", "logcamp"]

# Transform type pairs to compare
TTYPE_PAIRS = [("direct", "fast"), ("direct", "nfft"), ("nfft", "fast")]

# NFFT max gradient tolerance is much wider at 32x32 resolution
GRAD_MAX_TOL_NFFT = 10.0

# Tolerances (calibrated on 32x32 SgrA image)
CHISQ_FRAC_TOL = 0.01
GRAD_MEDIAN_TOL = 0.05
GRAD_MAX_TOL = 0.25

# ---------------------------------------------------------------------------
# chisqdata optional parameters (explicit for tracking across refactors)
# ---------------------------------------------------------------------------

# Common to all ttypes and dtypes
SYSTEMATIC_NOISE = 0.0
SNRCUT = 0.0
DEBIAS = True
WEIGHTING = "natural"

# Closure-specific
MAXSET = False
CP_UV_MIN = False
SYSTEMATIC_CPHASE_NOISE = 0.0

# FFT/NFFT-specific
FFT_PAD_FACTOR = 10
P_RAD = 12
CONV_FUNC = "gaussian"
FFT_INTERP_ORDER = 3


def _make_rect_image(xdim, ydim, psize=None):
    """Construct a Gaussian image with arbitrary (xdim, ydim) dimensions."""
    if psize is None:
        psize = 200 * eh.RADPERUAS / max(xdim, ydim)
    image_arr = np.zeros((ydim, xdim))
    for i in range(ydim):
        for j in range(xdim):
            x = (j - xdim / 2) * psize
            y = (i - ydim / 2) * psize
            sigma = 50 * eh.RADPERUAS
            image_arr[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    image_arr /= image_arr.sum()
    return eh.image.Image(
        image_arr, psize, 17.761, -29.0,
        polrep="stokes", pol_prim="I", rf=230e9,
    )


@pytest.fixture(scope="module")
def chisq_setup(sgra_im_small, eht_array):
    """Set up observation and test image for chi-squared comparison tests.

    Uses a single DFT observation as ground truth. The chi-squared functions
    are then evaluated with different transform matrices (direct, fast, nfft)
    on the same data, testing that the transform approximations agree.
    """
    im = sgra_im_small

    obs = im.observe(
        eht_array, TINT_SEC, TADV_SEC, TSTART_HR, TSTOP_HR, BW_HZ,
        sgrscat=False, ampcal=True, phasecal=False,
        ttype="direct", add_th_noise=False,
    )

    prior = eh.image.make_square(obs, im.xdim, im.xdim * im.psize)
    prior = prior.add_gauss(im.total_flux(), (50 * eh.RADPERUAS, 50 * eh.RADPERUAS, 0, 0, 0))

    im2 = prior.copy()
    rng = np.random.RandomState(42)
    im2.imvec *= 1.0 + (rng.rand(len(im2.imvec)) - 0.5) / 10.0
    im2.imvec += rng.rand(len(im2.imvec)) / 10.0 * im.imvec

    mask = im2.imvec > 0
    test_imvec = im2.imvec[mask] if np.any(~mask) else im2.imvec

    return {
        "obs": obs,
        "prior": prior,
        "test_imvec": test_imvec,
        "mask": mask,
        "im": im,
    }


@pytest.fixture(scope="module")
def chisq_setup_rect(eht_array):
    """Set up 32x48 rectangular Gaussian image observation for chi-squared tests."""
    im = _make_rect_image(32, 48)
    im.imvec = im.imvec * 2.0 / im.total_flux()  # normalize to 2 Jy

    obs = im.observe(
        eht_array, TINT_SEC, TADV_SEC, TSTART_HR, TSTOP_HR, BW_HZ,
        sgrscat=False, ampcal=True, phasecal=False,
        ttype="direct", add_th_noise=False,
    )

    prior = im.copy()

    im2 = prior.copy()
    rng = np.random.RandomState(4)
    im2.imvec *= 1.0 + (rng.rand(len(im2.imvec)) - 0.5) / 10.0
    im2.imvec += rng.rand(len(im2.imvec)) / 10.0 * im.imvec

    mask = im2.imvec > 0
    test_imvec = im2.imvec[mask] if np.any(~mask) else im2.imvec

    return {
        "obs": obs,
        "prior": prior,
        "test_imvec": test_imvec,
        "mask": mask,
        "im": im,
    }


def _chisq_kwargs(ttype):
    """Return extra kwargs for chisqdata based on ttype."""
    kwargs = {
        "systematic_noise": SYSTEMATIC_NOISE,
        "snrcut": SNRCUT,
        "debias": DEBIAS,
        "weighting": WEIGHTING,
        "maxset": MAXSET,
        "cp_uv_min": CP_UV_MIN,
        "systematic_cphase_noise": SYSTEMATIC_CPHASE_NOISE,
    }
    if ttype in ("fast", "nfft"):
        kwargs["fft_pad_factor"] = FFT_PAD_FACTOR
        kwargs["p_rad"] = P_RAD
        kwargs["conv_func"] = CONV_FUNC
        kwargs["order"] = FFT_INTERP_ORDER
    return kwargs


class TestChisqConsistency:
    """Chi-squared values agree across transform types.

    Same observation data is used for both ttypes. Only the transform matrix
    (DFT vs FFT) differs, so differences reflect the approximation error.
    """

    @pytest.mark.parametrize("dtype", DATATERMS)
    @pytest.mark.parametrize("pair", TTYPE_PAIRS, ids=lambda p: f"{p[0]}-{p[1]}")
    def test_chisq_values(self, chisq_setup, dtype, pair):
        ttype_a, ttype_b = pair
        if "nfft" in (ttype_a, ttype_b):
            pytest.importorskip("pynfft")
        obs = chisq_setup["obs"]
        prior = chisq_setup["prior"]
        mask = chisq_setup["mask"]
        test_imvec = chisq_setup["test_imvec"]

        cdata_a = chisqdata(obs, prior, mask, dtype, ttype=ttype_a, **_chisq_kwargs(ttype_a))
        cdata_b = chisqdata(obs, prior, mask, dtype, ttype=ttype_b, **_chisq_kwargs(ttype_b))

        chi_a = chisq(test_imvec, cdata_a[2], cdata_a[0], cdata_a[1], dtype, ttype=ttype_a, mask=mask)
        chi_b = chisq(test_imvec, cdata_b[2], cdata_b[0], cdata_b[1], dtype, ttype=ttype_b, mask=mask)

        frac_diff = abs((chi_a - chi_b) / abs(chi_a))
        print(f"  {dtype} {ttype_a}-{ttype_b}: chisq frac diff = {frac_diff:.6f}")
        assert frac_diff < CHISQ_FRAC_TOL, (
            f"{dtype} {ttype_a}-{ttype_b}: chisq frac diff = {frac_diff:.6f}"
        )


class TestChisqGradConsistency:
    """Chi-squared gradients agree across transform types."""

    @pytest.mark.parametrize("dtype", DATATERMS)
    @pytest.mark.parametrize("pair", TTYPE_PAIRS, ids=lambda p: f"{p[0]}-{p[1]}")
    def test_grad_median_frac_diff(self, chisq_setup, dtype, pair):
        median_frac, _ = _gradient_comparison(chisq_setup, dtype, pair)
        assert median_frac < GRAD_MEDIAN_TOL, (
            f"{dtype} {pair[0]}-{pair[1]}: grad median frac diff = {median_frac:.6f}"
        )

    @pytest.mark.parametrize("dtype", DATATERMS)
    @pytest.mark.parametrize("pair", TTYPE_PAIRS, ids=lambda p: f"{p[0]}-{p[1]}")
    def test_grad_max_frac_diff(self, chisq_setup, dtype, pair):
        _, max_frac = _gradient_comparison(chisq_setup, dtype, pair)
        tol = GRAD_MAX_TOL_NFFT if "nfft" in pair else GRAD_MAX_TOL
        assert max_frac < tol, (
            f"{dtype} {pair[0]}-{pair[1]}: grad max frac diff = {max_frac:.6f}"
        )


def _gradient_comparison(chisq_setup, dtype, pair):
    """Compute median and max fractional gradient diff between two ttypes."""
    ttype_a, ttype_b = pair
    if "nfft" in (ttype_a, ttype_b):
        pytest.importorskip("pynfft")
    obs = chisq_setup["obs"]
    prior = chisq_setup["prior"]
    mask = chisq_setup["mask"]
    test_imvec = chisq_setup["test_imvec"]

    cdata_a = chisqdata(obs, prior, mask, dtype, ttype=ttype_a, **_chisq_kwargs(ttype_a))
    cdata_b = chisqdata(obs, prior, mask, dtype, ttype=ttype_b, **_chisq_kwargs(ttype_b))

    grad_a = chisqgrad(test_imvec, cdata_a[2], cdata_a[0], cdata_a[1], dtype, ttype=ttype_a, mask=mask)
    grad_b = chisqgrad(test_imvec, cdata_b[2], cdata_b[0], cdata_b[1], dtype, ttype=ttype_b, mask=mask)

    compare_floor = np.min(np.abs(grad_a)) * 1e-20 + 1e-100
    frac_diff = np.abs((grad_a - grad_b) / (np.abs(grad_a) + compare_floor))
    print(f"  {dtype} {ttype_a}-{ttype_b}: grad median={np.median(frac_diff):.6f} max={np.max(frac_diff):.6f}")
    return np.median(frac_diff), np.max(frac_diff)


# ---------------------------------------------------------------------------
# Rectangular image tests (xdim != ydim)
# ---------------------------------------------------------------------------


class TestChisqConsistencyRect:
    """Chi-squared values agree across transform types on rectangular images."""

    @pytest.mark.parametrize("dtype", DATATERMS)
    @pytest.mark.parametrize("pair", TTYPE_PAIRS, ids=lambda p: f"{p[0]}-{p[1]}")
    def test_chisq_values(self, chisq_setup_rect, dtype, pair):
        ttype_a, ttype_b = pair
        if "nfft" in (ttype_a, ttype_b):
            pytest.importorskip("pynfft")
        obs = chisq_setup_rect["obs"]
        prior = chisq_setup_rect["prior"]
        mask = chisq_setup_rect["mask"]
        test_imvec = chisq_setup_rect["test_imvec"]

        cdata_a = chisqdata(obs, prior, mask, dtype, ttype=ttype_a, **_chisq_kwargs(ttype_a))
        cdata_b = chisqdata(obs, prior, mask, dtype, ttype=ttype_b, **_chisq_kwargs(ttype_b))

        chi_a = chisq(test_imvec, cdata_a[2], cdata_a[0], cdata_a[1], dtype, ttype=ttype_a, mask=mask)
        chi_b = chisq(test_imvec, cdata_b[2], cdata_b[0], cdata_b[1], dtype, ttype=ttype_b, mask=mask)

        frac_diff = abs((chi_a - chi_b) / abs(chi_a))
        print(f"  {dtype} {ttype_a}-{ttype_b} (rect): chisq frac diff = {frac_diff:.6f}")
        assert frac_diff < CHISQ_FRAC_TOL, (
            f"{dtype} {ttype_a}-{ttype_b} (rect): chisq frac diff = {frac_diff:.6f}"
        )


class TestChisqGradConsistencyRect:
    """Chi-squared gradients agree across transform types on rectangular images."""

    @pytest.mark.parametrize("dtype", DATATERMS)
    @pytest.mark.parametrize("pair", TTYPE_PAIRS, ids=lambda p: f"{p[0]}-{p[1]}")
    def test_grad_median_frac_diff(self, chisq_setup_rect, dtype, pair):
        median_frac, _ = _gradient_comparison(chisq_setup_rect, dtype, pair)
        assert median_frac < GRAD_MEDIAN_TOL, (
            f"{dtype} {pair[0]}-{pair[1]} (rect): grad median frac diff = {median_frac:.6f}"
        )

    @pytest.mark.parametrize("dtype", DATATERMS)
    @pytest.mark.parametrize("pair", TTYPE_PAIRS, ids=lambda p: f"{p[0]}-{p[1]}")
    def test_grad_max_frac_diff(self, chisq_setup_rect, dtype, pair):
        _, max_frac = _gradient_comparison(chisq_setup_rect, dtype, pair)
        tol = GRAD_MAX_TOL_NFFT if "nfft" in pair else GRAD_MAX_TOL
        assert max_frac < tol, (
            f"{dtype} {pair[0]}-{pair[1]} (rect): grad max frac diff = {max_frac:.6f}"
        )
