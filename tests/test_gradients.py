"""Tests for analytic gradient correctness via numeric finite differences.

Verifies that analytic chi-squared gradients and regularizer gradients
match numeric finite differences computed element-wise. All tests use a
32x48 image so xdim != ydim exercises the rectangular-image code paths.
"""

import numpy as np
import pytest

import ehtim as eh
import ehtim.imaging.imager_utils as iu
from ehtim.imaging.imager_utils import chisq, chisqdata, chisqgrad

# Data types, regularizers, and transform types to test
DATATERMS = ["vis", "bs", "amp", "cphase", "camp", "logcamp"]
REGULARIZERS = ["simple", "gs", "l1w", "tv", "tv2"]
TTYPES = ["direct"]  # expand to ["direct", "fast", "nfft"] to test other transforms

# Diagonalized closure gradients, checked on the supported transforms.
# ('fast'/plain-FFT is omitted; that mode is slated for deprecation.)
DATATERMS_DIAG = ["cphase_diag", "logcamp_diag"]
TTYPES_DIAG = ["direct", "nfft"]

# Tolerances (calibrated on 32x48 synthetic Gaussian with relative step size)
CHISQ_GRAD_MEDIAN_TOL = 0.001
CHISQ_GRAD_MAX_TOL = 0.01
REG_GRAD_MEDIAN_TOL = 0.01
REG_GRAD_MAX_TOL = 0.10

# ---------------------------------------------------------------------------
# Numeric gradient parameters
# ---------------------------------------------------------------------------
N_GRAD_SAMPLES = 100
RNG_SEED = 4
GRAD_DX_REL = 1e-8    # relative step size per pixel
GRAD_DX_FLOOR = 1e-12  # absolute minimum step size

# ---------------------------------------------------------------------------
# Regularizer optional parameters
# ---------------------------------------------------------------------------
BEAM_SIZE = 20.0 * eh.RADPERUAS
ALPHA_A = 5000.0
EPSILON_TV = 0.0

# ---------------------------------------------------------------------------
# Observation parameters
# ---------------------------------------------------------------------------
TINT_SEC = 5
TADV_SEC = 600
TSTART_HR = 0
TSTOP_HR = 24
BW_HZ = 4e9


@pytest.fixture(scope="module")
def grad_setup(eht_array, make_rect_image):
    """Set up observation and test image from 32x48 synthetic Gaussian.

    Uses xdim != ydim so the rectangular-image code paths are exercised.
    """
    im = make_rect_image(32, 48)
    im.imvec = im.imvec * 2.0 / im.total_flux()  # normalize to 2 Jy

    obs = im.observe(
        eht_array, TINT_SEC, TADV_SEC, TSTART_HR, TSTOP_HR, BW_HZ,
        sgrscat=False, ampcal=True, phasecal=True,
        ttype="direct", add_th_noise=False,
    )

    prior = im.copy()
    im2 = prior.copy()

    rng = np.random.default_rng(RNG_SEED)
    im2.imvec *= 1.0 + (rng.random(len(im2.imvec)) - 0.5) / 10.0
    im2.imvec += (1.0 + (rng.random(len(im2.imvec)) - 0.5) / 10.0) * np.mean(im2.imvec)

    mask = im2.imvec > 0.5 * np.median(im2.imvec)
    test_imvec = im2.imvec[mask] if np.any(~mask) else im2.imvec

    return {
        "obs": obs,
        "prior": prior,
        "test_imvec": test_imvec,
        "mask": mask,
        "im": im,
    }


class TestChisqGradientFiniteDiff:
    """Analytic chi-squared gradients match numeric finite differences."""

    @pytest.mark.parametrize("dtype", DATATERMS)
    @pytest.mark.parametrize("ttype", TTYPES)
    def test_median_frac_diff(self, grad_setup, dtype, ttype):
        median_frac, _ = _chisq_gradient_check(grad_setup, dtype, ttype)
        assert median_frac < CHISQ_GRAD_MEDIAN_TOL, (
            f"{dtype} ({ttype}) median fractional gradient diff = {median_frac:.6f}"
        )

    @pytest.mark.parametrize("dtype", DATATERMS)
    @pytest.mark.parametrize("ttype", TTYPES)
    def test_max_frac_diff(self, grad_setup, dtype, ttype):
        _, max_frac = _chisq_gradient_check(grad_setup, dtype, ttype)
        assert max_frac < CHISQ_GRAD_MAX_TOL, (
            f"{dtype} ({ttype}) max fractional gradient diff = {max_frac:.6f}"
        )


class TestChisqGradientFiniteDiffDiag:
    """Diagonalized-closure gradients match finite differences.

    Pins the vectorized per-time-block matvec in chisqgrad_{cphase,logcamp}_diag
    against numeric finite differences, at the same tolerance as the standard
    closures.
    """

    @pytest.mark.parametrize("dtype", DATATERMS_DIAG)
    @pytest.mark.parametrize("ttype", TTYPES_DIAG)
    def test_median_frac_diff(self, grad_setup, dtype, ttype):
        median_frac, _ = _chisq_gradient_check(grad_setup, dtype, ttype)
        assert median_frac < CHISQ_GRAD_MEDIAN_TOL, (
            f"{dtype} ({ttype}) median fractional gradient diff = {median_frac:.6f}"
        )

    @pytest.mark.parametrize("dtype", DATATERMS_DIAG)
    @pytest.mark.parametrize("ttype", TTYPES_DIAG)
    def test_max_frac_diff(self, grad_setup, dtype, ttype):
        _, max_frac = _chisq_gradient_check(grad_setup, dtype, ttype)
        assert max_frac < CHISQ_GRAD_MAX_TOL, (
            f"{dtype} ({ttype}) max fractional gradient diff = {max_frac:.6f}"
        )


def test_diag_chisq_nfft_matches_direct(grad_setup):
    """Block-diagonal nfft diag chisq agrees with the direct-DFT diag chisq.

    The nfft diagonalized-closure terms apply the per-block decorrelating
    transforms as one block-diagonal matmul; the direct terms loop. Both share
    the same transforms and measured closures and differ only by Fourier
    accuracy, so their chi^2 must agree closely. Guards the block-diagonal
    restructure against a self-consistent-but-wrong error (which finite
    differences alone would not catch).
    """
    obs, prior, mask = grad_setup["obs"], grad_setup["prior"], grad_setup["mask"]
    iv = grad_setup["test_imvec"]
    for dtype in DATATERMS_DIAG:
        cdir = chisqdata(obs, prior, mask, dtype, ttype="direct")
        cnf = chisqdata(obs, prior, mask, dtype, ttype="nfft")
        chi_dir = chisq(iv, cdir[2], cdir[0], cdir[1], dtype, ttype="direct", mask=mask)
        chi_nf = chisq(iv, cnf[2], cnf[0], cnf[1], dtype, ttype="nfft", mask=mask)
        assert abs(chi_dir - chi_nf) <= 1e-2 * abs(chi_dir), (
            f"{dtype}: direct={chi_dir:.6g} vs nfft={chi_nf:.6g}"
        )


class TestRegularizerGradientFiniteDiff:
    """Analytic regularizer gradients match numeric finite differences."""

    @pytest.mark.parametrize("rtype", REGULARIZERS)
    def test_median_frac_diff(self, grad_setup, rtype):
        median_frac, _ = _reg_gradient_check(grad_setup, rtype)
        assert median_frac < REG_GRAD_MEDIAN_TOL, (
            f"{rtype} median fractional gradient diff = {median_frac:.6f}"
        )

    @pytest.mark.parametrize("rtype", REGULARIZERS)
    def test_max_frac_diff(self, grad_setup, rtype):
        _, max_frac = _reg_gradient_check(grad_setup, rtype)
        assert max_frac < REG_GRAD_MAX_TOL, (
            f"{rtype} max fractional gradient diff = {max_frac:.6f}"
        )


def _chisq_gradient_check(grad_setup, dtype, ttype):
    """Compare analytic vs numeric chi-squared gradient on subsampled pixels."""
    obs = grad_setup["obs"]
    prior = grad_setup["prior"]
    mask = grad_setup["mask"]
    test_imvec = grad_setup["test_imvec"]

    cdata = chisqdata(obs, prior, mask, dtype, ttype=ttype)
    grad_exact = chisqgrad(test_imvec, cdata[2], cdata[0], cdata[1], dtype, ttype=ttype, mask=mask)
    y0 = chisq(test_imvec, cdata[2], cdata[0], cdata[1], dtype, ttype=ttype, mask=mask)

    rng = np.random.default_rng(RNG_SEED)
    sample_idx = rng.choice(len(test_imvec), size=N_GRAD_SAMPLES, replace=False)

    grad_numeric = np.zeros(N_GRAD_SAMPLES)
    for i, j in enumerate(sample_idx):
        dx = max(GRAD_DX_REL * abs(test_imvec[j]), GRAD_DX_FLOOR)
        imvec2 = test_imvec.copy()
        imvec2[j] += dx
        y1 = chisq(imvec2, cdata[2], cdata[0], cdata[1], dtype, ttype=ttype, mask=mask)
        grad_numeric[i] = (y1 - y0) / dx

    grad_sampled = grad_exact[sample_idx]
    compare_floor = np.min(np.abs(grad_sampled)) * 1e-20 + 1e-100
    frac_diff = np.abs((grad_numeric - grad_sampled) / (np.abs(grad_sampled) + compare_floor))
    return np.median(frac_diff), np.max(frac_diff)


def _reg_gradient_check(grad_setup, rtype):
    """Compare analytic vs numeric regularizer gradient on subsampled pixels."""
    test_imvec = grad_setup["test_imvec"]
    im = grad_setup["im"]

    nprior = np.ones_like(test_imvec)
    nprior = nprior * np.sum(test_imvec) / np.sum(nprior)
    mask = grad_setup["mask"]
    flux = np.sum(test_imvec)

    kwargs = dict(
        beam_size=BEAM_SIZE, alpha_A=ALPHA_A, epsilon_tv=EPSILON_TV, norm_reg=True,
    )

    y0 = iu.regularizer(test_imvec, nprior, mask, flux, im.xdim, im.ydim, im.psize, rtype, **kwargs)
    grad_exact = iu.regularizergrad(test_imvec, nprior, mask, flux, im.xdim, im.ydim, im.psize, rtype, **kwargs)

    rng = np.random.default_rng(RNG_SEED)
    sample_idx = rng.choice(len(test_imvec), size=N_GRAD_SAMPLES, replace=False)

    grad_numeric = np.zeros(N_GRAD_SAMPLES)
    for i, j in enumerate(sample_idx):
        dx = max(GRAD_DX_REL * abs(test_imvec[j]), GRAD_DX_FLOOR)
        imvec2 = test_imvec.copy()
        imvec2[j] += dx
        y1 = iu.regularizer(imvec2, nprior, mask, flux, im.xdim, im.ydim, im.psize, rtype, **kwargs)
        grad_numeric[i] = (y1 - y0) / dx

    grad_sampled = grad_exact[sample_idx]
    compare_floor = np.min(np.abs(grad_sampled)) * 1e-20 + 1e-100
    frac_diff = np.abs((grad_numeric - grad_sampled) / (np.abs(grad_sampled) + compare_floor))
    return np.median(frac_diff), np.max(frac_diff)
