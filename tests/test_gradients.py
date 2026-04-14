"""Tests for analytic gradient correctness via numeric finite differences.

Verifies that analytic chi-squared gradients and regularizer gradients
match numeric finite differences computed element-wise.
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

# Tolerances (calibrated on 32x32 M87 image with relative step size)
CHISQ_GRAD_MEDIAN_TOL = 0.001
CHISQ_GRAD_MAX_TOL = 0.01
REG_GRAD_MEDIAN_TOL = 0.01
REG_GRAD_MAX_TOL = 0.10

# ---------------------------------------------------------------------------
# Numeric gradient parameters
# ---------------------------------------------------------------------------
N_GRAD_SAMPLES = 100
GRAD_SEED = 42
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
def grad_setup(m87_im_small, eht_array):
    """Set up observation and test image for gradient verification."""
    im = m87_im_small

    obs = im.observe(
        eht_array, TINT_SEC, TADV_SEC, TSTART_HR, TSTOP_HR, BW_HZ,
        sgrscat=False, ampcal=True, phasecal=True,
        ttype="direct", add_th_noise=False,
    )

    prior = im.copy()
    im2 = prior.copy()

    rng = np.random.RandomState(42)
    im2.imvec *= 1.0 + (rng.rand(len(im2.imvec)) - 0.5) / 10.0
    im2.imvec += (1.0 + (rng.rand(len(im2.imvec)) - 0.5) / 10.0) * np.mean(im2.imvec)

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
        if ttype == "nfft":
            pytest.importorskip("pynfft")
        median_frac, _ = _chisq_gradient_check(grad_setup, dtype, ttype)
        assert median_frac < CHISQ_GRAD_MEDIAN_TOL, (
            f"{dtype} ({ttype}) median fractional gradient diff = {median_frac:.6f}"
        )

    @pytest.mark.parametrize("dtype", DATATERMS)
    @pytest.mark.parametrize("ttype", TTYPES)
    def test_max_frac_diff(self, grad_setup, dtype, ttype):
        if ttype == "nfft":
            pytest.importorskip("pynfft")
        _, max_frac = _chisq_gradient_check(grad_setup, dtype, ttype)
        assert max_frac < CHISQ_GRAD_MAX_TOL, (
            f"{dtype} ({ttype}) max fractional gradient diff = {max_frac:.6f}"
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

    rng = np.random.default_rng(GRAD_SEED)
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

    rng = np.random.default_rng(GRAD_SEED)
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
