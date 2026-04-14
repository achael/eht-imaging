"""Tests for ehtim regularizer functions.

Verifies that all regularizer types return finite values and that
analytic gradients match numeric finite differences.
"""

import numpy as np
import pytest

import ehtim as eh
import ehtim.imaging.imager_utils as iu

# Tolerances for gradient checks (calibrated on 32x32 SgrA image)
MEDIAN_FRAC_TOL = 0.05
MAX_FRAC_TOL = 0.6

# No regularizers skipped from gradient checks
SKIP_GRAD_RTYPES = set()

# Number of pixels to subsample for numeric gradient (seeded for reproducibility)
N_GRAD_SAMPLES = 100
GRAD_SEED = 42


@pytest.fixture(scope="module")
def reg_setup(sgra_im_small):
    """Set up regularizer test data from 32x32 SgrA image."""
    im = sgra_im_small.copy()
    im.pulse = eh.observing.pulses.deltaPulse2D
    mask = im.imvec > 0
    imvec = im.imvec
    nprior = np.ones_like(imvec)
    nprior = nprior / np.sum(nprior)
    flux = im.total_flux() * 0.95
    return im, imvec, nprior, mask, flux


class TestRegularizerValues:
    """All regularizer types return finite values."""

    @pytest.mark.parametrize("rtype", iu.REGULARIZERS)
    @pytest.mark.parametrize("norm_reg", [True, False], ids=["normalized", "unnormalized"])
    def test_returns_finite(self, reg_setup, rtype, norm_reg):
        im, imvec, nprior, mask, flux = reg_setup
        kwargs = dict(beam_size=20.0 * eh.RADPERUAS, alpha_A=5000.0, norm_reg=norm_reg)
        if rtype == "rgauss":
            kwargs["major"] = 50.0 * eh.RADPERUAS
            kwargs["minor"] = 60.0 * eh.RADPERUAS
            kwargs["PA"] = np.pi / 3
        val = iu.regularizer(
            imvec, nprior, mask, flux,
            im.xdim, im.ydim, im.psize, rtype, **kwargs,
        )
        assert np.isfinite(val), f"{rtype} (norm_reg={norm_reg}) returned {val}"


class TestRegularizerGradients:
    """Analytic regularizer gradients match numeric finite differences."""

    @pytest.mark.parametrize("rtype", [r for r in iu.REGULARIZERS if r not in SKIP_GRAD_RTYPES])
    def test_median_frac_diff(self, reg_setup, rtype):
        median_frac, _ = _gradient_check(reg_setup, rtype)
        assert median_frac < MEDIAN_FRAC_TOL, (
            f"{rtype} median fractional gradient diff = {median_frac:.6f}"
        )

    @pytest.mark.parametrize("rtype", [r for r in iu.REGULARIZERS if r not in SKIP_GRAD_RTYPES])
    def test_max_frac_diff(self, reg_setup, rtype):
        _, max_frac = _gradient_check(reg_setup, rtype)
        assert max_frac < MAX_FRAC_TOL, (
            f"{rtype} max fractional gradient diff = {max_frac:.6f}"
        )


def _gradient_check(reg_setup, rtype):
    """Compute median and max fractional diff between analytic and numeric gradient."""
    im, imvec, nprior, mask, flux = reg_setup
    kwargs = dict(
        beam_size=20.0 * eh.RADPERUAS, alpha_A=5000.0, norm_reg=True,
    )
    if rtype == "rgauss":
        kwargs["major"] = 50.0 * eh.RADPERUAS
        kwargs["minor"] = 60.0 * eh.RADPERUAS
        kwargs["PA"] = np.pi / 3

    y0 = iu.regularizer(imvec, nprior, mask, flux, im.xdim, im.ydim, im.psize, rtype, **kwargs)
    grad_exact = iu.regularizergrad(imvec, nprior, mask, flux, im.xdim, im.ydim, im.psize, rtype, **kwargs)

    rng = np.random.default_rng(GRAD_SEED)
    sample_idx = rng.choice(len(imvec), size=N_GRAD_SAMPLES, replace=False)

    grad_numeric = np.zeros(N_GRAD_SAMPLES)
    for i, j in enumerate(sample_idx):
        dx = max(1e-8 * abs(imvec[j]), 1e-12)
        imvec2 = imvec.copy()
        imvec2[j] += dx
        y1 = iu.regularizer(imvec2, nprior, mask, flux, im.xdim, im.ydim, im.psize, rtype, **kwargs)
        grad_numeric[i] = (y1 - y0) / dx

    grad_exact_sampled = grad_exact[sample_idx]
    compare_floor = np.min(np.abs(grad_exact_sampled)) * 1e-20 + 1e-100
    frac_diff = np.abs((grad_numeric - grad_exact_sampled) / (np.abs(grad_exact_sampled) + compare_floor))
    return np.median(frac_diff), np.max(frac_diff)
