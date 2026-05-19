"""Tests for ehtim regularizer functions.

Verifies that all regularizer types return finite values and that
analytic gradients match numeric finite differences. All tests use a
32x48 image so xdim != ydim exercises the rectangular-image code paths.
"""

import numpy as np
import pytest

import ehtim as eh
import ehtim.imaging.imager_backend as ib
import ehtim.imaging.imager_utils as iu
import ehtim.imaging.multifreq_imager_utils as mfu
import ehtim.imaging.pol_imager_utils as pu

# Tolerances for gradient checks (calibrated on 32x48 synthetic Gaussian)
MEDIAN_FRAC_TOL = 0.05
MAX_FRAC_TOL = 0.6

# ---------------------------------------------------------------------------
# Regularizer optional parameters (explicit for tracking across refactors)
# ---------------------------------------------------------------------------
BEAM_SIZE = 20.0 * eh.RADPERUAS
ALPHA_A = 5000.0
EPSILON_TV = 0.0

# rgauss-specific parameters
RGAUSS_MAJOR = 50.0 * eh.RADPERUAS
RGAUSS_MINOR = 60.0 * eh.RADPERUAS
RGAUSS_PA = np.pi / 3

# ---------------------------------------------------------------------------
# Numeric gradient parameters
# ---------------------------------------------------------------------------
N_GRAD_SAMPLES = 100
RNG_SEED = 4
GRAD_DX_REL = 1e-8    # relative step size per pixel
GRAD_DX_FLOOR = 1e-12  # absolute minimum step size


@pytest.fixture(scope="module")
def reg_setup(make_rect_image):
    """Set up regularizer test data from 32x48 synthetic Gaussian.

    Uses xdim != ydim so the rectangular-image code paths are exercised.
    """
    im = make_rect_image(32, 48)
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
        val = iu.regularizer(
            imvec, nprior, mask, flux,
            im.xdim, im.ydim, im.psize, rtype,
            **_reg_kwargs(rtype, norm_reg=norm_reg),
        )
        assert np.isfinite(val), f"{rtype} (norm_reg={norm_reg}) returned {val}"


class TestRegularizerGradients:
    """Analytic regularizer gradients match numeric finite differences."""

    @pytest.mark.parametrize("rtype", iu.REGULARIZERS)
    def test_median_frac_diff(self, reg_setup, rtype):
        median_frac, _ = _gradient_check(reg_setup, rtype)
        assert median_frac < MEDIAN_FRAC_TOL, (
            f"{rtype} median fractional gradient diff = {median_frac:.6f}"
        )

    @pytest.mark.parametrize("rtype", iu.REGULARIZERS)
    def test_max_frac_diff(self, reg_setup, rtype):
        _, max_frac = _gradient_check(reg_setup, rtype)
        assert max_frac < MAX_FRAC_TOL, (
            f"{rtype} max fractional gradient diff = {max_frac:.6f}"
        )


class TestCenterOfMassRegularizer:
    """`reg_cm` / `reggrad_cm` semantics on a full-grid imvec.

    The COM constraint is `(sum(I*x))^2 + (sum(I*y))^2`, normalised. An
    off-centre image yields a strictly larger penalty than the centred one,
    and the analytic gradient must match a finite-difference reference.
    """

    @staticmethod
    def _kw(im):
        return dict(xdim=im.xdim, ydim=im.ydim, psize=im.psize,
                    flux=im.total_flux(), norm_reg=True)

    def test_off_center_image_more_positive_than_centered(self, gauss_im):
        """Shifting the source off-centre makes the (positive) penalty strictly larger."""
        mask = np.ones_like(gauss_im.imvec, dtype=bool)
        arr = gauss_im.imvec.reshape(gauss_im.ydim, gauss_im.xdim)
        shifted = np.roll(arr, gauss_im.xdim // 4, axis=1).flatten()
        val_centered = ib.compute_regularizer_term(gauss_im.imvec, 'cm', mask,
                                                   **self._kw(gauss_im))
        val_off = ib.compute_regularizer_term(shifted, 'cm', mask,
                                              **self._kw(gauss_im))
        assert val_off > val_centered
        assert val_off > 1e-6

    def test_gradient_matches_finite_difference(self, gauss_im):
        """Analytic reggrad_cm matches a central finite-difference gradient at sample pixels."""
        mask = np.ones_like(gauss_im.imvec, dtype=bool)
        kw = self._kw(gauss_im)
        rng = np.random.default_rng(0)
        imvec = gauss_im.imvec + 0.01 * rng.standard_normal(gauss_im.imvec.shape)
        grad_analytic = ib.compute_regularizergrad_term(imvec, 'cm', mask, **kw)
        h = 1e-6
        sample_idx = rng.choice(imvec.size, size=10, replace=False)
        for i in sample_idx:
            ep = imvec.copy()
            ep[i] += h
            em = imvec.copy()
            em[i] -= h
            fd = (ib.compute_regularizer_term(ep, 'cm', mask, **kw)
                  - ib.compute_regularizer_term(em, 'cm', mask, **kw)) / (2 * h)
            np.testing.assert_allclose(grad_analytic[i], fd, rtol=1e-4, atol=1e-12)

    def test_gradient_matches_finite_difference_partial_mask(self, gauss_im):
        """Exercise the embed-pre / mask-post-slice path of reg_cm/reggrad_cm.

        With a non-trivial mask, `reg_cm` calls `embed(imvec, mask, randomfloor=True)`
        which (with default `clipfloor=0`) zero-fills masked positions deterministically.
        The analytic gradient returned by `reggrad_cm` is length-nmask (post-slice),
        and FD perturbations on the masked-only vector must agree.
        """
        full_mask = gauss_im.imvec > 0.1 * gauss_im.imvec.max()
        assert not full_mask.all(), "test requires a partial (non-trivial) mask"
        kw = self._kw(gauss_im)
        rng = np.random.default_rng(1)
        imvec_masked = gauss_im.imvec[full_mask] + 0.01 * rng.standard_normal(full_mask.sum())
        grad_analytic = ib.compute_regularizergrad_term(imvec_masked, 'cm', full_mask, **kw)
        assert grad_analytic.shape == (full_mask.sum(),), "gradient must be length-nmask"
        h = 1e-6
        sample_idx = rng.choice(imvec_masked.size, size=10, replace=False)
        for i in sample_idx:
            ep = imvec_masked.copy()
            ep[i] += h
            em = imvec_masked.copy()
            em[i] -= h
            fd = (ib.compute_regularizer_term(ep, 'cm', full_mask, **kw)
                  - ib.compute_regularizer_term(em, 'cm', full_mask, **kw)) / (2 * h)
            np.testing.assert_allclose(grad_analytic[i], fd, rtol=1e-4, atol=1e-12)


def _reg_kwargs(rtype, norm_reg=True):
    """Return kwargs for regularizer/regularizergrad based on rtype."""
    kwargs = dict(
        beam_size=BEAM_SIZE, alpha_A=ALPHA_A, epsilon_tv=EPSILON_TV, norm_reg=norm_reg,
    )
    if rtype == "rgauss":
        kwargs["major"] = RGAUSS_MAJOR
        kwargs["minor"] = RGAUSS_MINOR
        kwargs["PA"] = RGAUSS_PA
    return kwargs


def _gradient_check(reg_setup, rtype):
    """Compute median and max fractional diff between analytic and numeric gradient."""
    im, imvec, nprior, mask, flux = reg_setup
    kwargs = _reg_kwargs(rtype)

    y0 = iu.regularizer(imvec, nprior, mask, flux, im.xdim, im.ydim, im.psize, rtype, **kwargs)
    grad_exact = iu.regularizergrad(imvec, nprior, mask, flux, im.xdim, im.ydim, im.psize, rtype, **kwargs)

    rng = np.random.default_rng(RNG_SEED)
    sample_idx = rng.choice(len(imvec), size=N_GRAD_SAMPLES, replace=False)

    grad_numeric = np.zeros(N_GRAD_SAMPLES)
    for i, j in enumerate(sample_idx):
        dx = max(GRAD_DX_REL * abs(imvec[j]), GRAD_DX_FLOOR)
        imvec2 = imvec.copy()
        imvec2[j] += dx
        y1 = iu.regularizer(imvec2, nprior, mask, flux, im.xdim, im.ydim, im.psize, rtype, **kwargs)
        grad_numeric[i] = (y1 - y0) / dx

    grad_exact_sampled = grad_exact[sample_idx]
    compare_floor = np.min(np.abs(grad_exact_sampled)) * 1e-20 + 1e-100
    frac_diff = np.abs((grad_numeric - grad_exact_sampled) / (np.abs(grad_exact_sampled) + compare_floor))
    return np.median(frac_diff), np.max(frac_diff)


# polregularizer / polregularizergrad operate on a (4, nimage) imarr in
# solver space: imarr[0]=I, imarr[1]=rho, imarr[2]=phi=2*chi, imarr[3]=psi.
# Linear-pol regs solve slots (rho, phi); circular-pol regs solve psi only.
POL_LIN_REGS = ('msimple', 'hw', 'ptv')
POL_CIRC_REGS = ('vflux', 'l1v', 'l2v', 'vtv', 'vtv2')

POL_MEDIAN_FRAC_TOL = 0.05
POL_MAX_FRAC_TOL = 0.6
# TV-style regs (ptv, vtv, vtv2) have a sqrt-of-squared-differences denominator
# that goes to ~0 on smooth regions; a few FD samples near image edges land in
# the small-denominator regime where the linear approximation breaks down.
POL_MAX_FRAC_TOL_TV = 2.0


def _pol_solve_for(rtype):
    if rtype in POL_LIN_REGS:
        return pu.POL_SOLVE_DEFAULT
    return pu.POL_SOLVE_DEFAULT_V


def _pol_tols(rtype):
    if rtype in ('ptv', 'vtv', 'vtv2'):
        return POL_MEDIAN_FRAC_TOL, POL_MAX_FRAC_TOL_TV
    return POL_MEDIAN_FRAC_TOL, POL_MAX_FRAC_TOL


@pytest.fixture(scope="module")
def polreg_setup(make_rect_image):
    """32x48 Gaussian Stokes I with jittered pol structure.

    Per-pixel jitter on rho / phi / psi is required so TV-style regularizers
    (ptv, vtv, vtv2) get a non-degenerate denominator in their gradient
    (which is sqrt of squared spatial differences and goes to 0 on uniform
    fields).
    """
    im = make_rect_image(32, 48)
    im.pulse = eh.observing.pulses.deltaPulse2D

    mask = im.imvec > 0
    nimage = int(np.sum(mask))

    rng = np.random.default_rng(7)
    I = im.imvec[mask]
    rho = np.clip(0.3 + 0.05 * rng.standard_normal(nimage), 0.05, 0.95)
    phi = 0.5 + 0.1 * rng.standard_normal(nimage)
    psi = 0.2 + 0.05 * rng.standard_normal(nimage)
    imarr = np.array([I, rho, phi, psi])
    priorarr = np.zeros_like(imarr)

    flux = im.total_flux()
    return im, imarr, priorarr, mask, flux, flux, flux


def _polreg_kwargs(norm_reg=True):
    return dict(beam_size=BEAM_SIZE, norm_reg=norm_reg)


class TestPolRegularizerValues:
    """Each REGULARIZERS_POL regularizer returns a finite, non-zero value."""

    @pytest.mark.parametrize("rtype", pu.REGULARIZERS_POL)
    @pytest.mark.parametrize("norm_reg", [True, False], ids=["normalized", "unnormalized"])
    def test_returns_finite(self, polreg_setup, rtype, norm_reg):
        im, imarr, priorarr, mask, flux, pflux, vflux = polreg_setup
        val = pu.polregularizer(
            imarr, priorarr, mask, flux, pflux, vflux,
            im.xdim, im.ydim, im.psize, rtype,
            **_polreg_kwargs(norm_reg=norm_reg),
        )
        assert np.isfinite(val), f"{rtype} (norm_reg={norm_reg}) returned {val}"

    @pytest.mark.parametrize("rtype", pu.REGULARIZERS_POL)
    def test_returns_nonzero(self, polreg_setup, rtype):
        # Catches dispatch typos: a name listed in REGULARIZERS_POL whose
        # body branch never matches will silently return 0.
        im, imarr, priorarr, mask, flux, pflux, vflux = polreg_setup
        val = pu.polregularizer(
            imarr, priorarr, mask, flux, pflux, vflux,
            im.xdim, im.ydim, im.psize, rtype,
            **_polreg_kwargs(),
        )
        assert val != 0, f"{rtype} returned 0 - dispatch likely missed"


class TestPolRegularizerGradients:
    """Analytic gradients match numeric finite differences in pol_solve slots."""

    @pytest.mark.parametrize("rtype", pu.REGULARIZERS_POL)
    def test_median_frac_diff(self, polreg_setup, rtype):
        median_tol, _ = _pol_tols(rtype)
        median_frac, _ = _pol_gradient_check(polreg_setup, rtype)
        assert median_frac < median_tol, (
            f"{rtype} median fractional gradient diff = {median_frac:.6f} (tol={median_tol})"
        )

    @pytest.mark.parametrize("rtype", pu.REGULARIZERS_POL)
    def test_max_frac_diff(self, polreg_setup, rtype):
        _, max_tol = _pol_tols(rtype)
        _, max_frac = _pol_gradient_check(polreg_setup, rtype)
        assert max_frac < max_tol, (
            f"{rtype} max fractional gradient diff = {max_frac:.6f} (tol={max_tol})"
        )


def _pol_gradient_check(polreg_setup, rtype):
    """FD-vs-analytic check across N pixels in each pol_solve-active slot."""
    im, imarr, priorarr, mask, flux, pflux, vflux = polreg_setup
    kwargs = _polreg_kwargs()
    pol_solve = _pol_solve_for(rtype)

    y0 = pu.polregularizer(
        imarr, priorarr, mask, flux, pflux, vflux,
        im.xdim, im.ydim, im.psize, rtype, **kwargs,
    )
    grad_exact = pu.polregularizergrad(
        imarr, priorarr, mask, flux, pflux, vflux,
        im.xdim, im.ydim, im.psize, rtype,
        pol_solve=pol_solve, **kwargs,
    )

    rng = np.random.default_rng(RNG_SEED)
    nimage = imarr.shape[1]
    sample_idx = rng.choice(nimage, size=N_GRAD_SAMPLES, replace=False)

    frac_diffs = []
    for slot in range(4):
        if pol_solve[slot] == 0:
            continue
        for j in sample_idx:
            dx = max(GRAD_DX_REL * abs(imarr[slot, j]), GRAD_DX_FLOOR)
            imarr2 = imarr.copy()
            imarr2[slot, j] += dx
            y1 = pu.polregularizer(
                imarr2, priorarr, mask, flux, pflux, vflux,
                im.xdim, im.ydim, im.psize, rtype, **kwargs,
            )
            numeric = (y1 - y0) / dx
            exact = grad_exact[slot, j]
            compare_floor = max(abs(exact), 1e-100) * 1e-20 + 1e-100
            frac_diffs.append(abs((numeric - exact) / (abs(exact) + compare_floor)))

    frac_diffs = np.array(frac_diffs)
    return float(np.median(frac_diffs)), float(np.max(frac_diffs))


# regularizer_mf dispatches by string prefix: names starting with 'l2_'
# compute an L2 distance from the prior; names starting with 'tv_' compute
# spatial total variation. The 12 names in REGULARIZERS_SPECTRAL only differ
# in which spectral coefficient slot (alpha / beta / rm / cm / _p variants)
# the imager passes through; the computation itself is shared between them.


@pytest.fixture(scope="module")
def mfreg_setup(make_rect_image):
    """32x48 spectral coefficient vector with a half-amplitude prior."""
    im = make_rect_image(32, 48)
    imvec = im.imvec
    nprior = imvec * 0.5
    mask = imvec > 0
    return im, imvec, nprior, mask


class TestMFRegularizerValues:
    """REGULARIZERS_SPECTRAL regularizers return finite, non-zero values."""

    @pytest.mark.parametrize("rtype", ib.REGULARIZERS_SPECTRAL)
    @pytest.mark.parametrize("norm_reg", [True, False], ids=["normalized", "unnormalized"])
    def test_returns_finite(self, mfreg_setup, rtype, norm_reg):
        im, imvec, nprior, mask = mfreg_setup
        val = mfu.regularizer_mf(
            imvec, nprior, mask, im.xdim, im.ydim, im.psize, rtype,
            beam_size=BEAM_SIZE, norm_reg=norm_reg,
        )
        assert np.isfinite(val), f"{rtype} (norm_reg={norm_reg}) returned {val}"

    @pytest.mark.parametrize("rtype", ib.REGULARIZERS_SPECTRAL)
    def test_returns_nonzero(self, mfreg_setup, rtype):
        # A name that matches neither 'l2_' nor 'tv_' prefix silently returns 0.
        # Guards against a future addition to REGULARIZERS_SPECTRAL that breaks
        # the naming convention.
        im, imvec, nprior, mask = mfreg_setup
        val = mfu.regularizer_mf(
            imvec, nprior, mask, im.xdim, im.ydim, im.psize, rtype,
            beam_size=BEAM_SIZE,
        )
        assert val != 0, f"{rtype} returned 0 - dispatch likely missed"


class TestMFRegularizerGradients:
    """Analytic regularizergrad_mf matches numeric finite differences."""

    @pytest.mark.parametrize("rtype", ib.REGULARIZERS_SPECTRAL)
    def test_median_frac_diff(self, mfreg_setup, rtype):
        median_frac, _ = _mfreg_gradient_check(mfreg_setup, rtype)
        assert median_frac < MEDIAN_FRAC_TOL, (
            f"{rtype} median fractional gradient diff = {median_frac:.6f}"
        )

    @pytest.mark.parametrize("rtype", ib.REGULARIZERS_SPECTRAL)
    def test_max_frac_diff(self, mfreg_setup, rtype):
        _, max_frac = _mfreg_gradient_check(mfreg_setup, rtype)
        assert max_frac < MAX_FRAC_TOL, (
            f"{rtype} max fractional gradient diff = {max_frac:.6f}"
        )


def _mfreg_gradient_check(mfreg_setup, rtype):
    """Compare analytic and finite-difference gradients on N random pixels."""
    im, imvec, nprior, mask = mfreg_setup
    kwargs = dict(beam_size=BEAM_SIZE, norm_reg=True)

    y0 = mfu.regularizer_mf(
        imvec, nprior, mask, im.xdim, im.ydim, im.psize, rtype, **kwargs,
    )
    grad_exact = mfu.regularizergrad_mf(
        imvec, nprior, mask, im.xdim, im.ydim, im.psize, rtype, **kwargs,
    )

    rng = np.random.default_rng(RNG_SEED)
    sample_idx = rng.choice(len(imvec), size=N_GRAD_SAMPLES, replace=False)

    grad_numeric = np.zeros(N_GRAD_SAMPLES)
    for i, j in enumerate(sample_idx):
        dx = max(GRAD_DX_REL * abs(imvec[j]), GRAD_DX_FLOOR)
        imvec2 = imvec.copy()
        imvec2[j] += dx
        y1 = mfu.regularizer_mf(
            imvec2, nprior, mask, im.xdim, im.ydim, im.psize, rtype, **kwargs,
        )
        grad_numeric[i] = (y1 - y0) / dx

    grad_exact_sampled = grad_exact[sample_idx]
    compare_floor = np.min(np.abs(grad_exact_sampled)) * 1e-20 + 1e-100
    frac_diff = np.abs((grad_numeric - grad_exact_sampled) / (np.abs(grad_exact_sampled) + compare_floor))
    return float(np.median(frac_diff)), float(np.max(frac_diff))
