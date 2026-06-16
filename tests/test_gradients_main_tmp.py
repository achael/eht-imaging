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


# ===========================================================================
# (2) Boundary FD: first-row/col back-neighbor masking in the TV gradients
#     Pins the #306 stv_pol_grad boundary-mask fix (tv/vtv already had it).
#     Full 4x5 grid, central FD on EVERY pixel, so the first row/col -- where
#     the back-neighbor is the zero pad and the masking lives -- are checked.
# ===========================================================================
def _boundary_pol_imarr(nx, ny):
    rng = np.random.default_rng(11)
    npix = nx * ny
    I = 1.0 + 0.3 * rng.random(npix)
    rho = np.clip(0.3 + 0.10 * rng.standard_normal(npix), 0.05, 0.95)
    phi = 0.5 + 0.30 * rng.standard_normal(npix)
    psi = 0.2 + 0.10 * rng.standard_normal(npix)
    return np.array([I, rho, phi, psi])


@pytest.mark.parametrize("rtype", ["ptv", "vtv"])
def test_pol_tv_grad_matches_fd_on_boundary(rtype):
    nx, ny = 4, 5
    npix = nx * ny
    imarr = _boundary_pol_imarr(nx, ny)
    mask = np.ones(npix, dtype=bool)
    flux = float(np.sum(imarr[0]))
    kwargs = dict(norm_reg=False, beam_size=1.0)
    args = (np.zeros_like(imarr), mask, flux, flux, flux, nx, ny, 1.0, rtype)
    grad = pu.polregularizergrad(imarr, *args, pol_solve=np.array([1, 1, 1, 1]), **kwargs)

    eps = 1e-6
    frac = []
    for slot in (0, 1, 3):  # chi (slot 2) is absent from the |P|/V back-neighbor terms
        for j in range(npix):
            a, b = imarr.copy(), imarr.copy()
            a[slot, j] += eps
            b[slot, j] -= eps
            fd = (pu.polregularizer(a, *args, **kwargs)
                  - pu.polregularizer(b, *args, **kwargs)) / (2 * eps)
            frac.append(abs(grad[slot, j] - fd) / max(abs(fd), abs(grad[slot, j]), 1e-6))
    assert max(frac) < 1e-3, f"{rtype}: max boundary frac diff = {max(frac):.4g}"


def test_intensity_tv_grad_matches_fd_on_boundary():
    nx, ny = 4, 5
    npix = nx * ny
    rng = np.random.default_rng(11)
    imvec = 1.0 + 0.5 * rng.random(npix)
    mask = np.ones(npix, dtype=bool)
    flux = float(np.sum(imvec))
    kwargs = dict(norm_reg=False, beam_size=1.0, epsilon_tv=0.0)
    args = (np.zeros_like(imvec), mask, flux, nx, ny, 1.0, "tv")
    grad = iu.regularizergrad(imvec, *args, **kwargs)

    eps = 1e-6
    frac = []
    for j in range(npix):
        a, b = imvec.copy(), imvec.copy()
        a[j] += eps
        b[j] -= eps
        fd = (iu.regularizer(a, *args, **kwargs) - iu.regularizer(b, *args, **kwargs)) / (2 * eps)
        frac.append(abs(grad[j] - fd) / max(abs(fd), abs(grad[j]), 1e-6))
    assert max(frac) < 1e-3, f"tv: max boundary frac diff = {max(frac):.4g}"


# ===========================================================================
# (3) Transforms modify the gradient correctly -- chain rule vs FD, isolated
#     transform_gradients(grad_phys, ...) must equal d/dimarr [f(transform(imarr))]
#     for the mcv / vcv / polcv image transforms. Covers the off-diagonal
#     cross-terms the #306 fix added and the log-I-slot preservation.
# ===========================================================================
def _scalar_obj(phys):
    """Arbitrary smooth scalar f(phys) = sum(weights * phys**2)."""
    weights = np.array([0.7, 1.3, -0.5, 0.9])[:, None]
    return float(np.sum(weights * phys**2))


def _fd_grad_of_transform(imarr, transforms, which_solve):
    """Centered FD gradient of (scalar_obj o transform_imarr) in solver space."""
    eps = 1e-6
    grad = np.zeros_like(imarr)
    for k in range(imarr.shape[0]):
        for i in range(imarr.shape[1]):
            ip, im_ = imarr.copy(), imarr.copy()
            ip[k, i] += eps
            im_[k, i] -= eps
            grad[k, i] = (_scalar_obj(transform_imarr(ip, transforms, which_solve))
                          - _scalar_obj(transform_imarr(im_, transforms, which_solve))) / (2 * eps)
    return grad


def _solver_imarr(seed=0, n=8, v_pre=None, m_pre=None):
    """Build a (4, n) solver-space image array; pin a row to set up the
    diagonal-Jacobian operating points (mcv vfrac=0, vcv mfrac=0)."""
    rng = np.random.default_rng(seed)
    log_I = rng.uniform(-1.0, 1.0, size=n)
    m_arr = rng.uniform(-2.0, 2.0, size=n) if m_pre is None else np.full(n, m_pre)
    chi_arr = rng.uniform(-0.5, 0.5, size=n)
    v_arr = rng.uniform(-0.3, 0.3, size=n) if v_pre is None else np.full(n, v_pre)
    return np.array([log_I, m_arr, chi_arr, v_arr])


class TestTransformGradientsChainRule:
    """transform_gradients(grad_phys, ...) == FD of (f o transform_imarr)."""

    @pytest.mark.parametrize(
        "transforms,which_solve,imarr_kwargs,check_rows",
        [
            # real-usage diagonal operating points
            (["log", "mcv"], np.array([1, 1, 1, 0]), {"v_pre": 0.0}, [0, 1, 2]),
            (["log", "vcv"], np.array([1, 0, 0, 1]), {"m_pre": 0.0}, [0, 3]),
            (["log", "polcv"], np.array([1, 1, 0, 1]), {}, [0, 1, 3]),
            # general regimes -> nonzero off-diagonal cross-terms (the #306 fix)
            (["log", "mcv"], np.array([1, 1, 0, 0]), {}, [0, 1]),
            (["log", "vcv"], np.array([1, 0, 0, 1]), {}, [0, 3]),
            # sanity: log only, and pure pol-cv with no log
            (["log"], np.array([1, 0, 0, 0]), {}, [0]),
            (["mcv"], np.array([0, 1, 0, 0]), {"v_pre": 0.0}, [1]),
            (["vcv"], np.array([0, 0, 0, 1]), {"m_pre": 0.0}, [3]),
            (["polcv"], np.array([0, 1, 0, 1]), {}, [1, 3]),
        ],
        ids=["IP_vfrac0", "IV_mfrac0", "IPV", "IP_general", "IV_general",
             "log_only", "mcv_only", "vcv_only", "polcv_only"],
    )
    def test_chain_rule_matches_fd(self, transforms, which_solve, imarr_kwargs, check_rows):
        imarr = _solver_imarr(seed=0, **imarr_kwargs)
        phys = transform_imarr(imarr, transforms, which_solve)
        weights = np.array([0.7, 1.3, -0.5, 0.9])[:, None]
        grad_phys = 2.0 * weights * phys  # = d(_scalar_obj)/dphys

        grad_solver = transform_gradients(grad_phys, imarr, transforms, which_solve)
        grad_fd = _fd_grad_of_transform(imarr, transforms, which_solve)
        for k in check_rows:
            np.testing.assert_allclose(
                grad_solver[k], grad_fd[k], rtol=1e-5, atol=1e-8,
                err_msg=f"chain rule mismatch row {k} for transforms={transforms}")

    def test_log_chain_preserved_under_mcv(self):
        """log+mcv: outarr[0] = exp(imarr[0]); mcv must not clobber the I slot."""
        imarr = _solver_imarr(seed=1, v_pre=0.0)
        out = transform_gradients(np.ones_like(imarr), imarr, ["log", "mcv"], np.array([1, 1, 0, 0]))
        np.testing.assert_allclose(out[0], np.exp(imarr[0]), rtol=1e-12)

    def test_log_chain_preserved_under_polcv(self):
        """log+polcv: outarr[0] = exp(imarr[0]); polcv must not clobber the I slot."""
        imarr = _solver_imarr(seed=2)
        out = transform_gradients(np.ones_like(imarr), imarr, ["log", "polcv"], np.array([1, 1, 0, 1]))
        np.testing.assert_allclose(out[0], np.exp(imarr[0]), rtol=1e-12)


# ===========================================================================
# (3) Transforms modify the gradient correctly -- end-to-end through Imager
#     Builds a real Imager and compares objgrad to FD of objfunc, sampling the
#     POLARIZATION DOF block (past the leading Stokes-I block) where the mcv/vcv
#     cross-coupling lives -- the Stokes-I-dominated global sampling misses it.
#     This is the test that exposes the mcv/vcv cross-coupling bug end-to-end.
# ===========================================================================
# (pol, transform, data_term, reg_term)
POL_OBJFD_CASES = [
    ("IP", ["log", "mcv"], {"vis": 100, "pvis": 100, "m": 100}, {"simple": 1, "hw": 1}),
    ("IV", ["log", "vcv"], {"amp": 100, "vvis": 100}, {"simple": 1, "l2v": 1}),
    ("IQUV", ["log", "polcv"], {"vis": 100, "pvis": 100, "m": 100, "vvis": 100},
     {"simple": 1, "hw": 1, "l2v": 1}),
]


@pytest.fixture(scope="module")
def asym_pol_setup(array):
    """Asymmetric image, spatially-varying lin+circ pol, Stokes-I prior.
    Square grid (add_random_pol scattering rect bug on main)."""
    im = make_asym_image(32, 32)
    im.imvec = im.imvec * 2.0 / im.total_flux()
    prior = im.blur_circ(30 * UA)
    im_pol = im.add_random_pol(0.25, 40 * UA, cmag=0.06, ccorr=40 * UA, seed=7)
    obs = im_pol.observe(array, TINT, TADV, TSTART, TSTOP, BW,
                         ampcal=True, phasecal=True, ttype="direct", add_th_noise=False)
    return im_pol, obs, prior


class TestObjectiveGradPolarimetricFD:
    """imgr.objgrad matches central FD of imgr.objfunc on the polarization DOF
    block, for IP/mcv, IV/vcv, IQUV/polcv x {direct, nfft}."""

    @pytest.mark.parametrize("ttype", TTYPES)
    @pytest.mark.parametrize("pol,transform,data_term,reg_term", POL_OBJFD_CASES,
                             ids=[c[0] for c in POL_OBJFD_CASES])
    def test_pol_dof_grad_matches_fd(self, asym_pol_setup, pol, transform,
                                     data_term, reg_term, ttype):
        im_pol, obs, prior = asym_pol_setup
        imgr = eh.imager.Imager(
            obs, prior, prior_im=prior, flux=im_pol.total_flux(),
            data_term=data_term, reg_term=reg_term,
            ttype=ttype, pol=pol, transform=transform, maxit=10,
        )
        imgr.init_imager()
        nslots = int(np.sum(imgr._which_solve))
        x = np.asarray(imgr._xinit, float)
        rng = np.random.default_rng(4)
        x = x + 0.10 * rng.standard_normal(x.size)
        nimage = x.size // nslots

        g = np.asarray(imgr.objgrad(x))
        # pol DOF block = everything past the leading Stokes-I block (slot-major
        # packing); take the best-conditioned 30 for a stable fractional check.
        pol_idx = np.arange(nimage, x.size)
        idx = pol_idx[np.argsort(np.abs(g[pol_idx]))[-30:]]
        fd = np.empty(idx.size)
        for k, j in enumerate(idx):
            a, b = x.copy(), x.copy()
            a[j] += 1e-6
            b[j] -= 1e-6
            fd[k] = (imgr.objfunc(a) - imgr.objfunc(b)) / 2e-6
        frac = np.abs((fd - g[idx]) / (np.abs(g[idx]) + 1e-100))
        assert np.median(frac) < 1e-4, f"{pol}/{ttype}: median frac {np.median(frac):.3e}"
        assert np.max(frac) < 1e-2, f"{pol}/{ttype}: max frac {np.max(frac):.3e}"
