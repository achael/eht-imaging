"""Tests for chi-squared cross-transform consistency (direct, fast, nfft).

Verifies that chi-squared values and gradients AGREE across DFT, FFT, and NFFT for all
data types -- a cross-check, not a standalone correctness proof: the direct path is the
reference, finite-difference-validated in test_gradients.py (the canonical FD suite), so a
tight direct-vs-nfft gradient bound pins nfft gradient correctness. All tests use a 32x48
image so xdim != ydim exercises the rectangular-image code paths (rect subsumes square).
"""

import tracemalloc

import numpy as np
import pytest

import ehtim as eh
from ehtim.imaging.imager_backend import (
    ImagerConfig,
    MfConfig,
    compute_chisq_term,
    compute_chisqdata_term,
)
from ehtim.imaging.imager_utils import chisq, chisqdata, chisqgrad, chisqgrad_vis
from ehtim.imaging.pol_imager_utils import chisqgrad_m, chisqgrad_p, chisqgrad_vvis

# Observation parameters (must match conftest.py)
TINT_SEC = 5
TADV_SEC = 600
TSTART_HR = 0
TSTOP_HR = 24
BW_HZ = 4e9

# Data types to test. The `_diag` variants are the per-timestamp diagonalized
# closures (statistically independent components after orthogonalization).
DATATERMS = ["vis", "bs", "amp", "cphase", "cphase_diag",
             "camp", "logcamp", "logcamp_diag"]

# Transform type pairs to compare
TTYPE_PAIRS = [("direct", "fast"), ("direct", "nfft"), ("nfft", "fast")]

# Cross-transform gradient agreement. The direct DFT path is the trusted reference -- its
# gradients are finite-difference-validated in test_gradients.py -- so a tight direct-vs-nfft
# bound effectively pins nfft gradient correctness. Pairs that include the gridded 'fast' FFT
# are limited by its interpolation accuracy, not by a gradient bug, so they stay looser.
GRAD_MAX_TOL_DIRECT_NFFT = 1e-2
GRAD_MAX_TOL = 0.25            # any pair containing 'fast'

# Diagonalized closures orthogonalize per-timestamp covariance, which amplifies tail outliers.
GRAD_MAX_TOL_DIAG = 0.5
DIAG_DTYPES = {"cphase_diag", "logcamp_diag"}

# Tolerances (calibrated on 32x48 synthetic Gaussian)
CHISQ_FRAC_TOL = 0.01
GRAD_MEDIAN_TOL = 0.05

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

# Random seed for image perturbation in fixtures
RNG_SEED = 4


@pytest.fixture(scope="module")
def chisq_setup(eht_array, make_asym_image):
    """Set up observation and test image for chi-squared comparison tests.

    Uses a 32x48 asymmetric (offset double-Gaussian) image: xdim != ydim
    exercises the rectangular-image code paths, and the broken symmetry +
    edge flux surface axis-ordering bugs a centered Gaussian hides. A single
    DFT observation serves as ground truth; chi-squared is then evaluated with
    different transform matrices (direct, fast, nfft) to test that the
    transform approximations agree.
    """
    im = make_asym_image(32, 48)
    im.imvec = im.imvec * 2.0 / im.total_flux()  # normalize to 2 Jy

    obs = im.observe(
        eht_array, TINT_SEC, TADV_SEC, TSTART_HR, TSTOP_HR, BW_HZ,
        sgrscat=False, ampcal=True, phasecal=False,
        ttype="direct", add_th_noise=False,
    )

    prior = im.copy()

    im2 = prior.copy()
    rng = np.random.default_rng(RNG_SEED)
    im2.imvec *= 1.0 + (rng.random(len(im2.imvec)) - 0.5) / 10.0
    im2.imvec += rng.random(len(im2.imvec)) / 10.0 * im.imvec

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
        if pair == ("direct", "nfft"):
            tol = GRAD_MAX_TOL_DIAG if dtype in DIAG_DTYPES else GRAD_MAX_TOL_DIRECT_NFFT
        else:
            tol = GRAD_MAX_TOL_DIAG if dtype in DIAG_DTYPES else GRAD_MAX_TOL
        assert max_frac < tol, (
            f"{dtype} {pair[0]}-{pair[1]}: grad max frac diff = {max_frac:.6f}"
        )


def _gradient_comparison(chisq_setup, dtype, pair):
    """Compute median and max fractional gradient diff between two ttypes."""
    ttype_a, ttype_b = pair
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
# Polarimetric chi-squared (pvis / m / vvis). Pol has no 'fast' ttype.
# ---------------------------------------------------------------------------
POL_DATATERMS = ["pvis", "m", "vvis"]
POL_CHISQ_FRAC_TOL = 0.01   # direct-vs-nfft value agreement


def _pol_config(ttype):
    return ImagerConfig(
        pol="IP", transforms=[], ttype=ttype, mf=False,
        mf_config=MfConfig(mf_order=0, mf_order_pol=0, mf_rm=0, mf_cm=0),
    )


def _pol_data_tuple(obs, prior, mask, dtype, ttype):
    data, sigma, A = compute_chisqdata_term(obs, prior, mask, dtype, _pol_config(ttype))
    return A, data, sigma


@pytest.fixture(scope="module")
def chisq_setup_pol(eht_array, make_asym_image):
    """Asymmetric 32x48 polarized image + a jittered physical imcur [I,rho,phi,psi].

    add_random_pol gives a spatially-varying EVPA (random screen) and, with
    ccorr>0, a spatially-varying circular fraction -- so chi, vfrac, rho, and psi
    all vary across the image. A constant pol fraction would leave the
    polarization structure spatially uniform and never exercise those gradients.
    """
    im = make_asym_image(32, 48)
    im.imvec = im.imvec * 2.0 / im.total_flux()
    im = im.add_random_pol(0.25, 40 * eh.RADPERUAS,
                           cmag=0.06, ccorr=40 * eh.RADPERUAS, seed=7)
    prior = im.copy()

    obs = im.observe(
        eht_array, TINT_SEC, TADV_SEC, TSTART_HR, TSTOP_HR, BW_HZ,
        ampcal=True, phasecal=True, ttype="direct", add_th_noise=False,
    )
    mask = np.ones(im.imvec.size, dtype=bool)

    # Physical-space imcur jittered off the truth so residuals (and grads) are
    # nonzero; keep rho in (0,1) and psi away from 0 (tan(psi) in vvis grad).
    rng = np.random.default_rng(RNG_SEED)
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


class TestPolChisqConsistency:
    """Pol chi-squared values agree between direct and nfft (same obs, different A)."""

    @pytest.mark.parametrize("dtype", POL_DATATERMS)
    def test_chisq_values(self, chisq_setup_pol, dtype):
        obs, prior = chisq_setup_pol["obs"], chisq_setup_pol["prior"]
        mask, imcur = chisq_setup_pol["mask"], chisq_setup_pol["imcur"]
        vals = {}
        for tt in ("direct", "nfft"):
            A, data, sigma = _pol_data_tuple(obs, prior, mask, dtype, tt)
            vals[tt] = compute_chisq_term(imcur, dtype, A, data, sigma, ttype=tt, mask=mask)
        frac = abs((vals["direct"] - vals["nfft"]) / abs(vals["direct"]))
        assert frac < POL_CHISQ_FRAC_TOL, f"{dtype}: chisq frac diff = {frac:.6f}"


# ---------------------------------------------------------------------------
# Operator memory: the direct-transform gradients must not copy the operator
#
# `Amatrix.conj()` materializes a full (nvis, npix) copy of the Fourier operator;
# only the subsequent `.T` is free. Every gradient below needs nothing more than
# `Amatrix.conj().T @ vec`, which is the same arithmetic as conjugating the
# (nvis,) input and the (npix,) output, both orders of magnitude smaller. These
# tests bound the peak allocation of each gradient well under the operator.
# ---------------------------------------------------------------------------

OP_NVIS = 1200
OP_NPIX = 2048

# Measured on saturn before the fix: peak is 1.00x the operator for every
# gradient here. After it, peak is a handful of (nvis,) and (npix,) vectors,
# under 1% of the operator. A quarter of the operator sits clearly between the
# two and leaves room for BLAS scratch.
OP_PEAK_FRACTION = 0.25


def _peak_mib(fn):
    """Peak traced allocation in MiB during fn(). NumPy registers its own
    tracemalloc domain, so array allocations are counted."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 2**20


@pytest.fixture(scope="module")
def operator_setup():
    """A dense complex operator plus matching Stokes-I and polarimetric inputs.

    Synthetic rather than observation-derived: these tests measure allocation
    against a known operator size, so the operator has to be sized here.
    """
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((OP_NVIS, OP_NPIX))
         + 1j * rng.standard_normal((OP_NVIS, OP_NPIX)))
    data = rng.standard_normal(OP_NVIS) + 1j * rng.standard_normal(OP_NVIS)
    sigma = np.abs(rng.standard_normal(OP_NVIS)) + 0.1
    imvec = np.abs(rng.standard_normal(OP_NPIX)) + 0.1
    imarr = np.vstack([imvec,
                       0.3 * np.ones(OP_NPIX),      # rho
                       0.4 * np.ones(OP_NPIX),      # phi
                       0.2 * np.ones(OP_NPIX)])     # psi
    return {"A": A, "data": data, "sigma": sigma, "imvec": imvec, "imarr": imarr,
            "op_mib": A.nbytes / 2**20}


# (label, callable taking the setup dict) for every gradient that applies the
# adjoint operator on the direct path.
GRAD_CASES = [
    ("chisqgrad_vis",
     lambda s: chisqgrad_vis(s["imvec"], s["A"], s["data"], s["sigma"])),
    ("chisqgrad_p",
     lambda s: chisqgrad_p(s["imarr"], s["A"], s["data"], s["sigma"],
                           pol_solve=(1, 1, 1, 1))),
    ("chisqgrad_m",
     lambda s: chisqgrad_m(s["imarr"], s["A"], s["data"], s["sigma"],
                          pol_solve=(1, 1, 1, 1))),
    ("chisqgrad_vvis",
     lambda s: chisqgrad_vvis(s["imarr"], s["A"], s["data"], s["sigma"],
                              pol_solve=(1, 1, 0, 1))),
]


class TestGradientOperatorNotCopied:
    """Peak allocation during a direct-path gradient stays well under the operator."""

    def test_tracemalloc_sees_numpy_allocations(self, operator_setup):
        # Guard. If tracemalloc did not count numpy arrays every bound below
        # would pass while the copies were still happening. Conjugating the
        # operator outright must register as roughly its own size.
        A, op_mib = operator_setup["A"], operator_setup["op_mib"]
        peak = _peak_mib(lambda: A.conj())
        assert peak > 0.9 * op_mib, (
            f"tracemalloc reported {peak:.2f} MiB for an explicit copy of a "
            f"{op_mib:.2f} MiB operator; the bounds below would be vacuous")

    @pytest.mark.parametrize("label,call", GRAD_CASES, ids=[c[0] for c in GRAD_CASES])
    def test_peak_allocation_well_under_operator(self, operator_setup, label, call):
        op_mib = operator_setup["op_mib"]
        peak = _peak_mib(lambda: call(operator_setup))
        assert peak < OP_PEAK_FRACTION * op_mib, (
            f"{label} peaked at {peak:.2f} MiB against a {op_mib:.2f} MiB "
            f"operator ({peak/op_mib:.2f}x); it is copying the operator")
