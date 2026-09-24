"""Tests for ehtim.observing.obs_helpers."""
import os
import subprocess
import sys
import zlib

import numpy as np

import ehtim.const_def as ehc
from ehtim.observing import obs_helpers as obsh

# ---------------------------------------------------------------------------
# Hash-seeded random helpers: reproducibility across processes
#
# The hash-seeded random helpers must be reproducible across Python processes.
# CPython salts str hashing per process (PYTHONHASHSEED), so seeding numpy from
# the builtin hash() produced different simulated noise every run; _stable_seed
# (zlib.crc32) makes the draws identical across processes.
# ---------------------------------------------------------------------------


def test_stable_seed_matches_crc32():
    # Pins the implementation to crc32 (process-independent), guarding against a
    # regression back to the per-process-salted builtin hash().
    assert obsh._stable_seed("ALMA", "APEX", 12.5, "rr") == \
        zlib.crc32(b"'ALMA','APEX',12.5,'rr'")


def test_stable_seed_in_numpy_seed_range():
    # np.random.seed requires 0 <= seed < 2**32.
    seed = obsh._stable_seed("x", 1, 2.0, "im")
    assert 0 <= seed < 2**32


# --- cross-process reproducibility ---------

# Draw one thermal-noise value (cerror_hash) and one gain value (hashrandn) and
# print them; the values must not depend on PYTHONHASHSEED.
_SNIPPET = (
    "from ehtim.observing import obs_helpers as obsh;"
    "print('RESULT',"
    " repr(obsh.cerror_hash(2.0, 'ALMA', 'APEX', 12.5, 'rr', 42)),"
    " repr(obsh.hashrandn('SMT', 3.0, 'gain')))"
)


def _run_with_hashseed(hashseed):
    env = {**os.environ, "PYTHONHASHSEED": hashseed}
    out = subprocess.check_output([sys.executable, "-c", _SNIPPET], env=env, text=True)
    lines = [ln for ln in out.splitlines() if ln.startswith("RESULT")]
    assert lines, f"subprocess produced no RESULT line; output was:\n{out}"
    return lines[-1]


def test_hash_helpers_reproducible_across_processes():
    # With the old builtin hash() these would differ per PYTHONHASHSEED but with
    # crc32 the drawn values are byte-identical across salts.
    results = {_run_with_hashseed(hs) for hs in ("0", "1", "999999")}
    assert len(results) == 1, f"noise varied across PYTHONHASHSEED: {results}"


# ---------------------------------------------------------------------------
# NFFT plan configuration: accuracy and thread count
# ---------------------------------------------------------------------------


def _uv_points(n=64, seed=0):
    """n nonuniform (u, v) points in the range NFFTInfo expects, in lambda."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-2e9, 2e9, size=(n, 2))


def test_nfft_eps_default_is_1e_6():
    """The shipped NFFT accuracy is 1e-6.

    Pinned because it is a speed/accuracy trade, not an arbitrary constant:
    1e-9 cost ~20% more per imaging run for accuracy nothing in the
    chi-squared can see. Change it deliberately, with a benchmark.
    """
    assert ehc.NFFT_EPS_DEFAULT == 1e-6


def test_nfft_nthreads_default_lets_finufft_choose():
    """0 means finufft picks its own thread count."""
    assert ehc.NFFT_NTHREADS_DEFAULT == 0


def test_nthreads_is_a_recognized_finufft_option(recwarn):
    """Building a plan with nthreads must not warn about an unknown option.

    finufft only *warns* on an option name it does not recognize, so a typo or
    an upstream rename would silently leave the thread count unset. This fails
    if 'nthreads' ever stops being a real finufft_opts attribute.
    """
    obsh.FINUFFTPlan(16, 16, _uv_points(8) * 1e-9, eps=1e-6, nthreads=2)
    unknown = [w for w in recwarn if 'does not have attribute' in str(w.message)]
    assert not unknown, f"finufft rejected an option name: {[str(w.message) for w in unknown]}"


def test_nfftinfo_records_nthreads():
    """NFFTInfo keeps the thread count it was given, so callers can read it back."""
    info = obsh.NFFTInfo(16, 16, 1e-10, ehc.PULSE_DEFAULT, 2, 2, _uv_points(8), nthreads=3)
    assert info.nthreads == 3


def test_nfftinfo_defaults_to_library_settings():
    """With no overrides, NFFTInfo takes both knobs from const_def."""
    info = obsh.NFFTInfo(16, 16, 1e-10, ehc.PULSE_DEFAULT, 2, 2, _uv_points(8))
    assert info.eps == ehc.NFFT_EPS_DEFAULT
    assert info.nthreads == ehc.NFFT_NTHREADS_DEFAULT


def test_default_eps_transform_matches_exact_dft():
    """The default accuracy is far tighter than any realistic data noise.

    Guards the 1e-9 -> 1e-6 change. The bound discriminates: relative error
    measured 3.0e-6 at the 1e-6 default and 1.8e-4 at 1e-4, so a further
    relaxation fails here while the current default clears it ~3x over.
    """
    xdim = ydim = 16
    psize = 1e-10
    uv = _uv_points(32)
    info = obsh.NFFTInfo(xdim, ydim, psize, ehc.PULSE_DEFAULT, 2, 2, uv)

    rng = np.random.default_rng(1)
    imvec = rng.uniform(0, 1, size=xdim * ydim)

    # nufft2_backend returns the transform without the pulse/centering factor;
    # ftmatrix already carries it, so pulsefac is applied to the NFFT side.
    got = obsh.nufft2_backend(imvec, info) * info.pulsefac
    want = obsh.ftmatrix(psize, xdim, ydim, uv, pulse=ehc.PULSE_DEFAULT) @ imvec
    assert np.allclose(got, want, rtol=0, atol=1e-5 * np.abs(want).max())
