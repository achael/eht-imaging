"""Tests for ehtim.observing.obs_helpers."""
import os
import subprocess
import sys
import tracemalloc
import zlib

import numpy as np
import pytest

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
# adjoint_dot: apply A^H without materializing conj(A)
#
# `Amatrix.conj()` copies the whole (nvis, npix) operator; only the `.T` after
# it is free. adjoint_dot conjugates the (nvis,) input and the (npix,) output
# instead, which is the same arithmetic on far smaller vectors.
# ---------------------------------------------------------------------------

ADJ_SHAPES = [(500, 1024), (1200, 2048), (37, 64)]

# Pre-fix the equivalent expression peaks at 1.00x the operator; adjoint_dot
# allocates two vectors. A quarter of the operator separates the two cleanly.
ADJ_PEAK_FRACTION = 0.25


def _adj_operator(nvis, npix, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((nvis, npix))
         + 1j * rng.standard_normal((nvis, npix)))
    vec = rng.standard_normal(nvis) + 1j * rng.standard_normal(nvis)
    return A, vec


def _peak_mib(fn):
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 2**20


@pytest.mark.parametrize("nvis,npix", ADJ_SHAPES, ids=lambda v: str(v))
def test_adjoint_dot_matches_explicit_conjugate_transpose(nvis, npix):
    # Bit-identity, not just closeness: this is the whole claim of the helper,
    # and it kills the plausible wrong forms (dropping either conjugation, or
    # conjugating the operator instead of the vectors).
    A, vec = _adj_operator(nvis, npix)
    expected = np.dot(A.conj().T, vec)
    assert np.array_equal(obsh.adjoint_dot(A, vec), expected)


def test_adjoint_dot_tracemalloc_sees_numpy_allocations():
    # Guard: if numpy allocations were invisible to tracemalloc the memory
    # bound below would pass no matter what adjoint_dot did.
    A, _ = _adj_operator(1200, 2048)
    op_mib = A.nbytes / 2**20
    assert _peak_mib(lambda: A.conj()) > 0.9 * op_mib


def test_adjoint_dot_does_not_copy_the_operator():
    A, vec = _adj_operator(1200, 2048)
    op_mib = A.nbytes / 2**20
    peak = _peak_mib(lambda: obsh.adjoint_dot(A, vec))
    assert peak < ADJ_PEAK_FRACTION * op_mib, (
        f"adjoint_dot peaked at {peak:.2f} MiB against a {op_mib:.2f} MiB "
        f"operator ({peak/op_mib:.2f}x)")


# ---------------------------------------------------------------------------
# ftmatrix build cost
#
# The operator was assembled as a Python list of nvis separate (ydim, xdim)
# arrays and then copied into one contiguous block by np.array() while the list
# was still alive, so building it peaked at twice the array it returns. With a
# mask it was worse: the full unmasked stack was built and then sliced down.
# ---------------------------------------------------------------------------

# Post-fix the peak is the returned array plus a single (npix,) row; pre-fix it
# is 2.00x measured. 1.35x separates them with room for the row and scratch.
FT_PEAK_FRACTION = 1.35


def _ftmatrix_reference(pdim, xdim, ydim, uvlist, pulse=ehc.PULSE_DEFAULT, mask=[]):
    """The list-then-copy implementation, kept here as the value reference.

    Spelled out rather than imported so the test pins the actual arithmetic
    (sign convention, outer-product axis order, row ordering) and not just
    whatever ftmatrix currently happens to do.
    """
    xlist = np.arange(0, -xdim, -1)*pdim + (pdim*xdim)/2.0 - pdim/2.0
    ylist = np.arange(0, -ydim, -1)*pdim + (pdim*ydim)/2.0 - pdim/2.0
    mats = [pulse(2*np.pi*uv[0], 2*np.pi*uv[1], pdim, dom="F") *
            np.outer(np.exp(2j*np.pi*ylist*uv[1]), np.exp(2j*np.pi*xlist*uv[0]))
            for uv in uvlist]
    out = np.reshape(np.array(mats), (len(uvlist), xdim*ydim))
    if len(mask):
        out = out[:, mask]
    return out


def _uvlist(nvis, seed=3):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((nvis, 2)) * 1e9


# xdim != ydim throughout: a square grid hides axis-order mistakes.
FT_XDIM, FT_YDIM = 16, 24
FT_PSIZE = 200 * 4.848136811133344e-12 / FT_XDIM   # 200 uas FOV in radians


def test_ftmatrix_matches_reference_unmasked():
    uv = _uvlist(40)
    got = obsh.ftmatrix(FT_PSIZE, FT_XDIM, FT_YDIM, uv)
    assert np.array_equal(got, _ftmatrix_reference(FT_PSIZE, FT_XDIM, FT_YDIM, uv))


def test_ftmatrix_matches_reference_masked():
    # A partial mask is the interesting case: it is what the imager actually
    # passes, and it is where a preallocating rewrite is easiest to get wrong.
    uv = _uvlist(40)
    rng = np.random.default_rng(11)
    mask = rng.random(FT_XDIM * FT_YDIM) > 0.4
    got = obsh.ftmatrix(FT_PSIZE, FT_XDIM, FT_YDIM, uv, mask=mask)
    expected = _ftmatrix_reference(FT_PSIZE, FT_XDIM, FT_YDIM, uv, mask=mask)
    assert got.shape == (len(uv), int(mask.sum()))
    assert np.array_equal(got, expected)


def test_ftmatrix_build_peak_stays_near_its_result():
    uv = _uvlist(800)
    peak = _peak_mib(lambda: obsh.ftmatrix(FT_PSIZE, 32, 32, uv))
    result_mib = len(uv) * 32 * 32 * 16 / 2**20
    assert peak < FT_PEAK_FRACTION * result_mib, (
        f"ftmatrix peaked at {peak:.2f} MiB building a {result_mib:.2f} MiB "
        f"operator ({peak/result_mib:.2f}x)")


def test_ftmatrix_masked_build_does_not_allocate_the_full_stack():
    # With a mask the returned array is much smaller than the full grid, and
    # the build should never materialize the full one.
    uv = _uvlist(800)
    rng = np.random.default_rng(7)
    mask = rng.random(32 * 32) > 0.75          # keep roughly a quarter
    peak = _peak_mib(lambda: obsh.ftmatrix(FT_PSIZE, 32, 32, uv, mask=mask))
    result_mib = len(uv) * int(mask.sum()) * 16 / 2**20
    assert peak < FT_PEAK_FRACTION * result_mib, (
        f"ftmatrix peaked at {peak:.2f} MiB building a {result_mib:.2f} MiB "
        f"masked operator ({peak/result_mib:.2f}x)")
