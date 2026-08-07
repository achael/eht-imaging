"""Tests for ehtim.observing.obs_helpers."""
import os
import subprocess
import sys
import tracemalloc
import zlib

import numpy as np
import pytest

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
