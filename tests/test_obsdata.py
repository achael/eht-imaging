"""Tests for Obsdata methods.

The first batch is Phase 0 of the obsdata cleanup: with the closure-table
cache gone, every call to bispectra() / c_phases() / c_amplitudes() now
recomputes from scratch, so they must be deterministic.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Closure-quantity idempotence (Phase 0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vtype", ["vis", "qvis", "uvis", "vvis"])
def test_bispectra_idempotent_stokes(obs_pol_direct, vtype):
    a = obs_pol_direct.bispectra(mode="all", count="max", vtype=vtype)
    b = obs_pol_direct.bispectra(mode="all", count="max", vtype=vtype)
    assert a.dtype == b.dtype
    assert len(a) == len(b)
    for field in a.dtype.names:
        np.testing.assert_array_equal(a[field], b[field], err_msg=f"field={field}")


@pytest.mark.parametrize("vtype", ["rrvis", "llvis", "rlvis", "lrvis"])
def test_bispectra_idempotent_circ(obs_pol_direct, vtype):
    obs_circ = obs_pol_direct.switch_polrep("circ")
    a = obs_circ.bispectra(mode="all", count="max", vtype=vtype)
    b = obs_circ.bispectra(mode="all", count="max", vtype=vtype)
    assert a.dtype == b.dtype
    assert len(a) == len(b)
    for field in a.dtype.names:
        np.testing.assert_array_equal(a[field], b[field], err_msg=f"field={field}")


@pytest.mark.parametrize("vtype", ["vis", "qvis", "uvis", "vvis"])
def test_c_phases_idempotent(obs_pol_direct, vtype):
    a = obs_pol_direct.c_phases(mode="all", count="max", vtype=vtype)
    b = obs_pol_direct.c_phases(mode="all", count="max", vtype=vtype)
    assert a.dtype == b.dtype
    assert len(a) == len(b)
    for field in a.dtype.names:
        np.testing.assert_array_equal(a[field], b[field], err_msg=f"field={field}")


@pytest.mark.parametrize("ctype", ["camp", "logcamp"])
@pytest.mark.parametrize("vtype", ["vis", "qvis", "uvis", "vvis"])
def test_c_amplitudes_idempotent(obs_pol_direct, vtype, ctype):
    a = obs_pol_direct.c_amplitudes(mode="all", count="max", vtype=vtype, ctype=ctype)
    b = obs_pol_direct.c_amplitudes(mode="all", count="max", vtype=vtype, ctype=ctype)
    assert a.dtype == b.dtype
    assert len(a) == len(b)
    for field in a.dtype.names:
        np.testing.assert_array_equal(a[field], b[field], err_msg=f"field={field}")
