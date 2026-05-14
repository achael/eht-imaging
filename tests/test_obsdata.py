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

@pytest.mark.parametrize("vtype", ["vis", "qvis", "uvis", "vvis"])
def test_bispectra_tri_idempotent(obs_pol_direct, vtype):
    sites = list(obs_pol_direct.tarr['site'][:3])
    a = obs_pol_direct.bispectra_tri(*sites, vtype=vtype)
    b = obs_pol_direct.bispectra_tri(*sites, vtype=vtype)
    np.testing.assert_array_equal(a, b)

@pytest.mark.parametrize("vtype", ["vis", "qvis", "uvis", "vvis"])
def test_cphase_tri_idempotent(obs_pol_direct, vtype):
    sites = list(obs_pol_direct.tarr['site'][:3])
    a = obs_pol_direct.cphase_tri(*sites, vtype=vtype)
    b = obs_pol_direct.cphase_tri(*sites, vtype=vtype)
    np.testing.assert_array_equal(a, b)

@pytest.mark.parametrize("ctype", ["camp", "logcamp"])
@pytest.mark.parametrize("vtype", ["vis", "qvis", "uvis", "vvis"])
def test_camp_quad_idempotent(obs_pol_direct, vtype):
    sites = list(obs_pol_direct.tarr['site'][:4])
    a = obs_pol_direct.camp_quad(*sites, vtype=vtype, ctype=ctype)
    b = obs_pol_direct.camp_quad(*sites, vtype=vtype, ctype=ctype)
    np.testing.assert_array_equal(a, b)

def test_save_load_cphase_roundtrip(tmp_path, obs_pol_direct):
    from ehtim.io.save import save_dtype_txt
    from ehtim.io.load import load_dtype_txt
    cph = obs_pol_direct.c_phases(mode='all', count='max', vtype='vis')
    fname = str(tmp_path / "cph.txt")
    save_dtype_txt(obs_pol_direct, fname, data=cph, dtype='cphase')
    loaded = load_dtype_txt(fname, dtype='cphase')
    assert loaded.dtype == cph.dtype
    np.testing.assert_array_almost_equal(loaded['cphase'], cph['cphase'])
