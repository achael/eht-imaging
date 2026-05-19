"""Tests for the mixed-polarization data infrastructure.

Tests are grouped by the phase of obsdata_mixedpol_plan_v2.md they cover.
Subsequent phases (2, 2.5, 3, ...) should append their tests to this file
under matching headers.
"""
import os
import pickle

import numpy as np
import pytest

import ehtim.array as ea
import ehtim.caltable as ec
import ehtim.const_def as ehc
import ehtim.obsdata as eo
import ehtim.warnings as ehw

# Legacy (pre-Phase-1) dtypes — used to verify upgrade plumbing.
_LEGACY_DTARR = [('site', 'U32'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                 ('sefdr', 'f8'), ('sefdl', 'f8'), ('dr', 'c16'), ('dl', 'c16'),
                 ('fr_par', 'f8'), ('fr_elev', 'f8'), ('fr_off', 'f8')]

_LEGACY_DTPOL_CIRC = [('time', 'f8'), ('tint', 'f8'),
                     ('t1', 'U32'), ('t2', 'U32'),
                     ('tau1', 'f8'), ('tau2', 'f8'),
                     ('u', 'f8'), ('v', 'f8'),
                     ('rrvis', 'c16'), ('llvis', 'c16'),
                     ('rlvis', 'c16'), ('lrvis', 'c16'),
                     ('rrsigma', 'f8'), ('llsigma', 'f8'),
                     ('rlsigma', 'f8'), ('lrsigma', 'f8')]

_LEGACY_DTCAL = [('time', 'f8'), ('rscale', 'c16'), ('lscale', 'c16')]


def _legacy_tarr():
    ot = np.zeros(2, dtype=_LEGACY_DTARR)
    ot['site'] = ['A', 'B']
    ot['sefdr'] = [1000., 2000.]
    ot['sefdl'] = [1100., 2100.]
    return ot


def _legacy_circ_datatable():
    od = np.zeros(2, dtype=_LEGACY_DTPOL_CIRC)
    od['time'] = [0.0, 0.5]
    od['t1'] = ['A', 'A']
    od['t2'] = ['B', 'B']
    od['rrvis'] = [1 + 0j, 2 + 0j]
    od['rrsigma'] = [0.1, 0.2]
    return od


# ============================================================================
# Phase 1 — Schema additions and warnings module
# ============================================================================
#
# Plan reference: obsdata_mixedpol_plan_v2.md, "Phase 1".
# Scope:
#   - ehtim/warnings.py module + warning classes
#   - DTARR migration to 12 fields with feed_type
#   - DTPOL_CIRC / DTPOL_LIN / DTPOL_MIXED / DTCAL_CIRC / DTCAL_LIN dtypes
#   - feed_dtype_for_polrep, feed_poldict helpers
#   - Silent schema upgrade on __init__ and __setstate__ for Array,
#     Obsdata, and Caltable.
#
# These tests guard the backwards-compatibility commitment: legacy
# circular-feed data must keep producing numerically bit-identical
# numerical values, with the dtype object expanding only by additive
# columns and title aliases.

# ----- Warnings module -----------------------------------------------------

def test_phase1_warning_classes_are_user_warning_subclasses():
    assert issubclass(ehw.MixedPolConventionWarning, UserWarning)
    assert issubclass(ehw.MixedPolClosureSkipWarning, UserWarning)


# ----- DTARR ---------------------------------------------------------------

def test_phase1_dtarr_alias_equivalence():
    a = np.zeros(2, dtype=ehc.DTARR)
    a['sefd_p1'] = [1000., 2000.]
    assert np.array_equal(a['sefdr'], a['sefd_p1'])
    a['d_p1'] = [0.1 + 0.2j, 0.3 + 0.4j]
    assert np.array_equal(a['dr'], a['d_p1'])


def test_phase1_dtarr_primary_names_are_generic():
    names = np.dtype(ehc.DTARR).names
    assert 'sefd_p1' in names and 'sefd_p2' in names
    assert 'd_p1' in names and 'd_p2' in names
    assert 'feed_type' in names
    # legacy names not in dtype.names (they are title aliases)
    assert 'sefdr' not in names
    assert 'sefdl' not in names


def test_phase1_dtarr_default_feed_type_is_empty_until_filled():
    a = np.zeros(1, dtype=ehc.DTARR)
    # np.zeros gives an empty unicode string; ehtim's constructors fill 'rl'.
    assert a['feed_type'][0] == ''


# ----- DTPOL_CIRC ----------------------------------------------------------

def test_phase1_dtpol_circ_alias_equivalence():
    d = np.zeros(2, dtype=ehc.DTPOL_CIRC)
    d['rrvis'] = [1 + 0j, 2 + 0j]
    assert np.array_equal(d['p1p1vis'], d['rrvis'])
    d['rlvis'] = [3 + 1j, 4 - 1j]
    assert np.array_equal(d['p1p2vis'], d['rlvis'])
    d['rrsigma'] = [0.1, 0.2]
    assert np.array_equal(d['p1p1sigma'], d['rrsigma'])


def test_phase1_dtpol_circ_primary_names_are_physical():
    names = np.dtype(ehc.DTPOL_CIRC).names
    assert 'rrvis' in names and 'lrvis' in names
    assert 'p1p1vis' not in names  # generic is a title alias, not primary


def test_phase1_dtpol_circ_wrong_feed_access_raises():
    d = np.zeros(1, dtype=ehc.DTPOL_CIRC)
    with pytest.raises((ValueError, KeyError)):
        d['xxvis']


# ----- DTPOL_LIN -----------------------------------------------------------

def test_phase1_dtpol_lin_alias_equivalence():
    d = np.zeros(2, dtype=ehc.DTPOL_LIN)
    d['xxvis'] = [1 + 0j, 2 + 0j]
    assert np.array_equal(d['p1p1vis'], d['xxvis'])
    d['yxvis'] = [3 + 0j, 4 + 0j]
    assert np.array_equal(d['p2p1vis'], d['yxvis'])


def test_phase1_dtpol_lin_wrong_feed_access_raises():
    d = np.zeros(1, dtype=ehc.DTPOL_LIN)
    with pytest.raises((ValueError, KeyError)):
        d['rrvis']


# ----- DTPOL_MIXED ---------------------------------------------------------

def test_phase1_dtpol_mixed_has_only_generic_names_and_polbasis():
    names = np.dtype(ehc.DTPOL_MIXED).names
    assert 'p1p1vis' in names and 'p2p2vis' in names
    assert 'p1p2vis' in names and 'p2p1vis' in names
    assert 'polbasis' in names
    for n in ('rrvis', 'llvis', 'xxvis', 'yyvis'):
        assert n not in names


# ----- DTCAL ---------------------------------------------------------------

def test_phase1_dtcal_circ_alias_equivalence():
    c = np.zeros(2, dtype=ehc.DTCAL_CIRC)
    c['rscale'] = [1 + 0j, 2 + 0j]
    assert np.array_equal(c['p1scale'], c['rscale'])
    c['dr'] = [0.01 + 0.02j, 0.03 - 0.01j]
    assert np.array_equal(c['d_p1'], c['dr'])


def test_phase1_dtcal_lin_alias_equivalence():
    c = np.zeros(2, dtype=ehc.DTCAL_LIN)
    c['xscale'] = [1 + 0j, 2 + 0j]
    assert np.array_equal(c['p1scale'], c['xscale'])


def test_phase1_dtcal_legacy_alias_is_dtcal_circ():
    assert ehc.DTCAL is ehc.DTCAL_CIRC


# ----- Polrep / feed dispatch helpers --------------------------------------

def test_phase1_feed_dtype_for_polrep():
    assert ehc.feed_dtype_for_polrep('stokes') is ehc.DTPOL_STOKES
    assert ehc.feed_dtype_for_polrep('circ') is ehc.DTPOL_CIRC
    assert ehc.feed_dtype_for_polrep('lin') is ehc.DTPOL_LIN
    assert ehc.feed_dtype_for_polrep('mixed') is ehc.DTPOL_MIXED
    with pytest.raises(ValueError):
        ehc.feed_dtype_for_polrep('bogus')


def test_phase1_feed_poldict_circular_pair():
    d = ehc.feed_poldict('rl', 'rl')
    assert d['vis1'] == 'rrvis'
    assert d['vis2'] == 'llvis'
    assert d['vis3'] == 'rlvis'
    assert d['vis4'] == 'lrvis'


def test_phase1_feed_poldict_linear_pair():
    d = ehc.feed_poldict('xy', 'xy')
    assert d['vis1'] == 'xxvis'
    assert d['vis2'] == 'yyvis'
    assert d['vis3'] == 'xyvis'
    assert d['vis4'] == 'yxvis'


def test_phase1_feed_poldict_mixed_pair():
    d = ehc.feed_poldict('rl', 'xy')
    assert d['vis1'] == 'rxvis'
    assert d['vis2'] == 'lyvis'
    assert d['vis3'] == 'ryvis'
    assert d['vis4'] == 'lxvis'


def test_phase1_feed_poldict_unknown_raises():
    with pytest.raises(ValueError):
        ehc.feed_poldict('rl', '??')
    with pytest.raises(ValueError):
        ehc.feed_poldict('??', 'rl')


# ----- Schema upgrade helpers ---------------------------------------------

def test_phase1_upgrade_tarr_from_legacy():
    ot = np.zeros(2, dtype=_LEGACY_DTARR)
    ot['site'] = ['A', 'B']
    ot['sefdr'] = [1000., 2000.]
    ot['sefdl'] = [1100., 2100.]
    ot['dr'] = [0.01 + 0.02j, 0.03 + 0.04j]
    nt = ehc.upgrade_tarr(ot)
    assert 'feed_type' in nt.dtype.names
    assert np.array_equal(nt['feed_type'], np.array(['rl', 'rl']))
    # numerically bit-identical for the legacy columns
    assert np.array_equal(nt['sefd_p1'], ot['sefdr'])
    assert np.array_equal(nt['sefd_p2'], ot['sefdl'])
    assert np.array_equal(nt['d_p1'], ot['dr'])


def test_phase1_upgrade_tarr_idempotent():
    a = np.zeros(1, dtype=ehc.DTARR)
    a['feed_type'] = 'rl'
    assert ehc.upgrade_tarr(a) is a


def test_phase1_upgrade_dtpol_circ_zero_copy_view():
    od = np.zeros(3, dtype=_LEGACY_DTPOL_CIRC)
    od['rrvis'] = [1 + 0j, 2 + 0j, 3 + 0j]
    nd = ehc.upgrade_dtpol_circ(od)
    # view-cast: same underlying buffer
    assert nd.base is od
    # both legacy and generic accessors work
    assert np.array_equal(nd['rrvis'], od['rrvis'])
    assert np.array_equal(nd['p1p1vis'], od['rrvis'])


def test_phase1_upgrade_dtpol_circ_idempotent():
    d = np.zeros(1, dtype=ehc.DTPOL_CIRC)
    out = ehc.upgrade_dtpol_circ(d)
    assert out.dtype == d.dtype


def test_phase1_upgrade_dtcal_circ_from_legacy_adds_dterm_fields():
    oc = np.zeros(2, dtype=_LEGACY_DTCAL)
    oc['rscale'] = [1 + 0j, 2 + 0j]
    nc = ehc.upgrade_dtcal_circ(oc)
    assert nc.dtype.names == ('time', 'rscale', 'lscale', 'dr', 'dl')
    assert np.array_equal(nc['rscale'], oc['rscale'])
    # dr/dl default to 0
    assert np.all(nc['dr'] == 0)
    assert np.all(nc['dl'] == 0)


# ----- End-to-end: Array / Obsdata / Caltable upgrade through __init__ ----

def test_phase1_array_upgrades_legacy_tarr():
    arr = ea.Array(_legacy_tarr())
    assert 'feed_type' in arr.tarr.dtype.names
    assert np.all(arr.tarr['feed_type'] == 'rl')


def test_phase1_array_pickle_roundtrip_from_new():
    arr = ea.Array(_legacy_tarr())
    arr2 = pickle.loads(pickle.dumps(arr))
    assert np.all(arr2.tarr['feed_type'] == 'rl')
    assert np.array_equal(arr2.tarr['sefd_p1'], arr.tarr['sefd_p1'])


def test_phase1_array_setstate_upgrades_legacy_pickle():
    # Simulate a pickle made before the schema migration:
    legacy_state = {'_tarr': _legacy_tarr(), 'ephem': {},
                    'tkey': {'A': 0, 'B': 1}}
    arr = ea.Array.__new__(ea.Array)
    arr.__setstate__(legacy_state)
    assert 'feed_type' in arr.tarr.dtype.names
    assert np.all(arr.tarr['feed_type'] == 'rl')


def test_phase1_obsdata_upgrades_legacy_datatable_and_tarr():
    obs = eo.Obsdata(ra=0., dec=0., rf=230e9, bw=1e9,
                     datatable=_legacy_circ_datatable(),
                     tarr=_legacy_tarr(), polrep='circ')
    assert obs.polrep == 'circ'
    assert 'feed_type' in obs.tarr.dtype.names
    assert np.all(obs.tarr['feed_type'] == 'rl')
    # Generic accessor works on upgraded data
    assert np.array_equal(obs.data['p1p1vis'], obs.data['rrvis'])


def test_phase1_obsdata_pickle_roundtrip_legacy_recarrays():
    obs = eo.Obsdata(ra=0., dec=0., rf=230e9, bw=1e9,
                     datatable=_legacy_circ_datatable(),
                     tarr=_legacy_tarr(), polrep='circ')
    obs2 = pickle.loads(pickle.dumps(obs))
    assert np.all(obs2.tarr['feed_type'] == 'rl')
    assert np.array_equal(obs2.data['p1p1vis'], obs.data['rrvis'])


def test_phase1_caltable_upgrades_legacy_dtcal():
    oc = np.zeros(2, dtype=_LEGACY_DTCAL)
    oc['rscale'] = [1 + 0j, 1.1 + 0j]
    caltab = ec.Caltable(ra=0., dec=0., rf=230e9, bw=1e9,
                         datadict={'A': oc.copy(), 'B': oc.copy()},
                         tarr=_legacy_tarr())
    for site in ('A', 'B'):
        assert 'dr' in caltab.data[site].dtype.names
        assert 'd_p1' in caltab.data[site].dtype.fields  # title alias
        assert np.all(caltab.data[site]['dr'] == 0)


def test_phase1_caltable_pickle_roundtrip_legacy_dtcal():
    oc = np.zeros(2, dtype=_LEGACY_DTCAL)
    oc['rscale'] = [1 + 0j, 1.1 + 0j]
    caltab = ec.Caltable(ra=0., dec=0., rf=230e9, bw=1e9,
                         datadict={'A': oc.copy()}, tarr=_legacy_tarr())
    caltab2 = pickle.loads(pickle.dumps(caltab))
    assert 'dr' in caltab2.data['A'].dtype.names
    assert np.array_equal(caltab2.data['A']['rscale'],
                          caltab.data['A']['rscale'])


# ============================================================================
# Phase 2 — Array class migration
# ============================================================================
#
# Plan reference: obsdata_mixedpol_plan_v2.md, "Phase 2".
# Scope:
#   - Array query methods: is_homogeneous_feeds, feed_types,
#     sefd_for_feed, dterm_for_feed
#   - add_site / add_satellite_tle / add_satellite_elements feed_type kwarg
#   - TarrView wrapper raising on legacy R/L access for non-RL stations
#   - Array.obsdata(polrep=...) extended to 'lin' and 'mixed'
#   - save_array_txt / load_array_txt v2 versioned text format
#   - load_obs_txt embedded-tarr versioned header

def _mixed_tarr():
    """Two-site tarr: one 'rl', one 'xy'."""
    t = np.zeros(2, dtype=ehc.DTARR)
    t['site'] = ['ALMA', 'APEX']
    t['sefd_p1'] = [100., 4000.]
    t['sefd_p2'] = [110., 4100.]
    t['d_p1'] = [0.01 + 0.02j, 0.03 + 0.04j]
    t['d_p2'] = [0.05 + 0.06j, 0.07 + 0.08j]
    t['feed_type'] = ['xy', 'rl']
    return t


def _xy_tarr():
    """Two-site all-linear-feed tarr."""
    t = np.zeros(2, dtype=ehc.DTARR)
    t['site'] = ['A', 'B']
    t['sefd_p1'] = [1000., 2000.]
    t['sefd_p2'] = [1100., 2100.]
    t['d_p1'] = [0.1 + 0.0j, 0.2 + 0.0j]
    t['d_p2'] = [0.3 + 0.0j, 0.4 + 0.0j]
    t['feed_type'] = ['xy', 'xy']
    return t


# ----- Array query methods --------------------------------------------------

def test_phase2_is_homogeneous_feeds_legacy_array_true():
    arr = ea.Array(_legacy_tarr())
    assert arr.is_homogeneous_feeds() is True


def test_phase2_is_homogeneous_feeds_xy_array_true():
    arr = ea.Array(_xy_tarr())
    assert arr.is_homogeneous_feeds() is True


def test_phase2_is_homogeneous_feeds_mixed_array_false():
    arr = ea.Array(_mixed_tarr())
    assert arr.is_homogeneous_feeds() is False


def test_phase2_feed_types_legacy_array():
    arr = ea.Array(_legacy_tarr())
    assert arr.feed_types() == {'rl'}


def test_phase2_feed_types_mixed_array():
    arr = ea.Array(_mixed_tarr())
    assert arr.feed_types() == {'rl', 'xy'}


def test_phase2_sefd_for_feed_rl_station_returns_p1_p2():
    arr = ea.Array(_mixed_tarr())
    assert arr.sefd_for_feed('APEX', 'R') == 4000.
    assert arr.sefd_for_feed('APEX', 'L') == 4100.


def test_phase2_sefd_for_feed_xy_station_returns_p1_p2():
    arr = ea.Array(_mixed_tarr())
    assert arr.sefd_for_feed('ALMA', 'X') == 100.
    assert arr.sefd_for_feed('ALMA', 'Y') == 110.


def test_phase2_sefd_for_feed_case_insensitive():
    arr = ea.Array(_mixed_tarr())
    assert arr.sefd_for_feed('ALMA', 'x') == arr.sefd_for_feed('ALMA', 'X')


def test_phase2_sefd_for_feed_wrong_feed_raises():
    arr = ea.Array(_mixed_tarr())
    # ALMA is 'xy'; asking for R should raise.
    with pytest.raises(ValueError, match="feed 'R' not in feed_type 'xy'"):
        arr.sefd_for_feed('ALMA', 'R')


def test_phase2_sefd_for_feed_unknown_site_raises():
    arr = ea.Array(_mixed_tarr())
    with pytest.raises(KeyError, match="site 'NOSITE' not in array"):
        arr.sefd_for_feed('NOSITE', 'R')


def test_phase2_dterm_for_feed_rl_station_dispatch():
    arr = ea.Array(_mixed_tarr())
    assert arr.dterm_for_feed('APEX', 'R') == 0.03 + 0.04j
    assert arr.dterm_for_feed('APEX', 'L') == 0.07 + 0.08j


def test_phase2_dterm_for_feed_xy_station_dispatch():
    arr = ea.Array(_mixed_tarr())
    assert arr.dterm_for_feed('ALMA', 'X') == 0.01 + 0.02j
    assert arr.dterm_for_feed('ALMA', 'Y') == 0.05 + 0.06j


def test_phase2_dterm_for_feed_wrong_feed_raises():
    arr = ea.Array(_mixed_tarr())
    with pytest.raises(ValueError, match="not in feed_type"):
        arr.dterm_for_feed('APEX', 'X')


# ----- add_site signature extension -----------------------------------------

def _empty_array():
    return ea.Array(np.zeros(0, dtype=ehc.DTARR))


def test_phase2_add_site_default_kwargs_bit_identical_to_legacy():
    arr = _empty_array().add_site('A', (1.0, 2.0, 3.0))
    row = arr.tarr[arr.tkey['A']]
    assert row['sefd_p1'] == 10000.0
    assert row['sefd_p2'] == 10000.0
    assert row['d_p1'] == 0 + 0j
    assert row['d_p2'] == 0 + 0j
    assert str(row['feed_type']) == 'rl'


def test_phase2_add_site_legacy_sefd_and_dr_dl_still_work():
    arr = _empty_array().add_site('A', (1., 2., 3.),
                                  sefd=5000, dr=0.1 + 0.2j, dl=0.3 + 0.4j)
    row = arr.tarr[arr.tkey['A']]
    assert row['sefd_p1'] == 5000.0
    assert row['sefd_p2'] == 5000.0
    assert row['d_p1'] == 0.1 + 0.2j
    assert row['d_p2'] == 0.3 + 0.4j
    assert str(row['feed_type']) == 'rl'


def test_phase2_add_site_with_feed_type_xy_and_generic_sefds():
    arr = _empty_array().add_site('A', (1., 2., 3.),
                                  feed_type='xy',
                                  sefd_p1=100., sefd_p2=110.,
                                  d_p1=0.01 + 0j, d_p2=0.02 + 0j)
    row = arr.tarr[arr.tkey['A']]
    assert str(row['feed_type']) == 'xy'
    assert row['sefd_p1'] == 100.0
    assert row['sefd_p2'] == 110.0
    assert row['d_p1'] == 0.01 + 0j
    assert row['d_p2'] == 0.02 + 0j


def test_phase2_add_site_dr_with_non_rl_feed_raises():
    with pytest.raises(ValueError, match="dr/dl kwargs are only valid"):
        _empty_array().add_site('A', (0., 0., 0.),
                                feed_type='xy', dr=0.1 + 0j)


def test_phase2_add_site_dl_with_non_rl_feed_raises():
    with pytest.raises(ValueError, match="dr/dl kwargs are only valid"):
        _empty_array().add_site('A', (0., 0., 0.),
                                feed_type='xy', dl=0.1 + 0j)


def test_phase2_add_site_dr_and_d_p1_both_raises():
    with pytest.raises(ValueError, match="dr.*d_p1"):
        _empty_array().add_site('A', (0., 0., 0.),
                                dr=0.1 + 0j, d_p1=0.2 + 0j)


def test_phase2_add_site_dl_and_d_p2_both_raises():
    with pytest.raises(ValueError, match="dl.*d_p2"):
        _empty_array().add_site('A', (0., 0., 0.),
                                dl=0.1 + 0j, d_p2=0.2 + 0j)


def test_phase2_add_site_sefd_and_sefd_p1_both_raises():
    with pytest.raises(ValueError, match="sefd.*sefd_p1"):
        _empty_array().add_site('A', (0., 0., 0.),
                                sefd=5000, sefd_p1=100., sefd_p2=110.)


def test_phase2_add_site_sefd_p1_without_p2_raises():
    with pytest.raises(ValueError, match="sefd_p1 and sefd_p2"):
        _empty_array().add_site('A', (0., 0., 0.),
                                feed_type='xy', sefd_p1=100.)


def test_phase2_add_site_invalid_feed_type_raises():
    with pytest.raises(ValueError, match="feed_type must be one of"):
        _empty_array().add_site('A', (0., 0., 0.), feed_type='zz')


# ----- add_satellite_* signature extension ----------------------------------

def test_phase2_add_satellite_tle_default_feed_type_rl():
    tle = ['SAT1', 'line1', 'line2']
    arr = _empty_array().add_satellite_tle(tle, sefd=10000)
    row = arr.tarr[arr.tkey['SAT1']]
    assert str(row['feed_type']) == 'rl'
    assert row['sefd_p1'] == 10000.0
    assert row['sefd_p2'] == 10000.0


def test_phase2_add_satellite_tle_with_feed_type_xy():
    tle = ['SAT1', 'line1', 'line2']
    arr = _empty_array().add_satellite_tle(tle, feed_type='xy',
                                            sefd_p1=500., sefd_p2=600.)
    row = arr.tarr[arr.tkey['SAT1']]
    assert str(row['feed_type']) == 'xy'
    assert row['sefd_p1'] == 500.0
    assert row['sefd_p2'] == 600.0


def test_phase2_add_satellite_elements_default_feed_type_rl():
    arr = _empty_array().add_satellite_elements('SAT2', sefd=2000)
    row = arr.tarr[arr.tkey['SAT2']]
    assert str(row['feed_type']) == 'rl'
    assert row['sefd_p1'] == 2000.0


def test_phase2_add_satellite_elements_with_feed_type_xy_and_dterms():
    arr = _empty_array().add_satellite_elements('SAT2', feed_type='xy',
                                                 sefd_p1=300., sefd_p2=400.,
                                                 d_p1=0.05 + 0.01j,
                                                 d_p2=0.06 + 0.02j)
    row = arr.tarr[arr.tkey['SAT2']]
    assert str(row['feed_type']) == 'xy'
    assert row['sefd_p1'] == 300.0
    assert row['sefd_p2'] == 400.0
    assert row['d_p1'] == 0.05 + 0.01j
    assert row['d_p2'] == 0.06 + 0.02j


def test_phase2_add_satellite_tle_invalid_feed_type_raises():
    tle = ['SAT1', 'line1', 'line2']
    with pytest.raises(ValueError, match="feed_type must be one of"):
        _empty_array().add_satellite_tle(tle, feed_type='qq')


# ----- TarrView wrapper -----------------------------------------------------

def test_phase2_tarrview_is_returned_by_tarr_property():
    arr = ea.Array(_legacy_tarr())
    assert isinstance(arr.tarr, ea.TarrView)


def test_phase2_tarrview_sefdr_on_homogeneous_rl_works():
    arr = ea.Array(_legacy_tarr())
    # Legacy title-alias access still works on an all-RL array.
    np.testing.assert_array_equal(arr.tarr['sefdr'], [1000., 2000.])
    np.testing.assert_array_equal(arr.tarr['sefdl'], [1100., 2100.])


def test_phase2_tarrview_sefdr_on_xy_array_raises():
    arr = ea.Array(_xy_tarr())
    with pytest.raises(KeyError, match=r"tarr\['sefdr'\].*xy"):
        _ = arr.tarr['sefdr']


def test_phase2_tarrview_sefdl_on_xy_array_raises():
    arr = ea.Array(_xy_tarr())
    with pytest.raises(KeyError, match=r"tarr\['sefdl'\].*xy"):
        _ = arr.tarr['sefdl']


def test_phase2_tarrview_sefdr_on_mixed_array_raises():
    arr = ea.Array(_mixed_tarr())
    with pytest.raises(KeyError, match=r"tarr\['sefdr'\].*"):
        _ = arr.tarr['sefdr']


def test_phase2_tarrview_dr_on_mixed_array_raises():
    arr = ea.Array(_mixed_tarr())
    with pytest.raises(KeyError, match="dterm_for_feed"):
        _ = arr.tarr['dr']


def test_phase2_tarrview_generic_names_always_work():
    # Generic per-feed names are never guarded.
    arr = ea.Array(_mixed_tarr())
    np.testing.assert_array_equal(arr.tarr['sefd_p1'], [100., 4000.])
    np.testing.assert_array_equal(arr.tarr['sefd_p2'], [110., 4100.])


def test_phase2_tarrview_dtype_attribute_forwarded():
    arr = ea.Array(_legacy_tarr())
    assert 'feed_type' in arr.tarr.dtype.names
    assert arr.tarr.dtype == ehc.DTARR


def test_phase2_tarrview_len_iter_forwarded():
    arr = ea.Array(_legacy_tarr())
    assert len(arr.tarr) == 2
    sites = [str(row['site']) for row in arr.tarr]
    assert sites == ['A', 'B']


def test_phase2_tarrview_row_index_returns_void():
    arr = ea.Array(_legacy_tarr())
    row = arr.tarr[0]
    # Single-row index returns a numpy void; field access on it bypasses
    # the guard (documented limitation).
    assert str(row['site']) == 'A'


def test_phase2_tarrview_boolean_mask_returns_wrapped_view():
    arr = ea.Array(_mixed_tarr())
    mask = np.array([True, False])
    sub = arr.tarr[mask]
    assert isinstance(sub, ea.TarrView)
    assert len(sub) == 1
    assert str(sub[0]['site']) == 'ALMA'


def test_phase2_tarrview_boolean_mask_preserves_guard():
    # Slicing a mixed array down to one xy station: legacy 'sefdr'
    # still raises on the subview (feed_types still non-RL).
    arr = ea.Array(_mixed_tarr())
    sub = arr.tarr[np.array([True, False])]
    with pytest.raises(KeyError):
        _ = sub['sefdr']


def test_phase2_tarrview_equality_recarray_lhs():
    arr = ea.Array(_legacy_tarr())
    raw = arr.tarr._tarr.copy()
    cmp = raw == arr.tarr
    assert np.all(cmp)


def test_phase2_tarrview_equality_recarray_rhs():
    arr = ea.Array(_legacy_tarr())
    raw = arr.tarr._tarr.copy()
    cmp = arr.tarr == raw
    assert np.all(cmp)


def test_phase2_tarrview_equality_both_views():
    arr1 = ea.Array(_legacy_tarr())
    arr2 = ea.Array(_legacy_tarr())
    assert np.all(arr1.tarr == arr2.tarr)


def test_phase2_tarrview_pickle_roundtrip():
    arr = ea.Array(_legacy_tarr())
    view = arr.tarr
    view2 = pickle.loads(pickle.dumps(view))
    assert isinstance(view2, ea.TarrView)
    assert np.all(view == view2)


def test_phase2_tarrview_unhashable():
    arr = ea.Array(_legacy_tarr())
    with pytest.raises(TypeError):
        hash(arr.tarr)


def test_phase2_tarrview_numpy_interop_via_array_protocol():
    arr = ea.Array(_legacy_tarr())
    # np.asarray should produce a recarray view (no copy required by spec).
    asarr = np.asarray(arr.tarr)
    assert isinstance(asarr, np.ndarray)
    assert asarr.dtype.names == arr.tarr.dtype.names


def test_phase2_tarrview_setter_unwraps_view():
    # Assigning a TarrView via the property setter unwraps it so _tarr
    # remains a plain ndarray (the storage truth).
    arr = ea.Array(_legacy_tarr())
    view = arr.tarr  # TarrView
    arr.tarr = view  # round-trip via setter
    assert not isinstance(arr._tarr, ea.TarrView)
    assert isinstance(arr._tarr, np.ndarray)
    assert arr._tarr.dtype.names is not None  # structured dtype preserved


def test_phase2_tarrview_array_pickle_roundtrip_preserves_ndarray_storage():
    # The Array itself round-trips through pickle: storage stays as an
    # ndarray (not a TarrView), so the on-disk format is unchanged.
    arr = ea.Array(_legacy_tarr())
    arr2 = pickle.loads(pickle.dumps(arr))
    assert isinstance(arr2._tarr, np.ndarray)
    assert not isinstance(arr2._tarr, ea.TarrView)
    assert isinstance(arr2.tarr, ea.TarrView)
    assert np.all(arr2.tarr == arr.tarr)


# ----- Array.obsdata(polrep=...) validation ---------------------------------

def _eht_like_rl_array():
    """Two-site array with realistic ground-station coordinates."""
    t = np.zeros(2, dtype=ehc.DTARR)
    t['site'] = ['ALMA', 'APEX']
    t['x'] = [2225061.16, 2225039.53]
    t['y'] = [-5440057.37, -5441197.63]
    t['z'] = [-2481681.15, -2479303.36]
    t['sefd_p1'] = [100., 4000.]
    t['sefd_p2'] = [100., 4000.]
    t['feed_type'] = ['rl', 'rl']
    return t


def _eht_like_xy_array():
    t = _eht_like_rl_array()
    t['feed_type'] = ['xy', 'xy']
    return t


def _eht_like_mixed_array():
    t = _eht_like_rl_array()
    t['feed_type'] = ['xy', 'rl']
    return t


def _obs_kwargs():
    # Sgr A* coords with full 24h window so ALMA/APEX intersect.
    return dict(ra=17.76, dec=-29.0, rf=230e9, bw=1e9,
                tint=60.0, tadv=600.0, tstart=0.0, tstop=24.0)


def test_phase2_obsdata_polrep_invalid_raises():
    arr = ea.Array(_eht_like_rl_array())
    with pytest.raises(ValueError, match="polrep must be one of"):
        arr.obsdata(polrep='bogus', **_obs_kwargs())


def test_phase2_obsdata_polrep_stokes_on_rl_array_succeeds():
    arr = ea.Array(_eht_like_rl_array())
    obs = arr.obsdata(polrep='stokes', **_obs_kwargs())
    assert obs.polrep == 'stokes'


def test_phase2_obsdata_polrep_circ_on_rl_array_succeeds():
    arr = ea.Array(_eht_like_rl_array())
    obs = arr.obsdata(polrep='circ', **_obs_kwargs())
    assert obs.polrep == 'circ'


def test_phase2_obsdata_polrep_circ_on_xy_array_raises():
    arr = ea.Array(_eht_like_xy_array())
    with pytest.raises(ValueError,
                       match="polrep='circ' requires all stations.*'rl'"):
        arr.obsdata(polrep='circ', **_obs_kwargs())


def test_phase2_obsdata_polrep_circ_on_mixed_array_raises():
    arr = ea.Array(_eht_like_mixed_array())
    with pytest.raises(ValueError, match="polrep='circ'"):
        arr.obsdata(polrep='circ', **_obs_kwargs())


def test_phase2_obsdata_polrep_lin_on_rl_array_raises():
    arr = ea.Array(_eht_like_rl_array())
    with pytest.raises(ValueError,
                       match="polrep='lin' requires all stations.*'xy'"):
        arr.obsdata(polrep='lin', **_obs_kwargs())


def test_phase2_obsdata_polrep_lin_on_xy_array_raises_not_implemented():
    # Validation passes (all xy) but simulation backend doesn't support
    # 'lin' yet — should raise NotImplementedError pointing to Phase 5.
    arr = ea.Array(_eht_like_xy_array())
    with pytest.raises(NotImplementedError, match="Phase 5"):
        arr.obsdata(polrep='lin', **_obs_kwargs())


def test_phase2_obsdata_polrep_mixed_on_homogeneous_array_raises():
    arr = ea.Array(_eht_like_rl_array())
    with pytest.raises(ValueError,
                       match="polrep='mixed' requires at least two"):
        arr.obsdata(polrep='mixed', **_obs_kwargs())


def test_phase2_obsdata_polrep_mixed_on_mixed_array_raises_not_implemented():
    arr = ea.Array(_eht_like_mixed_array())
    with pytest.raises(NotImplementedError, match="Phase 5"):
        arr.obsdata(polrep='mixed', **_obs_kwargs())


# ----- save_array_txt v2 + load_array_txt round-trip ------------------------

def _full_tarr(feed_types):
    """Synthetic tarr with non-trivial values across all DTARR fields."""
    n = len(feed_types)
    t = np.zeros(n, dtype=ehc.DTARR)
    t['site'] = [f'S{i}' for i in range(n)]
    t['x'] = [1.0 * (i + 1) for i in range(n)]
    t['y'] = [2.0 * (i + 1) for i in range(n)]
    t['z'] = [3.0 * (i + 1) for i in range(n)]
    t['sefd_p1'] = [100.0 * (i + 1) for i in range(n)]
    t['sefd_p2'] = [110.0 * (i + 1) for i in range(n)]
    t['d_p1'] = [complex(0.01 * (i + 1), 0.02 * (i + 1)) for i in range(n)]
    t['d_p2'] = [complex(0.03 * (i + 1), 0.04 * (i + 1)) for i in range(n)]
    t['fr_par'] = [1.0 * (i + 1) for i in range(n)]
    t['fr_elev'] = [0.5 * (i + 1) for i in range(n)]
    t['fr_off'] = [0.1 * (i + 1) for i in range(n)]
    t['feed_type'] = feed_types
    return t


def _assert_tarr_round_trip(arr_before, arr_after, atol_sefd=1e-2,
                             atol_d=1e-4, atol_fr=1e-2, atol_coord=1e-5):
    """Compare numeric fields with tolerances matching save_array_txt's
    printf precision; feed_type and site are exact."""
    a, b = arr_before.tarr._tarr, arr_after.tarr._tarr
    assert np.array_equal(a['site'], b['site'])
    assert np.array_equal(a['feed_type'], b['feed_type'])
    np.testing.assert_allclose(a['x'], b['x'], atol=atol_coord)
    np.testing.assert_allclose(a['y'], b['y'], atol=atol_coord)
    np.testing.assert_allclose(a['z'], b['z'], atol=atol_coord)
    np.testing.assert_allclose(a['sefd_p1'], b['sefd_p1'], atol=atol_sefd)
    np.testing.assert_allclose(a['sefd_p2'], b['sefd_p2'], atol=atol_sefd)
    np.testing.assert_allclose(a['d_p1'].real, b['d_p1'].real, atol=atol_d)
    np.testing.assert_allclose(a['d_p1'].imag, b['d_p1'].imag, atol=atol_d)
    np.testing.assert_allclose(a['d_p2'].real, b['d_p2'].real, atol=atol_d)
    np.testing.assert_allclose(a['d_p2'].imag, b['d_p2'].imag, atol=atol_d)
    np.testing.assert_allclose(a['fr_par'], b['fr_par'], atol=atol_fr)
    np.testing.assert_allclose(a['fr_elev'], b['fr_elev'], atol=atol_fr)
    np.testing.assert_allclose(a['fr_off'], b['fr_off'], atol=atol_fr)


def test_phase2_save_array_txt_emits_v2_header(tmp_path):
    arr = ea.Array(_full_tarr(['rl', 'rl']))
    fname = str(tmp_path / 'a.txt')
    arr.save_txt(fname)
    with open(fname) as f:
        first = f.readline().strip()
    assert first == '# ehtim array format v2'


def test_phase2_save_load_array_txt_roundtrip_rl(tmp_path):
    arr = ea.Array(_full_tarr(['rl', 'rl']))
    fname = str(tmp_path / 'a.txt')
    arr.save_txt(fname)
    arr2 = ea.load_txt(fname)
    _assert_tarr_round_trip(arr, arr2)
    assert set(arr2.feed_types()) == {'rl'}


def test_phase2_save_load_array_txt_roundtrip_xy(tmp_path):
    arr = ea.Array(_full_tarr(['xy', 'xy', 'xy']))
    fname = str(tmp_path / 'a.txt')
    arr.save_txt(fname)
    arr2 = ea.load_txt(fname)
    _assert_tarr_round_trip(arr, arr2)
    assert set(arr2.feed_types()) == {'xy'}


def test_phase2_save_load_array_txt_roundtrip_mixed(tmp_path):
    arr = ea.Array(_full_tarr(['rl', 'xy', 'lx']))
    fname = str(tmp_path / 'a.txt')
    arr.save_txt(fname)
    arr2 = ea.load_txt(fname)
    _assert_tarr_round_trip(arr, arr2)
    assert arr2.feed_types() == {'rl', 'xy', 'lx'}


# Legacy unversioned files in arrays/ must continue to load bit-identically.
ARRAYS_DIR = os.path.join(os.path.dirname(__file__), '..', 'arrays')
_LEGACY_ARRAY_FILES = sorted([
    f for f in os.listdir(ARRAYS_DIR) if f.endswith('.txt')
]) if os.path.isdir(ARRAYS_DIR) else []


@pytest.mark.parametrize('fname', _LEGACY_ARRAY_FILES)
def test_phase2_load_array_txt_legacy_bit_identical(fname):
    """For every existing arrays/*.txt file, the loaded tarr's numeric
    columns match a fresh np.loadtxt read of the same file byte-for-byte
    (Phase 1 added a feed_type column; nothing else may differ).
    """
    path = os.path.join(ARRAYS_DIR, fname)
    # SITES.txt is a non-array reference file (site name → code table).
    if fname == 'SITES.txt':
        pytest.skip("SITES.txt is not an Array file")
    arr = ea.load_txt(path)

    raw = np.loadtxt(path, dtype=bytes, comments='#').astype(str)
    if raw[0][0].lower() == 'site':
        raw = raw[1:]
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    ncols = raw.shape[1]
    n = len(arr.tarr)
    assert ncols in (5, 13, 14)
    assert len(raw) == n

    # site, x, y, z always in cols 0..3
    np.testing.assert_array_equal(arr.tarr['site'], raw[:, 0])
    np.testing.assert_array_equal(arr.tarr._tarr['x'], raw[:, 1].astype(float))
    np.testing.assert_array_equal(arr.tarr._tarr['y'], raw[:, 2].astype(float))
    np.testing.assert_array_equal(arr.tarr._tarr['z'], raw[:, 3].astype(float))

    if ncols == 5:
        # SEFD symmetric, all other numeric fields zero, feed_type='rl'
        np.testing.assert_array_equal(arr.tarr._tarr['sefd_p1'],
                                      raw[:, 4].astype(float))
        np.testing.assert_array_equal(arr.tarr._tarr['sefd_p2'],
                                      raw[:, 4].astype(float))
        assert np.all(arr.tarr._tarr['fr_par'] == 0)
        assert np.all(arr.tarr._tarr['d_p1'] == 0)
        assert np.all(arr.tarr._tarr['d_p2'] == 0)
    else:
        np.testing.assert_array_equal(arr.tarr._tarr['sefd_p1'],
                                      raw[:, 4].astype(float))
        np.testing.assert_array_equal(arr.tarr._tarr['sefd_p2'],
                                      raw[:, 5].astype(float))
        np.testing.assert_array_equal(arr.tarr._tarr['fr_par'],
                                      raw[:, 6].astype(float))
        np.testing.assert_array_equal(arr.tarr._tarr['fr_elev'],
                                      raw[:, 7].astype(float))
        np.testing.assert_array_equal(arr.tarr._tarr['fr_off'],
                                      raw[:, 8].astype(float))
        d_p1_re = raw[:, 9].astype(float)
        d_p1_im = raw[:, 10].astype(float)
        d_p2_re = raw[:, 11].astype(float)
        d_p2_im = raw[:, 12].astype(float)
        np.testing.assert_array_equal(arr.tarr._tarr['d_p1'].real, d_p1_re)
        np.testing.assert_array_equal(arr.tarr._tarr['d_p1'].imag, d_p1_im)
        np.testing.assert_array_equal(arr.tarr._tarr['d_p2'].real, d_p2_re)
        np.testing.assert_array_equal(arr.tarr._tarr['d_p2'].imag, d_p2_im)

    # feed_type: 14-col files read it from disk; 5/13-col fall back to 'rl'
    if ncols == 14:
        np.testing.assert_array_equal(arr.tarr['feed_type'], raw[:, 13])
    else:
        assert np.all(arr.tarr['feed_type'] == 'rl')
