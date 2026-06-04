"""Tests for the mixed-polarization data infrastructure.

"""
import os
import pickle
import warnings

import numpy as np
import pytest

import ehtim.array as ea
import ehtim.caltable as ec
import ehtim.const_def as ehc
import ehtim.obsdata as eo
import ehtim.warnings as ehw

# Legacy dtypes — used to verify upgrade plumbing.
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
# Schema additions and warnings module
# ============================================================================
#
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

def test_warning_classes_are_user_warning_subclasses():
    assert issubclass(ehw.MixedPolConventionWarning, UserWarning)
    assert issubclass(ehw.MixedPolClosureSkipWarning, UserWarning)


# ----- DTARR ---------------------------------------------------------------

def test_dtarr_alias_equivalence():
    a = np.zeros(2, dtype=ehc.DTARR)
    a['sefd_p1'] = [1000., 2000.]
    assert np.array_equal(a['sefdr'], a['sefd_p1'])
    a['d_p1'] = [0.1 + 0.2j, 0.3 + 0.4j]
    assert np.array_equal(a['dr'], a['d_p1'])


def test_dtarr_primary_names_are_generic():
    names = np.dtype(ehc.DTARR).names
    assert 'sefd_p1' in names and 'sefd_p2' in names
    assert 'd_p1' in names and 'd_p2' in names
    assert 'feed_type' in names
    # legacy names not in dtype.names (they are title aliases)
    assert 'sefdr' not in names
    assert 'sefdl' not in names


def test_dtarr_default_feed_type_is_empty_until_filled():
    a = np.zeros(1, dtype=ehc.DTARR)
    # np.zeros gives an empty unicode string; ehtim's constructors fill 'rl'.
    assert a['feed_type'][0] == ''


# ----- DTPOL_CIRC ----------------------------------------------------------

def test_dtpol_circ_alias_equivalence():
    d = np.zeros(2, dtype=ehc.DTPOL_CIRC)
    d['rrvis'] = [1 + 0j, 2 + 0j]
    assert np.array_equal(d['p1p1vis'], d['rrvis'])
    d['rlvis'] = [3 + 1j, 4 - 1j]
    assert np.array_equal(d['p1p2vis'], d['rlvis'])
    d['rrsigma'] = [0.1, 0.2]
    assert np.array_equal(d['p1p1sigma'], d['rrsigma'])


def test_dtpol_circ_primary_names_are_physical():
    names = np.dtype(ehc.DTPOL_CIRC).names
    assert 'rrvis' in names and 'lrvis' in names
    assert 'p1p1vis' not in names  # generic is a title alias, not primary


def test_dtpol_circ_wrong_feed_access_raises():
    d = np.zeros(1, dtype=ehc.DTPOL_CIRC)
    with pytest.raises((ValueError, KeyError)):
        d['xxvis']


# ----- DTPOL_LIN -----------------------------------------------------------

def test_dtpol_lin_alias_equivalence():
    d = np.zeros(2, dtype=ehc.DTPOL_LIN)
    d['xxvis'] = [1 + 0j, 2 + 0j]
    assert np.array_equal(d['p1p1vis'], d['xxvis'])
    d['yxvis'] = [3 + 0j, 4 + 0j]
    assert np.array_equal(d['p2p1vis'], d['yxvis'])


def test_dtpol_lin_wrong_feed_access_raises():
    d = np.zeros(1, dtype=ehc.DTPOL_LIN)
    with pytest.raises((ValueError, KeyError)):
        d['rrvis']


# ----- DTPOL_MIXED ---------------------------------------------------------

def test_dtpol_mixed_has_only_generic_names_and_polbasis():
    names = np.dtype(ehc.DTPOL_MIXED).names
    assert 'p1p1vis' in names and 'p2p2vis' in names
    assert 'p1p2vis' in names and 'p2p1vis' in names
    assert 'polbasis' in names
    for n in ('rrvis', 'llvis', 'xxvis', 'yyvis'):
        assert n not in names


# ----- DTCAL ---------------------------------------------------------------

def test_dtcal_circ_alias_equivalence():
    c = np.zeros(2, dtype=ehc.DTCAL_CIRC)
    c['rscale'] = [1 + 0j, 2 + 0j]
    assert np.array_equal(c['p1scale'], c['rscale'])
    c['dr'] = [0.01 + 0.02j, 0.03 - 0.01j]
    assert np.array_equal(c['d_p1'], c['dr'])


def test_dtcal_lin_alias_equivalence():
    c = np.zeros(2, dtype=ehc.DTCAL_LIN)
    c['xscale'] = [1 + 0j, 2 + 0j]
    assert np.array_equal(c['p1scale'], c['xscale'])


def test_dtcal_legacy_alias_is_dtcal_circ():
    assert ehc.DTCAL is ehc.DTCAL_CIRC


# ----- Polrep / feed dispatch helpers --------------------------------------

def test_feed_dtype_for_polrep():
    assert ehc.feed_dtype_for_polrep('stokes') is ehc.DTPOL_STOKES
    assert ehc.feed_dtype_for_polrep('circ') is ehc.DTPOL_CIRC
    assert ehc.feed_dtype_for_polrep('lin') is ehc.DTPOL_LIN
    assert ehc.feed_dtype_for_polrep('mixed') is ehc.DTPOL_MIXED
    with pytest.raises(ValueError):
        ehc.feed_dtype_for_polrep('bogus')


def test_feed_poldict_circular_pair():
    d = ehc.feed_poldict('rl', 'rl')
    assert d['vis1'] == 'rrvis'
    assert d['vis2'] == 'llvis'
    assert d['vis3'] == 'rlvis'
    assert d['vis4'] == 'lrvis'


def test_feed_poldict_linear_pair():
    d = ehc.feed_poldict('xy', 'xy')
    assert d['vis1'] == 'xxvis'
    assert d['vis2'] == 'yyvis'
    assert d['vis3'] == 'xyvis'
    assert d['vis4'] == 'yxvis'


def test_feed_poldict_mixed_pair():
    d = ehc.feed_poldict('rl', 'xy')
    assert d['vis1'] == 'rxvis'
    assert d['vis2'] == 'lyvis'
    assert d['vis3'] == 'ryvis'
    assert d['vis4'] == 'lxvis'


def test_feed_poldict_unknown_raises():
    with pytest.raises(ValueError):
        ehc.feed_poldict('rl', '??')
    with pytest.raises(ValueError):
        ehc.feed_poldict('??', 'rl')


# ----- Schema upgrade helpers ---------------------------------------------

def test_upgrade_tarr_from_legacy():
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


def test_upgrade_tarr_idempotent():
    a = np.zeros(1, dtype=ehc.DTARR)
    a['feed_type'] = 'rl'
    assert ehc.upgrade_tarr(a) is a


def test_upgrade_dtpol_circ_zero_copy_view():
    od = np.zeros(3, dtype=_LEGACY_DTPOL_CIRC)
    od['rrvis'] = [1 + 0j, 2 + 0j, 3 + 0j]
    nd = ehc.upgrade_dtpol_circ(od)
    # view-cast: same underlying buffer
    assert nd.base is od
    # both legacy and generic accessors work
    assert np.array_equal(nd['rrvis'], od['rrvis'])
    assert np.array_equal(nd['p1p1vis'], od['rrvis'])


def test_upgrade_dtpol_circ_idempotent():
    d = np.zeros(1, dtype=ehc.DTPOL_CIRC)
    out = ehc.upgrade_dtpol_circ(d)
    assert out.dtype == d.dtype


def test_upgrade_dtcal_circ_from_legacy_adds_dterm_fields():
    oc = np.zeros(2, dtype=_LEGACY_DTCAL)
    oc['rscale'] = [1 + 0j, 2 + 0j]
    nc = ehc.upgrade_dtcal_circ(oc)
    assert nc.dtype.names == ('time', 'rscale', 'lscale', 'dr', 'dl')
    assert np.array_equal(nc['rscale'], oc['rscale'])
    # dr/dl default to 0
    assert np.all(nc['dr'] == 0)
    assert np.all(nc['dl'] == 0)


# ----- End-to-end: Array / Obsdata / Caltable upgrade through __init__ ----

def test_array_upgrades_legacy_tarr():
    arr = ea.Array(_legacy_tarr())
    assert 'feed_type' in arr.tarr.dtype.names
    assert np.all(arr.tarr['feed_type'] == 'rl')


def test_array_pickle_roundtrip_from_new():
    arr = ea.Array(_legacy_tarr())
    arr2 = pickle.loads(pickle.dumps(arr))
    assert np.all(arr2.tarr['feed_type'] == 'rl')
    assert np.array_equal(arr2.tarr['sefd_p1'], arr.tarr['sefd_p1'])


def test_array_setstate_upgrades_legacy_pickle():
    # Simulate a pickle made before the schema migration:
    legacy_state = {'_tarr': _legacy_tarr(), 'ephem': {},
                    'tkey': {'A': 0, 'B': 1}}
    arr = ea.Array.__new__(ea.Array)
    arr.__setstate__(legacy_state)
    assert 'feed_type' in arr.tarr.dtype.names
    assert np.all(arr.tarr['feed_type'] == 'rl')


def test_obsdata_upgrades_legacy_datatable_and_tarr():
    obs = eo.Obsdata(ra=0., dec=0., rf=230e9, bw=1e9,
                     datatable=_legacy_circ_datatable(),
                     tarr=_legacy_tarr(), polrep='circ')
    assert obs.polrep == 'circ'
    assert 'feed_type' in obs.tarr.dtype.names
    assert np.all(obs.tarr['feed_type'] == 'rl')
    # Generic accessor works on upgraded data
    assert np.array_equal(obs.data['p1p1vis'], obs.data['rrvis'])


def test_obsdata_pickle_roundtrip_legacy_recarrays():
    obs = eo.Obsdata(ra=0., dec=0., rf=230e9, bw=1e9,
                     datatable=_legacy_circ_datatable(),
                     tarr=_legacy_tarr(), polrep='circ')
    obs2 = pickle.loads(pickle.dumps(obs))
    assert np.all(obs2.tarr['feed_type'] == 'rl')
    assert np.array_equal(obs2.data['p1p1vis'], obs.data['rrvis'])


def test_caltable_upgrades_legacy_dtcal():
    oc = np.zeros(2, dtype=_LEGACY_DTCAL)
    oc['rscale'] = [1 + 0j, 1.1 + 0j]
    caltab = ec.Caltable(ra=0., dec=0., rf=230e9, bw=1e9,
                         datadict={'A': oc.copy(), 'B': oc.copy()},
                         tarr=_legacy_tarr())
    for site in ('A', 'B'):
        assert 'dr' in caltab.data[site].dtype.names
        assert 'd_p1' in caltab.data[site].dtype.fields  # title alias
        assert np.all(caltab.data[site]['dr'] == 0)


def test_caltable_pickle_roundtrip_legacy_dtcal():
    oc = np.zeros(2, dtype=_LEGACY_DTCAL)
    oc['rscale'] = [1 + 0j, 1.1 + 0j]
    caltab = ec.Caltable(ra=0., dec=0., rf=230e9, bw=1e9,
                         datadict={'A': oc.copy()}, tarr=_legacy_tarr())
    caltab2 = pickle.loads(pickle.dumps(caltab))
    assert 'dr' in caltab2.data['A'].dtype.names
    assert np.array_equal(caltab2.data['A']['rscale'],
                          caltab.data['A']['rscale'])


# ============================================================================
# Array class migration
# ============================================================================
#
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

def test_is_homogeneous_feeds_legacy_array_true():
    arr = ea.Array(_legacy_tarr())
    assert arr.is_homogeneous_feeds() is True


def test_is_homogeneous_feeds_xy_array_true():
    arr = ea.Array(_xy_tarr())
    assert arr.is_homogeneous_feeds() is True


def test_is_homogeneous_feeds_mixed_array_false():
    arr = ea.Array(_mixed_tarr())
    assert arr.is_homogeneous_feeds() is False


def test_feed_types_legacy_array():
    arr = ea.Array(_legacy_tarr())
    assert arr.feed_types() == {'rl'}


def test_feed_types_mixed_array():
    arr = ea.Array(_mixed_tarr())
    assert arr.feed_types() == {'rl', 'xy'}


def test_sefd_for_feed_rl_station_returns_p1_p2():
    arr = ea.Array(_mixed_tarr())
    assert arr.sefd_for_feed('APEX', 'R') == 4000.
    assert arr.sefd_for_feed('APEX', 'L') == 4100.


def test_sefd_for_feed_xy_station_returns_p1_p2():
    arr = ea.Array(_mixed_tarr())
    assert arr.sefd_for_feed('ALMA', 'X') == 100.
    assert arr.sefd_for_feed('ALMA', 'Y') == 110.


def test_sefd_for_feed_case_insensitive():
    arr = ea.Array(_mixed_tarr())
    assert arr.sefd_for_feed('ALMA', 'x') == arr.sefd_for_feed('ALMA', 'X')


def test_sefd_for_feed_wrong_feed_raises():
    arr = ea.Array(_mixed_tarr())
    # ALMA is 'xy'; asking for R should raise.
    with pytest.raises(ValueError, match="feed 'R' not in feed_type 'xy'"):
        arr.sefd_for_feed('ALMA', 'R')


def test_sefd_for_feed_unknown_site_raises():
    arr = ea.Array(_mixed_tarr())
    with pytest.raises(KeyError, match="site 'NOSITE' not in array"):
        arr.sefd_for_feed('NOSITE', 'R')


def test_dterm_for_feed_rl_station_dispatch():
    arr = ea.Array(_mixed_tarr())
    assert arr.dterm_for_feed('APEX', 'R') == 0.03 + 0.04j
    assert arr.dterm_for_feed('APEX', 'L') == 0.07 + 0.08j


def test_dterm_for_feed_xy_station_dispatch():
    arr = ea.Array(_mixed_tarr())
    assert arr.dterm_for_feed('ALMA', 'X') == 0.01 + 0.02j
    assert arr.dterm_for_feed('ALMA', 'Y') == 0.05 + 0.06j


def test_dterm_for_feed_wrong_feed_raises():
    arr = ea.Array(_mixed_tarr())
    with pytest.raises(ValueError, match="not in feed_type"):
        arr.dterm_for_feed('APEX', 'X')


# ----- add_site signature extension -----------------------------------------

def _empty_array():
    return ea.Array(np.zeros(0, dtype=ehc.DTARR))


def test_add_site_default_kwargs_bit_identical_to_legacy():
    arr = _empty_array().add_site('A', (1.0, 2.0, 3.0))
    row = arr.tarr[arr.tkey['A']]
    assert row['sefd_p1'] == 10000.0
    assert row['sefd_p2'] == 10000.0
    assert row['d_p1'] == 0 + 0j
    assert row['d_p2'] == 0 + 0j
    assert str(row['feed_type']) == 'rl'


def test_add_site_legacy_sefd_and_dr_dl_still_work():
    arr = _empty_array().add_site('A', (1., 2., 3.),
                                  sefd=5000, dr=0.1 + 0.2j, dl=0.3 + 0.4j)
    row = arr.tarr[arr.tkey['A']]
    assert row['sefd_p1'] == 5000.0
    assert row['sefd_p2'] == 5000.0
    assert row['d_p1'] == 0.1 + 0.2j
    assert row['d_p2'] == 0.3 + 0.4j
    assert str(row['feed_type']) == 'rl'


def test_add_site_with_feed_type_xy_and_generic_sefds():
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


def test_add_site_dr_with_non_rl_feed_raises():
    with pytest.raises(ValueError, match="dr/dl kwargs are only valid"):
        _empty_array().add_site('A', (0., 0., 0.),
                                feed_type='xy', dr=0.1 + 0j)


def test_add_site_dl_with_non_rl_feed_raises():
    with pytest.raises(ValueError, match="dr/dl kwargs are only valid"):
        _empty_array().add_site('A', (0., 0., 0.),
                                feed_type='xy', dl=0.1 + 0j)


def test_add_site_dr_and_d_p1_both_raises():
    with pytest.raises(ValueError, match="dr.*d_p1"):
        _empty_array().add_site('A', (0., 0., 0.),
                                dr=0.1 + 0j, d_p1=0.2 + 0j)


def test_add_site_dl_and_d_p2_both_raises():
    with pytest.raises(ValueError, match="dl.*d_p2"):
        _empty_array().add_site('A', (0., 0., 0.),
                                dl=0.1 + 0j, d_p2=0.2 + 0j)


def test_add_site_sefd_and_sefd_p1_both_raises():
    with pytest.raises(ValueError, match="sefd.*sefd_p1"):
        _empty_array().add_site('A', (0., 0., 0.),
                                sefd=5000, sefd_p1=100., sefd_p2=110.)


def test_add_site_sefd_p1_without_p2_raises():
    with pytest.raises(ValueError, match="sefd_p1 and sefd_p2"):
        _empty_array().add_site('A', (0., 0., 0.),
                                feed_type='xy', sefd_p1=100.)


def test_add_site_invalid_feed_type_raises():
    with pytest.raises(ValueError, match="feed_type must be one of"):
        _empty_array().add_site('A', (0., 0., 0.), feed_type='zz')


# ----- add_satellite_* signature extension ----------------------------------

def test_add_satellite_tle_default_feed_type_rl():
    tle = ['SAT1', 'line1', 'line2']
    arr = _empty_array().add_satellite_tle(tle, sefd=10000)
    row = arr.tarr[arr.tkey['SAT1']]
    assert str(row['feed_type']) == 'rl'
    assert row['sefd_p1'] == 10000.0
    assert row['sefd_p2'] == 10000.0


def test_add_satellite_tle_with_feed_type_xy():
    tle = ['SAT1', 'line1', 'line2']
    arr = _empty_array().add_satellite_tle(tle, feed_type='xy',
                                            sefd_p1=500., sefd_p2=600.)
    row = arr.tarr[arr.tkey['SAT1']]
    assert str(row['feed_type']) == 'xy'
    assert row['sefd_p1'] == 500.0
    assert row['sefd_p2'] == 600.0


def test_add_satellite_elements_default_feed_type_rl():
    arr = _empty_array().add_satellite_elements('SAT2', sefd=2000)
    row = arr.tarr[arr.tkey['SAT2']]
    assert str(row['feed_type']) == 'rl'
    assert row['sefd_p1'] == 2000.0


def test_add_satellite_elements_with_feed_type_xy_and_dterms():
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


def test_add_satellite_tle_invalid_feed_type_raises():
    tle = ['SAT1', 'line1', 'line2']
    with pytest.raises(ValueError, match="feed_type must be one of"):
        _empty_array().add_satellite_tle(tle, feed_type='qq')


# ----- TarrView wrapper -----------------------------------------------------

def test_tarrview_is_returned_by_tarr_property():
    arr = ea.Array(_legacy_tarr())
    assert isinstance(arr.tarr, ea.TarrView)


def test_tarrview_sefdr_on_homogeneous_rl_works():
    arr = ea.Array(_legacy_tarr())
    # Legacy title-alias access still works on an all-RL array.
    np.testing.assert_array_equal(arr.tarr['sefdr'], [1000., 2000.])
    np.testing.assert_array_equal(arr.tarr['sefdl'], [1100., 2100.])


def test_tarrview_sefdr_on_xy_array_raises():
    arr = ea.Array(_xy_tarr())
    with pytest.raises(KeyError, match=r"tarr\['sefdr'\].*xy"):
        _ = arr.tarr['sefdr']


def test_tarrview_sefdl_on_xy_array_raises():
    arr = ea.Array(_xy_tarr())
    with pytest.raises(KeyError, match=r"tarr\['sefdl'\].*xy"):
        _ = arr.tarr['sefdl']


def test_tarrview_sefdr_on_mixed_array_raises():
    arr = ea.Array(_mixed_tarr())
    with pytest.raises(KeyError, match=r"tarr\['sefdr'\].*"):
        _ = arr.tarr['sefdr']


def test_tarrview_dr_on_mixed_array_raises():
    arr = ea.Array(_mixed_tarr())
    with pytest.raises(KeyError, match="dterm_for_feed"):
        _ = arr.tarr['dr']


def test_tarrview_generic_names_always_work():
    # Generic per-feed names are never guarded.
    arr = ea.Array(_mixed_tarr())
    np.testing.assert_array_equal(arr.tarr['sefd_p1'], [100., 4000.])
    np.testing.assert_array_equal(arr.tarr['sefd_p2'], [110., 4100.])


def test_tarrview_dtype_attribute_forwarded():
    arr = ea.Array(_legacy_tarr())
    assert 'feed_type' in arr.tarr.dtype.names
    assert arr.tarr.dtype == ehc.DTARR


def test_tarrview_len_iter_forwarded():
    arr = ea.Array(_legacy_tarr())
    assert len(arr.tarr) == 2
    sites = [str(row['site']) for row in arr.tarr]
    assert sites == ['A', 'B']


def test_tarrview_row_index_returns_void():
    arr = ea.Array(_legacy_tarr())
    row = arr.tarr[0]
    # Single-row index returns a numpy void; field access on it bypasses
    # the guard (documented limitation).
    assert str(row['site']) == 'A'


def test_tarrview_boolean_mask_returns_wrapped_view():
    arr = ea.Array(_mixed_tarr())
    mask = np.array([True, False])
    sub = arr.tarr[mask]
    assert isinstance(sub, ea.TarrView)
    assert len(sub) == 1
    assert str(sub[0]['site']) == 'ALMA'


def test_tarrview_boolean_mask_preserves_guard():
    # Slicing a mixed array down to one xy station: legacy 'sefdr'
    # still raises on the subview (feed_types still non-RL).
    arr = ea.Array(_mixed_tarr())
    sub = arr.tarr[np.array([True, False])]
    with pytest.raises(KeyError):
        _ = sub['sefdr']


def test_tarrview_equality_recarray_lhs():
    arr = ea.Array(_legacy_tarr())
    raw = arr.tarr._tarr.copy()
    cmp = raw == arr.tarr
    assert np.all(cmp)


def test_tarrview_equality_recarray_rhs():
    arr = ea.Array(_legacy_tarr())
    raw = arr.tarr._tarr.copy()
    cmp = arr.tarr == raw
    assert np.all(cmp)


def test_tarrview_equality_both_views():
    arr1 = ea.Array(_legacy_tarr())
    arr2 = ea.Array(_legacy_tarr())
    assert np.all(arr1.tarr == arr2.tarr)


def test_tarrview_pickle_roundtrip():
    arr = ea.Array(_legacy_tarr())
    view = arr.tarr
    view2 = pickle.loads(pickle.dumps(view))
    assert isinstance(view2, ea.TarrView)
    assert np.all(view == view2)


def test_tarrview_unhashable():
    arr = ea.Array(_legacy_tarr())
    with pytest.raises(TypeError):
        hash(arr.tarr)


def test_tarrview_numpy_interop_via_array_protocol():
    arr = ea.Array(_legacy_tarr())
    # np.asarray should produce a recarray view (no copy required by spec).
    asarr = np.asarray(arr.tarr)
    assert isinstance(asarr, np.ndarray)
    assert asarr.dtype.names == arr.tarr.dtype.names


def test_tarrview_setter_unwraps_view():
    # Assigning a TarrView via the property setter unwraps it so _tarr
    # remains a plain ndarray (the storage truth).
    arr = ea.Array(_legacy_tarr())
    view = arr.tarr  # TarrView
    arr.tarr = view  # round-trip via setter
    assert not isinstance(arr._tarr, ea.TarrView)
    assert isinstance(arr._tarr, np.ndarray)
    assert arr._tarr.dtype.names is not None  # structured dtype preserved


def test_tarrview_array_pickle_roundtrip_preserves_ndarray_storage():
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


def test_obsdata_polrep_invalid_raises():
    arr = ea.Array(_eht_like_rl_array())
    with pytest.raises(ValueError, match="polrep must be one of"):
        arr.obsdata(polrep='bogus', **_obs_kwargs())


def test_obsdata_polrep_stokes_on_rl_array_succeeds():
    arr = ea.Array(_eht_like_rl_array())
    obs = arr.obsdata(polrep='stokes', **_obs_kwargs())
    assert obs.polrep == 'stokes'


def test_obsdata_polrep_circ_on_rl_array_succeeds():
    arr = ea.Array(_eht_like_rl_array())
    obs = arr.obsdata(polrep='circ', **_obs_kwargs())
    assert obs.polrep == 'circ'


def test_obsdata_polrep_circ_on_xy_array_raises():
    arr = ea.Array(_eht_like_xy_array())
    with pytest.raises(ValueError,
                       match="polrep='circ' requires all stations.*'rl'"):
        arr.obsdata(polrep='circ', **_obs_kwargs())


def test_obsdata_polrep_circ_on_mixed_array_raises():
    arr = ea.Array(_eht_like_mixed_array())
    with pytest.raises(ValueError, match="polrep='circ'"):
        arr.obsdata(polrep='circ', **_obs_kwargs())


def test_obsdata_polrep_lin_on_rl_array_raises():
    arr = ea.Array(_eht_like_rl_array())
    with pytest.raises(ValueError,
                       match="polrep='lin' requires all stations.*'xy'"):
        arr.obsdata(polrep='lin', **_obs_kwargs())


def test_obsdata_polrep_lin_on_xy_array_raises_not_implemented():
    # Validation passes (all xy) but simulation backend doesn't support
    # 'lin' yet — should raise NotImplementedError pointing to Phase 5.
    arr = ea.Array(_eht_like_xy_array())
    with pytest.raises(NotImplementedError, match="is not yet supported"):
        arr.obsdata(polrep='lin', **_obs_kwargs())


def test_obsdata_polrep_mixed_on_homogeneous_array_raises():
    arr = ea.Array(_eht_like_rl_array())
    with pytest.raises(ValueError,
                       match="polrep='mixed' requires at least two"):
        arr.obsdata(polrep='mixed', **_obs_kwargs())


def test_obsdata_polrep_mixed_on_mixed_array_raises_not_implemented():
    arr = ea.Array(_eht_like_mixed_array())
    with pytest.raises(NotImplementedError, match="is not yet supported"):
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


def test_save_array_txt_emits_v2_header(tmp_path):
    arr = ea.Array(_full_tarr(['rl', 'rl']))
    fname = str(tmp_path / 'a.txt')
    arr.save_txt(fname)
    with open(fname) as f:
        first = f.readline().strip()
    assert first == '# ehtim array format v2'


def test_save_load_array_txt_roundtrip_rl(tmp_path):
    arr = ea.Array(_full_tarr(['rl', 'rl']))
    fname = str(tmp_path / 'a.txt')
    arr.save_txt(fname)
    arr2 = ea.load_txt(fname)
    _assert_tarr_round_trip(arr, arr2)
    assert set(arr2.feed_types()) == {'rl'}


def test_save_load_array_txt_roundtrip_xy(tmp_path):
    arr = ea.Array(_full_tarr(['xy', 'xy', 'xy']))
    fname = str(tmp_path / 'a.txt')
    arr.save_txt(fname)
    arr2 = ea.load_txt(fname)
    _assert_tarr_round_trip(arr, arr2)
    assert set(arr2.feed_types()) == {'xy'}


def test_save_load_array_txt_roundtrip_mixed(tmp_path):
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
def test_load_array_txt_legacy_bit_identical(fname):
    """For every existing arrays/*.txt file, the loaded tarr's numeric
    columns match a fresh np.loadtxt read of the same file byte-for-byte
    (mixpol added a feed_type column; nothing else may differ).
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


# ----- save_obs_txt / load_obs_txt embedded-tarr round-trip -----------------

def test_save_load_obs_txt_embedded_tarr_with_feed_type(tmp_path):
    """The obs.txt embedded tarr block now emits feed_type; round-trip
    must preserve it for non-trivial values."""
    tarr = _full_tarr(['rl', 'xy', 'lx'])
    data = np.zeros(2, dtype=ehc.DTPOL_STOKES)
    data['t1'] = ['S0', 'S0']
    data['t2'] = ['S1', 'S2']
    data['u'] = [1e9, 2e9]
    data['v'] = [1e8, 2e8]
    data['vis'] = [1 + 0j, 0.5 + 0j]
    data['sigma'] = [0.1, 0.1]
    obs = eo.Obsdata(ra=17.76, dec=-29., rf=230e9, bw=1e9,
                     datatable=data, tarr=tarr, polrep='stokes')
    fname = str(tmp_path / 'obs.txt')
    obs.save_txt(fname)
    obs2 = eo.load_txt(fname, polrep='stokes')
    np.testing.assert_array_equal(obs2.tarr['feed_type'], obs.tarr['feed_type'])
    np.testing.assert_array_equal(obs2.tarr['site'], obs.tarr['site'])
    np.testing.assert_allclose(obs2.tarr['sefd_p1'], obs.tarr['sefd_p1'], atol=1e-2)
    np.testing.assert_allclose(obs2.tarr['sefd_p2'], obs.tarr['sefd_p2'], atol=1e-2)


# ============================================================================
# Obsdata polrep enum + dtype dispatch
# ============================================================================


# ----- Helpers --------------------------------------------------------------

def _tarr(feed_types, sites=None):
    """Minimal valid tarr with the given feed_type list (one per station)."""
    n = len(feed_types)
    if sites is None:
        sites = [f'S{i}' for i in range(n)]
    t = np.zeros(n, dtype=ehc.DTARR)
    t['site'] = sites
    t['feed_type'] = feed_types
    return t


def _lin_data(t1s, t2s, times=None):
    """DTPOL_LIN datatable with non-trivial visibility values."""
    n = len(t1s)
    d = np.zeros(n, dtype=ehc.DTPOL_LIN)
    d['time'] = times if times is not None else np.zeros(n)
    d['t1'] = t1s
    d['t2'] = t2s
    d['u'] = 1e6 * (np.arange(n) + 1)
    d['v'] = 1e5 * (np.arange(n) + 1)
    d['xxvis'] = (1 + 2j) * (np.arange(n) + 1)
    d['yyvis'] = (3 + 4j) * (np.arange(n) + 1)
    d['xyvis'] = (5 + 6j) * (np.arange(n) + 1)
    d['yxvis'] = (7 + 8j) * (np.arange(n) + 1)
    d['xxsigma'] = 0.1 * (np.arange(n) + 1)
    d['yysigma'] = 0.2 * (np.arange(n) + 1)
    d['xysigma'] = 0.3 * (np.arange(n) + 1)
    d['yxsigma'] = 0.4 * (np.arange(n) + 1)
    return d


def _mixed_data(t1s, t2s, times=None):
    """DTPOL_MIXED datatable; polbasis is left empty (populated in __init__)."""
    n = len(t1s)
    d = np.zeros(n, dtype=ehc.DTPOL_MIXED)
    d['time'] = times if times is not None else np.zeros(n)
    d['t1'] = t1s
    d['t2'] = t2s
    d['u'] = 1e6 * (np.arange(n) + 1)
    d['v'] = 1e5 * (np.arange(n) + 1)
    d['p1p1vis'] = (1 + 2j) * (np.arange(n) + 1)
    d['p2p2vis'] = (3 + 4j) * (np.arange(n) + 1)
    d['p1p2vis'] = (5 + 6j) * (np.arange(n) + 1)
    d['p2p1vis'] = (7 + 8j) * (np.arange(n) + 1)
    d['p1p1sigma'] = 0.1 * (np.arange(n) + 1)
    d['p2p2sigma'] = 0.2 * (np.arange(n) + 1)
    d['p1p2sigma'] = 0.3 * (np.arange(n) + 1)
    d['p2p1sigma'] = 0.4 * (np.arange(n) + 1)
    return d


def _lin_obs(t1s=('S0',), t2s=('S1',)):
    return eo.Obsdata(0., 0., 230e9, 1e9,
                      _lin_data(list(t1s), list(t2s)),
                      _tarr(['xy', 'xy']),
                      polrep='lin')


def _mixed_obs():
    """3-station mixed array: S0/S1 are 'rl', S2 is 'xy'. 3 baselines."""
    return eo.Obsdata(0., 0., 230e9, 1e9,
                      _mixed_data(['S0', 'S0', 'S1'],
                                          ['S1', 'S2', 'S2']),
                      _tarr(['rl', 'rl', 'xy']),
                      polrep='mixed')


# ----- Construction & validation --------------------------------------------

def test_init_lin_from_synthetic_table():
    obs = _lin_obs()
    assert obs.polrep == 'lin'
    assert obs.poltype is ehc.DTPOL_LIN
    assert obs.poldict is ehc.POLDICT_LIN
    np.testing.assert_array_equal(obs.data['p1p1vis'], obs.data['xxvis'])
    np.testing.assert_array_equal(obs.data['p2p1vis'], obs.data['yxvis'])


def test_init_mixed_from_synthetic_table():
    obs = _mixed_obs()
    assert obs.polrep == 'mixed'
    assert obs.poltype is ehc.DTPOL_MIXED
    assert obs.poldict is ehc.POLDICT_MIXED
    # polbasis is the lossless 4-char t1+t2 feed_type concat:
    # (S0,S1) -> rlrl, (S0,S2) -> rlxy, (S1,S2) -> rlxy
    assert set(obs.data['polbasis']) == {'rlrl', 'rlxy'}


def test_init_circ_warns_on_non_rl_tarr():
    d = np.zeros(1, dtype=ehc.DTPOL_CIRC)
    d['t1'] = 'S0'
    d['t2'] = 'S1'
    tarr = _tarr(['rl', 'xy'])
    with pytest.warns(UserWarning, match="polrep='circ' Obsdata constructed on a tarr"):
        eo.Obsdata(0., 0., 230e9, 1e9, d, tarr, polrep='circ')


def test_init_lin_warns_on_non_xy_tarr():
    d = _lin_data(['S0'], ['S1'])
    tarr = _tarr(['rl', 'rl'])
    with pytest.warns(UserWarning, match="polrep='lin' Obsdata constructed on a tarr"):
        eo.Obsdata(0., 0., 230e9, 1e9, d, tarr, polrep='lin')


def test_init_mixed_rejects_homogeneous_tarr():
    d = _mixed_data(['S0'], ['S1'])
    tarr = _tarr(['rl', 'rl'])
    with pytest.raises(ValueError, match="polrep='mixed' requires at least two distinct feed types"):
        eo.Obsdata(0., 0., 230e9, 1e9, d, tarr, polrep='mixed')


def test_init_rejects_unset_feed_type_qq():
    d = np.zeros(1, dtype=ehc.DTPOL_STOKES)
    d['t1'] = 'S0'
    d['t2'] = 'S1'
    tarr = _tarr(['??', '??'])
    with pytest.raises(ValueError, match="unset feed_type='\\?\\?'"):
        eo.Obsdata(0., 0., 230e9, 1e9, d, tarr, polrep='stokes')


def test_init_rejects_unknown_feed_type():
    d = np.zeros(1, dtype=ehc.DTPOL_STOKES)
    d['t1'] = 'S0'
    d['t2'] = 'S1'
    tarr = _tarr(['rl', 'zz'])
    with pytest.raises(ValueError, match="unknown feed_type"):
        eo.Obsdata(0., 0., 230e9, 1e9, d, tarr, polrep='stokes')


def test_init_mixed_rejects_missing_station():
    d = _mixed_data(['SX'], ['S1'])
    tarr = _tarr(['rl', 'xy'])
    with pytest.raises(ValueError, match="data references stations not in tarr"):
        eo.Obsdata(0., 0., 230e9, 1e9, d, tarr, polrep='mixed')


def test_dtype_polrep_mismatch_raises():
    d = np.zeros(1, dtype=ehc.DTPOL_STOKES)
    d['t1'] = 'S0'
    d['t2'] = 'S1'
    tarr = _tarr(['xy', 'xy'])
    with pytest.raises(Exception, match="does not match polrep='lin'"):
        eo.Obsdata(0., 0., 230e9, 1e9, d, tarr, polrep='lin')


# ----- Round-trip & query methods -------------------------------------------

def test_copy_roundtrip_lin():
    obs = _lin_obs()
    other = obs.copy()
    assert other.polrep == 'lin'
    assert other.poltype == ehc.DTPOL_LIN
    np.testing.assert_array_equal(other.data['xxvis'], obs.data['xxvis'])


def test_copy_roundtrip_mixed():
    obs = _mixed_obs()
    other = obs.copy()
    assert other.polrep == 'mixed'
    assert other.poltype == ehc.DTPOL_MIXED
    np.testing.assert_array_equal(other.data['polbasis'], obs.data['polbasis'])
    np.testing.assert_array_equal(other.data['p1p1vis'], obs.data['p1p1vis'])


def test_tlist_bllist_lin():
    obs = eo.Obsdata(0., 0., 230e9, 1e9,
                     _lin_data(['S0', 'S0'], ['S1', 'S1'],
                                       times=[0.0, 1.0]),
                     _tarr(['xy', 'xy']),
                     polrep='lin')
    tl = obs.tlist()
    bl = obs.bllist()
    assert len(tl) == 2
    assert len(bl) == 1


def test_tlist_bllist_mixed():
    obs = _mixed_obs()
    tl = obs.tlist()
    bl = obs.bllist()
    assert len(tl) == 1
    assert len(bl) == 3


def test_flag_uvdist_lin():
    obs = _lin_obs()
    kept = obs.flag_uvdist(uv_min=0., uv_max=1e20)
    assert len(kept.data) == len(obs.data)


def test_flag_sites_mixed():
    obs = _mixed_obs()
    kept = obs.flag_sites(['S2'])
    assert 'S2' not in set(kept.data['t1']) | set(kept.data['t2'])


# ----- reorder_baselines ----------------------------------------------------

def test_reorder_baselines_lin_swap():
    # t1=S1, t2=S0 is reversed; reorder swaps and applies LIN conj/cross-swap.
    d = _lin_data(['S1'], ['S0'])
    obs = eo.Obsdata(0., 0., 230e9, 1e9, d, _tarr(['xy', 'xy']),
                     polrep='lin')
    row = obs.data[0]
    assert row['t1'] == 'S0'
    assert row['t2'] == 'S1'
    assert row['u'] == -1e6
    assert row['xxvis'] == np.conj(1 + 2j)
    assert row['yyvis'] == np.conj(3 + 4j)
    # cross: new xyvis = conj(old yxvis); new yxvis = conj(old xyvis)
    assert row['xyvis'] == np.conj(7 + 8j)
    assert row['yxvis'] == np.conj(5 + 6j)
    assert row['xysigma'] == 0.4
    assert row['yxsigma'] == 0.3


def test_reorder_baselines_mixed_swap_and_polbasis_flip():
    # t1=S1 (xy), t2=S0 (rl); reorder swaps and flips polbasis 'xyrl' -> 'rlxy'.
    d = _mixed_data(['S1'], ['S0'])
    obs = eo.Obsdata(0., 0., 230e9, 1e9, d,
                     _tarr(['rl', 'xy']),
                     polrep='mixed')
    row = obs.data[0]
    assert row['t1'] == 'S0'
    assert row['t2'] == 'S1'
    assert row['u'] == -1e6
    assert row['polbasis'] == 'rlxy'
    assert row['p1p1vis'] == np.conj(1 + 2j)
    assert row['p2p2vis'] == np.conj(3 + 4j)
    assert row['p1p2vis'] == np.conj(7 + 8j)
    assert row['p2p1vis'] == np.conj(5 + 6j)
    assert row['p1p2sigma'] == 0.4
    assert row['p2p1sigma'] == 0.3


# ----- avg_coherent / avg_incoherent guards ---------------------------------

def test_avg_coherent_lin_raises():
    obs = _lin_obs()
    with pytest.raises(NotImplementedError, match="not yet supported on polrep='lin'"):
        obs.avg_coherent(60.0)


def test_avg_coherent_mixed_raises():
    obs = _mixed_obs()
    with pytest.raises(NotImplementedError, match="not yet supported on polrep='mixed'"):
        obs.avg_coherent(60.0)


def test_avg_incoherent_lin_raises():
    obs = _lin_obs()
    with pytest.raises(NotImplementedError, match="not yet supported on polrep='lin'"):
        obs.avg_incoherent(60.0)


def test_avg_incoherent_mixed_raises():
    obs = _mixed_obs()
    with pytest.raises(NotImplementedError, match="not yet supported on polrep='mixed'"):
        obs.avg_incoherent(60.0)


# ----- dirtyimage -----------------------------------------------------------

def test_dirtyimage_mixed_raises():
    obs = _mixed_obs()
    with pytest.raises(NotImplementedError, match="dirtyimage is not supported on polrep='mixed'"):
        obs.dirtyimage(npix=8, fov=1e-6)


# LIN dirtyimage smoke coverage is deferred: unpack_dat does not yet handle
# LIN sigma/Stokes-derived fields, so dirtyimage falls over before it can
# build the image. Will be exercised once unpack_dat gains the LIN branch.


# ============================================================================
#  Obsdata.switch_polrep extensions
# ============================================================================


# ----- Helpers --------------------------------------------------------------

def _stokes_obs(tarr_feeds=('rl', 'rl'), with_nan_v=False):
    """Stokes Obsdata on a 2-station tarr with the given feed_types."""
    tarr = _tarr(list(tarr_feeds))
    d = np.zeros(2, dtype=ehc.DTPOL_STOKES)
    d['time'] = [0.0, 0.5]
    d['t1'] = ['S0', 'S0']
    d['t2'] = ['S1', 'S1']
    d['u'] = [1e6, 2e6]
    d['v'] = [1e5, 2e5]
    d['vis'] = [1.0 + 0j, 1.2 + 0j]
    d['qvis'] = [0.3 + 0.1j, 0.4 + 0.1j]
    d['uvis'] = [0.2 - 0.1j, 0.25 - 0.1j]
    d['vvis'] = [0.05 + 0j, 0.06 + 0j] if not with_nan_v else [np.nan, np.nan]
    d['sigma'] = 0.01
    d['qsigma'] = 0.01
    d['usigma'] = 0.01
    d['vsigma'] = 0.01
    return eo.Obsdata(0., 0., 230e9, 1e9, d, tarr, polrep='stokes')


def _circ_obs():
    """A non-trivial circ Obsdata on an rl tarr."""
    return _stokes_obs().switch_polrep('circ')


def _lin_obs(with_nan_xx=False):
    """A non-trivial lin Obsdata on an xy tarr (built from stokes)."""
    obs_s = _stokes_obs(tarr_feeds=('xy', 'xy'))
    obs_l = obs_s.switch_polrep('lin', singlepol_hand='X')
    if with_nan_xx:
        obs_l.data['xxvis'] = np.nan
    return obs_l


# ----- Direct lin <-> stokes round-trips ------------------------------------

def test_switch_polrep_lin_stokes_lin_roundtrip():
    obs = _lin_obs()
    rt = obs.switch_polrep('stokes').switch_polrep('lin', singlepol_hand='X')
    for f in ('xxvis', 'yyvis', 'xyvis', 'yxvis'):
        np.testing.assert_allclose(rt.data[f], obs.data[f], atol=1e-12)


def test_switch_polrep_cross_convention_check():
    """Build matched circ + lin obs from the same Stokes; both -> stokes match."""
    obs_s = _stokes_obs()
    obs_c = obs_s.switch_polrep('circ')
    obs_l = obs_s.switch_polrep('lin', singlepol_hand='X')
    s_from_c = obs_c.switch_polrep('stokes')
    s_from_l = obs_l.switch_polrep('stokes')
    for f in ('vis', 'qvis', 'uvis', 'vvis'):
        np.testing.assert_allclose(s_from_c.data[f], s_from_l.data[f], atol=1e-12)


# ----- circ <-> lin composition ---------------------------------------------

def test_switch_polrep_circ_to_lin_equals_explicit():
    obs_c = _circ_obs()
    composed = obs_c.switch_polrep('lin', singlepol_hand='X')
    explicit = obs_c.switch_polrep('stokes').switch_polrep('lin', singlepol_hand='X')
    for f in ('xxvis', 'yyvis', 'xyvis', 'yxvis'):
        np.testing.assert_allclose(composed.data[f], explicit.data[f], atol=1e-12)


def test_switch_polrep_lin_to_circ_equals_explicit():
    obs_l = _lin_obs()
    composed = obs_l.switch_polrep('circ', singlepol_hand='R')
    explicit = obs_l.switch_polrep('stokes').switch_polrep('circ', singlepol_hand='R')
    for f in ('rrvis', 'llvis', 'rlvis', 'lrvis'):
        np.testing.assert_allclose(composed.data[f], explicit.data[f], atol=1e-12)


def test_switch_polrep_circ_lin_circ_roundtrip():
    obs_c = _circ_obs()
    rt = obs_c.switch_polrep('lin', singlepol_hand='X').switch_polrep('circ', singlepol_hand='R')
    for f in ('rrvis', 'llvis', 'rlvis', 'lrvis'):
        np.testing.assert_allclose(rt.data[f], obs_c.data[f], atol=1e-12)


# ----- Mixed source / invalid output ----------------------------------------

def test_switch_polrep_mixed_source_raises():
    obs_m = _mixed_obs()
    with pytest.raises(NotImplementedError, match="polrep='mixed' is not yet implemented"):
        obs_m.switch_polrep('stokes')


def test_switch_polrep_invalid_output_raises():
    obs_c = _circ_obs()
    with pytest.raises(Exception, match="polrep_out must be"):
        obs_c.switch_polrep('mixed')
    with pytest.raises(Exception, match="polrep_out must be"):
        obs_c.switch_polrep('bogus')


# ----- allow_singlepol on lin -> stokes -------------------------------------

def test_switch_polrep_allow_singlepol_lin_to_stokes_fills_xx_with_yy():
    obs = _lin_obs(with_nan_xx=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = obs.switch_polrep('stokes', allow_singlepol=True)
    # vis should equal yyvis where xxvis was NaN (the surviving parallel-hand)
    np.testing.assert_allclose(out.data['vis'], obs.data['yyvis'], atol=1e-12)


def test_switch_polrep_allow_singlepol_false_leaves_nan():
    obs = _lin_obs(with_nan_xx=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = obs.switch_polrep('stokes', allow_singlepol=False)
    assert np.all(np.isnan(out.data['vis']))


# ----- singlepol warning behaviour ------------------------------------------

def test_switch_polrep_singlepol_warns_on_substitution():
    obs = _stokes_obs(with_nan_v=True)
    with pytest.warns(UserWarning, match="allow_singlepol substituted"):
        obs.switch_polrep('circ', singlepol_hand='R')


def test_switch_polrep_singlepol_no_warn_when_disabled():
    obs = _stokes_obs(with_nan_v=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        obs.switch_polrep('circ', allow_singlepol=False, singlepol_hand='R')
        assert not any('allow_singlepol substituted' in str(wi.message) for wi in w)


def test_switch_polrep_singlepol_no_warn_when_no_substitution():
    obs = _stokes_obs(with_nan_v=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        obs.switch_polrep('circ', allow_singlepol=True, singlepol_hand='R')
        assert not any('allow_singlepol substituted' in str(wi.message) for wi in w)


# ----- singlepol_hand validation --------------------------------------------

def test_switch_polrep_singlepol_hand_X_invalid_for_circ():
    obs = _stokes_obs(with_nan_v=True)
    with pytest.raises(Exception, match="singlepol_hand must be 'R' or 'L'"):
        obs.switch_polrep('circ', singlepol_hand='X')


def test_switch_polrep_singlepol_hand_R_invalid_for_lin():
    obs = _stokes_obs(with_nan_v=True)
    with pytest.raises(Exception, match="singlepol_hand must be 'X' or 'Y'"):
        obs.switch_polrep('lin', singlepol_hand='R')


def test_switch_polrep_singlepol_hand_case_insensitive():
    obs = _stokes_obs(with_nan_v=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out_r = obs.switch_polrep('circ', singlepol_hand='r')
        out_x = obs.switch_polrep('lin', singlepol_hand='x')
    # lowercase accepted -- xxvis (lin) / rrvis (circ) should hold the Stokes I fill
    assert not np.all(np.isnan(out_r.data['rrvis']))
    assert not np.all(np.isnan(out_x.data['xxvis']))


def test_switch_polrep_singlepol_hand_unused_when_disabled():
    """When allow_singlepol=False, singlepol_hand is not validated."""
    obs = _stokes_obs(with_nan_v=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # 'X' would be invalid for circ if validated; with allow_singlepol=False
        # it's never used and never validated.
        obs.switch_polrep('circ', allow_singlepol=False, singlepol_hand='X')


# ----- tarr preservation ----------------------------------------------------

def test_switch_polrep_tarr_unchanged():
    obs_c = _circ_obs()
    obs_l = obs_c.switch_polrep('lin', singlepol_hand='X')
    # tarr.feed_type is physical-array description; switch_polrep does not
    # touch it.
    np.testing.assert_array_equal(obs_l.tarr['feed_type'], obs_c.tarr['feed_type'])


# ============================================================================
#  data_conj — conjugate-baseline generalization (lin / mixed)
# ============================================================================


def _circ_data_1bl():
    """Single-baseline (S0, S1) DTPOL_CIRC table with known values."""
    d = np.zeros(1, dtype=ehc.DTPOL_CIRC)
    d['t1'] = 'S0'
    d['t2'] = 'S1'
    d['u'] = 1e6
    d['v'] = 1e5
    d['rrvis'] = 1 + 2j
    d['llvis'] = 3 + 4j
    d['rlvis'] = 5 + 6j
    d['lrvis'] = 7 + 8j
    d['rrsigma'] = 0.1
    d['llsigma'] = 0.2
    d['rlsigma'] = 0.3
    d['lrsigma'] = 0.4
    return d


def _stokes_data_1bl():
    """Single-baseline (S0, S1) DTPOL_STOKES table with known values."""
    d = np.zeros(1, dtype=ehc.DTPOL_STOKES)
    d['t1'] = 'S0'
    d['t2'] = 'S1'
    d['u'] = 1e6
    d['v'] = 1e5
    d['vis'] = 1 + 2j
    d['qvis'] = 3 + 4j
    d['uvis'] = 5 + 6j
    d['vvis'] = 7 + 8j
    d['sigma'] = 0.1
    d['qsigma'] = 0.2
    d['usigma'] = 0.3
    d['vsigma'] = 0.4
    return d


def test_data_conj_circ_unchanged():
    # Regression: the circ conjugate transform is the same as before 4c.
    obs = eo.Obsdata(0., 0., 230e9, 1e9, _circ_data_1bl(),
                     _tarr(['rl', 'rl']), polrep='circ')
    dc = obs.data_conj()
    assert len(dc) == 2
    conj = dc[dc['u'] < 0][0]
    assert conj['t1'] == 'S1' and conj['t2'] == 'S0'
    # diagonal slots: conjugate only
    assert conj['rrvis'] == np.conj(1 + 2j)
    assert conj['llvis'] == np.conj(3 + 4j)
    # cross-hand slots: conjugate and swap
    assert conj['rlvis'] == np.conj(7 + 8j)
    assert conj['lrvis'] == np.conj(5 + 6j)
    # diagonal sigmas unchanged; cross-hand sigmas swap
    assert conj['rrsigma'] == 0.1
    assert conj['llsigma'] == 0.2
    assert conj['rlsigma'] == 0.4
    assert conj['lrsigma'] == 0.3


def test_data_conj_stokes_unchanged():
    # Regression: stokes conjugates all four vis fields, no sigma swap.
    obs = eo.Obsdata(0., 0., 230e9, 1e9, _stokes_data_1bl(),
                     _tarr(['rl', 'rl']), polrep='stokes')
    dc = obs.data_conj()
    conj = dc[dc['u'] < 0][0]
    assert conj['vis'] == np.conj(1 + 2j)
    assert conj['qvis'] == np.conj(3 + 4j)
    assert conj['uvis'] == np.conj(5 + 6j)
    assert conj['vvis'] == np.conj(7 + 8j)
    # stokes sigmas are not swapped
    assert conj['usigma'] == 0.3
    assert conj['vsigma'] == 0.4


def test_data_conj_lin():
    # xx/yy conjugate; xy <-> yx conjugate-and-swap; cross sigmas swap.
    obs = eo.Obsdata(0., 0., 230e9, 1e9, _lin_data(['S0'], ['S1']),
                     _tarr(['xy', 'xy']), polrep='lin')
    dc = obs.data_conj()
    conj = dc[dc['u'] < 0][0]
    assert conj['xxvis'] == np.conj(1 + 2j)
    assert conj['yyvis'] == np.conj(3 + 4j)
    assert conj['xyvis'] == np.conj(7 + 8j)
    assert conj['yxvis'] == np.conj(5 + 6j)
    assert conj['xxsigma'] == 0.1
    assert conj['yysigma'] == 0.2
    assert conj['xysigma'] == 0.4
    assert conj['yxsigma'] == 0.3


def test_data_conj_mixed_swap_and_polbasis_flip():
    # Heterogeneous baseline (S0='rl', S1='xy'): generic-slot conj/swap, and
    # the polbasis halves flip on the conjugate row ('rlxy' -> 'xyrl').
    obs = eo.Obsdata(0., 0., 230e9, 1e9, _mixed_data(['S0'], ['S1']),
                     _tarr(['rl', 'xy']), polrep='mixed')
    dc = obs.data_conj()
    orig = dc[dc['u'] > 0][0]
    conj = dc[dc['u'] < 0][0]
    assert orig['polbasis'] == 'rlxy'
    assert conj['polbasis'] == 'xyrl'
    assert conj['p1p1vis'] == np.conj(1 + 2j)
    assert conj['p2p2vis'] == np.conj(3 + 4j)
    assert conj['p1p2vis'] == np.conj(7 + 8j)
    assert conj['p2p1vis'] == np.conj(5 + 6j)
    assert conj['p1p2sigma'] == 0.4
    assert conj['p2p1sigma'] == 0.3


def test_data_conj_mixed_multibaseline_polbasis():
    # 3-station mixed array: every conjugate row's polbasis is the half-swap
    # of its original, and the table doubles in length.
    obs = eo.Obsdata(0., 0., 230e9, 1e9,
                     _mixed_data(['S0', 'S0', 'S1'], ['S1', 'S2', 'S2']),
                     _tarr(['rl', 'rl', 'xy']), polrep='mixed')
    dc = obs.data_conj()
    assert len(dc) == 2 * len(obs.data)
    for row in dc:
        pb = str(row['polbasis'])
        # the reverse-station row must also be present with swapped halves
        mate = pb[2:] + pb[:2]
        assert mate in set(str(s) for s in dc['polbasis'])


# ============================================================================
#  Closures (bispectra / c_phases / c_amplitudes) on lin / mixed
# ============================================================================


def _lin_tri_obs():
    """3-station all-'xy' LIN obs, one timestamp -> one triangle."""
    d = _lin_data(['S0', 'S0', 'S1'], ['S1', 'S2', 'S2'])
    return eo.Obsdata(0., 0., 230e9, 1e9, d, _tarr(['xy', 'xy', 'xy']),
                      polrep='lin')


def _lin_4st_obs():
    """4-station all-'xy' LIN obs, one timestamp -> non-trivial closure set."""
    t1s = ['S0', 'S0', 'S0', 'S1', 'S1', 'S2']
    t2s = ['S1', 'S2', 'S3', 'S2', 'S3', 'S3']
    d = _lin_data(t1s, t2s)
    return eo.Obsdata(0., 0., 230e9, 1e9, d, _tarr(['xy', 'xy', 'xy', 'xy']),
                      polrep='lin')


def _mixed_tri_obs():
    """3-station mixed obs (S0,S1='rl', S2='xy'): the only triangle is feed-mixing."""
    d = _mixed_data(['S0', 'S0', 'S1'], ['S1', 'S2', 'S2'])
    return eo.Obsdata(0., 0., 230e9, 1e9, d, _tarr(['rl', 'rl', 'xy']),
                      polrep='mixed')


def _mixed_4st_obs():
    """4-station mixed obs (S0,S1,S2='rl', S3='xy'): exactly one all-circular
       triangle (S0,S1,S2); the other three triangles touch the linear station."""
    t1s = ['S0', 'S0', 'S0', 'S1', 'S1', 'S2']
    t2s = ['S1', 'S2', 'S3', 'S2', 'S3', 'S3']
    d = _mixed_data(t1s, t2s)
    return eo.Obsdata(0., 0., 230e9, 1e9, d, _tarr(['rl', 'rl', 'rl', 'xy']),
                      polrep='mixed')


def _mixed_5st_obs():
    """5-station mixed obs (S0-S3='rl', S4='xy'): the all-circular S0-S3
       quadrangle survives; every baseline touching S4 is cross-feed and
       skipped. Exercises the diag covariance NaN-filter on a surviving quad."""
    sites = ['S0', 'S1', 'S2', 'S3', 'S4']
    t1s, t2s = [], []
    for i in range(5):
        for j in range(i + 1, 5):
            t1s.append(sites[i])
            t2s.append(sites[j])
    d = _mixed_data(t1s, t2s)
    return eo.Obsdata(0., 0., 230e9, 1e9, d,
                      _tarr(['rl', 'rl', 'rl', 'rl', 'xy']), polrep='mixed')


# ----- lin closures match the stokes-converted observation ------------------

def test_bispectra_lin_matches_stokes():
    obs_l = _lin_tri_obs()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bl = obs_l.bispectra(vtype='vis')
        bs = obs_l.switch_polrep('stokes').bispectra(vtype='vis')
    assert len(bl) == len(bs) == 1
    np.testing.assert_allclose(bl['bispec'], bs['bispec'], rtol=1e-10)


def test_bispectra_lin_native_slot():
    # vtype='xxvis' bispectrum is the product of xxvis around the triangle.
    obs_l = _lin_tri_obs()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        b = obs_l.bispectra(vtype='xxvis')
    d = obs_l.data
    expected = d['xxvis'][0] * d['xxvis'][2] * np.conj(d['xxvis'][1])
    # triangle (S0,S1),(S1,S2),(S2,S0): third leg is conj of (S0,S2)
    assert len(b) == 1
    np.testing.assert_allclose(np.abs(b['bispec'][0]), np.abs(expected), rtol=1e-10)


def test_c_phases_lin_matches_stokes():
    obs_l = _lin_tri_obs()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cl = obs_l.c_phases(vtype='vis')
        cs = obs_l.switch_polrep('stokes').c_phases(vtype='vis')
    assert len(cl) == len(cs) == 1
    np.testing.assert_allclose(cl['cphase'], cs['cphase'], rtol=1e-9)


# ----- vtype validity is polrep-specific ------------------------------------

def test_bispectra_lin_rejects_circular_vtype():
    obs_l = _lin_tri_obs()
    with pytest.raises(Exception, match="vtype"):
        obs_l.bispectra(vtype='rrvis')


def test_bispectra_circ_rejects_linear_vtype():
    obs_c = _stokes_obs().switch_polrep('circ')
    with pytest.raises(Exception, match="vtype"):
        obs_c.bispectra(vtype='xxvis')


# ----- mixed-feed: skip feed-mixing triangles -------------------------------

def test_bispectra_mixed_skips_cross_feed_triangle():
    obs = _mixed_tri_obs()
    with pytest.warns(ehw.MixedPolClosureSkipWarning):
        b = obs.bispectra(vtype='rrvis', count='max')
    assert len(b) == 0


def test_bispectra_mixed_keeps_homogeneous_subtriangle():
    obs = _mixed_4st_obs()
    with pytest.warns(ehw.MixedPolClosureSkipWarning):
        b = obs.bispectra(vtype='rrvis', count='max')
    # only the all-circular triangle survives
    assert len(b) == 1
    assert set((b['t1'][0], b['t2'][0], b['t3'][0])) == {'S0', 'S1', 'S2'}


def test_c_phases_mixed_skips_cross_feed_triangle():
    obs = _mixed_tri_obs()
    with pytest.warns(ehw.MixedPolClosureSkipWarning):
        c = obs.c_phases(vtype='rrvis', count='max')
    assert len(c) == 0


# ----- mixed-feed: Stokes / generic vtypes are not closure-able -------------

def test_bispectra_mixed_stokes_vtype_raises():
    obs = _mixed_tri_obs()
    with pytest.raises(Exception, match="mixed-feed"):
        obs.bispectra(vtype='vis')


def test_bispectra_mixed_generic_vtype_raises():
    obs = _mixed_tri_obs()
    with pytest.raises(Exception, match="mixed-feed"):
        obs.bispectra(vtype='p1p1vis')


# ----- minimal-set caveat is surfaced ---------------------------------------

def test_mixed_skip_warning_minimal_set_caveat():
    obs = _mixed_4st_obs()
    with pytest.warns(ehw.MixedPolClosureSkipWarning, match="baseline-aware"):
        obs.bispectra(vtype='rrvis', count='min')


def test_mixed_skip_warning_no_caveat_for_maxset():
    obs = _mixed_4st_obs()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        obs.bispectra(vtype='rrvis', count='max')
    msgs = [str(w.message) for w in rec
            if issubclass(w.category, ehw.MixedPolClosureSkipWarning)]
    assert msgs and all('baseline-aware' not in m for m in msgs)


# ----- single-triangle helper guards ----------------------------------------

def test_cphase_tri_mixed_heterogeneous_returns_empty():
    obs = _mixed_tri_obs()
    with pytest.warns(ehw.MixedPolClosureSkipWarning):
        out = obs.cphase_tri('S0', 'S1', 'S2', vtype='rrvis')
    assert len(out) == 0


# ----- diagonal closures on lin / mixed -------------------------------------

def test_c_phases_diag_lin_matches_stokes():
    obs_l = _lin_4st_obs()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dl = obs_l.c_phases_diag(vtype='vis')
        ds = obs_l.switch_polrep('stokes').c_phases_diag(vtype='vis')
    # one timestamp; compare the diagonalized closure-phase values
    np.testing.assert_allclose(dl[0][0]['cphase'], ds[0][0]['cphase'], rtol=1e-8)
    np.testing.assert_allclose(dl[0][0]['sigmacp'], ds[0][0]['sigmacp'], rtol=1e-8)


def test_c_log_amplitudes_diag_lin_matches_stokes():
    obs_l = _lin_4st_obs()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dl = obs_l.c_log_amplitudes_diag(vtype='vis')
        ds = obs_l.switch_polrep('stokes').c_log_amplitudes_diag(vtype='vis')
    np.testing.assert_allclose(dl[0][0]['camp'], ds[0][0]['camp'], rtol=1e-8)


def test_c_phases_diag_mixed_skips_cross_feed():
    # only the all-circular sub-triangle survives; diag runs on it with a warning
    obs = _mixed_4st_obs()
    with pytest.warns(ehw.MixedPolClosureSkipWarning):
        d = obs.c_phases_diag(vtype='rrvis')
    assert len(d) >= 1
    # cross-feed baselines must not poison the surviving triangle's covariance
    sigmacp = np.concatenate([x[0]['sigmacp'] for x in d])
    assert np.all(np.isfinite(sigmacp))


def test_c_log_amplitudes_diag_mixed_no_quad_returns_empty():
    # 4 stations, only 3 circular -> no all-circular quadrangle survives.
    # Must return empty without raising (previously crashed with IndexError).
    obs = _mixed_4st_obs()
    with pytest.warns(ehw.MixedPolClosureSkipWarning):
        d = obs.c_log_amplitudes_diag(vtype='rrvis')
    assert d == []


def test_c_log_amplitudes_diag_mixed_skips_cross_feed():
    # the all-circular S0-S3 quad survives; cross-feed baselines must not
    # poison its covariance
    obs = _mixed_5st_obs()
    with pytest.warns(ehw.MixedPolClosureSkipWarning):
        d = obs.c_log_amplitudes_diag(vtype='rrvis')
    assert len(d) >= 1
    sigmaca = np.concatenate([x[0]['sigmaca'] for x in d])
    assert np.all(np.isfinite(sigmaca))


# ============================================================================
#  unpack on lin / mixed
# ============================================================================


def test_unpack_lin_vis_matches_switch_polrep():
    # unpack('vis') on a lin obs must equal switch_polrep('stokes') then unpack
    # (both route through pol_conventions), the key lin<->circ symmetry.
    obs_l = _lin_obs()
    obs_s = obs_l.switch_polrep('stokes')
    for f in ['vis', 'qvis', 'uvis', 'vvis']:
        np.testing.assert_allclose(obs_l.unpack(f)[f], obs_s.unpack(f)[f], atol=1e-12)


def test_unpack_lin_native_and_amp():
    obs_l = _lin_obs()
    np.testing.assert_array_equal(obs_l.unpack('xxvis')['xxvis'], obs_l.data['xxvis'])
    # Stokes-I amplitude from linear feeds: 0.5|XX + YY|
    expected = np.abs(0.5 * (obs_l.data['xxvis'] + obs_l.data['yyvis']))
    np.testing.assert_allclose(obs_l.unpack('amp')['amp'], expected, atol=1e-12)


def test_unpack_generic_slot_lin():
    obs_l = _lin_obs()
    np.testing.assert_array_equal(obs_l.unpack('p1p1vis')['p1p1vis'], obs_l.data['xxvis'])
    np.testing.assert_array_equal(obs_l.unpack('p2p1vis')['p2p1vis'], obs_l.data['yxvis'])


def test_unpack_lin_rejects_circular_name():
    obs_l = _lin_obs()
    with pytest.raises(Exception, match="not supported"):
        obs_l.unpack('rrvis')


def test_unpack_generic_slot_stokes_raises():
    obs_s = _stokes_obs()
    with pytest.raises(Exception, match="no meaning"):
        obs_s.unpack('p1p1vis')


def test_unpack_mixed_stokes_recovered_all_rows():
    # Stokes I is recoverable on every row, including cross-feed (rlxy) ones.
    obs = _mixed_obs()
    vis = obs.unpack('vis')['vis']
    assert np.all(np.isfinite(vis))
    pb = obs.data['polbasis']
    m = (pb == 'rlrl')  # on circular baselines I = 0.5(RR + LL)
    np.testing.assert_allclose(
        vis[m], 0.5 * (obs.data['p1p1vis'][m] + obs.data['p2p2vis'][m]), atol=1e-12)


def test_unpack_mixed_generic_slot():
    obs = _mixed_obs()
    np.testing.assert_array_equal(obs.unpack('p1p1vis')['p1p1vis'], obs.data['p1p1vis'])
    np.testing.assert_array_equal(obs.unpack('p2p1vis')['p2p1vis'], obs.data['p2p1vis'])


def test_unpack_mixed_physical_nanfill_and_warns():
    # rrvis exists only on rl-rl baselines; rl-xy baselines have no RR -> NaN.
    obs = _mixed_obs()
    pb = obs.data['polbasis']
    with pytest.warns(ehw.MixedPolUnpackNaNWarning, match="RR correlation"):
        rr = obs.unpack('rrvis')['rrvis']
    np.testing.assert_array_equal(rr[pb == 'rlrl'], obs.data['p1p1vis'][pb == 'rlrl'])
    assert np.all(np.isnan(rr[pb != 'rlrl']))


def test_unpack_mixed_correlation_absent_everywhere():
    # No baseline has two linear stations, so XX is NaN on every row.
    obs = _mixed_obs()
    with pytest.warns(ehw.MixedPolUnpackNaNWarning):
        xx = obs.unpack('xxvis')['xxvis']
    assert np.all(np.isnan(xx))


# Stokes-derived fields (pvis/m/rrllvis) are recoverable on every mixed row
# under the ideal-feed assumption. Each is checked as the algebraic
# combination of the already-trusted Stokes components.

def test_unpack_mixed_pvis():
    obs = _mixed_obs()
    q = obs.unpack('qvis')['qvis']
    u = obs.unpack('uvis')['uvis']
    pvis = obs.unpack('pvis')['pvis']
    assert np.all(np.isfinite(pvis))
    np.testing.assert_allclose(pvis, q + 1j * u, atol=1e-12)


def test_unpack_mixed_m():
    obs = _mixed_obs()
    ivis = obs.unpack('vis')['vis']
    q = obs.unpack('qvis')['qvis']
    u = obs.unpack('uvis')['uvis']
    m = obs.unpack('m')['m']
    assert np.all(np.isfinite(m))
    np.testing.assert_allclose(m, (q + 1j * u) / ivis, atol=1e-12)


def test_unpack_mixed_rrllvis():
    obs = _mixed_obs()
    ivis = obs.unpack('vis')['vis']
    v = obs.unpack('vvis')['vvis']
    rrll = obs.unpack('rrllvis')['rrllvis']
    assert np.all(np.isfinite(rrll))
    np.testing.assert_allclose(rrll, (ivis + v) / (ivis - v), atol=1e-12)


# ============================================================================
#  load_uvfits mixed-pol detection stop-gap
# ============================================================================

def test_load_uvfits_rejects_noncircular_poltya():
    from astropy.io import fits
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample.uvfits')
    hdul = fits.open(path)
    hdul['AIPS AN'].data['POLTYA'][:] = 'X'   # pretend the AN table flags linear feeds
    with pytest.raises(NotImplementedError, match="mixed-pol"):
        eo.load_uvfits(hdul)


def test_load_uvfits_rejects_hybrid_poltyb():
    # POLTYA stays circular but POLTYB is linear (a hybrid R/X feed) -> caught
    from astropy.io import fits
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample.uvfits')
    hdul = fits.open(path)
    hdul['AIPS AN'].data['POLTYB'][:] = 'X'
    with pytest.raises(NotImplementedError, match="mixed-pol"):
        eo.load_uvfits(hdul)


def test_load_uvfits_circular_unaffected():
    # all-'R' POLTYA (the sample file) loads normally through the new check
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample.uvfits')
    obs = eo.load_uvfits(path)
    assert obs.polrep in ('stokes', 'circ')
    assert set(obs.tarr['feed_type']) == {'rl'}
