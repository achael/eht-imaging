"""Tests for the mixed-polarization data infrastructure.

Tests are grouped by the phase of obsdata_mixedpol_plan_v2.md they cover.
Subsequent phases (2, 2.5, 3, ...) should append their tests to this file
under matching headers.
"""
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
