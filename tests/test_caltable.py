"""Tests for ehtim.caltable.Caltable."""

import numpy as np
import pytest

import ehtim as eh
from ehtim.const_def import DTCAL


def _unity_caltable(obs):
    """A Caltable with rscale = lscale = 1 for every site, spanning the obs times.

    Kept local so the existing applycal test stays self-contained; new tests
    use the session-scoped ``unity_caltable`` fixture from conftest.
    """
    times = np.array([obs.data['time'].min() - 1.0, obs.data['time'].max() + 1.0])
    caldict = {}
    for site in obs.tarr['site']:
        caldict[site] = np.array(
            [(t, 1.0 + 0j, 1.0 + 0j) for t in times], dtype=DTCAL
        ).view(np.recarray)
    return eh.caltable.Caltable(
        obs.ra, obs.dec, obs.rf, obs.bw, caldict, obs.tarr,
        source=obs.source, mjd=obs.mjd, timetype=obs.timetype,
    )


def test_applycal_unity_preserves_data(obs_direct):
    """Unity-gain calibration returns every visibility unchanged.

    Exercises applycal's per-baseline assembly (collect then single hstack):
    the calibrated observation must keep all rows and, with gains of 1, leave
    the visibility amplitudes untouched.
    """
    obs = obs_direct
    calobs = _unity_caltable(obs).applycal(obs, interp='nearest')

    assert len(calobs.data) == len(obs.data)

    obs_c = obs.switch_polrep('circ')
    cal_c = calobs.switch_polrep('circ')
    for field in ('rrvis', 'llvis'):
        np.testing.assert_allclose(
            np.sort(np.abs(cal_c.data[field])),
            np.sort(np.abs(obs_c.data[field])),
            rtol=1e-10, atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Section 1: Construction
# ---------------------------------------------------------------------------


class TestCaltableConstruction:

    def test_init_sets_scalar_attrs(self, unity_caltable, obs_direct):
        ct = unity_caltable
        assert ct.source == obs_direct.source
        assert ct.ra == obs_direct.ra
        assert ct.dec == obs_direct.dec
        assert ct.rf == obs_direct.rf
        assert ct.bw == obs_direct.bw
        assert ct.mjd == obs_direct.mjd
        assert ct.timetype == obs_direct.timetype

    def test_init_builds_tkey_from_tarr(self, unity_caltable):
        ct = unity_caltable
        for i, row in enumerate(ct.tarr):
            assert ct.tkey[row['site']] == i

    def test_init_rejects_bad_timetype(self, obs_direct):
        with pytest.raises(Exception, match="GMST"):
            eh.caltable.Caltable(
                obs_direct.ra, obs_direct.dec, obs_direct.rf, obs_direct.bw,
                {}, obs_direct.tarr, timetype='TAI',
            )

    def test_init_data_keys_match_sites(self, unity_caltable, obs_direct):
        assert set(unity_caltable.data.keys()) == set(obs_direct.tarr['site'])

    def test_make_caltable_square_ntele_eq_ntimes(self, obs_direct):
        # ntele == ntimes: indexing gains[s*ntele + t] coincides with the
        # intended gains[s*ntimes + t] so the mapping is correct.
        sites = list(obs_direct.tarr['site'])[:3]
        times = [obs_direct.data['time'].min(),
                 (obs_direct.data['time'].min() + obs_direct.data['time'].max()) / 2,
                 obs_direct.data['time'].max()]
        gains = [complex(idx) for idx in range(9)]   # flat list, s*ntimes + t
        ct = eh.caltable.make_caltable(obs_direct, gains, sites, times)
        assert isinstance(ct, eh.caltable.Caltable)
        # site sites[1], time times[2] -> gains[1*3 + 2] = 5
        assert ct.data[sites[1]][2]['rscale'] == 5 + 0j
        assert ct.data[sites[1]][2]['lscale'] == 5 + 0j

    def test_make_caltable_returns_false_on_empty(self, obs_direct):
        assert eh.caltable.make_caltable(obs_direct, [], [], []) is False

    @pytest.mark.xfail(reason="make_caltable uses gains[s*ntele + t] instead of "
                              "s*ntimes + t; rect (ntele != ntimes) yields a "
                              "transposed/OOB indexing. Separate fix PR.",
                       strict=True)
    def test_make_caltable_rect_ntele_ne_ntimes(self, obs_direct):
        sites = list(obs_direct.tarr['site'])[:2]
        times = [obs_direct.data['time'].min(),
                 (obs_direct.data['time'].min() + obs_direct.data['time'].max()) / 2,
                 obs_direct.data['time'].max()]
        gains = [complex(idx) for idx in range(6)]   # s*ntimes + t
        ct = eh.caltable.make_caltable(obs_direct, gains, sites, times)
        # site sites[1], time times[0] -> gains[1*3 + 0] = 3
        assert ct.data[sites[1]][0]['rscale'] == 3 + 0j


# ---------------------------------------------------------------------------
# Section 2: Copy
# ---------------------------------------------------------------------------


class TestCaltableCopy:

    def test_copy_returns_caltable_instance(self, unity_caltable):
        assert isinstance(unity_caltable.copy(), eh.caltable.Caltable)

    def test_copy_preserves_scalar_attrs(self, unity_caltable):
        ct = unity_caltable.copy()
        for attr in ('source', 'ra', 'dec', 'rf', 'bw', 'mjd', 'timetype'):
            assert getattr(ct, attr) == getattr(unity_caltable, attr)

    @pytest.mark.xfail(reason="Caltable.copy() is shallow; data dict aliases the "
                              "original. Fixed in next commit.",
                       strict=True)
    def test_copy_data_is_independent(self, constant_gain_caltable_factory):
        original = constant_gain_caltable_factory(2.0 + 0j)
        cp = original.copy()
        first_site = next(iter(cp.data))
        cp.data[first_site]['rscale'] *= 10
        assert original.data[first_site]['rscale'][0] == 2 + 0j

    @pytest.mark.xfail(reason="Caltable.copy() is shallow; tarr aliases the "
                              "original. Fixed in next commit.",
                       strict=True)
    def test_copy_tarr_is_independent(self, unity_caltable):
        cp = unity_caltable.copy()
        cp.tarr['sefdr'][0] = -1234.0
        assert unity_caltable.tarr['sefdr'][0] != -1234.0


# ---------------------------------------------------------------------------
# Section 3: Invert gains
# ---------------------------------------------------------------------------


class TestInvertGains:

    def test_invert_unity_is_unity(self, unity_caltable):
        ct = unity_caltable.copy()
        ct.invert_gains()
        for site in ct.data:
            np.testing.assert_allclose(ct.data[site]['rscale'], 1 + 0j)
            np.testing.assert_allclose(ct.data[site]['lscale'], 1 + 0j)

    def test_invert_constant_g_yields_reciprocal(self, constant_gain_caltable_factory):
        ct = constant_gain_caltable_factory(2.0 + 0j)
        ct.invert_gains()
        for site in ct.data:
            np.testing.assert_allclose(ct.data[site]['rscale'], 0.5 + 0j)
            np.testing.assert_allclose(ct.data[site]['lscale'], 0.5 + 0j)

    def test_invert_twice_is_identity(self, injected_gain_caltable_factory):
        ct = injected_gain_caltable_factory(seed=7)
        ref = {site: ct.data[site]['rscale'].copy() for site in ct.data}
        ct.invert_gains()
        ct.invert_gains()
        for site, expected in ref.items():
            np.testing.assert_allclose(ct.data[site]['rscale'], expected, rtol=1e-12)

    def test_invert_returns_self(self, unity_caltable):
        ct = unity_caltable.copy()
        assert ct.invert_gains() is ct
