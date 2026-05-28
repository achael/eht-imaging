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

    def test_copy_data_is_independent(self, constant_gain_caltable_factory):
        original = constant_gain_caltable_factory(2.0 + 0j)
        cp = original.copy()
        first_site = next(iter(cp.data))
        cp.data[first_site]['rscale'] *= 10
        assert original.data[first_site]['rscale'][0] == 2 + 0j

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


# ---------------------------------------------------------------------------
# Section 4: applycal
# ---------------------------------------------------------------------------


def _caltable_with_times(obs, times, rscale=1.0 + 0j, lscale=1.0 + 0j):
    """Helper: build a Caltable with arbitrary per-time gains for every site."""
    times = np.asarray(times, dtype=float)
    n = len(times)
    r = np.broadcast_to(np.asarray(rscale, dtype=complex), (n,))
    ll = np.broadcast_to(np.asarray(lscale, dtype=complex), (n,))
    caldict = {
        site: np.array(
            [(times[k], r[k], ll[k]) for k in range(n)], dtype=DTCAL,
        ).view(np.recarray)
        for site in obs.tarr['site']
    }
    return eh.caltable.Caltable(
        obs.ra, obs.dec, obs.rf, obs.bw, caldict, obs.tarr,
        source=obs.source, mjd=obs.mjd, timetype=obs.timetype,
    )


class TestApplycalConstantGain:

    def test_real_gain_scales_amp_by_g_squared(self, obs_direct,
                                               constant_gain_caltable_factory):
        # Constant rscale = lscale = g (real) ⇒ vis_out = vis_in * g**2 on
        # every baseline (the per-station product is g_i * conj(g_j) = g**2).
        g = 2.0 + 0j
        out = constant_gain_caltable_factory(g).applycal(obs_direct, interp='nearest')
        for f in ('vis', 'qvis', 'uvis', 'vvis'):
            np.testing.assert_allclose(
                out.data[f], obs_direct.data[f] * np.abs(g) ** 2,
                rtol=1e-12, atol=1e-12,
            )

    def test_pure_phase_gain_preserves_amplitude(self, obs_direct):
        # rscale = lscale = exp(i phi) ⇒ |vis_out| = |vis_in| (phase is
        # absorbed since g_i * conj(g_j) magnitude is 1).
        phi = 0.37
        g = np.exp(1j * phi)
        ct = _caltable_with_times(
            obs_direct,
            [obs_direct.data['time'].min() - 1.0,
             obs_direct.data['time'].max() + 1.0],
            rscale=g, lscale=g,
        )
        out = ct.applycal(obs_direct, interp='nearest')
        np.testing.assert_allclose(
            np.abs(out.data['vis']), np.abs(obs_direct.data['vis']),
            rtol=1e-12, atol=1e-12,
        )


class TestApplycalSiteSubset:

    def test_missing_site_baselines_unscaled(self, obs_direct,
                                             constant_gain_caltable_factory):
        # If a site is absent from caltable.data, applycal warns and treats its
        # gain as 1 ⇒ baselines touching that site get scaled by g_other^1 only,
        # not g_other * conj(g_dropped) = g_other.
        g = 2.0 + 0j
        ct = constant_gain_caltable_factory(g)
        dropped = obs_direct.tarr['site'][0]
        ct.data.pop(dropped)

        out = ct.applycal(obs_direct, interp='nearest')
        # Baselines with neither station dropped scale by g**2; baselines with
        # one station dropped scale by g (the surviving station's g, conj(1)).
        both_present = (out.data['t1'] != dropped) & (out.data['t2'] != dropped)
        one_dropped = ~both_present
        if both_present.any():
            np.testing.assert_allclose(
                out.data['vis'][both_present],
                obs_direct.data['vis'][both_present] * np.abs(g) ** 2,
                rtol=1e-12,
            )
        if one_dropped.any():
            np.testing.assert_allclose(
                out.data['vis'][one_dropped],
                obs_direct.data['vis'][one_dropped] * g,
                rtol=1e-12,
            )


class TestApplycalPolrep:

    def test_stokes_input_returns_stokes(self, obs_direct, unity_caltable):
        out = unity_caltable.applycal(obs_direct, interp='nearest')
        assert out.polrep == 'stokes'

    def test_circ_input_returns_circ(self, obs_direct, unity_caltable):
        obs_circ = obs_direct.switch_polrep('circ')
        out = unity_caltable.applycal(obs_circ, interp='nearest')
        assert out.polrep == 'circ'


class TestApplycalRejectsMismatchedTarr:

    def test_different_tarr_raises(self, obs_direct, unity_caltable):
        obs_mut = obs_direct.copy()
        obs_mut.tarr['x'][0] += 1.0
        with pytest.raises(Exception, match="telescope array"):
            unity_caltable.applycal(obs_mut, interp='nearest')


class TestApplycalInterpModes:

    @pytest.mark.parametrize("interp", ["nearest", "linear"])
    def test_interp_at_boundary_matches_boundary_gain(self, obs_direct, interp):
        # When the caltable spans the obs and the gain is constant within
        # the spanned interval, all three interp modes recover the boundary
        # gain at every observed time.
        g = 1.7 + 0.3j
        ct = _caltable_with_times(
            obs_direct,
            [obs_direct.data['time'].min() - 1.0,
             obs_direct.data['time'].max() + 1.0],
            rscale=g, lscale=g,
        )
        out = ct.applycal(obs_direct, interp=interp)
        np.testing.assert_allclose(
            out.data['vis'], obs_direct.data['vis'] * np.abs(g) ** 2,
            rtol=1e-10, atol=1e-12,
        )

    def test_cubic_requires_four_points(self, obs_direct):
        # cubic interpolation needs >= 4 anchor points; with a 4-time caltable
        # of constant g, applycal returns g**2-scaled visibilities.
        g = 1.5 + 0j
        t0 = obs_direct.data['time'].min() - 1.0
        t1 = obs_direct.data['time'].max() + 1.0
        times = np.linspace(t0, t1, 4)
        ct = _caltable_with_times(obs_direct, times, rscale=g, lscale=g)
        out = ct.applycal(obs_direct, interp='cubic')
        np.testing.assert_allclose(
            out.data['vis'], obs_direct.data['vis'] * np.abs(g) ** 2,
            rtol=1e-10, atol=1e-12,
        )


class TestApplycalExtrapolation:

    @pytest.mark.xfail(reason="applycal silently fills out-of-span samples with "
                              "NaN when extrapolate is None and the caltable "
                              "doesn't span the obs times. Tracked for a "
                              "follow-up fix PR.",
                       strict=True)
    def test_undersized_caltable_does_not_silently_nan(self, obs_direct):
        # Caltable covers only the first half of the obs; with extrapolate=None
        # the second half should not silently come back NaN.
        t0 = obs_direct.data['time'].min()
        t_mid = (obs_direct.data['time'].min()
                 + obs_direct.data['time'].max()) / 2.0
        ct = _caltable_with_times(obs_direct, [t0 - 1.0, t_mid])
        out = ct.applycal(obs_direct, interp='linear', extrapolate=None)
        assert not np.any(np.isnan(out.data['vis']))

    def test_extrapolate_true_fills_outside_span(self, obs_direct):
        # With extrapolate=True, the scipy 'extrapolate' fill value is used
        # and the calibrated visibilities are finite everywhere.
        t0 = obs_direct.data['time'].min()
        t_mid = (obs_direct.data['time'].min()
                 + obs_direct.data['time'].max()) / 2.0
        ct = _caltable_with_times(obs_direct, [t0 - 1.0, t_mid])
        out = ct.applycal(obs_direct, interp='linear', extrapolate=True)
        assert np.all(np.isfinite(out.data['vis']))
