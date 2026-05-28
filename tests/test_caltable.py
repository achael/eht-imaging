"""Tests for ehtim.caltable.Caltable."""

import numpy as np
import pytest

import ehtim as eh
from ehtim.const_def import DTCAL

# ---------------------------------------------------------------------------
# Constants used across the module
# ---------------------------------------------------------------------------

# Bit-clean numerical-equality tolerances for applycal scaling and gain
# round-trips. 1e-12 is the float-roundoff floor for products and inversions
# of double-precision complex numbers across this module.
BIT_CLEAN_RTOL = 1e-12
BIT_CLEAN_ATOL = 1e-12

# Interpolated-path tolerance (scipy.interp1d adds a small kernel error).
INTERP_RTOL = 1e-10

# Save/load round-trip precision floors. Gains save as 17-digit floats so
# they read back near float-roundoff. Times go through MJD + time/24 then a
# subtraction, so the floor is ~1e-10.
GAIN_RTOL = BIT_CLEAN_RTOL
TIME_RTOL = 1e-9

# Synthetic gain values for characterization.
CONST_REAL_GAIN = 2.0 + 0j        # used in applycal, invert_gains, enforce
CONST_LOW_GAIN = 0.5 + 0j         # below DEFAULT_MIN_GAIN; triggers rescaling
CONST_INTERP_GAIN = 1.7 + 0.3j    # complex gain for interp-mode tests
CONST_CUBIC_GAIN = 1.5 + 0j       # used in the cubic-interp test
CONST_PHASE = 0.37                # pure-phase rotation for amp-preservation

# enforce_positive threshold (matches the Caltable default).
DEFAULT_MIN_GAIN = 0.9

# RNG seeds for reproducible injected-gain caltables.
SEED_INVERT_ROUNDTRIP = 7
SEED_SAVE_LOAD_ROUNDTRIP = 3
SEED_SAVE_TXT_MATCH = 5
SEED_SQRT_ROUNDTRIP = 11


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
            rtol=INTERP_RTOL, atol=BIT_CLEAN_ATOL,
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
        ct = constant_gain_caltable_factory(CONST_REAL_GAIN)
        ct.invert_gains()
        for site in ct.data:
            np.testing.assert_allclose(ct.data[site]['rscale'], 1 / CONST_REAL_GAIN)
            np.testing.assert_allclose(ct.data[site]['lscale'], 1 / CONST_REAL_GAIN)

    def test_invert_twice_is_identity(self, injected_gain_caltable_factory):
        ct = injected_gain_caltable_factory(seed=SEED_INVERT_ROUNDTRIP)
        ref = {site: ct.data[site]['rscale'].copy() for site in ct.data}
        ct.invert_gains()
        ct.invert_gains()
        for site, expected in ref.items():
            np.testing.assert_allclose(ct.data[site]['rscale'], expected,
                                       rtol=BIT_CLEAN_RTOL)

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
        # g_i * conj(g_j) = g**2 for every baseline when all stations share g.
        g = CONST_REAL_GAIN
        out = constant_gain_caltable_factory(g).applycal(obs_direct, interp='nearest')
        for f in ('vis', 'qvis', 'uvis', 'vvis'):
            np.testing.assert_allclose(
                out.data[f], obs_direct.data[f] * np.abs(g) ** 2,
                rtol=BIT_CLEAN_RTOL, atol=BIT_CLEAN_ATOL,
            )

    def test_pure_phase_gain_preserves_amplitude(self, obs_direct):
        # |g_i * conj(g_j)| = 1 for any common phase ⇒ amplitudes invariant.
        g = np.exp(1j * CONST_PHASE)
        ct = _caltable_with_times(
            obs_direct,
            [obs_direct.data['time'].min() - 1.0,
             obs_direct.data['time'].max() + 1.0],
            rscale=g, lscale=g,
        )
        out = ct.applycal(obs_direct, interp='nearest')
        np.testing.assert_allclose(
            np.abs(out.data['vis']), np.abs(obs_direct.data['vis']),
            rtol=BIT_CLEAN_RTOL, atol=BIT_CLEAN_ATOL,
        )


class TestApplycalSiteSubset:

    def test_missing_site_baselines_unscaled(self, obs_direct,
                                             constant_gain_caltable_factory):
        # Sites absent from caltable.data fall back to gain = 1.
        g = CONST_REAL_GAIN
        ct = constant_gain_caltable_factory(g)
        dropped = obs_direct.tarr['site'][0]
        ct.data.pop(dropped)

        out = ct.applycal(obs_direct, interp='nearest')
        # Both present: g**2.  One dropped: g (the survivor) * conj(1).
        both_present = (out.data['t1'] != dropped) & (out.data['t2'] != dropped)
        one_dropped = ~both_present
        if both_present.any():
            np.testing.assert_allclose(
                out.data['vis'][both_present],
                obs_direct.data['vis'][both_present] * np.abs(g) ** 2,
                rtol=BIT_CLEAN_RTOL,
            )
        if one_dropped.any():
            np.testing.assert_allclose(
                out.data['vis'][one_dropped],
                obs_direct.data['vis'][one_dropped] * g,
                rtol=BIT_CLEAN_RTOL,
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
    def test_constant_gain_recovered_at_every_time(self, obs_direct, interp):
        # Both modes return g**2 at every obs time when the caltable is flat.
        g = CONST_INTERP_GAIN
        ct = _caltable_with_times(
            obs_direct,
            [obs_direct.data['time'].min() - 1.0,
             obs_direct.data['time'].max() + 1.0],
            rscale=g, lscale=g,
        )
        out = ct.applycal(obs_direct, interp=interp)
        np.testing.assert_allclose(
            out.data['vis'], obs_direct.data['vis'] * np.abs(g) ** 2,
            rtol=INTERP_RTOL, atol=BIT_CLEAN_ATOL,
        )

    def test_cubic_requires_four_points(self, obs_direct):
        # cubic interp needs >= 4 anchor points; constant g recovers g**2.
        g = CONST_CUBIC_GAIN
        t0 = obs_direct.data['time'].min() - 1.0
        t1 = obs_direct.data['time'].max() + 1.0
        times = np.linspace(t0, t1, 4)
        ct = _caltable_with_times(obs_direct, times, rscale=g, lscale=g)
        out = ct.applycal(obs_direct, interp='cubic')
        np.testing.assert_allclose(
            out.data['vis'], obs_direct.data['vis'] * np.abs(g) ** 2,
            rtol=INTERP_RTOL, atol=BIT_CLEAN_ATOL,
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


# ---------------------------------------------------------------------------
# Section 5: Save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:

    def test_save_caltable_load_caltable_roundtrip(self, obs_direct,
                                                   injected_gain_caltable_factory,
                                                   tmp_path):
        ct = injected_gain_caltable_factory(seed=SEED_SAVE_LOAD_ROUNDTRIP)
        eh.caltable.save_caltable(ct, obs_direct, datadir=str(tmp_path))
        loaded = eh.caltable.load_caltable(obs_direct, str(tmp_path))
        assert isinstance(loaded, eh.caltable.Caltable)
        for site in ct.data:
            np.testing.assert_allclose(loaded.data[site]['rscale'],
                                       ct.data[site]['rscale'], rtol=GAIN_RTOL)
            np.testing.assert_allclose(loaded.data[site]['lscale'],
                                       ct.data[site]['lscale'], rtol=GAIN_RTOL)
            np.testing.assert_allclose(loaded.data[site]['time'],
                                       ct.data[site]['time'], rtol=TIME_RTOL)

    def test_sqrt_gains_roundtrip_preserves_squared_gain(self, obs_direct,
                                                        injected_gain_caltable_factory,
                                                        tmp_path):
        # On-disk quantity is the squared gain, so loaded**2 == original**2
        # is the exact round-trip. Comparing squares also sidesteps the
        # principal-branch sqrt sign flip outside (-pi/2, pi/2).
        ct = injected_gain_caltable_factory(seed=SEED_SQRT_ROUNDTRIP)
        eh.caltable.save_caltable(ct, obs_direct, datadir=str(tmp_path),
                                  sqrt_gains=True)
        loaded = eh.caltable.load_caltable(obs_direct, str(tmp_path),
                                           sqrt_gains=True)
        for site in ct.data:
            np.testing.assert_allclose(loaded.data[site]['rscale'] ** 2,
                                       ct.data[site]['rscale'] ** 2,
                                       rtol=GAIN_RTOL, atol=BIT_CLEAN_ATOL)
            np.testing.assert_allclose(loaded.data[site]['lscale'] ** 2,
                                       ct.data[site]['lscale'] ** 2,
                                       rtol=GAIN_RTOL, atol=BIT_CLEAN_ATOL)
            np.testing.assert_allclose(loaded.data[site]['time'],
                                       ct.data[site]['time'], rtol=TIME_RTOL)

    def test_save_txt_method_matches_module_function(self, obs_direct,
                                                     injected_gain_caltable_factory,
                                                     tmp_path):
        # Caltable.save_txt is a thin wrapper; files should be byte-identical.
        ct = injected_gain_caltable_factory(seed=SEED_SAVE_TXT_MATCH)
        out_a = tmp_path / "a"
        out_b = tmp_path / "b"
        out_a.mkdir()
        out_b.mkdir()
        ct.save_txt(obs_direct, datadir=str(out_a))
        eh.caltable.save_caltable(ct, obs_direct, datadir=str(out_b))
        for site in ct.data:
            fa = out_a / f"{obs_direct.source}_{site}.txt"
            fb = out_b / f"{obs_direct.source}_{site}.txt"
            assert fa.read_bytes() == fb.read_bytes()


# ---------------------------------------------------------------------------
# Section 6: enforce_positive
# ---------------------------------------------------------------------------


class TestEnforcePositive:

    def test_no_op_when_all_gains_above_min(self, unity_caltable):
        out = unity_caltable.enforce_positive(method='median', min_gain=0.5,
                                              verbose=False)
        for site in unity_caltable.data:
            np.testing.assert_allclose(out.data[site]['rscale'],
                                       unity_caltable.data[site]['rscale'])
            np.testing.assert_allclose(out.data[site]['lscale'],
                                       unity_caltable.data[site]['lscale'])

    def test_rescales_low_gains_above_threshold(self, constant_gain_caltable_factory):
        # All gains g = CONST_LOW_GAIN < DEFAULT_MIN_GAIN ⇒ median |g| = |g|,
        # rescale by 1/|g| ⇒ new |gain| = 1.
        ct = constant_gain_caltable_factory(CONST_LOW_GAIN)
        out = ct.enforce_positive(method='median', min_gain=DEFAULT_MIN_GAIN,
                                  verbose=False)
        for site in out.data:
            np.testing.assert_allclose(np.abs(out.data[site]['rscale']), 1.0)
            np.testing.assert_allclose(np.abs(out.data[site]['lscale']), 1.0)

    def test_unknown_method_returns_unchanged_copy(self, constant_gain_caltable_factory):
        ct = constant_gain_caltable_factory(CONST_LOW_GAIN)
        out = ct.enforce_positive(method='nope', min_gain=DEFAULT_MIN_GAIN,
                                  verbose=False)
        first = next(iter(out.data))
        np.testing.assert_allclose(out.data[first]['rscale'],
                                   ct.data[first]['rscale'])

    def test_sites_subset_only_affects_listed(self, constant_gain_caltable_factory):
        ct = constant_gain_caltable_factory(CONST_LOW_GAIN)
        first, second = list(ct.data.keys())[:2]
        out = ct.enforce_positive(method='median', min_gain=DEFAULT_MIN_GAIN,
                                  sites=[first], verbose=False)
        np.testing.assert_allclose(np.abs(out.data[first]['rscale']), 1.0)
        np.testing.assert_allclose(np.abs(out.data[second]['rscale']),
                                   np.abs(CONST_LOW_GAIN))

    def test_returns_independent_copy(self, constant_gain_caltable_factory):
        ct = constant_gain_caltable_factory(CONST_LOW_GAIN)
        out = ct.enforce_positive(method='median', min_gain=DEFAULT_MIN_GAIN,
                                  verbose=False)
        first = next(iter(ct.data))
        out.data[first]['rscale'] *= 7
        assert ct.data[first]['rscale'][0] == CONST_LOW_GAIN


# ---------------------------------------------------------------------------
# Section 7: relaxed_interp1d utility
# ---------------------------------------------------------------------------


class TestRelaxedInterp1d:

    def test_single_point_falls_back_to_constant(self):
        f = eh.caltable.relaxed_interp1d(np.array([3.0]), np.array([2.5]),
                                         kind='linear')
        # The helper expands a 1-point input to a 2-point [x-0.5, x+0.5]
        # constant fan so the value is recovered at the anchor and nearby.
        assert f(3.0) == pytest.approx(2.5)
        assert f(2.7) == pytest.approx(2.5)
        assert f(3.3) == pytest.approx(2.5)

    def test_scalar_x_y_are_promoted_to_arrays(self):
        # The bare-scalar code path catches TypeError on len(x) and recovers.
        f = eh.caltable.relaxed_interp1d(0.0, 1.5, kind='linear')
        assert f(0.0) == pytest.approx(1.5)

    def test_multi_point_matches_scipy_interp1d(self):
        # When len(x) > 1, relaxed_interp1d delegates straight to
        # scipy.interpolate.interp1d ⇒ bit-identical outputs.
        import scipy.interpolate as spi
        x = np.linspace(0.0, 1.0, 5)
        y = x ** 2
        f_ehtim = eh.caltable.relaxed_interp1d(x, y, kind='linear')
        f_scipy = spi.interp1d(x, y, kind='linear')
        probes = np.linspace(0.1, 0.9, 7)
        np.testing.assert_array_equal(f_ehtim(probes), f_scipy(probes))
