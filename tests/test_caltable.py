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
SEED_SCAN_AVG_PHASES = 0

# pad_scans synthetic-block geometry.
PAD_SCAN_NSAMPLES = 3
PAD_SCAN_DT_SEC = 10.0          # spacing between samples within one scan
PAD_SCAN_GAP_SEC = 300.0        # gap between consecutive scans (> maxdiff)
PAD_SCAN_MAXDIFF_SEC = 60       # default applied in pad_scans tests
# Per-scan median gains used in the median-padding test.
PAD_SCAN_MEDIAN_GAINS = (0.5 + 0j, 0.7 + 0j)
# Constant gain seeded into each per-scan block for the endval-padding test.
PAD_SCAN_ENDVAL_GAIN = 1.3 + 0j

# scan_avg synthetic data.
SCAN_INCOH_MAGNITUDES = (0.6, 1.2)
SCAN_COH_GAINS = (0.5 + 0.5j, -0.8 + 0.2j)

# merge test gains.
MERGE_GAIN_A = 1.7 + 0j
MERGE_GAIN_B = 0.5 + 0j


def _stack_gains(ct):
    """Return (rscale_stack, lscale_stack) concatenated across sites.

    Many tests assert a single property uniformly over all sites; flattening
    the per-site arrays into one buffer turns the assertion into one vectorised
    call instead of a per-site loop.
    """
    rs = np.concatenate([arr['rscale'] for arr in ct.data.values()])
    ls = np.concatenate([arr['lscale'] for arr in ct.data.values()])
    return rs, ls


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
        idx = np.fromiter((ct.tkey[s] for s in ct.tarr['site']),
                          dtype=int, count=len(ct.tarr))
        np.testing.assert_array_equal(idx, np.arange(len(ct.tarr)))

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

    def test_make_caltable_rect_ntele_ne_ntimes(self, obs_direct):
        # ntele=2, ntimes=3. With the correct s*ntimes + t indexing, gain at
        # (site i, time j) is gains[i*ntimes + j].
        sites = list(obs_direct.tarr['site'])[:2]
        times = [obs_direct.data['time'].min(),
                 (obs_direct.data['time'].min() + obs_direct.data['time'].max()) / 2,
                 obs_direct.data['time'].max()]
        gains = np.arange(6, dtype=complex)
        ct = eh.caltable.make_caltable(obs_direct, gains, sites, times)
        expected = gains.reshape(2, 3)
        got_r = np.stack([ct.data[site]['rscale'] for site in sites])
        got_l = np.stack([ct.data[site]['lscale'] for site in sites])
        np.testing.assert_array_equal(got_r, expected)
        np.testing.assert_array_equal(got_l, expected)


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
        r_stack, l_stack = _stack_gains(ct)
        np.testing.assert_allclose(r_stack, 1 + 0j)
        np.testing.assert_allclose(l_stack, 1 + 0j)

    def test_invert_constant_g_yields_reciprocal(self, constant_gain_caltable_factory):
        ct = constant_gain_caltable_factory(CONST_REAL_GAIN)
        ct.invert_gains()
        r_stack, l_stack = _stack_gains(ct)
        np.testing.assert_allclose(r_stack, 1 / CONST_REAL_GAIN)
        np.testing.assert_allclose(l_stack, 1 / CONST_REAL_GAIN)

    def test_invert_twice_is_identity(self, injected_gain_caltable_factory):
        ct = injected_gain_caltable_factory(seed=SEED_INVERT_ROUNDTRIP)
        ref_r, _ = _stack_gains(ct)
        ct.invert_gains()
        ct.invert_gains()
        r_stack, _ = _stack_gains(ct)
        np.testing.assert_allclose(r_stack, ref_r, rtol=BIT_CLEAN_RTOL)

    def test_invert_returns_self(self, unity_caltable):
        ct = unity_caltable.copy()
        assert ct.invert_gains() is ct


# ---------------------------------------------------------------------------
# Section 4: applycal
# ---------------------------------------------------------------------------


def _caltable_with_times(obs, times, rscale=1.0 + 0j, lscale=1.0 + 0j):
    """Helper: build a Caltable with arbitrary per-time gains for every site.

    All sites share the same gain template; per-site arrays are copies so
    mutating one site's gains doesn't leak across sites.
    """
    times = np.asarray(times, dtype=float)
    n = len(times)
    template = np.empty(n, dtype=DTCAL)
    template['time'] = times
    template['rscale'] = np.broadcast_to(np.asarray(rscale, dtype=complex), (n,))
    template['lscale'] = np.broadcast_to(np.asarray(lscale, dtype=complex), (n,))
    caldict = {site: template.copy().view(np.recarray)
               for site in obs.tarr['site']}
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

    # The applycal docstring only documents extrapolate=True; the False/None
    # behaviour is left to the underlying scipy.interpolate.interp1d default
    # (NaN fill). These tests pin that delegated behaviour rather than treat
    # it as a bug: with extrapolate=None, samples outside the caltable span
    # come back NaN, exactly on those rows.

    def test_extrapolate_none_yields_nan_only_outside_span(self, obs_direct):
        t0 = obs_direct.data['time'].min()
        t_mid = 0.5 * (obs_direct.data['time'].min()
                       + obs_direct.data['time'].max())
        ct = _caltable_with_times(obs_direct, [t0 - 1.0, t_mid])
        out = ct.applycal(obs_direct, interp='linear', extrapolate=None)
        outside = obs_direct.data['time'] > t_mid
        inside = ~outside
        # Outside-span rows are NaN; in-span rows remain finite.
        assert np.all(np.isnan(out.data['vis'][outside]))
        assert np.all(np.isfinite(out.data['vis'][inside]))

    def test_extrapolate_true_fills_outside_span(self, obs_direct):
        # With extrapolate=True, scipy's 'extrapolate' fill value is used and
        # the calibrated visibilities are finite everywhere.
        t0 = obs_direct.data['time'].min()
        t_mid = 0.5 * (obs_direct.data['time'].min()
                       + obs_direct.data['time'].max())
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
        out_r, out_l = _stack_gains(out)
        in_r, in_l = _stack_gains(unity_caltable)
        np.testing.assert_allclose(out_r, in_r)
        np.testing.assert_allclose(out_l, in_l)

    def test_rescales_low_gains_above_threshold(self, constant_gain_caltable_factory):
        # All gains g = CONST_LOW_GAIN < DEFAULT_MIN_GAIN ⇒ median |g| = |g|,
        # rescale by 1/|g| ⇒ new |gain| = 1.
        ct = constant_gain_caltable_factory(CONST_LOW_GAIN)
        out = ct.enforce_positive(method='median', min_gain=DEFAULT_MIN_GAIN,
                                  verbose=False)
        out_r, out_l = _stack_gains(out)
        np.testing.assert_allclose(np.abs(out_r), 1.0)
        np.testing.assert_allclose(np.abs(out_l), 1.0)

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


# ---------------------------------------------------------------------------
# Section 8: pad_scans
# ---------------------------------------------------------------------------


def _multi_scan_caltable(obs, n_scans, samples_per_scan=PAD_SCAN_NSAMPLES,
                         dt_in_scan_sec=PAD_SCAN_DT_SEC,
                         gap_sec=PAD_SCAN_GAP_SEC,
                         rscale=1.0 + 0j, lscale=1.0 + 0j):
    """Build a Caltable whose times form `n_scans` blocks separated by `gap_sec`.

    Each block has `samples_per_scan` points spaced by `dt_in_scan_sec`. The
    block structure is purely time-gap-driven and does not need to align with
    obs.scans (pad_scans only looks at gaps in the caltable times).
    """
    block_dt_hr = dt_in_scan_sec / 3600.0
    gap_hr = gap_sec / 3600.0
    t0 = obs.data['time'].min()
    block_offsets = t0 + np.arange(n_scans) * (samples_per_scan * block_dt_hr + gap_hr)
    inner = np.arange(samples_per_scan) * block_dt_hr
    times = (block_offsets[:, None] + inner[None, :]).reshape(-1)
    n = len(times)
    template = np.empty(n, dtype=DTCAL)
    template['time'] = times
    template['rscale'] = np.broadcast_to(np.asarray(rscale, dtype=complex), (n,))
    template['lscale'] = np.broadcast_to(np.asarray(lscale, dtype=complex), (n,))
    caldict = {site: template.copy().view(np.recarray) for site in obs.tarr['site']}
    return eh.caltable.Caltable(
        obs.ra, obs.dec, obs.rf, obs.bw, caldict, obs.tarr,
        source=obs.source, mjd=obs.mjd, timetype=obs.timetype,
    )


def _scan_aligned_caltable(obs, gains_per_scan, samples_per_scan=PAD_SCAN_NSAMPLES):
    """Build a Caltable whose times fall inside the first len(gains_per_scan)
    scans of `obs` (so scan_avg's per-scan bucketing finds the samples).

    `gains_per_scan` may be scalars (one constant per scan, broadcast across
    its samples) or already shape (n_scans, samples_per_scan).
    """
    obs2 = obs.copy()
    obs2.add_scans()
    assert len(obs2.scans) >= len(gains_per_scan), (
        f"obs has {len(obs2.scans)} scans but {len(gains_per_scan)} are needed"
    )
    scans = obs2.scans[:len(gains_per_scan)]
    inner = np.linspace(0.0, 1.0, samples_per_scan + 2)[1:-1]  # interior
    times = np.concatenate([
        scan_start + (scan_stop - scan_start) * inner
        for scan_start, scan_stop in scans
    ])
    gains_per_scan = np.asarray(gains_per_scan, dtype=complex)
    if gains_per_scan.ndim == 1:
        gains = np.repeat(gains_per_scan, samples_per_scan)
    else:
        gains = gains_per_scan.reshape(-1)
    n = len(times)
    template = np.empty(n, dtype=DTCAL)
    template['time'] = times
    template['rscale'] = gains
    template['lscale'] = gains
    caldict = {site: template.copy().view(np.recarray) for site in obs.tarr['site']}
    return eh.caltable.Caltable(
        obs.ra, obs.dec, obs.rf, obs.bw, caldict, obs.tarr,
        source=obs.source, mjd=obs.mjd, timetype=obs.timetype,
    )


class TestPadScans:

    def test_endval_padding_inserts_endpoint_gains(self, obs_direct):
        # Constant gains within each scan; endval padding ⇒ the pre and post
        # rows must equal the first/last sample of the scan (= the constant).
        n_scans = len(PAD_SCAN_MEDIAN_GAINS)
        ct = _multi_scan_caltable(obs_direct, n_scans=n_scans,
                                  rscale=PAD_SCAN_ENDVAL_GAIN,
                                  lscale=PAD_SCAN_ENDVAL_GAIN)
        out = ct.pad_scans(maxdiff=PAD_SCAN_MAXDIFF_SEC, padtype='endval')
        block_len = PAD_SCAN_NSAMPLES + 2
        for site in out.data:
            assert len(out.data[site]) == n_scans * block_len
        out_r, out_l = _stack_gains(out)
        np.testing.assert_allclose(out_r, PAD_SCAN_ENDVAL_GAIN)
        np.testing.assert_allclose(out_l, PAD_SCAN_ENDVAL_GAIN)

    def test_median_padding_uses_per_scan_median(self, obs_direct):
        # Each scan gets its own constant gain ⇒ that constant is the median.
        gains = np.repeat(np.array(PAD_SCAN_MEDIAN_GAINS), PAD_SCAN_NSAMPLES)
        ct = _multi_scan_caltable(obs_direct, n_scans=len(PAD_SCAN_MEDIAN_GAINS),
                                  rscale=gains, lscale=gains)
        out = ct.pad_scans(maxdiff=PAD_SCAN_MAXDIFF_SEC, padtype='median')
        block_len = PAD_SCAN_NSAMPLES + 2
        # Each padded scan repeats its scan-median across all block_len rows.
        expected = np.repeat(np.array(PAD_SCAN_MEDIAN_GAINS), block_len)
        for site in out.data:
            np.testing.assert_allclose(out.data[site]['rscale'], expected)
            np.testing.assert_allclose(out.data[site]['lscale'], expected)


# ---------------------------------------------------------------------------
# Section 9: scan_avg
# ---------------------------------------------------------------------------


class TestScanAvg:

    # scan_avg emits one row per scan in obs.scans (regardless of caltable
    # coverage). Scans without caltable data come back as NaN, so the tests
    # below assert (i) the first n_scans rows recover the input values and
    # (ii) every other row is NaN.

    def test_incoherent_avg_recovers_per_scan_magnitudes(self, obs_direct):
        magnitudes = np.array(SCAN_INCOH_MAGNITUDES)
        n_scans = len(magnitudes)
        rng = np.random.default_rng(SEED_SCAN_AVG_PHASES)
        phases = rng.uniform(-np.pi, np.pi,
                             size=(n_scans, PAD_SCAN_NSAMPLES))
        gains = magnitudes[:, None] * np.exp(1j * phases)
        ct = _scan_aligned_caltable(obs_direct, gains)
        out = ct.scan_avg(obs_direct, incoherent=True)
        for site in out.data:
            rscale = out.data[site]['rscale']
            lscale = out.data[site]['lscale']
            np.testing.assert_allclose(np.abs(rscale[:n_scans]), magnitudes,
                                       rtol=INTERP_RTOL)
            np.testing.assert_allclose(np.abs(lscale[:n_scans]), magnitudes,
                                       rtol=INTERP_RTOL)
            assert np.all(np.isnan(rscale[n_scans:]))
            assert np.all(np.isnan(lscale[n_scans:]))

    def test_coherent_avg_keeps_phase(self, obs_direct):
        per_scan = np.array(SCAN_COH_GAINS)
        n_scans = len(per_scan)
        ct = _scan_aligned_caltable(obs_direct, per_scan)
        out = ct.scan_avg(obs_direct, incoherent=False)
        for site in out.data:
            rscale = out.data[site]['rscale']
            lscale = out.data[site]['lscale']
            np.testing.assert_allclose(rscale[:n_scans], per_scan,
                                       rtol=INTERP_RTOL)
            np.testing.assert_allclose(lscale[:n_scans], per_scan,
                                       rtol=INTERP_RTOL)
            assert np.all(np.isnan(rscale[n_scans:]))
            assert np.all(np.isnan(lscale[n_scans:]))


# ---------------------------------------------------------------------------
# Section 10: merge
# ---------------------------------------------------------------------------


class TestMerge:

    def test_two_constant_caltables_multiply(self, obs_direct,
                                             constant_gain_caltable_factory):
        # Caltable.merge interpolates each input over the union of times and
        # multiplies pointwise. Two flat caltables a, b ⇒ merged gain = a*b.
        ct_a = constant_gain_caltable_factory(MERGE_GAIN_A)
        ct_b = constant_gain_caltable_factory(MERGE_GAIN_B)
        out = ct_a.merge([ct_b])
        out_r, out_l = _stack_gains(out)
        np.testing.assert_allclose(out_r, MERGE_GAIN_A * MERGE_GAIN_B,
                                   rtol=INTERP_RTOL)
        np.testing.assert_allclose(out_l, MERGE_GAIN_A * MERGE_GAIN_B,
                                   rtol=INTERP_RTOL)

    def test_disjoint_sites_are_unioned(self, obs_direct,
                                        constant_gain_caltable_factory):
        # Drop site x from ct_a and site y from ct_b ⇒ the merged caltable's
        # data dict must contain every site that appeared in either input.
        ct_a = constant_gain_caltable_factory(MERGE_GAIN_A)
        ct_b = constant_gain_caltable_factory(MERGE_GAIN_B)
        sites = list(ct_a.data.keys())
        x, y = sites[0], sites[1]
        ct_a.data.pop(x)
        ct_b.data.pop(y)
        out = ct_a.merge([ct_b])
        assert x in out.data
        assert y in out.data
