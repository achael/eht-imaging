"""Tests for ehtim.caltable.Caltable."""

import pickle
import warnings

import numpy as np
import pytest

import ehtim as eh
from ehtim.const_def import DTARR, DTCAL
from ehtim.warnings import MixedPolConventionWarning

# ---------------------------------------------------------------------------
# Constants used across the module
# ---------------------------------------------------------------------------

# Bit-clean numerical-equality tolerances for applycal scaling and gain
# round-trips. 1e-12 is the float-roundoff floor for products and inversions
# of double-precision complex numbers across this module.
# Gain used by the applycal D-term warning tests. Any well-conditioned value
# works; the tests assert on the warning and on output equality, not on it.
APPLYCAL_WARN_GAIN = 1.3 + 0.4j

# D-term persistence tests. DTERM_TIMES is deliberately not the gain time grid:
# the two tables are sampled independently. TIME_ATOL is in hours and covers
# the MJD round-trip, where a double holds ~1e-9 hr of resolution near MJD 5e4.
SEED_DTERM_ROUNDTRIP = 20260810
DTERM_TIMES = (0.5, 3.25)
SINGLE_ROW_TIME_HR = 2.0
TIME_ATOL = 1e-6

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
    """Building a table from a datadict, and the make_caltable helper."""

    def test_init_sets_scalar_attrs(self, unity_caltable, obs_direct):
        """The observation's scalar metadata is copied onto the table."""
        ct = unity_caltable
        assert ct.source == obs_direct.source
        assert ct.ra == obs_direct.ra
        assert ct.dec == obs_direct.dec
        assert ct.rf == obs_direct.rf
        assert ct.bw == obs_direct.bw
        assert ct.mjd == obs_direct.mjd
        assert ct.timetype == obs_direct.timetype

    def test_init_builds_tkey_from_tarr(self, unity_caltable):
        """tkey maps each site name to its row index in tarr."""
        ct = unity_caltable
        idx = np.fromiter((ct.tkey[s] for s in ct.tarr['site']),
                          dtype=int, count=len(ct.tarr))
        np.testing.assert_array_equal(idx, np.arange(len(ct.tarr)))

    def test_init_rejects_bad_timetype(self, obs_direct):
        """Only 'GMST' and 'UTC' are accepted as time conventions."""
        with pytest.raises(Exception, match="GMST"):
            eh.caltable.Caltable(
                obs_direct.ra, obs_direct.dec, obs_direct.rf, obs_direct.bw,
                {}, obs_direct.tarr, timetype='TAI',
            )

    def test_init_data_keys_match_sites(self, unity_caltable, obs_direct):
        """Every site in the array gets an entry in the gain table."""
        assert set(unity_caltable.data.keys()) == set(obs_direct.tarr['site'])

    def test_make_caltable_square_ntele_eq_ntimes(self, obs_direct):
        """make_caltable lays the flat gain list out as gains[site*ntimes + time].

        Square case, ntele == ntimes, where a transposed index would still agree;
        the rectangular test below is the one that can tell them apart.
        """
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
        """With no sites or times there is nothing to build, so it returns False."""
        assert eh.caltable.make_caltable(obs_direct, [], [], []) is False

    def test_make_caltable_rect_ntele_ne_ntimes(self, obs_direct):
        """Two sites and three times, where a transposed index would disagree."""
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
    """copy() is a deep copy: the result shares no array with the original."""

    def test_copy_returns_caltable_instance(self, unity_caltable):
        """copy() hands back a Caltable, not a bare dict."""
        assert isinstance(unity_caltable.copy(), eh.caltable.Caltable)

    def test_copy_preserves_scalar_attrs(self, unity_caltable):
        """The copy carries the same scalar metadata."""
        ct = unity_caltable.copy()
        for attr in ('source', 'ra', 'dec', 'rf', 'bw', 'mjd', 'timetype'):
            assert getattr(ct, attr) == getattr(unity_caltable, attr)

    def test_copy_data_is_independent(self, constant_gain_caltable_factory):
        """Mutating the copy's gains leaves the original alone."""
        original = constant_gain_caltable_factory(2.0 + 0j)
        cp = original.copy()
        first_site = next(iter(cp.data))
        cp.data[first_site]['rscale'] *= 10
        assert original.data[first_site]['rscale'][0] == 2 + 0j

    def test_copy_tarr_is_independent(self, unity_caltable):
        """The telescope array is deep-copied too, not shared."""
        cp = unity_caltable.copy()
        cp.tarr['sefdr'][0] = -1234.0
        assert unity_caltable.tarr['sefdr'][0] != -1234.0


# ---------------------------------------------------------------------------
# Section 3: Invert gains
# ---------------------------------------------------------------------------


class TestInvertGains:
    """invert_gains() replaces every gain by its reciprocal, in place."""

    def test_invert_unity_is_unity(self, unity_caltable):
        """Inverting unit gains leaves them at unity."""
        ct = unity_caltable.copy()
        ct.invert_gains()
        r_stack, l_stack = _stack_gains(ct)
        np.testing.assert_allclose(r_stack, 1 + 0j)
        np.testing.assert_allclose(l_stack, 1 + 0j)

    def test_invert_constant_g_yields_reciprocal(self, constant_gain_caltable_factory):
        """A constant gain g becomes 1/g."""
        ct = constant_gain_caltable_factory(CONST_REAL_GAIN)
        ct.invert_gains()
        r_stack, l_stack = _stack_gains(ct)
        np.testing.assert_allclose(r_stack, 1 / CONST_REAL_GAIN)
        np.testing.assert_allclose(l_stack, 1 / CONST_REAL_GAIN)

    def test_invert_twice_is_identity(self, injected_gain_caltable_factory):
        """Inverting twice returns the original gains, to roundoff."""
        ct = injected_gain_caltable_factory(seed=SEED_INVERT_ROUNDTRIP)
        ref_r, _ = _stack_gains(ct)
        ct.invert_gains()
        ct.invert_gains()
        r_stack, _ = _stack_gains(ct)
        np.testing.assert_allclose(r_stack, ref_r, rtol=BIT_CLEAN_RTOL)

    def test_invert_returns_self(self, unity_caltable):
        """It works in place and returns the same object, so calls can be chained."""
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
    """applycal scales each visibility by g_i * conj(g_j) for its two stations."""

    def test_real_gain_scales_amp_by_g_squared(self, obs_direct,
                                               constant_gain_caltable_factory):
        """A real gain shared by every station scales each visibility by |g|^2.

        Each baseline picks up g_i * conj(g_j).
        """
        g = CONST_REAL_GAIN
        out = constant_gain_caltable_factory(g).applycal(obs_direct, interp='nearest')
        for f in ('vis', 'qvis', 'uvis', 'vvis'):
            np.testing.assert_allclose(
                out.data[f], obs_direct.data[f] * np.abs(g) ** 2,
                rtol=BIT_CLEAN_RTOL, atol=BIT_CLEAN_ATOL,
            )

    def test_pure_phase_gain_preserves_amplitude(self, obs_direct):
        """A phase shared by every station leaves the amplitudes untouched.

        |g_i * conj(g_j)| = 1 for any phase common to all stations.
        """
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
    """Stations missing from the table are treated as already calibrated."""

    def test_missing_site_baselines_unscaled(self, obs_direct,
                                             constant_gain_caltable_factory):
        """A site with no gain entry falls back to gain 1.

        Its baselines then scale by the surviving station's g rather than g^2.
        """
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
    """applycal works in circular internally and hands back the caller's polrep."""

    def test_stokes_input_returns_stokes(self, obs_direct, unity_caltable):
        """A Stokes observation comes back as Stokes."""
        out = unity_caltable.applycal(obs_direct, interp='nearest')
        assert out.polrep == 'stokes'

    def test_circ_input_returns_circ(self, obs_direct, unity_caltable):
        """A circular observation comes back as circular."""
        obs_circ = obs_direct.switch_polrep('circ')
        out = unity_caltable.applycal(obs_circ, interp='nearest')
        assert out.polrep == 'circ'


class TestApplycalRejectsMismatchedTarr:
    """The table and the observation have to describe the same array."""

    def test_different_tarr_raises(self, obs_direct, unity_caltable):
        """One perturbed station coordinate makes applycal refuse rather than mis-apply."""
        obs_mut = obs_direct.copy()
        obs_mut.tarr['x'][0] += 1.0
        with pytest.raises(Exception, match="telescope array"):
            unity_caltable.applycal(obs_mut, interp='nearest')


class TestApplycalInterpModes:
    """The interp kwarg chooses how gains are sampled between cal-table times."""

    @pytest.mark.parametrize("interp", ["nearest", "linear"])
    def test_constant_gain_recovered_at_every_time(self, obs_direct, interp):
        """A flat table returns |g|^2 at every observation time, in either interp mode."""
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
        """Cubic interpolation needs four anchors; given them, a constant gain still comes back exactly."""
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
    """What happens outside the table's time span.

    applycal only documents extrapolate=True; the None case falls through to
    scipy's interp1d default, which is a NaN fill. These pin that inherited
    behaviour rather than treating it as a bug.
    """

    def test_extrapolate_none_yields_nan_only_outside_span(self, obs_direct):
        """Rows past the end of the table come back NaN, and only those rows."""
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
        """With extrapolate=True scipy fills the ends and every row stays finite."""
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
    """Gains survive a trip out to the per-site text files and back."""

    def test_save_caltable_load_caltable_roundtrip(self, obs_direct,
                                                   injected_gain_caltable_factory,
                                                   tmp_path):
        """Saving then loading recovers the gains and their times."""
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
        """With sqrt_gains the on-disk quantity is the squared gain.

        Comparing squares is the exact round-trip and also sidesteps the sqrt
        branch cut, which flips sign outside (-pi/2, pi/2).
        """
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
        """Caltable.save_txt is a thin wrapper, so it writes byte-identical files."""
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
    """enforce_positive rescales a whole gain curve so no station sits below min_gain."""

    def test_no_op_when_all_gains_above_min(self, unity_caltable):
        """Gains already above the floor are left alone."""
        out = unity_caltable.enforce_positive(method='median', min_gain=0.5,
                                              verbose=False)
        out_r, out_l = _stack_gains(out)
        in_r, in_l = _stack_gains(unity_caltable)
        np.testing.assert_allclose(out_r, in_r)
        np.testing.assert_allclose(out_l, in_l)

    def test_rescales_low_gains_above_threshold(self, constant_gain_caltable_factory):
        """A curve sitting below the floor is divided by its median, landing at |gain| = 1."""
        ct = constant_gain_caltable_factory(CONST_LOW_GAIN)
        out = ct.enforce_positive(method='median', min_gain=DEFAULT_MIN_GAIN,
                                  verbose=False)
        out_r, out_l = _stack_gains(out)
        np.testing.assert_allclose(np.abs(out_r), 1.0)
        np.testing.assert_allclose(np.abs(out_l), 1.0)

    def test_unknown_method_returns_unchanged_copy(self, constant_gain_caltable_factory):
        """An unrecognised method returns an unchanged copy instead of raising."""
        ct = constant_gain_caltable_factory(CONST_LOW_GAIN)
        out = ct.enforce_positive(method='nope', min_gain=DEFAULT_MIN_GAIN,
                                  verbose=False)
        first = next(iter(out.data))
        np.testing.assert_allclose(out.data[first]['rscale'],
                                   ct.data[first]['rscale'])

    def test_sites_subset_only_affects_listed(self, constant_gain_caltable_factory):
        """Passing sites= confines the rescaling to those stations."""
        ct = constant_gain_caltable_factory(CONST_LOW_GAIN)
        first, second = list(ct.data.keys())[:2]
        out = ct.enforce_positive(method='median', min_gain=DEFAULT_MIN_GAIN,
                                  sites=[first], verbose=False)
        np.testing.assert_allclose(np.abs(out.data[first]['rscale']), 1.0)
        np.testing.assert_allclose(np.abs(out.data[second]['rscale']),
                                   np.abs(CONST_LOW_GAIN))

    def test_returns_independent_copy(self, constant_gain_caltable_factory):
        """The result is a copy; mutating it does not reach the original."""
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
    """relaxed_interp1d wraps scipy's interp1d so degenerate inputs still work."""

    def test_single_point_falls_back_to_constant(self):
        """A one-point table is expanded to a flat two-point segment.

        The value is recovered at the anchor and on either side of it. Note the
        segment is only half a unit wide in x, which is why a one-row table is
        constant over a limited window rather than for all time.
        """
        f = eh.caltable.relaxed_interp1d(np.array([3.0]), np.array([2.5]),
                                         kind='linear')
        # The helper expands a 1-point input to a 2-point [x-0.5, x+0.5]
        # constant fan so the value is recovered at the anchor and nearby.
        assert f(3.0) == pytest.approx(2.5)
        assert f(2.7) == pytest.approx(2.5)
        assert f(3.3) == pytest.approx(2.5)

    def test_scalar_x_y_are_promoted_to_arrays(self):
        """Bare scalars are caught by the len() TypeError path and promoted to arrays."""
        f = eh.caltable.relaxed_interp1d(0.0, 1.5, kind='linear')
        assert f(0.0) == pytest.approx(1.5)

    def test_multi_point_matches_scipy_interp1d(self):
        """With more than one point it delegates straight to scipy, bit for bit."""
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
                         rscale=1.0 + 0j, lscale=1.0 + 0j, dterms=None):
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
        source=obs.source, mjd=obs.mjd, timetype=obs.timetype, dterms=dterms,
    )


def _scan_aligned_caltable(obs, gains_per_scan, samples_per_scan=PAD_SCAN_NSAMPLES,
                           dterms=None):
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
        source=obs.source, mjd=obs.mjd, timetype=obs.timetype, dterms=dterms,
    )


class TestPadScans:
    """pad_scans adds a row on each side of every scan.

    Gains are solved per chunk but applied by interpolation, so without the
    padding the interpolant ramps across the gaps between scans.
    """

    def test_endval_padding_inserts_endpoint_gains(self, obs_direct):
        """endval padding repeats each scan's first and last sample into the pad rows."""
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
        """median padding uses each scan's own median, not one median for the track."""
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
    """scan_avg collapses each scan to a single averaged gain.

    It emits one row per scan in obs.scans whatever the table covers, so scans
    with no cal data come back NaN. The tests check the covered scans recover
    their input and every later row is NaN.
    """

    def test_incoherent_avg_recovers_per_scan_magnitudes(self, obs_direct):
        """Averaging |g| across random phases recovers each scan's magnitude."""
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
        """Coherent averaging keeps the complex gain, phase included."""
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
    """merge multiplies two tables over the union of their time grids."""

    def test_two_constant_caltables_multiply(self, obs_direct,
                                             constant_gain_caltable_factory):
        """Two flat tables a and b merge to the pointwise product a*b."""
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
        """A site present in only one input still appears in the merged table."""
        ct_a = constant_gain_caltable_factory(MERGE_GAIN_A)
        ct_b = constant_gain_caltable_factory(MERGE_GAIN_B)
        sites = list(ct_a.data.keys())
        x, y = sites[0], sites[1]
        ct_a.data.pop(x)
        ct_b.data.pop(y)
        out = ct_a.merge([ct_b])
        assert x in out.data
        assert y in out.data


# ---------------------------------------------------------------------------
# Section 11: Gain / D-term storage split
# ---------------------------------------------------------------------------

# The row dtype from the era when D-terms shared the gain rows (PR #254),
# reproduced here so the migration paths can be driven from a caller's side.
_WELDED_DTCAL = [('time', 'f8'),
                 (('p1scale', 'rscale'), 'c16'), (('p2scale', 'lscale'), 'c16'),
                 (('d_p1', 'dr'), 'c16'), (('d_p2', 'dl'), 'c16')]

# Gains and leakage used across the split tests. The gain is complex and not
# unity so an applycal comparison has teeth; the D-terms are nonzero so the
# split actually extracts a table instead of dropping an all-zero one.
SPLIT_GAIN_R = 1.4 - 0.2j
SPLIT_GAIN_L = 0.8 + 0.5j
SPLIT_DR = 0.03 + 0.01j
SPLIT_DL = -0.02 + 0.04j
# A one-row D-term table is the storage form of a track-constant leakage.
SPLIT_DTERM_TIME = 0.0


def _welded_caldict(sites, times):
    """A pre-split datadict: gains and D-terms welded into one row dtype."""
    times = np.asarray(times, dtype=float)
    template = np.zeros(len(times), dtype=_WELDED_DTCAL)
    template['time'] = times
    template['rscale'] = SPLIT_GAIN_R
    template['lscale'] = SPLIT_GAIN_L
    template['dr'] = SPLIT_DR
    template['dl'] = SPLIT_DL
    return {site: template.copy() for site in sites}


def _clean_caldict(sites, times):
    """The gains-only equivalent of :func:`_welded_caldict`."""
    times = np.asarray(times, dtype=float)
    template = np.zeros(len(times), dtype=DTCAL)
    template['time'] = times
    template['rscale'] = SPLIT_GAIN_R
    template['lscale'] = SPLIT_GAIN_L
    return {site: template.copy() for site in sites}


def _span_times(obs):
    return [obs.data['time'].min() - 1.0, obs.data['time'].max() + 1.0]


def _first_sites(obs, n=2):
    """The first n site names of the observation's array."""
    return list(obs.tarr['site'])[:n]


class TestDataAliasesGains:
    """``data`` is the legacy name for the live gain table, not a snapshot."""

    def test_data_is_live_alias_of_gains(self, unity_caltable):
        """ct.data is the gains dict itself, not a copy of it."""
        assert unity_caltable.data is unity_caltable.gains

    def test_data_mutation_persists(self, constant_gain_caltable_factory):
        """Reassigning through ct.data reaches the table.

        This is the modeling_utils gain-assembly pattern: take .data, np.append
        into it per site, and expect the table itself to grow.
        """
        ct = constant_gain_caltable_factory(CONST_REAL_GAIN)
        site = next(iter(ct.data))
        n_before = len(ct.gains[site])
        caldict = ct.data
        caldict[site] = np.append(caldict[site], caldict[site])
        assert len(ct.gains[site]) == 2 * n_before

    def test_data_setter_replaces_gains(self, unity_caltable, obs_direct):
        """Assigning to ct.data swaps in a new gain table."""
        ct = unity_caltable.copy()
        replacement = _clean_caldict(_first_sites(obs_direct, 1),
                                     _span_times(obs_direct))
        ct.data = replacement
        assert ct.gains is replacement

    def test_dterms_defaults_empty(self, unity_caltable):
        """A table built without leakage has an empty dterms dict."""
        assert unity_caltable.dterms == {}


class TestConstructorSplit:
    """``__init__`` splits welded input and honours an explicit ``dterms=``."""

    def test_constructor_welded_datadict_autosplits(self, obs_direct):
        """A pre-split datadict is separated into the two tables on construction.

        The extracted D-terms keep the time column they were welded to, which is
        the faithful migration of an old table.
        """
        times = _span_times(obs_direct)
        sites = _first_sites(obs_direct)
        ct = eh.caltable.Caltable(
            obs_direct.ra, obs_direct.dec, obs_direct.rf, obs_direct.bw,
            _welded_caldict(sites, times), obs_direct.tarr,
            source=obs_direct.source, mjd=obs_direct.mjd,
        )
        for site in sites:
            assert ct.gains[site].dtype.names == ('time', 'rscale', 'lscale')
            np.testing.assert_array_equal(ct.gains[site]['rscale'], SPLIT_GAIN_R)
            np.testing.assert_array_equal(ct.gains[site]['lscale'], SPLIT_GAIN_L)
            np.testing.assert_array_equal(ct.dterms[site]['dr'], SPLIT_DR)
            np.testing.assert_array_equal(ct.dterms[site]['dl'], SPLIT_DL)
            # the extracted D-terms keep the time column they were welded to
            np.testing.assert_array_equal(ct.dterms[site]['time'], times)

    def test_constructor_dterms_kwarg(self, obs_direct, dterm_dict_factory):
        """Gains and D-terms passed separately keep their own independent time grids."""
        times = _span_times(obs_direct)
        sites = _first_sites(obs_direct)
        ct = eh.caltable.Caltable(
            obs_direct.ra, obs_direct.dec, obs_direct.rf, obs_direct.bw,
            _clean_caldict(sites, times), obs_direct.tarr,
            source=obs_direct.source, mjd=obs_direct.mjd,
            dterms=dterm_dict_factory(sites, times=(SPLIT_DTERM_TIME,),
                                      dr=SPLIT_DR, dl=SPLIT_DL),
        )
        for site in sites:
            # gains keep their own (two-point) grid, D-terms their one-row grid
            assert len(ct.gains[site]) == len(times)
            assert len(ct.dterms[site]) == 1
            assert ct.dterms[site]['time'][0] == SPLIT_DTERM_TIME
            np.testing.assert_array_equal(ct.gains[site]['rscale'], SPLIT_GAIN_R)

    def test_constructor_explicit_dterms_overrides_autosplit(self, obs_direct,
                                                             dterm_dict_factory):
        """An explicit dterms= wins over whatever a welded datadict would have contributed."""
        times = _span_times(obs_direct)
        sites = _first_sites(obs_direct, 1)
        site = sites[0]
        override = 0.5 + 0.5j
        ct = eh.caltable.Caltable(
            obs_direct.ra, obs_direct.dec, obs_direct.rf, obs_direct.bw,
            _welded_caldict(sites, times), obs_direct.tarr,
            source=obs_direct.source, mjd=obs_direct.mjd,
            dterms=dterm_dict_factory(sites, dr=override, dl=override),
        )
        # the explicit table replaces what the weld would have contributed
        assert len(ct.dterms[site]) == 1
        assert ct.dterms[site]['dr'][0] == override
        np.testing.assert_array_equal(ct.gains[site]['rscale'], SPLIT_GAIN_R)

    def test_constructor_single_record_list_site(self, obs_direct):
        """A site handed over as a bare list of records is normalised, not rejected.

        network_cal and polgains_cal leave this shape behind for a site seen in
        exactly one non-first scan.
        """
        times = _span_times(obs_direct)
        site = _first_sites(obs_direct, 1)[0]
        row = _clean_caldict([site], times)[site][0]
        ct = eh.caltable.Caltable(
            obs_direct.ra, obs_direct.dec, obs_direct.rf, obs_direct.bw,
            {site: [row]}, obs_direct.tarr,
            source=obs_direct.source, mjd=obs_direct.mjd,
        )
        assert len(ct.gains[site]) == 1
        assert ct.gains[site]['rscale'][0] == SPLIT_GAIN_R

    def test_constructor_rejects_non_dict_dterms(self, obs_direct, dterm_dict_factory):
        """A non-dict dterms= names the problem instead of failing inside dict()."""
        site = _first_sites(obs_direct, 1)[0]
        table = dterm_dict_factory([site])[site]     # the bare table, not a dict
        with pytest.raises(TypeError, match="dterms must be a dict"):
            eh.caltable.Caltable(
                obs_direct.ra, obs_direct.dec, obs_direct.rf, obs_direct.bw,
                _clean_caldict([site], _span_times(obs_direct)), obs_direct.tarr,
                source=obs_direct.source, mjd=obs_direct.mjd, dterms=table,
            )

    def test_constructor_non_dict_datadict_stored_verbatim(self, obs_direct):
        """A non-dict datadict is stored as-is, the way it always was."""
        site = _first_sites(obs_direct, 1)[0]
        arr = _clean_caldict([site], _span_times(obs_direct))[site]
        ct = eh.caltable.Caltable(
            obs_direct.ra, obs_direct.dec, obs_direct.rf, obs_direct.bw,
            arr, obs_direct.tarr, source=obs_direct.source, mjd=obs_direct.mjd,
        )
        assert ct.gains is arr
        assert ct.data is arr


class TestSplitStateRoundTrips:
    """copy / pickle carry both tables, and legacy states still migrate."""

    def _welded_caltable(self, obs):
        return eh.caltable.Caltable(
            obs.ra, obs.dec, obs.rf, obs.bw,
            _welded_caldict(_first_sites(obs), _span_times(obs)), obs.tarr,
            source=obs.source, mjd=obs.mjd,
        )

    def test_copy_preserves_dterms_independently(self, obs_direct):
        """copy() deep-copies the leakage table as well as the gains."""
        ct = self._welded_caltable(obs_direct)
        site = _first_sites(obs_direct, 1)[0]
        cp = ct.copy()
        np.testing.assert_array_equal(cp.dterms[site]['dr'], SPLIT_DR)
        cp.dterms[site]['dr'] *= 10
        np.testing.assert_array_equal(ct.dterms[site]['dr'], SPLIT_DR)

    def test_pickle_roundtrip_with_dterms(self, obs_direct):
        """Both tables survive a pickle round trip, and data comes back as an alias."""
        ct = self._welded_caltable(obs_direct)
        revived = pickle.loads(pickle.dumps(ct))
        assert set(revived.dterms) == set(ct.dterms)
        for site in ct.gains:
            np.testing.assert_array_equal(revived.gains[site], ct.gains[site])
            np.testing.assert_array_equal(revived.dterms[site], ct.dterms[site])
        # the alias survives the round trip as an alias, not a copy
        assert revived.data is revived.gains

    def test_setstate_non_dict_data_stored_verbatim(self, obs_direct):
        """A pre-split pickle whose 'data' was never a dict is passed through untouched."""
        site = _first_sites(obs_direct, 1)[0]
        arr = _clean_caldict([site], _span_times(obs_direct))[site]
        ct = self._welded_caltable(obs_direct)
        state = dict(ct.__dict__)
        state.pop('gains')
        state.pop('dterms')
        state['data'] = arr

        revived = eh.caltable.Caltable.__new__(eh.caltable.Caltable)
        revived.__setstate__(state)
        assert revived.gains is arr
        assert revived.dterms == {}

    def test_setstate_drops_legacy_data_key(self, obs_direct):
        """A legacy pickle's 'data' key is consumed, not left in the instance dict.

        data is a property now, and a property shadows an instance-dict entry of
        the same name, so leaving one behind would make those gains unreachable.
        """
        ct = self._welded_caltable(obs_direct)
        site = _first_sites(obs_direct, 1)[0]
        state = dict(ct.__dict__)
        state.pop('gains')
        state.pop('dterms')
        state['data'] = _clean_caldict([site], _span_times(obs_direct))

        revived = eh.caltable.Caltable.__new__(eh.caltable.Caltable)
        revived.__setstate__(state)
        assert 'data' not in revived.__dict__
        np.testing.assert_array_equal(revived.gains[site]['rscale'], SPLIT_GAIN_R)

    def test_setstate_leaves_the_caller_state_dict_alone(self, obs_direct):
        """__setstate__ migrates onto a copy, not onto the dict it was handed.

        Unpickling passes a throwaway dict, but a direct call passes one the
        caller still owns, so consuming its 'data' key in place is visible to
        them. It also breaks the next restore from that dict: the migration is
        skipped as already-done and both objects end up on one gains dict.

        The row arrays stay shared, which is the same zero-copy contract the
        constructor has -- hand it the same arrays and you get the same buffers.
        """
        site = _first_sites(obs_direct, 1)[0]
        ct = self._welded_caltable(obs_direct)
        state = dict(ct.__dict__)
        state.pop('gains')
        state.pop('dterms')
        state['data'] = _clean_caldict([site], _span_times(obs_direct))
        keys_before = set(state)

        first = eh.caltable.Caltable.__new__(eh.caltable.Caltable)
        first.__setstate__(state)
        assert set(state) == keys_before      # 'data' not consumed, no 'gains' added

        second = eh.caltable.Caltable.__new__(eh.caltable.Caltable)
        second.__setstate__(state)
        assert first.gains is not second.gains
        first.gains['EXTRASITE'] = None
        assert 'EXTRASITE' not in second.gains


class TestSplitLeavesApplycalUnchanged:
    """Splitting the input must not perturb gain application."""

    def test_applycal_welded_equals_clean_gains(self, obs_direct):
        """Splitting the input does not perturb gain application.

        A welded table carrying leakage calibrates identically to the equivalent
        gains-only table, since applycal has never applied D-terms.
        """
        times = _span_times(obs_direct)
        sites = obs_direct.tarr['site']
        kwargs = dict(source=obs_direct.source, mjd=obs_direct.mjd,
                      timetype=obs_direct.timetype)
        ct_welded = eh.caltable.Caltable(
            obs_direct.ra, obs_direct.dec, obs_direct.rf, obs_direct.bw,
            _welded_caldict(sites, times), obs_direct.tarr, **kwargs)
        ct_clean = eh.caltable.Caltable(
            obs_direct.ra, obs_direct.dec, obs_direct.rf, obs_direct.bw,
            _clean_caldict(sites, times), obs_direct.tarr, **kwargs)

        # the welded input carries leakage, the clean one does not
        assert ct_welded.dterms
        assert ct_clean.dterms == {}

        out_welded = ct_welded.applycal(obs_direct, interp='nearest')
        out_clean = ct_clean.applycal(obs_direct, interp='nearest')

        circ_w = out_welded.switch_polrep('circ')
        circ_c = out_clean.switch_polrep('circ')
        for field in ('rrvis', 'llvis', 'rlvis', 'lrvis'):
            np.testing.assert_array_equal(circ_w.data[field], circ_c.data[field])
        for field in ('rrsigma', 'llsigma', 'rlsigma', 'lrsigma'):
            np.testing.assert_array_equal(circ_w.data[field], circ_c.data[field])


class TestDtermPersistence:
    """D-terms round-trip through their own per-site files. The gain files keep
    their long-standing format, so directories written before the split still
    load and directories written now are still readable by older code."""

    def test_save_load_roundtrip_with_dterms(self, obs_direct,
                                             injected_gain_caltable_factory,
                                             dterm_dict_factory, tmp_path):
        """D-terms written on their own time grid come back on that same grid."""
        ct = injected_gain_caltable_factory(seed=SEED_DTERM_ROUNDTRIP)
        ct.dterms = dterm_dict_factory(list(ct.data), times=DTERM_TIMES)
        eh.caltable.save_caltable(ct, obs_direct, datadir=str(tmp_path))

        loaded = eh.caltable.load_caltable(obs_direct, str(tmp_path))
        assert set(loaded.dterms) == set(ct.dterms)
        for site in ct.dterms:
            np.testing.assert_allclose(loaded.dterms[site]['dr'],
                                       ct.dterms[site]['dr'], rtol=GAIN_RTOL)
            np.testing.assert_allclose(loaded.dterms[site]['dl'],
                                       ct.dterms[site]['dl'], rtol=GAIN_RTOL)
            np.testing.assert_allclose(loaded.dterms[site]['time'],
                                       ct.dterms[site]['time'], atol=TIME_ATOL)

    def test_single_row_dterm_file_roundtrip(self, obs_direct,
                                             injected_gain_caltable_factory,
                                             dterm_dict_factory, tmp_path):
        """A one-row file, the storage form of a track-constant leakage, round-trips."""
        ct = injected_gain_caltable_factory(seed=SEED_DTERM_ROUNDTRIP)
        ct.dterms = dterm_dict_factory(list(ct.data))
        eh.caltable.save_caltable(ct, obs_direct, datadir=str(tmp_path))

        loaded = eh.caltable.load_caltable(obs_direct, str(tmp_path))
        for site in ct.dterms:
            assert len(loaded.dterms[site]) == 1
            np.testing.assert_allclose(loaded.dterms[site]['dr'],
                                       ct.dterms[site]['dr'], rtol=GAIN_RTOL)

    def test_dterm_file_has_version_header(self, obs_direct,
                                           injected_gain_caltable_factory,
                                           dterm_dict_factory, tmp_path):
        """The D-term file leads with its version line."""
        ct = injected_gain_caltable_factory(seed=SEED_DTERM_ROUNDTRIP)
        site = _first_sites(obs_direct, 1)[0]
        ct.dterms = dterm_dict_factory([site])
        eh.caltable.save_caltable(ct, obs_direct, datadir=str(tmp_path))

        path = tmp_path / f"{ct.source}_{site}{eh.caltable.DTERM_FILE_SUFFIX}"
        first_line = path.read_text().splitlines()[0]
        assert first_line == eh.caltable.DTERM_FILE_HEADER

    def test_gains_only_dir_has_no_dterm_files(self, obs_direct,
                                               injected_gain_caltable_factory,
                                               tmp_path):
        """A table with no leakage writes exactly what it always did, and loads back empty."""
        ct = injected_gain_caltable_factory(seed=SEED_DTERM_ROUNDTRIP)
        assert ct.dterms == {}
        eh.caltable.save_caltable(ct, obs_direct, datadir=str(tmp_path))

        assert not list(tmp_path.glob("*" + eh.caltable.DTERM_FILE_SUFFIX))
        loaded = eh.caltable.load_caltable(obs_direct, str(tmp_path))
        assert loaded.dterms == {}

    def test_sqrt_gains_does_not_touch_dterms(self, obs_direct,
                                              injected_gain_caltable_factory,
                                              dterm_dict_factory, tmp_path):
        """sqrt_gains is a gain-side convention; leakage is written and read unchanged."""
        ct = injected_gain_caltable_factory(seed=SEED_DTERM_ROUNDTRIP)
        site = _first_sites(obs_direct, 1)[0]
        ct.dterms = dterm_dict_factory([site])
        eh.caltable.save_caltable(ct, obs_direct, datadir=str(tmp_path),
                                  sqrt_gains=True)

        loaded = eh.caltable.load_caltable(obs_direct, str(tmp_path),
                                           sqrt_gains=True)
        np.testing.assert_allclose(loaded.dterms[site]['dr'],
                                   ct.dterms[site]['dr'], rtol=GAIN_RTOL)

    def test_dterm_files_without_gains_returns_false(self, obs_direct,
                                                     injected_gain_caltable_factory,
                                                     dterm_dict_factory, tmp_path):
        """The directory gate is unchanged: no gain files still means no table."""
        ct = injected_gain_caltable_factory(seed=SEED_DTERM_ROUNDTRIP)
        ct.dterms = dterm_dict_factory(list(ct.data))
        eh.caltable.save_caltable(ct, obs_direct, datadir=str(tmp_path))
        for gain_file in tmp_path.glob(f"{ct.source}_*.txt"):
            if not gain_file.name.endswith(eh.caltable.DTERM_FILE_SUFFIX):
                gain_file.unlink()

        assert eh.caltable.load_caltable(obs_direct, str(tmp_path)) is False

    def test_gains_only_save_dir_byte_identical_to_pre_split(self, obs_direct,
                                                             tmp_path):
        """The gain files keep their historic headerless five-column layout.

        Every cal directory anyone already has on disk is in this format, so the
        split must leave it alone byte for byte. The times here land on whole
        MJDs and the gains on exact binary fractions, which makes the expected
        bytes something this test can spell out rather than recompute.
        """
        site = _first_sites(obs_direct, 1)[0]
        table = np.zeros(2, dtype=DTCAL)
        table['time'] = [0.0, 24.0]              # mjd exactly, then mjd + 1
        table['rscale'] = 2.0 + 0j
        table['lscale'] = 0.5 + 0j
        ct = eh.caltable.Caltable(
            obs_direct.ra, obs_direct.dec, obs_direct.rf, obs_direct.bw,
            {site: table}, obs_direct.tarr, source=obs_direct.source,
            mjd=obs_direct.mjd, timetype=obs_direct.timetype)
        eh.caltable.save_caltable(ct, obs_direct, datadir=str(tmp_path))

        mjd = float(obs_direct.mjd)
        written = (tmp_path / f"{obs_direct.source}_{site}.txt").read_text()
        assert written == (f"{mjd} 2.0 0.0 0.5 0.0\n"
                           f"{mjd + 1.0} 2.0 0.0 0.5 0.0\n")
        # and a leakage-free table grows no D-term sidecar
        assert not list(tmp_path.glob("*" + eh.caltable.DTERM_FILE_SUFFIX))


class TestLoadSingleRowAndLegacyGainFiles:
    """Regressions for the gain loader: one-row files used to fail with
    "format unknown" because a 1-D loadtxt result iterates as characters."""

    def test_single_row_gain_file_loads(self, obs_direct, tmp_path):
        """A one-row gain file loads.

        It used to fail with "format unknown": a 1-D loadtxt result iterates as
        characters, so len(row) measured a string length instead of a column count.
        """
        site = _first_sites(obs_direct, 1)[0]
        time_mjd = obs_direct.mjd + SINGLE_ROW_TIME_HR / 24.0
        path = tmp_path / f"{obs_direct.source}_{site}.txt"
        path.write_text(f"{time_mjd} 1.5 0.25 2.5 -0.5\n")

        loaded = eh.caltable.load_caltable(obs_direct, str(tmp_path))
        assert len(loaded.data[site]) == 1
        np.testing.assert_allclose(loaded.data[site]['rscale'][0], 1.5 + 0.25j)
        np.testing.assert_allclose(loaded.data[site]['lscale'][0], 2.5 - 0.5j)
        np.testing.assert_allclose(loaded.data[site]['time'][0],
                                   SINGLE_ROW_TIME_HR, atol=TIME_ATOL)

    def test_load_legacy_three_column_real_gains(self, obs_direct, tmp_path):
        """The oldest on-disk vintage, three columns of real-valued gains, still loads."""
        site = _first_sites(obs_direct, 1)[0]
        t0 = obs_direct.mjd + SINGLE_ROW_TIME_HR / 24.0
        t1 = obs_direct.mjd + (SINGLE_ROW_TIME_HR + 1.0) / 24.0
        path = tmp_path / f"{obs_direct.source}_{site}.txt"
        path.write_text(f"{t0} 1.5 2.5\n{t1} 3.5 4.5\n")

        loaded = eh.caltable.load_caltable(obs_direct, str(tmp_path))
        np.testing.assert_allclose(loaded.data[site]['rscale'], [1.5, 3.5])
        np.testing.assert_allclose(loaded.data[site]['lscale'], [2.5, 4.5])


class TestApplycalWarnsOnDterms:
    """applycal corrects gains only. If the table carries leakage, say so
    rather than letting it look like the data came back fully calibrated."""

    def test_applycal_warns_on_unapplied_dterms(self, obs_direct,
                                                constant_gain_caltable_factory,
                                                dterm_dict_factory):
        """A table carrying leakage says so, naming the sites it did not correct."""
        ct = constant_gain_caltable_factory(APPLYCAL_WARN_GAIN)
        site = _first_sites(obs_direct, 1)[0]
        ct.dterms = dterm_dict_factory([site])

        with pytest.warns(MixedPolConventionWarning, match=site):
            ct.applycal(obs_direct)

    def test_applycal_silent_without_dterms(self, obs_direct,
                                            constant_gain_caltable_factory):
        """A gains-only table calibrates without complaint."""
        ct = constant_gain_caltable_factory(APPLYCAL_WARN_GAIN)
        assert ct.dterms == {}

        with warnings.catch_warnings():
            warnings.simplefilter("error", MixedPolConventionWarning)
            ct.applycal(obs_direct)

    def test_warning_does_not_change_the_output(self, obs_direct,
                                                constant_gain_caltable_factory,
                                                dterm_dict_factory):
        """The warning is advisory: the data returned is the same gains-only result."""
        ct_plain = constant_gain_caltable_factory(APPLYCAL_WARN_GAIN)
        ct_leaky = constant_gain_caltable_factory(APPLYCAL_WARN_GAIN)
        ct_leaky.dterms = dterm_dict_factory(list(ct_leaky.data))

        out_plain = ct_plain.applycal(obs_direct)
        with pytest.warns(MixedPolConventionWarning):
            out_leaky = ct_leaky.applycal(obs_direct)

        for field in ('vis', 'sigma'):
            np.testing.assert_array_equal(out_plain.data[field],
                                          out_leaky.data[field])


class TestPlottingWithDterms:
    """The gain plot is unaffected by leakage riding along in the same table."""

    def test_plot_gains_smoke_with_dterms_present(self, obs_direct,
                                                 constant_gain_caltable_factory,
                                                 dterm_dict_factory):
        import matplotlib.pyplot as plt

        ct = constant_gain_caltable_factory(CONST_REAL_GAIN)
        ct.dterms = dterm_dict_factory(_first_sites(obs_direct, 1))

        original_backend = plt.get_backend()
        plt.switch_backend('Agg')                 # never open a window in CI
        try:
            axis = ct.plot_gains(_first_sites(obs_direct, 2), show=False)
            assert axis is not None
            assert len(axis.get_lines()) > 0
        finally:
            plt.close('all')
            plt.switch_backend(original_backend)


class TestTransformsCarryDterms:
    """pad_scans / scan_avg / merge resample gains; the leakage table rides
    along on its own time grid instead of being silently dropped."""

    def test_pad_scans_forwards_dterms_deepcopy(self, obs_direct, dterm_dict_factory):
        """Padding the gain grid carries the leakage through without resampling it."""
        site = _first_sites(obs_direct, 1)[0]
        dterms = dterm_dict_factory([site], dr=SPLIT_DR, dl=SPLIT_DL)
        ct = _multi_scan_caltable(obs_direct, n_scans=len(PAD_SCAN_MEDIAN_GAINS),
                                  rscale=PAD_SCAN_ENDVAL_GAIN,
                                  lscale=PAD_SCAN_ENDVAL_GAIN,
                                  dterms=dterms)
        out = ct.pad_scans(maxdiff=PAD_SCAN_MAXDIFF_SEC, padtype='endval')

        np.testing.assert_array_equal(out.dterms[site]['dr'], SPLIT_DR)
        # padding the gain grid must not resample leakage: one row in, one out
        assert len(out.dterms[site]) == 1
        # forwarded as a deep copy, so the output is not welded to the input
        out.dterms[site]['dr'] *= 10
        np.testing.assert_array_equal(ct.dterms[site]['dr'], SPLIT_DR)

    def test_scan_avg_forwards_dterms(self, obs_direct, dterm_dict_factory):
        """Averaging gains per scan leaves the leakage table intact."""
        site = _first_sites(obs_direct, 1)[0]
        dterms = dterm_dict_factory([site], dr=SPLIT_DR, dl=SPLIT_DL)
        ct = _scan_aligned_caltable(obs_direct, np.array(SCAN_COH_GAINS),
                                    dterms=dterms)
        out = ct.scan_avg(obs_direct, incoherent=False)

        np.testing.assert_array_equal(out.dterms[site]['dr'], SPLIT_DR)
        assert len(out.dterms[site]) == 1
        out.dterms[site]['dr'] *= 10
        np.testing.assert_array_equal(ct.dterms[site]['dr'], SPLIT_DR)

    def test_merge_forwards_one_sided_dterms_no_aliasing(
            self, obs_direct, constant_gain_caltable_factory, dterm_dict_factory):
        """Leakage from one side is copied into the merged table, not aliased to it."""
        ct_a = constant_gain_caltable_factory(MERGE_GAIN_A)
        ct_b = constant_gain_caltable_factory(MERGE_GAIN_B)
        site = _first_sites(obs_direct, 1)[0]
        ct_b.dterms = dterm_dict_factory([site], dr=SPLIT_DR, dl=SPLIT_DL)

        out = ct_a.merge([ct_b])

        np.testing.assert_array_equal(out.dterms[site]['dr'], SPLIT_DR)
        assert out.dterms[site] is not ct_b.dterms[site]
        out.dterms[site]['dr'] *= 10
        np.testing.assert_array_equal(ct_b.dterms[site]['dr'], SPLIT_DR)

    def test_merge_one_sided_dterms_leave_gains_alone(
            self, obs_direct, constant_gain_caltable_factory, dterm_dict_factory):
        """Carrying leakage across a merge does not disturb the gain multiplication."""
        ct_a = constant_gain_caltable_factory(MERGE_GAIN_A)
        ct_b = constant_gain_caltable_factory(MERGE_GAIN_B)
        ct_b.dterms = dterm_dict_factory(_first_sites(obs_direct, 1),
                                         dr=SPLIT_DR, dl=SPLIT_DL)
        out = ct_a.merge([ct_b])
        out_r, out_l = _stack_gains(out)
        np.testing.assert_allclose(out_r, MERGE_GAIN_A * MERGE_GAIN_B,
                                   rtol=INTERP_RTOL)
        np.testing.assert_allclose(out_l, MERGE_GAIN_A * MERGE_GAIN_B,
                                   rtol=INTERP_RTOL)

    def test_merge_both_sides_dterms_raises(
            self, obs_direct, constant_gain_caltable_factory, dterm_dict_factory):
        """Two leakage solutions for one site refuse to merge.

        They do compose, since J = G(I+D) is closed under multiplication, but not
        the way gains do and not commutatively. Raising beats a quietly wrong table.
        """
        ct_a = constant_gain_caltable_factory(MERGE_GAIN_A)
        ct_b = constant_gain_caltable_factory(MERGE_GAIN_B)
        sites = _first_sites(obs_direct, 1)
        ct_a.dterms = dterm_dict_factory(sites, dr=SPLIT_DR, dl=SPLIT_DL)
        ct_b.dterms = dterm_dict_factory(sites, dr=SPLIT_DL, dl=SPLIT_DR)

        with pytest.raises(NotImplementedError, match="D-terms"):
            ct_a.merge([ct_b])

    def test_merge_disjoint_dterm_sites_both_kept(
            self, obs_direct, constant_gain_caltable_factory, dterm_dict_factory):
        """Leakage for different sites on either side is unioned."""
        ct_a = constant_gain_caltable_factory(MERGE_GAIN_A)
        ct_b = constant_gain_caltable_factory(MERGE_GAIN_B)
        site_a, site_b = _first_sites(obs_direct, 2)
        ct_a.dterms = dterm_dict_factory([site_a], dr=SPLIT_DR, dl=SPLIT_DL)
        ct_b.dterms = dterm_dict_factory([site_b], dr=SPLIT_DL, dl=SPLIT_DR)

        out = ct_a.merge([ct_b])

        assert set(out.dterms) == {site_a, site_b}
        np.testing.assert_array_equal(out.dterms[site_a]['dr'], SPLIT_DR)
        np.testing.assert_array_equal(out.dterms[site_b]['dr'], SPLIT_DL)

    def test_merge_keeps_tarr_row_for_dterm_only_site(
            self, obs_direct, constant_gain_caltable_factory, dterm_dict_factory):
        # A site can carry leakage with no gain rows. Its array row has to come
        # along, or save_caltable -- which walks tarr -- drops the solved
        # leakage without a word.
        ct_a = constant_gain_caltable_factory(MERGE_GAIN_A)
        ct_b = constant_gain_caltable_factory(MERGE_GAIN_B)

        foreign = np.zeros(1, dtype=DTARR)
        foreign['site'] = ['NEWSITE']
        foreign['feed_type'] = ['rl']
        ct_b.tarr = np.append(ct_b.tarr, foreign)
        ct_b.tkey = {s: i for i, s in enumerate(ct_b.tarr['site'])}
        ct_b.dterms = dterm_dict_factory(['NEWSITE'], dr=SPLIT_DR, dl=SPLIT_DL)

        out = ct_a.merge([ct_b])

        assert 'NEWSITE' in out.dterms
        assert 'NEWSITE' in list(out.tarr['site'])
        assert 'NEWSITE' not in out.gains          # leakage only, no gains
        # exactly one row, i.e. the gain loop did not append it a second time
        assert list(out.tarr['site']).count('NEWSITE') == 1

    def test_merge_dterm_site_absent_from_both_tarrs_is_tolerated(
            self, obs_direct, constant_gain_caltable_factory, dterm_dict_factory):
        # No array row exists to copy, so merge cannot invent one; it must not
        # raise either. Persisting such a site is a separate problem.
        ct_a = constant_gain_caltable_factory(MERGE_GAIN_A)
        ct_b = constant_gain_caltable_factory(MERGE_GAIN_B)
        ct_b.dterms = dterm_dict_factory(['GHOST'], dr=SPLIT_DR, dl=SPLIT_DL)

        out = ct_a.merge([ct_b])

        assert 'GHOST' in out.dterms
        assert 'GHOST' not in list(out.tarr['site'])

    def test_merge_disjoint_site_gains_not_aliased(
            self, obs_direct, constant_gain_caltable_factory):
        """A gain array adopted from the other table is copied, so it cannot be mutated back."""
        ct_a = constant_gain_caltable_factory(MERGE_GAIN_A)
        ct_b = constant_gain_caltable_factory(MERGE_GAIN_B)
        site = _first_sites(obs_direct, 1)[0]
        ct_a.data.pop(site)

        out = ct_a.merge([ct_b])

        assert out.data[site] is not ct_b.data[site]
        out.data[site]['rscale'] *= 10
        np.testing.assert_allclose(ct_b.data[site]['rscale'], MERGE_GAIN_B)
