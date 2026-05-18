"""Tests for ehtim.statistics.averaging.

The three averaging routines historically lived in
`ehtim.statistics.dataframes` (pandas-based) and propagated visibility errors
via `sqrt(mean(sigma_i**2))`.  The migration to a pandas-free
`ehtim.statistics.averaging` swapped that formula for the correct
inverse-variance combination

    sigma_avg = 1 / sqrt(sum(1/sigma_i**2)).

Visibility values are also combined with inverse-variance weights by default
(``invvar_avg=True``); ``invvar_avg=False`` reproduces the legacy direct
(unweighted) complex mean.

This file pins both the migration (pandas-free, matching record-array
contracts) and the two weighting modes.
"""

import numpy as np
import pytest

import ehtim as eh
import ehtim.const_def as ehc
from ehtim.statistics.averaging import (
    coh_avg_vis,
    coh_moving_avg_vis,
    incoh_avg_vis,
)

# ---------------------------------------------------------------------------
# Synthetic Obsdata fixture
# ---------------------------------------------------------------------------


_STATIONS = (b"AA", b"AP", b"LM")


def _make_obs(times_hr, baselines=None, vis_fn=None, sigma=0.1, polrep="stokes"):
    """Build a minimal Obsdata with caller-specified times + visibility values.

    `baselines` is a list of (t1, t2) pairs of bytes-strings; one row per
    timestamp uses each baseline.  Defaults to a single baseline (AA, AP).
    `vis_fn(t, t1, t2)` returns the complex Stokes-I (or RR) visibility for
    each (time, baseline).  Defaults to `1+0j`.
    """
    if baselines is None:
        baselines = [(b"AA", b"AP")]
    if vis_fn is None:
        def vis_fn(t, t1, t2):
            return 1.0 + 0.0j
    times_hr = np.asarray(times_hr, dtype=float)
    n_times = len(times_hr)
    n_bls = len(baselines)
    n = n_times * n_bls

    if polrep == "stokes":
        dtype = ehc.DTPOL_STOKES
        vis_fields = ("vis", "qvis", "uvis", "vvis")
        sig_fields = ("sigma", "qsigma", "usigma", "vsigma")
    else:
        dtype = ehc.DTPOL_CIRC
        vis_fields = ("rrvis", "llvis", "rlvis", "lrvis")
        sig_fields = ("rrsigma", "llsigma", "rlsigma", "lrsigma")
    data = np.zeros(n, dtype=dtype)
    sigma_arr = np.broadcast_to(np.asarray(sigma, dtype=float), (n_times,))
    idx = 0
    for t1, t2 in baselines:
        sl = slice(idx, idx + n_times)
        data["time"][sl] = times_hr
        data["tint"][sl] = 1.0
        data["t1"][sl] = t1
        data["t2"][sl] = t2
        data["u"][sl] = 1e9
        data["v"][sl] = 1e9
        for vf in vis_fields:
            data[vf][sl] = [vis_fn(t, t1, t2) for t in times_hr]
        for sf in sig_fields:
            data[sf][sl] = sigma_arr
        idx += n_times

    tarr = np.zeros(len(_STATIONS), dtype=ehc.DTARR)
    for i, site in enumerate(_STATIONS):
        tarr[i]["site"] = site
        tarr[i]["sefdr"] = 1.0
        tarr[i]["sefdl"] = 1.0

    obs = eh.obsdata.Obsdata(
        ra=17.761, dec=-29.0, rf=230e9, bw=4e9,
        datatable=data, tarr=tarr, polrep=polrep,
    )
    return obs


# ---------------------------------------------------------------------------
# Section 1: coh_avg_vis — bin grouping + complex mean
# ---------------------------------------------------------------------------


def test_coh_avg_vis_dt_zero_returns_input():
    """dt=0, scan_avg=False is a no-op (returns input data)."""
    obs = _make_obs(times_hr=np.array([0.0, 0.01]))
    out = coh_avg_vis(obs, dt=0, scan_avg=False)
    assert out is obs.data


def test_coh_avg_vis_single_bin_complex_mean():
    """Two visibilities in the same 60s bin coherently average to their mean."""
    times = np.array([0.0, 5.0 / 3600.0])  # 0s and 5s after start
    obs = _make_obs(times_hr=times,
                    vis_fn=lambda t, t1, t2: 1.0 + 0.0j if t == 0.0 else 3.0 + 0.0j)
    out = coh_avg_vis(obs, dt=60.0)
    assert len(out) == 1
    assert out["vis"][0] == pytest.approx(2.0 + 0.0j)


def test_coh_avg_vis_two_bins_separated():
    """Two visibilities one minute apart land in different bins."""
    times = np.array([0.0, 70.0 / 3600.0])
    obs = _make_obs(times_hr=times,
                    vis_fn=lambda t, t1, t2: 1.0 + 0.0j if t == 0.0 else 3.0 + 0.0j)
    out = coh_avg_vis(obs, dt=60.0)
    assert len(out) == 2


def test_coh_avg_vis_tint_summed_in_bin():
    """tint accumulates inside the bin (units stay seconds)."""
    times = np.array([0.0, 5.0 / 3600.0, 10.0 / 3600.0])
    obs = _make_obs(times_hr=times)
    out = coh_avg_vis(obs, dt=60.0)
    assert out["tint"][0] == pytest.approx(3.0)  # three 1s integrations


# ---------------------------------------------------------------------------
# Section 2: inverse-variance sigma — the task-2.17 regression
# ---------------------------------------------------------------------------


def test_coh_avg_vis_uniform_sigma_drops_by_sqrt_n():
    """N visibilities with identical sigma s → averaged sigma = s/sqrt(N)."""
    times = np.array([0.0, 1.0 / 3600.0, 2.0 / 3600.0, 3.0 / 3600.0])  # 4 samples
    obs = _make_obs(times_hr=times, sigma=0.1)
    out = coh_avg_vis(obs, dt=60.0)
    assert out["sigma"][0] == pytest.approx(0.1 / np.sqrt(4), rel=1e-12)


def test_coh_avg_vis_inverse_variance_sigma_nonuniform():
    """Non-uniform sigmas combine via 1/sqrt(sum(1/sigma_i^2)), not sqrt(mean(sigma_i^2)).

    The old pandas-based meanerrF in dataframes.py used the latter (wrong).  For
    sigmas [0.1, 0.2] the correct inverse-variance combination is
    1/sqrt(1/0.01 + 1/0.04) = 1/sqrt(125) ≈ 0.0894, vs the old (wrong) value
    sqrt((0.01 + 0.04)/4) = 0.1118.  The 25% discrepancy makes this a strict
    regression test.
    """
    times = np.array([0.0, 1.0 / 3600.0])
    obs = _make_obs(times_hr=times)
    # Override sigmas by patching the recarray directly.
    obs.data["sigma"] = np.array([0.1, 0.2])
    obs.data["qsigma"] = np.array([0.1, 0.2])
    obs.data["usigma"] = np.array([0.1, 0.2])
    obs.data["vsigma"] = np.array([0.1, 0.2])

    out = coh_avg_vis(obs, dt=60.0)
    expected = 1.0 / np.sqrt(1.0 / 0.1**2 + 1.0 / 0.2**2)
    assert out["sigma"][0] == pytest.approx(expected, rel=1e-12)
    # Sanity: should NOT equal the historical buggy formula.
    wrong = np.sqrt((0.1**2 + 0.2**2) / 2**2)
    assert abs(out["sigma"][0] - wrong) > 1e-4


# ---------------------------------------------------------------------------
# Section 3: coh_moving_avg_vis — sliding-window
# ---------------------------------------------------------------------------


def test_coh_moving_avg_vis_constant_input_unchanged():
    """Moving average of a constant signal is the constant."""
    times = np.linspace(0.0, 0.05, 50)  # 50 samples spread over 3 minutes
    obs = _make_obs(times_hr=times)
    out = coh_moving_avg_vis(obs, dt=30.0)
    np.testing.assert_allclose(out["vis"], 1.0 + 0.0j, atol=1e-12)


def test_coh_moving_avg_vis_baselines_independent():
    """A second baseline's samples don't pollute the first baseline's window."""
    times = np.array([0.0, 1.0 / 3600.0])

    def vis_fn(t, t1, t2):
        # Baseline (AA, AP) carries 1+0j; baseline (AA, LM) carries 99+0j.
        return 99.0 + 0.0j if t2 == b"LM" else 1.0 + 0.0j

    obs = _make_obs(times_hr=times,
                    baselines=[(b"AA", b"AP"), (b"AA", b"LM")],
                    vis_fn=vis_fn)
    out = coh_moving_avg_vis(obs, dt=120.0)
    mask_AP = (out["t1"] == b"AA") & (out["t2"] == b"AP")
    mask_LM = (out["t1"] == b"AA") & (out["t2"] == b"LM")
    np.testing.assert_allclose(out["vis"][mask_AP], 1.0 + 0.0j, atol=1e-12)
    np.testing.assert_allclose(out["vis"][mask_LM], 99.0 + 0.0j, atol=1e-12)


# ---------------------------------------------------------------------------
# Section 4: incoh_avg_vis — high-SNR sanity
# ---------------------------------------------------------------------------


def test_incoh_avg_vis_high_snr_recovers_amplitude():
    """At very high SNR, incoherent averaging recovers the true amplitude."""
    times = np.array([0.0, 1.0 / 3600.0, 2.0 / 3600.0])
    obs = _make_obs(times_hr=times,
                    vis_fn=lambda t, t1, t2: 1.0 + 0.0j,
                    sigma=1e-6)  # very small noise
    out = incoh_avg_vis(obs, dt=60.0, debias=True)
    assert abs(out["vis"][0].real - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Section 5: end-to-end through Obsdata wrappers
# ---------------------------------------------------------------------------


def test_avg_coherent_through_obs_wrapper(obs_direct):
    """obs.avg_coherent() round-trips through ehavg.coh_avg_vis."""
    out = obs_direct.avg_coherent(60.0)
    assert out.data.dtype == obs_direct.data.dtype
    # Same or fewer rows than input.
    assert len(out.data) <= len(obs_direct.data)


def test_avg_incoherent_through_obs_wrapper(obs_direct):
    """obs.avg_incoherent() round-trips through ehavg.incoh_avg_vis."""
    out = obs_direct.avg_incoherent(60.0)
    assert out.data.dtype == obs_direct.data.dtype
    assert len(out.data) <= len(obs_direct.data)


# ---------------------------------------------------------------------------
# Section 6: parity vs the legacy pandas implementations in dataframes.py
#
# Per Andrew's suggestion: keep ehtim.statistics.dataframes around for now so
# we can A/B test the new pandas-free averaging against the original.  Once
# this PR has flushed through dev / dev-backend and PR #246's add_* removal
# has propagated, dataframes.py is deleted in a follow-up PR.
#
# Vis values are coherently averaged the same way in both implementations
# (complex mean) so they must match within float tolerance.  Sigmas
# intentionally differ: the new code uses inverse-variance combination, the
# old code used sqrt(mean(sigma_i**2)).  Both are asserted.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _legacy_ehdf():
    """Import the pandas-based legacy module; skip parity tests if pandas
    isn't installed in the env."""
    pytest.importorskip("pandas")
    from ehtim.statistics import dataframes as legacy
    return legacy


def test_parity_coh_avg_vis_complex_mean_matches_legacy(obs_direct, _legacy_ehdf):
    """Direct (invvar_avg=False) complex mean matches the legacy pandas
    implementation row-for-row, independently of how sigmas are distributed.

    obs_direct has uniform sigma within each (baseline, time-bucket) bin, so
    invvar_avg=True would also pass here by coincidence -- but invvar_avg=False
    is the genuinely equivalent comparison. The varied-sigma case is covered
    by test_parity_coh_avg_vis_varied_sigma below.
    """
    new = coh_avg_vis(obs_direct, dt=60.0, invvar_avg=False)
    old = _legacy_ehdf.coh_avg_vis(obs_direct, dt=60.0, return_type='rec')
    assert len(new) == len(old)
    # Sort both by (t1, t2, time) for an order-independent comparison.
    new_key = np.argsort(new, order=("t1", "t2", "time"))
    old_key = np.argsort(old, order=("t1", "t2", "time"))
    new_s, old_s = new[new_key], old[old_key]
    for vf in ("vis", "qvis", "uvis", "vvis"):
        np.testing.assert_allclose(new_s[vf], old_s[vf], rtol=1e-10, atol=1e-12,
                                   err_msg=f"vis mismatch in field {vf!r}")


def test_parity_coh_avg_vis_varied_sigma(_legacy_ehdf):
    """With varied intra-bin sigmas, invvar_avg=False matches the legacy
    direct mean exactly, while invvar_avg=True diverges as expected.

    Four samples in one 60 s bin on one baseline with varying vis values
    and varying sigmas exercise the weighting branch.
    """
    times = np.array([0.0, 10.0 / 3600.0, 20.0 / 3600.0, 30.0 / 3600.0])
    obs = _make_obs(times_hr=times,
                    vis_fn=lambda t, t1, t2: 1.0 + (t * 3600.0) * 0.01)
    obs.data["sigma"] = np.array([0.1, 0.4, 0.1, 0.4])
    obs.data["qsigma"] = np.array([0.1, 0.4, 0.1, 0.4])
    obs.data["usigma"] = np.array([0.1, 0.4, 0.1, 0.4])
    obs.data["vsigma"] = np.array([0.1, 0.4, 0.1, 0.4])

    new_direct = coh_avg_vis(obs, dt=60.0, invvar_avg=False)
    new_invvar = coh_avg_vis(obs, dt=60.0, invvar_avg=True)
    old = _legacy_ehdf.coh_avg_vis(obs, dt=60.0, return_type='rec')

    # Direct mean: bit-for-bit equal to legacy.
    np.testing.assert_allclose(
        new_direct["vis"], old["vis"], rtol=1e-12, atol=1e-14,
        err_msg="invvar_avg=False should match legacy on varied sigmas",
    )

    # Direct mean of vis values [1.0, 1.1, 1.2, 1.3] is 1.15.
    assert new_direct["vis"][0].real == pytest.approx(1.15, rel=1e-12)

    # Inverse-variance weighted mean: weights [100, 6.25, 100, 6.25] →
    # <V> = (1.0*100 + 1.1*6.25 + 1.2*100 + 1.3*6.25) / 212.5 ≈ 1.1.
    w = np.array([1/0.1**2, 1/0.4**2, 1/0.1**2, 1/0.4**2])
    expected_invvar = np.sum(np.array([1.0, 1.1, 1.2, 1.3]) * w) / np.sum(w)
    assert new_invvar["vis"][0].real == pytest.approx(expected_invvar, rel=1e-12)

    # The two means genuinely differ on varied-sigma data.
    assert abs(new_invvar["vis"][0] - new_direct["vis"][0]) > 1e-3


def test_parity_coh_avg_vis_sigma_differs_by_inverse_variance(_legacy_ehdf):
    """Sigma values intentionally differ — the new code fixes the bug.

    For two equal-time-bin samples with sigmas (0.1, 0.2), the new
    inverse-variance value is 1/sqrt(125) ≈ 0.0894; the legacy formula
    sqrt((0.01 + 0.04)/4) ≈ 0.1118.
    """
    times = np.array([0.0, 1.0 / 3600.0])
    obs = _make_obs(times_hr=times)
    obs.data["sigma"] = np.array([0.1, 0.2])
    obs.data["qsigma"] = np.array([0.1, 0.2])
    obs.data["usigma"] = np.array([0.1, 0.2])
    obs.data["vsigma"] = np.array([0.1, 0.2])

    new = coh_avg_vis(obs, dt=60.0)
    old = _legacy_ehdf.coh_avg_vis(obs, dt=60.0, return_type='rec')

    expected_new = 1.0 / np.sqrt(1.0 / 0.1**2 + 1.0 / 0.2**2)
    expected_old = np.sqrt((0.1**2 + 0.2**2) / 4)

    assert new["sigma"][0] == pytest.approx(expected_new, rel=1e-12)
    assert old["sigma"][0] == pytest.approx(expected_old, rel=1e-12)
    # And the two formulas must genuinely disagree on non-uniform sigmas.
    assert abs(new["sigma"][0] - old["sigma"][0]) > 1e-4


def test_parity_incoh_avg_vis_amplitude_matches_legacy(obs_direct, _legacy_ehdf):
    """Direct-mean (invvar_avg=False) incoherent amplitude matches legacy.

    Same caveat as the coherent parity test: with uniform intra-bin sigmas
    the invvar branch would also pass coincidentally. invvar_avg=False is
    the genuine equivalence.
    """
    new = incoh_avg_vis(obs_direct, dt=60.0, debias=True, invvar_avg=False)
    old = _legacy_ehdf.incoh_avg_vis(obs_direct, dt=60.0, debias=True, return_type='rec')
    assert len(new) == len(old)
    new_key = np.argsort(new, order=("t1", "t2", "time"))
    old_key = np.argsort(old, order=("t1", "t2", "time"))
    new_s, old_s = new[new_key], old[old_key]
    # Real parts hold the (debiased) amplitude.
    np.testing.assert_allclose(new_s["vis"].real, old_s["vis"].real,
                               rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# Section 7: additional coverage — scan_avg, polrep='circ', moving-avg sigma,
# err_type='measured'
# ---------------------------------------------------------------------------


def test_coh_avg_vis_scan_avg_groups_per_scan():
    """scan_avg=True puts samples into one bin per scan; rows outside any
    scan are dropped.

    Scan intervals follow pandas.cut semantics: ``(tstart, tstop]`` (open on
    the left, closed on the right), so a sample exactly on a scan boundary
    lands in the LATER scan only, not both.
    """
    # Six samples at 2, 5, 10, 15, 60, 75 seconds. Scans: (0, 15s] and (55s, 75s].
    # Sample at t=15s lands in scan 1 (right-inclusive); sample at t=75s in scan 2.
    times = np.array([2.0 / 3600.0, 5.0 / 3600.0, 10.0 / 3600.0,
                      15.0 / 3600.0, 60.0 / 3600.0, 75.0 / 3600.0])
    obs = _make_obs(times_hr=times)
    obs.scans = np.array([[0.0, 15.0 / 3600.0],
                          [55.0 / 3600.0, 75.0 / 3600.0]])
    out = coh_avg_vis(obs, dt=0, scan_avg=True)
    # Two scans, two output rows.
    assert len(out) == 2
    # tint per scan: scan 1 has 4 samples × 1s, scan 2 has 2 samples × 1s.
    tints = sorted(out["tint"])
    assert tints == [2.0, 4.0]


def test_coh_avg_vis_polrep_circ():
    """coh_avg_vis on a circular-polrep obs produces DTPOL_CIRC output."""
    times = np.array([0.0, 5.0 / 3600.0])
    obs = _make_obs(times_hr=times,
                    vis_fn=lambda t, t1, t2: 1.0 + 0.0j if t == 0.0 else 3.0 + 0.0j,
                    polrep="circ")
    out = coh_avg_vis(obs, dt=60.0)
    assert len(out) == 1
    assert out.dtype == ehc.DTPOL_CIRC
    # All four circ visibility fields receive the same complex mean.
    for vf in ("rrvis", "llvis", "rlvis", "lrvis"):
        assert out[vf][0] == pytest.approx(2.0 + 0.0j)


def test_coh_moving_avg_vis_inverse_variance_sigma():
    """Sliding-window sigma combines via inverse variance, not mean-of-squares.

    Two samples within a 120 s window with sigmas (0.1, 0.2) should reduce to
    1/sqrt(1/0.01 + 1/0.04) ≈ 0.0894 at both row positions.
    """
    times = np.array([0.0, 1.0 / 3600.0])  # 1 s apart
    obs = _make_obs(times_hr=times)
    obs.data["sigma"] = np.array([0.1, 0.2])
    obs.data["qsigma"] = np.array([0.1, 0.2])
    obs.data["usigma"] = np.array([0.1, 0.2])
    obs.data["vsigma"] = np.array([0.1, 0.2])

    out = coh_moving_avg_vis(obs, dt=120.0)
    expected = 1.0 / np.sqrt(1.0 / 0.1**2 + 1.0 / 0.2**2)
    np.testing.assert_allclose(out["sigma"], expected, rtol=1e-12)


def test_coh_avg_vis_err_type_measured_smoke():
    """The bootstrap err_type='measured' path runs end-to-end."""
    rng = np.random.default_rng(42)
    n = 8
    times = np.linspace(0.0, 7.0 / 3600.0, n)  # 8 samples within 60 s
    # Visibility = 1+0j with Gaussian noise of std 0.05.
    real_noise = rng.normal(0.0, 0.05, size=n)
    imag_noise = rng.normal(0.0, 0.05, size=n)

    def vis_fn(t, t1, t2):
        i = int(round(t * 3600.0))  # back to seconds index
        return (1.0 + real_noise[i]) + 1j * imag_noise[i]

    obs = _make_obs(times_hr=times, vis_fn=vis_fn, sigma=0.05)
    out = coh_avg_vis(obs, dt=60.0, err_type="measured", num_samples=200)
    assert len(out) == 1
    # Bootstrap sigma should be a positive finite number.
    assert np.isfinite(out["sigma"][0]) and out["sigma"][0] > 0
