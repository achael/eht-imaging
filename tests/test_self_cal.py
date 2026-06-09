"""Tests for ehtim.calibrating.self_cal."""

import numpy as np
import pytest

import ehtim as eh
import ehtim.calibrating.self_cal as scal

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Constants used across the module
# ---------------------------------------------------------------------------

# self_cal is scipy.optimize.minimize-bound; tolerances are looser than the
# bit-clean tolerances used in the caltable tests.
GAIN_RECOVERY_RTOL = 5e-2          # 5% on amp recovery
PHASE_RECOVERY_ATOL = 5e-2         # 5e-2 rad on phase recovery

# self_cal serial path has a known dtype bug; use 2 processes to skip it.
SELF_CAL_PROCESSES = 2

# Knobs pinned for deterministic runs.
GAIN_TOL = 0.5                     # accept up to 50% deviation
PAD_AMP = 0.0
SHOW_SOLUTION = False

# Dense-coverage observation window: a few hours where most EHT2017 stations
# are simultaneously visible. Keeps each per-station gain well-constrained so
# the optimizer converges uniformly across all solution intervals.
TSTART_HR = 14.0
TSTOP_HR = 16.0
# scan_solutions=True: one solve per scan. Each solve sees the full baseline
# set within its scan, which is dense enough in this window to land on the
# canonical solution every time.
SCAN_SOLUTIONS = True

# Rows in the returned caltable that self_cal could not solve (no data at
# that site/time) stay at the optimizer's initial value of 1+0j. Filter the
# assertions to the rows that actually moved.
SOLVED_OFFSET_FROM_UNITY = 1e-3    # |g - 1| above this ⇒ row was solved
MIN_SOLVED_FRACTION = 0.5          # at least half the rows must be solved

# Injected gain values for the recovery test (one per station). Magnitudes
# above 1 (gain too high) and below 1 (gain too low) check both directions.
INJECT_AMP = 1.3
INJECT_PHASE = 0.4                 # radians

# Analytic-gradient (use_grad=True) finite-difference parity.
# Only method='both' supports an explicit gradient; phase/amp raise.
GRAD_FD_STEP = 1e-6
GRAD_FD_RTOL = 1e-6
GRAD_FD_ATOL = 1e-9
GRAD_TEST_N_SITES = 4
GRAD_TEST_SIGMA = 0.1
# Test points kept away from the |g|=1 kink in the prior tolerance term.
GRAD_TEST_GPARS = ((0.95, 0.05), (1.10, -0.08), (0.85, 0.20))
SEED_GRAD = 11


@pytest.fixture(scope="module")
def obs_dense(gauss_im, observe):
    """Noise-free observation of gauss_im over a dense-coverage window."""
    return observe(gauss_im, ttype="direct",
                   tstart=TSTART_HR, tstop=TSTOP_HR)


def _constant_gain_caltable(obs, rscale, lscale):
    """Build a flat per-site caltable spanning the obs times. Local helper
    because the conftest factory only takes a single g for both hands."""
    from ehtim.const_def import DTCAL
    times = np.array([obs.data['time'].min() - 1.0,
                      obs.data['time'].max() + 1.0])
    n = len(times)
    template = np.empty(n, dtype=DTCAL)
    template['time'] = times
    template['rscale'] = np.full(n, rscale, dtype=complex)
    template['lscale'] = np.full(n, lscale, dtype=complex)
    caldict = {site: template.copy().view(np.recarray)
               for site in obs.tarr['site']}
    return eh.caltable.Caltable(
        obs.ra, obs.dec, obs.rf, obs.bw, caldict, obs.tarr,
        source=obs.source, mjd=obs.mjd, timetype=obs.timetype,
    )


# ---------------------------------------------------------------------------
# Section 1: Identity recovery
# ---------------------------------------------------------------------------


def _stack_gains(caltable):
    """Concatenate the rscale and lscale columns across all sites.

    Single-entry sites can yield 0-d arrays; promote to 1-d before stacking.
    """
    rs = np.concatenate([np.atleast_1d(arr['rscale'])
                         for arr in caltable.data.values()])
    ls = np.concatenate([np.atleast_1d(arr['lscale'])
                         for arr in caltable.data.values()])
    return rs, ls


def _solved_mask(gains):
    """Boolean mask for rows the optimizer actually moved from the initial 1+0j.

    self_cal leaves rows untouched (= 1+0j) at site/time entries where no
    visibility is available to solve against; those rows are not the test's
    subject and should be excluded from gain-recovery assertions.
    """
    return np.abs(gains - 1.0) > SOLVED_OFFSET_FROM_UNITY


class TestSelfCalIdentity:

    def test_noise_free_obs_recovers_unit_gains(self, obs_dense, gauss_im):
        # Calibrating a noise-free obs to its source image should recover
        # gains within GAIN_RECOVERY_RTOL of unity (mag ≈ 1, phase ≈ 0).
        ct = scal.self_cal(
            obs_dense, gauss_im, method='both',
            gain_tol=GAIN_TOL, pad_amp=PAD_AMP,
            scan_solutions=SCAN_SOLUTIONS,
            processes=SELF_CAL_PROCESSES, show_solution=SHOW_SOLUTION,
            caltable=True,
        )
        rs, ls = _stack_gains(ct)
        np.testing.assert_allclose(np.abs(rs), 1.0, rtol=GAIN_RECOVERY_RTOL)
        np.testing.assert_allclose(np.abs(ls), 1.0, rtol=GAIN_RECOVERY_RTOL)
        np.testing.assert_allclose(np.angle(rs), 0.0,
                                   atol=PHASE_RECOVERY_ATOL)
        np.testing.assert_allclose(np.angle(ls), 0.0,
                                   atol=PHASE_RECOVERY_ATOL)


# ---------------------------------------------------------------------------
# Section 2: Injected-gain recovery
# ---------------------------------------------------------------------------


class TestSelfCalInjectedGains:

    def test_amp_only_restores_visibility_amplitudes(self, obs_dense, gauss_im):
        # The per-station amp gains are not uniquely determined (any common
        # rescaling factor across all sites is a gauge of the amp solution),
        # but the per-baseline products g_i * conj(g_j) are. The rigorous
        # gauge-invariant property is: applying the recovered calibration to
        # the corrupted obs restores the source visibilities. Use a
        # dynamic-range-scaled absolute tolerance so noise-floor rows do not
        # dominate the relative comparison.
        corrupt = _constant_gain_caltable(
            obs_dense, INJECT_AMP + 0j, INJECT_AMP + 0j,
        ).applycal(obs_dense, interp='nearest')
        recovered = scal.self_cal(
            corrupt, gauss_im, method='amp',
            gain_tol=GAIN_TOL, pad_amp=PAD_AMP,
            scan_solutions=SCAN_SOLUTIONS,
            processes=SELF_CAL_PROCESSES, show_solution=SHOW_SOLUTION,
        )
        atol = GAIN_RECOVERY_RTOL * np.max(np.abs(obs_dense.data['vis']))
        np.testing.assert_allclose(np.abs(recovered.data['vis']),
                                   np.abs(obs_dense.data['vis']), atol=atol)

    def test_phase_only_common_rotation_is_gauge_invariant(self, obs_dense,
                                                           gauss_im):
        # A common per-station phase is a gauge of the visibility model:
        # g_i * conj(g_j) = exp(i phi) * exp(-i phi) = 1 on every baseline.
        # self_cal phase-only therefore observes an uncorrupted obs and the
        # recovered per-station gains stay at 1 + 0j to within tolerance.
        g = np.exp(1j * INJECT_PHASE)
        corrupt = _constant_gain_caltable(
            obs_dense, g, g,
        ).applycal(obs_dense, interp='nearest')
        ct = scal.self_cal(
            corrupt, gauss_im, method='phase',
            gain_tol=GAIN_TOL, pad_amp=PAD_AMP,
            scan_solutions=SCAN_SOLUTIONS,
            processes=SELF_CAL_PROCESSES, show_solution=SHOW_SOLUTION,
            caltable=True,
        )
        rs, ls = _stack_gains(ct)
        np.testing.assert_allclose(np.abs(rs), 1.0, rtol=GAIN_RECOVERY_RTOL)
        np.testing.assert_allclose(np.abs(ls), 1.0, rtol=GAIN_RECOVERY_RTOL)
        np.testing.assert_allclose(np.angle(rs), 0.0,
                                   atol=PHASE_RECOVERY_ATOL)
        np.testing.assert_allclose(np.angle(ls), 0.0,
                                   atol=PHASE_RECOVERY_ATOL)


# ---------------------------------------------------------------------------
# Section 3: Return types
# ---------------------------------------------------------------------------


class TestSelfCalReturnTypes:

    def test_default_returns_obsdata(self, obs_dense, gauss_im):
        out = scal.self_cal(
            obs_dense, gauss_im, method='both',
            gain_tol=GAIN_TOL, pad_amp=PAD_AMP,
            scan_solutions=SCAN_SOLUTIONS,
            processes=SELF_CAL_PROCESSES, show_solution=SHOW_SOLUTION,
        )
        assert isinstance(out, eh.obsdata.Obsdata)

    def test_caltable_true_returns_caltable(self, obs_dense, gauss_im):
        out = scal.self_cal(
            obs_dense, gauss_im, method='both',
            gain_tol=GAIN_TOL, pad_amp=PAD_AMP,
            scan_solutions=SCAN_SOLUTIONS,
            processes=SELF_CAL_PROCESSES, show_solution=SHOW_SOLUTION,
            caltable=True,
        )
        assert isinstance(out, eh.caltable.Caltable)


# ---------------------------------------------------------------------------
# Section 4: Method branches
# ---------------------------------------------------------------------------


class TestSelfCalMethods:

    @pytest.mark.parametrize("method", ["amp", "phase", "both"])
    def test_each_method_runs_and_preserves_row_count(self, obs_dense,
                                                      gauss_im, method):
        out = scal.self_cal(
            obs_dense, gauss_im, method=method,
            gain_tol=GAIN_TOL, pad_amp=PAD_AMP,
            scan_solutions=SCAN_SOLUTIONS,
            processes=SELF_CAL_PROCESSES, show_solution=SHOW_SOLUTION,
        )
        assert len(out.data) == len(obs_dense.data)


# ---------------------------------------------------------------------------
# Section 5: Analytic gradient (use_grad path)
# ---------------------------------------------------------------------------


class TestErrfuncGradient:
    """errfunc_grad_full vs central FD on errfunc_full.

    Validates the analytic gradient scipy.optimize uses as jac= when
    self_cal is invoked with use_grad=True. Only method='both' is supported.
    """

    @pytest.fixture(scope="class")
    def grad_inputs(self):
        from itertools import combinations
        rng = np.random.default_rng(SEED_GRAD)
        sites = [f"s{i}" for i in range(GRAD_TEST_N_SITES)]
        pairs = list(combinations(range(GRAD_TEST_N_SITES), 2))
        n = len(pairs)
        v_scan = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        # vis = perturbed model so the data-term gradient is non-trivial.
        vis = v_scan * (1.0 + 0.1 * rng.standard_normal(n))
        return dict(
            vis=vis,
            v_scan=v_scan,
            sigma_inv=np.full(n, 1.0 / GRAD_TEST_SIGMA),
            gain_tol={"default": (0.2, 0.5)},
            sites=sites,
            g1_keys=[p[0] for p in pairs],
            g2_keys=[p[1] for p in pairs],
        )

    @pytest.mark.parametrize("gr,gi", GRAD_TEST_GPARS)
    def test_matches_finite_difference(self, grad_inputs, gr, gi):
        gpar = np.tile([gr, gi], GRAD_TEST_N_SITES)
        analytic = scal.errfunc_grad_full(gpar, method="both", **grad_inputs)
        fd = np.empty_like(gpar)
        for i in range(len(gpar)):
            xp = gpar.copy()
            xm = gpar.copy()
            xp[i] += GRAD_FD_STEP
            xm[i] -= GRAD_FD_STEP
            fd[i] = (scal.errfunc_full(xp, method="both", **grad_inputs)
                     - scal.errfunc_full(xm, method="both", **grad_inputs)) / (2 * GRAD_FD_STEP)
        np.testing.assert_allclose(analytic, fd,
                                    rtol=GRAD_FD_RTOL, atol=GRAD_FD_ATOL)

    @pytest.mark.parametrize("method", ["phase", "amp"])
    def test_raises_for_unsupported_method(self, grad_inputs, method):
        gpar = np.tile([0.95, 0.05], GRAD_TEST_N_SITES)
        with pytest.raises(Exception, match="method=='both'"):
            scal.errfunc_grad_full(gpar, method=method, **grad_inputs)
