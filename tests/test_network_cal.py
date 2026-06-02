"""Tests for ehtim.calibrating.network_cal."""

import numpy as np
import pytest

import ehtim as eh
import ehtim.calibrating.network_cal as netcal

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Constants used across the module
# ---------------------------------------------------------------------------

# network_cal is scipy.optimize.minimize-bound; tolerances are looser than
# the bit-clean tolerances used in the caltable tests.
NETCAL_RECOVERY_RTOL = 5e-2        # 5% on amp recovery

# network_cal serial path has the same dtype bug as self_cal; use 2 processes.
NETCAL_PROCESSES = 2

# Knobs pinned for deterministic runs.
NETCAL_GAIN_TOL = 0.5              # accept up to 50% deviation
NETCAL_PAD_AMP = 0.0
NETCAL_SHOW_SOLUTION = False
# scan_solutions=True: one solve per scan.
NETCAL_SCAN_SOLUTIONS = True

# Dense-coverage observation window mirroring the self_cal tests.
TSTART_HR = 14.0
TSTOP_HR = 16.0

# Injected per-station amplitude scale for the recovery test.
INJECT_AMP = 1.3


@pytest.fixture(scope="module")
def obs_dense(gauss_im, observe):
    """Noise-free observation of gauss_im over a dense-coverage window."""
    return observe(gauss_im, ttype="direct",
                   tstart=TSTART_HR, tstop=TSTOP_HR)


@pytest.fixture(scope="module")
def zbl(gauss_im):
    """Zero-baseline flux equals the source total flux."""
    return float(gauss_im.total_flux())


def _constant_gain_caltable(obs, rscale, lscale):
    """Build a flat per-site caltable spanning the obs times."""
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


def _zbl_mask(obs):
    """Boolean mask selecting rows whose uv-distance is within ZBLCUTOFF.

    network_cal only constrains gains for sites that share a zero-baseline
    pair; other sites stay at gain=1, so the rigorous post-cal assertion
    targets the zbl rows.
    """
    uvdist = np.sqrt(obs.data['u'] ** 2 + obs.data['v'] ** 2)
    return uvdist < netcal.ZBLCUTOFF


class TestNetworkCalIdentity:

    def test_noise_free_obs_preserves_zero_baseline_flux(self, obs_dense, zbl):
        # A noise-free obs already has |zbl rows| = zbl; network_cal must
        # leave them at the same value within recovery tolerance.
        out = netcal.network_cal(
            obs_dense, zbl, method='amp',
            gain_tol=NETCAL_GAIN_TOL, pad_amp=NETCAL_PAD_AMP,
            scan_solutions=NETCAL_SCAN_SOLUTIONS,
            processes=NETCAL_PROCESSES, show_solution=NETCAL_SHOW_SOLUTION,
        )
        mask = _zbl_mask(out)
        assert mask.any()
        np.testing.assert_allclose(np.abs(out.data['vis'][mask]), zbl,
                                   rtol=NETCAL_RECOVERY_RTOL)


# ---------------------------------------------------------------------------
# Section 2: Injected-gain recovery
# ---------------------------------------------------------------------------


class TestNetworkCalInjectedGains:

    def test_amp_only_restores_zero_baseline_flux(self, obs_dense, zbl):
        # Common per-station amp gain corrupts every visibility by INJECT_AMP**2.
        # network_cal solves the per-site gains constrained by the zbl rows;
        # the rigorous post-cal property is that |zbl rows| ≈ zbl.
        corrupt = _constant_gain_caltable(
            obs_dense, INJECT_AMP + 0j, INJECT_AMP + 0j,
        ).applycal(obs_dense, interp='nearest')
        recovered = netcal.network_cal(
            corrupt, zbl, method='amp',
            gain_tol=NETCAL_GAIN_TOL, pad_amp=NETCAL_PAD_AMP,
            scan_solutions=NETCAL_SCAN_SOLUTIONS,
            processes=NETCAL_PROCESSES, show_solution=NETCAL_SHOW_SOLUTION,
        )
        mask = _zbl_mask(recovered)
        assert mask.any()
        np.testing.assert_allclose(np.abs(recovered.data['vis'][mask]), zbl,
                                   rtol=NETCAL_RECOVERY_RTOL)


# ---------------------------------------------------------------------------
# Section 3: Return types
# ---------------------------------------------------------------------------


class TestNetworkCalReturnTypes:

    def test_default_returns_obsdata(self, obs_dense, zbl):
        out = netcal.network_cal(
            obs_dense, zbl, method='amp',
            gain_tol=NETCAL_GAIN_TOL, pad_amp=NETCAL_PAD_AMP,
            scan_solutions=NETCAL_SCAN_SOLUTIONS,
            processes=NETCAL_PROCESSES, show_solution=NETCAL_SHOW_SOLUTION,
        )
        assert isinstance(out, eh.obsdata.Obsdata)

    def test_caltable_true_returns_caltable(self, obs_dense, zbl):
        out = netcal.network_cal(
            obs_dense, zbl, method='amp',
            gain_tol=NETCAL_GAIN_TOL, pad_amp=NETCAL_PAD_AMP,
            scan_solutions=NETCAL_SCAN_SOLUTIONS,
            processes=NETCAL_PROCESSES, show_solution=NETCAL_SHOW_SOLUTION,
            caltable=True,
        )
        assert isinstance(out, eh.caltable.Caltable)


# ---------------------------------------------------------------------------
# Section 4: Method branches
# ---------------------------------------------------------------------------


class TestNetworkCalMethods:

    @pytest.mark.parametrize("method", ["amp", "phase", "both"])
    def test_each_method_runs_and_preserves_row_count(self, obs_dense, zbl,
                                                      method):
        out = netcal.network_cal(
            obs_dense, zbl, method=method,
            gain_tol=NETCAL_GAIN_TOL, pad_amp=NETCAL_PAD_AMP,
            scan_solutions=NETCAL_SCAN_SOLUTIONS,
            processes=NETCAL_PROCESSES, show_solution=NETCAL_SHOW_SOLUTION,
        )
        assert len(out.data) == len(obs_dense.data)
