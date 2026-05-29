"""Tests for ehtim.calibrating.polgains_cal."""

import numpy as np
import pytest

import ehtim as eh
import ehtim.calibrating.polgains_cal as pgcal

pytestmark = pytest.mark.slow

# Tolerances + knobs.
RESIDUAL_RTOL = 5e-2
POLGAINS_PAD_AMP = 0.0
POLGAINS_PROCESSES = 2
POLGAINS_SHOW_SOLUTION = False
SCAN_SOLUTIONS = True

# Reference station for the EHT2017 array; LCP on baselines to this station
# stays unchanged so the polgains solution has a well-defined gauge.
REFERENCE_SITE = 'ALMA'

# Dense polarimetric window.
TSTART_HR = 14.0
TSTOP_HR = 16.0


@pytest.fixture(scope="module")
def obs_pol_dense(gauss_im_pol, observe):
    """Noise-free polarimetric obs in circ polrep."""
    obs = observe(gauss_im_pol, ttype="direct",
                  tstart=TSTART_HR, tstop=TSTOP_HR)
    return obs.switch_polrep("circ")


class TestPolgainsCalIdentity:

    def test_clean_obs_preserves_parallel_hand_amplitudes(self, obs_pol_dense):
        # polgains_cal aligns RCP/LCP phases (and optionally amps). On a
        # clean obs with consistent feeds, the parallel-hand amplitudes
        # should be preserved within the optimizer's convergence noise.
        out = pgcal.polgains_cal(
            obs_pol_dense, reference=REFERENCE_SITE, method='phase',
            pad_amp=POLGAINS_PAD_AMP, scan_solutions=SCAN_SOLUTIONS,
            processes=POLGAINS_PROCESSES, show_solution=POLGAINS_SHOW_SOLUTION,
        )
        atol = RESIDUAL_RTOL * np.max(np.abs(obs_pol_dense.data['rrvis']))
        np.testing.assert_allclose(np.abs(out.data['rrvis']),
                                   np.abs(obs_pol_dense.data['rrvis']),
                                   atol=atol)
        np.testing.assert_allclose(np.abs(out.data['llvis']),
                                   np.abs(obs_pol_dense.data['llvis']),
                                   atol=atol)


class TestPolgainsCalReturnTypes:

    def test_default_returns_obsdata(self, obs_pol_dense):
        out = pgcal.polgains_cal(
            obs_pol_dense, reference=REFERENCE_SITE, method='phase',
            pad_amp=POLGAINS_PAD_AMP, scan_solutions=SCAN_SOLUTIONS,
            processes=POLGAINS_PROCESSES, show_solution=POLGAINS_SHOW_SOLUTION,
        )
        assert isinstance(out, eh.obsdata.Obsdata)

    def test_caltable_true_returns_caltable(self, obs_pol_dense):
        out = pgcal.polgains_cal(
            obs_pol_dense, reference=REFERENCE_SITE, method='phase',
            pad_amp=POLGAINS_PAD_AMP, scan_solutions=SCAN_SOLUTIONS,
            processes=POLGAINS_PROCESSES, show_solution=POLGAINS_SHOW_SOLUTION,
            caltable=True,
        )
        assert isinstance(out, eh.caltable.Caltable)


class TestPolgainsCalMethods:

    @pytest.mark.parametrize("method", ["phase", "both"])
    def test_each_method_runs_and_preserves_row_count(self, obs_pol_dense, method):
        out = pgcal.polgains_cal(
            obs_pol_dense, reference=REFERENCE_SITE, method=method,
            pad_amp=POLGAINS_PAD_AMP, scan_solutions=SCAN_SOLUTIONS,
            processes=POLGAINS_PROCESSES, show_solution=POLGAINS_SHOW_SOLUTION,
        )
        assert len(out.data) == len(obs_pol_dense.data)
