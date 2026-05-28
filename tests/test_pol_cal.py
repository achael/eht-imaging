"""Tests for ehtim.calibrating.pol_cal.leakage_cal."""

import numpy as np
import pytest

import ehtim as eh
import ehtim.calibrating.pol_cal as polcal
import ehtim.observing.obs_simulate as simobs

# Tolerances + knobs.
LEAKAGE_RECOVERY_ATOL = 5e-2
LEAKAGE_TOL = 0.1
SHOW_SOLUTION = False

# Minimum cross-hand residual reduction. Sites whose PA coverage cannot lift
# the D-term/source-pol degeneracy cap how far the residual can drop; the
# observed reduction at EHT2017 + this declination is ~40%.
RESIDUAL_REDUCTION_FRAC = 0.3

# Full 24h obs so all sites get enough parallactic-angle coverage to lift the
# D-term / source-pol degeneracy.
TSTART_HR = 0.0
TSTOP_HR = 24.0

# Injected per-station D-term scale + seed.
INJECT_DTERM_AMP = 0.05
INJECT_DTERM_SEED = 123


@pytest.fixture(scope="module")
def obs_pol_dense(gauss_im_pol, observe):
    """Noise-free polarimetric obs, switched to circ for leakage_cal."""
    obs = observe(gauss_im_pol, ttype="direct",
                  tstart=TSTART_HR, tstop=TSTOP_HR)
    return obs.switch_polrep("circ")


@pytest.fixture(scope="module")
def obs_pol_dense_dterm_corrupted(obs_pol_dense):
    """obs_pol_dense with known per-station D-terms applied via Jones; tarr
    D-terms reset to 0 so leakage_cal solves from scratch."""
    rng = np.random.default_rng(INJECT_DTERM_SEED)
    n_sites = len(obs_pol_dense.tarr)
    dr = INJECT_DTERM_AMP * (rng.standard_normal(n_sites)
                             + 1j * rng.standard_normal(n_sites))
    dl = INJECT_DTERM_AMP * (rng.standard_normal(n_sites)
                             + 1j * rng.standard_normal(n_sites))

    seed = obs_pol_dense.copy()
    seed.tarr['dr'][:] = dr
    seed.tarr['dl'][:] = dl

    corrupt = seed.copy()
    corrupt.data = simobs.add_jones_and_noise(
        seed, add_th_noise=False,
        ampcal=True, phasecal=True, opacitycal=True, frcal=True, rlgaincal=True,
        dcal=False, dterm_offset=0.0, verbose=False,
    )
    corrupt.tarr['dr'][:] = 0
    corrupt.tarr['dl'][:] = 0
    corrupt.dcal = False
    return corrupt


def _cross_hand_residual(obs_a, obs_b):
    """L2 distance between two obs' cross-hand visibilities."""
    return np.sqrt(np.nansum(np.abs(obs_a.data['rlvis'] - obs_b.data['rlvis']) ** 2
                             + np.abs(obs_a.data['lrvis'] - obs_b.data['lrvis']) ** 2))


class TestLeakageCalIdentity:

    def test_clean_obs_recovers_dterms_near_zero(self, obs_pol_dense, gauss_im_pol):
        out = polcal.leakage_cal(
            obs_pol_dense, gauss_im_pol,
            leakage_tol=LEAKAGE_TOL, show_solution=SHOW_SOLUTION,
        )
        assert isinstance(out, eh.obsdata.Obsdata)
        np.testing.assert_allclose(np.abs(out.tarr['dr']), 0.0,
                                   atol=LEAKAGE_RECOVERY_ATOL)
        np.testing.assert_allclose(np.abs(out.tarr['dl']), 0.0,
                                   atol=LEAKAGE_RECOVERY_ATOL)


class TestLeakageCalInjectedDterms:

    def test_cross_hand_residual_drops_after_cal(
        self, obs_pol_dense_dterm_corrupted, obs_pol_dense, gauss_im_pol,
    ):
        # Gauge-invariant check: the cross-hand residual against the clean
        # source obs must shrink by at least RESIDUAL_REDUCTION_FRAC.
        corrupt = obs_pol_dense_dterm_corrupted
        before = _cross_hand_residual(corrupt, obs_pol_dense)
        out = polcal.leakage_cal(
            corrupt, gauss_im_pol,
            leakage_tol=LEAKAGE_TOL, show_solution=SHOW_SOLUTION,
        )
        after = _cross_hand_residual(out, obs_pol_dense)
        assert after < before * (1 - RESIDUAL_REDUCTION_FRAC)


class TestLeakageCalReturnType:

    def test_default_returns_obsdata_with_dcal_set(self, obs_pol_dense,
                                                   gauss_im_pol):
        out = polcal.leakage_cal(
            obs_pol_dense, gauss_im_pol,
            leakage_tol=LEAKAGE_TOL, show_solution=SHOW_SOLUTION,
        )
        assert isinstance(out, eh.obsdata.Obsdata)
        assert out.dcal is True
