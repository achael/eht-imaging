"""Tests for ImagerRunState dataclass and Imager history mechanism."""

import dataclasses
import os

import numpy as np
import pytest

import ehtim as eh
from ehtim.imager import Imager, ImagerRunState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ARRAY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "arrays", "EHT2017.txt"
)


def _make_test_imager(**kwargs):
    """Build a minimal Imager suitable for fast tests."""
    im = eh.image.make_empty(32, 200 * eh.RADPERUAS, 17.761, -29.0, rf=230e9)
    im.imvec = np.zeros(32 * 32)
    im.imvec[32 * 16 + 16] = 1.0

    array = eh.array.load_txt(ARRAY_PATH)
    obs = im.observe(
        array, 300, 10, 0.1, 12.0, 4e9,
        add_th_noise=False, ampcal=True, phasecal=True,
        ttype="direct",
    )
    defaults = dict(
        data_term={"vis": 1},
        reg_term={"simple": 1, "flux": 100},
        ttype="direct",
        maxit=5,
    )
    defaults.update(kwargs)
    return Imager(obs, im, prior_im=im, **defaults)


# ---------------------------------------------------------------------------
# ImagerRunState dataclass
# ---------------------------------------------------------------------------

class TestImagerRunStateDataclass:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(ImagerRunState)

    def test_expected_fields(self):
        fields = {f.name for f in dataclasses.fields(ImagerRunState)}
        expected = {
            "out", "obslist", "init", "prior", "reg_term", "dat_term",
            "maxit", "stop", "pol", "flux", "pflux", "vflux", "clipfloor",
            "snrcut", "debias", "systematic_noise", "systematic_cphase_noise",
            "transform", "weighting", "maxset", "cp_uv_min",
            "reffreq", "mf", "mf_order", "mf_order_pol", "mf_rm", "mf_cm",
        }
        assert fields == expected


# ---------------------------------------------------------------------------
# Before any run
# ---------------------------------------------------------------------------

class TestBeforeAnyRun:
    @pytest.fixture
    def imager(self):
        return _make_test_imager()

    def test_nruns_zero(self, imager):
        assert imager.nruns == 0

    def test_history_empty(self, imager):
        assert imager._history == []

    def _check_prints_no_runs_and_returns_none(self, method, imager, capsys):
        result = method(imager)
        captured = capsys.readouterr()
        assert "No imager runs yet!" in captured.out
        assert result is None

    def test_out_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.out_last, imager, capsys)

    def test_prior_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.prior_last, imager, capsys)

    def test_init_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.init_last, imager, capsys)

    def test_reg_terms_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.reg_terms_last, imager, capsys)

    def test_dat_terms_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.dat_terms_last, imager, capsys)

    def test_pol_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.pol_last, imager, capsys)

    def test_flux_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.flux_last, imager, capsys)

    def test_pflux_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.pflux_last, imager, capsys)

    def test_vflux_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.vflux_last, imager, capsys)

    def test_maxit_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.maxit_last, imager, capsys)

    def test_stop_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.stop_last, imager, capsys)

    def test_debias_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.debias_last, imager, capsys)

    def test_debias_next_default_is_false(self, imager):
        """debias defaults to False (flipped from True per task 2.15)."""
        assert imager.debias_next is False

    def test_snrcut_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.snrcut_last, imager, capsys)

    def test_weighting_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.weighting_last, imager, capsys)

    def test_systematic_noise_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.systematic_noise_last, imager, capsys)

    def test_systematic_cphase_noise_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.systematic_cphase_noise_last, imager, capsys)

    def test_maxset_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.maxset_last, imager, capsys)

    def test_cp_uv_min_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.cp_uv_min_last, imager, capsys)

    def test_clipfloor_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.clipfloor_last, imager, capsys)

    def test_transform_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.transform_last, imager, capsys)

    def test_mf_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.mf_last, imager, capsys)

    def test_reffreq_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.reffreq_last, imager, capsys)

    def test_mf_order_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.mf_order_last, imager, capsys)

    def test_mf_order_pol_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.mf_order_pol_last, imager, capsys)

    def test_mf_rm_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.mf_rm_last, imager, capsys)

    def test_mf_cm_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.mf_cm_last, imager, capsys)

    def test_obslist_last_no_runs(self, imager, capsys):
        self._check_prints_no_runs_and_returns_none(Imager.obslist_last, imager, capsys)

    def test_obs_last_no_runs_prints_message(self, imager, capsys):
        # _last() returns None and prints; then [0] raises TypeError before
        # obs_last() can return — verify the message is still emitted
        with pytest.raises(TypeError):
            imager.obs_last()
        captured = capsys.readouterr()
        assert "No imager runs yet!" in captured.out


# ---------------------------------------------------------------------------
# After one run
# ---------------------------------------------------------------------------

class TestAfterOneRun:
    @pytest.fixture(scope="class")
    def imager_after_run(self):
        im = _make_test_imager(
            maxit=5,
            clipfloor=0.0,
            maxset=False,
            stop=1e-3,
            debias=True,
            snrcut=0.0,
            weighting="natural",
            systematic_noise=0.0,
            systematic_cphase_noise=0.0,
        )
        im.make_image(pol="I", grads=False)
        return im

    def test_nruns_is_one(self, imager_after_run):
        assert imager_after_run.nruns == 1

    def test_history_has_one_entry(self, imager_after_run):
        assert len(imager_after_run._history) == 1

    def test_history_entry_is_imager_run_state(self, imager_after_run):
        assert isinstance(imager_after_run._history[0], ImagerRunState)

    def test_out_last_returns_image(self, imager_after_run):
        out = imager_after_run.out_last()
        assert out is not None
        assert hasattr(out, "imvec")

    def test_pol_last_is_I(self, imager_after_run):
        assert imager_after_run.pol_last() == "I"

    def test_maxit_last(self, imager_after_run):
        assert imager_after_run.maxit_last() == 5

    def test_stop_last(self, imager_after_run):
        assert imager_after_run.stop_last() == pytest.approx(1e-3)

    def test_debias_last(self, imager_after_run):
        assert imager_after_run.debias_last() is True

    def test_weighting_last(self, imager_after_run):
        assert imager_after_run.weighting_last() == "natural"

    def test_systematic_noise_last(self, imager_after_run):
        assert imager_after_run.systematic_noise_last() == pytest.approx(0.0)

    def test_systematic_cphase_noise_last(self, imager_after_run):
        assert imager_after_run.systematic_cphase_noise_last() == pytest.approx(0.0)

    def test_clipfloor_last(self, imager_after_run):
        assert imager_after_run.clipfloor_last() == pytest.approx(0.0)

    def test_mf_last_is_false(self, imager_after_run):
        assert imager_after_run.mf_last() is False

    def test_mf_order_last_is_zero(self, imager_after_run):
        assert imager_after_run.mf_order_last() == 0

    def test_mf_order_pol_last_is_zero(self, imager_after_run):
        assert imager_after_run.mf_order_pol_last() == 0

    def test_mf_rm_last_is_zero(self, imager_after_run):
        assert imager_after_run.mf_rm_last() == 0

    def test_mf_cm_last_is_zero(self, imager_after_run):
        assert imager_after_run.mf_cm_last() == 0

    def test_reg_terms_last_is_dict(self, imager_after_run):
        reg = imager_after_run.reg_terms_last()
        assert isinstance(reg, dict)
        assert "simple" in reg
        assert "flux" in reg

    def test_dat_terms_last_is_dict(self, imager_after_run):
        dat = imager_after_run.dat_terms_last()
        assert isinstance(dat, dict)
        assert "vis" in dat

    def test_prior_last_is_image(self, imager_after_run):
        prior = imager_after_run.prior_last()
        assert prior is not None
        assert hasattr(prior, "imvec")

    def test_init_last_is_image(self, imager_after_run):
        init = imager_after_run.init_last()
        assert init is not None
        assert hasattr(init, "imvec")

    def test_obslist_last_returns_list(self, imager_after_run):
        obslist = imager_after_run.obslist_last()
        assert isinstance(obslist, list)
        assert len(obslist) == 1

    def test_obs_last_returns_single_obsdata(self, imager_after_run):
        obs = imager_after_run.obs_last()
        assert obs is not None
        # obslist_last() wraps it; obs_last() unwraps the first element
        assert obs is imager_after_run.obslist_last()[0]

    def test_snrcut_last_is_dict(self, imager_after_run):
        snrcut = imager_after_run.snrcut_last()
        assert isinstance(snrcut, dict)

    def test_reffreq_last_matches_image_rf(self, imager_after_run):
        assert imager_after_run.reffreq_last() == pytest.approx(230e9)

    def test_flux_last_is_positive(self, imager_after_run):
        assert imager_after_run.flux_last() > 0


# ---------------------------------------------------------------------------
# Bug-fix: maxset_last() must return bool, not a float / clipfloor value
# ---------------------------------------------------------------------------

class TestMaxsetBugFix:
    def test_maxset_last_is_bool_false_by_default(self):
        imager = _make_test_imager(clipfloor=0.5)
        imager.make_image(pol="I", grads=False)
        result = imager.maxset_last()
        assert isinstance(result, bool), (
            f"maxset_last() returned {result!r} (type {type(result).__name__}), expected bool"
        )
        assert result is False

    def test_maxset_last_is_bool_true_when_set(self):
        imager = _make_test_imager(maxset=True)
        imager.make_image(pol="I", grads=False)
        result = imager.maxset_last()
        assert isinstance(result, bool)
        assert result is True

    def test_maxset_last_not_equal_to_clipfloor(self):
        clipfloor_val = 0.5
        imager = _make_test_imager(clipfloor=clipfloor_val, maxset=False)
        imager.make_image(pol="I", grads=False)
        # The old bug would have returned clipfloor_val (0.5) here
        assert imager.maxset_last() != clipfloor_val


# ---------------------------------------------------------------------------
# After two runs: _last() returns second run's values
# ---------------------------------------------------------------------------

class TestAfterTwoRuns:
    @pytest.fixture(scope="class")
    def imager_two_runs(self):
        imager = _make_test_imager(maxit=5)
        imager.make_image(pol="I", grads=False)
        # Change maxit and weighting for the second run
        imager.maxit_next = 3
        imager.weighting_next = "uniform"
        imager.make_image(pol="I", grads=False)
        return imager

    def test_nruns_is_two(self, imager_two_runs):
        assert imager_two_runs.nruns == 2

    def test_history_has_two_entries(self, imager_two_runs):
        assert len(imager_two_runs._history) == 2

    def test_both_history_entries_are_imager_run_state(self, imager_two_runs):
        for entry in imager_two_runs._history:
            assert isinstance(entry, ImagerRunState)

    def test_maxit_last_reflects_second_run(self, imager_two_runs):
        assert imager_two_runs.maxit_last() == 3

    def test_maxit_first_run_was_five(self, imager_two_runs):
        assert imager_two_runs._history[0].maxit == 5

    def test_weighting_last_reflects_second_run(self, imager_two_runs):
        assert imager_two_runs.weighting_last() == "uniform"

    def test_weighting_first_run_was_natural(self, imager_two_runs):
        assert imager_two_runs._history[0].weighting == "natural"

    def test_out_last_differs_between_runs(self, imager_two_runs):
        # The two out images are separate objects
        assert imager_two_runs._history[0].out is not imager_two_runs._history[1].out

    def test_pol_last_is_I_for_both_runs(self, imager_two_runs):
        assert imager_two_runs._history[0].pol == "I"
        assert imager_two_runs._history[1].pol == "I"


# ---------------------------------------------------------------------------
# obs_last() vs obslist_last() distinction
# ---------------------------------------------------------------------------

class TestObsLastVsObslistLast:
    @pytest.fixture(scope="class")
    def imager_run(self):
        im = _make_test_imager()
        im.make_image(pol="I", grads=False)
        return im

    def test_obslist_last_returns_list(self, imager_run):
        obslist = imager_run.obslist_last()
        assert isinstance(obslist, list)

    def test_obs_last_returns_first_element_of_obslist(self, imager_run):
        obs = imager_run.obs_last()
        obslist = imager_run.obslist_last()
        assert obs is obslist[0]

    def test_obs_last_has_data(self, imager_run):
        obs = imager_run.obs_last()
        assert hasattr(obs, "data")
        assert len(obs.data) > 0
