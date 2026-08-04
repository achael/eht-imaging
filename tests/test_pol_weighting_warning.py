"""Data weighting is applied to the Stokes-I terms only, and says so.

`chisqdata_pvis` / `chisqdata_m` / `chisqdata_vvis` take the standard weighting kwargs so
the dispatcher can pass them uniformly, and then use none of them. `Imager` builds its
`snrcut` dict with keys for the polarimetric terms too, so `snrcut={'pvis': 5}` looks
supported and silently does nothing.

Weighting the polarimetric data is a separate change: it would move every polarimetric
result, so it needs its own validation. These tests pin the warning, and pin the gap it
describes so the warning cannot quietly outlive it.
"""
import numpy as np
import pytest

import ehtim as eh
from ehtim.warnings import PolWeightingIgnoredWarning

POL_TERMS = ["pvis", "m", "vvis"]


@pytest.fixture(scope="module")
def pol_obs(eht_array, gauss_im):
    im = gauss_im.copy().add_random_pol(0.2, 0.05, seed=3)
    obs = im.observe(eht_array, 60, 600, 0, 24, 4e9, ampcal=True, phasecal=True,
                     ttype="direct", add_th_noise=True, seed=42)
    return im, obs


@pytest.fixture
def make_pol_imager(pol_obs, gauss_prior):
    im, obs = pol_obs

    def build(**kwargs):
        return eh.imager.Imager(obs, gauss_prior, prior_im=gauss_prior,
                                flux=im.total_flux(), data_term={"amp": 1, "pvis": 1},
                                reg_term={"simple": 1, "msimple": 1}, ttype="direct",
                                pol="IP", maxit=2, epsilon_tv=1e-10, **kwargs)
    return build


class TestTheWarningFires:
    @pytest.mark.parametrize("kwargs", [
        {"snrcut": {"pvis": 5.0}},
        {"debias": True},
        {"systematic_noise": 0.1},
        {"weighting": "uniform"},
    ], ids=["snrcut", "debias", "systematic_noise", "weighting"])
    def test_a_weighting_setting_with_a_pol_term_warns(self, make_pol_imager, kwargs):
        with pytest.warns(PolWeightingIgnoredWarning, match="Stokes-I"):
            make_pol_imager(**kwargs).init_imager()

    def test_the_warning_names_the_offending_terms(self, make_pol_imager):
        with pytest.warns(PolWeightingIgnoredWarning, match=r"\['pvis'\]"):
            make_pol_imager(systematic_noise=0.1).init_imager()

    def test_an_snrcut_on_a_stokes_i_term_alone_does_not_warn(self, make_pol_imager,
                                                             recwarn):
        # snrcut is per term: cutting amp is honoured, so there is nothing to warn about
        make_pol_imager(snrcut={"amp": 5.0}).init_imager()
        assert not [w for w in recwarn if issubclass(w.category, PolWeightingIgnoredWarning)]

    def test_default_weighting_does_not_warn(self, make_pol_imager, recwarn):
        make_pol_imager().init_imager()
        assert not [w for w in recwarn if issubclass(w.category, PolWeightingIgnoredWarning)]

    def test_a_stokes_i_run_does_not_warn(self, pol_obs, gauss_prior, recwarn):
        # no polarimetric term active, so the weighting is fully honoured
        im, obs = pol_obs
        eh.imager.Imager(obs, gauss_prior, prior_im=gauss_prior, flux=im.total_flux(),
                         data_term={"amp": 1}, reg_term={"simple": 1}, ttype="direct",
                         pol="I", maxit=2, epsilon_tv=1e-10,
                         systematic_noise=0.1).init_imager()
        assert not [w for w in recwarn if issubclass(w.category, PolWeightingIgnoredWarning)]


class TestTheGapTheWarningDescribes:
    """Pin the behaviour itself, so the warning cannot outlive the thing it warns about."""

    @pytest.mark.parametrize("kwargs,expect_i_changes", [
        ({"snrcut": {"pvis": 5.0, "amp": 5.0}}, True),
        ({"systematic_noise": 0.1}, True),
        ({"weighting": "uniform"}, True),
    ], ids=["snrcut", "systematic_noise", "weighting"])
    def test_pol_sigmas_are_untouched_while_stokes_i_moves(self, make_pol_imager, kwargs,
                                                           expect_i_changes):
        base = make_pol_imager()
        base.init_imager()
        with pytest.warns(PolWeightingIgnoredWarning):
            weighted = make_pol_imager(**kwargs)
            weighted.init_imager()

        def sig(imgr, term):
            return np.asarray(imgr._data_tuples[term][1])

        assert np.array_equal(sig(base, "pvis"), sig(weighted, "pvis"))
        if expect_i_changes:
            assert not np.array_equal(sig(base, "amp"), sig(weighted, "amp"))

    def test_debias_leaves_the_pol_data_alone(self, make_pol_imager):
        # debias changes the data rather than the sigmas
        base = make_pol_imager()
        base.init_imager()
        with pytest.warns(PolWeightingIgnoredWarning):
            deb = make_pol_imager(debias=True)
            deb.init_imager()
        assert np.array_equal(np.asarray(base._data_tuples["pvis"][0]),
                              np.asarray(deb._data_tuples["pvis"][0]))
        assert not np.array_equal(np.asarray(base._data_tuples["amp"][0]),
                                  np.asarray(deb._data_tuples["amp"][0]))

    def test_the_snrcut_dict_still_offers_pol_keys(self, make_pol_imager):
        # the Imager invites snrcut={'pvis': ...}; that is why the warning exists
        imgr = make_pol_imager()
        assert set(POL_TERMS) <= set(imgr.snrcut_next)
