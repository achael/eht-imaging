"""The imager recomputes its data products when the settings that feed them change.

`Imager.__init__` builds the data tuples immediately, so every later change has to be
noticed or the run silently uses data built for different settings. Two ways that failed:
`check_params` returned early before the first `make_image`, so anything assigned as an
attribute was ignored, and it never compared several settings that do feed the tuples
(clipfloor, maxset, ttype, the Fourier-grid params).

The control tests matter as much as the rest. Recomputing whenever anything is touched
would pass every staleness test here and make each run rebuild the operators, so the
no-change and unrelated-change cases are pinned too.
"""
import numpy as np
import pytest

import ehtim as eh


def _n_amp(imgr):
    imgr.init_imager()
    return int(np.asarray(imgr._data_tuples["amp"][0]).size)


@pytest.fixture(scope="module")
def noisy_obs(eht_array, gauss_im):
    # thermal noise so snrcut and debias have something to bite on
    return gauss_im.observe(eht_array, 60, 600, 0, 24, 4e9, ampcal=True, phasecal=True,
                            ttype="direct", add_th_noise=True, seed=42)


@pytest.fixture
def make_imager(noisy_obs, gauss_im, gauss_prior):
    def build(**kwargs):
        return eh.imager.Imager(noisy_obs, gauss_prior, prior_im=gauss_prior,
                                flux=gauss_im.total_flux(), data_term={"amp": 1},
                                reg_term={"simple": 1}, ttype="direct", maxit=2,
                                epsilon_tv=1e-10, **kwargs)
    return build


class TestSettingsAppliedAfterConstruction:
    """Assigning an attribute must have the same effect as passing the kwarg."""

    def test_snrcut_and_debias_match_the_constructor(self, make_imager):
        via_kwargs = make_imager(debias=True, snrcut={"amp": 5.0})
        via_attrs = make_imager()
        via_attrs.debias_next = True
        via_attrs.snrcut_next = {**via_attrs.snrcut_next, "amp": 5.0}
        assert _n_amp(via_attrs) == _n_amp(via_kwargs)

    def test_an_snrcut_edited_in_place_is_noticed(self, make_imager):
        # the run history used to store the live dict, so this compared it against itself
        imgr = make_imager()
        before = _n_amp(imgr)
        imgr.make_image(show_updates=False)
        imgr.snrcut_next["amp"] = 5.0
        imgr.check_params()
        assert imgr._change_imgr_params
        assert _n_amp(imgr) < before

    @pytest.mark.parametrize("attr,value", [
        ("clipfloor_next", 0.02),
        ("maxset_next", True),
        ("weighting_next", "uniform"),
        ("systematic_noise_next", 0.05),
    ])
    def test_settings_that_feed_the_data_tuples_are_noticed(self, make_imager, attr, value):
        imgr = make_imager()
        imgr.make_image(show_updates=False)
        setattr(imgr, attr, value)
        imgr.check_params()
        assert imgr._change_imgr_params, f"{attr} left the data products stale"

    def test_a_transform_change_is_noticed(self, make_imager):
        # ttype selects the Fourier operator, so the cached one is the wrong kind
        imgr = make_imager()
        imgr.make_image(show_updates=False)
        imgr._config = imgr._config._replace(ttype="nfft")
        imgr.check_params()
        assert imgr._change_imgr_params

    def test_a_fourier_grid_change_is_noticed(self, make_imager):
        imgr = make_imager(ttype="nfft")
        imgr.make_image(show_updates=False)
        imgr._nfft_eps = 1e-6
        imgr.check_params()
        assert imgr._change_imgr_params


class TestNothingIsRecomputedWithoutCause:
    """A fix that recomputes unconditionally would satisfy everything above."""

    def test_repeated_checks_leave_the_products_alone(self, make_imager):
        imgr = make_imager()
        imgr.make_image(show_updates=False)
        for _ in range(3):
            imgr.check_params()
            assert not imgr._change_imgr_params

    def test_the_data_tuples_are_not_rebuilt_between_identical_runs(self, make_imager):
        imgr = make_imager()
        imgr.make_image(show_updates=False)
        before = imgr._data_tuples["amp"][0]
        imgr.make_image(show_updates=False)
        assert imgr._data_tuples["amp"][0] is before

    def test_a_regularizer_weight_change_does_not_rebuild_the_data(self, make_imager):
        # regularizers do not touch the data products
        imgr = make_imager()
        imgr.make_image(show_updates=False)
        before = imgr._data_tuples["amp"][0]
        imgr.reg_term_next = {"simple": 5}
        imgr.check_params()
        imgr.init_imager()
        assert imgr._data_tuples["amp"][0] is before


class TestMultifrequencyFlag:
    def test_a_constructor_set_mf_survives_make_image(self, eht_array, gauss_im):
        # make_image's mf default used to overwrite whatever the constructor was given
        im = gauss_im.copy().add_const_mf(1.0, 0)
        prior = im.blur_circ(40 * eh.RADPERUAS)
        obslist = [im.get_image_mf(nu).observe(eht_array, 600, 600, 0, 24, 4e9, ampcal=True,
                                               phasecal=True, ttype="direct",
                                               add_th_noise=False, seed=42)
                   for nu in (220e9, 240e9)]
        imgr = eh.imager.Imager(obslist, prior, prior_im=prior, flux=im.total_flux(),
                                data_term={"vis": 1}, reg_term={"simple": 1},
                                ttype="direct", maxit=2, mf=True, mf_order=1,
                                epsilon_tv=1e-10)
        assert imgr._config.mf
        imgr.make_image(show_updates=False)
        assert imgr._config.mf

    def test_make_image_can_still_turn_mf_off(self, make_imager):
        imgr = make_imager()
        imgr.make_image(mf=False, show_updates=False)
        assert not imgr._config.mf
