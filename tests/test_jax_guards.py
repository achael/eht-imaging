"""Capability guards for the jax, optax and sharded paths.

Not every transform and data term has a jax kernel. Where one is missing the run used to
die somewhere inside jax with a raw tracer or argument-binding error, which says nothing
about what the user should do instead. These tests pin the refusal: a ValueError naming
the unsupported thing and the alternative that works.

The controls matter as much as the refusals. A guard that is too broad would block
configurations that run fine today, so each rejection is paired with a case that must
still be accepted.
"""
import numpy as np
import pytest

import ehtim as eh
from ehtim.imaging.imager_backend import check_jax_supported

pytestmark = pytest.mark.jax

pytest.importorskip("optax")

EPSILON_TV = 1e-10


def _imager(obs, gauss_im, prior, ttype="direct", data_term=None):
    return eh.imager.Imager(obs, prior, prior_im=prior, flux=gauss_im.total_flux(),
                            data_term=data_term or {"vis": 1},
                            reg_term={"simple": 1, "tv": 1}, ttype=ttype,
                            maxit=2, epsilon_tv=EPSILON_TV)


# ============================== the helper on its own ==============================
class TestCheckJaxSupported:
    """Direct tests of the predicate, independent of how the Imager calls it."""

    @pytest.mark.parametrize("ttype", ["direct", "nfft"])
    def test_supported_transforms_pass(self, ttype):
        check_jax_supported(ttype, ["vis", "amp", "cphase", "logcamp"])

    def test_fast_transform_is_refused(self):
        with pytest.raises(ValueError, match="ttype='fast'"):
            check_jax_supported("fast", ["vis"])

    def test_the_refusal_names_a_working_alternative(self):
        # a guard that only says no is not much better than the tracer error it replaces
        with pytest.raises(ValueError, match="nfft"):
            check_jax_supported("fast", ["vis"])

    @pytest.mark.parametrize("term", ["cphase_diag", "logcamp_diag"])
    def test_diag_closure_terms_are_refused(self, term):
        with pytest.raises(ValueError, match=term):
            check_jax_supported("direct", [term])

    def test_it_reports_every_unsupported_term_at_once(self):
        # fixing them one error at a time would be tedious
        with pytest.raises(ValueError) as e:
            check_jax_supported("direct", ["vis", "cphase_diag", "logcamp_diag"])
        assert "cphase_diag" in str(e.value) and "logcamp_diag" in str(e.value)

    def test_the_undiagonalized_closures_still_pass(self):
        # cphase and logcamp are jaxified; only their _diag variants are not
        check_jax_supported("direct", ["cphase", "logcamp", "camp", "bs"])


# ============================== through the Imager ==============================
class TestImagerRefusesUnsupportedJaxRuns:
    def test_fast_ttype_with_use_jax(self, obs_direct, gauss_im, gauss_prior):
        with pytest.raises(ValueError, match="ttype='fast'"):
            _imager(obs_direct, gauss_im, gauss_prior, ttype="fast").make_image(
                use_jax=True, show_updates=False)

    def test_fast_ttype_on_the_optax_path_without_use_jax(self, obs_direct, gauss_im,
                                                          gauss_prior):
        # the subtle one: an optax optimizer builds the jax objective itself, so use_jax
        # is not what decides whether jax runs. Keying the guard on the flag alone would
        # let this straight through to the tracer error.
        with pytest.raises(ValueError, match="ttype='fast'"):
            _imager(obs_direct, gauss_im, gauss_prior, ttype="fast").make_image(
                optimizer="optax-lbfgs", show_updates=False)

    @pytest.mark.parametrize("term", ["cphase_diag", "logcamp_diag"])
    def test_diag_dataterm_with_use_jax(self, obs_direct, gauss_im, gauss_prior, term):
        with pytest.raises(ValueError, match=term):
            _imager(obs_direct, gauss_im, gauss_prior, data_term={term: 1}).make_image(
                use_jax=True, show_updates=False)

    def test_fast_ttype_on_numpy_is_untouched(self, obs_direct, gauss_im, gauss_prior):
        # control: the guard must not fire on the default numpy path, where fast works
        with pytest.warns(DeprecationWarning):
            out = _imager(obs_direct, gauss_im, gauss_prior, ttype="fast").make_image(
                show_updates=False)
        assert np.all(np.isfinite(out.imvec))

    def test_diag_dataterm_on_numpy_is_untouched(self, obs_direct, gauss_im, gauss_prior):
        # control: _diag closures image fine on numpy, they are only missing jax kernels
        out = _imager(obs_direct, gauss_im, gauss_prior,
                      data_term={"cphase_diag": 1}).make_image(show_updates=False)
        assert np.all(np.isfinite(out.imvec))

    def test_direct_with_use_jax_still_runs(self, obs_direct, gauss_im, gauss_prior):
        # control: the supported combination must not be caught by the guard
        out = _imager(obs_direct, gauss_im, gauss_prior).make_image(
            use_jax=True, show_updates=False)
        assert np.all(np.isfinite(out.imvec))


# ============================== sharding and survey ==============================
class TestShardingGuards:
    def test_baseline_sharding_refuses_fast_ttype(self, obs_direct, gauss_im, gauss_prior):
        # the existing NotImplementedError is unreachable: for 'fast' the operator is a
        # tuple, so the isinstance branch is taken and the run dies in _pad_rows with
        # IndexError instead.
        jax = pytest.importorskip("jax")
        from ehtim.imaging.sharding import build_mesh, make_sharded_value_and_grad
        imgr = _imager(obs_direct, gauss_im, gauss_prior, ttype="fast")
        imgr.init_imager()
        args = (imgr._init_arr, imgr._config, imgr._which_solve, imgr._data_tuples,
                imgr._logfreqratio_list, len(imgr.obslist_next), imgr.dat_term_next,
                imgr.reg_term_next, imgr._prior_arr, imgr.norm_reg, imgr._regparams(),
                imgr._embed_mask)
        mesh = build_mesh(devices=jax.devices()[:1])
        with pytest.raises(NotImplementedError, match="fast"):
            make_sharded_value_and_grad(*args, mesh=mesh, shard_axis="baseline")

    def test_build_mesh_without_gpus_points_at_the_mesh_kwarg(self, monkeypatch):
        # monkeypatched rather than skipped, so it runs on GPU boxes too
        jax = pytest.importorskip("jax")
        import ehtim.imaging.sharding as sharding_mod

        def no_gpu(kind=None):
            raise RuntimeError("Unknown backend: 'gpu' requested")

        monkeypatch.setattr(jax, "devices", no_gpu)
        with pytest.raises(ValueError, match="mesh"):
            sharding_mod.build_mesh()


class TestSurveyGuards:
    def test_survey_refuses_multiple_observations(self, obs_direct, gauss_im, gauss_prior):
        # the survey keys its data terms by bare name; with n_obs > 1 they are already
        # suffixed and get suffixed again, which surfaced as KeyError: 'vis_0_0'.
        from ehtim.imaging.survey_gpu import run_survey_gpu
        imgr = eh.imager.Imager([obs_direct, obs_direct], gauss_prior, prior_im=gauss_prior,
                                flux=gauss_im.total_flux(), data_term={"vis": 1},
                                reg_term={"simple": 1, "tv": 1}, ttype="direct",
                                maxit=2, epsilon_tv=EPSILON_TV)
        with pytest.raises(ValueError, match="single observation"):
            run_survey_gpu(imgr, weight_grid={"tv": np.array([1.0])}, maxit=2)

    def test_survey_still_runs_on_one_observation(self, obs_direct, gauss_im, gauss_prior):
        # control: the guard must not fire on the supported case
        from ehtim.imaging.survey_gpu import run_survey_gpu
        images, objval, _, _ = run_survey_gpu(_imager(obs_direct, gauss_im, gauss_prior),
                                              weight_grid={"tv": np.array([1.0, 10.0])},
                                              maxit=4)
        assert images.shape[0] == 2 and np.all(np.isfinite(objval))
