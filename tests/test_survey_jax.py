"""Tests for GPU parameter surveys (ehtim.imaging.survey_gpu).

The survey objective must equal the fixed-weight objective at matching weights, a vmapped batch
must equal running each grid point on its own, and the prior-FWHM / systematic-noise outer axes
must broadcast over the scalar sub-grid and restore the imager. Runnable on CPU (no GPU needed)
-- the vmap over the hyperparameter grid is the thing under test.
"""
import numpy as np
import pytest

import ehtim as eh

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")
pytest.importorskip("optax")


def _imager(obs, gauss_im, prior):
    return eh.imager.Imager(obs, prior, prior_im=prior, flux=gauss_im.total_flux(),
                            data_term={"vis": 1}, reg_term={"simple": 1, "tv": 1},
                            ttype="direct", maxit=20, epsilon_tv=1e-10)


def test_survey_objective_matches_fixed_weights(obs_direct, gauss_im, gauss_prior):
    from ehtim.imaging.imager_backend import make_survey_value_and_grad, make_value_and_grad_jax
    imgr = _imager(obs_direct, gauss_im, gauss_prior)
    imgr.init_imager()
    head = (imgr._init_arr, imgr._config, imgr._which_solve, imgr._data_tuples,
            imgr._logfreqratio_list, len(imgr.obslist_next))
    tail = (imgr._prior_arr, imgr.norm_reg, imgr._regparams(), imgr._embed_mask)
    x = np.asarray(imgr._init_vec, float) + 0.05 * np.random.default_rng(0).standard_normal(
        imgr._init_vec.size)

    vgf, _, to = make_value_and_grad_jax(*head, imgr.dat_term_next, imgr.reg_term_next, *tail)
    v0, g0 = jax.jit(vgf)(to(x))

    vgs, _, put, _ = make_survey_value_and_grad(*head, *tail)
    hp = {"dat_term": dict(imgr.dat_term_next), "reg_term": dict(imgr.reg_term_next), "reg_params": {}}
    v1, g1 = jax.jit(lambda z: vgs(z, hp))(put(x))

    assert np.allclose(float(v0), float(v1), rtol=1e-9)
    assert np.allclose(np.asarray(g0), np.asarray(g1), rtol=1e-9)


def test_survey_runs_and_shapes(obs_direct, gauss_im, gauss_prior):
    from ehtim.imaging.survey_gpu import run_survey_gpu
    grid = {"tv": np.array([1.0, 10.0]), "simple": np.array([1.0, 5.0])}
    images, objval, rec, chis = run_survey_gpu(_imager(obs_direct, gauss_im, gauss_prior),
                                               weight_grid=grid, maxit=10)
    assert images.shape[0] == 4 and objval.shape == (4,)
    assert np.all(np.isfinite(images)) and np.all(np.isfinite(objval))
    assert rec["tv"].shape == (4,) and rec["simple"].shape == (4,)
    assert chis["vis"].shape == (4,) and np.all(chis["vis"] > 0)


def test_survey_batch_matches_single(obs_direct, gauss_im, gauss_prior):
    from ehtim.imaging.survey_gpu import run_survey_gpu
    tvs = np.array([1.0, 50.0])
    imgs, objs, _, _ = run_survey_gpu(_imager(obs_direct, gauss_im, gauss_prior),
                            weight_grid={"tv": tvs}, maxit=15)
    for b, tv in enumerate(tvs):
        i1, o1, _, _ = run_survey_gpu(_imager(obs_direct, gauss_im, gauss_prior),
                            weight_grid={"tv": np.array([tv])}, maxit=15)
        assert np.allclose(imgs[b], i1[0], rtol=1e-6, atol=1e-8)
        assert np.allclose(objs[b], o1[0], rtol=1e-6)


def test_survey_prior_fwhm_outer_axis(obs_direct, gauss_im, gauss_prior):
    from ehtim.imaging.survey_gpu import run_survey_gpu
    images, objval, rec, chis = run_survey_gpu(_imager(obs_direct, gauss_im, gauss_prior),
                                               weight_grid={"tv": np.array([1.0, 10.0])},
                                               prior_fwhm=[40.0, 60.0], maxit=8)
    assert images.shape[0] == 4 and objval.shape == (4,)
    assert rec["tv"].shape == (4,) and set(np.unique(rec["prior_fwhm"])) == {40.0, 60.0}
    assert chis["vis"].shape == (4,) and np.all(np.isfinite(images))


def test_survey_sys_noise_outer_axis_and_restore(obs_direct, gauss_im, gauss_prior):
    from ehtim.imaging.survey_gpu import run_survey_gpu
    imgr = _imager(obs_direct, gauss_im, gauss_prior)
    base_prior = imgr.prior_next
    images, objval, rec, _ = run_survey_gpu(imgr, weight_grid={"tv": np.array([1.0])},
                                            sys_noise=[0.0, 0.05], maxit=8)
    assert images.shape[0] == 2 and set(np.unique(rec["sys_noise"])) == {0.0, 0.05}
    assert imgr.prior_next is base_prior  # imager restored after the survey
