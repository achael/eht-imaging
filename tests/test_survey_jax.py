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


def test_survey_grid_changes_the_reconstruction(obs_direct, gauss_im, gauss_prior):
    """The grid must reach the objective: the checks above pass for any four images at all."""
    from ehtim.imaging.survey_gpu import run_survey_gpu
    tvs, simples = np.array([1.0, 100.0]), np.array([1.0, 50.0])
    images, _, rec, _ = run_survey_gpu(_imager(obs_direct, gauss_im, gauss_prior),
                                       weight_grid={"tv": tvs, "simple": simples}, maxit=20)
    np.testing.assert_allclose(np.sort(np.unique(rec["tv"])), tvs)
    np.testing.assert_allclose(np.sort(np.unique(rec["simple"])), simples)

    # meshgrid(indexing="ij") over (tv, simple): rows 0,1 share tv and differ in simple;
    # rows 0,2 share simple and differ in tv. Both weights must move the reconstruction.
    for a, b, label in ((0, 1, "simple"), (0, 2, "tv")):
        spread = np.max(np.abs(images[a] - images[b])) / np.max(np.abs(images[0]))
        assert spread > 1e-3, f"{label} did not change the image (max rel diff {spread:.2e})"


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
    imgr = _imager(obs_direct, gauss_im, gauss_prior)
    imgr.init_imager()
    # pin the start vector. prior_fwhm also sets init_next, and x0 defaults to imgr._init_vec,
    # so without this the rows differ before a single iteration runs and the spread below
    # measures the starting images rather than the prior reaching the objective.
    x0 = np.asarray(imgr._init_vec, float)
    images, objval, rec, chis = run_survey_gpu(imgr, weight_grid={"tv": np.array([1.0, 10.0])},
                                               prior_fwhm=[40.0, 60.0], maxit=8, x0=x0)
    assert images.shape[0] == 4 and objval.shape == (4,)
    assert rec["tv"].shape == (4,) and set(np.unique(rec["prior_fwhm"])) == {40.0, 60.0}
    assert chis["vis"].shape == (4,) and np.all(np.isfinite(images))

    # rec["prior_fwhm"] echoes the caller's own list, so it proves nothing. Rows 0 and 2 are
    # the same tv weight at fwhm 40 vs 60.
    spread = np.max(np.abs(images[0] - images[2])) / np.max(np.abs(images[0]))
    assert spread > 1e-6, f"prior_fwhm 40 and 60 gave the same image (max rel diff {spread:.2e})"


def test_survey_sys_noise_outer_axis_and_restore(obs_direct, gauss_im, gauss_prior):
    from ehtim.imaging.survey_gpu import run_survey_gpu
    imgr = _imager(obs_direct, gauss_im, gauss_prior)
    base_prior, base_init = imgr.prior_next, imgr.init_next
    base_obs = list(imgr.obslist_next)
    imgr.init_imager()
    base_sigma = np.array(imgr._data_tuples["vis"][1], copy=True)
    images, objval, rec, _ = run_survey_gpu(imgr, weight_grid={"tv": np.array([1.0])},
                                            sys_noise=[0.0, 0.05], maxit=8)
    assert images.shape[0] == 2 and set(np.unique(rec["sys_noise"])) == {0.0, 0.05}
    # A sys_noise sweep rebuilds obslist_next and leaves the prior alone, so checking
    # prior_next alone was the one of the three that could not fail.
    assert imgr.prior_next is base_prior
    assert imgr.init_next is base_init
    assert imgr.obslist_next == base_obs

    # and the data products, which are what those attributes exist to produce. Restoring the
    # attributes is not enough: the derived sigmas are cached, so the caller was left imaging
    # the last grid point's inflated errors.
    imgr.init_imager()
    np.testing.assert_allclose(imgr._data_tuples["vis"][1], base_sigma, rtol=1e-12)

    # sys_noise inflates the errors, so the two rows must not be the same reconstruction
    spread = np.max(np.abs(images[0] - images[1])) / np.max(np.abs(images[0]))
    assert spread > 1e-6, f"sys_noise made no difference (max rel diff {spread:.2e})"
