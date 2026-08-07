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


# ---------------------------------------------------------------------------
# L-BFGS history size
#
# `optax.lbfgs(memory_size=m)` state is ~2m+3 copies of the image vector, and
# _inner_survey vmaps the whole reconstruction, so the state is paid per batch
# element. At the shipped memory_size=50 that is 3.22 MiB each (103x the 4096
# pixel vector at float64): 3.22 GiB at B=1024 and 25.75 GiB at B=8192, the
# Paper IV grid size this module models itself on. It was a module constant
# with no way for a caller to change it.
# ---------------------------------------------------------------------------


def _spy_on_optdict(monkeypatch):
    """Capture the option dict survey_gpu hands to resolve_optax."""
    from ehtim.imaging import survey_gpu
    seen = {}
    real = survey_gpu.resolve_optax

    def spy(optimizer, optdict):
        seen.update(optdict)
        return real(optimizer, optdict)

    monkeypatch.setattr(survey_gpu, "resolve_optax", spy)
    return seen


def test_survey_lbfgs_history_defaults_to_the_module_constant(
        obs_direct, gauss_im, gauss_prior, monkeypatch):
    from ehtim.imaging.survey_gpu import NHIST, run_survey_gpu
    seen = _spy_on_optdict(monkeypatch)
    run_survey_gpu(_imager(obs_direct, gauss_im, gauss_prior),
                   weight_grid={"tv": np.array([1.0])}, maxit=3)
    assert seen["maxcor"] == NHIST


def test_survey_lbfgs_history_is_tunable(obs_direct, gauss_im, gauss_prior, monkeypatch):
    # The point of the knob: a large grid can trade convergence for memory.
    from ehtim.imaging.survey_gpu import run_survey_gpu
    seen = _spy_on_optdict(monkeypatch)
    run_survey_gpu(_imager(obs_direct, gauss_im, gauss_prior),
                   weight_grid={"tv": np.array([1.0])}, maxit=3, maxcor=7)
    assert seen["maxcor"] == 7


def test_survey_maxcor_sizes_the_optimizer_state():
    # Goes through resolve_optax, which is where the optdict "maxcor" key is
    # mapped onto optax's memory_size. Calling optax directly would not pin that
    # mapping: hardcoding memory_size=50 inside resolve_optax passes such a test
    # while making the knob inert.
    pytest.importorskip("optax")
    import jax.numpy as jnp

    from ehtim.imaging.optimizers import resolve_optax
    from ehtim.imaging.survey_gpu import MAXLS

    def state_bytes(m):
        gt, _ = resolve_optax("optax-lbfgs-bt",
                              {"maxiter": 1, "maxcor": m, "maxls": MAXLS})
        st = gt.init(jnp.zeros(4096, dtype=jnp.float64))
        return sum(np.asarray(x).nbytes for x in jax.tree_util.tree_leaves(st))

    # L-BFGS state is ~(2m+3) copies of the vector, so 50 must dominate 5.
    small, large = state_bytes(5), state_bytes(50)
    assert large > 4 * small, (
        f"maxcor should size the state through resolve_optax: "
        f"5 -> {small} B, 50 -> {large} B")


def test_survey_smaller_history_still_reconstructs(obs_direct, gauss_im, gauss_prior):
    # A shorter history converges more slowly, but must still reconstruct: an
    # all-zero or noise result would pass a shape-and-finiteness check.
    from ehtim.imaging.survey_gpu import run_survey_gpu
    imgs, objs, _, chis = run_survey_gpu(_imager(obs_direct, gauss_im, gauss_prior),
                                         weight_grid={"tv": np.array([1.0, 10.0])},
                                         maxit=60, maxcor=5)
    assert imgs.shape[0] == 2
    assert np.all(np.isfinite(imgs)) and np.all(np.isfinite(objs))
    assert np.all(chis["vis"] > 0)
    truth = gauss_im.imvec
    for b in range(imgs.shape[0]):
        a, t = imgs[b] - imgs[b].mean(), truth - truth.mean()
        nx = float(np.sum(a * t) / np.sqrt(np.sum(a * a) * np.sum(t * t)))
        assert nx > 0.8, f"grid point {b}: nxcorr {nx:.3f} against the source"


@pytest.mark.parametrize("bad", [0, -1, 7.9, None])
def test_survey_rejects_a_nonsense_maxcor(obs_direct, gauss_im, gauss_prior, bad):
    # Without this these reach optax (or int()) only after run_survey_gpu has
    # already rebuilt the prior and re-noised the observations for grid point 1.
    from ehtim.imaging.survey_gpu import run_survey_gpu
    with pytest.raises((ValueError, TypeError)):
        run_survey_gpu(_imager(obs_direct, gauss_im, gauss_prior),
                       weight_grid={"tv": np.array([1.0])}, maxit=3, maxcor=bad)
