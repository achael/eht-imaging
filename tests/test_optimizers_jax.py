"""Tests for the modular optimizer layer (ehtim.imaging.optimizers).

Covers: classify_optimizer routing; the scipy default is unchanged (the dispatcher's
scipy path reproduces a direct scipy.optimize.minimize call bit-for-bit); the on-device
value_and_grad matches the host make_objective_jax; optax-lbfgs and a custom optax
GradientTransformation recover the source; and a user callable plugs in via the escape
hatch. optax-running tests are marked slow.
"""
import warnings

import numpy as np
import pytest
import scipy.optimize

import ehtim as eh
from ehtim.imager import MAXLS, NHIST
from ehtim.imaging.imager_backend import make_objective_jax, make_value_and_grad_jax
from ehtim.imaging.optimizers import classify_optimizer, run_optimizer

pytestmark = pytest.mark.jax

optax = pytest.importorskip("optax")

VALUE_RTOL = 1e-9
GRAD_RTOL = 1e-9
NXCORR_FLOOR = 0.8
EPSILON_TV = 1e-10
RNG_SEED = 4
PERTURB = 0.10


def _nxcorr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 0 else 0.0


@pytest.fixture(scope="module")
def make_opt_imager(obs_direct, gauss_im, gauss_prior):
    """Factory: a fresh Stokes-I imager per call (make_image mutates the imager)."""
    def build():
        return eh.imager.Imager(
            obs_direct, gauss_prior, prior_im=gauss_prior, flux=gauss_im.total_flux(),
            data_term={"vis": 1}, reg_term={"simple": 1, "tv": 1},
            ttype="direct", pol="I", maxit=100, epsilon_tv=EPSILON_TV)
    return build


def _perturbed_x0(imgr):
    rng = np.random.default_rng(RNG_SEED)
    base = np.asarray(imgr._init_vec, dtype=np.float64)
    return base + PERTURB * rng.standard_normal(base.size)


def _backend_args(imgr):
    return (imgr._init_arr, imgr._config, imgr._which_solve, imgr._data_tuples,
            imgr._logfreqratio_list, len(imgr.obslist_next), imgr.dat_term_next,
            imgr.reg_term_next, imgr._prior_arr, imgr.norm_reg, imgr._regparams(),
            imgr._embed_mask)


# ============================== dispatch ==============================
def test_classify_optimizer():
    assert classify_optimizer(None) == "scipy"
    assert classify_optimizer("lbfgs") == "scipy"
    assert classify_optimizer("scipy-lbfgs") == "scipy"
    assert classify_optimizer("adam") == "optax"
    assert classify_optimizer("optax-lbfgs") == "optax"
    assert classify_optimizer(optax.adam(1e-2)) == "optax"
    assert classify_optimizer(lambda *a, **k: None) == "callable"
    with pytest.raises(ValueError):
        classify_optimizer("not-an-optimizer")


def test_device_vg_matches_host(make_opt_imager):
    # the on-device value_and_grad reproduces the validated host objective
    imgr = make_opt_imager()
    imgr.check_params()
    imgr.check_limits()
    imgr.init_imager()
    fun = make_objective_jax(*_backend_args(imgr))
    vg, _loss, to_device = make_value_and_grad_jax(*_backend_args(imgr))
    x = _perturbed_x0(imgr)
    v_host, g_host = fun(x)
    val, grad = vg(to_device(x))
    assert np.allclose(float(val), v_host, rtol=VALUE_RTOL)
    assert np.allclose(np.asarray(grad), g_host, rtol=GRAD_RTOL)


# ============================== scipy path (default unchanged) ==============================
# every scipy method the dispatcher accepts, and the options it should end up with. The
# imager always passes maxiter/ftol/gtol/maxcor/maxls, and scipy only *warns* about keys a
# method does not know, so an unfiltered optdict silently disables the stopping rule the
# caller asked for: TNC drops maxiter and runs uncapped, Newton-CG drops both tolerances.
SCIPY_METHOD_CASES = [
    ("lbfgs", "L-BFGS-B"), ("l-bfgs-b", "L-BFGS-B"), ("scipy", "L-BFGS-B"),
    ("scipy-lbfgs", "L-BFGS-B"), ("bfgs", "BFGS"), ("cg", "CG"),
    ("newton-cg", "Newton-CG"), ("tnc", "TNC"), ("slsqp", "SLSQP"),
]
SCIPY_METHOD_IDS = [c[0] for c in SCIPY_METHOD_CASES]


@pytest.mark.parametrize("name,method", SCIPY_METHOD_CASES, ids=SCIPY_METHOD_IDS)
def test_every_scipy_alias_maps_to_its_method(name, method):
    from ehtim.imaging.optimizers import _SCIPY_METHODS
    assert classify_optimizer(name) == "scipy"
    assert _SCIPY_METHODS[name.lower()] == method


@pytest.mark.parametrize("name,method", SCIPY_METHOD_CASES, ids=SCIPY_METHOD_IDS)
def test_no_option_is_silently_dropped(make_opt_imager, name, method):
    # scipy raises RuntimeWarning for options a method does not read. Running with
    # warnings as errors is what pins that the filter is right; without the filter this
    # fails for every method except L-BFGS-B.
    from ehtim.imaging.optimizers import _scipy_options
    optdict = {"maxiter": 2, "ftol": 1e-6, "gtol": 1e-6, "maxcor": NHIST, "maxls": MAXLS}
    imgr = make_opt_imager()
    imgr.check_params()
    imgr.check_limits()
    imgr.init_imager()
    # record with "always" rather than raising on the first: Python dedups warnings per
    # location, so a raising filter would only fire for whichever method ran first
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scipy.optimize.minimize(imgr.objfunc, imgr._init_vec, method=method,
                                jac=imgr.objgrad,
                                options=_scipy_options(method, optdict))
    dropped = [str(w.message) for w in caught if "Unknown solver options" in str(w.message)]
    assert not dropped, f"{method} silently dropped options: {dropped}"


def test_tnc_gets_an_evaluation_cap_and_newton_cg_a_tolerance():
    # both rename rather than drop: TNC counts evaluations, Newton-CG takes only xtol
    from ehtim.imaging.optimizers import _scipy_options
    optdict = {"maxiter": 7, "ftol": 1e-5, "gtol": 1e-6, "maxcor": 50, "maxls": 40}
    assert _scipy_options("TNC", optdict)["maxfun"] == 7
    assert "maxiter" not in _scipy_options("TNC", optdict)
    assert _scipy_options("Newton-CG", optdict)["xtol"] == 1e-5
    assert _scipy_options("L-BFGS-B", optdict) == optdict


@pytest.mark.slow
@pytest.mark.parametrize("name", ["bfgs", "cg", "newton-cg", "tnc", "slsqp"])
def test_the_other_methods_reduce_the_objective(make_opt_imager, name):
    imgr = make_opt_imager()
    imgr.check_params()
    imgr.check_limits()
    imgr.init_imager()
    x0 = imgr._init_vec
    f0 = float(imgr.objfunc(x0))
    optdict = {"maxiter": 20, "ftol": 1e-6, "gtol": 1e-6, "maxcor": NHIST, "maxls": MAXLS}
    res = run_optimizer(name, lambda: (imgr.objfunc, imgr.objgrad),
                        x0=x0, optdict=optdict, callback=None)
    assert float(imgr.objfunc(res.x)) < f0


def test_bfgs_warns_when_the_dense_hessian_is_large(make_opt_imager):
    # a 128x128 image needs 2 GiB for the inverse Hessian, 256x256 needs 32
    from ehtim.imaging.optimizers import _warn_if_bfgs_hessian_is_large
    with pytest.warns(ResourceWarning, match="dense"):
        _warn_if_bfgs_hessian_is_large("BFGS", 128 * 128)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_bfgs_hessian_is_large("BFGS", 32 * 32)      # 8 MiB, fine
        _warn_if_bfgs_hessian_is_large("L-BFGS-B", 256 * 256)  # not BFGS, no dense matrix


def test_resolve_optax_names_the_valid_optimizers():
    # reachable through run_survey_gpu's optimizer kwarg, where it used to be a bare KeyError
    from ehtim.imaging.optimizers import resolve_optax
    with pytest.raises(ValueError, match="unknown optax optimizer"):
        resolve_optax("lbfgs", {"maxcor": 50, "maxls": 40})


@pytest.mark.slow
def test_scipy_lane_matches_direct_scipy(make_opt_imager):
    # the dispatcher's default path is a bit-for-bit pass-through to scipy L-BFGS-B
    imgr = make_opt_imager()
    imgr.check_params()
    imgr.check_limits()
    imgr.init_imager()
    optdict = {"maxiter": imgr.maxit_next, "ftol": imgr.stop_next, "gtol": imgr.stop_next,
               "maxcor": NHIST, "maxls": MAXLS}
    x0 = imgr._init_vec
    res_dispatch = run_optimizer(None, lambda: (imgr.objfunc, imgr.objgrad),
                                 x0=x0, optdict=optdict, callback=None)
    res_direct = scipy.optimize.minimize(imgr.objfunc, x0, method="L-BFGS-B",
                                         jac=imgr.objgrad, options=optdict)
    np.testing.assert_array_equal(res_dispatch.x, res_direct.x)


@pytest.mark.slow
def test_default_recovers(make_opt_imager, gauss_im):
    out = make_opt_imager().make_image(show_updates=False)
    assert _nxcorr(out.imvec, gauss_im.imvec) > NXCORR_FLOOR


# ============================== optax + custom optimizers ==============================
@pytest.mark.slow
def test_optax_lbfgs_recovers(make_opt_imager, gauss_im):
    out = make_opt_imager().make_image(optimizer="optax-lbfgs", show_updates=False)
    assert _nxcorr(out.imvec, gauss_im.imvec) > NXCORR_FLOOR


@pytest.mark.slow
def test_custom_gradient_transformation_recovers(make_opt_imager, gauss_im):
    # any optax GradientTransformation works through the optax path
    out = make_opt_imager().make_image(optimizer=optax.adam(3e-2), show_updates=False)
    assert _nxcorr(out.imvec, gauss_im.imvec) > NXCORR_FLOOR


@pytest.mark.slow
def test_custom_callable_recovers(make_opt_imager, gauss_im):
    # the escape hatch: a user callable receives a host value_and_grad and returns
    # anything with .x / .fun. Here it plugs scipy CG.
    def my_optimizer(value_and_grad, x0, *, maxiter, tol, callback=None):
        return scipy.optimize.minimize(value_and_grad, x0, method="CG", jac=True,
                                       options={"maxiter": maxiter}, callback=callback)

    out = make_opt_imager().make_image(optimizer=my_optimizer, show_updates=False)
    assert _nxcorr(out.imvec, gauss_im.imvec) > NXCORR_FLOOR


def test_unknown_optimizer_raises(make_opt_imager):
    with pytest.raises(ValueError):
        make_opt_imager().make_image(optimizer="not-an-optimizer", show_updates=False)
