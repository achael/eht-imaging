"""Tests for the modular optimizer layer (ehtim.imaging.optimizers).

Covers: classify_optimizer routing; the scipy default is unchanged (the dispatcher's
scipy path reproduces a direct scipy.optimize.minimize call bit-for-bit); the on-device
value_and_grad matches the host make_objective_jax; optax-lbfgs and a custom optax
GradientTransformation recover the source; and a user callable plugs in via the escape
hatch. optax-running tests are marked slow.
"""
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
EPSILON_TV = 1e-10
RNG_SEED = 4
PERTURB = 0.10


def _nxcorr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 0 else 0.0


@pytest.fixture(scope="module")
def make_opt_imager(obs_direct, gauss_im, flat_prior):
    """Factory: a fresh Stokes-I imager per call (make_image mutates the imager).

    Init and prior are both featureless, so the reconstruction has to come from the data.
    """
    def build(maxit=100):
        return eh.imager.Imager(
            obs_direct, flat_prior, prior_im=flat_prior, flux=gauss_im.total_flux(),
            data_term={"vis": 1}, reg_term={"simple": 1, "tv": 1},
            ttype="direct", pol="I", maxit=maxit, epsilon_tv=EPSILON_TV)
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
    # both jax factories agree with each other AND with the numpy analytic. Without the numpy
    # anchor the two factories are built from the same args by the same family, so a shared
    # error would cancel.
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

    assert np.allclose(v_host, float(imgr.objfunc(x)), rtol=VALUE_RTOL)
    assert np.allclose(g_host, np.asarray(imgr.objgrad(x)), rtol=GRAD_RTOL)


# ============================== scipy path (default unchanged) ==============================
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
def test_default_recovers(make_opt_imager, gauss_im, recovery_floors):
    out = make_opt_imager().make_image(show_updates=False)
    assert _nxcorr(out.imvec, gauss_im.imvec) > recovery_floors.floor


# ============================== optax + custom optimizers ==============================
@pytest.mark.slow
def test_optax_lbfgs_recovers(make_opt_imager, gauss_im, recovery_floors):
    out = make_opt_imager().make_image(optimizer="optax-lbfgs", show_updates=False)
    assert _nxcorr(out.imvec, gauss_im.imvec) > recovery_floors.floor


@pytest.mark.slow
def test_custom_gradient_transformation_recovers(make_opt_imager, gauss_im, recovery_floors):
    # any optax GradientTransformation works through the optax path. adam is first-order, so
    # it lands well short of L-BFGS at the same iteration count.
    out = make_opt_imager().make_image(optimizer=optax.adam(3e-2), show_updates=False)
    assert _nxcorr(out.imvec, gauss_im.imvec) > recovery_floors.first_order


@pytest.mark.slow
def test_custom_callable_recovers(make_opt_imager, gauss_im, recovery_floors):
    # the escape hatch: a user callable receives a host value_and_grad and returns
    # anything with .x / .fun. Here it plugs scipy CG.
    def my_optimizer(value_and_grad, x0, *, maxiter, tol, callback=None):
        return scipy.optimize.minimize(value_and_grad, x0, method="CG", jac=True,
                                       options={"maxiter": maxiter}, callback=callback)

    out = make_opt_imager().make_image(optimizer=my_optimizer, show_updates=False)
    assert _nxcorr(out.imvec, gauss_im.imvec) > recovery_floors.floor


def test_recovery_floors_are_not_vacuous(flat_prior, gauss_im, recovery_floors):
    """Guard on the four tests above: the start must carry none of the source.

    They previously began from a blur of the truth, which scores 0.989 against it, so they
    passed on the untouched starting image. Asserted on the fixture rather than on a
    zero-iteration run: scipy L-BFGS-B completes one line search before it checks the
    iteration count, so `maxit=0` is not a no-op and scores 0.186, which would leave this
    guard measuring one optimizer step instead of the fixture.
    """
    # a featureless image has zero variance, so _nxcorr returns its 0.0 guard value
    start = _nxcorr(flat_prior.imvec, gauss_im.imvec)
    assert start < recovery_floors.no_op_ceiling, (
        f"the starting image scores {start:.3f} against the truth, so the recovery floors "
        "above no longer measure the reconstruction")


def test_unknown_optimizer_raises(make_opt_imager):
    with pytest.raises(ValueError):
        make_opt_imager().make_image(optimizer="not-an-optimizer", show_updates=False)
