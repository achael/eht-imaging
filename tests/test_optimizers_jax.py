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
from ehtim.imaging.optimizers import (
    _METHOD_OPTS as _METHOD_OPTS_KEYS,
)
from ehtim.imaging.optimizers import (
    classify_optimizer,
    run_optimizer,
)

# No module-level jax mark: most of this file is the scipy dispatch, which needs neither
# jax nor optax, and CI deselects the jax mark. The tests that really do reach jax (every
# optax one, since optax imports it) carry the mark individually.
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
@pytest.mark.jax
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


@pytest.mark.jax
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
    from ehtim.imaging.optimizers import resolve_backend
    assert classify_optimizer(name) == "scipy"
    assert resolve_backend(name).method == method


def test_a_new_optimizer_can_be_registered_without_editing_the_module():
    # the point of the registry: a caller adds a backend rather than a branch
    from ehtim.imaging.optimizers import (
        _BACKENDS,
        OptimizerBackend,
        register_optimizer,
        resolve_backend,
    )

    class CountingBackend(OptimizerBackend):
        kind = "scipy"

        def __init__(self):
            self.calls = 0

        def run(self, build_loss, x0, optdict, callback, device):
            self.calls += 1
            return scipy.optimize.OptimizeResult(x=np.asarray(x0), fun=0.0, nit=0,
                                                 success=True, message="counted")

    backend = CountingBackend()
    register_optimizer("counting", backend)
    try:
        assert classify_optimizer("counting") == "scipy"
        assert resolve_backend("counting") is backend
        res = run_optimizer("counting", lambda: (None, None), x0=np.zeros(3),
                            optdict={"maxiter": 1, "gtol": 1e-6}, callback=None)
        assert backend.calls == 1 and res.message == "counted"
    finally:
        del _BACKENDS["counting"]


@pytest.mark.parametrize("name", ["LBFGS", "BFGS", "Newton-CG", "TNC"])
def test_optimizer_names_are_case_insensitive(name):
    from ehtim.imaging.optimizers import resolve_backend
    assert resolve_backend(name) is resolve_backend(name.lower())


def test_a_registered_name_is_also_case_insensitive():
    from ehtim.imaging.optimizers import _BACKENDS, ScipyBackend, register_optimizer

    register_optimizer("MyCG", ScipyBackend("CG"))
    try:
        assert classify_optimizer("mycg") == "scipy"
        assert classify_optimizer("MYCG") == "scipy"
    finally:
        del _BACKENDS["mycg"]


def test_replacing_a_builtin_optimizer_warns():
    from ehtim.imaging.optimizers import _BACKENDS, ScipyBackend, register_optimizer

    original = _BACKENDS["bfgs"]
    try:
        with pytest.warns(UserWarning, match="replacing the built-in"):
            register_optimizer("bfgs", ScipyBackend("CG"))
    finally:
        _BACKENDS["bfgs"] = original


def test_registering_a_non_backend_is_refused():
    from ehtim.imaging.optimizers import register_optimizer
    with pytest.raises(TypeError, match="OptimizerBackend"):
        register_optimizer("nonsense", lambda *a, **k: None)


def test_a_scipy_backend_needs_an_option_table():
    # ScipyBackend is public, so a method with no table would otherwise fail deep inside
    # _scipy_options with a bare KeyError
    from ehtim.imaging.optimizers import ScipyBackend
    with pytest.raises(ValueError, match="no option table"):
        ScipyBackend("Powell")


def test_the_callback_is_forwarded_to_scipy(make_opt_imager):
    # make_image drives its progress display through this; dropping it is silent
    imgr = make_opt_imager()
    imgr.check_params()
    imgr.check_limits()
    imgr.init_imager()
    seen = []
    optdict = {"maxiter": 3, "ftol": 1e-12, "gtol": 1e-12, "maxcor": NHIST, "maxls": MAXLS}
    run_optimizer("lbfgs", lambda: (imgr.objfunc, imgr.objgrad), x0=imgr._init_vec,
                  optdict=optdict, callback=lambda xk: seen.append(np.asarray(xk).copy()))
    assert seen and all(s.shape == np.shape(imgr._init_vec) for s in seen)


def test_the_memory_guard_fires_through_the_backend(monkeypatch):
    # calling the helper directly would still pass if ScipyBackend forgot to call it
    import ehtim.imaging.optimizers as opt_mod

    def spy(fun, x0, **kwargs):
        return scipy.optimize.OptimizeResult(x=np.asarray(x0), fun=0.0, nit=0, success=True)

    monkeypatch.setattr(opt_mod.scipy.optimize, "minimize", spy)
    optdict = {"maxiter": 1, "ftol": 1e-6, "gtol": 1e-6, "maxcor": NHIST, "maxls": MAXLS}
    with pytest.warns(UserWarning, match="dense"):
        run_optimizer("bfgs", lambda: (None, None), x0=np.zeros(128 * 128),
                      optdict=optdict, callback=None)


def test_an_unknown_name_lists_the_registered_ones():
    with pytest.raises(ValueError, match="registered names are"):
        classify_optimizer("no-such-optimizer")


@pytest.mark.parametrize("optimizer,kind", [
    (None, "scipy"), ("bfgs", "scipy"), ("adam", "optax"),
    (lambda *a, **k: None, "callable"),
], ids=["default", "scipy-name", "optax-name", "user-callable"])
def test_each_optimizer_shape_routes_to_its_backend(optimizer, kind):
    from ehtim.imaging.optimizers import resolve_backend
    assert classify_optimizer(optimizer) == kind
    assert resolve_backend(optimizer).kind == kind


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


def _toy_quadratic(x):
    return float(np.sum((x - 1.0) ** 2))


def _toy_quadratic_grad(x):
    return 2.0 * (np.asarray(x) - 1.0)


@pytest.mark.parametrize("method", sorted(_METHOD_OPTS_KEYS))
def test_every_omitted_option_really_is_unknown_to_the_method(method):
    # the other half of test_no_option_is_silently_dropped, which can only catch keys
    # scipy does not know. This catches the opposite and more dangerous direction: a key
    # scipy DOES read that the table drops, silently disabling a stopping rule. Four of
    # the six table entries could be corrupted without failing anything before this.
    from ehtim.imaging.optimizers import _METHOD_OPTS
    optdict = {"maxiter": 2, "ftol": 1e-6, "gtol": 1e-6, "maxcor": NHIST, "maxls": MAXLS}
    for key, value in optdict.items():
        if key in _METHOD_OPTS[method] or key in {"maxfun", "xtol"}:
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scipy.optimize.minimize(_toy_quadratic, np.ones(4), method=method,
                                    jac=_toy_quadratic_grad, options={key: value})
        assert any("Unknown solver options" in str(w.message) for w in caught), \
            f"{method} reads {key!r}, but _METHOD_OPTS drops it"


def test_tnc_gets_an_evaluation_cap_and_newton_cg_a_tolerance():
    # both rename rather than drop: TNC counts evaluations, Newton-CG takes only xtol
    from ehtim.imaging.optimizers import _scipy_options
    optdict = {"maxiter": 7, "ftol": 1e-5, "gtol": 1e-6, "maxcor": 50, "maxls": 40}
    assert _scipy_options("TNC", optdict)["maxfun"] == 7
    assert "maxiter" not in _scipy_options("TNC", optdict)
    assert _scipy_options("Newton-CG", optdict)["xtol"] == 1e-5
    assert _scipy_options("L-BFGS-B", optdict) == optdict


@pytest.mark.parametrize("name,method", SCIPY_METHOD_CASES, ids=SCIPY_METHOD_IDS)
def test_the_requested_method_reaches_scipy(monkeypatch, name, method):
    # every method reduces the objective, so "it converged" cannot tell them apart: a
    # dispatcher that ignored the request and always ran L-BFGS-B would pass every other
    # test in this file. Watch the argument instead.
    import ehtim.imaging.optimizers as opt_mod
    seen = {}

    def spy(fun, x0, **kwargs):
        seen.update(kwargs)
        return scipy.optimize.OptimizeResult(x=np.asarray(x0), fun=0.0, nit=0,
                                             success=True, message="stub")

    monkeypatch.setattr(opt_mod.scipy.optimize, "minimize", spy)
    optdict = {"maxiter": 1, "ftol": 1e-6, "gtol": 1e-6, "maxcor": NHIST, "maxls": MAXLS}
    run_optimizer(name, lambda: ((lambda x: 0.0), (lambda x: np.zeros_like(x))),
                  x0=np.zeros(4), optdict=optdict, callback=None)
    assert seen["method"] == method


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
    # a 128x128 image needs 2 GiB for the inverse Hessian alone, and scipy holds
    # several matrices that size at once, so peak runs several times higher
    from ehtim.imaging.optimizers import _warn_if_bfgs_hessian_is_large
    # UserWarning, not ResourceWarning: python ignores that category by default, so the
    # user would never have seen it
    with pytest.warns(UserWarning, match="dense"):
        _warn_if_bfgs_hessian_is_large("BFGS", 128 * 128)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_bfgs_hessian_is_large("BFGS", 32 * 32)      # 8 MiB, fine
        _warn_if_bfgs_hessian_is_large("L-BFGS-B", 256 * 256)  # not BFGS, no dense matrix


@pytest.mark.jax
@pytest.mark.parametrize("name", ["adam", "adamw", "sgd", "rmsprop", "optax-lbfgs",
                                  "optax-lbfgs-bt"])
def test_each_optax_name_reaches_resolve_optax_with_its_own_spec(monkeypatch, name):
    # a backend that ignored self.spec and always built adam would pass everything else
    import ehtim.imaging.optimizers as opt_mod
    seen = {}

    def spy(optimizer, optdict):
        seen["spec"] = optimizer
        return object(), False

    monkeypatch.setattr(opt_mod, "resolve_optax", spy)
    monkeypatch.setattr(opt_mod, "_run_optax",
                        lambda *a, **k: scipy.optimize.OptimizeResult(x=a[4], fun=0.0))
    optdict = {"maxiter": 1, "ftol": 1e-6, "gtol": 1e-6, "maxcor": NHIST, "maxls": MAXLS}
    run_optimizer(name, lambda dev: (None, None, np.asarray, None), x0=np.zeros(3),
                  optdict=optdict, callback=None)
    assert seen["spec"] == name


@pytest.mark.jax
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
@pytest.mark.jax
def test_optax_lbfgs_recovers(make_opt_imager, gauss_im):
    out = make_opt_imager().make_image(optimizer="optax-lbfgs", show_updates=False)
    assert _nxcorr(out.imvec, gauss_im.imvec) > NXCORR_FLOOR


@pytest.mark.slow
@pytest.mark.jax
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
