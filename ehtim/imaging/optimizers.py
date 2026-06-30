"""Pluggable optimizer dispatch for the imaging objective.

`run_optimizer` lets `Imager.make_image` drive any optimizer through one seam:
the scipy L-BFGS-B default (unchanged), an optax optimizer running on-device, or
a user-supplied callable. It always returns a `scipy.optimize.OptimizeResult`, so
the caller unpacks `.x`/`.fun` the same way for every backend.
"""
from functools import partial

import numpy as np
import scipy.optimize

# Built-in optimizer names, resolved in their respective lanes. Listed here so
# classify_optimizer can route the names without importing optax.
_OPTAX_NAMES = frozenset({"optax-lbfgs", "optax-lbfgs-bt", "adam", "adamw", "sgd", "rmsprop"})
_SCIPY_NAMES = frozenset({"lbfgs", "l-bfgs-b", "scipy", "scipy-lbfgs"})


def classify_optimizer(optimizer):
    """Return the lane that handles `optimizer`: 'scipy', 'optax', or 'callable'.

    None and the scipy aliases map to 'scipy'; an optax built-in name or a
    GradientTransformation maps to 'optax'; any other callable maps to 'callable'.
    """
    if optimizer is None:
        return "scipy"
    if isinstance(optimizer, str):
        name = optimizer.lower()
        if name in _SCIPY_NAMES:
            return "scipy"
        if name in _OPTAX_NAMES:
            return "optax"
        raise ValueError(f"unknown optimizer name {optimizer!r}")
    # an optax GradientTransformation is a NamedTuple exposing init/update
    if hasattr(optimizer, "init") and hasattr(optimizer, "update"):
        return "optax"
    if callable(optimizer):
        return "callable"
    raise TypeError(f"unsupported optimizer {optimizer!r}")


def run_optimizer(optimizer, *, x0, optdict, callback=None, build_loss=None,
                  device=None, mesh=None):
    """Minimize the imaging objective with the chosen optimizer.

    The caller supplies one `build_loss` builder appropriate to the lane (the lane
    only ever needs one): the host builder for the scipy/callable lanes, or the
    on-device builder for the optax lane. Each lane below calls it with the
    signature it expects, so passing the wrong builder fails fast (arity mismatch).

    Parameters
    ----------
    optimizer : None, str, optax.GradientTransformation, or callable
        Selects the lane (see `classify_optimizer`). None keeps scipy L-BFGS-B.
    x0 : numpy.ndarray
        Initial solver vector.
    optdict : dict
        scipy L-BFGS-B options ('maxiter', 'ftol', 'gtol', 'maxcor', 'maxls').
        Also the source of the iteration cap and tolerances for the optax lane.
    callback : callable, optional
        Per-iteration callback(xk) for the host lanes (scipy / callable).
    build_loss : callable
        Lane-appropriate builder of the loss handles:
        - scipy / callable lanes: `() -> (fun, jac)` host handles. `fun` is `objfunc`
          or a jax objective returning (value, grad); `jac` is `objgrad`, True, or None.
        - optax lane: `(device) -> (value_and_grad, loss, to_device, aux)` on-device handles.
    device, mesh : optional
        Device / sharding mesh for the optax lane.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Carries at least `.x` and `.fun`.
    """
    kind = classify_optimizer(optimizer)

    if kind == "scipy":
        fun, jac = build_loss()
        return scipy.optimize.minimize(fun, x0, method="L-BFGS-B", jac=jac,
                                       options=optdict, callback=callback)

    elif kind == "callable":
        # The escape hatch: hand the user a host value_and_grad(x) -> (value, grad).
        fun, jac = build_loss()
        if jac is True:
            value_and_grad = fun           # fun already returns (value, grad)
        elif jac is None:
            def value_and_grad(x):
                return fun(x), None
        else:
            def value_and_grad(x):
                return fun(x), jac(x)
        return optimizer(value_and_grad, x0, maxiter=optdict["maxiter"],
                         tol=optdict["gtol"], callback=callback)

    else:
        # kind == "optax": run an optax optimizer entirely on device.
        gt, needs_ls = _resolve_optax(optimizer, optdict)
        value_and_grad, loss, to_device, aux = build_loss(device)
        return _run_optax(gt, needs_ls, value_and_grad, loss, to_device(x0), optdict, aux=aux)


_DEFAULT_LR = 1e-2  # step size for the first-order optax builtins (adam/sgd/...)


def _resolve_optax(optimizer, optdict):
    """Return (gradient_transformation, needs_linesearch) for the optax lane.

    A string names a built-in; an optax.GradientTransformation is used as given.
    L-BFGS honors maxcor (memory) and maxls (line-search steps) from optdict; the
    first-order builtins use _DEFAULT_LR (pass your own GradientTransformation to
    control the step size).
    """
    import optax

    if not isinstance(optimizer, str):
        needs_ls = isinstance(optimizer, optax.GradientTransformationExtraArgs)
        return optimizer, needs_ls

    name = optimizer.lower()
    if name == "optax-lbfgs":
        linesearch = optax.scale_by_zoom_linesearch(
            max_linesearch_steps=int(optdict["maxls"]))
        return optax.lbfgs(memory_size=int(optdict["maxcor"]),
                           linesearch=linesearch), True
    if name == "optax-lbfgs-bt":
        # Backtracking (Armijo) line search: a few value evals per step instead of zoom's
        # bracket+zoom, so a vmapped survey doesn't pay the batch worst-case trip count.
        linesearch = optax.scale_by_backtracking_linesearch(
            max_backtracking_steps=int(optdict["maxls"]), store_grad=True)
        return optax.lbfgs(memory_size=int(optdict["maxcor"]),
                           linesearch=linesearch), True
    builders = {"adam": optax.adam, "adamw": optax.adamw,
                "sgd": optax.sgd, "rmsprop": optax.rmsprop}
    return builders[name](_DEFAULT_LR), False


def _run_optax(gt, needs_ls, value_and_grad, loss, x0, optdict, aux=None):
    """Minimize on device with a single jitted while_loop.

    value_and_grad -> optax update -> apply_updates, converging on gradient norm
    and relative value change. x and the optimizer state stay on device for the
    whole loop (no per-step host sync). `aux` (None for single device) carries the
    sharded data as a jit argument -- closing over sharded arrays mis-partitions
    them. Returns a scipy OptimizeResult.
    """
    import jax
    import jax.numpy as jnp
    import optax

    maxiter = int(optdict["maxiter"])
    gtol = float(optdict["gtol"])
    ftol = float(optdict["ftol"])

    @partial(jax.jit, donate_argnums=(0,))
    def run(x_init, aux):
        if aux is not None:
            def vg(x):
                return value_and_grad(x, aux)

            def lossfn(x):
                return loss(x, aux)
        else:
            vg, lossfn = value_and_grad, loss

        def cond(carry):
            i, _, _, gnorm, _, dval = carry
            return (i < maxiter) & (gnorm > gtol) & (dval > ftol)

        def body(carry):
            i, x, st, _, prev, _ = carry
            val, grad = vg(x)
            if needs_ls:
                updates, st = gt.update(grad, st, x, value=val, grad=grad,
                                        value_fn=lossfn)
            else:
                updates, st = gt.update(grad, st, x)
            x = optax.apply_updates(x, updates)
            rel = jnp.abs(prev - val) / jnp.maximum(jnp.abs(val), 1.0)
            return (i + 1, x, st, jnp.linalg.norm(grad), val, rel)

        init = (0, x_init, gt.init(x_init), jnp.inf, jnp.inf, jnp.inf)
        i, x, _, _, val, _ = jax.lax.while_loop(cond, body, init)
        return x, val, i

    x, val, nit = run(x0, aux)
    return scipy.optimize.OptimizeResult(
        x=np.asarray(x, dtype=np.float64), fun=float(val), nit=int(nit),
        njev=int(nit), success=True, status=0,
        message="optax on-device convergence")


def optimize_fixed(value_and_grad, loss, x0, gt, needs_ls, maxiter):
    """Run exactly `maxiter` optax steps with no value-dependent stop.

    Pure jax (no jit), so it is safe to `vmap` over a batch -- parameter surveys vmap this
    over a hyperparameter grid, where a while_loop's per-element stop would be ill-defined.
    Returns (x, final_value).
    """
    import jax
    import optax

    def body(_, carry):
        x, st = carry
        val, grad = value_and_grad(x)
        if needs_ls:
            updates, st = gt.update(grad, st, x, value=val, grad=grad, value_fn=loss)
        else:
            updates, st = gt.update(grad, st, x)
        return optax.apply_updates(x, updates), st

    x, _ = jax.lax.fori_loop(0, maxiter, body, (x0, gt.init(x0)))
    return x, loss(x)
