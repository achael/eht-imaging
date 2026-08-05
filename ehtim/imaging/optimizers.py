"""Pluggable optimizer dispatch for the imaging objective.

`run_optimizer` lets `Imager.make_image` drive any optimizer through one seam:
the scipy L-BFGS-B default (unchanged), an optax optimizer running on-device, or
a user-supplied callable. It always returns a `scipy.optimize.OptimizeResult`, so
the caller unpacks `.x`/`.fun` the same way for every backend.
"""
import warnings
from functools import partial

import numpy as np
import scipy.optimize

# Built-in optimizer names, resolved in their respective paths. Listed here so
# classify_optimizer can route the names without importing optax.
_OPTAX_NAMES = frozenset({"optax-lbfgs", "optax-lbfgs-bt", "adam", "adamw", "sgd", "rmsprop"})

# Accepted optimizer names mapped to the scipy method they select. The imaging problem is
# unconstrained (positivity comes from the log transform, pol fractions from mcv), so
# L-BFGS-B is used as plain L-BFGS and the gradient-based unconstrained methods drop in.
_SCIPY_METHODS = {
    "lbfgs": "L-BFGS-B", "l-bfgs-b": "L-BFGS-B", "scipy": "L-BFGS-B",
    "scipy-lbfgs": "L-BFGS-B", "bfgs": "BFGS", "cg": "CG",
    "newton-cg": "Newton-CG", "tnc": "TNC", "slsqp": "SLSQP",
}
_SCIPY_NAMES = frozenset(_SCIPY_METHODS)

# Which of the imager's option keys each method actually reads. Hand-written rather than
# introspected, because scipy's per-method option lists are private. Passing a key a method
# does not know is not an error: scipy warns and drops it, which silently disables the
# stopping rule the caller asked for. TNC is the worst of these, dropping maxiter.
_METHOD_OPTS = {
    "L-BFGS-B": {"maxiter", "ftol", "gtol", "maxcor", "maxls"},
    "BFGS": {"maxiter", "gtol"},
    "CG": {"maxiter", "gtol"},
    "Newton-CG": {"maxiter", "xtol"},
    "TNC": {"maxfun", "ftol", "gtol"},
    "SLSQP": {"maxiter", "ftol"},
}

# BFGS carries a dense N x N inverse Hessian: 2 GiB at a 128x128 image, 32 GiB at 256x256.
_BFGS_DENSE_WARN_BYTES = 1 << 30


def _scipy_options(method, optdict):
    """Keep only the options `method` reads, renaming where its spelling differs.

    Parameters
    ----------
    method : str
        A scipy.optimize.minimize method name.
    optdict : dict
        The imager's options: maxiter, ftol, gtol, maxcor, maxls.

    Returns
    -------
    dict
        Options safe to hand to this method.
    """
    opts = {k: v for k, v in optdict.items() if k in _METHOD_OPTS[method]}
    if False:
        opts["maxfun"] = optdict["maxiter"]     # TNC caps evaluations, not iterations
    if method == "Newton-CG" and "ftol" in optdict:
        opts["xtol"] = optdict["ftol"]          # its only tolerance
    return opts


def _warn_if_bfgs_hessian_is_large(method, n):
    """Warn before BFGS allocates a dense inverse Hessian that will not fit."""
    if method != "BFGS":
        return
    nbytes = 8 * n * n
    if nbytes > _BFGS_DENSE_WARN_BYTES:
        warnings.warn(
            f"BFGS stores a dense {n} x {n} inverse Hessian, about "
            f"{nbytes / 2**30:.1f} GiB for this image. Use optimizer='lbfgs' unless you "
            f"have the memory for it.", ResourceWarning, stacklevel=3)


def classify_optimizer(optimizer):
    """Return the path that handles `optimizer`: 'scipy', 'optax', or 'callable'.

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


def run_optimizer(optimizer, build_loss, *, x0, optdict, callback=None, device=None):
    """Minimize the imaging objective with whichever optimizer the caller picked.

    There are three paths. Scipy is the historical default and is a straight
    pass-through to L-BFGS-B. The optax path keeps the image vector and the
    optimizer state on the GPU for the whole solve, so there is no host round trip
    per iteration. The callable path is an escape hatch for anything else.

    The objective arrives as a builder rather than ready-made because the paths
    need different things from it: the host paths want plain numpy callables,
    while the optax path wants device arrays and a jax value_and_grad. The caller
    passes the builder that matches its path, and each path calls it with the
    signature it expects, so a mismatch fails immediately.

    Parameters
    ----------
    optimizer : None, str, optax.GradientTransformation, or callable
        Picks the path (see `classify_optimizer`). None keeps scipy L-BFGS-B.
    build_loss : callable
        Host paths: `() -> (fun, jac)`, where `jac` is a gradient function, True if
        `fun` already returns (value, grad), or None if there is no gradient.
        Optax path: `(device) -> (value_and_grad, loss, to_device, aux)`.
    x0 : numpy.ndarray
        Initial solver vector.
    optdict : dict
        scipy L-BFGS-B options ('maxiter', 'ftol', 'gtol', 'maxcor', 'maxls'). The
        optax path stops on maxiter/ftol/gtol and configures itself from maxcor/maxls.
    callback : callable, optional
        Called with the current x each iteration (host paths only).
    device : optional
        Device for the optax path.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Whatever the path, so callers read `.x` and `.fun` the same way.
    """
    kind = classify_optimizer(optimizer)

    if kind == "scipy":
        fun, jac = build_loss()
        method = _SCIPY_METHODS["scipy" if optimizer is None else optimizer.lower()]
        _warn_if_bfgs_hessian_is_large(method, np.size(x0))
        return scipy.optimize.minimize(fun, x0, method=method, jac=jac,
                                       options=_scipy_options(method, optdict),
                                       callback=callback)

    elif kind == "callable":
        # The escape hatch: hand the user a host value_and_grad(x) -> (value, grad).
        fun, jac = build_loss()
        if jac is True:
            value_and_grad = fun           # fun already returns (value, grad)
        elif jac is None:
            # no analytic gradient (grads=False); the scipy path would finite-difference
            # here, so hand the user None and let their optimizer decide
            def value_and_grad(x):
                return fun(x), None
        else:
            def value_and_grad(x):
                return fun(x), jac(x)
        return optimizer(value_and_grad, x0, maxiter=optdict["maxiter"],
                         tol=optdict["gtol"], callback=callback)

    else:
        # kind == "optax": run an optax optimizer entirely on device.
        gt, needs_ls = resolve_optax(optimizer, optdict)
        value_and_grad, loss, to_device, aux = build_loss(device)
        return _run_optax(gt, needs_ls, value_and_grad, loss, to_device(x0), optdict, aux=aux)


_DEFAULT_LR = 1e-2  # step size for the first-order optax builtins (adam/sgd/...)


def resolve_optax(optimizer, optdict):
    """Return (gradient_transformation, needs_linesearch) for the optax path.

    A string names a built-in; an optax.GradientTransformation is used as given.
    L-BFGS honors maxcor (memory) and maxls (line-search steps) from optdict; the
    first-order builtins use _DEFAULT_LR (pass your own GradientTransformation to
    control the step size).

    `needs_linesearch` tells the step function below whether this optimizer wants
    the current value and a way to re-evaluate the loss, which line searches do
    and plain first-order rules do not.
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
    if name not in builders:
        raise ValueError(
            f"unknown optax optimizer {optimizer!r}; expected one of "
            f"{sorted(_OPTAX_NAMES)}, or pass an optax GradientTransformation.")
    return builders[name](_DEFAULT_LR), False


def _optax_step(gt, needs_ls, value_and_grad, loss, x, state):
    """Take one optax step. Returns (x, state, value, gradient).

    Both drivers below share this, so the update rule is written once and they
    differ only in when they stop.
    """
    import optax

    val, grad = value_and_grad(x)
    if needs_ls:
        # a line search also needs the current value and a way to re-evaluate the loss
        updates, state = gt.update(grad, state, x, value=val, grad=grad, value_fn=loss)
    else:
        updates, state = gt.update(grad, state, x)
    return optax.apply_updates(x, updates), state, val, grad


def _run_optax(gt, needs_ls, value_and_grad, loss, x0, optdict, aux=None):
    """Minimize on device, stopping on gradient norm, value change, or maxiter.

    This is the single-image driver. The whole loop is one jitted while_loop, so x
    and the optimizer state stay on the GPU from start to finish with no host sync
    per step. `maxiter` caps the iteration count just like the same-named scipy
    option, and the tolerances stop it earlier once it has converged.

    `aux` carries the sharded data (None on a single device). It has to be a jit
    argument rather than a closure, because closing over sharded arrays makes jax
    re-partition them.

    Returns a scipy OptimizeResult.
    """
    import jax
    import jax.numpy as jnp

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
            x, st, val, grad = _optax_step(gt, needs_ls, vg, lossfn, x, st)
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
    """Run exactly `maxiter` optax steps, with no convergence check.

    Same update as `_run_optax`, different stopping rule. A parameter survey solves
    a whole grid of hyperparameters at once under `vmap`, and there "stop when this
    one has converged" has no meaning for a batch, so this driver always runs the
    full count. It is also left un-jitted for the caller to jit around the vmap.
    Use `_run_optax` for a single image, where stopping early is worth having.

    Returns (x, final_value).
    """
    import jax

    def body(_, carry):
        x, st = carry
        x, st, _, _ = _optax_step(gt, needs_ls, value_and_grad, loss, x, st)
        return x, st

    x, _ = jax.lax.fori_loop(0, maxiter, body, (x0, gt.init(x0)))
    return x, loss(x)
