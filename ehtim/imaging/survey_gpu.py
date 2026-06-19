"""GPU parameter surveys for RML imaging.

`run_survey_gpu` reconstructs a grid of imaging hyperparameters -- the GPU counterpart to the
CPU `paramsurvey` tool in `ehtim/survey.py` (which it leaves untouched). The grid has two kinds
of axis, following the Paper IV survey (EHTC et al. 2019):

* scalar axes -- data/reg-term weights and RegParams scalars (e.g. total flux). These trace
  through the objective as plain multiplies, so the whole sub-grid runs as one vmapped on-device
  optimization.
* host-rebuild axes -- prior FWHM and fractional systematic noise. The prior FWHM rebuilds the
  Gaussian prior image and systematic noise re-derives the data sigmas (closure sigmas come from
  the constituent visibility sigmas, so they cannot just be scaled). Each is an outer loop that
  re-initializes the imager and runs the scalar sub-grid inside.

Static structure (active terms, ttype, pol, fov, prior shape) comes from the Imager. jax / optax
imports are lazy so `import ehtim` stays jax-free.
"""
import numpy as np

from ehtim.const_def import RADPERUAS
from ehtim.imaging.imager_backend import (
    make_survey_value_and_grad,
    transform_imarr,
    unpack_imarr,
)
from ehtim.imaging.optimizers import _resolve_optax, optimize_fixed

NHIST = 50   # optax-lbfgs memory
MAXLS = 5    # zoom line-search cap; small so vmap doesn't pay the batch-worst-case every step


def _gaussian_prior(base, flux, fwhm_rad):
    """Gaussian prior of total flux `flux` and the given FWHM on `base`'s grid.

    Follows the Paper IV recipe: a main Gaussian plus a 1e-3 offset Gaussian that removes the
    first-step gradient singularity. Built on `base`'s grid so it inherits its header and shape.
    """
    pri = base.copy()
    pri.imvec = np.zeros_like(pri.imvec)
    pri = pri.add_gauss(flux, (fwhm_rad, fwhm_rad, 0, 0, 0))
    return pri.add_gauss(flux * 1e-3, (fwhm_rad, fwhm_rad, 0, fwhm_rad, fwhm_rad))


def _inner_survey(imgr, weight_grid, regparam_grid, maxit, x0, optimizer, device):
    """Vmap the scalar sub-grid for the imager's current (already initialized) state.

    Returns (images[B, Npix], objval[B], grid{axis_name: [B]}, chisqs{dname: [B]}) for the
    Cartesian product of the scalar axes; `grid` records each row's swept values (RegParams
    fields prefixed `rp_`), `chisqs` the per-term reduced chi^2 of each reconstruction.
    """
    import jax
    import jax.numpy as jnp

    base_dat, base_reg = imgr.dat_term_next, imgr.reg_term_next
    names = list(weight_grid) + [("rp", f) for f in regparam_grid]
    arrs = [np.asarray(a, float) for a in (*weight_grid.values(), *regparam_grid.values())]
    flat = [m.reshape(-1) for m in np.meshgrid(*arrs, indexing="ij")] if arrs else []
    n_batch = flat[0].size if flat else 1
    swept = dict(zip(names, flat))

    def col(name, fixed):                    # length-B column: swept values or the fixed weight
        return jnp.asarray(swept[name] if name in swept else np.full(n_batch, fixed))

    hparams = {
        "dat_term": {d: col(d, base_dat[d]) for d in base_dat},
        "reg_term": {r: col(r, base_reg[r]) for r in base_reg},
        "reg_params": {f: jnp.asarray(swept[("rp", f)]) for f in regparam_grid},
    }

    vg, loss, put, chisq_dict = make_survey_value_and_grad(
        imgr._init_arr, imgr._config, imgr._which_solve, imgr._data_tuples,
        imgr._logfreqratio_list, len(imgr.obslist_next), imgr._prior_arr, imgr.norm_reg,
        imgr._regparams(), imgr._embed_mask, device=device)

    maxit = int(maxit if maxit is not None else imgr.maxit_next)
    gt, needs_ls = _resolve_optax(optimizer, {"maxiter": maxit, "maxcor": NHIST, "maxls": MAXLS})
    x0 = put(np.asarray(imgr._init_vec, float) if x0 is None else x0)
    init_d = put(np.asarray(imgr._init_arr))
    which_solve, transforms = imgr._which_solve, imgr._config.transforms

    def reconstruct(hp):
        x, val = optimize_fixed(lambda z: vg(z, hp), lambda z: loss(z, hp),
                                x0, gt, needs_ls, maxit)
        image = transform_imarr(unpack_imarr(x, init_d, which_solve), transforms, which_solve)
        return image, val, chisq_dict(image)

    images, objval, chisqs = jax.jit(jax.vmap(reconstruct))(hparams)
    grid = {(f"rp_{n[1]}" if isinstance(n, tuple) else n): v for n, v in swept.items()}
    return (np.asarray(images), np.asarray(objval), grid,
            {k: np.asarray(v) for k, v in chisqs.items()})


def run_survey_gpu(imgr, *, weight_grid=None, regparam_grid=None, prior_fwhm=None,
                   sys_noise=None, maxit=None, x0=None, optimizer="optax-lbfgs-bt",
                   device=None):
    """Reconstruct a hyperparameter grid, vmapping the scalar axes on device.

    Parameters
    ----------
    imgr : ehtim.imager.Imager
        Provides the static structure (data, config, prior, active terms). Restored on return.
    weight_grid : dict, optional
        {term_name: 1-D array} sweeping dat_term / reg_term weights, e.g.
        {"tv": [0, 1, 10, 100], "simple": [1, 10, 100]}.
    regparam_grid : dict, optional
        {field: 1-D array} sweeping RegParams scalars, e.g. {"flux": [0.4, 0.5, 0.6]}.
    prior_fwhm : sequence of float, optional
        Gaussian prior FWHMs in micro-arcseconds; an outer loop rebuilding the prior per value.
    sys_noise : sequence of float, optional
        Fractional systematic noise levels (e.g. 0.02); an outer loop re-deriving sigmas per value.
    maxit : int, optional
        Fixed optax iterations per reconstruction (default imgr.maxit_next).
    x0, optimizer, device
        Shared start vector, optax optimizer name/GradientTransformation, jax device.

    Returns
    -------
    images : np.ndarray, shape (B, Npix)
        Physical image vector for each of the B grid points (Cartesian product of every axis).
    objval : np.ndarray, shape (B,)
        Final objective value for each grid point.
    grid : dict
        {axis_name: 1-D array of length B} giving each row's swept value, for every active axis.
    chisqs : dict
        {data_term: 1-D array of length B} reduced chi^2 of each reconstruction, per data term.
    """
    weight_grid = weight_grid or {}
    regparam_grid = regparam_grid or {}
    fwhms = list(prior_fwhm) if prior_fwhm is not None else [None]
    syss = list(sys_noise) if sys_noise is not None else [None]

    base_prior, base_init = imgr.prior_next, imgr.init_next
    base_obs = list(imgr.obslist_next)
    flux = imgr.flux_next

    images, objval, grid, chisqs = [], [], {}, {}
    try:
        for fwhm in fwhms:
            if fwhm is not None:
                imgr.prior_next = imgr.init_next = _gaussian_prior(base_prior, flux,
                                                                   fwhm * RADPERUAS)
            for sysn in syss:
                if sysn is not None:
                    imgr.obslist_next = [o.add_fractional_noise(sysn) for o in base_obs]
                imgr.init_imager()
                im_b, ob_b, gr_b, ch_b = _inner_survey(imgr, weight_grid, regparam_grid,
                                                       maxit, x0, optimizer, device)
                images.append(im_b)
                objval.append(ob_b)
                for k, v in gr_b.items():
                    grid.setdefault(k, []).append(v)
                for k, v in ch_b.items():
                    chisqs.setdefault(k, []).append(v)
                if fwhm is not None:
                    grid.setdefault("prior_fwhm", []).append(np.full(ob_b.size, fwhm))
                if sysn is not None:
                    grid.setdefault("sys_noise", []).append(np.full(ob_b.size, sysn))
    finally:
        imgr.prior_next, imgr.init_next, imgr.obslist_next = base_prior, base_init, base_obs
        imgr.init_imager()

    grid = {k: np.concatenate(v) for k, v in grid.items()}
    chisqs = {k: np.concatenate(v) for k, v in chisqs.items()}
    return np.concatenate(images), np.concatenate(objval), grid, chisqs
