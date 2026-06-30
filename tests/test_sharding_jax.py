"""Tests for multi-GPU sharding (ehtim.imaging.sharding).

The sharded objective must equal the single-device jax objective and the numpy
analytic gradient bit-for-bit -- the visibility-axis padding + correction is exact,
not approximate, even when Nvis does not divide the device count. Requires >= 2
local GPUs. The value_and_grad is jitted: eager execution of a sharded graph stalls
on per-op collectives (the optimizer loop jits the whole iteration, as here).
"""
import numpy as np
import pytest

import ehtim as eh
from ehtim.imaging.imager_backend import make_value_and_grad_jax

pytestmark = [pytest.mark.jax, pytest.mark.gpu]

jax = pytest.importorskip("jax")
pytest.importorskip("optax")

try:
    _N_GPU = len(jax.devices("gpu"))
except RuntimeError:
    _N_GPU = 0
requires_2gpu = pytest.mark.skipif(_N_GPU < 2, reason="needs >= 2 GPUs")

VALUE_RTOL = 1e-9
GRAD_RTOL = 1e-9
NXCORR_FLOOR = 0.8
EPSILON_TV = 1e-10


def _nxcorr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 0 else 0.0


def _build_imager(obs, gauss_im, gauss_prior):
    return eh.imager.Imager(
        obs, gauss_prior, prior_im=gauss_prior, flux=gauss_im.total_flux(),
        data_term={"vis": 1}, reg_term={"simple": 1, "tv": 1},
        ttype="direct", maxit=100, epsilon_tv=EPSILON_TV)


def _backend_args(imgr):
    return (imgr._init_arr, imgr._config, imgr._which_solve, imgr._data_tuples,
            imgr._logfreqratio_list, len(imgr.obslist_next), imgr.dat_term_next,
            imgr.reg_term_next, imgr._prior_arr, imgr.norm_reg, imgr._regparams(),
            imgr._embed_mask)


def _perturbed_x0(imgr):
    rng = np.random.default_rng(4)
    base = np.asarray(imgr._init_vec, dtype=np.float64)
    return base + 0.1 * rng.standard_normal(base.size)


@requires_2gpu
@pytest.mark.parametrize("ttype", ["direct", "nfft"])
@pytest.mark.parametrize("data_term", [{"vis": 1}, {"amp": 1, "cphase": 1, "logcamp": 1}],
                         ids=["vis", "closures"])
def test_baseline_sharded_matches(obs_direct, gauss_im, gauss_prior, ttype, data_term):
    # baseline (visibility-axis) sharding for both transforms and both linear (vis) and
    # closure (cphase/logcamp) terms must match single-device + numpy bit-for-bit. nfft
    # routes its gradient through a custom_vjp; closures rely on the finite-sample padding.
    from ehtim.imaging.sharding import build_mesh, make_sharded_value_and_grad
    # tight nfft_eps so the jax/numpy nfft accuracy gap stays under GRAD_RTOL for the
    # sharded-vs-numpy check (closures amplify it); sharded-vs-single is exact regardless.
    imgr = eh.imager.Imager(
        obs_direct, gauss_prior, prior_im=gauss_prior, flux=gauss_im.total_flux(),
        data_term=data_term, reg_term={"simple": 1, "tv": 1},
        ttype=ttype, maxit=100, epsilon_tv=EPSILON_TV, nfft_eps=1e-12)
    imgr.init_imager()
    x = _perturbed_x0(imgr)

    vg0, _, to0 = make_value_and_grad_jax(*_backend_args(imgr))
    v0, g0 = jax.jit(vg0)(to0(x))
    v0, g0 = float(v0), np.asarray(g0)

    gnp = np.asarray(imgr.objgrad(np.asarray(x)))

    mesh = build_mesh()
    vg1, _, to1, aux1 = make_sharded_value_and_grad(*_backend_args(imgr), mesh=mesh, shard_axis="baseline")
    v1, g1 = jax.jit(vg1)(to1(x), aux1)
    v1, g1 = float(v1), np.asarray(g1)

    # exact: padding contributes zero and the correction restores the normalization
    assert np.allclose(v1, v0, rtol=VALUE_RTOL)
    assert np.linalg.norm(g1 - g0) / np.linalg.norm(g0) < GRAD_RTOL
    assert np.linalg.norm(g1 - gnp) / np.linalg.norm(gnp) < GRAD_RTOL


@requires_2gpu
@pytest.mark.slow
def test_sharded_make_image_recovers(obs_direct, gauss_im, gauss_prior):
    out = _build_imager(obs_direct, gauss_im, gauss_prior).make_image(shard=True, show_updates=False)
    assert _nxcorr(out.imvec, gauss_im.imvec) > NXCORR_FLOOR


@requires_2gpu
def test_frequency_sharded_matches_single_and_numpy(eht_array, gauss_im):
    # multifrequency: shard the channel axis (channel count padded to the mesh
    # size with a validity mask). Must match single-device + numpy bit-for-bit.
    from ehtim.imaging.sharding import build_mesh, make_sharded_value_and_grad
    im = gauss_im.copy().add_const_mf(1.0, 0)
    prior = im.blur_circ(40 * eh.RADPERUAS)
    obslist = [im.get_image_mf(nu).observe(eht_array, 5, 600, 0, 24, 4e9, ampcal=True,
                                           phasecal=True, ttype="direct", add_th_noise=True, seed=42)
               for nu in (220e9, 240e9)]
    imgr = eh.imager.Imager(obslist, prior, prior_im=prior, flux=im.total_flux(),
                            data_term={"vis": 1}, reg_term={"simple": 1, "tv": 1},
                            ttype="direct", pol="I", mf=True, mf_order=1, maxit=100, epsilon_tv=EPSILON_TV)
    imgr.init_imager()
    x = _perturbed_x0(imgr)

    vg0, _, to0 = make_value_and_grad_jax(*_backend_args(imgr))
    v0, g0 = jax.jit(vg0)(to0(x))
    v0, g0 = float(v0), np.asarray(g0)

    gnp = np.asarray(imgr.objgrad(np.asarray(x)))

    mesh = build_mesh()
    vg1, _, to1, aux1 = make_sharded_value_and_grad(*_backend_args(imgr), mesh=mesh, shard_axis="frequency")
    v1, g1 = jax.jit(vg1)(to1(x), aux1)
    v1, g1 = float(v1), np.asarray(g1)

    assert np.allclose(v1, v0, rtol=VALUE_RTOL)
    assert np.linalg.norm(g1 - g0) / np.linalg.norm(g0) < GRAD_RTOL
    assert np.linalg.norm(g1 - gnp) / np.linalg.norm(gnp) < GRAD_RTOL
