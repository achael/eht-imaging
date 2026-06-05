"""Stokes-I regularizers with a NumPy/JAX backend switch (embed-free terms).

Backend-agnostic value kernels (``xp = array_namespace(imvec)``) for the
regularizers that act directly on the image vector: flux, simple, l1, l1w, lA, gs.
Each mirrors its numpy leaf in ``ehtim.imaging.imager_utils`` so ``jax.grad`` of
the kernel matches the analytic ``reggrad_*`` (the oracle in ``_REGGRAD``).
Spatial regularizers (tv, tvlog, cm, ...) call ``embed`` and are deferred.

Params are kwargs: ``flux``, ``nprior`` (prior image vector), ``norm_reg``, and
for lA also ``psize``, ``beam_size``, ``alpha_A``. The image is ``imvec = exp(x)``.
"""
import math

import ehtim.const_def as ehc
from ehtim.backends import array_namespace
from ehtim.imaging.imager_utils import (
    reggrad_flux,
    reggrad_gs,
    reggrad_l1,
    reggrad_l1w,
    reggrad_lA,
    reggrad_simple,
)


def reg_flux(imvec, **kwargs):
    """Total-flux penalty, (sum(imvec) - flux)^2."""
    xp = array_namespace(imvec)
    flux = kwargs["flux"]
    norm = flux**2 if kwargs.get("norm_reg", True) else 1
    return (xp.sum(imvec) - flux) ** 2 / norm


def reg_simple(imvec, **kwargs):
    """Relative entropy to the prior, sum(imvec * log(imvec / nprior))."""
    xp = array_namespace(imvec)
    priorvec = kwargs["nprior"]
    flux = kwargs["flux"]
    norm = flux if kwargs.get("norm_reg", True) else 1
    return xp.sum(imvec * xp.log(imvec / priorvec)) / norm


def reg_l1(imvec, **kwargs):
    """l1 sparsity, sum(|imvec|)."""
    xp = array_namespace(imvec)
    flux = kwargs["flux"]
    norm = flux if kwargs.get("norm_reg", True) else 1
    return xp.sum(xp.abs(imvec)) / norm


def reg_l1w(imvec, **kwargs):
    """Prior-weighted l1, sum(sqrt(imvec^2 + eps) / (sqrt(nprior^2 + eps) + eps))."""
    xp = array_namespace(imvec)
    priorvec = kwargs["nprior"]
    num = xp.sqrt(imvec**2 + ehc.EP)
    denom = xp.sqrt(priorvec**2 + ehc.EP) + ehc.EP
    return xp.sum(num / denom)


def reg_lA(imvec, **kwargs):
    """Smoothed-l0 (binary-image) penalty, sum(fA(imvec, flux, alpha_A))."""
    xp = array_namespace(imvec)
    psize = kwargs["psize"]
    flux = kwargs["flux"]
    beam_size = kwargs.get("beam_size") or psize
    alpha_A = kwargs.get("alpha_A", 1.0)
    if kwargs.get("norm_reg", True):
        weight_l1 = 1.0 / (1.0 + alpha_A)
        weight_l0 = alpha_A
        norm = (weight_l1 + (beam_size / psize) ** 2 * weight_l0) / (weight_l0 + weight_l1)
    else:
        norm = 1
    fa = 2.0 / math.pi * (1.0 + alpha_A) / alpha_A * xp.arctan(math.pi * alpha_A / 2.0 * xp.abs(imvec) / flux)
    return xp.sum(fa) / norm


def reg_gs(imvec, **kwargs):
    """Gull-Skilling entropy, -sum(imvec - nprior - imvec * log(imvec / nprior))."""
    xp = array_namespace(imvec)
    priorvec = kwargs["nprior"]
    flux = kwargs["flux"]
    norm = flux if kwargs.get("norm_reg", True) else 1
    return -xp.sum(imvec - priorvec - imvec * xp.log(imvec / priorvec)) / norm


_REG = {
    "flux": reg_flux, "simple": reg_simple, "l1": reg_l1,
    "l1w": reg_l1w, "lA": reg_lA, "gs": reg_gs,
}
_REGGRAD = {
    "flux": reggrad_flux, "simple": reggrad_simple, "l1": reggrad_l1,
    "l1w": reggrad_l1w, "lA": reggrad_lA, "gs": reggrad_gs,
}


def reg_loss_log(rtype, x, **kwargs):
    """reg[rtype](exp(x), ...) in log-image coordinates (the autodiff target)."""
    xp = array_namespace(x)
    return _REG[rtype](xp.exp(x), **kwargs)
