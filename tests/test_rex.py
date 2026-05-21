"""Tests for ehtim.features.rex ring-profile extraction.

REX resamples the image with a grid interpolator built as
``RegularGridInterpolator((ys, xs), imarr)`` and samples a physical point
``(x, y)`` as ``rgi((y, x))`` (first axis is ``ys``, second is ``xs``). The
axis-convention test pins that contract; the remaining tests exercise the
interpolation through the public REX entry points.
"""

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

import ehtim as eh
from ehtim.features import rex

R_RING_UAS = 25.0      # radius of the synthetic test ring
RING_WIDTH_UAS = 4.0
NPIX = 64
FOV_UAS = 120.0


@pytest.fixture(scope="module")
def ring_image():
    """A centered Gaussian annulus of known radius, as an ehtim Image."""
    im = eh.image.make_empty(NPIX, FOV_UAS * eh.RADPERUAS, 17.761, -29.0, rf=230e9)
    psize_uas = im.psize / eh.RADPERUAS
    coords = np.arange(NPIX) * psize_uas
    # geometric center is fixed under REX's imarr row-flip, so the recovered
    # center and peak radius are unaffected by that flip.
    center = 0.5 * (NPIX - 1) * psize_uas
    xx, yy = np.meshgrid(coords, coords)
    rr = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    ring = np.exp(-0.5 * ((rr - R_RING_UAS) / RING_WIDTH_UAS) ** 2)
    im.imvec = ring.ravel()
    return im, center


def test_rex_module_imports():
    assert hasattr(rex, "compute_ring_profile")
    assert hasattr(rex, "findCenter")


def test_interp_axis_convention_rectangular():
    """rgi((ys, xs), imarr) sampled as rgi((y, x)) recovers the right value.

    Uses a rectangular grid (ydim != xdim) and a ramp that depends differently
    on x and y, so an axis transpose would either raise on the shape mismatch
    or return the wrong value.
    """
    ydim, xdim = 7, 11
    psize_uas = 2.0
    xs = np.arange(xdim) * psize_uas
    ys = np.arange(ydim) * psize_uas
    xx, yy = np.meshgrid(xs, ys)                 # shape (ydim, xdim)
    imarr = 3.0 * xx + 100.0 * yy                # imarr[i, j] = 3*xs[j] + 100*ys[i]

    rgi = RegularGridInterpolator(
        (ys, xs), imarr, method="linear", bounds_error=False, fill_value=None)

    # exact at grid nodes
    assert np.isclose(rgi((ys[5], xs[8])), imarr[5, 8])

    # interior point sampled as rgi((y, x))
    xq = xs[3] + 0.5 * psize_uas
    yq = ys[2] + 0.25 * psize_uas
    assert np.isclose(rgi((yq, xq)), 3.0 * xq + 100.0 * yq)


def test_compute_ring_profile_recovers_ring_radius(ring_image):
    im, center = ring_image
    pp = rex.compute_ring_profile(im, center, center, rmax=50, interptype="cubic")
    pp.calc_meanprof_and_stats()

    assert pp.meanprof.shape == (pp.nrs,)
    assert np.all(np.isfinite(pp.meanprof))
    # mean radial profile peaks at the ring radius
    assert abs(pp.pkrad - R_RING_UAS) < 3.0


def test_compute_ring_profile_polarized_runs(ring_image):
    im, center = ring_image
    im = im.copy()
    im.add_qu(0.10 * im.imarr(), 0.05 * im.imarr())
    pp = rex.compute_ring_profile(im, center, center, rmax=50, pol_profs=True)

    assert pp.profilesQ.shape[0] == len(pp.thetas)
    assert np.all(np.isfinite(pp.profilesQ))
    assert np.all(np.isfinite(pp.profilesU))


def test_findcenter_locates_ring_center(ring_image):
    im, center = ring_image
    res = rex.findCenter(im, rmin_search=15.0, rmax_search=60.0,
                         nrays_search=25, nrs_search=50)
    assert np.all(np.isfinite(res))
    assert abs(res[0] - center) < 8.0
    assert abs(res[1] - center) < 8.0
