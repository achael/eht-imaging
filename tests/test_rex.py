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


def test_compute_ring_profile_rectangular():
    """Ring extraction on a rectangular image (xdim != ydim).

    The interpolator is built as RGI((ys, xs), imarr) with imarr of shape
    (ydim, xdim); a swapped axis order would raise on the shape mismatch here
    even though it is silent on square images.
    """
    xdim, ydim = 48, 64
    psize = 200 * eh.RADPERUAS / max(xdim, ydim)
    psize_uas = psize / eh.RADPERUAS
    xs = np.arange(xdim) * psize_uas
    ys = np.arange(ydim) * psize_uas
    xc = 0.5 * (xdim - 1) * psize_uas
    yc = 0.5 * (ydim - 1) * psize_uas
    xx, yy = np.meshgrid(xs, ys)                 # (ydim, xdim)
    rr = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
    ring = np.exp(-0.5 * ((rr - R_RING_UAS) / RING_WIDTH_UAS) ** 2)
    im = eh.image.Image(ring, psize, 17.761, -29.0,
                        polrep="stokes", pol_prim="I", rf=230e9)

    pp = rex.compute_ring_profile(im, xc, yc, rmax=40, interptype="cubic")
    pp.calc_meanprof_and_stats()

    assert np.all(np.isfinite(pp.meanprof))
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


# ---------------------------------------------------------------------------
# Off-centre ring + FindProfile entry point
# ---------------------------------------------------------------------------


# Offset for the off-centre ring test: shift the ring by 8 uas in (+x, -y)
# so findCenter has to actually search rather than pick the geometric centre.
RING_OFFSET_UAS = (8.0, -8.0)


@pytest.fixture(scope="module")
def off_centre_ring_image():
    """Gaussian annulus centred off the image's geometric centre."""
    im = eh.image.make_empty(NPIX, FOV_UAS * eh.RADPERUAS, 17.761, -29.0, rf=230e9)
    psize_uas = im.psize / eh.RADPERUAS
    coords = np.arange(NPIX) * psize_uas
    cx = 0.5 * (NPIX - 1) * psize_uas + RING_OFFSET_UAS[0]
    cy = 0.5 * (NPIX - 1) * psize_uas + RING_OFFSET_UAS[1]
    xx, yy = np.meshgrid(coords, coords)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring = np.exp(-0.5 * ((rr - R_RING_UAS) / RING_WIDTH_UAS) ** 2)
    im.imvec = ring.ravel()
    return im, (cx, cy)


def test_findcenter_locates_off_centre_ring(off_centre_ring_image):
    im, (cx, cy) = off_centre_ring_image
    x0, y0 = rex.findCenter(im)[:2]
    # findCenter returns the ring centre in uas in REX's flipped-row frame.
    # Recovery within one psize is enough to pin "the search actually moves".
    psize_uas = im.psize / eh.RADPERUAS
    assert abs(x0 - cx) < psize_uas
    # y centre in flipped frame: the row flip mirrors y about the centre, so
    # compare to the mirrored coordinate.
    cy_flipped = (NPIX - 1) * psize_uas - cy
    assert abs(y0 - cy_flipped) < psize_uas


def test_findprofile_returns_ringprofile_with_populated_meanprof(ring_image):
    im, _ = ring_image
    pp = rex.FindProfile(im)
    assert hasattr(pp, "meanprof")
    assert np.any(pp.meanprof != 0)
    assert pp.meanprof.shape == pp.rs.shape


# ---------------------------------------------------------------------------
# RingProfile plotting — Agg smoke
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _agg_backend():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    yield plt
    plt.close("all")


@pytest.fixture(scope="module")
def ring_profile(ring_image):
    im, _ = ring_image
    return rex.FindProfile(im)


@pytest.mark.parametrize("method", [
    "plot_img", "plot_unwrapped", "plot_profs",
    "plot_prof_band", "plot_meanprof", "plot_meanprof_theta",
])
def test_ringprofile_plot_smoke(ring_profile, _agg_backend, method):
    # Render with save_png=False so no files land on disk.
    getattr(ring_profile, method)(save_png=False)
