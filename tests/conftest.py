"""Shared pytest fixtures for eht-imaging tests."""

import os

import pytest

import ehtim as eh
from ehtim.imaging.imager_backend import (
    POLARIZATION_MODES,
    transform_imarr,
    unpack_imarr,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = os.path.join(os.path.dirname(__file__), "..")
ARRAY_DIR = os.path.join(_ROOT, "arrays")
MODEL_DIR = os.path.join(_ROOT, "models")

# ---------------------------------------------------------------------------
# Common observation parameters
# ---------------------------------------------------------------------------

TINT_SEC = 5
TADV_SEC = 600
TSTART_HR = 0
TSTOP_HR = 24
BW_HZ = 4e9

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def eht_array():
    """Load the EHT 2017 array."""
    return eh.array.load_txt(os.path.join(ARRAY_DIR, "EHT2017.txt"))


@pytest.fixture(scope="session")
def gauss_im():
    """Create a 32x32 Gaussian test image (synthetic, no file dependency)."""
    im = eh.image.make_empty(32, 200 * eh.RADPERUAS, 17.761, -29.0, rf=230e9)
    im = im.add_gauss(1.0, (50 * eh.RADPERUAS, 50 * eh.RADPERUAS, 0, 0, 0))
    return im


@pytest.fixture(scope="session")
def gauss_im_pol(gauss_im):
    """Polarized Gaussian image: Stokes I plus scaled Q, U, V."""
    im = gauss_im.copy()
    im.add_qu(0.10 * im.imarr(), 0.05 * im.imarr())
    im.add_v(0.02 * im.imarr())
    return im


@pytest.fixture(scope="session")
def sgra_im():
    """Load the avery_sgra model image."""
    return eh.image.load_txt(os.path.join(MODEL_DIR, "avery_sgra_eofn.txt"))


@pytest.fixture(scope="session")
def sgra_im_small(sgra_im):
    """Regrid the SgrA model image to 32x32 for fast tests."""
    return sgra_im.regrid_image(sgra_im.fovx(), 32)


@pytest.fixture(scope="session")
def m87_im():
    """Load the jason_mad M87 model image."""
    return eh.image.load_txt(os.path.join(MODEL_DIR, "jason_mad_eofn.txt"))


@pytest.fixture(scope="session")
def m87_im_small(m87_im):
    """Regrid the M87 model image to 32x32 for fast tests."""
    return m87_im.regrid_image(m87_im.fovx(), 32)


@pytest.fixture(scope="session")
def obs_direct(gauss_im, eht_array):
    """Noise-free observation of Gaussian image using direct FT."""
    return gauss_im.observe(
        eht_array, TINT_SEC, TADV_SEC, TSTART_HR, TSTOP_HR, BW_HZ,
        ampcal=True, phasecal=True, ttype="direct", add_th_noise=False,
    )


@pytest.fixture(scope="session")
def obs_fast(gauss_im, eht_array):
    """Noise-free observation of Gaussian image using FFT."""
    return gauss_im.observe(
        eht_array, TINT_SEC, TADV_SEC, TSTART_HR, TSTOP_HR, BW_HZ,
        ampcal=True, phasecal=True, ttype="fast", add_th_noise=False,
    )


@pytest.fixture(scope="session")
def obs_nfft(gauss_im, eht_array):
    """Noise-free observation of Gaussian image using NFFT."""
    pytest.importorskip("pynfft")
    return gauss_im.observe(
        eht_array, TINT_SEC, TADV_SEC, TSTART_HR, TSTOP_HR, BW_HZ,
        ampcal=True, phasecal=True, ttype="nfft", add_th_noise=False,
    )


@pytest.fixture(scope="session")
def obs_noisy(gauss_im, eht_array):
    """Noisy observation of Gaussian image using direct FT."""
    return gauss_im.observe(
        eht_array, TINT_SEC, TADV_SEC, TSTART_HR, TSTOP_HR, BW_HZ,
        ampcal=True, phasecal=True, ttype="direct", add_th_noise=True,
        seed=42,
    )


@pytest.fixture(scope="session")
def obs_pol_direct(gauss_im_pol, eht_array):
    """Noise-free observation of the polarized Gaussian image using direct FT."""
    return gauss_im_pol.observe(
        eht_array, TINT_SEC, TADV_SEC, TSTART_HR, TSTOP_HR, BW_HZ,
        ampcal=True, phasecal=True, ttype="direct", add_th_noise=False,
    )


@pytest.fixture(scope="session")
def obs_gmst(obs_direct):
    """`obs_direct` switched to GMST timetype. Used by tests that exercise GMST branches."""
    return obs_direct.switch_timetype("GMST")


@pytest.fixture(scope="session")
def gauss_prior(gauss_im):
    """A blurred copy of the Gaussian image suitable for use as a prior."""
    return gauss_im.blur_circ(30 * eh.RADPERUAS)


@pytest.fixture(scope="session")
def make_rect_image():
    """Factory fixture: construct a Gaussian image with arbitrary (xdim, ydim)."""
    import numpy as np

    def _factory(xdim, ydim, psize=None):
        if psize is None:
            psize = 200 * eh.RADPERUAS / max(xdim, ydim)
        image_arr = np.zeros((ydim, xdim))
        for i in range(ydim):
            for j in range(xdim):
                x = (j - xdim / 2) * psize
                y = (i - ydim / 2) * psize
                sigma = 50 * eh.RADPERUAS
                image_arr[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        image_arr /= image_arr.sum()
        return eh.image.Image(
            image_arr, psize, 17.761, -29.0,
            polrep="stokes", pol_prim="I", rf=230e9,
        )
    return _factory


@pytest.fixture(scope="session")
def observe(eht_array):
    """Factory fixture: observation with project-standard defaults.

    Closes over `eht_array`. Tests call as `observe(im)` for noise-free output,
    or `observe(im, seed=42)` to enable thermal noise with a deterministic seed.
    """
    def _factory(im, ttype="direct", tstart=TSTART_HR, tstop=TSTOP_HR, seed=None):
        kwargs = dict(
            ampcal=True, phasecal=True, ttype=ttype,
            add_th_noise=seed is not None,
        )
        if seed is not None:
            kwargs["seed"] = seed
        return im.observe(
            eht_array, TINT_SEC, TADV_SEC, tstart, tstop, BW_HZ, **kwargs,
        )
    return _factory


@pytest.fixture(scope="session")
def initialize_imager():
    """Factory fixture: build an Imager and run init_imager.

    Returns (imgr, imcur) where imcur is the unpacked + transformed image
    array ready to pass into make_chisq_dict / compute_chisq_dict. Accepts
    either a single obs or a list of obs.
    """
    def _factory(obs, im, data_term, reg_term=None, pol="I", ttype="direct",
                 mf=False, mf_order=0, mf_flux=None, debias=False, snrcut=0.0,
                 transform=None):
        imgr_kw = dict(
            data_term=data_term, ttype=ttype, pol=pol,
            debias=debias, snrcut=snrcut,
            mf=mf, mf_order=mf_order,
        )
        if reg_term is not None:
            imgr_kw["reg_term"] = reg_term
        if mf_flux is not None:
            imgr_kw["mf_flux"] = mf_flux
        if transform is not None:
            imgr_kw["transform"] = transform
        imgr = eh.imager.Imager(
            obs, im, prior_im=im, flux=im.total_flux(), **imgr_kw,
        )
        if pol in POLARIZATION_MODES:
            imgr.prior_next = imgr.prior_next.switch_polrep(polrep_out="stokes", pol_prim_out="I")
            imgr.init_next = imgr.init_next.switch_polrep(polrep_out="stokes", pol_prim_out="I")
        imgr.check_params()
        imgr.check_limits()
        imgr.init_imager()

        imcur = unpack_imarr(imgr._init_vec, imgr._init_arr, imgr._which_solve)
        imcur = transform_imarr(imcur, imgr.transform_next, imgr._which_solve)
        return imgr, imcur
    return _factory
