"""Shared pytest fixtures for eht-imaging tests."""

import os

import pytest

import ehtim as eh

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
def gauss_prior(gauss_im):
    """A blurred copy of the Gaussian image suitable for use as a prior."""
    return gauss_im.blur_circ(30 * eh.RADPERUAS)
