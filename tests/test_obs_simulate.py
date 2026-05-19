"""Tests for ehtim.observing.obs_simulate.sample_vis."""
import os

import numpy as np
import pytest

import ehtim as eh

ARRAY_PATH = os.path.join(os.path.dirname(__file__), "..", "arrays", "EHT2017.txt")

TINT = 60.0
TADV = 600.0
TSTART = 0.0
TSTOP = 24.0
BW = 1e9


@pytest.fixture(scope="module")
def array():
    return eh.array.load_txt(ARRAY_PATH)


@pytest.fixture(scope="module")
def asymmetric_image():
    """Off-centre elongated Gaussian: distinguishes axes and tests centering."""
    im = eh.image.make_empty(64, 200 * eh.RADPERUAS, 17.761, -29.0, rf=230e9)
    im = im.add_gauss(1.0, (60 * eh.RADPERUAS, 20 * eh.RADPERUAS,
                            np.pi / 6, 30 * eh.RADPERUAS, -15 * eh.RADPERUAS))
    return im


@pytest.fixture(scope="module")
def asymmetric_image_pol(asymmetric_image):
    im = asymmetric_image.copy()
    im.add_qu(0.10 * im.imarr(), 0.05 * im.imarr())
    im.add_v(0.02 * im.imarr())
    return im


def _observe(im, array, ttype):
    return im.observe(
        array, TINT, TADV, TSTART, TSTOP, BW,
        ampcal=True, phasecal=True, ttype=ttype, add_th_noise=False,
    )


class TestSampleVisTtypeParity:
    """sample_vis output agrees across ttype='direct' and ttype='nfft'.

    Same uv coverage, same image, same pulse: visibilities should agree to
    finufft's accuracy floor (eps=1e-9 default).
    """

    def test_stokes_i(self, asymmetric_image, array):
        obs_direct = _observe(asymmetric_image, array, "direct")
        obs_nfft = _observe(asymmetric_image, array, "nfft")
        np.testing.assert_allclose(
            obs_nfft.data["vis"], obs_direct.data["vis"],
            rtol=1e-6, atol=1e-9,
        )

    def test_polarimetric_all_stokes(self, asymmetric_image_pol, array):
        obs_direct = _observe(asymmetric_image_pol, array, "direct")
        obs_nfft = _observe(asymmetric_image_pol, array, "nfft")
        for field in ("vis", "qvis", "uvis", "vvis"):
            np.testing.assert_allclose(
                obs_nfft.data[field], obs_direct.data[field],
                rtol=1e-6, atol=1e-9,
                err_msg=f"direct vs nfft mismatch on {field}",
            )
