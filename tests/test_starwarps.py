"""Rectangular-image (xdim != ydim) regression tests for starwarps."""
import pytest

import ehtim.imaging.starwarps as sw


RECT_XDIM = 32
RECT_YDIM = 48


class TestGkern:
    """gkern returns a normalised 2D Gaussian kernel of the requested size."""

    def test_returns_square_kernel_with_correct_shape(self):
        """gkern(kernlen=N) returns an (N, N) array."""
        kern = sw.gkern(kernlen=9, nsig=2)
        assert kern.shape == (9, 9)

    def test_kernel_is_normalized(self):
        """gkern is flux-preserving (sums to 1)."""
        kern = sw.gkern(kernlen=11, nsig=2)
        assert kern.sum() == pytest.approx(1.0, rel=1e-10)


class TestPadNewFOV:
    """padNewFOV pads a (possibly rect) image to a larger isotropic FOV."""

    def test_pads_square_image_to_larger_square(self, gauss_im):
        """Square image padded to ~1.5x FOV remains square and grows."""
        # The kwarg is misnamed `fov_arcseconds` but the function multiplies by
        # ehtim.RADPERUAS, so the value is effectively in microarcseconds.
        old_fov_uas = gauss_im.psize * gauss_im.xdim / 4.848136811e-12
        padded = sw.padNewFOV(gauss_im, fov_arcseconds=1.5 * old_fov_uas)
        assert padded.xdim == padded.ydim
        assert padded.xdim > gauss_im.xdim
