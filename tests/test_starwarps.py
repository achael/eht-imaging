"""Rectangular-image (xdim != ydim) regression tests for starwarps."""
import numpy as np
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


class TestFlipImg:
    """flipImg returns a flipped Image without mutating the input with a stray attribute."""

    def test_flip_lr_reverses_columns_on_rect(self, make_rect_image):
        """flip_lr produces the column-reversed (ydim, xdim) array."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        flipped = sw.flipImg(im, flip_lr=True, flip_ud=False)
        assert (flipped.xdim, flipped.ydim) == (im.xdim, im.ydim)
        orig_arr = im.imvec.reshape(im.ydim, im.xdim)
        flipped_arr = flipped.imvec.reshape(flipped.ydim, flipped.xdim)
        np.testing.assert_allclose(flipped_arr, orig_arr[:, ::-1], rtol=1e-12)

    def test_flip_ud_reverses_rows_on_rect(self, make_rect_image):
        """flip_ud produces the row-reversed (ydim, xdim) array."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        flipped = sw.flipImg(im, flip_lr=False, flip_ud=True)
        orig_arr = im.imvec.reshape(im.ydim, im.xdim)
        flipped_arr = flipped.imvec.reshape(flipped.ydim, flipped.xdim)
        np.testing.assert_allclose(flipped_arr, orig_arr[::-1, :], rtol=1e-12)

    def test_does_not_set_stray_imec_attribute(self, make_rect_image):
        """flipImg must not mutate the input with the stray `imec` attribute."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        sw.flipImg(im, flip_lr=True, flip_ud=False)
        assert not hasattr(im, "imec")


class TestRotateImg:
    """rotateImg returns a rotated Image; k=1 swaps the (ydim, xdim) axes."""

    def test_rotate_180_preserves_shape_on_rect(self, make_rect_image):
        """k=2 rotates by 180 degrees: shape unchanged."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        rotated = sw.rotateImg(im, k=2)
        assert (rotated.xdim, rotated.ydim) == (im.xdim, im.ydim)

    def test_rotate_90_swaps_dimensions_on_rect(self, make_rect_image):
        """k=1 rotates by 90 degrees: output xdim/ydim swap."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        rotated = sw.rotateImg(im, k=1)
        assert (rotated.xdim, rotated.ydim) == (im.ydim, im.xdim)

    def test_does_not_set_stray_imec_attribute(self, make_rect_image):
        """rotateImg must not mutate the input with the stray `imec` attribute."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        sw.rotateImg(im, k=2)
        assert not hasattr(im, "imec")
