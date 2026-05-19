"""Rectangular-image (xdim != ydim) regression tests for starwarps."""
import numpy as np
import pytest

import ehtim.imaging.starwarps as sw

RECT_XDIM = 32
RECT_YDIM = 48


class TestGkern:
    """gkern returns a normalised 2D Gaussian kernel of the requested size."""

    def test_returns_square_kernel_with_correct_shape(self):
        """gkern(kernlen=N) returns an (N, N) array (legacy default)."""
        kern = sw.gkern(kernlen=9, nsig=2)
        assert kern.shape == (9, 9)

    def test_kernel_is_normalized(self):
        """gkern is flux-preserving (sums to 1)."""
        kern = sw.gkern(kernlen=11, nsig=2)
        assert kern.sum() == pytest.approx(1.0, rel=1e-10)

    def test_rect_kernel_with_explicit_kernlen_y(self):
        """gkern(kernlen=Nx, kernlen_y=Ny) returns a (Ny, Nx) kernel that still sums to 1."""
        kern = sw.gkern(kernlen=8, kernlen_y=12, nsig=2)
        assert kern.shape == (12, 8)
        assert kern.sum() == pytest.approx(1.0, rel=1e-10)


class TestPadNewFOV:
    """padNewFOV pads a (possibly rect) image to a larger isotropic FOV.

    Output is always square at the target FOV regardless of input shape;
    square inputs keep bit-for-bit legacy behavior.
    """

    def test_pads_square_image_to_larger_square(self, gauss_im):
        """Square input -> larger square output (legacy contract preserved)."""
        old_fov_uas = gauss_im.psize * gauss_im.xdim / 4.848136811e-12
        padded = sw.padNewFOV(gauss_im, fov_arcseconds=1.5 * old_fov_uas)
        assert padded.xdim == padded.ydim
        assert padded.xdim > gauss_im.xdim
        assert padded.imvec.sum() == pytest.approx(gauss_im.imvec.sum(), rel=1e-10)

    def test_pads_rect_image_to_square_at_target_fov(self, make_rect_image):
        """Rect input -> square output with side = ceil(newfov / psize); flux preserved."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        old_fov_large_uas = max(im.xdim, im.ydim) * im.psize / 4.848136811e-12
        padded = sw.padNewFOV(im, fov_arcseconds=1.5 * old_fov_large_uas)
        assert padded.xdim == padded.ydim
        assert padded.xdim >= max(im.xdim, im.ydim)
        assert padded.imvec.sum() == pytest.approx(im.imvec.sum(), rel=1e-10)


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

    def test_rotate_180_matches_np_rot90_on_rect(self, make_rect_image):
        """k=2 (180-degree rotation) matches np.rot90 element-by-element."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        rotated = sw.rotateImg(im, k=2)
        orig_arr = im.imvec.reshape(im.ydim, im.xdim)
        rotated_arr = rotated.imvec.reshape(rotated.ydim, rotated.xdim)
        np.testing.assert_allclose(rotated_arr, np.rot90(orig_arr, k=2), rtol=1e-12)

    def test_rotate_90_swaps_dimensions_and_matches_np_rot90(self, make_rect_image):
        """k=1 swaps xdim/ydim and matches np.rot90 element-by-element."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        rotated = sw.rotateImg(im, k=1)
        assert (rotated.xdim, rotated.ydim) == (im.ydim, im.xdim)
        orig_arr = im.imvec.reshape(im.ydim, im.xdim)
        rotated_arr = rotated.imvec.reshape(rotated.ydim, rotated.xdim)
        np.testing.assert_allclose(rotated_arr, np.rot90(orig_arr, k=1), rtol=1e-12)

    def test_does_not_set_stray_imec_attribute(self, make_rect_image):
        """rotateImg must not mutate the input with the stray `imec` attribute."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        sw.rotateImg(im, k=2)
        assert not hasattr(im, "imec")


class TestGradMtx:
    """gradMtx builds a row-major finite-difference operator over the (h, w) grid."""

    def test_y_direction_uses_xdim_offset_on_rect(self):
        """For (w=2, h=3) the y-gradient subtracts the next-row pixel (flat offset w).

        Pre-fix used flat offset h, which is wrong on rect grids; on a square
        h == w it happened to be silently correct.
        """
        G = sw.gradMtx(w=2, h=3, dir='y')
        expected = np.eye(6)
        expected[0, 2] = -1
        expected[1, 3] = -1
        expected[2, 4] = -1
        expected[3, 5] = -1
        expected[4:6, :] = 0
        np.testing.assert_array_equal(G, expected)
