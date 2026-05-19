"""Rectangular-image (xdim != ydim) regression tests for dynamical_imaging.

Locks in the (ydim, xdim) reshape contract on the public image-operation
and averaging entry points so future shape-sensitive changes regress
loudly instead of silently corrupting movie frames.
"""
import numpy as np
import pytest

import ehtim.imaging.dynamical_imaging as di


RECT_XDIM = 32
RECT_YDIM = 48


class TestImageOperations:
    """Image-shape sanity checks for KLS, core position, centering, alignment."""

    def test_kls_identical_images_is_zero(self, make_rect_image):
        """Symmetric KL divergence between an image and itself is exactly zero."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        assert di.get_KLS(im, im) == pytest.approx(0.0, abs=1e-12)

    def test_core_position_is_centered(self, make_rect_image):
        """A centered Gaussian source on a rect grid peaks at the pixel-grid centre."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        ypix, xpix = di.get_core_position(im)
        assert abs(ypix - im.ydim // 2) <= 1
        assert abs(xpix - im.xdim // 2) <= 1

    def test_center_core_preserves_shape_and_flux(self, make_rect_image):
        """Rotating a centered source about its core preserves vector length and flux."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        rotated = di.center_core(im)
        assert rotated.imvec.shape == im.imvec.shape
        assert rotated.total_flux() == pytest.approx(im.total_flux(), rel=1e-10)

    def test_align_left_preserves_shape_and_flux(self, make_rect_image):
        """Horizontal alignment is a pure roll: vector length and flux unchanged."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        aligned = di.align_left(im)
        assert aligned.imvec.shape == im.imvec.shape
        assert aligned.total_flux() == pytest.approx(im.total_flux(), rel=1e-10)


class TestAveraging:
    """Frame-list averaging and Gaussian blurring on rectangular frames."""

    def test_average_of_identical_images_recovers_input(self, make_rect_image):
        """Averaging N copies of the same frame returns that frame bit-for-bit."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        avg = di.average_im_list([im, im, im])
        np.testing.assert_allclose(avg.imvec, im.imvec, rtol=1e-12)

    def test_blur_im_list_preserves_shape_and_total_flux(self, make_rect_image):
        """Gaussian (space + time) blur keeps the per-frame shape and approximate flux."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        blurred = di.blur_im_list([im, im, im], fwhm_x=2 * im.psize, fwhm_t=1.0)
        assert len(blurred) == 3
        for frame in blurred:
            assert frame.imvec.shape == im.imvec.shape
            assert frame.total_flux() == pytest.approx(
                im.total_flux(), rel=1e-2,
            )
