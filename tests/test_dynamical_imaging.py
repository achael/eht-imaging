"""Rectangular-image (xdim != ydim) regression tests for dynamical_imaging.

Locks in the (ydim, xdim) reshape contract on the public image-operation
and averaging entry points so future shape-sensitive changes regress
loudly instead of silently corrupting movie frames.
"""
import numpy as np
import pytest

import ehtim as eh
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

    def test_align_left_rolls_to_x_axis_center_on_rect(self, make_rect_image):
        """A single bright column at x=0 lands at the x-axis centre after align.

        Catches the bug where the roll amount uses the y-centre instead of
        the x-centre, which is silent on square images but visibly wrong on
        rectangular ones.
        """
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        arr = np.zeros((im.ydim, im.xdim))
        arr[:, 0] = 1.0
        arr /= arr.sum()
        bright = im.copy()
        bright.imvec = arr.flatten()

        aligned = di.align_left(bright, min_frac=0.1)

        new_arr = aligned.imvec.reshape(aligned.ydim, aligned.xdim)
        col_with_bright = int(np.argmax(new_arr.sum(axis=0)))
        assert col_with_bright == (aligned.xdim - 1) // 2


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


class TestStaticRegularizer:
    """static_regularizer + static_regularizer_gradient on rectangular frames.

    Exercises the explicit xdim/ydim kwargs that prevent silent square-image
    dimension inference from corrupting regularizer values on rect grids.
    """

    @staticmethod
    def _single_frame_inputs(im):
        """Pack an Image into the (1, ydim, xdim) frame stack the helpers expect."""
        frame = im.imvec.reshape(im.ydim, im.xdim)
        mask = np.ones(im.xdim * im.ydim, dtype=bool)
        return (
            np.array([frame]),
            np.array([frame]),
            np.array([mask]),
            im.total_flux(),
            im.psize,
        )

    def test_tv_value_is_finite_on_rect_frame(self, make_rect_image):
        """Total-variation regularizer on a rect frame returns a finite scalar."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        frames, priors, masks, flux, psize = self._single_frame_inputs(im)
        value = di.static_regularizer(
            frames, priors, masks, flux, psize,
            stype="tv", xdim=im.xdim, ydim=im.ydim,
        )
        assert np.isfinite(value)

    def test_tv_gradient_shape_matches_image_on_rect_frame(self, make_rect_image):
        """TV-regularizer gradient on a rect frame returns the full-pixel vector."""
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        frames, priors, masks, flux, psize = self._single_frame_inputs(im)
        grad = di.static_regularizer_gradient(
            frames, priors, masks, flux, psize,
            stype="tv", xdim=im.xdim, ydim=im.ydim,
        )
        assert grad.shape == im.imvec.shape
        assert np.all(np.isfinite(grad))


class TestContPlotter:
    """Cont contour-plotter axis ordering and extent on rectangular frames."""

    @staticmethod
    def _spy_contour(monkeypatch):
        """Return (captured_dict, side-effect-free fake) for plt.contour / plt.show."""
        import matplotlib.pyplot as plt
        captured = {}

        def fake_contour(Z, *args, **kwargs):
            captured["Z_shape"] = Z.shape
            captured["extent"] = kwargs.get("extent")
            return None

        monkeypatch.setattr(plt, "contour", fake_contour)
        monkeypatch.setattr(plt, "show", lambda: None)
        return captured

    def test_reshapes_rect_image_to_ydim_xdim_grid(self, make_rect_image,
                                                   monkeypatch):
        """Cont's internal 2D grid follows the (ydim, xdim) row-major convention."""
        captured = self._spy_contour(monkeypatch)
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        di.Cont(im)
        assert captured["Z_shape"] == (im.ydim, im.xdim)

    def test_extent_uses_separate_x_and_y_field_of_view(self, make_rect_image,
                                                        monkeypatch):
        """Contour extent is (-povx/2, povx/2, -povy/2, povy/2) with separate axes."""
        captured = self._spy_contour(monkeypatch)
        im = make_rect_image(RECT_XDIM, RECT_YDIM)
        di.Cont(im)
        pov_x_mas = im.xdim * im.psize / (eh.RADPERUAS * 1.e3)
        pov_y_mas = im.ydim * im.psize / (eh.RADPERUAS * 1.e3)
        expected = (-pov_x_mas / 2., pov_x_mas / 2.,
                    -pov_y_mas / 2., pov_y_mas / 2.)
        assert captured["extent"] == pytest.approx(expected, rel=1e-12)
