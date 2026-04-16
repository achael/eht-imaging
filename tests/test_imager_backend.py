"""Tests for pure functional backend in ehtim/imaging/imager_backend.py.

Each test verifies that a backend function produces identical output
to the corresponding Imager class method.
"""

import numpy as np
import pytest

import ehtim as eh
from ehtim.imaging.imager_backend import compute_embed


class TestComputeEmbed:
    """Tests for compute_embed (extracted from Imager.set_embed)."""

    def test_basic_gaussian(self, gauss_im):
        """All-positive Gaussian image gives full mask."""
        im = gauss_im
        embed_mask, coord_matrix = compute_embed(
            im.imvec, im.xdim, im.ydim, im.psize, clipfloor=0.0,
        )
        assert embed_mask.shape == (im.xdim * im.ydim,)
        assert np.all(embed_mask)
        assert coord_matrix.shape == (im.xdim * im.ydim, 2)

    def test_clipping(self, gauss_im):
        """High clipfloor masks out low-intensity pixels."""
        im = gauss_im
        clipfloor = np.median(im.imvec)
        embed_mask, coord_matrix = compute_embed(
            im.imvec, im.xdim, im.ydim, im.psize, clipfloor=clipfloor,
        )
        n_embed = np.sum(embed_mask)
        assert n_embed < im.xdim * im.ydim
        assert n_embed > 0
        assert coord_matrix.shape == (n_embed, 2)

    def test_clipfloor_too_large(self, gauss_im):
        """Clipfloor above max pixel raises exception."""
        im = gauss_im
        with pytest.raises(Exception, match="clipfloor too large"):
            compute_embed(
                im.imvec, im.xdim, im.ydim, im.psize,
                clipfloor=np.max(im.imvec) + 1.0,
            )

    def test_matches_imager(self, gauss_im, eht_array):
        """Backend compute_embed matches Imager.set_embed exactly."""
        obs = gauss_im.observe(
            eht_array, 5, 600, 0, 24, 4e9,
            ampcal=True, phasecal=True, ttype="direct", add_th_noise=False,
        )
        imgr = eh.imager.Imager(obs, gauss_im, gauss_im, gauss_im.total_flux())

        # Call the Imager method
        imgr.set_embed()

        # Call the backend function with the same args
        embed_mask, coord_matrix = compute_embed(
            imgr.prior_next.imvec, imgr.prior_next.xdim,
            imgr.prior_next.ydim, imgr.prior_next.psize,
            imgr.clipfloor_next,
        )

        np.testing.assert_array_equal(embed_mask, imgr._embed_mask)
        np.testing.assert_array_equal(coord_matrix, imgr._coord_matrix)

    def test_odd_dimensions(self):
        """Odd image dimensions are handled correctly."""
        im = eh.image.make_empty(31, 200 * eh.RADPERUAS, 0.0, 0.0)
        im = im.add_gauss(1.0, (50 * eh.RADPERUAS, 50 * eh.RADPERUAS, 0, 0, 0))
        embed_mask, coord_matrix = compute_embed(
            im.imvec, im.xdim, im.ydim, im.psize, clipfloor=0.0,
        )
        assert embed_mask.shape == (31 * 31,)
        assert coord_matrix.shape == (31 * 31, 2)

    def test_coord_matrix_units(self, gauss_im):
        """Coordinate matrix values are in radians (scaled by psize)."""
        im = gauss_im
        _, coord_matrix = compute_embed(
            im.imvec, im.xdim, im.ydim, im.psize, clipfloor=0.0,
        )
        # Coordinates should be within +/- (xdim/2) * psize
        max_coord = (im.xdim // 2) * im.psize
        assert np.all(np.abs(coord_matrix) <= max_coord + im.psize)
