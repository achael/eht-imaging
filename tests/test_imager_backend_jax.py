"""Tests for JAX-compatible backend mirrors in ehtim/imaging/imager_backend_jax.py."""

import inspect

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = pytest.importorskip("jax.numpy")

from ehtim.imaging.imager_backend import compute_embed as compute_embed_np  # noqa: E402
from ehtim.imaging.imager_backend import pack_imarr as pack_imarr_np  # noqa: E402
from ehtim.imaging.imager_backend_jax import compute_embed as compute_embed_jax  # noqa: E402
from ehtim.imaging.imager_backend_jax import pack_imarr as pack_imarr_jax  # noqa: E402


IMAGE_SHAPES = [
    (32, 32),  # square
    (32, 48),  # tall (ydim > xdim)
    (48, 32),  # wide (xdim > ydim)
    (31, 31),  # odd square
    (31, 33),  # odd rectangular
]


class TestComputeEmbedJax:
    """Tests for JAX compute_embed mirror."""

    def test_signature_matches_numpy_backend(self):
        """The JAX mirror keeps the public backend signature."""
        assert inspect.signature(compute_embed_jax) == inspect.signature(compute_embed_np)

    def test_basic_gaussian(self, gauss_im):
        """All-positive Gaussian image gives full mask."""
        im = gauss_im
        embed_mask, coord_matrix = compute_embed_jax(
            jnp.asarray(im.imvec), im.xdim, im.ydim, im.psize, clipfloor=0.0,
        )
        assert np.asarray(embed_mask).shape == (im.xdim * im.ydim,)
        assert np.all(np.asarray(embed_mask))
        assert np.asarray(coord_matrix).shape == (im.xdim * im.ydim, 2)

    def test_clipping(self, gauss_im):
        """High clipfloor masks out low-intensity pixels."""
        im = gauss_im
        clipfloor = np.median(im.imvec)
        embed_mask, coord_matrix = compute_embed_jax(
            jnp.asarray(im.imvec), im.xdim, im.ydim, im.psize, clipfloor=clipfloor,
        )
        n_embed = np.sum(np.asarray(embed_mask))
        assert n_embed < im.xdim * im.ydim
        assert n_embed > 0
        assert np.asarray(coord_matrix).shape == (n_embed, 2)

    def test_clipfloor_too_large(self, gauss_im):
        """Clipfloor above max pixel raises exception."""
        im = gauss_im
        with pytest.raises(Exception, match="clipfloor too large"):
            compute_embed_jax(
                jnp.asarray(im.imvec), im.xdim, im.ydim, im.psize,
                clipfloor=np.max(im.imvec) + 1.0,
            )

    @pytest.mark.parametrize("xdim,ydim", IMAGE_SHAPES)
    def test_shapes(self, make_rect_image, xdim, ydim):
        """JAX backend handles square, rectangular, and odd-dim images correctly."""
        im = make_rect_image(xdim, ydim)
        embed_mask, coord_matrix = compute_embed_jax(
            jnp.asarray(im.imvec), im.xdim, im.ydim, im.psize, clipfloor=0.0,
        )
        assert np.asarray(embed_mask).shape == (xdim * ydim,)
        assert np.asarray(coord_matrix).shape == (xdim * ydim, 2)

    def test_coord_matrix_units(self, gauss_im):
        """Coordinate matrix values are in radians (scaled by psize)."""
        im = gauss_im
        _, coord_matrix = compute_embed_jax(
            jnp.asarray(im.imvec), im.xdim, im.ydim, im.psize, clipfloor=0.0,
        )
        max_coord = (im.xdim // 2) * im.psize
        assert np.all(np.abs(np.asarray(coord_matrix)) <= max_coord + im.psize)


class TestPackImarrJax:
    """Tests for the JAX mirror of imager_backend.pack_imarr."""

    def test_signature_matches_numpy_backend(self):
        """The JAX mirror keeps the public backend signature."""
        assert inspect.signature(pack_imarr_jax) == inspect.signature(pack_imarr_np)

    def test_pack_stokes_i_only(self):
        """which_solve=[1,0,0,0] packs only row 0."""
        imarr = np.arange(40, dtype=float).reshape(4, 10)
        which_solve = np.array([1, 0, 0, 0])
        expected = pack_imarr_np(imarr, which_solve)
        result = pack_imarr_jax(jnp.asarray(imarr), jnp.asarray(which_solve))
        assert np.asarray(result).shape == (10,)
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_pack_full_iquv(self):
        """which_solve=[1,1,1,1] packs all four rows concatenated in order."""
        imarr = np.arange(20, dtype=float).reshape(4, 5)
        which_solve = np.array([1, 1, 1, 1])
        expected = pack_imarr_np(imarr, which_solve)
        result = pack_imarr_jax(jnp.asarray(imarr), jnp.asarray(which_solve))
        assert np.asarray(result).shape == (20,)
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_pack_mixed_solved(self):
        """which_solve=[1,1,0,0] packs rows 0 and 1 in order."""
        imarr = np.array([
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0, 18.0, 19.0],
        ])
        which_solve = np.array([1, 1, 0, 0])
        expected = pack_imarr_np(imarr, which_solve)
        result = pack_imarr_jax(jnp.asarray(imarr), jnp.asarray(which_solve))
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_pack_multifreq_pol_spectral(self):
        """nimage=10 layout: which_solve picks I, mprime, alpha, beta."""
        imarr = np.arange(60, dtype=float).reshape(10, 6)
        which_solve = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0])
        expected = pack_imarr_np(imarr, which_solve)
        result = pack_imarr_jax(jnp.asarray(imarr), jnp.asarray(which_solve))
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_pack_1d_input(self):
        """1D imarr with which_solve=[1] returns the input unchanged."""
        imarr = np.linspace(0.0, 1.0, 7)
        which_solve = np.array([1])
        expected = pack_imarr_np(imarr, which_solve)
        result = pack_imarr_jax(jnp.asarray(imarr), jnp.asarray(which_solve))
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_pack_no_solved_slots_returns_empty(self):
        """which_solve all zero yields an empty 1D array."""
        imarr = np.arange(12, dtype=float).reshape(3, 4)
        which_solve = np.array([0, 0, 0])
        expected = pack_imarr_np(imarr, which_solve)
        result = pack_imarr_jax(jnp.asarray(imarr), jnp.asarray(which_solve))
        assert np.asarray(result).shape == (0,)
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_pack_which_solve_length_mismatch_raises(self):
        """which_solve length must equal imarr's nsolve dimension."""
        imarr = jnp.asarray(np.zeros((4, 5)))
        which_solve = jnp.asarray([1, 0, 0])

        with pytest.raises(Exception, match="inconsistent shape with which_solve"):
            pack_imarr_jax(imarr, which_solve)

    def test_pack_wrong_dim_raises(self):
        """3D imarr is rejected; only 1D and 2D are supported."""
        with pytest.raises(Exception, match="should have one or two dimensions"):
            pack_imarr_jax(jnp.asarray(np.zeros((2, 2, 4))), jnp.asarray([1, 1]))
