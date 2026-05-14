"""Tests for JAX-compatible backend mirrors in ehtim/imaging/imager_backend_jax.py."""

import inspect

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = pytest.importorskip("jax.numpy")

from ehtim.imaging import imager_backend as backend_np  # noqa: E402
from ehtim.imaging import imager_backend_jax as backend_jax  # noqa: E402


IMAGE_SHAPES = [
    (32, 32),  # square
    (32, 48),  # tall (ydim > xdim)
    (48, 32),  # wide (xdim > ydim)
    (31, 31),  # odd square
    (31, 33),  # odd rectangular
]

DIRECT_STANDARD_DTYPES = [
    "vis", "amp", "logamp", "bs", "cphase", "cphase_diag",
    "camp", "logcamp", "logcamp_diag",
]
DIRECT_POL_DTYPES = ["pvis", "m", "vvis"]
FAST_STANDARD_DTYPES = DIRECT_STANDARD_DTYPES


class TestComputeEmbedJax:
    """Tests for JAX compute_embed mirror."""

    def test_signature_matches_numpy_backend(self):
        """The JAX mirror keeps the public backend signature."""
        assert inspect.signature(backend_jax.compute_embed) == inspect.signature(
            backend_np.compute_embed
        )

    def test_basic_gaussian(self, gauss_im):
        """All-positive Gaussian image gives full mask."""
        im = gauss_im
        embed_mask, coord_matrix = backend_jax.compute_embed(
            jnp.asarray(im.imvec), im.xdim, im.ydim, im.psize, clipfloor=0.0,
        )
        assert np.asarray(embed_mask).shape == (im.xdim * im.ydim,)
        assert np.all(np.asarray(embed_mask))
        assert np.asarray(coord_matrix).shape == (im.xdim * im.ydim, 2)

    def test_clipping(self, gauss_im):
        """High clipfloor masks out low-intensity pixels."""
        im = gauss_im
        clipfloor = np.median(im.imvec)
        embed_mask, coord_matrix = backend_jax.compute_embed(
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
            backend_jax.compute_embed(
                jnp.asarray(im.imvec), im.xdim, im.ydim, im.psize,
                clipfloor=np.max(im.imvec) + 1.0,
            )

    @pytest.mark.parametrize("xdim,ydim", IMAGE_SHAPES)
    def test_shapes(self, make_rect_image, xdim, ydim):
        """JAX backend handles square, rectangular, and odd-dim images correctly."""
        im = make_rect_image(xdim, ydim)
        embed_mask, coord_matrix = backend_jax.compute_embed(
            jnp.asarray(im.imvec), im.xdim, im.ydim, im.psize, clipfloor=0.0,
        )
        assert np.asarray(embed_mask).shape == (xdim * ydim,)
        assert np.asarray(coord_matrix).shape == (xdim * ydim, 2)

    def test_coord_matrix_units(self, gauss_im):
        """Coordinate matrix values are in radians (scaled by psize)."""
        im = gauss_im
        _, coord_matrix = backend_jax.compute_embed(
            jnp.asarray(im.imvec), im.xdim, im.ydim, im.psize, clipfloor=0.0,
        )
        max_coord = (im.xdim // 2) * im.psize
        assert np.all(np.abs(np.asarray(coord_matrix)) <= max_coord + im.psize)


class TestPackImarrJax:
    """Tests for the JAX mirror of imager_backend.pack_imarr."""

    def test_signature_matches_numpy_backend(self):
        """The JAX mirror keeps the public backend signature."""
        assert inspect.signature(backend_jax.pack_imarr) == inspect.signature(
            backend_np.pack_imarr
        )

    def test_pack_stokes_i_only(self):
        """which_solve=[1,0,0,0] packs only row 0."""
        imarr = np.arange(40, dtype=float).reshape(4, 10)
        which_solve = np.array([1, 0, 0, 0])
        expected = backend_np.pack_imarr(imarr, which_solve)
        result = backend_jax.pack_imarr(jnp.asarray(imarr), jnp.asarray(which_solve))
        assert np.asarray(result).shape == (10,)
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_pack_full_iquv(self):
        """which_solve=[1,1,1,1] packs all four rows concatenated in order."""
        imarr = np.arange(20, dtype=float).reshape(4, 5)
        which_solve = np.array([1, 1, 1, 1])
        expected = backend_np.pack_imarr(imarr, which_solve)
        result = backend_jax.pack_imarr(jnp.asarray(imarr), jnp.asarray(which_solve))
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
        expected = backend_np.pack_imarr(imarr, which_solve)
        result = backend_jax.pack_imarr(jnp.asarray(imarr), jnp.asarray(which_solve))
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_pack_multifreq_pol_spectral(self):
        """nimage=10 layout: which_solve picks I, mprime, alpha, beta."""
        imarr = np.arange(60, dtype=float).reshape(10, 6)
        which_solve = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0])
        expected = backend_np.pack_imarr(imarr, which_solve)
        result = backend_jax.pack_imarr(jnp.asarray(imarr), jnp.asarray(which_solve))
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_pack_1d_input(self):
        """1D imarr with which_solve=[1] returns the input unchanged."""
        imarr = np.linspace(0.0, 1.0, 7)
        which_solve = np.array([1])
        expected = backend_np.pack_imarr(imarr, which_solve)
        result = backend_jax.pack_imarr(jnp.asarray(imarr), jnp.asarray(which_solve))
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_pack_no_solved_slots_returns_empty(self):
        """which_solve all zero yields an empty 1D array."""
        imarr = np.arange(12, dtype=float).reshape(3, 4)
        which_solve = np.array([0, 0, 0])
        expected = backend_np.pack_imarr(imarr, which_solve)
        result = backend_jax.pack_imarr(jnp.asarray(imarr), jnp.asarray(which_solve))
        assert np.asarray(result).shape == (0,)
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_pack_which_solve_length_mismatch_raises(self):
        """which_solve length must equal imarr's nsolve dimension."""
        imarr = jnp.asarray(np.zeros((4, 5)))
        which_solve = jnp.asarray([1, 0, 0])

        with pytest.raises(Exception, match="inconsistent shape with which_solve"):
            backend_jax.pack_imarr(imarr, which_solve)

    def test_pack_wrong_dim_raises(self):
        """3D imarr is rejected; only 1D and 2D are supported."""
        with pytest.raises(Exception, match="should have one or two dimensions"):
            backend_jax.pack_imarr(
                jnp.asarray(np.zeros((2, 2, 4))), jnp.asarray([1, 1]),
            )


class TestComputeChisqTermDirectJax:
    """Direct and fast chi^2 parity tests for compute_chisq_term."""

    @staticmethod
    def _full_mask(im):
        return np.ones(im.imvec.size, dtype=bool)

    @staticmethod
    def _partial_mask(im, frac=0.7):
        mask = np.zeros(im.imvec.size, dtype=bool)
        cutoff = np.quantile(im.imvec, 1.0 - frac)
        mask[im.imvec >= cutoff] = True
        return mask

    @staticmethod
    def _imcur_pol(gauss_im_pol):
        iimage = gauss_im_pol.imvec
        qimage = 0.10 * iimage
        uimage = 0.05 * iimage
        vimage = 0.02 * iimage
        rho = np.sqrt(qimage ** 2 + uimage ** 2 + vimage ** 2) / iimage
        phi = np.arctan2(uimage, qimage)
        psi = np.arcsin(
            vimage / np.sqrt(qimage ** 2 + uimage ** 2 + vimage ** 2 + 1e-30)
        )
        return np.array([iimage, rho, phi, psi])

    @staticmethod
    def _data_tuple(obs, prior, mask, dtype, pol, ttype="direct", order=3):
        data, sigma, A = backend_np.compute_chisqdata_term(
            obs, prior, mask, dtype, ttype=ttype, pol=pol, order=order,
        )
        return A, data, sigma

    def test_signature_matches_numpy_backend(self):
        """The JAX mirror keeps the public backend signature."""
        assert inspect.signature(backend_jax.compute_chisq_term) == inspect.signature(
            backend_np.compute_chisq_term
        )

    @pytest.mark.parametrize("dtype", DIRECT_STANDARD_DTYPES)
    def test_parity_direct_standard(self, gauss_im, obs_direct, dtype):
        """Direct standard data terms match the NumPy backend."""
        mask = self._full_mask(gauss_im)
        A, data, sigma = self._data_tuple(obs_direct, gauss_im, mask, dtype, pol="I")
        imcur = np.array([gauss_im.imvec])

        expected = backend_np.compute_chisq_term(
            imcur, dtype, A, data, sigma, ttype="direct", mask=mask,
        )
        result = backend_jax.compute_chisq_term(
            jnp.asarray(imcur), dtype, A, data, sigma, ttype="direct",
            mask=jnp.asarray(mask),
        )

        np.testing.assert_allclose(
            np.asarray(result), expected, rtol=1e-12, atol=1e-15,
        )

    @pytest.mark.parametrize("dtype", DIRECT_POL_DTYPES)
    def test_parity_direct_pol(self, gauss_im_pol, obs_direct, dtype):
        """Direct polarimetric data terms match the NumPy backend."""
        mask = self._full_mask(gauss_im_pol)
        A, data, sigma = self._data_tuple(obs_direct, gauss_im_pol, mask, dtype, pol="IP")
        imcur = self._imcur_pol(gauss_im_pol)

        expected = backend_np.compute_chisq_term(
            imcur, dtype, A, data, sigma, ttype="direct", mask=mask,
        )
        result = backend_jax.compute_chisq_term(
            jnp.asarray(imcur), dtype, A, data, sigma, ttype="direct",
            mask=jnp.asarray(mask),
        )

        np.testing.assert_allclose(
            np.asarray(result), expected, rtol=1e-12, atol=1e-15,
        )

    @pytest.mark.parametrize("dtype", FAST_STANDARD_DTYPES)
    def test_parity_fast_standard_order1(self, gauss_im, obs_fast, dtype):
        """Fast standard data terms match the NumPy backend for linear sampling."""
        mask = self._partial_mask(gauss_im)
        A, data, sigma = self._data_tuple(
            obs_fast, gauss_im, mask, dtype, pol="I", ttype="fast", order=1,
        )
        imvec = gauss_im.imvec[mask]
        imcur = np.array([imvec])

        expected = backend_np.compute_chisq_term(
            imcur, dtype, A, data, sigma, ttype="fast", mask=mask,
        )
        result = backend_jax.compute_chisq_term(
            jnp.asarray(imcur), dtype, A, data, sigma, ttype="fast",
            mask=jnp.asarray(mask),
        )

        np.testing.assert_allclose(
            np.asarray(result), expected, rtol=1e-10, atol=1e-12,
        )

    def test_direct_vis_jit_with_static_dispatch(self, gauss_im, obs_direct):
        """A closed-over direct vis loss compiles when dispatch stays static."""
        mask = self._full_mask(gauss_im)
        A, data, sigma = self._data_tuple(obs_direct, gauss_im, mask, "vis", pol="I")
        expected = backend_np.compute_chisq_term(
            gauss_im.imvec, "vis", A, data, sigma, ttype="direct", mask=mask,
        )

        loss = jax.jit(
            lambda imvec: backend_jax.compute_chisq_term(
                imvec, "vis", A, data, sigma, ttype="direct", mask=jnp.asarray(mask),
            )
        )
        result = loss(jnp.asarray(gauss_im.imvec))

        np.testing.assert_allclose(
            np.asarray(result), expected, rtol=1e-12, atol=1e-15,
        )

    def test_fast_order3_not_implemented(self, gauss_im, obs_fast):
        """JAX map_coordinates does not support the backend's default cubic sampler."""
        mask = self._full_mask(gauss_im)
        A, data, sigma = self._data_tuple(
            obs_fast, gauss_im, mask, "vis", pol="I", ttype="fast", order=3,
        )
        with pytest.raises(NotImplementedError, match="order <= 1"):
            backend_jax.compute_chisq_term(
                jnp.asarray(gauss_im.imvec), "vis", A, data, sigma,
                ttype="fast", mask=jnp.asarray(mask),
            )

    def test_nfft_not_implemented(self, gauss_im):
        """NFFT losses are not implemented in this pass."""
        with pytest.raises(NotImplementedError, match="not supported"):
            backend_jax.compute_chisq_term(
                jnp.asarray(gauss_im.imvec), "vis", None, None, None,
                ttype="nfft",
            )

    def test_unknown_dtype_raises(self, gauss_im):
        """Unknown data terms raise like the NumPy backend."""
        with pytest.raises(Exception, match="data term .* not recognized"):
            backend_jax.compute_chisq_term(
                jnp.asarray(gauss_im.imvec), "bogus", None, None, None,
                ttype="direct",
            )
