"""Tests for Image methods.

Method-by-method coverage of ehtim/image.py. Sections mirror the structure
of the class itself and follow the pattern established by tests/test_obsdata.py:
session-scoped fixtures from conftest.py (gauss_im, gauss_im_pol, eht_array,
obs_direct, ...) are reused so the expensive constructors are amortised
across the whole module.
"""

import numpy as np
import pytest

import ehtim as eh

# ---------------------------------------------------------------------------
# Section 1: Construction, basic state, image_args, copy
# ---------------------------------------------------------------------------


def test_init_requires_2d_array():
    with pytest.raises(Exception, match="2D"):
        eh.image.Image(np.zeros(16), 1e-10, 17.761, -29.0)


def test_init_rejects_bad_polrep():
    with pytest.raises(Exception, match="polrep"):
        eh.image.Image(np.zeros((4, 4)), 1e-10, 17.761, -29.0, polrep="lin")


def test_init_rejects_bad_pol_prim_stokes():
    with pytest.raises(Exception, match="pol_prim"):
        eh.image.Image(np.zeros((4, 4)), 1e-10, 17.761, -29.0,
                       polrep="stokes", pol_prim="RR")


def test_init_rejects_bad_pol_prim_circ():
    with pytest.raises(Exception, match="pol_prim"):
        eh.image.Image(np.zeros((4, 4)), 1e-10, 17.761, -29.0,
                       polrep="circ", pol_prim="I")


def test_init_default_polrep_is_stokes(gauss_im):
    assert gauss_im.polrep == "stokes"
    assert gauss_im.pol_prim == "I"


def test_init_xdim_ydim_from_image_shape():
    arr = np.zeros((5, 7))  # (ydim, xdim)
    im = eh.image.Image(arr, 1e-10, 17.761, -29.0)
    assert im.ydim == 5
    assert im.xdim == 7


def test_init_time_over_24_wraps_into_mjd():
    arr = np.zeros((4, 4))
    im = eh.image.Image(arr, 1e-10, 17.761, -29.0,
                        mjd=58000, time=27.5)
    assert im.mjd == 58001
    assert im.time == pytest.approx(3.5)


def test_image_args_roundtrip(gauss_im):
    arglist, argdict = gauss_im.image_args()
    rebuilt = eh.image.Image(*arglist, **argdict)
    assert rebuilt.xdim == gauss_im.xdim
    assert rebuilt.ydim == gauss_im.ydim
    assert rebuilt.psize == gauss_im.psize
    assert rebuilt.polrep == gauss_im.polrep
    np.testing.assert_array_equal(rebuilt.imvec, gauss_im.imvec)


def test_copy_is_deep(gauss_im):
    other = gauss_im.copy()
    other._imdict[other.pol_prim][0] += 1.0
    assert other.imvec[0] != gauss_im.imvec[0]
    assert other is not gauss_im


def test_copy_preserves_pol_images(gauss_im_pol):
    other = gauss_im_pol.copy()
    np.testing.assert_array_equal(other.qvec, gauss_im_pol.qvec)
    np.testing.assert_array_equal(other.uvec, gauss_im_pol.uvec)
    np.testing.assert_array_equal(other.vvec, gauss_im_pol.vvec)


def test_copy_preserves_mflist(gauss_im):
    im = gauss_im.add_const_mf(alpha=2.5, beta=0.1)
    other = im.copy()
    np.testing.assert_array_equal(other.specvec, im.specvec)
    np.testing.assert_array_equal(other.curvvec, im.curvvec)


# ---------------------------------------------------------------------------
# Section 2: Imvec / property getters & setters
# ---------------------------------------------------------------------------


def test_imvec_get_matches_imdict(gauss_im):
    np.testing.assert_array_equal(gauss_im.imvec, gauss_im._imdict["I"])


def test_imvec_setter_rejects_wrong_size(gauss_im):
    im = gauss_im.copy()
    with pytest.raises(Exception, match="xdim"):
        im.imvec = np.zeros(5)


def test_imvec_setter_updates_imdict(gauss_im):
    im = gauss_im.copy()
    new = np.arange(im.xdim * im.ydim, dtype=float)
    im.imvec = new
    np.testing.assert_array_equal(im._imdict["I"], new)


@pytest.mark.parametrize("propname", ["specvec", "curvvec", "specvec_pol",
                                       "curvvec_pol", "rmvec", "cmvec"])
def test_mflist_setter_rejects_wrong_size(gauss_im, propname):
    im = gauss_im.copy()
    with pytest.raises(Exception, match="xdim"):
        setattr(im, propname, np.zeros(5))


@pytest.mark.parametrize("propname,idx", [
    ("specvec", 0), ("curvvec", 1), ("specvec_pol", 2),
    ("curvvec_pol", 3), ("rmvec", 4), ("cmvec", 5),
])
def test_mflist_setter_updates_mflist(gauss_im, propname, idx):
    im = gauss_im.copy()
    new = np.linspace(0.1, 0.2, im.xdim * im.ydim)
    setattr(im, propname, new)
    np.testing.assert_array_equal(im._mflist[idx], new)


@pytest.mark.parametrize("propname", ["qvec", "uvec", "vvec"])
def test_polvec_setter_rejects_non_stokes(gauss_im_pol, propname):
    im_circ = gauss_im_pol.switch_polrep("circ")
    with pytest.raises(Exception, match="polrep"):
        setattr(im_circ, propname, np.zeros(im_circ.xdim * im_circ.ydim))


@pytest.mark.parametrize("propname", ["rrvec", "llvec", "rlvec", "lrvec"])
def test_circ_polvec_setter_rejects_non_circ(gauss_im, propname):
    with pytest.raises(Exception, match="polrep"):
        setattr(gauss_im.copy(), propname,
                np.zeros(gauss_im.xdim * gauss_im.ydim))


def test_rrvec_derived_from_stokes_with_v(gauss_im):
    im = gauss_im.copy()
    im.add_v(0.0 * im.imarr())  # V = 0 -> RR = I
    np.testing.assert_allclose(im.rrvec, im.ivec)


def test_rrvec_empty_when_no_v(gauss_im):
    assert gauss_im.rrvec.size == 0


def test_rlvec_derived_from_q_and_u(gauss_im_pol):
    expected = gauss_im_pol.qvec + 1j * gauss_im_pol.uvec
    np.testing.assert_allclose(gauss_im_pol.rlvec, expected)


def test_pvec_matches_qplusiu(gauss_im_pol):
    expected = np.abs(gauss_im_pol.qvec + 1j * gauss_im_pol.uvec)
    np.testing.assert_allclose(gauss_im_pol.pvec, expected)


def test_mvec_matches_p_over_i(gauss_im_pol):
    expected = gauss_im_pol.pvec / gauss_im_pol.ivec
    np.testing.assert_allclose(gauss_im_pol.mvec, expected)


def test_chivec_matches_half_angle(gauss_im_pol):
    expected = 0.5 * np.angle(
        (gauss_im_pol.qvec + 1j * gauss_im_pol.uvec) / gauss_im_pol.ivec
    )
    np.testing.assert_allclose(gauss_im_pol.chivec, expected)


def test_phivec_is_twice_chivec(gauss_im_pol):
    np.testing.assert_allclose(gauss_im_pol.phivec, 2 * gauss_im_pol.chivec)


def test_evpavec_aliases_chivec(gauss_im_pol):
    np.testing.assert_array_equal(gauss_im_pol.evpavec, gauss_im_pol.chivec)


def test_rhovec_total_polfrac(gauss_im_pol):
    expected = np.sqrt(gauss_im_pol.qvec**2 + gauss_im_pol.uvec**2
                       + gauss_im_pol.vvec**2) / gauss_im_pol.ivec
    np.testing.assert_allclose(gauss_im_pol.rhovec, expected)


def test_psivec_circular_poincare(gauss_im_pol):
    expected = np.arctan2(gauss_im_pol.vvec, gauss_im_pol.pvec)
    np.testing.assert_allclose(gauss_im_pol.psivec, expected)


def test_evec_bvec_shape(gauss_im_pol):
    assert gauss_im_pol.evec.shape == gauss_im_pol.ivec.shape
    assert gauss_im_pol.bvec.shape == gauss_im_pol.ivec.shape
    assert np.all(np.isreal(gauss_im_pol.evec))
    assert np.all(np.isreal(gauss_im_pol.bvec))


def test_evec_bvec_zero_for_no_polarization(gauss_im):
    # With Q=U=0 the E and B modes must vanish.
    im = gauss_im.copy()
    im.add_qu(0.0 * im.imarr(), 0.0 * im.imarr())
    np.testing.assert_allclose(im.evec, 0.0, atol=1e-12)
    np.testing.assert_allclose(im.bvec, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Section 3: get_polvec, imarr, sourcevec, fovx/fovy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pol", ["I", "i"])
def test_get_polvec_case_insensitive(gauss_im, pol):
    np.testing.assert_array_equal(gauss_im.get_polvec(pol), gauss_im.ivec)


def test_get_polvec_default_uses_pol_prim(gauss_im):
    np.testing.assert_array_equal(gauss_im.get_polvec(None), gauss_im.imvec)


@pytest.mark.parametrize("pol", ["I", "Q", "U", "V", "P", "M", "chi",
                                  "evpa", "E", "B"])
def test_get_polvec_shapes(gauss_im_pol, pol):
    v = gauss_im_pol.get_polvec(pol)
    assert v.shape == gauss_im_pol.imvec.shape


def test_get_polvec_unknown_raises(gauss_im):
    with pytest.raises(Exception, match="not recognized"):
        gauss_im.get_polvec("banana")


def test_imarr_default_returns_primary(gauss_im):
    arr = gauss_im.imarr()
    assert arr.shape == (gauss_im.ydim, gauss_im.xdim)
    np.testing.assert_array_equal(arr.flatten(), gauss_im.imvec)


def test_imarr_returns_empty_for_missing_pol(gauss_im):
    arr = gauss_im.imarr("Q")  # no Q image present
    assert arr.size == 0


def test_sourcevec_unit_norm(gauss_im):
    sv = gauss_im.sourcevec()
    np.testing.assert_allclose(np.linalg.norm(sv), 1.0)
    assert sv[1] == 0.0


def test_fovx_fovy(gauss_im):
    assert gauss_im.fovx() == pytest.approx(gauss_im.psize * gauss_im.xdim)
    assert gauss_im.fovy() == pytest.approx(gauss_im.psize * gauss_im.ydim)


# ---------------------------------------------------------------------------
# Section 4: add_pol_image, add_qu, add_v, copy_pol_images
# ---------------------------------------------------------------------------


def test_add_pol_image_rejects_pol_prim(gauss_im):
    im = gauss_im.copy()
    with pytest.raises(Exception, match="pol_prim"):
        im.add_pol_image(im.imarr(), "I")


def test_add_pol_image_rejects_bad_shape(gauss_im):
    im = gauss_im.copy()
    with pytest.raises(Exception, match="shapes"):
        im.add_pol_image(np.zeros((3, 3)), "Q")


def test_add_pol_image_rejects_unknown_pol(gauss_im):
    im = gauss_im.copy()
    with pytest.raises(Exception, match="add_pol_image"):
        im.add_pol_image(np.zeros((im.ydim, im.xdim)), "X")


def test_add_qu_rejects_non_stokes(gauss_im_pol):
    im_circ = gauss_im_pol.switch_polrep("circ")
    with pytest.raises(Exception, match="stokes"):
        im_circ.add_qu(np.zeros((im_circ.ydim, im_circ.xdim)),
                       np.zeros((im_circ.ydim, im_circ.xdim)))


def test_add_v_rejects_non_stokes(gauss_im_pol):
    im_circ = gauss_im_pol.switch_polrep("circ")
    with pytest.raises(Exception, match="stokes"):
        im_circ.add_v(np.zeros((im_circ.ydim, im_circ.xdim)))


def test_copy_pol_images_brings_q_u_v(gauss_im, gauss_im_pol):
    empty = gauss_im.copy()
    empty.copy_pol_images(gauss_im_pol)
    np.testing.assert_array_equal(empty.qvec, gauss_im_pol.qvec)
    np.testing.assert_array_equal(empty.uvec, gauss_im_pol.uvec)
    np.testing.assert_array_equal(empty.vvec, gauss_im_pol.vvec)


# ---------------------------------------------------------------------------
# Section 5: switch_polrep
# ---------------------------------------------------------------------------


def test_switch_polrep_invalid_raises(gauss_im):
    with pytest.raises(Exception, match="polrep_out"):
        gauss_im.switch_polrep("lin")


def test_switch_polrep_noop_returns_copy(gauss_im):
    out = gauss_im.switch_polrep("stokes", "I")
    assert out is not gauss_im
    np.testing.assert_array_equal(out.imvec, gauss_im.imvec)


def test_switch_polrep_stokes_to_circ_roundtrip(gauss_im_pol):
    rt = gauss_im_pol.switch_polrep("circ").switch_polrep("stokes")
    for vec in ("ivec", "qvec", "uvec", "vvec"):
        np.testing.assert_allclose(
            getattr(rt, vec), getattr(gauss_im_pol, vec), atol=1e-12,
        )


def test_switch_polrep_stokes_to_circ_formulae(gauss_im_pol):
    out = gauss_im_pol.switch_polrep("circ")
    np.testing.assert_allclose(out.rrvec,
                               gauss_im_pol.ivec + gauss_im_pol.vvec, atol=1e-12)
    np.testing.assert_allclose(out.llvec,
                               gauss_im_pol.ivec - gauss_im_pol.vvec, atol=1e-12)
    np.testing.assert_allclose(out.rlvec,
                               gauss_im_pol.qvec + 1j * gauss_im_pol.uvec,
                               atol=1e-12)
    np.testing.assert_allclose(out.lrvec,
                               gauss_im_pol.qvec - 1j * gauss_im_pol.uvec,
                               atol=1e-12)


def test_switch_polrep_raises_if_output_pol_missing(gauss_im):
    # stokes-only image with pol_prim='I' cannot promote to circ pol_prim='RR'
    # because V is not defined.
    with pytest.raises(Exception, match="not defined"):
        gauss_im.switch_polrep("circ", "RR")


def test_orth_chi_flips_qu_signs(gauss_im_pol):
    im = gauss_im_pol.orth_chi()
    np.testing.assert_array_equal(im.qvec, -gauss_im_pol.qvec)
    np.testing.assert_array_equal(im.uvec, -gauss_im_pol.uvec)


def test_orth_chi_flips_rl_signs(gauss_im_pol):
    im_circ = gauss_im_pol.switch_polrep("circ")
    im = im_circ.orth_chi()
    np.testing.assert_array_equal(im.rlvec, -im_circ.rlvec)
    np.testing.assert_array_equal(im.lrvec, -im_circ.lrvec)


# ---------------------------------------------------------------------------
# Section 6: Summary quantities — total_flux, lin_polfrac, evpa, etc.
# ---------------------------------------------------------------------------


def test_total_flux_stokes(gauss_im):
    assert gauss_im.total_flux() == pytest.approx(1.0)


def test_total_flux_circ_average(gauss_im_pol):
    im_circ = gauss_im_pol.switch_polrep("circ")
    assert im_circ.total_flux() == pytest.approx(gauss_im_pol.total_flux(),
                                                  rel=1e-12)


def test_lin_polfrac_matches_circ(gauss_im_pol):
    f_stokes = gauss_im_pol.lin_polfrac()
    f_circ = gauss_im_pol.switch_polrep("circ").lin_polfrac()
    assert f_stokes == pytest.approx(f_circ, rel=1e-12)


def test_circ_polfrac_matches_circ(gauss_im_pol):
    f_stokes = gauss_im_pol.circ_polfrac()
    f_circ = gauss_im_pol.switch_polrep("circ").circ_polfrac()
    assert f_stokes == pytest.approx(f_circ, rel=1e-12)


def test_evpa_stokes_recovers_input_angle(gauss_im_pol):
    # gauss_im_pol has Q = 0.10*I, U = 0.05*I -> EVPA = 0.5*atan2(0.05, 0.10).
    expected = 0.5 * np.arctan2(0.05, 0.10)
    assert gauss_im_pol.evpa() == pytest.approx(expected, rel=1e-10)


def test_evpa_matches_across_polreps(gauss_im_pol):
    e_stokes = gauss_im_pol.evpa()
    e_circ = gauss_im_pol.switch_polrep("circ").evpa()
    assert e_stokes == pytest.approx(e_circ, rel=1e-12)


def test_mavg_geq_lin_polfrac(gauss_im_pol):
    # |<p>| <= <|p|> for any image (Jensen).
    assert gauss_im_pol.mavg() >= gauss_im_pol.lin_polfrac() - 1e-12


def test_vavg_matches_circ(gauss_im_pol):
    v_stokes = gauss_im_pol.vavg()
    v_circ = gauss_im_pol.switch_polrep("circ").vavg()
    assert v_stokes == pytest.approx(v_circ, rel=1e-12)


def test_betamodes_returns_complex_list(gauss_im_pol):
    out = gauss_im_pol.betamodes(ms=[2], verbose=False)
    assert isinstance(out, list)
    assert isinstance(out[0], complex) or np.iscomplexobj(out[0])


def test_betamodes_output_fluxes_shape(gauss_im_pol):
    out = gauss_im_pol.betamodes(ms=[2], output_fluxes=True, verbose=False)
    coeffs, flux, pflux, area = out
    assert len(coeffs) == 1
    assert flux > 0
    assert area > 0


def test_betamodes_rejects_non_integer_m(gauss_im_pol):
    with pytest.raises(Exception, match="integer"):
        gauss_im_pol.betamodes(ms=[2.5], verbose=False)


# ---------------------------------------------------------------------------
# Section 7: centroid / center
# ---------------------------------------------------------------------------


def test_centroid_zero_for_symmetric_gaussian(gauss_im):
    x0, y0 = gauss_im.centroid()
    # Symmetric Gaussian at the image center -> centroid at (0,0).
    assert abs(x0) < gauss_im.psize
    assert abs(y0) < gauss_im.psize


def test_centroid_raises_on_missing_pol(gauss_im):
    with pytest.raises(Exception, match="No"):
        gauss_im.centroid(pol="Q")


def test_center_shifts_centroid_to_origin():
    # An offset Gaussian should have a non-zero centroid; center() should
    # bring it close to zero.
    im = eh.image.make_empty(64, 200 * eh.RADPERUAS, 17.761, -29.0, rf=230e9)
    im = im.add_gauss(1.0, (40 * eh.RADPERUAS, 40 * eh.RADPERUAS, 0,
                            20 * eh.RADPERUAS, 0))
    centered = im.center()
    x0, y0 = centered.centroid()
    assert abs(x0) < im.psize
    assert abs(y0) < im.psize


# ---------------------------------------------------------------------------
# Section 8: Geometric transforms — pad, regrid, rotate, shift, shift_fft, resample
# ---------------------------------------------------------------------------


def test_pad_preserves_total_flux(gauss_im):
    out = gauss_im.pad(2 * gauss_im.fovx(), 2 * gauss_im.fovy())
    assert out.xdim > gauss_im.xdim
    assert out.total_flux() == pytest.approx(gauss_im.total_flux(), rel=1e-12)
    assert out.psize == pytest.approx(gauss_im.psize)


def test_pad_preserves_polarizations(gauss_im_pol):
    out = gauss_im_pol.pad(2 * gauss_im_pol.fovx(), 2 * gauss_im_pol.fovy())
    assert out.qvec.size == out.xdim * out.ydim
    assert np.sum(out.qvec) == pytest.approx(np.sum(gauss_im_pol.qvec))


def test_regrid_image_changes_dimensions(gauss_im):
    out = gauss_im.regrid_image(gauss_im.fovx(), 16)
    assert out.xdim == 16
    assert out.ydim == 16
    assert out.psize == pytest.approx(gauss_im.fovx() / 16)


def test_regrid_image_roughly_preserves_flux(gauss_im):
    out = gauss_im.regrid_image(gauss_im.fovx(), 16)
    # Coarser grid + interpolation -> within ~10%.
    assert out.total_flux() == pytest.approx(gauss_im.total_flux(), rel=0.1)


def test_rotate_preserves_flux(gauss_im):
    out = gauss_im.rotate(np.pi / 4)
    assert out.total_flux() == pytest.approx(gauss_im.total_flux(), rel=1e-3)


def test_rotate_rejects_non_scalar_pol_prim():
    arr = np.zeros((8, 8))
    arr[4, 4] = 1.0
    im = eh.image.Image(arr, 1e-10, 17.761, -29.0,
                        polrep="circ", pol_prim="RR")
    # Manually flip to a non-rotatable prim:
    im.pol_prim = "RL"
    with pytest.raises(Exception, match="scalar"):
        im.rotate(np.pi / 4)


def test_shift_integer_pixels(gauss_im):
    out = gauss_im.shift([2, 0])
    # A roll-based shift is a permutation, so total flux is exactly preserved.
    assert out.total_flux() == pytest.approx(gauss_im.total_flux())


def test_shift_fft_preserves_flux(gauss_im):
    out = gauss_im.shift_fft([5 * eh.RADPERUAS, 0.0])
    assert out.total_flux() == pytest.approx(gauss_im.total_flux(), rel=1e-10)


def test_resample_square_changes_xdim(gauss_im):
    out = gauss_im.resample_square(16)
    assert out.xdim == 16
    assert out.ydim == 16
    assert out.total_flux() == pytest.approx(gauss_im.total_flux(), rel=1e-12)


def test_resample_square_rejects_rect(make_rect_image):
    rect = make_rect_image(32, 16)
    with pytest.raises(Exception, match="square"):
        rect.resample_square(8)


def test_resample_square_warns_deprecated(gauss_im):
    with pytest.warns(DeprecationWarning, match="regrid_image"):
        gauss_im.resample_square(16)


# ---------------------------------------------------------------------------
# Section 9: Blur, mask, threshold, gradient
# ---------------------------------------------------------------------------


def test_blur_gauss_preserves_flux(gauss_im):
    out = gauss_im.blur_gauss([30 * eh.RADPERUAS, 30 * eh.RADPERUAS, 0])
    assert out.total_flux() == pytest.approx(gauss_im.total_flux(), rel=1e-3)


def test_blur_gauss_zero_frac_is_copy(gauss_im):
    out = gauss_im.blur_gauss([30 * eh.RADPERUAS, 30 * eh.RADPERUAS, 0],
                              frac=0.0)
    np.testing.assert_array_equal(out.imvec, gauss_im.imvec)


def test_blur_circ_preserves_flux(gauss_im):
    out = gauss_im.blur_circ(30 * eh.RADPERUAS)
    assert out.total_flux() == pytest.approx(gauss_im.total_flux(), rel=1e-3)


def test_blur_circ_butter_filter(gauss_im):
    out = gauss_im.blur_circ(30 * eh.RADPERUAS, filttype="butter")
    assert out.imvec.shape == gauss_im.imvec.shape


def test_blur_circ_rejects_bad_filttype(gauss_im):
    with pytest.raises(Exception, match="filttype"):
        gauss_im.blur_circ(30 * eh.RADPERUAS, filttype="square")


def test_blur_mf_rejects_bad_fit_order(gauss_im):
    with pytest.raises(Exception, match="fit_order"):
        gauss_im.blur_mf([230e9, 350e9], 30 * eh.RADPERUAS, fit_order=3)


def test_blur_mf_populates_specvec(gauss_im):
    im_mf = gauss_im.add_const_mf(alpha=2.5)
    out = im_mf.blur_mf([230e9, 350e9], 30 * eh.RADPERUAS, fit_order=1)
    assert out.specvec.size == out.xdim * out.ydim


def test_mask_returns_binary_image(gauss_im):
    out = gauss_im.mask(cutoff=0.5)
    assert set(np.unique(out.imvec)).issubset({0, 1})


def test_mask_with_beam_blurs_first(gauss_im):
    out = gauss_im.mask(cutoff=0.5, beamparams=30 * eh.RADPERUAS, frac=1.0)
    assert set(np.unique(out.imvec)).issubset({0, 1})


def test_apply_mask_dimension_mismatch(gauss_im):
    other = eh.image.make_empty(8, gauss_im.fovx() / 4,
                                17.761, -29.0, rf=230e9)
    with pytest.raises(Exception, match="dimensions"):
        gauss_im.apply_mask(other)


def test_apply_mask_zeros_outside(gauss_im):
    # NOTE: apply_mask mutates self.imvec in place (image.py:1895-1896), so
    # work off a copy to keep the session-scoped fixture untouched.
    im = gauss_im.copy()
    m = im.mask(cutoff=0.5)
    out = im.apply_mask(m, fill_val=0.0)
    # Pixels outside the mask must be zero.
    assert np.all(out.imvec[m.imvec == 0] == 0.0)


def test_threshold_floor_value(gauss_im):
    # threshold internally calls apply_mask, which mutates self.imvec; use a copy.
    out = gauss_im.copy().threshold(cutoff=0.5, fill_val=0.0)
    assert np.min(out.imvec) >= 0.0


def test_grad_shape(gauss_im):
    out = gauss_im.grad()
    assert out.imvec.shape == gauss_im.imvec.shape


def test_grad_zero_for_uniform_image():
    arr = np.ones((16, 16))
    im = eh.image.Image(arr, 1e-10, 17.761, -29.0)
    out = im.grad()
    # Sobel on a uniform field is zero everywhere.
    np.testing.assert_allclose(out.imvec, 0.0, atol=1e-12)


def test_grad_x_differs_from_abs():
    arr = np.zeros((16, 16))
    arr[8, 3] = 1.0
    im = eh.image.Image(arr, 1e-10, 17.761, -29.0)
    gx = im.grad(gradtype="x")
    gabs = im.grad(gradtype="abs")
    assert not np.allclose(gx.imvec, gabs.imvec)


# ---------------------------------------------------------------------------
# Section 10: Source-adding methods
# ---------------------------------------------------------------------------


def test_add_flat_increases_flux(gauss_im):
    out = gauss_im.add_flat(0.5)
    assert out.total_flux() == pytest.approx(gauss_im.total_flux() + 0.5,
                                              rel=1e-12)


def test_add_flat_rejects_missing_pol(gauss_im):
    with pytest.raises(Exception, match="no image"):
        gauss_im.add_flat(0.5, pol="Q")


def test_add_flat_rejects_unknown_pol(gauss_im):
    with pytest.raises(Exception, match="pol must"):
        gauss_im.add_flat(0.5, pol="banana")


def test_add_tophat_increases_flux(gauss_im):
    out = gauss_im.add_tophat(0.7, 50 * eh.RADPERUAS)
    assert out.total_flux() > gauss_im.total_flux()


def test_add_gauss_increases_flux(gauss_im):
    out = gauss_im.add_gauss(0.3, (20 * eh.RADPERUAS, 20 * eh.RADPERUAS, 0))
    assert out.total_flux() == pytest.approx(gauss_im.total_flux() + 0.3,
                                              rel=1e-3)


def test_add_crescent_positive_flux(gauss_im):
    out = gauss_im.add_crescent(0.5, 50 * eh.RADPERUAS, 25 * eh.RADPERUAS,
                                10 * eh.RADPERUAS, 0)
    assert out.total_flux() > gauss_im.total_flux()


def test_add_ring_m1_returns_image(gauss_im):
    out = gauss_im.add_ring_m1(1.0, 0.5, 40 * eh.RADPERUAS, 0,
                               5 * eh.RADPERUAS)
    assert out.imvec.shape == gauss_im.imvec.shape


def test_add_const_pol_sets_linfrac(gauss_im):
    out = gauss_im.add_const_pol(0.3, 0.0)
    # Background pixels with I = 0 give NaN mvec; check only where I > 0.
    mask = out.ivec > 1e-12
    np.testing.assert_allclose(out.mvec[mask], 0.3, atol=1e-10)


def test_add_const_pol_rejects_out_of_range(gauss_im):
    with pytest.raises(Exception, match="magnitude"):
        gauss_im.add_const_pol(1.5, 0.0)


def test_add_random_pol_seed_reproducible(gauss_im):
    a = gauss_im.add_random_pol(0.2, 50 * eh.RADPERUAS, seed=42)
    b = gauss_im.add_random_pol(0.2, 50 * eh.RADPERUAS, seed=42)
    np.testing.assert_array_equal(a.qvec, b.qvec)
    np.testing.assert_array_equal(a.uvec, b.uvec)


def test_add_const_mf_sets_avec_and_bvec(gauss_im):
    out = gauss_im.add_const_mf(alpha=2.5, beta=0.1)
    np.testing.assert_allclose(out.specvec, 2.5)
    np.testing.assert_allclose(out.curvvec, 0.1)


def test_add_const_mf_pol_terms(gauss_im_pol):
    out = gauss_im_pol.add_const_mf(alpha=2.5, alpha_pol=0.5, rm=10.0)
    np.testing.assert_allclose(out.specvec_pol, 0.5)
    np.testing.assert_allclose(out.rmvec, 10.0)
    assert out.curvvec_pol.size == 0  # not requested
    assert out.cmvec.size == 0


# ---------------------------------------------------------------------------
# Section 11: get_image_mf — multifrequency
# ---------------------------------------------------------------------------


def test_get_image_mf_scales_by_power_law(gauss_im):
    # I(nu) = I0 * (nu/nu0)^alpha with alpha=2 and nu=2*nu0 -> I scales 4x.
    im_mf = gauss_im.add_const_mf(alpha=2.0)
    out = im_mf.get_image_mf(2 * gauss_im.rf)
    np.testing.assert_allclose(out.imvec, 4.0 * gauss_im.imvec, rtol=1e-10)
    assert out.rf == pytest.approx(2 * gauss_im.rf)


def test_get_image_mf_identity_at_reference(gauss_im):
    im_mf = gauss_im.add_const_mf(alpha=2.5, beta=0.1)
    out = im_mf.get_image_mf(gauss_im.rf)
    np.testing.assert_allclose(out.imvec, gauss_im.imvec, rtol=1e-10)


def test_get_image_mf_with_polarization(gauss_im_pol):
    im_mf = gauss_im_pol.add_const_mf(alpha=2.0, alpha_pol=0.5, rm=10.0)
    out = im_mf.get_image_mf(0.5 * gauss_im_pol.rf)
    # Intensity should scale; polarization gets a Faraday rotation but stays real.
    assert out.qvec.size == out.imvec.size
    assert np.all(np.isfinite(out.qvec))


# ---------------------------------------------------------------------------
# Section 12: Sampling and observation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ttype", ["direct", "fast", "nfft"])
def test_sample_uv_zero_baseline_is_total_flux(gauss_im, ttype):
    uv = np.array([[0.0, 0.0]])
    data = gauss_im.sample_uv(uv, ttype=ttype, verbose=False)
    np.testing.assert_allclose(np.real(data[0][0]),
                               gauss_im.total_flux(), rtol=1e-3)


def test_sample_uv_rejects_bad_polrep(gauss_im):
    with pytest.raises(Exception, match="polrep_obs"):
        gauss_im.sample_uv(np.array([[0.0, 0.0]]), polrep_obs="lin",
                           verbose=False)


def test_observe_same_nonoise_matches_obs_baselines(gauss_im, obs_direct):
    out = gauss_im.observe_same_nonoise(obs_direct, ttype="direct",
                                         verbose=False)
    assert len(out.data) == len(obs_direct.data)
    np.testing.assert_array_equal(out.data["u"], obs_direct.data["u"])
    np.testing.assert_array_equal(out.data["v"], obs_direct.data["v"])


def test_observe_same_nonoise_zero_baseline_flux(gauss_im, obs_direct):
    # Synthesise visibilities, find a near-zero baseline, check flux.
    out = gauss_im.observe_same_nonoise(obs_direct, ttype="direct",
                                         verbose=False)
    uvdist = np.sqrt(out.data["u"] ** 2 + out.data["v"] ** 2)
    closest = np.argmin(uvdist)
    if uvdist[closest] < 1e9:  # only meaningful for short baselines
        assert np.abs(out.data["vis"][closest]) <= gauss_im.total_flux() + 1e-3


def test_observe_same_nonoise_coord_mismatch_raises(gauss_im, obs_direct):
    im_bad = gauss_im.copy()
    im_bad.ra = obs_direct.ra + 1.0
    with pytest.raises(Exception, match="coordinates"):
        im_bad.observe_same_nonoise(obs_direct, ttype="direct", verbose=False)


def test_observe_same_nonoise_freq_mismatch_raises(gauss_im, obs_direct):
    im_bad = gauss_im.copy()
    im_bad.rf = 2 * obs_direct.rf
    with pytest.raises(Exception, match="frequency"):
        im_bad.observe_same_nonoise(obs_direct, ttype="direct", verbose=False)


def test_observe_same_nonoise_bad_ttype_raises(gauss_im, obs_direct):
    with pytest.raises(Exception, match="ttype"):
        gauss_im.observe_same_nonoise(obs_direct, ttype="banana",
                                       verbose=False)


def test_observe_creates_obsdata(gauss_im, eht_array):
    obs = gauss_im.observe(eht_array, 5, 600, 0, 24, 4e9,
                            ttype="direct", add_th_noise=False, verbose=False)
    assert isinstance(obs, eh.obsdata.Obsdata)
    assert len(obs.data) > 0


def test_observe_with_noise_seed_reproducible(gauss_im, eht_array):
    a = gauss_im.observe(eht_array, 5, 600, 0, 24, 4e9,
                          ttype="direct", add_th_noise=True, seed=42,
                          verbose=False)
    b = gauss_im.observe(eht_array, 5, 600, 0, 24, 4e9,
                          ttype="direct", add_th_noise=True, seed=42,
                          verbose=False)
    np.testing.assert_array_equal(a.data["vis"], b.data["vis"])


# ---------------------------------------------------------------------------
# Section 13: compare_images, align_images, find_shift
# ---------------------------------------------------------------------------


def test_compare_images_self_is_identity(gauss_im):
    err, im1_pad, im2_shift = gauss_im.compare_images(gauss_im)
    nxcorr, nrmse, rssd = err
    # Self comparison: normalized cross-corr = 1, error metrics = 0.
    assert nxcorr == pytest.approx(1.0, abs=1e-6)
    assert nrmse == pytest.approx(0.0, abs=1e-9)
    assert rssd == pytest.approx(0.0, abs=1e-9)


def test_compare_images_polrep_handled_internally(gauss_im_pol):
    im_circ = gauss_im_pol.switch_polrep("circ")
    err, _, _ = gauss_im_pol.compare_images(im_circ)
    assert err[0] == pytest.approx(1.0, abs=1e-6)


def test_find_shift_returns_idx_and_xcorr(gauss_im):
    idx, xcorr, im1, im2 = gauss_im.find_shift(gauss_im)
    assert len(idx) == 2
    assert xcorr.shape == (im1.ydim, im1.xdim)


def test_find_shift_rejects_complex_pol(gauss_im_pol):
    with pytest.raises(Exception, match="complex"):
        gauss_im_pol.find_shift(gauss_im_pol, pol="RL")


def test_align_images_returns_tuple(gauss_im):
    im2 = gauss_im.copy()
    out_list, shifts, base = gauss_im.align_images([im2])
    assert len(out_list) == 1
    assert len(shifts) == 1


# ---------------------------------------------------------------------------
# Section 14: Gaussian fit
# ---------------------------------------------------------------------------


def test_fit_gauss_recovers_fwhm(gauss_im):
    fwhm_maj, fwhm_min, theta = gauss_im.fit_gauss()
    # gauss_im was built with 50 uas FWHM in both axes.
    assert fwhm_maj / eh.RADPERUAS == pytest.approx(50.0, rel=0.01)
    assert fwhm_min / eh.RADPERUAS == pytest.approx(50.0, rel=0.01)


def test_fit_gauss_natural_units(gauss_im):
    fwhm_maj, fwhm_min, pa = gauss_im.fit_gauss(units="natural")
    assert fwhm_maj == pytest.approx(50.0, rel=0.01)
    # Natural units use degrees for the position angle.
    assert 0 <= pa < 180


def test_fit_gauss_empirical_runs(gauss_im):
    out = gauss_im.fit_gauss_empirical()
    assert len(out) == 3
    # The major axis should be >= minor axis after sorting.
    assert out[0] >= out[1]


# ---------------------------------------------------------------------------
# Section 15: Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_txt_roundtrip(gauss_im, tmp_path):
    fname = tmp_path / "im.txt"
    gauss_im.save_txt(str(fname))
    loaded = eh.image.load_txt(str(fname))
    assert loaded.xdim == gauss_im.xdim
    assert loaded.ydim == gauss_im.ydim
    assert loaded.total_flux() == pytest.approx(gauss_im.total_flux(),
                                                 rel=1e-6)


def test_save_load_fits_roundtrip(gauss_im, tmp_path):
    fname = tmp_path / "im.fits"
    gauss_im.save_fits(str(fname))
    loaded = eh.image.load_fits(str(fname))
    assert loaded.xdim == gauss_im.xdim
    assert loaded.ydim == gauss_im.ydim
    assert loaded.total_flux() == pytest.approx(gauss_im.total_flux(),
                                                 rel=1e-6)
    np.testing.assert_allclose(loaded.imvec, gauss_im.imvec, rtol=1e-6,
                                atol=1e-10)


def test_load_image_dispatch_unknown_extension(tmp_path):
    f = tmp_path / "im.unknown"
    f.write_text("hello")
    out = eh.image.load_image(str(f))
    assert out is False


def test_load_image_passthrough_image_object(gauss_im):
    out = eh.image.load_image(gauss_im)
    assert out is gauss_im


# ---------------------------------------------------------------------------
# Section 16: Module-level constructors and helpers
# ---------------------------------------------------------------------------


def test_make_empty_zero_image():
    im = eh.image.make_empty(16, 200 * eh.RADPERUAS, 17.761, -29.0, rf=230e9)
    assert im.xdim == 16
    assert im.ydim == 16
    assert im.total_flux() == 0.0
    assert im.polrep == "stokes"
    assert im.pol_prim == "I"


def test_make_square_uses_obs_metadata(obs_direct):
    sq = eh.image.make_square(obs_direct, 16, 100 * eh.RADPERUAS)
    assert sq.xdim == 16
    assert sq.ra == pytest.approx(obs_direct.ra)
    assert sq.dec == pytest.approx(obs_direct.dec)
    assert sq.rf == pytest.approx(obs_direct.rf)


def test_avg_imlist_two_copies_returns_same_flux(gauss_im):
    avg = eh.image.avg_imlist([gauss_im.copy(), gauss_im.copy()])
    assert avg.total_flux() == pytest.approx(gauss_im.total_flux(), rel=1e-12)


def test_avg_imlist_polrep_mismatch_raises(gauss_im_pol):
    # Needs a fully polarized image because switch_polrep("circ") requires V.
    im_circ = gauss_im_pol.switch_polrep("circ")
    with pytest.raises(Exception, match="polrep"):
        eh.image.avg_imlist([gauss_im_pol.copy(), im_circ])


def test_avg_imlist_source_mismatch_raises(gauss_im):
    other = gauss_im.copy()
    other.source = "other"
    with pytest.raises(Exception, match="source"):
        eh.image.avg_imlist([gauss_im.copy(), other])


def test_avg_imlist_rf_mismatch_raises(gauss_im):
    other = gauss_im.copy()
    other.rf = 2 * gauss_im.rf
    with pytest.raises(Exception, match="rf"):
        eh.image.avg_imlist([gauss_im.copy(), other])


def test_get_specim_recovers_alpha(gauss_im):
    im_mf = gauss_im.add_const_mf(alpha=2.5)
    im_350 = im_mf.get_image_mf(350e9)
    out = eh.image.get_specim([im_mf.copy(), im_350], reffreq=gauss_im.rf,
                               fit_order=1)
    # The recovered spectral index should match the input alpha.
    np.testing.assert_allclose(out.specvec, 2.5, rtol=1e-6)


def test_get_specim_invalid_fit_order(gauss_im):
    im_mf = gauss_im.add_const_mf(alpha=2.5)
    im_350 = im_mf.get_image_mf(350e9)
    with pytest.raises(Exception):
        eh.image.get_specim([im_mf, im_350], reffreq=gauss_im.rf, fit_order=3)
