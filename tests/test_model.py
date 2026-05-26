"""Tests for ehtim.model.Model.

Focused on Phase 3 mixed-pol additions:
- Model.switch_polrep accepts polrep='lin'
- sample_1model_uv produces lin-basis visibilities consistent with
  pol_conventions.stokes_to_lin and pol_conventions.circ_to_lin
- get_const_polfac dispatches XX/YY/XY/YX

Two pre-Phase-3 Model issues constrain what this file can test directly;
both are tracked as future-cleanup TODOs separate from this PR:

1. Model.switch_polrep is intentionally a no-op — it validates the
   request and returns `self.copy()`, leaving the polrep unchanged.
   Worse, Model.__init__ hardcodes `self.polrep = 'stokes'` regardless
   of the `polrep=` constructor argument (model.py:1548). The polrep
   slot on a Model is therefore always 'stokes' in practice. We test
   only that switch_polrep('lin') doesn't raise, not that the output
   actually carries the new polrep.

2. Model.sample_uv and Model.observe_same_nonoise both flow through
   sample_model_uv (model.py:1467, 1470), which uses
   `np.sum(generator)` — removed in numpy 2.0. Affected end-to-end
   paths are marked xfail and tested at the module-level entry point
   sample_1model_uv instead.
"""

import numpy as np
import pytest

import ehtim as eh
from ehtim.model import get_const_polfac, sample_1model_uv
from ehtim.observing import pol_conventions as pc

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def polarized_gauss_params():
    return ("circ_gauss",
            {"F0": 1.0, "FWHM": 20 * eh.RADPERUAS, "x0": 0.0, "y0": 0.0,
             "pol_frac": 0.1, "pol_evpa": 0.3, "cpol_frac": 0.05})


@pytest.fixture
def uv_grid():
    u = np.array([1e9, 2e9, 3e9, 4e9])
    v = np.array([0.0, 5e8, 1e9, -5e8])
    return u, v


@pytest.fixture(autouse=True)
def _reset_warning():
    pc._reset_convention_warning()
    yield


# ---------------------------------------------------------------------------
# Model.switch_polrep
# ---------------------------------------------------------------------------


def test_switch_polrep_lin_accepted():
    """Phase 3 widening: 'lin' is a valid polrep_out.

    Note: Model.switch_polrep is intentionally a no-op (returns self.copy()),
    so the polrep on the output isn't actually changed — the test only
    asserts that validation accepts 'lin' and the call doesn't raise.
    """
    mod = eh.model.Model(ra=0, dec=0)
    out = mod.switch_polrep("lin")  # must not raise
    assert isinstance(out, eh.model.Model)


def test_switch_polrep_bogus_raises():
    mod = eh.model.Model(ra=0, dec=0)
    with pytest.raises(Exception, match="polrep_out"):
        mod.switch_polrep("bogus")


# ---------------------------------------------------------------------------
# get_const_polfac dispatches XX/YY/XY/YX
# ---------------------------------------------------------------------------


def test_get_const_polfac_xx_from_stokes(polarized_gauss_params):
    mt, params = polarized_gauss_params
    i_factor = get_const_polfac(mt, params, "I")
    q_factor = get_const_polfac(mt, params, "Q")
    xx_factor = get_const_polfac(mt, params, "XX")
    assert np.isclose(xx_factor, i_factor + q_factor)


def test_get_const_polfac_yy_from_stokes(polarized_gauss_params):
    mt, params = polarized_gauss_params
    i_factor = get_const_polfac(mt, params, "I")
    q_factor = get_const_polfac(mt, params, "Q")
    yy_factor = get_const_polfac(mt, params, "YY")
    assert np.isclose(yy_factor, i_factor - q_factor)


def test_get_const_polfac_xy_from_stokes(polarized_gauss_params):
    mt, params = polarized_gauss_params
    u_factor = get_const_polfac(mt, params, "U")
    v_factor = get_const_polfac(mt, params, "V")
    xy_factor = get_const_polfac(mt, params, "XY")
    assert np.isclose(xy_factor, u_factor + 1j * v_factor)


def test_get_const_polfac_yx_from_stokes(polarized_gauss_params):
    mt, params = polarized_gauss_params
    u_factor = get_const_polfac(mt, params, "U")
    v_factor = get_const_polfac(mt, params, "V")
    yx_factor = get_const_polfac(mt, params, "YX")
    assert np.isclose(yx_factor, u_factor - 1j * v_factor)


# ---------------------------------------------------------------------------
# sample_1model_uv XX/YY/XY/YX clauses
# ---------------------------------------------------------------------------


def test_sample_1model_xx_matches_stokes_to_lin(polarized_gauss_params, uv_grid):
    mt, params = polarized_gauss_params
    u, v = uv_grid
    I = sample_1model_uv(u, v, mt, params, pol="I")
    Q = sample_1model_uv(u, v, mt, params, pol="Q")
    XX = sample_1model_uv(u, v, mt, params, pol="XX")
    XX_e, _ = pc.stokes_to_lin_diag(I, Q)
    np.testing.assert_allclose(XX, XX_e, atol=1e-12)


def test_sample_1model_yy_matches_stokes_to_lin(polarized_gauss_params, uv_grid):
    mt, params = polarized_gauss_params
    u, v = uv_grid
    I = sample_1model_uv(u, v, mt, params, pol="I")
    Q = sample_1model_uv(u, v, mt, params, pol="Q")
    YY = sample_1model_uv(u, v, mt, params, pol="YY")
    _, YY_e = pc.stokes_to_lin_diag(I, Q)
    np.testing.assert_allclose(YY, YY_e, atol=1e-12)


def test_sample_1model_xy_matches_stokes_to_lin(polarized_gauss_params, uv_grid):
    mt, params = polarized_gauss_params
    u, v = uv_grid
    U = sample_1model_uv(u, v, mt, params, pol="U")
    V = sample_1model_uv(u, v, mt, params, pol="V")
    XY = sample_1model_uv(u, v, mt, params, pol="XY")
    XY_e, _ = pc.stokes_to_lin_offdiag(U, V)
    np.testing.assert_allclose(XY, XY_e, atol=1e-12)


def test_sample_1model_yx_matches_stokes_to_lin(polarized_gauss_params, uv_grid):
    mt, params = polarized_gauss_params
    u, v = uv_grid
    U = sample_1model_uv(u, v, mt, params, pol="U")
    V = sample_1model_uv(u, v, mt, params, pol="V")
    YX = sample_1model_uv(u, v, mt, params, pol="YX")
    _, YX_e = pc.stokes_to_lin_offdiag(U, V)
    np.testing.assert_allclose(YX, YX_e, atol=1e-12)


def test_sample_1model_lin_matches_circ_via_pol_conventions(
    polarized_gauss_params, uv_grid,
):
    """Cross-check: lin sampling equals circ sampling routed through pol_conventions."""
    mt, params = polarized_gauss_params
    u, v = uv_grid
    RR = sample_1model_uv(u, v, mt, params, pol="RR")
    LL = sample_1model_uv(u, v, mt, params, pol="LL")
    RL = sample_1model_uv(u, v, mt, params, pol="RL")
    LR = sample_1model_uv(u, v, mt, params, pol="LR")
    XX_e, YY_e, XY_e, YX_e = pc.circ_to_lin(RR, LL, RL, LR)
    XX = sample_1model_uv(u, v, mt, params, pol="XX")
    YY = sample_1model_uv(u, v, mt, params, pol="YY")
    XY = sample_1model_uv(u, v, mt, params, pol="XY")
    YX = sample_1model_uv(u, v, mt, params, pol="YX")
    np.testing.assert_allclose(XX, XX_e, atol=1e-12)
    np.testing.assert_allclose(YY, YY_e, atol=1e-12)
    np.testing.assert_allclose(XY, XY_e, atol=1e-12)
    np.testing.assert_allclose(YX, YX_e, atol=1e-12)


# ---------------------------------------------------------------------------
# Model.observe_same_nonoise lin path
# ---------------------------------------------------------------------------
# Pre-existing model.py:1470 uses np.sum(generator), removed in numpy 2.0.
# Once that's fixed, the xfail below should flip to passing on a lin obs.


@pytest.mark.xfail(reason="model.py:1470 np.sum(generator) numpy 2.0 incompat", strict=True)
def test_observe_same_nonoise_produces_lin_obsdata(polarized_gauss_params, uv_grid):
    mt, params = polarized_gauss_params
    mod = eh.model.Model(ra=0, dec=0)
    mod = mod.add_circ_gauss(F0=params["F0"], FWHM=params["FWHM"],
                             pol_frac=params["pol_frac"],
                             pol_evpa=params["pol_evpa"],
                             cpol_frac=params["cpol_frac"])
    arr = eh.array.load_txt("arrays/EHT2025.txt")
    obs = arr.obsdata(ra=0, dec=0, rf=230e9, bw=4e9,
                      tint=10, tadv=600, tstart=0, tstop=4, polrep="lin")
    out = mod.observe_same_nonoise(obs)
    assert out.polrep == "lin"
    assert "xxvis" in out.data.dtype.fields
