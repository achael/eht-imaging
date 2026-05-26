"""Tests for ehtim/observing/pol_conventions.py.

Covers:
* the pair-based primitives and their 4-in/4-out wrappers
* round-trips and cross-convention consistency
* sigma propagation (matches existing inline obsdata.py behavior)
* Jones-matrix scaffolding (identity, round-trip with small D-terms)
* MixedPolConventionWarning fires exactly once per session
"""

import warnings

import numpy as np
import pytest

from ehtim.observing import pol_conventions as pc
from ehtim.warnings import MixedPolConventionWarning


@pytest.fixture(autouse=True)
def _reset_warning():
    """Re-arm the once-per-session warning flag for each test."""
    pc._reset_convention_warning()
    yield
    pc._reset_convention_warning()


# ---------------------------------------------------------------------------
# Pair-based primitives
# ---------------------------------------------------------------------------


def test_circ_to_stokes_parallel_returns_pair():
    rr = np.array([1.0 + 0j, 2.0 + 0j])
    ll = np.array([0.8 + 0j, 1.6 + 0j])
    i, v = pc.circ_to_stokes_parallel(rr, ll)
    np.testing.assert_allclose(i, [0.9, 1.8])
    np.testing.assert_allclose(v, [0.1, 0.2])


def test_circ_to_stokes_cross_returns_complex():
    rl = np.array([0.05 + 0.02j])
    lr = np.array([0.05 - 0.02j])
    q, u = pc.circ_to_stokes_cross(rl, lr)
    np.testing.assert_allclose(q, [0.05 + 0j])
    np.testing.assert_allclose(u, [0.02 + 0j])


def test_circ_parallel_roundtrip():
    rr, ll = 1.0 + 0.1j, 0.9 - 0.1j
    i, v = pc.circ_to_stokes_parallel(rr, ll)
    rr2, ll2 = pc.stokes_to_circ_parallel(i, v)
    assert np.isclose(rr2, rr) and np.isclose(ll2, ll)


def test_circ_cross_roundtrip():
    rl, lr = 0.05 + 0.02j, 0.05 - 0.02j
    q, u = pc.circ_to_stokes_cross(rl, lr)
    rl2, lr2 = pc.stokes_to_circ_cross(q, u)
    assert np.isclose(rl2, rl) and np.isclose(lr2, lr)


def test_lin_diag_roundtrip():
    xx, yy = 1.5 + 0j, 0.5 + 0j
    i, q = pc.lin_to_stokes_diag(xx, yy)
    xx2, yy2 = pc.stokes_to_lin_diag(i, q)
    assert np.isclose(xx2, xx) and np.isclose(yy2, yy)


def test_lin_offdiag_roundtrip():
    xy, yx = 0.3 - 0.2j, 0.3 + 0.2j
    u, v = pc.lin_to_stokes_offdiag(xy, yx)
    xy2, yx2 = pc.stokes_to_lin_offdiag(u, v)
    assert np.isclose(xy2, xy) and np.isclose(yx2, yx)


# ---------------------------------------------------------------------------
# 4-in/4-out wrappers compose to the same as paired calls
# ---------------------------------------------------------------------------


def test_circ_to_stokes_wrapper_equals_pair_calls():
    rr, ll, rl, lr = 1.0 + 0.1j, 0.9 + 0j, 0.05 + 0.02j, 0.05 - 0.02j
    i_w, q_w, u_w, v_w = pc.circ_to_stokes(rr, ll, rl, lr)
    i_p, v_p = pc.circ_to_stokes_parallel(rr, ll)
    q_p, u_p = pc.circ_to_stokes_cross(rl, lr)
    assert (i_w, q_w, u_w, v_w) == (i_p, q_p, u_p, v_p)


def test_stokes_to_circ_wrapper_equals_pair_calls():
    i, q, u, v = 1.0 + 0j, 0.05 + 0j, -0.02 + 0j, 0.1 + 0j
    rr_w, ll_w, rl_w, lr_w = pc.stokes_to_circ(i, q, u, v)
    rr_p, ll_p = pc.stokes_to_circ_parallel(i, v)
    rl_p, lr_p = pc.stokes_to_circ_cross(q, u)
    assert (rr_w, ll_w, rl_w, lr_w) == (rr_p, ll_p, rl_p, lr_p)


def test_lin_to_stokes_wrapper_equals_pair_calls():
    xx, yy, xy, yx = 1.5 + 0j, 0.5 + 0j, 0.3 - 0.2j, 0.3 + 0.2j
    i_w, q_w, u_w, v_w = pc.lin_to_stokes(xx, yy, xy, yx)
    i_p, q_p = pc.lin_to_stokes_diag(xx, yy)
    u_p, v_p = pc.lin_to_stokes_offdiag(xy, yx)
    assert (i_w, q_w, u_w, v_w) == (i_p, q_p, u_p, v_p)


def test_stokes_to_lin_wrapper_equals_pair_calls():
    i, q, u, v = 1.0 + 0j, 0.5 + 0j, 0.3 + 0j, 0.2 + 0j
    xx_w, yy_w, xy_w, yx_w = pc.stokes_to_lin(i, q, u, v)
    xx_p, yy_p = pc.stokes_to_lin_diag(i, q)
    xy_p, yx_p = pc.stokes_to_lin_offdiag(u, v)
    assert (xx_w, yy_w, xy_w, yx_w) == (xx_p, yy_p, xy_p, yx_p)


# ---------------------------------------------------------------------------
# Cross-convention: stokes -> circ vs stokes -> lin produce the same physics
# ---------------------------------------------------------------------------


def test_circ_lin_consistency_via_stokes():
    """Known Stokes -> circ vs Stokes -> lin -> circ must agree."""
    i, q, u, v = 1.0 + 0j, 0.5 + 0j, 0.3 + 0j, 0.2 + 0j
    rr, ll, rl, lr = pc.stokes_to_circ(i, q, u, v)
    xx, yy, xy, yx = pc.stokes_to_lin(i, q, u, v)
    rr2, ll2, rl2, lr2 = pc.lin_to_circ(xx, yy, xy, yx)
    assert np.isclose(rr, rr2) and np.isclose(ll, ll2)
    assert np.isclose(rl, rl2) and np.isclose(lr, lr2)


def test_circ_to_lin_via_composition():
    """circ_to_lin = stokes_to_lin(circ_to_stokes(...))."""
    rr, ll, rl, lr = 1.2 + 0.1j, 0.8 - 0.1j, 0.05 + 0.02j, 0.05 - 0.02j
    xx, yy, xy, yx = pc.circ_to_lin(rr, ll, rl, lr)
    xx_e, yy_e, xy_e, yx_e = pc.stokes_to_lin(*pc.circ_to_stokes(rr, ll, rl, lr))
    assert np.isclose(xx, xx_e) and np.isclose(yy, yy_e)
    assert np.isclose(xy, xy_e) and np.isclose(yx, yx_e)


def test_basis_matrix_consistent_with_circ_lin_stokes_formulas():
    """BASIS_LIN_TO_CIRC must give the same Stokes whether you go
    (X,Y) -> Stokes via §5, or (X,Y) -> (R,L) -> Stokes via §4.

    Regression test for the convention bug where §2 declared one basis
    but §4 / §5 implemented the opposite (pre-fix, lin_to_stokes_offdiag
    had V with the wrong sign relative to BASIS_LIN_TO_CIRC).
    """
    # Pure +U linear wave (45 deg, in phase) — the discriminating case.
    EX, EY = 1.0 + 0j, 1.0 + 0j
    XX, YY = EX * np.conj(EX), EY * np.conj(EY)
    XY, YX = EX * np.conj(EY), EY * np.conj(EX)
    I_lin, Q_lin, U_lin, V_lin = pc.lin_to_stokes(XX, YY, XY, YX)

    ER, EL = pc.BASIS_LIN_TO_CIRC @ np.array([EX, EY])
    RR, LL = ER * np.conj(ER), EL * np.conj(EL)
    RL, LR = ER * np.conj(EL), EL * np.conj(ER)
    I_cir, Q_cir, U_cir, V_cir = pc.circ_to_stokes(RR, LL, RL, LR)

    assert np.isclose(I_lin, I_cir)
    assert np.isclose(Q_lin, Q_cir)
    assert np.isclose(U_lin, U_cir)
    assert np.isclose(V_lin, V_cir)


# ---------------------------------------------------------------------------
# IAU/HBS convention check via a known Stokes vector
# ---------------------------------------------------------------------------


def test_known_stokes_to_lin_iau_convention():
    """For I=1, Q=0, U=0, V=0 we expect XX=YY=1, XY=YX=0."""
    xx, yy, xy, yx = pc.stokes_to_lin(1.0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j)
    assert np.isclose(xx, 1.0) and np.isclose(yy, 1.0)
    assert np.isclose(xy, 0.0) and np.isclose(yx, 0.0)


def test_known_stokes_q_only_to_lin():
    """For I=0, Q=1, U=0, V=0 we expect XX=1, YY=-1, XY=YX=0."""
    xx, yy, xy, yx = pc.stokes_to_lin(0 + 0j, 1.0 + 0j, 0 + 0j, 0 + 0j)
    assert np.isclose(xx, 1.0) and np.isclose(yy, -1.0)
    assert np.isclose(xy, 0.0) and np.isclose(yx, 0.0)


def test_known_stokes_v_only_to_lin_iau_sign():
    """For pure V=1, IAU/HBS (engineering) gives XY=+i, YX=-i; XX=YY=0."""
    xx, yy, xy, yx = pc.stokes_to_lin(0 + 0j, 0 + 0j, 0 + 0j, 1.0 + 0j)
    assert np.isclose(xx, 0.0) and np.isclose(yy, 0.0)
    assert np.isclose(xy, +1.0j) and np.isclose(yx, -1.0j)


# ---------------------------------------------------------------------------
# Sigma propagation matches the existing inline obsdata.py formulas
# ---------------------------------------------------------------------------


def test_circ_to_stokes_sigma():
    s_iv_e = 0.5 * np.sqrt(0.1**2 + 0.1**2)
    s_qu_e = 0.5 * np.sqrt(0.2**2 + 0.2**2)
    si, sq, su, sv = pc.circ_to_stokes_sigma(0.1, 0.1, 0.2, 0.2)
    assert np.isclose(si, s_iv_e) and np.isclose(sv, s_iv_e)
    assert np.isclose(sq, s_qu_e) and np.isclose(su, s_qu_e)


def test_stokes_to_circ_sigma():
    rr_e = np.sqrt(0.1**2 + 0.05**2)
    rl_e = np.sqrt(0.2**2 + 0.2**2)
    sr, sl, srl, slr = pc.stokes_to_circ_sigma(0.1, 0.2, 0.2, 0.05)
    assert np.isclose(sr, rr_e) and np.isclose(sl, rr_e)
    assert np.isclose(srl, rl_e) and np.isclose(slr, rl_e)


def test_lin_to_stokes_sigma_mirrors_circ_pattern():
    """Same structure as circ_to_stokes_sigma but on (XX,YY) and (XY,YX)."""
    si, sq, su, sv = pc.lin_to_stokes_sigma(0.1, 0.1, 0.2, 0.2)
    assert np.isclose(si, 0.5 * np.sqrt(0.02))
    assert np.isclose(sq, 0.5 * np.sqrt(0.02))
    assert np.isclose(su, 0.5 * np.sqrt(0.08))
    assert np.isclose(sv, 0.5 * np.sqrt(0.08))


# ---------------------------------------------------------------------------
# Jones-matrix scaffolding
# ---------------------------------------------------------------------------


def test_jones_identity_with_unit_gains():
    J = pc.jones_matrix(1.0 + 0j, 1.0 + 0j, 0, 0)
    np.testing.assert_allclose(J, np.eye(2))


def test_jones_d_terms_off_diagonal():
    """J = G(I + D); with G=I and small D, J has D's off-diagonals."""
    J = pc.jones_matrix(1.0 + 0j, 1.0 + 0j, d_p1=0.05 + 0.01j, d_p2=-0.03 + 0.02j)
    np.testing.assert_allclose(J, np.array([[1.0, 0.05 + 0.01j],
                                            [-0.03 + 0.02j, 1.0]]))


def test_apply_inverse_jones_identity_preserves_vis():
    V = np.array([[1.0 + 0j, 0.05 + 0j], [0.05 + 0j, 0.8 + 0j]])
    J = np.eye(2, dtype=complex)
    V_corr = pc.apply_inverse_jones_to_coherency(V, J, J)
    np.testing.assert_allclose(V_corr, V)


def test_apply_inverse_jones_roundtrip_small_dterms():
    """V_obs = J1 V_true J2^dag; apply_inverse recovers V_true."""
    V_true = np.array([[1.0 + 0j, 0.05 + 0.02j],
                       [0.05 - 0.02j, 0.8 + 0j]])
    J1 = pc.jones_matrix(1.05 + 0.01j, 0.97 - 0.02j, 0.03, -0.02)
    J2 = pc.jones_matrix(0.98 - 0.01j, 1.02 + 0.02j, -0.01, 0.04)
    V_obs = J1 @ V_true @ J2.conj().T
    V_recovered = pc.apply_inverse_jones_to_coherency(V_obs, J1, J2)
    np.testing.assert_allclose(V_recovered, V_true, atol=1e-12)


# ---------------------------------------------------------------------------
# MixedPolConventionWarning fires exactly once per session
# ---------------------------------------------------------------------------


def test_warning_fires_on_first_transform():
    pc._reset_convention_warning()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=MixedPolConventionWarning)
        pc.circ_to_stokes_parallel(1.0 + 0j, 0.9 + 0j)
    assert any(issubclass(w.category, MixedPolConventionWarning) for w in caught)


def test_warning_only_fires_once_per_session():
    pc._reset_convention_warning()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=MixedPolConventionWarning)
        pc.circ_to_stokes_parallel(1.0 + 0j, 0.9 + 0j)
        pc.lin_to_stokes_diag(1.5 + 0j, 0.5 + 0j)  # second non-identity transform
        pc.stokes_to_lin_offdiag(0.3 + 0j, 0.2 + 0j)  # third
    fired = [w for w in caught if issubclass(w.category, MixedPolConventionWarning)]
    assert len(fired) == 1


def test_warning_fires_from_jones_application():
    pc._reset_convention_warning()
    V = np.eye(2, dtype=complex)
    J = np.eye(2, dtype=complex)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=MixedPolConventionWarning)
        pc.apply_inverse_jones_to_coherency(V, J, J)
    assert any(issubclass(w.category, MixedPolConventionWarning) for w in caught)
