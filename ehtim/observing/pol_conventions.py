# pol_conventions.py
# Single home for polarization-basis transforms and Jones-matrix math.
#
#    Copyright (C) 2026 Andrew Chael
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Polarization basis transforms and Jones-matrix utilities.

Convention (IAU / Hamaker-Bregman-Sault / CASA / AIPS, engineering
time dependence ``exp(+i omega t)`` as in TMS Ch. 4):

    R = (X + iY) / sqrt(2)
    L = (X - iY) / sqrt(2)

with the circular Stokes formulas:

    I = (RR + LL) / 2,    V = (RR - LL) / 2
    Q = (RL + LR) / 2,    U = i (LR - RL) / 2

and the linear-feed Stokes formulas:

    I = (XX + YY) / 2,    Q = (XX - YY) / 2
    U = (XY + YX) / 2,    V = -i (XY - YX) / 2

The sign of V flips under the opposite basis choice
(R = (X - iY) / sqrt(2), physics convention). Full derivation and
worked examples live in ``docs/polarization_conventions.md``.

A ``MixedPolConventionWarning`` fires once per session on the first
non-identity transform call.
"""

import functools
import warnings

import numpy as np

from ehtim.warnings import MixedPolConventionWarning

# ---------------------------------------------------------------------------
# Basis matrices
# ---------------------------------------------------------------------------

# Maps a feed-basis vector (X, Y) -> (R, L) under the IAU/HBS convention
# with the engineering / IEEE time-dependence convention exp(+i omega t)
# used throughout radio interferometry (cf. TMS Ch. 4). Under this
# convention R = (X + iY)/sqrt(2), L = (X - iY)/sqrt(2).
# Rows indexed by (R, L); columns by (X, Y).
# Reference only: the pair transforms below currently use the explicit
# arithmetic forms.
BASIS_LIN_TO_CIRC = np.array([[1.0, +1.0j],
                              [1.0, -1.0j]]) / np.sqrt(2.0)

# Unitary inverse: maps (R, L) -> (X, Y).
BASIS_CIRC_TO_LIN = BASIS_LIN_TO_CIRC.conj().T


# ---------------------------------------------------------------------------
# Session-level convention warning
# ---------------------------------------------------------------------------

_convention_warning_emitted = False


def _maybe_warn_convention():
    """Emit MixedPolConventionWarning at most once per session."""
    global _convention_warning_emitted
    if _convention_warning_emitted:
        return
    _convention_warning_emitted = True
    warnings.warn(
        "Polarization basis transform applied under the IAU/Hamaker-"
        "Bregman-Sault convention (R = (X + iY)/sqrt(2)). This "
        "transformation assumes ideal feeds (D = 0 or already "
        "calibrated). See docs/polarization_conventions.md.",
        category=MixedPolConventionWarning,
        stacklevel=3,
    )


def _reset_convention_warning():
    """Test helper: re-arm the once-per-session warning flag."""
    global _convention_warning_emitted
    _convention_warning_emitted = False


# ---------------------------------------------------------------------------
# Pair-based polarization transforms (the primitives)
# ---------------------------------------------------------------------------
#
# These transforms operate on the polarization basis itself, so they apply
# uniformly to any quantity carried in that basis: visibilities, image
# pixels, model parameters, etc. The basis transforms factor into pairs:
#   - circ:  parallel-hand (RR, LL) <-> (I, V);  cross-hand (RL, LR) <-> (Q, U)
#   - lin:   diagonal     (XX, YY) <-> (I, Q);   off-diagonal (XY, YX) <-> (U, V)
#
# Callers that need a single Stokes component (image/movie property
# getters) call the relevant pair helper and discard one output. The
# 4-in/4-out functions further down compose two pair calls and are
# convenience wrappers for callers (like Obsdata.switch_polrep) that
# want the full four-tuple.

def circ_to_stokes_parallel(rr, ll):
    """(I, V) from (RR, LL) parallel-hand pair.

    Returns ``(0.5*(RR+LL), 0.5*(RR-LL))``.
    """
    _maybe_warn_convention()
    return 0.5 * (rr + ll), 0.5 * (rr - ll)


def circ_to_stokes_cross(rl, lr):
    """(Q, U) from (RL, LR) cross-hand pair.

    Returns ``(0.5*(LR+RL), 0.5j*(LR-RL))``. The outputs are complex;
    image-domain callers should apply ``np.real()`` since the sky-domain
    Q, U are real-valued.
    """
    _maybe_warn_convention()
    return 0.5 * (lr + rl), 0.5j * (lr - rl)


def stokes_to_circ_parallel(i, v):
    """(RR, LL) from (I, V). Inverse of circ_to_stokes_parallel."""
    _maybe_warn_convention()
    return i + v, i - v


def stokes_to_circ_cross(q, u):
    """(RL, LR) from (Q, U). Inverse of circ_to_stokes_cross."""
    _maybe_warn_convention()
    return q + 1.0j * u, q - 1.0j * u


def lin_to_stokes_diag(xx, yy):
    """(I, Q) from (XX, YY) diagonal pair.

    Returns ``(0.5*(XX+YY), 0.5*(XX-YY))``.
    """
    _maybe_warn_convention()
    return 0.5 * (xx + yy), 0.5 * (xx - yy)


def lin_to_stokes_offdiag(xy, yx):
    """(U, V) from (XY, YX) off-diagonal pair.

    Returns ``(0.5*(XY+YX), 0.5j*(YX-XY))``. The outputs are complex;
    image-domain callers should apply ``np.real()`` since the sky-domain
    U, V are real-valued.
    """
    _maybe_warn_convention()
    return 0.5 * (xy + yx), 0.5j * (yx - xy)


def stokes_to_lin_diag(i, q):
    """(XX, YY) from (I, Q). Inverse of lin_to_stokes_diag."""
    _maybe_warn_convention()
    return i + q, i - q


def stokes_to_lin_offdiag(u, v):
    """(XY, YX) from (U, V). Inverse of lin_to_stokes_offdiag."""
    _maybe_warn_convention()
    return u + 1.0j * v, u - 1.0j * v


# ---------------------------------------------------------------------------
# 4-in/4-out polarization transforms (thin wrappers around the pair helpers)
# ---------------------------------------------------------------------------

def circ_to_stokes(rr, ll, rl, lr):
    """Convert all four circular-basis components to Stokes.

    Returns
    -------
    (i, q, u, v) : tuple of ndarrays
    """
    i, v = circ_to_stokes_parallel(rr, ll)
    q, u = circ_to_stokes_cross(rl, lr)
    return i, q, u, v


def stokes_to_circ(i, q, u, v):
    """Convert all four Stokes components to circular basis.

    Returns
    -------
    (rr, ll, rl, lr) : tuple of ndarrays
    """
    rr, ll = stokes_to_circ_parallel(i, v)
    rl, lr = stokes_to_circ_cross(q, u)
    return rr, ll, rl, lr


def lin_to_stokes(xx, yy, xy, yx):
    """Convert all four linear-basis components to Stokes (IAU/HBS).

    Returns
    -------
    (i, q, u, v) : tuple of ndarrays
    """
    i, q = lin_to_stokes_diag(xx, yy)
    u, v = lin_to_stokes_offdiag(xy, yx)
    return i, q, u, v


def stokes_to_lin(i, q, u, v):
    """Convert all four Stokes components to linear basis (IAU/HBS).

    Returns
    -------
    (xx, yy, xy, yx) : tuple of ndarrays
    """
    xx, yy = stokes_to_lin_diag(i, q)
    xy, yx = stokes_to_lin_offdiag(u, v)
    return xx, yy, xy, yx


def lin_to_circ(xx, yy, xy, yx):
    """Convert linear-basis components to circular basis via Stokes."""
    return stokes_to_circ(*lin_to_stokes(xx, yy, xy, yx))


def circ_to_lin(rr, ll, rl, lr):
    """Convert circular-basis components to linear basis via Stokes."""
    return stokes_to_lin(*circ_to_stokes(rr, ll, rl, lr))


# ---------------------------------------------------------------------------
# Sigma propagation (quadrature, not linear)
# ---------------------------------------------------------------------------
#
# TODO: these per-component sigma transforms are NOT invertible. The
# basis transforms are linear combinations of the input visibilities, so
# the output components are correlated even when the inputs are
# independent — but we only return marginal variances per output
# component, dropping the off-diagonal covariance terms. A
# round-trip (e.g. stokes_to_circ_sigma then circ_to_stokes_sigma)
# generally does not recover the input sigmas.
#
# Proper treatment needs a full 4x4 covariance matrix propagation:
#     Cov_out = M @ Cov_in @ M^dagger
# where M is the 4x4 transform matrix on the visibility vector. Migrate
# callers to that when a downstream consumer cares about the cross
# terms (uncertainty quantification, weighted closures, Bayesian
# inference).
#
# Until then these helpers replicate the existing inline obsdata.py
# behavior bit-for-bit, so the lossy independence assumption is
# preserved across the migration.

def circ_to_stokes_sigma(rrsigma, llsigma, rlsigma, lrsigma):
    """Propagate circular-feed sigmas to Stokes sigmas.

    sigma_I = sigma_V = 0.5 * sqrt(rrsigma^2 + llsigma^2)
    sigma_Q = sigma_U = 0.5 * sqrt(rlsigma^2 + lrsigma^2)
    """
    iv = 0.5 * np.sqrt(rrsigma**2 + llsigma**2)
    qu = 0.5 * np.sqrt(rlsigma**2 + lrsigma**2)
    return iv, qu, qu, iv


def stokes_to_circ_sigma(sigma, qsigma, usigma, vsigma):
    """Propagate Stokes sigmas to circular-feed sigmas.

    sigma_RR = sigma_LL = sqrt(sigma_I^2 + sigma_V^2)
    sigma_RL = sigma_LR = sqrt(sigma_Q^2 + sigma_U^2)
    """
    rr_ll = np.sqrt(sigma**2 + vsigma**2)
    rl_lr = np.sqrt(qsigma**2 + usigma**2)
    return rr_ll, rr_ll, rl_lr, rl_lr


def lin_to_stokes_sigma(xxsigma, yysigma, xysigma, yxsigma):
    """Propagate linear-feed sigmas to Stokes sigmas.

    sigma_I = sigma_Q = 0.5 * sqrt(xxsigma^2 + yysigma^2)
    sigma_U = sigma_V = 0.5 * sqrt(xysigma^2 + yxsigma^2)
    """
    iq = 0.5 * np.sqrt(xxsigma**2 + yysigma**2)
    uv = 0.5 * np.sqrt(xysigma**2 + yxsigma**2)
    return iq, iq, uv, uv


def stokes_to_lin_sigma(sigma, qsigma, usigma, vsigma):
    """Propagate Stokes sigmas to linear-feed sigmas."""
    xx_yy = np.sqrt(sigma**2 + qsigma**2)
    xy_yx = np.sqrt(usigma**2 + vsigma**2)
    return xx_yy, xx_yy, xy_yx, xy_yx


# ---------------------------------------------------------------------------
# General coherency <-> Stokes for arbitrary feed pairings (mixed-pol)
# ---------------------------------------------------------------------------
#
# Under ideal feeds (D = 0, gains calibrated -- the same assumption every
# transform here makes), a baseline between stations with feed matrices F1, F2
# measures the full coherency
#
#     V = F1 @ B @ F2^dagger,   B = [[XX, XY], [YX, YY]]   (sky brightness, XY basis)
#
# so B = F1^{-1} V (F2^dagger)^{-1} recovers Stokes from ANY feed pairing,
# including heterogeneous ones (e.g. circular x linear) and hybrid feeds whose
# F is invertible but not unitary. The feed matrices are built from
# BASIS_LIN_TO_CIRC (engineering convention R = (X + iY)/sqrt(2)), so
# these reduce to circ_to_stokes / lin_to_stokes on homogeneous pairings -- a
# property asserted by the cross-validation tests.
#
# Field rotation / parallactic angle is assumed already corrected (as for all
# transforms in this module). Slot order is the generic DTPOL_MIXED order
# (p1p1, p2p2, p1p2, p2p1); Stokes order is (I, Q, U, V).

# Each feed letter as its (X, Y) projection vector (rows of the feed matrix).
_FEED_VEC = {
    'x': np.array([1.0, 0.0], dtype=complex),
    'y': np.array([0.0, 1.0], dtype=complex),
    'r': BASIS_LIN_TO_CIRC[0].copy(),   # R = (X + iY)/sqrt(2)
    'l': BASIS_LIN_TO_CIRC[1].copy(),   # L = (X - iY)/sqrt(2)
}


@functools.cache
def feed_matrix(feed_type):
    """2x2 matrix F mapping an (X, Y) field to a station's two feeds.

    Rows are the station's feeds (in feed order), columns are (X, Y). Built
    from BASIS_LIN_TO_CIRC, so feed_matrix('rl') == BASIS_LIN_TO_CIRC and
    feed_matrix('xy') == identity. Always invertible; unitary for orthogonal
    feed pairs (rl/lr/xy/yx) but NOT for hybrid feeds (e.g. 'rx': R and X are
    non-orthogonal) -- which is fine, coherency recovery uses inv(), not the
    adjoint. The returned array is cached and read-only.
    """
    if (len(feed_type) != 2 or feed_type[0] == feed_type[1]
            or any(c not in _FEED_VEC for c in feed_type)):
        raise ValueError(f"feed_matrix: degenerate or unsupported feed_type {feed_type!r}")
    F = np.array([_FEED_VEC[feed_type[0]], _FEED_VEC[feed_type[1]]])
    F.flags.writeable = False
    return F


@functools.cache
def _coherency_matrix(t1_feed, t2_feed):
    """Cached 4x4 complex operator M with
       (I, Q, U, V) = M @ (p1p1, p2p2, p1p2, p2p1)
       for a baseline of the given station feed types. Read-only."""
    f1inv = np.linalg.inv(feed_matrix(t1_feed))
    f2invH = np.linalg.inv(feed_matrix(t2_feed)).conj().T
    M = np.zeros((4, 4), dtype=complex)
    for j in range(4):
        # j-th unit correlation in (p1p1, p2p2, p1p2, p2p1) order
        slots = np.zeros(4, dtype=complex)
        slots[j] = 1.0
        Vmat = np.array([[slots[0], slots[2]],
                         [slots[3], slots[1]]])
        B = f1inv @ Vmat @ f2invH          # sky coherency in (X, Y) basis
        M[0, j] = 0.5 * (B[0, 0] + B[1, 1])         # I = (XX + YY)/2
        M[1, j] = 0.5 * (B[0, 0] - B[1, 1])         # Q = (XX - YY)/2
        M[2, j] = 0.5 * (B[0, 1] + B[1, 0])         # U = (XY + YX)/2
        M[3, j] = 0.5j * (B[1, 0] - B[0, 1])        # V = i(YX - XY)/2
    M.flags.writeable = False
    return M


def coherency_to_stokes(p1p1, p2p2, p1p2, p2p1, t1_feed, t2_feed):
    """Recover (I, Q, U, V) from the four generic-slot correlations of a single
       feed pairing (arrays of equal shape, or scalars). Ideal-feed assumption."""
    _maybe_warn_convention()
    M = _coherency_matrix(t1_feed, t2_feed)
    corr = np.stack([np.asarray(p1p1), np.asarray(p2p2),
                     np.asarray(p1p2), np.asarray(p2p1)])
    stokes = np.tensordot(M, corr, axes=([1], [0]))
    return stokes[0], stokes[1], stokes[2], stokes[3]


def stokes_to_coherency(i, q, u, v, t1_feed, t2_feed):
    """Inverse of coherency_to_stokes: build the four generic-slot correlations
       (p1p1, p2p2, p1p2, p2p1) for a feed pairing from Stokes."""
    _maybe_warn_convention()
    Minv = np.linalg.inv(_coherency_matrix(t1_feed, t2_feed))
    stokes = np.stack([np.asarray(i), np.asarray(q),
                       np.asarray(u), np.asarray(v)])
    corr = np.tensordot(Minv, stokes, axes=([1], [0]))
    return corr[0], corr[1], corr[2], corr[3]


def coherency_to_stokes_sigma(s_p1p1, s_p2p2, s_p1p2, s_p2p1, t1_feed, t2_feed):
    """Marginal (diagonal, covariance-dropping) sigma propagation of the four
       slot sigmas to (sigma_I, sigma_Q, sigma_U, sigma_V), matching the lossy
       convention of the other *_sigma helpers here."""
    W = np.abs(_coherency_matrix(t1_feed, t2_feed))**2
    var = np.stack([np.asarray(s_p1p1)**2, np.asarray(s_p2p2)**2,
                    np.asarray(s_p1p2)**2, np.asarray(s_p2p1)**2])
    out = np.sqrt(np.tensordot(W, var, axes=([1], [0])))
    return out[0], out[1], out[2], out[3]


def correlation_slot(t1_feed, t2_feed, feed_a, feed_b):
    """Return the generic DTPOL_MIXED slot name ('p1p1vis'/'p1p2vis'/'p2p1vis'/
       'p2p2vis') holding the correlation feed_a * conj(feed_b) for a baseline
       with these station feed types, or None if the baseline does not measure
       it (a station lacks the required feed). Resolves by feed-letter position,
       so it is correct for reordered/hybrid feeds (e.g. 'lr', 'rx') too."""
    if len(t1_feed) != 2 or t1_feed[0] == t1_feed[1]:
        raise ValueError(f"correlation_slot: invalid feed {t1_feed!r}")
    if len(t2_feed) != 2 or t2_feed[0] == t2_feed[1]:
        raise ValueError(f"correlation_slot: invalid feed {t2_feed!r}")
    i = t1_feed.find(feed_a)
    j = t2_feed.find(feed_b)
    if i < 0 or j < 0:
        return None
    return {(0, 0): 'p1p1vis', (0, 1): 'p1p2vis',
            (1, 0): 'p2p1vis', (1, 1): 'p2p2vis'}[(i, j)]


# ---------------------------------------------------------------------------
# Jones-matrix scaffolding
# ---------------------------------------------------------------------------
#
# Factoring convention: J = G @ (I + D), where
#
#     G = diag(g_p1, g_p2)           (diagonal complex gains)
#     I + D = [[1,    d_p1],
#              [d_p2, 1   ]]         (D-term cross-coupling)
#
# Verify against existing pol_cal* code when the first consumer
# (full-Jones applycal) lands.

def jones_matrix(g_p1, g_p2, d_p1=0.0, d_p2=0.0):
    """Build a 2x2 Jones matrix from per-feed gains and D-terms.

    Parameters
    ----------
    g_p1, g_p2 : complex
        Per-feed complex gains. Diagonal of G.
    d_p1, d_p2 : complex, optional
        Off-diagonal D-term cross-coupling. Default 0 (ideal feeds).

    Returns
    -------
    J : (2, 2) complex ndarray
        ``J = G @ (I + D)``.
    """
    G = np.array([[g_p1, 0.0],
                  [0.0, g_p2]], dtype=complex)
    IpD = np.array([[1.0, d_p1],
                    [d_p2, 1.0]], dtype=complex)
    return G @ IpD


def invert_jones(J):
    """Return the inverse of a Jones matrix.

    Broadcasts over leading axes. Reference only: ``apply_inverse_jones_to_coherency``
    currently inlines ``np.linalg.inv``.
    """
    return np.linalg.inv(J)


def apply_inverse_jones_to_coherency(V_obs, J1, J2):
    """Apply the inverse Jones correction to an observed coherency matrix.

    Computes ``V_corr = J1^{-1} @ V_obs @ (J2^dagger)^{-1}``.

    Parameters
    ----------
    V_obs : (..., 2, 2) complex ndarray
        Observed coherency matrix. Leading axes broadcast.
    J1, J2 : (..., 2, 2) complex ndarray
        Jones matrices at stations 1 and 2.

    Returns
    -------
    V_corr : (..., 2, 2) complex ndarray
    """
    _maybe_warn_convention()
    J1_inv = np.linalg.inv(J1)
    J2_dagger_inv = np.linalg.inv(np.conjugate(np.swapaxes(J2, -1, -2)))
    return J1_inv @ V_obs @ J2_dagger_inv


def apply_jones_to_coherency(V_true, J1, J2):
    """Apply the forward Jones corruption to a true coherency matrix.

    Computes ``V_obs = J1 @ V_true @ J2^dagger`` (EHT Paper VII / docs sec 9).
    This is the exact inverse operation of
    :func:`apply_inverse_jones_to_coherency` and is what simulation uses to
    corrupt clean visibilities. Broadcasts over leading axes; pure (no in-place
    mutation).

    Parameters
    ----------
    V_true : (..., 2, 2) complex ndarray
        True (clean) coherency matrix. Leading axes broadcast.
    J1, J2 : (..., 2, 2) complex ndarray
        Jones matrices at stations 1 and 2.

    Returns
    -------
    V_obs : (..., 2, 2) complex ndarray
    """
    _maybe_warn_convention()
    return J1 @ V_true @ np.conjugate(np.swapaxes(J2, -1, -2))


# ---------------------------------------------------------------------------
# Forward Jones assembly (feeds obs_simulate corruption; calibration reuses it)
# ---------------------------------------------------------------------------
#
# The observing code (obs_simulate) generates the physical corruption
# parameters -- complex gains, D-terms, and the field-rotation angle from
# array geometry -- and this module assembles them into the 2x2 Jones matrix.
# Keeping assembly here (rather than inline in obs_simulate) gives calibration
# a single home to reuse and keeps the kernel pure/traceable for the JAX track.


def field_rotation_matrix(feed_type, fr_angle):
    """Field-rotation Jones factor ``Phi`` for a station's feeds.

    For circular feeds (``'rl'``) field rotation is a diagonal phase,
    ``Phi = diag(exp(-i*fr_angle), exp(+i*fr_angle))`` with R = p1 leading.
    This reproduces the field-rotation convention baked into
    ``obs_simulate.make_jones`` (cf. docs/polarization_conventions.md sec 8/12).

    For linear feeds (``'xy'``) field rotation is a real rotation of the feed
    axes, ``Phi = [[cos, sin], [-sin, cos]]``. This form is NOT a free choice:
    it is FIXED by consistency with the circular convention via
    ``Phi_lin = F^{-1} Phi_circ F`` where ``F = BASIS_LIN_TO_CIRC``.

    ASSUMPTION (to verify; see docs/polarization_conventions.md sec 10): the
    derived linear form and the sign of ``fr_angle`` (from tarr ``fr_elev``/
    ``fr_par``) match EHT Paper VII's field-rotation Jones. Hybrid feeds
    (e.g. ``'rx'``) have an undecided convention and raise.

    Parameters
    ----------
    feed_type : str
        Two-character station feed type. ``'rl'`` (circular) and ``'xy'``
        (linear) are supported; hybrid feeds raise ``NotImplementedError``.
    fr_angle : float or ndarray
        Field-rotation angle(s) in radians. Broadcasts over leading axes.

    Returns
    -------
    Phi : ndarray
        Complex array of shape ``np.shape(fr_angle) + (2, 2)``.
    """
    fr_angle = np.asarray(fr_angle, dtype=float)
    if feed_type == 'rl':
        # Circular feeds: field rotation is a diagonal phase.
        z = np.exp(1j * fr_angle)
        zc = np.exp(-1j * fr_angle)
        zeros = np.zeros_like(z)
        row0 = np.stack([zc, zeros], axis=-1)
        row1 = np.stack([zeros, z], axis=-1)
        return np.stack([row0, row1], axis=-2)
    elif feed_type == 'xy':
        # Linear feeds: real rotation of the feed axes. Form fixed by
        # consistency with the circular convention (F^{-1} Phi_circ F).
        c = np.cos(fr_angle)
        s = np.sin(fr_angle)
        zeros = np.zeros_like(c)
        row0 = np.stack([c, s], axis=-1)
        row1 = np.stack([-s, c], axis=-1)
        return np.stack([row0, row1], axis=-2).astype(complex)
    else:
        raise NotImplementedError(
            f"field_rotation_matrix: feed_type {feed_type!r} not supported. Only "
            "orthogonal same-basis feeds 'rl' (circular) and 'xy' (linear) are "
            "implemented; hybrid feeds (e.g. 'rx') have an undecided field-"
            "rotation convention (see docs/polarization_conventions.md sec 10).")


def assemble_jones(g_p1, g_p2, d_p1, d_p2, feed_type, fr_angle, fr_angle_D=0.0):
    """Assemble per-station Jones matrices ``J = G (I + D_rot) Phi``.

    Factoring (EHT Paper VII Eq. 7; docs/polarization_conventions.md sec 10):

        G     = diag(g_p1, g_p2)          complex per-feed gains
        I + D = [[1, d_p1], [d_p2, 1]]    leakage / D-terms
        Phi                              field rotation (see
                                         :func:`field_rotation_matrix`)

    ``fr_angle_D`` is the residual field-rotation angle applied to the leakage
    when field rotation has been corrected but leakage has not (the
    ``fr_angle_D`` term in ``obs_simulate.make_jones``); it rotates the D-terms
    by ``Phi_D^{-1}``. With ``fr_angle_D = 0`` this is the plain
    ``J = G (I + D) Phi``.

    Reproduces ``obs_simulate.make_jones``'s circular assembly exactly for
    ``feed_type='rl'``, and extends to linear feeds (``'xy'``) via the linear
    ``Phi`` in :func:`field_rotation_matrix`. Pure and broadcasting (no in-place
    mutation), so it is safe for the JAX backend.

    LIMITATION (ASSUMPTION, to verify; see docs/polarization_conventions.md sec 10):
    for non-circular feeds the one-sided ``Phi_D^{-1} @ D`` leakage rotation is
    only correct when ``Phi`` is diagonal (circular). For linear ``Phi`` the
    frcal-but-not-dcal case needs the full conjugation, so a nonzero
    ``fr_angle_D`` with a non-circular feed raises ``NotImplementedError``.

    Parameters
    ----------
    g_p1, g_p2 : complex or ndarray
        Per-feed complex gains (diagonal of G). Broadcast over leading axes
        (e.g. one entry per time).
    d_p1, d_p2 : complex or ndarray
        Per-feed D-terms (off-diagonal leakage).
    feed_type : str
        Station feed type; ``'rl'`` and ``'xy'`` supported.
    fr_angle : float or ndarray
        Field-rotation angle(s) in radians.
    fr_angle_D : float or ndarray, optional
        Residual leakage-rotation angle(s) in radians. Default 0. Must be 0 for
        non-circular feeds (see LIMITATION above).

    Returns
    -------
    J : ndarray
        Complex array of shape ``broadcast(inputs) + (2, 2)``.
    """
    fr_angle_D = np.asarray(fr_angle_D, dtype=float)
    if feed_type != 'rl' and np.any(fr_angle_D != 0.0):
        raise NotImplementedError(
            "assemble_jones: nonzero fr_angle_D (leakage double-rotation for the "
            "frcal-but-not-dcal case) is only implemented for circular feeds; "
            "the linear conjugation form is pending confirmation "
            "(see docs/polarization_conventions.md sec 10).")
    g_p1 = np.asarray(g_p1, dtype=complex)
    g_p2 = np.asarray(g_p2, dtype=complex)
    gshape = np.broadcast(g_p1, g_p2).shape
    gzeros = np.zeros(gshape, dtype=complex)
    G = np.stack([np.stack([np.broadcast_to(g_p1, gshape), gzeros], axis=-1),
                  np.stack([gzeros, np.broadcast_to(g_p2, gshape)], axis=-1)],
                 axis=-2)

    d_p1 = np.asarray(d_p1, dtype=complex)
    d_p2 = np.asarray(d_p2, dtype=complex)
    dshape = np.broadcast(d_p1, d_p2).shape
    dzeros = np.zeros(dshape, dtype=complex)
    D = np.stack([np.stack([dzeros, np.broadcast_to(d_p1, dshape)], axis=-1),
                  np.stack([np.broadcast_to(d_p2, dshape), dzeros], axis=-1)],
                 axis=-2)

    Phi = field_rotation_matrix(feed_type, fr_angle)
    Phi_D_inv = field_rotation_matrix(feed_type, -fr_angle_D)
    IpD = np.eye(2, dtype=complex) + Phi_D_inv @ D
    return G @ IpD @ Phi
