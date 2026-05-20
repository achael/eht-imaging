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

Convention (IAU / Hamaker-Bregman-Sault / CASA / AIPS):

    R = (X - iY) / sqrt(2)
    L = (X + iY) / sqrt(2)

with the circular Stokes formulas:

    I = (RR + LL) / 2,    V = (RR - LL) / 2
    Q = (RL + LR) / 2,    U = i (LR - RL) / 2

and the inverted linear-feed Stokes formulas:

    I = (XX + YY) / 2,    Q = (XX - YY) / 2
    U = (XY + YX) / 2,    V = +i (XY - YX) / 2

The sign of V flips under the opposite basis choice
(R = (X + iY) / sqrt(2)). Full derivation and worked examples live in
``docs/polarization_conventions.md``.

A ``MixedPolConventionWarning`` fires once per session on the first
non-identity transform call.
"""

import warnings

import numpy as np

from ehtim.warnings import MixedPolConventionWarning

# ---------------------------------------------------------------------------
# Basis matrices
# ---------------------------------------------------------------------------

# Maps a feed-basis vector (X, Y) -> (R, L) under the IAU/HBS convention.
# Rows indexed by (R, L); columns by (X, Y).
BASIS_LIN_TO_CIRC = np.array([[1.0, -1.0j],
                              [1.0, +1.0j]]) / np.sqrt(2.0)

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
        "Bregman-Sault convention (R = (X - iY)/sqrt(2)). This "
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
# Visibility transforms (vis only; sigmas handled by *_sigma helpers below)
# ---------------------------------------------------------------------------

def circ_to_stokes(rrvis, llvis, rlvis, lrvis):
    """Convert circular-feed visibilities to Stokes visibilities.

    Returns
    -------
    (vis, qvis, uvis, vvis) : tuple of ndarrays
    """
    _maybe_warn_convention()
    vis = 0.5 * (rrvis + llvis)
    qvis = 0.5 * (lrvis + rlvis)
    uvis = 0.5j * (lrvis - rlvis)
    vvis = 0.5 * (rrvis - llvis)
    return vis, qvis, uvis, vvis


def stokes_to_circ(vis, qvis, uvis, vvis):
    """Convert Stokes visibilities to circular-feed visibilities.

    Returns
    -------
    (rrvis, llvis, rlvis, lrvis) : tuple of ndarrays
    """
    _maybe_warn_convention()
    rrvis = vis + vvis
    llvis = vis - vvis
    rlvis = qvis + 1.0j * uvis
    lrvis = qvis - 1.0j * uvis
    return rrvis, llvis, rlvis, lrvis


def lin_to_stokes(xxvis, yyvis, xyvis, yxvis):
    """Convert linear-feed visibilities to Stokes (IAU/HBS).

    Returns
    -------
    (vis, qvis, uvis, vvis) : tuple of ndarrays
    """
    _maybe_warn_convention()
    vis = 0.5 * (xxvis + yyvis)
    qvis = 0.5 * (xxvis - yyvis)
    uvis = 0.5 * (xyvis + yxvis)
    vvis = 0.5j * (xyvis - yxvis)
    return vis, qvis, uvis, vvis


def stokes_to_lin(vis, qvis, uvis, vvis):
    """Convert Stokes visibilities to linear-feed visibilities (IAU/HBS).

    Inverse of ``lin_to_stokes``.

    Returns
    -------
    (xxvis, yyvis, xyvis, yxvis) : tuple of ndarrays
    """
    _maybe_warn_convention()
    xxvis = vis + qvis
    yyvis = vis - qvis
    xyvis = uvis - 1.0j * vvis
    yxvis = uvis + 1.0j * vvis
    return xxvis, yyvis, xyvis, yxvis


def lin_to_circ(xxvis, yyvis, xyvis, yxvis):
    """Convert linear-feed visibilities to circular-feed via Stokes."""
    return stokes_to_circ(*lin_to_stokes(xxvis, yyvis, xyvis, yxvis))


def circ_to_lin(rrvis, llvis, rlvis, lrvis):
    """Convert circular-feed visibilities to linear-feed via Stokes."""
    return stokes_to_lin(*circ_to_stokes(rrvis, llvis, rlvis, lrvis))


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

    Broadcasts over leading axes.
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
