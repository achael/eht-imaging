# pol_imager_utils.py
# General imager functions for polarimetric VLBI data
#
#    Copyright (C) 2018 Andrew Chael
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

import matplotlib.pyplot as plt
import numpy as np

from ehtim.backends import array_namespace

try:
    from pynfft.nfft import NFFT
    _HAS_NFFT = True
except ImportError:
    NFFT = None
    _HAS_NFFT = False

from ehtim.const_def import FFT_PAD_DEFAULT, GRIDDER_P_RAD_DEFAULT, NFFT_EPS_DEFAULT, RADPERAS
from ehtim.observing.obs_helpers import NFFTInfo, ftmatrix, ticks

TANWIDTH_M = 0.5
TANWIDTH_V = 1
TANWIDTH_PSI = 1
POL_SOLVE_DEFAULT = (0,1,1,0)
POL_SOLVE_DEFAULT_V = (0,0,0,1)
RANDOMFLOOR=True
##################################################################################################
# Constants & Definitions
##################################################################################################

NORM_REGULARIZER = True
DATATERMS_POL = ['pvis', 'm','vvis']
REGULARIZERS_POL = ['msimple', 'hw', 'ptv', 'l1v', 'l2v', 'vtv', 'vtv2', 'vflux']

nit = 0 # global variable to track the iteration number in the plotting callback

# CONVENTIONS
# P = M = RL = Q+iU = I \rho cos(\psi) e^(i\phi)
# \phi = 2\chi = 2*EVPA
# m = abs(Q+iU)/I = I \rho cos(\psi)
# rho = Sqrt(Q^2 + U^2 + V^2)/I
# V = I \rho sin(\psi)
# v = V/I = \rho sin(\psi)
# imarr = (I, \rho, \phi, \psi)

##################################################################################################
# Linear Polarimetric image representations and Change of Variables
##################################################################################################

def make_i_image(imarr):
    """return total intensity image
    """
    return imarr[0]

def make_p_image(imarr):
    """construct a polarimetric image P = RL = Q + iU
    """

    # NOTE! We replaced EVPA chi with phi=2chi in the imarr
    xp = array_namespace(imarr)
    pimage = imarr[0] * imarr[1] * xp.exp(1j*imarr[2]) * xp.cos(imarr[3])

    return pimage

def make_m_image(imarr):
    """construct a polarimetric ratrio image abs(P/I) = abs(Q + iU)/I
    """

    xp = array_namespace(imarr)
    mimage = imarr[1] * xp.cos(imarr[3])

    return mimage

def make_chi_image(imarr):
    """construct an EVPA image
    """
    # NOTE! We replaced EVPA chi with phi=2chi in the imarr
    chiimage = 0.5*imarr[2]

    return chiimage

def make_psi_image(imarr):
    """construct a circular polarization angle image
    """
    psiimage = imarr[3]

    return psiimage

def make_q_image(imarr):
    """construct an image of stokes Q
    """

    # NOTE! We replaced EVPA chi with phi=2chi in the imarr
    xp = array_namespace(imarr)
    qimage = imarr[0] * imarr[1] * xp.cos(imarr[2]) *  xp.cos(imarr[3])

    return qimage

def make_u_image(imarr):
    """construct an image of stokes U
    """
    # NOTE! We replaced EVPA chi with phi=2chi in the imarr
    xp = array_namespace(imarr)
    uimage = imarr[0] * imarr[1] * xp.sin(imarr[2]) *  xp.cos(imarr[3])

    return uimage

def make_v_image(imarr):
    """construct an image of stokes V
    """

    xp = array_namespace(imarr)
    vimage = imarr[0] * imarr[1] * xp.sin(imarr[3])

    return vimage

def make_vf_image(imarr):
    """construct an image of stokes V/I
    """

    xp = array_namespace(imarr)
    vfimage = imarr[1] * xp.sin(imarr[3])

    return vfimage

def polcv(imarr):
    r"""change of variables rho(rho') from range (-inf, inf) to (0,1)
       and \psi(\psi') from range (-inf, inf) to (-pi/2, pi/2)
       input is solver values, output is physical values
    """

    xp = array_namespace(imarr)
    rho_prime =  imarr[1]
    rho = 0.5 + xp.arctan(rho_prime/TANWIDTH_M)/np.pi

    psi_prime = imarr[3]
    psi = xp.arctan(psi_prime/TANWIDTH_PSI)

    out = xp.stack((imarr[0], rho, imarr[2], psi))
    return out

def polcv_r(imarr):
    r"""Reverse change of variables rho'(rho) from range (0,1) to (-inf, inf)
       and \psi'(\psi) from range (-pi/2, pi/2) to (-inf, inf)
       input is physical values, output is solver values
    """

    rho = imarr[1]
    rho_prime = TANWIDTH_M*np.tan(np.pi*(rho - 0.5))

    psi = imarr[3]
    psi_prime = TANWIDTH_PSI*np.tan(psi)

    out = np.array((imarr[0], rho_prime, imarr[2], psi_prime))

    return out

def polcv_grad(imarr, gradarr):
    """Apply J^T @ gradarr for polcv (rho ← arctan(rho_pre/T_M); psi ← arctan(psi_pre/T_PSI)).

    Diagonal Jacobian on slots 1 and 3; slot 2 (phi) passes through.

    Parameters
    ----------
    imarr : np.ndarray, shape (4, ...)
        Solver-space image array.
    gradarr : np.ndarray, shape (4, ...)
        Gradient w.r.t. physical components phys[0:4]. gradarr[0] is unused.

    Returns
    -------
    out : np.ndarray, shape (3, ...)
        Gradient w.r.t. solver slots (1, 2, 3).
    """
    rho_pre = imarr[1]
    psi_pre = imarr[3]
    drho_dpre = 1 / (TANWIDTH_M * np.pi * (1 + (rho_pre / TANWIDTH_M) ** 2))
    dpsi_dpre = 1 / (TANWIDTH_PSI * (1 + (psi_pre / TANWIDTH_PSI) ** 2))

    out = np.empty((3,) + gradarr.shape[1:])
    out[0] = drho_dpre * gradarr[1]
    out[1] = gradarr[2]
    out[2] = dpsi_dpre * gradarr[3]
    return out


def mcv(imarr):
    """change of variables m(n') from range (-inf, inf) to (0,1)
       input is solver values, output is physical values
    """

    xp = array_namespace(imarr)
    vfrac = imarr[3] # when using this transform, we interpret transformed imarr[3] as mfrac=\rho sin(\psi)
    mfrac_max = 1-xp.abs(vfrac)
    if xp is np and np.any(mfrac_max>1):  # dead guard (mfrac_max<=1 always); kept on numpy
        raise Exception("mfrac_max>1 in mcv!")

    mfrac_prime =  imarr[1]
    mfrac = mfrac_max*(0.5 + xp.arctan(mfrac_prime/TANWIDTH_M)/np.pi)

    rho = xp.sqrt(mfrac**2 + vfrac**2)
    psi = xp.arcsin(vfrac/rho)

    out = xp.stack((imarr[0], rho, imarr[2], psi))
    return out

def mcv_r(imarr):
    """Reverse change of variables m'(m) from range (0,1-|v|) to (-inf, inf)
       input is physical values, output is solver values
    """
    rho = imarr[1]
    psi = imarr[3]
    mfrac = rho*np.cos(psi)
    vfrac = rho*np.sin(psi)
    mfrac_max = 1-np.abs(vfrac)
    if np.any(mfrac_max>1):
        raise Exception("mfrac_max>1 in mcv!")

    mfrac_prime = TANWIDTH_M*np.tan(np.pi*(mfrac/mfrac_max - 0.5))
    out = np.array((imarr[0], mfrac_prime, imarr[2], vfrac))
    return out

def mcv_grad(imarr, gradarr):
    """Apply J^T @ gradarr for mcv (mprime → rho, psi; vfrac held constant).

    phys[1]=rho and phys[3]=psi both depend on mprime via mfrac, so the
    Jacobian has off-diagonal terms; both are included here. Slot 3 of
    the result is zero because vfrac is held constant.

    Parameters
    ----------
    imarr : np.ndarray, shape (4, ...)
        Solver-space image array.
    gradarr : np.ndarray, shape (4, ...)
        Gradient w.r.t. physical components phys[0:4]. gradarr[0] is unused.

    Returns
    -------
    out : np.ndarray, shape (3, ...)
        Gradient w.r.t. solver slots (1, 2, 3).
    """
    vfrac = imarr[3]
    mfrac_max = 1 - np.abs(vfrac)

    mprime = imarr[1]
    mfrac = mfrac_max * (0.5 + np.arctan(mprime / TANWIDTH_M) / np.pi)
    rho = np.sqrt(mfrac ** 2 + vfrac ** 2)

    dm_dmprime = mfrac_max / (TANWIDTH_M * np.pi * (1 + (mprime / TANWIDTH_M) ** 2))

    # Avoid 0/0 when total polarization is zero; in that limit both terms vanish.
    safe_rho = np.where(rho > 0, rho, 1.0)
    drho_dmprime = (mfrac / safe_rho) * dm_dmprime
    dpsi_dmprime = -(vfrac * np.sign(mfrac)) / (safe_rho ** 2) * dm_dmprime

    out = np.empty((3,) + gradarr.shape[1:])
    out[0] = drho_dmprime * gradarr[1] + dpsi_dmprime * gradarr[3]
    out[1] = gradarr[2]
    out[2] = 0.0
    return out

def vcv(imarr):
    """change of variables for v(v') from range (-inf,inf) to (-1+|m|,1-|m|) while keeping m=P/I fixed
       input is solver values, output is physical values"""

    xp = array_namespace(imarr)
    mfrac = imarr[1] # when using this transform, we interpret transformed imarr[1] as mfrac=\rho cos(\psi)
    vfrac_max = 1-xp.abs(mfrac)
    if xp is np and np.any(vfrac_max>1):  # dead guard (vfrac_max<=1 always); kept on numpy
        raise Exception("vfrac_max>1 in vcv!")

    vfrac_prime = imarr[3]
    vfrac = 2*vfrac_max*xp.arctan(vfrac_prime/TANWIDTH_V)/np.pi

    rho = xp.sqrt(mfrac**2 + vfrac**2)
    psi = xp.arcsin(vfrac/rho)

    out = xp.stack((imarr[0], rho, imarr[2], psi))
    return out

def vcv_r(imarr):
    """Reverse change of variables for v'(v) from range(-1+|m|,1-|m|) to (-inf,inf) keeping m=P/I fixed
       input is physical values, output is solver values"""
    rho = imarr[1]
    psi = imarr[3]
    mfrac = rho*np.cos(psi)
    vfrac = rho*np.sin(psi)
    vfrac_max = 1-np.abs(mfrac)
    if np.any(vfrac_max>1):
        raise Exception("vfrac_max>1 in vcv_r!")

    vfrac_prime = TANWIDTH_V*np.tan(np.pi*vfrac/(2*vfrac_max))
    out = np.array((imarr[0], mfrac, imarr[2], vfrac_prime))
    return out

def vcv_grad(imarr, gradarr):
    """Apply J^T @ gradarr for vcv (vprime → rho, psi; mfrac held constant).

    phys[1]=rho and phys[3]=psi both depend on vprime via vfrac, so the
    Jacobian has off-diagonal terms. At mfrac=0 (pol='IV' regime) psi is
    locally constant and the entire gradient flows through drho/dvprime.
    Slot 1 of the result is zero because mfrac is held constant.

    Parameters
    ----------
    imarr : np.ndarray, shape (4, ...)
        Solver-space image array.
    gradarr : np.ndarray, shape (4, ...)
        Gradient w.r.t. physical components phys[0:4]. gradarr[0] is unused.

    Returns
    -------
    out : np.ndarray, shape (3, ...)
        Gradient w.r.t. solver slots (1, 2, 3).
    """
    mfrac = imarr[1]
    vfrac_max = 1 - np.abs(mfrac)

    vprime = imarr[3]
    vfrac = 2 * vfrac_max * np.arctan(vprime / TANWIDTH_V) / np.pi
    rho = np.sqrt(mfrac ** 2 + vfrac ** 2)

    dv_dvprime = 2 * vfrac_max / (TANWIDTH_V * np.pi * (1 + (vprime / TANWIDTH_V) ** 2))

    safe_rho = np.where(rho > 0, rho, 1.0)
    drho_dvprime = (vfrac / safe_rho) * dv_dvprime
    dpsi_dvprime = (np.abs(mfrac) / (safe_rho ** 2)) * dv_dvprime

    out = np.empty((3,) + gradarr.shape[1:])
    out[0] = 0.0
    out[1] = gradarr[2]
    out[2] = drho_dvprime * gradarr[1] + dpsi_dvprime * gradarr[3]
    return out


##################################################################################################
# Wrapper Functions
##################################################################################################

def polchisq(imarr, A, data, sigma, dtype, ttype='direct', mask=[]):
    """Return chi^2 for a polarimetric data term.

    Thin shim around imager_backend.compute_chisq_term retained for backward
    compatibility. New code should call compute_chisq_term directly.
    """
    from ehtim.imaging.imager_backend import compute_chisq_term
    if dtype not in DATATERMS_POL:
        raise Exception(f"data term {dtype!r} is not a polarimetric data term")
    return compute_chisq_term(imarr, dtype, A, data, sigma,
                              ttype=ttype, mask=mask)


def polchisqgrad(imarr, A, data, sigma, dtype, ttype='direct', mask=[],
                 pol_solve=POL_SOLVE_DEFAULT):
    """Return chi^2 gradient for a polarimetric data term.

    Thin shim around imager_backend.compute_chisqgrad_term retained for
    backward compatibility. New code should call compute_chisqgrad_term
    directly.
    """
    from ehtim.imaging.imager_backend import compute_chisqgrad_term
    if dtype not in DATATERMS_POL:
        raise Exception(f"data term {dtype!r} is not a polarimetric data term")
    return compute_chisqgrad_term(imarr, dtype, A, data, sigma,
                                  ttype=ttype, mask=mask, pol_solve=pol_solve)


def polregularizer(imarr, priorarr, mask, flux, pflux, vflux, xdim, ydim, psize, stype, **kwargs):
    """Return the regularizer value for a polarimetric regularizer.

    Thin shim around imager_backend.compute_regularizer_term retained for
    backward compatibility. New code should call compute_regularizer_term
    directly. `priorarr` is currently not consumed by any pol regularizer.
    """
    from ehtim.imaging.imager_backend import compute_regularizer_term
    if stype not in REGULARIZERS_POL:
        raise Exception(f"regularizer term {stype!r} is not a polarimetric regularizer")
    return compute_regularizer_term(imarr, stype, mask,
                                    flux=flux, pflux=pflux, vflux=vflux,
                                    xdim=xdim, ydim=ydim, psize=psize, **kwargs)


def polregularizergrad(imarr, priorarr, mask, flux, pflux, vflux, xdim, ydim, psize, stype, **kwargs):
    """Return the regularizer gradient for a polarimetric regularizer.

    Thin shim around imager_backend.compute_regularizergrad_term retained for
    backward compatibility. New code should call compute_regularizergrad_term
    directly.
    """
    from ehtim.imaging.imager_backend import compute_regularizergrad_term
    if stype not in REGULARIZERS_POL:
        raise Exception(f"regularizer term {stype!r} is not a polarimetric regularizer")
    return compute_regularizergrad_term(imarr, stype, mask,
                                        flux=flux, pflux=pflux, vflux=vflux,
                                        xdim=xdim, ydim=ydim, psize=psize, **kwargs)


def polchisqdata(Obsdata, Prior, mask, dtype, **kwargs):
    """Return (data, sigma, A) for a polarimetric data term.

    Thin shim around imager_backend.compute_chisqdata_term retained for
    backward compatibility. New code should call compute_chisqdata_term
    directly.
    """
    from ehtim.imaging.imager_backend import (
        ImagerConfig,
        MfConfig,
        compute_chisqdata_term,
    )
    ttype = kwargs.pop('ttype', 'direct')
    pol = kwargs.pop('pol', 'I')
    if dtype not in DATATERMS_POL:
        raise Exception(f"data term {dtype!r} is not a polarimetric data term")
    config = ImagerConfig(
        pol=pol, transforms=[], ttype=ttype, mf=False,
        mf_config=MfConfig(mf_order=0, mf_order_pol=0, mf_rm=0, mf_cm=0),
    )
    return compute_chisqdata_term(Obsdata, Prior, mask, dtype, config, **kwargs)


##################################################################################################
# DFT Chi-squared and Gradient Functions
##################################################################################################

def chisq_p(imarr, Amatrix, p, sigmap):
    """Polarimetric visibility chi-squared
    """

    xp = array_namespace(imarr)
    pimage = make_p_image(imarr)
    psamples = xp.dot(Amatrix, pimage)
    chisq =  xp.sum(xp.abs(p - psamples)**2/(sigmap**2)) / (2*len(p))
    return chisq

def chisqgrad_p(imarr, Amatrix, p, sigmap,pol_solve=POL_SOLVE_DEFAULT):
    """Polarimetric visibility chi-squared gradient
    """
    gradout = np.zeros(imarr.shape)

    iimage = make_i_image(imarr)
    pimage = make_p_image(imarr)
    mimage = make_m_image(imarr)
    chiimage = make_chi_image(imarr)
    psiimage = make_psi_image(imarr)

    psamples = np.dot(Amatrix, pimage)
    pdiff = (p - psamples) / (sigmap**2)

    # TODO: for now, use previous gradients and modify with chain rule
    if pol_solve[0]!=0:
        gradi = -np.real(mimage * np.exp(-2j*chiimage) * np.dot(Amatrix.conj().T, pdiff)) / len(p)
        gradout[0] = gradi

    if pol_solve[1]!=0:
        gradm = -np.real(iimage * np.exp(-2j*chiimage) * np.dot(Amatrix.conj().T, pdiff)) / len(p)
        gradrho = gradm * np.cos(psiimage)
        gradout[1] = gradrho

    if pol_solve[2]!=0:
        gradchi = -2 * np.imag(pimage.conj() * np.dot(Amatrix.conj().T, pdiff)) / len(p)
        gradphi = 0.5*gradchi
        gradout[2] = gradphi

    # TODO check
    if pol_solve[3]!=0:
        gradm = -np.real(iimage * np.exp(-2j*chiimage) * np.dot(Amatrix.conj().T, pdiff)) / len(p)
        gradpsi = gradm * (-mimage*np.tan(psiimage))
        gradout[3] = gradpsi

    return gradout

def chisq_m(imarr, Amatrix, m, sigmam):
    """Polarimetric ratio chi-squared
    """

    xp = array_namespace(imarr)
    iimage = make_i_image(imarr)
    pimage = make_p_image(imarr)
    msamples = xp.dot(Amatrix, pimage) / xp.dot(Amatrix, iimage)
    chisq = xp.sum(xp.abs(m - msamples)**2/(sigmam**2)) / (2*len(m))
    return chisq

def chisqgrad_m(imarr, Amatrix, m, sigmam,pol_solve=POL_SOLVE_DEFAULT):
    """The gradient of the polarimetric ratio chisq
    """
    gradout = np.zeros(imarr.shape)

    iimage = make_i_image(imarr)
    pimage = make_p_image(imarr)
    mimage = make_m_image(imarr)
    chiimage = make_chi_image(imarr)
    psiimage = make_psi_image(imarr)

    isamples = np.dot(Amatrix, iimage)
    psamples = np.dot(Amatrix, pimage)
    msamples = psamples/isamples
    mdiff = (m - msamples) / (isamples.conj() * sigmam**2)

    if pol_solve[0]!=0:
        gradi = (-np.real(mimage * np.exp(-2j*chiimage) * np.dot(Amatrix.conj().T, mdiff)) / len(m) +
                  np.real(np.dot(Amatrix.conj().T, msamples.conj() * mdiff)) / len(m))
        gradout[0] = gradi

    if pol_solve[1]!=0:
        gradm = -np.real(iimage*np.exp(-2j*chiimage) * np.dot(Amatrix.conj().T, mdiff)) / len(m)
        gradrho = gradm * np.cos(psiimage)
        gradout[1] = gradrho

    if pol_solve[2]!=0:
        gradchi = -2 * np.imag(pimage.conj() * np.dot(Amatrix.conj().T, mdiff)) / len(m)
        gradphi = 0.5*gradchi
        gradout[2] = gradphi

    if pol_solve[3]!=0:
        gradm = -np.real(iimage*np.exp(-2j*chiimage) * np.dot(Amatrix.conj().T, mdiff)) / len(m)
        gradpsi = gradm * (-mimage*np.tan(psiimage))
        gradout[3] = gradpsi


    return gradout

# stokes v
def chisq_vvis(imarr, Amatrix, v, sigmav):
    """V visibility chi-squared
    """

    xp = array_namespace(imarr)
    vimage = make_v_image(imarr)
    vsamples = xp.dot(Amatrix, vimage)
    chisq =  xp.sum(xp.abs(v - vsamples)**2/(sigmav**2)) / (2*len(v))
    return chisq

def chisqgrad_vvis(imarr, Amatrix, v, sigmav, pol_solve=POL_SOLVE_DEFAULT_V):
    """V visibility chi-squared gradient
    """

    gradout = np.zeros(imarr.shape)

    iimage = make_i_image(imarr)
    vimage = make_v_image(imarr)
    vfimage = make_vf_image(imarr)
    psiimage = make_psi_image(imarr)

    vsamples = np.dot(Amatrix, vimage)
    vdiff = (v - vsamples) / (sigmav**2)

    if pol_solve[0]!=0:
        gradi = -np.real(vfimage * np.dot(Amatrix.conj().T, vdiff)) / len(v)
        gradout[0] = gradi

    if pol_solve[1]!=0:
        gradv = -np.real(iimage * np.dot(Amatrix.conj().T, vdiff)) / len(v)
        gradrho = gradv*np.sin(psiimage)
        gradout[1] = gradrho

    if pol_solve[3]!=0:
        gradv = -np.real(iimage * np.dot(Amatrix.conj().T, vdiff)) / len(v)
        gradpsi = gradv * (vfimage/np.tan(psiimage))
        gradout[3] = gradpsi

    return gradout

##################################################################################################
# NFFT Chi-squared and Gradient Functions
##################################################################################################
def chisq_p_nfft(imarr, A, p, sigmap):
    """P visibility chi-squared
    """
    pimage = make_p_image(imarr)

    #get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    #compute uniform --> nonuniform transform
    plan.f_hat = pimage.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    psamples = plan.f.copy()*pulsefac

    #compute chi^2
    chisq =  np.sum(np.abs(p - psamples)**2/(sigmap**2)) / (2*len(p))

    return chisq

def chisqgrad_p_nfft(imarr, A, p, sigmap,pol_solve=POL_SOLVE_DEFAULT):
    """P visibility chi-squared gradient
    """
    gradout = np.zeros(imarr.shape)

    iimage = make_i_image(imarr)
    pimage = make_p_image(imarr)
    mimage = make_m_image(imarr)
    chiimage = make_chi_image(imarr)
    psiimage = make_psi_image(imarr)

    #get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    #compute uniform --> nonuniform transform
    plan.f_hat = pimage.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    psamples = plan.f.copy()*pulsefac


    pdiff_vec = (-1.0/len(p)) * (p - psamples) / (sigmap**2) * pulsefac.conj()
    plan.f = pdiff_vec
    plan.adjoint()
    ppart = (plan.f_hat.copy().T).reshape(nfft_info.xdim*nfft_info.ydim)

    if pol_solve[0]!=0:
        gradi = np.real(mimage * np.exp(-2j*chiimage) * ppart)
        gradout[0] = gradi

    if pol_solve[1]!=0:
        gradm = np.real(iimage*np.exp(-2j*chiimage) *ppart)
        gradrho = gradm * np.cos(psiimage)
        gradout[1] = gradrho

    if pol_solve[2]!=0:
        gradchi = 2 * np.imag(pimage.conj() * ppart)
        gradphi = 0.5*gradchi
        gradout[2] = gradphi

    if pol_solve[3]!=0:
        gradm = np.real(iimage*np.exp(-2j*chiimage) *ppart)
        gradpsi = gradm * (-mimage*np.tan(psiimage))
        gradout[3] = gradpsi


    return gradout


def chisq_m_nfft(imarr, A, m, sigmam):
    """Polarimetric ratio chi-squared
    """
    iimage = make_i_image(imarr)
    pimage = make_p_image(imarr)

    #get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    #compute uniform --> nonuniform transform
    plan.f_hat = iimage.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    isamples = plan.f.copy()*pulsefac

    plan.f_hat = pimage.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    psamples = plan.f.copy()*pulsefac

    #compute chi^2
    msamples = psamples/isamples
    chisq = np.sum(np.abs(m - msamples)**2/(sigmam**2)) / (2*len(m))
    return chisq

def chisqgrad_m_nfft(imarr, A, m, sigmam,pol_solve=POL_SOLVE_DEFAULT):
    """Polarimetric ratio chi-squared gradient
    """
    gradout = np.zeros(imarr.shape)

    iimage = make_i_image(imarr)
    pimage = make_p_image(imarr)
    mimage = make_m_image(imarr)
    chiimage = make_chi_image(imarr)
    psiimage = make_psi_image(imarr)

    #get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    #compute uniform --> nonuniform transform
    plan.f_hat = iimage.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    isamples = plan.f.copy()*pulsefac

    plan.f_hat = pimage.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    psamples = plan.f.copy()*pulsefac

    msamples = psamples/isamples
    mdiff_vec = (-1./len(m))*(m - msamples) / (isamples.conj() * sigmam**2) * pulsefac.conj()
    plan.f = mdiff_vec
    plan.adjoint()
    mpart = (plan.f_hat.copy().T).reshape(nfft_info.xdim*nfft_info.ydim)

    if pol_solve[0]!=0: #TODO -- check!!
        plan.f = mdiff_vec * msamples.conj()
        plan.adjoint()
        mpart2 = (plan.f_hat.copy().T).reshape(nfft_info.xdim*nfft_info.ydim)

        gradi = (np.real(mimage * np.exp(-2j*chiimage) * mpart) - np.real(mpart2))
        gradout[0] = gradi

    if pol_solve[1]!=0:
        gradm = np.real(iimage*np.exp(-2j*chiimage) * mpart)
        gradrho = gradm * np.cos(psiimage)
        gradout[1] = gradrho

    if pol_solve[2]!=0:
        gradchi = 2 * np.imag(pimage.conj() * mpart)
        gradphi = 0.5*gradchi
        gradout[2] = gradphi

    if pol_solve[3]!=0:
        gradm = np.real(iimage*np.exp(-2j*chiimage) * mpart)
        gradpsi = gradm * (-mimage*np.tan(psiimage))
        gradout[3] = gradpsi

    return gradout

# stokes v
def chisq_vvis_nfft(imarr, A, v, sigmav):
    """V visibility chi-squared
    """
    vimage = make_v_image(imarr)

    #get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    #compute uniform --> nonuniform transform
    plan.f_hat = vimage.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    vsamples = plan.f.copy()*pulsefac

    #compute chi^2
    chisq =  np.sum(np.abs(v - vsamples)**2/(sigmav**2)) / (2*len(v))

    return chisq

def chisqgrad_vvis_nfft(imarr, A, v, sigmav,pol_solve=POL_SOLVE_DEFAULT):
    """V visibility chi-squared gradient
    """
    gradout = np.zeros(imarr.shape)

    iimage = make_i_image(imarr)
    vimage = make_v_image(imarr)
    vfimage = make_vf_image(imarr)
    psiimage = make_psi_image(imarr)

    #get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    #compute uniform --> nonuniform transform
    iimage = imarr[0]
    vimage = make_v_image(imarr)
    plan.f_hat = vimage.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    vsamples = plan.f.copy()*pulsefac

    vdiff_vec = (-1.0/len(v)) * (v - vsamples) / (sigmav**2) * pulsefac.conj()
    plan.f = vdiff_vec
    plan.adjoint()
    vpart = (plan.f_hat.copy().T).reshape(nfft_info.xdim*nfft_info.ydim)


    if pol_solve[0]!=0:
        gradi = np.real(vfimage*vpart)
        gradout[0] = gradi

    if pol_solve[1]!=0:
        gradv = np.real(iimage*vpart)
        gradrho = gradv*np.sin(psiimage)
        gradout[1] = gradrho

    if pol_solve[3]!=0:
        gradv = np.real(iimage*vpart)
        gradpsi = gradv * (vfimage/np.tan(psiimage))
        gradout[3] = gradpsi

    return gradout


##################################################################################################
# Polarimetric regularizers
#
# Each `reg_X` / `reggrad_X` implements a polarimetric regularizer with the
# uniform `(imarr, mask, **kwargs)` signature used by the `_REGULARIZER_DISPATCH`
# table in `imager_backend.py`. Each returns the penalty value (positive; the
# imager solver minimises it). Spatial regularizers (ptv, vtv, vtv2) use
# `embed_imarr` (not `embed`) for the pre-step and slice the gradient as
# `g[:, mask]` since the pol gradient is shaped (4, nimage) — one row per
# Stokes component, gated on `pol_solve[0..3]`.
##################################################################################################


def reg_msimple(imarr, mask, **kwargs):
    xp = array_namespace(imarr)
    flux = kwargs['flux']
    norm = flux if kwargs.get('norm_reg', True) else 1
    iimage = make_i_image(imarr)
    mimage = make_m_image(imarr)
    return xp.sum(iimage * xp.log(mimage)) / norm


def reggrad_msimple(imarr, mask, **kwargs):
    flux = kwargs['flux']
    pol_solve = kwargs.get('pol_solve', POL_SOLVE_DEFAULT)
    norm = flux if kwargs.get('norm_reg', True) else 1
    iimage = make_i_image(imarr)
    mimage = make_m_image(imarr)
    psiimage = make_psi_image(imarr)
    gradout = np.zeros(imarr.shape)
    if pol_solve[0] != 0:
        gradout[0] = np.log(mimage)
    if pol_solve[1] != 0:
        gradm = iimage / mimage
        gradout[1] = gradm * np.cos(psiimage)
    if pol_solve[3] != 0:
        gradm = iimage / mimage
        gradout[3] = gradm * (-mimage * np.tan(psiimage))
    return gradout / norm


def reg_hw(imarr, mask, **kwargs):
    xp = array_namespace(imarr)
    flux = kwargs['flux']
    norm = flux if kwargs.get('norm_reg', True) else 1
    iimage = make_i_image(imarr)
    mimage = make_m_image(imarr)
    return xp.sum(iimage * (((1+mimage)/2) * xp.log((1+mimage)/2)
                            + ((1-mimage)/2) * xp.log((1-mimage)/2))) / norm


def reggrad_hw(imarr, mask, **kwargs):
    flux = kwargs['flux']
    pol_solve = kwargs.get('pol_solve', POL_SOLVE_DEFAULT)
    norm = flux if kwargs.get('norm_reg', True) else 1
    iimage = make_i_image(imarr)
    mimage = make_m_image(imarr)
    psiimage = make_psi_image(imarr)
    gradout = np.zeros(imarr.shape)
    if pol_solve[0] != 0:
        gradout[0] = (((1+mimage)/2) * np.log((1+mimage)/2)
                      + ((1-mimage)/2) * np.log((1-mimage)/2))
    if pol_solve[1] != 0:
        gradm = iimage * np.arctanh(mimage)
        gradout[1] = gradm * np.cos(psiimage)
    if pol_solve[3] != 0:
        gradm = iimage * np.arctanh(mimage)
        gradout[3] = gradm * (-mimage * np.tan(psiimage))
    return gradout / norm


def reg_ptv(imarr, mask, **kwargs):
    from ehtim.imaging.imager_utils import embed_imarr
    xp = array_namespace(imarr)
    if np.any(np.invert(mask)):
        imarr = embed_imarr(imarr, mask, randomfloor=True)
    flux = kwargs['flux']
    nx, ny, psize = kwargs['xdim'], kwargs['ydim'], kwargs['psize']
    beam_size = kwargs.get('beam_size', 1) or psize
    norm = flux * psize / beam_size if kwargs.get('norm_reg', True) else 1
    pimage = make_p_image(imarr)
    im = pimage.reshape(ny, nx)
    impad = xp.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = xp.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = xp.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    # TODO(bug): no epsilon in the sqrt (cf. reg_tv's epsilon_tv), so the gradient
    # is singular at near-zero-|P|-difference pixels. reg_vtv / reg_vtv2 share this.
    return xp.sum(xp.sqrt(xp.abs(im_l1 - im)**2 + xp.abs(im_l2 - im)**2)) / norm


def reggrad_ptv(imarr, mask, **kwargs):
    from ehtim.imaging.imager_utils import embed_imarr
    do_slice = np.any(np.invert(mask))
    if do_slice:
        imarr = embed_imarr(imarr, mask, randomfloor=True)
    flux = kwargs['flux']
    nx, ny, psize = kwargs['xdim'], kwargs['ydim'], kwargs['psize']
    pol_solve = kwargs.get('pol_solve', POL_SOLVE_DEFAULT)
    beam_size = kwargs.get('beam_size', 1) or psize
    norm = flux * psize / beam_size if kwargs.get('norm_reg', True) else 1
    iimage = make_i_image(imarr)
    pimage = make_p_image(imarr)
    mimage = make_m_image(imarr)
    psiimage = make_psi_image(imarr)
    im = pimage.reshape(ny, nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    im_r1 = np.roll(impad, 1, axis=0)[1:ny+1, 1:nx+1]
    im_r2 = np.roll(impad, 1, axis=1)[1:ny+1, 1:nx+1]
    im_r1l2 = np.roll(np.roll(impad,  1, axis=0), -1, axis=1)[1:ny+1, 1:nx+1]
    im_l1r2 = np.roll(np.roll(impad, -1, axis=0),  1, axis=1)[1:ny+1, 1:nx+1]
    # Denominators: |forward-l1|+|forward-l2|, |back-r1|+|cross|, |back-r2|+|cross|.
    d1 = np.sqrt(np.abs(im_l1 - im)**2 + np.abs(im_l2 - im)**2)
    d2 = np.sqrt(np.abs(im_r1 - im)**2 + np.abs(im_r1l2 - im_r1)**2)
    d3 = np.sqrt(np.abs(im_r2 - im)**2 + np.abs(im_l1r2 - im_r2)**2)
    # Numerators below use cos/sin of the single-angle difference between
    # neighbors, from d|P_l1 - P|^2/d|P| = 2|P| - 2|P_l1|*cos(angle(P_l1) - angle(P)).
    gradout = np.zeros(imarr.shape)
    if pol_solve[0] != 0:
        # dS/dI numerators (chain through |P| = I*m)
        m1 = 2*np.abs(im*im) - np.abs(im*im_l1)*np.cos(np.angle(im_l1) - np.angle(im)) - np.abs(im*im_l2)*np.cos(np.angle(im_l2) - np.angle(im))
        m2 = np.abs(im*im) - np.abs(im*im_r1)*np.cos(np.angle(im) - np.angle(im_r1))
        m3 = np.abs(im*im) - np.abs(im*im_r2)*np.cos(np.angle(im) - np.angle(im_r2))
        gradout[0] = (1./iimage) * (m1/d1 + m2/d2 + m3/d3).flatten()
    if pol_solve[1] != 0:
        # dS/dm numerators; m enters via |P| = I*m, so dS/dm = I * dS/d|P|.
        # Then dm/dprimitive[1] = cos(psi) on the (m, psi) -> (m, v) chart.
        m1 = 2*np.abs(im) - np.abs(im_l1)*np.cos(np.angle(im_l1) - np.angle(im)) - np.abs(im_l2)*np.cos(np.angle(im_l2) - np.angle(im))
        m2 = np.abs(im) - np.abs(im_r1)*np.cos(np.angle(im) - np.angle(im_r1))
        m3 = np.abs(im) - np.abs(im_r2)*np.cos(np.angle(im) - np.angle(im_r2))
        gradm = iimage * (m1/d1 + m2/d2 + m3/d3).flatten()
        gradout[1] = gradm * np.cos(psiimage)
    if pol_solve[2] != 0:
        # dS/dphi numerators (primitive is phi = 2*chi, see PR #240).
        # `gradchi` is dS/dchi; chain through chi(phi) = phi/2 gives the *0.5.
        c1 = -2*np.abs(im*im_l1)*np.sin(np.angle(im_l1) - np.angle(im)) - 2*np.abs(im*im_l2)*np.sin(np.angle(im_l2) - np.angle(im))
        c2 = 2*np.abs(im*im_r1)*np.sin(np.angle(im) - np.angle(im_r1))
        c3 = 2*np.abs(im*im_r2)*np.sin(np.angle(im) - np.angle(im_r2))
        gradchi = (c1/d1 + c2/d2 + c3/d3).flatten()
        gradout[2] = 0.5 * gradchi
    if pol_solve[3] != 0:
        # dS/dpsi numerators; reuse dS/dm and chain through dm/dpsi = -m*tan(psi).
        m1 = 2*np.abs(im) - np.abs(im_l1)*np.cos(np.angle(im_l1) - np.angle(im)) - np.abs(im_l2)*np.cos(np.angle(im_l2) - np.angle(im))
        m2 = np.abs(im) - np.abs(im_r1)*np.cos(np.angle(im) - np.angle(im_r1))
        m3 = np.abs(im) - np.abs(im_r2)*np.cos(np.angle(im) - np.angle(im_r2))
        gradm = iimage * (m1/d1 + m2/d2 + m3/d3).flatten()
        gradout[3] = gradm * (-mimage * np.tan(psiimage))
    g = gradout / norm
    return g[:, mask] if do_slice else g


# --- circular-polarization (V) regularizers ---------------------------------
# TODO check!! The V-pol regularizers below were never thoroughly audited; the
# `gradv * (vfimage / np.tan(psiimage))` chain-rule form (dS/dpsi) is unusual
# and the test coverage only exercises generic random pol structure.
def reg_vflux(imarr, mask, **kwargs):
    xp = array_namespace(imarr)
    vflux = kwargs['vflux']
    norm = np.abs(vflux)**2 if kwargs.get('norm_reg', True) else 1
    vimage = make_v_image(imarr)
    return (xp.sum(vimage) - vflux)**2 / norm


def reggrad_vflux(imarr, mask, **kwargs):
    vflux = kwargs['vflux']
    pol_solve = kwargs.get('pol_solve', POL_SOLVE_DEFAULT_V)
    norm = np.abs(vflux)**2 if kwargs.get('norm_reg', True) else 1
    iimage = make_i_image(imarr)
    vimage = make_v_image(imarr)
    vfimage = make_vf_image(imarr)
    psiimage = make_psi_image(imarr)
    # base = dS/dV via V = I * vf, vf = m*sin(psi); pol_solve branches below
    # chain through dV/dI = V/I, dV/dm = I*sin(psi), dV/dpsi = I*m*cos(psi).
    base = 2 * (np.sum(vimage) - vflux) * np.ones(len(vimage))
    gradout = np.zeros(imarr.shape)
    if pol_solve[0] != 0:
        gradout[0] = (vimage / iimage) * base  # dS/dI
    if pol_solve[1] != 0:
        gradv = iimage * base
        gradout[1] = gradv * np.sin(psiimage)  # dS/dm
    if pol_solve[3] != 0:
        gradv = iimage * base
        gradout[3] = gradv * (vfimage / np.tan(psiimage))  # dS/dpsi; vf/tan(psi) = m*cos(psi)
    return gradout / norm


def reg_l1v(imarr, mask, **kwargs):
    xp = array_namespace(imarr)
    vflux = kwargs['vflux']
    norm = np.abs(vflux) if kwargs.get('norm_reg', True) else 1
    vimage = make_v_image(imarr)
    return xp.sum(xp.abs(vimage)) / norm


def reggrad_l1v(imarr, mask, **kwargs):
    vflux = kwargs['vflux']
    pol_solve = kwargs.get('pol_solve', POL_SOLVE_DEFAULT_V)
    norm = np.abs(vflux) if kwargs.get('norm_reg', True) else 1
    iimage = make_i_image(imarr)
    vimage = make_v_image(imarr)
    vfimage = make_vf_image(imarr)
    psiimage = make_psi_image(imarr)
    # base = dS/dV via V = I*vf; chain rules per reggrad_vflux.
    base = np.sign(vimage)
    gradout = np.zeros(imarr.shape)
    if pol_solve[0] != 0:
        gradout[0] = vfimage * base  # dS/dI (vfimage = V/I)
    if pol_solve[1] != 0:
        gradv = iimage * base
        gradout[1] = gradv * np.sin(psiimage)  # dS/dm
    if pol_solve[3] != 0:
        gradv = iimage * base
        gradout[3] = gradv * (vfimage / np.tan(psiimage))  # dS/dpsi
    return gradout / norm


def reg_l2v(imarr, mask, **kwargs):
    xp = array_namespace(imarr)
    vflux = kwargs['vflux']
    norm = np.abs(vflux**2) if kwargs.get('norm_reg', True) else 1
    vimage = make_v_image(imarr)
    return xp.sum(vimage**2) / norm


def reggrad_l2v(imarr, mask, **kwargs):
    vflux = kwargs['vflux']
    pol_solve = kwargs.get('pol_solve', POL_SOLVE_DEFAULT_V)
    norm = np.abs(vflux**2) if kwargs.get('norm_reg', True) else 1
    iimage = make_i_image(imarr)
    vimage = make_v_image(imarr)
    vfimage = make_vf_image(imarr)
    psiimage = make_psi_image(imarr)
    # base = dS/dV via V = I*vf; chain rules per reggrad_vflux.
    base = 2 * vimage
    gradout = np.zeros(imarr.shape)
    if pol_solve[0] != 0:
        gradout[0] = vfimage * base  # dS/dI
    if pol_solve[1] != 0:
        gradv = iimage * base
        gradout[1] = gradv * np.sin(psiimage)  # dS/dm
    if pol_solve[3] != 0:
        gradv = iimage * base
        gradout[3] = gradv * (vfimage / np.tan(psiimage))  # dS/dpsi
    return gradout / norm


def reg_vtv(imarr, mask, **kwargs):
    from ehtim.imaging.imager_utils import embed_imarr
    xp = array_namespace(imarr)
    if np.any(np.invert(mask)):
        imarr = embed_imarr(imarr, mask, randomfloor=True)
    vflux = kwargs['vflux']
    nx, ny, psize = kwargs['xdim'], kwargs['ydim'], kwargs['psize']
    beam_size = kwargs.get('beam_size', 1) or psize
    norm = np.abs(vflux) * psize / beam_size if kwargs.get('norm_reg', True) else 1
    vimage = make_v_image(imarr)
    im = vimage.reshape(ny, nx)
    impad = xp.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = xp.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = xp.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    return xp.sum(xp.sqrt(xp.abs(im_l1 - im)**2 + xp.abs(im_l2 - im)**2)) / norm


def reggrad_vtv(imarr, mask, **kwargs):
    from ehtim.imaging.imager_utils import embed_imarr
    do_slice = np.any(np.invert(mask))
    if do_slice:
        imarr = embed_imarr(imarr, mask, randomfloor=True)
    vflux = kwargs['vflux']
    nx, ny, psize = kwargs['xdim'], kwargs['ydim'], kwargs['psize']
    pol_solve = kwargs.get('pol_solve', POL_SOLVE_DEFAULT_V)
    beam_size = kwargs.get('beam_size', 1) or psize
    norm = np.abs(vflux) * psize / beam_size if kwargs.get('norm_reg', True) else 1
    epsilon = 0.
    iimage = make_i_image(imarr)
    vimage = make_v_image(imarr)
    vfimage = make_vf_image(imarr)
    psiimage = make_psi_image(imarr)
    im = vimage.reshape(ny, nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    im_r1 = np.roll(impad, 1, axis=0)[1:ny+1, 1:nx+1]
    im_r2 = np.roll(impad, 1, axis=1)[1:ny+1, 1:nx+1]
    im_r1l2 = np.roll(np.roll(impad,  1, axis=0), -1, axis=1)[1:ny+1, 1:nx+1]
    im_l1r2 = np.roll(np.roll(impad, -1, axis=0),  1, axis=1)[1:ny+1, 1:nx+1]
    g1 = (2*im - im_l1 - im_l2) / np.sqrt((im - im_l1)**2 + (im - im_l2)**2 + epsilon)
    g2 = (im - im_r1) / np.sqrt((im - im_r1)**2 + (im_r1l2 - im_r1)**2 + epsilon)
    g3 = (im - im_r2) / np.sqrt((im - im_r2)**2 + (im_l1r2 - im_r2)**2 + epsilon)
    mask1 = np.zeros(im.shape)
    mask2 = np.zeros(im.shape)
    mask1[0, :] = 1
    mask2[:, 0] = 1
    g2[mask1.astype(bool)] = 0
    g3[mask2.astype(bool)] = 0
    # base = dS/dV via V = I*vf; chain rules per reggrad_vflux.
    base = (g1 + g2 + g3).flatten()
    gradout = np.zeros(imarr.shape)
    if pol_solve[0] != 0:
        gradout[0] = vfimage * base  # dS/dI
    if pol_solve[1] != 0:
        gradv = iimage * base
        gradout[1] = gradv * np.sin(psiimage)  # dS/dm
    if pol_solve[3] != 0:
        gradv = iimage * base
        gradout[3] = gradv * (vfimage / np.tan(psiimage))  # dS/dpsi
    g = gradout / norm
    return g[:, mask] if do_slice else g


def reg_vtv2(imarr, mask, **kwargs):
    from ehtim.imaging.imager_utils import embed_imarr
    xp = array_namespace(imarr)
    if np.any(np.invert(mask)):
        imarr = embed_imarr(imarr, mask, randomfloor=True)
    vflux = kwargs['vflux']
    nx, ny, psize = kwargs['xdim'], kwargs['ydim'], kwargs['psize']
    beam_size = kwargs.get('beam_size', 1) or psize
    norm = psize**4 * np.abs(vflux**2) / beam_size**4 if kwargs.get('norm_reg', True) else 1
    vimage = make_v_image(imarr)
    im = vimage.reshape(ny, nx)
    impad = xp.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = xp.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = xp.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    return xp.sum((im_l1 - im)**2 + (im_l2 - im)**2) / norm


def reggrad_vtv2(imarr, mask, **kwargs):
    from ehtim.imaging.imager_utils import embed_imarr
    do_slice = np.any(np.invert(mask))
    if do_slice:
        imarr = embed_imarr(imarr, mask, randomfloor=True)
    vflux = kwargs['vflux']
    nx, ny, psize = kwargs['xdim'], kwargs['ydim'], kwargs['psize']
    pol_solve = kwargs.get('pol_solve', POL_SOLVE_DEFAULT_V)
    beam_size = kwargs.get('beam_size', 1) or psize
    norm = psize**4 * np.abs(vflux**2) / beam_size**4 if kwargs.get('norm_reg', True) else 1
    iimage = make_i_image(imarr)
    vimage = make_v_image(imarr)
    vfimage = make_vf_image(imarr)
    psiimage = make_psi_image(imarr)
    im = vimage.reshape(ny, nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    im_r1 = np.roll(impad, 1, axis=0)[1:ny+1, 1:nx+1]
    im_r2 = np.roll(impad, 1, axis=1)[1:ny+1, 1:nx+1]
    g1 = 2*im - im_l1 - im_l2
    g2 = im - im_r1
    g3 = im - im_r2
    mask1 = np.zeros(im.shape)
    mask2 = np.zeros(im.shape)
    mask1[0, :] = 1
    mask2[:, 0] = 1
    g2[mask1.astype(bool)] = 0
    g3[mask2.astype(bool)] = 0
    # base = dS/dV via V = I*vf; chain rules per reggrad_vflux.
    base = 2 * (g1 + g2 + g3).flatten()
    gradout = np.zeros(imarr.shape)
    if pol_solve[0] != 0:
        gradout[0] = vfimage * base  # dS/dI
    if pol_solve[1] != 0:
        gradv = iimage * base
        gradout[1] = gradv * np.sin(psiimage)  # dS/dm
    if pol_solve[3] != 0:
        gradv = iimage * base
        gradout[3] = gradv * (vfimage / np.tan(psiimage))  # dS/dpsi
    g = gradout / norm
    return g[:, mask] if do_slice else g


##################################################################################################
# Chi^2 Data functions
##################################################################################################
def chisqdata_pvis(Obsdata, Prior, mask, **kwargs):
    """Return the visibilities, sigmas, and fourier matrix for an observation, prior, mask.

    Accepts and ignores standard-chisqdata kwargs (pol, snrcut, debias, etc.) so the
    unified compute_chisqdata_term dispatcher can pass them uniformly across all dtypes.
    """

    data_arr = Obsdata.unpack(['u','v','pvis','psigma'], conj=True)
    uv = np.hstack((data_arr['u'].reshape(-1,1), data_arr['v'].reshape(-1,1)))
    vis = data_arr['pvis']
    sigma = data_arr['psigma']
    A = ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv, pulse=Prior.pulse, mask=mask)

    return (vis, sigma, A)

def chisqdata_pvis_nfft(Obsdata, Prior, **kwargs):
    """Return the visibilities, sigmas, and fourier matrix for an observation."""

    # unpack keyword args
    fft_pad_factor = kwargs.get('fft_pad_factor',FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', GRIDDER_P_RAD_DEFAULT)
    nfft_eps = kwargs.get('nfft_eps', NFFT_EPS_DEFAULT)

    # unpack data
    data_arr = Obsdata.unpack(['u','v','pvis','psigma'], conj=True)
    uv = np.hstack((data_arr['u'].reshape(-1,1), data_arr['v'].reshape(-1,1)))
    vis = data_arr['pvis']
    sigma = data_arr['psigma']

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv, eps=nfft_eps)
    A = [A1]

    return (vis, sigma, A)


def chisqdata_m(Obsdata, Prior, mask, **kwargs):
    """Return the pol ratios, sigmas, and fourier matrix for an observation, prior, mask.

    Accepts and ignores standard-chisqdata kwargs (pol, snrcut, debias, etc.) so the
    unified compute_chisqdata_term dispatcher can pass them uniformly across all dtypes.
    """

    mdata = Obsdata.unpack(['u','v','m','msigma'], conj=True)
    uv = np.hstack((mdata['u'].reshape(-1,1), mdata['v'].reshape(-1,1)))
    m = mdata['m']
    sigmam = mdata['msigma']
    A = ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv, pulse=Prior.pulse, mask=mask)

    return (m, sigmam, A)

def chisqdata_m_nfft(Obsdata, Prior, **kwargs):
    """Return the pol ratios, sigmas, and fourier matrix for an observation."""

    # unpack keyword args
    fft_pad_factor = kwargs.get('fft_pad_factor',FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', GRIDDER_P_RAD_DEFAULT)
    nfft_eps = kwargs.get('nfft_eps', NFFT_EPS_DEFAULT)

    # unpack data
    mdata = Obsdata.unpack(['u','v','m','msigma'], conj=True)
    uv = np.hstack((mdata['u'].reshape(-1,1), mdata['v'].reshape(-1,1)))
    m = mdata['m']
    sigmam = mdata['msigma']

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv, eps=nfft_eps)
    A = [A1]

    return (m, sigmam, A)


def chisqdata_vvis(Obsdata, Prior, mask, **kwargs):
    """Return the visibilities, sigmas, and fourier matrix for an observation, prior, mask.

    Accepts and ignores standard-chisqdata kwargs (pol, snrcut, debias, etc.) so the
    unified compute_chisqdata_term dispatcher can pass them uniformly across all dtypes.
    """

    data_arr = Obsdata.unpack(['u','v','vvis','vsigma'], conj=False)
    uv = np.hstack((data_arr['u'].reshape(-1,1), data_arr['v'].reshape(-1,1)))
    vis = data_arr['vvis']
    sigma = data_arr['vsigma']
    A = ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv, pulse=Prior.pulse, mask=mask)

    return (vis, sigma, A)

def chisqdata_vvis_nfft(Obsdata, Prior, **kwargs):
    """Return the visibilities, sigmas, and fourier matrix for an observation."""

    # unpack keyword args
    fft_pad_factor = kwargs.get('fft_pad_factor',FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', GRIDDER_P_RAD_DEFAULT)
    nfft_eps = kwargs.get('nfft_eps', NFFT_EPS_DEFAULT)

    # unpack data
    data_arr = Obsdata.unpack(['u','v','vvis','vsigma'], conj=False)
    uv = np.hstack((data_arr['u'].reshape(-1,1), data_arr['v'].reshape(-1,1)))
    vis = data_arr['vvis']
    sigma = data_arr['vsigma']

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv, eps=nfft_eps)
    A = [A1]

    return (vis, sigma, A)

##################################################################################################
# Plotting
##################################################################################################

def plot_m(polarr, Prior, nit, chi2_dict, **kwargs):

    cmap = kwargs.get('cmap','afmhot')
    interpolation = kwargs.get('interpolation', 'gaussian')
    pcut = kwargs.get('pcut', 0.05)
    nvec = kwargs.get('nvec', 15)
    scale = kwargs.get('scale',None)
    dynamic_range = kwargs.get('dynamic_range',1.e5)
    gamma = kwargs.get('dynamic_range',.5)

    plt.ion()
    plt.pause(1.e-6)
    plt.clf()

    # unpack
    im = make_i_image(polarr)
    mim = make_m_image(polarr)
    chiim = make_chi_image(polarr)
    imarr = im.reshape(Prior.ydim,Prior.xdim)

    if scale=='log':
        if (imarr < 0.0).any():
            print('clipping values less than 0')
            imarr[imarr<0.0] = 0.0
        imarr = np.log(imarr + np.max(imarr)/dynamic_range)
        #unit = 'log(' + cbar_unit[0] + ' per ' + cbar_unit[1] + ')'

    if scale=='gamma':
        if (imarr < 0.0).any():
            print('clipping values less than 0')
            imarr[imarr<0.0] = 0.0
        imarr = (imarr + np.max(imarr)/dynamic_range)**(gamma)
        #unit = '(' + cbar_unit[0] + ' per ' + cbar_unit[1] + ')^gamma'

    # Mask for low flux points
    thin = int(round(Prior.xdim/nvec))
    mask = imarr > pcut * np.max(im)
    mask2 = mask[::thin, ::thin]

    # Get vectors and ratio from current image
    x = np.array([[i for i in range(Prior.xdim)] for j in range(Prior.ydim)])[::thin, ::thin][mask2]
    y = np.array([[j for i in range(Prior.xdim)] for j in range(Prior.ydim)])[::thin, ::thin][mask2]
    q = im * mim * np.cos(2*chiim)
    u = im * mim * np.sin(2*chiim)
    a = -np.sin(np.angle(q+1j*u)/2).reshape(Prior.ydim, Prior.xdim)[::thin, ::thin][mask2]
    b = np.cos(np.angle(q+1j*u)/2).reshape(Prior.ydim, Prior.xdim)[::thin, ::thin][mask2]
    m = (np.abs(q + 1j*u)/im).reshape(Prior.ydim, Prior.xdim)
    m[~mask] = 0

    # Stokes I plot
    plt.subplot(121)
    plt.imshow(imarr, cmap=plt.get_cmap('afmhot'), interpolation='gaussian')
    plt.quiver(x, y, a, b,
               headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
               width=.01*Prior.xdim, units='x', pivot='mid', color='k', angles='uv', scale=1.0/thin)
    plt.quiver(x, y, a, b,
               headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
               width=.005*Prior.xdim, units='x', pivot='mid', color='w', angles='uv', scale=1.1/thin)

    xticks = ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
    yticks = ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel(r'Relative RA ($\mu$as)')
    plt.ylabel(r'Relative Dec ($\mu$as)')
    plt.title('Stokes I')

    # Ratio plot
    plt.subplot(122)
    plt.imshow(m, cmap=plt.get_cmap('winter'), interpolation='gaussian', vmin=0, vmax=1)
    plt.quiver(x, y, a, b,
               headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
               width=.01*Prior.xdim, units='x', pivot='mid', color='k', angles='uv', scale=1.0/thin)
    plt.quiver(x, y, a, b,
               headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
               width=.005*Prior.xdim, units='x', pivot='mid', color='w', angles='uv', scale=1.1/thin)

    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel(r'Relative RA ($\mu$as)')
    plt.ylabel(r'Relative Dec ($\mu$as)')
    plt.title(f'm (above {int(pcut*100)} % max flux)')

    # Create title
    plotstr = f"step: {nit}  "
    for key in chi2_dict.keys():
        plotstr += rf"$\chi^2_{{{key}}}$: {chi2_dict[key]:0.2f}  "
    plt.suptitle(plotstr, fontsize=18)


