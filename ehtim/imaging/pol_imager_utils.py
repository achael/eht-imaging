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
#    This program is distributed in the hope that it will be useful,s
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import range

import string
import time
import copy
import numpy as np
import scipy.optimize as opt
import scipy.ndimage as ndF
import scipy.ndimage.filters as filt
import matplotlib.pyplot as plt
try:
    from pynfft.nfft import NFFT
except ImportError:
    pass
    #print("Warning: No NFFT installed! Cannot use nfft functions")

import ehtim.image as image
from ehtim.const_def import *
from ehtim.observing.obs_helpers import *
from ehtim.imaging.imager_utils import embed, embed_arr

TANWIDTH_M = 0.5
TANWIDTH_V = 1
POL_SOLVE_DEFAULT = (0,1,1,0)
POL_SOLVE_DEFAULT_V = (0,0,0,1)
RANDOMFLOOR=True
##################################################################################################
# Constants & Definitions
##################################################################################################

NORM_REGULARIZER = True
DATATERMS_POL = ['pvis', 'm','vvis']
REGULARIZERS_POL = ['msimple', 'hw', 'ptv','l1v','l2v','vtv','v2tv2','vflux']

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
    Pimage = imarr[0] * imarr[1] * np.exp(1j*imarr[2]) * np.cos(imarr[3])

    return pimage

def make_m_image(imarr):
    """construct a polarimetric ratrio image abs(P/I) = abs(Q + iU)/I
    """

    mimage = imarr[1] * np.cos(imarr[3])

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
    qimage = imarr[0] * imarr[1] * np.cos(imarr[2]) *  np.cos(imarr[3])

    return qimage

def make_u_image(imarr):
    """construct an image of stokes U
    """
    # NOTE! We replaced EVPA chi with phi=2chi in the imarr
    uimage = imarr[0] * imarr[1] * np.sin(imarr[2]) *  np.cos(imarr[3])
    
    return uimage

def make_v_image(imarr):
    """construct an image of stokes V
    """

    vimage = imarr[0] * imarr[1] * np.sin(imarr[3])
        
    return vimage

def make_vfrac_image(imarr):
    """construct an image of stokes V/I
    """
    
    vimage = imarr[1] * np.sin(imarr[3])
        
    return vfimage
   
def pcv(imarr):
    """change of variables rho(rho') from range (-inf, inf) to (0,1)
    """
    
    rho_prime =  imarr[1]
    rho = 0.5 + np.arctan(rho_prime/TANWIDTH_M)/np.pi
    out = np.array((imarr[0], rho, imarr[2], imarr[3]))
    return out
    
def vcv(imarr):
    """change of variables for v(v') from range (-inf,inf) to (-1+m,1-m) keeping m=P/I fixed"""
    mfrac = imarr[1] # when using this transform, imarr[2] is mfrac=\rho cos(\psi), not \rho
    vfrac_max = np.abs(1-mfrac)
    
    vfrac_prime = imarr[3]
    vfrac = 2*vfrac_max*np.arctan(vfrac_prime/TANWIDTH_V)/np.pi    
    
    rho = np.sqrt(mfrac**2 + vfrac**2)
    psi = np.arcsin(vfrac/rho)
    
    out = np.array((imarr[0], rho, imarr[2], psi))
    return out

def pcv_r(imarr):
    """Reverse change of variables rho'(rho) from range (0,1) to (-inf, inf)
    """
    
    rho = imarr[1]
    rho_prime = TANWIDTH_M*np.tan(np.pi*(rho - 0.5))
    out = np.array((imarr[0], rho_prime, imarr[2], imarr[3]))
    return out
    
def vcv_r(imarr):
    """Reverse change of variables for v'(v) from range(-1+m,1-m) to (-inf,inf) keeping m=P/I fixed"""
    rho = imarr[1]
    psi = imarr[3]
    mfrac = rho*np.cos(psi)
    vfrac = rho*np.sin(psi)
    vfrac_max = np.abs(1-mfrac)
    
    vfrac_prime = TANWIDTH_V*np.tan(np.pi*vfrac/(2*vfrac_max))
    out = np.array((imarr[0], mfrac, imarr[2], vfrac_prime))
    
    return out

def pcv_chain(imarr):
    """chain rule term drho/drho' for pcv
    """
    out = np.ones(imarr.shape)
    
    rho_prime = imarr[1]
    out[1] = 1 / (TANWIDTH_M*np.pi*(1 + (rho_prime/TANWIDTH_M)**2))
    return out

def vcv_chain(imarr):
    """chain rule terms drho/dv' and dpsi/dv' for vcv
    """
    out = np.ones(imarr.shape)
    
    mfrac = imarr[1] # when using this transform, imarr[2] is mfrac=\rho cos(\psi), not \rho
    vfrac_max = np.abs(1-mfrac)

    vfrac_prime = imarr[3]    
    vfrac = 2*vfrac_max*np.arctan(vfrac_prime/TANWIDTH_V)/np.pi        
    rho = np.sqrt(mfrac**2 + vfrac**2)
    psi = np.arcsin(vfrac/rho)
    
    dv_dvprime = 2. * vfrac_max / (TANWIDTH_V*np.pi*(1 + (vfrac_prime/TANWIDTH_V)**2))
    
    drho_dv = vfrac / rho
    drho_dvprime = drho_dv * dv_dvprime
    out[1] = drho_dvprime
    
    dpsi_dv = mfrac / rho
    dpsi_dvprime = dpsi_dv * dv_dvprime
    out[3] = dpsi_dvprime       
    return out


##################################################################################################
# Wrapper Functions
##################################################################################################

def polchisq(imarr, A, data, sigma, dtype, ttype='direct', mask=[]):
    """return the chi^2 for the appropriate dtype
    """

    chisq = 1 
    if not dtype in DATATERMS_POL:
        return chisq
    if ttype not in ['direct','nfft']:
        raise Exception("Possible ttype values for polchisq are 'nfft' and 'direct'!")
        
    if ttype == 'direct':
        # linear
        if dtype == 'pvis':
            chisq = chisq_p(imarr, A, data, sigma)
        elif dtype == 'm':
            chisq = chisq_m(imarr, A, data, sigma)
            
        # circular
        elif dtype == 'vvis':
            chisq = chisq_vvis(imarr, A, data, sigma)        
            
    elif ttype== 'fast':
        raise Exception("FFT not yet implemented in polchisq!")

    elif ttype== 'nfft':
        if len(mask)>0 and np.any(np.invert(mask)):
            imarr = embed_arr(imarr, mask, randomfloor=RANDOMFLOOR)

        # linear
        if dtype == 'pvis':
            chisq = chisq_p_nfft(imarr, A, data, sigma)
        elif dtype == 'm':
            chisq = chisq_m_nfft(imarr, A, data, sigma)
            
        # circular
        elif dtype == 'vvis':
            chisq = chisq_vvis_nfft(imarr, A, data, sigma)        

    return chisq

def polchisqgrad(imarr, A, data, sigma, dtype, ttype='direct',mask=[],
                 pol_solve=POL_SOLVE_DEFAULT):
    
    """return the chi^2 gradient for the appropriate dtype
    """

    chisqgrad = np.zeros(imarr.shape)
    if not dtype in DATATERMS_POL:
        return chisqgrad
    if ttype not in ['direct','nfft']:
        raise Exception("Possible ttype values for polchisqgrad are 'nfft' and 'direct'!")

    if ttype == 'direct':
        # linear
        if dtype == 'pvis':
            chisqgrad = chisqgrad_p(imarr, A, data, sigma, pol_solve)
        elif dtype == 'm':
            chisqgrad = chisqgrad_m(imarr, A, data, sigma, pol_solve)

            
        # circular
        elif dtype == 'vvis':
            chisqgrad = chisqgrad_vvis(imarr, A, data, sigma, pol_solve)
                
    elif ttype== 'fast':
        raise Exception("FFT not yet implemented in polchisqgrad!")

    elif ttype== 'nfft':
        if len(mask)>0 and np.any(np.invert(mask)):
            imarr = embed_arr(imarr, mask, randomfloor=RANDOMFLOOR)
      
        # linear
        if dtype == 'pvis':
            chisqgrad = chisqgrad_p_nfft(imarr, A, data, sigma, pol_solve)
        elif dtype == 'm':
            chisqgrad = chisqgrad_m_nfft(imarr, A, data, sigma, pol_solve)


        # circular
        elif dtype == 'vvis':
            chisqgrad = chisqgrad_vvis_nfft(imarr, A, data, sigma, pol_solve)
            
        if len(mask)>0 and np.any(np.invert(mask)):
            chisqgrad = chisqgrad[:,mask]

    return chisqgrad


def polregularizer(imarr, mask, flux, pflux, vflux, xdim, ydim, psize, stype, **kwargs):

    norm_reg = kwargs.get('norm_reg', NORM_REGULARIZER)
    beam_size = kwargs.get('beam_size',1)
    
    reg = 0
    if not stype in REGULARIZERS_POL:
        return reg
            
    # linear
    if stype == "msimple":
        reg = -sm(imarr, flux, norm_reg=norm_reg)
    elif stype == "hw":
        reg = -shw(imarr, flux, norm_reg=norm_reg)
    elif stype == "ptv":
        if np.any(np.invert(mask)):
            imarr = embed_arr(imarr, mask, randomfloor=RANDOMFLOOR)
        reg = -stv_pol(imarr, flux, xdim, ydim, psize, 
                       norm_reg=norm_reg, beam_size=beam_size)
    # circular
    elif stype == 'vflux':
        reg = -svflux(imarr, vflux, norm_reg=norm_reg)    
    elif stype == "l1v":
        reg = -sl1v(imarr, vflux, norm_reg=norm_reg)
    elif stype == "l2v":
        reg = -sl2v(imarr, vflux, norm_reg=norm_reg)
    elif stype == "vtv":
        if np.any(np.invert(mask)):
            imarr = embed_arr(imarr, mask, randomfloor=RANDOMFLOOR)
        reg = -stv_v(imarr, vflux, xdim, ydim, psize, 
                     norm_reg=norm_reg, beam_size=beam_size)
    elif stype == "vtv2":
        if np.any(np.invert(mask)):
            imarr = embed_arr(imarr, mask, randomfloor=RANDOMFLOOR)
        reg = -stv2_v(imarr, vflux, xdim, ydim, psize, 
                      norm_reg=norm_reg, beam_size=beam_size)                               


    return reg

def polregularizergrad(imarr, mask, flux, pflux, vflux, xdim, ydim, psize, stype, **kwargs):

    norm_reg = kwargs.get('norm_reg', NORM_REGULARIZER)
    beam_size = kwargs.get('beam_size',1)
    pol_solve = kwargs.get('pol_solve', POL_SOLVE_DEFAULT)
    reggrad = np.zeros(imarr.shape)
    
    if not stype in REGULARIZERS_POL:
        return reg

    # linear
    if stype == "msimple":
        reggrad = -smgrad(imarr, flux, pol_solve, norm_reg=norm_reg)
    elif stype == "hw":
        reggrad = -shwgrad(imarr, flux, pol_solve, norm_reg=norm_reg)
    elif stype == "ptv":
        if np.any(np.invert(mask)):
            imarr = embed_arr(imarr, mask, randomfloor=RANDOMFLOOR)
        reggrad = -stv_pol_grad(imarr, flux, xdim, ydim, psize, pol_solve,
                                norm_reg=norm_reg, beam_size=beam_size)
        if np.any(np.invert(mask)):
            reggrad = reggrad[:,mask]


    # circular
    elif stype == 'vflux':
        reggrad = -svfluxgrad(imarr, vflux, pol_solve=pol_solve, norm_reg=norm_reg)        
    elif stype == "l1v":
        reggrad = -sl1vgrad(imarr, vflux, pol_solve=pol_solve, norm_reg=norm_reg)
    elif stype == "l2v":
        reggrad = -sl2vgrad(imarr, vflux, pol_solve=pol_solve, norm_reg=norm_reg)    
    elif stype == "vtv":
        if np.any(np.invert(mask)):
            imarr = embed_arr(imarr, mask, randomfloor=RANDOMFLOOR)
        reggrad = -stv_v_grad(imarr, vflux, xdim, ydim, psize,
                              pol_solve=pol_solve, norm_reg=norm_reg, beam_size=beam_size)
        if np.any(np.invert(mask)):
            reggrad = reggrad[:,mask]
            
    elif stype == "vtv2":
        if np.any(np.invert(mask)):
            imarr = embed_arr(imarr, mask, randomfloor=RANDOMFLOOR)
        reggrad = -stv2_v_grad(imarr, vflux, xdim, ydim, psize, 
                               pol_solve=pol_solve, norm_reg=norm_reg, beam_size=beam_size)
        if np.any(np.invert(mask)):
            reggrad = reggrad[:,mask]

    return reggrad


def polchisqdata(Obsdata, Prior, mask, dtype, **kwargs):

    """Return the data, sigma, and matrices for the appropriate dtype
    """

    ttype = kwargs.get('ttype','direct')

    (data, sigma, A) = (False, False, False)
    if ttype not in ['direct','nfft']:
        raise Exception("Possible ttype values for polchisqdata are 'nfft' and 'direct'!")
        
    if ttype=='direct':
        if dtype == 'pvis':
            (data, sigma, A) = chisqdata_pvis(Obsdata, Prior, mask)
        elif dtype == 'm':
            (data, sigma, A) = chisqdata_m(Obsdata, Prior, mask)
        elif dtype == 'vvis':
            (data, sigma, A) = chisqdata_vvis(Obsdata, Prior, mask)
            
    elif ttype=='fast':
        raise Exception("FFT not yet implemented in polchisqdata!")

    elif ttype=='nfft':
        if dtype == 'pvis':
            (data, sigma, A) = chisqdata_pvis_nfft(Obsdata, Prior, mask, **kwargs)
        elif dtype == 'm':
            (data, sigma, A) = chisqdata_m_nfft(Obsdata, Prior, mask, **kwargs)
        elif dtype == 'vvis':
            (data, sigma, A) = chisqdata_vvis_nfft(Obsdata, Prior, mask, **kwargs)
        
    return (data, sigma, A)


##################################################################################################
# DFT Chi-squared and Gradient Functions
##################################################################################################

def chisq_p(imarr, Amatrix, p, sigmap):
    """Polarimetric visibility chi-squared
    """

    pimage = make_p_image(imarr)
    psamples = np.dot(Amatrix, pimage) 
    chisq =  np.sum(np.abs((p - psamples))**2/(sigmap**2)) / (2*len(p))   
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

    iimage = make_i_image(imarr)
    pimage = make_p_image(imarr)
    msamples = np.dot(Amatrix, pimage) / np.dot(Amatrix, iimage)
    chisq = np.sum(np.abs((m - msamples))**2/(sigmam**2)) / (2*len(m))   
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
        gradi = (-np.real(   * np.dot(Amatrix.conj().T, mdiff)) / len(m) + 
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

    vimage = make_v_image(imarr)
    vsamples = np.dot(Amatrix, vimage) 
    chisq =  np.sum(np.abs((v - vsamples))**2/(sigmav**2)) / (2*len(v))   
    return chisq

def chisqgrad_vvis(imarr, Amatrix, v, sigmap,pol_solve=POL_SOLVE_DEFAULT_V):
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
    chisq =  np.sum(np.abs((p - psamples))**2/(sigmap**2)) / (2*len(p))   

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
    chisq = np.sum(np.abs((m - msamples))**2/(sigmam**2)) / (2*len(m))   
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
    chisq =  np.sum(np.abs((v - vsamples))**2/(sigmav**2)) / (2*len(v))   

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
# Polarimetric Entropy and Gradient Functions
##################################################################################################

def sm(imarr, flux, norm_reg=NORM_REGULARIZER):
    """I log m entropy
    """
    if norm_reg: norm = flux
    else: norm = 1

    iimage = make_i_image(imarr)
    mimage = make_m_image(imarr) 
    S = -np.sum(iimage * np.log(mimage))
    return S/norm

def smgrad(imarr, flux, pol_solve=POL_SOLVE_DEFAULT,
           norm_reg=NORM_REGULARIZER):
    """I log m entropy gradient
    """
    gradout = np.zeros(imarr.shape)
    
    if norm_reg: norm = flux
    else: norm = 1

    iimage = make_i_image(imarr)
    mimage = make_m_image(imarr) 
    psiimage = make_psi_image(imarr)

    if pol_solve[0]!=0:
        gradi = -np.log(mimage)
        outgrad[0] = gradi
        
    if pol_solve[1]!=0:
        gradm = -iimage / mimage
        gradrho = gradm * np.cos(psiimage)        
        gradout[1] = gradrho
    
    if pol_solve[3]!=0:
        gradm = -iimage / mimage
        gradpsi = gradm * (-mimage*np.tan(psiimage))
        gradout[3] = gradpsi
            
    return gradout/norm
          
def shw(imarr, flux, norm_reg=NORM_REGULARIZER):
    """Holdaway-Wardle polarimetric entropy
    """
    
    if norm_reg: norm = flux
    else: norm = 1

    iimage = make_i_image(imarr)
    mimage = make_m_image(imarr) 
    S = -np.sum(iimage * (((1+mimage)/2) * np.log((1+mimage)/2) + ((1-mimage)/2) * np.log((1-mimage)/2)))
    return S/norm

def shwgrad(imarr, flux,pol_solve=POL_SOLVE_DEFAULT,
            norm_reg=NORM_REGULARIZER):
    """Gradient of the Holdaway-Wardle polarimetric entropy
    """
    gradout = np.zeros(imarr.shape)
    
    if norm_reg: norm = flux
    else: norm = 1

    iimage = make_i_image(imarr)
    mimage = make_m_image(imarr) 
    psiimage = make_psi_image(imarr)

    if pol_solve[0]!=0:
        gradi =  -(((1+mimage)/2) * np.log((1+mimage)/2) + ((1-mimage)/2) * np.log((1-mimage)/2))
        gradout[0] = gradi
        
    if pol_solve[1]!=0:
        gradm = -iimage * np.arctanh(mimage)
        gradrho = gradm * np.cos(psiimage)        
        gradout[1] = gradrho

    if pol_solve[3]!=0:
        gradm = -iimage * np.arctanh(mimage)
        gradpsi = gradm * (-mimage*np.tan(psiimage))
        gradout[3] = gradpsi     
                
    return gradout/norm

def stv_pol(imarr, flux, nx, ny, psize, 
            norm_reg=NORM_REGULARIZER, beam_size=None):
    """Total variation of I*m*exp(2Ichi)"""
    
    if beam_size is None: beam_size = psize
    if norm_reg: norm = flux*psize / beam_size
    else: norm = 1

    pimage = make_p_image(imarr)
    im = pimage.reshape(ny, nx)

    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    S = -np.sum(np.sqrt(np.abs(im_l1 - im)**2 + np.abs(im_l2 - im)**2))
    return S/norm

def stv_pol_grad(imarr, flux, nx, ny, psize, pol_solve=POL_SOLVE_DEFAULT, 
             norm_reg=NORM_REGULARIZER, beam_size=None):
    """Total variation entropy gradient"""
    gradout = np.zeros(imarr.shape)
    
    if beam_size is None: beam_size = psize
    if norm_reg: norm = flux*psize / beam_size
    else: norm = 1

    iimage = make_i_image(imarr)
    pimage = make_p_image(imarr)
    psiimage = make_psi_image(imarr)

    im = pimage.reshape(ny, nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    im_r1 = np.roll(impad, 1, axis=0)[1:ny+1, 1:nx+1]
    im_r2 = np.roll(impad, 1, axis=1)[1:ny+1, 1:nx+1]
    im_r1l2 = np.roll(np.roll(impad, 1, axis=0), -1, axis=1)[1:ny+1, 1:nx+1]
    im_l1r2 = np.roll(np.roll(impad, 1, axis=0), -1, axis=1)[1:ny+1, 1:nx+1]
    
    # Denominators
    d1 = np.sqrt(np.abs(im_l1 - im)**2 + np.abs(im_l2 - im)**2)
    d2 = np.sqrt(np.abs(im_r1 - im)**2 + np.abs(im_r1l2 - im_r1)**2)
    d3 = np.sqrt(np.abs(im_r2 - im)**2 + np.abs(im_l1r2 - im_r2)**2)

    # dS/dI Numerators
    if pol_solve[0]!=0:
        m1 = 2*np.abs(im*im) - np.abs(im*im_l1)*np.cos(2*(np.angle(im_l1) - np.angle(im))) - np.abs(im*im_l2)*np.cos(2*(np.angle(im_l2) - np.angle(im)))
        m2 = np.abs(im*im) - np.abs(im*im_r1)*np.cos(2*(np.angle(im) - np.angle(im_r1)))
        m3 = np.abs(im*im) - np.abs(im*im_r2)*np.cos(2*(np.angle(im) - np.angle(im_r2)))
        gradi = -(1./iimage)*(m1/d1 + m2/d2 + m3/d3).flatten()
        gradout[0] = gradi

    # dS/dm numerators
    if pol_solve[1]!=0:
        m1 = 2*np.abs(im) - np.abs(im_l1)*np.cos(2*(np.angle(im_l1) - np.angle(im))) - np.abs(im_l2)*np.cos(2*(np.angle(im_l2) - np.angle(im)))
        m2 = np.abs(im) - np.abs(im_r1)*np.cos(2*(np.angle(im) - np.angle(im_r1)))
        m3 = np.abs(im) - np.abs(im_r2)*np.cos(2*(np.angle(im) - np.angle(im_r2)))
        gradm = -iimage*(m1/d1 + m2/d2 + m3/d3).flatten()
        gradrho = gradm * np.cos(psiimage)        
        gradout[1] = gradrho

    # dS/dchi numerators
    if pol_solve[2]!=0:
        c1 = -2*np.abs(im*im_l1)*np.sin(2*(np.angle(im_l1) - np.angle(im))) - 2*np.abs(im*im_l2)*np.sin(2*(np.angle(im_l2) - np.angle(im)))
        c2 = 2*np.abs(im*im_r1)*np.sin(2*(np.angle(im) - np.angle(im_r1)))
        c3 = 2*np.abs(im*im_r2)*np.sin(2*(np.angle(im) - np.angle(im_r2)))
        gradchi = -(c1/d1 + c2/d2 + c3/d3).flatten()
        gradphi = 0.5*gradchi
        gradout[2] = gradphi

    # dS/dpsi
    if pol_solve[3]!=0:
        m1 = 2*np.abs(im) - np.abs(im_l1)*np.cos(2*(np.angle(im_l1) - np.angle(im))) - np.abs(im_l2)*np.cos(2*(np.angle(im_l2) - np.angle(im)))
        m2 = np.abs(im) - np.abs(im_r1)*np.cos(2*(np.angle(im) - np.angle(im_r1)))
        m3 = np.abs(im) - np.abs(im_r2)*np.cos(2*(np.angle(im) - np.angle(im_r2)))
        gradm = -iimage*(m1/d1 + m2/d2 + m3/d3).flatten()
        gradpsi = gradm * (-mimage*np.tan(psiimage))
        gradout[3] = gradpsi     
                
    return out/norm

###############################################################
# circular polarization
# TODO check!!
def svflux(imarr, vflux, norm_reg=NORM_REGULARIZER):
    """Total flux constraint
    """
    if norm_reg: norm = np.abs(vflux)**2
    else: norm = 1
    
    vimage = make_v_image(imarr) 
    
    out = -(np.sum(vimage) - vflux)**2
    return out/norm


def svfluxgrad(imarr, vflux,  pol_solve=POL_SOLVE_DEFAULT_V, norm_reg=NORM_REGULARIZER):
    """Total flux constraint gradient
    """
    gradout = np.zeros(imarr.shape)
    
    if norm_reg: norm = np.abs(vflux)**2
    else: norm = 1
    
    iimage = make_i_image(imarr)  
    vimage = make_v_image(imarr)
    psiimage = make_psi_image(imarr) 
    grad = -2*(np.sum(vimage) - vflux)*np.ones(len(vimage))

        
    # dS/dI Numerators
    if pol_solve[0]!=0:
        gradi = (vimage/iimage)*grad
        gradout[0] = gradi

    # dS/dv numerators
    if pol_solve[1]!=0:
        gradv = iimage*grad
        gradrho = gradv*np.sin(psiimage)
        gradout[1] = gradrho         
            
    if pol_solve[3]!=0:
        gradv = iimage*grad
        gradpsi = gradv * (vfimage/np.tan(psiimage))
        gradout[3] = gradpsi

    return gradoutout/norm
    
def sl1v(imarr, vflux, norm_reg=NORM_REGULARIZER):
    """L1 norm regularizer on V
    """
    if norm_reg: norm = np.abs(vflux)
    else: norm = 1

    vimage = make_v_image(imarr) 
    l1 = -np.sum(np.abs(vimage))
    return l1/norm


def sl1vgrad(imarr, vflux, pol_solve=POL_SOLVE_DEFAULT_V,  norm_reg=NORM_REGULARIZER):
    """L1 norm gradient
    """
    gradout = np.zeros(imarr.shape)
        
    if norm_reg: norm = np.abs(vflux)
    else: norm = 1
     
    iimage = make_i_image(imarr)  
    vimage = make_v_image(imarr)
    psiimage = make_psi_image(imarr) 
    
    grad = -np.sign(vimage)
    
    # dS/dI Numerators
    if pol_solve[0]!=0:
        gradi = (vimage/iimage)*grad
        gradout[0] = gradi

    # dS/dv numerators
    if pol_solve[1]!=0:
        gradv = iimage*grad
        gradrho = gradv*np.sin(psiimage)
        gradout[1] = gradrho
        
    if pol_solve[3]!=0:
        gradv = iimage*grad
        gradpsi = gradv * (vfimage/np.tan(psiimage))
        gradout[3] = gradpsi
    
    return gradout/norm
    
def sl2v(imarr, vflux, norm_reg=NORM_REGULARIZER):
    """L1 norm regularizer on V
    """
    if norm_reg: norm = np.abs(vflux**2)
    else: norm = 1

    vimage = make_v_image(imarr) 
    l2 = -np.sum((vimage)**2)
    return l2/norm


def sl2vgrad(imarr, vflux,pol_solve=POL_SOLVE_DEFAULT_V, norm_reg=NORM_REGULARIZER):
    """L2 norm gradient
    """
    gradout = np.zeros(imarr.shape)
    
    if norm_reg: norm = np.abs(vflux**2)
    else: norm = 1
 
    iimage = make_i_image(imarr)  
    vimage = make_v_image(imarr)
    psiimage = make_psi_image(imarr) 
    
    grad = -2*vimage

    # dS/dI Numerators
    if pol_solve[0]!=0:
        gradi = (vimage/iimage)*grad
        gradout[0] = gradi

    # dS/dv numerators
    if pol_solve[1]!=0:
        gradv = iimage*grad
        gradrho = gradv*np.sin(psiimage)
        gradout[1] = gradrho         

    if pol_solve[3]!=0:
        gradv = iimage*grad
        gradpsi = gradv * (vfimage/np.tan(psiimage))
        gradout[3] = gradpsi

    return gradout/norm    

def stv_v(imarr, vflux, nx, ny, psize, 
          norm_reg=NORM_REGULARIZER, beam_size=None, epsilon=0.):
    """Total variation of I*vfrac"""
    
    if beam_size is None: beam_size = psize
    if norm_reg: norm = np.abs(vflux)*psize / beam_size
    else: norm = 1

    vimage = make_v_image(imarr)
    im = vimage.reshape(ny, nx)

    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    S = -np.sum(np.sqrt(np.abs(im_l1 - im)**2 + np.abs(im_l2 - im)**2+epsilon))
    return S/norm

def stv_v_grad(imarr, vflux, nx, ny, psize, pol_solve=POL_SOLVE_DEFAULT_V, 
               norm_reg=NORM_REGULARIZER, beam_size=None, epsilon=0.):
    """Total variation gradient"""
    gradout = np.zeros(imarr.shape)
    
    if beam_size is None: beam_size = psize
    if norm_reg: norm = np.abs(vflux)*psize / beam_size
    else: norm = 1

    iimage = make_i_image(imarr)
    vimage = make_v_image(imarr)
    psiimage = make_psi_image(imarr) 
    
    im = vimage.reshape(ny, nx)
    
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    im_r1 = np.roll(impad, 1, axis=0)[1:ny+1, 1:nx+1]
    im_r2 = np.roll(impad, 1, axis=1)[1:ny+1, 1:nx+1]

    # rotate images
    im_r1l2 = np.roll(np.roll(impad,  1, axis=0), -1, axis=1)[1:ny+1, 1:nx+1]
    im_l1r2 = np.roll(np.roll(impad, -1, axis=0), 1, axis=1)[1:ny+1, 1:nx+1]

    # add together terms and return
    g1 = (2*im - im_l1 - im_l2) / np.sqrt((im - im_l1)**2 + (im - im_l2)**2 + epsilon)
    g2 = (im - im_r1) / np.sqrt((im - im_r1)**2 + (im_r1l2 - im_r1)**2 + epsilon)
    g3 = (im - im_r2) / np.sqrt((im - im_r2)**2 + (im_l1r2 - im_r2)**2 + epsilon)

    # mask the first row column gradient terms that don't exist
    mask1 = np.zeros(im.shape)
    mask2 = np.zeros(im.shape)
    mask1[0, :] = 1
    mask2[:, 0] = 1
    g2[mask1.astype(bool)] = 0
    g3[mask2.astype(bool)] = 0

    # add terms together and return
    grad = -(g1 + g2 + g3).flatten()
    
    # dS/dI Numerators
    if pol_solve[0]!=0:
        gradi = (vimage/iimage)*grad
        gradout[0] = gradi
     
    # dS/dv numerators
    if pol_solve[1]!=0:
        gradv = iimage*grad
        gradrho = gradv*np.sin(psiimage)
        gradout[1] = gradrho         

    if pol_solve[3]!=0:
        gradv = iimage*grad
        gradpsi = gradv * (vfimage/np.tan(psiimage))
        gradout[3] = gradpsi

    return gradout/norm

def stv2_v(imarr, vflux, nx, ny, psize,
           norm_reg=NORM_REGULARIZER, beam_size=None):
    """Squared Total variation of I*vfrac
    """
    
    if beam_size is None:
        beam_size = psize
    if norm_reg:
        norm = psize**4 * np.abs(vflux**2) / beam_size**4
    else:
        norm = 1

    vimage = make_v_image(imarr)
    im = vimage.reshape(ny, nx)
    
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    out = -np.sum((im_l1 - im)**2 + (im_l2 - im)**2)
    return out/norm

def stv2_v_grad(imarr, vflux, nx, ny, psize, pol_solve=POL_SOLVE_DEFAULT_V, 
               norm_reg=NORM_REGULARIZER, beam_size=None):
    """Squared Total variation gradient
    """
    gradout = np.zeros(imarr.shape)
    if beam_size is None:
        beam_size = psize
    if norm_reg:
        norm = psize**4 * np.abs(vflux**2) / beam_size**4
    else:
        norm = 1

    iimage = make_i_image(imarr)
    vimage = make_v_image(imarr)
    psiimage = make_psi_image(imarr) 
    
    im = vimage.reshape(ny, nx)
    
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    im_r1 = np.roll(impad, 1, axis=0)[1:ny+1, 1:nx+1]
    im_r2 = np.roll(impad, 1, axis=1)[1:ny+1, 1:nx+1]

    g1 = (2*im - im_l1 - im_l2)
    g2 = (im - im_r1)
    g3 = (im - im_r2)

    # mask the first row column gradient terms that don't exist
    mask1 = np.zeros(im.shape)
    mask2 = np.zeros(im.shape)
    mask1[0, :] = 1
    mask2[:, 0] = 1
    g2[mask1.astype(bool)] = 0
    g3[mask2.astype(bool)] = 0

    # add together terms and return
    grad = -2*(g1 + g2 + g3).flatten()
    
    # dS/dI Numerators
    if pol_solve[0]!=0:
        gradi = (vimage/iimage)*grad
        gradout[0] = gradi

    # dS/dv numerators
    if pol_solve[1]!=0:
        gradv = iimage*grad
        gradrho = gradv*np.sin(psiimage)
        gradout[1] = gradrho         
        
    if pol_solve[3]!=0:
        gradv = iimage*grad
        gradpsi = gradv * (vfimage/np.tan(psiimage))
        gradout[3] = gradpsi

    return gradout/norm

##################################################################################################
# Chi^2 Data functions
##################################################################################################
def chisqdata_pvis(Obsdata, Prior, mask):
    """Return the visibilities, sigmas, and fourier matrix for an observation, prior, mask
    """

    data_arr = Obsdata.unpack(['u','v','pvis','psigma'], conj=True)
    uv = np.hstack((data_arr['u'].reshape(-1,1), data_arr['v'].reshape(-1,1)))
    vis = data_arr['pvis']
    sigma = data_arr['psigma']
    A = ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv, pulse=Prior.pulse, mask=mask)

    return (vis, sigma, A)

def chisqdata_pvis_nfft(Obsdata, Prior, mask, **kwargs):
    """Return the visibilities, sigmas, and fourier matrix for an observation, prior, mask
    """

    # unpack keyword args
    fft_pad_factor = kwargs.get('fft_pad_factor',FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', GRIDDER_P_RAD_DEFAULT)

    # unpack data
    data_arr = Obsdata.unpack(['u','v','pvis','psigma'], conj=True)
    uv = np.hstack((data_arr['u'].reshape(-1,1), data_arr['v'].reshape(-1,1)))
    vis = data_arr['pvis']
    sigma = data_arr['psigma']

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv)
    A = [A1]

    return (vis, sigma, A)


def chisqdata_m(Obsdata, Prior, mask):
    """Return the pol  ratios, sigmas, and fourier matrix for and observation, prior, mask
    """

    mdata = Obsdata.unpack(['u','v','m','msigma'], conj=True)
    uv = np.hstack((mdata['u'].reshape(-1,1), mdata['v'].reshape(-1,1)))
    m = mdata['m']
    sigmam = mdata['msigma']
    A = ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv, pulse=Prior.pulse, mask=mask)

    return (m, sigmam, A)

def chisqdata_m_nfft(Obsdata, Prior, mask, **kwargs):
    """Return the pol ratios, sigmas, and fourier matrix for an observation, prior, mask
    """

    # unpack keyword args
    fft_pad_factor = kwargs.get('fft_pad_factor',FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', GRIDDER_P_RAD_DEFAULT)

    # unpack data
    mdata = Obsdata.unpack(['u','v','m','msigma'], conj=True)
    uv = np.hstack((mdata['u'].reshape(-1,1), mdata['v'].reshape(-1,1)))
    m = mdata['m']
    sigmam = mdata['msigma']

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv)
    A = [A1]

    return (m, sigmam, A)


def chisqdata_vvis(Obsdata, Prior, mask):
    """Return the visibilities, sigmas, and fourier matrix for an observation, prior, mask
    """

    data_arr = Obsdata.unpack(['u','v','vvis','vsigma'], conj=False)
    uv = np.hstack((data_arr['u'].reshape(-1,1), data_arr['v'].reshape(-1,1)))
    vis = data_arr['vvis']
    sigma = data_arr['vsigma']
    A = ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv, pulse=Prior.pulse, mask=mask)

    return (vis, sigma, A)

def chisqdata_vvis_nfft(Obsdata, Prior, mask, **kwargs):
    """Return the visibilities, sigmas, and fourier matrix for an observation, prior, mask
    """

    # unpack keyword args
    fft_pad_factor = kwargs.get('fft_pad_factor',FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', GRIDDER_P_RAD_DEFAULT)

    # unpack data
    data_arr = Obsdata.unpack(['u','v','vvis','vsigma'], conj=False)
    uv = np.hstack((data_arr['u'].reshape(-1,1), data_arr['v'].reshape(-1,1)))
    vis = data_arr['vvis']
    sigma = data_arr['vsigma']

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv)
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
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
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
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title('m (above %i %% max flux)' % int(pcut*100))

    # Create title
    plotstr = "step: %i  " % nit
    for key in chi2_dict.keys():
        plotstr += "$\chi^2_{%s}$: %0.2f  " % (key, chi2_dict[key])
    plt.suptitle(plotstr, fontsize=18)


