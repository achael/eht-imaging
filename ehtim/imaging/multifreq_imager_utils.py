# multifreq_imager_utils.py
# imager functions for multifrequency VLBI data
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


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import range

import string
import time
import numpy as np
import scipy.optimize as opt
import scipy.ndimage as nd
import scipy.ndimage.filters as filt
import matplotlib.pyplot as plt
try:
    from pynfft.nfft import NFFT
except ImportError:
    pass
    #print("Warning: No NFFT installed! Cannot use nfft functions")

import ehtim.image as image
from . import linearize_energy as le

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *
#from ehtim.statistics.dataframes import *
from ehtim.imaging.imager_utils import embed

NORM_REGULARIZER = True
EPSILON = 1.e-12

##################################################################################################
# Mulitfrequency regularizers
##################################################################################################

def regularizer_mf(imvec, nprior, mask, xdim, ydim, psize, stype, **kwargs):
    """return the regularizer value on spectral index or curvature
    """

    norm_reg = kwargs.get('norm_reg', NORM_REGULARIZER)
    beam_size = kwargs.get('beam_size', psize)

    if "l2_" in stype:
        s = -l2_spec(imvec, nprior, norm_reg=norm_reg)
    elif "tv_" in stype:
        if np.any(np.invert(mask)):
            imvec = embed(imvec, mask, clipfloor=0, randomfloor=False)
        s = -tv_spec(imvec, xdim, ydim, psize, norm_reg=norm_reg, beam_size=beam_size)
    else:
        s = 0

    return s

def regularizergrad_mf(imvec, nprior, mask, xdim, ydim, psize, stype, **kwargs):
    """return the regularizer gradient
    """

    norm_reg = kwargs.get('norm_reg', NORM_REGULARIZER)
    beam_size = kwargs.get('beam_size', psize)

    if "l2_" in stype:
        s = -l2_spec_grad(imvec, nprior)
    elif "tv_" in stype:
        if np.any(np.invert(mask)):
            imvec = embed(imvec, mask, clipfloor=0, randomfloor=False)
        s = -tv_spec_grad(imvec, xdim, ydim, psize, norm_reg=norm_reg, beam_size=beam_size)
        s = s[mask]
    else:
        s = 0

    return s


def l2_spec(imvec, priorvec, norm_reg=NORM_REGULARIZER):
    """L2 norm on spectral index w/r/t prior
    """
    
    if norm_reg:
        norm = float(len(imvec))
    else:
        norm = 1
        
    out = -(np.sum((imvec - priorvec)**2))
    return out/norm

def l2_spec_grad(imvec, priorvec, norm_reg=NORM_REGULARIZER):
    """L2 norm on spectral index w/r/t prior
    """

    if norm_reg:
        norm = float(len(imvec))
    else:
        norm = 1
        
    out = -2*(np.sum(imvec - priorvec))*np.ones(len(imvec))
    return out/norm


def tv_spec(imvec, nx, ny, psize, norm_reg=NORM_REGULARIZER, beam_size=None):
    """Total variation regularizer
    """
    if beam_size is None: beam_size = psize
    if norm_reg: 
        norm = len(imvec)*psize / beam_size
    else: 
        norm = 1

    im = imvec.reshape(ny, nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    out = -np.sum(np.sqrt(np.abs(im_l1 - im)**2 + np.abs(im_l2 - im)**2 + EPSILON))

    return out/norm

def tv_spec_grad(imvec, nx, ny, psize, norm_reg=NORM_REGULARIZER, beam_size=None):
    """Total variation gradient
    """
    if beam_size is None: beam_size = psize
    if norm_reg: 
        norm = len(imvec)*psize / beam_size
    else: 
        norm = 1

    im = imvec.reshape(ny,nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    im_r1 = np.roll(impad, 1, axis=0)[1:ny+1, 1:nx+1]
    im_r2 = np.roll(impad, 1, axis=1)[1:ny+1, 1:nx+1]

    #rotate images
    im_r1l2 = np.roll(np.roll(impad,  1, axis=0),-1, axis=1)[1:ny+1, 1:nx+1]
    im_l1r2 = np.roll(np.roll(impad, -1, axis=0), 1, axis=1)[1:ny+1, 1:nx+1]

    #add together terms and return
    g1 = (2*im - im_l1 - im_l2) / np.sqrt((im - im_l1)**2 + (im - im_l2)**2 + EPSILON)
    g2 = (im - im_r1) / np.sqrt((im - im_r1)**2 + (im_r1l2 - im_r1)**2 + EPSILON)
    g3 = (im - im_r2) / np.sqrt((im - im_r2)**2 + (im_l1r2 - im_r2)**2 + EPSILON)

    #mask the first row column gradient terms that don't exist
    mask1 = np.zeros(im.shape)
    mask2 = np.zeros(im.shape)
    mask1[0,:] = 1
    mask2[:,0] = 1
    g2[mask1.astype(bool)] = 0
    g3[mask2.astype(bool)] = 0

    # add terms together and return
    out= -(g1 + g2 + g3).flatten()
    return out/norm

##################################################################################################

##################################################################################################
def image_at_freq(mftuple, log_freqratio):
        """Get the image or polarization image tuple from multifrequency data
        """
        # Stokes I 
        if len(mftuple==3):
            imvec_ref_log = np.log(mftuple[0])
            spectral_index = mftuple[1]
            curvature = mftuple[2]

            logimvec = imvec_ref_log + spectral_index*log_freqratio + curvature*log_freqratio*log_freqratio
            imvec = np.exp(logimvec)
            out = imvec
        # Polarization    
        elif len(mftuple==8):
            (I0, alpha, beta, m0, malpha, mbeta, chi0, rm) = mftuple                    
            imvec_ref_log = np.log(I0)
            logimvec = imvec_ref_log + alpha*log_freqratio + beta*log_freqratio*log_freqratio
            imvec = np.exp(logimvec)
            
            mvec_ref_log = np.log(m0)
            logmvec = mvec_ref_log + malpha*log_freqratio + mbeta*log_freqratio*log_freqratio
            mvec = np.exp(logmvec)
            
            # we use dimensionless rm scaled by lambda0^2 = c^2/nu0^2   
            chivec = chi0 + rm*(np.exp(-2*log_freqratio)-1)
            
            out = (imvec, mvec, chivec)
        else:
            raise Exception("in image_at_freq, len(mftuple) must be 3 or 8!")
            
        return out

def mf_all_grads_chain(funcgrad, image_cur, mftuple, log_freqratio):
        """Get the gradients of the reference image, spectral index, and curvature
           w/r/t the gradient of a function funcgrad to the image given frequency ref_freq*e(log_freqratio)
        """
        # Stokes I 
        if len(mftuple==3):
            imvec_cur = image_cur
            imvec_ref = mftuple[0]
            
            dfunc_dI0    = funcgrad * imvec_cur / imvec_ref
            dfunc_dalpha = funcgrad * imvec_cur * log_freqratio
            dfunc_dbeta  = funcgrad * imvec_cur * log_freqratio * log_freqratio

            out = np.array((dfunc_dI0, dfunc_dalpha, dfunc_dbeta))
            
        # Polarization
        elif len(mftuple==8):
            (dfunc_dI, dfunc_dm, dfunc_dchi) = funcgrad
            (imvec_cur, mvec_cur, chivec_cur) = image_cur   
            (I0, alpha, beta, m0, malpha, mbeta, chi0, rm) = mftuple    
            
            # apply chain rule to gradients w/r/t I 
            dfunc_dI0    = dfunc_dI * imvec_cur / I0
            dfunc_dalpha = dfunc_dI * imvec_cur * log_freqratio
            dfunc_dbeta  = dfunc_dI * imvec_cur * log_freqratio * log_freqratio    

            # apply chain rule for derivatives w/r/t m
            dfunc_dm0     = dfunc_dm * mvec_cur / m0
            dfunc_dmalpha = dfunc_dm * mvec_cur * log_freqratio
            dfunc_dmbeta  = dfunc_dm * mvec_cur * log_freqratio * log_freqratio

            # apply chain rule for derivatives w/r/t chi
            dfunc_dchi0 = dfunc_dchi
            dfunc_drm   = dfunc_dchi*(np.exp(-2*log_freqratio)-1)
            
            out = np.array((dfunc_dI0, dfunc_dalpha, dfunc_dbeta,
                            dfunc_dm0, dfunc_dmalpha, dfunc_dmbeta,
                            dfunc_dchi0, dfunc_drm))        
                                
        else:
            raise Exception("in image_at_freq, len(mftuple) must be 3 or 8!")  
                  
        return out
        

