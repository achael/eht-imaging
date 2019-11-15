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
    print("Warning: No NFFT installed! Cannot use nfft functions")

import ehtim.image as image
from . import linearize_energy as le

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *
from ehtim.statistics.dataframes import *

from IPython import display

NORM_REGULARIZER = True 

##################################################################################################
# Mulitfrequency regularizers
##################################################################################################

def regularizer_mf(imvec, nprior, mask, flux, xdim, ydim, psize, stype, **kwargs):
    """return the regularizer value on spectral index or curvature
    """

    norm_reg = kwargs.get('norm_reg', NORM_REGULARIZER)
    beam_size = kwargs.get('beam_size', psize)

    if stype == "l2_alpha":
        s = -l2_alpha(imvec, nprior)
    else:
        s = 0

    return s

def regularizergrad(imvec, nprior, mask, flux, xdim, ydim, psize, stype, **kwargs):
    """return the regularizer gradient
    """

    norm_reg = kwargs.get('norm_reg', NORM_REGULARIZER)
    beam_size = kwargs.get('beam_size', psize)

    if stype == "l2_alpha":
        s = -l2_alpha_grad(imvec, nprior)
    else:
        s = 0

    return s


def l2_alpha(imvec, priorvec):
    """L2 norm on spectral index w/r/t prior
    """

    norm = float(len(imvec))
    out = -(np.sum((imvec - priorvec)**2)
    return out/norm

def l2_alpha_grad(imvec, priorvec):
    """L2 norm on spectral index w/r/t prior
    """

    norm = float(len(imvec))
    out = -2*(np.sum(imvec) - flux)
    return out/norm


##################################################################################################

##################################################################################################
def unpack_mftuple(imvec, inittuple, nimage, mf_solve = (1,1,0)):
        """unpack imvec into tuple, 
           replaces quantities not iterated with their initial values
        """
        init0 = inittuple[0]
        init1 = inittuple[1]
        init2 = inittuple[2]

        imct = 0
        if mf_solve[0] == 0:
            im0 = init0
        else:
            im0 = imvec[imct*nimage:(imct+1)*nimage]
            imct += 1

        if mf_solve[1] == 0:
            im1 = init1
        else:
            im1 = imvec[imct*nimage:(imct+1)*nimage]
            imct += 1

        if mf_solve[2] == 0:
            im2 = init2
        else:
            im2 = imvec[imct*nimage:(imct+1)*nimage]
            imct += 1
        return np.array((im0, im1, im2))

def pack_poltuple(mftuple, mf_solve = (1,1,0)):       
        """pack multifreq data into image vector, 
           ignore quantities not iterated
        """

        vec = np.array([])
        if mf_solve[0] != 0:
            vec = np.hstack((vec,mftuple[0]))
        if mf_solve[1] != 0:
            vec = np.hstack((vec,mftuple[1]))
        if mf_solve[2] != 0:
            vec = np.hstack((vec,mftuple[2]))

        return vec

def embed_mf(imtuple, mask, clipfloor=0., randomfloor=False):
    """Embeds a multifrequency image tuple into the size of boolean embed mask
    """
    out0=np.zeros(len(mask))
    out1=np.zeros(len(mask))
    out2=np.zeros(len(mask))

    # Here's a much faster version than before 
    out0[mask.nonzero()] = imtuple[0]
    out1[mask.nonzero()] = imtuple[1]
    out2[mask.nonzero()] = imtuple[2]

    if clipfloor != 0.0:       
        if randomfloor: # prevent total variation gradient singularities
            out0[(mask-1).nonzero()] = clipfloor * np.abs(np.random.normal(size=len((mask-1).nonzero())))
            out1[(mask-1).nonzero()] = clipfloor * np.abs(np.random.normal(size=len((mask-1).nonzero())))
            out2[(mask-1).nonzero()] = clipfloor * np.abs(np.random.normal(size=len((mask-1).nonzero())))
        else:
            out0[(mask-1).nonzero()] = clipfloor
            out1[(mask-1).nonzero()] = clipfloor
            out2[(mask-1).nonzero()] = clipfloor

    return (out0, out1, out2)

def imvec_at_freq(mftuple, log_freqratio):
        """Get the image at a frequency given by ref_freq*e(log_freqratio)
           Remember spectral index is defined with a + sign!
        """
        imvec_ref_log = np.log(mftuple[0])
        spectral_index = mftuple[1]
        curvature = mftuple[2]

        logimvec = imvec_ref_log + spectral_index*log_freqratio + curvature*log_freqratio*log_freqratio
        imvec = np.exp(logimvec)
        return imvec

def mf_all_chisqgrads(chi2grad, imvec_cur, imvec_ref, log_freqratio):
        """Get the gradients of the reference image, spectral index, and curvature
           w/r/t the gradient of the chi^2 at a given frequency ref_freq*e(log_freqratio)
        """

        dchisq_dI0 = chi2grad * imvec_cur / imvec_ref 
        dchisq_dalpha = chi2grad * imvec_cur * log_freqratio
        dchisq_dbeta = dchisq_dalpha * log_freqratio
        
        return np.array((dchisq_dI0, dchisq_dalpha, dchisq_dbeta))

