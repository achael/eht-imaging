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


#TODO
# FIX NFFTS for  m and for p -- offsets? imag?

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
from  scipy.special import jv

import ehtim.image as image
from . import linearize_energy as le

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

TANWIDTH_M = 0.5
TANWIDTH_V = 1

##################################################################################################
# Constants & Definitions
##################################################################################################

NORM_REGULARIZER = False #ANDREW TODO change this default in the future

MAXLS = 100 # maximum number of line searches in L-BFGS-B
NHIST = 100 # number of steps to store for hessian approx
MAXIT = 100 # maximum number of iterations
STOP = 1.e-100 # convergence criterion

DATATERMS_POL = ['pvis', 'm', 'pbs','vvis']
REGULARIZERS_POL = ['msimple', 'hw', 'ptv','l1v','l2v','vtv','v2tv2']

nit = 0 # global variable to track the iteration number in the plotting callback

def qimage(iimage, mimage, chiimage):
    """Return the Q image from m and chi"""
    return iimage * mimage * np.cos(2*chiimage)
    
def uimage(iimage, mimage, chiimage):
    """Return the U image from m and chi"""
    return iimage * mimage * np.sin(2*chiimage) 


##################################################################################################
# Polarimetric Imager
##################################################################################################
def pol_imager_func(Obsdata, InitIm, Prior,
                    pol_trans=True, pol_solve = (0,1,1),
                    d1='pvis', d2=False, 
                    s1='msimple', s2=False,
                    alpha_d1=100, alpha_d2=100,
                    alpha_s1=1, alpha_s2=1,
                    **kwargs):

    """Run a polarimetric imager.

       Args:
           Obsdata (Obsdata): The Obsdata object with polarimetric VLBI data
           Prior (Image): The Image object with the prior image
           InitIm (Image): The Image object with the initial image for the minimization

           pol_trans (bool): polarization scaled to physical values or not
           pol_solve (tuple): len 3 tuple, solve for the corresponding pol image when not zero

           d1 (str): The first data term; options are 'pvis', 'm', 'pbs'
           d2 (str): The second data term; options are 'pvis', 'm', 'pbs'
           s1 (str): The first regularizer; options are 'msimple', 'hw', 'ptv'
           s2 (str): The second regularizer; options are 'msimple', 'hw', 'ptv'
           alpha_d1 (float): The first data term weighting
           alpha_d2 (float): The second data term weighting
           alpha_s1 (float): The first regularizer term weighting
           alpha_s2 (float): The second regularizer term weighting

           ttype (str): The Fourier transform type; options are 'fast' and 'direct'
           fft_pad_factor (float): The FFT will pre-pad the image by this factor x the original size
           p_rad (float): The pixel radius for the convolving function in gridding for FFTs

           clipfloor (float): The Stokes I Jy/pixel level above which prior image pixels are varied
           grads (bool): If True, analytic gradients are used
           norm_reg (bool): If True, normalizes regularizer terms
           beam_size (float): beam size in radians for normalizing the regularizers
           flux (float): The total flux of the output image in Jy

           maxit (int): Maximum number of minimizer iterations
           stop (float): The convergence criterion
           show_updates (bool): If True, displays the progress of the minimizer

       Returns:
           Image: Image object with result
    """

    # some kwarg default values
    maxit = kwargs.get('maxit', MAXIT)
    stop = kwargs.get('stop', STOP)
    clipfloor = kwargs.get('clipfloor', -1)
    ttype = kwargs.get('ttype','direct')
    grads = kwargs.get('grads',True)
    norm_init = kwargs.get('norm_init',False)
    show_updates = kwargs.get('show_updates',True)

    beam_size = kwargs.get('beam_size',Obsdata.res())
    flux = kwargs.get('flux',InitIm.total_flux())
    kwargs['beam_size'] = beam_size

    # Make sure data and regularizer options are ok
    if not d1 and not d2:
        raise Exception("Must have at least one data term!")
    if not s1 and not s2:
        raise Exception("Must have at least one regularizer term!")
    if (not ((d1 in DATATERMS_POL) or d1==False)) or (not ((d2 in DATATERMS_POL) or d2==False)):
        raise Exception("Invalid data term: valid data terms are: " + ' '.join(DATATERMS_POL))
    if (not ((s1 in REGULARIZERS_POL) or s1==False)) or (not ((s2 in REGULARIZERS_POL) or s2==False)):
        raise Exception("Invalid regularizer: valid regularizers are: " + ' '.join(REGULARIZERS_POL))
    if (Prior.psize != InitIm.psize) or (Prior.xdim != InitIm.xdim) or (Prior.ydim != InitIm.ydim):
        raise Exception("Initial image does not match dimensions of the prior image!")
    if (ttype not in ["direct","nfft"]):
        raise Exception("FFT no yet implemented in polarimetric imaging -- use NFFT!")
    if (not pol_trans):
        raise Exception("Only pol_trans==True supported!")
    if (len(pol_solve)!=3):
        raise Exception("pol_solve tuple must have 3 entries!")

    # Catch scale and dimension problems
    imsize = np.max([Prior.xdim, Prior.ydim]) * Prior.psize
    uvmax = 1.0/Prior.psize
    uvmin = 1.0/imsize
    uvdists = Obsdata.unpack('uvdist')['uvdist']
    maxbl = np.max(uvdists)
    minbl = np.max(uvdists[uvdists > 0])
    maxamp = np.max(np.abs(Obsdata.unpack('amp')['amp']))

    if uvmax < maxbl:
        print("Warning! Pixel Spacing is larger than smallest spatial wavelength!")
    if uvmin > minbl:
        print("Warning! Field of View is smaller than largest nonzero spatial wavelength!")

    # convert polrep to stokes
    # ANDREW todo -- make more general??
    Prior = Prior.switch_polrep(polrep_out='stokes', pol_prim_out='I')
    InitIm = InitIm.switch_polrep(polrep_out='stokes', pol_prim_out='I')

    # embedding mask 
    embed_mask = Prior.imvec > clipfloor

    # initial Stokes I image
    iimage = InitIm.imvec[embed_mask]
    nimage = len(iimage)

    # initial pol image
    if pol_trans:
        if len(InitIm.qvec) and (np.any(InitIm.qvec!=0) or np.any(InitIm.uvec!=0)): #TODO right? or should it be if any=0
            init1 = (np.abs(InitIm.qvec + 1j*InitIm.uvec) / InitIm.imvec)[embed_mask]
            init2 = (np.arctan2(InitIm.uvec, InitIm.qvec) / 2.0)[embed_mask]
        else:
            # !AC TODO get the actual zero baseline pol. frac from the data!??
            print("No polarimetric image in the initial image!")
            init1 = 0.2 * (np.ones(len(iimage)) + 1e-2 * np.random.rand(len(iimage)))
            init2 = np.zeros(len(iimage)) + 1e-2 * np.random.rand(len(iimage))

        # Change of variables    
        inittuple = np.array((iimage, init1, init2))
        xtuple =  mcv_r(inittuple)

    # Get data and fourier matrices for the data terms
    (data1, sigma1, A1) = polchisqdata(Obsdata, Prior, embed_mask, d1, **kwargs)
    (data2, sigma2, A2) = polchisqdata(Obsdata, Prior, embed_mask, d2, **kwargs)

    # Define the chi^2 and chi^2 gradient
    def chisq1(imtuple):
        return polchisq(imtuple, A1, data1, sigma1, d1, ttype=ttype,
                        mask=embed_mask, pol_trans=pol_trans)

    def chisq1grad(imtuple):
        return polchisqgrad(imtuple, A1, data1, sigma1, d1, ttype=ttype, mask=embed_mask,
                            pol_trans=pol_trans, pol_solve=pol_solve)

    def chisq2(imtuple):
        return polchisq(imtuple, A2, data2, sigma2, d2, ttype=ttype, mask=embed_mask,
                        pol_trans=pol_trans)

    def chisq2grad(imtuple):
        return polchisqgrad(imtuple, A2, data2, sigma2, d2, ttype=ttype, mask=embed_mask,
                            pol_trans=pol_trans,pol_solve=pol_solve)

    # Define the regularizer and regularizer gradient
    def reg1(imtuple):
        return polregularizer(imtuple, embed_mask, flux, 
                              Prior.xdim, Prior.ydim, Prior.psize, s1, **kwargs)
        return polregularizer(imtuple, embed_mask, Prior.xdim, Prior.ydim, Prior.psize, s1, 
                              pol_trans=pol_trans,norm_reg=norm_reg)

    def reg1grad(imtuple):
        return polregularizergrad(imtuple, embed_mask, flux, 
                                  Prior.xdim, Prior.ydim, Prior.psize, s1, **kwargs)

    def reg2(imtuple):
        return polregularizer(imtuple, embed_mask, flux, 
                              Prior.xdim, Prior.ydim, Prior.psize, s2, **kwargs)

    def reg2grad(imtuple):
        return  polregularizergrad(imtuple, embed_mask, flux, 
                                   Prior.xdim, Prior.ydim, Prior.psize, s2, **kwargs)


    # Define the objective function and gradient
    def objfunc(allvec):
        # unpack allvec into image tuple
        cvtuple = unpack_poltuple(allvec, xtuple, nimage, pol_solve)

        # change of variables
        if pol_trans:
            imtuple = mcv(cvtuple)
        else:
            raise  Exception()

        datterm = alpha_d1 * (chisq1(imtuple) - 1) + alpha_d2 * (chisq2(imtuple) - 1)
        regterm = alpha_s1 * reg1(imtuple) + alpha_s2 * reg2(imtuple)

        return datterm + regterm

    def objgrad(allvec):
        # unpack allvec into image tuple
        cvtuple = unpack_poltuple(allvec, xtuple, nimage, pol_solve)

        # change of variables
        if pol_trans:
            imtuple = mcv(cvtuple)
        else:
            raise Exception()

        datterm = alpha_d1 * chisq1grad(imtuple) + alpha_d2 * chisq2grad(imtuple)
        regterm = alpha_s1 * reg1grad(imtuple) + alpha_s2 * reg2grad(imtuple)
        gradarr = datterm + regterm

        # chain rule
        if pol_trans:
            chainarr = mchain(cvtuple)
            gradarr = gradarr*chainarr

        # repack grad into single vector
        grad = pack_poltuple(gradarr, pol_solve)

        return grad

    # Define plotting function for each iteration
    global nit
    nit = 0
    def plotcur(im_step):
        global nit
        cvtuple = unpack_poltuple(im_step, xtuple, nimage, pol_solve)
        if pol_trans:
            imtuple = mcv(cvtuple)  #change of variables
        else:
            raise Exception()

        if show_updates:
            #print( np.max(np.abs((imtuple[2]-xx_static)/xx_static)))
            chi2_1 = chisq1(imtuple)
            chi2_2 = chisq2(imtuple)
            s_1 = reg1(imtuple)
            s_2 = reg2(imtuple)
            if np.any(np.invert(embed_mask)): 
                imtuple = embed_pol(imtuple, embed_mask)
            plot_m(imtuple, Prior, nit, {d1:chi2_1, d2:chi2_2})
            print("i: %d chi2_1: %0.2f chi2_2: %0.2f s_1: %0.2f s_2: %0.2f" % (nit, chi2_1, chi2_2,s_1,s_2))
        nit += 1

    # Print stats
    print("Initial S_1: %f S_2: %f" % (reg1(inittuple), reg2(inittuple)))
    print("Initial Chi^2_1: %f Chi^2_2: %f" % (chisq1(inittuple), chisq2(inittuple)))
    if d1 in DATATERMS_POL:
        print("Total Data 1: ", (len(data1)))
    if d2 in DATATERMS_POL:
        print("Total Data 2: ", (len(data2)))
    print("Total Pixel #: ", (len(Prior.imvec)))
    print("Clipped Pixel #: ", nimage)
    print()

    # Plot Initial
    xinit = pack_poltuple(xtuple.copy(), pol_solve)
    plotcur(xinit)

    # Minimize
    optdict = {'maxiter':maxit, 'ftol':stop, 'maxcor':NHIST,'maxls':MAXLS,'gtol':stop,'maxfun':1.e100} # minimizer dict params
    tstart = time.time()
    if grads:
        res = opt.minimize(objfunc, xinit, method='L-BFGS-B', jac=objgrad,callback=plotcur,
                           options=optdict)
    else:
        res = opt.minimize(objfunc, xinit, method='L-BFGS-B',
                           options=optdict, callback=plotcur)



    tstop = time.time()
    
    # Format output
    outcv = unpack_poltuple(res.x, xtuple, nimage, pol_solve)
    if pol_trans:
        outcut = mcv(outcv)  #change of variables
    else:
        outcut = outcv

    if np.any(np.invert(embed_mask)): 
        out = embed_pol(out, embed_mask) #embed
    else:
        out = outcut

    iimage = out[0]
    qimage = make_q_image(out, pol_trans)
    uimage = make_u_image(out, pol_trans)

    outim = image.Image(iimage.reshape(Prior.ydim, Prior.xdim), Prior.psize,
                         Prior.ra, Prior.dec, rf=Prior.rf, source=Prior.source,
                         mjd=Prior.mjd, pulse=Prior.pulse)
    outim.add_qu(qimage.reshape(Prior.ydim, Prior.xdim), uimage.reshape(Prior.ydim, Prior.xdim))

    # Print stats
    print("time: %f s" % (tstop - tstart))
    print("J: %f" % res.fun)
    print("Final Chi^2_1: %f Chi^2_2: %f" % (chisq1(outcut), chisq2(outcut)))
    print(res.message)

    # Return Image object
    return outim

##################################################################################################
# Linear Polarimetric image representations and Change of Variables
##################################################################################################
def pack_poltuple(poltuple, pol_solve = (0,1,1)):       
        """pack polvec into image vector, 
           ignore quantities not iterated
        """

        vec = np.array([])
        if pol_solve[0] != 0:
            vec = np.hstack((vec,poltuple[0]))
        if pol_solve[1] != 0:
            vec = np.hstack((vec,poltuple[1]))
        if pol_solve[2] != 0:
            vec = np.hstack((vec,poltuple[2]))
        if len(pol_solve)==4 and pol_solve[3] != 0:
            vec = np.hstack((vec,poltuple[3]))
                    
        return vec


def unpack_poltuple(polvec, inittuple, nimage, pol_solve = (0,1,1)):       
        """unpack polvec into image tuple, 
           replaces quantities not iterated with  initial values
        """
        init0 = inittuple[0]
        init1 = inittuple[1]
        init2 = inittuple[2]
        if len(pol_solve)==4: 
            init3 = inittuple[3]
        
        imct = 0
        if pol_solve[0] == 0:
            im0 = init0
        else:
            im0 = polvec[imct*nimage:(imct+1)*nimage]
            imct += 1

        if pol_solve[1] == 0:
            im1 = init1
        else:
            im1 = polvec[imct*nimage:(imct+1)*nimage]
            imct += 1

        if pol_solve[2] == 0:
            im2 = init2
        else:
            im2 = polvec[imct*nimage:(imct+1)*nimage]
            imct += 1
            
        if len(pol_solve)==4:
            if pol_solve[3] == 0:
                im3 = init3
            else:
                im3 = polvec[imct*nimage:(imct+1)*nimage]
                imct += 1    
            out = np.array((im0, im1, im2, im3))        
        else:        
            out = np.array((im0, im1, im2)) 
               
        return out

def make_p_image(imtuple, pol_trans=True):
    """construct a polarimetric image P = Q + iU
    """

    if pol_trans:
        pimage = imtuple[0] * imtuple[1] * np.exp(2j*imtuple[2])
    else:
        pimage = imtuple[1] + 1j*imtuple[2]

    return pimage

def make_m_image(imtuple, pol_trans=True):
    """construct a polarimetric ratrio image abs(P/I) = abs(Q + iU)/I
    """

    if pol_trans:
        mimage = imtuple[1]
    else:
        mimage = np.abs((imtuple[1] + 1j*imtuple[2])/imtuple[0])
    return mimage

def make_chi_image(imtuple, pol_trans=True):
    """construct a polarimetric angle image 
    """

    if pol_trans:
        chiimage = imtuple[2]
    else: 
        chiimage = 0.5*np.angle((imtuple[1] + 1j*imtuple[2])/imtuple[0])

    return chiimage

def make_q_image(imtuple, pol_trans=True):
    """construct an image of stokes Q
    """

    if pol_trans:
        qimage = imtuple[0] * imtuple[1] * np.cos(2*imtuple[2])
    else:
        qimage = imtuple[1]

    return qimage

def make_u_image(imtuple, pol_trans=True):
    """construct an image of stokes U
    """

    if pol_trans:
        uimage = imtuple[0] * imtuple[1] * np.sin(2*imtuple[2])
    else:
        uimage = imtuple[2]

    return uimage

def make_v_image(imtuple, pol_trans=True):
    """construct an image of stokes V
    """

    if len(imtuple)==4:
        if pol_trans:
            vimage = imtuple[0] * imtuple[3]
        else:
            vimage = imtuple[3]
    else:
        vimage = np.zeros(imtuple[0].shape)
        
    return vimage

def make_vfrac_image(imtuple, pol_trans=True):
    """construct an image of stokes V
    """
    if len(imtuple)==4:
        if pol_trans:
            vfimage = imtuple[3]
        else:
            vfimage = imtuple[3]/imtuple[0]
    else:
        vfimage = np.zeros(imtuple[0].shape)
        
    return vfimage
    
        
# these change of variables only apply to polarimetric ratios
# !AC In these pol. changes of variables, might be useful to 
# take m -> m/100 by adjusting B (function becomes less steep around m' = 0)

def mcv(imtuple):
    """Change of pol. ratio from range (-inf, inf) to (0,1)
    """
    
    iimage = imtuple[0]
    mimage =  imtuple[1]
    chiimage = imtuple[2]

    mtrans = 0.5 + np.arctan(mimage/TANWIDTH_M)/np.pi
    if len(imtuple)==4:
        vfimage = imtuple[3]
        vtrans = 2*np.arctan(vfimage/TANWIDTH_V)/np.pi    
        out = np.array((iimage, mtrans, chiimage, vtrans))
    else:
        out = np.array((iimage, mtrans, chiimage))    
    

    return out

def mcv_r(imtuple):
    """Change of pol. ratio from range (0,1) to (-inf,inf)
    """
    iimage = imtuple[0]
    mimage =  imtuple[1]
    chiimage = imtuple[2]

    mtrans = TANWIDTH_M*np.tan(np.pi*(mimage - 0.5))
    if len(imtuple)==4:
        vfimage = imtuple[3]
        vtrans = TANWIDTH_V*np.tan(0.5*np.pi*(vfimage))    
        out = np.array((iimage, mtrans, chiimage, vtrans))
    else:
        out = np.array((iimage, mtrans, chiimage))
                
    return out

def mchain(imtuple):
    """The gradient change of variables, dm/dm'
    """
    iimage = imtuple[0]
    mimage =  imtuple[1]
    chiimage = imtuple[2]

    ichain = np.ones(len(iimage))
    mmchain = 1 / (TANWIDTH_M*np.pi*(1 + (mimage/TANWIDTH_M)**2))
    chichain = np.ones(len(chiimage))
    
    if len(imtuple)==4:
        vfimage = imtuple[3]
        vchain = 2. / (TANWIDTH_V*np.pi*(1 + (vfimage/TANWIDTH_V)**2))
        out = np.array((ichain, mmchain, chichain, vchain))
    else:
        out = np.array((ichain, mmchain, chichain))
        
    return np.array(out)


##################################################################################################
# Wrapper Functions
##################################################################################################

def polchisq(imtuple, A, data, sigma, dtype, ttype='direct', mask=[], pol_trans=True):
    """return the chi^2 for the appropriate dtype
    """

    chisq = 1 
    if not dtype in DATATERMS_POL:
        return chisq
    if ttype not in ['fast','direct','nfft']:
        raise Exception("Possible ttype values are 'fast' and 'direct'!")

    if ttype == 'direct':
        # linear
        if dtype == 'pvis':
            chisq = chisq_p(imtuple, A, data, sigma, pol_trans)
        elif dtype == 'm':
            chisq = chisq_m(imtuple, A, data, sigma, pol_trans)
        elif dtype == 'pbs':
            chisq = chisq_pbs(imtuple, A, data, sigma, pol_trans)
            
        # circular
        elif dtype == 'vvis':
            chisq = chisq_vvis(imtuple, A, data, sigma, pol_trans)        
            
    elif ttype== 'fast':
        raise Exception("FFT not yet implemented in polchisq!")

    elif ttype== 'nfft':
        if len(mask)>0 and np.any(np.invert(mask)):
            imtuple = embed_pol(imtuple, mask, randomfloor=True)

        # linear
        if dtype == 'pvis':
            chisq = chisq_p_nfft(imtuple, A, data, sigma, pol_trans)
        elif dtype == 'm':
            chisq = chisq_m_nfft(imtuple, A, data, sigma, pol_trans)
        elif dtype == 'pbs':
            chisq = chisq_pbs_nfft(imtuple, A, data, sigma, pol_trans)
            
        # circular
        elif dtype == 'vvis':
            chisq = chisq_vvis_nfft(imtuple, A, data, sigma, pol_trans)        

    return chisq

def polchisqgrad(imtuple, A, data, sigma, dtype, ttype='direct',
                 mask=[], pol_trans=True,pol_solve=(0,1,1)):
    
    """return the chi^2 gradient for the appropriate dtype
    """

    chisqgrad = np.zeros((3,len(imtuple[0])))
    if not dtype in DATATERMS_POL:
        return chisqgrad
    if ttype not in ['fast','direct','nfft']:
        raise Exception("Possible ttype values are 'fast' and 'direct'!")

    if ttype == 'direct':
        # linear
        if dtype == 'pvis':
            chisqgrad = chisqgrad_p(imtuple, A, data, sigma, pol_trans,pol_solve)
        elif dtype == 'm':
            chisqgrad = chisqgrad_m(imtuple, A, data, sigma, pol_trans,pol_solve)
        elif dtype == 'pbs':
            chisqgrad = chisqgrad_pbs(imtuple, A, data, sigma, pol_trans,pol_solve)
            
        # circular
        elif dtype == 'vvis':
            chisqgrad = chisqgrad_vvis(imtuple, A, data, sigma, pol_trans,pol_solve)
                
    elif ttype== 'fast':
        raise Exception("FFT not yet implemented in polchisqgrad!")

    elif ttype== 'nfft':
        if len(mask)>0 and np.any(np.invert(mask)):
            imtuple = embed_pol(imtuple, mask, randomfloor=True)
      
        # linear
        if dtype == 'pvis':
            chisqgrad = chisqgrad_p_nfft(imtuple, A, data, sigma, pol_trans,pol_solve)
        elif dtype == 'm':
            chisqgrad = chisqgrad_m_nfft(imtuple, A, data, sigma, pol_trans,pol_solve)
        elif dtype == 'pbs':
            chisqgrad = chisqgrad_pbs_nfft(imtuple, A, data, sigma, pol_trans,pol_solve)

        # circular
        elif dtype == 'vvis':
            chisqgrad = chisqgrad_vvis_nfft(imtuple, A, data, sigma, pol_trans,pol_solve)
            
        if len(mask)>0 and np.any(np.invert(mask)):
            if len(chisqgrad)==4:
                chisqgrad = np.array((chisqgrad[0][mask],chisqgrad[1][mask],chisqgrad[2][mask],chisqgrad[3][mask]))
            else:
                chisqgrad = np.array((chisqgrad[0][mask],chisqgrad[1][mask],chisqgrad[2][mask]))

    return chisqgrad


def polregularizer(imtuple, mask, flux, xdim, ydim, psize, stype, **kwargs):

    norm_reg = kwargs.get('norm_reg', NORM_REGULARIZER)
    pol_trans = kwargs.get('pol_trans', True)
    beam_size = kwargs.get('beam_size',1)

    # linear
    if stype == "msimple":
        reg = -sm(imtuple, flux, pol_trans, norm_reg=norm_reg)
    elif stype == "hw":
        reg = -shw(imtuple, flux, pol_trans, norm_reg=norm_reg)
    elif stype == "ptv":
        if np.any(np.invert(mask)):
            imtuple = embed_pol(imtuple, mask, randomfloor=True)
        reg = -stv_pol(imtuple, flux, xdim, ydim, psize, pol_trans, 
                       norm_reg=norm_reg, beam_size=beam_size)
    # circular
    elif stype == "l1v":
        reg = -sl1v(imtuple, flux, pol_trans, norm_reg=norm_reg)
    elif stype == "l2v":
        reg = -sl2v(imtuple, flux, pol_trans, norm_reg=norm_reg)
    elif stype == "vtv":
        if np.any(np.invert(mask)):
            imtuple = embed_pol(imtuple, mask, randomfloor=True)
        reg = -stv_v(imtuple, flux, xdim, ydim, psize, pol_trans, 
                     norm_reg=norm_reg, beam_size=beam_size)
    elif stype == "vtv2":
        if np.any(np.invert(mask)):
            imtuple = embed_pol(imtuple, mask, randomfloor=True)
        reg = -stv2_v(imtuple, flux, xdim, ydim, psize, pol_trans, 
                      norm_reg=norm_reg, beam_size=beam_size)                               
    else:
        reg = 0

    return reg

def polregularizergrad(imtuple, mask, flux, xdim, ydim, psize, stype, **kwargs):

    norm_reg = kwargs.get('norm_reg', NORM_REGULARIZER)
    pol_trans = kwargs.get('pol_trans', True)
    pol_solve = kwargs.get('pol_solve', (0,1,1))
    beam_size = kwargs.get('beam_size',1)

    # linear
    if stype == "msimple":
        reggrad = -smgrad(imtuple, flux, pol_trans, pol_solve, norm_reg=norm_reg)
    elif stype == "hw":
        reggrad = -shwgrad(imtuple, flux, pol_trans, pol_solve, norm_reg=norm_reg)
    elif stype == "ptv":
        if np.any(np.invert(mask)):
            imtuple = embed_pol(imtuple, mask, randomfloor=True)
        reggrad = -stv_pol_grad(imtuple, flux, xdim, ydim, psize, pol_trans, pol_solve,
                                norm_reg=norm_reg, beam_size=beam_size)
        if np.any(np.invert(mask)):
            if len(reggrad)==4: 
                reggrad = np.array((reggrad[0][mask],reggrad[1][mask],reggrad[2][mask],reggrad[3][mask]))
            else: 
                reggrad = np.array((reggrad[0][mask],reggrad[1][mask],reggrad[2][mask]))

    # circular
    elif stype == "l1v":
        reggrad = -sl1vgrad(imtuple, flux, pol_trans, pol_solve=pol_solve, norm_reg=norm_reg)
    elif stype == "l2v":
        reggrad = -sl2vgrad(imtuple, flux, pol_trans, pol_solve=pol_solve, norm_reg=norm_reg)    
    elif stype == "vtv":
        if np.any(np.invert(mask)):
            imtuple = embed_pol(imtuple, mask, randomfloor=True)
        reggrad = -stv_v_grad(imtuple, flux, xdim, ydim, psize, pol_trans,
                              pol_solve=pol_solve, norm_reg=norm_reg, beam_size=beam_size)
        if np.any(np.invert(mask)):
            reggrad = np.array((reggrad[0][mask],reggrad[1][mask],reggrad[2][mask],reggrad[3][mask]))
    elif stype == "vtv2":
        if np.any(np.invert(mask)):
            imtuple = embed_pol(imtuple, mask, randomfloor=True)
        reggrad = -stv2_v_grad(imtuple, flux, xdim, ydim, psize, pol_trans, 
                               pol_solve=pol_solve, norm_reg=norm_reg, beam_size=beam_size)
        if np.any(np.invert(mask)):
            reggrad = np.array((reggrad[0][mask],reggrad[1][mask],reggrad[2][mask],reggrad[3][mask]))

    else:
        reggrad = np.zeros((len(imtuple),len(imtuple[0])))

    return reggrad


def polchisqdata(Obsdata, Prior, mask, dtype, **kwargs):

    """Return the data, sigma, and matrices for the appropriate dtype
    """

    ttype=kwargs.get('ttype','direct')

    (data, sigma, A) = (False, False, False)
    if ttype not in ['fast','direct','nfft']:
        raise Exception("Possible ttype values are 'fast' and 'direct' and 'nfft'!")
    if ttype=='direct':
        if dtype == 'pvis':
            (data, sigma, A) = chisqdata_pvis(Obsdata, Prior, mask)
        elif dtype == 'm':
            (data, sigma, A) = chisqdata_m(Obsdata, Prior, mask)
        elif dtype == 'pbs':
            (data, sigma, A) = chisqdata_pbs(Obsdata, Prior, mask)
        elif dtype == 'vvis':
            (data, sigma, A) = chisqdata_vvis(Obsdata, Prior, mask)
    elif ttype=='fast':
        raise Exception("FFT not yet implemented in polchisqdata!")

    elif ttype=='nfft':
        if dtype == 'pvis':
            (data, sigma, A) = chisqdata_pvis_nfft(Obsdata, Prior, mask, **kwargs)
        elif dtype == 'm':
            (data, sigma, A) = chisqdata_m_nfft(Obsdata, Prior, mask, **kwargs)
        elif dtype == 'pbs':
            (data, sigma, A) = chisqdata_pbs_nfft(Obsdata, Prior, mask, **kwargs)
        elif dtype == 'vvis':
            (data, sigma, A) = chisqdata_vvis_nfft(Obsdata, Prior, mask, **kwargs)
        
    return (data, sigma, A)


##################################################################################################
# DFT Chi-squared and Gradient Functions
##################################################################################################

def chisq_p(imtuple, Amatrix, p, sigmap, pol_trans=True):
    """Polarimetric visibility chi-squared
    """

    pimage = make_p_image(imtuple, pol_trans)
    psamples = np.dot(Amatrix, pimage) 
    chisq =  np.sum(np.abs((p - psamples))**2/(sigmap**2)) / (2*len(p))   
    return chisq

def chisqgrad_p(imtuple, Amatrix, p, sigmap, pol_trans=True,pol_solve=(0,1,1)):
    """Polarimetric visibility chi-squared gradient
    """
    

    iimage = imtuple[0]
    pimage = make_p_image(imtuple, pol_trans)
    psamples = np.dot(Amatrix, pimage)
    pdiff = (p - psamples) / (sigmap**2)
    zeros =  np.zeros(len(iimage))
        
    if pol_trans:

        mimage = imtuple[1]
        chiimage = imtuple[2]
        
        if pol_solve[0]!=0:
            gradi = -np.real(mimage * np.exp(-2j*chiimage) * np.dot(Amatrix.conj().T, pdiff)) / len(p)
        else:
            gradi = zeros

        if pol_solve[1]!=0:
            gradm = -np.real(iimage * np.exp(-2j*chiimage) * np.dot(Amatrix.conj().T, pdiff)) / len(p)
        else:
            gradm = zeros

        if pol_solve[2]!=0:
            gradchi = -2 * np.imag(pimage.conj() * np.dot(Amatrix.conj().T, pdiff)) / len(p)
        else:
            gradchi = zeros

        # output tuple can be length 3 or 4 depending on V
        if len(imtuple)==4:
            gradout = np.array((gradi, gradm, gradchi, zeros))        
        else:
            gradout = np.array((gradi, gradm, gradchi))
            

    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_trans)

    return gradout

def chisq_m(imtuple, Amatrix, m, sigmam, pol_trans=True):
    """Polarimetric ratio chi-squared
    """

    iimage = imtuple[0]
    pimage = make_p_image(imtuple, pol_trans)
    msamples = np.dot(Amatrix, pimage) / np.dot(Amatrix, iimage)
    return np.sum(np.abs((m - msamples))**2/(sigmam**2)) / (2*len(m))   

def chisqgrad_m(imtuple, Amatrix, m, sigmam, pol_trans=True,pol_solve=(0,1,1)):
    """The gradient of the polarimetric ratio chisq
    """
    iimage = imtuple[0]
    pimage = make_p_image(imtuple, pol_trans)

    isamples = np.dot(Amatrix, iimage)
    psamples = np.dot(Amatrix, pimage)
    zeros  =  np.zeros(len(iimage))

    if pol_trans:

        mimage = imtuple[1]
        chiimage = imtuple[2]
        
        msamples = psamples/isamples
        mdiff = (m - msamples) / (isamples.conj() * sigmam**2)

        if pol_solve[0]!=0:
            gradi = (-np.real(mimage * np.exp(-2j*chiimage) * np.dot(Amatrix.conj().T, mdiff)) / len(m) + 
                      np.real(np.dot(Amatrix.conj().T, msamples.conj() * mdiff)) / len(m))
        else:
            gradi = zeros

        if pol_solve[1]!=0:
            gradm = -np.real(iimage*np.exp(-2j*chiimage) * np.dot(Amatrix.conj().T, mdiff)) / len(m)
        else:
            gradm = zeros

        if pol_solve[2]!=0:
            gradchi = -2 * np.imag(pimage.conj() * np.dot(Amatrix.conj().T, mdiff)) / len(m)
        else:
            gradchi = zeros

        # output tuple can be length 3 or 4 depending on V
        if len(imtuple)==4:
            gradout = np.array((gradi, gradm, gradchi, zeros))        
        else:
            gradout = np.array((gradi, gradm, gradchi))
            

    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_trans)

    return gradout

def chisq_pbs(imtuple, Amatrices, bis_p, sigma, pol_trans=True):
    """Polarimetric bispectrum chi-squared
    """
    
    iimage = imtuple[0]
    pimage = make_p_image(imtuple, pol_trans)
    bisamples_p = np.dot(Amatrices[0], pimage) * np.dot(Amatrices[1], pimage) * np.dot(Amatrices[2], pimage)
    chisq =  np.sum(np.abs((bis_p - bisamples_p)/sigma)**2) / (2.*len(bis_p))
    return chisq

def chisqgrad_pbs(imtuple, Amatrices, bis_p, sigma, pol_trans=True,pol_solve=(0,1,1)):
    """Polarimetric bispectrum chi-squared gradient
    """
    pimage = make_p_image(imtuple, pol_trans)
    bisamples_p = np.dot(Amatrices[0], pimage) * np.dot(Amatrices[1], pimage) * np.dot(Amatrices[2], pimage)

    wdiff = ((bis_p - bisamples_p).conj()) / (sigma**2)      
    pt1 = wdiff * np.dot(Amatrices[1],pimage) * np.dot(Amatrices[2],pimage)
    pt2 = wdiff * np.dot(Amatrices[0],pimage) * np.dot(Amatrices[2],pimage)
    pt3 = wdiff * np.dot(Amatrices[0],pimage) * np.dot(Amatrices[1],pimage)
    ptsum = np.dot(pt1, Amatrices[0]) + np.dot(pt2, Amatrices[1]) + np.dot(pt3, Amatrices[2])    

    if pol_trans:
        iimage = imtuple[0]
        mimage = imtuple[1]
        chiimage = imtuple[2]
        
        if pol_solve[0]!=0:
            gradi = -np.real(ptsum * mimage * np.exp(2j*chiimage)) / len(bis_p)
        else:
            gradi = zeros
        if pol_solve[1]!=0:
            gradm = -np.real(ptsum * iimage * np.exp(2j*chiimage)) / len(bis_p)
        else:
            gradm = zeros
        if pol_solve[2]!=0:
            gradchi = 2 * np.imag(ptsum * pimage) / len(bis_p)
        else:
            gradchi = zeros

        # output tuple can be length 3 or 4 depending on V
        if len(imtuple)==4:
            gradout = np.array((gradi, gradm, gradchi, zeros))        
        else:
            gradout = np.array((gradi, gradm, gradchi))
            
    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_trans)

    return gradout

# stokes v
def chisq_vvis(imtuple, Amatrix, v, sigmav, pol_trans=True):
    """V visibility chi-squared
    """

    vimage = make_v_image(imtuple, pol_trans)
    vsamples = np.dot(Amatrix, vimage) 
    chisq =  np.sum(np.abs((v - vsamples))**2/(sigmav**2)) / (2*len(v))   
    return chisq

def chisqgrad_vvis(imtuple, Amatrix, v, sigmap, pol_trans=True,pol_solve=(0,1,1)):
    """V visibility chi-squared gradient
    """
    

    iimage = imtuple[0]
    vimage = make_v_image(imtuple, pol_trans)
    vsamples = np.dot(Amatrix, vimage)
    pdiff = (v - vsamples) / (sigmav**2)
    zeros =  np.zeros(len(iimage))
        
    if pol_trans:
        vfimage = imtuple[3] # fractional   
        if pol_solve[0]!=0:
            gradi = -np.real(vfimage * np.dot(Amatrix.conj().T, vdiff)) / len(v)
        else:
            gradi = zeros

        if pol_solve[3]!=0:
            gradv = -np.real(iimage * np.dot(Amatrix.conj().T, vdiff)) / len(v)
        else:
            gradm = zeros


        gradout = np.array((gradi, zeros, zeros, gradv))

    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_trans)

    return gradout

##################################################################################################
# NFFT Chi-squared and Gradient Functions
##################################################################################################
def chisq_p_nfft(imtuple, A, p, sigmap, pol_trans=True):
    """P visibility chi-squared
    """

    #get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    #compute uniform --> nonuniform transform
    pimage = make_p_image(imtuple, pol_trans)
    plan.f_hat = pimage.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    psamples = plan.f.copy()*pulsefac

    #compute chi^2
    chisq =  np.sum(np.abs((p - psamples))**2/(sigmap**2)) / (2*len(p))   

    return chisq

def chisqgrad_p_nfft(imtuple, A, p, sigmap, pol_trans=True,pol_solve=(0,1,1)):
    """P visibility chi-squared gradient
    """
    
    #get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    #compute uniform --> nonuniform transform
    iimage = imtuple[0]
    pimage = make_p_image(imtuple, pol_trans)
    plan.f_hat = pimage.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    psamples = plan.f.copy()*pulsefac


    pdiff_vec = (-1.0/len(p)) * (p - psamples) / (sigmap**2) * pulsefac.conj()
    plan.f = pdiff_vec
    plan.adjoint()
    ppart = (plan.f_hat.copy().T).reshape(nfft_info.xdim*nfft_info.ydim)

    zeros =  np.zeros(len(iimage))
        
    if pol_trans:

        mimage = imtuple[1]
        chiimage = imtuple[2]
        
        if pol_solve[0]!=0:

            gradi = np.real(mimage * np.exp(-2j*chiimage) * ppart)
        else:
            gradi = zeros

        if pol_solve[1]!=0:
            gradm = np.real(iimage*np.exp(-2j*chiimage) *ppart) 
        else:
            gradm = zeros

        if pol_solve[2]!=0:
            gradchi = 2 * np.imag(pimage.conj() * ppart)
        else:
            gradchi = zeros

        if len(imtuple)==4:
            gradout = np.array((gradi, gradm, gradchi, zeros))        
        else:
            gradout = np.array((gradi, gradm, gradchi))

    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_trans)

    return gradout


def chisq_m_nfft(imtuple, A, m, sigmam, pol_trans=True):
    """Polarimetric ratio chi-squared
    """
    #get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    #compute uniform --> nonuniform transform
    iimage = imtuple[0]
    plan.f_hat = iimage.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    isamples = plan.f.copy()*pulsefac

    pimage = make_p_image(imtuple, pol_trans)
    plan.f_hat = pimage.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    psamples = plan.f.copy()*pulsefac

    #compute chi^2
    msamples = psamples/isamples
    chisq = np.sum(np.abs((m - msamples))**2/(sigmam**2)) / (2*len(m))   
    return chisq

def chisqgrad_m_nfft(imtuple, A, m, sigmam, pol_trans=True,pol_solve=(0,1,1)):
    """Polarimetric ratio chi-squared gradient
    """
    iimage = imtuple[0]
    pimage = make_p_image(imtuple, pol_trans)

    
    #get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    #compute uniform --> nonuniform transform
    iimage = imtuple[0]
    plan.f_hat = iimage.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    isamples = plan.f.copy()*pulsefac

    pimage = make_p_image(imtuple, pol_trans)
    plan.f_hat = pimage.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    psamples = plan.f.copy()*pulsefac

    zeros =  np.zeros(len(iimage))

    if pol_trans:

        mimage = imtuple[1]
        chiimage = imtuple[2]
        
        msamples = psamples/isamples
        mdiff_vec = (-1./len(m))*(m - msamples) / (isamples.conj() * sigmam**2) * pulsefac.conj()
        plan.f = mdiff_vec
        plan.adjoint()
        mpart = (plan.f_hat.copy().T).reshape(nfft_info.xdim*nfft_info.ydim)

        if pol_solve[0]!=0: #TODO -- not right??
            plan.f = mdiff_vec * msamples.conj() 
            plan.adjoint()
            mpart2 = (plan.f_hat.copy().T).reshape(nfft_info.xdim*nfft_info.ydim)

            gradi = (np.real(mimage * np.exp(-2j*chiimage) * mpart) - np.real(mpart2))

        else:
            gradi = zeros

        if pol_solve[1]!=0:
            gradm = np.real(iimage*np.exp(-2j*chiimage) * mpart) 
        else:
            gradm = zeros

        if pol_solve[2]!=0:
            gradchi = 2 * np.imag(pimage.conj() * mpart)
        else:
            gradchi = zeros

        # output tuple can be length 3 or 4 depending on V
        if len(imtuple)==4:
            gradout = np.array((gradi, gradm, gradchi, zeros))        
        else:
            gradout = np.array((gradi, gradm, gradchi))
            

    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_trans)

    return gradout


def chisq_pbs_nfft(imtuple, A, bis_p, sigma, pol_trans=True):
    """Polarimetric bispectrum chi-squared
    """
    
    #get nfft objects
    nfft_info1 = A[0]
    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac

    nfft_info2 = A[1]
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac

    nfft_info3 = A[2]
    plan3 = nfft_info3.plan
    pulsefac3 = nfft_info3.pulsefac

    #compute uniform --> nonuniform transforms
    pimage = make_p_image(imtuple, pol_trans)

    plan1.f_hat = pimage.copy().reshape((nfft_info1.ydim,nfft_info1.xdim)).T
    plan1.trafo()
    samples1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = pimage.copy().reshape((nfft_info2.ydim,nfft_info2.xdim)).T
    plan2.trafo()
    samples2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = pimage.copy().reshape((nfft_info3.ydim,nfft_info3.xdim)).T
    plan3.trafo()
    samples3 = plan3.f.copy()*pulsefac3

    #compute chi^2
    bisamples_p = samples1*samples2*samples3
    chisq =  np.sum(np.abs((bis_p - bisamples_p)/sigma)**2) / (2.*len(bis_p))

    return chisq

def chisqgrad_pbs(imtuple, Amatrices, bis_p, sigma, pol_trans=True,pol_solve=(0,1,1)):
    """Polarimetric bispectrum chi-squared gradient 
    """

    #get nfft objects
    nfft_info1 = A[0]
    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac

    nfft_info2 = A[1]
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac

    nfft_info3 = A[2]
    plan3 = nfft_info3.plan
    pulsefac3 = nfft_info3.pulsefac

    #compute uniform --> nonuniform transforms
    pimage = make_p_image(imtuple, pol_trans)

    plan1.f_hat = pimage.copy().reshape((nfft_info1.ydim,nfft_info1.xdim)).T
    plan1.trafo()
    v1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = pimage.copy().reshape((nfft_info2.ydim,nfft_info2.xdim)).T
    plan2.trafo()
    v2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = pimage.copy().reshape((nfft_info3.ydim,nfft_info3.xdim)).T
    plan3.trafo()
    v3 = plan3.f.copy()*pulsefac3

    # gradient vec  for  adjoint fft
    bisamples_p = v1*v2*v3
    wdiff = (-1./len(bis_p)) * ((bis_p - bisamples_p).conj()) / (sigma**2)   
    pt1 = wdiff * (v2 * v3).conj() * pulsefac1.conj()
    pt2 = wdiff * (v1 * v3).conj() * pulsefac2.conj()
    pt3 = wdiff * (v1 * v2).conj() * pulsefac3.conj()
   
    plan1.f = pt1
    plan1.adjoint()
    out1 = (plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim)

    plan2.f = pt2
    plan2.adjoint()
    out2 = (plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim)

    plan3.f = pt3
    plan3.adjoint()
    out3 = (plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim)

    ptsum = out1 + out2 + out3

    if pol_trans:
        iimage = imtuple[0]
        mimage = imtuple[1]
        chiimage = imtuple[2]
        
        if pol_solve[0]!=0:
            gradi = np.real(ptsum * mimage * np.exp(2j*chiimage)) 
        else:
            gradi = zeros
        if pol_solve[1]!=0:
            gradm = np.real(ptsum * iimage * np.exp(2j*chiimage)) 
        else:
            gradm = zeros
        if pol_solve[2]!=0:
            gradchi = -2 * np.imag(ptsum * pimage) 
        else:
            gradchi = zeros

        # output tuple can be length 3 or 4 depending on V
        if len(imtuple)==4:
            gradout = np.array((gradi, gradm, gradchi, zeros))        
        else:
            gradout = np.array((gradi, gradm, gradchi))
            

    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_trans)

    return gradout

# stokes v
def chisq_vvis_nfft(imtuple, A, v, sigmav, pol_trans=True):
    """V visibility chi-squared
    """

    #get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    #compute uniform --> nonuniform transform
    vimage = make_v_image(imtuple, pol_trans)
    plan.f_hat = vimage.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    vsamples = plan.f.copy()*pulsefac

    #compute chi^2
    chisq =  np.sum(np.abs((v - vsamples))**2/(sigmav**2)) / (2*len(v))   

    return chisq

def chisqgrad_vvis_nfft(imtuple, A, v, sigmav, pol_trans=True,pol_solve=(0,1,1)):
    """V visibility chi-squared gradient
    """
    
    #get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    #compute uniform --> nonuniform transform
    iimage = imtuple[0]
    vimage = make_v_image(imtuple, pol_trans)
    plan.f_hat = vimage.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    vsamples = plan.f.copy()*pulsefac


    vdiff_vec = (-1.0/len(v)) * (v - vsamples) / (sigmav**2) * pulsefac.conj()
    plan.f = vdiff_vec
    plan.adjoint()
    vpart = (plan.f_hat.copy().T).reshape(nfft_info.xdim*nfft_info.ydim)

    zeros =  np.zeros(len(iimage))
        
    if pol_trans:

        vfimage = imtuple[3] #fractional
        
        if pol_solve[0]!=0:
            gradi = np.real(vfimage*vpart)
        else:
            gradi = zeros

        if pol_solve[3]!=0:
            gradv = np.real(iimage*vpart) 
        else:
            gradv = zeros


        gradout = np.array((gradi, zeros, zeros, gradv))

    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_trans)

    return gradout

##################################################################################################
# Polarimetric Entropy and Gradient Functions
##################################################################################################

def sm(imtuple, flux, pol_trans=True, 
       norm_reg=NORM_REGULARIZER):
    """I log m entropy
    """
    if norm_reg: norm = flux
    else: norm = 1

    iimage = imtuple[0]    
    mimage = make_m_image(imtuple, pol_trans) 
    S = -np.sum(iimage * np.log(mimage))
    return S/norm

def smgrad(imtuple, flux, pol_trans=True,pol_solve=(0,1,1),
           norm_reg=NORM_REGULARIZER):
    """I log m entropy gradient
    """

    if norm_reg: norm = flux
    else: norm = 1

    iimage = imtuple[0]    
    zeros  =  np.zeros(len(iimage))
    mimage = make_m_image(imtuple, pol_trans) 

    if pol_trans:

        
        if pol_solve[0]!=0:
            gradi = -np.log(mimage)
        else:
            gradi = zeros
            
        if pol_solve[1]!=0:
            gradm = -iimage / mimage
        else:
            gradm = zeros
        
        gradchi = zeros

        if len(imtuple)==4:
            out = np.array((gradi, gradm, gradchi,zeros))
        else:        
            out = np.array((gradi, gradm, gradchi))
    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_trans)

    return np.array(out)/norm
          
def shw(imtuple, flux, pol_trans=True, norm_reg=NORM_REGULARIZER):
    """Holdaway-Wardle polarimetric entropy
    """
    
    if norm_reg: norm = flux
    else: norm = 1

    iimage = imtuple[0]    
    mimage = make_m_image(imtuple, pol_trans) 
    S = -np.sum(iimage * (((1+mimage)/2) * np.log((1+mimage)/2) + ((1-mimage)/2) * np.log((1-mimage)/2)))
    return S/norm

def shwgrad(imtuple, flux, pol_trans=True,pol_solve=(0,1,1),
            norm_reg=NORM_REGULARIZER):
    """Gradient of the Holdaway-Wardle polarimetric entropy
    """
    if norm_reg: norm = flux
    else: norm = 1

    iimage = imtuple[0]
    zeros =  np.zeros(len(iimage))
    mimage = make_m_image(imtuple, pol_trans)    
    if pol_trans:

        if pol_solve[0]!=0:
            gradi =  -(((1+mimage)/2) * np.log((1+mimage)/2) + ((1-mimage)/2) * np.log((1-mimage)/2))
        else:
            gradi = zeros
            
        if pol_solve[1]!=0:
            gradm = -iimage * np.arctanh(mimage)
        else:
            gradm = zeros
            
        gradchi = zeros
        
        if len(imtuple)==4:
            out = np.array((gradi, gradm, gradchi,zeros))
        else:        
            out = np.array((gradi, gradm, gradchi))
    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_trans)

    return np.array(out)/norm

def stv_pol(imtuple, flux, nx, ny, psize, pol_trans=True, 
            norm_reg=NORM_REGULARIZER, beam_size=None):
    """Total variation of I*m*exp(2Ichi)"""
    
    if beam_size is None: beam_size = psize
    if norm_reg: norm = flux*psize / beam_size
    else: norm = 1

    pimage = make_p_image(imtuple, pol_trans)
    im = pimage.reshape(ny, nx)

    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    S = -np.sum(np.sqrt(np.abs(im_l1 - im)**2 + np.abs(im_l2 - im)**2))
    return S/norm

def stv_pol_grad(imtuple, flux, nx, ny, psize, pol_trans=True, pol_solve=(0,1,1), 
             norm_reg=NORM_REGULARIZER, beam_size=None):
    """Total variation entropy gradient"""

    if beam_size is None: beam_size = psize
    if norm_reg: norm = flux*psize / beam_size
    else: norm = 1

    iimage = imtuple[0]   
    zeros =  np.zeros(len(iimage)) 
    pimage = make_p_image(imtuple, pol_trans)

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

    if pol_trans:

        # dS/dI Numerators
        if pol_solve[0]!=0:
            m1 = 2*np.abs(im*im) - np.abs(im*im_l1)*np.cos(2*(np.angle(im_l1) - np.angle(im))) - np.abs(im*im_l2)*np.cos(2*(np.angle(im_l2) - np.angle(im)))
            m2 = np.abs(im*im) - np.abs(im*im_r1)*np.cos(2*(np.angle(im) - np.angle(im_r1)))
            m3 = np.abs(im*im) - np.abs(im*im_r2)*np.cos(2*(np.angle(im) - np.angle(im_r2)))
            igrad = -(1./iimage)*(m1/d1 + m2/d2 + m3/d3).flatten()
        else:
            igrad = zeros

        # dS/dm numerators
        if pol_solve[1]!=0:
            m1 = 2*np.abs(im) - np.abs(im_l1)*np.cos(2*(np.angle(im_l1) - np.angle(im))) - np.abs(im_l2)*np.cos(2*(np.angle(im_l2) - np.angle(im)))
            m2 = np.abs(im) - np.abs(im_r1)*np.cos(2*(np.angle(im) - np.angle(im_r1)))
            m3 = np.abs(im) - np.abs(im_r2)*np.cos(2*(np.angle(im) - np.angle(im_r2)))
            mgrad = -iimage*(m1/d1 + m2/d2 + m3/d3).flatten()
        else: 
            mgrad=zeros

        # dS/dchi numerators
        if pol_solve[2]!=0:
            c1 = -2*np.abs(im*im_l1)*np.sin(2*(np.angle(im_l1) - np.angle(im))) - 2*np.abs(im*im_l2)*np.sin(2*(np.angle(im_l2) - np.angle(im)))
            c2 = 2*np.abs(im*im_r1)*np.sin(2*(np.angle(im) - np.angle(im_r1)))
            c3 = 2*np.abs(im*im_r2)*np.sin(2*(np.angle(im) - np.angle(im_r2)))
            chigrad = -(c1/d1 + c2/d2 + c3/d3).flatten()
        else:
            chigrad = zeros

        if len(imtuple)==4:
            out = np.array((igrad, mgrad, chigrad,zeros))
        else:        
            out = np.array((igrad, mgrad, chigrad))

    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_trans)

    return out/norm

# circular polarization
# TODO check!!
def sl1v(imtuple, flux, pol_trans=True, norm_reg=NORM_REGULARIZER):
    """L1 norm regularizer on V
    """
    if norm_reg: norm = flux
    else: norm = 1

    vimage = make_v_image(imtuple, pol_trans) 
    l1 = -np.sum(np.abs(vimage))
    return l1/norm


def sl1vgrad(imtuple, flux, pol_trans=True, pol_solve=(0,0,0,1),  norm_reg=NORM_REGULARIZER):
    """L1 norm gradient
    """
    if norm_reg: norm = flux
    else: norm = 1
     
    iimage = imtuple[0]    
    zeros  =  np.zeros(len(iimage))
    vimage = make_v_image(imtuple, pol_trans) 
    grad = -np.sign(vimage)
    
    if pol_trans:

        # dS/dI Numerators
        if pol_solve[0]!=0:
            igrad = (vimage/iimage)*grad
        else:
            igrad = zeros

        # dS/dm numerators
        if pol_solve[3]!=0:
            vgrad = iimage*grad
        else: 
            vgrad=zeros


        out = np.array((igrad, zeros, zeros, vgrad))
    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_trans)            
    
    return out/norm
    
def sl2v(imtuple, flux, pol_trans=True, norm_reg=NORM_REGULARIZER):
    """L1 norm regularizer on V
    """
    if norm_reg: norm = flux
    else: norm = 1

    iimage = imtuple[0]    
    vimage = make_v_image(imtuple, pol_trans) 
    l2 = -np.sum((vimage)**2)
    return l2/norm


def sl2vgrad(imtuple, flux, pol_trans=True, pol_solve=(0,0,0,1), norm_reg=NORM_REGULARIZER):
    """L2 norm gradient
    """
    if norm_reg: norm = flux
    else: norm = 1

    iimage = imtuple[0]    
    zeros  =  np.zeros(len(iimage))
    vimage = make_v_image(imtuple, pol_trans) 
    grad = -2*vimage

    if pol_trans:

        # dS/dI Numerators
        if pol_solve[0]!=0:
            igrad = (vimage/iimage)*grad
        else:
            igrad = zeros

        # dS/dm numerators
        if pol_solve[3]!=0:
            vgrad = iimage*grad
        else: 
            vgrad=zeros


        out = np.array((igrad, zeros, zeros, vgrad))
    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_trans)            
            
    return out/norm    

def stv_v(imtuple, flux, nx, ny, psize, pol_trans=True, 
          norm_reg=NORM_REGULARIZER, beam_size=None, epsilon=0.):
    """Total variation of I*vfrac"""
    
    if beam_size is None: beam_size = psize
    if norm_reg: norm = flux*psize / beam_size
    else: norm = 1

    vimage = make_v_image(imtuple, pol_trans)
    im = vimage.reshape(ny, nx)

    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    S = -np.sum(np.sqrt(np.abs(im_l1 - im)**2 + np.abs(im_l2 - im)**2+epsilon))
    return S/norm

def stv_v_grad(imtuple, flux, nx, ny, psize, pol_trans=True, pol_solve=(0,0,0,1), 
             norm_reg=NORM_REGULARIZER, beam_size=None, epsilon=0.):
    """Total variation gradient"""

    if beam_size is None: beam_size = psize
    if norm_reg: norm = flux*psize / beam_size
    else: norm = 1

    iimage = imtuple[0]   
    zeros =  np.zeros(len(iimage)) 
    vimage = make_v_image(imtuple, pol_trans)

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
    
    if pol_trans:

        # dS/dI Numerators
        if pol_solve[0]!=0:
            igrad = (vimage/iimage)*grad
        else:
            igrad = zeros

        # dS/dm numerators
        if pol_solve[3]!=0:
            vgrad = iimage*grad
        else: 
            vgrad=zeros


        out = np.array((igrad, zeros, zeros, vgrad))


    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_trans)

    return out/norm

def stv2_v(imtuple, flux, nx, ny, psize, pol_trans=True, 
           norm_reg=NORM_REGULARIZER, beam_size=None):
    """Squared Total variation of I*vfrac
    """
    
    if beam_size is None:
        beam_size = psize
    if norm_reg:
        norm = psize**4 * flux**2 / beam_size**4
    else:
        norm = 1

    vimage = make_v_image(imtuple, pol_trans)
    im = vimage.reshape(ny, nx)
    
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    out = -np.sum((im_l1 - im)**2 + (im_l2 - im)**2)
    return out/norm

def stv2_v_grad(imtuple, flux, nx, ny, psize, pol_trans=True, pol_solve=(0,0,0,1), 
               norm_reg=NORM_REGULARIZER, beam_size=None):
    """Squared Total variation gradient
    """
    if beam_size is None:
        beam_size = psize
    if norm_reg:
        norm = psize**4 * flux**2 / beam_size**4
    else:
        norm = 1

    iimage = imtuple[0]   
    zeros =  np.zeros(len(iimage)) 
    vimage = make_v_image(imtuple, pol_trans)
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
    
    if pol_trans:

        # dS/dI Numerators
        if pol_solve[0]!=0:
            igrad = (vimage/iimage)*grad
        else:
            igrad = zeros

        # dS/dm numerators
        if pol_solve[3]!=0:
            vgrad = iimage*grad
        else: 
            vgrad=zeros


        out = np.array((igrad, zeros, zeros, vgrad))


    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_trans)

    return out/norm

##################################################################################################
# Embedding and Chi^2 Data functions
##################################################################################################
def embed_pol(imtuple, mask, clipfloor=0., randomfloor=False):
    """Embeds a polarimetric image tuple into the size of boolean embed mask
    """

    out0=np.zeros(len(mask))
    out1=np.zeros(len(mask))
    out2=np.zeros(len(mask))
    if len(imtuple)==4: out3=np.zeros(len(mask))
    
    # Here's a much faster version than before 
    out0[mask.nonzero()] = imtuple[0]
    out1[mask.nonzero()] = imtuple[1]
    out2[mask.nonzero()] = imtuple[2]
    if len(imtuple)==4: 
        out3[mask.nonzero()] = imtuple[3]
        
    if clipfloor != 0.0:       
        out0[(mask-1).nonzero()] = clipfloor
        out1[(mask-1).nonzero()] = 0
        out2[(mask-1).nonzero()] = 0
        if len(imtuple)==4: out3[(mask-1).nonzero()] = 0
        if randomfloor: # prevent total variation gradient singularities
            out0[(mask-1).nonzero()] *= np.abs(np.random.normal(size=len((mask-1).nonzero())))   
            
    if len(imtuple)==4:
        out = (out0, out1, out2, out3)
    else:
        out = (out0, out1, out2)
          
    return out

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


def chisqdata_pbs(Obsdata, Prior, mask):
    """return the bispectra, sigmas, and fourier matrices for and observation, prior, mask
    """

    biarr = Obsdata.bispectra(mode="all", vtype='rlvis', count="min") #TODO CONJ??
    uv1 = np.hstack((biarr['u1'].reshape(-1,1), biarr['v1'].reshape(-1,1)))
    uv2 = np.hstack((biarr['u2'].reshape(-1,1), biarr['v2'].reshape(-1,1)))
    uv3 = np.hstack((biarr['u3'].reshape(-1,1), biarr['v3'].reshape(-1,1)))
    bi = biarr['bispec']
    sigma = biarr['sigmab']

    A3 = (ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse, mask=mask)
         )

    return (bi, sigma, A3)

def chisqdata_pbs_nfft(Obsdata, Prior, mask):
    """return the bispectra, sigmas, and fourier matrices for and observation, prior, mask
    """

    # unpack keyword args
    fft_pad_factor = kwargs.get('fft_pad_factor',FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', GRIDDER_P_RAD_DEFAULT)

    # unpack data
    biarr = Obsdata.bispectra(mode="all", vtype='rlvis', count="min")
    uv1 = np.hstack((biarr['u1'].reshape(-1,1), biarr['v1'].reshape(-1,1)))
    uv2 = np.hstack((biarr['u2'].reshape(-1,1), biarr['v2'].reshape(-1,1)))
    uv3 = np.hstack((biarr['u3'].reshape(-1,1), biarr['v3'].reshape(-1,1)))
    bi = biarr['bispec']
    sigma = biarr['sigmab']

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv1)
    A2 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv2)
    A3 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv3)
    A = [A1,A2,A3]

    return (bi, sigma, A)

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

#TODO this only works for pol_trans == "amp_cphase"
def plot_m(imtuple, Prior, nit, chi2_dict, **kwargs):

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
    im = imtuple[0]
    mim = imtuple[1]
    chiim = imtuple[2]
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
    q = qimage(im, mim, chiim)
    u = uimage(im, mim, chiim)
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


