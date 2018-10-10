# obsdata.py
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
# make this work!

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
from  scipy.special import jv

import ehtim.image as image
from . import linearize_energy as le

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

from IPython import display

##################################################################################################
# Constants & Definitions
##################################################################################################

NORM_REGULARIZER = False #ANDREW TODO change this default in the future

MAXLS = 100 # maximum number of line searches in L-BFGS-B
NHIST = 100 # number of steps to store for hessian approx
MAXIT = 100 # maximum number of iterations
STOP = 1.e-100 # convergence criterion

DATATERMS = ['pvis','m','pbs']
REGULARIZERS = ['msimple', 'hw', 'ptv']

GRIDDER_P_RAD_DEFAULT = 2
GRIDDER_CONV_FUNC_DEFAULT = 'gaussian'
FFT_PAD_DEFAULT = 2
FFT_INTERP_DEFAULT = 3

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
                    pol_prim="amp_phase", pol_solve = (0,1,1),
                    d1='pvis', d2=False, 
                    s1='msimple', s2=False,
                    alpha_d1=100, alpha_d2=100,
                    alpha_s1=1, alpha_s2=1,
                    **kwargs):

    """Run a general interferometric imager.

       Args:
           Obsdata (Obsdata): The Obsdata object with polarimetric VLBI data
           Prior (Image): The Image object with the prior image
           InitIm (Image): The Image object with the initial image for the minimization

           pol_prim (str): "amp_phase" I,m,chi "qu" for IQU, "qu_frac" for I,Q/I,U/I
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
           fft_interp (int): Interpolation order for sampling the FFT
           grid_conv_func (str): The convolving function for gridding; options are 'gaussian', 'pill', and 'spheroidal'
           grid_prad (float): The pixel radius for the convolving function in gridding for FFTs

           clipfloor (float): The Stokes I Jy/pixel level above which prior image pixels are varied
           grads (bool): If True, analytic gradients are used

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
    #mcv_transform = kwargs.get('logim',True) #transform m,chi with mcv (see below)
    norm_init = kwargs.get('norm_init',False)
    show_updates = kwargs.get('show_updates',True)

    # Make sure data and regularizer options are ok
    if not d1 and not d2:
        raise Exception("Must have at least one data term!")
    if not s1 and not s2:
        raise Exception("Must have at least one regularizer term!")
    if (not ((d1 in DATATERMS) or d1==False)) or (not ((d2 in DATATERMS) or d2==False)):
        raise Exception("Invalid data term: valid data terms are: " + ' '.join(DATATERMS))
    if (not ((s1 in REGULARIZERS) or s1==False)) or (not ((s2 in REGULARIZERS) or s2==False)):
        raise Exception("Invalid regularizer: valid regularizers are: " + ' '.join(REGULARIZERS))
    if (Prior.psize != InitIm.psize) or (Prior.xdim != InitIm.xdim) or (Prior.ydim != InitIm.ydim):
        raise Exception("Initial image does not match dimensions of the prior image!")
    if (ttype!="direct"):
        raise Exception("FFTs and NFFTs not yet implemented in polarimetric imaging!")
    if (pol_prim!="amp_phase"):
        raise Exception("Only amp_phase pol_prim currently supported!")
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
    if pol_prim ==  "amp_phase":
        if len(InitIm.qvec) and (np.any(InitIm.qvec!=0) or np.any(InitIm.uvec!=0)): #TODO right? or should it be if any=0
            init1 = (np.abs(InitIm.qvec + 1j*InitIm.uvec) / InitIm.imvec)#[embed_mask]
            init2 = (np.arctan2(InitIm.uvec, InitIm.qvec) / 2.0)#[embed_mask]
        else:
            raise Exception("not ready for random!")
            # !AC TODO get the actual zero baseline pol. frac from the data!??
            print("No polarimetric image in the initial image!")
            #init1 = 0.2 * (np.ones(len(iimage)))# + 1e-10 * np.random.rand(len(iimage)))
            #init2 = np.zeros(len(iimage)) + 1e-2 #* np.random.rand(len(iimage))
            init1 = 0.2 * (np.ones(len(iimage)) + 1e-2 * np.random.rand(len(iimage)))
            init2 = np.zeros(len(iimage)) + 1e-2 * np.random.rand(len(iimage))
            #init1 = (0.2 * (np.ones(len(iimage)) + 1e-1 * np.random.rand(nimage)))[embed_mask]
            #init2 = (.5*np.random.rand(nimage))[embed_mask]

        # Change of variables    
        inittuple = (iimage, init1, init2)
        xtuple =  mcv_r(inittuple)

    # Get data and fourier matrices for the data terms
    (data1, sigma1, A1) = polchisqdata(Obsdata, Prior, embed_mask, d1, **kwargs)
    (data2, sigma2, A2) = polchisqdata(Obsdata, Prior, embed_mask, d2, **kwargs)

    # Define the chi^2 and chi^2 gradient
    def chisq1(imtuple):
        return polchisq(imtuple, A1, data1, sigma1, d1, ttype=ttype, mask=embed_mask, pol_prim=pol_prim)

    def chisq1grad(imtuple):
        return polchisqgrad(imtuple, A1, data1, sigma1, d1, ttype=ttype, mask=embed_mask, pol_prim=pol_prim, pol_solve=pol_solve)

    def chisq2(imtuple):
        return polchisq(imtuple, A2, data2, sigma2, d2, ttype=ttype, mask=embed_mask, pol_prim=pol_prim)

    def chisq2grad(imtuple):
        return polchisqgrad(imtuple, A2, data2, sigma2, d2, ttype=ttype, mask=embed_mask, pol_prim=pol_prim,pol_solve=pol_solve)

    # Define the regularizer and regularizer gradient
    def reg1(imtuple):
        return polregularizer(imtuple, embed_mask, Prior.xdim, Prior.ydim, Prior.psize, s1, pol_prim=pol_prim)

    def reg1grad(imtuple):
        return polregularizergrad(imtuple, embed_mask, Prior.xdim, Prior.ydim, Prior.psize, s1, pol_prim=pol_prim,pol_solve=pol_solve)

    def reg2(imtuple):
        return polregularizer(imtuple, embed_mask, Prior.xdim, Prior.ydim, Prior.psize, s2, pol_prim=pol_prim)

    def reg2grad(imtuple):
        return  polregularizergrad(imtuple, embed_mask, Prior.xdim, Prior.ydim, Prior.psize, s2, pol_prim=pol_prim,pol_solve=pol_solve)


    # Define the objective function and gradient
    def objfunc(allvec):
        # unpack allvec into image tuple
        cvtuple = unpack_poltuple(allvec, xtuple, nimage, pol_solve)

        # change of variables
        if pol_prim == "amp_phase":
            imtuple = mcv(cvtuple)
        else:
            imtuple = cvtuple

        datterm = alpha_d1 * (chisq1(imtuple) - 1) + alpha_d2 * (chisq2(imtuple) - 1)
        regterm = alpha_s1 * reg1(imtuple) + alpha_s2 * reg2(imtuple)

        return datterm + regterm

    def objgrad(allvec):
        # unpack allvec into image tuple
        cvtuple = unpack_poltuple(allvec, xtuple, nimage, pol_solve)

        # change of variables
        if pol_prim == "amp_phase":
            imtuple = mcv(cvtuple)
        else:
            imtuple = cvtuple

        datterm = alpha_d1 * chisq1grad(imtuple) + alpha_d2 * chisq2grad(imtuple)
        regterm = alpha_s1 * reg1grad(imtuple) + alpha_s2 * reg2grad(imtuple)
        gradarr = datterm + regterm

        # chain rule
        if pol_prim == "amp_phase":
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
        print(nit)
        cvtuple = unpack_poltuple(im_step, xtuple, nimage, pol_solve)

        if pol_prim == "amp_phase":
            imtuple = mcv(cvtuple)  #change of variables
        else:
            imtuple = cvtuple

        if show_updates:
            chi2_1 = chisq1(imtuple)
            chi2_2 = chisq2(imtuple)
            s_1 = reg1(imtuple)
            s_2 = reg2(imtuple)
            #if np.any(np.invert(embed_mask)): 
            #    imtuple = embed(imtuple, embed_mask)
            #plot_m(imtuple, Prior, nit, chi2_1, chi2_2)
            print("i: %d chi2_1: %0.2f chi2_2: %0.2f s_1: %0.2f s_2: %0.2f" % (nit, chi2_1, chi2_2,s_1,s_2))
        nit += 1

    # Print stats
    print("Initial S_1: %f S_2: %f" % (reg1(inittuple), reg2(inittuple)))
    print("Initial Chi^2_1: %f Chi^2_2: %f" % (chisq1(inittuple), chisq2(inittuple)))
    if d1 in DATATERMS:
        print("Total Data 1: ", (len(data1)))
    if d2 in DATATERMS:
        print("Total Data 2: ", (len(data2)))
    print("Total Pixel #: ", (len(Prior.imvec)))
    print("Clipped Pixel #: ", nimage)
    print()

    # Plot Initial
    xinit = pack_poltuple(xtuple, pol_solve)
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
    print(res.message)


    tstop = time.time()
    
    # Format output
    outcv = unpack_poltuple(res.x, xtuple, nimage, pol_solve)
    if pol_prim == "amp_phase":
        out = mcv(outcv)  #change of variables
    else:
        out = outcv

    #if np.any(np.invert(embed_mask)): 
    #    out = embed(out, embed_mask) #embed

    iimage = out[0]
    qimage = make_q_image(out, pol_prim)
    uimage = make_u_image(out, pol_prim)

    outim = image.Image(iimage.reshape(Prior.ydim, Prior.xdim), Prior.psize,
                         Prior.ra, Prior.dec, rf=Prior.rf, source=Prior.source,
                         mjd=Prior.mjd, pulse=Prior.pulse)
    outim.add_qu(qimage.reshape(Prior.ydim, Prior.xdim), uimage.reshape(Prior.ydim, Prior.xdim))

    # Print stats
    print("time: %f s" % (tstop - tstart))
    #print("J: %f" % res.fun)
    #print("Final Chi^2_1: %f Chi^2_2: %f" % (chisq1(out[embed_mask]), chisq2(out[embed_mask])))
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

        return vec


def unpack_poltuple(polvec, inittuple, nimage, pol_solve = (0,1,1)):       
        """unpack polvec into image tuple, 
           replaces quantities not iterated with  initial values
        """
        (init0, init1, init2) = inittuple

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

        if pol_solve[0] == 0:
            im2 = init2
        else:
            im2 = polvec[imct*nimage:(imct+1)*nimage]
            imct += 1
        return (im0, im1, im2)

def make_p_image(imtuple, pol_prim="amp_phase"):
    """construct a polarimetric image P = Q + iU
    """

    if pol_prim=="amp_phase":
        pimage = imtuple[0] * imtuple[1] * np.exp(2j*imtuple[2])
    elif pol_prim=="qu":
        pimage = imtuple[1] + 1j*imtuple[2]
    elif pol_prim=="qu_frac":
        pimage = imtuple[0] * (imtuple[1] + 1j*imtuple[2])
    else:
        raise Exception("polarimetric representation %s not recognized in make_p_image!"%pol_prim)
    return pimage

def make_m_image(imtuple, pol_prim="amp_phase"):
    """construct a polarimetric ratrio image abs(P/I) = abs(Q + iU)/I
    """

    if pol_prim=="amp_phase":
        mimage = imtuple[1]
    elif pol_prim=="qu":
        mimage = np.abs((imtuple[1] + 1j*imtuple[2])/imtuple[0])
    elif pol_prim=="qu_frac":
        mimage = np.abs(imtuple[1] + 1j*imtuple[2])
    else:
        raise Exception("polarimetric representation %s not recognized in make_m_image!"%pol_prim)
    return mimage

def make_chi_image(imtuple, pol_prim="amp_phase"):
    """construct a polarimetric angle image 
    """

    if pol_prim=="amp_phase":
        mimage = imtuple[2]
    elif pol_prim=="qu":
        mimage = 0.5*np.angle((imtuple[1] + 1j*imtuple[2])/imtuple[0])
    elif pol_prim=="qu_frac":
        mimage = 0.5*np.angle(imtuple[1] + 1j*imtuple[2])
    else:
        raise Exception("polarimetric representation %s not recognized in make_chi_image!"%pol_prim)
    return mimage

def make_q_image(imtuple, pol_prim="amp_phase"):
    """construct an image of stokes Q
    """

    if pol_prim=="amp_phase":
        mimage = imtuple[0] * imtuple[1] * np.cos(2*imtuple[2])
    elif pol_prim=="qu":
        mimage = imtuple[1]
    elif pol_prim=="qu_frac":
        mimage = imtuple[1]*imtuple[0]
    else:
        raise Exception("polarimetric representation %s not recognized in make_m_image!"%pol_prim)
    return mimage

def make_u_image(imtuple, pol_prim="amp_phase"):
    """construct an image of stokes U
    """

    if pol_prim=="amp_phase":
        mimage = imtuple[0] * imtuple[1] * np.sin(2*imtuple[2])
    elif pol_prim=="qu":
        mimage = imtuple[2]
    elif pol_prim=="qu_frac":
        mimage =imtuple[2]*imtuple[0]
    else:
        raise Exception("polarimetric representation %s not recognized in make_chi_image!"%pol_prim)
    return mimage

# these change of variables only apply to polarimetric ratios
# !AC In these pol. changes of variables, might be useful to 
# take m -> m/100 by adjusting B (function becomes less steep around m' = 0)
B = .5
def mcv(imtuple):
    """Change of pol. ratio from range (-inf, inf) to (0,1)
    """
    iimage = imtuple[0]
    mimage =  imtuple[1]
    chiimage = imtuple[2]

    mtrans = 0.5 + np.arctan(mimage/B)/np.pi

    out = (iimage, mtrans, chiimage)
    return out

def mcv_r(imtuple):
    """Change of pol. ratio from range (0,1) to (-inf,inf)
    """
    iimage = imtuple[0]
    mimage =  imtuple[1]
    chiimage = imtuple[2]

    mtrans = B*np.tan(np.pi*(mimage - 0.5))

    out = (iimage, mtrans, chiimage)
    return out

def mchain(imtuple):
    """The gradient change of variables, dm/dm'
    """
    iimage = imtuple[0]
    mimage =  imtuple[1]
    chiimage = imtuple[2]

    mchain = 1 / (B*np.pi*(1 + (mimage/B)**2))
    chichain = np.ones(len(chiimage))
    ichain = np.ones(len(iimage))

    out = (ichain, mchain, chichain)
    return np.array(out)

    mchain = 1 / (B*np.pi*(1 + (polimage_cv[0:len(polimage_cv)/2]/B)**2))


##################################################################################################
# Wrapper Functions
##################################################################################################

def polchisq(imtuple, A, data, sigma, dtype, ttype='direct', mask=[], pol_prim="amp_phase"):
    """return the chi^2 for the appropriate dtype
    """

    chisq = 1 
    if not dtype in DATATERMS:
        return chisq
    if ttype not in ['fast','direct','nfft']:
        raise Exception("Possible ttype values are 'fast' and 'direct'!")

    if ttype == 'direct':
        if dtype == 'pvis':
            chisq = chisq_p(imtuple, A, data, sigma, pol_prim)

        elif dtype == 'm':
            chisq = chisq_m(imtuple, A, data, sigma, pol_prim)

        elif dtype == 'pbs':
            chisq = chisq_pbis(imtuple, A, data, sigma, pol_prim)
    
    elif ttype== 'fast':
        raise Exception("FFT not yet implemented in polchisq!")

    elif ttype== 'nfft':
        raise Exception("FFT not yet implemented in polchisq!")


    return chisq

def polchisqgrad(imtuple, A, data, sigma, dtype, ttype='direct', mask=[], pol_prim="amp_phase",pol_solve=(0,1,1)):
    
    """return the chi^2 gradient for the appropriate dtype
    """

    chisqgrad = np.zeros((3,len(imtuple[0])))
    if not dtype in DATATERMS:
        return chisqgrad
    if ttype not in ['fast','direct','nfft']:
        raise Exception("Possible ttype values are 'fast' and 'direct'!")

    if ttype == 'direct':
        if dtype == 'pvis':
            chisqgrad = chisqgrad_p(imtuple, A, data, sigma, pol_prim,pol_solve)

        elif dtype == 'm':
            chisqgrad = chisqgrad_m(imtuple, A, data, sigma, pol_prim,pol_solve)

        elif dtype == 'pbs':
            chisqgrad = chisqgrad_pbs(imtuple, A, data, sigma, pol_prim,pol_solve)
    
    elif ttype== 'fast':
        raise Exception("FFT not yet implemented in polchisqgrad!")

    elif ttype== 'nfft':
        raise Exception("FFT not yet implemented in polchisq!")

    return np.array(chisqgrad)


def polregularizer(imtuple, mask, xdim, ydim, psize, stype, pol_prim="amp_phase"):
    if stype == "msimple":
        reg = -sm(imtuple, pol_prim)
    elif stype == "hw":
        reg = -shw(imtuple, pol_prim)
    elif stype == "ptv":
        if np.any(np.invert(mask)):
            imtuple = embed(imtuple, mask, randomfloor=True)
        reg = -stv_pol(imtuple, xdim, ydim, pol_prim)
    else:
        reg = 0

    return reg

def polregularizergrad(imtuple, mask, xdim, ydim, psize, stype, pol_prim="amp_phase",pol_solve=(0,1,1)):

    if stype == "msimple":
        reggrad = -smgrad(imtuple, pol_prim,pol_solve)
    elif stype == "hw":
        reggrad = -shwgrad(imtuple, pol_prim,pol_solve)
    elif stype == "ptv":
        if np.any(np.invert(mask)):
            imtuple = embed(imtuple, mask, randomfloor=True)
        reggrad = -stv_pol_grad(imtuple, xdim, ydim, pol_prim,pol_solve)
        if np.any(np.invert(mask)):
            reggrad = (reggrad[0][mask],reggrad[1][mask],reggrad[2][mask])
    else:
        reggrad = np.zeros((3,len(imtuple[0])))

    return np.array(reggrad)


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

    elif ttype=='fast':
        raise Exception("FFT not yet implemented in polchisqdata!")
    elif ttype=='nfft':
        raise Exception("NFFT not yet implemented in polchisqdata!")      
        
    return (data, sigma, A)


##################################################################################################
# DFT Chi-squared and Gradient Functions
##################################################################################################

def chisq_p(imtuple, Amatrix, p, sigmap, pol_prim="amp_phase"):
    """Polarimetric ratio chi-squared
    """

    pimage = make_p_image(imtuple, pol_prim)
    psamples = np.dot(Amatrix, pimage) 
    chisq =  np.sum(np.abs((p - psamples))**2/(sigmap**2)) / (2*len(p))   
    return chisq

def chisqgrad_p(imtuple, Amatrix, p, sigmap, pol_prim="amp_phase",pol_solve=(0,1,1)):
    """Polarimetric ratio chi-squared gradient
    """
    

    iimage = imtuple[0]
    pimage = make_p_image(imtuple, pol_prim)
    psamples = np.dot(Amatrix, pimage)
    pdiff = (p - psamples) / (sigmap**2)
    zeros =  np.zeros(len(iimage))
        
    if pol_prim=="amp_phase":

        mimage = imtuple[1]
        chiimage = imtuple[2]
        
        if pol_solve[0]!=0:
            gradi = -np.real(mimage * np.exp(-2j*chiimage) * np.dot(Amatrix.conj().T, pdiff)) / len(p)
        else:
            gradi = zeros

        if pol_solve[1]!=0:
            gradm = -np.real(iimage*np.exp(-2j*chiimage) * np.dot(Amatrix.conj().T, pdiff)) / len(p)
        else:
            gradm = zeros

        if pol_solve[2]!=0:
            gradchi = -2 * np.imag(pimage.conj() * np.dot(Amatrix.conj().T, pdiff)) / len(p)
        else:
            gradchi = zeros

        gradout = (gradi, gradm, gradchi)

    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_prim)

    return np.array(gradout)

def chisq_m(imtuple, Amatrix, m, sigmam, pol_prim="amp_phase"):
    """Polarimetric ratio chi-squared
    """

    iimage = imtuple[0]
    pimage = make_p_image(imtuple, pol_prim)
    msamples = np.dot(Amatrix, pimage) / np.dot(Amatrix, iimage)
    return np.sum(np.abs((m - msamples))**2/(sigmam**2)) / (2*len(m))   

    #chisq = np.sum(np.abs((m - msamples))**2/(sigmam**2)) / (2*len(m))   
    #return chisq

def chisqgrad_m(imtuple, Amatrix, m, sigmam, pol_prim="amp_phase",pol_solve=(0,1,1)):
    """The gradient of the polarimetric ratio chisq
    """
    iimage = imtuple[0]
    isamples = np.dot(Amatrix, iimage)

    pimage = make_p_image(imtuple, pol_prim)
    psamples = np.dot(Amatrix, pimage)
    zeros  =  np.zeros(len(iimage))

    if pol_prim=="amp_phase":

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

        gradout = (gradi, gradm, gradchi)

    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_prim)

    return np.array(gradout)

def chisq_pbs(imtuple, Amatrices, bis_p, sigma, pol_prim="amp_phase"):
    """Polarimetric bispectrum chi-squared
    """
    
    iimage = imtuple[0]
    pimage = make_p_image(imtuple, pol_prim)
    bisamples_p = np.dot(Amatrices[0], pimage) * np.dot(Amatrices[1], pimage) * np.dot(Amatrices[2], pimage)
    chisq =  np.sum(np.abs((bis_p - bisamples_p)/sigma)**2) / (2.*len(bis_p))
    return chisq

def chisqgrad_pbs(imtuple, Amatrices, bis_p, sigma, pol_prim="amp_phase",pol_solve=(0,1,1)):
    """The gradient of the polarimetric bispectrum chisq 
    """
    pimage = make_p_image(imtuple, pol_prim)
    bisamples_p = np.dot(Amatrices[0], pimage) * np.dot(Amatrices[1], pimage) * np.dot(Amatrices[2], pimage)

    wdiff = ((bis_p - bisamples_p).conj()) / (sigma**2)      
    pt1 = wdiff * np.dot(Amatrices[1],pimage) * np.dot(Amatrices[2],pimage)
    pt2 = wdiff * np.dot(Amatrices[0],pimage) * np.dot(Amatrices[2],pimage)
    pt3 = wdiff * np.dot(Amatrices[0],pimage) * np.dot(Amatrices[1],pimage)
    ptsum = np.dot(pt1, Amatrices[0]) + np.dot(pt2, Amatrices[1]) + np.dot(pt3, Amatrices[2])    

    if pol_prim=="amp_phase":
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

        out = (gradi, gradm, gradchi)

    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_prim)

    return np.array(out)

    
##################################################################################################
# Polarimetric Entropy and Gradient Functions
##################################################################################################

def sm(imtuple, pol_prim="amp_phase"):
    """I log m entropy
    """
    iimage = imtuple[0]    
    mimage = make_m_image(imtuple, pol_prim) 
    S = -np.sum(iimage * np.log(mimage))
    return S

def smgrad(imtuple, pol_prim="amp_phase",pol_solve=(0,1,1)):
    """I log m entropy gradient
    """
    iimage = imtuple[0]    
    zeros  =  np.zeros(len(iimage))
    mimage = make_m_image(imtuple, pol_prim) 

    if pol_prim=="amp_phase":
        gradi = zeros
        gradchi = zeros
        if pol_solve[1]!=0:
            gradm = -iimage / mimage
        else:
            gradm = zeros

        out = (gradi, gradm, gradchi)
    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_prim)

    return np.array(out)
          
def shw(imtuple, pol_prim="amp_phase"):
    """Holdaway-Wardle polarimetric entropy
    """
    
    iimage = imtuple[0]    
    mimage = make_m_image(imtuple, pol_prim) 
    S = -np.sum(iimage * (((1+mimage)/2) * np.log((1+mimage)/2) + ((1-mimage)/2) * np.log((1-mimage)/2)))
    return S

def shwgrad(imtuple, pol_prim="amp_phase",pol_solve=(0,1,1)):
    """Gradient of the Holdaway-Wardle polarimetric entropy
    """

    iimage = imtuple[0]
    zeros =  np.zeros(len(iimage))
    mimage = make_m_image(imtuple, pol_prim)    
    if pol_prim=="amp_phase":
        gradi = zeros
        gradchi = zeros
        if pol_solve[1]!=0:
            gradm = -iimage * np.arctanh(mimage)
        else:
            gradm = zeros
        out = (gradi, gradm, gradchi)
    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_prim)

    return np.array(out)

def stv_pol(imtuple, nx, ny, pol_prim="amp_phase"):
    """Total variation of I*m*exp(2Ichi)"""
    
    pimage = make_p_image(imtuple, pol_prim)
    im = pimage.reshape(ny, nx)

    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    S = -np.sum(np.sqrt(np.abs(im_l1 - im)**2 + np.abs(im_l2 - im)**2))
    return S

def stv_pol_grad(imtuple, nx, ny, pol_prim="amp_phase",pol_solve=(0,1,1)):
    """Total variation entropy gradient"""

    iimage = imtuple[0]   
    zeros =  np.zeros(len(iimage)) 
    pimage = make_p_image(imtuple, pol_prim)

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

    if pol_prim=="amp_phase":

        # dS/dI Numerators
        if pol_solve[0]!=0:
            m1 = 2*np.abs(im*im) - np.abs(im*im_l1)*np.cos(2*(np.angle(im_l1) - np.angle(im))) - np.abs(im*im_l2)*np.cos(2*(np.angle(im_l2) - np.angle(im)))
            m2 = np.abs(im*im) - np.abs(im*im_r1)*np.cos(2*(np.angle(im) - np.angle(im_r1)))
            m3 = np.abs(im*im) - np.abs(im*im_r2)*np.cos(2*(np.angle(im) - np.angle(im_r2)))
            igrad = -((1./iimage)*(m1/d1 + m2/d2 + m3/d3)).flatten()
        else:
            igrad = zeros

        # dS/dm numerators
        if pol_solve[1]!=0:
            m1 = 2*np.abs(im) - np.abs(im_l1)*np.cos(2*(np.angle(im_l1) - np.angle(im))) - np.abs(im_l2)*np.cos(2*(np.angle(im_l2) - np.angle(im)))
            m2 = np.abs(im) - np.abs(im_r1)*np.cos(2*(np.angle(im) - np.angle(im_r1)))
            m3 = np.abs(im) - np.abs(im_r2)*np.cos(2*(np.angle(im) - np.angle(im_r2)))
            mgrad = -(iimage*(m1/d1 + m2/d2 + m3/d3)).flatten()
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

        out = (igrad, mgrad, chigrad)

    else:
        raise Exception("polarimetric representation %s not added to pol gradient yet!" % pol_prim)

    return out


##################################################################################################
# Embedding and Chi^2 Data functions
##################################################################################################
def embed(imtuple, mask, clipfloor=0., randomfloor=False):
    """Embeds a polarimetric image tuple into the size of boolean embed mask
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
            out1[(mask-1).nonzero()] = 0 #ANDREW TODO Right? 
            out2[(mask-1).nonzero()] = 0
        else:
            out0[(mask-1).nonzero()] = clipfloor
            out1[(mask-1).nonzero()] = 0
            out2[(mask-1).nonzero()] = 0

    return (out0, out1, out2)

def chisqdata_pvis(Obsdata, Prior, mask):
    """Return the visibilities, sigmas, and fourier matrix for an observation, prior, mask
    """

    data_arr = Obsdata.unpack(['u','v','pvis','psigma'])
    uv = np.hstack((data_arr['u'].reshape(-1,1), data_arr['v'].reshape(-1,1)))
    vis = data_arr['pvis']
    sigma = data_arr['psigma']
    A = ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv, pulse=Prior.pulse, mask=mask)

    return (vis, sigma, A)

def chisqdata_m(Obsdata, Prior, mask):
    """Return the amplitudes, sigmas, and fourier matrix for and observation, prior, mask
    """

    ampdata = Obsdata.unpack(['u','v','m','msigma'])
    uv = np.hstack((ampdata['u'].reshape(-1,1), ampdata['v'].reshape(-1,1)))
    m = ampdata['m']
    sigmam = ampdata['msigma']
    A = ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv, pulse=Prior.pulse, mask=mask)

    return (m, sigmam, A)

def chisqdata_pbs(Obsdata, Prior, mask):
    """return the bispectra, sigmas, and fourier matrices for and observation, prior, mask
    """

    biarr = Obsdata.bispectra(mode="all", vtype='rlvis', count="min")
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

##################################################################################################
# Plotting
##################################################################################################

#TODO this only works for pol_prim == "amp_cphase"
def plot_m(imtuple, Prior, nit, chi2, chi2m, pcut=0.05, nvec=15, ipynb=False):
    
    # unpack
    im = imtuple[0]
    mim = imtuple[1]
    chiim = imtuple[2]

    # Mask for low flux points
    thin = int(round(Prior.xdim/nvec))
    mask = im.reshape(Prior.ydim, Prior.xdim) > pcut * np.max(im)
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
    
    # Create figure and title
    plt.ion()
    plt.pause(0.00001)    
    plt.clf()

    plt.suptitle("step: %i  $\chi_{1}^2$: %f   $\chi_{2}^2$: %f" % (nit, chi2, chi2m), fontsize=20)
        
    # Stokes I plot
    plt.subplot(121)
    plt.imshow(im.reshape(Prior.ydim, Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian')
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
    
    # Display
    #plt.draw()
    if ipynb:
        display.clear_output()
        display.display(plt.gcf())   
