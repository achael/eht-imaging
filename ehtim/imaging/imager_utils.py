# imager_utils.py
# General imager functions for total intensity VLBI data
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
STOP = 1.e-8 # convergence criterion

DATATERMS = ['vis', 'bs', 'amp', 'cphase', 'camp', 'logcamp', 'logamp']
REGULARIZERS = ['gs', 'tv', 'tv2','l1', 'lA', 'patch', 'simple', 'compact', 'compact2','rgauss']

nit = 0 # global variable to track the iteration number in the plotting callback

##################################################################################################
# Total Intensity Imager
##################################################################################################
def imager_func(Obsdata, InitIm, Prior, flux,
                   d1='vis', d2=False, d3=False,
                   alpha_d1=100, alpha_d2=100, alpha_d3=100,
                   s1='simple', s2=False, s3=False,
                   alpha_s1=1, alpha_s2=1, alpha_s3=1,
                   alpha_flux=500, alpha_cm=500, 
                   **kwargs):


    """Run a general interferometric imager. Only works directly on the image's primary polarization. 

       Args:
           Obsdata (Obsdata): The Obsdata object with VLBI data
           InitIm (Image): The Image object with the initial image for the minimization
           Prior (Image): The Image object with the prior image
           flux (float): The total flux of the output image in Jy

           d1 (str): The first data term; options are 'vis', 'bs', 'amp', 'cphase', 'camp', 'logcamp'
           d2 (str): The second data term; options are 'vis', 'bs', 'amp', 'cphase', 'camp', 'logcamp'
           d3 (str): The third data term; options are 'vis', 'bs', 'amp', 'cphase', 'camp', 'logcamp'

           s1 (str): The first regularizer; options are 'simple', 'gs', 'tv', 'tv2', 'l1', 'patch','compact','compact2','rgauss'
           s2 (str): The second regularizer; options are 'simple', 'gs', 'tv', 'tv2','l1', 'patch','compact','compact2','rgauss'
           s3 (str): The third regularizer; options are 'simple', 'gs', 'tv', 'tv2','l1', 'patch','compact','compact2','rgauss'

           alpha_d1 (float): The first data term weighting
           alpha_d2 (float): The second data term weighting
           alpha_d2 (float): The third data term weighting

           alpha_s1 (float): The first regularizer term weighting
           alpha_s2 (float): The second regularizer term weighting
           alpha_s3 (float): The third regularizer term weighting

           alpha_flux (float): The weighting for the total flux constraint
           alpha_cm (float): The weighting for the center of mass constraint

           maxit (int): Maximum number of minimizer iterations
           stop (float): The convergence criterion
           clipfloor (float): The Jy/pixel level above which prior image pixels are varied

           grads (bool): If True, analytic gradients are used
           logim (bool): If True, uses I = exp(I') change of variables
           norm_reg (bool): If True, normalizes regularizer terms
           norm_init (bool): If True, normalizes initial image to given total flux
           show_updates (bool): If True, displays the progress of the minimizer

           weighting (str): 'natural' or 'uniform' 
           debias (bool): if True then apply debiasing to amplitudes/closure amplitudes
           systematic_noise (float): a fractional systematic noise tolerance to add to thermal sigmas
           snrcut (float): a  snr cutoff for including data in the chi^2 sum
           beam_size (float): beam size in radians for normalizing the regularizers

           maxset (bool):  if True, use maximal set instead of minimal for closure quantities
           systematic_cphase_noise (float): a value in degrees to add to the closure phase sigmas
           cp_uv_min (float): flag baselines shorter than this before forming closure quantities

           ttype (str): The Fourier transform type; options are 'fast', 'direct', 'nfft'
           fft_pad_factor (float): The FFT will pre-pad the image by this factor x the original size
           order (int): Interpolation order for sampling the FFT
           conv_func (str): The convolving function for gridding; options are 'gaussian', 'pill', and 'cubic'
           p_rad (int): The pixel radius for the convolving function in gridding for FFTs

       Returns:
           Image: Image object with result
    """

    # some kwarg default values
    maxit = kwargs.get('maxit', MAXIT)
    stop = kwargs.get('stop', STOP)
    clipfloor = kwargs.get('clipfloor', 0)
    ttype = kwargs.get('ttype','direct')

    grads = kwargs.get('grads',True)
    logim = kwargs.get('logim',True)
    norm_init = kwargs.get('norm_init',False)
    show_updates = kwargs.get('show_updates',True)

    beam_size = kwargs.get('beam_size',Obsdata.res())
    kwargs['beam_size'] = beam_size

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
    if (InitIm.polrep != Prior.polrep):
        raise Exception("Initial image pol. representation does not match pol. representation of the prior image!")
    if (logim and Prior.pol_prim in ['Q','U','V']):
        raise Exception("Cannot image Stokes Q,U,or V with log image transformation! Set logim=False in imager_func")

    pol = Prior.pol_prim
    print("Generating %s image..." % pol)

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
    if flux > 1.2*maxamp:
        print("Warning! Specified flux is > 120% of maximum visibility amplitude!")
    if flux < .8*maxamp:
        print("Warning! Specified flux is < 80% of maximum visibility amplitude!")

    # Define embedding mask
    embed_mask = Prior.imvec > clipfloor

    # Normalize prior image to total flux and limit imager range to prior values > clipfloor
    if (not norm_init):
        nprior = Prior.imvec[embed_mask]
        ninit = InitIm.imvec[embed_mask]
    else:
        nprior = (flux * Prior.imvec / np.sum((Prior.imvec)[embed_mask]))[embed_mask]
        ninit = (flux * InitIm.imvec / np.sum((InitIm.imvec)[embed_mask]))[embed_mask]

    if len(nprior)==0:
        raise Exception("clipfloor too large: all prior pixels have been clipped!")

    # Get data and fourier matrices for the data terms
    (data1, sigma1, A1) = chisqdata(Obsdata, Prior, embed_mask, d1, pol=pol, **kwargs)
    (data2, sigma2, A2) = chisqdata(Obsdata, Prior, embed_mask, d2, pol=pol, **kwargs)
    (data3, sigma3, A3) = chisqdata(Obsdata, Prior, embed_mask, d3, pol=pol, **kwargs)

    # Define the chi^2 and chi^2 gradient
    def chisq1(imvec):
        return chisq(imvec, A1, data1, sigma1, d1, ttype=ttype, mask=embed_mask)

    def chisq1grad(imvec):
        c = chisqgrad(imvec, A1, data1, sigma1, d1, ttype=ttype, mask=embed_mask)
        return c

    def chisq2(imvec):
        return chisq(imvec, A2, data2, sigma2, d2, ttype=ttype, mask=embed_mask)

    def chisq2grad(imvec):
        c = chisqgrad(imvec, A2, data2, sigma2, d2, ttype=ttype, mask=embed_mask)
        return c

    def chisq3(imvec):
        return chisq(imvec, A3, data3, sigma3, d3, ttype=ttype, mask=embed_mask)

    def chisq3grad(imvec):
        c = chisqgrad(imvec, A3, data3, sigma3, d3, ttype=ttype, mask=embed_mask)
        return c

    # Define the regularizer and regularizer gradient
    def reg1(imvec):
        return regularizer(imvec, nprior, embed_mask, flux, Prior.xdim, Prior.ydim, Prior.psize, s1, **kwargs)

    def reg1grad(imvec):
        return regularizergrad(imvec, nprior, embed_mask, flux, Prior.xdim, Prior.ydim, Prior.psize, s1, **kwargs)

    def reg2(imvec):
        return regularizer(imvec, nprior, embed_mask, flux, Prior.xdim, Prior.ydim, Prior.psize, s2, **kwargs)

    def reg2grad(imvec):
        return regularizergrad(imvec, nprior, embed_mask, flux, Prior.xdim, Prior.ydim, Prior.psize, s2, **kwargs)

    def reg3(imvec):
        return regularizer(imvec, nprior, embed_mask, flux, Prior.xdim, Prior.ydim, Prior.psize, s3, **kwargs)

    def reg3grad(imvec):
        return regularizergrad(imvec, nprior, embed_mask, flux, Prior.xdim, Prior.ydim, Prior.psize, s3, **kwargs)

    # Define constraint functions
    def flux_constraint(imvec):
        return regularizer(imvec, nprior, embed_mask, flux, Prior.xdim, Prior.ydim, Prior.psize, "flux", **kwargs)

    def flux_constraint_grad(imvec):
        return regularizergrad(imvec, nprior, embed_mask, flux, Prior.xdim, Prior.ydim, Prior.psize, "flux", **kwargs)

    def cm_constraint(imvec):
        return regularizer(imvec, nprior, embed_mask, flux, Prior.xdim, Prior.ydim, Prior.psize, "cm", **kwargs)

    def cm_constraint_grad(imvec):
        return regularizergrad(imvec, nprior, embed_mask, flux, Prior.xdim, Prior.ydim, Prior.psize, "cm", **kwargs)

    # Define the objective function and gradient
    def objfunc(imvec):
        if logim: imvec = np.exp(imvec)

        datterm  = alpha_d1 * (chisq1(imvec) - 1) + alpha_d2 * (chisq2(imvec) - 1) + alpha_d3 * (chisq3(imvec) - 1)
        regterm  = alpha_s1 * reg1(imvec) + alpha_s2 * reg2(imvec) + alpha_s3 * reg3(imvec)
        conterm  = alpha_flux * flux_constraint(imvec)  + alpha_cm * cm_constraint(imvec)

        return datterm + regterm + conterm

    def objgrad(imvec):
        if logim: imvec = np.exp(imvec)

        datterm  = alpha_d1 * chisq1grad(imvec) + alpha_d2 * chisq2grad(imvec) + alpha_d3 * chisq3grad(imvec)
        regterm  = alpha_s1 * reg1grad(imvec) + alpha_s2 * reg2grad(imvec) + alpha_s3 * reg3grad(imvec)
        conterm  = alpha_flux * flux_constraint_grad(imvec)  + alpha_cm * cm_constraint_grad(imvec)

        grad = datterm + regterm + conterm

        # chain rule term for change of variables
        if logim: grad *= imvec

        return grad

    # Define plotting function for each iteration
    global nit
    nit = 0
    def plotcur(im_step):
        global nit
        if logim: im_step = np.exp(im_step)
        if show_updates:
            chi2_1 = chisq1(im_step)
            chi2_2 = chisq2(im_step)
            chi2_3 = chisq3(im_step)
            s_1 = reg1(im_step)
            s_2 = reg2(im_step)
            s_3 = reg3(im_step)
            if np.any(np.invert(embed_mask)): im_step = embed(im_step, embed_mask)
            plot_i(im_step, Prior, nit, {d1:chi2_1, d2:chi2_2, d3:chi2_3}, pol=pol)
            print("i: %d chi2_1: %0.2f chi2_2: %0.2f chi2_3: %0.2f s_1: %0.2f s_2: %0.2f s_3: %0.2f" % (nit, chi2_1, chi2_2, chi2_3, s_1, s_2, s_3))
        nit += 1

    # Generate and the initial image
    if logim:
        xinit = np.log(ninit)
    else:
        xinit = ninit

    # Print stats
    print("Initial S_1: %f S_2: %f S_3: %f" % (reg1(ninit), reg2(ninit), reg3(ninit)))
    print("Initial Chi^2_1: %f Chi^2_2: %f Chi^2_3: %f" % (chisq1(ninit), chisq2(ninit), chisq3(ninit)))
    print("Initial Objective Function: %f" % (objfunc(xinit)))

    if d1 in DATATERMS:
        print("Total Data 1: ", (len(data1)))
    if d2 in DATATERMS:
        print("Total Data 2: ", (len(data2)))
    if d3 in DATATERMS:
        print("Total Data 3: ", (len(data3)))

    print("Total Pixel #: ",(len(Prior.imvec)))
    print("Clipped Pixel #: ",(len(ninit)))
    print()
    plotcur(xinit)

    # Minimize
    #stop2 = stop/(np.finfo(float).eps)
    optdict = {'maxiter':maxit, 'ftol':stop, 'maxcor':NHIST,'gtol':stop,'maxls':MAXLS} # minimizer dict params
    tstart = time.time()
    if grads:
        res = opt.minimize(objfunc, xinit, method='L-BFGS-B', jac=objgrad,
                       options=optdict, callback=plotcur)
    else:
        res = opt.minimize(objfunc, xinit, method='L-BFGS-B',
                       options=optdict, callback=plotcur)

    tstop = time.time()

    # Format output
    out = res.x
    if logim: out = np.exp(res.x)
    if np.any(np.invert(embed_mask)): out = embed(out, embed_mask)

    outim = image.Image(out.reshape(Prior.ydim, Prior.xdim),
                        Prior.psize, Prior.ra, Prior.dec,
                        rf=Prior.rf, source=Prior.source,
                        polrep=Prior.polrep, pol_prim=pol, 
                        mjd=Prior.mjd, time=Prior.time, pulse=Prior.pulse)


    # copy over other polarizations
    outim.copy_pol_images(InitIm)

    # Print stats
    print("time: %f s" % (tstop - tstart))
    print("J: %f" % res.fun)
    print("Final Chi^2_1: %f Chi^2_2: %f  Chi^2_3: %f" % (chisq1(out[embed_mask]), chisq2(out[embed_mask]), chisq3(out[embed_mask])))
    print(res.message)

    # Return Image object
    return outim

##################################################################################################
# Wrapper Functions
##################################################################################################

def chisq(imvec, A, data, sigma, dtype, ttype='direct', mask=None):
    """return the chi^2 for the appropriate dtype
    """

    if mask is None: mask=[]
    chisq = 1
    if not dtype in DATATERMS:
        return chisq

    if ttype not in ['fast','direct','nfft']:
        raise Exception("Possible ttype values are 'fast', 'direct'!, 'nfft!'")

    if ttype == 'direct':
        if dtype == 'vis':
            chisq = chisq_vis(imvec, A, data, sigma)
        elif dtype == 'amp':
            chisq = chisq_amp(imvec, A, data, sigma)
        elif dtype == 'logamp':
            chisq = chisq_logamp(imvec, A, data, sigma)
        elif dtype == 'bs':
            chisq = chisq_bs(imvec, A, data, sigma)
        elif dtype == 'cphase':
            chisq = chisq_cphase(imvec, A, data, sigma)
        elif dtype == 'camp':
            chisq = chisq_camp(imvec, A, data, sigma)
        elif dtype == 'logcamp':
            chisq = chisq_logcamp(imvec, A, data, sigma)

    elif ttype== 'fast':
        if len(mask)>0 and np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)

        vis_arr = fft_imvec(imvec, A[0])
        if dtype == 'vis':
            chisq = chisq_vis_fft(vis_arr, A, data, sigma)
        elif dtype == 'amp':
            chisq = chisq_amp_fft(vis_arr, A, data, sigma)
        elif dtype == 'logamp':
            chisq = chisq_logamp_fft(vis_arr, A, data, sigma)
        elif dtype == 'bs':
            chisq = chisq_bs_fft(vis_arr, A, data, sigma)
        elif dtype == 'cphase':
            chisq = chisq_cphase_fft(vis_arr, A, data, sigma)
        elif dtype == 'camp':
            chisq = chisq_camp_fft(vis_arr, A, data, sigma)
        elif dtype == 'logcamp':
            chisq = chisq_logcamp_fft(vis_arr, A, data, sigma)

    elif ttype== 'nfft':
        if len(mask)>0 and np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)

        if dtype == 'vis':
            chisq = chisq_vis_nfft(imvec, A, data, sigma)
        elif dtype == 'amp':
            chisq = chisq_amp_nfft(imvec, A, data, sigma)
        elif dtype == 'logamp':
            chisq = chisq_logamp_nfft(imvec, A, data, sigma)
        elif dtype == 'bs':
            chisq = chisq_bs_nfft(imvec, A, data, sigma)
        elif dtype == 'cphase':
            chisq = chisq_cphase_nfft(imvec, A, data, sigma)
        elif dtype == 'camp':
            chisq = chisq_camp_nfft(imvec, A, data, sigma)
        elif dtype == 'logcamp':
            chisq = chisq_logcamp_nfft(imvec, A, data, sigma)

    return chisq

def chisqgrad(imvec, A, data, sigma, dtype, ttype='direct', mask=None):
    """return the chi^2 gradient for the appropriate dtype
    """

    if mask is None: mask=[]
    chisqgrad = np.zeros(len(imvec))
    if not dtype in DATATERMS:
        return chisqgrad

    if ttype not in ['fast','direct','nfft']:
        raise Exception("Possible ttype values are 'fast', 'direct','nfft'!")

    if ttype == 'direct':
        if dtype == 'vis':
            chisqgrad = chisqgrad_vis(imvec, A, data, sigma)
        elif dtype == 'amp':
            chisqgrad = chisqgrad_amp(imvec, A, data, sigma)
        elif dtype == 'logamp':
            chisqgrad = chisqgrad_logamp(imvec, A, data, sigma)
        elif dtype == 'bs':
            chisqgrad = chisqgrad_bs(imvec, A, data, sigma)
        elif dtype == 'cphase':
            chisqgrad = chisqgrad_cphase(imvec, A, data, sigma)
        elif dtype == 'camp':
            chisqgrad = chisqgrad_camp(imvec, A, data, sigma)
        elif dtype == 'logcamp':
            chisqgrad = chisqgrad_logcamp(imvec, A, data, sigma)

    elif ttype== 'fast':
        if len(mask)>0 and np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)
        vis_arr = fft_imvec(imvec, A[0])

        if dtype == 'vis':
            chisqgrad = chisqgrad_vis_fft(vis_arr, A, data, sigma)
        elif dtype == 'amp':
            chisqgrad = chisqgrad_amp_fft(vis_arr, A, data, sigma)
        elif dtype == 'logamp':
            chisqgrad = chisqgrad_logamp_fft(vis_arr, A, data, sigma)
        elif dtype == 'bs':
            chisqgrad = chisqgrad_bs_fft(vis_arr, A, data, sigma)
        elif dtype == 'cphase':
            chisqgrad = chisqgrad_cphase_fft(vis_arr, A, data, sigma)
        elif dtype == 'camp':
            chisqgrad = chisqgrad_camp_fft(vis_arr, A, data, sigma)
        elif dtype == 'logcamp':
            chisqgrad = chisqgrad_logcamp_fft(vis_arr, A, data, sigma)

        if len(mask)>0 and np.any(np.invert(mask)):
            chisqgrad = chisqgrad[mask]

    elif ttype== 'nfft':
        if len(mask)>0 and np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)

        if dtype == 'vis':
            chisqgrad = chisqgrad_vis_nfft(imvec, A, data, sigma)
        elif dtype == 'amp':
            chisqgrad = chisqgrad_amp_nfft(imvec, A, data, sigma)
        elif dtype == 'logamp':
            chisqgrad = chisqgrad_logamp_nfft(imvec, A, data, sigma)
        elif dtype == 'bs':
            chisqgrad = chisqgrad_bs_nfft(imvec, A, data, sigma)
        elif dtype == 'cphase':
            chisqgrad = chisqgrad_cphase_nfft(imvec, A, data, sigma)
        elif dtype == 'camp':
            chisqgrad = chisqgrad_camp_nfft(imvec, A, data, sigma)
        elif dtype == 'logcamp':
            chisqgrad = chisqgrad_logcamp_nfft(imvec, A, data, sigma)

        if len(mask)>0 and np.any(np.invert(mask)):
            chisqgrad = chisqgrad[mask]

    return chisqgrad


def regularizer(imvec, nprior, mask, flux, xdim, ydim, psize, stype, **kwargs):
    """return the regularizer value
    """

    norm_reg = kwargs.get('norm_reg', NORM_REGULARIZER)
    beam_size = kwargs.get('beam_size', psize)
    alpha_A = kwargs.get('alpha_A', 1.0)

    if stype == "flux":
        s = -sflux(imvec, nprior, flux, norm_reg=norm_reg)
    elif stype == "cm":
        s = -scm(imvec, xdim, ydim, psize, flux, mask, norm_reg=norm_reg, beam_size=beam_size)
    elif stype == "simple":
        s = -ssimple(imvec, nprior, flux, norm_reg=norm_reg)
    elif stype == "l1":
        s = -sl1(imvec, nprior, flux, norm_reg=norm_reg)
    elif stype == "lA":
        s = -slA(imvec, nprior, psize, flux, beam_size, alpha_A, norm_reg)
    elif stype == "gs":
        s = -sgs(imvec, nprior, flux, norm_reg=norm_reg)
    elif stype == "patch":
        s = -spatch(imvec, nprior, flux, norm_reg=norm_reg)
    elif stype == "tv":
        if np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)
        s = -stv(imvec, xdim, ydim, psize, flux, norm_reg=norm_reg, beam_size=beam_size)
    elif stype == "tv2":
        if np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)
        s = -stv2(imvec, xdim, ydim, psize, flux, norm_reg=norm_reg, beam_size=beam_size)
    elif stype == "compact":
        if np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)
        s = -scompact(imvec, xdim, ydim, psize, flux, norm_reg=norm_reg)
    elif stype == "compact2":
        if np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)
        s = -scompact2(imvec, xdim, ydim, psize, flux, norm_reg=norm_reg)
    elif stype == "rgauss":
        #additional key words for gaussian regularizer 
        major = kwargs.get('major',1.0)
        minor = kwargs.get('minor',1.0)
        PA = kwargs.get('PA',1.0)

        if np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)         
        s = -sgauss(imvec, xdim, ydim, psize, major=major, minor=minor, PA=PA)
    else:
        s = 0

    return s

def regularizergrad(imvec, nprior, mask, flux, xdim, ydim, psize, stype, **kwargs):
    """return the regularizer gradient
    """

    norm_reg = kwargs.get('norm_reg', NORM_REGULARIZER)
    beam_size = kwargs.get('beam_size', psize)
    alpha_A = kwargs.get('alpha_A', 1.0)

    if stype == "flux":
        s = -sfluxgrad(imvec, nprior, flux, norm_reg=norm_reg)
    elif stype == "cm":
        s = -scmgrad(imvec, xdim, ydim, psize, flux, mask, norm_reg=norm_reg, beam_size=beam_size)
    elif stype == "simple":
        s = -ssimplegrad(imvec, nprior, flux, norm_reg=norm_reg)
    elif stype == "l1":
        s = -sl1grad(imvec, nprior, flux, norm_reg=norm_reg)
    elif stype == "lA":
        s = -slAgrad(imvec, nprior, psize, flux, beam_size, alpha_A, norm_reg)
    elif stype == "gs":
        s = -sgsgrad(imvec, nprior, flux, norm_reg=norm_reg)
    elif stype == "patch":
        s = -spatchgrad(imvec, nprior, flux, norm_reg=norm_reg)
    elif stype == "tv":
        if np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)
        s = -stvgrad(imvec, xdim, ydim, psize, flux, norm_reg=norm_reg, beam_size=beam_size)
        s = s[mask]
    elif stype == "tv2":
        if np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)
        s = -stv2grad(imvec, xdim, ydim, psize, flux, norm_reg=norm_reg, beam_size=beam_size)
        s = s[mask]
    elif stype == "compact":
        if np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)
        s = -scompactgrad(imvec, xdim, ydim, psize, flux, norm_reg=norm_reg)
        s = s[mask]
    elif stype == "compact2":
        if np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)
        s = -scompact2grad(imvec, xdim, ydim, psize, flux, norm_reg=norm_reg)
        s = s[mask]
    elif stype == "rgauss":
        #additional key words for gaussian regularizer 
        major = kwargs.get('major',1.0)
        minor = kwargs.get('minor',1.0)
        PA = kwargs.get('PA',1.0)

        if np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)
        s = -sgauss_grad(imvec, xdim, ydim, psize, major, minor, PA)
        s = s[mask]
    else:
        s = np.zeros(len(imvec))

    return s



def chisqdata(Obsdata, Prior, mask, dtype, pol='I', **kwargs):

    """Return the data, sigma, and matrices for the appropriate dtype
    """

    ttype=kwargs.get('ttype','direct')
    (data, sigma, A) = (False, False, False)
    if ttype not in ['fast','direct','nfft']:
        raise Exception("Possible ttype values are 'fast', 'direct','nfft'!")

    if ttype=='direct':
        if dtype == 'vis':
            (data, sigma, A) = chisqdata_vis(Obsdata, Prior, mask, pol=pol, **kwargs)
        elif dtype == 'amp' or dtype == 'logamp':
            (data, sigma, A) = chisqdata_amp(Obsdata, Prior, mask, pol=pol,**kwargs)
        elif dtype == 'bs':
            (data, sigma, A) = chisqdata_bs(Obsdata, Prior, mask, pol=pol,**kwargs)
        elif dtype == 'cphase':
            (data, sigma, A) = chisqdata_cphase(Obsdata, Prior, mask, pol=pol,**kwargs)
        elif dtype == 'camp':
            (data, sigma, A) = chisqdata_camp(Obsdata, Prior, mask, pol=pol,**kwargs)
        elif dtype == 'logcamp':
            (data, sigma, A) = chisqdata_logcamp(Obsdata, Prior, mask, pol=pol,**kwargs)

    elif ttype=='fast':
        if dtype=='vis':
            (data, sigma, A) = chisqdata_vis_fft(Obsdata, Prior, pol=pol, **kwargs)
        elif dtype == 'amp' or dtype == 'logamp':
            (data, sigma, A) = chisqdata_amp_fft(Obsdata, Prior, pol=pol, **kwargs)
        elif dtype == 'bs':
            (data, sigma, A) = chisqdata_bs_fft(Obsdata, Prior, pol=pol, **kwargs)
        elif dtype == 'cphase':
            (data, sigma, A) = chisqdata_cphase_fft(Obsdata, Prior, pol=pol, **kwargs)
        elif dtype == 'camp':
            (data, sigma, A) = chisqdata_camp_fft(Obsdata, Prior, pol=pol, **kwargs)
        elif dtype == 'logcamp':
            (data, sigma, A) = chisqdata_logcamp_fft(Obsdata, Prior, pol=pol, **kwargs)

    elif ttype=='nfft':
        if dtype=='vis':
            (data, sigma, A) = chisqdata_vis_nfft(Obsdata, Prior, pol=pol, **kwargs)
        elif dtype == 'amp' or dtype == 'logamp':
            (data, sigma, A) = chisqdata_amp_nfft(Obsdata, Prior, pol=pol, **kwargs)
        elif dtype == 'bs':
            (data, sigma, A) = chisqdata_bs_nfft(Obsdata, Prior, pol=pol, **kwargs)
        elif dtype == 'cphase':
            (data, sigma, A) = chisqdata_cphase_nfft(Obsdata, Prior, pol=pol, **kwargs)
        elif dtype == 'camp':
            (data, sigma, A) = chisqdata_camp_nfft(Obsdata, Prior, pol=pol, **kwargs)
        elif dtype == 'logcamp':
            (data, sigma, A) = chisqdata_logcamp_nfft(Obsdata, Prior, pol=pol, **kwargs)


    return (data, sigma, A)


##################################################################################################
# DFT Chi-squared and Gradient Functions
##################################################################################################

def chisq_vis(imvec, Amatrix, vis, sigma):
    """Visibility chi-squared"""

    samples = np.dot(Amatrix, imvec)
    return np.sum(np.abs((samples-vis)/sigma)**2)/(2*len(vis))

def chisqgrad_vis(imvec, Amatrix, vis, sigma):
    """The gradient of the visibility chi-squared"""

    samples = np.dot(Amatrix, imvec)
    wdiff = (vis - samples)/(sigma**2)

    out = -np.real(np.dot(Amatrix.conj().T, wdiff))/len(vis)
    return out

def chisq_amp(imvec, A, amp, sigma):
    """Visibility Amplitudes (normalized) chi-squared"""

    amp_samples = np.abs(np.dot(A, imvec))
    return np.sum(np.abs((amp - amp_samples)/sigma)**2)/len(amp)

def chisqgrad_amp(imvec, A, amp, sigma):
    """The gradient of the amplitude chi-squared"""

    i1 = np.dot(A, imvec)
    amp_samples = np.abs(i1)

    pp = ((amp - amp_samples) * amp_samples) / (sigma**2) / i1
    out = (-2.0/len(amp)) * np.real(np.dot(pp, A))
    return out

def chisq_bs(imvec, Amatrices, bis, sigma):
    """Bispectrum chi-squared"""

    bisamples = np.dot(Amatrices[0], imvec) * np.dot(Amatrices[1], imvec) * np.dot(Amatrices[2], imvec)
    chisq= np.sum(np.abs(((bis - bisamples)/sigma))**2)/(2.*len(bis))
    return chisq

def chisqgrad_bs(imvec, Amatrices, bis, sigma):
    """The gradient of the bispectrum chi-squared"""

    bisamples = np.dot(Amatrices[0], imvec) * np.dot(Amatrices[1], imvec) * np.dot(Amatrices[2], imvec)
    wdiff = ((bis - bisamples).conj())/(sigma**2)
    pt1 = wdiff * np.dot(Amatrices[1],imvec) * np.dot(Amatrices[2],imvec)
    pt2 = wdiff * np.dot(Amatrices[0],imvec) * np.dot(Amatrices[2],imvec)
    pt3 = wdiff * np.dot(Amatrices[0],imvec) * np.dot(Amatrices[1],imvec)
    out = -np.real(np.dot(pt1, Amatrices[0]) + np.dot(pt2, Amatrices[1]) + np.dot(pt3, Amatrices[2]))/len(bis)
    return out

def chisq_cphase(imvec, Amatrices, clphase, sigma):
    """Closure Phases (normalized) chi-squared"""
    clphase = clphase * DEGREE
    sigma = sigma * DEGREE
    clphase_samples = np.angle(np.dot(Amatrices[0], imvec) * np.dot(Amatrices[1], imvec) * np.dot(Amatrices[2], imvec))
    chisq= (2.0/len(clphase)) * np.sum((1.0 - np.cos(clphase-clphase_samples))/(sigma**2))
    return chisq

def chisqgrad_cphase(imvec, Amatrices, clphase, sigma):
    """The gradient of the closure phase chi-squared"""
    clphase = clphase * DEGREE
    sigma = sigma * DEGREE

    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    clphase_samples = np.angle(i1 * i2 * i3)

    pref = np.sin(clphase - clphase_samples)/(sigma**2)
    pt1  = pref/i1
    pt2  = pref/i2
    pt3  = pref/i3
    out  = -(2.0/len(clphase)) * np.imag(np.dot(pt1, Amatrices[0]) + np.dot(pt2, Amatrices[1]) + np.dot(pt3, Amatrices[2]))
    return out

def chisq_camp(imvec, Amatrices, clamp, sigma):
    """Closure Amplitudes (normalized) chi-squared"""

    clamp_samples = np.abs(np.dot(Amatrices[0], imvec) * np.dot(Amatrices[1], imvec) / (np.dot(Amatrices[2], imvec) * np.dot(Amatrices[3], imvec)))
    chisq = np.sum(np.abs((clamp - clamp_samples)/sigma)**2)/len(clamp)
    return chisq

def chisqgrad_camp(imvec, Amatrices, clamp, sigma):
    """The gradient of the closure amplitude chi-squared"""

    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    i4 = np.dot(Amatrices[3], imvec)
    clamp_samples = np.abs((i1 * i2)/(i3 * i4))

    pp = ((clamp - clamp_samples) * clamp_samples)/(sigma**2)
    pt1 =  pp/i1
    pt2 =  pp/i2
    pt3 = -pp/i3
    pt4 = -pp/i4
    out = (-2.0/len(clamp)) * np.real(np.dot(pt1, Amatrices[0]) + np.dot(pt2, Amatrices[1]) + np.dot(pt3, Amatrices[2]) + np.dot(pt4, Amatrices[3]))
    return out

def chisq_logcamp(imvec, Amatrices, log_clamp, sigma):
    """Log Closure Amplitudes (normalized) chi-squared"""

    a1 = np.abs(np.dot(Amatrices[0], imvec))
    a2 = np.abs(np.dot(Amatrices[1], imvec))
    a3 = np.abs(np.dot(Amatrices[2], imvec))
    a4 = np.abs(np.dot(Amatrices[3], imvec))

    samples = np.log(a1) + np.log(a2) - np.log(a3) - np.log(a4)
    chisq = np.sum(np.abs((log_clamp - samples)/sigma)**2) / (len(log_clamp))
    return  chisq

def chisqgrad_logcamp(imvec, Amatrices, log_clamp, sigma):
    """The gradient of the Log closure amplitude chi-squared"""

    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    i4 = np.dot(Amatrices[3], imvec)
    log_clamp_samples = np.log(np.abs(i1)) + np.log(np.abs(i2)) - np.log(np.abs(i3)) - np.log(np.abs(i4))

    pp = (log_clamp - log_clamp_samples) / (sigma**2)
    pt1 = pp / i1
    pt2 = pp / i2
    pt3 = -pp / i3
    pt4 = -pp / i4
    out = (-2.0/len(log_clamp)) * np.real(np.dot(pt1, Amatrices[0]) + np.dot(pt2, Amatrices[1]) + np.dot(pt3, Amatrices[2]) + np.dot(pt4, Amatrices[3]))
    return out

def chisq_logamp(imvec, A, amp, sigma):
    """Log Visibility Amplitudes (normalized) chi-squared"""

    # to lowest order the variance on the logarithm of a quantity x is
    # sigma^2_log(x) = sigma^2/x^2
    logsigma = sigma / amp

    amp_samples = np.abs(np.dot(A, imvec))
    return np.sum(np.abs((np.log(amp) - np.log(amp_samples))/logsigma)**2)/len(amp)

def chisqgrad_logamp(imvec, A, amp, sigma):
    """The gradient of the Log amplitude chi-squared"""

    # to lowest order the variance on the logarithm of a quantity x is
    # sigma^2_log(x) = sigma^2/x^2
    logsigma = sigma / amp

    i1 = np.dot(A, imvec)
    amp_samples = np.abs(i1)

    pp = ((np.log(amp) - np.log(amp_samples))) / (logsigma**2) / i1
    out = (-2.0/len(amp)) * np.real(np.dot(pp, A))
    return out

##################################################################################################
# FFT Chi-squared and Gradient Functions
##################################################################################################
def chisq_vis_fft(vis_arr, A, vis, sigma):
    """Visibility chi-squared from fft
    """

    im_info, sampler_info_list, gridder_info_list = A
    samples = sampler(vis_arr, sampler_info_list, sample_type="vis")

    chisq = np.sum(np.abs((samples-vis)/sigma)**2)/(2*len(vis))

    return chisq

def chisqgrad_vis_fft(vis_arr, A, vis, sigma):

    """The gradient of the visibility chi-squared from fft
    """

    im_info, sampler_info_list, gridder_info_list = A

    # samples and gradient FT
    pulsefac = sampler_info_list[0].pulsefac
    samples = sampler(vis_arr, sampler_info_list, sample_type="vis")
    wdiff_vec = (-1.0/len(vis)*(vis - samples)/(sigma**2)) * pulsefac.conj()

    # Setup and perform the inverse FFT
    wdiff_arr = gridder([wdiff_vec], gridder_info_list)
    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff_arr)))
    grad_arr = grad_arr * (im_info.npad * im_info.npad)

    # extract relevant cells and flatten
    # TODO or is x<-->y??
    out = np.real(grad_arr[im_info.padvalx1:-im_info.padvalx2, im_info.padvaly1:-im_info.padvaly2].flatten())

    return out

def chisq_amp_fft(vis_arr, A, amp, sigma):
    """Visibility amplitude chi-squared from fft
    """

    im_info, sampler_info_list, gridder_info_list = A
    amp_samples = np.abs(sampler(vis_arr, sampler_info_list, sample_type="vis"))
    chisq = np.sum(np.abs((amp_samples-amp)/sigma)**2)/(len(amp))
    return chisq

def chisqgrad_amp_fft(vis_arr, A, amp, sigma):

    """The gradient of the amplitude chi-kernesquared
    """

    im_info, sampler_info_list, gridder_info_list = A

    # samples
    samples = sampler(vis_arr, sampler_info_list, sample_type="vis")
    amp_samples = np.abs(samples)

    # gradient FT
    pulsefac = sampler_info_list[0].pulsefac
    wdiff_vec = (-2.0/len(amp)*((amp - amp_samples) * amp_samples) / (sigma**2) / samples.conj()) * pulsefac.conj()

    # Setup and perform the inverse FFT
    wdiff_arr = gridder([wdiff_vec], gridder_info_list)
    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff_arr)))
    grad_arr = grad_arr * (im_info.npad * im_info.npad)

    # extract relevent cells and flatten
    # TODO or is x<-->y??
    out = np.real(grad_arr[im_info.padvalx1:-im_info.padvalx2,im_info.padvaly1:-im_info.padvaly2].flatten())

    return out

def chisq_bs_fft(vis_arr, A, bis, sigma):
    """Bispectrum chi-squared from fft"""

    im_info, sampler_info_list, gridder_info_list = A
    bisamples = sampler(vis_arr, sampler_info_list, sample_type="bs")

    return np.sum(np.abs(((bis - bisamples)/sigma))**2)/(2.*len(bis))

def chisqgrad_bs_fft(vis_arr, A, bis, sigma):

    """The gradient of the amplitude chi-squared
    """
    im_info, sampler_info_list, gridder_info_list = A

    v1 = sampler(vis_arr, [sampler_info_list[0]], sample_type="vis")
    v2 = sampler(vis_arr, [sampler_info_list[1]], sample_type="vis")
    v3 = sampler(vis_arr, [sampler_info_list[2]], sample_type="vis")
    bisamples = v1*v2*v3

    wdiff = -1.0/len(bis)*(bis - bisamples)/(sigma**2)

    pt1 = wdiff * (v2 * v3).conj() * sampler_info_list[0].pulsefac.conj()
    pt2 = wdiff * (v1 * v3).conj() * sampler_info_list[1].pulsefac.conj()
    pt3 = wdiff * (v1 * v2).conj() * sampler_info_list[2].pulsefac.conj()

    # Setup and perform the inverse FFT
    wdiff = gridder([pt1,pt2,pt3], gridder_info_list)
    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff)))
    grad_arr = grad_arr * (im_info.npad * im_info.npad)

    # extract relevant cells and flatten
    # TODO or is x<-->y??
    out = np.real(grad_arr[im_info.padvalx1:-im_info.padvalx2,im_info.padvaly1:-im_info.padvaly2].flatten())
    return out

def chisq_cphase_fft(vis_arr, A, clphase, sigma):
    """Closure Phases (normalized) chi-squared from fft
    """

    clphase = clphase * DEGREE
    sigma = sigma * DEGREE

    im_info, sampler_info_list, gridder_info_list = A
    clphase_samples = np.angle(sampler(vis_arr, sampler_info_list, sample_type="bs"))

    chisq = (2.0/len(clphase)) * np.sum((1.0 - np.cos(clphase-clphase_samples))/(sigma**2))
    return chisq

def chisqgrad_cphase_fft(vis_arr, A, clphase, sigma):
    """The gradient of the closure phase chi-squared from fft"""

    clphase = clphase * DEGREE
    sigma = sigma * DEGREE
    im_info, sampler_info_list, gridder_info_list = A

    #sample visibilities and closure phases
    v1 = sampler(vis_arr, [sampler_info_list[0]], sample_type="vis")
    v2 = sampler(vis_arr, [sampler_info_list[1]], sample_type="vis")
    v3 = sampler(vis_arr, [sampler_info_list[2]], sample_type="vis")
    clphase_samples = np.angle(v1*v2*v3)

    pref = (2.0/len(clphase)) * np.sin(clphase - clphase_samples)/(sigma**2)
    pt1  = pref/v1.conj() * sampler_info_list[0].pulsefac.conj()
    pt2  = pref/v2.conj() * sampler_info_list[1].pulsefac.conj()
    pt3  = pref/v3.conj() * sampler_info_list[2].pulsefac.conj()

    # Setup and perform the inverse FFT
    wdiff = gridder([pt1,pt2,pt3], gridder_info_list)
    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff)))
    grad_arr = grad_arr * (im_info.npad * im_info.npad)

    # extract relevant cells and flatten
    # TODO or is x<-->y??
    out = np.imag(grad_arr[im_info.padvalx1:-im_info.padvalx2,im_info.padvaly1:-im_info.padvaly2].flatten())

    return out

def chisq_camp_fft(vis_arr, A, clamp, sigma):
    """Closure Amplitudes (normalized) chi-squared from fft
    """

    im_info, sampler_info_list, gridder_info_list = A
    clamp_samples = sampler(vis_arr, sampler_info_list, sample_type="camp")
    chisq = np.sum(np.abs((clamp - clamp_samples)/sigma)**2)/len(clamp)
    return chisq

def chisqgrad_camp_fft(vis_arr, A, clamp, sigma):

    """The gradient of the closure amplitude chi-squared from fft
    """

    im_info, sampler_info_list, gridder_info_list = A

    # sampled visibility and closure amplitudes
    v1 = sampler(vis_arr, [sampler_info_list[0]], sample_type="vis")
    v2 = sampler(vis_arr, [sampler_info_list[1]], sample_type="vis")
    v3 = sampler(vis_arr, [sampler_info_list[2]], sample_type="vis")
    v4 = sampler(vis_arr, [sampler_info_list[3]], sample_type="vis")
    clamp_samples = np.abs((v1 * v2)/(v3 * v4))

    # gradient components
    pp = (-2.0/len(clamp)) * ((clamp - clamp_samples) * clamp_samples)/(sigma**2)
    pt1 =  pp/v1.conj()* sampler_info_list[0].pulsefac.conj()
    pt2 =  pp/v2.conj()* sampler_info_list[1].pulsefac.conj()
    pt3 = -pp/v3.conj()* sampler_info_list[2].pulsefac.conj()
    pt4 = -pp/v4.conj()* sampler_info_list[3].pulsefac.conj()

    # Setup and perform the inverse FFT
    wdiff = gridder([pt1,pt2,pt3,pt4], gridder_info_list)
    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff)))
    grad_arr = grad_arr * (im_info.npad * im_info.npad)

    # extract relevant cells and flatten
    # TODO or is x<-->y??
    out = np.real(grad_arr[im_info.padvalx1:-im_info.padvalx2,im_info.padvaly1:-im_info.padvaly2].flatten())

    return out

def chisq_logcamp_fft(vis_arr, A, log_clamp, sigma):
    """Closure Amplitudes (normalized) chi-squared from fft
    """

    im_info, sampler_info_list, gridder_info_list = A
    log_clamp_samples = np.log(sampler(vis_arr, sampler_info_list, sample_type='camp'))

    chisq = np.sum(np.abs((log_clamp - log_clamp_samples)/sigma)**2) / (len(log_clamp))

    return chisq

def chisqgrad_logcamp_fft(vis_arr, A, log_clamp, sigma):

    """The gradient of the closure amplitude chi-squared from fft
    """

    im_info, sampler_info_list, gridder_info_list = A

    # sampled visibility and closure amplitudes
    v1 = sampler(vis_arr, [sampler_info_list[0]], sample_type="vis")
    v2 = sampler(vis_arr, [sampler_info_list[1]], sample_type="vis")
    v3 = sampler(vis_arr, [sampler_info_list[2]], sample_type="vis")
    v4 = sampler(vis_arr, [sampler_info_list[3]], sample_type="vis")

    log_clamp_samples = np.log(np.abs((v1 * v2)/(v3 * v4)))

    # gradient components
    pp = (-2.0/len(log_clamp)) * (log_clamp - log_clamp_samples) / (sigma**2)
    pt1 = pp / v1.conj()* sampler_info_list[0].pulsefac.conj()
    pt2 = pp / v2.conj()* sampler_info_list[1].pulsefac.conj()
    pt3 = -pp / v3.conj()* sampler_info_list[2].pulsefac.conj()
    pt4 = -pp / v4.conj()* sampler_info_list[3].pulsefac.conj()

    # Setup and perform the inverse FFT
    wdiff = gridder([pt1,pt2,pt3,pt4], gridder_info_list)
    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff)))
    grad_arr = grad_arr * (im_info.npad * im_info.npad)

    # extract relevant cells and flatten
    # TODO or is x<-->y??
    out = np.real(grad_arr[im_info.padvalx1:-im_info.padvalx2,im_info.padvaly1:-im_info.padvaly2].flatten())

    return out


def chisq_logamp_fft(vis_arr, A, amp, sigma):
    """Visibility amplitude chi-squared from fft
    """

    # to lowest order the variance on the logarithm of a quantity x is
    # sigma^2_log(x) = sigma^2/x^2
    logsigma = sigma / amp

    im_info, sampler_info_list, gridder_info_list = A
    amp_samples = np.abs(sampler(vis_arr, sampler_info_list, sample_type="vis"))
    chisq = np.sum(np.abs((np.log(amp_samples)-np.log(amp))/logsigma)**2)/(len(amp))
    return chisq

def chisqgrad_logamp_fft(vis_arr, A, amp, sigma):

    """The gradient of the amplitude chi-kernesquared
    """

    im_info, sampler_info_list, gridder_info_list = A

    # samples
    samples = sampler(vis_arr, sampler_info_list, sample_type="vis")
    amp_samples = np.abs(samples)

    # gradient FT
    logsigma = sigma / amp
    pulsefac = sampler_info_list[0].pulsefac
    wdiff_vec = (-2.0/len(amp)*((np.log(amp) - np.log(amp_samples))) / (logsigma**2) / samples.conj()) * pulsefac.conj()

    # Setup and perform the inverse FFT
    wdiff_arr = gridder([wdiff_vec], gridder_info_list)
    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff_arr)))
    grad_arr = grad_arr * (im_info.npad * im_info.npad)

    # extract relevent cells and flatten
    # TODO or is x<-->y??
    out = np.real(grad_arr[im_info.padvalx1:-im_info.padvalx2,im_info.padvaly1:-im_info.padvaly2].flatten())

    return out

##################################################################################################
# NFFT Chi-squared and Gradient Functions
##################################################################################################
def chisq_vis_nfft(imvec, A, vis, sigma):
    """Visibility chi-squared from nfft
    """

    #get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    #compute uniform --> nonuniform transform
    plan.f_hat = imvec.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    samples = plan.f.copy()*pulsefac

    #compute chi^2
    chisq = np.sum(np.abs((samples-vis)/sigma)**2)/(2*len(vis))

    return chisq

def chisqgrad_vis_nfft(imvec, A, vis, sigma):

    """The gradient of the visibility chi-squared from nfft
    """

    #get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    #compute uniform --> nonuniform transform
    plan.f_hat = imvec.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    samples = plan.f.copy()*pulsefac

    # gradient vec for adjoint FT
    wdiff_vec = (-1.0/len(vis)*(vis - samples)/(sigma**2)) * pulsefac.conj()
    plan.f = wdiff_vec
    plan.adjoint()
    grad = np.real((plan.f_hat.copy().T).reshape(nfft_info.xdim*nfft_info.ydim))

    return grad

def chisq_amp_nfft(imvec, A, amp, sigma):
    """Visibility amplitude chi-squared from nfft
    """
    #get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    #compute uniform --> nonuniform transform
    plan.f_hat = imvec.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    samples = plan.f.copy()*pulsefac

    #compute chi^2
    amp_samples = np.abs(samples)
    chisq = np.sum(np.abs((amp_samples-amp)/sigma)**2)/(len(amp))

    return chisq

def chisqgrad_amp_nfft(imvec, A, amp, sigma):

    """The gradient of the amplitude chi-squared from nfft
    """

    #get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    #compute uniform --> nonuniform transform
    plan.f_hat = imvec.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    samples = plan.f.copy()*pulsefac
    amp_samples=np.abs(samples)

    # gradient vec for adjoint FT
    wdiff_vec = (-2.0/len(amp)*((amp - amp_samples) * samples) / (sigma**2) / amp_samples) * pulsefac.conj()
    plan.f = wdiff_vec
    plan.adjoint()
    out = np.real((plan.f_hat.copy().T).reshape(nfft_info.xdim*nfft_info.ydim))

    return out

def chisq_bs_nfft(imvec, A, bis, sigma):
    """Bispectrum chi-squared from fft"""

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
    plan1.f_hat = imvec.copy().reshape((nfft_info1.ydim,nfft_info1.xdim)).T
    plan1.trafo()
    samples1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim,nfft_info2.xdim)).T
    plan2.trafo()
    samples2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim,nfft_info3.xdim)).T
    plan3.trafo()
    samples3 = plan3.f.copy()*pulsefac3

    #compute chi^2
    bisamples = samples1*samples2*samples3
    chisq = np.sum(np.abs(((bis - bisamples)/sigma))**2)/(2.*len(bis))
    return chisq

def chisqgrad_bs_nfft(imvec, A, bis, sigma):
    """The gradient of the amplitude chi-squared from the nfft
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
    plan1.f_hat = imvec.copy().reshape((nfft_info1.ydim,nfft_info1.xdim)).T
    plan1.trafo()
    v1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim,nfft_info2.xdim)).T
    plan2.trafo()
    v2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim,nfft_info3.xdim)).T
    plan3.trafo()
    v3 = plan3.f.copy()*pulsefac3

    # gradient vec for adjoint FT
    bisamples = v1*v2*v3
    wdiff = -1.0/len(bis)*(bis - bisamples)/(sigma**2)
    pt1 = wdiff * (v2 * v3).conj() * pulsefac1.conj()
    pt2 = wdiff * (v1 * v3).conj() * pulsefac2.conj()
    pt3 = wdiff * (v1 * v2).conj() * pulsefac3.conj()

    # Setup and perform the inverse FFT
    plan1.f = pt1
    plan1.adjoint()
    out1 = np.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))

    plan2.f = pt2
    plan2.adjoint()
    out2 = np.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))

    plan3.f = pt3
    plan3.adjoint()
    out3 = np.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

    out = out1 + out2 + out3
    return out

def chisq_cphase_nfft(imvec, A, clphase, sigma):
    """Closure Phases (normalized) chi-squared from nfft
    """

    clphase = clphase * DEGREE
    sigma = sigma * DEGREE

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
    plan1.f_hat = imvec.copy().reshape((nfft_info1.ydim,nfft_info1.xdim)).T
    plan1.trafo()
    samples1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim,nfft_info2.xdim)).T
    plan2.trafo()
    samples2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim,nfft_info3.xdim)).T
    plan3.trafo()
    samples3 = plan3.f.copy()*pulsefac3

    #compute chi^2
    clphase_samples = np.angle(samples1*samples2*samples3)
    chisq = (2.0/len(clphase)) * np.sum((1.0 - np.cos(clphase-clphase_samples))/(sigma**2))

    return chisq

def chisqgrad_cphase_nfft(imvec, A, clphase, sigma):
    """The gradient of the closure phase chi-squared from nfft"""

    clphase = clphase * DEGREE
    sigma = sigma * DEGREE

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
    plan1.f_hat = imvec.copy().reshape((nfft_info1.ydim,nfft_info1.xdim)).T
    plan1.trafo()
    v1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim,nfft_info2.xdim)).T
    plan2.trafo()
    v2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim,nfft_info3.xdim)).T
    plan3.trafo()
    v3 = plan3.f.copy()*pulsefac3

    # gradient vec for adjoint FT
    clphase_samples = np.angle(v1*v2*v3)
    pref = (2.0/len(clphase)) * np.sin(clphase - clphase_samples)/(sigma**2)
    pt1  = pref/v1.conj() * pulsefac1.conj()
    pt2  = pref/v2.conj() * pulsefac2.conj()
    pt3  = pref/v3.conj() * pulsefac3.conj()

    # Setup and perform the inverse FFT
    plan1.f = pt1
    plan1.adjoint()
    out1 = np.imag((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))

    plan2.f = pt2
    plan2.adjoint()
    out2 = np.imag((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))

    plan3.f = pt3
    plan3.adjoint()
    out3 = np.imag((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

    out = out1 + out2 + out3
    return out

def chisq_camp_nfft(imvec, A, clamp, sigma):
    """Closure Amplitudes (normalized) chi-squared from fft
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

    nfft_info4 = A[3]
    plan4 = nfft_info4.plan
    pulsefac4 = nfft_info4.pulsefac

    #compute uniform --> nonuniform transforms
    plan1.f_hat = imvec.copy().reshape((nfft_info1.ydim,nfft_info1.xdim)).T
    plan1.trafo()
    samples1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim,nfft_info2.xdim)).T
    plan2.trafo()
    samples2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim,nfft_info3.xdim)).T
    plan3.trafo()
    samples3 = plan3.f.copy()*pulsefac3

    plan4.f_hat = imvec.copy().reshape((nfft_info4.ydim,nfft_info4.xdim)).T
    plan4.trafo()
    samples4 = plan4.f.copy()*pulsefac4

    #compute chi^2
    clamp_samples = np.abs((samples1*samples2)/(samples3*samples4))
    chisq = np.sum(np.abs((clamp - clamp_samples)/sigma)**2)/len(clamp)
    return chisq

def chisqgrad_camp_nfft(imvec, A, clamp, sigma):

    """The gradient of the closure amplitude chi-squared from fft
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

    nfft_info4 = A[3]
    plan4 = nfft_info4.plan
    pulsefac4 = nfft_info4.pulsefac

    #compute uniform --> nonuniform transforms
    plan1.f_hat = imvec.copy().reshape((nfft_info1.ydim,nfft_info1.xdim)).T
    plan1.trafo()
    v1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim,nfft_info2.xdim)).T
    plan2.trafo()
    v2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim,nfft_info3.xdim)).T
    plan3.trafo()
    v3 = plan3.f.copy()*pulsefac3

    plan4.f_hat = imvec.copy().reshape((nfft_info4.ydim,nfft_info4.xdim)).T
    plan4.trafo()
    v4 = plan4.f.copy()*pulsefac4

    # gradient vec for adjoint FT
    clamp_samples = np.abs((v1 * v2)/(v3 * v4))

    pp = (-2.0/len(clamp)) * ((clamp - clamp_samples) * clamp_samples)/(sigma**2)
    pt1 =  pp/v1.conj()* pulsefac1.conj()
    pt2 =  pp/v2.conj()* pulsefac2.conj()
    pt3 = -pp/v3.conj()* pulsefac3.conj()
    pt4 = -pp/v4.conj()* pulsefac4.conj()

    # Setup and perform the inverse FFT
    plan1.f = pt1
    plan1.adjoint()
    out1 = np.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))

    plan2.f = pt2
    plan2.adjoint()
    out2 = np.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))

    plan3.f = pt3
    plan3.adjoint()
    out3 = np.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

    plan4.f = pt4
    plan4.adjoint()
    out4 = np.real((plan4.f_hat.copy().T).reshape(nfft_info4.xdim*nfft_info4.ydim))

    out = out1 + out2 + out3 + out4
    return out

def chisq_logcamp_nfft(imvec, A, log_clamp, sigma):
    """Closure Amplitudes (normalized) chi-squared from fft
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

    nfft_info4 = A[3]
    plan4 = nfft_info4.plan
    pulsefac4 = nfft_info4.pulsefac

    #compute uniform --> nonuniform transforms
    plan1.f_hat = imvec.copy().reshape((nfft_info1.ydim,nfft_info1.xdim)).T
    plan1.trafo()
    samples1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim,nfft_info2.xdim)).T
    plan2.trafo()
    samples2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim,nfft_info3.xdim)).T
    plan3.trafo()
    samples3 = plan3.f.copy()*pulsefac3

    plan4.f_hat = imvec.copy().reshape((nfft_info4.ydim,nfft_info4.xdim)).T
    plan4.trafo()
    samples4 = plan4.f.copy()*pulsefac4

    #compute chi^2
    log_clamp_samples = (np.log(np.abs(samples1)) + np.log(np.abs(samples2))
                         - np.log(np.abs(samples3)) - np.log(np.abs(samples4)))
    chisq = np.sum(np.abs((log_clamp - log_clamp_samples)/sigma)**2) / (len(log_clamp))
    return chisq

def chisqgrad_logcamp_nfft(imvec, A, log_clamp, sigma):

    """The gradient of the closure amplitude chi-squared from fft
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

    nfft_info4 = A[3]
    plan4 = nfft_info4.plan
    pulsefac4 = nfft_info4.pulsefac

    #compute uniform --> nonuniform transforms
    plan1.f_hat = imvec.copy().reshape((nfft_info1.ydim,nfft_info1.xdim)).T
    plan1.trafo()
    v1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim,nfft_info2.xdim)).T
    plan2.trafo()
    v2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim,nfft_info3.xdim)).T
    plan3.trafo()
    v3 = plan3.f.copy()*pulsefac3

    plan4.f_hat = imvec.copy().reshape((nfft_info4.ydim,nfft_info4.xdim)).T
    plan4.trafo()
    v4 = plan4.f.copy()*pulsefac4

    # gradient vec for adjoint FT
    log_clamp_samples = np.log(np.abs((v1 * v2)/(v3 * v4)))

    pp = (-2.0/len(log_clamp)) * (log_clamp - log_clamp_samples) / (sigma**2)
    pt1 = pp / v1.conj()* pulsefac1.conj()
    pt2 = pp / v2.conj()* pulsefac2.conj()
    pt3 = -pp / v3.conj()* pulsefac3.conj()
    pt4 = -pp / v4.conj()* pulsefac4.conj()

    # Setup and perform the inverse FFT
    plan1.f = pt1
    plan1.adjoint()
    out1 = np.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))

    plan2.f = pt2
    plan2.adjoint()
    out2 = np.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))

    plan3.f = pt3
    plan3.adjoint()
    out3 = np.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

    plan4.f = pt4
    plan4.adjoint()
    out4 = np.real((plan4.f_hat.copy().T).reshape(nfft_info4.xdim*nfft_info4.ydim))

    out = out1 + out2 + out3 + out4
    return out



def chisq_logamp_nfft(imvec, A, amp, sigma):
    """Visibility log amplitude chi-squared from nfft
    """

    #get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    #compute uniform --> nonuniform transform
    plan.f_hat = imvec.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    samples = plan.f.copy()*pulsefac

    # to lowest order the variance on the logarithm of a quantity x is
    # sigma^2_log(x) = sigma^2/x^2
    logsigma = sigma / amp

    #compute chi^2
    amp_samples = np.abs(samples)
    chisq = np.sum(np.abs((np.log(amp_samples)-np.log(amp))/logsigma)**2)/(len(amp))

    return chisq

def chisqgrad_logamp_nfft(imvec, A, amp, sigma):

    """The gradient of the log amplitude chi-squared from nfft
    """

    #get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    #compute uniform --> nonuniform transform
    plan.f_hat = imvec.copy().reshape((nfft_info.ydim,nfft_info.xdim)).T
    plan.trafo()
    samples = plan.f.copy()*pulsefac
    amp_samples=np.abs(samples)

    # gradient vec for adjoint FT
    logsigma = sigma / amp
    wdiff_vec = (-2.0/len(amp)*((np.log(amp) - np.log(amp_samples))) / (logsigma**2) / samples.conj()) * pulsefac.conj()
    plan.f = wdiff_vec
    plan.adjoint()
    out = np.real((plan.f_hat.copy().T).reshape(nfft_info.xdim*nfft_info.ydim))

    return out


##################################################################################################
# Regularizer and Gradient Functions
##################################################################################################

def sflux(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER):
    """Total flux constraint
    """
    if norm_reg: norm = flux**2
    else: norm = 1

    out = -(np.sum(imvec) - flux)**2
    return out/norm

def sfluxgrad(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER):
    """Total flux constraint gradient
    """
    if norm_reg: norm = flux**2
    else: norm = 1

    out = -2*(np.sum(imvec) - flux)
    return out/ norm

def scm(imvec, nx, ny, psize, flux, embed_mask, norm_reg=NORM_REGULARIZER, beam_size=None):
    """Center-of-mass constraint
    """
    if beam_size is None: beam_size = psize
    if norm_reg: norm = beam_size**2 * flux**2
    else: norm = 1

    xx, yy = np.meshgrid(range(nx//2,-nx//2,-1), range(ny//2,-ny//2,-1))
    xx = psize*xx.flatten()[embed_mask]
    yy = psize*yy.flatten()[embed_mask]

    out = -(np.sum(imvec*xx)**2 + np.sum(imvec*yy)**2)
    return out/norm


def scmgrad(imvec, nx, ny, psize, flux, embed_mask, norm_reg=NORM_REGULARIZER, beam_size=None):
    """Center-of-mass constraint gradient
    """
    if beam_size is None: beam_size = psize
    if norm_reg: norm = beam_size**2 * flux**2
    else: norm = 1

    xx, yy = np.meshgrid(range(nx//2,-nx//2,-1), range(ny//2,-ny//2,-1))
    xx = psize*xx.flatten()[embed_mask]
    yy = psize*yy.flatten()[embed_mask]

    out = -2*(np.sum(imvec*xx)*xx + np.sum(imvec*yy)*yy)
    return out/norm

def ssimple(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER):
    """Simple entropy
    """
    if norm_reg: norm = flux
    else: norm = 1

    entropy = -np.sum(imvec*np.log(imvec/priorvec))
    return entropy/norm

def ssimplegrad(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER):
    """Simple entropy gradient
    """
    if norm_reg: norm = flux
    else: norm = 1

    entropygrad = -np.log(imvec/priorvec) - 1
    return entropygrad/norm

def sl1(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER):
    """L1 norm regularizer
    """
    if norm_reg: norm = flux
    else: norm = 1

    #l1 = -np.sum(np.abs(imvec - priorvec))
    l1 =  -np.sum(np.abs(imvec))
    return l1/norm

def sl1grad(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER):
    """L1 norm gradient
    """
    if norm_reg: norm = flux
    else: norm = 1

    #l1grad = -np.sign(imvec - priorvec)
    l1grad = -np.sign(imvec)
    return l1grad/norm

def fA(imvec, I_ref = 1.0, alpha_A = 1.0):
    """Function to take imvec to itself in the limit alpha_A -> 0 and to a binary representation in the limit alpha_A -> infinity
    """
    return 2.0/np.pi * (1.0 + alpha_A)/alpha_A * np.arctan( np.pi*alpha_A/2.0*np.abs(imvec)/I_ref)

def fAgrad(imvec, I_ref = 1.0, alpha_A = 1.0):
    """Function to take imvec to itself in the limit alpha_A -> 0 and to a binary representation in the limit alpha_A -> infinity
    """
    return (1.0 + alpha_A) / ( I_ref * (1.0 + (np.pi*alpha_A/2.0*imvec/I_ref)**2))

def slA(imvec, priorvec, psize, flux, beam_size=None, alpha_A = 1.0, norm_reg=NORM_REGULARIZER):
    """\ell_A regularizer
    """

    # The appropriate I_ref is something like the total flux divided by the number of pixels per beam
    if beam_size is None: beam_size = psize
    I_ref = flux

    if norm_reg:
        norm_l1 = 1.0                  # as alpha_A ->0
        norm_l0 = (beam_size/psize)**2 # as alpha_A ->\infty
        weight_l1 = 1.0/(1.0 + alpha_A)
        weight_l0 = alpha_A
        norm = (norm_l1 * weight_l1 + norm_l0 * weight_l0)/(weight_l0 + weight_l1)
    else:
        norm = 1

    return -np.sum(fA(imvec, I_ref, alpha_A))/norm

def slAgrad(imvec, priorvec, psize, flux, beam_size=None, alpha_A = 1.0, norm_reg=NORM_REGULARIZER):
    """\ell_A gradient
    """

    # The appropriate I_ref is something like the total flux divided by the number of pixels per beam
    if beam_size is None: beam_size = psize
    I_ref = flux

    if norm_reg:
        norm_l1 = 1.0                  # as alpha_A ->0
        norm_l0 = (beam_size/psize)**2 # as alpha_A ->\infty
        weight_l1 = 1.0/(1.0 + alpha_A)
        weight_l0 = alpha_A
        norm = (norm_l1 * weight_l1 + norm_l0 * weight_l0)/(weight_l0 + weight_l1)
    else:
        norm = 1

    return -fAgrad(imvec, I_ref, alpha_A)/norm

def sgs(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER):
    """Gull-skilling entropy
    """
    if norm_reg: norm = flux
    else: norm = 1

    entropy = np.sum(imvec - priorvec - imvec*np.log(imvec/priorvec))
    return entropy/norm


def sgsgrad(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER):
    """Gull-Skilling gradient
    """
    if norm_reg: norm = flux
    else: norm = 1

    entropygrad= -np.log(imvec/priorvec)
    return entropygrad/norm

def stv(imvec, nx, ny, psize, flux, norm_reg=NORM_REGULARIZER, beam_size=None):
    """Total variation regularizer
    """
    if beam_size is None: beam_size = psize
    if norm_reg: norm = flux*psize / beam_size
    else: norm = 1

    im = imvec.reshape(ny, nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    out = -np.sum(np.sqrt(np.abs(im_l1 - im)**2 + np.abs(im_l2 - im)**2))
    return out/norm

def stvgrad(imvec, nx, ny, psize, flux, norm_reg=NORM_REGULARIZER, beam_size=None):
    """Total variation gradient
    """
    if beam_size is None: beam_size = psize
    if norm_reg: norm = flux*psize / beam_size
    else: norm = 1

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
    g1 = (2*im - im_l1 - im_l2) / np.sqrt((im - im_l1)**2 + (im - im_l2)**2)
    g2 = (im - im_r1) / np.sqrt((im - im_r1)**2 + (im_r1l2 - im_r1)**2)
    g3 = (im - im_r2) / np.sqrt((im - im_r2)**2 + (im_l1r2 - im_r2)**2)

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

def stv2(imvec, nx, ny, psize, flux, norm_reg=NORM_REGULARIZER, beam_size=None):
    """Squared Total variation regularizer
    """
    if beam_size is None: beam_size = psize
    if norm_reg: norm = psize**4 * flux**2 / beam_size**4
    else: norm = 1

    im = imvec.reshape(ny, nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    out = -np.sum((im_l1 - im)**2 + (im_l2 - im)**2)
    return out/norm

def stv2grad(imvec, nx, ny, psize, flux, norm_reg=NORM_REGULARIZER, beam_size=None):
    """Squared Total variation gradient
    """
    if beam_size is None: beam_size = psize
    if norm_reg: norm = psize**4 * flux**2 / beam_size**4
    else: norm = 1

    im = imvec.reshape(ny,nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    im_r1 = np.roll(impad, 1, axis=0)[1:ny+1, 1:nx+1]
    im_r2 = np.roll(impad, 1, axis=1)[1:ny+1, 1:nx+1]

    g1 = (2*im - im_l1 - im_l2)
    g2 = (im - im_r1)
    g3 = (im - im_r2)

    #mask the first row column gradient terms that don't exist
    mask1 = np.zeros(im.shape)
    mask2 = np.zeros(im.shape)
    mask1[0,:] = 1
    mask2[:,0] = 1
    g2[mask1.astype(bool)] = 0
    g3[mask2.astype(bool)] = 0

    #add together terms and return
    out= -2*(g1 + g2 + g3).flatten()
    return out/norm

def spatch(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER):
    """Patch prior regularizer
    """
    if norm_reg: norm = flux**2
    else: norm = 1

    out = -0.5*np.sum( ( imvec - priorvec) ** 2)
    return out/norm

def spatchgrad(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER):
    """Patch prior gradient
    """
    if norm_reg: norm = flux**2
    else: norm = 1

    out = -(imvec  - priorvec)
    return out/norm

#TODO FIGURE OUT NORMALIZATIONS FOR COMPACT 1 & 2 REGULARIZERS
def scompact(imvec, nx, ny, psize, flux, norm_reg=NORM_REGULARIZER, beam_size=None):
    """I r^2 source size regularizer
    """

    if beam_size is None: beam_size = psize
    if norm_reg: norm = flux * (beam_size**2)
    else: norm = 1


    im = imvec.reshape(ny, nx)
    #im = im/flux

    xx, yy = np.meshgrid(range(nx), range(ny))
    xx = xx - (nx-1)/2.0
    yy = yy - (ny-1)/2.0
    xxpsize = xx * psize
    yypsize = yy * psize

    x0 = np.sum(np.sum(im * xxpsize))/flux
    y0 = np.sum(np.sum(im * yypsize))/flux

    out = -np.sum(np.sum(im * ( (xxpsize - x0 )**2 + (yypsize - y0  )**2 ) ) )
    return out/norm


def scompactgrad(imvec, nx, ny, psize, flux, norm_reg=NORM_REGULARIZER, beam_size=None):
    """Gradient for I r^2 source size regularizer
    """

    if beam_size is None: beam_size = psize
    if norm_reg: norm = flux * beam_size**2
    else: norm = 1


    im = imvec.reshape(ny, nx)
    #im = im/flux

    xx, yy = np.meshgrid(range(nx), range(ny))
    xx = xx - (nx-1)/2.0
    yy = yy - (ny-1)/2.0
    xxpsize = xx * psize
    yypsize = yy * psize

    x0 = np.sum(np.sum(im * xxpsize))/flux
    y0 = np.sum(np.sum(im * yypsize))/flux

    term1 = np.sum(np.sum( im * ( (xxpsize - x0) ) ) )
    term2 = np.sum(np.sum( im * ( (yypsize - y0) ) ) )

    grad = -2*xxpsize*term1 - 2*yypsize*term2  + (xxpsize - x0 )**2 + (yypsize - y0)**2

    return -grad.reshape(-1)/norm

def scompact2(imvec, nx, ny, psize, flux, norm_reg=NORM_REGULARIZER, beam_size=None):
    """I^2r^2 source size regularizer
    """

    if beam_size is None: beam_size = psize
    if norm_reg: norm = flux**2 * beam_size**2
    else: norm = 1

    im = imvec.reshape(ny, nx)

    xx, yy = np.meshgrid(range(nx), range(ny))
    xx = xx - (nx-1)/2.0
    yy = yy - (ny-1)/2.0
    xxpsize = xx * psize
    yypsize = yy * psize

    out = -np.sum(np.sum(im**2 * ( xxpsize**2 + yypsize**2 ) ) )
    return out/norm

def scompact2grad(imvec, nx, ny, psize, flux, norm_reg=NORM_REGULARIZER, beam_size=None):
    """Gradient for I^2r^2 source size regularizer
    """

    if beam_size is None: beam_size = psize
    if norm_reg: norm = flux**2 * beam_size**2
    else: norm = 1


    im = imvec.reshape(ny, nx)

    xx, yy = np.meshgrid(range(nx), range(ny))
    xx = xx - (nx-1)/2.0
    yy = yy - (ny-1)/2.0
    xxpsize = xx * psize
    yypsize = yy * psize

    grad = -2*im*(xxpsize**2 + yypsize**2)

    return grad.reshape(-1)/norm

def sgauss(imvec, xdim, ydim, psize, major, minor, PA):
    """Gaussian source size regularizer
    """

    #major, minor and PA are all in radians
    phi = PA 

    #eigenvalues of covariance matrix
    lambda1 = minor**2./(8.*np.log(2.))
    lambda2 = major**2./(8.*np.log(2.))

    #now compute covariance matrix elements from user inputs
    sigxx_prime = lambda1*(np.cos(phi)**2.) + lambda2*(np.sin(phi)**2.) 
    sigyy_prime = lambda1*(np.sin(phi)**2.) + lambda2*(np.cos(phi)**2.)
    sigxy_prime = (lambda2 - lambda1)*np.cos(phi)*np.sin(phi) 

    #we get the dimensions and image vector     
    pdim = psize
    im = imvec.reshape(xdim, ydim)
    xlist, ylist = np.meshgrid(range(xdim),range(ydim))
    xlist = xlist - (xdim-1)/2.0
    ylist = ylist - (ydim-1)/2.0

    xx = xlist * psize
    yy = ylist * psize

    #the centroid parameters
    x0 = np.sum(xx*im) / np.sum(im)
    y0 = np.sum(yy*im) / np.sum(im)

    #we calculate the elements of the covariance matrix
    sigxx = (np.sum((xx - x0)**2.*im)/np.sum(im))
    sigyy = (np.sum((yy - y0)**2.*im)/np.sum(im))
    sigxy = (np.sum((xx - x0)*(yy- y0)*im)/np.sum(im))
      
    #We calculate the regularizer #this line was CHANGED
    rgauss = -( (sigxx - sigxx_prime)**2. + (sigyy - sigyy_prime)**2. + 2*(sigxy - sigxy_prime)**2. )
    rgauss = rgauss/(major**2. * minor**2.) #normalization will need to be redone, right now requires alpha~1000 
    return rgauss


def sgauss_grad(imvec, xdim, ydim, psize, major, minor, PA):
    """Gradient for Gaussian source size regularizer
    """

    #major, minor and PA are all in radians
    phi = PA

    #computing eigenvalues of the covariance matrix
    lambda1 = (minor**2.)/(8.*np.log(2.))
    lambda2 = (major**2.)/(8.*np.log(2.))

    #now compute covariance matrix elements from user inputs

    sigxx_prime = lambda1*(np.cos(phi)**2.) + lambda2*(np.sin(phi)**2.) 
    sigyy_prime = lambda1*(np.sin(phi)**2.) + lambda2*(np.cos(phi)**2.)
    sigxy_prime = (lambda2 - lambda1)*np.cos(phi)*np.sin(phi) 


    #we get the dimensions and image vector     
    pdim = psize
    im = imvec.reshape(xdim, ydim)
    xlist, ylist = np.meshgrid(range(xdim),range(ydim))
    xlist = xlist - (xdim-1)/2.0
    ylist = ylist - (ydim-1)/2.0

    xx = xlist * psize
    yy = ylist * psize


    #the centroid parameters
    x0 = np.sum(xx*im) / np.sum(im)
    y0 = np.sum(yy*im) / np.sum(im)

    #we calculate the elements of the covariance matrix of the image 
    sigxx = (np.sum((xx - x0)**2.*im)/np.sum(im))
    sigyy = (np.sum((yy - y0)**2.*im)/np.sum(im))
    sigxy = (np.sum((xx - x0)*(yy- y0)*im)/np.sum(im))


    #now we compute the gradients of all quantities 
    #gradient of centroid 
    dx0 = ( xx -  x0) / np.sum(im)
    dy0 = ( yy -  y0) / np.sum(im)

    #gradients of covariance matrix elements 
    dxx = (( (xx - x0)**2. - 2.*(xx - x0)*dx0*im ) - sigxx ) / np.sum(im) 

    dyy = ( ( (yy - y0)**2. - 2.*(yy - y0)*dx0*im ) - sigyy ) / np.sum(im) 	

    dxy = ( ( (xx - x0)*(yy - y0) - (yy - y0)*dx0*im - (xx - x0)*dy0*im ) - sigxy ) / np.sum(im) 

    #gradient of the regularizer #this line was CHANGED
    drgauss = ( 2.*(sigxx - sigxx_prime)*dxx + 2.*(sigyy - sigyy_prime)*dyy + 4.*(sigxy - sigxy_prime)*dxy )
    drgauss = drgauss/(major**2. * minor**2.) #normalization will need to be redone, right now requires alpha~1000 

    return -drgauss.reshape(-1)


##################################################################################################
# Chi^2 Data functions
##################################################################################################
def apply_systematic_noise_snrcut(data_arr, systematic_noise, snrcut, pol):
    """apply systematic noise to VISIBILITIES or AMPLITUDES
       data_arr should have fields 't1','t2','u','v','vis','amp','sigma'

       returns: (uv, vis, amp, sigma)
    """

    vtype=vis_poldict[pol]
    atype=amp_poldict[pol]
    etype=sig_poldict[pol]

    t1 = data_arr['t1']
    t2 = data_arr['t2']

    sigma = data_arr[etype]
    amp = data_arr[atype]
    try:
        vis = data_arr[vtype]
    except ValueError:
        vis = amp.astype('c16')

    snrmask = np.abs(amp/sigma) >= snrcut

    if type(systematic_noise) is dict:
        sys_level = np.zeros(len(t1))
        for i in range(len(t1)):
            if t1[i] in systematic_noise.keys():
                t1sys = systematic_noise[t1[i]]
            else:
                t1sys = 0.
            if t2[i] in systematic_noise.keys():
                t2sys = systematic_noise[t2[i]]
            else:
                t2sys = 0.

            if t1sys<0 or t2sys<0:
                sys_level[i] = -1
            else:
                sys_level[i] = np.sqrt(t1sys**2 + t2sys**2)
    else:
        sys_level = np.sqrt(2)*systematic_noise*np.ones(len(t1))

    mask = sys_level>=0.
    mask = snrmask * mask

    sigma = np.linalg.norm([sigma, sys_level*np.abs(amp)], axis=0)[mask]
    vis = vis[mask]
    amp = amp[mask]
    uv = np.hstack((data_arr['u'].reshape(-1,1), data_arr['v'].reshape(-1,1)))[mask]
    return (uv, vis, amp, sigma)

def chisqdata_vis(Obsdata, Prior, mask, pol='I', **kwargs):
    """Return the data, sigmas, and fourier matrix for visibilities
    """

    # unpack keyword args
    systematic_noise = kwargs.get('systematic_noise',0.)
    snrcut = kwargs.get('snrcut',0.)
    debias = kwargs.get('debias',True)
    weighting = kwargs.get('weighting','natural')

    # unpack data
    vtype=vis_poldict[pol]
    atype=amp_poldict[pol]
    etype=sig_poldict[pol]
    data_arr = Obsdata.unpack(['t1','t2','u','v',vtype,atype,etype], debias=debias)
    (uv, vis, amp, sigma) = apply_systematic_noise_snrcut(data_arr, systematic_noise, snrcut, pol)

    # make fourier matrix
    A = ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv, pulse=Prior.pulse, mask=mask)

    return (vis, sigma, A)

def chisqdata_amp(Obsdata, Prior, mask, pol='I',**kwargs):
    """Return the data, sigmas, and fourier matrix for visibility amplitudes
    """

    # unpack keyword args
    systematic_noise = kwargs.get('systematic_noise',0.)
    snrcut = kwargs.get('snrcut',0.)
    debias = kwargs.get('debias',True)
    weighting = kwargs.get('weighting','natural')

    # unpack data
    vtype=vis_poldict[pol]
    atype=amp_poldict[pol]
    etype=sig_poldict[pol]
    if (Obsdata.amp is None) or (len(Obsdata.amp)==0) or pol!='I':
        data_arr = Obsdata.unpack(['t1','t2','u','v',vtype,atype,etype], debias=debias)

    else: # TODO -- pre-computed  with not stokes I? 
        print("Using pre-computed amplitude table in amplitude chi^2!")
        if not type(Obsdata.amp) in [np.ndarray, np.recarray]:
            raise Exception("pre-computed amplitude table is not a numpy rec array!")
        data_arr = Obsdata.amp

   
    # apply systematic noise and SNR cut
    # TODO -- after pre-computed??
    (uv, vis, amp, sigma) = apply_systematic_noise_snrcut(data_arr, systematic_noise, snrcut, pol)

    # data weighting
    if weighting=='uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # make fourier matrix
    A = ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv, pulse=Prior.pulse, mask=mask)

    return (amp, sigma, A)

def chisqdata_bs(Obsdata, Prior, mask, pol='I',**kwargs):
    """return the data, sigmas, and fourier matrices for bispectra
    """

    # unpack keyword args
    #systematic_noise = kwargs.get('systematic_noise',0.) #this will break with a systematic noise dict
    maxset = kwargs.get('maxset',False)
    if maxset: count='max'
    else: count='min'

    snrcut = kwargs.get('snrcut',0.)
    debias = kwargs.get('debias',True)
    weighting = kwargs.get('weighting','natural')

    # unpack data
    vtype=vis_poldict[pol]
    if (Obsdata.bispec is None) or (len(Obsdata.bispec)==0) or pol!='I':
        biarr = Obsdata.bispectra(mode="all", vtype=vtype, count=count,snrcut=snrcut)

    else: # TODO -- pre-computed  with not stokes I? 
        print("Using pre-computed bispectrum table in cphase chi^2!")
        if not type(Obsdata.bispec) in [np.ndarray, np.recarray]:
            raise Exception("pre-computed bispectrum table is not a numpy rec array!")
        biarr = Obsdata.bispec
        # reduce to a minimal set 
        if count!='max':   
            biarr = reduce_tri_minimal(Obsdata, biarr)

    uv1 = np.hstack((biarr['u1'].reshape(-1,1), biarr['v1'].reshape(-1,1)))
    uv2 = np.hstack((biarr['u2'].reshape(-1,1), biarr['v2'].reshape(-1,1)))
    uv3 = np.hstack((biarr['u3'].reshape(-1,1), biarr['v3'].reshape(-1,1)))
    bi = biarr['bispec']
    sigma = biarr['sigmab']

    #add systematic noise
    #sigma = np.linalg.norm([biarr['sigmab'], systematic_noise*np.abs(biarr['bispec'])], axis=0)

    # data weighting
    if weighting=='uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))


    # make fourier matrices
    A3 = (ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse, mask=mask)
         )

    return (bi, sigma, A3)

def chisqdata_cphase(Obsdata, Prior, mask, pol='I',**kwargs):
    """Return the data, sigmas, and fourier matrices for closure phases
    """

    # unpack keyword args
    maxset = kwargs.get('maxset',False)
    uv_min = kwargs.get('cp_uv_min', False)
    if maxset: count='max'
    else: count='min'

    snrcut = kwargs.get('snrcut',0.)
    systematic_cphase_noise = kwargs.get('systematic_cphase_noise',0.)
    weighting = kwargs.get('weighting','natural')

    # unpack data
    vtype=vis_poldict[pol]
    if (Obsdata.cphase is None) or (len(Obsdata.cphase)==0) or pol!='I':
        clphasearr = Obsdata.c_phases(mode="all", vtype=vtype, count=count, uv_min=uv_min, snrcut=snrcut)
    else: #TODO precomputed with not Stokes I
        print("Using pre-computed cphase table in cphase chi^2!")
        if not type(Obsdata.cphase) in [np.ndarray, np.recarray]:
            raise Exception("pre-computed closure phase table is not a numpy rec array!")
        clphasearr = Obsdata.cphase
        # reduce to a minimal set
        if count!='max':       
            clphasearr = reduce_tri_minimal(Obsdata, clphasearr)

    uv1 = np.hstack((clphasearr['u1'].reshape(-1,1), clphasearr['v1'].reshape(-1,1)))
    uv2 = np.hstack((clphasearr['u2'].reshape(-1,1), clphasearr['v2'].reshape(-1,1)))
    uv3 = np.hstack((clphasearr['u3'].reshape(-1,1), clphasearr['v3'].reshape(-1,1)))
    clphase = clphasearr['cphase']
    sigma = clphasearr['sigmacp']

    #add systematic cphase noise (in DEGREES)
    sigma = np.linalg.norm([sigma, systematic_cphase_noise*np.ones(len(sigma))], axis=0)

    # data weighting
    if weighting=='uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # make fourier matrices
    A3 = (ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse, mask=mask)
         )
    return (clphase, sigma, A3)

def chisqdata_camp(Obsdata, Prior, mask, pol='I',**kwargs):
    """Return the data, sigmas, and fourier matrices for closure amplitudes
    """
    # unpack keyword args
    maxset = kwargs.get('maxset',False)
    if maxset: count='max'
    else: count='min'

    snrcut = kwargs.get('snrcut',0.)
    debias = kwargs.get('debias',True)
    weighting = kwargs.get('weighting','natural')

    # unpack data & mask low snr points
    vtype=vis_poldict[pol]
    if (Obsdata.camp is None) or (len(Obsdata.camp)==0) or pol!='I':
        clamparr = Obsdata.c_amplitudes(mode='all', count=count, ctype='camp', debias=debias, snrcut=snrcut)
    else: # TODO -- pre-computed  with not stokes I? 
        print("Using pre-computed closure amplitude table in closure amplitude chi^2!")
        if not type(Obsdata.camp) in [np.ndarray, np.recarray]:
            raise Exception("pre-computed closure amplitude table is not a numpy rec array!")
        clamparr = Obsdata.camp
        # reduce to a minimal set
        if count!='max':       
            clamparr = reduce_quad_minimal(Obsdata, clamparr, ctype='camp')

    uv1 = np.hstack((clamparr['u1'].reshape(-1,1), clamparr['v1'].reshape(-1,1)))
    uv2 = np.hstack((clamparr['u2'].reshape(-1,1), clamparr['v2'].reshape(-1,1)))
    uv3 = np.hstack((clamparr['u3'].reshape(-1,1), clamparr['v3'].reshape(-1,1)))
    uv4 = np.hstack((clamparr['u4'].reshape(-1,1), clamparr['v4'].reshape(-1,1)))
    clamp = clamparr['camp']
    sigma = clamparr['sigmaca']

    # data weighting
    if weighting=='uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # make fourier matrices
    A4 = (ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv4, pulse=Prior.pulse, mask=mask)
         )

    return (clamp, sigma, A4)

def chisqdata_logcamp(Obsdata, Prior, mask,pol='I', **kwargs):
    """Return the data, sigmas, and fourier matrices for log closure amplitudes
    """
    # unpack keyword args
    maxset = kwargs.get('maxset',False)
    if maxset: count='max'
    else: count='min'

    snrcut = kwargs.get('snrcut',0.)
    debias = kwargs.get('debias',True)
    weighting = kwargs.get('weighting','natural')

    # unpack data & mask low snr points
    vtype=vis_poldict[pol]
    if (Obsdata.logcamp is None) or (len(Obsdata.logcamp)==0)  or pol!='I':
        clamparr = Obsdata.c_amplitudes(mode='all', count=count, vtype=vtype, ctype='logcamp', debias=debias, snrcut=snrcut)
    else: # TODO -- pre-computed  with not stokes I? 
        print("Using pre-computed log closure amplitude table in log closure amplitude chi^2!")
        if not type(Obsdata.logcamp) in [np.ndarray, np.recarray]:
            raise Exception("pre-computed log closure amplitude table is not a numpy rec array!")
        clamparr = Obsdata.logcamp
        # reduce to a minimal set
        if count!='max':       
            clamparr = reduce_quad_minimal(Obsdata, clamparr, ctype='logcamp')

    uv1 = np.hstack((clamparr['u1'].reshape(-1,1), clamparr['v1'].reshape(-1,1)))
    uv2 = np.hstack((clamparr['u2'].reshape(-1,1), clamparr['v2'].reshape(-1,1)))
    uv3 = np.hstack((clamparr['u3'].reshape(-1,1), clamparr['v3'].reshape(-1,1)))
    uv4 = np.hstack((clamparr['u4'].reshape(-1,1), clamparr['v4'].reshape(-1,1)))
    clamp = clamparr['camp']
    sigma = clamparr['sigmaca']

    # data weighting
    if weighting=='uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # make fourier matrices
    A4 = (ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv4, pulse=Prior.pulse, mask=mask)
         )

    return (clamp, sigma, A4)


##################################################################################################
# FFT Chi^2 Data functions
##################################################################################################
def chisqdata_vis_fft(Obsdata, Prior,pol='I', **kwargs):
    """Return the data, sigmas, uv points, and FFT info for visibilities
    """

    # unpack keyword args
    systematic_noise = kwargs.get('systematic_noise',0.)
    snrcut = kwargs.get('snrcut',0.)
    debias = kwargs.get('debias',True)
    weighting = kwargs.get('weighting','natural')
    fft_pad_factor = kwargs.get('fft_pad_factor',FFT_PAD_DEFAULT)
    conv_func = kwargs.get('conv_func', GRIDDER_CONV_FUNC_DEFAULT)
    p_rad = kwargs.get('p_rad', GRIDDER_P_RAD_DEFAULT)
    order = kwargs.get('order', FFT_INTERP_DEFAULT)

    # unpack data
    vtype=vis_poldict[pol]
    atype=amp_poldict[pol]
    etype=sig_poldict[pol]
    data_arr = Obsdata.unpack(['t1','t2','u','v',vtype,atype,etype], debias=debias)
    (uv, vis, amp, sigma) = apply_systematic_noise_snrcut(data_arr, systematic_noise, snrcut, pol)

    # data weighting
    if weighting=='uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # prepare image and fft info objects
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    im_info = ImInfo(Prior.xdim, Prior.ydim, npad, Prior.psize, Prior.pulse)
    gs_info = make_gridder_and_sampler_info(im_info, uv, conv_func=conv_func, p_rad=p_rad, order=order)

    sampler_info_list = [gs_info[0]]
    gridder_info_list = [gs_info[1]]
    A = (im_info, sampler_info_list, gridder_info_list)

    return (vis, sigma, A)

def chisqdata_amp_fft(Obsdata, Prior,pol='I', **kwargs):

    """Return the data, sigmas, uv points, and FFT info for visibility amplitudes
    """

    # unpack keyword args
    systematic_noise = kwargs.get('systematic_noise',0.)
    snrcut = kwargs.get('snrcut',0.)
    debias = kwargs.get('debias',True)
    weighting = kwargs.get('weighting','natural')
    fft_pad_factor = kwargs.get('fft_pad_factor',FFT_PAD_DEFAULT)
    conv_func = kwargs.get('conv_func', GRIDDER_CONV_FUNC_DEFAULT)
    p_rad = kwargs.get('p_rad', GRIDDER_P_RAD_DEFAULT)
    order = kwargs.get('order', FFT_INTERP_DEFAULT)

    # unpack data
    vtype=vis_poldict[pol]
    atype=amp_poldict[pol]
    etype=sig_poldict[pol]
    if (Obsdata.amp is None) or (len(Obsdata.amp)==0) or pol!='I':
        data_arr = Obsdata.unpack(['t1','t2','u','v',vtype,atype,etype], debias=debias)
    else: # TODO -- pre-computed  with not stokes I? 
        print("Using pre-computed amplitude table in amplitude chi^2!")
        if not type(Obsdata.amp) in [np.ndarray, np.recarray]:
            raise Exception("pre-computed amplitude table is not a numpy rec array!")
        data_arr = Obsdata.amp

    # apply systematic noise
    (uv, vis, amp, sigma) = apply_systematic_noise_snrcut(data_arr, systematic_noise,snrcut, pol)

    # data weighting
    if weighting=='uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # prepare image and fft info objects
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    im_info = ImInfo(Prior.xdim, Prior.ydim, npad, Prior.psize, Prior.pulse)
    gs_info = make_gridder_and_sampler_info(im_info, uv, conv_func=conv_func, p_rad=p_rad, order=order)

    sampler_info_list = [gs_info[0]]
    gridder_info_list = [gs_info[1]]
    A = (im_info, sampler_info_list, gridder_info_list)

    return (amp, sigma, A)

def chisqdata_bs_fft(Obsdata, Prior, pol='I',**kwargs):

    """Return the data, sigmas, uv points, and FFT info for bispectra
    """

    # unpack keyword args
    #systematic_noise = kwargs.get('systematic_noise',0.) #this will break with a systematic noise dict
    maxset = kwargs.get('maxset',False)
    if maxset: count='max'
    else: count='min'

    snrcut = kwargs.get('snrcut',0.)
    weighting = kwargs.get('weighting','natural')
    fft_pad_factor = kwargs.get('fft_pad_factor',FFT_PAD_DEFAULT)
    conv_func = kwargs.get('conv_func', GRIDDER_CONV_FUNC_DEFAULT)
    p_rad = kwargs.get('p_rad', GRIDDER_P_RAD_DEFAULT)
    order = kwargs.get('order', FFT_INTERP_DEFAULT)

    # unpack data
    vtype=vis_poldict[pol]
    if (Obsdata.bispec is None) or (len(Obsdata.bispec)==0) or pol!='I':
        biarr = Obsdata.bispectra(mode="all", vtype=vtype, count=count, snrcut=snrcut)
    else: # TODO -- pre-computed  with not stokes I? 
        print("Using pre-computed bispectrum table in cphase chi^2!")
        if not type(Obsdata.bispec) in [np.ndarray, np.recarray]:
            raise Exception("pre-computed bispectrum table is not a numpy rec array!")
        biarr = Obsdata.bispec
        # reduce to a minimal set 
        if count!='max':   
            biarr = reduce_tri_minimal(Obsdata, biarr)

    uv1 = np.hstack((biarr['u1'].reshape(-1,1), biarr['v1'].reshape(-1,1)))
    uv2 = np.hstack((biarr['u2'].reshape(-1,1), biarr['v2'].reshape(-1,1)))
    uv3 = np.hstack((biarr['u3'].reshape(-1,1), biarr['v3'].reshape(-1,1)))
    bi = biarr['bispec']
    sigma = biarr['sigmab']

    #add systematic noise
    #sigma = np.linalg.norm([biarr['sigmab'], systematic_noise*np.abs(biarr['bispec'])], axis=0)
    
    # data weighting
    if weighting=='uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # prepare image and fft info objects
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    im_info = ImInfo(Prior.xdim, Prior.ydim, npad, Prior.psize, Prior.pulse)
    gs_info1 = make_gridder_and_sampler_info(im_info, uv1, conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info2 = make_gridder_and_sampler_info(im_info, uv2, conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info3 = make_gridder_and_sampler_info(im_info, uv3, conv_func=conv_func, p_rad=p_rad, order=order)

    sampler_info_list = [gs_info1[0],gs_info2[0],gs_info3[0]]
    gridder_info_list = [gs_info1[1],gs_info2[1],gs_info3[1]]
    A = (im_info, sampler_info_list, gridder_info_list)

    return (bi, sigma, A)

def chisqdata_cphase_fft(Obsdata, Prior, pol='I',**kwargs):

    """Return the data, sigmas, uv points, and FFT info for closure phases
    """
    # unpack keyword args
    maxset = kwargs.get('maxset',False)
    uv_min = kwargs.get('cp_uv_min', False)
    if maxset: count='max'
    else: count='min'

    snrcut = kwargs.get('snrcut',0.)
    weighting = kwargs.get('weighting','natural')
    systematic_cphase_noise = kwargs.get('systematic_cphase_noise', 0.)
    fft_pad_factor = kwargs.get('fft_pad_factor',FFT_PAD_DEFAULT)
    conv_func = kwargs.get('conv_func', GRIDDER_CONV_FUNC_DEFAULT)
    p_rad = kwargs.get('p_rad', GRIDDER_P_RAD_DEFAULT)
    order = kwargs.get('order', FFT_INTERP_DEFAULT)

    # unpack data
    vtype=vis_poldict[pol]
    if (Obsdata.cphase is None) or (len(Obsdata.cphase)==0) or pol!='I':
        clphasearr = Obsdata.c_phases(mode="all", vtype=vtype, count=count, uv_min=uv_min, snrcut=snrcut)
    else: #TODO precomputed with not Stokes I
        print("Using pre-computed cphase table in cphase chi^2!")
        if not type(Obsdata.cphase) in [np.ndarray, np.recarray]:
            raise Exception("pre-computed closure phase table is not a numpy rec array!")
        clphasearr = Obsdata.cphase
        # reduce to a minimal set
        if count!='max':       
            clphasearr = reduce_tri_minimal(Obsdata, clphasearr)


    uv1 = np.hstack((clphasearr['u1'].reshape(-1,1), clphasearr['v1'].reshape(-1,1)))
    uv2 = np.hstack((clphasearr['u2'].reshape(-1,1), clphasearr['v2'].reshape(-1,1)))
    uv3 = np.hstack((clphasearr['u3'].reshape(-1,1), clphasearr['v3'].reshape(-1,1)))
    clphase = clphasearr['cphase']
    sigma = clphasearr['sigmacp']

    #add systematic cphase noise (in DEGREES)
    sigma = np.linalg.norm([sigma, systematic_cphase_noise*np.ones(len(sigma))], axis=0)

    # data weighting
    if weighting=='uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # prepare image and fft info objects
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    im_info = ImInfo(Prior.xdim, Prior.ydim, npad, Prior.psize, Prior.pulse)
    gs_info1 = make_gridder_and_sampler_info(im_info, uv1, conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info2 = make_gridder_and_sampler_info(im_info, uv2, conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info3 = make_gridder_and_sampler_info(im_info, uv3, conv_func=conv_func, p_rad=p_rad, order=order)

    sampler_info_list = [gs_info1[0],gs_info2[0],gs_info3[0]]
    gridder_info_list = [gs_info1[1],gs_info2[1],gs_info3[1]]
    A = (im_info, sampler_info_list, gridder_info_list)

    return (clphase, sigma, A)

def chisqdata_camp_fft(Obsdata, Prior, pol='I',**kwargs):

    """Return the data, sigmas, uv points, and FFT info for closure amplitudes
    """

    # unpack keyword args
    maxset = kwargs.get('maxset',False)
    if maxset: count='max'
    else: count='min'

    snrcut = kwargs.get('snrcut',0.)
    debias = kwargs.get('debias',True)
    weighting = kwargs.get('weighting','natural')
    fft_pad_factor = kwargs.get('fft_pad_factor',FFT_PAD_DEFAULT)
    conv_func = kwargs.get('conv_func', GRIDDER_CONV_FUNC_DEFAULT)
    p_rad = kwargs.get('p_rad', GRIDDER_P_RAD_DEFAULT)
    order = kwargs.get('order', FFT_INTERP_DEFAULT)

    # unpack data
    vtype=vis_poldict[pol]
    if (Obsdata.camp is None) or (len(Obsdata.camp)==0) or pol!='I':
        clamparr = Obsdata.c_amplitudes(mode='all', count=count, ctype='camp', debias=debias,snrcut=snrcut)
    else: # TODO -- pre-computed  with not stokes I? 
        print("Using pre-computed closure amplitude table in closure amplitude chi^2!")
        if not type(Obsdata.camp) in [np.ndarray, np.recarray]:
            raise Exception("pre-computed closure amplitude table is not a numpy rec array!")
        clamparr = Obsdata.camp
        # reduce to a minimal set
        if count!='max':       
            clamparr = reduce_quad_minimal(Obsdata, clamparr, ctype='camp')


    uv1 = np.hstack((clamparr['u1'].reshape(-1,1), clamparr['v1'].reshape(-1,1)))
    uv2 = np.hstack((clamparr['u2'].reshape(-1,1), clamparr['v2'].reshape(-1,1)))
    uv3 = np.hstack((clamparr['u3'].reshape(-1,1), clamparr['v3'].reshape(-1,1)))
    uv4 = np.hstack((clamparr['u4'].reshape(-1,1), clamparr['v4'].reshape(-1,1)))
    clamp = clamparr['camp']
    sigma = clamparr['sigmaca']

    # data weighting
    if weighting=='uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # prepare image and fft info objects
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    im_info = ImInfo(Prior.xdim, Prior.ydim, npad, Prior.psize, Prior.pulse)
    gs_info1 = make_gridder_and_sampler_info(im_info, uv1, conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info2 = make_gridder_and_sampler_info(im_info, uv2, conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info3 = make_gridder_and_sampler_info(im_info, uv3, conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info4 = make_gridder_and_sampler_info(im_info, uv4, conv_func=conv_func, p_rad=p_rad, order=order)

    sampler_info_list = [gs_info1[0],gs_info2[0],gs_info3[0],gs_info4[0]]
    gridder_info_list = [gs_info1[1],gs_info2[1],gs_info3[1],gs_info4[1]]
    A = (im_info, sampler_info_list, gridder_info_list)

    return (clamp, sigma, A)

def chisqdata_logcamp_fft(Obsdata, Prior,pol='I', **kwargs):

    """Return the data, sigmas, uv points, and FFT info for log closure amplitudes
    """
    # unpack keyword args
    maxset = kwargs.get('maxset',False)
    if maxset: count='max'
    else: count='min'

    snrcut = kwargs.get('snrcut',0.)
    debias = kwargs.get('debias',True)
    weighting = kwargs.get('weighting','natural')
    fft_pad_factor = kwargs.get('fft_pad_factor',FFT_PAD_DEFAULT)
    conv_func = kwargs.get('conv_func', GRIDDER_CONV_FUNC_DEFAULT)
    p_rad = kwargs.get('p_rad', GRIDDER_P_RAD_DEFAULT)
    order = kwargs.get('order', FFT_INTERP_DEFAULT)

    # unpack data
    vtype=vis_poldict[pol]
    if (Obsdata.logcamp is None) or (len(Obsdata.logcamp)==0)  or pol!='I':
        clamparr = Obsdata.c_amplitudes(mode='all', count=count, vtype=vtype, ctype='logcamp', debias=debias, snrcut=snrcut)
    else: # TODO -- pre-computed  with not stokes I? 
        print("Using pre-computed log closure amplitude table in log closure amplitude chi^2!")
        if not type(Obsdata.logcamp) in [np.ndarray, np.recarray]:
            raise Exception("pre-computed log closure amplitude table is not a numpy rec array!")
        clamparr = Obsdata.logcamp
        # reduce to a minimal set
        if count!='max':       
            clamparr = reduce_quad_minimal(Obsdata, clamparr, ctype='logcamp')

    uv1 = np.hstack((clamparr['u1'].reshape(-1,1), clamparr['v1'].reshape(-1,1)))
    uv2 = np.hstack((clamparr['u2'].reshape(-1,1), clamparr['v2'].reshape(-1,1)))
    uv3 = np.hstack((clamparr['u3'].reshape(-1,1), clamparr['v3'].reshape(-1,1)))
    uv4 = np.hstack((clamparr['u4'].reshape(-1,1), clamparr['v4'].reshape(-1,1)))
    clamp = clamparr['camp']
    sigma = clamparr['sigmaca']

    # data weighting
    if weighting=='uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # prepare image and fft info objects
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    im_info = ImInfo(Prior.xdim, Prior.ydim, npad, Prior.psize, Prior.pulse)
    gs_info1 = make_gridder_and_sampler_info(im_info, uv1, conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info2 = make_gridder_and_sampler_info(im_info, uv2, conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info3 = make_gridder_and_sampler_info(im_info, uv3, conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info4 = make_gridder_and_sampler_info(im_info, uv4, conv_func=conv_func, p_rad=p_rad, order=order)

    sampler_info_list = [gs_info1[0],gs_info2[0],gs_info3[0],gs_info4[0]]
    gridder_info_list = [gs_info1[1],gs_info2[1],gs_info3[1],gs_info4[1]]
    A = (im_info, sampler_info_list, gridder_info_list)

    return (clamp, sigma, A)

##################################################################################################
# NFFT Chi^2 Data functions
##################################################################################################
def chisqdata_vis_nfft(Obsdata, Prior, pol='I',**kwargs):

    """Return the visibilities, sigmas, uv points, and nfft info
    """
    if (Prior.xdim%2 or Prior.ydim%2):
        raise Exception("NFFT doesn't work with odd image dimensions!")

    # unpack keyword args
    systematic_noise = kwargs.get('systematic_noise',0.)
    snrcut = kwargs.get('snrcut',0.)
    debias = kwargs.get('debias',True)
    weighting = kwargs.get('weighting','natural')
    fft_pad_factor = kwargs.get('fft_pad_factor',FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', GRIDDER_P_RAD_DEFAULT)

    # unpack data
    vtype=vis_poldict[pol]
    atype=amp_poldict[pol]
    etype=sig_poldict[pol]
    data_arr = Obsdata.unpack(['t1','t2','u','v',vtype,atype,etype], debias=debias)
    (uv, vis, amp, sigma) = apply_systematic_noise_snrcut(data_arr, systematic_noise, snrcut, pol)

    # data weighting
    if weighting=='uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv)
    A = [A1]

    return (vis, sigma, A)

def chisqdata_amp_nfft(Obsdata, Prior, pol='I',**kwargs):

    """Return the amplitudes, sigmas, uv points, and nfft info
    """
    if (Prior.xdim%2 or Prior.ydim%2):
        raise Exception("NFFT doesn't work with odd image dimensions!")

    # unpack keyword args
    systematic_noise = kwargs.get('systematic_noise',0.)
    snrcut = kwargs.get('snrcut',0.)
    debias = kwargs.get('debias',True)
    weighting = kwargs.get('weighting','natural')
    fft_pad_factor = kwargs.get('fft_pad_factor',FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', GRIDDER_P_RAD_DEFAULT)

    # unpack data
    vtype=vis_poldict[pol]
    atype=amp_poldict[pol]
    etype=sig_poldict[pol]
    if (Obsdata.amp is None) or (len(Obsdata.amp)==0) or pol!='I':
        data_arr = Obsdata.unpack(['t1','t2','u','v',vtype,atype,etype], debias=debias)
    else: # TODO -- pre-computed  with not stokes I? 
        print("Using pre-computed amplitude table in amplitude chi^2!")
        if not type(Obsdata.amp) in [np.ndarray, np.recarray]:
            raise Exception("pre-computed amplitude table is not a numpy rec array!")
        data_arr = Obsdata.amp
    
    # apply systematic noise 
    (uv, vis, amp, sigma) = apply_systematic_noise_snrcut(data_arr, systematic_noise, snrcut, pol)

    # data weighting
    if weighting=='uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv)
    A = [A1]

    return (amp, sigma, A)

def chisqdata_bs_nfft(Obsdata, Prior, pol='I',**kwargs):

    """Return the bispectra, sigmas, uv points, and nfft info
    """
    if (Prior.xdim%2 or Prior.ydim%2):
        raise Exception("NFFT doesn't work with odd image dimensions!")

    # unpack keyword args
    #systematic_noise = kwargs.get('systematic_noise',0.) #this will break with a systematic_noise dict
    maxset = kwargs.get('maxset',False)
    if maxset: count='max'
    else: count='min'

    snrcut = kwargs.get('snrcut',0.)
    weighting = kwargs.get('weighting','natural')
    fft_pad_factor = kwargs.get('fft_pad_factor',FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', GRIDDER_P_RAD_DEFAULT)

    # unpack data
    vtype=vis_poldict[pol]
    if (Obsdata.bispec is None) or (len(Obsdata.bispec)==0) or pol!='I':
        biarr = Obsdata.bispectra(mode="all", vtype=vtype, count=count, snrcut=snrcut)
    else: # TODO -- pre-computed  with not stokes I? 
        print("Using pre-computed bispectrum table in cphase chi^2!")
        if not type(Obsdata.bispec) in [np.ndarray, np.recarray]:
            raise Exception("pre-computed bispectrum table is not a numpy rec array!")
        biarr = Obsdata.bispec
        # reduce to a minimal set 
        if count!='max':   
            biarr = reduce_tri_minimal(Obsdata, biarr)

    uv1 = np.hstack((biarr['u1'].reshape(-1,1), biarr['v1'].reshape(-1,1)))
    uv2 = np.hstack((biarr['u2'].reshape(-1,1), biarr['v2'].reshape(-1,1)))
    uv3 = np.hstack((biarr['u3'].reshape(-1,1), biarr['v3'].reshape(-1,1)))
    bi = biarr['bispec']
    sigma = biarr['sigmab']

    #add systematic noise
    #sigma = np.linalg.norm([biarr['sigmab'], systematic_noise*np.abs(biarr['bispec'])], axis=0)

    # data weighting
    if weighting=='uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv1)
    A2 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv2)
    A3 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv3)
    A = [A1,A2,A3]

    return (bi, sigma, A)

def chisqdata_cphase_nfft(Obsdata, Prior, pol='I',**kwargs):

    """Return the closure phases, sigmas, uv points, and nfft info
    """
    if (Prior.xdim%2 or Prior.ydim%2):
        raise Exception("NFFT doesn't work with odd image dimensions!")

    # unpack keyword args    
    maxset = kwargs.get('maxset',False)
    uv_min = kwargs.get('cp_uv_min', False)
    if maxset: count='max'
    else: count='min'

    snrcut = kwargs.get('snrcut',0.)
    weighting = kwargs.get('weighting','natural')
    systematic_cphase_noise = kwargs.get('systematic_cphase_noise',0.)
    fft_pad_factor = kwargs.get('fft_pad_factor',FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', GRIDDER_P_RAD_DEFAULT)

    # unpack data
    vtype=vis_poldict[pol]
    if (Obsdata.cphase is None) or (len(Obsdata.cphase)==0) or pol!='I':
        clphasearr = Obsdata.c_phases(mode="all", vtype=vtype, count=count, uv_min=uv_min, snrcut=snrcut)
    else: #TODO precomputed with not Stokes I
        print("Using pre-computed cphase table in cphase chi^2!")
        if not type(Obsdata.cphase) in [np.ndarray, np.recarray]:
            raise Exception("pre-computed closure phase table is not a numpy rec array!")
        clphasearr = Obsdata.cphase
        # reduce to a minimal set
        if count!='max':       
            clphasearr = reduce_tri_minimal(Obsdata, clphasearr)

    uv1 = np.hstack((clphasearr['u1'].reshape(-1,1), clphasearr['v1'].reshape(-1,1)))
    uv2 = np.hstack((clphasearr['u2'].reshape(-1,1), clphasearr['v2'].reshape(-1,1)))
    uv3 = np.hstack((clphasearr['u3'].reshape(-1,1), clphasearr['v3'].reshape(-1,1)))
    clphase = clphasearr['cphase']
    sigma = clphasearr['sigmacp']

    #add systematic cphase noise (in DEGREES)
    sigma = np.linalg.norm([sigma, systematic_cphase_noise*np.ones(len(sigma))], axis=0)

    # data weighting
    if weighting=='uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv1)
    A2 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv2)
    A3 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv3)
    A = [A1,A2,A3]

    return (clphase, sigma, A)

def chisqdata_camp_nfft(Obsdata, Prior, pol='I',**kwargs):

    """Return the closure phases, sigmas, uv points, and nfft info
    """
    if (Prior.xdim%2 or Prior.ydim%2):
        raise Exception("NFFT doesn't work with odd image dimensions!")

    # unpack keyword args
    maxset = kwargs.get('maxset',False)
    if maxset: count='max'
    else: count='min'

    snrcut = kwargs.get('snrcut',0.)
    debias = kwargs.get('debias',True)
    weighting = kwargs.get('weighting','natural')
    fft_pad_factor = kwargs.get('fft_pad_factor',FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', GRIDDER_P_RAD_DEFAULT)

    # unpack data
    vtype=vis_poldict[pol]
    if (Obsdata.camp is None) or (len(Obsdata.camp)==0) or pol!='I':
        clamparr = Obsdata.c_amplitudes(mode='all', count=count, ctype='camp', debias=debias, snrcut=snrcut)
    else: # TODO -- pre-computed  with not stokes I? 
        print("Using pre-computed closure amplitude table in closure amplitude chi^2!")
        if not type(Obsdata.camp) in [np.ndarray, np.recarray]:
            raise Exception("pre-computed closure amplitude table is not a numpy rec array!")
        clamparr = Obsdata.camp
        # reduce to a minimal set
        if count!='max':       
            clamparr = reduce_quad_minimal(Obsdata, clamparr, ctype='camp')

    uv1 = np.hstack((clamparr['u1'].reshape(-1,1), clamparr['v1'].reshape(-1,1)))
    uv2 = np.hstack((clamparr['u2'].reshape(-1,1), clamparr['v2'].reshape(-1,1)))
    uv3 = np.hstack((clamparr['u3'].reshape(-1,1), clamparr['v3'].reshape(-1,1)))
    uv4 = np.hstack((clamparr['u4'].reshape(-1,1), clamparr['v4'].reshape(-1,1)))
    clamp = clamparr['camp']
    sigma = clamparr['sigmaca']

    # data weighting
    if weighting=='uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv1)
    A2 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv2)
    A3 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv3)
    A4 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv4)
    A = [A1,A2,A3,A4]

    return (clamp, sigma, A)

def chisqdata_logcamp_nfft(Obsdata, Prior,pol='I', **kwargs):

    """Return the closure phases, sigmas, uv points, and nfft info
    """
    if (Prior.xdim%2 or Prior.ydim%2):
        raise Exception("NFFT doesn't work with odd image dimensions!")

    # unpack keyword args
    maxset = kwargs.get('maxset',False)
    if maxset: count='max'
    else: count='min'

    snrcut = kwargs.get('snrcut',0.)
    debias = kwargs.get('debias',True)
    weighting = kwargs.get('weighting','natural')
    fft_pad_factor = kwargs.get('fft_pad_factor',FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', GRIDDER_P_RAD_DEFAULT)

    # unpack data
    vtype=vis_poldict[pol]
    if (Obsdata.logcamp is None) or (len(Obsdata.logcamp)==0)  or pol!='I':
        clamparr = Obsdata.c_amplitudes(mode='all', count=count, vtype=vtype, ctype='logcamp', debias=debias, snrcut=snrcut)
    else: # TODO -- pre-computed  with not stokes I? 
        print("Using pre-computed log closure amplitude table in log closure amplitude chi^2!")
        if not type(Obsdata.logcamp) in [np.ndarray, np.recarray]:
            raise Exception("pre-computed log closure amplitude table is not a numpy rec array!")
        clamparr = Obsdata.logcamp
        # reduce to a minimal set
        if count!='max':       
            clamparr = reduce_quad_minimal(Obsdata, clamparr, ctype='logcamp')

    uv1 = np.hstack((clamparr['u1'].reshape(-1,1), clamparr['v1'].reshape(-1,1)))
    uv2 = np.hstack((clamparr['u2'].reshape(-1,1), clamparr['v2'].reshape(-1,1)))
    uv3 = np.hstack((clamparr['u3'].reshape(-1,1), clamparr['v3'].reshape(-1,1)))
    uv4 = np.hstack((clamparr['u4'].reshape(-1,1), clamparr['v4'].reshape(-1,1)))
    clamp = clamparr['camp']
    sigma = clamparr['sigmaca']

    # data weighting
    if weighting=='uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv1)
    A2 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv2)
    A3 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv3)
    A4 = NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv4)
    A = [A1,A2,A3,A4]

    return (clamp, sigma, A)


##################################################################################################
# Restoring ,Embedding, and Plotting Functions
##################################################################################################
def plot_i(im, Prior, nit, chi2_dict, **kwargs):
    """Plot the total intensity image at each iteration
    """

    cmap = kwargs.get('cmap','afmhot')
    interpolation = kwargs.get('interpolation', 'gaussian')
    pol = kwargs.get('pol', '')
    scale = kwargs.get('scale',None)
    dynamic_range = kwargs.get('dynamic_range',1.e5)
    gamma = kwargs.get('dynamic_range',.5)

    plt.ion()
    plt.pause(1.e-6)
    plt.clf()

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

    plt.imshow(imarr, cmap=plt.get_cmap(cmap), interpolation=interpolation)
    xticks = ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
    yticks = ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plotstr = str(pol) + " : step: %i  " % nit
    for key in chi2_dict.keys():
        plotstr += "$\chi^2_{%s}$: %0.2f  " % (key, chi2_dict[key])
    plt.title(plotstr, fontsize=18)
    #plt.draw()


def embed(im, mask, clipfloor=0., randomfloor=False):
    """Embeds a 1d image array into the size of boolean embed mask
    """

    out=np.zeros(len(mask))

    # Here's a much faster version than before
    out[mask.nonzero()] = im

    if clipfloor != 0.0:
        if randomfloor: # prevent total variation gradient singularities
            out[(mask-1).nonzero()] = clipfloor * np.abs(np.random.normal(size=len((mask-1).nonzero())))
        else:
            out[(mask-1).nonzero()] = clipfloor

    return out

