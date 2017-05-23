# imager_dft.py
# Andrew Chael, 3/11/2017
# General imager for total intensity VLBI data

#TODO 
# add ffts and rename!
# add more general linearized energy functions
# debias closure amplitudes
# closure amplitude and phase covariance

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

import ehtim.image as image
from . import linearize_energy as le

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

from IPython import display

def conv_func_pill(x,y): 
    if abs(x) < 0.5 and abs(y) < 0.5: 
        out = 1.
    else: 
        out = 0.
    return out

def conv_func_gauss(x,ys):
    return np.exp(-(x**2 + y**2))

# There's a bug in my version of the spheroidal function of order 0! - gives nans for eta<1
def conv_func_spheroidal(x,y,p,m):
    etax = 2.*x/float(p)
    etay = 2.*x/float(p)
    psix =  abs(1-etax**2)**m * scipy.special.pro_rad1(m,0,0.5*np.pi*p,etax)[0] 
    psiy = abs(1-etay**2)**m * scipy.special.pro_rad1(m,0,0.5*np.pi*p,etay)[0]
    
    return psix*psiy

def sampler(uv, griddata, psize, order=3):
    """
    Samples griddata sampled at uv points in a npix x npix grid
    psize is the image domain pixel size of the corresponding real space image
    order is the order of the spline interpolation
    the griddata should already be rotated so u,v = 0,0 is in the center
    to agree with dft result another phase shift may be required
    """

    if griddata.shape[0] != griddata.shape[0]: 
        raise Exception("griddata should be a square array!")

    npix = griddata.shape[0]
    vu2  = np.hstack((uv[:,1].reshape(-1,1), uv[:,0].reshape(-1,1)))
    du   = 1.0/(npix*psize)
    vu2  = (vu2/du + 0.5*npix).T

    datare = nd.map_coordinates(np.real(vis_im), vu2, order=order)
    dataim = nd.map_coordinates(np.imag(vis_im), vu2, order=order)
    data = visdatare + 1j*visdataim

    return data

def gridder(uv, data, npix, psize, conv_func="pillbox", p_rad=1.):
    """
    Grids data sampled at uv points on a square npix x npix array
    psize is the image domain pixel size of the corresponding real space image
    conv_func is the convolution function: current options are "pillbox" and "gaussian"
    p_rad is the radius inside wich the conv_func is nonzero 
    """

    if len(uv) != len(data): 
        raise Exception("uv and data are not the same length!")
    if not (conv_func in ['pillbox','gaussian']):
        raise Exception("conv_func must be either 'pillbox' or 'gaussian'")

    vu2 = np.hstack((uv[:,1].reshape(-1,1), uv[:,0].reshape(-1,1)))
    du  = 1.0/(npix*psize)
    vu2 = (vu2/du + 0.5*npix)

    datagrid = np.zeros((npad, npad)).astype('c16')
    for k in range(len(data)):
        point = vu2[k]
        vispoint = data[k]

        vumin = np.ceil(point - prad).astype(int)
        vumax = np.floor(point + prad).astype(int)

        #print vumin, vumax
        for i in np.arange(vumin[0], vumax[0]+1):
            for j in np.arange(vumin[1], vumax[1]+1):
                if conv_func == 'pillbox':
                    visgrid[i,j] += conv_func_pill(j-point[1], i-point[0]) * vispoint

                elif conv_func == 'gaussian':
                    visgrid[i,j] += conv_func_gauss(j-point[1], i-point[0]) * vispoint
    
    return datagrid

##################################################################################################
# Constants; TODO: no need to keep them?
##################################################################################################
C = 299792458.0 
DEGREE = np.pi/180.0
RADPERAS = DEGREE/3600.0
RADPERUAS = RADPERAS/1e6
NHIST = 100 # number of steps to store for hessian approx

DATATERMS = ['vis', 'bs', 'amp', 'cphase', 'camp']
REGULARIZERS = ['gs', 'tv', 'l1', 'patch', 'simple']

nit = 0 # global variable to track the iteration number in the plotting callback

##################################################################################################
# Total Intensity Imager
##################################################################################################
def imager(Obsdata, InitIm, Prior, flux,
           d1='vis', d2=False, s1='simple', s2=False,
           alpha_s1=1, alpha_s2=1,
           alpha_d1=100, alpha_d2=100, 
           alpha_flux=500, alpha_cm=500, 
           clipfloor=0., datamin="gd", grads=True, logim=True, 
           maxit=100, stop=1e-10, ipynb=False, show_updates=True):
   
    """Run a general interferometric imager.
       
       Args:
           Obsdata (Obsdata): The Obsdata object with VLBI data
           Prior (Image): The Image object with the prior image 
           InitIm (Image): The Image object with the initial image for the minimization
           flux (float): The total flux of the output image in Jy
           d1 (str): The first data term; options are 'vis', 'bs', 'amp', 'cphase', 'camp'
           d2 (str): The second data term; options are 'vis', 'bs', 'amp', 'cphase', 'camp'
           s1 (str): The first regularizer; options are 'simple', 'gs', 'tv', 'l1', 'patch'
           s2 (str): The second regularizer; options are 'simple', 'gs', 'tv', 'l1', 'patch'
           alpha_d1 (float): The first data term weighting
           alpha_d2 (float): The second data term weighting
           alpha_s1 (float): The first regularizer term weighting
           alpha_s2 (float): The second regularizer term weighting
           alpha_flux (float): The weighting for the total flux constraint 
           alpha_cm (float): The weighting for the center of mass constraint
           
           clipfloor (float): The Jy/pixel level above which prior image pixels are varied
           datamin (str): If 'lin', linearized energy is used (currently only compatible with CHIRP)
           grads (bool): If True, analytic gradients are used
           logim (bool): If True, uses I = exp(I') change of variables

           maxit (int): Maximum number of minimizer iterations
           stop (float): The convergence criterion
           show_updates (bool): If True, displays the progress of the minimizer
           ipynb (bool): If True, adjusts the plotting for the ipython/jupyter notebook 

       Returns:
           Image: Image object with result
    """
    
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

    # Normalize prior image to total flux and limit imager range to prior values > clipfloor
    embed_mask = Prior.imvec > clipfloor
    nprior = (flux * Prior.imvec / np.sum((Prior.imvec)[embed_mask]))[embed_mask]
    ninit = (flux * InitIm.imvec / np.sum((InitIm.imvec)[embed_mask]))[embed_mask]

    # Get data and fourier matrices for the data terms
    (data1, sigma1, A1) = chisqdata(Obsdata, Prior, embed_mask, d1)
    (data2, sigma2, A2) = chisqdata(Obsdata, Prior, embed_mask, d2)

    # Coordinate matrix for center-of-mass constraint
    coord = Prior.psize * np.array([[[x,y] for x in np.arange(Prior.xdim//2,-Prior.xdim//2,-1)]
                                           for y in np.arange(Prior.ydim//2,-Prior.ydim//2,-1)])
    coord = coord.reshape(Prior.ydim*Prior.xdim, 2)        
    coord = coord[embed_mask]

    # Katie - if you are using the linearized energy then compute the A and b of your 
    # linearized equation given your current image
    #TODO generalize this linearization for the other data terms!
    if d1=='bs' and datamin=='lin':
        (Alin1, blin1) = le.computeLinTerms_bi(ninit, A1, data1, sigma1, 
                               len(ninit), alpha=alpha_d1)
    if d2=='bs' and datamin=='lin':
        (Alin2, blin2) = le.computeLinTerms_bi(ninit, A2, data2, sigma2, 
                               len(ninit), alpha=alpha_d2)


    # Define the chi^2 and chi^2 gradient
    def chisq1(imvec):
        return chisq(imvec, A1, data1, sigma1, d1)
    
    def chisq1grad(imvec):
        if d1=='bs' and datamin=='lin':
            c = 2.0/(2.0*len(sigma1)) * np.dot(Alin1.T, np.dot(Alin1, imvec) - blin1)
        else:
            c = chisqgrad(imvec, A1, data1, sigma1, d1)
        return c

    def chisq2(imvec):
        return chisq(imvec, A2, data2, sigma2, d2)
    
    def chisq2grad(imvec):
        if d1=='bs' and datamin=='lin':
            c = 2.0/(2.0*len(sigma2)) * np.dot(Alin2.T, np.dot(Alin2, imvec) - blin2)
        else:
            c = chisqgrad(imvec, A2, data2, sigma2, d2)
        return c

    # Define the regularizer and regularizer gradient
    def reg1(imvec):
        return regularizer(imvec, nprior, embed_mask, flux, Prior.xdim, Prior.ydim, Prior.psize, s1)

    def reg1grad(imvec):
        return regularizergrad(imvec, nprior, embed_mask, flux, Prior.xdim, Prior.ydim, Prior.psize, s1)

    def reg2(imvec):
        return regularizer(imvec, nprior, embed_mask, flux, Prior.xdim, Prior.ydim, Prior.psize, s2)

    def reg2grad(imvec):
        return regularizergrad(imvec, nprior, embed_mask, flux, Prior.xdim, Prior.ydim, Prior.psize, s2)

    # Define constraint functions
    def flux_constraint(imvec):
        #norm = flux**2
        norm = 1        
        return (np.sum(imvec) - flux)**2/norm

    def flux_constraint_grad(imvec):
        #norm = flux**2
        norm = 1
        return 2*(np.sum(imvec) - flux) / norm

    def cm_constraint(imvec):
        #norm = flux**2 * Prior.psize**2
        norm = 1        
        return (np.sum(imvec*coord[:,0])**2 + np.sum(imvec*coord[:,1])**2)/norm

    def cm_constraint_grad(imvec):
        #norm = flux**2 * Prior.psize**2
        norm = 1
        return 2*(np.sum(imvec*coord[:,0])*coord[:,0] + np.sum(imvec*coord[:,1])*coord[:,1]) / norm

    # Define the objective function and gradient
    def objfunc(imvec):
        if logim: imvec = np.exp(imvec)

        datterm  = alpha_d1 * (chisq1(imvec) - 1) + alpha_d2 * (chisq2(imvec) - 1)
        regterm  = alpha_s1 * reg1(imvec) + alpha_s2 * reg2(imvec)
        conterm  = alpha_flux * flux_constraint(imvec)  + alpha_cm * cm_constraint(imvec)

        return datterm + regterm + conterm

    def objgrad(imvec):
        if logim: imvec = np.exp(imvec)
        
        datterm  = alpha_d1 * chisq1grad(imvec) + alpha_d2 * chisq2grad(imvec)
        regterm  = alpha_s1 * reg1grad(imvec) + alpha_s2 * reg2grad(imvec)
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
            s_1 = reg1(im_step) 
            s_2 = reg2(im_step) 
            #fluxreg = flux_constraint(im_step) 
            #cmreg = cm_constraint(im_step) 
            if np.any(np.invert(embed_mask)): im_step = embed(im_step, embed_mask)
            plot_i(im_step, Prior, nit, chi2_1, chi2_2, ipynb=ipynb)
            print("i: %d chi2_1-1: %0.2f chi2_2-1: %0.2f s_1: %0.2f s_2: %0.2f" % (nit, chi2_1-1, chi2_2-1,s_1,s_2))
        nit += 1
   
    # Generate and the initial image
    if logim:
        xinit = np.log(ninit)
    else: 
        xinit = ninit       

    
    # Print stats
    print("Initial Chi^2_1: %f Chi^2_2: %f" % (chisq1(ninit), chisq2(ninit)))
    print("Total Pixel #: ",(len(Prior.imvec)))
    print("Clipped Pixel #: ",(len(ninit)))
    plotcur(xinit)
    
    # Minimize
    optdict = {'maxiter':maxit, 'ftol':stop, 'maxcor':NHIST} # minimizer params
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
 
    outim = image.Image(out.reshape(Prior.ydim, Prior.xdim), Prior.psize, 
                     Prior.ra, Prior.dec, rf=Prior.rf, source=Prior.source, 
                     mjd=Prior.mjd, pulse=Prior.pulse)
    
    if len(Prior.qvec):
        print("Preserving image complex polarization fractions!")
        qvec = Prior.qvec * out / Prior.imvec
        uvec = Prior.uvec * out / Prior.imvec
        outim.add_qu(qvec.reshape(Prior.ydim, Prior.xdim), uvec.reshape(Prior.ydim, Prior.xdim))
   
    # Print stats
    print("time: %f s" % (tstop - tstart))
    print("J: %f" % res.fun)
    print("Final Chi^2_1: %f Chi^2_2: %f" % (chisq1(out[embed_mask]), chisq2(out[embed_mask])))
    print(res.message)
    
    # Return Image object
    return outim
          
##################################################################################################
# Chi-squared and Gradient Functions
##################################################################################################

def chisq(imvec, A, data, sigma, dtype):
    """return the chi^2 for the appropriate dtype"""
    
    if dtype == 'vis':
        chisq = chisq_vis(imvec, A, data, sigma)
    
    elif dtype == 'amp':
        chisq = chisq_amp(imvec, A, data, sigma)

    elif dtype == 'bs':
        chisq = chisq_bs(imvec, A, data, sigma)

    elif dtype == 'cphase':
        chisq = chisq_cphase(imvec, A, data, sigma)

    elif dtype == 'camp':
        chisq = chisq_camp(imvec, A, data, sigma)

    else:
        chisq = 1.

    return chisq

def chisqgrad(imvec, A, data, sigma, dtype):
    """return the chi^2 gradient for the appropriate dtype"""
    
    if dtype == 'vis':
        chisqgrad = chisqgrad_vis(imvec, A, data, sigma)
    
    elif dtype == 'amp':
        chisqgrad = chisqgrad_amp(imvec, A, data, sigma)

    elif dtype == 'bs':
        chisqgrad = chisqgrad_bs(imvec, A, data, sigma)

    elif dtype == 'cphase':
        chisqgrad = chisqgrad_cphase(imvec, A, data, sigma)

    elif dtype == 'camp':
        chisqgrad = chisqgrad_camp(imvec, A, data, sigma)

    else:
        chisqgrad = np.zeros(len(imvec))

    return chisqgrad

# Visibility phase and amplitude chi-squared
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

#Visibility Amplitudes chi-squared
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

# Bispectrum chi-squared
def chisq_bs(imvec, Amatrices, bis, sigma):
    """Bispectrum chi-squared"""
    
    bisamples = np.dot(Amatrices[0], imvec) * np.dot(Amatrices[1], imvec) * np.dot(Amatrices[2], imvec)
    return np.sum(np.abs(((bis - bisamples)/sigma))**2)/(2.*len(bis))
    
def chisqgrad_bs(imvec, Amatrices, bis, sigma):
    """The gradient of the bispectrum chi-squared"""
    
    bisamples = np.dot(Amatrices[0], imvec) * np.dot(Amatrices[1], imvec) * np.dot(Amatrices[2], imvec)
    wdiff = ((bis - bisamples).conj())/(sigma**2)
    pt1 = wdiff * np.dot(Amatrices[1],imvec) * np.dot(Amatrices[2],imvec)
    pt2 = wdiff * np.dot(Amatrices[0],imvec) * np.dot(Amatrices[2],imvec)
    pt3 = wdiff * np.dot(Amatrices[0],imvec) * np.dot(Amatrices[1],imvec)
    out = -np.real(np.dot(pt1, Amatrices[0]) + np.dot(pt2, Amatrices[1]) + np.dot(pt3, Amatrices[2]))/len(bis)
    return out

#Closure phases chi-squared
def chisq_cphase(imvec, Amatrices, clphase, sigma):
    """Closure Phases (normalized) chi-squared"""
    clphase = clphase * DEGREE
    sigma = sigma * DEGREE
    clphase_samples = np.angle(np.dot(Amatrices[0], imvec) * np.dot(Amatrices[1], imvec) * np.dot(Amatrices[2], imvec))
    return (2.0/len(clphase)) * np.sum((1.0 - np.cos(clphase-clphase_samples))/(sigma**2))


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
 
#def chisqgrad_cphase_2(imvec, Amatrices, clphase, sigma):
#    clphase = clphase * DEGREE
#    sigma = sigma * DEGREE
#    
#    i1 = np.dot(Amatrices[0], imvec)
#    i2 = np.dot(Amatrices[1], imvec)
#    i3 = np.dot(Amatrices[2], imvec)
#    clphase_samples = np.angle(i1 * i2 * i3)
#    xx = np.imag(Amatrices[0]/(i1[:,None]) + Amatrices[1]/(i2[:,None]) + Amatrices[2]/(i3[:,None]))
#    cc = 1j*np.exp(1j*clphase_samples)*(np.exp(-1j*clphase)-np.exp(-1j*clphase_samples))/ (sigma**2)
#    out = -(2.0/len(clphase)) * np.real(np.dot(cc,xx))
#    return out 

#Closure Amplitudes chi-squared
def chisq_camp(imvec, Amatrices, clamp, sigma):
    """Closure Amplitudes (normalized) chi-squared"""
    
    clamp_samples = np.abs(np.dot(Amatrices[0], imvec) * np.dot(Amatrices[1], imvec) / (np.dot(Amatrices[2], imvec) * np.dot(Amatrices[3], imvec)))
    return np.sum(np.abs((clamp - clamp_samples)/sigma)**2)/len(clamp)

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


##################################################################################################
# Entropy and Gradient Functions
##################################################################################################

def regularizer(imvec, nprior, mask, flux, xdim, ydim, psize, stype):
    if stype == "simple":
        s = -ssimple(imvec, nprior, flux)
    elif stype == "l1":
        s = -sl1(imvec, nprior, flux)
    elif stype == "gs":
        s = -sgs(imvec, nprior, flux)
    elif stype == "patch":
        s = -spatch(imvec, nprior, flux)
    
    elif stype == "tv":
        if np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)
        s = -stv(imvec, xdim, ydim, flux)
    else:
        s = 0

    return s

def regularizergrad(imvec, nprior, mask, flux, xdim, ydim, psize, stype):

    if stype == "simple":
        s = -ssimplegrad(imvec, nprior, flux)
    elif stype == "l1":
        s = -sl1grad(imvec, nprior, flux)
    elif stype == "gs":
        s = -sgsgrad(imvec, nprior, flux)
    elif stype == "patch":
        s = -spatchgrad(imvec, nprior, flux)

    elif stype == "tv":
        if np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)
        s = -stvgrad(imvec, xdim, ydim, flux)[mask]
    else:
        s = np.zeros(len(imvec))

    return s


def ssimple(imvec, priorvec, flux):
    """Simple entropy
    """
    norm = flux
    norm = 1
    return -np.sum(imvec*np.log(imvec/priorvec))/norm

def ssimplegrad(imvec, priorvec, flux):
    """Simple entropy gradient
    """
    norm = flux
    norm =1
    return (-np.log(imvec/priorvec) - 1)/norm
    
def sl1(imvec, priorvec, flux):
    """L1 norm regularizer
    """
    norm = flux
    norm = 1    
    return -np.sum(np.abs(imvec - priorvec))/norm

def sl1grad(imvec, priorvec, flux):
    """L1 norm gradient
    """
    norm = flux
    norm = 1    
    return -np.sign(imvec - priorvec)/norm
    
def sgs(imvec, priorvec, flux):
    """Gull-skilling entropy
    """
    norm = flux
    norm =1    
    return np.sum(imvec - priorvec - imvec*np.log(imvec/priorvec))/norm


def sgsgrad(imvec, priorvec, flux):
    """Gull-Skilling gradient
    """  
    norm = flux
    norm = 1
    return -np.log(imvec/priorvec)/norm


def stv(imvec, nx, ny, flux):
    """Total variation regularizer
    """
    norm = flux
    norm = 1    
    im = imvec.reshape(ny, nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    out = -np.sum(np.sqrt(np.abs(im_l1 - im)**2 + np.abs(im_l2 - im)**2))
    return out/norm

def stvgrad(imvec, nx, ny, flux):
    """Total variation gradient
    """
    norm = flux
    norm = 1    
    im = imvec.reshape(ny,nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    im_r1 = np.roll(impad, 1, axis=0)[1:ny+1, 1:nx+1]
    im_r2 = np.roll(impad, 1, axis=1)[1:ny+1, 1:nx+1]
    im_r1l2 = np.roll(np.roll(impad, 1, axis=0), -1, axis=1)[1:ny+1, 1:nx+1]
    im_l1r2 = np.roll(np.roll(impad, 1, axis=0), -1, axis=1)[1:ny+1, 1:nx+1]
    
    g1 = (2*im - im_l1 - im_l2)/np.sqrt((im_l1 - im)**2 + (im_l2 - im)**2)
    g2 = (im - im_r1)/np.sqrt((im_r1 - im)**2 + (im_r1l2 - im_r1)**2)
    g3 = (im - im_r2)/np.sqrt((im_r2 - im)**2 + (im_l1r2 - im_r2)**2)
    out= -(g1 + g2 + g3).flatten()
    return out/norm
    
def spatch(imvec, priorvec, flux):
    """Patch prior regularizer
    """
    norm = flux**2
    norm = 1    
    out = -0.5*np.sum( ( imvec - priorvec) ** 2)
    return out/norm

def spatchgrad(imvec, priorvec, flux):
    """Patch prior gradient
    """
    norm = flux**2
    norm = 1    
    out = -(imvec  - priorvec)
    return out/norm
   
##################################################################################################
# Misc Functions
##################################################################################################
def embed(im, mask, clipfloor=0., randomfloor=False):
    """Embeds a 1d image array into the size of boolean embed mask
    """
    j=0
    out=np.zeros(len(mask))
    for i in range(len(mask)):
        if mask[i]:
            out[i] = im[j]
            j += 1
        else:
            # prevent total variation gradient singularities
            if randomfloor: out[i] = clipfloor * np.random.normal()
            else: out[i] = clipfloor

    return out

def chisqdata(Obsdata, Prior, mask, dtype):
    """Return the data, sigma, and matrices for the appropriate dtype
    """
    
    if dtype == 'vis':
        (data, sigma, A) = chisqdata_vis(Obsdata, Prior, mask)
    
    elif dtype == 'amp':
        (data, sigma, A) = chisqdata_amp(Obsdata, Prior, mask)

    elif dtype == 'bs':
        (data, sigma, A) = chisqdata_bs(Obsdata, Prior, mask)

    elif dtype == 'cphase':
        (data, sigma, A) = chisqdata_cphase(Obsdata, Prior, mask)

    elif dtype == 'camp':
        (data, sigma, A) = chisqdata_camp(Obsdata, Prior, mask)

    else:
        (data, sigma, A) = (False, False, False)

    return (data, sigma, A)

def chisqdata_vis(Obsdata, Prior, mask):
    """Return the visibilities, sigmas, and fourier matrix for and observation, prior, mask
    """
    
    data_arr = Obsdata.unpack(['u','v','vis','sigma'])
    uv = np.hstack((data_arr['u'].reshape(-1,1), data_arr['v'].reshape(-1,1)))    
    vis = data_arr['vis']
    sigma = data_arr['sigma']
    A = ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv, pulse=Prior.pulse, mask=mask)
    
    return (vis, sigma, A)

def chisqdata_amp(Obsdata, Prior, mask):
    """Return the amplitudes, sigmas, and fourier matrix for and observation, prior, mask
    """

    ampdata = Obsdata.unpack(['u','v','amp','sigma'])
    uv = np.hstack((ampdata['u'].reshape(-1,1), ampdata['v'].reshape(-1,1)))
    amp = ampdata['amp']
    sigma = ampdata['sigma']
    A = ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv, pulse=Prior.pulse, mask=mask)
    
    # Debias the amplitudes
    amp = amp_debias(amp, sigma)
    
    return (amp, sigma, A)

def chisqdata_bs(Obsdata, Prior, mask):
    """return the bispectra, sigmas, and fourier matrices for and observation, prior, mask
    """

    biarr = Obsdata.bispectra(mode="all", count="min")
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

def chisqdata_cphase(Obsdata, Prior, mask):
    """Return the closure phases, sigmas, and fourier matrices for and observation, prior, mask
    """

    clphasearr = Obsdata.c_phases(mode="all", count="min")
    uv1 = np.hstack((clphasearr['u1'].reshape(-1,1), clphasearr['v1'].reshape(-1,1)))
    uv2 = np.hstack((clphasearr['u2'].reshape(-1,1), clphasearr['v2'].reshape(-1,1)))
    uv3 = np.hstack((clphasearr['u3'].reshape(-1,1), clphasearr['v3'].reshape(-1,1)))
    clphase = clphasearr['cphase']
    sigma = clphasearr['sigmacp']
    
    A3 = (ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse, mask=mask)
         )
    return (clphase, sigma, A3)

def chisqdata_camp(Obsdata, Prior, mask):
    """Return the closure amplitudes, sigmas, and fourier matrices for and observation, prior, mask
    """

    clamparr = Obsdata.c_amplitudes(mode="all", count="min")
    uv1 = np.hstack((clamparr['u1'].reshape(-1,1), clamparr['v1'].reshape(-1,1)))
    uv2 = np.hstack((clamparr['u2'].reshape(-1,1), clamparr['v2'].reshape(-1,1)))
    uv3 = np.hstack((clamparr['u3'].reshape(-1,1), clamparr['v3'].reshape(-1,1)))
    uv4 = np.hstack((clamparr['u4'].reshape(-1,1), clamparr['v4'].reshape(-1,1)))
    clamp = clamparr['camp']
    sigma = clamparr['sigmaca']

    A4 = (ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv4, pulse=Prior.pulse, mask=mask)
         )

    return (clamp, sigma, A4)

def plot_i(im, Prior, nit, chi2_1, chi2_2, ipynb=False):
    """Plot the total intensity image at each iteration
    """

    plt.ion()
    plt.pause(0.00001)    
    plt.clf()
    
    plt.imshow(im.reshape(Prior.ydim,Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian')     
    xticks = ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
    yticks = ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title("step: %i  $\chi^2_1$: %f  $\chi^2_2$: %f" % (nit, chi2_1, chi2_2), fontsize=20)
    #plt.draw()

    if ipynb:
        display.clear_output()
        display.display(plt.gcf())

##################################################################################################
# Restoring Functions
##################################################################################################
def threshold(image, frac_i=1.e-5, frac_pol=1.e-3):
    """Apply a hard threshold
    """ 

    imvec = np.copy(image.imvec)

    thresh = frac_i*np.abs(np.max(imvec))
    lowval = thresh
    flux = np.sum(imvec)

    for j in range(len(imvec)):
        if imvec[j] < thresh: 
            imvec[j]=lowval

    imvec = flux*imvec/np.sum(imvec)
    out = image.Image(imvec.reshape(image.ydim,image.xdim), image.psize, 
                   image.ra, image.dec, rf=image.rf, source=image.source, mjd=image.mjd)
    return out

def blur_circ(image, fwhm_i, fwhm_pol=0):
    """Apply a circular gaussian filter to the I image.
       fwhm is in radians
    """ 
    
    # Blur Stokes I
    sigma = fwhm_i/(2. * np.sqrt(2. * np.log(2.)))
    sigmap = sigma/image.psize
    im = filt.gaussian_filter(image.imvec.reshape(image.ydim, image.xdim), (sigmap, sigmap))
    out = image.Image(im, image.psize, image.ra, image.dec, rf=image.rf, source=image.source, mjd=image.mjd)
   
    # Blur Stokes Q and U
    if len(image.qvec) and fwhm_pol:
        sigma = fwhm_pol/(2. * np.sqrt(2. * np.log(2.)))
        sigmap = sigma/image.psize
        imq = filt.gaussian_filter(image.qvec.reshape(image.ydim,image.xdim), (sigmap, sigmap))
        imu = filt.gaussian_filter(image.uvec.reshape(image.ydim,image.xdim), (sigmap, sigmap))
        out.add_qu(imq, imu)
        
    return out
          
