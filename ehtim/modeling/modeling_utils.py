# modeling_utils.py
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
from  scipy.special import jv

import ehtim.image as image
#from . import linearize_energy as le

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *
from ehtim.statistics.dataframes import *

from IPython import display


##################################################################################################
# Constants & Definitions
##################################################################################################

MAXLS = 100 # maximum number of line searches in L-BFGS-B
NHIST = 100 # number of steps to store for hessian approx
MAXIT = 100 # maximum number of iterations
STOP = 1.e-8 # convergence criterion

DATATERMS = ['vis', 'bs', 'amp', 'cphase', 'cphase_diag', 'camp', 'logcamp', 'logcamp_diag', 'logamp']

nit = 0 # global variable to track the iteration number in the plotting callback

##################################################################################################
# Priors
##################################################################################################


##################################################################################################
# Modeler
##################################################################################################
def modeler_func(Obsdata, model, model_prior,
                   d1='vis', d2=False, d3=False,
                   alpha_d1=100, alpha_d2=100, alpha_d3=100,
                   **kwargs):

    """Fit a specified model. 

       Args:
           Obsdata (Obsdata): The Obsdata object with VLBI data
           model (Model): The Model object to fit
           model_prior (dict): Priors for each model parameter

           d1 (str): The first data term; options are 'vis', 'bs', 'amp', 'cphase', 'cphase_diag' 'camp', 'logcamp', 'logcamp_diag'
           d2 (str): The second data term; options are 'vis', 'bs', 'amp', 'cphase', 'cphase_diag' 'camp', 'logcamp', 'logcamp_diag'
           d3 (str): The third data term; options are 'vis', 'bs', 'amp', 'cphase', 'cphase_diag' 'camp', 'logcamp', 'logcamp_diag'

           alpha_d1 (float): The first data term weighting
           alpha_d2 (float): The second data term weighting
           alpha_d2 (float): The third data term weighting

           maxit (int): Maximum number of minimizer iterations
           stop (float): The convergence criterion

           show_updates (bool): If True, displays the progress of the minimizer

           debias (bool): if True then apply debiasing to amplitudes/closure amplitudes
           systematic_noise (float): a fractional systematic noise tolerance to add to thermal sigmas
           snrcut (float): a  snr cutoff for including data in the chi^2 sum

           maxset (bool):  if True, use maximal set instead of minimal for closure quantities
           systematic_cphase_noise (float): a value in degrees to add to the closure phase sigmas
           cp_uv_min (float): flag baselines shorter than this before forming closure quantities

       Returns:
           Model: Model object with result
    """

    # some kwarg default values
    maxit = kwargs.get('maxit', MAXIT)
    stop = kwargs.get('stop', STOP)

    show_updates = kwargs.get('show_updates',True)

    # Make sure data and regularizer options are ok
    if not d1 and not d2:
        raise Exception("Must have at least one data term!")
    if (not ((d1 in DATATERMS) or d1==False)) or (not ((d2 in DATATERMS) or d2==False)):
        raise Exception("Invalid data term: valid data terms are: " + ' '.join(DATATERMS))

    # Create the trial model
    trial_model = model.copy()

    # Define the mapping between solved parameters and the model
    # Each fitted model parameter is rescaled to give values closer to order unity
    param_map = []
    param_mask = []
    for j in range(model.N_models()):
        if model.models[j] == 'ring':
            param_map.append([j,'F0',1,'Jy'])
            param_mask.append(True)
            param_map.append([j,'d',RADPERUAS,'uas'])
            param_mask.append(True)
            param_map.append([j,'x0',RADPERUAS,'uas'])
            param_mask.append(True)
            param_map.append([j,'y0',RADPERUAS,'uas'])
            param_mask.append(True)

    # Get data and info for the data terms
    (data1, sigma1, uv1) = chisqdata(Obsdata, d1)
    (data2, sigma2, uv2) = chisqdata(Obsdata, d2)
    (data3, sigma3, uv3) = chisqdata(Obsdata, d3)

    # Define the chi^2 and chi^2 gradient
    def chisq1():
        return chisq(trial_model, uv1, data1, sigma1, d1)

    def chisq1grad():
        c = chisqgrad(trial_model, uv1, data1, sigma1, d1, param_mask)
        return c

    def chisq2():
        return chisq(trial_model, uv2, data2, sigma2, d2)

    def chisq2grad():
        c = chisqgrad(trial_model, uv2, data2, sigma2, d2, param_mask)
        return c

    def chisq3():
        return chisq(trial_model, uv3, data3, sigma3, d3)

    def chisq3grad():
        c = chisqgrad(trial_model, uv3, data3, sigma3, d3, param_mask)
        return c

    def set_params(params):
        for j in range(len(params)):
            trial_model.params[param_map[j][0]][param_map[j][1]] = params[j] * param_map[j][2]

    # Define prior
    def prior():
        return np.log(1.0) #prior(params, model_prior)

    def prior_grad():
        return np.log(1.0) #prior(params, model_prior)

    # Define the objective function and gradient
    def objfunc(params):
        set_params(params)
        datterm  = alpha_d1 * (chisq1() - 1) + alpha_d2 * (chisq2() - 1) + alpha_d3 * (chisq3() - 1)
        priterm  = prior()

        return datterm + priterm

    def objgrad(params):
        set_params(params)
        datterm  = alpha_d1 * chisq1grad() + alpha_d2 * chisq2grad() + alpha_d3 * chisq3grad()
        priterm  = prior_grad()

        grad = datterm + priterm

        for j in range(len(params)):
            grad[j] *= param_map[j][2]

        return grad

    # Define plotting function for each iteration
    global nit
    nit = 0
    def plotcur(params_step):
        global nit
        if show_updates:
            print('Params:',params_step)
            chi2_1 = chisq1()
            chi2_2 = chisq2()
            chi2_3 = chisq3()
            #plot_i(im_step, Prior, nit, {d1:chi2_1, d2:chi2_2, d3:chi2_3}, pol=pol)   # could sample model and plot it
            print("i: %d chi2_1: %0.2f chi2_2: %0.2f chi2_3: %0.2f" % (nit, chi2_1, chi2_2, chi2_3))
        nit += 1

    # Initial parameters
    param_init = np.array([model.params[pm[0]][pm[1]]/pm[2] for pm in param_map])

    # Print stats
    print("Initial Chi^2_1: %f Chi^2_2: %f Chi^2_3: %f" % (chisq1(), chisq2(), chisq3()))
    print("Initial Objective Function: %f" % (objfunc(param_init)))

    if d1 in DATATERMS:
        print("Total Data 1: ", (len(data1)))
    if d2 in DATATERMS:
        print("Total Data 2: ", (len(data2)))
    if d3 in DATATERMS:
        print("Total Data 3: ", (len(data3)))

    print("Total Fitted Real Parameters #: ",(len(param_init)))
    plotcur(param_init)

    # Minimize
    optdict = {'maxiter':maxit, 'ftol':stop, 'maxcor':NHIST,'gtol':stop,'maxls':MAXLS} # minimizer dict params
    tstart = time.time()
    res = opt.minimize(objfunc, param_init, method='L-BFGS-B', jac=objgrad,
                       options=optdict, callback=plotcur)

    tstop = time.time()

    # Format output
    out = res.x
    set_params(out)

    print("\nFitted Parameters:")
    cur_idx = -1
    for j in range(len(param_map)):
        if param_map[j][0] != cur_idx:
            cur_idx = param_map[j][0]
            print(model.models[cur_idx] + ' (component %d/%d):' % (cur_idx+1,model.N_models()))
        print(('\t' + param_map[j][1] + ': %f ' + param_map[j][3]) %  out[j])
    print('\n')

    # Print stats
    print("time: %f s" % (tstop - tstart))
    print("J: %f" % res.fun)
    print(res.message)

    # Return fitted model
    return trial_model

##################################################################################################
# Wrapper Functions
##################################################################################################

def chisq(model, uv, data, sigma, dtype):
    """return the chi^2 for the appropriate dtype
    """

    chisq = 1
    if not dtype in DATATERMS:
        return chisq

    if dtype == 'vis':
        chisq = chisq_vis(model, uv, data, sigma)
    elif dtype == 'amp':
        chisq = chisq_amp(model, uv, data, sigma)
    elif dtype == 'logamp':
        chisq = chisq_logamp(model, uv, data, sigma)
    elif dtype == 'bs':
        chisq = chisq_bs(model, uv, data, sigma)
    elif dtype == 'cphase':
        chisq = chisq_cphase(model, uv, data, sigma)
    elif dtype == 'cphase_diag':
        chisq = chisq_cphase_diag(model, uv, data, sigma)
    elif dtype == 'camp':
        chisq = chisq_camp(model, uv, data, sigma)
    elif dtype == 'logcamp':
        chisq = chisq_logcamp(model, uv, data, sigma)
    elif dtype == 'logcamp_diag':
        chisq = chisq_logcamp_diag(model, uv, data, sigma)

    return chisq

def chisqgrad(model, uv, data, sigma, dtype, param_mask):
    """return the chi^2 gradient for the appropriate dtype
    """

    chisqgrad = np.sum(param_mask)
    if not dtype in DATATERMS:
        return chisqgrad

    if dtype == 'vis':
        chisqgrad = chisqgrad_vis(model, uv, data, sigma)
    elif dtype == 'amp':
        chisqgrad = chisqgrad_amp(model, uv, data, sigma)
    elif dtype == 'logamp':
        chisqgrad = chisqgrad_logamp(model, uv, data, sigma)
    elif dtype == 'bs':
        chisqgrad = chisqgrad_bs(model, uv, data, sigma)
    elif dtype == 'cphase':
        chisqgrad = chisqgrad_cphase(model, uv, data, sigma)
#    elif dtype == 'cphase_diag':
#        chisqgrad = chisqgrad_cphase_diag(imvec, A, data, sigma)
    elif dtype == 'camp':
        chisqgrad = chisqgrad_camp(model, uv, data, sigma)
    elif dtype == 'logcamp':
        chisqgrad = chisqgrad_logcamp(model, uv, data, sigma)
#    elif dtype == 'logcamp_diag':
#        chisqgrad = chisqgrad_logcamp_diag(imvec, A, data, sigma)

    return chisqgrad[param_mask]

def chisqdata(Obsdata, dtype, pol='I', **kwargs):

    """Return the data, sigma, and matrices for the appropriate dtype
    """

    (data, sigma, uv) = (False, False, False)

    if dtype == 'vis':
        (data, sigma, uv) = chisqdata_vis(Obsdata, pol=pol, **kwargs)
    elif dtype == 'amp' or dtype == 'logamp':
        (data, sigma, uv) = chisqdata_amp(Obsdata, pol=pol,**kwargs)
    elif dtype == 'bs':
        (data, sigma, uv) = chisqdata_bs(Obsdata, pol=pol,**kwargs)
    elif dtype == 'cphase':
        (data, sigma, uv) = chisqdata_cphase(Obsdata, pol=pol,**kwargs)
#    elif dtype == 'cphase_diag':
#        (data, sigma, uv) = chisqdata_cphase_diag(Obsdata, Prior, mask, pol=pol,**kwargs)
    elif dtype == 'camp':
        (data, sigma, uv) = chisqdata_camp(Obsdata, pol=pol,**kwargs)
    elif dtype == 'logcamp':
        (data, sigma, uv) = chisqdata_logcamp(Obsdata, pol=pol,**kwargs)
#    elif dtype == 'logcamp_diag':
#        (data, sigma, A) = chisqdata_logcamp_diag(Obsdata, Prior, mask, pol=pol,**kwargs)

    return (data, sigma, uv)


##################################################################################################
# Chi-squared and Gradient Functions
##################################################################################################

def chisq_vis(model, uv, vis, sigma):
    """Visibility chi-squared"""

    samples = model.sample_uv(uv[:,0],uv[:,1])
    return np.sum(np.abs((samples-vis)/sigma)**2)/(2*len(vis))

def chisqgrad_vis(model, uv, vis, sigma):
    """The gradient of the visibility chi-squared"""

    samples = model.sample_uv(uv[:,0],uv[:,1])
    wdiff   = (vis - samples)/(sigma**2)
    grad    = model.sample_grad_uv(uv[:,0],uv[:,1])

    out = -np.real(np.dot(grad.conj(), wdiff))/len(vis)
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

def chisq_cphase_diag(imvec, Amatrices, clphase_diag, sigma):
    """Diagonalized closure phases (normalized) chi-squared"""
    clphase_diag = np.concatenate(clphase_diag) * DEGREE
    sigma = np.concatenate(sigma) * DEGREE

    A3_diag = Amatrices[0]
    tform_mats = Amatrices[1]

    clphase_diag_samples = []
    for iA, A3 in enumerate(A3_diag):
        clphase_samples = np.angle(np.dot(A3[0], imvec) * np.dot(A3[1], imvec) * np.dot(A3[2], imvec))
        clphase_diag_samples.append(np.dot(tform_mats[iA],clphase_samples))
    clphase_diag_samples = np.concatenate(clphase_diag_samples)

    chisq = (2.0/len(clphase_diag)) * np.sum((1.0 - np.cos(clphase_diag-clphase_diag_samples))/(sigma**2))
    return chisq

def chisqgrad_cphase_diag(imvec, Amatrices, clphase_diag, sigma):
    """The gradient of the diagonalized closure phase chi-squared"""
    clphase_diag = clphase_diag * DEGREE
    sigma = sigma * DEGREE

    A3_diag = Amatrices[0]
    tform_mats = Amatrices[1]

    deriv = np.zeros_like(imvec)
    for iA, A3 in enumerate(A3_diag):

        i1 = np.dot(A3[0], imvec)
        i2 = np.dot(A3[1], imvec)
        i3 = np.dot(A3[2], imvec)
        clphase_samples = np.angle(i1 * i2 * i3)
        clphase_diag_samples = np.dot(tform_mats[iA],clphase_samples)

        clphase_diag_measured = clphase_diag[iA]
        clphase_diag_sigma = sigma[iA]

        term1 = np.dot(np.dot((np.sin(clphase_diag_measured-clphase_diag_samples)/(clphase_diag_sigma**2.0)),(tform_mats[iA]/i1)),A3[0])
        term2 = np.dot(np.dot((np.sin(clphase_diag_measured-clphase_diag_samples)/(clphase_diag_sigma**2.0)),(tform_mats[iA]/i2)),A3[1])
        term3 = np.dot(np.dot((np.sin(clphase_diag_measured-clphase_diag_samples)/(clphase_diag_sigma**2.0)),(tform_mats[iA]/i3)),A3[2])
        deriv += -2.0*np.imag(term1 + term2 + term3)

    deriv *= 1.0/np.float(len(np.concatenate(clphase_diag)))

    return deriv

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

def chisq_logcamp_diag(imvec, Amatrices, log_clamp_diag, sigma):
    """Diagonalized log closure amplitudes (normalized) chi-squared"""

    log_clamp_diag = np.concatenate(log_clamp_diag)
    sigma = np.concatenate(sigma)

    A4_diag = Amatrices[0]
    tform_mats = Amatrices[1]

    log_clamp_diag_samples = []
    for iA, A4 in enumerate(A4_diag):

        a1 = np.abs(np.dot(A4[0], imvec))
        a2 = np.abs(np.dot(A4[1], imvec))
        a3 = np.abs(np.dot(A4[2], imvec))
        a4 = np.abs(np.dot(A4[3], imvec))

        log_clamp_samples = np.log(a1) + np.log(a2) - np.log(a3) - np.log(a4)
        log_clamp_diag_samples.append(np.dot(tform_mats[iA],log_clamp_samples))

    log_clamp_diag_samples = np.concatenate(log_clamp_diag_samples)

    chisq = np.sum(np.abs((log_clamp_diag - log_clamp_diag_samples)/sigma)**2) / (len(log_clamp_diag))
    return  chisq

def chisqgrad_logcamp_diag(imvec, Amatrices, log_clamp_diag, sigma):
    """The gradient of the diagonalized log closure amplitude chi-squared"""

    A4_diag = Amatrices[0]
    tform_mats = Amatrices[1]

    deriv = np.zeros_like(imvec)
    for iA, A4 in enumerate(A4_diag):

        i1 = np.dot(A4[0], imvec)
        i2 = np.dot(A4[1], imvec)
        i3 = np.dot(A4[2], imvec)
        i4 = np.dot(A4[3], imvec)
        log_clamp_samples = np.log(np.abs(i1)) + np.log(np.abs(i2)) - np.log(np.abs(i3)) - np.log(np.abs(i4))
        log_clamp_diag_samples = np.dot(tform_mats[iA],log_clamp_samples)

        log_clamp_diag_measured = log_clamp_diag[iA]
        log_clamp_diag_sigma = sigma[iA]

        term1 = np.dot(np.dot(((log_clamp_diag_measured-log_clamp_diag_samples)/(log_clamp_diag_sigma**2.0)),(tform_mats[iA]/i1)),A4[0])
        term2 = np.dot(np.dot(((log_clamp_diag_measured-log_clamp_diag_samples)/(log_clamp_diag_sigma**2.0)),(tform_mats[iA]/i2)),A4[1])
        term3 = np.dot(np.dot(((log_clamp_diag_measured-log_clamp_diag_samples)/(log_clamp_diag_sigma**2.0)),(tform_mats[iA]/i3)),A4[2])
        term4 = np.dot(np.dot(((log_clamp_diag_measured-log_clamp_diag_samples)/(log_clamp_diag_sigma**2.0)),(tform_mats[iA]/i4)),A4[3])
        deriv += -2.0*np.real(term1 + term2 - term3 - term4)

    deriv *= 1.0/np.float(len(np.concatenate(log_clamp_diag)))

    return deriv

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

def chisqdata_vis(Obsdata, pol='I', **kwargs):
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

    return (vis, sigma, uv)

def chisqdata_amp(Obsdata, pol='I',**kwargs):
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

    return (amp, sigma, uv)

def chisqdata_bs(Obsdata, pol='I',**kwargs):
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

    return (bi, sigma, (uv1, uv2, uv3))

def chisqdata_cphase(Obsdata, pol='I',**kwargs):
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

    return (clphase, sigma, (uv1, uv2, uv3))

def chisqdata_camp(Obsdata, pol='I',**kwargs):
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

    return (clamp, sigma, (uv1, uv2, uv3, uv4))

def chisqdata_logcamp(Obsdata, pol='I', **kwargs):
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

    return (clamp, sigma, (uv1, uv2, uv3, uv4))
