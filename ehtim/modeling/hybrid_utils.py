# hybrid_utils.py
# Hybrid imaging/modeling functions for total intensity VLBI data

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
import ehtim.model as model

import ehtim.imaging.imager_utils as iu
import ehtim.modeling.modeling_utils as mu

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

nit = 0 # global variable to track the iteration number in the plotting callback  

##################################################################################################
# Hybrid Imager/Modeler
##################################################################################################
def hybrid_func(Obsdata, model_init, model_prior, InitIm, PriorIm, im_flux,
                   d1='vis', d2=False, d3=False,
                   alpha_d1=100, alpha_d2=100, alpha_d3=100,
                   s1='simple', s2=False, s3=False,
                   alpha_s1=1, alpha_s2=1, alpha_s3=1,
                   alpha_flux=500, alpha_cm=500, 
                   minimizer_method='L-BFGS-B', test_gradient=False, ttype='nfft', **kwargs):

    """Fit a hybrid model+image to a specified observation. 

       Args:
           Obsdata (Obsdata): The Obsdata object with VLBI data

           model_init (Model): The Model object to fit, which also defines initial model parameters
           model_prior (dict): Priors for each model parameter

           InitIm (Image): The Image object with the initial image for the minimization
           Prior (Image): The Image object with the prior image
           im_flux (float): The total flux of the output image in Jy

           d1 (str): The first data term; options are 'vis', 'bs', 'amp', 'cphase', 'cphase_diag' 'camp', 'logcamp', 'logcamp_diag'
           d2 (str): The second data term; options are 'vis', 'bs', 'amp', 'cphase', 'cphase_diag' 'camp', 'logcamp', 'logcamp_diag'
           d3 (str): The third data term; options are 'vis', 'bs', 'amp', 'cphase', 'cphase_diag' 'camp', 'logcamp', 'logcamp_diag'

           s1 (str): The first regularizer; options are 'simple', 'gs', 'tv', 'tv2', 'l1', 'patch','compact','compact2','rgauss'
           s2 (str): The second regularizer; options are 'simple', 'gs', 'tv', 'tv2','l1', 'patch','compact','compact2','rgauss'
           s3 (str): The third regularizer; options are 'simple', 'gs', 'tv', 'tv2','l1', 'patch','compact','compact2','rgauss'

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
    clipfloor = kwargs.get('clipfloor', 0)

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
    if (not ((d1 in iu.DATATERMS) or d1==False)) or (not ((d2 in iu.DATATERMS) or d2==False)):
        raise Exception("Invalid data term: valid data terms are: " + ' '.join(iu.DATATERMS))
    if (not ((s1 in iu.REGULARIZERS) or s1==False)) or (not ((s2 in iu.REGULARIZERS) or s2==False)):
        raise Exception("Invalid regularizer: valid regularizers are: " + ' '.join(iu.REGULARIZERS))
    if (PriorIm.psize != InitIm.psize) or (PriorIm.xdim != InitIm.xdim) or (PriorIm.ydim != InitIm.ydim):
        raise Exception("Initial image does not match dimensions of the prior image!")
    if (InitIm.polrep != PriorIm.polrep):
        raise Exception("Initial image pol. representation does not match pol. representation of the prior image!")
    if (logim and PriorIm.pol_prim in ['Q','U','V']):
        raise Exception("Cannot image Stokes Q,U,or V with log image transformation! Set logim=False in imager_func")

    pol = PriorIm.pol_prim
    print("Generating %s image..." % pol)

    # Create the trial model
    trial_model = model_init.copy()

    # Define the mapping between solved parameters and the model
    # Each fitted model parameter is rescaled to give values closer to order unity
    param_map = []  # Define mapping for every fitted parameter: model component #, parameter name, rescale multiplier, unit
    param_mask = [] # True or False for whether to fit each model parameter    
    for j in range(model_init.N_models()):
        params = model.model_params(model_init.models[j],model_init.params[j])
        for param in params:
            if model_prior[j][param]['prior_type'] != 'fixed':
                param_mask.append(True)
                try:
                    param_map.append([j,param,mu.PARAM_DETAILS[param][0],mu.PARAM_DETAILS[param][1]])
                except:
                    param_map.append([j,param,1,''])
                    pass
            else:
                param_mask.append(False)

    # Define embedding mask
    embed_mask = PriorIm.imvec > clipfloor

    # Normalize prior image to total flux and limit imager range to prior values > clipfloor
    if (not norm_init):
        nprior = PriorIm.imvec[embed_mask]
        ninit = InitIm.imvec[embed_mask]
    else:
        nprior = (im_flux * PriorIm.imvec / np.sum((PriorIm.imvec)[embed_mask]))[embed_mask]
        ninit = (im_flux * InitIm.imvec / np.sum((InitIm.imvec)[embed_mask]))[embed_mask]

    if len(nprior)==0:
        raise Exception("clipfloor too large: all prior pixels have been clipped!")

    # Model: Get data and info for the data terms
    (data1, sigma1, uv1) = mu.chisqdata(Obsdata, d1)
    (data2, sigma2, uv2) = mu.chisqdata(Obsdata, d2)
    (data3, sigma3, uv3) = mu.chisqdata(Obsdata, d3)

    # Image: Get data and fourier matrices for the data terms
    (data1, sigma1, A1) = iu.chisqdata(Obsdata, PriorIm, embed_mask, d1, pol=pol, ttype=ttype, **kwargs)
    (data2, sigma2, A2) = iu.chisqdata(Obsdata, PriorIm, embed_mask, d2, pol=pol, ttype=ttype, **kwargs)
    (data3, sigma3, A3) = iu.chisqdata(Obsdata, PriorIm, embed_mask, d3, pol=pol, ttype=ttype, **kwargs)

    # Define the chi^2 and chi^2 gradient
    def chisq1(imvec):
        return chisq(trial_model, imvec, uv1, A1, data1, sigma1, d1, ttype=ttype, im_mask=embed_mask)

    def chisq1grad(imvec):
        c = chisqgrad(trial_model, imvec, uv1, A1, data1, sigma1, d1, param_mask, ttype=ttype, im_mask=embed_mask)
        return c

    def chisq2(imvec):
        return chisq(trial_model, imvec, uv2, A2, data2, sigma2, d2, ttype=ttype, im_mask=embed_mask)

    def chisq2grad(imvec):
        c = chisqgrad(trial_model, imvec, uv2, A2, data2, sigma2, d2, param_mask, ttype=ttype, im_mask=embed_mask)
        return c

    def chisq3(imvec):
        return chisq(trial_model, imvec, uv3, A3, data3, sigma3, d3, ttype=ttype, im_mask=embed_mask)

    def chisq3grad(imvec):
        c = chisqgrad(trial_model, imvec, uv3, A3, data3, sigma3, d3, param_mask, ttype=ttype, im_mask=embed_mask)
        return c

    # Define the regularizer and regularizer gradient
    def reg1(imvec):
        return iu.regularizer(imvec, nprior, embed_mask, im_flux, PriorIm.xdim, PriorIm.ydim, PriorIm.psize, s1, **kwargs)

    def reg1grad(imvec):
        return iu.regularizergrad(imvec, nprior, embed_mask, im_flux, PriorIm.xdim, PriorIm.ydim, PriorIm.psize, s1, **kwargs)

    def reg2(imvec):
        return iu.regularizer(imvec, nprior, embed_mask, im_flux, PriorIm.xdim, PriorIm.ydim, PriorIm.psize, s2, **kwargs)

    def reg2grad(imvec):
        return iu.regularizergrad(imvec, nprior, embed_mask, im_flux, PriorIm.xdim, PriorIm.ydim, PriorIm.psize, s2, **kwargs)

    def reg3(imvec):
        return iu.regularizer(imvec, nprior, embed_mask, im_flux, PriorIm.xdim, PriorIm.ydim, PriorIm.psize, s3, **kwargs)

    def reg3grad(imvec):
        return iu.regularizergrad(imvec, nprior, embed_mask, im_flux, PriorIm.xdim, PriorIm.ydim, PriorIm.psize, s3, **kwargs)

    # Define constraint functions
    def flux_constraint(imvec):
        return iu.regularizer(imvec, nprior, embed_mask, im_flux, PriorIm.xdim, PriorIm.ydim, PriorIm.psize, "flux", **kwargs)

    def flux_constraint_grad(imvec):
        return iu.regularizergrad(imvec, nprior, embed_mask, im_flux, PriorIm.xdim, PriorIm.ydim, PriorIm.psize, "flux", **kwargs)

    def cm_constraint(imvec):
        return iu.regularizer(imvec, nprior, embed_mask, im_flux, PriorIm.xdim, PriorIm.ydim, PriorIm.psize, "cm", **kwargs)

    def cm_constraint_grad(imvec):
        return iu.regularizergrad(imvec, nprior, embed_mask, im_flux, PriorIm.xdim, PriorIm.ydim, PriorIm.psize, "cm", **kwargs)

    def transform_params(params, inverse=True):
        return [mu.transform_param(params[j], model_prior[param_map[j][0]][param_map[j][1]], inverse=inverse) for j in range(len(params))]

    def set_params(params):
        tparams = transform_params(params)
        for j in range(len(params)):
            if param_map[j][1] in trial_model.params[param_map[j][0]].keys():
                trial_model.params[param_map[j][0]][param_map[j][1]] = tparams[j] * param_map[j][2]
            else: # In this case, the parameter is a list of complex numbers, so the real and imaginary parts need to be assigned
                param_type = 'beta_list' 
                idx = int(param_map[j][1][4:-3]) - 1
                curval = trial_model.params[param_map[j][0]][param_type][idx]
                if param_map[j][1][-2:] == 're':
                    trial_model.params[param_map[j][0]][param_type][idx] = tparams[j] * param_map[j][2]      + np.imag(curval)*1j
                elif param_map[j][1][-2:] == 'im':
                    trial_model.params[param_map[j][0]][param_type][idx] = tparams[j] * param_map[j][2] * 1j + np.real(curval)
                else:
                    print('Parameter ' + param_map[j][1] + ' not understood!')

    # Define model prior
    def prior(params):
        tparams = transform_params(params)
        return np.sum([np.log(mu.prior_func(tparams[j]*param_map[j][2], model_prior[param_map[j][0]][param_map[j][1]])) for j in range(len(params))])

    def prior_grad(params):
        tparams = transform_params(params)        
        f  = np.array([mu.prior_func(tparams[j]*param_map[j][2], model_prior[param_map[j][0]][param_map[j][1]]) for j in range(len(params))])
        df = np.array([mu.prior_grad_func(tparams[j]*param_map[j][2], model_prior[param_map[j][0]][param_map[j][1]]) for j in range(len(params))])
        return df/f

    # Define the objective function and gradient
    def objfunc(params):
        model_params = params[:len(mod_init)]
        imvec = params[len(mod_init):]

        if logim: imvec = np.exp(imvec)

        set_params(model_params)
        datterm  = alpha_d1 * (chisq1(imvec) - 1) + alpha_d2 * (chisq2(imvec) - 1) + alpha_d3 * (chisq3(imvec) - 1)
        priterm  = prior(model_params)
        regterm  = alpha_s1 * reg1(imvec) + alpha_s2 * reg2(imvec) + alpha_s3 * reg3(imvec)
        conterm  = alpha_flux * flux_constraint(imvec)  + alpha_cm * cm_constraint(imvec)

        return datterm + regterm + conterm - priterm

    def objgrad(params):
        model_params = params[:len(mod_init)]
        imvec = params[len(mod_init):]

        if logim: imvec = np.exp(imvec)

        set_params(model_params)
        datterm  = alpha_d1 * chisq1grad(imvec) + alpha_d2 * chisq2grad(imvec) + alpha_d3 * chisq3grad(imvec)
        priterm  = prior_grad(model_params)
        regterm  = alpha_s1 * reg1grad(imvec) + alpha_s2 * reg2grad(imvec) + alpha_s3 * reg3grad(imvec)
        conterm  = alpha_flux * flux_constraint_grad(imvec)  + alpha_cm * cm_constraint_grad(imvec)

        grad = datterm 
        grad[:len(model_params)] -= priterm
        grad[len(model_params):] += regterm + conterm 

        for j in range(len(model_params)):
            grad[j] *= param_map[j][2] * mu.transform_grad_param(model_params[j], model_prior[param_map[j][0]][param_map[j][1]])

        # chain rule term for image change of variables
        if logim: grad[len(mod_init):] *= imvec

        if test_gradient:
            import copy
            dx = 1e-9
            grad_numeric = np.zeros(len(grad))
            f1 = objfunc(params)
            print('\nNumeric Gradient Check: ')
            for j in range(len(grad)):
                params2 = copy.deepcopy(params)
                params2[j] += dx                
                f2 = objfunc(params2)
                grad_numeric[j] = (f2 - f1)/dx
                if j < len(mod_init):
                    print(param_map[j][0],param_map[j][1],grad[j],grad_numeric[j])
                else:
                    print(grad[j],grad_numeric[j])

        return grad

    # Define plotting function for each iteration
    global nit
    nit = 0
    def plotcur(params_step):
        model_step = params_step[:len(mod_init)]
        im_step = params_step[len(mod_init):]
        global nit
        if logim: im_step = np.exp(im_step)
        if show_updates:
            print('Model Params:',transform_params(model_step))
            chi2_1 = chisq1(im_step)
            chi2_2 = chisq2(im_step)
            chi2_3 = chisq3(im_step)
            s_1 = reg1(im_step)
            s_2 = reg2(im_step)
            s_3 = reg3(im_step)
            if np.any(np.invert(embed_mask)): im_step = embed(im_step, embed_mask)
            #plot_i(im_step, PriorIm, nit, {d1:chi2_1, d2:chi2_2, d3:chi2_3}, pol=pol)   # could sample model and plot it
            print("i: %d chi2_1: %0.2f chi2_2: %0.2f chi2_3: %0.2f prior: %0.2f s_1: %0.2f s_2: %0.2f s_3: %0.2f" % (nit, chi2_1, chi2_2, chi2_3, prior(model_step), s_1, s_2, s_3))
        nit += 1

    # Initial model parameters
    mod_init = []

    for j in range(len(param_map)):
        pm = param_map[j]
        if param_map[j][1] in trial_model.params[param_map[j][0]].keys():
            mod_init.append(mu.transform_param(model_init.params[pm[0]][pm[1]]/pm[2], model_prior[pm[0]][pm[1]],inverse=False))
        else: # In this case, the parameter is a list of complex numbers, so the real and imaginary parts need to be assigned
            param_type = 'beta_list' #param_map[j][1][:-3]
            idx = int(param_map[j][1][4:-3]) - 1
            curval = model_init.params[param_map[j][0]][param_type][idx]
            if param_map[j][1][-2:] == 're':
                mod_init.append(mu.transform_param(np.real(model_init.params[pm[0]][param_type][idx]/pm[2]), model_prior[pm[0]][pm[1]],inverse=False))
            elif param_map[j][1][-2:] == 'im':
                mod_init.append(mu.transform_param(np.imag(model_init.params[pm[0]][param_type][idx]/pm[2]), model_prior[pm[0]][pm[1]],inverse=False))
            else:
                print('Parameter ' + param_map[j][1] + ' not understood!')  

    # Initial image
    if logim:
        im_init = np.log(ninit)
    else:
        im_init = ninit

    # Full initialization
    param_init = np.concatenate([mod_init, im_init])

    # Print stats
    print("Initial S_1: %f S_2: %f S_3: %f" % (reg1(ninit), reg2(ninit), reg3(ninit)))
    print("Initial Chi^2_1: %f Chi^2_2: %f Chi^2_3: %f" % (chisq1(ninit), chisq2(ninit), chisq3(ninit)))
    print("Initial Objective Function: %f" % (objfunc(param_init)))

    if d1 in iu.DATATERMS:
        print("Total Data 1: ", (len(data1)))
    if d2 in iu.DATATERMS:
        print("Total Data 2: ", (len(data2)))
    if d3 in iu.DATATERMS:
        print("Total Data 3: ", (len(data3)))

    print("Total Fitted Real Model Parameters #: ",(len(mod_init)))
    print("Total Pixel #: ",(len(PriorIm.imvec)))
    print("Clipped Pixel #: ",(len(ninit)))
    print()
    plotcur(param_init)

    # Minimize
    optdict = {'maxiter':maxit, 'ftol':stop, 'maxcor':NHIST,'gtol':stop,'maxls':MAXLS} # minimizer dict params
    tstart = time.time()
    res = opt.minimize(objfunc, param_init, method=minimizer_method, jac=objgrad,
                       options=optdict, callback=plotcur)

    tstop = time.time()

    # Format output
    out = res.x
    model_out = out[:len(mod_init)]
    im_out = out[len(mod_init):]

    if logim: im_out = np.exp(im_out)
    if np.any(np.invert(embed_mask)): im_out = embed(im_out, embed_mask)

    outim = image.Image(im_out.reshape(PriorIm.ydim, PriorIm.xdim),
                        PriorIm.psize, PriorIm.ra, PriorIm.dec,
                        rf=PriorIm.rf, source=PriorIm.source,
                        polrep=PriorIm.polrep, pol_prim=pol, 
                        mjd=PriorIm.mjd, time=PriorIm.time, pulse=PriorIm.pulse)

    set_params(model_out)
    tparams = transform_params(model_out)

    # copy over other polarizations
    outim.copy_pol_images(InitIm)

    print("\nFitted Parameters:")
    cur_idx = -1
    for j in range(len(param_map)):
        if param_map[j][0] != cur_idx:
            cur_idx = param_map[j][0]
            print(model_init.models[cur_idx] + ' (component %d/%d):' % (cur_idx+1,model_init.N_models()))
        print(('\t' + param_map[j][1] + ': %f ' + param_map[j][3]) % tparams[j])
    print('\n')

    # Print stats
    print("time: %f s" % (tstop - tstart))
    print("J: %f" % res.fun)
    print("Final Chi^2_1: %f Chi^2_2: %f  Chi^2_3: %f" % (chisq1(im_out[embed_mask]), chisq2(im_out[embed_mask]), chisq3(im_out[embed_mask])))
    print(res.message)

    # Return fitted hybrid model
    return {'model':trial_model, 'image':outim}

##################################################################################################
# Wrapper Functions
##################################################################################################

def chisq(model, imvec, uv, A, data, sigma, dtype, ttype='direct', im_mask=None):
    """return the chi^2 for the appropriate dtype
    """

    if im_mask is None: im_mask=np.ones(len(imvec), dtype=bool)
    #mask = np.concatenate([param_mask,im_mask])
    chisq = 1
    if not dtype in iu.DATATERMS:
        return chisq

    if ttype not in ['fast','direct','nfft']:
        raise Exception("Possible ttype values are 'fast', 'direct'!, 'nfft!'")

    if ttype == 'direct':
        if dtype == 'vis':
            chisq = chisq_vis(model, imvec, uv, A, data, sigma)
        elif dtype == 'amp':
            chisq = chisq_amp(model, imvec, uv, A, data, sigma)
        elif dtype == 'logamp':
            chisq = chisq_logamp(model, imvec, uv, A, data, sigma)
        elif dtype == 'bs':
            chisq = chisq_bs(model, imvec, uv, A, data, sigma)
        elif dtype == 'cphase':
            chisq = chisq_cphase(model, imvec, uv, A, data, sigma)
        elif dtype == 'cphase_diag':
            chisq = chisq_cphase_diag(model, imvec, uv, A, data, sigma)
        elif dtype == 'camp':
            chisq = chisq_camp(model, imvec, uv, A, data, sigma)
        elif dtype == 'logcamp':
            chisq = chisq_logcamp(model, imvec, uv, A, data, sigma)
        elif dtype == 'logcamp_diag':
            chisq = chisq_logcamp_diag(model, imvec, uv, A, data, sigma)
    elif ttype== 'nfft':
        if len(im_mask)>0 and np.any(np.invert(im_mask)):
            imvec = embed(imvec, im_mask, randomfloor=True)

        if dtype == 'vis':
            chisq = chisq_vis_nfft(model, imvec, uv, A, data, sigma)
        elif dtype == 'amp':
            chisq = chisq_amp_nfft(model, imvec, uv, A, data, sigma)
        elif dtype == 'logamp':
            chisq = chisq_logamp_nfft(model, imvec, uv, A, data, sigma)
        elif dtype == 'bs':
            chisq = chisq_bs_nfft(model, imvec, uv, A, data, sigma)
        elif dtype == 'cphase':
            chisq = chisq_cphase_nfft(model, imvec, uv, A, data, sigma)
        elif dtype == 'cphase_diag':
            chisq = chisq_cphase_diag_nfft(model, imvec, uv, A, data, sigma)
        elif dtype == 'camp':
            chisq = chisq_camp_nfft(model, imvec, uv, A, data, sigma)
        elif dtype == 'logcamp':
            chisq = chisq_logcamp_nfft(model, imvec, uv, A, data, sigma)
        elif dtype == 'logcamp_diag':
            chisq = chisq_logcamp_diag_nfft(model, imvec, uv, A, data, sigma)
    else:
        print('ttype not yet implemented!')

    return chisq

def chisqgrad(model, imvec, uv, A, data, sigma, dtype, param_mask, ttype='direct', im_mask=None):
    """return the chi^2 gradient for the appropriate dtype
    """

    if im_mask is None: im_mask=np.ones(len(imvec), dtype=bool)
    mask = np.concatenate([param_mask,im_mask])

    chisqgrad = np.zeros(len(param_mask) + len(imvec))
    if not dtype in iu.DATATERMS:
        return chisqgrad[mask]

    if ttype not in ['fast','direct','nfft']:
        raise Exception("Possible ttype values are 'fast', 'direct', 'nfft'!")

    if ttype == 'direct':
        if dtype == 'vis':
            chisqgrad = chisqgrad_vis(model, imvec, uv, A, data, sigma)
        elif dtype == 'amp':
            chisqgrad = chisqgrad_amp(model, imvec, uv, A, data, sigma)
        elif dtype == 'logamp':
            chisqgrad = chisqgrad_logamp(model, imvec, uv, A, data, sigma)
        elif dtype == 'bs':
            chisqgrad = chisqgrad_bs(model, imvec, uv, A, data, sigma)
        elif dtype == 'cphase':
            chisqgrad = chisqgrad_cphase(model, imvec, uv, A, data, sigma)
        elif dtype == 'cphase_diag':
            chisqgrad = chisqgrad_cphase_diag(model, imvec, uv, A, data, sigma)
        elif dtype == 'camp':
            chisqgrad = chisqgrad_camp(model, imvec, uv, A, data, sigma)
        elif dtype == 'logcamp':
            chisqgrad = chisqgrad_logcamp(model, imvec, uv, A, data, sigma)
        elif dtype == 'logcamp_diag':
            chisqgrad = chisqgrad_logcamp_diag(model, imvec, uv, A, data, sigma)
    elif ttype== 'nfft':
        if len(im_mask)>0 and np.any(np.invert(im_mask)):
            imvec = embed(imvec, im_mask, randomfloor=True)

        if dtype == 'vis':
            chisqgrad = chisqgrad_vis_nfft(model, imvec, uv, A, data, sigma)
        elif dtype == 'amp':
            chisqgrad = chisqgrad_amp_nfft(model, imvec, uv, A, data, sigma)
        elif dtype == 'logamp':
            chisqgrad = chisqgrad_logamp_nfft(model, imvec, uv, A, data, sigma)
        elif dtype == 'bs':
            chisqgrad = chisqgrad_bs_nfft(model, imvec, uv, A, data, sigma)
        elif dtype == 'cphase':
            chisqgrad = chisqgrad_cphase_nfft(model, imvec, uv, A, data, sigma)
        elif dtype == 'cphase_diag':
            chisqgrad = chisqgrad_cphase_diag_nfft(model, imvec, uv, A, data, sigma)
        elif dtype == 'camp':
            chisqgrad = chisqgrad_camp_nfft(model, imvec, uv, A, data, sigma)
        elif dtype == 'logcamp':
            chisqgrad = chisqgrad_logcamp_nfft(model, imvec, uv, A, data, sigma)
        elif dtype == 'logcamp_diag':
            chisqgrad = chisqgrad_logcamp_diag_nfft(model, imvec, uv, A, data, sigma)

#        if len(im_mask)>0 and np.any(np.invert(im_mask)):
#            chisqgrad = chisqgrad[im_mask]

    return chisqgrad[mask]

##################################################################################################
# Chi-squared and Gradient Functions
##################################################################################################

def chisq_vis(model, imvec, uv, A, vis, sigma):
    """Visibility chi-squared"""

    samples = model.sample_uv(uv[:,0],uv[:,1]) + np.dot(A, imvec)

    return np.sum(np.abs((samples-vis)/sigma)**2)/(2*len(vis))

def chisqgrad_vis(model, imvec, uv, A, vis, sigma):
    """The gradient of the visibility chi-squared"""

    samples = model.sample_uv(uv[:,0],uv[:,1]) + np.dot(A, imvec)
    wdiff   = (vis - samples)/(sigma**2)
    grad    = model.sample_grad_uv(uv[:,0],uv[:,1])

    out = np.concatenate([
          -np.real(np.dot(grad.conj(), wdiff))/len(vis),
          -np.real(np.dot(A.conj().T, wdiff))/len(vis)
          ])
    return out

def chisq_amp(model, imvec, uv, A, amp, sigma):
    """Visibility Amplitudes (normalized) chi-squared"""

    samples = model.sample_uv(uv[:,0],uv[:,1]) + np.dot(A, imvec)
    amp_samples = np.abs(samples)
    return np.sum(np.abs((amp - amp_samples)/sigma)**2)/len(amp)

def chisqgrad_amp(model, imvec, uv, A, amp, sigma):
    """The gradient of the amplitude chi-squared"""

    samples = model.sample_uv(uv[:,0],uv[:,1]) + np.dot(A, imvec)
    amp_samples = np.abs(samples)

    pp = ((amp - amp_samples) * amp_samples) / (sigma**2) / samples
    grad = model.sample_grad_uv(uv[:,0],uv[:,1])
    out = (-2.0/len(amp)) * np.concatenate([ np.real(np.dot(grad, pp)), np.real(np.dot(pp, A)) ])
    return out

def chisq_bs(model, imvec, uv, A, bis, sigma):
    """Bispectrum chi-squared"""

    V1 = model.sample_uv(uv[0][:,0],uv[0][:,1]) + np.dot(A[0], imvec)
    V2 = model.sample_uv(uv[1][:,0],uv[1][:,1]) + np.dot(A[1], imvec)
    V3 = model.sample_uv(uv[2][:,0],uv[2][:,1]) + np.dot(A[2], imvec)

    bisamples = V1 * V2 * V3 
    chisq = np.sum(np.abs(((bis - bisamples)/sigma))**2)/(2.*len(bis))
    return chisq

def chisqgrad_bs(model, imvec, uv, A, bis, sigma):
    """The gradient of the bispectrum chi-squared"""

    V1 = model.sample_uv(uv[0][:,0],uv[0][:,1]) + np.dot(A[0], imvec)
    V2 = model.sample_uv(uv[1][:,0],uv[1][:,1]) + np.dot(A[1], imvec)
    V3 = model.sample_uv(uv[2][:,0],uv[2][:,1]) + np.dot(A[2], imvec)
    V1_grad = model.sample_grad_uv(uv[0][:,0],uv[0][:,1])
    V2_grad = model.sample_grad_uv(uv[1][:,0],uv[1][:,1])
    V3_grad = model.sample_grad_uv(uv[2][:,0],uv[2][:,1])
    bisamples = V1 * V2 * V3 
    wdiff = ((bis - bisamples).conj())/(sigma**2)
    pt1 = wdiff * V2 * V3
    pt2 = wdiff * V1 * V3
    pt3 = wdiff * V1 * V2
    out = np.concatenate([
             -np.real(np.dot(pt1, V1_grad.T) + np.dot(pt2, V2_grad.T) + np.dot(pt3, V3_grad.T))/len(bis)
             -np.real(np.dot(pt1, Amatrices[0]) + np.dot(pt2, Amatrices[1]) + np.dot(pt3, Amatrices[2]))/len(bis)
                        ])
    return out

def chisq_cphase(model, imvec, uv, A, clphase, sigma):
    """Closure Phases (normalized) chi-squared"""
    clphase = clphase * DEGREE
    sigma = sigma * DEGREE

    V1 = model.sample_uv(uv[0][:,0],uv[0][:,1]) + np.dot(A[0], imvec)
    V2 = model.sample_uv(uv[1][:,0],uv[1][:,1]) + np.dot(A[1], imvec)
    V3 = model.sample_uv(uv[2][:,0],uv[2][:,1]) + np.dot(A[2], imvec)

    clphase_samples = np.angle(V1 * V2 * V3)
    chisq = (2.0/len(clphase)) * np.sum((1.0 - np.cos(clphase-clphase_samples))/(sigma**2))
    return chisq

def chisqgrad_cphase(model, imvec, uv, A, clphase, sigma):
    """The gradient of the closure phase chi-squared"""
    clphase = clphase * DEGREE
    sigma = sigma * DEGREE

    V1 = model.sample_uv(uv[0][:,0],uv[0][:,1]) + np.dot(A[0], imvec)
    V2 = model.sample_uv(uv[1][:,0],uv[1][:,1]) + np.dot(A[1], imvec)
    V3 = model.sample_uv(uv[2][:,0],uv[2][:,1]) + np.dot(A[2], imvec)
    V1_grad = model.sample_grad_uv(uv[0][:,0],uv[0][:,1])
    V2_grad = model.sample_grad_uv(uv[1][:,0],uv[1][:,1])
    V3_grad = model.sample_grad_uv(uv[2][:,0],uv[2][:,1])

    clphase_samples = np.angle(V1 * V2 * V3)

    pref = np.sin(clphase - clphase_samples)/(sigma**2)
    pt1  = pref/V1
    pt2  = pref/V2
    pt3  = pref/V3
    out  = -(2.0/len(clphase)) * np.concatenate([
                                    np.imag(np.dot(pt1, V1_grad.T) + np.dot(pt2, V2_grad.T) + np.dot(pt3, V3_grad.T)),
                                    np.imag(np.dot(pt1, A[0]) + np.dot(pt2, A[1]) + np.dot(pt3, A[2]))
                                ])
    return out

def chisq_cphase_diag(model, imvec, uv, A, vis, sigma):
    """Diagonalized closure phases (normalized) chi-squared"""
    clphase_diag = np.concatenate(clphase_diag) * DEGREE
    sigma = np.concatenate(sigma) * DEGREE

    uv_diag = uv[0]
    tform_mats = uv[1]

    clphase_diag_samples = []
    for iA, uv3 in enumerate(uv_diag):
        i1 = model.sample_uv(uv3[0][:,0],uv3[0][:,1])    
        i2 = model.sample_uv(uv3[1][:,0],uv3[1][:,1])    
        i3 = model.sample_uv(uv3[2][:,0],uv3[2][:,1])  

        clphase_samples = np.angle(i1 * i2 * i3)
        clphase_diag_samples.append(np.dot(tform_mats[iA],clphase_samples))
    clphase_diag_samples = np.concatenate(clphase_diag_samples)

    chisq = (2.0/len(clphase_diag)) * np.sum((1.0 - np.cos(clphase_diag-clphase_diag_samples))/(sigma**2))
    return chisq

def chisqgrad_cphase_diag(model, imvec, uv, A, vis, sigma):
    """The gradient of the diagonalized closure phase chi-squared"""
    clphase_diag = clphase_diag * DEGREE
    sigma = sigma * DEGREE

    uv_diag = uv[0]
    tform_mats = uv[1]

    deriv = np.zeros(len(model.sample_grad_uv(0,0)))
    for iA, uv3 in enumerate(uv_diag):

        i1 = model.sample_uv(uv3[0][:,0],uv3[0][:,1])    
        i2 = model.sample_uv(uv3[1][:,0],uv3[1][:,1])    
        i3 = model.sample_uv(uv3[2][:,0],uv3[2][:,1])   

        i1_grad = model.sample_grad_uv(uv3[0][:,0],uv3[0][:,1])
        i2_grad = model.sample_grad_uv(uv3[1][:,0],uv3[1][:,1])
        i3_grad = model.sample_grad_uv(uv3[2][:,0],uv3[2][:,1])
 
        clphase_samples = np.angle(i1 * i2 * i3)
        clphase_diag_samples = np.dot(tform_mats[iA],clphase_samples)

        clphase_diag_measured = clphase_diag[iA]
        clphase_diag_sigma = sigma[iA]

        term1 = np.dot(np.dot((np.sin(clphase_diag_measured-clphase_diag_samples)/(clphase_diag_sigma**2.0)),(tform_mats[iA]/i1)),i1_grad.T)
        term2 = np.dot(np.dot((np.sin(clphase_diag_measured-clphase_diag_samples)/(clphase_diag_sigma**2.0)),(tform_mats[iA]/i2)),i2_grad.T)
        term3 = np.dot(np.dot((np.sin(clphase_diag_measured-clphase_diag_samples)/(clphase_diag_sigma**2.0)),(tform_mats[iA]/i3)),i3_grad.T)
        deriv += -2.0*np.imag(term1 + term2 + term3)

    deriv *= 1.0/np.float(len(np.concatenate(clphase_diag)))

    return deriv

def chisq_camp(model, imvec, uv, A, clamp, sigma):
    """Closure Amplitudes (normalized) chi-squared"""

    V1 = model.sample_uv(uv[0][:,0],uv[0][:,1]) + np.dot(A[0], imvec)
    V2 = model.sample_uv(uv[1][:,0],uv[1][:,1]) + np.dot(A[1], imvec)
    V3 = model.sample_uv(uv[2][:,0],uv[2][:,1]) + np.dot(A[2], imvec)
    V4 = model.sample_uv(uv[3][:,0],uv[3][:,1]) + np.dot(A[3], imvec)

    clamp_samples = np.abs(V1 * V2 / (V3 * V4))
    chisq = np.sum(np.abs((clamp - clamp_samples)/sigma)**2)/len(clamp)
    return chisq

def chisqgrad_camp(model, imvec, uv, A, clamp, sigma):
    """The gradient of the closure amplitude chi-squared"""

    V1 = model.sample_uv(uv[0][:,0],uv[0][:,1]) + np.dot(A[0], imvec)
    V2 = model.sample_uv(uv[1][:,0],uv[1][:,1]) + np.dot(A[1], imvec)
    V3 = model.sample_uv(uv[2][:,0],uv[2][:,1]) + np.dot(A[2], imvec)
    V4 = model.sample_uv(uv[3][:,0],uv[3][:,1]) + np.dot(A[3], imvec)
    V1_grad = model.sample_grad_uv(uv[0][:,0],uv[0][:,1])
    V2_grad = model.sample_grad_uv(uv[1][:,0],uv[1][:,1])
    V3_grad = model.sample_grad_uv(uv[2][:,0],uv[2][:,1])
    V4_grad = model.sample_grad_uv(uv[3][:,0],uv[3][:,1])

    clamp_samples = np.abs((V1 * V2)/(V3 * V4))

    pp = ((clamp - clamp_samples) * clamp_samples)/(sigma**2)
    pt1 =  pp/V1
    pt2 =  pp/V2
    pt3 = -pp/V3
    pt4 = -pp/V4
    out = (-2.0/len(clamp)) * np.concatenate([
                                 np.real(np.dot(pt1, V1_grad.T) + np.dot(pt2, V2_grad.T) + np.dot(pt3, V3_grad.T) + np.dot(pt4, V4_grad.T)),
                                 np.real(np.dot(pt1, A[0]) + np.dot(pt2, A[1]) + np.dot(pt3, A[2]) + np.dot(pt4, A[3]))
                              ])
    return out

def chisq_logcamp(model, imvec, uv, A, log_clamp, sigma):
    """Log Closure Amplitudes (normalized) chi-squared"""

    V1 = model.sample_uv(uv[0][:,0],uv[0][:,1]) + np.dot(A[0], imvec)
    V2 = model.sample_uv(uv[1][:,0],uv[1][:,1]) + np.dot(A[1], imvec)
    V3 = model.sample_uv(uv[2][:,0],uv[2][:,1]) + np.dot(A[2], imvec)
    V4 = model.sample_uv(uv[3][:,0],uv[3][:,1]) + np.dot(A[3], imvec)

    a1 = np.abs(V1)
    a2 = np.abs(V2)
    a3 = np.abs(V3)
    a4 = np.abs(V4)

    samples = np.log(a1) + np.log(a2) - np.log(a3) - np.log(a4)
    chisq = np.sum(np.abs((log_clamp - samples)/sigma)**2) / (len(log_clamp))
    return  chisq

def chisqgrad_logcamp(model, imvec, uv, A, log_clamp, sigma):
    """The gradient of the Log closure amplitude chi-squared"""

    V1 = model.sample_uv(uv[0][:,0],uv[0][:,1]) + np.dot(A[0], imvec)
    V2 = model.sample_uv(uv[1][:,0],uv[1][:,1]) + np.dot(A[1], imvec)
    V3 = model.sample_uv(uv[2][:,0],uv[2][:,1]) + np.dot(A[2], imvec)
    V4 = model.sample_uv(uv[3][:,0],uv[3][:,1]) + np.dot(A[3], imvec)
    V1_grad = model.sample_grad_uv(uv[0][:,0],uv[0][:,1])
    V2_grad = model.sample_grad_uv(uv[1][:,0],uv[1][:,1])
    V3_grad = model.sample_grad_uv(uv[2][:,0],uv[2][:,1])
    V4_grad = model.sample_grad_uv(uv[3][:,0],uv[3][:,1])

    log_clamp_samples = np.log(np.abs(V1)) + np.log(np.abs(V2)) - np.log(np.abs(V3)) - np.log(np.abs(V4))

    pp = (log_clamp - log_clamp_samples) / (sigma**2)
    pt1 = pp / V1
    pt2 = pp / V2
    pt3 = -pp / V3
    pt4 = -pp / V4
    out = (-2.0/len(log_clamp)) * np.concatenate([
                                    np.real(np.dot(pt1, V1_grad.T) + np.dot(pt2, V2_grad.T) + np.dot(pt3, V3_grad.T) + np.dot(pt4, V4_grad.T)),
                                    np.real(np.dot(pt1, A[0]) + np.dot(pt2, A[1]) + np.dot(pt3, A[2]) + np.dot(pt4, A[3]))
                                    ])
    return out

def chisq_logcamp_diag(model, imvec, uv, A, vis, sigma):
    """Diagonalized log closure amplitudes (normalized) chi-squared"""

    log_clamp_diag = np.concatenate(log_clamp_diag)
    sigma = np.concatenate(sigma)

    uv_diag = uv[0]
    tform_mats = uv[1]

    log_clamp_diag_samples = []
    for iA, uv4 in enumerate(uv_diag):

        a1 = np.abs(model.sample_uv(uv4[0][:,0],uv4[0][:,1]))          
        a2 = np.abs(model.sample_uv(uv4[1][:,0],uv4[1][:,1])) 
        a3 = np.abs(model.sample_uv(uv4[2][:,0],uv4[2][:,1])) 
        a4 = np.abs(model.sample_uv(uv4[3][:,0],uv4[3][:,1])) 

        log_clamp_samples = np.log(a1) + np.log(a2) - np.log(a3) - np.log(a4)
        log_clamp_diag_samples.append(np.dot(tform_mats[iA],log_clamp_samples))

    log_clamp_diag_samples = np.concatenate(log_clamp_diag_samples)

    chisq = np.sum(np.abs((log_clamp_diag - log_clamp_diag_samples)/sigma)**2) / (len(log_clamp_diag))
    return  chisq

def chisqgrad_logcamp_diag(model, imvec, uv, A, vis, sigma):
    """The gradient of the diagonalized log closure amplitude chi-squared"""

    uv_diag = uv[0]
    tform_mats = uv[1]

    deriv = np.zeros(len(model.sample_grad_uv(0,0)))
    for iA, uv4 in enumerate(uv_diag):

        i1 = model.sample_uv(uv4[0][:,0],uv4[0][:,1])
        i2 = model.sample_uv(uv4[1][:,0],uv4[1][:,1])
        i3 = model.sample_uv(uv4[2][:,0],uv4[2][:,1])
        i4 = model.sample_uv(uv4[3][:,0],uv4[3][:,1])

        i1_grad = model.sample_grad_uv(uv4[0][:,0],uv4[0][:,1])
        i2_grad = model.sample_grad_uv(uv4[1][:,0],uv4[1][:,1])
        i3_grad = model.sample_grad_uv(uv4[2][:,0],uv4[2][:,1])
        i4_grad = model.sample_grad_uv(uv4[3][:,0],uv4[3][:,1])

        log_clamp_samples = np.log(np.abs(i1)) + np.log(np.abs(i2)) - np.log(np.abs(i3)) - np.log(np.abs(i4))
        log_clamp_diag_samples = np.dot(tform_mats[iA],log_clamp_samples)

        log_clamp_diag_measured = log_clamp_diag[iA]
        log_clamp_diag_sigma = sigma[iA]

        term1 = np.dot(np.dot(((log_clamp_diag_measured-log_clamp_diag_samples)/(log_clamp_diag_sigma**2.0)),(tform_mats[iA]/i1)),i1_grad.T)
        term2 = np.dot(np.dot(((log_clamp_diag_measured-log_clamp_diag_samples)/(log_clamp_diag_sigma**2.0)),(tform_mats[iA]/i2)),i2_grad.T)
        term3 = np.dot(np.dot(((log_clamp_diag_measured-log_clamp_diag_samples)/(log_clamp_diag_sigma**2.0)),(tform_mats[iA]/i3)),i3_grad.T)
        term4 = np.dot(np.dot(((log_clamp_diag_measured-log_clamp_diag_samples)/(log_clamp_diag_sigma**2.0)),(tform_mats[iA]/i4)),i4_grad.T)
        deriv += -2.0*np.real(term1 + term2 - term3 - term4)

    deriv *= 1.0/np.float(len(np.concatenate(log_clamp_diag)))

    return deriv

def chisq_logamp(model, imvec, uv, A, amp, sigma):
    """Log Visibility Amplitudes (normalized) chi-squared"""

    # to lowest order the variance on the logarithm of a quantity x is
    # sigma^2_log(x) = sigma^2/x^2
    logsigma = sigma / amp

    amp_samples = np.abs(model.sample_uv(uv[:,0],uv[:,1]))
    return np.sum(np.abs((np.log(amp) - np.log(amp_samples))/logsigma)**2)/len(amp)

def chisqgrad_logamp(model, imvec, uv, A, amp, sigma):
    """The gradient of the Log amplitude chi-squared"""

    # to lowest order the variance on the logarithm of a quantity x is
    # sigma^2_log(x) = sigma^2/x^2
    logsigma = sigma / amp

    i1 = model.sample_uv(uv[:,0],uv[:,1])
    amp_samples = np.abs(i1)

    V_grad = model.sample_grad_uv(uv[:,0],uv[:,1])

    pp = ((np.log(amp) - np.log(amp_samples))) / (logsigma**2) / i1
    out = (-2.0/len(amp)) * np.real(np.dot(pp, V_grad.T))
    return out

##################################################################################################
# NFFT Chi-squared and Gradient Functions
##################################################################################################
def chisq_vis_nfft(model, imvec, uv, A, vis, sigma):
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
    samples += model.sample_uv(uv[:,0],uv[:,1]) 

    #compute chi^2
    chisq = np.sum(np.abs((samples-vis)/sigma)**2)/(2*len(vis))

    return chisq

def chisqgrad_vis_nfft(model, imvec, uv, A, vis, sigma):

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
    samples += model.sample_uv(uv[:,0],uv[:,1]) 

    # model gradient
    mod_grad = model.sample_grad_uv(uv[:,0],uv[:,1])

    # gradient vec for adjoint FT
    wdiff_vec = (-1.0/len(vis)*(vis - samples)/(sigma**2)) 
    plan.f = wdiff_vec * pulsefac.conj()
    plan.adjoint()
    grad = np.concatenate([
           np.real(np.dot(mod_grad.conj(), wdiff_vec)),
           np.real((plan.f_hat.copy().T).reshape(nfft_info.xdim*nfft_info.ydim))
           ])

    return grad

def chisq_amp_nfft(model, imvec, uv, A, amp, sigma):
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
    samples += model.sample_uv(uv[:,0],uv[:,1]) 

    #compute chi^2
    amp_samples = np.abs(samples)
    chisq = np.sum(np.abs((amp_samples-amp)/sigma)**2)/(len(amp))

    return chisq

def chisqgrad_amp_nfft(model, imvec, uv, A, amp, sigma):

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
    samples += model.sample_uv(uv[:,0],uv[:,1]) 
    amp_samples=np.abs(samples)

    # model gradient
    mod_grad = model.sample_grad_uv(uv[:,0],uv[:,1])

    # gradient vec for adjoint FT
    wdiff_vec = (-2.0/len(amp)*((amp - amp_samples) * samples) / (sigma**2) / amp_samples) 
    plan.f = wdiff_vec * pulsefac.conj()
    plan.adjoint()
    out = np.concatenate([ np.real(np.dot(mod_grad.conj(), wdiff_vec)), 
                           np.real((plan.f_hat.copy().T).reshape(nfft_info.xdim*nfft_info.ydim)) ])

    return out

def chisq_bs_nfft(model, imvec, uv, A, bis, sigma):
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
    samples1 += model.sample_uv(uv[0][:,0],uv[0][:,1])

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim,nfft_info2.xdim)).T
    plan2.trafo()
    samples2 = plan2.f.copy()*pulsefac2
    samples2 += model.sample_uv(uv[1][:,0],uv[1][:,1])

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim,nfft_info3.xdim)).T
    plan3.trafo()
    samples3 = plan3.f.copy()*pulsefac3
    samples3 += model.sample_uv(uv[2][:,0],uv[2][:,1])

    #compute chi^2
    bisamples = samples1*samples2*samples3
    chisq = np.sum(np.abs(((bis - bisamples)/sigma))**2)/(2.*len(bis))
    return chisq

def chisqgrad_bs_nfft(model, imvec, uv, A, bis, sigma):
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
    v1 += model.sample_uv(uv[0][:,0],uv[0][:,1])

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim,nfft_info2.xdim)).T
    plan2.trafo()
    v2 = plan2.f.copy()*pulsefac2
    v2 += model.sample_uv(uv[1][:,0],uv[1][:,1])

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim,nfft_info3.xdim)).T
    plan3.trafo()
    v3 = plan3.f.copy()*pulsefac3
    v3 += model.sample_uv(uv[2][:,0],uv[2][:,1])

    # gradient vec for adjoint FT
    bisamples = v1*v2*v3
    wdiff = -1.0/len(bis)*(bis - bisamples)/(sigma**2)
    pt1 = wdiff * (v2 * v3).conj()
    pt2 = wdiff * (v1 * v3).conj()
    pt3 = wdiff * (v1 * v2).conj()

    # Model gradient 
    V1_grad = model.sample_grad_uv(uv[0][:,0],uv[0][:,1])
    V2_grad = model.sample_grad_uv(uv[1][:,0],uv[1][:,1])
    V3_grad = model.sample_grad_uv(uv[2][:,0],uv[2][:,1])

    # Setup and perform the inverse FFT
    plan1.f = pt1 * pulsefac1.conj()
    plan1.adjoint()
    out1 = np.concatenate([np.real(np.dot(pt1, V1_grad.conj().T)),
                           np.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))])

    plan2.f = pt2 * pulsefac2.conj()
    plan2.adjoint()
    out2 = np.concatenate([np.real(np.dot(pt2, V2_grad.conj().T)),
                           np.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))])

    plan3.f = pt3 * pulsefac3.conj()
    plan3.adjoint()
    out3 = np.concatenate([np.real(np.dot(pt3, V3_grad.conj().T)),
                           np.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))])

    out = out1 + out2 + out3
    return out

def chisq_cphase_nfft(model, imvec, uv, A, clphase, sigma):
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
    samples1 += model.sample_uv(uv[0][:,0],uv[0][:,1])

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim,nfft_info2.xdim)).T
    plan2.trafo()
    samples2 = plan2.f.copy()*pulsefac2
    samples2 += model.sample_uv(uv[1][:,0],uv[1][:,1])

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim,nfft_info3.xdim)).T
    plan3.trafo()
    samples3 = plan3.f.copy()*pulsefac3
    samples3 += model.sample_uv(uv[2][:,0],uv[2][:,1])

    #compute chi^2
    clphase_samples = np.angle(samples1*samples2*samples3)
    chisq = (2.0/len(clphase)) * np.sum((1.0 - np.cos(clphase-clphase_samples))/(sigma**2))

    return chisq

def chisqgrad_cphase_nfft(model, imvec, uv, A, clphase, sigma):
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
    v1 += model.sample_uv(uv[0][:,0],uv[0][:,1])

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim,nfft_info2.xdim)).T
    plan2.trafo()
    v2 = plan2.f.copy()*pulsefac2
    v2 += model.sample_uv(uv[1][:,0],uv[1][:,1])

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim,nfft_info3.xdim)).T
    plan3.trafo()
    v3 = plan3.f.copy()*pulsefac3
    v3 += model.sample_uv(uv[2][:,0],uv[2][:,1])

    # gradient vec for adjoint FT
    clphase_samples = np.angle(v1*v2*v3)
    pref = (2.0/len(clphase)) * np.sin(clphase - clphase_samples)/(sigma**2)
    pt1  = pref/v1.conj()
    pt2  = pref/v2.conj()
    pt3  = pref/v3.conj()

    # Model gradient 
    V1_grad = model.sample_grad_uv(uv[0][:,0],uv[0][:,1])
    V2_grad = model.sample_grad_uv(uv[1][:,0],uv[1][:,1])
    V3_grad = model.sample_grad_uv(uv[2][:,0],uv[2][:,1])

    # Setup and perform the inverse FFT
    plan1.f = pt1 * pulsefac1.conj()
    plan1.adjoint()
    out1 = np.concatenate([np.imag(np.dot(pt1, V1_grad.conj().T)),
                           np.imag((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))])

    plan2.f = pt2 * pulsefac2.conj()
    plan2.adjoint()
    out2 = np.concatenate([np.imag(np.dot(pt2, V2_grad.conj().T)),
                           np.imag((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))])

    plan3.f = pt3 * pulsefac3.conj()
    plan3.adjoint()
    out3 = np.concatenate([np.imag(np.dot(pt3, V3_grad.conj().T)),
                           np.imag((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))])

    out = out1 + out2 + out3
    return out

def chisq_cphase_diag_nfft(model, imvec, uv, A, clphase_diag, sigma):
    """Diagonalized closure phases (normalized) chi-squared from nfft
    """

    clphase_diag = np.concatenate(clphase_diag) * DEGREE
    sigma = np.concatenate(sigma) * DEGREE

    A3 = A[0]
    tform_mats = A[1]

    #get nfft objects
    nfft_info1 = A3[0]
    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac

    nfft_info2 = A3[1]
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac

    nfft_info3 = A3[2]
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

    clphase_samples = np.angle(samples1*samples2*samples3)

    count = 0
    clphase_diag_samples = []
    for tform_mat in tform_mats:
        clphase_samples_here = clphase_samples[count:count+len(tform_mat)]
        clphase_diag_samples.append(np.dot(tform_mat,clphase_samples_here))
        count += len(tform_mat)

    clphase_diag_samples = np.concatenate(clphase_diag_samples)

    #compute chi^2
    chisq= (2.0/len(clphase_diag)) * np.sum((1.0 - np.cos(clphase_diag-clphase_diag_samples))/(sigma**2))

    return chisq

def chisqgrad_cphase_diag_nfft(model, imvec, uv, A, clphase_diag, sigma):
    """The gradient of the diagonalized closure phase chi-squared from nfft"""

    clphase_diag = np.concatenate(clphase_diag) * DEGREE
    sigma = np.concatenate(sigma) * DEGREE

    A3 = A[0]
    tform_mats = A[1]

    #get nfft objects
    nfft_info1 = A3[0]
    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac

    nfft_info2 = A3[1]
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac

    nfft_info3 = A3[2]
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

    clphase_samples = np.angle(v1*v2*v3)

    # gradient vec for adjoint FT
    count = 0
    pref = np.zeros_like(clphase_samples)
    for tform_mat in tform_mats:

        clphase_diag_samples = np.dot(tform_mat,clphase_samples[count:count+len(tform_mat)])
        clphase_diag_measured = clphase_diag[count:count+len(tform_mat)]
        clphase_diag_sigma = sigma[count:count+len(tform_mat)]

        for j in range(len(clphase_diag_measured)):
            pref[count:count+len(tform_mat)] += 2.0 * tform_mat[j,:] * np.sin(clphase_diag_measured[j] - clphase_diag_samples[j])/(clphase_diag_sigma[j]**2)

        count += len(tform_mat)

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

    deriv = out1 + out2 + out3
    deriv *= 1.0/np.float(len(clphase_diag))

    return deriv

def chisq_camp_nfft(model, imvec, uv, A, clamp, sigma):
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
    samples1 += model.sample_uv(uv[0][:,0],uv[0][:,1])

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim,nfft_info2.xdim)).T
    plan2.trafo()
    samples2 = plan2.f.copy()*pulsefac2
    samples2 += model.sample_uv(uv[1][:,0],uv[1][:,1])

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim,nfft_info3.xdim)).T
    plan3.trafo()
    samples3 = plan3.f.copy()*pulsefac3
    samples3 += model.sample_uv(uv[2][:,0],uv[2][:,1])

    plan4.f_hat = imvec.copy().reshape((nfft_info4.ydim,nfft_info4.xdim)).T
    plan4.trafo()
    samples4 = plan4.f.copy()*pulsefac4
    samples4 += model.sample_uv(uv[3][:,0],uv[3][:,1])

    #compute chi^2
    clamp_samples = np.abs((samples1*samples2)/(samples3*samples4))
    chisq = np.sum(np.abs((clamp - clamp_samples)/sigma)**2)/len(clamp)
    return chisq

def chisqgrad_camp_nfft(model, imvec, uv, A, clamp, sigma):

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
    v1 += model.sample_uv(uv[0][:,0],uv[0][:,1])

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim,nfft_info2.xdim)).T
    plan2.trafo()
    v2 = plan2.f.copy()*pulsefac2
    v2 += model.sample_uv(uv[1][:,0],uv[1][:,1])

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim,nfft_info3.xdim)).T
    plan3.trafo()
    v3 = plan3.f.copy()*pulsefac3
    v3 += model.sample_uv(uv[2][:,0],uv[2][:,1])

    plan4.f_hat = imvec.copy().reshape((nfft_info4.ydim,nfft_info4.xdim)).T
    plan4.trafo()
    v4 = plan4.f.copy()*pulsefac4
    v4 += model.sample_uv(uv[3][:,0],uv[3][:,1])

    # Model gradient 
    V1_grad = model.sample_grad_uv(uv[0][:,0],uv[0][:,1])
    V2_grad = model.sample_grad_uv(uv[1][:,0],uv[1][:,1])
    V3_grad = model.sample_grad_uv(uv[2][:,0],uv[2][:,1])
    V4_grad = model.sample_grad_uv(uv[3][:,0],uv[3][:,1])

    # gradient vec for adjoint FT
    clamp_samples = np.abs((v1 * v2)/(v3 * v4))

    pp = (-2.0/len(clamp)) * ((clamp - clamp_samples) * clamp_samples)/(sigma**2)
    pt1 =  pp/v1.conj()
    pt2 =  pp/v2.conj()
    pt3 = -pp/v3.conj()
    pt4 = -pp/v4.conj()

    # Setup and perform the inverse FFT
    plan1.f = pt1 * pulsefac1.conj()
    plan1.adjoint()
    out1 = np.concatenate([np.real(np.dot(pt1, V1_grad.conj().T)),
                           np.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))])

    plan2.f = pt2 * pulsefac2.conj()
    plan2.adjoint()
    out2 = np.concatenate([np.real(np.dot(pt2, V2_grad.conj().T)),
                           np.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))])

    plan3.f = pt3 * pulsefac3.conj()
    plan3.adjoint()
    out3 = np.concatenate([np.real(np.dot(pt3, V3_grad.conj().T)),
                           np.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))])

    plan4.f = pt4 * pulsefac4.conj()
    plan4.adjoint()
    out4 = np.concatenate([np.real(np.dot(pt4, V4_grad.conj().T)),
                           np.real((plan4.f_hat.copy().T).reshape(nfft_info4.xdim*nfft_info4.ydim))])

    out = out1 + out2 + out3 + out4
    return out

def chisq_logcamp_nfft(model, imvec, uv, A, log_clamp, sigma):
    """Log Closure Amplitudes (normalized) chi-squared from fft
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
    samples1 += model.sample_uv(uv[0][:,0],uv[0][:,1])

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim,nfft_info2.xdim)).T
    plan2.trafo()
    samples2 = plan2.f.copy()*pulsefac2
    samples2 += model.sample_uv(uv[1][:,0],uv[1][:,1])

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim,nfft_info3.xdim)).T
    plan3.trafo()
    samples3 = plan3.f.copy()*pulsefac3
    samples3 += model.sample_uv(uv[2][:,0],uv[2][:,1])

    plan4.f_hat = imvec.copy().reshape((nfft_info4.ydim,nfft_info4.xdim)).T
    plan4.trafo()
    samples4 = plan4.f.copy()*pulsefac4
    samples4 += model.sample_uv(uv[3][:,0],uv[3][:,1])

    #compute chi^2
    log_clamp_samples = (np.log(np.abs(samples1)) + np.log(np.abs(samples2))
                         - np.log(np.abs(samples3)) - np.log(np.abs(samples4)))
    chisq = np.sum(np.abs((log_clamp - log_clamp_samples)/sigma)**2) / (len(log_clamp))
    return chisq

def chisqgrad_logcamp_nfft(model, imvec, uv, A, log_clamp, sigma):

    """The gradient of the log closure amplitude chi-squared from fft
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
    v1 += model.sample_uv(uv[0][:,0],uv[0][:,1])

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim,nfft_info2.xdim)).T
    plan2.trafo()
    v2 = plan2.f.copy()*pulsefac2
    v2 += model.sample_uv(uv[1][:,0],uv[1][:,1])

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim,nfft_info3.xdim)).T
    plan3.trafo()
    v3 = plan3.f.copy()*pulsefac3
    v3 += model.sample_uv(uv[2][:,0],uv[2][:,1])

    plan4.f_hat = imvec.copy().reshape((nfft_info4.ydim,nfft_info4.xdim)).T
    plan4.trafo()
    v4 = plan4.f.copy()*pulsefac4
    v4 += model.sample_uv(uv[3][:,0],uv[3][:,1])

    # Model gradient 
    V1_grad = model.sample_grad_uv(uv[0][:,0],uv[0][:,1])
    V2_grad = model.sample_grad_uv(uv[1][:,0],uv[1][:,1])
    V3_grad = model.sample_grad_uv(uv[2][:,0],uv[2][:,1])
    V4_grad = model.sample_grad_uv(uv[3][:,0],uv[3][:,1])

    # gradient vec for adjoint FT
    log_clamp_samples = np.log(np.abs((v1 * v2)/(v3 * v4)))

    pp = (-2.0/len(log_clamp)) * (log_clamp - log_clamp_samples) / (sigma**2)
    pt1 = pp / v1.conj()
    pt2 = pp / v2.conj()
    pt3 = -pp / v3.conj()
    pt4 = -pp / v4.conj() 

    # Setup and perform the inverse FFT
    plan1.f = pt1 * pulsefac1.conj()
    plan1.adjoint()
    out1 = np.concatenate([np.real(np.dot(pt1, V1_grad.conj().T)),
                           np.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))])

    plan2.f = pt2 * pulsefac2.conj()
    plan2.adjoint()
    out2 = np.concatenate([np.real(np.dot(pt2, V2_grad.conj().T)),
                           np.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))])

    plan3.f = pt3 * pulsefac3.conj()
    plan3.adjoint()
    out3 = np.concatenate([np.real(np.dot(pt3, V3_grad.conj().T)),
                           np.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))])

    plan4.f = pt4 * pulsefac4.conj()
    plan4.adjoint()
    out4 = np.concatenate([np.real(np.dot(pt4, V4_grad.conj().T)),
                           np.real((plan4.f_hat.copy().T).reshape(nfft_info4.xdim*nfft_info4.ydim))])

    out = out1 + out2 + out3 + out4
    return out

def chisq_logcamp_diag_nfft(model, imvec, uv, A, log_clamp_diag, sigma):
    """Diagonalized log closure amplitudes (normalized) chi-squared from nfft
    """

    log_clamp_diag = np.concatenate(log_clamp_diag)
    sigma = np.concatenate(sigma)

    A4 = A[0]
    tform_mats = A[1]

    #get nfft objects
    nfft_info1 = A4[0]
    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac

    nfft_info2 = A4[1]
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac

    nfft_info3 = A4[2]
    plan3 = nfft_info3.plan
    pulsefac3 = nfft_info3.pulsefac

    nfft_info4 = A4[3]
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

    log_clamp_samples = (np.log(np.abs(samples1)) + np.log(np.abs(samples2))
                        - np.log(np.abs(samples3)) - np.log(np.abs(samples4)))

    count = 0
    log_clamp_diag_samples = []
    for tform_mat in tform_mats:
        log_clamp_samples_here = log_clamp_samples[count:count+len(tform_mat)]
        log_clamp_diag_samples.append(np.dot(tform_mat,log_clamp_samples_here))
        count += len(tform_mat)

    log_clamp_diag_samples = np.concatenate(log_clamp_diag_samples)

    #compute chi^2
    chisq = np.sum(np.abs((log_clamp_diag - log_clamp_diag_samples)/sigma)**2) / (len(log_clamp_diag))

    return chisq

def chisqgrad_logcamp_diag_nfft(model, imvec, uv, A, log_clamp_diag, sigma):

    """The gradient of the diagonalized log closure amplitude chi-squared from fft
    """

    log_clamp_diag = np.concatenate(log_clamp_diag)
    sigma = np.concatenate(sigma)

    A4 = A[0]
    tform_mats = A[1]

    #get nfft objects
    nfft_info1 = A4[0]
    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac

    nfft_info2 = A4[1]
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac

    nfft_info3 = A4[2]
    plan3 = nfft_info3.plan
    pulsefac3 = nfft_info3.pulsefac

    nfft_info4 = A4[3]
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

    log_clamp_samples = np.log(np.abs((v1 * v2)/(v3 * v4)))

    # gradient vec for adjoint FT
    count = 0
    pp = np.zeros_like(log_clamp_samples)
    for tform_mat in tform_mats:

        log_clamp_diag_samples = np.dot(tform_mat,log_clamp_samples[count:count+len(tform_mat)])
        log_clamp_diag_measured = log_clamp_diag[count:count+len(tform_mat)]
        log_clamp_diag_sigma = sigma[count:count+len(tform_mat)]

        for j in range(len(log_clamp_diag_measured)):
            pp[count:count+len(tform_mat)] += -2.0 * tform_mat[j,:] * (log_clamp_diag_measured[j] - log_clamp_diag_samples[j])/(log_clamp_diag_sigma[j]**2)

        count += len(tform_mat)

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

    deriv = out1 + out2 + out3 + out4
    deriv *= 1.0/np.float(len(log_clamp_diag))

    return deriv

def chisq_logamp_nfft(model, imvec, uv, A, amp, sigma):
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
    samples += model.sample_uv(uv[:,0],uv[:,1])

    # to lowest order the variance on the logarithm of a quantity x is
    # sigma^2_log(x) = sigma^2/x^2
    logsigma = sigma / amp

    #compute chi^2
    amp_samples = np.abs(samples)
    chisq = np.sum(np.abs((np.log(amp_samples)-np.log(amp))/logsigma)**2)/(len(amp))

    return chisq

def chisqgrad_logamp_nfft(model, imvec, uv, A, amp, sigma):

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
    samples += model.sample_uv(uv[:,0],uv[:,1])
    amp_samples=np.abs(samples)

    # model gradient
    V_grad = model.sample_grad_uv(uv[:,0],uv[:,1])

    # gradient vec for adjoint FT
    logsigma = sigma / amp
    wdiff_vec = (-2.0/len(amp)*((np.log(amp) - np.log(amp_samples))) / (logsigma**2) / samples.conj())
    plan.f = wdiff_vec * pulsefac.conj()
    plan.adjoint()
    out = np.concatenate([np.real(np.dot(wdiff_vec, V_grad.conj().T)),
                          np.real((plan.f_hat.copy().T).reshape(nfft_info.xdim*nfft_info.ydim))])

    return out

