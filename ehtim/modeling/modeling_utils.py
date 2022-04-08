# modeling_utils.py
# General modeling functions for total intensity VLBI data
#
#    Copyright (C) 2020 Michael Johnson
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

## TODO ##
# >> return jonesdict for all data types <- requires significant modification to eht-imaging
# >> Deal with nans in fitting (mask chisqdata) <- mostly done
# >> Add optional transform for leakage and gains

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
import scipy.special as sps
import scipy.stats as stats
import copy 

import ehtim.obsdata as obsdata
import ehtim.image as image
import ehtim.model as model
import ehtim.caltable as caltable

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *
from ehtim.statistics.dataframes import *

#from IPython import display

##################################################################################################
# Constants & Definitions
##################################################################################################

MAXLS = 100  # maximum number of line searches in L-BFGS-B
NHIST = 100  # number of steps to store for hessian approx
MAXIT = 100  # maximum number of iterations
STOP = 1.e-8 # convergence criterion

BOUNDS_MIN = -1e4
BOUNDS_MAX = 1e4
BOUNDS_GAUSS_NSIGMA = 10.
BOUNDS_EXP_NSIGMA = 10.

PRIOR_MIN = 1e-200 # to avoid problems with log-prior

DATATERMS = ['vis', 'bs', 'amp', 'cphase', 'cphase_diag', 'camp', 'logcamp', 'logcamp_diag', 'logamp', 'pvis', 'm', 'rlrr', 'rlll', 'lrrr', 'lrll','rrll','llrr','polclosure']

nit = 0       # global variable to track the iteration number in the plotting callback
globdict = {} # global dictionary with all parameters related to the model fitting (mainly for efficient parallelization, but also very useful for debugging)

# Details on each fitted parameter (convenience rescaling factor and associated unit)
PARAM_DETAILS = {'F0':[1.,'Jy'], 'FWHM':[RADPERUAS,'uas'], 'FWHM_maj':[RADPERUAS,'uas'], 'FWHM_min':[RADPERUAS,'uas'], 
                 'd':[RADPERUAS,'uas'], 'PA':[np.pi/180.,'deg'], 'alpha':[RADPERUAS,'uas'], 'ff':[1.,''], 
                 'x0':[RADPERUAS,'uas'], 'y0':[RADPERUAS,'uas'], 'stretch':[1.,''], 'stretch_PA':[np.pi/180.,'deg'], 
                 'arg':[np.pi/180.,'deg'], 'evpa':[np.pi/180.,'deg'], 'phi':[np.pi/180.,'deg']}

GAIN_PRIOR_DEFAULT = {'prior_type':'lognormal','sigma':0.1,'mu':0.0,'shift':-1.0} 
LEAKAGE_PRIOR_DEFAULT = {'prior_type':'flat','min':-0.5,'max':0.5} 
N_POSTERIOR_SAMPLES = 100

##################################################################################################
# Priors
##################################################################################################

def cdf(x, prior_params):
    """Compute the cumulative distribution function CDF(x) of a given prior at a given point x

       Args:
           x (float): Value at which to compute the CDF 
           prior_params (dict): Dictionary with information about the prior

       Returns:
           float: CDF(x)
    """   
    if prior_params['prior_type'] == 'flat':
        return (  (x > prior_params['max']) * 1.0   
                + (x > prior_params['min']) * (x < prior_params['max']) * (x - prior_params['min'])/(prior_params['max'] - prior_params['min']))
    elif prior_params['prior_type'] == 'gauss':
        return 0.5 * (1.0 + sps.erf( (x - prior_params['mean'])/(prior_params['std'] * np.sqrt(2.0)) ))
    elif prior_params['prior_type'] == 'exponential':
        return (1.0 - np.exp(-x/prior_params['std'])) * (x >= 0.0)
    elif prior_params['prior_type'] == 'lognormal':
        return (x > prior_params['shift']) * (0.5 * sps.erfc( (prior_params['mu'] - np.log(x - prior_params['shift']))/(np.sqrt(2.0) * prior_params['sigma'])))
    elif prior_params['prior_type'] == 'positive':
        raise Exception('CDF is not defined for prior type "positive"')
    elif prior_params['prior_type'] == 'none':
        raise Exception('CDF is not defined for prior type "none"')
    elif prior_params['prior_type'] == 'fixed':
        raise Exception('CDF is not defined for prior type "fixed"')
    else:
        raise Exception('Prior type ' + prior_params['prior_type'] + ' not recognized!')

def cdf_inverse(x, prior_params):
    """Compute the inverse cumulative distribution function of a given prior at a given point 0 <= x <= 1

       Args:
           x (float): Value at which to compute the inverse CDF 
           prior_params (dict): Dictionary with information about the prior

       Returns:
           float: Inverse CDF at x
    """   
    if prior_params['prior_type'] == 'flat':
        return prior_params['min'] * (1.0 - x) + prior_params['max'] * x
    elif prior_params['prior_type'] == 'gauss':
        return prior_params['mean'] - np.sqrt(2.0) * prior_params['std'] * sps.erfcinv(2.0 * x)
    elif prior_params['prior_type'] == 'exponential':
        return prior_params['std'] * np.log(1.0/(1.0 - x))
    elif prior_params['prior_type'] == 'lognormal':
        return np.exp( prior_params['mu'] - np.sqrt(2.0) * prior_params['sigma'] * sps.erfcinv(2.0 * x)) + prior_params['shift']
    elif prior_params['prior_type'] == 'positive':
        raise Exception('CDF is not defined for prior type "positive"')
    elif prior_params['prior_type'] == 'none':
        raise Exception('CDF is not defined for prior type "none"')
    elif prior_params['prior_type'] == 'fixed':
        raise Exception('CDF is not defined for prior type "fixed"')
    else:
        raise Exception('Prior type ' + prior_params['prior_type'] + ' not recognized!')

def param_bounds(prior_params):
    """Compute the parameter boundaries associated with a given prior

       Args:
           prior_params (dict): Dictionary with information about the prior

       Returns:
           list: 2-element list specifying the allowed parameter range: [min,max]
    """   
    if prior_params.get('transform','') == 'cdf':
        bounds = [0.0, 1.0]
    elif prior_params['prior_type'] == 'flat':
        bounds = [prior_params['min'],prior_params['max']]
    elif prior_params['prior_type'] == 'gauss':
        bounds = [prior_params['mean'] - prior_params['std'] * BOUNDS_GAUSS_NSIGMA, prior_params['mean'] + prior_params['std'] * BOUNDS_GAUSS_NSIGMA]
    elif prior_params['prior_type'] == 'exponential':
        bounds = [PRIOR_MIN, BOUNDS_EXP_NSIGMA * prior_params['std']]
    elif prior_params['prior_type'] == 'lognormal':
        bounds = [prior_params['shift'], prior_params['shift'] + np.exp(prior_params['mu'] +  BOUNDS_GAUSS_NSIGMA * prior_params['sigma'])]
    elif prior_params['prior_type'] == 'positive':
        bounds = [PRIOR_MIN, BOUNDS_MAX]
    elif prior_params['prior_type'] == 'none':
        bounds = [BOUNDS_MIN,BOUNDS_MAX]
    elif prior_params['prior_type'] == 'fixed':
        bounds = [1.0, 1.0]
    else:
        print('Prior type not recognized!')
        bounds = [BOUNDS_MIN,BOUNDS_MAX]

    return bounds

def prior_func(x, prior_params):
    """Compute the value of a 1-D prior P(x) at a specified value x.

       Args:
           x (float): Value at which to compute the prior
           prior_params (dict): Dictionary with information about the prior

       Returns:
           float: Prior value P(x) 
    """   

    if prior_params['prior_type'] == 'flat':
        return (x >= prior_params['min']) * (x <= prior_params['max']) * 1.0/(prior_params['max'] - prior_params['min']) + PRIOR_MIN
    elif prior_params['prior_type'] == 'gauss':
        return 1./((2.*np.pi)**0.5 * prior_params['std']) * np.exp(-(x - prior_params['mean'])**2/(2.*prior_params['std']**2))
    elif prior_params['prior_type'] == 'exponential':
        return (1./prior_params['std'] * np.exp(-x/prior_params['std'])) * (x >= 0.0) + PRIOR_MIN
    elif prior_params['prior_type'] == 'lognormal':
        return (x > prior_params['shift']) * (
                  1.0/((2.0*np.pi)**0.5 * prior_params['sigma'] * (x - prior_params['shift'])) 
                * np.exp( -(np.log(x - prior_params['shift']) - prior_params['mu'])**2/(2.0 * prior_params['sigma']**2) ) )
    elif prior_params['prior_type'] == 'positive':
        return (x >= 0.0) * 1.0 + PRIOR_MIN
    elif prior_params['prior_type'] == 'none':
        return 1.0
    elif prior_params['prior_type'] == 'fixed':
        return 1.0
    else:
        print('Prior not recognized!')
        return 1.0

def prior_grad_func(x, prior_params):
    """Compute the value of the derivative of a 1-D prior, dP/dx at a specified value x.

       Args:
           x (float): Value at which to compute the prior derivative
           prior_params (dict): Dictionary with information about the prior

       Returns:
           float: Prior derivative value dP/dx(x) 
    """   

    if prior_params['prior_type'] == 'flat':
        return 0.0
    elif prior_params['prior_type'] == 'gauss':
        return -(x - prior_params['mean'])/((2.*np.pi)**0.5 * prior_params['std']**3) * np.exp(-(x - prior_params['mean'])**2/(2.*prior_params['std']**2))
    elif prior_params['prior_type'] == 'exponential':
        return (-1./prior_params['std']**2 * np.exp(-x/prior_params['std'])) * (x >= 0.0)
    elif prior_params['prior_type'] == 'lognormal':
        return (x > prior_params['shift']) * (
                  (prior_params['mu'] - prior_params['sigma']**2 - np.log(x - prior_params['shift']))
                / ((2.0*np.pi)**0.5 * prior_params['sigma']**3 * (x - prior_params['shift'])**2) 
                * np.exp( -(np.log(x - prior_params['shift']) - prior_params['mu'])**2/(2.0 * prior_params['sigma']**2) ) )
    elif prior_params['prior_type'] == 'positive':
        return 0.0
    elif prior_params['prior_type'] == 'none':
        return 0.0
    elif prior_params['prior_type'] == 'fixed':
        return 0.0
    else:
        print('Prior not recognized!')
        return 0.0

def transform_param(x, x_prior, inverse=True):
    """Compute a specified coordinate transformation T(x) of a parameter value x

       Args:
           x (float): Untransformed value
           x_prior (dict): Dictionary with information about the transformation
           inverse (bool): Whether to compute the forward or inverse transform. 

       Returns:
           float: Transformed parameter value
    """  

    try:
        transform = x_prior['transform']
    except:
        transform = 'none'
        pass

    if transform == 'log':
        if inverse:
            return np.exp(x)
        else:
            return np.log(x)
    elif transform == 'cdf':
        if inverse:
            return cdf_inverse(x, x_prior)
        else:
            return cdf(x, x_prior)
    else:
        return x

def transform_grad_param(x, x_prior):
    """Compute the gradient of a specified coordinate transformation T(x) of a parameter value x

       Args:
           x (float): Untransformed value
           x_prior (dict): Dictionary with information about the transformation

       Returns:
           float: Gradient of transformation, dT/dx(x)
    """  

    try:
        transform = x_prior['transform']
    except:
        transform = 'none'
        pass

    if transform == 'log':
        return np.exp(x)
    elif transform == 'cdf':
        return 1.0/prior_func(transform_param(x,x_prior),x_prior)
    else:
        return 1.0        

##################################################################################################
# Helper functions
##################################################################################################
def shrink_prior(prior, model, shrink=0.1):
    """Shrink a specified prior volume by centering on a specified fitted model

       Args:
           prior (list): Model prior (list of dictionaries, one per model component)
           model (Model): Model to draw central values from
           shrink (float): Factor to shrink each prior width by

       Returns:
           prior (list): Model prior with restricted volume
    """ 

    prior_shrunk = copy.deepcopy(prior)
    f = 1.0

    #TODO: this doesn't work for beta lists yet!

    for j in range(len(prior_shrunk)):
        for key in prior_shrunk[j].keys():
            if prior_shrunk[j][key]['prior_type'] == 'flat':
                x = model.params[j][key]
                w = prior_shrunk[j][key]['max'] - prior_shrunk[j][key]['min']
                prior_shrunk[j][key]['min'] = x - w/2
                prior_shrunk[j][key]['max'] = x + w/2
                if prior_shrunk[j][key]['min'] < prior[j][key]['min']: prior_shrunk[j][key]['min'] = prior[j][key]['min']
                if prior_shrunk[j][key]['max'] > prior[j][key]['max']: prior_shrunk[j][key]['max'] = prior[j][key]['max']
                f *= (prior_shrunk[j][key]['max'] - prior_shrunk[j][key]['min'])/w
            else:
                pass

    print('(New Prior Volume)/(Original Prior Volume:',f)

    return prior_shrunk

def selfcal(Obsdata, model,
            gain_init=None, gain_prior=None,
            minimizer_func='scipy.optimize.minimize', minimizer_kwargs=None,
            bounds=None, use_bounds=True,
            processes=-1, msgtype='bar', quiet=True, **kwargs):
    """Self-calibrate a specified observation to a given model, accounting for gain priors

       Args:

       Returns:
    """  

    # This is just a convenience function. It will call modeler_func() scan-by-scan fitting only gains.
    # This function differs from ehtim.calibrating.self_cal in the inclusion of gain priors
    tlist = Obsdata.tlist()
    res_list = []
    for j in range(len(tlist)):
        if msgtype not in ['none','']:
            prog_msg(j, len(tlist), msgtype, j-1)
        obs = Obsdata.copy()
        obs.data = tlist[j]
        res_list.append(modeler_func(obs, model, model_prior=None, d1='amp', 
                   fit_model=False, fit_gains=True,gain_init=gain_init,gain_prior=gain_prior,
                   minimizer_func=minimizer_func, minimizer_kwargs=minimizer_kwargs,
                   bounds=bounds, use_bounds=use_bounds, processes=-1, quiet=quiet, **kwargs))

    # Assemble a single caltable to return
    allsites = Obsdata.tarr['site']
    caldict = res_list[0]['caltable'].data
    for j in range(1,len(tlist)):
        row = res_list[j]['caltable'].data
        for site in allsites:
            try: dat = row[site]
            except KeyError: continue

            try: caldict[site] = np.append(caldict[site], row[site])
            except KeyError: caldict[site] = dat

    ct = caltable.Caltable(obs.ra, obs.dec, obs.rf, obs.bw, caldict, obs.tarr,
                                       source=obs.source, mjd=obs.mjd, timetype=obs.timetype)

    return ct

def make_param_map(model_init, model_prior, minimizer_func, fit_model, fit_pol=False, fit_cpol=False):
    # Define the mapping between solved parameters and the model
    # Each fitted model parameter can be rescaled to give values closer to order unity
    param_map = []  # Define mapping for every fitted parameter: model component #, parameter name, rescale multiplier internal, unit, rescale multiplier external
    param_mask = [] # True or False for whether to fit each model parameter (because the gradient is computed for all model parameters)
    for j in range(model_init.N_models()):
        params = model.model_params(model_init.models[j],model_init.params[j], fit_pol=fit_pol, fit_cpol=fit_cpol)
        for param in params:
            if fit_model == False:
                param_mask.append(False)        
            elif model_prior[j][param]['prior_type'] != 'fixed':
                param_mask.append(True)
                param_type = param
                if len(param_type.split('_')) == 2 and param_type not in PARAM_DETAILS:
                    param_type = param_type.split('_')[1]
                try:
                    if model_prior[j][param].get('transform','') == 'cdf' or minimizer_func in ['dynesty_static','dynesty_dynamic','pymc3']:
                        param_map.append([j,param,1,PARAM_DETAILS[param_type][1],PARAM_DETAILS[param_type][0]])
                    else:
                        param_map.append([j,param,PARAM_DETAILS[param_type][0],PARAM_DETAILS[param_type][1],PARAM_DETAILS[param_type][0]])
                except:
                    param_map.append([j,param,1,'',1])
                    pass
            else:
                param_mask.append(False)
    return (param_map, param_mask)

def compute_likelihood_constants(d1, d2, d3, sigma1, sigma2, sigma3):
    # Compute the correct data weights (hyperparameters) and the correct extra constant for the log-likelihood
    alpha_d1 = alpha_d2 = alpha_d3 = ln_norm1 = ln_norm2 = ln_norm3 = 0.0

    try: 
        alpha_d1 = 0.5 * len(sigma1)
        ln_norm1 = -np.sum(np.log((2.0*np.pi)**0.5 * sigma1))
    except: pass
    try: 
        alpha_d2 = 0.5 * len(sigma2)
        ln_norm2 = -np.sum(np.log((2.0*np.pi)**0.5 * sigma2))
    except: pass
    try: 
        alpha_d3 = 0.5 * len(sigma3)        
        ln_norm3 = -np.sum(np.log((2.0*np.pi)**0.5 * sigma3))
    except: pass

    # If using closure phase, the sigma is given in degrees, not radians!
    # Use the correct von Mises normalization if using closure phase
    if d1 in ['cphase','cphase_diag']:
        ln_norm1 = -np.sum(np.log(2.0*np.pi*sps.ive(0, 1.0/(sigma1 * DEGREE)**2)))
    if d2 in ['cphase','cphase_diag']:
        ln_norm2 = -np.sum(np.log(2.0*np.pi*sps.ive(0, 1.0/(sigma2 * DEGREE)**2)))
    if d3 in ['cphase','cphase_diag']:
        ln_norm3 = -np.sum(np.log(2.0*np.pi*sps.ive(0, 1.0/(sigma3 * DEGREE)**2)))

    if d1 in ['vis','bs','m','pvis','rrll','llrr','lrll','rlll','lrrr','rlrr','polclosure']:
        alpha_d1 *= 2
        ln_norm1 *= 2
    if d2 in ['vis','bs','m','pvis','rrll','llrr','lrll','rlll','lrrr','rlrr','polclosure']:
        alpha_d2 *= 2
        ln_norm2 *= 2
    if d3 in ['vis','bs','m','pvis','rrll','llrr','lrll','rlll','lrrr','rlrr','polclosure']:
        alpha_d3 *= 2
        ln_norm3 *= 2
    ln_norm = ln_norm1 + ln_norm2 + ln_norm3

    return (alpha_d1, alpha_d2, alpha_d3, ln_norm)

def default_gain_prior(sites):
    print('No gain prior specified. Defaulting to ' + str(GAIN_PRIOR_DEFAULT) + ' for all sites.')
    gain_prior = {}
    for site in sites:
        gain_prior[site] = GAIN_PRIOR_DEFAULT
    return gain_prior

def caltable_to_gains(caltab, gain_list):
    # Generate an ordered list of gains from a caltable
    # gain_list is a set of tuples (time, site)
    gains = [np.abs(caltab.data[site]['rscale'][caltab.data[site]['time'] == time][0]) - 1.0 for (time, site) in gain_list]
    return gains

def make_gain_map(Obsdata, gain_prior): 
    # gain_list gives all unique (time,site) pairs
    # gains_t1 gives the gain index for the first site in each measurement
    # gains_t2 gives the gain index for the second site in each measurement
    gain_list = []
    for j in range(len(Obsdata.data)):
        if ([Obsdata.data[j]['time'],Obsdata.data[j]['t1']] not in gain_list) and (gain_prior[Obsdata.data[j]['t1']]['prior_type'] != 'fixed'):
            gain_list.append([Obsdata.data[j]['time'],Obsdata.data[j]['t1']])
        if ([Obsdata.data[j]['time'],Obsdata.data[j]['t2']] not in gain_list) and (gain_prior[Obsdata.data[j]['t2']]['prior_type'] != 'fixed'):
            gain_list.append([Obsdata.data[j]['time'],Obsdata.data[j]['t2']])

    # Now determine the appropriate mapping; use the final index for all ignored gains, which default to 1
    def gain_index(j, tnum):
        try:
            return gain_list.index([Obsdata.data[j]['time'],Obsdata.data[j][tnum]])
        except:
            return len(gain_list)

    gains_t1 = [gain_index(j, 't1') for j in range(len(Obsdata.data))]
    gains_t2 = [gain_index(j, 't2') for j in range(len(Obsdata.data))]        

    return (gain_list, gains_t1, gains_t2)

def make_bounds(model_prior, param_map, gain_prior, gain_list, n_gains, leakage_fit, leakage_prior):
    bounds = []
    for j in range(len(param_map)):
        pm = param_map[j]
        pb = param_bounds(model_prior[pm[0]][pm[1]])
        if (model_prior[pm[0]][pm[1]]['prior_type'] not in ['positive','none','fixed']) and (model_prior[pm[0]][pm[1]].get('transform','') != 'cdf'):
            pb[0] = transform_param(pb[0]/pm[2], model_prior[pm[0]][pm[1]], inverse=False)
            pb[1] = transform_param(pb[1]/pm[2], model_prior[pm[0]][pm[1]], inverse=False)
        bounds.append(pb)
    for j in range(n_gains):
        pb = param_bounds(gain_prior[gain_list[j][1]])
        if (gain_prior[gain_list[j][1]]['prior_type'] not in ['positive','none','fixed']) and (gain_prior[gain_list[j][1]].get('transform','') != 'cdf'):
            pb[0] = transform_param(pb[0], gain_prior[gain_list[j][1]], inverse=False)
            pb[1] = transform_param(pb[1], gain_prior[gain_list[j][1]], inverse=False)
        bounds.append(pb)
    for j in range(len(leakage_fit)):
        for cpart in ['re','im']:
            prior = leakage_prior[leakage_fit[j][0]][leakage_fit[j][1]][cpart]
            pb = param_bounds(prior)
            if (prior['prior_type'] not in ['positive','none','fixed']) and (prior.get('transform','') != 'cdf'):
                pb[0] = transform_param(pb[0], prior, inverse=False)
                pb[1] = transform_param(pb[1], prior, inverse=False)
            bounds.append(pb)

    return np.array(bounds)

# Determine multiplicative factor for the gains (amplitude only)
def gain_factor(dtype,gains,gains_t1,gains_t2, fit_or_marginalize_gains):
    global globdict

    if not fit_or_marginalize_gains:
        if globdict['gain_init'] == None:
            return 1
        else:
            gains = globdict['gain_init']

    if globdict['marginalize_gains']:
        gains = globdict['gain_init']

    if dtype in ['amp','vis']:    
        gains_wzero = np.append(gains,0.0)
        return (1.0 + gains_wzero[gains_t1])*(1.0 + gains_wzero[gains_t2])
    else:
        return 1

def gain_factor_separate(dtype,gains,gains_t1,gains_t2, fit_or_marginalize_gains):
    # Determine the pair of multiplicative factors for the gains (amplitude only)
    # Note: these are not displaced by unity!
    global globdict

    if not fit_or_marginalize_gains:
        if globdict['gain_init'] == None:
            return (0., 0.)
        else:
            gains = globdict['gain_init']

    if globdict['marginalize_gains']:
        gains = globdict['gain_init']

    if dtype in ['amp','vis']:    
        gains_wzero = np.append(gains,0.0)
        return (gains_wzero[gains_t1], gains_wzero[gains_t2])
    else:
        return (0, 0)

def prior_leakage(leakage, leakage_fit, leakage_prior, fit_leakage):
    # Compute the log-prior contribution from the fitted leakage terms
    if fit_leakage:
        cparts = ['re','im']
        return np.sum([np.log(prior_func(leakage[j], leakage_prior[leakage_fit[j//2][0]][leakage_fit[j//2][1]][cparts[j%2]])) for j in range(len(leakage))])
    else:
        return 0.0

def prior_leakage_grad(leakage, leakage_fit, leakage_prior, fit_leakage):
    # Compute the log-prior contribution to the gradient from the leakages
    if fit_leakage:
        cparts = ['re','im']
        f  = np.array([prior_func(leakage[j], leakage_prior[leakage_fit[j//2][0]][leakage_fit[j//2][1]][cparts[j%2]]) for j in range(len(leakage))])
        df = np.array([prior_grad_func(leakage[j], leakage_prior[leakage_fit[j//2][0]][leakage_fit[j//2][1]][cparts[j%2]]) for j in range(len(leakage))])
        return df/f
    else:
        return []

def prior_gain(gains, gain_list, gain_prior, fit_gains):
    # Compute the log-prior contribution from the gains
    if fit_gains:
        return np.sum([np.log(prior_func(gains[j], gain_prior[gain_list[j][1]])) for j in range(len(gains))])
    else:
        return 0.0

def prior_gain_grad(gains, gain_list, gain_prior, fit_gains):
    # Compute the log-prior contribution to the gradient from the gains
    if fit_gains:
        f  = np.array([prior_func(gains[j], gain_prior[gain_list[j][1]]) for j in range(len(gains))])
        df = np.array([prior_grad_func(gains[j], gain_prior[gain_list[j][1]]) for j in range(len(gains))])
        return df/f
    else:
        return []

def transform_params(params, param_map, minimizer_func, model_prior, inverse=True):
    if minimizer_func not in ['dynesty_static','dynesty_dynamic','pymc3']:
        return [transform_param(params[j], model_prior[param_map[j][0]][param_map[j][1]], inverse=inverse) for j in range(len(params))]
    else:
        # For dynesty or pymc3, over-ride all specified parameter transformations to assume CDF mapping to the hypercube
        # However, the passed parameters to the objective function and gradient are *not* transformed (i.e., they are not in the hypercube), thus the transformation does not need to be inverted
        return params

def set_params(params, trial_model, param_map, minimizer_func, model_prior):
    tparams = transform_params(params, param_map, minimizer_func, model_prior)

    for j in range(len(params)):
        if param_map[j][1] in trial_model.params[param_map[j][0]].keys():
            trial_model.params[param_map[j][0]][param_map[j][1]] = tparams[j] * param_map[j][2]
        else: # In this case, the parameter is a list of complex numbers, so the real/imaginary or abs/arg components need to be assigned
            if param_map[j][1].find('cpol') != -1:
                param_type = 'beta_list_cpol' 
                idx = int(param_map[j][1].split('_')[0][8:]) 
            elif param_map[j][1].find('pol') != -1:
                param_type = 'beta_list_pol'                
                idx = int(param_map[j][1].split('_')[0][7:]) + (len(trial_model.params[param_map[j][0]][param_type])-1)//2 
            elif param_map[j][1].find('beta') != -1:
                param_type = 'beta_list' 
                idx = int(param_map[j][1].split('_')[0][4:]) - 1
            else:
                raise Exception('Unsure how to interpret ' + param_map[j][1])

            curval = trial_model.params[param_map[j][0]][param_type][idx]
            if '_' not in param_map[j][1]: # This is for beta0 of cpol
                trial_model.params[param_map[j][0]][param_type][idx] = tparams[j] * param_map[j][2]
            elif param_map[j][1][-2:] == 're':
                trial_model.params[param_map[j][0]][param_type][idx] = tparams[j] * param_map[j][2]      + np.imag(curval)*1j
            elif param_map[j][1][-2:] == 'im':
                trial_model.params[param_map[j][0]][param_type][idx] = tparams[j] * param_map[j][2] * 1j + np.real(curval)
            elif param_map[j][1][-3:] == 'abs':
                trial_model.params[param_map[j][0]][param_type][idx] = tparams[j] * param_map[j][2] * np.exp(1j * np.angle(curval))
            elif param_map[j][1][-3:] == 'arg': 
                trial_model.params[param_map[j][0]][param_type][idx] = np.abs(curval) * np.exp(1j * tparams[j] * param_map[j][2])
            else:
                print('Parameter ' + param_map[j][1] + ' not understood!')

# Define prior
def prior(params, param_map, model_prior, minimizer_func):
    tparams = transform_params(params, param_map, minimizer_func, model_prior)
    return np.sum([np.log(prior_func(tparams[j]*param_map[j][2], model_prior[param_map[j][0]][param_map[j][1]])) for j in range(len(params))])

def prior_grad(params, param_map, model_prior, minimizer_func):
    tparams = transform_params(params, param_map, minimizer_func, model_prior)        
    f  = np.array([prior_func(tparams[j]*param_map[j][2], model_prior[param_map[j][0]][param_map[j][1]]) for j in range(len(params))])
    df = np.array([prior_grad_func(tparams[j]*param_map[j][2], model_prior[param_map[j][0]][param_map[j][1]]) for j in range(len(params))])
    return df/f

# Define constraint functions
def flux_constraint(trial_model, alpha_flux, flux):
    if alpha_flux == 0.0: 
        return 0.0

    return ((trial_model.total_flux() - flux)/flux)**2

def flux_constraint_grad(trial_model, alpha_flux, flux, params, param_map):
    if alpha_flux == 0.0: 
        return 0.0

    fluxmask = np.zeros_like(params)
    for j in range(len(param_map)):
        if param_map[j][1] == 'F0':
            fluxmask[j] = 1.0

    return 2.0 * (trial_model.total_flux() - flux)/flux * fluxmask

##################################################################################################
# Define the chi^2 and chi^2 gradient functions
##################################################################################################
def laplace_approximation(trial_model, dtype, data, uv, sigma, gains_t1, gains_t2):
    # Compute the approximate contribution to the log-likelihood by marginalizing over gains
    global globdict

    if globdict['marginalize_gains'] == True and dtype == 'amp':
        # Add the log-likelihood term from analytic gain marginalization
        # Create the Hessian matrix for the argument of the exponential
        gain_hess = np.zeros((len(globdict['gain_list']), len(globdict['gain_list'])))

        # Add the terms from the likelihood
        gain = gain_factor(dtype,None,gains_t1,gains_t2,True)
        amp_model = np.abs(trial_model.sample_uv(uv[:,0],uv[:,1])) # TODO: Add polarization!
        amp_bar = gain*data
        sigma_bar = gain*sigma
        (g1, g2) = gain_factor_separate(dtype,None,gains_t1,gains_t2,True)

        # Each amplitude *measurement* (not fitted gain parameter!) contributes to the hessian in four places; two diagonal and two off-diagonal
        for j in range(len(gain)):
            gain_hess[gains_t1[j],gains_t1[j]] += amp_model[j] * (3.0 * amp_model[j] - 2.0 * amp_bar[j])/((1.0 + g1[j])**2 * sigma_bar[j]**2)
            gain_hess[gains_t2[j],gains_t2[j]] += amp_model[j] * (3.0 * amp_model[j] - 2.0 * amp_bar[j])/((1.0 + g2[j])**2 * sigma_bar[j]**2)
            gain_hess[gains_t1[j],gains_t2[j]] += amp_model[j] * (2.0 * amp_model[j] - amp_bar[j])/((1.0 + g1[j])*(1.0 + g2[j]) * sigma_bar[j]**2)
            gain_hess[gains_t2[j],gains_t1[j]] += amp_model[j] * (2.0 * amp_model[j] - amp_bar[j])/((1.0 + g1[j])*(1.0 + g2[j]) * sigma_bar[j]**2)

        # Add contributions from the prior to the diagonal. This ranges over the fitted gain parameters.
        # Note: for the Laplace approximation, only Gaussian gain priors have any effect!
        for j in range(len(globdict['gain_list'])):
            t = globdict['gain_list'][j][1]
            if globdict['gain_prior'][t]['prior_type'] == 'gauss':
                gain_hess[j,j] += 1.0/globdict['gain_prior'][t]['std']
            elif globdict['gain_prior'][t]['prior_type'] == 'flat':
                gain_hess[j,j] += 0.0
            elif globdict['gain_prior'][t]['prior_type'] == 'exponential':
                gain_hess[j,j] += 0.0
            elif globdict['gain_prior'][t]['prior_type'] == 'fixed':                
                gain_hess[j,j] += 0.0
            else:
                raise Exception('Gain prior not implemented!')                
        return np.log((2.0 * np.pi)**(len(gain)/2.0) * np.abs(np.linalg.det(gain_hess))**-0.5)
    else:
        return 0.0

def laplace_list():
    global globdict
    l1 = laplace_approximation(globdict['trial_model'], globdict['d1'], globdict['data1'], globdict['uv1'], globdict['sigma1'], globdict['gains_t1'], globdict['gains_t2'])
    l2 = laplace_approximation(globdict['trial_model'], globdict['d2'], globdict['data2'], globdict['uv2'], globdict['sigma2'], globdict['gains_t1'], globdict['gains_t2'])
    l3 = laplace_approximation(globdict['trial_model'], globdict['d3'], globdict['data3'], globdict['uv3'], globdict['sigma3'], globdict['gains_t1'], globdict['gains_t2'])
    return (l1, l2, l3)

def chisq_wgain(trial_model, dtype, data, uv, sigma, pol, jonesdict, gains, gains_t1, gains_t2, fit_or_marginalize_gains):
    global globdict
    gain = gain_factor(dtype,gains,gains_t1,gains_t2,fit_or_marginalize_gains)
    log_likelihood = chisq(trial_model, uv, gain*data, gain*sigma, dtype, pol, jonesdict)
    return log_likelihood

def chisqgrad_wgain(trial_model, dtype, data, uv, sigma, jonesdict, gains, gains_t1, gains_t2, fit_or_marginalize_gains, param_mask, pol='I', fit_pol=False, fit_cpol=False, fit_leakage=False):
    gain = gain_factor(dtype,gains,gains_t1,gains_t2,fit_or_marginalize_gains)
    return chisqgrad(trial_model, uv, gain*data, gain*sigma, jonesdict, dtype, param_mask, pol, fit_or_marginalize_gains, gains, gains_t1, gains_t2, fit_pol, fit_cpol, fit_leakage)

def chisq_list(gains):
    global globdict
    chi2_1 = chisq_wgain(globdict['trial_model'], globdict['d1'], globdict['data1'], globdict['uv1'], globdict['sigma1'], globdict['pol1'], globdict['jonesdict1'], gains, globdict['gains_t1'], globdict['gains_t2'], globdict['fit_gains'] + globdict['marginalize_gains'])
    chi2_2 = chisq_wgain(globdict['trial_model'], globdict['d2'], globdict['data2'], globdict['uv2'], globdict['sigma2'], globdict['pol2'], globdict['jonesdict2'], gains, globdict['gains_t1'], globdict['gains_t2'], globdict['fit_gains'] + globdict['marginalize_gains'])
    chi2_3 = chisq_wgain(globdict['trial_model'], globdict['d3'], globdict['data3'], globdict['uv3'], globdict['sigma3'], globdict['pol3'], globdict['jonesdict3'], gains, globdict['gains_t1'], globdict['gains_t2'], globdict['fit_gains'] + globdict['marginalize_gains'])
    return (chi2_1, chi2_2, chi2_3)

def update_leakage(leakage):
    # This function updates the 'jonesdict' entries based on current leakage estimates
    # leakage is list of the fitted parameters (re and im are separate)
    # station_leakages is the dictionary containing all station leakages, some of which may be fixed
    global globdict
    if len(leakage) == 0: return

    station_leakages = globdict['station_leakages']
    leakage_fit = globdict['leakage_fit']
    # First, update the entries in the leakage dictionary
    for j in range(len(leakage)//2):
        station_leakages[leakage_fit[j][0]][leakage_fit[j][1]] = leakage[2*j] + 1j * leakage[2*j + 1]

    # Now, recompute the jonesdict objects
    for j in range(1,4):
        jonesdict = globdict['jonesdict' + str(j)]
        if jonesdict is not None:
            if type(jonesdict) is dict:
                jonesdict['DR1'] = np.array([station_leakages[jonesdict['t1'][_]]['R'] for _ in range(len(jonesdict['t1']))])
                jonesdict['DR2'] = np.array([station_leakages[jonesdict['t2'][_]]['R'] for _ in range(len(jonesdict['t1']))])
                jonesdict['DL1'] = np.array([station_leakages[jonesdict['t1'][_]]['L'] for _ in range(len(jonesdict['t1']))])
                jonesdict['DL2'] = np.array([station_leakages[jonesdict['t2'][_]]['L'] for _ in range(len(jonesdict['t1']))])
                jonesdict['leakage_fit'] = globdict['leakage_fit']
            else:
                # In this case, the data product requires a list of jonesdicts
                for jonesdict2 in jonesdict:
                    jonesdict2['DR1'] = np.array([station_leakages[jonesdict2['t1'][_]]['R'] for _ in range(len(jonesdict2['t1']))])
                    jonesdict2['DR2'] = np.array([station_leakages[jonesdict2['t2'][_]]['R'] for _ in range(len(jonesdict2['t1']))])
                    jonesdict2['DL1'] = np.array([station_leakages[jonesdict2['t1'][_]]['L'] for _ in range(len(jonesdict2['t1']))])
                    jonesdict2['DL2'] = np.array([station_leakages[jonesdict2['t2'][_]]['L'] for _ in range(len(jonesdict2['t1']))])
                    jonesdict2['leakage_fit'] = globdict['leakage_fit']
        
##################################################################################################
# Define the objective function and gradient
##################################################################################################
def objfunc(params, force_posterior=False): 
    global globdict
    # Note: model parameters can have transformations applied; gains and leakage do not
    set_params(params[:globdict['n_params']], globdict['trial_model'], globdict['param_map'], globdict['minimizer_func'], globdict['model_prior'])
    gains = params[globdict['n_params']:(globdict['n_params'] + globdict['n_gains'])]
    leakage = params[(globdict['n_params'] + globdict['n_gains']):]
    update_leakage(leakage)

    if globdict['marginalize_gains']:
        # Ugh, the use of global variables totally messes this up
        _globdict = globdict
        # This doesn't handle the passed gain_init properly because the dimensions are incorrect
        _globdict['gain_init'] = caltable_to_gains(selfcal(globdict['Obsdata'], globdict['trial_model'], gain_init=None, gain_prior=globdict['gain_prior'], msgtype='none'),globdict['gain_list'])
        globdict = _globdict

    (chi2_1, chi2_2, chi2_3) = chisq_list(gains)
    datterm  = ( globdict['alpha_d1'] * chi2_1 
               + globdict['alpha_d2'] * chi2_2
               + globdict['alpha_d3'] * chi2_3)

    if globdict['marginalize_gains']:
        (l1, l2, l3) = laplace_list()
        datterm += l1 + l2 + l3

    if (globdict['minimizer_func'] not in ['dynesty_static','dynesty_dynamic','pymc3']) or force_posterior:
        priterm  = prior(params[:globdict['n_params']], globdict['param_map'], globdict['model_prior'], globdict['minimizer_func']) 
        priterm += prior_gain(params[globdict['n_params']:(globdict['n_params'] + globdict['n_gains'])], globdict['gain_list'], globdict['gain_prior'], globdict['fit_gains'])
        priterm += prior_leakage(params[(globdict['n_params'] + globdict['n_gains']):], globdict['leakage_fit'], globdict['leakage_prior'], globdict['fit_leakage'])
    else:
        priterm  = 0.0
    fluxterm = globdict['alpha_flux'] * flux_constraint(globdict['trial_model'], globdict['alpha_flux'], globdict['flux'])

    return datterm - priterm + fluxterm - globdict['ln_norm']

def objgrad(params):
    global globdict
    set_params(params[:globdict['n_params']], globdict['trial_model'], globdict['param_map'], globdict['minimizer_func'], globdict['model_prior'])
    gains = params[globdict['n_params']:(globdict['n_params'] + globdict['n_gains'])]
    leakage = params[(globdict['n_params'] + globdict['n_gains']):]
    update_leakage(leakage)

    datterm  = ( globdict['alpha_d1'] * chisqgrad_wgain(globdict['trial_model'], globdict['d1'], globdict['data1'], globdict['uv1'], globdict['sigma1'], globdict['jonesdict1'], gains, globdict['gains_t1'], globdict['gains_t2'], globdict['fit_gains'] + globdict['marginalize_gains'], globdict['param_mask'], globdict['pol1'], globdict['fit_pol'], globdict['fit_cpol'], globdict['fit_leakage']) 
               + globdict['alpha_d2'] * chisqgrad_wgain(globdict['trial_model'], globdict['d2'], globdict['data2'], globdict['uv2'], globdict['sigma2'], globdict['jonesdict2'], gains, globdict['gains_t1'], globdict['gains_t2'], globdict['fit_gains'] + globdict['marginalize_gains'], globdict['param_mask'], globdict['pol2'], globdict['fit_pol'], globdict['fit_cpol'], globdict['fit_leakage']) 
               + globdict['alpha_d3'] * chisqgrad_wgain(globdict['trial_model'], globdict['d3'], globdict['data3'], globdict['uv3'], globdict['sigma3'], globdict['jonesdict3'], gains, globdict['gains_t1'], globdict['gains_t2'], globdict['fit_gains'] + globdict['marginalize_gains'], globdict['param_mask'], globdict['pol3'], globdict['fit_pol'], globdict['fit_cpol'], globdict['fit_leakage']))

    if globdict['minimizer_func'] not in ['dynesty_static','dynesty_dynamic','pymc3']:
        priterm  = np.concatenate([prior_grad(params[:globdict['n_params']], globdict['param_map'], 
                                              globdict['model_prior'], globdict['minimizer_func']), 
                                  prior_gain_grad(params[globdict['n_params']:(globdict['n_params'] + globdict['n_gains'])], 
                                              globdict['gain_list'], globdict['gain_prior'], globdict['fit_gains']),
                                  prior_leakage_grad(params[(globdict['n_params'] + globdict['n_gains']):], globdict['leakage_fit'], 
                                              globdict['leakage_prior'], globdict['fit_leakage'])])
    else:
        priterm  = 0.0
    fluxterm = globdict['alpha_flux'] * flux_constraint_grad(params, globdict['alpha_flux'], globdict['flux'], params, globdict['param_map'])

    grad = datterm - priterm + fluxterm

    if globdict['minimizer_func'] not in ['dynesty_static','dynesty_dynamic','pymc3']:
        for j in range(globdict['n_params']):
            grad[j] *= globdict['param_map'][j][2] * transform_grad_param(params[j], globdict['model_prior'][globdict['param_map'][j][0]][globdict['param_map'][j][1]])
    else:
        # For dynesty or pymc3, over-ride all specified parameter transformations to assume CDF
        # However, the passed parameters are *not* transformed (i.e., they are not in the hypercube)
        # The Jacobian still needs to account for the parameter transformation
        for j in range(len(params)):
            if j < globdict['n_params']:
                j2 = j
                x  = params[j2]
                prior_params = globdict['model_prior'][globdict['param_map'][j2][0]][globdict['param_map'][j2][1]]
                grad[j] /= prior_func(x,prior_params)
            elif j < globdict['n_params'] + globdict['n_gains']:
                j2 = j-globdict['n_params']
                x = gains[j2]
                prior_params = globdict['gain_prior'][globdict['gain_list'][j2][1]]
                grad[j] /= prior_func(x, prior_params)                    
            else:
                cparts = ['re','im']
                j2 = j-globdict['n_params']-globdict['n_gains']
                x = leakage[j2]
                prior_params = globdict['leakage_prior'][globdict['leakage_fit'][j2//2][0]][globdict['leakage_fit'][j2//2][1]][cparts[j2%2]]
                grad[j] /= prior_func(x, prior_params)                          

    if globdict['test_gradient']:
        print('Testing the gradient at ',params)
        import copy
        dx = 1e-5
        grad_numeric = np.zeros(len(grad))
        f1 = objfunc(params)
        print('Objective Function:',f1)
        print('\nNumeric Gradient Check: Analytic Numeric')
        for j in range(len(grad)):
            if globdict['minimizer_func'] in ['dynesty_static','dynesty_dynamic','pymc3']:
                dx = np.abs(params[j]) * 1e-6

            params2 = copy.deepcopy(params)
            params2[j] += dx                
            f2 = objfunc(params2)
            grad_numeric[j] = (f2 - f1)/dx

            if globdict['minimizer_func'] in ['dynesty_static','dynesty_dynamic','pymc3']:
                if j < globdict['n_params']:
                    j2 = j
                    x  = params[j2]
                    prior_params = globdict['model_prior'][globdict['param_map'][j2][0]][globdict['param_map'][j2][1]]
                    grad_numeric[j] /= prior_func(x,prior_params)
                elif j < globdict['n_params'] + globdict['n_gains']:
                    j2 = j-globdict['n_params']
                    x = gains[j2]
                    prior_params = globdict['gain_prior'][globdict['gain_list'][j2][1]]
                    grad_numeric[j] /= prior_func(x, prior_params)                    
                else:
                    cparts = ['re','im']
                    j2 = j-globdict['n_params']-globdict['n_gains']
                    x = leakage[j2]
                    prior_params = globdict['leakage_prior'][globdict['leakage_fit'][j2//2][0]][globdict['leakage_fit'][j2//2][1]][cparts[j2%2]]
                    grad_numeric[j] /= prior_func(x, prior_params)    

            if j < globdict['n_params']:
                print('\nNumeric Gradient Check:',globdict['param_map'][j][0],globdict['param_map'][j][1],grad[j],grad_numeric[j])
            else:
                print('\nNumeric Gradient Check:',grad[j],grad_numeric[j])

    return grad

##################################################################################################
# Modeler
##################################################################################################
def modeler_func(Obsdata, model_init, model_prior,
                   d1='vis', d2=False, d3=False,
                   pol1='I', pol2='I', pol3='I',
                   normchisq = False, alpha_d1=0, alpha_d2=0, alpha_d3=0,                   
                   flux=1.0, alpha_flux=0,
                   fit_model=True, fit_pol=False, fit_cpol=False, 
                   fit_gains=False,marginalize_gains=False,gain_init=None,gain_prior=None,
                   fit_leakage=False, leakage_init=None, leakage_prior=None,
                   fit_noise_model=False,
                   minimizer_func='scipy.optimize.minimize',
                   minimizer_kwargs=None,
                   bounds=None, use_bounds=False,
                   processes=-1,
                   test_gradient=False, quiet=False, **kwargs):

    """Fit a specified model. 

       Args:
           Obsdata (Obsdata): The Obsdata object with VLBI data
           model_init (Model): The Model object to fit
           model_prior (dict): Priors for each model parameter

           d1 (str): The first data term; options are 'vis', 'bs', 'amp', 'cphase', 'cphase_diag' 'camp', 'logcamp', 'logcamp_diag', 'm'
           d2 (str): The second data term; options are 'vis', 'bs', 'amp', 'cphase', 'cphase_diag' 'camp', 'logcamp', 'logcamp_diag', 'm'
           d3 (str): The third data term; options are 'vis', 'bs', 'amp', 'cphase', 'cphase_diag' 'camp', 'logcamp', 'logcamp_diag', 'm'

           normchisq (bool): If False (default), automatically assign weights alpha_d1-3 to match the true log-likelihood. 
           alpha_d1 (float): The first data term weighting. 
           alpha_d2 (float): The second data term weighting. Default value of zero will automatically assign weights to match the true log-likelihood.
           alpha_d2 (float): The third data term weighting. Default value of zero will automatically assign weights to match the true log-likelihood.

           flux (float): Total flux of the fitted model
           alpha_flux (float): Hyperparameter controlling how strongly to constrain that the total flux matches the specified flux.

           fit_model (bool): Whether or not to fit the model parameters
           fit_pol (bool): Whether or not to fit linear polarization parameters
           fit_cpol (bool): Whether or not to fit circular polarization parameters
           fit_gains (bool): Whether or not to fit time-dependent amplitude gains for each station
           marginalize_gains (bool): Whether or not to perform analytic gain marginalization (via the Laplace approximation to the posterior)
    
           gain_init (list or caltable): Initial gain amplitudes to apply; these can be specified even if gains aren't fitted
           gain_prior (dict): Dictionary with the gain prior for each site. 

           minimizer_func (str): Minimizer function to use. Current options are:
                                'scipy.optimize.minimize'
                                'scipy.optimize.dual_annealing'
                                'scipy.optimize.basinhopping'
                                'dynesty_static'
                                'dynesty_dynamic'
                                'pymc3'
           minimizer_kwargs (dict): kwargs passed to the minimizer. 

           bounds (list): List of parameter bounds for the fitted parameters (will automatically compute if needed)
           use_bounds (bool): Whether or not to use bounds when fitting (required for some minimizers)

           processes (int): Number of processes to use for a multiprocessing pool. -1 disables multiprocessing; 0 uses all that are available. Only used for dynesty.

       Returns:
           dict: Dictionary with fitted model ('model') and other diagnostics that are minimizer-dependent
    """    

    global nit, globdict
    nit = n_params = 0
    ln_norm = 0.0

    if fit_model == False and fit_gains == False and fit_leakage == False:
        raise Exception('Both fit_model, fit_gains, and fit_leakage are False. Must fit something!')

    if fit_gains == True and marginalize_gains == True:
        raise Exception('Both fit_gains and marginalize_gains are True. Cannot do both!')

    if fit_gains == False and marginalize_gains == False and gain_init is not None:
        if not quiet: print('Both fit_gains and marginalize_gains are False but gain_init was passed. Applying these gains as a fixed correction!')

    if minimizer_kwargs is None:
        minimizer_kwargs = {}

    # Specifications for verbosity during fits
    show_updates = kwargs.get('show_updates',True)
    update_interval = kwargs.get('update_interval',1)
    run_nested_kwargs = kwargs.get('run_nested_kwargs',{})

    # Make sure data and regularizer options are ok
    if not d1 and not d2 and not d3:
        raise Exception("Must have at least one data term!")
    if (not ((d1 in DATATERMS) or d1==False)) or (not ((d2 in DATATERMS) or d2==False)):
        raise Exception("Invalid data term: valid data terms are: " + ' '.join(DATATERMS))

    # Create the trial model
    trial_model = model_init.copy()

    # Define mapping for every fitted parameter: model component index, parameter name, rescale multiplier, unit
    (param_map, param_mask) = make_param_map(model_init, model_prior, minimizer_func, fit_model, fit_pol, fit_cpol)

    # Get data and info for the data terms
    if type(Obsdata) is obsdata.Obsdata:
        (data1, sigma1, uv1, jonesdict1) = chisqdata(Obsdata, d1, pol=pol1)
        (data2, sigma2, uv2, jonesdict2) = chisqdata(Obsdata, d2, pol=pol2)
        (data3, sigma3, uv3, jonesdict3) = chisqdata(Obsdata, d3, pol=pol3)
    elif type(Obsdata) is list:
        # Combine a list of observations into one. 
        # Allow these to be from multiple sources for polarimetric zero-baseline purposes.
        # Main thing for different sources is to compute field rotation before combining
        def combine_data(d1,s1,u1,j1,d2,s2,u2,j2):
            d = np.concatenate([d1,d2])
            s = np.concatenate([s1,s2])
            u = np.concatenate([u1,u2])
            j = j1.copy()
            for key in ['fr1', 'fr2', 't1', 't2', 'DR1', 'DR2', 'DL1', 'DL2']:
                j[key] = np.concatenate([j1[key],j2[key]])
            return (d, s, u, j)        
            
        (data1, sigma1, uv1, jonesdict1) = chisqdata(Obsdata[0], d1, pol=pol1)
        (data2, sigma2, uv2, jonesdict2) = chisqdata(Obsdata[0], d2, pol=pol2)
        (data3, sigma3, uv3, jonesdict3) = chisqdata(Obsdata[0], d3, pol=pol3)
        for j in range(1,len(Obsdata)):
            (data1b, sigma1b, uv1b, jonesdict1b) = chisqdata(Obsdata[j], d1, pol=pol1)
            (data2b, sigma2b, uv2b, jonesdict2b) = chisqdata(Obsdata[j], d2, pol=pol2)
            (data3b, sigma3b, uv3b, jonesdict3b) = chisqdata(Obsdata[j], d3, pol=pol3)

            if data1b is not False: 
                (data1, sigma1, uv1, jonesdict1) = combine_data(data1,sigma1,uv1,jonesdict1,data1b,sigma1b,uv1b,jonesdict1b)
            if data2b is not False: 
                (data2, sigma2, uv2, jonesdict2) = combine_data(data2,sigma2,uv2,jonesdict2,data2b,sigma2b,uv2b,jonesdict2b)
            if data3b is not False: 
                (data3, sigma3, uv3, jonesdict3) = combine_data(data3,sigma3,uv3,jonesdict3,data3b,sigma3b,uv3b,jonesdict3b)

        alldata = np.concatenate([_.data for _ in Obsdata])
        Obsdata = Obsdata[0]
        Obsdata.data = alldata
    else:
        raise Exception("Observation format not recognized!")

    if fit_leakage or leakage_init is not None:
        # Determine what leakage terms must be fitted. At most, this would be L & R complex leakages terms for every site
        # leakage_fit is a list of tuples [site, hand] that will be fitted
        leakage_fit = []
        if fit_leakage:
            import copy # Error on the next line if this isn't done again. Why python, why?!?
            # Start with the list of all sites
            sites = list(set(np.concatenate(Obsdata.unpack(['t1','t2']).tolist())))

            # Add missing entries to leakage_prior
            # leakage_prior is a nested dictionary with keys of station, hand, re/im
            leakage_prior_init = copy.deepcopy(leakage_prior)
            if leakage_prior_init is None: leakage_prior_init = {}                
            leakage_prior = {}
            for s in sites:
                leakage_prior[s] = {}
                for pol in ['R','L']:
                    leakage_prior[s][pol] = {}
                    for cpart in ['re','im']:
                        # check to see if a prior is specified for the complex part, the pol, or the site (in that order)                        
                        if leakage_prior_init.get(s,{}).get(pol,{}).get(cpart,{}).get('prior_type','') != '':
                            leakage_prior[s][pol][cpart] = leakage_prior_init[s][pol][cpart]
                        elif leakage_prior_init.get(s,{}).get(pol,{}).get('prior_type','') != '':
                            leakage_prior[s][pol][cpart] = copy.deepcopy(leakage_prior_init[s][pol])
                        elif leakage_prior_init.get(s,{}).get('prior_type','') != '':
                            leakage_prior[s][pol][cpart] = copy.deepcopy(leakage_prior_init[s])
                        else:
                            leakage_prior[s][pol][cpart] = copy.deepcopy(LEAKAGE_PRIOR_DEFAULT)

            if Obsdata.polrep == 'stokes':                
                for s in sites:
                    for pol in ['R','L']:
                        if leakage_prior[s][pol]['re']['prior_type'] == 'fixed': continue                            
                        leakage_fit.append([s,pol])
            else:
                vislist = Obsdata.unpack(['t1','t2','rlvis','lrvis'])
                # Only fit leakage for sites that include cross hand visibilities
                DR = list(set(np.concatenate([vislist[~np.isnan(vislist['rlvis'])]['t1'], vislist[~np.isnan(vislist['lrvis'])]['t2']])))
                DL = list(set(np.concatenate([vislist[~np.isnan(vislist['lrvis'])]['t1'], vislist[~np.isnan(vislist['rlvis'])]['t2']])))
                [leakage_fit.append([s,'R']) for s in DR if leakage_prior[s]['R']['re']['prior_type'] != 'fixed']
                [leakage_fit.append([s,'L']) for s in DL if leakage_prior[s]['L']['re']['prior_type'] != 'fixed']
                sites = list(set(np.concatenate([DR,DL])))
            
            if type(leakage_init) is dict:
                station_leakages = copy.deepcopy(leakage_init)
            else:
                station_leakages = {}

            # Add missing entries to station_leakages
            for s in sites:
                for pol in ['R','L']:
                    if s not in station_leakages.keys():
                        station_leakages[s] = {}
                    if 'R' not in station_leakages[s].keys():
                        station_leakages[s]['R'] = 0.0
                    if 'L' not in station_leakages[s].keys():
                        station_leakages[s]['L'] = 0.0        
    else:
        # Disable leakage computations
        jonesdict1 = jonesdict2 = jonesdict3 = None
        leakage_fit = []
        station_leakages = None

    if normchisq == False:
        if not quiet: print('Assigning data weights to give the correct log-likelihood...')
        (alpha_d1, alpha_d2, alpha_d3, ln_norm) = compute_likelihood_constants(d1, d2, d3, sigma1, sigma2, sigma3)
    else:
        ln_norm = 0.0

    # Determine the mapping between solution gains and the input visibilities
    # Use passed gains even if fit_gains=False and marginalize_gains=False
    # NOTE: THERE IS A PROBLEM IN THIS IMPLEMENTATION. A fixed gain prior is ignored. However, gain_init may still want to apply a constant correction, especially when passing a caltable.
    # We should maybe have two gain lists: one for constant gains and one for fitted gains
    mean_g1 = mean_g2 = 0.0
    if fit_gains or marginalize_gains:
        if gain_prior is None:
            gain_prior = default_gain_prior(Obsdata.tarr['site'])
        (gain_list, gains_t1, gains_t2) = make_gain_map(Obsdata, gain_prior)
        if type(gain_init) == caltable.Caltable:
            if not quiet: print('Converting gain_init from caltable to a list')
            gain_init = caltable_to_gains(gain_init, gain_list)
        if gain_init is None:
            if not quiet: print('Initializing all gain corrections to be zero')
            gain_init = np.zeros(len(gain_list))
        else:
            if len(gain_init) != len(gain_list):
                raise Exception('Gain initialization has incorrect dimensions! %d %d' % (len(gain_init), len(gain_list)))
        if fit_gains:
            n_gains = len(gain_list)
        elif marginalize_gains:
            n_gains = 0
    else:
        if gain_init is None:
            n_gains = 0
            gain_list = []
            gains_t1 = gains_t2 = None
        else:
            if gain_prior is None:
                gain_prior = default_gain_prior(Obsdata.tarr['site'])
            (gain_list, gains_t1, gains_t2) = make_gain_map(Obsdata, gain_prior)
            if type(gain_init) == caltable.Caltable:
                if not quiet: print('Converting gain_init from caltable to a list')
                gain_init = caltable_to_gains(gain_init, gain_list)

    if fit_leakage:
        leakage_init = np.zeros(len(leakage_fit) * 2)  
        for j in range(len(leakage_init)//2):            
            leakage_init[2*j] = np.real(station_leakages[leakage_fit[j][0]][leakage_fit[j][1]])
            leakage_init[2*j + 1] = np.imag(station_leakages[leakage_fit[j][0]][leakage_fit[j][1]])
    else:
        leakage_init = []

    # Initial parameters
    param_init = []
    for j in range(len(param_map)):
        pm = param_map[j]
        if param_map[j][1] in trial_model.params[param_map[j][0]].keys():
            param_init.append(transform_param(model_init.params[pm[0]][pm[1]]/pm[2], model_prior[pm[0]][pm[1]],inverse=False))
        else: # In this case, the parameter is a list of complex numbers, so the real/imaginary or abs/arg components need to be assigned
            if param_map[j][1].find('cpol') != -1:
                param_type = 'beta_list_cpol' 
                idx = int(param_map[j][1].split('_')[0][8:]) 
            elif param_map[j][1].find('pol') != -1:
                param_type = 'beta_list_pol'                
                idx = int(param_map[j][1].split('_')[0][7:]) + (len(trial_model.params[param_map[j][0]][param_type])-1)//2 
            elif param_map[j][1].find('beta') != -1:
                param_type = 'beta_list' 
                idx = int(param_map[j][1].split('_')[0][4:]) - 1
            else:
                raise Exception('Unsure how to interpret ' + param_map[j][1])

            curval = model_init.params[param_map[j][0]][param_type][idx]
            if '_' not in param_map[j][1]:
                param_init.append(transform_param(np.real( model_init.params[pm[0]][param_type][idx]/pm[2]), model_prior[pm[0]][pm[1]],inverse=False))
            elif   param_map[j][1][-2:] == 're':
                param_init.append(transform_param(np.real( model_init.params[pm[0]][param_type][idx]/pm[2]), model_prior[pm[0]][pm[1]],inverse=False))
            elif param_map[j][1][-2:] == 'im':
                param_init.append(transform_param(np.imag( model_init.params[pm[0]][param_type][idx]/pm[2]), model_prior[pm[0]][pm[1]],inverse=False))
            elif param_map[j][1][-3:] == 'abs':
                param_init.append(transform_param(np.abs(  model_init.params[pm[0]][param_type][idx]/pm[2]), model_prior[pm[0]][pm[1]],inverse=False))
            elif param_map[j][1][-3:] == 'arg':
                param_init.append(transform_param(np.angle(model_init.params[pm[0]][param_type][idx])/pm[2], model_prior[pm[0]][pm[1]],inverse=False))
            else:
                if not quiet: print('Parameter ' + param_map[j][1] + ' not understood!')  
    n_params = len(param_init)

    # Note: model parameters can have transformations applied; gains and leakage do not
    if fit_gains: # Do not add these if marginalize_gains == True
        param_init += list(gain_init)
    if fit_leakage:
        param_init += list(leakage_init)
 
    if minimizer_func not in ['dynesty_static','dynesty_dynamic','pymc3']:
        # Define bounds (irrelevant for dynesty or pymc3)
        if use_bounds == False and minimizer_func in ['scipy.optimize.dual_annealing']:
            if not quiet: print('Bounds are required for ' + minimizer_func + '! Setting use_bounds=True.')
            use_bounds = True
        if use_bounds == False and bounds is not None:
            if not quiet: print('Bounds passed but use_bounds=False; setting use_bounds=True.')
            use_bounds = True
        if bounds is None and use_bounds:
            if not quiet: print('No bounds passed. Setting nominal bounds.')
            bounds = make_bounds(model_prior, param_map, gain_prior, gain_list, n_gains, leakage_fit, leakage_prior)
        if use_bounds == False:
            bounds = None

    # Gather global variables into a dictionary 
    globdict = {'trial_model':trial_model, 
                'd1':d1, 'd2':d2, 'd3':d3, 
                'pol1':pol1, 'pol2':pol2, 'pol3':pol3,
                'data1':data1, 'sigma1':sigma1, 'uv1':uv1, 'jonesdict1':jonesdict1,
                'data2':data2, 'sigma2':sigma2, 'uv2':uv2, 'jonesdict2':jonesdict2,
                'data3':data3, 'sigma3':sigma3, 'uv3':uv3, 'jonesdict3':jonesdict3,
                'alpha_d1':alpha_d1, 'alpha_d2':alpha_d2, 'alpha_d3':alpha_d3, 
                'n_params': n_params, 'n_gains':n_gains, 'n_leakage':len(leakage_init),
                'model_prior':model_prior, 'param_map':param_map, 'param_mask':param_mask, 
                'gain_prior':gain_prior, 'gain_list':gain_list, 'gain_init':gain_init,
                'fit_leakage':fit_leakage, 'leakage_init':leakage_init, 'leakage_fit':leakage_fit, 'station_leakages':station_leakages, 'leakage_prior':leakage_prior,
                'show_updates':show_updates, 'update_interval':update_interval, 'gains_t1':gains_t1, 'gains_t2':gains_t2, 
                'minimizer_func':minimizer_func,'Obsdata':Obsdata,
                'fit_pol':fit_pol, 'fit_cpol':fit_cpol,
                'flux':flux, 'alpha_flux':alpha_flux, 'fit_gains':fit_gains, 'marginalize_gains':marginalize_gains, 'ln_norm':ln_norm, 'param_init':param_init, 'test_gradient':test_gradient}    
    if fit_leakage:
        update_leakage(leakage_init)


    # Define the function that reports progress
    def plotcur(params_step, *args):
        global nit, globdict
        if globdict['show_updates'] and (nit % globdict['update_interval'] == 0) and (quiet == False):
            if globdict['n_params'] > 0:
                print('Params:',params_step[:globdict['n_params']])
                print('Transformed Params:',transform_params(params_step[:globdict['n_params']], globdict['param_map'], globdict['minimizer_func'], globdict['model_prior']))
            gains = params_step[globdict['n_params']:(globdict['n_params'] + globdict['n_gains'])]
            leakage = params_step[(globdict['n_params'] + globdict['n_gains']):]
            if len(leakage):
                print('leakage:',leakage)
            update_leakage(leakage)
            (chi2_1, chi2_2, chi2_3) = chisq_list(gains)
            print("i: %d chi2_1: %0.2f chi2_2: %0.2f chi2_3: %0.2f prior: %0.2f" % (nit, chi2_1, chi2_2, chi2_3, prior(params_step[:globdict['n_params']], globdict['param_map'], globdict['model_prior'], globdict['minimizer_func'])))
        nit += 1

    # Print initial statistics
    if not quiet: 
        print("Initial Objective Function: %f" % (objfunc(param_init)))
        if d1 in DATATERMS:
            print("Total Data 1: ", (len(data1)))
        if d2 in DATATERMS:
            print("Total Data 2: ", (len(data2)))
        if d3 in DATATERMS:
            print("Total Data 3: ", (len(data3)))
        print("Total Fitted Real Parameters #: ",(len(param_init)))
        print("Fitted Model Parameters: ",[_[1] for _ in param_map])
        print('Fitting Leakage Terms for:',leakage_fit)
    plotcur(param_init)

    # Run the minimization
    tstart = time.time()
    ret = {}
    if minimizer_func == 'scipy.optimize.minimize':
        min_kwargs = {'method':minimizer_kwargs.get('method','L-BFGS-B'),
                      'options':{'maxiter':MAXIT, 'ftol':STOP, 'maxcor':NHIST,'gtol':STOP,'maxls':MAXLS}}

        if 'options' in minimizer_kwargs.keys():
            for key in minimizer_kwargs['options'].keys():
                min_kwargs['options'][key] = minimizer_kwargs['options'][key]

        for key in minimizer_kwargs.keys():
            if key in ['options','method']:
                continue
            else:
                min_kwargs[key] = minimizer_kwargs[key]

        res = opt.minimize(objfunc, param_init, jac=objgrad, callback=plotcur, bounds=bounds, **min_kwargs)
    elif minimizer_func == 'scipy.optimize.dual_annealing':
        min_kwargs = {}
        min_kwargs['local_search_options'] = {'jac':objgrad, 
                                              'method':'L-BFGS-B','options':{'maxiter':MAXIT, 'ftol':STOP, 'maxcor':NHIST,'gtol':STOP,'maxls':MAXLS}}
        if 'local_search_options' in minimizer_kwargs.keys():
            for key in minimizer_kwargs['local_search_options'].keys():
                min_kwargs['local_search_options'][key] = minimizer_kwargs['local_search_options'][key]

        for key in minimizer_kwargs.keys():
            if key in ['local_search_options']:
                continue
            min_kwargs[key] = minimizer_kwargs[key]
              
        res = opt.dual_annealing(objfunc, x0=param_init, bounds=bounds, callback=plotcur, **min_kwargs)
    elif minimizer_func == 'scipy.optimize.basinhopping':
        min_kwargs = {}
        for key in minimizer_kwargs.keys():
            min_kwargs[key] = minimizer_kwargs[key]

        res = opt.basinhopping(objfunc, param_init, **min_kwargs)  
    elif minimizer_func == 'pymc3':
        ########################
        ## Sample using pymc3 ##
        ########################
        import pymc3 as pm
        import theano
        import theano.tensor as tt

        # To simplfy things, we'll use cdf transforms to map everything to a hypercube, as in dynesty

        # First, define a theano Op for our likelihood function
        # This is based on the example here: https://docs.pymc.io/notebooks/blackbox_external_likelihood.html
        class LogLike(tt.Op):
            itypes = [tt.dvector] # expects a vector of parameter values when called
            otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

            def __init__(self, objfunc, objgrad):
                # add inputs as class attributes
                self.objfunc = objfunc
                self.objgrad = objgrad
                self.logpgrad = LogLikeGrad(objfunc, objgrad)

            def prior_transform(self, u):
                # This function transforms samples from the unit hypercube (u) to the target prior (x)
                global globdict  
                cparts = ['re','im']
                model_params_u   = u[:n_params]    
                gain_params_u    = u[n_params:(n_params+n_gains)]
                leakage_params_u = u[(n_params+n_gains):]
                model_params_x   = [cdf_inverse(  model_params_u[j], globdict['model_prior'][globdict['param_map'][j][0]][globdict['param_map'][j][1]]) for j in range(len(model_params_u))]
                gain_params_x    = [cdf_inverse(   gain_params_u[j], globdict['gain_prior'][globdict['gain_list'][j][1]]) for j in range(len(gain_params_u))]
                leakage_params_x = [cdf_inverse(leakage_params_u[j], globdict['leakage_prior'][globdict['leakage_fit'][j//2][0]][leakage_fit[j//2][1]][cparts[j%2]]) for j in range(len(leakage_params_u))]
                return np.concatenate([model_params_x, gain_params_x, leakage_params_x])

            def perform(self, node, inputs, outputs):
                # the method that is used when calling the Op
                theta, = inputs  # this will contain my variables
                # Transform from the hypercube to the prior
                x = self.prior_transform(theta)

                # call the log-likelihood function
                logl = -self.objfunc(x)

                outputs[0][0] = np.array(logl) # output the log-likelihood

            def grad(self, inputs, g):
                # the method that calculates the vector-Jacobian product
                # http://deeplearning.net/software/theano_versions/dev/extending/op.html#grad
                theta, = inputs 
                return [g[0]*self.logpgrad(theta)]

        class LogLikeGrad(tt.Op):
            """
            This Op will be called with a vector of values and also return a vector of
            values - the gradients in each dimension.
            """
            itypes = [tt.dvector]
            otypes = [tt.dvector]

            def __init__(self, objfunc, objgrad):
                self.objfunc = objfunc
                self.objgrad = objgrad

            def prior_transform(self, u):
                # This function transforms samples from the unit hypercube (u) to the target prior (x)
                global globdict  
                cparts = ['re','im']
                model_params_u   = u[:n_params]    
                gain_params_u    = u[n_params:(n_params+n_gains)]
                leakage_params_u = u[(n_params+n_gains):]
                model_params_x   = [cdf_inverse(  model_params_u[j], globdict['model_prior'][globdict['param_map'][j][0]][globdict['param_map'][j][1]]) for j in range(len(model_params_u))]
                gain_params_x    = [cdf_inverse(   gain_params_u[j], globdict['gain_prior'][globdict['gain_list'][j][1]]) for j in range(len(gain_params_u))]
                leakage_params_x = [cdf_inverse(leakage_params_u[j], globdict['leakage_prior'][globdict['leakage_fit'][j//2][0]][leakage_fit[j//2][1]][cparts[j%2]]) for j in range(len(leakage_params_u))]
                return np.concatenate([model_params_x, gain_params_x, leakage_params_x])

            def perform(self, node, inputs, outputs):
                theta, = inputs
                x = self.prior_transform(theta)
                outputs[0][0] = -self.objgrad(x)

        # create the log-likelihood Op
        logl = LogLike(objfunc, objgrad)

        # Define the sampler keywords
        min_kwargs = {}
        for key in minimizer_kwargs.keys():
            min_kwargs[key] = minimizer_kwargs[key]

        # Define the initial value if not passed 
        if 'start' not in min_kwargs.keys():
            cparts = ['re','im']
            model_params_x   = param_init[:n_params]    
            gain_params_x    = param_init[n_params:(n_params+n_gains)]
            leakage_params_x = param_init[(n_params+n_gains):]
            model_params_u   = [cdf(  model_params_x[j], globdict['model_prior'][globdict['param_map'][j][0]][globdict['param_map'][j][1]]) for j in range(len(model_params_x))]
            gain_params_u    = [cdf(   gain_params_x[j], globdict['gain_prior'][globdict['gain_list'][j][1]]) for j in range(len(gain_params_x))]
            leakage_params_u = [cdf(leakage_params_x[j], globdict['leakage_prior'][globdict['leakage_fit'][j//2][0]][leakage_fit[j//2][1]][cparts[j%2]]) for j in range(len(leakage_params_x))]
            param_init_u = np.concatenate([model_params_u, gain_params_u, leakage_params_u])
            min_kwargs['start'] = {}
            for j in range(len(param_init)):
                min_kwargs['start']['var' + str(j)] = param_init_u[j]

        # Setup the sampler
        with pm.Model() as model:
            theta = tt.as_tensor_variable([ pm.Uniform('var' + str(j), lower=0., upper=1.) for j in range(len(param_init)) ])
            pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
            trace = pm.sample(**min_kwargs)            

        # Extract useful sampling diagnostics.
        samples_u = np.vstack([trace['var' + str(j)] for j in range(len(param_init))]).T # samples in the hypercube
        samples = np.array([logl.prior_transform(u) for u in samples_u]) # samples 
        mean    = np.mean(samples,axis=0)
        var     = np.var(samples,axis=0)

        # Compute the log-posterior
        if not quiet: print('Calculating the posterior values for the samples...')
        logposterior = np.array([-objfunc(x, force_posterior=True) for x in samples])

        # Select the MAP
        j_MAP = np.argmax(logposterior)
        MAP  = samples[j_MAP]

        # Return a model determined by the MAP
        set_params(MAP[:n_params], trial_model, param_map, minimizer_func, model_prior)                
        gains = MAP[n_params:(n_params+n_gains)]
        leakage = MAP[(n_params+n_gains):]
        update_leakage(leakage)

        # Return the sampler
        ret['trace'] = trace
        ret['mean'] = mean
        ret['map'] = MAP
        ret['std']  = var**0.5
        ret['samples'] = samples
        ret['logposterior'] = logposterior

        # Return a set of models from the posterior
        posterior_models = []
        for j in range(N_POSTERIOR_SAMPLES):
            posterior_model = trial_model.copy()
            set_params(samples[-j][:n_params], posterior_model, param_map, minimizer_func, model_prior)  
            posterior_models.append(posterior_model)   
        ret['posterior_models'] = posterior_models    

        # Return data that has been rescaled based on 'natural' units for each parameter
        import copy
        samples_natural = copy.deepcopy(samples)
        samples_natural[:,:n_params] /= np.array([_[4] for _ in param_map])    
        ret['samples_natural'] = samples_natural

        # Return the names of the fitted parameters
        labels = []
        labels_natural = []
        for _ in param_map:
            labels.append(_[1].replace('_','-'))
            labels_natural.append(_[1].replace('_','-'))
            if _[3] != '':
                labels_natural[-1] += ' (' + _[3] + ')'
        for _ in gain_list:
            labels.append(str(_[0]) + ' ' + _[1])
            labels_natural.append(str(_[0]) + ' ' + _[1])
        for _ in leakage_fit:
            for coord in ['re','im']:
                labels.append(_[0] + ',' + _[1] + ',' + coord)
                labels_natural.append(_[0] + ',' + _[1] + ',' + coord)

        ret['labels'] = labels
        ret['labels_natural'] = labels_natural
    elif minimizer_func in ['dynesty_static','dynesty_dynamic']:
        ##########################
        ## Sample using dynesty ##
        ##########################
        import dynesty
        from dynesty import utils as dyfunc
        # Define the functions that dynesty requires
        def prior_transform(u):
            # This function transforms samples from the unit hypercube (u) to the target prior (x)
            global globdict  
            cparts = ['re','im']
            model_params_u   = u[:n_params]    
            gain_params_u    = u[n_params:(n_params+n_gains)]
            leakage_params_u = u[(n_params+n_gains):]
            model_params_x   = [cdf_inverse(  model_params_u[j], globdict['model_prior'][globdict['param_map'][j][0]][globdict['param_map'][j][1]]) for j in range(len(model_params_u))]
            gain_params_x    = [cdf_inverse(   gain_params_u[j], globdict['gain_prior'][globdict['gain_list'][j][1]]) for j in range(len(gain_params_u))]
            leakage_params_x = [cdf_inverse(leakage_params_u[j], globdict['leakage_prior'][globdict['leakage_fit'][j//2][0]][leakage_fit[j//2][1]][cparts[j%2]]) for j in range(len(leakage_params_u))]
            return np.concatenate([model_params_x, gain_params_x, leakage_params_x])

        def loglike(x):
            return -objfunc(x)

        def grad(x):
            return -objgrad(x)

        # Setup a multiprocessing pool if needed 
        if processes >= 0:
            import pathos.multiprocessing as mp
            from multiprocessing import cpu_count
            if processes == 0: processes = int(cpu_count())

            # Ensure efficient memory allocation among the processes and separate trial models for each
            def init(_globdict):
                global globdict 
                globdict = _globdict
                if processes >= 0:
                    globdict['trial_model'] = globdict['trial_model'].copy()

                return

            pool = mp.Pool(processes=processes, initializer=init, initargs=(globdict,))
            if not quiet: print('Using a pool with %d processes' % processes)
        else:
            pool = processes = None
            
        # Setup the sampler
        if minimizer_func == 'dynesty_static':
            sampler = dynesty.NestedSampler(loglike, prior_transform, ndim=len(param_init), gradient=grad, pool=pool, queue_size=processes, **minimizer_kwargs)
        else:
            sampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim=len(param_init), gradient=grad, pool=pool, queue_size=processes, **minimizer_kwargs)

        # Run the sampler
        sampler.run_nested(**run_nested_kwargs)

        # Print the sampler summary
        res = sampler.results
        if not quiet: 
            try: res.summary()
            except: pass

        # Extract useful sampling diagnostics.
        samples = res.samples                             # samples
        weights = np.exp(res.logwt - res.logz[-1])        # normalized weights
        mean, cov = dyfunc.mean_and_cov(samples, weights)

        # Compute the log-posterior
        if not quiet: print('Calculating the posterior values for the samples...')
        if pool is not None:
            from functools import partial
            def logpost(j):
                return -objfunc(samples[j], force_posterior=True)

            logposterior = pool.map(logpost, range(len(samples)))
        else:
            logposterior = np.array([-objfunc(x, force_posterior=True) for x in samples])

        # Close the pool (this may not be the desired behavior if the sampling is to be iterative!)
        if pool is not None:
            pool.close()

        # Select the MAP
        j_MAP = np.argmax(logposterior)
        MAP  = samples[j_MAP]

        # Resample from the posterior
        samples = dyfunc.resample_equal(samples, weights) 

        # Return a model determined by the MAP
        set_params(MAP[:n_params], trial_model, param_map, minimizer_func, model_prior)                
        gains = MAP[n_params:(n_params+n_gains)]
        leakage = MAP[(n_params+n_gains):]
        update_leakage(leakage)

        # Return the sampler
        ret['sampler'] = sampler
        ret['mean'] = mean
        ret['map'] = MAP
        ret['std']  = cov.diagonal()**0.5
        ret['samples'] = samples
        ret['logposterior'] = logposterior

        # Return a set of models from the posterior
        posterior_models = []
        for j in range(N_POSTERIOR_SAMPLES):
            posterior_model = trial_model.copy()
            set_params(samples[j][:n_params], posterior_model, param_map, minimizer_func, model_prior)  
            posterior_models.append(posterior_model)   
        ret['posterior_models'] = posterior_models    

        # Return data that has been rescaled based on 'natural' units for each parameter
        import copy
        res_natural = copy.deepcopy(res)
        res_natural.samples[:,:n_params] /= np.array([_[4] for _ in param_map])    
        samples_natural = samples[:,:n_params]/np.array([_[4] for _ in param_map])    
        ret['res_natural'] = res_natural
        ret['samples_natural'] = samples_natural

        # Return the names of the fitted parameters
        labels = []
        labels_natural = []
        for _ in param_map:
            labels.append(_[1].replace('_','-'))
            labels_natural.append(_[1].replace('_','-'))
            if _[3] != '':
                labels_natural[-1] += ' (' + _[3] + ')'
        for _ in gain_list:
            labels.append(str(_[0]) + ' ' + _[1])
            labels_natural.append(str(_[0]) + ' ' + _[1])
        for _ in leakage_fit:
            for coord in ['re','im']:
                labels.append(_[0] + ',' + _[1] + ',' + coord)
                labels_natural.append(_[0] + ',' + _[1] + ',' + coord)

        ret['labels'] = labels
        ret['labels_natural'] = labels_natural
    else:
        raise Exception('Minimizer function ' + minimizer_func + ' is not recognized!')

    # Format and print summary and fitted parameters
    tstop = time.time()
    trial_model = globdict['trial_model']

    if not quiet: 
        print("\ntime: %f s" % (tstop - tstart))
        print("\nFitted Parameters:")
    if minimizer_func not in ['dynesty_static','dynesty_dynamic','pymc3']:
        out = res.x
        set_params(out[:n_params], trial_model, param_map, minimizer_func, model_prior)
        gains = out[n_params:(n_params + n_gains)]
        leakage = out[(n_params + n_gains):]
        update_leakage(leakage)
        tparams = transform_params(out[:n_params], param_map, minimizer_func, model_prior)
        if not quiet: 
            cur_idx = -1
            if len(param_map):
                print('Model Parameters:')
                for j in range(len(param_map)):
                    if param_map[j][0] != cur_idx:
                        cur_idx = param_map[j][0]
                        print(model_init.models[cur_idx] + ' (component %d/%d):' % (cur_idx+1,model_init.N_models()))
                    print(('\t' + param_map[j][1] + ': %f ' + param_map[j][3]) % (tparams[j] * param_map[j][2]/param_map[j][4]))
                print('\n')
            
            if len(leakage_fit):
                print('Leakage (%; re, im):')
                for j in range(len(leakage_fit)):
                    print('\t' + leakage_fit[j][0] + ', ' + leakage_fit[j][1] + ': %2.2f %2.2f' % (leakage[2*j]*100,leakage[2*j + 1]*100))
                print('\n')

            print("Final Chi^2_1: %f Chi^2_2: %f  Chi^2_3: %f" % chisq_list(gains))
            print("J: %f" % res.fun)
            print(res.message)
    else:
        if not quiet: 
            cur_idx = -1
            if len(param_map):
                print('Model Parameters (mean and std):')
                for j in range(len(param_map)):
                    if param_map[j][0] != cur_idx:
                        cur_idx = param_map[j][0]
                        print(model_init.models[cur_idx] + ' (component %d/%d):' % (cur_idx+1,model_init.N_models()))
                    print(('\t' + param_map[j][1] + ': %f +/- %f ' + param_map[j][3]) % (ret['mean'][j] * param_map[j][2]/param_map[j][4], ret['std'][j] * param_map[j][2]/param_map[j][4]))
                print('\n')

            if len(leakage_fit):
                print('Leakage (%; re, im):')
                for j in range(len(leakage_fit)):
                    j2 = 2*j + n_params + n_gains
                    print(('\t' + leakage_fit[j][0] + ', ' + leakage_fit[j][1]
                                + ': %2.2f +/- %2.2f, %2.2f +/- %2.2f') 
                                % (mean[j2]*100,cov[j2,j2]**0.5 * 100,mean[j2+1]*100,cov[j2+1,j2+1]**0.5 * 100))
                print('\n')

    # Return fitted model
    ret['model']     = trial_model
    ret['param_map'] = param_map    
    ret['chisq_list'] = chisq_list(gains)
    try: ret['res'] = res
    except: pass

    if fit_gains:
        ret['gains'] = gains
    
        # Create and return a caltable
        caldict = {}
        for site in set(np.array(gain_list)[:,1]):
            caldict[site] = []
    
        for j in range(len(gains)):
            caldict[gain_list[j][1]].append((gain_list[j][0], (1.0 + gains[j]), (1.0 + gains[j])))

        for site in caldict.keys():
            caldict[site] = np.array(caldict[site], dtype=DTCAL)

        ct = caltable.Caltable(Obsdata.ra, Obsdata.dec, Obsdata.rf, Obsdata.bw, caldict, Obsdata.tarr,
                                           source=Obsdata.source, mjd=Obsdata.mjd, timetype=Obsdata.timetype)
        ret['caltable'] = ct 

    # If relevant, return useful quantities associated with the leakage
    if station_leakages is not None:
        ret['station_leakages'] = station_leakages
        tarr = Obsdata.tarr.copy()
        for s in station_leakages.keys():
            if 'R' in station_leakages[s].keys(): tarr[Obsdata.tkey[s]]['dr'] = station_leakages[s]['R']
            if 'L' in station_leakages[s].keys(): tarr[Obsdata.tkey[s]]['dl'] = station_leakages[s]['L']
        ret['tarr'] = tarr
                
    return ret

##################################################################################################
# Wrapper Functions
##################################################################################################

def chisq(model, uv, data, sigma, dtype, pol='I', jonesdict=None):
    """return the chi^2 for the appropriate dtype
    """

    chisq = 1
    if not dtype in DATATERMS:
        return chisq

    if dtype == 'vis':
        chisq = chisq_vis(model, uv, data, sigma, pol=pol, jonesdict=jonesdict)
    elif dtype == 'amp':
        chisq = chisq_amp(model, uv, data, sigma, pol=pol, jonesdict=jonesdict)
    elif dtype == 'logamp':
        chisq = chisq_logamp(model, uv, data, sigma, pol=pol, jonesdict=jonesdict)
    elif dtype == 'bs':
        chisq = chisq_bs(model, uv, data, sigma, pol=pol, jonesdict=jonesdict)
    elif dtype == 'cphase':
        chisq = chisq_cphase(model, uv, data, sigma, pol=pol, jonesdict=jonesdict)
    elif dtype == 'cphase_diag':
        chisq = chisq_cphase_diag(model, uv, data, sigma, pol=pol, jonesdict=jonesdict)
    elif dtype == 'camp':
        chisq = chisq_camp(model, uv, data, sigma, pol=pol, jonesdict=jonesdict)
    elif dtype == 'logcamp':
        chisq = chisq_logcamp(model, uv, data, sigma, pol=pol, jonesdict=jonesdict)
    elif dtype == 'logcamp_diag':
        chisq = chisq_logcamp_diag(model, uv, data, sigma, pol=pol, jonesdict=jonesdict)
    elif dtype == 'pvis':
        chisq = chisq_pvis(model, uv, data, sigma, jonesdict=jonesdict)
    elif dtype == 'm':
        chisq = chisq_m(model, uv, data, sigma, jonesdict=jonesdict)
    elif dtype in ['rrll','llrr','rlrr','rlll','lrrr','lrll']:
        chisq = chisq_fracpol(dtype[:2],dtype[2:],model, uv, data, sigma, jonesdict=jonesdict)
    elif dtype == 'polclosure':
        chisq = chisq_polclosure(model, uv, data, sigma, jonesdict=jonesdict)

    return chisq

def chisqgrad(model, uv, data, sigma, jonesdict, dtype, param_mask, pol='I', fit_gains=False, gains=None, gains_t1=None, gains_t2=None, fit_pol=False, fit_cpol=False, fit_leakage=False):
    """return the chi^2 gradient for the appropriate dtype
    """
    global globdict

    n_chisqgrad = len(param_mask)
    if fit_leakage:
        n_chisqgrad += 2*len(globdict['leakage_fit'])

    chisqgrad = np.zeros(n_chisqgrad)
    if fit_gains:
        gaingrad = np.zeros_like(gains)
    else:
        gaingrad = np.array([])

    # Now we need to be sure to put the gradient in the correct order: model parameters, then gains, then leakage
    param_mask_full   = np.zeros(len(chisqgrad), dtype=bool) 
    leakage_mask_full = np.zeros(len(chisqgrad), dtype=bool) 
    param_mask_full[:len(param_mask)] = param_mask
    leakage_mask_full[len(param_mask):] = ~leakage_mask_full[len(param_mask):]

    if not dtype in DATATERMS:
        return np.concatenate([chisqgrad[param_mask_full],gaingrad,chisqgrad[leakage_mask_full]])

    if dtype == 'vis':
        chisqgrad = chisqgrad_vis(model, uv, data, sigma, pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol, fit_leakage=fit_leakage, jonesdict=jonesdict)
    elif dtype == 'amp':
        chisqgrad = chisqgrad_amp(model, uv, data, sigma, pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol, fit_leakage=fit_leakage, jonesdict=jonesdict)

        if fit_gains:
            i1 = model.sample_uv(uv[:,0],uv[:,1], pol=pol, jonesdict=jonesdict)
            amp_samples = np.abs(i1)
            amp = data
            pp = ((amp - amp_samples) * amp_samples) / (sigma**2)
            gaingrad = 2.0/(1.0 + np.array(gains)) * np.array([np.sum(pp[(np.array(gains_t1) == j) + (np.array(gains_t2) == j)]) for j in range(len(gains))])/len(data)
    elif dtype == 'logamp':
        chisqgrad = chisqgrad_logamp(model, uv, data, sigma, pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol, fit_leakage=fit_leakage, jonesdict=jonesdict)
    elif dtype == 'bs':
        chisqgrad = chisqgrad_bs(model, uv, data, sigma, pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol, fit_leakage=fit_leakage, jonesdict=jonesdict)
    elif dtype == 'cphase':
        chisqgrad = chisqgrad_cphase(model, uv, data, sigma, pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol, fit_leakage=fit_leakage, jonesdict=jonesdict)
    elif dtype == 'cphase_diag':
        chisqgrad = chisqgrad_cphase_diag(model, uv, data, sigma, pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol, fit_leakage=fit_leakage, jonesdict=jonesdict)
    elif dtype == 'camp':
        chisqgrad = chisqgrad_camp(model, uv, data, sigma, pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol, fit_leakage=fit_leakage, jonesdict=jonesdict)
    elif dtype == 'logcamp':
        chisqgrad = chisqgrad_logcamp(model, uv, data, sigma, pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol, fit_leakage=fit_leakage, jonesdict=jonesdict)
    elif dtype == 'logcamp_diag':
        chisqgrad = chisqgrad_logcamp_diag(model, uv, data, sigma, pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol, fit_leakage=fit_leakage, jonesdict=jonesdict)
    elif dtype == 'pvis':
        chisqgrad = chisqgrad_pvis(model, uv, data, sigma, fit_pol=fit_pol, fit_cpol=fit_cpol, fit_leakage=fit_leakage, jonesdict=jonesdict)
    elif dtype == 'm':
        chisqgrad = chisqgrad_m(model, uv, data, sigma, fit_pol=fit_pol, fit_cpol=fit_cpol, fit_leakage=fit_leakage, jonesdict=jonesdict)
    elif dtype in ['rrll','llrr','rlrr','rlll','lrrr','lrll']:
        chisqgrad = chisqgrad_fracpol(dtype[:2],dtype[2:],model, uv, data, sigma, jonesdict=jonesdict, fit_pol=fit_pol, fit_cpol=fit_cpol, fit_leakage=fit_leakage)
    elif dtype == 'polclosure':
        chisqgrad = chisqgrad_polclosure(model, uv, data, sigma, fit_pol=fit_pol, fit_cpol=fit_cpol, fit_leakage=fit_leakage, jonesdict=jonesdict)

    return np.concatenate([chisqgrad[param_mask_full],gaingrad,chisqgrad[leakage_mask_full]])

def chisqdata(Obsdata, dtype, pol='I', **kwargs):

    """Return the data, sigma, and matrices for the appropriate dtype
    """

    (data, sigma, uv, jonesdict) = (False, False, False, None)

    if dtype == 'vis':
        (data, sigma, uv, jonesdict) = chisqdata_vis(Obsdata, pol=pol, **kwargs)
    elif dtype == 'amp' or dtype == 'logamp':
        (data, sigma, uv, jonesdict) = chisqdata_amp(Obsdata, pol=pol,**kwargs)
    elif dtype == 'bs':
        (data, sigma, uv, jonesdict) = chisqdata_bs(Obsdata, pol=pol,**kwargs)
    elif dtype == 'cphase':
        (data, sigma, uv, jonesdict) = chisqdata_cphase(Obsdata, pol=pol,**kwargs)
    elif dtype == 'cphase_diag':
        (data, sigma, uv, jonesdict) = chisqdata_cphase_diag(Obsdata, pol=pol,**kwargs)
    elif dtype == 'camp':
        (data, sigma, uv, jonesdict) = chisqdata_camp(Obsdata, pol=pol,**kwargs)
    elif dtype == 'logcamp':
        (data, sigma, uv, jonesdict) = chisqdata_logcamp(Obsdata, pol=pol,**kwargs)
    elif dtype == 'logcamp_diag':
        (data, sigma, uv, jonesdict) = chisqdata_logcamp_diag(Obsdata, pol=pol,**kwargs)
    elif dtype == 'pvis':
        (data, sigma, uv, jonesdict) = chisqdata_pvis(Obsdata, pol=pol,**kwargs)
    elif dtype == 'm':
        (data, sigma, uv, jonesdict) = chisqdata_m(Obsdata, pol=pol,**kwargs)
    elif dtype in ['rrll','llrr','rlrr','rlll','lrrr','lrll']:
        (data, sigma, uv, jonesdict) = chisqdata_fracpol(Obsdata,dtype[:2],dtype[2:],jonesdict=jonesdict)
    elif dtype == 'polclosure':
        (data, sigma, uv, jonesdict) = chisqdata_polclosure(Obsdata,jonesdict=jonesdict)

    return (data, sigma, uv, jonesdict)


##################################################################################################
# Chi-squared and Gradient Functions
##################################################################################################

def chisq_vis(model, uv, vis, sigma, pol='I', jonesdict=None):
    """Visibility chi-squared"""

    samples = model.sample_uv(uv[:,0],uv[:,1], pol=pol, jonesdict=jonesdict)
    return np.sum(np.abs((samples-vis)/sigma)**2)/(2*len(vis))

def chisqgrad_vis(model, uv, vis, sigma, pol='I', fit_pol=False, fit_cpol=False, fit_leakage=False, jonesdict=None):
    """The gradient of the visibility chi-squared"""

    samples = model.sample_uv(uv[:,0],uv[:,1], pol=pol, jonesdict=jonesdict)
    wdiff   = (vis - samples)/(sigma**2)
    grad    = model.sample_grad_uv(uv[:,0],uv[:,1],fit_pol=fit_pol,fit_cpol=fit_cpol,fit_leakage=fit_leakage,jonesdict=jonesdict)

    out = -np.real(np.dot(grad.conj(), wdiff))/len(vis)
    return out

def chisq_amp(model, uv, amp, sigma, pol='I', jonesdict=None):
    """Visibility Amplitudes (normalized) chi-squared"""

    amp_samples = np.abs(model.sample_uv(uv[:,0],uv[:,1], pol=pol, jonesdict=jonesdict))
    return np.sum(np.abs((amp - amp_samples)/sigma)**2)/len(amp)

def chisqgrad_amp(model, uv, amp, sigma, pol='I', fit_pol=False, fit_cpol=False, fit_leakage=False, jonesdict=None):
    """The gradient of the amplitude chi-squared"""

    i1 = model.sample_uv(uv[:,0],uv[:,1], jonesdict=jonesdict)
    amp_samples = np.abs(i1)

    pp = ((amp - amp_samples) * amp_samples) / (sigma**2) / i1
    grad = model.sample_grad_uv(uv[:,0],uv[:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,fit_leakage=fit_leakage,jonesdict=jonesdict)
    out = (-2.0/len(amp)) * np.real(np.dot(grad, pp))
    return out

def chisq_bs(model, uv, bis, sigma, pol='I', jonesdict=None):
    """Bispectrum chi-squared"""

    bisamples = model.sample_uv(uv[0][:,0],uv[0][:,1],pol=pol,jonesdict=jonesdict) * model.sample_uv(uv[1][:,0],uv[1][:,1],pol=pol,jonesdict=jonesdict) * model.sample_uv(uv[2][:,0],uv[2][:,1],pol=pol,jonesdict=jonesdict)
    chisq= np.sum(np.abs(((bis - bisamples)/sigma))**2)/(2.*len(bis))
    return chisq

def chisqgrad_bs(model, uv, bis, sigma, pol='I', fit_pol=False, fit_cpol=False, fit_leakage=False, jonesdict=None):
    """The gradient of the bispectrum chi-squared"""

    V1 = model.sample_uv(uv[0][:,0],uv[0][:,1],pol=pol,jonesdict=jonesdict)
    V2 = model.sample_uv(uv[1][:,0],uv[1][:,1],pol=pol,jonesdict=jonesdict)
    V3 = model.sample_uv(uv[2][:,0],uv[2][:,1],pol=pol,jonesdict=jonesdict)
    V1_grad = model.sample_grad_uv(uv[0][:,0],uv[0][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)
    V2_grad = model.sample_grad_uv(uv[1][:,0],uv[1][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)
    V3_grad = model.sample_grad_uv(uv[2][:,0],uv[2][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)
    bisamples = V1 * V2 * V3 
    wdiff = ((bis - bisamples).conj())/(sigma**2)
    pt1 = wdiff * V2 * V3
    pt2 = wdiff * V1 * V3
    pt3 = wdiff * V1 * V2
    out = -np.real(np.dot(pt1, V1_grad.T) + np.dot(pt2, V2_grad.T) + np.dot(pt3, V3_grad.T))/len(bis)
    return out

def chisq_cphase(model, uv, clphase, sigma, pol='I', jonesdict=None):
    """Closure Phases (normalized) chi-squared"""
    clphase = clphase * DEGREE
    sigma = sigma * DEGREE

    V1 = model.sample_uv(uv[0][:,0],uv[0][:,1],pol=pol,jonesdict=jonesdict)
    V2 = model.sample_uv(uv[1][:,0],uv[1][:,1],pol=pol,jonesdict=jonesdict)
    V3 = model.sample_uv(uv[2][:,0],uv[2][:,1],pol=pol,jonesdict=jonesdict)

    clphase_samples = np.angle(V1 * V2 * V3)
    chisq= (2.0/len(clphase)) * np.sum((1.0 - np.cos(clphase-clphase_samples))/(sigma**2))
    return chisq

def chisqgrad_cphase(model, uv, clphase, sigma, pol='I', fit_pol=False, fit_cpol=False, fit_leakage=False, jonesdict=None):
    """The gradient of the closure phase chi-squared"""
    clphase = clphase * DEGREE
    sigma = sigma * DEGREE

    V1 = model.sample_uv(uv[0][:,0],uv[0][:,1],pol=pol,jonesdict=jonesdict)
    V2 = model.sample_uv(uv[1][:,0],uv[1][:,1],pol=pol,jonesdict=jonesdict)
    V3 = model.sample_uv(uv[2][:,0],uv[2][:,1],pol=pol,jonesdict=jonesdict)
    V1_grad = model.sample_grad_uv(uv[0][:,0],uv[0][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)
    V2_grad = model.sample_grad_uv(uv[1][:,0],uv[1][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)
    V3_grad = model.sample_grad_uv(uv[2][:,0],uv[2][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)

    clphase_samples = np.angle(V1 * V2 * V3)

    pref = np.sin(clphase - clphase_samples)/(sigma**2)
    pt1  = pref/V1
    pt2  = pref/V2
    pt3  = pref/V3
    out  = -(2.0/len(clphase)) * np.imag(np.dot(pt1, V1_grad.T) + np.dot(pt2, V2_grad.T) + np.dot(pt3, V3_grad.T))
    return out

def chisq_cphase_diag(model, uv, clphase_diag, sigma, pol='I', jonesdict=None):
    """Diagonalized closure phases (normalized) chi-squared"""
    clphase_diag = np.concatenate(clphase_diag) * DEGREE
    sigma = np.concatenate(sigma) * DEGREE

    uv_diag = uv[0]
    tform_mats = uv[1]

    clphase_diag_samples = []
    for iA, uv3 in enumerate(uv_diag):
        i1 = model.sample_uv(uv3[0][:,0],uv3[0][:,1],pol=pol,jonesdict=jonesdict)    
        i2 = model.sample_uv(uv3[1][:,0],uv3[1][:,1],pol=pol,jonesdict=jonesdict)    
        i3 = model.sample_uv(uv3[2][:,0],uv3[2][:,1],pol=pol,jonesdict=jonesdict)  

        clphase_samples = np.angle(i1 * i2 * i3)
        clphase_diag_samples.append(np.dot(tform_mats[iA],clphase_samples))
    clphase_diag_samples = np.concatenate(clphase_diag_samples)

    chisq = (2.0/len(clphase_diag)) * np.sum((1.0 - np.cos(clphase_diag-clphase_diag_samples))/(sigma**2))
    return chisq

def chisqgrad_cphase_diag(model, uv, clphase_diag, sigma, pol='I', fit_pol=False, fit_cpol=False, fit_leakage=False, jonesdict=None):
    """The gradient of the diagonalized closure phase chi-squared"""
    clphase_diag = clphase_diag * DEGREE
    sigma = sigma * DEGREE

    uv_diag = uv[0]
    tform_mats = uv[1]

    deriv = np.zeros(len(model.sample_grad_uv(0,0,pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)))
    for iA, uv3 in enumerate(uv_diag):

        i1 = model.sample_uv(uv3[0][:,0],uv3[0][:,1],pol=pol,jonesdict=jonesdict)    
        i2 = model.sample_uv(uv3[1][:,0],uv3[1][:,1],pol=pol,jonesdict=jonesdict)    
        i3 = model.sample_uv(uv3[2][:,0],uv3[2][:,1],pol=pol,jonesdict=jonesdict)   

        i1_grad = model.sample_grad_uv(uv3[0][:,0],uv3[0][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)
        i2_grad = model.sample_grad_uv(uv3[1][:,0],uv3[1][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)
        i3_grad = model.sample_grad_uv(uv3[2][:,0],uv3[2][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)
 
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

def chisq_camp(model, uv, clamp, sigma, pol='I', jonesdict=None):
    """Closure Amplitudes (normalized) chi-squared"""

    V1 = model.sample_uv(uv[0][:,0],uv[0][:,1],pol=pol,jonesdict=jonesdict)
    V2 = model.sample_uv(uv[1][:,0],uv[1][:,1],pol=pol,jonesdict=jonesdict)
    V3 = model.sample_uv(uv[2][:,0],uv[2][:,1],pol=pol,jonesdict=jonesdict)
    V4 = model.sample_uv(uv[3][:,0],uv[3][:,1],pol=pol,jonesdict=jonesdict)

    clamp_samples = np.abs(V1 * V2 / (V3 * V4))
    chisq = np.sum(np.abs((clamp - clamp_samples)/sigma)**2)/len(clamp)
    return chisq

def chisqgrad_camp(model, uv, clamp, sigma, pol='I', fit_pol=False, fit_cpol=False, fit_leakage=False, jonesdict=None):
    """The gradient of the closure amplitude chi-squared"""

    V1 = model.sample_uv(uv[0][:,0],uv[0][:,1],pol=pol,jonesdict=jonesdict)
    V2 = model.sample_uv(uv[1][:,0],uv[1][:,1],pol=pol,jonesdict=jonesdict)
    V3 = model.sample_uv(uv[2][:,0],uv[2][:,1],pol=pol,jonesdict=jonesdict)
    V4 = model.sample_uv(uv[3][:,0],uv[3][:,1],pol=pol,jonesdict=jonesdict)
    V1_grad = model.sample_grad_uv(uv[0][:,0],uv[0][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)
    V2_grad = model.sample_grad_uv(uv[1][:,0],uv[1][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)
    V3_grad = model.sample_grad_uv(uv[2][:,0],uv[2][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)
    V4_grad = model.sample_grad_uv(uv[3][:,0],uv[3][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)

    clamp_samples = np.abs((V1 * V2)/(V3 * V4))

    pp = ((clamp - clamp_samples) * clamp_samples)/(sigma**2)
    pt1 =  pp/V1
    pt2 =  pp/V2
    pt3 = -pp/V3
    pt4 = -pp/V4
    out = (-2.0/len(clamp)) * np.real(np.dot(pt1, V1_grad.T) + np.dot(pt2, V2_grad.T) + np.dot(pt3, V3_grad.T) + np.dot(pt4, V4_grad.T))
    return out

def chisq_logcamp(model, uv, log_clamp, sigma, pol='I', jonesdict=None):
    """Log Closure Amplitudes (normalized) chi-squared"""

    a1 = np.abs(model.sample_uv(uv[0][:,0],uv[0][:,1],pol=pol,jonesdict=jonesdict))
    a2 = np.abs(model.sample_uv(uv[1][:,0],uv[1][:,1],pol=pol,jonesdict=jonesdict))
    a3 = np.abs(model.sample_uv(uv[2][:,0],uv[2][:,1],pol=pol,jonesdict=jonesdict))
    a4 = np.abs(model.sample_uv(uv[3][:,0],uv[3][:,1],pol=pol,jonesdict=jonesdict))

    samples = np.log(a1) + np.log(a2) - np.log(a3) - np.log(a4)
    chisq = np.sum(np.abs((log_clamp - samples)/sigma)**2) / (len(log_clamp))
    return  chisq

def chisqgrad_logcamp(model, uv, log_clamp, sigma, pol='I', fit_pol=False, fit_cpol=False, fit_leakage=False, jonesdict=None):
    """The gradient of the Log closure amplitude chi-squared"""

    V1 = model.sample_uv(uv[0][:,0],uv[0][:,1],pol=pol,jonesdict=jonesdict)
    V2 = model.sample_uv(uv[1][:,0],uv[1][:,1],pol=pol,jonesdict=jonesdict)
    V3 = model.sample_uv(uv[2][:,0],uv[2][:,1],pol=pol,jonesdict=jonesdict)
    V4 = model.sample_uv(uv[3][:,0],uv[3][:,1],pol=pol,jonesdict=jonesdict)
    V1_grad = model.sample_grad_uv(uv[0][:,0],uv[0][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)
    V2_grad = model.sample_grad_uv(uv[1][:,0],uv[1][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)
    V3_grad = model.sample_grad_uv(uv[2][:,0],uv[2][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)
    V4_grad = model.sample_grad_uv(uv[3][:,0],uv[3][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)

    log_clamp_samples = np.log(np.abs(V1)) + np.log(np.abs(V2)) - np.log(np.abs(V3)) - np.log(np.abs(V4))

    pp = (log_clamp - log_clamp_samples) / (sigma**2)
    pt1 = pp / V1
    pt2 = pp / V2
    pt3 = -pp / V3
    pt4 = -pp / V4
    out = (-2.0/len(log_clamp)) * np.real(np.dot(pt1, V1_grad.T) + np.dot(pt2, V2_grad.T) + np.dot(pt3, V3_grad.T) + np.dot(pt4, V4_grad.T))
    return out

def chisq_logcamp_diag(model, uv, log_clamp_diag, sigma, pol='I', jonesdict=None):
    """Diagonalized log closure amplitudes (normalized) chi-squared"""

    log_clamp_diag = np.concatenate(log_clamp_diag)
    sigma = np.concatenate(sigma)

    uv_diag = uv[0]
    tform_mats = uv[1]

    log_clamp_diag_samples = []
    for iA, uv4 in enumerate(uv_diag):

        a1 = np.abs(model.sample_uv(uv4[0][:,0],uv4[0][:,1],pol=pol,jonesdict=jonesdict))          
        a2 = np.abs(model.sample_uv(uv4[1][:,0],uv4[1][:,1],pol=pol,jonesdict=jonesdict)) 
        a3 = np.abs(model.sample_uv(uv4[2][:,0],uv4[2][:,1],pol=pol,jonesdict=jonesdict)) 
        a4 = np.abs(model.sample_uv(uv4[3][:,0],uv4[3][:,1],pol=pol,jonesdict=jonesdict)) 

        log_clamp_samples = np.log(a1) + np.log(a2) - np.log(a3) - np.log(a4)
        log_clamp_diag_samples.append(np.dot(tform_mats[iA],log_clamp_samples))

    log_clamp_diag_samples = np.concatenate(log_clamp_diag_samples)

    chisq = np.sum(np.abs((log_clamp_diag - log_clamp_diag_samples)/sigma)**2) / (len(log_clamp_diag))
    return  chisq

def chisqgrad_logcamp_diag(model, uv, log_clamp_diag, sigma, pol='I', fit_pol=False, fit_cpol=False, fit_leakage=False, jonesdict=None):
    """The gradient of the diagonalized log closure amplitude chi-squared"""

    uv_diag = uv[0]
    tform_mats = uv[1]

    deriv = np.zeros(len(model.sample_grad_uv(0,0,pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)))
    for iA, uv4 in enumerate(uv_diag):

        i1 = model.sample_uv(uv4[0][:,0],uv4[0][:,1],pol=pol,jonesdict=jonesdict)
        i2 = model.sample_uv(uv4[1][:,0],uv4[1][:,1],pol=pol,jonesdict=jonesdict)
        i3 = model.sample_uv(uv4[2][:,0],uv4[2][:,1],pol=pol,jonesdict=jonesdict)
        i4 = model.sample_uv(uv4[3][:,0],uv4[3][:,1],pol=pol,jonesdict=jonesdict)

        i1_grad = model.sample_grad_uv(uv4[0][:,0],uv4[0][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)
        i2_grad = model.sample_grad_uv(uv4[1][:,0],uv4[1][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)
        i3_grad = model.sample_grad_uv(uv4[2][:,0],uv4[2][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)
        i4_grad = model.sample_grad_uv(uv4[3][:,0],uv4[3][:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)

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

def chisq_logamp(model, uv, amp, sigma, pol='I', jonesdict=None):
    """Log Visibility Amplitudes (normalized) chi-squared"""

    # to lowest order the variance on the logarithm of a quantity x is
    # sigma^2_log(x) = sigma^2/x^2
    logsigma = sigma / amp

    amp_samples = np.abs(model.sample_uv(uv[:,0],uv[:,1],pol=pol,jonesdict=jonesdict))
    return np.sum(np.abs((np.log(amp) - np.log(amp_samples))/logsigma)**2)/len(amp)

def chisqgrad_logamp(model, uv, amp, sigma, pol='I', fit_pol=False, fit_cpol=False, fit_leakage=False, jonesdict=None):
    """The gradient of the Log amplitude chi-squared"""

    # to lowest order the variance on the logarithm of a quantity x is
    # sigma^2_log(x) = sigma^2/x^2
    logsigma = sigma / amp

    i1 = model.sample_uv(uv[:,0],uv[:,1],pol=pol,jonesdict=jonesdict)
    amp_samples = np.abs(i1)

    V_grad = model.sample_grad_uv(uv[:,0],uv[:,1],pol=pol,fit_pol=fit_pol,fit_cpol=fit_cpol,jonesdict=jonesdict)

    pp = ((np.log(amp) - np.log(amp_samples))) / (logsigma**2) / i1
    out = (-2.0/len(amp)) * np.real(np.dot(pp, V_grad.T))
    return out


def chisq_pvis(model, uv, pvis, psigma, jonesdict=None):
    """Polarimetric visibility chi-squared
    """

    psamples = model.sample_uv(uv[:,0],uv[:,1],pol='P',jonesdict=jonesdict)
    return np.sum(np.abs((psamples-pvis)/psigma)**2)/(2*len(pvis))

def chisqgrad_pvis(model, uv, pvis, psigma, fit_pol=False, fit_cpol=False, fit_leakage=False, jonesdict=None):
    """Polarimetric visibility chi-squared gradient
    """
    samples = model.sample_uv(uv[:,0],uv[:,1],pol='P',jonesdict=jonesdict)
    wdiff   = (pvis - samples)/(psigma**2)
    grad    = model.sample_grad_uv(uv[:,0],uv[:,1],pol='P',fit_pol=fit_pol,fit_cpol=fit_cpol,fit_leakage=fit_leakage,jonesdict=jonesdict)

    out = -np.real(np.dot(grad.conj(), wdiff))/len(pvis)
    return out

def chisq_m(model, uv, m, msigma, jonesdict=None):
    """Polarimetric ratio chi-squared
    """

    msamples = model.sample_uv(uv[:,0],uv[:,1],pol='P',jonesdict=jonesdict)/model.sample_uv(uv[:,0],uv[:,1],pol='I',jonesdict=jonesdict)

    return np.sum(np.abs((m - msamples))**2/(msigma**2)) / (2*len(m))   

def chisqgrad_m(model, uv, mvis, msigma, fit_pol=False, fit_cpol=False, fit_leakage=False, jonesdict=None):
    """The gradient of the polarimetric ratio chisq
    """

    samp_P   = model.sample_uv(uv[:,0],uv[:,1],pol='P',jonesdict=jonesdict)
    samp_I   = model.sample_uv(uv[:,0],uv[:,1],pol='I',jonesdict=jonesdict)
    grad_P   = model.sample_grad_uv(uv[:,0],uv[:,1],pol='P',fit_pol=fit_pol,fit_cpol=fit_cpol,fit_leakage=fit_leakage,jonesdict=jonesdict)
    grad_I   = model.sample_grad_uv(uv[:,0],uv[:,1],pol='I',fit_pol=fit_pol,fit_cpol=fit_cpol,fit_leakage=fit_leakage,jonesdict=jonesdict)

    msamples = samp_P/samp_I
    wdiff   = (mvis - msamples)/(msigma**2)
    # Get the gradient from the quotient rule
    grad    = ( grad_P * samp_I - grad_I * samp_P)/samp_I**2

    return -np.real(np.dot(grad.conj(), wdiff))/len(mvis)

def chisq_fracpol(upper, lower, model, uv, m, msigma, jonesdict=None):
    """Polarimetric ratio chi-squared
    """

    msamples = model.sample_uv(uv[:,0],uv[:,1],pol=upper.upper(),jonesdict=jonesdict)/model.sample_uv(uv[:,0],uv[:,1],pol=lower.upper(),jonesdict=jonesdict)

    return np.sum(np.abs((m - msamples))**2/(msigma**2)) / (2*len(m))   

def chisqgrad_fracpol(upper, lower, model, uv, mvis, msigma, fit_pol=False, fit_cpol=False, fit_leakage=False, jonesdict=None):
    """The gradient of the polarimetric ratio chisq
    """

    samp_upper = model.sample_uv(uv[:,0],uv[:,1],pol=upper.upper(),jonesdict=jonesdict)
    samp_lower = model.sample_uv(uv[:,0],uv[:,1],pol=lower.upper(),jonesdict=jonesdict)
    grad_upper = model.sample_grad_uv(uv[:,0],uv[:,1],pol=upper.upper(),fit_pol=fit_pol,fit_cpol=fit_cpol,fit_leakage=fit_leakage,jonesdict=jonesdict)
    grad_lower = model.sample_grad_uv(uv[:,0],uv[:,1],pol=lower.upper(),fit_pol=fit_pol,fit_cpol=fit_cpol,fit_leakage=fit_leakage,jonesdict=jonesdict)

    msamples = samp_upper/samp_lower
    wdiff   = (mvis - msamples)/(msigma**2)
    # Get the gradient from the quotient rule
    grad    = ( grad_upper * samp_lower - grad_lower * samp_upper)/samp_lower**2

    return -np.real(np.dot(grad.conj(), wdiff))/len(mvis)

def chisq_polclosure(model, uv, vis, sigma, jonesdict=None):
    """Polarimetric ratio chi-squared
    """

    RL = model.sample_uv(uv[:,0],uv[:,1],pol='RL',jonesdict=jonesdict)
    LR = model.sample_uv(uv[:,0],uv[:,1],pol='LR',jonesdict=jonesdict)
    RR = model.sample_uv(uv[:,0],uv[:,1],pol='RR',jonesdict=jonesdict)
    LL = model.sample_uv(uv[:,0],uv[:,1],pol='LL',jonesdict=jonesdict)
    samples = (RL * LR)/(RR * LL)

    return np.sum(np.abs((vis - samples))**2/(sigma**2)) / (2*len(vis))   

def chisqgrad_polclosure(model, uv, vis, sigma, fit_pol=False, fit_cpol=False, fit_leakage=False, jonesdict=None):
    """The gradient of the polarimetric ratio chisq
    """

    RL = model.sample_uv(uv[:,0],uv[:,1],pol='RL',jonesdict=jonesdict)
    LR = model.sample_uv(uv[:,0],uv[:,1],pol='LR',jonesdict=jonesdict)
    RR = model.sample_uv(uv[:,0],uv[:,1],pol='RR',jonesdict=jonesdict)
    LL = model.sample_uv(uv[:,0],uv[:,1],pol='LL',jonesdict=jonesdict)

    dRL = model.sample_grad_uv(uv[:,0],uv[:,1],pol='RL',fit_pol=fit_pol,fit_cpol=fit_cpol,fit_leakage=fit_leakage,jonesdict=jonesdict)
    dLR = model.sample_grad_uv(uv[:,0],uv[:,1],pol='LR',fit_pol=fit_pol,fit_cpol=fit_cpol,fit_leakage=fit_leakage,jonesdict=jonesdict)
    dRR = model.sample_grad_uv(uv[:,0],uv[:,1],pol='RR',fit_pol=fit_pol,fit_cpol=fit_cpol,fit_leakage=fit_leakage,jonesdict=jonesdict)
    dLL = model.sample_grad_uv(uv[:,0],uv[:,1],pol='LL',fit_pol=fit_pol,fit_cpol=fit_cpol,fit_leakage=fit_leakage,jonesdict=jonesdict)

    samples = (RL * LR)/(RR * LL)
    wdiff   = (vis - samples)/(sigma**2)

    # Get the gradient from the quotient rule
    samp_upper = RL * LR
    samp_lower = RR * LL
    grad_upper = RL * dLR + dRL * LR
    grad_lower = RR * dLL + dRR * LL
    grad    = ( grad_upper * samp_lower - grad_lower * samp_upper)/samp_lower**2

    return -np.real(np.dot(grad.conj(), wdiff))/len(vis)


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

def make_jonesdict(Obsdata, data_arr):
    # Make a dictionary with entries needed to form the Jones matrices
    # Currently, this only works for data types on a single baseline (e.g., closure quantities aren't supported yet)

    # Get the names of each station for every measurement
    t1 = data_arr['t1']
    t2 = data_arr['t2']

    # Get the elevation of each station
    el1  = data_arr['el1']*np.pi/180.
    el2  = data_arr['el2']*np.pi/180.

    # Get the parallactic angle of each station
    par1 = data_arr['par_ang1']*np.pi/180.
    par2 = data_arr['par_ang2']*np.pi/180.

    # Compute the full field rotation angle for each site, based information in the Obsdata Array
    fr_elev1 = np.array([Obsdata.tarr[Obsdata.tkey[o['t1']]]['fr_elev'] for o in data_arr])
    fr_elev2 = np.array([Obsdata.tarr[Obsdata.tkey[o['t2']]]['fr_elev'] for o in data_arr])
    fr_par1  = np.array([Obsdata.tarr[Obsdata.tkey[o['t1']]]['fr_par']  for o in data_arr])
    fr_par2  = np.array([Obsdata.tarr[Obsdata.tkey[o['t2']]]['fr_par']  for o in data_arr])
    fr_off1  = np.array([Obsdata.tarr[Obsdata.tkey[o['t1']]]['fr_off']  for o in data_arr])
    fr_off2  = np.array([Obsdata.tarr[Obsdata.tkey[o['t2']]]['fr_off']  for o in data_arr])
    fr1 = fr_elev1*el1 + fr_par1*par1 + fr_off1*np.pi/180.
    fr2 = fr_elev2*el2 + fr_par2*par2 + fr_off2*np.pi/180.

    # Now populate the left and right D-term entries based on the Obsdata Array
    DR1 = np.array([Obsdata.tarr[Obsdata.tkey[o['t1']]]['dr'] for o in data_arr])
    DL1 = np.array([Obsdata.tarr[Obsdata.tkey[o['t1']]]['dl'] for o in data_arr])
    DR2 = np.array([Obsdata.tarr[Obsdata.tkey[o['t2']]]['dr'] for o in data_arr])
    DL2 = np.array([Obsdata.tarr[Obsdata.tkey[o['t2']]]['dl'] for o in data_arr])

    return {'fr1':fr1,'fr2':fr2,'t1':t1,'t2':t2, 
            'DR1':DR1, 'DR2':DR2, 'DL1':DL1, 'DL2':DL2}

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
    data_arr = Obsdata.unpack(['t1','t2','u','v',vtype,atype,etype,'el1','el2','par_ang1','par_ang2'], debias=debias)
    (uv, vis, amp, sigma) = apply_systematic_noise_snrcut(data_arr, systematic_noise, snrcut, pol)

    jonesdict = make_jonesdict(Obsdata, data_arr)

    return (vis, sigma, uv, jonesdict)

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
        data_arr = Obsdata.unpack(['time','t1','t2','u','v',vtype,atype,etype,'el1','el2','par_ang1','par_ang2'], debias=debias)

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
        
    jonesdict = make_jonesdict(Obsdata, data_arr)

    return (amp, sigma, uv, jonesdict)

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

    return (bi, sigma, (uv1, uv2, uv3), None)

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

    return (clphase, sigma, (uv1, uv2, uv3), None)

def chisqdata_cphase_diag(Obsdata, pol='I',**kwargs):
    """Return the data, sigmas, and fourier matrices for diagonalized closure phases
    """

    # unpack keyword args
    maxset = kwargs.get('maxset',False)
    uv_min = kwargs.get('cp_uv_min', False)
    if maxset: count='max'
    else: count='min'

    snrcut = kwargs.get('snrcut',0.)

    # unpack data
    vtype=vis_poldict[pol]
    clphasearr = Obsdata.c_phases_diag(vtype=vtype,count=count,snrcut=snrcut,uv_min=uv_min)

    # loop over timestamps
    clphase_diag = []
    sigma_diag = []
    uv_diag = []
    tform_mats = []
    for ic, cl in enumerate(clphasearr):

        # get diagonalized closure phases and errors
        clphase_diag.append(cl[0]['cphase'])
        sigma_diag.append(cl[0]['sigmacp'])
        
        # get uv arrays
        u1 = cl[2][:,0].astype('float')
        v1 = cl[3][:,0].astype('float')
        uv1 = np.hstack((u1.reshape(-1,1), v1.reshape(-1,1)))

        u2 = cl[2][:,1].astype('float')
        v2 = cl[3][:,1].astype('float')
        uv2 = np.hstack((u2.reshape(-1,1), v2.reshape(-1,1)))

        u3 = cl[2][:,2].astype('float')
        v3 = cl[3][:,2].astype('float')
        uv3 = np.hstack((u3.reshape(-1,1), v3.reshape(-1,1)))

        # compute Fourier matrices
        uv = (uv1,
              uv2,
              uv3
             )
        uv_diag.append(uv)

        # get transformation matrix for this timestamp
        tform_mats.append(cl[4].astype('float'))

    # combine Fourier and transformation matrices into tuple for outputting
    uvmatrices = (np.array(uv_diag),np.array(tform_mats))

    return (np.array(clphase_diag), np.array(sigma_diag), uvmatrices, None)

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

    return (clamp, sigma, (uv1, uv2, uv3, uv4), None)

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

    return (clamp, sigma, (uv1, uv2, uv3, uv4), None)

def chisqdata_logcamp_diag(Obsdata, pol='I', **kwargs):
    """Return the data, sigmas, and fourier matrices for diagonalized log closure amplitudes
    """
    # unpack keyword args
    maxset = kwargs.get('maxset',False)
    if maxset: count='max'
    else: count='min'

    snrcut = kwargs.get('snrcut',0.)
    debias = kwargs.get('debias',True)

    # unpack data & mask low snr points
    vtype=vis_poldict[pol]
    clamparr = Obsdata.c_log_amplitudes_diag(vtype=vtype,count=count,debias=debias,snrcut=snrcut)

    # loop over timestamps
    clamp_diag = []
    sigma_diag = []
    uv_diag = []
    tform_mats = []
    for ic, cl in enumerate(clamparr):

        # get diagonalized log closure amplitudes and errors
        clamp_diag.append(cl[0]['camp'])
        sigma_diag.append(cl[0]['sigmaca'])

        # get uv arrays
        u1 = cl[2][:,0].astype('float')
        v1 = cl[3][:,0].astype('float')
        uv1 = np.hstack((u1.reshape(-1,1), v1.reshape(-1,1)))

        u2 = cl[2][:,1].astype('float')
        v2 = cl[3][:,1].astype('float')
        uv2 = np.hstack((u2.reshape(-1,1), v2.reshape(-1,1)))

        u3 = cl[2][:,2].astype('float')
        v3 = cl[3][:,2].astype('float')
        uv3 = np.hstack((u3.reshape(-1,1), v3.reshape(-1,1)))

        u4 = cl[2][:,3].astype('float')
        v4 = cl[3][:,3].astype('float')
        uv4 = np.hstack((u4.reshape(-1,1), v4.reshape(-1,1)))

        # compute Fourier matrices
        uv = (uv1,
              uv2,
              uv3,
              uv4
             )
        uv_diag.append(uv)

        # get transformation matrix for this timestamp
        tform_mats.append(cl[4].astype('float'))

    # combine Fourier and transformation matrices into tuple for outputting
    uvmatrices = (np.array(uv_diag),np.array(tform_mats))

    return (np.array(clamp_diag), np.array(sigma_diag), uvmatrices, None)

def chisqdata_pvis(Obsdata, pol='I', **kwargs):
    data_arr = Obsdata.unpack(['t1','t2','u','v','pvis','psigma','el1','el2','par_ang1','par_ang2'], conj=True)
    uv = np.hstack((data_arr['u'].reshape(-1,1), data_arr['v'].reshape(-1,1)))
    mask = np.isfinite(data_arr['pvis'] + data_arr['psigma']) # don't include nan (missing data) or inf (division by zero)
    jonesdict = make_jonesdict(Obsdata, data_arr[mask])
    return (data_arr['pvis'][mask], data_arr['psigma'][mask], uv[mask], jonesdict)

def chisqdata_m(Obsdata, pol='I',**kwargs):
    debias = kwargs.get('debias',True)
    data_arr = Obsdata.unpack(['t1','t2','u','v','m','msigma','el1','el2','par_ang1','par_ang2'], conj=True, debias=False)
    uv = np.hstack((data_arr['u'].reshape(-1,1), data_arr['v'].reshape(-1,1)))
    mask = np.isfinite(data_arr['m'] + data_arr['msigma']) # don't include nan (missing data) or inf (division by zero)
    jonesdict = make_jonesdict(Obsdata, data_arr[mask])
    return (data_arr['m'][mask], data_arr['msigma'][mask], uv[mask], jonesdict)

def chisqdata_fracpol(Obsdata, pol_upper,pol_lower,**kwargs):
    debias = kwargs.get('debias',True)
    data_arr = Obsdata.unpack(['t1','t2','u','v','m','msigma','el1','el2','par_ang1','par_ang2','rrvis','rlvis','lrvis','llvis','rramp','rlamp','lramp','llamp','rrsigma','rlsigma','lrsigma','llsigma'], conj=False, debias=True)
    uv = np.hstack((data_arr['u'].reshape(-1,1), data_arr['v'].reshape(-1,1)))

    upper = data_arr[pol_upper + 'vis']
    lower = data_arr[pol_lower + 'vis']
    upper_amp = data_arr[pol_upper + 'amp']
    lower_amp = data_arr[pol_lower + 'amp']
    upper_sig = data_arr[pol_upper + 'sigma']
    lower_sig = data_arr[pol_lower + 'sigma']    

    sig = ((upper_sig/lower_amp)**2 + (lower_sig*upper_amp/lower_amp**2)**2)**0.5

    # Mask bad data
    mask = np.isfinite(upper + lower + sig) # don't include nan (missing data) or inf (division by zero)
    jonesdict = make_jonesdict(Obsdata, data_arr[mask])

    return ((upper/lower)[mask], sig[mask], uv[mask], jonesdict)

def chisqdata_polclosure(Obsdata, **kwargs):
    debias = kwargs.get('debias',True)
    data_arr = Obsdata.unpack(['t1','t2','u','v','m','msigma','el1','el2','par_ang1','par_ang2','rrvis','rlvis','lrvis','llvis','rramp','rlamp','lramp','llamp','rrsigma','rlsigma','lrsigma','llsigma'], conj=False, debias=True)
    uv = np.hstack((data_arr['u'].reshape(-1,1), data_arr['v'].reshape(-1,1)))

    RL = data_arr['rlvis']
    LR = data_arr['lrvis']
    RR = data_arr['rrvis']
    LL = data_arr['llvis']
    vis = (RL * LR)/(RR * LL)
    sig = (np.abs(LR/(LL*RR) * data_arr['rlsigma'])**2
          +np.abs(RL/(LL*RR) * data_arr['lrsigma'])**2
          +np.abs(LR*RL/(RR**2*LL) * data_arr['rrsigma'])**2
          +np.abs(RL*LR/(LL**2*RR) * data_arr['llsigma'])**2)**0.5

    # Mask bad data
    mask = np.isfinite(vis + sig) # don't include nan (missing data) or inf (division by zero)
    jonesdict = make_jonesdict(Obsdata, data_arr[mask])

    return (vis[mask], sig[mask], uv[mask], jonesdict)
