# model.py
# an interferometric model class

# To do:
# 1. Polarimetric models
# 2. Fix diagonal closure stuff

from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object

import numpy as np
import scipy.special as sps
import copy

import ehtim.observing.obs_simulate as simobs
import ehtim.observing.pulses

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *
from ehtim.modeling.modeling_utils import *

from ehtim.const_def import *

LINE_THICKNESS = 2 # Thickness of 1D models on the image, in pixels
FOV_DEFAULT = 100.*RADPERUAS
NPIX_DEFAULT = 256

###########################################################################################################################################
#Model object
###########################################################################################################################################

def model_params(model_type, model_params=None):
    """Return the ordered list of model parameters for a specified model type. This order must match that of the gradient function, sample_1model_grad_uv.
    """

    if model_type == 'point':
        params = ['F0','x0','y0']
    elif model_type == 'circ_gauss':
        params = ['F0','FWHM','x0','y0']
    elif model_type == 'gauss':
        params = ['F0','FWHM_maj','FWHM_min','PA','x0','y0']
    elif model_type == 'disk':
        params = ['F0','d','x0','y0']
    elif model_type == 'ring':
        params = ['F0','d','x0','y0']
    elif model_type == 'stretched_ring':
        params = ['F0','d','x0','y0','stretch','stretch_PA']
    elif model_type == 'thick_ring':
        params = ['F0','d','alpha','x0','y0']
    elif model_type == 'stretched_thick_ring':
        params = ['F0','d','alpha','x0','y0','stretch','stretch_PA']
    elif model_type == 'mring':
        params = ['F0','d','x0','y0']
        for j in range(len(model_params['beta_list'])):
            params.append('beta' + str(j+1) + '_re')
            params.append('beta' + str(j+1) + '_im')
    elif model_type == 'stretched_mring':
        params = ['F0','d','x0','y0']
        for j in range(len(model_params['beta_list'])):
            params.append('beta' + str(j+1) + '_re')
            params.append('beta' + str(j+1) + '_im')
        params.append('stretch')
        params.append('stretch_PA')
    elif model_type == 'thick_mring':
        params = ['F0','d','alpha', 'x0','y0']
        for j in range(len(model_params['beta_list'])):
            params.append('beta' + str(j+1) + '_re')
            params.append('beta' + str(j+1) + '_im')
    elif model_type == 'stretched_thick_mring':
        params = ['F0','d','alpha', 'x0','y0']
        for j in range(len(model_params['beta_list'])):
            params.append('beta' + str(j+1) + '_re')
            params.append('beta' + str(j+1) + '_im')
        params.append('stretch')
        params.append('stretch_PA')
    else:
        print('Model ' + model_init.models[j] + ' not recognized.')
        params = []

    return params

def default_prior(model_type,model_params=None):
    """Return the default model prior and transformation for a specified model type
    """

    prior = {'F0':{'prior_type':'none','transform':'log'}, 
             'x0':{'prior_type':'none'}, 
             'y0':{'prior_type':'none'}}
    if model_type == 'point':
        pass
    elif model_type == 'circ_gauss':
        prior['FWHM'] = {'prior_type':'none','transform':'log'}
    elif model_type == 'gauss':
        prior['FWHM_maj'] = {'prior_type':'positive','transform':'log'}
        prior['FWHM_min'] = {'prior_type':'positive','transform':'log'}
        prior['PA'] = {'prior_type':'none'}
    elif model_type == 'disk':
        prior['d'] = {'prior_type':'positive','transform':'log'}
    elif model_type == 'ring':
        prior['d'] = {'prior_type':'positive','transform':'log'}
    elif model_type == 'stretched_ring':
        prior['d'] = {'prior_type':'positive','transform':'log'}
        prior['stretch'] = {'prior_type':'positive','transform':'log'}
        prior['stretch_PA'] = {'prior_type':'none'}
    elif model_type == 'thick_ring':
        prior['d'] = {'prior_type':'positive','transform':'log'}
        prior['alpha'] = {'prior_type':'positive','transform':'log'}
    elif model_type == 'stretched_thick_ring':
        prior['d'] = {'prior_type':'positive','transform':'log'}
        prior['alpha'] = {'prior_type':'positive','transform':'log'}
        prior['stretch'] = {'prior_type':'positive','transform':'log'}
        prior['stretch_PA'] = {'prior_type':'none'}
    elif model_type == 'mring':
        prior['d'] = {'prior_type':'positive','transform':'log'}
        for j in range(len(model_params['beta_list'])):
            prior['beta' + str(j+1) + '_re'] = {'prior_type':'flat','min':-0.5,'max':0.5}
            prior['beta' + str(j+1) + '_im'] = {'prior_type':'flat','min':-0.5,'max':0.5}
    elif model_type == 'stretched_mring':
        prior['d'] = {'prior_type':'positive','transform':'log'}
        for j in range(len(model_params['beta_list'])):
            prior['beta' + str(j+1) + '_re'] = {'prior_type':'flat','min':-0.5,'max':0.5}
            prior['beta' + str(j+1) + '_im'] = {'prior_type':'flat','min':-0.5,'max':0.5}
        prior['stretch'] = {'prior_type':'positive','transform':'log'}
        prior['stretch_PA'] = {'prior_type':'none'}
    elif model_type == 'thick_mring':
        prior['d'] = {'prior_type':'positive','transform':'log'}
        prior['alpha'] = {'prior_type':'positive','transform':'log'}
        for j in range(len(model_params['beta_list'])):
            prior['beta' + str(j+1) + '_re'] = {'prior_type':'flat','min':-0.5,'max':0.5}
            prior['beta' + str(j+1) + '_im'] = {'prior_type':'flat','min':-0.5,'max':0.5}
    elif model_type == 'stretched_thick_mring':
        prior['d'] = {'prior_type':'positive','transform':'log'}
        prior['alpha'] = {'prior_type':'positive','transform':'log'}
        for j in range(len(model_params['beta_list'])):
            prior['beta' + str(j+1) + '_re'] = {'prior_type':'flat','min':-0.5,'max':0.5}
            prior['beta' + str(j+1) + '_im'] = {'prior_type':'flat','min':-0.5,'max':0.5}
        prior['stretch'] = {'prior_type':'positive','transform':'log'}
        prior['stretch_PA'] = {'prior_type':'none'}
    else:
        print('Model not recognized!')

    return prior

def stretch_xy(x, y, params):
    x_stretch = ((x - params['x0']) * (np.cos(params['stretch_PA'])**2 + np.sin(params['stretch_PA'])**2 / params['stretch'])
               + (y - params['y0']) * np.cos(params['stretch_PA']) * np.sin(params['stretch_PA']) * (1.0/params['stretch'] - 1.0))
    y_stretch = ((y - params['y0']) * (np.cos(params['stretch_PA'])**2 / params['stretch'] + np.sin(params['stretch_PA'])**2)
               + (x - params['x0']) * np.cos(params['stretch_PA']) * np.sin(params['stretch_PA']) * (1.0/params['stretch'] - 1.0))
    return (params['x0'] + x_stretch,params['y0'] + y_stretch)

def stretch_uv(u, v, params):
    u_stretch = (u * (np.cos(params['stretch_PA'])**2 + np.sin(params['stretch_PA'])**2 * params['stretch'])
               + v * np.cos(params['stretch_PA']) * np.sin(params['stretch_PA']) * (params['stretch'] - 1.0))
    v_stretch = (v * (np.cos(params['stretch_PA'])**2 * params['stretch'] + np.sin(params['stretch_PA'])**2)
               + u * np.cos(params['stretch_PA']) * np.sin(params['stretch_PA']) * (params['stretch'] - 1.0))
    return (u_stretch,v_stretch)

def sample_1model_xy(x, y, model_type, params, psize=1.*RADPERUAS):
    if model_type == 'point':
        return params['F0'] * (np.abs( x - params['x0']) < psize/2.0) * (np.abs( y - params['y0']) < psize/2.0)
    elif model_type == 'circ_gauss':
        sigma = params['FWHM'] / (2. * np.sqrt(2. * np.log(2.)))
        return (params['F0']*psize**2 * 4.0 * np.log(2.)/(np.pi * params['FWHM']**2) * 
               np.exp(-((x - params['x0'])**2 + (y - params['y0'])**2)/(2*sigma**2)))
    elif model_type == 'gauss':
        sigma_maj = params['FWHM_maj'] / (2. * np.sqrt(2. * np.log(2.)))
        sigma_min = params['FWHM_min'] / (2. * np.sqrt(2. * np.log(2.)))
        cth = np.cos(params['PA'])
        sth = np.sin(params['PA'])
        return (params['F0']*psize**2 * 4.0 * np.log(2.)/(np.pi * params['FWHM_maj'] * params['FWHM_min']) * 
               np.exp(-((y - params['y0'])*np.cos(params['PA']) + (x - params['x0'])*np.sin(params['PA']))**2/(2*sigma_maj**2) +
                      -((x - params['x0'])*np.cos(params['PA']) - (y - params['y0'])*np.sin(params['PA']))**2/(2*sigma_min**2)))
    elif model_type == 'disk':
        return params['F0']*psize**2/(np.pi*params['d']**2/4.) * (np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2) < params['d']/2.0) 
    elif model_type == 'ring':
        return (params['F0']*psize**2/(np.pi*params['d']*psize*LINE_THICKNESS)
                * (params['d']/2.0 - psize*LINE_THICKNESS/2 < np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2))
                * (params['d']/2.0 + psize*LINE_THICKNESS/2 > np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2)))
    elif model_type == 'thick_ring':
        r = np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2)
        z = 4.*np.log(2.) * r * params['d']/params['alpha']**2
        return (params['F0']*psize**2 * 4.0 * np.log(2.)/(np.pi * params['alpha']**2)
                * np.exp(-4.*np.log(2.)/params['alpha']**2*(r**2 + params['d']**2/4.) + z)
                * sps.ive(0, z)) 
    elif model_type == 'mring':
        phi = np.angle((y - params['y0']) + 1j*(x - params['x0']))
        return (params['F0']*psize**2/(np.pi*params['d']*psize*LINE_THICKNESS)
                * (1.0 + np.sum([2.*np.real(params['beta_list'][m-1] * np.exp(1j * m * phi)) for m in range(1,len(params['beta_list'])+1)],axis=0))  
                * (params['d']/2.0 - psize*LINE_THICKNESS/2 < np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2))
                * (params['d']/2.0 + psize*LINE_THICKNESS/2 > np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2)))
    elif model_type == 'thick_mring':
        phi = np.angle((y - params['y0']) + 1j*(x - params['x0']))
        r = np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2)
        z = 4.*np.log(2.) * r * params['d']/params['alpha']**2
        return (params['F0']*psize**2 * 4.0 * np.log(2.)/(np.pi * params['alpha']**2)
                * np.exp(-4.*np.log(2.)/params['alpha']**2*(r**2 + params['d']**2/4.) + z)
                * (sps.ive(0, z) + np.sum([2.*np.real(sps.ive(m, z) * params['beta_list'][m-1] * np.exp(1j * m * phi)) for m in range(1,len(params['beta_list'])+1)],axis=0)))
    elif model_type[:9] == 'stretched':
        params_stretch = params.copy()
        params_stretch['F0'] /= params['stretch']
        return sample_1model_xy(*stretch_xy(x, y, params), model_type[10:], params_stretch, psize)
    else:
        print('Model ' + model_type + ' not recognized!')
        return 0.0

def sample_1model_uv(u, v, model_type, params):
    if model_type == 'point':
        return params['F0'] * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))
    elif model_type == 'circ_gauss':
        return (params['F0'] 
               * np.exp(-np.pi**2/(4.*np.log(2.)) * (u**2 + v**2) * params['FWHM']**2)
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
    elif model_type == 'gauss':
        u_maj = u*np.sin(params['PA']) + v*np.cos(params['PA'])
        u_min = u*np.cos(params['PA']) - v*np.sin(params['PA'])
        return (params['F0'] 
               * np.exp(-np.pi**2/(4.*np.log(2.)) * ((u_maj * params['FWHM_maj'])**2 + (u_min * params['FWHM_min'])**2))
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
    elif model_type == 'disk':
        z = np.pi * params['d'] * (u**2 + v**2)**0.5
        #Add a small offset to avoid issues with division by zero
        z += (z == 0.0) * 1e-10
        return (params['F0'] * 2.0/z * sps.jv(1, z) 
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
    elif model_type == 'ring':
        z = np.pi * params['d'] * (u**2 + v**2)**0.5
        return (params['F0'] * sps.jv(0, z) 
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
    elif model_type == 'thick_ring':
        z = np.pi * params['d'] * (u**2 + v**2)**0.5
        return (params['F0'] * sps.jv(0, z) 
               * np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
    elif model_type == 'mring':
        phi = np.angle(v + 1j*u)
        # Flip the baseline sign to match eht-imaging conventions
        phi += np.pi
        z = np.pi * params['d'] * (u**2 + v**2)**0.5
        return (params['F0'] * (sps.jv(0, z) 
               + np.sum([params['beta_list'][m-1]          * sps.jv( m, z) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
               + np.sum([np.conj(params['beta_list'][m-1]) * sps.jv(-m, z) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0))
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
    elif model_type == 'thick_mring':
        phi = np.angle(v + 1j*u)
        # Flip the baseline sign to match eht-imaging conventions
        phi += np.pi
        z = np.pi * params['d'] * (u**2 + v**2)**0.5
        return (params['F0'] * (sps.jv(0, z) 
               + np.sum([params['beta_list'][m-1] * sps.jv( m, z) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
               + np.sum([np.conj(params['beta_list'][m-1]) * sps.jv(-m, z) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0))
               * np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
    elif model_type[:9] == 'stretched':
        params_stretch = params.copy()        
        params_stretch['x0'] = 0.0
        params_stretch['y0'] = 0.0
        return sample_1model_uv(*stretch_uv(u,v,params), model_type[10:], params_stretch) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])) 
    else:
        print('Model ' + model_type + ' not recognized!')
        return 0.0

def sample_1model_graduv_uv(u, v, model_type, params):
    # Gradient of the visibility function, (dV/du, dV/dv)
    # This function makes it convenient to, e.g., compute gradients of stretched images and to compute the model centroid

    vis = sample_1model_uv(u, v, model_type, params)
    if model_type == 'point': 
        return np.array([ 1j * 2.0 * np.pi * params['x0'] * vis,
                          1j * 2.0 * np.pi * params['y0'] * vis])
    elif model_type == 'circ_gauss': 
        return np.array([ (1j * 2.0 * np.pi * params['x0'] - params['FWHM']**2 * np.pi**2 * u/(2. * np.log(2.))) * vis,
                          (1j * 2.0 * np.pi * params['y0'] - params['FWHM']**2 * np.pi**2 * v/(2. * np.log(2.))) * vis])                          
    elif model_type == 'gauss': 
        u_maj = u*np.sin(params['PA']) + v*np.cos(params['PA'])
        u_min = u*np.cos(params['PA']) - v*np.sin(params['PA'])
        return np.array([ (1j * 2.0 * np.pi * params['x0'] - params['FWHM_maj']**2 * np.pi**2 * u_maj/(2. * np.log(2.)) * np.sin(params['PA']) - params['FWHM_min']**2 * np.pi**2 * u_min/(2. * np.log(2.)) * np.cos(params['PA'])) * vis,
                          (1j * 2.0 * np.pi * params['y0'] - params['FWHM_maj']**2 * np.pi**2 * u_maj/(2. * np.log(2.)) * np.cos(params['PA']) + params['FWHM_min']**2 * np.pi**2 * u_min/(2. * np.log(2.)) * np.sin(params['PA'])) * vis])       
    elif model_type == 'disk': 
        # Take care of the degenerate origin point by a small offset
        u += (u==0.)*(v==0.)*1e-10
        uvdist = (u**2 + v**2)**0.5
        z = np.pi * params['d'] * uvdist
        bessel_deriv = 0.5 * (sps.jv( 0, z) - sps.jv( 2, z))
        return np.array([ (1j * 2.0 * np.pi * params['x0'] - u/uvdist**2) * vis 
                            + params['F0'] * 2./z * np.pi * params['d'] * u/uvdist * bessel_deriv * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])),
                          (1j * 2.0 * np.pi * params['y0'] - v/uvdist**2) * vis 
                            + params['F0'] * 2./z * np.pi * params['d'] * v/uvdist * bessel_deriv * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])) ])
    elif model_type == 'ring':
        # Take care of the degenerate origin point by a small offset
        u += (u==0.)*(v==0.)*1e-10
        uvdist = (u**2 + v**2)**0.5
        z = np.pi * params['d'] * uvdist
        return np.array([ 1j * 2.0 * np.pi * params['x0'] * vis 
                            - params['F0'] * np.pi*params['d']*u/uvdist * sps.jv(1, z) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])),
                          1j * 2.0 * np.pi * params['y0'] * vis 
                            - params['F0'] * np.pi*params['d']*v/uvdist * sps.jv(1, z) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])) ])
    elif model_type == 'thick_ring':
        uvdist = (u**2 + v**2)**0.5
        #Add a small offset to avoid issues with division by zero
        uvdist += (uvdist == 0.0) * 1e-10
        z = np.pi * params['d'] * uvdist
        return np.array([ (1j * 2.0 * np.pi * params['x0'] - params['alpha']**2 * np.pi**2 * u/(2. * np.log(2.))) * vis 
                        - params['F0'] * np.pi*params['d']*u/uvdist * sps.jv(1, z) 
                                * np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.))) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])),
                          (1j * 2.0 * np.pi * params['y0'] - params['alpha']**2 * np.pi**2 * v/(2. * np.log(2.))) * vis 
                        - params['F0'] * np.pi*params['d']*v/uvdist * sps.jv(1, z) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))
                                * np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.))) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])) ])
    elif model_type == 'mring': 
        # Take care of the degenerate origin point by a small offset
        u += (u==0.)*(v==0.)*1e-10
        phi = np.angle(v + 1j*u)
        # Flip the baseline sign to match eht-imaging conventions
        phi += np.pi
        uvdist = (u**2 + v**2)**0.5
        dphidu =  v/uvdist**2 
        dphidv = -u/uvdist**2 
        z = np.pi * params['d'] * uvdist
        return np.array([ 
                 1j * 2.0 * np.pi * params['x0'] * vis 
                    + params['F0'] * (-np.pi * params['d'] * u/uvdist * sps.jv(1, z)
                    + np.sum([params['beta_list'][m-1]          * sps.jv( m, z) * ( 1j * m * dphidu) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([np.conj(params['beta_list'][m-1]) * sps.jv(-m, z) * (-1j * m * dphidu) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([params['beta_list'][m-1]          * 0.5 * (sps.jv( m-1, z) - sps.jv( m+1, z)) * np.pi * params['d'] * u/uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([np.conj(params['beta_list'][m-1]) * 0.5 * (sps.jv(-m-1, z) - sps.jv(-m+1, z)) * np.pi * params['d'] * u/uvdist * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0))
                    * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])),
                 1j * 2.0 * np.pi * params['y0'] * vis 
                    + params['F0'] * (-np.pi * params['d'] * v/uvdist * sps.jv(1, z)
                    + np.sum([params['beta_list'][m-1]          * sps.jv( m, z) * ( 1j * m * dphidv) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([np.conj(params['beta_list'][m-1]) * sps.jv(-m, z) * (-1j * m * dphidv) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([params['beta_list'][m-1]          * 0.5 * (sps.jv( m-1, z) - sps.jv( m+1, z)) * np.pi * params['d'] * v/uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([np.conj(params['beta_list'][m-1]) * 0.5 * (sps.jv(-m-1, z) - sps.jv(-m+1, z)) * np.pi * params['d'] * v/uvdist * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0))
                    * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])) ])
    elif model_type == 'thick_mring':
        # Take care of the degenerate origin point by a small offset
        u += (u==0.)*(v==0.)*1e-10
        phi = np.angle(v + 1j*u)
        # Flip the baseline sign to match eht-imaging conventions
        phi += np.pi
        uvdist = (u**2 + v**2)**0.5
        dphidu =  v/uvdist**2
        dphidv = -u/uvdist**2
        z = np.pi * params['d'] * uvdist
        return np.array([ 
                 (1j * 2.0 * np.pi * params['x0'] - params['alpha']**2 * np.pi**2 * u/(2. * np.log(2.))) * vis 
                    + params['F0'] * (-np.pi * params['d'] * u/uvdist * sps.jv(1, z)
                    + np.sum([params['beta_list'][m-1]          * sps.jv( m, z) * ( 1j * m * dphidu) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([np.conj(params['beta_list'][m-1]) * sps.jv(-m, z) * (-1j * m * dphidu) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([params['beta_list'][m-1]          * 0.5 * (sps.jv( m-1, z) - sps.jv( m+1, z)) * np.pi * params['d'] * u/uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([np.conj(params['beta_list'][m-1]) * 0.5 * (sps.jv(-m-1, z) - sps.jv(-m+1, z)) * np.pi * params['d'] * u/uvdist * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0))
                    * np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))
                    * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])),
                 (1j * 2.0 * np.pi * params['y0'] - params['alpha']**2 * np.pi**2 * v/(2. * np.log(2.))) * vis 
                    + params['F0'] * (-np.pi * params['d'] * v/uvdist * sps.jv(1, z)
                    + np.sum([params['beta_list'][m-1]          * sps.jv( m, z) * ( 1j * m * dphidv) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([np.conj(params['beta_list'][m-1]) * sps.jv(-m, z) * (-1j * m * dphidv) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([params['beta_list'][m-1]          * 0.5 * (sps.jv( m-1, z) - sps.jv( m+1, z)) * np.pi * params['d'] * v/uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([np.conj(params['beta_list'][m-1]) * 0.5 * (sps.jv(-m-1, z) - sps.jv(-m+1, z)) * np.pi * params['d'] * v/uvdist * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0))
                    * np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))
                    * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])) ])
    elif model_type[:9] == 'stretched':
        # Take care of the degenerate origin point by a small offset
        u += (u==0.)*(v==0.)*1e-10
        params_stretch = params.copy()        
        params_stretch['x0'] = 0.0
        params_stretch['y0'] = 0.0
        (u_stretch, v_stretch) = stretch_uv(u,v,params)
        
        # First calculate the gradient of the unshifted but stretched image
        grad0 = sample_1model_graduv_uv(u_stretch, v_stretch, model_type[10:], params_stretch) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))
        grad  = grad0.copy() * 0.0
        grad[0] = ( grad0[0] * (np.cos(params['stretch_PA'])**2 + np.sin(params['stretch_PA'])**2*params['stretch'])
                  + grad0[1] * ((params['stretch'] - 1.0) * np.cos(params['stretch_PA']) * np.sin(params['stretch_PA'])))
        grad[1] = ( grad0[1] * (np.cos(params['stretch_PA'])**2*params['stretch'] + np.sin(params['stretch_PA'])**2)
                  + grad0[0] * ((params['stretch'] - 1.0) * np.cos(params['stretch_PA']) * np.sin(params['stretch_PA'])))

        # Add the gradient term from the shift
        vis = sample_1model_uv(u_stretch, v_stretch, model_type[10:], params_stretch) 
        grad[0] += vis * 1j * 2.0 * np.pi * params['x0'] * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])) 
        grad[1] += vis * 1j * 2.0 * np.pi * params['y0'] * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])) 

        return grad              
    else:
        print('Model ' + model_type + ' not recognized!')
        return 0.0

def sample_1model_grad_uv(u, v, model_type, params):
    # Gradient of the model for each model parameter
    if model_type == 'point': # F0, x0, y0
        return np.array([ np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])),
                 1j * 2.0 * np.pi * u * params['F0'] * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])),
                 1j * 2.0 * np.pi * v * params['F0'] * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))])
    elif model_type == 'circ_gauss': # F0, FWHM, x0, y0
        gauss = (params['F0'] * np.exp(-np.pi**2/(4.*np.log(2.)) * (u**2 + v**2) * params['FWHM']**2)
                *np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])))
        return np.array([ 1.0/params['F0'] * gauss,
                         -np.pi**2/(2.*np.log(2.)) * (u**2 + v**2) * params['FWHM'] * gauss,
                          1j * 2.0 * np.pi * u * gauss,
                          1j * 2.0 * np.pi * v * gauss])                          
    elif model_type == 'gauss': # F0, FWHM_maj, FWHM_min, PA, x0, y0
        u_maj = u*np.sin(params['PA']) + v*np.cos(params['PA'])
        u_min = u*np.cos(params['PA']) - v*np.sin(params['PA'])
        vis = (params['F0'] 
               * np.exp(-np.pi**2/(4.*np.log(2.)) * ((u_maj * params['FWHM_maj'])**2 + (u_min * params['FWHM_min'])**2))
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
        return np.array([ 1.0/params['F0'] * vis,
                         -np.pi**2/(2.*np.log(2.)) * params['FWHM_maj'] * u_maj**2 * vis,
                         -np.pi**2/(2.*np.log(2.)) * params['FWHM_min'] * u_min**2 * vis,
                         -np.pi**2/(2.*np.log(2.)) * (params['FWHM_maj']**2 - params['FWHM_min']**2) * u_maj * u_min * vis,
                          1j * 2.0 * np.pi * u * vis,
                          1j * 2.0 * np.pi * v * vis])    
    elif model_type == 'disk': # F0, d, x0, y0
        z = np.pi * params['d'] * (u**2 + v**2)**0.5
        #Add a small offset to avoid issues with division by zero
        z += (z == 0.0) * 1e-10
        vis = (params['F0'] * 2.0/z * sps.jv(1, z) 
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
        return np.array([ 1.0/params['F0'] * vis,
                         -(params['F0'] * 2.0/z * sps.jv(2, z) * np.pi * (u**2 + v**2)**0.5 * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) ,
                          1j * 2.0 * np.pi * u * vis,
                          1j * 2.0 * np.pi * v * vis])    
    elif model_type == 'ring': # F0, d, x0, y0
        z = np.pi * params['d'] * (u**2 + v**2)**0.5
        return np.array([ sps.jv(0, z) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])), 
                -np.pi * (u**2 + v**2)**0.5 * params['F0'] * sps.jv(1, z) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])), 
                 2.0 * np.pi * 1j * u * params['F0'] * sps.jv(0, z) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])), 
                 2.0 * np.pi * 1j * v * params['F0'] * sps.jv(0, z) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))])
    elif model_type == 'thick_ring': # F0, d, alpha, x0, y0
        z = np.pi * params['d'] * (u**2 + v**2)**0.5
        vis = (params['F0'] * sps.jv(0, z) 
               * np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
        return np.array([ 1.0/params['F0'] * vis,
                         -(params['F0'] * np.pi * (u**2 + v**2)**0.5 * sps.jv(1, z) 
                            * np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))
                            * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))),
                         -np.pi**2 * (u**2 + v**2) * params['alpha']/(2.*np.log(2.)) * vis,
                          1j * 2.0 * np.pi * u * vis,
                          1j * 2.0 * np.pi * v * vis])    
    elif model_type == 'mring': # F0, d, x0, y0, beta1_re, beta1_im, beta2_re, beta2_im, ...
        phi = np.angle(v + 1j*u)
        # Flip the baseline sign to match eht-imaging conventions
        phi += np.pi
        uvdist = (u**2 + v**2)**0.5
        z = np.pi * params['d'] * uvdist
        vis = (params['F0'] * (sps.jv(0, z) 
               + np.sum([params['beta_list'][m-1]          * sps.jv( m, z) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
               + np.sum([np.conj(params['beta_list'][m-1]) * sps.jv(-m, z) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0))
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
        grad = [ 1.0/params['F0'] * vis, 
                (params['F0'] * (-np.pi * uvdist * sps.jv(1, z) 
               + np.sum([params['beta_list'][m-1]          * 0.5 * (sps.jv( m-1, z) - sps.jv( m+1, z)) * np.pi * uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
               + np.sum([np.conj(params['beta_list'][m-1]) * 0.5 * (sps.jv( -m-1, z) - sps.jv( -m+1, z)) * np.pi * uvdist * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0))
                * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))),
                 1j * 2.0 * np.pi * u * vis, 
                 1j * 2.0 * np.pi * v * vis]
        # Add derivatives of the beta terms 
        for m in range(1,len(params['beta_list'])+1):
            beta_grad_re = params['F0'] * (
               sps.jv( m, z) * np.exp( 1j * m * (phi - np.pi/2.)) + sps.jv(-m, z) * np.exp(-1j * m * (phi - np.pi/2.)) 
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
            beta_grad_im = params['F0'] * (
               sps.jv( m, z) * np.exp( 1j * m * (phi - np.pi/2.)) - sps.jv(-m, z) * np.exp(-1j * m * (phi - np.pi/2.)) 
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
            grad.append(beta_grad_re)
            grad.append(1j * beta_grad_im)
        return np.array(grad)
    elif model_type == 'thick_mring': # F0, d, alpha, x0, y0, beta1_re, beta1_im, beta2_re, beta2_im, ...
        phi = np.angle(v + 1j*u)
        # Flip the baseline sign to match eht-imaging conventions
        phi += np.pi
        uvdist = (u**2 + v**2)**0.5
        z = np.pi * params['d'] * uvdist
        vis = (params['F0'] * (sps.jv(0, z) 
               + np.sum([params['beta_list'][m-1]          * sps.jv( m, z) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
               + np.sum([np.conj(params['beta_list'][m-1]) * sps.jv(-m, z) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0))
               * np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
        grad = [ 1.0/params['F0'] * vis, 
                 (params['F0'] * (-np.pi * uvdist * sps.jv(1, z) 
               + np.sum([params['beta_list'][m-1]          * 0.5 * (sps.jv( m-1, z) - sps.jv( m+1, z)) * np.pi * uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
               + np.sum([np.conj(params['beta_list'][m-1]) * 0.5 * (sps.jv(-m-1, z) - sps.jv(-m+1, z)) * np.pi * uvdist * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0))
               * np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))),
                -np.pi**2/(2.*np.log(2)) * uvdist**2 * params['alpha'] * vis, 
                 1j * 2.0 * np.pi * u * vis, 
                 1j * 2.0 * np.pi * v * vis]
        # Add derivatives of the beta terms 
        for m in range(1,len(params['beta_list'])+1):
            beta_grad_re = (params['F0'] * (sps.jv( m, z) * np.exp( 1j * m * (phi - np.pi/2.)) + sps.jv(-m, z) * np.exp(-1j * m * (phi - np.pi/2.)))
               * np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])))  
            beta_grad_im = (params['F0'] * (sps.jv( m, z) * np.exp( 1j * m * (phi - np.pi/2.)) - sps.jv(-m, z) * np.exp(-1j * m * (phi - np.pi/2.)))
               * np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
            grad.append(beta_grad_re)
            grad.append(1j * beta_grad_im)
        return np.array(grad)
    elif model_type[:9] == 'stretched':
        # Start with the model visibility
        vis  = sample_1model_uv(u, v, model_type, params)

        # Next, calculate the gradient wrt model parameters other than stretch and stretch_PA
        # These are the same as the gradient of the unstretched model on stretched baselines
        params_stretch = params.copy()        
        params_stretch['x0'] = 0.0
        params_stretch['y0'] = 0.0
        (u_stretch, v_stretch) = stretch_uv(u,v,params)
        grad = (sample_1model_grad_uv(u_stretch, v_stretch, model_type[10:], params_stretch)
                * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])))

        # Add the gradient terms for the centroid
        grad[model_params(model_type, params).index('x0')] = 1j * 2.0 * np.pi * u * vis
        grad[model_params(model_type, params).index('y0')] = 1j * 2.0 * np.pi * v * vis

        # Now calculate the gradient with respect to stretch and stretch PA
        grad_uv = sample_1model_graduv_uv(u_stretch, v_stretch, model_type[10:], params_stretch)
        #grad_uv = sample_1model_graduv_uv(u, v, model_type, params)                   
        grad_stretch = grad_uv.copy() * 0.0 
        grad_stretch[0] = ( grad_uv[0] * (u * np.sin(params['stretch_PA'])**2 + v * np.sin(params['stretch_PA']) * np.cos(params['stretch_PA']))
                          + grad_uv[1] * (v * np.cos(params['stretch_PA'])**2 + u * np.sin(params['stretch_PA']) * np.cos(params['stretch_PA'])))
        grad_stretch[0] *= np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))

        grad_stretch[1] = ( grad_uv[0] * (params['stretch'] - 1.0) * ( u * np.sin(2.0 * params['stretch_PA']) + v * np.cos(2.0 * params['stretch_PA']))
                          + grad_uv[1] * (params['stretch'] - 1.0) * (-v * np.sin(2.0 * params['stretch_PA']) + u * np.cos(2.0 * params['stretch_PA'])))
        grad_stretch[1] *= np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))
#        grad  = grad0.copy() * 0.0
#        grad[0] = ( grad0[0] * (np.cos(params['stretch_PA'])**2 + np.sin(params['stretch_PA'])**2*params['stretch'])
#                  + grad0[1] * ((params['stretch'] - 1.0) * np.cos(params['stretch_PA']) * np.sin(params['stretch_PA'])))
#        grad[1] = ( grad0[1] * (np.cos(params['stretch_PA'])**2*params['stretch'] + np.sin(params['stretch_PA'])**2)
#                  + grad0[0] * ((params['stretch'] - 1.0) * np.cos(params['stretch_PA']) * np.sin(params['stretch_PA'])))

#        # Add the gradient term from the shift
#        vis = sample_1model_uv(u_stretch, v_stretch, model_type[10:], params_stretch) 
#        grad[0] += vis * 1j * 2.0 * np.pi * params['x0'] * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])) 
#        grad[1] += vis * 1j * 2.0 * np.pi * params['y0'] * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])) 

        return np.concatenate([grad, grad_stretch])
    else:
        print('Model ' + model_type + ' not recognized!')
        return 0.0

def sample_model_xy(models, params, x, y, psize=1.*RADPERUAS):
    return np.sum(sample_1model_xy(x, y, models[j], params[j], psize=psize) for j in range(len(models)))

def sample_model_uv(models, params, u, v):
    return np.sum(sample_1model_uv(u, v, models[j], params[j]) for j in range(len(models)))

def sample_model_graduv_uv(models, params, u, v):
    # Gradient of a sum of models wrt (u,v)
    return np.sum([sample_1model_graduv_uv(u, v, models[j], params[j]) for j in range(len(models))],axis=0)

def sample_model_grad_uv(models, params, u, v):
    # Gradient of a sum of models for each parameter
    return np.concatenate([sample_1model_grad_uv(u, v, models[j], params[j]) for j in range(len(models))])

def blur_circ_1model(model_type, params, fwhm):
    """Blur a single model, returning new model type and associated parameters

       Args:
            fwhm (float) : Full width at half maximum of the kernel (radians)

       Returns:
            (dict) : Dictionary with new 'model_type' and new 'params'
    """

    model_type_blur = model_type
    params_blur = params.copy()

    if model_type == 'point':
        model_type_blur = 'circ_gauss'
        params_blur['FWHM'] = fwhm
    elif model_type == 'circ_gauss':
        params_blur['FWHM'] = (params_blur['FWHM']**2 + fwhm**2)**0.5
    elif model_type == 'gauss':
        params_blur['FWHM_maj'] = (params_blur['FWHM_maj']**2 + fwhm**2)**0.5
        params_blur['FWHM_min'] = (params_blur['FWHM_min']**2 + fwhm**2)**0.5
    elif 'thick' in model_type:
        params_blur['alpha'] = (params_blur['alpha']**2 + fwhm**2)**0.5
    elif model_type == 'ring' or model_type == 'mring':
        model_type_blur = 'thick_' + model_type
        params_blur['alpha'] = fwhm
    elif model_type == 'stretched_ring' or model_type == 'stretched_mring':
        model_type_blur = 'stretched_thick_' + model_type[10:]
        params_blur['alpha'] = fwhm
    else:
        raise Exception("A blurred " + model_type + " is not yet a supported model!")
    
    return {'model_type':model_type_blur, 'params':params_blur}

class Model(object):
    """A model with analytic representations in the image and visibility domains.

       Attributes:
    """

    def __init__(self, ra=RA_DEFAULT, dec=DEC_DEFAULT, pa=0.0,
                       polrep='stokes', pol_prim=None,
                       rf=RF_DEFAULT, source=SOURCE_DEFAULT,
                       mjd=MJD_DEFAULT, time=0.):

        """A model with analytic representations in the image and visibility domains.

           Args:

           Returns:
        """

        # The model is a sum of component models, each defined by a tag and associated parameters
        self.pol_prim =  'I'
        self.polrep   = 'stokes'
        self._imdict = {'I':{'models':[],'params':[]},'Q':{'models':[],'params':[]},'U':{'models':[],'params':[]},'V':{'models':[],'params':[]}}

        # Save the image metadata
        self.ra  = float(ra)
        self.dec = float(dec)
        self.pa  = float(pa)
        self.rf  = float(rf)
        self.source = str(source)
        self.mjd = int(mjd)
        if time > 24:
            self.mjd += int((time - time % 24)/24)
            self.time = float(time % 24)
        else:
            self.time = time

    @property
    def models(self):
        return self._imdict[self.pol_prim]['models']

    @models.setter
    def models(self, model_list):
        self._imdict[self.pol_prim]['models'] = model_list

    @property
    def params(self):
        return self._imdict[self.pol_prim]['params']

    @params.setter
    def params(self, param_list):
        self._imdict[self.pol_prim]['params'] = param_list

    def copy(self):
        """Return a copy of the Model object.

           Args:

           Returns:
               (Model): copy of the Model.
        """
        out = Model(ra=self.ra, dec=self.dec, pa=self.pa, polrep=self.polrep, pol_prim=self.pol_prim,rf=self.rf,source=self.source,mjd=self.mjd,time=self.time)
        out.models = copy.deepcopy(self.models)
        out.params = copy.deepcopy(self.params.copy())
        return out

    def switch_polrep(self, polrep_out='stokes', pol_prim_out=None):

        """Return a new model with the polarization representation changed
           Args:
               polrep_out (str):  the polrep of the output data
               pol_prim_out (str): The default image: I,Q,U or V for Stokes, RR,LL,LR,RL for circ

           Returns:
               (Model): new Model object with potentially different polrep
        """

        # Note: this currently does nothing, but it is put here for compatibility with functions such as selfcal
        if polrep_out not in ['stokes','circ']:
            raise Exception("polrep_out must be either 'stokes' or 'circ'")
        if pol_prim_out is None:
            if polrep_out=='stokes': pol_prim_out = 'I'
            elif polrep_out=='circ': pol_prim_out = 'RR'

        return self.copy()

    def N_models(self):
        """Return the number of model components

           Args:

           Returns:
                (int): number of model components
        """
        return len(self.models)

    def total_flux(self):
        """Return the total flux of the model in Jy.

           Args:

           Returns:
                (float) : model total flux (Jy)
        """
        return np.sum([self.params[j]['F0'] for j in range(self.N_models())])

    def blur_circ(self, fwhm_i, fwhm_pol=0):
        """Return a new model, equal to the current one convolved with a circular Gaussian kernel

           Args:
                fwhm (float) : Full width at half maximum of the kernel (radians)
                fwhm_pol (float): Beam size for Stokes Q,U,V blurring kernel

           Returns:
                (Model) : Blurred model
        """

        out = self.copy()

        # Blur the primary image
        for j in range(len(out.models)):
            blur_model = blur_circ_1model(out.models[j], out.params[j], fwhm_i)
            out.models[j] = blur_model['model_type']
            out.params[j] = blur_model['params']

        # Blur all polarizations and copy over
        for pol in list(self._imdict.keys()):
            if pol==out.pol_prim: continue
            for j in range(len(out._imdict[pol]['models'])):
                blur_model = blur_circ_1model(out._imdict[pol]['models'][j], out._imdict[pol]['params'], fwhm_i)
                out._imdict[pol]['models'][j] = blur_model['model_type']
                out._imdict[pol]['models'][j] = blur_model['params']

        return out



    def add_point(self, F0 = 1.0, x0 = 0.0, y0 = 0.0):
        """Add a point source model.

           Args:
               F0 (float): The total flux of the point source (Jy)
               x0 (float): The x-coordinate (radians)
               y0 (float): The y-coordinate (radians)

           Returns:
                (Model): Updated Model
        """

        out = self.copy()
        out.models.append('point')
        out.params.append({'F0':F0,'x0':x0,'y0':y0})
        return out

    def add_circ_gauss(self, F0 = 1.0, FWHM = 50.*RADPERUAS, x0 = 0.0, y0 = 0.0):
        """Add a circular Gaussian model.

           Args:
               F0 (float): The total flux of the Gaussian (Jy)
               FWHM (float): The FWHM of the Gaussian (radians)
               x0 (float): The x-coordinate (radians)
               y0 (float): The y-coordinate (radians)

           Returns:
                (Model): Updated Model
        """
        out = self.copy()
        out.models.append('circ_gauss')
        out.params.append({'F0':F0,'FWHM':FWHM,'x0':x0,'y0':y0})
        return out

    def add_gauss(self, F0 = 1.0, FWHM_maj = 50.*RADPERUAS, FWHM_min = 50.*RADPERUAS, PA = 0.0, x0 = 0.0, y0 = 0.0):
        """Add an anisotropic Gaussian model.

           Args:
               F0 (float): The total flux of the Gaussian (Jy)
               FWHM_maj (float): The FWHM of the Gaussian major axis (radians)
               FWHM_min (float): The FWHM of the Gaussian minor axis (radians)
               PA (float): Position angle of the major axis, east of north (radians)
               x0 (float): The x-coordinate (radians)
               y0 (float): The y-coordinate (radians)

           Returns:
                (Model): Updated Model
        """
        out = self.copy()
        out.models.append('gauss')
        out.params.append({'F0':F0,'FWHM_maj':FWHM_maj,'FWHM_min':FWHM_min,'PA':PA,'x0':x0,'y0':y0})
        return out

    def add_disk(self, F0 = 1.0, d = 50.*RADPERUAS, x0 = 0.0, y0 = 0.0):
        """Add a circular disk model.

           Args:
               F0 (float): The total flux of the disk (Jy)
               d (float): The diameter (radians)
               x0 (float): The x-coordinate (radians)
               y0 (float): The y-coordinate (radians)

           Returns:
                (Model): Updated Model
        """
        out = self.copy()
        out.models.append('disk')
        out.params.append({'F0':F0,'d':d,'x0':x0,'y0':y0})
        return out

    def add_ring(self, F0 = 1.0, d = 50.*RADPERUAS, x0 = 0.0, y0 = 0.0):
        """Add a ring model with infinitesimal thickness.

           Args:
               F0 (float): The total flux of the ring (Jy)
               d (float): The diameter (radians)
               x0 (float): The x-coordinate (radians)
               y0 (float): The y-coordinate (radians)

           Returns:
                (Model): Updated Model
        """
        out = self.copy()
        out.models.append('ring')
        out.params.append({'F0':F0,'d':d,'x0':x0,'y0':y0})
        return out

    def add_stretched_ring(self, F0 = 1.0, d = 50.*RADPERUAS, x0 = 0.0, y0 = 0.0, stretch = 1.0, stretch_PA = 0.0):
        """Add a stretched ring model with infinitesimal thickness.

           Args:
               F0 (float): The total flux of the ring (Jy)
               d (float): The diameter (radians)
               x0 (float): The x-coordinate (radians)
               y0 (float): The y-coordinate (radians)
               stretch (float): The stretch to apply (1.0 = no stretch)
               stretch_PA (float): Position angle of the stretch, east of north (radians)

           Returns:
                (Model): Updated Model
        """
        out = self.copy()
        out.models.append('stretched_ring')
        out.params.append({'F0':F0,'d':d,'x0':x0,'y0':y0,'stretch':stretch,'stretch_PA':stretch_PA})
        return out

    def add_thick_ring(self, F0 = 1.0, d = 50.*RADPERUAS, alpha = 10.*RADPERUAS, x0 = 0.0, y0 = 0.0):
        """Add a ring model with finite thickness, determined by circular Gaussian convolution of a thin ring.
           For details, see Appendix G of https://iopscience.iop.org/article/10.3847/2041-8213/ab0e85/pdf

           Args:
               F0 (float): The total flux of the ring (Jy)
               d (float): The ring diameter (radians)
               alpha (float): The ring thickness (FWHM of Gaussian convolution) (radians)
               x0 (float): The x-coordinate (radians)
               y0 (float): The y-coordinate (radians)

           Returns:
                (Model): Updated Model
        """
        out = self.copy()
        out.models.append('thick_ring')
        out.params.append({'F0':F0,'d':d,'alpha':alpha,'x0':x0,'y0':y0})
        return out

    def add_stretched_thick_ring(self, F0 = 1.0, d = 50.*RADPERUAS, alpha = 10.*RADPERUAS, x0 = 0.0, y0 = 0.0, stretch = 1.0, stretch_PA = 0.0):
        """Add a ring model with finite thickness, determined by circular Gaussian convolution of a thin ring.
           For details, see Appendix G of https://iopscience.iop.org/article/10.3847/2041-8213/ab0e85/pdf

           Args:
               F0 (float): The total flux of the ring (Jy)
               d (float): The ring diameter (radians)
               alpha (float): The ring thickness (FWHM of Gaussian convolution) (radians)
               x0 (float): The x-coordinate (radians)
               y0 (float): The y-coordinate (radians)
               stretch (float): The stretch to apply (1.0 = no stretch)
               stretch_PA (float): Position angle of the stretch, east of north (radians)

           Returns:
                (Model): Updated Model
        """
        out = self.copy()
        out.models.append('stretched_thick_ring')
        out.params.append({'F0':F0,'d':d,'alpha':alpha,'x0':x0,'y0':y0,'stretch':stretch,'stretch_PA':stretch_PA})
        return out

    def add_mring(self, F0 = 1.0, d = 50.*RADPERUAS, x0 = 0.0, y0 = 0.0, beta_list = None):
        """Add a ring model with azimuthal brightness variations determined by a Fourier mode expansion.
           For details, see Eq. 18-20 of https://arxiv.org/abs/1907.04329

           Args:
               F0 (float): The total flux of the ring (Jy), which is also beta_0.
               d (float): The diameter (radians)
               x0 (float): The x-coordinate (radians)
               y0 (float): The y-coordinate (radians)
               beta_list (list): List of complex Fourier coefficients, [beta_1, beta_2, ...]. 
                                 Negative indices are determined by the condition beta_{-m} = beta_m*.
                                 Indices are all scaled by F0 = beta_0, so they are dimensionless. 
           Returns:
                (Model): Updated Model
        """
        out = self.copy()
        if beta_list is None:
            beta_list = [0.0]
        out.models.append('mring')
        out.params.append({'F0':F0,'d':d,'beta_list':beta_list,'x0':x0,'y0':y0})
        return out

    def add_stretched_mring(self, F0 = 1.0, d = 50.*RADPERUAS, x0 = 0.0, y0 = 0.0, beta_list = None, stretch = 1.0, stretch_PA = 0.0):
        """Add a stretched ring model with azimuthal brightness variations determined by a Fourier mode expansion.
           For details, see Eq. 18-20 of https://arxiv.org/abs/1907.04329

           Args:
               F0 (float): The total flux of the ring (Jy), which is also beta_0.
               d (float): The diameter (radians)
               x0 (float): The x-coordinate (radians)
               y0 (float): The y-coordinate (radians)
               beta_list (list): List of complex Fourier coefficients, [beta_1, beta_2, ...]. 
                                 Negative indices are determined by the condition beta_{-m} = beta_m*.
                                 Indices are all scaled by F0 = beta_0, so they are dimensionless. 
               stretch (float): The stretch to apply (1.0 = no stretch)
               stretch_PA (float): Position angle of the stretch, east of north (radians)
           Returns:
                (Model): Updated Model
        """
        out = self.copy()
        if beta_list is None:
            beta_list = [0.0]
        out.models.append('stretched_mring')
        out.params.append({'F0':F0,'d':d,'beta_list':beta_list,'x0':x0,'y0':y0,'stretch':stretch,'stretch_PA':stretch_PA})
        return out

    def add_thick_mring(self, F0 = 1.0, d = 50.*RADPERUAS, alpha = 10.*RADPERUAS, x0 = 0.0, y0 = 0.0, beta_list = None):
        """Add a ring model with azimuthal brightness variations determined by a Fourier mode expansion and thickness determined by circular Gaussian convolution.
           For details, see Eq. 18-20 of https://arxiv.org/abs/1907.04329
           The Gaussian convolution calculation is a trivial generalization of Appendix G of https://iopscience.iop.org/article/10.3847/2041-8213/ab0e85/pdf

           Args:
               F0 (float): The total flux of the ring (Jy), which is also beta_0.
               d (float): The ring diameter (radians)
               alpha (float): The ring thickness (FWHM of Gaussian convolution) (radians)
               x0 (float): The x-coordinate (radians)
               y0 (float): The y-coordinate (radians)
               beta_list (list): List of complex Fourier coefficients, [beta_1, beta_2, ...]. 
                                 Negative indices are determined by the condition beta_{-m} = beta_m*.
                                 Indices are all scaled by F0 = beta_0, so they are dimensionless. 
           Returns:
                (Model): Updated Model
        """
        out = self.copy()
        if beta_list is None:
            beta_list = [0.0]
        out.models.append('thick_mring')
        out.params.append({'F0':F0,'d':d,'beta_list':beta_list,'alpha':alpha,'x0':x0,'y0':y0})
        return out

    def add_stretched_thick_mring(self, F0 = 1.0, d = 50.*RADPERUAS, alpha = 10.*RADPERUAS, x0 = 0.0, y0 = 0.0, beta_list = None, stretch = 1.0, stretch_PA = 0.0):
        """Add a ring model with azimuthal brightness variations determined by a Fourier mode expansion and thickness determined by circular Gaussian convolution.
           For details, see Eq. 18-20 of https://arxiv.org/abs/1907.04329
           The Gaussian convolution calculation is a trivial generalization of Appendix G of https://iopscience.iop.org/article/10.3847/2041-8213/ab0e85/pdf

           Args:
               F0 (float): The total flux of the ring (Jy), which is also beta_0.
               d (float): The ring diameter (radians)
               alpha (float): The ring thickness (FWHM of Gaussian convolution) (radians)
               x0 (float): The x-coordinate (radians)
               y0 (float): The y-coordinate (radians)
               beta_list (list): List of complex Fourier coefficients, [beta_1, beta_2, ...]. 
                                 Negative indices are determined by the condition beta_{-m} = beta_m*.
                                 Indices are all scaled by F0 = beta_0, so they are dimensionless. 
               stretch (float): The stretch to apply (1.0 = no stretch)
               stretch_PA (float): Position angle of the stretch, east of north (radians)
           Returns:
                (Model): Updated Model
        """
        out = self.copy()
        if beta_list is None:
            beta_list = [0.0]
        out.models.append('stretched_thick_mring')
        out.params.append({'F0':F0,'d':d,'beta_list':beta_list,'alpha':alpha,'x0':x0,'y0':y0,'stretch':stretch,'stretch_PA':stretch_PA})
        return out

    def sample_xy(self, x, y, psize=1.*RADPERUAS):
        """Sample model image on the specified x and y coordinates

           Args:
               x (float): x coordinate (dimensionless)
               y (float): y coordinate (dimensionless)

           Returns:
               (float): Image brightness (Jy/radian^2)
        """  
        return sample_model_xy(self.models, self.params, x, y, psize=psize)

    def sample_uv(self, u, v):
        """Sample model visibility on the specified u and v coordinates

           Args:
               u (float): u coordinate (dimensionless)
               v (float): v coordinate (dimensionless)

           Returns:
               (complex): complex visibility (Jy)
        """   
        return sample_model_uv(self.models, self.params, u, v)

    def sample_graduv_uv(self, u, v):
        """Sample model visibility gradient on the specified u and v coordinates wrt (u,v)

           Args:
               u (float): u coordinate (dimensionless)
               v (float): v coordinate (dimensionless)

           Returns:
               (complex): complex visibility (Jy)
        """   
        return sample_model_graduv_uv(self.models, self.params, u, v)

    def sample_grad_uv(self, u, v):
        """Sample model visibility gradient on the specified u and v coordinates wrt all model parameters

           Args:
               u (float): u coordinate (dimensionless)
               v (float): v coordinate (dimensionless)

           Returns:
               (complex): complex visibility (Jy)
        """   
        return sample_model_grad_uv(self.models, self.params, u, v)

    def centroid(self, pol=None):
        """Compute the location of the image centroid (corresponding to the polarization pol)
           Note: This quantity is only well defined for total intensity

           Args:
                pol (str): The polarization for which to find the image centroid

           Returns:
               (np.array): centroid positions (x0,y0) in radians
        """

        if pol is None: pol=self.pol_prim
        if not (pol in list(self._imdict.keys())):
            raise Exception("for polrep==%s, pol must be in " % 
                             self.polrep + ",".join(list(self._imdict.keys())))

        return np.real(self.sample_graduv_uv(0,0)/(2.*np.pi*1j))/self.total_flux()

    def default_prior(self):
        return [default_prior(self.models[j],self.params[j]) for j in range(self.N_models())]        

    def display(self, fov=FOV_DEFAULT, npix=NPIX_DEFAULT, **kwargs):        
        return self.make_image(fov, npix, **kwargs).display(**kwargs)

    def make_image(self, fov, npix, ra=RA_DEFAULT, dec=DEC_DEFAULT, rf=RF_DEFAULT, source=SOURCE_DEFAULT,
               polrep='stokes', pol_prim=None, pulse=PULSE_DEFAULT,
               mjd=MJD_DEFAULT, time=0.):
        """Sample the model onto a square image.

           Args:
               fov (float): the field of view of each axis in radians
               npix (int): the number of pixels on each axis
               ra (float): The source Right Ascension in fractional hours
               dec (float): The source declination in fractional degrees
               rf (float): The image frequency in Hz

               source (str): The source name
               polrep (str): polarization representation, either 'stokes' or 'circ'
               pol_prim (str): The default image: I,Q,U or V for Stokes, RR,LL,LR,RL for Circular
               pulse (function): The function convolved with the pixel values for continuous image.

               mjd (int): The integer MJD of the image
               time (float): The observing time of the image (UTC hours)

           Returns:
               (Image): an image object
        """

        pdim = fov/float(npix)
        npix = int(npix)
        imarr = np.zeros((npix,npix))
        outim = image.Image(imarr, pdim, ra, dec,
                      polrep=polrep, pol_prim=pol_prim,
                      rf=rf, source=source, mjd=mjd, time=time, pulse=pulse)
        
        return self.image_same(outim)

    def image_same(self, im):
        """Create an image of the model with parameters equal to a reference image.

           Args:
               im (Image): the reference image

           Returns:
               (Image): image of the model
        """        
        out = im.copy()
        xlist = np.arange(0,-im.xdim,-1)*im.psize + (im.psize*im.xdim)/2.0 - im.psize/2.0
        ylist = np.arange(0,-im.ydim,-1)*im.psize + (im.psize*im.ydim)/2.0 - im.psize/2.0

        x_grid, y_grid = np.meshgrid(xlist, ylist)
        imarr = self.sample_xy(x_grid, y_grid, im.psize)    

        out.imvec = imarr.flatten() # Change this to init with image_args 

        return out

    def observe_same_nonoise(self, obs, **kwargs):
        """Observe the model on the same baselines as an existing observation, without noise.

           Args:
               obs (Obsdata): the existing observation 

           Returns:
               (Obsdata): an observation object with no noise
        """

        # Copy data to be safe
        obsdata = copy.deepcopy(obs.data)

        # Compute visibilities
        data = self.sample_uv(obs.data['u'], obs.data['v'])

        # put visibilities into the obsdata
        if obs.polrep=='stokes':
            obsdata['vis'] = data
        elif obs.polrep=='circ':
            obsdata['rrvis'] = data

        obs_no_noise = ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata, obs.tarr,
                                             source=obs.source, mjd=obs.mjd, polrep=obs.polrep,
                                             ampcal=True, phasecal=True, opacitycal=True,
                                             dcal=True, frcal=True,
                                             timetype=obs.timetype, scantable=obs.scans)

        return obs_no_noise

    def observe_same(self, obs_in, add_th_noise=True, sgrscat=False, ttype=False, # Note: sgrscat and ttype are kept for consistency with comp_plots
                           opacitycal=True, ampcal=True, phasecal=True, 
                           dcal=True, frcal=True, rlgaincal=True,
                           stabilize_scan_phase=False, stabilize_scan_amp=False, neggains=False,
                           jones=False, inv_jones=False,
                           tau=TAUDEF, taup=GAINPDEF,
                           gain_offset=GAINPDEF, gainp=GAINPDEF,
                           dterm_offset=DTERMPDEF, caltable_path=None, seed=False, **kwargs):

        """Observe the image on the same baselines as an existing observation object and add noise.

           Args:
               obs_in (Obsdata): the existing observation 

               add_th_noise (bool): if True, baseline-dependent thermal noise is added
               opacitycal (bool): if False, time-dependent gaussian errors are added to opacities
               ampcal (bool): if False, time-dependent gaussian errors are added to station gains
               phasecal (bool): if False, time-dependent station-based random phases are added
               frcal (bool): if False, feed rotation angle terms are added to Jones matrices. 
               dcal (bool): if False, time-dependent gaussian errors added to D-terms. 

               stabilize_scan_phase (bool): if True, random phase errors are constant over scans
               stabilize_scan_amp (bool): if True, random amplitude errors are constant over scans
               neggains (bool): if True, force the applied gains to be <1
                                meaning that you have overestimated your telescope's performance


               jones (bool): if True, uses Jones matrix to apply mis-calibration effects 
               inv_jones (bool): if True, applies estimated inverse Jones matrix 
                                 (not including random terms) to a priori calibrate data

               tau (float): the base opacity at all sites, 
                            or a dict giving one opacity per site
               taup (float): the fractional std. dev. of the random error on the opacities
               gainp (float): the fractional std. dev. of the random error on the gains
               gain_offset (float): the base gain offset at all sites, 
                                    or a dict giving one offset per site
               dterm_offset (float): the base dterm offset at all sites, 
                                     or a dict giving one dterm offset per site

               caltable_path (string): The path and prefix of a saved caltable

               seed (int): seeds the random component of the noise terms. DO NOT set to 0!

           Returns:
               (Obsdata): an observation object
        """

        if seed!=False:
            np.random.seed(seed=seed)

        obs = self.observe_same_nonoise(obs_in)

        # Jones Matrix Corruption & Calibration
        if jones:
            obsdata = simobs.add_jones_and_noise(obs, add_th_noise=add_th_noise,
                                                 opacitycal=opacitycal, ampcal=ampcal,
                                                 phasecal=phasecal, dcal=dcal, frcal=frcal, 
                                                 rlgaincal=rlgaincal,
                                                 stabilize_scan_phase=stabilize_scan_phase,
                                                 stabilize_scan_amp=stabilize_scan_amp, 
                                                 neggains=neggains,
                                                 gainp=gainp, taup=taup, gain_offset=gain_offset,
                                                 dterm_offset=dterm_offset,
                                                 caltable_path=caltable_path, seed=seed)

            obs =  ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata, obs.tarr,
                                         source=obs.source, mjd=obs.mjd, polrep=obs_in.polrep,
                                         ampcal=ampcal, phasecal=phasecal, opacitycal=opacitycal, 
                                         dcal=dcal, frcal=frcal,
                                         timetype=obs.timetype, scantable=obs.scans)

            if inv_jones:
                obsdata = simobs.apply_jones_inverse(obs, 
                                                     opacitycal=opacitycal, dcal=dcal, frcal=frcal)

                obs =  ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata, obs.tarr,
                                             source=obs.source, mjd=obs.mjd, polrep=obs_in.polrep,
                                             ampcal=ampcal, phasecal=phasecal,
                                             opacitycal=True, dcal=True, frcal=True,
                                             timetype=obs.timetype, scantable=obs.scans)
                                             #these are always set to True after inverse jones call


        # No Jones Matrices, Add noise the old way
        # NOTE There is an asymmetry here - in the old way, we don't offer the ability to *not*  
        # unscale estimated noise.
        else:

            if caltable_path:
                print('WARNING: the caltable is only saved if you apply noise with a Jones Matrix')

            obsdata = simobs.add_noise(obs, add_th_noise=add_th_noise,
                                       ampcal=ampcal, phasecal=phasecal, opacitycal=opacitycal,
                                       stabilize_scan_phase=stabilize_scan_phase,
                                       stabilize_scan_amp=stabilize_scan_amp,
                                       gainp=gainp, taup=taup, gain_offset=gain_offset, seed=seed)

            obs =  ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata, obs.tarr,
                                         source=obs.source, mjd=obs.mjd, polrep=obs_in.polrep,
                                         ampcal=ampcal, phasecal=phasecal,
                                         opacitycal=True, dcal=True, frcal=True,
                                         timetype=obs.timetype, scantable=obs.scans)
                                         #these are always set to True after inverse jones call

        return obs

    def observe(self, array, tint, tadv, tstart, tstop, bw,
                      mjd=None, timetype='UTC', polrep_obs=None,
                      elevmin=ELEV_LOW, elevmax=ELEV_HIGH,
                      fix_theta_GMST=False, add_th_noise=True,
                      opacitycal=True, ampcal=True, phasecal=True, 
                      dcal=True, frcal=True, rlgaincal=True,
                      stabilize_scan_phase=False, stabilize_scan_amp=False,
                      jones=False, inv_jones=False,
                      tau=TAUDEF, taup=GAINPDEF,
                      gainp=GAINPDEF, gain_offset=GAINPDEF,
                      dterm_offset=DTERMPDEF, seed=False, **kwargs):

        """Generate baselines from an array object and observe the image.

           Args:
               array (Array): an array object containing sites with which to generate baselines
               tint (float): the scan integration time in seconds
               tadv (float): the uniform cadence between scans in seconds
               tstart (float): the start time of the observation in hours
               tstop (float): the end time of the observation in hours
               bw (float): the observing bandwidth in Hz

               mjd (int): the mjd of the observation, if set as different from the image mjd
               timetype (str): how to interpret tstart and tstop; either 'GMST' or 'UTC'
               elevmin (float): station minimum elevation in degrees
               elevmax (float): station maximum elevation in degrees

               polrep_obs (str): 'stokes' or 'circ' sets the data polarimetric representation

               fix_theta_GMST (bool): if True, stops earth rotation to sample fixed u,v 
               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A*  kernel
               add_th_noise (bool): if True, baseline-dependent thermal noise is added 
               opacitycal (bool): if False, time-dependent gaussian errors are added to opacities
               ampcal (bool): if False, time-dependent gaussian errors are added to station gains
               phasecal (bool): if False, time-dependent station-based random phases are added 
               frcal (bool): if False, feed rotation angle terms are added to Jones matrices. 
               dcal (bool): if False, time-dependent gaussian errors added to Jones matrices D-terms. 

               stabilize_scan_phase (bool): if True, random phase errors are constant over scans
               stabilize_scan_amp (bool): if True, random amplitude errors are constant over scans
               jones (bool): if True, uses Jones matrix to apply mis-calibration effects 
                             otherwise uses old formalism without D-terms
               inv_jones (bool): if True, applies estimated inverse Jones matrix 
                                 (not including random terms) to calibrate data

               tau (float): the base opacity at all sites, 
                            or a dict giving one opacity per site
               taup (float): the fractional std. dev. of the random error on the opacities
               gain_offset (float): the base gain offset at all sites, 
                                    or a dict giving one gain offset per site
               gainp (float): the fractional std. dev. of the random error on the gains

               dterm_offset (float): the base dterm offset at all sites, 
                                     or a dict giving one dterm offset per site

               seed (int): seeds the random component of noise added. DO NOT set to 0!

           Returns:
               (Obsdata): an observation object
        """

        # Generate empty observation
        print("Generating empty observation file . . . ")

        if mjd == None:
            mjd = self.mjd
        if polrep_obs is None:
            polrep_obs=self.polrep

        obs = array.obsdata(self.ra, self.dec, self.rf, bw, tint, tadv, tstart, tstop, mjd=mjd, 
                            polrep=polrep_obs, tau=tau, timetype=timetype, 
                            elevmin=elevmin, elevmax=elevmax, fix_theta_GMST=fix_theta_GMST)

        # Observe on the same baselines as the empty observation and add noise
        obs = self.observe_same(obs, add_th_noise=add_th_noise, 
                                     opacitycal=opacitycal,ampcal=ampcal,
                                     phasecal=phasecal,dcal=dcal,
                                     frcal=frcal, rlgaincal=rlgaincal,
                                     stabilize_scan_phase=stabilize_scan_phase,
                                     stabilize_scan_amp=stabilize_scan_amp,
                                     gainp=gainp,gain_offset=gain_offset,
                                     tau=tau, taup=taup,
                                     dterm_offset=dterm_offset,
                                     jones=jones, inv_jones=inv_jones, seed=seed)

        obs.mjd = mjd

        return obs

    def save_txt(self,filename):
        # Note: need to save extra information (ra, dec, etc.)
        out = []
        for j in range(self.N_models()):
            out.append(self.models[j])
            out.append(str(self.params[j]))
        np.savetxt(filename, out, fmt="%s")

    def load_txt(self,filename):
        # Note: need to load extra information (ra, dec, etc.)
        lines = open(filename).read().splitlines()
        self.models = lines[::2]
        self.params = [eval(x) for x in lines[1::2]]

def load_txt(filename):
    # Note: need to load extra information (ra, dec, etc.)
    out = Model()

    lines = open(filename).read().splitlines()
    out.models = lines[::2]
    out.params = [eval(x) for x in lines[1::2]]
    return out
