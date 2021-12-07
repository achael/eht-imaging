# model.py
# an interferometric model class

from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object

import numpy as np
import scipy.special as sps
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import copy

import ehtim.observing.obs_simulate as simobs
import ehtim.observing.pulses

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *
#from ehtim.modeling.modeling_utils import *

import ehtim.image as image

from ehtim.const_def import *

LINE_THICKNESS = 2 # Thickness of 1D models on the image, in pixels
FOV_DEFAULT = 100.*RADPERUAS
NPIX_DEFAULT = 256
COMPLEX_BASIS = 'abs-arg' # Basis for representing (most) complex quantities: 'abs-arg' or 're-im'

###########################################################################################################################################
#Model object
###########################################################################################################################################

def model_params(model_type, model_params=None, fit_pol=False, fit_cpol=False):
    """Return the ordered list of model parameters for a specified model type. This order must match that of the gradient function, sample_1model_grad_uv.
    """

    if COMPLEX_BASIS == 're-im':
        complex_labels = ['_re','_im']
    elif COMPLEX_BASIS == 'abs-arg':
        complex_labels = ['_abs','_arg']
    else:
        raise Exception('COMPLEX_BASIS ' + COMPLEX_BASIS + ' not recognized!')

    params = []

    # Function to add polarimetric parameters; these must be added before stretch parameters
    def add_pol():
        if fit_pol:
            if model_type.find('mring') == -1:
                params.append('pol_frac')
                params.append('pol_evpa')
            else:            
                for j in range(-(len(model_params['beta_list_pol'])-1)//2,(len(model_params['beta_list_pol'])+1)//2):
                    params.append('betapol' + str(j) + complex_labels[0])
                    params.append('betapol' + str(j) + complex_labels[1])
        if fit_cpol:
            if model_type.find('mring') == -1:
                params.append('cpol_frac')
            else:
                for j in range(len(model_params['beta_list_cpol'])):
                    if j==0:
                        params.append('betacpol0')
                    else:
                        params.append('betacpol' + str(j) + complex_labels[0])
                        params.append('betacpol' + str(j) + complex_labels[1])

    if model_type == 'point':
        params = ['F0','x0','y0']
        add_pol()
    elif model_type == 'circ_gauss':
        params = ['F0','FWHM','x0','y0']
        add_pol()
    elif model_type == 'gauss':
        params = ['F0','FWHM_maj','FWHM_min','PA','x0','y0']
        add_pol()
    elif model_type == 'disk':
        params = ['F0','d','x0','y0']
        add_pol()
    elif model_type == 'blurred_disk':
        params = ['F0','d','alpha','x0','y0']
        add_pol()
    elif model_type == 'crescent':
        params = ['F0','d', 'fr', 'fo', 'ff', 'phi','x0','y0']
        add_pol()
    elif model_type == 'blurred_crescent':
        params = ['F0','d','alpha','fr', 'fo', 'ff', 'phi','x0','y0']
        add_pol()
    elif model_type == 'ring':
        params = ['F0','d','x0','y0']
        add_pol()
    elif model_type == 'stretched_ring':
        params = ['F0','d','x0','y0','stretch','stretch_PA']
        add_pol()
    elif model_type == 'thick_ring':
        params = ['F0','d','alpha','x0','y0']
        add_pol()
    elif model_type == 'stretched_thick_ring':
        params = ['F0','d','alpha','x0','y0','stretch','stretch_PA']
        add_pol()
    elif model_type == 'mring':
        params = ['F0','d','x0','y0']
        for j in range(len(model_params['beta_list'])):
            params.append('beta' + str(j+1) + complex_labels[0])
            params.append('beta' + str(j+1) + complex_labels[1])
        add_pol()
    elif model_type == 'stretched_mring':
        params = ['F0','d','x0','y0']
        for j in range(len(model_params['beta_list'])):
            params.append('beta' + str(j+1) + complex_labels[0])
            params.append('beta' + str(j+1) + complex_labels[1])
        add_pol()
        params.append('stretch')
        params.append('stretch_PA')
    elif model_type == 'thick_mring':
        params = ['F0','d','alpha','x0','y0']            
        for j in range(len(model_params['beta_list'])):
            params.append('beta' + str(j+1) + complex_labels[0])
            params.append('beta' + str(j+1) + complex_labels[1])
        add_pol()
    elif model_type == 'thick_mring_floor':
        params = ['F0','d','alpha','ff','x0','y0']            
        for j in range(len(model_params['beta_list'])):
            params.append('beta' + str(j+1) + complex_labels[0])
            params.append('beta' + str(j+1) + complex_labels[1])
        add_pol()
    elif model_type == 'thick_mring_Gfloor':
        params = ['F0','d','alpha','ff','FWHM','x0','y0']            
        for j in range(len(model_params['beta_list'])):
            params.append('beta' + str(j+1) + complex_labels[0])
            params.append('beta' + str(j+1) + complex_labels[1])
        add_pol()

    elif model_type == 'stretched_thick_mring':
        params = ['F0','d','alpha', 'x0','y0']
        for j in range(len(model_params['beta_list'])):
            params.append('beta' + str(j+1) + complex_labels[0])
            params.append('beta' + str(j+1) + complex_labels[1])
        add_pol()
        params.append('stretch')
        params.append('stretch_PA')
    elif model_type == 'stretched_thick_mring_floor':
        params = ['F0','d','alpha','ff', 'x0','y0']
        for j in range(len(model_params['beta_list'])):
            params.append('beta' + str(j+1) + complex_labels[0])
            params.append('beta' + str(j+1) + complex_labels[1])
        add_pol()
        params.append('stretch')
        params.append('stretch_PA')
    else:
        print('Model ' + model_init.models[j] + ' not recognized.')
        params = []

    return params

def default_prior(model_type,model_params=None,fit_pol=False,fit_cpol=False):
    """Return the default model prior and transformation for a specified model type
    """

    if COMPLEX_BASIS == 're-im':
        complex_labels = ['_re','_im']
        complex_priors  = [{'prior_type':'flat','min':-0.5,'max':0.5}, {'prior_type':'flat','min':-0.5,'max':0.5}]
        complex_priors2 = [{'prior_type':'flat','min':-1,'max':1}, {'prior_type':'flat','min':-1,'max':1}]
    elif COMPLEX_BASIS == 'abs-arg':
        complex_labels = ['_abs','_arg']
        # Note: angle range here must match np.angle(). Need to properly define wrapped distributions
        complex_priors  = [{'prior_type':'flat','min':0.0,'max':0.5}, {'prior_type':'flat','min':-np.pi, 'max':np.pi}] 
        complex_priors2 = [{'prior_type':'flat','min':0.0,'max':1.0}, {'prior_type':'flat','min':-np.pi, 'max':np.pi}] 
    else:
        raise Exception('COMPLEX_BASIS ' + COMPLEX_BASIS + ' not recognized!')

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
    elif model_type == 'blurred_disk':
        prior['d'] = {'prior_type':'positive','transform':'log'}
        prior['alpha'] = {'prior_type':'positive','transform':'log'}
    elif model_type == 'crescent': 
        prior['d'] = {'prior_type':'positive','transform':'log'}
        prior['fr'] = {'prior_type':'flat','min':0,'max':1}
        prior['fo'] = {'prior_type':'flat','min':0,'max':1}
        prior['ff'] = {'prior_type':'flat','min':0,'max':1}
        prior['phi'] = {'prior_type':'flat','min':0,'max':2.*np.pi}
    elif model_type == 'blurred_crescent':
        prior['d'] = {'prior_type':'positive','transform':'log'}
        prior['alpha'] = {'prior_type':'positive','transform':'log'}
        prior['fr'] = {'prior_type':'flat','min':0,'max':1}
        prior['fo'] = {'prior_type':'flat','min':0,'max':1}
        prior['ff'] = {'prior_type':'flat','min':0,'max':1}
        prior['phi'] = {'prior_type':'flat','min':0,'max':2.*np.pi}
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
            prior['beta' + str(j+1) + complex_labels[0]] = complex_priors[0]
            prior['beta' + str(j+1) + complex_labels[1]] = complex_priors[1]
    elif model_type == 'stretched_mring':
        prior['d'] = {'prior_type':'positive','transform':'log'}
        for j in range(len(model_params['beta_list'])):
            prior['beta' + str(j+1) + complex_labels[0]] = complex_priors[0]
            prior['beta' + str(j+1) + complex_labels[1]] = complex_priors[1]
        prior['stretch'] = {'prior_type':'positive','transform':'log'}
        prior['stretch_PA'] = {'prior_type':'none'}
    elif model_type == 'thick_mring':
        prior['d'] = {'prior_type':'positive','transform':'log'}
        prior['alpha'] = {'prior_type':'positive','transform':'log'}
        for j in range(len(model_params['beta_list'])):
            prior['beta' + str(j+1) + complex_labels[0]] = complex_priors[0]
            prior['beta' + str(j+1) + complex_labels[1]] = complex_priors[1]
    elif model_type == 'thick_mring_floor':
        prior['d'] = {'prior_type':'positive','transform':'log'}
        prior['alpha'] = {'prior_type':'positive','transform':'log'}
        prior['ff'] = {'prior_type':'flat','min':0,'max':1}
        for j in range(len(model_params['beta_list'])):
            prior['beta' + str(j+1) + complex_labels[0]] = complex_priors[0]
            prior['beta' + str(j+1) + complex_labels[1]] = complex_priors[1]
    elif model_type == 'thick_mring_Gfloor':
        prior['d'] = {'prior_type':'positive','transform':'log'}
        prior['alpha'] = {'prior_type':'positive','transform':'log'}
        prior['ff'] = {'prior_type':'flat','min':0,'max':1}
        prior['FWHM'] = {'prior_type':'positive','transform':'log'}
        for j in range(len(model_params['beta_list'])):
            prior['beta' + str(j+1) + complex_labels[0]] = complex_priors[0]
            prior['beta' + str(j+1) + complex_labels[1]] = complex_priors[1]
    elif model_type == 'stretched_thick_mring':
        prior['d'] = {'prior_type':'positive','transform':'log'}
        prior['alpha'] = {'prior_type':'positive','transform':'log'}
        for j in range(len(model_params['beta_list'])):
            prior['beta' + str(j+1) + complex_labels[0]] = complex_priors[0]
            prior['beta' + str(j+1) + complex_labels[1]] = complex_priors[1]
        prior['stretch'] = {'prior_type':'positive','transform':'log'}
        prior['stretch_PA'] = {'prior_type':'none'}
    elif model_type == 'stretched_thick_mring_floor':
        prior['d'] = {'prior_type':'positive','transform':'log'}
        prior['alpha'] = {'prior_type':'positive','transform':'log'}
        prior['ff'] = {'prior_type':'flat','min':0,'max':1}
        for j in range(len(model_params['beta_list'])):
            prior['beta' + str(j+1) + complex_labels[0]] = complex_priors[0]
            prior['beta' + str(j+1) + complex_labels[1]] = complex_priors[1]
        prior['stretch'] = {'prior_type':'positive','transform':'log'}
        prior['stretch_PA'] = {'prior_type':'none'}
    else:
        print('Model not recognized!')

    if fit_pol:
        if model_type.find('mring') == -1:
            prior['pol_frac'] = {'prior_type':'flat','min':0.0,'max':1.0}
            prior['pol_evpa'] = {'prior_type':'flat','min':0.0,'max':np.pi}
        else:            
            for j in range(-(len(model_params['beta_list_pol'])-1)//2,(len(model_params['beta_list_pol'])+1)//2):
                prior['betapol' + str(j) + complex_labels[0]] = complex_priors2[0]
                prior['betapol' + str(j) + complex_labels[1]] = complex_priors2[1]

    if fit_cpol:
        if model_type.find('mring') == -1:
            prior['cpol_frac'] = {'prior_type':'flat','min':-1.0,'max':1.0}
        else:
            for j in range(len(model_params['beta_list_cpol'])):
                if j > 0:
                    prior['betacpol' + str(j) + complex_labels[0]] = complex_priors2[0]
                    prior['betacpol' + str(j) + complex_labels[1]] = complex_priors2[1]
                else:
                    prior['betacpol0'] = {'prior_type':'flat','min':-1.0,'max':1.0}

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

def get_const_polfac(model_type, params, pol):
    # Return the scaling factor for models with constant fractional polarization

    if model_type.find('mring') != -1:
        # mring models have polarization information specified differently than a constant scaling factor
        return 1.0

    try:
        if pol == 'I':
            return 1.0
        elif pol == 'Q':
            return params['pol_frac'] * np.cos(2.0 * params['pol_evpa'])
        elif pol == 'U':
            return params['pol_frac'] * np.sin(2.0 * params['pol_evpa'])
        elif pol == 'V':
            return params['cpol_frac']
        elif pol == 'P':
            return params['pol_frac'] * np.exp(1j * 2.0 * params['pol_evpa'])
        elif pol == 'RR':
            return get_const_polfac(model_type, params, 'I') + get_const_polfac(model_type, params, 'V')
        elif pol == 'RL':
            return get_const_polfac(model_type, params, 'Q') + 1j*get_const_polfac(model_type, params, 'U')
        elif pol == 'LR':
            return get_const_polfac(model_type, params, 'Q') - 1j*get_const_polfac(model_type, params, 'U')
        elif pol == 'LL':
            return get_const_polfac(model_type, params, 'I') - get_const_polfac(model_type, params, 'V')
    except Exception:
        pass

    return 0.0

def sample_1model_xy(x, y, model_type, params, psize=1.*RADPERUAS, pol='I'):   
    if pol == 'Q':
        return np.real(sample_1model_xy(x, y, model_type, params, psize=psize, pol='P'))
    elif pol == 'U':
        return np.imag(sample_1model_xy(x, y, model_type, params, psize=psize, pol='P'))
    elif pol in ['I','V','P']:
        pass
    else:
        raise Exception('Polarization ' + pol + ' not implemented!')

    if model_type == 'point':
        val = params['F0'] * (np.abs( x - params['x0']) < psize/2.0) * (np.abs( y - params['y0']) < psize/2.0)
    elif model_type == 'circ_gauss':
        sigma = params['FWHM'] / (2. * np.sqrt(2. * np.log(2.)))
        val = (params['F0']*psize**2 * 4.0 * np.log(2.)/(np.pi * params['FWHM']**2) * 
               np.exp(-((x - params['x0'])**2 + (y - params['y0'])**2)/(2*sigma**2)))
    elif model_type == 'gauss':
        sigma_maj = params['FWHM_maj'] / (2. * np.sqrt(2. * np.log(2.)))
        sigma_min = params['FWHM_min'] / (2. * np.sqrt(2. * np.log(2.)))
        cth = np.cos(params['PA'])
        sth = np.sin(params['PA'])
        val = (params['F0']*psize**2 * 4.0 * np.log(2.)/(np.pi * params['FWHM_maj'] * params['FWHM_min']) * 
               np.exp(-((y - params['y0'])*np.cos(params['PA']) + (x - params['x0'])*np.sin(params['PA']))**2/(2*sigma_maj**2) +
                      -((x - params['x0'])*np.cos(params['PA']) - (y - params['y0'])*np.sin(params['PA']))**2/(2*sigma_min**2)))
    elif model_type == 'disk':
        val = params['F0']*psize**2/(np.pi*params['d']**2/4.) * (np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2) < params['d']/2.0) 
    elif model_type == 'blurred_disk':
        # Note: the exact form of a blurred disk requires numeric integration

        # This is the peak brightness of the blurred disk
        I_peak = 4.0/(np.pi*params['d']**2) * (1.0 - 2.0**(-params['d']**2/params['alpha']**2))

        # Constant prefactor
        prefac = 32.0 * np.log(2.0)/(np.pi * params['alpha']**2 * params['d']**2) 

        def f(r):
            return integrate.quad(lambda rp: 
                prefac * rp * np.exp( -4.0 * np.log(2.0)/params['alpha']**2 * (r**2 + rp**2 - 2.0*r * rp) ) 
                * sps.ive(0, 8.0*np.log(2.0) * r * rp/params['alpha']**2), 
                0, params['d']/2.0, limit=1000, epsabs=I_peak/1e9, epsrel=1.0e-6)[0]
        f=np.vectorize(f)
        r = np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2)

        # For images, it's much quicker to do the 1-D problem and interpolate
        if np.ndim(r) > 0:
            r_min = np.min(r)
            r_max = np.max(r)
            r_list = np.linspace(r_min, r_max, int((r_max-r_min)/(params['alpha']) * 20))
            if len(r_list) < len(np.ravel(r))/2 and len(r) > 100: 
                f = interpolate.interp1d(r_list, f(r_list), kind='cubic')
        val = params['F0'] * psize**2 * f(r)
    elif model_type == 'crescent':
        phi = params['phi']
        fr = params['fr']
        fo = params['fo']
        ff = params['ff']
        r = params['d'] / 2.
        params0 = {'F0': 1.0/(1.0-(1-ff)*fr**2)*params['F0'],   'd':params['d'],    'x0': params['x0'], 'y0': params['y0']}
        params1 = {'F0': (1-ff)*fr**2/(1.0-(1-ff)*fr**2)*params['F0'], 'd':params['d']*fr, 'x0': params['x0'] + r*(1-fr)*fo*np.sin(phi), 'y0': params['y0'] + r*(1-fr)*fo*np.cos(phi)}
        val =  sample_1model_xy(x, y, 'disk', params0, psize=psize, pol=pol) 
        val -= sample_1model_xy(x, y, 'disk', params1, psize=psize, pol=pol)
    elif model_type == 'blurred_crescent':
        phi = params['phi']
        fr = params['fr']
        fo = params['fo']
        ff = params['ff']
        r = params['d'] / 2.
        params0 = {'F0': 1.0/(1.0-(1-ff)*fr**2)*params['F0'],   'd':params['d'], 'alpha':params['alpha'], 'x0': params['x0'], 'y0': params['y0']}
        params1 = {'F0': (1-ff)*fr**2/(1.0-(1-ff)*fr**2)*params['F0'], 'd':params['d']*fr, 'alpha':params['alpha'], 'x0': params['x0'] + r*(1-fr)*fo*np.sin(phi), 'y0': params['y0'] + r*(1-fr)*fo*np.cos(phi)}
        val =  sample_1model_xy(x, y, 'blurred_disk', params0, psize=psize, pol=pol) 
        val -= sample_1model_xy(x, y, 'blurred_disk', params1, psize=psize, pol=pol)
    elif model_type == 'ring':
        val = (params['F0']*psize**2/(np.pi*params['d']*psize*LINE_THICKNESS)
                * (params['d']/2.0 - psize*LINE_THICKNESS/2 < np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2))
                * (params['d']/2.0 + psize*LINE_THICKNESS/2 > np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2)))
    elif model_type == 'thick_ring':
        r = np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2)
        z = 4.*np.log(2.) * r * params['d']/params['alpha']**2
        val = (params['F0']*psize**2 * 4.0 * np.log(2.)/(np.pi * params['alpha']**2)
                * np.exp(-4.*np.log(2.)/params['alpha']**2*(r**2 + params['d']**2/4.) + z)
                * sps.ive(0, z)) 
    elif model_type == 'mring':
        phi = np.angle((y - params['y0']) + 1j*(x - params['x0']))
        if pol == 'I':
            beta_factor = (1.0 + np.sum([2.*np.real(params['beta_list'][m-1] * np.exp(1j * m * phi)) for m in range(1,len(params['beta_list'])+1)],axis=0))
        elif pol == 'V' and len(params['beta_list_cpol']) > 0:
            beta_factor = np.real(params['beta_list_cpol'][0]) + np.sum([2.*np.real(params['beta_list_cpol'][m] * np.exp(1j * m * phi)) for m in range(1,len(params['beta_list_cpol']))],axis=0)
        elif pol == 'P' and len(params['beta_list_pol']) > 0:
            num_coeff = len(params['beta_list_pol'])
            beta_factor = np.sum([params['beta_list_pol'][m + (num_coeff-1)//2] * np.exp(1j * m * phi) for m in range(-(num_coeff-1)//2,(num_coeff+1)//2)],axis=0)
        else:
            beta_factor = 0.0

        val = (params['F0']*psize**2/(np.pi*params['d']*psize*LINE_THICKNESS)
                * beta_factor
                * (params['d']/2.0 - psize*LINE_THICKNESS/2 < np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2))
                * (params['d']/2.0 + psize*LINE_THICKNESS/2 > np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2)))
    elif model_type == 'thick_mring':
        phi = np.angle((y - params['y0']) + 1j*(x - params['x0']))
        r = np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2)
        z = 4.*np.log(2.) * r * params['d']/params['alpha']**2
        if pol == 'I':
            beta_factor = (sps.ive(0, z) + np.sum([2.*np.real(sps.ive(m, z) * params['beta_list'][m-1] * np.exp(1j * m * phi)) for m in range(1,len(params['beta_list'])+1)],axis=0))
        elif pol == 'V' and len(params['beta_list_cpol']) > 0:
            beta_factor = (sps.ive(0, z) * np.real(params['beta_list_cpol'][0]) + np.sum([2.*np.real(sps.ive(m, z) * params['beta_list_cpol'][m] * np.exp(1j * m * phi)) for m in range(1,len(params['beta_list_cpol']))],axis=0))
        elif pol == 'P' and len(params['beta_list_pol']) > 0:
            num_coeff = len(params['beta_list_pol'])
            beta_factor = np.sum([params['beta_list_pol'][m + (num_coeff-1)//2] * sps.ive(m, z) * np.exp(1j * m * phi) for m in range(-(num_coeff-1)//2,(num_coeff+1)//2)],axis=0)
        else:
            # Note: not all polarizations accounted for yet (need RR, RL, LR, LL; do these by calling for linear combinations of I, Q, U, V)!
            beta_factor = 0.0

        val = (params['F0']*psize**2 * 4.0 * np.log(2.)/(np.pi * params['alpha']**2)
                * np.exp(-4.*np.log(2.)/params['alpha']**2*(r**2 + params['d']**2/4.) + z)
                * beta_factor)
    elif model_type == 'thick_mring_floor':
        val  = (1.0 - params['ff']) * sample_1model_xy(x, y, 'thick_mring', params, psize=psize, pol=pol)
        val += params['ff'] * sample_1model_xy(x, y, 'blurred_disk', params, psize=psize, pol=pol)
    elif model_type == 'thick_mring_Gfloor':
        val  = (1.0 - params['ff']) * sample_1model_xy(x, y, 'thick_mring', params, psize=psize, pol=pol)
        val += params['ff'] * sample_1model_xy(x, y, 'circ_gauss', params, psize=psize, pol=pol)
    elif model_type[:9] == 'stretched':
        params_stretch = params.copy()
        params_stretch['F0'] /= params['stretch']
        val = sample_1model_xy(*stretch_xy(x, y, params), model_type[10:], params_stretch, psize, pol=pol)
    else:
        print('Model ' + model_type + ' not recognized!')
        val = 0.0
    return val * get_const_polfac(model_type, params, pol)

def sample_1model_uv(u, v, model_type, params, pol='I', jonesdict=None):
    if jonesdict is not None:
        # Define the various lists
        fr1 = jonesdict['fr1'] # Field rotation of site 1
        fr2 = jonesdict['fr2'] # Field rotation of site 2
        DR1 = jonesdict['DR1'] # Right leakage term of site 1
        DL1 = jonesdict['DL1'] # Left leakage term of site 1
        DR2 = np.conj(jonesdict['DR2']) # Right leakage term of site 2
        DL2 = np.conj(jonesdict['DL2']) # Left leakage term of site 2
        # Sample the model without leakage
        RR = sample_1model_uv(u, v, model_type, params, pol='RR')
        RL = sample_1model_uv(u, v, model_type, params, pol='RL')
        LR = sample_1model_uv(u, v, model_type, params, pol='LR')
        LL = sample_1model_uv(u, v, model_type, params, pol='LL')
        # Apply the Jones matrices
        RRp = RR + LR * DR1 * np.exp( 2j*fr1) + RL * DR2 * np.exp(-2j*fr2) + LL * DR1 * DR2 * np.exp( 2j*(fr1-fr2))
        RLp = RL + LL * DR1 * np.exp( 2j*fr1) + RR * DL2 * np.exp( 2j*fr2) + LR * DR1 * DL2 * np.exp( 2j*(fr1+fr2))
        LRp = LR + RR * DL1 * np.exp(-2j*fr1) + LL * DR2 * np.exp(-2j*fr2) + RL * DL1 * DR2 * np.exp(-2j*(fr1+fr2))
        LLp = LL + LR * DL2 * np.exp( 2j*fr2) + RL * DL1 * np.exp(-2j*fr1) + RR * DL1 * DL2 * np.exp(-2j*(fr1-fr2))
        # Return the specified polarization
        if   pol == 'RR': return RRp
        elif pol == 'RL': return RLp
        elif pol == 'LR': return LRp
        elif pol == 'LL': return LLp
        elif pol == 'I':  return 0.5 * (RRp + LLp)
        elif pol == 'Q':  return 0.5 * (LRp + RLp)
        elif pol == 'U':  return 0.5j* (LRp - RLp)
        elif pol == 'V':  return 0.5 * (RRp - LLp)
        elif pol == 'P':  return RLp
        else:
            raise Exception('Polarization ' + pol + ' not recognized!')

    if pol == 'Q':
        return 0.5 * (sample_1model_uv(u, v, model_type, params, pol='P') + np.conj(sample_1model_uv(-u, -v, model_type, params, pol='P')))
    elif pol == 'U':
        return -0.5j * (sample_1model_uv(u, v, model_type, params, pol='P') - np.conj(sample_1model_uv(-u, -v, model_type, params, pol='P')))
    elif pol in ['I','V','P']:
        pass
    elif pol == 'RR':
        return sample_1model_uv(u, v, model_type, params, pol='I') + sample_1model_uv(u, v, model_type, params, pol='V')
    elif pol == 'LL':
        return sample_1model_uv(u, v, model_type, params, pol='I') - sample_1model_uv(u, v, model_type, params, pol='V')
    elif pol == 'RL':
        return sample_1model_uv(u, v, model_type, params, pol='Q') + 1j*sample_1model_uv(u, v, model_type, params, pol='U')
    elif pol == 'LR':
        return sample_1model_uv(u, v, model_type, params, pol='Q') - 1j*sample_1model_uv(u, v, model_type, params, pol='U')
    else:
        raise Exception('Polarization ' + pol + ' not implemented!')

    if model_type == 'point':
        val = params['F0'] * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))
    elif model_type == 'circ_gauss':
        val = (params['F0'] 
               * np.exp(-np.pi**2/(4.*np.log(2.)) * (u**2 + v**2) * params['FWHM']**2)
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
    elif model_type == 'gauss':
        u_maj = u*np.sin(params['PA']) + v*np.cos(params['PA'])
        u_min = u*np.cos(params['PA']) - v*np.sin(params['PA'])
        val = (params['F0'] 
               * np.exp(-np.pi**2/(4.*np.log(2.)) * ((u_maj * params['FWHM_maj'])**2 + (u_min * params['FWHM_min'])**2))
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
    elif model_type == 'disk':
        z = np.pi * params['d'] * (u**2 + v**2)**0.5
        #Add a small offset to avoid issues with division by zero
        z += (z == 0.0) * 1e-10
        val = (params['F0'] * 2.0/z * sps.jv(1, z) 
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
    elif model_type == 'blurred_disk':
        z = np.pi * params['d'] * (u**2 + v**2)**0.5
        #Add a small offset to avoid issues with division by zero
        z += (z == 0.0) * 1e-10
        val = (params['F0'] * 2.0/z * sps.jv(1, z) 
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))
               * np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))) 
    elif model_type == 'crescent':
        phi = params['phi']
        fr = params['fr']
        fo = params['fo']
        ff = params['ff']
        r = params['d'] / 2.
        params0 = {'F0': 1.0/(1.0-(1-ff)*fr**2)*params['F0'],   'd':params['d'],    'x0': params['x0'], 'y0': params['y0']}
        params1 = {'F0': (1-ff)*fr**2/(1.0-(1-ff)*fr**2)*params['F0'], 'd':params['d']*fr, 'x0': params['x0'] + r*(1-fr)*fo*np.sin(phi), 'y0': params['y0'] + r*(1-fr)*fo*np.cos(phi)}
        val =  sample_1model_uv(u, v, 'disk', params0, pol=pol, jonesdict=jonesdict) 
        val -= sample_1model_uv(u, v, 'disk', params1, pol=pol, jonesdict=jonesdict)
    elif model_type == 'blurred_crescent':
        phi = params['phi']
        fr = params['fr']
        fo = params['fo']
        ff = params['ff']
        r = params['d'] / 2.
        params0 = {'F0': 1.0/(1.0-(1-ff)*fr**2)*params['F0'],   'd':params['d'], 'alpha':params['alpha'], 'x0': params['x0'], 'y0': params['y0']}
        params1 = {'F0': (1-ff)*fr**2/(1.0-(1-ff)*fr**2)*params['F0'], 'd':params['d']*fr, 'alpha':params['alpha'], 'x0': params['x0'] + r*(1-fr)*fo*np.sin(phi), 'y0': params['y0'] + r*(1-fr)*fo*np.cos(phi)}
        val =  sample_1model_uv(u, v, 'blurred_disk', params0, pol=pol, jonesdict=jonesdict) 
        val -= sample_1model_uv(u, v, 'blurred_disk', params1, pol=pol, jonesdict=jonesdict) 
    elif model_type == 'ring':
        z = np.pi * params['d'] * (u**2 + v**2)**0.5
        val = (params['F0'] * sps.jv(0, z) 
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
    elif model_type == 'thick_ring':
        z = np.pi * params['d'] * (u**2 + v**2)**0.5
        val = (params['F0'] * sps.jv(0, z) 
               * np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
    elif model_type == 'mring':
        phi = np.angle(v + 1j*u)
        # Flip the baseline sign to match eht-imaging conventions
        phi += np.pi
        z = np.pi * params['d'] * (u**2 + v**2)**0.5

        if pol == 'I':
            beta_factor = (sps.jv(0, z) 
               + np.sum([params['beta_list'][m-1]          * sps.jv( m, z) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
               + np.sum([np.conj(params['beta_list'][m-1]) * sps.jv(-m, z) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0))
        elif pol == 'V' and len(params['beta_list_cpol']) > 0:
            beta_factor = (np.real(params['beta_list_cpol'][0]) * sps.jv(0, z) 
               + np.sum([params['beta_list_cpol'][m]          * sps.jv( m, z) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0)
               + np.sum([np.conj(params['beta_list_cpol'][m]) * sps.jv(-m, z) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0))
        elif pol == 'P' and len(params['beta_list_pol']) > 0:
            num_coeff = len(params['beta_list_pol'])
            beta_factor = np.sum([params['beta_list_pol'][m + (num_coeff-1)//2] * sps.jv( m, z) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(-(num_coeff-1)//2,(num_coeff+1)//2)],axis=0)
        else:
            beta_factor = 0.0

        val = params['F0'] * beta_factor * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))
    elif model_type == 'thick_mring':
        phi = np.angle(v + 1j*u)
        # Flip the baseline sign to match eht-imaging conventions
        phi += np.pi
        z = np.pi * params['d'] * (u**2 + v**2)**0.5

        if pol == 'I':
            beta_factor = (sps.jv(0, z) 
               + np.sum([params['beta_list'][m-1]          * sps.jv( m, z) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
               + np.sum([np.conj(params['beta_list'][m-1]) * sps.jv(-m, z) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0))
        elif pol == 'V' and len(params['beta_list_cpol']) > 0:
            beta_factor = (np.real(params['beta_list_cpol'][0]) * sps.jv(0, z) 
               + np.sum([params['beta_list_cpol'][m]          * sps.jv( m, z) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0)
               + np.sum([np.conj(params['beta_list_cpol'][m]) * sps.jv(-m, z) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0))
        elif pol == 'P' and len(params['beta_list_pol']) > 0:
            num_coeff = len(params['beta_list_pol'])
            beta_factor = np.sum([params['beta_list_pol'][m + (num_coeff-1)//2] * sps.jv( m, z) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(-(num_coeff-1)//2,(num_coeff+1)//2)],axis=0)
        else:
            beta_factor = 0.0

        val = (params['F0'] * beta_factor
               * np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
    elif model_type == 'thick_mring_floor':
        val  = (1.0 - params['ff']) * sample_1model_uv(u, v, 'thick_mring', params, pol=pol, jonesdict=jonesdict)
        val += params['ff'] * sample_1model_uv(u, v, 'blurred_disk', params, pol=pol, jonesdict=jonesdict)
    elif model_type == 'thick_mring_Gfloor':
        val  = (1.0 - params['ff']) * sample_1model_uv(u, v, 'thick_mring', params, pol=pol, jonesdict=jonesdict)
        val += params['ff'] * sample_1model_uv(u, v, 'circ_gauss', params, pol=pol, jonesdict=jonesdict)
    elif model_type[:9] == 'stretched':
        params_stretch = params.copy()        
        params_stretch['x0'] = 0.0
        params_stretch['y0'] = 0.0
        val = sample_1model_uv(*stretch_uv(u,v,params), model_type[10:], params_stretch, pol=pol) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])) 
    else:
        print('Model ' + model_type + ' not recognized!')
        val = 0.0
    return val * get_const_polfac(model_type, params, pol)

def sample_1model_graduv_uv(u, v, model_type, params, pol='I', jonesdict=None):
    # Gradient of the visibility function, (dV/du, dV/dv)
    # This function makes it convenient to, e.g., compute gradients of stretched images and to compute the model centroid

    if jonesdict is not None:
        # Define the various lists
        fr1 = jonesdict['fr1'] # Field rotation of site 1
        fr2 = jonesdict['fr2'] # Field rotation of site 2
        DR1 = jonesdict['DR1'] # Right leakage term of site 1
        DL1 = jonesdict['DL1'] # Left leakage term of site 1
        DR2 = np.conj(jonesdict['DR2']) # Right leakage term of site 2
        DL2 = np.conj(jonesdict['DL2']) # Left leakage term of site 2
        # Sample the model without leakage
        RR = sample_1model_graduv_uv(u, v, model_type, params, pol='RR').reshape(2,len(u))
        RL = sample_1model_graduv_uv(u, v, model_type, params, pol='RL').reshape(2,len(u))
        LR = sample_1model_graduv_uv(u, v, model_type, params, pol='LR').reshape(2,len(u))
        LL = sample_1model_graduv_uv(u, v, model_type, params, pol='LL').reshape(2,len(u))
        # Apply the Jones matrices
        RRp = (RR + LR * DR1 * np.exp( 2j*fr1) + RL * DR2 * np.exp(-2j*fr2) + LL * DR1 * DR2 * np.exp( 2j*(fr1-fr2)))
        RLp = (RL + LL * DR1 * np.exp( 2j*fr1) + RR * DL2 * np.exp( 2j*fr2) + LR * DR1 * DL2 * np.exp( 2j*(fr1+fr2)))
        LRp = (LR + RR * DL1 * np.exp(-2j*fr1) + LL * DR2 * np.exp(-2j*fr2) + RL * DL1 * DR2 * np.exp(-2j*(fr1+fr2)))
        LLp = (LL + LR * DL2 * np.exp( 2j*fr2) + RL * DL1 * np.exp(-2j*fr1) + RR * DL1 * DL2 * np.exp(-2j*(fr1-fr2)))
        # Return the specified polarization
        if   pol == 'RR': return RRp
        elif pol == 'RL': return RLp
        elif pol == 'LR': return LRp
        elif pol == 'LL': return LLp
        elif pol == 'I':  return 0.5 * (RRp + LLp)
        elif pol == 'Q':  return 0.5 * (LRp + RLp)
        elif pol == 'U':  return 0.5j* (LRp - RLp)
        elif pol == 'V':  return 0.5 * (RRp - LLp)
        elif pol == 'P':  return RLp
        else:
            raise Exception('Polarization ' + pol + ' not recognized!')

    if pol == 'Q':
        return 0.5 * (sample_1model_graduv_uv(u, v, model_type, params, pol='P') + np.conj(sample_1model_graduv_uv(-u, -v, model_type, params, pol='P')))
    elif pol == 'U':
        return -0.5j * (sample_1model_graduv_uv(u, v, model_type, params, pol='P') - np.conj(sample_1model_graduv_uv(-u, -v, model_type, params, pol='P')))
    elif pol in ['I','V','P']:
        pass
    elif pol == 'RR':
        return sample_1model_graduv_uv(u, v, model_type, params, pol='I') + sample_1model_graduv_uv(u, v, model_type, params, pol='V')
    elif pol == 'LL':
        return sample_1model_graduv_uv(u, v, model_type, params, pol='I') - sample_1model_graduv_uv(u, v, model_type, params, pol='V')
    elif pol == 'RL':
        return sample_1model_graduv_uv(u, v, model_type, params, pol='Q') + 1j*sample_1model_graduv_uv(u, v, model_type, params, pol='U')
    elif pol == 'LR':
        return sample_1model_graduv_uv(u, v, model_type, params, pol='Q') - 1j*sample_1model_graduv_uv(u, v, model_type, params, pol='U')
    else:
        raise Exception('Polarization ' + pol + ' not implemented!')

    vis = sample_1model_uv(u, v, model_type, params, jonesdict=jonesdict)
    if model_type == 'point': 
        val = np.array([ 1j * 2.0 * np.pi * params['x0'] * vis,
                          1j * 2.0 * np.pi * params['y0'] * vis])
    elif model_type == 'circ_gauss': 
        val = np.array([ (1j * 2.0 * np.pi * params['x0'] - params['FWHM']**2 * np.pi**2 * u/(2. * np.log(2.))) * vis,
                          (1j * 2.0 * np.pi * params['y0'] - params['FWHM']**2 * np.pi**2 * v/(2. * np.log(2.))) * vis])                          
    elif model_type == 'gauss': 
        u_maj = u*np.sin(params['PA']) + v*np.cos(params['PA'])
        u_min = u*np.cos(params['PA']) - v*np.sin(params['PA'])
        val = np.array([ (1j * 2.0 * np.pi * params['x0'] - params['FWHM_maj']**2 * np.pi**2 * u_maj/(2. * np.log(2.)) * np.sin(params['PA']) - params['FWHM_min']**2 * np.pi**2 * u_min/(2. * np.log(2.)) * np.cos(params['PA'])) * vis,
                          (1j * 2.0 * np.pi * params['y0'] - params['FWHM_maj']**2 * np.pi**2 * u_maj/(2. * np.log(2.)) * np.cos(params['PA']) + params['FWHM_min']**2 * np.pi**2 * u_min/(2. * np.log(2.)) * np.sin(params['PA'])) * vis])       
    elif model_type == 'disk': 
        # Take care of the degenerate origin point by a small offset
        #v += (u==0.)*(v==0.)*1e-10
        uvdist = (u**2 + v**2 + (u==0.)*(v==0.)*1e-10)**0.5
        z = np.pi * params['d'] * uvdist
        bessel_deriv = 0.5 * (sps.jv( 0, z) - sps.jv( 2, z))
        val = np.array([  (1j * 2.0 * np.pi * params['x0'] - u/uvdist**2) * vis 
                            + params['F0'] * 2./z * np.pi * params['d'] * u/uvdist * bessel_deriv * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])),
                          (1j * 2.0 * np.pi * params['y0'] - v/uvdist**2) * vis 
                            + params['F0'] * 2./z * np.pi * params['d'] * v/uvdist * bessel_deriv * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])) ])
    elif model_type == 'blurred_disk': 
        # Take care of the degenerate origin point by a small offset
        #u += (u==0.)*(v==0.)*1e-10
        uvdist = (u**2 + v**2 + (u==0.)*(v==0.)*1e-10)**0.5
        z = np.pi * params['d'] * uvdist
        blur = np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))
        bessel_deriv = 0.5 * (sps.jv( 0, z) - sps.jv( 2, z))
        val = np.array([ (1j * 2.0 * np.pi * params['x0'] - u/uvdist**2 - params['alpha']**2 * np.pi**2 * u/(2. * np.log(2.))) * vis 
                            + params['F0'] * 2./z * np.pi * params['d'] * u/uvdist * bessel_deriv * blur * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])),
                         (1j * 2.0 * np.pi * params['y0'] - v/uvdist**2 - params['alpha']**2 * np.pi**2 * v/(2. * np.log(2.))) * vis 
                            + params['F0'] * 2./z * np.pi * params['d'] * v/uvdist * bessel_deriv * blur * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])) ])
    elif model_type == 'crescent':
        phi = params['phi']
        fr = params['fr']
        fo = params['fo']
        ff = params['ff']
        r = params['d'] / 2.
        params0 = {'F0': 1.0/(1.0-(1-ff)*fr**2)*params['F0'],   'd':params['d'],    'x0': params['x0'], 'y0': params['y0']}
        params1 = {'F0': (1-ff)*fr**2/(1.0-(1-ff)*fr**2)*params['F0'], 'd':params['d']*fr, 'x0': params['x0'] + r*(1-fr)*fo*np.sin(phi), 'y0': params['y0'] + r*(1-fr)*fo*np.cos(phi)}
        val =  sample_1model_graduv_uv(u, v, 'disk', params0, pol=pol, jonesdict=jonesdict) 
        val -= sample_1model_graduv_uv(u, v, 'disk', params1, pol=pol, jonesdict=jonesdict)
    elif model_type == 'blurred_crescent':
        phi = params['phi']
        fr = params['fr']
        fo = params['fo']
        ff = params['ff']
        r = params['d'] / 2.
        params0 = {'F0': 1.0/(1.0-(1-ff)*fr**2)*params['F0'],   'd':params['d'], 'alpha':params['alpha'], 'x0': params['x0'], 'y0': params['y0']}
        params1 = {'F0': (1-ff)*fr**2/(1.0-(1-ff)*fr**2)*params['F0'], 'd':params['d']*fr, 'alpha':params['alpha'], 'x0': params['x0'] + r*(1-fr)*fo*np.sin(phi), 'y0': params['y0'] + r*(1-fr)*fo*np.cos(phi)}
        val =  sample_1model_graduv_uv(u, v, 'blurred_disk', params0, pol=pol, jonesdict=jonesdict) 
        val -= sample_1model_graduv_uv(u, v, 'blurred_disk', params1, pol=pol, jonesdict=jonesdict) 
    elif model_type == 'ring':
        # Take care of the degenerate origin point by a small offset
        u += (u==0.)*(v==0.)*1e-10
        uvdist = (u**2 + v**2 + (u==0.)*(v==0.)*1e-10)**0.5
        z = np.pi * params['d'] * uvdist
        val = np.array([ 1j * 2.0 * np.pi * params['x0'] * vis 
                            - params['F0'] * np.pi*params['d']*u/uvdist * sps.jv(1, z) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])),
                          1j * 2.0 * np.pi * params['y0'] * vis 
                            - params['F0'] * np.pi*params['d']*v/uvdist * sps.jv(1, z) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])) ])
    elif model_type == 'thick_ring':
        uvdist = (u**2 + v**2)**0.5
        #Add a small offset to avoid issues with division by zero
        uvdist += (uvdist == 0.0) * 1e-10
        z = np.pi * params['d'] * uvdist
        val = np.array([ (1j * 2.0 * np.pi * params['x0'] - params['alpha']**2 * np.pi**2 * u/(2. * np.log(2.))) * vis 
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
        uvdist = (u**2 + v**2 + (u==0.)*(v==0.)*1e-10)**0.5
        dphidu =  v/uvdist**2 
        dphidv = -u/uvdist**2 
        z = np.pi * params['d'] * uvdist

        if pol == 'I':
            beta_factor_u = (-np.pi * params['d'] * u/uvdist * sps.jv(1, z)
                    + np.sum([params['beta_list'][m-1]          * sps.jv( m, z) * ( 1j * m * dphidu) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([np.conj(params['beta_list'][m-1]) * sps.jv(-m, z) * (-1j * m * dphidu) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([params['beta_list'][m-1]          * 0.5 * (sps.jv( m-1, z) - sps.jv( m+1, z)) * np.pi * params['d'] * u/uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([np.conj(params['beta_list'][m-1]) * 0.5 * (sps.jv(-m-1, z) - sps.jv(-m+1, z)) * np.pi * params['d'] * u/uvdist * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0))
            beta_factor_v = (-np.pi * params['d'] * v/uvdist * sps.jv(1, z)
                    + np.sum([params['beta_list'][m-1]          * sps.jv( m, z) * ( 1j * m * dphidv) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([np.conj(params['beta_list'][m-1]) * sps.jv(-m, z) * (-1j * m * dphidv) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([params['beta_list'][m-1]          * 0.5 * (sps.jv( m-1, z) - sps.jv( m+1, z)) * np.pi * params['d'] * v/uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([np.conj(params['beta_list'][m-1]) * 0.5 * (sps.jv(-m-1, z) - sps.jv(-m+1, z)) * np.pi * params['d'] * v/uvdist * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0))

        elif pol == 'V' and len(params['beta_list_cpol']) > 0:
            beta_factor_u = (-np.pi * params['d'] * u/uvdist * sps.jv(1, z) * np.real(params['beta_list_cpol'][0])
                    + np.sum([params['beta_list_cpol'][m]          * sps.jv( m, z) * ( 1j * m * dphidu) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0)
                    + np.sum([np.conj(params['beta_list_cpol'][m]) * sps.jv(-m, z) * (-1j * m * dphidu) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0)
                    + np.sum([params['beta_list_cpol'][m]          * 0.5 * (sps.jv( m-1, z) - sps.jv( m+1, z)) * np.pi * params['d'] * u/uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0)
                    + np.sum([np.conj(params['beta_list_cpol'][m]) * 0.5 * (sps.jv(-m-1, z) - sps.jv(-m+1, z)) * np.pi * params['d'] * u/uvdist * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0))
            beta_factor_v = (-np.pi * params['d'] * v/uvdist * sps.jv(1, z)
                    + np.sum([params['beta_list_cpol'][m]          * sps.jv( m, z) * ( 1j * m * dphidv) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0)
                    + np.sum([np.conj(params['beta_list_cpol'][m]) * sps.jv(-m, z) * (-1j * m * dphidv) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0)
                    + np.sum([params['beta_list_cpol'][m]          * 0.5 * (sps.jv( m-1, z) - sps.jv( m+1, z)) * np.pi * params['d'] * v/uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0)
                    + np.sum([np.conj(params['beta_list_cpol'][m]) * 0.5 * (sps.jv(-m-1, z) - sps.jv(-m+1, z)) * np.pi * params['d'] * v/uvdist * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0))
        elif pol == 'P' and len(params['beta_list_pol']) > 0:
            num_coeff = len(params['beta_list_pol'])
            beta_factor_u = (
                      np.sum([params['beta_list_pol'][m + (num_coeff-1)//2] * sps.jv( m, z) * ( 1j * m * dphidu) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(-(num_coeff-1)//2,(num_coeff+1)//2)],axis=0)
                    + np.sum([params['beta_list_pol'][m + (num_coeff-1)//2] * 0.5 * (sps.jv( m-1, z) - sps.jv( m+1, z)) * np.pi * params['d'] * u/uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(-(num_coeff-1)//2,(num_coeff+1)//2)],axis=0))
            beta_factor_v = (
                      np.sum([params['beta_list_pol'][m + (num_coeff-1)//2] * sps.jv( m, z) * ( 1j * m * dphidv) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(-(num_coeff-1)//2,(num_coeff+1)//2)],axis=0)
                    + np.sum([params['beta_list_pol'][m + (num_coeff-1)//2] * 0.5 * (sps.jv( m-1, z) - sps.jv( m+1, z)) * np.pi * params['d'] * v/uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(-(num_coeff-1)//2,(num_coeff+1)//2)],axis=0))
        else:
            beta_factor_u = beta_factor_v = 0.0

        val = np.array([ 
                 1j * 2.0 * np.pi * params['x0'] * vis 
                    + params['F0'] * beta_factor_u
                    * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])),
                 1j * 2.0 * np.pi * params['y0'] * vis 
                    + params['F0'] * beta_factor_v
                    * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])) ])
    elif model_type == 'thick_mring':
        # Take care of the degenerate origin point by a small offset
        u += (u==0.)*(v==0.)*1e-10
        phi = np.angle(v + 1j*u)
        # Flip the baseline sign to match eht-imaging conventions
        phi += np.pi
        uvdist = (u**2 + v**2 + (u==0.)*(v==0.)*1e-10)**0.5
        dphidu =  v/uvdist**2
        dphidv = -u/uvdist**2
        z = np.pi * params['d'] * uvdist

        if pol == 'I':
            beta_factor_u = (-np.pi * params['d'] * u/uvdist * sps.jv(1, z)
                    + np.sum([params['beta_list'][m-1]          * sps.jv( m, z) * ( 1j * m * dphidu) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([np.conj(params['beta_list'][m-1]) * sps.jv(-m, z) * (-1j * m * dphidu) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([params['beta_list'][m-1]          * 0.5 * (sps.jv( m-1, z) - sps.jv( m+1, z)) * np.pi * params['d'] * u/uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([np.conj(params['beta_list'][m-1]) * 0.5 * (sps.jv(-m-1, z) - sps.jv(-m+1, z)) * np.pi * params['d'] * u/uvdist * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0))
            beta_factor_v = (-np.pi * params['d'] * v/uvdist * sps.jv(1, z)
                    + np.sum([params['beta_list'][m-1]          * sps.jv( m, z) * ( 1j * m * dphidv) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([np.conj(params['beta_list'][m-1]) * sps.jv(-m, z) * (-1j * m * dphidv) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([params['beta_list'][m-1]          * 0.5 * (sps.jv( m-1, z) - sps.jv( m+1, z)) * np.pi * params['d'] * v/uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
                    + np.sum([np.conj(params['beta_list'][m-1]) * 0.5 * (sps.jv(-m-1, z) - sps.jv(-m+1, z)) * np.pi * params['d'] * v/uvdist * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0))

        elif pol == 'V' and len(params['beta_list_cpol']) > 0:
            beta_factor_u = (-np.pi * params['d'] * u/uvdist * sps.jv(1, z) * np.real(params['beta_list_cpol'][0])
                    + np.sum([params['beta_list_cpol'][m]          * sps.jv( m, z) * ( 1j * m * dphidu) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0)
                    + np.sum([np.conj(params['beta_list_cpol'][m]) * sps.jv(-m, z) * (-1j * m * dphidu) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0)
                    + np.sum([params['beta_list_cpol'][m]          * 0.5 * (sps.jv( m-1, z) - sps.jv( m+1, z)) * np.pi * params['d'] * u/uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0)
                    + np.sum([np.conj(params['beta_list_cpol'][m]) * 0.5 * (sps.jv(-m-1, z) - sps.jv(-m+1, z)) * np.pi * params['d'] * u/uvdist * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0))
            beta_factor_v = (-np.pi * params['d'] * v/uvdist * sps.jv(1, z)
                    + np.sum([params['beta_list_cpol'][m]          * sps.jv( m, z) * ( 1j * m * dphidv) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0)
                    + np.sum([np.conj(params['beta_list_cpol'][m]) * sps.jv(-m, z) * (-1j * m * dphidv) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0)
                    + np.sum([params['beta_list_cpol'][m]          * 0.5 * (sps.jv( m-1, z) - sps.jv( m+1, z)) * np.pi * params['d'] * v/uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0)
                    + np.sum([np.conj(params['beta_list_cpol'][m]) * 0.5 * (sps.jv(-m-1, z) - sps.jv(-m+1, z)) * np.pi * params['d'] * v/uvdist * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0))
        elif pol == 'P' and len(params['beta_list_pol']) > 0:
            num_coeff = len(params['beta_list_pol'])
            beta_factor_u = (0.0
                    + np.sum([params['beta_list_pol'][m + (num_coeff-1)//2] * sps.jv( m, z) * ( 1j * m * dphidu) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(-(num_coeff-1)//2,(num_coeff+1)//2)],axis=0)
                    + np.sum([params['beta_list_pol'][m + (num_coeff-1)//2] * 0.5 * (sps.jv( m-1, z) - sps.jv( m+1, z)) * np.pi * params['d'] * u/uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(-(num_coeff-1)//2,(num_coeff+1)//2)],axis=0))
            beta_factor_v = (0.0
                    + np.sum([params['beta_list_pol'][m + (num_coeff-1)//2] * sps.jv( m, z) * ( 1j * m * dphidv) * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(-(num_coeff-1)//2,(num_coeff+1)//2)],axis=0)
                    + np.sum([params['beta_list_pol'][m + (num_coeff-1)//2] * 0.5 * (sps.jv( m-1, z) - sps.jv( m+1, z)) * np.pi * params['d'] * v/uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(-(num_coeff-1)//2,(num_coeff+1)//2)],axis=0))
        else:
            beta_factor_u = beta_factor_v = 0.0

        val = np.array([ 
                 (1j * 2.0 * np.pi * params['x0'] - params['alpha']**2 * np.pi**2 * u/(2. * np.log(2.))) * vis 
                    + params['F0'] * beta_factor_u
                    * np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))
                    * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])),
                 (1j * 2.0 * np.pi * params['y0'] - params['alpha']**2 * np.pi**2 * v/(2. * np.log(2.))) * vis 
                    + params['F0'] * beta_factor_v
                    * np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))
                    * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])) ])
    elif model_type == 'thick_mring_floor':
        val  = (1.0 - params['ff']) * sample_1model_graduv_uv(u, v, 'thick_mring', params, pol=pol, jonesdict=jonesdict)
        val += params['ff'] * sample_1model_graduv_uv(u, v, 'blurred_disk', params, pol=pol, jonesdict=jonesdict)
    elif model_type == 'thick_mring_Gfloor':
        val  = (1.0 - params['ff']) * sample_1model_graduv_uv(u, v, 'thick_mring', params, pol=pol, jonesdict=jonesdict)
        val += params['ff'] * sample_1model_graduv_uv(u, v, 'circ_gauss', params, pol=pol, jonesdict=jonesdict)
    elif model_type[:9] == 'stretched':
        # Take care of the degenerate origin point by a small offset
        u += (u==0.)*(v==0.)*1e-10
        params_stretch = params.copy()        
        params_stretch['x0'] = 0.0
        params_stretch['y0'] = 0.0
        (u_stretch, v_stretch) = stretch_uv(u,v,params)
        
        # First calculate the gradient of the unshifted but stretched image
        grad0 = sample_1model_graduv_uv(u_stretch, v_stretch, model_type[10:], params_stretch, pol=pol) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))
        grad  = grad0.copy() * 0.0
        grad[0] = ( grad0[0] * (np.cos(params['stretch_PA'])**2 + np.sin(params['stretch_PA'])**2*params['stretch'])
                  + grad0[1] * ((params['stretch'] - 1.0) * np.cos(params['stretch_PA']) * np.sin(params['stretch_PA'])))
        grad[1] = ( grad0[1] * (np.cos(params['stretch_PA'])**2*params['stretch'] + np.sin(params['stretch_PA'])**2)
                  + grad0[0] * ((params['stretch'] - 1.0) * np.cos(params['stretch_PA']) * np.sin(params['stretch_PA'])))

        # Add the gradient term from the shift
        vis = sample_1model_uv(u_stretch, v_stretch, model_type[10:], params_stretch, jonesdict=jonesdict) 
        grad[0] += vis * 1j * 2.0 * np.pi * params['x0'] * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])) 
        grad[1] += vis * 1j * 2.0 * np.pi * params['y0'] * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])) 

        val = grad              
    else:
        print('Model ' + model_type + ' not recognized!')
        val = 0.0
    return val * get_const_polfac(model_type, params, pol)

def sample_1model_grad_leakage_uv_re(u, v, model_type, params, pol, site, hand, jonesdict):
    # Convenience function to calculate the gradient with respect to the real part of a specified site/hand leakage

    # Define the various lists
    fr1 = jonesdict['fr1'] # Field rotation of site 1
    fr2 = jonesdict['fr2'] # Field rotation of site 2
    DR1 = jonesdict['DR1'] # Right leakage term of site 1
    DL1 = jonesdict['DL1'] # Left leakage term of site 1
    DR2 = np.conj(jonesdict['DR2']) # Right leakage term of site 2
    DL2 = np.conj(jonesdict['DL2']) # Left leakage term of site 2
    # Sample the model without leakage
    RR = sample_1model_uv(u, v, model_type, params, pol='RR')
    RL = sample_1model_uv(u, v, model_type, params, pol='RL')
    LR = sample_1model_uv(u, v, model_type, params, pol='LR')
    LL = sample_1model_uv(u, v, model_type, params, pol='LL')

    # Figure out which terms to include in the gradient
    DR1mask = 0.0 + (hand == 'R') * (jonesdict['t1'] == site)
    DR2mask = 0.0 + (hand == 'R') * (jonesdict['t2'] == site)
    DL1mask = 0.0 + (hand == 'L') * (jonesdict['t1'] == site)
    DL2mask = 0.0 + (hand == 'L') * (jonesdict['t2'] == site)

    # These are the leakage gradient terms
    RRp = LR * DR1mask * np.exp( 2j*fr1) + RL * DR2mask * np.exp(-2j*fr2) + LL * DR1mask * DR2 * np.exp( 2j*(fr1-fr2)) + LL * DR1 * DR2mask * np.exp( 2j*(fr1-fr2))
    RLp = LL * DR1mask * np.exp( 2j*fr1) + RR * DL2mask * np.exp( 2j*fr2) + LR * DR1mask * DL2 * np.exp( 2j*(fr1+fr2)) + LR * DR1 * DL2mask * np.exp( 2j*(fr1+fr2))
    LRp = RR * DL1mask * np.exp(-2j*fr1) + LL * DR2mask * np.exp(-2j*fr2) + RL * DL1mask * DR2 * np.exp(-2j*(fr1+fr2)) + RL * DL1 * DR2mask * np.exp(-2j*(fr1+fr2))
    LLp = LR * DL2mask * np.exp( 2j*fr2) + RL * DL1mask * np.exp(-2j*fr1) + RR * DL1mask * DL2 * np.exp(-2j*(fr1-fr2)) + RR * DL1 * DL2mask * np.exp(-2j*(fr1-fr2))

    # Return the specified polarization
    if   pol == 'RR': return RRp
    elif pol == 'RL': return RLp
    elif pol == 'LR': return LRp
    elif pol == 'LL': return LLp
    elif pol == 'I':  return 0.5 * (RRp + LLp)
    elif pol == 'Q':  return 0.5 * (LRp + RLp)
    elif pol == 'U':  return 0.5j* (LRp - RLp)
    elif pol == 'V':  return 0.5 * (RRp - LLp)
    elif pol == 'P':  return RLp
    else:
        raise Exception('Polarization ' + pol + ' not recognized!')

def sample_1model_grad_leakage_uv_im(u, v, model_type, params, pol, site, hand, jonesdict):
    # Convenience function to calculate the gradient with respect to the imaginary part of a specified site/hand leakage
    # The tricky thing here is the conjugation of the second leakage site, flipping the sign of the gradient

    # Define the various lists
    fr1 = jonesdict['fr1'] # Field rotation of site 1
    fr2 = jonesdict['fr2'] # Field rotation of site 2
    DR1 = jonesdict['DR1'] # Right leakage term of site 1
    DL1 = jonesdict['DL1'] # Left leakage term of site 1
    DR2 = np.conj(jonesdict['DR2']) # Right leakage term of site 2
    DL2 = np.conj(jonesdict['DL2']) # Left leakage term of site 2
    # Sample the model without leakage
    RR = sample_1model_uv(u, v, model_type, params, pol='RR')
    RL = sample_1model_uv(u, v, model_type, params, pol='RL')
    LR = sample_1model_uv(u, v, model_type, params, pol='LR')
    LL = sample_1model_uv(u, v, model_type, params, pol='LL')

    # Figure out which terms to include in the gradient
    DR1mask = 0.0 + (hand == 'R') * (jonesdict['t1'] == site)
    DR2mask = 0.0 + (hand == 'R') * (jonesdict['t2'] == site)
    DL1mask = 0.0 + (hand == 'L') * (jonesdict['t1'] == site)
    DL2mask = 0.0 + (hand == 'L') * (jonesdict['t2'] == site)

    # These are the leakage gradient terms
    RRp = 1j*( LR * DR1mask * np.exp( 2j*fr1) - RL * DR2mask * np.exp(-2j*fr2) + LL * DR1mask * DR2 * np.exp( 2j*(fr1-fr2)) - LL * DR1 * DR2mask * np.exp( 2j*(fr1-fr2)))
    RLp = 1j*( LL * DR1mask * np.exp( 2j*fr1) - RR * DL2mask * np.exp( 2j*fr2) + LR * DR1mask * DL2 * np.exp( 2j*(fr1+fr2)) - LR * DR1 * DL2mask * np.exp( 2j*(fr1+fr2)))
    LRp = 1j*( RR * DL1mask * np.exp(-2j*fr1) - LL * DR2mask * np.exp(-2j*fr2) + RL * DL1mask * DR2 * np.exp(-2j*(fr1+fr2)) - RL * DL1 * DR2mask * np.exp(-2j*(fr1+fr2)))
    LLp = 1j*(-LR * DL2mask * np.exp( 2j*fr2) + RL * DL1mask * np.exp(-2j*fr1) + RR * DL1mask * DL2 * np.exp(-2j*(fr1-fr2)) - RR * DL1 * DL2mask * np.exp(-2j*(fr1-fr2)))

    # Return the specified polarization
    if   pol == 'RR': return RRp
    elif pol == 'RL': return RLp
    elif pol == 'LR': return LRp
    elif pol == 'LL': return LLp
    elif pol == 'I':  return 0.5 * (RRp + LLp)
    elif pol == 'Q':  return 0.5 * (LRp + RLp)
    elif pol == 'U':  return 0.5j* (LRp - RLp)
    elif pol == 'V':  return 0.5 * (RRp - LLp)
    elif pol == 'P':  return RLp
    else:
        raise Exception('Polarization ' + pol + ' not recognized!')

def sample_1model_grad_uv(u, v, model_type, params, pol='I', fit_pol=False, fit_cpol=False, fit_leakage=False, jonesdict=None):
    # Gradient of the model for each model parameter 

    if jonesdict is not None:
        # Define the various lists
        fr1 = jonesdict['fr1'] # Field rotation of site 1
        fr2 = jonesdict['fr2'] # Field rotation of site 2
        DR1 = jonesdict['DR1'] # Right leakage term of site 1
        DL1 = jonesdict['DL1'] # Left leakage term of site 1
        DR2 = np.conj(jonesdict['DR2']) # Right leakage term of site 2
        DL2 = np.conj(jonesdict['DL2']) # Left leakage term of site 2
        # Sample the gradients without leakage
        RR = sample_1model_grad_uv(u, v, model_type, params, pol='RR', fit_pol=fit_pol, fit_cpol=fit_cpol)
        RL = sample_1model_grad_uv(u, v, model_type, params, pol='RL', fit_pol=fit_pol, fit_cpol=fit_cpol)
        LR = sample_1model_grad_uv(u, v, model_type, params, pol='LR', fit_pol=fit_pol, fit_cpol=fit_cpol)
        LL = sample_1model_grad_uv(u, v, model_type, params, pol='LL', fit_pol=fit_pol, fit_cpol=fit_cpol)
        # Apply the Jones matrices
        RRp = (RR + LR * DR1 * np.exp( 2j*fr1) + RL * DR2 * np.exp(-2j*fr2) + LL * DR1 * DR2 * np.exp( 2j*(fr1-fr2)))
        RLp = (RL + LL * DR1 * np.exp( 2j*fr1) + RR * DL2 * np.exp( 2j*fr2) + LR * DR1 * DL2 * np.exp( 2j*(fr1+fr2)))
        LRp = (LR + RR * DL1 * np.exp(-2j*fr1) + LL * DR2 * np.exp(-2j*fr2) + RL * DL1 * DR2 * np.exp(-2j*(fr1+fr2)))
        LLp = (LL + LR * DL2 * np.exp( 2j*fr2) + RL * DL1 * np.exp(-2j*fr1) + RR * DL1 * DL2 * np.exp(-2j*(fr1-fr2)))
        # Return the specified polarization
        if   pol == 'RR': grad = RRp
        elif pol == 'RL': grad = RLp
        elif pol == 'LR': grad = LRp
        elif pol == 'LL': grad = LLp
        elif pol == 'I':  grad = 0.5 * (RRp + LLp)
        elif pol == 'Q':  grad = 0.5 * (LRp + RLp)
        elif pol == 'U':  grad = 0.5j* (LRp - RLp)
        elif pol == 'V':  grad = 0.5 * (RRp - LLp)
        elif pol == 'P':  grad = RLp
        else:
            raise Exception('Polarization ' + pol + ' not recognized!')
        # If necessary, add the gradient components from the leakage terms
        # Each leakage term has two corresponding gradient terms: d/dRe and d/dIm. 
        if fit_leakage:
            # 'leakage_fit' is a list of tuples [site, 'R' or 'L'] denoting the fitted leakage terms
            for (site, hand) in jonesdict['leakage_fit']:
                grad = np.vstack([grad, sample_1model_grad_leakage_uv_re(u, v, model_type, params, pol, site, hand, jonesdict), sample_1model_grad_leakage_uv_im(u, v, model_type, params, pol, site, hand, jonesdict)])

        return grad

    if pol == 'Q':
        return   0.5 * (sample_1model_grad_uv(u, v, model_type, params, pol='P', fit_pol=fit_pol, fit_cpol=fit_cpol) + np.conj(sample_1model_grad_uv(-u, -v, model_type, params, pol='P', fit_pol=fit_pol, fit_cpol=fit_cpol)))
    elif pol == 'U':
        return -0.5j * (sample_1model_grad_uv(u, v, model_type, params, pol='P', fit_pol=fit_pol, fit_cpol=fit_cpol) - np.conj(sample_1model_grad_uv(-u, -v, model_type, params, pol='P', fit_pol=fit_pol, fit_cpol=fit_cpol)))
    elif pol in ['I','V','P']:
        pass
    elif pol == 'RR':
        return sample_1model_grad_uv(u, v, model_type, params, pol='I', fit_pol=fit_pol, fit_cpol=fit_cpol) + sample_1model_grad_uv(u, v, model_type, params, pol='V', fit_pol=fit_pol, fit_cpol=fit_cpol)
    elif pol == 'LL':
        return sample_1model_grad_uv(u, v, model_type, params, pol='I', fit_pol=fit_pol, fit_cpol=fit_cpol) - sample_1model_grad_uv(u, v, model_type, params, pol='V', fit_pol=fit_pol, fit_cpol=fit_cpol)
    elif pol == 'RL':
        return sample_1model_grad_uv(u, v, model_type, params, pol='Q', fit_pol=fit_pol, fit_cpol=fit_cpol) + 1j*sample_1model_grad_uv(u, v, model_type, params, pol='U', fit_pol=fit_pol, fit_cpol=fit_cpol)
    elif pol == 'LR':
        return sample_1model_grad_uv(u, v, model_type, params, pol='Q', fit_pol=fit_pol, fit_cpol=fit_cpol) - 1j*sample_1model_grad_uv(u, v, model_type, params, pol='U', fit_pol=fit_pol, fit_cpol=fit_cpol)
    else:
        raise Exception('Polarization ' + pol + ' not implemented!')

    if model_type == 'point': # F0, x0, y0
        val = np.array([ np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])),
                 1j * 2.0 * np.pi * u * params['F0'] * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])),
                 1j * 2.0 * np.pi * v * params['F0'] * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))])
    elif model_type == 'circ_gauss': # F0, FWHM, x0, y0
        gauss = (params['F0'] * np.exp(-np.pi**2/(4.*np.log(2.)) * (u**2 + v**2) * params['FWHM']**2)
                *np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])))
        val = np.array([ 1.0/params['F0'] * gauss,
                         -np.pi**2/(2.*np.log(2.)) * (u**2 + v**2) * params['FWHM'] * gauss,
                          1j * 2.0 * np.pi * u * gauss,
                          1j * 2.0 * np.pi * v * gauss])                          
    elif model_type == 'gauss': # F0, FWHM_maj, FWHM_min, PA, x0, y0
        u_maj = u*np.sin(params['PA']) + v*np.cos(params['PA'])
        u_min = u*np.cos(params['PA']) - v*np.sin(params['PA'])
        vis = (params['F0'] 
               * np.exp(-np.pi**2/(4.*np.log(2.)) * ((u_maj * params['FWHM_maj'])**2 + (u_min * params['FWHM_min'])**2))
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
        val = np.array([ 1.0/params['F0'] * vis,
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
        val = np.array([ 1.0/params['F0'] * vis,
                         -(params['F0'] * 2.0/z * sps.jv(2, z) * np.pi * (u**2 + v**2)**0.5 * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) ,
                          1j * 2.0 * np.pi * u * vis,
                          1j * 2.0 * np.pi * v * vis]) 
    elif model_type == 'blurred_disk': # F0, d, alpha, x0, y0
        z = np.pi * params['d'] * (u**2 + v**2)**0.5
        #Add a small offset to avoid issues with division by zero
        z += (z == 0.0) * 1e-10
        blur = np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))
        vis = (params['F0'] * 2.0/z * sps.jv(1, z) 
               * blur
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
        val = np.array([ 1.0/params['F0'] * vis,
                         -params['F0'] * 2.0/z * sps.jv(2, z) * np.pi * (u**2 + v**2)**0.5 * blur * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])),
                         -np.pi**2 * (u**2 + v**2) * params['alpha']/(2.*np.log(2.)) * vis,
                          1j * 2.0 * np.pi * u * vis,
                          1j * 2.0 * np.pi * v * vis])  
    elif model_type == 'crescent': #['F0','d', 'fr', 'fo', 'phi','x0','y0']
        phi = params['phi'] 
        fr = params['fr']
        fo = params['fo']
        ff = params['ff']
        r = params['d'] / 2.
        params0 = {'F0': 1.0/(1.0-(1-ff)*fr**2)*params['F0'],   'd':params['d'],    'x0': params['x0'], 'y0': params['y0']}
        params1 = {'F0': (1-ff)*fr**2/(1.0-(1-ff)*fr**2)*params['F0'], 'd':params['d']*fr, 'x0': params['x0'] + r*(1-fr)*fo*np.sin(phi), 'y0': params['y0'] + r*(1-fr)*fo*np.cos(phi)}

        grad0 = sample_1model_grad_uv(u, v, 'disk', params0, pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol, fit_leakage=fit_leakage, jonesdict=jonesdict)
        grad1 = sample_1model_grad_uv(u, v, 'disk', params1, pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol, fit_leakage=fit_leakage, jonesdict=jonesdict)

        # Add the derivatives one by one
        grad = []

        # F0
        grad.append( 1.0/(1.0-(1-ff)*fr**2)*grad0[0] - (1-ff)*fr**2/(1.0-(1-ff)*fr**2)*grad1[0] )

        # d
        grad.append( grad0[1] - fr*grad1[1] - 0.5 * (1.0 - fr) * fo * (np.sin(phi) * grad1[2] + np.cos(phi) * grad1[3])  )

        # fr
        grad.append( 2.0*params['F0']*(1-ff)*fr/(1.0 - (1-ff)*fr**2)**2 * (grad0[0] - grad1[0]) - params['d']*grad1[1] + r * fo * (np.sin(phi) * grad1[2] + np.cos(phi) * grad1[3]) ) 

        # fo
        grad.append( -r * (1-fr) * (np.sin(phi) * grad1[2] + np.cos(phi) * grad1[3]) ) 

        # ff
        grad.append( -params['F0']*fr**2/(1.0 - (1-ff)*fr**2)**2 * (grad0[0] - grad1[0]) ) 

        # phi
        grad.append( -r*(1-fr)*fo* (np.cos(phi) * grad1[2] - np.sin(phi) * grad1[3]) ) 

        # x0, y0
        grad.append( grad0[2] - grad1[2] ) 
        grad.append( grad0[3] - grad1[3] ) 

        val = np.array(grad)
    elif model_type == 'blurred_crescent': #['F0','d','alpha','fr', 'fo', 'phi','x0','y0']
        phi = params['phi']
        fr = params['fr']
        fo = params['fo']
        ff = params['ff']
        r = params['d'] / 2.
        params0 = {'F0': 1.0/(1.0-(1-ff)*fr**2)*params['F0'],   'd':params['d'], 'alpha':params['alpha'], 'x0': params['x0'], 'y0': params['y0']}
        params1 = {'F0': (1-ff)*fr**2/(1.0-(1-ff)*fr**2)*params['F0'], 'd':params['d']*fr, 'alpha':params['alpha'], 'x0': params['x0'] + r*(1-fr)*fo*np.sin(phi), 'y0': params['y0'] + r*(1-fr)*fo*np.cos(phi)}

        grad0 = sample_1model_grad_uv(u, v, 'blurred_disk', params0, pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol, fit_leakage=fit_leakage, jonesdict=jonesdict)
        grad1 = sample_1model_grad_uv(u, v, 'blurred_disk', params1, pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol, fit_leakage=fit_leakage, jonesdict=jonesdict)

        # Add the derivatives one by one
        grad = []

        # F0
        grad.append( 1.0/(1.0-(1-ff)*fr**2)*grad0[0] - (1-ff)*fr**2/(1.0-(1-ff)*fr**2)*grad1[0] )

        # d
        grad.append( grad0[1] - fr*grad1[1] - 0.5 * (1.0 - fr) * fo * (np.sin(phi) * grad1[3] + np.cos(phi) * grad1[4])  )

        # alpha
        grad.append( grad0[2] - grad1[2] ) 

        # fr
        grad.append( 2.0*params['F0']*(1-ff)*fr/(1.0 - (1-ff)*fr**2)**2 * (grad0[0] - grad1[0]) - params['d']*grad1[1] + r * fo * (np.sin(phi) * grad1[3] + np.cos(phi) * grad1[4]) ) 

        # fo
        grad.append( -r * (1-fr) * (np.sin(phi) * grad1[3] + np.cos(phi) * grad1[4]) ) 

        # ff
        grad.append( -params['F0']*fr**2/(1.0 - (1-ff)*fr**2)**2 * (grad0[0] - grad1[0]) ) 

        # phi
        grad.append( -r*(1-fr)*fo* (np.cos(phi) * grad1[3] - np.sin(phi) * grad1[4]) ) 

        # x0, y0
        grad.append( grad0[3] - grad1[3] ) 
        grad.append( grad0[4] - grad1[4] ) 

        val = np.array(grad)

    elif model_type == 'ring': # F0, d, x0, y0
        z = np.pi * params['d'] * (u**2 + v**2)**0.5
        val = np.array([ sps.jv(0, z) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])), 
                -np.pi * (u**2 + v**2)**0.5 * params['F0'] * sps.jv(1, z) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])), 
                 2.0 * np.pi * 1j * u * params['F0'] * sps.jv(0, z) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])), 
                 2.0 * np.pi * 1j * v * params['F0'] * sps.jv(0, z) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))])
    elif model_type == 'thick_ring': # F0, d, alpha, x0, y0
        z = np.pi * params['d'] * (u**2 + v**2)**0.5
        vis = (params['F0'] * sps.jv(0, z) 
               * np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
        val = np.array([ 1.0/params['F0'] * vis,
                         -(params['F0'] * np.pi * (u**2 + v**2)**0.5 * sps.jv(1, z) 
                            * np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))
                            * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))),
                         -np.pi**2 * (u**2 + v**2) * params['alpha']/(2.*np.log(2.)) * vis,
                          1j * 2.0 * np.pi * u * vis,
                          1j * 2.0 * np.pi * v * vis])    
    elif model_type in ['mring','thick_mring']: # F0, d, [alpha], x0, y0, beta1_re/abs, beta1_im/arg, beta2_re/abs, beta2_im/arg, ...
        phi = np.angle(v + 1j*u)
        # Flip the baseline sign to match eht-imaging conventions
        phi += np.pi
        uvdist = (u**2 + v**2)**0.5
        z = np.pi * params['d'] * uvdist
        if model_type == 'thick_mring':
            alpha_factor = np.exp(-(np.pi * params['alpha'] * (u**2 + v**2)**0.5)**2/(4. * np.log(2.)))
        else:
            alpha_factor = 1

        # Only one of the beta_lists will affect the measurement and have non-zero gradients. Figure out which:
        # These are for the derivatives wrt diameter
        if pol == 'I':
            beta_factor = (-np.pi * uvdist * sps.jv(1, z) 
               + np.sum([params['beta_list'][m-1]          * 0.5 * (sps.jv( m-1, z)  - sps.jv(  m+1, z)) * np.pi * uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
               + np.sum([np.conj(params['beta_list'][m-1]) * 0.5 * (sps.jv( -m-1, z) - sps.jv( -m+1, z)) * np.pi * uvdist * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0))
        elif pol == 'P' and len(params['beta_list_pol']) > 0:
            num_coeff = len(params['beta_list_pol'])
            beta_factor = np.sum([params['beta_list_pol'][m + (num_coeff-1)//2] * 0.5 * (sps.jv( m-1, z)  - sps.jv(  m+1, z)) * np.pi * uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(-(num_coeff-1)//2,(num_coeff+1)//2)],axis=0)
        elif pol == 'V' and len(params['beta_list_cpol']) > 0:
            beta_factor = (-np.pi * uvdist * sps.jv(1, z) * np.real(params['beta_list_cpol'][0]) 
               + np.sum([params['beta_list_cpol'][m]          * 0.5 * (sps.jv( m-1, z)  - sps.jv(  m+1, z)) * np.pi * uvdist * np.exp( 1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0)
               + np.sum([np.conj(params['beta_list_cpol'][m]) * 0.5 * (sps.jv( -m-1, z) - sps.jv( -m+1, z)) * np.pi * uvdist * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list_cpol']))],axis=0))
        else:
            beta_factor = 0.0
            
        vis  = sample_1model_uv(u, v, model_type, params, pol=pol, jonesdict=jonesdict)
        grad = [ 1.0/params['F0'] * vis, 
                 (params['F0'] * alpha_factor * beta_factor * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])))]
        if model_type == 'thick_mring':
            grad.append(-np.pi**2/(2.*np.log(2)) * uvdist**2 * params['alpha'] * vis)
        grad.append(1j * 2.0 * np.pi * u * vis)
        grad.append(1j * 2.0 * np.pi * v * vis)

        if pol=='I':
            # Add derivatives of the beta terms 
            for m in range(1,len(params['beta_list'])+1):
                beta_grad_re =      params['F0'] * alpha_factor * (
                   sps.jv( m, z) * np.exp( 1j * m * (phi - np.pi/2.)) + sps.jv(-m, z) * np.exp(-1j * m * (phi - np.pi/2.)) 
                   * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
                beta_grad_im = 1j * params['F0'] * alpha_factor * (
                   sps.jv( m, z) * np.exp( 1j * m * (phi - np.pi/2.)) - sps.jv(-m, z) * np.exp(-1j * m * (phi - np.pi/2.)) 
                   * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
                if COMPLEX_BASIS == 're-im':
                    grad.append(beta_grad_re)
                    grad.append(beta_grad_im)
                elif COMPLEX_BASIS == 'abs-arg':
                    beta_abs = np.abs(params['beta_list'][m-1])
                    beta_arg = np.angle(params['beta_list'][m-1])
                    grad.append(beta_grad_re * np.cos(beta_arg) + beta_grad_im * np.sin(beta_arg))
                    grad.append(-beta_abs * np.sin(beta_arg) * beta_grad_re + beta_abs * np.cos(beta_arg) * beta_grad_im)
                else:
                    raise Exception('COMPLEX_BASIS ' + COMPLEX_BASIS + ' not recognized!')
        else:
            [grad.append(np.zeros_like(grad[0])) for _ in range(2*len(params['beta_list']))]

        if pol=='P' and fit_pol:
            # Add derivatives of the beta terms 
            num_coeff = len(params['beta_list_pol'])
            for m in range(-(num_coeff-1)//2,(num_coeff+1)//2):
                beta_grad_re = params['F0'] * alpha_factor * sps.jv( m, z) * np.exp( 1j * m * (phi - np.pi/2.)) 
                beta_grad_im = 1j * beta_grad_re
                if COMPLEX_BASIS == 're-im':
                    grad.append(beta_grad_re)
                    grad.append(beta_grad_im)
                elif COMPLEX_BASIS == 'abs-arg':
                    beta_abs = np.abs(params['beta_list_pol'][m+(num_coeff-1)//2])
                    beta_arg = np.angle(params['beta_list_pol'][m+(num_coeff-1)//2])
                    grad.append(beta_grad_re * np.cos(beta_arg) + beta_grad_im * np.sin(beta_arg))
                    grad.append(-beta_abs * np.sin(beta_arg) * beta_grad_re + beta_abs * np.cos(beta_arg) * beta_grad_im)
                else:
                    raise Exception('COMPLEX_BASIS ' + COMPLEX_BASIS + ' not recognized!')
        elif pol!='P' and fit_pol:
            [grad.append(np.zeros_like(grad[0])) for _ in range(2*len(params['beta_list_pol']))]

        val = np.array(grad)
    elif model_type == 'thick_mring_floor': # F0, d, [alpha], ff, x0, y0, beta1_re/abs, beta1_im/arg, beta2_re/abs, beta2_im/arg, ...
        # We need to stich together the two gradients for the mring and the disk; we also need to add the gradient for the floor fraction ff
        grad_mring = (1.0 - params['ff']) * sample_1model_grad_uv(u, v, 'thick_mring', params, pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol)
        grad_disk  = params['ff'] * sample_1model_grad_uv(u, v, 'blurred_disk', params, pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol)

        # mring: F0, d, alpha, x0, y0, beta1_re/abs, beta1_im/arg, beta2_re/abs, beta2_im/arg, ...
        # disk:  F0, d, alpha, x0, y0

        # Here are derivatives for F0, d, and alpha
        grad = []
        for j in range(3):
            grad.append(grad_mring[j] + grad_disk[j])

        # Here is the derivative for ff
        grad.append( params['F0'] * (grad_disk[0]/params['ff'] - grad_mring[0]/(1.0 - params['ff'])) )

        # Now the derivatives for x0 and y0
        grad.append(grad_mring[3] + grad_disk[3])
        grad.append(grad_mring[4] + grad_disk[4])

        # Add remaining gradients for the mring
        for j in range(5,len(grad_mring)):
            grad.append(grad_mring[j])

        val = np.array(grad)
    elif model_type == 'thick_mring_Gfloor': # F0, d, [alpha], ff, FWHM, x0, y0, beta1_re/abs, beta1_im/arg, beta2_re/abs, beta2_im/arg, ...
        # We need to stich together the two gradients for the mring and the gaussian; we also need to add the gradient for the floor fraction ff
        grad_mring = (1.0 - params['ff']) * sample_1model_grad_uv(u, v, 'thick_mring', params, pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol)
        grad_gauss  = params['ff'] * sample_1model_grad_uv(u, v, 'circ_gauss', params, pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol)

        # mring: F0, d, alpha, x0, y0, beta1_re/abs, beta1_im/arg, beta2_re/abs, beta2_im/arg, ...
        # gauss: F0, [d, alpha] FWHM, x0, y0

        grad = []
        grad.append(grad_mring[0] + grad_gauss[0]) # Here are derivatives for F0 

        # Here are derivatives for d, and alpha 
        grad.append(grad_mring[1])
        grad.append(grad_mring[2])

        # Here is the derivative for ff
        grad.append( params['F0'] * (grad_gauss[0]/params['ff'] - grad_mring[0]/(1.0 - params['ff'])) )

        # Now the derivatives for FWHM
        grad.append(grad_gauss[1])

        # Now the derivatives for x0 and y0
        grad.append(grad_mring[3] + grad_gauss[2])
        grad.append(grad_mring[4] + grad_gauss[3])

        # Add remaining gradients for the mring
        for j in range(5,len(grad_mring)):
            grad.append(grad_mring[j])

        val = np.array(grad)
    elif model_type[:9] == 'stretched':
        # Start with the model visibility
        vis  = sample_1model_uv(u, v, model_type, params, pol=pol, jonesdict=jonesdict)

        # Next, calculate the gradient wrt model parameters other than stretch and stretch_PA
        # These are the same as the gradient of the unstretched model on stretched baselines
        params_stretch = params.copy()
        params_stretch['x0'] = 0.0
        params_stretch['y0'] = 0.0
        (u_stretch, v_stretch) = stretch_uv(u,v,params)
        grad = (sample_1model_grad_uv(u_stretch, v_stretch, model_type[10:], params_stretch, pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol)
                * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])))

        # Add the gradient terms for the centroid
        grad[model_params(model_type, params).index('x0')] = 1j * 2.0 * np.pi * u * vis
        grad[model_params(model_type, params).index('y0')] = 1j * 2.0 * np.pi * v * vis

        # Now calculate the gradient with respect to stretch and stretch PA
        grad_uv = sample_1model_graduv_uv(u_stretch, v_stretch, model_type[10:], params_stretch, pol=pol)
        grad_stretch = grad_uv.copy() * 0.0 
        grad_stretch[0] = ( grad_uv[0] * (u * np.sin(params['stretch_PA'])**2 + v * np.sin(params['stretch_PA']) * np.cos(params['stretch_PA']))
                          + grad_uv[1] * (v * np.cos(params['stretch_PA'])**2 + u * np.sin(params['stretch_PA']) * np.cos(params['stretch_PA'])))
        grad_stretch[0] *= np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))

        grad_stretch[1] = ( grad_uv[0] * (params['stretch'] - 1.0) * ( u * np.sin(2.0 * params['stretch_PA']) + v * np.cos(2.0 * params['stretch_PA']))
                          + grad_uv[1] * (params['stretch'] - 1.0) * (-v * np.sin(2.0 * params['stretch_PA']) + u * np.cos(2.0 * params['stretch_PA'])))
        grad_stretch[1] *= np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))

        val = np.concatenate([grad, grad_stretch])
    else:
        print('Model ' + model_type + ' not recognized!')
        val = 0.0

    grad = val * get_const_polfac(model_type, params, pol)

    if (fit_pol or fit_cpol) and model_type.find('mring') == -1:
        # Add gradient contributions for models that have constant polarization
        if fit_pol:
            # Add gradient wrt pol_frac if the polarization is P, otherwise ignore
            grad_params = copy.deepcopy(params)
            grad_params['pol_frac'] = 1.0
            grad = np.vstack([grad, (pol == 'P') * sample_1model_uv(u, v, model_type, grad_params, pol=pol, jonesdict=jonesdict)])

            # Add gradient wrt pol_evpa if the polarization is P, otherwise ignore
            grad_params = copy.deepcopy(params)
            grad_params['pol_frac'] *= 2j
            grad = np.vstack([grad, (pol == 'P') * sample_1model_uv(u, v, model_type, grad_params, pol=pol, jonesdict=jonesdict)])
        if fit_cpol:
            # Add gradient wrt cpol_frac
            grad_params = copy.deepcopy(params)
            grad_params['cpol_frac'] = 1.0
            grad = np.vstack([grad, (pol == 'V') * sample_1model_uv(u, v, model_type, grad_params, pol=pol, jonesdict=jonesdict)])

    return grad

def sample_model_xy(models, params, x, y, psize=1.*RADPERUAS, pol='I'):
    return np.sum(sample_1model_xy(x, y, models[j], params[j], psize=psize,pol=pol) for j in range(len(models)))

def sample_model_uv(models, params, u, v, pol='I', jonesdict=None):
    return np.sum(sample_1model_uv(u, v, models[j], params[j], pol=pol, jonesdict=jonesdict) for j in range(len(models)))

def sample_model_graduv_uv(models, params, u, v, pol='I', jonesdict=None):
    # Gradient of a sum of models wrt (u,v)
    return np.sum([sample_1model_graduv_uv(u, v, models[j], params[j], pol=pol, jonesdict=jonesdict) for j in range(len(models))],axis=0)

def sample_model_grad_uv(models, params, u, v, pol='I', fit_pol=False, fit_cpol=False, fit_leakage=False, jonesdict=None):
    # Gradient of a sum of models for each parameter
    if fit_leakage == False:
        return np.concatenate([sample_1model_grad_uv(u, v, models[j], params[j], pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol, fit_leakage=fit_leakage, jonesdict=jonesdict) for j in range(len(models))])
    else:
        # Need to sum the leakage contributions
        allgrad = [sample_1model_grad_uv(u, v, models[j], params[j], pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol, fit_leakage=fit_leakage, jonesdict=jonesdict) for j in range(len(models))]
        n_leakage = len(jonesdict['leakage_fit'])*2
        grad = np.concatenate([allgrad[j][:-n_leakage] for j in range(len(models))])
        grad_leakage = np.sum([allgrad[j][-n_leakage:] for j in range(len(models))],axis=0)
        return np.concatenate([grad, grad_leakage])

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
    elif 'thick' in model_type or 'blurred' in model_type:
        params_blur['alpha'] = (params_blur['alpha']**2 + fwhm**2)**0.5
    elif model_type == 'disk':
        model_type_blur = 'blurred_' + model_type
        params_blur['alpha'] = fwhm
    elif model_type == 'crescent':
        model_type_blur = 'blurred_' + model_type
        params_blur['alpha'] = fwhm
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
        return np.real(self.sample_uv(0,0))

    def blur_circ(self, fwhm):
        """Return a new model, equal to the current one convolved with a circular Gaussian kernel

           Args:
                fwhm (float) : Full width at half maximum of the kernel (radians)

           Returns:
                (Model) : Blurred model
        """

        out = self.copy()

        for j in range(len(out.models)):
            blur_model = blur_circ_1model(out.models[j], out.params[j], fwhm)
            out.models[j] = blur_model['model_type']
            out.params[j] = blur_model['params']

        return out

    def add_point(self, F0 = 1.0, x0 = 0.0, y0 = 0.0, pol_frac = 0.0, pol_evpa = 0.0, cpol_frac = 0.0):
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
        out.params.append({'F0':F0,'x0':x0,'y0':y0,'pol_frac':pol_frac,'pol_evpa':pol_evpa,'cpol_frac':cpol_frac})
        return out

    def add_circ_gauss(self, F0 = 1.0, FWHM = 50.*RADPERUAS, x0 = 0.0, y0 = 0.0, pol_frac = 0.0, pol_evpa = 0.0, cpol_frac = 0.0):
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
        out.params.append({'F0':F0,'FWHM':FWHM,'x0':x0,'y0':y0,'pol_frac':pol_frac,'pol_evpa':pol_evpa,'cpol_frac':cpol_frac})
        return out

    def add_gauss(self, F0 = 1.0, FWHM_maj = 50.*RADPERUAS, FWHM_min = 50.*RADPERUAS, PA = 0.0, x0 = 0.0, y0 = 0.0, pol_frac = 0.0, pol_evpa = 0.0, cpol_frac = 0.0):
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
        out.params.append({'F0':F0,'FWHM_maj':FWHM_maj,'FWHM_min':FWHM_min,'PA':PA,'x0':x0,'y0':y0,'pol_frac':pol_frac,'pol_evpa':pol_evpa,'cpol_frac':cpol_frac})
        return out

    def add_disk(self, F0 = 1.0, d = 50.*RADPERUAS, x0 = 0.0, y0 = 0.0, pol_frac = 0.0, pol_evpa = 0.0, cpol_frac = 0.0):
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
        out.params.append({'F0':F0,'d':d,'x0':x0,'y0':y0,'pol_frac':pol_frac,'pol_evpa':pol_evpa,'cpol_frac':cpol_frac})
        return out

    def add_blurred_disk(self, F0 = 1.0, d = 50.*RADPERUAS, alpha = 10.*RADPERUAS, x0 = 0.0, y0 = 0.0, pol_frac = 0.0, pol_evpa = 0.0, cpol_frac = 0.0):
        """Add a circular disk model that is blurred with a circular Gaussian kernel.

           Args:
               F0 (float): The total flux of the disk (Jy)
               d (float): The diameter (radians)
               alpha (float): The blurring (FWHM of Gaussian convolution) (radians)
               x0 (float): The x-coordinate (radians)
               y0 (float): The y-coordinate (radians)

           Returns:
                (Model): Updated Model
        """
        out = self.copy()
        out.models.append('blurred_disk')
        out.params.append({'F0':F0,'d':d,'alpha':alpha,'x0':x0,'y0':y0,'pol_frac':pol_frac,'pol_evpa':pol_evpa,'cpol_frac':cpol_frac})
        return out

    def add_crescent(self, F0 = 1.0, d = 50.*RADPERUAS, fr = 0.0, fo = 0.0, ff = 0.0, phi = 0.0, x0 = 0.0, y0 = 0.0, pol_frac = 0.0, pol_evpa = 0.0, cpol_frac = 0.0):
        """Add a crescent model.

           Args:
               F0 (float): The total flux of the disk (Jy)
               d (float): The diameter (radians)
               fr (float): Fractional radius of the inner subtracted disk with respect to the radius of the outer disk
               fo (float): Fractional offset of the inner disk from the center of the outer disk
               ff (float): Fractional brightness of the inner disk
               phi (float): angle of offset of the inner disk
               x0 (float): The x-coordinate (radians)
               y0 (float): The y-coordinate (radians)

           Returns:
                (Model): Updated Model
        """
        out = self.copy()
        out.models.append('crescent')
        out.params.append({'F0':F0,'d':d,'fr':fr, 'fo':fo, 'ff':ff, 'phi':phi, 'x0':x0, 'y0':y0, 'pol_frac':pol_frac, 'pol_evpa':pol_evpa, 'cpol_frac':cpol_frac})
        return out

    def add_blurred_crescent(self, F0 = 1.0, d = 50.*RADPERUAS, alpha = 10.*RADPERUAS, fr = 0.0, fo = 0.0, ff = 0.0, phi = 0.0, x0 = 0.0, y0 = 0.0, pol_frac = 0.0, pol_evpa = 0.0, cpol_frac = 0.0):

        """Add a circular disk model that is blurred with a circular Gaussian kernel.

           Args:
               F0 (float): The total flux of the disk (Jy)
               d (float): The diameter (radians)
               alpha (float) :The blurring (FWHM of Gaussian convolution) (radians)
               fr (float): Fractional radius of the inner subtracted disk with respect to the radius of the outer disk
               fo (float): Fractional offset of the inner disk from the center of the outer disk
               ff (float): Fractional brightness of the inner disk
               phi (float): angle of offset of the inner disk
               x0 (float): The x-coordinate (radians)
               y0 (float): The y-coordinate (radians)

           Returns:
                (Model): Updated Model
        """
        out = self.copy()
        out.models.append('blurred_crescent')
        out.params.append({'F0':F0,'d':d,'alpha':alpha, 'fr':fr, 'fo':fo, 'ff':ff, 'phi':phi, 'x0':x0, 'y0':y0, 'pol_frac':pol_frac, 'pol_evpa':pol_evpa, 'cpol_frac':cpol_frac})
        return out

    def add_ring(self, F0 = 1.0, d = 50.*RADPERUAS, x0 = 0.0, y0 = 0.0, pol_frac = 0.0, pol_evpa = 0.0, cpol_frac = 0.0):
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
        out.params.append({'F0':F0,'d':d,'x0':x0,'y0':y0,'pol_frac':pol_frac,'pol_evpa':pol_evpa,'cpol_frac':cpol_frac})
        return out

    def add_stretched_ring(self, F0 = 1.0, d = 50.*RADPERUAS, x0 = 0.0, y0 = 0.0, stretch = 1.0, stretch_PA = 0.0, pol_frac = 0.0, pol_evpa = 0.0, cpol_frac = 0.0):
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
        out.params.append({'F0':F0,'d':d,'x0':x0,'y0':y0,'stretch':stretch,'stretch_PA':stretch_PA,'pol_frac':pol_frac,'pol_evpa':pol_evpa,'cpol_frac':cpol_frac})
        return out

    def add_thick_ring(self, F0 = 1.0, d = 50.*RADPERUAS, alpha = 10.*RADPERUAS, x0 = 0.0, y0 = 0.0, pol_frac = 0.0, pol_evpa = 0.0, cpol_frac = 0.0):
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
        out.params.append({'F0':F0,'d':d,'alpha':alpha,'x0':x0,'y0':y0,'pol_frac':pol_frac,'pol_evpa':pol_evpa,'cpol_frac':cpol_frac})
        return out

    def add_stretched_thick_ring(self, F0 = 1.0, d = 50.*RADPERUAS, alpha = 10.*RADPERUAS, x0 = 0.0, y0 = 0.0, stretch = 1.0, stretch_PA = 0.0, pol_frac = 0.0, pol_evpa = 0.0, cpol_frac = 0.0):
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
        out.params.append({'F0':F0,'d':d,'alpha':alpha,'x0':x0,'y0':y0,'stretch':stretch,'stretch_PA':stretch_PA,'pol_frac':pol_frac,'pol_evpa':pol_evpa,'cpol_frac':cpol_frac})
        return out

    def add_mring(self, F0 = 1.0, d = 50.*RADPERUAS, x0 = 0.0, y0 = 0.0, beta_list = None, beta_list_pol = None, beta_list_cpol = None):
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
        if beta_list is None: beta_list = []
        if beta_list_pol is None: beta_list_pol = []
        if beta_list_cpol is None: beta_list_cpol = []

        out = self.copy()
        if beta_list is None:
            beta_list = [0.0]
        out.models.append('mring')
        out.params.append({'F0':F0,'d':d,'beta_list':np.array(beta_list, dtype=np.complex_),'beta_list_pol':np.array(beta_list_pol, dtype=np.complex_),'beta_list_cpol':np.array(beta_list_cpol, dtype=np.complex_),'x0':x0,'y0':y0})
        return out

    def add_stretched_mring(self, F0 = 1.0, d = 50.*RADPERUAS, x0 = 0.0, y0 = 0.0, beta_list = None, beta_list_pol = None, beta_list_cpol = None, stretch = 1.0, stretch_PA = 0.0):
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
        if beta_list is None: beta_list = []
        if beta_list_pol is None: beta_list_pol = []
        if beta_list_cpol is None: beta_list_cpol = []

        out = self.copy()
        if beta_list is None:
            beta_list = [0.0]
        out.models.append('stretched_mring')
        out.params.append({'F0':F0,'d':d,'beta_list':np.array(beta_list, dtype=np.complex_),'beta_list_pol':np.array(beta_list_pol, dtype=np.complex_),'beta_list_cpol':np.array(beta_list_cpol, dtype=np.complex_),'x0':x0,'y0':y0,'stretch':stretch,'stretch_PA':stretch_PA})
        return out

    def add_thick_mring(self, F0 = 1.0, d = 50.*RADPERUAS, alpha = 10.*RADPERUAS, x0 = 0.0, y0 = 0.0, beta_list = None, beta_list_pol = None, beta_list_cpol = None):
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
        if beta_list is None: beta_list = []
        if beta_list_pol is None: beta_list_pol = []
        if beta_list_cpol is None: beta_list_cpol = []

        out = self.copy()
        if beta_list is None:
            beta_list = [0.0]
        out.models.append('thick_mring')
        out.params.append({'F0':F0,'d':d,'beta_list':np.array(beta_list, dtype=np.complex_),'beta_list_pol':np.array(beta_list_pol, dtype=np.complex_),'beta_list_cpol':np.array(beta_list_cpol, dtype=np.complex_),'alpha':alpha,'x0':x0,'y0':y0})
        return out

    def add_thick_mring_floor(self, F0 = 1.0, d = 50.*RADPERUAS, alpha = 10.*RADPERUAS, ff=0.0, x0 = 0.0, y0 = 0.0, beta_list = None, beta_list_pol = None, beta_list_cpol = None):
        """Add a ring model with azimuthal brightness variations determined by a Fourier mode expansion, thickness determined by circular Gaussian convolution, and a floor
           The floor is a blurred disk, with diameter d and blurred FWHM alpha
           For details, see Eq. 18-20 of https://arxiv.org/abs/1907.04329
           The Gaussian convolution calculation is a trivial generalization of Appendix G of https://iopscience.iop.org/article/10.3847/2041-8213/ab0e85/pdf

           Args:
               F0 (float): The total flux of the ring (Jy), which is also beta_0.
               d (float): The ring diameter (radians)
               alpha (float): The ring thickness (FWHM of Gaussian convolution) (radians)
               x0 (float): The x-coordinate (radians)
               y0 (float): The y-coordinate (radians)
               ff (float): The fraction of the total flux in the floor
               beta_list (list): List of complex Fourier coefficients, [beta_1, beta_2, ...]. 
                                 Negative indices are determined by the condition beta_{-m} = beta_m*.
                                 Indices are all scaled by F0 = beta_0, so they are dimensionless. 
           Returns:
                (Model): Updated Model
        """
        if beta_list is None: beta_list = []
        if beta_list_pol is None: beta_list_pol = []
        if beta_list_cpol is None: beta_list_cpol = []

        out = self.copy()
        if beta_list is None:
            beta_list = [0.0]
        out.models.append('thick_mring_floor')
        out.params.append({'F0':F0,'d':d,'beta_list':np.array(beta_list, dtype=np.complex_),'beta_list_pol':np.array(beta_list_pol, dtype=np.complex_),'beta_list_cpol':np.array(beta_list_cpol, dtype=np.complex_),'alpha':alpha,'x0':x0,'y0':y0,'ff':ff})
        return out

    def add_thick_mring_Gfloor(self, F0 = 1.0, d = 50.*RADPERUAS, alpha = 10.*RADPERUAS, ff=0.0, FWHM = 50.*RADPERUAS, x0 = 0.0, y0 = 0.0, beta_list = None, beta_list_pol = None, beta_list_cpol = None):
        """Add a ring model with azimuthal brightness variations determined by a Fourier mode expansion, thickness determined by circular Gaussian convolution, and a floor
           The floor is a circular Gaussian, with size FWHM
           For details, see Eq. 18-20 of https://arxiv.org/abs/1907.04329
           The Gaussian convolution calculation is a trivial generalization of Appendix G of https://iopscience.iop.org/article/10.3847/2041-8213/ab0e85/pdf

           Args:
               F0 (float): The total flux of the model
               d (float): The ring diameter (radians)
               alpha (float): The ring thickness (FWHM of Gaussian convolution) (radians)
               FWHM (float): The Gaussian FWHM
               x0 (float): The x-coordinate (radians)
               y0 (float): The y-coordinate (radians)
               ff (float): The fraction of the total flux in the floor
               beta_list (list): List of complex Fourier coefficients, [beta_1, beta_2, ...]. 
                                 Negative indices are determined by the condition beta_{-m} = beta_m*.
                                 Indices are all scaled by F0 = beta_0, so they are dimensionless. 
           Returns:
                (Model): Updated Model
        """
        if beta_list is None: beta_list = []
        if beta_list_pol is None: beta_list_pol = []
        if beta_list_cpol is None: beta_list_cpol = []

        out = self.copy()
        if beta_list is None:
            beta_list = [0.0]
        out.models.append('thick_mring_Gfloor')
        out.params.append({'F0':F0,'d':d,'beta_list':np.array(beta_list, dtype=np.complex_),'beta_list_pol':np.array(beta_list_pol, dtype=np.complex_),'beta_list_cpol':np.array(beta_list_cpol, dtype=np.complex_),'alpha':alpha,'x0':x0,'y0':y0,'ff':ff,'FWHM':FWHM})
        return out

    def add_stretched_thick_mring(self, F0 = 1.0, d = 50.*RADPERUAS, alpha = 10.*RADPERUAS, x0 = 0.0, y0 = 0.0, beta_list = None, beta_list_pol = None, beta_list_cpol = None, stretch = 1.0, stretch_PA = 0.0):
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
        if beta_list is None: beta_list = []
        if beta_list_pol is None: beta_list_pol = []
        if beta_list_cpol is None: beta_list_cpol = []

        out = self.copy()
        if beta_list is None:
            beta_list = [0.0]
        out.models.append('stretched_thick_mring')
        out.params.append({'F0':F0,'d':d,'beta_list':np.array(beta_list, dtype=np.complex_),'beta_list_pol':np.array(beta_list_pol, dtype=np.complex_),'beta_list_cpol':np.array(beta_list_cpol, dtype=np.complex_),'alpha':alpha,'x0':x0,'y0':y0,'stretch':stretch,'stretch_PA':stretch_PA})
        return out

    def add_stretched_thick_mring_floor(self, F0 = 1.0, d = 50.*RADPERUAS, alpha = 10.*RADPERUAS, ff=0.0, x0 = 0.0, y0 = 0.0, beta_list = None, beta_list_pol = None, beta_list_cpol = None, stretch = 1.0, stretch_PA = 0.0):
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
        if beta_list is None: beta_list = []
        if beta_list_pol is None: beta_list_pol = []
        if beta_list_cpol is None: beta_list_cpol = []

        out = self.copy()
        if beta_list is None:
            beta_list = [0.0]
        out.models.append('stretched_thick_mring_floor')
        out.params.append({'F0':F0,'d':d,'beta_list':np.array(beta_list, dtype=np.complex_),'beta_list_pol':np.array(beta_list_pol, dtype=np.complex_),'beta_list_cpol':np.array(beta_list_cpol, dtype=np.complex_),'alpha':alpha,'x0':x0,'y0':y0,'stretch':stretch,'stretch_PA':stretch_PA,'ff':ff})
        return out

    def sample_xy(self, x, y, psize=1.*RADPERUAS, pol='I'):
        """Sample model image on the specified x and y coordinates

           Args:
               x (float): x coordinate (dimensionless)
               y (float): y coordinate (dimensionless)

           Returns:
               (float): Image brightness (Jy/radian^2)
        """  
        return sample_model_xy(self.models, self.params, x, y, psize=psize, pol=pol)

    def sample_uv(self, u, v, polrep_obs='Stokes', pol='I', jonesdict=None):
        """Sample model visibility on the specified u and v coordinates

           Args:
               u (float): u coordinate (dimensionless)
               v (float): v coordinate (dimensionless)

           Returns:
               (complex): complex visibility (Jy)
        """   
        return sample_model_uv(self.models, self.params, u, v, pol=pol, jonesdict=jonesdict)

    def sample_graduv_uv(self, u, v, pol='I', jonesdict=None):
        """Sample model visibility gradient on the specified u and v coordinates wrt (u,v)

           Args:
               u (float): u coordinate (dimensionless)
               v (float): v coordinate (dimensionless)

           Returns:
               (complex): complex visibility (Jy)
        """   
        return sample_model_graduv_uv(self.models, self.params, u, v, pol=pol, jonesdict=jonesdict)

    def sample_grad_uv(self, u, v, pol='I', fit_pol=False, fit_cpol=False, fit_leakage=False, jonesdict=None):
        """Sample model visibility gradient on the specified u and v coordinates wrt all model parameters

           Args:
               u (float): u coordinate (dimensionless)
               v (float): v coordinate (dimensionless)

           Returns:
               (complex): complex visibility (Jy)
        """   
        return sample_model_grad_uv(self.models, self.params, u, v, pol=pol, fit_pol=fit_pol, fit_cpol=fit_cpol, fit_leakage=fit_leakage, jonesdict=jonesdict)

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

    def default_prior(self,fit_pol=False,fit_cpol=False):
        return [default_prior(self.models[j],self.params[j],fit_pol=fit_pol,fit_cpol=fit_cpol) for j in range(self.N_models())]        

    def display(self, fov=FOV_DEFAULT, npix=NPIX_DEFAULT, polrep='stokes', pol_prim=None, pulse=PULSE_DEFAULT, time=0., **kwargs):        
        return self.make_image(fov, npix, polrep, pol_prim, pulse, time).display(**kwargs)

    def make_image(self, fov, npix, polrep='stokes', pol_prim=None, pulse=PULSE_DEFAULT, time=0.):
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
        outim = image.Image(imarr, pdim, self.ra, self.dec,
                      polrep=polrep, pol_prim=pol_prim,
                      rf=self.rf, source=self.source, mjd=self.mjd, time=time, pulse=pulse)
        
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

        # Add the remaining polarizations
        for pol in ['Q','U','V']:
            out.add_pol_image(self.sample_xy(x_grid, y_grid, im.psize, pol=pol), pol)

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

        # Load optional parameters
        jonesdict = kwargs.get('jonesdict',None)

        # Compute visibilities and put them into the obsdata
        if obs.polrep=='stokes':
            obsdata['vis']  = self.sample_uv(obs.data['u'], obs.data['v'], pol='I', jonesdict=jonesdict)
            obsdata['qvis'] = self.sample_uv(obs.data['u'], obs.data['v'], pol='Q', jonesdict=jonesdict)
            obsdata['uvis'] = self.sample_uv(obs.data['u'], obs.data['v'], pol='U', jonesdict=jonesdict)
            obsdata['vvis'] = self.sample_uv(obs.data['u'], obs.data['v'], pol='V', jonesdict=jonesdict)
        elif obs.polrep=='circ':
            obsdata['rrvis'] = self.sample_uv(obs.data['u'], obs.data['v'], pol='RR', jonesdict=jonesdict)
            obsdata['rlvis'] = self.sample_uv(obs.data['u'], obs.data['v'], pol='RL', jonesdict=jonesdict)
            obsdata['lrvis'] = self.sample_uv(obs.data['u'], obs.data['v'], pol='LR', jonesdict=jonesdict)
            obsdata['llvis'] = self.sample_uv(obs.data['u'], obs.data['v'], pol='LL', jonesdict=jonesdict)

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
                           dterm_offset=DTERMPDEF,                            
                           rlratio_std=0.,rlphase_std=0.,
                           caltable_path=None, seed=False, **kwargs):

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
                              or a dict giving one std. dev. per site      

               gain_offset (float): the base gain offset at all sites,
                                    or a dict giving one gain offset per site
               dterm_offset (float): the base std. dev. of random additive error at all sites,
                                    or a dict giving one std. dev. per site

               rlratio_std (float): the fractional std. dev. of the R/L gain offset
                                    or a dict giving one std. dev. per site                                          
               rlphase_std (float): std. dev. of R/L phase offset, 
                                    or a dict giving one std. dev. per site
                                    a negative value samples from uniform                                          

               caltable_path (string): The path and prefix of a saved caltable

               seed (int): seeds the random component of the noise terms. DO NOT set to 0!

           Returns:
               (Obsdata): an observation object
        """

        if seed!=False:
            np.random.seed(seed=seed)

        obs = self.observe_same_nonoise(obs_in, **kwargs)

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
                                                 rlratio_std=rlratio_std,rlphase_std=rlphase_std,
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
                      dterm_offset=DTERMPDEF, rlratio_std=0.,rlphase_std=0.,
                      seed=False, **kwargs):

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
               gainp (float): the fractional std. dev. of the random error on the gains
                              or a dict giving one std. dev. per site      

               gain_offset (float): the base gain offset at all sites,
                                    or a dict giving one gain offset per site
               dterm_offset (float): the base std. dev. of random additive error at all sites,
                                    or a dict giving one std. dev. per site

               rlratio_std (float): the fractional std. dev. of the R/L gain offset
                                    or a dict giving one std. dev. per site                                          
               rlphase_std (float): std. dev. of R/L phase offset, 
                                    or a dict giving one std. dev. per site
                                    a negative value samples from uniform                                          

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
                                     rlratio_std=rlratio_std,rlphase_std=rlphase_std,
                                     jones=jones, inv_jones=inv_jones, seed=seed, **kwargs)

        obs.mjd = mjd

        return obs

    def save_txt(self,filename):
        # Header
        import ehtim.observing.obs_helpers as obshelp
        mjd = float(self.mjd)
        time = self.time
        mjd += (time/24.)

        head = ("SRC: %s \n" % self.source +
                "RA: " + obshelp.rastring(self.ra) + "\n" + 
                "DEC: " + obshelp.decstring(self.dec) + "\n" +
                "MJD: %.6f \n" % (float(mjd)) +  
                "RF: %.4f GHz" % (self.rf/1e9))
        # Models
        out = []
        for j in range(self.N_models()):
            out.append(self.models[j])
            out.append(str(self.params[j]).replace('\n','').replace('complex128','np.complex128').replace('array','np.array'))
        np.savetxt(filename, out, header=head, fmt="%s")

    def load_txt(self,filename):
        lines = open(filename).read().splitlines()
        
        src = ' '.join(lines[0].split()[2:])
        ra = lines[1].split()
        self.ra = float(ra[2]) + float(ra[4])/60.0 + float(ra[6])/3600.0
        dec = lines[2].split()
        self.dec = np.sign(float(dec[2])) * (abs(float(dec[2])) + float(dec[4])/60.0 + float(dec[6])/3600.0)
        mjd_float = float(lines[3].split()[2])
        self.mjd = int(mjd_float)
        self.time = (mjd_float - self.mjd) * 24
        self.rf = float(lines[4].split()[2]) * 1e9

        self.models = lines[5::2]
        self.params = [eval(x) for x in lines[6::2]]

def load_txt(filename):
    out = Model()
    out.load_txt(filename)
    return out
