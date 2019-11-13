# model.py
# an interferometric model class

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

###########################################################################################################################################
# Model types
###########################################################################################################################################
# point: point source
#   F0: total flux density
#   (x0, y0): location
#
# gauss: Gaussian 
#   F0: total flux density
#   (x0, y0): centroid
#   FWHM_maj: 
#   FWHM_min:
#   PA: position angle 
#
# disk: disk with uniform brightness
#   F0: total flux density
#   (x0, y0): centroid
#   d: disk diameter 
#
# ring: ring with brightness asymmetry
#   F0: total flux density
#   (x0, y0): centroid
#   d: ring diameter 
#   beta_list: list of angular mode coefficients [\beta_1, \beta_2, ...]. Total flux density F0 is \beta_0.
#
# thick_ring: uniform ring with finite thickness (Gaussian convolution with FWHM alpha; see EHTC Paper IV)
#   F0: total flux density
#   (x0, y0): centroid
#   d: ring diameter 
#   alpha: FWHM of blurring kernel
#
# mring: ring with brightness asymmetry (see https://arxiv.org/abs/1907.04329)
#   F0: total flux density
#   (x0, y0): centroid
#   d: ring diameter 
#   beta_list: list of angular mode coefficients [\beta_1, \beta_2, ...]. \beta_0 is assumed to be 1, with everything scaled by total flux density F0.
###########################################################################################################################################

###########################################################################################################################################
#Model object
###########################################################################################################################################

def sample_1model_xy(x, y, model_type, params, psize=1.*RADPERUAS):
    if model_type == 'point':
        return params['F0'] * (np.abs( x - params['x0']) < psize/2.0) * (np.abs( y - params['y0']) < psize/2.0)
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
        return (params['F0']*psize**2/(np.pi*params['d']*psize)
                * (params['d']/2.0 - psize/2 < np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2))
                * (params['d']/2.0 + psize/2 > np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2)))
    elif model_type == 'thick_ring':
        r = np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2)
        return (params['F0']*psize**2 * 4.0 * np.log(2.)/(np.pi * params['alpha']**2)
                * np.exp(-4.*np.log(2.)/params['alpha']**2*(r**2 + params['d']**2/4.))
                * sps.iv(0, 4.*np.log(2.) * r * params['d']/params['alpha']**2)) 
    elif model_type == 'mring':
        phi = np.angle((y - params['y0']) + 1j*(x - params['x0']))
        return (params['F0']*psize**2/(np.pi*params['d']*psize)
                * (1.0 + np.sum([2.*np.real(params['beta_list'][m-1] * np.exp(1j * m * phi)) for m in range(1,len(params['beta_list'])+1)],axis=0))  
                * (params['d']/2.0 - psize/2 < np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2))
                * (params['d']/2.0 + psize/2 > np.sqrt((x - params['x0'])**2 + (y - params['y0'])**2)))
    else:
        return 0.0

def sample_1model_uv(u, v, model_type, params):
    if model_type == 'point':
        return params['F0'] * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))
    elif model_type == 'gauss':
        u_maj = u*np.sin(params['PA']) + v*np.cos(params['PA'])
        u_min = u*np.cos(params['PA']) - v*np.sin(params['PA'])
        return (params['F0'] 
               * np.exp(-np.pi**2/(4.*np.log(2.)) * ((u_maj * params['FWHM_maj'])**2 + (u_min * params['FWHM_min'])**2))
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
    elif model_type == 'disk':
        z = np.pi * params['d'] * (u**2 + v**2)**0.5
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
               + np.sum([params['beta_list'][m-1] * sps.jv(m, z) * np.exp(1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
               + np.sum([params['beta_list'][m-1] * sps.jv(-m, z) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0))
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
    else:
        return 0.0

def sample_1model_grad_uv(u, v, model_type, params):
    # Gradient of the model for each parameter
    if model_type == 'point': # F0, x0, y0
        return np.array([ params['F0'] * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])),
                 1j * 2.0 * np.pi * u * params['F0'] * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])),
                 1j * 2.0 * np.pi * v * params['F0'] * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))])
    elif model_type == 'gauss': 
        u_maj = u*np.sin(params['PA']) + v*np.cos(params['PA'])
        u_min = u*np.cos(params['PA']) - v*np.sin(params['PA'])
        return (params['F0'] 
               * np.exp(-np.pi**2/(4.*np.log(2.)) * ((u_maj * params['FWHM_maj'])**2 + (u_min * params['FWHM_min'])**2))
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
    elif model_type == 'disk':
        z = np.pi * params['d'] * (u**2 + v**2)**0.5
        return (params['F0'] * 2.0/z * sps.jv(1, z) 
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
    elif model_type == 'ring': # F0, d, x0, y0
        z = np.pi * params['d'] * (u**2 + v**2)**0.5
        return np.array([ sps.jv(0, z) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])), 
                -np.pi * (u**2 + v**2)**0.5 * params['F0'] * sps.jv(1, z) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])), 
                 2.0 * np.pi * 1j * u * params['F0'] * sps.jv(0, z) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0'])), 
                 2.0 * np.pi * 1j * v * params['F0'] * sps.jv(0, z) * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))])
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
               + np.sum([params['beta_list'][m-1] * sps.jv(m, z) * np.exp(1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0)
               + np.sum([params['beta_list'][m-1] * sps.jv(-m, z) * np.exp(-1j * m * (phi - np.pi/2.)) for m in range(1,len(params['beta_list'])+1)],axis=0))
               * np.exp(1j * 2.0 * np.pi * (u * params['x0'] + v * params['y0']))) 
    else:
        return 0.0

def sample_model_xy(models, params, x, y, psize=1.*RADPERUAS):
    return np.sum(sample_1model_xy(x, y, models[j], params[j], psize=psize) for j in range(len(models)))

def sample_model_uv(models, params, u, v):
    return np.sum(sample_1model_uv(u, v, models[j], params[j]) for j in range(len(models)))

def sample_model_grad_uv(models, params, u, v):
    # Gradient of a sum of models for each parameter
    return np.concatenate([sample_1model_grad_uv(u, v, models[j], params[j]) for j in range(len(models))])

class Model(object):
    """A model with analytic representations in the image and visibility domains.

       Attributes:
    """

    def __init__(self):

        """A model with analytic representations in the image and visibility domains.

           Args:

           Returns:
        """

        # The model is a sum of component models, each defined by a tag and associated parameters
        self.models = []
        self.params = []

    def copy(self):
        out = Model()
        out.models = copy.deepcopy(self.models)
        out.params = copy.deepcopy(self.params.copy())
        return out

    def N_models(self):
        return len(self.models)

    def total_flux(self):
        return np.sum([self.params[j]['F0'] for j in range(self.N_models())])

    def add_point(self, F0 = 1.0, x0 = 0.0, y0 = 0.0):
        self.models.append('point')
        self.params.append({'F0':F0,'x0':x0,'y0':y0})
        return

    def add_circ_gauss(self, F0 = 1.0, FWHM = 50.*RADPERUAS, x0 = 0.0, y0 = 0.0):
        self.models.append('gauss')
        self.params.append({'F0':F0,'FWHM_maj':FWHM,'FWHM_min':FWHM,'PA':0.0,'x0':x0,'y0':y0})
        return

    def add_gauss(self, F0 = 1.0, FWHM_maj = 50.*RADPERUAS, FWHM_min = 50.*RADPERUAS, PA = 0.0, x0 = 0.0, y0 = 0.0):
        self.models.append('gauss')
        self.params.append({'F0':F0,'FWHM_maj':FWHM_maj,'FWHM_min':FWHM_min,'PA':PA,'x0':x0,'y0':y0})
        return

    def add_disk(self, F0 = 1.0, d = 50.*RADPERUAS, x0 = 0.0, y0 = 0.0):
        self.models.append('disk')
        self.params.append({'F0':F0,'d':d,'x0':x0,'y0':y0})
        return

    def add_ring(self, F0 = 1.0, d = 50.*RADPERUAS, x0 = 0.0, y0 = 0.0):
        self.models.append('ring')
        self.params.append({'F0':F0,'d':d,'x0':x0,'y0':y0})
        return

    def add_thick_ring(self, F0 = 1.0, d = 50.*RADPERUAS, alpha = 10.*RADPERUAS, x0 = 0.0, y0 = 0.0):
        self.models.append('thick_ring')
        self.params.append({'F0':F0,'d':d,'alpha':alpha,'x0':x0,'y0':y0})
        return

    def add_mring(self, F0 = 1.0, d = 50.*RADPERUAS, beta_list = None, x0 = 0.0, y0 = 0.0):
        if beta_list is None:
            beta_list = [0.0]
        self.models.append('mring')
        self.params.append({'F0':F0,'d':d,'beta_list':beta_list,'x0':x0,'y0':y0})
        return

    def sample_xy(self, x, y, psize=1.*RADPERUAS):
        return sample_model_xy(self.models, self.params, x, y, psize=psize)

    def sample_uv(self, u, v):
        """Sample model visibility on the specified u and v coordinates

           Args:
               u (float): u coordinate (dimensionless)
               v (float): v coordinate (dimensionless)

           Returns:
               (complex): complex visibility
        """   
        return sample_model_uv(self.models, self.params, u, v)

    def sample_grad_uv(self, u, v):
        """Sample model visibility gradient on the specified u and v coordinates wrt all parameters

           Args:
               u (float): u coordinate (dimensionless)
               v (float): v coordinate (dimensionless)

           Returns:
               (complex): complex visibility
        """   
        return sample_model_grad_uv(self.models, self.params, u, v)

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

    def observe_same_nonoise(self, obs):
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
                           dterm_offset=DTERMPDEF, caltable_path=None, seed=False):

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
                      dterm_offset=DTERMPDEF, seed=False):

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
        out = []
        for j in range(self.N_models()):
            out.append(self.models[j])
            out.append(str(self.params[j]))
        np.savetxt(filename, out, fmt="%s")

    def load_txt(self,filename):
        lines = open(filename).read().splitlines()
        self.models = lines[::2]
        self.params = [eval(x) for x in lines[1::2]]

def load_txt(filename):
    out = Model()

    lines = open(filename).read().splitlines()
    out.models = lines[::2]
    out.params = [eval(x) for x in lines[1::2]]
    return out
