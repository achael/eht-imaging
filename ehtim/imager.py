# imager.py
# a general interferometric imager class
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

# TODO FIX INDEXING MESS FOR MULTIFREQUENCY POLARIZATION
# TODO for polarization imaging give better control than just initializing at 20% 
#       better initialization for all terms!! in init_imager

from __future__ import division
from __future__ import print_function

from builtins import str
from builtins import range
from builtins import object

import copy
import time
import numpy as np
import scipy.optimize as opt

import ehtim.imaging.imager_utils as imutils
import ehtim.imaging.pol_imager_utils as polutils
import ehtim.imaging.multifreq_imager_utils as mfutils
import ehtim.image
import ehtim.const_def as ehc

MAXIT = 200  # number of iterations
NHIST = 50   # number of steps to store for hessian approx
MAXLS = 40   # maximum number of line search steps in BFGS-B
STOP = 1e-6  # convergence criterion
EPS = 1e-8

DATATERMS = ['vis', 'bs', 'amp', 'cphase', 'cphase_diag', 'camp', 'logcamp', 'logcamp_diag']
DATATERMS_POL = ['pvis', 'm', 'vvis']

REGULARIZERS = ['gs', 'tv', 'tvlog','tv2', 'tv2log', 'l1', 'l1w', 'lA', 'patch',
                'flux', 'cm', 'simple', 'compact', 'compact2', 'rgauss']
REGULARIZERS_POL = ['msimple', 'hw', 'ptv','l1v','l2v','vtv','vtv2','vflux']

REGULARIZERS_SPECIND = ['l2_alpha', 'tv_alpha']
REGULARIZERS_CURV = ['l2_beta', 'tv_beta']
REGULARIZERS_SPECIND_P = ['l2_alphap', 'tv_alphap']
REGULARIZERS_CURV_P = ['l2_betap', 'tv_betap']
REGULARIZERS_RM = ['l2_rm', 'tv_rm']
REGULARIZERS_CM = ['l2_cm', 'tv_cm']
REGULARIZERS_ISPECTRAL = REGULARIZERS_SPECIND + REGULARIZERS_CURV 
REGULARIZERS_POLSPECTRAL = REGULARIZERS_SPECIND_P + REGULARIZERS_CURV_P + REGULARIZERS_RM + REGULARIZERS_CM
REGULARIZERS_SPECTRAL = REGULARIZERS_ISPECTRAL + REGULARIZERS_ISPECTRAL

GRIDDER_P_RAD_DEFAULT = 2
GRIDDER_CONV_FUNC_DEFAULT = 'gaussian'
FFT_PAD_DEFAULT = 2
FFT_INTERP_DEFAULT = 3

REG_DEFAULT = {'simple': 1}
DAT_DEFAULT = {'vis': 100}

REGPARAMS_DEFAULT = {'major':50*ehc.RADPERUAS,
                     'minor':50*ehc.RADPERUAS,
                     'PA':0.,
                     'alpha_A':1.0,
                     'epsilon_tv':0.0}

POLARIZATION_MODES = ['P','QU','IP','IQU','V','IV','IQUV','IPV'] # TODO: treatment of V may be inconsistent
MEANPOL_INIT = 0.2 # mean initial polarization if not in initial image
SIGMAPOL_INIT = 1.e-2 # perturbations to initial polarization if not in initial image
###################################################################################################
# Imager object
###################################################################################################


class Imager(object):
    """A general interferometric imager.
    """

    def __init__(self, obs_in, init_im,
                 prior_im=None, flux=None, data_term=DAT_DEFAULT, reg_term=REG_DEFAULT, **kwargs):

        self.logstr = ""
        self._out_list = []
        
        self._obs_list = []
        self._init_list = []
        self._prior_list = []

        self._reg_term_list = []
        self._dat_term_list = []

        self._maxit_list = []
        self._stop_list = []

        self._pol_list = []
        
        self._flux_list = []
        self._pflux_list = []
        self._vflux_list = []

        self._clipfloor_list = []        
        self._snrcut_list = []
        self._debias_list = []
        self._systematic_noise_list = []
        self._systematic_cphase_noise_list = []
        self._transform_list = []
        self._weighting_list = []

        self._maxset_list = []
        self._cp_uv_min_list = []

        self._reffreq_list = []
        self._mf_list = []
        self._mf_order_list = []
        self._mf_order_pol_list = []
        self._mf_rm_list = []
        self._mf_cm_list = []
                
        # iterations / convergence
        self.maxit_next = kwargs.get('maxit', MAXIT)
        self.stop_next = kwargs.get('stop', STOP)
        
        # Regularizer/data terms for the next imaging iteration
        self.reg_term_next = reg_term   # e.g. [('simple',1), ('l1',10), ('flux',500), ('cm',500)]
        self.dat_term_next = data_term  # e.g. [('amp', 1000), ('cphase',100)]

        # Observations, frequencies
        self.reffreq = init_im.rf
        if isinstance(obs_in, list) or isinstance(obs_in, np.ndarray):
            self._obslist_next = obs_in
            self.obslist_next = obs_in
        else:
            self._obslist_next = [obs_in]
            self.obslist_next = [obs_in]

        # Initial image, prior, flux
        self.init_next = init_im

        if prior_im is None:
            self.prior_next = self.init_next
        else:
            self.prior_next = prior_im

        if flux is None:
            self.flux_next = self.prior_next.total_flux()
        else:
            self.flux_next = flux

        # set polarimetric flux values equal to Stokes I flux by default
        # used in polarimetric regularizer normalization
        self.pflux_next = kwargs.get('pflux', flux)
        self.vflux_next = kwargs.get('vflux', flux)
                
        # Polarization and image transforms
        self.pol_next = kwargs.get('pol', self.init_next.pol_prim)
        self.transform_next = kwargs.get('transform', ['log','mcv'])
        self.transform_next = np.array([self.transform_next]).flatten() #so we can handle multiple transforms
        
        # Weighting/debiasing/snr cut/systematic noise
        self.debias_next = kwargs.get('debias', True)
        snrcut = kwargs.get('snrcut', 0.)
        self.snrcut_next = {key: 0. for key in set(DATATERMS+DATATERMS_POL)}

        if type(snrcut) is dict:
            for key in snrcut.keys():
                self.snrcut_next[key] = snrcut[key]
        else:
            for key in self.snrcut_next.keys():
                self.snrcut_next[key] = snrcut

        self.systematic_noise_next = kwargs.get('systematic_noise', 0.)
        self.systematic_cphase_noise_next = kwargs.get('systematic_cphase_noise', 0.)
        self.weighting_next = kwargs.get('weighting', 'natural')
        
        # Maximal/minimal closure set
        self.maxset_next = kwargs.get('maxset', False)
        self.cp_uv_min = kwargs.get('cp_uv_min', False)  # UV minimum for closure phase

        # clipfloor/initial image normalization
        self.clipfloor_next = kwargs.get('clipfloor', 0.)
        self.norm_init = kwargs.get('norm_init', True)
        self.norm_reg = kwargs.get('norm_reg', False)
        self.beam_size = self.obslist_next[0].res()
        self.regparams = {k: kwargs.get(k, REGPARAMS_DEFAULT[k]) for k in REGPARAMS_DEFAULT.keys()}

        self.chisq_transform = False
        self.chisq_offset_gradient = 0.0

        # FFT parameters
        self._ttype = kwargs.get('ttype', 'nfft')
        self._fft_gridder_prad = kwargs.get('fft_gridder_prad', GRIDDER_P_RAD_DEFAULT)
        self._fft_conv_func = kwargs.get('fft_conv_func', GRIDDER_CONV_FUNC_DEFAULT)
        self._fft_pad_factor = kwargs.get('fft_pad_factor', FFT_PAD_DEFAULT)
        self._fft_interp_order = kwargs.get('fft_interp_order', FFT_INTERP_DEFAULT)
        
        # multifrequency
        self.mf_next = kwargs.get('mf',False)

        self.mf_order = kwargs.get('mf_order',0)
        self.mf_order_pol = kwargs.get('mf_order_pol',0)
        self.mf_rm = kwargs.get('mf_rm',0)
        self.mf_cm = kwargs.get('mf_cm',0)

        # Imager history
        self._change_imgr_params = True
        self.nruns = 0
                
        # Set embedding matrices and prepare imager
        self.check_params()
        self.check_limits()
        self.init_imager()
        

    @property
    def obslist_next(self):
        return self._obslist_next

    @obslist_next.setter
    def obslist_next(self, obslist):
        if not isinstance(obslist, list):
            raise Exception("obslist_next must be a list!")
        self._obslist_next = obslist
        self.freq_list = [obs.rf for obs in self.obslist_next]
        self._logfreqratio_list = [np.log(nu/self.reffreq) for nu in self.freq_list]

    @property
    def obs_next(self):
        """the next Obsdata to be used in imaging
        """
        return self.obslist_next[0]

    @obs_next.setter
    def obs_next(self, obs):
        """the next Obsdata to be used in imaging
        """
        self.obslist_next = [obs]

    def reg_terms_last(self):
        """Return last used regularizer terms.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._reg_term_list[-1]

    def dat_terms_last(self):
        """Return last used data terms.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._dat_term_list[-1]

    def obslist_last(self):
        """Return last used observation.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._obs_list[-1]

    def obs_last(self):
        """Return last used observation.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._obs_list[-1][0]

    def prior_last(self):
        """Return last used prior image.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._prior_list[-1]

    def out_last(self):
        """Return last result.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._out_list[-1]


    def init_last(self):
        """Return last initial image.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._init_list[-1]

    def flux_last(self):
        """Return last total flux constraint.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._flux_list[-1]

    def pflux_last(self):
        """Return last total linear polarimetric flux constraint.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._pflux_list[-1]

    def vflux_last(self):
        """Return last total circular polarimetric flux constraint.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._vflux_list[-1]

    def clipfloor_last(self):
        """Return last clip floor.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._clipfloor_list[-1]

    def pol_last(self):
        """Return last polarization imaged.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._pol_list[-1]

    def maxit_last(self):
        """Return last max_iterations value.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._maxit_list[-1]

    def stop_last(self):
        """Return last convergence value.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._stop_list[-1]
        
    def debias_last(self):
        """Return last debias value.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._debias_list[-1]

    def snrcut_last(self):
        """Return last snrcut value.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._snrcut_list[-1]

    def weighting_last(self):
        """Return last weighting value.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._weighting_list[-1]

    def systematic_noise_last(self):
        """Return last systematic_noise value.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._systematic_noise_list[-1]

    def systematic_cphase_noise_last(self):
        """Return last closure phase systematic noise value (in degree).
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._systematic_cphase_noise_list[-1]

    def maxset_last(self):
        """Return last choice of closure phase maximal/minimal set
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._maxset_list[-1]

    def cp_uv_min_last(self):
        """Return last choice of minimal uvdistance for closure phases
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._cp_uv_min_list[-1]

    def mf_last(self):
        """Return last choice for multifrequency imaging
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._mf_list[-1]

    def reffreq_last(self):
        """Return last choice of order for multifrequency imaging
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._reffreq_list[-1]
        
    def mf_order_last(self):
        """Return last choice of order for multifrequency imaging
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._mf_order_list[-1]

    def mf_order_pol_last(self):
        """Return last choice of order for multifrequency polarimetric imaging
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._mf_order_pol_list[-1]

    def mf_rm_last(self):
        """Return last choice for RM imaging
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._mf_rm_list[-1]

    def mf_cm_last(self):
        """Return last choice for CM imaging
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._mf_cm_list[-1]
                               
    def transform_last(self):
        """Return last image transform used.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._transform_list[-1]

    def converge(self, niter, blur_frac, pol, grads=True, **kwargs):

        blur = blur_frac * self.obs_next.res()
        for repeat in range(niter-1):
            init = self.out_last()
            init = init.blur_circ(blur, blur)
            self.init_next = init
            self.make_image(pol=pol, grads=grads, **kwargs)

        
    def make_image_I(self, grads=True, niter=1, blur_frac=1, **kwargs):
        """Make Stokes I image using current imager settings.
        """
        pol = 'I'
        self.make_image(pol=pol, grads=grads, **kwargs)
        self.converge(niter, blur_frac, pol, grads, **kwargs)

        return self.out_last()


    def make_image_P(self, grads=True, niter=1, blur_frac=1, **kwargs):
        """Make Stokes P polarimetric image using current imager settings.
        """
        pol = 'P'
        self.make_image(pol=pol, grads=grads, **kwargs)
        self.converge(niter, blur_frac, pol, grads, **kwargs)

        return self.out_last()


    def make_image_IP(self, grads=True, niter=1, blur_frac=1, **kwargs):
        """Make Stokes I and P polarimetric image simultaneously using current imager settings.
        """
        pol = 'IP'
        self.make_image(pol=pol, grads=grads, **kwargs)
        self.converge(niter, blur_frac, pol, grads, **kwargs)

        return self.out_last()

    def make_image_V(self, grads=True, niter=1, blur_frac=1, **kwargs):
        """Make Stokes I image using current imager settings.
        """
        pol = 'V'
        self.make_image(pol=pol, grads=grads, **kwargs)
        self.converge(niter, blur_frac, pol, grads, **kwargs)

        return self.out_last()

    def make_image_IV(self, grads=True, niter=1, blur_frac=1, **kwargs):
        """Make Stokes I image using current imager settings.
        """
        pol = 'IV'
        self.make_image(pol=pol, grads=grads, **kwargs)
        self.converge(niter, blur_frac, pol, grads, **kwargs)

        return self.out_last()
                
    def make_image(self, pol=None, grads=True, mf=False, **kwargs):
        """Make an image using current imager settings.

           Args:
               pol (str): which polarization to image
               grads (bool): whether or not to use image gradients
               
               show_updates (bool): whether or not to show imager progress 
               update_interval (int): step interval for plotting if show_updates=True 
               
               mf (bool): whether or not to do multifrequency (spectral index only for now)
               mf_order (int): order for multifrequency spectral index imaging
               mf_order_pol (int): order for multifrequency polarization fraction imaging
               mf_rm (int): order for rotation measure imaging
               mf_cm (int): order for conversion measure imaging
               
           Returns:
               (Image): output image

        """

        print("==============================")
        print("Imager run %i " % (int(self.nruns)+1))
        
        # multifrequency parameters
        self.mf_next = mf
        self.mf_order = kwargs.get('mf_order', self.mf_order)
        self.mf_order_pol = kwargs.get('mf_order_pol', self.mf_order_pol)
        self.mf_rm = kwargs.get('mf_rm', self.mf_rm)
        self.mf_cm = kwargs.get('mf_cm', self.mf_cm)
        
        # polarization parameters
        if pol is None:
            pol_prim = self.pol_next
        else:
            self.pol_next = pol
            pol_prim = pol

        # For polarimetric imaging, we must switch polrep to Stokes
        if self.pol_next in POLARIZATION_MODES:
            print("Imaging Polarization: switching image polrep to Stokes!")
            self.prior_next = self.prior_next.switch_polrep(polrep_out='stokes', pol_prim_out='I')
            self.init_next = self.init_next.switch_polrep(polrep_out='stokes', pol_prim_out='I')
            pol_prim = 'I'

        # Checks and initialize
        self.check_params()
        self.check_limits()
        self.init_imager()

        # Print initial stats
        self._nit = 0
        self._show_updates = kwargs.get('show_updates', True)
        self._update_interval = kwargs.get('update_interval', 1)

        # Plot initial image
        self.plotcur(self._xinit, **kwargs)

        # Minimize
        optdict = {'maxiter': self.maxit_next,
                   'ftol': self.stop_next, 'gtol': self.stop_next,
                   'maxcor': NHIST, 'maxls': MAXLS}
        def callback_func(xcur):
            self.plotcur(xcur, **kwargs)

        print("Imaging . . .")
        tstart = time.time()
        if grads:
            res = opt.minimize(self.objfunc, self._xinit, method='L-BFGS-B', jac=self.objgrad,
                               options=optdict, callback=callback_func)
        else:
            res = opt.minimize(self.objfunc, self._xinit, method='L-BFGS-B',
                               options=optdict, callback=callback_func)
        tstop = time.time()

        # Format output
        out = res.x[:]
        self.tmpout = res.x
         
        # unpack output vector into outarr
        outarr = unpack_imarr(out, self._xarr, self._which_solve)

        # apply image transform to bounded values
        outarr = transform_imarr(outarr, self.transform_next, self._which_solve)

        # get and print final statistics
        outstr = ""
        chi2_term_dict = self.make_chisq_dict(outarr)
        for dname in sorted(self.dat_term_next.keys()):
            for i, obs in enumerate(self.obslist_next):
                if len(self.obslist_next)==1:
                    dname_key = dname
                else:
                    dname_key = dname + ('_%i' % i)
                outstr += "chi2_%s : %0.2f " % (dname_key, chi2_term_dict[dname_key])

        try:
            print("time: %f s" % (tstop - tstart))
            print("J: %f" % res.fun)
            print(outstr)
            if isinstance(res.message,str): print(res.message)
            else: print(res.message.decode())
        except: # TODO -- issues for some users with res.message
            pass

        print("==============================")

        # Format final image object
        outim = self.format_outim(outarr, pol_prim=pol_prim)
        
        # Append to history
        logstr = str(self.nruns) + ": make_image(pol=%s)" % pol
        self._append_image_history(outim, logstr)
        self.nruns += 1

        # Return Image object
        return outim


    def check_params(self):
        """Check parameter consistency.
        """
        if ((self.prior_next.psize != self.init_next.psize) or
            (self.prior_next.xdim != self.init_next.xdim) or
            (self.prior_next.ydim != self.init_next.ydim)):
            raise Exception("Initial image does not match dimensions of the prior image!")

        if ((self.prior_next.rf != self.init_next.rf)):
            raise Exception("Initial image does not have same frequency as prior image!")

        if (self.prior_next.polrep != self.init_next.polrep):
            raise Exception(
                "Initial image polrep does not match prior polrep!")

        if (self.prior_next.polrep == 'circ' and not(self.pol_next in ['RR', 'LL'])):
            raise Exception("Initial image polrep is 'circ': pol_next must be 'RR' or 'LL'")

        if (self.prior_next.polrep == 'stokes' and not(self.pol_next in ['I', 'Q', 'U', 'V', 'P','IP','IQU','IV','IQUV'])):
            raise Exception(
                "Initial image polrep is 'stokes': pol_next must be in 'I', 'Q', 'U', 'V', 'P','IP','IQU','IV','IQUV'!")

        if ('log' in self.transform_next and self.pol_next in ['Q', 'U', 'V']):
            raise Exception("Cannot image Stokes Q, U, V with log image transformation!")

        if(self.pol_next in ['Q', 'U', 'V'] and
           ('gs' in self.reg_term_next.keys() or 'simple' in self.reg_term_next.keys())):
            raise Exception(
                "'simple' and 'gs' regularizers do not work with Stokes Q, U, or V images!")

        if self._ttype not in ['fast', 'direct', 'nfft']:
            raise Exception("Possible ttype values are 'fast', 'direct','nfft'!")
            
        # Catch errors in multifrequency imaging setup
        if self.mf_next:
            if len(set(self.freq_list)) < 2:
                raise Exception("Must have observations at at least two frequencies for multifrequency imaging!")
            if self.mf_order not in [1,2]:
                raise Exception("mf_order must be in [1,2]!")
                
            if (self.pol_next in POLARIZATION_MODES):
                if not (self.pol_next in ['P','QU']):
                    raise Exception("Currently we only support pol_next=P for polarization multifrequency imaging!")
                if self.mf_order_pol not in [0,1,2]:
                    raise Exception("mf_order_pol must be in [0,1,2]!")
                                      
        # Catch errors for polarimetric imaging setup
        if self.pol_next in POLARIZATION_MODES:
            if (self.pol_next in ['P', 'QU','IP','IQU']):
                if 'mcv' not in self.transform_next:
                    raise Exception("%s imaging requires 'mcv' transform!"%self.pol_next)
                if 'vcv' in self.transform_next:
                    raise Exception("Cannot do %s imaging with 'vcv' transform!"%self.pol_next)
                if 'polcv' in self.transform_next:
                    raise Exception("Cannot do %s imaging only with 'polcv' transform!"%self.pol_next)
                                        
            if (self.pol_next in ['V','IV']):
                if 'vcv' not in self.transform_next:
                    raise Exception("%s imaging requires 'vcv' transform!"%self.pol_next)
                if 'mcv' in self.transform_next:
                    raise Exception("Cannot do %s imaging only with 'mcv' transform!"%self.pol_next)
                if 'polcv' in self.transform_next:
                    raise Exception("Cannot do %s imaging only with 'polcv' transform!"%self.pol_next)                  
                    
            if self.pol_next in ['IPV','IQUV']:
                if 'polcv' not in self.transform_next:
                    raise Exception("Linear+Circular polarization imaging requires 'polcv' transform!")  
                                                     
            if (self._ttype not in ["direct", "nfft"]):
                raise Exception("FFT not yet implemented in polarimetric imaging -- use NFFT!")

        # catch errors in general imaging setup               
        if self.mf_next:
            if self.pol_next in POLARIZATION_MODES:
                if 'I' in self.pol_next:
                    rlist = REGULARIZERS + REGULARIZERS_POL + REGULARIZERS_SPECTRAL
                    dlist = DATATERMS + DATATERMS_POL
                else:
                    rlist = REGULARIZERS_POL + REGULARIZERS_POLSPECTRAL
                    dlist = DATATERMS_POL
            else:
                rlist = REGULARIZERS_
                dlist = DATATERMS        
        else:     
            if self.pol_next in POLARIZATION_MODES:
                if 'I' in self.pol_next:
                    rlist = REGULARIZERS + REGULARIZERS_POL
                    dlist = DATATERMS + DATATERMS_POL
                else:
                    rlist = REGULARIZERS_POL
                    dlist = DATATERMS_POL
            else:
                rlist = REGULARIZERS
                dlist = DATATERMS
 

        dt_here = False
        dt_type = True
        for term in sorted(self.dat_term_next.keys()):
            if (term is not None) and (term is not False):
                dt_here = True
            if not ((term in dlist) or (term is False)):
                dt_type = False

        st_here = False
        st_type = True
        for term in sorted(self.reg_term_next.keys()):
            if (term is not None) and (term is not False):
                st_here = True
            if not ((term in rlist) or (term is False)):
                st_type = False

        if not dt_here:
            raise Exception("Must have at least one data term!")
        if not st_here:
            raise Exception("Must have at least one regularizer term!")
        if not dt_type:
            raise Exception("Invalid data term: valid data terms are: " + ','.join(dlist))
        if not st_type:
            raise Exception("Invalid regularizer: valid regularizers are: " + ','.join(rlist))

                    
        # Determine if we need to recompute the saved imager parameters on the next imager run
        if self.nruns == 0:
            return

        if self.pol_next != self.pol_last():
            print("changed polarization!")
            self._change_imgr_params = True
            return

        if self.obslist_next != self.obslist_last():
            print("changed observation!")
            self._change_imgr_params = True
            return

        if len(self.reg_term_next) != len(self.reg_terms_last()):
            print("changed number of regularizer terms!")
            self._change_imgr_params = True
            return

        if len(self.dat_term_next) != len(self.dat_terms_last()):
            print("changed number of data terms!")
            self._change_imgr_params = True
            return

        for term in sorted(self.dat_term_next.keys()):
            if term not in self.dat_terms_last().keys():
                print("added %s to data terms" % term)
                self._change_imgr_params = True
                return

        for term in sorted(self.reg_term_next.keys()):
            if term not in self.reg_terms_last().keys():
                print("added %s to regularizers!" % term)
                self._change_imgr_params = True
                return

        if ((self.prior_next.psize != self.prior_last().psize) or
            (self.prior_next.xdim != self.prior_last().xdim) or
            (self.prior_next.ydim != self.prior_last().ydim)):
            print("changed prior dimensions!")
            self._change_imgr_params = True
            return
            
        if self.debias_next != self.debias_last():
            print("changed debiasing!")
            self._change_imgr_params = True
            return
        if self.snrcut_next != self.snrcut_last():
            print("changed snrcut!")
            self._change_imgr_params = True
            return
        if self.weighting_next != self.weighting_last():
            print("changed data weighting!")
            self._change_imgr_params = True
            return
        if self.systematic_noise_next != self.systematic_noise_last():
            print("changed systematic noise!")
            self._change_imgr_params = True
            return
        if self.systematic_cphase_noise_next != self.systematic_cphase_noise_last():
            print("changed systematic cphase noise!")
            self._change_imgr_params = True
            return
        if self.cp_uv_min != self.cp_uv_min_last():
            print("changed cphase maximal/minimal set!")
            self._change_imgr_params = True
            return
        if self.reffreq != self.reffreq_last():
            print("changed refrence frequency!")
            self._change_imgr_params = True
            return
        if self.mf_next != self.mf_last():
            print("changed multifrequncy strategy!")
            self._change_imgr_params = True
            return
        if self.mf_order != self.mf_order_last():
            print("changed multifrequncy order!")
            self._change_imgr_params = True
            return
        if self.mf_order_pol != self.mf_order_pol_last():
            print("changed pol. multifrequncy order!")
            self._change_imgr_params = True
            return
        if self.mf_rm != self.mf_rm_last():
            print("changed pol. rm imaging order!")
            self._change_imgr_params = True
            return                                                           
        if self.mf_cm != self.mf_cm_last():
            print("changed pol. cm imaging order!")
            self._change_imgr_params = True
            return                                                           
                                                                                                                                  
        return
        
    def check_limits(self):
        """Check image parameter consistency with observation.
        """
        uvmax = 1.0/self.prior_next.psize
        uvmin = 1.0/(self.prior_next.psize*np.max((self.prior_next.xdim, self.prior_next.ydim)))
        uvdists = self.obs_next.unpack('uvdist')['uvdist']
        maxbl = np.max(uvdists)
        minbl = np.max(uvdists[uvdists > 0])

        if uvmax < maxbl:
            print("Warning! Pixel size is larger than smallest spatial wavelength!")
        if uvmin > minbl:
            print("Warning! Field of View is smaller than largest nonzero spatial wavelength!")

        if self.pol_next in ['I', 'RR', 'LL']:
            maxamp = np.max(np.abs(self.obs_next.unpack('amp')['amp']))
            if self.flux_next > 1.2*maxamp:
                print("Warning! Specified flux is > 120% of maximum visibility amplitude!")
            if self.flux_next < .8*maxamp:
                print("Warning! Specified flux is < 80% of maximum visibility amplitude!")
        return


    def init_imager(self):
        """Set up Stokes I imager.
        """
        # Set embedding mask
        self.set_embed()
        self._nimage = np.sum(self._embed_mask)
        
        # Set prior & initial image vectors for multifrequency imaging
        if self.mf_next:

            # set reference frequency to same as prior
            self.reffreq = self.init_next.rf 
            
            # reset logfreqratios in case the reference frequency changed
            self._logfreqratio_list = [np.log(nu/self.reffreq) for nu in self.freq_list]

            # determine self._which_solve
            # TODO is there a nicer way to do this? 
            if self.mf_order==2:
                do_a = 1; do_b = 1;
            elif self.mf_order==1:
                do_a = 1; do_b = 0;
            else:
                raise Exception("Imager.mf_order must be 1 or 2!")
                           
            # polarization multi-frequency
            if self.pol_next in POLARIZATION_MODES:

                # determine self._which_solve
                # TODO is there a nicer way to do this?            
                if self.mf_order_pol == 2:
                    do_ap=1; do_bp=1     
                elif self.mf_order_pol == 1:   
                    do_ap=1; do_bp=0  
                elif self.mf_order_pol == 0:
                    do_ap=0; do_bp=0
                else:
                    raise Exception("Imager.mf_order_pol must be 0, 1, or 2!")
                
                if self.mf_rm:
                    do_rm = 1
                else:
                    do_rm = 0
                    
                if self.mf_cm:
                    do_cm = 1
                else:
                    do_cm = 0

                if 'I' in self.pol_next:
                    do_i = 1
                else:
                    do_i = 0
                
                # TODO: No Stokes V imaging for multifrequency yet
                do_rho = 1; do_phi=1              
                do_psi = 0
                if not (('P' in self.pol_next) or ('QU' in self.pol_next)):
                    raise Exception("Multifrequency polarization imaging currently requires pol_next=P!")
                if ('V' in self.pol_next):
                    raise Exception("Stokes V not yet implemented in multifrequency polarization imaging!")
                    
                # set_which_solve vector                
                self._which_solve = np.array([do_i,do_rho,do_phi,do_psi,
                                              do_a,do_b,do_ap,do_bp,
                                              do_rm,do_cm])
        
                # make initial and prior images
                randompol_circ = randompol_lin=False
                if 'V' in self.pol_next: 
                    randompol_circ=True
                if ('P' in self.pol_next) or ('QU' in self.pol_next):
                    randompol_lin=True
                    
                initarr = make_initarr(self.init_next, self._embed_mask, 
                                       norm_init=self.norm_init, flux=self.flux_next, 
                                       mf=True, pol=True, 
                                       randompol_lin=randompol_lin, randompol_circ=randompol_circ, 
                                       meanpol=MEANPOL_INIT, sigmapol=SIGMAPOL_INIT)
                priorarr = make_initarr(self.prior_next, self._embed_mask, 
                                        norm_init=self.norm_init, flux=self.flux_next, 
                                        mf=True, pol=True, 
                                        randompol_lin=False, randompol_circ=False)                    
                # prior
                self._xprior = priorarr

                # transform the initial image to solved for values and assign to self._xarr
                initarr = transform_imarr_inverse(initarr, self.transform_next, self._which_solve)
                self._xarr = initarr
                         
            # Stokes I multi-frequency
            else:

                # set_which_solve vector                
                self._which_solve = np.array([1,do_a,do_b])
                                              
                # make initial and prior images                    
                initarr = make_initarr(self.init_next, self._embed_mask, 
                                       norm_init=self.norm_init, flux=self.flux_next, 
                                       mf=True, pol=False)
                priorarr = make_initarr(self.prior_next, self._embed_mask, 
                                       norm_init=self.norm_init, flux=self.flux_next, 
                                       mf=True, pol=False) 
            
                # prior
                self._xprior = priorarr


                # transform the initial image to solved for values and assign to self._xarr
                initarr = transform_imarr_inverse(initarr, self.transform_next, self._which_solve)
                self._xarr = initarr
                    
            # Pack multi-frequency tuple into single vector
            self._xinit = pack_imarr(self._xarr, self._which_solve)
                
        # Set prior & initial image vectors for single-frequency imaging
        else:
            # single-frequency polarimetric imaging
            if self.pol_next in POLARIZATION_MODES:

                # Determine self._which_solve
                if ('I' in self.pol_next):
                    do_i = 1
                else:
                    do_i = 0
                    
                if ('P' in self.pol_next) or ('QU' in self.pol_next):
                    do_rho = 1; do_phi=1
                else:
                    do_rho = 0; do_phi=0
                    
                if ('V' in self.pol_next):
                    do_psi = 1;
                else:
                    do_psi = 0;    
                
                    
                # set self._which_solve vector                
                self._which_solve = np.array([do_i,do_rho,do_phi,do_psi])
                
                # make initial and prior images
                randompol_circ = randompol_lin=False
                if 'V' in self.pol_next: 
                    randompol_circ=True
                if ('P' in self.pol_next) or ('QU' in self.pol_next):
                    randompol_lin=True
                    
                initarr = make_initarr(self.init_next, self._embed_mask, 
                                       norm_init=self.norm_init, flux=self.flux_next, 
                                       mf=False, pol=True, 
                                       randompol_lin=randompol_lin, randompol_circ=randompol_circ, 
                                       meanpol=MEANPOL_INIT, sigmapol=SIGMAPOL_INIT)
                priorarr = make_initarr(self.prior_next, self._embed_mask, 
                                       norm_init=self.norm_init, flux=self.flux_next, 
                                       mf=False, pol=True, 
                                       randompol_lin=False, randompol_circ=False)                    
                # prior
                self._xprior = priorarr

                # transform the initial image to solved for values and assign to self._xarr
                initarr = transform_imarr_inverse(initarr, self.transform_next, self._which_solve)
                self._xarr = initarr
                                            
                # Pack into single vector
                self._xinit = pack_imarr(self._xarr, self._which_solve)

            # regular single-frequency single-stokes (or RR, LL) imaging
            else:
                # set self._which_solve vector                
                self._which_solve = np.array([1])

                # make initial and prior images                            
                initarr = make_initarr(self.init_next, self._embed_mask, 
                                       norm_init=self.norm_init, flux=self.flux_next, 
                                       mf=False, pol=False)
                priorarr = make_initarr(self.prior_next, self._embed_mask, 
                                       norm_init=self.norm_init, flux=self.flux_next, 
                                       mf=False, pol=False)                    
                                       
                # Prior
                self._xprior = priorarr

                # transform the initial image to solved for values and assign to self._xarr
                initarr = transform_imarr_inverse(initarr, self.transform_next, self._which_solve)
                self._xarr = initarr
                                            
                # Pack into single vector
                self._xinit = pack_imarr(self._xarr, self._which_solve)

        # Make data term tuples
        if self._change_imgr_params:
            if self.nruns == 0:
                print("Initializing imager data products . . .")
            if self.nruns > 0:
                print("Recomputing imager data products . . .")

            self._data_tuples = {}

            # Loop over all data term types
            for dname in sorted(self.dat_term_next.keys()):

                # Loop over all observations in the list
                for i, obs in enumerate(self.obslist_next):
                    # Each entry in the dterm dictionary past the first has an appended number
                    if len(self.obslist_next)==1:
                        dname_key = dname
                    else:
                        dname_key = dname + ('_%i' % i)

                    # Polarimetric data products
                    if dname in DATATERMS_POL:
                        tup = polutils.polchisqdata(obs, self.prior_next, self._embed_mask, dname,
                                                    ttype=self._ttype,
                                                    fft_pad_factor=self._fft_pad_factor,
                                                    conv_func=self._fft_conv_func,
                                                    p_rad=self._fft_gridder_prad)

                    # Single polarization data products
                    elif dname in DATATERMS:
                        if self.pol_next in POLARIZATION_MODES:
                            if not 'I' in self.pol_next:
                                raise Exception("cannot use dterm %s with pol=%s"%(dname,self.pol_next))
                            pol_next = 'I'                            
                        else:
                            pol_next = self.pol_next
                            
                        tup = imutils.chisqdata(obs, self.prior_next, self._embed_mask, dname,
                                                pol=pol_next, maxset=self.maxset_next,
                                                debias=self.debias_next,
                                                snrcut=self.snrcut_next[dname],
                                                weighting=self.weighting_next,
                                                systematic_noise=self.systematic_noise_next,
                                                systematic_cphase_noise=self.systematic_cphase_noise_next,
                                                ttype=self._ttype, order=self._fft_interp_order,
                                                fft_pad_factor=self._fft_pad_factor,
                                                conv_func=self._fft_conv_func,
                                                p_rad=self._fft_gridder_prad,
                                                cp_uv_min=self.cp_uv_min)
                    else:
                        raise Exception("data term %s not recognized!" % dname)

                    self._data_tuples[dname_key] = tup

            self._change_imgr_params = False

        return

    def set_embed(self):
        """Set embedding matrix.
        """
        self._embed_mask = (self.prior_next.imvec > self.clipfloor_next)
        if not np.any(self._embed_mask):
            raise Exception("clipfloor_next too large: all prior pixels have been clipped!")

        xmax = self.prior_next.xdim//2
        ymax = self.prior_next.ydim//2
        
        if self.prior_next.xdim % 2: xmin=-xmax-1
        else: xmin=-xmax
        
        if self.prior_next.ydim % 2: ymin=-ymax-1
        else: ymin=-ymax
        
        coord = np.array([[[x, y]
                           for x in np.arange(xmax, xmin, -1)]
                           for y in np.arange(ymax, ymin, -1)])

        coord = coord.reshape(self.prior_next.ydim * self.prior_next.xdim, 2)
        coord = coord * self.prior_next.psize

        self._coord_matrix = coord[self._embed_mask]

        return


    def make_chisq_dict(self, imcur):
        """Make a dictionary of current chi^2 term values
           input is image array transformed to bounded values
        """

        chi2_dict = {}
        for dname in sorted(self.dat_term_next.keys()):
            # Loop over all observations in the list
            for i, obs in enumerate(self.obslist_next):
                if len(self.obslist_next)==1:
                    dname_key = dname
                else:
                    dname_key = dname + ('_%i' % i)

                # get data products
                (data, sigma, A) = self._data_tuples[dname_key]

                # get current multifrequency image
                if self.mf_next:
                    logfreqratio = self._logfreqratio_list[i]
                    imcur_nu = mfutils.image_at_freq(imcur, logfreqratio)
                else:
                    imcur_nu = imcur
                         
                # Polarization chi^2 terms
                if dname in DATATERMS_POL: 
                    chi2 = polutils.polchisq(imcur_nu, A, data, sigma, dname,
                                             ttype=self._ttype, mask=self._embed_mask)

                # Single Polarization chi^2 terms
                elif dname in DATATERMS:
                    if self.pol_next in POLARIZATION_MODES: 
                        imcur_nu = imcur_nu[0] 

                    chi2 = imutils.chisq(imcur_nu, A, data, sigma, dname,
                                         ttype=self._ttype, mask=self._embed_mask)

                else:
                    raise Exception("data term %s not recognized!" % dname)

                chi2_dict[dname_key] = chi2

        return chi2_dict

    def make_chisqgrad_dict(self, imcur):
        """Make a dictionary of current chi^2 term gradient values
           input is image array transformed to bounded values
        """
        chi2grad_dict = {}
        for dname in sorted(self.dat_term_next.keys()):
            # Loop over all observations in the list
            for i, obs in enumerate(self.obslist_next):
                if len(self.obslist_next)==1:
                    dname_key = dname
                else:
                    dname_key = dname + ('_%i' % i)

                # get data products
                (data, sigma, A) = self._data_tuples[dname_key]

                # get current multifrequency image
                if self.mf_next:
                    logfreqratio = self._logfreqratio_list[i]
                    imcur_nu = mfutils.image_at_freq(imcur, logfreqratio)
                else:
                    imcur_nu = imcur
                
                # Polarimetric chi^2 gradients
                if dname in DATATERMS_POL:
                    if self.mf_next:
                        pol_solve = self._which_solve[0:4]
                    else:
                        pol_solve = self._which_solve
                    chi2grad = polutils.polchisqgrad(imcur_nu, A, data, sigma, dname,
                                                     ttype=self._ttype, mask=self._embed_mask,
                                                     pol_solve=pol_solve)
                                                     
                # Single polarization chi^2 gradients
                elif dname in DATATERMS:
                    if self.pol_next in POLARIZATION_MODES: # polarization
                        imcur_nu = imcur_nu[0]
                        
                    chi2grad = imutils.chisqgrad(imcur_nu, A, data, sigma, dname,
                                                 ttype=self._ttype, mask=self._embed_mask)

                    # If imaging Stokes I with polarization simultaneously, bundle the gradient
                    if self.pol_next in POLARIZATION_MODES: 
                        chi2grad = np.array((chi2grad,np.zeros(self._nimage),np.zeros(self._nimage),np.zeros(self._nimage)))

                else:
                    raise Exception("data term %s not recognized!" % dname)

                # If multifrequency imaging,
                # transform the image gradients for all the solved quantities
                if self.mf_next:
                    logfreqratio = self._logfreqratio_list[i]
                    chi2grad = mfutils.mf_all_grads_chain(chi2grad, imcur_nu, imcur, logfreqratio)

                chi2grad_dict[dname_key] = np.array(chi2grad)

        return chi2grad_dict

    def make_reg_dict(self, imcur):
        """Make a dictionary of current regularizer values
           input is image array transformed to bounded values 
        """
        reg_dict = {}
                           
        for regname in sorted(self.reg_term_next.keys()):
        
            # Multifrequency regularizers
            if self.mf_next:
            
                # Polarimetric regularizers
                if regname in REGULARIZERS_POL:
                    # we only regularize the reference frequency image
                    imcur_pol = imcur[0:4]
                    prior_pol = self._xprior[0:4] 
                    reg = polutils.polregularizer(imcur_pol, prior_pol, self._embed_mask, 
                                                  self.flux_next, self.pflux_next, self.vflux_next,
                                                  self.prior_next.xdim, self.prior_next.ydim, self.prior_next.psize, 
                                                  regname,
                                                  norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                  **self.regparams) 
                                                                             
                # Stokes I regularizers
                elif regname in REGULARIZERS:         
                    # we only regularize the reference frequency image
                    reg = imutils.regularizer(imcur[0], self._xprior[0], self._embed_mask,
                                              self.flux_next, self.prior_next.xdim,
                                              self.prior_next.ydim, self.prior_next.psize,
                                              regname,
                                              norm_reg=self.norm_reg, beam_size=self.beam_size,
                                              **self.regparams)
                
                # Spectral regularizers
                if regname in REGULARIZERS_SPECTRAL:
                    if regname in REGULARIZERS_SPECIND:
                        if len(imcur)==10: idx = 4
                        else: idx=1
                    elif regname in REGULARIZERS_CURV:
                        if len(imcur)==10: idx = 5
                        else: idx=2
                    elif regname in REGULARIZERS_SPECIND_P:
                        idx = 6
                    elif regname in REGULARIZERS_CURV_P:
                        idx = 7
                    elif regname in REGULARIZERS_RM:
                        idx = 8
                    elif regname in REGULARIZERS_CM:
                        idx = 9
                                            
                    reg = mfutils.regularizer_mf(imcur[idx], self._xprior[idx], self._embed_mask,
                                                 self.prior_next.xdim, self.prior_next.ydim, self.prior_next.psize,
                                                 regname,
                                                 norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                 **self.regparams)
                else:
                    raise Exception("regularizer term %s not recognized!" % regname)                            

            # Single-frequency polarimetric regularizer
            elif regname in REGULARIZERS_POL:
                reg = polutils.polregularizer(imcur, self._xprior, self._embed_mask, 
                                              self.flux_next, self.pflux_next, self.vflux_next,
                                              self.prior_next.xdim, self.prior_next.ydim, self.prior_next.psize, 
                                              regname,
                                              norm_reg=self.norm_reg, beam_size=self.beam_size,
                                              **self.regparams) 

            # Single-frequency, single-polarization regularizer                                              
            elif regname in REGULARIZERS:
                if self.pol_next in POLARIZATION_MODES:
                    imcur0 = imcur[0]
                    prior0 = self._xprior[0]
                else:
                    imcur0 = imcur
                    prior0 = self._xprior
                    
                reg = imutils.regularizer(imcur0, prior0, self._embed_mask,
                                          self.flux_next, 
                                          self.prior_next.xdim, self.prior_next.ydim, self.prior_next.psize,
                                          regname,
                                          norm_reg=self.norm_reg, beam_size=self.beam_size,
                                          **self.regparams)
            else:
                raise Exception("regularizer term %s not recognized!" % regname)

            # put regularizer terms in the dictionary
            reg_dict[regname] = reg

        return reg_dict

    def make_reggrad_dict(self, imcur):
        """Make a dictionary of current regularizer gradient values
        """
                
        reggrad_dict = {}
                      
        for regname in sorted(self.reg_term_next.keys()):
            
            # Multifrequency regularizers 
            if self.mf_next:
            
                # Polarimetric regularizers
                if regname in REGULARIZERS_POL:
                    # we only regularize reference frequency image
                    imcur_pol = imcur[0:4]
                    prior_pol = self._xprior[0:4] 
                    pol_solve = self._which_solve[0:4]
                    regp = polutils.polregularizergrad(imcur_pol, prior_pol, self._embed_mask, 
                                                      self.flux_next, self.pflux_next, self.vflux_next,
                                                      self.prior_next.xdim, self.prior_next.ydim, self.prior_next.psize, 
                                                      regname,
                                                      norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                      pol_solve=pol_solve,
                                                      **self.regparams) 
                    reggrad = np.zeros((len(imcur), self._nimage))
                    reggrad[0:4] = regp
                    
                # Stokes I regularizers
                elif regname in REGULARIZERS:         
                    # we only regularize reference frequency image
                    regi = imutils.regularizergrad(imcur[0], self._xprior[0], 
                                                   self._embed_mask, self.flux_next, 
                                                   self.prior_next.xdim, self.prior_next.ydim, self.prior_next.psize,
                                                   regname,
                                                   norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                   **self.regparams)
                    reggrad = np.zeros((len(imcur), self._nimage))
                    reggrad[0] = regi  
                                   
                # Spectral regularizers
                if regname in REGULARIZERS_SPECTRAL:
                    if regname in REGULARIZERS_SPECIND:
                        if len(imcur)==10: idx = 4
                        else: idx=1
                    elif regname in REGULARIZERS_CURV:
                        if len(imcur)==10: idx = 5
                        else: idx=2
                    elif regname in REGULARIZERS_SPECIND_P:
                        idx = 6
                    elif regname in REGULARIZERS_CURV_P:
                        idx = 7
                    elif regname in REGULARIZERS_RM:
                        idx = 8
                    elif regname in REGULARIZERS_CM:
                        idx = 9
                    
                    regmf = mfutils.regularizergrad_mf(imcur[idx], self_xprior[idx], self._embed_mask,
                                                       self.prior_next.xdim, self.prior_next.ydim, self.prior_next.psize,
                                                       regname,
                                                       norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                       **self.regparams)
                    
                    reggrad = np.zeros((len(imcur), self._nimage))
                    reggrad[idx] = regmf  
                else:
                    raise Exception("regularizer term %s not recognized!" % regname)                            
            

                    
            else:        
                # Single-frequency polarimetric regularizer
                if regname in REGULARIZERS_POL:
                    reggrad = polutils.polregularizergrad(imcur, self._xprior, self._embed_mask, 
                                                      self.flux_next, self.pflux_next, self.vflux_next,
                                                      self.prior_next.xdim, self.prior_next.ydim,
                                                      self.prior_next.psize, regname,
                                                      norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                      pol_solve=self._which_solve,
                                                      **self.regparams)


                # Single-frequency, single polarization regularizer
                elif regname in REGULARIZERS:
                    if self.pol_next in POLARIZATION_MODES:
                        imcur0 = imcur[0]
                        prior0 = self._xprior[0]
                    else:
                        imcur0 = imcur
                        prior0 = self._xprior
                    reggrad = imutils.regularizergrad(imcur0, prior0, self._embed_mask, self.flux_next,
                                                      self.prior_next.xdim, self.prior_next.ydim,
                                                      self.prior_next.psize,
                                                      regname,
                                                      norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                      **self.regparams)
                                                  
                    # If imaging Stokes I with polarization simultaneously, bundle the gradient
                    if self.pol_next in POLARIZATION_MODES: 
                        reggrad = np.array((reggrad,np.zeros(self._nimage),np.zeros(self._nimage),np.zeros(self._nimage)))

                else:
                    raise Exception("regularizer term %s not recognized!" % regname)

            # put regularizer terms in the dictionary
            reggrad_dict[regname] = reggrad

        return reggrad_dict

    def objfunc(self, imvec):
        """Current objective function.
        """
        
        # Unpack polarimetric/multifrequency vector into an array
        imcur =  unpack_imarr(imvec, self._xarr, self._which_solve)

        # apply image transform to bounded values
        imcur = transform_imarr(imcur, self.transform_next, self._which_solve)
        
        # Data terms
        datterm = 0.
        chi2_term_dict = self.make_chisq_dict(imcur)
        for dname in sorted(self.dat_term_next.keys()):
            hyperparameter = self.dat_term_next[dname]

            for i, obs in enumerate(self.obslist_next):
                if len(self.obslist_next)==1:
                    dname_key = dname
                else:
                    dname_key = dname + ('_%i' % i)

                chi2 = chi2_term_dict[dname_key]

                if self.chisq_transform:
                    datterm += hyperparameter * (chi2 + 1./chi2 - 1.)
                else:
                    datterm += hyperparameter * (chi2 - 1.)

        # Regularizer terms
        regterm = 0
        reg_term_dict = self.make_reg_dict(imcur)
        for regname in sorted(self.reg_term_next.keys()):
            hyperparameter = self.reg_term_next[regname]
            regularizer = reg_term_dict[regname]
            regterm += hyperparameter * regularizer

        # Total cost
        cost = datterm + regterm
        
        return cost

    def objgrad(self, imvec):
        """Current objective function gradient.
        """

        # Unpack polarimetric/multifrequency vector into an array
        imcur =  unpack_imarr(imvec, self._xarr, self._which_solve)

        # apply image transform to bounded values
        imcur_prime = imcur.copy()
        imcur = transform_imarr(imcur, self.transform_next, self._which_solve)
        
        # Data terms
        datterm = 0.
        chi2_term_dict = self.make_chisqgrad_dict(imcur)
        if self.chisq_transform:
            chi2_value_dict = self.make_chisq_dict(imcur)
        for dname in sorted(self.dat_term_next.keys()):
            hyperparameter = self.dat_term_next[dname]

            for i, obs in enumerate(self.obslist_next):
                if len(self.obslist_next)==1:
                    dname_key = dname
                else:
                    dname_key = dname + ('_%i' % i)

                chi2_grad = chi2_term_dict[dname_key]

                if self.chisq_transform:
                    chi2_val = chi2_value_dict[dname]
                    datterm += hyperparameter * chi2_grad * (1. - 1./(chi2_val**2))
                else:
                    datterm += hyperparameter * (chi2_grad + self.chisq_offset_gradient)

        # Regularizer terms
        regterm = 0
        reg_term_dict = self.make_reggrad_dict(imcur)
        for regname in sorted(self.reg_term_next.keys()):
            hyperparameter = self.reg_term_next[regname]
            regularizer_grad = reg_term_dict[regname]
            regterm += hyperparameter * regularizer_grad

        # Total gradient
        grad = datterm + regterm

        # Chain rule term for change of variables
        grad = transform_gradients(grad, imcur_prime, self.transform_next, self._which_solve)
        
        # repack gradient
        grad = pack_imarr(grad, self._which_solve)
        
        return grad

    def plotcur(self, imvec, **kwargs):
        """Plot current image.
        """

        if self._show_updates:
            if self._nit % self._update_interval == 0:
                # Unpack polarimetric/multifrequency vector into an array
                imcur =  unpack_imarr(imvec, self._xarr, self._which_solve)
                
                # apply image transform to bounded values
                imcur_prime = imcur.copy()
                imcur = transform_imarr(imcur, self.transform_next, self._which_solve)
        
                # Get chi^2 and regularizer
                chi2_term_dict = self.make_chisq_dict(imcur)
                reg_term_dict = self.make_reg_dict(imcur)

                # Format print string
                outstr = "------------------------------------------------------------------"
                outstr += "\n%4d | " % self._nit
                for dname in sorted(self.dat_term_next.keys()):
                    for i, obs in enumerate(self.obslist_next):
                        if len(self.obslist_next)==1:
                            dname_key = dname
                        else:
                            dname_key = dname + ('_%i' % i)
                        outstr += "chi2_%s : %0.2f " % (dname_key, chi2_term_dict[dname_key])
                outstr += "\n        "
                for dname in sorted(self.dat_term_next.keys()):
                    for i, obs in enumerate(self.obslist_next):
                        if len(self.obslist_next)==1:
                            dname_key = dname
                        else:
                            dname_key = dname + ('_%i' % i)
                        dval = chi2_term_dict[dname_key]*self.dat_term_next[dname]
                        outstr += "%s : %0.1f " % (dname_key, dval)

                outstr += "\n        "
                for regname in sorted(self.reg_term_next.keys()):
               
                    rval = reg_term_dict[regname]*self.reg_term_next[regname]
                    outstr += "%s : %0.1f " % (regname, rval)

                # Embed and plot the image
                if np.any(np.invert(self._embed_mask)):                
                    implot = embed_imarr(imcur, self._embed_mask) 
                else:
                    implot = imcur
                    
                if self.pol_next in POLARIZATION_MODES:
                    polutils.plot_m(implot, self.prior_next, self._nit, chi2_term_dict, **kwargs)

                else:
                    imutils.plot_i(implot, self.prior_next, self._nit,
                                   chi2_term_dict, pol=self.pol_next, **kwargs)

                if self._nit == 0:
                    print()
                print(outstr)

        self._nit += 1

                
    def format_outim(self, outarr, pol_prim='I'): 
        """format the final image data into an Image object"""

        # embed the image into the full frame
        if np.any(np.invert(self._embed_mask)):
            outarr = embed_imarr(outarr, self._embed_mask) 
                           
        if self.mf_next:
            # multi-frequency polarization
            if self.pol_next in POLARIZATION_MODES:
                iimage_out = outarr[0]
                polarr_out = (outarr[0], outarr[1], outarr[2], outarr[3])
                specind_out = outarr[4]
                curv_out = outarr[5]
                specind_out_pol = outarr[6]
                curv_out_pol = outarr[7]
                rm_out = outarr[8]
                cm_out = outarr[9]

                qimage_out = polutils.make_q_image(polarr_out)
                uimage_out = polutils.make_u_image(polarr_out)
                vimage_out = polutils.make_v_image(polarr_out)  
                
            # multi-frequency Stokes I
            else:
                iimage_out = outarr[0]
                specind_out = outarr[1]
                curv_out = outarr[2]
        else:
            # single frequency polarization
            if self.pol_next in POLARIZATION_MODES:
                iimage_out = outarr[0]
                polarr_out = (outarr[0], outarr[1], outarr[2], outarr[3])
                
                qimage_out = polutils.make_q_image(polarr_out)
                uimage_out = polutils.make_u_image(polarr_out)
                vimage_out = polutils.make_v_image(polarr_out)
                
            # single frequency Stokes I 
            else: 
                iimage_out = outarr

        # Create Image object
        arglist, argdict = self.prior_next.image_args()
        arglist[0] = iimage_out.reshape(self.prior_next.ydim, self.prior_next.xdim)
        argdict['pol_prim'] = pol_prim
        outim = ehtim.image.Image(*arglist, **argdict)

        # Add polarizations to the output image
        for pol2 in list(outim._imdict.keys()):

            # Is it the base image?
            if pol2 == outim.pol_prim:
                continue

            # Did we solve for polarimeric image or are we copying over old polarization data?
            if self.pol_next in POLARIZATION_MODES and pol2 == 'Q':
                polvec = qimage_out
            elif self.pol_next in POLARIZATION_MODES and pol2 == 'U':
                polvec = uimage_out
            elif self.pol_next in POLARIZATION_MODES and pol2 == 'V':
                polvec = vimage_out
            else:
                polvec = self.init_next._imdict[pol2]

            # Add the current polarization
            if len(polvec):
                outim.add_pol_image(polvec.reshape(outim.ydim, outim.xdim), pol2)

        # Copy over spectral information to the output image
        outim._mflist = copy.deepcopy(self.init_next._mflist)
        if self.mf_next:
            outim._mflist[0] = specind_out
            outim._mflist[1] = curv_out
            
            # polarization multi-frequency
            if self.pol_next in POLARIZATION_MODES: 
                outim._mflist[2] = specind_out_pol
                outim._mflist[3] = curv_out_pol
                outim._mflist[4] = rm_out
                outim._mflist[5] = cm_out
                            
        # Return Image object
        return outim


    def _append_image_history(self, outim, logstr):
        self.logstr += (logstr + "\n")
        self._obs_list.append(self.obslist_next)
        self._init_list.append(self.init_next)
        self._prior_list.append(self.prior_next)
        self._debias_list.append(self.debias_next)
        self._weighting_list.append(self.weighting_next)
        self._systematic_noise_list.append(self.systematic_noise_next)
        self._systematic_cphase_noise_list.append(self.systematic_cphase_noise_next)
        self._snrcut_list.append(self.snrcut_next)
        self._flux_list.append(self.flux_next)
        self._pflux_list.append(self.pflux_next)
        self._vflux_list.append(self.vflux_next)        
        self._pol_list.append(self.pol_next)
        self._clipfloor_list.append(self.clipfloor_next)
        self._maxset_list.append(self.clipfloor_next)
        self._maxit_list.append(self.maxit_next)
        self._stop_list.append(self.stop_next)       
        self._cp_uv_min_list.append(self.cp_uv_min)
                
        self._mf_list.append(self.mf_next)
        self._reffreq_list.append(self.reffreq)        
        self._mf_order_list.append(self.mf_order)
        self._mf_order_pol_list.append(self.mf_order_pol)
        self._mf_rm_list.append(self.mf_rm)
        self._mf_cm_list.append(self.mf_cm)
                                
        self._transform_list.append(self.transform_next)
        self._reg_term_list.append(self.reg_term_next)
        self._dat_term_list.append(self.dat_term_next)

        self._out_list.append(outim)
        return
    
    
#############################################################   
# Helper functions
#############################################################
def embed_imarr(imarr, mask, clipfloor=0., randomfloor=False):
    """Embeds a multidimensional image array into the size of boolean embed mask
    """

    imarrdim = len(imarr.shape)   
    if imarrdim==2:
        nsolve = imarr.shape[0]    
        nimage = imarr.shape[1]
    elif imarrdim==1:
        nsolve = 1
        nimage = imarr.shape[0]
        imarr = imarr.reshape((nsolve,nimage))
    else:
        raise Exception("in embed_imarr, imarr should have one or two dimensions!")
               
    if nimage!=np.sum(mask):
        raise Exception("in embed_imarr, number of masked pixels is not consistent with imarr shape!")

    nimage_out = len(mask)
    outarr = np.empty((nsolve,nimage_out)) 
    # TODO does this require the for loop? 
    for kk in range(nsolve):
        outarr[kk] = imutils.embed(imarr[kk], mask, clipfloor=clipfloor, randomfloor=randomfloor)
    
    if imarrdim==1:
        outarr = outarr[0]
                
    return outarr
    
def pack_imarr(imarr, which_solve):
    """pack image array imarr into 1D array vec for minimizaiton
       ignore quantities not solved for
    """
    imarrdim = len(imarr.shape)   
    if imarrdim==2:
        nsolve = imarr.shape[0]    
        nimage = imarr.shape[1]
    elif imarrdim==1:
        nsolve = 1
        nimage = imarr.shape[0]
        imarr = imarr.reshape((nsolve,nimage))
    else:
        raise Exception("in pack_imarr, imarr should have one or two dimensions!")
        
    if nsolve != len(which_solve):
        raise Exception("in pack_imarr, imarr has inconsistent shape with which_solve!")
        
    vec = np.array([])
    for kk in range(nsolve):
        if which_solve[kk]!=0:
            vec = np.hstack((vec,imarr[kk]))        

    return vec
    
    
def unpack_imarr(vec, priorarr, which_solve):
    """unpack minimized vector vec into array,
       replace quantities not solved for with their initial values
    """
    imarrdim = len(priorarr.shape)
    if imarrdim==2:
        nsolve = priorarr.shape[0]    
        nimage = priorarr.shape[1]
    elif imarrdim==1:
        nsolve = 1
        nimage = priorarr.shape[0]
        imarr = priorarr.reshape((nsolve,nimage))
    else:
        raise Exception("in unpack_imarr, priorarr should have one or two dimensions !")

    if nsolve != len(which_solve):
        raise Exception("in unpack_imarr, priorarr has inconsistent shape with which_solve!")

    imct = 0
    imarr = np.empty((nsolve, nimage))
    for kk in range(nsolve):
        if which_solve[kk]==0:
            imarr[kk] = priorarr[kk]
        else:
            imarr[kk] = vec[imct*nimage:(imct+1)*nimage]
            imct += 1
            
    if imarrdim==1:
        imarr = imarr[0]
    return imarr
    

def transform_imarr(imarr, transforms, which_solve):
    """Apply transformation from solver to physical values for all polarizations"""           
    if ('polcv' in transforms):
        if ('vcv' in transforms) or ('mcv' in transforms):
            raise Exception("'mcv' and 'vcv' are not compatible with 'polcv' image transforms!")
    elif ('vcv' in transforms) and ('mcv' in transforms):
        raise Exception("'mcv' and 'vcv' are not compatible with each other in image transforms!")
        
    imarrdim = len(imarr.shape)   
    if imarrdim==2:
        nimage = imarr.shape[0]
    elif imarrdim==1:
        nimage = 1

    if nimage==1 or nimage==3:
        pol_which_solve = np.array((1,0,0,0)) # single polarization imaging
    elif nimage==4:
        pol_which_solve = which_solve         # single-frequency, multi-polarization imaging
    elif nimage==10:
        pol_which_solve = which_solve[0:4]    # multi-frequency, multi-polarization imaging
    else:
        raise Exception("transform_imarr requires imarr.shape[0] be either 1, 3, 4, or 10!")    

    outarr = imarr.copy()    
    if nimage==1 and ('log' in transforms):
        outarr = np.exp(outarr)   
    elif nimage==3 and ('log' in transforms):
        outarr[0] = np.exp(outarr[0])
    else:

        if pol_which_solve[0]==1 and ('log' in transforms):  # full polarization, including stokes I imaging
            outarr[0] = np.exp(outarr[0])
            
        if (pol_which_solve[1]==1 and pol_which_solve[3]==1 and ('polcv' in transforms)):
            outarr[0:4] = polutils.polcv(outarr)
        elif (pol_which_solve[1]==1) and ('mcv' in transforms):
            outarr[0:4] = polutils.mcv(outarr)
        elif (pol_which_solve[3]==1) and ('vcv' in transforms):
            outarr[0:4] = polutils.vcv(outarr)
             
    return outarr

def transform_imarr_inverse(imarr, transforms, which_solve):
    """Apply inverse transformation from physical to solver values for all polarizations"""           
    if ('polcv' in transforms):
        if ('vcv' in transforms) or ('mcv' in transforms):
            raise Exception("'mcv' and 'vcv' are not compatible with 'polcv' image transforms!")
    elif ('vcv' in transforms) and ('mcv' in transforms):
        raise Exception("'mcv' and 'vcv' are not compatible with each other in image transforms!")
         
    imarrdim = len(imarr.shape)   
    if imarrdim==2:
        nimage = imarr.shape[0]
    elif imarrdim==1:
        nimage = 1
       
    if nimage==1 or nimage==3:
        pol_which_solve = np.array((1,0,0,0)) # single polarization imaging
    elif nimage==4:
        pol_which_solve = which_solve         # single-frequency, multi-polarization imaging
    elif nimage==10:
        pol_which_solve = which_solve[0:4]    # multi-frequency, multi-polarization imaging
    else:
        raise Exception("transform_imarr requires imarr.shape[0] be either 1, 3, 4, or 10!")    
        
    outarr = imarr.copy()
    if nimage==1 and ('log' in transforms):
        outarr = np.log(imarr)         
    elif nimage==3 and ('log' in transforms):
        outarr[0] = np.exp(outarr[0])           
    else:

        if pol_which_solve[0]==1 and ('log' in transforms):  # full polarization, including stokes I imaging
            outarr[0] = np.log(outarr[0])
            
        if (pol_which_solve[1]==1 and pol_which_solve[3]==1 and ('polcv' in transforms)):
            outarr[0:4] = polutils.polcv_r(outarr[0:4])
        elif pol_which_solve[1]==1 and ('mcv' in transforms):
            outarr[0:4] = polutils.mcv_r(outarr[0:4])
        elif pol_which_solve[3]==1 and ('vcv' in transforms):
            outarr[0:4] = polutils.vcv_r(outarr[0:4])
             
    return outarr

def transform_gradients(gradarr, imarr, transforms, which_solve):
    """Apply chain rule gradients for solver values for all polarizations
       gradarr is objective func gradients w/r/t physical variables
       imarr is the current image in solver variables """           

    if ('polcv' in transforms):
        if ('vcv' in transforms) or ('mcv' in transforms):
            raise Exception("'mcv' and 'vcv' are not compatible with 'polcv' image transforms!")
    elif ('vcv' in transforms) and ('mcv' in transforms):
        raise Exception("'mcv' and 'vcv' are not compatible with each other in image transforms!")
        
    imarrdim = len(imarr.shape)   
    if imarrdim==2:
        nimage = imarr.shape[0]
    elif imarrdim==1:
        nimage = 1

    if nimage==1 or nimage==3:
        pol_which_solve = np.array((1,0,0,0)) # single polarization imaging
    elif nimage==4:
        pol_which_solve = which_solve         # single-frequency, multi-polarization imaging
    elif nimage==10:
        pol_which_solve = which_solve[0:4]    # multi-frequency, multi-polarization imaging
    else:
        raise Exception("transform_imarr requires imarr.shape[0] be either 1, 3, 4, or 10!")    

    outarr = gradarr.copy()    
    if nimage==1 and ('log' in transforms):
        outarr = np.exp(imarr) * gradarr
    elif nimage==3 and ('log' in transforms):
        outarr[0] = np.exp(imarr[0]) * gradarr[0]
    else:

        if pol_which_solve[0]==1 and ('log' in transforms):  # full polarization, including stokes I imaging
            outarr[0] = np.exp(imarr[0]) * gradarr[0]
            
        if (pol_which_solve[1]==1 and pol_which_solve[3]==1 and ('polcv' in transforms)):
            outarr[0:4] = polutils.polcv_chain(imarr[0:4]) * gradarr[0:4]
        elif (pol_which_solve[1]==1) and ('mcv' in transforms):
            outarr[0:4] = polutils.mcv_chain(imarr[0:4]) * gradarr[0:4]
        elif (pol_which_solve[3]==1) and ('vcv' in transforms):
            outarr[0:4] = polutils.vcv_chain(imarr[0:4]) * gradarr[0:4]
             
             
    return outarr


def make_initarr(image, mask, norm_init=False, flux=1, 
                 mf=False, pol=False, 
                 randompol_lin=False, randompol_circ=False, 
                 meanpol=0.2, sigmapol=1.e-2):
    """Make initial image array from image object, or initialize with default values"""           
    # set initial and prior images
    init_I = image.imvec[mask]
    nimage = len(init_I)
                    
    if norm_init:
        normfac = init_I / (np.sum(init_I)) 
        init_I = flux * normfac         
    else:
        normfac = 1

    # TODO -- apply a floor to init_I? 

    # single-frequency, single-polarization
    if not(pol) and not(mf):
        initarr = np.array(init_I)
            
    # polarization        
    if pol:
        if len(image.qvec):
            init_q = normfac*image.qvec[mask]
        else:
            init_q = np.zeros(nimage)
        if len(image.uvec):
            init_u = normfac*image.uvec[mask]
        else:
            init_u = np.zeros(nimage)            
        if len(image.vvec):
            init_v = normfac*image.vvec[mask]
        else:
            init_v = normfac*np.zeros(nimage)

        init_P = np.sqrt(init_q**2 + init_u**2)
                     
        init_rho = np.sqrt(init_q**2 + init_u**2 + init_v**2) / init_I
        init_phi = np.arctan2(init_u, init_q)       
        init_psi = np.arctan2(init_v, init_P)
                                
        if not(np.any(init_rho!=0)) and randompol_lin:
            print("No polarimetric image in init!")
            print("--initializing with 20% pol and random orientation!")
            init_rho = meanpol * (np.ones(nimage) + sigmapol * np.random.rand(nimage))
            init_phi = np.zeros(nimage) + sigmapol * np.random.rand(nimage)
            
        if not(np.any(init_psi!=0)) and randompol_circ:                
            print("No circular polarization image in init!")
            print("--initializing with random values!")                
            init_rho = meanpol * (np.ones(nimage) + sigmapol * np.random.rand(nimage))
            init_psi = np.zeros(nimage) + sigmapol * np.random.rand(nimage)
        
        if not(mf):
            initarr = np.array((init_I, init_rho, init_phi, init_psi))
     
    # multi-frequency        
    if mf: 
        if len(image.specvec):
            init_a = image.specvec[mask]
        else:
            init_a = np.zeros(nimage)
            
        if len(image.curvvec):
            init_b = image.curvvec[mask]
        else:
            init_b = np.zeros(nimage)
            
        # multi-frequency, multi-polarization    
        if pol:
            if len(image.specvec_pol):
                init_ap = image.specvec_pol[mask]
            else:
                init_ap = np.zeros(nimage)

            if len(image.curvvec_pol):
                init_bp = imate.curvvec_pol[mask]
            else:
                init_bp = np.zeros(nimage)

            # TODO what do we want to initialize RM and CM to? 
            if len(image.rmvec):
                init_rm = image.rmvec[mask]
            else:
                init_rm = np.zeros(nimage)

            if len(image.cmvec):
                init_cm = image.cmvec[mask]
            else:
                init_cm = np.zeros(nimage)

            initarr = np.array((init_I, init_rho, init_phi, init_psi,
                                init_a, init_b, init_ap, init_bp,
                                init_rm, init_cm))
       
        else:
            initarr = np.array((init_I, init_a, init_b))

    return initarr












