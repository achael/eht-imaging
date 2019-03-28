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

from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import object

import numpy as np
import matplotlib.pyplot as plt
import time

import ehtim.observing.pulses
import ehtim.scattering as so

from ehtim.imaging.imager_utils import *
from ehtim.imaging.pol_imager_utils import *
from ehtim.const_def import *
from ehtim.observing.obs_helpers import *


MAXIT = 200 # number of iterations
NHIST = 50 # number of steps to store for hessian approx
MAXLS = 40 # maximum number of line search steps in BFGS-B
STOP = 1e-6 # convergence criterion
EPS = 1e-8

DATATERMS = ['vis', 'bs', 'amp', 'cphase', 'camp', 'logcamp', 'm']
REGULARIZERS = ['gs', 'tv', 'tv2','l1', 'lA', 'patch', 'flux', 'cm', 'simple', 'compact', 'compact2','rgauss', 'hw', 'ptv']

DATATERMS_POL = ['pvis','m','pbs']
REGULARIZERS_POL = ['msimple', 'hw', 'ptv']

GRIDDER_P_RAD_DEFAULT = 2
GRIDDER_CONV_FUNC_DEFAULT = 'gaussian'
FFT_PAD_DEFAULT = 2
FFT_INTERP_DEFAULT = 3

REG_DEFAULT = {'simple':1}
DAT_DEFAULT = {'vis':100}

POL_PRIM = "amp_phase" # this means we solve for polarization in the m, chi basis
POL_SOLVE = (0,1,1) # this means that pol imaging solves for m & chi (not I), for now

#TODO -- pol_next tracks polarization of next image to make
#     -- checks on the polrep of the prior & init --> exception to switch if necessary
#     -- warnings not to use log transform if imaging V, Q, U, warnings that  RL & LR  have to wait for pol
#     -- get data terms corresponding to  right pols

###########################################################################################################################################
#Imager object
###########################################################################################################################################
class Imager(object):
    """A general interferometric imager.
    """

    def __init__(self, obsdata, init_im, prior_im=None, flux=None, data_term=DAT_DEFAULT, reg_term=REG_DEFAULT, **kwargs):

        self.logstr = ""
        self._obs_list = []
        self._init_list = []
        self._prior_list = []
        self._out_list = []
        self._out_list_epsilon   = []
        self._out_list_scattered = []
        self._flux_list = {}
        self._reg_term_list = []
        self._dat_term_list = []
        self._clipfloor_list = []
        self._pol_list = []
        self._maxit_list = []
        self._stop_list = []
        self._flux_list = []
        self._snrcut_list = []
        self._debias_list = []
        self._systematic_noise_list = []
        self._systematic_cphase_noise_list = []
        self._transform_list = []
        self._weighting_list = []

        # regularizer/data terms for the next imaging iteration
        self.reg_term_next = reg_term  #e.g. [('simple',1), ('l1',10), ('flux',500), ('cm',500)]
        self.dat_term_next = data_term #e.g. [('amp', 1000), ('cphase',100)]

        # obs, prior, flux, init
        self.obs_next = obsdata
        self.init_next = init_im

        if prior_im is None:
            self.prior_next = self.init_next
        else:
            self.prior_next = prior_im

        if flux is None:
            self.flux_next = self.prior_next.total_flux()
        else:
            self.flux_next = flux

        #polarization
        self.pol_next=kwargs.get('pol',self.init_next.pol_prim)

        #weighting/debiasing/snr cut/systematic noise
        self.debias_next=kwargs.get('debias',True)
        snrcut = kwargs.get('snrcut',0.)
        self.snrcut_next = {key: 0. for key in DATATERMS}

        if type(snrcut) is dict:
            for key in snrcut.keys():
                self.snrcut_next[key] = snrcut[key]
        else:
            for key in self.snrcut_next.keys():
                self.snrcut_next[key] = snrcut
        
        self.systematic_noise_next = kwargs.get('systematic_noise',0.)
        self.systematic_cphase_noise_next = kwargs.get('systematic_cphase_noise',0.)
        self.weighting_next = kwargs.get('weighting','natural')

        # clippping
        self.clipfloor_next = kwargs.get('clipfloor',0.)
        self.maxit_next = kwargs.get('maxit',MAXIT)
        self.stop_next = kwargs.get('stop',STOP)
        self.transform_next = kwargs.get('transform','log')

        # normalize or not
        self.norm_init=kwargs.get('norm_init',True)
        self.norm_reg=kwargs.get('norm_reg',False)
        self.beam_size=self.obs_next.res()
        self.regparams = {k:kwargs.get(k, 1.0) for k in ('major', 'minor', 'PA', 'alpha_A')}

        self.chisq_transform = False
        self.chisq_offset_gradient = 0.0

        # FFT parameters
        self._ttype = kwargs.get('ttype','nfft')
        self._fft_gridder_prad = kwargs.get('fft_gridder_prad',GRIDDER_P_RAD_DEFAULT)
        self._fft_conv_func = kwargs.get('fft_conv_func',GRIDDER_CONV_FUNC_DEFAULT)
        self._fft_pad_factor = kwargs.get('fft_pad_factor',FFT_PAD_DEFAULT)
        self._fft_interp_order = kwargs.get('fft_interp_order',FFT_INTERP_DEFAULT)

        # UV minimum for closure phases
        self.cp_uv_min = kwargs.get('cp_uv_min',False)

        # Parameters related to scattering
        self.epsilon_list_next = []
        self.scattering_model = kwargs.get('scattering_model', None)
        self._sqrtQ = None
        self._ea_ker = None
        self._ea_ker_gradient_x = None
        self._ea_ker_gradient_y = None
        self._alpha_phi_list = []
        self.alpha_phi_next = kwargs.get('alpha_phi',1e4)

        # imager history
        self._change_imgr_params = True
        self.nruns = 0

        #set embedding matrices and prepare imager
        self.check_params()
        self.check_limits()
        self.init_imager()

    def make_image(self, pol=None, grads=True, **kwargs):
        """Make an image using current imager settings.
        """

        if pol is None:
            pol_prim = self.pol_next
        else:
            self.pol_next = pol
            pol_prim = pol

        print("==============================")
        print("Imager run %i " % (int(self.nruns)+1))

        # for polarimetric imaging, switch polrep to Stokes
        # TODO: make polarimetric imaging more general
        if self.pol_next=='P':
            print("Imaging P: switching to Stokes!")
            self.prior_next = self.prior_next.switch_polrep(polrep_out='stokes', pol_prim_out='I')
            self.init_next = self.prior_next.switch_polrep(polrep_out='stokes', pol_prim_out='I')
            pol_prim = 'I'

        # Checks and initialize
        self.check_params()
        self.check_limits()
        self.init_imager()

        # Print initial stats
        self._nit = 0
        self._show_updates=kwargs.get('show_updates',True)
        self._update_interval=kwargs.get('update_interval',1)

        # Plot initial image
        self.plotcur(self._xinit, **kwargs)

        # Minimize
        optdict = {'maxiter':self.maxit_next, 'ftol':self.stop_next, 'gtol':self.stop_next,
                   'maxcor':NHIST, 'maxls':MAXLS}
        tstart = time.time()
        if grads:
            res = opt.minimize(self.objfunc, self._xinit, method='L-BFGS-B', jac=self.objgrad,
                               options=optdict, callback=self.plotcur)
        else:
            res = opt.minimize(self.objfunc, self._xinit, method='L-BFGS-B',
                               options=optdict, callback=self.plotcur)
        tstop = time.time()

        # Format output
        out = res.x[:]
        self.tmpout = res.x

        if self.pol_next=='P':
            out = unpack_poltuple(out, self._xtuple, self._nimage, POL_SOLVE)
            if self.transform_next == 'mcv':
                out = mcv(out)

        elif self.transform_next == 'log':
            out = np.exp(out)

        # Print final stats
        outstr = ""
        chi2_term_dict = self.make_chisq_dict(out)
        for dname in sorted(self.dat_term_next.keys()):
            outstr += "chi2_%s : %0.2f " % (dname, chi2_term_dict[dname])

        print("time: %f s" % (tstop - tstart))
        print("J: %f" % res.fun)
        print(outstr)
        print(res.message.decode())
        print("==============================")

        # embed image
        if self.pol_next=='P':
            if np.any(np.invert(self._embed_mask)):
                out = embed_pol(out, self._embed_mask)
            iimage_out = out[0]
            qimage_out = make_q_image(out, POL_PRIM)
            uimage_out = make_u_image(out, POL_PRIM)

        else:
            if np.any(np.invert(self._embed_mask)):
                out = embed(out, self._embed_mask)
            iimage_out = out


        # return image
        outim = image.Image(iimage_out.reshape(self.prior_next.ydim, self.prior_next.xdim), self.prior_next.psize,
                            self.prior_next.ra, self.prior_next.dec, self.prior_next.pa,
                            rf=self.prior_next.rf, source=self.prior_next.source,
                            polrep=self.prior_next.polrep, pol_prim=pol_prim,
                            mjd=self.prior_next.mjd, time=self.prior_next.time,
                            pulse=self.prior_next.pulse)


        # copy over other polarizations
        for pol2 in list(outim._imdict.keys()):

            # Is it the base image?
            if pol2==outim.pol_prim: continue

            # Did we solve for polarimeric image or are we copying over old pols?
            if self.pol_next=='P' and pol2=='Q':
                polvec=qimage_out
            elif self.pol_next=='P' and pol2=='U':
                polvec=uimage_out
            else:
                polvec = self.init_next._imdict[pol2]

            if len(polvec):
                polarr=polvec.reshape(outim.ydim, outim.xdim)
                outim.add_pol_image(polarr, pol2)

        # Append to history
        logstr = str(self.nruns) + ": make_image(pol=%s)"%pol #TODO - what should the log string be?
        self._append_image_history(outim, logstr)
        self.nruns += 1

        # Return Image object
        return outim

    def make_image_I(self, grads=True, **kwargs):
        """Make Stokes I image using current imager settings.
        """
        return self.make_image(pol='I', grads=grads, **kwargs)

    def make_image_P(self, grads=True, **kwargs):
        """Make Stokes P polarimetric image using current imager settings.
        """
        return self.make_image(pol='P', grads=grads, **kwargs)

    def set_embed(self):
        """Set embedding matrix.
        """

        self._embed_mask = self.prior_next.imvec > self.clipfloor_next
        if not np.any(self._embed_mask):
            raise Exception("clipfloor_next too large: all prior pixels have been clipped!")

        coord = np.array([[[x,y] for x in np.arange(self.prior_next.xdim//2,-self.prior_next.xdim//2,-1)]
                                 for y in np.arange(self.prior_next.ydim//2,-self.prior_next.ydim//2,-1)])
        coord = self.prior_next.psize * coord.reshape(self.prior_next.ydim * self.prior_next.xdim, 2)
        self._coord_matrix = coord[self._embed_mask]

        return

    def check_params(self):
        """Check parameter consistency.
        """

        if ((self.prior_next.psize != self.init_next.psize) or
            (self.prior_next.xdim != self.init_next.xdim) or
            (self.prior_next.ydim != self.prior_next.ydim)):
            raise Exception("Initial image does not match dimensions of the prior image!")

        if (self.prior_next.polrep != self.init_next.polrep):
            raise Exception("Initial image pol. representation does not match pol. representation of the prior image!")

        if (self.prior_next.polrep == 'circ' and not(self.pol_next in ['P','RR','LL'])):
            raise Exception("Initial image polrep is 'circ': pol_next must be 'RR' or 'LL' or 'P'!")

        if (self.prior_next.polrep == 'stokes' and not(self.pol_next in ['I','Q','U','V','P'])):
            raise Exception("Initial image polrep is 'stokes': pol_next must be in 'I','Q','U','V','P'!")

        if (self.transform_next=='log' and self.pol_next in ['Q','U','V','P']):
            raise Exception("Cannot image Stokes Q, U, V, P with log image transformation!")

        if self._ttype not in ['fast','direct','nfft']:
            raise Exception("Possible ttype values are 'fast', 'direct','nfft'!")

        if(self.pol_next in ['Q','U','V'] and
          ('gs' in self.reg_term_next.keys() or 'simple' in self.reg_term_next.keys())):
            raise Exception("'simple' and 'gs' MEM methods do not work with potentially negative Stokes Q, U, or V images!")

        # Catch errors for polarimetric imaging setup
        if (self.pol_next=='P'):
            if self.transform_next!='mcv':
                raise Exception("P imaging only works with 'mcv' transform!")
            if (self._ttype not in ["direct","nfft"]):
                raise Exception("FFT no yet implemented in polarimetric imaging -- use NFFT!")

            dt_here = False
            dt_type = True
            for term in sorted(self.dat_term_next.keys()):
                if (term != None) and (term != False): dt_here = True
                if not ((term in DATATERMS_POL) or term==False): dt_type = False

            st_here = False
            st_type = True
            for term in sorted(self.reg_term_next.keys()):
                if (term != None) and (term != False): st_here = True
                if not ((term in REGULARIZERS_POL) or term == False): st_type = False

            if not dt_here:
                raise Exception("Must have at least one data term!")
            if not st_here:
                raise Exception("Must have at least one regularizer term!")
            if not dt_type:
                raise Exception("Invalid data term for P imaging: valid data terms are: " + ','.join(DATATERMS_POL))
            if not st_type:
                raise Exception("Invalid regularizer for P imaging: valid regularizers are: " + ','.join(REGULARIZERS_POL))

        else:
            dt_here = False
            dt_type = True
            for term in sorted(self.dat_term_next.keys()):
                if (term != None) and (term != False): dt_here = True
                if not ((term in DATATERMS) or term==False): dt_type = False

            st_here = False
            st_type = True
            for term in sorted(self.reg_term_next.keys()):
                if (term != None) and (term != False): st_here = True
                if not ((term in REGULARIZERS) or term == False): st_type = False

            if not dt_here:
                raise Exception("Must have at least one data term!")
            if not st_here:
                raise Exception("Must have at least one regularizer term!")
            if not dt_type:
                raise Exception("Invalid data term: valid data terms are: " + ','.join(DATATERMS))
            if not st_type:
                raise Exception("Invalid regularizer: valid regularizers are: " + ','.join(REGULARIZERS))

        # determine if we need to recompute the saved imager parameters on the next imager run
        if self.nruns == 0:
            return

        if self.pol_next != self.pol_last():
            print("changed polarization!")
            self._change_imgr_params = True
            return

        if self.obs_next != self.obs_last():
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


    def check_limits(self):
        """Check image parameter consistency with observation.
        """

        uvmax = 1.0/self.prior_next.psize
        uvmin = 1.0/(self.prior_next.psize*np.max((self.prior_next.xdim, self.prior_next.ydim)))
        uvdists = self.obs_next.unpack('uvdist')['uvdist']
        maxbl = np.max(uvdists)
        minbl = np.max(uvdists[uvdists > 0])


        if uvmax < maxbl:
            print("Warning! Pixel size is than smallest spatial wavelength!")
        if uvmin > minbl:
            print("Warning! Field of View is smaller than largest nonzero spatial wavelength!")

        if self.pol_next in ['I','RR','LL']:
            maxamp = np.max(np.abs(self.obs_next.unpack('amp')['amp']))
            if self.flux_next > 1.2*maxamp:
                print("Warning! Specified flux is > 120% of maximum visibility amplitude!")
            if self.flux_next < .8*maxamp:
                print("Warning! Specified flux is < 80% of maximum visibility amplitude!")

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

    def obs_last(self):
        """Return last used observation.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._obs_list[-1]

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

    def out_scattered_last(self):
        """Return last result with scattering.
        """
        if self.nruns == 0 or len(self._out_list_scattered) == 0:
            print("No stochastic optics imager runs yet!")
            return
        return self._out_list_scattered[-1]

    def out_epsilon_last(self):
        """Return last result with scattering.
        """
        if self.nruns == 0 or len(self._out_list_epsilon) == 0:
            print("No stochastic optics imager runs yet!")
            return
        return self._out_list_epsilon[-1]

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

    def stop_last(self):
        """Return last convergence value.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._stop_list[-1]

    def transform_last(self):
        """Return last image transfrom used.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._transform_list[-1]

    def init_imager(self):
        """Set up Stokes I imager.
        """

        # set embedding
        self.set_embed()

        # set prior & initial image vectors
        if self.pol_next=='P': # polarimetric imaging
            # initial I, kept constant
            self._iinit = self.init_next.imvec[self._embed_mask]
            self._nimage = len(self._iinit)

            # initialize m & phi
            if len(self.init_next.qvec) and (np.any(self.init_next.qvec!=0) or np.any(self.init_next.uvec!=0)):
                init1 = (np.abs(self.init_next.qvec + 1j*self.init_next.uvec) / self.init_next.imvec)[self._embed_mask]
                init2 = (np.arctan2(self.init_next.uvec, self.init_next.qvec) / 2.0)[self._embed_mask]
            else:
                # !AC TODO get the actual zero baseline pol. frac from the data!??
                print("No polarimetric image in init_next -- initializing with 10% pol and random orientation!")
                init1 = 0.2 * (np.ones(self._nimage) + 1e-2 * np.random.rand(self._nimage))
                init2 = np.zeros(self._nimage) + 1e-2 * np.random.rand(self._nimage)
            self._inittuple = np.array((self._iinit, init1, init2))

            # change of variables
            if self.transform_next == 'mcv':
                self._xtuple =  mcv_r(self._inittuple)
            else:
                raise Exception("Polarimetric imaging only works with transform_next='mcv'")

            # pack into single vector
            self._xinit = pack_poltuple(self._xtuple, POL_SOLVE)

        else: # single stokes or RR/LL imaging
            if self.norm_init:
                self._nprior = (self.flux_next * self.prior_next.imvec / np.sum((self.prior_next.imvec)[self._embed_mask]))[self._embed_mask]
                self._ninit = (self.flux_next * self.init_next.imvec / np.sum((self.init_next.imvec)[self._embed_mask]))[self._embed_mask]
            else:
                self._nprior = self.prior_next.imvec[self._embed_mask]
                self._ninit = self.init_next.imvec[self._embed_mask]

            # change of variables
            if self.transform_next == 'log':
                self._xinit = np.log(self._ninit)
            else: self._xinit = self._ninit

        # data term tuples
        if self._change_imgr_params:
            if self.nruns==0:
                print("Initializing imager data products . . .")
            if self.nruns>0:
                print("Recomputing imager data products . . .")
            self._data_tuples = {}
            for dname in sorted(self.dat_term_next.keys()):
                if self.pol_next=='P':
                    tup = polchisqdata(self.obs_next, self.prior_next, self._embed_mask, dname, pol=self.pol_next,
                                        debias=self.debias_next, snrcut=self.snrcut_next[dname], weighting=self.weighting_next,
                                        systematic_noise=self.systematic_noise_next, systematic_cphase_noise=self.systematic_cphase_noise_next,
                                        ttype=self._ttype, order=self._fft_interp_order, fft_pad_factor=self._fft_pad_factor,
                                        conv_func=self._fft_conv_func, p_rad=self._fft_gridder_prad, cp_uv_min=self.cp_uv_min)
                else:
                    tup = chisqdata(self.obs_next, self.prior_next, self._embed_mask, dname, pol=self.pol_next,
                                    debias=self.debias_next, snrcut=self.snrcut_next[dname], weighting=self.weighting_next,
                                    systematic_noise=self.systematic_noise_next, systematic_cphase_noise=self.systematic_cphase_noise_next,
                                    ttype=self._ttype, order=self._fft_interp_order, fft_pad_factor=self._fft_pad_factor,
                                    conv_func=self._fft_conv_func, p_rad=self._fft_gridder_prad, cp_uv_min=self.cp_uv_min)

                self._data_tuples[dname] = tup
            self._change_imgr_params = False

        return

    def init_imager_scattering(self):
        """Set up scattering imager.
        """
        N = self.prior_next.xdim

        if self.scattering_model == None:
            self.scattering_model = so.ScatteringModel()

        # First some preliminary definitions
        wavelength = C/self.obs_next.rf*100.0 #Observing wavelength [cm]
        wavelengthbar = wavelength/(2.0*np.pi) #lambda/(2pi) [cm]
        N = self.prior_next.xdim
        FOV = self.prior_next.psize * N * self.scattering_model.observer_screen_distance #Field of view, in cm, at the scattering screen

        # The ensemble-average convolution kernel and its gradients
        self._ea_ker = self.scattering_model.Ensemble_Average_Kernel(self.prior_next, wavelength_cm = wavelength)
        ea_ker_gradient = so.Wrapped_Gradient(self._ea_ker/(FOV/N))
        self._ea_ker_gradient_x = -ea_ker_gradient[1]
        self._ea_ker_gradient_y = -ea_ker_gradient[0]

        # The power spectrum (note: rotation is not currently implemented; the gradients would need to be modified slightly)
        self._sqrtQ = np.real(self.scattering_model.sqrtQ_Matrix(self.prior_next,t_hr=0.0))

        # Generate the initial image+screen vector. By default, the screen is re-initialized to zero each time.
        if len(self.epsilon_list_next) == 0:
            self._xinit = np.concatenate((self._xinit,np.zeros(N**2-1)))
        else:
            self._xinit = np.concatenate((self._xinit,self.epsilon_list_next))

    def make_chisq_dict(self, imcur):
        """make dictionary of current chi^2 term values
        """
        chi2_dict = {}
        for dname in sorted(self.dat_term_next.keys()):
            data = self._data_tuples[dname][0]
            sigma = self._data_tuples[dname][1]
            A = self._data_tuples[dname][2]

            if self.pol_next=='P':
                chi2 = polchisq(imcur, A, data, sigma, dname,
                                ttype=self._ttype, mask=self._embed_mask,
                                pol_prim=POL_PRIM)
            else:
                chi2 = chisq(imcur, A, data, sigma, dname,
                             ttype=self._ttype, mask=self._embed_mask)
            chi2_dict[dname] = chi2

        return chi2_dict

    def make_chisqgrad_dict(self, imcur):
        """make dictionary of current chi^2 term gradient values
        """
        chi2grad_dict = {}
        for dname in sorted(self.dat_term_next.keys()):
            data = self._data_tuples[dname][0]
            sigma = self._data_tuples[dname][1]
            A = self._data_tuples[dname][2]

            if self.pol_next=='P':
                chi2grad = polchisqgrad(imcur, A, data, sigma, dname,
                                        ttype=self._ttype, mask=self._embed_mask,
                                        pol_prim=POL_PRIM, pol_solve=POL_SOLVE)
            else:
                chi2grad = chisqgrad(imcur, A, data, sigma, dname,
                                     ttype=self._ttype, mask=self._embed_mask)

            chi2grad_dict[dname] = chi2grad

        return chi2grad_dict

    def make_reg_dict(self, imcur):
        """make dictionary of current regularizer values
        """
        reg_dict = {}
        for regname in sorted(self.reg_term_next.keys()):
            if self.pol_next=='P':
                reg = polregularizer(imcur, self._embed_mask,
                                     self.prior_next.xdim, self.prior_next.ydim,
                                     self.prior_next.psize, regname,
                                     pol_prim=POL_PRIM
                                     )
            else:
                reg = regularizer(imcur, self._nprior, self._embed_mask,
                                  self.flux_next, self.prior_next.xdim,
                                  self.prior_next.ydim, self.prior_next.psize,
                                  regname,
                                  norm_reg=self.norm_reg, beam_size=self.beam_size,
                                  **self.regparams)
            reg_dict[regname] = reg

        return reg_dict

    def make_reggrad_dict(self, imcur):
        """make dictionary of current regularizer gradient values
        """
        reggrad_dict = {}
        for regname in sorted(self.reg_term_next.keys()):

            if self.pol_next=='P':
                reg = polregularizergrad(imcur, self._embed_mask,
                                     self.prior_next.xdim, self.prior_next.ydim,
                                     self.prior_next.psize, regname,
                                     pol_prim=POL_PRIM, pol_solve=POL_SOLVE
                                     )
            else:
                reg = regularizergrad(imcur, self._nprior, self._embed_mask,
                                      self.flux_next, self.prior_next.xdim,
                                      self.prior_next.ydim, self.prior_next.psize,
                                      regname,
                                      norm_reg=self.norm_reg, beam_size=self.beam_size,
                                      **self.regparams)
            reggrad_dict[regname] = reg

        return reggrad_dict

    def objfunc(self, imvec):
        """Current objective function.
        """
        # unpack polarimetric vector into an array
        if self.pol_next=='P':
            imcur = unpack_poltuple(imvec, self._xtuple, self._nimage, POL_SOLVE)
        else:
            imcur = imvec

        # image change of variables
        if self.transform_next == 'log':
            imcur = np.exp(imcur)
        elif self.transform_next== 'mcv':
            imcur = mcv(imcur)

        # data terms
        datterm = 0.
        chi2_term_dict = self.make_chisq_dict(imcur)
        for dname in sorted(self.dat_term_next.keys()):
            if self.chisq_transform:
                datterm += self.dat_term_next[dname] * (chi2_term_dict[dname] + 1./chi2_term_dict[dname] - 1.)
            else:
                datterm += self.dat_term_next[dname] * (chi2_term_dict[dname] - 1.)

        # regularizer terms
        regterm = 0
        reg_term_dict = self.make_reg_dict(imcur)
        for regname in sorted(self.reg_term_next.keys()):
            regterm += self.reg_term_next[regname] * reg_term_dict[regname]

        return datterm + regterm


    def objgrad(self, imvec):
        """Current objective function gradient.
        """
        # unpack polarimetric vector into an array
        if self.pol_next=='P':
            imcur = unpack_poltuple(imvec, self._xtuple, self._nimage, POL_SOLVE)
        else:
            imcur = imvec

        # image change of variabbles
        if self.transform_next == 'log':
            imcur = np.exp(imcur)
        elif self.transform_next== 'mcv':
            cvcur = imcur.copy()
            imcur = mcv(imcur)

        datterm = 0.
        chi2_term_dict = self.make_chisqgrad_dict(imcur)
        if self.chisq_transform:
            chi2_value_dict =  self.make_chisq_dict(imcur)

        for dname in sorted(self.dat_term_next.keys()):

            if self.chisq_transform:
                datterm += self.dat_term_next[dname] * chi2_term_dict[dname] * (1 - 1./(chi2_value_dict[dname]**2))
            else:
                datterm += self.dat_term_next[dname] * (chi2_term_dict[dname] + self.chisq_offset_gradient)

        regterm = 0
        reg_term_dict = self.make_reggrad_dict(imcur)
        for regname in sorted(self.reg_term_next.keys()):
            regterm += self.reg_term_next[regname] * reg_term_dict[regname]

        grad = datterm + regterm

        # chain rule term for change of variables
        if self.transform_next == 'log':
            grad *= imcur
        elif self.transform_next == 'mcv':
            grad *= mchain(cvcur)

        # repack gradient for polarimetric imaging
        if self.pol_next=='P':
            grad = pack_poltuple(grad, POL_SOLVE)

        return grad

    def plotcur(self, imvec, **kwargs):
        if self._show_updates:
            if self._nit % self._update_interval == 0:
                if self.pol_next=='P':
                    imcur = unpack_poltuple(imvec, self._xtuple, self._nimage, POL_SOLVE)
                else:
                    imcur = imvec

                #change of variables
                if self.transform_next == 'log':
                    imcur = np.exp(imcur)
                elif self.transform_next == "mcv":
                    imcur = mcv(imcur)

                # get chi^2 and regularizer
                chi2_term_dict = self.make_chisq_dict(imcur)
                reg_term_dict = self.make_reg_dict(imcur)

                chi2_keys = sorted(chi2_term_dict.keys())
                chi2_1 = chi2_term_dict[chi2_keys[0]]
                chi2_2 = 0.
                if len(chi2_term_dict) > 1:
                    chi2_2 = chi2_term_dict[chi2_keys[1]]

                # format print string
                outstr = "------------------------------------------------------------------"
                outstr += "\n%4d | " % self._nit
                for dname in sorted(self.dat_term_next.keys()):
                    outstr += "chi2_%s : %0.2f " % (dname, chi2_term_dict[dname])

                outstr += "\n        "
                for dname in sorted(self.dat_term_next.keys()):
                    outstr += "%s : %0.1f " % (dname, chi2_term_dict[dname]*self.dat_term_next[dname])
                outstr += "\n        "
                for regname in sorted(self.reg_term_next.keys()):
                    outstr += "%s : %0.1f " % (regname, reg_term_dict[regname]*self.reg_term_next[regname])

                # embed and plot
                if self.pol_next=='P':
                    if np.any(np.invert(self._embed_mask)):
                        imcur = embed_pol(imcur, self._embed_mask)
                    plot_m(imcur, self.prior_next, self._nit, chi2_term_dict, **kwargs)

                else:
                    if np.any(np.invert(self._embed_mask)):
                        imcur = embed(imcur, self._embed_mask)

                    plot_i(imcur, self.prior_next, self._nit, chi2_term_dict, pol=self.pol_next, **kwargs)

                if self._nit == 0: print()
                print(outstr)
        self._nit += 1


    def objfunc_scattering(self, minvec):
        """Current stochastic optics objective function.
        """
        N = self.prior_next.xdim

        imvec       = minvec[:N**2]
        EpsilonList = minvec[N**2:]
        if self.transform_next == 'log':
            imvec = np.exp(imvec)

        IM = ehtim.image.Image(imvec.reshape(N,N), self.prior_next.psize,
                               self.prior_next.ra, self.prior_next.dec, self.prior_next.pa,
                               rf=self.obs_next.rf, source=self.prior_next.source, mjd=self.prior_next.mjd)

        #the scattered image vector
        scatt_im = self.scattering_model.Scatter(IM, Epsilon_Screen=so.MakeEpsilonScreenFromList(EpsilonList, N),
                                                 ea_ker = self._ea_ker, sqrtQ=self._sqrtQ,
                                                 Linearized_Approximation=True).imvec

        # Calculate the chi^2 using the scattered image
        datterm = 0.
        chi2_term_dict = self.make_chisq_dict(scatt_im)
        for dname in sorted(self.dat_term_next.keys()):
            datterm += self.dat_term_next[dname] * (chi2_term_dict[dname] - 1.)

        # Calculate the entropy using the unscattered image
        regterm = 0
        reg_term_dict = self.make_reg_dict(imvec)
        #make dict also for scattered image
        reg_term_dict_scatt = self.make_reg_dict(scatt_im)

        for regname in sorted(self.reg_term_next.keys()):
            if regname == 'rgauss':
                #get gradient of the scattered image vector
                regterm += self.reg_term_next[regname] * reg_term_dict_scatt[regname]

            else:
                regterm += self.reg_term_next[regname] * reg_term_dict[regname]

        # Scattering screen regularization term
        chisq_epsilon = sum(EpsilonList*EpsilonList)/((N*N-1.0)/2.0)
        regterm_scattering = self.alpha_phi_next * (chisq_epsilon - 1.0)

        return datterm + regterm + regterm_scattering

    def objgrad_scattering(self, minvec):
        """Current stochastic optics objective function gradient
        """
        wavelength = C/self.obs_next.rf*100.0 #Observing wavelength [cm]
        wavelengthbar = wavelength/(2.0*np.pi) #lambda/(2pi) [cm]
        N = self.prior_next.xdim
        #Field of view, in cm, at the scattering screen
        FOV = self.prior_next.psize * N * self.scattering_model.observer_screen_distance
        rF = self.scattering_model.rF(wavelength)

        imvec       = minvec[:N**2]
        EpsilonList = minvec[N**2:]
        if self.transform_next == 'log':
            imvec = np.exp(imvec)

        IM = ehtim.image.Image(imvec.reshape(N,N), self.prior_next.psize,
                               self.prior_next.ra, self.prior_next.dec, self.prior_next.pa,
                               rf=self.obs_next.rf, source=self.prior_next.source, mjd=self.prior_next.mjd)
        #the scattered image vector
        scatt_im = self.scattering_model.Scatter(IM, Epsilon_Screen=so.MakeEpsilonScreenFromList(EpsilonList, N),
                                                 ea_ker = self._ea_ker, sqrtQ=self._sqrtQ,
                                                 Linearized_Approximation=True).imvec

        EA_Image = self.scattering_model.Ensemble_Average_Blur(IM, ker = self._ea_ker)
        EA_Gradient = so.Wrapped_Gradient((EA_Image.imvec/(FOV/N)).reshape(N, N))
        #The gradient signs don't actually matter, but let's make them match intuition (i.e., right to left, bottom to top)
        EA_Gradient_x = -EA_Gradient[1]
        EA_Gradient_y = -EA_Gradient[0]

        Epsilon_Screen = so.MakeEpsilonScreenFromList(EpsilonList, N)
        phi = self.scattering_model.MakePhaseScreen(Epsilon_Screen, IM, obs_frequency_Hz=self.obs_next.rf,sqrtQ_init=self._sqrtQ).imvec.reshape((N, N))
        phi_Gradient = so.Wrapped_Gradient(phi/(FOV/N))
        phi_Gradient_x = -phi_Gradient[1]
        phi_Gradient_y = -phi_Gradient[0]

        #Entropy gradient; wrt unscattered image so unchanged by scattering
        regterm = 0
        reg_term_dict = self.make_reggrad_dict(imvec)
        #make dict also for scattered image
        reg_term_dict_scatt = self.make_reggrad_dict(scatt_im)

        for regname in sorted(self.reg_term_next.keys()):
            #we need an exception if the regularizer is 'rgauss'
            if regname == 'rgauss':
                #get gradient of the scattered image vector
                gaussterm = self.reg_term_next[regname] * reg_term_dict_scatt[regname]
                dgauss_dIa = gaussterm.reshape((N,N))

                # Now the chain rule factor to get the gauss gradient wrt the unscattered image
                gx = (rF**2.0 * so.Wrapped_Convolve(self._ea_ker_gradient_x[::-1,::-1], phi_Gradient_x * (dgauss_dIa))).flatten()
                gy = (rF**2.0 * so.Wrapped_Convolve(self._ea_ker_gradient_y[::-1,::-1], phi_Gradient_y * (dgauss_dIa))).flatten()

                # Now we add the gradient for the unscattered image
                regterm += so.Wrapped_Convolve(self._ea_ker[::-1,::-1], (dgauss_dIa)).flatten() + gx + gy

            else:
                regterm += self.reg_term_next[regname] * reg_term_dict[regname]

        # Chi^2 gradient wrt the unscattered image
        # First, the chi^2 gradient wrt to the scattered image
        datterm = 0.
        chi2_term_dict = self.make_chisqgrad_dict(scatt_im)
        for dname in sorted(self.dat_term_next.keys()):
            datterm += self.dat_term_next[dname] * (chi2_term_dict[dname])
        dchisq_dIa = datterm.reshape((N,N))
        # Now the chain rule factor to get the chi^2 gradient wrt the unscattered image
        gx = (rF**2.0 * so.Wrapped_Convolve(self._ea_ker_gradient_x[::-1,::-1], phi_Gradient_x * (dchisq_dIa))).flatten()
        gy = (rF**2.0 * so.Wrapped_Convolve(self._ea_ker_gradient_y[::-1,::-1], phi_Gradient_y * (dchisq_dIa))).flatten()
        chisq_grad_im = so.Wrapped_Convolve(self._ea_ker[::-1,::-1], (dchisq_dIa)).flatten() + gx + gy

        # Gradient of the data chi^2 wrt to the epsilon screen
        #Preliminary Definitions
        chisq_grad_epsilon = np.zeros(N**2-1)
        i_grad = 0
        ell_mat = np.zeros((N,N))
        m_mat   = np.zeros((N,N))
        for ell in range(0, N):
            for m in range(0, N):
                ell_mat[ell,m] = ell
                m_mat[ell,m] = m

        #Real part; top row
        for t in range(1, (N+1)//2):
            s=0
            grad_term = so.Wrapped_Gradient(wavelengthbar/FOV*self._sqrtQ[s][t]*2.0*np.cos(2.0*np.pi/N*(ell_mat*s + m_mat*t))/(FOV/N))
            grad_term_x = -grad_term[1]
            grad_term_y = -grad_term[0]
            chisq_grad_epsilon[i_grad] = np.sum( dchisq_dIa * rF**2 * ( EA_Gradient_x * grad_term_x + EA_Gradient_y * grad_term_y ) )
            i_grad = i_grad + 1

        #Real part; remainder
        for s in range(1,(N+1)//2):
            for t in range(N):
                grad_term = so.Wrapped_Gradient(wavelengthbar/FOV*self._sqrtQ[s][t]*2.0*np.cos(2.0*np.pi/N*(ell_mat*s + m_mat*t))/(FOV/N))
                grad_term_x = -grad_term[1]
                grad_term_y = -grad_term[0]
                chisq_grad_epsilon[i_grad] = np.sum( dchisq_dIa * rF**2 * ( EA_Gradient_x * grad_term_x + EA_Gradient_y * grad_term_y ) )
                i_grad = i_grad + 1

        #Imaginary part; top row
        for t in range(1, (N+1)//2):
            s=0
            grad_term = so.Wrapped_Gradient(-wavelengthbar/FOV*self._sqrtQ[s][t]*2.0*np.sin(2.0*np.pi/N*(ell_mat*s + m_mat*t))/(FOV/N))
            grad_term_x = -grad_term[1]
            grad_term_y = -grad_term[0]
            chisq_grad_epsilon[i_grad] = np.sum( dchisq_dIa * rF**2 * ( EA_Gradient_x * grad_term_x + EA_Gradient_y * grad_term_y ) )
            i_grad = i_grad + 1

        #Imaginary part; remainder
        for s in range(1,(N+1)//2):
            for t in range(N):
                grad_term = so.Wrapped_Gradient(-wavelengthbar/FOV*self._sqrtQ[s][t]*2.0*np.sin(2.0*np.pi/N*(ell_mat*s + m_mat*t))/(FOV/N))
                grad_term_x = -grad_term[1]
                grad_term_y = -grad_term[0]
                chisq_grad_epsilon[i_grad] = np.sum( dchisq_dIa * rF**2 * ( EA_Gradient_x * grad_term_x + EA_Gradient_y * grad_term_y ) )
                i_grad = i_grad + 1

        # Gradient of the chi^2 regularization term for the epsilon screen
        chisq_epsilon_grad = self.alpha_phi_next * 2.0*EpsilonList/((N*N-1)/2.0)

        # chain rule term for change of variables
        if self.transform_next == 'log':
            regterm       *= imvec
            chisq_grad_im *= imvec

        return np.concatenate(((regterm + chisq_grad_im),(chisq_grad_epsilon + chisq_epsilon_grad)))

    def plotcur_scattering(self, minvec):
        if self._show_updates:
            if self._nit % self._update_interval == 0:
                N = self.prior_next.xdim

                imvec       = minvec[:N**2]
                EpsilonList = minvec[N**2:]
                if self.transform_next == 'log':
                    imvec = np.exp(imvec)

                IM = ehtim.image.Image(imvec.reshape(N,N), self.prior_next.psize,
                                       self.prior_next.ra, self.prior_next.dec, self.prior_next.pa,
                                       rf=self.obs_next.rf, source=self.prior_next.source, mjd=self.prior_next.mjd)
                #the scattered image vector
                scatt_im = self.scattering_model.Scatter(IM, Epsilon_Screen=so.MakeEpsilonScreenFromList(EpsilonList, N),
                                                         ea_ker = self._ea_ker, sqrtQ=self._sqrtQ, Linearized_Approximation=True).imvec

                # Calculate the chi^2 using the scattered image
                datterm = 0.
                chi2_term_dict = self.make_chisq_dict(scatt_im)
                for dname in sorted(self.dat_term_next.keys()):
                    datterm += self.dat_term_next[dname] * (chi2_term_dict[dname] - 1.)

                # Calculate the entropy using the unscattered image
                regterm = 0
                reg_term_dict = self.make_reg_dict(imvec)
                for regname in sorted(self.reg_term_next.keys()):
                    regterm += self.reg_term_next[regname] * reg_term_dict[regname]

                # Scattering screen regularization term
                chisq_epsilon = sum(EpsilonList*EpsilonList)/((N*N-1.0)/2.0)
                regterm_scattering = self.alpha_phi_next * (chisq_epsilon - 1.0)

                outstr = "i: %d " % self._nit

                for dname in sorted(self.dat_term_next.keys()):
                    outstr += "%s : %0.2f " % (dname, chi2_term_dict[dname])
                for regname in sorted(self.reg_term_next.keys()):
                    outstr += "%s : %0.2f " % (regname, reg_term_dict[regname])
                outstr += "Epsilon chi^2 : %0.2f " % (chisq_epsilon)
                outstr += "Max |Epsilon| : %0.2f " % (max(abs(EpsilonList)))
                print(outstr)

        self._nit += 1

    def make_image_I_stochastic_optics(self, grads=True, **kwargs):
        """Reconstructs an image of total flux density using the stochastic optics scattering mitigation technique.
           Uses the scattering model of the imager. If none has been specified, it will default to a standard model for Sgr A*.
           Returns the estimated unscattered image.

           Args:
                grads (bool): Flag for whether or not to use analytic gradients.
                show_updates (bool): Flag for whether or not to show updates for each step of convergence.
           Returns:
               out (Image): The estimated *unscattered* image.
        """

        N = self.prior_next.xdim

        # Checks and initialize
        self.check_params()
        self.check_limits()
        self.init_imager()
        self.init_imager_scattering()
        self._nit = 0

        # Print stats
        self._show_updates=kwargs.get('show_updates',True)
        self._update_interval=kwargs.get('update_interval',1)
        self.plotcur_scattering(self._xinit)

        # Minimize
        optdict = {'maxiter':self.maxit_next, 'ftol':self.stop_next, 'maxcor':NHIST}
        tstart = time.time()
        if grads:
            res = opt.minimize(self.objfunc_scattering, self._xinit, method='L-BFGS-B', jac=self.objgrad_scattering,
                               options=optdict, callback=self.plotcur_scattering)
        else:
            res = opt.minimize(self.objfunc_scattering, self._xinit, method='L-BFGS-B',
                               options=optdict, callback=self.plotcur_scattering)
        tstop = time.time()

        # Format output
        out = res.x[:N**2]
        if self.transform_next == 'log': out = np.exp(out)
        if np.any(np.invert(self._embed_mask)):
            raise Exception("Embedding is not currently implemented!")
            out = embed(out, self._embed_mask)

        outim = image.Image(out.reshape(N, N), self.prior_next.psize,
                            self.prior_next.ra, self.prior_next.dec, self.prior_next.pa,
                            rf=self.prior_next.rf, source=self.prior_next.source, mjd=self.prior_next.mjd, pulse=self.prior_next.pulse)
        outep = res.x[N**2:]
        outscatt = self.scattering_model.Scatter(outim, Epsilon_Screen=so.MakeEpsilonScreenFromList(outep, N),
                                                 ea_ker = self._ea_ker, sqrtQ=self._sqrtQ,
                                                 Linearized_Approximation=True)

        # Preserving image complex polarization fractions
        if len(self.prior_next.qvec):
            qvec = self.prior_next.qvec * out / self.prior_next.imvec
            uvec = self.prior_next.uvec * out / self.prior_next.imvec
            outim.add_qu(qvec.reshape(N, N),
                         uvec.reshape(N, N))

        # Print stats
        print("time: %f s" % (tstop - tstart))
        print("J: %f" % res.fun)
        print(res.message)

        # Append to history
        logstr = str(self.nruns) + ": make_image_I_stochastic_optics()"
        self._append_image_history(outim, logstr)
        self._out_list_epsilon.append(res.x[N**2:])
        self._out_list_scattered.append(outscatt)

        self.nruns += 1

        # Return Image object
        return outim

    def _append_image_history(self, outim, logstr):
        self.logstr += (logstr + "\n")
        self._obs_list.append(self.obs_next)
        self._init_list.append(self.init_next)
        self._prior_list.append(self.prior_next)
        self._debias_list.append(self.debias_next)
        self._weighting_list.append(self.weighting_next)
        self._systematic_noise_list.append(self.systematic_noise_next)
        self._systematic_cphase_noise_list.append(self.systematic_cphase_noise_next)
        self._snrcut_list.append(self.snrcut_next)
        self._flux_list.append(self.flux_next)
        self._pol_list.append(self.pol_next)
        self._clipfloor_list.append(self.clipfloor_next)
        self._maxit_list.append(self.maxit_next)
        self._stop_list.append(self.stop_next)
        self._transform_list.append(self.transform_next)
        self._reg_term_list.append(self.reg_term_next)
        self._dat_term_list.append(self.dat_term_next)
        self._alpha_phi_list.append(self.alpha_phi_next)

        self._out_list.append(outim)
        return

    def make_image_scat(self):
        return
