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
from builtins import range
from builtins import object

import copy
import time
import numpy as np
import scipy.optimize as opt

import ehtim.scattering as so
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
REGULARIZERS = ['gs', 'tv', 'tv2', 'l1', 'l1w', 'lA', 'patch',
                'flux', 'cm', 'simple', 'compact', 'compact2', 'rgauss', 'hw']
REGULARIZERS_SPECIND = ['l2_alpha', 'tv_alpha']
REGULARIZERS_CURV = ['l2_beta', 'tv_beta']

DATATERMS_POL = ['pvis', 'm', 'pbs']
REGULARIZERS_POL = ['msimple', 'hw', 'ptv']

GRIDDER_P_RAD_DEFAULT = 2
GRIDDER_CONV_FUNC_DEFAULT = 'gaussian'
FFT_PAD_DEFAULT = 2
FFT_INTERP_DEFAULT = 3

REG_DEFAULT = {'simple': 1}
DAT_DEFAULT = {'vis': 100}

POL_PRIM_SOLVE = "amp_phase"  # this means we solve for polarization in the m, chi basis
POL_WHICH_SOLVE = (0, 1, 1)   # this means that pol imaging solves for m & chi (not I), for now
MF_WHICH_SOLVE = (1, 1, 0)    # this means that mf imaging solves for I0 and alpha (not beta), for now
                              # DEFAULT ONLY: object now uses self.mf_which_solve

REGPARAMS_DEFAULT = {'major':50*ehc.RADPERUAS,
                     'minor':50*ehc.RADPERUAS,
                     'PA':0.,
                     'alpha_A':1.0,
                     'epsilon_tv':0.0}

###################################################################################################
# Imager object
###################################################################################################


class Imager(object):
    """A general interferometric imager.
    """

    def __init__(self, obs_in, init_im,
                 prior_im=None, flux=None, data_term=DAT_DEFAULT, reg_term=REG_DEFAULT, **kwargs):

        self.logstr = ""
        self._obs_list = []
        self._init_list = []
        self._prior_list = []
        self._out_list = []
        self._out_list_epsilon = []
        self._out_list_scattered = []
        self._flux_list = {}
        self._reg_term_list = []
        self._dat_term_list = []
        self._clipfloor_list = []
        self._maxset_list = []
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

        # Regularizer/data terms for the next imaging iteration
        self.reg_term_next = reg_term  # e.g. [('simple',1), ('l1',10), ('flux',500), ('cm',500)]
        self.dat_term_next = data_term  # e.g. [('amp', 1000), ('cphase',100)]

        # Observations, frequencies
        self.reffreq = init_im.rf
        if isinstance(obs_in, list):
            self._obslist_next = obs_in
            self.obslist_next = obs_in
        else:
            self._obslist_next = [obs_in]
            self.obslist_next = [obs_in]

        # Init, prior, flux
        self.init_next = init_im

        if prior_im is None:
            self.prior_next = self.init_next
        else:
            self.prior_next = prior_im

        if flux is None:
            self.flux_next = self.prior_next.total_flux()
        else:
            self.flux_next = flux

        # Polarization
        self.pol_next = kwargs.get('pol', self.init_next.pol_prim)

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

        # Clippping
        self.clipfloor_next = kwargs.get('clipfloor', 0.)
        self.maxit_next = kwargs.get('maxit', MAXIT)
        self.stop_next = kwargs.get('stop', STOP)
        self.transform_next = kwargs.get('transform', ['log','mcv'])
        self.transform_next = np.array([self.transform_next]).flatten() #so we can handle multiple transforms

        # Normalize or not?
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

        # UV minimum for closure phases
        self.cp_uv_min = kwargs.get('cp_uv_min', False)

        # Parameters related to scattering
        self.epsilon_list_next = []
        self.scattering_model = kwargs.get('scattering_model', None)
        self._sqrtQ = None
        self._ea_ker = None
        self._ea_ker_gradient_x = None
        self._ea_ker_gradient_y = None
        self._alpha_phi_list = []
        self.alpha_phi_next = kwargs.get('alpha_phi', 1e4)

        # Imager history
        self._change_imgr_params = True
        self.nruns = 0

        # multifrequency
        self.mf_next = False
        self.mf_which_solve = kwargs.get('mf_which_solve',MF_WHICH_SOLVE)

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
        #self.reffreq = self.freq_list[0] #Changed so that reffreq is determined by initial image/prior rf
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

    def make_image(self, pol=None, grads=True, mf=False, **kwargs):
        """Make an image using current imager settings.

           Args:
               pol (str): which polarization to image
               grads (bool): whether or not to use image gradients
               mf (bool): whether or not to do multifrequency (spectral index only for now)

           Returns:
               (Image): output image

        """

        self.mf_next = mf
        self.mf_which_solve = kwargs.get('mf_which_solve', self.mf_which_solve)

        if pol is None:
            pol_prim = self.pol_next
        else:
            self.pol_next = pol
            pol_prim = pol

        print("==============================")
        print("Imager run %i " % (int(self.nruns)+1))

        # For polarimetric imaging, switch polrep to Stokes
        if self.pol_next == 'P' or self.pol_next == 'IP' or self.pol_next == 'IQU':
            print("Imaging P: switching to Stokes!")
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

        if self.pol_next == 'P':
            out = polutils.unpack_poltuple(out, self._xtuple, self._nimage, (0,1,1))
            if 'mcv' in self.transform_next:
                out = polutils.mcv(out)

        elif self.pol_next == 'IP' or self.pol_next == 'IQU':
            out = polutils.unpack_poltuple(out, self._xtuple, self._nimage, (1,1,1))
            if 'mcv' in self.transform_next:
                out = polutils.mcv(out)
            if 'log' in self.transform_next:
                out[0] = np.exp(out[0])

        elif self.mf_next:
            out = mfutils.unpack_mftuple(out, self._xtuple, self._nimage, self.mf_which_solve)
            if 'log' in self.transform_next:
                out[0] = np.exp(out[0])

        elif 'log' in self.transform_next:
            out = np.exp(out)

        # Print final stats
        outstr = ""
        chi2_term_dict = self.make_chisq_dict(out)
        for dname in sorted(self.dat_term_next.keys()):
            for i, obs in enumerate(self.obslist_next):
                if i == 0:
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

        # Embed image
        if self.pol_next == 'P' or self.pol_next == 'IP' or self.pol_next == 'IQU':
            if np.any(np.invert(self._embed_mask)):
                out = polutils.embed_pol(out, self._embed_mask)
            iimage_out = out[0]
            qimage_out = polutils.make_q_image(out, POL_PRIM_SOLVE)
            uimage_out = polutils.make_u_image(out, POL_PRIM_SOLVE)

        elif self.mf_next:
            if np.any(np.invert(self._embed_mask)):
                out = mfutils.embed_mf(out, self._embed_mask)
            iimage_out = out[0]
            specind_out = out[1]
            curv_out = out[2]
        else:
            if np.any(np.invert(self._embed_mask)):
                out = imutils.embed(out, self._embed_mask)
            iimage_out = out

        # Return image
        arglist, argdict = self.prior_next.image_args()
        arglist[0] = iimage_out.reshape(self.prior_next.ydim, self.prior_next.xdim)
        argdict['pol_prim'] = pol_prim
        outim = ehtim.image.Image(*arglist, **argdict)

        # Copy over other polarizations
        for pol2 in list(outim._imdict.keys()):

            # Is it the base image?
            if pol2 == outim.pol_prim:
                continue

            # Did we solve for polarimeric image or are we copying over old pols?
            if self.pol_next in ['P','IP','IQU'] and pol2 == 'Q':
                polvec = qimage_out
            elif self.pol_next in ['P','IP','IQU'] and pol2 == 'U':
                polvec = uimage_out
            else:
                polvec = self.init_next._imdict[pol2]

            if len(polvec):
                polarr = polvec.reshape(outim.ydim, outim.xdim)
                outim.add_pol_image(polarr, pol2)

        # Copy over spectral index information
        outim._mflist = copy.deepcopy(self.init_next._mflist)
        if self.mf_next:
            outim._mflist[0] = specind_out
            outim._mflist[1] = curv_out

        # Append to history
        logstr = str(self.nruns) + ": make_image(pol=%s)" % pol
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

    def make_image_IP(self, grads=True, **kwargs):
        """Make Stokes I and P polarimetric image simultaneously using current imager settings.
        """
        return self.make_image(pol='IP', grads=grads, **kwargs)

    def set_embed(self):
        """Set embedding matrix.
        """
        self._embed_mask = self.prior_next.imvec > self.clipfloor_next
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

    def check_params(self):
        """Check parameter consistency.
        """
        if ((self.prior_next.psize != self.init_next.psize) or
            (self.prior_next.xdim != self.init_next.xdim) or
                (self.prior_next.ydim != self.prior_next.ydim)):
            raise Exception("Initial image does not match dimensions of the prior image!")

        if ((self.prior_next.rf != self.init_next.rf)):
            raise Exception("Initial image does not have same frequency as prior image!")

        if (self.prior_next.polrep != self.init_next.polrep):
            raise Exception(
                "Initial image polrep does not match prior polrep!")

        if (self.prior_next.polrep == 'circ' and not(self.pol_next in ['P', 'RR', 'LL'])):
            raise Exception("Initial image polrep is 'circ': pol_next must be 'RR' or 'LL' or 'P'!")

        if (self.prior_next.polrep == 'stokes' and not(self.pol_next in ['I', 'Q', 'U', 'V', 'P','IP','IQU'])):
            raise Exception(
                "Initial image polrep is 'stokes': pol_next must be in 'I','Q','U','V','P','IP','IQU'!")

        if ('log' in self.transform_next and self.pol_next in ['Q', 'U', 'V']):
            raise Exception("Cannot image Stokes Q, U, V with log image transformation!")

        if self._ttype not in ['fast', 'direct', 'nfft']:
            raise Exception("Possible ttype values are 'fast', 'direct','nfft'!")

        if(self.pol_next in ['Q', 'U', 'V'] and
           ('gs' in self.reg_term_next.keys() or 'simple' in self.reg_term_next.keys())):
            raise Exception(
                "'simple' and 'gs' methods do not work with Stokes Q, U, or V images!")

        # Catch errors in multifrequency imaging setup
        if self.mf_next and len(set(self.freq_list)) < 2:
            raise Exception(
                "must have observations at at least two frequencies for multifrequency imaging!")

        # Catch errors for polarimetric imaging setup
        if (self.pol_next == 'P'):
            if 'mcv' not in self.transform_next:
                raise Exception("P imaging needs 'mcv' transform!")
            if (self._ttype not in ["direct", "nfft"]):
                raise Exception("FFT no yet implemented in polarimetric imaging -- use NFFT!")

            dt_here = False
            dt_type = True
            for term in sorted(self.dat_term_next.keys()):
                if (term is not None) and (term is not False):
                    dt_here = True
                if not ((term in DATATERMS_POL) or (term is False)):
                    dt_type = False

            st_here = False
            st_type = True
            for term in sorted(self.reg_term_next.keys()):
                if (term is not None) and (term is not False):
                    st_here = True
                if not ((term in REGULARIZERS_POL) or (term is False)):
                    st_type = False

            if not dt_here:
                raise Exception("Must have at least one data term!")
            if not st_here:
                raise Exception("Must have at least one regularizer term!")
            if not dt_type:
                raise Exception(
                    "Invalid data term for P imaging: " +
                    "valid data terms are: " + ','.join(DATATERMS_POL))
            if not st_type:
                raise Exception(
                    "Invalid regularizer for P imaging: " +
                    "valid regularizers are: " + ','.join(REGULARIZERS_POL))

        # Catch errors for simultaneous I + polarimetric imaging setup
        elif (self.pol_next == 'IP' or self.pol_next == 'IQU'):
            if 'mcv' not in self.transform_next:
                raise Exception("P imaging needs 'mcv' transform!")
            if (self._ttype not in ["direct", "nfft"]):
                raise Exception("FFT no yet implemented in polarimetric imaging -- use NFFT!")

            dt_here = False
            dt_type = True
            dt_pol = False
            for term in sorted(self.dat_term_next.keys()):
                if (term is not None) and (term is not False):
                    dt_here = True
                if not ((term in DATATERMS_POL) or (term in DATATERMS) or (term is False)):
                    dt_type = False
                if term in DATATERMS_POL:
                    dt_pol = True

            st_here = False
            st_type = True
            for term in sorted(self.reg_term_next.keys()):
                if (term is not None) and (term is not False):
                    st_here = True
                if not ((term in REGULARIZERS_POL) or (term in REGULARIZERS) or (term is False)):
                    st_type = False

            if not dt_here:
                raise Exception("Must have at least one data term!")
            if not st_here:
                raise Exception("Must have at least one regularizer term!")
            if not dt_type:
                raise Exception(
                    "Invalid data term for IP imaging: " +
                    "valid data terms are: " + ','.join(DATATERMS_POL + DATATERMS))
            if not st_type:
                raise Exception(
                    "Invalid regularizer for IP imaging: " +
                    "valid regularizers are: " + ','.join(REGULARIZERS_POL + REGULARIZERS))
            if not dt_pol:
                raise Exception(
                    "IP imaging must have at least one pol data term from: " +
                    ','.join(DATATERMS_POL))

        # Catch errors in single pol imaging setup
        else:
            dt_here = False
            dt_type = True
            for term in sorted(self.dat_term_next.keys()):
                if (term is not None) and (term is not False):
                    dt_here = True
                if not ((term in DATATERMS) or (term is False)):
                    dt_type = False

            st_here = False
            st_type = True
            for term in sorted(self.reg_term_next.keys()):
                if (term is not None) and (term is not False):
                    st_here = True
                if not ((term in REGULARIZERS or
                         term in REGULARIZERS_SPECIND or
                         term in REGULARIZERS_CURV) or
                        term is False):
                    st_type = False

            if not dt_here:
                raise Exception("Must have at least one data term!")
            if not st_here:
                raise Exception("Must have at least one regularizer term!")
            if not dt_type:
                raise Exception("Invalid data term: valid data terms are: " + ','.join(DATATERMS))
            if not st_type:
                raise Exception("Invalid regularizer: valid regularizers are: " +
                                ','.join(REGULARIZERS))

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
            print("Warning! Pixel size is larger than smallest spatial wavelength!")
        if uvmin > minbl:
            print("Warning! Field of View is smaller than largest nonzero spatial wavelength!")

        if self.pol_next in ['I', 'RR', 'LL']:
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
        """Return last image transform used.
        """
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return self._transform_list[-1]

    def init_imager(self):
        """Set up Stokes I imager.
        """
        # Set embedding
        self.set_embed()

        # Set prior & initial image vectors for polarimetric imaging
        if self.pol_next == 'P' or self.pol_next == 'IP' or self.pol_next == 'IQU':

            # initial I, kept constant if pol_next=='P'
            if self.norm_init and (self.pol_next == 'IP' or self.pol_next == 'IQU'):
                self._nprior = (self.flux_next * self.prior_next.imvec /
                                np.sum((self.prior_next.imvec)[self._embed_mask]))[self._embed_mask]
                iinit = (self.flux_next * self.init_next.imvec /
                               np.sum((self.init_next.imvec)[self._embed_mask]))[self._embed_mask]
            else:
                self._nprior = self.prior_next.imvec[self._embed_mask]
                iinit = self.init_next.imvec[self._embed_mask]

            self._nimage = len(iinit)

            # Initialize m & phi
            if (len(self.init_next.qvec) and
                    (np.any(self.init_next.qvec != 0) or np.any(self.init_next.uvec != 0))):
                init1 = (np.abs(self.init_next.qvec + 1j*self.init_next.uvec) /
                         self.init_next.imvec)[self._embed_mask]
                init2 = (np.arctan2(self.init_next.uvec,
                                    self.init_next.qvec) / 2.0)[self._embed_mask]
            else:
                # !AC TODO get the actual zero baseline pol. frac from the data?
                print("No polarimetric image in init_next!")
                print("--initializing with 10% pol and random orientation!")
                init1 = 0.2 * (np.ones(self._nimage) + 1e-2 * np.random.rand(self._nimage))
                init2 = np.zeros(self._nimage) + 1e-2 * np.random.rand(self._nimage)
            self._inittuple = np.array((iinit, init1, init2))

            # Change of variables
            if 'mcv' in self.transform_next:
                if self.pol_next == 'P' or self.pol_next == 'IP' or self.pol_next == 'IQU':
                    self._xtuple = polutils.mcv_r(self._inittuple)
            else:
                raise Exception("Polarimetric imaging only works with mcv transform!")

            # Only apply log transformation to Stokes I if simultaneous imaging
            if ('log' in self.transform_next) and self.pol_next != 'P':
                self._xtuple[0] = np.log(self._xtuple[0])

            # Pack into single vector
            if self.pol_next == 'P':
                pol_which_solve = (0,1,1) # solve only for polarization, fix I
            else:
                pol_which_solve = (1,1,1) # solve simultaneously for full lin pol field

            self._xinit = polutils.pack_poltuple(self._xtuple, pol_which_solve)

        # Set prior & initial image vectors for multifrequency imaging
        elif self.mf_next:

            self.reffreq = self.init_next.rf # set reference frequency to same as prior
            # reset logfreqratios in case reference frequency changed
            self._logfreqratio_list = [np.log(nu/self.reffreq) for nu in self.freq_list]

            if self.norm_init:
                nprior_I = (self.flux_next * self.prior_next.imvec /
                            np.sum((self.prior_next.imvec)[self._embed_mask]))[self._embed_mask]
                ninit_I = (self.flux_next * self.init_next.imvec /
                           np.sum((self.init_next.imvec)[self._embed_mask]))[self._embed_mask]
            else:
                nprior_I = self.prior_next.imvec[self._embed_mask]
                ninit_I = self.init_next.imvec[self._embed_mask]

            if len(self.init_next.specvec):
                ninit_a = self.init_next.specvec[self._embed_mask]
            else:
                ninit_a = np.zeros(self._nimage)[self._embed_mask]
            if len(self.prior_next.specvec):
                nprior_a = self.prior_next.specvec[self._embed_mask]
            else:
                nprior_a = np.zeros(self._nimage)[self._embed_mask]

            if len(self.init_next.curvvec):
                ninit_b = self.init_next.curvvec[self._embed_mask]
            else:
                ninit_b = np.zeros(self._nimage)[self._embed_mask]
            if len(self.prior_next.curvvec):
                nprior_b = self.init_next.curvvec[self._embed_mask]
            else:
                nprior_b = np.zeros(self._nimage)[self._embed_mask]

            self._nimage = len(ninit_I)

            self.inittuple = np.array((ninit_I, ninit_a, ninit_b))
            self.priortuple = np.array((nprior_I, nprior_a, nprior_b))

            # Change of variables
            if 'log' in self.transform_next:
                self._xtuple = np.array((np.log(ninit_I), ninit_a, ninit_b))
            else:
                self._xtuple = self.inittuple

            # Pack into single vector
            self._xinit = mfutils.pack_mftuple(self._xtuple, self.mf_which_solve)

        # Set prior & initial image vectors for single stokes or RR/LL imaging
        else:

            if self.norm_init:
                self._nprior = (self.flux_next * self.prior_next.imvec /
                                np.sum((self.prior_next.imvec)[self._embed_mask]))[self._embed_mask]
                ninit = (self.flux_next * self.init_next.imvec /
                               np.sum((self.init_next.imvec)[self._embed_mask]))[self._embed_mask]
            else:
                self._nprior = self.prior_next.imvec[self._embed_mask]
                ninit = self.init_next.imvec[self._embed_mask]

            self._nimage = len(ninit)
            # Change of variables
            if 'log' in self.transform_next:
                self._xinit = np.log(ninit)
            else:
                self._xinit = ninit

        # Make data term tuples
        if self._change_imgr_params:
            if self.nruns == 0:
                print("Initializing imager data products . . .")
            if self.nruns > 0:
                print("Recomputing imager data products . . .")
#            if hasattr(self, "_data_tuples"):
#                del self._data_tuples
            self._data_tuples = {}

            # Loop over all data term types
            for dname in sorted(self.dat_term_next.keys()):

                # Loop over all observations in the list
                for i, obs in enumerate(self.obslist_next):
                    # Each entry in the dterm dictionary past the first has an appended number
                    if i == 0:
                        dname_key = dname
                    else:
                        dname_key = dname + ('_%i' % i)

                    # Polarimetric data products
                    if dname in DATATERMS_POL:
                    #if self.pol_next == 'P':
                        tup = polutils.polchisqdata(obs, self.prior_next, self._embed_mask, dname,
                                                    ttype=self._ttype,
                                                    fft_pad_factor=self._fft_pad_factor,
                                                    conv_func=self._fft_conv_func,
                                                    p_rad=self._fft_gridder_prad)

                    # Single polarization data products
                    elif dname in DATATERMS:
                        if self.pol_next == 'IP' or self.pol_next == 'IQU':
                            pol_next = 'I'
                        elif self.pol_next == 'P':
                            raise Exception("cannot use dterm %s with pol=P - did you mean to use pol=IP?")
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

    def init_imager_scattering(self):
        """Set up scattering imager.
        """
        N = self.prior_next.xdim

        if self.scattering_model is None:
            self.scattering_model = so.ScatteringModel()

        # First some preliminary definitions
        wavelength = ehc.C/self.obs_next.rf*100.0  # Observing wavelength [cm]
        N = self.prior_next.xdim

        # Field of view, in cm, at the scattering screen
        FOV = self.prior_next.psize * N * self.scattering_model.observer_screen_distance

        # The ensemble-average convolution kernel and its gradients
        self._ea_ker = self.scattering_model.Ensemble_Average_Kernel(
            self.prior_next, wavelength_cm=wavelength)
        ea_ker_gradient = so.Wrapped_Gradient(self._ea_ker/(FOV/N))
        self._ea_ker_gradient_x = -ea_ker_gradient[1]
        self._ea_ker_gradient_y = -ea_ker_gradient[0]

        # The power spectrum
        # Note: rotation is not currently implemented;
        # the gradients would need to be modified slightly
        self._sqrtQ = np.real(self.scattering_model.sqrtQ_Matrix(self.prior_next, t_hr=0.0))

        # Generate the initial image+screen vector.
        # By default, the screen is re-initialized to zero each time.
        if len(self.epsilon_list_next) == 0:
            self._xinit = np.concatenate((self._xinit, np.zeros(N**2-1)))
        else:
            self._xinit = np.concatenate((self._xinit, self.epsilon_list_next))

    def make_chisq_dict(self, imcur):
        """Make a dictionary of current chi^2 term values
           i indexes the observation number in self.obslist_next
        """

        chi2_dict = {}
        for dname in sorted(self.dat_term_next.keys()):
            # Loop over all observations in the list
            for i, obs in enumerate(self.obslist_next):
                if i == 0:
                    dname_key = dname
                else:
                    dname_key = dname + ('_%i' % i)

                (data, sigma, A) = self._data_tuples[dname_key]

                if dname in DATATERMS_POL:
                #if self.pol_next == 'P':
                    chi2 = polutils.polchisq(imcur, A, data, sigma, dname,
                                             ttype=self._ttype, mask=self._embed_mask,
                                             pol_prim=POL_PRIM_SOLVE)

                elif dname in DATATERMS:
                    # If multifrequency imaging, get the image at the right frequency
                    if self.mf_next:
                        logfreqratio = self._logfreqratio_list[i]
                        imcur_nu = mfutils.imvec_at_freq(imcur, logfreqratio)
                    elif self.pol_next == 'IP' or self.pol_next == 'IQU':
                        imcur_nu = imcur[0]
                    else:
                        imcur_nu = imcur

                    chi2 = imutils.chisq(imcur_nu, A, data, sigma, dname,
                                         ttype=self._ttype, mask=self._embed_mask)

                else:
                    raise Exception("data term %s not recognized!" % dname)

                chi2_dict[dname_key] = chi2

        return chi2_dict

    def make_chisqgrad_dict(self, imcur, i=0):
        """Make a dictionary of current chi^2 term gradient values
           i indexes the observation number in self.obslist_next
        """
        chi2grad_dict = {}
        for dname in sorted(self.dat_term_next.keys()):
            # Loop over all observations in the list
            for i, obs in enumerate(self.obslist_next):
                if i == 0:
                    dname_key = dname
                else:
                    dname_key = dname + ('_%i' % i)

                (data, sigma, A) = self._data_tuples[dname_key]

                # Polarimetric data products
                if dname in DATATERMS_POL:
                    chi2grad = polutils.polchisqgrad(imcur, A, data, sigma, dname,
                                                     ttype=self._ttype, mask=self._embed_mask,
                                                     pol_prim=POL_PRIM_SOLVE,
                                                     pol_solve=POL_WHICH_SOLVE)

                # Single polarization data products
                elif dname in DATATERMS:
                    # If multifrequency imaging, get the image at the right frequency
                    if self.mf_next:
                        logfreqratio = self._logfreqratio_list[i]
                        imref = imcur[0]
                        imcur_nu = mfutils.imvec_at_freq(imcur, logfreqratio)
                    elif self.pol_next == 'IP' or self.pol_next == 'IQU':
                        imcur_nu = imcur[0]
                    else:
                        imcur_nu = imcur

                    chi2grad = imutils.chisqgrad(imcur_nu, A, data, sigma, dname,
                                                 ttype=self._ttype, mask=self._embed_mask)

                    # If multifrequency imaging,
                    # transform the image gradients for all the solved quantities
                    if self.mf_next:
                        logfreqratio = self._logfreqratio_list[i]
                        chi2grad = mfutils.mf_all_chisqgrads(
                            chi2grad, imcur_nu, imref, logfreqratio)

                    # If imaging polarization simultaneously, bundle the gradient properly
                    if self.pol_next == 'IP' or self.pol_next == 'IQU': # TODO AC AA MAKE MORE ELEGANT
                        chi2grad = np.array((chi2grad, np.zeros(self._nimage), np.zeros(self._nimage)))

                else:
                    raise Exception("data term %s not recognized!" % dname)

                chi2grad_dict[dname_key] = np.array(chi2grad)

        return chi2grad_dict

    def make_reg_dict(self, imcur):
        """Make a dictionary of current regularizer values
        """
        reg_dict = {}
        for regname in sorted(self.reg_term_next.keys()):
            # Polarimetric regularizer
            #if self.pol_next == 'P':
            if regname in REGULARIZERS_POL:
                reg = polutils.polregularizer(imcur, self._embed_mask, self.flux_next,
                                              self.prior_next.xdim, self.prior_next.ydim,
                                              self.prior_next.psize, regname,
                                              norm_reg=self.norm_reg, beam_size=self.beam_size,
                                              pol_prim=POL_PRIM_SOLVE, pol_solve=POL_WHICH_SOLVE)

            # Multifrequency regularizer
            elif self.mf_next:
                if regname in REGULARIZERS:
                    reg = imutils.regularizer(imcur[0], self.priortuple[0], self._embed_mask,
                                              self.flux_next, self.prior_next.xdim,
                                              self.prior_next.ydim, self.prior_next.psize,
                                              regname,
                                              norm_reg=self.norm_reg, beam_size=self.beam_size,
                                              **self.regparams)
                elif regname in REGULARIZERS_SPECIND:
                    reg = mfutils.regularizer_mf(imcur[1], self.priortuple[1], self._embed_mask,
                                                 self.flux_next, self.prior_next.xdim,
                                                 self.prior_next.ydim, self.prior_next.psize,
                                                 regname,
                                                 norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                 **self.regparams)

                elif regname in REGULARIZERS_CURV:
                    reg = mfutils.regularizer_mf(imcur[2], self.priortuple[2], self._embed_mask,
                                                 self.flux_next, self.prior_next.xdim,
                                                 self.prior_next.ydim, self.prior_next.psize,
                                                 regname,
                                                 norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                 **self.regparams)

            # Normal regularizer
            elif regname in REGULARIZERS:
                if self.pol_next == 'IP' or self.pol_next == 'IQU':
                    imcur0 = imcur[0]
                else:
                    imcur0 = imcur

                reg = imutils.regularizer(imcur0, self._nprior, self._embed_mask,
                                          self.flux_next, self.prior_next.xdim,
                                          self.prior_next.ydim, self.prior_next.psize,
                                          regname,
                                          norm_reg=self.norm_reg, beam_size=self.beam_size,
                                          **self.regparams)
            else:
                raise Exception("regularizer term %s not recognized!" % regname)

            reg_dict[regname] = reg

        return reg_dict

    def make_reggrad_dict(self, imcur):
        """Make a dictionary of current regularizer gradient values
        """

        reggrad_dict = {}
        for regname in sorted(self.reg_term_next.keys()):

            # Polarimetric regularizer
            #if self.pol_next == 'P':
            if regname in REGULARIZERS_POL:
                reg = polutils.polregularizergrad(imcur, self._embed_mask, self.flux_next,
                                                  self.prior_next.xdim, self.prior_next.ydim,
                                                  self.prior_next.psize, regname,
                                                  norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                  pol_prim=POL_PRIM_SOLVE,
                                                  pol_solve=POL_WHICH_SOLVE)

            # Multifrequency regularizer
            elif self.mf_next:

                if regname in REGULARIZERS:
                    reg = imutils.regularizergrad(imcur[0], self.priortuple[0],
                                                  self._embed_mask, self.flux_next,
                                                  self.prior_next.xdim, self.prior_next.ydim,
                                                  self.prior_next.psize, regname,
                                                  norm_reg=self.norm_reg,
                                                  beam_size=self.beam_size,
                                                  **self.regparams)
                    reg = np.array((reg, np.zeros(self._nimage), np.zeros(self._nimage)))

                elif regname in REGULARIZERS_SPECIND:
                    reg = mfutils.regularizergrad_mf(imcur[1], self.priortuple[1],
                                                     self._embed_mask, self.flux_next,
                                                     self.prior_next.xdim, self.prior_next.ydim,
                                                     self.prior_next.psize, regname,
                                                     norm_reg=self.norm_reg,
                                                     beam_size=self.beam_size,
                                                     **self.regparams)
                    reg = np.array((np.zeros(self._nimage), reg, np.zeros(self._nimage)))

                elif regname in REGULARIZERS_CURV:
                    reg = mfutils.regularizergrad_mf(imcur[2], self.priortuple[2],
                                                     self._embed_mask, self.flux_next,
                                                     self.prior_next.xdim, self.prior_next.ydim,
                                                     self.prior_next.psize, regname,
                                                     norm_reg=self.norm_reg,
                                                     beam_size=self.beam_size,
                                                     **self.regparams)
                    reg = np.array((np.zeros(self._nimage), np.zeros(self._nimage), reg))

            # Normal regularizer
            elif regname in REGULARIZERS:
                if self.pol_next == 'IP' or self.pol_next == 'IQU': # TODO AC AA MAKE MORE ELEGANT
                    imcur0 = imcur[0]
                else:
                    imcur0 = imcur
                reg = imutils.regularizergrad(imcur0, self._nprior, self._embed_mask, self.flux_next,
                                              self.prior_next.xdim, self.prior_next.ydim,
                                              self.prior_next.psize,
                                              regname,
                                              norm_reg=self.norm_reg, beam_size=self.beam_size,
                                              **self.regparams)
                if self.pol_next == 'IP' or self.pol_next == 'IQU': # TODO AC AA MAKE MORE ELEGANT
                    reg = np.array((reg, np.zeros(self._nimage), np.zeros(self._nimage)))
            else:
                raise Exception("regularizer term %s not recognized!" % regname)

            reggrad_dict[regname] = reg

        return reggrad_dict

    def objfunc(self, imvec):
        """Current objective function.
        """

        # Unpack polarimetric/multifrequency vector into an array
        if self.pol_next == 'P' or self.pol_next == 'IP' or self.pol_next == 'IQU':
            if self.pol_next == 'P':
                pol_which_solve = (0,1,1) # solve only for polarization, fix I
            else:
                pol_which_solve = (1,1,1) # solve simultaneously for full lin pol field
            imcur = polutils.unpack_poltuple(imvec, self._xtuple, self._nimage, pol_which_solve)
        elif self.mf_next:
            imcur = mfutils.unpack_mftuple(imvec, self._xtuple, self._nimage, self.mf_which_solve)
        else:
            imcur = imvec

        # Image change of variables
        if 'mcv' in self.transform_next:
            if self.pol_next == 'P' or self.pol_next == 'IP' or self.pol_next == 'IQU':
                imcur = polutils.mcv(imcur)

        if 'log' in self.transform_next:
            if self.pol_next == 'P' or self.pol_next == 'IP' or self.pol_next == 'IQU':
                imcur[0] = np.exp(imcur[0])
            elif self.mf_next:
                imcur[0] = np.exp(imcur[0])
            else:
                imcur = np.exp(imcur)

        # Data terms
        datterm = 0.
        chi2_term_dict = self.make_chisq_dict(imcur)
        for dname in sorted(self.dat_term_next.keys()):
            hyperparameter = self.dat_term_next[dname]

            for i, obs in enumerate(self.obslist_next):
                if i == 0:
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

        return datterm + regterm

    def objgrad(self, imvec):
        """Current objective function gradient.
        """

        # Unpack polarimetric/multifrequency vector into an array
        if self.pol_next == 'P' or self.pol_next == 'IP' or self.pol_next == 'IQU':
            if self.pol_next == 'P':
                pol_which_solve = (0,1,1) # solve only for polarization, fix I
            else:
                pol_which_solve = (1,1,1) # solve simultaneously for full lin pol field
            imcur = polutils.unpack_poltuple(imvec, self._xtuple, self._nimage, pol_which_solve)
        elif self.mf_next:
            imcur = mfutils.unpack_mftuple(imvec, self._xtuple, self._nimage, self.mf_which_solve)
        else:
            imcur = imvec

        # Image change of variables
        if 'mcv' in self.transform_next:
            if self.pol_next == 'P' or self.pol_next == 'IP' or self.pol_next == 'IQU':
                cvcur = imcur.copy()
                imcur = polutils.mcv(imcur)

        if 'log' in self.transform_next:
            if self.pol_next == 'P' or self.pol_next == 'IP' or self.pol_next == 'IQU':
                imcur[0] = np.exp(imcur[0])
            elif self.mf_next:
                imcur[0] = np.exp(imcur[0])
            else:
                imcur = np.exp(imcur)


        # Data terms
        datterm = 0.
        chi2_term_dict = self.make_chisqgrad_dict(imcur)
        if self.chisq_transform:
            chi2_value_dict = self.make_chisq_dict(imcur)
        for dname in sorted(self.dat_term_next.keys()):
            hyperparameter = self.dat_term_next[dname]

            for i, obs in enumerate(self.obslist_next):
                if i == 0:
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

        grad = datterm + regterm

        # Chain rule term for change of variables
        if 'mcv' in self.transform_next:
            if self.pol_next == 'P' or self.pol_next == 'IP' or self.pol_next == 'IQU':
                grad *= polutils.mchain(cvcur) # note: mchain is only not 1 for 'm' variables

        if 'log' in self.transform_next:
            if self.pol_next == 'P' or self.pol_next == 'IP' or self.pol_next == 'IQU':
                grad[0] *= imcur[0]
            elif self.mf_next:
                grad[0] *= imcur[0]
            else:
                grad *= imcur

        # Repack gradient for polarimetric imaging
        if self.pol_next == 'P' or self.pol_next == 'IP' or self.pol_next == 'IQU':
            if self.pol_next == 'P': pol_which_solve = (0,1,1)
            else: pol_which_solve = (1,1,1)
            grad = polutils.pack_poltuple(grad, pol_which_solve)

        # repack gradient for multifrequency imaging
        elif self.mf_next:
            grad = mfutils.pack_mftuple(grad, self.mf_which_solve)

        return grad

    def plotcur(self, imvec, **kwargs):
        """Plot current image.
        """

        if self._show_updates:
            if self._nit % self._update_interval == 0:
                if self.pol_next == 'P' or self.pol_next == 'IP' or self.pol_next == 'IQU':
                    if self.pol_next == 'P': pol_which_solve = (0,1,1)
                    else: pol_which_solve = (1,1,1)
                    imcur = polutils.unpack_poltuple(
                        imvec, self._xtuple, self._nimage, pol_which_solve)
                elif self.mf_next:
                    imcur = mfutils.unpack_mftuple(
                        imvec, self._xtuple, self._nimage, self.mf_which_solve)
                else:
                    imcur = imvec

                # Image change of variables

                if 'mcv' in self.transform_next:
                    if self.pol_next == 'P' or self.pol_next == 'IP' or self.pol_next == 'IQU':
                        imcur = polutils.mcv(imcur)

                if 'log' in self.transform_next:
                    if self.pol_next == 'P' or self.pol_next == 'IP' or self.pol_next == 'IQU':
                        imcur[0] = np.exp(imcur[0])
                    elif self.mf_next:
                        imcur[0] = np.exp(imcur[0])
                    else:
                        imcur = np.exp(imcur)

                # Get chi^2 and regularizer
                chi2_term_dict = self.make_chisq_dict(imcur)
                reg_term_dict = self.make_reg_dict(imcur)

                # Format print string
                outstr = "------------------------------------------------------------------"
                outstr += "\n%4d | " % self._nit
                for dname in sorted(self.dat_term_next.keys()):
                    for i, obs in enumerate(self.obslist_next):
                        if i == 0:
                            dname_key = dname
                        else:
                            dname_key = dname + ('_%i' % i)
                        outstr += "chi2_%s : %0.2f " % (dname_key, chi2_term_dict[dname_key])
                outstr += "\n        "
                for dname in sorted(self.dat_term_next.keys()):
                    for i, obs in enumerate(self.obslist_next):
                        if i == 0:
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
                if self.pol_next == 'P' or self.pol_next == 'IP' or self.pol_next == 'IQU':
                    if np.any(np.invert(self._embed_mask)):
                        imcur = polutils.embed_pol(imcur, self._embed_mask)
                    polutils.plot_m(imcur, self.prior_next, self._nit, chi2_term_dict, **kwargs)

                else:
                    if self.mf_next:
                        implot = imcur[0]
                    else:
                        implot = imcur
                    if np.any(np.invert(self._embed_mask)):
                        implot = imutils.embed(implot, self._embed_mask)

                    imutils.plot_i(implot, self.prior_next, self._nit,
                                   chi2_term_dict, pol=self.pol_next, **kwargs)

                if self._nit == 0:
                    print()
                print(outstr)

        self._nit += 1

    def objfunc_scattering(self, minvec):
        """Current stochastic optics objective function.
        """
        N = self.prior_next.xdim

        imvec = minvec[:N**2]
        EpsilonList = minvec[N**2:]
        if 'log' in self.transform_next:
            imvec = np.exp(imvec)

        IM = ehtim.image.Image(imvec.reshape(N, N), self.prior_next.psize,
                               self.prior_next.ra, self.prior_next.dec,
                               self.prior_next.pa, rf=self.obs_next.rf,
                               source=self.prior_next.source, mjd=self.prior_next.mjd)

        # The scattered image vector
        screen = so.MakeEpsilonScreenFromList(EpsilonList, N)
        scatt_im = self.scattering_model.Scatter(IM, Epsilon_Screen=screen,
                                                 ea_ker=self._ea_ker, sqrtQ=self._sqrtQ,
                                                 Linearized_Approximation=True)
        scatt_im = scatt_im.imvec

        # Calculate the chi^2 using the scattered image
        datterm = 0.
        chi2_term_dict = self.make_chisq_dict(scatt_im)
        for dname in sorted(self.dat_term_next.keys()):
            datterm += self.dat_term_next[dname] * (chi2_term_dict[dname] - 1.)

        # Calculate the entropy using the unscattered image
        regterm = 0
        reg_term_dict = self.make_reg_dict(imvec)

        # Make dict also for scattered image
        reg_term_dict_scatt = self.make_reg_dict(scatt_im)

        for regname in sorted(self.reg_term_next.keys()):
            if regname == 'rgauss':
                # Get gradient of the scattered image vector
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
        wavelength = ehc.C/self.obs_next.rf*100.0  # Observing wavelength [cm]
        wavelengthbar = wavelength/(2.0*np.pi)     # lambda/(2pi) [cm]
        N = self.prior_next.xdim

        # Field of view, in cm, at the scattering screen
        FOV = self.prior_next.psize * N * self.scattering_model.observer_screen_distance
        rF = self.scattering_model.rF(wavelength)

        imvec = minvec[:N**2]
        EpsilonList = minvec[N**2:]
        if 'log' in self.transform_next:
            imvec = np.exp(imvec)

        IM = ehtim.image.Image(imvec.reshape(N, N), self.prior_next.psize,
                               self.prior_next.ra, self.prior_next.dec,
                               self.prior_next.pa, rf=self.obs_next.rf,
                               source=self.prior_next.source, mjd=self.prior_next.mjd)

        # The scattered image vector
        screen = so.MakeEpsilonScreenFromList(EpsilonList, N)
        scatt_im = self.scattering_model.Scatter(IM, Epsilon_Screen=screen,
                                                 ea_ker=self._ea_ker, sqrtQ=self._sqrtQ,
                                                 Linearized_Approximation=True)
        scatt_im = scatt_im.imvec

        EA_Image = self.scattering_model.Ensemble_Average_Blur(IM, ker=self._ea_ker)
        EA_Gradient = so.Wrapped_Gradient((EA_Image.imvec/(FOV/N)).reshape(N, N))

        # The gradient signs don't actually matter, but let's make them match intuition
        # (i.e., right to left, bottom to top)
        EA_Gradient_x = -EA_Gradient[1]
        EA_Gradient_y = -EA_Gradient[0]

        Epsilon_Screen = so.MakeEpsilonScreenFromList(EpsilonList, N)
        phi_scr = self.scattering_model.MakePhaseScreen(Epsilon_Screen, IM,
                                                        obs_frequency_Hz=self.obs_next.rf,
                                                        sqrtQ_init=self._sqrtQ)
        phi = phi_scr.imvec.reshape((N, N))
        phi_Gradient = so.Wrapped_Gradient(phi/(FOV/N))
        phi_Gradient_x = -phi_Gradient[1]
        phi_Gradient_y = -phi_Gradient[0]

        # Entropy gradient; wrt unscattered image so unchanged by scattering
        regterm = 0
        reg_term_dict = self.make_reggrad_dict(imvec)

        # Make dict also for scattered image
        reg_term_dict_scatt = self.make_reggrad_dict(scatt_im)

        for regname in sorted(self.reg_term_next.keys()):
            # We need an exception if the regularizer is 'rgauss'
            if regname == 'rgauss':
                # Get gradient of the scattered image vector
                gaussterm = self.reg_term_next[regname] * reg_term_dict_scatt[regname]
                dgauss_dIa = gaussterm.reshape((N, N))

                # Now the chain rule factor to get the gauss gradient wrt the unscattered image
                gx = so.Wrapped_Convolve(
                    self._ea_ker_gradient_x[::-1, ::-1], phi_Gradient_x * (dgauss_dIa))
                gx = (rF**2.0 * gx).flatten()

                gy = so.Wrapped_Convolve(
                    self._ea_ker_gradient_y[::-1, ::-1], phi_Gradient_y * (dgauss_dIa))
                gy = (rF**2.0 * gy).flatten()

                # Now we add the gradient for the unscattered image
                regterm += so.Wrapped_Convolve(self._ea_ker[::-1, ::-1],
                                               (dgauss_dIa)).flatten() + gx + gy

            else:
                regterm += self.reg_term_next[regname] * reg_term_dict[regname]

        # Chi^2 gradient wrt the unscattered image
        # First, the chi^2 gradient wrt to the scattered image
        datterm = 0.
        chi2_term_dict = self.make_chisqgrad_dict(scatt_im)
        for dname in sorted(self.dat_term_next.keys()):
            datterm += self.dat_term_next[dname] * (chi2_term_dict[dname])
        dchisq_dIa = datterm.reshape((N, N))

        # Now the chain rule factor to get the chi^2 gradient wrt the unscattered image
        gx = so.Wrapped_Convolve(self._ea_ker_gradient_x[::-1, ::-1], phi_Gradient_x * (dchisq_dIa))
        gx = (rF**2.0 * gx).flatten()

        gy = so.Wrapped_Convolve(self._ea_ker_gradient_y[::-1, ::-1], phi_Gradient_y * (dchisq_dIa))
        gy = (rF**2.0 * gy).flatten()

        chisq_grad_im = so.Wrapped_Convolve(
            self._ea_ker[::-1, ::-1], (dchisq_dIa)).flatten() + gx + gy

        # Gradient of the data chi^2 wrt to the epsilon screen
        # Preliminary Definitions
        chisq_grad_epsilon = np.zeros(N**2-1)
        i_grad = 0
        ell_mat = np.zeros((N, N))
        m_mat = np.zeros((N, N))
        for ell in range(0, N):
            for m in range(0, N):
                ell_mat[ell, m] = ell
                m_mat[ell, m] = m

        # Real part; top row
        for t in range(1, (N+1)//2):
            s = 0
            grad_term = (wavelengthbar/FOV*self._sqrtQ[s][t] *
                         2.0*np.cos(2.0*np.pi/N*(ell_mat*s + m_mat*t))/(FOV/N))
            grad_term = so.Wrapped_Gradient(grad_term)
            grad_term_x = -grad_term[1]
            grad_term_y = -grad_term[0]

            cge_term = (EA_Gradient_x * grad_term_x + EA_Gradient_y * grad_term_y)
            chisq_grad_epsilon[i_grad] = np.sum(dchisq_dIa * rF**2 * cge_term)

            i_grad = i_grad + 1

        # Real part; remainder
        for s in range(1, (N+1)//2):
            for t in range(N):
                grad_term = (wavelengthbar/FOV*self._sqrtQ[s][t] *
                             2.0*np.cos(2.0*np.pi/N*(ell_mat*s + m_mat*t))/(FOV/N))
                grad_term = so.Wrapped_Gradient(grad_term)
                grad_term_x = -grad_term[1]
                grad_term_y = -grad_term[0]

                cge_term = (EA_Gradient_x * grad_term_x + EA_Gradient_y * grad_term_y)
                chisq_grad_epsilon[i_grad] = np.sum(dchisq_dIa * rF**2 * cge_term)

                i_grad = i_grad + 1

        # Imaginary part; top row
        for t in range(1, (N+1)//2):
            s = 0
            grad_term = (-wavelengthbar/FOV*self._sqrtQ[s][t] *
                         2.0*np.sin(2.0*np.pi/N*(ell_mat*s + m_mat*t))/(FOV/N))
            grad_term = so.Wrapped_Gradient(grad_term)
            grad_term_x = -grad_term[1]
            grad_term_y = -grad_term[0]

            cge_term = (EA_Gradient_x * grad_term_x + EA_Gradient_y * grad_term_y)
            chisq_grad_epsilon[i_grad] = np.sum(dchisq_dIa * rF**2 * cge_term)

            i_grad = i_grad + 1

        # Imaginary part; remainder
        for s in range(1, (N+1)//2):
            for t in range(N):
                grad_term = (-wavelengthbar/FOV*self._sqrtQ[s][t] *
                             2.0*np.sin(2.0*np.pi/N*(ell_mat*s + m_mat*t))/(FOV/N))
                grad_term = so.Wrapped_Gradient(grad_term)
                grad_term_x = -grad_term[1]
                grad_term_y = -grad_term[0]

                cge_term = (EA_Gradient_x * grad_term_x + EA_Gradient_y * grad_term_y)
                chisq_grad_epsilon[i_grad] = np.sum(dchisq_dIa * rF**2 * cge_term)
                i_grad = i_grad + 1

        # Gradient of the chi^2 regularization term for the epsilon screen
        chisq_epsilon_grad = self.alpha_phi_next * 2.0*EpsilonList/((N*N-1)/2.0)

        # Chain rule term for change of variables
        if 'log' in self.transform_next:
            regterm *= imvec
            chisq_grad_im *= imvec

        out = np.concatenate(((regterm + chisq_grad_im), (chisq_grad_epsilon + chisq_epsilon_grad)))
        return out

    def plotcur_scattering(self, minvec):
        """Plot current stochastic optics image/screen
        """
        if self._show_updates:
            if self._nit % self._update_interval == 0:
                N = self.prior_next.xdim

                imvec = minvec[:N**2]
                EpsilonList = minvec[N**2:]
                if 'log' in self.transform_next:
                    imvec = np.exp(imvec)

                IM = ehtim.image.Image(imvec.reshape(N, N), self.prior_next.psize,
                                       self.prior_next.ra, self.prior_next.dec,
                                       self.prior_next.pa, rf=self.obs_next.rf,
                                       source=self.prior_next.source, mjd=self.prior_next.mjd)

                # The scattered image vector
                screen = so.MakeEpsilonScreenFromList(EpsilonList, N)
                scatt_im = self.scattering_model.Scatter(IM, Epsilon_Screen=screen,
                                                         ea_ker=self._ea_ker, sqrtQ=self._sqrtQ,
                                                         Linearized_Approximation=True)
                scatt_im = scatt_im.imvec

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
                # regterm_scattering = self.alpha_phi_next * (chisq_epsilon - 1.0)

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
        """Reconstructs an image of total flux density
           using the stochastic optics scattering mitigation technique.

           Uses the scattering model in Imager.scattering_model.
           If none has been specified, defaults to standard model for Sgr A*.
           Returns the estimated unscattered image.

           Args:
                grads (bool): Flag for whether or not to use analytic gradients.
                show_updates (bool): Flag for whether or not to show updates
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
        self._show_updates = kwargs.get('show_updates', True)
        self._update_interval = kwargs.get('update_interval', 1)
        self.plotcur_scattering(self._xinit)

        # Minimize
        optdict = {'maxiter': self.maxit_next, 'ftol': self.stop_next, 'maxcor': NHIST}
        tstart = time.time()
        if grads:
            res = opt.minimize(self.objfunc_scattering, self._xinit, method='L-BFGS-B',
                               jac=self.objgrad_scattering, options=optdict,
                               callback=self.plotcur_scattering)
        else:
            res = opt.minimize(self.objfunc_scattering, self._xinit, method='L-BFGS-B',
                               options=optdict, callback=self.plotcur_scattering)
        tstop = time.time()

        # Format output
        out = res.x[:N**2]
        if 'log' in self.transform_next:
            out = np.exp(out)
        if np.any(np.invert(self._embed_mask)):
            raise Exception("Embedding is not currently implemented!")
            out = imutils.embed(out, self._embed_mask)

        outim = ehtim.image.Image(out.reshape(N, N), self.prior_next.psize,
                                  self.prior_next.ra, self.prior_next.dec, self.prior_next.pa,
                                  rf=self.prior_next.rf, source=self.prior_next.source,
                                  mjd=self.prior_next.mjd, pulse=self.prior_next.pulse)
        outep = res.x[N**2:]
        screen = so.MakeEpsilonScreenFromList(outep, N)
        outscatt = self.scattering_model.Scatter(outim,
                                                 Epsilon_Screen=screen,
                                                 ea_ker=self._ea_ker, sqrtQ=self._sqrtQ,
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
        self._obs_list.append(self.obslist_next)
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
        self._maxset_list.append(self.clipfloor_next)
        self._maxit_list.append(self.maxit_next)
        self._stop_list.append(self.stop_next)
        self._transform_list.append(self.transform_next)
        self._reg_term_list.append(self.reg_term_next)
        self._dat_term_list.append(self.dat_term_next)
        self._alpha_phi_list.append(self.alpha_phi_next)

        self._out_list.append(outim)
        return
