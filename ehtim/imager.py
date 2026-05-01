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


import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.optimize as opt

import ehtim.const_def as ehc
import ehtim.image
import ehtim.imaging.imager_utils as imutils
import ehtim.imaging.pol_imager_utils as polutils
from ehtim.imaging.imager_backend import (
    DATATERMS,
    DATATERMS_POL,
    POLARIZATION_MODES,
    REGULARIZERS,
    REGULARIZERS_ISPECTRAL,
    REGULARIZERS_POL,
    REGULARIZERS_POLSPECTRAL,
    REGULARIZERS_SPECTRAL,
    compute_chisq_dict,
    compute_chisqgrad_dict,
    compute_embed,
    compute_reg_dict,
    compute_reggrad_dict,
    embed_imarr,
    make_initarr,
    pack_imarr,
    transform_gradients,
    transform_imarr,
    transform_imarr_inverse,
    unpack_imarr,
)

MAXIT = 200  # number of iterations
NHIST = 50   # number of steps to store for hessian approx
MAXLS = 40   # maximum number of line search steps in BFGS-B
STOP = 1e-6  # convergence criterion
EPS = 1e-8

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

MEANPOL_INIT = 0.2 # mean initial polarization if not in initial image
SIGMAPOL_INIT = 1.e-2 # perturbations to initial polarization if not in initial image

###################################################################################################
@dataclass
class ImagerRunState:
    """Snapshot of parameters and result from one imaging run."""
    out: Any
    obslist: list
    init: Any
    prior: Any
    reg_term: dict
    dat_term: dict
    maxit: int
    stop: float
    pol: str
    flux: float
    pflux: Any
    vflux: Any
    clipfloor: float
    snrcut: dict
    debias: bool
    systematic_noise: float
    systematic_cphase_noise: float
    transform: Any
    weighting: str
    maxset: bool
    cp_uv_min: Any
    reffreq: float
    mf: bool
    mf_order: int
    mf_order_pol: int
    mf_rm: int
    mf_cm: int


# Imager object
###################################################################################################


class Imager:
    """A general interferometric imager.
    """

    def __init__(self, obs_in, init_im,
                 prior_im=None, flux=None, data_term=DAT_DEFAULT, reg_term=REG_DEFAULT, **kwargs):

        self.logstr = ""
        self._history: list[ImagerRunState] = []

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
        self.debias_next = kwargs.get('debias', False)
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
        self.mf_flux = kwargs.get('mf_flux',[self.flux_next]) # TODO: merge these

        if kwargs.get('mf_which_solve') is not None:
            raise Exception("'mf_which_solve' argument for multifrequency imaging is deprecated -- use 'mf_order' instead!")
        if kwargs.get('reg_all_freq_mf') is not None:
            raise Exception("'reg_all_freq_mf' argument for multifrequency imaging is deprecated")

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

    def _last(self, field: str) -> Any:
        if self.nruns == 0:
            print("No imager runs yet!")
            return
        return getattr(self._history[-1], field)

    def reg_terms_last(self):
        """Return last used regularizer terms."""
        return self._last('reg_term')

    def dat_terms_last(self):
        """Return last used data terms."""
        return self._last('dat_term')

    def obslist_last(self):
        """Return last used observation list."""
        return self._last('obslist')

    def obs_last(self):
        """Return last used observation."""
        return self._last('obslist')[0]

    def prior_last(self):
        """Return last used prior image."""
        return self._last('prior')

    def out_last(self):
        """Return last result."""
        return self._last('out')

    def init_last(self):
        """Return last initial image."""
        return self._last('init')

    def flux_last(self):
        """Return last total flux constraint."""
        return self._last('flux')

    def pflux_last(self):
        """Return last total linear polarimetric flux constraint."""
        return self._last('pflux')

    def vflux_last(self):
        """Return last total circular polarimetric flux constraint."""
        return self._last('vflux')

    def clipfloor_last(self):
        """Return last clip floor."""
        return self._last('clipfloor')

    def pol_last(self):
        """Return last polarization imaged."""
        return self._last('pol')

    def maxit_last(self):
        """Return last max_iterations value."""
        return self._last('maxit')

    def stop_last(self):
        """Return last convergence value."""
        return self._last('stop')

    def debias_last(self):
        """Return last debias value."""
        return self._last('debias')

    def snrcut_last(self):
        """Return last snrcut value."""
        return self._last('snrcut')

    def weighting_last(self):
        """Return last weighting value."""
        return self._last('weighting')

    def systematic_noise_last(self):
        """Return last systematic_noise value."""
        return self._last('systematic_noise')

    def systematic_cphase_noise_last(self):
        """Return last closure phase systematic noise value (in degrees)."""
        return self._last('systematic_cphase_noise')

    def maxset_last(self):
        """Return last choice of closure phase maximal/minimal set."""
        return self._last('maxset')

    def cp_uv_min_last(self):
        """Return last choice of minimal uvdistance for closure phases."""
        return self._last('cp_uv_min')

    def mf_last(self):
        """Return last choice for multifrequency imaging."""
        return self._last('mf')

    def reffreq_last(self):
        """Return last reference frequency for multifrequency imaging."""
        return self._last('reffreq')

    def mf_order_last(self):
        """Return last choice of order for multifrequency imaging."""
        return self._last('mf_order')

    def mf_order_pol_last(self):
        """Return last choice of order for multifrequency polarimetric imaging."""
        return self._last('mf_order_pol')

    def mf_rm_last(self):
        """Return last choice for RM imaging."""
        return self._last('mf_rm')

    def mf_cm_last(self):
        """Return last choice for CM imaging."""
        return self._last('mf_cm')

    def transform_last(self):
        """Return last image transform used."""
        return self._last('transform')

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
        print(f"Imager run {int(self.nruns)+1} ")

        # multifrequency parameters
        self.mf_next = mf
        self.mf_order = kwargs.get('mf_order', self.mf_order)
        self.mf_order_pol = kwargs.get('mf_order_pol', self.mf_order_pol)
        self.mf_rm = kwargs.get('mf_rm', self.mf_rm)
        self.mf_cm = kwargs.get('mf_cm', self.mf_cm)
        if kwargs.get('mf_which_solve') is not None:
            raise Exception("'mf_which_solve' argument for multifrequency imaging is deprecated -- use 'mf_order' instead!")
        if kwargs.get('reg_all_freq_mf') is not None:
            raise Exception("'reg_all_freq_mf' argument for multifrequency imaging is deprecated")

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
        print("Imaging . . .")
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
                    dname_key = dname + (f'_{i}')
                outstr += f"chi2_{dname_key} : {chi2_term_dict[dname_key]:0.2f} "

        try:
            print(f"time: {tstop - tstart:f} s")
            print(f"J: {res.fun:f}")
            print(outstr)
            if isinstance(res.message, str):
                print(res.message)
            else:
                print(res.message.decode())
        except Exception:  # TODO -- issues for some users with res.message
            pass

        print("==============================")

        # Format final image object
        outim = self.format_outim(outarr, pol_prim=pol_prim)

        # Append to history
        logstr = str(self.nruns) + f": make_image(pol={pol})"
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

        if (self.prior_next.rf != self.init_next.rf):
            raise Exception("Initial image does not have same frequency as prior image!")

        if (self.prior_next.polrep != self.init_next.polrep):
            raise Exception(
                "Initial image polrep does not match prior polrep!")

        if (self.prior_next.polrep == 'circ' and self.pol_next not in ['RR', 'LL']):
            raise Exception("Initial image polrep is 'circ': pol_next must be 'RR' or 'LL'")

        if (self.prior_next.polrep == 'stokes'
            and self.pol_next not in ['I', 'Q', 'U', 'V', 'P','IP','IQU','IV','IQUV']):
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
            if self.mf_order not in [0,1,2]:
                raise Exception("mf_order must be in [0,1,2]!")

            if (self.pol_next in POLARIZATION_MODES):
                if self.pol_next not in ['P','QU']:
                    raise Exception("Currently we only support pol_next=P for polarization multifrequency imaging!")
                if self.mf_order_pol not in [0,1,2]:
                    raise Exception("mf_order_pol must be in [0,1,2]!")

        # Catch errors for polarimetric imaging setup
        if self.pol_next in POLARIZATION_MODES:
            if (self.pol_next in ['P', 'QU','IP','IQU']):
                if 'mcv' not in self.transform_next:
                    raise Exception(f"{self.pol_next} imaging requires 'mcv' transform!")
                if 'vcv' in self.transform_next:
                    raise Exception(f"Cannot do {self.pol_next} imaging with 'vcv' transform!")
                if 'polcv' in self.transform_next:
                    raise Exception(f"Cannot do {self.pol_next} imaging only with 'polcv' transform!")

            if (self.pol_next in ['V','IV']):
                if 'vcv' not in self.transform_next:
                    raise Exception(f"{self.pol_next} imaging requires 'vcv' transform!")
                if 'mcv' in self.transform_next:
                    raise Exception(f"Cannot do {self.pol_next} imaging only with 'mcv' transform!")
                if 'polcv' in self.transform_next:
                    raise Exception(f"Cannot do {self.pol_next} imaging only with 'polcv' transform!")

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
                rlist = REGULARIZERS + REGULARIZERS_ISPECTRAL
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
                print(f"added {term} to data terms")
                self._change_imgr_params = True
                return

        for term in sorted(self.reg_term_next.keys()):
            if term not in self.reg_terms_last().keys():
                print(f"added {term} to regularizers!")
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
        for i,obs in enumerate(self.obslist_next):
            uvmax = 1.0/self.prior_next.psize
            uvmin = 1.0/(self.prior_next.psize*np.max((self.prior_next.xdim, self.prior_next.ydim)))
            uvdists = obs.unpack('uvdist')['uvdist']
            maxbl = np.max(uvdists)
            minbl = np.max(uvdists[uvdists > 0])

            if uvmax < maxbl:
                print("Warning! Pixel size is larger than smallest spatial wavelength for freq %.1f GHz!"%(obs.rf/1.e9))
            if uvmin > minbl:
                print("Warning! Field of View is smaller than largest nonzero spatial wavelength for freq %.1f GHz!"%(obs.rf/1.e9))

            if self.pol_next in ['I', 'RR', 'LL']:
                maxamp = np.max(np.abs(obs.unpack('amp')['amp']))

                # TODO: better handling of mf fluxes
                if len(self.mf_flux)==len(self.obslist_next):
                    flux = self.mf_flux[i]
                else:
                    flux = self.flux_next

                if flux > 1.2*maxamp:
                    print(f"Warning! Specified flux {flux:.1f} is > 120% of maximum visibility amplitude for freq {obs.rf/1.e9:.1f} GHz!")
                if flux < .8*maxamp:
                    print(f"Warning! Specified flux {flux:.1f} is < 80% of maximum visibility amplitude for freq {obs.rf/1.e9:.1f} GHz!")
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
            if self.mf_order == 2:
                do_a = 1
                do_b = 1
            elif self.mf_order == 1:
                do_a = 1
                do_b = 0
            elif self.mf_order == 0:
                do_a = 0
                do_b = 0
            else:
                raise Exception("Imager.mf_order must be 0, 1, or 2!")

            # polarization multi-frequency
            if self.pol_next in POLARIZATION_MODES:

                # determine self._which_solve
                # TODO is there a nicer way to do this?
                if self.mf_order_pol == 2:
                    do_ap = 1
                    do_bp = 1
                elif self.mf_order_pol == 1:
                    do_ap = 1
                    do_bp = 0
                elif self.mf_order_pol == 0:
                    do_ap = 0
                    do_bp = 0
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
                do_rho = 1
                do_phi = 1
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
                    do_rho = 1
                    do_phi = 1
                else:
                    do_rho = 0
                    do_phi = 0

                if ('V' in self.pol_next):
                    do_psi = 1
                else:
                    do_psi = 0


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
                        dname_key = dname + (f'_{i}')

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
                            if 'I' not in self.pol_next:
                                raise Exception(f"cannot use dterm {dname} with pol={self.pol_next}")
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
                        raise Exception(f"data term {dname} not recognized!")

                    self._data_tuples[dname_key] = tup

            self._change_imgr_params = False

        return

    def set_embed(self):
        """Set embedding matrix."""
        self._embed_mask, self._coord_matrix = compute_embed(
            self.prior_next.imvec, self.prior_next.xdim,
            self.prior_next.ydim, self.prior_next.psize,
            self.clipfloor_next,
        )


    def make_chisq_dict(self, imcur):
        """Make a dictionary of current chi^2 term values
           input is image array transformed to bounded values
        """
        return compute_chisq_dict(
            imcur, sorted(self.dat_term_next.keys()), self._data_tuples,
            self.obslist_next, self._logfreqratio_list, self.mf_next,
            self.pol_next, self._ttype, self._embed_mask,
        )

    def make_chisqgrad_dict(self, imcur):
        """Make a dictionary of current chi^2 term gradient values
           input is image array transformed to bounded values
        """
        return compute_chisqgrad_dict(
            imcur, sorted(self.dat_term_next.keys()), self._data_tuples,
            self.obslist_next, self._logfreqratio_list, self.mf_next,
            self.pol_next, self._ttype, self._embed_mask,
            self._which_solve, self._nimage,
        )

    def _full_regparams(self):
        """Bundle all regularizer params into a single dict for the backend."""
        return {
            'flux': self.flux_next,
            'pflux': self.pflux_next,
            'vflux': self.vflux_next,
            'xdim': self.prior_next.xdim,
            'ydim': self.prior_next.ydim,
            'psize': self.prior_next.psize,
            'beam_size': self.beam_size,
            'mf_flux': self.mf_flux,
            **self.regparams,
        }

    def make_reg_dict(self, imcur):
        """Make a dictionary of current regularizer values
           input is image array transformed to bounded values
        """
        return compute_reg_dict(
            imcur, sorted(self.reg_term_next.keys()), self._xprior, self._embed_mask,
            self.mf_next, self.obslist_next, self._logfreqratio_list, self.pol_next,
            self.norm_reg, self._full_regparams(),
        )

    def make_reggrad_dict(self, imcur):
        """Make a dictionary of current regularizer gradient values
           input is image array transformed to bounded values
        """
        return compute_reggrad_dict(
            imcur, sorted(self.reg_term_next.keys()), self._xprior, self._embed_mask,
            self.mf_next, self.obslist_next, self._logfreqratio_list, self.pol_next,
            self.norm_reg, self._full_regparams(),
            self._which_solve, self._nimage,
        )

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
                    dname_key = dname + (f'_{i}')

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
                    dname_key = dname + (f'_{i}')

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
                outstr += f"\n{self._nit:4d} | "
                for dname in sorted(self.dat_term_next.keys()):
                    for i, obs in enumerate(self.obslist_next):
                        if len(self.obslist_next)==1:
                            dname_key = dname
                        else:
                            dname_key = dname + (f'_{i}')
                        outstr += f"chi2_{dname_key} : {chi2_term_dict[dname_key]:0.2f} "
                outstr += "\n        "
                for dname in sorted(self.dat_term_next.keys()):
                    for i, obs in enumerate(self.obslist_next):
                        if len(self.obslist_next)==1:
                            dname_key = dname
                        else:
                            dname_key = dname + (f'_{i}')
                        dval = chi2_term_dict[dname_key]*self.dat_term_next[dname]
                        outstr += f"{dname_key} : {dval:0.1f} "

                outstr += "\n        "
                for regname in sorted(self.reg_term_next.keys()):

                    rval = reg_term_dict[regname]*self.reg_term_next[regname]
                    outstr += f"{regname} : {rval:0.1f} "

                # Embed and plot the image
                if not self.mf_next: # TODO plot multi-frequency?
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
        self._history.append(ImagerRunState(
            out=outim,
            obslist=self.obslist_next,
            init=self.init_next,
            prior=self.prior_next,
            reg_term=self.reg_term_next,
            dat_term=self.dat_term_next,
            maxit=self.maxit_next,
            stop=self.stop_next,
            pol=self.pol_next,
            flux=self.flux_next,
            pflux=self.pflux_next,
            vflux=self.vflux_next,
            clipfloor=self.clipfloor_next,
            snrcut=self.snrcut_next,
            debias=self.debias_next,
            systematic_noise=self.systematic_noise_next,
            systematic_cphase_noise=self.systematic_cphase_noise_next,
            transform=self.transform_next,
            weighting=self.weighting_next,
            maxset=self.maxset_next,
            cp_uv_min=self.cp_uv_min,
            reffreq=self.reffreq,
            mf=self.mf_next,
            mf_order=self.mf_order,
            mf_order_pol=self.mf_order_pol,
            mf_rm=self.mf_rm,
            mf_cm=self.mf_cm,
        ))


