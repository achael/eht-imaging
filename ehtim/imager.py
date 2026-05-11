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
    compute_chisq_dict,
    compute_chisqgrad_dict,
    compute_init_state,
    compute_logfreqratios,
    compute_objective,
    compute_objective_grad,
    compute_reg_dict,
    compute_reggrad_dict,
    transform_imarr,
    unpack_imarr,
    validate_limits,
    validate_params,
)
from ehtim.imaging.imager_utils import embed_imarr

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
        self._logfreqratio_list = compute_logfreqratios(self.freq_list, self.reffreq)

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
        self.plotcur(self._init_vec, **kwargs)

        # Minimize
        print("Imaging . . .")
        optdict = {'maxiter': self.maxit_next,
                   'ftol': self.stop_next, 'gtol': self.stop_next,
                   'maxcor': NHIST, 'maxls': MAXLS}
        def callback_func(xcur):
            self.plotcur(xcur, **kwargs)


        tstart = time.time()
        if grads:
            res = opt.minimize(self.objfunc, self._init_vec, method='L-BFGS-B', jac=self.objgrad,
                               options=optdict, callback=callback_func)
        else:
            res = opt.minimize(self.objfunc, self._init_vec, method='L-BFGS-B',
                               options=optdict, callback=callback_func)
        tstop = time.time()

        # Format output
        out = res.x[:]
        self.tmpout = res.x

        # unpack output vector into outarr
        outarr = unpack_imarr(out, self._init_arr, self._which_solve)

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
        validate_params(
            self.prior_next, self.init_next, self.pol_next,
            self.transform_next,
            self.dat_term_next.keys(), self.reg_term_next.keys(),
            self._ttype, self.mf_next, self.mf_order, self.mf_order_pol,
            self.freq_list,
        )

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
        for msg in validate_limits(
            self.prior_next, self.obslist_next, self.pol_next,
            self.flux_next, self.mf_flux,
        ):
            print(msg)


    def init_imager(self):
        """Initialize the solver-ready state on this Imager.

        Thin wrapper around compute_init_state. Produces and assigns the
        nine attributes consumed by compute_objective / compute_objective_grad:
        _init_arr, _init_vec, _prior_arr, _data_tuples, _embed_mask,
        _coord_matrix, _logfreqratio_list, _nimage, _which_solve.

        See the naming-convention block at the top of imager_backend.py for
        init_* vs prior_*, *_phys vs *_solver, *_arr vs *_vec.
        """
        n_obs = len(self.obslist_next)
        if not (len(self.freq_list) == n_obs
                and len(self._logfreqratio_list) == n_obs):
            raise Exception(
                "Imager state inconsistent: len(obslist_next), len(freq_list), "
                "len(_logfreqratio_list) must all match."
            )

        if self._change_imgr_params:
            msg = ("Initializing imager data products . . ."
                   if self.nruns == 0
                   else "Recomputing imager data products . . .")
            print(msg)

        state = compute_init_state(
            self.obslist_next, self.init_next, self.prior_next,
            self.freq_list, self.reffreq,
            self.pol_next, self.mf_next, self.transform_next,
            self.mf_order, self.mf_order_pol, self.mf_rm, self.mf_cm,
            self.norm_init, self.flux_next, self.clipfloor_next,
            sorted(self.dat_term_next.keys()), self._ttype,
            self._full_data_weighting_params(), self._full_fft_params(),
            compute_data=self._change_imgr_params,
            prior_data_tuples=getattr(self, "_data_tuples", None),
        )

        self._init_arr = state.init_arr
        self._init_vec = state.init_vec
        self._prior_arr = state.prior_arr
        self._data_tuples = state.data_tuples
        self._embed_mask = state.embed_mask
        self._coord_matrix = state.coord_matrix
        self._logfreqratio_list = state.logfreqratio_list
        self._nimage = state.nimage
        self._which_solve = state.which_solve
        self.reffreq = state.reffreq

        if self._change_imgr_params:
            self._change_imgr_params = False

    def make_chisq_dict(self, imcur):
        """Make a dictionary of current chi^2 term values
           input is image array transformed to bounded values
        """
        return compute_chisq_dict(
            imcur, sorted(self.dat_term_next.keys()),
            self.mf_next, self.pol_next,
            self._data_tuples, self._logfreqratio_list, len(self.obslist_next),
            self._ttype, self._embed_mask,
        )

    def make_chisqgrad_dict(self, imcur):
        """Make a dictionary of current chi^2 term gradient values
           input is image array transformed to bounded values
        """
        return compute_chisqgrad_dict(
            imcur, sorted(self.dat_term_next.keys()),
            self.mf_next, self.pol_next,
            self._data_tuples, self._logfreqratio_list, len(self.obslist_next),
            self._ttype, self._embed_mask,
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

    def _full_data_weighting_params(self):
        """Bundle data-weighting params into a single dict for the backend."""
        return {
            'maxset': self.maxset_next,
            'debias': self.debias_next,
            'snrcut': self.snrcut_next,
            'weighting': self.weighting_next,
            'systematic_noise': self.systematic_noise_next,
            'systematic_cphase_noise': self.systematic_cphase_noise_next,
            'cp_uv_min': self.cp_uv_min,
        }

    def _full_fft_params(self):
        """Bundle FFT/NFFT params into a single dict for the backend."""
        return {
            'fft_pad_factor': self._fft_pad_factor,
            'fft_conv_func': self._fft_conv_func,
            'fft_gridder_prad': self._fft_gridder_prad,
            'fft_interp_order': self._fft_interp_order,
        }

    def make_reg_dict(self, imcur):
        """Make a dictionary of current regularizer values
           input is image array transformed to bounded values
        """
        return compute_reg_dict(
            imcur, sorted(self.reg_term_next.keys()),
            self.mf_next, self.pol_next,
            self._logfreqratio_list, len(self.obslist_next),
            self._prior_arr, self.norm_reg, self._full_regparams(),
            self._embed_mask,
        )

    def make_reggrad_dict(self, imcur):
        """Make a dictionary of current regularizer gradient values
           input is image array transformed to bounded values
        """
        return compute_reggrad_dict(
            imcur, sorted(self.reg_term_next.keys()),
            self.mf_next, self.pol_next,
            self._logfreqratio_list, len(self.obslist_next),
            self._prior_arr, self.norm_reg, self._full_regparams(),
            self._embed_mask,
            self._which_solve, self._nimage,
        )

    def objfunc(self, imvec):
        """Current objective function."""
        return compute_objective(
            imvec, self._init_arr,
            self.mf_next, self.pol_next,
            self._which_solve, self._data_tuples,
            self._logfreqratio_list, len(self.obslist_next),
            self.dat_term_next, self.reg_term_next,
            self._prior_arr, self.norm_reg, self._full_regparams(),
            self.transform_next, self._embed_mask, self._ttype,
        )

    def objgrad(self, imvec):
        """Current objective function gradient."""
        return compute_objective_grad(
            imvec, self._init_arr,
            self.mf_next, self.pol_next,
            self._which_solve, self._data_tuples,
            self._logfreqratio_list, len(self.obslist_next),
            self.dat_term_next, self.reg_term_next,
            self._prior_arr, self.norm_reg, self._full_regparams(),
            self.transform_next, self._embed_mask, self._ttype, self._nimage,
        )

    def plotcur(self, imvec, **kwargs):
        """Plot current image.
        """

        if self._show_updates:
            if self._nit % self._update_interval == 0:
                # Unpack polarimetric/multifrequency vector into an array
                imcur =  unpack_imarr(imvec, self._init_arr, self._which_solve)

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


