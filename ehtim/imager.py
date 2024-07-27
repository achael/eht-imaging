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
# TODO unpack_poltuple and unpack_mftuple are basically the same thing and we should consolidate
# TODO for polarization imaging give better control than just initializing at 20% 
# TODO better initialization for all terms!! in init_imager

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
REGULARIZERS_SPECTRAL = REGULARIZERS_SPECIND + REGULARIZERS_CURV + REGULARIZERS_SPECIND_P + REGULARIZERS_CURV_P + REGULARIZERS_RM

GRIDDER_P_RAD_DEFAULT = 2
GRIDDER_CONV_FUNC_DEFAULT = 'gaussian'
FFT_PAD_DEFAULT = 2
FFT_INTERP_DEFAULT = 3

REG_DEFAULT = {'simple': 1}
DAT_DEFAULT = {'vis': 100}

POL_TRANS = True  # this means we solve for polarization in the m, chi basis
#MF_WHICH_SOLVE = (1, 1, 0)    # this means that mf imaging solves for I0 and alpha (not beta), for now
#replaced with separate mf arguments     

REGPARAMS_DEFAULT = {'major':50*ehc.RADPERUAS,
                     'minor':50*ehc.RADPERUAS,
                     'PA':0.,
                     'alpha_A':1.0,
                     'epsilon_tv':0.0}

POLARIZATION_MODES = ['P','QU','IP','IQU','V','IV','IQUV','IPV'] # TODO: treatment of V may be inconsistent

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
        self._reg_term_list = []
        self._dat_term_list = []
        self._clipfloor_list = []
        self._maxset_list = []
        self._pol_list = []
        self._maxit_list = []
        self._stop_list = []
        self._flux_list = []
        self._pflux_list = []
        self._vflux_list = []
        self._snrcut_list = []
        self._debias_list = []
        self._systematic_noise_list = []
        self._systematic_cphase_noise_list = []
        self._transform_list = []
        self._weighting_list = []

        # Regularizer/data terms for the next imaging iteration
        self.reg_term_next = reg_term   # e.g. [('simple',1), ('l1',10), ('flux',500), ('cm',500)]
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

        # set polarimetric flux values equal to Stokes I flux by default
        # used in regularizer normalization
        self.pflux_next = kwargs.get('pflux', flux)
        self.vflux_next = kwargs.get('vflux', flux)
                
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

        # Imager history
        self._change_imgr_params = True
        self.nruns = 0

        # multifrequency
        self.mf_next = False
        self.reg_all_freq_mf = kwargs.get('reg_all_freq_mf',False)
        #self.mf_which_solve = kwargs.get('mf_which_solve',MF_WHICH_SOLVE)

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
        self.reg_all_freq_mf = kwargs.get('reg_all_freq_mf', self.reg_all_freq_mf)
        #self.mf_which_solve = kwargs.get('mf_which_solve', self.mf_which_solve)

        if pol is None:
            pol_prim = self.pol_next
        else:
            self.pol_next = pol
            pol_prim = pol

        print("==============================")
        print("Imager run %i " % (int(self.nruns)+1))

        # For polarimetric imaging, switch polrep to Stokes
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

        if self.mf_next:
            # multi-frequency polarization
            if self.pol_next in POLARIZATION_MODES: # polarization       
                if self.pol_next == 'P':
                    out = mfutils.unpack_mftuple(out, self._xtuple, self._nimage, self._mf_which_solve)
                    if 'mcv' in self.transform_next:
                        polout = polutils.mcv((out[0],out[3],out[6])) # TODO make a separate mcv_mf?? 
                        out[0] = polout[0]
                        out[3] = polout[1]
                        out[6] = polout[2]                                

            # multi-frequency Stokes I 
            else:
                out = mfutils.unpack_mftuple(out, self._xtuple, self._nimage, self._mf_which_solve)
                if 'log' in self.transform_next:
                    out[0] = np.exp(out[0])  
        else: 
            # single-frequency polarization 
            if self.pol_next in POLARIZATION_MODES: s
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

                elif self.pol_next == 'V':
                    out = polutils.unpack_poltuple(out, self._xtuple, self._nimage, (0,0,0,1))
                    if 'mcv' in self.transform_next:
                        out = polutils.mcv(out)

                elif self.pol_next == 'IV':
                    out = polutils.unpack_poltuple(out, self._xtuple, self._nimage, (1,0,0,1))
                    if 'mcv' in self.transform_next:
                        out = polutils.mcv(out)
                    if 'log' in self.transform_next:
                        out[0] = np.exp(out[0])

                elif self.pol_next == 'IQUV':
                    out = polutils.unpack_poltuple(out, self._xtuple, self._nimage, (1,1,1,1))
                    if 'mcv' in self.transform_next:
                        out = polutils.mcv(out)
                    if 'log' in self.transform_next:
                        out[0] = np.exp(out[0])
                                       

            # single-frequency Stokes I 
            elif 'log' in self.transform_next:
                out = np.exp(out)

        # Print final stats
        outstr = ""
        chi2_term_dict = self.make_chisq_dict(out)
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

        # Embed image
        if self._mf_next:
            # polarization multi-frequency
            if self.pol_next in POLARIZATION_MODES:
                if np.any(np.invert(self._embed_mask)):
                    out = imutils.embed_arr(out, self._embed_mask) 
                iimage_out = out[0]
                specind_out = out[1]
                curv_out = out[2]
                pimage_out = out[3]
                pspecind_out = out[4]
                pcurv_out = out[5]
                chiimage_out = out[6]
                rm_out = out[7]

                poltuple = (iimage_out, pimage_out, chiimage_out)
                qimage_out = polutils.make_q_image(poltuple, True)
                uimage_out = polutils.make_u_image(poltuple, True)
                vimage_out = polutils.make_v_image(poltuple, True)  
            # Stokes I multi-frequency
            else:
                if np.any(np.invert(self._embed_mask)):
                    out = imutils.embed_arr(out, self._embed_mask)
                iimage_out = out[0]
                specind_out = out[1]
                curv_out = out[2]
        else:
            # single frequency polarization
            if self.pol_next in POLARIZATION_MODES:
                if np.any(np.invert(self._embed_mask)):
                    out = imutils.embed_arr(out, self._embed_mask)
                iimage_out = out[0]
                qimage_out = polutils.make_q_image(out, POL_TRANS)
                uimage_out = polutils.make_u_image(out, POL_TRANS)
                vimage_out = polutils.make_v_image(out, POL_TRANS)
            
            else: # simple single-pol
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

            # Did we solve for polarimeric image or are we copying over old polarization data?
            if self.pol_next in POLARIZATION_MODES and pol2 == 'Q':
                polvec = qimage_out
            elif self.pol_next in POLARIZATION_MODES and pol2 == 'U':
                polvec = uimage_out
            elif self.pol_next in POLARIZATION_MODES and pol2 == 'V':
                polvec = vimage_out
            else:
                polvec = self.init_next._imdict[pol2]

            if len(polvec):
                polarr = polvec.reshape(outim.ydim, outim.xdim)
                outim.add_pol_image(polarr, pol2)

        # Copy over spectral information
        outim._mflist = copy.deepcopy(self.init_next._mflist)
        if self.mf_next:
            outim._mflist[0] = specind_out
            outim._mflist[1] = curv_out
            
            if self.pol_next in POLARIZATION_MODES: # polarization multi-frequency
                outim._mflist_p[0] = pspecind_out
                outim._mflist_p[1] = pcurv_out
                outim._mflist_chi[0] = rm_out
            
        # Append to history
        logstr = str(self.nruns) + ": make_image(pol=%s)" % pol
        self._append_image_history(outim, logstr)
        self.nruns += 1

        # Return Image object
        return outim

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

        # TODO single-polarization imaging. should we still support? 
        if ('log' in self.transform_next and self.pol_next in ['Q', 'U', 'V']):
            raise Exception("Cannot image Stokes Q, U, V with log image transformation!")

        if(self.pol_next in ['Q', 'U', 'V'] and
           ('gs' in self.reg_term_next.keys() or 'simple' in self.reg_term_next.keys())):
            raise Exception(
                "'simple' and 'gs' methods do not work with Stokes Q, U, or V images!")

        if self._ttype not in ['fast', 'direct', 'nfft']:
            raise Exception("Possible ttype values are 'fast', 'direct','nfft'!")
            
        # Catch errors in multifrequency imaging setup
        if self.mf_next and len(set(self.freq_list)) < 2:
            raise Exception(
                "must have observations at at least two frequencies for multifrequency imaging!")

        # Catch errors for polarimetric imaging setup
        if self.pol_next in POLARIZATION_MODES:
            if 'mcv' not in self.transform_next:
                raise Exception("Polarimetric imaging needs 'mcv' transform!")
            if (self._ttype not in ["direct", "nfft"]):
                raise Exception("FFT not yet implemented in polarimetric imaging -- use NFFT!")
            if 'I' in self.pol_next:
                rlist = REGULARIZERS + REGULARIZERS_POL
                dlist = DATATERMS + DATATERMS_POL
            else:
                rlist = REGULARIZERS_POL
                dlist = DATATERMS_POL
        else:
            rlist = REGULARIZERS + REGULARIZERS_SPECIND + REGULARIZERS_CURV
            dlist = DATATERMS    
 
        # catch errors in general imaging setup
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
        
        # Set prior & initial image vectors for multifrequency imaging
        elif self.mf_next:

            # set reference frequency to same as prior
            self.reffreq = self.init_next.rf 
            
            # reset logfreqratios in case the reference frequency changed
            self._logfreqratio_list = [np.log(nu/self.reffreq) for nu in self.freq_list]

            # set initial and prior images
            nprior_I = self.prior_next.imvec[self._embed_mask]
            ninit_I = self.init_next.imvec[self._embed_mask]
            self._nimage = len(ninit_I)
                            
            if self.norm_init:
                nprior_I = self.flux_next * nprior_I / (np.sum(nprior_I))
                ninit_I = self.flux_next * ninit_I / (np.sum(nprior_I))                

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


            # polarization multi-frequency
            if self.pol_next in POLARIZATION_MODES:
                if self.pol_next=='P':
                    if (len(self.init_next.qvec) and (np.any(self.init_next.qvec != 0) or np.any(self.init_next.uvec != 0))):
                        ninit_p = (np.abs(self.init_next.qvec + 1j*self.init_next.uvec) / self.init_next.imvec)[self._embed_mask]
                        ninit_chi = (np.arctan2(self.init_next.uvec, self.init_next.qvec) / 2.0)[self._embed_mask]

                    else:
                        # !AC TODO better initialization
                        print("No polarimetric image in init_next!")
                        print("--initializing with 20% pol and random orientation!")
                        ninit_p = 0.2 * (np.ones(self._nimage) + 1e-2 * np.random.rand(self._nimage))
                        ninit_chi = np.zeros(self._nimage) + 1e-2 * np.random.rand(self._nimage)
                    
                    if len(self.init_next.specvec_p):
                        ninit_ap = self.init_next.specvec_p[self._embed_mask]
                    else:
                        ninit_ap = np.zeros(self._nimage)[self._embed_mask]  
                    if len(self.prior_next.specvec_p):
                        nprior_ap = self.prior_next.specvec_p[self._embed_mask]
                    else:
                        nprior_ap = np.zeros(self._nimage)[self._embed_mask]                        

                    if len(self.init_next.curvvec_p):
                        ninit_bp = self.init_next.curvvec_p[self._embed_mask]
                    else:
                        ninit_bp = np.zeros(self._nimage)[self._embed_mask]
                    if len(self.prior_next.curvvec_p):
                        nprior_bp = self.init_next.curvvec_p[self._embed_mask]
                    else:
                        nprior_bp = np.zeros(self._nimage)[self._embed_mask]

                    # TODO what do we want to initialize RM to? 
                    if len(self.init_next.rmvec):
                        ninit_rm = self.init_next.rmvec[self._embed_mask]
                    else:
                        ninit_rm = np.zeros(self._nimage)[self._embed_mask]
                    if len(self.prior_next.rmvec):
                        nprior_rm = self.init_next.rmvec[self._embed_mask]
                    else:
                        nprior_rm = np.zeros(self._nimage)[self._embed_mask]

                    # prior
                    self._nprior = np.array((nprior_I, nprior_a, nprior_b, nprior_p, nprior_ap, nprior_bp, nprior_chi, nprior_rm))

                    # initial image w/ change of variables
                    if 'mcv' in self.transform_next:
                        (ninit_I, ninit_p, ninit_chi) = polutils.mcv_r((ninit_I, ninit_p, ninit_chi))

                    self._xtuple = np.array((ninit_I, ninit_a, ninit_b, ninit_p, ninit_ap, ninit_bp, ninit_chi, ninit_rm))      
                    
                    # determine mf_which_solve
                    # TODO is there a nicer way to do this? 
                    if self.mf_order_p == 0:
                        do_ap=0; do_bp=0     
                    elif self.mf_order_p == 1:   
                        do_ap=1; do_bp=0  
                    elif self.mf_order_p == 2
                        do_ap=1; do_bp=1
                    else:
                        raise Exception("Imager.mf_order_p must be 0, 1, or 2!")
                    
                    if self.mf_order_chi == 0:
                        do_rm = 0
                    elif self.mf_order_chi == 1:
                        do_rm = 1 
                    else:
                        raise Exception("Imager.mf_order_rm must be 0 or 1!")
                        
                    self._mf_which_solve = (0,0,0,1,do_ap,do_bp,1,do_rm)
                    
                else:
                    raise Exception("only P imaging supported for multifreqency polarzation for now!")       
                        
            # Stokes I multi-frequency
            else:
                # prior
                self._nprior = np.array((nprior_I, nprior_a, nprior_b))

                # initial image w/ change of variables
                if 'log' in self.transform_next:
                    ninit_I = np.log(ninit_I)

                self._xtuple = np.array((ninit_I, ninit_a, ninit_b))

                # Determine mf_which_solve
                if self.mf_order==2:
                    self._mf_which_solve = (1, 1, 0)
                elif self.mf_order==1:
                    self._mf_which_solve = (1, 1, 1)
                else:
                    raise Exception("Imager.mf_order must be 1 or 2!")
                    
            # Pack multi-frequency tuple into single vector
            self._xinit = mfutils.pack_mftuple(self._xtuple, self._mf_which_solve)
            
         
        # Set prior & initial image vectors for single-frequency imaging
        else:
            # single-frequency polarimetric imaging
            if self.pol_next in POLARIZATION_MODES:

                # initial I image
                self._nprior = self.prior_next.imvec[self._embed_mask]
                initI = self.init_next.imvec[self._embed_mask]
                if self.norm_init and ('I' in self.pol_next):
                    self._nprior = self.flux_next * self._nprior / np.sum(self._nprior)
                    initI = self.flux_next * initI / np.sum(initI)
                    
                self._nimage = len(initI)

                # Initialize m & chi & v
                if (len(self.init_next.qvec) and (np.any(self.init_next.qvec != 0) or np.any(self.init_next.uvec != 0))):
                    initp = (np.abs(self.init_next.qvec + 1j*self.init_next.uvec) / self.init_next.imvec)[self._embed_mask]
                    initchi = (np.arctan2(self.init_next.uvec, self.init_next.qvec) / 2.0)[self._embed_mask]

                else:
                    # !AC TODO get the actual zero baseline polarization fraction from the data?
                    print("No polarimetric image in init_next!")
                    print("--initializing with 20% pol and random orientation!")
                    initp = 0.2 * (np.ones(self._nimage) + 1e-2 * np.random.rand(self._nimage))
                    initchi = np.zeros(self._nimage) + 1e-2 * np.random.rand(self._nimage)
                
                # Initialize v if requested
                if 'V' in self.pol_next:
                    if len(self.init_next.vvec) and (np.any(self.init_next.vvec != 0)):
                        initv = (self.init_next.vvec / self.init_next.imvec)[self._embed_mask]
                    else:
                        # !AC TODO get the actual zero baseline polarization fraction from the data?
                        print("No V polarimetric image in init_next!")
                        print("--initializing with random vector")
                        initv = 0.01 * (np.ones(self._nimage) + 1e-2 * np.random.rand(self._nimage))
                    inittuple = np.array((initI, initp, initchi, initv))                    
                else:                    
                    inittuple = np.array((initI, initp, initchi))

                # Change of variables
                if 'mcv' in self.transform_next:
                    self._xtuple = polutils.mcv_r(inittuple)
                else:
                    raise Exception("Polarimetric imaging only works with mcv transform!")

                # Only apply log transformation to Stokes I if doing simultaneous imaging
                if ('log' in self.transform_next) and ('I' in self.pol_next):
                    self._xtuple[0] = np.log(self._xtuple[0])

                # Determine pol_which_solve
                if self.pol_next in ['P','QU']:
                    self._pol_which_solve = (0,1,1) 
                elif self.pol_next in ['IP','IQU']:
                    self._pol_which_solve = (1,1,1)           
                elif self.pol_next in ['V']:
                    self._pol_which_solve = (0,0,0,1)                       
                elif self.pol_next in ['IV']:
                    self._pol_which_solve = (1,0,0,1)                                   
                elif self.pol_next in ['IQUV']:
                    self._pol_which_solve = (1,1,1,1)                                   
                else: 
                    raise Exception("Do not know correct pol_which_solve for self.pol_next=%s!"%self.pol_next)
                
                # Pack into single vector
                self._xinit = polutils.pack_poltuple(self._xtuple, self._pol_which_solve)

            # regular single-frequency single-stokes (or RR, LL) imaging
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


    def make_chisq_dict(self, imcur):
        """Make a dictionary of current chi^2 term values
           i indexes the observation number in self.obslist_next
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
                         
                # get chi^2 term
                if dname in DATATERMS_POL:
                    chi2 = polutils.polchisq(imcur_nu, A, data, sigma, dname,
                                             ttype=self._ttype, mask=self._embed_mask,
                                             pol_trans=POL_TRANS)

                elif dname in DATATERMS:
                    if self.pol_next in POLARIZATION_MODES: # polarization
                        imcur_nu = imcur_nu[0]

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
                    chi2grad = polutils.polchisqgrad(imcur_nu, A, data, sigma, dname,
                                                     ttype=self._ttype, mask=self._embed_mask,
                                                     pol_solve=self._pol_which_solve,
                                                     pol_trans=POL_TRANS)

                # Single polarization chi^2 gradients
                elif dname in DATATERMS:
                    if self.pol_next in POLARIZATION_MODES: # polarization
                        imcur_nu = imcur_nu[0]
                        
                    chi2grad = imutils.chisqgrad(imcur_nu, A, data, sigma, dname,
                                                 ttype=self._ttype, mask=self._embed_mask)

                    # If imaging polarization simultaneously, bundle the gradient properly
                    # TODO need a more flexible approach here
                    if self.pol_next in POLARIZATION_MODES: 
                        if 'V' in self.pol_next:
                            chi2grad = np.array((chi2grad, np.zeros(self._nimage), np.zeros(self._nimage), np.zeros(self._nimage)))                        
                        else:
                            chi2grad = np.array((chi2grad, np.zeros(self._nimage), np.zeros(self._nimage)))

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
        """
        reg_dict = {}
                           
        for regname in sorted(self.reg_term_next.keys()):
        
            # Multifrequency regularizers
            if self.mf_next:
            
                # Polarimetric regularizers
                if regname in REGULARIZERS_POL:
                    # option to regularize ALL the images in multifrequency imaging
                    # TODO total fluxes not right? 
                    if self.reg_all_freq_mf:
                        for i in range(len(self.obslist_next)):
                            regname_key = regname + ('_%i' % i)
                            logfreqratio = self._logfreqratio_list[i]
                            imcur_nu = mfutils.image_at_freq(imcur, logfreqratio)
                            reg = polutils.polregularizer(imcur_nu, self._embed_mask, 
                                                          self.flux_next, self.pflux_next, self.vflux_next,
                                                          self.prior_next.xdim, self.prior_next.ydim, self.prior_next.psize, 
                                                          regname,
                                                          norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                          pol_trans=POL_TRANS)                            
                            reg_dict[regname_key] = reg                                
                    
                    # normally we only regularize reference frequency image
                    else:
                        imcur_pol = (imcur[0], imcur[3], imcur[6]) # TODO be more careful
                        reg = polutils.polregularizer(imcur_pol, self._embed_mask, 
                                                      self.flux_next, self.pflux_next, self.vflux_next,
                                                      self.prior_next.xdim, self.prior_next.ydim, self.prior_next.psize, 
                                                      regname,
                                                      norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                      pol_trans=POL_TRANS)                            
                # Stokes I regularizers
                elif regname in REGULARIZERS:         
                    # option to regularize ALL the images in multifrequency imaging
                    # TODO total fluxes not right? 
                    if self.reg_all_freq_mf:
                        for i in range(len(self.obslist_next)):
                            regname_key = regname + ('_%i' % i)
                            logfreqratio = self._logfreqratio_list[i]
                            imcur_nu = mfutils.image_at_freq(imcur, logfreqratio)
                            prior_nu = mfutils.image_at_freq(self._nprior, logfreqratio)
                            if len(imcur==8): # multifrequency polarizaiton 
                                imcur_nu = imcur_nu[0]
                                prior_nu = prior_nu[0]                            
                            reg = imutils.regularizer(imcur_nu, prior_nu, self._embed_mask,
                                                      self.flux_next, self.prior_next.xdim,
                                                      self.prior_next.ydim, self.prior_next.psize,
                                                      regname,
                                                      norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                      **self.regparams)
                            reg_dict[regname_key] = reg                                
                    
                    # normally we only regularize reference frequency image
                    else:
                        reg = imutils.regularizer(imcur[0], self._nprior[0], self._embed_mask,
                                                  self.flux_next, self.prior_next.xdim,
                                                  self.prior_next.ydim, self.prior_next.psize,
                                                  regname,
                                                  norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                  **self.regparams)
                
                # Spectral regularizers
                if regname in REGULARIZERS_SPECTRAL:
                    if regname in REGULARIZERS_SPECIND:
                        idx = 1
                    elif regname in REGULARIZERS_CURV:
                        idx = 2
                    elif regname in REGULARIZERS_SPECIND_P:
                        idx = 4
                    elif regname in REGULARIZERS_CURV_P:
                        idx = 5
                    elif regname in REGULARIZERS_RM:
                        idx = 7
                    
                    reg = mfutils.regularizer_mf(imcur[idx], self._nprior[idx], self._embed_mask,
                                                 self.prior_next.xdim, self.prior_next.ydim, self.prior_next.psize,
                                                 regname,
                                                 norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                 **self.regparams)
                else:
                    raise Exception("regularizer term %s not recognized!" % regname)                            

            # Single-frequency polarimetric regularizer
            elif regname in REGULARIZERS_POL:
                reg = polutils.polregularizer(imcur, self._embed_mask, 
                                              self.flux_next, self.pflux_next, self.vflux_next,
                                              self.prior_next.xdim, self.prior_next.ydim, self.prior_next.psize, 
                                              regname,
                                              norm_reg=self.norm_reg, beam_size=self.beam_size,
                                              pol_trans=POL_TRANS)

            # Single-frequency, single-polarization regularizer                                              
            elif regname in REGULARIZERS:
                if self.pol_next in POLARIZATION_MODES:
                    imcur0 = imcur[0]
                else:
                    imcur0 = imcur

                reg = imutils.regularizer(imcur0, self._nprior, self._embed_mask,
                                          self.flux_next, 
                                          self.prior_next.xdim, self.prior_next.ydim, self.prior_next.psize,
                                          regname,
                                          norm_reg=self.norm_reg, beam_size=self.beam_size,
                                          **self.regparams)
            else:
                raise Exception("regularizer term %s not recognized!" % regname)

            # put regularizer terms in the dictionary
            # if we regularize all images with self.reg_all_freq_mf they are already there          
            if not(self.mf_next and self.reg_all_freq_mf and (regname in REGULARIZERS or regname in REGULARIZERS_POL)): 
                reg_dict[regname] = reg

        return reg_dict

    def make_reggrad_dict(self, imcur):
        """Make a dictionary of current regularizer gradient values
        """
                
        reggrad_dict = {}
        
                    
        for regname in sorted(self.reg_term_next.keys()):
            reggradzero = np.zeros((len(imcur), self._nimage))
            
            # Multifrequency regularizers 
            if self.mf_next:
                # Polarimetric regularizers
                if regname in REGULARIZERS_POL:
                    # option to regularize ALL the images in multifrequency imaging
                    # TODO total fluxes not right? 
                    if self.reg_all_freq_mf:
                        for i in range(len(self.obslist_next)):
                            regname_key = regname + ('_%i' % i)
                            logfreqratio = self._logfreqratio_list[i]
                            imcur_nu = mfutils.image_at_freq(imcur, logfreqratio)
                            reg = polutils.polregularizergrad(imcur_nu, self._embed_mask, 
                                                              self.flux_next, self.pflux_next, self.vflux_next,
                                                              self.prior_next.xdim, self.prior_next.ydim, self.prior_next.psize, 
                                                              regname,
                                                              norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                              pol_solve=self._pol_which_solve,
                                                              pol_trans=POL_TRANS)                            
                            reg = mfutils.mf_all_grads_chain(reg, imcur_nu, imcur, logfreqratio)
                            reg_dict[regname_key] = reg   
                            
                    # normally we only regularize reference frequency image
                    else:
                        imcur_pol = (imcur[0], imcur[3], imcur[6]) # TODO be more careful with indices
                        regp = polutils.polregularizergrad(imcur_pol, self._embed_mask, 
                                                          self.flux_next, self.pflux_next, self.vflux_next,
                                                          self.prior_next.xdim, self.prior_next.ydim, self.prior_next.psize, 
                                                          regname,
                                                          norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                          pol_solve=self._pol_which_solve,
                                                          pol_trans=POL_TRANS)
                        reg = np.zeros((len(imcur), self._nimage))
                        reg[0] = reg[0]
                        reg[3] = reg[1]
                        reg[6] = reg[2]

                # Stokes I regularizers
                elif regname in REGULARIZERS:         
                    # option to regularize ALL the images in multifrequency imaging
                    # TODO total fluxes not right? 
                    if self.reg_all_freq_mf:
                        for i in range(len(self.obslist_next)):
                            regname_key = regname + ('_%i' % i)
                            logfreqratio = self._logfreqratio_list[i]
                            imcur_nu = mfutils.image_at_freq(imcur, logfreqratio)
                            prior_nu = mfutils.image_at_freq(self._nprior, logfreqratio)
                            if len(imcur==8): # multifrequency polarizaiton 
                                imcur_nu = imcur_nu[0]
                                prior_nu = prior_nu[0]
                            reg = imutils.regularizergrad(imcur_nu, prior_nu, 
                                                          self._embed_mask, self.flux_next, 
                                                          self.prior_next.xdim, self.prior_next.ydim, self.prior_next.psize,
                                                          regname,
                                                          norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                          **self.regparams)
                            reg = mfutils.mf_all_grads_chain(reg, imcur_nu, imcur, logfreqratio)
                            reg_dict[regname_key] = reg                                

                    # normally we only regularize reference frequency image
                    else:
                        regi = imutils.regularizergrad(imcur[0], self._nprior[0], 
                                                       self._embed_mask, self.flux_next, 
                                                       self.prior_next.xdim, self.prior_next.ydim, self.prior_next.psize,
                                                       regname,
                                                       norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                       **self.regparams)
                        reg = np.zeros((len(imcur), self._nimage))
                        reg[0] = regi  
                                   
                # Spectral regularizers
                if regname in REGULARIZERS_SPECTRAL:
                    if regname in REGULARIZERS_SPECIND:
                        idx = 1
                    elif regname in REGULARIZERS_CURV:
                        idx = 2
                    elif regname in REGULARIZERS_SPECIND_P:
                        idx = 4
                    elif regname in REGULARIZERS_CURV_P:
                        idx = 5
                    elif regname in REGULARIZERS_RM:
                        idx = 7
                    
                    regmf = mfutils.regularizergrad_mf(imcur[idx], self_nprior[idx], self._embed_mask,
                                                       self.prior_next.xdim, self.prior_next.ydim, self.prior_next.psize,
                                                       regname,
                                                       norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                       **self.regparams)
                    
                    reg = np.zeros((len(imcur), self._nimage))
                    reg[idx] = regmf  
                else:
                    raise Exception("regularizer term %s not recognized!" % regname)                            
            

                    
            else:        
                # Single-frequency polarimetric regularizer
                if regname in REGULARIZERS_POL:
                    reg = polutils.polregularizergrad(imcur, self._embed_mask, 
                                                      self.flux_next, self.pflux_next, self.vflux_next,
                                                      self.prior_next.xdim, self.prior_next.ydim,
                                                      self.prior_next.psize, regname,
                                                      norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                      pol_solve=self._pol_which_solve,
                                                      pol_trans=POL_TRANS)


                # Single-frequency, single polarization regularizer
                elif regname in REGULARIZERS:
                    if self.pol_next in POLARIZATION_MODES:
                        imcur0 = imcur[0]
                    else:
                        imcur0 = imcur
                    reg = imutils.regularizergrad(imcur0, self._nprior, self._embed_mask, self.flux_next,
                                                  self.prior_next.xdim, self.prior_next.ydim,
                                                  self.prior_next.psize,
                                                  regname,
                                                  norm_reg=self.norm_reg, beam_size=self.beam_size,
                                                  **self.regparams)
                    if self.pol_next in POLARIZATION_MODES:
                        if 'V' in self.pol_next:
                            reg = np.array((reg, np.zeros(self._nimage), np.zeros(self._nimage), np.zeros(self._nimage)))                    
                        else:
                            reg = np.array((reg, np.zeros(self._nimage), np.zeros(self._nimage)))
                else:
                    raise Exception("regularizer term %s not recognized!" % regname)

            # put regularizer terms in the dictionary
            # if we regularize all images with self.reg_all_freq_mf they are already there          
            if not(self.mf_next and self.reg_all_freq_mf and (regname in REGULARIZERS or regname in REGULARIZERS_POL)): 
                reggrad_dict[regname] = reg

        return reggrad_dict

############################################################################################### PINPINPIN######################
    def objfunc(self, imvec):
        """Current objective function.
        """

        # Unpack polarimetric/multifrequency vector into an array
        # TODO these unpack functions are basically the same, should consolidate
        if self.mf_next:
            imcur = mfutils.unpack_mftuple(imvec, self._xtuple, self._nimage, self._mf_which_solve)
        elif  self.pol_next in POLARIZATION_MODES:
            imcur = polutils.unpack_poltuple(imvec, self._xtuple, self._nimage, self._pol_which_solve)        
        else:
            imcur = imvec

        # Image change of variables
        if self.pol_next in POLARIZATION_MODES and 'mcv' in self.transform_next:
            imcur = polutils.mcv(imcur)

        if 'log' in self.transform_next:
            if self.pol_next in POLARIZATION_MODES:
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
            # multifrequency imaging, regularize every frequency
            if self.mf_next and self.reg_all_freq_mf and (regname in REGULARIZERS):
                for i in range(len(self.obslist_next)):
                    regname_key = regname + ('_%i' % i)
                    regularizer = reg_term_dict[regname_key]        
                    regterm += hyperparameter * regularizer
                    
            # but normally just one regularizer                    
            else:        
                regularizer = reg_term_dict[regname]
                regterm += hyperparameter * regularizer

        # Total cost
        cost = datterm + regterm
        
        return cost

    def objgrad(self, imvec):
        """Current objective function gradient.
        """

        # Unpack polarimetric/multifrequency vector into an array
        if self.pol_next in POLARIZATION_MODES:
            imcur = polutils.unpack_poltuple(imvec, self._xtuple, self._nimage, self._pol_which_solve)
        elif self.mf_next:
            imcur = mfutils.unpack_mftuple(imvec, self._xtuple, self._nimage, self._mf_which_solve)
        else:
            imcur = imvec

        # Image change of variables
        if 'mcv' in self.transform_next:
            if self.pol_next in POLARIZATION_MODES:
                cvcur = imcur.copy()
                imcur = polutils.mcv(imcur)

        if 'log' in self.transform_next:
            if self.pol_next in POLARIZATION_MODES:
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
            
            # multifrequency imaging, regularize every frequency
            if self.mf_next and self.reg_all_freq_mf and (regname in REGULARIZERS):
                for i in range(len(self.obslist_next)):
                    regname_key = regname + ('_%i' % i)
                    regularizer = reg_term_dict[regname_key]        
                    regterm += hyperparameter * regularizer
                    
            # but normally just one regularizer
            else:                    
                regularizer_grad = reg_term_dict[regname]
                regterm += hyperparameter * regularizer_grad

        # Total gradient
        grad = datterm + regterm

        # Chain rule term for change of variables
        if 'mcv' in self.transform_next:
            if self.pol_next in POLARIZATION_MODES:
                grad *= polutils.mchain(cvcur)
                
        if 'log' in self.transform_next:
            if self.pol_next in POLARIZATION_MODES:
                grad[0] *= imcur[0]
            elif self.mf_next:
                grad[0] *= imcur[0]
            else:
                grad *= imcur

        # Repack gradient for polarimetric imaging
        if self.pol_next in POLARIZATION_MODES:
            grad = polutils.pack_poltuple(grad, self._pol_which_solve)

        # repack gradient for multifrequency imaging
        elif self.mf_next:
            grad = mfutils.pack_mftuple(grad, self._mf_which_solve)

        return grad

    def plotcur(self, imvec, **kwargs):
        """Plot current image.
        """

        if self._show_updates:
            if self._nit % self._update_interval == 0:
                if self.pol_next in POLARIZATION_MODES:

                    imcur = polutils.unpack_poltuple(imvec, self._xtuple, self._nimage, self._pol_which_solve)
                elif self.mf_next:
                    imcur = mfutils.unpack_mftuple(
                        imvec, self._xtuple, self._nimage, self._mf_which_solve)
                else:
                    imcur = imvec

                # Image change of variables

                if 'mcv' in self.transform_next:
                    if self.pol_next in POLARIZATION_MODES:
                        imcur = polutils.mcv(imcur)

                if 'log' in self.transform_next:
                    if self.pol_next in POLARIZATION_MODES:
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
                
                    if self.mf_next and self.reg_all_freq_mf and (regname in REGULARIZERS):
                        for i in range(len(self.obslist_next)):
                            regname_key = regname + ('_%i' % i)
                            rval = reg_term_dict[regname_key]*self.reg_term_next[regname]
                            outstr += "%s : %0.1f " % (regname_key, rval)
                    else:
                        rval = reg_term_dict[regname]*self.reg_term_next[regname]
                        outstr += "%s : %0.1f " % (regname, rval)

                # Embed and plot the image
                if self.pol_next in POLARIZATION_MODES:
                    if np.any(np.invert(self._embed_mask)):
                        imcur = imutils.embed_arr(imcur, self._embed_mask)
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
        self._transform_list.append(self.transform_next)
        self._reg_term_list.append(self.reg_term_next)
        self._dat_term_list.append(self.dat_term_next)

        self._out_list.append(outim)
        return
