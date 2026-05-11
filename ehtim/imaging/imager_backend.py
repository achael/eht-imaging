# imager_backend.py
# Pure functional backend for imager.py
# Extracted from imager.py; zero functional changes.
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

from typing import NamedTuple

import numpy as np

import ehtim.imaging.imager_utils as imutils
import ehtim.imaging.multifreq_imager_utils as mfutils
import ehtim.imaging.pol_imager_utils as polutils

# -----------------------------------------------------------------------------
# Naming convention for image arguments throughout this module
# -----------------------------------------------------------------------------
# Functions here distinguish between the *initial* image and the *prior* image:
#   initvec / init_arr : the initial-value image used to fill not-solved-for
#                        slots in unpack_imarr (the L-BFGS-B start point).
#   priorvec           : the regularizer prior image, referenced by terms like
#                        'simple', 'l1', 'rgauss'. Can differ from the initial
#                        image entirely.
# Suffix conventions:
#   *_arr / init_arr   : multi-D unwrapped array (one row per Stokes/freq term).
#   *_vec / initvec    : same content as *_arr but conceptually a vector
#                        in the function signature (kept for Imager-side
#                        readability; shape is identical).
#   imvec              : the 1D packed solver vector (just the solved-for slots).
# Imager-side equivalents: self._init_arr, self._prior_arr, self._init_vec.
# -----------------------------------------------------------------------------

# Data term and polarization-mode names recognized by the chi^2 dispatchers.
# Imported by ehtim.imager for backward compatibility.
DATATERMS = ['vis', 'bs', 'amp', 'cphase', 'cphase_diag', 'camp', 'logcamp', 'logcamp_diag']
DATATERMS_POL = ['pvis', 'm', 'vvis']
POLARIZATION_MODES = ['P', 'QU', 'IP', 'IQU', 'V', 'IV', 'IQUV', 'IPV']  # TODO: treatment of V may be inconsistent

# Regularizer term names recognized by the regularizer dispatchers.
# Imported by ehtim.imager for backward compatibility.
REGULARIZERS = ['gs', 'tv', 'tvlog', 'tv2', 'tv2log', 'l1', 'l1w', 'lA', 'patch',
                'flux', 'cm', 'simple', 'compact', 'compact2', 'rgauss']
REGULARIZERS_POL = ['msimple', 'hw', 'ptv', 'l1v', 'l2v', 'vtv', 'vtv2', 'vflux']

REGULARIZERS_ALLFREQS_I = ['flux_mf']
REGULARIZERS += REGULARIZERS_ALLFREQS_I

REGULARIZERS_SPECIND = ['l2_alpha', 'tv_alpha']
REGULARIZERS_CURV = ['l2_beta', 'tv_beta']
REGULARIZERS_SPECIND_P = ['l2_alphap', 'tv_alphap']
REGULARIZERS_CURV_P = ['l2_betap', 'tv_betap']
REGULARIZERS_RM = ['l2_rm', 'tv_rm']
REGULARIZERS_CM = ['l2_cm', 'tv_cm']
REGULARIZERS_ISPECTRAL = REGULARIZERS_SPECIND + REGULARIZERS_CURV
REGULARIZERS_POLSPECTRAL = REGULARIZERS_SPECIND_P + REGULARIZERS_CURV_P + REGULARIZERS_RM + REGULARIZERS_CM
REGULARIZERS_SPECTRAL = REGULARIZERS_ISPECTRAL + REGULARIZERS_POLSPECTRAL

# Default initial-polarization parameters used when init image has no Q/U/V.
MEANPOL_INIT = 0.2     # mean polarization fraction
SIGMAPOL_INIT = 1.e-2  # perturbation scale


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


def unpack_imarr(vec, init_arr, which_solve):
    """Unpack minimized 1D vector `vec` into a multi-D array.

    For each Stokes / spectral slot k:
      - if which_solve[k] == 1, take the next nimage values from `vec`;
      - if which_solve[k] == 0, fall back to `init_arr[k]` (the *initial*
        image, NOT a regularizer prior).
    """

    imarrdim = len(init_arr.shape)
    if imarrdim==2:
        nsolve = init_arr.shape[0]
        nimage = init_arr.shape[1]
    elif imarrdim==1:
        nsolve = 1
        nimage = init_arr.shape[0]
        imarr = init_arr.reshape((nsolve,nimage))
    else:
        raise Exception("in unpack_imarr, init_arr should have one or two dimensions !")

    if nsolve != len(which_solve):
        raise Exception("in unpack_imarr, init_arr has inconsistent shape with which_solve!")

    imct = 0
    imarr = np.empty((nsolve, nimage))
    for kk in range(nsolve):
        if which_solve[kk]==0:
            imarr[kk] = init_arr[kk]
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
        outarr[0] = np.log(outarr[0])
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

        # Polarimetric chain rule. mcv (m-only) and vcv (V-only) are legacy
        # single-axis parametrizations of the polarization sphere; polcv
        # parametrizes (rho, psi) jointly and is the preferred form for full
        # multi-Stokes imaging going forward. See polutils.{polcv,mcv,vcv}_grad
        # docstrings for the per-transform Jacobian.
        # Each *_grad returns shape (3, ...) so outarr[0] (the log term above) survives.
        if (pol_which_solve[1]==1 and pol_which_solve[3]==1 and ('polcv' in transforms)):
            outarr[1:4] = polutils.polcv_grad(imarr[0:4], gradarr[0:4])
        elif (pol_which_solve[1]==1) and ('mcv' in transforms):
            outarr[1:4] = polutils.mcv_grad(imarr[0:4], gradarr[0:4])
        elif (pol_which_solve[3]==1) and ('vcv' in transforms):
            outarr[1:4] = polutils.vcv_grad(imarr[0:4], gradarr[0:4])

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
        normfac = flux / (np.sum(init_I))
        init_I = normfac * init_I
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

        # Caller (compute_init_state) decides when random-pol init applies.
        # Here we just honor the flag: True means "use random pol initialization
        # regardless of init image content"; False means "use init image's pol".
        if randompol_lin:
            print("Initializing linear polarization with 20% pol and random orientation!")
            init_rho = meanpol * (np.ones(nimage) + sigmapol * np.random.rand(nimage))
            init_phi = np.zeros(nimage) + sigmapol * np.random.rand(nimage)

        if randompol_circ:
            print("Initializing circular polarization with random values!")
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
                init_bp = image.curvvec_pol[mask]
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


def compute_logfreqratios(freq_list, reffreq):
    """Log-frequency ratios for multi-frequency imaging.

    Parameters
    ----------
    freq_list : list of float
        Reference frequencies (Hz) of each observation in the obslist.
    reffreq : float
        Reference frequency (Hz) of the multi-frequency image expansion.

    Returns
    -------
    list of float
        log(nu_i / reffreq) for each nu_i in freq_list.
    """
    return [np.log(nu / reffreq) for nu in freq_list]


def compute_which_solve(pol, mf,
                        mf_order=0, mf_order_pol=0,
                        mf_rm=False, mf_cm=False):
    """Build the which-solve flag array indicating which Stokes / spectral
    parameter slots are optimized vs held fixed.

    Layout of the returned array:
        single-freq Stokes I:     [1]
        single-freq polarimetric: [I, rho, phi, psi]
        multi-freq Stokes I:      [1, alpha, beta]
        multi-freq polarimetric:  [I, rho, phi, psi, alpha, beta,
                                   alpha_p, beta_p, RM, CM]

    Parameters
    ----------
    pol : str
        Polarization mode (e.g. 'I', 'IP', 'IV'). Polarimetric modes are those
        listed in POLARIZATION_MODES.
    mf : bool
        Multi-frequency imaging flag.
    mf_order : {0, 1, 2}, optional
        Stokes-I spectral expansion order. 0 -> none, 1 -> alpha only,
        2 -> alpha + beta. Only used when mf=True.
    mf_order_pol : {0, 1, 2}, optional
        Polarimetric spectral expansion order (alpha_p, beta_p). Only used
        when mf=True and pol is polarimetric.
    mf_rm : bool, optional
        Solve for rotation measure (RM). Only used when mf=True and pol is
        polarimetric.
    mf_cm : bool, optional
        Solve for conversion measure (CM). Only used when mf=True and pol is
        polarimetric.

    Returns
    -------
    np.ndarray of int (0/1)
    """
    is_pol = pol in POLARIZATION_MODES

    if mf:
        if mf_order == 2:
            do_a, do_b = 1, 1
        elif mf_order == 1:
            do_a, do_b = 1, 0
        elif mf_order == 0:
            do_a, do_b = 0, 0
        else:
            raise Exception("Imager.mf_order must be 0, 1, or 2!")

        if is_pol:
            if mf_order_pol == 2:
                do_ap, do_bp = 1, 1
            elif mf_order_pol == 1:
                do_ap, do_bp = 1, 0
            elif mf_order_pol == 0:
                do_ap, do_bp = 0, 0
            else:
                raise Exception("Imager.mf_order_pol must be 0, 1, or 2!")

            do_rm = 1 if mf_rm else 0
            do_cm = 1 if mf_cm else 0
            do_i = 1 if 'I' in pol else 0

            # TODO: No Stokes V imaging for multifrequency yet
            do_rho = 1
            do_phi = 1
            do_psi = 0
            if not (('P' in pol) or ('QU' in pol)):
                raise Exception("Multifrequency polarization imaging currently requires pol_next=P!")
            if 'V' in pol:
                raise Exception("Stokes V not yet implemented in multifrequency polarization imaging!")

            return np.array([do_i, do_rho, do_phi, do_psi,
                             do_a, do_b, do_ap, do_bp,
                             do_rm, do_cm])

        return np.array([1, do_a, do_b])

    # single frequency
    if is_pol:
        do_i = 1 if 'I' in pol else 0
        if ('P' in pol) or ('QU' in pol):
            do_rho = 1
            do_phi = 1
        else:
            do_rho = 0
            do_phi = 0
        do_psi = 1 if 'V' in pol else 0
        return np.array([do_i, do_rho, do_phi, do_psi])

    return np.array([1])


class ImagerInitState(NamedTuple):
    """Solver-ready state produced by compute_init_state.

    Each field corresponds to an Imager._* attribute consumed by
    compute_objective / compute_objective_grad.
    """
    init_arr: np.ndarray         # multi-D, solver space
    init_vec: np.ndarray         # 1D packed solver vector
    prior_arr: np.ndarray        # multi-D, physical space
    data_tuples: dict            # keyed by dname or f"{dname}_{i}"
    embed_mask: np.ndarray       # boolean
    coord_matrix: np.ndarray     # pixel-coord companion to embed_mask
    logfreqratio_list: list      # log(nu_i / reffreq)
    nimage: int                  # number of active pixels = sum(embed_mask)
    which_solve: np.ndarray      # int 0/1 flags
    reffreq: float               # may be re-bound to init.rf when mf=True


def compute_data_tuples(obslist, prior, embed_mask, dat_term_keys, pol,
                        maxset, debias, snrcut, weighting,
                        systematic_noise, systematic_cphase_noise,
                        cp_uv_min, ttype,
                        fft_pad_factor, fft_conv_func, fft_gridder_prad,
                        fft_interp_order):
    """Pre-compute (data, sigma, A) tuples for every (data-term, observation) pair.

    Dispatches to polutils.polchisqdata for polarimetric terms (in
    DATATERMS_POL) and imutils.chisqdata for standard terms (in DATATERMS).

    Parameters
    ----------
    obslist : list of Obsdata
    prior : Image
    embed_mask : np.ndarray of bool
    dat_term_keys : iterable of str
        Sorted dat_term names. Each must be in DATATERMS or DATATERMS_POL.
    pol : str
        Polarization mode of the imager (e.g. 'I', 'IP', 'IV').
    maxset, debias, snrcut, weighting, systematic_noise,
    systematic_cphase_noise, cp_uv_min : scalar / dict
        Per-data-term knobs forwarded to imutils.chisqdata. snrcut is a
        dict keyed by dname.
    ttype : {'direct', 'fast', 'nfft'}
    fft_pad_factor, fft_conv_func, fft_gridder_prad, fft_interp_order :
        FFT/NFFT-specific parameters.

    Returns
    -------
    dict
        Keys are dname (single-obs) or f"{dname}_{i}" (multi-obs).
        Values are (data, sigma, A) tuples returned by chisqdata/polchisqdata.
    """
    n_obs = len(obslist)
    data_tuples = {}

    for dname in dat_term_keys:
        for i, obs in enumerate(obslist):
            dname_key = dname if n_obs == 1 else f"{dname}_{i}"

            if dname in DATATERMS_POL:
                tup = polutils.polchisqdata(obs, prior, embed_mask, dname,
                                            ttype=ttype,
                                            fft_pad_factor=fft_pad_factor,
                                            conv_func=fft_conv_func,
                                            p_rad=fft_gridder_prad)

            elif dname in DATATERMS:
                if pol in POLARIZATION_MODES:
                    if 'I' not in pol:
                        raise Exception(
                            f"cannot use dterm {dname} with pol={pol}")
                    dterm_pol = 'I'
                else:
                    dterm_pol = pol

                tup = imutils.chisqdata(obs, prior, embed_mask, dname,
                                        pol=dterm_pol, maxset=maxset,
                                        debias=debias,
                                        snrcut=snrcut[dname],
                                        weighting=weighting,
                                        systematic_noise=systematic_noise,
                                        systematic_cphase_noise=systematic_cphase_noise,
                                        ttype=ttype, order=fft_interp_order,
                                        fft_pad_factor=fft_pad_factor,
                                        conv_func=fft_conv_func,
                                        p_rad=fft_gridder_prad,
                                        cp_uv_min=cp_uv_min)
            else:
                raise Exception(f"data term {dname} not recognized!")

            data_tuples[dname_key] = tup

    return data_tuples


def compute_init_state(
    obslist, init_image, prior_image,
    freq_list, reffreq,
    pol, mf, transforms,
    mf_order, mf_order_pol, mf_rm, mf_cm,
    norm_init, flux, clipfloor,
    dat_term_keys, maxset, debias, snrcut, weighting,
    systematic_noise, systematic_cphase_noise, cp_uv_min,
    ttype, fft_pad_factor, fft_conv_func, fft_gridder_prad, fft_interp_order,
    *, compute_data=True, prior_data_tuples=None,
):
    """Build solver-ready imager state. Pure function.

    Composes compute_embed + compute_logfreqratios + compute_which_solve +
    make_initarr + transform_imarr_inverse + pack_imarr + compute_data_tuples
    into the single state bundle consumed by compute_objective /
    compute_objective_grad.

    JAX note: when jitted, expect static_argnames=('pol', 'mf', 'transforms',
    'mf_order', 'mf_order_pol', 'mf_rm', 'mf_cm', 'ttype', 'dat_term_keys',
    'compute_data') plus static dict-key structure on snrcut.

    Parameters
    ----------
    obslist : list of Obsdata
    init_image, prior_image : Image
        Initial image (L-BFGS-B start point) and regularizer prior image.
    freq_list : list of float
        Reference frequencies (Hz) of each obs in obslist.
    reffreq : float
        Reference frequency (Hz) of the multi-frequency expansion. When
        mf=True this is overridden by init_image.rf.
    pol, mf, transforms : str, bool, list of str
        Polarization mode, multi-frequency flag, transform stack
        (e.g. ['log', 'mcv']).
    mf_order, mf_order_pol, mf_rm, mf_cm :
        Multi-frequency expansion orders + RM/CM solve flags.
    norm_init : bool
    flux, clipfloor : float
    dat_term_keys, maxset, debias, snrcut, weighting,
    systematic_noise, systematic_cphase_noise, cp_uv_min, ttype,
    fft_pad_factor, fft_conv_func, fft_gridder_prad, fft_interp_order :
        Forwarded to compute_data_tuples; see its docstring.

    Other Parameters
    ----------------
    compute_data : bool, optional
        When False, skip compute_data_tuples and reuse prior_data_tuples
        (used by Imager when _change_imgr_params is False).
    prior_data_tuples : dict, optional
        Pre-existing data_tuples dict reused when compute_data=False.

    Returns
    -------
    ImagerInitState
    """
    n_obs = len(obslist)
    if len(freq_list) != n_obs:
        raise Exception(
            "compute_init_state: len(obslist) and len(freq_list) must match.")

    embed_mask, coord_matrix = compute_embed(
        prior_image.imvec, prior_image.xdim, prior_image.ydim,
        prior_image.psize, clipfloor,
    )
    nimage = int(np.sum(embed_mask))

    # multi-freq: reffreq is re-bound to the init image's rf (matches the
    # legacy Imager.init_imager behavior at the start of the mf branch).
    reffreq_eff = init_image.rf if mf else reffreq
    logfreqratio_list = compute_logfreqratios(freq_list, reffreq_eff)

    which_solve = compute_which_solve(
        pol, mf, mf_order=mf_order, mf_order_pol=mf_order_pol,
        mf_rm=mf_rm, mf_cm=mf_cm,
    )

    is_pol = pol in POLARIZATION_MODES

    # Decide here (not inside make_initarr) whether random-pol initialization
    # is needed. Random init only kicks in when (a) the imager is in
    # polarimetric mode for that Stokes block, AND (b) init_image has no
    # nonzero polarization to use as a starting point. If init_image already
    # carries Q/U/V, make_initarr will use those values directly.
    init_has_pol_lin = is_pol and (
        (len(init_image.qvec) > 0 and np.any(init_image.qvec != 0))
        or (len(init_image.uvec) > 0 and np.any(init_image.uvec != 0))
        or (len(init_image.vvec) > 0 and np.any(init_image.vvec != 0))
    )
    init_has_pol_circ = is_pol and (
        len(init_image.vvec) > 0 and np.any(init_image.vvec != 0)
    )
    randompol_lin = is_pol and (('P' in pol) or ('QU' in pol)) and not init_has_pol_lin
    randompol_circ = is_pol and ('V' in pol) and not init_has_pol_circ

    init_phys = make_initarr(
        init_image, embed_mask,
        norm_init=norm_init, flux=flux,
        mf=mf, pol=is_pol,
        randompol_lin=randompol_lin, randompol_circ=randompol_circ,
        meanpol=MEANPOL_INIT, sigmapol=SIGMAPOL_INIT,
    )
    prior_phys = make_initarr(
        prior_image, embed_mask,
        norm_init=norm_init, flux=flux,
        mf=mf, pol=is_pol,
        randompol_lin=False, randompol_circ=False,
    )

    init_solver = transform_imarr_inverse(init_phys, transforms, which_solve)
    init_vec = pack_imarr(init_solver, which_solve)

    if compute_data:
        data_tuples = compute_data_tuples(
            obslist, prior_image, embed_mask, dat_term_keys, pol,
            maxset, debias, snrcut, weighting,
            systematic_noise, systematic_cphase_noise, cp_uv_min,
            ttype, fft_pad_factor, fft_conv_func, fft_gridder_prad,
            fft_interp_order,
        )
    else:
        data_tuples = prior_data_tuples

    return ImagerInitState(
        init_arr=init_solver, init_vec=init_vec, prior_arr=prior_phys,
        data_tuples=data_tuples, embed_mask=embed_mask,
        coord_matrix=coord_matrix, logfreqratio_list=logfreqratio_list,
        nimage=nimage, which_solve=which_solve, reffreq=reffreq_eff,
    )


def compute_embed(imvec, xdim, ydim, psize, clipfloor):
    """Compute embedding mask and coordinate matrix from a prior image vector.

    Parameters
    ----------
    imvec : np.ndarray
        Prior image vector (full, not embedded).
    xdim : int
        Image x dimension.
    ydim : int
        Image y dimension.
    psize : float
        Pixel size in radians.
    clipfloor : float
        Minimum pixel value; pixels below this are masked out.

    Returns
    -------
    embed_mask : np.ndarray of bool
        Boolean mask, True for pixels above clipfloor.
    coord_matrix : np.ndarray, shape (n_embed, 2)
        Pixel coordinates (in radians) for unmasked pixels.
    """

    embed_mask = (imvec > clipfloor)
    if not np.any(embed_mask):
        raise Exception("clipfloor too large: all prior pixels have been clipped!")

    xmax = xdim // 2
    ymax = ydim // 2

    if xdim % 2:
        xmin = -xmax - 1
    else:
        xmin = -xmax

    if ydim % 2:
        ymin = -ymax - 1
    else:
        ymin = -ymax

    coord = np.array([[[x, y]
                        for x in np.arange(xmax, xmin, -1)]
                        for y in np.arange(ymax, ymin, -1)])

    coord = coord.reshape(ydim * xdim, 2)
    coord = coord * psize

    coord_matrix = coord[embed_mask]

    return embed_mask, coord_matrix


def compute_chisq_dict(imcur, dat_term_keys,
                       mf, pol,
                       data_tuples, logfreqratio_list, n_obs,
                       ttype, embed_mask):
    """Compute chi^2 value for each data term across all observations.

    Parameters
    ----------
    imcur : np.ndarray
        Current image array transformed to bounded values.
    dat_term_keys : list of str
        Data term names to evaluate, already sorted.
    mf : bool
        Whether multifrequency imaging is enabled.
    pol : str
        Polarization mode string.
    data_tuples : dict
        Pre-computed data products keyed by dname or dname_i,
        each value is a (data, sigma, A) tuple.
    logfreqratio_list : list of float
        Log frequency ratios log(nu_i/reffreq); one per obs.
    n_obs : int
        Number of observations (frequencies/epochs). Must equal
        len(logfreqratio_list); validated by Imager.init_imager.
    ttype : str
        Transform type ('direct', 'fast', 'nfft').
    embed_mask : np.ndarray of bool
        Pixel embedding mask.

    Returns
    -------
    chi2_dict : dict
        Mapping from dname (or dname_i for multi-obs) to chi^2 scalar.
    """
    chi2_dict = {}
    for dname in dat_term_keys:
        # Loop over all observations
        for i in range(n_obs):
            if n_obs == 1:
                dname_key = dname
            else:
                dname_key = dname + (f'_{i}')

            # get data products
            (data, sigma, A) = data_tuples[dname_key]

            # get current multifrequency image
            if mf:
                logfreqratio = logfreqratio_list[i]
                imcur_nu = mfutils.image_at_freq(imcur, logfreqratio)
            else:
                imcur_nu = imcur

            # Polarization chi^2 terms
            if dname in DATATERMS_POL:
                chi2 = polutils.polchisq(imcur_nu, A, data, sigma, dname,
                                         ttype=ttype, mask=embed_mask)

            # Single Polarization chi^2 terms
            elif dname in DATATERMS:
                if pol in POLARIZATION_MODES:
                    imcur_nu_I = imcur_nu[0]
                else:
                    imcur_nu_I = imcur_nu
                chi2 = imutils.chisq(imcur_nu_I, A, data, sigma, dname,
                                     ttype=ttype, mask=embed_mask)

            else:
                raise Exception(f"data term {dname} not recognized!")

            chi2_dict[dname_key] = chi2

    return chi2_dict


def compute_chisqgrad_dict(imcur, dat_term_keys,
                           mf, pol,
                           data_tuples, logfreqratio_list, n_obs,
                           ttype, embed_mask,
                           which_solve, nimage):
    """Compute chi^2 gradient for each data term across all observations.

    Parameters
    ----------
    imcur : np.ndarray
        Current image array transformed to bounded values.
    dat_term_keys : list of str
        Data term names to evaluate, already sorted.
    mf : bool
        Whether multifrequency imaging is enabled.
    pol : str
        Polarization mode string.
    data_tuples : dict
        Pre-computed data products keyed by dname or dname_i,
        each value is a (data, sigma, A) tuple.
    logfreqratio_list : list of float
        Log frequency ratios log(nu_i/reffreq); one per obs.
    n_obs : int
        Number of observations (frequencies/epochs). Must equal
        len(logfreqratio_list); validated by Imager.init_imager.
    ttype : str
        Transform type ('direct', 'fast', 'nfft').
    embed_mask : np.ndarray of bool
        Pixel embedding mask.
    which_solve : np.ndarray of int
        Binary flags for which parameters are solved.
    nimage : int
        Number of active pixels (sum of embed_mask).

    Returns
    -------
    chi2grad_dict : dict
        Mapping from dname (or dname_i for multi-obs) to chi^2 gradient array.
    """
    chi2grad_dict = {}
    # Zero row reused in the polarization-bundled Stokes-I gradient; safe to share
    # because np.array((...)) below copies into a new (4, nimage) array each time.
    zero_row = np.zeros(nimage)
    for dname in dat_term_keys:
        # Loop over all observations
        for i in range(n_obs):
            if n_obs == 1:
                dname_key = dname
            else:
                dname_key = dname + (f'_{i}')

            # get data products
            (data, sigma, A) = data_tuples[dname_key]

            # get current multifrequency image
            if mf:
                logfreqratio = logfreqratio_list[i]
                imcur_nu = mfutils.image_at_freq(imcur, logfreqratio)
            else:
                imcur_nu = imcur

            # Polarimetric chi^2 gradients
            if dname in DATATERMS_POL:
                if mf:
                    pol_solve = which_solve[0:4]
                else:
                    pol_solve = which_solve
                chi2grad = polutils.polchisqgrad(imcur_nu, A, data, sigma, dname,
                                                 ttype=ttype, mask=embed_mask,
                                                 pol_solve=pol_solve)

            # Single polarization chi^2 gradients
            elif dname in DATATERMS:
                if pol in POLARIZATION_MODES:  # polarization
                    imcur_nu_I = imcur_nu[0]
                else:
                    imcur_nu_I = imcur_nu

                chi2grad = imutils.chisqgrad(imcur_nu_I, A, data, sigma, dname,
                                             ttype=ttype, mask=embed_mask)

                # If imaging Stokes I with polarization simultaneously, bundle the gradient
                if pol in POLARIZATION_MODES:
                    chi2grad = np.array((chi2grad, zero_row, zero_row, zero_row))

            else:
                raise Exception(f"data term {dname} not recognized!")

            # If multifrequency imaging,
            # transform the image gradients for all the solved quantities
            if mf:
                logfreqratio = logfreqratio_list[i]
                chi2grad = mfutils.mf_all_grads_chain(chi2grad, imcur_nu, imcur, logfreqratio)

            chi2grad_dict[dname_key] = np.array(chi2grad)

    return chi2grad_dict


def compute_reg_dict(imcur, reg_term_keys,
                     mf, pol,
                     logfreqratio_list, n_obs,
                     priorvec, norm_reg, regparams,
                     embed_mask):
    """Compute regularizer value for each regularizer term.

    Parameters
    ----------
    imcur : np.ndarray
        Current image array transformed to bounded values.
    reg_term_keys : list of str
        Regularizer term names to evaluate, already sorted.
    mf : bool
        Whether multifrequency imaging is enabled.
    pol : str
        Polarization mode string.
    logfreqratio_list : list of float
        Log frequency ratios log(nu_i/reffreq); one per obs.
    n_obs : int
        Number of observations (frequencies/epochs). Must equal
        len(logfreqratio_list); validated by Imager.init_imager.
    priorvec : np.ndarray
        Prior image array (same shape as imcur). Used by regularizer terms
        that depend on a reference image; distinct from `initvec` (the
        initial image used for not-solved-for slots in `unpack_imarr`).
    norm_reg : bool
        Whether to apply per-regularizer normalization.
    regparams : dict
        Bundle of regularizer parameters: must contain `flux`, `pflux`, `vflux`,
        `xdim`, `ydim`, `psize`, `beam_size`, and `mf_flux` (list of per-freq
        fluxes, required when any REGULARIZERS_ALLFREQS_I term is active);
        plus any term-specific kwargs (e.g. `major`, `minor`, `PA`, `alpha_A`,
        `epsilon_tv`). Spread into the underlying dispatchers as **kwargs.

    Returns
    -------
    reg_dict : dict
        Mapping from regname to regularizer scalar.
    """
    mf_flux = regparams.get('mf_flux')
    reg_dict = {}

    for regname in reg_term_keys:

        # Multifrequency regularizers
        if mf:

            # Polarimetric regularizers
            if regname in REGULARIZERS_POL:
                imcur_pol = imcur[0:4]
                prior_pol = priorvec[0:4]
                reg = polutils.polregularizer(imcur_pol, prior_pol, embed_mask,
                                              stype=regname, norm_reg=norm_reg,
                                              **regparams)

            # Stokes I regularizers
            elif regname in REGULARIZERS:

                if regname in REGULARIZERS_ALLFREQS_I:

                    # TODO move this to checks?
                    if (not isinstance(mf_flux, list)) or len(mf_flux) != n_obs:
                        raise Exception(f"when using regularizer '{regname}', "
                                        + "mf_flux must be a list of same length as n_obs!")

                    regname_base = '_'.join(regname.split('_')[:-1])  # remove the '_mf' tag
                    for i in range(n_obs):

                        logfreqratio = logfreqratio_list[i]
                        imcur_nu = mfutils.image_at_freq(imcur, logfreqratio)
                        prior_nu = mfutils.image_at_freq(priorvec, logfreqratio)

                        regi = imutils.regularizer(imcur_nu, prior_nu, embed_mask,
                                                   stype=regname_base, norm_reg=norm_reg,
                                                   **{**regparams, 'flux': mf_flux[i]})

                        reg = regi if i == 0 else reg + regi

                else:
                    reg = imutils.regularizer(imcur[0], priorvec[0], embed_mask,
                                              stype=regname, norm_reg=norm_reg,
                                              **regparams)

            # Spectral regularizers
            elif regname in REGULARIZERS_SPECTRAL:

                if regname in REGULARIZERS_SPECIND:
                    idx = 4 if len(imcur) == 10 else 1
                elif regname in REGULARIZERS_CURV:
                    idx = 5 if len(imcur) == 10 else 2
                elif regname in REGULARIZERS_SPECIND_P:
                    idx = 6
                elif regname in REGULARIZERS_CURV_P:
                    idx = 7
                elif regname in REGULARIZERS_RM:
                    idx = 8
                elif regname in REGULARIZERS_CM:
                    idx = 9

                reg = mfutils.regularizer_mf(imcur[idx], priorvec[idx], embed_mask,
                                             stype=regname, norm_reg=norm_reg,
                                             **regparams)
            else:
                raise Exception(f"regularizer term {regname} not recognized!")

        # Single-frequency polarimetric regularizer
        elif regname in REGULARIZERS_POL:
            reg = polutils.polregularizer(imcur, priorvec, embed_mask,
                                          stype=regname, norm_reg=norm_reg,
                                          **regparams)

        # Single-frequency, single-polarization regularizer
        elif regname in REGULARIZERS:
            if pol in POLARIZATION_MODES:
                imcur0 = imcur[0]
                prior0 = priorvec[0]
            else:
                imcur0 = imcur
                prior0 = priorvec

            reg = imutils.regularizer(imcur0, prior0, embed_mask,
                                      stype=regname, norm_reg=norm_reg,
                                      **regparams)
        else:
            raise Exception(f"regularizer term {regname} not recognized!")

        reg_dict[regname] = reg

    return reg_dict


def compute_reggrad_dict(imcur, reg_term_keys,
                         mf, pol,
                         logfreqratio_list, n_obs,
                         priorvec, norm_reg, regparams,
                         embed_mask,
                         which_solve, nimage):
    """Compute regularizer gradient for each regularizer term.

    Parameters
    ----------
    imcur, reg_term_keys, mf, pol, logfreqratio_list, n_obs,
    priorvec, norm_reg, regparams, embed_mask : see compute_reg_dict.
    which_solve : np.ndarray of bool
        Per-Stokes solve mask (used by polregularizergrad).
    nimage : int
        Number of solved-for pixels (rows of the gradient array).

    Returns
    -------
    reggrad_dict : dict
        Mapping from regname to gradient array of shape (nimage,)
        for single-pol, (4, nimage) for pol-bundled, or
        (len(imcur), nimage) for multifrequency.
    """
    mf_flux = regparams.get('mf_flux')
    reggrad_dict = {}

    for regname in reg_term_keys:

        # Multifrequency regularizers
        if mf:

            # Polarimetric regularizers
            if regname in REGULARIZERS_POL:
                imcur_pol = imcur[0:4]
                prior_pol = priorvec[0:4]
                pol_solve = which_solve[0:4]
                regp = polutils.polregularizergrad(imcur_pol, prior_pol, embed_mask,
                                                   stype=regname, norm_reg=norm_reg,
                                                   pol_solve=pol_solve,
                                                   **regparams)
                reggrad = np.zeros((len(imcur), nimage))
                reggrad[0:4] = regp

            # Stokes I regularizers
            elif regname in REGULARIZERS:

                if regname in REGULARIZERS_ALLFREQS_I:

                    # TODO move this to checks?
                    if (not isinstance(mf_flux, list)) or len(mf_flux) != n_obs:
                        raise Exception(f"when using regularizer '{regname}', "
                                        + "mf_flux must be a list of same length as n_obs!")

                    regname_base = '_'.join(regname.split('_')[:-1])  # remove the '_mf' tag
                    for i in range(n_obs):

                        logfreqratio = logfreqratio_list[i]
                        imcur_nu = mfutils.image_at_freq(imcur, logfreqratio)
                        prior_nu = mfutils.image_at_freq(priorvec, logfreqratio)

                        regi = imutils.regularizergrad(imcur_nu, prior_nu, embed_mask,
                                                       stype=regname_base, norm_reg=norm_reg,
                                                       **{**regparams, 'flux': mf_flux[i]})
                        reggrad_i = mfutils.mf_all_grads_chain(regi, imcur_nu, imcur, logfreqratio)
                        reggrad = reggrad_i if i == 0 else reggrad + reggrad_i

                else:
                    regi = imutils.regularizergrad(imcur[0], priorvec[0], embed_mask,
                                                   stype=regname, norm_reg=norm_reg,
                                                   **regparams)
                    reggrad = np.zeros((len(imcur), nimage))
                    reggrad[0] = regi

            elif regname in REGULARIZERS_SPECTRAL:
                if regname in REGULARIZERS_SPECIND:
                    idx = 4 if len(imcur) == 10 else 1
                elif regname in REGULARIZERS_CURV:
                    idx = 5 if len(imcur) == 10 else 2
                elif regname in REGULARIZERS_SPECIND_P:
                    idx = 6
                elif regname in REGULARIZERS_CURV_P:
                    idx = 7
                elif regname in REGULARIZERS_RM:
                    idx = 8
                elif regname in REGULARIZERS_CM:
                    idx = 9

                regmf = mfutils.regularizergrad_mf(imcur[idx], priorvec[idx], embed_mask,
                                                   stype=regname, norm_reg=norm_reg,
                                                   **regparams)

                reggrad = np.zeros((len(imcur), nimage))
                reggrad[idx] = regmf
            else:
                raise Exception(f"regularizer term {regname} not recognized!")

        else:
            # Single-frequency polarimetric regularizer
            if regname in REGULARIZERS_POL:
                reggrad = polutils.polregularizergrad(imcur, priorvec, embed_mask,
                                                      stype=regname, norm_reg=norm_reg,
                                                      pol_solve=which_solve,
                                                      **regparams)

            # Single-frequency, single polarization regularizer
            elif regname in REGULARIZERS:
                if pol in POLARIZATION_MODES:
                    imcur0 = imcur[0]
                    prior0 = priorvec[0]
                else:
                    imcur0 = imcur
                    prior0 = priorvec
                reggrad = imutils.regularizergrad(imcur0, prior0, embed_mask,
                                                  stype=regname, norm_reg=norm_reg,
                                                  **regparams)

                if pol in POLARIZATION_MODES:
                    reggrad = np.array((reggrad,
                                        np.zeros(nimage),
                                        np.zeros(nimage),
                                        np.zeros(nimage)))

            else:
                raise Exception(f"regularizer term {regname} not recognized!")

        reggrad_dict[regname] = reggrad

    return reggrad_dict


def compute_objective(imvec, initvec,
                      mf, pol,
                      which_solve, data_tuples, logfreqratio_list, n_obs,
                      dat_term, reg_term,
                      priorvec, norm_reg, regparams,
                      transforms, embed_mask, ttype):
    """Compute the scalar imaging objective: data fidelity + regularization.

    Pure-functional version of Imager.objfunc. Unpacks the 1D solver vector
    `imvec` into a multi-D image array (filling not-solved-for slots from
    `initvec`), applies the bounded-value transforms, evaluates each chi^2
    and regularizer term via the dispatcher dicts, and returns the
    hyperparameter-weighted sum.

    Parameters
    ----------
    imvec : np.ndarray
        1D solver vector (length sum(which_solve) * nimage), the optimization
        variable for L-BFGS-B.
    initvec : np.ndarray
        Initial-image array (unwrapped, multi-D); used by `unpack_imarr` to
        fill any not-solved-for slots. Distinct from `priorvec` (the
        regularizer prior).
    mf : bool
        Multifrequency-imaging flag.
    pol : str
        Polarization mode (e.g. 'I', 'P', 'IP', 'V', 'IV', ...).
    which_solve : np.ndarray of int
        Per-Stokes / per-spectral-term solve mask.
    data_tuples : dict
        Pre-computed (data, sigma, A) tuples keyed by dname or dname_i.
    logfreqratio_list : list of float
        log(nu_i / reffreq) for each observation.
    n_obs : int
        Number of observations. Must equal len(logfreqratio_list); enforced
        upstream by Imager.init_imager.
    dat_term, reg_term : dict
        Hyperparameter weights per data term and per regularizer term.
    priorvec : np.ndarray
        Prior-image array consumed by regularizers like 'simple', 'l1',
        'rgauss'. Distinct from `initvec` (the optimization start point).
    norm_reg : bool
        Whether to apply per-regularizer normalization.
    regparams : dict
        Bundle of regularizer parameters. See compute_reg_dict.
    transforms : list of str
        Image transform list (e.g. ['log', 'mcv']) applied to imcur.
    embed_mask : np.ndarray of bool
        Pixel embedding mask.
    ttype : str
        Fourier-transform type ('direct', 'fast', 'nfft').

    Returns
    -------
    cost : float
        Scalar objective value, sum of weighted chi^2 deviations and
        regularizer contributions.
    """

    dat_term_keys = sorted(dat_term.keys())
    reg_term_keys = sorted(reg_term.keys())

    # Unpack solver vector into image array, then apply bounded-value transform.
    # `initvec` provides the values used for any not-solved-for slots (it is
    # the *initial* image, NOT the regularizer prior `priorvec`).
    imcur = unpack_imarr(imvec, initvec, which_solve)
    imcur = transform_imarr(imcur, transforms, which_solve)

    chi2_dict = compute_chisq_dict(
        imcur, dat_term_keys,
        mf, pol,
        data_tuples, logfreqratio_list, n_obs,
        ttype, embed_mask,
    )
    reg_dict = compute_reg_dict(
        imcur, reg_term_keys,
        mf, pol,
        logfreqratio_list, n_obs,
        priorvec, norm_reg, regparams,
        embed_mask,
    )

    datterm = 0.0
    for dname in dat_term_keys:
        weight = dat_term[dname]
        for i in range(n_obs):
            key = dname if n_obs == 1 else f"{dname}_{i}"
            datterm = datterm + weight * (chi2_dict[key] - 1.0)

    regterm = 0.0
    for regname in reg_term_keys:
        regterm = regterm + reg_term[regname] * reg_dict[regname]

    return datterm + regterm


def compute_objective_grad(imvec, initvec,
                           mf, pol,
                           which_solve, data_tuples, logfreqratio_list, n_obs,
                           dat_term, reg_term,
                           priorvec, norm_reg, regparams,
                           transforms, embed_mask, ttype, nimage):
    """Compute the gradient of the imaging objective with respect to imvec.

    Pure-functional version of Imager.objgrad. Computes the chi^2 and
    regularizer gradients via the dispatcher dicts, sums them with the
    hyperparameter weights, applies the chain rule through the bounded-value
    transform, and packs the result back into solver space.

    Designed so that `jax.grad(compute_objective)` is a drop-in replacement
    once the underlying backends are JAXified (Phase 5D three-way check
    against finite differences and analytic).

    Parameters
    ----------
    imvec, initvec, mf, pol, which_solve, data_tuples, logfreqratio_list,
    n_obs, dat_term, reg_term, priorvec, norm_reg, regparams, transforms,
    embed_mask, ttype : see compute_objective.
    nimage : int
        Number of active pixels (sum of embed_mask). Used to size the
        pre-pack gradient array.

    Returns
    -------
    grad : np.ndarray
        1D gradient vector (same length as imvec), suitable as the `jac`
        argument to scipy.optimize.minimize.
    """

    dat_term_keys = sorted(dat_term.keys())
    reg_term_keys = sorted(reg_term.keys())

    # Unpack solver vector; keep a pre-transform copy for the chain rule.
    # `initvec` is the initial image (for not-solved-for slots), distinct
    # from `priorvec` which is the regularizer prior.
    imcur = unpack_imarr(imvec, initvec, which_solve)
    imcur_prime = imcur.copy()
    imcur = transform_imarr(imcur, transforms, which_solve)

    chi2grad_dict = compute_chisqgrad_dict(
        imcur, dat_term_keys,
        mf, pol,
        data_tuples, logfreqratio_list, n_obs,
        ttype, embed_mask,
        which_solve, nimage,
    )

    reggrad_dict = compute_reggrad_dict(
        imcur, reg_term_keys,
        mf, pol,
        logfreqratio_list, n_obs,
        priorvec, norm_reg, regparams,
        embed_mask,
        which_solve, nimage,
    )

    datterm = 0.0
    for dname in dat_term_keys:
        weight = dat_term[dname]
        for i in range(n_obs):
            key = dname if n_obs == 1 else f"{dname}_{i}"
            datterm = datterm + weight * chi2grad_dict[key]

    regterm = 0.0
    for regname in reg_term_keys:
        regterm = regterm + reg_term[regname] * reggrad_dict[regname]

    grad = datterm + regterm
    grad = transform_gradients(grad, imcur_prime, transforms, which_solve)
    return pack_imarr(grad, which_solve)
