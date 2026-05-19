# imager_backend.py
# Pure functional backend for imager.py

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

import ehtim.imaging.imager_utils as imutils
import ehtim.imaging.multifreq_imager_utils as mfutils
import ehtim.imaging.pol_imager_utils as polutils
import ehtim.observing.obs_helpers as obsh

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

# Whether mask-embedding under fast/nfft fills non-mask pixels with random
# noise (vs a constant clipfloor). Set True historically to break TV-regularizer
# gradient singularities; the newer `epsilon_tv` mechanism likely makes this
# unnecessary -- revisit once epsilon_tv coverage is broader.
EMBED_RANDOMFLOOR = True

# `logamp` aliases `amp` (same data product). Pol dtypes have no `fast`.
_CHISQDATA_DISPATCH = {
    'vis':          {'direct': imutils.chisqdata_vis,
                     'fast':   imutils.chisqdata_vis_fft,
                     'nfft':   imutils.chisqdata_vis_nfft},
    'amp':          {'direct': imutils.chisqdata_amp,
                     'fast':   imutils.chisqdata_amp_fft,
                     'nfft':   imutils.chisqdata_amp_nfft},
    'logamp':       {'direct': imutils.chisqdata_amp,
                     'fast':   imutils.chisqdata_amp_fft,
                     'nfft':   imutils.chisqdata_amp_nfft},
    'bs':           {'direct': imutils.chisqdata_bs,
                     'fast':   imutils.chisqdata_bs_fft,
                     'nfft':   imutils.chisqdata_bs_nfft},
    'cphase':       {'direct': imutils.chisqdata_cphase,
                     'fast':   imutils.chisqdata_cphase_fft,
                     'nfft':   imutils.chisqdata_cphase_nfft},
    'cphase_diag':  {'direct': imutils.chisqdata_cphase_diag,
                     'fast':   imutils.chisqdata_cphase_diag_fft,
                     'nfft':   imutils.chisqdata_cphase_diag_nfft},
    'camp':         {'direct': imutils.chisqdata_camp,
                     'fast':   imutils.chisqdata_camp_fft,
                     'nfft':   imutils.chisqdata_camp_nfft},
    'logcamp':      {'direct': imutils.chisqdata_logcamp,
                     'fast':   imutils.chisqdata_logcamp_fft,
                     'nfft':   imutils.chisqdata_logcamp_nfft},
    'logcamp_diag': {'direct': imutils.chisqdata_logcamp_diag,
                     'fast':   imutils.chisqdata_logcamp_diag_fft,
                     'nfft':   imutils.chisqdata_logcamp_diag_nfft},
    'pvis':         {'direct': polutils.chisqdata_pvis,
                     'nfft':   polutils.chisqdata_pvis_nfft},
    'm':            {'direct': polutils.chisqdata_m,
                     'nfft':   polutils.chisqdata_m_nfft},
    'vvis':         {'direct': polutils.chisqdata_vvis,
                     'nfft':   polutils.chisqdata_vvis_nfft},
}

# _diag leaves index the gridded image directly per-baseline; their `fast`
# variants take imvec rather than the pre-FFT'd vis_arr.
_DIAG_DTYPES = frozenset({'cphase_diag', 'logcamp_diag'})

# Pol dtypes have no `fast`; entries are (chisq_fn, chisqgrad_fn).
_CHISQ_DISPATCH = {
    'vis':          {'is_pol': False,
                     'direct': (imutils.chisq_vis,          imutils.chisqgrad_vis),
                     'fast':   (imutils.chisq_vis_fft,      imutils.chisqgrad_vis_fft),
                     'nfft':   (imutils.chisq_vis_nfft,     imutils.chisqgrad_vis_nfft)},
    'amp':          {'is_pol': False,
                     'direct': (imutils.chisq_amp,          imutils.chisqgrad_amp),
                     'fast':   (imutils.chisq_amp_fft,      imutils.chisqgrad_amp_fft),
                     'nfft':   (imutils.chisq_amp_nfft,     imutils.chisqgrad_amp_nfft)},
    'logamp':       {'is_pol': False,
                     'direct': (imutils.chisq_logamp,       imutils.chisqgrad_logamp),
                     'fast':   (imutils.chisq_logamp_fft,   imutils.chisqgrad_logamp_fft),
                     'nfft':   (imutils.chisq_logamp_nfft,  imutils.chisqgrad_logamp_nfft)},
    'bs':           {'is_pol': False,
                     'direct': (imutils.chisq_bs,           imutils.chisqgrad_bs),
                     'fast':   (imutils.chisq_bs_fft,       imutils.chisqgrad_bs_fft),
                     'nfft':   (imutils.chisq_bs_nfft,      imutils.chisqgrad_bs_nfft)},
    'cphase':       {'is_pol': False,
                     'direct': (imutils.chisq_cphase,       imutils.chisqgrad_cphase),
                     'fast':   (imutils.chisq_cphase_fft,   imutils.chisqgrad_cphase_fft),
                     'nfft':   (imutils.chisq_cphase_nfft,  imutils.chisqgrad_cphase_nfft)},
    'cphase_diag':  {'is_pol': False,
                     'direct': (imutils.chisq_cphase_diag,      imutils.chisqgrad_cphase_diag),
                     'fast':   (imutils.chisq_cphase_diag_fft,  imutils.chisqgrad_cphase_diag_fft),
                     'nfft':   (imutils.chisq_cphase_diag_nfft, imutils.chisqgrad_cphase_diag_nfft)},
    'camp':         {'is_pol': False,
                     'direct': (imutils.chisq_camp,         imutils.chisqgrad_camp),
                     'fast':   (imutils.chisq_camp_fft,     imutils.chisqgrad_camp_fft),
                     'nfft':   (imutils.chisq_camp_nfft,    imutils.chisqgrad_camp_nfft)},
    'logcamp':      {'is_pol': False,
                     'direct': (imutils.chisq_logcamp,      imutils.chisqgrad_logcamp),
                     'fast':   (imutils.chisq_logcamp_fft,  imutils.chisqgrad_logcamp_fft),
                     'nfft':   (imutils.chisq_logcamp_nfft, imutils.chisqgrad_logcamp_nfft)},
    'logcamp_diag': {'is_pol': False,
                     'direct': (imutils.chisq_logcamp_diag,      imutils.chisqgrad_logcamp_diag),
                     'fast':   (imutils.chisq_logcamp_diag_fft,  imutils.chisqgrad_logcamp_diag_fft),
                     'nfft':   (imutils.chisq_logcamp_diag_nfft, imutils.chisqgrad_logcamp_diag_nfft)},
    'pvis':         {'is_pol': True,
                     'direct': (polutils.chisq_p,           polutils.chisqgrad_p),
                     'nfft':   (polutils.chisq_p_nfft,      polutils.chisqgrad_p_nfft)},
    'm':            {'is_pol': True,
                     'direct': (polutils.chisq_m,           polutils.chisqgrad_m),
                     'nfft':   (polutils.chisq_m_nfft,      polutils.chisqgrad_m_nfft)},
    'vvis':         {'is_pol': True,
                     'direct': (polutils.chisq_vvis,        polutils.chisqgrad_vvis),
                     'nfft':   (polutils.chisq_vvis_nfft,   polutils.chisqgrad_vvis_nfft)},
}


# Dispatch table: regname -> (value_fn, grad_fn, family).
# Each entry points at a `reg_X` / `reggrad_X` wrapper in the regularizer's home
# file (imager_utils / pol_imager_utils / multifreq_imager_utils). The wrappers
# share a uniform (imvec_or_imarr, mask, **kwargs) signature and own their
# embed-pre / mask-post-slice logic. family is informational for compute_reg_dict
# to decide how to slice the imcur input and how to bundle the gradient back
# into a multi-Stokes shape when the imager runs in polarization mode.
_REGULARIZER_DISPATCH = {
    # Stokes-I (REGULARIZERS)
    'flux':     (imutils.reg_flux,     imutils.reggrad_flux,     'stokes_i'),
    # flux_mf shares the flux machinery; compute_reg_dict strips '_mf' and
    # loops over frequencies, but direct callers still need to resolve the name.
    'flux_mf':  (imutils.reg_flux,     imutils.reggrad_flux,     'stokes_i'),
    'simple':   (imutils.reg_simple,   imutils.reggrad_simple,   'stokes_i'),
    'l1':       (imutils.reg_l1,       imutils.reggrad_l1,       'stokes_i'),
    'l1w':      (imutils.reg_l1w,      imutils.reggrad_l1w,      'stokes_i'),
    'lA':       (imutils.reg_lA,       imutils.reggrad_lA,       'stokes_i'),
    'gs':       (imutils.reg_gs,       imutils.reggrad_gs,       'stokes_i'),
    'patch':    (imutils.reg_patch,    imutils.reggrad_patch,    'stokes_i'),
    'cm':       (imutils.reg_cm,       imutils.reggrad_cm,       'stokes_i'),
    'tv':       (imutils.reg_tv,       imutils.reggrad_tv,       'stokes_i'),
    'tvlog':    (imutils.reg_tvlog,    imutils.reggrad_tvlog,    'stokes_i'),
    'tv2':      (imutils.reg_tv2,      imutils.reggrad_tv2,      'stokes_i'),
    'tv2log':   (imutils.reg_tv2log,   imutils.reggrad_tv2log,   'stokes_i'),
    'compact':  (imutils.reg_compact,  imutils.reggrad_compact,  'stokes_i'),
    'compact2': (imutils.reg_compact2, imutils.reggrad_compact2, 'stokes_i'),
    'rgauss':   (imutils.reg_rgauss,   imutils.reggrad_rgauss,   'stokes_i'),
    # Pol (REGULARIZERS_POL): operate on a (4, nimage) imarr.
    'msimple':  (polutils.reg_msimple, polutils.reggrad_msimple, 'pol'),
    'hw':       (polutils.reg_hw,      polutils.reggrad_hw,      'pol'),
    'ptv':      (polutils.reg_ptv,     polutils.reggrad_ptv,     'pol'),
    'vflux':    (polutils.reg_vflux,   polutils.reggrad_vflux,   'pol'),
    'l1v':      (polutils.reg_l1v,     polutils.reggrad_l1v,     'pol'),
    'l2v':      (polutils.reg_l2v,     polutils.reggrad_l2v,     'pol'),
    'vtv':      (polutils.reg_vtv,     polutils.reggrad_vtv,     'pol'),
    'vtv2':     (polutils.reg_vtv2,    polutils.reggrad_vtv2,    'pol'),
    # Multifrequency (REGULARIZERS_SPECTRAL): all `l2_*` names share one wrapper,
    # all `tv_*` names share another. compute_reg_dict pulls the right slot from
    # imcur via mfutils.spectral_slot before passing the 1D vector through.
    **{name: (mfutils.reg_l2_spec, mfutils.reggrad_l2_spec, 'mf')
       for name in REGULARIZERS_SPECIND + REGULARIZERS_CURV + REGULARIZERS_SPECIND_P
                   + REGULARIZERS_CURV_P + REGULARIZERS_RM + REGULARIZERS_CM
       if name.startswith('l2_')},
    **{name: (mfutils.reg_tv_spec, mfutils.reggrad_tv_spec, 'mf')
       for name in REGULARIZERS_SPECIND + REGULARIZERS_CURV + REGULARIZERS_SPECIND_P
                   + REGULARIZERS_CURV_P + REGULARIZERS_RM + REGULARIZERS_CM
       if name.startswith('tv_')},
}


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


class RegParams(NamedTuple):
    """Bundle of regularizer parameters passed to compute_reg_dict / compute_reggrad_dict.

    Built once per imager run by Imager._regparams() and forwarded into each
    regularizer function via ``**reg_params._asdict()``. Regularizer functions
    take what they need from this bundle; extra fields are ignored via their
    trailing ``**kwargs``.

    Field groups:
      Image scale          : flux, pflux, vflux
      Image geometry       : xdim, ydim, psize, beam_size
      Multifrequency       : mf_flux
      Term-specific kwargs : major, minor, PA, alpha_A, epsilon_tv
    """
    flux: float
    pflux: float | None
    vflux: float | None
    xdim: int
    ydim: int
    psize: float
    beam_size: float
    mf_flux: list | None         # length n_obs when REGULARIZERS_ALLFREQS_I active
    major: float
    minor: float
    PA: float
    alpha_A: float
    epsilon_tv: float


class DataWeighting(NamedTuple):
    """Bundle of data-weighting parameters passed to compute_data_tuples.

    Built once per imager run by Imager._data_weighting_params() and forwarded
    into compute_data_tuples, which reads its fields directly.

    Field groups:
      Closure-set selection   : maxset
      Noise corrections       : debias, systematic_noise, systematic_cphase_noise
      SNR filtering           : snrcut
      Weighting scheme        : weighting
      UV closure-phase filter : cp_uv_min
    """
    maxset: bool                    # use the maximal independent set of closure quantities
    debias: bool                    # apply noise-bias correction to visibility amplitudes
    snrcut: dict                    # per-data-term SNR threshold (dname -> float)
    weighting: str                  # baseline weighting scheme ('natural', 'uniform')
    systematic_noise: float         # fractional sys-noise added in quadrature to amplitudes
    systematic_cphase_noise: float  # absolute sys-noise added in quadrature to closure phases (deg)
    cp_uv_min: float | bool         # minimum UV-separation cut on closure phases (False = no cut)


class FourierGridParams(NamedTuple):
    """Bundle of Fourier-grid parameters used by ttype='fast' and ttype='nfft'.

    These configure the gridding kernel + grid padding + interpolation order
    shared between the FFT and NFFT transform paths. Built once per imager
    run by Imager._fft_params() and passed into compute_data_tuples.

    Field groups:
      Grid           : fft_pad_factor
      Gridding kernel: fft_conv_func, fft_gridder_prad
      Interpolation  : fft_interp_order
      NFFT accuracy  : nfft_eps
    """
    fft_pad_factor: float    # zero-padding factor for the Fourier grid (typical: 2)
    fft_conv_func: str       # gridding convolution kernel ('gaussian', 'pillbox', ...)
    fft_gridder_prad: float  # gridding kernel half-support in pixels
    fft_interp_order: int    # interpolation order for grid-sample lookups
    nfft_eps: float          # requested NFFT relative accuracy (finufft eps)


class MfConfig(NamedTuple):
    """Multifrequency solver configuration bundled inside ImagerConfig.

    Carries the spectral-expansion orders read by compute_which_solve and
    related multifrequency dispatch code paths. Unused when ImagerConfig.mf
    is False, but always populated with defaults.

    Field groups:
      Spectral expansion : mf_order, mf_order_pol
      RM / CM solves     : mf_rm, mf_cm
    """
    mf_order: int      # spectral-index expansion order for Stokes I (0=off, 1=alpha, 2=alpha+beta)
    mf_order_pol: int  # spectral-index expansion order for polarization (0 / 1 / 2)
    mf_rm: int         # solve for Faraday rotation measure (0=off, 1=on)
    mf_cm: int         # solve for Faraday conversion measure (0=off, 1=on)


class ImagerConfig(NamedTuple):
    """Static imaging configuration bundled at Imager construction.

    Replaces the flat self.pol_next / self.transform_next / self._ttype /
    self.mf_next / self.mf_* attrs on the Imager class. Crosses into every
    backend function that previously took these as individual args.

    Field groups:
      Polarization   : pol, transforms
      Transform type : ttype
      Multifrequency : mf, mf_config (nested MfConfig)

    JAX note: this bundle is a *static* pytree, not a traced one. The string
    leaves (pol, ttype, transform names) and the dict leaf in the sibling
    DataWeighting bundle (snrcut) aren't valid JAX traceable values. When
    jitting backend functions that take a config, pass it as a
    ``static_argname='config'`` so the structure participates in cache keying
    rather than tracing.
    """
    pol: str                       # imager polarization mode ('I', 'IP', 'IV', 'IPV', 'QU', ...)
    transforms: Sequence[str]      # bounded-value transform stack applied to imcur (e.g. ['log', 'mcv'])
    ttype: str                     # Fourier transform type ('direct', 'fast', 'nfft')
    mf: bool                       # multifrequency-imaging master flag
    mf_config: MfConfig            # nested multifrequency expansion config


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
        init_arr = init_arr.reshape((nsolve, nimage))
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


def validate_params(prior, init, config, dat_term_keys, reg_term_keys, freq_list):
    """Validate imager configuration. Raises Exception on bad config.

    Pure validator extracted from Imager.check_params. Exception messages
    are preserved verbatim from the original implementation.

    Parameters
    ----------
    prior, init : ehtim.image.Image
        Prior (regularizer reference) and initial-value images. Must share
        psize / xdim / ydim / rf / polrep.
    config : ImagerConfig
        Imager configuration bundle. See the ImagerConfig docstring for the
        field list. Reads pol, transforms, ttype, mf, mf_config.mf_order,
        mf_config.mf_order_pol.
    dat_term_keys, reg_term_keys : iterable of str
        Data term and regularizer term keys actually requested.
    freq_list : list of float
        Per-observation reference frequencies. For mf=True, must contain
        at least two distinct values.
    """
    pol = config.pol
    transforms = config.transforms
    ttype = config.ttype
    mf = config.mf
    mf_order = config.mf_config.mf_order
    mf_order_pol = config.mf_config.mf_order_pol

    if ((prior.psize != init.psize) or
        (prior.xdim != init.xdim) or
        (prior.ydim != init.ydim)):
        raise Exception("Initial image does not match dimensions of the prior image!")

    if (prior.rf != init.rf):
        raise Exception("Initial image does not have same frequency as prior image!")

    if (prior.polrep != init.polrep):
        raise Exception(
            "Initial image polrep does not match prior polrep!")

    if (prior.polrep == 'circ' and pol not in ['RR', 'LL']):
        raise Exception("Initial image polrep is 'circ': pol_next must be 'RR' or 'LL'")

    if (prior.polrep == 'stokes'
        and pol not in ['I', 'Q', 'U', 'V', 'P','IP','IQU','IV','IQUV']):
        raise Exception(
            "Initial image polrep is 'stokes': pol_next must be in 'I', 'Q', 'U', 'V', 'P','IP','IQU','IV','IQUV'!")

    if ('log' in transforms and pol in ['Q', 'U', 'V']):
        raise Exception("Cannot image Stokes Q, U, V with log image transformation!")

    if(pol in ['Q', 'U', 'V'] and
       ('gs' in reg_term_keys or 'simple' in reg_term_keys)):
        raise Exception(
            "'simple' and 'gs' regularizers do not work with Stokes Q, U, or V images!")

    if ttype not in ['fast', 'direct', 'nfft']:
        raise Exception("Possible ttype values are 'fast', 'direct','nfft'!")

    # Catch errors in multifrequency imaging setup
    if mf:
        if len(set(freq_list)) < 2:
            raise Exception("Must have observations at at least two frequencies for multifrequency imaging!")
        if mf_order not in [0,1,2]:
            raise Exception("mf_order must be in [0,1,2]!")

        if (pol in POLARIZATION_MODES):
            if pol not in ['P','QU']:
                raise Exception("Currently we only support pol_next=P for polarization multifrequency imaging!")
            if mf_order_pol not in [0,1,2]:
                raise Exception("mf_order_pol must be in [0,1,2]!")

    # Catch errors for polarimetric imaging setup
    if pol in POLARIZATION_MODES:
        if (pol in ['P', 'QU','IP','IQU']):
            if 'mcv' not in transforms:
                raise Exception(f"{pol} imaging requires 'mcv' transform!")
            if 'vcv' in transforms:
                raise Exception(f"Cannot do {pol} imaging with 'vcv' transform!")
            if 'polcv' in transforms:
                raise Exception(f"Cannot do {pol} imaging only with 'polcv' transform!")

        if (pol in ['V','IV']):
            if 'vcv' not in transforms:
                raise Exception(f"{pol} imaging requires 'vcv' transform!")
            if 'mcv' in transforms:
                raise Exception(f"Cannot do {pol} imaging only with 'mcv' transform!")
            if 'polcv' in transforms:
                raise Exception(f"Cannot do {pol} imaging only with 'polcv' transform!")

        if pol in ['IPV','IQUV']:
            if 'polcv' not in transforms:
                raise Exception("Linear+Circular polarization imaging requires 'polcv' transform!")

        if (ttype not in ["direct", "nfft"]):
            raise Exception("FFT not yet implemented in polarimetric imaging -- use NFFT!")

    # catch errors in general imaging setup
    if mf:
        if pol in POLARIZATION_MODES:
            if 'I' in pol:
                rlist = REGULARIZERS + REGULARIZERS_POL + REGULARIZERS_SPECTRAL
                dlist = DATATERMS + DATATERMS_POL
            else:
                rlist = REGULARIZERS_POL + REGULARIZERS_POLSPECTRAL
                dlist = DATATERMS_POL
        else:
            rlist = REGULARIZERS + REGULARIZERS_ISPECTRAL
            dlist = DATATERMS
    else:
        if pol in POLARIZATION_MODES:
            if 'I' in pol:
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
    for term in sorted(dat_term_keys):
        if (term is not None) and (term is not False):
            dt_here = True
        if not ((term in dlist) or (term is False)):
            dt_type = False

    st_here = False
    st_type = True
    for term in sorted(reg_term_keys):
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


def validate_limits(prior, obslist, pol, flux, mf_flux):
    """Check image-grid vs observation uv-coverage and flux limits.

    Returns a list of warning strings (does not print). Caller is
    responsible for emitting the warnings. Pure function.

    Parameters
    ----------
    prior : ehtim.image.Image
        Image whose psize / xdim / ydim define the uv-resolution and FOV.
    obslist : iterable of ehtim.obsdata.Obsdata
        Observations whose uv-distances are compared against the grid.
    pol : str
        Polarization mode. Flux-vs-maxamp warnings only fire for total-flux
        pols ('I', 'RR', 'LL').
    flux : float
        Specified total flux used as a fallback when mf_flux does not match
        the number of observations.
    mf_flux : iterable of float
        Per-observation flux overrides for multifrequency imaging. Used
        instead of `flux` when len(mf_flux) == len(obslist).

    Returns
    -------
    list of str
        One warning string per offending (obs, condition) pair. Empty if
        the grid and flux are consistent with all observations.
    """
    warnings = []
    uvmax = 1.0 / prior.psize
    uvmin = 1.0 / (prior.psize * np.max((prior.xdim, prior.ydim)))
    obslist = list(obslist)
    use_mf_flux = len(mf_flux) == len(obslist)
    for i, obs in enumerate(obslist):
        uvdists = obs.unpack('uvdist')['uvdist']
        maxbl = np.max(uvdists)
        minbl = np.max(uvdists[uvdists > 0])

        if uvmax < maxbl:
            warnings.append(
                "Warning! Pixel size is larger than smallest spatial wavelength for freq %.1f GHz!"
                % (obs.rf / 1.e9)
            )
        if uvmin > minbl:
            warnings.append(
                "Warning! Field of View is smaller than largest nonzero spatial wavelength for freq %.1f GHz!"
                % (obs.rf / 1.e9)
            )

        if pol in ['I', 'RR', 'LL']:
            maxamp = np.max(np.abs(obs.unpack('amp')['amp']))
            obs_flux = mf_flux[i] if use_mf_flux else flux
            if obs_flux > 1.2 * maxamp:
                warnings.append(
                    f"Warning! Specified flux {obs_flux:.1f} is > 120% of "
                    f"maximum visibility amplitude for freq {obs.rf/1.e9:.1f} GHz!"
                )
            if obs_flux < .8 * maxamp:
                warnings.append(
                    f"Warning! Specified flux {obs_flux:.1f} is < 80% of "
                    f"maximum visibility amplitude for freq {obs.rf/1.e9:.1f} GHz!"
                )
    return warnings


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


def compute_which_solve(config):
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
    config : ImagerConfig
        Bundled static imager config; reads pol, mf, and mf_config fields.

    Returns
    -------
    np.ndarray of int (0/1)
    """
    pol = config.pol
    mf = config.mf
    mf_order = config.mf_config.mf_order
    mf_order_pol = config.mf_config.mf_order_pol
    mf_rm = config.mf_config.mf_rm
    mf_cm = config.mf_config.mf_cm

    is_pol = pol in POLARIZATION_MODES

    if mf:
        # TODO: when we get to multifreq pol imaging, generalize this to
        # an arbitrary number of spectral terms (currently hard-coded to
        # alpha + beta). Likewise for the mf_order_pol branch below.
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


def compute_chisqdata_term(obs, prior, mask, dtype, config, **kwargs):
    """Single chisqdata dispatcher unifying chisqdata + polchisqdata.

    Standard dtypes route to imutils.chisqdata_*; pol dtypes route to
    polutils.chisqdata_*. Mask handling is asymmetric across the legacy
    leaves (standard `fast`/`nfft` do not accept a mask param); this
    function hides that.

    Parameters
    ----------
    obs : ehtim.obsdata.Obsdata
    prior : ehtim.image.Image
    mask : np.ndarray of bool
        Embedding mask. Passed positionally to direct + pol-nfft helpers;
        ignored for standard fast/nfft (those helpers do not take mask).
    dtype : str
        Must be a key of _CHISQDATA_DISPATCH (12 dtypes).
    config : ImagerConfig
        Bundled imager config; reads ttype and pol fields.
    **kwargs
        Per-dtype tuning knobs forwarded to the leaf helper. See
        imager_utils.chisqdata for the standard kwarg list and
        pol_imager_utils.chisqdata_pvis_nfft for the FFT-related kwargs.

    Returns
    -------
    (data, sigma, A) tuple, same shape as the legacy chisqdata dispatchers.
    """
    ttype = config.ttype
    pol = config.pol

    if ttype not in ('direct', 'fast', 'nfft'):
        raise Exception("Possible ttype values are 'fast', 'direct', 'nfft'!")

    try:
        by_ttype = _CHISQDATA_DISPATCH[dtype]
    except KeyError:
        raise Exception(f"data term {dtype} not recognized!")

    if ttype not in by_ttype:
        raise Exception(f"ttype={ttype!r} not supported for dtype={dtype!r}")

    helper = by_ttype[ttype]
    is_pol = dtype in DATATERMS_POL

    # Standard data terms consume a single Stokes letter ('I'/'Q'/'U'/'V'). When
    # the imager is in a multi-Stokes mode (e.g. 'IP', 'IQUV'), the leaf needs
    # pol='I' to unpack Stokes-I visibilities; pol modes without 'I' (e.g. 'P')
    # are incompatible with standard data terms.
    if not is_pol and pol in POLARIZATION_MODES:
        if 'I' not in pol:
            raise Exception(f"cannot use dterm {dtype} with pol={pol}")
        pol = 'I'

    # Only `direct` leaves take mask positionally; fast/nfft leaves index uv
    # coords from obs directly and ignore the embed mask.
    if ttype == 'direct':
        return helper(obs, prior, mask, pol=pol, **kwargs)
    return helper(obs, prior, pol=pol, **kwargs)


def _pol_solve_block(which_solve, pol):
    """Slice the polarimetric (Stokes) block from which_solve.

    Multi-frequency polarimetric `which_solve` has layout
    [I, rho, phi, psi, alpha, beta, alpha_p, beta_p, RM, CM] (10 slots);
    pol chi^2 gradient kernels only consume the first four (the Stokes
    block). Single-frequency polarimetric which_solve is already 4-wide,
    so this is a no-op there.

    TODO(Phase 6): replace with a `WhichSolve(stokes, spectral)` NamedTuple
    so arbitrary spectral layouts do not need this hardcoded slice. Flagged
    by Andrew in #227.
    """
    if pol in POLARIZATION_MODES and len(which_solve) > 4:
        return which_solve[:4]
    return which_solve


def _lookup_chisq_entry(dtype, ttype):
    """Look up (chisq_fn, chisqgrad_fn, is_pol) for (dtype, ttype). Raises on miss."""
    if ttype not in ('direct', 'fast', 'nfft'):
        raise Exception("Possible ttype values are 'fast', 'direct', 'nfft'!")
    try:
        entry = _CHISQ_DISPATCH[dtype]
    except KeyError:
        raise Exception(f"data term {dtype} not recognized!")
    if ttype not in entry:
        raise Exception(f"ttype={ttype!r} not supported for dtype={dtype!r}")
    chisq_fn, chisqgrad_fn = entry[ttype]
    return chisq_fn, chisqgrad_fn, entry['is_pol']


def compute_chisq_term(imcur, dtype, A, data, sigma, ttype='direct', mask=None):
    """Single chi^2 dispatcher unifying chisq + polchisq.

    Parameters
    ----------
    imcur : np.ndarray
        Solver-space image. Pol dtypes consume the full multi-row array
        (1D for non-pol mode, 2D (nsolve, npix) for pol/mf-pol with
        nsolve in {3, 4, 10}). Standard dtypes consume the Stokes-I row:
        `imcur[0]` when 2D, `imcur` itself when 1D.
    dtype : str
        One of the keys of _CHISQ_DISPATCH (12 dtypes).
    A, data, sigma
        Per-dtype Fourier matrix or matrix-tuple, observed values, and
        sigma values. Same shapes as the legacy leaf-helper signatures.
    ttype : {'direct', 'fast', 'nfft'}
        'fast' is not supported for pol dtypes.
    mask : np.ndarray of bool, optional
        Embedding mask used by fast / nfft leaves to expand the solver
        image back onto the full grid. None or empty => no embedding.

    Returns
    -------
    float
    """
    chisq_fn, _, is_pol = _lookup_chisq_entry(dtype, ttype)

    imvec = imcur if (is_pol or imcur.ndim == 1) else imcur[0]

    if (ttype != 'direct' and mask is not None
            and len(mask) > 0 and np.any(np.invert(mask))):
        if is_pol:
            imvec = imutils.embed_imarr(imvec, mask, randomfloor=EMBED_RANDOMFLOOR)
        else:
            imvec = imutils.embed(imvec, mask, randomfloor=EMBED_RANDOMFLOOR)

    if ttype == 'fast' and dtype not in _DIAG_DTYPES:
        imvec = obsh.fft_imvec(imvec, A[0])

    return chisq_fn(imvec, A, data, sigma)


def compute_chisqgrad_term(imcur, dtype, A, data, sigma, ttype='direct',
                           mask=None, pol_solve=None):
    """Single chi^2-gradient dispatcher. See compute_chisq_term.

    For pol dtypes, `pol_solve` (the Stokes block of which_solve) must be
    supplied explicitly -- there is no principled default since the right
    value depends on the imager's pol mode. Standard grad helpers ignore
    `pol_solve`. Returned gradient is sliced back through `mask` for
    fast / nfft to match the legacy chisqgrad / polchisqgrad return shapes:
    pol grads are (4, n_masked); standard grads are (n_masked,).
    """
    _, chisqgrad_fn, is_pol = _lookup_chisq_entry(dtype, ttype)

    imvec = imcur if (is_pol or imcur.ndim == 1) else imcur[0]
    has_partial_mask = (mask is not None and len(mask) > 0
                        and np.any(np.invert(mask)))

    if ttype != 'direct' and has_partial_mask:
        if is_pol:
            imvec = imutils.embed_imarr(imvec, mask, randomfloor=EMBED_RANDOMFLOOR)
        else:
            imvec = imutils.embed(imvec, mask, randomfloor=EMBED_RANDOMFLOOR)

    if ttype == 'fast' and dtype not in _DIAG_DTYPES:
        imvec = obsh.fft_imvec(imvec, A[0])

    if is_pol:
        if pol_solve is None:
            raise Exception(
                f"compute_chisqgrad_term requires explicit pol_solve for "
                f"polarimetric dtype {dtype!r}"
            )
        grad = chisqgrad_fn(imvec, A, data, sigma, pol_solve)
    else:
        grad = chisqgrad_fn(imvec, A, data, sigma)

    if ttype != 'direct' and has_partial_mask:
        grad = grad[:, mask] if is_pol else grad[mask]

    return grad


def compute_regularizer_term(imvec_or_imarr, regname, mask, **kwargs):
    """Single regularizer-value dispatcher.

    Parameters
    ----------
    imvec_or_imarr : np.ndarray
        1D image vector for Stokes-I and mf regularizers; (4, nimage) array for pol regularizers.
    regname : str
        Regularizer name. Must be a key of _REGULARIZER_DISPATCH.
    mask : np.ndarray of bool
        Pixel embedding mask. Spatial regularizers embed through this before computing.
    **kwargs
        Per-regularizer parameters (flux, xdim, ydim, psize, beam_size, alpha_A,
        epsilon_tv, major, minor, PA, nprior, priorarr, vflux, pflux, pol_solve).
        Each regularizer takes what it needs and ignores the rest.

    Returns
    -------
    float
        Regularizer value, negated so positive values indicate a penalty to minimize.
    """
    if regname not in _REGULARIZER_DISPATCH:
        raise Exception(f"regularizer term {regname} not recognized!")
    val_fn, _, _ = _REGULARIZER_DISPATCH[regname]
    return val_fn(imvec_or_imarr, mask=mask, **kwargs)


def compute_regularizergrad_term(imvec_or_imarr, regname, mask, **kwargs):
    """Single regularizer-gradient dispatcher.

    Parameters
    ----------
    imvec_or_imarr : np.ndarray
        1D image vector for Stokes-I and mf regularizers; (4, nimage) array for pol regularizers.
    regname : str
        Regularizer name. Must be a key of _REGULARIZER_DISPATCH.
    mask : np.ndarray of bool
        Pixel embedding mask. Spatial regularizers embed through this before computing
        and slice the gradient back through ``mask`` after.
    **kwargs
        Per-regularizer parameters; see compute_regularizer_term.

    Returns
    -------
    np.ndarray
        Gradient with shape (nimage,) for Stokes-I and mf regularizers, or
        (4, nimage) for pol regularizers. Bundling a Stokes-I gradient into
        the (4, nimage) shape for pol-mode imaging is the caller's responsibility.
    """
    if regname not in _REGULARIZER_DISPATCH:
        raise Exception(f"regularizer term {regname} not recognized!")
    _, grad_fn, _ = _REGULARIZER_DISPATCH[regname]
    return grad_fn(imvec_or_imarr, mask=mask, **kwargs)


def compute_data_tuples(obslist, prior, embed_mask, dat_term_keys, config,
                        data_weighting, fourier_grid):
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
    config : ImagerConfig
        Provides pol and ttype for dispatch.
    data_weighting : DataWeighting
        Per-data-term knobs forwarded to imutils.chisqdata. See the DataWeighting
        docstring for the field list.
    fourier_grid : FourierGridParams
        Gridding parameters for ttype='fast' / 'nfft'. See the FourierGridParams
        docstring for the field list.

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
            data_tuples[dname_key] = compute_chisqdata_term(
                obs, prior, embed_mask, dname, config,
                maxset=data_weighting.maxset,
                debias=data_weighting.debias,
                snrcut=data_weighting.snrcut[dname],
                weighting=data_weighting.weighting,
                systematic_noise=data_weighting.systematic_noise,
                systematic_cphase_noise=data_weighting.systematic_cphase_noise,
                cp_uv_min=data_weighting.cp_uv_min,
                order=fourier_grid.fft_interp_order,
                fft_pad_factor=fourier_grid.fft_pad_factor,
                conv_func=fourier_grid.fft_conv_func,
                p_rad=fourier_grid.fft_gridder_prad,
                nfft_eps=fourier_grid.nfft_eps,
            )

    return data_tuples


def compute_init_state(
    obslist, init_image, prior_image,
    freq_list, reffreq,
    config,
    norm_init, flux, clipfloor,
    dat_term_keys,
    data_weighting, fourier_grid,
    *, compute_data=True, prior_data_tuples=None,
):
    """Build solver-ready imager state. Pure function.

    Composes compute_embed + compute_logfreqratios + compute_which_solve +
    make_initarr + transform_imarr_inverse + pack_imarr + compute_data_tuples
    into the single state bundle consumed by compute_objective /
    compute_objective_grad.

    JAX note: when jitted, expect static_argnames=('config', 'dat_term_keys',
    'compute_data') with config as a frozen pytree.

    Parameters
    ----------
    obslist : list of Obsdata
    init_image, prior_image : Image
        Initial image (L-BFGS-B start point) and regularizer prior image.
    freq_list : list of float
        Reference frequencies (Hz) of each obs in obslist.
    reffreq : float
        Reference frequency (Hz) of the multi-frequency expansion. When
        config.mf=True this is overridden by init_image.rf.
    config : ImagerConfig
        Static imager configuration. Provides pol, transforms, ttype, mf,
        and the nested mf_config bundle.
    norm_init : bool
    flux, clipfloor : float
    dat_term_keys : iterable of str
        Sorted dat_term names. Each must be in DATATERMS or DATATERMS_POL.
    data_weighting : DataWeighting
        Per-data-term knobs forwarded to compute_data_tuples.
    fourier_grid : FourierGridParams
        Gridding parameters forwarded to compute_data_tuples.

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
    pol = config.pol
    mf = config.mf
    transforms = config.transforms
    ttype = config.ttype
    mf_cfg = config.mf_config

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

    which_solve = compute_which_solve(config)

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
            obslist, prior_image, embed_mask, dat_term_keys, config,
            data_weighting, fourier_grid,
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


def compute_chisq_dict(imcur, dat_term_keys, config,
                       data_tuples, logfreqratio_list, n_obs,
                       embed_mask):
    """Compute chi^2 value for each data term across all observations.

    Parameters
    ----------
    imcur : np.ndarray
        Current image array transformed to bounded values.
    dat_term_keys : list of str
        Data term names to evaluate, already sorted.
    config : ImagerConfig
        Bundled imager config; reads mf, pol, ttype fields.
    data_tuples : dict
        Pre-computed data products keyed by dname or dname_i,
        each value is a (data, sigma, A) tuple.
    logfreqratio_list : list of float
        Log frequency ratios log(nu_i/reffreq); one per obs.
    n_obs : int
        Number of observations (frequencies/epochs). Must equal
        len(logfreqratio_list); validated by Imager.init_imager.
    embed_mask : np.ndarray of bool
        Pixel embedding mask.

    Returns
    -------
    chi2_dict : dict
        Mapping from dname (or dname_i for multi-obs) to chi^2 scalar.
    """
    mf = config.mf
    ttype = config.ttype

    chi2_dict = {}
    for dname in dat_term_keys:
        for i in range(n_obs):
            dname_key = dname if n_obs == 1 else f"{dname}_{i}"
            data, sigma, A = data_tuples[dname_key]
            imcur_nu = (
                mfutils.image_at_freq(imcur, logfreqratio_list[i])
                if mf else imcur
            )
            chi2_dict[dname_key] = compute_chisq_term(
                imcur_nu, dname, A, data, sigma,
                ttype=ttype, mask=embed_mask,
            )

    return chi2_dict


def compute_chisqgrad_dict(imcur, dat_term_keys, config,
                           data_tuples, logfreqratio_list, n_obs,
                           embed_mask,
                           which_solve, nimage):
    """Compute chi^2 gradient for each data term across all observations.

    Parameters
    ----------
    imcur : np.ndarray
        Current image array transformed to bounded values.
    dat_term_keys : list of str
        Data term names to evaluate, already sorted.
    config : ImagerConfig
        Bundled imager config; reads mf, pol, ttype fields.
    data_tuples : dict
        Pre-computed data products keyed by dname or dname_i,
        each value is a (data, sigma, A) tuple.
    logfreqratio_list : list of float
        Log frequency ratios log(nu_i/reffreq); one per obs.
    n_obs : int
        Number of observations (frequencies/epochs). Must equal
        len(logfreqratio_list); validated by Imager.init_imager.
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
    mf = config.mf
    pol = config.pol
    ttype = config.ttype

    chi2grad_dict = {}
    pol_solve = _pol_solve_block(which_solve, pol)
    # np.array((...)) below copies, so sharing zero_row across iterations is safe.
    zero_row = np.zeros(nimage)
    is_pol_mode = pol in POLARIZATION_MODES
    for dname in dat_term_keys:
        for i in range(n_obs):
            dname_key = dname if n_obs == 1 else f"{dname}_{i}"
            data, sigma, A = data_tuples[dname_key]
            imcur_nu = (
                mfutils.image_at_freq(imcur, logfreqratio_list[i])
                if mf else imcur
            )

            chi2grad = compute_chisqgrad_term(
                imcur_nu, dname, A, data, sigma,
                ttype=ttype, mask=embed_mask, pol_solve=pol_solve,
            )

            # Pad standard (nimage,) grad to (4, nimage) Stokes block so the
            # regularizer + mf chains see a uniform layout.
            if dname in DATATERMS and is_pol_mode:
                chi2grad = np.array((chi2grad, zero_row, zero_row, zero_row))

            if mf:
                chi2grad = mfutils.mf_all_grads_chain(
                    chi2grad, imcur_nu, imcur, logfreqratio_list[i],
                )

            chi2grad_dict[dname_key] = np.array(chi2grad)

    return chi2grad_dict


def compute_reg_dict(imcur, reg_term_keys, config,
                     logfreqratio_list, n_obs,
                     priorvec, norm_reg, reg_params,
                     embed_mask):
    """Compute regularizer value for each regularizer term.

    Parameters
    ----------
    imcur : np.ndarray
        Current image array transformed to bounded values.
    reg_term_keys : list of str
        Regularizer term names to evaluate, already sorted.
    config : ImagerConfig
        Bundled imager config; reads mf and pol fields.
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
    reg_params : RegParams
        Bundle of regularizer parameters. See the RegParams docstring for the
        field list. Forwarded into each regularizer call via
        ``**reg_params._asdict()``.
    embed_mask : np.ndarray of bool
        Pixel embedding mask.

    Returns
    -------
    reg_dict : dict
        Mapping from regname to regularizer scalar.
    """
    mf = config.mf
    pol = config.pol

    mf_flux = reg_params.mf_flux
    reg_kwargs = reg_params._asdict()
    reg_dict = {}

    for regname in reg_term_keys:

        if mf:
            if regname in REGULARIZERS_POL:
                reg = compute_regularizer_term(imcur[0:4], regname, embed_mask,
                                               norm_reg=norm_reg, **reg_kwargs)

            elif regname in REGULARIZERS:
                if regname in REGULARIZERS_ALLFREQS_I:
                    if (not isinstance(mf_flux, list)) or len(mf_flux) != n_obs:
                        raise Exception(f"when using regularizer '{regname}', "
                                        + "mf_flux must be a list of same length as n_obs!")

                    regname_base = '_'.join(regname.split('_')[:-1])  # strip '_mf' tag
                    reg = 0
                    for i in range(n_obs):
                        logfreqratio = logfreqratio_list[i]
                        imcur_nu = mfutils.image_at_freq(imcur, logfreqratio)
                        prior_nu = mfutils.image_at_freq(priorvec, logfreqratio)
                        reg = reg + compute_regularizer_term(
                            imcur_nu, regname_base, embed_mask,
                            nprior=prior_nu, norm_reg=norm_reg,
                            **{**reg_kwargs, 'flux': mf_flux[i]})
                else:
                    reg = compute_regularizer_term(imcur[0], regname, embed_mask,
                                                   nprior=priorvec[0],
                                                   norm_reg=norm_reg, **reg_kwargs)

            elif regname in REGULARIZERS_SPECTRAL:
                idx = mfutils.spectral_slot(regname, len(imcur))
                reg = compute_regularizer_term(imcur[idx], regname, embed_mask,
                                               nprior=priorvec[idx],
                                               norm_reg=norm_reg, **reg_kwargs)
            else:
                raise Exception(f"regularizer term {regname} not recognized!")

        elif regname in REGULARIZERS_POL:
            reg = compute_regularizer_term(imcur, regname, embed_mask,
                                           norm_reg=norm_reg, **reg_kwargs)

        elif regname in REGULARIZERS:
            if pol in POLARIZATION_MODES:
                imcur0, prior0 = imcur[0], priorvec[0]
            else:
                imcur0, prior0 = imcur, priorvec
            reg = compute_regularizer_term(imcur0, regname, embed_mask,
                                           nprior=prior0,
                                           norm_reg=norm_reg, **reg_kwargs)
        else:
            raise Exception(f"regularizer term {regname} not recognized!")

        reg_dict[regname] = reg

    return reg_dict


def compute_reggrad_dict(imcur, reg_term_keys, config,
                         logfreqratio_list, n_obs,
                         priorvec, norm_reg, reg_params,
                         embed_mask,
                         which_solve, nimage):
    """Compute regularizer gradient for each regularizer term.

    Parameters
    ----------
    imcur, reg_term_keys, config, logfreqratio_list, n_obs,
    priorvec, norm_reg, reg_params, embed_mask : see compute_reg_dict.
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
    mf = config.mf
    pol = config.pol

    mf_flux = reg_params.mf_flux
    reg_kwargs = reg_params._asdict()
    reggrad_dict = {}

    for regname in reg_term_keys:

        if mf:
            if regname in REGULARIZERS_POL:
                regp = compute_regularizergrad_term(
                    imcur[0:4], regname, embed_mask,
                    pol_solve=which_solve[0:4],
                    norm_reg=norm_reg, **reg_kwargs)
                reggrad = np.zeros((len(imcur), nimage))
                reggrad[0:4] = regp

            elif regname in REGULARIZERS:
                if regname in REGULARIZERS_ALLFREQS_I:
                    if (not isinstance(mf_flux, list)) or len(mf_flux) != n_obs:
                        raise Exception(f"when using regularizer '{regname}', "
                                        + "mf_flux must be a list of same length as n_obs!")

                    regname_base = '_'.join(regname.split('_')[:-1])  # strip '_mf' tag
                    reggrad = 0
                    for i in range(n_obs):
                        logfreqratio = logfreqratio_list[i]
                        imcur_nu = mfutils.image_at_freq(imcur, logfreqratio)
                        prior_nu = mfutils.image_at_freq(priorvec, logfreqratio)
                        regi = compute_regularizergrad_term(
                            imcur_nu, regname_base, embed_mask,
                            nprior=prior_nu, norm_reg=norm_reg,
                            **{**reg_kwargs, 'flux': mf_flux[i]})
                        reggrad = reggrad + mfutils.mf_all_grads_chain(
                            regi, imcur_nu, imcur, logfreqratio)
                else:
                    regi = compute_regularizergrad_term(
                        imcur[0], regname, embed_mask,
                        nprior=priorvec[0], norm_reg=norm_reg, **reg_kwargs)
                    reggrad = np.zeros((len(imcur), nimage))
                    reggrad[0] = regi

            elif regname in REGULARIZERS_SPECTRAL:
                idx = mfutils.spectral_slot(regname, len(imcur))
                regmf = compute_regularizergrad_term(
                    imcur[idx], regname, embed_mask,
                    nprior=priorvec[idx], norm_reg=norm_reg, **reg_kwargs)
                reggrad = np.zeros((len(imcur), nimage))
                reggrad[idx] = regmf
            else:
                raise Exception(f"regularizer term {regname} not recognized!")

        elif regname in REGULARIZERS_POL:
            reggrad = compute_regularizergrad_term(
                imcur, regname, embed_mask,
                pol_solve=which_solve, norm_reg=norm_reg, **reg_kwargs)

        elif regname in REGULARIZERS:
            if pol in POLARIZATION_MODES:
                imcur0, prior0 = imcur[0], priorvec[0]
            else:
                imcur0, prior0 = imcur, priorvec
            reggrad = compute_regularizergrad_term(
                imcur0, regname, embed_mask,
                nprior=prior0, norm_reg=norm_reg, **reg_kwargs)
            # In polarization mode, pad the Stokes-I gradient into a (4, nimage)
            # array with zeros for Q, U, V slots so it can be summed alongside
            # polregularizergrad outputs.
            if pol in POLARIZATION_MODES:
                reggrad = np.array((reggrad,
                                    np.zeros(nimage),
                                    np.zeros(nimage),
                                    np.zeros(nimage)))
        else:
            raise Exception(f"regularizer term {regname} not recognized!")

        reggrad_dict[regname] = reggrad

    return reggrad_dict


def compute_objective(imvec, initvec, config,
                      which_solve, data_tuples, logfreqratio_list, n_obs,
                      dat_term, reg_term,
                      priorvec, norm_reg, reg_params,
                      embed_mask):
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
    config : ImagerConfig
        Bundled imager config; reads pol, mf, transforms, ttype fields.
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
    reg_params : RegParams
        Bundle of regularizer parameters. See compute_reg_dict.
    embed_mask : np.ndarray of bool
        Pixel embedding mask.

    Returns
    -------
    cost : float
        Scalar objective value, sum of weighted chi^2 deviations and
        regularizer contributions.
    """
    transforms = config.transforms

    dat_term_keys = sorted(dat_term.keys())
    reg_term_keys = sorted(reg_term.keys())

    # Unpack solver vector into image array, then apply bounded-value transform.
    # `initvec` provides the values used for any not-solved-for slots (it is
    # the *initial* image, NOT the regularizer prior `priorvec`).
    imcur = unpack_imarr(imvec, initvec, which_solve)
    imcur = transform_imarr(imcur, transforms, which_solve)

    chi2_dict = compute_chisq_dict(
        imcur, dat_term_keys, config,
        data_tuples, logfreqratio_list, n_obs,
        embed_mask,
    )
    reg_dict = compute_reg_dict(
        imcur, reg_term_keys, config,
        logfreqratio_list, n_obs,
        priorvec, norm_reg, reg_params,
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


def compute_objective_grad(imvec, initvec, config,
                           which_solve, data_tuples, logfreqratio_list, n_obs,
                           dat_term, reg_term,
                           priorvec, norm_reg, reg_params,
                           embed_mask, nimage):
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
    imvec, initvec, config, which_solve, data_tuples, logfreqratio_list,
    n_obs, dat_term, reg_term, priorvec, norm_reg, reg_params,
    embed_mask : see compute_objective.
    nimage : int
        Number of active pixels (sum of embed_mask). Used to size the
        pre-pack gradient array.

    Returns
    -------
    grad : np.ndarray
        1D gradient vector (same length as imvec), suitable as the `jac`
        argument to scipy.optimize.minimize.
    """
    transforms = config.transforms

    dat_term_keys = sorted(dat_term.keys())
    reg_term_keys = sorted(reg_term.keys())

    # Unpack solver vector; keep a pre-transform copy for the chain rule.
    # `initvec` is the initial image (for not-solved-for slots), distinct
    # from `priorvec` which is the regularizer prior.
    imcur = unpack_imarr(imvec, initvec, which_solve)
    imcur_prime = imcur.copy()
    imcur = transform_imarr(imcur, transforms, which_solve)

    chi2grad_dict = compute_chisqgrad_dict(
        imcur, dat_term_keys, config,
        data_tuples, logfreqratio_list, n_obs,
        embed_mask,
        which_solve, nimage,
    )

    reggrad_dict = compute_reggrad_dict(
        imcur, reg_term_keys, config,
        logfreqratio_list, n_obs,
        priorvec, norm_reg, reg_params,
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
