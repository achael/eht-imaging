# imager_utils.py
# General imager functions for total intensity VLBI data
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




import matplotlib.pyplot as plt
import numpy as np

import ehtim.const_def as ehc
import ehtim.observing.obs_helpers as obsh

##################################################################################################
# Constants & Definitions
##################################################################################################

NORM_REGULARIZER = True
DATATERMS = ['vis', 'bs', 'amp', 'cphase', 'cphase_diag',
             'camp', 'logcamp', 'logcamp_diag', 'logamp']
REGULARIZERS = ['gs', 'tv', 'tvlog','tv2', 'tv2log','l1w', 'lA', 'patch', 'simple', 'compact', 'compact2', 'rgauss']

nit = 0  # global variable to track the iteration number in the plotting callback

##################################################################################################
# Wrapper Functions
##################################################################################################


def chisq(imvec, A, data, sigma, dtype, ttype='direct', mask=None):
    """Return chi^2 for a standard data term.

    Thin shim around imager_backend.compute_chisq_term retained for backward
    compatibility. New code should call compute_chisq_term directly.
    """
    # Imported here to avoid a module-load cycle with pol_imager_utils.
    from ehtim.imaging.imager_backend import compute_chisq_term
    if dtype not in DATATERMS:
        raise Exception(f"data term {dtype!r} is not a standard data term")
    return compute_chisq_term(imvec, dtype, A, data, sigma,
                              ttype=ttype, mask=mask)


def chisqgrad(imvec, A, data, sigma, dtype, ttype='direct', mask=None):
    """Return chi^2 gradient for a standard data term.

    Thin shim around imager_backend.compute_chisqgrad_term retained for
    backward compatibility. New code should call compute_chisqgrad_term
    directly.
    """
    from ehtim.imaging.imager_backend import compute_chisqgrad_term
    if dtype not in DATATERMS:
        raise Exception(f"data term {dtype!r} is not a standard data term")
    return compute_chisqgrad_term(imvec, dtype, A, data, sigma,
                                  ttype=ttype, mask=mask)


def regularizer(imvec, nprior, mask, flux, xdim, ydim, psize, stype, **kwargs):
    """Return the regularizer value for a Stokes-I regularizer.

    Thin shim around imager_backend.compute_regularizer_term retained for
    backward compatibility. New code should call compute_regularizer_term
    directly.
    """
    from ehtim.imaging.imager_backend import REGULARIZERS as _BACKEND_REGS
    from ehtim.imaging.imager_backend import compute_regularizer_term
    if stype not in _BACKEND_REGS:
        raise Exception(f"regularizer term {stype!r} is not a Stokes-I regularizer")
    return compute_regularizer_term(imvec, stype, mask,
                                    nprior=nprior, flux=flux,
                                    xdim=xdim, ydim=ydim, psize=psize, **kwargs)


def regularizergrad(imvec, nprior, mask, flux, xdim, ydim, psize, stype, **kwargs):
    """Return the regularizer gradient for a Stokes-I regularizer.

    Thin shim around imager_backend.compute_regularizergrad_term retained for
    backward compatibility. New code should call compute_regularizergrad_term
    directly.
    """
    from ehtim.imaging.imager_backend import REGULARIZERS as _BACKEND_REGS
    from ehtim.imaging.imager_backend import compute_regularizergrad_term
    if stype not in _BACKEND_REGS:
        raise Exception(f"regularizer term {stype!r} is not a Stokes-I regularizer")
    return compute_regularizergrad_term(imvec, stype, mask,
                                        nprior=nprior, flux=flux,
                                        xdim=xdim, ydim=ydim, psize=psize, **kwargs)


def chisqdata(Obsdata, Prior, mask, dtype, pol='I', **kwargs):
    """Return (data, sigma, A) for a standard data term.

    Thin shim around imager_backend.compute_chisqdata_term retained for
    backward compatibility. New code should call compute_chisqdata_term
    directly.
    """
    from ehtim.imaging.imager_backend import (
        ImagerConfig,
        MfConfig,
        compute_chisqdata_term,
    )
    ttype = kwargs.pop('ttype', 'direct')
    if dtype not in DATATERMS:
        raise Exception(f"data term {dtype!r} is not a standard data term")
    config = ImagerConfig(
        pol=pol, transforms=[], ttype=ttype, mf=False,
        mf_config=MfConfig(mf_order=0, mf_order_pol=0, mf_rm=0, mf_cm=0),
    )
    return compute_chisqdata_term(Obsdata, Prior, mask, dtype, config, **kwargs)


##################################################################################################
# DFT Chi-squared and Gradient Functions
##################################################################################################

def chisq_vis(imvec, Amatrix, vis, sigma):
    """Visibility chi-squared"""

    samples = np.dot(Amatrix, imvec)
    chisq = np.sum(np.abs((samples-vis)/sigma)**2)/(2*len(vis))
    return chisq

def chisqgrad_vis(imvec, Amatrix, vis, sigma):
    """The gradient of the visibility chi-squared"""

    samples = np.dot(Amatrix, imvec)
    wdiff = (vis - samples)/(sigma**2)

    out = -np.real(np.dot(Amatrix.conj().T, wdiff))/len(vis)
    return out


def chisq_amp(imvec, A, amp, sigma):
    """Visibility Amplitudes (normalized) chi-squared"""

    amp_samples = np.abs(np.dot(A, imvec))
    return np.sum(np.abs((amp - amp_samples)/sigma)**2)/len(amp)


def chisqgrad_amp(imvec, A, amp, sigma):
    """The gradient of the amplitude chi-squared"""

    i1 = np.dot(A, imvec)
    amp_samples = np.abs(i1)

    pp = ((amp - amp_samples) * amp_samples) / (sigma**2) / i1
    out = (-2.0/len(amp)) * np.real(np.dot(pp, A))
    return out


def chisq_bs(imvec, Amatrices, bis, sigma):
    """Bispectrum chi-squared"""

    bisamples = (np.dot(Amatrices[0], imvec) *
                 np.dot(Amatrices[1], imvec) *
                 np.dot(Amatrices[2], imvec))
    chisq = np.sum(np.abs((bis - bisamples)/sigma)**2)/(2.*len(bis))
    return chisq


def chisqgrad_bs(imvec, Amatrices, bis, sigma):
    """The gradient of the bispectrum chi-squared"""

    bisamples = (np.dot(Amatrices[0], imvec) *
                 np.dot(Amatrices[1], imvec) *
                 np.dot(Amatrices[2], imvec))

    wdiff = ((bis - bisamples).conj())/(sigma**2)
    pt1 = wdiff * np.dot(Amatrices[1], imvec) * np.dot(Amatrices[2], imvec)
    pt2 = wdiff * np.dot(Amatrices[0], imvec) * np.dot(Amatrices[2], imvec)
    pt3 = wdiff * np.dot(Amatrices[0], imvec) * np.dot(Amatrices[1], imvec)
    out = (np.dot(pt1, Amatrices[0]) +
           np.dot(pt2, Amatrices[1]) +
           np.dot(pt3, Amatrices[2]))

    out = -np.real(out) / len(bis)
    return out


def chisq_cphase(imvec, Amatrices, clphase, sigma):
    """Closure Phases (normalized) chi-squared"""
    clphase = clphase * ehc.DEGREE
    sigma = sigma * ehc.DEGREE

    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    clphase_samples = np.angle(i1 * i2 * i3)

    chisq = (2.0/len(clphase)) * np.sum((1.0 - np.cos(clphase-clphase_samples))/(sigma**2))
    return chisq


def chisqgrad_cphase(imvec, Amatrices, clphase, sigma):
    """The gradient of the closure phase chi-squared"""
    clphase = clphase * ehc.DEGREE
    sigma = sigma * ehc.DEGREE

    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    clphase_samples = np.angle(i1 * i2 * i3)

    pref = np.sin(clphase - clphase_samples)/(sigma**2)
    pt1 = pref/i1
    pt2 = pref/i2
    pt3 = pref/i3
    out = np.dot(pt1, Amatrices[0]) + np.dot(pt2, Amatrices[1]) + np.dot(pt3, Amatrices[2])
    out = (-2.0/len(clphase)) * np.imag(out)
    return out


def chisq_cphase_diag(imvec, Amatrices, clphase_diag, sigma):
    """Diagonalized closure phases (normalized) chi-squared"""
    clphase_diag = np.concatenate(clphase_diag) * ehc.DEGREE
    sigma = np.concatenate(sigma) * ehc.DEGREE

    A3_diag = Amatrices[0]
    tform_mats = Amatrices[1]

    clphase_diag_samples = []
    for iA, A3 in enumerate(A3_diag):
        clphase_samples = np.angle(np.dot(A3[0], imvec) *
                                   np.dot(A3[1], imvec) *
                                   np.dot(A3[2], imvec))
        clphase_diag_samples.append(np.dot(tform_mats[iA], clphase_samples))
    clphase_diag_samples = np.concatenate(clphase_diag_samples)

    chisq = np.sum((1.0 - np.cos(clphase_diag-clphase_diag_samples))/(sigma**2))
    chisq *= (2.0/len(clphase_diag))
    return chisq


def chisqgrad_cphase_diag(imvec, Amatrices, clphase_diag, sigma):
    """The gradient of the diagonalized closure phase chi-squared"""
    clphase_diag = clphase_diag * ehc.DEGREE
    sigma = sigma * ehc.DEGREE

    A3_diag = Amatrices[0]
    tform_mats = Amatrices[1]

    deriv = np.zeros_like(imvec)
    for iA, A3 in enumerate(A3_diag):

        i1 = np.dot(A3[0], imvec)
        i2 = np.dot(A3[1], imvec)
        i3 = np.dot(A3[2], imvec)
        clphase_samples = np.angle(i1 * i2 * i3)
        clphase_diag_samples = np.dot(tform_mats[iA], clphase_samples)

        clphase_diag_measured = clphase_diag[iA]
        clphase_diag_sigma = sigma[iA]

        term1 = np.dot(np.dot((np.sin(clphase_diag_measured-clphase_diag_samples) /
                               (clphase_diag_sigma**2.0)), (tform_mats[iA]/i1)), A3[0])
        term2 = np.dot(np.dot((np.sin(clphase_diag_measured-clphase_diag_samples) /
                               (clphase_diag_sigma**2.0)), (tform_mats[iA]/i2)), A3[1])
        term3 = np.dot(np.dot((np.sin(clphase_diag_measured-clphase_diag_samples) /
                               (clphase_diag_sigma**2.0)), (tform_mats[iA]/i3)), A3[2])
        deriv += -2.0*np.imag(term1 + term2 + term3)

    deriv *= 1.0/float(len(np.concatenate(clphase_diag)))

    return deriv


def chisq_camp(imvec, Amatrices, clamp, sigma):
    """Closure Amplitudes (normalized) chi-squared"""

    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    i4 = np.dot(Amatrices[3], imvec)
    clamp_samples = np.abs((i1 * i2)/(i3 * i4))

    chisq = np.sum(np.abs((clamp - clamp_samples)/sigma)**2)/len(clamp)
    return chisq


def chisqgrad_camp(imvec, Amatrices, clamp, sigma):
    """The gradient of the closure amplitude chi-squared"""

    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    i4 = np.dot(Amatrices[3], imvec)
    clamp_samples = np.abs((i1 * i2)/(i3 * i4))

    pp = ((clamp - clamp_samples) * clamp_samples)/(sigma**2)
    pt1 = pp/i1
    pt2 = pp/i2
    pt3 = -pp/i3
    pt4 = -pp/i4

    out = (np.dot(pt1, Amatrices[0]) +
           np.dot(pt2, Amatrices[1]) +
           np.dot(pt3, Amatrices[2]) +
           np.dot(pt4, Amatrices[3]))
    out = (-2.0/len(clamp)) * np.real(out)
    return out


def chisq_logcamp(imvec, Amatrices, log_clamp, sigma):
    """Log Closure Amplitudes (normalized) chi-squared"""

    a1 = np.abs(np.dot(Amatrices[0], imvec))
    a2 = np.abs(np.dot(Amatrices[1], imvec))
    a3 = np.abs(np.dot(Amatrices[2], imvec))
    a4 = np.abs(np.dot(Amatrices[3], imvec))

    samples = np.log(a1) + np.log(a2) - np.log(a3) - np.log(a4)
    chisq = np.sum(np.abs((log_clamp - samples)/sigma)**2) / (len(log_clamp))
    return chisq


def chisqgrad_logcamp(imvec, Amatrices, log_clamp, sigma):
    """The gradient of the Log closure amplitude chi-squared"""

    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    i4 = np.dot(Amatrices[3], imvec)
    log_clamp_samples = (np.log(np.abs(i1)) +
                         np.log(np.abs(i2)) -
                         np.log(np.abs(i3)) -
                         np.log(np.abs(i4)))

    pp = (log_clamp - log_clamp_samples) / (sigma**2)
    pt1 = pp / i1
    pt2 = pp / i2
    pt3 = -pp / i3
    pt4 = -pp / i4
    out = (np.dot(pt1, Amatrices[0]) +
           np.dot(pt2, Amatrices[1]) +
           np.dot(pt3, Amatrices[2]) +
           np.dot(pt4, Amatrices[3]))
    out = (-2.0/len(log_clamp)) * np.real(out)
    return out


def chisq_logcamp_diag(imvec, Amatrices, log_clamp_diag, sigma):
    """Diagonalized log closure amplitudes (normalized) chi-squared"""

    log_clamp_diag = np.concatenate(log_clamp_diag)
    sigma = np.concatenate(sigma)

    A4_diag = Amatrices[0]
    tform_mats = Amatrices[1]

    log_clamp_diag_samples = []
    for iA, A4 in enumerate(A4_diag):

        a1 = np.abs(np.dot(A4[0], imvec))
        a2 = np.abs(np.dot(A4[1], imvec))
        a3 = np.abs(np.dot(A4[2], imvec))
        a4 = np.abs(np.dot(A4[3], imvec))

        log_clamp_samples = np.log(a1) + np.log(a2) - np.log(a3) - np.log(a4)
        log_clamp_diag_samples.append(np.dot(tform_mats[iA], log_clamp_samples))

    log_clamp_diag_samples = np.concatenate(log_clamp_diag_samples)

    chisq = np.sum(np.abs((log_clamp_diag - log_clamp_diag_samples)/sigma)**2)
    chisq /= (len(log_clamp_diag))

    return chisq


def chisqgrad_logcamp_diag(imvec, Amatrices, log_clamp_diag, sigma):
    """The gradient of the diagonalized log closure amplitude chi-squared"""

    A4_diag = Amatrices[0]
    tform_mats = Amatrices[1]

    deriv = np.zeros_like(imvec)
    for iA, A4 in enumerate(A4_diag):

        i1 = np.dot(A4[0], imvec)
        i2 = np.dot(A4[1], imvec)
        i3 = np.dot(A4[2], imvec)
        i4 = np.dot(A4[3], imvec)
        log_clamp_samples = np.log(np.abs(i1)) + np.log(np.abs(i2)) - \
            np.log(np.abs(i3)) - np.log(np.abs(i4))
        log_clamp_diag_samples = np.dot(tform_mats[iA], log_clamp_samples)

        log_clamp_diag_measured = log_clamp_diag[iA]
        log_clamp_diag_sigma = sigma[iA]

        term1 = np.dot(np.dot(((log_clamp_diag_measured-log_clamp_diag_samples) /
                               (log_clamp_diag_sigma**2.0)), (tform_mats[iA]/i1)), A4[0])
        term2 = np.dot(np.dot(((log_clamp_diag_measured-log_clamp_diag_samples) /
                               (log_clamp_diag_sigma**2.0)), (tform_mats[iA]/i2)), A4[1])
        term3 = np.dot(np.dot(((log_clamp_diag_measured-log_clamp_diag_samples) /
                               (log_clamp_diag_sigma**2.0)), (tform_mats[iA]/i3)), A4[2])
        term4 = np.dot(np.dot(((log_clamp_diag_measured-log_clamp_diag_samples) /
                               (log_clamp_diag_sigma**2.0)), (tform_mats[iA]/i4)), A4[3])
        deriv += -2.0*np.real(term1 + term2 - term3 - term4)

    deriv *= 1.0/float(len(np.concatenate(log_clamp_diag)))

    return deriv


def chisq_logamp(imvec, A, amp, sigma):
    """Log Visibility Amplitudes (normalized) chi-squared"""

    # to lowest order the variance on the logarithm of a quantity x is
    # sigma^2_log(x) = sigma^2/x^2
    logsigma = sigma / amp

    amp_samples = np.abs(np.dot(A, imvec))
    chisq = np.sum(np.abs((np.log(amp) - np.log(amp_samples))/logsigma)**2)/len(amp)
    return chisq

def chisqgrad_logamp(imvec, A, amp, sigma):
    """The gradient of the Log amplitude chi-squared"""

    # to lowest order the variance on the logarithm of a quantity x is
    # sigma^2_log(x) = sigma^2/x^2
    logsigma = sigma / amp

    i1 = np.dot(A, imvec)
    amp_samples = np.abs(i1)

    pp = (np.log(amp) - np.log(amp_samples)) / (logsigma**2) / i1
    out = (-2.0/len(amp)) * np.real(np.dot(pp, A))
    return out

##################################################################################################
# FFT Chi-squared and Gradient Functions
##################################################################################################


def chisq_vis_fft(vis_arr, A, vis, sigma):
    """Visibility chi-squared from fft
    """

    im_info, sampler_info_list, gridder_info_list = A
    samples = obsh.sampler(vis_arr, sampler_info_list, sample_type="vis")

    chisq = np.sum(np.abs((samples-vis)/sigma)**2)/(2*len(vis))

    return chisq


def chisqgrad_vis_fft(vis_arr, A, vis, sigma):
    """The gradient of the visibility chi-squared from fft
    """

    im_info, sampler_info_list, gridder_info_list = A

    # samples and gradient FT
    pulsefac = sampler_info_list[0].pulsefac
    samples = obsh.sampler(vis_arr, sampler_info_list, sample_type="vis")
    wdiff_vec = (-1.0/len(vis)*(vis - samples)/(sigma**2)) * pulsefac.conj()

    # Setup and perform the inverse FFT
    wdiff_arr = obsh.gridder([wdiff_vec], gridder_info_list)
    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff_arr)))
    grad_arr = grad_arr * (im_info.npad * im_info.npad)

    # extract relevant cells and flatten
    out = np.real(grad_arr[im_info.padvaly1:-im_info.padvaly2,
                           im_info.padvalx1:-im_info.padvalx2].flatten())

    return out


def chisq_amp_fft(vis_arr, A, amp, sigma):
    """Visibility amplitude chi-squared from fft
    """

    im_info, sampler_info_list, gridder_info_list = A
    amp_samples = np.abs(obsh.sampler(vis_arr, sampler_info_list, sample_type="vis"))
    chisq = np.sum(np.abs((amp_samples-amp)/sigma)**2)/(len(amp))
    return chisq


def chisqgrad_amp_fft(vis_arr, A, amp, sigma):
    """The gradient of the amplitude chi-kernesquared
    """

    im_info, sampler_info_list, gridder_info_list = A

    # samples
    samples = obsh.sampler(vis_arr, sampler_info_list, sample_type="vis")
    amp_samples = np.abs(samples)

    # gradient FT
    pulsefac = sampler_info_list[0].pulsefac
    wdiff_vec = (-2.0/len(amp)*((amp - amp_samples) * amp_samples) /
                 (sigma**2) / samples.conj()) * pulsefac.conj()

    # Setup and perform the inverse FFT
    wdiff_arr = obsh.gridder([wdiff_vec], gridder_info_list)
    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff_arr)))
    grad_arr = grad_arr * (im_info.npad * im_info.npad)

    # extract relevent cells and flatten
    out = np.real(grad_arr[im_info.padvaly1:-im_info.padvaly2,
                           im_info.padvalx1:-im_info.padvalx2].flatten())

    return out


def chisq_bs_fft(vis_arr, A, bis, sigma):
    """Bispectrum chi-squared from fft"""

    im_info, sampler_info_list, gridder_info_list = A
    bisamples = obsh.sampler(vis_arr, sampler_info_list, sample_type="bs")

    return np.sum(np.abs((bis - bisamples)/sigma)**2)/(2.*len(bis))


def chisqgrad_bs_fft(vis_arr, A, bis, sigma):
    """The gradient of the amplitude chi-squared
    """
    im_info, sampler_info_list, gridder_info_list = A

    v1 = obsh.sampler(vis_arr, [sampler_info_list[0]], sample_type="vis")
    v2 = obsh.sampler(vis_arr, [sampler_info_list[1]], sample_type="vis")
    v3 = obsh.sampler(vis_arr, [sampler_info_list[2]], sample_type="vis")
    bisamples = v1*v2*v3

    wdiff = -1.0/len(bis)*(bis - bisamples)/(sigma**2)

    pt1 = wdiff * (v2 * v3).conj() * sampler_info_list[0].pulsefac.conj()
    pt2 = wdiff * (v1 * v3).conj() * sampler_info_list[1].pulsefac.conj()
    pt3 = wdiff * (v1 * v2).conj() * sampler_info_list[2].pulsefac.conj()

    # Setup and perform the inverse FFT
    wdiff = obsh.gridder([pt1, pt2, pt3], gridder_info_list)
    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff)))
    grad_arr = grad_arr * (im_info.npad * im_info.npad)

    # extract relevant cells and flatten
    out = np.real(grad_arr[im_info.padvaly1:-im_info.padvaly2,
                           im_info.padvalx1:-im_info.padvalx2].flatten())
    return out


def chisq_cphase_fft(vis_arr, A, clphase, sigma):
    """Closure Phases (normalized) chi-squared from fft
    """

    clphase = clphase * ehc.DEGREE
    sigma = sigma * ehc.DEGREE

    im_info, sampler_info_list, gridder_info_list = A
    clphase_samples = np.angle(obsh.sampler(vis_arr, sampler_info_list, sample_type="bs"))

    chisq = (2.0/len(clphase)) * np.sum((1.0 - np.cos(clphase-clphase_samples))/(sigma**2))
    return chisq


def chisqgrad_cphase_fft(vis_arr, A, clphase, sigma):
    """The gradient of the closure phase chi-squared from fft"""

    clphase = clphase * ehc.DEGREE
    sigma = sigma * ehc.DEGREE
    im_info, sampler_info_list, gridder_info_list = A

    # sample visibilities and closure phases
    v1 = obsh.sampler(vis_arr, [sampler_info_list[0]], sample_type="vis")
    v2 = obsh.sampler(vis_arr, [sampler_info_list[1]], sample_type="vis")
    v3 = obsh.sampler(vis_arr, [sampler_info_list[2]], sample_type="vis")
    clphase_samples = np.angle(v1*v2*v3)

    pref = (2.0/len(clphase)) * np.sin(clphase - clphase_samples)/(sigma**2)
    pt1 = pref/v1.conj() * sampler_info_list[0].pulsefac.conj()
    pt2 = pref/v2.conj() * sampler_info_list[1].pulsefac.conj()
    pt3 = pref/v3.conj() * sampler_info_list[2].pulsefac.conj()

    # Setup and perform the inverse FFT
    wdiff = obsh.gridder([pt1, pt2, pt3], gridder_info_list)
    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff)))
    grad_arr = grad_arr * (im_info.npad * im_info.npad)

    # extract relevant cells and flatten
    out = np.imag(grad_arr[im_info.padvaly1:-im_info.padvaly2,
                           im_info.padvalx1:-im_info.padvalx2].flatten())

    return out


def chisq_cphase_diag_fft(imvec, A, clphase_diag, sigma):
    """Diagonalized closure phases (normalized) chi-squared from fft
    """

    clphase_diag = np.concatenate(clphase_diag) * ehc.DEGREE
    sigma = np.concatenate(sigma) * ehc.DEGREE

    A3 = A[0]
    tform_mats = A[1]

    im_info, sampler_info_list, gridder_info_list = A3
    vis_arr = obsh.fft_imvec(imvec, A3[0])
    clphase_samples = np.angle(obsh.sampler(vis_arr, sampler_info_list, sample_type="bs"))

    count = 0
    clphase_diag_samples = []
    for tform_mat in tform_mats:
        clphase_samples_here = clphase_samples[count:count+len(tform_mat)]
        clphase_diag_samples.append(np.dot(tform_mat, clphase_samples_here))
        count += len(tform_mat)

    clphase_diag_samples = np.concatenate(clphase_diag_samples)

    chisq = np.sum((1.0 - np.cos(clphase_diag-clphase_diag_samples))/(sigma**2))
    chisq *= (2.0/len(clphase_diag))
    return chisq


def chisqgrad_cphase_diag_fft(imvec, A, clphase_diag, sigma):
    """The gradient of the closure phase chi-squared from fft"""

    clphase_diag = np.concatenate(clphase_diag) * ehc.DEGREE
    sigma = np.concatenate(sigma) * ehc.DEGREE

    A3 = A[0]
    tform_mats = A[1]

    im_info, sampler_info_list, gridder_info_list = A3
    vis_arr = obsh.fft_imvec(imvec, A3[0])

    # sample visibilities and closure phases
    v1 = obsh.sampler(vis_arr, [sampler_info_list[0]], sample_type="vis")
    v2 = obsh.sampler(vis_arr, [sampler_info_list[1]], sample_type="vis")
    v3 = obsh.sampler(vis_arr, [sampler_info_list[2]], sample_type="vis")
    clphase_samples = np.angle(v1*v2*v3)

    # gradient vec stuff
    count = 0
    pref = np.zeros_like(clphase_samples)
    for tform_mat in tform_mats:

        clphase_diag_samples = np.dot(tform_mat, clphase_samples[count:count+len(tform_mat)])
        clphase_diag_measured = clphase_diag[count:count+len(tform_mat)]
        clphase_diag_sigma = sigma[count:count+len(tform_mat)]

        for j in range(len(clphase_diag_measured)):
            pref[count:count+len(tform_mat)] += 2.0 * tform_mat[j, :] * np.sin(
                clphase_diag_measured[j] - clphase_diag_samples[j])/(clphase_diag_sigma[j]**2)

        count += len(tform_mat)

    pt1 = pref/v1.conj() * sampler_info_list[0].pulsefac.conj()
    pt2 = pref/v2.conj() * sampler_info_list[1].pulsefac.conj()
    pt3 = pref/v3.conj() * sampler_info_list[2].pulsefac.conj()

    # Setup and perform the inverse FFT
    wdiff = obsh.gridder([pt1, pt2, pt3], gridder_info_list)
    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff)))
    grad_arr = grad_arr * (im_info.npad * im_info.npad)

    # extract relevant cells and flatten
    deriv = np.imag(grad_arr[im_info.padvaly1:-im_info.padvaly2,
                             im_info.padvalx1:-im_info.padvalx2].flatten())
    deriv *= 1.0/float(len(clphase_diag))

    return deriv


def chisq_camp_fft(vis_arr, A, clamp, sigma):
    """Closure Amplitudes (normalized) chi-squared from fft
    """

    im_info, sampler_info_list, gridder_info_list = A
    clamp_samples = obsh.sampler(vis_arr, sampler_info_list, sample_type="camp")
    chisq = np.sum(np.abs((clamp - clamp_samples)/sigma)**2)/len(clamp)
    return chisq


def chisqgrad_camp_fft(vis_arr, A, clamp, sigma):
    """The gradient of the closure amplitude chi-squared from fft
    """

    im_info, sampler_info_list, gridder_info_list = A

    # sampled visibility and closure amplitudes
    v1 = obsh.sampler(vis_arr, [sampler_info_list[0]], sample_type="vis")
    v2 = obsh.sampler(vis_arr, [sampler_info_list[1]], sample_type="vis")
    v3 = obsh.sampler(vis_arr, [sampler_info_list[2]], sample_type="vis")
    v4 = obsh.sampler(vis_arr, [sampler_info_list[3]], sample_type="vis")
    clamp_samples = np.abs((v1 * v2)/(v3 * v4))

    # gradient components
    pp = (-2.0/len(clamp)) * ((clamp - clamp_samples) * clamp_samples)/(sigma**2)
    pt1 = pp/v1.conj() * sampler_info_list[0].pulsefac.conj()
    pt2 = pp/v2.conj() * sampler_info_list[1].pulsefac.conj()
    pt3 = -pp/v3.conj() * sampler_info_list[2].pulsefac.conj()
    pt4 = -pp/v4.conj() * sampler_info_list[3].pulsefac.conj()

    # Setup and perform the inverse FFT
    wdiff = obsh.gridder([pt1, pt2, pt3, pt4], gridder_info_list)
    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff)))
    grad_arr = grad_arr * (im_info.npad * im_info.npad)

    # extract relevant cells and flatten
    out = np.real(grad_arr[im_info.padvaly1:-im_info.padvaly2,
                           im_info.padvalx1:-im_info.padvalx2].flatten())

    return out


def chisq_logcamp_fft(vis_arr, A, log_clamp, sigma):
    """Closure Amplitudes (normalized) chi-squared from fft
    """

    im_info, sampler_info_list, gridder_info_list = A
    log_clamp_samples = np.log(obsh.sampler(vis_arr, sampler_info_list, sample_type='camp'))

    chisq = np.sum(np.abs((log_clamp - log_clamp_samples)/sigma)**2) / (len(log_clamp))

    return chisq


def chisqgrad_logcamp_fft(vis_arr, A, log_clamp, sigma):
    """The gradient of the closure amplitude chi-squared from fft
    """

    im_info, sampler_info_list, gridder_info_list = A

    # sampled visibility and closure amplitudes
    v1 = obsh.sampler(vis_arr, [sampler_info_list[0]], sample_type="vis")
    v2 = obsh.sampler(vis_arr, [sampler_info_list[1]], sample_type="vis")
    v3 = obsh.sampler(vis_arr, [sampler_info_list[2]], sample_type="vis")
    v4 = obsh.sampler(vis_arr, [sampler_info_list[3]], sample_type="vis")

    log_clamp_samples = np.log(np.abs((v1 * v2)/(v3 * v4)))

    # gradient components
    pp = (-2.0/len(log_clamp)) * (log_clamp - log_clamp_samples) / (sigma**2)
    pt1 = pp / v1.conj() * sampler_info_list[0].pulsefac.conj()
    pt2 = pp / v2.conj() * sampler_info_list[1].pulsefac.conj()
    pt3 = -pp / v3.conj() * sampler_info_list[2].pulsefac.conj()
    pt4 = -pp / v4.conj() * sampler_info_list[3].pulsefac.conj()

    # Setup and perform the inverse FFT
    wdiff = obsh.gridder([pt1, pt2, pt3, pt4], gridder_info_list)
    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff)))
    grad_arr = grad_arr * (im_info.npad * im_info.npad)

    # extract relevant cells and flatten
    out = np.real(grad_arr[im_info.padvaly1:-im_info.padvaly2,
                           im_info.padvalx1:-im_info.padvalx2].flatten())

    return out


def chisq_logcamp_diag_fft(imvec, A, log_clamp_diag, sigma):
    """Diagonalized log closure amplitudes (normalized) chi-squared from fft
    """

    log_clamp_diag = np.concatenate(log_clamp_diag)
    sigma = np.concatenate(sigma)

    A4 = A[0]
    tform_mats = A[1]

    im_info, sampler_info_list, gridder_info_list = A4
    vis_arr = obsh.fft_imvec(imvec, A4[0])
    log_clamp_samples = np.log(obsh.sampler(vis_arr, sampler_info_list, sample_type='camp'))

    count = 0
    log_clamp_diag_samples = []
    for tform_mat in tform_mats:
        log_clamp_samples_here = log_clamp_samples[count:count+len(tform_mat)]
        log_clamp_diag_samples.append(np.dot(tform_mat, log_clamp_samples_here))
        count += len(tform_mat)
    log_clamp_diag_samples = np.concatenate(log_clamp_diag_samples)

    chisq = np.sum(np.abs((log_clamp_diag - log_clamp_diag_samples)/sigma)**2)
    chisq /= (len(log_clamp_diag))
    return chisq


def chisqgrad_logcamp_diag_fft(imvec, A, log_clamp_diag, sigma):
    """The gradient of the diagonalized log closure amplitude chi-squared from fft
    """

    log_clamp_diag = np.concatenate(log_clamp_diag)
    sigma = np.concatenate(sigma)

    A4 = A[0]
    tform_mats = A[1]

    im_info, sampler_info_list, gridder_info_list = A4
    vis_arr = obsh.fft_imvec(imvec, A4[0])

    # sampled visibility and closure amplitudes
    v1 = obsh.sampler(vis_arr, [sampler_info_list[0]], sample_type="vis")
    v2 = obsh.sampler(vis_arr, [sampler_info_list[1]], sample_type="vis")
    v3 = obsh.sampler(vis_arr, [sampler_info_list[2]], sample_type="vis")
    v4 = obsh.sampler(vis_arr, [sampler_info_list[3]], sample_type="vis")
    log_clamp_samples = np.log(np.abs((v1 * v2)/(v3 * v4)))

    # gradient vec stuff
    count = 0
    pref = np.zeros_like(log_clamp_samples)
    for tform_mat in tform_mats:

        log_clamp_diag_samples = np.dot(tform_mat, log_clamp_samples[count:count+len(tform_mat)])
        log_clamp_diag_measured = log_clamp_diag[count:count+len(tform_mat)]
        log_clamp_diag_sigma = sigma[count:count+len(tform_mat)]

        for j in range(len(log_clamp_diag_measured)):
            pref[count:count+len(tform_mat)] += -2.0 * tform_mat[j, :] * \
                (log_clamp_diag_measured[j] - log_clamp_diag_samples[j]) / \
                (log_clamp_diag_sigma[j]**2)

        count += len(tform_mat)

    pt1 = pref / v1.conj() * sampler_info_list[0].pulsefac.conj()
    pt2 = pref / v2.conj() * sampler_info_list[1].pulsefac.conj()
    pt3 = -pref / v3.conj() * sampler_info_list[2].pulsefac.conj()
    pt4 = -pref / v4.conj() * sampler_info_list[3].pulsefac.conj()

    # Setup and perform the inverse FFT
    wdiff = obsh.gridder([pt1, pt2, pt3, pt4], gridder_info_list)
    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff)))
    grad_arr = grad_arr * (im_info.npad * im_info.npad)

    # extract relevant cells and flatten
    deriv = np.real(grad_arr[im_info.padvaly1:-im_info.padvaly2,
                             im_info.padvalx1:-im_info.padvalx2].flatten())
    deriv *= 1.0/float(len(log_clamp_diag))

    return deriv


def chisq_logamp_fft(vis_arr, A, amp, sigma):
    """Visibility amplitude chi-squared from fft
    """

    # to lowest order the variance on the logarithm of a quantity x is
    # sigma^2_log(x) = sigma^2/x^2
    logsigma = sigma / amp

    im_info, sampler_info_list, gridder_info_list = A
    amp_samples = np.abs(obsh.sampler(vis_arr, sampler_info_list, sample_type="vis"))
    chisq = np.sum(np.abs((np.log(amp_samples)-np.log(amp))/logsigma)**2)/(len(amp))
    return chisq


def chisqgrad_logamp_fft(vis_arr, A, amp, sigma):
    """The gradient of the amplitude chi-kernesquared
    """

    im_info, sampler_info_list, gridder_info_list = A

    # samples
    samples = obsh.sampler(vis_arr, sampler_info_list, sample_type="vis")
    amp_samples = np.abs(samples)

    # gradient FT
    logsigma = sigma / amp
    pulsefac = sampler_info_list[0].pulsefac
    wdiff_vec = (-2.0/len(amp)*(np.log(amp) - np.log(amp_samples)) /
                 (logsigma**2) / samples.conj()) * pulsefac.conj()

    # Setup and perform the inverse FFT
    wdiff_arr = obsh.gridder([wdiff_vec], gridder_info_list)
    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff_arr)))
    grad_arr = grad_arr * (im_info.npad * im_info.npad)

    # extract relevent cells and flatten
    out = np.real(grad_arr[im_info.padvaly1:-im_info.padvaly2,
                           im_info.padvalx1:-im_info.padvalx2].flatten())

    return out

##################################################################################################
# NFFT Chi-squared and Gradient Functions
##################################################################################################


def chisq_vis_nfft(imvec, A, vis, sigma):
    """Visibility chi-squared from nfft
    """

    # get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    # compute uniform --> nonuniform transform
    plan.f_hat = imvec.copy().reshape((nfft_info.ydim, nfft_info.xdim)).T
    plan.trafo()
    samples = plan.f.copy()*pulsefac

    # compute chi^2
    chisq = np.sum(np.abs((samples-vis)/sigma)**2)/(2*len(vis))

    return chisq


def chisqgrad_vis_nfft(imvec, A, vis, sigma):
    """The gradient of the visibility chi-squared from nfft
    """

    # get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    # compute uniform --> nonuniform transform
    plan.f_hat = imvec.copy().reshape((nfft_info.ydim, nfft_info.xdim)).T
    plan.trafo()
    samples = plan.f.copy()*pulsefac

    # gradient vec for adjoint FT
    wdiff_vec = (-1.0/len(vis)*(vis - samples)/(sigma**2)) * pulsefac.conj()
    plan.f = wdiff_vec
    plan.adjoint()
    grad = np.real((plan.f_hat.copy().T).reshape(nfft_info.xdim*nfft_info.ydim))

    return grad


def chisq_amp_nfft(imvec, A, amp, sigma):
    """Visibility amplitude chi-squared from nfft
    """
    # get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    # compute uniform --> nonuniform transform
    plan.f_hat = imvec.copy().reshape((nfft_info.ydim, nfft_info.xdim)).T
    plan.trafo()
    samples = plan.f.copy()*pulsefac

    # compute chi^2
    amp_samples = np.abs(samples)
    chisq = np.sum(np.abs((amp_samples-amp)/sigma)**2)/(len(amp))

    return chisq


def chisqgrad_amp_nfft(imvec, A, amp, sigma):
    """The gradient of the amplitude chi-squared from nfft
    """

    # get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    # compute uniform --> nonuniform transform
    plan.f_hat = imvec.copy().reshape((nfft_info.ydim, nfft_info.xdim)).T
    plan.trafo()
    samples = plan.f.copy()*pulsefac
    amp_samples = np.abs(samples)

    # gradient vec for adjoint FT
    wdiff_vec = (-2.0/len(amp)*((amp - amp_samples) * samples) /
                 (sigma**2) / amp_samples) * pulsefac.conj()
    plan.f = wdiff_vec
    plan.adjoint()
    out = np.real((plan.f_hat.copy().T).reshape(nfft_info.xdim*nfft_info.ydim))

    return out


def chisq_bs_nfft(imvec, A, bis, sigma):
    """Bispectrum chi-squared from fft"""

    # get nfft objects
    nfft_info1 = A[0]
    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac

    nfft_info2 = A[1]
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac

    nfft_info3 = A[2]
    plan3 = nfft_info3.plan
    pulsefac3 = nfft_info3.pulsefac

    # compute uniform --> nonuniform transforms
    plan1.f_hat = imvec.copy().reshape((nfft_info1.ydim, nfft_info1.xdim)).T
    plan1.trafo()
    samples1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim, nfft_info2.xdim)).T
    plan2.trafo()
    samples2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim, nfft_info3.xdim)).T
    plan3.trafo()
    samples3 = plan3.f.copy()*pulsefac3

    # compute chi^2
    bisamples = samples1*samples2*samples3
    chisq = np.sum(np.abs((bis - bisamples)/sigma)**2)/(2.*len(bis))
    return chisq


def chisqgrad_bs_nfft(imvec, A, bis, sigma):
    """The gradient of the amplitude chi-squared from the nfft
    """
    # get nfft objects
    nfft_info1 = A[0]
    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac

    nfft_info2 = A[1]
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac

    nfft_info3 = A[2]
    plan3 = nfft_info3.plan
    pulsefac3 = nfft_info3.pulsefac

    # compute uniform --> nonuniform transforms
    plan1.f_hat = imvec.copy().reshape((nfft_info1.ydim, nfft_info1.xdim)).T
    plan1.trafo()
    v1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim, nfft_info2.xdim)).T
    plan2.trafo()
    v2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim, nfft_info3.xdim)).T
    plan3.trafo()
    v3 = plan3.f.copy()*pulsefac3

    # gradient vec for adjoint FT
    bisamples = v1*v2*v3
    wdiff = -1.0/len(bis)*(bis - bisamples)/(sigma**2)
    pt1 = wdiff * (v2 * v3).conj() * pulsefac1.conj()
    pt2 = wdiff * (v1 * v3).conj() * pulsefac2.conj()
    pt3 = wdiff * (v1 * v2).conj() * pulsefac3.conj()

    # Setup and perform the inverse FFT
    plan1.f = pt1
    plan1.adjoint()
    out1 = np.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))

    plan2.f = pt2
    plan2.adjoint()
    out2 = np.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))

    plan3.f = pt3
    plan3.adjoint()
    out3 = np.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

    out = out1 + out2 + out3
    return out


def chisq_cphase_nfft(imvec, A, clphase, sigma):
    """Closure Phases (normalized) chi-squared from nfft
    """

    clphase = clphase * ehc.DEGREE
    sigma = sigma * ehc.DEGREE

    # get nfft objects
    nfft_info1 = A[0]
    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac

    nfft_info2 = A[1]
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac

    nfft_info3 = A[2]
    plan3 = nfft_info3.plan
    pulsefac3 = nfft_info3.pulsefac

    # compute uniform --> nonuniform transforms
    plan1.f_hat = imvec.copy().reshape((nfft_info1.ydim, nfft_info1.xdim)).T
    plan1.trafo()
    samples1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim, nfft_info2.xdim)).T
    plan2.trafo()
    samples2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim, nfft_info3.xdim)).T
    plan3.trafo()
    samples3 = plan3.f.copy()*pulsefac3

    # compute chi^2
    clphase_samples = np.angle(samples1*samples2*samples3)
    chisq = (2.0/len(clphase)) * np.sum((1.0 - np.cos(clphase-clphase_samples))/(sigma**2))

    return chisq


def chisqgrad_cphase_nfft(imvec, A, clphase, sigma):
    """The gradient of the closure phase chi-squared from nfft"""

    clphase = clphase * ehc.DEGREE
    sigma = sigma * ehc.DEGREE

    # get nfft objects
    nfft_info1 = A[0]
    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac

    nfft_info2 = A[1]
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac

    nfft_info3 = A[2]
    plan3 = nfft_info3.plan
    pulsefac3 = nfft_info3.pulsefac

    # compute uniform --> nonuniform transforms
    plan1.f_hat = imvec.copy().reshape((nfft_info1.ydim, nfft_info1.xdim)).T
    plan1.trafo()
    v1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim, nfft_info2.xdim)).T
    plan2.trafo()
    v2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim, nfft_info3.xdim)).T
    plan3.trafo()
    v3 = plan3.f.copy()*pulsefac3

    # gradient vec for adjoint FT
    clphase_samples = np.angle(v1*v2*v3)
    pref = (2.0/len(clphase)) * np.sin(clphase - clphase_samples)/(sigma**2)
    pt1 = pref/v1.conj() * pulsefac1.conj()
    pt2 = pref/v2.conj() * pulsefac2.conj()
    pt3 = pref/v3.conj() * pulsefac3.conj()

    # Setup and perform the inverse FFT
    plan1.f = pt1
    plan1.adjoint()
    out1 = np.imag((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))

    plan2.f = pt2
    plan2.adjoint()
    out2 = np.imag((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))

    plan3.f = pt3
    plan3.adjoint()
    out3 = np.imag((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

    out = out1 + out2 + out3
    return out


def chisq_cphase_diag_nfft(imvec, A, clphase_diag, sigma):
    """Diagonalized closure phases (normalized) chi-squared from nfft
    """

    clphase_diag = np.concatenate(clphase_diag) * ehc.DEGREE
    sigma = np.concatenate(sigma) * ehc.DEGREE

    A3 = A[0]
    tform_mats = A[1]

    # get nfft objects
    nfft_info1 = A3[0]
    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac

    nfft_info2 = A3[1]
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac

    nfft_info3 = A3[2]
    plan3 = nfft_info3.plan
    pulsefac3 = nfft_info3.pulsefac

    # compute uniform --> nonuniform transforms
    plan1.f_hat = imvec.copy().reshape((nfft_info1.ydim, nfft_info1.xdim)).T
    plan1.trafo()
    samples1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim, nfft_info2.xdim)).T
    plan2.trafo()
    samples2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim, nfft_info3.xdim)).T
    plan3.trafo()
    samples3 = plan3.f.copy()*pulsefac3

    clphase_samples = np.angle(samples1*samples2*samples3)

    count = 0
    clphase_diag_samples = []
    for tform_mat in tform_mats:
        clphase_samples_here = clphase_samples[count:count+len(tform_mat)]
        clphase_diag_samples.append(np.dot(tform_mat, clphase_samples_here))
        count += len(tform_mat)

    clphase_diag_samples = np.concatenate(clphase_diag_samples)

    # compute chi^2
    chisq = (2.0/len(clphase_diag)) * \
        np.sum((1.0 - np.cos(clphase_diag-clphase_diag_samples))/(sigma**2))

    return chisq


def chisqgrad_cphase_diag_nfft(imvec, A, clphase_diag, sigma):
    """The gradient of the diagonalized closure phase chi-squared from nfft"""

    clphase_diag = np.concatenate(clphase_diag) * ehc.DEGREE
    sigma = np.concatenate(sigma) * ehc.DEGREE

    A3 = A[0]
    tform_mats = A[1]

    # get nfft objects
    nfft_info1 = A3[0]
    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac

    nfft_info2 = A3[1]
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac

    nfft_info3 = A3[2]
    plan3 = nfft_info3.plan
    pulsefac3 = nfft_info3.pulsefac

    # compute uniform --> nonuniform transforms
    plan1.f_hat = imvec.copy().reshape((nfft_info1.ydim, nfft_info1.xdim)).T
    plan1.trafo()
    v1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim, nfft_info2.xdim)).T
    plan2.trafo()
    v2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim, nfft_info3.xdim)).T
    plan3.trafo()
    v3 = plan3.f.copy()*pulsefac3

    clphase_samples = np.angle(v1*v2*v3)

    # gradient vec for adjoint FT
    count = 0
    pref = np.zeros_like(clphase_samples)
    for tform_mat in tform_mats:

        clphase_diag_samples = np.dot(tform_mat, clphase_samples[count:count+len(tform_mat)])
        clphase_diag_measured = clphase_diag[count:count+len(tform_mat)]
        clphase_diag_sigma = sigma[count:count+len(tform_mat)]

        for j in range(len(clphase_diag_measured)):
            pref[count:count+len(tform_mat)] += 2.0 * tform_mat[j, :] * np.sin(
                clphase_diag_measured[j] - clphase_diag_samples[j])/(clphase_diag_sigma[j]**2)

        count += len(tform_mat)

    pt1 = pref/v1.conj() * pulsefac1.conj()
    pt2 = pref/v2.conj() * pulsefac2.conj()
    pt3 = pref/v3.conj() * pulsefac3.conj()

    # Setup and perform the inverse FFT
    plan1.f = pt1
    plan1.adjoint()
    out1 = np.imag((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))

    plan2.f = pt2
    plan2.adjoint()
    out2 = np.imag((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))

    plan3.f = pt3
    plan3.adjoint()
    out3 = np.imag((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

    deriv = out1 + out2 + out3
    deriv *= 1.0/float(len(clphase_diag))

    return deriv


def chisq_camp_nfft(imvec, A, clamp, sigma):
    """Closure Amplitudes (normalized) chi-squared from fft
    """

    # get nfft objects
    nfft_info1 = A[0]
    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac

    nfft_info2 = A[1]
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac

    nfft_info3 = A[2]
    plan3 = nfft_info3.plan
    pulsefac3 = nfft_info3.pulsefac

    nfft_info4 = A[3]
    plan4 = nfft_info4.plan
    pulsefac4 = nfft_info4.pulsefac

    # compute uniform --> nonuniform transforms
    plan1.f_hat = imvec.copy().reshape((nfft_info1.ydim, nfft_info1.xdim)).T
    plan1.trafo()
    samples1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim, nfft_info2.xdim)).T
    plan2.trafo()
    samples2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim, nfft_info3.xdim)).T
    plan3.trafo()
    samples3 = plan3.f.copy()*pulsefac3

    plan4.f_hat = imvec.copy().reshape((nfft_info4.ydim, nfft_info4.xdim)).T
    plan4.trafo()
    samples4 = plan4.f.copy()*pulsefac4

    # compute chi^2
    clamp_samples = np.abs((samples1*samples2)/(samples3*samples4))
    chisq = np.sum(np.abs((clamp - clamp_samples)/sigma)**2)/len(clamp)
    return chisq


def chisqgrad_camp_nfft(imvec, A, clamp, sigma):
    """The gradient of the closure amplitude chi-squared from fft
    """

    # get nfft objects
    nfft_info1 = A[0]
    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac

    nfft_info2 = A[1]
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac

    nfft_info3 = A[2]
    plan3 = nfft_info3.plan
    pulsefac3 = nfft_info3.pulsefac

    nfft_info4 = A[3]
    plan4 = nfft_info4.plan
    pulsefac4 = nfft_info4.pulsefac

    # compute uniform --> nonuniform transforms
    plan1.f_hat = imvec.copy().reshape((nfft_info1.ydim, nfft_info1.xdim)).T
    plan1.trafo()
    v1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim, nfft_info2.xdim)).T
    plan2.trafo()
    v2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim, nfft_info3.xdim)).T
    plan3.trafo()
    v3 = plan3.f.copy()*pulsefac3

    plan4.f_hat = imvec.copy().reshape((nfft_info4.ydim, nfft_info4.xdim)).T
    plan4.trafo()
    v4 = plan4.f.copy()*pulsefac4

    # gradient vec for adjoint FT
    clamp_samples = np.abs((v1 * v2)/(v3 * v4))

    pp = (-2.0/len(clamp)) * ((clamp - clamp_samples) * clamp_samples)/(sigma**2)
    pt1 = pp/v1.conj() * pulsefac1.conj()
    pt2 = pp/v2.conj() * pulsefac2.conj()
    pt3 = -pp/v3.conj() * pulsefac3.conj()
    pt4 = -pp/v4.conj() * pulsefac4.conj()

    # Setup and perform the inverse FFT
    plan1.f = pt1
    plan1.adjoint()
    out1 = np.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))

    plan2.f = pt2
    plan2.adjoint()
    out2 = np.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))

    plan3.f = pt3
    plan3.adjoint()
    out3 = np.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

    plan4.f = pt4
    plan4.adjoint()
    out4 = np.real((plan4.f_hat.copy().T).reshape(nfft_info4.xdim*nfft_info4.ydim))

    out = out1 + out2 + out3 + out4
    return out


def chisq_logcamp_nfft(imvec, A, log_clamp, sigma):
    """Log Closure Amplitudes (normalized) chi-squared from fft
    """

    # get nfft objects
    nfft_info1 = A[0]
    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac

    nfft_info2 = A[1]
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac

    nfft_info3 = A[2]
    plan3 = nfft_info3.plan
    pulsefac3 = nfft_info3.pulsefac

    nfft_info4 = A[3]
    plan4 = nfft_info4.plan
    pulsefac4 = nfft_info4.pulsefac

    # compute uniform --> nonuniform transforms
    plan1.f_hat = imvec.copy().reshape((nfft_info1.ydim, nfft_info1.xdim)).T
    plan1.trafo()
    samples1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim, nfft_info2.xdim)).T
    plan2.trafo()
    samples2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim, nfft_info3.xdim)).T
    plan3.trafo()
    samples3 = plan3.f.copy()*pulsefac3

    plan4.f_hat = imvec.copy().reshape((nfft_info4.ydim, nfft_info4.xdim)).T
    plan4.trafo()
    samples4 = plan4.f.copy()*pulsefac4

    # compute chi^2
    log_clamp_samples = (np.log(np.abs(samples1)) + np.log(np.abs(samples2)) -
                         np.log(np.abs(samples3)) - np.log(np.abs(samples4)))
    chisq = np.sum(np.abs((log_clamp - log_clamp_samples)/sigma)**2) / (len(log_clamp))
    return chisq


def chisqgrad_logcamp_nfft(imvec, A, log_clamp, sigma):
    """The gradient of the log closure amplitude chi-squared from fft
    """

    # get nfft objects
    nfft_info1 = A[0]
    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac

    nfft_info2 = A[1]
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac

    nfft_info3 = A[2]
    plan3 = nfft_info3.plan
    pulsefac3 = nfft_info3.pulsefac

    nfft_info4 = A[3]
    plan4 = nfft_info4.plan
    pulsefac4 = nfft_info4.pulsefac

    # compute uniform --> nonuniform transforms
    plan1.f_hat = imvec.copy().reshape((nfft_info1.ydim, nfft_info1.xdim)).T
    plan1.trafo()
    v1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim, nfft_info2.xdim)).T
    plan2.trafo()
    v2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim, nfft_info3.xdim)).T
    plan3.trafo()
    v3 = plan3.f.copy()*pulsefac3

    plan4.f_hat = imvec.copy().reshape((nfft_info4.ydim, nfft_info4.xdim)).T
    plan4.trafo()
    v4 = plan4.f.copy()*pulsefac4

    # gradient vec for adjoint FT
    log_clamp_samples = np.log(np.abs((v1 * v2)/(v3 * v4)))

    pp = (-2.0/len(log_clamp)) * (log_clamp - log_clamp_samples) / (sigma**2)
    pt1 = pp / v1.conj() * pulsefac1.conj()
    pt2 = pp / v2.conj() * pulsefac2.conj()
    pt3 = -pp / v3.conj() * pulsefac3.conj()
    pt4 = -pp / v4.conj() * pulsefac4.conj()

    # Setup and perform the inverse FFT
    plan1.f = pt1
    plan1.adjoint()
    out1 = np.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))

    plan2.f = pt2
    plan2.adjoint()
    out2 = np.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))

    plan3.f = pt3
    plan3.adjoint()
    out3 = np.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

    plan4.f = pt4
    plan4.adjoint()
    out4 = np.real((plan4.f_hat.copy().T).reshape(nfft_info4.xdim*nfft_info4.ydim))

    out = out1 + out2 + out3 + out4
    return out


def chisq_logcamp_diag_nfft(imvec, A, log_clamp_diag, sigma):
    """Diagonalized log closure amplitudes (normalized) chi-squared from nfft
    """

    log_clamp_diag = np.concatenate(log_clamp_diag)
    sigma = np.concatenate(sigma)

    A4 = A[0]
    tform_mats = A[1]

    # get nfft objects
    nfft_info1 = A4[0]
    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac

    nfft_info2 = A4[1]
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac

    nfft_info3 = A4[2]
    plan3 = nfft_info3.plan
    pulsefac3 = nfft_info3.pulsefac

    nfft_info4 = A4[3]
    plan4 = nfft_info4.plan
    pulsefac4 = nfft_info4.pulsefac

    # compute uniform --> nonuniform transforms
    plan1.f_hat = imvec.copy().reshape((nfft_info1.ydim, nfft_info1.xdim)).T
    plan1.trafo()
    samples1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim, nfft_info2.xdim)).T
    plan2.trafo()
    samples2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim, nfft_info3.xdim)).T
    plan3.trafo()
    samples3 = plan3.f.copy()*pulsefac3

    plan4.f_hat = imvec.copy().reshape((nfft_info4.ydim, nfft_info4.xdim)).T
    plan4.trafo()
    samples4 = plan4.f.copy()*pulsefac4

    log_clamp_samples = (np.log(np.abs(samples1)) + np.log(np.abs(samples2)) -
                         np.log(np.abs(samples3)) - np.log(np.abs(samples4)))

    count = 0
    log_clamp_diag_samples = []
    for tform_mat in tform_mats:
        log_clamp_samples_here = log_clamp_samples[count:count+len(tform_mat)]
        log_clamp_diag_samples.append(np.dot(tform_mat, log_clamp_samples_here))
        count += len(tform_mat)

    log_clamp_diag_samples = np.concatenate(log_clamp_diag_samples)

    # compute chi^2
    chisq = np.sum(np.abs((log_clamp_diag - log_clamp_diag_samples)/sigma)**2) / \
        (len(log_clamp_diag))

    return chisq


def chisqgrad_logcamp_diag_nfft(imvec, A, log_clamp_diag, sigma):
    """The gradient of the diagonalized log closure amplitude chi-squared from fft
    """

    log_clamp_diag = np.concatenate(log_clamp_diag)
    sigma = np.concatenate(sigma)

    A4 = A[0]
    tform_mats = A[1]

    # get nfft objects
    nfft_info1 = A4[0]
    plan1 = nfft_info1.plan
    pulsefac1 = nfft_info1.pulsefac

    nfft_info2 = A4[1]
    plan2 = nfft_info2.plan
    pulsefac2 = nfft_info2.pulsefac

    nfft_info3 = A4[2]
    plan3 = nfft_info3.plan
    pulsefac3 = nfft_info3.pulsefac

    nfft_info4 = A4[3]
    plan4 = nfft_info4.plan
    pulsefac4 = nfft_info4.pulsefac

    # compute uniform --> nonuniform transforms
    plan1.f_hat = imvec.copy().reshape((nfft_info1.ydim, nfft_info1.xdim)).T
    plan1.trafo()
    v1 = plan1.f.copy()*pulsefac1

    plan2.f_hat = imvec.copy().reshape((nfft_info2.ydim, nfft_info2.xdim)).T
    plan2.trafo()
    v2 = plan2.f.copy()*pulsefac2

    plan3.f_hat = imvec.copy().reshape((nfft_info3.ydim, nfft_info3.xdim)).T
    plan3.trafo()
    v3 = plan3.f.copy()*pulsefac3

    plan4.f_hat = imvec.copy().reshape((nfft_info4.ydim, nfft_info4.xdim)).T
    plan4.trafo()
    v4 = plan4.f.copy()*pulsefac4

    log_clamp_samples = np.log(np.abs((v1 * v2)/(v3 * v4)))

    # gradient vec for adjoint FT
    count = 0
    pp = np.zeros_like(log_clamp_samples)
    for tform_mat in tform_mats:

        log_clamp_diag_samples = np.dot(tform_mat, log_clamp_samples[count:count+len(tform_mat)])
        log_clamp_diag_measured = log_clamp_diag[count:count+len(tform_mat)]
        log_clamp_diag_sigma = sigma[count:count+len(tform_mat)]

        for j in range(len(log_clamp_diag_measured)):
            pp[count:count+len(tform_mat)] += -2.0 * tform_mat[j, :] * \
                (log_clamp_diag_measured[j] - log_clamp_diag_samples[j]) / \
                (log_clamp_diag_sigma[j]**2)

        count += len(tform_mat)

    pt1 = pp / v1.conj() * pulsefac1.conj()
    pt2 = pp / v2.conj() * pulsefac2.conj()
    pt3 = -pp / v3.conj() * pulsefac3.conj()
    pt4 = -pp / v4.conj() * pulsefac4.conj()

    # Setup and perform the inverse FFT
    plan1.f = pt1
    plan1.adjoint()
    out1 = np.real((plan1.f_hat.copy().T).reshape(nfft_info1.xdim*nfft_info1.ydim))

    plan2.f = pt2
    plan2.adjoint()
    out2 = np.real((plan2.f_hat.copy().T).reshape(nfft_info2.xdim*nfft_info2.ydim))

    plan3.f = pt3
    plan3.adjoint()
    out3 = np.real((plan3.f_hat.copy().T).reshape(nfft_info3.xdim*nfft_info3.ydim))

    plan4.f = pt4
    plan4.adjoint()
    out4 = np.real((plan4.f_hat.copy().T).reshape(nfft_info4.xdim*nfft_info4.ydim))

    deriv = out1 + out2 + out3 + out4
    deriv *= 1.0/float(len(log_clamp_diag))

    return deriv


def chisq_logamp_nfft(imvec, A, amp, sigma):
    """Visibility log amplitude chi-squared from nfft
    """

    # get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    # compute uniform --> nonuniform transform
    plan.f_hat = imvec.copy().reshape((nfft_info.ydim, nfft_info.xdim)).T
    plan.trafo()
    samples = plan.f.copy()*pulsefac

    # to lowest order the variance on the logarithm of a quantity x is
    # sigma^2_log(x) = sigma^2/x^2
    logsigma = sigma / amp

    # compute chi^2
    amp_samples = np.abs(samples)
    chisq = np.sum(np.abs((np.log(amp_samples)-np.log(amp))/logsigma)**2)/(len(amp))

    return chisq


def chisqgrad_logamp_nfft(imvec, A, amp, sigma):
    """The gradient of the log amplitude chi-squared from nfft
    """

    # get nfft object
    nfft_info = A[0]
    plan = nfft_info.plan
    pulsefac = nfft_info.pulsefac

    # compute uniform --> nonuniform transform
    plan.f_hat = imvec.copy().reshape((nfft_info.ydim, nfft_info.xdim)).T
    plan.trafo()
    samples = plan.f.copy()*pulsefac
    amp_samples = np.abs(samples)

    # gradient vec for adjoint FT
    logsigma = sigma / amp
    wdiff_vec = (-2.0/len(amp)*(np.log(amp) - np.log(amp_samples)) /
                 (logsigma**2) / samples.conj()) * pulsefac.conj()
    plan.f = wdiff_vec
    plan.adjoint()
    out = np.real((plan.f_hat.copy().T).reshape(nfft_info.xdim*nfft_info.ydim))

    return out


##################################################################################################
# Regularizer and Gradient Functions
##################################################################################################

def sflux(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER):
    """Total flux constraint
    """
    if norm_reg:
        norm = flux**2
    else:
        norm = 1

    out = -(np.sum(imvec) - flux)**2
    return out/norm


def sfluxgrad(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER):
    """Total flux constraint gradient
    """
    if norm_reg:
        norm = flux**2
    else:
        norm = 1

    out = -2*(np.sum(imvec) - flux)*np.ones(len(imvec))
    return out / norm


def scm(imvec, nx, ny, psize, flux, embed_mask, norm_reg=NORM_REGULARIZER, beam_size=None):
    """Center-of-mass constraint
    """
    if beam_size is None:
        beam_size = psize
    if norm_reg:
        norm = beam_size**2 * flux**2
    else:
        norm = 1

    xx, yy = np.meshgrid(range(nx//2, -nx//2, -1), range(ny//2, -ny//2, -1))
    xx = psize*xx.flatten()[embed_mask]
    yy = psize*yy.flatten()[embed_mask]

    out = -(np.sum(imvec*xx)**2 + np.sum(imvec*yy)**2)
    return out/norm


def scmgrad(imvec, nx, ny, psize, flux, embed_mask, norm_reg=NORM_REGULARIZER, beam_size=None):
    """Center-of-mass constraint gradient
    """
    if beam_size is None:
        beam_size = psize
    if norm_reg:
        norm = beam_size**2 * flux**2
    else:
        norm = 1

    xx, yy = np.meshgrid(range(nx//2, -nx//2, -1), range(ny//2, -ny//2, -1))
    xx = psize*xx.flatten()[embed_mask]
    yy = psize*yy.flatten()[embed_mask]

    out = -2*(np.sum(imvec*xx)*xx + np.sum(imvec*yy)*yy)
    return out/norm


def ssimple(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER):
    """Simple entropy
    """
    if norm_reg:
        norm = flux
    else:
        norm = 1

    entropy = -np.sum(imvec*np.log(imvec/priorvec))
    return entropy/norm


def ssimplegrad(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER):
    """Simple entropy gradient
    """
    if norm_reg:
        norm = flux
    else:
        norm = 1

    entropygrad = -np.log(imvec/priorvec) - 1
    return entropygrad/norm


def sl1(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER):
    """L1 norm regularizer
    """
    if norm_reg:
        norm = flux
    else:
        norm = 1

    # l1 = -np.sum(np.abs(imvec - priorvec))
    l1 = -np.sum(np.abs(imvec))
    return l1/norm


def sl1grad(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER):
    """L1 norm gradient
    """
    if norm_reg:
        norm = flux
    else:
        norm = 1

    # l1grad = -np.sign(imvec - priorvec)
    l1grad = -np.sign(imvec)
    return l1grad/norm


def sl1w(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER, epsilon=ehc.EP):
    """Weighted L1 norm regularizer a la SMILI
    """

    if norm_reg:
        norm = 1  # should be ok?
        # This is SMILI normalization
        # norm = np.sum((np.sqrt(priorvec**2 + epsilon) + epsilon)/np.sqrt(priorvec**2 + epsilon))
    else:
        norm = 1

    num = np.sqrt(imvec**2 + epsilon)
    denom = np.sqrt(priorvec**2 + epsilon) + epsilon

    l1w = -np.sum(num/denom)
    return l1w/norm


def sl1wgrad(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER, epsilon=ehc.EP):
    """Weighted L1 norm gradient
    """
    if norm_reg:
        norm = 1  # should be ok?
        # This is SMILI normalization
        # norm = np.sum((np.sqrt(priorvec**2 + epsilon) + epsilon)/np.sqrt(priorvec**2 + epsilon))
    else:
        norm = 1

    num = imvec / np.sqrt(imvec**2 + epsilon)
    denom = np.sqrt(priorvec**2 + epsilon) + epsilon

    l1wgrad = - num / denom
    return l1wgrad/norm


def fA(imvec, I_ref=1.0, alpha_A=1.0):
    """Function to take imvec to itself in the limit alpha_A -> 0
       and to a binary representation in the limit alpha_A -> infinity
    """
    return 2.0/np.pi * (1.0 + alpha_A)/alpha_A * np.arctan(np.pi*alpha_A/2.0*np.abs(imvec)/I_ref)


def fAgrad(imvec, I_ref=1.0, alpha_A=1.0):
    """Function to take imvec to itself in the limit alpha_A -> 0
       and to a binary representation in the limit alpha_A -> infinity
    """
    return (1.0 + alpha_A) / (I_ref * (1.0 + (np.pi*alpha_A/2.0*imvec/I_ref)**2))


def slA(imvec, priorvec, psize, flux, beam_size=None, alpha_A=1.0, norm_reg=NORM_REGULARIZER):
    """l_A regularizer
    """

    # The appropriate I_ref is something like the total flux divided by the # of pixels per beam
    if beam_size is None:
        beam_size = psize
    I_ref = flux

    if norm_reg:
        norm_l1 = 1.0                  # as alpha_A ->0
        norm_l0 = (beam_size/psize)**2  # as alpha_A ->\infty
        weight_l1 = 1.0/(1.0 + alpha_A)
        weight_l0 = alpha_A
        norm = (norm_l1 * weight_l1 + norm_l0 * weight_l0)/(weight_l0 + weight_l1)
    else:
        norm = 1

    return -np.sum(fA(imvec, I_ref, alpha_A))/norm


def slAgrad(imvec, priorvec, psize, flux, beam_size=None, alpha_A=1.0, norm_reg=NORM_REGULARIZER):
    """l_A gradient
    """

    # The appropriate I_ref is something like the total flux divided by the # of pixels per beam
    if beam_size is None:
        beam_size = psize
    I_ref = flux

    if norm_reg:
        norm_l1 = 1.0                  # as alpha_A ->0
        norm_l0 = (beam_size/psize)**2  # as alpha_A ->\infty
        weight_l1 = 1.0/(1.0 + alpha_A)
        weight_l0 = alpha_A
        norm = (norm_l1 * weight_l1 + norm_l0 * weight_l0)/(weight_l0 + weight_l1)
    else:
        norm = 1

    return -fAgrad(imvec, I_ref, alpha_A)/norm


def sgs(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER):
    """Gull-skilling entropy
    """
    if norm_reg:
        norm = flux
    else:
        norm = 1

    entropy = np.sum(imvec - priorvec - imvec*np.log(imvec/priorvec))
    return entropy/norm


def sgsgrad(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER):
    """Gull-Skilling gradient
    """
    if norm_reg:
        norm = flux
    else:
        norm = 1

    entropygrad = -np.log(imvec/priorvec)
    return entropygrad/norm

# TODO: epsilon is 0 by default for backwards compatibilitys
def stv(imvec, nx, ny, psize, flux, norm_reg=NORM_REGULARIZER, beam_size=None, epsilon=0.):
    """Total variation regularizer
    """
    if beam_size is None:
        beam_size = psize
    if norm_reg:
        norm = flux*psize / beam_size
    else:
        norm = 1

    im = imvec.reshape(ny, nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    out = -np.sum(np.sqrt(np.abs(im_l1 - im)**2 + np.abs(im_l2 - im)**2 + epsilon))
    return out/norm

# TODO: epsilon is 0 by default for backwards compatibility
def stvgrad(imvec, nx, ny, psize, flux, norm_reg=NORM_REGULARIZER, beam_size=None, epsilon=0.):
    """Total variation gradient
    """
    if beam_size is None:
        beam_size = psize
    if norm_reg:
        norm = flux*psize / beam_size
    else:
        norm = 1

    im = imvec.reshape(ny, nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    im_r1 = np.roll(impad, 1, axis=0)[1:ny+1, 1:nx+1]
    im_r2 = np.roll(impad, 1, axis=1)[1:ny+1, 1:nx+1]

    # rotate images
    im_r1l2 = np.roll(np.roll(impad,  1, axis=0), -1, axis=1)[1:ny+1, 1:nx+1]
    im_l1r2 = np.roll(np.roll(impad, -1, axis=0), 1, axis=1)[1:ny+1, 1:nx+1]

    # add together terms and return
    g1 = (2*im - im_l1 - im_l2) / np.sqrt((im - im_l1)**2 + (im - im_l2)**2 + epsilon)
    g2 = (im - im_r1) / np.sqrt((im - im_r1)**2 + (im_r1l2 - im_r1)**2 + epsilon)
    g3 = (im - im_r2) / np.sqrt((im - im_r2)**2 + (im_l1r2 - im_r2)**2 + epsilon)

    # mask the first row column gradient terms that don't exist
    mask1 = np.zeros(im.shape)
    mask2 = np.zeros(im.shape)
    mask1[0, :] = 1
    mask2[:, 0] = 1
    g2[mask1.astype(bool)] = 0
    g3[mask2.astype(bool)] = 0

    # add terms together and return
    out = -(g1 + g2 + g3).flatten()
    return out/norm


def stv2(imvec, nx, ny, psize, flux, norm_reg=NORM_REGULARIZER, beam_size=None):
    """Squared Total variation regularizer
    """
    if beam_size is None:
        beam_size = psize
    if norm_reg:
        norm = psize**4 * flux**2 / beam_size**4
    else:
        norm = 1

    im = imvec.reshape(ny, nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    out = -np.sum((im_l1 - im)**2 + (im_l2 - im)**2)
    return out/norm


def stv2grad(imvec, nx, ny, psize, flux, norm_reg=NORM_REGULARIZER, beam_size=None):
    """Squared Total variation gradient
    """
    if beam_size is None:
        beam_size = psize
    if norm_reg:
        norm = psize**4 * flux**2 / beam_size**4
    else:
        norm = 1

    im = imvec.reshape(ny, nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    im_r1 = np.roll(impad, 1, axis=0)[1:ny+1, 1:nx+1]
    im_r2 = np.roll(impad, 1, axis=1)[1:ny+1, 1:nx+1]

    g1 = (2*im - im_l1 - im_l2)
    g2 = (im - im_r1)
    g3 = (im - im_r2)

    # mask the first row column gradient terms that don't exist
    mask1 = np.zeros(im.shape)
    mask2 = np.zeros(im.shape)
    mask1[0, :] = 1
    mask2[:, 0] = 1
    g2[mask1.astype(bool)] = 0
    g3[mask2.astype(bool)] = 0

    # add together terms and return
    out = -2*(g1 + g2 + g3).flatten()
    return out/norm


def stvlog(imvec, nx, ny, psize, flux, norm_reg=NORM_REGULARIZER, beam_size=None, epsilon=0.):
    """Total variation regularizer applied to log(imvec).

    Forwards to stv with the log-transformed image and a transformed flux
    `logflux = npix * |log(flux/npix)|` so the regularizer has a sensible
    norm when norm_reg=True.
    """
    npix = nx * ny
    logflux = npix * np.abs(np.log(flux / npix))
    return stv(np.log(imvec), nx, ny, psize, logflux,
               norm_reg=norm_reg, beam_size=beam_size, epsilon=epsilon)


def stvloggrad(imvec, nx, ny, psize, flux, norm_reg=NORM_REGULARIZER, beam_size=None, epsilon=0.):
    """Gradient of stvlog: applies stvgrad to log(imvec), then chain rule 1/imvec.
    """
    npix = nx * ny
    logflux = npix * np.abs(np.log(flux / npix))
    g = stvgrad(np.log(imvec), nx, ny, psize, logflux,
                norm_reg=norm_reg, beam_size=beam_size, epsilon=epsilon)
    return g / imvec


def stv2log(imvec, nx, ny, psize, flux, norm_reg=NORM_REGULARIZER, beam_size=None):
    """Squared total variation regularizer applied to log(imvec).

    Forwards to stv2 with the log-transformed image and the transformed flux
    used by stvlog.
    """
    npix = nx * ny
    logflux = npix * np.abs(np.log(flux / npix))
    return stv2(np.log(imvec), nx, ny, psize, logflux,
                norm_reg=norm_reg, beam_size=beam_size)


def stv2loggrad(imvec, nx, ny, psize, flux, norm_reg=NORM_REGULARIZER, beam_size=None):
    """Gradient of stv2log: applies stv2grad to log(imvec), then chain rule 1/imvec.
    """
    npix = nx * ny
    logflux = npix * np.abs(np.log(flux / npix))
    g = stv2grad(np.log(imvec), nx, ny, psize, logflux,
                 norm_reg=norm_reg, beam_size=beam_size)
    return g / imvec


def spatch(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER):
    """Patch prior regularizer
    """
    if norm_reg:
        norm = flux**2
    else:
        norm = 1

    out = -0.5*np.sum((imvec - priorvec) ** 2)
    return out/norm


def spatchgrad(imvec, priorvec, flux, norm_reg=NORM_REGULARIZER):
    """Patch prior gradient
    """
    if norm_reg:
        norm = flux**2
    else:
        norm = 1

    out = -(imvec - priorvec)
    return out/norm

# TODO FIGURE OUT NORMALIZATIONS FOR COMPACT 1 & 2 REGULARIZERS


def scompact(imvec, nx, ny, psize, flux, norm_reg=NORM_REGULARIZER, beam_size=None):
    """I r^2 source size regularizer
    """

    if beam_size is None:
        beam_size = psize
    if norm_reg:
        norm = flux * (beam_size**2)
    else:
        norm = 1

    im = imvec.reshape(ny, nx)

    xx, yy = np.meshgrid(range(nx), range(ny))
    xx = xx - (nx-1)/2.0
    yy = yy - (ny-1)/2.0
    xxpsize = xx * psize
    yypsize = yy * psize

    x0 = np.sum(np.sum(im * xxpsize))/flux
    y0 = np.sum(np.sum(im * yypsize))/flux

    out = -np.sum(np.sum(im * ((xxpsize - x0)**2 + (yypsize - y0)**2)))
    return out/norm


def scompactgrad(imvec, nx, ny, psize, flux, norm_reg=NORM_REGULARIZER, beam_size=None):
    """Gradient for I r^2 source size regularizer
    """

    if beam_size is None:
        beam_size = psize
    if norm_reg:
        norm = flux * beam_size**2
    else:
        norm = 1

    im = imvec.reshape(ny, nx)

    xx, yy = np.meshgrid(range(nx), range(ny))
    xx = xx - (nx-1)/2.0
    yy = yy - (ny-1)/2.0
    xxpsize = xx * psize
    yypsize = yy * psize

    x0 = np.sum(np.sum(im * xxpsize))/flux
    y0 = np.sum(np.sum(im * yypsize))/flux

    term1 = np.sum(np.sum(im * (xxpsize - x0)))
    term2 = np.sum(np.sum(im * (yypsize - y0)))

    grad = -2*xxpsize*term1 - 2*yypsize*term2 + (xxpsize - x0)**2 + (yypsize - y0)**2

    return -grad.reshape(-1)/norm


def scompact2(imvec, nx, ny, psize, flux, norm_reg=NORM_REGULARIZER, beam_size=None):
    """I^2r^2 source size regularizer
    """

    if beam_size is None:
        beam_size = psize
    if norm_reg:
        norm = flux**2 * beam_size**2
    else:
        norm = 1

    im = imvec.reshape(ny, nx)

    xx, yy = np.meshgrid(range(nx), range(ny))
    xx = xx - (nx-1)/2.0
    yy = yy - (ny-1)/2.0
    xxpsize = xx * psize
    yypsize = yy * psize

    out = -np.sum(np.sum(im**2 * (xxpsize**2 + yypsize**2)))
    return out/norm


def scompact2grad(imvec, nx, ny, psize, flux, norm_reg=NORM_REGULARIZER, beam_size=None):
    """Gradient for I^2r^2 source size regularizer
    """

    if beam_size is None:
        beam_size = psize
    if norm_reg:
        norm = flux**2 * beam_size**2
    else:
        norm = 1

    im = imvec.reshape(ny, nx)

    xx, yy = np.meshgrid(range(nx), range(ny))
    xx = xx - (nx-1)/2.0
    yy = yy - (ny-1)/2.0
    xxpsize = xx * psize
    yypsize = yy * psize

    grad = -2*im*(xxpsize**2 + yypsize**2)

    return grad.reshape(-1)/norm


def sgauss(imvec, xdim, ydim, psize, major, minor, PA):
    """Gaussian source size regularizer
    """

    # major, minor and PA are all in radians
    phi = PA

    # eigenvalues of covariance matrix
    lambda1 = minor**2./(8.*np.log(2.))
    lambda2 = major**2./(8.*np.log(2.))

    # now compute covariance matrix elements from user inputs
    sigxx_prime = lambda1*(np.cos(phi)**2.) + lambda2*(np.sin(phi)**2.)
    sigyy_prime = lambda1*(np.sin(phi)**2.) + lambda2*(np.cos(phi)**2.)
    sigxy_prime = (lambda2 - lambda1)*np.cos(phi)*np.sin(phi)

    # we get the dimensions and image vector
    im = imvec.reshape(ydim, xdim)
    xlist, ylist = np.meshgrid(range(xdim), range(ydim))
    xlist = xlist - (xdim-1)/2.0
    ylist = ylist - (ydim-1)/2.0

    xx = xlist * psize
    yy = ylist * psize

    # the centroid parameters
    x0 = np.sum(xx*im) / np.sum(im)
    y0 = np.sum(yy*im) / np.sum(im)

    # we calculate the elements of the covariance matrix
    sigxx = (np.sum((xx - x0)**2.*im)/np.sum(im))
    sigyy = (np.sum((yy - y0)**2.*im)/np.sum(im))
    sigxy = (np.sum((xx - x0)*(yy - y0)*im)/np.sum(im))

    # We calculate the regularizer #this line was CHANGED
    rgauss = -((sigxx - sigxx_prime)**2. + (sigyy - sigyy_prime)**2. + 2*(sigxy - sigxy_prime)**2.)
    # normalization will need to be redone, right now requires alpha~1000
    rgauss = rgauss/(major**2. * minor**2.)
    return rgauss


def sgauss_grad(imvec, xdim, ydim, psize, major, minor, PA):
    """Gradient for Gaussian source size regularizer
    """

    # major, minor and PA are all in radians
    phi = PA

    # computing eigenvalues of the covariance matrix
    lambda1 = (minor**2.)/(8.*np.log(2.))
    lambda2 = (major**2.)/(8.*np.log(2.))

    # now compute covariance matrix elements from user inputs

    sigxx_prime = lambda1*(np.cos(phi)**2.) + lambda2*(np.sin(phi)**2.)
    sigyy_prime = lambda1*(np.sin(phi)**2.) + lambda2*(np.cos(phi)**2.)
    sigxy_prime = (lambda2 - lambda1)*np.cos(phi)*np.sin(phi)

    # we get the dimensions and image vector
    im = imvec.reshape(ydim, xdim)
    xlist, ylist = np.meshgrid(range(xdim), range(ydim))
    xlist = xlist - (xdim-1)/2.0
    ylist = ylist - (ydim-1)/2.0

    xx = xlist * psize
    yy = ylist * psize

    # the centroid parameters
    x0 = np.sum(xx*im) / np.sum(im)
    y0 = np.sum(yy*im) / np.sum(im)

    # we calculate the elements of the covariance matrix of the image
    sigxx = (np.sum((xx - x0)**2.*im)/np.sum(im))
    sigyy = (np.sum((yy - y0)**2.*im)/np.sum(im))
    sigxy = (np.sum((xx - x0)*(yy - y0)*im)/np.sum(im))

    # gradients of covariance matrix elements
    # d(sig_ab)/d(im_k) = [(a_k - a0)(b_k - b0) - sig_ab] / sum(im)
    # cross-term through centroid dependence sums to zero by definition of centroid
    S = np.sum(im)
    dxx = ((xx - x0)**2. - sigxx) / S
    dyy = ((yy - y0)**2. - sigyy) / S
    dxy = ((xx - x0)*(yy - y0) - sigxy) / S

    # gradient of the regularizer #this line was CHANGED
    drgauss = (2.*(sigxx - sigxx_prime)*dxx +
               2.*(sigyy - sigyy_prime)*dyy +
               4.*(sigxy - sigxy_prime)*dxy)

    # normalization will need to be redone, right now requires alpha~1000
    drgauss = drgauss/(major**2. * minor**2.)

    return -drgauss.reshape(-1)


##################################################################################################
# Imager-backend wrappers
#
# Each `reg_X` / `reggrad_X` adapts an `sX` / `sXgrad` regularizer above to the
# uniform `(imvec, mask, **kwargs)` signature used by the `_REGULARIZER_DISPATCH`
# table in `imager_backend.py`. The wrappers add the sign convention (the
# imager minimizes `-regularizer`), unpack kwargs into the leaf's positional
# args, and handle the embed-pre / mask-post-slice pattern for spatial regs.
##################################################################################################


def reg_flux(imvec, mask, **kwargs):
    return -sflux(imvec, kwargs['nprior'], kwargs['flux'],
                  norm_reg=kwargs.get('norm_reg', True))


def reggrad_flux(imvec, mask, **kwargs):
    return -sfluxgrad(imvec, kwargs['nprior'], kwargs['flux'],
                      norm_reg=kwargs.get('norm_reg', True))


def reg_simple(imvec, mask, **kwargs):
    return -ssimple(imvec, kwargs['nprior'], kwargs['flux'],
                    norm_reg=kwargs.get('norm_reg', True))


def reggrad_simple(imvec, mask, **kwargs):
    return -ssimplegrad(imvec, kwargs['nprior'], kwargs['flux'],
                        norm_reg=kwargs.get('norm_reg', True))


def reg_l1(imvec, mask, **kwargs):
    return -sl1(imvec, kwargs['nprior'], kwargs['flux'],
                norm_reg=kwargs.get('norm_reg', True))


def reggrad_l1(imvec, mask, **kwargs):
    return -sl1grad(imvec, kwargs['nprior'], kwargs['flux'],
                    norm_reg=kwargs.get('norm_reg', True))


def reg_l1w(imvec, mask, **kwargs):
    return -sl1w(imvec, kwargs['nprior'], kwargs['flux'],
                 norm_reg=kwargs.get('norm_reg', True))


def reggrad_l1w(imvec, mask, **kwargs):
    return -sl1wgrad(imvec, kwargs['nprior'], kwargs['flux'],
                     norm_reg=kwargs.get('norm_reg', True))


def reg_lA(imvec, mask, **kwargs):
    return -slA(imvec, kwargs['nprior'], kwargs['psize'], kwargs['flux'],
                kwargs.get('beam_size'), kwargs.get('alpha_A', 1.0),
                kwargs.get('norm_reg', True))


def reggrad_lA(imvec, mask, **kwargs):
    return -slAgrad(imvec, kwargs['nprior'], kwargs['psize'], kwargs['flux'],
                    kwargs.get('beam_size'), kwargs.get('alpha_A', 1.0),
                    kwargs.get('norm_reg', True))


def reg_gs(imvec, mask, **kwargs):
    return -sgs(imvec, kwargs['nprior'], kwargs['flux'],
                norm_reg=kwargs.get('norm_reg', True))


def reggrad_gs(imvec, mask, **kwargs):
    return -sgsgrad(imvec, kwargs['nprior'], kwargs['flux'],
                    norm_reg=kwargs.get('norm_reg', True))


def reg_patch(imvec, mask, **kwargs):
    return -spatch(imvec, kwargs['nprior'], kwargs['flux'],
                   norm_reg=kwargs.get('norm_reg', True))


def reggrad_patch(imvec, mask, **kwargs):
    return -spatchgrad(imvec, kwargs['nprior'], kwargs['flux'],
                       norm_reg=kwargs.get('norm_reg', True))


# scm / scmgrad take the embed mask as a positional and handle masking internally,
# so these wrappers skip the embed-pre / mask-post pattern used by other spatial regs.
def reg_cm(imvec, mask, **kwargs):
    return -scm(imvec, kwargs['xdim'], kwargs['ydim'], kwargs['psize'],
                kwargs['flux'], mask,
                norm_reg=kwargs.get('norm_reg', True),
                beam_size=kwargs.get('beam_size'))


def reggrad_cm(imvec, mask, **kwargs):
    return -scmgrad(imvec, kwargs['xdim'], kwargs['ydim'], kwargs['psize'],
                    kwargs['flux'], mask,
                    norm_reg=kwargs.get('norm_reg', True),
                    beam_size=kwargs.get('beam_size'))


def reg_tv(imvec, mask, **kwargs):
    if np.any(np.invert(mask)):
        imvec = embed(imvec, mask, randomfloor=True)
    return -stv(imvec, kwargs['xdim'], kwargs['ydim'], kwargs['psize'], kwargs['flux'],
                norm_reg=kwargs.get('norm_reg', True),
                beam_size=kwargs.get('beam_size'),
                epsilon=kwargs.get('epsilon_tv', 0.))


def reggrad_tv(imvec, mask, **kwargs):
    if np.any(np.invert(mask)):
        imvec = embed(imvec, mask, randomfloor=True)
    g = -stvgrad(imvec, kwargs['xdim'], kwargs['ydim'], kwargs['psize'], kwargs['flux'],
                 norm_reg=kwargs.get('norm_reg', True),
                 beam_size=kwargs.get('beam_size'),
                 epsilon=kwargs.get('epsilon_tv', 0.))
    return g[mask]


# tvlog / tv2log use clipfloor=epsilon_tv (not the default 0) so the log
# transform inside stvlog / stv2log stays defined where mask filled in values.
def reg_tvlog(imvec, mask, **kwargs):
    epsilon = kwargs.get('epsilon_tv', 0.)
    if np.any(np.invert(mask)):
        imvec = embed(imvec, mask, clipfloor=epsilon, randomfloor=True)
    return -stvlog(imvec, kwargs['xdim'], kwargs['ydim'], kwargs['psize'], kwargs['flux'],
                   norm_reg=kwargs.get('norm_reg', True),
                   beam_size=kwargs.get('beam_size'),
                   epsilon=epsilon)


def reggrad_tvlog(imvec, mask, **kwargs):
    epsilon = kwargs.get('epsilon_tv', 0.)
    if np.any(np.invert(mask)):
        imvec = embed(imvec, mask, clipfloor=epsilon, randomfloor=True)
    g = -stvloggrad(imvec, kwargs['xdim'], kwargs['ydim'], kwargs['psize'], kwargs['flux'],
                    norm_reg=kwargs.get('norm_reg', True),
                    beam_size=kwargs.get('beam_size'),
                    epsilon=epsilon)
    return g[mask]


def reg_tv2(imvec, mask, **kwargs):
    if np.any(np.invert(mask)):
        imvec = embed(imvec, mask, randomfloor=True)
    return -stv2(imvec, kwargs['xdim'], kwargs['ydim'], kwargs['psize'], kwargs['flux'],
                 norm_reg=kwargs.get('norm_reg', True),
                 beam_size=kwargs.get('beam_size'))


def reggrad_tv2(imvec, mask, **kwargs):
    if np.any(np.invert(mask)):
        imvec = embed(imvec, mask, randomfloor=True)
    g = -stv2grad(imvec, kwargs['xdim'], kwargs['ydim'], kwargs['psize'], kwargs['flux'],
                  norm_reg=kwargs.get('norm_reg', True),
                  beam_size=kwargs.get('beam_size'))
    return g[mask]


def reg_tv2log(imvec, mask, **kwargs):
    if np.any(np.invert(mask)):
        imvec = embed(imvec, mask, randomfloor=True)
    return -stv2log(imvec, kwargs['xdim'], kwargs['ydim'], kwargs['psize'], kwargs['flux'],
                    norm_reg=kwargs.get('norm_reg', True),
                    beam_size=kwargs.get('beam_size'))


def reggrad_tv2log(imvec, mask, **kwargs):
    if np.any(np.invert(mask)):
        imvec = embed(imvec, mask, randomfloor=True)
    g = -stv2loggrad(imvec, kwargs['xdim'], kwargs['ydim'], kwargs['psize'], kwargs['flux'],
                     norm_reg=kwargs.get('norm_reg', True),
                     beam_size=kwargs.get('beam_size'))
    return g[mask]


def reg_compact(imvec, mask, **kwargs):
    if np.any(np.invert(mask)):
        imvec = embed(imvec, mask, randomfloor=True)
    return -scompact(imvec, kwargs['xdim'], kwargs['ydim'], kwargs['psize'], kwargs['flux'],
                     norm_reg=kwargs.get('norm_reg', True))


def reggrad_compact(imvec, mask, **kwargs):
    if np.any(np.invert(mask)):
        imvec = embed(imvec, mask, randomfloor=True)
    g = -scompactgrad(imvec, kwargs['xdim'], kwargs['ydim'], kwargs['psize'], kwargs['flux'],
                      norm_reg=kwargs.get('norm_reg', True))
    return g[mask]


def reg_compact2(imvec, mask, **kwargs):
    if np.any(np.invert(mask)):
        imvec = embed(imvec, mask, randomfloor=True)
    return -scompact2(imvec, kwargs['xdim'], kwargs['ydim'], kwargs['psize'], kwargs['flux'],
                      norm_reg=kwargs.get('norm_reg', True))


def reggrad_compact2(imvec, mask, **kwargs):
    if np.any(np.invert(mask)):
        imvec = embed(imvec, mask, randomfloor=True)
    g = -scompact2grad(imvec, kwargs['xdim'], kwargs['ydim'], kwargs['psize'], kwargs['flux'],
                       norm_reg=kwargs.get('norm_reg', True))
    return g[mask]


def reg_rgauss(imvec, mask, **kwargs):
    if np.any(np.invert(mask)):
        imvec = embed(imvec, mask, randomfloor=True)
    return -sgauss(imvec, kwargs['xdim'], kwargs['ydim'], kwargs['psize'],
                   major=kwargs.get('major', 1.0),
                   minor=kwargs.get('minor', 1.0),
                   PA=kwargs.get('PA', 1.0))


def reggrad_rgauss(imvec, mask, **kwargs):
    if np.any(np.invert(mask)):
        imvec = embed(imvec, mask, randomfloor=True)
    g = -sgauss_grad(imvec, kwargs['xdim'], kwargs['ydim'], kwargs['psize'],
                     kwargs.get('major', 1.0),
                     kwargs.get('minor', 1.0),
                     kwargs.get('PA', 1.0))
    return g[mask]


##################################################################################################
# Chi^2 Data functions
##################################################################################################
def apply_systematic_noise_snrcut(data_arr, systematic_noise, snrcut, pol):
    """apply systematic noise to VISIBILITIES or AMPLITUDES
       data_arr should have fields 't1','t2','u','v','vis','amp','sigma'

       returns: (uv, vis, amp, sigma)
    """

    vtype = ehc.vis_poldict[pol]
    atype = ehc.amp_poldict[pol]
    etype = ehc.sig_poldict[pol]

    t1 = data_arr['t1']
    t2 = data_arr['t2']

    sigma = data_arr[etype]
    amp = data_arr[atype]
    try:
        vis = data_arr[vtype]
    except ValueError:
        vis = amp.astype('c16')

    snrmask = np.abs(amp/sigma) >= snrcut

    if type(systematic_noise) is dict:
        sys_level = np.zeros(len(t1))
        for i in range(len(t1)):
            if t1[i] in systematic_noise.keys():
                t1sys = systematic_noise[t1[i]]
            else:
                t1sys = 0.
            if t2[i] in systematic_noise.keys():
                t2sys = systematic_noise[t2[i]]
            else:
                t2sys = 0.

            if t1sys < 0 or t2sys < 0:
                sys_level[i] = -1
            else:
                sys_level[i] = np.sqrt(t1sys**2 + t2sys**2)
    else:
        sys_level = np.sqrt(2)*systematic_noise*np.ones(len(t1))

    mask = sys_level >= 0.
    mask = snrmask * mask

    sigma = np.linalg.norm([sigma, sys_level*np.abs(amp)], axis=0)[mask]
    vis = vis[mask]
    amp = amp[mask]
    uv = np.hstack((data_arr['u'].reshape(-1, 1), data_arr['v'].reshape(-1, 1)))[mask]
    return (uv, vis, amp, sigma)


def chisqdata_vis(Obsdata, Prior, mask, pol='I', **kwargs):
    """Return the data, sigmas, and fourier matrix for visibilities
    """

    # unpack keyword args
    systematic_noise = kwargs.get('systematic_noise', 0.)
    snrcut = kwargs.get('snrcut', 0.)
    debias = kwargs.get('debias', False)
    weighting = kwargs.get('weighting', 'natural')

    # unpack data
    vtype = ehc.vis_poldict[pol]
    atype = ehc.amp_poldict[pol]
    etype = ehc.sig_poldict[pol]
    data_arr = Obsdata.unpack(['t1', 't2', 'u', 'v', vtype, atype, etype], debias=debias)
    (uv, vis, amp, sigma) = apply_systematic_noise_snrcut(data_arr, systematic_noise, snrcut, pol)

    # data weighting
    if weighting == 'uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # make fourier matrix
    A = obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv, pulse=Prior.pulse, mask=mask)

    return (vis, sigma, A)


def chisqdata_amp(Obsdata, Prior, mask, pol='I', **kwargs):
    """Return the data, sigmas, and fourier matrix for visibility amplitudes
    """

    # unpack keyword args
    systematic_noise = kwargs.get('systematic_noise', 0.)
    snrcut = kwargs.get('snrcut', 0.)
    debias = kwargs.get('debias', False)
    weighting = kwargs.get('weighting', 'natural')

    # unpack data
    vtype = ehc.vis_poldict[pol]
    atype = ehc.amp_poldict[pol]
    etype = ehc.sig_poldict[pol]
    data_arr = Obsdata.unpack(['t1', 't2', 'u', 'v', vtype, atype, etype], debias=debias)

    # apply systematic noise and SNR cut
    # TODO -- after pre-computed??
    (uv, vis, amp, sigma) = apply_systematic_noise_snrcut(data_arr, systematic_noise, snrcut, pol)

    # data weighting
    if weighting == 'uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # make fourier matrix
    A = obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv, pulse=Prior.pulse, mask=mask)

    return (amp, sigma, A)


def chisqdata_bs(Obsdata, Prior, mask, pol='I', **kwargs):
    """return the data, sigmas, and fourier matrices for bispectra
    """

    # unpack keyword args
    # systematic_noise = kwargs.get('systematic_noise',0.)
    maxset = kwargs.get('maxset', False)
    if maxset:
        count = 'max'
    else:
        count = 'min'

    snrcut = kwargs.get('snrcut', 0.)
    weighting = kwargs.get('weighting', 'natural')

    # unpack data
    vtype = ehc.vis_poldict[pol]
    biarr = Obsdata.bispectra(mode="all", vtype=vtype, count=count, snrcut=snrcut)

    uv1 = np.hstack((biarr['u1'].reshape(-1, 1), biarr['v1'].reshape(-1, 1)))
    uv2 = np.hstack((biarr['u2'].reshape(-1, 1), biarr['v2'].reshape(-1, 1)))
    uv3 = np.hstack((biarr['u3'].reshape(-1, 1), biarr['v3'].reshape(-1, 1)))
    bi = biarr['bispec']
    sigma = biarr['sigmab']

    # add systematic noise
    # sigma = np.linalg.norm([biarr['sigmab'], systematic_noise*np.abs(biarr['bispec'])], axis=0)

    # data weighting
    if weighting == 'uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # make fourier matrices
    A3 = (obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse, mask=mask),
          obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse, mask=mask),
          obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse, mask=mask)
         )

    return (bi, sigma, A3)


def chisqdata_cphase(Obsdata, Prior, mask, pol='I', **kwargs):
    """Return the data, sigmas, and fourier matrices for closure phases
    """

    # unpack keyword args
    maxset = kwargs.get('maxset', False)
    uv_min = kwargs.get('cp_uv_min', False)
    if maxset:
        count = 'max'
    else:
        count = 'min'

    snrcut = kwargs.get('snrcut', 0.)
    systematic_cphase_noise = kwargs.get('systematic_cphase_noise', 0.)
    weighting = kwargs.get('weighting', 'natural')

    # unpack data
    vtype = ehc.vis_poldict[pol]
    clphasearr = Obsdata.c_phases(mode="all", vtype=vtype,
                                  count=count, uv_min=uv_min, snrcut=snrcut)

    uv1 = np.hstack((clphasearr['u1'].reshape(-1, 1), clphasearr['v1'].reshape(-1, 1)))
    uv2 = np.hstack((clphasearr['u2'].reshape(-1, 1), clphasearr['v2'].reshape(-1, 1)))
    uv3 = np.hstack((clphasearr['u3'].reshape(-1, 1), clphasearr['v3'].reshape(-1, 1)))
    clphase = clphasearr['cphase']
    sigma = clphasearr['sigmacp']

    # add systematic cphase noise (in DEGREES)
    sigma = np.linalg.norm([sigma, systematic_cphase_noise*np.ones(len(sigma))], axis=0)

    # data weighting
    if weighting == 'uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # make fourier matrices
    A3 = (obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse, mask=mask),
          obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse, mask=mask),
          obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse, mask=mask)
          )
    return (clphase, sigma, A3)


def chisqdata_cphase_diag(Obsdata, Prior, mask, pol='I', **kwargs):
    """Return the data, sigmas, and fourier matrices for diagonalized closure phases
    """

    # unpack keyword args
    maxset = kwargs.get('maxset', False)
    uv_min = kwargs.get('cp_uv_min', False)
    if maxset:
        count = 'max'
    else:
        count = 'min'

    snrcut = kwargs.get('snrcut', 0.)

    # unpack data
    vtype = ehc.vis_poldict[pol]
    clphasearr = Obsdata.c_phases_diag(vtype=vtype, count=count, snrcut=snrcut, uv_min=uv_min)

    # loop over timestamps
    clphase_diag = []
    sigma_diag = []
    A3_diag = []
    tform_mats = []
    for ic, cl in enumerate(clphasearr):

        # get diagonalized closure phases and errors
        clphase_diag.append(cl[0]['cphase'])
        sigma_diag.append(cl[0]['sigmacp'])

        # get uv arrays
        u1 = cl[2][:, 0].astype('float')
        v1 = cl[3][:, 0].astype('float')
        uv1 = np.hstack((u1.reshape(-1, 1), v1.reshape(-1, 1)))

        u2 = cl[2][:, 1].astype('float')
        v2 = cl[3][:, 1].astype('float')
        uv2 = np.hstack((u2.reshape(-1, 1), v2.reshape(-1, 1)))

        u3 = cl[2][:, 2].astype('float')
        v3 = cl[3][:, 2].astype('float')
        uv3 = np.hstack((u3.reshape(-1, 1), v3.reshape(-1, 1)))

        # compute Fourier matrices
        A3 = (obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse, mask=mask),
              obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse, mask=mask),
              obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse, mask=mask)
              )
        A3_diag.append(A3)

        # get transformation matrix for this timestamp
        tform_mats.append(cl[4].astype('float'))

    # combine Fourier and transformation matrices into tuple for outputting.
    # Per-timestamp baseline counts vary, so the outer arrays are ragged --
    # NumPy >= 1.24 requires explicit dtype=object for inhomogeneous shape.
    Amatrices = (np.array(A3_diag, dtype=object), np.array(tform_mats, dtype=object))

    return (np.array(clphase_diag, dtype=object), np.array(sigma_diag, dtype=object), Amatrices)


def chisqdata_camp(Obsdata, Prior, mask, pol='I', **kwargs):
    """Return the data, sigmas, and fourier matrices for closure amplitudes
    """
    # unpack keyword args
    maxset = kwargs.get('maxset', False)
    if maxset:
        count = 'max'
    else:
        count = 'min'

    snrcut = kwargs.get('snrcut', 0.)
    debias = kwargs.get('debias', False)
    weighting = kwargs.get('weighting', 'natural')

    # unpack data & mask low snr points
    vtype = ehc.vis_poldict[pol]
    clamparr = Obsdata.c_amplitudes(mode='all', count=count,
                                    vtype=vtype, ctype='camp', debias=debias, snrcut=snrcut)

    uv1 = np.hstack((clamparr['u1'].reshape(-1, 1), clamparr['v1'].reshape(-1, 1)))
    uv2 = np.hstack((clamparr['u2'].reshape(-1, 1), clamparr['v2'].reshape(-1, 1)))
    uv3 = np.hstack((clamparr['u3'].reshape(-1, 1), clamparr['v3'].reshape(-1, 1)))
    uv4 = np.hstack((clamparr['u4'].reshape(-1, 1), clamparr['v4'].reshape(-1, 1)))
    clamp = clamparr['camp']
    sigma = clamparr['sigmaca']

    # data weighting
    if weighting == 'uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # make fourier matrices
    A4 = (obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse, mask=mask),
          obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse, mask=mask),
          obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse, mask=mask),
          obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv4, pulse=Prior.pulse, mask=mask)
          )

    return (clamp, sigma, A4)


def chisqdata_logcamp(Obsdata, Prior, mask, pol='I', **kwargs):
    """Return the data, sigmas, and fourier matrices for log closure amplitudes
    """
    # unpack keyword args
    maxset = kwargs.get('maxset', False)
    if maxset:
        count = 'max'
    else:
        count = 'min'

    snrcut = kwargs.get('snrcut', 0.)
    debias = kwargs.get('debias', False)
    weighting = kwargs.get('weighting', 'natural')

    # unpack data & mask low snr points
    vtype = ehc.vis_poldict[pol]
    clamparr = Obsdata.c_amplitudes(mode='all', count=count,
                                    vtype=vtype, ctype='logcamp', debias=debias, snrcut=snrcut)

    uv1 = np.hstack((clamparr['u1'].reshape(-1, 1), clamparr['v1'].reshape(-1, 1)))
    uv2 = np.hstack((clamparr['u2'].reshape(-1, 1), clamparr['v2'].reshape(-1, 1)))
    uv3 = np.hstack((clamparr['u3'].reshape(-1, 1), clamparr['v3'].reshape(-1, 1)))
    uv4 = np.hstack((clamparr['u4'].reshape(-1, 1), clamparr['v4'].reshape(-1, 1)))
    clamp = clamparr['camp']
    sigma = clamparr['sigmaca']

    # data weighting
    if weighting == 'uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # make fourier matrices
    A4 = (obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse, mask=mask),
          obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse, mask=mask),
          obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse, mask=mask),
          obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv4, pulse=Prior.pulse, mask=mask)
          )

    return (clamp, sigma, A4)


def chisqdata_logcamp_diag(Obsdata, Prior, mask, pol='I', **kwargs):
    """Return the data, sigmas, and fourier matrices for diagonalized log closure amplitudes
    """
    # unpack keyword args
    maxset = kwargs.get('maxset', False)
    if maxset:
        count = 'max'
    else:
        count = 'min'

    snrcut = kwargs.get('snrcut', 0.)
    debias = kwargs.get('debias', False)

    # unpack data & mask low snr points
    vtype = ehc.vis_poldict[pol]
    clamparr = Obsdata.c_log_amplitudes_diag(vtype=vtype, count=count, debias=debias, snrcut=snrcut)

    # loop over timestamps
    clamp_diag = []
    sigma_diag = []
    A4_diag = []
    tform_mats = []
    for ic, cl in enumerate(clamparr):

        # get diagonalized log closure amplitudes and errors
        clamp_diag.append(cl[0]['camp'])
        sigma_diag.append(cl[0]['sigmaca'])

        # get uv arrays
        u1 = cl[2][:, 0].astype('float')
        v1 = cl[3][:, 0].astype('float')
        uv1 = np.hstack((u1.reshape(-1, 1), v1.reshape(-1, 1)))

        u2 = cl[2][:, 1].astype('float')
        v2 = cl[3][:, 1].astype('float')
        uv2 = np.hstack((u2.reshape(-1, 1), v2.reshape(-1, 1)))

        u3 = cl[2][:, 2].astype('float')
        v3 = cl[3][:, 2].astype('float')
        uv3 = np.hstack((u3.reshape(-1, 1), v3.reshape(-1, 1)))

        u4 = cl[2][:, 3].astype('float')
        v4 = cl[3][:, 3].astype('float')
        uv4 = np.hstack((u4.reshape(-1, 1), v4.reshape(-1, 1)))

        # compute Fourier matrices
        A4 = (obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse, mask=mask),
              obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse, mask=mask),
              obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse, mask=mask),
              obsh.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv4, pulse=Prior.pulse, mask=mask)
              )
        A4_diag.append(A4)

        # get transformation matrix for this timestamp
        tform_mats.append(cl[4].astype('float'))

    # combine Fourier and transformation matrices into tuple for outputting.
    # Per-timestamp baseline counts vary, so the outer arrays are ragged --
    # NumPy >= 1.24 requires explicit dtype=object for inhomogeneous shape.
    Amatrices = (np.array(A4_diag, dtype=object), np.array(tform_mats, dtype=object))

    return (np.array(clamp_diag, dtype=object), np.array(sigma_diag, dtype=object), Amatrices)

##################################################################################################
# FFT Chi^2 Data functions
##################################################################################################


def chisqdata_vis_fft(Obsdata, Prior, pol='I', **kwargs):
    """Return the data, sigmas, uv points, and FFT info for visibilities
    """

    # unpack keyword args
    systematic_noise = kwargs.get('systematic_noise', 0.)
    snrcut = kwargs.get('snrcut', 0.)
    debias = kwargs.get('debias', False)
    weighting = kwargs.get('weighting', 'natural')
    fft_pad_factor = kwargs.get('fft_pad_factor', ehc.FFT_PAD_DEFAULT)
    conv_func = kwargs.get('conv_func', ehc.GRIDDER_CONV_FUNC_DEFAULT)
    p_rad = kwargs.get('p_rad', ehc.GRIDDER_P_RAD_DEFAULT)
    order = kwargs.get('order', ehc.FFT_INTERP_DEFAULT)

    # unpack data
    vtype = ehc.vis_poldict[pol]
    atype = ehc.amp_poldict[pol]
    etype = ehc.sig_poldict[pol]
    data_arr = Obsdata.unpack(['t1', 't2', 'u', 'v', vtype, atype, etype], debias=debias)
    (uv, vis, amp, sigma) = apply_systematic_noise_snrcut(data_arr, systematic_noise, snrcut, pol)

    # data weighting
    if weighting == 'uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # prepare image and fft info objects
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    im_info = obsh.ImInfo(Prior.xdim, Prior.ydim, npad, Prior.psize, Prior.pulse)
    gs_info = obsh.make_gridder_and_sampler_info(
        im_info, uv, conv_func=conv_func, p_rad=p_rad, order=order)

    sampler_info_list = [gs_info[0]]
    gridder_info_list = [gs_info[1]]
    A = (im_info, sampler_info_list, gridder_info_list)

    return (vis, sigma, A)


def chisqdata_amp_fft(Obsdata, Prior, pol='I', **kwargs):
    """Return the data, sigmas, uv points, and FFT info for visibility amplitudes
    """

    # unpack keyword args
    systematic_noise = kwargs.get('systematic_noise', 0.)
    snrcut = kwargs.get('snrcut', 0.)
    debias = kwargs.get('debias', False)
    weighting = kwargs.get('weighting', 'natural')
    fft_pad_factor = kwargs.get('fft_pad_factor', ehc.FFT_PAD_DEFAULT)
    conv_func = kwargs.get('conv_func', ehc.GRIDDER_CONV_FUNC_DEFAULT)
    p_rad = kwargs.get('p_rad', ehc.GRIDDER_P_RAD_DEFAULT)
    order = kwargs.get('order', ehc.FFT_INTERP_DEFAULT)

    # unpack data
    vtype = ehc.vis_poldict[pol]
    atype = ehc.amp_poldict[pol]
    etype = ehc.sig_poldict[pol]
    data_arr = Obsdata.unpack(['t1', 't2', 'u', 'v', vtype, atype, etype], debias=debias)

    # apply systematic noise
    (uv, vis, amp, sigma) = apply_systematic_noise_snrcut(data_arr, systematic_noise, snrcut, pol)

    # data weighting
    if weighting == 'uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # prepare image and fft info objects
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    im_info = obsh.ImInfo(Prior.xdim, Prior.ydim, npad, Prior.psize, Prior.pulse)
    gs_info = obsh.make_gridder_and_sampler_info(im_info, uv,
                                                 conv_func=conv_func, p_rad=p_rad, order=order)

    sampler_info_list = [gs_info[0]]
    gridder_info_list = [gs_info[1]]
    A = (im_info, sampler_info_list, gridder_info_list)

    return (amp, sigma, A)


def chisqdata_bs_fft(Obsdata, Prior, pol='I', **kwargs):
    """Return the data, sigmas, uv points, and FFT info for bispectra
    """

    # unpack keyword args
    # systematic_noise = kwargs.get('systematic_noise',0.)
    maxset = kwargs.get('maxset', False)
    if maxset:
        count = 'max'
    else:
        count = 'min'

    snrcut = kwargs.get('snrcut', 0.)
    weighting = kwargs.get('weighting', 'natural')
    fft_pad_factor = kwargs.get('fft_pad_factor', ehc.FFT_PAD_DEFAULT)
    conv_func = kwargs.get('conv_func', ehc.GRIDDER_CONV_FUNC_DEFAULT)
    p_rad = kwargs.get('p_rad', ehc.GRIDDER_P_RAD_DEFAULT)
    order = kwargs.get('order', ehc.FFT_INTERP_DEFAULT)

    # unpack data
    vtype = ehc.vis_poldict[pol]
    biarr = Obsdata.bispectra(mode="all", vtype=vtype, count=count, snrcut=snrcut)

    uv1 = np.hstack((biarr['u1'].reshape(-1, 1), biarr['v1'].reshape(-1, 1)))
    uv2 = np.hstack((biarr['u2'].reshape(-1, 1), biarr['v2'].reshape(-1, 1)))
    uv3 = np.hstack((biarr['u3'].reshape(-1, 1), biarr['v3'].reshape(-1, 1)))
    bi = biarr['bispec']
    sigma = biarr['sigmab']

    # add systematic noise
    # sigma = np.linalg.norm([biarr['sigmab'], systematic_noise*np.abs(biarr['bispec'])], axis=0)

    # data weighting
    if weighting == 'uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # prepare image and fft info objects
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    im_info = obsh.ImInfo(Prior.xdim, Prior.ydim, npad, Prior.psize, Prior.pulse)
    gs_info1 = obsh.make_gridder_and_sampler_info(im_info, uv1,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info2 = obsh.make_gridder_and_sampler_info(im_info, uv2,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info3 = obsh.make_gridder_and_sampler_info(im_info, uv3,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)

    sampler_info_list = [gs_info1[0], gs_info2[0], gs_info3[0]]
    gridder_info_list = [gs_info1[1], gs_info2[1], gs_info3[1]]
    A = (im_info, sampler_info_list, gridder_info_list)

    return (bi, sigma, A)


def chisqdata_cphase_fft(Obsdata, Prior, pol='I', **kwargs):
    """Return the data, sigmas, uv points, and FFT info for closure phases
    """
    # unpack keyword args
    maxset = kwargs.get('maxset', False)
    uv_min = kwargs.get('cp_uv_min', False)
    if maxset:
        count = 'max'
    else:
        count = 'min'

    snrcut = kwargs.get('snrcut', 0.)
    weighting = kwargs.get('weighting', 'natural')
    systematic_cphase_noise = kwargs.get('systematic_cphase_noise', 0.)
    fft_pad_factor = kwargs.get('fft_pad_factor', ehc.FFT_PAD_DEFAULT)
    conv_func = kwargs.get('conv_func', ehc.GRIDDER_CONV_FUNC_DEFAULT)
    p_rad = kwargs.get('p_rad', ehc.GRIDDER_P_RAD_DEFAULT)
    order = kwargs.get('order', ehc.FFT_INTERP_DEFAULT)

    # unpack data
    vtype = ehc.vis_poldict[pol]
    clphasearr = Obsdata.c_phases(mode="all", vtype=vtype,
                                  count=count, uv_min=uv_min, snrcut=snrcut)

    uv1 = np.hstack((clphasearr['u1'].reshape(-1, 1), clphasearr['v1'].reshape(-1, 1)))
    uv2 = np.hstack((clphasearr['u2'].reshape(-1, 1), clphasearr['v2'].reshape(-1, 1)))
    uv3 = np.hstack((clphasearr['u3'].reshape(-1, 1), clphasearr['v3'].reshape(-1, 1)))
    clphase = clphasearr['cphase']
    sigma = clphasearr['sigmacp']

    # add systematic cphase noise (in DEGREES)
    sigma = np.linalg.norm([sigma, systematic_cphase_noise*np.ones(len(sigma))], axis=0)

    # data weighting
    if weighting == 'uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # prepare image and fft info objects
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    im_info = obsh.ImInfo(Prior.xdim, Prior.ydim, npad, Prior.psize, Prior.pulse)
    gs_info1 = obsh.make_gridder_and_sampler_info(im_info, uv1,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info2 = obsh.make_gridder_and_sampler_info(im_info, uv2,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info3 = obsh.make_gridder_and_sampler_info(im_info, uv3,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)

    sampler_info_list = [gs_info1[0], gs_info2[0], gs_info3[0]]
    gridder_info_list = [gs_info1[1], gs_info2[1], gs_info3[1]]
    A = (im_info, sampler_info_list, gridder_info_list)

    return (clphase, sigma, A)


def chisqdata_cphase_diag_fft(Obsdata, Prior, pol='I', **kwargs):
    """Return the data, sigmas, uv points, and FFT info for diagonalized closure phases
    """
    # unpack keyword args
    maxset = kwargs.get('maxset', False)
    uv_min = kwargs.get('cp_uv_min', False)
    if maxset:
        count = 'max'
    else:
        count = 'min'

    snrcut = kwargs.get('snrcut', 0.)
    fft_pad_factor = kwargs.get('fft_pad_factor', ehc.FFT_PAD_DEFAULT)
    conv_func = kwargs.get('conv_func', ehc.GRIDDER_CONV_FUNC_DEFAULT)
    p_rad = kwargs.get('p_rad', ehc.GRIDDER_P_RAD_DEFAULT)
    order = kwargs.get('order', ehc.FFT_INTERP_DEFAULT)

    # unpack data
    vtype = ehc.vis_poldict[pol]
    clphasearr = Obsdata.c_phases_diag(vtype=vtype, count=count, snrcut=snrcut, uv_min=uv_min)

    # loop over timestamps
    clphase_diag = []
    sigma_diag = []
    tform_mats = []
    u1 = []
    v1 = []
    u2 = []
    v2 = []
    u3 = []
    v3 = []
    for ic, cl in enumerate(clphasearr):

        # get diagonalized closure phases and errors
        clphase_diag.append(cl[0]['cphase'])
        sigma_diag.append(cl[0]['sigmacp'])

        # get u and v values
        u1.append(cl[2][:, 0].astype('float'))
        v1.append(cl[3][:, 0].astype('float'))
        u2.append(cl[2][:, 1].astype('float'))
        v2.append(cl[3][:, 1].astype('float'))
        u3.append(cl[2][:, 2].astype('float'))
        v3.append(cl[3][:, 2].astype('float'))

        # get transformation matrix for this timestamp
        tform_mats.append(cl[4].astype('float'))

    # fix formatting of arrays
    u1 = np.concatenate(u1)
    v1 = np.concatenate(v1)
    u2 = np.concatenate(u2)
    v2 = np.concatenate(v2)
    u3 = np.concatenate(u3)
    v3 = np.concatenate(v3)

    # get uv arrays
    uv1 = np.hstack((u1.reshape(-1, 1), v1.reshape(-1, 1)))
    uv2 = np.hstack((u2.reshape(-1, 1), v2.reshape(-1, 1)))
    uv3 = np.hstack((u3.reshape(-1, 1), v3.reshape(-1, 1)))

    # prepare image and fft info objects
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    im_info = obsh.ImInfo(Prior.xdim, Prior.ydim, npad, Prior.psize, Prior.pulse)
    gs_info1 = obsh.make_gridder_and_sampler_info(im_info, uv1,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info2 = obsh.make_gridder_and_sampler_info(im_info, uv2,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info3 = obsh.make_gridder_and_sampler_info(im_info, uv3,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)

    sampler_info_list = [gs_info1[0], gs_info2[0], gs_info3[0]]
    gridder_info_list = [gs_info1[1], gs_info2[1], gs_info3[1]]
    A3 = (im_info, sampler_info_list, gridder_info_list)

    # Per-timestamp transform matrices have varying shapes; NumPy >= 1.24
    # requires explicit dtype=object for inhomogeneous arrays.
    Amatrices = (A3, np.array(tform_mats, dtype=object))

    return (np.array(clphase_diag, dtype=object), np.array(sigma_diag, dtype=object), Amatrices)


def chisqdata_camp_fft(Obsdata, Prior, pol='I', **kwargs):
    """Return the data, sigmas, uv points, and FFT info for closure amplitudes
    """

    # unpack keyword args
    maxset = kwargs.get('maxset', False)
    if maxset:
        count = 'max'
    else:
        count = 'min'

    snrcut = kwargs.get('snrcut', 0.)
    debias = kwargs.get('debias', False)
    weighting = kwargs.get('weighting', 'natural')
    fft_pad_factor = kwargs.get('fft_pad_factor', ehc.FFT_PAD_DEFAULT)
    conv_func = kwargs.get('conv_func', ehc.GRIDDER_CONV_FUNC_DEFAULT)
    p_rad = kwargs.get('p_rad', ehc.GRIDDER_P_RAD_DEFAULT)
    order = kwargs.get('order', ehc.FFT_INTERP_DEFAULT)

    # unpack data
    vtype = ehc.vis_poldict[pol]
    clamparr = Obsdata.c_amplitudes(mode='all', count=count,
                                    vtype=vtype, ctype='camp', debias=debias, snrcut=snrcut)

    uv1 = np.hstack((clamparr['u1'].reshape(-1, 1), clamparr['v1'].reshape(-1, 1)))
    uv2 = np.hstack((clamparr['u2'].reshape(-1, 1), clamparr['v2'].reshape(-1, 1)))
    uv3 = np.hstack((clamparr['u3'].reshape(-1, 1), clamparr['v3'].reshape(-1, 1)))
    uv4 = np.hstack((clamparr['u4'].reshape(-1, 1), clamparr['v4'].reshape(-1, 1)))
    clamp = clamparr['camp']
    sigma = clamparr['sigmaca']

    # data weighting
    if weighting == 'uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # prepare image and fft info objects
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    im_info = obsh.ImInfo(Prior.xdim, Prior.ydim, npad, Prior.psize, Prior.pulse)
    gs_info1 = obsh.make_gridder_and_sampler_info(im_info, uv1,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info2 = obsh.make_gridder_and_sampler_info(im_info, uv2,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info3 = obsh.make_gridder_and_sampler_info(im_info, uv3,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info4 = obsh.make_gridder_and_sampler_info(im_info, uv4,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)

    sampler_info_list = [gs_info1[0], gs_info2[0], gs_info3[0], gs_info4[0]]
    gridder_info_list = [gs_info1[1], gs_info2[1], gs_info3[1], gs_info4[1]]
    A = (im_info, sampler_info_list, gridder_info_list)

    return (clamp, sigma, A)


def chisqdata_logcamp_fft(Obsdata, Prior, pol='I', **kwargs):
    """Return the data, sigmas, uv points, and FFT info for log closure amplitudes
    """
    # unpack keyword args
    maxset = kwargs.get('maxset', False)
    if maxset:
        count = 'max'
    else:
        count = 'min'

    snrcut = kwargs.get('snrcut', 0.)
    debias = kwargs.get('debias', False)
    weighting = kwargs.get('weighting', 'natural')
    fft_pad_factor = kwargs.get('fft_pad_factor', ehc.FFT_PAD_DEFAULT)
    conv_func = kwargs.get('conv_func', ehc.GRIDDER_CONV_FUNC_DEFAULT)
    p_rad = kwargs.get('p_rad', ehc.GRIDDER_P_RAD_DEFAULT)
    order = kwargs.get('order', ehc.FFT_INTERP_DEFAULT)

    # unpack data
    vtype = ehc.vis_poldict[pol]
    clamparr = Obsdata.c_amplitudes(mode='all', count=count,
                                    vtype=vtype, ctype='logcamp', debias=debias, snrcut=snrcut)

    uv1 = np.hstack((clamparr['u1'].reshape(-1, 1), clamparr['v1'].reshape(-1, 1)))
    uv2 = np.hstack((clamparr['u2'].reshape(-1, 1), clamparr['v2'].reshape(-1, 1)))
    uv3 = np.hstack((clamparr['u3'].reshape(-1, 1), clamparr['v3'].reshape(-1, 1)))
    uv4 = np.hstack((clamparr['u4'].reshape(-1, 1), clamparr['v4'].reshape(-1, 1)))
    clamp = clamparr['camp']
    sigma = clamparr['sigmaca']

    # data weighting
    if weighting == 'uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # prepare image and fft info objects
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    im_info = obsh.ImInfo(Prior.xdim, Prior.ydim, npad, Prior.psize, Prior.pulse)
    gs_info1 = obsh.make_gridder_and_sampler_info(im_info, uv1,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info2 = obsh.make_gridder_and_sampler_info(im_info, uv2,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info3 = obsh.make_gridder_and_sampler_info(im_info, uv3,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info4 = obsh.make_gridder_and_sampler_info(im_info, uv4,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)

    sampler_info_list = [gs_info1[0], gs_info2[0], gs_info3[0], gs_info4[0]]
    gridder_info_list = [gs_info1[1], gs_info2[1], gs_info3[1], gs_info4[1]]
    A = (im_info, sampler_info_list, gridder_info_list)

    return (clamp, sigma, A)


def chisqdata_logcamp_diag_fft(Obsdata, Prior, pol='I', **kwargs):
    """Return the data, sigmas, uv points, and FFT info for diagonalized log closure amplitudes
    """
    # unpack keyword args
    maxset = kwargs.get('maxset', False)
    if maxset:
        count = 'max'
    else:
        count = 'min'

    snrcut = kwargs.get('snrcut', 0.)
    debias = kwargs.get('debias', False)
    fft_pad_factor = kwargs.get('fft_pad_factor', ehc.FFT_PAD_DEFAULT)
    conv_func = kwargs.get('conv_func', ehc.GRIDDER_CONV_FUNC_DEFAULT)
    p_rad = kwargs.get('p_rad', ehc.GRIDDER_P_RAD_DEFAULT)
    order = kwargs.get('order', ehc.FFT_INTERP_DEFAULT)

    # unpack data & mask low snr points
    vtype = ehc.vis_poldict[pol]
    clamparr = Obsdata.c_log_amplitudes_diag(vtype=vtype, count=count,
                                             debias=debias, snrcut=snrcut)

    # loop over timestamps
    clamp_diag = []
    sigma_diag = []
    tform_mats = []
    u1 = []
    v1 = []
    u2 = []
    v2 = []
    u3 = []
    v3 = []
    u4 = []
    v4 = []
    for ic, cl in enumerate(clamparr):

        # get diagonalized log closure amplitudes and errors
        clamp_diag.append(cl[0]['camp'])
        sigma_diag.append(cl[0]['sigmaca'])

        # get u and v values
        u1.append(cl[2][:, 0].astype('float'))
        v1.append(cl[3][:, 0].astype('float'))
        u2.append(cl[2][:, 1].astype('float'))
        v2.append(cl[3][:, 1].astype('float'))
        u3.append(cl[2][:, 2].astype('float'))
        v3.append(cl[3][:, 2].astype('float'))
        u4.append(cl[2][:, 3].astype('float'))
        v4.append(cl[3][:, 3].astype('float'))

        # get transformation matrix for this timestamp
        tform_mats.append(cl[4].astype('float'))

    # fix formatting of arrays
    u1 = np.concatenate(u1)
    v1 = np.concatenate(v1)
    u2 = np.concatenate(u2)
    v2 = np.concatenate(v2)
    u3 = np.concatenate(u3)
    v3 = np.concatenate(v3)
    u4 = np.concatenate(u4)
    v4 = np.concatenate(v4)

    # get uv arrays
    uv1 = np.hstack((u1.reshape(-1, 1), v1.reshape(-1, 1)))
    uv2 = np.hstack((u2.reshape(-1, 1), v2.reshape(-1, 1)))
    uv3 = np.hstack((u3.reshape(-1, 1), v3.reshape(-1, 1)))
    uv4 = np.hstack((u4.reshape(-1, 1), v4.reshape(-1, 1)))

    # prepare image and fft info objects
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    im_info = obsh.ImInfo(Prior.xdim, Prior.ydim, npad, Prior.psize, Prior.pulse)
    gs_info1 = obsh.make_gridder_and_sampler_info(im_info, uv1,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info2 = obsh.make_gridder_and_sampler_info(im_info, uv2,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info3 = obsh.make_gridder_and_sampler_info(im_info, uv3,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)
    gs_info4 = obsh.make_gridder_and_sampler_info(im_info, uv4,
                                                  conv_func=conv_func, p_rad=p_rad, order=order)

    sampler_info_list = [gs_info1[0], gs_info2[0], gs_info3[0], gs_info4[0]]
    gridder_info_list = [gs_info1[1], gs_info2[1], gs_info3[1], gs_info4[1]]
    A = (im_info, sampler_info_list, gridder_info_list)

    # Per-timestamp transform matrices have varying shapes; NumPy >= 1.24
    # requires explicit dtype=object for inhomogeneous arrays.
    Amatrices = (A, np.array(tform_mats, dtype=object))

    return (np.array(clamp_diag, dtype=object), np.array(sigma_diag, dtype=object), Amatrices)

##################################################################################################
# NFFT Chi^2 Data functions
##################################################################################################


def chisqdata_vis_nfft(Obsdata, Prior, pol='I', **kwargs):
    """Return the visibilities, sigmas, uv points, and nfft info
    """
    if (Prior.xdim % 2 or Prior.ydim % 2):
        raise Exception("NFFT doesn't work with odd image dimensions!")

    # unpack keyword args
    systematic_noise = kwargs.get('systematic_noise', 0.)
    snrcut = kwargs.get('snrcut', 0.)
    debias = kwargs.get('debias', False)
    weighting = kwargs.get('weighting', 'natural')
    fft_pad_factor = kwargs.get('fft_pad_factor', ehc.FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', ehc.GRIDDER_P_RAD_DEFAULT)
    nfft_eps = kwargs.get('nfft_eps', ehc.NFFT_EPS_DEFAULT)

    # unpack data
    vtype = ehc.vis_poldict[pol]
    atype = ehc.amp_poldict[pol]
    etype = ehc.sig_poldict[pol]
    data_arr = Obsdata.unpack(['t1', 't2', 'u', 'v', vtype, atype, etype], debias=debias)
    (uv, vis, amp, sigma) = apply_systematic_noise_snrcut(data_arr, systematic_noise, snrcut, pol)

    # data weighting
    if weighting == 'uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv, eps=nfft_eps)
    A = [A1]

    return (vis, sigma, A)


def chisqdata_amp_nfft(Obsdata, Prior, pol='I', **kwargs):
    """Return the amplitudes, sigmas, uv points, and nfft info
    """
    if (Prior.xdim % 2 or Prior.ydim % 2):
        raise Exception("NFFT doesn't work with odd image dimensions!")

    # unpack keyword args
    systematic_noise = kwargs.get('systematic_noise', 0.)
    snrcut = kwargs.get('snrcut', 0.)
    debias = kwargs.get('debias', False)
    weighting = kwargs.get('weighting', 'natural')
    fft_pad_factor = kwargs.get('fft_pad_factor', ehc.FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', ehc.GRIDDER_P_RAD_DEFAULT)
    nfft_eps = kwargs.get('nfft_eps', ehc.NFFT_EPS_DEFAULT)

    # unpack data
    vtype = ehc.vis_poldict[pol]
    atype = ehc.amp_poldict[pol]
    etype = ehc.sig_poldict[pol]
    data_arr = Obsdata.unpack(['t1', 't2', 'u', 'v', vtype, atype, etype], debias=debias)

    # apply systematic noise
    (uv, vis, amp, sigma) = apply_systematic_noise_snrcut(data_arr, systematic_noise, snrcut, pol)

    # data weighting
    if weighting == 'uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv, eps=nfft_eps)
    A = [A1]

    return (amp, sigma, A)


def chisqdata_bs_nfft(Obsdata, Prior, pol='I', **kwargs):
    """Return the bispectra, sigmas, uv points, and nfft info
    """
    if (Prior.xdim % 2 or Prior.ydim % 2):
        raise Exception("NFFT doesn't work with odd image dimensions!")

    # unpack keyword args
    # systematic_noise = kwargs.get('systematic_noise',0.)
    maxset = kwargs.get('maxset', False)
    if maxset:
        count = 'max'
    else:
        count = 'min'

    snrcut = kwargs.get('snrcut', 0.)
    weighting = kwargs.get('weighting', 'natural')
    fft_pad_factor = kwargs.get('fft_pad_factor', ehc.FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', ehc.GRIDDER_P_RAD_DEFAULT)
    nfft_eps = kwargs.get('nfft_eps', ehc.NFFT_EPS_DEFAULT)

    # unpack data
    vtype = ehc.vis_poldict[pol]
    biarr = Obsdata.bispectra(mode="all", vtype=vtype, count=count, snrcut=snrcut)

    uv1 = np.hstack((biarr['u1'].reshape(-1, 1), biarr['v1'].reshape(-1, 1)))
    uv2 = np.hstack((biarr['u2'].reshape(-1, 1), biarr['v2'].reshape(-1, 1)))
    uv3 = np.hstack((biarr['u3'].reshape(-1, 1), biarr['v3'].reshape(-1, 1)))
    bi = biarr['bispec']
    sigma = biarr['sigmab']

    # add systematic noise
    # sigma = np.linalg.norm([biarr['sigmab'], systematic_noise*np.abs(biarr['bispec'])], axis=0)

    # data weighting
    if weighting == 'uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv1, eps=nfft_eps)
    A2 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv2, eps=nfft_eps)
    A3 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv3, eps=nfft_eps)
    A = [A1, A2, A3]

    return (bi, sigma, A)


def chisqdata_cphase_nfft(Obsdata, Prior, pol='I', **kwargs):
    """Return the closure phases, sigmas, uv points, and nfft info
    """
    if (Prior.xdim % 2 or Prior.ydim % 2):
        raise Exception("NFFT doesn't work with odd image dimensions!")

    # unpack keyword args
    maxset = kwargs.get('maxset', False)
    uv_min = kwargs.get('cp_uv_min', False)
    if maxset:
        count = 'max'
    else:
        count = 'min'

    snrcut = kwargs.get('snrcut', 0.)
    weighting = kwargs.get('weighting', 'natural')
    systematic_cphase_noise = kwargs.get('systematic_cphase_noise', 0.)
    fft_pad_factor = kwargs.get('fft_pad_factor', ehc.FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', ehc.GRIDDER_P_RAD_DEFAULT)
    nfft_eps = kwargs.get('nfft_eps', ehc.NFFT_EPS_DEFAULT)

    # unpack data
    vtype = ehc.vis_poldict[pol]
    clphasearr = Obsdata.c_phases(mode="all", vtype=vtype,
                                  count=count, uv_min=uv_min, snrcut=snrcut)

    uv1 = np.hstack((clphasearr['u1'].reshape(-1, 1), clphasearr['v1'].reshape(-1, 1)))
    uv2 = np.hstack((clphasearr['u2'].reshape(-1, 1), clphasearr['v2'].reshape(-1, 1)))
    uv3 = np.hstack((clphasearr['u3'].reshape(-1, 1), clphasearr['v3'].reshape(-1, 1)))
    clphase = clphasearr['cphase']
    sigma = clphasearr['sigmacp']

    # add systematic cphase noise (in DEGREES)
    sigma = np.linalg.norm([sigma, systematic_cphase_noise*np.ones(len(sigma))], axis=0)

    # data weighting
    if weighting == 'uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv1, eps=nfft_eps)
    A2 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv2, eps=nfft_eps)
    A3 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv3, eps=nfft_eps)
    A = [A1, A2, A3]

    return (clphase, sigma, A)


def chisqdata_cphase_diag_nfft(Obsdata, Prior, pol='I', **kwargs):
    """Return the diagonalized closure phases, sigmas, uv points, and nfft info
    """
    if (Prior.xdim % 2 or Prior.ydim % 2):
        raise Exception("NFFT doesn't work with odd image dimensions!")

    # unpack keyword args
    maxset = kwargs.get('maxset', False)
    uv_min = kwargs.get('cp_uv_min', False)
    if maxset:
        count = 'max'
    else:
        count = 'min'

    snrcut = kwargs.get('snrcut', 0.)
    fft_pad_factor = kwargs.get('fft_pad_factor', ehc.FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', ehc.GRIDDER_P_RAD_DEFAULT)
    nfft_eps = kwargs.get('nfft_eps', ehc.NFFT_EPS_DEFAULT)

    # unpack data
    vtype = ehc.vis_poldict[pol]
    clphasearr = Obsdata.c_phases_diag(vtype=vtype, count=count, snrcut=snrcut, uv_min=uv_min)

    # loop over timestamps
    clphase_diag = []
    sigma_diag = []
    tform_mats = []
    u1 = []
    v1 = []
    u2 = []
    v2 = []
    u3 = []
    v3 = []
    for ic, cl in enumerate(clphasearr):

        # get diagonalized closure phases and errors
        clphase_diag.append(cl[0]['cphase'])
        sigma_diag.append(cl[0]['sigmacp'])

        # get u and v values
        u1.append(cl[2][:, 0].astype('float'))
        v1.append(cl[3][:, 0].astype('float'))
        u2.append(cl[2][:, 1].astype('float'))
        v2.append(cl[3][:, 1].astype('float'))
        u3.append(cl[2][:, 2].astype('float'))
        v3.append(cl[3][:, 2].astype('float'))

        # get transformation matrix for this timestamp
        tform_mats.append(cl[4].astype('float'))

    # fix formatting of arrays
    u1 = np.concatenate(u1)
    v1 = np.concatenate(v1)
    u2 = np.concatenate(u2)
    v2 = np.concatenate(v2)
    u3 = np.concatenate(u3)
    v3 = np.concatenate(v3)

    # get uv arrays
    uv1 = np.hstack((u1.reshape(-1, 1), v1.reshape(-1, 1)))
    uv2 = np.hstack((u2.reshape(-1, 1), v2.reshape(-1, 1)))
    uv3 = np.hstack((u3.reshape(-1, 1), v3.reshape(-1, 1)))

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv1, eps=nfft_eps)
    A2 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv2, eps=nfft_eps)
    A3 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv3, eps=nfft_eps)
    A = [A1, A2, A3]

    # Per-timestamp transform matrices have varying shapes; NumPy >= 1.24
    # requires explicit dtype=object for inhomogeneous arrays.
    Amatrices = (A, np.array(tform_mats, dtype=object))

    return (np.array(clphase_diag, dtype=object), np.array(sigma_diag, dtype=object), Amatrices)


def chisqdata_camp_nfft(Obsdata, Prior, pol='I', **kwargs):
    """Return the closure amplitudes, sigmas, uv points, and nfft info
    """
    if (Prior.xdim % 2 or Prior.ydim % 2):
        raise Exception("NFFT doesn't work with odd image dimensions!")

    # unpack keyword args
    maxset = kwargs.get('maxset', False)
    if maxset:
        count = 'max'
    else:
        count = 'min'

    snrcut = kwargs.get('snrcut', 0.)
    debias = kwargs.get('debias', False)
    weighting = kwargs.get('weighting', 'natural')
    fft_pad_factor = kwargs.get('fft_pad_factor', ehc.FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', ehc.GRIDDER_P_RAD_DEFAULT)
    nfft_eps = kwargs.get('nfft_eps', ehc.NFFT_EPS_DEFAULT)

    # unpack data
    vtype = ehc.vis_poldict[pol]
    clamparr = Obsdata.c_amplitudes(mode='all', count=count,
                                    vtype=vtype, ctype='camp', debias=debias, snrcut=snrcut)

    uv1 = np.hstack((clamparr['u1'].reshape(-1, 1), clamparr['v1'].reshape(-1, 1)))
    uv2 = np.hstack((clamparr['u2'].reshape(-1, 1), clamparr['v2'].reshape(-1, 1)))
    uv3 = np.hstack((clamparr['u3'].reshape(-1, 1), clamparr['v3'].reshape(-1, 1)))
    uv4 = np.hstack((clamparr['u4'].reshape(-1, 1), clamparr['v4'].reshape(-1, 1)))
    clamp = clamparr['camp']
    sigma = clamparr['sigmaca']

    # data weighting
    if weighting == 'uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv1, eps=nfft_eps)
    A2 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv2, eps=nfft_eps)
    A3 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv3, eps=nfft_eps)
    A4 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv4, eps=nfft_eps)
    A = [A1, A2, A3, A4]

    return (clamp, sigma, A)


def chisqdata_logcamp_nfft(Obsdata, Prior, pol='I', **kwargs):
    """Return the log closure amplitudes, sigmas, uv points, and nfft info
    """
    if (Prior.xdim % 2 or Prior.ydim % 2):
        raise Exception("NFFT doesn't work with odd image dimensions!")

    # unpack keyword args
    maxset = kwargs.get('maxset', False)
    if maxset:
        count = 'max'
    else:
        count = 'min'

    snrcut = kwargs.get('snrcut', 0.)
    debias = kwargs.get('debias', False)
    weighting = kwargs.get('weighting', 'natural')
    fft_pad_factor = kwargs.get('fft_pad_factor', ehc.FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', ehc.GRIDDER_P_RAD_DEFAULT)
    nfft_eps = kwargs.get('nfft_eps', ehc.NFFT_EPS_DEFAULT)

    # unpack data
    vtype = ehc.vis_poldict[pol]
    clamparr = Obsdata.c_amplitudes(mode='all', count=count,
                                    vtype=vtype, ctype='logcamp', debias=debias, snrcut=snrcut)

    uv1 = np.hstack((clamparr['u1'].reshape(-1, 1), clamparr['v1'].reshape(-1, 1)))
    uv2 = np.hstack((clamparr['u2'].reshape(-1, 1), clamparr['v2'].reshape(-1, 1)))
    uv3 = np.hstack((clamparr['u3'].reshape(-1, 1), clamparr['v3'].reshape(-1, 1)))
    uv4 = np.hstack((clamparr['u4'].reshape(-1, 1), clamparr['v4'].reshape(-1, 1)))
    clamp = clamparr['camp']
    sigma = clamparr['sigmaca']

    # data weighting
    if weighting == 'uniform':
        sigma = np.median(sigma) * np.ones(len(sigma))

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv1, eps=nfft_eps)
    A2 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv2, eps=nfft_eps)
    A3 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv3, eps=nfft_eps)
    A4 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv4, eps=nfft_eps)
    A = [A1, A2, A3, A4]

    return (clamp, sigma, A)


def chisqdata_logcamp_diag_nfft(Obsdata, Prior, pol='I', **kwargs):
    """Return the diagonalized log closure amplitudes, sigmas, uv points, and nfft info
    """
    if (Prior.xdim % 2 or Prior.ydim % 2):
        raise Exception("NFFT doesn't work with odd image dimensions!")

    # unpack keyword args
    maxset = kwargs.get('maxset', False)
    if maxset:
        count = 'max'
    else:
        count = 'min'

    snrcut = kwargs.get('snrcut', 0.)
    debias = kwargs.get('debias', False)
    fft_pad_factor = kwargs.get('fft_pad_factor', ehc.FFT_PAD_DEFAULT)
    p_rad = kwargs.get('p_rad', ehc.GRIDDER_P_RAD_DEFAULT)
    nfft_eps = kwargs.get('nfft_eps', ehc.NFFT_EPS_DEFAULT)

    # unpack data & mask low snr points
    vtype = ehc.vis_poldict[pol]
    clamparr = Obsdata.c_log_amplitudes_diag(vtype=vtype, count=count, debias=debias, snrcut=snrcut)

    # loop over timestamps
    clamp_diag = []
    sigma_diag = []
    tform_mats = []
    u1 = []
    v1 = []
    u2 = []
    v2 = []
    u3 = []
    v3 = []
    u4 = []
    v4 = []
    for ic, cl in enumerate(clamparr):

        # get diagonalized log closure amplitudes and errors
        clamp_diag.append(cl[0]['camp'])
        sigma_diag.append(cl[0]['sigmaca'])

        # get u and v values
        u1.append(cl[2][:, 0].astype('float'))
        v1.append(cl[3][:, 0].astype('float'))
        u2.append(cl[2][:, 1].astype('float'))
        v2.append(cl[3][:, 1].astype('float'))
        u3.append(cl[2][:, 2].astype('float'))
        v3.append(cl[3][:, 2].astype('float'))
        u4.append(cl[2][:, 3].astype('float'))
        v4.append(cl[3][:, 3].astype('float'))

        # get transformation matrix for this timestamp
        tform_mats.append(cl[4].astype('float'))

    # fix formatting of arrays
    u1 = np.concatenate(u1)
    v1 = np.concatenate(v1)
    u2 = np.concatenate(u2)
    v2 = np.concatenate(v2)
    u3 = np.concatenate(u3)
    v3 = np.concatenate(v3)
    u4 = np.concatenate(u4)
    v4 = np.concatenate(v4)

    # get uv arrays
    uv1 = np.hstack((u1.reshape(-1, 1), v1.reshape(-1, 1)))
    uv2 = np.hstack((u2.reshape(-1, 1), v2.reshape(-1, 1)))
    uv3 = np.hstack((u3.reshape(-1, 1), v3.reshape(-1, 1)))
    uv4 = np.hstack((u4.reshape(-1, 1), v4.reshape(-1, 1)))

    # get NFFT info
    npad = int(fft_pad_factor * np.max((Prior.xdim, Prior.ydim)))
    A1 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv1, eps=nfft_eps)
    A2 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv2, eps=nfft_eps)
    A3 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv3, eps=nfft_eps)
    A4 = obsh.NFFTInfo(Prior.xdim, Prior.ydim, Prior.psize, Prior.pulse, npad, p_rad, uv4, eps=nfft_eps)
    A = [A1, A2, A3, A4]

    # Per-timestamp transform matrices have varying shapes; NumPy >= 1.24
    # requires explicit dtype=object for inhomogeneous arrays.
    Amatrices = (A, np.array(tform_mats, dtype=object))

    return (np.array(clamp_diag, dtype=object), np.array(sigma_diag, dtype=object), Amatrices)

##################################################################################################
# Plotting Functions
##################################################################################################


def plot_i(im, Prior, nit, chi2_dict, **kwargs):
    """Plot the total intensity image at each iteration
    """
    cmap = kwargs.get('cmap', 'afmhot')
    interpolation = kwargs.get('interpolation', 'gaussian')
    pol = kwargs.get('pol', '')
    scale = kwargs.get('scale', None)
    dynamic_range = kwargs.get('dynamic_range', 1.e5)
    gamma = kwargs.get('dynamic_range', .5)

    plt.ion()
    plt.pause(1.e-6)
    plt.clf()

    imarr = im.reshape(Prior.ydim, Prior.xdim)

    if scale == 'log':
        if (imarr < 0.0).any():
            print('clipping values less than 0')
            imarr[imarr < 0.0] = 0.0
        imarr = np.log(imarr + np.max(imarr)/dynamic_range)

    if scale == 'gamma':
        if (imarr < 0.0).any():
            print('clipping values less than 0')
            imarr[imarr < 0.0] = 0.0
        imarr = (imarr + np.max(imarr)/dynamic_range)**(gamma)

    plt.imshow(imarr, cmap=plt.get_cmap(cmap), interpolation=interpolation)
    xticks = obsh.ticks(Prior.xdim, Prior.psize/ehc.RADPERAS/1e-6)
    yticks = obsh.ticks(Prior.ydim, Prior.psize/ehc.RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel(r'Relative RA ($\mu$as)')
    plt.ylabel(r'Relative Dec ($\mu$as)')
    plotstr = str(pol) + f" : step: {nit}  "
    for key in chi2_dict.keys():
        plotstr += rf"$\chi^2_{{{key}}}$: {chi2_dict[key]:0.2f}  "
    plt.title(plotstr, fontsize=18)

##################################################################################################
# Embedding functions
##################################################################################################




# TODO(achael): consolidate `embed` (1D, this function) and `embed_imarr`
# (1D or 2D, below) into a single implementation -- their bodies overlap.
def embed(imvec, mask, clipfloor=0., randomfloor=False):
    """Embeds a 1d image vector into the size of boolean embed mask
    """

    out = np.zeros(len(mask))

    # Here's a much faster version than before
    out[mask.nonzero()] = imvec

    #if clipfloor != 0.0:
    if randomfloor:  # prevent total variation gradient singularities
        out[(mask-1).nonzero()] = clipfloor * \
            np.abs(np.random.normal(size=len((mask-1).nonzero()[0])))
    else:
        out[(mask-1).nonzero()] = clipfloor

    return out


def embed_imarr(imarr, mask, clipfloor=0., randomfloor=False):
    """Embed a packed image array back onto the full image grid.

    Multi-row generalization of `embed`: each row of `imarr` is independently
    embedded back into a full-grid representation using `mask`. Pixels outside
    the mask are filled with `clipfloor` (constant) or `clipfloor * |N(0, 1)|`
    when `randomfloor=True`.

    Lives in imager_utils alongside `embed` rather than in imager_backend.py
    to avoid a module-load cycle (pol_imager_utils -> imager_backend imports).

    Parameters
    ----------
    imarr : np.ndarray
        Either 1D of length `sum(mask)` (Stokes-I only), or 2D of shape
        (nsolve, sum(mask)) where nsolve is the number of Stokes / spectral
        slots being solved for (1 for Stokes-I only, 4 for full polarization,
        3 or 10 for the multifrequency variants).
    mask : np.ndarray of bool
        Embed mask of length npix_total. Number of True entries must equal
        the second axis of `imarr` (or its length, for 1D input).
    clipfloor : float, optional
        Value placed at non-mask pixels when `randomfloor=False`. Default 0.0.
    randomfloor : bool, optional
        If True, non-mask pixels get `clipfloor * |N(0, 1)|` instead of a
        constant. Used to break gradient singularities in total-variation
        regularizers. Default False.

    Returns
    -------
    out : np.ndarray
        Same dimensionality as `imarr` (1D or 2D), but the second axis
        (or only axis) is extended to len(mask).

    Raises
    ------
    Exception
        If `imarr` is not 1D or 2D, or if its mask-axis length does not
        equal `sum(mask)`.
    """
    imarrdim = len(imarr.shape)
    if imarrdim == 2:
        nsolve = imarr.shape[0]
        nimage = imarr.shape[1]
    elif imarrdim == 1:
        nsolve = 1
        nimage = imarr.shape[0]
        imarr = imarr.reshape((nsolve, nimage))
    else:
        raise Exception("in embed_imarr, imarr should have one or two dimensions!")

    if nimage != np.sum(mask):
        raise Exception("in embed_imarr, number of masked pixels is not consistent with imarr shape!")

    nimage_out = len(mask)
    outarr = np.empty((nsolve, nimage_out))
    # Vectorized over the nsolve axis: scatter imarr into the masked columns
    # of outarr, then fill non-mask columns with clipfloor (or random).
    not_mask = ~mask.astype(bool)
    outarr[:, mask.astype(bool)] = imarr
    if randomfloor:
        outarr[:, not_mask] = clipfloor * np.abs(
            np.random.normal(size=(nsolve, int(not_mask.sum())))
        )
    else:
        outarr[:, not_mask] = clipfloor

    if imarrdim == 1:
        outarr = outarr[0]

    return outarr





