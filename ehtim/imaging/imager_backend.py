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

import numpy as np

import ehtim.imaging.imager_utils as imutils
import ehtim.imaging.multifreq_imager_utils as mfutils
import ehtim.imaging.pol_imager_utils as polutils


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


def compute_chisq_dict(imcur, dat_term_keys, data_tuples, obslist_next,
                       logfreqratio_list, mf_next, pol_next, ttype, embed_mask,
                       dataterms, dataterms_pol, polarization_modes):
    """Compute chi^2 value for each data term across all observations.

    Parameters
    ----------
    imcur : np.ndarray
        Current image array transformed to bounded values.
    dat_term_keys : list of str
        Data term names to evaluate, already sorted.
    data_tuples : dict
        Pre-computed data products keyed by dname or dname_i,
        each value is a (data, sigma, A) tuple.
    obslist_next : list
        List of Obsdata objects (one per frequency/epoch).
    logfreqratio_list : list of float
        Log frequency ratios log(nu_i/reffreq); one per obs.
    mf_next : bool
        Whether multifrequency imaging is enabled.
    pol_next : str
        Polarization mode string.
    ttype : str
        Transform type ('direct', 'fast', 'nfft').
    embed_mask : np.ndarray of bool
        Pixel embedding mask.
    dataterms : list of str
        Single-polarization data term names.
    dataterms_pol : list of str
        Polarimetric data term names.
    polarization_modes : list of str
        Polarization modes that bundle Stokes I with other terms.

    Returns
    -------
    chi2_dict : dict
        Mapping from dname (or dname_i for multi-obs) to chi^2 scalar.
    """
    chi2_dict = {}
    for dname in dat_term_keys:
        # Loop over all observations in the list
        for i, obs in enumerate(obslist_next):
            if len(obslist_next) == 1:
                dname_key = dname
            else:
                dname_key = dname + (f'_{i}')

            # get data products
            (data, sigma, A) = data_tuples[dname_key]

            # get current multifrequency image
            if mf_next:
                logfreqratio = logfreqratio_list[i]
                imcur_nu = mfutils.image_at_freq(imcur, logfreqratio)
            else:
                imcur_nu = imcur

            # Polarization chi^2 terms
            if dname in dataterms_pol:
                chi2 = polutils.polchisq(imcur_nu, A, data, sigma, dname,
                                         ttype=ttype, mask=embed_mask)

            # Single Polarization chi^2 terms
            elif dname in dataterms:
                if pol_next in polarization_modes:
                    imcur_nu_I = imcur_nu[0]
                else:
                    imcur_nu_I = imcur_nu
                chi2 = imutils.chisq(imcur_nu_I, A, data, sigma, dname,
                                     ttype=ttype, mask=embed_mask)

            else:
                raise Exception(f"data term {dname} not recognized!")

            chi2_dict[dname_key] = chi2

    return chi2_dict


def compute_chisqgrad_dict(imcur, dat_term_keys, data_tuples, obslist_next,
                           logfreqratio_list, mf_next, pol_next, ttype, embed_mask,
                           which_solve, nimage,
                           dataterms, dataterms_pol, polarization_modes):
    """Compute chi^2 gradient for each data term across all observations.

    Parameters
    ----------
    imcur : np.ndarray
        Current image array transformed to bounded values.
    dat_term_keys : list of str
        Data term names to evaluate, already sorted.
    data_tuples : dict
        Pre-computed data products keyed by dname or dname_i,
        each value is a (data, sigma, A) tuple.
    obslist_next : list
        List of Obsdata objects (one per frequency/epoch).
    logfreqratio_list : list of float
        Log frequency ratios log(nu_i/reffreq); one per obs.
    mf_next : bool
        Whether multifrequency imaging is enabled.
    pol_next : str
        Polarization mode string.
    ttype : str
        Transform type ('direct', 'fast', 'nfft').
    embed_mask : np.ndarray of bool
        Pixel embedding mask.
    which_solve : np.ndarray of int
        Binary flags for which parameters are solved.
    nimage : int
        Number of active pixels (sum of embed_mask).
    dataterms : list of str
        Single-polarization data term names.
    dataterms_pol : list of str
        Polarimetric data term names.
    polarization_modes : list of str
        Polarization modes that bundle Stokes I with other terms.

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
        # Loop over all observations in the list
        for i, obs in enumerate(obslist_next):
            if len(obslist_next) == 1:
                dname_key = dname
            else:
                dname_key = dname + (f'_{i}')

            # get data products
            (data, sigma, A) = data_tuples[dname_key]

            # get current multifrequency image
            if mf_next:
                logfreqratio = logfreqratio_list[i]
                imcur_nu = mfutils.image_at_freq(imcur, logfreqratio)
            else:
                imcur_nu = imcur

            # Polarimetric chi^2 gradients
            if dname in dataterms_pol:
                if mf_next:
                    pol_solve = which_solve[0:4]
                else:
                    pol_solve = which_solve
                chi2grad = polutils.polchisqgrad(imcur_nu, A, data, sigma, dname,
                                                 ttype=ttype, mask=embed_mask,
                                                 pol_solve=pol_solve)

            # Single polarization chi^2 gradients
            elif dname in dataterms:
                if pol_next in polarization_modes:  # polarization
                    imcur_nu_I = imcur_nu[0]
                else:
                    imcur_nu_I = imcur_nu

                chi2grad = imutils.chisqgrad(imcur_nu_I, A, data, sigma, dname,
                                             ttype=ttype, mask=embed_mask)

                # If imaging Stokes I with polarization simultaneously, bundle the gradient
                if pol_next in polarization_modes:
                    chi2grad = np.array((chi2grad, zero_row, zero_row, zero_row))

            else:
                raise Exception(f"data term {dname} not recognized!")

            # If multifrequency imaging,
            # transform the image gradients for all the solved quantities
            if mf_next:
                logfreqratio = logfreqratio_list[i]
                chi2grad = mfutils.mf_all_grads_chain(chi2grad, imcur_nu, imcur, logfreqratio)

            chi2grad_dict[dname_key] = np.array(chi2grad)

    return chi2grad_dict
