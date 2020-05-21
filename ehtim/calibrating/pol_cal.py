# pol_cal.py
# functions for polarimetric-calibration
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

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time

import ehtim.imaging.imager_utils as iu
import ehtim.observing.obs_simulate as simobs
import ehtim.const_def as ehc

MAXIT = 1000  # maximum number of iterations in pol-cal minimizer

###################################################################################################
# Polarimetric Calibration
###################################################################################################


def leakage_cal(obs, im=None, sites=[], leakage_tol=.1, pol_fit=['RL', 'LR'], dtype='vis',
                const_fpol=False, inverse=False,  minimizer_method='L-BFGS-B',
                ttype='direct', fft_pad_factor=2, show_solution=True, obs_apply=False):
    """Polarimetric calibration (detects and removes polarimetric leakage,
       based on consistency with a given image)

       Args:
           obs (Obsdata): The observation to be calibrated
           im (Image): the reference image used for calibration
                      (not needed if using const_fpol = True)
           sites (list): list of sites to include in the polarimetric calibration.
                         empty list calibrates all sites

           leakage_tol (float): leakage values exceeding this value will be disfavored by the prior
           pol_fit (list): list of visibilities to use; e.g., ['RL','LR'] or ['RR','LL','RL','LR']
           dtype (str): Type of data to fit ('vis' for complex visibilities;
                                             'amp' for just the amplitudes)
           const_fpol (bool): If true, solve for a single fractional polarization
                              across all baselines in addition to leakage.
                              For this option, the passed image is not used.
           minimizer_method (str): Method for scipy.optimize.minimize (e.g., 'CG', 'BFGS')
           ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT
           fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in FFT

           show_solution (bool): if True, display the solution as it is calculated

       Returns:
           (Obsdata): the calibrated observation, with computed leakage values added to the obs.tarr
    """
    tstart = time.time()

    mask = []

    # Do everything in a circular basis
    if not const_fpol:
        im_circ = im.switch_polrep('circ')
    else:
        im_circ = None

    if dtype not in ['vis', 'amp']:
        raise Exception('dtype must be vis or amp')

    # Create the obsdata object for searching
    obs_test = obs.copy()
    obs_test = obs_test.switch_polrep('circ')

    # Check to see if the field rotation is corrected
    if obs_test.frcal is False:
        print("Field rotation angles have not been corrected. Correcting now...")
        obs_test.data = simobs.apply_jones_inverse(obs_test, frcal=False, dcal=True, verbose=False)
        obs_test.frcal = True

    # List of all sites present in the observation
    allsites = list(set(np.hstack((obs.data['t1'], obs.data['t2']))))

    if len(sites) == 0:
        print("No stations specified for leakage calibration: defaulting to calibrating all !")
        sites = allsites

    # Set all leakage terms in obs_test to zero
    # (we will only correct leakage for those sites with new solutions)
    for j in range(len(obs_test.tarr)):
        if obs_test.tarr[j]['site'] in sites:
            continue
        obs_test.tarr[j]['dr'] = obs_test.tarr[j]['dl'] = 0.0j

    # only include sites that are present
    sites = [s for s in sites if s in allsites]
    site_index = [list(obs.tarr['site']).index(s) for s in sites]

    if not const_fpol:
        (dataRR, sigmaRR, ARR) = iu.chisqdata(obs, im_circ, mask=mask, dtype=dtype, pol='RR',
                                              ttype=ttype, fft_pad_factor=fft_pad_factor)
        (dataLL, sigmaLL, ALL) = iu.chisqdata(obs, im_circ, mask=mask, dtype=dtype, pol='LL',
                                              ttype=ttype, fft_pad_factor=fft_pad_factor)
        (dataRL, sigmaRL, ARL) = iu.chisqdata(obs, im_circ, mask=mask, dtype=dtype, pol='RL',
                                              ttype=ttype, fft_pad_factor=fft_pad_factor)
        (dataLR, sigmaLR, ALR) = iu.chisqdata(obs, im_circ, mask=mask, dtype=dtype, pol='LR',
                                              ttype=ttype, fft_pad_factor=fft_pad_factor)

    # If inverse modeling, pre-compute field rotation angles
    if inverse:
        el1 = obs.unpack(['el1'], ang_unit='rad')['el1']
        el2 = obs.unpack(['el2'], ang_unit='rad')['el2']
        par1 = obs.unpack(['par_ang1'], ang_unit='rad')['par_ang1']
        par2 = obs.unpack(['par_ang2'], ang_unit='rad')['par_ang2']

        fr_elev1 = np.array([obs.tarr[obs.tkey[o['t1']]]['fr_elev'] for o in obs.data])
        fr_elev2 = np.array([obs.tarr[obs.tkey[o['t2']]]['fr_elev'] for o in obs.data])
        fr_par1 = np.array([obs.tarr[obs.tkey[o['t1']]]['fr_par'] for o in obs.data])
        fr_par2 = np.array([obs.tarr[obs.tkey[o['t2']]]['fr_par'] for o in obs.data])
        fr_off1 = np.array([obs.tarr[obs.tkey[o['t1']]]['fr_off'] for o in obs.data])
        fr_off2 = np.array([obs.tarr[obs.tkey[o['t2']]]['fr_off'] for o in obs.data])

        fr1 = fr_elev1*el1 + fr_par1*par1 + fr_off1*np.pi/180.
        fr2 = fr_elev2*el2 + fr_par2*par2 + fr_off2*np.pi/180.

    def chisq_total(data, im, D, inverse=False):
        if const_fpol:
            if inverse:
                # Note: These are linearized approximations (in leakage and source polarization)
                fpol_model = D[-2]
                cpol_model = np.real(D[-1])

                D1R = [D[2*sites.index(o['t1'])] for o in data]
                D1L = [D[2*sites.index(o['t1'])+1] for o in data]
                D2R = [D[2*sites.index(o['t2'])] for o in data]
                D2L = [D[2*sites.index(o['t2'])+1] for o in data]

                lrll = data['lrvis']/data['llvis']
                lrrr = data['lrvis']/data['rrvis']
                rlll = data['rlvis']/data['llvis']
                rlrr = data['rlvis']/data['rrvis']

                lrll_sigma = data['lrsigma']/np.abs(data['llvis'])
                lrrr_sigma = data['lrsigma']/np.abs(data['rrvis'])
                rlll_sigma = data['rlsigma']/np.abs(data['llvis'])
                rlrr_sigma = data['rlsigma']/np.abs(data['rrvis'])

                lrll_model = (np.conjugate(fpol_model) * (1.0 + cpol_model) +
                              D1L * np.exp(-2j*fr1) * (1.0 + 2.0*cpol_model) +
                              np.conjugate(D2R) * np.exp(-2j*fr2))
                lrrr_model = (np.conjugate(fpol_model) * (1.0 - cpol_model) +
                              D1L * np.exp(-2j*fr1) +
                              np.conjugate(D2R) * np.exp(-2j*fr2) * (1.0 - 2.0*cpol_model))
                rlll_model = (fpol_model * (1.0 + cpol_model) +
                              D1R * np.exp(2j*fr1) +
                              np.conjugate(D2L) * np.exp(2j*fr2) * (1.0 + 2.0*cpol_model))
                rlrr_model = (fpol_model * (1.0 - cpol_model) +
                              D1R * np.exp(2j*fr1) * (1.0 - 2.0*cpol_model) +
                              np.conjugate(D2L) * np.exp(2j*fr2))

                chisq = np.concatenate([np.abs((lrll - lrll_model)/lrll_sigma)**2,
                                        np.abs((lrrr - lrrr_model)/lrrr_sigma)**2,
                                        np.abs((rlll - rlll_model)/rlll_sigma)**2,
                                        np.abs((rlrr - rlrr_model)/rlrr_sigma)**2])
                chisq = chisq[~np.isnan(chisq)]
                return np.mean(chisq)
            else:
                fpol_model = D[-2]
                cpol_model = np.real(D[-1])

                fpol_data_1 = 2.0 * data['rlvis']/(data['rrvis'] + data['llvis'])
                fpol_data_2 = 2.0 * np.conj(data['lrvis']/(data['rrvis'] + data['llvis']))
                fpol_sigma_1 = 2.0/np.abs(data['rrvis'] + data['llvis']) * data['rlsigma']
                fpol_sigma_2 = 2.0/np.abs(data['rrvis'] + data['llvis']) * data['lrsigma']
                return 0.5*np.mean(np.abs((fpol_model - fpol_data_1)/fpol_sigma_1)**2
                                   + np.abs((fpol_model - fpol_data_2)/fpol_sigma_2)**2)
        else:
            chisq_RR = chisq_LL = chisq_RL = chisq_LR = 0.0
            if 'RR' in pol_fit:
                chisq_RR = iu.chisq(im.rrvec, ARR,
                                    obs_test.unpack_dat(data, ['rr' + dtype])['rr' + dtype],
                                    data['rrsigma'], dtype=dtype, ttype=ttype, mask=mask)
            if 'LL' in pol_fit:
                chisq_LL = iu.chisq(im.llvec, ALL,
                                    obs_test.unpack_dat(data, ['ll' + dtype])['ll' + dtype],
                                    data['llsigma'], dtype=dtype, ttype=ttype, mask=mask)
            if 'RL' in pol_fit:
                chisq_RL = iu.chisq(im.rlvec, ARL,
                                    obs_test.unpack_dat(data, ['rl' + dtype])['rl' + dtype],
                                    data['rlsigma'], dtype=dtype, ttype=ttype, mask=mask)
            if 'LR' in pol_fit:
                chisq_LR = iu.chisq(im.lrvec, ALR,
                                    obs_test.unpack_dat(data, ['lr' + dtype])['lr' + dtype],
                                    data['lrsigma'], dtype=dtype, ttype=ttype, mask=mask)
            return (chisq_RR + chisq_LL + chisq_RL + chisq_LR)/len(pol_fit)

    print("Finding leakage for sites:", sites)

    def errfunc(Dpar):
        # all the D-terms (complex). If const_fpol, fpol is the last parameter.
        D = Dpar.astype(np.float64).view(dtype=np.complex128)

        if not inverse:
            for isite in range(len(sites)):
                obs_test.tarr['dr'][site_index[isite]] = D[2*isite]
                obs_test.tarr['dl'][site_index[isite]] = D[2*isite+1]
            data = simobs.apply_jones_inverse(obs_test, dcal=False, verbose=False)
        else:
            data = obs.data

        # goodness-of-fit for the leakage-corrected data
        chisq = chisq_total(data, im_circ, D, inverse=inverse)

        # prior on the D terms
        chisq_D = np.sum(np.abs(D/leakage_tol)**2)

        return chisq + chisq_D

    # Now, we will minimize the total chi-squared. We need two complex leakage terms for each site
    optdict = {'maxiter': MAXIT}  # minimizer params
    Dpar_guess = np.zeros((len(sites) + const_fpol*2)*2, dtype=np.complex128).view(dtype=np.float64)
    print("Minimizing...")
    res = opt.minimize(errfunc, Dpar_guess, method=minimizer_method, options=optdict)

    # get solution
    D_fit = res.x.astype(np.float64).view(dtype=np.complex128)  # all the D-terms (complex)

    # Apply the solution
    for isite in range(len(sites)):
        obs_test.tarr['dr'][site_index[isite]] = D_fit[2*isite]
        obs_test.tarr['dl'][site_index[isite]] = D_fit[2*isite+1]
    obs_test.data = simobs.apply_jones_inverse(obs_test, dcal=False, verbose=False)
    obs_test.dcal = True

    # Re-populate any additional leakage terms that were present
    for j in range(len(obs_test.tarr)):
        if obs_test.tarr[j]['site'] in sites:
            continue
        obs_test.tarr[j]['dr'] = obs.tarr[j]['dr']
        obs_test.tarr[j]['dl'] = obs.tarr[j]['dl']

    if show_solution:
        if inverse is False:
            chisq_orig = chisq_total(obs.switch_polrep('circ').data,
                                     im_circ, D_fit, inverse=inverse)
            chisq_new = chisq_total(obs_test.data, im_circ, D_fit, inverse=inverse)
        else:
            chisq_orig = chisq_total(obs.switch_polrep('circ').data,
                                     im_circ, D_fit*0, inverse=inverse)
            chisq_new = chisq_total(obs.switch_polrep('circ').data, im_circ, D_fit, inverse=inverse)

        print("Original chi-squared: {:.4f}".format(chisq_orig))
        print("New chi-squared: {:.4f}\n".format(chisq_new))
        for isite in range(len(sites)):
            print(sites[isite])
            print('   D_R: {:.4f}'.format(D_fit[2*isite]))
            print('   D_L: {:.4f}\n'.format(D_fit[2*isite+1]))
        if const_fpol:
            print('Source Fractional Polarization Magnitude: {:.4f}'.format(np.abs(D_fit[-2])))
            print('Source Fractional Polarization EVPA [deg]: {:.4f}\n'.format(
                90./np.pi*np.angle(D_fit[-2])))
            if inverse:
                print('Source Fractional Circular Polarization: {:.4f}'.format(np.real(D_fit[-1])))

    tstop = time.time()
    print("\nleakage_cal time: %f s" % (tstop - tstart))

    if obs_apply is not False:
        # Apply the solution to another observation
        obs_test = obs_apply.copy()
        obs_test.tarr['dr'] *= 0.0
        obs_test.tarr['dl'] *= 0.0

        # Copy the solved D-terms
        for isite in range(len(sites)):
            if sites[isite] in list(obs_test.tarr['site']):
                i_site = list(obs_test.tarr['site']).index(sites[isite])
                obs_test.tarr['dr'][i_site] = D_fit[2*isite]
                obs_test.tarr['dl'][i_site] = D_fit[2*isite+1]

        obs_test.data = simobs.apply_jones_inverse(obs_test, dcal=False, verbose=False)
        obs_test.dcal = True

        # Copy in the remaining D-terms that were there before
        for j in range(len(obs_test.tarr)):
            if obs_test.tarr[j]['site'] in sites:
                continue
            obs_test.tarr[j]['dr'] = obs_apply.tarr[j]['dr']
            obs_test.tarr[j]['dl'] = obs_apply.tarr[j]['dl']
    else:
        obs_test = obs_test.switch_polrep(obs.polrep)

    if not const_fpol:
        return obs_test
    else:
        if inverse:
            return [obs_test, D_fit[-2], D_fit[-1]]
        else:
            return [obs_test, D_fit[-2]]


def plot_leakage(obs, sites=[], axis=False, rangex=False, rangey=False,
                 markers=['o', 's'], markersize=6,
                 export_pdf="", axislabels=True, legend=True, sort_tarr=True, show=True):
    """Plot polarimetric leakage terms in an observation

       Args:
           obs (Obsdata): observation (or Array) containing the tarr
           axis (matplotlib.axes.Axes): add plot to this axis
           rangex (list): [xmin, xmax] x-axis limits
           rangey (list): [ymin, ymax] y-axis limits
           markers (str): pair of matplotlib plot markers (for RCP and LCP)
           markersize (int): size of plot markers
           label (str): plot legend label

           export_pdf (str): path to pdf file to save figure
           axislabels (bool): Show axis labels if True
           legend (bool): Show legend if True
           show (bool): Display the plot if true

       Returns:
           (matplotlib.axes.Axes): Axes object with data plot
    """

    tarr = obs.tarr.copy()
    if sort_tarr:
        tarr.sort(axis=0)

    if len(sites):
        mask = [t in sites for t in tarr['site']]
        tarr = tarr[mask]

    clist = ehc.SCOLORS

    # make plot(s)
    if axis:
        fig = axis.figure
        x = axis
    else:
        fig = plt.figure()
        x = fig.add_subplot(1, 1, 1)

    plt.axhline(0, color='black')
    plt.axvline(0, color='black')

    xymax = np.max([np.abs(tarr['dr']), np.abs(tarr['dl'])])*100.0

    plot_points = []
    for i in range(len(tarr)):
        color = clist[i % len(clist)]
        label = tarr['site'][i]
        dre, = x.plot(np.real(tarr['dr'][i])*100.0, np.imag(tarr['dr'][i])*100.0, markers[0],
                      markersize=markersize, color=color, label=label)
        dim, = x.plot(np.real(tarr['dl'][i])*100.0, np.imag(tarr['dl'][i])*100.0, markers[1],
                      markersize=markersize, color=color, label=label)
        plot_points.append([dre, dim])

    # Data ranges
    if not rangex:
        rangex = [-xymax*1.1-0.01, xymax*1.1+0.01]

    if not rangey:
        rangey = [-xymax*1.1-0.01, xymax*1.1+0.01]

#    if not rangex and not rangey:
#        plt.axes().set_aspect('equal', 'datalim')

    x.set_xlim(rangex)
    x.set_ylim(rangey)

    # label and save
    if axislabels:
        x.set_xlabel('Re[$D$] (\%)')
        x.set_ylabel('Im[$D$] (\%)')
    if legend:
        legend1 = plt.legend([l[0] for l in plot_points], tarr['site'], ncol=1, loc=1)
        plt.legend(plot_points[0], ['$D_R$ (\%)', '$D_L$ (\%)'], loc=4)
        plt.gca().add_artist(legend1)
    if export_pdf != "":  # and not axis:
        fig.savefig(export_pdf, bbox_inches='tight')
    if show:
        plt.show(block=False)

    return x
