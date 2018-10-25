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

import numpy as np
import ehtim.imaging.imager_utils as iu
import ehtim.observing.obs_simulate as simobs
import scipy.optimize as opt
import time

MAXIT=50
###################################################################################################################################
#Polarimetric Calibration
###################################################################################################################################

def leakage_cal(obs, im, sites=[], leakage_tol=.1, pol_fit = ['RL','LR'], dtype='vis', minimizer_method='L-BFGS-B',
             ttype='direct', fft_pad_factor=2, show_solution=True, obs_apply=False):

    """Polarimetric calibration (detects and removes polarimetric leakage, based on consistency with a given image)

       Args:
           obs (Obsdata): The observation to be calibrated
           im (Image): the reference image used for calibration
           sites (list): list of sites to include in the polarimetric calibration. empty list calibrates all sites

           leakage_tol (float): leakage values that exceed this value will be disfavored by the prior
           pol_fit (list): list of visibilities to use; e.g., ['RL','LR'] or ['RR','LL','RL','LR']
           dtype (str): Type of data to fit ('vis' for complex visibilities; 'amp' for just the amplitudes)
           minimizer_method (str): Method for scipy.optimize.minimize (e.g., 'CG', 'BFGS', 'Nelder-Mead', etc.)
           ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT
           fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in FFT

           show_solution (bool): if True, display the solution as it is calculated

       Returns:
           (Obsdata): the calibrated observation, with computed leakage values added to the obs.tarr
    """
    tstart = time.time()

    mask=[]

    # Do everything in a circular basis
    im_circ = im.switch_polrep('circ')        

    if dtype not in ['vis','amp']:
        raise Exception('dtype must be vis or amp')

    # Create the obsdata object for searching 
    obs_test = obs.copy()
    obs_test = obs_test.switch_polrep('circ')

    # Check to see if the field rotation is corrected
    if obs_test.frcal == False:
        print("Field rotation angles have not been corrected. Correcting now...")
        obs_test.data = simobs.apply_jones_inverse(obs_test,frcal=False,dcal=True,verbose=False)
        obs_test.frcal = True

    # List of all sites present in the observation
    allsites = list(set(np.hstack((obs.data['t1'], obs.data['t2']))))

    if len(sites) == 0:
        print("No stations specified for leakage calibration: defaulting to calibrating all !")
        sites = allsites

    # only include sites that are present
    sites = [s for s in sites if s in allsites]
    site_index = [list(obs.tarr['site']).index(s) for s in sites]

    (dataRR, sigmaRR, ARR) = iu.chisqdata(obs, im_circ, mask=mask, dtype=dtype, pol='RR', ttype=ttype, fft_pad_factor=fft_pad_factor)
    (dataLL, sigmaLL, ALL) = iu.chisqdata(obs, im_circ, mask=mask, dtype=dtype, pol='LL', ttype=ttype, fft_pad_factor=fft_pad_factor)
    (dataRL, sigmaRL, ARL) = iu.chisqdata(obs, im_circ, mask=mask, dtype=dtype, pol='RL', ttype=ttype, fft_pad_factor=fft_pad_factor)
    (dataLR, sigmaLR, ALR) = iu.chisqdata(obs, im_circ, mask=mask, dtype=dtype, pol='LR', ttype=ttype, fft_pad_factor=fft_pad_factor)

    def chisq_total(data, im):
        chisq_RR = chisq_LL = chisq_RL = chisq_LR = 0.0
        if 'RR' in pol_fit: chisq_RR = iu.chisq(im.rrvec, ARR, obs_test.unpack_dat(data,['rr' + dtype])['rr' + dtype], data['rrsigma'], dtype=dtype, ttype=ttype, mask=mask)
        if 'LL' in pol_fit: chisq_LL = iu.chisq(im.llvec, ALL, obs_test.unpack_dat(data,['ll' + dtype])['ll' + dtype], data['llsigma'], dtype=dtype, ttype=ttype, mask=mask)
        if 'RL' in pol_fit: chisq_RL = iu.chisq(im.rlvec, ARL, obs_test.unpack_dat(data,['rl' + dtype])['rl' + dtype], data['rlsigma'], dtype=dtype, ttype=ttype, mask=mask)
        if 'LR' in pol_fit: chisq_LR = iu.chisq(im.lrvec, ALR, obs_test.unpack_dat(data,['lr' + dtype])['lr' + dtype], data['lrsigma'], dtype=dtype, ttype=ttype, mask=mask)
        return (chisq_RR + chisq_LL + chisq_RL + chisq_LR)/len(pol_fit)

    print("Finding leakage for sites:",sites)

    def errfunc(Dpar):
        D = Dpar.astype(np.float64).view(dtype=np.complex128) # all the D-terms (complex)

        for isite in range(len(sites)):
            obs_test.tarr['dr'][site_index[isite]] = D[2*isite]
            obs_test.tarr['dl'][site_index[isite]] = D[2*isite+1]
 
        data = simobs.apply_jones_inverse(obs_test,dcal=False,verbose=False)

        # goodness-of-fit for gains 
        chisq = chisq_total(data, im_circ)

        # prior on the D terms
        chisq_D = np.sum(np.abs(D/leakage_tol)**2)

        return chisq + chisq_D

    # Now, we will minimize the total chi-squared. We need two complex leakage terms for each site
    optdict = {'maxiter' : MAXIT} # minimizer params
    Dpar_guess = np.zeros(len(sites)*2, dtype=np.complex128).view(dtype=np.float64)
    print("Minimizing...")
    res = opt.minimize(errfunc, Dpar_guess, method=minimizer_method, options=optdict)
    
    # get solution
    D_fit = res.x.astype(np.float64).view(dtype=np.complex128) # all the D-terms (complex)

    # Apply the solution
    for isite in range(len(sites)):
        obs_test.tarr['dr'][site_index[isite]] = D_fit[2*isite]
        obs_test.tarr['dl'][site_index[isite]] = D_fit[2*isite+1]    
    obs_test.data = simobs.apply_jones_inverse(obs_test,dcal=False,verbose=False)
    obs_test.dcal = True

    if show_solution:
        print("Original chi-squared: {:.4f}".format(chisq_total(obs.switch_polrep('circ').data, im_circ)))
        print("New chi-squared: {:.4f}\n".format(chisq_total(obs_test.data, im_circ)))
        for isite in range(len(sites)):       
            print(sites[isite])
            print('   D_R: {:.4f}'.format(D_fit[2*isite]))
            print('   D_L: {:.4f}\n'.format(D_fit[2*isite+1]))

    tstop = time.time()
    print("\nleakage_cal time: %f s" % (tstop - tstart))

    if not obs_apply==False:
        obs_test = obs_apply.copy()
        # Apply the solution
        for isite in range(len(sites)):
            obs_test.tarr['dr'][site_index[isite]] = D_fit[2*isite]
            obs_test.tarr['dl'][site_index[isite]] = D_fit[2*isite+1]
        obs_test.data = simobs.apply_jones_inverse(obs_test,dcal=False,verbose=False)
        obs_test.dcal = True
    else:
        obs_test = obs_test.switch_polrep(obs.polrep)

    return obs_test
