# pol_cal.py
# functions for D-term calibration
# new version that should be faster (2024)
#
#    Copyright (C) 2024 Andrew Chael
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

MAXIT = 10000  # maximum number of iterations in self-cal minimizer
NHIST = 50   # number of steps to store for hessian approx
MAXLS = 40   # maximum number of line search steps in BFGS-B
STOP = 1e-6  # convergence criterion

###################################################################################################
# Polarimetric Calibration
###################################################################################################

# TODO - other chi^2 terms, not just 'vis'?
# TODO - do we want to start with some nonzero D-term initial guess? 
# TODO - option to not frcal?
# TODO - pass other kwargs to the chisq? 
# TODO - handle gain cal == False, read in gains from a caltable

def leakage_cal_new(obs, im, sites=[], leakage_tol=.1, rescale_leakage_tol=False,
                    pol_fit=['RL', 'LR'], dtype='vis',
                    minimizer_method='L-BFGS-B', 
                    ttype='direct', fft_pad_factor=2, 
                    use_grad=True,
                    show_solution=True):
    """Polarimetric calibration (detects and removes polarimetric leakage,
       based on consistency with a given image)

       Args:
           obs (Obsdata): The observation to be calibrated
           im (Image): the reference image used for calibration

           sites (list): list of sites to include in the polarimetric calibration.
                         empty list calibrates all sites

           leakage_tol (float): leakage values exceeding this value will be disfavored by the prior
           rescale_leakage_tol (bool): if True, properly scale leakage tol for number of sites 
                                       (not done correctly in old version)
           
           pol_fit (list): list of visibilities to use; e.g., ['RL','LR'] or ['RR','LL','RL','LR']

           minimizer_method (str): Method for scipy.optimize.minimize (e.g., 'CG', 'BFGS')
           ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT
           fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in FFT

           use_grad (bool): if True, use gradients in minimizer
           
           show_solution (bool): if True, display the solution as it is calculated

           
       Returns:
           (Obsdata): the calibrated observation, with computed leakage values added to the obs.tarr
    """
    
    if not(obs.ampcal and obs.phasecal):
        raise Exception("obs must be amplitude and phase calibrated before leakage_cal! (TODO: generalize)")
        
        
    tstart = time.time()

    mask = []     # TODO: add image masks? 
    dtype = 'vis' # TODO: add other data terms? 
    
    # Do everything in a circular basis
    im_circ = im.switch_polrep('circ')
    obs_circ = obs.copy().switch_polrep('circ')

    # Check to see if the field rotation is corrected
    if obs_circ.frcal is False:
        print("Field rotation angles have not been corrected. Correcting now...")
        obs_circ.data = simobs.apply_jones_inverse(obs_circ, frcal=False, dcal=True, opacitycal=True, verbose=False)
        obs_circ.frcal = True 

    # List of all sites present in the observation. Make sure they are all in the tarr
    allsites = list(set(np.hstack((obs_circ.data['t1'], obs_circ.data['t2']))))
    for site in allsites:
        if not (site in obs_circ.tarr['site']):
            raise Exception("site %s not in obs.tarr!"%site)
            
    if len(sites) == 0:
        print("No stations specified for leakage calibration: defaulting to calibrating all sites !")
        sites = allsites
    # only include sites that are present in obs.tarr
    sites = [s for s in sites if s in allsites]
    site_index = [list(obs_circ.tarr['site']).index(s) for s in sites]
    
    # TODO do we want to start with some nonzero D-terms? 
    # Set all leakage terms in obs_circ to zero
    # (we will only correct leakage for those sites with new solutions)
    for j in range(len(obs_circ.tarr)):
        if obs_circ.tarr[j]['site'] in sites:
            continue
        obs_circ.tarr[j]['dr'] = obs_circ.tarr[j]['dl'] = 0.0j

    print("Finding leakage for sites:", sites)

    print("Precomputing visibilities...")    
    # get stations
    t1 = obs_circ.unpack('t1')['t1']
    t2 = obs_circ.unpack('t2')['t2']
    
    # index sites in t1, t2 position. If no calibrated site is used in a baseline, -1
    idx1 = np.array([sites.index(t) if (t in sites) else -1 for t in t1])
    idx2 = np.array([sites.index(t) if (t in sites) else -1 for t in t2])
        
    # get real data and sigmas
    # TODO add other chisqdata parameters? 
    # TODO modify chisqdata function to have the option to return samples? 
    
    (vis_RR, sigma_RR, _) = iu.chisqdata(obs_circ, im_circ, mask=mask, dtype=dtype, pol='RR',
                                         ttype=ttype, fft_pad_factor=fft_pad_factor)
    (vis_LL, sigma_LL, _) = iu.chisqdata(obs_circ, im_circ, mask=mask, dtype=dtype, pol='LL',
                                         ttype=ttype, fft_pad_factor=fft_pad_factor)
    (vis_RL, sigma_RL, _) = iu.chisqdata(obs_circ, im_circ, mask=mask, dtype=dtype, pol='RL',
                                         ttype=ttype, fft_pad_factor=fft_pad_factor)
    (vis_LR, sigma_LR, _) = iu.chisqdata(obs_circ, im_circ, mask=mask, dtype=dtype, pol='LR',
                                         ttype=ttype, fft_pad_factor=fft_pad_factor)

    # get simulated data (from simple Fourier transform)
    obs_sim = im_circ.observe_same_nonoise(obs_circ, 
                                           ttype=ttype, fft_pad_factor=fft_pad_factor, 
                                           zero_empty_pol=True,verbose=False)
                                           
    (ft_RR, _, _) = iu.chisqdata(obs_sim, im_circ, mask=mask, dtype=dtype, pol='RR',
                                 ttype=ttype, fft_pad_factor=fft_pad_factor)
    (ft_LL, _, _) = iu.chisqdata(obs_sim, im_circ, mask=mask, dtype=dtype, pol='LL',
                                 ttype=ttype, fft_pad_factor=fft_pad_factor)
    (ft_RL, _, _) = iu.chisqdata(obs_sim, im_circ, mask=mask, dtype=dtype, pol='RL',
                                 ttype=ttype, fft_pad_factor=fft_pad_factor)
    (ft_LR, _, _) = iu.chisqdata(obs_sim, im_circ, mask=mask, dtype=dtype, pol='LR',
                                 ttype=ttype, fft_pad_factor=fft_pad_factor)
                                                                              
    # field rotation angles 
    el1 = obs_circ.unpack(['el1'], ang_unit='rad')['el1']
    el2 = obs_circ.unpack(['el2'], ang_unit='rad')['el2']
    par1 = obs_circ.unpack(['par_ang1'], ang_unit='rad')['par_ang1']
    par2 = obs_circ.unpack(['par_ang2'], ang_unit='rad')['par_ang2']

    fr_elev1 = np.array([obs_circ.tarr[obs_circ.tkey[o['t1']]]['fr_elev'] for o in obs.data])
    fr_elev2 = np.array([obs_circ.tarr[obs_circ.tkey[o['t2']]]['fr_elev'] for o in obs.data])
    fr_par1  = np.array([obs_circ.tarr[obs_circ.tkey[o['t1']]]['fr_par']  for o in obs.data])
    fr_par2  = np.array([obs_circ.tarr[obs_circ.tkey[o['t2']]]['fr_par']  for o in obs.data])
    fr_off1  = np.array([obs_circ.tarr[obs_circ.tkey[o['t1']]]['fr_off']  for o in obs.data])
    fr_off2  = np.array([obs_circ.tarr[obs_circ.tkey[o['t2']]]['fr_off']  for o in obs.data])

    fr1 = fr_elev1*el1 + fr_par1*par1 + fr_off1*np.pi/180.
    fr2 = fr_elev2*el2 + fr_par2*par2 + fr_off2*np.pi/180.

    Delta = fr1 - fr2
    Phi = fr1 + fr2

    # TODO: read in gains from caltable? 
    # gains
    GR1 = np.ones(fr1.shape)
    GL1 = np.ones(fr1.shape)
    GR2 = np.ones(fr2.shape)
    GL2 = np.ones(fr2.shape)
        
    if not(len(Delta)==len(vis_RR)==len(sigma_LL)==len(ft_RR)==len(t1)):
        raise Exception("not all data columns the right length in pol_cal!")
    Nvis = len(vis_RR)
    
    # define the error function
    def chisq_total(Dpar):
        # all the D-terms as complex numbers. If const_fpol, fpol is the last parameter.
        D = Dpar.astype(np.float64).view(dtype=np.complex128)

        # current D-terms for each baseline, zero for stations not calibrated (TODO faster?)
        DR1 = np.asarray([D[2*sites.index(s)] if s in sites else 0. for s in t1])
        DL1 = np.asarray([D[2*sites.index(s)+1] if s in sites else 0. for s in t1])
        
        DR2 = np.asarray([D[2*sites.index(s)] if s in sites else 0. for s in t2])
        DL2 = np.asarray([D[2*sites.index(s)+1] if s in sites else 0. for s in t2])
        
        # simulated visibilities and chisqs with leakage
        chisq_RR = chisq_LL = chisq_RL = chisq_LR = 0.0
        if 'RR' in pol_fit:
            vis_RR_leak = ft_RR + DR1*DR2.conj()*np.exp(2j*Delta)*ft_LL + DR1*np.exp(2j*fr1)*ft_LR + DR2.conj()*np.exp(-2j*fr2)*ft_RL
            vis_RR_leak *= GR1*GR2.conj()
            
            chisq_RR = np.sum(np.abs(vis_RR - vis_RR_leak)**2 / (sigma_RR**2))
            chisq_RR = chisq_RR / (2.*Nvis)
        if 'LL' in pol_fit:
            vis_LL_leak = ft_LL + DL1*DL2.conj()*np.exp(-2j*Delta)*ft_RR + DL1*np.exp(-2j*fr1)*ft_RL + DL2.conj()*np.exp(2j*fr2)*ft_LR
            vis_LL_leak *= GL1*GL2.conj()
            
            chisq_LL = np.sum(np.abs(vis_LL - vis_LL_leak)**2 / (sigma_LL**2))
            chisq_LL = chisq_LL / (2.*Nvis)
        if 'RL' in pol_fit:
            vis_RL_leak = ft_RL + DR1*DL2.conj()*np.exp(2j*Phi)*ft_LR + DR1*np.exp(2j*fr1)*ft_LL + DL2.conj()*np.exp(2j*fr2)*ft_RR          
            vis_RL_leak *= GR1*GL2.conj()  
            
            chisq_RL = np.sum(np.abs(vis_RL - vis_RL_leak)**2 / (sigma_RL**2))
            chisq_RL = chisq_RL / (2.*Nvis)
        if 'LR' in pol_fit:
            vis_LR_leak = ft_LR + DL1*DR2.conj()*np.exp(-2j*Phi)*ft_RL + DL1*np.exp(-2j*fr1)*ft_RR + DR2.conj()*np.exp(-2j*fr2)*ft_LL                       
            vis_LR_leak *= GL1*GR2.conj()
                 
            chisq_LR = np.sum(np.abs(vis_LR - vis_LR_leak)**2 / (sigma_LR**2))            
            chisq_LR = chisq_LR / (2.*Nvis)
            
        chisq_tot = (chisq_RR + chisq_LL + chisq_RL + chisq_LR)/len(pol_fit)
        return chisq_tot
        
    def errfunc(Dpar):
        # chi-squared        
        chisq_tot = chisq_total(Dpar)
        
        # prior on the D terms
        # TODO 
        prior = np.sum((np.abs(Dpar)**2)/(leakage_tol**2))
        if rescale_leakage_tol:
            prior = prior / (len(Dpar))
            
        return  chisq_tot + prior             

    # define the error function gradient

    def chisq_total_grad(Dpar):
        
       # residual and dV/dD terms
        if 'RR' in pol_fit:
            vis_RR_leak = ft_RR + DR1*DR2.conj()*np.exp(2j*Delta)*ft_LL + DR1*np.exp(2j*fr1)*ft_LR + DR2.conj()*np.exp(-2j*fr2)*ft_RL
            vis_RR_leak *= GR1*GR2.conj()

            resid_RR = (vis_RR - vis_RR_leak).conj() 

            dRR_dReDR1 = DR2.conj()*np.exp(2j*Delta)*ft_LL + np.exp(2j*fr1)*ft_LR
            dRR_dReDR1 *= GR1*GR2.conj()
            
            dRR_dReDR2 = DR1*np.exp(2j*Delta)*ft_LL + np.exp(-2j*fr2)*ft_RL
            dRR_dReDR2 *= GR1*GR2.conj()

        if 'LL' in pol_fit:
            vis_LL_leak = ft_LL + DL1*DL2.conj()*np.exp(-2j*Delta)*ft_RR + DL1*np.exp(-2j*fr1)*ft_RL + DL2.conj()*np.exp(2j*fr2)*ft_LR
            vis_LL_leak *= GL1*GL2.conj()

            resid_LL = (vis_LL - vis_LL_leak).conj() 

            dLL_dReDL1 = DL2.conj()*np.exp(-2j*Delta)*ft_RR + np.exp(-2j*fr1)*ft_RL
            dLL_dReDL1 *= GL1*GL2.conj()

            dLL_dReDL2 = DL1*np.exp(-2j*Delta)*ft_RR + np.exp(2j*fr2)*ft_LR
            dLL_dReDL2 *= GL1*GL2.conj()


        if 'RL' in pol_fit:
            vis_RL_leak = ft_RL + DR1*DL2.conj()*np.exp(2j*Phi)*ft_LR + DR1*np.exp(2j*fr1)*ft_LL + DL2.conj()*np.exp(2j*fr2)*ft_RR          
            vis_RL_leak *= GR1*GL2.conj()  

            resid_RL = (vis_RL - vis_RL_leak).conj() 

            dRL_dReDR1 = DL2.conj()*np.exp(2j*Phi)*ft_LR + np.exp(2j*fr1)*ft_LL
            dRL_dReDR1 *= GR1*GL2.conj()
    
            dRL_dReDL2 = DR1*np.exp(2j*Phi)*ft_LR + np.exp(2j*fr2)*ft_RR
            dRL_dReDL2 *= GR1*GL2.conj()

            
        if 'LR' in pol_fit:
            vis_LR_leak = ft_LR + DL1*DR2.conj()*np.exp(-2j*Phi)*ft_RL + DL1*np.exp(-2j*fr1)*ft_RR + DR2.conj()*np.exp(-2j*fr2)*ft_LL                       
            vis_LR_leak *= GL1*GR2.conj()

            resid_LR = (vis_LR - vis_LR_leak).conj() 

            dLR_dReDL1 = DR2.conj()*np.exp(-2j*Phi)*ft_RL + np.exp(-2j*fr1)*ft_RR
            dLR_dReDL1 *= GL1*GR2.conj()
 
            dLR_dReDR2 = DL1*np.exp(-2j*Phi)*ft_RL + np.exp(-2j*fr2)*ft_LL
            dLR_dReDR2 *= GL1*GR2.conj()

                    
        # gradients, sum over baselines
        # TODO remove for loop with some fancy vectorization? 
        for isite in range(len(sites)):
            mask1 = (idx1 == isite)
            mask2 = (idx2 == isite)
            
            # DR
            regrad = 0
            imgrad = 0
            if 'RR' in pol_fit:
                terms = -1 * resid_RR[mask1] * dRR_dReDR1[mask1] / (sigma_RR[mask1]**2)
                regrad += np.sum(np.real(terms))/Nvis
                imgrad += -1*np.sum(np.imag(terms))/Nvis
                
                terms = -1 * resid_RR[mask2] * dRR_dReDR2[mask2] / (sigma_RR[mask2]**2)
                regrad += np.sum(np.real(terms))/Nvis
                imgrad += np.sum(np.imag(terms))/Nvis
          
            if 'RL' in pol_fit:
                terms = -1 * resid_RL[mask1] * dRL_dReDR1[mask1] / (sigma_RL[mask1]**2)
                regrad += np.sum(np.real(terms))/Nvis
                imgrad += -1*np.sum(np.imag(terms))/Nvis
          
            if 'LR' in pol_fit: 
                terms = -1 * resid_LR[mask2] * dLR_dReDR2[mask2] / (sigma_LR[mask2]**2)
                regrad += np.sum(np.real(terms))/Nvis
                imgrad += np.sum(np.imag(terms))/Nvis
          
            chisqgrad[4*isite] += regrad  # Re(DR)
            chisqgrad[4*isite+1] += imgrad  # Im(DR)
                          
            # DL
            regrad = 0
            imgrad = 0
            if 'LL' in pol_fit:
                terms = -1 * resid_LL[mask1] * dLL_dReDL1[mask1] / (sigma_LL[mask1]**2)
                regrad += np.sum(np.real(terms))/Nvis
                imgrad += -1*np.sum(np.imag(terms))/Nvis

                terms = -1 * resid_LL[mask2] * dLL_dReDL2[mask2] / (sigma_LL[mask2]**2)
                regrad += np.sum(np.real(terms))/Nvis
                imgrad += np.sum(np.imag(terms))/Nvis
          
            if 'RL' in pol_fit: 
                terms = -1 * resid_RL[mask2] * dRL_dReDL2[mask2] / (sigma_RL[mask2]**2)
                regrad += np.sum(np.real(terms))/Nvis
                imgrad += np.sum(np.imag(terms))/Nvis
                          
            if 'LR' in pol_fit:
                terms = -1 * resid_LR[mask1] * dLR_dReDL1[mask1] / (sigma_LR[mask1]**2)
                regrad += np.sum(np.real(terms))/Nvis
                imgrad += -1*np.sum(np.imag(terms))/Nvis
                
            chisqgrad[4*isite+2] += regrad  # Re(DL)
            chisqgrad[4*isite+3] += imgrad  # Im(DL)
           
        chisqgrad /= len(pol_fit)
                    
        return chisqgrad
        
    def errfunc_grad(Dpar):
        # gradient of the chi^2
        chisqgrad = chisq_total_grad(Dpar)
        
        # gradient of the prior
        priorgrad = 2*Dpar / (leakage_tol**2)
        if rescale_leakage_tol:
            priorgrad = priorgrad / (len(Dpar))
            
        return  chisqgrad + priorgrad      

    # Gradient test - remove!
#    def test_grad(Dpar):
#        grad_ana = errfunc_grad(Dpar)
#        grad_num1 = np.zeros(len(Dpar))
#        for i in range(len(Dpar)):
#            dd = 1.e-8
#            Dpar_dd = Dpar.copy()
#            Dpar_dd[i] += dd
#            grad_num1[i] = (errfunc(Dpar_dd) - errfunc(Dpar))/dd
#        grad_num2 = np.zeros(len(Dpar))
#        for i in range(len(Dpar)):
#            dd = -1.e-8
#            Dpar_dd = Dpar.copy()
#            Dpar_dd[i] += dd
#            grad_num2[i] = (errfunc(Dpar_dd) - errfunc(Dpar))/dd
#                        
#        plt.close('all')
#        plt.ion()
#        plt.figure()
#        plt.plot(np.arange(len(Dpar)), grad_ana, 'ro')
#        plt.plot(np.arange(len(Dpar)), grad_num1, 'b.')     
#        plt.plot(np.arange(len(Dpar)), grad_num2, 'bx')   
#        plt.xticks(np.arange(0,len(Dpar),4), sites)
#                
#        plt.figure()
#        zscal = 1.e-32*np.min(np.abs(grad_ana)[grad_ana!=0])
#        plt.plot(np.arange(len(Dpar)), 100-100*(grad_num1+zscal)/(grad_ana+zscal),'b.') 
#        plt.plot(np.arange(len(Dpar)), 100-100*(grad_num2+zscal)/(grad_ana+zscal),'bx') 
#        plt.xticks(np.arange(0,len(Dpar),4), sites)
#        plt.ylim(-1,1)
#        plt.show()
#        return
      
#    Dpar_guess = .1*np.random.randn(len(sites)*4)
#    test_grad(Dpar_guess)
    
    print("Calibrating D-terms...")    
    # Now, we will finally minimize the total error term. We need two complex leakage terms for each site
    if minimizer_method=='L-BFGS-B':
        optdict = {'maxiter': MAXIT,
                   'ftol': STOP, 'gtol': STOP,
                   'maxcor': NHIST, 'maxls': MAXLS}
    else:
        optdict = {'maxiter': MAXIT}  
        
    Dpar_guess = np.zeros(len(sites)*2, dtype=np.complex128).view(dtype=np.float64)
    if use_grad:                   
        res = opt.minimize(errfunc, Dpar_guess, method=minimizer_method, options=optdict, jac=errfunc_grad)
    else:
        res = opt.minimize(errfunc, Dpar_guess, method=minimizer_method, options=optdict)    

    print(errfunc(Dpar_guess),errfunc(res.x))
    
    # get solution
    Dpar_fit  =  res.x.astype(np.float64)
    D_fit = Dpar_fit.view(dtype=np.complex128)  # all the D-terms (complex)

    # Apply the solution
    obs_out = obs_circ.copy() # TODO or overwrite directly?
    for isite in range(len(sites)):
        obs_out.tarr['dr'][site_index[isite]] = D_fit[2*isite]
        obs_out.tarr['dl'][site_index[isite]] = D_fit[2*isite+1]
    obs_out.data = simobs.apply_jones_inverse(obs_out, dcal=False, frcal=True, opacitycal=True, verbose=False)
    obs_out.dcal = True

    # Re-populate any additional leakage terms that were present
    for j in range(len(obs_out.tarr)):
        if obs_out.tarr[j]['site'] in sites:
            continue
        obs_out.tarr[j]['dr'] = obs.tarr[j]['dr']
        obs_out.tarr[j]['dl'] = obs.tarr[j]['dl']

    # TODO are these diagnostics correct? 
    if show_solution:

        chisq_orig = chisq_total(Dpar_fit*0)
        chisq_new  = chisq_total(Dpar_fit)

        print("Original chi-squared: {:.4f}".format(chisq_orig))
        print("New chi-squared: {:.4f}\n".format(chisq_new))
        for isite in range(len(sites)):
            print(sites[isite])
            print('   D_R: {:.4f}'.format(D_fit[2*isite]))
            print('   D_L: {:.4f}\n'.format(D_fit[2*isite+1]))

    tstop = time.time()
    print("\nleakage_cal time: %f s" % (tstop - tstart))


    obs_out = obs_out.switch_polrep(obs.polrep)


    return obs_out
    
    


