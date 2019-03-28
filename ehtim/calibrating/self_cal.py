# selfcal.py
# functions for self-calibration
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
import scipy.special as spec
import scipy.optimize as opt
import sys
import itertools as it
import time
import copy

import ehtim.obsdata
from ehtim.observing.obs_helpers import *
import ehtim.imaging.imager_utils as iu

from multiprocessing import cpu_count
from multiprocessing import Pool
from multiprocessing import Process, Value, Lock

from ehtim.calibrating.cal_helpers import *
import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in log")

MAXIT=10000 # maximum number of iterations in self-cal minimizer

###################################################################################################################################
#Self-Calibration
###################################################################################################################################
def self_cal(obs, im, sites=[], method="both", pol='I', minimizer_method='BFGS',
             pad_amp=0., gain_tol=.2, solution_interval=0.0, scan_solutions=False, 
             ttype='direct', fft_pad_factor=2, caltable=False, debias=True, copy_closure_tables=True,
             processes=-1,show_solution=False,msgtype='bar'):
    """Self-calibrate a dataset to an image.

       Args:
           obs (Obsdata): The observation to be calibrated
           im (Image): the image to be calibrated  to
           sites (list): list of sites to include in the self calibration. empty list calibrates all sites

           method (str): chooses what to calibrate, 'amp', 'phase', or 'both' 
           minimizer_method (str): Method for scipy.optimize.minimize (e.g., 'CG', 'BFGS', 'Nelder-Mead', etc.)
           pol (str): which image polarization to self-calibrate visibilities to 

           pad_amp (float): adds fractional uncertainty to amplitude sigmas in quadrature
           gain_tol (float or list): gains that exceed this value will be disfavored by the prior
                                     for asymmetric gain_tol for corrections below/above unity, pass a 2-element list
           solution_interval (float): solution interval in seconds; one gain is derived for each interval.
                                      If 0.0, a solution is determined for each unique time in the observation.
           scan_solutions (bool): If True, determine one gain per site per scan (supersedes solution_interval)

           caltable (bool): if True, returns a Caltable instead of an Obsdata 
           processes (int): number of cores to use in multiprocessing

           ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT
           fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in FFT

           debias (bool): If True, debias the amplitudes
           show_solution (bool): if True, display the solution as it is calculated
           msgtype (str): type of progress message to be printed, default is 'bar'

       Returns:
           (Obsdata): the calibrated observation, if caltable==False
           (Caltable): the derived calibration table, if caltable==True
    """

    if pol not in ['I','Q','U','V','RR','LL']:
        raise Exception("Can only self-calibrate to I, Q, U, V, RR, or LL images!")
    if pol in ['I','Q','U','V']:
        if obs.polrep!='stokes':
            raise Exception("selfcal pol is a stokes parameter, but obs.polrep!='stokes'")
        im = im.switch_polrep('stokes',pol)
    elif pol in ['RR','LL','RRLL']:
        if obs.polrep!='circ':
            raise Exception("selfcal pol is RR or LL, but obs.polrep!='circ'")
        im = im.switch_polrep('circ',pol)

    # V = model visibility, V' = measured visibility, G_i = site gain
    # G_i * conj(G_j) * V_ij = V'_ij
    if len(sites) == 0:
        print("No stations specified in self cal: defaulting to calibrating all stations!")
        sites = obs.tarr['site']

    # First, sample the model visibilities of the specified polarization
    print("Computing the Model Visibilities with " + ttype + " Fourier Transform...")
    obs_clean = im.observe_same_nonoise(obs, ttype=ttype, fft_pad_factor=fft_pad_factor)
    V = obs_clean.data[vis_poldict[pol]]

    # Partition the list of observed visibilities into scans
    scans     = obs.tlist(t_gather=solution_interval, scan_gather=scan_solutions)
    scans_cal = copy.copy(scans)

    # Partition the list of model visibilities into scans
    V_scans      = [o[vis_poldict[pol]] for o in obs_clean.tlist(t_gather=solution_interval, scan_gather=scan_solutions)]

    # Make the pool for parallel processing
    if processes > 0:
        counter = Counter(initval=0, maxval=len(scans))
        print("Using Multiprocessing with %d Processes" % processes)
        pool = Pool(processes=processes, initializer=init, initargs=(counter,))
    elif processes == 0:
        counter = Counter(initval=0, maxval=len(scans))
        processes = int(cpu_count())
        print("Using Multiprocessing with %d Processes" % processes)
        pool = Pool(processes=processes, initializer=init, initargs=(counter,))
    else:
        print("Not Using Multiprocessing")

    # loop over scans and calibrate
    tstart = time.time()
    if processes > 0: # run on multiple cores with multiprocessing
        scans_cal = np.array(pool.map(get_selfcal_scan_cal, [[i, len(scans), scans[i], im, V_scans[i], sites, obs.polrep, pol, method, minimizer_method, 
                                                              show_solution, pad_amp, gain_tol, caltable, debias, msgtype
                                                             ] for i in range(len(scans))]))

    else: # run on a single core
        for i in range(len(scans)):
            prog_msg(i, len(scans),msgtype=msgtype,nscan_last=i-1)
            scans_cal[i] = self_cal_scan(scans[i], im, V_scan=V_scans[i], sites=sites, polrep=obs.polrep, pol=pol,
                                 method=method, minimizer_method=minimizer_method, show_solution=show_solution, debias=debias,
                                 pad_amp=pad_amp, gain_tol=gain_tol, caltable=caltable)


    tstop = time.time()
    print("\nself_cal time: %f s" % (tstop - tstart))

    if caltable: # assemble the caltable to return
        allsites = obs.tarr['site']
        caldict = scans_cal[0]
        for i in range(1,len(scans_cal)):
            row = scans_cal[i]
            for site in allsites:
                try: dat = row[site]
                except KeyError: continue

                try: caldict[site] = np.append(caldict[site], row[site])
                except KeyError: caldict[site] = dat

        caltable = ehtim.caltable.Caltable(obs.ra, obs.dec, obs.rf, obs.bw, caldict, obs.tarr,
                                           source=obs.source, mjd=obs.mjd, timetype=obs.timetype)
        out = caltable
    else: # return a calibrated observation
        arglist, argdict = obs.obsdata_args()
        arglist[4] = np.concatenate(scans_cal)
        out = ehtim.obsdata.Obsdata(*arglist, **argdict)
        if copy_closure_tables:
            out.camp = obs.camp
            out.logcamp = obs.logcamp
            out.cphase = obs.cphase

    # close multiprocessing jobs
    if processes != -1:
        pool.close()

    return out

def self_cal_scan(scan, im, V_scan=[], sites=[], polrep='stokes', pol='I', method="both", 
                  minimizer_method='BFGS', show_solution=False, 
                  pad_amp=0., gain_tol=.2, debias=True, caltable=False):
    """Self-calibrate a scan to an image.

       Args:
           scan (np.recarray): data array of type DTPOL_STOKES or DTPOL_CIRC containing the scan visibility data
           im (Image): the image to be calibrated  to
           sites (list): list of sites to include in the self calibration. empty list calibrates all sites
           V_scan (list) : precomputed scan visibilities

           polrep (str): 'stokes' or 'circ' to specify the  polarization products in scan
           pol (str): which image polarization to self-calibrate visibilities to 
           method (str): chooses what to calibrate, 'amp', 'phase', or 'both' 
           minimizer_method (str): Method for scipy.optimize.minimize (e.g., 'CG', 'BFGS', 'Nelder-Mead', etc.)
           pad_amp (float): adds fractional uncertainty to amplitude sigmas in quadrature
           gain_tol (float or list): gains that exceed this value will be disfavored by the prior
                                     for asymmetric gain_tol for corrections below/above unity, pass a 2-element list

           debias (bool): If True, debias the amplitudes
           caltable (bool): if True, returns a Caltable instead of an Obsdata 
           show_solution (bool): if True, display the solution as it is calculated
           
       Returns:
           (Obsdata): the calibrated observation, if caltable==False
           (Caltable): the derived calibration table, if caltable==True
    """

    if len(sites) == 0:
        print("No stations specified in self cal: defaulting to calibrating all !")
        sites = list(set(scan['t1']).union(set(scan['t2'])))

    if len(V_scan) < 1:
        # This is not correct. Need to update to use polarization dictionary
        uv = np.hstack((scan['u'].reshape(-1,1), scan['v'].reshape(-1,1)))
        A = ftmatrix(im.psize, im.xdim, im.ydim, uv, pulse=im.pulse)
        V_scan = np.dot(A, im.imvec)

    if type(gain_tol) == float or type(gain_tol) == int:
        gain_tol = [gain_tol, gain_tol]

    # create a dictionary to keep track of gains
    tkey = {b:a for a,b in enumerate(sites)}

    # make a list of gain keys that relates scan bl gains to solved site ones
    # -1 means that this station does not have a gain that is being solved for
    g1_keys = []
    g2_keys = []
    scan_keys = []
    for row in scan:
        try:
            g1_keys.append(tkey[row['t1']])
        except KeyError:
            g1_keys.append(-1)
        try:
            g2_keys.append(tkey[row['t2']])
        except KeyError:
            g2_keys.append(-1)

    # no sites to calibrate on this scan!
    if np.all(g1_keys == -1) and np.all(g2_keys == -1):
        return scan

    # get scan visibilities of the specified polarization
    vis = scan[vis_poldict[pol]]
    sigma = scan[sig_poldict[pol]]

    if method=='amp':
        if debias:
            vis = amp_debias(np.abs(vis), np.abs(sigma))
        else:
            vis = np.abs(vis)

    sigma_inv = 1.0/np.sqrt(sigma**2+ (pad_amp*np.abs(vis))**2)


    # initial guess for gains
    gpar_guess = np.ones(len(sites), dtype=np.complex128).view(dtype=np.float64)

    # error function
    def errfunc(gpar):
        g = gpar.astype(np.float64).view(dtype=np.complex128) # all the forward site gains (complex)

        if method=="phase":
            g = g/np.abs(g) 
        if method=="amp":
            g = np.abs(np.real(g))

        # append the default values to g for missing gains
        g = np.append(g, 1.)
        g1 = g[g1_keys]
        g2 = g[g2_keys]

        if method=='amp':
            verr = np.abs(vis) - g1*g2.conj() * np.abs(V_scan)
        else:
            verr = vis - g1*g2.conj() * V_scan

        nan_mask = [not np.isnan(v) for v in verr]
        verr = verr[nan_mask]   

        # goodness-of-fit for gains 
        chisq = np.sum((verr.real * sigma_inv[nan_mask])**2) + np.sum((verr.imag * sigma_inv[nan_mask])**2)

        # prior on the gains
        chisq_g = np.sum(np.log(np.abs(g))**2 / ((np.abs(g) > 1) * gain_tol[0] + (np.abs(g) <= 1) * gain_tol[1])**2)

        return chisq + chisq_g

    # use gradient descent to find the gains
    optdict = {'maxiter': MAXIT} # minimizer params
    res = opt.minimize(errfunc, gpar_guess, method=minimizer_method, options=optdict)

    # save the solution
    g_fit = res.x.view(np.complex128)

    if show_solution == True:
        print (np.abs(g_fit))

    if method=="phase":
        g_fit = g_fit / np.abs(g_fit)
    if method=="amp":
        g_fit = np.abs(np.real(g_fit))

    g_fit = np.append(g_fit, 1.)

    # Derive a calibration table or apply the solution to the scan
    if caltable: 
        allsites = list(set(scan['t1']).union(set(scan['t2'])))

        caldict = {}
        for site in allsites:
            if site in sites:
                site_key = tkey[site]
            else:
                site_key = -1

            # We will *always* set the R and L gain corrections to be equal in self calibration, to avoid breaking polarization consistency relationships
            rscale = g_fit[site_key]**-1
            lscale = g_fit[site_key]**-1

#            # If we selfcal on a stokes image, set R&L gains equal.
#            if pol in ['I','Q','U','V']:
#                rscale = g_fit[site_key]**-1
#                lscale = g_fit[site_key]**-1

#            # TODO is this right??
#            # But if we selfcal  on RR or LL, only set the appropriate gain
#            elif pol=='RR':
#                rscale = g_fit[site_key]**-1
#                lscale = 1
#            elif pol=='LL':
#                lscale = g_fit[site_key]**-1
#                rscale = 1

            # Note: we may want to give two entries for the start/stop times when a non-zero solution interval is used
            caldict[site] = np.array((scan['time'][0], rscale, lscale), dtype=DTCAL)

        out = caldict

    else: 
        g1_fit = g_fit[g1_keys]
        g2_fit = g_fit[g2_keys]   

        gij_inv = (g1_fit * g2_fit.conj())**(-1)    

        if polrep=='stokes': 
            # scale visibilities
            for vistype in ['vis','qvis','uvis','vvis']:
                scan[vistype]  *= gij_inv
            # scale sigmas
            for sigtype in ['sigma','qsigma','usigma','vsigma']:
                scan[sigtype]  *= np.abs(gij_inv)
        elif polrep=='circ': 
            # scale visibilities
            for vistype in ['rrvis','llvis','rlvis','lrvis']:
                scan[vistype]  *= gij_inv
            # scale sigmas
            for sigtype in ['rrsigma','llsigma','rlsigma','lrsigma']:
                scan[sigtype]  *= np.abs(gij_inv)  

        
#        # TODO is this right??
#        elif polrep=='circ': 

#            if pol=='RR':
#                scan['rrvis'] *= (g1_fit * g2_fit.conj())**(-1)
#                scan['llvis'] *= 1
#                scan['rlvis'] *= g1_fit**(-1)
#                scan['lrvis'] *= g2_fit.conj()**(-1)

#                scan['rrsigma'] *= np.abs((g1_fit * g2_fit.conj())**(-1))
#                scan['llsigma'] *= 1
#                scan['rlsigma'] *= np.abs(g1_fit**(-1))
#                scan['lrsigma'] *= np.abs(g2_fit.conj()**(-1))

#            elif pol=='LL':
#                scan['rrvis'] *= 1
#                scan['llvis'] *= (g1_fit * g2_fit.conj())**(-1)
#                scan['rlvis'] *= g2_fit.conj()**(-1)
#                scan['lrvis'] *= g1_fit**(-1)

#                scan['rrsigma'] *= 1
#                scan['llsigma'] *= np.abs((g1_fit * g2_fit.conj())**(-1))
#                scan['rlsigma'] *= np.abs(g2_fit.conj()**(-1))
#                scan['lrsigma'] *= np.abs(g1_fit**(-1))

        out = scan

    return out

def init(x):
    global counter
    counter = x

def get_selfcal_scan_cal(args):
    return get_selfcal_scan_cal2(*args)

def get_selfcal_scan_cal2(i, n, scan, im, V_scan, sites, polrep, pol, method, minimizer_method, show_solution, pad_amp, gain_tol, caltable,debias,msgtype):
    if n > 1:
        global counter
        counter.increment()
        prog_msg(counter.value(), counter.maxval,msgtype,counter.value()-1)

    return self_cal_scan(scan, im, V_scan=V_scan, sites=sites, polrep=polrep, pol=pol, method=method, minimizer_method=minimizer_method, show_solution=show_solution, 
                         pad_amp=pad_amp, gain_tol=gain_tol, caltable=caltable, debias=debias)











