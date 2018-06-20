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
import copy

import ehtim.obsdata
from ehtim.observing.obs_helpers import *
import ehtim.imaging.imager_utils as iu

from multiprocessing import cpu_count
from multiprocessing import Pool
from multiprocessing import Process, Value, Lock

import itertools
import time
ZBLCUTOFF = 1.e7;

###################################################################################################################################
#Self-Calibration
###################################################################################################################################
def self_cal(obs, im, sites=[], method="both", show_solution=False, pad_amp=0., gain_tol=.2, 
             ttype='nfft', fft_pad_factor=2, caltable=False, processes=-1):
    """Self-calibrate a dataset to an image.

       Args:
           obs (Obsdata): The observation to be calibrated
           im (Image): the image to be calibrated  to
           sites (list): list of sites to include in the network calibration. empty list calibrates all sites

           method (str): chooses what to calibrate, 'amp', 'phase', or 'both' 
           pad_amp (float): adds fractional uncertainty to amplitude sigmas in quadrature
           gain_tol (float): gains that exceed this value will be disfavored by the prior

           caltable (bool): if True, returns a Caltable instead of an Obsdata 
           processes (int): number of cores to use in multiprocessing

           ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT
           fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in FFT

           show_solution (bool): if True, display the solution as it is calculated
           

       Returns:
           (Obsdata): the calibrated observation, if caltable==False
           (Caltable): the derived calibration table, if caltable==True
    """

    # V = model visibility, V' = measured visibility, G_i = site gain
    # G_i * conj(G_j) * V_ij = V'_ij
    if len(sites) < 2:
        print("less than 2 stations specified in self cal: defaulting to calibrating all stations!")
        sites = obs.tarr['site']

    # First, sample the model visibilities
    print("Computing the Model Visibilities with " + ttype + " Fourier Transform...")
    if ttype == 'direct':
        data_arr = obs.unpack(['u','v','vis','sigma'])
        uv = np.hstack((data_arr['u'].reshape(-1,1), data_arr['v'].reshape(-1,1)))
        A = ftmatrix(im.psize, im.xdim, im.ydim, uv, pulse=im.pulse)
        V = np.dot(A, im.imvec)
    else:
        (data, sigma, fft_A) = iu.chisqdata_vis_fft(obs, im, fft_pad_factor=fft_pad_factor)
        im_info, sampler_info_list, gridder_info_list = fft_A
        vis_arr = iu.fft_imvec(im.imvec, im_info)
        V = iu.sampler(vis_arr, sampler_info_list, sample_type='vis')
    print("Done!")

    # Partition the list of model visibilities into scans
    from itertools import islice
    it = iter(V)
    scan_lengths = [len(o) for o in obs.tlist()]
    V_scans      = [list(islice(it, 0, i)) for i in scan_lengths]

    # get scans
    scans     = obs.tlist()
    scans_cal = copy.copy(scans)

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

        scans_cal = np.array(pool.map(get_selfcal_scan_cal, [[i, len(scans), scans[i], im, V_scans[i], sites, method,
                                                              show_solution, pad_amp, gain_tol, caltable, 
                                                             ] for i in range(len(scans))]))

    else: # run on a single core

        for i in range(len(scans)):
            cal_prog_msg(i, len(scans))
            scans_cal[i] = self_cal_scan(scans[i], im, V_scan=V_scans[i], sites=sites,
                                 method=method, show_solution=show_solution,
                                 pad_amp=pad_amp, gain_tol=gain_tol, caltable=caltable)

    print('DONE')
    tstop = time.time()
    print("self_cal time: %f s" % (tstop - tstart))

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
                                           source = obs.source, mjd=obs.mjd, timetype=obs.timetype)
        out = caltable
    else: # return a calibrated observation
        obs_cal = ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw,
                                        np.concatenate(scans_cal), obs.tarr, source=obs.source, mjd=obs.mjd,
                                        ampcal=obs.ampcal, phasecal=obs.phasecal, dcal=obs.dcal, frcal=obs.frcal,
                                        timetype=obs.timetype)

        out = obs_cal

    # close multiprocessing jobs
    if processes != -1:
        pool.close()

    return out

def self_cal_scan(scan, im, V_scan=[], sites=[], method="both", show_solution=False, pad_amp=0., gain_tol=.2, caltable=False):
    """Self-calibrate a scan to an image.

       Args:
           scan (np.recarray): data array of type DTPOL containing the scan visibility data
           im (Image): the image to be calibrated  to
           sites (list): list of sites to include in the network calibration. empty list calibrates all sites
           V_scan (list) : precomputed scan visibilities

           method (str): chooses what to calibrate, 'amp', 'phase', or 'both' 
           pad_amp (float): adds fractional uncertainty to amplitude sigmas in quadrature
           gain_tol (float): gains that exceed this value will be disfavored by the prior

           caltable (bool): if True, returns a Caltable instead of an Obsdata 
           show_solution (bool): if True, display the solution as it is calculated
           
       Returns:
           (Obsdata): the calibrated observation, if caltable==False
           (Caltable): the derived calibration table, if caltable==True
    """

    if len(sites) < 2:
        print("less than 2 stations specified in network cal: defaulting to calibrating all !")
        sites = list(set(scan['t1']).union(set(scan['t2'])))

    if len(V_scan) < 1:
        uv = np.hstack((scan['u'].reshape(-1,1), scan['v'].reshape(-1,1)))
        A = ftmatrix(im.psize, im.xdim, im.ydim, uv, pulse=im.pulse)
        V_scan = np.dot(A, im.imvec)

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

    sigma_inv = 1.0/(scan['sigma'] + pad_amp*np.abs(scan['vis']))

    # initial guess for gains
    gpar_guess = np.ones(len(sites), dtype=np.complex128).view(dtype=np.float64)

    # error function
    def errfunc(gpar):
        g = gpar.astype(np.float64).view(dtype=np.complex128) # all the forward site gains (complex)

        if method=="phase":
            g = g/np.abs(g) # TODO: use exp(i*np.arg())?
        if method=="amp":
             g = np.abs(np.real(g))
            #g = np.abs(g)

        # append the default values to g for missing gains
        g = np.append(g, 1.)
        g1 = g[g1_keys]
        g2 = g[g2_keys]

        #TODO debias!
        if method=='amp':
            verr = np.abs(scan['vis']) - g1*g2.conj() * np.abs(V_scan)
        else:
            verr = scan['vis'] - g1*g2.conj() * V_scan

        # goodness-of-fit for gains 
        chisq = np.sum((verr.real * sigma_inv)**2) + np.sum((verr.imag * sigma_inv)**2)
        # prior on the gains
        chisq_g = np.sum((np.log(np.abs(g))**2 / gain_tol**2))

        return chisq + chisq_g

    # use gradient descent to find the gains
    optdict = {'maxiter': 5000} # minimizer params
    res = opt.minimize(errfunc, gpar_guess, method='Powell', options=optdict)

    # save the solution
    g_fit = res.x.view(np.complex128)

    if show_solution == True:
        print (np.abs(g_fit))

    if method=="phase":
        g_fit = g_fit / np.abs(g_fit)
    if method=="amp":
        g_fit = np.abs(np.real(g_fit))

    g_fit = np.append(g_fit, 1.)

    if caltable: # derive a calibration table
        allsites = list(set(scan['t1']).union(set(scan['t2'])))

        caldict = {}
        for site in allsites:
            if site in sites:
                site_key = tkey[site]
            else:
                site_key = -1
            caldict[site] = np.array((scan['time'][0], g_fit[site_key]**-1, g_fit[site_key]**-1), dtype=DTCAL)

        out = caldict

    else: # apply the solution to the scan
        g1_fit = g_fit[g1_keys]
        g2_fit = g_fit[g2_keys]
        gij_inv = (g1_fit * g2_fit.conj())**(-1)
        scan['vis']  = gij_inv * scan['vis']
        scan['qvis'] = gij_inv * scan['qvis']
        scan['uvis'] = gij_inv * scan['uvis']
        scan['vvis'] = gij_inv * scan['vvis']
        scan['sigma']  = np.abs(gij_inv) * scan['sigma']
        scan['qsigma'] = np.abs(gij_inv) * scan['qsigma']
        scan['usigma'] = np.abs(gij_inv) * scan['usigma']
        scan['vsigma'] = np.abs(gij_inv) * scan['vsigma']
        out = scan

    return out

def init(x):
    global counter
    counter = x

def get_selfcal_scan_cal(args):
    return get_selfcal_scan_cal2(*args)

def get_selfcal_scan_cal2(i, n, scan, im, V_scan, sites, method, show_solution, pad_amp, gain_tol, caltable):
    if n > 1:
        global counter
        counter.increment()
        cal_prog_msg(counter.value(), counter.maxval)

    return self_cal_scan(scan, im, V_scan=V_scan, sites=sites, method=method, show_solution=show_solution, 
                         pad_amp=pad_amp, gain_tol=gain_tol, caltable=caltable)


def cal_prog_msg(nscan, totscans):
    complete_percent = int(100*float(nscan)/float(totscans))
    sys.stdout.write('\rCalibrating Scan %i/%i : %i%% done . . .' % (nscan, totscans, complete_percent))
    sys.stdout.flush()

# counter object for sharing among multiprocessing jobs
class Counter(object):
    def __init__(self,initval=0,maxval=0):
        self.val = Value('i',initval)
        self.maxval = maxval
        self.lock = Lock()
    def increment(self):
        with self.lock:
            self.val.value += 1
    def value(self):
        with self.lock:
            return self.val.value











