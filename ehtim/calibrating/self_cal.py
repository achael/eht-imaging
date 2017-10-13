from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.special as spec
import scipy.optimize as opt
import sys
import itertools as it

import ehtim.obsdata
from ehtim.observing.obs_helpers import *
import ehtim.imaging.imager_utils as iu

from multiprocessing import cpu_count
from multiprocessing import Pool

import itertools

ZBLCUTOFF = 1.e7;

###################################################################################################################################
#Network-Calibration
###################################################################################################################################
def network_cal(obs, zbl, sites=[], zbl_uvdist_max=ZBLCUTOFF, method="both", show_solution=False, pad_amp=0.,gain_tol=.2, processes=-1, caltable=False):
    """Network-calibrate a dataset with zbl constraints
    """
    # V = model visibility, V' = measured visibility, G_i = site gain
    # G_i * conj(G_j) * V_ij = V'_ij
    if len(sites) < 2:
        print("less than 2 stations specified in network cal: defaulting to calibrating all stations!")
        sites = obs.tarr['site']

    # Make the pool for parallel processing
    if processes > 0:
        print("Using Multiprocessing")
        pool = Pool(processes=processes)
    elif processes == 0:
        processes = int(cpu_count())
        print("Using Multiprocessing with %d Processes" % processes)
        pool = Pool(processes=processes)
    else:
        print("Not Using Multiprocessing")

    # find colocated sites and put into list allclusters
    cluster_data = make_cluster_data(obs, zbl_uvdist_max)

    # loop over scans and calibrate
    scans     = obs.tlist()
    scans_cal = scans.copy()

    if processes > 0:
        scans_cal = np.array(pool.map(get_scan_cal, [[i, len(scans), scans[i], zbl, sites, cluster_data, method, pad_amp, gain_tol, caltable, show_solution] for i in range(len(scans))]))
        print('DONE')
    else:
        for i in range(len(scans)):
            sys.stdout.write('\rCalibrating Scan %i/%i...' % (i,len(scans)))
            sys.stdout.flush()
            scans_cal[i] = network_cal_scan(scans[i], zbl, sites, cluster_data,
                                            method=method, show_solution=show_solution, caltable=caltable,
                                            pad_amp=pad_amp, gain_tol=gain_tol)

    if caltable:
        allsites = obs.tarr['site']
        caldict = scans_cal[0]
        for i in range(1,len(scans_cal)):
            row = scans_cal[i]
            if len(row) == 0: 
                continue 

            for site in allsites:
                try: dat = row[site]
                except KeyError: continue

                try: caldict[site] = np.append(caldict[site], row[site])
                except KeyError: caldict[site] = dat

        caltable = ehtim.caltable.Caltable(obs.ra, obs.dec, obs.rf, obs.bw, caldict, obs.tarr,
                                           source = obs.source, mjd=obs.mjd, timetype=obs.timetype)
        out = caltable
    else:
        obs_cal = ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw,
                                        np.concatenate(scans_cal), obs.tarr, source=obs.source, mjd=obs.mjd,
                                        ampcal=obs.ampcal, phasecal=obs.phasecal, dcal=obs.dcal, frcal=obs.frcal,
                                        timetype=obs.timetype)
        out = obs_cal
    return out

def network_cal_scan(scan, zbl, sites, clustered_sites, zbl_uvidst_max=ZBLCUTOFF, method="both", show_solution=False, pad_amp=0., gain_tol=.2, caltable=False):
    """Network-calibrate a scan with zbl constraints
    """
    if len(sites) < 2:
        print("less than 2 stations specified in network cal: defaulting to calibrating all !")
        sites = list(set(np.hstack((scan['t1'], scan['t2']))))

    #sites = list(set(scan['t1']).union(set(scan['t2'])))

    # clustered site information
    allclusters = clustered_sites[0]
    clusterdict = clustered_sites[1]
    clusterbls = clustered_sites[2]

    # create a dictionary to keep track of gains
    tkey = {b:a for a,b in enumerate(sites)}
    clusterkey = clusterdict

    # make a list of gain keys that relates scan bl gains to solved site ones
    # -1 means that this station does not have a gain that is being solved for

    # and make a list of scan keys that relates scan bl visibilities to solved cluster ones
    # -1 means it's a zero baseline!

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

        clusternum1 = clusterkey[row['t1']]
        clusternum2 = clusterkey[row['t2']]
        if clusternum1 == clusternum2: # sites are in the same cluster
            scan_keys.append(-1)
        else: #sites are not in the same cluster
            bl_index = clusterbls.index(set((clusternum1, clusternum2)))
            scan_keys.append(bl_index)


    # no sites to calibrate on this scan!
    if np.all(g1_keys == -1):
        return scan

    # scan visibilities and sigmas with extra padding
    vis = scan['vis']
    sigma_inv = 1.0/np.sqrt(scan['sigma']**2+ (pad_amp*np.abs(scan['vis']))**2)

    # initial guesses for parameters
    n_gains = len(sites)
    n_clusterbls = len(clusterbls)
    gpar_guess = np.ones(n_gains, dtype=np.complex128).view(dtype=np.float64)

    vpar_guess = np.ones(n_clusterbls, dtype=np.complex128)
    for i in range(len(scan_keys)):
        if scan_keys[i] < 0: continue
        vpar_guess[scan_keys[i]] = vis[i]

    vpar_guess = vpar_guess.view(dtype=np.float64)

    gvpar_guess = np.hstack((gpar_guess, vpar_guess))


    # error function
    def errfunc(gvpar):

        # all the forward site gains (complex)
        g = gvpar[0:2*n_gains].astype(np.float64).view(dtype=np.complex128)
        # all the intercluster visibilities (complex)
        v = gvpar[2*n_gains:].astype(np.float64).view(dtype=np.complex128)

        # choose to only scale ampliltudes or phases
        if method=="phase":
            g = g/np.abs(g) # TODO: use exp(i*np.arg())?
        if method=="amp":
             g = np.abs(np.real(g))
            #g = np.abs(g)

        # append the default values to g for missing points
        # and to v for the zero baseline points
        g = np.append(g, 1.)
        v = np.append(v, zbl)

        # scan visibilities are either an intercluster visibility or the fixed zbl

        v_scan = v[scan_keys]
        g1 = g[g1_keys]
        g2 = g[g2_keys]

        #TODO DEBIAS
        if method=='amp':
            verr = np.abs(vis) - g1*g2.conj() * np.abs(v_scan)
        else:
            verr = vis - g1*g2.conj() * v_scan

        chisq = np.sum((verr.real * sigma_inv)**2) + np.sum((verr.imag * sigma_inv)**2)

        g_fracerr = gain_tol #0.3
        sharpness = 5.
        chisq_g = np.sum((np.log(np.abs(g))**2 / g_fracerr**2))
        chisq_v = np.sum((np.abs(v)/zbl)**4)
        #chisq_v = np.sum(-np.log(spec.expit(sharpness*(1-np.abs(v)/zbl))))

        return chisq + chisq_g + chisq_v

    #print ("errfunc init: ", errfunc(gvpar_guess))
    optdict = {'maxiter' : 500} # minimizer params
    res = opt.minimize(errfunc, gvpar_guess, method='CG', options=optdict)

    # get solution
    #print ("errfunc end: " ,errfunc(res.x))
    g_fit = res.x[0:2*n_gains].view(np.complex128)
    v_fit = res.x[2*n_gains:].view(np.complex128)

    if method=="phase":
        g_fit = g_fit / np.abs(g_fit)
    if method=="amp":
        g_fit = np.abs(np.real(g_fit))

    if show_solution == True:
        print (np.abs(g_fit))
        print (np.abs(v_fit))


    g_fit = np.append(g_fit, 1.)
    v_fit = np.append(v_fit, zbl)

    if caltable:
        allsites = list(set(scan['t1']).union(set(scan['t2']))) if g_fit[-2] != 1.0 else []

        caldict = {}
        for site in allsites:
            if site in sites:
                site_key = tkey[site]
            else:
                site_key = -1
            caldict[site] = np.array((scan['time'][0], g_fit[site_key]**-1, g_fit[site_key]**-1), dtype=DTCAL)
        out = caldict

    else:
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


def get_scan_cal(args):
    return get_scan_cal2(*args)

def get_scan_cal2(i, n, scan, zbl, sites, cluster_data, method, pad_amp,gain_tol,caltable, show_solution):
    if n > 1:
        sys.stdout.write('.')
        sys.stdout.flush()

    return network_cal_scan(scan, zbl, sites, cluster_data, zbl_uvidst_max=ZBLCUTOFF, method=method,caltable=caltable, show_solution=show_solution, pad_amp=pad_amp, gain_tol=gain_tol)


###################################################################################################################################
#Self-Calibration
###################################################################################################################################
def self_cal(obs, im, sites=[], method="both", show_solution=False, pad_amp=0., ttype='direct', fft_pad_frac=2, gain_tol=.2, caltable=False):
    """Self-calibrate a dataset to a fixed image.
    """
    # V = model visibility, V' = measured visibility, G_i = site gain
    # G_i * conj(G_j) * V_ij = V'_ij
    if len(sites) < 2:
        print("less than 2 stations specified in self cal: defaulting to calibrating all stations!")
        sites = obs.tarr['site']

    # First, sample the model visibilities
    if ttype == 'direct':
        data_arr = obs.unpack(['u','v','vis','sigma'])
        uv = np.hstack((data_arr['u'].reshape(-1,1), data_arr['v'].reshape(-1,1)))
        A = ftmatrix(im.psize, im.xdim, im.ydim, uv, pulse=im.pulse)
        V = np.dot(A, im.imvec)
    else:
        (data, sigma, fft_A) = iu.chisqdata_vis_fft(obs, im, fft_pad_frac=fft_pad_frac)
        im_info, sampler_info_list, gridder_info_list = fft_A
        vis_arr = iu.fft_imvec(im.imvec, im_info)
        V = iu.sampler(vis_arr, sampler_info_list, sample_type='vis')

    scans = obs.tlist()
    n = len(scans)
    data_cal = []
    i = 0
    data_index = 0
    for scan in scans:
        i += 1
        if not show_solution:
            sys.stdout.write('\rCalibrating Scan %i/%i...' % (i,n))
            sys.stdout.flush()

        scan_cal = self_cal_scan(scan, im, V_scan=V[data_index:(data_index+len(scan))], sites=sites,
                                 method=method, show_solution=show_solution, caltable=caltable,
                                 pad_amp=pad_amp, gain_tol=gain_tol)
        data_index += len(scan)
        data_cal.append(scan_cal)

    if caltable:
        allsites = obs.tarr['site']
        caldict = data_cal[0]
        for i in range(1,len(data_cal)):
            row = data_cal[i]
            for site in allsites:
                try: dat = row[site]
                except KeyError: continue

                try: caldict[site] = np.append(caldict[site], row[site])
                except KeyError: caldict[site] = dat

        caltable = ehtim.caltable.Caltable(obs.ra, obs.dec, obs.rf, obs.bw, caldict, obs.tarr,
                                           source = obs.source, mjd=obs.mjd, timetype=obs.timetype)
        out = caltable
    else:
        obs_cal = ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw,
                                        np.concatenate(data_cal), obs.tarr, source=obs.source, mjd=obs.mjd,
                                        ampcal=obs.ampcal, phasecal=obs.phasecal, dcal=obs.dcal, frcal=obs.frcal,
                                        timetype=obs.timetype)

        out = obs_cal
    return out

def self_cal_scan(scan, im, V_scan=[], sites=[], method="both", show_solution=False, pad_amp=0., gain_tol=.2, caltable=False):
    """Self-calibrate a scan to a fixed  image.
       if caltable==True, returns a caltable dictionary. If False, returns the  scan.
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

    gpar_guess = np.ones(len(sites), dtype=np.complex128).view(dtype=np.float64)

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

        #TODO DEBIAS
        if method=='amp':
            verr = np.abs(scan['vis']) - g1*g2.conj() * np.abs(V_scan)
        else:
            verr = scan['vis'] - g1*g2.conj() * V_scan

        chisq = np.sum((verr.real * sigma_inv)**2) + np.sum((verr.imag * sigma_inv)**2)
        chisq_g = np.sum((np.log(np.abs(g))**2 / gain_tol**2))

        return chisq

    optdict = {'maxiter': 5000} # minimizer params
    res = opt.minimize(errfunc, gpar_guess, method='Powell', options=optdict)
    g_fit = res.x.view(np.complex128)

    if show_solution == True:
        print (np.abs(g_fit))

    if method=="phase":
        g_fit = g_fit / np.abs(g_fit)
    if method=="amp":
        g_fit = np.abs(np.real(g_fit))

    g_fit = np.append(g_fit, 1.)

    if caltable:
        allsites = list(set(scan['t1']).union(set(scan['t2'])))

        caldict = {}
        for site in allsites:
            if site in sites:
                site_key = tkey[site]
            else:
                site_key = -1
            caldict[site] = np.array((scan['time'][0], g_fit[site_key]**-1, g_fit[site_key]**-1), dtype=DTCAL)

        out = caldict

    else:
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


###################################################################################################################################
#Misc
###################################################################################################################################

def make_cluster_data(obs, zbl_uvdist_max=ZBLCUTOFF):
    clusters = []
    clustered_sites = []
    for i1 in range(len(obs.tarr)):
        t1 = obs.tarr[i1]

        if t1['site'] in clustered_sites:
            continue

        csites = [t1['site']]
        clustered_sites.append(t1['site'])
        for i2 in range(len(obs.tarr))[i1:]:
            t2 = obs.tarr[i2]
            if t2['site'] in clustered_sites:
                continue

            site1coord = np.array([t1['x'], t1['y'], t1['z']])
            site2coord = np.array([t2['x'], t2['y'], t2['z']])
            uvdist = np.sqrt(np.sum((site1coord-site2coord)**2)) / (C / obs.rf)

            if uvdist < zbl_uvdist_max:
                csites.append(t2['site'])
                clustered_sites.append(t2['site'])
        clusters.append(csites)

    clusterdict = {}
    for site in obs.tarr['site']:
        for k in range(len(clusters)):
            if site in  clusters[k]:
                clusterdict[site] = k

    clusterbls = [set(comb) for comb in it.combinations(range(len(clusterdict)),2)]

    cluster_data = (clusters, clusterdict, clusterbls)

    return cluster_data

def norm_zbl(obs, flux=1.):
    """Normalize scans to zero baseline
    """
    # V = model visibility, V' = measured visibility, G_i = site gain
    # G_i * conj(G_j) * V_ij = V'_ij

    scans = obs.tlist()
    n = len(scans)
    data_norm = []
    i = 0
    for scan in scans:
        i += 1
        uvdist = np.sqrt(scan['u']**2 + scan['v']**2)
        #print(np.min(uvdist)/1.e4)
        scan_zbl = np.abs(scan['vis'][np.argmin(uvdist)])

        scan['vis'] = scan['vis']/scan_zbl
        scan['sigma'] = scan['sigma']/scan_zbl
        scan['qvis'] = scan['qvis']/scan_zbl
        scan['qsigma'] = scan['qsigma']/scan_zbl
        scan['uvis'] = scan['uvis']/scan_zbl
        scan['usigma'] = scan['usigma']/scan_zbl
        scan['vvis'] = scan['vvis']/scan_zbl
        scan['vsigma'] = scan['vsigma']/scan_zbl

        if len(data_norm):
            data_norm = np.hstack((data_norm, scan))
        else:
            data_norm = scan

    obs_cal = ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, data_norm, obs.tarr,
                                    source=obs.source, mjd=obs.mjd, timetype=obs.timetype,
                                    ampcal=obs.ampcal, phasecal=obs.phasecal, dcal=obs.dcal, frcal=obs.frcal)
    return obs_cal

