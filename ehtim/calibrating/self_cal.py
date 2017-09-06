from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.optimize as opt
import sys

import ehtim.obsdata
from ehtim.observing.obs_helpers import *

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

    obs_cal = ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, data_norm, obs.tarr, source=obs.source,
                                    mjd=obs.mjd, ampcal=obs.ampcal, phasecal=obs.phasecal, dcal=obs.dcal, frcal=obs.frcal)
    return obs_cal


def self_cal(obs, im, method="both", show_solution=False):
    """Self-calibrate a dataset to a fixed image.
    """
    # V = model visibility, V' = measured visibility, G_i = site gain
    # G_i * conj(G_j) * V_ij = V'_ij

    scans = obs.tlist()
    n = len(scans)
    data_cal = []
    i = 0
    for scan in scans:
        i += 1
        if not show_solution:
            sys.stdout.write('\rCalibrating Scan %i/%i...' % (i,n))
            sys.stdout.flush()
        scan_cal = self_cal_scan(scan, im, method=method, show_solution=show_solution)

        if len(data_cal):
            data_cal = np.hstack((data_cal, scan_cal))
        else:
            data_cal = scan_cal

    obs_cal = ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, data_cal, obs.tarr, source=obs.source,
                                    mjd=obs.mjd, ampcal=obs.ampcal, phasecal=obs.phasecal, dcal=obs.dcal, frcal=obs.frcal)
    return obs_cal

def self_cal_scan(scan, im, method="both", show_solution=False):
    """Self-calibrate a scan to a fixed  image.
    """

    # calculating image true visibs (beware no scattering here..)
    uv = np.hstack((scan['u'].reshape(-1,1), scan['v'].reshape(-1,1)))
    A = ftmatrix(im.psize, im.xdim, im.ydim, uv, pulse=im.pulse)
    V = np.dot(A, im.imvec)

    sites = list(set(scan['t1']).union(set(scan['t2'])))

    tkey = {b:a for a,b in enumerate(sites)}
    tidx1 = [tkey[row['t1']] for row in scan]
    tidx2 = [tkey[row['t2']] for row in scan]
    sigma_inv = 1.0/scan['sigma']

    def errfunc(gpar):
        g = gpar.astype(np.float64).view(dtype=np.complex128) # all the forward site gains (complex)
        if method=="phase":
            g = g/np.abs(g) # TODO: use np.sign()?
        if method=="amp":
            g = np.abs(g)

        Verr = scan['vis'] - g[tidx1]*g[tidx2].conj() * V
        chisq = np.sum((Verr.real * sigma_inv)**2) + np.sum((Verr.imag * sigma_inv)**2)
        return chisq

    gpar_guess = np.ones(len(sites), dtype=np.complex128).view(dtype=np.float64)

    optdict = {'maxiter':1000} # minimizer params
    res = opt.minimize(errfunc, gpar_guess, method='Powell',options=optdict)
    g_fit = res.x.view(np.complex128)

    if method=="phase":
        g_fit = g_fit/np.abs(g_fit) # TODO: use np.sign()?
    if method=="amp":
        g_fit = np.abs(g_fit)

    gij_inv = (g_fit[tidx1] * g_fit[tidx2].conj())**(-1)

    if show_solution == True:
        print(np.abs(gij_inv))

    scan['vis'] = gij_inv * scan['vis']
    scan['qvis'] = gij_inv * scan['qvis']
    scan['uvis'] = gij_inv * scan['uvis']
    scan['sigma'] = np.abs(gij_inv) * scan['sigma']

    return scan
