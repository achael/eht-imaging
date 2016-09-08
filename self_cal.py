import vlbi_imaging_utils as vb
import numpy as np
import scipy.optimize as opt

def self_cal(obs, im):
    """Self-calibrate a dataset to a fixed image""" 
    # V = model visibility, V' = measured visibility, G_i = site gain
    # G_i * conj(G_j) * V_ij = V'_ij

    scans = obs.tlist()

    data_cal = []
    for scan in scans:
        scan_cal = self_cal_scan(scan, im)

        if len(data_cal):
            data_cal = np.hstack((data_cal, scan_cal))
        else:
            data_cal = scan_cal
    
    obs_cal = vb.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, data_cal, obs.tarr, source=obs.source, 
                                           mjd=obs.mjd, ampcal=obs.ampcal, phasecal=obs.phasecal)
    return obs_cal

def self_cal_scan(scan, im):
    """Self-calibrate a scan"""

    # calculating model visibs
    uv = np.hstack((scan['u'].reshape(-1,1), scan['v'].reshape(-1,1)))
    A = vb.ftmatrix(im.psize, im.xdim, im.ydim, uv, pulse=im.pulse)
    V = np.dot(A, im.imvec)

    sites = list(set(scan['t1']).union(set(scan['t2'])))
    
    tkey = {b:a for a,b in enumerate(sites)}

    def errfunc(gpar):
        # ginv = np.sum(gpar.reshape((-1, 2)) * np.array((1, 1j))[np.newaxis,:], axis=-1)
        ginv = gpar.astype(np.double).view(dtype=np.complex128)
        Vpred = np.array([row['vis']*ginv[tkey[row['t1']]]*ginv[tkey[row['t2']]].conj() for row in scan])
        chisq = np.sum(np.abs((Vpred - V)/scan['sigma'])**2)
        return chisq

    gpar_guess = np.ones(len(sites), dtype=np.complex128).view(dtype=np.double)

    # optdict = {'maxiter':maxit, 'ftol':stop, 'maxcor':NHIST} # minimizer params
    res = opt.minimize(errfunc, gpar_guess, method='Nelder-Mead')
    ginv_fit = res.x.view(np.complex128)

    Vpred = np.array([row['vis']*ginv_fit[tkey[row['t1']]]*ginv_fit[tkey[row['t2']]].conj() for row in scan])

    scan['vis'] = Vpred
    scan['sigma'] =  np.array([row['sigma']*np.abs(ginv_fit[tkey[row['t1']]]*ginv_fit[tkey[row['t2']]].conj()) for row in scan])
    return scan
