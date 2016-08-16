# maxen.py
# Andrew Chael, 10/15/2015
# Maximum Entropy imagers for VLBI data

import sys
import time
import numpy as np
import scipy.optimize as opt
import scipy.ndimage.filters as filt
import scipy.signal
import matplotlib.pyplot as plt
import itertools as it
import vlbi_imaging_utils as vb
import pulses
import linearize_energy as le
from IPython import display

##################################################################################################
# Constants
##################################################################################################
C = 299792458.0 
DEGREE = np.pi/180.
RADPERAS = DEGREE/3600.
RADPERUAS = RADPERAS/1e6
NHIST = 5000 # number of steps to store for hessian approx
nit = 0 # global variable to track the iteration number in the plotting callback

##################################################################################################
# Imagers
##################################################################################################
def maxen(Obsdata, InitIm, Prior, maxit=100, alpha=1e5, entropy="gs", stop=1e-10, ipynb=False):
    """Run maximum entropy with full amplitudes and phases. 
       Uses I = exp(I') change of variables.
       Obsdata is an Obsdata object, and Prior is an Image object.
       Returns Image object.
       The lagrange multiplier alpha is not a free parameter
    """
    
    print "Imaging I with visibility amplitudes and phases . . ."
        
    # Catch problem if uvrange < largest baseline
    uvrange = 1/Prior.psize
    maxbl = np.max(Obsdata.unpack(['uvdist'])['uvdist'])
    if uvrange < maxbl:
        raise Exception("pixel spacing is larger than smallest spatial wavelength!")
    if (Prior.psize != InitIm.psize) or (Prior.xdim != InitIm.xdim) or (Prior.ydim != InitIm.ydim):
        raise Exception("initial image does not match dimensions of the prior image!")

    # Normalize prior image to total flux
    zbl = np.max(Obsdata.unpack(['amp'])['amp'])
    nprior = zbl * Prior.imvec / np.sum(Prior.imvec)
    ninit = zbl * InitIm.imvec / np.sum(InitIm.imvec)
    logprior = np.log(nprior)
    loginit = np.log(ninit)
    
    # Data
    data = Obsdata.unpack(['u','v','vis','sigma'])
    uv = np.hstack((data['u'].reshape(-1,1), data['v'].reshape(-1,1)))
    vis = data['vis']
    sigma = data['sigma']
    
    # Compute the Fourier matrix
    A = vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv, pulse=Prior.pulse)
    
    # Define the objective function and gradient
    def objfunc(logim):
        im = np.exp(logim)
        if entropy == "simple":
            s = -ssimple(im, nprior)
        elif entropy == "l1":
            s = -sl1(im, nprior)
        elif entropy == "gs":
            s = -sgs(im, nprior)
        elif entropy == "tv":
            s = -stv(im, Prior.xdim, Prior.ydim)
        elif entropy == "patch":
            s = -spatch(im, nprior)
             
        return s + alpha * (chisq(im, A, vis, sigma) - 1)
        
    def objgrad(logim):
        im = np.exp(logim)
        if entropy == "simple":
            s = -ssimplegrad(im, nprior)
        elif entropy == "l1":
            s = -sl1grad(im, nprior)
        elif entropy == "gs":
            s = -sgsgrad(im, nprior) 
        elif entropy == "tv":
            s = -stvgrad(im, Prior.xdim, Prior.ydim)
        elif entropy == "patch":
            s = -spatchgrad(im, nprior)
            
        return  (s + alpha * chisqgrad(im, A, vis, sigma)) * im
    
    # Plotting function for each iteration
    global nit
    nit = 0
    def plotcur(logim_step):
        global nit
        im_step = np.exp(logim_step)
        chi2 = chisq(im_step, A, vis, sigma)
        plot_i(im_step, Prior, nit, chi2, ipynb=ipynb)
        nit += 1
   
    # Plot the prior

    plotcur(loginit)
        
    # Minimize
    optdict = {'maxiter':maxit, 'ftol':stop, 'maxcor':NHIST} # minimizer params
    tstart = time.time()
    res = opt.minimize(objfunc, loginit, method='L-BFGS-B', jac=objgrad, 
                       options=optdict, callback=plotcur)
    tstop = time.time()
    out = np.exp(res.x)
    
    # Print stats
    print "time: %f s" % (tstop - tstart)
    print "J: %f" % res.fun
    print "Chi^2: %f" % chisq(out, A, vis, sigma)
    print res.message
    
    # Return Image object
    outim = vb.Image(out.reshape(Prior.ydim, Prior.xdim), Prior.psize, Prior.ra, Prior.dec, rf=Prior.rf, source=Prior.source, mjd=Prior.mjd, pulse=Prior.pulse)
    if len(Prior.qvec):
        print "Preserving image complex polarization fractions!"
        qvec = Prior.qvec * out / Prior.imvec
        uvec = Prior.uvec * out / Prior.imvec
        outim.add_qu(qvec.reshape(Prior.ydim, Prior.xdim), uvec.reshape(Prior.ydim, Prior.xdim))
    return outim

def maxen_bs(Obsdata, InitIm, Prior, flux, maxit=100, alpha=100, gamma=500, delta=500, beta=1.0, entropy="gs", datamin="gd", stop=1e-10, ipynb=False):
    """Run maximum entropy on the bispectrum with an exponential change of variables 
       Obsdata is an Obsdata object, and Prior is an Image object.
       Returns Image object.
       "Lagrange multipliers" are not free parameters.
    """
    
    print "Imaging I with bispectrum . . ." 
    
    # Catch problem if uvrange < largest baseline
    uvrange = 1/Prior.psize
    maxbl = np.max(Obsdata.unpack(['uvdist'])['uvdist'])
    if uvrange < maxbl:
        raise Exception("Pixel spacing is larger than smallest spatial wavelength!")
    if (Prior.psize != InitIm.psize) or (Prior.xdim != InitIm.xdim) or (Prior.ydim != InitIm.ydim):
        raise Exception("initial image does not match dimensions of the prior image!")
        
    # Normalize prior image to total flux  TODO: DO WE STILL NEED THIS?
    nprior = flux * Prior.imvec / np.sum(Prior.imvec)
    ninit = flux * InitIm.imvec / np.sum(InitIm.imvec)
    logprior = np.log(nprior)
    loginit = np.log(ninit)
    
    # Get bispectra data    
    biarr = Obsdata.bispectra(mode="all", count="min")
    uv1 = np.hstack((biarr['u1'].reshape(-1,1), biarr['v1'].reshape(-1,1)))
    uv2 = np.hstack((biarr['u2'].reshape(-1,1), biarr['v2'].reshape(-1,1)))
    uv3 = np.hstack((biarr['u3'].reshape(-1,1), biarr['v3'].reshape(-1,1)))
    bi = biarr['bispec']
    sigs = biarr['sigmab']
    #sigs_2 = scaled_bisigs(Obsdata) # Katie's correction for overcounting number of DOF
    
    # Compute the fourier matrices
    A3 = (vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse),
          vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse),
          vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse)
         )
    
    # Coordinate matrix for COM constraint
    coord = Prior.psize * np.array([[[x,y] for x in np.arange(Prior.xdim/2,-Prior.xdim/2,-1)]
                                           for y in np.arange(Prior.ydim/2,-Prior.ydim/2,-1)])
    coord = coord.reshape(Prior.ydim*Prior.xdim, 2)        
    
    # Katie - if you are using the linearized energy then compute the A and b of your 
    # linearized equation given your current image
    if datamin=="lin":
        (Alin, blin) = le.computeLinTerms_bi(ninit, A3, bi, sigs, InitIm.xdim*InitIm.ydim, alpha=alpha)                           
    
    # Define the objective function and gradient
    def objfunc(logim):
        im = np.exp(logim)
        if entropy == "simple":
            s = -ssimple(im, nprior)
        elif entropy == "l1":
            s = -sl1(im, nprior)
        elif entropy == "gs":
            s = -sgs(im, nprior)
        elif entropy == "tv":
            s = -stv(im, Prior.xdim, Prior.ydim)
        elif entropy == "patch":
            s = -spatch(im, nprior)
            
        if datamin == "gd":
            c = alpha * (chisq_bi(im, A3, bi, sigs) - 1)
        elif datamin== "lin":
            c = alpha * (chisq_bi(im, A3, bi, sigs) - 1)
        
        s = s * beta
        t = gamma * (np.sum(im) - flux)**2
        cm = delta * (np.sum(im * coord[:,0]) + np.sum(im * coord[:,1]))**2
        return  s + c + t + cm
        
    def objgrad(logim):
        im = np.exp(logim)
        if entropy == "simple":
            s = -ssimplegrad(im, nprior)
        elif entropy == "l1":
            s = -sl1grad(im, nprior)
        elif entropy == "gs":
            s = -sgsgrad(im, nprior)
        elif entropy == "tv":
            s = -stvgrad(im, Prior.xdim, Prior.ydim)
        elif entropy == "patch":
            s = -spatchgrad(im, nprior)
            
        if datamin == "gd":
            c = alpha * chisqgrad_bi(im, A3, bi, sigs)
        elif datamin== "lin":
            c = 2.0/(2.0*len(sigs)) * np.dot(Alin.T, np.dot(Alin, im) - blin) #negative here? 944202941822.00635
        
        s = s * beta
        t = 2 * gamma * (np.sum(im) - flux)
        cm = 2 * delta * (np.sum(im * coord[:,0]) + np.sum(im * coord[:,1])) * (coord[:,0] + coord[:,1])
        return  (s + c + t + cm) * im
    
    # Plotting function for each iteration
    global nit
    nit = 0
    def plotcur(logim_step):
        global nit
        im_step = np.exp(logim_step)
        chi2 = chisq_bi(im_step, A3, bi, sigs)
        plot_i(im_step, Prior, nit, chi2, ipynb=ipynb)
        nit += 1
   
    plotcur(loginit)
       
    # Minimize    
    optdict = {'maxiter':maxit, 'ftol':stop, 'maxcor':NHIST}
    tstart = time.time()
    res = opt.minimize(objfunc, loginit, method='L-BFGS-B', jac=objgrad, 
                       options=optdict, callback=plotcur)
    tstop = time.time()
    out = np.exp(res.x)
    
    # Print stats
    print "time: %f s" % (tstop - tstart)
    print "J: %f" % res.fun
    print "Chi^2: %f" % chisq_bi(out, A3, bi, sigs)
    print res.message
    
    # Return Image object
    outim = vb.Image(out.reshape(Prior.ydim, Prior.xdim), Prior.psize, Prior.ra, Prior.dec, rf=Prior.rf, source=Prior.source, mjd=Prior.mjd, pulse=Prior.pulse)
    if len(Prior.qvec):
        print "Preserving complex polarization fractions"
        qvec = Prior.qvec * out / Prior.imvec
        uvec = Prior.uvec * out / Prior.imvec
        outim.add_qu(qvec.reshape(Prior.ydim, Prior.xdim), uvec.reshape(Prior.ydim, Prior.xdim))
    return outim 
    
def maxen_amp_cphase(Obsdata, InitIm, Prior, flux = 1.0, maxit=100, alpha_clphase=100, alpha_visamp=100, gamma=500, delta=500, entropy="gs", stop=1e-5, grads=True, ipynb=False):
    """Run maximum entropy on visibility amplitude and closure phase with an exponential change of variables 
       Obsdata is an Obsdata object, and Prior is an Image object.
       Returns Image object.
       "Lagrange multipliers" are not free parameters.
    """
    
    print "Imaging I with visibility amplitudes and closure phases . . ." 
   
    # Catch problem if uvrange < largest baseline
    uvrange = 1/Prior.psize
    maxbl = np.max(Obsdata.unpack(['uvdist'])['uvdist'])
    if uvrange < maxbl:
        raise Exception("pixel spacing is larger than smallest spatial wavelength!")
    if (Prior.psize != InitIm.psize) or (Prior.xdim != InitIm.xdim) or (Prior.ydim != InitIm.ydim):
        raise Exception("initial image does not match dimensions of the prior image!")

    # Normalize prior image to total flux (this doesn't actually matter because total flux is unconstrained)
    nprior = flux * Prior.imvec / np.sum(Prior.imvec)
    ninit = flux * InitIm.imvec / np.sum(InitIm.imvec)
    logprior = np.log(nprior)
    loginit = np.log(ninit)

    # Get closure phase data
    clphasearr = Obsdata.c_phases(mode="all", count="min")
    uv1 = np.hstack((clphasearr['u1'].reshape(-1,1), clphasearr['v1'].reshape(-1,1)))
    uv2 = np.hstack((clphasearr['u2'].reshape(-1,1), clphasearr['v2'].reshape(-1,1)))
    uv3 = np.hstack((clphasearr['u3'].reshape(-1,1), clphasearr['v3'].reshape(-1,1)))
    clphase = clphasearr['cphase']
    sigs_clphase = clphasearr['sigmacp']
    
    
    # Compute the fourier matrices
    A3 = (vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse),
          vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse),
          vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse)
         )
    
    # Get amplitude data
    ampdata = Obsdata.unpack(['u','v','amp','sigma'])
    uv = np.hstack((ampdata['u'].reshape(-1,1), ampdata['v'].reshape(-1,1)))
    amp = ampdata['amp']
    sigs_amp = ampdata['sigma']
    
    A = vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv, pulse=Prior.pulse)

    del clphasearr
    del uv1, uv2, uv3, uv
    
    # Coordinate matrix for COM constraint
    coord = Prior.psize * np.array([[[x,y] for x in np.arange(Prior.xdim/2,-Prior.xdim/2,-1)]
                                           for y in np.arange(Prior.ydim/2,-Prior.ydim/2,-1)])
    coord = coord.reshape(Prior.ydim*Prior.xdim, 2)                                   
    
    # Define the objective function and gradient
    def objfunc(logim):
        im = np.exp(logim)
        if entropy == "simple":
            s = -ssimple(im, nprior)
        elif entropy == "l1":
            s = -sl1(im, nprior)
        elif entropy == "gs":
            s = -sgs(im, nprior)
        elif entropy == "tv":
            s = -stv(im, Prior.xdim, Prior.ydim)
            
        c_clphase = alpha_clphase * (chisq_clphase(im, A3, clphase, sigs_clphase) - 1)
        c_amp   = alpha_visamp   * (chisq_visamp(im, A, amp, sigs_amp) - 1)
        t = gamma * (np.sum(im) - flux)**2
        cm = delta * (np.sum(im * coord[:,0]) + np.sum(im * coord[:,1]))**2
        return  s + c_clphase + c_amp + t + cm
    
    def objgrad(logim):
        im = np.exp(logim)
        if entropy == "simple":
            s = -ssimplegrad(im, nprior)
        elif entropy == "l1":
            s = -sl1grad(im, nprior)
        elif entropy == "gs":
            s = -sgsgrad(im, nprior) 
        elif entropy == "tv":
            s = -stvgrad(im, Prior.xdim, Prior.ydim)
        
        c_clphase = alpha_clphase * chisqgrad_clphase(im, A3, clphase, sigs_clphase)
        c_amp = alpha_visamp * chisqgrad_visamp(im, A, amp, sigs_amp)
        t = 2 * gamma * (np.sum(im) - flux)
        cm = 2 * delta * (np.sum(im * coord[:,0]) + np.sum(im * coord[:,1])) * (coord[:,0] + coord[:,1])
        return  (s + c_clphase + c_amp + t + cm) * im
		       
    # Plotting function for each iteration
    global nit
    nit = 0
    def plotcur(logim_step):
        global nit
        im_step = np.exp(logim_step)
        chi2_clphase = chisq_clphase(im_step, A3, clphase, sigs_clphase)
        chi2_amp   = chisq_visamp(im_step, A, amp, sigs_amp)
	print("chi2_clphase: ",chi2_clphase)
	print("chi2_amp: ",chi2_amp)
        plot_i(im_step, Prior, nit, chi2_amp, ipynb=ipynb)
        nit += 1
   
    plotcur(loginit)
       
    # Minimize
    optdict = {'maxiter':maxit, 'ftol':stop, 'maxcor':NHIST}
    tstart = time.time()
    res = opt.minimize(objfunc, loginit, method='L-BFGS-B', jac=objgrad, 
                       options=optdict, callback=plotcur)

    tstop = time.time()
    out = np.exp(res.x)
    
    # Print stats
    print "time: %f s" % (tstop - tstart)
    print "J: %f" % res.fun
    print "Closure Phase Chi^2: %f" % chisq_clphase(out, A3, clphase, sigs_clphase)
    print "Amplitude Chi^2: %f" % chisq_visamp(out, A, amp, sigs_amp)
    print res.message
    
    # Return Image object
    outim = vb.Image(out.reshape(Prior.ydim, Prior.xdim), Prior.psize, Prior.ra, Prior.dec, rf=Prior.rf, source=Prior.source, mjd=Prior.mjd, pulse=Prior.pulse)
    if len(Prior.qvec):
        print "Preserving complex polarization fractions"
        qvec = Prior.qvec * out / Prior.imvec
        uvec = Prior.uvec * out / Prior.imvec
        outim.add_qu(qvec.reshape(Prior.ydim, Prior.xdim), uvec.reshape(Prior.ydim, Prior.xdim))
    return outim   
    
def maxen_onlyclosure(Obsdata, InitIm, Prior, flux = 1.0, maxit=100, alpha_clphase=100, alpha_clamp=100, gamma=500, delta=500, entropy="gs", stop=1e-5, grads=True, ipynb=False):
    """Run maximum entropy on only closure quantities with an exponential change of variables 
       Obsdata is an Obsdata object, and Prior is an Image object.
       Returns Image object.
       "Lagrange multipliers" are not free parameters.
    """
    
    print "Imaging I with only closure quantities . . ." 
    
    # Note: total flux is completely unconstrained by only using closure quantities

    # Catch problem if uvrange < largest baseline
    uvrange = 1/Prior.psize
    maxbl = np.max(Obsdata.unpack(['uvdist'])['uvdist'])
    if uvrange < maxbl:
        raise Exception("pixel spacing is larger than smallest spatial wavelength!")
    if (Prior.psize != InitIm.psize) or (Prior.xdim != InitIm.xdim) or (Prior.ydim != InitIm.ydim):
        raise Exception("initial image does not match dimensions of the prior image!")

    # Normalize prior image to total flux (this doesn't actually matter because total flux is unconstrained)
    nprior = flux * Prior.imvec / np.sum(Prior.imvec)
    ninit = flux * InitIm.imvec / np.sum(InitIm.imvec)
    logprior = np.log(nprior)
    loginit = np.log(ninit)

    # Get closure phase data
    clphasearr = Obsdata.c_phases(mode="all", count="min")
    uv1 = np.hstack((clphasearr['u1'].reshape(-1,1), clphasearr['v1'].reshape(-1,1)))
    uv2 = np.hstack((clphasearr['u2'].reshape(-1,1), clphasearr['v2'].reshape(-1,1)))
    uv3 = np.hstack((clphasearr['u3'].reshape(-1,1), clphasearr['v3'].reshape(-1,1)))
    clphase = clphasearr['cphase']
    sigs_clphase = clphasearr['sigmacp']
    
    
    # Compute the fourier matrices
    A3 = (vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse),
          vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse),
          vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse)
         )
    
    # Get closure amplitude data
    clamparr = Obsdata.c_amplitudes(mode="all", count="min")
    uv1 = np.hstack((clamparr['u1'].reshape(-1,1), clamparr['v1'].reshape(-1,1)))
    uv2 = np.hstack((clamparr['u2'].reshape(-1,1), clamparr['v2'].reshape(-1,1)))
    uv3 = np.hstack((clamparr['u3'].reshape(-1,1), clamparr['v3'].reshape(-1,1)))
    uv4 = np.hstack((clamparr['u4'].reshape(-1,1), clamparr['v4'].reshape(-1,1)))
    clamp = clamparr['camp']
    sigs_clamp = clamparr['sigmaca']

    A4 = (vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse),
          vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse),
          vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse),
          vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv4, pulse=Prior.pulse)
         )

    del clphasearr
    del clamparr
    del uv1, uv2, uv3, uv4
    
    # Coordinate matrix for COM constraint
    coord = Prior.psize * np.array([[[x,y] for x in np.arange(Prior.xdim/2,-Prior.xdim/2,-1)]
                                           for y in np.arange(Prior.ydim/2,-Prior.ydim/2,-1)])
    coord = coord.reshape(Prior.ydim*Prior.xdim, 2)                                   
    
    # Define the objective function and gradient
    def objfunc(logim):
        im = np.exp(logim)
        if entropy == "simple":
            s = -ssimple(im, nprior)
        elif entropy == "l1":
            s = -sl1(im, nprior)
        elif entropy == "gs":
            s = -sgs(im, nprior)
        elif entropy == "tv":
            s = -stv(im, Prior.xdim, Prior.ydim)
            
        c_clphase = alpha_clphase * (chisq_clphase(im, A3, clphase, sigs_clphase) - 1)
        c_clamp   = alpha_clamp   * (chisq_clamp(im, A4, clamp, sigs_clamp) - 1)
        t = gamma * (np.sum(im) - flux)**2
        cm = delta * (np.sum(im * coord[:,0]) + np.sum(im * coord[:,1]))**2
        return  s + c_clphase + c_clamp + t + cm
    
    def objgrad(logim):
        im = np.exp(logim)
        if entropy == "simple":
            s = -ssimplegrad(im, nprior)
        elif entropy == "l1":
            s = -sl1grad(im, nprior)
        elif entropy == "gs":
            s = -sgsgrad(im, nprior) 
        elif entropy == "tv":
            s = -stvgrad(im, Prior.xdim, Prior.ydim)
        
        c_clphase = alpha_clphase * chisqgrad_clphase(im, A3, clphase, sigs_clphase)
        c_clamp = alpha_clamp * chisqgrad_clamp(im, A4, clamp, sigs_clamp)
        t = 2 * gamma * (np.sum(im) - flux)
        cm = 2 * delta * (np.sum(im * coord[:,0]) + np.sum(im * coord[:,1])) * (coord[:,0] + coord[:,1])
        return  (s + c_clphase + c_clamp + t + cm) * im
    
    def objgrad_num(x):  #This calculates the gradient numerically
		dx = 0.000001* np.mean(x) 
		J0 = objfunc(x)
		Jgrad = np.copy(x)
		for i in range(len(x)):
			xp = np.copy(x)
			xp[i] = xp[i] + dx
			J1 = objfunc(xp)
			Jgrad[i] = (J1-J0)/dx
		return Jgrad
		       
    # Plotting function for each iteration
    global nit
    nit = 0
    def plotcur(logim_step):
        global nit
        im_step = np.exp(logim_step)
        chi2_clphase = chisq_clphase(im_step, A3, clphase, sigs_clphase)
        chi2_clamp   = chisq_clamp(im_step, A4, clamp, sigs_clamp)
	print("chi2_clphase: ",chi2_clphase)
	print("chi2_clamp: ",chi2_clamp)
        plot_i(im_step, Prior, nit, chi2_clamp, ipynb=ipynb)
        nit += 1
   
    plotcur(loginit)
       
    # Minimize
    optdict = {'maxiter':maxit, 'ftol':stop, 'maxcor':NHIST}
    tstart = time.time()
    if grads:
        res = opt.minimize(objfunc, loginit, method='L-BFGS-B', jac=objgrad, 
                       options=optdict, callback=plotcur)
    else:
        res = opt.minimize(objfunc, loginit, method='L-BFGS-B', 
                       options=optdict, callback=plotcur)
    tstop = time.time()
    out = np.exp(res.x)
    
    # Print stats
    print "time: %f s" % (tstop - tstart)
    print "J: %f" % res.fun
    print "Closure Phase Chi^2: %f" % chisq_clphase(out, A3, clphase, sigs_clphase)
    print "Closure Amplitude Chi^2: %f" % chisq_clamp(out, A4, clamp, sigs_clamp)
    print res.message
    
    # Return Image object
    outim = vb.Image(out.reshape(Prior.ydim, Prior.xdim), Prior.psize, Prior.ra, Prior.dec, rf=Prior.rf, source=Prior.source, mjd=Prior.mjd, pulse=Prior.pulse)
    if len(Prior.qvec):
        print "Preserving complex polarization fractions"
        qvec = Prior.qvec * out / Prior.imvec
        uvec = Prior.uvec * out / Prior.imvec
        outim.add_qu(qvec.reshape(Prior.ydim, Prior.xdim), uvec.reshape(Prior.ydim, Prior.xdim))
    return outim   
    
def maxen_p(Obsdata, Prior, maxit=100, beta=1e4, polentropy="hw", stop=1e-500, nvec=15, pcut=0.05, prior=True, ipynb=False):
    """Run maximum entropy on pol. amplitude and phase
       Obsdata is an Obsdata object,
       Prior is an Image object containing the Stokes I image and Q & U priors.
       Returns an Image object.
       Operates on m and chi images (NOT Q and U)
       The lagrange multiplier beta is not a free parameter
    """
    
    print "Imaging Q, U with pol. amplitude and phase . . ."
    
    # Catch problem if uvrange < largest baseline
    uvrange = 1/Prior.psize
    maxbl = np.max(Obsdata.unpack(['uvdist'])['uvdist'])
    if uvrange < maxbl:
        raise Exception("pixel spacing is larger than smallest spatial wavelength!")
    
    # Set up priors
    iimage = Prior.imvec
    if len(Prior.qvec):
        mprior = np.abs(Prior.qvec + 1j*Prior.uvec) / Prior.imvec
        chiprior = np.arctan2(Prior.uvec, Prior.qvec) / 2.0
        allprior = np.hstack((mprior, chiprior))
    else:
        # !AC get the actual zero baseline pol. frac from the data!
        mprior = 0.2 * (np.ones(len(iimage)) + 1e-10 * np.random.rand(len(iimage)))
        chiprior = np.zeros(len(iimage)) + 1e-10 * np.random.rand(len(iimage))
        allprior = np.hstack((mprior, chiprior))
    
    # Change of variables    
    allprior = mcv_r(allprior)
    
    # Data
    data = Obsdata.unpack(['u','v','pvis','sigma','m'])
    uv = np.hstack((data['u'].reshape(-1,1), data['v'].reshape(-1,1)))
    p = data['pvis']
    m = data['m']

    sigmap = np.sqrt(2) * data['sigma']
    
    # Compute the fourier matrix
    A = vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv, pulse=Prior.pulse)

    # Account for the bispectrum phase shift
    modelvis = np.dot(A, iimage)
    p = m * modelvis

    # Define the objective function and gradient
    def objfunc(cvimage):
        mimage = mcv(cvimage)
        if polentropy == "hw":
            s = -shw(mimage, iimage)
        elif polentropy == "logm":
            s = -sm(mimage, iimage)
        elif polentropy == "tv":
            s = -stv_pol(mimage, iimage, Prior.xdim, Prior.ydim)
            
        return s + beta * (chisq_p(mimage, iimage, A, p, sigmap) - 1)
   
    def objgrad(cvimage):
        mimage = mcv(cvimage)
        if polentropy == "hw":
            s = -shwgrad(mimage, iimage)
        elif polentropy == "logm":
            s = -smgrad(mimage, iimage)
        elif polentropy == "tv":
            s = -stv_pol_grad(mimage, iimage, Prior.xdim, Prior.ydim)
            
        return  (s + beta * chisqgrad_p(mimage, iimage, A, p, sigmap)) * mchainlist(cvimage)
    
    # Plotting function for each iteration
    global nit
    nit = 0
    def plotcur(mcv_step):
        global nit
        m_step = mcv(mcv_step)
        chi2 = 0
        chi2p = chisq_p(m_step, iimage, A, p, sigmap)
        plot_m(iimage, m_step, Prior, nit, chi2, chi2p, pcut=pcut, nvec=nvec, ipynb=ipynb)
        nit += 1

    plotcur(allprior)
    
    # Minimize
    optdict = {'maxiter':maxit, 'ftol':stop, 'maxcor':NHIST}
    tstart = time.time()
    res = opt.minimize(objfunc, allprior, method='L-BFGS-B', jac=objgrad, 
                       options=optdict, callback=plotcur)
    tstop = time.time()
    out = mcv(res.x)
    
    # Print stats
    print "time: %f s" % (tstop - tstart)
    print "J: %f" % res.fun
    print "Chi^2: %f" % chisq_p(out, iimage, A, p, sigmap)
    print res.message
    
    # Return Image object
    qimfinal = qimage(iimage, out[0:len(iimage)], out[len(iimage):])
    uimfinal = uimage(iimage, out[0:len(iimage)], out[len(iimage):])
    outim = vb.Image(iimage.reshape(Prior.ydim, Prior.xdim), Prior.psize, Prior.ra, Prior.dec, 
                     rf=Prior.rf, source=Prior.source, mjd=Prior.mjd, pulse=Prior.pulse) 
    outim.add_qu(qimfinal.reshape(Prior.ydim, Prior.xdim), uimfinal.reshape(Prior.ydim, Prior.xdim))
    return outim
             
def maxen_m(Obsdata, Prior, maxit=100, beta=1e4, polentropy="hw", stop=1e-100, nvec=15, pcut=0.05, prior=True, ipynb=False):
    """Run maximum entropy on pol. ratios. 
       Obsdata is an Obsdata object,
       Prior is an Image object containing the Stokes I image and Q & U priors.
       Returns an Image object.
       Operates on m and chi images (NOT Q and U)
       The "lagrange multiplier" is not a free parameter
    """
    
    print "Imaging Q, U with pol. ratios . . ."
    
    # Catch problem if uvrange < largest baseline
    uvrange = 1/Prior.psize
    maxbl = np.max(Obsdata.unpack(['uvdist'])['uvdist'])
    if uvrange < maxbl:
        raise Exception("pixel spacing is larger than smallest spatial wavelength!")
    
    # Set up priors
    iimage = Prior.imvec
    if len(Prior.qvec) and prior:
        mprior = np.abs(Prior.qvec + 1j*Prior.uvec) / Prior.imvec
        chiprior = np.arctan2(Prior.uvec, Prior.qvec) / 2.0
        allprior = np.hstack((mprior, chiprior))
    else:
        # !AC get the actual zero baseline pol. frac from the data!
        mprior = 0.2 * (np.ones(len(iimage)) + 1e-10 * np.random.rand(len(iimage)))
        chiprior = np.zeros(len(iimage)) + 1e-10 * np.random.rand(len(iimage))
        allprior = np.hstack((mprior, chiprior))
    
    # Change of variables    
    allprior = mcv_r(allprior)
    
    # Data
    data = Obsdata.unpack(['u','v','vis','m','sigma'])
    uv = np.hstack((data['u'].reshape(-1,1), data['v'].reshape(-1,1)))
    m = data['m']
    sigmam = vb.merr(data['sigma'], data['vis'], data['m'])
    
    # Compute the Fourier matrix
    A = vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv, pulse=Prior.pulse)
    
    # Define the objective function and gradient
    def objfunc(cvimage):
        mimage = mcv(cvimage)
        if polentropy == "hw":
            s = -shw(mimage, iimage)
        elif polentropy == "logm":
            s = -sm(mimage, iimage)
        elif polentropy == "tv":
            s = -stv_pol(mimage, iimage, Prior.xdim, Prior.ydim)
            
        return s + beta * (chisq_m(mimage, iimage, A, m, sigmam) - 1)
   
    def objgrad(cvimage):
        mimage = mcv(cvimage)
        if polentropy == "hw":
            s = -shwgrad(mimage, iimage)
        elif polentropy == "logm":
            s = -smgrad(mimage, iimage)
        elif polentropy == "tv":
            s = -stv_pol_grad(mimage, iimage, Prior.xdim, Prior.ydim)
        
        return  (s + beta * chisqgrad_m(mimage, iimage, A, m, sigmam)) * mchainlist(cvimage)
    
    # Plotting function for each iteration
    global nit
    nit = 0
    def plotcur(mcv_step):
        global nit
        m_step = mcv(mcv_step)
        chi2 = 0
        chi2m = chisq_m(m_step, iimage, A, m, sigmam)
        plot_m(iimage, m_step, Prior, nit, chi2, chi2m, pcut=pcut, nvec=nvec, ipynb=ipynb)
        nit += 1
    plotcur(allprior)
        
    # Minimize
    optdict = {'maxiter':maxit, 'ftol':stop, 'maxcor':NHIST}
    tstart = time.time()
    res = opt.minimize(objfunc, allprior, method='L-BFGS-B', jac=objgrad, 
                       options=optdict, callback=plotcur)
    tstop = time.time()
    out = mcv(res.x)
    
    # Print stats
    print "time: %f s" % (tstop - tstart)
    print "J: %f" % res.fun
    print "Chi^2: %f" % chisq_m(out, iimage, A, m, sigmam)
    print res.message
    
    # Return Image object
    qimfinal = qimage(iimage, out[0:len(iimage)], out[len(iimage):])
    uimfinal = uimage(iimage, out[0:len(iimage)], out[len(iimage):])
    outim = vb.Image(iimage.reshape(Prior.ydim, Prior.xdim), Prior.psize, Prior.ra, Prior.dec, 
                     rf=Prior.rf, source=Prior.source, mjd=Prior.mjd, pulse=Prior.pulse) 
    outim.add_qu(qimfinal.reshape(Prior.ydim, Prior.xdim), uimfinal.reshape(Prior.ydim, Prior.xdim))
    return outim

def maxen_bs_m(Obsdata, Prior, flux, maxit=100, alpha=1e6, beta=7.5e5, gamma=1.5e6, delta=1e5,
               entropy="gs", polentropy="hw", stop=1e-500, nvec=15, pcut=0.05, ipynb=False):
    """Run maximum entropy SIMULTANEOUSLY on bispectrum and on pol. ratios. 
       Obsdata is an Obsdata object,
       Prior is an Image object containing the Stokes I image and Q & U priors.
       returns an Image object.
       Lagrange multipliers are not free parameters
    """
    
    print "Imaging I, Q, U with bispectrum and pol. ratios simulaneously . . ."
    
    # Catch problem if uvrange < largest baseline
    uvrange = 1/Prior.psize
    maxbl = np.max(Obsdata.unpack(['uvdist'])['uvdist'])
    if uvrange < maxbl:
        raise Exception("pixel spacing is larger than smallest spatial wavelength!")
    if (Prior.psize != InitIm.psize) or (Prior.xdim != InitIm.xdim) or (Prior.ydim != InitIm.ydim):
        raise Exception("initial image does not match dimensions of the prior image!")
    
    # Set up priors
    nprior = flux * Prior.imvec / np.sum(Prior.imvec)
    if len(Prior.qvec):
        mprior = np.abs(Prior.qvec + 1j*Prior.uvec) / Prior.imvec
        chiprior = np.arctan2(Prior.uvec, Prior.qvec) / 2.0
        polprior = np.hstack((mprior, chiprior))
    else:
        # !AC get the actual zero baseline pol. frac from the data!
        mprior = 0.2 * (np.ones(len(iimage)) + 1e-10 * np.random.rand(len(iimage)))
        chiprior = np.zeros(len(iimage)) + 1e-10 * np.random.rand(len(iimage))
        polprior = np.hstack((mprior, chiprior))
    
    # Change variables and package full prior
    allprior = np.hstack((np.log(nprior), mcv_r(polprior)))
      
    # Get data
    biarr = Obsdata.bispectra(mode="all")
    uv1 = np.hstack((biarr['u1'].reshape(-1,1), biarr['v1'].reshape(-1,1)))
    uv2 = np.hstack((biarr['u2'].reshape(-1,1), biarr['v2'].reshape(-1,1)))
    uv3 = np.hstack((biarr['u3'].reshape(-1,1), biarr['v3'].reshape(-1,1)))
    bi = biarr['bispec']
    sigsb = biarr['sigmab']
    #sigsb_2 = scaled_bisigs(Obsdata) # Correction for overcounting NDOF
    
    poldata = Obsdata.unpack(['u','v','vis','m','sigma'])
    uvpol = np.hstack((poldata['u'].reshape(-1,1), poldata['v'].reshape(-1,1)))
    m = poldata['m']
    sigsm = vb.merr(poldata['sigma'], poldata['vis'], poldata['m'])
    
    # Compute the Fourier matrices
    A3 = (vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse),
          vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse),
          vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse)
         )
    Apol = vb.ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uvpol, pulse=Prior.pulse)
    
    # Coordinate matrix for COM constraint
    coord = Prior.psize * np.array([[[x,y] for x in np.arange(Prior.xdim/2,-Prior.xdim/2,-1)]
                                           for y in np.arange(Prior.ydim/2,-Prior.ydim/2,-1)])
    coord = coord.reshape(Prior.ydim*Prior.xdim, 2)                                       

    # Define the bispectrum objective function and gradient
    def objfunc_b(iim):
        if entropy == "simple":
            s = -ssimple(iim, nprior)
        elif entropy == "l1":
            s = -sl1(iim, nprior)
        elif entropy == "gs":
            s = -sgs(iim, nprior) 
        elif entropy == "tv":
            s = -stv(iim, Prior.xdim, Prior.ydim)
            
        c = alpha * (chisq_bi(iim, A3, bi, sigsb) - 1)
        t = gamma * (np.sum(iim) - flux)**2
        cm = delta * (np.sum(iim * coord[:,0]) + np.sum(iim * coord[:,1]))**2
        return  s + c + t + cm
        
    def objgrad_b(iim):
        if entropy == "simple":
            s = -ssimplegrad(iim, nprior)
        elif entropy == "l1":
            s = -sl1grad(iim, nprior)
        elif entropy == "gs":
            s = -sgsgrad(iim, nprior)
        elif entropy == "tv":
            s = -stvgrad(iim, Prior.xdim, Prior.ydim)
            
        c = alpha * chisqgrad_bi(iim, A3, bi, sigsb)
        
        t = 2 * gamma * (np.sum(iim) - flux)
        cm = 2 * delta * (coord[:,0] + coord[:,1]) * (np.sum(iim * coord[:,0]) + np.sum(iim * coord[:,1]))
        return  (s + c + t + cm)
            
    # Define the total polarimetric objective function and gradient
    def objfunc(allimage):
        iim = np.exp(allimage[0:len(nprior)])
        mim = mcv(allimage[len(nprior):])
        objb = objfunc_b(iim)
        
        if polentropy == "hw":
            s = -shw(mim, iim)
        elif polentropy == "logm":
            s = -sm(mim, iim)
        elif polentropy == "tv":
            s = -stv_pol(mim, iim, Prior.xdim, Prior.ydim)
            
        c = beta * (chisq_m(mim, iim, Apol, m, sigsm) - 1)
        return  s + c + objb
        
    def objgrad(allimage):
        iim = np.exp(allimage[0:len(nprior)])
        mim = mcv(allimage[len(nprior):])
        gradb = (objgrad_b(iim) + beta * chisqgrad_m_i(mim, iim, Apol, m, sigsm)) * iim
        
        if polentropy == "hw":
            s = -shwgrad(mim, iim)
        elif polentropy == "logm":
            s = -smgrad(mim, iim)
        elif polentropy == "tv":
            s = -stv_pol_grad(mim, iim, Prior.xdim, Prior.ydim)    
            gradb = gradb - stv_pol_grad_i(mim, iim, Prior.xdim, Prior.ydim) * iim
            
        c = beta * chisqgrad_m(mim, iim, Apol, m, sigsm)
        gradm = (s + c) * mchainlist(allimage[len(nprior):])
        
        return  np.hstack((gradb, gradm))
           
    # Plotting function for each iteration
    global nit
    nit = 0
    def plotcur(all_step):
        global nit
        i_step = np.exp(all_step[0:len(nprior)])
        m_step = mcv(all_step[len(nprior):])
        chi2 = chisq_bi(i_step, A3, bi, sigsb)
        chi2m = chisq_m(m_step, i_step, Apol, m, sigsm)
        plot_m(i_step, m_step, Prior, nit, chi2, chi2m, pcut=pcut, nvec=nvec, ipynb=ipynb)
        nit += 1

    plotcur(allprior)
        
    # Minimize
    optdict = {'maxiter':maxit, 'ftol':stop, 'maxcor':NHIST}
    tstart = time.time()
    res = opt.minimize(objfunc, allprior, method='L-BFGS-B', jac=objgrad, 
                       options=optdict, callback=plotcur)
    tstop = time.time()
    outi = np.exp(res.x[0: len(nprior)])
    outp = mcv(res.x[len(nprior):])
    
    # Print stats
    print "time: %f s" % (tstop - tstart)
    print "J: %f" % res.fun
    print "Chi^2_b: %f" % chisq_bi(outi, A3, bi, sigsb)
    print "Chi^2_m: %f" % chisq_m(outp, outi, Apol, m, sigsm)
    print res.message
    
    # Return Image object
    qimfinal = qimage(outi, outp[0:len(outi)], outp[len(outi):])
    uimfinal = uimage(outi, outp[0:len(outi)], outp[len(outi):])
    outim = vb.Image(outi.reshape(Prior.ydim, Prior.xdim), Prior.psize, Prior.ra, Prior.dec, 
                     rf=Prior.rf, source=Prior.source, mjd=Prior.mjd, pulse=Prior.pulse) 
    outim.add_qu(qimfinal.reshape(Prior.ydim, Prior.xdim), uimfinal.reshape(Prior.ydim, Prior.xdim))
    return outim     

##################################################################################################
# Blurring Function
##################################################################################################

def blur_circ(image, fwhm_i, fwhm_pol=0):
    """Apply a circular gaussian filter to the I image
       fwhm is in radians
    """ 
    
    # Blur Stokes I
    sigma = fwhm_i / (2. * np.sqrt(2. * np.log(2.)))
    sigmap = sigma / image.psize
    im = filt.gaussian_filter(image.imvec.reshape(image.ydim, image.xdim), (sigmap, sigmap))
    out = vb.Image(im, image.psize, image.ra, image.dec, rf=image.rf, source=image.source, mjd=image.mjd)
   
    # Blur Stokes Q and U
    if len(image.qvec) and fwhm_pol:
        sigma = fwhm_pol / (2. * np.sqrt(2. * np.log(2.)))
        sigmap = sigma / image.psize
        imq = filt.gaussian_filter(image.qvec.reshape(image.ydim,image.xdim), (sigmap, sigmap))
        imu = filt.gaussian_filter(image.uvec.reshape(image.ydim,image.xdim), (sigmap, sigmap))
        out.add_qu(imq, imu)
        
    return out
          
##################################################################################################
# Chi-squared and Gradient Functions
##################################################################################################

# Visibility phase and amplitude chi-squared
def chisq(imvec, Amatrix, vis, sigma):
    """Visibility chi-squared"""
   
    samples = np.dot(Amatrix, imvec)
    return np.sum(np.abs((samples-vis)/sigma)**2) / (2*len(vis))
    
def chisqgrad(imvec, Amatrix, vis, sigma):
    """The gradient of the visibility chi-squared"""
    
    samples = np.dot(Amatrix, imvec)
    wdiff = (vis - samples) / (sigma**2) 
    out = -np.real(np.dot(Amatrix.conj().T, wdiff)) / len(vis)
    return out

# Bispectrum chi-squared
def chisq_bi(imvec, Amatrices, bis, sigma):
    """Bispectrum chi-squared"""
    
    bisamples = np.dot(Amatrices[0], imvec) * np.dot(Amatrices[1], imvec) * np.dot(Amatrices[2], imvec)
    return np.sum(np.abs((bis - bisamples)/sigma)**2) / (2.*len(bis))
    
def chisqgrad_bi(imvec, Amatrices, bis, sigma):
    """The gradient of the bispectrum chi-squared"""
    
    bisamples = np.dot(Amatrices[0], imvec) * np.dot(Amatrices[1], imvec) * np.dot(Amatrices[2], imvec)
    wdiff = ((bis - bisamples).conj()) / (sigma**2)
    pt1 = wdiff * np.dot(Amatrices[1],imvec) * np.dot(Amatrices[2],imvec)
    pt2 = wdiff * np.dot(Amatrices[0],imvec) * np.dot(Amatrices[2],imvec)
    pt3 = wdiff * np.dot(Amatrices[0],imvec) * np.dot(Amatrices[1],imvec)
    out = -np.real(np.dot(pt1, Amatrices[0]) + np.dot(pt2, Amatrices[1]) + np.dot(pt3, Amatrices[2])) / len(bis)
    return out

#Closure Amplitudes chi-squared
def chisq_clamp(imvec, Amatrices, clamp, sigma):
    """Closure Amplitudes (normalized) chi-squared"""
    
    clamp_samples = np.abs(np.dot(Amatrices[0], imvec) * np.dot(Amatrices[1], imvec) / (np.dot(Amatrices[2], imvec) * np.dot(Amatrices[3], imvec)))
    return np.sum(np.abs((clamp - clamp_samples)/sigma)**2) / (len(clamp))

def chisqgrad_clamp(imvec, Amatrices, clamp, sigma):
    
    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    i4 = np.dot(Amatrices[3], imvec)
    clamp_samples = np.abs((i1 * i2) / (i3 * i4))
    
    pp = ((clamp - clamp_samples) * clamp_samples) / (sigma**2)
    pt1 = pp / i1
    pt2 = pp / i2
    pt3 = -pp / i3
    pt4 = -pp / i4
    out = (-2.0/len(clamp)) * np.real(np.dot(pt1, Amatrices[0]) + np.dot(pt2, Amatrices[1]) + np.dot(pt3, Amatrices[2]) + np.dot(pt4, Amatrices[3]))
    return out

#Visibility Amplitudes chi-squared
def chisq_visamp(imvec, A, amp, sigma):
    """Closure Amplitudes (normalized) chi-squared"""
    
    amp_samples = np.abs(np.dot(A, imvec))
    return np.sum(np.abs((amp - amp_samples)/sigma)**2) / (len(amp))

def chisqgrad_visamp(imvec, A, amp, sigma):
    
    i1 = np.dot(A, imvec)
    amp_samples = np.abs(i1)
    
    pp = ((amp - amp_samples) * amp_samples) / (sigma**2) / i1
    out = (-2.0/len(amp)) * np.real(np.dot(pp, A))
    return out
    
#Closure phases chi-squared
def chisq_clphase(imvec, Amatrices, clphase, sigma):
    """Closure Phases (normalized) chi-squared"""
    clphase = clphase * DEGREE
    sigma = sigma * DEGREE
    clphase_samples = np.angle(np.dot(Amatrices[0], imvec) * np.dot(Amatrices[1], imvec) * np.dot(Amatrices[2], imvec))
    return (2.0/len(clphase)) * np.sum((1.0 - np.cos(clphase-clphase_samples)) / (sigma**2))


def chisqgrad_clphase(imvec, Amatrices, clphase, sigma):
    clphase = clphase * DEGREE
    sigma = sigma * DEGREE
    
    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    clphase_samples = np.angle(i1 * i2 * i3)
    
    pref = np.sin(clphase - clphase_samples) / (sigma**2)
    pt1 = pref / i1
    pt2 = pref / i2
    pt3 = pref / i3
    out = -(2.0/len(clphase)) * np.imag(np.dot(pt1, Amatrices[0]) + np.dot(pt2, Amatrices[1]) + np.dot(pt3, Amatrices[2]))
    return out
    
# Polarimetric Amplitude and Phase chi-squared
def chisq_p(polimage, iimage, Amatrix, p, sigmap):
    """Pol. ratio chi-squared"""
    pimage = iimage * polimage[0:len(iimage)] * np.exp(2j*polimage[len(iimage):])
    psamples = np.dot(Amatrix, pimage) 
    return np.sum(np.abs((p - psamples))**2/(sigmap**2)) / (2*len(p))   

def chisqgrad_p(polimage, iimage, Amatrix, p, sigmap):
    """The gradient of the pol. ratio chisq. w/r/t m and chi"""
    
    mimage = polimage[0:len(iimage)]
    chiimage = polimage[len(iimage):]
    pimage = iimage * mimage * np.exp(2j*chiimage)
    
    psamples = np.dot(Amatrix, pimage)
    pdiff = (p - psamples) / (sigmap**2)
    
    gradm = -np.real(iimage*np.exp(-2j*chiimage) * np.dot(Amatrix.conj().T, pdiff)) / len(p)
    gradchi = -2 * np.imag(pimage.conj() * np.dot(Amatrix.conj().T, pdiff)) / len(p)
    return np.hstack((gradm, gradchi))

def chisqgrad_p_i(polimage, iimage, Amatrix, p, sigmap):
    """The gradient of the pol. ratio chisq w/r/t I"""
    
    mimage = polimage[0:len(iimage)]
    chiimage = polimage[len(iimage):]
    pimage = iimage * mimage * np.exp(2j*chiimage)
    
    psamples = np.dot(Amatrix, pimage)
    pdiff = (p - psamples) / (sigmap**2)
    
    grad = -np.real(mimage * np.exp(-2j*chiimage) * np.dot(Amatrix.conj().T, pdiff)) / len(p)
    return grad
    
# Polarimetric ratio chi-squared
def chisq_m(polimage, iimage, Amatrix, m, sigmam):
    """Pol. ratio chi-squared"""
    pimage = iimage * polimage[0:len(iimage)] * np.exp(2j*polimage[len(iimage):])
    msamples = np.dot(Amatrix, pimage) / np.dot(Amatrix, iimage)
    return np.sum(np.abs((m - msamples))**2/(sigmam**2)) / (2*len(m))   

def chisqgrad_m(polimage, iimage, Amatrix, m, sigmam):
    """The gradient of the pol. ratio chisq. w/r/t m and chi"""
    
    mimage = polimage[0:len(iimage)]
    chiimage = polimage[len(iimage):]
    pimage = iimage * mimage * np.exp(2j*chiimage)
    
    psamples = np.dot(Amatrix, pimage)
    isamples = np.dot(Amatrix, iimage)
    msamples = psamples/isamples
    mdiff = (m - msamples) / (isamples.conj() * sigmam**2)
    
    gradm = -np.real(iimage*np.exp(-2j*chiimage) * np.dot(Amatrix.conj().T, mdiff)) / len(m)
    gradchi = -2 * np.imag(pimage.conj() * np.dot(Amatrix.conj().T, mdiff)) / len(m)
    return np.hstack((gradm, gradchi))

def chisqgrad_m_i(polimage, iimage, Amatrix, m, sigmam):
    """The gradient of the pol. ratio chisq w/r/t I"""
    
    mimage = polimage[0:len(iimage)]
    chiimage = polimage[len(iimage):]
    pimage = iimage * mimage * np.exp(2j*chiimage)
    
    psamples = np.dot(Amatrix, pimage)
    isamples = np.dot(Amatrix, iimage)
    msamples = psamples/isamples         
    mdiff = (m - msamples) / (isamples.conj() * sigmam**2)
    
    grad1 = -np.real(mimage * np.exp(-2j*chiimage) * np.dot(Amatrix.conj().T, mdiff)) / len(m)
    grad2 = np.real(np.dot(Amatrix.conj().T, msamples.conj() * mdiff)) / len(m)
    return grad1 + grad2

# Polarimetric Bispectrum chi-squared
def chisq_pbi(polimage, iimage, Amatrices, bis_p, sigma):
    """Pol. bispectrum chi-squared"""
    
    mimage = polimage[0:len(iimage)]
    chiimage = polimage[len(iimage):]
    pimage = iimage * mimage * np.exp(2j*chiimage)
    
    bisamples_p = np.dot(Amatrices[0], pimage) * np.dot(Amatrices[1], pimage) * np.dot(Amatrices[2], pimage)
    return np.sum(np.abs((bis_p - bisamples_p)/sigma)**2) / (2.*len(bis_p))
    
def chisqgrad_pbi(polimage, iimage, Amatrices, bis_p, sigma):
    """The gradient of the pol. bispectrum chisq w/r/t m and chi"""
    
    mimage = polimage[0:len(iimage)]
    chiimage = polimage[len(iimage):]
    pimage = iimage * mimage * np.exp(2j*chiimage)
    
    bisamples_p = np.dot(Amatrices[0], pimage) * np.dot(Amatrices[1], pimage) * np.dot(Amatrices[2], pimage)
    wdiff = ((bis_p - bisamples_p).conj()) / (sigma**2)
    pt1 = wdiff * np.dot(Amatrices[1],pimage) * np.dot(Amatrices[2],pimage)
    pt2 = wdiff * np.dot(Amatrices[0],pimage) * np.dot(Amatrices[2],pimage)
    pt3 = wdiff * np.dot(Amatrices[0],pimage) * np.dot(Amatrices[1],pimage)
    ptsum = np.dot(pt1, Amatrices[0]) + np.dot(pt2, Amatrices[1]) + np.dot(pt3, Amatrices[2])
    
    gradm = -np.real(ptsum * iimage * np.exp(2j*chiimage)) / len(bis_p)
    gradchi = 2 * np.imag(ptsum * pimage) / len(bis_p)
    return np.hstack((gradm, gradchi))

def chisqgrad_pbi_i(polimage, iimage, Amatrices, bis_p, sigma):
    """The gradient of the pol. bispectrum chisq w/r/t I"""
    
    mimage = polimage[0:len(iimage)]
    chiimage = polimage[len(iimage):]
    pimage = iimage * mimage * np.exp(2j*chiimage)
    
    bisamples_p = np.dot(Amatrices[0], pimage) * np.dot(Amatrices[1], pimage) * np.dot(Amatrices[2], pimage)
    wdiff = ((bis_p - bisamples_p).conj()) / (sigma**2)
    pt1 = wdiff * np.dot(Amatrices[1],pimage) * np.dot(Amatrices[2],pimage)
    pt2 = wdiff * np.dot(Amatrices[0],pimage) * np.dot(Amatrices[2],pimage)
    pt3 = wdiff * np.dot(Amatrices[0],pimage) * np.dot(Amatrices[1],pimage)
    ptsum = np.dot(pt1, Amatrices[0]) + np.dot(pt2, Amatrices[1]) + np.dot(pt3, Amatrices[2])
    
    gradi = -np.real(ptsum * mimage * np.exp(2j*chiimage)) / len(bis_p)
    return gradi
    
##################################################################################################
# Entropy and Gradient Functions
##################################################################################################

# !AC All images should be 1d arrays 
# !AC May want to pass 2d arrays (esp. in TV)
# polimage should be [mimage, chiimage]

# Total intensity Entropys
def spatch(imvec, priorvec):
    return -0.5*np.sum( ( imvec - priorvec) ** 2)

def spatchgrad(imvec, priorvec):
    return -(imvec  - priorvec)

def ssimple(imvec, priorvec):
    return -np.sum(imvec*np.log(imvec/priorvec))

def ssimplegrad(imvec, priorvec):
    return -np.log(imvec/priorvec) - 1
    
def sl1(imvec, priorvec):
    return -np.sum(np.abs(imvec - priorvec))

def sl1grad(imvec, priorvec):
    return -np.sign(imvec - priorvec)
    
def sgs(imvec, priorvec):
    return np.sum(imvec - priorvec - imvec*np.log(imvec/priorvec))


def sgsgrad(imvec, priorvec):
    return -np.log(imvec/priorvec)


def stv(imvec, nx, ny):
    """Total variation entropy"""
    
    im = imvec.reshape(ny, nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    out = -np.sum(np.sqrt(np.abs(im_l1 - im)**2 + np.abs(im_l2 - im)**2))
    return out

def stvgrad(imvec, nx, ny):
    """Total variation gradient"""
    
    im = imvec.reshape(ny,nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    im_r1 = np.roll(impad, 1, axis=0)[1:ny+1, 1:nx+1]
    im_r2 = np.roll(impad, 1, axis=1)[1:ny+1, 1:nx+1]
    im_r1l2 = np.roll(np.roll(impad, 1, axis=0), -1, axis=1)[1:ny+1, 1:nx+1]
    im_l1r2 = np.roll(np.roll(impad, 1, axis=0), -1, axis=1)[1:ny+1, 1:nx+1]
    
    g1 = (2*im - im_l1 - im_l2) / np.sqrt((im_l1 - im)**2 + (im_l2 - im)**2)
    g2 = (im - im_r1) / np.sqrt((im_r1 - im)**2 + (im_r1l2 - im_r1)**2)
    g3 = (im - im_r2) / np.sqrt((im_r2 - im)**2 + (im_l1r2 - im_r2)**2)
    return -(g1 + g2 + g3).flatten()
    
# Polarimetric Entropys
def sm(polimage, iimage):
    """I log m entropy"""
    
    mimage = polimage[0:len(iimage)]
    return -np.sum(iimage * np.log(mimage))

def smgrad(polimage, iimage):
    """I log m entropy gradient"""
    
    mimage = polimage[0:len(iimage)]
    chigrad = np.zeros(len(iimage))
    mgrad = -iimage / mimage
    return np.hstack((mgrad, chigrad))
          
def shw(polimage, iimage):
    """Holdaway-Wardle polarimetric entropy"""
    
    mimage = polimage[0:len(iimage)]
    return -np.sum(iimage * (((1+mimage)/2) * np.log((1+mimage)/2) + ((1-mimage)/2) * np.log((1-mimage)/2)))

def shwgrad(polimage, iimage):
    """Gradient of the Holdaway-Wardle polarimetric entropy"""
    
    mimage = polimage[0:len(iimage)]
    chigrad = np.zeros(len(iimage))
    mgrad = -iimage * np.arctanh(mimage)
    return np.hstack((mgrad, chigrad))

def stv_pol(polimage, iimage, nx, ny):
    """Total variation of I*m*exp(2Ichi)"""
    
    iimage = iimage.reshape(ny, nx)
    mimage = polimage[0:nx*ny].reshape(ny,nx)
    chiimage = polimage[nx*ny:].reshape(ny,nx)
    im = iimage * mimage * np.exp(2j*chiimage)
    
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    out = -np.sum(np.sqrt(np.abs(im_l1 - im)**2 + np.abs(im_l2 - im)**2))
    return out

def stv_pol_grad(polimage, iimage, nx, ny):
    """Total variation entropy gradient"""
    
    iimage = iimage.reshape(ny, nx)
    mimage = polimage[0:nx*ny].reshape(ny,nx)
    chiimage = polimage[nx*ny:].reshape(ny,nx)
    im = iimage * mimage * np.exp(2j*chiimage)
    
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    im_r1 = np.roll(impad, 1, axis=0)[1:ny+1, 1:nx+1]
    im_r2 = np.roll(impad, 1, axis=1)[1:ny+1, 1:nx+1]
    im_r1l2 = np.roll(np.roll(impad, 1, axis=0), -1, axis=1)[1:ny+1, 1:nx+1]
    im_l1r2 = np.roll(np.roll(impad, 1, axis=0), -1, axis=1)[1:ny+1, 1:nx+1]
    
    # Denominators
    d1 = np.sqrt(np.abs(im_l1 - im)**2 + np.abs(im_l2 - im)**2)
    d2 = np.sqrt(np.abs(im_r1 - im)**2 + np.abs(im_r1l2 - im_r1)**2)
    d3 = np.sqrt(np.abs(im_r2 - im)**2 + np.abs(im_l1r2 - im_r2)**2)
    
    # dS/dm numerators
    m1 = 2*np.abs(im) - np.abs(im_l1)*np.cos(2*(np.angle(im_l1) - np.angle(im))) - np.abs(im_l2)*np.cos(2*(np.angle(im_l2) - np.angle(im)))
    m2 = np.abs(im) - np.abs(im_r1)*np.cos(2*(np.angle(im) - np.angle(im_r1)))
    m3 = np.abs(im) - np.abs(im_r2)*np.cos(2*(np.angle(im) - np.angle(im_r2)))
    mgrad = -(iimage*(m1/d1 + m2/d2 + m3/d3)).flatten()
    
    # dS/dchi numerators
    c1 = -2*np.abs(im*im_l1)*np.sin(2*(np.angle(im_l1) - np.angle(im))) - 2*np.abs(im*im_l2)*np.sin(2*(np.angle(im_l2) - np.angle(im)))
    c2 = 2*np.abs(im*im_r1)*np.sin(2*(np.angle(im) - np.angle(im_r1)))
    c3 = 2*np.abs(im*im_r2)*np.sin(2*(np.angle(im) - np.angle(im_r2)))
    chigrad = -(c1/d1 + c2/d2 + c3/d3).flatten()
    
    return np.hstack((mgrad, chigrad))

def stv_pol_grad_i(polimage, iimage, nx, ny):
    """Total variation entropy gradient w/r/t Stokes I"""
    
    iimage = iimage.reshape(ny, nx)
    mimage = polimage[0:nx*ny].reshape(ny,nx)
    chiimage = polimage[nx*ny:].reshape(ny,nx)
    im = iimage * mimage * np.exp(2j*chiimage)
    
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    im_r1 = np.roll(impad, 1, axis=0)[1:ny+1, 1:nx+1]
    im_r2 = np.roll(impad, 1, axis=1)[1:ny+1, 1:nx+1]
    im_r1l2 = np.roll(np.roll(impad, 1, axis=0), -1, axis=1)[1:ny+1, 1:nx+1]
    im_l1r2 = np.roll(np.roll(impad, 1, axis=0), -1, axis=1)[1:ny+1, 1:nx+1]
    
    # Denominators
    d1 = np.sqrt(np.abs(im_l1 - im)**2 + np.abs(im_l2 - im)**2)
    d2 = np.sqrt(np.abs(im_r1 - im)**2 + np.abs(im_r1l2 - im_r1)**2)
    d3 = np.sqrt(np.abs(im_r2 - im)**2 + np.abs(im_l1r2 - im_r2)**2)
    
    # Numerators
    m1 = 2*np.abs(im*im) - np.abs(im*im_l1)*np.cos(2*(np.angle(im_l1) - np.angle(im))) - np.abs(im*im_l2)*np.cos(2*(np.angle(im_l2) - np.angle(im)))
    m2 = np.abs(im*im) - np.abs(im*im_r1)*np.cos(2*(np.angle(im) - np.angle(im_r1)))
    m3 = np.abs(im*im) - np.abs(im*im_r2)*np.cos(2*(np.angle(im) - np.angle(im_r2)))
    
    grad = -((1./iimage)*(m1/d1 + m2/d2 + m3/d3)).flatten()
  
    return grad   
      
##################################################################################################
# Plotting Functions
##################################################################################################
def plot_i(im, Prior, nit, chi2, ipynb=False):
    
    plt.ion()
    plt.clf()
    
    plt.imshow(im.reshape(Prior.ydim,Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian')     
    xticks = vb.ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
    yticks = vb.ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title("step: %i  $\chi^2$: %f" % (nit, chi2), fontsize=20)
    plt.draw()

    if ipynb:
        display.clear_output()
        display.display(plt.gcf())

def plot_m(im, mim, Prior, nit, chi2, chi2m, pcut=0.05, nvec=15, ipynb=False):
    
    # Mask for low flux points
    thin = Prior.xdim/nvec
    mask = im.reshape(Prior.ydim, Prior.xdim) > pcut * np.max(im)
    mask2 = mask[::thin, ::thin]
    
    # Get vectors and ratio from current image
    x = np.array([[i for i in range(Prior.xdim)] for j in range(Prior.ydim)])[::thin, ::thin][mask2]
    y = np.array([[j for i in range(Prior.xdim)] for j in range(Prior.ydim)])[::thin, ::thin][mask2]
    q = qimage(im, mim[0:len(im)], mim[len(im):])
    u = uimage(im, mim[0:len(im)], mim[len(im):])
    a = -np.sin(np.angle(q+1j*u)/2).reshape(Prior.ydim, Prior.xdim)[::thin, ::thin][mask2]
    b = np.cos(np.angle(q+1j*u)/2).reshape(Prior.ydim, Prior.xdim)[::thin, ::thin][mask2]
    m = (np.abs(q + 1j*u)/im).reshape(Prior.ydim, Prior.xdim)
    m[-mask] = 0
    
    # Create figure and title
    plt.ion()
    plt.clf()
    plt.suptitle("step: %i  $\chi_b^2$: %f   $\chi_m^2$: %f" % (nit, chi2, chi2m), fontsize=20)
        
    # Stokes I plot
    plt.subplot(121)
    plt.imshow(im.reshape(Prior.ydim, Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian')
    plt.quiver(x, y, a, b,
               headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
               width=.01*Prior.xdim, units='x', pivot='mid', color='k', angles='uv', scale=1.0/thin)
    plt.quiver(x, y, a, b,
               headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
               width=.005*Prior.xdim, units='x', pivot='mid', color='w', angles='uv', scale=1.1/thin)

    xticks = vb.ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
    yticks = vb.ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title('Stokes I')
    
    # Ratio plot
    plt.subplot(122)
    plt.imshow(m, cmap=plt.get_cmap('winter'), interpolation='gaussian', vmin=0, vmax=1)
    plt.quiver(x, y, a, b,
               headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
               width=.01*Prior.xdim, units='x', pivot='mid', color='k', angles='uv', scale=1.0/thin)
    plt.quiver(x, y, a, b,
               headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
               width=.005*Prior.xdim, units='x', pivot='mid', color='w', angles='uv', scale=1.1/thin)

    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title('m (above %i %% max flux)' % int(pcut*100))
    
    # Display
    plt.draw()
    if ipynb:
        display.clear_output()
        display.display(plt.gcf())   
            
##################################################################################################
# Other Functions
##################################################################################################
def qimage(iimage, mimage, chiimage):
    """Return the Q image from m and chi"""
    return iimage * mimage * np.cos(2*chiimage)
    
def uimage(iimage, mimage, chiimage):
    """Return the U image from m and chi"""
    return iimage * mimage * np.sin(2*chiimage) 

#def scaled_bisigs(obsdata, vtype="vis"):
#    """scale bispectrum errors for overcounting degrees of freedom
#       see Bouman 2014
#    """
#    bilist = obsdata.bispectra(mode="time", vtype=vtype)
#    overct = []
#    for entry in bilist:
#        nscope = len(set(list(entry['t1']) + list(entry['t2']) + list(entry['t3'])))
#        for i in range(len(entry)):
#            overct.append(3.0/nscope)
#    overct = np.array(overct)   
#    biarr = obsdata.bispectra(mode="all", vtype=vtype)
#    sigsb = biarr['sigmab']/np.sqrt(overct)
#    
#    return sigsb
    
# !AC In these pol. changes of variables, might be useful to 
# take m -> m/100 by adjusting B (function becomes less steep around m' = 0)
B = .5
def mcv(polimage):
    """Change of pol. variables from range (-inf, inf) to (0,1)"""
   
    mtrans = 0.5 + np.arctan(polimage[0:len(polimage)/2]/B)/np.pi
    chitrans = polimage[len(polimage)/2::]
    return np.hstack((mtrans, chitrans))
    
def mcv_r(polimage):
    """Change of pol. variables from range (0,1) to (-inf,inf)"""
   
    mtrans = B*np.tan(np.pi*(polimage[0:len(polimage)/2] - 0.5))
    chitrans = polimage[len(polimage)/2::]
    return np.hstack((mtrans, chitrans))    

def mchainlist(polimage_cv):
    """The gradient change of variables, dm/dm'"""

    mchain = 1 / (B*np.pi*(1 + (polimage_cv[0:len(polimage_cv)/2]/B)**2))
    chichain = np.ones(len(polimage_cv)/2)
    return np.hstack((mchain, chichain))
  
    
