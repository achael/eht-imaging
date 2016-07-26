# maxen_dynamic.py

import sys
import time
import numpy as np
import maxen as mx
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
# Time-Variable Imagers
##################################################################################################

def Wrapped_Convolve(sig,ker):
    N = sig.shape[0]
    return scipy.signal.fftconvolve(np.pad(sig,((N, N), (N, N)), 'wrap'), np.pad(ker,((N, N), (N, N)), 'constant'),mode='same')[N:(2*N),N:(2*N)]

def dynamic_regularizer(Frames, ker):
    Blur_Frames = np.array([Wrapped_Convolve(f, ker) for f in Frames])
    return np.sum(np.diff(Blur_Frames,axis=0)**2) #for now, no sqrt to make the gradients easier

def dynamic_regularizer_gradient(Frames, ker):
    Blur_Frames = np.array([Wrapped_Convolve(f, ker) for f in Frames]) 

    quad = np.copy(Blur_Frames)
    quad = quad*0.0

    for i in range(Frames.shape[0]-1):
	quad[i]   = quad[i]   + 2.0*(Blur_Frames[i] - Blur_Frames[i+1])
	quad[i+1] = quad[i+1] + 2.0*(Blur_Frames[i+1] - Blur_Frames[i])

    grad = np.array([Wrapped_Convolve(quad[i], ker)*Frames[i] for i in range(Frames.shape[0])]).flatten()

    return grad

# !AC TODO: Add support for chirp pulses
def maxen_dynamic_bs(Obsdata_List, Prior_List, Flux_List, maxit=100, alpha=100, gamma=500, delta=500, beta=5, beta_blur=5, entropy="simple", stop=1e-10, ipynb=False, refresh_interval = 1000):
    """Run maximum entropy with the bispectrum
       Uses I = exp(I') change of variables.
       Obsdata is an Obsdata object, and Prior is an Image object.
       Returns Image object.
       The lagrange multiplier alpha is not a free parameter
    """
    print "Dynamic Imaging with the bispectrum. . ."
        
    gauss = np.array([[np.exp(-1.0*i**2/(2*beta_blur**2) - 1.0*j**2/(2.*beta_blur**2))
		                  for i in np.linspace((Prior_List[0].xdim-1)/2., -(Prior_List[0].xdim-1)/2., num=Prior_List[0].xdim)]  
		                  for j in np.linspace((Prior_List[0].ydim-1)/2., -(Prior_List[0].ydim-1)/2., num=Prior_List[0].ydim)]) #linespace makes thing symmetric
    gauss = gauss[0:Prior_List[0].ydim, 0:Prior_List[0].xdim]
    gauss = gauss / np.sum(gauss) # normalize to 1

    N_frame = len(Obsdata_List)
    N_pixel = Prior_List[0].xdim #pixel dimension

    # Catch problem if uvrange < largest baseline
    uvrange = 1./Prior_List[0].psize
    maxbl = np.max(Obsdata_List[0].unpack(['uvdist'])['uvdist'])
    if uvrange < maxbl:
        raise Exception("pixel spacing is larger than smallest spatial wavelength!")    
    
    logprior_List = [Prior_List[i].imvec for i in range(N_frame)]
    nprior_List = [Prior_List[i].imvec for i in range(N_frame)]
    A_List = [Prior_List[i].imvec for i in range(N_frame)]
    bi_List = [None,] * N_frame
    sigma_List = [None,] * N_frame
    for i in range(N_frame):
	# Normalize prior image to total flux        
        nprior_List[i] = Flux_List[i] * Prior_List[i].imvec / np.sum(Prior_List[i].imvec)
        logprior_List[i] = np.log(nprior_List[i])    
	# Data
	biarr = Obsdata_List[i].bispectra(mode="all", count="min")
	bi_List[i] = biarr['bispec']
	sigma_List[i] = biarr['sigmab']	
	# Compute the Fourier matrix
	uv1 = np.hstack((biarr['u1'].reshape(-1,1), biarr['v1'].reshape(-1,1)))
	uv2 = np.hstack((biarr['u2'].reshape(-1,1), biarr['v2'].reshape(-1,1)))
	uv3 = np.hstack((biarr['u3'].reshape(-1,1), biarr['v3'].reshape(-1,1)))
	A3 = (
	      vb.ftmatrix(Prior_List[i].psize, Prior_List[i].xdim, Prior_List[i].ydim, uv1),
	      vb.ftmatrix(Prior_List[i].psize, Prior_List[i].xdim, Prior_List[i].ydim, uv2),
	      vb.ftmatrix(Prior_List[i].psize, Prior_List[i].xdim, Prior_List[i].ydim, uv3)
	     )
	A_List[i] = A3
    

    # Define the objective function and gradient
    def objfunc(logim):
        Frames = np.exp(logim.reshape((-1, N_pixel, N_pixel)))     
        if entropy == "simple":
            s = np.sum(-mx.ssimple(Frames[i].ravel(), nprior_List[i]) for i in range(N_frame))
        elif entropy == "l1":
            s = np.sum(-mx.sl1(Frames[i].ravel(), nprior_List[i]) for i in range(N_frame))
        elif entropy == "gs":
            s = np.sum(-mx.sgs(Frames[i].ravel(), nprior_List[i]) for i in range(N_frame))
        elif entropy == "tv":
            s = np.sum(-mx.stv(Frames[i].ravel(), Prior_List[0].xdim, Prior_List[0].ydim) for i in range(N_frame))

        chisq_total = np.sum(mx.chisq_bi(Frames[i].ravel(), A_List[i], bi_List[i], sigma_List[i]) for i in range(N_frame))/N_frame  
        s_dynamic = dynamic_regularizer(Frames, gauss)
	t = np.sum( [ gamma * (np.sum(Frames[i].ravel()) - Flux_List[i])**2 for i in range(N_frame)] )/N_frame  
        return s + beta * s_dynamic + alpha * (chisq_total - 1.) + t
        
    def objgrad(logim):
	Frames = np.exp(logim.reshape((-1, N_pixel, N_pixel)))
        if entropy == "simple":
            s = np.concatenate([-mx.ssimplegrad(Frames[i].ravel(), nprior_List[i])*Frames[i].ravel() for i in range(N_frame)])
        elif entropy == "l1":
            s = np.concatenate([-mx.sl1grad(Frames[i].ravel(), nprior_List[i])*Frames[i].ravel() for i in range(N_frame)])
        elif entropy == "gs":
            s = np.concatenate([-mx.sgsgrad(Frames[i].ravel(), nprior_List[i])*Frames[i].ravel() for i in range(N_frame)])
        elif entropy == "tv":
            s = np.concatenate([-mx.stvgrad(Frames[i].ravel(), Prior_List[0].xdim, Prior_List[0].ydim)*Frames[i].ravel() for i in range(N_frame)])
            
	chisq_total_grad = np.concatenate([mx.chisqgrad_bi(Frames[i].ravel(), A_List[i], bi_List[i], sigma_List[i])*Frames[i].ravel()/N_frame for i in range(N_frame)])  

	s_dynamic_grad = dynamic_regularizer_gradient(Frames, gauss)

	t = np.concatenate( [ 2.0 * gamma * (np.sum(Frames[i].ravel()) - Flux_List[i]) * Frames[i].ravel()/N_frame   for i in range(N_frame)] )

        return (s + beta * s_dynamic_grad + alpha * chisq_total_grad + t) 

    def objgrad_numeric(x):  #This calculates the gradient numerically
	#dx = 10.0**-8
	J0 = objfunc(x)
	Jgrad = np.copy(x)
	for i in range(len(x)):
	    xp = np.copy(x)
	    dx = xp[i]*10.0**-8#+10.0**-8
	    xp[i] = xp[i] + dx
	    J1 = objfunc(xp)
	    Jgrad[i] = (J1-J0)/dx
	return Jgrad

    # Plotting function for each iteration
    global nit
    nit = 0
    def plotcur(logim_step):
        global nit
        nit += 1
	if nit%10 == 0:        
		print "iteration %d" % nit
		Frames = np.exp(logim_step.reshape((-1, N_pixel, N_pixel)))
		chi2 = np.sum(mx.chisq_bi(Frames[i].ravel(), A_List[i], bi_List[i], sigma_List[i]) for i in range(N_frame))/N_frame
		s = np.sum(-mx.ssimple(Frames[i].ravel(), nprior_List[i]) for i in range(N_frame))
		s_dynamic = dynamic_regularizer(Frames, gauss)
		print "chi2: %f" % chi2
		print "s: %f" % s
		print "s_dynamic: %f" % s_dynamic
		#plot_i(Frames[0], Prior_List[0], nit, chi2, ipynb=ipynb)
		if nit%refresh_interval == 0:
			plot_i_dynamic(Frames[::5], Prior_List[0], nit, chi2, s, s_dynamic, ipynb=ipynb)        

		if 1==0:
		    numeric_gradient = objgrad_numeric(logim_step)
		    analytic_gradient = objgrad(logim_step)
		    print "Numeric Gradient:"
		    print numeric_gradient 
		    print "Analytic Gradient:"
		    print analytic_gradient
		    print "Max Fractional Difference in gradients:"
		    print max(abs((numeric_gradient - analytic_gradient)/numeric_gradient))
   
    # Plot the prior
    #print(logprior_List.flatten().shape)
    logprior = np.hstack(logprior_List).flatten()
    plotcur(logprior)
        
    # Minimize
    optdict = {'maxiter':maxit, 'ftol':stop, 'maxcor':NHIST} # minimizer params
    tstart = time.time()
    res = opt.minimize(objfunc, logprior, method='L-BFGS-B', jac=objgrad,
                       options=optdict, callback=plotcur)
    tstop = time.time()
    #out = np.exp(res.x)
    Frames = np.exp(res.x.reshape((-1, N_pixel, N_pixel)))

    # Print stats
    print "time: %f s" % (tstop - tstart)
    print "J: %f" % res.fun
    #print "Chi^2: %f" % mx.chisq(out, A, vis, sigma)
    print res.message
    
    #Return Frames
    outim = [vb.Image(Frames[i].reshape(Prior_List[0].ydim, Prior_List[0].xdim), Prior_List[0].psize, Prior_List[0].ra, Prior_List[0].dec, rf=Prior_List[0].rf, source=Prior_List[0].source, mjd=Prior_List[0].mjd) for i in range(N_frame)]

    return outim

def maxen_dynamic_minimal(Obsdata_List, Prior_List, maxit=100, alpha=100, beta=5, beta_blur=5, entropy="simple", stop=1e-10, ipynb=False, refresh_interval = 1000):
    """Run maximum entropy with full amplitudes and phases. 
       Uses I = exp(I') change of variables.
       Obsdata is an Obsdata object, and Prior is an Image object.
       Returns Image object.
       The lagrange multiplier alpha is not a free parameter
    """
    print "Imaging I with visibility amplitudes and phases . . ."
        
    gauss = np.array([[np.exp(-1.0*i**2/(2*beta_blur**2) - 1.0*j**2/(2.*beta_blur**2))
		                  for i in np.linspace((Prior_List[0].xdim-1)/2., -(Prior_List[0].xdim-1)/2., num=Prior_List[0].xdim)]  
		                  for j in np.linspace((Prior_List[0].ydim-1)/2., -(Prior_List[0].ydim-1)/2., num=Prior_List[0].ydim)]) #linespace makes thing symmetric
    gauss = gauss[0:Prior_List[0].ydim, 0:Prior_List[0].xdim]
    gauss = gauss / np.sum(gauss) # normalize to 1

    N_frame = len(Obsdata_List)
    N_pixel = Prior_List[0].xdim #pixel dimension

    # Catch problem if uvrange < largest baseline
    uvrange = 1./Prior_List[0].psize
    maxbl = np.max(Obsdata_List[0].unpack(['uvdist'])['uvdist'])
    if uvrange < maxbl:
        raise Exception("pixel spacing is larger than smallest spatial wavelength!")    
   
    logprior_List = [Prior_List[i].imvec for i in range(N_frame)]
    nprior_List = [Prior_List[i].imvec for i in range(N_frame)]
    A_List = [Prior_List[i].imvec for i in range(N_frame)]
    vis_List = [None,] * N_frame
    sigma_List = [None,] * N_frame
    for i in range(N_frame):
	# Normalize prior image to total flux
        zbl = np.max(Obsdata_List[i].unpack(['amp'])['amp'])
        nprior_List[i] = zbl * Prior_List[i].imvec / np.sum(Prior_List[i].imvec)
        logprior_List[i] = np.log(nprior_List[i])    
	# Data
	data = Obsdata_List[i].unpack(['u','v','vis','sigma'])
	uv = np.hstack((data['u'].reshape(-1,1), data['v'].reshape(-1,1)))
	vis_List[i] = data['vis']
	sigma_List[i] = data['sigma']    
	# Compute the Fourier matrix
	A_List[i] = vb.ftmatrix(Prior_List[i].psize, Prior_List[i].xdim, Prior_List[i].ydim, uv)
    
    # Define the objective function and gradient
    def objfunc(logim):
        Frames = np.exp(logim.reshape((-1, N_pixel, N_pixel)))     
        if entropy == "simple":
            s = np.sum(-mx.ssimple(Frames[i].ravel(), nprior_List[i]) for i in range(N_frame))
        elif entropy == "l1":
            s = np.sum(-mx.sl1(Frames[i].ravel(), nprior_List[i]) for i in range(N_frame))
        elif entropy == "gs":
            s = np.sum(-mx.sgs(Frames[i].ravel(), nprior_List[i]) for i in range(N_frame))
        elif entropy == "tv":
            s = np.sum(-mx.stv(Frames[i].ravel(), Prior_List[0].xdim, Prior_List[0].ydim) for i in range(N_frame))

        chisq_total = np.sum(mx.chisq(Frames[i].ravel(), A_List[i], vis_List[i], sigma_List[i]) for i in range(N_frame))/N_frame  
	#print ("chisq_total ",chisq_total)
        s_dynamic = dynamic_regularizer(Frames, gauss)
	#print ("s_dynamic ",s_dynamic)
	#return s_dynamic
        return s + beta * s_dynamic + alpha * (chisq_total - 1.)
        
    def objgrad(logim):
	Frames = np.exp(logim.reshape((-1, N_pixel, N_pixel)))
        if entropy == "simple":
            s = np.concatenate([-mx.ssimplegrad(Frames[i].ravel(), nprior_List[i])*Frames[i].ravel() for i in range(N_frame)])
        elif entropy == "l1":
            s = np.concatenate([-mx.sl1grad(Frames[i].ravel(), nprior_List[i])*Frames[i].ravel() for i in range(N_frame)])
        elif entropy == "gs":
            s = np.concatenate([-mx.sgsgrad(Frames[i].ravel(), nprior_List[i])*Frames[i].ravel() for i in range(N_frame)])
        elif entropy == "tv":
            s = np.concatenate([-mx.stvgrad(Frames[i].ravel(), Prior_List[0].xdim, Prior_List[0].ydim)*Frames[i].ravel() for i in range(N_frame)])
            
	#s = [grad for i in range(N_frame) for grad in -stvgrad(Frames[i].ravel(), Prior_List[0].xdim, Prior_List[0].ydim)*Frames[i].ravel()]
	#s = np.concatenate(s)

	chisq_total_grad = np.concatenate([mx.chisqgrad(Frames[i].ravel(), A_List[i], vis_List[i], sigma_List[i])*Frames[i].ravel()/N_frame for i in range(N_frame)])  

	s_dynamic_grad = dynamic_regularizer_gradient(Frames, gauss)
	#return s_dynamic_grad
        return (s + beta * s_dynamic_grad + alpha * chisq_total_grad) 

    def objgrad_numeric(x):  #This calculates the gradient numerically
	#dx = 10.0**-8
	J0 = objfunc(x)
	Jgrad = np.copy(x)
	for i in range(len(x)):
	    xp = np.copy(x)
	    dx = xp[i]*10.0**-8#+10.0**-8
	    xp[i] = xp[i] + dx
	    J1 = objfunc(xp)
	    Jgrad[i] = (J1-J0)/dx
	return Jgrad

    # Plotting function for each iteration
    global nit
    nit = 0
    def plotcur(logim_step):
        global nit
        nit += 1
	#print "iteration %d" % nit
	#refresh_interval = 100
	if nit%10 == 0:        
		print "iteration %d" % nit
		Frames = np.exp(logim_step.reshape((-1, N_pixel, N_pixel)))
		chi2 = np.sum(mx.chisq(Frames[i].ravel(), A_List[i], vis_List[i], sigma_List[i]) for i in range(N_frame))/N_frame
		s = np.sum(-mx.ssimple(Frames[i].ravel(), nprior_List[i]) for i in range(N_frame))
		s_dynamic = dynamic_regularizer(Frames, gauss)
		print "chi2: %f" % chi2
		print "s: %f" % s
		print "s_dynamic: %f" % s_dynamic
		#plot_i(Frames[0], Prior_List[0], nit, chi2, ipynb=ipynb)
		if nit%refresh_interval == 0:
			plot_i_dynamic(Frames, Prior_List[0], nit, chi2, s, s_dynamic, ipynb=ipynb)        

		if 1==0:
		    numeric_gradient = objgrad_numeric(logim_step)
		    analytic_gradient = objgrad(logim_step)
		    print "Numeric Gradient:"
		    print numeric_gradient 
		    print "Analytic Gradient:"
		    print analytic_gradient
		    print "Max Fractional Difference in gradients:"
		    print max(abs((numeric_gradient - analytic_gradient)/numeric_gradient))
   
    # Plot the prior
    #print(logprior_List.flatten().shape)
    logprior = np.hstack(logprior_List).flatten()
    plotcur(logprior)
        
    # Minimize
    optdict = {'maxiter':maxit, 'ftol':stop, 'maxcor':NHIST} # minimizer params
    tstart = time.time()
    res = opt.minimize(objfunc, logprior, method='L-BFGS-B', jac=objgrad,
                       options=optdict, callback=plotcur)
    tstop = time.time()
    #out = np.exp(res.x)
    Frames = np.exp(res.x.reshape((-1, N_pixel, N_pixel)))

    # Print stats
    print "time: %f s" % (tstop - tstart)
    print "J: %f" % res.fun
    #print "Chi^2: %f" % mx.chisq(out, A, vis, sigma)
    print res.message
    
    #Return Frames
    outim = [vb.Image(Frames[i].reshape(Prior_List[0].ydim, Prior_List[0].xdim), Prior_List[0].psize, Prior_List[0].ra, Prior_List[0].dec, rf=Prior_List[0].rf, source=Prior_List[0].source, mjd=Prior_List[0].mjd) for i in range(N_frame)]

    return outim
    
def plot_i_dynamic(im_List, Prior, nit, chi2, s, s_dynamic, ipynb=False):
    
    plt.ion()
    plt.clf()
    
    for i in range(len(im_List)):
        plt.subplot(1, len(im_List), i+1)	
        plt.imshow(im_List[i].reshape(Prior.ydim,Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian')     
        xticks = vb.ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
        yticks = vb.ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
        plt.xticks(xticks[0], xticks[1])
        plt.yticks(yticks[0], yticks[1])
	if i == 0:
            plt.xlabel('Relative RA ($\mu$as)')
            plt.ylabel('Relative Dec ($\mu$as)')
            plt.title("step: %i  $\chi^2$: %f  $s$: %f  $s_{t}$: %f" % (nit, chi2, s, s_dynamic), fontsize=20)
	else:
            plt.xlabel('')
            plt.ylabel('')
            plt.title('')
	

        plt.draw()

    if ipynb:
        display.clear_output()
        display.display(plt.gcf())
        
def plot_im_List(im_List, ipynb=False):

    plt.title("Test", fontsize=20)    
    plt.ion()
    plt.clf()
    
    Prior = im_List[0]

    for i in range(len(im_List)):
        plt.subplot(1, len(im_List), i+1)	
        plt.imshow(im_List[i].imvec.reshape(Prior.ydim,Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian')     
        xticks = vb.ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
        yticks = vb.ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
        plt.xticks(xticks[0], xticks[1])
        plt.yticks(yticks[0], yticks[1])
	if i == 0:
            plt.xlabel('Relative RA ($\mu$as)')
            plt.ylabel('Relative Dec ($\mu$as)')
	else:
            plt.xlabel('')
            plt.ylabel('')
            plt.title('')
	

        plt.draw()

    if ipynb:
        display.clear_output()
        display.display(plt.gcf())
        
def plot_im_List_Set(im_List_List, ipynb=False):

    plt.ion()
    plt.clf()
    
    Prior = im_List_List[0][0]

    xnum = len(im_List_List[0])
    ynum = len(im_List_List)

    for i in range(xnum*ynum):
        plt.subplot(ynum, xnum, i+1)	
        im = im_List_List[(i-i%xnum)/xnum][i%xnum]
        plt.imshow(im.imvec.reshape(im.ydim,im.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian')     
        xticks = vb.ticks(im.xdim, im.psize/RADPERAS/1e-6)
        yticks = vb.ticks(im.ydim, im.psize/RADPERAS/1e-6)
        plt.xticks(xticks[0], xticks[1])
        plt.yticks(yticks[0], yticks[1])
	if i == 0:
            plt.xlabel('Relative RA ($\mu$as)')
            plt.ylabel('Relative Dec ($\mu$as)')
	else:
            plt.xlabel('')
            plt.ylabel('')
            plt.title('')
	

        plt.draw()

    if ipynb:
        display.clear_output()
        display.display(plt.gcf())



