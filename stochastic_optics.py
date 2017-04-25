# Michael Johnson, 2/15/2017
# See http://adsabs.harvard.edu/abs/2016ApJ...833...74J for details about this module

import sys
import time
import numpy as np
import scipy.optimize as opt
import scipy.ndimage.filters as filt
import scipy.signal
import matplotlib.pyplot as plt
import itertools as it
import vlbi_imaging_utils as vb
import maxen as mx
import pulses
import linearize_energy as le
from IPython import display

import math
import cmath

C = 299792458.0 #m/s, consistent with other modules
DEGREE = np.pi/180.
RADPERAS = DEGREE/3600.
RADPERUAS = RADPERAS/1e6

no_linear_shift = False #flag for whether or not to allow image shift in the phase screen

def Q(qx, qy, wavelength, scatt_alpha, r0_maj, r0_min, POS_ANG, r_in, r_out): #Power spectrum of phase fluctuations
    #x is aligned with the major axis; y is aligned with the minor axis
    wavelengthbar = wavelength/(2.0*np.pi)
    qmin = 2.0*np.pi/r_out
    qmax = 2.0*np.pi/r_in
    #rotate qx and qy as needed
    PA = (90 - vb.POS_ANG) * np.pi/180.0
    qx_rot =  qx*np.cos(PA) + qy*np.sin(PA)
    qy_rot = -qx*np.sin(PA) + qy*np.cos(PA)
    return 2.0**scatt_alpha * np.pi * scatt_alpha * scipy.special.gamma(1.0 + scatt_alpha/2.0)/scipy.special.gamma(1.0 - scatt_alpha/2.0)*wavelengthbar**-2.0*(r0_maj*r0_min)**-(scatt_alpha/2.0) * ( (r0_maj/r0_min)*qx_rot**2.0 + (r0_min/r0_maj)*qy_rot**2.0 + qmin**2.0)**(-(scatt_alpha+2.0)/2.0) * np.exp(-((qx_rot**2.0 + qy_rot**2.0)/qmax**2.0)**0.5)

def Wrapped_Convolve(sig,ker):
    N = sig.shape[0]
    return scipy.signal.fftconvolve(np.pad(sig,((N, N), (N, N)), 'wrap'), np.pad(ker,((N, N), (N, N)), 'constant'),mode='same')[N:(2*N),N:(2*N)]

def Wrapped_Gradient(M):
    G = np.gradient(np.pad(M,((1, 1), (1, 1)), 'wrap'))
    Gx = G[0][1:-1,1:-1]
    Gy = G[1][1:-1,1:-1]
    return (Gx, Gy)

def reverse_array(M):
    N = M.shape[0]
    M_rot = np.copy(M)
    for x in range(N):
        for y in range(N):
            x2 = N - x
            y2 = N - y
            if x2 == N:
                x2 = 0
            if y2 == N:
                y2 = 0
            M_rot[y][x] = M[y2][x2]
    return M_rot
    
def MakeEpsilonScreenFromList(EpsilonList, N):
    epsilon = np.zeros((N,N),dtype=np.complex)
    #There are (N^2-1)/2 real elements followed by (N^2-1)/2 complex elements

    #The first (N-1)/2 are the top row
    N_re = (N*N-1)/2
    i = 0
    for x in range(1,(N+1)/2):
        epsilon[0][x] = EpsilonList[i] + 1j * EpsilonList[i+N_re]
        epsilon[0][N-x] = np.conjugate(epsilon[0][x])
        i=i+1

    #The next N(N-1)/2 are filling the next N rows
    for y in range(1,(N+1)/2):
        for x in range(N):
            epsilon[y][x] = EpsilonList[i] + 1j * EpsilonList[i+N_re]

            x2 = N - x
            y2 = N - y
            if x2 == N:
                x2 = 0
            if y2 == N:
                y2 = 0

            epsilon[y2][x2] = np.conjugate(epsilon[y][x])
            i=i+1    

    if no_linear_shift == True:
        epsilon[0,0] = 0
        epsilon[1,0] = 0
        epsilon[0,1] = 0
        epsilon[-1,0] = 0
        epsilon[0,-1] = 0

    return epsilon

def MakeEpsilonScreen(Nx, Ny, rngseed = 0):
    if rngseed != 0:
        np.random.seed( rngseed )

    epsilon = np.random.normal(loc=0.0, scale=1.0/math.sqrt(2), size=(Nx,Ny)) + 1j * np.random.normal(loc=0.0, scale=1.0/math.sqrt(2), size=(Nx,Ny))
    epsilon[0][0] = 0.0

    #Now let's ensure that it has the necessary conjugation symmetry
    for x in range(Nx):
        if x > (Nx-1)/2:
            epsilon[0][x] = np.conjugate(epsilon[0][Nx-x])
        for y in range((Ny-1)/2, Ny):
            x2 = Nx - x
            y2 = Ny - y
            if x2 == Nx:
                x2 = 0
            if y2 == Ny:
                y2 = 0
            epsilon[y][x] = np.conjugate(epsilon[y2][x2])

    if no_linear_shift == True:
        epsilon[0,0] = 0
        epsilon[1,0] = 0
        epsilon[0,1] = 0
        epsilon[-1,0] = 0
        epsilon[0,-1] = 0

    return epsilon

def MakePhaseScreen(EpsilonScreen, Reference_Image, obs_frequency_Hz=0.0, scatt_alpha=5.0/3.0, r0_maj=0.0, r0_min=0.0, POS_ANG=None, r_in=0.0, r_out=0.0, observer_screen_distance=0.0, source_screen_distance=0.0, Vx_km_per_s=50.0, Vy_km_per_s=0.0, t_hr=0.0, Verbose=False):
    """Create a refractive phase screen from standardized Fourier components 
       All lengths should be specified in centimeters
       If the observing frequency (obs_frequency_Hz) is not specified, then it will be taken to be equal to the frequency of the Reference_Image
       Unspecified scattering parameters are taken to be equal to those of Sgr A*
       scatt_alpha is the power-law index (Kolmogorov is 5/3)
       POS_ANG is the position angle of the major axis in degrees East of North
       Motion of the screen can be handled with the screen velocity {Vx_km_per_s, Vy_km_per_s} and time parameter t_hr
    """
    #Observing wavelength
    if obs_frequency_Hz == 0.0:
        obs_frequency_Hz = Reference_Image.rf

    wavelength = C/obs_frequency_Hz*100.0 #Observing wavelength [cm]
    wavelengthbar = wavelength/(2.0*np.pi) #lambda/(2pi) [cm]

    #Scattering properties
    if observer_screen_distance == 0.0: 
        observer_screen_distance = 8.023*10**21 #Observer-Scattering distance [cm]

    if source_screen_distance == 0.0:
        source_screen_distance = 1.790*10**22 #Source-Scattering distance [cm]    

    if r0_maj == 0.0:
        r0_maj = (wavelength/0.13)**-1.0*3.134*10**8 #Phase coherence length [cm]

    if r0_min == 0.0:
        r0_min = (wavelength/0.13)**-1.0*6.415*10**8 #Phase coherence length [cm]

    if POS_ANG == None:
        POS_ANG = vb.POS_ANG

    if r_in == 0.0:
        r_in = 1000*10**5 #inner scale [cm]

    if r_out == 0.0:
        r_out = 10**20 #outer scale [cm]    

    #Derived parameters
    Mag = observer_screen_distance/source_screen_distance
    rF = (source_screen_distance*observer_screen_distance/(source_screen_distance + observer_screen_distance)*wavelengthbar)**0.5 #Fresnel scale [cm]    
    FOV = Reference_Image.psize * Reference_Image.xdim * observer_screen_distance #Field of view, in cm, at the scattering screen
    Nx = EpsilonScreen.shape[1]
    Ny = EpsilonScreen.shape[0]

    if Nx%2 == 0:
        print "The image dimension should really be odd..."

    #Now we'll calculate the power spectrum for each pixel in Fourier space
    sqrtQ = 1j*np.zeros((Ny,Nx)) #just to get the dimensions correct
    dq = 2.0*np.pi/FOV #this is the spacing in wavenumber

    screen_x_offset_pixels = (Vx_km_per_s * 1.e5) * (t_hr*3600.0) / (FOV/float(Nx))
    screen_y_offset_pixels = (Vy_km_per_s * 1.e5) * (t_hr*3600.0) / (FOV/float(Nx))

    if Verbose and (screen_x_offset_pixels != 0.0 or screen_y_offset_pixels != 0.0):
        print "Screen Offset x (pixels): ",screen_x_offset_pixels
        print "Screen Offset y (pixels): ",screen_y_offset_pixels

    for s in range(0, Nx):
        for t in range(0, Ny):
            s2 = s
            t2 = t
            if s2 > (Nx-1)/2:
                s2 = s2 - Nx
            if t2 > (Ny-1)/2:
                t2 = t2 - Ny 

            sqrtQ[t][s] = Q(dq*s2, dq*t2, wavelength, scatt_alpha, r0_maj, r0_min, POS_ANG, r_in, r_out)**0.5 
            sqrtQ[t][s] *= np.exp(2.0*np.pi*1j*(float(s2)*screen_x_offset_pixels + float(t2)*screen_y_offset_pixels)/float(Nx)) #The exponential term rotates the screen (after a Fourier transform to the image domain). Note that it uses {s2,t2}, which is important to get the conjugation symmetry correct. 
          
    sqrtQ[0][0] = 0.0 #A DC offset doesn't affect scattering

    #Now calculate the phase screen
    phi = np.real(wavelengthbar/FOV*EpsilonScreen.shape[0]*EpsilonScreen.shape[1]*np.fft.ifft2(sqrtQ*EpsilonScreen))
    phi_Image = vb.Image(phi, Reference_Image.psize, Reference_Image.ra, Reference_Image.dec, rf=Reference_Image.rf, source=Reference_Image.source, mjd=Reference_Image.mjd)

    return phi_Image


def Scatter(Unscattered_Image, Epsilon_Screen=np.array([]), obs_frequency_Hz=0.0, scatt_alpha=5.0/3.0, r0_maj=0.0, r0_min=0.0, POS_ANG=None, r_in=0.0, r_out=0.0, observer_screen_distance=0.0, source_screen_distance=0.0, Vx_km_per_s=50.0, Vy_km_per_s=0.0, t_hr=0.0, Linearized_Approximation=True, DisplayPhi=False, DisplayImage=False, Force_Positivity=False, Verbose=False): 
    """Create a refractive phase screen from standardized Fourier components 
       All lengths should be specified in centimeters
       If the observing frequency (obs_frequency_Hz) is not specified, then it will be taken to be equal to the frequency of the Unscattered_Image
       Unspecified scattering parameters are taken to be equal to those of Sgr A*
       scatt_alpha is the power-law index (Kolmogorov is 5/3)
       POS_ANG is the position angle of the major axis in degrees East of North
       Note: an odd image dimension is required!
    """

    #Observing wavelength
    if obs_frequency_Hz == 0.0:
        obs_frequency_Hz = Unscattered_Image.rf

    wavelength = C/obs_frequency_Hz*100.0 #Observing wavelength [cm]
    wavelengthbar = wavelength/(2.0*np.pi) #lambda/(2pi) [cm]

    #Scattering properties
    if observer_screen_distance == 0.0: 
        observer_screen_distance = 8.023*10**21 #Observer-Scattering distance [cm]

    if source_screen_distance == 0.0:
        source_screen_distance = 1.790*10**22 #Source-Scattering distance [cm]    

    if r0_maj == 0.0:
        r0_maj = (wavelength/0.13)**-1.0*3.134*10**8 #Phase coherence length [cm]

    if r0_min == 0.0:
        r0_min = (wavelength/0.13)**-1.0*6.415*10**8 #Phase coherence length [cm]
 
    if POS_ANG == None:
        POS_ANG = vb.POS_ANG

    #Derived parameters
    rF = (source_screen_distance*observer_screen_distance/(source_screen_distance + observer_screen_distance)*wavelengthbar)**0.5 #Fresnel scale [cm]    
    FOV = Unscattered_Image.psize * Unscattered_Image.xdim * observer_screen_distance #Field of view, in cm, at the scattering screen
    Mag = observer_screen_distance/source_screen_distance

    #Optional: Print info
    if Verbose == True:
        print "Major Axis r0/rF: ",(r0_maj/rF)
        print "Minor Axis r0/rF: ",(r0_maj/rF)
        print "Major Axis Ensemble-average FWHM: ",((2.0*np.log(2.0))**0.5/np.pi*wavelength/r0_maj*10**6*3600*180/np.pi)
        print "Minor Axis Ensemble-average FWHM: ",((2.0*np.log(2.0))**0.5/np.pi*wavelength/r0_min*10**6*3600*180/np.pi)

    #First we need to calculate the ensemble-average image by blurring the unscattered image with the correct kernel
    blurring_kernel_params = [ (2.0*np.log(2.0))**0.5/np.pi*wavelength/(r0_maj*(1.0 + Mag)), (2.0*np.log(2.0))**0.5/np.pi*wavelength/(r0_min*(1.0 + Mag)), POS_ANG*np.pi/180.0 ]
    EA_Image = vb.blur_gauss(Unscattered_Image, blurring_kernel_params, frac=1.0, frac_pol=0)
   
    if Epsilon_Screen.shape[0] == 0:
        return EA_Image
    else:
        Nx = Epsilon_Screen.shape[1]
        Ny = Epsilon_Screen.shape[0]

        if Nx%2 == 0:
            print "The image dimension should really be odd..."
    
        #We'll now calculate the phase screen.       
        phi_Image = MakePhaseScreen(Epsilon_Screen, Unscattered_Image, obs_frequency_Hz, scatt_alpha, r0_maj, r0_min, POS_ANG, r_in, r_out, observer_screen_distance, source_screen_distance, Vx_km_per_s=Vx_km_per_s, Vy_km_per_s=Vy_km_per_s, t_hr=t_hr, Verbose=Verbose)
        phi = phi_Image.imvec.reshape(Ny,Nx)
        
        if DisplayPhi:
            phi_Image.display()

        #Next, we need the gradient of the ensemble-average image
        phi_Gradient = Wrapped_Gradient(phi/(FOV/Nx))    
        #The gradient signs don't actually matter, but let's make them match intuition (i.e., right to left, bottom to top)
        phi_Gradient_x = -phi_Gradient[1]
        phi_Gradient_y = -phi_Gradient[0]

        if Linearized_Approximation == True: #Use Equation 10 of Johnson & Narayan (2016)
            #Calculate the gradient of the ensemble-average image
            EA_Gradient = Wrapped_Gradient((EA_Image.imvec/(FOV/Nx)).reshape(EA_Image.ydim, EA_Image.xdim))    
            #The gradient signs don't actually matter, but let's make them match intuition (i.e., right to left, bottom to top)
            EA_Gradient_x = -EA_Gradient[1]
            EA_Gradient_y = -EA_Gradient[0]
            #Now we can patch together the average image
            AI = (EA_Image.imvec).reshape(Ny,Nx) + rF**2.0 * ( EA_Gradient_x*phi_Gradient_x + EA_Gradient_y*phi_Gradient_y )
            if len(Unscattered_Image.qvec):
                    # Scatter the Q image
                    EA_Gradient = Wrapped_Gradient((EA_Image.qvec/(FOV/Nx)).reshape(EA_Image.ydim, EA_Image.xdim)) 
                    EA_Gradient_x = -EA_Gradient[1]
                    EA_Gradient_y = -EA_Gradient[0]
                    AI_Q = (EA_Image.qvec).reshape(Ny,Nx) + rF**2.0 * ( EA_Gradient_x*phi_Gradient_x + EA_Gradient_y*phi_Gradient_y )
                    # Scatter the U image
                    EA_Gradient = Wrapped_Gradient((EA_Image.uvec/(FOV/Nx)).reshape(EA_Image.ydim, EA_Image.xdim)) 
                    EA_Gradient_x = -EA_Gradient[1]
                    EA_Gradient_y = -EA_Gradient[0]
                    AI_U = (EA_Image.uvec).reshape(Ny,Nx) + rF**2.0 * ( EA_Gradient_x*phi_Gradient_x + EA_Gradient_y*phi_Gradient_y )
            if len(Unscattered_Image.vvec):
                    # Scatter the V image
                    EA_Gradient = Wrapped_Gradient((EA_Image.vvec/(FOV/Nx)).reshape(EA_Image.ydim, EA_Image.xdim)) 
                    EA_Gradient_x = -EA_Gradient[1]
                    EA_Gradient_y = -EA_Gradient[0]
                    AI_V = (EA_Image.vvec).reshape(Ny,Nx) + rF**2.0 * ( EA_Gradient_x*phi_Gradient_x + EA_Gradient_y*phi_Gradient_y )
        else: #Use Equation 9 of Johnson & Narayan (2016)
            EA_im = (EA_Image.imvec).reshape(Ny,Nx)
            AI = np.copy((EA_Image.imvec).reshape(Ny,Nx))
            if len(Unscattered_Image.qvec):
                AI_Q = np.copy((EA_Image.imvec).reshape(Ny,Nx))
                AI_U = np.copy((EA_Image.imvec).reshape(Ny,Nx))
                EA_im_Q = (EA_Image.qvec).reshape(Ny,Nx)
                EA_im_U = (EA_Image.uvec).reshape(Ny,Nx)
            if len(Unscattered_Image.vvec):
                AI_V = np.copy((EA_Image.imvec).reshape(Ny,Nx))
                EA_im_V = (EA_Image.vvec).reshape(Ny,Nx)
            for rx in range(Nx):
                for ry in range(Ny):
                    # Annoyingly, the signs here must be negative to match the other approximation. I'm not sure which is correct, but it really shouldn't matter anyway because -phi has the same power spectrum as phi. However, getting the *relative* sign for the x- and y-directions correct is important. 
                    rxp = int(np.round(rx - rF**2.0 * phi_Gradient_x[ry,rx]/observer_screen_distance/Unscattered_Image.psize))%Nx
                    ryp = int(np.round(ry - rF**2.0 * phi_Gradient_y[ry,rx]/observer_screen_distance/Unscattered_Image.psize))%Ny
                    AI[ry,rx] = EA_im[ryp,rxp]   
                    if len(Unscattered_Image.qvec):  
                        AI_Q[ry,rx] = EA_im_Q[ryp,rxp]          
                        AI_U[ry,rx] = EA_im_U[ryp,rxp]      
                    if len(Unscattered_Image.vvec): 
                        AI_V[ry,rx] = EA_im_V[ryp,rxp]

        #Optional: eliminate negative flux
        if Force_Positivity == True: 
           AI = abs(AI)

        #Make it into a proper image format
        AI_Image = vb.Image(AI, EA_Image.psize, EA_Image.ra, EA_Image.dec, rf=EA_Image.rf, source=EA_Image.source, mjd=EA_Image.mjd)
        if len(Unscattered_Image.qvec): 
            AI_Image.add_qu(AI_Q, AI_U)
        if len(Unscattered_Image.vvec):
            AI_Image.add_v(AI_V)

        if DisplayImage:
            plot_scatt(Unscattered_Image.imvec, EA_Image.imvec, AI_Image.imvec, phi_Image.imvec, Unscattered_Image, 0, 0, ipynb=False)

        return AI_Image

##################################################################################################
# Plotting Functions
##################################################################################################
def plot_scatt_dual(im_unscatt1, im_unscatt2, im_scatt1, im_scatt2, im_phase1, im_phase2, Prior, nit, chi2, ipynb=False):
    #plot_scatt_dual(im1, im2, scatt_im1, scatt_im2, phi1, phi2, Prior1, 0, 0, ipynb=False)
    # Get vectors and ratio from current image
    x = np.array([[i for i in range(Prior.xdim)] for j in range(Prior.ydim)])
    y = np.array([[j for i in range(Prior.xdim)] for j in range(Prior.ydim)])
    
    # Create figure and title
    plt.ion()
    plt.clf()
    #plt.suptitle("step: %i  $\chi^2$: %f " % (nit, chi2), fontsize=20)
        
    # Unscattered Image
    plt.subplot(231)
    plt.imshow(im_unscatt1.reshape(Prior.ydim, Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian', vmin=0)
    xticks = vb.ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
    yticks = vb.ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title('Unscattered')
    
    # Scattered
    plt.subplot(232)
    plt.imshow(im_scatt1.reshape(Prior.ydim, Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian', vmin=0)
    xticks = vb.ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
    yticks = vb.ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title('Average Image')
    
    # Phase
    plt.subplot(233)
    plt.imshow(im_phase1.reshape(Prior.ydim, Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian')
    xticks = vb.ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
    yticks = vb.ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title('Phase Screen')

      
    # Unscattered Image
    plt.subplot(234)
    plt.imshow(im_unscatt2.reshape(Prior.ydim, Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian', vmin=0)
    xticks = vb.ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
    yticks = vb.ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title('Unscattered')
    
    # Scattered
    plt.subplot(235)
    plt.imshow(im_scatt2.reshape(Prior.ydim, Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian', vmin=0)
    xticks = vb.ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
    yticks = vb.ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title('Average Image')
    
    # Phase
    plt.subplot(236)
    plt.imshow(im_phase2.reshape(Prior.ydim, Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian')
    xticks = vb.ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
    yticks = vb.ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title('Phase Screen')

    # Display
    plt.draw()
    if ipynb:
        display.clear_output()
        display.display(plt.gcf())  

def plot_scatt(im_unscatt, im_ea, im_scatt, im_phase, Prior, nit, chi2, ipynb=False):
    
    # Get vectors and ratio from current image
    x = np.array([[i for i in range(Prior.xdim)] for j in range(Prior.ydim)])
    y = np.array([[j for i in range(Prior.xdim)] for j in range(Prior.ydim)])
    
    # Create figure and title
    plt.ion()
    plt.clf()
    plt.suptitle("step: %i  $\chi^2$: %f " % (nit, chi2), fontsize=20)
        
    # Unscattered Image
    plt.subplot(141)
    plt.imshow(im_unscatt.reshape(Prior.ydim, Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian', vmin=0)
    xticks = vb.ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
    yticks = vb.ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title('Unscattered')
    

    # Ensemble Average
    plt.subplot(142)
    plt.imshow(im_ea.reshape(Prior.ydim, Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian', vmin=0)
    xticks = vb.ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
    yticks = vb.ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title('Ensemble Average')


    # Scattered
    plt.subplot(143)
    plt.imshow(im_scatt.reshape(Prior.ydim, Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian', vmin=0)
    xticks = vb.ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
    yticks = vb.ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title('Average Image')
    
    # Phase
    plt.subplot(144)
    plt.imshow(im_phase.reshape(Prior.ydim, Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian')
    xticks = vb.ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
    yticks = vb.ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title('Phase Screen')


    # Display
    plt.draw()
    if ipynb:
        display.clear_output()
        display.display(plt.gcf())  

