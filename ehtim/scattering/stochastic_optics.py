# Michael Johnson, 2/15/2017
# See http://adsabs.harvard.edu/abs/2016ApJ...833...74J for details about this module

from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy.signal
import scipy.special as sps
import scipy.integrate as integrate
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from IPython import display

import ehtim.image as image
import ehtim.movie as movie
import ehtim.obsdata as obsdata
from ehtim.observing.obs_helpers import *
from ehtim.const_def import * #Note: C is m/s rather than cm/s.

import math
import cmath

################################################################################
# The class ScatteringModel enscompasses a generic scattering model, determined by the power spectrum Q and phase structure function Dphi
################################################################################

class ScatteringModel(object):
    """A scattering model based on a thin-screen approximation.

       Models include:
           ('von_Mises', 'boxcar', 'dipole'): These scattering models are motivated by observations of Sgr A*.
            Each gives a Gaussian at long wavelengths that matches the model defined
            by {theta_maj_mas_ref, theta_min_mas_ref, POS_ANG} at the reference wavelength wavelength_reference_cm
            with a lambda^2 scaling. The source sizes {theta_maj, theta_min} are the image FWHM in milliarcseconds
            at the reference wavelength. Note that this may not match the ensemble-average kernel at the reference wavelength,
            if the reference wavelength is short enough to be beyond the lambda^2 regime!
            This model also includes an inner and outer scale and will thus transition to scattering with scatt_alpha at shorter wavelengths
            Note: This model *requires* a finite inner scale
           'power-law': This scattering model gives a pure power law at all wavelengths. There is no inner scale, but there can be an outer scale.
            The ensemble-average image is given by {theta_maj_mas_ref, theta_min_mas_ref, POS_ANG} at the reference wavelength wavelength_reference_cm.
            The ensemble-average image size is proportional to wavelength^(1+2/scatt_alpha) = wavelength^(11/5) for Kolmogorov

       Attributes:
            model (string): The type of scattering model (determined by the power spectrum of phase fluctuations).
            scatt_alpha (float): The power-law index of the phase fluctuations (Kolmogorov is 5/3).
            observer_screen_distance (float): The distance from the observer to the scattering screen in cm.
            source_screen_distance (float): The distance from the source to the scattering screen in cm.
            theta_maj_mas_ref (float): FWHM in mas of the major axis angular broadening at the specified reference wavelength.
            theta_min_mas_ref (float): FWHM in mas of the minor axis angular broadening at the specified reference wavelength.
            POS_ANG (float): The position angle of the major axis of the scattering.
            wavelength_reference_cm (float): The reference wavelength for the scattering model in cm.
            r_in (float): The inner scale of the scattering screen in cm.
            r_out (float): The outer scale of the scattering screen in cm.
            rF (function): The Fresnel scale of the scattering screen at the specific wavelength.
    """

    def __init__(self, model = 'dipole', scatt_alpha = 1.38, observer_screen_distance = 2.82 * 3.086e21, source_screen_distance = 5.53 * 3.086e21, theta_maj_mas_ref = 1.380, theta_min_mas_ref = 0.703, POS_ANG = 81.9, wavelength_reference_cm = 1.0, r_in = 800e5, r_out = 1e20):
        """To initialize the scattering model, specify:

           Attributes:
                model (string): The type of scattering model (determined by the power spectrum of phase fluctuations). Options are 'von_Mises', 'boxcar', 'dipole', and 'power-law'
                scatt_alpha (float): The power-law index of the phase fluctuations (Kolmogorov is 5/3).
                observer_screen_distance (float): The distance from the observer to the scattering screen in cm.
                source_screen_distance (float): The distance from the source to the scattering screen in cm.
                theta_maj_mas_ref (float): FWHM in mas of the major axis angular broadening at the specified reference wavelength.
                theta_min_mas_ref (float): FWHM in mas of the minor axis angular broadening at the specified reference wavelength.
                POS_ANG (float): The position angle of the major axis of the scattering.
                wavelength_reference_cm (float): The reference wavelength for the scattering model in cm.
                r_in (float): The inner scale of the scattering screen in cm.
                r_out (float): The outer scale of the scattering screen in cm.
        """

        self.model = model
        self.POS_ANG = POS_ANG #Major axis position angle [degrees, east of north]
        self.observer_screen_distance = observer_screen_distance #cm
        self.source_screen_distance   = source_screen_distance   #cm
        M = observer_screen_distance/source_screen_distance
        self.wavelength_reference = wavelength_reference_cm #Reference wavelength [cm]
        self.r_in    = r_in #inner scale [cm]
        self.r_out   = r_out     #outer scale [cm]
        self.scatt_alpha = scatt_alpha

        FWHM_fac = (2.0 * np.log(2.0))**0.5/np.pi
        self.Qbar = 2.0/sps.gamma((2.0 - self.scatt_alpha)/2.0) * (self.r_in**2*(1.0 + M)/(FWHM_fac*(self.wavelength_reference/(2.0*np.pi))**2) )**2 * ( (theta_maj_mas_ref**2 + theta_min_mas_ref**2)*(1.0/1000.0/3600.0*np.pi/180.0)**2)
        self.C_scatt_0 = (self.wavelength_reference/(2.0*np.pi))**2 * self.Qbar*sps.gamma(1.0 - self.scatt_alpha/2.0)/(8.0*np.pi**2*self.r_in**2)
        A = theta_maj_mas_ref/theta_min_mas_ref # Anisotropy, >=1, as lambda->infinity 
        self.phi0 = (90 - self.POS_ANG) * np.pi/180.0

        # Parameters for the approximate phase structure function 
        theta_maj_rad_ref = theta_maj_mas_ref/1000.0/3600.0*np.pi/180.0
        theta_min_rad_ref = theta_min_mas_ref/1000.0/3600.0*np.pi/180.0
        self.Amaj_0 = ( self.r_in*(1.0 + M) * theta_maj_rad_ref/(FWHM_fac * (self.wavelength_reference/(2.0*np.pi)) * 2.0*np.pi ))**2
        self.Amin_0 = ( self.r_in*(1.0 + M) * theta_min_rad_ref/(FWHM_fac * (self.wavelength_reference/(2.0*np.pi)) * 2.0*np.pi ))**2

        if model == 'von_Mises':
            def avM_Anisotropy(kzeta):                
                return np.abs( (kzeta*sps.i0(kzeta)/sps.i1(kzeta) - 1.0)**0.5 - A )

            self.kzeta = minimize(avM_Anisotropy, A**2, method='nelder-mead', options={'xtol': 1e-8, 'disp': False}).x
            self.P_phi_prefac = 1.0/(2.0*np.pi*sps.i0(self.kzeta))  
        elif model == 'boxcar':
            def boxcar_Anisotropy(kzeta):                
                return np.abs( np.sin(np.pi/(1.0 + kzeta))/(np.pi/(1.0 + kzeta)) - (theta_maj_mas_ref**2 - theta_min_mas_ref**2)/(theta_maj_mas_ref**2 + theta_min_mas_ref**2) )       

            self.kzeta = minimize(boxcar_Anisotropy, A, method='nelder-mead', options={'xtol': 1e-8, 'disp': False}).x
            self.P_phi_prefac = (1.0 + self.kzeta)/(2.0*np.pi)   
        elif model == 'dipole':
            def dipole_Anisotropy(kzeta):                
                return np.abs( sps.hyp2f1((self.scatt_alpha + 2.0)/2.0, 0.5, 2.0, -kzeta)/sps.hyp2f1((self.scatt_alpha + 2.0)/2.0, 1.5, 2.0, -kzeta) - A**2 )  

            self.kzeta = minimize(dipole_Anisotropy, A, method='nelder-mead', options={'xtol': 1e-8, 'disp': False}).x
            self.P_phi_prefac = 1.0/(2.0*np.pi*sps.hyp2f1((self.scatt_alpha + 2.0)/2.0, 0.5, 1.0, -self.kzeta))       
        else:
            print("Scattering Model Not Recognized!")

        # More parameters for the approximate phase structure function 
        int_maj = integrate.quad(lambda phi_q: np.abs( np.cos( self.phi0 - phi_q ) )**self.scatt_alpha * self.P_phi(phi_q), 0, 2.0*np.pi, limit=250)[0]  
        int_min = integrate.quad(lambda phi_q: np.abs( np.sin( self.phi0 - phi_q ) )**self.scatt_alpha * self.P_phi(phi_q), 0, 2.0*np.pi, limit=250)[0]      
        B_prefac = self.C_scatt_0 * 2.0**(2.0 - self.scatt_alpha) * np.pi**0.5/(self.scatt_alpha * sps.gamma((self.scatt_alpha + 1.0)/2.0))
        self.Bmaj_0 = B_prefac*int_maj
        self.Bmin_0 = B_prefac*int_min

        #Check normalization:
        #print("Checking Normalization:",integrate.quad(lambda phi_q: self.P_phi(phi_q), 0, 2.0*np.pi)[0])  

        return

    def P_phi(self, phi):
        if self.model == 'von_Mises':
            return self.P_phi_prefac * np.cosh(self.kzeta*np.cos(phi - self.phi0))  
        elif self.model == 'boxcar':
            return self.P_phi_prefac * (1.0 - ((np.pi/(2.0*(1.0 + self.kzeta)) < (phi - self.phi0) % np.pi) & ((phi - self.phi0) % np.pi < np.pi - np.pi/(2.0*(1.0 + self.kzeta)))))
        elif self.model == 'dipole':
            return self.P_phi_prefac * (1.0 + self.kzeta*np.sin(phi - self.phi0)**2)**(-(self.scatt_alpha + 2.0)/2.0)

    def rF(self, wavelength):
        """Returns the Fresnel scale [cm] of the scattering screen at the specified wavelength [cm].

           Args:
               wavelength (float): The desired wavelength [cm]

           Returns:
               rF (float): The Fresnel scale [cm]
        """
        return (self.source_screen_distance*self.observer_screen_distance/(self.source_screen_distance + self.observer_screen_distance)*wavelength/(2.0*np.pi))**0.5

    def Mag(self):
        """Returns the effective magnification the scattering screen: (observer-screen distance)/(source-screen distance).

           Returns:
               M (float): The effective magnification of the scattering screen.
        """
        return self.observer_screen_distance/self.source_screen_distance

    def dDphi_dz(self, r, phi, phi_q, wavelength):
        """differential contribution to the phase structure function
        """
        return 4.0 * (wavelength/self.wavelength_reference)**2 * self.C_scatt_0/self.scatt_alpha * (sps.hyp1f1(-self.scatt_alpha/2.0, 0.5, -r**2/(4.0*self.r_in**2)*np.cos(phi - phi_q)**2) - 1.0)

    def Dphi_exact(self, x, y, wavelength_cm):
        r = (x**2 + y**2)**0.5
        phi = np.arctan2(y, x)

        return integrate.quad(lambda phi_q: self.dDphi_dz(r, phi, phi_q, wavelength_cm)*self.P_phi(phi_q), 0, 2.0*np.pi)[0]        

    def Dmaj(self, r, wavelength_cm):
        return (wavelength_cm/self.wavelength_reference)**2 * self.Bmaj_0 * (2.0 * self.Amaj_0/(self.scatt_alpha * self.Bmaj_0))**(-self.scatt_alpha/(2.0 - self.scatt_alpha)) * ((1.0 + (2.0*self.Amaj_0/(self.scatt_alpha * self.Bmaj_0))**(2.0/(2.0 - self.scatt_alpha)) * (r/self.r_in)**2 )**(self.scatt_alpha/2.0) - 1.0)

    def Dmin(self, r, wavelength_cm):
        return (wavelength_cm/self.wavelength_reference)**2 * self.Bmin_0 * (2.0 * self.Amin_0/(self.scatt_alpha * self.Bmin_0))**(-self.scatt_alpha/(2.0 - self.scatt_alpha)) * ((1.0 + (2.0*self.Amin_0/(self.scatt_alpha * self.Bmin_0))**(2.0/(2.0 - self.scatt_alpha)) * (r/self.r_in)**2 )**(self.scatt_alpha/2.0) - 1.0)

    def Dphi_approx(self, x, y, wavelength_cm):
        r = (x**2 + y**2)**0.5
        phi = np.arctan2(y, x)

        Dmaj_eval = self.Dmaj(r, wavelength_cm)
        Dmin_eval = self.Dmin(r, wavelength_cm)

        return (Dmaj_eval + Dmin_eval)/2.0 + (Dmaj_eval - Dmin_eval)/2.0*np.cos(2.0*(phi - self.phi0))

    def Q(self, qx, qy):
        """Computes the power spectrum of the scattering model at a wavenumber {qx,qy} (in 1/cm).
        The power spectrum is part of what defines the scattering model (along with Dphi).
        Q(qx,qy) is independent of the observing wavelength.

        Args:
            qx (float): x coordinate of the wavenumber in 1/cm.
            qy (float): y coordinate of the wavenumber in 1/cm.
        Returns:
            (float): The power spectrum Q(qx,qy)
        """

        q = (qx**2 + qy**2)**0.5 + 1e-12/self.r_in #Add a small offset to avoid division by zero
        phi_q = np.arctan2(qy, qx)

        return self.Qbar * (q*self.r_in)**(-(self.scatt_alpha + 2.0)) * np.exp(-(q * self.r_in)**2) * self.P_phi(phi_q)


    def sqrtQ_Matrix(self, Reference_Image, Vx_km_per_s=50.0, Vy_km_per_s=0.0, t_hr=0.0):
        """Computes the square root of the power spectrum on a discrete grid. Because translation of the screen is done most conveniently in Fourier space, a screen translation can also be included.

           Args:
                Reference_Image (Image): Reference image to determine image and pixel dimensions and wavelength.
                Vx_km_per_s (float): Velocity of the scattering screen in the x direction (toward East) in km/s.
                Vy_km_per_s (float): Velocity of the scattering screen in the y direction (toward North) in km/s.
                t_hr (float): The current time of the scattering in hours.
           Returns:
               sqrtQ (2D complex ndarray): The square root of the power spectrum of the screen with an additional phase for rotation of the screen.
            """

        #Derived parameters
        FOV = Reference_Image.psize * Reference_Image.xdim * self.observer_screen_distance #Field of view, in cm, at the scattering screen
        N = Reference_Image.xdim
        dq = 2.0*np.pi/FOV #this is the spacing in wavenumber
        screen_x_offset_pixels = (Vx_km_per_s * 1.e5) * (t_hr*3600.0) / (FOV/float(N))
        screen_y_offset_pixels = (Vy_km_per_s * 1.e5) * (t_hr*3600.0) / (FOV/float(N))

        s, t = np.meshgrid(np.fft.fftfreq(N, d=1.0/N), np.fft.fftfreq(N, d=1.0/N))
        sqrtQ = np.sqrt(self.Q(dq*s, dq*t)) * np.exp(2.0*np.pi*1j*(s*screen_x_offset_pixels +
                                                                   t*screen_y_offset_pixels)/float(N))
        sqrtQ[0][0] = 0.0 #A DC offset doesn't affect scattering

        return sqrtQ

    def Ensemble_Average_Kernel(self, Reference_Image, wavelength_cm = None, use_approximate_form=True):
        """The ensemble-average convolution kernel for images; returns a 2D array corresponding to the image dimensions of the reference image

           Args:
                Reference_Image (Image): Reference image to determine image and pixel dimensions and wavelength.
                wavelength_cm (float): The observing wavelength for the scattering kernel in cm. If unspecified, this will default to the wavelength of the Reference image.

           Returns:
               ker (2D ndarray): The ensemble-average scattering kernel in the image domain.
            """

        if wavelength_cm == None:
            wavelength_cm = C/Reference_Image.rf*100.0 #Observing wavelength [cm]

        uvlist = np.fft.fftfreq(Reference_Image.xdim)/Reference_Image.psize # assume square kernel.  FIXME: create ulist and vlist, and construct u_grid and v_grid with the correct dimension
        if use_approximate_form == True:
            u_grid, v_grid = np.meshgrid(uvlist, uvlist)
            ker_uv = self.Ensemble_Average_Kernel_Visibility(u_grid, v_grid, wavelength_cm, use_approximate_form=use_approximate_form)
        else:
            ker_uv = np.array([[self.Ensemble_Average_Kernel_Visibility(u, v, wavelength_cm, use_approximate_form=use_approximate_form) for u in uvlist] for v in uvlist]) 

        ker = np.real(np.fft.fftshift(np.fft.fft2(ker_uv)))
        ker = ker / np.sum(ker) # normalize to 1
        return ker

    def Ensemble_Average_Kernel_Visibility(self, u, v, wavelength_cm, use_approximate_form=True):
        """The ensemble-average multiplicative scattering kernel for visibilities at a particular {u,v} coordinate

           Args:
                u (float): u baseline coordinate (dimensionless)
                v (float): v baseline coordinate (dimensionless)
                wavelength_cm (float): The observing wavelength for the scattering kernel in cm.

           Returns:
               float: The ensemble-average kernel at the specified {u,v} point and observing wavelength.
            """
        if use_approximate_form == True:
            return np.exp(-0.5*self.Dphi_approx(u*wavelength_cm/(1.0+self.Mag()), v*wavelength_cm/(1.0+self.Mag()), wavelength_cm))
        else:
            return np.exp(-0.5*self.Dphi_exact(u*wavelength_cm/(1.0+self.Mag()), v*wavelength_cm/(1.0+self.Mag()), wavelength_cm))

    def Ensemble_Average_Blur(self, im, wavelength_cm = None, ker = None, use_approximate_form=True):
        """Blurs an input Image with the ensemble-average scattering kernel.

           Args:
                im (Image): The unscattered image.
                wavelength_cm (float): The observing wavelength for the scattering kernel in cm. If unspecified, this will default to the wavelength of the input image.
                ker (2D ndarray): The user can optionally pass a pre-computed ensemble-average blurring kernel.

           Returns:
               out (Image): The ensemble-average scattered image.
            """

        # Inputs an unscattered image and an ensemble-average blurring kernel (2D array); returns the ensemble-average image
        # The pre-computed kernel can optionally be specified (ker)

        if wavelength_cm == None:
            wavelength_cm = C/im.rf*100.0 #Observing wavelength [cm]

        if ker is None:
            ker = self.Ensemble_Average_Kernel(im, wavelength_cm, use_approximate_form)

        Iim = Wrapped_Convolve((im.imvec).reshape(im.ydim, im.xdim), ker)
        out = image.Image(Iim, im.psize, im.ra, im.dec, rf=C/(wavelength_cm/100.0), source=im.source, mjd=im.mjd, pulse=im.pulse)
        if len(im.qvec):
            Qim = Wrapped_Convolve((im.qvec).reshape(im.ydim, im.xdim), ker)
            Uim = Wrapped_Convolve((im.uvec).reshape(im.ydim, im.xdim), ker)
            out.add_qu(Qim, Uim)
        if len(im.vvec):
            Vim = Wrapped_Convolve((im.vvec).reshape(im.ydim, im.xdim), ker)
            out.add_v(Vim)

        return out

    def Deblur_obs(self, obs, use_approximate_form=True):
        """Deblurs the observation obs by dividing visibilities by the ensemble-average scattering kernel. See Fish et al. (2014): arXiv:1409.4690.

           Args:
                obs (Obsdata): The observervation data (including scattering).

           Returns:
               obsdeblur (Obsdata): The deblurred observation.
            """

        # make a copy of observation data
        datatable = (obs.copy()).data

        vis = datatable['vis']
        qvis = datatable['qvis']
        uvis = datatable['uvis']
        vvis = datatable['vvis']
        sigma = datatable['sigma']
        qsigma = datatable['qsigma']
        usigma = datatable['usigma']
        vsigma = datatable['vsigma']
        u = datatable['u']
        v = datatable['v']

        # divide visibilities by the scattering kernel
        for i in range(len(vis)):
            ker = self.Ensemble_Average_Kernel_Visibility(u[i], v[i], wavelength_cm = C/obs.rf*100.0, use_approximate_form=use_approximate_form)
            vis[i] = vis[i] / ker
            qvis[i] = qvis[i] / ker
            uvis[i] = uvis[i] / ker
            vvis[i] = vvis[i] / ker
            sigma[i] = sigma[i] / ker
            qsigma[i] = qsigma[i] / ker
            usigma[i] = usigma[i] / ker
            vsigma[i] = vsigma[i] / ker

        datatable['vis'] = vis
        datatable['qvis'] = qvis
        datatable['uvis'] = uvis
        datatable['vvis'] = vvis
        datatable['sigma'] = sigma
        datatable['qsigma'] = qsigma
        datatable['usigma'] = usigma
        datatable['vsigma'] = vsigma

        obsdeblur = obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, datatable, obs.tarr, source=obs.source, mjd=obs.mjd,
                            ampcal=obs.ampcal, phasecal=obs.phasecal, opacitycal=obs.opacitycal, dcal=obs.dcal, frcal=obs.frcal)
        return obsdeblur

    def MakePhaseScreen(self, EpsilonScreen, Reference_Image, obs_frequency_Hz=0.0, Vx_km_per_s=50.0, Vy_km_per_s=0.0, t_hr=0.0, sqrtQ_init=None):
        """Create a refractive phase screen from standardized Fourier components (the EpsilonScreen).
           All lengths should be specified in centimeters
           If the observing frequency (obs_frequency_Hz) is not specified, then it will be taken to be equal to the frequency of the Reference_Image
           Note: an odd image dimension is required!

           Args:
                EpsilonScreen (2D ndarray): Optionally, the scattering screen can be specified. If none is given, a random one will be generated.
                Reference_Image (Image): The reference image.
                obs_frequency_Hz (float): The observing frequency, in Hz. By default, it will be taken to be equal to the frequency of the Unscattered_Image.
                Vx_km_per_s (float): Velocity of the scattering screen in the x direction (toward East) in km/s.
                Vy_km_per_s (float): Velocity of the scattering screen in the y direction (toward North) in km/s.
                t_hr (float): The current time of the scattering in hours.
                ea_ker (2D ndarray): The used can optionally pass a precomputed array of the ensemble-average blurring kernel.
                sqrtQ_init (2D ndarray): The used can optionally pass a precomputed array of the square root of the power spectrum.

           Returns:
               phi_Image (Image): The phase screen.
            """

        #Observing wavelength
        if obs_frequency_Hz == 0.0:
            obs_frequency_Hz = Reference_Image.rf

        wavelength = C/obs_frequency_Hz*100.0 #Observing wavelength [cm]
        wavelengthbar = wavelength/(2.0*np.pi) #lambda/(2pi) [cm]

        #Derived parameters
        FOV = Reference_Image.psize * Reference_Image.xdim * self.observer_screen_distance #Field of view, in cm, at the scattering screen
        rF  = self.rF(wavelength)
        Nx = EpsilonScreen.shape[1]
        Ny = EpsilonScreen.shape[0]

        if Nx%2 == 0:
            print("The image dimension should really be odd...")

        #Now we'll calculate the power spectrum for each pixel in Fourier space
        screen_x_offset_pixels = (Vx_km_per_s*1.e5) * (t_hr*3600.0) / (FOV/float(Nx))
        screen_y_offset_pixels = (Vy_km_per_s*1.e5) * (t_hr*3600.0) / (FOV/float(Nx))

        if sqrtQ_init is None:
            sqrtQ = self.sqrtQ_Matrix(Reference_Image, Vx_km_per_s=Vx_km_per_s, Vy_km_per_s=Vy_km_per_s, t_hr=t_hr)
        else:
            #If a matrix for sqrtQ_init is passed, we still potentially need to rotate it

            if screen_x_offset_pixels != 0.0 or screen_y_offset_pixels != 0.0:
                s, t = np.meshgrid(np.fft.fftfreq(Nx, d=1.0/Nx), np.fft.fftfreq(Ny, d=1.0/Ny))
                sqrtQ = sqrtQ_init * np.exp(2.0*np.pi*1j*(s*screen_x_offset_pixels +
                                                          t*screen_y_offset_pixels)/float(Nx))
            else:
                sqrtQ = sqrtQ_init

        #Now calculate the phase screen
        phi = np.real(wavelengthbar/FOV*EpsilonScreen.shape[0]*EpsilonScreen.shape[1]*np.fft.ifft2(sqrtQ*EpsilonScreen))
        phi_Image = image.Image(phi, Reference_Image.psize, Reference_Image.ra, Reference_Image.dec, rf=Reference_Image.rf, source=Reference_Image.source, mjd=Reference_Image.mjd)

        return phi_Image

    def Scatter(self, Unscattered_Image, Epsilon_Screen=np.array([]), obs_frequency_Hz=0.0, Vx_km_per_s=50.0, Vy_km_per_s=0.0, t_hr=0.0, ea_ker=None, sqrtQ=None, Linearized_Approximation=False, DisplayImage=False, Force_Positivity=False, use_approximate_form=True):
        """Scatter an image using the specified epsilon screen.
           All lengths should be specified in centimeters
           If the observing frequency (obs_frequency_Hz) is not specified, then it will be taken to be equal to the frequency of the Unscattered_Image
           Note: an odd image dimension is required!

           Args:
                Unscattered_Image (Image): The unscattered image.
                Epsilon_Screen (2D ndarray): Optionally, the scattering screen can be specified. If none is given, a random one will be generated.
                obs_frequency_Hz (float): The observing frequency, in Hz. By default, it will be taken to be equal to the frequency of the Unscattered_Image.
                Vx_km_per_s (float): Velocity of the scattering screen in the x direction (toward East) in km/s.
                Vy_km_per_s (float): Velocity of the scattering screen in the y direction (toward North) in km/s.
                t_hr (float): The current time of the scattering in hours.
                ea_ker (2D ndarray): The used can optionally pass a precomputed array of the ensemble-average blurring kernel.
                sqrtQ (2D ndarray): The used can optionally pass a precomputed array of the square root of the power spectrum.
                Linearized_Approximation (bool): If True, uses a linearized approximation for the scattering (Eq. 10 of Johnson & Narayan 2016). If False, uses Eq. 9 of that paper.
                DisplayImage (bool): If True, show a plot of the unscattered, ensemble-average, and scattered images as well as the phase screen.
                Force_Positivity (bool): If True, eliminates negative flux from the scattered image from the linearized approximation.
                Return_Image_List (bool): If True, returns a list of the scattered frames. If False, returns a movie object.

           Returns:
               AI_Image (Image): The scattered image.
            """

        #Observing wavelength
        if obs_frequency_Hz == 0.0:
            obs_frequency_Hz = Unscattered_Image.rf

        wavelength = C/obs_frequency_Hz*100.0 #Observing wavelength [cm]
        wavelengthbar = wavelength/(2.0*np.pi) #lambda/(2pi) [cm]

        #Derived parameters
        FOV = Unscattered_Image.psize * Unscattered_Image.xdim * self.observer_screen_distance #Field of view, in cm, at the scattering screen
        rF  = self.rF(wavelength)
        Nx = Unscattered_Image.xdim
        Ny = Unscattered_Image.ydim

        #First we need to calculate the ensemble-average image by blurring the unscattered image with the correct kernel
        EA_Image = self.Ensemble_Average_Blur(Unscattered_Image, wavelength, ker = ea_ker, use_approximate_form=use_approximate_form)

        # If no epsilon screen is specified, then generate a random realization
        if Epsilon_Screen.shape[0] == 0:
            Epsilon_Screen = MakeEpsilonScreen(Nx, Ny)

        #We'll now calculate the phase screen.
        phi_Image = self.MakePhaseScreen(Epsilon_Screen, Unscattered_Image, obs_frequency_Hz, Vx_km_per_s=Vx_km_per_s, Vy_km_per_s=Vy_km_per_s, t_hr=t_hr, sqrtQ_init=sqrtQ)
        phi = phi_Image.imvec.reshape(Ny,Nx)

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
                    rxp = int(np.round(rx - rF**2.0 * phi_Gradient_x[ry,rx]/self.observer_screen_distance/Unscattered_Image.psize))%Nx
                    ryp = int(np.round(ry - rF**2.0 * phi_Gradient_y[ry,rx]/self.observer_screen_distance/Unscattered_Image.psize))%Ny
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
        AI_Image = image.Image(AI, EA_Image.psize, EA_Image.ra, EA_Image.dec, rf=EA_Image.rf, source=EA_Image.source, mjd=EA_Image.mjd)
        if len(Unscattered_Image.qvec):
            AI_Image.add_qu(AI_Q, AI_U)
        if len(Unscattered_Image.vvec):
            AI_Image.add_v(AI_V)

        if DisplayImage:
            plot_scatt(Unscattered_Image.imvec, EA_Image.imvec, AI_Image.imvec, phi_Image.imvec, Unscattered_Image, 0, 0, ipynb=False)

        return AI_Image

    def Scatter_Movie(self, Unscattered_Movie, Epsilon_Screen=np.array([]), obs_frequency_Hz=0.0, Vx_km_per_s=50.0, Vy_km_per_s=0.0, framedur_sec=None, N_frames = None, sqrtQ=None, Linearized_Approximation=False, Force_Positivity=False,Return_Image_List=False):
        """Scatter a movie using the specified epsilon screen. The movie can either be a movie object, an image list, or a static image
           If scattering a list of images or static image, the frame duration in seconds (framedur_sec) must be specified
           If scattering a static image, the total number of frames must be specified (N_frames)
           All lengths should be specified in centimeters
           If the observing frequency (obs_frequency_Hz) is not specified, then it will be taken to be equal to the frequency of the Unscattered_Movie
           Note: an odd image dimension is required!

           Args:
                Unscattered_Movie: This can be a movie object, an image list, or a static image
                Epsilon_Screen (2D ndarray): Optionally, the scattering screen can be specified. If none is given, a random one will be generated.
                obs_frequency_Hz (float): The observing frequency, in Hz. By default, it will be taken to be equal to the frequency of the Unscattered_Movie.
                Vx_km_per_s (float): Velocity of the scattering screen in the x direction (toward East) in km/s.
                Vy_km_per_s (float): Velocity of the scattering screen in the y direction (toward North) in km/s.
                framedur_sec (float): Duration of each frame, in seconds. Only needed if Unscattered_Movie is not a movie object.
                N_frames (int): Total number of frames. Only needed if Unscattered_Movie is a static image object.
                sqrtQ (2D ndarray): The used can optionally pass a precomputed array of the square root of the power spectrum.
                Linearized_Approximation (bool): If True, uses a linearized approximation for the scattering (Eq. 10 of Johnson & Narayan 2016). If False, uses Eq. 9 of that paper.
                Force_Positivity (bool): If True, eliminates negative flux from the scattered image from the linearized approximation.
                Return_Image_List (bool): If True, returns a list of the scattered frames. If False, returns a movie object.

           Returns:
               Scattered_Movie: Either a movie object or a list of images, depending on the flag Return_Image_List.
            """

        if type(Unscattered_Movie) != movie.Movie and framedur_sec == None:
            print("If scattering a list of images of static image, the framedur must be specified!")
            return

        if type(Unscattered_Movie) == image.Image and N_frames == None:
            print("If scattering a static image, the total number of frames must be specified (N_frames)!")
            return

        if framedur_sec == None:
            framedur_sec = Unscattered_Movie.framedur

        print("Frame Duration (seconds):",framedur_sec)

        if type(Unscattered_Movie) == movie.Movie:
            N = Unscattered_Movie.xdim
            N_frames = len(Unscattered_Movie.frames)
            psize = Unscattered_Movie.psize
            ra = Unscattered_Movie.ra
            dec = Unscattered_Movie.dec
            rf = Unscattered_Movie.rf
            pulse=Unscattered_Movie.pulse
            source=Unscattered_Movie.source
            mjd=Unscattered_Movie.mjd
            start_hr=Unscattered_Movie.start_hr
            has_pol = len(Unscattered_Movie.qframes)
        elif type(Unscattered_Movie) == list:
            N = Unscattered_Movie[0].xdim
            N_frames = len(Unscattered_Movie)
            psize = Unscattered_Movie[0].psize
            ra = Unscattered_Movie[0].ra
            dec = Unscattered_Movie[0].dec
            rf = Unscattered_Movie[0].rf
            pulse=Unscattered_Movie[0].pulse
            source=Unscattered_Movie[0].source
            mjd=Unscattered_Movie[0].mjd
            start_hr=0.0
            has_pol = len(Unscattered_Movie[0].qvec)
        else:
            N = Unscattered_Movie.xdim
            psize = Unscattered_Movie.psize
            ra = Unscattered_Movie.ra
            dec = Unscattered_Movie.dec
            rf = Unscattered_Movie.rf
            pulse=Unscattered_Movie.pulse
            source=Unscattered_Movie.source
            mjd=Unscattered_Movie.mjd
            start_hr=0.0
            has_pol = len(Unscattered_Movie.qvec)

        def get_frame(j):
            if type(Unscattered_Movie) == movie.Movie:
                im = image.Image(Unscattered_Movie.frames[j].reshape((N,N)), psize, ra, dec, rf, pulse, source, mjd)
                if len(Unscattered_Movie.qframes) > 0:
                    im.add_qu(Unscattered_Movie.qframes[j].reshape((N,N)), Unscattered_Movie.uframes[j].reshape((N,N)))

                return im
            elif type(Unscattered_Movie) == list:
                return Unscattered_Movie[j]
            else:
                return Unscattered_Movie

        #If it isn't specified, calculate the matrix sqrtQ for efficiency
        if sqrtQ is None:
            sqrtQ = self.sqrtQ_Matrix(get_frame(0))

        # If no epsilon screen is specified, then generate a random realization
        if Epsilon_Screen.shape[0] == 0:
            Epsilon_Screen = MakeEpsilonScreen(N, N)

        scattered_im_List = [ self.Scatter(get_frame(j), Epsilon_Screen, obs_frequency_Hz = obs_frequency_Hz, Vx_km_per_s = Vx_km_per_s, Vy_km_per_s = Vy_km_per_s, t_hr=framedur_sec/3600.0*j, sqrtQ=sqrtQ, Linearized_Approximation=Linearized_Approximation, Force_Positivity=Force_Positivity) for j in range(N_frames)]

        if Return_Image_List == True:
            return scattered_im_List

        Scattered_Movie = movie.Movie( [im.imvec.reshape((im.xdim,im.ydim)) for im in scattered_im_List], framedur = framedur_sec, psize = psize, ra = ra, dec = dec, rf=rf, pulse=pulse, source=source, mjd=mjd, start_hr=start_hr)

        if has_pol:
            Scattered_Movie_Q = [im.qvec.reshape((im.xdim,im.ydim)) for im in scattered_im_List]
            Scattered_Movie_U = [im.uvec.reshape((im.xdim,im.ydim)) for im in scattered_im_List]
            Scattered_Movie.add_qu(Scattered_Movie_Q, Scattered_Movie_U)

        return Scattered_Movie

################################################################################
# These are helper functions
################################################################################

def Wrapped_Convolve(sig,ker):
    N = sig.shape[0]
    return scipy.signal.fftconvolve(np.pad(sig,((N, N), (N, N)), 'wrap'), np.pad(ker,((N, N), (N, N)), 'constant'),mode='same')[N:(2*N),N:(2*N)]

def Wrapped_Gradient(M):
    G = np.gradient(np.pad(M,((1, 1), (1, 1)), 'wrap'))
    Gx = G[0][1:-1,1:-1]
    Gy = G[1][1:-1,1:-1]
    return (Gx, Gy)

def MakeEpsilonScreenFromList(EpsilonList, N):
    epsilon = np.zeros((N,N),dtype=np.complex)
    #If N is odd: there are (N^2-1)/2 real elements followed by their corresponding (N^2-1)/2 imaginary elements
    #If N is even: there are (N^2+2)/2 of each, although 3 of these must be purely real, also giving a total of N^2-1 degrees of freedom
    #This is because of conjugation symmetry in Fourier space to ensure a real Fourier transform

    #The first (N-1)/2 are the top row
    N_re = (N*N-1)//2 # FIXME: check logic if N is even
    i = 0
    for x in range(1,(N+1)//2): # FIXME: check logic if N is even
        epsilon[0][x] = EpsilonList[i] + 1j * EpsilonList[i+N_re]
        epsilon[0][N-x] = np.conjugate(epsilon[0][x])
        i=i+1

    #The next N(N-1)/2 are filling the next N rows
    for y in range(1,(N+1)//2): # FIXME: check logic if N is even
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

    return epsilon

def MakeEpsilonScreen(Nx, Ny, rngseed = 0):
    """Create a standardized Fourier representation of a scattering screen

       Args:
           Nx (int): Number of pixels in the x direction
           Ny (int): Number of pixels in the y direction
           rngseed (int): Seed for the random number generator

       Returns:
           epsilon: A 2D numpy ndarray.
    """

    if rngseed != 0:
        np.random.seed( rngseed )

    epsilon = np.random.normal(loc=0.0, scale=1.0/math.sqrt(2), size=(Ny,Nx)) + 1j * np.random.normal(loc=0.0, scale=1.0/math.sqrt(2), size=(Ny,Nx))

    # The zero frequency doesn't affect scattering
    epsilon[0][0] = 0.0

    #Now let's ensure that it has the necessary conjugation symmetry
    if Nx%2 == 0:
        epsilon[0][Nx//2] = np.real(epsilon[0][Nx//2])
    if Ny%2 == 0:
        epsilon[Ny//2][0] = np.real(epsilon[Ny//2][0])
    if Nx%2 == 0 and Ny%2 == 0:
        epsilon[Ny//2][Nx//2] = np.real(epsilon[Ny//2][Nx//2])

    for x in range(Nx):
        if x > (Nx-1)//2: 
            epsilon[0][x] = np.conjugate(epsilon[0][Nx-x])
        for y in range((Ny-1)//2, Ny): 
            x2 = Nx - x
            y2 = Ny - y
            if x2 == Nx:
                x2 = 0
            if y2 == Ny:
                y2 = 0
            epsilon[y][x] = np.conjugate(epsilon[y2][x2])

    return epsilon

##################################################################################################
# Plotting Functions
##################################################################################################

def plot_scatt(im_unscatt, im_ea, im_scatt, im_phase, Prior, nit, chi2, ipynb=False):
    # Get vectors and ratio from current image
    x = np.array([[i for i in range(Prior.xdim)] for j in range(Prior.ydim)])
    y = np.array([[j for i in range(Prior.xdim)] for j in range(Prior.ydim)])

    # Create figure and title
    plt.ion()
    plt.clf()
    if chi2 > 0.0:
        plt.suptitle("step: %i  $\chi^2$: %f " % (nit, chi2), fontsize=20)

    # Unscattered Image
    plt.subplot(141)
    plt.imshow(im_unscatt.reshape(Prior.ydim, Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian', vmin=0)
    xticks = ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
    yticks = ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title('Unscattered')

    # Ensemble Average
    plt.subplot(142)
    plt.imshow(im_ea.reshape(Prior.ydim, Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian', vmin=0)
    xticks = ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
    yticks = ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title('Ensemble Average')

    # Scattered
    plt.subplot(143)
    plt.imshow(im_scatt.reshape(Prior.ydim, Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian', vmin=0)
    xticks = ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
    yticks = ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title('Average Image')

    # Phase
    plt.subplot(144)
    plt.imshow(im_phase.reshape(Prior.ydim, Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian')
    xticks = ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
    yticks = ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
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
