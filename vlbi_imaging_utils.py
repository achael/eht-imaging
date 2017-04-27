# vlbi_imaging_utils.py
# Andrew Chael, 10/15/2015
# Utilities for generating and manipulating VLBI images, datasets, and arrays

# TODO 
# how do we scale sefds/sigmas in jones noise vs normal noise? 
# discuss the calibration flags -- do they make sense / are they being applied properly? 

import string
import numpy as np
import numpy.lib.recfunctions as rec
import matplotlib.pyplot as plt
import scipy.signal
import scipy.optimize as opt
import scipy.ndimage as nd
import itertools as it
import astropy.io.fits as fits
import datetime
import writeData
import oifits_new as oifits
import astropy.time as at
import time as ttime
import pulses

#from mpl_toolkits.basemap import Basemap # for plotting baselines on globe

##################################################################################################
# Constants
##################################################################################################
EP = 1.0e-15
C = 299792458.0
DEGREE = np.pi/180.
HOUR = (180./12.)*DEGREE
RADPERAS = DEGREE/3600
RADPERUAS = RADPERAS*1.e-6

# Default Parameters
SOURCE_DEFAULT = "SgrA"
RA_DEFAULT = 17.761122472222223
DEC_DEFAULT = -28.992189444444445
RF_DEFAULT = 230e9
MJD_DEFAULT = 51544
PULSE_DEFAULT = pulses.trianglePulse2D

# Telescope elevation cuts (degrees) 
ELEV_LOW = 10.0
ELEV_HIGH = 85.0

# Default Optical Depth and std. dev % on gain
TAUDEF = 0.1
GAINPDEF = 0.1
DTERMPDEF = 0.1 # rms amplitude of D-terms if not specified in array file
DTERMPDEF_RESID = 0.01 # rms *residual* amplitude of D-terms (random, unknown contribution)

# Sgr A* Kernel Values (Bower et al., in uas/cm^2)
FWHM_MAJ = 1.309 * 1000 # in uas
FWHM_MIN = 0.64 * 1000
POS_ANG = 78 # in degree, E of N

# Observation recarray datatypes
DTARR = [('site', 'a32'), ('x','f8'), ('y','f8'), ('z','f8'), 
         ('sefdr','f8'),('sefdl','f8'),('dr','c16'),('dl','c16'),
         ('fr_par','f8'),('fr_elev','f8'),('fr_off','f8')]

DTPOL = [('time','f8'),('tint','f8'),
         ('t1','a32'),('t2','a32'),
         ('tau1','f8'),('tau2','f8'),
         ('u','f8'),('v','f8'),
         ('vis','c16'),('qvis','c16'),('uvis','c16'),('vvis','c16'),
         ('sigma','f8'),('qsigma','f8'),('usigma','f8'),('vsigma','f8')]

DTBIS = [('time','f8'),('t1','a32'),('t2','a32'),('t3','a32'),
         ('u1','f8'),('v1','f8'),('u2','f8'),('v2','f8'),('u3','f8'),('v3','f8'),
         ('bispec','c16'),('sigmab','f8')]
                                             
DTCPHASE = [('time','f8'),('t1','a32'),('t2','a32'),('t3','a32'),
            ('u1','f8'),('v1','f8'),('u2','f8'),('v2','f8'),('u3','f8'),('v3','f8'),
            ('cphase','f8'),('sigmacp','f8')]
            
DTCAMP = [('time','f8'),('t1','a32'),('t2','a32'),('t3','a32'),('t4','a32'),
          ('u1','f8'),('v1','f8'),('u2','f8'),('v2','f8'),
          ('u3','f8'),('v3','f8'),('u4','f8'),('v4','f8'),
          ('camp','f8'),('sigmaca','f8')]

# Observation fields for plotting and retrieving data        
FIELDS = ['time','tint','u','v','uvdist',
          't1','t2','tau1','tau2',
          'el1','el2','hr_ang1','hr_ang2','par_ang1','par_ang2',
          'vis','amp','phase','snr',
          'qvis','qamp','qphase','qsnr',
          'uvis','uamp','uphase','usnr',
          'vvis','vamp','vphase','vsnr',
          'sigma','qsigma','usigma','vsigma',
          'sigma_phase','qsigma_phase','usigma_phase','vsigma_phase',
          'psigma_phase','msigma_phase',
          'pvis','pamp','pphase','psnr',
          'm','mamp','mphase','msnr']
                  
##################################################################################################
# Classes
##################################################################################################

class Image(object):
    """A radio frequency image array (in Jy/pixel).
    
       Attributes:
    	pulse: The function convolved with pixel value dirac comb for continuous image rep. (function from pulses.py)
        psize: The pixel dimension in radians (float)
        xdim: The number of pixels along the x dimension (int)
        ydim: The number of pixels along the y dimension (int)
        ra: The source Right Ascension (frac hours)
        dec: The source Declination (frac degrees)
        rf: The radio frequency (Hz)
        imvec: The xdim*ydim vector of jy/pixel values (array)
        source: The astrophysical source name (string)
    	mjd: The mjd of the image 
    """
    
    def __init__(self, image, psize, ra, dec, rf=RF_DEFAULT, pulse=PULSE_DEFAULT, source=SOURCE_DEFAULT, mjd=MJD_DEFAULT):
        if len(image.shape) != 2: 
            raise Exception("image must be a 2D numpy array") 
        
        self.pulse = pulse       
        self.psize = float(psize)
        self.xdim = image.shape[1]
        self.ydim = image.shape[0]
        self.imvec = image.flatten() 
                
        self.ra = float(ra) 
        self.dec = float(dec)
        self.rf = float(rf)
        self.source = str(source)
        self.mjd = int(mjd) 
        
        self.qvec = []
        self.uvec = []
        self.vvec = []
        
    def add_qu(self, qimage, uimage):
        """Add Q and U images
        """
        
        if len(qimage.shape) != len(uimage.shape):
            raise Exception("image must be a 2D numpy array")
        if qimage.shape != uimage.shape != (self.ydim, self.xdim):
            raise Exception("Q & U image shapes incompatible with I image!") 
        self.qvec = qimage.flatten()
        self.uvec = uimage.flatten()
    
    def add_v(self, vimage):
        """Add V image
        """
        if vimage.shape != (self.ydim, self.xdim):
            raise Exception("V image shape incompatible with I image!") 
        self.vvec = vimage.flatten()
    
    def add_pol(self, qimage, uimage, vimage):
        """Add Q, U, V images
        """
        self.add_qu(qimage, uimage)
        self.add_v(vimage)
        
    def copy(self):
        """Copy the image object
        """
        newim = Image(self.imvec.reshape(self.ydim,self.xdim), self.psize, self.ra, self.dec, rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)
        if len(self.qvec):
            newim.add_qu(self.qvec.reshape(self.ydim,self.xdim), self.uvec.reshape(self.ydim,self.xdim))
        return newim

    def imarr(self, stokes="I"):
        """Returns the image 2D array of type stokes"""

        imarr = np.array([])
        if stokes=="I": imarr=im.imvec.reshape(im.ydim, im.xdim)
        elif stokes=="Q" and len(im.qvec): imarr=im.qvec.reshape(im.ydim, im.xdim)
        elif stokes=="U" and len(im.uvec): imarr=im.uvec.reshape(im.ydim, im.xdim)
        elif stokes=="V" and len(im.vvec): imarr=im.vvec.reshape(im.ydim, im.xdim)
        return imarr

    def sourcevec(self):
        """Returns the source position vector in geocentric coordinates (at 0h GMST)
        """
        return np.array([np.cos(self.dec*DEGREE), 0, np.sin(self.dec*DEGREE)])
                
    def flip_chi(self):
        """Change between different conventions for measuring position angle (East of North vs up from x axis)
        """
        self.qvec = - self.qvec
        return

    def observe_same_nonoise(self, obs, sgrscat=False, ft="direct", pad_frac=0.5):
        """Observe the image on the same baselines as an existing observation object
           if sgrscat==True, the visibilites will be blurred by the Sgr A* scattering kernel
           Does NOT add noise
        """

        # Check for agreement in coordinates and frequency 
        if (self.ra!= obs.ra) or (self.dec != obs.dec):
            raise Exception("Image coordinates are not the same as observtion coordinates!")
        if (self.rf != obs.rf):
            raise Exception("Image frequency is not the same as observation frequency!")
        
        if ft=='direct' or ft=='fast':        
            print "Producing clean visibilities from image with " + ft + " FT . . . "
        else:
            raise Exception("ft=%s, options for ft are 'direct' and 'fast'"%ft)

        # Get data (must make a copy!)              
        obsdata = obs.copy().data
                          
        # Extract uv data
        uv = obsdata[['u','v']].view(('f8',2))

        umin = np.min(np.sqrt(uv[:,0]**2 + uv[:,1]**2))
        umax = np.max(np.sqrt(uv[:,0]**2 + uv[:,1]**2))

        if not self.psize < 1/(2*umax): 
            print "    Warning!: longest baseline > 1/2 x maximum image spatial wavelength!"
        if not self.psize*np.sqrt(self.xdim*self.ydim) > 1/(0.5*umin): 
            print "    Warning!: shortest baseline < 2 x minimum image spatial wavelength!"
        
        vis = np.zeros(len(uv))
        qvis = np.zeros(len(uv))
        uvis = np.zeros(len(uv))
        vvis = np.zeros(len(uv))

        #visibilities from FFT
        if ft=="fast":

            # Pad image
            npad = int(np.ceil(pad_frac*1./(self.psize*umin)))
            npad = power_of_two(npad)

            padvalx1 = padvalx2 = int(np.floor((npad - self.xdim)/2.))
            if self.xdim % 2:
                padvalx2 += 1
            padvaly1 = padvaly2 = int(np.floor((npad - self.ydim)/2.))
            if self.ydim % 2:
                padvaly2 += 1

            imarr = self.imvec.reshape(self.ydim, self.xdim)
            imarr = np.pad(imarr, ((padvalx1,padvalx2),(padvaly1,padvaly2)), 'constant', constant_values=0.0)
            npad = imarr.shape[0]
            if imarr.shape[0]!=imarr.shape[1]: 
                raise Exception("FFT padding did not return a square image!")

            # Scaled uv points
            du = 1./(npad*self.psize)
            uv2 = np.hstack((uv[:,1].reshape(-1,1), uv[:,0].reshape(-1,1)))
            uv2 = (uv2 / du + 0.5*npad).T

            # FFT for visibilities
            vis_im = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(imarr)))

            # Sample the visibilities
            visre = nd.map_coordinates(np.real(vis_im), uv2)
            visim = nd.map_coordinates(np.imag(vis_im), uv2)
            vis = visre + 1j*visim

            #extra phase to match centroid convention -- right?
            phase = np.exp(-1j*np.pi*self.psize*(uv[:,0]+uv[:,1]))
            vis = vis * phase

            if len(self.qvec):
                qarr = self.qvec.reshape(self.ydim, self.xdim)
                qarr = np.pad(qarr, ((padvalx1,padvalx2),(padvaly1,padvaly2)), 'constant', constant_values=0.0)
                uarr = self.uvec.reshape(self.ydim, self.xdim)
                uarr = np.pad(uarr, ((padvalx1,padvalx2),(padvaly1,padvaly2)), 'constant', constant_values=0.0)
                
                qvis_im = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(qarr)))
                uvis_im = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(uarr)))

                qvisre = nd.map_coordinates(np.real(qvis_im), uv2)
                qvisim = nd.map_coordinates(np.imag(qvis_im), uv2)
                qvis = phase*(qvisre + 1j*qvisim)

                uvisre = nd.map_coordinates(np.real(uvis_im), uv2)
                uvisim = nd.map_coordinates(np.imag(uvis_im), uv2)
                uvis = phase*(uvisre + 1j*uvisim)

            if len(self.vvec):
                varr = self.vvec.reshape(self.ydim, self.xdim)
                varr = np.pad(varr, ((padvalx1,padvalx2),(padvaly1,padvaly2)), 'constant', constant_values=0.0)
              
                vvis_im = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(varr)))

                vvisre = nd.map_coordinates(np.real(vvis_im), uv2)
                vvisim = nd.map_coordinates(np.imag(vvis_im), uv2)
                vvis = phase*(vvisre + 1j*qvisim)

        #visibilities from DFT
        else:
            mat = ftmatrix(self.psize, self.xdim, self.ydim, uv, pulse=self.pulse)
            vis = np.dot(mat, self.imvec)
        
            if len(self.qvec):
                qvis = np.dot(mat, self.qvec)
                uvis = np.dot(mat, self.uvec)
            if len(self.vvec):
                vvis = np.dot(mat, self.vvec)
                    
        # Scatter the visibilities with the SgrA* kernel
        if sgrscat:
            print 'Scattering Visibilities with Sgr A* kernel!'
            for i in range(len(vis)):
                ker = sgra_kernel_uv(self.rf, uv[i,0], uv[i,1])
                vis[i]  *= ker
                qvis[i] *= ker
                uvis[i] *= ker
                vvis[i] *= ker
                
        # Put the visibilities back in the obsdata array
        obsdata['vis'] = vis
        obsdata['qvis'] = qvis
        obsdata['uvis'] = uvis
        obsdata['vvis'] = vvis
        
        # Return observation object (all calibration flags should be True)
        obs_no_noise = Obsdata(self.ra, self.dec, self.rf, obs.bw, obsdata, obs.tarr, source=self.source, mjd=obs.mjd)
        
        return obs_no_noise 
          
    def observe_same(self, obs, ft='direct', pad_frac=0.5, sgrscat=False, add_th_noise=True, 
                                opacitycal=True, ampcal=True, phasecal=True, frcal=True,dcal=True,
                                tau=TAUDEF, gainp=GAINPDEF, gain_offset=GAINPDEF, dtermp=DTERMPDEF,
                                jones=False, inv_jones=False):
                                                                  
        """Observe the image on the same baselines as an existing observation object
           if sgrscat==True, the visibilites will be blurred by the Sgr A* scattering kernel
           Does NOT add noise
           
           gain_offset can be optionally set as a dictionary that specifies the percentage offset 
           for each telescope site. 
        """

        obs_out = self.observe_same_nonoise(obs, sgrscat=sgrscat, ft=ft, pad_frac=pad_frac)    
        
        # Jones Matrix Corruption
        if jones:
            obs_out = add_jones_and_noise(obs_out, add_th_noise=add_th_noise, opacitycal=opacitycal, 
                                                   ampcal=ampcal, phasecal=phasecal, dcal=dcal, frcal=frcal,
                                                   gainp=gainp, gain_offset=gain_offset, dtermp=dtermp)
            
            #!AC TODO constant gain_offset is NOT calibrated away in this step. Is this inconsistent?
            if inv_jones:
                obs_out = apply_jones_inverse(obs_out, opacitycal=opacitycal, ampcal=ampcal, phasecal=phasecal, 
                                                       dcal=dcal, frcal=frcal)
        
        # No Jones Matrices, Add noise the old way        
        #!AC There is an asymmetry here - in the old way, we don't offer the ability to *not* unscale estimated noise.                                              
        elif add_th_noise:                
            obs_out = add_noise(obs_out, add_th_noise=add_th_noise, opacitycal=opacitycal, 
                                         ampcal=ampcal, phasecal=phasecal, gainp=gainp,
                                         gain_offset=gain_offset)
        
        return obs_out
        
    def observe(self, array, tint, tadv, tstart, tstop, bw, 
                      mjd=None, timetype='UTC', elevmin=ELEV_LOW, elevmax=ELEV_HIGH,
                      ft='direct', pad_frac=0.5, sgrscat=False, add_th_noise=True, 
                      opacitycal=True, ampcal=True, phasecal=True, frcal=True, dcal=True,
                      tau=TAUDEF, gainp=GAINPDEF, gain_offset=GAINPDEF, dtermp=DTERMPDEF,
                      jones=False, inv_jones=False):
                
        """Observe the image with an array object to produce an obsdata object.
	       tstart and tstop should be hrs in UTC.
           tint and tadv should be seconds.
           tau is the estimated optical depth. This can be a single number or a dictionary giving one tau per site
           if sgrscat==True, the visibilites will be blurred by the Sgr A* scattering kernel at the appropriate frequency
           
           gain_offset can be optionally set as a dictionary that specifies the percentage offset 
           for each telescope site. This can only be done currently if jones=False. 
           If gain_offset is a single value than it is the standard deviation 
           of a randomly selected gain offset. 
	    """
        
        # Generate empty observation
        print "Generating empty observation file . . . "
        if mjd == None:
            mjd = self.mjd

        obs = array.obsdata(self.ra, self.dec, self.rf, bw, tint, tadv, tstart, tstop, mjd=mjd, 
                            tau=tau, timetype=timetype, elevmin=elevmin, elevmax=elevmax)

        # Observe on the same baselines as the empty observation and add noise
        obs = self.observe_same(obs, ft=ft, pad_frac=pad_frac, sgrscat=sgrscat, add_th_noise=add_th_noise,
                                     opacitycal=opacitycal,ampcal=ampcal,phasecal=phasecal,dcal=dcal,frcal=frcal,
                                     gainp=gainp,gain_offset=gain_offset,dtermp=dtermp,
                                     jones=jones, inv_jones=inv_jones,)   
        
        return obs
        
    def display(self, cfun='afmhot', nvec=20, pcut=0.01, plotp=False, interp='gaussian', scale='lin',gamma=0.5):
        """Display the image
        """

        # TODO Display circular polarization 
        
        if (interp in ['gauss', 'gaussian', 'Gaussian', 'Gauss']):
            interp = 'gaussian'
        else:
            interp = 'nearest'
            
        plt.figure()
        plt.clf()
        
        imarr = (self.imvec).reshape(self.ydim, self.xdim)
        unit = 'Jy/pixel'
        if scale=='log':
            imarr = np.log(imarr)
            unit = 'log(Jy/pixel)'
        
        if scale=='gamma':
            imarr = imarr**(gamma)
            unit = '(Jy/pixel)^gamma'    
                   
        if len(self.qvec) and plotp:
            thin = self.xdim/nvec 
            mask = (self.imvec).reshape(self.ydim, self.xdim) > pcut * np.max(self.imvec)
            mask2 = mask[::thin, ::thin]
            x = (np.array([[i for i in range(self.xdim)] for j in range(self.ydim)])[::thin, ::thin])[mask2]
            y = (np.array([[j for i in range(self.xdim)] for j in range(self.ydim)])[::thin, ::thin])[mask2]
            a = (-np.sin(np.angle(self.qvec+1j*self.uvec)/2).reshape(self.ydim, self.xdim)[::thin, ::thin])[mask2]
            b = (np.cos(np.angle(self.qvec+1j*self.uvec)/2).reshape(self.ydim, self.xdim)[::thin, ::thin])[mask2]

            m = (np.abs(self.qvec + 1j*self.uvec)/self.imvec).reshape(self.ydim, self.xdim)
            m[-mask] = 0
            
            plt.suptitle('%s   MJD %i  %.2f GHz' % (self.source, self.mjd, self.rf/1e9), fontsize=20)
            
            # Stokes I plot
            plt.subplot(121)
            im = plt.imshow(imarr, cmap=plt.get_cmap(cfun), interpolation=interp)
            plt.colorbar(im, fraction=0.046, pad=0.04, label=unit)
            plt.quiver(x, y, a, b,
                       headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                       width=.01*self.xdim, units='x', pivot='mid', color='k', angles='uv', scale=1.0/thin)
            plt.quiver(x, y, a, b,
                       headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                       width=.005*self.xdim, units='x', pivot='mid', color='w', angles='uv', scale=1.1/thin)

            xticks = ticks(self.xdim, self.psize/RADPERAS/1e-6)
            yticks = ticks(self.ydim, self.psize/RADPERAS/1e-6)
            plt.xticks(xticks[0], xticks[1])
            plt.yticks(yticks[0], yticks[1])
            plt.xlabel('Relative RA ($\mu$as)')
            plt.ylabel('Relative Dec ($\mu$as)')
            plt.title('Stokes I')
        
            # m plot
            plt.subplot(122)
            im = plt.imshow(m, cmap=plt.get_cmap('winter'), interpolation=interp, vmin=0, vmax=1)
            plt.colorbar(im, fraction=0.046, pad=0.04, label='|m|')
            plt.quiver(x, y, a, b,
                   headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                   width=.01*self.xdim, units='x', pivot='mid', color='k', angles='uv', scale=1.0/thin)
            plt.quiver(x, y, a, b,
                   headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                   width=.005*self.xdim, units='x', pivot='mid', color='w', angles='uv', scale=1.1/thin)
            plt.xticks(xticks[0], xticks[1])
            plt.yticks(yticks[0], yticks[1])
            plt.xlabel('Relative RA ($\mu$as)')
            plt.ylabel('Relative Dec ($\mu$as)')
            plt.title('m (above %0.2f max flux)' % pcut)
        
        else:
            plt.subplot(111)    
            plt.title('%s   MJD %i  %.2f GHz' % (self.source, self.mjd, self.rf/1e9), fontsize=20)
            im = plt.imshow(imarr, cmap=plt.get_cmap(cfun), interpolation=interp)
            plt.colorbar(im, fraction=0.046, pad=0.04, label=unit)
            xticks = ticks(self.xdim, self.psize/RADPERAS/1e-6)
            yticks = ticks(self.ydim, self.psize/RADPERAS/1e-6)
            plt.xticks(xticks[0], xticks[1])
            plt.yticks(yticks[0], yticks[1])
            plt.xlabel('Relative RA ($\mu$as)')
            plt.ylabel('Relative Dec ($\mu$as)')   
        
        plt.show(block=False)
            
    def save_txt(self, fname):
        """Save image data to text file
        """
        
        # Coordinate values
        pdimas = self.psize/RADPERAS
        xs = np.array([[j for j in range(self.xdim)] for i in range(self.ydim)]).reshape(self.xdim*self.ydim,1)
        xs = pdimas * (xs[::-1] - self.xdim/2.0)
        ys = np.array([[i for j in range(self.xdim)] for i in range(self.ydim)]).reshape(self.xdim*self.ydim,1)
        ys = pdimas * (ys[::-1] - self.xdim/2.0)
        
        # If V values but no Q/U values, make Q/U zero  
        if len(self.vvec) and not len(self.qvec): 
            self.qvec = 0*self.vvec
            self.uvec = 0*self.vvec
            
        # Format Data            
        if len(self.qvec) and len(self.vvec):
            outdata = np.hstack((xs, ys, (self.imvec).reshape(self.xdim*self.ydim, 1),
                                         (self.qvec).reshape(self.xdim*self.ydim, 1),
                                         (self.uvec).reshape(self.xdim*self.ydim, 1), 
                                         (self.vvec).reshape(self.xdim*self.ydim, 1)))
            hf = "x (as)     y (as)       I (Jy/pixel)  Q (Jy/pixel)  U (Jy/pixel)  V (Jy/pixel)"

            fmts = "%10.10f %10.10f %10.10f %10.10f %10.10f %10.10f"
            
        elif len(self.qvec):
            outdata = np.hstack((xs, ys, (self.imvec).reshape(self.xdim*self.ydim, 1),
                                         (self.qvec).reshape(self.xdim*self.ydim, 1),
                                         (self.uvec).reshape(self.xdim*self.ydim, 1)))
            hf = "x (as)     y (as)       I (Jy/pixel)  Q (Jy/pixel)  U (Jy/pixel)"

            fmts = "%10.10f %10.10f %10.10f %10.10f %10.10f"
            
        else:
            outdata = np.hstack((xs, ys, (self.imvec).reshape(self.xdim*self.ydim, 1)))
            hf = "x (as)     y (as)       I (Jy/pixel)"
            fmts = "%10.10f %10.10f %10.10f"
     
        # Header
        head = ("SRC: %s \n" % self.source +
                    "RA: " + rastring(self.ra) + "\n" + "DEC: " + decstring(self.dec) + "\n" +
                    "MJD: %i \n" % self.mjd + 
                    "RF: %.4f GHz \n" % (self.rf/1e9) + 
                    "FOVX: %i pix %f as \n" % (self.xdim, pdimas * self.xdim) +
                    "FOVY: %i pix %f as \n" % (self.ydim, pdimas * self.ydim) +
                    "------------------------------------\n" + hf)
         
        # Save
        np.savetxt(fname, outdata, header=head, fmt=fmts)

    def save_fits(self, fname):
        """Save image data to FITS file
        """
                
        # Create header and fill in some values
        header = fits.Header()
        header['OBJECT'] = self.source
        header['CTYPE1'] = 'RA---SIN'
        header['CTYPE2'] = 'DEC--SIN'
        header['CDELT1'] = -self.psize/DEGREE
        header['CDELT2'] = self.psize/DEGREE
        header['OBSRA'] = self.ra * 180/12.
        header['OBSDEC'] = self.dec
        header['FREQ'] = self.rf
        header['MJD'] = float(self.mjd)
        header['TELESCOP'] = 'VLBI'
        header['BUNIT'] = 'JY/PIXEL'
        header['STOKES'] = 'I'
        
        # Create the fits image
        image = np.reshape(self.imvec,(self.ydim,self.xdim))[::-1,:] #flip y axis!
        hdu = fits.PrimaryHDU(image, header=header)
        hdulist = [hdu]
        if len(self.qvec):
            qimage = np.reshape(self.qvec,(self.xdim,self.ydim))[::-1,:]
            uimage = np.reshape(self.uvec,(self.xdim,self.ydim))[::-1,:]
            header['STOKES'] = 'Q'
            hduq = fits.ImageHDU(qimage, name='Q', header=header)
            header['STOKES'] = 'U'
            hduu = fits.ImageHDU(uimage, name='U', header=header)
            hdulist = [hdu, hduq, hduu]       
        if len(self.vvec): 
            vimage = np.reshape(self.vvec,(self.xdim,self.ydim))[::-1,:]            
            header['STOKES'] = 'V'
            hduv = fits.ImageHDU(vimage, name='V', header=header)
            hdulist.append(hduv)     
        
        hdulist = fits.HDUList(hdulist)
      
        # Save fits 
        hdulist.writeto(fname, overwrite=True)
        
        return
                
##################################################################################################        
class Array(object):
    """A VLBI array of telescopes with locations and SEFDs
    
        Attributes:
        tarr: The array of telescope data (name, x, y, z, sefdr,sefdl,dr,dl, fr_par_angle, fr_elev_angle, fr_offset)
        where x,y,z are geocentric coordinates.
    """   
    
    def __init__(self, tarr):
        self.tarr = tarr
        
        # Dictionary of array indices for site names
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}
            
    def listbls(self):
        """List all baselines
        """
 
        bls = []
        for i1 in sorted(self.tarr['site']):
            for i2 in sorted(self.tarr['site']):
                if not ([i1,i2] in bls) and not ([i2,i1] in bls) and i1 != i2:
                    bls.append([i1,i2])
                    
        return np.array(bls)
            
    def obsdata(self, ra, dec, rf, bw, tint, tadv, tstart, tstop, mjd=MJD_DEFAULT, 
                      tau=TAUDEF, elevmin=ELEV_LOW, elevmax=ELEV_HIGH, timetype='UTC'):
        """Generate u,v points and baseline errors for the array.
           Return an Observation object with no visibilities.
           tstart and tstop are hrs in UTC
           tint and tadv are seconds.
           rf and bw are Hz
           ra is fractional hours
           dec is fractional degrees
           tau can be a single number or a dictionary giving one per site
        """
        
        #if mjdtogmt(mjd)-tstart > 1e-9:
        #    raise Exception("Initial time is greater than given mjd!")            

        # Set up coordinate system
        sourcevec = np.array([np.cos(dec*DEGREE), 0, np.sin(dec*DEGREE)])
        projU = np.cross(np.array([0,0,1]), sourcevec)
        projU = projU / np.linalg.norm(projU)
        projV = -np.cross(projU, sourcevec)
        
        # Set up time start and steps
        tstep = tadv/3600.0
        if tstop < tstart:
            tstop = tstop + 24.0;       

        # Wavelength
        l = C/rf 
        
        # Observing times
        # TODO: Scale - utc or tt? 
        times = np.arange(tstart, tstop, tstep)
        if timetype == 'UTC':
            times_sidereal = utc_to_gmst(times, mjd)
        elif timetype == 'GMST':
            times_sidereal = times
        else:
            print "Time Type Not Recognized! Assuming UTC!"
            times_sidereal = utc_to_gmst(times, mjd)

        # Generate uv points at all times
        outlist = []        
        for k in xrange(len(times)):
            time = times[k]
            time_sidereal = times_sidereal[k]
            theta = np.mod((time_sidereal-ra)*HOUR, 2*np.pi)
            blpairs = []
            for i1 in xrange(len(self.tarr)):
                for i2 in xrange(len(self.tarr)):
                    coord1 = np.array((self.tarr[i1]['x'], self.tarr[i1]['y'], self.tarr[i1]['z']))
                    coord2 = np.array((self.tarr[i2]['x'], self.tarr[i2]['y'], self.tarr[i2]['z']))
                    if (i1!=i2 and
                        i1 < i2 and # This is the right condition for uvfits save
                        #self.tarr[i1]['z'] <= self.tarr[i2]['z'] and # Choose the north one first
                        not ((i2, i1) in blpairs) and # This cuts out the conjugate baselines
                        elevcut(earthrot(coord1, theta), sourcevec, elevmin, elevmax)[0] and 
                        elevcut(earthrot(coord2, theta), sourcevec, elevmin, elevmax)[0]
                       ):
                        
                        # Optical Depth
                        if type(tau) == dict:
                            try: 
                                tau1 = tau[i1]
                                tau2 = tau[i2]
                            except KeyError:
                                tau1 = tau2 = TAUDEF
                        else:
                            tau1 = tau2 = tau
                        
                        
                        # Noise on the correlations
                        sig_rr = blnoise(self.tarr[i1]['sefdr'], self.tarr[i2]['sefdr'], tint, bw)
                        sig_ll = blnoise(self.tarr[i1]['sefdl'], self.tarr[i2]['sefdl'], tint, bw)
                        sig_rl = blnoise(self.tarr[i1]['sefdr'], self.tarr[i2]['sefdl'], tint, bw)
                        sig_lr = blnoise(self.tarr[i1]['sefdl'], self.tarr[i2]['sefdr'], tint, bw)
                        
                        # Append data to list   
                        blpairs.append((i1,i2))
                        outlist.append(np.array((
                                  time,
                                  tint, # Integration 
                                  self.tarr[i1]['site'], # Station 1
                                  self.tarr[i2]['site'], # Station 2
                                  tau1, # Station 1 optical depth
                                  tau2, # Station 1 optical depth
                                  np.dot(earthrot(coord1 - coord2, theta)/l, projU), # u (lambda)
                                  np.dot(earthrot(coord1 - coord2, theta)/l, projV), # v (lambda)
                                  0.0, # I Visibility (Jy)
                                  0.0, # Q Visibility
                                  0.0, # U Visibility
                                  0.0, # V Visibilities
                                  0.5*np.sqrt(sig_rr**2 + sig_ll**2), # I Sigma (Jy)
                                  0.5*np.sqrt(sig_rl**2 + sig_lr**2), # Q Sigma 
                                  0.5*np.sqrt(sig_rl**2 + sig_lr**2), # U Sigma
                                  0.5*np.sqrt(sig_rr**2 + sig_ll**2)  # V Sigma
                                ), dtype=DTPOL
                                ))
                                
        obsarr = np.array(outlist)
         
        if not len(obsarr):
            raise Exception("No mutual visibilities in the specified time range!")
            
        # Return
        obs = Obsdata(ra, dec, rf, bw, np.array(outlist), self.tarr, source=str(ra) + ":" + str(dec), mjd=mjd, timetype=timetype)      
        return obs
     
    def save_array(self, fname):
        """Save the array data in a text file
        """

        out = ("#Site      X(m)             Y(m)             Z(m)           "+
                    "SEFDR      SEFDL     FR_PAR   FR_EL   FR_OFF  "+
                    "DR_RE    DR_IM    DL_RE    DL_IM   \n")
        for scope in range(len(self.tarr)):
            dat = (self.tarr[scope]['site'], 
                   self.tarr[scope]['x'], self.tarr[scope]['y'], self.tarr[scope]['z'], 
                   self.tarr[scope]['sefdr'], self.tarr[scope]['sefdl'],
                   self.tarr[scope]['fr_par'], self.tarr[scope]['fr_elev'], self.tarr[scope]['fr_off'],
                   self.tarr[scope]['dr'].real, self.tarr[scope]['dr'].imag, self.tarr[scope]['dl'].real, self.tarr[scope]['dl'].imag
                  )
            out += "%-8s %15.5f  %15.5f  %15.5f  %8.2f   %8.2f  %5.2f   %5.2f   %5.2f  %8.4f %8.4f %8.4f %8.4f \n" % dat
        f = open(fname,'w')
        f.write(out)
        f.close()
        return 
         
#    def plotbls(self):
#        """Plot all baselines on a globe"""
#        
#        lat = []
#        lon = []
#        for t1 in range(len(tarr)):
#            (x,y,z) = (self.tarr[t1]['x'], self.tarr[t1]['y'], self.tarr[t1]['z'])
#            lon.append(np.arctan2(y, x)/DEGREE)
#            lat.append(90 - np.arccos(z/np.sqrt(x**2 + y**2 + z**2))/DEGREE)

#        map = Basemap(projection='moll', lon_0=-90)
#        map.drawmapboundary(fill_color='blue')
#        map.fillcontinents(color='green', lake_color='blue')
#        map.drawcoastlines()
#        for i in range(len(lon)):
#            for j in range(len(lon)):
#                x,y = map([lon[i],lon[j]], [lat[i],lat[j]])
#                map.plot(x, y, marker='D', color='r')
#        
#        plt.show()
        
##################################################################################################        
class Obsdata(object):
    """A VLBI observation of visibility amplitudes and phases. 
    
       Attributes:
        source: the source name
        ra: the source right ascension (frac. hours)
        dec: the source declination (frac. degrees)
        mjd: the observation start date 
        tstart: the observation start time (UTC, hr.)
        tstop: the observation end time (UTC, hr.)
        rf: the observing frequency (Hz)
        bw: the observing bandwidth (Hz)
        ampcal: amplitudes calibrated T/F
        phasecal: phases calibrated T/F
        data: recarray with the data (time, t1, t2, tint, u, v, vis, qvis, uvis, vvis, sigma, qsigma, usigma, vsigma)
    """
    
    def __init__(self, ra, dec, rf, bw, datatable, tarr, source=SOURCE_DEFAULT, mjd=MJD_DEFAULT, ampcal=True, phasecal=True, opacitycal=True, dcal=True, frcal=True, timetype='UTC'):
        
        if len(datatable) == 0:
            raise Exception("No data in input table!")
        if (datatable.dtype != DTPOL):
            raise Exception("Data table should be a recarray with datatable.dtype = %s" % DTPOL)
        
        # Set the various parameters
        self.source = str(source)
        self.ra = float(ra)
        self.dec = float(dec)
        self.rf = float(rf)
        self.bw = float(bw)
        self.ampcal = bool(ampcal)
        self.phasecal = bool(phasecal)
        self.opacitycal = bool(opacitycal)
        self.dcal = bool(dcal)
        self.frcal = bool(frcal)
        self.timetype = timetype
        self.tarr = tarr
        
        # Dictionary of array indices for site names
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}
        
        # Time partition the datatable
        datalist = []
        for key, group in it.groupby(datatable, lambda x: x['time']):
            datalist.append(np.array([obs for obs in group]))
        
        # Remove conjugate baselines
        obsdata = []
        for tlist in datalist:
            blpairs = []
            for dat in tlist:
                if not (set((dat['t1'], dat['t2']))) in blpairs:

                     # Reverse the baseline in the right order for uvfits:
                     if(self.tkey[dat['t1']] < self.tkey[dat['t2']]):                        
                        (dat['t1'], dat['t2']) = (dat['t2'], dat['t1'])
                        dat['u'] = -dat['u']
                        dat['v'] = -dat['v']
                        dat['vis'] = np.conj(dat['vis'])
                        dat['uvis'] = np.conj(dat['uvis'])
                        dat['qvis'] = np.conj(dat['qvis'])
                        dat['vvis'] = np.conj(dat['vvis'])

                     # Append the data point
                     blpairs.append(set((dat['t1'],dat['t2'])))    
                     obsdata.append(dat) 

        obsdata = np.array(obsdata, dtype=DTPOL)
        
        # Sort the data by time
        obsdata = obsdata[np.argsort(obsdata, order=['time','t1'])]
        
        # Save the data             
        self.data = obsdata
            
        # Get tstart, mjd and tstop
        times = self.unpack(['time'])['time']
        self.tstart = times[0]
        self.mjd = int(mjd)
        #self.mjd = fracmjd(mjd, self.tstart)
        self.tstop = times[-1]
        if self.tstop < self.tstart: 
            self.tstop += 24.0
  
    def copy(self):
        """Copy the observation object
        """
        newobs = Obsdata(self.ra, self.dec, self.rf, self.bw, self.data, self.tarr, source=self.source, mjd=self.mjd, 
                         ampcal=self.ampcal, phasecal=self.phasecal, opacitycal=self.opacitycal, dcal=self.dcal, frcal=self.frcal)
        return newobs
        
    def data_conj(self):
        """Return a data array of same format as self.data but including all conjugate baselines
        """
        
        data = np.empty(2*len(self.data), dtype=DTPOL)        
        
        # Add the conjugate baseline data
        for f in DTPOL:
            f = f[0]
            if f in ["t1", "t2", "tau1", "tau2"]:
                if f[-1]=='1': f2 = f[:-1]+'2'
                else: f2 = f[:-1]+'1'
                data[f] = np.hstack((self.data[f], self.data[f2]))
            elif f in ["u","v"]:
                data[f] = np.hstack((self.data[f], -self.data[f]))
            elif f in ["vis","qvis","uvis","vvis"]:
                data[f] = np.hstack((self.data[f], np.conj(self.data[f])))
            else:
                data[f] = np.hstack((self.data[f], self.data[f]))
        
        # Sort the data by time
        #!AC TODO should we apply some sorting within equal times? 
        data = data[np.argsort(data['time'])]
        return data

    def tlist(self, conj=False):
        """Return partitioned data in a list of equal time observations
        """
        
        if conj: 
            data = self.data_conj()
        else: 
            data = self.data
        
        # Use itertools groupby function to partition the data
        datalist = []
        for key, group in it.groupby(data, lambda x: x['time']):
            datalist.append(np.array([obs for obs in group]))
        
        return np.array(datalist)
                
    def unpack_bl(self, site1, site2, in_fields, ang_unit='deg'):
        """Unpack the data over time on the selected baseline
        """

        # If we only specify one field
        fields=['time']
        if type(in_fields) == str: fields.append(in_fields)
        else: 
            for i in range(len(in_fields)): fields.append(in_fields[i])
            
        # Get field data on selected baseline   
        allout = []    
        
        # Get the data from data table on the selected baseline
        tlist = self.tlist(conj=True)
        for scan in tlist:
            for obs in scan:
                if (obs['t1'], obs['t2']) == (site1, site2):
                    obs = np.array([obs])
                    out = self.unpack_dat(obs, fields, ang_unit=ang_unit)             
                    allout.append(out)
        return np.array(allout)            
    
    def unpack(self, fields, conj=False, ang_unit='deg', mode='all'):
        
        if not mode in ('time', 'all'):
            raise Exception("possible options for mode are 'time' and 'all'")
                    
        # If we only specify one field
        if type(fields) == str: fields = [fields]
        
        if mode=='all':    
            if conj:
                data = self.data_conj()     
            else:
                data = self.data
            allout=self.unpack_dat(data, fields, ang_unit=ang_unit)
        elif mode=='time':
            allout=[]
            tlist = self.tlist(conj=True)
            for scan in tlist:
                out=self.unpack_dat(scan, fields, ang_unit=ang_unit)
                allout.append(out)
        
        return np.array(allout)
    
    def unpack_dat(self, data, fields, conj=False, ang_unit='deg'):
        """Return a recarray of all the data for the given fields from the data table
           If conj=True, will return conjugate baselines
        """
       
        if ang_unit=='deg': angle=DEGREE
        else: angle = 1.0
        
        # Get field data    
        allout = []    
        for field in fields:
            if field in ["u","v","tint","time","tau1","tau2"]: 
                out = data[field]
                ty = 'f8'
            elif field in ["uvdist"]: 
                out = np.abs(data['u'] + 1j * data['v'])
                ty = 'f8'
            elif field in ["t1","el1","par_ang1","hr_ang1"]: 
                sites = data["t1"]
                keys = [self.tkey[site] for site in sites]
                tdata = self.tarr[keys]
                out = sites
                ty = 'a32'
            elif field in ["t2","el2","par_ang2","hr_ang2"]: 
                sites = data["t2"]
                keys = [self.tkey[site] for site in sites]
                tdata = self.tarr[keys]
                out = sites
                ty = 'a32'
            elif field in ["vis","amp","phase","snr","sigma","sigma_phase"]: 
                out = data['vis']
                sig = data['sigma']
                ty = 'c16'
            elif field in ["qvis","qamp","qphase","qsnr","qsigma","qsigma_phase"]: 
                out = data['qvis']
                sig = data['qsigma']
                ty = 'c16'
            elif field in ["uvis","uamp","uphase","usnr","usigma","usigma_phase"]: 
                out = data['uvis']
                sig = data['usigma']
                ty = 'c16'
            elif field in ["vvis","vamp","vphase","vsnr","vsigma","vsigma_phase"]: 
                out = data['vvis']
                sig = data['vsigma']
                ty = 'c16'               
            elif field in ["pvis","pamp","pphase","psnr","psigma","psigma_phase"]: 
                out = data['qvis'] + 1j * data['uvis']
                sig = np.sqrt(data['qsigma']**2 + data['usigma']**2)
                ty = 'c16'
            elif field in ["m","mamp","mphase","msnr","msigma","msigma_phase"]: 
                out = (data['qvis'] + 1j * data['uvis']) / data['vis']
                sig = merr(data['sigma'], data['qsigma'], data['usigma'], data['vis'], out)
                ty = 'c16'
            
            else: raise Exception("%s is not valid field \n" % field + 
                                  "valid field values are " + string.join(FIELDS)) 

            # Elevation and Parallactic Angles
            if field in ["el1","el2","hr_ang1","hr_ang2","par_ang1","par_ang2"]:
                if self.timetype=='GMST':
                    times_sid = data['time']
                else:
                    times_sid = utc_to_gmst(data['time'], self.mjd)

                thetas = np.mod((times_sid - self.ra)*HOUR, 2*np.pi)
                coords = tdata[['x','y','z']].view(('f8', 3))
                el_angle = elev(earthrot(coords, thetas), self.sourcevec())
                latlon = xyz_2_latlong(coords)
                hr_angles = hr_angle(times_sid*HOUR, latlon[:,1], self.ra*HOUR)

                if field in ["el1","el2"]:
                    out=el_angle/angle
                    ty = 'f8'
                if field in ["hr_ang1","hr_ang2"]:
                    out = hr_angles/angle
                    ty = 'f8'
                if field in ["par_ang1","par_ang2"]:
                    par_ang = par_angle(hr_angles, latlon[:,0], self.dec*DEGREE)
                    out = par_ang/angle
                    ty = 'f8'
                                
            # Get arg/amps/snr
            if field in ["amp", "qamp", "uamp","vamp","pamp","mamp"]: 
                out = np.abs(out)
                ty = 'f8'
            elif field in ["phase", "qphase", "uphase", "vphase","pphase", "mphase"]: 
                out = np.angle(out)/angle
                ty = 'f8'
            elif field in ["sigma","qsigma","usigma","vsigma","psigma","msigma"]:
                out = np.abs(sig)
                ty = 'f8'
            elif field in ["sigma_phase","qsigma_phase","usigma_phase","vsigma_phase","psigma_phase","msigma_phase"]:
                out = np.abs(sig)/np.abs(out)/angle
                ty = 'f8'                                                
            elif field in ["snr", "qsnr", "usnr", "vsnr", "psnr", "msnr"]:
                out = np.abs(out)/np.abs(sig)
                ty = 'f8'
                                            
            # Reshape and stack with other fields
            out = np.array(out, dtype=[(field, ty)])

            if len(allout) > 0: #N.B.: This throws an error sometimes 
                allout = rec.merge_arrays((allout, out), asrecarray=True, flatten=True)
            else:
                allout = out
            
        return allout
    
    def sourcevec(self):
        """Returns the source position vector in geocentric coordinates (at 0h GMST)
        """
        return np.array([np.cos(self.dec*DEGREE), 0, np.sin(self.dec*DEGREE)])
        
    def res(self):
        """Return the nominal resolution of the observation in radian
        """
        return 1.0/np.max(self.unpack('uvdist')['uvdist'])
        
    def bispectra(self, vtype='vis', mode='time', count='min'):
        """Return all independent equal time bispectrum values
           Independent triangles are chosen to contain the minimum sefd station in the scan
           Set count='max' to return all bispectrum values
           Get Q, U, P bispectra by changing vtype
        """

        if not mode in ('time', 'all'):
            raise Exception("possible options for mode are 'time' and 'all'")
        if not count in ('min', 'max'):
            raise Exception("possible options for count are 'min' and 'max'")
        if not vtype in ('vis','qvis','uvis','vvis','pvis'):
            raise Exception("possible options for vtype are 'vis','qvis','uvis','vvis','pvis'")
            
        # Generate the time-sorted data with conjugate baselines
        tlist = self.tlist(conj=True)    
        outlist = []
        bis = []
        for tdata in tlist:
            time = tdata[0]['time']
            sites = list(set(np.hstack((tdata['t1'],tdata['t2']))))
                                        
            # Create a dictionary of baselines at the current time incl. conjugates;
            l_dict = {}
            for dat in tdata:
                l_dict[(dat['t1'], dat['t2'])] = dat
            
            # Determine the triangles in the time step
            if count == 'min':
                # If we want a minimal set, choose triangles with the minimum sefd reference
                # Unless there is no sefd data, in which case choose the northernmost
                # !AC TODO This should probably be an sefdr + sefdl average
                if len(set(self.tarr['sefdr'])) > 1:
                    ref = sites[np.argmin([self.tarr[self.tkey[site]]['sefdr'] for site in sites])]
                else:
                    ref = sites[np.argmax([self.tarr[self.tkey[site]]['z'] for site in sites])]
                sites.remove(ref)
                
                # Find all triangles that contain the ref                    
                tris = list(it.combinations(sites,2))
                tris = [(ref, t[0], t[1]) for t in tris]
            elif count == 'max':
                # Find all triangles
                tris = list(it.combinations(sites,3))
            
            # Generate bispectra for each triangle
            for tri in tris:
                # The ordering is north-south
                a1 = np.argmax([self.tarr[self.tkey[site]]['z'] for site in tri])
                a3 = np.argmin([self.tarr[self.tkey[site]]['z'] for site in tri])
                a2 = 3 - a1 - a3
                tri = (tri[a1], tri[a2], tri[a3])
                    
                # Select triangle entries in the data dictionary
                try:
                    l1 = l_dict[(tri[0], tri[1])]
                    l2 = l_dict[(tri[1],tri[2])]
                    l3 = l_dict[(tri[2], tri[0])]
                except KeyError:
                    continue
                    
                # Choose the appropriate polarization and compute the bs and err
                if vtype in ["vis", "qvis", "uvis","vvis"]:
                    if vtype=='vis':  sigmatype='sigma'
                    if vtype=='qvis': sigmatype='qsigma'
                    if vtype=='uvis': sigmatype='usigma'
                    if vtype=='vvis': sigmatype='vsigma'
                    
                    p1 = l1[vtype]
                    p2 = l2[vtype]
                    p3 = l3[vtype]
                    
                    var1 = l1[sigmatype]**2 
                    var2 = l2[sigmatype]**2                                                        
                    var3 = l3[sigmatype]**2                                                        
                                                                                                                                                      
                elif vtype == "pvis":
                    p1 = l1['qvis'] + 1j*l2['uvis']
                    p2 = l2['qvis'] + 1j*l2['uvis']
                    p3 = l3['qvis'] + 1j*l3['uvis']
                    bi = p1 * p2 * p3
                    
                    var1 = l1['qsigma']**2 + l1['usigma']**2
                    var2 = l2['qsigma']**2 + l2['usigma']**2
                    var3 = l3['qsigma']**2 + l3['usigma']**2                                                                                                     
                
                bi = p1*p2*p3
                bisig = np.abs(bi) * np.sqrt(var1/np.abs(p1)**2 +  
                                             var2/np.abs(p2)**2 + 
                                             var3/np.abs(p3)**2)
                #Katie's 2nd + 3rd order corrections - see CHIRP supplement
                bisig = np.sqrt(bisig**2 + var1*var2*np.abs(p3)**2 +  
                                           var1*var3*np.abs(p2)**2 +  
                                           var2*var3*np.abs(p1)**2 +  
                                           var1*var2*var3)                                                               
                
                # Append to the equal-time list
                bis.append(np.array((time, tri[0], tri[1], tri[2], 
                                     l1['u'], l1['v'], l2['u'], l2['v'], l3['u'], l3['v'],
                                     bi, bisig), dtype=DTBIS))                 
            
            # Append equal time bispectra to outlist    
            if mode=='time' and len(bis) > 0:
                outlist.append(np.array(bis))
                bis = []    
     
        if mode=='all':
            outlist = np.array(bis)
        
        return np.array(outlist)
   
        
    def c_phases(self, vtype='vis', mode='time', count='min', ang_unit='deg'):
        """Return all independent equal time closure phase values
           Independent triangles are chosen to contain the minimum sefd station in the scan
           Set count='max' to return all closure phases
        """

        if not mode in ('time', 'all'):
            raise Exception("possible options for mode are 'time' and 'all'")  
        if not count in ('max', 'min'):
            raise Exception("possible options for count are 'max' and 'min'")  
        if not vtype in ('vis','qvis','uvis','vvis','pvis'):
            raise Exception("possible options for vtype are 'vis','qvis','uvis','vvis','pvis'")
                
        if ang_unit=='deg': angle=DEGREE
        else: angle = 1.0
                    
        # Get the bispectra data
        bispecs = self.bispectra(vtype=vtype, mode='time', count=count)
        
        # Reformat into a closure phase list/array
        outlist = []
        cps = []
        for bis in bispecs:
            for bi in bis:
                if len(bi) == 0: continue
                bi.dtype.names = ('time','t1','t2','t3','u1','v1','u2','v2','u3','v3','cphase','sigmacp')
                bi['sigmacp'] = bi['sigmacp']/np.abs(bi['cphase'])/angle
                bi['cphase'] = (np.angle(bi['cphase'])/angle).real
                cps.append(bi.astype(np.dtype(DTCPHASE)))
            if mode == 'time' and len(cps) > 0:
                outlist.append(np.array(cps))
                cps = []
                
        if mode == 'all':
            outlist = np.array(cps)

        return np.array(outlist)    

    def unique_c_phases(self):
        """Return all unique closure phase triangles
        """
        biarr = self.bispectra(mode="all", count="min")
        catsites = np.vstack((np.vstack((biarr['t1'],biarr['t2'])), biarr['t3'] ))
        uniqueclosure = np.vstack({tuple(row) for row in catsites.T})

        return uniqueclosure
         
    def c_amplitudes(self, vtype='vis', mode='time', count='min'):
        """Return equal time closure amplitudes
           Set count='max' to return all closure amplitudes up to inverses
        """ 
        
        if not mode in ('time','all'):
            raise Exception("possible options for mode are 'time' and 'all'")
        if not count in ('max', 'min'):
            raise Exception("possible options for count are 'max' and 'min'")  
        if not vtype in ('vis','qvis','uvis','vvis','pvis'):
            raise Exception("possible options for vtype are 'vis','qvis','uvis','vvis','pvis'")
                    
        tlist = self.tlist(conj=True) 
        outlist = []
        cas = []
        for tdata in tlist:
            time = tdata[0]['time']
            sites = np.array(list(set(np.hstack((tdata['t1'],tdata['t2'])))))
            if len(sites) < 4:
                continue
                                            
            # Create a dictionary of baselines at the current time incl. conjugates;
            l_dict = {}
            for dat in tdata:
                l_dict[(dat['t1'], dat['t2'])] = dat
            
            if count == 'min':
                # If we want a minimal set, choose the minimum sefd reference
                # !AC TODO this should probably be an sefdr + sefdl average
                sites = sites[np.argsort([self.tarr[self.tkey[site]]['sefdr'] for site in sites])]
                ref = sites[0]
                
                # Loop over other sites >=3 and form minimal closure amplitude set
                for i in xrange(3, len(sites)):
                    blue1 = l_dict[ref, sites[i]] #!!
                    for j in xrange(1, i):
                        if j == i-1: k = 1
                        else: k = j+1
                        
                        red1 = l_dict[sites[i], sites[j]]
                        red2 = l_dict[ref, sites[k]]
                        blue2 = l_dict[sites[j], sites[k]] 
                        
                        # Compute the closure amplitude and the error
                        if vtype in ["vis", "qvis", "uvis", "vvis"]:
                            if vtype=='vis':  sigmatype='sigma'
                            if vtype=='qvis': sigmatype='qsigma'
                            if vtype=='uvis': sigmatype='usigma'
                            if vtype=='vvis': sigmatype='vsigma'
                            
                            var1 = blue1[sigmatype]**2
                            var2 = blue2[sigmatype]**2
                            var3 = red1[sigmatype]**2
                            var4 = red2[sigmatype]**2
                            
                            p1 = amp_debias(blue1[vtype], np.sqrt(var1))
                            p2 = amp_debias(blue2[vtype], np.sqrt(var2))
                            p3 = amp_debias(red1[vtype], np.sqrt(var3))
                            p4 = amp_debias(red2[vtype], np.sqrt(var4))
                                                                             
                        elif vtype == "pvis":
                            var1 = blue1['qsigma']**2 + blue1['usigma']**2
                            var2 = blue2['qsigma']**2 + blue2['usigma']**2
                            var3 = red1['qsigma']**2 + red1['usigma']**2
                            var4 = red2['qsigma']**2 + red2['usigma']**2

                            p1 = amp_debias(blue1['qvis'] + 1j*blue1['uvis'], np.sqrt(var1))
                            p2 = amp_debias(blue2['qvis'] + 1j*blue2['uvis'], np.sqrt(var2))
                            p3 = amp_debias(red1['qvis'] + 1j*red1['uvis'], np.sqrt(var3))
                            p4 = amp_debias(red2['qvis'] + 1j*red2['uvis'], np.sqrt(var4))
                                                   
                        camp = np.abs((p1*p2)/(p3*p4))
                        camperr = camp * np.sqrt(var1/np.abs(p1)**2 +  
                                                 var2/np.abs(p2)**2 + 
                                                 var3/np.abs(p3)**2 +
                                                 var4/np.abs(p4)**2)
                                        
                        # Add the closure amplitudes to the equal-time list  
                        # Our site convention is (12)(34)/(14)(23)       
                        cas.append(np.array((time, 
                                             ref, sites[i], sites[j], sites[k],
                                             blue1['u'], blue1['v'], blue2['u'], blue2['v'], 
                                             red1['u'], red1['v'], red2['u'], red2['v'],
                                             camp, camperr),
                                             dtype=DTCAMP)) 

            # !AC TODO Find a different way to do min/max sets so we don't have to duplicate code here?
            elif count == 'max':
                # Find all quadrangles
                quadsets = list(it.combinations(sites,4))
                for q in quadsets:
                    # Loop over 3 closure amplitudes
                    # Our site convention is (12)(34)/(14)(23)
                    for quad in (q, [q[0],q[2],q[1],q[3]], [q[0],q[1],q[3],q[2]]): 
                        
                        # Blue is numerator, red is denominator
                        blue1 = l_dict[quad[0], quad[1]]
                        blue2 = l_dict[quad[2], quad[3]]
                        red1 = l_dict[quad[0], quad[3]]
                        red2 = l_dict[quad[1], quad[2]]
                                      
                        # Compute the closure amplitude and the error
                        if vtype in ["vis", "qvis", "uvis", "vvis"]:
                            if vtype=='vis':  sigmatype='sigma'
                            if vtype=='qvis': sigmatype='qsigma'
                            if vtype=='uvis': sigmatype='usigma'
                            if vtype=='vvis': sigmatype='vsigma'
                            
                            var1 = blue1[sigmatype]**2
                            var2 = blue2[sigmatype]**2
                            var3 = red1[sigmatype]**2
                            var4 = red2[sigmatype]**2
                            
                            p1 = amp_debias(blue1[vtype], np.sqrt(var1))
                            p2 = amp_debias(blue2[vtype], np.sqrt(var2))
                            p3 = amp_debias(red1[vtype], np.sqrt(var3))
                            p4 = amp_debias(red2[vtype], np.sqrt(var4))
                                                                             
                        elif vtype == "pvis":
                            var1 = blue1['qsigma']**2 + blue1['usigma']**2
                            var2 = blue2['qsigma']**2 + blue2['usigma']**2
                            var3 = red1['qsigma']**2 + red1['usigma']**2
                            var4 = red2['qsigma']**2 + red2['usigma']**2

                            p1 = amp_debias(blue1['qvis'] + 1j*blue1['uvis'], np.sqrt(var1))
                            p2 = amp_debias(blue2['qvis'] + 1j*blue2['uvis'], np.sqrt(var2))
                            p3 = amp_debias(red1['qvis'] + 1j*red1['uvis'], np.sqrt(var3))
                            p4 = amp_debias(red2['qvis'] + 1j*red2['uvis'], np.sqrt(var4))
                                                   
                        camp = np.abs((p1*p2)/(p3*p4))
                        camperr = camp * np.sqrt(var1/np.abs(p1)**2 +  
                                                 var2/np.abs(p2)**2 + 
                                                 var3/np.abs(p3)**2 +
                                                 var4/np.abs(p4)**2)
                                        
                                        
                        # Add the closure amplitudes to the equal-time list         
                        cas.append(np.array((time, 
                                             quad[0], quad[1], quad[2], quad[3],
                                             blue1['u'], blue1['v'], blue2['u'], blue2['v'], 
                                             red1['u'], red1['v'], red2['u'], red2['v'],
                                             camp, camperr),
                                             dtype=DTCAMP)) 

            # Append all equal time closure amps to outlist    
            if mode=='time':
                outlist.append(np.array(cas))
                cas = []    
            elif mode=='all':
                outlist = np.array(cas)
        
        return np.array(outlist)

    #!AC TODO Add covariance matrices!
    def log_c_amplitudes(self, cov=True, vtype='vis', mode='time', count='min'):
        """Return equal time log closure amplitudes
           Set count='max' to return all closure amplitudes up to inverses
        """
     
        if not mode in ('time','all'):
            raise Exception("possible options for mode are 'time' and 'all'")
        if not count in ('max', 'min'):
            raise Exception("possible options for count are 'max' and 'min'")  
        if not vtype in ('vis','qvis','uvis','vvis','pvis'):
            raise Exception("possible options for vtype are 'vis','qvis','uvis','vvis','pvis'")
                    
        tlist = self.tlist(conj=True) 
        covs = []
        outlist = []
        cas = []
        for tdata in tlist:
            time = tdata[0]['time']
            sites = np.array(list(set(np.hstack((tdata['t1'],tdata['t2'])))))
            if len(sites) < 4:
                continue
                                            
            # Create a dictionary of baselines at the current time incl. conjugates;
            l_dict = {}
            for dat in tdata:
                l_dict[(dat['t1'], dat['t2'])] = dat
            
            if count == 'min':
                # If we want a minimal set, choose the minimum sefd reference
                # !AC this should probably be an sefdr + sefdl average
                sites = sites[np.argsort([self.tarr[self.tkey[site]]['sefdr'] for site in sites])]
                ref = sites[0]
                
                # Loop over other sites >=3 and form minimal closure amplitude set
                for i in xrange(3, len(sites)):
                    blue1 = l_dict[ref, sites[i]] #!!
                    for j in xrange(1, i):
                        if j == i-1: k = 1
                        else: k = j+1
                        
                        red1 = l_dict[sites[i], sites[j]]
                        red2 = l_dict[ref, sites[k]]
                        blue2 = l_dict[sites[j], sites[k]] 
                        
                        # Compute the closure amplitude and the error
                        if vtype in ["vis", "qvis", "uvis"]:
                            if vtype=='vis':  sigmatype='sigma'
                            if vtype=='qvis': sigmatype='qsigma'
                            if vtype=='uvis': sigmatype='usigma'
                            if vtype=='vvis': sigmatype='vsigma'
                            
                            var1 = blue1[sigmatype]**2
                            var2 = blue2[sigmatype]**2
                            var3 = red1[sigmatype]**2
                            var4 = red2[sigmatype]**2
                            
                            p1 = amp_debias(blue1[vtype], np.sqrt(var1)) #!AC TODO debias or not in the closure amplitude?
                            p2 = amp_debias(blue2[vtype], np.sqrt(var2))
                            p3 = amp_debias(red1[vtype], np.sqrt(var3))
                            p4 = amp_debias(red2[vtype], np.sqrt(var4))
                                                                             
                        elif vtype == "pvis":
                            var1 = blue1['qsigma']**2 + blue1['usigma']**2
                            var2 = blue2['qsigma']**2 + blue2['usigma']**2
                            var3 = red1['qsigma']**2 + red1['usigma']**2
                            var4 = red2['qsigma']**2 + red2['usigma']**2
                            
                            p1 = amp_debias(blue1['qvis'] + 1j*blue1['uvis'], np.sqrt(var1))
                            p2 = amp_debias(blue2['qvis'] + 1j*blue2['uvis'], np.sqrt(var2))
                            p3 = amp_debias(red1['qvis'] + 1j*red1['uvis'], np.sqrt(var3))
                            p4 = amp_debias(red2['qvis'] + 1j*red2['uvis'], np.sqrt(var4))
                            
                        
                        logcamp = np.log(np.abs(p1)) + np.log(np.abs(p2)) - np.log(np.abs(p3)) - np.log(np.abs(p4));

                        logcamperr = np.sqrt(var1/np.abs(p1)**2 +  
                                             var2/np.abs(p2)**2 + 
                                             var3/np.abs(p3)**2 +
                                             var4/np.abs(p4)**2)
                                        
                        # Add the closure amplitudes to the equal-time list  
                        # Our site convention is (12)+(34)-(14)-(23)       
                        cas.append(np.array((time, 
                                             ref, sites[i], sites[j], sites[k],
                                             blue1['u'], blue1['v'], blue2['u'], blue2['v'], 
                                             red1['u'], red1['v'], red2['u'], red2['v'],
                                             logcamp, logcamperr),
                                             dtype=DTCAMP)) 
           
                #!AC TODO
                # Make the covariance matrix after making all the scan data points    
                #if cov: 
                #    clbl = np.array([[[self.tkey[cl['t1']],self.tkey[cl['t2']]],
                #                      [self.tkey[cl['t3']],self.tkey[cl['t4']]],
                #                      [self.tkey[cl['t1']],self.tkey[cl['t4']]],
                #                      [self.tkey[cl['t2']],self.tkey[cl['t3']]]
                #                    ] for cl in cas])
                #    covmatrix = make_cov(clbl, l_dict)
                #    covs.append(covmatrix)

            #!AC TODO find a better way to loop over min/max sets so we don't need to duplicate so much code?
            elif count == 'max':
                # Find all quadrangles
                quadsets = list(it.combinations(sites,4))
                for q in quadsets:
                    # Loop over 3 closure amplitudes
                    # Our site convention is (12)(34)/(14)(23)
                    for quad in (q, [q[0],q[2],q[1],q[3]], [q[0],q[1],q[3],q[2]]): 
                        
                        # Blue is numerator, red is denominator
                        blue1 = l_dict[quad[0], quad[1]]
                        blue2 = l_dict[quad[2], quad[3]]
                        red1 = l_dict[quad[0], quad[3]]
                        red2 = l_dict[quad[1], quad[2]]
                                      
                        # Compute the closure amplitude and the error
                        if vtype in ["vis", "qvis", "uvis"]:
                            if vtype=='vis':  sigmatype='sigma'
                            if vtype=='qvis': sigmatype='qsigma'
                            if vtype=='uvis': sigmatype='usigma'
                            if vtype=='vvis': sigmatype='vsigma'
                            
                            var1 = blue1[sigmatype]**2
                            var2 = blue2[sigmatype]**2
                            var3 = red1[sigmatype]**2
                            var4 = red2[sigmatype]**2
                            
                            p1 = amp_debias(blue1[vtype], e1)
                            p2 = amp_debias(blue2[vtype], e2)
                            p3 = amp_debias(red1[vtype], e3)
                            p4 = amp_debias(red2[vtype], e4)
                                                                             
                        elif vtype == "pvis":
                            var1 = blue1['qsigma']**2 + blue1['usigma']**2
                            var2 = blue2['qsigma']**2 + blue2['usigma']**2
                            var3 = red1['qsigma']**2 + red1['usigma']**2
                            var4 = red2['qsigma']**2 + red2['usigma']**2

                            p1 = amp_debias(blue1['qvis'] + 1j*blue1['uvis'], e1)
                            p2 = amp_debias(blue2['qvis'] + 1j*blue2['uvis'], e2)
                            p3 = amp_debias(red1['qvis'] + 1j*red1['uvis'], e3)
                            p4 = amp_debias(red2['qvis'] + 1j*red2['uvis'], e4)
                            
                        
                        logcamp = np.log(np.abs(p1)) + np.log(np.abs(p2)) - np.log(np.abs(p3)) - np.log(np.abs(p4));

                        logcamperr = np.sqrt(var1/np.abs(p1)**2 +  
                                             var2/np.abs(p2)**2 + 
                                             var3/np.abs(p3)**2 +
                                             var4/np.abs(p4)**2)
                                        
                        # Add the closure amplitudes to the equal-time list  
                        # Our site convention is (12)+(34)-(14)-(23)       
                        cas.append(np.array((time, 
                                             ref, sites[i], sites[j], sites[k],
                                             blue1['u'], blue1['v'], blue2['u'], blue2['v'], 
                                             red1['u'], red1['v'], red2['u'], red2['v'],
                                             logcamp, logcamperr),
                                             dtype=DTCAMP)) 
                
                #!AC TODO
                # Make the covariance matrix after making all the scan data points    
                #if cov: 
                #    covs.append(covmatrix)
                #    clbl = np.array([[[self.tkey[cl['t1']],self.tkey[cl['t2']]],
                #                      [self.tkey[cl['t3']],self.tkey[cl['t4']]],
                #                      [self.tkey[cl['t1']],self.tkey[cl['t4']]],
                #                      [self.tkey[cl['t2']],self.tkey[cl['t3']]]
                #                    ] for cl in cas])
                #    covmatrix = make_cov(clbl, l_dict)
 
            # Append all equal time closure amps to outlist    
            if mode=='time':
                outlist.append(np.array(cas))
                cas = []    
            elif mode=='all':
                outlist = np.array(cas)               
        
        return np.array(outlist)

    
    def dirtybeam(self, npix, fov, pulse=PULSE_DEFAULT):
        """Return a square Image object of the observation dirty beam
           fov is in radian
        """

        # !AC TODO add different types of beam weighting
        pdim = fov/npix
        u = self.unpack('u')['u']
        v = self.unpack('v')['v']
        
        xlist = np.arange(0,-npix,-1)*pdim + (pdim*npix)/2.0 - pdim/2.0
        
        im = np.array([[np.mean(np.cos(2*np.pi*(i*u + j*v)))
                  for i in xlist] 
                  for j in xlist])    
        
        im = im[0:npix, 0:npix]
        
        # Normalize to a total beam power of 1
        im = im/np.sum(im)
        
        src = self.source + "_DB"
        return Image(im, pdim, self.ra, self.dec, rf=self.rf, source=src, mjd=self.mjd, pulse=pulse)
    
    def dirtyimage(self, npix, fov, pulse=PULSE_DEFAULT):
        """Return a square Image object of the observation dirty image
           fov is in radian
        """

        # !AC TODO add different types of beam weighting  
        pdim = fov/npix
        u = self.unpack('u')['u']
        v = self.unpack('v')['v']
        vis = self.unpack('vis')['vis']
        qvis = self.unpack('qvis')['qvis']
        uvis = self.unpack('uvis')['uvis']
        vvis = self.unpack('vvis')['vvis']
        
        xlist = np.arange(0,-npix,-1)*pdim + (pdim*npix)/2.0 - pdim/2.0

        # Take the DTFTS
        # Shouldn't need to real about conjugate baselines b/c unpack does not return them
        im  = np.array([[np.mean(np.real(vis)*np.cos(2*np.pi*(i*u + j*v)) - 
                                 np.imag(vis)*np.sin(2*np.pi*(i*u + j*v)))
                  for i in xlist] 
                  for j in xlist])    
        qim = np.array([[np.mean(np.real(qvis)*np.cos(2*np.pi*(i*u + j*v)) - 
                                 np.imag(qvis)*np.sin(2*np.pi*(i*u + j*v)))
                  for i in xlist] 
                  for j in xlist])     
        uim = np.array([[np.mean(np.real(uvis)*np.cos(2*np.pi*(i*u + j*v)) - 
                                 np.imag(uvis)*np.sin(2*np.pi*(i*u + j*v)))
                  for i in xlist] 
                  for j in xlist])    
        vim = np.array([[np.mean(np.real(vvis)*np.cos(2*np.pi*(i*u + j*v)) - 
                                 np.imag(vvis)*np.sin(2*np.pi*(i*u + j*v)))
                  for i in xlist] 
                  for j in xlist])    
                                                             
        dim = np.array([[np.mean(np.cos(2*np.pi*(i*u + j*v)))
                  for i in xlist] 
                  for j in xlist])   
        
        # Normalization   
        im = im/np.sum(dim)
        qim = qim/np.sum(dim)
        uim = uim/np.sum(dim)
        vim = vim/np.sum(dim)
        
        im = im[0:npix, 0:npix]
        qim = qim[0:npix, 0:npix]
        uim = uim[0:npix, 0:npix]   
        vim = vim[0:npix, 0:npix]
        
        out = Image(im, pdim, self.ra, self.dec, rf=self.rf, source=self.source, mjd=self.mjd, pulse=pulse)
        out.add_qu(qim, uim)
        out.add_v(vim)
        
        return out
    
    def cleanbeam(self, npix, fov, pulse=PULSE_DEFAULT):
        """Return a square Image object of the observation fitted (clean) beam
           fov is in radian
        """
        # !AC TODO include other beam weightings
        im = make_square(self, npix, fov, pulse=pulse)
        beamparams = self.fit_beam()
        im = add_gauss(im, 1.0, beamparams)
        return im
        
    def fit_beam(self):
        """Fit a gaussian to the dirty beam and return the parameters (fwhm_maj, fwhm_min, theta).
           All params are in radian and theta is measured E of N.
           Fit the quadratic expansion of the Gaussian (normalized to 1 at the peak) 
           to the expansion of dirty beam with the same normalization
        """    
        # !AC TODO include other beam weightings
        # Define the sum of squares function that compares the quadratic expansion of the dirty image
        # with the quadratic expansion of an elliptical gaussian
        def fit_chisq(beamparams, db_coeff):
            
            (fwhm_maj2, fwhm_min2, theta) = beamparams
            a = 4 * np.log(2) * (np.cos(theta)**2/fwhm_min2 + np.sin(theta)**2/fwhm_maj2)
            b = 4 * np.log(2) * (np.cos(theta)**2/fwhm_maj2 + np.sin(theta)**2/fwhm_min2)
            c = 8 * np.log(2) * np.cos(theta) * np.sin(theta) * (1/fwhm_maj2 - 1/fwhm_min2)
            gauss_coeff = np.array((a,b,c))
            
            chisq = np.sum((np.array(db_coeff) - gauss_coeff)**2)
            
            return chisq
        
        # These are the coefficients (a,b,c) of a quadratic expansion of the dirty beam
        # For a point (x,y) in the image plane, the dirty beam expansion is 1-ax^2-by^2-cxy
        u = self.unpack('u')['u']
        v = self.unpack('v')['v']
        n = float(len(u))
        abc = (2.*np.pi**2/n) * np.array([np.sum(u**2), np.sum(v**2), 2*np.sum(u*v)])                
        abc = 1e-20 * abc # Decrease size of coefficients
        
        # Fit the beam 
        guess = [(50)**2, (50)**2, 0.0]
        params = opt.minimize(fit_chisq, guess, args=(abc,), method='Powell')
        
        # Return parameters, adjusting fwhm_maj and fwhm_min if necessary
        if params.x[0] > params.x[1]:
            fwhm_maj = 1e-10*np.sqrt(params.x[0])
            fwhm_min = 1e-10*np.sqrt(params.x[1])
            theta = np.mod(params.x[2], np.pi)
        else:
            fwhm_maj = 1e-10*np.sqrt(params.x[1])
            fwhm_min = 1e-10*np.sqrt(params.x[0])
            theta = np.mod(params.x[2] + np.pi/2, np.pi)

        return np.array((fwhm_maj, fwhm_min, theta))
    
    def plotall(self, field1, field2, ebar=True, rangex=False, rangey=False, conj=False, show=True, axis=False, color='b', ang_unit='deg'):
        """Make a scatter plot of 2 real observation fields with errors
           If conj==True, display conjugate baselines
        """
        
        # Determine if fields are valid
        if (field1 not in FIELDS) and (field2 not in FIELDS):
            raise Exception("valid fields are " + string.join(FIELDS))
                              
        # Unpack x and y axis data
        data = self.unpack([field1, field2], conj=conj, ang_unit=ang_unit)
        
        # X error bars
        if sigtype(field1):
            sigx = self.unpack(sigtype(field2), conj=conj, ang_unit=ang_unit)[sigtype(field1)]
        else:
            sigx = None
            
        # Y error bars
        if sigtype(field2):
            sigy = self.unpack(sigtype(field2), conj=conj, ang_unit=ang_unit)[sigtype(field2)]
        else:
            sigy = None
        
        # Debias amplitudes if appropriate:
        if field1 in ['amp', 'qamp', 'uamp', 'vamp', 'pamp', 'mamp']:
            print "De-biasing amplitudes for plot x values!"
            data[field1] = amp_debias(data[field1], sigx)
        
        if field2 in ['amp', 'qamp', 'uamp', 'vamp', 'pamp', 'mamp']:
            print "De-biasing amplitudes for plot y values!"
            data[field2] = amp_debias(data[field2], sigy)
           
        # Data ranges
        if not rangex:
            rangex = [np.min(data[field1]) - 0.2 * np.abs(np.min(data[field1])), 
                      np.max(data[field1]) + 0.2 * np.abs(np.max(data[field1]))] 
        if not rangey:
            rangey = [np.min(data[field2]) - 0.2 * np.abs(np.min(data[field2])), 
                      np.max(data[field2]) + 0.2 * np.abs(np.max(data[field2]))] 
        
        # Plot the data
        if axis:
            x = axis
        else:
            fig=plt.figure()
            x = fig.add_subplot(1,1,1)
         
        if ebar and (np.any(sigy) or np.any(sigx)):
            x.errorbar(data[field1], data[field2], xerr=sigx, yerr=sigy, fmt='.', color=color)
        else:
            x.plot(data[field1], data[field2], '.', color=color)
            
        x.set_xlim(rangex)
        x.set_ylim(rangey)
        x.set_xlabel(field1)
        x.set_ylabel(field2)

        if show:
            plt.show(block=False)
        return x
                        
    def plot_bl(self, site1, site2, field, ebar=True, rangex=False, rangey=False, show=True, axis=False, color='b', ang_unit='deg'):
        """Plot a field over time on a baseline
        """
                
        if ang_unit=='deg': angle=DEGREE
        else: angle = 1.0
        
        # Determine if fields are valid
        if field not in FIELDS:
            raise Exception("valid fields are " + string.join(FIELDS))
        
        plotdata = self.unpack_bl(site1, site2, field, ang_unit=ang_unit)
        if not rangex: 
            rangex = [self.tstart,self.tstop]
        if not rangey:
            rangey = [np.min(plotdata[field]) - 0.2 * np.abs(np.min(plotdata[field])), 
                      np.max(plotdata[field]) + 0.2 * np.abs(np.max(plotdata[field]))] 
        # Plot the data                        
        if axis:
            x = axis
        else:
            fig = plt.figure()
            x = fig.add_subplot(1,1,1)       

        if ebar and sigtype(field)!=False:
            errdata = self.unpack_bl(site1, site2, sigtype(field), ang_unit=ang_unit)
            x.errorbar(plotdata['time'][:,0], plotdata[field][:,0], yerr=errdata[sigtype(field)][:,0], fmt='b.', color=color)
        else:
            x.plot(plotdata['time'][:,0], plotdata[field][:,0],'b.', color=color)
            
        x.set_xlim(rangex)
        x.set_ylim(rangey)
        x.set_xlabel('hr')
        x.set_ylabel(field)
        x.set_title('%s - %s'%(site1,site2))
        
        if show:
            plt.show(block=False)    
        return x
                
                
    def plot_cphase(self, site1, site2, site3, vtype='vis', ebar=True, rangex=False, rangey=False, show=True, axis=False, color='b', ang_unit='deg'):
        """Plot closure phase over time on a triangle
        """
                
        if ang_unit=='deg': angle=DEGREE
        else: angle = 1.0
        
        # Get closure phases (maximal set)
        cphases = self.c_phases(mode='time', count='max', vtype=vtype)
        
        # Get requested closure phases over time
        tri = (site1, site2, site3)
        plotdata = []
        for entry in cphases:
            for obs in entry:
                obstri = (obs['t1'],obs['t2'],obs['t3'])
                if set(obstri) == set(tri):
                    # Flip the sign of the closure phase if necessary
                    parity = paritycompare(tri, obstri) 
                    plotdata.append([obs['time'], parity*obs['cphase'], obs['sigmacp']])
                    continue
        
        plotdata = np.array(plotdata)
        
        if len(plotdata) == 0: 
            print "No closure phases on this triangle!"
            return
        
        # Data ranges
        if not rangex: 
            rangex = [self.tstart,self.tstop]
        if not rangey:
            rangey = [np.min(plotdata[:,1]) - 0.2 * np.abs(np.min(plotdata[:,1])), 
                      np.max(plotdata[:,1]) + 0.2 * np.abs(np.max(plotdata[:,1]))] 
        
        # Plot the data                        
        if axis:
            x = axis
        else:
            fig=plt.figure()
            x = fig.add_subplot(1,1,1)       

        if ebar and np.any(plotdata[:,2]):
            x.errorbar(plotdata[:,0], plotdata[:,1], yerr=plotdata[:,2], fmt='b.', color=color)
        else:
            x.plot(plotdata[:,0], plotdata[:,1],'b.', color=color)
            
        x.set_xlim(rangex)
        x.set_ylim(rangey)
        x.set_xlabel('GMT (h)')
        x.set_ylabel('Closure Phase (deg)')
        x.set_title('%s - %s - %s' % (site1,site2,site3))
        if show:
            plt.show(block=False)    
        return x
        
    def plot_camp(self, site1, site2, site3, site4, vtype='vis', ebar=True, rangex=False, rangey=False, show=True, axis=False, color='b'):
        """Plot closure amplitude over time on a quadrange
           (1-2)(3-4)/(1-4)(2-3)
        """
        quad = (site1, site2, site3, site4)
        b1 = set((site1, site2))
        r1 = set((site1, site4))
              
        # Get the closure amplitudes
        camps = self.c_amplitudes(mode='time', count='max', vtype='vis')
        plotdata = []
        for entry in camps:
            for obs in entry:
                obsquad = (obs['t1'],obs['t2'],obs['t3'],obs['t4'])
                if set(quad) == set(obsquad):
                    num = [set((obs['t1'], obs['t2'])), set((obs['t3'], obs['t4']))] 
                    denom = [set((obs['t1'], obs['t4'])), set((obs['t2'], obs['t3']))]
                    
                    if (b1 in num) and (r1 in denom):
                        plotdata.append([obs['time'], obs['camp'], obs['sigmaca']])
                    elif (r1 in num) and (b1 in denom):
                        plotdata.append([obs['time'], 1./obs['camp'], obs['sigmaca']/(obs['camp']**2)])
                    continue
                
                    
        plotdata = np.array(plotdata)
        if len(plotdata) == 0: 
            print "No closure amplitudes on this quadrangle!"
            return

        # Data ranges
        if not rangex: 
            rangex = [self.tstart,self.tstop]
        if not rangey:
            rangey = [np.min(plotdata[:,1]) - 0.2 * np.abs(np.min(plotdata[:,1])), 
                      np.max(plotdata[:,1]) + 0.2 * np.abs(np.max(plotdata[:,1]))] 
        
        # Plot the data                        
        if axis:
            x = axis
        else:
            fig=plt.figure()
            x = fig.add_subplot(1,1,1)       
            
        if ebar and np.any(plotdata[:,2]):
            x.errorbar(plotdata[:,0], plotdata[:,1], yerr=plotdata[:,2], fmt='b.', color=color)
        else:
            x.plot(plotdata[:,0], plotdata[:,1],'b.', color=color)
            
        x.set_xlim(rangex)
        x.set_ylim(rangey)
        x.set_xlabel('GMT (h)')
        x.set_ylabel('Closure Amplitude')
        x.set_title('(%s - %s)(%s - %s)/(%s - %s)(%s - %s)'%(site1,site2,site3,site4,
                                                           site1,site4,site2,site3))
        if show:
            plt.show(block=False)    
            return
        else:
            return x
    
    def fit_gauss(self, flux=1.0, fittype='amp', paramguess=(100*RADPERUAS, 100*RADPERUAS, 0.)):
        """Fit a gaussian to either Stokes I complex visibilities or Stokes I visibility amplitudes
           TODO bispectra/closure phase?
        """
        vis = self.data['vis']
        u = self.data['u']
        v = self.data['v']
        sig = self.data['sigma']
        
        # error function
        if fittype=='amp':
            def errfunc(p):
            	vismodel = gauss_uv(u,v, flux, p, x=0., y=0.)
            	err = np.sum((np.abs(vis)-np.abs(vismodel))**2/sig**2)
            	return err
        else:
            def errfunc(p):
            	vismodel = gauss_uv(u,v, flux, p, x=0., y=0.)
            	err = np.sum(np.abs(vis-vismodel)**2/sig**2)
            	return err
        
        optdict = {'maxiter':5000} # minimizer params
        res = opt.minimize(errfunc, paramguess, method='Powell',options=optdict)
        return res.x
            	        
    def save_txt(self, fname):
        """Save visibility data to a text file"""

        # Get the necessary data and the header
        outdata = self.unpack(['time', 'tint', 't1', 't2','tau1','tau2',
                               'u', 'v', 'amp', 'phase', 'qamp', 'qphase', 'uamp', 'uphase', 'vamp', 'vphase',
                               'sigma', 'qsigma', 'usigma', 'vsigma'])
        head = ("SRC: %s \n" % self.source +
                    "RA: " + rastring(self.ra) + "\n" + "DEC: " + decstring(self.dec) + "\n" +
                    "MJD: %i \n" % self.mjd + 
                    "RF: %.4f GHz \n" % (self.rf/1e9) + 
                    "BW: %.4f GHz \n" % (self.bw/1e9) +
                    "PHASECAL: %i \n" % self.phasecal + 
                    "AMPCAL: %i \n" % self.ampcal +
                    "OPACITYCAL: %i \n" % self.opacitycal +                    
                    "DCAL: %i \n" % self.dcal + 
                    "FRCAL: %i \n" % self.frcal +  
                    "----------------------------------------------------------------------"+
                    "------------------------------------------------------------------\n" +
                    "Site       X(m)             Y(m)             Z(m)           "+
                    "SEFDR      SEFDL     FR_PAR   FR_EL   FR_OFF  "+
                    "DR_RE    DR_IM    DL_RE    DL_IM   \n"
                )
        
        for i in range(len(self.tarr)):
            head += ("%-8s %15.5f  %15.5f  %15.5f  %8.2f   %8.2f  %5.2f   %5.2f   %5.2f  %8.4f %8.4f %8.4f %8.4f \n" % (self.tarr[i]['site'], 
                                                                      self.tarr[i]['x'], self.tarr[i]['y'], self.tarr[i]['z'], 
                                                                      self.tarr[i]['sefdr'], self.tarr[i]['sefdl'],
                                                                      self.tarr[i]['fr_par'], self.tarr[i]['fr_elev'], self.tarr[i]['fr_off'],
                                                                      (self.tarr[i]['dr']).real, (self.tarr[i]['dr']).imag, 
                                                                      (self.tarr[i]['dl']).real, (self.tarr[i]['dl']).imag
                                                                     ))

        head += (
                "----------------------------------------------------------------------"+
                "------------------------------------------------------------------\n" +
                "time (hr) tint    T1     T2    Tau1   Tau2   U (lambda)       V (lambda)         "+
                "Iamp (Jy)    Iphase(d)  Qamp (Jy)    Qphase(d)   Uamp (Jy)    Uphase(d)   Vamp (Jy)    Vphase(d)   "+
                "Isigma (Jy)   Qsigma (Jy)   Usigma (Jy)   Vsigma (Jy)"
                )
          
        # Format and save the data
        fmts = ("%011.8f %4.2f %6s %6s  %4.2f   %4.2f  %16.4f %16.4f    "+
               "%10.8f %10.4f   %10.8f %10.4f    %10.8f %10.4f    %10.8f %10.4f    "+
               "%10.8f    %10.8f    %10.8f    %10.8f")
        np.savetxt(fname, outdata, header=head, fmt=fmts)
        return
    
    #!AC TODO how can we save dterm and field rotation arrays to uvfits?
    def save_uvfits(self, fname):
        """Save visibility data to uvfits
           Needs template.UVP file
        """

        # Open template UVFITS
        hdulist = fits.open('./template.UVP')

        ########################################################################
        # Antenna table
        
        # Load the array data
        tarr = self.tarr
        tnames = tarr['site']
        tnums = np.arange(1, len(tarr)+1)
        xyz = np.array([[tarr[i]['x'],tarr[i]['y'],tarr[i]['z']] for i in np.arange(len(tarr))])
        sefd = tarr['sefdr']
        
        nsta = len(tnames)
        col1 = fits.Column(name='ANNAME', format='8A', array=tnames)
        col2 = fits.Column(name='STABXYZ', format='3D', unit='METERS', array=xyz)
        col3 = fits.Column(name='NOSTA', format='1J', array=tnums)
        colfin = fits.Column(name='SEFD', format='1D', array=sefd)
        
        #!AC TODO these antenna fields+header are questionable - look into them

        col4 = fits.Column(name='MNTSTA', format='1J', array=np.zeros(nsta))
        col5 = fits.Column(name='STAXOF', format='1E', unit='METERS', array=np.zeros(nsta))
        col6 = fits.Column(name='POLTYA', format='1A', array=np.array(['R' for i in range(nsta)], dtype='|S1'))
        col7 = fits.Column(name='POLAA', format='1E', unit='DEGREES', array=np.zeros(nsta))
        col8 = fits.Column(name='POLCALA', format='3E', array=np.zeros((nsta,3)))
        col9 = fits.Column(name='POLTYB', format='1A', array=np.array(['L' for i in range(nsta)], dtype='|S1'))
        col10 = fits.Column(name='POLAB', format='1E', unit='DEGREES', array=(90.*np.ones(nsta)))
        col11 = fits.Column(name='POLCALB', format='3E', array=np.zeros((nsta,3)))
        col25= fits.Column(name='ORBPARM', format='1E', array=np.zeros(0))
        
        #Antenna Header params - do I need to change more of these?? 
        #head = fits.Header()
        head = hdulist['AIPS AN'].header
        head['EXTVER'] = 1
        head['GSTIA0'] = 119.85 # !AC TODO ?? for mjd 48277
        head['FREQ']= self.rf
        head['ARRNAM'] = 'ALMA' #!AC TODO Can we change this field? 
        head['XYZHAND'] = 'RIGHT'
        head['ARRAYX'] = 0.e0
        head['ARRAYY'] = 0.e0
        head['ARRAYZ'] = 0.e0
        head['DEGPDY'] = 360.985
        head['POLARX'] = 0.e0
        head['POLARY'] = 0.e0
        head['UT1UTC'] = 0.e0
        head['DATUTC'] = 0.e0
        head['TIMESYS'] = 'UTC'
        head['NUMORB'] = 0
        head['NO_IF'] = 1
        head['NOPCAL'] = 2
        head['POLTYPE'] = 'APPROX'
        head['FREQID'] = 1
        tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs([col1,col2,col25,col3,col4,col5,col6,col7,col8,col9,col10,col11,colfin]), name='AIPS AN', header=head)
        hdulist['AIPS AN'] = tbhdu

        ########################################################################
        # Data table
        # Data header (based on the BU format)
        #header = fits.Header()
        header = hdulist[0].header
        header['OBSRA'] = self.ra * 180./12.
        header['OBSDEC'] = self.dec
        header['OBJECT'] = self.source
        header['MJD'] = float(self.mjd)
        header['BUNIT'] = 'JY'
        header['VELREF'] = 3 #??
        header['ALTRPIX'] = 1.e0
        header['TELESCOP'] = 'ALMA' # !AC TODO Can we change this field?  
        header['INSTRUME'] = 'ALMA'
        header['CTYPE2'] = 'COMPLEX'
        header['CRVAL2'] = 1.e0
        header['CDELT2'] = 1.e0
        header['CRPIX2'] = 1.e0
        header['CROTA2'] = 0.e0
        header['CTYPE3'] = 'STOKES'
        header['CRVAL3'] = -1.e0
        header['CRDELT3'] = -1.e0
        header['CRPIX3'] = 1.e0
        header['CROTA3'] = 0.e0
        header['CTYPE4'] = 'FREQ'
        header['CRVAL4'] = self.rf
        header['CDELT4'] = self.bw
        header['CRPIX4'] = 1.e0
        header['CROTA4'] = 0.e0
        header['CTYPE6'] = 'RA'
        header['CRVAL6'] = header['OBSRA']
        header['CDELT6'] = 1.e0
        header['CRPIX6'] = 1.e0
        header['CROTA6'] = 0.e0
        header['CTYPE7'] = 'DEC'
        header['CRVAL7'] = header['OBSDEC']
        header['CDELT7'] = 1.e0
        header['CRPIX7'] = 1.e0
        header['CROTA7'] = 0.e0
        header['PTYPE1'] = 'UU---SIN'
        header['PSCAL1'] = 1/self.rf
        header['PZERO1'] = 0.e0
        header['PTYPE2'] = 'VV---SIN'
        header['PSCAL2'] = 1/self.rf
        header['PZERO2'] = 0.e0
        header['PTYPE3'] = 'WW---SIN'
        header['PSCAL3'] = 1/self.rf
        header['PZERO3'] = 0.e0
        header['PTYPE4'] = 'BASELINE'
        header['PSCAL4'] = 1.e0
        header['PZERO4'] = 0.e0
        header['PTYPE5'] = 'DATE'
        header['PSCAL5'] = 1.e0
        header['PZERO5'] = 0.e0
        header['PTYPE6'] = 'DATE'
        header['PSCAL6'] = 1.e0
        header['PZERO6'] = 0.0
        header['PTYPE7'] = 'INTTIM'
        header['PSCAL7'] = 1.e0
        header['PZERO7'] = 0.e0
        header['PTYPE8'] = 'TAU1'
        header['PSCAL8'] = 1.e0
        header['PZERO8'] = 0.e0
        header['PTYPE9'] = 'TAU2'
        header['PSCAL9'] = 1.e0
        header['PZERO9'] = 0.e0
                
        # Get data
        obsdata = self.unpack(['time','tint','u','v','vis','qvis','uvis','vvis','sigma','qsigma','usigma','vsigma','t1','t2','tau1','tau2'])
        ndat = len(obsdata['time'])
        
        # times and tints
        #jds = (self.mjd + 2400000.5) * np.ones(len(obsdata))
        #fractimes = (obsdata['time'] / 24.0) 
        jds = (2400000.5 + self.mjd) * np.ones(len(obsdata))
        fractimes = (obsdata['time'] / 24.0) 
        #jds = jds + fractimes
        #fractimes = np.zeros(len(obsdata))        
        tints = obsdata['tint']

        # Baselines            
        t1 = [self.tkey[scope] + 1 for scope in obsdata['t1']]
        t2 = [self.tkey[scope] + 1 for scope in obsdata['t2']]
        bl = 256*np.array(t1) + np.array(t2)
           
        # opacities
        tau1 = obsdata['tau1']
        tau2 = obsdata['tau2']
        
        # uv are in lightseconds
        u = obsdata['u']
        v = obsdata['v']
        
        # rr, ll, lr, rl, weights
        rr = obsdata['vis'] + obsdata['vvis']
        ll = obsdata['vis'] - obsdata['vvis']
        rl = obsdata['qvis'] + 1j*obsdata['uvis']
        lr = obsdata['qvis'] - 1j*obsdata['uvis']
        
        weightrr = 1 / (obsdata['sigma']**2 + obsdata['vsigma']**2)
        weightll = 1 / (obsdata['sigma']**2 + obsdata['vsigma']**2)
        weightrl = 1 / (obsdata['qsigma']**2 + obsdata['usigma']**2)
        weightlr = 1 / (obsdata['qsigma']**2 + obsdata['usigma']**2)
                                
        # Data array
        outdat = np.zeros((ndat, 1, 1, 1, 1, 4, 3))
        outdat[:,0,0,0,0,0,0] = np.real(rr)
        outdat[:,0,0,0,0,0,1] = np.imag(rr)
        outdat[:,0,0,0,0,0,2] = weightrr
        outdat[:,0,0,0,0,1,0] = np.real(ll)
        outdat[:,0,0,0,0,1,1] = np.imag(ll)
        outdat[:,0,0,0,0,1,2] = weightll
        outdat[:,0,0,0,0,2,0] = np.real(rl)
        outdat[:,0,0,0,0,2,1] = np.imag(rl)
        outdat[:,0,0,0,0,2,2] = weightrl
        outdat[:,0,0,0,0,3,0] = np.real(lr)
        outdat[:,0,0,0,0,3,1] = np.imag(lr)
        outdat[:,0,0,0,0,3,2] = weightlr
        
        # Save data
        
        pars = ['UU---SIN', 'VV---SIN', 'WW---SIN', 'BASELINE', 'DATE', 'DATE',
                'INTTIM', 'TAU1', 'TAU2']
        x = fits.GroupData(outdat, parnames=pars,
            pardata=[u, v, np.zeros(ndat), bl, jds, fractimes, tints,tau1,tau2],
            bitpix=-32)

                
        hdulist[0].data = x
        hdulist[0].header = header
 
        ##################################################################################
        # AIPS FQ TABLE -- Thanks to Kazu
        # Convert types & columns

        nif=1
        col1 = np.array(1, dtype=np.int32).reshape([nif]) #frqsel
        col2 = np.array(0.0, dtype=np.float64).reshape([nif]) #iffreq
        col3 = np.array([self.bw], dtype=np.float32).reshape([nif]) #chwidth
        col4 = np.array([self.bw], dtype=np.float32).reshape([nif]) #bw
        col5 = np.array([1], dtype=np.int32).reshape([nif]) #sideband

        col1 = fits.Column(name="FRQSEL", format="1J", array=col1)
        col2 = fits.Column(name="IF FREQ", format="%dD"%(nif), array=col2)
        col3 = fits.Column(name="CH WIDTH",format="%dE"%(nif),array=col3)
        col4 = fits.Column(name="TOTAL BANDWIDTH",format="%dE"%(nif),array=col4)
        col5 = fits.Column(name="SIDEBAND",format="%dJ"%(nif),array=col5)
        cols = fits.ColDefs([col1, col2,col3,col4,col5])

        # create table
        tbhdu = fits.BinTableHDU.from_columns(cols)
        
        # add header information
        tbhdu.header.append(("NO_IF", nif, "Number IFs"))
        tbhdu.header.append(("EXTNAME","AIPS FQ"))
        hdulist.append(tbhdu)
         
        # Write final HDUList to file
        hdulist.writeto(fname, overwrite=True)
                
        return
    
    def save_oifits(self, fname, flux=1.0):
        """ Save visibility data to oifits
            Polarization data is NOT saved
            Antenna diameter currently incorrect and the exact times are not correct in the datetime object
            Please contact Katie Bouman (klbouman@mit.edu) for any questions on this function 
        """
        #todo: Add polarization to oifits??
        print 'Warning: save_oifits does NOT save polarimetric visibility data!'
        
        # Normalizing by the total flux passed in - note this is changing the data inside the obs structure
        self.data['vis'] /= flux
        self.data['sigma'] /= flux
        
        data = self.unpack(['u','v','amp','phase', 'sigma', 'time', 't1', 't2', 'tint'])
        biarr = self.bispectra(mode="all", count="min")

        # extract the telescope names and parameters
        antennaNames = self.tarr['site'] #np.array(self.tkey.keys())
        sefd = self.tarr['sefdr']
        antennaX = self.tarr['x']
        antennaY = self.tarr['y']
        antennaZ = self.tarr['z']
        #antennaDiam = -np.ones(antennaX.shape) #todo: this is incorrect and there is just a dummy variable here
        antennaDiam = sefd # replace antennaDiam with SEFD for radio observtions
        
        # create dictionary
        union = {};
        union = writeData.arrayUnion(antennaNames, union)

        # extract the integration time
        intTime = data['tint'][0]
        if not all(data['tint'][0] == item for item in np.reshape(data['tint'], (-1)) ):
            raise TypeError("The time integrations for each visibility are different")

        # get visibility information
        amp = data['amp']
        phase = data['phase']
        viserror = data['sigma']
        u = data['u']
        v = data['v']
        
        # convert antenna name strings to number identifiers
        ant1 = writeData.convertStrings(data['t1'], union)
        ant2 = writeData.convertStrings(data['t2'], union)
        
        # convert times to datetime objects
        time = data['time']
        dttime = np.array([datetime.datetime.utcfromtimestamp(x*60*60) for x in time]); #TODO: these do not correspond to the acutal times
        
        # get the bispectrum information
        bi = biarr['bispec']
        t3amp = np.abs(bi);
        t3phi = np.angle(bi, deg=1)
        t3amperr = biarr['sigmab']
        t3phierr = 180.0/np.pi * (1/t3amp) * t3amperr;
        uClosure = np.transpose(np.array([np.array(biarr['u1']), np.array(biarr['u2'])]));
        vClosure = np.transpose(np.array([np.array(biarr['v1']), np.array(biarr['v2'])]));
        
        # convert times to datetime objects
        timeClosure = biarr['time']
        dttimeClosure = np.array([datetime.datetime.utcfromtimestamp(x) for x in timeClosure]); #TODO: these do not correspond to the acutal times

        # convert antenna name strings to number identifiers
        biarr_ant1 = writeData.convertStrings(biarr['t1'], union)
        biarr_ant2 = writeData.convertStrings(biarr['t2'], union)
        biarr_ant3 = writeData.convertStrings(biarr['t3'], union)
        antOrder = np.transpose(np.array([biarr_ant1, biarr_ant2, biarr_ant3]))

        # todo: check that putting the negatives on the phase and t3phi is correct
        writeData.writeOIFITS(fname, self.ra, self.dec, self.rf, self.bw, intTime, amp, viserror, phase, viserror, u, v, ant1, ant2, dttime, 
                              t3amp, t3amperr, t3phi, t3phierr, uClosure, vClosure, antOrder, dttimeClosure, antennaNames, antennaDiam, antennaX, antennaY, antennaZ)
   
        # Un-Normalizing by the total flux passed in - note this is changing the data inside the obs structure back to what it originally was
        self.data['vis'] *= flux
        self.data['sigma'] *= flux
        
        return
        
##################################################################################################
# Object Construction Functions
##################################################################################################    
           
def load_array(filename):
    """Read an array from a text file and return an Array object
    """
    
    tdata = np.loadtxt(filename,dtype=str,comments='#')
    if (tdata.shape[1] != 5 and tdata.shape[1] != 13):
        raise Exception("Array file should have format: "+ 
                        "(name, x, y, z, SEFDR, SEFDL "+
                        "FR_PAR_ANGLE FR_ELEV_ANGLE FR_OFFSET" +
                        "DR_RE   DR_IM   DL_RE    DL_IM )") 
    if tdata.shape[1] == 5:                    
    	tdataout = [np.array((x[0],float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[4]),0.0, 0.0, 0.0, 0.0, 0.0),
                           dtype=DTARR) for x in tdata]
    elif tdata.shape[1] == 13:                    
    	tdataout = [np.array((x[0],float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5]),
                           float(x[9])+1j*float(x[10]), float(x[11])+1j*float(x[12]), 
                           float(x[6]), float(x[7]), float(x[8])), 
                           dtype=DTARR) for x in tdata]                           

    tdataout = np.array(tdataout)
    return Array(tdataout)
      
def load_obs_txt(filename):
    """Read an observation from a text file and return an Obsdata object
       text file has the same format as output from Obsdata.savedata()
    """
    
    # Read the header parameters
    file = open(filename)
    src = string.join(file.readline().split()[2:])
    ra = file.readline().split()
    ra = float(ra[2]) + float(ra[4])/60. + float(ra[6])/3600.
    dec = file.readline().split()
    dec = np.sign(float(dec[2])) *(abs(float(dec[2])) + float(dec[4])/60. + float(dec[6])/3600.)
    mjd = float(file.readline().split()[2])
    rf = float(file.readline().split()[2]) * 1e9
    bw = float(file.readline().split()[2]) * 1e9
    phasecal = bool(file.readline().split()[2])
    ampcal = bool(file.readline().split()[2])
    
    # New Header Parameters
    x = file.readline().split()
    if x[1] == 'OPACITYCAL:':
        opacitycal = bool(x[2])
        dcal = bool(file.readline().split()[2])
        frcal = bool(file.readline().split()[2])
        file.readline()
    else: 
        opacitycal = True
        dcal = True
        frcal = True
    file.readline()
    
    # read the tarr
    line = file.readline().split()
    tarr = []
    while line[1][0] != "-":
        if len(line) == 6: 
        	tarr.append(np.array((line[1], line[2], line[3], line[4], line[5], line[5], 0, 0, 0, 0, 0), dtype=DTARR))
        elif len(line) == 14: 
        	tarr.append(np.array((line[1], line[2], line[3], line[4], line[5], line[6], 
        	                      float(line[10])+1j*float(line[11]), float(line[12])+1j*float(line[13]), 
        	                      line[7], line[8], line[9]), dtype=DTARR))
        else: raise Exception("Telescope header doesn't have the right number of fields!")
        line = file.readline().split()
    tarr = np.array(tarr, dtype=DTARR)   
    file.close()

    # Load the data, convert to list format, return object
    datatable = np.loadtxt(filename, dtype=str)
    datatable2 = []
    for row in datatable:
        time = float(row[0])
        tint = float(row[1])
        t1 = row[2]
        t2 = row[3]
        
        #Old datatable formats
        if datatable.shape[1] < 20:
            tau1 = float(row[6])
            tau2 = float(row[7])
            u = float(row[8])
            v = float(row[9])
            vis = float(row[10]) * np.exp(1j * float(row[11]) * DEGREE)
            if datatable.shape[1] == 19:
                qvis = float(row[12]) * np.exp(1j * float(row[13]) * DEGREE)
                uvis = float(row[14]) * np.exp(1j * float(row[15]) * DEGREE)
                vvis = float(row[16]) * np.exp(1j * float(row[17]) * DEGREE)
                sigma = qsigma = usigma = vsigma = float(row[18])
            elif datatable.shape[1] == 17:
                qvis = float(row[12]) * np.exp(1j * float(row[13]) * DEGREE)
                uvis = float(row[14]) * np.exp(1j * float(row[15]) * DEGREE)
                vvis = 0+0j
                sigma = qsigma = usigma = vsigma = float(row[16])
            elif datatable.shape[1] == 15:
                qvis = 0+0j
                uvis = 0+0j
                vvis = 0+0j
                sigma = qsigma = usigma = vsigma = float(row[12])
            else:
                raise Exception('Text file does not have the right number of fields!')
        
        # Current datatable format
        elif datatable.shape[1] == 20:
            tau1 = float(row[4])
            tau2 = float(row[5])
            u = float(row[6])
            v = float(row[7])
            vis = float(row[8]) * np.exp(1j * float(row[9]) * DEGREE)
            qvis = float(row[10]) * np.exp(1j * float(row[11]) * DEGREE)
            uvis = float(row[12]) * np.exp(1j * float(row[14]) * DEGREE)
            vvis = float(row[14]) * np.exp(1j * float(row[15]) * DEGREE)
            sigma = float(row[16])
            qsigma = float(row[17])
            usigma = float(row[18])            
            vsigma = float(row[19])
                                                
        else:
            raise Exception('Text file does not have the right number of fields!')            
        
        
        datatable2.append(np.array((time, tint, t1, t2, tau1, tau2, 
                                    u, v, vis, qvis, uvis, vvis, 
                                    sigma, qsigma, usigma, vsigma), dtype=DTPOL))
    
    # Return the data object                      
    datatable2 = np.array(datatable2)
    out =  Obsdata(ra, dec, rf, bw, datatable2, tarr, source=src, mjd=mjd, 
                   ampcal=ampcal, phasecal=phasecal, opacitycal=opacitycal, dcal=dcal, frcal=frcal)        
    return out
    
def load_obs_maps(arrfile, obsspec, ifile, qfile=0, ufile=0, vfile=0, src=SOURCE_DEFAULT, mjd=MJD_DEFAULT, ampcal=False, phasecal=False):
    """Read an observation from a maps text file and return an Obsdata object
       text file has the same format as output from Obsdata.savedata()
    """
    # Read telescope parameters from the array file
    tdata = np.loadtxt(arrfile, dtype=str)
    tdata = [np.array((x[0],float(x[1]),float(x[2]),float(x[3]),float(x[-1])), dtype=DTARR) for x in tdata]
    tdata = np.array(tdata)

    # Read parameters from the obs_spec
    f = open(obsspec)
    stop = False
    while not stop:
        line = f.readline().split()
        if line==[] or line[0]=='\\':
            continue
        elif line[0] == 'FOV_center_RA':
            x = line[2].split(':')
            ra = float(x[0]) + float(x[1])/60. + float(x[2])/3600.
        elif line[0] == 'FOV_center_Dec':
            x = line[2].split(':')
            dec = np.sign(float(x[0])) * (abs(float(x[0])) + float(x[1])/60. + float(x[2])/3600.)
        elif line[0] == 'Corr_int_time':
            tint = float(line[2])
        elif line[0] == 'Corr_chan_bw':  #!AC TODO what if multiple channels?
            bw = float(line[2]) * 1e6 #in MHz
        elif line[0] == 'Channel': #!AC TODO what if multiple scans with different params?
            rf = float(line[2].split(':')[0]) * 1e6
        elif line[0] == 'Scan_start':
            x = line[2].split(':') #!AC properly compute MJD! 
        elif line[0] == 'Endscan':
            stop=True
    f.close()
    
    # Load the data, convert to list format, return object
    datatable = []
    f = open(ifile)
    
    for line in f:
        line = line.split()
        if not (line[0] in ['UV', 'Scan','\n']):
            time = line[0].split(':')
            time = float(time[2]) + float(time[3])/60. + float(time[4])/3600.
            u = float(line[1]) * 1000
            v = float(line[2]) * 1000
            bl = line[4].split('-')
            t1 = tdata[int(bl[0])-1]['site']
            t2 = tdata[int(bl[1])-1]['site']
            tau1 = 0.
            tau2 = 0.
            vis = float(line[7][:-1]) * np.exp(1j*float(line[8][:-1])*DEGREE)
            sigma = float(line[10])
            datatable.append(np.array((time, tint, t1, t2, tau1, tau2, 
                                        u, v, vis, 0.0, 0.0, 0.0, 
                                        sigma, 0.0, 0.0, 0.0), dtype=DTPOL))
    
    datatable = np.array(datatable)
    
    #!AC TODO qfile ufile and vfile must have exactly the same format as ifile: add some consistency check 
    if not qfile==0:
        f = open(qfile)
        i = 0
        for line in f:
            line = line.split()
            if not (line[0] in ['UV', 'Scan','\n']):
                datatable[i]['qvis'] = float(line[7][:-1]) * np.exp(1j*float(line[8][:-1])*DEGREE)
                datatable[i]['qsigma'] = float(line[10])
                i += 1
            
    if not ufile==0:
        f = open(ufile)
        i = 0
        for line in f:
            line = line.split()
            if not (line[0] in ['UV', 'Scan','\n']):
                datatable[i]['uvis'] = float(line[7][:-1]) * np.exp(1j*float(line[8][:-1])*DEGREE)
                datatable[i]['usigma'] = float(line[10])                
                i += 1                                
    
    if not vfile==0:
        f = open(vfile)
        i = 0
        for line in f:
            line = line.split()
            if not (line[0] in ['UV', 'Scan','\n']):
                datatable[i]['vvis'] = float(line[7][:-1]) * np.exp(1j*float(line[8][:-1])*DEGREE)
                datatable[i]['vsigma'] = float(line[10])                
                i += 1         
                                           
    # Return the datatable                      
    return Obsdata(ra, dec, rf, bw, datatable, tdata, source=src, mjd=mjd)        

#!AC TODO can we save new telescope array terms and flags to uvfits and load them?
def load_obs_uvfits(filename, flipbl=False):
    """Load uvfits data from a uvfits file.
    """
        
    # Load the uvfits file
    hdulist = fits.open(filename)
    header = hdulist[0].header
    data = hdulist[0].data
    
    # Load the array data
    tnames = hdulist['AIPS AN'].data['ANNAME']
    tnums = hdulist['AIPS AN'].data['NOSTA'] - 1
    xyz = hdulist['AIPS AN'].data['STABXYZ']
    try:
        sefdr = np.real(hdulist['AIPS AN'].data['SEFD'])
        sefdl = np.real(hdulist['AIPS AN'].data['SEFD']) #!AC TODO add sefdl to uvfits?
    except KeyError:
        print "Warning! no SEFD data in UVfits file"
        sefdr = np.zeros(len(tnames))
        sefdl = np.zeros(len(tnames))
    
    #!AC TODO - get the *real* values of these telescope parameters
    fr_par = np.zeros(len(tnames))
    fr_el = np.zeros(len(tnames))
    fr_off = np.zeros(len(tnames))
    dr = np.zeros(len(tnames)) + 1j*np.zeros(len(tnames))
    dl = np.zeros(len(tnames)) + 1j*np.zeros(len(tnames))
        
    tarr = [np.array((tnames[i], xyz[i][0], xyz[i][1], xyz[i][2], 
            sefdr[i], sefdl[i], fr_par[i], fr_el[i], fr_off[i],
            dr[i], dl[i]),
            dtype=DTARR) for i in range(len(tnames))]
            
    tarr = np.array(tarr)
    
    # Various header parameters
    ra = header['OBSRA'] * 12./180.
    dec = header['OBSDEC']   
    src = header['OBJECT']
    if header['CTYPE4'] == 'FREQ':
        rf = header['CRVAL4']
        bw = header['CDELT4'] 
    else: raise Exception('Cannot find observing frequency!')
    
    
    # Mask to screen bad data
    rrweight = data['DATA'][:,0,0,0,0,0,2]
    llweight = data['DATA'][:,0,0,0,0,1,2]
    rlweight = data['DATA'][:,0,0,0,0,2,2]
    lrweight = data['DATA'][:,0,0,0,0,3,2]
    mask = (rrweight > 0) * (llweight > 0) * (rlweight > 0) * (lrweight > 0)
    
    # Obs Times
    jds = data['DATE'][mask].astype('d') + data['_DATE'][mask].astype('d')
    mjd = int(np.min(jds)-2400000.5)
    times = (jds - 2400000.5 - mjd) * 24.0

    # old time conversion
    #jds = data['DATE'][mask]
    #mjd = int(jdtomjd(np.min(jds)))
    #print len(set(data['DATE']))
    #if len(set(data['DATE'])) > 2:
    #    times = np.array([mjdtogmt(jdtomjd(jd)) for jd in jds])
    #else:
    #    times = data['_DATE'][mask] * 24.0
    
    # Integration times
    tints = data['INTTIM'][mask]
    
    # Sites - add names
    t1 = data['BASELINE'][mask].astype(int)/256
    t2 = data['BASELINE'][mask].astype(int) - t1*256
    t1 = t1 - 1
    t2 = t2 - 1
    scopes_num = np.sort(list(set(np.hstack((t1,t2)))))
    t1 = np.array([tarr[i]['site'] for i in t1])
    t2 = np.array([tarr[i]['site'] for i in t2])

    # Opacities (not in BU files)
    try: 
        tau1 = data['TAU1'][mask]
        tau2 = data['TAU2'][mask]
    except KeyError:
        tau1 = tau2 = np.zeros(len(t1))
        
    # Convert uv in lightsec to lambda by multiplying by rf
    try:
        u = data['UU---SIN'][mask] * rf
        v = data['VV---SIN'][mask] * rf    
    except KeyError:   
        try:
            u = data['UU'][mask] * rf
            v = data['VV'][mask] * rf
        except KeyError:
            try:
                u = data['UU--'][mask] * rf
                v = data['VV--'][mask] * rf
            except KeyError:
                raise Exception("Cant figure out column label for UV coords")
                    
    # Get vis data
    rr = data['DATA'][:,0,0,0,0,0,0][mask] + 1j*data['DATA'][:,0,0,0,0,0,1][mask]
    ll = data['DATA'][:,0,0,0,0,1,0][mask] + 1j*data['DATA'][:,0,0,0,0,1,1][mask]
    rl = data['DATA'][:,0,0,0,0,2,0][mask] + 1j*data['DATA'][:,0,0,0,0,2,1][mask]
    lr = data['DATA'][:,0,0,0,0,3,0][mask] + 1j*data['DATA'][:,0,0,0,0,3,1][mask]
    rrsig = 1/np.sqrt(rrweight[mask])
    llsig = 1/np.sqrt(llweight[mask])
    rlsig = 1/np.sqrt(rlweight[mask])
    lrsig = 1/np.sqrt(lrweight[mask])
    
    # Form stokes parameters
    ivis = 0.5 * (rr + ll)
    qvis = 0.5 * (rl + lr)
    uvis = 0.5j * (lr - rl)
    vvis = 0.5 * (rr - ll)
    sigma = 0.5 * np.sqrt(rrsig**2 + llsig**2)
    qsigma = 0.5* np.sqrt(rlsig**2 + lrsig**2)
    usigma = qsigma
    vsigma = sigma
   
    # Reverse sign of baselines for correct imaging?
    if flipbl:
        u = -u
        v = -v
    
    # Make a datatable
    datatable = []
    for i in xrange(len(times)):
        datatable.append(np.array
                         ((
                           times[i], tints[i], 
                           t1[i], t2[i], tau1[i], tau2[i], 
                           u[i], v[i],
                           ivis[i], qvis[i], uvis[i], vvis[i],
                           sigma[i], qsigma[i], usigma[i], vsigma[i],
                           ), dtype=DTPOL
                         ))
    datatable = np.array(datatable)
    
    #!AC TODO get calibration flags from uvfits?
    return Obsdata(ra, dec, rf, bw, datatable, tarr, source=src, mjd=mjd)

def load_obs_oifits(filename, flux=1.0):
    """Load data from an oifits file
       Does NOT currently support polarization
    """
    
    print 'Warning: load_obs_oifits does NOT currently support polarimetric data!' 
    
    # open oifits file and get visibilities
    oidata=oifits.open(filename)
    vis_data = oidata.vis

    # get source info
    src = oidata.target[0].target
    ra = oidata.target[0].raep0.angle
    dec = oidata.target[0].decep0.angle
    
    # get annena info
    nAntennas = len(oidata.array[oidata.array.keys()[0]].station)
    sites = np.array([oidata.array[oidata.array.keys()[0]].station[i].sta_name for i in range(nAntennas)])
    arrayX = oidata.array[oidata.array.keys()[0]].arrxyz[0]
    arrayY = oidata.array[oidata.array.keys()[0]].arrxyz[1]
    arrayZ = oidata.array[oidata.array.keys()[0]].arrxyz[2]
    x = np.array([arrayX + oidata.array[oidata.array.keys()[0]].station[i].staxyz[0] for i in range(nAntennas)])
    y = np.array([arrayY + oidata.array[oidata.array.keys()[0]].station[i].staxyz[1] for i in range(nAntennas)])
    z = np.array([arrayZ + oidata.array[oidata.array.keys()[0]].station[i].staxyz[2] for i in range(nAntennas)])
    
    # get wavelength and corresponding frequencies
    wavelength = oidata.wavelength[oidata.wavelength.keys()[0]].eff_wave
    nWavelengths = wavelength.shape[0]
    bandpass = oidata.wavelength[oidata.wavelength.keys()[0]].eff_band
    frequency = C/wavelength
    
    # !AC TODO: this result seems wrong...
    bw = np.mean(2*(np.sqrt( bandpass**2*frequency**2 + C**2) - C)/bandpass)
    rf = np.mean(frequency)
    
    # get the u-v point for each visibility
    u = np.array([vis_data[i].ucoord/wavelength for i in range(len(vis_data))])
    v = np.array([vis_data[i].vcoord/wavelength for i in range(len(vis_data))])
    
    # get visibility info - currently the phase error is not being used properly
    amp = np.array([vis_data[i]._visamp for i in range(len(vis_data))])
    phase = np.array([vis_data[i]._visphi for i in range(len(vis_data))])
    amperr = np.array([vis_data[i]._visamperr for i in range(len(vis_data))])
    visphierr = np.array([vis_data[i]._visphierr for i in range(len(vis_data))])
    timeobs = np.array([vis_data[i].timeobs for i in range(len(vis_data))]) #convert to single number
    
    #return timeobs
    time = np.transpose(np.tile(np.array([(ttime.mktime((timeobs[i] + datetime.timedelta(days=1)).timetuple()))/(60.0*60.0) 
                                        for i in range(len(timeobs))]), [nWavelengths, 1]))
    
    # integration time
    tint = np.array([vis_data[i].int_time for i in range(len(vis_data))])
    if not all(tint[0] == item for item in np.reshape(tint, (-1)) ):
        raise TypeError("The time integrations for each visibility are different")
    tint = tint[0]
    tint = tint * np.ones( amp.shape )

    # get telescope names for each visibility
    t1 = np.transpose(np.tile( np.array([ vis_data[i].station[0].sta_name for i in range(len(vis_data))]), [nWavelengths,1]))
    t2 = np.transpose(np.tile( np.array([ vis_data[i].station[1].sta_name for i in range(len(vis_data))]), [nWavelengths,1]))

    # dummy variables
    tau1 = np.zeros(amp.shape)
    tau2 = np.zeros(amp.shape)
    qvis = np.zeros(amp.shape)
    uvis = np.zeros(amp.shape)
    vvis = np.zeros(amp.shape)
    sefdr = np.zeros(x.shape)
    sefdl = np.zeros(x.shape)
    fr_par = np.zeros(x.shape)   
    fr_el = np.zeros(x.shape)
    fr_off = np.zeros(x.shape)
    dr = np.zeros(x.shape) + 1j*np.zeros(x.shape)
    dl = np.zeros(x.shape) + 1j*np.zeros(x.shape)
             
    # vectorize
    time = time.ravel()
    tint = tint.ravel()
    t1 = t1.ravel()
    t2 = t2.ravel()

    tau1 = tau1.ravel()
    tau2 = tau2.ravel()
    u = u.ravel()
    v = v.ravel()
    vis = amp.ravel() * np.exp ( -1j * phase.ravel() * np.pi/180.0 )
    qvis = qvis.ravel()
    uvis = uvis.ravel()
    vvis = vvis.ravel()
    amperr = amperr.ravel()

    #!AC TODO - check that we are properly using the error from the amplitude and phase
    # create data tables
    datatable = np.array([(time[i], tint[i], t1[i], t2[i], tau1[i], tau2[i], u[i], v[i], 
                           flux*vis[i], qvis[i], uvis[i], vvis[i], 
                           flux*amperr[i], flux*amperr[i], flux*amperr[i], flux*amperr[i]
                          ) for i in range(len(vis))
                         ], dtype=DTPOL)

    tarr = np.array([(sites[i], x[i], y[i], z[i], 
                       sefdr[i], sefdl[i], fr_par[i], fr_el[i], fr_off[i],
                       dr[i], dl[i]
                     ) for i in range(nAntennas)
                    ], dtype=DTARR)
    
    # return object

    return Obsdata(ra, dec, rf, bw, datatable, tarr, source=src, mjd=time[0])
    
def load_im_txt(filename, pulse=PULSE_DEFAULT):
    """Read in an image from a text file and create an Image object
       Text file should have the same format as output from Image.save_txt()
       Make sure the header has exactly the same form!
    """
    
    # Read the header
    file = open(filename)
    src = string.join(file.readline().split()[2:])
    ra = file.readline().split()
    ra = float(ra[2]) + float(ra[4])/60. + float(ra[6])/3600.
    dec = file.readline().split()
    dec = np.sign(float(dec[2])) *(abs(float(dec[2])) + float(dec[4])/60. + float(dec[6])/3600.)
    mjd = int(float(file.readline().split()[2]))
    rf = float(file.readline().split()[2]) * 1e9
    xdim = file.readline().split()
    xdim_p = int(xdim[2])
    psize_x = float(xdim[4])*RADPERAS/xdim_p
    ydim = file.readline().split()
    ydim_p = int(ydim[2])
    psize_y = float(ydim[4])*RADPERAS/ydim_p
    file.close()
    
    if psize_x != psize_y:
        raise Exception("Pixel dimensions in x and y are inconsistent!")
    
    # Load the data, convert to list format, make object
    datatable = np.loadtxt(filename, dtype=float)
    image = datatable[:,2].reshape(ydim_p, xdim_p)
    outim = Image(image, psize_x, ra, dec, rf=rf, source=src, mjd=mjd, pulse=pulse)
    
    # Look for Stokes Q and U
    qimage = uimage = vimage = np.zeros(image.shape)
    if datatable.shape[1] == 6:
        qimage = datatable[:,3].reshape(ydim_p, xdim_p)
        uimage = datatable[:,4].reshape(ydim_p, xdim_p)
        vimage = datatable[:,5].reshape(ydim_p, xdim_p)
    elif datatable.shape[1] == 5:
        qimage = datatable[:,3].reshape(ydim_p, xdim_p)
        uimage = datatable[:,4].reshape(ydim_p, xdim_p)    
    
    if np.any((qimage != 0) + (uimage != 0)) and np.any((vimage != 0)):
        print 'Loaded Stokes I, Q, U, and V Images'
        outim.add_qu(qimage, uimage)
        outim.add_v(vimage)
    elif np.any((vimage != 0)):
        print 'Loaded Stokes I and V Images'
        outim.add_v(vimage)
    elif np.any((qimage != 0) + (uimage != 0)):   
        print 'Loaded Stokes I, Q, and U Images'
        outim.add_qu(qimage, uimage)     
    else:
        print 'Loaded Stokes I Image Only'
    
    return outim
    
def load_im_fits(filename, punit="deg", pulse=PULSE_DEFAULT):
    """Read in an image from a FITS file and create an Image object
    """

    # Radian or Degree?
    if punit=="deg":
        pscl = DEGREE
    elif punit=="rad":
        pscl = 1.0
    elif punit=="uas":
        pscl = RADPERUAS
        
    # Open the FITS file
    hdulist = fits.open(filename)
    
    # Assume stokes I is the primary hdu
    header = hdulist[0].header
    
    # Read some header values
    ra = header['OBSRA']*12/180.
    dec = header['OBSDEC']
    xdim_p = header['NAXIS1']
    psize_x = np.abs(header['CDELT1']) * pscl
    dim_p = header['NAXIS2']
    psize_y = np.abs(header['CDELT2']) * pscl
    
    if 'MJD' in header.keys(): mjd = header['MJD']
    else: mjd = 0.0 
    
    if 'FREQ' in header.keys(): rf = header['FREQ']
    else: rf = 0.0
    
    if 'OBJECT' in header.keys(): src = header['OBJECT']
    else: src = ''
    
    # Get the image and create the object
    data = hdulist[0].data
    data = data.reshape((data.shape[-2],data.shape[-1]))
    image = data[::-1,:] # flip y-axis!
    
    # normalize the flux
    normalizer = 1.0;
    if 'BUNIT' in header.keys():
        if header['BUNIT'] == 'JY/BEAM':
            beamarea = (2.0*np.pi*header['BMAJ']*header['BMIN']/(8.0*np.log(2)))
            normalizer = (header['CDELT2'])**2 / beamarea
    image *= normalizer
            
    # make image object            
    outim = Image(image, psize_x, ra, dec, rf=rf, source=src, mjd=mjd, pulse=pulse)
    
    # Look for Stokes Q and U
    qimage = uimage = vimage = np.array([])
    for hdu in hdulist[1:]:
        header = hdu.header
        data = hdu.data
        try: data = data.reshape((data.shape[-2],data.shape[-1]))
        except IndexError: continue
        
        if 'STOKES' in header.keys() and header['STOKES'] == 'Q':
            qimage = normalizer*data[::-1,:] # flip y-axis!
        if 'STOKES' in header.keys() and header['STOKES'] == 'U':
            uimage = normalizer*data[::-1,:] # flip y-axis!
        if 'STOKES' in header.keys() and header['STOKES'] == 'V':
            vimage = normalizer*data[::-1,:] # flip y-axis!
    
    if qimage.shape == uimage.shape == vimage.shape == image.shape:
        print 'Loaded Stokes I, Q, U, and V Images'
        outim.add_qu(qimage, uimage)
        outim.add_v(vimage)
    elif vimage.shape == image.shape:
        print 'Loaded Stokes I and V Images'
        outim.add_v(vimage)
    elif qimage.shape == uimage.shape == image.shape:   
        print 'Loaded Stokes I, Q, and U Images'
        outim.add_qu(qimage, uimage)     
    else:
        print 'Loaded Stokes I Image Only'
                            
    return outim

#!AC TODO - Michael...what exactly is this for? 
def load_im_manual_fits(filename, timesrot90=0, punit="deg", fov=-1, ra=RA_DEFAULT, dec=DEC_DEFAULT,
                        rf=RF_DEFAULT, src=SOURCE_DEFAULT, mjd=MJD_DEFAULT , pulse=PULSE_DEFAULT):

    # Radian or Degree?
    if punit=="deg":
        pscl = DEGREE
    elif punit=="rad":
        pscl = 1.0
    elif punit=="uas":
        pscl = RADPERUAS
    elif punit=="mas":
        pscl = RADPERUAS * 1000.0


    hdulist = fits.open(filename)
    header = hdulist[0].header
    data = hdulist[0].data

    if 'NAXIS1' in header.keys(): xdim_p = header['NAXIS1']
    else: xdim_p = data.shape[-2]

    if 'CDELT1' in header.keys(): 
        psize_x = np.abs(header['CDELT1']) * pscl
    else: 
        psize_x = (float(fov) / data.shape[-2]) * pscl
        if fov==-1:
            print 'WARNING: Must provide a field of view for the image'

    normalizer = 1.0; 
    if 'BUNIT' in header.keys():
        if header['BUNIT'] == 'JY/BEAM':
            beamarea = (2.0*np.pi*header['BMAJ']*header['BMIN']/(8.0*np.log(2)))
            normalizer = (header['CDELT2'])**2 / beamarea

    data = data.reshape((data.shape[-2],data.shape[-1]))

    image = data[::-1,:] # flip y-axis!
    image = np.rot90(image, k=timesrot90)
    outim = Image(image*normalizer, psize_x, ra, dec, rf=rf, source=src, mjd=mjd, pulse=pulse)
    
    print 'Loaded Stokes I image only'
    return outim


##################################################################################################
# Image Construction Functions
##################################################################################################

def resample_square(im, xdim_new, ker_size=5):
    """Return a new image object that is resampled to the new dimensions xdim_new x xdim_new
    """
    
    if im.xdim != im.ydim:
        raise Exception("Image must be square!")
    if im.pulse == pulses.deltaPulse2D:
        raise Exception("This function only works on continuously parametrized images: does not work with delta pulses!")
    
    ydim_new = xdim_new
    fov = im.xdim * im.psize
    psize_new = fov / xdim_new
    ij = np.array([[[i*im.psize + (im.psize*im.xdim)/2.0 - im.psize/2.0, j*im.psize + (im.psize*im.ydim)/2.0 - im.psize/2.0]
                    for i in np.arange(0, -im.xdim, -1)] 
                    for j in np.arange(0, -im.ydim, -1)]).reshape((im.xdim*im.ydim, 2))
    def im_new(x,y):
        mask = (((x - ker_size*im.psize/2.0) < ij[:,0]) * (ij[:,0] < (x + ker_size*im.psize/2.0)) * ((y-ker_size*im.psize/2.0) < ij[:,1]) * (ij[:,1] < (y+ker_size*im.psize/2.0))).flatten()
        return np.sum([im.imvec[n] * im.pulse(x-ij[n,0], y-ij[n,1], im.psize, dom="I") for n in np.arange(len(im.imvec))[mask]])
    
    out = np.array([[im_new(x*psize_new + (psize_new*xdim_new)/2.0 - psize_new/2.0, y*psize_new + (psize_new*ydim_new)/2.0 - psize_new/2.0)
                      for x in np.arange(0, -xdim_new, -1)] 
                      for y in np.arange(0, -ydim_new, -1)] )                     

                      
    # Normalize
    scaling = np.sum(im.imvec) / np.sum(out)
    out *= scaling
    outim = Image(out, psize_new, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse)
    
    # Q and U images
    if len(im.qvec):
        def im_new_q(x,y):
            mask = (((x - ker_size*im.psize/2.0) < ij[:,0]) * (ij[:,0] < (x + ker_size*im.psize/2.0)) * 
                    ((y - ker_size*im.psize/2.0) < ij[:,1]) * (ij[:,1] < (y + ker_size*im.psize/2.0))).flatten()
            return np.sum([im.qvec[n] * im.pulse(x-ij[n,0], y-ij[n,1], im.psize, dom="I") for n in np.arange(len(im.imvec))[mask]])
        def im_new_u(x,y):
            mask = (((x - ker_size*im.psize/2.0) < ij[:,0]) * (ij[:,0] < (x + ker_size*im.psize/2.0)) * 
                    ((y-ker_size*im.psize/2.0) < ij[:,1]) * (ij[:,1] < (y+ker_size*im.psize/2.0))).flatten()
            return np.sum([im.uvec[n] * im.pulse(x-ij[n,0], y-ij[n,1], im.psize, dom="I") for n in np.arange(len(im.imvec))[mask]])
        
        outq = np.array([[im_new_q(x*psize_new + (psize_new*xdim_new)/2.0 - psize_new/2.0, y*psize_new + (psize_new*ydim_new)/2.0 - psize_new/2.0)
                      for x in np.arange(0, -xdim_new, -1)] 
                      for y in np.arange(0, -ydim_new, -1)] ) 
        outu = np.array([[im_new_u(x*psize_new + (psize_new*xdim_new)/2.0 - psize_new/2.0, y*psize_new + (psize_new*ydim_new)/2.0 - psize_new/2.0)
                      for x in np.arange(0, -xdim_new, -1)] 
                      for y in np.arange(0, -ydim_new, -1)] )
        outq *= scaling
        outu *= scaling
        outim.add_qu(outq, outu)
    
    if len(im.vvec):
        def im_new_v(x,y):
            mask = (((x - ker_size*im.psize/2.0) < ij[:,0]) * (ij[:,0] < (x + ker_size*im.psize/2.0)) * 
                    ((y-ker_size*im.psize/2.0) < ij[:,1]) * (ij[:,1] < (y+ker_size*im.psize/2.0))).flatten()
            return np.sum([im.vvec[n] * im.pulse(x-ij[n,0], y-ij[n,1], im.psize, dom="I") for n in np.arange(len(im.imvec))[mask]])
        
        outv = np.array([[im_new_v(x*psize_new + (psize_new*xdim_new)/2.0 - psize_new/2.0, y*psize_new + (psize_new*ydim_new)/2.0 - psize_new/2.0)
                      for x in np.arange(0, -xdim_new, -1)] 
                      for y in np.arange(0, -ydim_new, -1)] ) 
        outv *= scaling
        outim.add_v(outv)        
    
    return outim

def im_pad(im, fovx, fovy):
    """Pad an image to new fov
    """ 

    fovoldx=im.psize*im.xdim
    fovoldy=im.psize*im.ydim
    padx=int(0.5*(fovx-fovoldx)/im.psize)
    pady=int(0.5*(fovy-fovoldy)/im.psize)
    imarr=im.imvec.reshape(im.ydim, im.xdim)
    imarr=np.pad(imarr,((padx,padx),(pady,pady)),'constant')
    outim=Image(imarr, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse)

    if len(im.qvec):
        qarr=im.qvec.reshape(im.ydim,im.xdim)
        qarr=np.pad(qarr,((padx,padx),(pady,pady)),'constant')
        uarr=im.uvec.reshape(im.ydim,im.xdim)
        uarr=np.pad(uarr,((padx,padx),(pady,pady)),'constant')
        outim.add_qu(qarr,uarr)
    if len(im.vvec):
        varr=im.vvec.reshape(im.ydim,im.xdim)
        varr=np.pad(qarr,((padx,padx),(pady,pady)),'constant')
        outim.add_v(varr)
    return outim
    
def make_square(obs, npix, fov,pulse=PULSE_DEFAULT):
    """Make an empty prior image
       obs is an observation object
       fov is in radians
    """ 
    pdim = fov/npix
    im = np.zeros((npix,npix))
    return Image(im, pdim, obs.ra, obs.dec, rf=obs.rf, source=obs.source, mjd=obs.mjd, pulse=pulse)

def add_flat(im, flux):
    """Add flat background flux to an image
    """ 
    
    imout = (im.imvec + (flux/float(len(im.imvec))) * np.ones(len(im.imvec))).reshape(im.ydim,im.xdim)
    out = Image(imout, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse) 
    return out

def add_tophat(im, flux, radius):
    """Add tophat flux to an image
    """
 
    xfov = im.xdim * im.psize
    yfov = im.ydim * im.psize
    
    xlist = np.arange(0,-im.xdim,-1)*im.psize + (im.psize*im.xdim)/2.0 - im.psize/2.0
    ylist = np.arange(0,-im.ydim,-1)*im.psize + (im.psize*im.ydim)/2.0 - im.psize/2.0

    hat = np.array([[1.0 if np.sqrt(i**2+j**2) <= radius else EP
                      for i in xlist] 
                      for j in ylist])        
    
    hat = hat[0:im.ydim, 0:im.xdim]
    
    imout = im.imvec.reshape(im.ydim, im.xdim) + (hat * flux/np.sum(hat))
    out = Image(imout, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd) 
    return out

def add_gauss(im, flux, beamparams):
    """Add a gaussian to an image
       beamparams is [fwhm_maj, fwhm_min, theta, x, y], all in rad
       theta is the orientation angle measured E of N
    """ 
    
    try: 
    	x=beamparams[3]
    	y=beamparams[4] 
    except IndexError:
    	x=y=0.0
    	 
    sigma_maj = beamparams[0] / (2. * np.sqrt(2. * np.log(2.))) 
    sigma_min = beamparams[1] / (2. * np.sqrt(2. * np.log(2.)))
    cth = np.cos(beamparams[2])
    sth = np.sin(beamparams[2])
    
    xfov = im.xdim * im.psize
    yfov = im.ydim * im.psize    
    xlist = np.arange(0,-im.xdim,-1)*im.psize + (im.psize*im.xdim)/2.0 - im.psize/2.0
    ylist = np.arange(0,-im.ydim,-1)*im.psize + (im.psize*im.ydim)/2.0 - im.psize/2.0
    
    gauss = np.array([[np.exp(-((j-y)*cth + (i-x)*sth)**2/(2*sigma_maj**2) - ((i-x)*cth - (j-y)*sth)**2/(2.*sigma_min**2))
                      for i in xlist] 
                      for j in ylist]) 
  
    gauss = gauss[0:im.ydim, 0:im.xdim]
    
    imout = im.imvec.reshape(im.ydim, im.xdim) + (gauss * flux/np.sum(gauss))
    out = Image(imout, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse)
    return out

def add_crescent(im, flux, Rp, Rn, a, b, x=0, y=0):
    """Add a crescent to an image; see Kamruddin & Dexter (2013)
       all parameters in rad
       Rp is larger radius
       Rn is smaller radius
       a is relative x offset of smaller disk
       b is relative y offset of smaller disk
       x,y are center coordinates of the larger disk
    """ 
    
    xfov = im.xdim * im.psize
    yfov = im.ydim * im.psize    
    xlist = np.arange(0,-im.xdim,-1)*im.psize + (im.psize*im.xdim)/2.0 - im.psize/2.0
    ylist = np.arange(0,-im.ydim,-1)*im.psize + (im.psize*im.ydim)/2.0 - im.psize/2.0
    
    def mask(x2, y2):
        if (x2-a)**2 + (y2-b)**2 > Rn**2 and x2**2 + y2**2 < Rp**2:
            return 1.0
        else:
            return 0.0

    crescent = np.array([[mask(i-x, j-y)
                      for i in xlist] 
                      for j in ylist]) 
  
    crescent = crescent[0:im.ydim, 0:im.xdim]
    
    imout = im.imvec.reshape(im.ydim, im.xdim) + (crescent * flux/np.sum(crescent))
    out = Image(imout, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse)
    return out

def add_const_m(im, mag, angle):
    """Add a constant fractional linear polarization to image
       angle is in radians
    """ 
    
    if not (0 < mag < 1):
        raise Exception("fractional polarization magnitude must be beween 0 and 1!")
    
    imi = im.imvec.reshape(im.ydim,im.xdim)    
    imq = qimage(im.imvec, mag * np.ones(len(im.imvec)), angle*np.ones(len(im.imvec))).reshape(im.ydim,im.xdim)
    imu = uimage(im.imvec, mag * np.ones(len(im.imvec)), angle*np.ones(len(im.imvec))).reshape(im.ydim,im.xdim)
    out = Image(imi, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse)
    out.add_qu(imq, imu)
    return out
    
##################################################################################################
# Image domain blurring Functions
##################################################################################################
def blur_gauss(image, beamparams, frac, frac_pol=0):
    """Blur image with a Gaussian beam defined by beamparams
       beamparams is [FWHMmaj, FWHMmin, theta], all in radian
    """
    
    im = (image.imvec).reshape(image.ydim, image.xdim)
    if len(image.qvec):
        qim = (image.qvec).reshape(image.ydim, image.xdim)
        uim = (image.uvec).reshape(image.ydim, image.xdim)
    if len(image.vvec):
        vim = (image.vvec).reshape(image.ydim, image.xdim)
    xfov = image.xdim * image.psize
    yfov = image.ydim * image.psize
    xlist = np.arange(0,-image.xdim,-1)*image.psize + (image.psize*image.xdim)/2.0 - image.psize/2.0
    ylist = np.arange(0,-image.ydim,-1)*image.psize + (image.psize*image.ydim)/2.0 - image.psize/2.0        

    if beamparams[0] > 0.0:
        sigma_maj = frac * beamparams[0] / (2. * np.sqrt(2. * np.log(2.))) 
        sigma_min = frac * beamparams[1] / (2. * np.sqrt(2. * np.log(2.))) 
        cth = np.cos(beamparams[2])
        sth = np.sin(beamparams[2])

        gauss = np.array([[np.exp(-(j*cth + i*sth)**2/(2*sigma_maj**2) - (i*cth - j*sth)**2/(2.*sigma_min**2))
                                  for i in xlist] 
                                  for j in ylist])

        gauss = gauss[0:image.ydim, 0:image.xdim]
        gauss = gauss / np.sum(gauss) # normalize to 1

        # Convolve
        im = scipy.signal.fftconvolve(gauss, im, mode='same')



    if frac_pol:
        if not len(image.qvec):
            raise Exception("There is no polarized image!")
                
        sigma_maj = frac_pol * beamparams[0] / (2. * np.sqrt(2. * np.log(2.))) 
        sigma_min = frac_pol * beamparams[1] / (2. * np.sqrt(2. * np.log(2.))) 
        cth = np.cos(beamparams[2])
        sth = np.sin(beamparams[2])
        gauss = np.array([[np.exp(-(j*cth + i*sth)**2/(2*sigma_maj**2) - (i*cth - j*sth)**2/(2.*sigma_min**2))
                                  for i in xlist] 
                                  for j in ylist])
        

        gauss = gauss[0:image.ydim, 0:image.xdim]
        gauss = gauss / np.sum(gauss) # normalize to 1        
        
        # Convolve
        qim = scipy.signal.fftconvolve(gauss, qim, mode='same')
        uim = scipy.signal.fftconvolve(gauss, uim, mode='same')
        
        if len(image.vvec):
            vim = scipy.signal.fftconvolve(gauss, vim, mode='same')                                  
    
    out = Image(im, image.psize, image.ra, image.dec, rf=image.rf, source=image.source, mjd=image.mjd, pulse=image.pulse)                        
    if len(image.qvec):
        out.add_qu(qim, uim)
    if len(image.vvec):
        out.add_v(vim)
    return out  
        
##################################################################################################
# Scattering Functions
##################################################################################################
def deblur(obs):
    """Deblur the observation obs by dividing with the Sgr A* scattering kernel.
       Returns a new observation.
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
        ker = sgra_kernel_uv(obs.rf, u[i], v[i])
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
    
    obsdeblur = Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, datatable, obs.tarr, source=obs.source, mjd=obs.mjd, 
                        ampcal=obs.ampcal, phasecal=obs.phasecal, opacitycal=obs.opacitycal, dcal=obs.dcal, frcal=obs.frcal)
    return obsdeblur

def gauss_uv(u, v, flux, beamparams, x=0., y=0.):
    """Return the value of the Gaussian FT with 
       beamparams is [FWHMmaj, FWHMmin, theta, x, y], all in radian
       theta is the orientation angle measured E of N
    """

    sigma_maj = beamparams[0] / (2*np.sqrt(2*np.log(2))) 
    sigma_min = beamparams[1] / (2*np.sqrt(2*np.log(2)))
    theta = beamparams[2]
    #try: 
    #	x=beamparams[3]
    #	y=beamparams[4] 
    #except IndexError:
    #	x=y=0.0
    
    
    # Covariance matrix
    a = (sigma_min * np.cos(theta))**2 + (sigma_maj*np.sin(theta))**2
    b = (sigma_maj * np.cos(theta))**2 + (sigma_min*np.sin(theta))**2
    c = (sigma_min**2 - sigma_maj**2) * np.cos(theta) * np.sin(theta)
    m = np.array([[a, c], [c, b]])
    
    uv = np.array([[u[i],v[i]] for i in xrange(len(u))])
    x2 = np.array([np.dot(uvi,np.dot(m,uvi)) for uvi in uv])   
    #x2 = np.dot(uv, np.dot(m, uv.T))
    g = np.exp(-2 * np.pi**2 * x2)
    p = np.exp(-2j * np.pi * (u*x + v*y))

    return flux * g * p
    
def sgra_kernel_uv(rf, u, v):
    """Return the value of the Sgr A* scattering kernel at a given u,v pt (in lambda), 
       at a given frequency rf (in Hz).
       Values from Bower et al.
    """
    
    lcm = (C/rf) * 100 # in cm
    sigma_maj = FWHM_MAJ * (lcm**2) / (2*np.sqrt(2*np.log(2))) * RADPERUAS
    sigma_min = FWHM_MIN * (lcm**2) / (2*np.sqrt(2*np.log(2))) * RADPERUAS
    theta = POS_ANG * DEGREE
    
    
    # Covariance matrix
    a = (sigma_min * np.cos(theta))**2 + (sigma_maj*np.sin(theta))**2
    b = (sigma_maj * np.cos(theta))**2 + (sigma_min*np.sin(theta))**2
    c = (sigma_min**2 - sigma_maj**2) * np.cos(theta) * np.sin(theta)
    m = np.array([[a, c], [c, b]])
    uv = np.array([u,v])
    
    x2 = np.dot(uv, np.dot(m, uv))
    g = np.exp(-2 * np.pi**2 * x2)
    
    return g

def sgra_kernel_params(rf):
    """Return elliptical gaussian parameters in radian for the Sgr A* scattering ellipse at a given frequency
       Values from Bower et al.
    """
    
    lcm = (C/rf) * 100 # in cm
    fwhm_maj_rf = FWHM_MAJ * (lcm**2)  * RADPERUAS
    fwhm_min_rf = FWHM_MIN * (lcm**2)  * RADPERUAS
    theta = POS_ANG * DEGREE
    
    return np.array([fwhm_maj_rf, fwhm_min_rf, theta])
##################################################################################################
# Observation & Noise Functions
##################################################################################################        

def blnoise(sefd1, sefd2, tint, bw):
    """Determine the standard deviation of Gaussian thermal noise on a baseline 
       This is the noise on the rr/ll/rl/lr correlation, not the stokes parameter
       2-bit quantization is responsible for the 0.88 factor
    """
    
    #!AC TODO Is the factor of sqrt(2) correct? 
    noise = np.sqrt(sefd1*sefd2/(2*bw*tint))/0.88
    #noise = np.sqrt(sefd1*sefd2/(bw*tint))/0.88
    return noise

def cerror(sigma):
    """Return a complex number drawn from a circular complex Gaussian of zero mean
    """
    return np.random.normal(loc=0,scale=sigma) + 1j*np.random.normal(loc=0,scale=sigma)

def hashrandn(*args):
    """set the seed according to a collection of arguments and return random gaussian var
    """
    np.random.seed(hash(",".join(map(repr,args))) % 4294967295)
    return np.random.randn()

def hashrand(*args):
    """set the seed according to a collection of arguments and return random number in 0,1
    """
    np.random.seed(hash(",".join(map(repr,args))) % 4294967295)
    return np.random.rand()

def image_centroid(im):
    """Return the image centroid (in radians)
    """

    xlist = np.arange(0,-im.xdim,-1)*im.psize + (im.psize*im.xdim)/2.0 - im.psize/2.0
    ylist = np.arange(0,-im.ydim,-1)*im.psize + (im.psize*im.ydim)/2.0 - im.psize/2.0

    x0 = np.sum(np.outer(0.0*ylist+1.0, xlist).ravel()*im.imvec)/np.sum(im.imvec)
    y0 = np.sum(np.outer(ylist, 0.0*xlist+1.0).ravel()*im.imvec)/np.sum(im.imvec)

    return np.array([x0, y0])

def ftmatrix_centered(im, pdim, xdim, ydim, uvlist, pulse=pulses.deltaPulse2D):
    """Return a DFT matrix for the xdim*ydim image with pixel width pdim
       that extracts spatial frequencies of the uv points in uvlist.
       in this version, it puts the image centroid at the origin
    """

    # !AC TODO : there is a residual value for the center being around 0, maybe we should chop this off to be exactly 0
    # Coordinate matrix for COM constraint
    xlist = np.arange(0,-xdim,-1)*pdim + (pdim*xdim)/2.0 - pdim/2.0
    ylist = np.arange(0,-ydim,-1)*pdim + (pdim*ydim)/2.0 - pdim/2.0
    x0 = np.sum(np.outer(0.0*ylist+1.0, xlist).ravel()*im)/np.sum(im)
    y0 = np.sum(np.outer(ylist, 0.0*xlist+1.0).ravel()*im)/np.sum(im)

    #Now shift the lists
    xlist = xlist - x0
    ylist = ylist - y0

    ftmatrices = [pulse(2*np.pi*uv[0], 2*np.pi*uv[1], pdim, dom="F") * np.outer(np.exp(-2j*np.pi*ylist*uv[1]), np.exp(-2j*np.pi*xlist*uv[0])) for uv in uvlist] #list of matrices at each freq
    ftmatrices = np.reshape(np.array(ftmatrices), (len(uvlist), xdim*ydim))
    return ftmatrices
      
def ftmatrix(pdim, xdim, ydim, uvlist, pulse=pulses.deltaPulse2D, mask=[]):
    """Return a DFT matrix for the xdim*ydim image with pixel width pdim
       that extracts spatial frequencies of the uv points in uvlist.
    """

    xlist = np.arange(0,-xdim,-1)*pdim + (pdim*xdim)/2.0 - pdim/2.0
    ylist = np.arange(0,-ydim,-1)*pdim + (pdim*ydim)/2.0 - pdim/2.0

    # original sign convention
    #ftmatrices = [pulse(2*np.pi*uv[0], 2*np.pi*uv[1], pdim, dom="F") * np.outer(np.exp(-2j*np.pi*ylist*uv[1]), np.exp(-2j*np.pi*xlist*uv[0])) for uv in uvlist] #list of matrices at each freq
    
    # changed the sign convention to agree with BU data (Jan 2017)
    ftmatrices = [pulse(2*np.pi*uv[0], 2*np.pi*uv[1], pdim, dom="F") * np.outer(np.exp(2j*np.pi*ylist*uv[1]), np.exp(2j*np.pi*xlist*uv[0])) for uv in uvlist] #list of matrices at each freq
    
    ftmatrices = np.reshape(np.array(ftmatrices), (len(uvlist), xdim*ydim))

    if len(mask):
        ftmatrices = ftmatrices[:,mask]
        
    return ftmatrices


def make_jones(obs, ampcal=True, opacitycal=True, gainp=GAINPDEF, gain_offset=GAINPDEF, phasecal=True, 
                    dcal=True, dtermp=DTERMPDEF, dtermp_resid=DTERMPDEF_RESID, frcal=True):
    """Compute ALL Jones Matrices for a list of times (non repeating), with gain and dterm errors.
       ra and dec should be in hours / degrees
       Will return a nested dictionary of matrices indexed by the site, then by the time 

       gain_offset can be optionally set as a dictionary that specifies the constant percentage offset 
       for each telescope site. If it is a single value than it is the standard deviation 
       of a randomly selected gain offset. 
    """   

    tlist = obs.tlist()
    tarr = obs.tarr
    ra = obs.ra
    dec = obs.dec
    sourcevec = np.array([np.cos(dec*DEGREE), 0, np.sin(dec*DEGREE)])
    
    # Create a dictionary of taus and a list of unique times
    nsites = len(obs.tarr['site'])
    taudict = {site : np.array([]) for site in obs.tarr['site']}
    times = np.array([])
    for scan in tlist:
        time = scan['time'][0]
        times = np.append(times, time)
        sites_in = np.array([])
        for bl in scan:
            # Should we screen for conflicting same-time measurements of tau? 
            if len(sites_in) >= nsites: break
            
            if (not len(sites_in)) or (not bl['t1'] in sites_in):
                taudict[bl['t1']] = np.append(taudict[bl['t1']], bl['tau1'])
                sites_in = np.append(sites_in, bl['t1'])
            
            if (not len(sites_in)) or (not bl['t2'] in sites_in):
                taudict[bl['t2']] = np.append(taudict[bl['t2']], bl['tau2'])
                sites_in = np.append(sites_in, bl['t2'])                
        if len(sites_in) < nsites:
            for site in obs.tarr['site']:
                if site not in sites_in: 
                    taudict[site] = np.append(taudict[site], 0.0)   

    # Compute Sidereal Times
    if obs.timetype=='GMST':
        times_sid=times
    else:
        times_sid = utc_to_gmst(times, obs.mjd)
    
    # Generate Jones Matrices at each time for each telescope 
    out = {}
    for i in xrange(len(tarr)):
        site = tarr[i]['site']
        coords = np.array([tarr[i]['x'],tarr[i]['y'],tarr[i]['z']])
        latlon = xyz_2_latlong(coords)
            
        # Elevation Angles
        thetas = np.mod((times_sid - ra)*HOUR, 2*np.pi)
        el_angles = elev(earthrot(coords, thetas), sourcevec) 
        
        # Parallactic Angles 
        hr_angles = hr_angle(times_sid*HOUR, latlon[:,1], ra*HOUR)
        par_angles = par_angle(hr_angles, latlon[:,0], dec*DEGREE)        
        
        # Amplitude gain        
        tproc = str(ttime.time())
        gainR = gainL = np.ones(len(times))
        if not ampcal:
            # Amplitude gain
            if type(gain_offset) == dict:
                goff = gain_offset[site]
            else: 
                goff=gain_offset

            gainR = np.sqrt(np.abs(np.array([1.0 +  0.01*goff + gainp * hashrandn(site, 'gain', time, tproc) 
                                     for time in times])))
            gainL = np.sqrt(np.abs(np.array([1.0 +  0.01*goff + gainp * hashrandn(site, 'gain', time, tproc) 
                                     for time in times])))
        
        # Opacity attenuation of amplitude gain
        if not opacitycal:
            taus = np.abs(np.array([taudict[site][j] * (1.0 + gainp * hashrandn(site, 'tau', times[j], tproc)) for j in xrange(len(times))]))                
            atten = np.exp(-taus/(EP + 2.0*np.sin(el_angles)))
            
            gainR = gainR * atten
            gainL = gainL * atten
                 
        # Atmospheric Phase
        if not phasecal:
            phase = np.array([2 * np.pi * hashrand(site, 'phase', time, tproc) for time in times])
            gainR = gainR * np.exp(1j*phase) 
            gainL = gainL * np.exp(1j*phase)
            
        # D Term errors
        dR = dL = 0.0
        if not dcal: 
            dR = tarr[i]['dr']
            if dR == 0.0: dR = dtermp * (hashrandn(site, 'drreal', tproc) + 1j * hashrandn(site, 'drim', tproc))
            dL = tarr[i]['dl']
            if dL == 0.0: dL = dtermp * (hashrandn(site, 'dlreal', tproc) + 1j * hashrandn(site, 'dlim', tproc))
            dR = dR + dtermp_resid * (hashrandn(site, 'drreal_resid', tproc) + 1j * hashrandn(site, 'drim_resid', tproc))
            dL = dL + dtermp_resid * (hashrandn(site, 'dlreal_resid', tproc) + 1j * hashrandn(site, 'dlim_resid', tproc)) 

        # Feed Rotation Angles
        fr_angle = np.zeros(len(times))
        if not frcal:
            fr_angle = tarr[i]['fr_elev']*el_angles + tarr[i]['fr_par']*par_angles + tarr[i]['fr_off']*DEGREE
                     
        # Assemble the Jones Matrices and save to dictionary
        # TODO: indexed by utc or sideral time?
        j_matrices = {times[j]: np.array([
                                [np.exp(-1j*fr_angle[j])*gainR[j], np.exp(1j*fr_angle[j])*dR*gainR[j]],
                                [np.exp(-1j*fr_angle[j])*dL*gainL[j], np.exp(1j*fr_angle[j])*gainL[j]]
                                ]) 
                                for j in xrange(len(times))}

        out[site] = j_matrices
    
    return out 
    
def make_jones_inverse(obs, ampcal=True, phasecal=True, opacitycal=True, dcal=True, frcal=True):
    """Compute all Inverse Jones Matrices for a list of times (non repeating), with NO gain and dterm errors.
       ra and dec should be in hours / degrees
       Will return a dictionary of matrices for each time, indexed by the site, with the matrices in time order  

       gain_offset can be optionally set as a dictionary that specifies the constant percentage offset 
       for each telescope site. 
    """   

    # Get data
    tlist = obs.tlist()
    tarr = obs.tarr
    ra = obs.ra
    dec = obs.dec
    sourcevec = np.array([np.cos(dec*DEGREE), 0, np.sin(dec*DEGREE)])
    
    # Create a dictionary of taus and a list of unique times
    nsites = len(obs.tarr['site'])
    taudict = {site : np.array([]) for site in obs.tarr['site']}
    times = np.array([])
    for scan in tlist:
        time = scan['time'][0]
        times = np.append(times, time)
        sites_in = np.array([])
        for bl in scan:
            # Should we screen for conflicting same-time measurements of tau? 
            if len(sites_in) >= nsites: break
            
            if (not len(sites_in)) or (not bl['t1'] in sites_in):
                taudict[bl['t1']] = np.append(taudict[bl['t1']], bl['tau1'])
                sites_in = np.append(sites_in, bl['t1'])
            
            if (not len(sites_in)) or (not bl['t2'] in sites_in):
                taudict[bl['t2']] = np.append(taudict[bl['t2']], bl['tau2'])
                sites_in = np.append(sites_in, bl['t2'])                
        if len(sites_in) < nsites:
            for site in obs.tarr['site']:
                if site not in sites_in: 
                    taudict[site] = np.append(taudict[site], 0.0)   
    
    # Compute Sidereal Times
    if obs.timetype=='GMST':
        times_sid=times
    else:
        times_sid = utc_to_gmst(times, obs.mjd)
                                  
    # Make inverse Jones Matrices
    out = {}
    for i in xrange(len(tarr)):
        site = tarr[i]['site']
        coords = np.array([tarr[i]['x'],tarr[i]['y'],tarr[i]['z']])
        latlon = xyz_2_latlong(coords)
        
        # Elevation Angles
        thetas = np.mod((times_sid - ra)*HOUR, 2*np.pi)
        el_angles = elev(earthrot(coords, thetas), sourcevec)

        # Parallactic Angles (positive longitude EAST)
        hr_angles = hr_angle(times_sid*HOUR, latlon[:,1], ra*HOUR)
        par_angles = par_angle(hr_angles, latlon[:,0], dec*DEGREE)         
        
        # Amplitude gain - one by default now
        # !AC TODO this assumes all gains 1 - should we put in a fixed gain term? 
        gainR = gainL = np.ones(len(times))        
        
        # Amplitude gain        
        gainR = gainL = np.ones(len(times))
        
        #!AC TODO gain_offset not implemented in inverse Jones
        #          should be added to tarr? 
        #if not ampcal:
        #    # Amplitude gain
        #    if type(gain_offset) == dict:
        #        goff = gain_offset[site]
        #    else: 
        #        goff=gain_offset
        #
        #    gainR *= np.sqrt(1.0 +  0.01*goff)
        #    gainL *= np.sqrt(1.0 +  0.01*goff)
        
        # Opacity attenuation of amplitude gain
        if not opacitycal:
            taus = np.abs(np.array(taudict[site]))                
            atten = np.exp(-taus/(EP + 2.0*np.sin(el_angles)))
            
            gainR = gainR * atten
            gainL = gainL * atten            
        
        # D Terms 
        dR = dL = 0.0
        if not dcal: 
            dR = tarr[i]['dr']
            dL = tarr[i]['dl']
            
        # Feed Rotation Angles
        fr_angle = np.zeros(len(times))
        if not frcal:                 
            # Total Angle (Radian)
            fr_angle = tarr[i]['fr_elev']*el_angles + tarr[i]['fr_par']*par_angles + tarr[i]['fr_off']*DEGREE
                     
        # Assemble the Jones Matrices and save to dictionary
        pref = 1.0 / (gainL*gainR*(1.0 - dL*dR))
        j_matrices_inv = {times[j]: pref[j]*np.array([
                                     [np.exp(1j*fr_angle[j])*gainL[j], -np.exp(1j*fr_angle[j])*dR*gainR[j]],
                                     [-np.exp(-1j*fr_angle[j])*dL*gainL[j], np.exp(-1j*fr_angle[j])*gainR[j]]
                                     ]) for j in xrange(len(times))}
      
        out[site] = j_matrices_inv
    
    return out 
         
def add_jones_and_noise(obs, add_th_noise=True, opacitycal=True, ampcal=True, gainp=GAINPDEF, phasecal=True, dcal=True, dtermp=DTERMPDEF, frcal=True):
    """Corrupt visibilities in obs with jones matrices and add thermal noise
    """  

    print "Applying Jones Matrices to data . . . "

    # Build Jones Matrices
    jm_dict = make_jones(obs, 
                         ampcal=ampcal, opacitycal=opacitycal, gainp=gainp, phasecal=phasecal, 
                         dcal=dcal, dtermp=dtermp, frcal=frcal)    
    # Unpack Data
    obsdata = obs.data
    times = obsdata['time']                     
    t1 = obsdata['t1']
    t2 = obsdata['t2']
    tints = obsdata['tint']
    
    # Visibility Data
    rr = obsdata['vis'] + obsdata['vvis']
    ll = obsdata['vis'] - obsdata['vvis']
    rl = obsdata['qvis'] + 1j*obsdata['uvis']
    lr = obsdata['qvis'] - 1j*obsdata['uvis']   
    
    # Recompute the noise std. deviations from the SEFDs
    sig_rr = np.array([blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdr'], obs.tarr[obs.tkey[t2[i]]]['sefdr'], tints[i], obs.bw) for i in xrange(len(rr))])
    sig_ll = np.array([blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdl'], obs.tarr[obs.tkey[t2[i]]]['sefdl'], tints[i], obs.bw) for i in xrange(len(ll))])
    sig_rl = np.array([blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdr'], obs.tarr[obs.tkey[t2[i]]]['sefdl'], tints[i], obs.bw) for i in xrange(len(rl))])
    sig_lr = np.array([blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdl'], obs.tarr[obs.tkey[t2[i]]]['sefdr'], tints[i], obs.bw) for i in xrange(len(lr))])                  
    
    #print "------------------------------------------------------------------------------------------------------------------------"
    if not opacitycal: 
        print "   Applying opacity attenuation: opacitycal-->False"
    if not ampcal: 
        print "   Applying gain corruption: gaincal-->False"
    if not phasecal: 
        print "   Applying atmospheric phase corruption: phasecal-->False" 
    if not dcal: 
        print "   Applying D Term mixing: dcal-->False"
    if not frcal: 
        print "   Applying Field Rotation: frcal-->False"
    if add_th_noise: 
        print "Adding thermal noise to data . . . "
    #print "------------------------------------------------------------------------------------------------------------------------"
    
    # Corrupt each IQUV visibilty set with the jones matrices and add noise
    for i in xrange(len(times)):            
        # Form the visibility correlation matrix
        corr_matrix = np.array([[rr[i], rl[i]], [lr[i], ll[i]]])
        
        # Get the jones matrices and corrupt the corr_matrix
        j1 = jm_dict[t1[i]][times[i]]
        j2 = jm_dict[t2[i]][times[i]]
        
        corr_matrix_corrupt = np.dot(j1, np.dot(corr_matrix, np.conjugate(j2.T)))
        
        # Add noise
        if add_th_noise:
            noise_matrix = np.array([[cerror(sig_rr[i]), cerror(sig_rl[i])], [cerror(sig_lr[i]), cerror(sig_ll[i])]])              
            corr_matrix_corrupt += noise_matrix
                
        # Put the corrupted data back into the data table 
        obsdata['vis'][i]  = 0.5*(corr_matrix_corrupt[0][0] + corr_matrix_corrupt[1][1])
        obsdata['vvis'][i] = 0.5*(corr_matrix_corrupt[0][0] - corr_matrix_corrupt[1][1])
        obsdata['qvis'][i] = 0.5*(corr_matrix_corrupt[0][1] + corr_matrix_corrupt[1][0])
        obsdata['uvis'][i] = -0.5j*(corr_matrix_corrupt[0][1] - corr_matrix_corrupt[1][0])
        
        # Put the recomputed sigmas back into the data table
        obsdata['sigma'][i] = 0.5*np.sqrt(sig_rr[i]**2 + sig_ll[i]**2)        
        obsdata['vsigma'][i] = 0.5*np.sqrt(sig_rr[i]**2 + sig_ll[i]**2)
        obsdata['qsigma'][i] = 0.5*np.sqrt(sig_rl[i]**2 + sig_lr[i]**2)
        obsdata['usigma'][i] = 0.5*np.sqrt(sig_rl[i]**2 + sig_lr[i]**2)
    
    # Return observation object
    out =  Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata, obs.tarr, source=obs.source, mjd=obs.mjd, 
                   ampcal=ampcal, phasecal=phasecal, opacitycal=opacitycal, dcal=dcal, frcal=frcal)
    return out
                       
def apply_jones_inverse(obs, ampcal=True, opacitycal=True, phasecal=True, dcal=True, frcal=True):
    """Corrupt visibilities in obs with jones matrices and add thermal noise
    """  

    print "Applying a priori calibration with estimated Jones matrices . . . "

    # Build Inverse Jones Matrices
    jm_dict = make_jones_inverse(obs,
                                 ampcal=ampcal, phasecal=phasecal, opacitycal=opacitycal,
                                 dcal=dcal, frcal=frcal)   
    # Unpack Data
    obsdata = obs.data
    times = obsdata['time']                     
    t1 = obsdata['t1']
    t2 = obsdata['t2']
    tints = obsdata['tint']
    
    # Visibility Data
    rr = obsdata['vis'] + obsdata['vvis']
    ll = obsdata['vis'] - obsdata['vvis']
    rl = obsdata['qvis'] + 1j*obsdata['uvis']
    lr = obsdata['qvis'] - 1j*obsdata['uvis']   
    
    # Recompute the noise std. deviations from the SEFDs
    #!AC should we instead get them from the file? 
    sig_rr = np.array([blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdr'], obs.tarr[obs.tkey[t2[i]]]['sefdr'], tints[i], obs.bw) for i in xrange(len(rr))])
    sig_ll = np.array([blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdl'], obs.tarr[obs.tkey[t2[i]]]['sefdl'], tints[i], obs.bw) for i in xrange(len(ll))])
    sig_rl = np.array([blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdr'], obs.tarr[obs.tkey[t2[i]]]['sefdl'], tints[i], obs.bw) for i in xrange(len(rl))])
    sig_lr = np.array([blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdl'], obs.tarr[obs.tkey[t2[i]]]['sefdr'], tints[i], obs.bw) for i in xrange(len(lr))])                  
    
    ampcal = obs.ampcal
    phasecal = obs.phasecal
    #print "------------------------------------------------------------------------------------------------------------------------"
    if not opacitycal: 
        print "   Applying opacity corrections: opacitycal-->True"
        opacitycal=True
    if not dcal: 
        print "   Applying D Term corrections: dcal-->True"
        dcal=True
    if not frcal: 
        print "   Applying Field Rotation corrections: frcal-->True"
        frcal=True
    #print "------------------------------------------------------------------------------------------------------------------------"
             
    # Apply the inverse Jones matrices to each visibility
    for i in xrange(len(times)):
        # Get the inverse jones matrices
        inv_j1 = jm_dict[t1[i]][times[i]]
        inv_j2 = jm_dict[t2[i]][times[i]]
        
        # Form the visibility correlation matrix
        corr_matrix = np.array([[rr[i], rl[i]], [lr[i], ll[i]]])

        # Form the sigma matrices
        sig_rr_matrix = np.array([[sig_rr[i], 0.0], [0.0, 0.0]])
        sig_ll_matrix = np.array([[0.0, 0.0], [0.0, sig_ll[i]]])
        sig_rl_matrix = np.array([[0.0, sig_rl[i]], [0.0, 0.0]])
        sig_lr_matrix = np.array([[0.0, 0.0], [sig_lr[i], 0.0]])
                                        
        # Apply the Jones Matrices to the visibility correlation matrix and sigma matrices                                
        corr_matrix_new = np.dot(inv_j1, np.dot(corr_matrix, np.conjugate(inv_j2.T)))
        
        sig_rr_matrix_new = np.dot(inv_j1, np.dot(sig_rr_matrix, np.conjugate(inv_j2.T)))
        sig_ll_matrix_new = np.dot(inv_j1, np.dot(sig_ll_matrix, np.conjugate(inv_j2.T)))
        sig_rl_matrix_new = np.dot(inv_j1, np.dot(sig_rl_matrix, np.conjugate(inv_j2.T)))
        sig_lr_matrix_new = np.dot(inv_j1, np.dot(sig_lr_matrix, np.conjugate(inv_j2.T)))
        
        # !AC TODO is this correct?
        # Get the final sigma matrix as a quadrature sum
        sig_matrix_new = np.sqrt(np.abs(sig_rr_matrix_new)**2 + np.abs(sig_ll_matrix_new)**2 + 
                                 np.abs(sig_rl_matrix_new)**2 + np.abs(sig_lr_matrix_new)**2)
                                         
        # Put the data back into the data table
        obsdata['vis'][i]  = 0.5*(corr_matrix_new[0][0] + corr_matrix_new[1][1])
        obsdata['vvis'][i] = 0.5*(corr_matrix_new[0][0] - corr_matrix_new[1][1])
        obsdata['qvis'][i] = 0.5*(corr_matrix_new[0][1] + corr_matrix_new[1][0])
        obsdata['uvis'][i] = -0.5j*(corr_matrix_new[0][1] - corr_matrix_new[1][0])
        
        # Put the recomputed sigmas back into the data table
        obsdata['sigma'][i] = 0.5*np.sqrt(sig_matrix_new[0][0]**2 + sig_matrix_new[1][1]**2)	        
        obsdata['vsigma'][i] = 0.5*np.sqrt(sig_matrix_new[0][0]**2 + sig_matrix_new[1][1]**2)
        obsdata['qsigma'][i] = 0.5*np.sqrt(sig_matrix_new[0][1]**2 + sig_matrix_new[1][0]**2)
        obsdata['usigma'][i] = 0.5*np.sqrt(sig_matrix_new[0][1]**2 + sig_matrix_new[1][0]**2)        
	
	# Return observation object
    out =  Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata, obs.tarr, source=obs.source, mjd=obs.mjd, 
                   ampcal=ampcal, phasecal=phasecal, opacitycal=opacitycal, dcal=dcal, frcal=frcal)
    return out

# The old noise generating function.     
def add_noise(obs, ampcal=True, opacitycal=True, phasecal=True, add_th_noise=True, gainp=GAINPDEF, gain_offset=GAINPDEF):
    """Re-compute sigmas from SEFDS and add noise with gain & phase errors
       Returns signals & noises scaled by estimated gains, including opacity attenuation. 
       Be very careful using outside of Image.observe!
       
       gain_offset can be optionally set as a dictionary that specifies the constant percentage offset 
       for each telescope site. If it is a single value than it is the standard deviation 
       of a randomly selected gain offset. 
    """   

    print "Adding gain + phase errors to data and applying a priori calibration . . . "

    #print "------------------------------------------------------------------------------------------------------------------------"
    opacitycalout=True #With old noise function, output is always calibrated to estimated opacity
                       #!AC TODO Could change this 
    if not opacitycal: 
        print "   Applying opacity attenuation AND estimated opacity corrections: opacitycal-->True"
    if not ampcal: 
        print "   Applying gain corruption: gaincal-->False"
    if not phasecal:
        print "   Applying atmospheric phase corruption: phasecal-->False" 
    if add_th_noise: 
        print "Adding thermal noise to data . . . "
    #print "------------------------------------------------------------------------------------------------------------------------"
               
    # Get data
    obsdata = obs.data
    sites = obsdata[['t1','t2']].view(('a32',2))
    time = obsdata[['time']].view(('f8',1))
    tint = obsdata[['tint']].view(('f8',1))
    uv = obsdata[['u','v']].view(('f8',2))
    vis = obsdata[['vis']].view(('c16',1))
    qvis = obsdata[['qvis']].view(('c16',1))
    uvis = obsdata[['uvis']].view(('c16',1))
    vvis = obsdata[['vvis']].view(('c16',1))
    
    taus = np.abs(obsdata[['tau1','tau2']].view(('f8',2)))   
    elevs = obs.unpack(['el1','el2'],ang_unit='deg').view(('f8',2)) 
    bw = obs.bw
        
    # Recompute perfect sigmas from SEFDs
    # Multiply 1/sqrt(2) for sum of polarizations 
    sigma_perf = np.array([blnoise(obs.tarr[obs.tkey[sites[i][0]]]['sefdr'], obs.tarr[obs.tkey[sites[i][1]]]['sefdr'], tint[i], bw)/np.sqrt(2.0) 
                            for i in range(len(tint))])
                                                                                  
    # Use estimated opacity to compute the ESTIMATED noise
    sigma_est = sigma_perf
    if not opacitycal:
        sigma_est = sigma_est * np.sqrt(np.exp(taus[:,0]/(EP+np.sin(elevs[:,0]*DEGREE)) + taus[:,1]/(EP+np.sin(elevs[:,1]*DEGREE))))
        
    # Add gain and opacity fluctuations to the TRUE noise
    sigma_true = sigma_perf
    tproc = str(ttime.time())
    if not ampcal:
        # Amplitude gain
        if type(gain_offset) == dict:
            goff1 = gain_offset[sites[i,0]]
            goff2 = gain_offset[sites[i,1]]
        else: goff1=goff2=gain_offset

        gain1 = np.abs(np.array([1.0 +  0.01*goff1 + gainp * hashrandn(sites[i,0], 'gain', time[i], tproc) 
                                 for i in xrange(len(time))]))
        gain2 = np.abs(np.array([1.0 +  0.01*goff2 + gainp * hashrandn(sites[i,1], 'gain', time[i], tproc) 
                                 for i in xrange(len(time))]))  

        sigma_true = sigma_true / np.sqrt(gain1 * gain2)

    if not opacitycal:
        # Opacity Errors
        tau1 = np.abs(np.array([taus[i,0]* (1.0 + gainp * hashrandn(sites[i,0], 'tau', time[i], tproc)) for i in xrange(len(time))]))
        tau2 = np.abs(np.array([taus[i,1]* (1.0 + gainp * hashrandn(sites[i,1], 'tau', time[i], tproc)) for i in xrange(len(time))]))
        
        # Correct noise RMS for opacity
        sigma_true = sigma_true * np.sqrt(np.exp(tau1/(EP+np.sin(elevs[:,0]*DEGREE)) + tau2/(EP+np.sin(elevs[:,1]*DEGREE))))
        
    # Add the noise and the gain error to the true visibilities
    vis  = (vis + cerror(sigma_true))  * (sigma_est/sigma_true) #sigma_est/sigma_true = gain
    qvis = (qvis + cerror(sigma_true)) * (sigma_est/sigma_true)
    uvis = (uvis + cerror(sigma_true)) * (sigma_est/sigma_true)
    vvis = (vvis + cerror(sigma_true)) * (sigma_est/sigma_true)
        
    # Add random atmospheric phases    
    if not phasecal:
        phase1 = np.array([2 * np.pi * hashrand(sites[i,0], 'phase', time[i], tproc) for i in xrange(len(time))])
        phase2 = np.array([2 * np.pi * hashrand(sites[i,1], 'phase', time[i], tproc) for i in xrange(len(time))])
        
        vis *= np.exp(1j * (phase2-phase1))
        qvis *= np.exp(1j * (phase2-phase1))
        uvis *= np.exp(1j * (phase2-phase1))
        vvis *= np.exp(1j * (phase2-phase1))     
        
    # Put the visibilities estimated errors back in the obsdata array
    obsdata['vis'] = vis
    obsdata['qvis'] = qvis
    obsdata['uvis'] = uvis
    obsdata['vvis'] = vvis
    obsdata['sigma'] = sigma_est
    
    # This function doesn't use different visibility sigmas!
    obsdata['qsigma'] = obsdata['usigma'] = obsdata['vsigma'] = sigma_est
    
	# Return observation object
    out =  Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata, obs.tarr, source=obs.source, mjd=obs.mjd, ampcal=ampcal, phasecal=phasecal, opacitycal=opacitycalout)
    return out


##################################################################################################
# Plot Related Functions
##################################################################################################        
def ticks(axisdim, psize, nticks=8):
    """Return a list of ticklocs and ticklabels
       psize should be in desired units
    """
    
    axisdim = int(axisdim)
    nticks = int(nticks)
    if not axisdim % 2: axisdim += 1
    if nticks % 2: nticks -= 1
    tickspacing = float((axisdim-1))/nticks
    ticklocs = np.arange(0, axisdim+1, tickspacing)
    ticklabels= np.around(psize * np.arange((axisdim-1)/2., -(axisdim)/2., -tickspacing), decimals=1)
    return (ticklocs, ticklabels)
                                     
##################################################################################################
# Other Functions
##################################################################################################

#!AC TODO
# Lindy's function to generate covariance matrices
#def make_cov(clnoisedata, noisebldict):
#    iloc = {cl:i for i, cl in enumerate(cldata)}
#    cov = np.zeros((len(cldata), len(cldata)))
#    bldict = {bl:[] for bl in bldict.iterkeys()}
#    for cl in cldata:
#        if len(cl) == 3: # closure phase parity
#            for bl in cl:
#                parity = 1 if bl[1] > bl[0] else -1
#                bldict[bl[::parity]].append((cl, parity)) # convert key to sorted order
#        elif len(cl) == 4: # closure amplitude parity
#            for bl in cl[:2]:
#                bldict[bl].append((cl, 1)) # numerator
#            for bl in cl[2:]:
#                bldict[bl].append((cl, -1)) # denominator
#    for (bl, cqlist) in bldict.iteritems():
#        for (cq1, cq2) in product(cqlist, cqlist):
#            cov[iloc[cq1[0]], iloc[cq2[0]]] += cq1[1] * cq2[1] * noisebldict[bl]['sigmaca']**2 ##ANDREW FIX THIS FOR PHASES
#    return cov

def power_of_two(target):
    """Finds the next greatest power of two
    """
    cur = 1
    if target > 1:
        for i in xrange(0, int(target)):
            if (cur >= target):
                return cur
            else: cur *= 2
    else:
        return 1


def paritycompare(perm1, perm2):
    """Compare the parity of two permutations.
       Assume both lists are equal length and with same elements
       Copied from: http://stackoverflow.com/questions/1503072/how-to-check-if-permutations-have-equal-parity
    """
    
    perm2 = list(perm2)
    perm2_map = dict((v, i) for i,v in enumerate(perm2))
    transCount=0
    for loc, p1 in enumerate(perm1):
        p2 = perm2[loc]
        if p1 != p2:
            sloc = perm2_map[p1]
            perm2[loc], perm2[sloc] = p1, p2
            perm2_map[p1], perm2_map[p2] = sloc, loc
            transCount += 1
    
    if not (transCount % 2): return 1
    else: return  -1

def amp_debias(vis, sigma):
    """Return debiased visibility amplitudes
    """
    
    # !AC TODO: what to do if deb2 < 0? Currently we do nothing
    deb2 = np.abs(vis)**2 - np.abs(sigma)**2
    if type(deb2) == float or type(deb2)==np.float64:
        if deb2 < 0.0: return np.abs(vis)
        else: return np.sqrt(deb2)
    else:
        lowsnr = deb2 < 0.0
        deb2[lowsnr] = np.abs(vis[lowsnr])**2
        return np.sqrt(deb2)
        
def sigtype(datatype):
    """Return the type of noise corresponding to the data type
    """
    
    datatype = str(datatype)
    if datatype in ['vis', 'amp']: sigmatype='sigma'
    elif datatype in ['qvis', 'qamp']: sigmatype='qsigma'
    elif datatype in ['uvis', 'uamp']: sigmatype='usigma'
    elif datatype in ['vvis', 'vamp']: sigmatype='vsigma'
    elif datatype in ['pvis', 'pamp']: sigmatype='psigma'                
    elif datatype in ['pvis', 'pamp']: sigmatype='psigma'
    elif datatype in ['m', 'mamp']: sigmatype='msigma'
    elif datatype in ['phase']: sigmatype='sigma_phase'
    elif datatype in ['qphase']: sigmatype='qsigma_phase'
    elif datatype in ['uphase']: sigmatype='usigma_phase'
    elif datatype in ['vphase']: sigmatype='vsigma_phase'
    elif datatype in ['pphase']: sigmatype='psigma_phase'
    elif datatype in ['mphase']: sigmatype='msigma_phase'
    else: sigmatype = False
    
    return sigmatype                                    
    
def merr(sigma, qsigma, usigma, I, m):
    """Return the error in mbreve real and imaginary parts"""

    err = np.sqrt((qsigma**2 + usigma**2 + (sigma*np.abs(m))**2)/ (np.abs(I) ** 2))
    # old formula assumes all sigmas the same
    #err = sigma * np.sqrt((2 + np.abs(m)**2)/ (np.abs(I) ** 2))     
    return err
           
def rastring(ra):
    """Convert a ra in fractional hours to formatted string
    """
    h = int(ra)
    m = int((ra-h)*60.)
    s = (ra-h-m/60.)*3600.
    out = "%2i h %2i m %2.4f s" % (h,m,s)
    return out 

def decstring(dec):
    """Convert a dec in fractional degrees to formatted string
    """
    
    deg = int(dec)
    m = int((abs(dec)-abs(deg))*60.)
    s = (abs(dec)-abs(deg)-m/60.)*3600.
    out = "%2i deg %2i m %2.4f s" % (deg,m,s)
    return out

def gmtstring(gmt):
    """Convert a gmt in fractional hours to formatted string
    """
    
    if gmt > 24.0: gmt = gmt-24.0
    h = int(gmt)
    m = int((gmt-h)*60.)
    s = (gmt-h-m/60.)*3600.
    out = "%02i:%02i:%2.4f" % (h,m,s)
    return out 

#def fracmjd(mjd, gmt):
#    """Convert a int mjd + gmt (frac. hr.) into a fractional mjd"""
#    
#    return int(mjd) + gmt/24.
#
#def mjdtogmt(mjd):
#    """Return the gmt of a fractional mjd, in days"""
#  
#    return (mjd - int(mjd)) * 24.0
#    
#def jdtomjd(jd):
#    """Return the mjd of a jd"""
#  
#    return jd - 2400000.5

def utc_to_gmst(utc, mjd): 
    """Convert utc times in hours to gmst using astropy
    """

    mjd=int(mjd) #MJD should always be an integer, but was float in older versions of the code
    time_obj = at.Time(utc/24.0 + np.floor(mjd), format='mjd', scale='utc') 
    time_sidereal = time_obj.sidereal_time('mean','greenwich').hour
    return time_sidereal
    
def earthrot(vecs, thetas):
    """Rotate a vector / array of vectors about the z-direction by theta / array of thetas (radian)
    """

    if len(vecs.shape)==1: 
        vecs = np.array([vecs])
    if np.isscalar(thetas):
        thetas = np.array([thetas for i in xrange(len(vecs))])

    # equal numbers of sites and angles
    if len(thetas) == len(vecs):
        rotvec = np.array([np.dot(np.array(((np.cos(thetas[i]),-np.sin(thetas[i]),0),(np.sin(thetas[i]),np.cos(thetas[i]),0),(0,0,1))), vecs[i]) 
                       for i in xrange(len(vecs))])
    # only one rotation angle, many sites
    elif len(thetas) == 1:
        rotvec = np.array([np.dot(np.array(((np.cos(thetas[0]),-np.sin(thetas[0]),0),(np.sin(thetas[0]),np.cos(thetas[0]),0),(0,0,1))), vecs[i]) 
                       for i in xrange(len(vecs))])
    # only one site, many angles
    elif len(vecs) == 1:
        rotvec = np.array([np.dot(np.array(((np.cos(thetas[i]),-np.sin(thetas[i]),0),(np.sin(thetas[i]),np.cos(thetas[i]),0),(0,0,1))), vecs[0]) 
                       for i in xrange(len(thetas))]) 
    else:
        raise Exception("Unequal numbers of vectors and angles in earthrot(vecs, thetas)!")
                                            
    #if rotvec.shape[0]==1: rotvec = rotvec[0]
    return rotvec

def elev(obsvecs, sourcevec):
    """Return the elevation of a source with respect to an observer/observers in radians
       obsvec can be an array of vectors but sourcevec can ONLY be a single vector
    """
       
    if len(obsvecs.shape)==1:
        obsvecs=np.array([obsvecs])

    anglebtw = np.array([np.dot(obsvec,sourcevec)/np.linalg.norm(obsvec)/np.linalg.norm(sourcevec) for obsvec in obsvecs])
    el = 0.5*np.pi - np.arccos(anglebtw)

    return el
        
def elevcut(obsvecs, sourcevec, elevmin=ELEV_LOW, elevmax=ELEV_HIGH):
    """Return True if a source is observable by a telescope vector
    """
    
    angles = elev(obsvecs, sourcevec)/DEGREE
    
    return (angles > elevmin) * (angles < elevmax)

def hr_angle(gst, lon, ra):
    """Computes the hour angle for a source at RA, observer at longitude long, and GMST time gst
       gst in hours, ra & lon ALL in radian
       longitude positive east
    """

    hr_angle = np.mod(gst + lon - ra, 2*np.pi)
    return hr_angle
    
def par_angle(hr_angle, lat, dec):
    """Compute the parallactic angle for a source at hr_angle and dec for an observer with latitude lat. 
       All angles in radian
    """
       
    num = np.sin(hr_angle)*np.cos(lat)
    denom = np.sin(lat)*np.cos(dec) - np.cos(lat)*np.sin(dec)*np.cos(hr_angle)
    
    return np.arctan2(num, denom)

def xyz_2_latlong(obsvecs): 
    """Compute the (geocentric) latitude and longitude of a site at geocentric position x,y,z 
       The output is in radians
    """

    if len(obsvecs.shape)==1:
        obsvecs=np.array([obsvecs])        
    out = []
    for obsvec in obsvecs:
        x = obsvec[0]
        y = obsvec[1]
        z = obsvec[2]
        lon = np.array(np.arctan2(y,x))
        lat = np.array(np.arctan2(z, np.sqrt(x**2+y**2)))
        out.append([lat,lon])
        
    out = np.array(out)

    #if out.shape[0]==1: out = out[0]
    return out




##################################################################################################
# Convenience Functions for Data Processing
##################################################################################################

def split_obs(obs):
    """Split single observation into multiple observation files, one per scan
    """

    print "Splitting Observation File into " + str(len(obs.tlist())) + " scans"

    #Note that the tarr of the output includes all sites, even those that don't participate in the scan
    splitlist = [Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, tdata, obs.tarr, source=obs.source,
                         mjd=obs.mjd, ampcal=obs.ampcal, phasecal=obs.phasecal) 
                 for tdata in obs.tlist() 
                ]
    return splitlist

def merge_obs(obs_List):
    """Merge a list of observations into a single observation file
    """

    if (len(set([obs.ra for obs in obs_List])) > 1 or 
        len(set([obs.dec for obs in obs_List])) > 1 or 
        len(set([obs.rf for obs in obs_List])) > 1 or 
        len(set([obs.bw for obs in obs_List])) > 1 or 
        len(set([obs.source for obs in obs_List])) > 1 or 
        len(set([np.floor(obs.mjd) for obs in obs_List])) > 1):

        print "All observations must have the same parameters!"
        return 

    #The important things to merge are the mjd, the data, and the list of telescopes
    data_merge = np.hstack([obs.data for obs in obs_List]) 

    mergeobs = Obsdata(obs_List[0].ra, obs_List[0].dec, obs_List[0].rf, obs_List[0].bw, data_merge, np.unique(np.concatenate([obs.tarr for obs in obs_List])), 
                       source=obs_List[0].source, mjd=obs_List[0].mjd, ampcal=obs_List[0].ampcal, phasecal=obs_List[0].phasecal) 

    return mergeobs

def make_subarray(array, sites):
    """Make a subarray from the Array object array that only includes the sites listed in sites
    """
    all_sites = [t[0] for t in array.tarr]   
    mask = np.array([t in sites for t in all_sites])
    return Array(array.tarr[mask])

