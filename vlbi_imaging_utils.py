# vlbi_imaging_utils.py
# Andrew Chael, 02/24/16
# Utilities for generating and manipulating VLBI images, datasets, and arrays
# 03/24 Added new prescription for computing closures
# 02/24 Added dirty beam, dirty image, and fitting for the clean beam
# 01/20 Added gain and phase errors

# TODO: 
#       Fix for case where there are no data points
#       Screen for 0 errors 
#       Add amplitude debiasing
#       Add closure amplitude debiasing
#       Add different i,q,u,v SEFDs and calibration errors?

import string
import numpy as np
import numpy.lib.recfunctions as rec
import matplotlib.pyplot as plt
import scipy.signal
import scipy.optimize
import itertools as it
import astropy.io.fits as fits
#from mpl_toolkits.basemap import Basemap # for plotting baselines on globe

##################################################################################################
# Constants
##################################################################################################
C = 299792458.0
DEGREE = np.pi/180.
RADPERAS = DEGREE/3600
RADPERUAS = RADPERAS/1e6

# Telescope elevation cuts (degrees) 
ELEV_LOW = 15.0
ELEV_HIGH = 85.0

# Default Optical Depth and std. dev % on gain
TAUDEF = 0.1
GAINPDEF = 0.1

# Sgr A* Kernel Values (Bower et al., in uas/cm^2)
FWHM_MAJ = 1.309 * 1000 # in uas
FWHM_MIN = 0.64 * 1000
POS_ANG = 78 # in degree, E of N

# Observation recarray datatypes
DTARR = [('site', 'a32'), ('x','f8'), ('y','f8'), ('z','f8'), ('sefd','f8')]

DTPOL = [('time','f8'),('tint','f8'),
         ('t1','a32'),('t2','a32'),
         ('el1','f8'),('el2','f8'),
         ('tau1','f8'),('tau2','f8'),
         ('u','f8'),('v','f8'),
         ('vis','c16'),('qvis','c16'),('uvis','c16'),('sigma','f8')]

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
          't1','t2','el1','el2','tau1','tau2',
          'vis','amp','phase','snr','sigma',
          'qvis','qamp','qphase','qsnr',
          'uvis','uamp','uphase','usnr',
          'pvis','pamp','pphase',
          'm','mamp','mphase']
                  
##################################################################################################
# Classes
##################################################################################################

class Image(object):
    """A radio frequency image array (in Jy/pixel).
    
    Attributes:
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
    
    def __init__(self, image, psize, ra, dec, rf=230e9, source="SgrA", mjd="48277"):
        if len(image.shape) != 2: 
            raise Exception("image must be a 2D numpy array") 
               
        self.psize = float(psize)
        self.xdim = image.shape[1]
        self.ydim = image.shape[0]
        self.imvec = image.flatten() 
                
        self.ra = float(ra) 
        self.dec = float(dec)
        self.rf = float(rf)
        self.source = str(source)
        self.mjd = float(mjd)
        
        self.qvec = []
        self.uvec = []
        
    def add_qu(self, qimage, uimage):
        """Add Q and U images
        """
        
        if len(qimage.shape) != len(uimage.shape):
            raise Exception("image must be a 2D numpy array")
        if qimage.shape != uimage.shape != (self.ydim, self.xdim):
            raise Exception("Q & U image shapes incompatible with I image!") 
        self.qvec = qimage.flatten()
        self.uvec = uimage.flatten()
    
    def copy(self):
        """Copy the image object"""
        newim = Image(self.imvec.reshape(self.ydim,self.xdim), self.psize, self.ra, self.dec, self.rf, self.source, self.mjd)
        newim.add_qu(self.qvec.reshape(self.ydim,self.xdim), self.uvec.reshape(self.ydim,self.xdim))
        return newim
        
    def flip_chi(self):
        """Change between different conventions for measuring position angle (East of North vs up from x axis)
        """
        self.qvec = - self.qvec
        return
           
    def observe_same(self, obs, gainp=GAINPDEF, ampcal="True", phasecal="True", sgrscat=False):
        """Observe the image on the same baselines with the same noise as an existing observation object
           if sgrscat==True, the visibilites will be blurred by the Sgr A* scattering kernel
           gainp is the percent error in gain and tau
        """
        
        # Check for agreement in coordinates and frequency 
        if (self.ra!= obs.ra) or (self.dec != obs.dec):
            raise Exception("Image coordinates are not the same as observtion coordinates!")
	    if (self.rf != obs.rf):
	        raise Exception("Image frequency is not the same as observation frequency!")
        
        # Get data
        obsdata = obs.data
        obslist = obs.tlist()
        
        # Remove possible conjugate baselines:
        obsdata = []
        blpairs = []
        for tlist in obslist:
            for dat in tlist:
                if not ((dat['t1'], dat['t2']) in blpairs 
                     or (dat['t2'], dat['t1']) in blpairs):
                     obsdata.append(dat)
                     
        obsdata = np.array(obsdata, dtype=DTPOL)
                          
        # Extract data
        sites = obsdata[['t1','t2']].view(('a32',2))
        elevs = obsdata[['el1','el2']].view(('f8',2))
        taus = obsdata[['tau1','tau2']].view(('f8',2))
        time = obsdata['time'].view(('f8',1))
        uv = obsdata[['u','v']].view(('f8',2))
        sigma_clean = obsdata['sigma'].view(('f8',1))

        # Perform DFT
        mat = ftmatrix(self.psize, self.xdim, self.ydim, uv)
        vis = np.dot(mat, self.imvec)
        
        # Estimated noise using no gain and estimated opacity
        sigma_est = sigma_clean * np.sqrt(np.exp(taus[:,0]/np.sin(elevs[:,0]*DEGREE) + taus[:,1]/np.sin(elevs[:,1]*DEGREE)))
        
        # If there are polarized images, observe them:
        qvis = np.zeros(len(vis))
        uvis = np.zeros(len(vis))
        if len(self.qvec):
            qvis = np.dot(mat, self.qvec)
            uvis = np.dot(mat, self.uvec)
        
        # Scatter the visibilities with the SgrA* kernel
        if sgrscat:
            for i in range(len(vis)):
                ker = sgra_kernel_uv(self.rf, uv[i,0], uv[i,1])
                vis[i]  *= ker
                qvis[i] *= ker
                uvis[i] *= ker
        
        # Add gain and opacity uncertanities to the RMS noise
        if not ampcal:
            # Amplitude gain
            gain1 = np.abs(np.array([1.0 + gainp * hashrandn(sites[i,0], 'gain') 
                            + gainp * hashrandn(sites[i,0], 'gain', time[i]) for i in xrange(len(time))]))
            gain2 = np.abs(np.array([1.0 + gainp * hashrandn(sites[i,1], 'gain') 
                            + gainp * hashrandn(sites[i,1], 'gain', time[i]) for i in xrange(len(time))]))
           
            # Opacity
            tau1 = np.array([taus[i,0]*(1 + gainp * hashrandn(sites[i,0], 'tau', time[i])) for i in xrange(len(time))])
            tau2 = np.array([taus[i,1]*(1 + gainp * hashrandn(sites[i,1], 'tau', time[i])) for i in xrange(len(time))])

            # Correct noise RMS for gain variation and opacity
            sigma_true = sigma_clean / np.sqrt(gain1 * gain2)
            sigma_true = sigma_true * np.sqrt(np.exp(tau1/np.sin(elevs[:,0]*DEGREE) + tau2/np.sin(elevs[:,1]*DEGREE)))
        
        else: 
            sigma_true = sigma_est
        
        # Add the noise the gain error
        vis  = (vis + cerror(sigma_true))  * (sigma_est/sigma_true)
        qvis = (qvis + cerror(sigma_true)) * (sigma_est/sigma_true)
        uvis = (uvis + cerror(sigma_true)) * (sigma_est/sigma_true)

        # Add random atmospheric phases    
        if not phasecal:
            phase1 = np.array([2 * np.pi * hashrand(sites[i,0], 'phase', time[i]) for i in xrange(len(time))])
            phase2 = np.array([2 * np.pi * hashrand(sites[i,1], 'phase', time[i]) for i in xrange(len(time))])
            
            vis *= np.exp(1j * (phase2-phase1))
            qvis *= np.exp(1j * (phase2-phase1))
            uvis *= np.exp(1j * (phase2-phase1))
              
        # Put the visibilities estimated errors back in the obsdata array
        obsdata['vis'] = vis
        obsdata['qvis'] = qvis
        obsdata['uvis'] = uvis
        obsdata['sigma'] = sigma_est
        
        # Return observation object
        return Obsdata(self.ra, self.dec, self.rf, obs.bw, obsdata, obs.tarr, source=self.source, mjd=self.mjd, ampcal=ampcal, phasecal=phasecal)
    
    def observe(self, array, tint, tadv, tstart, tstop, bw, tau=TAUDEF, gainp=GAINPDEF, ampcal="True", phasecal="True", sgrscat=False):
        """Observe the image with an array object to produce an obsdata object.
	       tstart and tstop should be hrs in GMST.
           tint and tadv should be seconds.
           tau is the estimated optical depth. This can be a single number or a dictionary giving one tau per site
           if sgrscat==True, the visibilites will be blurred by the Sgr A* scattering kernel at the appropriate frequency
	    """
        
        # Generate empty observation
        obs = array.obsdata(self.ra, self.dec, self.rf, bw, tint, tadv, tstart, tstop, tau=tau)
        
        # Observe
        obs = self.observe_same(obs, ampcal=ampcal, phasecal=phasecal, sgrscat=sgrscat, gainp=gainp)    
        return obs
        
    def display(self, cfun='afmhot', nvec=20, pcut=0.01, plotp=False):
        """Display the image with matplotlib
        """
        
        plt.figure()
        plt.clf()
        
        image = self.imvec;
        
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
            im = plt.imshow(image.reshape(self.ydim, self.xdim), cmap=plt.get_cmap(cfun), interpolation='nearest')
            plt.colorbar(im, fraction=0.046, pad=0.04, label='Jy/pixel')
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
            im = plt.imshow(m, cmap=plt.get_cmap('winter'), interpolation='nearest', vmin=0, vmax=1)
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
            
            im = plt.imshow(image.reshape(self.ydim,self.xdim), cmap=plt.get_cmap(cfun), interpolation='nearest')
            plt.colorbar(im, fraction=0.046, pad=0.04, label='Jy/pixel')
            xticks = ticks(self.xdim, self.psize/RADPERAS/1e-6)
            yticks = ticks(self.ydim, self.psize/RADPERAS/1e-6)
            plt.xticks(xticks[0], xticks[1])
            plt.yticks(yticks[0], yticks[1])
            plt.xlabel('Relative RA ($\mu$as)')
            plt.ylabel('Relative Dec ($\mu$as)')   
        
        plt.show(block=False)
            
    def save_txt(self, fname):
        """Save image data to text file"""
        
        # Coordinate values
        pdimas = self.psize/RADPERAS
        xs = np.array([[j for j in range(self.xdim)] for i in range(self.ydim)]).reshape(self.xdim*self.ydim,1)
        xs = pdimas * (xs[::-1] - self.xdim/2.0)
        ys = np.array([[i for j in range(self.xdim)] for i in range(self.ydim)]).reshape(self.xdim*self.ydim,1)
        ys = pdimas * (ys[::-1] - self.xdim/2.0)
        
        # Data
        if len(self.qvec):
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
                    "MJD: %.4f \n" % self.mjd + 
                    "RF: %.4f GHz \n" % (self.rf/1e9) + 
                    "FOVX: %i pix %f as \n" % (self.xdim, pdimas * self.xdim) +
                    "FOVY: %i pix %f as \n" % (self.ydim, pdimas * self.ydim) +
                    "------------------------------------\n" + hf)
         
        # Save
        np.savetxt(fname, outdata, header=head, fmt=fmts)

    def save_fits(self, fname):
        """Save image data to FITS file"""
                
        # Create header and fill in some values
        header = fits.Header()
        header['OBJECT'] = self.source
        header['CTYPE1'] = 'RA---SIN'
        header['CTYPE2'] = 'DEC--SIN'
        header['CDELT1'] = -self.psize
        header['CDELT2'] = self.psize
        header['OBSRA'] = self.ra * 180/12.
        header['OBSDEC'] = self.dec
        header['FREQ'] = self.rf
        header['MJD'] = self.mjd
        header['TELESCOP'] = 'VLBI'
        header['BUNIT'] = 'JY/PIXEL'
        header['STOKES'] = 'I'
        
        # Create the fits image
        image = np.reshape(self.imvec,(self.ydim,self.xdim))[::-1,:] #flip y axis!
        hdu = fits.PrimaryHDU(image, header=header)
        if len(self.qvec):
            qimage = np.reshape(self.qvec,(self.xdim,self.ydim))[::-1,:]
            uimage = np.reshape(self.uvec,(self.xdim,self.ydim))[::-1,:]
            header['STOKES'] = 'Q'
            hduq = fits.ImageHDU(qimage, name='Q', header=header)
            header['STOKES'] = 'U'
            hduu = fits.ImageHDU(uimage, name='U', header=header)
            hdulist = fits.HDUList([hdu, hduq, hduu])
        else: hdulist = fits.HDUList([hdu])
      
        # Save fits 
        hdulist.writeto(fname, clobber=True)
        
        return
                
##################################################################################################        
class Array(object):
    """A VLBI array of telescopes with locations and SEFDs
    
        Attributes:
        tarr: The array of telescope data (name, x, y, z, SEFD) where x,y,z are geocentric coordinates.
    """   
    
    def __init__(self, tarr):
        self.tarr = tarr
        
        # Dictionary of array indices for site names
        # !AC better way?
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}
            
    def listbls(self):
        """List all baselines"""
 
        bls = []
        for i1 in sorted(self.tarr['site']):
            for i2 in sorted(self.tarr['site']):
                if not ([i1,i2] in bls) and not ([i2,i1] in bls) and i1 != i2:
                    bls.append([i1,i2])
                    
        return np.array(bls)
            
    def obsdata(self, ra, dec, rf, bw, tint, tadv, tstart, tstop, tau=TAUDEF):
        """Generate u,v points and baseline errors for the array.
           Return an Observation object with no visibilities.
           tstart and tstop are hrs in GMST
           tint and tadv are seconds.
           rf and bw are Hz
           ra is fractional hours
           dec is fractional degrees
           tau can be a single number or a dictionary giving one per site
        """
        
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
        times = np.arange(tstart, tstop+tstep, tstep)
       
        # Generate uv points at all times

        outlist = []        
        for k in xrange(len(times)):
            time = times[k]
            theta = np.mod((time-ra)*360./24, 360)
            blpairs = []
            for i1 in xrange(len(self.tarr)):
                for i2 in xrange(len(self.tarr)):
                    coord1 = np.array((self.tarr[i1]['x'], self.tarr[i1]['y'], self.tarr[i1]['z']))
                    coord2 = np.array((self.tarr[i2]['x'], self.tarr[i2]['y'], self.tarr[i2]['z']))
                    if (i1!=i2 and 
                        self.tarr[i1]['z'] <= self.tarr[i2]['z'] and # Choose the north one first
                        not ((i2, i1) in blpairs) and # This cuts out the conjugate baselines
                        elevcut(earthrot(coord1, theta),sourcevec) and 
                        elevcut(earthrot(coord2, theta),sourcevec)):
                        
                        # Optical Depth
                        if type(tau) == dict:
                            try: 
                                tau1 = tau[i1]
                                tau2 = tau[i2]
                            except KeyError:
                                tau1 = tau2 = TAUDEF
                        else:
                            tau1 = tau2 = tau
                        
                        # Append data to list   
                        blpairs.append((i1,i2))
                        outlist.append(np.array((
                                  time,
                                  tint, # Integration 
                                  self.tarr[i1]['site'], # Station 1
                                  self.tarr[i2]['site'], # Station 2
                                  elev(earthrot(coord1, theta),sourcevec), # Station 1 elevation
                                  elev(earthrot(coord2, theta),sourcevec), # Station 2 elevation
                                  tau1, # Station 1 optical depth
                                  tau2, # Station 1 optical depth
                                  np.dot(earthrot(coord1 - coord2, theta)/l, projU), # u (lambda)
                                  np.dot(earthrot(coord1 - coord2, theta)/l, projV), # v (lambda)
                                  0.0, 0.0, 0.0, # Stokes I, Q, U visibilities (Jy)
                                  blnoise(self.tarr[i1]['sefd'], self.tarr[i2]['sefd'], tint, bw) # Sigma (Jy)
                                ), dtype=DTPOL
                                ))

        obs = Obsdata(ra, dec, rf, bw, np.array(outlist), self.tarr, source="0", mjd=0, ampcal=True, phasecal=True)      
        return obs
     
    def save_array(self, fname):
        """Save the array data in a text file"""
         
        out = ""
        for scope in range(len(self.tarr)):
            dat = (self.tarr[scope]['site'], 
                   self.tarr[scope]['x'], self.tarr[scope]['y'], 
                   self.tarr[scope]['z'], self.tarr[scope]['sefd']
                  )
            out += "%-8s %15.5f  %15.5f  %15.5f  %6.4f \n" % dat
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
        tstart: the observation start time (GMT, frac. hr.)
        tstop: the observation end time (GMT, frac. hr.)
        rf: the observing frequency (Hz)
        bw: the observing bandwidth (Hz)
        ampcal: amplitudes calibrated T/F
        phasecal: phases calibrated T/F
        data: recarray with the data (time, t1, t2, tint, u, v, vis, qvis, uvis, sigma)
    """
    
    def __init__(self, ra, dec, rf, bw, datatable, tarr, source="SgrA", mjd=48277, ampcal=True, phasecal=True):
        
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
        self.tarr = tarr
        
        # Dictionary of array indices for site names
        # !AC better way?
        self.tkey = {self.tarr[i]['site']:i for i in range(len(self.tarr))}
        
        # Time partition the datatable
        datalist = []
        for key, group in it.groupby(datatable, lambda x: x['time']):
            datalist.append(np.array([obs for obs in group]))
        
        # Remove conjugate baselines
        # Make the north site first in each pair
        obsdata = []
        for tlist in datalist:
            blpairs = []
            for dat in tlist:
                if not (set((dat['t1'], dat['t2']))) in blpairs:
                     # Reverse the baseline if not north
                     if self.tarr[self.tkey[dat['t1']]]['z'] <= self.tarr[self.tkey[dat['t1']]]['z']:
                        (dat['t1'], dat['t2']) = (dat['t2'], dat['t1'])
                        (dat['el1'], dat['el2']) = (dat['el2'], dat['el1'])
                        dat['u'] = -dat['u']
                        dat['v'] = -dat['v']
                        dat['vis'] = np.conj(dat['vis'])
                        dat['uvis'] = np.conj(dat['uvis'])
                        dat['qvis'] = np.conj(dat['qvis'])
                     # Append the data point
                     blpairs.append(set((dat['t1'],dat['t2'])))    
                     obsdata.append(dat) 
        
        obsdata = np.array(obsdata, dtype=DTPOL)
        
        # Sort the data by time
        obsdata = obsdata[np.argsort(obsdata['time'])]

        # Save the data             
        self.data = obsdata
            
        # Get tstart, mjd and tstop
        times = self.unpack(['time'])['time']
        self.tstart = times[0]
        self.mjd = fracmjd(mjd, self.tstart)
        self.tstop = times[-1]
        if self.tstop < self.tstart: 
            self.tstop += 24.0
    
    def data_conj(self):
        #!AC
        """Return a data array of same format as self.data but including all conjugate baselines"""
        
        data = np.empty(2*len(self.data),dtype=DTPOL)        
        
        # Add the conjugate baseline data
        for f in DTPOL:
            f = f[0]
            if f in ["t1", "t2", "el1", "el2", "tau1", "tau2"]:
                if f[-1]=='1': f2 = f[:-1]+'2'
                else: f2 = f[:-1]+'1'
                data[f] = np.hstack((self.data[f], self.data[f2]))
            elif f in ["u","v"]:
                data[f] = np.hstack((self.data[f], -self.data[f]))
            elif f in ["vis","qvis","uvis"]:
                data[f] = np.hstack((self.data[f], np.conj(self.data[f])))
            else:
                data[f] = np.hstack((self.data[f], self.data[f]))
        
        # Sort the data
        #!AC sort within equal times? 
        data = data[np.argsort(data['time'])]
        return data
        
    def unpack(self, fields, conj=False):
        """Return a recarray of all the data for the given fields from the data table
           If conj=True, will return conjugate baselines"""
        
        # If we only specify one field
        if type(fields) == str: fields = [fields]
            
        # Get conjugates
        if conj:
            data = self.data_conj()     
        else:
            data = self.data
        
        # Get field data    
        allout = []    
        for field in fields:
             
            if field in ["u","v","sigma","tint","time","el1","el2","tau1","tau2"]: 
                out = data[field]
                ty = 'f8'
            elif field in ["t1","t2"]: 
                out = data[field]
                ty = 'a32'
            elif field in ["vis","amp","phase","snr"]: 
                out = data['vis']
                ty = 'c16'
            elif field in ["qvis","qamp","qphase","qsnr"]: 
                out = data['qvis']
                ty = 'c16'
            elif field in ["uvis","uamp","uphase","usnr"]: 
                out = data['uvis']
                ty = 'c16'
            elif field in ["pvis","pamp","pphase"]: 
                out = data['qvis'] + 1j * data['uvis']
                ty = 'c16'
            elif field in ["m","mamp","mphase"]: 
                out = (data['qvis'] + 1j * data['uvis']) / data['vis']
                ty = 'c16'
            elif field in ["uvdist"]: 
                out = np.abs(data['u'] + 1j * data['v'])
                ty = 'f8'
            else: raise Exception("%s is not valid field \n" % field + 
                                  "valid field values are " + string.join(FIELDS)) 

            # Get arg/amps/snr
            if field in ["amp", "qamp", "uamp", "pamp", "mamp"]: 
                out = np.abs(out)
                ty = 'f8'
            elif field in ["phase", "qphase", "uphase", "pphase", "mphase"]: 
                out = np.angle(out)/DEGREE
                ty = 'f8'
            elif field in ["snr","qsnr","usnr"]:
                out = np.abs(out)/data['sigma']
                ty = 'f8'
             
                    
            # Reshape and stack with other fields
            out = np.array(out, dtype=[(field, ty)])
            if len(allout) > 0:
                allout = rec.merge_arrays((allout, out), asrecarray=True, flatten=True)
            else:
                allout = out
            
        return allout
    
    def tlist(self, conj=False):
        """Return partitioned data in a list of equal time observations"""
        
        if conj: 
            data = self.data_conj()
        else: 
            data = self.data
        
        # Use itertools groupby function to partition the data
        datalist = []
        for key, group in it.groupby(data, lambda x: x['time']):
            datalist.append(np.array([obs for obs in group]))
        
        return datalist
    
    def res(self):
        """Return the nominal resolution of the observation in radian"""
        return 1.0/np.max(self.unpack('uvdist')['uvdist'])
        
    def bispectra(self, vtype='vis', mode='time', count='min'):
        """Return all independent equal time bispectrum values
           Independent triangles are chosen to contain the minimum sefd station in the scan
           Set count='max' to return all bispectrum values
           Get Q, U, P bispectra by changing vtype
        """
        #!AC Error formula for bispectrum only true in high SNR limit!
        if not mode in ('time', 'all'):
            raise Exception("possible options for mode are 'time' and 'all'")
        if not count in ('min', 'max'):
            raise Exception("possible options for count are 'min' and 'max'")
        
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
            
            
            if count == 'min':
                # If we want a minimal set, choose triangles with the minimum sefd reference
                # Unless there is no sefd data, in which case choose the northernmost
                if len(set(self.tarr['sefd'])) > 1:
                    ref = sites[np.argmin([self.tarr[self.tkey[site]]['sefd'] for site in sites])]
                else:
                    ref = sites[np.argmax([self.tarr[self.tkey[site]]['z'] for site in sites])]
                sites.remove(ref)
                
                # Find all triangles that contain the ref                    
                tris = list(it.combinations(sites,2))
                tris = [(ref, t[0], t[1]) for t in tris]
            elif count == 'max':
                # Find all triangles
                tris = list(it.combinations(sites,3))
            
            for tri in tris:
                # The ordering is north-south
                a1 = np.argmax([self.tarr[self.tkey[site]]['z'] for site in tri])
                a3 = np.argmin([self.tarr[self.tkey[site]]['z'] for site in tri])
                a2 = 3 - a1 - a3
                tri = (tri[a1], tri[a2], tri[a3])
                    
                # Select triangle entries in the data dictionary
                l1 = l_dict[(tri[0], tri[1])]
                l2 = l_dict[(tri[1],tri[2])]
                l3 = l_dict[(tri[2], tri[0])]
                
                # Choose the appropriate polarization and compute the bs and err
                if vtype in ["vis", "qvis", "uvis"]:
                    bi = l1[vtype]*l2[vtype]*l3[vtype]  
                    bisig = np.abs(bi) * np.sqrt((l1['sigma']/np.abs(l1[vtype]))**2 +  
                                                 (l2['sigma']/np.abs(l2[vtype]))**2 + 
                                                 (l3['sigma']/np.abs(l3[vtype]))**2)   
                elif vtype == "pvis":
                    p1 = l1['qvis'] + 1j*l2['uvis']
                    p2 = l2['qvis'] + 1j*l2['uvis']
                    p3 = l3['qvis'] + 1j*l3['uvis']
                    bi = p1 * p2 * p3
                    bisig = np.abs(bi) * np.sqrt((l1['sigma']/np.abs(p1))**2 +  
                                                 (l2['sigma']/np.abs(p2))**2 + 
                                                 (l3['sigma']/np.abs(p3))**2)
                    bisig = np.sqrt(2) * bisig
                
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
        
        return outlist
   
        
    def c_phases(self, vtype='vis', mode='time', count='min'):
        """Return all independent equal time closure phase values
           Independent triangles are chosen to contain the minimum sefd station in the scan
           Set count='max' to return all closure phases
        """
        #!AC Error formula for closure phases only true in high SNR limit!
        if not mode in ('time', 'all'):
            raise Exception("possible options for mode are 'time' and 'all'")  
        if not count in ('max', 'min'):
            raise Exception("possible options for count are 'max' and 'min'")  
        
        # Get the bispectra data
        bispecs = self.bispectra(vtype=vtype, mode='time', count=count)
        
        # Reformat into a closure phase list/array
        outlist = []
        cps = []
        for bis in bispecs:
            for bi in bis:
                if len(bi) == 0: continue
                bi.dtype.names = ('time','t1','t2','t3','u1','v1','u2','v2','u3','v3','cphase','sigmacp')
                bi['sigmacp'] = bi['sigmacp']/np.abs(bi['cphase'])/DEGREE
                bi['cphase'] = np.angle(bi['cphase'])/DEGREE
                cps.append(bi.real.astype(np.dtype(DTCPHASE)))
            if mode == 'time' and len(cps) > 0:
                outlist.append(np.array(cps))
                cps = []
                
        if mode == 'all':
            outlist = np.array(cps)

        return outlist    
         
    def c_amplitudes(self, vtype='vis', mode='time', count='min'):
        """Return equal time closure amplitudes
           Set count='max' to return all closure amplitudes up to inverses
        """ 
        
        #!AC Add amplitude debiasing! (and elsewhere!)
        #!AC Error formula for closure amplitudes only true in high SNR limit!
        if not mode in ('time','all'):
            raise Exception("possible options for mode are 'time' and 'all'")
        if not count in ('max', 'min'):
            raise Exception("possible options for count are 'max' and 'min'")  
        
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
                # !AC sites are ordered by sefd - does that make sense?
                sites = sites[np.argsort([self.tarr[self.tkey[site]]['sefd'] for site in sites])]
                ref = sites[0]
                
                # Loop over other sites >3 and form minimal closure amplitude set
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
                            camp = np.abs((blue1[vtype]*blue2[vtype])/(red1[vtype]*red2[vtype]))
                            camperr = camp * np.sqrt((blue1['sigma']/np.abs(blue1[vtype]))**2 +  
                                                     (blue2['sigma']/np.abs(blue2[vtype]))**2 + 
                                                     (red1['sigma']/np.abs(red1[vtype]))**2 +
                                                     (red2['sigma']/np.abs(red2[vtype]))**2)
                                                 
                        elif vtype == "pvis":
                            p1 = blue1['qvis'] + 1j*blue1['uvis']
                            p2 = blue2['qvis'] + 1j*blue2['uvis']
                            p3 = red1['qvis'] + 1j*red1['uvis']
                            p4 = red2['qvis'] + 1j*red2['uvis']
                            
                            camp = np.abs((p1*p2)/(p3*p4))
                            camperr = np.abs(bi) * np.sqrt((blue1['sigma']/np.abs(p1))**2 +  
                                                           (blue2['sigma']/np.abs(p2))**2 + 
                                                           (red1['sigma']/np.abs(p3))**2 +
                                                           (red2['sigma']/np.abs(p3))**2)
                            camperr = np.sqrt(2) * camperr
                                        
                        # Add the closure amplitudes to the equal-time list  
                        # Our site convention is (12)(34)/(14)(23)       
                        cas.append(np.array((time, 
                                             ref, sites[i], sites[j], sites[k],
                                             blue1['u'], blue1['v'], blue2['u'], blue2['v'], 
                                             red1['u'], red1['v'], red2['u'], red2['v'],
                                             camp, camperr),
                                             dtype=DTCAMP)) 


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
                            camp = np.abs((blue1[vtype]*blue2[vtype])/(red1[vtype]*red2[vtype]))
                            camperr = camp * np.sqrt((blue1['sigma']/np.abs(blue1[vtype]))**2 +  
                                                     (blue2['sigma']/np.abs(blue2[vtype]))**2 + 
                                                     (red1['sigma']/np.abs(red1[vtype]))**2 +
                                                     (red2['sigma']/np.abs(red2[vtype]))**2)
                                                 
                        elif vtype == "pvis":
                            p1 = blue1['qvis'] + 1j*blue1['uvis']
                            p2 = blue2['qvis'] + 1j*blue2['uvis']
                            p3 = red1['qvis'] + 1j*red1['uvis']
                            p4 = red2['qvis'] + 1j*red2['uvis']
                            
                            camp = np.abs((p1*p2)/(p3*p4))
                            camperr = np.abs(bi) * np.sqrt((blue1['sigma']/np.abs(p1))**2 +  
                                                           (blue2['sigma']/np.abs(p2))**2 + 
                                                           (red1['sigma']/np.abs(p3))**2 +
                                                           (red2['sigma']/np.abs(p3))**2)
                            camperr = np.sqrt(2) * camperr
                                        
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
        
        return outlist
    
    def dirtybeam(self, npix, fov):
        """Return a square Image object of the observation dirty beam
           fov is in radian
        """
        # !AC this is a slow way of doing this
        # !AC add different types of beam weighting
        pdim = fov/npix
        u = self.unpack('u')['u']
        v = self.unpack('v')['v']

        im = np.array([[np.mean(np.cos(2*np.pi*(i*u + j*v)))
                  for i in np.arange(fov/2., -fov/2., -pdim)] 
                  for j in np.arange(fov/2., -fov/2., -pdim)])    
        
        # !AC think more carefully about the different image size cases
        im = im[0:npix, 0:npix]
        
        # Normalize to a total beam power of 1
        im = im/np.sum(im)
        
        src = self.source + "_DB"
        return Image(im, pdim, self.ra, self.dec, rf=self.rf, source=src, mjd=self.mjd)
    
    def dirtyimage(self, npix, fov):
       

        """Return a square Image object of the observation dirty image
           fov is in radian
        """
        # !AC this is a slow way of doing this
        # !AC add different types of beam weighting
        # !AC is it possible for Q^2 + U^2 > I^2 in the dirty image?
        
        pdim = fov/npix
        u = self.unpack('u')['u']
        v = self.unpack('v')['v']
        vis = self.unpack('vis')['vis']
        qvis = self.unpack('qvis')['qvis']
        uvis = self.unpack('uvis')['uvis']
        
        # Take the DFTS
        # Shouldn't need to real about conjugate baselines b/c unpack
        # does not return them
        im  = np.array([[np.mean(np.real(vis)*np.cos(2*np.pi*(i*u + j*v)) - 
                                 np.imag(vis)*np.sin(2*np.pi*(i*u + j*v)))
                  for i in np.arange(fov/2., -fov/2., -pdim)] 
                  for j in np.arange(fov/2., -fov/2., -pdim)])    
        qim = np.array([[np.mean(np.real(qvis)*np.cos(2*np.pi*(i*u + j*v)) - 
                                 np.imag(qvis)*np.sin(2*np.pi*(i*u + j*v)))
                  for i in np.arange(fov/2., -fov/2., -pdim)] 
                  for j in np.arange(fov/2., -fov/2., -pdim)])     
        uim = np.array([[np.mean(np.real(uvis)*np.cos(2*np.pi*(i*u + j*v)) - 
                                 np.imag(uvis)*np.sin(2*np.pi*(i*u + j*v)))
                  for i in np.arange(fov/2., -fov/2., -pdim)] 
                  for j in np.arange(fov/2., -fov/2., -pdim)])    
                                           
        dim = np.array([[np.mean(np.cos(2*np.pi*(i*u + j*v)))
                  for i in np.arange(fov/2., -fov/2., -pdim)] 
                  for j in np.arange(fov/2., -fov/2., -pdim)])   
           
        # !AC is this the correct normalization?
        im = im/np.sum(dim)
        qim = qim/np.sum(dim)
        uim = uim/np.sum(dim)
 
        # !AC think more carefully about the different image size cases here       
        im = im[0:npix, 0:npix]
        qim = qim[0:npix, 0:npix]
        uim = uim[0:npix, 0:npix]   
        
        out = Image(im, pdim, self.ra, self.dec, rf=self.rf, source=self.source, mjd=self.mjd)
        out.add_qu(qim, uim)
        return out
    
    def cleanbeam(self, npix, fov):
        """Return a square Image object of the observation fitted (clean) beam
           fov is in radian
        """
        # !AC include other beam weightings
        im = make_square(self, npix, fov)
        beamparams = self.fit_beam()
        im = add_gauss(im, 1.0, beamparams)
        return im
        
    def fit_beam(self):
        """Fit a gaussian to the dirty beam and return the parameters (fwhm_maj, fwhm_min, theta).
           All params are in radian and theta is measured E of N.
           Fit the quadratic expansion of the Gaussian (normalized to 1 at the peak) 
           to the expansion of dirty beam with the same normalization
        """    
        # !AC include other beam weightings
          
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
        params = scipy.optimize.minimize(fit_chisq, guess, args=(abc,), method='Powell')
        
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
            
    def plotall(self, field1, field2, rangex=False, rangey=False, conj=False):
        """Make a scatter plot of 2 real observation fields with errors
           If conj==True, display conjugate baselines"""
        
        # Determine if fields are valid
        if (field1 not in FIELDS) and (field2 not in FIELDS):
            raise Exception("valid fields are " + string.join(FIELDS))
                              
        # Unpack x and y axis data
        data = self.unpack([field1,field2], conj=conj)
        
        # X error bars
        if field1 in ['amp', 'qamp', 'uamp']:
            sigx = self.unpack('sigma',conj=conj)['sigma']
        elif field1 in ['phase', 'uphase', 'qphase']:
            sigx = (self.unpack('sigma',conj=conj)['sigma'])/(self.unpack('amp',conj=conj)['amp'])/DEGREE
        elif field1 == 'pamp':
            sigx = np.sqrt(2)*self.unpack('sigma',conj=conj)['sigma']
        elif field1 == 'pphase':
            sigx = np.sqrt(2)*(self.unpack('sigma',conj=conj)['sigma'])/(self.unpack('pamp',conj=conj)['pamp'])/DEGREE
        elif field1 == 'mamp':
            sigx = merr(self.unpack('sigma',conj=conj)['sigma'], 
                        self.unpack('amp',conj=conj)['amp'], 
                        self.unpack('mamp',conj=conj)['mamp'])
        elif field1 == 'mphase':
            sigx = merr(self.unpack('sigma',conj=conj)['sigma'], 
                        self.unpack('amp',conj=conj)['amp'], 
                        self.unpack('mamp',conj=conj)['mamp']) / self.unpack('mamp',conj=conj)['mamp']
        else:
            sigx = None
            
        # Y error bars
        if field2 in ['amp', 'qamp', 'uamp']:
            sigy = self.unpack('sigma',conj=conj)['sigma']
        elif field2 in ['phase', 'uphase', 'qphase']:
            sigy = (self.unpack('sigma',conj=conj)['sigma'])/(self.unpack('amp',conj=conj)['amp'])/DEGREE
        elif field2 == 'pamp':
            sigy = np.sqrt(2)*self.unpack('sigma',conj=conj)['sigma']
        elif field2 == 'pphase':
            sigy = np.sqrt(2)*(self.unpack('sigma',conj=conj)['sigma'])/(self.unpack('pamp',conj=conj)['pamp'])/DEGREE
        elif field2 == 'mamp':
            sigy = merr(self.unpack('sigma',conj=conj)['sigma'],
                        self.unpack('amp',conj=conj)['amp'],
                        self.unpack('mamp',conj=conj)['mamp'])
        elif field2 == 'mphase':
            sigy = merr(self.unpack('sigma',conj=conj)['sigma'],
                        self.unpack('amp',conj=conj)['amp'], 
                        self.unpack('mamp',conj=conj)['mamp']) / self.unpack('mamp',conj=conj)['mamp']
        else:
            sigy = None
        
        # Data ranges
        if not rangex:
            rangex = [np.min(data[field1]) - 0.2 * np.abs(np.min(data[field1])), 
                      np.max(data[field1]) + 0.2 * np.abs(np.max(data[field1]))] 
        if not rangey:
            rangey = [np.min(data[field2]) - 0.2 * np.abs(np.min(data[field2])), 
                      np.max(data[field2]) + 0.2 * np.abs(np.max(data[field2]))] 
        
        # Plot the data
        plt.figure()
        plt.cla()
        plt.errorbar(data[field1], data[field2], xerr=sigx, yerr=sigy, fmt='b.')
        plt.xlim(rangex)
        plt.ylim(rangey)
        plt.xlabel(field1)
        plt.ylabel(field2)
        plt.show(block=False)
        return
        
    def plot_bl(self, site1, site2, field, rangey=False):
        """Plot a field over time on a baseline"""
        
        # Determine if fields are valid
        if field not in FIELDS:
            raise Exception("valid fields are " + string.join(FIELDS))
        
        # Get the data from data table on the selected baseline
        plotdata = []
        tlist = self.tlist(conj=True)
        for scan in tlist:
            for obs in scan:
                if (obs['t1'], obs['t2']) == (site1, site2):
                    time = obs['time']
                    if field == 'uvdist':
                        plotdata.append([time, np.abs(obs['u'] + 1j*obs['v']), 0])
                    
                    elif field in ['amp', 'qamp', 'uamp']:
                        if field == 'amp': l = 'vis'
                        elif field == 'qamp': l = 'qvis'
                        elif field == 'uamp': l = 'uvis'
                        plotdata.append([time, np.abs(obs[l]), obs['sigma']])
                    
                    elif field in ['phase', 'qphase', 'uphase']:
                        if field == 'phase': l = 'vis'
                        elif field == 'qphase': l = 'qvis'
                        elif field == 'uphase': l = 'uvis'
                        plotdata.append([time, np.angle(obs[l])/DEGREE, obs['sigma']/np.abs(obs[l])/DEGREE])
                    
                    elif field == 'pamp':
                        plotdata.append([time, np.abs(obs['qvis'] + 1j*obs['uvis']), np.sqrt(2)*obs['sigma']])
                    
                    elif field == 'pphase':
                        plotdata.append([time, 
                                         np.angle(obs['qvis'] + 1j*obs['uvis'])/DEGREE, 
                                         np.sqrt(2)*obs['sigma']/np.abs(obs['qvis'] + 1j*obs['uvis'])/DEGREE
                                       ])
                    
                    elif field == 'mamp':
                        plotdata.append([time,
                                         np.abs((obs['qvis'] + 1j*obs['uvis'])/obs['vis']), 
                                         merr(obs['sigma'], obs['vis'], (obs['qvis']+1j*obs['uvis'])/obs['vis'])
                                       ])
                    
                    elif field == 'mphase':
                        plotdata.append([time, 
                                        np.angle((obs['qvis'] + 1j*obs['uvis'])/obs['vis'])/DEGREE, 
                                        (merr(obs['sigma'], obs['vis'], (obs['qvis']+1j*obs['uvis'])/obs['vis'])/
                                           np.abs((obs['qvis']+1j*obs['uvis'])/obs['vis'])/DEGREE)
                                       ])
                    
                    else:
                        plotdata.append([time, obs[field], 0])
                    
                    # Assume only one relevant entry per scan
                    break
        
        # Plot the data                        
        plotdata = np.array(plotdata)
    
        if not rangey:
            rangey = [np.min(plotdata[:,1]) - 0.2 * np.abs(np.min(plotdata[:,1])), 
                      np.max(plotdata[:,1]) + 0.2 * np.abs(np.max(plotdata[:,1]))] 

        plt.figure()    
        plt.cla()
        plt.errorbar(plotdata[:,0], plotdata[:,1], yerr=plotdata[:,2], fmt='.')
        plt.xlim([self.tstart,self.tstop])
        plt.ylim(rangey)
        plt.xlabel('GMT (h)')
        plt.ylabel(field)
        plt.title('%s - %s'%(site1,site2))
        plt.show(block=False)    
                
    def plot_cphase(self, site1, site2, site3, rangey=False):
        """Plot closure phase over time on a triangle"""

        # Get closure phases (maximal set)
        
        cphases = self.c_phases(mode='time', count='max')
        
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
        if not rangey:
            rangey = [np.min(plotdata[:,1]) - 0.2 * np.abs(np.min(plotdata[:,1])), 
                      np.max(plotdata[:,1]) + 0.2 * np.abs(np.max(plotdata[:,1]))] 
        
        # Plot
        plt.figure()
        plt.cla()
        plt.errorbar(plotdata[:,0], plotdata[:,1], yerr=plotdata[:,2], fmt='.')
        plt.xlim([self.tstart,self.tstop])
        plt.ylim(rangey)
        plt.xlabel('GMT (h)')
        plt.ylabel('Closure Phase (deg)')
        plt.title('%s - %s - %s' % (site1,site2,site3))
        plt.show(block=False)              
        
    def plot_camp(self, site1, site2, site3, site4, rangey=False):
        """Plot closure amplitude over time on a quadrange
           (1-2)(3-4)/(1-4)(2-3)
        """
        quad = (site1, site2, site3, site4)
        b1 = set((site1, site2))
        r1 = set((site1, site4))
              
        # Get the closure amplitudes
        camps = self.c_amplitudes(mode='time', count='max')
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
        if not rangey:
            rangey = [np.min(plotdata[:,1]) - 0.2 * np.abs(np.min(plotdata[:,1])), 
                      np.max(plotdata[:,1]) + 0.2 * np.abs(np.max(plotdata[:,1]))] 
        
        # Plot                            
        plotdata = np.array(plotdata)
        plt.figure
        plt.cla()
        plt.errorbar(plotdata[:,0], plotdata[:,1], yerr=plotdata[:,2], fmt='.')
        plt.xlim([self.tstart,self.tstop])
        plt.ylim(rangey)
        plt.xlabel('GMT (h)')
        plt.ylabel('Closure Amplitude')
        plt.title('(%s - %s)(%s - %s)/(%s - %s)(%s - %s)'%(site1,site2,site3,site4,
                                                           site1,site4,site2,site3))
        plt.show(block=False)       

    def save_txt(self, fname):
        """Save visibility data to a text file"""
        
        # Get the necessary data and the header
        outdata = self.unpack(['time', 'tint', 't1', 't2', 'el1', 'el2', 'tau1','tau2',
                               'u', 'v', 'amp', 'phase', 'qamp', 'qphase', 'uamp', 'uphase', 'sigma'])
        head = ("SRC: %s \n" % self.source +
                    "RA: " + rastring(self.ra) + "\n" + "DEC: " + decstring(self.dec) + "\n" +
                    "MJD: %.4f - %.4f \n" % (fracmjd(self.mjd,self.tstart), fracmjd(self.mjd,self.tstop)) + 
                    "RF: %.4f GHz \n" % (self.rf/1e9) + 
                    "BW: %.4f GHz \n" % (self.bw/1e9) +
                    "PHASECAL: %i \n" % self.phasecal + 
                    "AMPCAL: %i \n" % self.ampcal + 
                    "----------------------------------------------------------------\n" +
                    "Site       X(m)             Y(m)             Z(m)           SEFD\n"
                )
        
        for i in range(len(self.tarr)):
            head += "%-8s %15.5f  %15.5f  %15.5f  %6.4f \n" % (self.tarr[i]['site'], 
                                                               self.tarr[i]['x'], self.tarr[i]['y'], self.tarr[i]['z'], 
                                                               self.tarr[i]['sefd'])

        head += ("----------------------------------------------------------------\n" +
                "time (hr) tint    T1     T2    Elev1   Elev2  Tau1   Tau2   U (lambda)       V (lambda)         "+
                "Iamp (Jy)    Iphase(d)  Qamp (Jy)    Qphase(d)   Uamp (Jy)    Uphase(d)   sigma (Jy)"
                )
          
        # Format and save the data
        fmts = ("%011.8f %4.2f %6s %6s  %4.2f   %4.2f  %4.2f   %4.2f  %16.4f %16.4f    "+
               "%10.8f %10.4f   %10.8f %10.4f    %10.8f %10.4f    %10.8f")
        np.savetxt(fname, outdata, header=head, fmt=fmts)
        return

    def save_uvfits(self, fname):
        """Save visibility data to uvfits
           Needs template.UVP file
           Antenna table is currently incorrect"""
        
        # Open template UVFITS
        hdulist = fits.open('./template.UVP')
        
        # Load the array data
        tarr = self.tarr
        tnames = tarr['site']
        tnums = np.arange(1, len(tarr)+1)
        xyz = np.array([[tarr[i]['x'],tarr[i]['y'],tarr[i]['z']] for i in np.arange(len(tarr))])
        sefd = tarr['sefd']
        
	col1 = fits.Column(name='ANNAME', format='8A', array=tnames)
	col2 = fits.Column(name='STABXYZ', format='3D', array=xyz)
	col3 = fits.Column(name='NOSTA', format='IJ', array=tnums)
	col4 = fits.Column(name='SEFD', format='1D', array=sefd)
        tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs([col1,col2,col3,col4]), name='AIPS AN')
        hdulist['AIPS AN'] = tbhdu
    
        # Header (based on the BU format)
        header = hdulist[0].header
        
        header['OBSRA'] = self.ra * 180./12.
        header['OBSDEC'] = self.dec
        header['OBJECT'] = self.source
        header['MJD'] = self.mjd
        header['TELESCOP'] = 'VLBI' # !AC Can I change this??
        header['INSTRUME'] = 'VLBI'
        header['CTYPE4'] = 'FREQ'
        header['CRVAL4'] = self.rf
        header['CDELT4'] = self.bw   
        header['CRPIX4'] = 1.e0
        header['CTYPE6'] = 'RA'
        header['CRVAL6'] = header['OBSRA']
        header['CTYPE7'] = 'DEC'
        header['CRVAL7'] = header['OBSRA']
        header['PTYPE1'] = 'UU---SIN'
        header['PSCAL1'] = 1/self.rf
        header['PTYPE2'] = 'VV---SIN'
        header['PSCAL2'] = 1/self.rf
        header['PTYPE3'] = 'WW---SIN'
        header['PSCAL3'] = 1/self.rf
             
        # Get data
        obsdata = self.unpack(['time','tint','u','v','vis','qvis','uvis','sigma','t1','t2','el1','el2','tau1','tau2'])
        ndat = len(obsdata['time'])
        
        # times and tints
        jds = (self.mjd + 2400000.5) + (obsdata['time'] / 24.0) 
        tints = obsdata['tint']
        
        # Baselines            
        # !AC These HAVE to be correct for CLEAN to work. Why?
        t1 = [self.tkey[scope] + 1 for scope in obsdata['t1']]
        t2 = [self.tkey[scope] + 1 for scope in obsdata['t2']]
        bl = 256*np.array(t1) + np.array(t2)
        
        # elevations
        el1 = obsdata['el1']
        el2 = obsdata['el2']
        
        # opacities
        tau1 = obsdata['tau1']
        tau2 = obsdata['tau2']
        
        # uv are in lightseconds
        u = obsdata['u']
        v = obsdata['v']
        
        # rr, ll, lr, rl, weights
        # !AC Assume V = 0 (linear polarization only)
        rr = ll = obsdata['vis'] # complex
        rl = obsdata['qvis'] + 1j*obsdata['uvis']
        lr = obsdata['qvis'] - 1j*obsdata['uvis']
        weight = 1 / (2 * obsdata['sigma']**2)
        
        # Data array
        outdat = np.zeros((ndat, 1, 1, 1, 1, 4, 3))
        outdat[:,0,0,0,0,0,0] = np.real(rr)
        outdat[:,0,0,0,0,0,1] = np.imag(rr)
        outdat[:,0,0,0,0,0,2] = weight
        outdat[:,0,0,0,0,1,0] = np.real(ll)
        outdat[:,0,0,0,0,1,1] = np.imag(ll)
        outdat[:,0,0,0,0,1,2] = weight
        outdat[:,0,0,0,0,2,0] = np.real(rl)
        outdat[:,0,0,0,0,2,1] = np.imag(rl)
        outdat[:,0,0,0,0,2,2] = weight
        outdat[:,0,0,0,0,3,0] = np.real(lr)
        outdat[:,0,0,0,0,3,1] = np.imag(lr)
        outdat[:,0,0,0,0,3,2] = weight    
        
        # Save data
        pars = ['UU---SIN', 'VV---SIN', 'WW---SIN', 'BASELINE', 'DATE', '_DATE', 
                'INTTIM', 'ELEV1', 'ELEV2', 'TAU1', 'TAU2']
        x = fits.GroupData(outdat, parnames=pars, 
                           pardata=[u, v, np.zeros(ndat), bl, jds, np.zeros(ndat), tints, el1, el2,tau1,tau2], 
                           bitpix=-64)
        hdulist[0].data = x
        hdulist[0].header = header
        hdulist.writeto(fname, clobber=True)
        
        return
        
##################################################################################################
# Object Construction Functions
##################################################################################################    
           
def load_array(filename):
    """Read an array from a text file and return an Array object
    """
    
    tdata = np.loadtxt(filename,dtype=str)
    if tdata.shape[1] != 5:
        raise Exception("Array file should have format: (name, x, y, z, SEFD)") 
    tdata = [np.array((x[0],float(x[1]),float(x[2]),float(x[3]),float(x[4])), dtype=DTARR) for x in tdata]
    tdata = np.array(tdata)
    return Array(tdata)
      
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
    file.readline()
    file.readline()
    
    # read the tarr
    line = file.readline().split()
    tarr = []
    while line[1][0] != "-":
        tarr.append(np.array((line[1], line[2], line[3], line[4], line[5]), dtype=DTARR))
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
        el1 = float(row[4])
        el2 = float(row[5])
        tau1 = float(row[6])
        tau2 = float(row[7])
        u = float(row[8])
        v = float(row[9])
        vis = float(row[10]) * np.exp(1j * float(row[11]) * DEGREE)
        if datatable.shape[1] == 17:
            qvis = float(row[12]) * np.exp(1j * float(row[13]) * DEGREE)
            uvis = float(row[14]) * np.exp(1j * float(row[15]) * DEGREE)
            sigma = float(row[16])
        elif datatable.shape[1] == 13:
            qvis = 0+0j
            uvis = 0+0j
            sigma = float(row[12])
        else:
            raise Exception('Text file does not have the right number of fields!')
            
        datatable2.append(np.array((time, tint, t1, t2, el1, el2, tau1, tau2, 
                                    u, v, vis, qvis, uvis, sigma), dtype=DTPOL))
    
    # Return the datatable                      
    datatable2 = np.array(datatable2)
    return Obsdata(ra, dec, rf, bw, datatable2, tarr, source=src, mjd=mjd, ampcal=ampcal, phasecal=phasecal)        

def load_obs_maps(arrfile, obsspec, ifile, qfile=0, ufile=0, src='SgrA', mjd=48277, ampcal=False, phasecal=False):
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
        elif line[0] == 'Corr_chan_bw':  #!AC what if multiple channels?
            bw = float(line[2]) * 1e6 #MHz
        elif line[0] == 'Channel': #!AC what if multiple scans with different params?
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
            el1 = 0.
            el2 = 0.
            tau1 = 0.
            tau2 = 0.
            vis = float(line[7][:-1]) * np.exp(1j*float(line[8][:-1])*DEGREE)
            sigma = float(line[10])
            datatable.append(np.array((time, tint, t1, t2, el1, el2, tau1, tau2, 
                                        u, v, vis, 0.0,0.0, sigma), dtype=DTPOL))
    
    datatable = np.array(datatable)
    #!AC: qfile and ufile must have exactly the same format as ifile
    #!AC: add some consistency check 
    if not qfile==0:
        f = open(qfile)
        i = 0
        for line in f:
            line = line.split()
            if not (line[0] in ['UV', 'Scan','\n']):
                datatable[i]['qvis'] = float(line[7][:-1]) * np.exp(1j*float(line[8][:-1])*DEGREE)
                i += 1
            
    if not ufile==0:
        f = open(ufile)
        i = 0
        for line in f:
            line = line.split()
            if not (line[0] in ['UV', 'Scan','\n']):
                datatable[i]['uvis'] = float(line[7][:-1]) * np.exp(1j*float(line[8][:-1])*DEGREE)
                i += 1                                
    
    # Return the datatable                      
    return Obsdata(ra, dec, rf, bw, datatable, tdata, source=src, mjd=mjd)        

def load_obs_uvfits(filename, flipbl=False):
    """Load uvfits data from a uvfits file
       Need an associated array file  text file with the telescope parameters
       The sites must be in the order in the array file that corresponds numbers they have in the uvfits file
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
        sefd = hdulist['AIPS AN'].data['SEFD']
    except KeyError:
        print "Warning! no SEFD data in UVfits file"
        sefd = np.zeros(len(tnames))
        
    tarr = [np.array((tnames[i], xyz[i][0], xyz[i][1], xyz[i][2], sefd[i]),
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
    jds = data['DATE'][mask]
    mjd = int(jdtomjd(np.min(jds)))
    
    #!AC: There seems to be different behavior here - 
    #!AC: BU puts date in _DATE 
    if len(set(data['DATE'])) > 2:
        times = np.array([mjdtogmt(jdtomjd(jd)) for jd in jds])
    else:
        times = data['_DATE'][mask] * 24.0
    
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
    
    # Elevations (not in BU files)
    try: 
        el1 = data['ELEV1'][mask]
        el2 = data['ELEV2'][mask]
    except KeyError:
        el1 = el2 = np.zeros(len(t1))

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
        u = data['UU'][mask] * rf
        v = data['VV'][mask] * rf
           
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
    ivis = (rr + ll)/2.0
    qvis = (rl + lr)/2.0
    uvis = (rl - lr)/(2.0j)
    isig = np.sqrt(rrsig**2 + llsig**2)/2.0
    qsig = np.sqrt(rlsig**2 + lrsig**2)/2.0
    usig = qsig
    
    # !AC Should sigma be the avg of the stokes sigmas, or just the I sigma?  
    sigma = isig 
    
    # !AC reverse sign of baselines for correct imaging?
    if flipbl:
        u = -u
        v = -v
    
    # Make a datatable
    # !AC Can I make this faster?
    datatable = []
    for i in xrange(len(times)):
        datatable.append(np.array((
                           times[i], tints[i], 
                           t1[i], t2[i], el1[i], el2[i], tau1[i], tau2[i], 
                           u[i], v[i],
                           ivis[i], qvis[i], uvis[i], sigma[i]
                           ), dtype=DTPOL
                         ))
    datatable = np.array(datatable)

    return Obsdata(ra, dec, rf, bw, datatable, tarr, source=src, mjd=mjd, ampcal=True, phasecal=True)

def load_im_txt(filename):
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
    mjd = float(file.readline().split()[2])
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
    outim = Image(image, psize_x, ra, dec, rf=rf, source=src, mjd=mjd)
    
    # Look for Stokes Q and U
    qimage = uimage = np.zeros(image.shape)
    if datatable.shape[1] == 5:
        qimage = datatable[:,3].reshape(ydim_p, xdim_p)
        uimage = datatable[:,4].reshape(ydim_p, xdim_p)
    
    if np.any((qimage != 0) + (uimage != 0)):
        print 'Loaded Stokes I, Q, and U images'
        outim.add_qu(qimage, uimage)
    else:
        print 'Loaded Stokes I image only'
    
    return outim
    
def load_im_fits(filename, punit="deg"):
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
    else: mjd = 48277.0 
    
    if 'FREQ' in header.keys(): rf = header['FREQ']
    else: rf = 230e9
    
    if 'OBJECT' in header.keys(): src = header['OBJECT']
    else: src = 'SgrA'
    
    # Get the image and create the object
    data = hdulist[0].data
    data = data.reshape((data.shape[-2],data.shape[-1]))
    image = data[::-1,:] # flip y-axis!
    outim = Image(image, psize_x, ra, dec, rf=rf, source=src, mjd=mjd)
    
    # Look for Stokes Q and U
    qimage = uimage = np.array([])
    for hdu in hdulist[1:]:
        header = hdu.header
        data = hdu.data
        data = data.reshape((data.shape[-2],data.shape[-1]))
        if 'STOKES' in header.keys() and header['STOKES'] == 'Q':
            qimage = data[::-1,:] # flip y-axis!
        if 'STOKES' in header.keys() and header['STOKES'] == 'U':
            uimage = data[::-1,:] # flip y-axis!
    if qimage.shape == uimage.shape == image.shape:
        print 'Loaded Stokes I, Q, and U images'
        outim.add_qu(qimage, uimage)
    else:
        print 'Loaded Stokes I image only'
                
    return outim

##################################################################################################
# Image Construction Functions
##################################################################################################

def make_square(obs, npix, fov):
    """Make an empty prior image
       obs is an observation object
       fov is in radians
    """ 
    pdim = fov/npix
    im = np.zeros((npix,npix))
    return Image(im, pdim, obs.ra, obs.dec, rf=obs.rf, source=obs.source, mjd=obs.mjd)

def add_flat(im, flux):
    """Add flat background to an image""" 
    
    imout = (im.imvec + (flux/float(len(im.imvec))) * np.ones(len(im.imvec))).reshape(im.ydim,im.xdim)
    out = Image(imout, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd) 
    return out

def add_gauss(im, flux, beamparams, x=0, y=0):
    """Add a gaussian to an image
       beamparams is [fwhm_maj, fwhm_min, theta], all in rad
       x,y are gaussian position in rad
       theta is the orientation angle measured E of N
    """ 
    
    xfov = im.xdim * im.psize
    yfov = im.ydim * im.psize
    sigma_maj = beamparams[0] / (2. * np.sqrt(2. * np.log(2.))) 
    sigma_min = beamparams[1] / (2. * np.sqrt(2. * np.log(2.)))
    cth = np.cos(beamparams[2])
    sth = np.sin(beamparams[2])

    gauss = np.array([[np.exp(-((j-y)*cth + (i-x)*sth)**2/(2*sigma_maj**2) - ((i-x)*cth - (j-y)*sth)**2/(2.*sigma_min**2))
                      for i in np.arange(xfov/2., -xfov/2., -im.psize)] 
                      for j in np.arange(yfov/2., -yfov/2., -im.psize)])    
  
    # !AC think more carefully about the different cases for array size here
    gauss = gauss[0:im.ydim, 0:im.xdim]
    
    imout = im.imvec.reshape(im.ydim, im.xdim) + (gauss * flux/np.sum(gauss))
    out = Image(imout, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd)
    return out


def add_const_m(im, mag, angle):
    """Add a constant fractional polarization to image
       angle is in radians""" 
    
    if not (0 < mag < 1):
        raise Exception("fractional polarization magnitude must be beween 0 and 1!")
    
    imi = im.imvec.reshape(im.ydim,im.xdim)    
    imq = qimage(im.imvec, mag * np.ones(len(im.imvec)), angle*np.ones(len(im.imvec))).reshape(im.ydim,im.xdim)
    imu = uimage(im.imvec, mag * np.ones(len(im.imvec)), angle*np.ones(len(im.imvec))).reshape(im.ydim,im.xdim)
    out = Image(imi, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd)
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
    xfov = image.xdim * image.psize
    yfov = image.ydim * image.psize
    
    if beamparams[0] > 0.0:
        sigma_maj = frac * beamparams[0] / (2. * np.sqrt(2. * np.log(2.))) 
        sigma_min = frac * beamparams[1] / (2. * np.sqrt(2. * np.log(2.))) 
        cth = np.cos(beamparams[2])
        sth = np.sin(beamparams[2])
        gauss = np.array([[np.exp(-(j*cth + i*sth)**2/(2*sigma_maj**2) - (i*cth - j*sth)**2/(2.*sigma_min**2))
                                  for i in np.arange(xfov/2., -xfov/2., -image.psize)] 
                                  for j in np.arange(yfov/2., -yfov/2., -image.psize)])

        # !AC think more carefully about the different image size cases here
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
                                  for i in np.arange(xfov/2., -xfov/2., -image.psize)] 
                                  for j in np.arange(yfov/2., -yfov/2., -image.psize)])
        
        # !AC think more carefully about the different cases here
        gauss = gauss[0:self.ydim, 0:self.xdim]
        gauss = gauss / np.sum(gauss) # normalize to 1        
        
        # Convolve
        qim = scipy.signal.fftconvolve(gauss, qim, mode='same')
        uim = scipy.signal.fftconvolve(gauss, uim, mode='same')
                                  
    
    out = Image(im, image.psize, image.ra, image.dec, rf=image.rf, source=image.source, mjd=image.mjd)                        
    if len(image.qvec):
        out.add_qu(qim, uim)
    return out  
        
##################################################################################################
# Scattering Functions
##################################################################################################
def deblur(obs):
    """Deblur the observation obs by dividing with the Sgr A* scattering kernel.
       Returns a new observation.
    """
    
    datatable = np.array(obs.data, copy=True)
    vis = datatable['vis']
    qvis = datatable['qvis']
    uvis = datatable['uvis']
    sigma = datatable['sigma']
    u = datatable['u']
    v = datatable['v']
    
    for i in range(len(vis)):
        ker = sgra_kernel_uv(obs.rf, u[i], v[i])
        vis[i] = vis[i] / ker
        qvis[i] = qvis[i] / ker
        uvis[i] = uvis[i] / ker
        sigma[i] = sigma[i] / ker
    
    datatable['vis'] = vis
    datatable['qvis'] = qvis
    datatable['uvis'] = uvis
    datatable['sigma'] = sigma
    
    obsdeblur = Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, datatable, obs.tarr)
    return obsdeblur
    
def sgra_kernel_uv(rf, u, v):
    """Return the value of the Sgr A* scattering kernel at a given u,v pt (in lambda), 
       at a given frequency rf (in Hz).
       Values from Bower et al.
    """
    
    lcm = (C/rf) * 100 # in cm
    sigma_maj = FWHM_MAJ * (lcm**2) / (2*np.sqrt(2*np.log(2))) * RADPERUAS
    sigma_min = FWHM_MIN * (lcm**2) / (2*np.sqrt(2*np.log(2))) * RADPERUAS
    theta = POS_ANG * DEGREE
    
    
    # Covarience matrix
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
# Other Functions
##################################################################################################

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
    
def merr(sigma, I, m):
    """Return the error in mbreve real and imaginary parts"""
    return sigma * np.sqrt((2 + np.abs(m)**2)/ (np.abs(I) ** 2))
       
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
    
def rastring(ra):
    """Convert a ra in fractional hours to formatted string"""
    h = int(ra)
    m = int((ra-h)*60.)
    s = (ra-h-m/60.)*3600.
    out = "%2i h %2i m %2.4f s" % (h,m,s)
    return out 

def decstring(dec):
    """Convert a dec in fractional degrees to formatted string"""
    
    deg = int(dec)
    m = int((abs(dec)-abs(deg))*60.)
    s = (abs(dec)-abs(deg)-m/60.)*3600.
    out = "%2i deg %2i m %2.4f s" % (deg,m,s)
    return out

def gmtstring(gmt):
    """Convert a gmt in fractional hours to formatted string"""
    
    if gmt > 24.0: gmt = gmt-24.0
    h = int(gmt)
    m = int((gmt-h)*60.)
    s = (gmt-h-m/60.)*3600.
    out = "%02i:%02i:%2.4f" % (h,m,s)
    return out 

def fracmjd(mjd, gmt):
    """Convert a int mjd + gmt (frac. hr.) into a fractional mjd"""
    
    return int(mjd) + gmt/24.

def mjdtogmt(mjd):
    """Return the gmt of a fractional mjd, in days"""
    
    return (mjd - int(mjd)) * 24.0
    
def jdtomjd(jd):
    """Return the mjd of a jd"""
    
    return jd - 2400000.5
    
def earthrot(vec, theta):
    """Rotate a vector about the z-direction by theta (degrees)"""
    
    x = theta * DEGREE
    return np.dot(np.array(((np.cos(x),-np.sin(x),0),(np.sin(x),np.cos(x),0),(0,0,1))),vec)

def elev(obsvec, sourcevec):
    """Return the elevation of a source with respect to an observer in degrees"""
    anglebtw = np.dot(obsvec,sourcevec)/np.linalg.norm(obsvec)/np.linalg.norm(sourcevec)
    el = 90 - np.arccos(anglebtw)/DEGREE
    return el
        
def elevcut(obsvec,sourcevec):
    """Return True if a source is observable by a telescope vector"""
    angle = elev(obsvec, sourcevec)
    return ELEV_LOW < angle < ELEV_HIGH

        
def blnoise(sefd1, sefd2, tint, bw):
    """Determine the standard deviation of Gaussian thermal noise on a baseline (2-bit quantization)"""
    
    return np.sqrt(sefd1*sefd2/(2*bw*tint))/0.88


def cerror(sigma):
    """Return a complex number drawn from a circular complex Gaussian of zero mean"""
    
    return np.random.normal(loc=0,scale=sigma) + 1j*np.random.normal(loc=0,scale=sigma)

def hashrandn(*args):
    """set the seed according to a collection of arguments and return random gaussian var"""
    np.random.seed(hash(",".join(map(repr,args))) % 4294967295)
    return np.random.randn()

def hashrand(*args):
    """set the seed according to a collection of arguments and return random number in 0,1"""
    np.random.seed(hash(",".join(map(repr,args))) % 4294967295)
    return np.random.rand()

      
def ftmatrix(pdim, xdim, ydim, uvlist):
    """Return a DFT matrix for the xdim*ydim image with pixel width pdim
       that extracts spatial frequencies of the uv points in uvlist.
    """
   
    if xdim % 2:
        xlist = pdim * np.arange((xdim-1)/2, -(xdim+1)/2, -1)
    else: 
        xlist = pdim * np.arange(xdim/2-1, -xdim/2-1, -1)
    
    if ydim % 2:
        ylist = pdim * np.arange((ydim-1)/2, -(ydim+1)/2, -1)
    else: 
        ylist = pdim * np.arange(ydim/2-1, -ydim/2-1, -1)
    
    # Fortunately, this works for both uvlist recarrays and ndarrays, but be careful! 
    ftmatrices = np.array([np.outer(np.exp(-2j*np.pi*ylist*uv[1]), np.exp(-2j*np.pi*xlist*uv[0])) for uv in uvlist])
    return np.reshape(ftmatrices, (len(uvlist), xdim*ydim))

   
