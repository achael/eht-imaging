# vlbi_imaging_utils.py
# Andrew Chael, 01/12/16
# Utilities for generating and manipulating VLBI images, datasets, and arrays

# TODO: 
#       Add ability to load different scattering kernel
#       Fix case where there are no data points
#       Screen for 0 errors 
#       Add closure amplitude debiasing
#       Closure amplitude combinations
#       Make forming a data table clearer
#       Add different i,q,u,v SEFDs and calibration errors?
#       reseed random number generators each time


import string
import numpy as np
import numpy.lib.recfunctions as rec
import matplotlib.pyplot as plt
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

# Sgr A* Kernel Values (Bower et al., in uas/cm^2)
FWHM_MAJ = 1.309 * 1000 # in uas
FWHM_MIN = 0.64 * 1000
POS_ANG = -78 * DEGREE

# Observation recarray datatypes
DTPOL = [('time','f8'),('tint','f8'),
         ('t1','a32'),('t2','a32'),
         ('el1','f8'),('el2','f8'),
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
FIELDS = ['time','tint','u','v','uvdist','t1','t2','el1','el2',
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
    
    def flip_chi(self):
        """Change between different conventions for measuring position angle (East of North vs up from x axis)
        """
        self.qvec = - self.qvec
        return
        
    def observe_same(self, obs, ampcal="True", phasecal="True", sgrscat=False):
        """Observe the image on the same baselines with the same noise as an existing observation object
           if sgrscat==True, the visibilites will be blurred by the Sgr A* scattering kernel
        """
        
        # Check for agreement in coordinates and frequency 
        if (self.ra!= obs.ra) or (self.dec != obs.dec):
            raise Exception("Image coordinates are not the same as observtion coordinates!")
	    if (self.rf != obs.rf):
	        raise Exception("Image frequency is not the same as observation frequency!")
        
        # Get data
        obsdata = obs.data
        
        sites = obsdata[['t1','t2']].view(('a32',2))
        elevs = obsdata[['el1','el2']].view(('f8',2))
        time = obsdata['time'].view(('f8',1))
        uv = obsdata[['u','v']].view(('f8',2))
        sigma_true = sigma_est = obsdata['sigma'].view(('f8',1))
        
        # Perform DFT
        mat = ftmatrix(self.psize, self.xdim, self.ydim, uv)
        vis = np.dot(mat, self.imvec)
        
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
        
        # !AC Check the order of operations here
        # Add gain and opacity uncertanities to the RMS noise
        if not ampcal:
            # amplitude gain (forward gain on power) = 1.0 + 10% fixed + 10% time-dependent
            gain1 = np.array([1.0 + 0.1 * hashrandn(sites[i,0], 'gain') + 0.1 * hashrandn(sites[i,0], 'gain', time[i]) for i in xrange(len(time))])
            gain2 = np.array([1.0 + 0.1 * hashrandn(sites[i,1], 'gain') + 0.1 * hashrandn(sites[i,1], 'gain', time[i]) for i in xrange(len(time))])
           
            # opacity = 0.1 + 10% time-dependent
            tau1 = np.array([0.1 + 0.01 * hashrandn(sites[i,0], 'tau', time[i]) for i in xrange(len(time))])
            tau2 = np.array([0.1 + 0.01 * hashrandn(sites[i,1], 'tau', time[i]) for i in xrange(len(time))])
            
            # Correct noise RMS for gain variation and opacity
            sigma_true = sigma_true / np.sqrt(gain1 * gain2)
            sigma_true = sigma_true * np.sqrt(np.exp(tau1/np.sin(elevs[:,0]*DEGREE) + tau2/np.sin(elevs[:,1]*DEGREE))) 
            
            # Estimated noise using 10% gain and 0.1 opacity
            sigma_est = sigma_est * np.sqrt(np.exp(0.1/np.sin(elevs[:,0]*DEGREE) + 0.1/np.sin(elevs[:,1]*DEGREE)))
            
        # Add the noise the gain error
        vis  = (vis + cerror(sigma_true)) * (sigma_est/sigma_true)
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
        return Obsdata(self.ra, self.dec, self.rf, obs.bw, obsdata)
    
    def observe(self, array, tint, tadv, tstart, tstop, bw, ampcal="True", phasecal="True", sgrscat=False):
        """Observe the image with an array object to produce an obsdata object.
	       tstart and tstop should be hrs in GMST.
           tint and tadv should be seconds.
           if sgrscat==True, the visibilites will be blurred by the Sgr A* scattering kernel.
	    """
        
        # Generate empty observation
        obs = array.obsdata(self.ra, self.dec, self.rf, bw, tint, tadv, tstart, tstop)
        
        # Observe
        obs = self.observe_same(obs, ampcal=ampcal, phasecal=phasecal, sgrscat=sgrscat)    
        return obs
        
    def display(self, cfun='afmhot', nvec=20, pcut=0.01, plotvec=True):
        """Display the image with matplotlib
        """
        
        plt.figure()
        plt.clf()
        
        image = self.imvec;
        
        if len(self.qvec) and plotvec:
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
        header['BUINT'] = 'JY/PIXEL'
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
        tdict: The dictionary of telescope tuples (x, y, z, SEFD) where x,y,z are geocentric coordinates.
    """    
    
    def __init__(self, tdict):
        self.tdict = tdict
        
    def listscopes(self):
        """List all telescopes"""
        
        print [i for i in sorted(self.tdict.keys())]
            
    def listbls(self):
        """List all baselines"""
 
        bls = []
        for i1 in sorted(self.tdict.keys()):
            for i2 in sorted(self.tdict.keys()):
                if not ([i1,i2] in bls) and not ([i2,i1] in bls) and i1 != i2:
                    bls.append([i1,i2])
                    
        return np.array(bls)
            
    def obsdata(self, ra, dec, rf, bw, tint, tadv, tstart, tstop):
        """Generate u,v points and baseline errors for the array.
           Return an Observation object.
           tstart and tstop are hrs in GMST
           tint and tadv are seconds.
           rf and bw are Hz
           ra is fractional hours
           dec is fractional degrees
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
        # !AC can this be made faster? 
        scopes = sorted(self.tdict.keys())
        outlist = []
        
        for k in xrange(len(times)):
            time = times[k]
            theta = np.mod((time-ra)*360./24, 360)
            for i1 in scopes:
                for i2 in scopes:
                    if (i1!=i2 and elevcut(earthrot(self.tdict[i1][0],theta),sourcevec) and 
                        elevcut(earthrot(self.tdict[i2][0],theta),sourcevec)):
                        outlist.append(np.array((
                                  time,
                                  tint, # Integration 
                                  i1, # Station 1
                                  i2, # Station 2
                                  elev(earthrot(self.tdict[i1][0],theta),sourcevec), # Station 1 elevation
                                  elev(earthrot(self.tdict[i2][0],theta),sourcevec), # Station 2 elevation
                                  np.dot(earthrot(self.tdict[i1][0]-self.tdict[i2][0],theta)/l, projU), # U (lambda)
                                  np.dot(earthrot(self.tdict[i1][0]-self.tdict[i2][0],theta)/l, projV), # V (lambda)
                                  0.0, 0.0, 0.0, # Stokes I, Q, U visibilities (Jy)
                                  blnoise(self.tdict[i1][1],self.tdict[i2][1], tint, bw) # Sigma (Jy)
                                ),dtype=DTPOL
                                ))

        obs = Obsdata(ra, dec, rf, bw, np.array(outlist))      
        return obs
    
#    def plotbls(self):
#        """Plot all baselines on a globe"""
#        
#        lat = []
#        lon = []
#        for t1 in sorted(self.tdict.keys()):
#            (x,y,z) = self.tdict[t1][0]
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
    
    def __init__(self, ra, dec, rf, bw, datatable, source="SgrA", mjd=48277, ampcal=True, phasecal=True):

        self.source = str(source)
        self.ra = float(ra)
        self.dec = float(dec)
        self.rf = float(rf)
        self.bw = float(bw)
        self.ampcal = bool(ampcal)
        self.phasecal = bool(phasecal)
        
        # Time sort the datatable
        if (datatable.dtype != DTPOL):
            raise Exception("Data table should be a recarray with datatable.dtype = %s" % DTPOL)
        self.data = np.array(sorted(datatable, key=lambda x: x['time']))
        
        times = self.unpack(['time'])['time']
        self.tstart = times[0]
        self.mjd = fracmjd(mjd, self.tstart)
        self.tstop = times[-1]
        if self.tstop < self.tstart: 
            self.tstop += 24.0
    
    def unpack(self, fields):
        """Return a recarray of all the data for the given fields from the data table"""
        
        # Take care of the case where we specify only one field
        if type(fields) == str: fields = [fields]
        
        # dictionary of types
        ty_dict = {f : 'f8' for f in ["u","v","sigma","tint","time","el1","el2",
                                    "amp", "qamp", "uamp", "pamp", "mamp", "uvdist",
                                    "phase", "qphase", "uphase", "pphase", "mphase",
                                    "snr","qsnr","usnr"]}
        ty_dict.update({f : 'a32' for f in ["t1","t2"]})
        ty_dict.update({f : 'c16' for f in ["vis","qvis","uvis","pvis","m"]})
        
        # dictionary of data
        out_dict = {f : self.data[f] for f in ["u","v","sigma","tint","time","el1","el2", 
                                               "t1", "t2", "vis", "qvis", "uvis"]}
        out_dict["uvdist"] = np.abs(out_dict["u"] + 1j*out_dict["v"])
        out_dict["pvis"] = out_dict["qvis"] + 1j * out_dict["uvis"]
        out_dict["m"] = out_dict["pvis"] / out_dict["vis"]
        out_dict["amp"] = np.abs(out_dict["vis"])
        out_dict["qamp"] = np.abs(out_dict["qvis"])
        out_dict["uamp"] = np.abs(out_dict["uvis"])
        out_dict["pamp"] = np.abs(out_dict["pvis"])
        out_dict["mamp"] = np.abs(out_dict["m"])
        out_dict["phase"] = np.angle(out_dict["vis"])/DEGREE
        out_dict["qphase"] = np.angle(out_dict["qvis"])/DEGREE
        out_dict["uphase"] = np.angle(out_dict["uvis"])/DEGREE
        out_dict["pphase"] = np.angle(out_dict["pvis"])/DEGREE
        out_dict["mphase"] = np.angle(out_dict["m"])/DEGREE
        out_dict["snr"] = out_dict["amp"] / out_dict["sigma"]
        out_dict["qsnr"] = out_dict["qamp"] / out_dict["sigma"]
        out_dict["usnr"] = out_dict["uamp"] / out_dict["sigma"]
        
        allout = []
        
        for field in fields:
             if not (field in FIELDS):
                raise Exception("%s is not valid field \n" % field + 
                                "valid field values are " + string.join(FIELDS)) 

             out = out_dict[field]
             ty = ty_dict[field]
             
#            # Get field data
#            if field in ["u","v","sigma","tint","time","elev1","elev2"]: 
#                out = self.data[field]
#                ty = 'f8'
#            elif field in ["t1","t2"]: 
#                out = self.data[field]
#                ty = 'a32'
#            elif field in ["vis","amp","phase","snr"]: 
#                out = self.data['vis']
#                ty = 'c16'
#            elif field in ["qvis","qamp","qphase","qsnr"]: 
#                out = self.data['qvis']
#                ty = 'c16'
#            elif field in ["uvis","uamp","uphase","usnr"]: 
#                out = self.data['uvis']
#                ty = 'c16'
#            elif field in ["pvis","pamp","pphase"]: 
#                out = self.data['qvis'] + 1j * self.data['uvis']
#                ty = 'c16'
#            elif field in ["m","mamp","mphase"]: 
#                out = (self.data['qvis'] + 1j * self.data['uvis']) / self.data['vis']
#                ty = 'c16'
#            elif field in ["uvdist"]: 
#                out = self.data['u'] + 1j * self.data['v']
#                ty = 'c16'
#            else: raise Exception("%s is not valid field \n" % field + 
#                                  "valid field values are " + string.join(FIELDS)) 

#            # Get arg or amps or snr
#            if field in ["amp", "qamp", "uamp", "pamp", "mamp", "uvdist"]: 
#                out = np.abs(out)
#                ty = 'f8'
#            elif field in ["phase", "qphase", "uphase", "pphase", "mphase"]: 
#                out = np.angle(out)/DEGREE
#                ty = 'f8'
#            elif field in ["snr","qsnr","usnr"]:
#                out = np.abs(out)/self.data['sigma']
#                ty = 'f8'
                    
             # Reshape and stack with other fields
             out = np.array(out, dtype=[(field, ty)])
             if len(allout) > 0:
                 allout = rec.merge_arrays((allout, out), asrecarray=True, flatten=True)
             else:
                 allout = out
            
        return allout
    
    def tlist(self):
        """Partition the data into a list of equal time observations"""
        
        # Use itertools groupby function
        datalist = []
        for key, group in it.groupby(self.data, lambda x: x['time']):
            datalist.append(np.array([obs for obs in group]))
        
        return datalist
        
    def bispectra(self, vtype="vis", mode='time'):
        """Return all independent equal time bispectrum values
           Get Q, U, P bispectra by changing vtype
        """
        
        if not mode in ('time', 'all'):
            raise Exception("possible options for mode are 'time' and 'all'")
        
        tlist = self.tlist()    
        outlist = []
        bis = []
        for tdata in tlist:
            time = tdata[0]['time']
            
            # Find all unique triangles and calculate the bispectra
            
            sites = set(np.hstack((tdata['t1'].view(('a32',1)), tdata['t2'].view(('a32',1)))))
            tris = list(it.combinations(sites,3))
            l_dict = {(datapt['t1'],datapt['t2']) : datapt for datapt in tdata}
            for tri in tris:
                # Select triangle points.
                # !AC Correct for the case where we don't have all inverse baselines!
                l1 = l_dict[(tri[0],tri[1])]
                l2 = l_dict[(tri[1],tri[2])]
                l3 = l_dict[(tri[2],tri[0])]
                    
                # Get the appropriate polarization
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
                                     bi, bisig),dtype=DTBIS))                 
            
            # Append all equal time bispectra to outlist    
            if mode=='time' and len(bis) > 0:
                outlist.append(np.array(bis))
                bis = []    
        
        if mode=='all':
            outlist = np.array(bis)
        
        return outlist
   
        
    def c_phases(self, mode='time'):
        """Return all independent equal time closure phase values
        """
           
        #!AC Should I include Q,U,P closure phases? 
        
        bis = self.bispectra(mode=mode)
        if mode == 'all':
            bis.dtype.names = ('time','t1','t2','t3','u1','v1','u2','v2','u3','v3','cphase','sigmacp')
            bis['sigmacp'] = bis['sigmacp']/np.abs(bis['cphase'])/DEGREE
            bis['cphase'] = np.angle(bis['cphase'])/DEGREE
            out = bis.astype(np.dtype(DTCPHASE))
            
        elif mode=='time':
            out = []
            for bi in bis:
                if len(bi) == 0: continue
                bi.dtype.names = ('time','t1','t2','t3','u1','v1','u2','v2','u3','v3','cphase','sigmacp')
                bi['sigmacp'] = bi['sigmacp']/np.abs(bi['cphase'])/DEGREE
                bi['cphase'] = np.angle(bi['cphase'])/DEGREE
                out.append(bi.astype(np.dtype(DTCPHASE)))
        else: 
            raise Exception("possible options for mode are 'time' and 'all'")
        
        return out    
         
    def c_amplitudes(self, mode="time"):
        """Return all independent equal time closure amplitudes""" 
        
        #!AC Add amplitude debiasing! (and elsewhere!)
        
        if not mode in ('time','all'):
            raise Exception("possible options for mode are 'time' and 'all'")
        
        tlist = self.tlist() 
        outlist = []
        cas = []
        for tdata in tlist:
            time = tdata[0]['time']

            # Find all unique quadrangles and calculate the closure amplitudes
            # !AC Can this be made faster?
            sites = set(np.hstack((tdata['t1'].view(('a32',1)), tdata['t2'].view(('a32',1)))))
            quadsets = list(it.combinations(sites,4))
            
            # Loop over quadrangles
            for q in quadsets:
                # Loop over both independent closure amplitudes
                for quad in (q, [q[0],q[3],q[1],q[2]]): 
                
                    # !AC Correct for case where we don't have all the inverse baselines in the array 
                    l1 = [datapt for datapt in tdata if datapt['t1'] == quad[0] and datapt['t2'] == quad[1]]
                    l2 = [datapt for datapt in tdata if datapt['t1'] == quad[2] and datapt['t2'] == quad[3]]
                    l3 = [datapt for datapt in tdata if datapt['t1'] == quad[0] and datapt['t2'] == quad[2]]
                    l4 = [datapt for datapt in tdata if datapt['t1'] == quad[1] and datapt['t2'] == quad[3]]
                    
                    #!AC There should be one entry max in these lists
                    # Check earlier for no duplicate data points? 
                    if len(l1) > 0 and len(l2) > 0 and len(l3) > 0 and len(l4) > 0:
                        l1 = l1[0]
                        l2 = l2[0]
                        l3 = l3[0]
                        l4 = l4[0]
                    else:
                        continue
                    
                    # Compute the closure amplitude and the error
                    camp = np.abs((l1['vis']*l2['vis'])/(l3['vis']*l4['vis']))
                    camperr = camp * np.sqrt((l1['sigma']/np.abs(l1['vis']))**2 +  
                                             (l2['sigma']/np.abs(l2['vis']))**2 + 
                                             (l3['sigma']/np.abs(l3['vis']))**2 +
                                             (l4['sigma']/np.abs(l4['vis']))**2)
                                             
                                            
                    # Add the closure amplitudes to the equal-time list         
                    cas.append(np.array((time, quad[0], quad[1], quad[2], quad[3],
                                         l1['u'], l1['v'], l2['u'], l2['v'], 
                                         l3['u'], l3['v'], l4['u'], l4['v'],
                                         camp, camperr),dtype=DTCAMP))                 
            
            # Append all equal time closure amps to outlist    
            if mode=='time':
                outlist.append(np.array(cas))
                cas = []    
            elif mode=='all':
                outlist = np.array(cas)
        
        return outlist
        
    def plotall(self, field1, field2, rangex=False, rangey=False):
        """Make a scatter plot of 2 real observation fields with errors"""
        
        # Determine if fields are valid
        if (field1 not in FIELDS) and (field2 not in FIELDS):
            raise Exception("valid fields are " + string.join(FIELDS))
                              
        # Unpack x and y axis data
        data = self.unpack([field1,field2])
        
        # X error bars
        if field1 in ['amp', 'qamp', 'uamp']:
            sigx = self.unpack('sigma')['sigma']
        elif field1 in ['phase', 'uphase', 'qphase']:
            sigx = (self.unpack('sigma')['sigma'])/(self.unpack(['amp'])['amp'])/DEGREE
        elif field1 == 'pamp':
            sigx = np.sqrt(2) * self.unpack('sigma')['sigma']
        elif field1 == 'pphase':
            sigx = np.sqrt(2) * (self.unpack('sigma')['sigma'])/(self.unpack('pamp')['pamp'])/DEGREE
        elif field1 == 'mamp':
            sigx = merr(self.unpack('sigma')['sigma'], self.unpack('amp')['amp'], self.unpack('mamp')['mamp'])
        elif field1 == 'mphase':
            sigx = merr(self.unpack('sigma')['sigma'], self.unpack('amp')['amp'], self.unpack('mamp')['mamp']) / self.unpack('mamp')['mamp']
        else:
            sigx = None
            
        # Y error bars
        if field2 in ['amp', 'qamp', 'uamp']:
            sigy = self.unpack('sigma')['sigma']
        elif field2 in ['phase', 'uphase', 'qphase']:
            sigy = (self.unpack('sigma')['sigma'])/(self.unpack(['amp'])['amp'])/DEGREE
        elif field2 == 'pamp':
            sigy = np.sqrt(2) * self.unpack('sigma')['sigma']
        elif field2 == 'pphase':
            sigy = np.sqrt(2) * (self.unpack('sigma')['sigma'])/(self.unpack('pamp')['pamp'])/DEGREE
        elif field2 == 'mamp':
            sigy = merr(self.unpack('sigma')['sigma'], self.unpack('amp')['amp'], self.unpack('mamp')['mamp'])
        elif field2 == 'mphase':
            sigy = merr(self.unpack('sigma')['sigma'], self.unpack('amp')['amp'], self.unpack('mamp')['mamp']) / self.unpack('mamp')['mamp']
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
        for obs in self.data:
            if (obs['t1'], obs['t2']) == (site1, site2):
                time = obs['time']
                if field == 'uvdist':
                    plotdata.append([time, np.abs(obs['u'] + 1j*obs['v']), obs['sigma']])
                
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
                    plotdata.append([time, np.angle(obs['qvis'] + 1j*obs['uvis'])/DEGREE, np.sqrt(2)*obs['sigma']/np.abs(obs['qvis'] + 1j*obs['uvis'])/DEGREE])
                
                elif field == 'mamp':
                    plotdata.append([time, np.abs((obs['qvis'] + 1j*obs['uvis'])/obs['vis']), merr(obs['sigma'], obs['vis'], (obs['qvis']+1j*obs['uvis'])/obs['vis'])])
                
                elif field == 'mphase':
                    plotdata.append([time, np.angle((obs['qvis'] + 1j*obs['uvis'])/obs['vis'])/DEGREE, 
                                    merr(obs['sigma'], obs['vis'], (obs['qvis']+1j*obs['uvis'])/obs['vis'])/np.abs((obs['qvis']+1j*obs['uvis'])/obs['vis'])/DEGREE])
                            
                elif field == 'time':
                    plotdata.append([time, time, 0])
                
                else:
                    plotdata.append([time, obs[field], 0])
                
                continue
                                
        plotdata = np.array(plotdata)
    
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
        plt.ylabel(field)
        plt.title('%s - %s'%(site1,site2))
        plt.show(block=False)    
                
    def plot_cphase(self, site1, site2, site3, rangey=False):
        """Plot closure phase over time on a triangle"""
        
        # Get closure phases
        tri = (site1, site2, site3)
        cphases = self.c_phases()
        plotdata = []
        for entry in cphases:
            for obs in entry:
                obstri = (obs['t1'],obs['t2'],obs['t3'])
                if set(obstri) == set(tri):
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
           (1-2)(3-4)/(1-3)(2-4)
        """
        # !AC find a comprehensive way to assemble requested closure amplitude
        # from the 2 independent ones we may have. 
        
        # Get the closure amplitudes
        quad = (site1, site2, site3, site4)
        camps = self.c_amplitudes()
        plotdata = []
        for entry in camps:
            for obs in entry:
                obsquad = (obs['t1'],obs['t2'],obs['t3'],obs['t4'])
                if quad == obsquad:
                    plotdata.append([obs['time'], obs['camp'], obs['sigmaca']])
                    continue
                
                # Check reversing labels 1-3, 2-4
                obsquad = (obs['t3'],obs['t4'],obs['t1'],obs['t2'])
                if quad == obsquad:
                    plotdata.append([obs['time'], obs['camp'], obs['sigmaca']])
                    continue
                
                # !AC and other combinations...??
                    
        plotdata = np.array(plotdata)
        if len(plotdata) == 0: 
            print "No closure amplitudes on this quadrangle!"
            print "Try again with a different scope order?"
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
                                                           site1,site3,site2,site4))
        plt.show(block=False)       

    def save_txt(self, fname):
        """Save visibility data to a text file"""
        
        # Get the necessary data and the header
        outdata = self.unpack(['time', 'tint', 't1', 't2', 'el1', 'el2', 'u', 'v', 'amp', 'phase', 'qamp', 'qphase', 'uamp', 'uphase', 'sigma'])
        head = ("SRC: %s \n" % self.source +
                    "RA: " + rastring(self.ra) + "\n" + "DEC: " + decstring(self.dec) + "\n" +
                    "MJD: %.4f - %.4f \n" % (fracmjd(self.mjd,self.tstart), fracmjd(self.mjd,self.tstop)) + 
                    "RF: %.4f GHz \n" % (self.rf/1e9) + 
                    "BW: %.4f GHz \n" % (self.bw/1e9) +
                    "PHASECAL: %i \n" % self.phasecal + 
                    "AMPCAL: %i \n" % self.ampcal + 
                    "-------------------------------------------------------------------------------------------------------------------------------------------------\n" +
                    "time (hr) tint    T1     T2    Elev1   Elev2   U (lambda)       V (lambda)         Iamp (Jy)    Iphase(d)  Qamp (Jy)    Qphase(d)   Uamp (Jy)    Uphase(d)   sigma (Jy)")
        
        # Format and save the data
        fmts = "%011.8f %4.2f %6s %6s  %4.2f   %4.2f  %16.4f %16.4f    %10.8f %10.4f   %10.8f %10.4f    %10.8f %10.4f    %10.8f"
        np.savetxt(fname, outdata, header=head, fmt=fmts)
        return

    def save_uvfits(self, fname):
        # !AC ELEVATION
        """Save visibility data to uvfits
           Needs template.UVP file
           Antenna table is currently incorrect"""
        
        # Template UVFITS
        hdulist = fits.open('./template.UVP')
        
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
        obsdata = self.unpack(['time','tint','u','v','vis','qvis','uvis','sigma','t1','t2','el1','el2'])
        ndat = len(obsdata['time'])
        
        # times and tints
        jds = (self.mjd + 2400000.5) + (obsdata['time'] / 24.0) 
        tints = obsdata['tint']
        
        # Baselines
        # First convert scopes to numbers
        scopes = list(set(np.hstack((obsdata['t1'], obsdata['t2']))))
        scopes.sort()
        tdict = {}
        i = 1
        for scope in scopes:
            tdict[scope] = i
            i += 1
            
        # !AC These HAVE to be correct for CLEAN to work. Why?
        t1 = [tdict[scope] for scope in obsdata['t1']]
        t2 = [tdict[scope] for scope in obsdata['t2']]
        bl = 256*np.array(t1) + np.array(t2)
        
        # elevations
        el1 = obsdata['el1']
        el2 = obsdata['el2']
        
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
        x = fits.GroupData(outdat, parnames=['UU---SIN', 'VV---SIN', 'WW---SIN', 'BASELINE', 'DATE', '_DATE', 'INTTIM', 'ELEV1', 'ELEV2'], 
                           pardata=[u, v, np.zeros(ndat), bl, jds, np.zeros(ndat), tints, el1, el2], bitpix=-32)
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
    tdict = {}
    if tdata.shape[1] != 5:
        raise Exception("Array file should have format: (name, x, y, z, SEFD)") 
    for row in tdata:
        tdict[row[0]] = (np.array((row[1], row[2], row[3]), dtype=float), float(row[4]))
    
    return Array(tdict)
    
def load_challenge_obs(filename):
    """Read an observation from an imaging challenge text file and return an Obsdata object
    """
    
    src = "unknown"
    ra = 0
    dec = 0
    mjd = 48277
    rf = 230*1e9
    bw = 2e9
    phasecal = False
    ampcal = False
    
    # Load the data, convert to list format, return object
    datatable = np.loadtxt(filename, dtype=str)
    datatable2 = []
    for row in datatable:
        time = float(row[0])
        tint = 30 #?
        t1 = row[1]
        t2 = row[2]
        el1 = float(row[3])
        el2 = float(row[4])
        u = float(row[5])
        v = float(row[6])
        vis = float(row[7]) + 1j*float(row[8])
        qvis = 0
        uvis = 0
        sigma = float(row[9])
        datatable2.append(np.array((time, tint, t1, t2, el1, el2, u, v, vis, qvis, uvis, sigma), dtype=DTPOL))
                         
    datatable2 = np.array(datatable2)

    return Obsdata(ra, dec, rf, bw, datatable2, source=src, mjd=mjd, ampcal=ampcal, phasecal=phasecal)        
    
def load_obs_txt(filename):
    """Read an observation from a text file and return an Obsdata object
       text file has the same format as output from Obsdata.savedata()
    """
    
    # Read the header
    file = open(filename)
    src = string.join(file.readline().split()[2:])
    ra = file.readline().split()
    ra = float(ra[2]) + float(ra[4])/60. + float(ra[6])/3600.
    dec = file.readline().split()
    dec = float(dec[2]) + float(dec[4])/60. + float(dec[6])/3600.
    mjd = float(file.readline().split()[2])
    rf = float(file.readline().split()[2]) * 1e9
    bw = float(file.readline().split()[2]) * 1e9
    phasecal = bool(file.readline().split()[2])
    ampcal = bool(file.readline().split()[2])
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
        u = float(row[6])
        v = float(row[7])
        vis = float(row[8]) * np.exp(1j * float(row[9]) * DEGREE)
        if datatable.shape[1] == 15:
            qvis = float(row[10]) * np.exp(1j * float(row[11]) * DEGREE)
            uvis = float(row[12]) * np.exp(1j * float(row[13]) * DEGREE)
            sigma = float(row[14])
        elif datatable.shape[1] == 11:
            qvis = 0+0j
            uvis = 0+0j
            sigma = float(row[10])
        else:
            raise Exception('Text file does not have the right number of fields!')
            
        datatable2.append(np.array((time, tint, t1, t2, el1, el2, u, v, vis, qvis, uvis, sigma), dtype=DTPOL))
                         
    datatable2 = np.array(datatable2)

    return Obsdata(ra, dec, rf, bw, datatable2, source=src, mjd=mjd, ampcal=ampcal, phasecal=phasecal)        

def load_obs_uvfits(filename):
    """Load uvfits data from a BU Blazar datafile. 
    """
    
    hdulist = fits.open(filename)
    header = hdulist[0].header
    data = hdulist[0].data
    
    # Load various header parameters
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
    
    # Times
    jds = data['DATE'][mask]
    mjd = int(jdtomjd(jds[0]))
    times = np.array([mjdtogmt(jdtomjd(jd)) for jd in jds])
    tints = data['INTTIM'][mask]
    
    # Scopes
    t1 = data['BASELINE'][mask].astype(int)/256
    t2 = data['BASELINE'][mask].astype(int) - t1*256
    
    # Elevations
    try: 
        el1 = data['ELEV1'][mask]
        el2 = data['ELEV2'][mask]
    except KeyError:
        el1 = el2 = np.zeros(len(t1))
        
    # Convert uv in lightsec to lambda by multiplying by rf
    u = data['UU---SIN'][mask] * rf
    v = data['VV---SIN'][mask] * rf    
    
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
    
    # Include the negative baselines
    # !AC First test to see if if the negative bls are already in there!
    times_d = np.array([[time, time] for time in times]).flatten()
    t1_d = np.array([[t1[i], t2[i]] for i in range(len(t1))]).flatten()
    t2_d = np.array([[t2[i], t1[i]] for i in range(len(t1))]).flatten()
    el1_d = np.array([[el1[i], el2[i]] for i in range(len(t1))]).flatten()
    el2_d = np.array([[el2[i], el1[i]] for i in range(len(t1))]).flatten()
    tints_d = np.array([[tint, tint] for tint in tints]).flatten()
    u_d = np.array([[-uu, uu] for uu in u]).flatten()
    v_d = np.array([[-vv, vv] for vv in v]).flatten()
    ivis_d = np.array([[i, np.conj(i)] for i in ivis]).flatten() # !AC Reverse the phases ??
    qvis_d = np.array([[q, np.conj(q)] for q in qvis]).flatten() # ??
    uvis_d = np.array([[u, np.conj(u)] for u in uvis]).flatten() # ??
    sigma_d = np.array([[s, s] for s in sigma]).flatten()
    
    
    # Make a datatable
    # !AC Can I make this faster?
    datatable = []
    for i in xrange(len(times_d)):
        datatable.append(np.array((
                           times_d[i], tints_d[i], t1_d[i], t2_d[i], el1_d[i], el2_d[i],
                           u_d[i], v_d[i],
                           ivis_d[i], qvis_d[i], uvis_d[i], sigma_d[i]
                           ), dtype=DTPOL
                         ))
    datatable = np.array(datatable)
    
    return Obsdata(ra, dec, rf, bw, datatable, source=src, mjd=mjd, ampcal=True, phasecal=True)

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
    dec = float(dec[2]) + float(dec[4])/60. + float(dec[6])/3600.
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
    
def load_im_fits(filename):
    """Read in an image from a FITS file and create an Image object
    """
    
    # Open the FITS file
    hdulist = fits.open(filename)
    
    # Assume stokes I is the primary hdu
    header = hdulist[0].header
    
    # Read some header values
    ra = header['OBSRA']*12/180.
    dec = header['OBSDEC']
    xdim_p = header['NAXIS1']
    psize_x = np.abs(header['CDELT1'])
    dim_p = header['NAXIS2']
    psize_y = np.abs(header['CDELT2'])
    
    if 'MJD' in header.keys(): mjd = header['MJD']
    else: mjd = 48277.0 
    
    if 'FREQ' in header.keys(): rf = header['FREQ']
    else: rf = 230e9
    
    if 'OBJECT' in header.keys(): src = header['OBJECT']
    else: src = 'SgrA'
    
    # Get the image and create the object
    image = hdulist[0].data[::-1,:] # flip y-axis!
    outim = Image(image, psize_x, ra, dec, rf=rf, source=src, mjd=mjd)
    
    # Look for Stokes Q and U
    qimage = uimage = np.array([])
    for hdu in hdulist[1:]:
        header = hdu.header
        if 'STOKES' in header.keys() and header['STOKES'] == 'Q':
            qimage = hdu.data[::-1,:] # flip y-axis!
        if 'STOKES' in header.keys() and header['STOKES'] == 'U':
            uimage = hdu.data[::-1,:] # flip y-axis!
    if qimage.shape == uimage.shape == image.shape:
        print 'Loaded Stokes I, Q, and U images'
        outim.add_qu(qimage, uimage)
    else:
        print 'Loaded Stokes I image only'
                
    return outim
    
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
    
    obsdeblur = Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, datatable)
    return obsdeblur
    
def sgra_kernel_uv(rf, u, v):
    """Return the value of the Sgr A* scattering kernel at a given u,v pt (in lambda), 
       at a given frequency rf (in Hz).
       Values form Bower et al.
    """
    
    lcm = (C/rf) * 100 # in cm
    sigma_maj = FWHM_MAJ * (lcm**2) / (2*np.sqrt(2*np.log(2))) * RADPERUAS
    sigma_min = FWHM_MIN * (lcm**2) / (2*np.sqrt(2*np.log(2))) * RADPERUAS
    theta = POS_ANG
    
    
    # Covarience matrix
    a = (sigma_min * np.cos(theta))**2 + (sigma_maj*np.sin(theta))**2
    b = (sigma_maj * np.cos(theta))**2 + (sigma_min*np.sin(theta))**2
    c = (sigma_min**2 - sigma_maj**2) * np.cos(theta) * np.sin(theta)
    m = np.array([[a, c], [c, b]])
    uv = np.array([u,v])
    
    
    x2 = np.dot(uv, np.dot(m, uv))
    g = np.exp(-2 * np.pi**2 * x2)
    
    return g
                                 
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
    
    return (mjd - int(mjd)) * 24
    
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

   
