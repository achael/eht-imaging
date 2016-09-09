# movie_utils.py
# Andrew Chael, 09/06/2016

import string
import numpy as np
import numpy.lib.recfunctions as rec
import matplotlib.pyplot as plt
import scipy.signal
import scipy.optimize
import itertools as it
import astropy.io.fits as fits
import datetime
import writeData
import oifits_new as oifits
import time as ttime
import pulses
import vlbi_imaging_utils as vb
#from mpl_toolkits.basemap import Basemap # for plotting baselines on globe


##################################################################################################
# Constants
##################################################################################################
EP = 1.0e-10
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
          
class Movie(object):
    """A list of image arrays (in Jy/pixel).
    
    Attributes:
    	pulse: The function convolved with pixel value dirac comb for continuous image rep. (function from pulses.py)
    	framedur: The frame duration (sec)
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
    
    def __init__(self, movie, framedur, psize, ra, dec, rf=230e9, pulse=pulses.trianglePulse2D, source="SgrA", mjd="0"):
        if len(movie[0].shape) != 2: 
            raise Exception("image must be a 2D numpy array") 
        
        self.framedur = float(framedur)
        self.pulse = pulse       
        self.psize = float(psize)
        # !AC Check on consistency here
        self.xdim = movie[0].shape[1]
        self.ydim = movie[0].shape[0]
        
        self.ra = float(ra) 
        self.dec = float(dec)
        self.rf = float(rf)
        self.source = str(source)
        self.mjd = float(mjd)
        
        #the list of frames
        self.frames = [image.flatten() for image in movie] 
        self.qframes = []
        self.uframes = []
        
    def add_qu(self, qmovie, umovie):
        """Add Q and U movies.
        """
        
        if not(len(qmovie) == len(umovie) == len(self.frames)):
            raise Exception("Q & U movies must have same length as I movie!")
        
        self.qframes = [0 for i in xrange(len(self.frames))]
        self.uframes = [0 for i in xrange(len(self.frames))]
        
        for i in xrange(len(self.frames)): 
            qimage = qmovie[i]
            uimage = umovie[i]
            if len(qimage.shape) != len(uimage.shape):
                raise Exception("image must be a 2D numpy array")
            if qimage.shape != uimage.shape != (self.ydim, self.xdim):
                raise Exception("Q & U image shapes incompatible with I image!") 
            self.qframes[i] = qimage.flatten()
            self.uframes[i] = uimage.flatten()
    
    def copy(self):
        """Copy the Movie object"""
        new = Movie([imvec.reshape(self.ydim,self.xdim) for imvec in self.frames], self.psize, self.framedur, self.ra, self.dec, rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)
        newim.add_qu([qvec.reshape(self.ydim,self.xdim) for qvec in self.qframes], [uvec.reshape(self.ydim,self.xdim) for uvec in self.uframes])
        return newim
        
    def flip_chi(self):
        """Change between different conventions for measuring position angle (East of North vs up from x axis).
        """
        self.qframes = [-qvec for qvec in self.qframes]
        return
           
    def observe_same(self, obs, sgrscat=False, repeat=False):
        """Observe the movie on the same baselines as an existing observation object
           if sgrscat==True, the visibilites will be blurred by the Sgr A* scattering kernel
           Does NOT add noise
        """
        
        # Check for agreement in coordinates and frequency 
        if (self.ra!= obs.ra) or (self.dec != obs.dec):
            raise Exception("Image coordinates are not the same as observtion coordinates!")
	    if (self.rf != obs.rf):
	        raise Exception("Image frequency is not the same as observation frequency!")
        
        mjdstart = self.mjd
        mjdend = self.mjd + (len(self.frames)*self.framedur) / 86400.0 #!AC use astropy date conversion
        
        # Get data
        obslist = obs.tlist()
        
        # Observation MJDs in range?
        # !AC use astropy date conversion!!
        obsmjds = np.array([(np.floor(obs.mjd) + (obsdata[0]['time'])/24.0) for obsdata in obslist])

        if (not repeat) and ((obsmjds < mjdstart) + (obsmjds > mjdend)).any():
            raise Exception("Obs times outside of movie range of MJD %f - %f" % (mjdstart, mjdend))
                   
        # Observe nearest frame
        obsdata_out = []
        for i in xrange(len(obslist)):
            obsdata = obslist[i]
            
            # Frame number
            mjd = obsmjds[i]
            n = int(np.floor((mjd - mjdstart) * 86400.0 / self.framedur))
            
            if (n >= len(self.frames)):
                if repeat: n = np.mod(n, len(self.frames))
                else: raise Exception("Obs times outside of movie range of MJD %f - %f" % (mjdstart, mjdend))
                
            # Extract uv data & perform DFT
            uv = obsdata[['u','v']].view(('f8',2))
            mat = vb.ftmatrix(self.psize, self.xdim, self.ydim, uv, pulse=self.pulse)
            vis = np.dot(mat, self.frames[n])
        
            # If there are polarized images, observe them:
            qvis = np.zeros(len(vis))
            uvis = np.zeros(len(vis))
            if len(self.qframes):
                qvis = np.dot(mat, self.qframes[n])
                uvis = np.dot(mat, self.uframes[n])
        
            # Scatter the visibilities with the SgrA* kernel
            if sgrscat:
                for i in range(len(vis)):
                    ker = sgra_kernel_uv(self.rf, uv[i,0], uv[i,1])
                    vis[i]  *= ker
                    qvis[i] *= ker
                    uvis[i] *= ker
   
            # Put the visibilities back in the obsdata array
            obsdata['vis'] = vis
            obsdata['qvis'] = qvis
            obsdata['uvis'] = uvis
            
            if len(obsdata_out):
                obsdata_out = np.hstack((obsdata_out, obsdata))
            else:
                obsdata_out = obsdata
                
        # Return observation object
        obs_no_noise = vb.Obsdata(self.ra, self.dec, self.rf, obs.bw, obsdata_out, obs.tarr, source=self.source, mjd=self.mjd)
        return obs_no_noise
        
    def observe(self, array, tint, tadv, tstart, tstop, bw, tau=TAUDEF, gainp=GAINPDEF, opacity_errs=True, ampcal=True, phasecal=True, sgrscat=False, repeat=False):
        """Observe the image with an array object to produce an obsdata object.
	       tstart and tstop should be hrs in GMST.
           tint and tadv should be seconds.
           tau is the estimated optical depth. This can be a single number or a dictionary giving one tau per site
           if sgrscat==True, the visibilites will be blurred by the Sgr A* scattering kernel at the appropriate frequency
	    """
        
        # Generate empty observation
        obs = array.obsdata(self.ra, self.dec, self.rf, bw, tint, tadv, tstart, tstop, tau=tau, opacity_errs=opacity_errs, mjd=self.mjd)
        
        # Observe
        obs = self.observe_same(obs, sgrscat=sgrscat, repeat=repeat)    
        
        # Add noise
        obs = vb.add_noise(obs, opacity_errs=opacity_errs, ampcal=ampcal, phasecal=phasecal, gainp=gainp)
        
        return obs
        
#    def display(self, cfun='afmhot', nvec=20, pcut=0.01, plotp=False, interp='nearest'):
#        """Display the image with matplotlib
#        """
#        
#        if (interp in ['gauss', 'gaussian', 'Gaussian', 'Gauss']):
#            interp = 'gaussian'
#        else:
#            interp = 'nearest'
#            
#        plt.figure()
#        plt.clf()
#        
#        image = self.imvec;
#        
#        if len(self.qvec) and plotp:
#            thin = self.xdim/nvec 
#            mask = (self.imvec).reshape(self.ydim, self.xdim) > pcut * np.max(self.imvec)
#            mask2 = mask[::thin, ::thin]
#            x = (np.array([[i for i in range(self.xdim)] for j in range(self.ydim)])[::thin, ::thin])[mask2]
#            y = (np.array([[j for i in range(self.xdim)] for j in range(self.ydim)])[::thin, ::thin])[mask2]
#            a = (-np.sin(np.angle(self.qvec+1j*self.uvec)/2).reshape(self.ydim, self.xdim)[::thin, ::thin])[mask2]
#            b = (np.cos(np.angle(self.qvec+1j*self.uvec)/2).reshape(self.ydim, self.xdim)[::thin, ::thin])[mask2]

#            m = (np.abs(self.qvec + 1j*self.uvec)/self.imvec).reshape(self.ydim, self.xdim)
#            m[-mask] = 0
#            
#            plt.suptitle('%s   MJD %i  %.2f GHz' % (self.source, self.mjd, self.rf/1e9), fontsize=20)
#            
#            # Stokes I plot
#            plt.subplot(121)
#            im = plt.imshow(image.reshape(self.ydim, self.xdim), cmap=plt.get_cmap(cfun), interpolation=interp)
#            plt.colorbar(im, fraction=0.046, pad=0.04, label='Jy/pixel')
#            plt.quiver(x, y, a, b,
#                       headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
#                       width=.01*self.xdim, units='x', pivot='mid', color='k', angles='uv', scale=1.0/thin)
#            plt.quiver(x, y, a, b,
#                       headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
#                       width=.005*self.xdim, units='x', pivot='mid', color='w', angles='uv', scale=1.1/thin)

#            xticks = ticks(self.xdim, self.psize/RADPERAS/1e-6)
#            yticks = ticks(self.ydim, self.psize/RADPERAS/1e-6)
#            plt.xticks(xticks[0], xticks[1])
#            plt.yticks(yticks[0], yticks[1])
#            plt.xlabel('Relative RA ($\mu$as)')
#            plt.ylabel('Relative Dec ($\mu$as)')
#            plt.title('Stokes I')
#        
#            # m plot
#            plt.subplot(122)
#            im = plt.imshow(m, cmap=plt.get_cmap('winter'), interpolation=interp, vmin=0, vmax=1)
#            plt.colorbar(im, fraction=0.046, pad=0.04, label='|m|')
#            plt.quiver(x, y, a, b,
#                   headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
#                   width=.01*self.xdim, units='x', pivot='mid', color='k', angles='uv', scale=1.0/thin)
#            plt.quiver(x, y, a, b,
#                   headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
#                   width=.005*self.xdim, units='x', pivot='mid', color='w', angles='uv', scale=1.1/thin)
#            plt.xticks(xticks[0], xticks[1])
#            plt.yticks(yticks[0], yticks[1])
#            plt.xlabel('Relative RA ($\mu$as)')
#            plt.ylabel('Relative Dec ($\mu$as)')
#            plt.title('m (above %0.2f max flux)' % pcut)
#        
#        else:
#            plt.subplot(111)    
#            plt.title('%s   MJD %i  %.2f GHz' % (self.source, self.mjd, self.rf/1e9), fontsize=20)
#            
#            im = plt.imshow(image.reshape(self.ydim,self.xdim), cmap=plt.get_cmap(cfun), interpolation=interp)
#            plt.colorbar(im, fraction=0.046, pad=0.04, label='Jy/pixel')
#            xticks = ticks(self.xdim, self.psize/RADPERAS/1e-6)
#            yticks = ticks(self.ydim, self.psize/RADPERAS/1e-6)
#            plt.xticks(xticks[0], xticks[1])
#            plt.yticks(yticks[0], yticks[1])
#            plt.xlabel('Relative RA ($\mu$as)')
#            plt.ylabel('Relative Dec ($\mu$as)')   
#        
#        plt.show(block=False)
            
    def save_txt(self, fname):
        """Save image data to text files"""
        
        # Coordinate values
        pdimas = self.psize/RADPERAS
        xs = np.array([[j for j in range(self.xdim)] for i in range(self.ydim)]).reshape(self.xdim*self.ydim,1)
        xs = pdimas * (xs[::-1] - self.xdim/2.0)
        ys = np.array([[i for j in range(self.xdim)] for i in range(self.ydim)]).reshape(self.xdim*self.ydim,1)
        ys = pdimas * (ys[::-1] - self.xdim/2.0)
        
        for i in xrange(len(self.frames)):
            fname_frame = fname + "%05d" % i
            # Data
            if len(self.qframes):
                outdata = np.hstack((xs, ys, (self.frames[i]).reshape(self.xdim*self.ydim, 1),
                                             (self.qframes[i]).reshape(self.xdim*self.ydim, 1),
                                             (self.uframes[i]).reshape(self.xdim*self.ydim, 1)))
                hf = "x (as)     y (as)       I (Jy/pixel)  Q (Jy/pixel)  U (Jy/pixel)"

                fmts = "%10.10f %10.10f %10.10f %10.10f %10.10f"
            else:
                outdata = np.hstack((xs, ys, (self.frames[i]).reshape(self.xdim*self.ydim, 1)))
                hf = "x (as)     y (as)       I (Jy/pixel)"
                fmts = "%10.10f %10.10f %10.10f"
     
            # Header
            head = ("SRC: %s \n" % self.source +
                        "RA: " + rastring(self.ra) + "\n" + "DEC: " + decstring(self.dec) + "\n" +
                        "MJD: %.4f \n" % (self.mjd + i*self.framedur/86400.0) + #!AC astropy date conversions!! 
                        "RF: %.4f GHz \n" % (self.rf/1e9) + 
                        "FOVX: %i pix %f as \n" % (self.xdim, pdimas * self.xdim) +
                        "FOVY: %i pix %f as \n" % (self.ydim, pdimas * self.ydim) +
                        "------------------------------------\n" + hf)
             
            # Save
            np.savetxt(fname_frame, outdata, header=head, fmt=fmts)

#    def save_fits(self, fname):
#        """Save image data to FITS file"""
#                
#        # Create header and fill in some values
#        header = fits.Header()
#        header['OBJECT'] = self.source
#        header['CTYPE1'] = 'RA---SIN'
#        header['CTYPE2'] = 'DEC--SIN'
#        header['CDELT1'] = -self.psize/DEGREE
#        header['CDELT2'] = self.psize/DEGREE
#        header['OBSRA'] = self.ra * 180/12.
#        header['OBSDEC'] = self.dec
#        header['FREQ'] = self.rf
#        header['MJD'] = self.mjd
#        header['TELESCOP'] = 'VLBI'
#        header['BUNIT'] = 'JY/PIXEL'
#        header['STOKES'] = 'I'
#        
#        # Create the fits image
#        image = np.reshape(self.imvec,(self.ydim,self.xdim))[::-1,:] #flip y axis!
#        hdu = fits.PrimaryHDU(image, header=header)
#        if len(self.qvec):
#            qimage = np.reshape(self.qvec,(self.xdim,self.ydim))[::-1,:]
#            uimage = np.reshape(self.uvec,(self.xdim,self.ydim))[::-1,:]
#            header['STOKES'] = 'Q'
#            hduq = fits.ImageHDU(qimage, name='Q', header=header)
#            header['STOKES'] = 'U'
#            hduu = fits.ImageHDU(uimage, name='U', header=header)
#            hdulist = fits.HDUList([hdu, hduq, hduu])
#        else: hdulist = fits.HDUList([hdu])
#      
#        # Save fits 
#        hdulist.writeto(fname, clobber=True)
#        
#        return

#!AC this needs a lot of work
#!AC think about how to do the filenames
def load_movie_txt(basename, nframes, framedur, pulse=pulses.trianglePulse2D):
    """Read in a movie from text files and create a Movie object
       Text files should be filename + 00001, etc. 
       Text files should have the same format as output from Image.save_txt()
       Make sure the header has exactly the same form!
    """
    
    framelist = []
    qlist = []
    ulist = []
    for i in xrange(nframes):
        filename = basename + "%05d" % i
        
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
        if i == 0:
            src0 = src
            ra0 = ra
            dec0 = dec
            mjd0 = mjd
            rf0 = rf
            xdim0 = xdim
            ydim0 = ydim
            psize0 = psize_x
        else:
            pass
            #some checks on header consistency
            
        # Load the data, convert to list format, make object
        datatable = np.loadtxt(filename, dtype=float)
        image = datatable[:,2].reshape(ydim_p, xdim_p)
        framelist.append(image)
        
        # Look for Stokes Q and U
        qimage = uimage = np.zeros(image.shape)
        if datatable.shape[1] == 5:
            qimage = datatable[:,3].reshape(ydim_p, xdim_p)
            uimage = datatable[:,4].reshape(ydim_p, xdim_p)
            qlist.append(qimage)
            ulist.append(uimage)
    
    out_mov = Movie(framelist, framedur, psize0, ra0, dec0, rf=rf0, source=src0, mjd=mjd0, pulse=pulse)
    
    if len(qlist):
        print 'Loaded Stokes I, Q, and U movies'
        out_mov.add_qu(qlist, ulist)
    else:
        print 'Loaded Stokes I movie only'
    
    return out_mov
