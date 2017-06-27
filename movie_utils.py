# movie_utils.py
# Andrew Chael, 09/06/2016

#TODO add jones matrices to observe

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
HOUR = (180./12.)*DEGREE
RADPERAS = DEGREE/3600
RADPERUAS = RADPERAS/1e6

# Default Parameters
SOURCE_DEFAULT = "SgrA"
RA_DEFAULT = 17.761122472222223
DEC_DEFAULT = -28.992189444444445
RF_DEFAULT = 230e9
MJD_DEFAULT = 51544
PULSE_DEFAULT = pulses.trianglePulse2D

# Default Optical Depth and std. dev % on gain
TAUDEF = 0.1
GAINPDEF = 0.1
DTERMPDEF = 0.1 # rms amplitude of D-terms if not specified in array file
DTERMPDEF_RESID = 0.01 # rms *residual* amplitude of D-terms (random, unknown contribution)

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
    	mjd: The starting integer mjd of the image 
        start_hr: Fractional start hour of the movie [default = -1.0, meaning it is inherited from the observation]
    """
    
    def __init__(self, movie, framedur, psize, ra, dec, rf=RF_DEFAULT, pulse=PULSE_DEFAULT, source=SOURCE_DEFAULT, mjd=MJD_DEFAULT, start_hr=0.0):
        if len(movie[0].shape) != 2: 
            raise Exception("image must be a 2D numpy array") 
        
        self.framedur = float(framedur)
        self.pulse = pulse       
        self.psize = float(psize)
        self.xdim = movie[0].shape[1]
        self.ydim = movie[0].shape[0]
        
        self.ra = float(ra) 
        self.dec = float(dec)
        self.rf = float(rf)
        self.source = str(source)
        self.mjd = int(mjd)
        self.start_hr = float(start_hr)
        
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
        new = Movie([imvec.reshape(self.ydim,self.xdim) for imvec in self.frames], 
                     self.framedur, self.psize, self.ra, self.dec, rf=self.rf, 
                     source=self.source, mjd=self.mjd, start_hr=self.start_hr, pulse=self.pulse)
        
        if len(self.qframes):
            new.add_qu([qvec.reshape(self.ydim,self.xdim) for qvec in self.qframes], 
                         [uvec.reshape(self.ydim,self.xdim) for uvec in self.uframes])

        return new
        
    def flip_chi(self):
        """Change between different conventions for measuring position angle (East of North vs up from x axis).
        """
        self.qframes = [-qvec for qvec in self.qframes]
        return
           
    def observe_same_nonoise(self, obs, sgrscat=False, repeat=False):
        """Observe the movie on the same baselines as an existing observation object
           if sgrscat==True, the visibilites will be blurred by the Sgr A* scattering kernel
           Does NOT add noise
        """
        
        # Check for agreement in coordinates and frequency 
        if (self.ra!= obs.ra) or (self.dec != obs.dec):
            raise Exception("Image coordinates are not the same as observation coordinates!")
	    if (self.rf != obs.rf):
	        raise Exception("Image frequency is not the same as observation frequency!")
        
        mjdstart = float(self.mjd) + float(self.start_hr/24.0)
        mjdend = mjdstart + (len(self.frames)*self.framedur) / 86400.0 
        
        # Get data
        obslist = obs.tlist()
        
        # times
        obsmjds = np.array([(np.floor(obs.mjd) + (obsdata[0]['time'])/24.0) for obsdata in obslist])

        print obsmjds

        if (not repeat) and ((obsmjds < mjdstart) + (obsmjds > mjdend)).any():
            raise Exception("Obs times outside of movie range of MJD %f - %f" % (mjdstart, mjdend))
                   
        # Observe nearest frame
        obsdata_out = []
        for i in xrange(len(obslist)):
            obsdata = obslist[i]
            
            # Frame number
            mjd = obsmjds[i]
            n = int(np.floor((mjd - mjdstart) * 86400. / self.framedur))
            
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
        obs_no_noise = vb.Obsdata(self.ra, self.dec, self.rf, obs.bw, obsdata_out, obs.tarr, source=self.source, mjd=np.floor(obs.mjd))
        return obs_no_noise

    def observe_same(self, obs, sgrscat=False, add_th_noise=True, ampcal=True, opacitycal=True, gainp=GAINPDEF, gain_offset=GAINPDEF, phasecal=True,
                                                                  jones=False, dcal=True, dtermp=DTERMPDEF, frcal=True,
                                                                  inv_jones=False, repeat=False):                                                     
        
        """Observe the movie on the same baselines as an existing observation object
           if sgrscat==True, the visibilites will be blurred by the Sgr A* scattering kernel
           Does NOT add noise
           
           gain_offset can be optionally set as a dictionary that specifies the percentage offset 
           for each telescope site. This can only be done currently if jones=False. 
           If gain_offset is a single value than it is the standard deviation 
           of a randomly selected gain offset. 
        """
        print "Producing clean visibilities from movie . . . "
        obs_out = self.observe_same_nonoise(obs, sgrscat=sgrscat, repeat=repeat)    
        
        # Jones Matrix Corruption
        if jones:
            if type(gain_offset)== dict:
                print 'WARNING: cannot use a dictionary gain offset when using jones matrices'
        
            print "Applying Jones Matrices to data . . . "
            obs_out = vb.add_jones_and_noise(obs_out, add_th_noise=add_th_noise, opacitycal=opacitycal, 
                                             ampcal=ampcal, gainp=gainp, phasecal=phasecal, dcal=dcal, dtermp=dtermp, frcal=frcal)
            
            
            if inv_jones:
                print "Applying a priori calibration with estimated Jones matrices . . . "
                obs_out = vb.apply_jones_inverse(obs_out, ampcal=ampcal, opacitycal=opacitycal, phasecal=phasecal, dcal=dcal, frcal=frcal)
        
        # No Jones Matrices, Add noise the old way        
        #!AC There is an asymmetry here - in the old way, we don't offer the ability to *not* unscale estimated noise.                                              
        elif add_th_noise:                
            print "Adding gain + phase errors to data and applying a priori calibration . . . "
            obs_out = vb.add_noise(obs_out, opacitycal=opacitycal, ampcal=ampcal, phasecal=phasecal,
                                   gainp=gainp, gain_offset=gain_offset, add_th_noise=add_th_noise)
        
        return obs_out
        
    def observe(self, array, tint, tadv, tstart, tstop, bw, mjd=None, 
                      sgrscat=False, add_th_noise=True, tau=TAUDEF, gainp=GAINPDEF, gain_offset=GAINPDEF, opacitycal=True, ampcal=True, phasecal=True,
                      jones=False, inv_jones=False, dcal=True, dtermp=DTERMPDEF, frcal=True, timetype='UTC',
                      repeat=False):

        """Observe the movie with an array object to produce an obsdata object.
	       tstart and tstop should be hrs in UTC.
           tint and tadv should be seconds.
           tau is the estimated optical depth. This can be a single number or a dictionary giving one tau per site
           if sgrscat==True, the visibilites will be blurred by the Sgr A* scattering kernel at the appropriate frequency
           
           gain_offset can be optionally set as a dictionary that specifies the percentage offset 
           for each telescope site. This can only be done currently if jones=False. 
           If gain_offset is a single value than it is the standard deviation 
           of a randomly selected gain offset. 

           repeat=True will repeat the movie if necessary to fill the observation time
	    """
        
        # Generate empty observation
        print "Generating empty observation file . . . "
        if mjd == None:
            mjd = self.mjd

        obs = array.obsdata(self.ra, self.dec, self.rf, bw, tint, tadv, tstart, tstop, tau=tau, mjd=mjd, timetype=timetype)
        
        # Observe on the same baselines as the empty observation and add noise
        obs = self.observe_same(obs, sgrscat=sgrscat, add_th_noise=add_th_noise, opacitycal=opacitycal,
                                ampcal=ampcal, gainp=gainp, phasecal=phasecal, gain_offset=gain_offset, 
                                jones=jones, inv_jones=inv_jones, dcal=dcal, dtermp=dtermp, frcal=frcal,
                                repeat=repeat)   
        
        return obs
                    
    def save_txt(self, fname):
        """Save movie data to text files"""
        
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
                        "MJD: %i \n" % (float(self.mjd) + self.start_hr/24.0 + i*self.framedur/86400.0) + 
                        "RF: %.4f GHz \n" % (self.rf/1e9) + 
                        "FOVX: %i pix %f as \n" % (self.xdim, pdimas * self.xdim) +
                        "FOVY: %i pix %f as \n" % (self.ydim, pdimas * self.ydim) +
                        "------------------------------------\n" + hf)
             
            # Save
            np.savetxt(fname_frame, outdata, header=head, fmt=fmts)


#!AC TODO this needs a lot of work
#!AC TODO think about how to do the filenames
def load_movie_txt(basename, nframes, framedur=-1, pulse=PULSE_DEFAULT):
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
        mjd_frac = float(file.readline().split()[2])
        mjd = np.floor(mjd_frac)
        hour = (mjd_frac - mjd)*24.0
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
            hour0 = hour
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

    if frame_dur == -1:
        frame_dur = (hour - hour0)/float(nframes)*3600.0

    out_mov = Movie(framelist, framedur, psize0, ra0, dec0, rf=rf0, source=src0, mjd=mjd0, start_hr=hour0, pulse=pulse)
    
    if len(qlist):
        print 'Loaded Stokes I, Q, and U movies'
        out_mov.add_qu(qlist, ulist)
    else:
        print 'Loaded Stokes I movie only'
    
    return out_mov
