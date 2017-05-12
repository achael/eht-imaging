import numpy as np
import matplotlib.pyplot as plt

import ehtim.obsdata
import ehtim.observing.obs_simulate as simobs
import ehtim.io.save 
import ehtim.io.load

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

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
        self.vframes = []

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
                raise Exception("image must be a 2D numpy array!")
            if qimage.shape != uimage.shape != (self.ydim, self.xdim):
                raise Exception("Q & U image shapes incompatible with I image!") 
            self.qframes[i] = qimage.flatten()
            self.uframes[i] = uimage.flatten()
        return

    def add_v(self, vmovie):
        """Add V movie
        """
        if not(len(vmovie) == len(self.frames)):
            raise Exception("V movie must have same length as I movie!")

        self.vframes = [0 for i in xrange(len(self.frames))]

        for i in xrange(len(self.frames)): 
            vimage = vmovie[i]
            if vimage.shape != (self.ydim, self.xdim):
                raise Exception("Q & U image shapes incompatible with I image!") 
            self.vframes[i] = vimage.flatten()
        return

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

        obsdata = simobs.observe_movie_nonoise(self, obs, sgrscat=sgrscat, repeat=repeat)

        obs_no_noise = ehtim.obsdata.Obsdata(self.ra, self.dec, self.rf, obs.bw, obsdata,
                                             obs.tarr, source=self.source, mjd=np.floor(obs.mjd))
        return obs_no_noise
    
    def observe_same(self, obsin, ft='direct', pad_frac=0.5, sgrscat=False, add_th_noise=True, 
                                opacitycal=True, ampcal=True, phasecal=True, frcal=True,dcal=True,
                                tau=TAUDEF, gainp=GAINPDEF, gain_offset=GAINPDEF, dtermp=DTERMPDEF,
                                jones=False, inv_jones=False, repeat=False):
        
        """Observe the movie on the same baselines as an existing observation object
           if sgrscat==True, the visibilites will be blurred by the Sgr A* scattering kernel
           Does NOT add noise
           
           gain_offset can be optionally set as a dictionary that specifies the percentage offset 
           for each telescope site. This can only be done currently if jones=False. 
           If gain_offset is a single value than it is the standard deviation 
           of a randomly selected gain offset. 
        """

        print "Producing clean visibilities from movie . . . "
        obs = self.observe_same_nonoise(obsin, sgrscat=sgrscat, ft=ft, pad_frac=pad_frac, repeat=repeat)    
        
        # Jones Matrix Corruption & Calibration
        if jones:
            print "Applying Jones Matrices to data . . . "
            obsdata = simobs.add_jones_and_noise(obs, add_th_noise=add_th_noise, 
                                                 opacitycal=opacitycal, ampcal=ampcal, 
                                                 phasecal=phasecal, dcal=dcal, frcal=frcal, 
                                                 gainp=gainp, dtermp=dtermp, gain_offset=gain_offset)
            
            obs =  ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata, 
                                             obs.tarr, source=obs.source, mjd=obs.mjd, 
                                             ampcal=ampcal, phasecal=phasecal,  
                                             opacitycal=opacitycal, dcal=dcal, frcal=frcal)
            if inv_jones:
                print "Applying a priori calibration with estimated Jones matrices . . . "
                obsdata = simobs.apply_jones_inverse(obs, 
                                                     ampcal=ampcal, opacitycal=opacitycal, 
                                                     phasecal=phasecal, dcal=dcal, frcal=frcal)

                obs =  ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata, 
                                                 obs.tarr, source=obs.source, mjd=obs.mjd, 
                                                 ampcal=ampcal, phasecal=phasecal,  
                                                 opacitycal=True, dcal=True, frcal=True) 
                                                 #these are always set to True after inverse jones call
        
        # No Jones Matrices, Add noise the old way        
        # !AC There is an asymmetry here - in the old way, we don't offer the ability to *not* unscale estimated noise.                                              
        elif add_th_noise:                
            print "Adding gain + phase errors to data and applying a priori calibration . . . "
            obsdata = simobs.add_noise(obs, add_th_noise=add_th_noise,
                                       ampcal=ampcal, phasecal=phasecal, opacitycal=opacitycal, 
                                       gainp=gainp, gain_offset=gain_offset)
            
            obs =  ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata, 
                                             obs.tarr, source=obs.source, mjd=obs.mjd, 
                                             ampcal=ampcal, phasecal=phasecal, 
                                             opacitycal=True, dcal=True, frcal=True) 
                                             #these are always set to True after inverse jones call
        return obs
        
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
                 
    def observe_vex(vex, source, synchronize_start = True, t_int = 0.0, sgrscat=False, add_th_noise=True, opacitycal=True, ampcal=True, phasecal=True, frcal=True,
                    tau=TAUDEF, gainp=GAINPDEF, gain_offset=GAINPDEF, dtermp=DTERMPDEF,
                    jones=False, inv_jones=False, dcal=True):
        """Generates an observation corresponding to a given vex objectS
           vex is a vex object
           source is the source string identifier in the vex object, e.g., 'SGRA'
           synchronize_start is a flag that determines whether the start of the movie should be defined to be the start of the observations
           t_int overrides the vex scans to produce visibilities for each t_int seconds
        """

        obs_List=[]
        movie = self.copy() #!AC TODO costly??

        if synchronize_start:
            movie.mjd = vex.sched[0]['mjd_floor']
            movie.start_hr = vex.sched[0]['start_hr']

        movie_start = float(movie.mjd) + movie.start_hr/24.0
        movie_end   = movie_start + len(movie.frames)*movie.framedur/24.0/3600.0

        print "Movie MJD Range: ",movie_start,movie_end

        snapshot = 1.0
        if t_int > 0.0: 
            snapshot = 0.0

        for i_scan in range(len(vex.sched)):
            if vex.sched[i_scan]['source'] != source:
                continue
            subarray = vex.array.make_subarray([vex.sched[i_scan]['scan'][key]['site'] for key in vex.sched[i_scan]['scan'].keys()])

            if snapshot == 1.0:
                t_int = np.max(np.array([vex.sched[i_scan]['scan'][site]['scan_sec'] for site in vex.sched[i_scan]['scan'].keys()]))
                print t_int
                #vex.sched[i_scan]['scan'][0]['scan_sec']

            vex_scan_start_mjd = float(vex.sched[i_scan]['mjd_floor']) + vex.sched[i_scan]['start_hr']/24.0
            vex_scan_stop_mjd  = vex_scan_start_mjd + vex.sched[i_scan]['scan'][0]['scan_sec']/3600.0/24.0

            print "Scan MJD Range: ",vex_scan_start_mjd,vex_scan_stop_mjd

            if vex_scan_start_mjd < movie_start or vex_scan_stop_mjd > movie_end:
                continue

            obs = subarray.obsdata(movie.ra, movie.dec, movie.rf, vex.bw_hz, t_int, t_int, 
                                       vex.sched[i_scan]['start_hr'], vex.sched[i_scan]['start_hr'] + vex.sched[i_scan]['scan'][0]['scan_sec']/3600.0 - EP, 
                                       mjd=vex.sched[i_scan]['mjd_floor'],
                                       elevmin=.01, elevmax=89.99, timetype='UTC')   
            obs_List.append(obs)

        if len(obs_List) == 0:
            raise Exception("Movie has no overlap with the vex file and source=" + source) 

        obs = ehtim.obsdata.merge_obs(obs_List)

        return movie.observe_same(obs, sgrscat=sgrscat, add_th_noise=add_th_noise, opacitycal=opacitycal,
                                    ampcal=ampcal, gainp=gainp, phasecal=phasecal, gain_offset=gain_offset, 
                                    jones=jones, inv_jones=inv_jones, dcal=dcal, dtermp=dtermp, frcal=frcal,
                                    repeat=False)      
    def save_txt(self, fname):
        """Save movie data to text files"""
        
        ehtim.io.ioutils.save_mov_txt(self, fname)
        return

##################################################################################################
# Movie creation functions
##################################################################################################

def load_txt(basename, nframes, framedur=-1, pulse=PULSE_DEFAULT):
    return load_movie_txt(basename, nframes, framedur=framedur, pulse=pulse)

