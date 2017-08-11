from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object

import numpy as np
import matplotlib.pyplot as plt

import ehtim.obsdata
import ehtim.observing.obs_simulate as simobs
import ehtim.io.save
import ehtim.io.load

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

class Movie(object):
    """A polarimetric movie (in units of Jy/pixel).

       Attributes:
           pulse (function): The function convolved with the pixel values for continuous image
    	   framedur (float): The frame duration in seconds
           psize (float): The pixel dimension in radians
           xdim (int): The number of pixels along the x dimension
           ydim (int): The number of pixels along the y dimension
           mjd (int): The integer MJD of the image
           start_hr (float): The start UTC hour of the observation (default = -1.0, meaning it is inherited from the observation)
           source (str): The astrophysical source name
           ra (float): The source Right Ascension in fractional hours
           dec (float): The source declination in fractional degrees
           rf (float): The image frequency in Hz
           frames (list): The list of frame vectors of stokes I values in Jy/pixel (each of len xdim*ydim)
           qframes (list): The list of frame vectors of stokes Q values in Jy/pixel (each of len xdim*ydim)
           uframes (list): The list of frame vectors of stokes U values in Jy/pixel (each of len xdim*ydim)
           vframes (list): The list of frame vectors of stokes V values in Jy/pixel (each of len xdim*ydim)
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
        """Add Stokes Q and U movies.
        """

        if not(len(qmovie) == len(umovie) == len(self.frames)):
            raise Exception("Q & U movies must have same length as I movie!")

        self.qframes = [0 for i in range(len(self.frames))]
        self.uframes = [0 for i in range(len(self.frames))]

        for i in range(len(self.frames)):
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
        """Add Stokes V movie.
        """
        if not(len(vmovie) == len(self.frames)):
            raise Exception("V movie must have same length as I movie!")

        self.vframes = [0 for i in range(len(self.frames))]

        for i in range(len(self.frames)):
            vimage = vmovie[i]
            if vimage.shape != (self.ydim, self.xdim):
                raise Exception("Q & U image shapes incompatible with I image!")
            self.vframes[i] = vimage.flatten()
        return

    def copy(self):
        """Return a copy of the Movie object.
        """
        new = Movie([imvec.reshape(self.ydim,self.xdim) for imvec in self.frames],
                     self.framedur, self.psize, self.ra, self.dec, rf=self.rf,
                     source=self.source, mjd=self.mjd, start_hr=self.start_hr, pulse=self.pulse)

        if len(self.qframes):
            new.add_qu([qvec.reshape(self.ydim,self.xdim) for qvec in self.qframes],
                         [uvec.reshape(self.ydim,self.xdim) for uvec in self.uframes])

        return new

    def flip_chi(self):
        """Flip between the different conventions for measuring position angle (East of North vs up from x axis).
        """
        self.qframes = [-qvec for qvec in self.qframes]
        return

    def observe_same_nonoise(self, obs, ft="direct", pad_frac=0.5,  repeat=False, sgrscat=False):
        """Observe the movie on the same baselines as an existing observation object without adding noise.

           Args:
               obs (Obsdata): the existing observation with  baselines where the image FT will be sampled
               ft (str): if "fast", use FFT to produce visibilities. Else "direct" for DTFT
               pad_frac (float): zero pad the image so that pad_frac*shortest baseline is captured in FFT
               repeat (bool): if True, repeat the movie to fill up the observation interval
               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel

           Returns:
               Obsdata: an observation object
        """

        obsdata = simobs.observe_movie_nonoise(self, obs, ft="direct", pad_frac=0.5, sgrscat=sgrscat, repeat=repeat)

        obs_no_noise = ehtim.obsdata.Obsdata(self.ra, self.dec, self.rf, obs.bw, obsdata,
                                             obs.tarr, source=self.source, mjd=np.floor(obs.mjd))
        return obs_no_noise

    def observe_same(self, obsin, ft='direct', pad_frac=0.5,  repeat=False,
                           sgrscat=False, add_th_noise=True,
                           opacitycal=True, ampcal=True, phasecal=True, frcal=True,dcal=True,
                           jones=False, inv_jones=False,
                           tau=TAUDEF, gainp=GAINPDEF, gain_offset=GAINPDEF, dtermp=DTERMPDEF):
        """Observe the image on the same baselines as an existing observation object and add noise.

           Args:
               obsin (Obsdata): the existing observation with  baselines where the image FT will be sampled
               ft (str): if "fast", use FFT to produce visibilities. Else "direct" for DTFT
               pad_frac (float): zero pad the image so that pad_frac*shortest baseline is captured in FFT
               repeat (bool): if True, repeat the movie to fill up the observation interval
               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
               add_th_noise (bool): if True, baseline-dependent thermal noise is added to each data point
               opacitycal (bool): if False, time-dependent gaussian errors are added to station opacities
               ampcal (bool): if False, time-dependent gaussian errors are added to station gains
               phasecal (bool): if False, time-dependent station-based random phases are added to data points
               frcal (bool): if False, feed rotation angle terms are added to Jones matrices. Must have jones=True
               dcal (bool): if False, time-dependent gaussian errors added to Jones matrices D-terms. Must have jones=True
               jones (bool): if True, uses Jones matrix to apply mis-calibration effects (gains, phases, Dterms), otherwise uses old formalism without D-terms
               inv_jones (bool): if True, applies estimated inverse Jones matrix (not including random terms) to calibrate data
               tau (float): the base opacity at all sites, or a dict giving one opacity per site
               gain_offset (float): the base gain offset at all sites, or a dict giving one gain offset per site
               gainp (float): the fractional std. dev. of the random error on the gains and opacities
               dtermp (float): the fractional std. dev. of the random error on the D-terms

           Returns:
               Obsdata: an observation object

        """

        print("Producing clean visibilities from movie . . . ")
        obs = self.observe_same_nonoise(obsin, sgrscat=sgrscat, ft=ft, pad_frac=pad_frac, repeat=repeat)

        # Jones Matrix Corruption & Calibration
        if jones:
            print("Applying Jones Matrices to data . . . ")
            obsdata = simobs.add_jones_and_noise(obs, add_th_noise=add_th_noise,
                                                 opacitycal=opacitycal, ampcal=ampcal,
                                                 phasecal=phasecal, dcal=dcal, frcal=frcal,
                                                 gainp=gainp, dtermp=dtermp, gain_offset=gain_offset)

            obs =  ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata,
                                             obs.tarr, source=obs.source, mjd=obs.mjd,
                                             ampcal=ampcal, phasecal=phasecal,
                                             opacitycal=opacitycal, dcal=dcal, frcal=frcal)
            if inv_jones:
                print("Applying a priori calibration with estimated Jones matrices . . . ")
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
            print("Adding gain + phase errors to data and applying a priori calibration . . . ")
            obsdata = simobs.add_noise(obs, add_th_noise=add_th_noise,
                                       ampcal=ampcal, phasecal=phasecal, opacitycal=opacitycal,
                                       gainp=gainp, gain_offset=gain_offset)

            obs =  ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata,
                                             obs.tarr, source=obs.source, mjd=obs.mjd,
                                             ampcal=ampcal, phasecal=phasecal,
                                             opacitycal=True, dcal=True, frcal=True)
                                             #these are always set to True after inverse jones call
        return obs

    def observe(self, array, tint, tadv, tstart, tstop, bw, repeat=False,
                      mjd=None, timetype='UTC', elevmin=ELEV_LOW, elevmax=ELEV_HIGH,
                      ft='direct', pad_frac=0.5, sgrscat=False, add_th_noise=True,
                      opacitycal=True, ampcal=True, phasecal=True, frcal=True, dcal=True,
                      jones=False, inv_jones=False,
                      tau=TAUDEF, gainp=GAINPDEF, gain_offset=GAINPDEF, dtermp=DTERMPDEF):

        """Generate baselines from an array object and observe the movie.

           Args:
               array (Array): an array object containing sites with which to generate baselines
               tint (float): the scan integration time in seconds
               tadv (float): the uniform cadence between scans in seconds
               tstart (float): the start time of the observation in hours
               tstop (float): the end time of the observation in hours
               bw (float): the observing bandwidth in Hz
               repeat (bool): if True, repeat the movie to fill up the observation interval
               mjd (int): the mjd of the observation, if different from the image mjd
               timetype (str): how to interpret tstart and tstop; either 'GMST' or 'UTC'
               elevmin (float): station minimum elevation in degrees
               elevmax (float): station maximum elevation in degrees
               ft (str): if "fast", use FFT to produce visibilities. Else "direct" for DTFT
               pad_frac (float): zero pad the image so that pad_frac*shortest baseline is captured in FFT
               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
               add_th_noise (bool): if True, baseline-dependent thermal noise is added to each data point
               opacitycal (bool): if False, time-dependent gaussian errors are added to station opacities
               ampcal (bool): if False, time-dependent gaussian errors are added to station gains
               phasecal (bool): if False, time-dependent station-based random phases are added to data points
               frcal (bool): if False, feed rotation angle terms are added to Jones matrices. Must have jones=True
               dcal (bool): if False, time-dependent gaussian errors added to Jones matrices D-terms. Must have jones=True
               jones (bool): if True, uses Jones matrix to apply mis-calibration effects (gains, phases, Dterms), otherwise uses old formalism without D-terms
               inv_jones (bool): if True, applies estimated inverse Jones matrix (not including random terms) to calibrate data
               tau (float): the base opacity at all sites, or a dict giving one opacity per site
               gain_offset (float): the base gain offset at all sites, or a dict giving one gain offset per site
               gainp (float): the fractional std. dev. of the random error on the gains and opacities
               dtermp (float): the fractional std. dev. of the random error on the D-terms

           Returns:
               Obsdata: an observation object

        """

        # Generate empty observation
        print("Generating empty observation file . . . ")
        if mjd == None:
            mjd = self.mjd

        obs = array.obsdata(self.ra, self.dec, self.rf, bw, tint, tadv, tstart, tstop, tau=tau, mjd=mjd, timetype=timetype)

        # Observe on the same baselines as the empty observation and add noise
        obs = self.observe_same(obs, sgrscat=sgrscat, add_th_noise=add_th_noise, opacitycal=opacitycal,
                                ampcal=ampcal, gainp=gainp, phasecal=phasecal, gain_offset=gain_offset,
                                jones=jones, inv_jones=inv_jones, dcal=dcal, dtermp=dtermp, frcal=frcal,
                                repeat=repeat)

        return obs

    def observe_vex(self, vex, source, synchronize_start=True, t_int=0.0,
                          sgrscat=False, add_th_noise=True,
                          opacitycal=True, ampcal=True, phasecal=True, frcal=True, dcal=True,
                          jones=False, inv_jones=False,
                          tau=TAUDEF, gainp=GAINPDEF, gain_offset=GAINPDEF, dtermp=DTERMPDEF):

        """Generate baselines from a vex file and observes the movie.

           Args:
               vex (Vex): an vex object containing sites and scan information
               source (str): the source string identifier in the vex object, e.g., 'SGRA'
               synchronize_start (bool): if True, the start of the movie will be defined to be the start of the observations
               t_int (float): if not zero, overrides the vex scans to produce visibilities for each t_int seconds

               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
               add_th_noise (bool): if True, baseline-dependent thermal noise is added to each data point
               opacitycal (bool): if False, time-dependent gaussian errors are added to station opacities
               ampcal (bool): if False, time-dependent gaussian errors are added to station gains
               phasecal (bool): if False, time-dependent station-based random phases are added to data points
               frcal (bool): if False, feed rotation angle terms are added to Jones matrices. Must have jones=True
               dcal (bool): if False, time-dependent gaussian errors added to Jones matrices D-terms. Must have jones=True
               jones (bool): if True, uses Jones matrix to apply mis-calibration effects (gains, phases, Dterms), otherwise uses old formalism without D-terms
               inv_jones (bool): if True, applies estimated inverse Jones matrix (not including random terms) to calibrate data
               tau (float): the base opacity at all sites, or a dict giving one opacity per site
               gain_offset (float): the base gain offset at all sites, or a dict giving one gain offset per site
               gainp (float): the fractional std. dev. of the random error on the gains and opacities
               dtermp (float): the fractional std. dev. of the random error on the D-terms

           Returns:
               Obsdata: an observation object

        """

        obs_List=[]
        movie = self.copy() #!AC TODO costly??

        if synchronize_start:
            movie.mjd = vex.sched[0]['mjd_floor']
            movie.start_hr = vex.sched[0]['start_hr']

        movie_start = float(movie.mjd) + movie.start_hr/24.0
        movie_end   = movie_start + len(movie.frames)*movie.framedur/24.0/3600.0

        print("Movie MJD Range: ",movie_start,movie_end)

        snapshot = 1.0
        if t_int > 0.0:
            snapshot = 0.0

        for i_scan in range(len(vex.sched)):
            if vex.sched[i_scan]['source'] != source:
                continue
            subarray = vex.array.make_subarray([vex.sched[i_scan]['scan'][key]['site'] for key in list(vex.sched[i_scan]['scan'].keys())])

            if snapshot == 1.0:
                t_int = np.max(np.array([vex.sched[i_scan]['scan'][site]['scan_sec'] for site in list(vex.sched[i_scan]['scan'].keys())]))
                print(t_int)
                #vex.sched[i_scan]['scan'][0]['scan_sec']

            vex_scan_start_mjd = float(vex.sched[i_scan]['mjd_floor']) + vex.sched[i_scan]['start_hr']/24.0
            vex_scan_stop_mjd  = vex_scan_start_mjd + vex.sched[i_scan]['scan'][0]['scan_sec']/3600.0/24.0

            print("Scan MJD Range: ",vex_scan_start_mjd,vex_scan_stop_mjd)

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
        """Save the Movie data to text files with basename fname and filenames basename + 00001, etc. """

        ehtim.io.ioutils.save_mov_txt(self, fname)
        return

    def export_mp4(self, out='movie.mp4', fps=10, dpi=120, scale='linear', dynamic_range=1000.0, pad_factor=1, verbose=False):
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig = plt.figure()
        
        extent = self.psize/RADPERUAS*self.xdim*np.array((1,-1,-1,1)) / 2.
        maxi = np.max(np.concatenate([im for im in self.frames]))

        def im_data(n):
            n_data = (n-n%pad_factor)/pad_factor
            if scale == 'linear':
                return self.frames[n_data].reshape((self.ydim,self.xdim))
            else:
                return np.log(self.frames[n_data][n_data].reshape((self.ydim,self.xdim)) + maxi/plot_dynamic_range)

        plt_im = plt.imshow(im_data(0), extent=extent, cmap=plt.get_cmap('afmhot'), interpolation='gaussian') 
        if scale == 'linear':
            plt_im.set_clim([0,maxi])
        else:
            plt_im.set_clim([np.log(maxi/dynamic_range),np.log(maxi)])

        plt.xlabel('Relative RA ($\mu$as)')
        plt.ylabel('Relative Dec ($\mu$as)')
        fig.set_size_inches([5,5])
        plt.tight_layout()

        def update_img(n):
            if verbose:
                print("processing frame {0} of {1}".format(n, len(self.frames)*pad_factor))
            plt_im.set_data(im_data(n))
            return plt_im

        ani = animation.FuncAnimation(fig,update_img,len(self.frames)*pad_factor,interval=1e3/fps)
        writer = animation.writers['ffmpeg'](fps=fps, bitrate=1e6)
        ani.save(out,writer=writer,dpi=dpi)



##################################################################################################
# Movie creation and export functions
##################################################################################################

def load_txt(basename, nframes, framedur=-1, pulse=PULSE_DEFAULT):
    """Read in a movie from text files and create a Movie object.

       Args:
           basename (str): The base name of individual movie frames. Files should have names basename + 00001, etc.
           nframes (int): The total number of frames
           framedur (float): The frame duration (default = -1, corresponding to framedur taken from file headers)
           pulse (function): The function convolved with the pixel values for continuous image

       Returns:
           Movie: a Movie object
    """
    return load_movie_txt(basename, nframes, framedur=framedur, pulse=pulse)




