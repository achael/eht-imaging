# movie.py
# a interferometric movie class
#
#    Copyright (C) 2018 Andrew Chael
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object

import numpy as np
import matplotlib.pyplot as plt

import ehtim.image
import ehtim.obsdata
import ehtim.observing.obs_simulate as simobs
import ehtim.io.save
import ehtim.io.load

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

###########################################################################################################################################
#Movie object
###########################################################################################################################################
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

    def __init__(self, movie, framedur, psize, ra, dec, rf=RF_DEFAULT,
                       polrep='stokes', pol_prim=None,
                       pulse=PULSE_DEFAULT, source=SOURCE_DEFAULT,
                       mjd=MJD_DEFAULT, start_hr=0.0):

        """A polarimetric image (in units of Jy/pixel).

           Args:
               movie (list): The list of 2D frames, each is a Jy/pixel array
    	       framedur (float): The frame duration in seconds
               psize (float): The pixel dimension in radians
               ra (float): The source Right Ascension in fractional hours
               dec (float): The source declination in fractional degrees
               rf (float): The image frequency in Hz
               polrep (str): polarization representation, either 'stokes' or 'circ'
               pol_prim (str): The default image: I,Q,U or V for Stokes, RR,LL,LR,RL for Circular
               pulse (function): The function convolved with the pixel values for continuous image.
               source (str): The source name
               mjd (int): The integer MJD of the image
               start_hr (float): The start time of image (UTC hours)

           Returns:
               (Image): the Image object
        """

        if len(movie[0].shape) != 2:
            raise Exception("frames must each be a 2D numpy array")

        #the list of frames
        frames = [image.flatten() for image in movie]
        self.nframes = len(self.frames)
        if polrep=='stokes':
            if pol_prim is None: pol_prim = 'I'
            if pol_prim=='I':
                self._movdict = {'I':frames,'Q':[],'U':[],'V':[]}
            elif pol_prim=='V':
                self._movdict = {'I':[],'Q':[],'U':[],'V':frames}
            elif pol_prim=='Q':
                self._movdict = {'I':[],'Q':frames,'U':[],'V':[]}
            elif pol_prim=='U':
                self._movdict = {'I':[],'Q':[],'U':frames,'V':[]}
            else:
                raise Exception("for polrep=='stokes', pol_prim must be 'I','Q','U', or 'V'!")

        elif polrep=='circ':
            if pol_prim is None:
                print("polrep is 'circ' and no pol_prim specified! Setting pol_prim='RR'")
                pol_prim = 'RR'
            if pol_prim=='RR':
                self._movdict = {'RR':frames,'LL':[],'RL':[],'LR':[]}
            elif pol_prim=='LL':
                self._movdict = {'RR':[],'LL':frames,'RL':[],'LR':[]}
            else:
                raise Exception("for polrep=='circ', pol_prim must be 'RR' or 'LL'!")

        self.pol_prim =  pol_prim

        self.framedur = float(framedur)
        self.polrep = polrep
        self.pulse = pulse
        self.psize = float(psize)
        self.xdim = movie[0].shape[1]
        self.ydim = movie[0].shape[0]

        self.ra = float(ra)
        self.dec = float(dec)
        self.rf = float(rf)
        self.source = str(source)
        self.mjd = int(mjd)
        if start_hr > 24:
            self.mjd += int((start_hr - start_hr % 24)/24)
            self.start_hr = float(start_hr % 24)
        else:
            self.start_hr = start_hr


    @property
    def frames(self):
        frames = self._movdict[self.pol_prim]
        return frames

    @frames.setter
    def frames(self, frames):
        if len(frames[0]) != self.xdim*self.ydim:
            raise Exception("imvec size is not consistent with xdim*ydim!")
        #TODO -- more checks on consistency with the existing pol data???

        self._movdict[self.pol_prim] =  frames

    @property
    def iframes(self):

        if self.polrep!='stokes':
            raise Exception("iframes is not defined unless self.polrep=='stokes' -- try self.switch_polrep()")

        frames = self._movdict['I']
        return frames

    @iframes.setter
    def iframes(self, frames):

        if len(frames[0]) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        #TODO -- more checks on the consistency of the imvec with the existing pol data???
        self._movdict['I'] =  frames

    @property
    def qframes(self):

        if self.polrep!='stokes':
            raise Exception("qframes is not defined unless self.polrep=='stokes' -- try self.switch_polrep()")

        frames = self._movdict['Q']
        return frames

    @qframes.setter
    def qframes(self, frames):

        if len(frames[0]) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        #TODO -- more checks on the consistency of the imvec with the existing pol data???
        self._movdict['Q'] =  frames

    @property
    def uframes(self):

        if self.polrep!='stokes':
            raise Exception("uframes is not defined unless self.polrep=='stokes' -- try self.switch_polrep()")

        frames = self._movdict['U']
        return frames

    @uframes.setter
    def uframes(self, frames):

        if len(frames[0]) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        #TODO -- more checks on the consistency of the imvec with the existing pol data???
        self._movdict['U'] =  frames

    @property
    def vframes(self):

        if self.polrep!='stokes':
            raise Exception("vframes is not defined unless self.polrep=='stokes' -- try self.switch_polrep()")

        frames = self._movdict['V']
        return frames

    @vframes.setter
    def vframes(self, frames):

        if len(frames[0]) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        #TODO -- more checks on the consistency of the imvec with the existing pol data???
        self._movdict['V'] =  frames

    @property
    def rrframes(self):

        if self.polrep!='circ':
            raise Exception("rrframes is not defined unless self.polrep=='circ' -- try self.switch_polrep()")

        frames = self._movdict['RR']
        return frames

    @rrframes.setter
    def rrframes(self, frames):

        if len(frames[0]) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        #TODO -- more checks on the consistency of the imvec with the existing pol data???
        self._movdict['RR'] =  frames

    @property
    def llframes(self):

        if self.polrep!='circ':
            raise Exception("llframes is not defined unless self.polrep=='circ' -- try self.switch_polrep()")

        frames = self._movdict['LL']
        return frames

    @llframes.setter
    def llframes(self, frames):

        if len(frames[0]) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        #TODO -- more checks on the consistency of the imvec with the existing pol data???
        self._movdict['LL'] =  frames

    @property
    def rlvec(self):

        if self.polrep!='circ':
            raise Exception("rlframes is not defined unless self.polrep=='circ' -- try self.switch_polrep()")

        frames = self._movdict['RL']
        return frames

    @rlvec.setter
    def rlvec(self, frames):

        if len(frames[0]) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        #TODO -- more checks on the consistency of the imvec with the existing pol data???
        self._movdict['RL'] =  frames

    @property
    def lrvec(self):

        if self.polrep!='circ':
            raise Exception("lrframes is not defined unless self.polrep=='circ' -- try self.switch_polrep()")

        frames = self._movdict['LR']
        return frames

    @lrvec.setter
    def lrvec(self, frames):

        if len(frames[0]) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        #TODO -- more checks on the consistency of the imvec with the existing pol data???
        self._movdict['LR'] =  frames

    def copy(self):

        """Return a copy of the Movie object.

           Args:

           Returns:
               (Image): copy of the Image.
        """

        # Make new  movie with primary polarization
        newmov = Movie([imvec.reshape(self.ydim,self.xdim) for imvec in self.frames],
                       self.framedur, self.psize, self.ra, self.dec,
                       polrep=self.polrep, pol_prim=self.pol_prim, start_hr=self.start_hr,
                       rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Copy over all polarization movies
        for pol in list(self._movdict.keys()):
            if pol==self.pol_prim: continue
            polframes = self._movdict[pol]
            if len(polframes):
                newmov.add_pol_movie([polvec.reshape(self.ydim,self.xdim) for polvec in polframes], pol)

        return newmov


    def add_pol_movie(self, movie, pol):

        """Add another movie polarization. f

           Args:
               movie (list): list of 2D frames (possibly complex) in a Jy/pixel array
               pol (str): The image type: 'I','Q','U','V' for stokes, 'RR','LL','RL','LR' for circ
        """
        if not(len(movie) == self.nframes):
            raise Exception("new pol movies must have same length as primary movie!")

        if pol==self.pol_prim:
            raise Exception("new pol in add_pol_movie is the same as pol_prim!")
        if np.any(np.array([image.shape != (self.ydim, self.xdim) for image in movie])):
            raise Exception("add_pol_movie image shapes incompatible with primary image!")
        if not (pol in list(self._movdict.keys())):
            raise Exception("for polrep==%s, pol in add_pol_movie must be in "%self.polrep + ",".join(list(self._movdict.keys())))

        if self.polrep=='stokes':
            if pol=='I': self.iframes = [image.flatten() for image in movie]
            elif pol=='Q': self.qframes = [image.flatten() for image in movie]
            elif pol=='U': self.uframes = [image.flatten() for image in movie]
            elif pol=='V': self.vframes = [image.flatten() for image in movie]
            self._movdict = {'I':self.iframes,'Q':self.qframes,'U':self.uframes,'V':self.vframes}
        elif self.polrep=='circ':
            if pol=='RR': self.rrframes = [image.flatten() for image in movie]
            elif pol=='LL': self.llframes = [image.flatten() for image in movie]
            elif pol=='RL': self.rlframes = [image.flatten() for image in movie]
            elif pol=='LR': self.lrframes = [image.flatten() for image in movie]
            self._movdict = {'RR':self.rrframes,'LL':self.llframes,'RL':self.rlframes,'LR':self.lrframes}

        return

    # TODO deprecated -- replace with generic add_pol_movie
    def add_qu(self, qmovie, umovie):
        """Add Stokes Q and U movies. self.polrep must be 'stokes'

           Args:
               qmovie (list): list of 2D Stokes Q frames in Jy/pixel array
               umovie (list): list of 2D Stokes U frames in Jy/pixel array

           Returns:
        """

        if self.polrep!='stokes':
            raise Excpetion("polrep must be 'stokes' for add_qu() !")
        self.add_pol_movie(qmovie,'Q')
        self.add_pol_movie(umovie,'U')

        return

    # TODO deprecated -- replace with generic add_pol_movie
    def add_v(self, vmovie):
        """Add Stokes V movie. self.polrep must be 'stokes'

           Args:
               vmovie (list): list of 2D Stokes Q frames in Jy/pixel array

           Returns:
        """

        if self.polrep!='stokes':
            raise Excpetion("polrep must be 'stokes' for add_v() !")
        self.add_pol_movie(vmovie,'V')

        return


    def switch_polrep(self, polrep_out='stokes', pol_prim_out=None):

        """Return a new movie with the polarization representation changed
           Args:
               polrep_out (str):  the polrep of the output data
               pol_prim_out (str): The default movie: I,Q,U or V for Stokes, RR,LL,LR,RL for Circular

           Returns:
               (Movie): new movie object with potentially different polrep
        """

        if polrep_out not in ['stokes','circ']:
            raise Exception("polrep_out must be either 'stokes' or 'circ'")
        if pol_prim_out is None:
            if polrep_out=='stokes': pol_prim_out = 'I'
            elif polrep_out=='circ': pol_prim_out = 'RR'

        # Simply copy if the polrep is unchanged
        if polrep_out==self.polrep and pol_prim_out==self.pol_prim:
            return self.copy()

        # Assemble a dictionary of new polarization vectors
        if polrep_out=='stokes':
            if self.polrep=='stokes':
                movdict = {'I':self.iframes,'Q':self.qframes,'U':self.uframes,'V':self.vframes}
            else:
                if len(self.rrframes)==0 or len(self.llframes)==0:
                    iframes = []
                    vframes = []
                else:
                    iframes = [0.5*(self.rrframes[i] + self.llframes[i]).reshape(self.ydim, self.xdim) for i in range(self.nframes)]
                    vframes = [0.5*(self.rrframes[i] - self.llframes[i]).reshape(self.ydim, self.xdim) for i in range(self.nframes)]

                if len(self.rlframes)==0 or len(self.lrframes)==0:
                    qframes = []
                    uframes = []
                else:
                    qframes = [np.real(0.5*(self.lrframes[i] + self.rlframes[i])).reshape(self.ydim, self.xdim) for i in range(self.nframes)]
                    uframes = [np.real(0.5j*(self.lrframes[i] - self.rlframes[i])).reshape(self.ydim, self.xdim) for i in range(self.nframes)]

                movdict = {'I':iframes,'Q':qframes,'U':uframes,'V':vframes}

        elif polrep_out=='circ':
            if self.polrep=='circ':
                movdict = {'RR':self.rrframes,'LL':self.llframes,'RL':self.rlframes,'LR':self.lrframes}
            else:
                if len(self.iframes)==0 or len(self.vframes)==0:
                    rrframes = []
                    llframes = []
                else:
                    rrframes = [(self.iframes[i] + self.vframes[i]).reshape(self.ydim, self.xdim) for i in range(self.nframes)]
                    llframes = [(self.iframes[i] - self.vframes[i]).reshape(self.ydim, self.xdim) for i in range(self.nframes)]

                if len(self.qframes)==0 or len(self.uframes)==0:
                    rlframes = []
                    lrframes = []
                else:
                    rlframes = [(self.qframes[i] + 1j*self.uframes[i]).reshape(self.ydim, self.xdim) for i in range(self.nframes)]
                    lrframes = [(self.qframes[i] - 1j*self.uframes[i]).reshape(self.ydim, self.xdim) for i in range(self.nframes)]

                movdict = {'RR':rrframes,'LL':llframes,'RL':rlframes,'LR':lrframes}

        # Assemble the new movie
        frames = movdict[pol_prim_out]
        if len(frames)==0:
            raise Exception("for switch_polrep to %s with pol_prim_out=%s, \n"%(polrep_out,pol_prim_out) +
                            "output movie is not defined")

        newmov = Movie(frames,
                       self.framedur, self.psize, self.ra, self.dec,
                       polrep=polrep_out, pol_prim=pol_prim_out, start_hr=self.start_hr,
                       rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Add in any other polarizations
        for pol in list(movdict.keys()):
            if pol==pol_prim_out: continue
            polframes = movdict[pol]
            if len(polframes):
                newmov.add_pol_movie([polvec.reshape(self.ydim,self.xdim) for polvec in polframes], pol)

        return newmov


    def flip_chi(self):

        """Flip between the different conventions for measuring the EVPA (E of N vs N of E).

           Args:

           Returns:
               (Image): movie with flipped EVPA
        """


        mov = self.copy()
        if mov.polrep=='stokes':
            mov.qframes *= [-qvec for qvec in mov.qframes]

        elif mov.polrep=='circ':
            mov.lrframes *= [-np.conjugate(lrvec) for lrvec in mov.lrframes]
            mov.rlframes *= [-np.conjugate(rlvel) for rlvec in mov.rlframes]

        return mov

    def orth_chi(self):

        """Rotate the EVPA 90 degrees

           Args:

           Returns:
               (Image): movie with rotated EVPA
        """
        mov = self.copy()
        if im.polrep=='stokes':
            mov.qframes *= [-uvec for uvec in mov.vframes]

        elif mov.polrep=='circ':
            mov.lrframes *= [np.conjugate(lrvec) for lrvec in mov.lrframes]
            mov.rlframes *= [np.conjugate(rlvel) for rlvec in mov.rlframes]

        return mov

    def frametimes(self):

        """Return the list of movie frame times in hours

           Args:

           Returns:
                (numpy.ndarray) : 1d array of movie frame times
        """

        return self.start_hr + np.arange(self.nframes)*self.framedur/3600.


    def fovx(self):

        """Return the movie fov in x direction in radians.

           Args:

           Returns:
                (float) : movie fov in x direction (radian)
        """

        return self.psize * self.xdim

    def fovy(self):

        """Returns the movie fov in y direction in radians.

           Args:

           Returns:
                (float) : movie fov in y direction (radian)
        """

        return self.psize * self.ydim

    def lightcurve(self):

        """Return the total flux over time of the image in Jy.

           Args:

           Returns:
                (numpy.Array) :  image total flux (Jy) over time
        """
        if self.polrep=='stokes':
            flux = [np.sum(ivec) for ivec in self.iframes]
        elif self.polrep=='circ':
            flux = [0.5*(np.sum(self.rrframes[i])+np.sum(self.llframes[i])) for i in range(self.nframes)]

        return np.array(flux)

    def lin_polfrac_curve(self):

        """Return the total fractional linear polarized flux over time

           Args:

           Returns:
                (numpy.ndarray) : image fractional linear polarized flux per frame
        """
        if self.polrep=='stokes':
            frac = [np.abs(np.sum(self.qframes[i] + 1j*self.uframes[i])) / np.abs(np.sum(self.iframes[i]))
                    for i in range(self.nframes)]
        elif self.polrep=='circ':
            frac = [2*np.abs(np.sum(self.rlframes[i])) / np.abs(np.sum(self.rrframes[i]+self.llframes[i]))
                    for i in range(self.nframes)]
        return np.array(frac)

    def circ_polfrac_curve(self):

        """Return the (signed) total fractional circular polarized flux over time

           Args:

           Returns:
                (numpy.ndarray) : image fractional circular polarized flux per frame
        """
        if self.polrep=='stokes':
            frac = [np.sum(self.vframes[i]) / np.abs(np.sum(self.iframes[i]))
                    for i in range(self.nframes)]
        elif self.polrep=='circ':
            frac = [np.sum(self.rrframes[i]-self.llframes[i]) / np.abs(np.sum(self.rrframes[i] + self.llframes[i]))
                    for i in range(self.nframes)]

        return np.array(frac)


    def get_frame(self, n):
        """Return an Image of the nth frame

           Args:
               n (int): the frame number

           Returns:
               (Image): the Image object of the nth frame
        """

        if n<0 or n>=len(self.frames):
            raise Exception("n must be in the range 0 - %i"% self.nframes)

        time = self.start_hr + n * self.framedur/3600

        # Make the primary image
        imarr = self.frames[n].reshape(self.ydim, self.xdim)
        outim = ehtim.image.Image(imarr, self.psize, self.ra, self.dec, self.pa,
                                    polrep=self.polrep, pol_prim=self.pol_prim, time=time,
                                    rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Copy over the rest of the polarizations
        for pol in list(self._movdict.keys()):
            if pol==self.pol_prim: continue
            polframes = self._movdict[pol]
            if len(polframes):
                polvec = polframes[n]
                polarr = polvec.reshape(self.ydim, self.xdim).copy()
                outim.add_pol_image(polarr, pol)

        return outim


    def im_list(self):
        """Return a list of the movie frames

           Args:

           Returns:
               (list): list of Image objects
        """

        return [self.get_frame(j) for j in range(self.nframes)]

    def avg_frame(self):
        """Coherently Average the movie frames into a single image.

           Returns:
                (Image) : averaged image of all frames
        """

        time = self.start_hr

        # Make the primary image
        avg_imvec = np.mean(np.array(self.frames),axis=0)
        avg_imarr = avg_imvec.reshape(self.ydim, self.xdim)
        outim = ehtim.image.Image(avg_imarr, self.psize, self.ra, self.dec, self.pa,
                                  polrep=self.polrep, pol_prim=self.pol_prim, time=self.start_hr,
                                  rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Copy over the rest of the average polarizations
        for pol in list(self._movdict.keys()):
            if pol==self.pol_prim: continue
            polframes = self._movdict[pol]
            if len(polframes):
                avg_polvec = np.mean(np.array(polframes),axis=0)
                avg_polarr = avg_polvec.reshape(self.ydim, self.xdim)
                outim.add_pol_image(avg_polarr, pol)

        return outim


    def observe_same_nonoise(self, obs, sgrscat=False, ttype="nfft", fft_pad_factor=2, repeat=False):
        """Observe the movie on the same baselines as an existing observation object without adding noise.

           Args:
               obs (Obsdata): the existing observation with  baselines where the image FT will be sampled
               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
               ttype (str): if "fast", use FFT to produce visibilities. Else "direct" for DTFT
               fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in FFT
               repeat (bool): if True, repeat the movie to fill up the observation interval

           Returns:
               (Obsdata): an observation object
        """


        # Check for agreement in coordinates and frequency
        tolerance = 1e-8
        if (np.abs(self.ra - obs.ra) > tolerance) or (np.abs(self.dec - obs.dec) > tolerance):
            raise Exception("Movie coordinates are not the same as observtion coordinates!")
        if (np.abs(self.rf - obs.rf)/obs.rf > tolerance):
            raise Exception("Movie frequency is not the same as observation frequency!")

        if ttype=='direct' or ttype=='fast' or ttype=='nfft':
            print("Producing clean visibilities from movie with " + ttype + " FT . . . ")
        else:
            raise Exception("ttype=%s, options for ttype are 'direct', 'fast', 'nfft'"%ttype)

        mjdstart = float(self.mjd) + float(self.start_hr/24.0)
        mjdend = mjdstart + (len(self.frames)*self.framedur)/86400.0

        # Get data
        obslist = obs.tlist()

        # times
        obsmjds = np.array([(np.floor(obs.mjd) + (obsdata[0]['time'])/24.0) for obsdata in obslist])

        if (not repeat) and ((obsmjds < mjdstart) + (obsmjds > mjdend)).any():
            raise Exception("Obs times outside of movie range of MJD %f - %f" % (mjdstart, mjdend))

        # Observe nearest frame
        obsdata_out = []
        for i in range(len(obslist)):
            obsdata = obslist[i]

            # Frame number
            mjd = obsmjds[i]
            n = int(np.floor((mjd - mjdstart) * 86400. / self.framedur))

            if (n >= len(self.frames)):
                if repeat: n = np.mod(n, len(self.frames))
                else: raise Exception("Obs times outside of movie range of MJD %f - %f" % (mjdstart, mjdend))


            # Get the frame visibilities
            uv = recarr_to_ndarr(obsdata[['u','v']],'f8')
            im = self.get_frame(n)
            data = simobs.sample_vis(im, uv, sgrscat=sgrscat, polrep_obs=obs.polrep,
                                         ttype=ttype, fft_pad_factor=fft_pad_factor, zero_empty_pol=True)

            # Put visibilities into the obsdata
            if obs.polrep=='stokes':
                obsdata['vis'] = data[0]
                if not(data[1] is None):
                    obsdata['qvis'] = data[1]
                    obsdata['uvis'] = data[2]
                    obsdata['vvis'] = data[3]

            elif obs.polrep=='circ':
                obsdata['rrvis'] = data[0]
                if not(data[1] is None):
                    obsdata['llvis'] = data[1]
                if not(data[2] is None):
                    obsdata['rlvis'] = data[2]
                    obsdata['lrvis'] = data[3]


            if len(obsdata_out):
                obsdata_out = np.hstack((obsdata_out, obsdata))
            else:
                obsdata_out = obsdata

        obsdata_out = np.array(obsdata_out, dtype=obs.poltype)
        obs_no_noise = ehtim.obsdata.Obsdata(self.ra, self.dec, self.rf, obs.bw, obsdata_out, obs.tarr,
                               source=self.source, mjd=np.floor(obs.mjd), polrep=obs.polrep,
                               ampcal=True, phasecal=True, opacitycal=True, dcal=True, frcal=True,
                               timetype=obs.timetype, scantable=obs.scans)

        return obs_no_noise

    def observe_same(self, obs_in, ttype='direct', fft_pad_factor=2,  repeat=False,
                           sgrscat=False, add_th_noise=True,
                           opacitycal=True, ampcal=True, phasecal=True, frcal=True,dcal=True,
                           stabilize_scan_phase=False, stabilize_scan_amp=False,
                           jones=False, inv_jones=False,
                           tau=TAUDEF, taup=GAINPDEF,
                           gainp=GAINPDEF, gain_offset=GAINPDEF,
                           dtermp=DTERMPDEF, dterm_offset=DTERMPDEF):

        """Observe the image on the same baselines as an existing observation object and add noise.

           Args:
               obs_in (Obsdata): the existing observation with  baselines where the image FT will be sampled
               ttype (str): if "fast" or "nfft", use FFT to produce visibilities. Else "direct" for DTFT
               fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in FFT

               repeat (bool): if True, repeat the movie to fill up the observation interval
               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
               add_th_noise (bool): if True, baseline-dependent thermal noise is added to each data point
               opacitycal (bool): if False, time-dependent gaussian errors are added to station opacities
               ampcal (bool): if False, time-dependent gaussian errors are added to station gains
               phasecal (bool): if False, time-dependent station-based random phases are added to data points
               frcal (bool): if False, feed rotation angle terms are added to Jones matrices. Must have jones=True
               stabilize_scan_phase (bool): if True, random phase errors are constant over scans
               stabilize_scan_amp (bool): if True, random amplitude errors are constant over scans
               dcal (bool): if False, time-dependent gaussian errors added to Jones matrices D-terms. Must have jones=True

               jones (bool): if True, uses Jones matrix to apply mis-calibration effects (gains, phases, Dterms), otherwise uses old formalism without D-terms
               inv_jones (bool): if True, applies estimated inverse Jones matrix (not including random terms) to calibrate data

               tau (float): the base opacity at all sites, or a dict giving one opacity per site
               gainp (float): the fractional std. dev. of the random error on the gains
               gain_offset (float): the base gain offset at all sites, or a dict giving one gain offset per site
               taup (float): the fractional std. dev. of the random error on the opacities
               dtermp (float): the fractional std. dev. of the random error on the D-terms
               dterm_offset (float): the base dterm offset at all sites, or a dict giving one dterm offset per site

            Returns:
               (Obsdata): an observation object

        """

        print("Producing clean visibilities from movie . . . ")
        obs = self.observe_same_nonoise(obs_in, sgrscat=sgrscat, ttype=ttype, fft_pad_factor=fft_pad_factor, repeat=repeat)

        # Jones Matrix Corruption & Calibration
        if jones:
            print("Applying Jones Matrices to data . . . ")
            obsdata = simobs.add_jones_and_noise(obs, add_th_noise=add_th_noise,
                                                 opacitycal=opacitycal, ampcal=ampcal,
                                                 phasecal=phasecal, dcal=dcal, frcal=frcal,
                                                 stabilize_scan_phase=stabilize_scan_phase,
                                                 stabilize_scan_amp=stabilize_scan_amp,
                                                 gainp=gainp, taup=taup, gain_offset=gain_offset,
                                                 dtermp=dtermp,dterm_offset=dterm_offset)

            obs =  ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata, obs.tarr,
                                         source=obs.source, mjd=obs.mjd, polrep=obs_in.polrep,
                                         ampcal=ampcal, phasecal=phasecal, opacitycal=opacitycal, dcal=dcal, frcal=frcal,
                                         timetype=obs.timetype, scantable=obs.scans)

            if inv_jones:
                obsdata = simobs.apply_jones_inverse(obs, opacitycal=opacitycal, dcal=dcal, frcal=frcal)

                obs =  ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata, obs.tarr,
                                             source=obs.source, mjd=obs.mjd, polrep=obs_in.polrep,
                                             ampcal=ampcal, phasecal=phasecal,
                                             opacitycal=True, dcal=True, frcal=True,
                                             timetype=obs.timetype, scantable=obs.scans)
                                             #these are always set to True after inverse jones call

        # No Jones Matrices, Add noise the old way
        # TODO There is an asymmetry here - in the old way, we don't offer the ability to *not* unscale estimated noise.
        elif add_th_noise:
            obsdata = simobs.add_noise(obs, add_th_noise=add_th_noise,
                                       ampcal=ampcal, phasecal=phasecal, opacitycal=opacitycal,
                                       gainp=gainp, taup=taup, gain_offset=gain_offset)

            obs =  ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata, obs.tarr,
                                         source=obs.source, mjd=obs.mjd, polrep=obs_in.polrep,
                                         ampcal=ampcal, phasecal=phasecal, opacitycal=True, dcal=True, frcal=True,
                                         timetype=obs.timetype, scantable=obs.scans)
                                         #these are always set to True after inverse jones call

        return obs

    def observe(self, array, tint, tadv, tstart, tstop, bw, repeat=False,
                      mjd=None, timetype='UTC', polrep_obs=None,
                      elevmin=ELEV_LOW, elevmax=ELEV_HIGH,
                      ttype='nfft', fft_pad_factor=2,
                      fix_theta_GMST=False, sgrscat=False, add_th_noise=True,
                      opacitycal=True, ampcal=True, phasecal=True, frcal=True, dcal=True,
                      stabilize_scan_phase=False, stabilize_scan_amp=False,
                      jones=False, inv_jones=False,
                      tau=TAUDEF, taup=GAINPDEF,
                      gainp=GAINPDEF, gain_offset=GAINPDEF,
                      dtermp=DTERMPDEF, dterm_offset=DTERMPDEF):


        """Generate baselines from an array object and observe the movie.

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

               polrep_obs (str): 'stokes' or 'circ' sets the data polarimetric representtion
               ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT
               fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in the FFT

               fix_theta_GMST (bool): if True, stops earth rotation to sample fixed u,v points through time
               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
               add_th_noise (bool): if True, baseline-dependent thermal noise is added to each data point
               opacitycal (bool): if False, time-dependent gaussian errors are added to station opacities
               ampcal (bool): if False, time-dependent gaussian errors are added to station gains
               phasecal (bool): if False, time-dependent station-based random phases are added to data points
               frcal (bool): if False, feed rotation angle terms are added to Jones matrices. Must have jones=True
               dcal (bool): if False, time-dependent gaussian errors added to Jones matrices D-terms. Must have jones=True
               stabilize_scan_phase (bool): if True, random phase errors are constant over scans
               stabilize_scan_amp (bool): if True, random amplitude errors are constant over scans
               jones (bool): if True, uses Jones matrix to apply mis-calibration effects (gains, phases, Dterms), otherwise uses old formalism without D-terms
               inv_jones (bool): if True, applies estimated inverse Jones matrix (not including random terms) to calibrate data

               tau (float): the base opacity at all sites, or a dict giving one opacity per site
               gain_offset (float): the base gain offset at all sites, or a dict giving one gain offset per site
               gainp (float): the fractional std. dev. of the random error on the gains
               taup (float): the fractional std. dev. of the random error on the opacities
               dtermp (float): the fractional std. dev. of the random error on the D-terms
               dterm_offset (float): the base dterm offset at all sites, or a dict giving one dterm offset per site

           Returns:
               (Obsdata): an observation object
        """


        # Generate empty observation
        print("Generating empty observation file . . . ")
        if mjd == None:
            mjd = self.mjd
        if polrep_obs is None:
            polrep_obs=self.polrep

        obs = array.obsdata(self.ra, self.dec, self.rf, bw, tint, tadv, tstart, tstop, mjd=mjd, polrep=polrep_obs,
                            tau=tau, timetype=timetype, elevmin=elevmin, elevmax=elevmax, fix_theta_GMST=fix_theta_GMST)


        # Observe on the same baselines as the empty observation and add noise
        obs = self.observe_same(obs, ttype=ttype, fft_pad_factor=fft_pad_factor, repeat=repeat,
                                     sgrscat=sgrscat, add_th_noise=add_th_noise,
                                     opacitycal=opacitycal,ampcal=ampcal,phasecal=phasecal,dcal=dcal,frcal=frcal,
                                     stabilize_scan_phase=stabilize_scan_phase, stabilize_scan_amp=stabilize_scan_amp,
                                     gainp=gainp,gain_offset=gain_offset,
                                     tau=tau, taup=taup,
                                     dtermp=dtermp, dterm_offset=dterm_offset,
                                     jones=jones, inv_jones=inv_jones)


        return obs

    def observe_vex(self, vex, source, synchronize_start=True, t_int=0.0,
                          polrep_obs=None, ttype='nfft', fft_pad_factor=2,
                          sgrscat=False, add_th_noise=True,
                          opacitycal=True, ampcal=True, phasecal=True, frcal=True, dcal=True,
                          stabilize_scan_phase=False, stabilize_scan_amp=False,
                          jones=False, inv_jones=False,
                          tau=TAUDEF, taup=GAINPDEF, gainp=GAINPDEF, gain_offset=GAINPDEF,
                          dtermp=DTERMPDEF, dterm_offset=DTERMPDEF, fix_theta_GMST=False):

        """Generate baselines from a vex file and observe the movie.

           Args:
               vex (Vex): an vex object containing sites and scan information
               source (str): the source to observe
               synchronize_start (bool): if True, the start of the movie will be defined to be the start of the observations

               t_int (float): if not zero, overrides the vex scans to produce visibilities for each t_int seconds

               polrep_obs (str): 'stokes' or 'circ' sets the data polarimetric representtion
               ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT
               fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in FFT

               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
               add_th_noise (bool): if True, baseline-dependent thermal noise is added to each data point
               opacitycal (bool): if False, time-dependent gaussian errors are added to station opacities
               ampcal (bool): if False, time-dependent gaussian errors are added to station gains
               phasecal (bool): if False, time-dependent station-based random phases are added to data points
               frcal (bool): if False, feed rotation angle terms are added to Jones matrices. Must have jones=True
               dcal (bool): if False, time-dependent gaussian errors added to Jones matrices D-terms. Must have jones=True
               stabilize_scan_phase (bool): if True, random phase errors are constant over scans
               stabilize_scan_amp (bool): if True, random amplitude errors are constant over scans
               jones (bool): if True, uses Jones matrix to apply mis-calibration effects (gains, phases, Dterms), otherwise uses old formalism without D-terms
               inv_jones (bool): if True, applies estimated inverse Jones matrix (not including random terms) to calibrate data

               tau (float): the base opacity at all sites, or a dict giving one opacity per site
               gain_offset (float): the base gain offset at all sites, or a dict giving one gain offset per site
               gainp (float): the fractional std. dev. of the random error on the gains
               taup (float): the fractional std. dev. of the random error on the opacities
               dterm_offset (float): the base dterm offset at all sites, or a dict giving one dterm offset per site
               dtermp (float): the fractional std. dev. of the random error on the D-terms

           Returns:
               (Obsdata): an observation object

        """

        if polrep_obs is None:
            polrep_obs=self.polrep

        obs_List=[]
        movie = self.copy()

        if synchronize_start:
            movie.mjd = vex.sched[0]['mjd_floor']
            movie.start_hr = vex.sched[0]['start_hr']

        movie_start = float(movie.mjd) + movie.start_hr/24.0
        movie_end   = movie_start + len(movie.frames)*movie.framedur/86400.

        print("Movie MJD Range: ",movie_start,movie_end)

        snapshot = 1.0
        if t_int > 0.0:
            snapshot = 0.0

        for i_scan in range(len(vex.sched)):
            if vex.sched[i_scan]['source'] != source:
                continue
            scankeys = list(vex.sched[i_scan]['scan'].keys())
            subarray = vex.array.make_subarray([vex.sched[i_scan]['scan'][key]['site'] for key in scankeys])

            if snapshot == 1.0:
                t_int = np.max(np.array([vex.sched[i_scan]['scan'][site]['scan_sec'] for site in scankeys]))
                print(t_int)

            vex_scan_start_mjd = float(vex.sched[i_scan]['mjd_floor']) + vex.sched[i_scan]['start_hr']/24.0
            vex_scan_stop_mjd  = vex_scan_start_mjd + vex.sched[i_scan]['scan'][0]['scan_sec']/3600.0/24.0

            print("Scan MJD Range: ",vex_scan_start_mjd,vex_scan_stop_mjd)

            if vex_scan_start_mjd < movie_start or vex_scan_stop_mjd > movie_end:
                continue

            t_start = vex.sched[i_scan]['start_hr']
            t_stop = vex.sched[i_scan]['start_hr'] + vex.sched[i_scan]['scan'][0]['scan_sec']/3600.0 - EP
            mjd = vex.sched[i_scan]['mjd_floor']
            obs = subarray.obsdata(movie.ra, movie.dec, movie.rf, vex.bw_hz, t_int, t_int, t_start, t_stop,
                                   mjd=mjd, polrep=polrep_obs, tau=tau,
                                   elevmin=.01, elevmax=89.99, timetype='UTC', fix_theta_GMST=fix_theta_GMST)
            obs_List.append(obs)

        if len(obs_List) == 0:
            raise Exception("Movie has no overlap with the vex file")

        obs = ehtim.obsdata.merge_obs(obs_List)

        obsout = movie.observe_same(obs, ttype=ttype, fft_pad_factor=fft_pad_factor, repeat=False,
                                    sgrscat=sgrscat, add_th_noise=add_th_noise,
                                    opacitycal=opacitycal, ampcal=ampcal, phasecal=phasecal, frcal=frcal, dcal=dcal,
                                    stabilize_scan_phase=stabilize_scan_phase, stabilize_scan_amp=stabilize_scan_amp,
                                    jones=jones, inv_jones=inv_jones,
                                    tau=tau, taup=taup,
                                    gainp=gainp, gain_offset=gain_offset,
                                    dtermp=dtermp, dterm_offset=dterm_offset)

        return obsout


    def save_txt(self, fname):
        """Save the Movie data to individual text files with filenames basename + 00001, etc.

           Args:
              fname (str): basename of output files

           Returns:
        """

        ehtim.io.save.save_mov_txt(self, fname)

        return

    def save_fits(self, fname):
        """Save the Movie data to individual fits files with filenames basename + 00001, etc.

           Args:
              fname (str): basename of output files

           Returns:
        """

        ehtim.io.save.save_mov_fits(self, fname)
        return

    def export_mp4(self, out='movie.mp4', fps=10, dpi=120,
                         interp='gaussian', scale='lin', dynamic_range=1000.0, cfun='afmhot',
                         nvec=20, pcut=0.01, plotp=False, gamma=0.5, frame_pad_factor=1, verbose=False):
        """Save the Movie to an mp4 file
        """

        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        im = im.switch_polrep('stokes','I')

        if (interp in ['gauss', 'gaussian', 'Gaussian', 'Gauss']):
            interp = 'gaussian'
        else:
            interp = 'nearest'

        if scale == 'lin':
            unit = 'Jy/pixel'
        elif scale == 'log':
            unit = 'log(Jy/pixel)'
        elif scale=='gamma':
            unit = '(Jy/pixel)^gamma'
        else:
            raise Exception("Scale not recognized!")

        fig = plt.figure()
        maxi = np.max(np.concatenate([im for im in self.frames]))

        if len(self.qframes) and plotp:
            thin = self.xdim//nvec
            mask = (self.frames[0]).reshape(self.ydim, self.xdim) > pcut * np.max(self.frames[0])
            mask2 = mask[::thin, ::thin]
            x = (np.array([[i for i in range(self.xdim)] for j in range(self.ydim)])[::thin, ::thin])[mask2]
            y = (np.array([[j for i in range(self.xdim)] for j in range(self.ydim)])[::thin, ::thin])[mask2]
            a = (-np.sin(np.angle(self.qframes[0]+1j*self.uframes[0])/2).reshape(self.ydim, self.xdim)[::thin, ::thin])[mask2]
            b = ( np.cos(np.angle(self.qframes[0]+1j*self.uframes[0])/2).reshape(self.ydim, self.xdim)[::thin, ::thin])[mask2]

            m = (np.abs(self.qframes[0] + 1j*self.uframes[0])/self.frames[0]).reshape(self.ydim, self.xdim)
            m[np.logical_not(mask)] = 0

            Q1 = plt.quiver(x, y, a, b,
                       headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                       width=.01*self.xdim, units='x', pivot='mid', color='k', angles='uv', scale=1.0/thin)
            Q2 = plt.quiver(x, y, a, b,
                       headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                       width=.005*self.xdim, units='x', pivot='mid', color='w', angles='uv', scale=1.1/thin)

        def im_data(n):

            n_data = int((n-n%frame_pad_factor)/frame_pad_factor)

            if len(self.qframes) and plotp:
                a = (-np.sin(np.angle(self.qframes[n_data]+1j*self.uframes[n_data])/2).reshape(self.ydim, self.xdim)[::thin, ::thin])[mask2]
                b = ( np.cos(np.angle(self.qframes[n_data]+1j*self.uframes[n_data])/2).reshape(self.ydim, self.xdim)[::thin, ::thin])[mask2]

                Q1.set_UVC(a,b)
                Q2.set_UVC(a,b)

            if scale == 'lin':
                return self.frames[n_data].reshape((self.ydim,self.xdim))
            elif scale == 'log':
                return np.log(self.frames[n_data].reshape((self.ydim,self.xdim)) + maxi/dynamic_range)
            elif scale=='gamma':
                return (self.frames[n_data]**(gamma)).reshape((self.ydim,self.xdim))

        plt_im = plt.imshow(im_data(0), cmap=plt.get_cmap(cfun), interpolation=interp)
        plt.colorbar(plt_im, fraction=0.046, pad=0.04, label=unit)

        if scale == 'lin':

            plt_im.set_clim([0,maxi])
        else:
            plt_im.set_clim([np.log(maxi/dynamic_range),np.log(maxi)])

        xticks = ticks(self.xdim, self.psize/RADPERAS/1e-6)
        yticks = ticks(self.ydim, self.psize/RADPERAS/1e-6)
        plt.xticks(xticks[0], xticks[1])
        plt.yticks(yticks[0], yticks[1])
        plt.xlabel('Relative RA ($\mu$as)')
        plt.ylabel('Relative Dec ($\mu$as)')

        fig.set_size_inches([5,5])
        plt.tight_layout()

        def update_img(n):
            if verbose:
                print("processing frame {0} of {1}".format(n, len(self.frames)*frame_pad_factor))
            plt_im.set_data(im_data(n))
            return plt_im

        ani = animation.FuncAnimation(fig,update_img,len(self.frames)*frame_pad_factor,interval=1e3/fps)
        writer = animation.writers['ffmpeg'](fps=fps, bitrate=1e6)
        ani.save(out,writer=writer,dpi=dpi)


##################################################################################################
# Movie creation functions
##################################################################################################
def merge_im_list(imlist, framedur=-1):
    """Merge a list of image objects into a movie object.

       Args:
           imlist (list): list of Image objects
           framedur (float): duration of a movie frame in seconds

       Returns:
           (Movie): a Movie object assembled from the images
    """
    framelist = []
    qlist = []
    ulist = []
    vlist = []
    nframes = len(imlist)

    print ("\nMerging %i frames from MJD %i %.2f hr to MJD %i %.2f hr"%(
            nframes,imlist[0].mjd,imlist[0].time, imlist[-1].mjd, imlist[-1].time))

    for i in range(nframes):
        im = imlist[i]
        if i==0:
            polrep0 = im.polrep
            pol_prim0 =  im.pol_prim
            movdict = {key:[] for key in  list(im._imdict.keys())}
            psize0 = im.psize
            xdim0 = im.xdim
            ydim0 = im.ydim
            ra0 = im.ra
            dec0 = im.dec
            rf0 = im.rf
            src0 = im.source
            mjd0 = im.mjd
            hour0 = im.time
            pulse = im.pulse
        else:
            if (im.polrep!=polrep0):
                raise Exception("polrep of image %i != polrep of image 0!" % i)
            if (im.psize!=psize0):
                raise Exception("psize of image %i != psize of image 0!" % i)
            if (im.xdim!=xdim0):
                raise Exception("xdim of image %i != xdim of image 0!" % i)
            if (im.ydim!=ydim0):
                raise Exception("ydim of image %i != ydim of image 0!" % i)
            if (im.ra!=ra0):
                raise Exception("RA of image %i != RA of image 0!" % i)
            if (im.dec!=dec0):
                raise Exception("DEC of image %i != DEC of image 0!" % i)
            if (im.rf!=rf0):
                raise Exception("rf of image %i != rf of image 0!" % i)
            if (im.source!=src0):
                raise Exception("source of image %i != src of image 0!" % i)
            if (im.mjd < mjd0):
                raise Exception("mjd of image %i < mjd of image 0!" % i)

            hour = im.time
            if im.mjd > mjd0:
                hour += 24*(im.mjd - mjd0)

        imarr = im.imvec.reshape(ydim0, xdim0)
        framelist.append(imarr)

        # Look for other  polarizations
        for pol in list(movdict.keys()):
            polvec = im._imdict[pol]
            if len(polvec):
                polarr = polvec.reshape(ydim0, xdim0)
                movdict[pol].append(polarr)
            else:
                if movdict[pol]:
                    raise Exception("all frames in merge_im_list must have the same pol layout: error in  frame %i"%i)

    if framedur == -1:
        framedur = ((hour - hour0)/float(nframes))*3600.0

    # Make new  movie with primary polarization
    newmov = Movie(framelist,
                   framedur, psize0, ra0, dec0,
                   polrep=polrep0, pol_prim=pol_prim0, start_hr=hour0,
                   rf=rf0, source=src0, mjd=mjd0, pulse=pulse)

    # Copy over all polarization movies
    for pol in list(movdict.keys()):
        if pol==newmov.pol_prim: continue
        polframes = movdict[pol]
        if len(polframes):
            newmov.add_pol_movie(polframes, pol)

    return newmov



def load_hdf5(file_name, framedur_sec=-1, psize=-1, ra=17.761122472222223, dec=-28.992189444444445, rf=230e9, source='SgrA',
              pulse=PULSE_DEFAULT, polrep='stokes', pol_prim=None,  zero_pol=True):

    """Read in a movie from an hdf5 file and create a Movie object.

       Args:
           file_name (str): The name of the hdf5 file.
           framedur_sec (float): The frame duration in seconds (default=-1, corresponding to framedur tahen from file header)
           psize (float): Pixel size in radian, (default=-1, corresponding to framedur taken from file header)
           ra (float): The movie right ascension
           dec (float): The movie declination
           rf (float): The movie frequency
           source (str) : The source name
           pulse (function): The function convolved with the pixel values for continuous image
           polrep (str): polarization representation, either 'stokes' or 'circ'
           pol_prim (str): The default image: I,Q,U or V for Stokes, RR,LL,LR,RL for Circular
           zero_pol (bool): If True, loads any missing polarizations as zeros

       Returns:
           Movie: a Movie object
    """

    return ehtim.io.load.load_movie_hdf5(file_name, framedur_sec=framedur_sec, psize=psize, ra=ra, dec=dec, rf=rf, source=source,
                                         pulse=pulse, polrep=polrep, pol_prim=pol_prim, zero_pol=zero_pol)


def load_txt(basename, nframes, framedur=-1, pulse=PULSE_DEFAULT, polrep='stokes', pol_prim=None,  zero_pol=True):
    """Read in a movie from text files and create a Movie object.

       Args:
           basename (str): The base name of individual movie frames. Files should have names basename + 00001, etc.
           nframes (int): The total number of frames
           framedur (float): The frame duration in seconds (default = -1, corresponding to framedur taken from file headers)
           pulse (function): The function convolved with the pixel values for continuous image
           polrep (str): polarization representation, either 'stokes' or 'circ'
           pol_prim (str): The default image: I,Q,U or V for Stokes, RR,LL,LR,RL for Circular
           zero_pol (bool): If True, loads any missing polarizations as zeros

       Returns:
           Movie: a Movie object
    """

    return ehtim.io.load.load_movie_txt(basename, nframes, framedur=framedur, pulse=pulse,
                                         polrep=polrep, pol_prim=pol_prim, zero_pol=zero_pol)


def load_fits(basename, nframes, framedur=-1, pulse=PULSE_DEFAULT, polrep='stokes', pol_prim=None,  zero_pol=True):
    """Read in a movie from fits files and create a Movie object.

       Args:
           basename (str): The base name of individual movie frames. Files should have names basename + 00001, etc.
           nframes (int): The total number of frames
           framedur (float): The frame duration in seconds (default = -1, corresponding to framedur taken from file headers)
           pulse (function): The function convolved with the pixel values for continuous image
           polrep (str): polarization representation, either 'stokes' or 'circ'
           pol_prim (str): The default image: I,Q,U or V for Stokes, RR,LL,LR,RL for Circular
           zero_pol (bool): If True, loads any missing polarizations as zeros

       Returns:
           Movie: a Movie object
    """

    return ehtim.io.load.load_movie_fits(basename, nframes, framedur=framedur, pulse=pulse,
                                         polrep=polrep, pol_prim=pol_prim, zero_pol=zero_pol)
