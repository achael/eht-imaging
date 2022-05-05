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

import string
import numpy as np
import scipy.interpolate
import scipy.ndimage.filters as filt

import ehtim.image
import ehtim.obsdata
import ehtim.observing.obs_simulate as simobs
import ehtim.io.save
import ehtim.io.load
import ehtim.const_def as ehc
import ehtim.observing.obs_helpers as obsh

INTERPOLATION_KINDS = ['linear', 'nearest', 'zero', 'slinear',
                       'quadratic', 'cubic', 'previous', 'next']

###################################################################################################
# Movie object
###################################################################################################


class Movie(object):

    """A polarimetric movie (in units of Jy/pixel).

       Attributes:
           pulse (function): The function convolved with the pixel values for continuous image
           psize (float): The pixel dimension in radians
           xdim (int): The number of pixels along the x dimension
           ydim (int): The number of pixels along the y dimension
           mjd (int): The integer MJD of the image
           source (str): The astrophysical source name
           ra (float): The source Right Ascension in fractional hours
           dec (float): The source declination in fractional degrees
           rf (float): The image frequency in Hz

           polrep (str): polarization representation, either 'stokes' or 'circ'
           pol_prim (str): The default image: I,Q,U or V for Stokes, or RR,LL,LR,RL for Circular
           interp (str): Interpolation method, for scipy.interpolate.interp1d 'kind' keyword
                         (e.g. 'linear', 'nearest', 'quadratic', 'cubic', 'previous', 'next'...)
           bounds_error (bool): if False, return nearest frame when outside [start_hr, stop_hr]


           times (list): The list of frame time stamps in hours

           _movdict (dict): The dictionary with the lists of frames
    """

    def __init__(self, frames, times, psize, ra, dec,
                 rf=ehc.RF_DEFAULT, polrep='stokes', pol_prim=None,
                 pulse=ehc.PULSE_DEFAULT, source=ehc.SOURCE_DEFAULT,
                 mjd=ehc.MJD_DEFAULT,
                 bounds_error=ehc.BOUNDS_ERROR, interp=ehc.INTERP_DEFAULT):
        """A polarimetric image (in units of Jy/pixel).

           Args:
               frames (list): The list of 2D frames; each is a Jy/pixel array
               times (list): The list of frame time stamps in hours
               psize (float): The pixel dimension in radians
               ra (float): The source Right Ascension in fractional hours
               dec (float): The source declination in fractional degrees
               rf (float): The image frequency in Hz

               polrep (str): polarization representation, either 'stokes' or 'circ'
               pol_prim (str): The default image: I,Q,U or V for Stokes, RR,LL,LR,RL for Circular
               pulse (function): The function convolved with the pixel values for continuous image.
               source (str): The source name
               mjd (int): The integer MJD of the image
               interp (str): Interpolation method, for scipy.interpolate.interp1d 'kind' keyword
                             (e.g. 'linear', 'nearest', 'quadratic', 'cubic', 'previous', 'next'...)
               bounds_error (bool): if False, return nearest frame when outside [start_hr, stop_hr]

         Returns:
               (Image): the Image object
        """

        if len(frames[0].shape) != 2:
            raise Exception("frames must each be a 2D numpy array")

        if len(frames) != len(times):
            raise Exception("len(frames) != len(times) !")

        if not (interp in INTERPOLATION_KINDS):
            raise Exception(
                "'interp' must be a valid argument for scipy.interpolate.interp1d: " +
                string.join(INTERPOLATION_KINDS))

        self.times = times
        start_hr = np.min(self.times)
        self.mjd = int(mjd)
        if start_hr > 24:
            self.mjd += int((start_hr - start_hr % 24)/24)
            self.start_hr = float(start_hr % 24)
        else:
            self.start_hr = start_hr
        self.stop_hr = np.max(self.times)
        self.duration = self.stop_hr - self.start_hr

        # frame shape parameters
        self.nframes = len(frames)
        self.polrep = polrep
        self.pulse = pulse
        self.psize = float(psize)
        self.xdim = frames[0].shape[1]
        self.ydim = frames[0].shape[0]

        # the list of frames
        frames = np.array([image.flatten() for image in frames])
        self.interp = interp
        self.bounds_error = bounds_error

        fill_value = (frames[0], frames[-1])
        fun = scipy.interpolate.interp1d(self.times, frames.T, kind=interp,
                                         bounds_error=bounds_error, fill_value=fill_value)

        if polrep == 'stokes':
            if pol_prim is None:
                pol_prim = 'I'
            if pol_prim == 'I':
                self._movdict = {'I': frames, 'Q': [], 'U': [], 'V': []}
                self._fundict = {'I': fun, 'Q': None, 'U': None, 'V': None}
            elif pol_prim == 'V':
                self._movdict = {'I': [], 'Q': [], 'U': [], 'V': frames}
                self._fundict = {'I': None, 'Q': None, 'U': None, 'V': fun}
            elif pol_prim == 'Q':
                self._movdict = {'I': [], 'Q': frames, 'U': [], 'V': []}
                self._fundict = {'I': None, 'Q': fun, 'U': None, 'V': None}
            elif pol_prim == 'U':
                self._movdict = {'I': [], 'Q': [], 'U': frames, 'V': []}
                self._fundict = {'I': None, 'Q': None, 'U': frames, 'V': None}
            else:
                raise Exception("for polrep=='stokes', pol_prim must be 'I','Q','U', or 'V'!")

        elif polrep == 'circ':
            if pol_prim is None:
                print("polrep is 'circ' and no pol_prim specified! Setting pol_prim='RR'")
                pol_prim = 'RR'
            if pol_prim == 'RR':
                self._movdict = {'RR': frames, 'LL': [], 'RL': [], 'LR': []}
                self._fundict = {'RR': fun, 'LL': None, 'RL': None, 'LR': None}
            elif pol_prim == 'LL':
                self._movdict = {'RR': [], 'LL': frames, 'RL': [], 'LR': []}
                self._fundict = {'RR': None, 'LL': fun, 'RL': None, 'LR': None}
            else:
                raise Exception("for polrep=='circ', pol_prim must be 'RR' or 'LL'!")

        self.pol_prim = pol_prim

        self.ra = float(ra)
        self.dec = float(dec)
        self.rf = float(rf)
        self.source = str(source)
        self.pa = 0.0  # TODO: The pa needs to be properly implemented in the movie object
        # TODO: What is this doing??

    @property
    def frames(self):
        frames = self._movdict[self.pol_prim]
        return frames

    @frames.setter
    def frames(self, frames):
        if len(frames[0]) != self.xdim*self.ydim:
            raise Exception("imvec size is not consistent with xdim*ydim!")
        # TODO -- more checks on consistency with the existing pol data???

        frames = np.array(frames)
        self._movdict[self.pol_prim] = frames

        fill_value = (frames[0], frames[-1])
        fun = scipy.interpolate.interp1d(self.times, frames.T, kind=self.interp,
                                         bounds_error=self.bounds_error, fill_value=fill_value)
        self._fundict[self.pol_prim] = fun

    @property
    def iframes(self):

        if self.polrep != 'stokes':
            raise Exception(
                "iframes is not defined unless self.polrep=='stokes' -- try self.switch_polrep()")

        frames = self._movdict['I']
        return frames

    @iframes.setter
    def iframes(self, frames):

        if len(frames[0]) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        # TODO -- more checks on the consistency of the imvec with the existing pol data???
        frames = np.array(frames)
        self._movdict['I'] = frames
        fill_value = (frames[0], frames[-1])
        fun = scipy.interpolate.interp1d(self.times, frames.T, kind=self.interp,
                                         bounds_error=self.bounds_error, fill_value=fill_value)
        self._fundict['I'] = fun

    @property
    def qframes(self):

        if self.polrep != 'stokes':
            raise Exception(
                "qframes is not defined unless self.polrep=='stokes' -- try self.switch_polrep()")

        frames = self._movdict['Q']
        return frames

    @qframes.setter
    def qframes(self, frames):

        if len(frames[0]) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        # TODO -- more checks on the consistency of the imvec with the existing pol data???
        frames = np.array(frames)
        self._movdict['Q'] = frames
        fill_value = (frames[0], frames[-1])
        fun = scipy.interpolate.interp1d(self.times, frames.T, kind=self.interp,
                                         bounds_error=self.bounds_error, fill_value=fill_value)
        self._fundict['Q'] = fun

    @property
    def uframes(self):

        if self.polrep != 'stokes':
            raise Exception(
                "uframes is not defined unless self.polrep=='stokes' -- try self.switch_polrep()")

        frames = self._movdict['U']

        return frames

    @uframes.setter
    def uframes(self, frames):

        if len(frames[0]) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        # TODO -- more checks on the consistency of the imvec with the existing pol data???
        frames = np.array(frames)
        self._movdict['U'] = frames
        fill_value = (frames[0], frames[-1])
        fun = scipy.interpolate.interp1d(self.times, frames.T, kind=self.interp,
                                         bounds_error=self.bounds_error, fill_value=fill_value)
        self._fundict['U'] = fun

    @property
    def vframes(self):

        if self.polrep != 'stokes':
            raise Exception(
                "vframes is not defined unless self.polrep=='stokes' -- try self.switch_polrep()")

        frames = self._movdict['V']

        return frames

    @vframes.setter
    def vframes(self, frames):

        if len(frames[0]) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        # TODO -- more checks on the consistency of the imvec with the existing pol data???
        frames = np.array(frames)
        self._movdict['V'] = frames
        fill_value = (frames[0], frames[-1])
        fun = scipy.interpolate.interp1d(self.times, frames.T, kind=self.interp,
                                         bounds_error=self.bounds_error, fill_value=fill_value)
        self._fundict['V'] = fun

    @property
    def rrframes(self):

        if self.polrep != 'circ':
            raise Exception(
                "rrframes is not defined unless self.polrep=='circ' -- try self.switch_polrep()")

        frames = self._movdict['RR']
        return frames

    @rrframes.setter
    def rrframes(self, frames):

        if len(frames[0]) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        # TODO -- more checks on the consistency of the imvec with the existing pol data???
        frames = np.array(frames)
        self._movdict['RR'] = frames
        fill_value = (frames[0], frames[-1])
        fun = scipy.interpolate.interp1d(self.times, frames.T, kind=self.interp,
                                         bounds_error=self.bounds_error, fill_value=fill_value)
        self._fundict['RR'] = fun

    @property
    def llframes(self):

        if self.polrep != 'circ':
            raise Exception(
                "llframes is not defined unless self.polrep=='circ' -- try self.switch_polrep()")

        frames = self._movdict['LL']
        return frames

    @llframes.setter
    def llframes(self, frames):

        if len(frames[0]) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        # TODO -- more checks on the consistency of the imvec with the existing pol data???
        frames = np.array(frames)
        self._movdict['LL'] = frames
        fill_value = (frames[0], frames[-1])
        fun = scipy.interpolate.interp1d(self.times, frames.T, kind=self.interp,
                                         bounds_error=self.bounds_error, fill_value=fill_value)
        self._fundict['LL'] = fun

    @property
    def rlvec(self):

        if self.polrep != 'circ':
            raise Exception(
                "rlframes is not defined unless self.polrep=='circ' -- try self.switch_polrep()")

        frames = self._movdict['RL']
        return frames

    @rlvec.setter
    def rlvec(self, frames):

        if len(frames[0]) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        # TODO -- more checks on the consistency of the imvec with the existing pol data???
        frames = np.array(frames)
        self._movdict['RL'] = frames
        fill_value = (frames[0], frames[-1])
        fun = scipy.interpolate.interp1d(self.times, frames.T, kind=self.interp,
                                         bounds_error=self.bounds_error, fill_value=fill_value)
        self._fundict['RL'] = fun

    @property
    def lrvec(self):

        if self.polrep != 'circ':
            raise Exception(
                "lrframes is not defined unless self.polrep=='circ' -- try self.switch_polrep()")

        frames = self._movdict['LR']
        return frames

    @lrvec.setter
    def lrvec(self, frames):

        if len(frames[0]) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        # TODO -- more checks on the consistency of the imvec with the existing pol data???
        frames = np.array(frames)
        self._movdict['LR'] = frames
        fill_value = (frames[0], frames[-1])
        fun = scipy.interpolate.interp1d(self.times, frames.T, kind=self.interp,
                                         bounds_error=self.bounds_error, fill_value=fill_value)
        self._fundict['LR'] = fun

    def movie_args(self):
        """"Copy arguments for making a  new Movie into a list and dictonary
        """

        frames2D = self.frames.reshape((self.nframes, self.ydim, self.xdim))
        arglist = [frames2D.copy(), self.times.copy(), self.psize, self.ra,  self.dec]
        #arglist = [frames2D, self.times, self.psize, self.ra,  self.dec]
        argdict = {'rf': self.rf, 'polrep': self.polrep,  'pol_prim': self.pol_prim,
                   'pulse': self.pulse, 'source': self.source,
                   'mjd': self.mjd, 'interp': self.interp, 'bounds_error': self.bounds_error}

        return (arglist, argdict)

    def copy(self):
        """Return a copy of the Movie object.

           Args:

           Returns:
               (Movie): copy of the Image.
        """

        arglist, argdict = self.movie_args()

        # Make new  movie with primary polarization
        newmov = Movie(*arglist, **argdict)

        # Copy over all polarization movies
        for pol in list(self._movdict.keys()):
            if pol == self.pol_prim:
                continue
            polframes = self._movdict[pol]
            if len(polframes):
                polframes = polframes.reshape((self.nframes, self.ydim, self.xdim))
                newmov.add_pol_movie(polframes, pol)

        return newmov

    def reset_interp(self, interp=None, bounds_error=None):
        """Reset the movie interpolation function to change the interp. type or change the frames

           Args:
              interp (str): Interpolation method, input to scipy.interpolate.interp1d kind keyword
              bounds_error (bool): if False, return nearest frame outside [start_hr, stop_hr]
        """

        if interp is None:
            interp = self.interp
        if bounds_error is None:
            bounds_error = self.bounds_error

        # Copy over all polarization movies
        for pol in list(self._movdict.keys()):
            polframes = self._movdict[pol]
            if len(polframes):
                fill_value = (polframes[0], polframes[-1])
                fun = scipy.interpolate.interp1d(self.times, polframes.T, kind=interp,
                                                 fill_value=fill_value, bounds_error=bounds_error)
                self._fundict[pol] = fun
            else:
                self._fundict[pol] = None

        self.interp = interp
        self.bounds_error = bounds_error
        return

    def offset_time(self, t_offset):
        """Offset the movie in time by t_offset

           Args:
               t_offset (float): offset time in hours
           Returns:

        """
        mov = self.copy()
        mov.start_hr += t_offset
        mov.stop_hr += t_offset
        mov.times += t_offset
        mov.reset_interp(interp=mov.interp, bounds_error=mov.bounds_error)
        return mov

    def add_pol_movie(self, movie, pol):
        """Add another movie polarization. f

           Args:
               movie (list): list of 2D frames (possibly complex) in a Jy/pixel array
               pol (str): The image type: 'I','Q','U','V' for stokes, 'RR','LL','RL','LR' for circ
        """
        if not(len(movie) == self.nframes):
            raise Exception("new pol movies must have same length as primary movie!")

        if pol == self.pol_prim:
            raise Exception("new pol in add_pol_movie is the same as pol_prim!")
        if np.any(np.array([image.shape != (self.ydim, self.xdim) for image in movie])):
            raise Exception("add_pol_movie image shapes incompatible with primary image!")
        if not (pol in list(self._movdict.keys())):
            raise Exception("for polrep==%s, pol in add_pol_movie must be in " %
                            self.polrep + ",".join(list(self._movdict.keys())))

        if self.polrep == 'stokes':
            if pol == 'I':
                self.iframes = [image.flatten() for image in movie]
            elif pol == 'Q':
                self.qframes = [image.flatten() for image in movie]
            elif pol == 'U':
                self.uframes = [image.flatten() for image in movie]
            elif pol == 'V':
                self.vframes = [image.flatten() for image in movie]

            if len(self.iframes) > 0:
                fill_value = (self.iframes[0], self.iframes[-1])
                ifun = scipy.interpolate.interp1d(self.times, self.iframes.T, kind=self.interp,
                                                  fill_value=fill_value,
                                                  bounds_error=self.bounds_error)
            else:
                ifun = None
            if len(self.vframes) > 0:
                fill_value = (self.vframes[0], self.vframes[-1])
                vfun = scipy.interpolate.interp1d(self.times, self.vframes.T, kind=self.interp,
                                                  fill_value=fill_value,
                                                  bounds_error=self.bounds_error)
            else:
                vfun = None
            if len(self.qframes) > 0:
                fill_value = (self.qframes[0], self.qframes[-1])
                qfun = scipy.interpolate.interp1d(self.times, self.qframes.T, kind=self.interp,
                                                  fill_value=fill_value,
                                                  bounds_error=self.bounds_error)
            else:
                qfun = None
            if len(self.uframes) > 0:
                fill_value = (self.uframes[0], self.uframes[-1])
                ufun = scipy.interpolate.interp1d(self.times, self.uframes.T, kind=self.interp,
                                                  fill_value=fill_value,
                                                  bounds_error=self.bounds_error)
            else:
                ufun = None

            self._movdict = {'I': self.iframes, 'Q': self.qframes,
                             'U': self.uframes, 'V': self.vframes}
            self._fundict = {'I': ifun, 'Q': qfun, 'U': ufun, 'V': vfun}

        elif self.polrep == 'circ':
            if pol == 'RR':
                self.rrframes = [image.flatten() for image in movie]
            elif pol == 'LL':
                self.llframes = [image.flatten() for image in movie]
            elif pol == 'RL':
                self.rlframes = [image.flatten() for image in movie]
            elif pol == 'LR':
                self.lrframes = [image.flatten() for image in movie]

            if len(self.rrframes) > 0:
                fill_value = (self.rrframes[0], self.rrframes[-1])
                rrfun = scipy.interpolate.interp1d(self.times, self.rrframes.T, kind=self.interp,
                                                   fill_value=fill_value,
                                                   bounds_error=self.bounds_error)
            else:
                rrfun = None
            if len(self.llframes) > 0:
                fill_value = (self.llframes[0], self.llframes[-1])
                llfun = scipy.interpolate.interp1d(self.times, self.llframes.T, kind=self.interp,
                                                   fill_value=fill_value,
                                                   bounds_error=self.bounds_error)
            else:
                llfun = None
            if len(self.rlframes) > 0:
                fill_value = (self.rlframes[0], self.rlframes[-1])
                rlfun = scipy.interpolate.interp1d(self.times, self.rlframes.T, kind=self.interp,
                                                   fill_value=fill_value,
                                                   bounds_error=self.bounds_error)
            else:
                rlfun = None
            if len(self.lrframes) > 0:
                fill_value = (self.lrframes[0], self.lrframes[-1])
                lrfun = scipy.interpolate.interp1d(self.times, self.lrframes.T, kind=self.interp,
                                                   fill_value=fill_value,
                                                   bounds_error=self.bounds_error)
            else:
                lrfun = None

            self._movdict = {'RR': self.rrframes, 'LL': self.llframes,
                             'RL': self.rlframes, 'LR': self.lrframes}
            self._fundict = {'RR': rrfun, 'LL': llfun, 'RL': rlfun, 'LR': lrfun}
        return

    # TODO deprecated -- replace with generic add_pol_movie
    def add_qu(self, qmovie, umovie):
        """Add Stokes Q and U movies. self.polrep must be 'stokes'

           Args:
               qmovie (list): list of 2D Stokes Q frames in Jy/pixel array
               umovie (list): list of 2D Stokes U frames in Jy/pixel array

           Returns:
        """

        if self.polrep != 'stokes':
            raise Exception("polrep must be 'stokes' for add_qu() !")
        self.add_pol_movie(qmovie, 'Q')
        self.add_pol_movie(umovie, 'U')

        return

    # TODO deprecated -- replace with generic add_pol_movie
    def add_v(self, vmovie):
        """Add Stokes V movie. self.polrep must be 'stokes'

           Args:
               vmovie (list): list of 2D Stokes Q frames in Jy/pixel array

           Returns:
        """

        if self.polrep != 'stokes':
            raise Exception("polrep must be 'stokes' for add_v() !")
        self.add_pol_movie(vmovie, 'V')

        return

    def switch_polrep(self, polrep_out='stokes', pol_prim_out=None):
        """Return a new movie with the polarization representation changed

           Args:
               polrep_out (str):  the polrep of the output data
               pol_prim_out (str): The default movie: I,Q,U or V for Stokes,
                                   RR,LL,LR,RL for Circular

           Returns:
               (Movie): new movie object with potentially different polrep
        """

        if polrep_out not in ['stokes', 'circ']:
            raise Exception("polrep_out must be either 'stokes' or 'circ'")
        if pol_prim_out is None:
            if polrep_out == 'stokes':
                pol_prim_out = 'I'
            elif polrep_out == 'circ':
                pol_prim_out = 'RR'

        # Simply copy if the polrep is unchanged
        if polrep_out == self.polrep and pol_prim_out == self.pol_prim:
            return self.copy()

        # Assemble a dictionary of new polarization vectors
        framedim = (self.nframes, self.ydim, self.xdim)
        if polrep_out == 'stokes':
            if self.polrep == 'stokes':
                movdict = {'I': self.iframes, 'Q': self.qframes,
                           'U': self.uframes, 'V': self.vframes}
            else:
                if len(self.rrframes) == 0 or len(self.llframes) == 0:
                    iframes = []
                    vframes = []
                else:
                    iframes = 0.5*(self.rrframes.reshape(framedim) +
                                   self.llframes.reshape(framedim))
                    vframes = 0.5*(self.rrframes.reshape(framedim) -
                                   self.llframes.reshape(framedim))

                if len(self.rlframes) == 0 or len(self.lrframes) == 0:
                    qframes = []
                    uframes = []
                else:
                    qframes = np.real(0.5*(self.lrframes.reshape(framedim) +
                                           self.rlframes.reshape(framedim)))
                    uframes = np.real(0.5j*(self.lrframes.reshape(framedim) -
                                            self.rlframes.reshape(framedim)))

                movdict = {'I': iframes, 'Q': qframes, 'U': uframes, 'V': vframes}

        elif polrep_out == 'circ':
            if self.polrep == 'circ':
                movdict = {'RR': self.rrframes, 'LL': self.llframes,
                           'RL': self.rlframes, 'LR': self.lrframes}
            else:
                if len(self.iframes) == 0 or len(self.vframes) == 0:
                    rrframes = []
                    llframes = []
                else:
                    rrframes = (self.iframes.reshape(framedim) + self.vframes.reshape(framedim))
                    llframes = (self.iframes.reshape(framedim) - self.vframes.reshape(framedim))

                if len(self.qframes) == 0 or len(self.uframes) == 0:
                    rlframes = []
                    lrframes = []
                else:
                    rlframes = (self.qframes.reshape(framedim) + 1j*self.uframes.reshape(framedim))
                    lrframes = (self.qframes.reshape(framedim) - 1j*self.uframes.reshape(framedim))

                movdict = {'RR': rrframes, 'LL': llframes, 'RL': rlframes, 'LR': lrframes}

        # Assemble the new movie
        frames = movdict[pol_prim_out]
        if len(frames) == 0:
            raise Exception("switch_polrep to " +
                            "%s with pol_prim_out=%s, \n" % (polrep_out, pol_prim_out) +
                            "output movie is not defined")

        # Make new  movie with primary polarization
        arglist, argdict = self.movie_args()
        arglist[0] = frames
        argdict['polrep'] = polrep_out
        argdict['pol_prim'] = pol_prim_out
        newmov = Movie(*arglist, **argdict)

        # Add in any other polarizations
        for pol in list(movdict.keys()):
            if pol == pol_prim_out:
                continue
            polframes = movdict[pol]
            if len(polframes):
                polframes = polframes.reshape((self.nframes, self.ydim, self.xdim))
                newmov.add_pol_movie(polframes, pol)

        return newmov

    def flip_chi(self):
        """Flip between the different conventions for measuring the EVPA (E of N vs N of E).

           Args:

           Returns:
               (Image): movie with flipped EVPA
        """

        mov = self.copy()
        if mov.polrep == 'stokes':
            mov.qframes *= [-qvec for qvec in mov.qframes]

        elif mov.polrep == 'circ':
            mov.lrframes *= [-np.conjugate(lrvec) for lrvec in mov.lrframes]
            mov.rlframes *= [-np.conjugate(rlvec) for rlvec in mov.rlframes]

        return mov

    def orth_chi(self):
        """Rotate the EVPA 90 degrees

           Args:

           Returns:
               (Image): movie with rotated EVPA
        """
        mov = self.copy()
        if mov.polrep == 'stokes':
            mov.qframes *= [-uvec for uvec in mov.vframes]

        elif mov.polrep == 'circ':
            mov.lrframes *= [np.conjugate(lrvec) for lrvec in mov.lrframes]
            mov.rlframes *= [np.conjugate(rlvec) for rlvec in mov.rlframes]

        return mov

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

    @property
    def lightcurve(self):
        """Return the total flux over time of the image in Jy.

           Args:

           Returns:
                (numpy.Array) :  image total flux (Jy) over time
        """
        if self.polrep == 'stokes':
            flux = [np.sum(ivec) for ivec in self.iframes]
        elif self.polrep == 'circ':
            flux = [0.5*(np.sum(self.rrframes[i])+np.sum(self.llframes[i]))
                    for i in range(self.nframes)]

        return np.array(flux)

    def lin_polfrac_curve(self):
        """Return the total fractional linear polarized flux over time

           Args:

           Returns:
                (numpy.ndarray) : image fractional linear polarized flux per frame
        """
        if self.polrep == 'stokes':
            frac = [np.abs(np.sum(self.qframes[i] + 1j*self.uframes[i])) /
                    np.abs(np.sum(self.iframes[i]))
                    for i in range(self.nframes)]
        elif self.polrep == 'circ':
            frac = [2*np.abs(np.sum(self.rlframes[i])) /
                    np.abs(np.sum(self.rrframes[i]+self.llframes[i]))
                    for i in range(self.nframes)]
        return np.array(frac)

    def circ_polfrac_curve(self):
        """Return the (signed) total fractional circular polarized flux over time

           Args:

           Returns:
                (numpy.ndarray) : image fractional circular polarized flux per frame
        """
        if self.polrep == 'stokes':
            frac = [np.sum(self.vframes[i]) / np.abs(np.sum(self.iframes[i]))
                    for i in range(self.nframes)]
        elif self.polrep == 'circ':
            frac = [np.sum(self.rrframes[i]-self.llframes[i]) /
                    np.abs(np.sum(self.rrframes[i] + self.llframes[i]))
                    for i in range(self.nframes)]

        return np.array(frac)

    def get_image(self, time):
        """Return an Image at time

           Args:
               time (float): the time  in hours

           Returns:
               (Image): the Image object at the given time
        """

        if (time < self.start_hr):
            if not(self.bounds_error):
                pass
                # print ("time %f before movie start time %f" % (time, self.start_hr))
                # print ("returning constant frame 0! \n")
            else:
                raise Exception("time %f must be in the range %f - %f" %
                                (time, self.start_hr, self.stop_hr))

        if (time > self.stop_hr):
            if not(self.bounds_error):
                pass
                # print ("time %f after movie stop time %f" % (time, self.stop_hr))
                # print ("returning constant frame -1! \n")
            else:
                raise Exception("time %f must be in the range %f - %f" %
                                (time, self.start_hr, self.stop_hr))

        # interpolate the imvec to the given time
        imvec = self._fundict[self.pol_prim](time)

        # Make the primary image
        imarr = imvec.reshape(self.ydim, self.xdim)
        outim = ehtim.image.Image(imarr, self.psize, self.ra, self.dec, self.pa,
                                  polrep=self.polrep, pol_prim=self.pol_prim, time=time,
                                  rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Copy over the rest of the polarizations
        for pol in list(self._movdict.keys()):
            if pol == self.pol_prim:
                continue
            polframes = self._movdict[pol]
            if len(polframes):
                polvec = self._fundict[pol](time)
                polarr = polvec.reshape(self.ydim, self.xdim).copy()
                outim.add_pol_image(polarr, pol)

        return outim

    def get_frame(self, n):
        """Return an Image of the nth frame

           Args:
               n (int): the frame number

           Returns:
               (Image): the Image object of the nth frame
        """

        if n < 0 or n >= len(self.frames):
            raise Exception("n must be in the range 0 - %i" % self.nframes)

        time = self.times[n]

        # Make the primary image
        imarr = self.frames[n].reshape(self.ydim, self.xdim)
        outim = ehtim.image.Image(imarr, self.psize, self.ra, self.dec, self.pa,
                                  polrep=self.polrep, pol_prim=self.pol_prim, time=time,
                                  rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Copy over the rest of the polarizations
        for pol in list(self._movdict.keys()):
            if pol == self.pol_prim:
                continue
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

        # Make the primary image
        avg_imvec = np.mean(np.array(self.frames), axis=0)
        avg_imarr = avg_imvec.reshape(self.ydim, self.xdim)
        outim = ehtim.image.Image(avg_imarr, self.psize, self.ra, self.dec, self.pa,
                                  polrep=self.polrep, pol_prim=self.pol_prim, time=self.start_hr,
                                  rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Copy over the rest of the average polarizations
        for pol in list(self._movdict.keys()):
            if pol == self.pol_prim:
                continue
            polframes = self._movdict[pol]
            if len(polframes):
                avg_polvec = np.mean(np.array(polframes), axis=0)
                avg_polarr = avg_polvec.reshape(self.ydim, self.xdim)
                outim.add_pol_image(avg_polarr, pol)

        return outim

    def blur_circ(self, fwhm_x, fwhm_t, fwhm_x_pol=0):
        """Apply a Gaussian filter to a list of images.

           Args:
               fwhm_x (float): circular beam size for spatial blurring in radians
               fwhm_t (float): temporal blurring in frames
               fwhm_x_pol (float): circular beam size for Stokes Q,U,V spatial blurring in radians
           Returns:
               (Image): output image list
        """

        # Unpack the frames
        frames = self.im_list()

        # Blur Stokes I
        sigma_x = fwhm_x / self.psize / (2. * np.sqrt(2. * np.log(2.)))
        sigma_t = fwhm_t / (2. * np.sqrt(2. * np.log(2.)))
        sigma_x_pol = fwhm_x_pol / self.psize / (2. * np.sqrt(2. * np.log(2.)))

        arr = np.array([im.imvec.reshape(self.ydim, self.xdim) for im in frames])
        arr = filt.gaussian_filter(arr, (sigma_t, sigma_x, sigma_x))

        # Make a new blurred movie
        arglist, argdict = self.movie_args()
        arglist[0] = arr
        movie_blur = Movie(*arglist, **argdict)

        # Process the remaining polarizations
        for pol in list(self._movdict.keys()):
            if pol == self.pol_prim:
                continue
            polframes = self._movdict[pol]

            if len(polframes):
                arr = np.array([imvec.reshape(self.ydim, self.xdim) for imvec in polframes])
                arr = filt.gaussian_filter(arr, (sigma_t, sigma_x_pol, sigma_x_pol))
                movie_blur.add_pol_movie(arr, pol)

        return movie_blur

    def observe_same_nonoise(self, obs, repeat=False, sgrscat=False, 
                             ttype="nfft", fft_pad_factor=2, 
                             zero_empty_pol=True, verbose=True):
        """Observe the movie on the same baselines as an existing observation
           without adding noise.

           Args:
               obs (Obsdata): existing observation with baselines where the FT will be sampled
               repeat (bool): if True, repeat the movie to fill up the observation interval
               sgrscat (bool): if True, the visibilites are blurred by the Sgr A* scattering kernel
               ttype (str): if "fast", use FFT to produce visibilities. Else "direct" for DTFT
               fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in FFT

               zero_empty_pol (bool): if True, returns zero vec if the polarization doesn't exist.
                                      Otherwise return None
               verbose (bool): Boolean value controls output prints.

           Returns:
               (Obsdata): an observation object
        """

        # Check for agreement in coordinates and frequency
        tolerance = 1e-8
        if (np.abs(self.ra - obs.ra) > tolerance) or (np.abs(self.dec - obs.dec) > tolerance):
            raise Exception("Movie coordinates are not the same as observtion coordinates!")
        if (np.abs(self.rf - obs.rf)/obs.rf > tolerance):
            raise Exception("Movie frequency is not the same as observation frequency!")

        if ttype == 'direct' or ttype == 'fast' or ttype == 'nfft':
            print("Producing clean visibilities from movie with " + ttype + " FT . . . ")
        else:
            raise Exception("ttype=%s, options for ttype are 'direct', 'fast', 'nfft'" % ttype)

        # Get data
        obslist = obs.tlist()

        obstimes = np.array([obsdata[0]['time'] for obsdata in obslist])

        if (obstimes < self.start_hr).any():
            if repeat:
                print("Some observation times before movie start time %f" % self.start_hr)
                print("Looping movie before start\n")
            elif not(self.bounds_error):
                print("Some observation times before movie start time %f" % self.start_hr)
                print("bounds_error is  False:  using constant frame 0 before start_hr! \n")
            else:
                raise Exception("Some observation times before movie start time %f" % self.start_hr)
        if (obstimes > self.stop_hr).any():
            if repeat:
                print("Some observation times after movie stop time %f" % self.stop_hr)
                print("Looping movie after stop\n")
            elif not(self.bounds_error):
                print("Some observation times after movie stop time %f" % self.stop_hr)
                print("bounds_error is  False:  using constant frame -1 after stop_hr! \n")
            else:
                raise Exception("Some observation times after movie stop time %f" % self.stop_hr)

        # Observe nearest frame
        obsdata_out = []

        for i in range(len(obslist)):
            obsdata = obslist[i]

            # Frame number
            time = obsdata[0]['time']

            if self.bounds_error:
                if (time < self.start_hr or time > self.stop_hr):
                    if repeat:
                        time = self.start_hr + np.mod(time - self.start_hr, self.duration)
                    else:
                        raise Exception("Obs time %f outside movie range %f--%f" %
                                        (time, self.start_hr, self.stop_hr))

            # Get the frame visibilities
            uv = obsh.recarr_to_ndarr(obsdata[['u', 'v']], 'f8')

            try:
                im = self.get_image(time)
            except ValueError:
                raise Exception("Interpolation error for time %f: movie range %f--%f" %
                                (time, self.start_hr, self.stop_hr))

            data = simobs.sample_vis(im, uv, sgrscat=sgrscat, polrep_obs=obs.polrep,
                                     ttype=ttype, fft_pad_factor=fft_pad_factor,
                                     zero_empty_pol=zero_empty_pol, verbose=verbose)
            verbose = False # only print for one frame

            # Put visibilities into the obsdata
            if obs.polrep == 'stokes':
                obsdata['vis'] = data[0]
                if not(data[1] is None):
                    obsdata['qvis'] = data[1]
                    obsdata['uvis'] = data[2]
                    obsdata['vvis'] = data[3]

            elif obs.polrep == 'circ':
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
        obs_no_noise = ehtim.obsdata.Obsdata(self.ra, self.dec, self.rf, obs.bw,
                                             obsdata_out, obs.tarr,
                                             source=self.source, mjd=np.floor(obs.mjd),
                                             polrep=obs.polrep,
                                             ampcal=True, phasecal=True, opacitycal=True,
                                             dcal=True, frcal=True,
                                             timetype=obs.timetype, scantable=obs.scans)

        return obs_no_noise

    def observe_same(self, obs_in, repeat=False,
                     ttype='nfft', fft_pad_factor=2,
                     sgrscat=False, add_th_noise=True,
                     jones=False, inv_jones=False,
                     opacitycal=True, ampcal=True, phasecal=True,
                     frcal=True, dcal=True, rlgaincal=True,
                     stabilize_scan_phase=False, stabilize_scan_amp=False,
                     neggains=False,
                     taup=ehc.GAINPDEF,
                     gain_offset=ehc.GAINPDEF, gainp=ehc.GAINPDEF, 
                     dterm_offset=ehc.DTERMPDEF,
                     rlratio_std=0.,rlphase_std=0.,
                     caltable_path=None, seed=False, sigmat=None, verbose=True):
        """Observe the image on the same baselines as an existing observation object and add noise.

           Args:
               obs_in (Obsdata): existing observation with baselines where the FT will be sampled
               repeat (bool): if True, repeat the movie to fill up the observation interval
               ttype (str):  "fast" or "nfft" or "direct"
               fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in FFT

               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A*  kernel
               add_th_noise (bool): if True, baseline-dependent thermal noise is added

               jones (bool): if True, uses Jones matrix to apply mis-calibration effects
               inv_jones (bool): if True, applies estimated inverse Jones matrix
                                 (not including random terms) to a priori calibrate data
               opacitycal (bool): if False, time-dependent gaussian errors are added to opacities
               ampcal (bool): if False, time-dependent gaussian errors are added to station gains
               phasecal (bool): if False, time-dependent station-based random phases are added
               frcal (bool): if False, feed rotation angle terms are added to Jones matrices.
               dcal (bool): if False, time-dependent gaussian errors added to D-terms.
               rlgaincal (bool): if False, time-dependent gains are not equal for R and L pol
               stabilize_scan_phase (bool): if True, random phase errors are constant over scans
               stabilize_scan_amp (bool): if True, random amplitude errors are constant over scans
               neggains (bool): if True, force the applied gains to be <1

               taup (float): the fractional std. dev. of the random error on the opacities
               gainp (float): the fractional std. dev. of the random error on the gains
                              or a dict giving one std. dev. per site      

               gain_offset (float): the base gain offset at all sites,
                                    or a dict giving one gain offset per site
               dterm_offset (float): the base std. dev. of random additive error at all sites,
                                    or a dict giving one std. dev. per site

               rlratio_std (float): the fractional std. dev. of the R/L gain offset
                                    or a dict giving one std. dev. per site                                          
               rlphase_std (float): std. dev. of R/L phase offset, 
                                    or a dict giving one std. dev. per site
                                    a negative value samples from uniform                                          
                                    
               caltable_path (string): If not None, path and prefix for saving the applied caltable
               seed (int): seeds the random component of the noise terms. DO NOT set to 0!
               sigmat (float): temporal std for a Gaussian Process used to generate gain noise.
                               if sigmat=None then an iid gain noise is applied.
               verbose (bool): print updates and warnings

            Returns:
               (Obsdata): an observation object

        """

        if seed:
            np.random.seed(seed=seed)

        # print("Producing clean visibilities from movie . . . ")
        obs = self.observe_same_nonoise(obs_in, repeat=repeat, sgrscat=sgrscat,
                                        ttype=ttype, fft_pad_factor=fft_pad_factor,
                                        zero_empty_pol=True, verbose=verbose)

        # Jones Matrix Corruption & Calibration
        if jones:
            obsdata = simobs.add_jones_and_noise(obs, add_th_noise=add_th_noise,
                                                 opacitycal=opacitycal, ampcal=ampcal,
                                                 phasecal=phasecal, frcal=frcal, dcal=dcal,
                                                 rlgaincal=rlgaincal,
                                                 stabilize_scan_phase=stabilize_scan_phase,
                                                 stabilize_scan_amp=stabilize_scan_amp,
                                                 neggains=neggains,
                                                 taup=taup,
                                                 gain_offset=gain_offset, gainp=gainp,
                                                 dterm_offset=dterm_offset,
                                                 rlratio_std=rlratio_std, rlphase_std=rlphase_std,
                                                 caltable_path=caltable_path,
                                                 seed=seed, sigmat=sigmat, verbose=verbose)

            obs = ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata, obs.tarr,
                                        source=obs.source, mjd=obs.mjd, polrep=obs_in.polrep,
                                        ampcal=ampcal, phasecal=phasecal, opacitycal=opacitycal,
                                        dcal=dcal, frcal=frcal,
                                        timetype=obs.timetype, scantable=obs.scans)

            if inv_jones:
                obsdata = simobs.apply_jones_inverse(obs,
                                                     opacitycal=opacitycal, dcal=dcal, frcal=frcal,
                                                     verbose=verbose)

                obs = ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata, obs.tarr,
                                            source=obs.source, mjd=obs.mjd, polrep=obs_in.polrep,
                                            ampcal=ampcal, phasecal=phasecal,
                                            opacitycal=True, dcal=True, frcal=True,
                                            timetype=obs.timetype, scantable=obs.scans)

        # No Jones Matrices, Add noise the old way
        # TODO There is an asymmetry here - in the old way, we don't offer the ability to
        # *not* unscale estimated noise.
        else:
            if caltable_path:
                print('WARNING: the caltable is only saved if you apply noise with a Jones Matrix')

            obsdata = simobs.add_noise(obs, add_th_noise=add_th_noise,
                                       opacitycal=opacitycal, ampcal=ampcal, phasecal=phasecal, 
                                       stabilize_scan_phase=stabilize_scan_phase,
                                       stabilize_scan_amp=stabilize_scan_amp,
                                       neggains=neggains,
                                       taup=taup, gain_offset=gain_offset, gainp=gainp,
                                       caltable_path=caltable_path, seed=seed, sigmat=sigmat,
                                       verbose=verbose)

            obs = ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata, obs.tarr,
                                        source=obs.source, mjd=obs.mjd, polrep=obs_in.polrep,
                                        ampcal=ampcal, phasecal=phasecal,
                                        opacitycal=True, dcal=True, frcal=True,
                                        timetype=obs.timetype, scantable=obs.scans)

        return obs

    def observe(self, array, tint, tadv, tstart, tstop, bw, repeat=False,
                mjd=None, timetype='UTC', polrep_obs=None,
                elevmin=ehc.ELEV_LOW, elevmax=ehc.ELEV_HIGH,
                ttype='nfft', fft_pad_factor=2, fix_theta_GMST=False,
                sgrscat=False, add_th_noise=True,
                jones=False, inv_jones=False,
                opacitycal=True, ampcal=True, phasecal=True,
                frcal=True, dcal=True, rlgaincal=True,
                stabilize_scan_phase=False, stabilize_scan_amp=False,
                neggains=False,
                tau=ehc.TAUDEF, taup=ehc.GAINPDEF,
                gain_offset=ehc.GAINPDEF, gainp=ehc.GAINPDEF, 
                dterm_offset=ehc.DTERMPDEF,
                rlratio_std=0.,rlphase_std=0.,
                caltable_path=None, seed=False, sigmat=None, verbose=True):
        """Generate baselines from an array object and observe the movie.

           Args:
               array (Array): an array object containing sites with which to generate baselines
               tint (float): the scan integration time in seconds
               tadv (float): the uniform cadence between scans in seconds
               tstart (float): the start time of the observation in hours
               tstop (float): the end time of the observation in hours
               bw (float): the observing bandwidth in Hz
               repeat (bool): if True, repeat the movie to fill up the observation interval

               mjd (int): the mjd of the observation, if set as different from the image mjd
               timetype (str): how to interpret tstart and tstop; either 'GMST' or 'UTC'
               polrep_obs (str): 'stokes' or 'circ' sets the data polarimetric representation
               elevmin (float): station minimum elevation in degrees
               elevmax (float): station maximum elevation in degrees

               ttype (str): "fast", "nfft" or "dtft"
               fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in the FFT
               fix_theta_GMST (bool): if True, stops earth rotation to sample fixed u,v

               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A*  kernel
               add_th_noise (bool): if True, baseline-dependent thermal noise is added

               jones (bool): if True, uses Jones matrix to apply mis-calibration effects
                             otherwise uses old formalism without D-terms
               inv_jones (bool): if True, applies estimated inverse Jones matrix
                                 (not including random terms) to calibrate data

               opacitycal (bool): if False, time-dependent gaussian errors are added to opacities
               ampcal (bool): if False, time-dependent gaussian errors are added to station gains
               phasecal (bool): if False, time-dependent station-based random phases are added
               frcal (bool): if False, feed rotation angle terms are added to Jones matrix.
               dcal (bool): if False, time-dependent gaussian errors added to Jones matrix D-terms.
               rlgaincal (bool): if False, time-dependent gains are not equal for R and L pol
               stabilize_scan_phase (bool): if True, random phase errors are constant over scans
               stabilize_scan_amp (bool): if True, random amplitude errors are constant over scans
               neggains (bool): if True, force the applied gains to be <1

               tau (float): the base opacity at all sites, or a dict giving one opacity per site
               taup (float): the fractional std. dev. of the random error on the opacities
               gainp (float): the fractional std. dev. of the random error on the gains
                              or a dict giving one std. dev. per site      

               gain_offset (float): the base gain offset at all sites,
                                    or a dict giving one gain offset per site
               dterm_offset (float): the base std. dev. of random additive error at all sites,
                                    or a dict giving one std. dev. per site

               rlratio_std (float): the fractional std. dev. of the R/L gain offset
                                    or a dict giving one std. dev. per site                                          
               rlphase_std (float): std. dev. of R/L phase offset, 
                                    or a dict giving one std. dev. per site
                                    a negative value samples from uniform                                          


               caltable_path (string): If not None, path and prefix for saving the applied caltable
               seed (int): seeds the random component of the noise terms. DO NOT set to 0!
               sigmat (float): temporal std for a Gaussian Process used to generate gain noise.
                               if sigmat=None then an iid gain noise is applied.
               verbose (bool): print updates and warnings

           Returns:
               (Obsdata): an observation object

        """

        # Generate empty observation
        print("Generating empty observation file . . . ")
        if mjd is None:
            mjd = self.mjd
        if polrep_obs is None:
            polrep_obs = self.polrep

        obs = array.obsdata(self.ra, self.dec, self.rf, bw, tint, tadv, tstart, tstop,
                            mjd=mjd, polrep=polrep_obs,
                            tau=tau, timetype=timetype,
                            elevmin=elevmin, elevmax=elevmax,
                            fix_theta_GMST=fix_theta_GMST)

        # Observe on the same baselines as the empty observation and add noise
        obs = self.observe_same(obs, repeat=repeat, 
                                ttype=ttype, fft_pad_factor=fft_pad_factor,
                                sgrscat=sgrscat,
                                add_th_noise=add_th_noise,
                                jones=jones, inv_jones=inv_jones,
                                opacitycal=opacitycal, ampcal=ampcal,
                                phasecal=phasecal, dcal=dcal,
                                frcal=frcal, rlgaincal=rlgaincal,
                                stabilize_scan_phase=stabilize_scan_phase,
                                stabilize_scan_amp=stabilize_scan_amp,
                                neggains=neggains,
                                taup=taup,
                                gain_offset=gain_offset, gainp=gainp, 
                                dterm_offset=dterm_offset,
                                rlratio_std=rlratio_std,rlphase_std=rlphase_std,
                                caltable_path=caltable_path, seed=seed, sigmat=sigmat,
                                verbose=verbose)

        return obs

    def observe_vex(self, vex, source, synchronize_start=True, t_int=0.0,
                    polrep_obs=None, ttype='nfft', fft_pad_factor=2,
                    fix_theta_GMST=False, 
                    sgrscat=False, add_th_noise=True,
                    jones=False, inv_jones=False,
                    opacitycal=True, ampcal=True, phasecal=True,
                    frcal=True, dcal=True, rlgaincal=True,
                    stabilize_scan_phase=False, stabilize_scan_amp=False,
                    neggains=False,
                    tau=ehc.TAUDEF, taup=ehc.GAINPDEF,
                    gain_offset=ehc.GAINPDEF, gainp=ehc.GAINPDEF, 
                    dterm_offset=ehc.DTERMPDEF,
                    caltable_path=None, seed=False, sigmat=None, verbose=True):
        """Generate baselines from a vex file and observe the movie.

           Args:
               vex (Vex): an vex object containing sites and scan information
               source (str): the source to observe
               synchronize_start (bool): if True, the start of the movie is defined
                                         as the start of the observations
               t_int (float): if not zero, overrides the vex scan lengths

               polrep_obs (str): 'stokes' or 'circ' sets the data polarimetric representation
               ttype (str): "fast" or "nfft" or "dtft"
               fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in FFT
               fix_theta_GMST (bool): if True, stops earth rotation to sample fixed u,v

               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A*  kernel
               add_th_noise (bool): if True, baseline-dependent thermal noise is added

               jones (bool): if True, uses Jones matrix to apply mis-calibration effects
                             otherwise uses old formalism without D-terms
               inv_jones (bool): if True, applies estimated inverse Jones matrix
                                 (not including random terms) to calibrate data
               opacitycal (bool): if False, time-dependent gaussian errors are added to opacities
               ampcal (bool): if False, time-dependent gaussian errors are added to station gains
               phasecal (bool): if False, time-dependent station-based random phases are added
               frcal (bool): if False, feed rotation angle terms are added to Jones matrix.
               dcal (bool): if False, time-dependent gaussian errors added to Jones matrix D-terms.
               rlgaincal (bool): if False, time-dependent gains are not equal for R and L pol
               stabilize_scan_phase (bool): if True, random phase errors are constant over scans
               stabilize_scan_amp (bool): if True, random amplitude errors are constant over scans
               neggains (bool): if True, force the applied gains to be <1

               tau (float): the base opacity at all sites,
                            or a dict giving one opacity per site
               taup (float): the fractional std. dev. of the random error on the opacities
               gain_offset (float): the base gain offset at all sites,
                                    or a dict giving one gain offset per site
               gainp (float): the fractional std. dev. of the random error on the gains
               dterm_offset (float): the base dterm offset at all sites,
                                     or a dict giving one dterm offset per site

               caltable_path (string): If not None, path and prefix for saving the applied caltable
               seed (int): seeds the random component of the noise terms. DO NOT set to 0!
               sigmat (float): temporal std for a Gaussian Process used to generate gain noise.
                               if sigmat=None then an iid gain noise is applied.
               verbose (bool): print updates and warnings

           Returns:
               (Obsdata): an observation object

        """

        if polrep_obs is None:
            polrep_obs = self.polrep

        obs_List = []
        movie = self.copy()

        if synchronize_start:
            movie.mjd = vex.sched[0]['mjd_floor']
            movie.start_hr = vex.sched[0]['start_hr']

        movie_start = float(movie.mjd) + movie.start_hr/24.0
        movie_end = float(movie.mjd) + movie.stop_hr/24.0

        print("Movie MJD Range: ", movie_start, movie_end)

        snapshot = 1.0
        if t_int > 0.0:
            snapshot = 0.0

        for i_scan in range(len(vex.sched)):
            if vex.sched[i_scan]['source'] != source:
                continue
            scankeys = list(vex.sched[i_scan]['scan'].keys())
            subarray = vex.array.make_subarray([vex.sched[i_scan]['scan'][key]['site']
                                                for key in scankeys])

            if snapshot == 1.0:
                t_int = np.max(np.array([vex.sched[i_scan]['scan'][site]
                                         ['scan_sec'] for site in scankeys]))
                print(t_int)

            vex_scan_start_mjd = float(vex.sched[i_scan]['mjd_floor'])
            vex_scan_start_mjd += vex.sched[i_scan]['start_hr']/24.0

            vex_scan_length_mjd = vex.sched[i_scan]['scan'][0]['scan_sec']/3600.0/24.0
            vex_scan_stop_mjd = vex_scan_start_mjd + vex_scan_length_mjd

            print("Scan MJD Range: ", vex_scan_start_mjd, vex_scan_stop_mjd)

            if vex_scan_start_mjd < movie_start or vex_scan_stop_mjd > movie_end:
                continue

            t_start = vex.sched[i_scan]['start_hr']
            t_stop = t_start + vex.sched[i_scan]['scan'][0]['scan_sec']/3600.0 - ehc.EP

            mjd = vex.sched[i_scan]['mjd_floor']
            obs = subarray.obsdata(movie.ra, movie.dec, movie.rf, vex.bw_hz,
                                   t_int, t_int, t_start, t_stop,
                                   mjd=mjd, polrep=polrep_obs, tau=tau,
                                   elevmin=.01, elevmax=89.99, timetype='UTC',
                                   fix_theta_GMST=fix_theta_GMST)
            obs_List.append(obs)

        if len(obs_List) == 0:
            raise Exception("Movie has no overlap with the vex file")

        obs = ehtim.obsdata.merge_obs(obs_List)

        obsout = movie.observe_same(obs, repeat=False,
                                    ttype=ttype, fft_pad_factor=fft_pad_factor,
                                    sgrscat=sgrscat, add_th_noise=add_th_noise,
                                    jones=jones, inv_jones=inv_jones,
                                    opacitycal=opacitycal, ampcal=ampcal, phasecal=phasecal,
                                    frcal=frcal, dcal=dcal, rlgaincal=rlgaincal,
                                    stabilize_scan_phase=stabilize_scan_phase,
                                    stabilize_scan_amp=stabilize_scan_amp,
                                    neggains=neggains,
                                    taup=taup,
                                    gain_offset=gain_offset, gainp=gainp,
                                    dterm_offset=dterm_offset,
                                    caltable_path=caltable_path, seed=seed,sigmat=sigmat,
                                    verbose=verbose)

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

    def save_hdf5(self, fname):
        """Save the Movie data to a single hdf5 file.

           Args:
              fname (str): output file name

           Returns:
        """

        ehtim.io.save.save_mov_hdf5(self, fname)
        return

    def export_mp4(self, out='movie.mp4', fps=10, dpi=120,
                   interp='gaussian', scale='lin', dynamic_range=1000.0, cfun='afmhot',
                   nvec=20, pcut=0.01, plotp=False, gamma=0.5, frame_pad_factor=1,
                   label_time=False, verbose=False):
        """Save the Movie to an mp4 file
        """

        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        if self.polrep != 'stokes':
            raise Exception("export_mp4 requires self.polrep=='stokes' -- try self.switch_polrep()")

        if (interp in ['gauss', 'gaussian', 'Gaussian', 'Gauss']):
            interp = 'gaussian'
        else:
            interp = 'nearest'

        if scale == 'lin':
            unit = 'Jy/pixel'
        elif scale == 'log':
            unit = 'log(Jy/pixel)'
        elif scale == 'gamma':
            unit = '(Jy/pixel)^gamma'
        else:
            raise Exception("Scale not recognized!")

        fig = plt.figure()
        maxi = np.max(np.concatenate([im for im in self.frames]))

        if len(self.qframes) and plotp:
            thin = self.xdim//nvec
            mask = (self.frames[0]).reshape(self.ydim, self.xdim) > pcut * np.max(self.frames[0])
            mask2 = mask[::thin, ::thin]
            x = (np.array([[i for i in range(self.xdim)]
                           for j in range(self.ydim)])[::thin, ::thin])[mask2]
            y = (np.array([[j for i in range(self.xdim)]
                           for j in range(self.ydim)])[::thin, ::thin])[mask2]
            a = (-np.sin(np.angle(self.qframes[0]+1j*self.uframes[0]) /
                         2).reshape(self.ydim, self.xdim)[::thin, ::thin])[mask2]
            b = (np.cos(np.angle(self.qframes[0]+1j*self.uframes[0]) /
                        2).reshape(self.ydim, self.xdim)[::thin, ::thin])[mask2]

            m = (np.abs(self.qframes[0] + 1j*self.uframes[0]) /
                 self.frames[0]).reshape(self.ydim, self.xdim)
            m[np.logical_not(mask)] = 0

            Q1 = plt.quiver(x, y, a, b,
                            headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                            width=.01*self.xdim, units='x', pivot='mid', color='k',
                            angles='uv', scale=1.0/thin)
            Q2 = plt.quiver(x, y, a, b,
                            headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                            width=.005*self.xdim, units='x', pivot='mid', color='w',
                            angles='uv', scale=1.1/thin)

        def im_data(n):

            n_data = int((n-n % frame_pad_factor)/frame_pad_factor)

            if len(self.qframes) and plotp:
                a = (-np.sin(np.angle(self.qframes[n_data]+1j*self.uframes[n_data]
                                      )/2).reshape(self.ydim, self.xdim)[::thin, ::thin])[mask2]
                b = (np.cos(np.angle(self.qframes[n_data]+1j*self.uframes[n_data]
                                     )/2).reshape(self.ydim, self.xdim)[::thin, ::thin])[mask2]

                Q1.set_UVC(a, b)
                Q2.set_UVC(a, b)

            if scale == 'lin':
                return self.frames[n_data].reshape((self.ydim, self.xdim))
            elif scale == 'log':
                return np.log(self.frames[n_data].reshape(
                    (self.ydim, self.xdim)) + maxi/dynamic_range)
            elif scale == 'gamma':
                return (self.frames[n_data]**(gamma)).reshape((self.ydim, self.xdim))

        plt_im = plt.imshow(im_data(0), cmap=plt.get_cmap(cfun), interpolation=interp)
        plt.colorbar(plt_im, fraction=0.046, pad=0.04, label=unit)

        if scale == 'lin':

            plt_im.set_clim([0, maxi])
        else:
            plt_im.set_clim([np.log(maxi/dynamic_range), np.log(maxi)])

        xticks = obsh.ticks(self.xdim, self.psize/ehc.RADPERAS/1e-6)
        yticks = obsh.ticks(self.ydim, self.psize/ehc.RADPERAS/1e-6)
        plt.xticks(xticks[0], xticks[1])
        plt.yticks(yticks[0], yticks[1])
        plt.xlabel(r'Relative RA ($\mu$as)')
        plt.ylabel(r'Relative Dec ($\mu$as)')

        fig.set_size_inches([5, 5])
        plt.tight_layout()

        def update_img(n):
            if verbose:
                print("processing frame {0} of {1}".format(n, len(self.frames)*frame_pad_factor))
            plt_im.set_data(im_data(n))

            if label_time:
                time = self.times[n]
                time_str = ("%02d:%02d:%02d" % (int(time), (time*60) % 60, (time*3600) % 60))
                fig.suptitle(time_str)

            return plt_im

        ani = animation.FuncAnimation(fig, update_img, len(
            self.frames)*frame_pad_factor, interval=1e3/fps)
        writer = animation.writers['ffmpeg'](fps=fps, bitrate=1e6)
        ani.save(out, writer=writer, dpi=dpi)

##################################################################################################
# Movie creation functions
##################################################################################################


def export_multipanel_mp4(input_list, out='movie.mp4', start_hr=None, stop_hr=None, nframes=100,
                          fov=None, npix=None,
                          nrows=1, fps=10, dpi=120, verbose=False, titles=None,
                          panel_size=4.0, common_scale=False, scale='linear', label_type='scale',
                          has_cbar=False, **kwargs):
    """Export a movie comparing multiple movies in a grid.

       Args:
            input_list (list): The  list of  input Movies or Images
            out (string): The output filename
            start_hr (float): The start time in hours. If None, defaults to first start time
            end_hr (float): The end time in hours. If None, defaults to last start time
            nframes (int): The number of frames in the output movie
            fov (float): If specified, use this field of view for all panels
            npix (int): If specified, use this linear pixel dimension for all panels
            nrows (int): Number of rows in movie
            fps (int): Frames per second
            titles (list): List of panel titles for input_list
            panel_size (float): Size of individual panels (inches)

    """
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    if start_hr is None:
        try:
            start_hr = np.min([x.start_hr for x in input_list if hasattr(x, 'start_hr')])
        except ValueError:
            raise Exception("no movies in input_list!")

    if stop_hr is None:
        try:
            stop_hr = np.max([x.stop_hr for x in input_list if hasattr(x, 'stop_hr')])
        except ValueError:
            raise Exception("no movies in input_list!")

    print("%s will have %i frames in the  range %f-%f hr" % (out, nframes, start_hr, stop_hr))

    ncols = int(np.ceil(len(input_list)/nrows))
    suptitle_space = 0.6  # inches
    w = panel_size*ncols
    h = panel_size*nrows + suptitle_space
    tgap = suptitle_space / h
    bgap = .1
    rgap = .1
    lgap = .1
    subw = (1-lgap-rgap)/ncols
    subh = (1-tgap-bgap)/nrows
    print("Rows: " + str(nrows))
    print("Cols: " + str(ncols))

    fig = plt.figure(figsize=(w, h))
    ax_all = [[] for j in range(nrows)]
    for y in range(nrows):
        for x in range(ncols):
            ax = fig.add_axes([lgap+subw*x, bgap+subh*(nrows-y-1), subw, subh])
            ax_all[y].append(ax)

    times = np.linspace(start_hr, stop_hr, nframes)
    hr_step = times[1]-times[0]
    mjd_step = hr_step/24.

    im_List_Set = [[x.get_image(time) if hasattr(x, 'get_image') else x.copy() for time in times]
                   for x in input_list]

    if fov and npix:
        im_List_Set = [[x.regrid_image(fov, npix) for x in y] for y in im_List_Set]
    else:
        print('not rescaling images to common fov and npix!')

    maxi = [np.max([im.imvec for im in im_List_Set[j]]) for j in range(len(im_List_Set))]
    if common_scale:
        maxi = np.max(maxi) + 0.0*maxi

    i = 0
    for y in range(nrows):
        for x in range(ncols):
            if i >= len(im_List_Set):
                ax_all[y][x].set_visible(False)
            else:
                kwargs.get('ttype', 'nfft')
                if (y == nrows-1 and x == 0) or fov is None:
                    label_type_cur = label_type
                else:
                    label_type_cur = 'none'

                im_List_Set[i][0].display(axis=ax_all[y][x], scale=scale,
                                          label_type=label_type_cur, has_cbar=has_cbar, **kwargs)
                if y == nrows-1 and x == 0:
                    plt.xlabel(r'Relative RA ($\mu$as)')
                    plt.ylabel(r'Relative Dec ($\mu$as)')
                else:
                    plt.xlabel('')
                    plt.ylabel('')
                if not titles:
                    ax_all[y][x].set_title('')
                else:
                    ax_all[y][x].set_title(titles[i])
            i = i+1

    def im_data(i, n):
        if scale == 'linear':
            return im_List_Set[i][n].imvec.reshape((im_List_Set[i][n].ydim, im_List_Set[i][n].xdim))
        else:
            return np.log(im_List_Set[i][n].imvec.reshape(
                (im_List_Set[i][n].ydim, im_List_Set[i][n].xdim)) + 1e-20)

    def update_img(n):
        if verbose:
            print("processing frame {0} of {1}".format(n, len(im_List_Set[0])))
        i = 0
        for y in range(nrows):
            for x in range(ncols):
                ax_all[y][x].images[0].set_data(im_data(i, n))
                i = i+1
                if i >= len(im_List_Set):
                    break

        if mjd_step > 0.1:
            # , verticalalignment=verticalalignment)
            fig.suptitle('MJD: ' + str(im_List_Set[0][n].mjd))
        else:
            time = im_List_Set[0][n].time
            time_str = ("%d:%02d.%02d" % (int(time), (time*60) % 60, (time*3600) % 60))
            fig.suptitle(time_str)

        return

    ani = animation.FuncAnimation(fig, update_img, len(im_List_Set[0]), interval=1e3/fps)
    writer = animation.writers['ffmpeg'](fps=fps, bitrate=1e6)
    ani.save(out, writer=writer, dpi=dpi)


def merge_im_list(imlist, framedur=-1, interp=ehc.INTERP_DEFAULT, bounds_error=ehc.BOUNDS_ERROR):
    """Merge a list of image objects into a movie object.

       Args:
           imlist (list): list of Image objects
           framedur (float): duration of a movie frame in seconds
                             use to override times in the individual movies
           interp (str): Interpolation method, input to scipy.interpolate.interp1d kind keyword
           bounds_error (bool): if False, return nearest frame outside interval [start_hr, stop_hr]

        Returns:
           (Movie): a Movie object assembled from the images
    """
    framelist = []
    nframes = len(imlist)

    print("\nMerging %i frames from MJD %i %.2f hr to MJD %i %.2f hr" % (
        nframes, imlist[0].mjd, imlist[0].time, imlist[-1].mjd, imlist[-1].time))

    for i in range(nframes):
        im = imlist[i]
        if i == 0:
            polrep0 = im.polrep
            pol_prim0 = im.pol_prim
            movdict = {key: [] for key in list(im._imdict.keys())}
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
            times = [hour0]
        else:
            if (im.polrep != polrep0):
                raise Exception("polrep of image %i != polrep of image 0!" % i)
            if (im.psize != psize0):
                raise Exception("psize of image %i != psize of image 0!" % i)
            if (im.xdim != xdim0):
                raise Exception("xdim of image %i != xdim of image 0!" % i)
            if (im.ydim != ydim0):
                raise Exception("ydim of image %i != ydim of image 0!" % i)
            if (im.ra != ra0):
                raise Exception("RA of image %i != RA of image 0!" % i)
            if (im.dec != dec0):
                raise Exception("DEC of image %i != DEC of image 0!" % i)
            if (im.rf != rf0):
                raise Exception("rf of image %i != rf of image 0!" % i)
            if (im.source != src0):
                raise Exception("source of image %i != src of image 0!" % i)
            if (im.mjd < mjd0):
                raise Exception("mjd of image %i < mjd of image 0!" % i)

            hour = im.time
            if im.mjd > mjd0:
                hour += 24*(im.mjd - mjd0)
            times.append(hour)

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
                    raise Exception("all frames in merge_im_list must have the same pol layout: " +
                                    "error in  frame %i" % i)

    # assume equispaced with a given framedur instead of reading the individual image times
    if framedur != -1:
        framedur_hr = framedur/3600.
        tstart = hour0
        tstop = hour0 + framedur_hr*nframes
        times = np.linspace(tstart, tstop, nframes)

    elif len(set(times)) < len(framelist):
        raise Exception("image times have duplicates!")

    # Make new  movie with primary polarization
    newmov = Movie(framelist, times,
                   psize0, ra0, dec0, interp=interp, bounds_error=bounds_error,
                   polrep=polrep0, pol_prim=pol_prim0,
                   rf=rf0, source=src0, mjd=mjd0, pulse=pulse)

    # Copy over all polarization movies
    for pol in list(movdict.keys()):
        if pol == newmov.pol_prim:
            continue
        polframes = np.array(movdict[pol])
        if len(polframes):
            polframes = polframes.reshape((newmov.nframes, newmov.ydim, newmov.xdim))
            newmov.add_pol_movie(polframes, pol)

    return newmov


def load_hdf5(file_name,
              pulse=ehc.PULSE_DEFAULT, interp=ehc.INTERP_DEFAULT, bounds_error=ehc.BOUNDS_ERROR):
    """Read in a movie from an hdf5 file and create a Movie object.

       Args:
           file_name (str): The name of the hdf5 file.
           pulse (function): The function convolved with the pixel values for continuous image
           interp (str): Interpolation method, input to scipy.interpolate.interp1d kind keyword
           bounds_error (bool): if False, return nearest frame outside interval [start_hr, stop_hr]

       Returns:
           Movie: a Movie object
    """

    return ehtim.io.load.load_movie_hdf5(file_name, pulse=pulse, interp=interp,
                                         bounds_error=bounds_error)


def load_txt(basename, nframes,
             framedur=-1, pulse=ehc.PULSE_DEFAULT,
             polrep='stokes', pol_prim=None,  zero_pol=True,
             interp=ehc.INTERP_DEFAULT, bounds_error=ehc.BOUNDS_ERROR):
    """Read in a movie from text files and create a Movie object.

       Args:
           basename (str): The base name of individual movie frames.
                           Files should have names basename + 00001, etc.
           nframes (int): The total number of frames
           framedur (float): The frame duration in seconds
                             if famedur==-1, frame duration taken from file headers
           pulse (function): The function convolved with the pixel values for continuous image
           polrep (str): polarization representation, either 'stokes' or 'circ'
           pol_prim (str): The default image: I,Q,U or V for Stokes, RR,LL,LR,RL for Circular
           zero_pol (bool): If True, loads any missing polarizations as zeros
           interp (str): Interpolation method, input to scipy.interpolate.interp1d kind keyword
           bounds_error (bool): if False, return nearest frame outside interval [start_hr, stop_hr]

       Returns:
           Movie: a Movie object
    """

    return ehtim.io.load.load_movie_txt(basename, nframes, framedur=framedur, pulse=pulse,
                                        polrep=polrep, pol_prim=pol_prim, zero_pol=zero_pol,
                                        interp=interp, bounds_error=bounds_error)


def load_fits(basename, nframes,
              framedur=-1, pulse=ehc.PULSE_DEFAULT,
              polrep='stokes', pol_prim=None,  zero_pol=True,
              interp=ehc.INTERP_DEFAULT, bounds_error=ehc.BOUNDS_ERROR):
    """Read in a movie from fits files and create a Movie object.

       Args:
           basename (str): The base name of individual movie frames.
                           Files should have names basename + 00001, etc.
           nframes (int): The total number of frames
           framedur (float): The frame duration in seconds.
                             if famedur==-1, frame duration taken from file headers
           pulse (function): The function convolved with the pixel values for continuous image
           polrep (str): polarization representation, either 'stokes' or 'circ'
           pol_prim (str): The default image: I,Q,U or V for Stokes, RR,LL,LR,RL for Circular
           zero_pol (bool): If True, loads any missing polarizations as zeros
           interp (str): Interpolation method, input to scipy.interpolate.interp1d kind keyword
           bounds_error (bool): if False, return nearest frame outside interval [start_hr, stop_hr]

       Returns:
           Movie: a Movie object
    """

    return ehtim.io.load.load_movie_fits(basename, nframes, framedur=framedur, pulse=pulse,
                                         polrep=polrep, pol_prim=pol_prim, zero_pol=zero_pol,
                                         interp=interp, bounds_error=bounds_error)
