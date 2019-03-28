# image.py
# an interferometric image class
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
import numpy.matlib as matlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.signal
import scipy.ndimage.filters as filt
import scipy.interpolate
from scipy import ndimage as ndi
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks

import ehtim.observing.obs_simulate as simobs
import ehtim.observing.pulses
import ehtim.io.save
import ehtim.io.load

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

#import emcee
#TODO : add time to all images
#TODO : add arbitrary center location

###########################################################################################################################################
#Image object
###########################################################################################################################################

class Image(object):
    """A polarimetric image (in units of Jy/pixel).

       Attributes:
           pulse (function): The function convolved with the pixel values for continuous image.
           psize (float): The pixel dimension in radians
           xdim (int): The number of pixels along the x dimension
           ydim (int): The number of pixels along the y dimension
           mjd (int): The integer MJD of the image
           time (float): The observing time of the image (UTC hours)
           source (str): The astrophysical source name
           ra (float): The source Right Ascension in fractional hours
           dec (float): The source declination in fractional degrees
           rf (float): The image frequency in Hz

           imvec (array): The vector of pixel I values in Jy/pixel (len xdim*ydim)
           polrep (str): polarization representation, either 'stokes' or 'circ'
           pol_prim (str): The default image: I,Q,U or V for Stokes, or RR,LL,LR,RL for Circular

    """

    def __init__(self, image, psize, ra, dec, pa=0.0,
                       polrep='stokes', pol_prim=None,
                       rf=RF_DEFAULT, pulse=PULSE_DEFAULT, source=SOURCE_DEFAULT,
                       mjd=MJD_DEFAULT, time=0.):

        """A polarimetric image (in units of Jy/pixel).

           Args:
               image (numpy.array): The 2D intensity values in a Jy/pixel array
               polrep (str): polarization representation, either 'stokes' or 'circ'
               pol_prim (str): The default image: I,Q,U or V for Stokes, RR,LL,LR,RL for Circular

               psize (float): The pixel dimension in radians
               ra (float): The source Right Ascension in fractional hours
               dec (float): The source declination in fractional degrees
               pa (float): logical positional angle of the image
               rf (float): The image frequency in Hz
               pulse (function): The function convolved with the pixel values for continuous image.
               source (str): The source name
               mjd (int): The integer MJD of the image
               time (float): The observing time of the image (UTC hours)

           Returns:
               (Image): the Image object
        """

        if len(image.shape) != 2:
            raise Exception("image must be a 2D numpy array")
        if not polrep in ['stokes','circ']:
            raise Exception("only 'stokes' and 'circ' are supported polreps!")

        # Save the image vector
        imvec = image.flatten()

        if polrep=='stokes':
            if pol_prim is None: pol_prim = 'I'
            if pol_prim=='I':
                self._imdict = {'I':imvec,'Q':[],'U':[],'V':[]}
            elif pol_prim=='V':
                self._imdict = {'I':[],'Q':[],'U':[],'V':imvec}
            elif pol_prim=='Q':
                self._imdict = {'I':[],'Q':imvec,'U':[],'V':[]}
            elif pol_prim=='U':
                self._imdict = {'I':[],'Q':[],'U':imvec,'V':[]}
            else:
                raise Exception("for polrep=='stokes', pol_prim must be 'I','Q','U', or 'V'!")

        elif polrep=='circ':
            if pol_prim is None:
                print("polrep is 'circ' and no pol_prim specified! Setting pol_prim='RR'")
                pol_prim = 'RR'
            if pol_prim=='RR':
                self._imdict = {'RR':imvec,'LL':[],'RL':[],'LR':[]}
            elif pol_prim=='LL':
                self._imdict = {'RR':[],'LL':imvec,'RL':[],'LR':[]}
            else:
                raise Exception("for polrep=='circ', pol_prim must be 'RR' or 'LL'!")
        else:
            raise Excpetion("polrep must be 'circ' or 'stokes'!")

        # Save the image dimension data
        self.pol_prim =  pol_prim
        self.polrep = polrep
        self.pulse = pulse
        self.psize = float(psize)
        self.xdim = image.shape[1]
        self.ydim = image.shape[0]

        # Save the image metadata
        self.ra  = float(ra)
        self.dec = float(dec)
        self.pa  = float(pa)
        self.rf  = float(rf)
        self.source = str(source)
        self.mjd = int(mjd)

        # Cached FFT of the image
        self.cached_fft = {}

        if time > 24:
            self.mjd += int((time - time % 24)/24)
            self.time = float(time % 24)
        else:
            self.time = time

    @property
    def imvec(self):
        imvec = self._imdict[self.pol_prim]
        return imvec

    @imvec.setter
    def imvec(self, vec):
        if len(vec) != self.xdim*self.ydim:
            raise Exception("imvec size is not consistent with xdim*ydim!")
        #TODO -- more checks on the consistency of the imvec with the existing pol data???
        self._imdict[self.pol_prim] =  vec

    @property
    def ivec(self):
        if self.polrep!='stokes':
            raise Exception("ivec is not defined unless self.polrep=='stokes' -- try self.switch_polrep()")

        ivec = self._imdict['I']
        return ivec

    @ivec.setter
    def ivec(self, vec):
        if len(vec) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        #TODO -- more checks on the consistency of the imvec with the existing pol data???
        self._imdict['I'] =  vec

    @property
    def qvec(self):
        if self.polrep!='stokes':
            raise Exception("qvec is not defined unless self.polrep=='stokes' -- try self.switch_polrep()")

        qvec = self._imdict['Q']
        return qvec

    @qvec.setter
    def qvec(self, vec):
        if len(vec) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        #TODO -- more checks on the consistency of the imvec with the existing pol data???
        self._imdict['Q'] =  vec

    @property
    def uvec(self):
        if self.polrep!='stokes':
            raise Exception("uvec is not defined unless self.polrep=='stokes' -- try self.switch_polrep()")

        uvec = self._imdict['U']
        return uvec

    @uvec.setter
    def uvec(self, vec):
        if len(vec) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        #TODO -- more checks on the consistency of the imvec with the existing pol data???
        self._imdict['U'] =  vec

    @property
    def vvec(self):
        if self.polrep!='stokes':
            raise Exception("vvec is not defined unless self.polrep=='stokes' -- try self.switch_polrep()")

        vvec = self._imdict['V']
        return vvec

    @vvec.setter
    def vvec(self, vec):
        if len(vec) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        #TODO -- more checks on the consistency of the imvec with the existing pol data???
        self._imdict['V'] =  vec

    @property
    def rrvec(self):
        if self.polrep!='circ':
            raise Exception("rrvec is not defined unless self.polrep=='circ' -- try self.switch_polrep()")

        rrvec = self._imdict['RR']
        return rrvec

    @rrvec.setter
    def rrvec(self, vec):
        if len(vec) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        #TODO -- more checks on the consistency of the imvec with the existing pol data???
        self._imdict['RR'] =  vec

    @property
    def llvec(self):
        if self.polrep!='circ':
            raise Exception("rrvec is not defined unless self.polrep=='circ' -- try self.switch_polrep()")

        llvec = self._imdict['LL']
        return llvec

    @llvec.setter
    def llvec(self, vec):
        if len(vec) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        #TODO -- more checks on the consistency of the imvec with the existing pol data???
        self._imdict['LL'] =  vec

    @property
    def rlvec(self):
        if self.polrep!='circ':
            raise Exception("rlvec is not defined unless self.polrep=='circ' -- try self.switch_polrep()")

        rlvec = self._imdict['RL']
        return rlvec

    @rlvec.setter
    def rlvec(self, vec):
        if len(vec) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        #TODO -- more checks on the consistency of the imvec with the existing pol data???
        self._imdict['RL'] =  vec

    @property
    def lrvec(self):
        """Return the imvec of LR"""
        if self.polrep!='circ':
            raise Exception("lrvec is not defined unless self.polrep=='circ' -- try self.switch_polrep()")

        lrvec = self._imdict['LR']
        return lrvec

    @lrvec.setter
    def lrvec(self, vec):
        """Set the imvec"""

        if len(vec) != self.xdim*self.ydim:
            raise Exception("vec size is not consistent with xdim*ydim!")

        #TODO -- more checks on the consistency of the imvec with the existing pol data???
        self._imdict['LR'] =  vec

    def copy(self):

        """Return a copy of the Image object.

           Args:

           Returns:
               (Image): copy of the Image.
        """

        # Make new  image with primary polarization
        newim = Image(self.imvec.reshape(self.ydim,self.xdim), self.psize, self.ra, self.dec, self.pa,
                      polrep=self.polrep, pol_prim=self.pol_prim, time=self.time,
                      rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Copy over all polarization images
        newim.copy_pol_images(self)

        return newim

    def copy_pol_images(self, old_image):

        """Copy polarization images from old_image over to self.

           Args:
               old_image (Image): image object to copy from

        """

        for pol in list(self._imdict.keys()):

            if (pol==self.pol_prim):
                continue

            polvec = old_image._imdict[pol]
            if len(polvec):
                self.add_pol_image(polvec.reshape(self.ydim,self.xdim), pol)

    def add_pol_image(self, image, pol):

        """Add another image polarization.

           Args:
               image (list): 2D image frame (possibly complex) in a Jy/pixel array
               pol (str): The image type: 'I','Q','U','V' for stokes, 'RR','LL','RL','LR' for circ
        """

        if pol==self.pol_prim:
            raise Exception("new pol in add_pol_image is the same as pol_prim!")
        if image.shape != (self.ydim, self.xdim):
            raise Exception("add_pol_image image shapes incompatible with primary image!")
        if not (pol in list(self._imdict.keys())):
            raise Exception("for polrep==%s, pol in add_pol_image in "%self.polrep + ",".join(list(self._imdict.keys())))

        if self.polrep=='stokes':
            if pol=='I': self.ivec = image.flatten()
            elif pol=='Q': self.qvec = image.flatten()
            elif pol=='U': self.uvec = image.flatten()
            elif pol=='V': self.vvec = image.flatten()

        elif self.polrep=='circ':
            if pol=='RR': self.rrvec = image.flatten()
            elif pol=='LL': self.llvec = image.flatten()
            elif pol=='RL': self.rlvec = image.flatten()
            elif pol=='LR': self.lrvec = image.flatten()

        return

    # TODO deprecated -- replace with generic add_pol_image
    def add_qu(self, qimage, uimage):

        """Add Stokes Q and U images. self.polrep must be 'stokes'

           Args:
               qimage (numpy.array): The 2D Stokes Q values in Jy/pixel array
               uimage (numpy.array): The 2D Stokes U values in Jy/pixel array

           Returns:
        """

        if self.polrep!='stokes':
            raise Excpetion("polrep must be 'stokes' for add_qu() !")
        self.add_pol_image(qimage,'Q')
        self.add_pol_image(uimage,'U')

        return

    # TODO deprecated -- replace with generic add_pol_image
    def add_v(self, vimage):

        """Add Stokes V image. self.polrep must be 'stokes'

           Args:
               vimage (numpy.array): The 2D Stokes Q values in Jy/pixel array
        """

        if self.polrep!='stokes':
            raise Excpetion("polrep must be 'stokes' for add_v() !")
        self.add_pol_image(vimage,'V')

        return

    def switch_polrep(self, polrep_out='stokes', pol_prim_out=None):

        """Return a new image with the polarization representation changed
           Args:
               polrep_out (str):  the polrep of the output data
               pol_prim_out (str): The default image: I,Q,U or V for Stokes, RR,LL,LR,RL for Circular

           Returns:
               (Image): new Image object with potentially different polrep
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
                imdict = {'I':self.ivec,'Q':self.qvec,'U':self.uvec,'V':self.vvec}
            else:
                if len(self.rrvec)==0 or len(self.llvec)==0:
                    ivec = []
                    vvec = []
                else:
                    ivec = 0.5*(self.rrvec + self.llvec)
                    vvec = 0.5*(self.rrvec - self.llvec)

                if len(self.rlvec)==0 or len(self.lrvec)==0:
                    qvec = []
                    uvec = []
                else:
                    qvec = np.real(0.5*(self.lrvec + self.rlvec))
                    uvec = np.real(0.5j*(self.lrvec - self.rlvec))

                imdict = {'I':ivec,'Q':qvec,'U':uvec,'V':vvec}

        elif polrep_out=='circ':
            if self.polrep=='circ':
                imdict = {'RR':self.rrvec,'LL':self.llvec,'RL':self.rlvec,'LR':self.lrvec}
            else:
                if len(self.ivec)==0 or len(self.vvec)==0:
                    rrvec = []
                    llvec = []
                else:
                    rrvec = (self.ivec + self.vvec)
                    llvec = (self.ivec - self.vvec)

                if len(self.qvec)==0 or len(self.uvec)==0:
                    rlvec = []
                    lrvec = []
                else:
                    rlvec = (self.qvec + 1j*self.uvec)
                    lrvec = (self.qvec - 1j*self.uvec)

                imdict = {'RR':rrvec,'LL':llvec,'RL':rlvec,'LR':lrvec}

        # Assemble the new image
        imvec = imdict[pol_prim_out]
        if len(imvec)==0:
            raise Exception("for switch_polrep to %s with pol_prim_out=%s, \n"%(polrep_out,pol_prim_out) +
                            "output image is not defined")

        newim = Image(imvec.reshape(self.ydim,self.xdim), self.psize, self.ra, self.dec, self.pa,
                      polrep=polrep_out, pol_prim=pol_prim_out, time=self.time,
                      rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Add in any other polarizations
        for pol in list(imdict.keys()):
            if pol==newim.pol_prim: continue
            polvec = imdict[pol]
            if len(polvec):
                newim.add_pol_image(polvec.reshape(self.ydim,self.xdim), pol)

        return newim

    #TODO -- right convention?
    def flip_chi(self):

        """Flip between the different conventions for measuring the EVPA (E of N vs N of E).

           Args:

           Returns:
               (Image): image with flipped EVPA
        """

        im = self.copy()
        if im.polrep=='stokes':
            im.qvec *= -1

        elif im.polrep=='circ':
            im.lrvec = -np.conjugate(im.lrvec)
            im.rlvec = -np.conjugate(im.rlvec)

        return im

    def orth_chi(self):

        """Rotate the EVPA 90 degrees

           Args:

           Returns:
               (Image): image with rotated EVPA
        """
        im = self.copy()
        if im.polrep=='stokes':
            im.uvec *= -1
        elif im.polrep=='circ':
            im.lrvec = np.conjugate(im.rlvec)
            im.rlvec = np.conjugate(im.rlvec)

        return im

    def imarr(self, pol=None):

        """Return the 2D image array of a given pol parameter.

           Args:
               pol (str): I,Q,U or V for Stokes, or RR,LL,LR,RL for Circular

           Returns:
               (numpy.array): 2D image array of dimension (ydim, xdim)
        """

        if pol is None: pol=self.pol_prim

        imarr = np.array([])
        if self.polrep=='stokes':
            if   pol=="I" and len(self.ivec): imarr=self.ivec.reshape(self.ydim, self.xdim)
            elif pol=="Q" and len(self.qvec): imarr=self.qvec.reshape(self.ydim, self.xdim)
            elif pol=="U" and len(self.uvec): imarr=self.uvec.reshape(self.ydim, self.xdim)
            elif pol=="V" and len(self.vvec): imarr=self.vvec.reshape(self.ydim, self.xdim)
        elif self.polrep=='circ':
            if   pol=="RR" and len(self.rrvec): imarr=self.rrvec.reshape(self.ydim, self.xdim)
            elif pol=="LL" and len(self.llvec): imarr=self.llvec.reshape(self.ydim, self.xdim)
            elif pol=="RL" and len(self.rlvec): imarr=self.rlvec.reshape(self.ydim, self.xdim)
            elif pol=="LR" and len(self.lrvec): imarr=self.lrvec.reshape(self.ydim, self.xdim)

        return imarr

    def sourcevec(self):

        """Return the source position vector in geocentric coordinates at 0h GMST.

           Args:

           Returns:
                (numpy.array): normal vector pointing to source in geocentric coordinates (m)
        """

        sourcevec = np.array([np.cos(self.dec*DEGREE), 0, np.sin(self.dec*DEGREE)])
        return sourcevec


    def fovx(self):

        """Return the image fov in x direction in radians.

           Args:

           Returns:
                (float) : image fov in x direction (radian)
        """

        return self.psize * self.xdim

    def fovy(self):

        """Returns the image fov in y direction in radians.

           Args:

           Returns:
                (float) : image fov in y direction (radian)
        """

        return self.psize * self.ydim


    def total_flux(self):

        """Return the total flux of the image in Jy.

           Args:

           Returns:
                (float) : image total flux (Jy)
        """
        if self.polrep=='stokes':
            flux = np.sum(self.ivec)
        elif self.polrep=='circ':
            flux = 0.5*(np.sum(self.rrvec)+np.sum(self.llvec))

        return flux

    def lin_polfrac(self):

        """Return the total fractional linear polarized flux

           Args:

           Returns:
                (float) : image fractional linear polarized flux
        """
        if self.polrep=='stokes':
            frac = np.abs(np.sum(self.qvec + 1j*self.uvec)) / np.abs(np.sum(self.ivec))
        elif self.polrep=='circ':
            frac = 2*np.abs(np.sum(self.rlvec)) / np.abs(np.sum(self.rrvec+self.llvec))

        return frac

    def circ_polfrac(self):

        """Return the total fractional circular polarized flux

           Args:

           Returns:
                (float) : image fractional circular polarized flux
        """
        if self.polrep=='stokes':
            frac = np.sum(self.vvec) / np.abs(np.sum(self.ivec))
        elif self.polrep=='circ':
            frac = np.sum(self.rrvec-self.llvec) / np.abs(np.sum(self.rrvec + self.llvec))

        return frac


    def centroid(self, pol=None):

        """Compute the location of the image centroid (corresponding to the polarization pol)

           Args:
                pol (str): The polarization for which to find the image centroid

           Returns:
               (np.array): centroid positions (x0,y0) in radians
        """

        if pol is None: pol=self.pol_prim
        if not (pol in list(self._imdict.keys())):
            raise Exception("for polrep==%s, pol must be in "%self.polrep + ",".join(list(self._imdict.keys())))

        pdim = self.psize
        imvec = self._imdict[pol]
        if len(imvec):
            xlist = np.arange(0,-self.xdim,-1)*pdim + (pdim*self.xdim)/2.0 - pdim/2.0
            ylist = np.arange(0,-self.ydim,-1)*pdim + (pdim*self.ydim)/2.0 - pdim/2.0
            x0 = np.sum(np.outer(0.0*ylist+1.0, xlist).ravel()*imvec)/np.sum(imvec)
            y0 = np.sum(np.outer(ylist, 0.0*xlist+1.0).ravel()*imvec)/np.sum(imvec)
            centroid = np.array([x0, y0])
        else:
            raise Exception("No %s image found!"  % pol)

        return centroid


    def pad(self, fovx, fovy):

        """Pad an image to new fov_x by fov_y in radian.
           Args:
                fovx  (float): new fov in x dimension (rad)
                fovy  (float): new fov in y dimension (rad)

           Returns:
                im_pad (Image): padded image
        """

        # Find pad widths
        fovoldx=self.fovx()
        fovoldy=self.fovy()
        padx=int(0.5*(fovx-fovoldx)/self.psize)
        pady=int(0.5*(fovy-fovoldy)/self.psize)

        # Pad main image vector
        imarr=self.imvec.reshape(self.ydim, self.xdim)
        imarr=np.pad(imarr,((pady,pady),(padx,padx)),'constant')

        # Make new image
        outim = Image(imarr, self.psize, self.ra, self.dec, self.pa,
                      polrep=self.polrep, pol_prim=self.pol_prim, time=self.time,
                      rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Pad all polarizations and copy over
        for pol in list(self._imdict.keys()):
            if pol==self.pol_prim: continue
            polvec = self._imdict[pol]
            if len(polvec):
                polarr=polvec.reshape(self.ydim, self.xdim)
                polarr=np.pad(polarr,((pady,pady),(padx,padx)), 'constant')
                outim.add_pol_image(polarr, pol)

        return outim


    def resample_square(self, xdim_new, ker_size=5):

        """Exactly resample a square image to new dimensions using the pulse function.

           Args:
                xdim_new  (int): new pixel dimension
                ker_size  (int): kernel size for resampling

           Returns:
                im_resampled (Image): resampled image
        """

        im = self
        if self.xdim != self.ydim:
            raise Exception("Image must be square to use Image.resample_square!")
        if self.pulse == ehtim.observing.pulses.deltaPulse2D:
            raise Exception("Image.resample_squre only works on continuously parametrized images: does not work with delta pulses!")

        ydim_new = xdim_new
        fov = self.xdim * self.psize
        psize_new = float(fov) / float(xdim_new)

        # Define an interpolation function using the pulse
        ij = np.array([[[i*self.psize + (self.psize*self.xdim)/2.0 - self.psize/2.0, j*self.psize + (self.psize*self.ydim)/2.0 - self.psize/2.0]
                        for i in np.arange(0, -self.xdim, -1)]
                        for j in np.arange(0, -self.ydim, -1)]).reshape((self.xdim*self.ydim, 2))

        def im_new_val(imvec,x_idx,y_idx):
            x = x_idx*psize_new + (psize_new*xdim_new)/2.0 - psize_new/2.0
            y = y_idx*psize_new + (psize_new*ydim_new)/2.0 - psize_new/2.0
            mask = (((x - ker_size*self.psize/2.0) < ij[:,0]) * (ij[:,0] < (x + ker_size*self.psize/2.0)) *
                    ((y - ker_size*self.psize/2.0) < ij[:,1]) * (ij[:,1] < (y + ker_size*self.psize/2.0))
                   ).flatten()
            interp = np.sum([imvec[n] * self.pulse(x-ij[n,0], y-ij[n,1], self.psize, dom="I") for n in np.arange(len(imvec))[mask]])
            return interp

        def im_new(imvec):
            imarr_new = np.array([[im_new_val(imvec, x_idx , y_idx)
                                   for x_idx in np.arange(0, -xdim_new, -1)]
                                   for y_idx in np.arange(0, -ydim_new, -1)])
            return imarr_new

        # Compute new primary image vector
        imarr_new = im_new(self.imvec)

        # Normalize
        scaling = np.sum(self.imvec) / np.sum(imarr_new)
        imarr_new *= scaling

        # Make new image
        outim = Image(imarr_new, psize_new, self.ra, self.dec, self.pa,
                      polrep=self.polrep, pol_prim=self.pol_prim, time=self.time,
                      rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Interpolate all polarizations and copy over
        for pol in list(self._imdict.keys()):
            if pol==self.pol_prim: continue
            polvec = self._imdict[pol]
            if len(polvec):
                polarr_new = im_new(polvec)
                polarr_new *= scaling
                outim.add_pol_image(polarr_new, pol)

        return outim

    def regrid_image(self, targetfov, npix, interp='linear'):

        """Resample the image to new (square) dimensions.

           Args:
                targetfov  (float): new field of view (radian)
                npix  (int): new pixel dimension
                interp ('linear', 'cubic', 'quintic'): type of interpolation. default is linear

           Returns:
                (Image): resampled image
        """

        psize_new = float(targetfov)/float(npix)
        fov_x = self.fovx()
        fov_y = self.fovy()

        # define an interpolation function
        x = np.linspace(-fov_x/2, fov_x/2, self.xdim)
        y = np.linspace(-fov_y/2, fov_y/2, self.ydim)

        xtarget = np.linspace(-targetfov/2, targetfov/2, npix)
        ytarget = np.linspace(-targetfov/2, targetfov/2, npix)

        def interp_imvec(imvec):
            if np.any(np.imag(imvec)!=0):
                return interp_imvec(np.real(imvec)) + 1j*interp_imvec(np.imag(imvec))
            #interpfunc = scipy.interpolate.RectBivariateSpline( y, x, np.reshape(imvec, (self.ydim, self.xdim) ) )
            interpfunc = scipy.interpolate.interp2d(y, x, np.reshape(imvec, (self.ydim, self.xdim) ) , kind=interp)
            tmpimg = interpfunc(ytarget, xtarget)
            tmpimg[np.abs(xtarget)>fov_x/2.,:] = 0.0
            tmpimg[:,np.abs(ytarget)>fov_y/2.] = 0.0
            tmpimg = tmpimg * (psize_new)**2 / self.psize**2
            return tmpimg

        # Make new image
        imarr_new = interp_imvec(self.imvec)

        outim = Image(imarr_new, psize_new, self.ra, self.dec, self.pa,
                      polrep=self.polrep, pol_prim=self.pol_prim, time=self.time,
                      rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Interpolate all polarizations and copy over
        for pol in list(self._imdict.keys()):
            if pol==self.pol_prim: continue
            polvec = self._imdict[pol]
            if len(polvec):
                polarr_new = interp_imvec(polvec)
                outim.add_pol_image(polarr_new, pol)

        return outim


    def rotate(self, angle, interp='cubic'):

        """Rotate the image counterclockwise by the specified angle.

           Args:
                angle  (float): CCW angle to rotate the image (radian)
                interp ('linear', 'cubic', 'quintic'): type of interpolation. default is cubic
           Returns:
                (Image): resampled image
        """
        order=3
        if interp=='linear': order=1
        elif interp=='cubic': order=3
        elif interp=='quintic': order=5

        # Define an interpolation function
        def rot_imvec(imvec):
            if np.any(np.imag(imvec)!=0):
                return rot_imvec(np.real(imvec)) + 1j*rot_imvec(np.imag(imvec))
            imarr_rot = scipy.ndimage.interpolation.rotate(imvec.reshape((self.ydim, self.xdim)),
                                                           angle*180.0/np.pi, reshape=False, order=order,
                                                           mode='constant', cval=0.0, prefilter=True)

            return imarr_rot

        # Make new image
        imarr_rot = rot_imvec(self.imvec)
        outim = Image(imarr_rot, self.psize, self.ra, self.dec, self.pa,
                      polrep=self.polrep, pol_prim=self.pol_prim, time=self.time,
                      rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Rotate all polarizations and copy over
        for pol in list(self._imdict.keys()):
            if pol==self.pol_prim: continue
            polvec = self._imdict[pol]
            if len(polvec):
                polarr_rot = rot_imvec(polvec)
                outim.add_pol_image(polarr_rot, pol)

        return outim

    def shift(self, shiftidx):

        """Shift the image by a given number of pixels.

         Args:
             shiftidx (list): pixel offsets [x_offset, y_offset] for the image shift

         Returns:
             (Image): shifted images
        """

        # Define shifting function
        def shift_imvec(imvec):
            im_shift = np.roll(imvec.reshape(self.ydim, self.xdim), shiftidx[0], axis=0)
            im_shift = np.roll(im_shift, shiftidx[1], axis=1)
            return im_shift

        # Make new image
        imarr_shift = shift_imvec(self.imvec)
        outim = Image(imarr_shift, self.psize, self.ra, self.dec, self.pa,
                      polrep=self.polrep, pol_prim=self.pol_prim, time=self.time,
                      rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Shift all polarizations and copy over
        for pol in list(self._imdict.keys()):
            if pol==self.pol_prim: continue
            polvec = self._imdict[pol]
            if len(polvec):
                polarr_shift = shift_imvec(polvec)
                outim.add_pol_image(polarr_shift, pol)

        return outim


    def blur_gauss(self, beamparams, frac=1., frac_pol=0):

        """Blur image with a Gaussian beam defined by beamparams [fwhm_max, fwhm_min, theta] in radians.

           Args:
               beamparams (list): the gaussian parameters, [fwhm_maj, fwhm_min, theta, x, y], all in radians
               frac (float): fractional beam size for blurring the main image (polarization defined by self.pol_prim)
               frac_pol (float): fractional beam size for blurring the other polarizations

           Returns:
               (Image): output image
        """

        if frac <= 0.0 or beamparams[0]<=0:
            return self.copy()

        # Make a Gaussian image
        xlist = np.arange(0,-self.xdim,-1)*self.psize + (self.psize*self.xdim)/2.0 - self.psize/2.0
        ylist = np.arange(0,-self.ydim,-1)*self.psize + (self.psize*self.ydim)/2.0 - self.psize/2.0
        sigma_maj = beamparams[0] / (2. * np.sqrt(2. * np.log(2.)))
        sigma_min = beamparams[1] / (2. * np.sqrt(2. * np.log(2.)))
        cth = np.cos(beamparams[2])
        sth = np.sin(beamparams[2])

        def gaussim(blurfrac):
            gauss = np.array([[np.exp(-(j*cth + i*sth)**2/(2*(blurfrac*sigma_maj)**2) - (i*cth - j*sth)**2/(2.*(blurfrac*sigma_min)**2))
                               for i in xlist]
                               for j in ylist])
            gauss = gauss[0:self.ydim, 0:self.xdim]
            gauss = gauss / np.sum(gauss) # normalize to 1
            return gauss

        gauss = gaussim(frac)
        if frac_pol:
            gausspol = gaussim(frac_pol)

        # Define a convolution function
        def blur(imarr, gauss):
            #if np.any(np.imag(imvec)!=0):
            #    return blur(np.real(imvec)) + 1j*blur(np.imag(imvec))
            imarr_blur = scipy.signal.fftconvolve(gauss, imarr, mode='same')
            return imarr_blur

        # Convolve the primary image
        imarr = (self.imvec).reshape(self.ydim, self.xdim)


        outim = Image(blur(imarr,gauss), self.psize, self.ra, self.dec, self.pa,
                      polrep=self.polrep, pol_prim=self.pol_prim, time=self.time,
                      rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Blur all polarizations and copy over
        for pol in list(self._imdict.keys()):
            if pol==self.pol_prim: continue
            polvec = self._imdict[pol]
            if len(polvec):
                polarr = polvec.reshape(self.ydim, self.xdim)
                if frac_pol:
                    polarr = blur(polarr,gauss)
                outim.add_pol_image(polarr, pol)

        return outim

    def blur_circ(self, fwhm_i, fwhm_pol=0):

        """Apply a circular gaussian filter to the image, with FWHM in radians.

           Args:
               fwhm_i (float): circular beam size for Stokes I  blurring in  radian
               fwhm_pol (float): circular beam size for Stokes Q,U,V  blurring in  radian

           Returns:
               (Image): output image
        """


        sigma = fwhm_i / (2. * np.sqrt(2. * np.log(2.)))
        sigmap = sigma / self.psize

        # Define a convolution function
        def blur(imarr, sigma):
            if np.any(np.imag(imarr)!=0):
                return blur(np.real(imarr),sigma) + 1j*blur(np.imag(imarr),sigma)
            imarr_blur = filt.gaussian_filter(imarr, (sigma, sigma))
            return imarr_blur

        # Blur the primary image
        imarr = self.imvec.reshape(self.ydim, self.xdim)
        imarr = blur(imarr,sigmap)
        outim = Image(imarr, self.psize, self.ra, self.dec, self.pa,
                      polrep=self.polrep, pol_prim=self.pol_prim, time=self.time,
                      rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Blur all polarizations and copy over
        for pol in list(self._imdict.keys()):
            if pol==self.pol_prim: continue
            polvec = self._imdict[pol]
            if len(polvec):
                polarr = polvec.reshape(self.ydim, self.xdim)
                if fwhm_pol:
                    print ("Blurring polarization")
                    sigma = fwhm_pol / (2. * np.sqrt(2. * np.log(2.)))
                    sigmap = sigma / self.psize
                    polarr = blur(polarr, sigmap)
                outim.add_pol_image(polarr, pol)

        return outim

    def grad(self, gradtype='abs'):

        """Return the gradient image

           Args:
               gradtype (str): 'x','y',or 'abs' for the image gradient dimension
           Returns:
               Image : an image object containing the gradient image
        """

        # Define the desired gradient function
        def gradim(imvec):
            if np.any(np.imag(imvec)!=0):
                return gradim(np.real(imvec)) + 1j*gradim(np.imag(imvec))

            imarr = imvec.reshape(self.ydim, self.xdim)
            #gradlist = np.gradient(imarr)
            #[sy,sx] = gradlist
            sx = ndi.sobel(imarr, axis=0, mode='constant')
            sy = ndi.sobel(imarr, axis=1, mode='constant')

            #TODO: are these in the right order??
            if gradtype=='x':
                gradarr = sx
            if gradtype=='y':
                gradarr = sy
            else:
                gradarr = np.hypot(sx, sy)
            return gradarr

        # Find the gradient for the primary image
        gradarr = gradim(self.imvec)
        outim = Image(gradarr, self.psize, self.ra, self.dec, self.pa,
                      polrep=self.polrep, pol_prim=self.pol_prim, time=self.time,
                      rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Find the gradient for all polarizations and copy over
        for pol in list(self._imdict.keys()):
            if pol==self.pol_prim: continue
            polvec = self._imdict[pol]
            if len(polvec):
                gradarr = gradim(polvec)
                outim.add_pol_image(gradarr, pol)

        return outim

    def mask(self, cutoff=0.05, beamparams=None, frac=0.0):

        """Produce an image mask that shows all pixels above the specified cutoff percentage of the max flux
           Works off the primary image

           Args:
               cutoff (float): Primary image pixels with intensities greater than the cuttoff * max intensity are masked
               beamparams (list): either [fwhm_maj, fwhm_min, pos_ang] parameters of an elliptical gaussian, or a single fwhm
               frac (float): the fraction of nominal beam to blur with

           Returns:
               (Image): output mask image

        """

        # Blur the image
        if not beamparams is None:
            try: len(beamparams)
            except TypeError:
                beamparams = [beamparams, beamparams, 0]
            if len(beamparams)==3:
                mask = self.blur_gauss(beamparams, frac)
            else:
                raise Exception("beamparams should be a length 3 array [maj, min, posang]!")
        else:
            mask = self.copy()

        # Mask pixels outside the desired intensity range
        maxval = np.max(mask.imvec)
        minval = np.min(mask.imvec)
        #minval = np.max((np.min(mask.imvec),0.))
        intensityrange = maxval - minval
        thresh = intensityrange * cutoff +  minval
        maskvec  = (mask.imvec > thresh).astype(int)

        # make the primary image
        maskarr = maskvec.reshape(mask.ydim, mask.xdim)
        mask = Image(maskarr, self.psize, self.ra, self.dec, self.pa,
                      polrep=self.polrep, pol_prim=self.pol_prim, time=self.time,
                      rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Replace all polarization imvecs with mask
        for pol in list(self._imdict.keys()):
            if pol==self.pol_prim: continue
            mask.add_pol_image(maskarr, pol)

        return mask

    #TODO make this work with a mask image with different dimensions & fov
    def apply_mask(self, mask_im, fill_val=0.):

        """Apply a mask to the image

           Args:
               mask_im (Image): a mask image with the same dimensions as the Image
               fill_val (float): masked pixels of all polarizations are set to this value

           Returns:
               (Image): the masked image

        """
        if (self.psize != mask_im.psize) or (self.xdim != mask_im.xdim) or (self.ydim != mask_im.ydim):
            raise Exception("mask image does not match dimensions of the current image!")

        # Get the mask vector
        maskvec = mask_im.imvec.astype(bool)
        maskvec[maskvec <= 0] = 0
        maskvec[maskvec > 0] = 1

        # Mask the primary image
        imvec = self.imvec
        imvec[~maskvec] = fill_val
        outim = Image(imvec.reshape(self.ydim,self.xdim), self.psize, self.ra, self.dec, self.pa,
                      polrep=self.polrep, pol_prim=self.pol_prim,  time=self.time,
                      rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Find the gradient for all polarizations and copy over
        for pol in list(self._imdict.keys()):
            if pol==self.pol_prim: continue
            polvec = self._imdict[pol]
            if len(polvec):
                polvec[~maskvec] = fill_val
                outim.add_pol_image(polvec.reshape(self.ydim,self.xdim), pol)

        return outim

    def threshold(self, cutoff=0.05, frac_i=None, beamparams=None, frac=0.0, fill_val=None):

        """Apply a hard threshold to the primary polarization image. Leave other polarizations untouched.

           Args:
               cutoff (float): Pixels with intensities greater than the cuttoff * max intensity are masked
               frac_i (float): the old name for cutoff: should not be used except in old scripts!
               beamparams (list): either [fwhm_maj, fwhm_min, pos_ang] parameters of an elliptical gaussian, or a single fwhm
               frac (float): the fraction of nominal beam to blur with
               fill_val (float): masked pixels are set to this value. If fill_val==None, they are set to the min unmasked value

           Returns:
               (Image): output mask image
        """

        if not (frac_i is None):
            cutoff=frac_i
            print("Warning!: using frac_i=%f as cutoff in threshold(). Rename 'frac_i' to 'cutoff' in future scripts!")

        if fill_val is None or fill_val==False:
            maxval = np.max(self.imvec)
            minval = np.min(self.imvec)
            #minval = np.max((np.min(self.imvec),0.))
            intensityrange = maxval - minval
            fill_val = (intensityrange * cutoff +  minval)

        mask = self.mask(cutoff=cutoff, beamparams=beamparams, frac=frac)
        out = self.apply_mask(mask, fill_val=fill_val)
        return out

    def add_flat(self, flux, pol=None):

        """Add a flat background flux to the main polarization image.

           Args:
                flux  (float): total flux to add to image
                pol (str): the polarization to add the flux to. None defaults to pol_prim.
           Returns:
                (Image): output image
        """

        if pol is None: pol=self.pol_prim
        if not (pol in list(self._imdict.keys())):
            raise Exception("for polrep==%s, pol must be in "%self.polrep + ",".join(list(self._imdict.keys())))
        if not len(self._imdict[pol]):
            raise Exception("no image for pol %s"%pol)

        # Make a flat image array
        flatarr = ((flux/float(len(self.imvec))) * np.ones(len(self.imvec))).reshape(self.ydim, self.xdim)

        # Add to the main image and create the new image object
        imarr = self.imvec.reshape(self.ydim, self.xdim).copy()
        if pol==self.pol_prim:
            imarr += flatarr

        outim = Image(imarr, self.psize, self.ra, self.dec, self.pa,
                      polrep=self.polrep, pol_prim=self.pol_prim, time=self.time,
                      rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Copy over the rest of the polarizations
        for pol2 in list(self._imdict.keys()):
            if pol2==self.pol_prim: continue
            polvec = self._imdict[pol2]
            if len(polvec):
                polarr = polvec.reshape(self.ydim, self.xdim).copy()
                if pol2==pol:
                    polarr += flatarr
                outim.add_pol_image(polarr, pol2)

        return outim

    def add_tophat(self, flux, radius, pol=None):

        """Add centered tophat flux to the Stokes I image inside a given radius.

           Args:
                flux  (float): total flux to add to image
                radius  (float): radius of top hat flux in radians
                pol (str): the polarization to add the flux to. None defaults to pol_prim

           Returns:
                (Image): output image
        """

        if pol is None: pol=self.pol_prim
        if not (pol in list(self._imdict.keys())):
            raise Exception("for polrep==%s, pol must be in "%self.polrep + ",".join(list(self._imdict.keys())))
        if not len(self._imdict[pol]):
            raise Exception("no image for pol %s"%pol)

        # Make a tophat image array
        xlist = np.arange(0,-self.xdim,-1)*self.psize + (self.psize*self.xdim)/2.0 - self.psize/2.0
        ylist = np.arange(0,-self.ydim,-1)*self.psize + (self.psize*self.ydim)/2.0 - self.psize/2.0

        hatarr = np.array([[1.0 if np.sqrt(i**2+j**2) <= radius else 0.
                            for i in xlist]
                            for j in ylist])

        hatarr = hatarr[0:self.ydim, 0:self.xdim]
        hatarr *= flux / np.sum(hatarr)

        # Add to the main image and create the new image object
        imarr = self.imvec.reshape(self.ydim, self.xdim).copy()
        if pol==self.pol_prim:
            imarr += hatarr

        outim = Image(imarr, self.psize, self.ra, self.dec, self.pa,
                      polrep=self.polrep, pol_prim=self.pol_prim, time=self.time,
                      rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Copy over the rest of the polarizations
        for pol2 in list(self._imdict.keys()):
            if pol2==self.pol_prim: continue
            polvec = self._imdict[pol2]
            if len(polvec):
                polarr = polvec.reshape(self.ydim, self.xdim).copy()
                if pol2==pol:
                    polarr += hatarr
                outim.add_pol_image(polarr, pol2)

        return outim

    def add_gauss(self, flux, beamparams, pol=None):

        """Add a gaussian to an image.

           Args:
               flux (float): the total flux contained in the Gaussian in Jy
               beamparams (list): the gaussian parameters, [fwhm_maj, fwhm_min, theta, x, y], all in radians
               pol (str): the polarization to add the flux to. None defaults to pol_prim.

           Returns:
                (Image): output image
        """

        if pol is None: pol=self.pol_prim
        if not (pol in list(self._imdict.keys())):
            raise Exception("for polrep==%s, pol must be in "%self.polrep + ",".join(list(self._imdict.keys())))
        if not len(self._imdict[pol]):
            raise Exception("no image for pol %s"%pol)

        # Make a Gaussian image
        try:
            x=beamparams[3]
            y=beamparams[4]
        except IndexError:
            x=y=0.0

        sigma_maj = beamparams[0] / (2. * np.sqrt(2. * np.log(2.)))
        sigma_min = beamparams[1] / (2. * np.sqrt(2. * np.log(2.)))
        cth = np.cos(beamparams[2])
        sth = np.sin(beamparams[2])
        xlist = np.arange(0,-self.xdim,-1)*self.psize + (self.psize*self.xdim)/2.0 - self.psize/2.0
        ylist = np.arange(0,-self.ydim,-1)*self.psize + (self.psize*self.ydim)/2.0 - self.psize/2.0

        def gaussian(x2, y2):
            gauss = np.exp(-((y2)*cth + (x2)*sth)**2/(2*sigma_maj**2) +
                           -((x2)*cth - (y2)*sth)**2/(2*sigma_min**2))
            return gauss

        gaussarr = np.array([[gaussian(i-x, j-y) for i in xlist] for j in ylist])
        gaussarr = gaussarr[0:self.ydim, 0:self.xdim]
        gaussarr *= flux/np.sum(gaussarr)

        # TODO: if we want to add a gaussian to V, we might also want to make sure we add it to I
        # Add to the main image and create the new image object
        imarr = self.imvec.reshape(self.ydim, self.xdim).copy()
        if pol==self.pol_prim:
            imarr += gaussarr

        outim = Image(imarr, self.psize, self.ra, self.dec, self.pa,
                      polrep=self.polrep, pol_prim=self.pol_prim, time=self.time,
                      rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Copy over the rest of the polarizations
        for pol2 in list(self._imdict.keys()):
            if pol2 == self.pol_prim: continue
            polvec = self._imdict[pol2]
            if len(polvec):
                polarr = polvec.reshape(self.ydim, self.xdim).copy()
                if pol2==pol:
                    polarr += gaussarr
                outim.add_pol_image(polarr, pol2)

        return outim

    def add_crescent(self, flux, Rp, Rn, a, b, x=0, y=0, pol=None):

        """Add a crescent to an image; see Kamruddin & Dexter (2013).

           Args:
               flux (float): the total flux contained in the crescent in Jy
               Rp (float): the larger radius in radians
               Rn (float): the smaller radius in radians
               a (float): the relative x offset of smaller disk in radians
               b (float): the relative y offset of smaller disk in radians
               x (float): the center x coordinate of the larger disk in radians
               y (float): the center y coordinate of the larger disk in radians
               pol (str): the polarization to add the flux to. None defaults to pol_prim.

           Returns:
               (Image): output image add_gaus
        """

        if pol is None: pol=self.pol_prim
        if not (pol in list(self._imdict.keys())):
            raise Exception("for polrep==%s, pol must be in "%self.polrep + ",".join(list(self._imdict.keys())))
        if not len(self._imdict[pol]):
            raise Exception("no image for pol %s"%pol)


        # Make a crescent image
        xlist = np.arange(0,-self.xdim,-1)*self.psize + (self.psize*self.xdim)/2.0 - self.psize/2.0
        ylist = np.arange(0,-self.ydim,-1)*self.psize + (self.psize*self.ydim)/2.0 - self.psize/2.0

        def crescent(x2, y2):
            if (x2-a)**2 + (y2-b)**2 > Rn**2 and x2**2 + y2**2 < Rp**2:
                return 1.0
            else:
                return 0.0

        crescarr = np.array([[crescent(i-x, j-y) for i in xlist] for j in ylist])
        crescarr = crescarr[0:self.ydim, 0:self.xdim]
        crescarr *= flux/np.sum(crescarr)

        # Add to the main image and create the new image object
        imarr = self.imvec.reshape(self.ydim, self.xdim).copy()
        if pol==self.pol_prim:
            imarr += crescarr

        outim = Image(imarr, self.psize, self.ra, self.dec, self.pa,
                      polrep=self.polrep, pol_prim=self.pol_prim,  time=self.time,
                      rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Copy over the rest of the polarizations
        for pol2 in list(self._imdict.keys()):
            if pol2==self.pol_prim: continue
            polvec = self._imdict[pol2]
            if len(polvec):
                polarr = polvec.reshape(self.ydim, self.xdim).copy()
                if pol2==pol:
                    polarr += crescarr
                outim.add_pol_image(polarr, pol2)

        return outim

    def add_ring_m1(self, I0, I1, r0, phi, sigma, x=0, y=0, pol=None):

        """Add a ring to an image with an m=1 mode

           Args:
               I0 (float):
               I1 (float):
               r0 (float): the radius
               phi (float): angle of m1 mode
               sigma (float): the blurring size
               x (float): the center x coordinate of the larger disk in radians
               y (float): the center y coordinate of the larger disk in radians
               pol (str): the polarization to add the flux to. None defaults to pol_prim.
           Returns:
               (Image): output image add_gaus
        """

        if pol is None: pol=self.pol_prim
        if not (pol in list(self._imdict.keys())):
            raise Exception("for polrep==%s, pol must be in "%self.polrep + ",".join(list(self._imdict.keys())))
        if not len(self._imdict[pol]):
            raise Exception("no image for pol %s"%pol)

        # Make a ring image
        flux = I0 - 0.5*I1
        phi = phi + np.pi
        psize = self.psize
        xlist = np.arange(0,-self.xdim,-1)*self.psize + (self.psize*self.xdim)/2.0 - self.psize/2.0
        ylist = np.arange(0,-self.ydim,-1)*self.psize + (self.psize*self.ydim)/2.0 - self.psize/2.0

        def ringm1(x2, y2):
            if (x2**2 + y2**2) > (r0-psize)**2 and (x2**2 + y2**2) < (r0+psize)**2:
                theta = np.arctan2(y2,x2)
                flux = (I0-0.5*I1*(1+np.cos(theta-phi)))/(2*np.pi*r0)
                return flux
            else:
                return 0.0

        ringarr = np.array([[ringm1(i-x, j-y)
                          for i in xlist]
                          for j in ylist])
        ringarr = ringarr[0:self.ydim, 0:self.xdim]
        tmp = Image(ringarr, self.psize, self.ra, self.dec, self.pa, rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)
        tmp = tmp.blur_circ(sigma)
        tmp.imvec *= flux/(tmp.total_flux())
        ringarr = tmp.imvec.reshape(self.ydim, self.xdim)

        # Add to the main image and create the new image object
        imarr = self.imvec.reshape(self.ydim, self.xdim).copy()
        if pol==self.pol_prim:
            imarr += ringarr

        outim = Image(imarr, self.psize, self.ra, self.dec, self.pa,
                      polrep=self.polrep, pol_prim=self.pol_prim, time=self.time,
                      rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Copy over the rest of the polarizations
        for pol2 in list(self._imdict.keys()):
            if pol2==self.pol_prim: continue
            polvec = self._imdict[pol2]
            if len(polvec):
                polarr = polvec.reshape(self.ydim, self.xdim).copy()
                if pol2==pol:
                    polarr += ringarr
                outim.add_pol_image(polarr, pol2)

        return outim

    def add_const_pol(self, mag, angle, cmag=0, csign=1):

        """Return an image with the same total intensity but fractional linear and circular polarization set to constant values

           Args:
               mag (float): constant polarization fraction to add to the image
               angle (float): constant EVPA
               cmag (float): constant circular polarization fraction to add to the image
               cmag (int): constant circular polarization sign +/- 1

           Returns:
                (Image): output image
        """

        if not (0 <= mag < 1):
            raise Exception("fractional polarization magnitude must be between 0 and 1!")

        if not (0 <= cmag < 1):
            raise Exception("circular polarization magnitude must be between 0 and 1!")

        im = self

        if self.polrep=='stokes':
            im_stokes = self
        elif self.polrep=='circ':
            im_stokes = self.switch_polrep(polrep_out='stokes')
        ivec = im_stokes.ivec.copy()
        qvec = qimage(ivec, mag * np.ones(len(ivec)), angle*np.ones(len(ivec)))
        uvec = uimage(ivec, mag * np.ones(len(ivec)), angle*np.ones(len(ivec)))
        vvec = cmag * np.sign(csign) * ivec

        # create the new stokes image object
        iarr = ivec.reshape(im.ydim,im.xdim).copy()
        outim = Image(iarr, self.psize, self.ra, self.dec, self.pa,
                      polrep='stokes', pol_prim='I', time=self.time,
                      rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Copy over the rest of the polarizations
        imdict = {'I':ivec,'Q':qvec,'U':uvec,'V':vvec}
        for pol in list(imdict.keys()):
            if pol=='I': continue
            polvec = imdict[pol]
            if len(polvec):
                polarr = polvec.reshape(self.ydim, self.xdim).copy()
                outim.add_pol_image(polarr, pol)

        return outim

    def add_random_pol(self, mag, corr, cmag=0., ccorr=0., seed=0):

        """Return an image with the same total intensity but random linear and circular polarization images with specified correlation lengths

           Args:
               mag   (float): linear polarization fraction
               corr  (float): EVPA correlation length (radians)
               cmag  (float): circular polarization fraction
               ccorr (float): CP correlation length (radians)
               seed    (int): Seed for random number generation

           Returns:
                (Image): output image
        """
        import ehtim.scattering.stochastic_optics as so

        if not (0 <= mag < 1):
            raise Exception("fractional polarization magnitude must be between 0 and 1!")

        if not (0 <= cmag < 1):
            raise Exception("circular polarization magnitude must be between 0 and 1!")

        im = self

        if self.polrep=='stokes':
            im_stokes = self
        elif self.polrep=='circ':
            im_stokes = self.switch_polrep(polrep_out='stokes')
        ivec = im_stokes.ivec.copy()

        # create the new stokes image object
        iarr = ivec.reshape(im.ydim,im.xdim).copy()
        outim = Image(iarr, self.psize, self.ra, self.dec, self.pa,
                      polrep='stokes', pol_prim='I', time=self.time,
                      rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        # Make a random phase screen using the scattering tools
        # Use this screen to define the EVPA
        dist = 1.0 * 3.086e21
        rdiff = np.abs(corr) * dist / 1e3
        theta_mas = 0.37 * 1.0/rdiff * 1000. * 3600. * 180./np.pi
        sm = so.ScatteringModel(scatt_alpha = 1.67, observer_screen_distance = dist, source_screen_distance = 1.e5 * dist, theta_maj_mas_ref = theta_mas, theta_min_mas_ref = theta_mas, r_in = rdiff*2, r_out = 1e30)
        ep = so.MakeEpsilonScreen(im.xdim, im.ydim, rngseed = seed)
        ps = np.array(sm.MakePhaseScreen(ep, im, obs_frequency_Hz=29.979e9).imvec)/1000**(1.66/2)
        qvec = ivec * mag * np.sin(ps)
        uvec = ivec * mag * np.cos(ps)

        # Make a random phase screen using the scattering tools
        # Use this screen to define the CP magnitude
        if cmag != 0.0 and ccorr > 0.0:
            dist = 1.0 * 3.086e21
            rdiff = np.abs(ccorr) * dist / 1e3
            theta_mas = 0.37 * 1.0/rdiff * 1000. * 3600. * 180./np.pi
            sm = so.ScatteringModel(scatt_alpha = 1.67, observer_screen_distance = dist, source_screen_distance = 1.e5 * dist, theta_maj_mas_ref = theta_mas, theta_min_mas_ref = theta_mas, r_in = rdiff*2, r_out = 1e30)
            ep = so.MakeEpsilonScreen(im.xdim, im.ydim, rngseed = seed*2)
            ps = np.array(sm.MakePhaseScreen(ep, im, obs_frequency_Hz=29.979e9).imvec)/1000**(1.66/2)
            vvec = ivec * cmag * np.sin(ps)
        else:
            vvec = ivec * cmag

        # Copy over the rest of the polarizations
        imdict = {'I':ivec,'Q':qvec,'U':uvec,'V':vvec}
        for pol in list(imdict.keys()):
            if pol=='I': continue
            polvec = imdict[pol]
            if len(polvec):
                polarr = polvec.reshape(self.ydim, self.xdim).copy()
                outim.add_pol_image(polarr, pol)

        return outim
    
    def add_zblterm(self, obs, uv_min, zblval=None, new_fov=False, gauss_sz=False, gauss_sz_factor=0.75, debias=True):
        
        """Add a Gaussian term to account for missing flux in the 0 baseline.
            
            Args:
            obs : an Obsdata object that the min non-zero baseline and 0-bl flux are determined from
            uv_min (float): The cuttoff in Glambada used to determine what is a 0-bl
            new_fov (rad): The size of the padded image once the Gaussian is added (safest to keep False, then it will be set to 3 x the gaussian fwhm)
            gauss_sz (rad): The size of the Gaussian added to add flux to the 0-bl. (safest to keep False and it is computed from the min non-zero baseline)
            gauss_sz_factor (float): The fraction of the min non-zero baseline that is used to caluclate the Gaussian FWHM.
            debias (bool): True if you use debiased amplitudes to caluclate the 0-bl flux in Jy
            
            Returns:
            (Image): a padded image with a large Gaussian component
            """
        
        if gauss_sz == False:
            obs_flag = obs.flag_uvdist(uv_min = uv_min)
            minuvdist = np.min( np.sqrt(obs_flag.data['u']**2 + obs_flag.data['v']**2) )
            gauss_sz_sigma = (1/(gauss_sz_factor*minuvdist))
            gauss_sz = gauss_sz_sigma * 2.355 # convert from stdev to fwhm
        
        factor = 5.0
        if new_fov == False:
            im_fov = np.max((self.xdim*self.psize, self.ydim*self.psize))
            new_fov = np.max((factor*(gauss_sz / 2.355), im_fov))
        
        if new_fov < factor*(gauss_sz / 2.355):
            print('WARNING! The specified new fov may not be large enough for the gaussian size and may cause higher frequency effects')
        
        # calculate the amount of flux to include in the Gaussian
        obs_zerobl = obs.flag_uvdist(uv_max=uv_min)
        obs_zerobl.add_amp(debias=debias)
        orig_totflux = np.sum(obs_zerobl.amp['amp']*(1/obs_zerobl.amp['sigma']**2))/np.sum(1/obs_zerobl.amp['sigma']**2)
        
        if zblval == None:
            addedflux = orig_totflux - np.sum(self.imvec)
        else:
            addedflux = orig_totflux - zblval
        
        print('Adding a ' + str(addedflux) + ' Jy circular Gaussian of FWHM size ' + str(gauss_sz/RADPERUAS) + ' uas')
        
        im_new = self.copy()
        im_new = im_new.pad(new_fov, new_fov)
        im_new = im_new.add_gauss(addedflux, (gauss_sz, gauss_sz, 0, 0, 0))
        return im_new


    def sample_uv(self, uv, sgrscat=False, polrep_obs='stokes', ttype='nfft', cache=False, fft_pad_factor=2):

        """Sample the image on the selected uv points without adding noise.

           Args:
               uv (ndarray): an array of uv points
               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
               polrep_obs (str): 'stokes' or 'circ' sets the data polarimetric representtion

               ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT
               fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in FFT

           Returns:
               (list): a list of [I,Q,U,V] visibilities
        """

        if polrep_obs not in ['stokes','circ']:
            raise Exception("polrep_obs must be either 'stokes' or 'circ'")

        data = simobs.sample_vis(self, uv, sgrscat=sgrscat, polrep_obs=polrep_obs,
                                       ttype=ttype, cache=cache, fft_pad_factor=fft_pad_factor)
        return data

    def observe_same_nonoise(self, obs, sgrscat=False,  ttype="nfft", cache=False, fft_pad_factor=2):

        """Observe the image on the same baselines as an existing observation object without adding noise.

           Args:
               obs (Obsdata): the existing observation with  baselines where the image FT will be sampled
               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel

               ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT
               fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in FFT

           Returns:
               (Obsdata): an observation object with no noise
        """


        # Check for agreement in coordinates and frequency
        tolerance = 1e-8
        if (np.abs(self.ra - obs.ra) > tolerance) or (np.abs(self.dec - obs.dec) > tolerance):
            raise Exception("Image coordinates are not the same as observtion coordinates!")
        if (np.abs(self.rf - obs.rf)/obs.rf > tolerance):
            raise Exception("Image frequency is not the same as observation frequency!")

        if ttype=='direct' or ttype=='fast' or ttype=='nfft':
            print("Producing clean visibilities from image with " + ttype + " FT . . . ")
        else:
            raise Exception("ttype=%s, options for ttype are 'direct', 'fast', 'nfft'"%ttype)

        # Copy data to be safe
        obsdata = copy.deepcopy(obs.data)

        # Extract uv datasample
        uv = recarr_to_ndarr(obsdata[['u','v']],'f8')
        data = simobs.sample_vis(self, uv, sgrscat=sgrscat, polrep_obs=obs.polrep,
                                       ttype=ttype, cache=cache, fft_pad_factor=fft_pad_factor)

        # put visibilities into the obsdata
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

        obs_no_noise = ehtim.obsdata.Obsdata(self.ra, self.dec, obs.rf, obs.bw, obsdata, obs.tarr,
                                             source=self.source, mjd=self.mjd, polrep=obs.polrep,
                                             ampcal=True, phasecal=True, opacitycal=True, dcal=True, frcal=True,
                                             timetype=obs.timetype, scantable=obs.scans)

        return obs_no_noise

    def observe_same(self, obs_in, ttype='nfft', fft_pad_factor=2,
                           sgrscat=False, add_th_noise=True,
                           opacitycal=True, ampcal=True, phasecal=True, dcal=True, frcal=True, rlgaincal=True,
                           stabilize_scan_phase=False, stabilize_scan_amp=False, neggains=False,
                           jones=False, inv_jones=False,
                           tau=TAUDEF, taup=GAINPDEF,
                           gain_offset=GAINPDEF, gainp=GAINPDEF,
                           dterm_offset=DTERMPDEF, caltable_path=None, seed=False):

        """Observe the image on the same baselines as an existing observation object and add noise.

           Args:
               obs_in (Obsdata): the existing observation with  baselines where the image FT will be sampled

               ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT
               fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in FFT

               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
               add_th_noise (bool): if True, baseline-dependent thermal noise is added to each data point
               opacitycal (bool): if False, time-dependent gaussian errors are added to station opacities
               ampcal (bool): if False, time-dependent gaussian errors are added to station gains
               phasecal (bool): if False, time-dependent station-based random phases are added to data points
               frcal (bool): if False, feed rotation angle terms are added to Jones matrices. Must have jones=True
               stabilize_scan_phase (bool): if True, random phase errors are constant over scans
               stabilize_scan_amp (bool): if True, random amplitude errors are constant over scans
               neggains (bool): if True then force the applied gains to be negative meaning that you have overestimated your telescope's performance
               dcal (bool): if False, time-dependent gaussian errors added to Jones matrices D-terms. Must have jones=True
               jones (bool): if True, uses Jones matrix to apply mis-calibration effects (gains, phases, Dterms), otherwise uses old formalism without D-terms
               inv_jones (bool): if True, applies estimated inverse Jones matrix (not including random terms) to calibrate data

               tau (float): the base opacity at all sites, or a dict giving one opacity per site
               gainp (float): the fractional std. dev. of the random error on the gains
               gain_offset (float): the base gain offset at all sites, or a dict giving one gain offset per site
               taup (float): the fractional std. dev. of the random error on the opacities
               dterm_offset (float): the base dterm offset at all sites, or a dict giving one dterm offset per site

               caltable_path (string): The path and prefix that a caltable is saved with if adding noise through a Jones matrix
               seed (int): seeds the random component of the noise terms. do not set to 0!

           Returns:
               (Obsdata): an observation object
        """

        if seed!=False:
            np.random.seed(seed=seed)

        obs = self.observe_same_nonoise(obs_in, sgrscat=sgrscat, ttype=ttype, fft_pad_factor=fft_pad_factor)

        # Jones Matrix Corruption & Calibration
        if jones:
            obsdata = simobs.add_jones_and_noise(obs, add_th_noise=add_th_noise,
                                                 opacitycal=opacitycal, ampcal=ampcal,
                                                 phasecal=phasecal, dcal=dcal, frcal=frcal, rlgaincal=rlgaincal,
                                                 stabilize_scan_phase=stabilize_scan_phase,
                                                 stabilize_scan_amp=stabilize_scan_amp, neggains=neggains,
                                                 gainp=gainp, taup=taup, gain_offset=gain_offset,
                                                 dterm_offset=dterm_offset,
                                                 caltable_path=caltable_path, seed=seed)

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
        # NOTE There is an asymmetry here - in the old way, we don't offer the ability to *not* unscale estimated noise.
        else:

            if caltable_path:
                print('WARNING: the caltable is currently only saved if you apply noise with a Jones Matrix')

            obsdata = simobs.add_noise(obs, add_th_noise=add_th_noise,
                                       ampcal=ampcal, phasecal=phasecal, opacitycal=opacitycal,
                                       stabilize_scan_phase=stabilize_scan_phase,
                                       stabilize_scan_amp=stabilize_scan_amp,
                                       gainp=gainp, taup=taup, gain_offset=gain_offset, seed=seed)

            obs =  ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata, obs.tarr,
                                         source=obs.source, mjd=obs.mjd, polrep=obs_in.polrep,
                                         ampcal=ampcal, phasecal=phasecal, opacitycal=True, dcal=True, frcal=True,
                                         timetype=obs.timetype, scantable=obs.scans)
                                         #these are always set to True after inverse jones call

        return obs

    def observe(self, array, tint, tadv, tstart, tstop, bw,
                      mjd=None, timetype='UTC', polrep_obs=None,
                      elevmin=ELEV_LOW, elevmax=ELEV_HIGH,
                      ttype='nfft', fft_pad_factor=2,
                      fix_theta_GMST=False, sgrscat=False, add_th_noise=True,
                      opacitycal=True, ampcal=True, phasecal=True, 
                      dcal=True, frcal=True, rlgaincal=True,
                      stabilize_scan_phase=False, stabilize_scan_amp=False,
                      jones=False, inv_jones=False,
                      tau=TAUDEF, taup=GAINPDEF,
                      gainp=GAINPDEF, gain_offset=GAINPDEF,
                      dterm_offset=DTERMPDEF, seed=False):

        """Generate baselines from an array object and observe the image.

           Args:
               array (Array): an array object containing sites with which to generate baselines
               tint (float): the scan integration time in seconds
               tadv (float): the uniform cadence between scans in seconds
               tstart (float): the start time of the observation in hours
               tstop (float): the end time of the observation in hours
               bw (float): the observing bandwidth in Hz

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
               dterm_offset (float): the base dterm offset at all sites, or a dict giving one dterm offset per site
               seed (int): seeds the random component of noise added. do not set to 0!

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
        obs = self.observe_same(obs, ttype=ttype, fft_pad_factor=fft_pad_factor,
                                     sgrscat=sgrscat, add_th_noise=add_th_noise,
                                     opacitycal=opacitycal,ampcal=ampcal,phasecal=phasecal,dcal=dcal,frcal=frcal, rlgaincal=rlgaincal,
                                     stabilize_scan_phase=stabilize_scan_phase,
                                     stabilize_scan_amp=stabilize_scan_amp,
                                     gainp=gainp,gain_offset=gain_offset,
                                     tau=tau, taup=taup,
                                     dterm_offset=dterm_offset,
                                     jones=jones, inv_jones=inv_jones, seed=seed)

        return obs

    def observe_vex(self, vex, source, t_int=0.0, tight_tadv=False,
                          polrep_obs=None, ttype='nfft', fft_pad_factor=2,
                          sgrscat=False, add_th_noise=True,
                          opacitycal=True, ampcal=True, phasecal=True, frcal=True, dcal=True, rlgaincal=True,
                          stabilize_scan_phase=False, stabilize_scan_amp=False,
                          jones=False, inv_jones=False,
                          tau=TAUDEF, taup=GAINPDEF, gainp=GAINPDEF, gain_offset=GAINPDEF,
                          dterm_offset=DTERMPDEF):

        """Generate baselines from a vex file and observes the image.

           Args:
               vex (Vex): an vex object containing sites and scan information
               source (str): the source to observe

               t_int (float): if not zero, overrides the vex scans to produce visibilities for each t_int seconds
               tight_tadv (float): if True, advance right after each integration, otherwise advance after 2x the scan length

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

           Returns:
               (Obsdata): an observation object

        """

        if polrep_obs is None:
            polrep_obs=self.polrep

        t_int_flag = False
        if t_int == 0.0:
            t_int_flag = True

        # Loop over all scans and assemble a list of scan observations
        obs_List=[]
        for i_scan in range(len(vex.sched)):

            if t_int_flag:
                 t_int = vex.sched[i_scan]['scan'][0]['scan_sec']
            if tight_tadv:
                t_adv = t_int
            else:
                t_adv = 2.0*vex.sched[i_scan]['scan'][0]['scan_sec']

            # If this scan doesn't observe the source, advance
            if vex.sched[i_scan]['source'] != source:
                continue

            # What subarray is observing now?
            scankeys = list(vex.sched[i_scan]['scan'].keys())
            subarray = vex.array.make_subarray([vex.sched[i_scan]['scan'][key]['site'] for key in scankeys])

            # Observe with the subarray over the scan interval
            t_start = vex.sched[i_scan]['start_hr']
            t_stop = vex.sched[i_scan]['start_hr'] + vex.sched[i_scan]['scan'][0]['scan_sec']/3600.0
            obs = self.observe(subarray, t_int, t_adv, t_start, t_stop, vex.bw_hz,
                                       mjd=vex.sched[i_scan]['mjd_floor'], elevmin=.01, elevmax=89.99,
                                       polrep_obs=polrep_obs,
                                       ttype=ttype, fft_pad_factor=fft_pad_factor, sgrscat=sgrscat, add_th_noise=add_th_noise,
                                       opacitycal=opacitycal,ampcal=ampcal,phasecal=phasecal,dcal=dcal,frcal=frcal,rlgaincal=rlgaincal,
                                       stabilize_scan_phase=stabilize_scan_phase,
                                       stabilize_scan_amp=stabilize_scan_amp,
                                       taup=taup, gainp=gainp,gain_offset=gain_offset,dterm_offset=dterm_offset,
                                       jones=jones, inv_jones=inv_jones)


            obs_List.append(obs)

        # Merge the scans together
        obs = ehtim.obsdata.merge_obs(obs_List)

        return obs

    def compare_images(self, im2, psize=None, target_fov=None,blur_frac=0.0,
                             beamparams=[1., 1., 1.], metric=['nxcorr', 'nrmse', 'rssd'],
                             blursmall=False, shift=True):

        """Compare to another image by computing normalized cross correlation, normalized root mean squared error, or square root of the sum of squared differences.
           Returns metrics only for the primary polarization imvec!

           Args:
               psize (float): pixel size of comparison image (rad). If None it is the smallest of the input image pizel sizes
               target_fov (float): fov of the comparison image (rad). If None it is twice the largest fov of the input images

               beamparams (list): the nominal Gaussian beam parameters [fovx, fovy, position angle]
               blur_frac (float): fractional beam to blur each image to before comparison

               metric (list) : a list of fidelity metrics from ['nxcorr','nrmse','rssd']
               blursmall (bool) : True to blur the unpadded image rather than the large image.
               shift (int): manual image shift, otherwise use shift from maximum cross-correlation

           Returns:
               (tuple): [errormetric, im1_pad, im2_shift] of computed error metric and shifted/resized comparison images
        """

        im1 = self.copy()
        if im1.polrep!=im2.polrep:
            raise Exception("In find_shift, im1 and im2 must have the same polrep!")
        if im1.pol_prim!=im2.pol_prim:
            raise Exception("In find_shift, im1 and im2 must have the same pol_prim!")

        # Shift the comparison image to maximize normalized cross-corr.
        [idx, xcorr, im1_pad, im2_pad] = im1.find_shift(im2, psize=psize, target_fov=target_fov, beamparams=beamparams, blur_frac=blur_frac, blursmall=blursmall)

        if type(shift)!=bool:
            idx = shift

        im2_shift = im2_pad.shift(idx)

        # Compute error metrics
        error = []
        if 'nxcorr' in metric:
            error.append( xcorr[ idx[0], idx[1] ] / (im1_pad.xdim * im1_pad.ydim) )
        if 'nrmse' in metric:
            error.append( np.sqrt( np.sum( ( (im1_pad.imvec - im2_shift.imvec)**2 * im1_pad.psize**2  ) ) / np.sum( (im1_pad.imvec )**2 * im1_pad.psize**2 ) ) )
        if 'rssd' in metric:
            error.append( np.sqrt( np.sum(  (im1_pad.imvec - im2_shift.imvec)**2 ) * im1_pad.psize**2 ) )

        return (error, im1_pad, im2_shift)

    def align_images(self, im_list, shift=True, final_fov=False, scale='lin', gamma=0.5, dynamic_range=[1.e3]):

        """Align all the images in im_list to the current image (self)
           Aligns all images by comparison of the primary pol image.

           Args:
               im_list (list): list of images to align to the current image
               shift (list): list of manual image shifts, otherwise use the shift from maximum cross-correlation
               final_fov (float): fov of the comparison image (rad). If False it is the largestinput image fov

               scale (str) : compare images in 'log','lin',or 'gamma' scale
               gamma (float): exponent for gamma scale comparison
               dynamic_range (float): dynamic range for log and gamma scale comparisons

           Returns:
               (tuple): (im_list_shift, shifts, im0_pad) of shifted images, shift vectors, and padded reference image
        """

        im0 = self.copy()
        if not np.all(im0.polrep == np.array([im.polrep for im in im_list])):
            raise Exception("In align_images, all images must have the same polrep!")
        if not np.all(im0.pol_prim == np.array([im.pol_prim for im in im_list])):
            raise Exception("In find_shift, all images must have the same pol_prim!")

        if len(dynamic_range)==1:
            dynamic_range = dynamic_range * np.ones(len(im_list)+1)

        useshift = True
        if type(shift)==bool:
            useshift = False

        # Find the minimum psize and the maximum field of view
        psize = im0.psize
        max_fov = np.max([im0.xdim*im0.psize, im0.ydim*im0.psize])
        for i in range(0, len(im_list)):
            psize = np.min([psize, im_list[i].psize])
            max_fov = np.max([max_fov, im_list[i].xdim*im_list[i].psize, im_list[i].ydim*im_list[i].psize])

        if not final_fov:
            final_fov = max_fov

        # Shift all images in the list
        im_list_shift = []
        shifts = []
        for i in range(0, len(im_list)):
            (idx, _, im0_pad_orig, im_pad) = im0.find_shift(im_list[i], target_fov=2*max_fov, psize=psize,
                                                            scale=scale, gamma=gamma, dynamic_range=dynamic_range[i+1])

            if i==0:
                npix = int(im0_pad_orig.xdim/2)
                im0_pad = im0_pad_orig.regrid_image(final_fov, npix)
            if useshift:
                idx = shift[i]

            tmp = im_pad.shift(idx)
            shifts.append(idx)
            im_list_shift.append(tmp.regrid_image(final_fov, npix))

        return (im_list_shift, shifts, im0_pad)

    def find_shift(self, im2, psize=None, target_fov=None,
                         beamparams=[1., 1., 1.], blur_frac = 0.0, blursmall=False,
                         scale='lin', gamma=0.5, dynamic_range=1.e3):

        """Find image shift that maximizes normalized cross correlation with a second image im2.
           Finds shift only by comparison of the primary pol image.

           Args:
               im2 (Image): image with respect with to switch
               psize (float): pixel size of comparison image (rad). If None it is the smallest of the input image pizel sizes
               target_fov (float): fov of the comparison image (rad). If None it is twice the largest fov of the input images

               beamparams (list): the nominal Gaussian beam parameters [fovx, fovy, position angle]
               blur_frac (float): fractional beam to blur each image to before comparison
               blursmall (bool) : True to blur the unpadded image rather than the large image.

               scale (str) : compare images in 'log','lin',or 'gamma' scale
               gamma (float): exponent for gamma scale comparison
               dynamic_range (float): dynamic range for log and gamma scale comparisons

           Returns:
               (tuple): (errormetric, im1_pad, im2_shift) of computed error metric and shifted/resized comparison images

        """

        im1 = self.copy()
        if im1.polrep!=im2.polrep:
            raise Exception("In find_shift, im1 and im2 must have the same polrep!")
        if im1.pol_prim!=im2.pol_prim:
            raise Exception("In find_shift, im1 and im2 must have the same pol_prim!")

        # Find maximum FOV and minimum pixel size for comparison
        if target_fov==None:
            max_fov = np.max([im1.fovx(), im1.fovy(), im2.fovx(), im2.fovy()])
            target_fov = 2*max_fov
        if psize==None:
            psize = np.min([im1.psize, im2.psize])

        npix = int( target_fov / psize )

        # Blur images, then pad
        if ( (blur_frac > 0.0) * (blursmall==True) ):
            im1 = im1.blur_gauss(beamparams, blur_frac)
            im2 = im2.blur_gauss(beamparams, blur_frac)

        im1_pad = im1.regrid_image(target_fov, npix)
        im2_pad = im2.regrid_image(target_fov, npix)

        # or, pad images, then blur
        if ((blur_frac > 0.0) * (blursmall==False)):
            im1_pad = im1_pad.blur_gauss(beamparams, blur_frac)
            im2_pad = im2_pad.blur_gauss(beamparams, blur_frac)

        # Rescale the image vectors into log or gamma scale
        # TODO -- what about negative values? threshold?
        im1_pad_vec = im1_pad.imvec
        im2_pad_vec = im2_pad.imvec
        if scale=='log':
            im1_pad_vec[im1_pad_vec<0.0] = 0.0
            im1_pad_vec = np.log(im1_pad_vec + np.max(im1_pad_vec)/dynamic_range)
            im2_pad_vec[im2_pad_vec<0.0] = 0.0
            im2_pad_vec = np.log(im2_pad_vec + np.max(im2_pad_vec)/dynamic_range)
        if scale=='gamma':
            im1_pad_vec[im1_pad_vec<0.0] = 0.0
            im1_pad_vec = (im1_pad_vec + np.max(im1_pad_vec)/dynamic_range)**(gamma)
            im2_pad_vec[im2_pad_vec<0.0] = 0.0
            im2_pad_vec = (im2_pad_vec + np.max(im2_pad_vec)/dynamic_range)**(gamma)

        # Normalize images and compute cross correlation with FFT
        im1_norm = ( im1_pad_vec.reshape(im1_pad.ydim, im1_pad.xdim) - np.mean(im1_pad_vec) ) / np.std(im1_pad_vec)
        im2_norm = ( im2_pad_vec.reshape(im2_pad.ydim, im2_pad.xdim) - np.mean(im2_pad_vec) ) / np.std(im2_pad_vec)

        fft_im1 = np.fft.fft2( im1_norm )
        fft_im2 = np.fft.fft2( im2_norm )

        xcorr =  np.real( np.fft.ifft2( fft_im1 * np.conj(fft_im2) ) )

        # Find idx of shift that maximized cross-correlation
        idx = np.unravel_index(xcorr.argmax(), xcorr.shape)

        return [idx, xcorr, im1_pad, im2_pad]

    def hough_ring(self, edgetype='canny',thresh=0.2, num_circles=3, radius_range=None,
                         return_type='rad', display_results=True):

        """Use a circular hough transform to find a circle in the image
           Returns metrics only for the primary polarization imvec!

           Args:
               num_circles (int) : number of circles to return
               radius_range (tuple): range of radii to search in Hough transform, in radian
               edgetype (str): edge detection type, 'gradient' or 'canny'
               thresh(float): fractional threshold for the gradient image
               display_results (bool): True to display results of the fit
               return_type (str): 'rad' to return results in radian, 'pixel' to return in pixel units
           Returns:
               list : a list of fitted circle (xpos, ypos, radius, objFunc), with coordinates and radius in radian
        """

        # coordinate values
        pdim = self.psize
        xlist = np.arange(0,-self.xdim,-1)*pdim + (pdim*self.xdim)/2.0 - pdim/2.0
        ylist = np.arange(0,-self.ydim,-1)*pdim + (pdim*self.ydim)/2.0 - pdim/2.0

        #normalize to range 0, 1
        im = self.copy()
        maxval = np.max(im.imvec)
        meanval = np.mean(im.imvec)
        im_norm = im.imvec / (maxval + .01*meanval)
        im_norm = im_norm
        im_norm = im_norm.astype('float') # is it a problem if it's double??
        im_norm[np.isnan(im.imvec)] = 0 #mask nans to 0
        im.imvec = im_norm

        # detect edges
        if edgetype=='canny':
            imarr = im.imvec.reshape(self.ydim,self.xdim)
            edges = canny(imarr, sigma=0, high_threshold=thresh,  low_threshold=0.01)
            im_edges = self.copy()
            im_edges.imvec = edges.flatten()

        elif edgetype=='grad':
            im_edges = self.grad()
            if not (thresh is None):
                thresh_val = thresh*np.max(im_edges.imvec)
                mask = im_edges.imvec > thresh_val
                #im_edges.imvec[mask] = 1
                im_edges.imvec[~mask] = 0
                edges = im_edges.imvec.reshape(self.ydim, self.xdim)
        else:
            im_edges = im.copy()
            if not (thresh is None):
                thresh_val = thresh*np.max(im_edges.imvec)
                mask = im_edges.imvec > thresh_val
                #im_edges.imvec[mask] = 1f
                im_edges.imvec[~mask] = 0
                edges = im_edges.imvec.reshape(self.ydim, self.xdim)

        # define radius range for Hough transform search
        if radius_range is None:
            hough_radii = np.arange(int(10*RADPERUAS/self.psize), int(50*RADPERUAS/self.psize))
        else:
            hough_radii = np.linspace(radius_range[0]/self.psize, radius_range[0]/self.psize, 25)

        # perform the hough transform and select the most prominent circles
        hough_res = hough_circle(edges, hough_radii)
        accums, cy, cx, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=num_circles)
        accum_tot = np.sum(accums)

        # print results, plot circles, and return
        outlist = []
        if display_results:
            plt.ion()
            fig = self.display()
            ax = fig.gca()

        i=0
        colors = ['b','r','w','lime','magenta','aqua']
        for accum, center_y, center_x, radius in zip(accums, cy, cx, radii):
            accum_frac = accum/accum_tot
            if return_type=='rad':
                x_rad = xlist[int(np.round(center_x))]
                y_rad = ylist[int(np.round(center_y))]
                r_rad = radius*self.psize
                outlist.append([x_rad,y_rad,r_rad,accum_frac])
            else:
                outlist.append([center_x,center_y,radius,accum_frac])
            print(accum_frac)
            print("%i ring diameter: %0.1f microarcsec"% (i, 2*radius*pdim/RADPERUAS))
            if display_results:
                if i>len(colors): color=colors[-1]
                else: color = colors[i]
                circ = mpl.patches.Circle((center_y,center_x),radius, fill=False, color=color)
                ax.add_patch(circ)
            i+=1

        return outlist

    def fit_gauss(self, units='rad'):

        """Determine the Gaussian parameters that short baselines would measure for the source by diagonalizing the image covariance matrix.
           Returns parameters only for the primary polarization imvec!

           Args:
               units (string): 'rad' returns values in radians, 'natural' returns FWHM in uas and PA in degrees

           Returns:
               (tuple) : a tuple (fwhm_maj, fwhm_min, theta) of the fit Gaussian parameters in radians or natural units.
        """

        (x1,y1) = self.centroid()
        pdim = self.psize
        im = self.imvec

        xlist = np.arange(0,-self.xdim,-1)*pdim + (pdim*self.xdim)/2.0 - pdim/2.0
        ylist = np.arange(0,-self.ydim,-1)*pdim + (pdim*self.ydim)/2.0 - pdim/2.0

        x2 = (np.sum(np.outer(0.0*ylist+1.0, (xlist - x1)**2).ravel()*im)/np.sum(im))
        y2 = (np.sum(np.outer((ylist - y1)**2, 0.0*xlist+1.0).ravel()*im)/np.sum(im))
        xy = (np.sum(np.outer(ylist - y1, xlist - x1).ravel()*im)/np.sum(im))

        eig = np.linalg.eigh(np.array(((x2,xy),(xy,y2))))
        gauss_params = np.array((eig[0][1]**0.5*(8.*np.log(2.))**0.5, eig[0][0]**0.5*(8.*np.log(2.))**0.5, np.mod(np.arctan2(eig[1][1][0],eig[1][1][1]) + np.pi, np.pi)))
        if units == 'natural':
            gauss_params[0] /= RADPERUAS
            gauss_params[1] /= RADPERUAS
            gauss_params[2] *= 180./np.pi

        return gauss_params

    def fit_gauss_empirical(self, paramguess=None):

        """Determine the Gaussian parameters that short baselines would measure for the source by fitting short baselines.
           Returns parameters only for the primary polarization imvec!

           Args:
                paramguess (tuple): Initial guess (fwhm_maj, fwhm_min, theta) of fit parameters

           Returns:
               (tuple) : a tuple (fwhm_maj, fwhm_min, theta) of the fit Gaussian parameters in radians.
        """


        # This could be done using moments of the intensity distribution (self.fit_gauss)
        # but we'll use the visibility approach
        u_max = 1.0/(self.psize * self.xdim)/5.0
        uv = np.array([[u, v] for u in np.arange(-u_max,u_max*1.001,u_max/4.0) for v in np.arange(-u_max,u_max*1.001,u_max/4.0)])
        u = uv[:,0]
        v = uv[:,1]
        vis = np.dot(ehtim.obsdata.ftmatrix(self.psize, self.xdim, self.ydim, uv, pulse=self.pulse), self.imvec)

        if paramguess == None:
            paramguess = (self.psize * self.xdim / 4.0, self.psize * self.xdim / 4.0, 0.)

        def errfunc(p):
            vismodel = gauss_uv(u,v, self.total_flux(), p, x=0., y=0.)
            err = np.sum((np.abs(vis)-np.abs(vismodel))**2)
            return err

        optdict = {'maxiter':5000, 'maxfev':5000, 'xtol': paramguess[0]/1e9, 'ftol': 1e-10} # minimizer params
        res = opt.minimize(errfunc, paramguess, method='Nelder-Mead',options=optdict)

        # Return in the form [maj, min, PA]
        x = res.x
        x[0] = np.abs(x[0])
        x[1] = np.abs(x[1])
        x[2] = np.mod(x[2], np.pi)
        if x[0] < x[1]:
            maj = x[1]
            x[1] = x[0]
            x[0] = maj
            x[2] = np.mod(x[2] + np.pi/2.0, np.pi)

        return x

    def contour(self, contour_levels=[0.1, 0.25, 0.5, 0.75], 
                      contour_cfun=None, color='w', legend=True, show_im=True,
                      cfun='afmhot',scale='lin', interp='gaussian', gamma=0.5, dynamic_range=1.e3,
                      plotp=False, nvec=20, pcut=0.01, label_type='ticks', has_title=True,
                      has_cbar=True, cbar_lims=(), cbar_unit = ('Jy', 'pixel'),
                      contour_im=False, power=0, beamcolor='w',
                      export_pdf="", show=True, beamparams=None, cbar_orientation="vertical", 
                      scale_lw=1, beam_lw=1, cbar_fontsize=12, axis=None, scale_fontsize=12):

        """Display the image in a contour plot.

           Args:
               contour_levels (arr): the fractional contour levels relative to the max flux plotted
               contour_cfun (pyplot colormap function): the function used to get the RGB colors 
               legend (bool): True to show a legend that says what each contour line corresponds to
               cfun (str): matplotlib.pyplot color function
               scale (str): image scaling in ['log','gamma','lin']
               interp (str): image interpolation 'gauss' or 'lin'

               gamma (float): index for gamma scaling
               dynamic_range (float): dynamic range for log and gamma scaling

               plotp (bool): True to plot linear polarimetic image
               nvec (int): number of polarimetric vectors to plot
               pcut (float): minimum stokes P value for displaying polarimetric vectors 
                             as fraction of maximum Stokes I pixel
               label_type (string): specifies the type of axes labeling: 'ticks', 'scale', 'none'
               has_title (bool): True if you want a title on the plot
               has_cbar (bool): True if you want a colorbar on the plot
               cbar_lims (tuple): specify the lower and upper limit of the colorbar
               cbar_unit (tuple of strings): the unit of each pixel for the colorbar: 'Jy', 'm-Jy', '$\mu$Jy'

               export_pdf (str): path to exported PDF with plot
               show (bool): Display the plot if true
               show_im (bool): Display the image with the contour plot if True

           Returns:
               (matplotlib.figure.Figure): figure object with image

        """


        image = self.copy()

        #or some generalized version for image sizes
        y = np.linspace(0, image.ydim, image.ydim)
        x = np.linspace(0, image.xdim, image.xdim)


        #make the image grid
        z = image.imvec.reshape((image.ydim ,image.xdim ))
        maxz = max(image.imvec)
        if axis is None:
            ax = plt.gca()

        elif axis is not None:
            ax = axis
            plt.sca(axis)

        if show_im:
            if axis is not None:
                axis = image.display(cfun=cfun,scale=scale, interp=interp, gamma=gamma, dynamic_range=dynamic_range,
                      plotp=plotp, nvec=nvec, pcut=pcut, label_type=label_type, has_title=has_title,
                      has_cbar=has_cbar, cbar_lims=cbar_lims, cbar_unit=cbar_unit, beamparams=beamparams, 
                      cbar_orientation=cbar_orientation, scale_lw=1, beam_lw=1, cbar_fontsize=cbar_fontsize, axis=axis,             
                      scale_fontsize=scale_fontsize,power=power,beamcolor=beamcolor)
            else:   
                image.display(cfun=cfun,scale=scale, interp=interp, gamma=gamma, dynamic_range=dynamic_range,
                      plotp=plotp, nvec=nvec, pcut=pcut, label_type=label_type, has_title=has_title,
                      has_cbar=has_cbar, cbar_lims=cbar_lims, cbar_unit=cbar_unit, beamparams=beamparams,
                      cbar_orientation=cbar_orientation, scale_lw=1, beam_lw=1, cbar_fontsize=cbar_fontsize, 
                      axis=None, scale_fontsize=scale_fontsize, power=power,beamcolor=beamcolor)
        else:
            if contour_im==False:
                image.imvec = 0.0*image.imvec
            else:
                image = contour_im.copy()
            
            if axis is not None:
                axis = image.display(cfun=cfun,scale=scale, interp=interp, gamma=gamma, dynamic_range=dynamic_range,
                      plotp=plotp, nvec=nvec, pcut=pcut, label_type=label_type, has_title=has_title,
                      has_cbar=has_cbar, cbar_lims=cbar_lims, cbar_unit=cbar_unit, beamparams=beamparams, 
                      cbar_orientation=cbar_orientation, scale_lw=1, beam_lw=1, cbar_fontsize=cbar_fontsize, 
                      axis=axis, scale_fontsize=scale_fontsize,power=power,beamcolor=beamcolor)
            else:   
                image.display(cfun=cfun,scale=scale, interp=interp, gamma=gamma, dynamic_range=dynamic_range,
                      plotp=plotp, nvec=nvec, pcut=pcut, label_type=label_type, has_title=has_title,
                      has_cbar=has_cbar, cbar_lims=cbar_lims, cbar_unit=cbar_unit, beamparams=beamparams,
                      cbar_orientation=cbar_orientation, scale_lw=1, beam_lw=1, cbar_fontsize=cbar_fontsize, axis=None,
                      scale_fontsize=scale_fontsize,power=power,beamcolor=beamcolor)

        if axis is None: ax = plt.gcf()
        if axis is not None: ax = axis

        if axis is not None:
            ax = axis
            plt.sca(axis)

        count = 0.

        for level in contour_levels:
            if not(contour_cfun is None):
                rgbval = contour_cfun(count/len(contour_levels))
                rgbstring = '#%02x%02x%02x' % (rgbval[0]*256, rgbval[1]*256, rgbval[2]*256)
            else:
                rgbstring=color
            cs = plt.contour(x,y,z, levels=[level*maxz],colors=rgbstring, cmap=None )
            count += 1
            cs.collections[0].set_label(str(int(level*100)) + '%')
        if legend:
            plt.legend()

        if show:
            plt.ion()
            plt.show(block=False)

        if export_pdf != "":
            ax.savefig(export_pdf, bbox_inches='tight', pad_inches = 0)

        elif axis is not None:
            return axis
        return ax


    def display(self, pol=None, cfun='afmhot', interp='gaussian',
                      scale='lin',gamma=0.5, dynamic_range=1.e3,
                      plotp=False, nvec=20, pcut=0.1, log_offset=False,
                      label_type='ticks', has_title=True, alpha=1,
                      has_cbar=True, only_cbar=False, cbar_lims=(), cbar_unit = ('Jy', 'pixel'),
                      export_pdf="", pdf_pad_inches=0.0, show=True, beamparams=None, cbar_orientation="vertical", scinot=False,
                      scale_lw=1, beam_lw=1, cbar_fontsize=12, axis=None, scale_fontsize=12, power=0, beamcolor='w', dpi=500):

        """Display the image.

           Args:
               pol (str): which polarization image to plot. Default is self.pol_prim
               cfun (str): matplotlib.pyplot color function
               interp (str): image interpolation 'gauss' or 'lin'

               scale (str): image scaling in ['log','gamma','lin']
               gamma (float): index for gamma scaling
               dynamic_range (float): dynamic range for log and gamma scaling

               plotp (bool): True to plot linear polarimetic image
               nvec (int): number of polarimetric vectors to plot
               pcut (float): minimum stokes P value for displaying polarimetric vectors as fraction of maximum Stokes I pixel

               label_type (string): specifies the type of axes labeling: 'ticks', 'scale', 'none'
               has_title (bool): True if you want a title on the plot
               has_cbar (bool): True if you want a colorbar on the plot
               cbar_lims (tuple): specify the lower and upper limit of the colorbar
               cbar_unit (tuple): specifies the unit of the colorbar: eg ('Jy','pixel'),('m-Jy','$\mu$as$^2$'),['Tb'] (i.e., brightness temperature)
               beamparams (list): [fwhm_maj, fwhm_min, theta], set to plot beam contour

               export_pdf (str): path to exported PDF with plot
               show (bool): Display the plot if true
               scinot (bool): Display numbers/units in scientific notation
               scale_lw (float): Linewidth of the scale overlay
               beam_lw (float): Linewidth of the beam overlay
               cbar_fontsize (float): Fontsize of the text elements of the colorbar
               axis (matplotlib.axes.Axes): An axis object, which will be edited so as to not overwrite your plot with a figure object. If none, creates a new figure.
               scale_fontsize (float): Fontsize of the scale label
               power (float): Passed to colorbar for division of ticks by 1e(power)
               beamcolor (str): color of the beam overlay 

           Returns:
               (matplotlib.figure.Figure): figure object with image

        """

        if (interp in ['gauss', 'gaussian', 'Gaussian', 'Gauss']):
            interp = 'gaussian'
        else:
            interp = 'linear'

        if not(beamparams is None or beamparams is False):
            if beamparams[0]>self.fovx() or beamparams[1]>self.fovx():
                raise Exception("beam FWHM must be smaller than fov!")

        if self.polrep=='stokes' and pol is None: pol='I'
        elif self.polrep=='circ' and pol is None: pol='RR'

        if only_cbar:
            has_cbar = True
            label_type = 'none'
            has_title = False

        if axis is None:
            f = plt.figure()
            plt.clf()

        if axis is not None:
            plt.sca(axis)

        # Get unit scale factor
        factor = 1.
        fluxunit='Jy'
        areaunit='pixel'

        if cbar_unit[0] in ['m-Jy','mJy']:
            fluxunit = 'mJy'
            factor *= 1.e3
        elif cbar_unit[0] in ['muJy','$\mu$-Jy','$\mu$Jy']:
            fluxunit = '$\mu$Jy'
            factor *= 1.e6
        elif cbar_unit[0] == 'Tb':
            factor = 3.254e13/(self.rf**2 * self.psize**2)
            fluxunit = 'Brightness Temperature (K)'
            areaunit = ''
            if power!=0:
                fluxunit = (r'Brightness Temperature ($10^{{' + str(power) + '}}$ K)')
            else:
                fluxunit = 'Brightness Temperature (K)'
        elif cbar_unit[0] in ['Jy']:
            fluxunit = 'Jy'
            factor *= 1.
        else:
            factor = 1
            fluxunit = cbar_unit[0]
            areaunit = ''
        #elif cbar_unit[0] != 'Jy':
            #raise ValueError('cbar_unit ' + cbar_unit[0] + ' is not a possible option')

        if len(cbar_unit) == 1 or cbar_unit[0] == 'Tb':
            factor *= 1.

        elif cbar_unit[1] == 'pixel':
            factor *= 1.
            if power!=0:
                areaunit = areaunit + (r' ($10^{{' + str(power) + '}}$ K)')

        elif cbar_unit[1] in ['$arcseconds$^2$','as$^2$','as2']:
            areaunit='as$^2$'
            fovfactor = self.xdim*self.psize*(1/RADPERAS)
            factor *= (1./fovfactor)**2 / (1./self.xdim)**2
            if power!=0:
                areaunit = areaunit + (r' ($10^{{' + str(power) + '}}$ K)')

        elif cbar_unit[1] in ['$\m-arcseconds$^2$','mas$^2$','mas2']:
            areaunit='mas$^2$'
            fovfactor = self.xdim*self.psize*(1/RADPERUAS) / 1000.
            factor *= (1./fovfactor)**2 / (1./self.xdim)**2
            if power!=0:
                areaunit = areaunit + (r' ($10^{{' + str(power) + '}}$ K)')

        elif cbar_unit[1] in ['$\mu$-arcseconds$^2$','$\mu$as$^2$','muas2']:
            areaunit='$\mu$as$^2$'
            fovfactor = self.xdim*self.psize*(1/RADPERUAS)
            factor *= (1./fovfactor)**2 / (1./self.xdim)**2
            if power!=0:
                areaunit = areaunit + (r' ($10^{{' + str(power) + '}}$ K)')

        elif cbar_unit[1]=='beam':
            if (beamparams is None or beamparams==False):
                print("Cannot convert to Jy/beam without beamparams!")
            else:
                areaunit='beam'
                beamarea=(2.0*np.pi*beamparams[0]*beamparams[1]/(8.0*np.log(2)))
                factor = beamarea/(self.psize**2)
                if power!=0:
                    areaunit = areaunit + (r' ($10^{{' + str(power) + '}}$ K)')

        else:
            raise ValueError('cbar_unit ' + cbar_unit[1] + ' is not a possible option')

        if not plotp:
            # Plot single polarization image
            try:
                imvec = np.array(self._imdict[pol]).reshape(-1) / (10.**power)
            except KeyError:
                try:
                    if self.polrep=='stokes': im2 = self.switch_polrep('circ')
                    elif self.polrep=='circ': im2 = self.switch_polrep('stokes')
                    imvec = np.array(im2._imdict[pol]).reshape(-1) / (10.**power)
                except KeyError:
                    raise Exception("Cannot make pol %s image in display()!" % pol)

            if np.any(np.imag(imvec)):
                print('casting complex image to abs value')
                imvec = np.real(imvec)

            imvec = imvec * factor
            imarr = imvec.reshape(self.ydim, self.xdim)
            unit = fluxunit
            if areaunit != '':
                unit += ' / ' + areaunit

            if scale=='log':
                if (imarr < 0.0).any():
                    print('clipping values less than 0 in display')
                    imarr[imarr<0.0] = 0.0
                if log_offset:
                    imarr = np.log10(imarr + log_offset / dynamic_range)
                else:
                    imarr = np.log10(imarr + np.max(imarr) / dynamic_range)
                unit = r'$\log_{10}$(' + unit + ')'

            if scale=='gamma':
                if (imarr < 0.0).any():
                    print('clipping values less than 0 in display')
                    imarr[imarr<0.0] = 0.0
                imarr = (imarr + np.max(imarr)/dynamic_range)**(gamma)
                unit = '(' + unit + ')^' + str(gamma)

            if cbar_lims:
                cbar_lims[0] = cbar_lims[0] / (10.**power)
                cbar_lims[1] = cbar_lims[1] / (10.**power)
                imarr[imarr>cbar_lims[1]] = cbar_lims[1]
                imarr[imarr<cbar_lims[0]] = cbar_lims[0]

            if has_title:
                plt.title("%s %.2f GHz %s" % (self.source, self.rf/1e9, pol), fontsize=16)

            if cbar_lims:
                im = plt.imshow(imarr, alpha=alpha, cmap=plt.get_cmap(cfun), interpolation=interp, vmin=cbar_lims[0], vmax=cbar_lims[1])
            else:
                im = plt.imshow(imarr, alpha=alpha, cmap=plt.get_cmap(cfun), interpolation=interp)

            if not(beamparams is None or beamparams is False):
                beamparams = [beamparams[0], beamparams[1], beamparams[2],
                              -.35*self.fovx(), -.35*self.fovy()]
                beamimage = self.copy()
                beamimage.imvec *= 0
                beamimage = beamimage.add_gauss(1, beamparams)
                halflevel = 0.5*np.max(beamimage.imvec)
                beamimarr = (beamimage.imvec).reshape(beamimage.ydim,beamimage.xdim)
                plt.contour(beamimarr, levels=[halflevel], colors=beamcolor, linewidths=beam_lw)


            if has_cbar:
                if only_cbar: im.set_visible(False)
                cb = plt.colorbar(im, fraction=0.046, pad=0.04, orientation=cbar_orientation)
                cb.set_label(unit, fontsize=float(cbar_fontsize))
                        
                if cbar_fontsize != 12: cb.set_label(unit, fontsize=float(cbar_fontsize)/1.5)
                cb.ax.tick_params(labelsize=cbar_fontsize)
                        
                if cbar_lims:
                    plt.clim(cbar_lims[0],cbar_lims[1])
                if scinot:
                    cb.formatter.set_powerlimits((0,0))
                    cb.update_ticks()
        

        else: #plot Stokes parameters!
            im_stokes = self.switch_polrep(polrep_out='stokes')
            imvec = np.array(im_stokes.imvec).reshape(-1) / (10**power)
            qvec = np.array(im_stokes.qvec).reshape(-1) / (10**power)
            uvec = np.array(im_stokes.uvec).reshape(-1) / (10**power)
            vvec = np.array(im_stokes.vvec).reshape(-1) / (10**power)

            if len(imvec)==0: imvec = np.zeros(im_stokes.ydim*im_stokes.xdim)
            if len(qvec)==0: qvec = np.zeros(im_stokes.ydim*im_stokes.xdim)
            if len(uvec)==0: uvec = np.zeros(im_stokes.ydim*im_stokes.xdim)
            if len(vvec)==0: vvec = np.zeros(im_stokes.ydim*im_stokes.xdim)

            imvec *= factor
            qvec *= factor
            uvec *= factor
            vvec *= factor

            imarr = (imvec).reshape(im_stokes.ydim, im_stokes.xdim)
            qarr = (qvec).reshape(im_stokes.ydim, im_stokes.xdim)
            uarr = (uvec).reshape(im_stokes.ydim, im_stokes.xdim)
            varr = (vvec).reshape(im_stokes.ydim, im_stokes.xdim)

            unit = fluxunit + ' / ' + areaunit

            # only the  stokes I image gets transformed! TODO
            imarr2 = imarr.copy()
            if scale=='log':
                if (imarr2 < 0.0).any():
                    print('clipping values less than 0 in display')
                    imarr2[imarr2<0.0] = 0.0
                imarr2 = np.log10(imarr2 + np.max(imarr2) / dynamic_range)
                unit = r'$\log_{10}$(' + unit + ')'

            if scale=='gamma':
                if (imarr2 < 0.0).any():
                    print('clipping values less than 0 in display')
                    imarr2[imarr2<0.0] = 0.0
                imarr2 = (imarr2 + np.max(imarr2)/dynamic_range)**(gamma)
                unit = '(' + unit + ')^gamma'

            if cbar_lims:
                cbar_lims[0] = cbar_lims[0] / (10.**power)
                cbar_lims[1] = cbar_lims[1] / (10.**power)
                imarr2[imarr2>cbar_lims[1]] = cbar_lims[1]
                imarr2[imarr2<cbar_lims[0]] = cbar_lims[0]

            # polarization ticks
            thin = self.xdim//nvec
            mask = (imvec).reshape(self.ydim, self.xdim) > pcut * np.max(imvec)
            mask2 = mask[::thin, ::thin]
            x = (np.array([[i for i in range(self.xdim)] for j in range(self.ydim)])[::thin, ::thin])[mask2]
            y = (np.array([[j for i in range(self.xdim)] for j in range(self.ydim)])[::thin, ::thin])[mask2]
            a = (-np.sin(np.angle(qvec+1j*uvec)/2).reshape(self.ydim, self.xdim)[::thin, ::thin])[mask2]
            b = ( np.cos(np.angle(qvec+1j*uvec)/2).reshape(self.ydim, self.xdim)[::thin, ::thin])[mask2]

            m = (np.abs(qvec + 1j*uvec)/imvec).reshape(self.ydim, self.xdim)
            m[np.logical_not(mask)] = 0

            voi = (vvec/imvec).reshape(self.ydim, self.xdim)
            voi[np.logical_not(mask)] = 0

            # Little pol plots
            maxval = 1.1*np.max((np.max(np.abs(uarr)),np.max(np.abs(qarr)),np.max(np.abs(varr))))

            ax = plt.subplot2grid((2,5),(0,0))
            plt.imshow(imarr, cmap=plt.get_cmap('bwr'), interpolation=interp, vmin=-maxval, vmax=maxval)
            plt.contour(imarr, colors='k',linewidth=.25)
            ax.set_xticks([])
            ax.set_yticks([])
            if has_title: plt.title('I')

            ax = plt.subplot2grid((2,5),(0,1))
            plt.imshow(varr, cmap=plt.get_cmap('bwr'), interpolation=interp, vmin=-maxval, vmax=maxval)
            #plt.contour(varr, colors='k',linewidth=.25)
            ax.set_xticks([])
            ax.set_yticks([])
            if has_title: plt.title('V')

            ax = plt.subplot2grid((2,5),(1,0))
            plt.imshow(qarr, cmap=plt.get_cmap('bwr'), interpolation=interp, vmin=-maxval, vmax=maxval)
            plt.contour(qarr, colors='k',linewidth=.25)
            ax.set_xticks([])
            ax.set_yticks([])
            if has_title: plt.title('Q')

            ax = plt.subplot2grid((2,5),(1,1))
            plt.imshow(uarr, cmap=plt.get_cmap('bwr'), interpolation=interp, vmin=-maxval, vmax=maxval)
            plt.contour(uarr, colors='k',linewidth=.25)
            ax.set_xticks([])
            ax.set_yticks([])
            if has_title: plt.title('U')

            # V/I plot
            ax = plt.subplot2grid((2,5),(0,2))
            plt.imshow(voi, cmap=plt.get_cmap('seismic'), interpolation=interp, vmin=-1, vmax=1)
            if has_title: plt.title('V/I')
            ax.set_xticks([])
            ax.set_yticks([])

            # m plot
            ax = plt.subplot2grid((2,5),(1,2))
            plt.imshow(m, cmap=plt.get_cmap('seismic'), interpolation=interp, vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if has_title: plt.title('m')
            plt.quiver(x, y, a, b,
                   headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                   width=.01*self.xdim, units='x', pivot='mid', color='k', angles='uv', scale=1.0/thin)
            plt.quiver(x, y, a, b,
                   headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                   width=.005*self.xdim, units='x', pivot='mid', color='w', angles='uv', scale=1.1/thin)

            # Big Stokes I plot
            ax = plt.subplot2grid((2,5),(0,3), rowspan=2, colspan=2)
            if cbar_lims:
                im = plt.imshow(imarr2, cmap=plt.get_cmap(cfun), interpolation=interp, vmin=cbar_lims[0], vmax=cbar_lims[1])
            else:
                im = plt.imshow(imarr2, cmap=plt.get_cmap(cfun), interpolation=interp)

            plt.quiver(x, y, a, b,
                   headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                   width=.01*self.xdim, units='x', pivot='mid', color='k', angles='uv', scale=1.0/thin)
            plt.quiver(x, y, a, b,
                   headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                   width=.005*self.xdim, units='x', pivot='mid', color='w', angles='uv', scale=1.1/thin)

            if not(beamparams is None or beamparams==False):
                beamparams = [beamparams[0], beamparams[1], beamparams[2],
                              -.35*self.fovx(), -.35*self.fovy()]
                beamimage = self.copy()
                beamimage.imvec *= 0
                beamimage = beamimage.add_gauss(1, beamparams)
                halflevel = 0.5*np.max(beamimage.imvec)
                beamimarr = (beamimage.imvec).reshape(beamimage.ydim,beamimage.xdim)
                plt.contour(beamimarr, levels=[halflevel], colors=beamcolor, linewidths=beam_lw)

            if has_cbar:
                cbar = plt.colorbar(im, fraction=0.046, pad=0.04, label=unit, orientation=cbar_orientation)
                cbar.ax.tick_params(labelsize=cbar_fontsize) 
                if cbar_lims:
                    plt.clim(cbar_lims[0],cbar_lims[1])
            if has_title:
                plt.title("%s %.2f GHz %s" % (self.source, self.rf/1e9, 'I'), fontsize=12)
            f.subplots_adjust(hspace=-.5,wspace=0.1)


        # Label the plot
        ax = plt.gca()
        if label_type=='ticks':
            xticks = ticks(self.xdim, self.psize/RADPERAS/1e-6)
            yticks = ticks(self.ydim, self.psize/RADPERAS/1e-6)
            plt.xticks(xticks[0], xticks[1])
            plt.yticks(yticks[0], yticks[1])
            plt.xlabel('Relative RA ($\mu$as)')
            plt.ylabel('Relative Dec ($\mu$as)')

        elif label_type=='scale':
            plt.axis('off')
            fov_uas = self.xdim * self.psize / RADPERUAS # get the fov in uas
            roughfactor = 1./3. # make the bar about 1/3 the fov
            fov_scale = int( math.ceil(fov_uas * roughfactor / 10.0 ) ) * 10 # round around 1/3 the fov to nearest 10
            start = self.xdim * roughfactor / 3.0 # select the start location
            end = start + fov_scale/fov_uas * self.xdim # determine the end location based on the size of the bar
            plt.plot([start, end], [self.ydim-start-5, self.ydim-start-5], color="white", lw=scale_lw) # plot line
            plt.text(x=(start+end)/2.0, y=self.ydim-start+self.ydim/30, s= str(fov_scale) + " $\mu$as", color="white", ha="center", va="center", fontsize=scale_fontsize)
            ax = plt.gca()
            if axis is None:
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)

        elif label_type=='none':
            plt.axis('off')
            ax = plt.gca()
            if axis is None:
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)

        # Show or save to file
        if axis is not None:
            return axis
        if show:
            plt.show(block=False)

        if export_pdf != "":
            f.savefig(export_pdf, bbox_inches='tight', pad_inches = pdf_pad_inches, dpi=dpi)

        return f

    def overlay_display(self, im_list, color_coding = np.array([[1, 0, 1], [0, 1, 0]]),
                              export_pdf="", show=True, f=False,
                              shift=[0,0], final_fov=False, interp='gaussian',
                              scale='lin', gamma=0.5, dynamic_range=[1.e3], rescale=True):

        """Overlay primary polarization images of a list of images to compare structures.

           Args:
               im_list (list): list of images to align to the current image
               color_coding (numpy.array): Color coding of each image in the composite

               f (matplotlib.pyplot.figure): Figure to overlay on top of
               export_pdf (str): path to exported PDF with plot
               show (bool): Display the plot if true

               shift (list): list of manual image shifts, otherwise use the shift from maximum cross-correlation
               final_fov (float): fov of the comparison image (rad). If False it is the largestinput image fov

               scale (str) : compare images in 'log','lin',or 'gamma' scale
               gamma (float): exponent for gamma scale comparison
               dynamic_range (float): dynamic range for log and gamma scale comparisons

           Returns:
               (matplotlib.figure.Figure): figure object with image

        """

        if not f:
            f = plt.figure()
        plt.clf()

        if len(dynamic_range)==1:
            dynamic_range = dynamic_range * np.ones(len(im_list)+1)

        if type(shift) != np.ndarray and type(shift) != bool:
            shift = matlib.repmat(shift, len(im_list), 1)

        psize = self.psize
        max_fov = np.max([self.xdim*self.psize, self.ydim*self.psize])
        for i in range(0, len(im_list)):
            psize = np.min([psize, im_list[i].psize])
            max_fov = np.max([max_fov, im_list[i].xdim*im_list[i].psize, im_list[i].ydim*im_list[i].psize])

        if not final_fov:
            final_fov = max_fov

        (im_list_shift, shifts, im0_pad) = self.align_images(im_list, shift=shift, final_fov=final_fov, scale=scale, gamma=gamma,  dynamic_range=dynamic_range)


        unit = 'Jy/pixel'
        if scale=='log':
            unit = 'log(Jy/pixel)'
            im0_pad.imvec = np.log10(im0_pad.imvec + np.max(im0_pad.imvec)/dynamic_range[0])
            for i in range(0, len(im_list)):
                im_list_shift[i].imvec = np.log10(im_list_shift[i].imvec + np.max(im_list_shift[i].imvec)/dynamic_range[i+1])

        if scale=='gamma':
            unit = '(Jy/pixel)^gamma'
            im0_pad.imvec = (im0_pad.imvec + np.max(im0_pad.imvec)/dynamic_range[0])**(gamma)
            for i in range(0, len(im_list)):
                im_list_shift[i].imvec = (im_list_shift[i].imvec + np.max(im_list_shift[i].imvec)/dynamic_range[i+1])**(gamma)

        composite_img = np.zeros((im0_pad.ydim, im0_pad.xdim,3))
        for i in range(-1, len(im_list)):

            if i==-1:
                immtx = im0_pad.imvec.reshape(im0_pad.ydim, im0_pad.xdim)
            else:
                immtx = im_list_shift[i].imvec.reshape(im0_pad.ydim, im0_pad.xdim)

            if rescale:
                immtx = immtx - np.min(np.min(immtx))
                immtx = immtx / np.max(np.max(immtx))

            for c in range(0,3):
                composite_img[:,:,c] = composite_img[:,:,c] + (color_coding[i+1,c] * immtx)

        if rescale==False:
            composite_img  = composite_img - np.min(np.min(np.min(composite_img)))
            composite_img = composite_img / np.max(np.max(np.max(composite_img)))

        plt.subplot(111)
        plt.title('%s   MJD %i  %.2f GHz' % (self.source, self.mjd, self.rf/1e9), fontsize=20)
        im = plt.imshow(composite_img, interpolation=interp)
        #plt.colorbar(im, fraction=0.046, pad=0.04, label=unit)
        xticks = ticks(im0_pad.xdim, im0_pad.psize/RADPERAS/1e-6)
        yticks = ticks(im0_pad.ydim, im0_pad.psize/RADPERAS/1e-6)
        plt.xticks(xticks[0], xticks[1])
        plt.yticks(yticks[0], yticks[1])
        plt.xlabel('Relative RA ($\mu$as)')
        plt.ylabel('Relative Dec ($\mu$as)')

        if show:
            plt.show(block=False)

        if export_pdf != "":
            f.savefig(export_pdf, bbox_inches='tight')

        return (f, shift)

    def save_txt(self, fname):

        """Save image data to text file.

           Args:
                fname (str): path to output text file

           Returns:
        """

        ehtim.io.save.save_im_txt(self, fname)
        return

    def save_fits(self, fname):

        """Save image data to a fits file.

           Args:
                fname (str): path to output fits file

           Returns:
        """
        ehtim.io.save.save_im_fits(self, fname)
        return


###########################################################################################################################################
#Image creation functions
###########################################################################################################################################
def make_square(obs, npix, fov, pulse=PULSE_DEFAULT, polrep='stokes', pol_prim=None):

    """Make an empty square image.

       Args:
           obs (Obsdata): an obsdata object with the image metadata
           npix (int): the pixel size of each axis
           fov (float): the field of view of each axis in radians
           pulse (function): the function convolved with the pixel values for continuous image

           polrep (str): polarization representation, either 'stokes' or 'circ'
           pol_prim (str): The default image: I,Q,U or V for Stokes, or RR,LL,LR,RL for Circular
       Returns:
           (Image): an image object
    """

    outim = make_empty(npix, fov, obs.ra, obs.dec, rf=obs.rf, source=obs.source,
                       polrep=polrep, pol_prim=pol_prim, pulse=pulse,
                       mjd=obs.mjd, time=obs.tstart)

    return outim


def make_empty(npix, fov, ra, dec, rf=RF_DEFAULT,source=SOURCE_DEFAULT,
               polrep='stokes', pol_prim=None,pulse=PULSE_DEFAULT,
               mjd=MJD_DEFAULT, time=0.):

    """Make an empty square image.

       Args:
           npix (int): the pixel size of each axis
           fov (float): the field of view of each axis in radians
           ra (float): The source Right Ascension in fractional hours
           dec (float): The source declination in fractional degrees
           rf (float): The image frequency in Hz

           source (str): The source name
           polrep (str): polarization representation, either 'stokes' or 'circ'
           pol_prim (str): The default image: I,Q,U or V for Stokes, RR,LL,LR,RL for Circular
           pulse (function): The function convolved with the pixel values for continuous image.

           mjd (int): The integer MJD of the image
           time (float): The observing time of the image (UTC hours)

       Returns:
           (Image): an image object
    """

    pdim = fov/float(npix)
    npix = int(npix)
    imarr = np.zeros((npix,npix))
    outim = Image(imarr, pdim, ra, dec,
                  polrep=polrep, pol_prim=pol_prim,
                  rf=rf, source=source, mjd=mjd, time=time, pulse=pulse)
    return outim


def load_image(image, display=False, aipscc=False):

    """Read in an image from a text, .fits, or ehtim.image.Image object

       Args:
            image (str/Image): path to input file
            display (boolean): determine whether to display the image default
            aipscc (boolean): if True, then AIPS CC table will be loaded instead
                              of the original brightness distribution.
       Returns:
            (Image):    loaded image object
            (boolean):  False if the image cannot be read
    """

    if type(image) == type("str"):
      if image.endswith('.fits'):
        im = ehtim.io.load.load_im_fits(image, aipscc=aipscc)
      elif image.endswith('.txt'):
        im = ehtim.io.load.load_im_txt(image)
      else:
        print("Image format is not recognized. Was expecting .fits, .txt, or Image. Got <.{0}>. Returning False.".format(image.split('.')[-1]))
        return False


    elif isinstance(image, ehtim.image.Image):
      im = image

    else:
      print("Image format is not recognized. Was expecting .fits, .txt, or Image. Got {0}. Returning False.".format(type(image)))
      return False

    if display:
      im.display()

    return im

def load_txt(fname, polrep='stokes', pol_prim=None, pulse=PULSE_DEFAULT, zero_pol=True):


    """Read in an image from a text file.

       Args:
            fname (str): path to input text file
            pulse (function): The function convolved with the pixel values for continuous image.
            polrep (str): polarization representation, either 'stokes' or 'circ'
            pol_prim (str): The default image: I,Q,U or V for Stokes, RR,LL,LR,RL for Circular
            zero_pol (bool): If True, loads any missing polarizations as zeros

       Returns:
            (Image): loaded image object
    """

    return ehtim.io.load.load_im_txt(fname, pulse=pulse, polrep=polrep, pol_prim=pol_prim, zero_pol=True)

def load_fits(fname, aipscc=False, pulse=PULSE_DEFAULT,
              polrep='stokes', pol_prim=None,  zero_pol=False):

    """Read in an image from a FITS file.

       Args:
           fname (str): path to input fits file
           aipscc (bool): if True, then AIPS CC table will be loaded
           pulse (function): The function convolved with the pixel values for continuous image.
           polrep (str): polarization representation, either 'stokes' or 'circ'
           pol_prim (str): The default image: I,Q,U or V for Stokes, RR,LL,LR,RL for Circular
           zero_pol (bool): If True, loads any missing polarizations as zeros

       Returns:
           (Image): loaded image object
    """

    return ehtim.io.load.load_im_fits(fname, aipscc=aipscc,pulse=pulse,
                                      polrep=polrep, pol_prim=pol_prim,  zero_pol=zero_pol)
