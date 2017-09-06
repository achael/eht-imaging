from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage.filters as filt
import scipy.interpolate

import ehtim.observing.obs_simulate as simobs
import ehtim.observing.pulses
import ehtim.io.save
import ehtim.io.load

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

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
           source (str): The astrophysical source name
           ra (float): The source Right Ascension in fractional hours
           dec (float): The source declination in fractional degrees
           rf (float): The image frequency in Hz
           imvec (array): The vector of stokes I values in Jy/pixel (len xdim*ydim)
           qvec (array): The vector of stokes Q values in Jy/pixel (len xdim*ydim)
           uvec (array): The vector of stokes U values in Jy/pixel (len xdim*ydim)
           vvec (array): The vector of stokes V values in Jy/pixel (len xdim*ydim)
    """

    def __init__(self, image, psize, ra, dec, rf=RF_DEFAULT, pulse=PULSE_DEFAULT, source=SOURCE_DEFAULT, mjd=MJD_DEFAULT):
        """A polarimetric image (in units of Jy/pixel).

           Args:
               image (numpy.array): The 2D Stokes I values in Jy/pixel array
               psize (float): The pixel dimension in radians
               ra (float): The source Right Ascension in fractional hours
               dec (float): The source declination in fractional degrees
               rf (float): The image frequency in Hz
               pulse (function): The function convolved with the pixel values for continuous image.
               source (str): The astrophysical source name
               mjd (int): The integer MJD of the image

           Returns:
               image (Image): the Image object    
        """

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
        """Add Stokes Q and U image.
           Args:
               qimage (numpy.array): The 2D Stokes Q values in Jy/pixel array
               uimage (numpy.array): The 2D Stokes U values in Jy/pixel array
        """

        if len(qimage.shape) != len(uimage.shape):
            raise Exception("image must be a 2D numpy array")
        if qimage.shape != uimage.shape != (self.ydim, self.xdim):
            raise Exception("Q & U image shapes incompatible with I image!")
        self.qvec = qimage.flatten()
        self.uvec = uimage.flatten()

    def add_v(self, vimage):
        """Add Stokes V image.
           Args:
               vimage (numpy.array): The 2D Stokes Q values in Jy/pixel array
        """
        if vimage.shape != (self.ydim, self.xdim):
            raise Exception("V image shape incompatible with I image!")
        self.vvec = vimage.flatten()

    def add_pol(self, qimage, uimage, vimage):
        """Add all 3 Stokes Q, U, and V images.
           Args:
               qimage (numpy.array): The 2D Stokes Q values in Jy/pixel array
               uimage (numpy.array): The 2D Stokes U values in Jy/pixel array
               vimage (numpy.array): The 2D Stokes U values in Jy/pixel array
        """
        self.add_qu(qimage, uimage)
        self.add_v(vimage)

    def copy(self):
        """Return a copy of the image object.
           Returns: 
               newim (Image): copy of the Image.
        """
        newim = Image(self.imvec.reshape(self.ydim,self.xdim), self.psize, self.ra, self.dec, rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)
        if len(self.qvec):
            newim.add_qu(self.qvec.reshape(self.ydim,self.xdim), self.uvec.reshape(self.ydim,self.xdim))
        return newim

    def sourcevec(self):
        """Return the source position vector in geocentric coordinates at 0h GMST.
           Returns:
                vec (numpy.array): normal vector pointing to source in geocentric coordinates (m)
        """

        return np.array([np.cos(self.dec*DEGREE), 0, np.sin(self.dec*DEGREE)])

    def imarr(self, stokes="I"):
        """Return the 2D image array of a given Stokes parameter.
           Args: 
               stokes (str): "I","Q","U","V" for a given Stokes parameter
           Returns:
               imarr (numpy.array): 2D image array of dimension (ydim, xdim)
        """

        imarr = np.array([])
        if stokes=="I": imarr=im.imvec.reshape(im.ydim, im.xdim)
        elif stokes=="Q" and len(im.qvec): imarr=im.qvec.reshape(im.ydim, im.xdim)
        elif stokes=="U" and len(im.uvec): imarr=im.uvec.reshape(im.ydim, im.xdim)
        elif stokes=="V" and len(im.vvec): imarr=im.vvec.reshape(im.ydim, im.xdim)
        return imarr

    def fovx(self):
        """Return the image fov in x direction in radians.
           Returns: 
                fovx (float) : image fov in x direction (radian)
        """

        return self.psize * self.xdim

    def fovy(self):
        """Returns the image fov in y direction in radians.
           Returns: 
                fovy (float) : image fov in y direction (radian)
        """
        return self.psize * self.ydim

    def total_flux(self):
        """Return the total flux of the Stokes I image in Jy.
           Returns: 
                flux (float) : image total flux (Jy)
        """
        return np.sum(self.imvec)

    def flip_chi(self):
        """Flip between the different conventions for measuring the EVPA (E of N vs N of E).
        """
        self.qvec = - self.qvec
        return

    def observe_same_nonoise(self, obs, sgrscat=False, ft="direct", fft_pad_factor=1):
        """Observe the image on the same baselines as an existing observation object without adding noise.

           Args:
               obs (Obsdata): the existing observation with  baselines where the image FT will be sampled
               ft (str): if "fast", use FFT to produce visibilities. Else "direct" for DTFT
               fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in FFT
               sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel

           Returns:
               obs (Obsdata): an observation object with no noise
        """

        data = simobs.observe_image_nonoise(self, obs, sgrscat=sgrscat, ft=ft, fft_pad_factor=fft_pad_factor)

        obs_no_noise = ehtim.obsdata.Obsdata(self.ra, self.dec, obs.rf, obs.bw, data,
                                             obs.tarr, source=self.source, mjd=obs.mjd)
        return obs_no_noise

    def observe_same(self, obsin, ft='direct', fft_pad_factor=1,
                           sgrscat=False, add_th_noise=True,
                           opacitycal=True, ampcal=True, phasecal=True, frcal=True,dcal=True,
                           jones=False, inv_jones=False,
                           tau=TAUDEF, taup=GAINPDEF, gain_offset=GAINPDEF, gainp=GAINPDEF, 
                           dtermp=DTERMPDEF, dtermp_resid=DTERMPDEF_RESID):

        """Observe the image on the same baselines as an existing observation object and add noise.

           Args:
               obsin (Obsdata): the existing observation with  baselines where the image FT will be sampled
               ft (str): if "fast", use FFT to produce visibilities. Else "direct" for DTFT
               fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in FFT
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
               gainp (float): the fractional std. dev. of the random error on the gains
               taup (float): the fractional std. dev. of the random error on the opacities
               dtermp (float): the fractional std. dev. of the random error on the D-terms
               dtermp_resid (float): the fractional std. dev. of the random error on the D-terms

           Returns:
               obs (Obsdata): an observation object

        """

        obs = self.observe_same_nonoise(obsin, sgrscat=sgrscat, ft=ft, fft_pad_factor=fft_pad_factor)

        # Jones Matrix Corruption & Calibration
        if jones:
            print("Applying Jones Matrices to data . . . ")
            obsdata = simobs.add_jones_and_noise(obs, add_th_noise=add_th_noise,
                                                 opacitycal=opacitycal, ampcal=ampcal,
                                                 phasecal=phasecal, dcal=dcal, frcal=frcal,
                                                 gainp=gainp, taup=taup, gain_offset=gain_offset,
                                                 dtermp=dtermp,dtermp_resid=dtermp_resid)

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
        # There is an asymmetry here - in the old way, we don't offer the ability to *not* unscale estimated noise.
        else:
            print("Adding gain + phase errors to data and applying a priori calibration . . . ")
            obsdata = simobs.add_noise(obs, add_th_noise=add_th_noise,
                                       ampcal=ampcal, phasecal=phasecal, opacitycal=opacitycal,
                                       gainp=gainp, taup=taup, gain_offset=gain_offset)

            obs =  ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, obsdata,
                                             obs.tarr, source=obs.source, mjd=obs.mjd,
                                             ampcal=ampcal, phasecal=phasecal,
                                             opacitycal=True, dcal=True, frcal=True)
                                             #these are always set to True after inverse jones cal
        return obs

    def observe(self, array, tint, tadv, tstart, tstop, bw, mjd=None, timetype='UTC',
                      elevmin=ELEV_LOW, elevmax=ELEV_HIGH,
                      ft='direct', fft_pad_factor=1, sgrscat=False, add_th_noise=True,
                      opacitycal=True, ampcal=True, phasecal=True, frcal=True, dcal=True,
                      jones=False, inv_jones=False,
                      tau=TAUDEF, taup=GAINPDEF, gainp=GAINPDEF, gain_offset=GAINPDEF, 
                      dtermp=DTERMPDEF, dtermp_resid=DTERMPDEF_RESID):

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
               ft (str): if "fast", use FFT to produce visibilities. Else "direct" for DTFT
               fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in the FFT
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
               gainp (float): the fractional std. dev. of the random error on the gains
               taup (float): the fractional std. dev. of the random error on the opacities
               dtermp (float): the fractional std. dev. of the random error on the D-terms
               dtermp_resid (float): the fractional std. dev. of the random error on the D-terms

           Returns:
               obs (Obsdata): an observation object

        """


        # Generate empty observation
        print("Generating empty observation file . . . ")
        if mjd == None:
            mjd = self.mjd

        obs = array.obsdata(self.ra, self.dec, self.rf, bw, tint, tadv, tstart, tstop, mjd=mjd,
                            tau=tau, timetype=timetype, elevmin=elevmin, elevmax=elevmax)

        # Observe on the same baselines as the empty observation and add noise
        obs = self.observe_same(obs, ft=ft, fft_pad_factor=fft_pad_factor, sgrscat=sgrscat, add_th_noise=add_th_noise,
                                     opacitycal=opacitycal,ampcal=ampcal,phasecal=phasecal,dcal=dcal,frcal=frcal,
                                     gainp=gainp,gain_offset=gain_offset,dtermp=dtermp,taup=taup,dtermp_resid=dtermp_resid,
                                     jones=jones, inv_jones=inv_jones)

        return obs

    def observe_vex(self, vex, source,
                      ft='direct', fft_pad_factor=1, sgrscat=False, add_th_noise=True,
                      opacitycal=True, ampcal=True, phasecal=True, frcal=True, dcal=True,
                      jones=False, inv_jones=False,
                      tau=TAUDEF, gainp=GAINPDEF, taup=GAINPDEF, gain_offset=GAINPDEF, 
                      dtermp=DTERMPDEF, dtermp_resid=DTERMPDEF_RESID):

        """Generate baselines from a vex file and observes the image.

           Args:
               vex (Vex): an vex object containing sites and scan information
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
               gainp (float): the fractional std. dev. of the random error on the gains
               taup (float): the fractional std. dev. of the random error on the opacities
               dtermp (float): the fractional std. dev. of the random error on the D-terms
               dtermp_resid (float): the fractional std. dev. of the random error on the D-terms


           Returns:
               obs (Obsdata): an observation object

        """

        obs_List=[]
        for i_scan in range(len(vex.sched)):
            if vex.sched[i_scan]['source'] != source:
                continue
            subarray = vex.array.make_subarray([vex.sched[i_scan]['scan'][key]['site'] for key in list(vex.sched[i_scan]['scan'].keys())])

            obs = self.observe(subarray, vex.sched[i_scan]['scan'][0]['scan_sec'], 2.0*vex.sched[i_scan]['scan'][0]['scan_sec'],
                                       vex.sched[i_scan]['start_hr'], vex.sched[i_scan]['start_hr'] + vex.sched[i_scan]['scan'][0]['scan_sec']/3600.0,
                                       vex.bw_hz, mjd=vex.sched[i_scan]['mjd_floor'],
                                       elevmin=.01, elevmax=89.99,
                                       ft=ft, fft_pad_factor=fft_pad_factor, sgrscat=sgrscat, add_th_noise=add_th_noise,
                                       opacitycal=opacitycal,ampcal=ampcal,phasecal=phasecal,dcal=dcal,frcal=frcal,
                                       taup=taup, gainp=gainp,gain_offset=gain_offset,dtermp=dtermps,dtermp_resid=termp_resid,
                                       jones=jones, inv_jones=inv_jones)


            obs_List.append(obs)

        return eht.obsdata.merge_obs(obs_List)

    def regrid_image(self, targetfov, npix):
        """Resample the image to new (square) dimensions
           Args:
                targetfov  (float): new field of view (radian) 
                npix  (int): new pixel dimension 
           Returns:
                im_regrid (Image): resampled image 
        """
        
        fov_x = self.xdim * self.psize
        fov_y = self.ydim * iselfm.psize
        
        x = np.linspace(-fov_x/2, fov_x/2, self.xdim)
        y = np.linspace(-fov_y/2, fov_y/2, self.ydim)

        xtarget = np.linspace(-targetfov/2, targetfov/2, npix)
        ytarget = np.linspace(-targetfov/2, targetfov/2, npix)

        interpfunc = scipy.interpolate.RectBivariateSpline( y, x, np.reshape(self.imvec, (self.ydim, self.xdim) ) )
        tmpimg = interpfunc(ytarget, xtarget)
        tmpimg[np.abs(xtarget)>fov_x/2.,:] = 0.0
        tmpimg[:,np.abs(ytarget)>fov_y/2.] = 0.0
        tmpimg = tmpimg * (targetfov/npix)**2 /self.psize**2

        outim = Image(tmpimg, targetfov/npix, self.ra, self.dec, rf=self.rf, source=self.source, mjd=self.mjd, pulse=self.pulse)

        if len(im.qvec):
            interpfunc = scipy.interpolate.RectBivariateSpline( y, x, np.reshape(self.qvec, (self.ydim, self.xdim) ) )
            tmpimg = interpfunc(ytarget, xtarget)
            tmpimg[np.abs(xtarget)>fov_x/2.,:] = 0.0
            tmpimg[:,np.abs(ytarget)>fov_y/2.] = 0.0
            outq = tmpimg * (targetfov/npix)**2 / self.psize**2

            interpfunc = scipy.interpolate.RectBivariateSpline( y, x, np.reshape(self.uvec, (self.ydim, self.xdim) ) )
            tmpimg = interpfunc(ytarget, xtarget)
            tmpimg[np.abs(xtarget)>fov_x/2.,:] = 0.0
            tmpimg[:,np.abs(ytarget)>fov_y/2.] = 0.0
            outu = tmpimg * (targetfov/npix)**2 / self.psize**2

            outim.add_qu(outq, outu)

        if len(im.vvec):
            interpfunc = scipy.interpolate.RectBivariateSpline( y, x, np.reshape(self.vvec, (self.ydim, self.xdim) ) )
            tmpimg = interpfunc(ytarget, xtarget)
            tmpimg[np.abs(xtarget)>fov_x/2.,:] = 0.0
            tmpimg[:,np.abs(ytarget)>fov_y/2.] = 0.0
            outv = tmpimg * (targetfov/npix)**2 / self.psize**2

            outim.add_v(outv)

        return outim


    def compare_images(self, im2, psize=None, target_fov=None, beamparams = [1., 1., 1.], blur_frac = 0.0, metric = ['nxcorr', 'nrmse', 'rssd'], blursmall=False):
      """Compare to another image by computing normalized cross correlation, normalized root mean squared error, or square root of the sum of squared differences
         Args:
             psize (float): pixel size of comparison image (rad). If None it is the smallest of the input image pizel sizes
             target_fov (float): fov of the comparison image (rad). If None it is twice the largest fov of the input images

             beamparams (list): the nominal Gaussian beam parameters [fovx, fovy, position angle]
             blur_frac (float): fractional beam to blur each image to before comparison

             metric (list) : a list of fidelity metrics from ['nxcorr','nrmse','rssd']
             blursmall (bool) : True to blur the unpadded image rather than the large image.

         Returns:
             out (list): [errormetric, im1_pad, im2_shift] of computed error metric and shifted/resized comparison images

      """
        
        im1 = self.copy()
        if target_fov==None:
            max_fov = np.max([im1.xdim * im1.psize, im1.ydim * im1.psize, im2.xdim * im2.psize, im2.ydim * im2.psize])
            target_fov = 2*max_fov

        if psize==None:
            psize = np.min([im1.psize, im2.psize])

        npix = int( target_fov / psize )

        if ( (blur_frac > 0.0) * (blursmall==True) ):
            im1 = im1.blur_gauss(beamparams, blur_frac)
            im2 = im2.blur_gauss(beamparams, blur_frac)

        im1_pad = im1.regrid_image(im1, target_fov, npix)
        im2_pad = im2.regrid_image(im2, target_fov, npix)

        if ((blur_frac > 0.0) * (blursmall==False)):
            im1_pad = im1_pad.blur_gauss(beamparams, blur_frac)
            im2_pad = im2_pad.blur_gauss(beamparams, blur_frac)

        im1_norm = ( im1_pad.imvec.reshape(im1_pad.ydim, im1_pad.xdim) - np.mean(im1_pad.imvec) ) / np.std(im1_pad.imvec)
        im2_norm = ( im2_pad.imvec.reshape(im2_pad.ydim, im2_pad.xdim) - np.mean(im2_pad.imvec) ) / np.std(im2_pad.imvec)

        fft_im1 = np.fft.fft2( im1_norm )
        fft_im2 = np.fft.fft2( im2_norm )

        xcorr =  np.real( np.fft.ifft2( fft_im1 * np.conj(fft_im2) ) )
        idx = np.unravel_index(xcorr.argmax(), xcorr.shape)

        im2_shift = np.roll(im2_pad.imvec.reshape(im2_pad.ydim, im2_pad.xdim),   idx[0], axis=0)
        im2_shift = np.roll(im2_shift, idx[1], axis=1)
        im2_shift = eh.image.Image( im2_shift, im2_pad.psize, im2_pad.ra, im2_pad.dec, rf=im2_pad.rf, source=im2_pad.source, mjd=im2_pad.mjd, pulse=im2_pad.pulse)

        error = []
        if 'nxcorr' in metric:
            error.append( np.max(xcorr.reshape(-1)) / (im1_pad.xdim * im1_pad.ydim) )
        if 'nrmse' in metric:
            error.append( np.sqrt( np.sum( ( (im1_pad.imvec - im2_shift.imvec)**2 * im1_pad.psize**2  ) ) / np.sum( (im1_pad.imvec )**2 * im1_pad.psize**2 ) ) )
        if 'rssd' in metric:
            error.append( np.sqrt( np.sum(  (im1_pad.imvec - im2_shift.imvec)**2 ) * im1_pad.psize**2 ) )

        return (error, im1_pad, im2_shift)

#    def resample_square(self, xdim_new, ker_size=5):
#        """Resample the image to new (square) dimensions
#           Args:
#                xdim_new  (int): new pixel dimension 
#                ker_size  (int): kernel size for resampling
#           Returns:
#                im_resampled (Image): resampled image 
#        """
#
#        im = self
#        if im.xdim != im.ydim:
#            raise Exception("Image must be square!")
#        if im.pulse == ehtim.observing.pulses.deltaPulse2D:
#            raise Exception("This function only works on continuously parametrized images: does not work with delta pulses!")
#
#        ydim_new = xdim_new
#        fov = im.xdim * im.psize
#        psize_new = fov / xdim_new
#        ij = np.array([[[i*im.psize + (im.psize*im.xdim)/2.0 - im.psize/2.0, j*im.psize + (im.psize*im.ydim)/2.0 - im.psize/2.0]
#                        for i in np.arange(0, -im.xdim, -1)]
#                        for j in np.arange(0, -im.ydim, -1)]).reshape((im.xdim*im.ydim, 2))
#        def im_new(x,y):
#            mask = (((x - ker_size*im.psize/2.0) < ij[:,0]) * (ij[:,0] < (x + ker_size*im.psize/2.0)) * ((y-ker_size*im.psize/2.0) < ij[:,1]) * (ij[:,1] < (y+ker_size*im.psize/2.0))).flatten()
#            return np.sum([im.imvec[n] * im.pulse(x-ij[n,0], y-ij[n,1], im.psize, dom="I") for n in np.arange(len(im.imvec))[mask]])
#
#        out = np.array([[im_new(x*psize_new + (psize_new*xdim_new)/2.0 - psize_new/2.0, y*psize_new + (psize_new*ydim_new)/2.0 - psize_new/2.0)
#                          for x in np.arange(0, -xdim_new, -1)]
#                          for y in np.arange(0, -ydim_new, -1)] )
#
#
#        # Normalize
#        scaling = np.sum(im.imvec) / np.sum(out)
#        out *= scaling
#        outim = Image(out, psize_new, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse)
#
#        # Q and U images
#        if len(im.qvec):
#            def im_new_q(x,y):
#                mask = (((x - ker_size*im.psize/2.0) < ij[:,0]) * (ij[:,0] < (x + ker_size*im.psize/2.0)) *
#                        ((y - ker_size*im.psize/2.0) < ij[:,1]) * (ij[:,1] < (y + ker_size*im.psize/2.0))).flatten()
#                return np.sum([im.qvec[n] * im.pulse(x-ij[n,0], y-ij[n,1], im.psize, dom="I") for n in np.arange(len(im.imvec))[mask]])
#            def im_new_u(x,y):
#                mask = (((x - ker_size*im.psize/2.0) < ij[:,0]) * (ij[:,0] < (x + ker_size*im.psize/2.0)) *
#                        ((y-ker_size*im.psize/2.0) < ij[:,1]) * (ij[:,1] < (y+ker_size*im.psize/2.0))).flatten()
#                return np.sum([im.uvec[n] * im.pulse(x-ij[n,0], y-ij[n,1], im.psize, dom="I") for n in np.arange(len(im.imvec))[mask]])
#            outq = np.array([[im_new_q(x*psize_new + (psize_new*xdim_new)/2.0 - psize_new/2.0, y*psize_new + (psize_new*ydim_new)/2.0 - psize_new/2.0)
#                          for x in np.arange(0, -xdim_new, -1)]
#                          for y in np.arange(0, -ydim_new, -1)] )
#            outu = np.array([[im_new_u(x*psize_new + (psize_new*xdim_new)/2.0 - psize_new/2.0, y*psize_new + (psize_new*ydim_new)/2.0 - psize_new/2.0)
#                          for x in np.arange(0, -xdim_new, -1)]
#                          for y in np.arange(0, -ydim_new, -1)] )
#            outq *= scaling
#            outu *= scaling
#            outim.add_qu(outq, outu)
#        if len(im.vvec):
#            def im_new_v(x,y):
#                mask = (((x - ker_size*im.psize/2.0) < ij[:,0]) * (ij[:,0] < (x + ker_size*im.psize/2.0)) *
#                        ((y-ker_size*im.psize/2.0) < ij[:,1]) * (ij[:,1] < (y+ker_size*im.psize/2.0))).flatten()
#                return np.sum([im.vvec[n] * im.pulse(x-ij[n,0], y-ij[n,1], im.psize, dom="I") for n in np.arange(len(im.imvec))[mask]])
#            outv = np.array([[im_new_v(x*psize_new + (psize_new*xdim_new)/2.0 - psize_new/2.0, y*psize_new + (psize_new*ydim_new)/2.0 - psize_new/2.0)
#                          for x in np.arange(0, -xdim_new, -1)]
#                          for y in np.arange(0, -ydim_new, -1)] )
#            outv *= scaling
#            outim.add_v(outv)
#        return outim
#
#    def im_pad(self, fovx, fovy):
#        """Pad an image to new fov_x by fov_y in radian.
#           Args:
#                fovx  (float): new fov in x dimension (rad) 
#                fovy  (float): new fov in y dimension (rad) 
#           Returns:
#                im_pad (Image): padded image 
#        """
#        im = self
#        fovoldx=im.psize*im.xdim
#        fovoldy=im.psize*im.ydim
#        padx=int(0.5*(fovx-fovoldx)/im.psize)
#        pady=int(0.5*(fovy-fovoldy)/im.psize)
#        imarr=im.imvec.reshape(im.ydim, im.xdim)
#        imarr=np.pad(imarr,((padx,padx),(pady,pady)),'constant')
#        outim=Image(imarr, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse)#
#        if len(im.qvec):
#            qarr=im.qvec.reshape(im.ydim,im.xdim)
#            qarr=np.pad(qarr,((padx,padx),(pady,pady)),'constant')
#            uarr=im.uvec.reshape(im.ydim,im.xdim)
#            uarr=np.pad(uarr,((padx,padx),(pady,pady)),'constant')
#            outim.add_qu(qarr,uarr)
#        if len(im.vvec):
#            varr=im.vvec.reshape(im.ydim,im.xdim)
#            varr=np.pad(qarr,((padx,padx),(pady,pady)),'constant')
#            outim.add_v(varr)
#        return outim


    def add_flat(self, flux):
        """Add a flat background flux to the Stokes I image.
           Args:
                flux  (float): total flux to add to image
           Returns:
                im_flat (Image): output image 
        """

        im = self
        imout = (im.imvec + (flux/float(len(im.imvec))) * np.ones(len(im.imvec))).reshape(im.ydim,im.xdim)
        out = Image(imout, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse)
        return out

    def add_tophat(self, flux, radius):
        """Add centered tophat flux to the Stokes I image inside a given radius.
           Args:
                flux  (float): total flux to add to image
                radius  (float): radius of top hat flux in radians
           Returns:
                im_tophat (Image): output image 
        """

        im = self
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

    def add_gauss(self, flux, beamparams):
        """Add a gaussian to an image.

           Args:
               flux (float): the total flux contained in the Gaussian in Jy
               beamparams (list): the gaussian parameters, [fwhm_maj, fwhm_min, theta, x, y], all in radians
           Returns:
                im_gauss (Image): output image 
        """

        im = self
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

    def add_crescent(self, flux, Rp, Rn, a, b, x=0, y=0):
        """Add a crescent to an image; see Kamruddin & Dexter (2013).

           Args:
               flux (float): the total flux contained in the crescent in Jy
               Rp (float): the larger radius in radians
               Rn (float): the smaller radius in radians
               a (float): the relative x offset of smaller disk in radians
               b (float): the relative y offset of smaller disk in radians
               x (float): the center x coordinate of the larger disk in radians
               y (float): the center y coordinate of the larger disk in radians
           Returns:
               im_cresc (Image): output image 
        """

        im = self
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

    def add_const_m(self, mag, angle):
        """Add a constant fractional linear polarization to image

           Args:
               mag (float): constant polarization fraction to add to the image
               angle (float): constant EVPA
           Returns:
                im_out (Image): output image 
        """
        im = self
        if not (0 < mag < 1):
            raise Exception("fractional polarization magnitude must be beween 0 and 1!")

        imi = im.imvec.reshape(im.ydim,im.xdim)
        imq = qimage(im.imvec, mag * np.ones(len(im.imvec)), angle*np.ones(len(im.imvec))).reshape(im.ydim,im.xdim)
        imu = uimage(im.imvec, mag * np.ones(len(im.imvec)), angle*np.ones(len(im.imvec))).reshape(im.ydim,im.xdim)
        out = Image(imi, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse)
        out.add_qu(imq, imu)
        return out

    def blur_gauss(self, beamparams, frac=1., frac_pol=0):
        """Blur image with a Gaussian beam defined by beamparams [fwhm_max, fwhm_min, theta] in radians.
           Args:
               beamparams (list): the gaussian parameters, [fwhm_maj, fwhm_min, theta, x, y], all in radians
               frac (float): fractional beam size for Stokes I  blurring
               frac_pol (float): fractional beam size for Stokes Q,U,V  blurring
           Returns:
               im_out (Image): output image 
        """

        image = self
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

    def blur_circ(self, fwhm_i, fwhm_pol=0):
        """Apply a circular gaussian filter to the image, with FWHM in radians.
           Args:
               fwhm_i (float): circular beam size for Stokes I  blurring in  radian
               fwhm_pol (float): circular beam size for Stokes Q,U,V  blurring in  radian
           Returns:
               im_out (Image): output image 
        """

        image = self
        # Blur Stokes I
        sigma = fwhm_i / (2. * np.sqrt(2. * np.log(2.)))
        sigmap = sigma / image.psize
        im = filt.gaussian_filter(image.imvec.reshape(image.ydim, image.xdim), (sigmap, sigmap))
        out = Image(im, image.psize, image.ra, image.dec, rf=image.rf, source=image.source, mjd=image.mjd)

        # Blur Stokes Q and U
        if len(image.qvec) and fwhm_pol:
            sigma = fwhm_pol / (2. * np.sqrt(2. * np.log(2.)))
            sigmap = sigma / image.psize
            imq = filt.gaussian_filter(image.qvec.reshape(image.ydim,image.xdim), (sigmap, sigmap))
            imu = filt.gaussian_filter(image.uvec.reshape(image.ydim,image.xdim), (sigmap, sigmap))
            out.add_qu(imq, imu)

        return out

    def threshold(self, frac_i=1.e-5):
        """Apply a hard threshold to the image.
           Args:
               frac_i (float): Stokes I floor as a fraction of maximum stokes I point
           Returns:
               im_out (Image): output image 
        """

        image=self
        imvec = np.copy(image.imvec)

        thresh = frac_i*np.abs(np.max(imvec))
        lowval = thresh
        flux = np.sum(imvec)

        for j in range(len(imvec)):
            if imvec[j] < thresh:
                imvec[j]=lowval

        imvec = flux*imvec/np.sum(imvec)
        out = Image(imvec.reshape(image.ydim,image.xdim), image.psize,
                       image.ra, image.dec, rf=image.rf, source=image.source, mjd=image.mjd)
        return out



    def display(self, cfun='afmhot',scale='lin', interp='gaussian', gamma=0.5, dynamic_range=1.e3, plotp=False, nvec=20, pcut=0.01,export_pdf="", show=True):
        """Display the image.

           Args:
               cfun (str): matplotlib.pyplot color function
               scale (str): image scaling in ['log','gamma','lin']
               interp (str): image interpolation 'gauss' or 'lin'

               gamma (float): index for gamma scaling
               dynamic_range (float): dynamic range for log and gamma scaling

               plotp (bool): True to plot linear polarimetic image
               nvec (int): number of polarimetric vectors to plot
               pcut (float): minimum stokes P value for displaying polarimetric vectors as fraction of maximum Stokes I pixel
               export_pdf (str): path to exported PDF with plot
               show (bool): Display the plot if true

           Returns:
               fig (matplotlib.figure.Figure): figure object with image

        """

        if (interp in ['gauss', 'gaussian', 'Gaussian', 'Gauss']):
            interp = 'gaussian'
        else:
            interp = 'linear'

        f = plt.figure()
        plt.clf()

        imarr = (self.imvec).reshape(self.ydim, self.xdim)
        unit = 'Jy/pixel'
        if scale=='log':
            imarr = np.log(imarr + np.max(imarr)/dynamic_range)
            unit = 'log(Jy/pixel)'

        if scale=='gamma':
            imarr = (imarr + np.max(imarr)/dynamic_range)**(gamma)
            unit = '(Jy/pixel)^gamma'

        if len(self.qvec) and plotp:
            thin = self.xdim//nvec
            mask = (self.imvec).reshape(self.ydim, self.xdim) > pcut * np.max(self.imvec)
            mask2 = mask[::thin, ::thin]
            x = (np.array([[i for i in range(self.xdim)] for j in range(self.ydim)])[::thin, ::thin])[mask2]
            y = (np.array([[j for i in range(self.xdim)] for j in range(self.ydim)])[::thin, ::thin])[mask2]
            a = (-np.sin(np.angle(self.qvec+1j*self.uvec)/2).reshape(self.ydim, self.xdim)[::thin, ::thin])[mask2]
            b = ( np.cos(np.angle(self.qvec+1j*self.uvec)/2).reshape(self.ydim, self.xdim)[::thin, ::thin])[mask2]

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

        if export_pdf != "":
            f.savefig(export_pdf, bbox_inches='tight')

        return f

    def save_txt(self, fname):
        """Save image data to text file.
           Args:
                fname (str): path to output text file
        """

        ehtim.io.save.save_im_txt(self, fname)
        return

    def save_fits(self, fname):
        """Save image data to a fits file.
           Args:
                fname (str): path to output fits file
        """
        ehtim.io.save.save_im_fits(self, fname)
        return

###########################################################################################################################################
#Image creation functions
###########################################################################################################################################
def make_square(obs, npix, fov, pulse=PULSE_DEFAULT):
    """Make an empty square image.

       Args:
           obs (Obsdata): an obsdata object with the image metadata
           npix (int): the pixel size of each axis
           fov (float): the field of view of each axis in radians
           pulse (function): the function convolved with the pixel values for continuous image
       Returns:
           empty_image (Image): an image object
    """

    pdim = fov/float(npix)
    npix = int(npix)
    im = np.zeros((npix,npix))
    return Image(im, pdim, obs.ra, obs.dec, rf=obs.rf, source=obs.source, mjd=obs.mjd, pulse=pulse)

def load_txt(fname):
    """Read in an image from a text file.
    
       Args:
            fname (str): path to input text file
       Returns: 
            image (Image): loaded image object
    """

    return ehtim.io.load.load_im_txt(fname)

def load_fits(fname):
    """Read in an image from a FITS file.

       Args:
            fname (str): path to input fits file
       Returns: 
            image (Image): loaded image object
    """

    return ehtim.io.load.load_im_fits(fname)
