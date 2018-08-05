# obs_simulate.py
# functions to simulate interferometric observations
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

import astropy.time as at
import time as ttime
import scipy.ndimage as nd
import numpy as np
import datetime
try:
    import ephem
except ImportError:
    print("Warning: ephem not installed: cannot simulate space VLBI")
import astropy.coordinates as coords
import copy
try:
    from pynfft.nfft import NFFT
except ImportError:
    print("Warning: No NFFT installed!")

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

##################################################################################################
# Generate U-V Points
##################################################################################################

def make_uvpoints(array, ra, dec, rf, bw, tint, tadv, tstart, tstop,
                  mjd=MJD_DEFAULT,tau=TAUDEF, elevmin=ELEV_LOW, elevmax=ELEV_HIGH,
                  timetype='UTC', fix_theta_GMST = False):

    """Generate u,v points and baseline sigmas for a given array.

       Args:
           array (Array): the array object
           ra (float): The source Right Ascension in fractional hours
           dec (float): The source declination in fractional degrees
           rf (float): The observation frequency in Hz
           bw (float): The observation bandwidth in Hz
           tint (float): the scan integration time in seconds
           tadv (float): the uniform cadence between scans in seconds
           tstart (float): the start time of the observation in hours
           tstop (float): the end time of the observation in hours
           mjd (int): the mjd of the observation, if different from the image mjd
           timetype (str): how to interpret tstart and tstop; either 'GMST' or 'UTC'
           elevmin (float): station minimum elevation in degrees
           elevmax (float): station maximum elevation in degrees
           tau (float): the base opacity at all sites, or a dict giving one opacity per site
           fix_theta_GMST (bool): if True, stops earth rotation to sample fixed u,v points through time

       Returns:
           (Obsdata): an observation object with all visibilities zeroed
    """

    # Set up time start and steps
    tstep = tadv/3600.0
    if tstop < tstart:
        tstop = tstop + 24.0;

    # Observing times
    # TODO: Scale - utc or tt?
    times = np.arange(tstart, tstop, tstep)
    if timetype not in ['UTC', 'GMST']:
        print("Time Type Not Recognized! Assuming UTC!")
        timetype = 'UTC'

    # Generate uv points at all times
    outlist = []
    blpairs = []
    for i1 in range(len(array.tarr)):
        for i2 in range(len(array.tarr)):
            if (i1!=i2 and
                i1 < i2 and # This is the right condition for uvfits save order
                not ((i2, i1) in blpairs)): # This cuts out the conjugate baselines

                blpairs.append((i1,i2))

                #sites
                site1 = array.tarr[i1]['site']
                site2 = array.tarr[i2]['site']
                coord1 = ((array.tarr[i1]['x'], array.tarr[i1]['y'], array.tarr[i1]['z']))
                coord2 = ((array.tarr[i2]['x'], array.tarr[i2]['y'], array.tarr[i2]['z']))
                # Optical Depth
                if type(tau) == dict:
                    try:
                        tau1 = tau[i1]
                        tau2 = tau[i2]
                    except KeyError:
                        tau1 = tau2 = TAUDEF
                else:
                    tau1 = tau2 = tau
                #no optical depth for space sites
                if coord1 == (0.,0.,0.): tau1 = 0.
                if coord2 == (0.,0.,0.): tau2 = 0.

                # Noise on the correlations
                if np.any(obs.tarr['sefdr'] <= 0) or np.any(obs.tarr['sefdl'] <=0):
                    print("Warning!: in make_uvpoints, some SEFDs are <= 0!")

                sig_rr = blnoise(array.tarr[i1]['sefdr'], array.tarr[i2]['sefdr'], tint, bw)
                sig_ll = blnoise(array.tarr[i1]['sefdl'], array.tarr[i2]['sefdl'], tint, bw)
                sig_rl = blnoise(array.tarr[i1]['sefdr'], array.tarr[i2]['sefdl'], tint, bw)
                sig_lr = blnoise(array.tarr[i1]['sefdl'], array.tarr[i2]['sefdr'], tint, bw)
                sig_iv = 0.5*np.sqrt(sig_rr**2 + sig_ll**2)
                sig_qu = 0.5*np.sqrt(sig_rl**2 + sig_lr**2)


                (timesout,uout,vout) = compute_uv_coordinates(array, site1, site2, times, mjd,
                                                              ra, dec, rf, timetype=timetype,
                                                              elevmin=elevmin, elevmax=elevmax,
                                                              fix_theta_GMST=fix_theta_GMST)

                for k in range(len(timesout)):
                    outlist.append(np.array((
                              timesout[k],
                              tint, # Integration
                              site1, # Station 1
                              site2, # Station 2
                              tau1, # Station 1 zenith optical depth
                              tau2, # Station 1 zenith optical depth
                              uout[k], # u (lambda)
                              vout[k], # v (lambda)
                              0.0, # I Visibility (Jy)
                              0.0, # Q Visibility
                              0.0, # U Visibility
                              0.0, # V Visibilities
                              sig_iv, # I Sigma (Jy)
                              sig_qu, # Q Sigma
                              sig_qu, # U Sigma
                              sig_iv  # V Sigma
                            ), dtype=DTPOL
                            ))

    obsarr = np.array(outlist)

    if not len(obsarr):
        raise Exception("No mutual visibilities in the specified time range!")

    return obsarr

##################################################################################################
# Observe w/o noise
##################################################################################################

def sample_vis(im, uv, sgrscat=False, ttype="direct", fft_pad_factor=2):

    """Observe a image on given baselines with no noise.

       Args:
           im (Image): the image to be observed
           uv (ndarray): an array of u,v coordinates
           sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
           ttype (str): if "fast" or 'nfft', use FFT to produce visibilities. Else "direct" for DTFT
           fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in FFT

       Returns:
           (Obsdata): an observation object
    """

    #TODO -- all imports should go at the top
    #but circular imports cause headaches....
    from ehtim.obsdata import Obsdata

    uv = np.array(uv)
    if uv.shape[1] != 2:
        raise Exception("When given as a list of uv points, the obs should be a list of pairs of u-v coordinates!")

    umin = np.min(np.sqrt(uv[:,0]**2 + uv[:,1]**2))
    umax = np.max(np.sqrt(uv[:,0]**2 + uv[:,1]**2))

    if not im.psize < 1.0/(2.0*umax):
        print("    Warning!: longest baseline > 1/2 x maximum image spatial wavelength!")
    if not im.psize*np.sqrt(im.xdim*im.ydim) > 1.0/(0.5*umin):
        print("    Warning!: shortest baseline < 2 x minimum image spatial wavelength!")

    vis = np.zeros(len(uv))
    qvis = np.zeros(len(uv))
    uvis = np.zeros(len(uv))
    vvis = np.zeros(len(uv))

    # Get visibilities from straightforward FFT
    if ttype=="fast":

        # Pad image
        npad = fft_pad_factor * np.max((im.xdim, im.ydim))
        npad = power_of_two(npad)

        padvalx1 = padvalx2 = int(np.floor((npad - im.xdim)/2.0))
        if im.xdim % 2:
            padvalx2 += 1
        padvaly1 = padvaly2 = int(np.floor((npad - im.ydim)/2.0))
        if im.ydim % 2:
            padvaly2 += 1

        imarr = im.imvec.reshape(im.ydim, im.xdim)
        imarr = np.pad(imarr, ((padvalx1,padvalx2),(padvaly1,padvaly2)), 'constant', constant_values=0.0)
        npad = imarr.shape[0]
        if imarr.shape[0]!=imarr.shape[1]:
            raise Exception("FFT padding did not return a square image!")

        # Scaled uv points
        du = 1.0/(npad*im.psize)
        uv2 = np.hstack((uv[:,1].reshape(-1,1), uv[:,0].reshape(-1,1)))
        uv2 = (uv2/du + 0.5*npad).T

        # FFT for visibilities
        vis_im = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(imarr)))

        # Sample the visibilities
        # default is cubic spline interpolation
        visre = nd.map_coordinates(np.real(vis_im), uv2)
        visim = nd.map_coordinates(np.imag(vis_im), uv2)
        vis = visre + 1j*visim

        # Extra phase to match centroid convention
        phase = np.exp(-1j*np.pi*im.psize*((1+im.xdim%2)*uv[:,0] + (1+im.ydim%2)*uv[:,1]))
        vis = vis * phase

        # Multiply by the pulse function
        pulsefac = np.array([im.pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], im.psize, dom="F") for uvpt in uv])
        vis = vis * pulsefac

        # FFT of polarimetric quantities
        if len(im.qvec):
            qarr = im.qvec.reshape(im.ydim, im.xdim)
            qarr = np.pad(qarr, ((padvalx1,padvalx2),(padvaly1,padvaly2)), 'constant', constant_values=0.0)
            uarr = im.uvec.reshape(im.ydim, im.xdim)
            uarr = np.pad(uarr, ((padvalx1,padvalx2),(padvaly1,padvaly2)), 'constant', constant_values=0.0)

            qvis_im = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(qarr)))
            uvis_im = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(uarr)))

            qvisre = nd.map_coordinates(np.real(qvis_im), uv2)
            qvisim = nd.map_coordinates(np.imag(qvis_im), uv2)
            qvis = phase*(qvisre + 1j*qvisim)
            qvis = qvis*pulsefac

            uvisre = nd.map_coordinates(np.real(uvis_im), uv2)
            uvisim = nd.map_coordinates(np.imag(uvis_im), uv2)
            uvis = phase*(uvisre + 1j*uvisim)
            uvis = uvis*pulsefac

        if len(im.vvec):
            varr = im.vvec.reshape(im.ydim, im.xdim)
            varr = np.pad(varr, ((padvalx1,padvalx2),(padvaly1,padvaly2)), 'constant', constant_values=0.0)

            vvis_im = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(varr)))

            vvisre = nd.map_coordinates(np.real(vvis_im), uv2)
            vvisim = nd.map_coordinates(np.imag(vvis_im), uv2)
            vvis = phase*(vvisre + 1j*vvisim)
            vvis = vvis*pulsefac

    # Get visibilities from NFFT library
    elif ttype=="nfft":

        uvdim = len(uv)
        if (im.xdim%2 or im.ydim%2):
            raise Exception("NFFT doesn't work with odd image dimensions!")

        npad = fft_pad_factor * np.max((im.xdim, im.ydim))

        #TODO kernel size??
        nker = np.floor(np.min((im.xdim,im.ydim))/5)
        if (nker>50):
            nker = 50
        elif (im.xdim<50 or im.ydim<50):
            nker = np.min((im.xdim,im.ydim))/2

        #TODO y & x reversed?
        plan = NFFT([im.xdim,im.ydim],uvdim, m=nker, n=[npad,npad])

        #sampled points
        uvlist = uv*im.psize

        #precompute
        plan.x = uvlist
        plan.precompute()

        #phase and pulsefac
        phase = np.exp(-1j*np.pi*(uvlist[:,0] + uvlist[:,1]))
        pulsefac = np.array([im.pulse(2*np.pi*uvlist[i,0], 2*np.pi*uvlist[i,1], 1., dom="F") for i in range(uvdim)])

        #compute uniform --> nonuniform transform
        plan.f_hat = im.imvec.copy().reshape((im.ydim,im.xdim)).T
        plan.trafo()
        vis = plan.f.copy()*phase*pulsefac

        if len(im.qvec):
            plan.f_hat = im.qvec.copy().reshape((im.ydim,im.xdim)).T
            plan.trafo()
            qvis = plan.f.copy()*phase*pulsefac

            plan.f_hat = im.uvec.copy().reshape((im.ydim,im.xdim)).T
            plan.trafo()
            uvis = plan.f.copy()*phase*pulsefac

        if len(im.vvec):
            plan.f_hat = im.vvec.copy().reshape((im.ydim,im.xdim)).T
            plan.trafo()
            vvis = plan.f.copy()*phase*pulsefac

    # Get visibilities from DTFT
    else:
        mat = ftmatrix(im.psize, im.xdim, im.ydim, uv, pulse=im.pulse)
        vis = np.dot(mat, im.imvec)

        if len(im.qvec):
            qvis = np.dot(mat, im.qvec)
            uvis = np.dot(mat, im.uvec)
        if len(im.vvec):
            vvis = np.dot(mat, im.vvec)

    # Scatter the visibilities with the SgrA* kernel
    if sgrscat:
        print('Scattering Visibilities with Sgr A* kernel!')
        for i in range(len(vis)):
            ker = sgra_kernel_uv(im.rf, uv[i,0], uv[i,1])
            vis[i]  *= ker
            qvis[i] *= ker
            uvis[i] *= ker
            vvis[i] *= ker

    # Put the visibilities back in the obsdata array
    if len(im.qvec):
        obsdata = [vis, qvis, uvis, vvis]
    else:
        obsdata = [vis, None, None, None]

    return obsdata

#TODO make this more similar to sample_vis for an image
#TODO is it even possible given that we need time information?
def observe_movie_nonoise(mov, obs, sgrscat=False, ttype="direct", fft_pad_factor=2, repeat=False):

    """Observe a movie on the same baselines as an existing observation object with no noise.

       Args:
           mov (Movie): the movie to be observed
           obs (Obsdata): The empty observation object
           sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
           ttype (str): if "fast", or 'nfft', use FFT to produce visibilities. Else "direct" for DTFT
           fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in FFT
           repeat (bool): if True, repeat the movie to fill up the observation interval

       Returns:
           (Obsdata): an observation object
    """

    mjdstart = float(mov.mjd) + float(mov.start_hr/24.0)
    mjdend = mjdstart + (len(mov.frames)*mov.framedur)/86400.0

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
        n = int(np.floor((mjd - mjdstart) * 86400. / mov.framedur))

        if (n >= len(mov.frames)):
            if repeat: n = np.mod(n, len(mov.frames))
            else: raise Exception("Obs times outside of movie range of MJD %f - %f" % (mjdstart, mjdend))

        # Extract uv data & perform DFT
        uv = recarr_to_ndarr(obsdata[['u','v']],'f8')
        umin = np.min(np.sqrt(uv[:,0]**2 + uv[:,1]**2))
        umax = np.max(np.sqrt(uv[:,0]**2 + uv[:,1]**2))

        if not mov.psize < 1.0/(2.0*umax):
            print("    Warning!: longest baseline > 1/2 x maximum image spatial wavelength!")
        if not mov.psize*np.sqrt(mov.xdim*mov.ydim) > 1.0/(0.5*umin):
            print("    Warning!: shortest baseline < 2 x minimum image spatial wavelength!")

        vis = np.zeros(len(uv))
        qvis = np.zeros(len(uv))
        uvis = np.zeros(len(uv))
        vvis = np.zeros(len(uv))

        # Get visibilities from FFT with interpolation
        if ttype=="fast":

            # Pad image
            #npad = int(np.ceil(pad_frac*1./(mov.psize*umin)))
            npad = fft_pad_factor * np.max((mov.xdim, mov.ydim))
            npad = power_of_two(npad)

            padvalx1 = padvalx2 = int(np.floor((npad - mov.xdim)/2.0))
            if mov.xdim % 2:
                padvalx2 += 1
            padvaly1 = padvaly2 = int(np.floor((npad - mov.ydim)/2.0))
            if mov.ydim % 2:
                padvaly2 += 1

            imarr = mov.frames[n].reshape(mov.ydim, mov.xdim)
            imarr = np.pad(imarr, ((padvalx1,padvalx2),(padvaly1,padvaly2)), 'constant', constant_values=0.0)
            npad = imarr.shape[0]
            if imarr.shape[0]!=imarr.shape[1]:
                raise Exception("FFT padding did not return a square image!")

            # Scaled uv points
            du = 1.0/(npad*mov.psize)
            uv2 = np.hstack((uv[:,1].reshape(-1,1), uv[:,0].reshape(-1,1)))
            uv2 = (uv2/du + 0.5*npad).T

            # FFT for visibilities
            vis_im = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(imarr)))

            # Sample the visibilities
            visre = nd.map_coordinates(np.real(vis_im), uv2)
            visim = nd.map_coordinates(np.imag(vis_im), uv2)
            vis = visre + 1j*visim

            #extra phase to match centroid convention -- right?
            phase = np.exp(-1j*np.pi*mov.psize*(uv[:,0]+uv[:,1]))
            vis = vis * phase

            # Multiply by the pulse function
            pulsefac = np.array([mov.pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], mov.psize, dom="F") for uvpt in uv])
            vis = vis * pulsefac

            if len(mov.qframes):
                qarr = mov.qframes[n].reshape(mov.ydim, mov.xdim)
                qarr = np.pad(qarr, ((padvalx1,padvalx2),(padvaly1,padvaly2)), 'constant', constant_values=0.0)
                uarr = mov.uframes[n].reshape(mov.ydim, mov.xdim)
                uarr = np.pad(uarr, ((padvalx1,padvalx2),(padvaly1,padvaly2)), 'constant', constant_values=0.0)

                qvis_im = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(qarr)))
                uvis_im = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(uarr)))

                qvisre = nd.map_coordinates(np.real(qvis_im), uv2)
                qvisim = nd.map_coordinates(np.imag(qvis_im), uv2)
                qvis = phase*(qvisre + 1j*qvisim)
                qvis = qvis*pulsefac

                uvisre = nd.map_coordinates(np.real(uvis_im), uv2)
                uvisim = nd.map_coordinates(np.imag(uvis_im), uv2)
                uvis = phase*(uvisre + 1j*uvisim)
                uvis = uvis*pulsefac

            if len(mov.vframes):
                varr = mov.frames[n].reshape(mov.ydim, mov.xdim)
                varr = np.pad(varr, ((padvalx1,padvalx2),(padvaly1,padvaly2)), 'constant', constant_values=0.0)

                vvis_im = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(varr)))

                vvisre = nd.map_coordinates(np.real(vvis_im), uv2)
                vvisim = nd.map_coordinates(np.imag(vvis_im), uv2)
                vvis = phase*(vvisre + 1j*qvisim)
                qvis = vvis*pulsefac

        # Get visibilities from NFFT
        elif ttype=="nfft":

            uvdim = len(uv)
            if (mov.xdim%2 or mov.ydim%2):
                raise Exception("NFFT doesn't work with odd image dimensions!")

            npad = fft_pad_factor * np.max((mov.xdim, mov.ydim))

            #TODO kernel size??
            nker = np.floor(np.min((im.xdim,im.ydim))/5)
            if (nker>50):
                nker = 50
            elif (im.xdim<50 or im.ydim<50):
                nker = np.min((im.xdim,im.ydim))/2
            plan = NFFT([mov.xdim,mov.ydim],uvdim, m=nker, n=[npad,npad])

            #sampled points
            uvlist = uv*mov.psize

            #precompute
            plan.x = uvlist
            plan.precompute()

            #phase and pulsefac
            phase = np.exp(-1j*np.pi*(uvlist[:,0] + uvlist[:,1]))
            pulsefac = np.array([mov.pulse(2*np.pi*uvlist[i,0], 2*np.pi*uvlist[i,1], 1., dom="F") for i in range(uvdim)])

            #compute uniform --> nonuniform transform
            plan.f_hat = mov.frames[n].copy().reshape((mov.ydim,mov.xdim)).T
            plan.trafo()
            vis = plan.f.copy()*phase*pulsefac

            if len(mov.qframes):
                plan.f_hat = mov.qframes[n].copy().reshape((mov.ydim,mov.xdim)).T
                plan.trafo()
                qvis = plan.f.copy()*phase*pulsefac

                plan.f_hat = mov.uframes[n].copy().reshape((mov.ydim,mov.xdim)).T
                plan.trafo()
                uvis = plan.f.copy()*phase*pulsefac
            if len(mov.vframes):
                plan.f_hat = mov.vframes[n].copy().reshape((mov.ydim,mov.xdim)).T
                plan.trafo()
                vvis = plan.f.copy()*phase*pulsefac

        # Get visibilities from DTFT
        else:
            mat = ftmatrix(mov.psize, mov.xdim, mov.ydim, uv, pulse=mov.pulse)
            vis = np.dot(mat, mov.frames[n])

            if len(mov.qframes):
                qvis = np.dot(mat, mov.qframes[n])
                uvis = np.dot(mat, mov.uframes[n])
            if len(mov.vframes):
                vvis = np.dot(mat, mov.vframes[n])

        # Scatter the visibilities with the SgrA* kernel
        if sgrscat:
            for i in range(len(vis)):
                ker = sgra_kernel_uv(mov.rf, uv[i,0], uv[i,1])
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

    # Return observation data
    return obsdata_out

##################################################################################################
# Noise + miscalibration funcitons
##################################################################################################

def make_jones(obs, opacitycal=True, ampcal=True, phasecal=True, dcal=True, frcal=True,
               taup=GAINPDEF, gainp=GAINPDEF, gain_offset=GAINPDEF,
               dtermp=DTERMPDEF, dterm_offset=DTERMPDEF,
               seed=False):

    """Computes Jones Matrices for a list of times (non repeating), with gain and dterm errors.

       Args:
           obs (Obsdata): the observation with scans for the Jones matrices to be computed
           opacitycal (bool): if False, time-dependent gaussian errors are added to station opacities
           ampcal (bool): if False, time-dependent gaussian errors are added to complex station gains
           phasecal (bool): if False, time-dependent station-based random phases are added to complex station gains
           dcal (bool): if False, time-dependent gaussian errors are added to D-terms.
           frcal (bool): if False, feed rotation angle terms are added to Jones matrices.
           taup (float): the fractional std. dev. of the random error on the opacities
           gainp (float): the fractional std. dev. of the random error on the gains
           gain_offset (float): the base gain offset at all sites, or a dict giving one gain offset per site
           dtermp (float): the fractional std. dev. of the random error on the D-terms
           dterm_offset (float): the base dterm offset at all sites, or a dict giving one dterm offset per site
           seed : a seed for the random number generators, uses system time if false

       Returns:
           (dict): a nested dictionary of matrices indexed by the site, then by the time
    """

    tlist = obs.tlist()
    tarr = obs.tarr
    ra = obs.ra
    dec = obs.dec
    sourcevec = np.array([np.cos(dec*DEGREE), 0, np.sin(dec*DEGREE)])
    tproc = str(ttime.time())

    # Create a dictionary of taus and a list of unique times
    nsites = len(obs.tarr['site'])
    taudict = {site : np.array([]) for site in obs.tarr['site']}
    times = np.array([])
    for scan in tlist:
        time = scan['time'][0]
        times = np.append(times, time)
        sites_in = np.array([])
        for bl in scan:
            # Should we screen for conflicting same-time measurements of tau?
            if len(sites_in) >= nsites: break

            if (not len(sites_in)) or (not bl['t1'] in sites_in):
                taudict[bl['t1']] = np.append(taudict[bl['t1']], bl['tau1'])
                sites_in = np.append(sites_in, bl['t1'])

            if (not len(sites_in)) or (not bl['t2'] in sites_in):
                taudict[bl['t2']] = np.append(taudict[bl['t2']], bl['tau2'])
                sites_in = np.append(sites_in, bl['t2'])
        if len(sites_in) < nsites:
            for site in obs.tarr['site']:
                if site not in sites_in:
                    taudict[site] = np.append(taudict[site], 0.0)

    # Compute Sidereal Times
    if obs.timetype=='GMST':
        times_sid=times
    else:
        times_sid = utc_to_gmst(times, obs.mjd)

    # Seed for random number generators
    if seed==False:
        seed=str(ttime.time())

    # Generate Jones Matrices at each time for each telescope
    out = {}
    for i in range(len(tarr)):
        site = tarr[i]['site']
        coords = np.array([tarr[i]['x'],tarr[i]['y'],tarr[i]['z']])
        latlon = xyz_2_latlong(coords)

        # Elevation Angles
        thetas = np.mod((times_sid - ra)*HOUR, 2*np.pi)
        el_angles = elev(earthrot(coords, thetas), sourcevec)

        # Parallactic Angles
        hr_angles = hr_angle(times_sid*HOUR, latlon[:,1], ra*HOUR)
        par_angles = par_angle(hr_angles, latlon[:,0], dec*DEGREE)

        # Amplitude gains
        gainR = gainL = np.ones(len(times))
        if not ampcal:
            if type(gain_offset) == dict:
                goff = gain_offset[site]
            else:
                goff = gain_offset

            gainR = np.sqrt(np.abs(np.array([(1.0 +  goff*hashrandn(site,'gainR',str(goff), seed))*(1.0 + gainp * hashrandn(site, 'gainR', time, str(gainp), seed))
                                     for time in times])))
            gainL = np.sqrt(np.abs(np.array([(1.0 +  goff*hashrandn(site,'gainL',str(goff), seed))*(1.0 + gainp * hashrandn(site, 'gainL', time, str(gainp), seed))
                                     for time in times])))

        # Opacity attenuation of amplitude gain
        if not opacitycal:
            taus = np.abs(np.array([taudict[site][j] * (1.0 + taup * hashrandn(site, 'tau', times[j], seed)) for j in range(len(times))]))
            atten = np.exp(-taus/(EP + 2.0*np.sin(el_angles)))

            gainR = gainR * atten
            gainL = gainL * atten

        # Atmospheric Phase
        if not phasecal:
            phase = np.array([2 * np.pi * hashrand(site, 'phase', time, seed) for time in times])
            gainR = gainR * np.exp(1j*phase)
            gainL = gainL * np.exp(1j*phase)

        # D Term errors
        dR = dL = 0.0
        if not dcal:
            if type(dterm_offset) == dict:
                doff = dterm_offset[site]
            else:
                doff = dterm_offset

            dR = tarr[i]['dr']
            dL = tarr[i]['dl']

            dR += doff * (hashrandn(site, 'dRre', seed) + 1j * hashrandn(site, 'dRim', seed))
            dL += doff * (hashrandn(site, 'dLre', seed) + 1j * hashrandn(site, 'dLim', seed))

            dR *= (1 + dtermp * (hashrandn(site, 'dRre_resid', seed) + 1j * hashrandn(site, 'dRim_resid', seed)))
            dL *= (1 + dtermp * (hashrandn(site, 'dLre_resid', seed) + 1j * hashrandn(site, 'dLim_resid', seed)))

        #print ('dterms:', np.abs(dR),np.abs(dL))
        # Feed Rotation Angles
        fr_angle = np.zeros(len(times))
        if not frcal:
            fr_angle = tarr[i]['fr_elev']*el_angles + tarr[i]['fr_par']*par_angles + tarr[i]['fr_off']*DEGREE

        # Assemble the Jones Matrices and save to dictionary
        # TODO: indexed by utc or sideral time?
        j_matrices = {times[j]: np.array([
                                [np.exp(-1j*fr_angle[j])*gainR[j], np.exp(1j*fr_angle[j])*dR*gainR[j]],
                                [np.exp(-1j*fr_angle[j])*dL*gainL[j], np.exp(1j*fr_angle[j])*gainL[j]]
                                ])
                                for j in range(len(times))
                     }

        out[site] = j_matrices

    return out

def make_jones_inverse(obs, opacitycal=True, dcal=True, frcal=True):

    """Computes inverse Jones Matrices for a list of times (non repeating), with NO gain and dterm errors.

       Args:
           obs (Obsdata): the observation with scans for the inverse Jones matrices to be computed
           opacitycal (bool): if False, estimated opacity terms are applied in the inverse gains
           dcal (bool): if False, estimated inverse d-terms are applied to the inverse Jones matrices
           frcal (bool): if False, inverse feed rotation angle terms are applied to Jones matrices.

       Returns:
           (dict): a nested dictionary of matrices indexed by the site, then by the time
    """

    # Get data
    tlist = obs.tlist()
    tarr = obs.tarr
    ra = obs.ra
    dec = obs.dec
    sourcevec = np.array([np.cos(dec*DEGREE), 0, np.sin(dec*DEGREE)])

    # Create a dictionary of taus and a list of unique times
    nsites = len(obs.tarr['site'])
    taudict = {site : np.array([]) for site in obs.tarr['site']}
    times = np.array([])
    for scan in tlist:
        time = scan['time'][0]
        times = np.append(times, time)
        sites_in = np.array([])
        for bl in scan:

            # Should we screen for conflicting same-time measurements of tau?
            if len(sites_in) >= nsites: break

            if (not len(sites_in)) or (not bl['t1'] in sites_in):
                taudict[bl['t1']] = np.append(taudict[bl['t1']], bl['tau1'])
                sites_in = np.append(sites_in, bl['t1'])

            if (not len(sites_in)) or (not bl['t2'] in sites_in):
                taudict[bl['t2']] = np.append(taudict[bl['t2']], bl['tau2'])
                sites_in = np.append(sites_in, bl['t2'])
        if len(sites_in) < nsites:
            for site in obs.tarr['site']:
                if site not in sites_in:
                    taudict[site] = np.append(taudict[site], 0.0)

    # Compute Sidereal Times
    if obs.timetype=='GMST':
        times_sid=times
    else:
        times_sid = utc_to_gmst(times, obs.mjd)

    # Make inverse Jones Matrices
    out = {}
    for i in range(len(tarr)):
        site = tarr[i]['site']
        coords = np.array([tarr[i]['x'],tarr[i]['y'],tarr[i]['z']])
        latlon = xyz_2_latlong(coords)

        # Elevation Angles
        thetas = np.mod((times_sid - ra)*HOUR, 2*np.pi)
        el_angles = elev(earthrot(coords, thetas), sourcevec)

        # Parallactic Angles (positive longitude EAST)
        hr_angles = hr_angle(times_sid*HOUR, latlon[:,1], ra*HOUR)
        par_angles = par_angle(hr_angles, latlon[:,0], dec*DEGREE)

        # Amplitude gain assumed 1
        gainR = gainL = np.ones(len(times))

        # Opacity attenuation of amplitude gain
        if not opacitycal:
            taus = np.abs(np.array(taudict[site]))
            atten = np.exp(-taus/(EP + 2.0*np.sin(el_angles)))

            gainR = gainR * atten
            gainL = gainL * atten

        # D Terms
        dR = dL = 0.0
        if not dcal:
            dR = tarr[i]['dr']
            dL = tarr[i]['dl']

        # Feed Rotation Angles
        fr_angle = np.zeros(len(times))
        if not frcal:
            # Total Angle (Radian)
            fr_angle = tarr[i]['fr_elev']*el_angles + tarr[i]['fr_par']*par_angles + tarr[i]['fr_off']*DEGREE

        # Assemble the Jones Matrices and save to dictionary
        pref = 1.0/(gainL*gainR*(1.0 - dL*dR))
        j_matrices_inv = {times[j]: pref[j]*np.array([
                                     [np.exp(1j*fr_angle[j])*gainL[j], -np.exp(1j*fr_angle[j])*dR*gainR[j]],
                                     [-np.exp(-1j*fr_angle[j])*dL*gainL[j], np.exp(-1j*fr_angle[j])*gainR[j]]
                                     ]) for j in range(len(times))
                         }

        out[site] = j_matrices_inv

    return out

def add_jones_and_noise(obs, add_th_noise=True,
                        opacitycal=True, ampcal=True, phasecal=True, dcal=True, frcal=True,
                        taup=GAINPDEF, gainp=GAINPDEF, gain_offset=GAINPDEF, dtermp=DTERMPDEF, dterm_offset=DTERMPDEF,
                        seed=False, deepcopy=True):

    """Corrupt visibilities in obs with jones matrices and add thermal noise

       Args:
           obs (Obsdata): the original observation
           add_th_noise (bool): if True, baseline-dependent thermal noise is added to each data point
           opacitycal (bool): if False, time-dependent gaussian errors are added to station opacities
           ampcal (bool): if False, time-dependent gaussian errors are added to complex station gains
           phasecal (bool): if False, time-dependent station-based random phases are added to complex station gains
           dcal (bool): if False, time-dependent gaussian errors are added to D-terms.
           frcal (bool): if False, feed rotation angle terms are added to Jones matrices.
           taup (float): the fractional std. dev. of the random error on the opacities
           gainp (float): the fractional std. dev. of the random error on the gains
           gain_offset (float): the base gain offset at all sites, or a dict giving one gain offset per site
           dtermp (float): the fractional std. dev. of the random error on the D-terms
           dterm_offset (float): the base dterm offset at all sites, or a dict giving one dterm offset per site
           seed : a seed for the random number generators, uses system time if false
           deepcopy (bool) : if True, does a deep copy of the observation before unpacking

       Returns:
           (np.array): an observation  data array
    """


    print("Applying Jones Matrices to data . . . ")
    # Build Jones Matrices
    jm_dict = make_jones(obs,
                         ampcal=ampcal, opacitycal=opacitycal, phasecal=phasecal,dcal=dcal,frcal=frcal,
                         gainp=gainp, taup=taup, gain_offset=gain_offset, dtermp=dtermp, dterm_offset=dterm_offset,
                         seed=seed)
    # Unpack Data
    if deepcopy:
        obsdata = copy.deepcopy(obs.data)
    else:
        obsdata = obs.data

    times = obsdata['time']
    t1 = obsdata['t1']
    t2 = obsdata['t2']
    tints = obsdata['tint']

    # Visibility Data
    rr = obsdata['vis'] + obsdata['vvis']
    ll = obsdata['vis'] - obsdata['vvis']
    rl = obsdata['qvis'] + 1j*obsdata['uvis']
    lr = obsdata['qvis'] - 1j*obsdata['uvis']

    # Recompute the noise std. deviations from the SEFDs
    if np.any(obs.tarr['sefdr'] <= 0) or np.any(obs.tarr['sefdl'] <=0):
        print("Warning!: in add_jones_and_noise, some SEFDs are <= 0!, resorting to data point sigmas which may add too much systematic noise!")
        sigmas = obs.unpack(['rrsigma','llsigma','rlsigma','lrsigma'])
        sig_rr = sigmas['rrsigma']
        sig_ll = sigmas['llsigma']
        sig_rl = sigmas['rlsigma']
        sig_lr = sigmas['lrsigma']
    else:
        sig_rr = np.sqrt(2.)*np.array([blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdr'], obs.tarr[obs.tkey[t2[i]]]['sefdr'], tints[i], obs.bw) for i in range(len(rr))])
        sig_ll = np.sqrt(2.)*np.array([blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdl'], obs.tarr[obs.tkey[t2[i]]]['sefdl'], tints[i], obs.bw) for i in range(len(ll))])
        sig_rl = np.sqrt(2.)*np.array([blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdr'], obs.tarr[obs.tkey[t2[i]]]['sefdl'], tints[i], obs.bw) for i in range(len(rl))])
        sig_lr = np.sqrt(2.)*np.array([blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdl'], obs.tarr[obs.tkey[t2[i]]]['sefdr'], tints[i], obs.bw) for i in range(len(lr))])

    #print "------------------------------------------------------------------------------------------------------------------------"
    if not opacitycal:
        print("   Applying opacity attenuation: opacitycal-->False")
    if not ampcal:
        print("   Applying gain corruption: ampcal-->False")
    if not phasecal:
        print("   Applying atmospheric phase corruption: phasecal-->False")
    if not dcal:
        print("   Applying D Term mixing: dcal-->False")
    if not frcal:
        print("   Applying Field Rotation: frcal-->False")
    if add_th_noise:
        print("Adding thermal noise to data . . . ")
    #print "------------------------------------------------------------------------------------------------------------------------"

    # Corrupt each IQUV visibilty set with the jones matrices and add noise
    for i in range(len(times)):
        # Form the visibility correlation matrix
        corr_matrix = np.array([[rr[i], rl[i]], [lr[i], ll[i]]])

        # Get the jones matrices and corrupt the corr_matrix
        j1 = jm_dict[t1[i]][times[i]]
        j2 = jm_dict[t2[i]][times[i]]

        corr_matrix_corrupt = np.dot(j1, np.dot(corr_matrix, np.conjugate(j2.T)))

        # Add noise
        if add_th_noise:
            noise_matrix = np.array([[cerror(sig_rr[i]), cerror(sig_rl[i])], [cerror(sig_lr[i]), cerror(sig_ll[i])]])
            corr_matrix_corrupt += noise_matrix

        # Put the corrupted data back into the data table
        obsdata['vis'][i]  = 0.5*(corr_matrix_corrupt[0][0] + corr_matrix_corrupt[1][1])
        obsdata['vvis'][i] = 0.5*(corr_matrix_corrupt[0][0] - corr_matrix_corrupt[1][1])
        obsdata['qvis'][i] = 0.5*(corr_matrix_corrupt[0][1] + corr_matrix_corrupt[1][0])
        obsdata['uvis'][i] = -0.5j*(corr_matrix_corrupt[0][1] - corr_matrix_corrupt[1][0])

        # Put the recomputed sigmas back into the data table
        obsdata['sigma'][i] = 0.5*np.sqrt(sig_rr[i]**2 + sig_ll[i]**2)
        obsdata['vsigma'][i] = 0.5*np.sqrt(sig_rr[i]**2 + sig_ll[i]**2)
        obsdata['qsigma'][i] = 0.5*np.sqrt(sig_rl[i]**2 + sig_lr[i]**2)
        obsdata['usigma'][i] = 0.5*np.sqrt(sig_rl[i]**2 + sig_lr[i]**2)

    # Return observation data
    return obsdata

def apply_jones_inverse(obs, opacitycal=True, dcal=True, frcal=True, deepcopy=True):
    """Apply inverse jones matrices to an observation

       Args:
           obs (Obsdata): the original observation
           add_th_noise (bool): if True, baseline-dependent thermal noise is added to each data point
           dcal (bool): if False, time-dependent gaussian errors are added to D-terms.
           frcal (bool): if False, feed rotation angle terms are added to Jones matrices.
           deepcopy (bool) : if True, does a deep copy of the observation before unpacking

       Returns:
           (np.array): an observation data array
    """

    print("Applying a priori calibration with estimated Jones matrices . . . ")
    # Build Inverse Jones Matrices
    jm_dict = make_jones_inverse(obs, opacitycal=opacitycal, dcal=dcal, frcal=frcal)

    # Unpack Data
    if deepcopy:
        obsdata = copy.deepcopy(obs.data)
    else:
        obsdata = obs.data
    times = obsdata['time']
    t1 = obsdata['t1']
    t2 = obsdata['t2']
    tints = obsdata['tint']

    # Visibility Data
    rr = obsdata['vis'] + obsdata['vvis']
    ll = obsdata['vis'] - obsdata['vvis']
    rl = obsdata['qvis'] + 1j*obsdata['uvis']
    lr = obsdata['qvis'] - 1j*obsdata['uvis']

    # Recompute the noise std. deviations from the SEFDs
    if np.any(obs.tarr['sefdr'] <= 0) or np.any(obs.tarr['sefdl'] <=0):
        print("Warning!: in add_jones_and_noise, some SEFDs are <= 0!, resorting to data point sigmas which may add too much systematic noise!")
        sigmas = obs.unpack(['rrsigma','llsigma','rlsigma','lrsigma'])
        sig_rr = sigmas['rrsigma']
        sig_ll = sigmas['llsigma']
        sig_rl = sigmas['rlsigma']
        sig_lr = sigmas['lrsigma']
    else:
        sig_rr = np.array([blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdr'], obs.tarr[obs.tkey[t2[i]]]['sefdr'], tints[i], obs.bw) for i in range(len(rr))])
        sig_ll = np.array([blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdl'], obs.tarr[obs.tkey[t2[i]]]['sefdl'], tints[i], obs.bw) for i in range(len(ll))])
        sig_rl = np.array([blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdr'], obs.tarr[obs.tkey[t2[i]]]['sefdl'], tints[i], obs.bw) for i in range(len(rl))])
        sig_lr = np.array([blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdl'], obs.tarr[obs.tkey[t2[i]]]['sefdr'], tints[i], obs.bw) for i in range(len(lr))])

    ampcal = obs.ampcal
    phasecal = obs.phasecal
    #print "------------------------------------------------------------------------------------------------------------------------"
    if not opacitycal:
        print("   Applying opacity corrections: opacitycal-->True")
        opacitycal=True
    if not dcal:
        print("   Applying D Term corrections: dcal-->True")
        dcal=True
    if not frcal:
        print("   Applying Field Rotation corrections: frcal-->True")
        frcal=True
    #print "------------------------------------------------------------------------------------------------------------------------"

    # Apply the inverse Jones matrices to each visibility
    for i in range(len(times)):
        # Get the inverse jones matrices
        inv_j1 = jm_dict[t1[i]][times[i]]
        inv_j2 = jm_dict[t2[i]][times[i]]

        # Form the visibility correlation matrix
        corr_matrix = np.array([[rr[i], rl[i]], [lr[i], ll[i]]])

        # Form the sigma matrices
        sig_rr_matrix = np.array([[sig_rr[i], 0.0], [0.0, 0.0]])
        sig_ll_matrix = np.array([[0.0, 0.0], [0.0, sig_ll[i]]])
        sig_rl_matrix = np.array([[0.0, sig_rl[i]], [0.0, 0.0]])
        sig_lr_matrix = np.array([[0.0, 0.0], [sig_lr[i], 0.0]])

        # Apply the Jones Matrices to the visibility correlation matrix and sigma matrices
        corr_matrix_new = np.dot(inv_j1, np.dot(corr_matrix, np.conjugate(inv_j2.T)))

        sig_rr_matrix_new = np.dot(inv_j1, np.dot(sig_rr_matrix, np.conjugate(inv_j2.T)))
        sig_ll_matrix_new = np.dot(inv_j1, np.dot(sig_ll_matrix, np.conjugate(inv_j2.T)))
        sig_rl_matrix_new = np.dot(inv_j1, np.dot(sig_rl_matrix, np.conjugate(inv_j2.T)))
        sig_lr_matrix_new = np.dot(inv_j1, np.dot(sig_lr_matrix, np.conjugate(inv_j2.T)))

        # Get the final sigma matrix as a quadrature sum
        sig_matrix_new = np.sqrt(np.abs(sig_rr_matrix_new)**2 + np.abs(sig_ll_matrix_new)**2 +
                                 np.abs(sig_rl_matrix_new)**2 + np.abs(sig_lr_matrix_new)**2)

        # Put the data back into the data table
        obsdata['vis'][i]  = 0.5*(corr_matrix_new[0][0] + corr_matrix_new[1][1])
        obsdata['vvis'][i] = 0.5*(corr_matrix_new[0][0] - corr_matrix_new[1][1])
        obsdata['qvis'][i] = 0.5*(corr_matrix_new[0][1] + corr_matrix_new[1][0])
        obsdata['uvis'][i] = -0.5j*(corr_matrix_new[0][1] - corr_matrix_new[1][0])

        # Put the recomputed sigmas back into the data table
        obsdata['sigma'][i] = 0.5*np.sqrt(sig_matrix_new[0][0]**2 + sig_matrix_new[1][1]**2)
        obsdata['vsigma'][i] = 0.5*np.sqrt(sig_matrix_new[0][0]**2 + sig_matrix_new[1][1]**2)
        obsdata['qsigma'][i] = 0.5*np.sqrt(sig_matrix_new[0][1]**2 + sig_matrix_new[1][0]**2)
        obsdata['usigma'][i] = 0.5*np.sqrt(sig_matrix_new[0][1]**2 + sig_matrix_new[1][0]**2)

    # Return observation data
    return obsdata

# The old noise generating function.
def add_noise(obs, add_th_noise=True, opacitycal=True, ampcal=True, phasecal=True,
              taup=GAINPDEF, gainp=GAINPDEF, gain_offset=GAINPDEF,
              seed=False, deepcopy=True):

    """Add thermal noise and gain & phase calibration errors to a dataset. Old routine replaced by add_jones_and_noise.

       Args:
           obs (Obsdata): the original observation
           add_th_noise (bool): if True, baseline-dependent thermal noise is added to each data point
           opacitycal (bool): if False, time-dependent gaussian errors are added to station opacities
           ampcal (bool): if False, time-dependent gaussian errors are added to complex station gains
           phasecal (bool): if False, time-dependent station-based random phases are added to complex station gains
           taup (float): the fractional std. dev. of the random error on the opacities
           gainp (float): the fractional std. dev. of the random error on the gains
           gain_offset (float): the base gain offset at all sites, or a dict giving one gain offset per site
           seed : a seed for the random number generators, uses system time if false
           deepcopy (bool) : if True, does a deep copy of the observation before unpacking

       Returns:
           (np.array): an observation data array
    """

    print("Adding gain + phase errors to data and applying a priori calibration . . . ")
    #print "------------------------------------------------------------------------------------------------------------------------"
    if not opacitycal:
        print("   Applying opacity attenuation AND estimated opacity corrections: opacitycal-->True")
    if not ampcal:
        print("   Applying gain corruption: ampcal-->False")
    if not phasecal:
        print("   Applying atmospheric phase corruption: phasecal-->False")
    if add_th_noise:
        print("Adding thermal noise to data . . . ")
    #print "------------------------------------------------------------------------------------------------------------------------"

    # Get data
    if deepcopy:
        obsdata = copy.deepcopy(obs.data)
    else:
        obsdata = obs.data

    sites = recarr_to_ndarr(obsdata[['t1','t2']],'U32')
    uv = recarr_to_ndarr(obsdata[['u','v']],'f8')
    taus = np.abs(recarr_to_ndarr(obsdata[['tau1','tau2']],'f8'))
    elevs = recarr_to_ndarr(obs.unpack(['el1','el2'],ang_unit='deg'),'f8')
    time = obsdata['time']
    tint = obsdata['tint']
    vis  = obsdata['vis']
    qvis = obsdata['qvis']
    uvis = obsdata['uvis']
    vvis = obsdata['vvis']

    # Recompute perfect sigmas from SEFDs
    bw = obs.bw
    if np.any(obs.tarr['sefdr'] <= 0):
        print("Warning!: in add_noise, some SEFDs are <= 0 -- not recomputing sigmas, which may result in double systematic noise")
        sigma_perf = np.array(obsdata['sigma'])
    else:
        sigma_perf = np.array([blnoise(obs.tarr[obs.tkey[sites[i][0]]]['sefdr'], obs.tarr[obs.tkey[sites[i][1]]]['sefdr'], tint[i], bw)
                               for i in range(len(tint))])

    # Seed for random number generators
    if seed==False:
        seed=str(ttime.time())

    # Add gain and opacity fluctuations to the TRUE noise
    if not ampcal:
        # Amplitude gain
        if type(gain_offset) == dict:
            goff1 = gain_offset[sites[i,0]]
            goff2 = gain_offset[sites[i,1]]
        else:
            goff1=goff2=gain_offset

        gain1 = np.abs(np.array([(1.0 + goff1 * hashrandn(sites[i,0], 'gain', seed))*(1.0 + gainp * hashrandn(sites[i,0], 'gain', time[i], seed))
                                 for i in range(len(time))]))
        gain2 = np.abs(np.array([(1.0 + goff2 * hashrandn(sites[i,1], 'gain', seed))*(1.0 + gainp * hashrandn(sites[i,1], 'gain', time[i], seed))
                                 for i in range(len(time))]))
        gain_true = np.sqrt(gain1 * gain2)
    else:
        gain_true = 1

    if not opacitycal:
        # Use estimated opacity to compute the ESTIMATED noise
        tau_est = np.sqrt(np.exp(taus[:,0]/(EP+np.sin(elevs[:,0]*DEGREE)) + taus[:,1]/(EP+np.sin(elevs[:,1]*DEGREE))))

        # Opacity Errors
        tau1 = np.abs(np.array([taus[i,0]* (1.0 + taup * hashrandn(sites[i,0], 'tau', time[i], seed)) for i in range(len(time))]))
        tau2 = np.abs(np.array([taus[i,1]* (1.0 + taup * hashrandn(sites[i,1], 'tau', time[i], seed)) for i in range(len(time))]))

        # Correct noise RMS for opacity
        tau_true = np.sqrt(np.exp(tau1/(EP+np.sin(elevs[:,0]*DEGREE)) + tau2/(EP+np.sin(elevs[:,1]*DEGREE))))
    else:
        tau_true = tau_est = 1

    # Add the noise
    sigma_true = sigma_perf
    sigma_est = sigma_perf * gain_true * tau_est

    if add_th_noise:
        vis  = (vis  + cerror(sigma_true))
        qvis = (qvis + cerror(sigma_true))
        uvis = (uvis + cerror(sigma_true))
        vvis = (vvis + cerror(sigma_true))

    # Add the gain error to the true visibilities
    vis =   vis * gain_true * tau_est / tau_true
    qvis = qvis * gain_true * tau_est / tau_true
    uvis = uvis * gain_true * tau_est / tau_true
    vvis = vvis * gain_true * tau_est / tau_true

    # Add random atmospheric phases
    if not phasecal:
        phase1 = np.array([2 * np.pi * hashrand(sites[i,0], 'phase', time[i], seed) for i in range(len(time))])
        phase2 = np.array([2 * np.pi * hashrand(sites[i,1], 'phase', time[i], seed) for i in range(len(time))])

        vis  *= np.exp(1j * (phase2-phase1))
        qvis *= np.exp(1j * (phase2-phase1))
        uvis *= np.exp(1j * (phase2-phase1))
        vvis *= np.exp(1j * (phase2-phase1))

    # Put the visibilities estimated errors back in the obsdata array
    obsdata['vis'] = vis
    obsdata['qvis'] = qvis
    obsdata['uvis'] = uvis
    obsdata['vvis'] = vvis
    obsdata['sigma'] = sigma_est

    # This function doesn't use different Stokes sigmas!
    obsdata['qsigma'] = obsdata['usigma'] = obsdata['vsigma'] = sigma_est

	# Return observation data
    return obsdata
