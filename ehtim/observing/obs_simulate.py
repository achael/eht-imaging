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

def make_uvpoints(array, ra, dec, rf, bw, tint, tadv, tstart, tstop, polrep='stokes',
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
           polrep (str): 'stokes' or 'circ' sets the data polarimetric representtion
           mjd (int): the mjd of the observation, if different from the image mjd
           timetype (str): how to interpret tstart and tstop; either 'GMST' or 'UTC'
           elevmin (float): station minimum elevation in degrees
           elevmax (float): station maximum elevation in degrees
           tau (float): the base opacity at all sites, or a dict giving one opacity per site
           fix_theta_GMST (bool): if True, stops earth rotation to sample fixed u,v points through time

       Returns:
           (Obsdata): an observation object with all visibilities zeroed
    """

    if polrep=='stokes':
        poltype = DTPOL_STOKES
    elif polrep=='circ':
        poltype = DTPOL_CIRC
    else:
        raise Exception("only 'stokes' and 'circ' are supported polreps!")

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
                if np.any(array.tarr['sefdr'] <= 0) or np.any(array.tarr['sefdl'] <=0):
                    print("Warning!: in make_uvpoints, some SEFDs are <= 0!")

                sig_rr = blnoise(array.tarr[i1]['sefdr'], array.tarr[i2]['sefdr'], tint, bw)
                sig_ll = blnoise(array.tarr[i1]['sefdl'], array.tarr[i2]['sefdl'], tint, bw)
                sig_rl = blnoise(array.tarr[i1]['sefdr'], array.tarr[i2]['sefdl'], tint, bw)
                sig_lr = blnoise(array.tarr[i1]['sefdl'], array.tarr[i2]['sefdr'], tint, bw)
                if polrep=='stokes':
                    sig_iv = 0.5*np.sqrt(sig_rr**2 + sig_ll**2)
                    sig_qu = 0.5*np.sqrt(sig_rl**2 + sig_lr**2)
                    sig1 = sig_iv
                    sig2 = sig_qu
                    sig3 = sig_qu
                    sig4 = sig_iv
                elif polrep=='circ':
                    sig1 = sig_rr
                    sig2 = sig_ll
                    sig3 = sig_rl
                    sig4 = sig_lr

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
                              0.0, # 1st Visibility (Jy)
                              0.0, # 2nd Visibility
                              0.0, # 3rd Visibility
                              0.0, # 4th Visibility
                              sig1, # 1st Sigma (Jy)
                              sig2, # 2nd Sigma
                              sig3, # 3rd Sigma
                              sig4  # 4th Sigma
                            ), dtype=poltype
                            ))

    obsarr = np.array(outlist)

    if not len(obsarr):
        raise Exception("No mutual visibilities in the specified time range!")

    return obsarr

##################################################################################################
# Observe w/o noise
##################################################################################################

def sample_vis(im_org, uv, sgrscat=False, polrep_obs='stokes',
               ttype="nfft", cache=False, fft_pad_factor=2, zero_empty_pol=True):

    """Observe a image on given baselines with no noise.

       Args:
           im (Image): the image to be observed
           uv (ndarray): an array of u,v coordinates
           sgrscat (bool): if True, the visibilites will be blurred by the Sgr A* scattering kernel
           polrep_obs (str): 'stokes' or 'circ' sets the data polarimetric representtion
           ttype (str): if "fast" or 'nfft', use FFT to produce visibilities. Else "direct" for DTFT
           fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in FFT
           zero_empty_pol (bool): if True, returns zero vec if the polarization doesn't exist. Otherwise return None

       Returns:
           (Obsdata): an observation object
    """

    #TODO -- all imports should go at the top
    # but circular imports cause headaches....
    from ehtim.obsdata import Obsdata

    if polrep_obs=='stokes':
        im = im_org.switch_polrep('stokes','I')
        pollist = ['I','Q','U','V'] #TODO what if we have to I image?
    elif polrep_obs=='circ':
        im = im_org.switch_polrep('circ','RR')
        pollist = ['RR','LL','RL','LR'] #TODO what if we have to RR image?
    else:
        raise Exception("only 'stokes' and 'circ' are supported polreps!")

    uv = np.array(uv)
    if uv.shape[1] != 2:
        raise Exception("When given as a list of uv points, the obs should be a list of pairs of u-v coordinates!")
    if im.pa != 0.0:
        c = np.cos(im.pa)
        s = np.sin(im.pa)
        u = uv[:,0]
        v = uv[:,1]
        uv = np.column_stack([c * u - s * v,
                              s * u + c * v])

#    umin = np.min(np.sqrt(uv[:,0]**2 + uv[:,1]**2))
#    umax = np.max(np.sqrt(uv[:,0]**2 + uv[:,1]**2))

#    if not im.psize < 1.0/(2.0*umax):
#        print("    Warning!: longest baseline > 1/2 x maximum image spatial wavelength!")
#    if not im.psize*np.sqrt(im.xdim*im.ydim) > 1.0/(0.5*umin):
#        print("    Warning!: shortest baseline < 2 x minimum image spatial wavelength!")

    obsdata = []

    # Get visibilities from straightforward FFT
    if ttype=="fast":

        # Padded image size
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

        # Extra phase to match centroid convention
        phase = np.exp(-1j*np.pi*im.psize*((1+im.xdim%2)*uv[:,0] + (1+im.ydim%2)*uv[:,1]))

        # Pulse function
        pulsefac = np.fromiter((im.pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], im.psize, dom="F") for uvpt in uv),'c16')

        for i in range(4):
            pol = pollist[i]
            imvec = im._imdict[pol]
            if imvec is None or len(imvec)==0:
                if zero_empty_pol:
                    obsdata.append(np.zeros(len(uv)))
                else:
                    obsdata.append(None)
            else:
                # FFT for visibilities
                if pol in im_org.cached_fft:
                    vis_im = im_org.cached_fft[pol]
                else:
                    imarr = imvec.reshape(im.ydim, im.xdim)
                    imarr = np.pad(imarr, ((padvalx1,padvalx2),(padvaly1,padvaly2)), 'constant', constant_values=0.0)
                    vis_im = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(imarr)))
                    if cache == 'auto':
                        im_org.cached_fft[pol] = vis_im

                # Sample the visibilities
                # default is cubic spline interpolation
                visre = nd.map_coordinates(np.real(vis_im), uv2)
                visim = nd.map_coordinates(np.imag(vis_im), uv2)
                vis = visre + 1j*visim

                # Extra phase and pulse factor
                vis = vis * phase * pulsefac

                # Return visibilities
                obsdata.append(vis)


    # Get visibilities from the NFFT
    elif ttype=="nfft":

        uvdim = len(uv)
        if (im.xdim%2 or im.ydim%2):
            raise Exception("NFFT doesn't work with odd image dimensions!")

        npad = fft_pad_factor * np.max((im.xdim, im.ydim))

        #TODO what is a good kernel size??
        nker = np.floor(np.min((im.xdim,im.ydim))/5)
        if (nker>50):
            nker = 50
        elif (im.xdim<50 or im.ydim<50):
            nker = np.min((im.xdim,im.ydim))/2

        #TODO are y & x reversed?
        plan = NFFT([im.xdim,im.ydim],uvdim, m=nker, n=[npad,npad])

        # Sampled uv points
        uvlist = uv*im.psize

        # Precompute
        plan.x = uvlist
        plan.precompute()

        # Extra phase and pulsefac
        phase = np.exp(-1j*np.pi*(uvlist[:,0] + uvlist[:,1]))
        pulsefac = np.fromiter((im.pulse(2*np.pi*uvlist[i,0], 2*np.pi*uvlist[i,1], 1., dom="F") for i in range(uvdim)),'c16')

        # Compute the uniform --> nonuniform transform for different polarizations
        for i in range(4):
            pol = pollist[i]
            imvec = im._imdict[pol]
            if imvec is None or len(imvec)==0:
                if zero_empty_pol:
                    obsdata.append(np.zeros(len(uv)))
                else:
                    obsdata.append(None)
            else:
                plan.f_hat = imvec.copy().reshape((im.ydim,im.xdim)).T
                plan.trafo()
                vis = plan.f.copy()*phase*pulsefac

                obsdata.append(vis)

    # Get visibilities from DTFT
    else:
        # Construct Fourier matrix
        mat = ftmatrix(im.psize, im.xdim, im.ydim, uv, pulse=im.pulse)

        # Compute DTFT for different polarizations
        for i in range(4):
            pol = pollist[i]
            imvec = im._imdict[pol]
            if imvec is None or len(imvec)==0:
                if zero_empty_pol:
                    obsdata.append(np.zeros(len(uv)))
                else:
                    obsdata.append(None)
            else:
                vis = np.dot(mat, imvec)
                obsdata.append(vis)

    # Scatter the visibilities with the SgrA* kernel
    # TODO: fix sgra_kernel_uv so you can take this out of a  loop!
    if sgrscat:
        print('Scattering Visibilities with Sgr A* kernel!')
        ker =  np.fromiter((sgra_kernel_uv(im.rf, uv[i,0], uv[i,1]) for i in range(len(vis))),'c16')
        for data in obsdata:
            if data is None: continue
            data *= ker

    return obsdata

##################################################################################################
# Noise + miscalibration funcitons
##################################################################################################

def make_jones(obs, opacitycal=True, ampcal=True, phasecal=True, dcal=True, 
               frcal=True, rlgaincal=True,
               stabilize_scan_phase=False, stabilize_scan_amp=False, neggains=False,
               taup=GAINPDEF, gainp=GAINPDEF, gain_offset=GAINPDEF, dterm_offset=DTERMPDEF,
               caltable_path=None, seed=False):

    """Computes Jones Matrices for a list of times (non repeating), with gain and dterm errors.

       Args:
           obs (Obsdata): the observation with scans for the Jones matrices to be computed
           opacitycal (bool): if False, time-dependent gaussian errors are added to station opacities
           ampcal (bool): if False, time-dependent gaussian errors are added to complex station gains
           phasecal (bool): if False, time-dependent station-based random phases are added to complex station gains
           dcal (bool): if False, time-dependent gaussian errors are added to D-terms.
           frcal (bool): if False, feed rotation angle terms are added to Jones matrices.
           stabilize_scan_phase (bool): if True, random phase errors are constant over scans
           stabilize_scan_amp (bool): if True, random amplitude errors are constant over scans
           taup (float): the fractional std. dev. of the random error on the opacities
           gainp (float): the fractional std. dev. of the random error on the gains
           gain_offset (float): the base gain offset at all sites, or a dict giving one gain offset per site
           dterm_offset (float): the base std. dev. of random additive error at all sites, or a dict giving one std. dev. per site
           seed : a seed for the random number generators, uses system time if false

       Returns:
           (dict): a nested dictionary of matrices indexed by the site, then by the time
    """

    obs_tmp = obs.copy()
    tlist = obs_tmp.tlist()
    tarr = obs_tmp.tarr
    ra = obs_tmp.ra
    dec = obs_tmp.dec
    sourcevec = np.array([np.cos(dec*DEGREE), 0, np.sin(dec*DEGREE)])
    tproc = str(ttime.time())

    # Create a dictionary of taus and a list of unique times
    nsites = len(obs_tmp.tarr['site'])
    taudict = {site : np.array([]) for site in obs_tmp.tarr['site']}
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
            for site in obs_tmp.tarr['site']:
                if site not in sites_in:
                    taudict[site] = np.append(taudict[site], 0.0)

    # Now define a list that accounts for periods where the phase or amplitude errors
    # are stable (e.g., over scans if stabilize_scan_phase==True)
    times_stable_phase = times.copy()
    times_stable_amp = times.copy()
    times_stable = times.copy()
    if stabilize_scan_phase==True or stabilize_scan_amp==True:
        scans = obs_tmp.scans
        if np.all(scans) == None or len(scans) == 0:
            obs_scans = obs.copy()
            obs_scans.add_scans()
            scans = obs_scans.scans
        for j in range(len(times_stable)):
            for scan in scans:
                if scan[0] <= times_stable[j] and scan[1] >= times_stable[j]:
                    times_stable[j] = scan[0]
                    break

    if stabilize_scan_phase==True: times_stable_phase = times_stable.copy()
    if stabilize_scan_amp==True: times_stable_amp = times_stable.copy()

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
    datatables = {}
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

            # gain offset: time independent, polarization dependent
            if type(gain_offset) == dict:
                goff = gain_offset[site]
            else:
                goff = gain_offset

            # gain_mult: time dependent, polarization independent
            if type(gainp) == dict:
                gain_mult = gainp[site]
            else:
                gain_mult = gainp

            # Note: R/L gain ratio is independent of time for each site
            # TODO: enforce gainR and gainL < 1

            if rlgaincal:
                gainr_string = 'gain'
                gainl_string = 'gain'
            else:
                gainr_string = 'gainR'
                gainl_string = 'gainL'
            
            
            gainR_constant = goff * hashrandn(site,gainr_string,str(goff),seed)
            gainL_constant = goff * hashrandn(site,gainl_string,str(goff),seed)
            if neggains:
                gainR_constant = -np.abs(gainR_constant)
                gainL_constant = -np.abs(gainL_constant)


            gainR = np.sqrt(np.abs(np.fromiter((
                                                (1.0 + gainR_constant) *
                                                (1.0 + gain_mult * hashrandn(site,'gain',str(time),str(gain_mult),seed))
                                                for time in times_stable_amp
                                               ),float)))

            gainL = np.sqrt(np.abs(np.fromiter((
                                                (1.0 + gainL_constant) *
                                                (1.0 + gain_mult * hashrandn(site,'gain',str(time),str(gain_mult),seed))
                                                for time in times_stable_amp
                                               ),float)))

        # Opacity attenuation of amplitude gain
        if not opacitycal:
            taus = np.abs(np.fromiter((
                                       taudict[site][j] * (1.0 + taup * hashrandn(site, 'tau', times_stable_amp[j], seed))
                                       for j in range(len(times))
                                      ), float))
            atten = np.exp(-taus/(EP + 2.0*np.sin(el_angles)))

            gainR = gainR * atten
            gainL = gainL * atten

        # Atmospheric Phase
        if not phasecal:
            phase = np.fromiter((2 * np.pi * hashrand(site, 'phase', time, seed) for time in times_stable_phase),float)
            gainR = gainR * np.exp(1j*phase)
            gainL = gainL * np.exp(1j*phase)


        # D Term errors
        dR = dL = 0.0
        if not dcal:

            # D terms are always time-independent
            if type(dterm_offset) == dict:
                doff = dterm_offset[site]
            else:
                doff = dterm_offset

            dR = tarr[i]['dr']
            dL = tarr[i]['dl']

            dR += doff * (hashrandn(site, 'dRre', seed) + 1j * hashrandn(site, 'dRim', seed))
            dL += doff * (hashrandn(site, 'dLre', seed) + 1j * hashrandn(site, 'dLim', seed))

        # Feed Rotation Angles
        fr_angle = np.zeros(len(times))
        fr_angle_D = np.zeros(len(times))

        # Field rotation has not been corrected
        if not frcal:
            fr_angle = tarr[i]['fr_elev']*el_angles + tarr[i]['fr_par']*par_angles + tarr[i]['fr_off']*DEGREE

        # If field rotation has been corrected, but leakage has not been corrected, the leakage needs to rotate doubly
        elif frcal and not dcal:

            fr_angle_D = 2.0*(tarr[i]['fr_elev']*el_angles + tarr[i]['fr_par']*par_angles + tarr[i]['fr_off']*DEGREE)

        # Assemble the Jones Matrices and save to dictionary
        j_matrices = {times[j]: np.array([
                                [np.exp(-1j*fr_angle[j])*gainR[j], np.exp(1j*(fr_angle[j]+fr_angle_D[j]))*dR*gainR[j]],
                                [np.exp(-1j*(fr_angle[j]+fr_angle_D[j]))*dL*gainL[j], np.exp(1j*fr_angle[j])*gainL[j]]
                                ])
                                for j in range(len(times))
                     }

        out[site] = j_matrices

        if caltable_path:
            obs_tmp.tarr[i]['dr'] = dR
            obs_tmp.tarr[i]['dl'] = dL
            datatable = []
            for j in range(len(times)):
                datatable.append(np.array((times[j], gainR[j], gainL[j]), dtype=DTCAL))
            datatables[site] = np.array(datatable)

    # Save a calibration table with the synthetic gains and dterms added
    if caltable_path and len(datatables)>0:
        caltable = ehtim.caltable.Caltable(obs_tmp.ra, obs_tmp.dec, obs_tmp.rf, obs_tmp.bw, datatables, obs_tmp.tarr, source=obs_tmp.source, mjd=obs_tmp.mjd, timetype=obs_tmp.timetype)
        caltable.save_txt(obs_tmp, datadir=caltable_path+'_simdata_caltable')

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
        fr_angle_D = np.zeros(len(times)) # This is for when field rotation is corrected but not leakage
        if not frcal:
            # Total Angle (Radian)
            fr_angle = tarr[i]['fr_elev']*el_angles + tarr[i]['fr_par']*par_angles + tarr[i]['fr_off']*DEGREE

        elif frcal and not dcal:
            # If the field rotation angle has been removed but leakage hasn't, we still need to rotate the leakage terms appropriately (by *twice* the field rotation angle)
            fr_angle_D = 2.0*(tarr[i]['fr_elev']*el_angles + tarr[i]['fr_par']*par_angles + tarr[i]['fr_off']*DEGREE)

        # Assemble the inverse Jones Matrices and save to dictionary
        pref = 1.0/(gainL*gainR*(1.0 - dL*dR))
        j_matrices_inv = {times[j]: pref[j]*np.array([
                                     [ np.exp( 1j*fr_angle[j])*gainL[j],   -np.exp( 1j*(fr_angle[j] + fr_angle_D[j]))*dR*gainR[j]],
                                     [-np.exp(-1j*(fr_angle[j] + fr_angle_D[j]))*dL*gainL[j], np.exp(-1j*fr_angle[j])*gainR[j]]
                                     ]) for j in range(len(times))
                         }

        out[site] = j_matrices_inv

    return out

def add_jones_and_noise(obs, add_th_noise=True,
                        opacitycal=True, ampcal=True, phasecal=True, dcal=True, frcal=True, rlgaincal=True,
                        stabilize_scan_phase=False, stabilize_scan_amp=False, neggains=False,
                        taup=GAINPDEF, gainp=GAINPDEF, gain_offset=GAINPDEF, dterm_offset=DTERMPDEF,
                        caltable_path=None, seed=False):


    """Corrupt visibilities in obs with jones matrices and add thermal noise

       Args:
           obs (Obsdata): the original observation
           add_th_noise (bool): if True, baseline-dependent thermal noise is added to each data point
           opacitycal (bool): if False, time-dependent gaussian errors are added to station opacities
           ampcal (bool): if False, time-dependent gaussian errors are added to complex station gains
           phasecal (bool): if False, time-dependent station-based random phases are added to complex station gains
           dcal (bool): if False, time-dependent gaussian errors are added to D-terms.
           frcal (bool): if False, feed rotation angle terms are added to Jones matrices.
           stabilize_scan_phase (bool): if True, random phase errors are constant over scans
           stabilize_scan_amp (bool): if True, random amplitude errors are constant over scans
           taup (float): the fractional std. dev. of the random error on the opacities
           gainp (float): the fractional std. dev. of the random error on the gains
           gain_offset (float): the base gain offset at all sites, or a dict giving one gain offset per site
           dterm_offset (float): the base std. dev. of random additive error at all sites, or a dict giving one std. dev. per site
           seed : a seed for the random number generators, uses system time if false


       Returns:
           (np.array): an observation  data array
    """


    print("Applying Jones Matrices to data . . . ")
    # Build Jones Matrices
    jm_dict = make_jones(obs,
                         ampcal=ampcal, opacitycal=opacitycal, phasecal=phasecal,dcal=dcal,frcal=frcal,rlgaincal=rlgaincal,
                         stabilize_scan_phase=stabilize_scan_phase,stabilize_scan_amp=stabilize_scan_amp, neggains=neggains,
                         gainp=gainp, taup=taup, gain_offset=gain_offset, dterm_offset=dterm_offset,
                         caltable_path=caltable_path, seed=seed)

    # Change pol rep:
    obs_circ = obs.switch_polrep('circ')
    obsdata = copy.copy(obs_circ.data)

    times = obsdata['time']
    t1    = obsdata['t1']
    t2    = obsdata['t2']
    tints = obsdata['tint']
    rr    = obsdata['rrvis']
    ll    = obsdata['llvis']
    rl    = obsdata['rlvis']
    lr    = obsdata['lrvis']

    # Recompute the noise std. deviations from the SEFDs
    if np.any(obs.tarr['sefdr'] <= 0) or np.any(obs.tarr['sefdl'] <=0):
        print("Warning!: in add_jones_and_noise, some SEFDs are <= 0!, resorting to data point sigmas which may add too much systematic noise!")
        sig_rr = obsdata['rrsigma']
        sig_ll = obsdata['llsigma']
        sig_rl = obsdata['rlsigma']
        sig_lr = obsdata['lrsigma']
    else:
        sig_rr = np.fromiter((blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdr'], obs.tarr[obs.tkey[t2[i]]]['sefdr'], tints[i], obs.bw) for i in range(len(rr))),float)
        sig_ll = np.fromiter((blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdl'], obs.tarr[obs.tkey[t2[i]]]['sefdl'], tints[i], obs.bw) for i in range(len(ll))),float)
        sig_rl = np.fromiter((blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdr'], obs.tarr[obs.tkey[t2[i]]]['sefdl'], tints[i], obs.bw) for i in range(len(rl))),float)
        sig_lr = np.fromiter((blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdl'], obs.tarr[obs.tkey[t2[i]]]['sefdr'], tints[i], obs.bw) for i in range(len(lr))),float)

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

    # Corrupt each IQUV visibility set with the jones matrices and add noise
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
        obsdata['rrvis'][i] = corr_matrix_corrupt[0][0]
        obsdata['llvis'][i] = corr_matrix_corrupt[1][1]
        obsdata['rlvis'][i] = corr_matrix_corrupt[0][1]
        obsdata['lrvis'][i] = corr_matrix_corrupt[1][0]

        # Put the recomputed sigmas back into the data table
        obsdata['rrsigma'][i] = sig_rr[i]
        obsdata['llsigma'][i] = sig_ll[i]
        obsdata['rlsigma'][i] = sig_rl[i]
        obsdata['lrsigma'][i] = sig_lr[i]

    # put back into input polvec
    obs_circ.data = obsdata
    obs_back = obs_circ.switch_polrep(obs.polrep)
    obsdata_back = obs_back.data

    # Return observation data
    return obsdata_back

def apply_jones_inverse(obs, opacitycal=True, dcal=True, frcal=True, verbose=True):

    """Apply inverse jones matrices to an observation

       Args:
           obs (Obsdata): the original observation
           dcal (bool): if False, time-dependent gaussian errors are added to D-terms.
           frcal (bool): if False, feed rotation angle terms are added to Jones matrices.

       Returns:
           (np.array): an observation data array
    """

    if verbose: print("Applying a priori calibration with estimated Jones matrices . . . ")
    # Build Inverse Jones Matrices
    jm_dict = make_jones_inverse(obs, opacitycal=opacitycal, dcal=dcal, frcal=frcal)

    # Change pol rep:
    obs_circ = obs.switch_polrep('circ')
    ampcal = obs.ampcal
    phasecal = obs.phasecal

    # Get data
    obsdata = copy.deepcopy(obs_circ.data)
    times = obsdata['time']
    t1 = obsdata['t1']
    t2 = obsdata['t2']
    tints = obsdata['tint']
    rr = obsdata['rrvis']
    ll = obsdata['llvis']
    rl = obsdata['rlvis']
    lr = obsdata['lrvis']


    # Recompute the noise std. deviations from the SEFDs
    if np.any(obs.tarr['sefdr'] <= 0) or np.any(obs.tarr['sefdl'] <=0):
        if verbose: print("Warning!: in add_jones_and_noise, some SEFDs are <= 0!, resorting to data point sigmas which may add too much systematic noise!")
        if verbose: print("Warning!: in add_jones_and_noise, some SEFDs are <= 0!, resorting to data point sigmas which may add too much systematic noise!")
        sig_rr = obsdata['rrsigma']
        sig_ll = obsdata['llsigma']
        sig_rl = obsdata['rlsigma']
        sig_lr = obsdata['lrsigma']
    else:
        # TODO why are there sqrt(2)s here and not below?
        sig_rr = np.fromiter((blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdr'], obs.tarr[obs.tkey[t2[i]]]['sefdr'], tints[i], obs.bw) for i in range(len(rr))),float)
        sig_ll = np.fromiter((blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdl'], obs.tarr[obs.tkey[t2[i]]]['sefdl'], tints[i], obs.bw) for i in range(len(ll))),float)
        sig_rl = np.fromiter((blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdr'], obs.tarr[obs.tkey[t2[i]]]['sefdl'], tints[i], obs.bw) for i in range(len(rl))),float)
        sig_lr = np.fromiter((blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdl'], obs.tarr[obs.tkey[t2[i]]]['sefdr'], tints[i], obs.bw) for i in range(len(lr))),float)

    #print "------------------------------------------------------------------------------------------------------------------------"
    if not opacitycal:
        if verbose: print("   Applying opacity corrections: opacitycal-->True")
        opacitycal=True
    if not dcal:
        if verbose: print("   Applying D Term corrections: dcal-->True")
        dcal=True
    if not frcal:
        if verbose: print("   Applying Field Rotation corrections: frcal-->True")
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

        # Apply the inverse Jones Matrices to the visibility correlation matrix and sigma matrices
        corr_matrix_new = np.dot(inv_j1, np.dot(corr_matrix, np.conjugate(inv_j2.T)))

        sig_rr_matrix_new = np.dot(inv_j1, np.dot(sig_rr_matrix, np.conjugate(inv_j2.T)))
        sig_ll_matrix_new = np.dot(inv_j1, np.dot(sig_ll_matrix, np.conjugate(inv_j2.T)))
        sig_rl_matrix_new = np.dot(inv_j1, np.dot(sig_rl_matrix, np.conjugate(inv_j2.T)))
        sig_lr_matrix_new = np.dot(inv_j1, np.dot(sig_lr_matrix, np.conjugate(inv_j2.T)))

        # Get the final sigma matrix as a quadrature sum
        # TODO is this right?
        sig_matrix_new = np.sqrt(np.abs(sig_rr_matrix_new)**2 + np.abs(sig_ll_matrix_new)**2 +
                                 np.abs(sig_rl_matrix_new)**2 + np.abs(sig_lr_matrix_new)**2)

        # Put the corrupted data back into the data table
        obsdata['rrvis'][i] = corr_matrix_new[0][0]
        obsdata['llvis'][i] = corr_matrix_new[1][1]
        obsdata['rlvis'][i] = corr_matrix_new[0][1]
        obsdata['lrvis'][i] = corr_matrix_new[1][0]

        # Put the recomputed sigmas back into the data table
        obsdata['rrsigma'][i] = sig_matrix_new[0][0]
        obsdata['llsigma'][i] = sig_matrix_new[1][1]
        obsdata['rlsigma'][i] = sig_matrix_new[0][1]
        obsdata['lrsigma'][i] = sig_matrix_new[1][0]

    # put back into input polvec
    obs_circ.data = obsdata
    obs_back = obs_circ.switch_polrep(obs.polrep)
    obsdata_back = obs_back.data

    # Return observation data
    return obsdata_back

# The old noise generating function.
def add_noise(obs, add_th_noise=True, opacitycal=True, ampcal=True, phasecal=True,
              stabilize_scan_amp=False, stabilize_scan_phase=False,
              taup=GAINPDEF, gainp=GAINPDEF, gain_offset=GAINPDEF,
              seed=False):

    """Add thermal noise and gain & phase calibration errors to a dataset. Old routine replaced by add_jones_and_noise.

       Args:
           obs (Obsdata): the original observation
           add_th_noise (bool): if True, baseline-dependent thermal noise is added to each data point
           opacitycal (bool): if False, time-dependent gaussian errors are added to station opacities
           ampcal (bool): if False, time-dependent gaussian errors are added to complex station gains
           phasecal (bool): if False, time-dependent station-based random phases are added to complex station gains
           stabilize_scan_phase (bool): if True, random phase errors are constant over scans
           stabilize_scan_amp (bool): if True, random amplitude errors are constant over scans
           taup (float): the fractional std. dev. of the random error on the opacities
           gainp (float): the fractional std. dev. of the random error on the gains
           gain_offset (float): the base gain offset at all sites, or a dict giving one gain offset per site
           seed : a seed for the random number generators, uses system time if false

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
    obsdata = copy.deepcopy(obs.data)

    sites = recarr_to_ndarr(obsdata[['t1','t2']],'U32')
    uv = recarr_to_ndarr(obsdata[['u','v']],'f8')
    taus = np.abs(recarr_to_ndarr(obsdata[['tau1','tau2']],'f8'))
    elevs = recarr_to_ndarr(obs.unpack(['el1','el2'],ang_unit='deg'),'f8')
    times = obsdata['time']
    tint = obsdata['tint']
    vis1 = obsdata[obs.poldict['vis1']]
    vis2 = obsdata[obs.poldict['vis2']]
    vis3 = obsdata[obs.poldict['vis3']]
    vis4 = obsdata[obs.poldict['vis4']]

    times_stable_phase = times.copy()
    times_stable_amp = times.copy()
    times_stable = times.copy()

    if stabilize_scan_phase==True or stabilize_scan_amp==True:
        scans = obs.scans
        if np.all(scans) == None or len(scans) == 0:
            print('ADDING SCANS')
            obs_scans = obs.copy()
            obs_scans.add_scans()
            scans = obs_scans.scans
        for j in range(len(times_stable)):
            for scan in scans:
                if scan[0] <= times_stable[j] and scan[1] >= times_stable[j]:
                    times_stable[j] = scan[0]
                    break

    if stabilize_scan_phase==True: times_stable_phase = times_stable.copy()
    if stabilize_scan_amp==True: times_stable_amp = times_stable.copy()

    # Recompute perfect sigmas from SEFDs
    bw = obs.bw
    if np.any(obs.tarr['sefdr'] <= 0):
        print("Warning!: in add_noise, some SEFDs are <= 0 -- not recomputing sigmas, which may result in double systematic noise")
        sigma_perf1 = obsdata[obs.poldict['sigma1']]
        sigma_perf2 = obsdata[obs.poldict['sigma2']]
        sigma_perf3 = obsdata[obs.poldict['sigma3']]
        sigma_perf4 = obsdata[obs.poldict['sigma4']]
    else:
        sig_rr = np.fromiter((blnoise(obs.tarr[obs.tkey[sites[i][0]]]['sefdr'], obs.tarr[obs.tkey[sites[i][1]]]['sefdr'], tint[i], bw)
                               for i in range(len(tint))),float)
        sig_ll = np.fromiter((blnoise(obs.tarr[obs.tkey[sites[i][0]]]['sefdr'], obs.tarr[obs.tkey[sites[i][1]]]['sefdl'], tint[i], bw)
                               for i in range(len(tint))),float)
        sig_rl = np.fromiter((blnoise(obs.tarr[obs.tkey[sites[i][0]]]['sefdr'], obs.tarr[obs.tkey[sites[i][1]]]['sefdl'], tint[i], bw)
                               for i in range(len(tint))),float)
        sig_lr = np.fromiter((blnoise(obs.tarr[obs.tkey[sites[i][0]]]['sefdl'], obs.tarr[obs.tkey[sites[i][1]]]['sefdr'], tint[i], bw)
                               for i in range(len(tint))),float)
        if obs.polrep=='stokes':
            sig_iv = 0.5*np.sqrt(sig_rr**2 + sig_ll**2)
            sig_qu = 0.5*np.sqrt(sig_rl**2 + sig_lr**2)
            sigma_perf1 = sig_iv
            sigma_perf2 = sig_qu
            sigma_perf3 = sig_qu
            sigma_perf4 = sig_iv
        elif obs.polrep=='circ':
            sigma_perf1 = sig_rr
            sigma_perf2 = sig_ll
            sigma_perf3 = sig_rl
            sigma_perf4 = sig_lr

    # Seed for random number generators
    if seed==False:
        seed=str(ttime.time())

    # Add gain and opacity fluctuations to the TRUE noise
    if not ampcal:
        # Amplitude gain
        if type(gain_offset) == dict:
            goff1 = np.fromiter((gain_offset[sites[i,0]] for i in range(len(times))), float)
            goff2 = np.fromiter((gain_offset[sites[i,1]] for i in range(len(times))), float)
        else:
            goff1 = np.fromiter((gain_offset for i in range(len(times))), float)
            goff2 = np.fromiter((gain_offset for i in range(len(times))), float)

        if type(gainp) is dict:
            gain_mult_1 = np.fromiter((gainp[sites[i,0]] for i in range(len(times))), float)
            gain_mult_2 = np.fromiter((gainp[sites[i,1]] for i in range(len(times))), float)
        else:
            gain_mult_1 = np.fromiter((gainp for i in range(len(times))), float)
            gain_mult_2 = np.fromiter((gainp for i in range(len(times))), float)

        gain1 = np.abs(np.fromiter(
                      ((1.0 + goff1[i] * np.abs(hashrandn(sites[i,0], 'gain', seed)))*(1.0 + gain_mult_1[i] * hashrandn(sites[i,0], 'gain', times_stable_amp[i], seed))
                       for i in range(len(times))
                      ), float))
        gain2 = np.abs(np.fromiter(
                      ((1.0 + goff2[i] * np.abs(hashrandn(sites[i,1], 'gain', seed)))*(1.0 + gain_mult_2[i] * hashrandn(sites[i,1], 'gain', times_stable_amp[i], seed))
                        for i in range(len(times))
                      ), float))
        gain_true = np.sqrt(gain1 * gain2)
    else:
        gain_true = 1

    if not opacitycal:

        # Use estimated opacity to compute the ESTIMATED noise
        tau_est = np.sqrt(np.exp(taus[:,0]/(EP+np.sin(elevs[:,0]*DEGREE)) + taus[:,1]/(EP+np.sin(elevs[:,1]*DEGREE))))

        # Opacity Errors
        tau1 = np.abs(np.fromiter((taus[i,0]* (1.0 + taup * hashrandn(sites[i,0], 'tau', times_stable_amp[i], seed)) for i in range(len(times))),float))
        tau2 = np.abs(np.fromiter((taus[i,1]* (1.0 + taup * hashrandn(sites[i,1], 'tau', times_stable_amp[i], seed)) for i in range(len(times))),float))

        # Correct noise RMS for opacity
        tau_true = np.sqrt(np.exp(tau1/(EP+np.sin(elevs[:,0]*DEGREE)) + tau2/(EP+np.sin(elevs[:,1]*DEGREE))))
    else:
        tau_true = tau_est = 1

    # Add the noise
    sigma_true1 = sigma_perf1
    sigma_true2 = sigma_perf2
    sigma_true3 = sigma_perf3
    sigma_true4 = sigma_perf4

    sigma_est1 = sigma_perf1 * gain_true * tau_est
    sigma_est2 = sigma_perf2 * gain_true * tau_est
    sigma_est3 = sigma_perf3 * gain_true * tau_est
    sigma_est4 = sigma_perf4 * gain_true * tau_est

    if add_th_noise:
        vis1 = (vis1 + cerror(sigma_true1))
        vis2 = (vis2 + cerror(sigma_true2))
        vis3 = (vis3 + cerror(sigma_true3))
        vis4 = (vis4 + cerror(sigma_true4))

    # Add the gain error to the true visibilities
    vis1 = vis1 * gain_true * tau_est / tau_true
    vis2 = vis2 * gain_true * tau_est / tau_true
    vis3 = vis3 * gain_true * tau_est / tau_true
    vis4 = vis4 * gain_true * tau_est / tau_true

    # Add random atmospheric phases
    if not phasecal:
        phase1 = np.fromiter((2 * np.pi * hashrand(sites[i,0], 'phase', times_stable_phase[i], seed) for i in range(len(times))),float)
        phase2 = np.fromiter((2 * np.pi * hashrand(sites[i,1], 'phase', times_stable_phase[i], seed) for i in range(len(times))),float)

        vis1 *= np.exp(1j * (phase2-phase1))
        vis2 *= np.exp(1j * (phase2-phase1))
        vis3 *= np.exp(1j * (phase2-phase1))
        vis4 *= np.exp(1j * (phase2-phase1))

    # Put the visibilities estimated errors back in the obsdata array
    obsdata[obs.poldict['vis1']] = vis1
    obsdata[obs.poldict['vis2']] = vis2
    obsdata[obs.poldict['vis3']] = vis3
    obsdata[obs.poldict['vis4']] = vis4

    obsdata[obs.poldict['sigma1']] = sigma_est1
    obsdata[obs.poldict['sigma2']] = sigma_est2
    obsdata[obs.poldict['sigma3']] = sigma_est3
    obsdata[obs.poldict['sigma4']] = sigma_est4

	# Return observation data
    return obsdata
