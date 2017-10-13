from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range

import astropy.time as at
import time as ttime
import scipy.ndimage as nd
import numpy as np
import datetime
import ephem
import astropy.coordinates as coords
import copy 

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

##################################################################################################
# Generate U-V Points
##################################################################################################

def make_uvpoints2(array, ra, dec, rf, bw, tint, tadv, tstart, tstop, mjd=MJD_DEFAULT,
                      tau=TAUDEF, elevmin=ELEV_LOW, elevmax=ELEV_HIGH, timetype='UTC'):
    """Generate u,v points and baseline errors for the array.
       Return an obsdata array with no visibilities.
       tstart and tstop are hrs in UTC
       tint and tadv are seconds.
       rf and bw are Hz
       ra is fractional hours
       dec is fractional degrees
       tau can be a single number or a dictionary giving one per site
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
                
                # Optical Depth
                if type(tau) == dict:
                    try:
                        tau1 = tau[i1]
                        tau2 = tau[i2]
                    except KeyError:
                        tau1 = tau2 = TAUDEF
                else:
                    tau1 = tau2 = tau

                # Noise on the correlations
                sig_rr = blnoise(array.tarr[i1]['sefdr'], array.tarr[i2]['sefdr'], tint, bw)
                sig_ll = blnoise(array.tarr[i1]['sefdl'], array.tarr[i2]['sefdl'], tint, bw)
                sig_rl = blnoise(array.tarr[i1]['sefdr'], array.tarr[i2]['sefdl'], tint, bw)
                sig_lr = blnoise(array.tarr[i1]['sefdl'], array.tarr[i2]['sefdr'], tint, bw)
                sig_iv = 0.5*np.sqrt(sig_rr**2 + sig_ll**2)
                sig_qu = 0.5*np.sqrt(sig_rl**2 + sig_lr**2)

                site1 = array.tarr[i1]['site']
                site2 = array.tarr[i2]['site']
                (timesout,uout,vout) = compute_uv_coordinates(array, site1, site2, times, mjd, ra, dec, rf, timetype=timetype, elevmin=elevmin, elevmax=elevmax)
                for k in range(len(timesout)):
                    # Append data to list
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



def make_uvpoints(array, ra, dec, rf, bw, tint, tadv, tstart, tstop, mjd=MJD_DEFAULT,
                      tau=TAUDEF, elevmin=ELEV_LOW, elevmax=ELEV_HIGH, timetype='UTC'):
    """Generate u,v points and baseline errors for the array.
       Return an obsdata array with no visibilities.
       tstart and tstop are hrs in UTC
       tint and tadv are seconds.
       rf and bw are Hz
       ra is fractional hours
       dec is fractional degrees
       tau can be a single number or a dictionary giving one per site
    """

    #if mjdtogmt(mjd)-tstart > 1e-9:
    #    raise Exception("Initial time is greater than given mjd!")

    # Set up coordinate system
    sourcevec = np.array([np.cos(dec*DEGREE), 0, np.sin(dec*DEGREE)])
    projU = np.cross(np.array([0,0,1]), sourcevec)
    projU = projU/np.linalg.norm(projU)
    projV = -np.cross(projU, sourcevec)

    # Set up time start and steps
    tstep = tadv/3600.0
    if tstop < tstart:
        tstop = tstop + 24.0;

    # Wavelength
    l = C/rf

    # Observing times
    # TODO: Scale - utc or tt?
    times = np.arange(tstart, tstop, tstep)
    if timetype == 'UTC':
        times_sidereal = utc_to_gmst(times, mjd)
    elif timetype == 'GMST':
        times_sidereal = times
    else:
        print("Time Type Not Recognized! Assuming UTC!")
        times_sidereal = utc_to_gmst(times, mjd)

    #print mjd
    #print np.min(times_sidereal), np.max(times_sidereal)
    # Generate uv points at all times
    outlist = []
    for k in range(len(times)):
        time = times[k]
        fracmjd = np.floor(mjd) + time/24.
        dto = (at.Time(fracmjd, format='mjd')).datetime
        time_sidereal = times_sidereal[k]
        theta = np.mod((time_sidereal - ra)*HOUR, 2*np.pi)
        blpairs = []

        for i1 in range(len(array.tarr)):
            for i2 in range(len(array.tarr)):
                coord1 = np.array((array.tarr[i1]['x'], array.tarr[i1]['y'], array.tarr[i1]['z']))
                coord2 = np.array((array.tarr[i2]['x'], array.tarr[i2]['y'], array.tarr[i2]['z']))
                site1 =  array.tarr[i1]['site']
                site2 =  array.tarr[i2]['site']

                # use spacecraft ephemeris so get position of site 1
                if np.all(coord1 == (0.,0.,0.)):
                    if timetype=='GMST':
                        raise Exception("Spacecraft ephemeris only work with UTC!")
                    sat = ephem.readtle(array.ephem[site1][0],array.ephem[site1][1],array.ephem[site1][2])
                    sat.compute(dto) # often complains if ephemeris out of date!
                    elev = sat.elevation
                    lat = sat.sublat / DEGREE
                    lon = sat.sublong / DEGREE
                    # pyephem doesn't use an ellipsoid earth model!
                    coord1 = coords.EarthLocation.from_geodetic(lon, lat, elev, ellipsoid=None)
                    coord1 = np.array((coord1.x.value, coord1.y.value, coord1.z.value))

                # use spacecraft ephemeris to get position of site 2
                if np.all(coord2 == (0.,0.,0.)):
                    if timetype=='GMST':
                        raise Exception("Spacecraft ephemeris only work with UTC!")
                    sat = ephem.readtle(array.ephem[site2][0],array.ephem[site2][1],array.ephem[site2][2])
                    sat.compute(dto) # often complains if ephemeris out of date!
                    elev = sat.elevation
                    lat = sat.sublat  / DEGREE
                    lon = sat.sublong / DEGREE
                    # pyephem doesn't use an ellipsoid earth model!
                    coord2 = coords.EarthLocation.from_geodetic(lon, lat, elev, ellipsoid=None)
                    coord2 = np.array((coord2.x.value, coord2.y.value, coord2.z.value))

                # rotate the station coordinates
                coord1 = earthrot(coord1, theta)
                coord2 = earthrot(coord2, theta)

                if (i1!=i2 and
                    i1 < i2 and # This is the right condition for uvfits save order
                    not ((i2, i1) in blpairs) and # This cuts out the conjugate baselines
                    elevcut(coord1, sourcevec, elevmin=elevmin, elevmax=elevmax)[0] and
                    elevcut(coord2, sourcevec, elevmin=elevmin, elevmax=elevmax)[0]
                   ):

                    # Optical Depth
                    if type(tau) == dict:
                        try:
                            tau1 = tau[i1]
                            tau2 = tau[i2]
                        except KeyError:
                            tau1 = tau2 = TAUDEF
                    else:
                        tau1 = tau2 = tau


                    # Noise on the correlations
                    sig_rr = blnoise(array.tarr[i1]['sefdr'], array.tarr[i2]['sefdr'], tint, bw)
                    sig_ll = blnoise(array.tarr[i1]['sefdl'], array.tarr[i2]['sefdl'], tint, bw)
                    sig_rl = blnoise(array.tarr[i1]['sefdr'], array.tarr[i2]['sefdl'], tint, bw)
                    sig_lr = blnoise(array.tarr[i1]['sefdl'], array.tarr[i2]['sefdr'], tint, bw)

                    # Append data to list
                    blpairs.append((i1,i2))
                    outlist.append(np.array((
                              time,
                              tint, # Integration
                              array.tarr[i1]['site'], # Station 1
                              array.tarr[i2]['site'], # Station 2
                              tau1, # Station 1 optical depth
                              tau2, # Station 1 optical depth
                              np.dot((coord1 - coord2)/l, projU), # u (lambda)
                              np.dot((coord1 - coord2)/l, projV), # v (lambda)
                              0.0, # I Visibility (Jy)
                              0.0, # Q Visibility
                              0.0, # U Visibility
                              0.0, # V Visibilities
                              0.5*np.sqrt(sig_rr**2 + sig_ll**2), # I Sigma (Jy)
                              0.5*np.sqrt(sig_rl**2 + sig_lr**2), # Q Sigma
                              0.5*np.sqrt(sig_rl**2 + sig_lr**2), # U Sigma
                              0.5*np.sqrt(sig_rr**2 + sig_ll**2)  # V Sigma
                            ), dtype=DTPOL
                            ))

    obsarr = np.array(outlist)

    if not len(obsarr):
        raise Exception("No mutual visibilities in the specified time range!")

    return obsarr


##################################################################################################
# Observe w/o noise
##################################################################################################

def observe_image_nonoise(im, obs, sgrscat=False, ttype="direct", fft_pad_factor=1):
    """Observe the image on the same baselines as an existing observation object
       if sgrscat==True, the visibilites will be blurred by the Sgr A* scattering kernel
       Does NOT add noise
    """

    # Check for agreement in coordinates and frequency
    if (im.ra!= obs.ra) or (im.dec != obs.dec):
        raise Exception("Image coordinates are not the same as observtion coordinates!")
    if (im.rf != obs.rf):
        raise Exception("Image frequency is not the same as observation frequency!")

    if ttype=='direct' or ttype=='fast':
        print("Producing clean visibilities from image with " + ttype + " FT . . . ")
    else:
        raise Exception("ttype=%s, options for ttype are 'direct' and 'fast'"%ttype)

    # Copy data to be safe 
    obsdata = obs.copy().data

    # Extract uv data
    uv = obsdata[['u','v']].view(('f8',2))

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

    #visibilities from FFT
    if ttype=="fast":

        # Pad image
        #npad = int(np.ceil(pad_frac*1./(im.psize*umin)))
        npad = fft_pad_factor * np.max((im.xdim, im.ydim))
        npad = power_of_two(npad) #TODO good in all cases??

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
        # TODO can we get rid of the fftshifts?
        vis_im = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(imarr)))

        # Sample the visibilities
        # default is cubic spline interpolation
        visre = nd.map_coordinates(np.real(vis_im), uv2)
        visim = nd.map_coordinates(np.imag(vis_im), uv2)
        vis = visre + 1j*visim

        # Extra phase to match centroid convention -- right??
        #phase = np.exp(-1j*np.pi*im.psize*(uv[:,0] + uv[:,1]))
        phase = np.exp(-1j*np.pi*im.psize*((1+im.xdim%2)*uv[:,0] + (1+im.ydim%2)*uv[:,1])) 
        vis = vis * phase

        # Multiply by the pulse function
        # TODO make faster?
        pulsefac = np.array([im.pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], im.psize, dom="F") for uvpt in uv])
        vis = vis * pulsefac

        # FT of polarimetric quantities
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

    #visibilities from DFT
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
    obsdata['vis'] = vis
    obsdata['qvis'] = qvis
    obsdata['uvis'] = uvis
    obsdata['vvis'] = vvis

    return obsdata

def observe_movie_nonoise(mov, obs, sgrscat=False, ttype="direct", pad_frac=0.5, repeat=False):
        """Observe the movie on the same baselines as an existing observation object
           if sgrscat==True, the visibilites will be blurred by the Sgr A* scattering kernel
           Does NOT add noise
        """

        # Check for agreement in coordinates and frequency
        if (mov.ra!= obs.ra) or (mov.dec != obs.dec):
            raise Exception("Image coordinates are not the same as observation coordinates!")
            if (mov.rf != obs.rf):
                raise Exception("Image frequency is not the same as observation frequency!")

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
            uv = obsdata[['u','v']].view(('f8',2))
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

            #visibilities from FFT
            if ttype=="fast":

                # Pad image
                npad = int(np.ceil(pad_frac*1./(mov.psize*umin)))
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

                    uvisre = nd.map_coordinates(np.real(uvis_im), uv2)
                    uvisim = nd.map_coordinates(np.imag(uvis_im), uv2)
                    uvis = phase*(uvisre + 1j*uvisim)

                if len(mov.vframes):
                    varr = mov.frames[n].reshape(mov.ydim, mov.xdim)
                    varr = np.pad(varr, ((padvalx1,padvalx2),(padvaly1,padvaly2)), 'constant', constant_values=0.0)

                    vvis_im = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(varr)))

                    vvisre = nd.map_coordinates(np.real(vvis_im), uv2)
                    vvisim = nd.map_coordinates(np.imag(vvis_im), uv2)
                    vvis = phase*(vvisre + 1j*qvisim)

            #visibilities from DFT
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

def make_jones(obs, ampcal=True, opacitycal=True, phasecal=True, dcal=True, frcal=True, 
               gainp=GAINPDEF, taup=GAINPDEF, gain_offset=GAINPDEF, dtermp=DTERMPDEF, dtermp_resid=DTERMPDEF_RESID):
    """Compute ALL Jones Matrices for a list of times (non repeating), with gain and dterm errors.
       ra and dec should be in hours / degrees
       Will return a nested dictionary of matrices indexed by the site, then by the time

       gain_offset can be optionally set as a dictionary that specifies the standard deviation 
       of the time independtent offset drawn for each telescope site. If it is a single value than 
       the standard deviation of this random  variable is the same for all sites. 
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

        # Amplitude gain

        gainR = gainL = np.ones(len(times))
        if not ampcal:
            # Amplitude gain
            if type(gain_offset) == dict:
                goff = gain_offset[site]
            else:
                goff=gain_offset

            gainR = np.sqrt(np.abs(np.array([(1.0 +  goff*hashrandn(site,'gain',str(goff), tproc))*(1.0 + gainp * hashrandn(site, 'gain', time, str(gainp), tproc))
                                     for time in times]))) 
            gainL = np.sqrt(np.abs(np.array([(1.0 +  goff*hashrandn(site,'gain',str(goff), tproc))*(1.0 + gainp * hashrandn(site, 'gain', time, str(gainp), tproc))
                                     for time in times])))

        # Opacity attenuation of amplitude gain
        if not opacitycal:
            taus = np.abs(np.array([taudict[site][j] * (1.0 + taup * hashrandn(site, 'tau', times[j], tproc)) for j in range(len(times))]))
            atten = np.exp(-taus/(EP + 2.0*np.sin(el_angles)))

            gainR = gainR * atten
            gainL = gainL * atten

        # Atmospheric Phase
        if not phasecal:
            phase = np.array([2 * np.pi * hashrand(site, 'phase', time, tproc) for time in times])
            gainR = gainR * np.exp(1j*phase)
            gainL = gainL * np.exp(1j*phase)

        # D Term errors
        dR = dL = 0.0
        if not dcal:
            dR = tarr[i]['dr']
            if dR == 0.0: dR = dtermp * (hashrandn(site, 'drreal', tproc) + 1j * hashrandn(site, 'drim', tproc))
            dL = tarr[i]['dl']
            if dL == 0.0: dL = dtermp * (hashrandn(site, 'dlreal', tproc) + 1j * hashrandn(site, 'dlim', tproc))
            dR = dR + dtermp_resid * (hashrandn(site, 'drreal_resid', tproc) + 1j * hashrandn(site, 'drim_resid', tproc))
            dL = dL + dtermp_resid * (hashrandn(site, 'dlreal_resid', tproc) + 1j * hashrandn(site, 'dlim_resid', tproc))

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
                                for j in range(len(times))}

        out[site] = j_matrices

    return out

def make_jones_inverse(obs, ampcal=True, phasecal=True, opacitycal=True, dcal=True, frcal=True):
    """Compute all Inverse Jones Matrices for a list of times (non repeating), with NO gain and dterm errors.
       ra and dec should be in hours / degrees
       Will return a dictionary of matrices for each time, indexed by the site, with the matrices in time order

       for each telescope site.
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
                                     ]) for j in range(len(times))}

        out[site] = j_matrices_inv

    return out

def add_jones_and_noise(obs, add_th_noise=True, opacitycal=True, ampcal=True, phasecal=True, dcal=True, frcal=True, 
                        gainp=GAINPDEF, gain_offset=GAINPDEF, taup=GAINPDEF, dtermp=DTERMPDEF, dtermp_resid=DTERMPDEF_RESID):
    """Corrupt visibilities in obs with jones matrices and add thermal noise
    """

    print("Applying Jones Matrices to data . . . ")

    # Build Jones Matrices
    jm_dict = make_jones(obs,
                         ampcal=ampcal, opacitycal=opacitycal, phasecal=phasecal,dcal=dcal,frcal=frcal,
                         gainp=gainp, taup=taup, gain_offset=gain_offset, dtermp=dtermp, dtermp_resid=dtermp_resid)
    # Unpack Data
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
    sig_rr = np.array([blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdr'], obs.tarr[obs.tkey[t2[i]]]['sefdr'], tints[i], obs.bw) for i in range(len(rr))])
    sig_ll = np.array([blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdl'], obs.tarr[obs.tkey[t2[i]]]['sefdl'], tints[i], obs.bw) for i in range(len(ll))])
    sig_rl = np.array([blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdr'], obs.tarr[obs.tkey[t2[i]]]['sefdl'], tints[i], obs.bw) for i in range(len(rl))])
    sig_lr = np.array([blnoise(obs.tarr[obs.tkey[t1[i]]]['sefdl'], obs.tarr[obs.tkey[t2[i]]]['sefdr'], tints[i], obs.bw) for i in range(len(lr))])

    #print "------------------------------------------------------------------------------------------------------------------------"
    if not opacitycal:
        print("   Applying opacity attenuation: opacitycal-->False")
    if not ampcal:
        print("   Applying gain corruption: gaincal-->False")
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

def apply_jones_inverse(obs, ampcal=True, opacitycal=True, phasecal=True, dcal=True, frcal=True):
    """Corrupt visibilities in obs with jones matrices and add thermal noise
    """

    print("Applying a priori calibration with estimated Jones matrices . . . ")

    # Build Inverse Jones Matrices
    jm_dict = make_jones_inverse(obs,
                                 ampcal=ampcal, phasecal=phasecal, opacitycal=opacitycal,
                                 dcal=dcal, frcal=frcal)
    # Unpack Data
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
    #!AC should we instead get them from the file?
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

        # !AC TODO is this correct?
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
def add_noise(obs, ampcal=True, opacitycal=True, phasecal=True, add_th_noise=True, gainp=GAINPDEF, taup=GAINPDEF, gain_offset=GAINPDEF, seed=False):
    """Re-compute sigmas from SEFDS and add noise with gain & phase errors
       Returns signals & noises scaled by estimated gains, including opacity attenuation.
       Be very careful using outside of Image.observe!

       gain_offset can be optionally set as a dictionary that specifies the standard deviation 
       of the constant offset for each telescope site. If it is a single value than it is the standard deviation
       for all sites. 
    """

    print("Adding gain + phase errors to data and applying a priori calibration . . . ")

    #print "------------------------------------------------------------------------------------------------------------------------"
    opacitycalout=True #With old noise function, output is always calibrated to estimated opacity
                       #!AC TODO Could change this
    if not opacitycal:
        print("   Applying opacity attenuation AND estimated opacity corrections: opacitycal-->True")
    if not ampcal:
        print("   Applying gain corruption: gaincal-->False")
    if not phasecal:
        print("   Applying atmospheric phase corruption: phasecal-->False")
    if add_th_noise:
        print("Adding thermal noise to data . . . ")
    #print "------------------------------------------------------------------------------------------------------------------------"

    # Get data
    obsdata = copy.deepcopy(obs.data)
    sites = obsdata[['t1','t2']].view(('a32',2))
    time = obsdata[['time']].view(('f8',1))
    tint = obsdata[['tint']].view(('f8',1))
    uv = obsdata[['u','v']].view(('f8',2))
    vis = obsdata[['vis']].view(('c16',1))
    qvis = obsdata[['qvis']].view(('c16',1))
    uvis = obsdata[['uvis']].view(('c16',1))
    vvis = obsdata[['vvis']].view(('c16',1))

    taus = np.abs(obsdata[['tau1','tau2']].view(('f8',2)))
    elevs = obs.unpack(['el1','el2'], ang_unit='deg').view(('f8',2))
    bw = obs.bw

    # Recompute perfect sigmas from SEFDs
    # Multiply 1/sqrt(2) for sum of polarizations
    sigma_perf = np.array([blnoise(obs.tarr[obs.tkey[sites[i][0]]]['sefdr'], obs.tarr[obs.tkey[sites[i][1]]]['sefdr'], tint[i], bw)/np.sqrt(2.0)
                            for i in range(len(tint))])


    # Add gain and opacity fluctuations to the TRUE noise
    if seed==False:
        seed=str(ttime.time())
    
    if not ampcal:
        print ("GAIN")
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
    #ANDREW TODO -- sigma perf here??
    #tau_true = tau_est
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

    # This function doesn't use different visibility sigmas!
    obsdata['qsigma'] = obsdata['usigma'] = obsdata['vsigma'] = sigma_est


	# Return observation data
    return obsdata


