# obs_helpers.py
# helper functions for simulating and manipulating observations 
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
from builtins import str
from builtins import map
from builtins import range
import ephem

import astropy.time as at
import astropy.coordinates as coords
import numpy as np
import scipy.special as ss

from ehtim.const_def import *

##################################################################################################
# Other Functions
##################################################################################################

def compute_uv_coordinates(array, site1, site2, time, mjd, ra, dec, rf, timetype='UTC', elevmin=ELEV_LOW,  elevmax=ELEV_HIGH, fix_theta_GMST = False):

    if not isinstance(time, np.ndarray): time = np.array([time]).flatten()
    if not isinstance(site1, np.ndarray): site1 = np.array([site1]).flatten()
    if not isinstance(site2, np.ndarray): site2 = np.array([site2]).flatten()

    if len(site1) == len(site2) == 1:
        site1 = np.array([site1[0] for i in range(len(time))])
        site2 = np.array([site2[0] for i in range(len(time))])
    elif not (len(site1) == len(site2) == len(time)):
        raise Exception("site1, site2, and time not the same dimension in compute_uv_coordinates!") 

    # Source vector
    sourcevec = np.array([np.cos(dec*DEGREE), 0, np.sin(dec*DEGREE)])
    projU = np.cross(np.array([0,0,1]), sourcevec)
    projU = projU/np.linalg.norm(projU)
    projV = -np.cross(projU, sourcevec)

    # Wavelength
    l = C/rf

    if timetype=='GMST':
        time_sidereal = time
        time_utc = gmst_to_utc(time, mjd)
    elif timetype=='UTC':
        time_sidereal = utc_to_gmst(time, mjd)
        time_utc = time
    else: raise Exception("timetype must be UTC or GMST!")

    fracmjd = np.floor(mjd) + time/24.
    dto = (at.Time(fracmjd, format='mjd')).datetime
    theta = np.mod((time_sidereal - ra)*HOUR, 2*np.pi)
    if type(fix_theta_GMST) != bool:
        theta = np.mod((fix_theta_GMST - ra)*HOUR, 2*np.pi)

    i1 = np.array([array.tkey[site] for site in site1])
    i2 = np.array([array.tkey[site] for site in site2])

    coord1 = np.vstack((array.tarr[i1]['x'], array.tarr[i1]['y'], array.tarr[i1]['z'])).T
    coord2 = np.vstack((array.tarr[i2]['x'], array.tarr[i2]['y'], array.tarr[i2]['z'])).T

    # TODO SPEED UP!??
    # use spacecraft ephemeris to get position of site 1
    spacemask1 = [np.all(coord == (0.,0.,0.)) for coord in coord1]
    if np.any(spacemask1):
        if timetype=='GMST':
            raise Exception("Spacecraft ephemeris only work with UTC!")

        site1space_list = site2[spacemask1]
        site1space_dtolist = dto[spacemask1]
        coord1space = []
        for k in range(len(site1space_list)):
            site1space = site1space_list[k]
            dto_now = site1space_dtolist[k]
            sat = ephem.readtle(array.ephem[site1space][0],array.ephem[site1space][1],array.ephem[site1space][2])
            sat.compute(dto_now) # often complains if ephemeris out of date!
            elev = sat.elevation
            lat = sat.sublat / DEGREE
            lon = sat.sublong / DEGREE
            # pyephem doesn't use an ellipsoid earth model!
            c1 = coords.EarthLocation.from_geodetic(lon, lat, elev, ellipsoid=None)
            c1 = np.array((c1.x.value, c1.y.value, c1.z.value))
            coord1space.append(c1)
        coord1space = np.array(coord1space)
        coord1[spacemask1] = coord1space

    spacemask2 = [np.all(coord == (0.,0.,0.)) for coord in coord2]
    if np.any(spacemask2):
        if timetype=='GMST':
            raise Exception("Spacecraft ephemeris only work with UTC!")

        site2space_list = site2[spacemask2]
        site2space_dtolist = dto[spacemask2]
        coord2space = []
        for k in range(len(site2space_list)):
            site2space = site2space_list[k]
            dto_now = site2space_dtolist[k]
            sat = ephem.readtle(array.ephem[site2space][0],array.ephem[site2space][1],array.ephem[site2space][2])
            sat.compute(dto_now) # often complains if ephemeris out of date!
            elev = sat.elevation
            lat = sat.sublat / DEGREE
            lon = sat.sublong / DEGREE
            # pyephem doesn't use an ellipsoid earth model!
            c2 = coords.EarthLocation.from_geodetic(lon, lat, elev, ellipsoid=None)
            c2 = np.array((c2.x.value, c2.y.value, c2.z.value))
            coord2space.append(c2)
        coord2space = np.array(coord2space)
        coord2[spacemask2] = coord2space

    # rotate the station coordinates with the earth
    coord1 = earthrot(coord1, theta)
    coord2 = earthrot(coord2, theta)

    # u,v coordinates
    u = np.dot((coord1 - coord2)/l, projU) # u (lambda)
    v = np.dot((coord1 - coord2)/l, projV) # v (lambda)

    # mask out below elevation cut
    mask = (elevcut(coord1, sourcevec, elevmin=elevmin, elevmax=elevmax) *
            elevcut(coord2, sourcevec, elevmin=elevmin, elevmax=elevmax))

    time = time[mask]
    u = u[mask]
    v = v[mask]
    
    # return times and uv points where we have  data
    return (time, u, v)

def make_bispectrum(l1, l2, l3,vtype):
    """make a list of bispectra and errors
       l1,l2,l3 are full datatables of visibility entries
       vtype is visibility types
    """
    # Choose the appropriate polarization and compute the bs and err
    if vtype in ["vis", "qvis", "uvis","vvis"]:
        if vtype=='vis':  sigmatype='sigma'
        if vtype=='qvis': sigmatype='qsigma'
        if vtype=='uvis': sigmatype='usigma'
        if vtype=='vvis': sigmatype='vsigma'

        p1 = l1[vtype]
        p2 = l2[vtype]
        p3 = l3[vtype]

        var1 = l1[sigmatype]**2
        var2 = l2[sigmatype]**2
        var3 = l3[sigmatype]**2

    elif vtype == "rrvis":
        p1 = l1['vis'] + l1['vvis']
        p2 = l2['vis'] + l2['vvis']
        p3 = l3['vis'] + l3['vvis']
        
        var1 = l1['sigma']**2 + l1['vsigma']**2
        var2 = l2['sigma']**2 + l2['vsigma']**2
        var3 = l3['sigma']**2 + l3['vsigma']**2

    elif vtype == "llvis":
        p1 = l1['vis'] - l1['vvis']
        p2 = l2['vis'] - l2['vvis']
        p3 = l3['vis'] - l3['vvis']
        
        var1 = l1['sigma']**2 + l1['vsigma']**2
        var2 = l2['sigma']**2 + l2['vsigma']**2
        var3 = l3['sigma']**2 + l3['vsigma']**2

    elif vtype == "lrvis":
        p1 = l1['qvis'] - 1j*l1['uvis']
        p2 = l2['qvis'] - 1j*l2['uvis']
        p3 = l3['qvis'] - 1j*l3['uvis']
        
        var1 = l1['qsigma']**2 + l1['usigma']**2
        var2 = l2['qsigma']**2 + l2['usigma']**2
        var3 = l3['qsigma']**2 + l3['usigma']**2

    elif vtype in ["pvis","rlvis"]:
        p1 = l1['qvis'] + 1j*l2['uvis']
        p2 = l2['qvis'] + 1j*l2['uvis']
        p3 = l3['qvis'] + 1j*l3['uvis']
        bi = p1 * p2 * p3

        var1 = l1['qsigma']**2 + l1['usigma']**2
        var2 = l2['qsigma']**2 + l2['usigma']**2
        var3 = l3['qsigma']**2 + l3['usigma']**2

    bi = p1*p2*p3
    bisig = np.abs(bi) * np.sqrt(var1/np.abs(p1)**2 +
                                 var2/np.abs(p2)**2 +
                                 var3/np.abs(p3)**2)
    # Katie's 2nd + 3rd order corrections - see CHIRP supplement
    #bisig = np.sqrt(bisig**2 + var1*var2*np.abs(p3)**2 +
    #                           var1*var3*np.abs(p2)**2 +
    #                           var2*var3*np.abs(p1)**2 +
    #                           var1*var2*var3)
    return (bi, bisig)

def make_closure_amplitude(red1, red2, blue1, blue2, vtype, ctype='camp', debias=True, debias_type='ExactLog'):
    """make a list of closure amplitudes and errors
       red1 and red2 are full datatables of numerator entries
       blue1 and blue2 are full datatables denominator entries
       vtype is the  visibility type
       we always debias the individual amplitudes
       debias controls if we debias the closure amplitude at the end
       DebiasType controls the type of debisaing, 'ExactLog' means 
       exact debiasing in log space, it will turn off any debiasing in 'amp_debias',
       and apply debiasing only to closure quantities 
    """

    DebiasType = debias_type

    if not (ctype in ['camp', 'logcamp']):
        raise Exception("closure amplitude type must be 'camp' or 'logcamp'!")

    if vtype in ["vis", "qvis", "uvis", "vvis"]:
        if vtype=='vis':  sigmatype='sigma'
        if vtype=='qvis': sigmatype='qsigma'
        if vtype=='uvis': sigmatype='usigma'
        if vtype=='vvis': sigmatype='vsigma'

        sig1 = blue1[sigmatype]
        sig2 = blue2[sigmatype]
        sig3 = red1[sigmatype]
        sig4 = red2[sigmatype]

        p1 = amp_debias(blue1[vtype], sig1,DebiasType)
        p2 = amp_debias(blue2[vtype], sig2,DebiasType)
        p3 = amp_debias(red1[vtype], sig3,DebiasType)
        p4 = amp_debias(red2[vtype], sig4,DebiasType)

    elif vtype == "rrvis":
        sig1 = np.sqrt(blue1['sigma']**2 + blue1['vsigma']**2)
        sig2 = np.sqrt(blue2['sigma']**2 + blue2['vsigma']**2)
        sig3 = np.sqrt(red1['sigma']**2 + red1['vsigma']**2)
        sig4 = np.sqrt(red2['sigma']**2 + red2['vsigma']**2)

        p1 = amp_debias(blue1['vis'] + blue1['vvis'], sig1,DebiasType)
        p2 = amp_debias(blue2['vis'] + blue2['vvis'], sig2,DebiasType)
        p3 = amp_debias(red1['vis'] + red1['vvis'], sig3,DebiasType)
        p4 = amp_debias(red2['vis'] + red2['vvis'], sig4,DebiasType)

    elif vtype == "llvis":
        sig1 = np.sqrt(blue1['sigma']**2 + blue1['vsigma']**2)
        sig2 = np.sqrt(blue2['sigma']**2 + blue2['vsigma']**2)
        sig3 = np.sqrt(red1['sigma']**2 + red1['vsigma']**2)
        sig4 = np.sqrt(red2['sigma']**2 + red2['vsigma']**2)

        p1 = amp_debias(blue1['vis'] - blue1['vvis'], sig1,DebiasType)
        p2 = amp_debias(blue2['vis'] - blue2['vvis'], sig2,DebiasType)
        p3 = amp_debias(red1['vis'] - red1['vvis'], sig3,DebiasType)
        p4 = amp_debias(red2['vis'] - red2['vvis'], sig4,DebiasType)

    elif vtype == "lrvis":
        sig1 = np.sqrt(blue1['qsigma']**2 + blue1['usigma']**2,DebiasType)
        sig2 = np.sqrt(blue2['qsigma']**2 + blue2['usigma']**2,DebiasType)
        sig3 = np.sqrt(red1['qsigma']**2 + red1['usigma']**2,DebiasType)
        sig4 = np.sqrt(red2['qsigma']**2 + red2['usigma']**2,DebiasType)

        p1 = amp_debias(blue1['qvis'] - 1j*blue1['uvis'], sig1)
        p2 = amp_debias(blue2['qvis'] - 1j*blue2['uvis'], sig2)
        p3 = amp_debias(red1['qvis'] - 1j*red1['uvis'], sig3)
        p4 = amp_debias(red2['qvis'] - 1j*red2['uvis'], sig4)

    elif vtype in ["pvis","rlvis"]:
        sig1 = np.sqrt(blue1['qsigma']**2 + blue1['usigma']**2,DebiasType)
        sig2 = np.sqrt(blue2['qsigma']**2 + blue2['usigma']**2,DebiasType)
        sig3 = np.sqrt(red1['qsigma']**2 + red1['usigma']**2,DebiasType)
        sig4 = np.sqrt(red2['qsigma']**2 + red2['usigma']**2,DebiasType)

        p1 = amp_debias(blue1['qvis'] + 1j*blue1['uvis'], sig1,DebiasType)
        p2 = amp_debias(blue2['qvis'] + 1j*blue2['uvis'], sig2,DebiasType)
        p3 = amp_debias(red1['qvis'] + 1j*red1['uvis'], sig3,DebiasType)
        p4 = amp_debias(red2['qvis'] + 1j*red2['uvis'], sig4,DebiasType)

    snr1 = p1/sig1
    snr2 = p2/sig2
    snr3 = p3/sig3
    snr4 = p4/sig4

    if ctype=='camp':
        camp = np.abs((p1*p2)/(p3*p4))
        camperr = camp * np.sqrt(1./(snr1**2) + 1./(snr2**2) + 1./(snr3**2) + 1./(snr4**2))

        # Debias
        if debias:
            if DebiasType=='ExactLog':
                snr1 = get_snr(snr1)
                snr2 = get_snr(snr2)
                snr3 = get_snr(snr3)
                snr4 = get_snr(snr4)
                camp = camp_debias(camp, snr3, snr4,snr1,snr2,'ExactLog')
            else:
                camp = camp_debias(camp, snr3, snr4)

    elif ctype=='logcamp':
        camp = np.log(np.abs(p1)) + np.log(np.abs(p2)) - np.log(np.abs(p3)) - np.log(np.abs(p4))
        camperr = np.sqrt(1./(snr1**2) + 1./(snr2**2) + 1./(snr3**2) + 1./(snr4**2))

        # Debias
        if debias:
            if DebiasType=='ExactLog':
                snr1 = get_snr(snr1)
                snr2 = get_snr(snr2)
                snr3 = get_snr(snr3)
                snr4 = get_snr(snr4)
                camp = logcamp_debias(camp, snr1, snr2,snr3,snr4,'ExactLog')
            else:    
                camp = logcamp_debias(camp, snr1, snr2, snr3, snr4)

    return (camp, camperr)

#MW---OCT---2017
def get_snr_help(Esnr):
    """estimates snr given a single biased snr measurement
    """
    if Esnr**2 >= 2.0: 
        return np.sqrt(Esnr**2 - 1.0)
    else:
        return 1.0

def get_snr(Esnr):
    """"applies get_snr_help on vector
    """
    if type(Esnr) == float or type(Esnr)==np.float64:
        return get_snr_help(Esnr)
    else:
        return np.asarray(map(get_snr_help,Esnr))

def log_debias(snr0):
    """debias log snr
    """
    snr0 = np.asarray(snr0)
    return -ss.expi(-snr0**2/2.)/2.


def amp_debias(amp, sigma, DebiasType='old'):
    """Return debiased visibility amplitudes
    """

    if DebiasType=='ExactLog':
        #don't debias at all in this case, all debiasing will happen later for closure quantities
        #snr0 = amp/sigma
        #return amp*np.exp(-log_debias(snr0))
        return amp
    
    else:
        deb2 = np.abs(amp)**2 - np.abs(sigma)**2
        if type(deb2) == float or type(deb2)==np.float64:
            if deb2 < 0.0: return np.abs(amp)
            else: return np.sqrt(deb2)
        else:
            lowsnr = deb2 < 0.0
            deb2[lowsnr] = np.abs(amp[lowsnr])**2
            return np.sqrt(deb2)

def camp_debias(camp, snr3, snr4,snr1=1e5,snr2=1e5,DebiasType='old'):
    """Debias closure amplitudes
       snr3 and snr4 are snr of visibility amplitudes # 3 and 4.
    """
    if DebiasType=='ExactLog':
        camp_debias = camp*np.exp( - log_debias(snr1) - log_debias(snr2) + log_debias(snr3) + log_debias(snr4) )
    else:
        camp_debias = camp / (1 + 1./(snr3**2) + 1./(snr4**2))
    return camp_debias

def logcamp_debias(log_camp, snr1, snr2, snr3, snr4,DebiasType='old'):
    """Debias log closure amplitudes
       The snrs are the snr of visibility amplitudes
    """
    if DebiasType=='ExactLog':
        log_camp_debias = log_camp  - log_debias(snr1) - log_debias(snr2) + log_debias(snr3) + log_debias(snr4)
    else:
        log_camp_debias = log_camp + 0.5*(1./(snr1**2) + 1./(snr2**2) - 1./(snr3**2) - 1./(snr4**2))
    return log_camp_debias

def gauss_uv(u, v, flux, beamparams, x=0., y=0.):
    """Return the value of the Gaussian FT with
       beamparams is [FWHMmaj, FWHMmin, theta, x, y], all in radian
       theta is the orientation angle measured E of N
    """

    sigma_maj = beamparams[0]/(2*np.sqrt(2*np.log(2)))
    sigma_min = beamparams[1]/(2*np.sqrt(2*np.log(2)))
    theta = -beamparams[2] # theta needs to be negative in this convention!
    #try:
    #	x=beamparams[3]
    #	y=beamparams[4]
    #except IndexError:
    #	x=y=0.0

    # Covariance matrix
    a = (sigma_min * np.cos(theta))**2 + (sigma_maj*np.sin(theta))**2
    b = (sigma_maj * np.cos(theta))**2 + (sigma_min*np.sin(theta))**2
    c = (sigma_min**2 - sigma_maj**2) * np.cos(theta) * np.sin(theta)
    m = np.array([[a, c], [c, b]])

    uv = np.array([[u[i],v[i]] for i in range(len(u))])
    x2 = np.array([np.dot(uvi,np.dot(m,uvi)) for uvi in uv])
    #x2 = np.dot(uv, np.dot(m, uv.T))
    g = np.exp(-2 * np.pi**2 * x2)
    p = np.exp(-2j * np.pi * (u*x + v*y))

    return flux * g * p

def sgra_kernel_uv(rf, u, v):
    """Return the value of the Sgr A* scattering kernel at a given u,v pt (in lambda),
       at a given frequency rf (in Hz).
       Values from Bower et al.
    """

    lcm = (C/rf) * 100 # in cm
    sigma_maj = FWHM_MAJ * (lcm**2) / (2*np.sqrt(2*np.log(2))) * RADPERUAS
    sigma_min = FWHM_MIN * (lcm**2) / (2*np.sqrt(2*np.log(2))) * RADPERUAS
    theta = -POS_ANG * DEGREE # theta needs to be negative in this convention!

    #bp = [fwhm_maj, fwhm_min, theta]
    #g = gauss_uv(u, v, 1., bp, x=0., y=0.)

    # Covariance matrix
    a = (sigma_min * np.cos(theta))**2 + (sigma_maj*np.sin(theta))**2
    b = (sigma_maj * np.cos(theta))**2 + (sigma_min*np.sin(theta))**2
    c = (sigma_min**2 - sigma_maj**2) * np.cos(theta) * np.sin(theta)
    m = np.array([[a, c], [c, b]])
    uv = np.array([u,v])

    x2 = np.dot(uv, np.dot(m, uv))
    g = np.exp(-2 * np.pi**2 * x2)

    return g

def sgra_kernel_params(rf):
    """Return elliptical gaussian parameters in radian for the Sgr A* scattering ellipse at a given frequency
       Values from Bower et al.
    """

    lcm = (C/rf) * 100 # in cm
    fwhm_maj_rf = FWHM_MAJ * (lcm**2)  * RADPERUAS
    fwhm_min_rf = FWHM_MIN * (lcm**2)  * RADPERUAS
    theta = POS_ANG * DEGREE

    return np.array([fwhm_maj_rf, fwhm_min_rf, theta])


def blnoise(sefd1, sefd2, tint, bw):
    """Determine the standard deviation of Gaussian thermal noise on a baseline
       This is the noise on the rr/ll/rl/lr correlation, not the stokes parameter
       2-bit quantization is responsible for the 0.88 factor
    """


    noise = np.sqrt(sefd1*sefd2/(2*bw*tint))/0.88
    #noise = np.sqrt(sefd1*sefd2/(bw*tint))/0.88

    return noise

def merr(sigma, qsigma, usigma, I, m):
    """Return the error in mbreve real and imaginary parts"""

    err = np.sqrt((qsigma**2 + usigma**2 + (sigma*np.abs(m))**2)/(np.abs(I) ** 2))
    # old formula assumes all sigmas the same
    #err = sigma * np.sqrt((2 + np.abs(m)**2)/ (np.abs(I) ** 2))
    return err

def cerror(sigma):
    """Return a complex number drawn from a circular complex Gaussian of zero mean
    """
    return np.random.normal(loc=0,scale=sigma) + 1j*np.random.normal(loc=0,scale=sigma)

def hashrandn(*args):
    """set the seed according to a collection of arguments and return random gaussian var
    """
    np.random.seed(hash(",".join(map(repr,args))) % 4294967295)
    return np.random.randn()

def hashrand(*args):
    """set the seed according to a collection of arguments and return random number in 0,1
    """
    np.random.seed(hash(",".join(map(repr,args))) % 4294967295)
    return np.random.rand()

def image_centroid(im):
    """Return the image centroid (in radians)
    """

    xlist = np.arange(0,-im.xdim,-1)*im.psize + (im.psize*im.xdim)/2.0 - im.psize/2.0
    ylist = np.arange(0,-im.ydim,-1)*im.psize + (im.psize*im.ydim)/2.0 - im.psize/2.0

    x0 = np.sum(np.outer(0.0*ylist+1.0, xlist).ravel()*im.imvec)/np.sum(im.imvec)
    y0 = np.sum(np.outer(ylist, 0.0*xlist+1.0).ravel()*im.imvec)/np.sum(im.imvec)

    return np.array([x0, y0])

def ftmatrix(pdim, xdim, ydim, uvlist, pulse=PULSE_DEFAULT, mask=[]):
    """Return a DFT matrix for the xdim*ydim image with pixel width pdim
       that extracts spatial frequencies of the uv points in uvlist.
    """

    xlist = np.arange(0,-xdim,-1)*pdim + (pdim*xdim)/2.0 - pdim/2.0
    ylist = np.arange(0,-ydim,-1)*pdim + (pdim*ydim)/2.0 - pdim/2.0

    # original sign convention
    #ftmatrices = [pulse(2*np.pi*uv[0], 2*np.pi*uv[1], pdim, dom="F") * np.outer(np.exp(-2j*np.pi*ylist*uv[1]), np.exp(-2j*np.pi*xlist*uv[0])) for uv in uvlist] #list of matrices at each freq

    # changed the sign convention to agree with BU data (Jan 2017)
    ftmatrices = [pulse(2*np.pi*uv[0], 2*np.pi*uv[1], pdim, dom="F") * np.outer(np.exp(2j*np.pi*ylist*uv[1]), np.exp(2j*np.pi*xlist*uv[0])) for uv in uvlist] #list of matrices at each freq

    ftmatrices = np.reshape(np.array(ftmatrices), (len(uvlist), xdim*ydim))

    if len(mask):
        ftmatrices = ftmatrices[:,mask]

    return ftmatrices

def ftmatrix_centered(im, pdim, xdim, ydim, uvlist, pulse=PULSE_DEFAULT):
    """Return a DFT matrix for the xdim*ydim image with pixel width pdim
       that extracts spatial frequencies of the uv points in uvlist.
       in this version, it puts the image centroid at the origin
    """

    # TODO : there is a residual value for the center being around 0, maybe we should chop this off to be exactly 0
    # Coordinate matrix for COM constraint
    xlist = np.arange(0,-xdim,-1)*pdim + (pdim*xdim)/2.0 - pdim/2.0
    ylist = np.arange(0,-ydim,-1)*pdim + (pdim*ydim)/2.0 - pdim/2.0
    x0 = np.sum(np.outer(0.0*ylist+1.0, xlist).ravel()*im)/np.sum(im)
    y0 = np.sum(np.outer(ylist, 0.0*xlist+1.0).ravel()*im)/np.sum(im)

    #Now shift the lists
    xlist = xlist - x0
    ylist = ylist - y0

    ftmatrices = [pulse(2*np.pi*uv[0], 2*np.pi*uv[1], pdim, dom="F") * np.outer(np.exp(-2j*np.pi*ylist*uv[1]), np.exp(-2j*np.pi*xlist*uv[0])) for uv in uvlist] #list of matrices at each freq
    ftmatrices = np.reshape(np.array(ftmatrices), (len(uvlist), xdim*ydim))
    return ftmatrices



def ticks(axisdim, psize, nticks=8):
    """Return a list of ticklocs and ticklabels
       psize should be in desired units
    """

    axisdim = int(axisdim)
    nticks = int(nticks)
    if not axisdim % 2: axisdim += 1
    if nticks % 2: nticks -= 1
    tickspacing = float((axisdim-1))/nticks
    ticklocs = np.arange(0, axisdim+1, tickspacing) - 0.5
    ticklabels= np.around(psize * np.arange((axisdim-1)/2.0, -(axisdim)/2.0, -tickspacing), decimals=1)
    return (ticklocs, ticklabels)

def power_of_two(target):
    """Finds the next greatest power of two
    """
    cur = 1
    if target > 1:
        for i in range(0, int(target)):
            if (cur >= target):
                return cur
            else: cur *= 2
    else:
        return 1


def paritycompare(perm1, perm2):
    """Compare the parity of two permutations.
       Assume both lists are equal length and with same elements
       Copied from: http://stackoverflow.com/questions/1503072/how-to-check-if-permutations-have-equal-parity
    """

    perm2 = list(perm2)
    perm2_map = dict((v, i) for i,v in enumerate(perm2))
    transCount=0
    for loc, p1 in enumerate(perm1):
        p2 = perm2[loc]
        if p1 != p2:
            sloc = perm2_map[p1]
            perm2[loc], perm2[sloc] = p1, p2
            perm2_map[p1], perm2_map[p2] = sloc, loc
            transCount += 1

    if not (transCount % 2): return 1
    else: return  -1


def sigtype(datatype):
    """Return the type of noise corresponding to the data type
    """

    datatype = str(datatype)
    if datatype in ['vis', 'amp']: sigmatype='sigma'
    elif datatype in ['qvis', 'qamp']: sigmatype='qsigma'
    elif datatype in ['uvis', 'uamp']: sigmatype='usigma'
    elif datatype in ['vvis', 'vamp']: sigmatype='vsigma'
    elif datatype in ['pvis', 'pamp']: sigmatype='psigma'
    elif datatype in ['pvis', 'pamp']: sigmatype='psigma'
    elif datatype in ['rrvis', 'rramp']: sigmatype='rrsigma'
    elif datatype in ['llvis', 'llamp']: sigmatype='llsigma'
    elif datatype in ['rlvis', 'rlamp']: sigmatype='rlsigma'
    elif datatype in ['lrvis', 'lramp']: sigmatype='lrsigma'
    elif datatype in ['m', 'mamp']: sigmatype='msigma'
    elif datatype in ['phase']: sigmatype='sigma_phase'
    elif datatype in ['qphase']: sigmatype='qsigma_phase'
    elif datatype in ['uphase']: sigmatype='usigma_phase'
    elif datatype in ['vphase']: sigmatype='vsigma_phase'
    elif datatype in ['pphase']: sigmatype='psigma_phase'
    elif datatype in ['mphase']: sigmatype='msigma_phase'
    elif datatype in ['rrphase']: sigmatype='rrsigma_phase'
    elif datatype in ['llphase']: sigmatype='llsigma_phase'
    elif datatype in ['rlphase']: sigmatype='rlsigma_phase'
    elif datatype in ['lrphase']: sigmatype='lrsigma_phase'

    else: sigmatype = False

    return sigmatype


def rastring(ra):
    """Convert a ra in fractional hours to formatted string
    """
    h = int(ra)
    m = int((ra-h)*60.)
    s = (ra-h-m/60.)*3600.
    out = "%2i h %2i m %2.4f s" % (h,m,s)
    return out

def decstring(dec):
    """Convert a dec in fractional degrees to formatted string
    """

    deg = int(dec)
    m = int((abs(dec)-abs(deg))*60.)
    s = (abs(dec)-abs(deg)-m/60.)*3600.
    out = "%2i deg %2i m %2.4f s" % (deg,m,s)
    return out

def gmtstring(gmt):
    """Convert a gmt in fractional hours to formatted string
    """

    if gmt > 24.0: gmt = gmt-24.0
    h = int(gmt)
    m = int((gmt-h)*60.)
    s = (gmt-h-m/60.)*3600.
    out = "%02i:%02i:%2.4f" % (h,m,s)
    return out

#TODO fix this hacky way to do it!!
def gmst_to_utc(gmst,mjd):
    """Convert gmst times in hours to utc hours using astropy
    """

    mjd=int(mjd)
    time_obj_ref = at.Time(mjd, format='mjd', scale='utc')
    time_sidereal_ref = time_obj_ref.sidereal_time('mean', 'greenwich').hour
    time_utc = (gmst - time_sidereal_ref) * 0.9972695601848 
    return time_utc

def utc_to_gmst(utc, mjd):
    """Convert utc times in hours to gmst using astropy
    """
    mjd=int(mjd) #MJD should always be an integer, but was float in older versions of the code
    time_obj = at.Time(utc/24.0 + np.floor(mjd), format='mjd', scale='utc')
    time_sidereal = time_obj.sidereal_time('mean','greenwich').hour
    return time_sidereal

def earthrot(vecs, thetas):
    """Rotate a vector / array of vectors about the z-direction by theta / array of thetas (radian)
    """

    if len(vecs.shape)==1:
        vecs = np.array([vecs])
    if np.isscalar(thetas):
        thetas = np.array([thetas for i in range(len(vecs))])

    # equal numbers of sites and angles
    if len(thetas) == len(vecs):
        rotvec = np.array([np.dot(np.array(((np.cos(thetas[i]),-np.sin(thetas[i]),0),(np.sin(thetas[i]),np.cos(thetas[i]),0),(0,0,1))), vecs[i])
                       for i in range(len(vecs))])

    # only one rotation angle, many sites
    elif len(thetas) == 1:
        rotvec = np.array([np.dot(np.array(((np.cos(thetas[0]),-np.sin(thetas[0]),0),(np.sin(thetas[0]),np.cos(thetas[0]),0),(0,0,1))), vecs[i])
                       for i in range(len(vecs))])
    # only one site, many angles
    elif len(vecs) == 1:
        rotvec = np.array([np.dot(np.array(((np.cos(thetas[i]),-np.sin(thetas[i]),0),(np.sin(thetas[i]),np.cos(thetas[i]),0),(0,0,1))), vecs[0])
                       for i in range(len(thetas))])
    else:
        raise Exception("Unequal numbers of vectors and angles in earthrot(vecs, thetas)!")

    #if rotvec.shape[0]==1: rotvec = rotvec[0]
    return rotvec

def elev(obsvecs, sourcevec):
    """Return the elevation of a source with respect to an observer/observers in radians
       obsvec can be an array of vectors but sourcevec can ONLY be a single vector
    """

    if len(obsvecs.shape)==1:
        obsvecs=np.array([obsvecs])

    anglebtw = np.array([np.dot(obsvec,sourcevec)/np.linalg.norm(obsvec)/np.linalg.norm(sourcevec) for obsvec in obsvecs])
    el = 0.5*np.pi - np.arccos(anglebtw)

    return el

def elevcut(obsvecs, sourcevec, elevmin=ELEV_LOW, elevmax=ELEV_HIGH):
    """Return True if a source is observable by a telescope vector
    """

    angles = elev(obsvecs, sourcevec)/DEGREE

    return (angles > elevmin) * (angles < elevmax)

def hr_angle(gst, lon, ra):
    """Computes the hour angle for a source at RA, observer at longitude long, and GMST time gst
       gst in hours, ra & lon ALL in radian
       longitude positive east
    """

    hr_angle = np.mod(gst + lon - ra, 2*np.pi)
    return hr_angle

def par_angle(hr_angle, lat, dec):
    """Compute the parallactic angle for a source at hr_angle and dec for an observer with latitude lat.
       All angles in radian
    """

    num = np.sin(hr_angle)*np.cos(lat)
    denom = np.sin(lat)*np.cos(dec) - np.cos(lat)*np.sin(dec)*np.cos(hr_angle)

    return np.arctan2(num, denom)

def xyz_2_latlong(obsvecs):
    """Compute the (geocentric) latitude and longitude of a site at geocentric position x,y,z
       The output is in radians
    """

    if len(obsvecs.shape)==1:
        obsvecs=np.array([obsvecs])
    out = []
    for obsvec in obsvecs:
        x = obsvec[0]
        y = obsvec[1]
        z = obsvec[2]
        lon = np.array(np.arctan2(y,x))
        lat = np.array(np.arctan2(z, np.sqrt(x**2+y**2)))
        out.append([lat,lon])

    out = np.array(out)

    #if out.shape[0]==1: out = out[0]
    return out

def roundTime(dt=None, roundTo=60):
    from astropy.time import Time
    import datetime as datetime
    if dt == None: 
        dt = datetime.datetime.now()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)

def jd2expt2017(jd):
    '''
    Function translating from jd to expt for April 2017 EHT
    '''
    if (jd > 2457853.470)&(jd < 2457854.132 ):
        return 3600
    elif (jd > 2457849.531)&(jd < 2457850.177):
        return 3598
    elif (jd > 2457850.667)&(jd < 2457851.363):
        return 3599
    elif (jd > 2457848.438)&(jd < 2457849.214):
        return 3597
    elif (jd > 2457854.427)&(jd < 2457855.141):
        return 3601
    else:
        return None

def make_scan_list_EHT2017(fpath):
    '''
    generates data frame with information about scans for EHT2017
    '''
    import ehtim.vex as vex
    import os
    import pandas as pd
    from astropy.time import Time, TimeDelta
    import datetime as datetime
    nam2lett = {'ALMA':'A','APEX':'X','LMT':'L','PICOVEL':'P','SMTO':'Z','SPT':'T','JCMT':'J','SMAP':'S'}
    track2expt ={'D':3597,'B':3598, 'C':3599,'A':3600,'E':3601}
    list_files = os.listdir(fpath)
    scans = pd.DataFrame({'source' : []})

    for fi in list_files:#loop over vex files in folder
        track_loc = fi[3].upper()
        vpath = fpath+fi
        aa = vex.Vex(vpath)
        dec = []
        for cou in range(len(aa.source)):
            dec_h = float(aa.source[cou]['dec'].split('d')[0])
            dec_m = float((aa.source[cou]['dec'].split('d')[1])[0:2])
            dec_s = float((aa.source[cou]['dec'].split('d')[1])[3:-1])
            dec.append(tuple((dec_h,dec_m,dec_s)))    
        ra = []
        for cou in range(len(aa.source)):
            ra_d = float(aa.source[cou]['ra'].split('h')[0])
            ra_m = float(aa.source[cou]['ra'].split('h')[1].split('m')[0])
            ra_s = float(aa.source[cou]['ra'].split('h')[1].split('m')[1][:-1])
            ra.append(tuple((ra_d,ra_m,ra_s)))      
        sour_name = [aa.source[x]['source'] for x in range(len(aa.source))]
        dict_ra = dict(zip(sour_name,ra))
        dict_dec = dict(zip(sour_name,dec))
        t_min = [aa.sched[x]['start_hr'] for x in range(len(aa.sched))]
        sour = [aa.sched[x]['source'] for x in range(len(aa.sched))]
        datet = []
        elev = []
        antenas = []
        duration=[]
        for x in range(len(aa.sched)):#loop over scans in given file
            t = Time(aa.sched[x]['mjd_floor'], format='mjd', scale='utc')
            tiso = Time(t, format='iso', scale='utc')
            tiso = tiso + TimeDelta(t_min[x]*3600., format='sec')
            datet.append(tiso)
            ant_foo = set([nam2lett[aa.sched[x]['scan'][y]['site']] for y in range(len(aa.sched[x]['scan']))])
            antenas.append(ant_foo)
            duration_foo =max([aa.sched[x]['scan'][y]['scan_sec'] for y in range(len(aa.sched[x]['scan']))])
            duration.append(duration_foo)
        #time_min = [pd.tslib.Timestamp(datet[x].datetime) for x in range(len(datet))]
        time_min = [pd.Timestamp(datet[x].datetime) for x in range(len(datet))]
        time_max = [time_min[x] + datetime.timedelta(seconds=duration[x]) for x in range(len(aa.sched))]
        foo = pd.DataFrame(aa.sched)
        foo = foo[['source','mjd_floor','start_hr']]
        foo['time_min']=time_min
        foo['time_max']=time_max
        foo['scan_no'] = foo.index
        foo['scan_no'] = list(map(int,foo['scan_no']))
        foo['track'] = [track_loc]*foo.shape[0]
        foo['expt'] = [int(track2expt[track_loc])]*foo.shape[0]
        foo['antenas'] = antenas
        foo['duration'] = duration
        scans = pd.concat([scans,foo], ignore_index=True)
    scans = scans.reindex_axis(['mjd_floor','expt','track','scan_no','source','time_min','time_max','duration','antenas'],axis=1)
    scans = scans.sort_values('time_max')
    scans = scans.reset_index(drop=True)
    scans['scan_no_tot'] = scans.index
    return scans

def match_scans(scans,data):
    '''
    matches data with scans
    '''
    import pandas as pd
    from astropy.time import Time, TimeDelta
    import datetime as datetime
    bins_labels = [None]*(2*scans.shape[0]-1)
    bins_labels[1::2] = map(lambda x: -x-1,list(scans['scan_no_tot'])[:-1])
    bins_labels[::2] = list(scans['scan_no_tot'])
    dtmin = datetime.timedelta(seconds = 2.) 
    dtmax = datetime.timedelta(seconds = 2.) 
    binsT = [None]*(2*scans.shape[0])
    binsT[::2] = list(map(lambda x: x - dtmin,list(scans.time_min)))
    binsT[1::2] = list(map(lambda x: x + dtmax,list(scans.time_max))) 
    ordered_labels = pd.cut(data.datetime, binsT,labels = bins_labels)
    data['scan_no_tot'] = ordered_labels
    data = data[list(map(lambda x: x >= 0, data['scan_no_tot']))]
    data['scan_id'] = data['scan_no_tot']
    data.drop('scan_no_tot',axis=1,inplace=True)
    return data

def make_cphase_df(obsdata,pathVex=''):
    '''
    adds cphase data in form of pandas dataframe to enable easier exploration
    '''
    data = obsdata.c_phases(mode='all')
    import pandas as pd
    from astropy.time import Time
    AZ2Z = {'AZ': 'Z', 'PV': 'P', 'SM':'S', 'SR':'R','JC':'J', 'AA':'A','AP':'X', 'LM':'L','SP':'Y'}
    polar = 'unknown'; band = 'unknown' #for now fixed
    sour=obsdata.source
    df = pd.DataFrame(data=data)
    df['fmjd'] = df['time']/24.
    df['mjd'] = obsdata.mjd + df['fmjd']
    df['triangle'] = list(map(lambda x: AZ2Z[x[0].decode('unicode_escape')]+AZ2Z[x[1].decode('unicode_escape')]+AZ2Z[x[2].decode('unicode_escape')],zip(df['t1'],df['t2'],df['t3'])))
    df['datetime'] = Time(df['mjd'], format='mjd').datetime
    df['datetime'] =list(map(lambda x: roundTime(x,roundTo=1),df['datetime']))
    df['jd'] = Time(df['mjd'], format='mjd').jd
    df['polarization'] = polar
    df['band'] = band
    df['expt_no'] = list(map(jd2expt2017,df['jd']))
    df['source'] = sour
    if pathVex!='':
        scans = make_scan_list_EHT2017(pathVex)
        df = match_scans(scans,df)
    return df

def scan_average_cphases_bs(cdf,return_type='rec',num_samples=int(1e3),robust=False):
    '''
    averages cphases across scan (circular mean), with bootstrapped sigma
    '''
    import pandas as pd
    groupV=['scan_id','expt_no','polarization','band','triangle','t1','t2','t3']
    cdf['number'] = cdf['cphase']
    cdf = cdf.groupby(groupV).agg({'cphase': lambda x: circular_mean(x, robust=robust), 'sigmacp': lambda x: bootstrap(x, lambda y: std_stat(y,robust=robust), num_samples=num_samples), 'time': np.mean,
    'number': lambda x: len(x), 'u1':np.mean, 'u2': np.mean, 'u3':np.mean,'v1':np.mean, 'v2': np.mean, 'v3':np.mean}).reset_index()
    cdf.drop(cdf[cdf.number < 5.].index, inplace=True)
    cdf.drop(cdf[cdf.sigmacp < 0.1].index, inplace=True)
    if return_type=='rec':
        return cphase_df_to_rec(cdf)
    elif return_type=='df':
        return cdf

def cphase_df_to_rec(cdf):
    '''
    converts cphase pandas dataframe to cphase numpy recarray
    '''
    import pandas as pd
    return cdf[['time','t1','t2','t3','u1','v1','u2','v2','u3','v3','cphase','sigmacp']].to_records(index=False)

def circular_mean(theta,robust=False):
    '''
    circular mean statistics
    theta in deg, returns deg, may use robust statistics
    '''
    theta = np.asarray(theta, dtype=np.float32)*np.pi/180.
    theta = theta[theta==theta]
    if len(theta)==0:
        return None
    else:
        if robust==False:
            C = np.mean(np.cos(theta))
            S = np.mean(np.sin(theta))
        else:
            C = np.median(np.cos(theta))
            S = np.median(np.sin(theta))
        mt = np.arctan2(S,C)*180./np.pi
        return np.mod(mt,360)
    
def std_stat(theta,robust=False):
    '''
    mean variability of mean
    theta in deg, returns deg, may use robust statistics
    '''
    if robust==False:
        return np.std(theta)/np.sqrt(len(theta))
    else:
        theta = np.asarray(theta,dtype=np.float32)
        m = circular_mean(theta,robust=True)
        foo = np.exp(1j*theta*np.pi/180)*np.exp(-1j*m*np.pi/180)
        foo = np.angle(foo)*180/np.pi
        madev = 1.4826*np.median(np.abs(foo))/np.sqrt(len(foo))
        return madev

def bootstrap(data, statistic, num_samples=int(1e3)):
    """bootstrap estimator of a given statistic"""
    import numpy.random as npr
    stat = np.zeros(num_samples)
    data = np.asarray(data)
    n = len(data)
    idx = npr.randint(0, n, (num_samples, n))
    samples = data[idx]
    for cou in range(num_samples):
        stat[cou] = statistic(samples[cou,:])
    stat = np.sort(stat)
    return np.median(stat)

