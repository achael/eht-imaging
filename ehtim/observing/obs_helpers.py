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

try:
    import ephem
except ImportError:
    print("Warning: ephem not installed: cannot simulate space VLBI")
try:
    from pynfft.nfft import NFFT
except ImportError:
    print("Warning: No NFFT installed! Cannot use nfft functions")

import astropy.time as at
import astropy.coordinates as coords
import numpy as np
import scipy.special as ss
import itertools as it
import copy
import sys
import os

import ehtim.const_def
from ehtim.const_def import *
import scipy.ndimage as nd



import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars")

##################################################################################################
# Other Functions
##################################################################################################

def compute_uv_coordinates(array, site1, site2, time, mjd, ra, dec, rf, timetype='UTC',
                           elevmin=ELEV_LOW,  elevmax=ELEV_HIGH, fix_theta_GMST=False):
    """Compute u,v coordinates for an array at a given time for a source at a given ra,dec,rf
    """

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

    # TODO speed up?
    # use spacecraft ephemeris to get position of site 1
    spacemask1 = [np.all(coord == (0.,0.,0.)) for coord in coord1]
    if np.any(spacemask1):
        if timetype=='GMST':
            raise Exception("Spacecraft ephemeris only work with UTC!")

        site1space_list = site1[spacemask1]
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

def make_bispectrum(l1, l2, l3, vtype, polrep='stokes'):
    """make a list of bispectra and errors
       l1,l2,l3 are full datatables of visibility entries
       vtype is visibility types
    """

    # Choose the appropriate polarization and compute the bs and err
    if polrep=='stokes':
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
            p1 = l1['qvis'] + 1j*l1['uvis']
            p2 = l2['qvis'] + 1j*l2['uvis']
            p3 = l3['qvis'] + 1j*l3['uvis']

            var1 = l1['qsigma']**2 + l1['usigma']**2
            var2 = l2['qsigma']**2 + l2['usigma']**2
            var3 = l3['qsigma']**2 + l3['usigma']**2

    elif polrep=='circ':
        if vtype in ["rrvis", "llvis", "rlvis","lrvis",'pvis']:
            if vtype=='pvis': vtype='rlvis'

            if vtype=='rrvis': sigmatype='rrsigma'
            if vtype=='llvis': sigmatype='llsigma'
            if vtype=='rlvis': sigmatype='rlsigma'
            if vtype=='lrvis': sigmatype='lrsigma'

            p1 = l1[vtype]
            p2 = l2[vtype]
            p3 = l3[vtype]

            var1 = l1[sigmatype]**2
            var2 = l2[sigmatype]**2
            var3 = l3[sigmatype]**2

        elif vtype == "vis":
            p1 = 0.5*(l1['rrvis'] + l1['llvis'])
            p2 = 0.5*(l2['rrvis'] + l2['llvis'])
            p3 = 0.5*(l3['rrvis'] + l3['llvis'])

            var1 = 0.25*(l1['rrsigma']**2 + l1['llsigma']**2)
            var2 = 0.25*(l2['rrsigma']**2 + l2['llsigma']**2)
            var3 = 0.25*(l3['rrsigma']**2 + l3['llsigma']**2)

        elif vtype == "vvis":
            p1 = 0.5*(l1['rrvis'] - l1['llvis'])
            p2 = 0.5*(l2['rrvis'] - l2['llvis'])
            p3 = 0.5*(l3['rrvis'] - l3['llvis'])

            var1 = 0.25*(l1['rrsigma']**2 + l1['llsigma']**2)
            var2 = 0.25*(l2['rrsigma']**2 + l2['llsigma']**2)
            var3 = 0.25*(l3['rrsigma']**2 + l3['llsigma']**2)

        elif vtype == "qvis":
            p1 = 0.5*(l1['lrvis'] + l1['rlvis'])
            p2 = 0.5*(l2['lrvis'] + l2['rlvis'])
            p3 = 0.5*(l3['lrvis'] + l3['rlvis'])

            var1 = 0.25*(l1['lrsigma']**2 + l1['rlsigma']**2)
            var2 = 0.25*(l2['lrsigma']**2 + l2['rlsigma']**2)
            var3 = 0.25*(l3['lrsigma']**2 + l3['rlsigma']**2)

        elif vtype == "uvis":
            p1 = 0.5j*(l1['lrvis'] - l1['rlvis'])
            p2 = 0.5j*(l2['lrvis'] - l2['rlvis'])
            p3 = 0.5j*(l3['lrvis'] - l3['rlvis'])

            var1 = 0.25*(l1['lrsigma']**2 + l1['rlsigma']**2)
            var2 = 0.25*(l2['lrsigma']**2 + l2['rlsigma']**2)
            var3 = 0.25*(l3['lrsigma']**2 + l3['rlsigma']**2)
    else:
        raise Exception("only 'stokes' and 'circ' are supported polreps!")

    # Make the bispectrum and its uncertainty
    bi = p1*p2*p3
    bisig = np.abs(bi) * np.sqrt(var1/np.abs(p1)**2 +
                                 var2/np.abs(p2)**2 +
                                 var3/np.abs(p3)**2)

    return (bi, bisig)


def make_closure_amplitude(blue1, blue2, red1, red2, vtype, ctype='camp', debias=True, polrep='stokes'):

    """make a list of closure amplitudes and errors
       blue1 and blue2 are full datatables numerator entries
       red1 and red2 are full datatables of denominator entries
       vtype is the  visibility type
    """

    if not (ctype in ['camp', 'logcamp']):
        raise Exception("closure amplitude type must be 'camp' or 'logcamp'!")

    if polrep=='stokes':

        if vtype in ["vis", "qvis", "uvis", "vvis"]:
            if vtype=='vis':  sigmatype='sigma'
            if vtype=='qvis': sigmatype='qsigma'
            if vtype=='uvis': sigmatype='usigma'
            if vtype=='vvis': sigmatype='vsigma'

            sig1 = blue1[sigmatype]
            sig2 = blue2[sigmatype]
            sig3 = red1[sigmatype]
            sig4 = red2[sigmatype]

            p1 = np.abs(blue1[vtype])
            p2 = np.abs(blue2[vtype])
            p3 = np.abs(red1[vtype])
            p4 = np.abs(red2[vtype])

        elif vtype == "rrvis":
            sig1 = np.sqrt(blue1['sigma']**2 + blue1['vsigma']**2)
            sig2 = np.sqrt(blue2['sigma']**2 + blue2['vsigma']**2)
            sig3 = np.sqrt(red1['sigma']**2 + red1['vsigma']**2)
            sig4 = np.sqrt(red2['sigma']**2 + red2['vsigma']**2)

            p1 = np.abs(blue1['vis'] + blue1['vvis'])
            p2 = np.abs(blue2['vis'] + blue2['vvis'])
            p3 = np.abs(red1['vis'] + red1['vvis'])
            p4 = np.abs(red2['vis'] + red2['vvis'])

        elif vtype == "llvis":
            sig1 = np.sqrt(blue1['sigma']**2 + blue1['vsigma']**2)
            sig2 = np.sqrt(blue2['sigma']**2 + blue2['vsigma']**2)
            sig3 = np.sqrt(red1['sigma']**2 + red1['vsigma']**2)
            sig4 = np.sqrt(red2['sigma']**2 + red2['vsigma']**2)

            p1 = np.abs(blue1['vis'] - blue1['vvis'])
            p2 = np.abs(blue2['vis'] - blue2['vvis'])
            p3 = np.abs(red1['vis'] - red1['vvis'])
            p4 = np.abs(red2['vis'] - red2['vvis'])

        elif vtype == "lrvis":
            sig1 = np.sqrt(blue1['qsigma']**2 + blue1['usigma']**2)
            sig2 = np.sqrt(blue2['qsigma']**2 + blue2['usigma']**2)
            sig3 = np.sqrt(red1['qsigma']**2 + red1['usigma']**2)
            sig4 = np.sqrt(red2['qsigma']**2 + red2['usigma']**2)

            p1 = np.abs(blue1['qvis'] - 1j*blue1['uvis'])
            p2 = np.abs(blue2['qvis'] - 1j*blue2['uvis'])
            p3 = np.abs(red1['qvis'] - 1j*red1['uvis'])
            p4 = np.abs(red2['qvis'] - 1j*red2['uvis'])

        elif vtype in ["pvis","rlvis"]:
            sig1 = np.sqrt(blue1['qsigma']**2 + blue1['usigma']**2)
            sig2 = np.sqrt(blue2['qsigma']**2 + blue2['usigma']**2)
            sig3 = np.sqrt(red1['qsigma']**2 + red1['usigma']**2)
            sig4 = np.sqrt(red2['qsigma']**2 + red2['usigma']**2)

            p1 = np.abs(blue1['qvis'] + 1j*blue1['uvis'])
            p2 = np.abs(blue2['qvis'] + 1j*blue2['uvis'])
            p3 = np.abs(red1['qvis'] + 1j*red1['uvis'])
            p4 = np.abs(red2['qvis'] + 1j*red2['uvis'])

    elif polrep=='circ':
        if vtype in ["rrvis", "llvis", "rlvis","lrvis",'pvis']:
            if vtype=='pvis': vtype='rlvis' # p = rl

            if vtype=='rrvis': sigmatype='rrsigma'
            if vtype=='llvis': sigmatype='llsigma'
            if vtype=='rlvis': sigmatype='rlsigma'
            if vtype=='lrvis': sigmatype='lrsigma'

            sig1 = blue1[sigmatype]
            sig2 = blue2[sigmatype]
            sig3 = red1[sigmatype]
            sig4 = red2[sigmatype]

            p1 = np.abs(blue1[vtype])
            p2 = np.abs(blue2[vtype])
            p3 = np.abs(red1[vtype])
            p4 = np.abs(red2[vtype])

        elif vtype == "vis":
            sig1 = 0.5*np.sqrt(blue1['rrsigma']**2 + blue1['llsigma']**2)
            sig2 = 0.5*np.sqrt(blue2['rrsigma']**2 + blue2['llsigma']**2)
            sig3 = 0.5*np.sqrt(red1['rrsigma']**2 + red1['llsigma']**2)
            sig4 = 0.5*np.sqrt(red2['rrsigma']**2 + red2['llsigma']**2)

            p1 = 0.5*np.abs(blue1['rrvis'] + blue1['llvis'])
            p2 = 0.5*np.abs(blue2['rrvis'] + blue2['llvis'])
            p3 = 0.5*np.abs(red1['rrvis'] + red1['llvis'])
            p4 = 0.5*np.abs(red2['rrvis'] + red2['llvis'])

        elif vtype == "vvis":
            sig1 = 0.5*np.sqrt(blue1['rrsigma']**2 + blue1['llsigma']**2)
            sig2 = 0.5*np.sqrt(blue2['rrsigma']**2 + blue2['llsigma']**2)
            sig3 = 0.5*np.sqrt(red1['rrsigma']**2 + red1['llsigma']**2)
            sig4 = 0.5*np.sqrt(red2['rrsigma']**2 + red2['llsigma']**2)

            p1 = 0.5*np.abs(blue1['rrvis'] - blue1['llvis'])
            p2 = 0.5*np.abs(blue2['rrvis'] - blue2['llvis'])
            p3 = 0.5*np.abs(red1['rrvis'] - red1['llvis'])
            p4 = 0.5*np.abs(red2['rrvis'] - red2['llvis'])

        elif vtype == "qvis":
            sig1 = 0.5*np.sqrt(blue1['lrsigma']**2 + blue1['rlsigma']**2)
            sig2 = 0.5*np.sqrt(blue2['lrsigma']**2 + blue2['rlsigma']**2)
            sig3 = 0.5*np.sqrt(red1['lrsigma']**2 + red1['rlsigma']**2)
            sig4 = 0.5*np.sqrt(red2['lrsigma']**2 + red2['rlsigma']**2)

            p1 = 0.5*np.abs(blue1['lrvis'] + blue1['rlvis'])
            p2 = 0.5*np.abs(blue2['lrvis'] + blue2['rlvis'])
            p3 = 0.5*np.abs(red1['lrvis'] + red1['rlvis'])
            p4 = 0.5*np.abs(red2['lrvis'] + red2['rlvis'])

        elif vtype == "uvis":
            sig1 = 0.5*np.sqrt(blue1['lrsigma']**2 + blue1['rlsigma']**2)
            sig2 = 0.5*np.sqrt(blue2['lrsigma']**2 + blue2['rlsigma']**2)
            sig3 = 0.5*np.sqrt(red1['lrsigma']**2 + red1['rlsigma']**2)
            sig4 = 0.5*np.sqrt(red2['lrsigma']**2 + red2['rlsigma']**2)

            p1 = 0.5*np.abs(blue1['lrvis'] - blue1['rlvis'])
            p2 = 0.5*np.abs(blue2['lrvis'] - blue2['rlvis'])
            p3 = 0.5*np.abs(red1['lrvis'] - red1['rlvis'])
            p4 = 0.5*np.abs(red2['lrvis'] - red2['rlvis'])
    else:
        raise Exception("only 'stokes' and 'circ' are supported polreps!")

    # Debias the amplitude
    if debias:
        p1 = amp_debias(p1, sig1, force_nonzero=True)
        p2 = amp_debias(p2, sig2, force_nonzero=True)
        p3 = amp_debias(p3, sig3, force_nonzero=True)
        p4 = amp_debias(p4, sig4, force_nonzero=True)
    else:
        p1 = np.abs(p1)
        p2 = np.abs(p2)
        p3 = np.abs(p3)
        p4 = np.abs(p4)

    # Get snrs
    snr1 = p1/sig1
    snr2 = p2/sig2
    snr3 = p3/sig3
    snr4 = p4/sig4

    # Compute the closure amplitude and its uncertainty
    if ctype=='camp':
        camp = np.abs((p1*p2)/(p3*p4))
        camperr = camp * np.sqrt(1./(snr1**2) + 1./(snr2**2) + 1./(snr3**2) + 1./(snr4**2))

        # Debias
        if debias:
            camp = camp_debias(camp, snr3, snr4)

    elif ctype=='logcamp':
        camp = np.log(np.abs(p1)) + np.log(np.abs(p2)) - np.log(np.abs(p3)) - np.log(np.abs(p4))
        camperr = np.sqrt(1./(snr1**2) + 1./(snr2**2) + 1./(snr3**2) + 1./(snr4**2))

        # Debias
        if debias:
            camp = logcamp_debias(camp, snr1, snr2, snr3, snr4)

    return (camp, camperr)

def amp_debias(amp, sigma, force_nonzero=False):
    """Return debiased visibility amplitudes
    """

    deb2 = np.abs(amp)**2 - np.abs(sigma)**2

    # puts amplitude at 0 if snr < 1
    deb2 *= (np.nan_to_num(np.abs(amp)) > np.nan_to_num(np.abs(sigma)))

    # raises amplitude to sigma to force nonzero
    if force_nonzero:
        deb2 += (np.nan_to_num(np.abs(amp)) < np.nan_to_num(np.abs(sigma))) * np.abs(sigma)**2
    out = np.sqrt(deb2)

    return out


def camp_debias(camp, snr3, snr4):
    """Debias closure amplitudes
       snr3 and snr4 are snr of visibility amplitudes #3 and 4.
    """

    camp_debias = camp / (1 + 1./(snr3**2) + 1./(snr4**2))
    return camp_debias

def logcamp_debias(log_camp, snr1, snr2, snr3, snr4):
    """Debias log closure amplitudes
       The snrs are the snr of the component visibility amplitudes
    """

    log_camp_debias = log_camp + 0.5*(1./(snr1**2) + 1./(snr2**2) - 1./(snr3**2) - 1./(snr4**2))
    return log_camp_debias

def gauss_uv(u, v, flux, beamparams, x=0., y=0.):
    """Return the value of the Gaussian FT with
       beamparams is [FWHMmaj, FWHMmin, theta, x, y], all in radian
       x,y are the center coordinates
    """

    sigma_maj = beamparams[0]/(2*np.sqrt(2*np.log(2)))
    sigma_min = beamparams[1]/(2*np.sqrt(2*np.log(2)))
    theta = -beamparams[2] # theta needs to be negative in this convention!

    # Covariance matrix
    a = (sigma_min * np.cos(theta))**2 + (sigma_maj*np.sin(theta))**2
    b = (sigma_maj * np.cos(theta))**2 + (sigma_min*np.sin(theta))**2
    c = (sigma_min**2 - sigma_maj**2) * np.cos(theta) * np.sin(theta)
    m = np.array([[a, c], [c, b]])

    uv = np.array([[u[i],v[i]] for i in range(len(u))])
    x2 = np.array([np.dot(uvi,np.dot(m,uvi)) for uvi in uv])

    g = np.exp(-2 * np.pi**2 * x2)
    p = np.exp(-2j * np.pi * (u*x + v*y))

    return flux * g * p

def sgra_kernel_uv(rf, u, v):
    """Return the value of the Sgr A* scattering kernel at a given u,v pt (in lambda),
    """

    lcm = (C/rf) * 100 # in cm
    sigma_maj = FWHM_MAJ * (lcm**2) / (2*np.sqrt(2*np.log(2))) * RADPERUAS
    sigma_min = FWHM_MIN * (lcm**2) / (2*np.sqrt(2*np.log(2))) * RADPERUAS
    theta = -POS_ANG * DEGREE # theta needs to be negative in this convention!

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
    """

    lcm = (C/rf) * 100 # in cm
    fwhm_maj_rf = FWHM_MAJ * (lcm**2)  * RADPERUAS
    fwhm_min_rf = FWHM_MIN * (lcm**2)  * RADPERUAS
    theta = POS_ANG * DEGREE

    return np.array([fwhm_maj_rf, fwhm_min_rf, theta])


def blnoise(sefd1, sefd2, tint, bw):
    """Determine the standard deviation of Gaussian thermal noise on a baseline
       This is the noise on the rr/ll/rl/lr product, not the Stokes parameter
       2-bit quantization is responsible for the 0.88 factor
    """

    noise = np.sqrt(sefd1*sefd2/(2*bw*tint))/0.88
    #noise = np.sqrt(sefd1*sefd2/(bw*tint))/0.88

    return noise

def merr(sigma, qsigma, usigma, I, m):
    """Return the error in mbreve real and imaginary parts given stokes input
    """

    err = np.sqrt((qsigma**2 + usigma**2 + (sigma*np.abs(m))**2)/(np.abs(I) ** 2))

    return err

def merr2(rlsigma, rrsigma, llsigma, I, m):
    """Return the error in mbreve real and imaginary parts given polprod input
    """

    err = np.sqrt((rlsigma**2 + (rrsigma**2 + llsigma**2)*np.abs(m)**2)/(np.abs(I) ** 2))

    return err

def cerror(sigma):
    """Return a complex number drawn from a circular complex Gaussian of zero mean
    """

    noise = np.random.normal(loc=0,scale=sigma) + 1j*np.random.normal(loc=0,scale=sigma)
    return noise

def cerror_hash(sigma,*args):
    """Return a complex number drawn from a circular complex Gaussian of zero mean
    """

    reargs = list(args)
    reargs.append('re')
    np.random.seed(hash(",".join(map(repr,reargs))) % 4294967295)
    re = np.random.randn()

    imargs = list(args)
    imargs.append('im')
    np.random.seed(hash(",".join(map(repr,imargs))) % 4294967295)
    im = np.random.randn()

    err = sigma * (re + 1j*im)

    return err

def hashrandn(*args):
    """set the seed according to a collection of arguments and return random gaussian var
    """

    np.random.seed(hash(",".join(map(repr,args))) % 4294967295)
    noise = np.random.randn()
    return noise

def hashrand(*args):
    """set the seed according to a collection of arguments and return random number in 0,1
    """
    
    np.random.seed(hash(",".join(map(repr,args))) % 4294967295)
    noise = np.random.rand()
    return noise

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
    xlist = np.arange(0,-xdim,-1)*pdim + (pdim*xdim)/2.0 - pdim/2.0
    ylist = np.arange(0,-ydim,-1)*pdim + (pdim*ydim)/2.0 - pdim/2.0
    x0 = np.sum(np.outer(0.0*ylist+1.0, xlist).ravel()*im)/np.sum(im)
    y0 = np.sum(np.outer(ylist, 0.0*xlist+1.0).ravel()*im)/np.sum(im)

    #Now shift the lists
    xlist = xlist - x0
    ylist = ylist - y0

    #list of matrices at each spatial freq
    ftmatrices = [pulse(2*np.pi*uv[0], 2*np.pi*uv[1], pdim, dom="F") * np.outer(np.exp(-2j*np.pi*ylist*uv[1]), np.exp(-2j*np.pi*xlist*uv[0])) for uv in uvlist]
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
       gst in hours, ra & lon ALL in radian, longitude positive east
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

    return out

def tri_minimal_set(sites, tarr, tkey):
    """returns a minimal set of triangles for bispectra and closure phase"""

    # determine ordering and reference site based on order of  self.tarr
    sites_ordered = [x for x in tarr['site'] if x in sites]
    ref = sites_ordered[0]
    sites_ordered.remove(ref)

    # Find all triangles that contain the ref
    tris = list(it.combinations(sites_ordered,2))
    tris = [(ref, t[0], t[1]) for t in tris]

    return tris

def quad_minimal_set(sites, tarr, tkey):
    """returns a minimal set of quadrangels for closure amplitude"""

    # determine ordering and reference site based on order of  self.tarr
    sites_ordered = np.array([x for x in tarr['site'] if x in sites])
    ref = sites_ordered[0]

    # Loop over other sites >=3 and form minimal closure amplitude set
    quads = []
    for i in range(3, len(sites_ordered)):
        for j in range(1, i):
            if j == i-1: k = 1
            else: k = j+1

            # convetion is (12)(34)/(14)(23)
            quad = (ref, sites_ordered[i], sites_ordered[j], sites_ordered[k])
            quads.append(quad)

    return quads


# TODO This returns A minimal set if input is maximal, but it is not necessarily the same
# minimal set as we would from  calling c_phases(count='min'). This is because of sign flips.
def reduce_tri_minimal(obs, datarr):
    """reduce a bispectrum or closure phase data array to a minimal set
       datarr can be either a bispectrum array of type DTBIS
       or a closure phase array of type DTCPHASE, or a time sorted list of either
    """

    # time sort or not
    if not (type(datarr) is list):
        datalist = []
        dtype = datarr.dtype
        for key, group in it.groupby(datarr, lambda x: x['time']):
            datalist.append(np.array([gp for gp in group],dtype=dtype))
        returnType='all'
    else:
        dtype = datarr[0].dtype
        datalist=datarr
        returnType='time'

    out = []

    for timegroup in datalist:
        if returnType=='all':
            outgroup = out
        else:
            outgroup = []

        # determine a minimal set of trinagles
        sites = list(set(np.hstack((timegroup['t1'],timegroup['t2'],timegroup['t3']))))
        tris = tri_minimal_set(sites, obs.tarr, obs.tkey)
        tris = [set(tri) for tri in tris]

        # add data points from original array to new array if in minimal set
        for dp in timegroup:
            # TODO: sign flips?
            if set((dp['t1'],dp['t2'],dp['t3'])) in tris:
                outgroup.append(dp)

        if returnType=='time':
            out.append(np.array(outgroup,dtype=dtype))
        else:
            out = outgroup

    if returnType=='all':
        out = np.array(out,dtype=dtype)
    return out

# TODO This returns A minimal set if input is maximal, but it is not necessarily the same
# minimal set as we would from  calling c_amplitudes(count='min'). This is because of  inverses.
def reduce_quad_minimal(obs, datarr,ctype='camp'):
    """reduce a closure amplitude or log closure amplitude array FROM a maximal set TO a minimal set
    """

    if not ctype in ['camp','logcamp']:
        raise Exception("ctype must be 'camp' or 'logcamp'")

    # time sort or not
    if not (type(datarr) is list):
        datalist = []
        dtype = datarr.dtype
        for key, group in it.groupby(datarr, lambda x: x['time']):
            datalist.append(np.array([x for x in group]))
        returnType='all'
    else:
        dtype = datarr[0].dtype
        datalist=datarr
        returnType='time'

    out = []
    for timegroup in datalist:
        if returnType=='all':
            outgroup = out
        else:
            outgroup = []

        # determine a minimal set of quadrangles
        sites = np.array(list(set(np.hstack((timegroup['t1'],timegroup['t2'],timegroup['t3'],timegroup['t4'])))))
        if len(sites) < 4:
            continue
        quads = quad_minimal_set(sites, obs.tarr, obs.tkey)

        # add data points from original camp array to new array if in minimal set
        # ANDREW TODO: do we need to change the ordering ??
        for dp in timegroup:

            # this is all same closure amplitude, but the ordering of labels is different
            if ((dp['t1'],dp['t2'],dp['t3'],dp['t4']) in quads or
                (dp['t2'],dp['t1'],dp['t4'],dp['t3']) in quads or
                (dp['t3'],dp['t4'],dp['t1'],dp['t2']) in quads or
                (dp['t4'],dp['t3'],dp['t2'],dp['t1']) in quads):

                outgroup.append(np.array(dp,dtype=DTCAMP))

            # flip the inverse closure amplitude
            elif ((dp['t1'],dp['t4'],dp['t3'],dp['t2']) in quads or
                  (dp['t2'],dp['t3'],dp['t4'],dp['t1']) in quads or
                  (dp['t3'],dp['t2'],dp['t1'],dp['t4']) in quads or
                  (dp['t4'],dp['t1'],dp['t2'],dp['t3']) in quads):

                dp2 = copy.deepcopy(dp)
                campold = dp['camp']
                sigmaold = dp['sigmaca']
                t1old = dp['t1']
                t2old = dp['t2']
                t3old = dp['t3']
                t4old = dp['t4']
                u1old = dp['u1']
                u2old = dp['u2']
                u3old = dp['u3']
                u4old = dp['u4']
                v1old = dp['v1']
                v2old = dp['v2']
                v3old = dp['v3']
                v4old = dp['v4']

                dp2['t1'] = t1old
                dp2['t2'] = t4old
                dp2['t3'] = t3old
                dp2['t4'] = t2old

                dp2['u1'] = u3old
                dp2['v1'] = v3old

                dp2['u2'] = -u4old
                dp2['v2'] = -v4old

                dp2['u3'] = u1old
                dp2['v3'] = v1old

                dp2['u4'] = -u2old
                dp2['v4'] = -v2old

                if ctype=='camp':
                    dp2['camp'] = 1./campold
                    dp2['sigmaca'] = sigmaold/(campold**2)

                elif ctype=='logcamp':
                    dp2['camp'] = -campold
                    dp2['sigmaca'] = sigmaold

                outgroup.append(dp2)

        if returnType=='time':
            out.append(np.array(outgroup,dtype=dtype))
        else:
            out = outgroup

    if returnType=='all':
        out = np.array(out,dtype=dtype)
    return out

def qimage(iimage, mimage, chiimage):
    """Return the Q image from m and chi"""
    return iimage * mimage * np.cos(2*chiimage)
    
def uimage(iimage, mimage, chiimage):
    """Return the U image from m and chi"""
    return iimage * mimage * np.sin(2*chiimage) 


##################################################################################################
# FFT & NFFT helper functions
##################################################################################################
class NFFTInfo(object):
    def __init__(self, xdim, ydim, psize, pulse, npad, p_rad, uv):
        self.xdim = int(xdim)
        self.ydim = int(ydim)
        self.psize = psize
        self.pulse = pulse

        self.npad = int(npad)
        self.p_rad = int(p_rad)
        self.uv = uv
        self.uvdim = len(uv)

        # set nfft plan
        uv_scaled = uv*psize
        nfft_plan = NFFT([xdim, ydim], self.uvdim, m=p_rad, n=[npad,npad])
        nfft_plan.x = uv_scaled
        nfft_plan.precompute()
        self.plan = nfft_plan

        # compute phase and pulsefac
        phases = np.exp(-1j*np.pi*(uv_scaled[:,0]+uv_scaled[:,1]))
        pulses = np.fromiter((pulse(2*np.pi*uv_scaled[i,0], 2*np.pi*uv_scaled[i,1], 1., dom="F")
                              for i in range(self.uvdim)),'c16')
        self.pulsefac = (pulses*phases)

class SamplerInfo(object):
    def __init__(self, order, uv, pulsefac):
        self.order = int(order)
        self.uv = uv
        self.pulsefac = pulsefac

class GridderInfo(object):
    def __init__(self, npad, func, p_rad, coords, weights):
        self.npad = int(npad)
        self.conv_func = func
        self.p_rad = int(p_rad)
        self.coords = coords
        self.weights = weights

class ImInfo(object):
    def __init__(self, xdim, ydim, npad, psize, pulse):
        self.xdim = int(xdim)
        self.ydim = int(ydim)
        self.npad = int(npad)
        self.psize = psize
        self.pulse = pulse

        padvalx1 = padvalx2 = int(np.floor((npad - xdim)/2.0))
        if xdim % 2:
            padvalx2 += 1
        padvaly1 = padvaly2 = int(np.floor((npad - ydim)/2.0))
        if ydim % 2:
            padvaly2 += 1

        self.padvalx1 = padvalx1
        self.padvalx2 = padvalx2
        self.padvaly1 = padvaly1
        self.padvaly2 = padvaly2

def conv_func_pill(x,y):
    if abs(x) < 0.5 and abs(y) < 0.5:
        out = 1.
    else:
        out = 0.
    return out

def conv_func_gauss(x,y):
    return np.exp(-(x**2 + y**2))

def conv_func_cubicspline(x,y):
    if abs(x) <= 1:
        fx = 1.5*abs(x)**3 - 2.5*abs(x)**2 + 1
    elif abs(x) < 2:
        fx = -0.5*abs(x)**3 + 2.5*abs(x)**2 -4*abs(x) + 2
    else:
        fx = 0

    if abs(y) <= 1:
        fy = 1.5*abs(y)**3 - 2.5*abs(y)**2 + 1
    elif abs(y) < 2:
        fy = -0.5*abs(y)**3 + 2.5*abs(y)**2 - 4*abs(y) + 2
    else:
        fy = 0

    return fx*fy

##There's a bug in scipy spheroidal function of order 0! - gives nans for eta<1
#def conv_func_spheroidal(x,y,p,m):
#    etax = 2.*x/float(p)
#    etay = 2.*x/float(p)
#    psix =  abs(1-etax**2)**m * scipy.special.pro_rad1(m,0,0.5*np.pi*p,etax)[0]
#    psiy = abs(1-etay**2)**m * scipy.special.pro_rad1(m,0,0.5*np.pi*p,etay)[0]
#    return psix*psiy

def fft_imvec(imvec, im_info):
    """
    Returns fft of imvec on  grid
    im_info = (xdim, ydim, npad, psize, pulse)
    order is the order of the spline interpolation
    """

    xdim = im_info.xdim
    ydim = im_info.ydim
    padvalx1 = im_info.padvalx1
    padvalx2 = im_info.padvalx2
    padvaly1 = im_info.padvaly1
    padvaly2 = im_info.padvaly2

    imarr = imvec.reshape(ydim, xdim)
    imarr = np.pad(imarr, ((padvalx1,padvalx2),(padvaly1,padvaly2)), 'constant', constant_values=0.0)
    npad = imarr.shape[0]
    if imarr.shape[0]!=imarr.shape[1]:
        raise Exception("FFT padding did not return a square image!")

    # FFT for visibilities
    vis_im = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(imarr)))

    return vis_im

def sampler(griddata, sampler_info_list, sample_type="vis"):
    """
    Samples griddata (e.g. the FFT of an image) at uv points
    the griddata should already be rotated so u,v = 0,0 in the center
    sampler_info_list is an appropriately ordered list of 4 sampler_info objects
    order is the order of the spline interpolation
    """
    if sample_type not in ["vis","bs","camp"]:
        raise Exception("sampler sample_type should be either 'vis','bs',or 'camp'!")
    if griddata.shape[0] != griddata.shape[1]:
        raise Exception("griddata should be a square array!")

    dataset = []
    for sampler_info in sampler_info_list:

        vu2 = sampler_info.uv
        pulsefac = sampler_info.pulsefac

        datare = nd.map_coordinates(np.real(griddata), vu2, order=sampler_info.order)
        dataim = nd.map_coordinates(np.imag(griddata), vu2, order=sampler_info.order)

        data = datare + 1j*dataim
        data = data * pulsefac

        dataset.append(data)

    if sample_type=="vis":
        out = dataset[0]
    if sample_type=="bs":
        out = dataset[0]*dataset[1]*dataset[2]
    if sample_type=="camp":
        out = np.abs((dataset[0]*dataset[1])/(dataset[2]*dataset[3]))
    return out

def gridder(data_list, gridder_info_list):
    """
    Grid the data sampled at uv points on a square array
    gridder_info_list is an list of gridder_info objects
    """

    if len(data_list) != len(gridder_info_list):
        raise Exception("length of data_list in gridder() " +
                         "is not equal to length of gridder_info_list!")

    npad = gridder_info_list[0].npad
    datagrid = np.zeros((npad, npad)).astype('c16')

    for k in range(len(gridder_info_list)):
        gridder_info = gridder_info_list[k]
        data = data_list[k]

        if gridder_info.npad != npad:
            raise Exception("npad values not consistent in gridder_info_list!")

        p_rad = gridder_info.p_rad
        coords = gridder_info.coords
        weights = gridder_info.weights

        p_rad = int(p_rad)
        for i in range(2*p_rad+1):
            dy = i - p_rad
            for j in range(2*p_rad+1):
                dx = j - p_rad
                weight = weights[i][j]
                np.add.at(datagrid, tuple(map(tuple, (coords + [dy, dx]).transpose())), data*weight)

    return datagrid

def make_gridder_and_sampler_info(im_info, uv, conv_func=GRIDDER_CONV_FUNC_DEFAULT, p_rad=GRIDDER_P_RAD_DEFAULT, order=FFT_INTERP_DEFAULT):
    """
    Prep norms and weights for gridding data sampled at uv points on a square array
    im_info tuple contains (xdim, ydim, npad, psize, pulse) of the grid
    conv_func is the convolution function: current options are "pillbox", "gaussian"
    p_rad is the pixel radius inside wich the conv_func is nonzero
    """

    if not (conv_func in ['pillbox','gaussian','cubic']):
        raise Exception("conv_func must be either 'pillbox', 'gaussian', or, 'cubic'")

    xdim = im_info.xdim
    ydim = im_info.ydim
    npad = im_info.npad
    psize = im_info.psize
    pulse = im_info.pulse

    #compute grid u,v coordinates
    vu2 = np.hstack((uv[:,1].reshape(-1,1), uv[:,0].reshape(-1,1)))
    du  = 1.0/(npad*psize)
    vu2 = (vu2/du + 0.5*npad)

    coords = np.round(vu2).astype(int)
    dcoords = vu2 - np.round(vu2).astype(int)
    vu2  = vu2.T

    # TODO: phase rotations should be done separately for x and y if the image isn't square
    # e.g.,
    phase = np.exp(-1j*np.pi*psize*((1+im_info.xdim%2)*uv[:,0] + (1+im_info.ydim%2)*uv[:,1]))

    pulsefac = np.fromiter((pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], psize, dom="F") for uvpt in uv),'c16')
    pulsefac = pulsefac * phase

    #compute gridder norm
    weights = []
    norm = np.zeros_like(len(coords))
    for i in range(2*p_rad+1):
        weights.append([])
        dy = i - p_rad
        for j in range(2*p_rad+1):
            dx = j - p_rad
            if conv_func == 'gaussian':
                norm = norm + conv_func_gauss(dy - dcoords[:,0], dx - dcoords[:,1])
            elif conv_func == 'pillbox':
                norm = norm + conv_func_pill(dy - dcoords[:,0], dx - dcoords[:,1])
            elif conv_func == 'cubic':
                norm = norm + conv_func_cubicspline(dy - dcoords[:,0], dx - dcoords[:,1])

            weights[i].append(None)

    #compute weights for gridding
    for i in range(2*p_rad+1):
        dy = i - p_rad
        for j in range(2*p_rad+1):
            dx = j - p_rad
            if conv_func == 'gaussian':
                weight = conv_func_gauss(dy - dcoords[:,0], dx - dcoords[:,1])/norm
            elif conv_func == 'pillbox':
                weight = conv_func_pill(dy - dcoords[:,0], dx - dcoords[:,1])/norm
            elif conv_func == 'cubic':
                weight = conv_func_cubicspline(dy - dcoords[:,0], dx - dcoords[:,1])/norm

            weights[i][j] = weight

    #output the coordinates, norms, and weights
    sampler_info = SamplerInfo(order, vu2, pulsefac)
    gridder_info = GridderInfo(npad, conv_func, p_rad, coords, weights)
    return (sampler_info, gridder_info)
