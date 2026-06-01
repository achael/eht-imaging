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



try:
    import skyfield.api
    from sgp4.api import WGS72, Satrec
except ImportError:
    print("Warning: skyfield not installed: cannot simulate space VLBI")

import copy
import itertools as it
import sys
import warnings

import astropy.time as at
import finufft
import numpy as np
import scipy.ndimage as nd
import scipy.spatial.distance

import ehtim.const_def as ehc
import ehtim.observing.pol_conventions as pc
import ehtim.warnings as ehw

warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars")

##################################################################################################
# Observing & uv Functions
##################################################################################################

def compute_uv_coordinates(array, site1, site2, time, mjd, ra, dec, rf, timetype='UTC',
                           elevmin=ehc.ELEV_LOW,  elevmax=ehc.ELEV_HIGH, no_elevcut_space=False,
                           fix_theta_GMST=False, earthshadow_space=True):

    """Compute u,v coordinates for an array at a given time for a source at a given ra,dec,rf
    """

    if not isinstance(time, np.ndarray):
        time = np.array([time]).flatten()
    if not isinstance(site1, np.ndarray):
        site1 = np.array([site1]).flatten()
    if not isinstance(site2, np.ndarray):
        site2 = np.array([site2]).flatten()

    if len(site1) == len(site2) == 1:
        site1 = np.array([site1[0] for i in range(len(time))])
        site2 = np.array([site2[0] for i in range(len(time))])
    elif not (len(site1) == len(site2) == len(time)):
        raise Exception("site1, site2, and time not the same dimension in compute_uv_coordinates!")

    if ra>24 or ra<0:
        raise Exception(f'RA {ra:.2f} in compute_uv_coordinates should be in decimal hours from 0 to 24!')
    if dec>180 or dec<-180:
        raise Exception(f'DEC {dec:.2f} in compute_uv_coordinates should be in decimal degrees from -180 to 180!')

    # Source vector
    sourcevec = np.array([np.cos(dec*ehc.DEGREE), 0, np.sin(dec*ehc.DEGREE)])
    projU = np.cross(np.array([0, 0, 1]), sourcevec)
    projU = projU/np.linalg.norm(projU)
    projV = -np.cross(projU, sourcevec)

    # Wavelength
    wvl = ehc.C/rf

    if timetype == 'GMST':
        time_sidereal = time
        # time_utc = gmst_to_utc(time, mjd)
    elif timetype == 'UTC':
        time_sidereal = utc_to_gmst(time, mjd)
        # time_utc = time
    else:
        raise Exception("timetype must be UTC or GMST!")

    fracmjd = np.floor(mjd) + time/24.
    dto = (at.Time(fracmjd, format='mjd')).datetime
    theta = np.mod((time_sidereal - ra)*ehc.HOUR, 2*np.pi)
    if not isinstance(fix_theta_GMST, bool):
        theta = np.mod((fix_theta_GMST - ra)*ehc.HOUR, 2*np.pi)

    i1 = np.array([array.tkey[site] for site in site1])
    i2 = np.array([array.tkey[site] for site in site2])

    coord1 = np.vstack((array.tarr[i1]['x'], array.tarr[i1]['y'], array.tarr[i1]['z'])).T
    coord2 = np.vstack((array.tarr[i2]['x'], array.tarr[i2]['y'], array.tarr[i2]['z'])).T

    # Satellites: new method
    spacemask1 = [np.all(coord == (0., 0., 0.)) for coord in coord1]
    spacemask2 = [np.all(coord == (0., 0., 0.)) for coord in coord2]
    shadowmask1 = spacemask1.copy()
    shadowmask2 = spacemask2.copy()

    satnames = array.ephem.keys()
    satdict = {satname: sat_skyfield_from_ephementry(satname, array.ephem, mjd) for satname in satnames}
    for satname in satnames:
        sat = satdict[satname]

        mask1 = (site1==satname)
        c1 = orbit_skyfield(sat, fracmjd[mask1], whichout='itrs')
        coord1[mask1] = c1.T
        mask2 = (site2==satname)
        c2 = orbit_skyfield(sat, fracmjd[mask2], whichout='itrs')
        coord2[mask2] = c2.T

    # Satellites: old method
    """
    if np.any(spacemask1):
        if timetype == 'GMST':
            raise Exception("Spacecraft ephemeris only work with UTC!")

        site1space_list = site1[spacemask1]
        site1space_fracmjdlist = fracmjd[spacemask1]
        site1space_dtolist = dto[spacemask1]
        coord1space = []

        for k in range(len(site1space_list)):


            # old method with pyephem

            site1space = site1space_list[k]
            dto_now = site1space_dtolist[k]
            sat = ephem.readtle(array.ephem[site1space][0],
                                array.ephem[site1space][1], array.ephem[site1space][2])
            sat.compute(dto_now)  # often complains if ephemeris out of date!
            elev = sat.elevation
            lat = sat.sublat / ehc.DEGREE
            lon = sat.sublong / ehc.DEGREE
            # pyephem doesn't use an ellipsoid earth model!
            c1 = coords.EarthLocation.from_geodetic(lon, lat, elev, ellipsoid=None)
            c1 = np.array((c1.x.value, c1.y.value, c1.z.value))
            coord1space.append(c1)

        coord1space = np.array(coord1space)
        coord1[spacemask1] = coord1space

    # use spacecraft ephemeris to get position of site 2
    spacemask2 = [np.all(coord == (0., 0., 0.)) for coord in coord2]
    if np.any(spacemask2):
        if timetype == 'GMST':
            raise Exception("Spacecraft ephemeris only work with UTC!")

        site2space_list = site2[spacemask2]
        site2space_fracmjdlist = fracmjd[spacemask2]
        site2space_dtolist = dto[spacemask2]
        coord2space = []
        for k in range(len(site2space_list)):

            site2space = site2space_list[k]
            dto_now = site2space_dtolist[k]
            sat = ephem.readtle(array.ephem[site2space][0],
                               array.ephem[site2space][1],
                                array.ephem[site2space][2])
            sat.compute(dto_now)  # often complains if ephemeris out of date!
            elev = sat.elevation
            lat = sat.sublat / ehc.DEGREE
            lon = sat.sublong / ehc.DEGREE
            # pyephem doesn't use an ellipsoid earth model!
            c2 = coords.EarthLocation.from_geodetic(lon, lat, elev, ellipsoid=None)
            c2 = np.array((c2.x.value, c2.y.value, c2.z.value))
            coord2space.append(c2)

        coord2space = np.array(coord2space)
        coord2[spacemask2] = coord2space
    """

    # rotate the station coordinates with the earth
    coord1 = earthrot(coord1, theta)
    coord2 = earthrot(coord2, theta)

    # u,v coordinates
    u = np.dot((coord1 - coord2)/wvl, projU)  # u (lambda)
    v = np.dot((coord1 - coord2)/wvl, projV)  # v (lambda)

    # mask out below elevation cut
    mask_elev_1 = elevcut(coord1, sourcevec, elevmin=elevmin, elevmax=elevmax)
    mask_elev_2 = elevcut(coord2, sourcevec, elevmin=elevmin, elevmax=elevmax)

    # do NOT apply elevation cut for space orbiters
    if no_elevcut_space:
        mask_elev_1[spacemask1] = 1
        mask_elev_2[spacemask2] = 1



    # apply elevation mask
    mask = mask_elev_1 * mask_elev_2

    if earthshadow_space:
        spacevecs1 = coord1[spacemask1]
        spacevecs2 = coord2[spacemask2]

        shadowmask1 = earthshadow_mask(spacevecs1, sourcevec)
        shadowmask2 = earthshadow_mask(spacevecs2, sourcevec)
        mask[spacemask1] = mask[spacemask1]*shadowmask1
        mask[spacemask2] = mask[spacemask2]*shadowmask2

    time = time[mask]
    u = u[mask]
    v = v[mask]

    # return times and uv points where we have  data
    return (time, u, v)

def earthrot(vecs, thetas):
    """Rotate a vector / array of vectors about the z-direction by theta / array of thetas (radian)
    """

    if len(vecs.shape) == 1:
        vecs = np.array([vecs])
    if np.isscalar(thetas):
        thetas = np.array([thetas for i in range(len(vecs))])

    # equal numbers of sites and angles
    if len(thetas) == len(vecs):
        rotvec = np.array([np.dot(np.array(((np.cos(thetas[i]), -np.sin(thetas[i]), 0),
                                            (np.sin(thetas[i]), np.cos(thetas[i]), 0),
                                            (0, 0, 1))),
                                  vecs[i])
                           for i in range(len(vecs))])

    # only one rotation angle, many sites
    elif len(thetas) == 1:
        rotvec = np.array([np.dot(np.array(((np.cos(thetas[0]), -np.sin(thetas[0]), 0),
                                            (np.sin(thetas[0]), np.cos(thetas[0]), 0),
                                            (0, 0, 1))),
                                  vecs[i])
                           for i in range(len(vecs))])

    # only one site, many angles
    elif len(vecs) == 1:
        rotvec = np.array([np.dot(np.array(((np.cos(thetas[i]), -np.sin(thetas[i]), 0),
                                            (np.sin(thetas[i]), np.cos(thetas[i]), 0),
                                            (0, 0, 1))),
                                  vecs[0])
                           for i in range(len(thetas))])
    else:
        raise Exception("Unequal numbers of vectors and angles in earthrot(vecs, thetas)!")

    return rotvec


def earthshadow_mask(obsvecs, sourcevec):
    """Return a mask corresponding to obsvecs which are not Earth-shadowed
       along the line of sight to sourcevec.
       obsvec can be an array of vectors but sourcevec can ONLY be a single vector

    """

    if len(obsvecs.shape) == 1:
        obsvecs = np.array([obsvecs])
    LOS_projection = np.array([np.dot(obsvec,sourcevec) for obsvec in obsvecs])
    norms = np.array([np.linalg.norm(obsvec) for obsvec in obsvecs])
    projected_radii = np.sqrt(norms**2-LOS_projection**2)
    #Earth-shadowed points must have projected radii < earth radius and a negative LOS projection
    bad_mask = (LOS_projection<0) * (projected_radii < 6.371e6)
    return ~bad_mask


def elev(obsvecs, sourcevec):
    """Return the elevation of a source with respect to an observer/observers in radians
       obsvec can be an array of vectors but sourcevec can ONLY be a single vector
    """

    if len(obsvecs.shape) == 1:
        obsvecs = np.array([obsvecs])

    anglebtw = np.array([np.dot(obsvec, sourcevec)/np.linalg.norm(obsvec) /
                         np.linalg.norm(sourcevec) for obsvec in obsvecs])
    el = 0.5*np.pi - np.arccos(anglebtw)

    return el


def elevcut(obsvecs, sourcevec, elevmin=ehc.ELEV_LOW, elevmax=ehc.ELEV_HIGH):
    """Return True if a source is observable by a telescope vector
    """

    angles = elev(obsvecs, sourcevec)/ehc.DEGREE

    return (angles > elevmin) * (angles < elevmax)


def hr_angle(gst, lon, ra):
    """Computes the hour angle for a source at RA, observer at longitude long, and GMST time gst
       gst in hours, ra & lon ALL in radian, longitude positive east
    """

    hr_angle = np.mod(gst + lon - ra, 2*np.pi)

    return hr_angle


def par_angle(hr_angle, lat, dec):
    """Compute the parallactic angle for a source at hr_angle and dec
       for an observer with latitude lat.
       All angles in radian
    """

    num = np.sin(hr_angle)*np.cos(lat)
    denom = np.sin(lat)*np.cos(dec) - np.cos(lat)*np.sin(dec)*np.cos(hr_angle)

    return np.arctan2(num, denom)


def xyz_2_latlong(obsvecs):
    """Compute the (geocentric) latitude and longitude of a site at geocentric position x,y,z
       The output is in radians
    """

    if len(obsvecs.shape) == 1:
        obsvecs = np.array([obsvecs])
    out = []
    for obsvec in obsvecs:
        x = obsvec[0]
        y = obsvec[1]
        z = obsvec[2]
        lon = np.array(np.arctan2(y, x))
        lat = np.array(np.arctan2(z, np.sqrt(x**2+y**2)))
        out.append([lat, lon])

    out = np.array(out)

    return out

def sat_skyfield_from_elements(satname, epoch_mjd, perigee_mjd,
                               period_days, eccentricity,
                               inclination, arg_perigee, long_ascending):

    """skyfield EarthSatellite object from keplerian orbital elements
       perfect keplerian orbit is assumed, no derivatives
       epoch, pericenter given in mjd
       period given in days
       inclination, arg_perigee, long_ascending given in degrees
    """

    if not(0<=eccentricity<1):
        raise Exception("eccentricity must be between 0 and 1")
    if not(0<=inclination<=180):
        raise Exception("inclination must be between 0 and 180")
    if not(0<=arg_perigee<=180):
        raise Exception("arg_perigee must be between 0 and 180")
    if not(0<=long_ascending<=360):
        raise Exception("arg_perigee must be between 0 and 360")

    satrec = Satrec()
    ts = skyfield.api.load.timescale()
    ref_mjd = 33281. # mjd 1949 December 31 00:00 UT
    epoch_wrt_ref = epoch_mjd - ref_mjd

    inclination_rad = inclination*ehc.DEGREE
    arg_perigee_rad = arg_perigee*ehc.DEGREE
    long_ascending_rad = long_ascending*ehc.DEGREE

    mean_motion = 2*np.pi/(period_days*24.*60.) # radians/minute

    mean_anomaly = mean_motion*(epoch_mjd - perigee_mjd)
    mean_anomaly = np.mod(mean_anomaly, 2*np.pi)

    satrec.sgp4init(
        WGS72,           # gravity model
        'i',             # 'a' = old AFSPC mode, 'i' = improved mode
        1,               # satnum: Satellite number
        epoch_wrt_ref,     # epoch: days since 1949 December 31 00:00 UT
        0.0,             # bstar: drag coefficient (/earth radii)
        0.0,             # ndot: ballistic coefficient (revs/day)
        0.0,             # nddot: second derivative of mean motion (revs/day^3)
        eccentricity,    # ecco: eccentricity
        arg_perigee_rad,     # argpo: argument of perigee (radians)
        inclination_rad,     # inclo: inclination (radians)
        mean_anomaly,    # mo: mean anomaly (radians)
        mean_motion,     # no_kozai: mean motion (radians/minute)
        long_ascending_rad,    # nodeo: right ascension of ascending node (radians)
    )
    sat_skyfield = skyfield.api.EarthSatellite.from_satrec(satrec, ts)

    return sat_skyfield

def sat_skyfield_from_tle(satname, line1, line2):
    ts = skyfield.api.load.timescale()
    sat_skyfield = skyfield.api.EarthSatellite(line1, line2, satname, ts)
    return sat_skyfield

def sat_skyfield_from_ephementry(satname, ephem, epoch_mjd):
    if len(ephem[satname])==3: # TLE
        line1 = ephem[satname][1]
        line2 = ephem[satname][2]
        sat = sat_skyfield_from_tle(satname, line1, line2)
    elif len(ephem[satname])==6: #keplerian elements
        elements = ephem[satname]
        sat = sat_skyfield_from_elements(satname, epoch_mjd,
                                         elements[0],elements[1],elements[2],elements[3],elements[4],elements[5])
    else:
        raise Exception(f"ephemeris format not recognized for {satname}")

    return sat

def orbit_skyfield(sat, fracmjds, whichout='itrs'):

    """uses skyfield to propagate a earth satellite orbit and return x,y,z coordinates in co-rotating earth frame
       sat is a skyfield.sgp4lib.EarthSatellite object
       times is a list of fractional mjds
       whichout is 'itrs' (co-rotating) or 'gcrs' (fixed x-axis to equinox)
    """

    #fractional days of orbit in jd
    mjd_to_jd = 2400000.5
    ts = skyfield.api.load.timescale()
    t = ts.ut1_jd(mjd_to_jd+fracmjds)

    # propagate orbit
    time_data = sat.at(t)

    if whichout=='gcrs':
        # GCRS coordinates in km
        positions = time_data.xyz.m

    elif whichout=='itrs':
        # get coordinates in earth frame (WGS84 ellipsiod)
        geographic_position = skyfield.api.wgs84.geographic_position_of(time_data)
        positions = geographic_position.itrs_xyz.m

    else:
        raise Exception("orbit_skyfield whichout must be 'itrs' or 'gcrs'")

    return positions
##################################################################################################
# Unpack Helpers
##################################################################################################

_STOKES_VTYPES = ('vis', 'qvis', 'uvis', 'vvis')
_CIRC_VTYPES = ('rrvis', 'llvis', 'rlvis', 'lrvis')
_LIN_VTYPES = ('xxvis', 'yyvis', 'xyvis', 'yxvis')
_GENERIC_VTYPES = ('p1p1vis', 'p2p2vis', 'p1p2vis', 'p2p1vis')

# Maps a single-correlation feed-basis vtype to the generic DTPOL_MIXED slot
# that holds it on a homogeneous baseline of the matching feed basis.
_MIXED_SLOT = {'rrvis': 'p1p1vis', 'xxvis': 'p1p1vis',
               'llvis': 'p2p2vis', 'yyvis': 'p2p2vis',
               'rlvis': 'p1p2vis', 'xyvis': 'p1p2vis',
               'lrvis': 'p2p1vis', 'yxvis': 'p2p1vis',
               'pvis': 'p1p2vis'}

# field-name suffixes shared by every vis-family group
_UNPACK_SUFFIXES = ('vis', 'amp', 'phase', 'snr', 'sigma', 'sigma_phase')


def vis_component(l, vtype, polrep):
    """Return (visibility, sigma) for a single correlation `vtype` from one
       datatable `l`. Native correlations are read directly; correlations in a
       different basis are synthesized through ehtim.observing.pol_conventions
       so the basis-transform conventions live in exactly one place.
    """
    if vtype == 'pvis':  # polarized visibility == RL == Q + iU
        vtype = 'rlvis'

    if polrep == 'mixed':
        slot = _MIXED_SLOT.get(vtype)
        if slot is None:
            raise Exception("mixed-feed closures support only single-correlation "
                            f"feed-basis vtypes; got {vtype!r}")
        return l[slot], l[slot[:-3] + 'sigma']

    if polrep == 'stokes':
        if vtype in _STOKES_VTYPES:
            return l[vtype], l[vtype[:-3] + 'sigma']
        i, q, u, v = l['vis'], l['qvis'], l['uvis'], l['vvis']
        si, sq, su, sv = l['sigma'], l['qsigma'], l['usigma'], l['vsigma']
        if vtype in _CIRC_VTYPES:
            vals = dict(zip(_CIRC_VTYPES, pc.stokes_to_circ(i, q, u, v)))
            sigs = dict(zip(_CIRC_VTYPES, pc.stokes_to_circ_sigma(si, sq, su, sv)))
            return vals[vtype], sigs[vtype]
        if vtype in _LIN_VTYPES:
            vals = dict(zip(_LIN_VTYPES, pc.stokes_to_lin(i, q, u, v)))
            sigs = dict(zip(_LIN_VTYPES, pc.stokes_to_lin_sigma(si, sq, su, sv)))
            return vals[vtype], sigs[vtype]

    elif polrep == 'circ':
        if vtype in _CIRC_VTYPES:
            return l[vtype], l[vtype[:-3] + 'sigma']
        if vtype in _STOKES_VTYPES:
            rr, ll, rl, lr = l['rrvis'], l['llvis'], l['rlvis'], l['lrvis']
            srr, sll, srl, slr = l['rrsigma'], l['llsigma'], l['rlsigma'], l['lrsigma']
            vals = dict(zip(_STOKES_VTYPES, pc.circ_to_stokes(rr, ll, rl, lr)))
            sigs = dict(zip(_STOKES_VTYPES, pc.circ_to_stokes_sigma(srr, sll, srl, slr)))
            return vals[vtype], sigs[vtype]

    elif polrep == 'lin':
        if vtype in _LIN_VTYPES:
            return l[vtype], l[vtype[:-3] + 'sigma']
        if vtype in _STOKES_VTYPES:
            xx, yy, xy, yx = l['xxvis'], l['yyvis'], l['xyvis'], l['yxvis']
            sxx, syy, sxy, syx = l['xxsigma'], l['yysigma'], l['xysigma'], l['yxsigma']
            vals = dict(zip(_STOKES_VTYPES, pc.lin_to_stokes(xx, yy, xy, yx)))
            sigs = dict(zip(_STOKES_VTYPES, pc.lin_to_stokes_sigma(sxx, syy, sxy, syx)))
            return vals[vtype], sigs[vtype]

    raise Exception(f"vtype {vtype!r} not supported for polrep {polrep!r}")


def unpack_generic_slot(data, slot, polrep):
    """Return (vis, sigma) for a generic feed slot ('p1p1'/'p2p2'/'p1p2'/'p2p1')
       read directly via the DTPOL title alias. Defined for circ/lin/mixed (the
       slot aliases the physical correlation); stokes has no feed slots."""
    if polrep == 'stokes':
        raise Exception(f"unpack: generic slot {slot}vis has no meaning "
                        "for polrep 'stokes'")
    return data[slot + 'vis'], data[slot + 'sigma']


def unpack_mixed_stokes(data):
    """Recover per-row (I, Q, U, V) and their sigmas from a MIXED datatable via
       coherency_to_stokes, grouped by polbasis. Valid on every row (including
       heterogeneous baselines) under the ideal-feed assumption."""
    pb = data['polbasis']
    n = len(data)
    ivis = np.empty(n, dtype='c16')
    q = np.empty(n, dtype='c16')
    u = np.empty(n, dtype='c16')
    v = np.empty(n, dtype='c16')
    si = np.empty(n, dtype='f8')
    sq = np.empty(n, dtype='f8')
    su = np.empty(n, dtype='f8')
    sv = np.empty(n, dtype='f8')
    for code in np.unique(pb):    # loop over unique polbasis values
        m = (pb == code)
        t1f, t2f = str(code)[:2], str(code)[2:]    # t1 and t2 feed types
        iv, qv, uv, vv = pc.coherency_to_stokes(
            data['p1p1vis'][m], data['p2p2vis'][m],
            data['p1p2vis'][m], data['p2p1vis'][m], t1f, t2f)
        siv, sqv, suv, svv = pc.coherency_to_stokes_sigma(
            data['p1p1sigma'][m], data['p2p2sigma'][m],
            data['p1p2sigma'][m], data['p2p1sigma'][m], t1f, t2f)
        ivis[m], q[m], u[m], v[m] = iv, qv, uv, vv
        si[m], sq[m], su[m], sv[m] = siv, sqv, suv, svv
    return ivis, q, u, v, si, sq, su, sv


def unpack_mixed_correlation(data, feed_a, feed_b):
    """Per-row physical correlation feed_a * conj(feed_b) (e.g. 'r','r' -> RR)
       from a MIXED datatable: the matching generic slot where the row's feeds
       provide it, NaN elsewhere. Returns (vis, sigma, n_nanfilled)."""
    pb = data['polbasis']
    n = len(data)
    out = np.full(n, np.nan, dtype='c16')
    sig = np.full(n, np.nan, dtype='f8')
    n_nan = 0
    for code in np.unique(pb):    # loop over unique polbasis values
        m = (pb == code)
        # which generic slot holds this correlation on this feed pairing (or None)
        slot = pc.correlation_slot(str(code)[:2], str(code)[2:], feed_a, feed_b)
        if slot is None:
            n_nan += int(np.sum(m))
            continue
        out[m] = data[slot][m]
        sig[m] = data[slot[:-3] + 'sigma'][m]
    return out, sig, n_nan


def unpack_vis_mixed(data, field):
    """Return (out, sig, ty) for a vis-family field on a MIXED datatable.
       Row-aligned NaN-fill; warns per field when a physical correlation is
       absent on some rows. See the dispatch note in Obsdata.unpack_dat."""
    # generic slots: direct read, all rows
    for slot in ('p1p1', 'p2p2', 'p1p2', 'p2p1'):
        if field in [slot + s for s in _UNPACK_SUFFIXES]:
            return data[slot + 'vis'], data[slot + 'sigma'], 'c16'

    # physical / cross correlation names: matching slot where present, else NaN
    for visname in _CIRC_VTYPES + _LIN_VTYPES:
        pref = visname[:-3]    # 'rrvis' -> 'rr'; the two chars are the feed letters
        if field in [pref + s for s in _UNPACK_SUFFIXES]:
            out, sig, n_nan = unpack_mixed_correlation(data, pref[0], pref[1])
            if n_nan > 0:
                warnings.warn(
                    f"unpack({field!r}) on a mixed-feed obs: {n_nan} rows have "
                    f"no {pref.upper()} correlation and were returned as NaN.",
                    ehw.MixedPolUnpackNaNWarning)
            return out, sig, 'c16'

    # Stokes-derived: recover (I, Q, U, V) per row, then form the field
    ivis, q, u, v, si, sq, su, sv = unpack_mixed_stokes(data)
    if field in ['vis', 'amp', 'phase', 'snr', 'sigma', 'sigma_phase']:
        return ivis, si, 'c16'
    if field in ['qvis', 'qamp', 'qphase', 'qsnr', 'qsigma', 'qsigma_phase']:
        return q, sq, 'c16'
    if field in ['uvis', 'uamp', 'uphase', 'usnr', 'usigma', 'usigma_phase']:
        return u, su, 'c16'
    if field in ['vvis', 'vamp', 'vphase', 'vsnr', 'vsigma', 'vsigma_phase']:
        return v, sv, 'c16'
    if field in ['pvis', 'pamp', 'pphase', 'psnr', 'psigma', 'psigma_phase']:
        return q + 1j * u, np.sqrt(sq**2 + su**2), 'c16'
    if field in ['m', 'mamp', 'mphase', 'msnr', 'msigma', 'msigma_phase']:
        out = (q + 1j * u) / ivis
        return out, merr(si, sq, su, ivis, out), 'c16'
    if field in ['evis', 'eamp', 'ephase', 'esnr', 'esigma', 'esigma_phase']:
        ang = np.arctan2(data['u'], data['v'])
        out = np.cos(2 * ang) * q + np.sin(2 * ang) * u
        sig = np.sqrt(0.5 * ((np.cos(2 * ang) * sq)**2 + (np.sin(2 * ang) * su)**2))
        return out, sig, 'c16'
    if field in ['bvis', 'bamp', 'bphase', 'bsnr', 'bsigma', 'bsigma_phase']:
        ang = np.arctan2(data['u'], data['v'])
        out = -np.sin(2 * ang) * q + np.cos(2 * ang) * u
        sig = np.sqrt(0.5 * ((np.sin(2 * ang) * sq)**2 + (np.cos(2 * ang) * su)**2))
        return out, sig, 'c16'
    if field in ['rrllvis', 'rrllamp', 'rrllphase', 'rrllsnr',
                 'rrllsigma', 'rrllsigma_phase']:
        out = (ivis + v) / (ivis - v)
        sig = (2.0**0.5 * (np.abs(ivis)**2 + np.abs(v)**2)**0.5
               / np.abs(ivis - v)**2 * (si**2 + sv**2)**0.5)
        return out, sig, 'c16'

    raise Exception(f"{field} is not a valid field for polrep 'mixed'")


def unpack_vis_standard(data, field, polrep):
    """Return (out, sig, ty) for a vis-family field on a stokes/circ/lin
       datatable. Direct correlations route through vis_component (so all basis
       transforms come from pol_conventions); the derived fields (pvis/m/evis/
       bvis/rrllvis) keep their per-basis algebra. Raises for unsupported fields
       and (field, polrep) combinations."""
    if field in ['vis', 'amp', 'phase', 'snr', 'sigma', 'sigma_phase']:
        out, sig = vis_component(data, 'vis', polrep)
        return out, sig, 'c16'
    if field in ['qvis', 'qamp', 'qphase', 'qsnr', 'qsigma', 'qsigma_phase']:
        out, sig = vis_component(data, 'qvis', polrep)
        return out, sig, 'c16'
    if field in ['uvis', 'uamp', 'uphase', 'usnr', 'usigma', 'usigma_phase']:
        out, sig = vis_component(data, 'uvis', polrep)
        return out, sig, 'c16'
    if field in ['vvis', 'vamp', 'vphase', 'vsnr', 'vsigma', 'vsigma_phase']:
        out, sig = vis_component(data, 'vvis', polrep)
        return out, sig, 'c16'
    if field in ['pvis', 'pamp', 'pphase', 'psnr', 'psigma', 'psigma_phase']:
        if polrep in ('stokes', 'circ'):
            out, sig = vis_component(data, 'rlvis', polrep)  # P = RL
        elif polrep == 'lin':
            q, qsig = vis_component(data, 'qvis', 'lin')
            u, usig = vis_component(data, 'uvis', 'lin')
            out = q + 1j * u
            sig = np.sqrt(qsig**2 + usig**2)
        return out, sig, 'c16'
    if field in ['m', 'mamp', 'mphase', 'msnr', 'msigma', 'msigma_phase']:
        if polrep == 'stokes':
            out = (data['qvis'] + 1j * data['uvis']) / data['vis']
            sig = merr(data['sigma'], data['qsigma'], data['usigma'], data['vis'], out)
        elif polrep == 'circ':
            out = 2 * data['rlvis'] / (data['rrvis'] + data['llvis'])
            sig = merr2(data['rlsigma'], data['rrsigma'], data['llsigma'],
                        0.5 * (data['rrvis'] + data['llvis']), out)
        elif polrep == 'lin':
            ivis, isig = vis_component(data, 'vis', 'lin')
            q, qsig = vis_component(data, 'qvis', 'lin')
            u, usig = vis_component(data, 'uvis', 'lin')
            out = (q + 1j * u) / ivis
            sig = merr(isig, qsig, usig, ivis, out)
        return out, sig, 'c16'
    if field in ['evis', 'eamp', 'ephase', 'esnr', 'esigma', 'esigma_phase']:
        ang = np.arctan2(data['u'], data['v'])  # TODO: correct convention EofN?
        q, qsig = vis_component(data, 'qvis', polrep)
        u, usig = vis_component(data, 'uvis', polrep)
        out = (np.cos(2 * ang) * q + np.sin(2 * ang) * u)
        sig = np.sqrt(0.5 * ((np.cos(2 * ang) * qsig)**2 + (np.sin(2 * ang) * usig)**2))
        return out, sig, 'c16'
    if field in ['bvis', 'bamp', 'bphase', 'bsnr', 'bsigma', 'bsigma_phase']:
        ang = np.arctan2(data['u'], data['v'])  # TODO: correct convention EofN?
        q, qsig = vis_component(data, 'qvis', polrep)
        u, usig = vis_component(data, 'uvis', polrep)
        out = (-np.sin(2 * ang) * q + np.cos(2 * ang) * u)
        sig = np.sqrt(0.5 * ((np.sin(2 * ang) * qsig)**2 + (np.cos(2 * ang) * usig)**2))
        return out, sig, 'c16'
    if field in ['rrvis', 'rramp', 'rrphase', 'rrsnr', 'rrsigma', 'rrsigma_phase']:
        out, sig = vis_component(data, 'rrvis', polrep)
        return out, sig, 'c16'
    if field in ['llvis', 'llamp', 'llphase', 'llsnr', 'llsigma', 'llsigma_phase']:
        out, sig = vis_component(data, 'llvis', polrep)
        return out, sig, 'c16'
    if field in ['rlvis', 'rlamp', 'rlphase', 'rlsnr', 'rlsigma', 'rlsigma_phase']:
        out, sig = vis_component(data, 'rlvis', polrep)
        return out, sig, 'c16'
    if field in ['lrvis', 'lramp', 'lrphase', 'lrsnr', 'lrsigma', 'lrsigma_phase']:
        out, sig = vis_component(data, 'lrvis', polrep)
        return out, sig, 'c16'
    if field in ['xxvis', 'xxamp', 'xxphase', 'xxsnr', 'xxsigma', 'xxsigma_phase']:
        out, sig = vis_component(data, 'xxvis', polrep)
        return out, sig, 'c16'
    if field in ['yyvis', 'yyamp', 'yyphase', 'yysnr', 'yysigma', 'yysigma_phase']:
        out, sig = vis_component(data, 'yyvis', polrep)
        return out, sig, 'c16'
    if field in ['xyvis', 'xyamp', 'xyphase', 'xysnr', 'xysigma', 'xysigma_phase']:
        out, sig = vis_component(data, 'xyvis', polrep)
        return out, sig, 'c16'
    if field in ['yxvis', 'yxamp', 'yxphase', 'yxsnr', 'yxsigma', 'yxsigma_phase']:
        out, sig = vis_component(data, 'yxvis', polrep)
        return out, sig, 'c16'
    if field in ['rrllvis', 'rrllamp', 'rrllphase', 'rrllsnr',
                 'rrllsigma', 'rrllsigma_phase']:
        if polrep == 'stokes':
            out = (data['vis'] + data['vvis']) / (data['vis'] - data['vvis'])
            sig = (2.0**0.5 * (np.abs(data['vis'])**2 + np.abs(data['vvis'])**2)**0.5
                   / np.abs(data['vis'] - data['vvis'])**2
                   * (data['sigma']**2 + data['vsigma']**2)**0.5)
        elif polrep == 'circ':
            out = data['rrvis'] / data['llvis']
            sig = np.sqrt(np.abs(data['rrsigma'] / data['llvis'])**2
                          + np.abs(data['llsigma'] * data['rrvis'] / data['llvis'])**2)
        else:
            raise Exception(f"unpack: field {field!r} not supported for "
                            f"polrep {polrep!r}")
        return out, sig, 'c16'
    if field in ['p1p1vis', 'p1p1amp', 'p1p1phase', 'p1p1snr',
                 'p1p1sigma', 'p1p1sigma_phase']:
        out, sig = unpack_generic_slot(data, 'p1p1', polrep)
        return out, sig, 'c16'
    if field in ['p2p2vis', 'p2p2amp', 'p2p2phase', 'p2p2snr',
                 'p2p2sigma', 'p2p2sigma_phase']:
        out, sig = unpack_generic_slot(data, 'p2p2', polrep)
        return out, sig, 'c16'
    if field in ['p1p2vis', 'p1p2amp', 'p1p2phase', 'p1p2snr',
                 'p1p2sigma', 'p1p2sigma_phase']:
        out, sig = unpack_generic_slot(data, 'p1p2', polrep)
        return out, sig, 'c16'
    if field in ['p2p1vis', 'p2p1amp', 'p2p1phase', 'p2p1snr',
                 'p2p1sigma', 'p2p1sigma_phase']:
        out, sig = unpack_generic_slot(data, 'p2p1', polrep)
        return out, sig, 'c16'

    raise Exception(f"{field} is not a valid field \n" +
                    "valid field values are: " + ' '.join(ehc.FIELDS))


##################################################################################################
# Closure Quantity Construction
##################################################################################################


def valid_closure_vtypes(polrep):
    """Visibility types from which closure quantities can be formed for a polrep.

       Stokes is the complete basis: it can synthesize both circular and linear
       correlations. Each feed basis can give its own correlations plus Stokes,
       but not the other feed basis directly (that would need the two-step
       circ<->stokes<->lin composition, which is out of scope for the kernels).
    """
    if polrep == 'stokes':
        return _STOKES_VTYPES + _CIRC_VTYPES + _LIN_VTYPES
    if polrep == 'circ':
        return _CIRC_VTYPES + _STOKES_VTYPES
    if polrep == 'lin':
        return _LIN_VTYPES + _STOKES_VTYPES
    if polrep == 'mixed':
        # Permissive: feed-basis names pass here so the per-triangle filter
        # (closure_skip_polbasis) can reject Stokes/generic vtypes in the body.
        return _STOKES_VTYPES + _CIRC_VTYPES + _LIN_VTYPES + _GENERIC_VTYPES
    raise Exception(f"unsupported polrep {polrep!r} for closure quantities")


def closure_skip_polbasis(vtype):
    """For a mixed-feed observation, the homogeneous per-baseline polbasis that a
       closure of this vtype requires. Raises for vtypes with no single-baseline
       feed-basis meaning (Stokes or generic p1p1vis-style)."""
    if vtype in _CIRC_VTYPES or vtype == 'pvis':
        return 'rlrl'
    if vtype in _LIN_VTYPES:
        return 'xyxy'
    raise Exception(
        f"closure quantity vtype={vtype!r} is not physical on a mixed-feed "
        "observation: only single-correlation feed-basis vtypes "
        "(rrvis/llvis/rlvis/lrvis or xxvis/yyvis/xyvis/yxvis) can be formed per "
        "triangle. Stokes and generic p1p1vis-style closures need a homogeneous "
        "feed basis; use switch_polrep('stokes') after calibration instead.")


def closure_skip_message(n_skipped, count, kind='triangles'):
    """Message for MixedPolClosureSkipWarning when feed-mixing baselines are
       dropped. For minimal sets, note that the (site-based) minimal set is not
       baseline-aware, so some recoverable closures may be lost."""
    msg = (f"{n_skipped} {kind} skipped: closures require baselines between "
           "stations with the same polarization basis.")
    if count in ('min', 'min-cut0bl'):
        msg += " Minimal set is site-based, not baseline-aware; use count='max' to keep all."
    return msg


def make_bispectrum(l1, l2, l3, vistype, polrep='stokes'):
    """Make a list of bispectra and errors
       l1,l2,l3 are full datatables of visibility entries
       vtype is visibility types
    """

    # Per-correlation values and sigmas (synthesis routed via pol_conventions)
    p1, s1 = vis_component(l1, vistype, polrep)
    p2, s2 = vis_component(l2, vistype, polrep)
    p3, s3 = vis_component(l3, vistype, polrep)

    var1, var2, var3 = s1**2, s2**2, s3**2

    # Make the bispectrum and its uncertainty
    bi = p1*p2*p3
    bisig = np.abs(bi) * np.sqrt(var1/np.abs(p1)**2 +
                                 var2/np.abs(p2)**2 +
                                 var3/np.abs(p3)**2)

    return (bi, bisig)


def make_closure_amplitude(blue1, blue2, red1, red2, vtype,
                           ctype='camp', debias=True, polrep='stokes'):
    """Make a list of closure amplitudes and errors
       blue1 and blue2 are full datatables numerator entries
       red1 and red2 are full datatables of denominator entries
       vtype is the  visibility type
    """

    if ctype not in ['camp', 'logcamp']:
        raise Exception("closure amplitude type must be 'camp' or 'logcamp'!")

    # Per-correlation values and sigmas (synthesis routed via pol_conventions).
    # blue1/blue2 are numerator baselines; red1/red2 are denominator baselines.
    cblue1, sigblue1 = vis_component(blue1, vtype, polrep)
    cblue2, sigblue2 = vis_component(blue2, vtype, polrep)
    cred1, sigred1 = vis_component(red1, vtype, polrep)
    cred2, sigred2 = vis_component(red2, vtype, polrep)

    # Debias the amplitude
    if debias:
        pblue1 = amp_debias(np.abs(cblue1), sigblue1, force_nonzero=True)
        pblue2 = amp_debias(np.abs(cblue2), sigblue2, force_nonzero=True)
        pred1 = amp_debias(np.abs(cred1), sigred1, force_nonzero=True)
        pred2 = amp_debias(np.abs(cred2), sigred2, force_nonzero=True)
    else:
        pblue1 = np.abs(cblue1)
        pblue2 = np.abs(cblue2)
        pred1 = np.abs(cred1)
        pred2 = np.abs(cred2)

    # Get snrs
    snrblue1 = pblue1/sigblue1
    snrblue2 = pblue2/sigblue2
    snrred1 = pred1/sigred1
    snrred2 = pred2/sigred2

    # Compute the closure amplitude and its uncertainty
    if ctype == 'camp':
        camp = np.abs((pblue1*pblue2)/(pred1*pred2))
        camperr = camp * np.sqrt(1./(snrblue1**2) + 1./(snrblue2**2) +
                                 1./(snrred1**2) + 1./(snrred2**2))

        # Debias
        if debias:
            camp = camp_debias(camp, snrred1, snrred2)

    elif ctype == 'logcamp':
        camp = (np.log(np.abs(pblue1)) + np.log(np.abs(pblue2)) -
                np.log(np.abs(pred1)) - np.log(np.abs(pred2)))
        camperr = np.sqrt(1./(snrblue1**2) + 1./(snrblue2**2) +
                          1./(snrred1**2) + 1./(snrred2**2))

        # Debias
        if debias:
            camp = logcamp_debias(camp, snrblue1, snrblue2, snrred1, snrred2)

    return (camp, camperr)

def tri_minimal_set(sites, tarr, tkey):
    """returns a minimal set of triangles for bispectra and closure phase"""

    # determine ordering and reference site based on order of  self.tarr
    sites_ordered = [x for x in tarr['site'] if x in sites]
    ref = sites_ordered[0]
    sites_ordered.remove(ref)

    # Find all triangles that contain the ref
    tris = list(it.combinations(sites_ordered, 2))
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
            if j == i-1:
                k = 1
            else:
                k = j+1

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
    if type(datarr) is not list:
        datalist = []
        dtype = datarr.dtype
        for key, group in it.groupby(datarr, lambda x: x['time']):
            datalist.append(np.array([gp for gp in group], dtype=dtype))
        returnType = 'all'
    else:
        dtype = datarr[0].dtype
        datalist = datarr
        returnType = 'time'

    out = []

    for timegroup in datalist:
        if returnType == 'all':
            outgroup = out
        else:
            outgroup = []

        # determine a minimal set of trinagles
        sites = list(set(np.hstack((timegroup['t1'], timegroup['t2'], timegroup['t3']))))
        tris = tri_minimal_set(sites, obs.tarr, obs.tkey)
        tris = [set(tri) for tri in tris]

        # add data points from original array to new array if in minimal set
        for dp in timegroup:
            # TODO: sign flips?
            if set((dp['t1'], dp['t2'], dp['t3'])) in tris:
                outgroup.append(dp)

        if returnType == 'time':
            out.append(np.array(outgroup, dtype=dtype))
        else:
            out = outgroup

    if returnType == 'all':
        out = np.array(out, dtype=dtype)
    return out

# TODO This returns A minimal set if input is maximal, but it is not necessarily the same
# minimal set as we would from  calling c_amplitudes(count='min'). This is because of  inverses.


def reduce_quad_minimal(obs, datarr, ctype='camp'):
    """Reduce a closure amplitude or log closure amplitude array
       FROM a maximal set TO a minimal set
    """

    if ctype not in ['camp', 'logcamp']:
        raise Exception("ctype must be 'camp' or 'logcamp'")

    # time sort or not
    if type(datarr) is not list:
        datalist = []
        dtype = datarr.dtype
        for key, group in it.groupby(datarr, lambda x: x['time']):
            datalist.append(np.array([x for x in group]))
        returnType = 'all'
    else:
        dtype = datarr[0].dtype
        datalist = datarr
        returnType = 'time'

    out = []
    for timegroup in datalist:
        if returnType == 'all':
            outgroup = out
        else:
            outgroup = []

        # determine a minimal set of quadrangles
        sites = np.array(list(set(np.hstack((timegroup['t1'],
                                             timegroup['t2'],
                                             timegroup['t3'],
                                             timegroup['t4'])))))
        if len(sites) < 4:
            continue
        quads = quad_minimal_set(sites, obs.tarr, obs.tkey)

        # add data points from original camp array to new array if in minimal set
        # ANDREW TODO: do we need to change the ordering ??
        for dp in timegroup:

            # this is all same closure amplitude, but the ordering of labels is different
            if ((dp['t1'], dp['t2'], dp['t3'], dp['t4']) in quads or
                (dp['t2'], dp['t1'], dp['t4'], dp['t3']) in quads or
                (dp['t3'], dp['t4'], dp['t1'], dp['t2']) in quads or
                    (dp['t4'], dp['t3'], dp['t2'], dp['t1']) in quads):

                outgroup.append(np.array(dp, dtype=ehc.DTCAMP))

            # flip the inverse closure amplitude
            elif ((dp['t1'], dp['t4'], dp['t3'], dp['t2']) in quads or
                  (dp['t2'], dp['t3'], dp['t4'], dp['t1']) in quads or
                  (dp['t3'], dp['t2'], dp['t1'], dp['t4']) in quads or
                  (dp['t4'], dp['t1'], dp['t2'], dp['t3']) in quads):

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

                if ctype == 'camp':
                    dp2['camp'] = 1./campold
                    dp2['sigmaca'] = sigmaold/(campold**2)

                elif ctype == 'logcamp':
                    dp2['camp'] = -campold
                    dp2['sigmaca'] = sigmaold

                outgroup.append(dp2)

        if returnType == 'time':
            out.append(np.array(outgroup, dtype=dtype))
        else:
            out = outgroup

    if returnType == 'all':
        out = np.array(out, dtype=dtype)
    return out

##################################################################################################
# Debiasing Functions
##################################################################################################
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

##################################################################################################
# Scattering Functions
##################################################################################################

def gauss_uv(u, v, flux, beamparams, x=0., y=0.):
    """Return the value of the Gaussian FT with
       beamparams is [FWHMmaj, FWHMmin, theta, x, y], all in radian
       x,y are the center coordinates
    """

    sigma_maj = beamparams[0]/(2*np.sqrt(2*np.log(2)))
    sigma_min = beamparams[1]/(2*np.sqrt(2*np.log(2)))
    theta = -beamparams[2]  # theta needs to be negative in this convention!

    # Covariance matrix
    a = (sigma_min * np.cos(theta))**2 + (sigma_maj*np.sin(theta))**2
    b = (sigma_maj * np.cos(theta))**2 + (sigma_min*np.sin(theta))**2
    c = (sigma_min**2 - sigma_maj**2) * np.cos(theta) * np.sin(theta)
    m = np.array([[a, c], [c, b]])

    uv = np.array([[u[i], v[i]] for i in range(len(u))])
    x2 = np.array([np.dot(uvi, np.dot(m, uvi)) for uvi in uv])

    g = np.exp(-2 * np.pi**2 * x2)
    p = np.exp(-2j * np.pi * (u*x + v*y))

    return flux * g * p


def rbf_kernel_covariance(x, sigma):
    """Compute a covariance matrix from an RBF kernel

    Args:
        x (ndarray): 1D data points for which to compute the covariance
        sigma (float): std for the covariance. Controls correlation length / time.

    Returns:
       cov (ndarray): Covariance matrix
    """
    x = np.expand_dims(x, 1) if x.ndim == 1 else x
    norm = -0.5 * scipy.spatial.distance.cdist(x, x, 'sqeuclidean') / sigma**2
    cov = np.exp(norm)
    cov *= 1.0 / cov.sum(axis=0)
    return cov


def sgra_kernel_uv(rf, u, v):
    """Return the value of the Sgr A* scattering kernel at a given u,v (in lambda)

    Args:
        rf (float): The observation frequency in Hz
        u (float or ndarray): an array of u coordinates
        v (float or ndarray): an array of v coordinates

    Returns:
       g (float ndarray): Sgr A* scattering kernel
    """
    u = np.array(u)
    v = np.array(v)
    assert u.size == v.size, 'u and v should have the same size'

    lcm = (ehc.C / rf) * 100  # in cm
    sigma_maj = ehc.FWHM_MAJ * (lcm ** 2) / (2 * np.sqrt(2 * np.log(2))) * ehc.RADPERUAS
    sigma_min = ehc.FWHM_MIN * (lcm ** 2) / (2 * np.sqrt(2 * np.log(2))) * ehc.RADPERUAS
    theta = -ehc.POS_ANG * ehc.DEGREE  # theta needs to be negative in this convention!

    # Covariance matrix
    a = (sigma_min * np.cos(theta)) ** 2 + (sigma_maj * np.sin(theta)) ** 2
    b = (sigma_maj * np.cos(theta)) ** 2 + (sigma_min * np.sin(theta)) ** 2
    c = (sigma_min ** 2 - sigma_maj ** 2) * np.cos(theta) * np.sin(theta)
    m = np.array([[a, c], [c, b]])
    uv = np.array([u, v])

    x2 = (uv * np.dot(m, uv)).sum(axis=0)
    g = np.exp(-2 * np.pi ** 2 * x2)

    return g


def sgra_kernel_params(rf):
    """Return elliptical gaussian parameters in radian for the Sgr A* scattering ellipse
       at a given frequency rf
    """

    lcm = (ehc.C/rf) * 100  # in cm
    fwhm_maj_rf = ehc.FWHM_MAJ * (lcm**2) * ehc.RADPERUAS
    fwhm_min_rf = ehc.FWHM_MIN * (lcm**2) * ehc.RADPERUAS
    theta = ehc.POS_ANG * ehc.DEGREE

    return np.array([fwhm_maj_rf, fwhm_min_rf, theta])

##################################################################################################
# Noise Functions
##################################################################################################

def blnoise(sefd1, sefd2, tint, bw):
    """Determine the standard deviation of Gaussian thermal noise on a baseline
       This is the noise on the rr/ll/rl/lr product, not the Stokes parameter
       2-bit quantization is responsible for the 0.88 factor
    """

    noise = np.sqrt(sefd1*sefd2/(2*bw*tint))/0.88
    # noise = np.sqrt(sefd1*sefd2/(bw*tint))/0.88

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

    noise = np.random.normal(loc=0, scale=sigma) + 1j*np.random.normal(loc=0, scale=sigma)
    return noise


def cerror_hash(sigma, *args):
    """Return a complex number drawn from a circular complex Gaussian of zero mean
    """

    reargs = list(args)
    reargs.append('re')
    np.random.seed(hash(",".join(map(repr, reargs))) % 4294967295)
    re = np.random.randn()

    imargs = list(args)
    imargs.append('im')
    np.random.seed(hash(",".join(map(repr, imargs))) % 4294967295)
    im = np.random.randn()

    err = sigma * (re + 1j*im)

    return err


def hashmultivariaterandn(size, cov, *args):
    """set the seed according to a collection of arguments and return random multivariate gaussian var
    """
    np.random.seed(hash(",".join(map(repr, args))) % 4294967295)
    mean = np.zeros(size)
    noise = np.random.multivariate_normal(mean, cov, check_valid='ignore')
    return noise


def hashrandn(*args):
    """set the seed according to a collection of arguments and return random gaussian var
    """

    np.random.seed(hash(",".join(map(repr, args))) % 4294967295)
    noise = np.random.randn()
    return noise


def hashrand(*args):
    """set the seed according to a collection of arguments and return random number in 0,1
    """

    np.random.seed(hash(",".join(map(repr, args))) % 4294967295)
    noise = np.random.rand()
    return noise

##################################################################################################
# Time Functions
##################################################################################################

def gmst_to_utc(gmst, mjd):
    """Convert gmst times in hours to utc hours using astropy.

    Inverse of :func:`utc_to_gmst` on the canonical solar day of ``mjd``:
    returns UTC in [0, 24). The local sidereal rate is sampled directly
    from astropy across that day (mean GMST is polynomial in UT, so
    within a day it's linear to ~1e-14 hours -- no iteration needed).

    Note: an obs that spans > ~23.93 solar hours aliases in GMST and
    cannot be uniquely round-tripped through this inverse; callers
    must keep their obs inside a single sidereal day.
    """

    mjd = int(mjd)
    t0 = at.Time(mjd, format='mjd', scale='utc')
    t1 = at.Time(mjd + 1, format='mjd', scale='utc')
    s0 = t0.sidereal_time('mean', 'greenwich').hour
    s1 = t1.sidereal_time('mean', 'greenwich').hour
    # Sidereal hours elapsed in 24 solar hours (~24.0657)
    sidereal_per_day = ((s1 - s0) % 24.0) + 24.0
    # Wrap into [0, 24) sidereal so utc lands in [0, 24) of mjd
    delta_sidereal = (gmst - s0) % 24.0
    time_utc = delta_sidereal * 24.0 / sidereal_per_day

    return time_utc


def utc_to_gmst(utc, mjd):
    """Convert utc times in hours to gmst using astropy
    """

    mjd = int(mjd)  # MJD should always be an integer, but was float in older versions of the code
    time_obj = at.Time(utc/24.0 + np.floor(mjd), format='mjd', scale='utc')
    time_sidereal = time_obj.sidereal_time('mean', 'greenwich').hour

    return time_sidereal

##################################################################################################
# DFT Functions
##################################################################################################

def image_centroid(im):
    """Return the image centroid (in radians)
    """

    xlist = np.arange(0, -im.xdim, -1)*im.psize + (im.psize*im.xdim)/2.0 - im.psize/2.0
    ylist = np.arange(0, -im.ydim, -1)*im.psize + (im.psize*im.ydim)/2.0 - im.psize/2.0

    x0 = np.sum(np.outer(0.0*ylist+1.0, xlist).ravel()*im.imvec)/np.sum(im.imvec)
    y0 = np.sum(np.outer(ylist, 0.0*xlist+1.0).ravel()*im.imvec)/np.sum(im.imvec)

    return np.array([x0, y0])


def ftmatrix(pdim, xdim, ydim, uvlist, pulse=ehc.PULSE_DEFAULT, mask=[]):
    """Return a DFT matrix for the xdim*ydim image with pixel width pdim
       that extracts spatial frequencies of the uv points in uvlist.
    """

    xlist = np.arange(0, -xdim, -1)*pdim + (pdim*xdim)/2.0 - pdim/2.0
    ylist = np.arange(0, -ydim, -1)*pdim + (pdim*ydim)/2.0 - pdim/2.0

    # original sign convention
    # ftmatrices = [pulse(2*np.pi*uv[0], 2*np.pi*uv[1], pdim, dom="F") *
    #              np.outer(np.exp(-2j*np.pi*ylist*uv[1]), np.exp(-2j*np.pi*xlist*uv[0]))
    #              for uv in uvlist]

    # changed the sign convention to agree with BU data (Jan 2017)
    # this is correct for a u,v definition from site 1-2 as (x1-x2)/lambda
    ftmatrices = [pulse(2*np.pi*uv[0], 2*np.pi*uv[1], pdim, dom="F") *
                  np.outer(np.exp(2j*np.pi*ylist*uv[1]), np.exp(2j*np.pi*xlist*uv[0]))
                  for uv in uvlist]
    ftmatrices = np.reshape(np.array(ftmatrices), (len(uvlist), xdim*ydim))

    if len(mask):
        ftmatrices = ftmatrices[:, mask]

    return ftmatrices


def ftmatrix_centered(im, pdim, xdim, ydim, uvlist, pulse=ehc.PULSE_DEFAULT):
    """Return a DFT matrix for the xdim*ydim image with pixel width pdim
       that extracts spatial frequencies of the uv points in uvlist.
       in this version, it puts the image centroid at the origin
    """

    # TODO : there is a residual value for the center being around 0,
    # maybe we should chop this off to be exactly 0?
    xlist = np.arange(0, -xdim, -1)*pdim + (pdim*xdim)/2.0 - pdim/2.0
    ylist = np.arange(0, -ydim, -1)*pdim + (pdim*ydim)/2.0 - pdim/2.0
    x0 = np.sum(np.outer(0.0*ylist+1.0, xlist).ravel()*im)/np.sum(im)
    y0 = np.sum(np.outer(ylist, 0.0*xlist+1.0).ravel()*im)/np.sum(im)

    # Now shift the lists
    xlist = xlist - x0
    ylist = ylist - y0

    # list of matrices at each spatial freq
    ftmatrices = [pulse(2*np.pi*uv[0], 2*np.pi*uv[1], pdim, dom="F") *
                  np.outer(np.exp(-2j*np.pi*ylist*uv[1]), np.exp(-2j*np.pi*xlist*uv[0]))
                  for uv in uvlist]
    ftmatrices = np.reshape(np.array(ftmatrices), (len(uvlist), xdim*ydim))
    return ftmatrices

##################################################################################################
# FFT & NFFT helper functions
##################################################################################################
class FINUFFTPlan:
    """Stateful 2D NFFT at fixed nonuniform (u, v) points.

    Set .f_hat (shape (xdim, ydim), complex128) and call .trafo() for the
    forward transform; the result lands in .f. Set .f (length len(uv),
    complex128) and call .adjoint(); the result lands in .f_hat.

    The .f_hat / .f / .trafo() / .adjoint() API surface is preserved for
    backwards compatibility with the old pynfft.NFFT implementation, so
    every existing NFFT consumer in the codebase stays untouched.
    """

    def __init__(self, xdim, ydim, uv_finufft, eps=ehc.NFFT_EPS_DEFAULT):
        x = np.ascontiguousarray(uv_finufft[:, 0])
        y = np.ascontiguousarray(uv_finufft[:, 1])
        self._fwd = finufft.Plan(2, (xdim, ydim), n_trans=1, eps=eps,
                                 isign=-1, dtype='complex128')
        self._fwd.setpts(x, y)
        self._adj = finufft.Plan(1, (xdim, ydim), n_trans=1, eps=eps,
                                 isign=+1, dtype='complex128')
        self._adj.setpts(x, y)
        self.f_hat = None
        self.f = None

    def trafo(self):
        # pynfft silently promoted real f_hat to complex; finufft requires
        # matching dtype, so cast here.
        self.f = self._fwd.execute(np.ascontiguousarray(self.f_hat, dtype='complex128'))

    def adjoint(self):
        self.f_hat = self._adj.execute(np.ascontiguousarray(self.f, dtype='complex128'))


class NFFTInfo:
    """Precomputed NFFT plan + per-point pulse/centering factor.

    eps is the requested relative accuracy of the NFFT. Default 1e-9 is
    safe for high-dynamic-range imaging (ALMA polarimetry, SKA-scale
    arrays). Tighten to 1e-12 for ~1e6 dynamic range; relax to 1e-6 for
    faster low-SNR work where data noise dominates.

    npad and p_rad are accepted for backwards compatibility with the
    old pynfft.NFFT implementation but are unused under finufft, which
    chooses its own oversampling and kernel width from eps.
    """

    def __init__(self, xdim, ydim, psize, pulse, npad, p_rad, uv,
                 eps=ehc.NFFT_EPS_DEFAULT):
        self.xdim = int(xdim)
        self.ydim = int(ydim)
        self.psize = psize
        self.pulse = pulse
        self.npad = int(npad)
        self.p_rad = int(p_rad)
        self.uv = uv
        self.uvdim = len(uv)

        # uv in (-0.5, 0.5] is used by the centering phase below;
        # (-pi, pi] form is what FINUFFTPlan expects.
        uv_scaled = uv * psize
        uv_finufft = 2 * np.pi * uv_scaled
        self.plan = FINUFFTPlan(self.xdim, self.ydim, uv_finufft, eps=eps)

        phases = np.exp(-1j*np.pi*(uv_scaled[:, 0] + uv_scaled[:, 1]))
        pulses = np.fromiter((pulse(2*np.pi*uv_scaled[i, 0], 2*np.pi*uv_scaled[i, 1], 1., dom="F")
                              for i in range(self.uvdim)), 'c16')
        self.pulsefac = pulses * phases

class SamplerInfo:
    def __init__(self, order, uv, pulsefac):
        self.order = int(order)
        self.uv = uv
        self.pulsefac = pulsefac


class GridderInfo:
    def __init__(self, npad, func, p_rad, coords, weights):
        self.npad = int(npad)
        self.conv_func = func
        self.p_rad = int(p_rad)
        self.coords = coords
        self.weights = weights


class ImInfo:
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


def conv_func_pill(x, y):
    if abs(x) < 0.5 and abs(y) < 0.5:
        out = 1.
    else:
        out = 0.
    return out


def conv_func_gauss(x, y):
    return np.exp(-(x**2 + y**2))


def conv_func_cubicspline(x, y):
    if abs(x) <= 1:
        fx = 1.5*abs(x)**3 - 2.5*abs(x)**2 + 1
    elif abs(x) < 2:
        fx = -0.5*abs(x)**3 + 2.5*abs(x)**2 - 4*abs(x) + 2
    else:
        fx = 0

    if abs(y) <= 1:
        fy = 1.5*abs(y)**3 - 2.5*abs(y)**2 + 1
    elif abs(y) < 2:
        fy = -0.5*abs(y)**3 + 2.5*abs(y)**2 - 4*abs(y) + 2
    else:
        fy = 0

    return fx*fy

# There's a bug in scipy spheroidal function of order 0! - gives nans for eta<1
# def conv_func_spheroidal(x,y,p,m):
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
    # imarr.shape is (ydim, xdim). First axis (rows) is y, second (cols) is x.
    # So padvaly pads the y-axis and padvalx pads the x-axis.
    imarr = np.pad(imarr, ((padvaly1, padvaly2), (padvalx1, padvalx2)),
                   'constant', constant_values=0.0)

    if imarr.shape[0] != imarr.shape[1]:
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
    if sample_type not in ["vis", "bs", "camp"]:
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

    if sample_type == "vis":
        out = dataset[0]
    if sample_type == "bs":
        out = dataset[0]*dataset[1]*dataset[2]
    if sample_type == "camp":
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


def make_gridder_and_sampler_info(im_info, uv, conv_func=ehc.GRIDDER_CONV_FUNC_DEFAULT,
                                  p_rad=ehc.GRIDDER_P_RAD_DEFAULT, order=ehc.FFT_INTERP_DEFAULT):
    """
    Prep norms and weights for gridding data sampled at uv points on a square array
    im_info tuple contains (xdim, ydim, npad, psize, pulse) of the grid
    conv_func is the convolution function: current options are "pillbox", "gaussian"
    p_rad is the pixel radius inside wich the conv_func is nonzero
    """

    if conv_func not in ['pillbox', 'gaussian', 'cubic']:
        raise Exception("conv_func must be either 'pillbox', 'gaussian', or, 'cubic'")

    npad = im_info.npad
    psize = im_info.psize
    pulse = im_info.pulse

    # compute grid u,v coordinates
    vu2 = np.hstack((uv[:, 1].reshape(-1, 1), uv[:, 0].reshape(-1, 1)))
    du = 1.0/(npad*psize)
    vu2 = (vu2/du + 0.5*npad)

    coords = np.round(vu2).astype(int)
    dcoords = vu2 - np.round(vu2).astype(int)
    vu2 = vu2.T

    # TODO: phase rotations should be done separately for x and y if the image isn't square
    # e.g.,
    phase = np.exp(-1j*np.pi*psize*((1+im_info.xdim % 2)*uv[:, 0] + (1+im_info.ydim % 2)*uv[:, 1]))

    pulsefac = np.fromiter(
        (pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], psize, dom="F") for uvpt in uv), 'c16')
    pulsefac = pulsefac * phase

    # compute gridder norm
    weights = []
    norm = np.zeros_like(len(coords))
    for i in range(2*p_rad+1):
        weights.append([])
        dy = i - p_rad
        for j in range(2*p_rad+1):
            dx = j - p_rad
            if conv_func == 'gaussian':
                norm = norm + conv_func_gauss(dy - dcoords[:, 0], dx - dcoords[:, 1])
            elif conv_func == 'pillbox':
                norm = norm + conv_func_pill(dy - dcoords[:, 0], dx - dcoords[:, 1])
            elif conv_func == 'cubic':
                norm = norm + conv_func_cubicspline(dy - dcoords[:, 0], dx - dcoords[:, 1])

            weights[i].append(None)

    # compute weights for gridding
    for i in range(2*p_rad+1):
        dy = i - p_rad
        for j in range(2*p_rad+1):
            dx = j - p_rad
            if conv_func == 'gaussian':
                weight = conv_func_gauss(dy - dcoords[:, 0], dx - dcoords[:, 1])/norm
            elif conv_func == 'pillbox':
                weight = conv_func_pill(dy - dcoords[:, 0], dx - dcoords[:, 1])/norm
            elif conv_func == 'cubic':
                weight = conv_func_cubicspline(dy - dcoords[:, 0], dx - dcoords[:, 1])/norm

            weights[i][j] = weight

    # output the coordinates, norms, and weights
    sampler_info = SamplerInfo(order, vu2, pulsefac)
    gridder_info = GridderInfo(npad, conv_func, p_rad, coords, weights)
    return (sampler_info, gridder_info)


##################################################################################################
# miscellaneous functions
##################################################################################################

# TODO this makes a copy -- is there a faster robust way?
def recarr_to_ndarr(x, typ):
    """converts a record array x to a normal ndarray with all fields converted to datatype typ
    """

    fields = x.dtype.names
    shape = x.shape + (len(fields),)
    dt = [(name, typ) for name in fields]
    y = x.astype(dt).view(typ).reshape(shape)
    return y

def qimage(iimage, mimage, chiimage):
    """Return the Q image from m and chi"""
    return iimage * mimage * np.cos(2*chiimage)


def uimage(iimage, mimage, chiimage):
    """Return the U image from m and chi"""
    return iimage * mimage * np.sin(2*chiimage)


def ticks(axisdim, psize, nticks=8):
    """Return a list of ticklocs and ticklabels
       psize should be in desired units
    """

    axisdim = int(axisdim)
    nticks = int(nticks)
    if not axisdim % 2:
        axisdim += 1
    if nticks % 2:
        nticks -= 1
    tickspacing = float(axisdim-1)/nticks
    ticklocs = np.arange(0, axisdim+1, tickspacing) - 0.5
    ticklabels = np.around(psize * np.arange((axisdim-1)/2.0, -
                                             (axisdim)/2.0, -tickspacing), decimals=1)

    return (ticklocs, ticklabels)


def power_of_two(target):
    """Finds the next greatest power of two
    """
    cur = 1
    if target > 1:
        for i in range(0, int(target)):
            if (cur >= target):
                return cur
            else:
                cur *= 2
    else:
        return 1


def paritycompare(perm1, perm2):
    """Compare the parity of two permutations.
       Assume both lists are equal length and with same elements
       Copied from: http://stackoverflow.com/questions/1503072/how-to-check-if-permutations-have-equal-parity
    """

    perm2 = list(perm2)
    perm2_map = dict((v, i) for i, v in enumerate(perm2))
    transCount = 0
    for loc, p1 in enumerate(perm1):
        p2 = perm2[loc]
        if p1 != p2:
            sloc = perm2_map[p1]
            perm2[loc], perm2[sloc] = p1, p2
            perm2_map[p1], perm2_map[p2] = sloc, loc
            transCount += 1

    if not (transCount % 2):
        return 1
    else:
        return -1


def sigtype(datatype):
    """Return the type of noise corresponding to the data type
    """

    datatype = str(datatype)
    if datatype in ['vis', 'amp']:
        sigmatype = 'sigma'
    elif datatype in ['qvis', 'qamp']:
        sigmatype = 'qsigma'
    elif datatype in ['uvis', 'uamp']:
        sigmatype = 'usigma'
    elif datatype in ['vvis', 'vamp']:
        sigmatype = 'vsigma'
    elif datatype in ['pvis', 'pamp']:
        sigmatype = 'psigma'
    elif datatype in ['evis', 'eamp']:
        sigmatype = 'esigma'
    elif datatype in ['bvis', 'bamp']:
        sigmatype = 'esigma'
    elif datatype in ['rrvis', 'rramp']:
        sigmatype = 'rrsigma'
    elif datatype in ['llvis', 'llamp']:
        sigmatype = 'llsigma'
    elif datatype in ['rlvis', 'rlamp']:
        sigmatype = 'rlsigma'
    elif datatype in ['lrvis', 'lramp']:
        sigmatype = 'lrsigma'
    elif datatype in ['rrllvis', 'rrllamp']:
        sigmatype = 'rrllsigma'
    elif datatype in ['m', 'mamp']:
        sigmatype = 'msigma'
    elif datatype in ['phase']:
        sigmatype = 'sigma_phase'
    elif datatype in ['qphase']:
        sigmatype = 'qsigma_phase'
    elif datatype in ['uphase']:
        sigmatype = 'usigma_phase'
    elif datatype in ['vphase']:
        sigmatype = 'vsigma_phase'
    elif datatype in ['pphase']:
        sigmatype = 'psigma_phase'
    elif datatype in ['ephase']:
        sigmatype = 'esigma_phase'
    elif datatype in ['bphase']:
        sigmatype = 'bsigma_phase'
    elif datatype in ['mphase']:
        sigmatype = 'msigma_phase'
    elif datatype in ['rrphase']:
        sigmatype = 'rrsigma_phase'
    elif datatype in ['llphase']:
        sigmatype = 'llsigma_phase'
    elif datatype in ['rlphase']:
        sigmatype = 'rlsigma_phase'
    elif datatype in ['lrphase']:
        sigmatype = 'lrsigma_phase'
    elif datatype in ['rrllphase']:
        sigmatype = 'rrllsigma_phase'

    else:
        sigmatype = False

    return sigmatype


def rastring(ra):
    """Convert a ra in fractional hours to formatted string
    """
    h = int(ra)
    m = int((ra-h)*60.)
    s = (ra-h-m/60.)*3600.
    out = f"{h:2d} h {m:2d} m {s:2.4f} s"

    return out


def decstring(dec):
    """Convert a dec in fractional degrees to formatted string
    """

    deg = int(dec)
    m = int((abs(dec)-abs(deg))*60.)
    s = (abs(dec)-abs(deg)-m/60.)*3600.
    out = f"{deg:2d} deg {m:2d} m {s:2.4f} s"

    return out


def gmtstring(gmt):
    """Convert a gmt in fractional hours to formatted string
    """

    if gmt > 24.0:
        gmt = gmt-24.0
    h = int(gmt)
    m = int((gmt-h)*60.)
    s = (gmt-h-m/60.)*3600.
    out = f"{h:02d}:{m:02d}:{s:2.4f}"

    return out

def prog_msg(nscan, totscans, msgtype='bar', nscan_last=0):
    """print a progress method for calibration
    """
    complete_percent_last = int(100*float(nscan_last)/float(totscans))
    complete_percent = int(100*float(nscan)/float(totscans))
    ndigit = str(len(str(totscans)))

    if msgtype == 'bar':
        bar_width = 30
        progress = int(bar_width * complete_percent/float(100))
        barparams = (nscan, totscans, ("-"*progress) +
                     (" " * (bar_width-progress)), complete_percent)

        printstr = "\rScan %0"+ndigit+"i/%i : [%s]%i%%"
        sys.stdout.write(printstr % barparams)
        sys.stdout.flush()

    elif msgtype == 'bar2':
        bar_width = 30
        progress = int(bar_width * complete_percent/float(100))
        barparams = (nscan, totscans, ("/"*progress) +
                     (" " * (bar_width-progress)), complete_percent)

        printstr = "\rScan %0"+ndigit+"i/%i : [%s]%i%%"
        sys.stdout.write(printstr % barparams)
        sys.stdout.flush()

    elif msgtype == 'casa':
        message_list = [".", ".", ".", "10", ".", ".", ".", "20",
                        ".", ".", ".", "30", ".", ".", ".", "40",
                        ".", ".", ".", "50", ".", ".", ".", "60",
                        ".", ".", ".", "70", ".", ".", ".", "80",
                        ".", ".", ".", "90", ".", ".", ".", "DONE"]
        bar_width = len(message_list)
        progress = int(bar_width * complete_percent/float(100))
        message = ''.join(message_list[:progress])

        barparams = (nscan, totscans, message)
        printstr = "\rScan %0"+ndigit+"i/%i : %s"
        sys.stdout.write(printstr % barparams)
        sys.stdout.flush()

    elif msgtype == 'itcrowd':
        message_list = ["0", "1", "1", "8", " ", "9", "9", "9", " ", "8", "8", "1", "9", "9", " ",
                        "9", "1", "1", "9", " ", "7", "2", "5", " ", " ", " ", "3"]
        bar_width = len(message_list)
        progress = int(bar_width * complete_percent/float(100))
        message = ''.join(message_list[:progress])
        if complete_percent < 100:
            message += "."
            message += " "*(bar_width-progress-1)

        barparams = (nscan, totscans, message)

        printstr = "\rScan %0"+ndigit+"i/%i : [%s]"
        sys.stdout.write(printstr % barparams)
        sys.stdout.flush()

    elif msgtype == 'bh':
        message_all = ehc.BHIMAGE
        bar_width = len(message_all)
        progress = int(np.floor(bar_width * complete_percent/float(100)))-1
        progress_last = int(np.floor(bar_width * complete_percent_last/float(100)))-1
        if progress > progress_last:
            for i in range(progress_last+1, progress+1):
                message_line = ''.join(message_all[i])
                message_line = f'{int(complete_percent):03d}' + message_line
                print(message_line)

    elif msgtype == 'eht':
        message_all = ehc.EHTIMAGE
        bar_width = len(message_all)
        progress = int(np.floor(bar_width * complete_percent/float(100)))-1
        progress_last = int(np.floor(bar_width * complete_percent_last/float(100)))-1
        if progress > progress_last:
            for i in range(progress_last+1, progress+1):
                message_line = ''.join(message_all[i])
                message_line = f'{int(complete_percent):03d}' + message_line
                print(message_line)

    elif msgtype == 'dots':
        sys.stdout.write('.')
        sys.stdout.flush()

    else:  # msgtype=='default':
        barparams = (nscan, totscans, complete_percent)
        printstr = "\rScan %0"+ndigit+"i/%i : %i%% done . . ."
        sys.stdout.write(printstr % barparams)
        sys.stdout.flush()



