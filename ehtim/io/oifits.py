"""
A module for reading/writing OIFITS files

This module is NOT related to the OIFITS Python module provided at
http://www.mrao.cam.ac.uk/research/OAS/oi_data/oifits.html
It is a (better) alternative.

To open an existing OIFITS file, use the oifits.open(filename)
function.  This will return an oifits object with the following
members (any of which can be empty dictionaries or numpy arrays):

   array: a dictionary of interferometric arrays, as defined by the
   OI_ARRAY tables.  The dictionary key is the name of the array
   (ARRNAME).

   target: a numpy array of targets, as defined by the rows of the
   OI_TARGET table.

   wavelength: a dictionary of wavelength tables (OI_WAVELENGTH).  The
   dictionary key is the name of the instrument/settings (INSNAME).

   vis, vis2 and t3: numpy arrays of objects containing all the
   measurement information.  Each list member corresponds to a row in
   an OI_VIS/OI_VIS2/OI_T3 table.

This module makes an ad-hoc, backwards-compatible change to the OIFITS
revision 1 standard originally described by Pauls et al., 2005, PASP,
117, 1255.  The OI_VIS and OI_VIS2 tables in OIFITS files produced by
this file contain two additional columns for the correlated flux,
CFLUX and CFLUXERR , which are arrays with a length corresponding to
the number of wavelength elements (just as VISAMP/VIS2DATA).

The main purpose of this module is to allow easy access to your OIFITS
data within Python, where you can then analyze it in any way you want.
As of version 0.3, the module can now be used to create OIFITS files
from scratch without serious pain.  Be warned, creating an array table
from scratch is probably like nailing jelly to a tree.  In a future
verison this will become easier.

The module also provides a simple mechanism for combining multiple
oifits objects, achieved by using the '+' operator on two oifits
objects: result = a + b.  The result can then be written to a file
using result.save(filename).

Many of the parameters and their meanings are not specifically
documented here.  However, the nomenclature mirrors that of the OIFITS
standard, so it is recommended to use this module with the PASP
reference above in hand.

Beginning with version 0.3, the OI_VIS/OI_VIS2/OI_T3 classes now use
masked arrays for convenience, where the mask is defined via the
'flag' member of these classes.  Beware of the following subtlety: as
before, the array data are accessed via (for example) OI_VIS.visamp;
however, OI_VIS.visamp is just a method which constructs (on the fly)
a masked array from OI_VIS._visamp, which is where the data are
actually stored.  This is done transparently, and the data can be
accessed and modified transparently via the "visamp" hidden attribute.
The same goes for correlated fluxes, differential/closure phases,
triple products, etc.  See the notes on the individual classes for a
list of all the "hidden" attributes.

For further information, contact Paul Boley (boley@mpia-hd.mpg.de).

"""
from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import object

import numpy as np
from numpy import double, bool, ma
try:
    import pyfits
except:
    from astropy.io import fits as pyfits
import datetime
import copy

__author__ = "Paul Boley"
__email__ = "boley@mpia-hd.mpg.de"
__date__ ='1 October 2012'
__version__ = '0.3.1'
_mjdzero = datetime.datetime(1858, 11, 17)

matchtargetbyname = False
matchstationbyname = False
refdate = datetime.datetime(2000, 1, 1)

def _plurals(count):
    if count != 1: return 's'
    return ''

def _array_eq(a, b):
    "Test whether all the elements of two arrays are equal."

    try:
        return not (a != b).any()
    except:
        return not (a != b)

class _angpoint(float):
    "Convenience object for representing angles."

    def __init__(self, angle):
        self.angle = angle

    def __repr__(self):
        return '_angpoint(%s)'%self.angle.__repr__()

    def __str__(self):
        return "%g degrees"%(self.angle)

    def __eq__(self, other):
        return self.angle == other.angle

    def __ne__(self, other):
        return not self.__eq__(other)

    def asdms(self):
        """Return the value as a string in dms format,
        e.g. +25:30:22.55.  Useful for declination."""
        angle = self.angle
        if angle < 0:
            negative = True
            angle *= -1.0
        else:
            negative = False
        degrees = np.floor(angle)
        minutes = np.floor((angle - degrees)*60.0)
        seconds = (angle - degrees - minutes/60.0)*3600.0
        try:
            if negative:
                return "-%02d:%02d:%05.2f"%(degrees,minutes,seconds)
            else:
                return "+%02d:%02d:%05.2f"%(degrees,minutes,seconds)
        except TypeError:
            return self.__repr__()

    def ashms(self):
        """Return the value as a string in hms format,
        e.g. 5:12:17.21.  Useful for right ascension."""
        angle = self.angle*24.0/360.0

        hours = np.floor(angle)
        minutes = np.floor((angle - hours)*60.0)
        seconds = (angle - hours - minutes/60.0)*3600.0
        try:
            return "%02d:%02d:%05.2f"%(hours,minutes,seconds)
        except TypeError:
            return self.__repr__()

class OI_TARGET(object):

    def __init__(self, target, raep0, decep0, equinox=2000.0, ra_err=0.0, dec_err=0.0,
                 sysvel=0.0, veltyp='TOPCENT', veldef='OPTICAL', pmra=0.0, pmdec=0.0,
                 pmra_err=0.0, pmdec_err=0.0, parallax=0.0, para_err=0.0, spectyp='UNKNOWN'):
        self.target = target
        self.raep0 = _angpoint(raep0)
        self.decep0 = _angpoint(decep0)
        self.equinox = equinox
        self.ra_err = ra_err
        self.dec_err = dec_err
        self.sysvel = sysvel
        self.veltyp = veltyp
        self.veldef = veldef
        self.pmra = pmra
        self.pmdec = pmdec
        self.pmra_err = pmra_err
        self.pmdec_err = pmdec_err
        self.parallax = parallax
        self.para_err = para_err
        self.spectyp = spectyp

    def __eq__(self, other):

        if type(self) != type(other): return False

        return not (
            (self.target    != other.target)    or
            (self.raep0     != other.raep0)     or
            (self.decep0    != other.decep0)    or
            (self.equinox   != other.equinox)   or
            (self.ra_err    != other.ra_err)    or
            (self.dec_err   != other.dec_err)   or
            (self.sysvel    != other.sysvel)    or
            (self.veltyp    != other.veltyp)    or
            (self.veldef    != other.veldef)    or
            (self.pmra      != other.pmra)      or
            (self.pmdec     != other.pmdec)     or
            (self.pmra_err  != other.pmra_err)  or
            (self.pmdec_err != other.pmdec_err) or
            (self.parallax  != other.parallax)  or
            (self.para_err  != other.para_err)  or
            (self.spectyp   != other.spectyp))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return "%s: %s %s (%g)"%(self.target, self.raep0.ashms(), self.decep0.asdms(), self.equinox)

    def info(self):
        print(str(self))

class OI_WAVELENGTH(object):

    def __init__(self, eff_wave, eff_band=None):
        self.eff_wave = np.array(eff_wave, dtype=double).reshape(-1)
        if eff_band == None:
            eff_band = np.zeros_like(eff_wave)
        self.eff_band = np.array(eff_band, dtype=double).reshape(-1)

    def __eq__(self, other):

        if type(self) != type(other): return False

        return not (
            (not _array_eq(self.eff_wave, other.eff_wave)) or
            (not _array_eq(self.eff_band, other.eff_band)))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "%d wavelength%s (%.3g-%.3g um)"%(len(self.eff_wave), _plurals(len(self.eff_wave)), 1e6*np.min(self.eff_wave),1e6*np.max(self.eff_wave))

    def info(self):
        print(str(self))


class OI_VIS(object):
    """
    Class for storing visibility amplitude and differential phase data.
    To access the data, use the following hidden attributes:

    visamp, visamperr, visphi, visphierr, flag;
    and possibly cflux, cfluxerr.

    """

    def __init__(self, timeobs, int_time, visamp, visamperr, visphi, visphierr, flag, ucoord,
                 vcoord, wavelength, target, array=None, station=(None,None), cflux=None, cfluxerr=None):
        self.timeobs = timeobs
        self.array = array
        self.wavelength = wavelength
        self.target = target
        self.int_time = int_time
        self._visamp = np.array(visamp, dtype=double).reshape(-1)
        self._visamperr = np.array(visamperr, dtype=double).reshape(-1)
        self._visphi = np.array(visphi, dtype=double).reshape(-1)
        self._visphierr = np.array(visphierr, dtype=double).reshape(-1)
        if cflux != None: self._cflux = np.array(cflux, dtype=double).reshape(-1)
        else: self._cflux = None
        if cfluxerr != None: self._cfluxerr = np.array(cfluxerr, dtype=double).reshape(-1)
        else: self._cfluxerr = None
        self.flag = np.array(flag, dtype=bool).reshape(-1)
        self.ucoord = ucoord
        self.vcoord = vcoord
        self.station = station

    def __eq__(self, other):

        if type(self) != type(other): return False

        return not (
            (self.timeobs    != other.timeobs)    or
            (self.array      != other.array)      or
            (self.wavelength != other.wavelength) or
            (self.target     != other.target)     or
            (self.int_time   != other.int_time)   or
            (self.ucoord     != other.ucoord)     or
            (self.vcoord     != other.vcoord)     or
            (self.array      != other.array)      or
            (self.station    != other.station)    or
            (not _array_eq(self.visamp, other.visamp)) or
            (not _array_eq(self.visamperr, other.visamperr)) or
            (not _array_eq(self.visphi, other.visphi)) or
            (not _array_eq(self.visphierr, other.visphierr)) or
            (not _array_eq(self.flag, other.flag)))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getattr__(self, attrname):
        if attrname in ('visamp', 'visamperr', 'visphi', 'visphierr'):
            return ma.masked_array(self.__dict__['_' + attrname], mask=self.flag)
        elif attrname in ('cflux', 'cfluxerr'):
            if (self.__dict__['_' + attrname] != None):
                return ma.masked_array(self.__dict__['_' + attrname], mask=self.flag)
            else:
                return None
        else:
            raise AttributeError(attrname)

    def __setattr__(self, attrname, value):
        if attrname in ('visamp', 'visamperr', 'visphi', 'visphierr', 'cflux', 'cfluxerr'):
            self.__dict__['_' + attrname] = value
        else:
            self.__dict__[attrname] = value

    def __repr__(self):
        meanvis = ma.mean(self.visamp)
        if self.station[0] and self.station[1]:
            baselinename = ' (' + self.station[0].sta_name + self.station[1].sta_name + ')'
        else:
            baselinename = ''
        return '%s %s%s: %d point%s (%d masked), B = %5.1f m, PA = %5.1f deg, <V> = %4.2g'%(self.target.target, self.timeobs.strftime('%F %T'), baselinename, len(self.visamp), _plurals(len(self.visamp)), np.sum(self.flag), np.sqrt(self.ucoord**2 + self.vcoord**2), np.arctan(self.ucoord/self.vcoord) * 180.0 / np.pi % 180.0, meanvis)

    def info(self):
        print(str(self))

class OI_VIS2(object):
    """
    Class for storing squared visibility amplitude data.
    To access the data, use the following hidden attributes:

    vis2data, vis2err

    """
    def __init__(self, timeobs, int_time, vis2data, vis2err, flag, ucoord, vcoord, wavelength,
                 target, array=None, station=(None, None)):
        self.timeobs = timeobs
        self.array = array
        self.wavelength = wavelength
        self.target = target
        self.int_time = int_time
        self._vis2data = np.array(vis2data, dtype=double).reshape(-1)
        self._vis2err = np.array(vis2err, dtype=double).reshape(-1)
        self.flag = np.array(flag, dtype=bool).reshape(-1)
        self.ucoord = ucoord
        self.vcoord = vcoord
        self.station = station

    def __eq__(self, other):

        if type(self) != type(other): return False

        return not (
            (self.timeobs    != other.timeobs)    or
            (self.array      != other.array)      or
            (self.wavelength != other.wavelength) or
            (self.target     != other.target)     or
            (self.int_time   != other.int_time)   or
            (self.ucoord     != other.ucoord)     or
            (self.vcoord     != other.vcoord)     or
            (self.array      != other.array)      or
            (self.station    != other.station)    or
            (not _array_eq(self.vis2data, other.vis2data)) or
            (not _array_eq(self.vis2err, other.vis2err)) or
            (not _array_eq(self.flag, other.flag)))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getattr__(self, attrname):
        if attrname in ('vis2data', 'vis2err'):
            return ma.masked_array(self.__dict__['_' + attrname], mask=self.flag)
        else:
            raise AttributeError(attrname)

    def __setattr__(self, attrname, value):
        if attrname in ('vis2data', 'vis2err'):
            self.__dict__['_' + attrname] = value
        else:
            self.__dict__[attrname] = value

    def __repr__(self):
        meanvis = ma.mean(self.vis2data)
        if self.station[0] and self.station[1]:
            baselinename = ' (' + self.station[0].sta_name + self.station[1].sta_name + ')'
        else:
            baselinename = ''
        return "%s %s%s: %d point%s (%d masked), B = %5.1f m, PA = %5.1f deg, <V^2> = %4.2g"%(self.target.target, self.timeobs.strftime('%F %T'), baselinename, len(self.vis2data), _plurals(len(self.vis2data)), np.sum(self.flag), np.sqrt(self.ucoord**2 + self.vcoord**2), np.arctan(self.ucoord/self.vcoord) * 180.0 / np.pi % 180.0, meanvis)

    def info(self):
        print(str(self))


class OI_T3(object):
    """
    Class for storing triple product and closure phase data.
    To access the data, use the following hidden attributes:

    t3amp, t3amperr, t3phi, t3phierr

    """

    def __init__(self, timeobs, int_time, t3amp, t3amperr, t3phi, t3phierr, flag, u1coord,
                 v1coord, u2coord, v2coord, wavelength, target, array=None, station=(None,None,None)):
        self.timeobs = timeobs
        self.array = array
        self.wavelength = wavelength
        self.target = target
        self.int_time = int_time
        self._t3amp = np.array(t3amp, dtype=double).reshape(-1)
        self._t3amperr = np.array(t3amperr, dtype=double).reshape(-1)
        self._t3phi = np.array(t3phi, dtype=double).reshape(-1)
        self._t3phierr = np.array(t3phierr, dtype=double).reshape(-1)
        self.flag = np.array(flag, dtype=bool).reshape(-1)
        self.u1coord = u1coord
        self.v1coord = v1coord
        self.u2coord = u2coord
        self.v2coord = v2coord
        self.station = station

    def __eq__(self, other):

        if type(self) != type(other): return False

        return not (
            (self.timeobs    != other.timeobs)    or
            (self.array      != other.array)      or
            (self.wavelength != other.wavelength) or
            (self.target     != other.target)     or
            (self.int_time   != other.int_time)   or
            (self.u1coord    != other.u1coord)    or
            (self.v1coord    != other.v1coord)    or
            (self.u2coord    != other.u2coord)    or
            (self.v2coord    != other.v2coord)    or
            (self.array      != other.array)      or
            (self.station    != other.station)    or
            (not _array_eq(self.t3amp, other.t3amp)) or
            (not _array_eq(self.t3amperr, other.t3amperr)) or
            (not _array_eq(self.t3phi, other.t3phi)) or
            (not _array_eq(self.t3phierr, other.t3phierr)) or
            (not _array_eq(self.flag, other.flag)))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getattr__(self, attrname):
        if attrname in ('t3amp', 't3amperr', 't3phi', 't3phierr'):
            return ma.masked_array(self.__dict__['_' + attrname], mask=self.flag)
        else:
            raise AttributeError(attrname)

    def __setattr__(self, attrname, value):
        if attrname in ('vis2data', 'vis2err'):
            self.__dict__['_' + attrname] = value
        else:
            self.__dict__[attrname] = value

    def __repr__(self):
        meant3 = np.mean(self.t3amp[np.where(self.flag == False)])
        if self.station[0] and self.station[1] and self.station[2]:
            baselinename = ' (' + self.station[0].sta_name + self.station[1].sta_name + self.station[2].sta_name + ')'
        else:
            baselinename = ''
        return "%s %s%s: %d point%s (%d masked), B = %5.1fm, %5.1fm, <T3> = %4.2g"%(self.target.target, self.timeobs.strftime('%F %T'), baselinename, len(self.t3amp), _plurals(len(self.t3amp)), np.sum(self.flag), np.sqrt(self.u1coord**2 + self.v1coord**2), np.sqrt(self.u2coord**2 + self.v2coord**2), meant3)

    def info(self):
        print(str(self))

class OI_STATION(object):
    """ This class corresponds to a single row (i.e. single
    station/telescope) of an OI_ARRAY table."""

    def __init__(self, tel_name=None, sta_name=None, diameter=None, staxyz=[None, None, None]):
        self.tel_name = tel_name
        self.sta_name = sta_name
        self.diameter = diameter
        self.staxyz = staxyz

    def __eq__(self, other):

        if type(self) != type(other): return False

        return not (
            (self.tel_name != other.tel_name) or
            (self.sta_name != other.sta_name) or
            (self.diameter != other.diameter) or
            (not _array_eq(self.staxyz, other.staxyz)))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return '%s/%s (%g m)'%(self.sta_name, self.tel_name, self.diameter)

class OI_ARRAY(object):
    """Contains all the data for a single OI_ARRAY table.  Note the
    hidden convenience attributes latitude, longitude, and altitude."""

    def __init__(self, frame, arrxyz, stations=()):
        self.frame = frame
        self.arrxyz = arrxyz
        #self.station = stations;
        self.station = np.empty(0)
        for station in stations:
            tel_name, sta_name, sta_index, diameter, staxyz = station
            self.station = np.append(self.station, OI_STATION(tel_name=tel_name, sta_name=sta_name, diameter=diameter, staxyz=staxyz))

    def __eq__(self, other):

        if type(self) != type(other): return False

        equal = not (
            (self.frame   != other.frame)   or
            (not _array_eq(self.arrxyz, other.arrxyz)))

        if not equal: return False

        # If position appears to be the same, check that the stations
        # (and ordering) are also the same
        if (self.station != other.station).any():
            return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getattr__(self, attrname):
        if attrname == 'latitude':
            radius = np.sqrt((self.arrxyz**2).sum())
            return _angpoint(np.arcsin(self.arrxyz[2]/radius)*180.0/np.pi)
        elif attrname == 'longitude':
            radius = np.sqrt((self.arrxyz**2).sum())
            xylen = np.sqrt(self.arrxyz[0]**2+self.arrxyz[1]**2)
            return _angpoint(np.arcsin(self.arrxyz[1]/xylen)*180.0/np.pi)
        elif attrname == 'altitude':
            radius = np.sqrt((self.arrxyz**2).sum())
            return radius - 6378100.0
        else:
            raise AttributeError(attrname)

    def __repr__(self):
        return '%s %s %g m, %d station%s'%(self.latitude.asdms(), self.longitude.asdms(), self.altitude, len(self.station), _plurals(len(self.station)))

    def info(self, verbose=0):
        """Print the array's center coordinates.  If verbosity >= 1,
        print information about each station."""
        print(str(self))
        if verbose >= 1:
            for station in self.station:
                print("   %s"%str(station))

    def get_station_by_name(self, name):

        for station in self.station:
            if station.sta_name == name:
                return station

        raise LookupError('No such station %s'%name)

class oifits(object):

    def __init__(self):

        self.wavelength = {}
        self.target = np.empty(0)
        self.array = {}
        self.vis = np.empty(0)
        self.vis2 = np.empty(0)
        self.t3 = np.empty(0)

    def __add__(self, other):
        """Consistently combine two separate oifits objects.  Note
        that targets can be matched by name only (e.g. if coordinates
        differ) by setting oifits.matchtargetbyname to True.  The same
        goes for stations of the array (controlled by
        oifits.matchstationbyname)"""
        # Don't do anything if the two oifits objects are not CONSISTENT!
        if self.isconsistent() == False or other.isconsistent() == False:
            print('oifits objects are not consistent, bailing.')
            return

        new = copy.deepcopy(self)
        if len(other.wavelength):
            wavelengthmap = {}
            for key in list(other.wavelength.keys()):
                if key not in list(new.wavelength.keys()):
                    new.wavelength[key] = copy.deepcopy(other.wavelength[key])
                elif new.wavelength[key] != other.wavelength[key]:
                    raise ValueError('Wavelength tables have the same key but differing contents.')
                wavelengthmap[id(other.wavelength[key])] = new.wavelength[key]

        if len(other.target):
            targetmap = {}
            for otarget in other.target:
                for ntarget in new.target:
                    if matchtargetbyname and ntarget.target == otarget.target:
                        targetmap[id(otarget)] = ntarget
                        break
                    elif ntarget == otarget:
                        targetmap[id(otarget)] = ntarget
                        break
                    elif ntarget.target == otarget.target:
                        print('Found a target with a matching name, but some differences in the target specification.  Creating a new target.  Set oifits.matchtargetbyname to True to override this behavior.')
                # If 'id(otarget)' is not in targetmap, then this is a new
                # target and should be added to the array of targets
                if id(otarget) not in list(targetmap.keys()):
                    try:
                        newkey = list(new.target.keys())[-1]+1
                    except:
                        newkey = 1
                    target = copy.deepcopy(otarget)
                    new.target = np.append(new.target, target)
                    targetmap[id(otarget)] = target

        if len(other.array):
            stationmap = {}
            arraymap = {}
            for key, otharray in other.array.items():
                arraymap[id(otharray)] = key
                if key not in list(new.array.keys()):
                    new.array[key] = copy.deepcopy(other.array[key])
                # If arrays have the same name but seem to differ, try
                # to combine the two (by including the union of both
                # sets of stations)
                for othsta in other.array[key].station:
                    for newsta in new.array[key].station:
                        if newsta == othsta:
                            stationmap[id(othsta)] = newsta
                            break
                        elif matchstationbyname and newsta.sta_name == othsta.sta_name:
                            stationmap[id(othsta)] = newsta
                            break
                        elif newsta.sta_name == othsta.sta_name and matchstationbyname == False:
                            raise ValueError('Stations have matching names but conflicting data.')
                    # If 'id(othsta)' is not in the stationmap
                    # dictionary, then this is a new station and
                    # should be added to the current array
                    if id(othsta) not in list(stationmap.keys()):
                        newsta = copy.deepcopy(othsta)
                        new.array[key].station = np.append(new.array[key].station, newsta)
                        stationmap[id(othsta)] = newsta
                        # Make sure that staxyz of the new station is relative to the new array center
                        newsta.staxyz = othsta.staxyz - other.array[key].arrxyz + new.array[key].arrxyz

        for vis in other.vis:
            if vis not in new.vis:
                newvis = copy.copy(vis)
                # The wavelength, target, array and station objects
                # should point to the appropriate objects inside the
                # 'new' structure
                newvis.wavelength = wavelengthmap[id(vis.wavelength)]
                newvis.target = targetmap[id(vis.target)]
                if (vis.array):
                    newvis.array = new.array[arraymap[id(vis.array)]]
                    newvis.station = [None, None]
                    newvis.station[0] = stationmap[id(vis.station[0])]
                    newvis.station[1] = stationmap[id(vis.station[1])]
                new.vis = np.append(new.vis, newvis)

        for vis2 in other.vis2:
            if vis2 not in new.vis2:
                newvis2 = copy.copy(vis2)
                # The wavelength, target, array and station objects
                # should point to the appropriate objects inside the
                # 'new' structure
                newvis2.wavelength = wavelengthmap[id(vis2.wavelength)]
                newvis2.target = targetmap[id(vis2.target)]
                if (vis2.array):
                    newvis2.array = new.array[arraymap[id(vis2.array)]]
                    newvis2.station = [None, None]
                    newvis2.station[0] = stationmap[id(vis2.station[0])]
                    newvis2.station[1] = stationmap[id(vis2.station[1])]
                new.vis2 = np.append(new.vis2, newvis2)

        for t3 in other.t3:
            if t3 not in new.t3:
                newt3 = copy.copy(t3)
                # The wavelength, target, array and station objects
                # should point to the appropriate objects inside the
                # 'new' structure
                newt3.wavelength = wavelengthmap[id(t3.wavelength)]
                newt3.target = targetmap[id(t3.target)]
                if (t3.array):
                    newt3.array = new.array[arraymap[id(t3.array)]]
                    newt3.station = [None, None, None]
                    newt3.station[0] = stationmap[id(t3.station[0])]
                    newt3.station[1] = stationmap[id(t3.station[1])]
                    newt3.station[2] = stationmap[id(t3.station[2])]
                new.t3 = np.append(new.t3, newt3)

        return(new)


    def __eq__(self, other):

        if type(self) != type(other): return False

        return not (
            (self.wavelength != other.wavelength)   or
            (self.target     != other.target).any() or
            (self.array      != other.array)      or
            (self.vis        != other.vis).any()  or
            (self.vis2       != other.vis2).any() or
            (self.t3         != other.t3).any())

    def __ne__(self, other):
        return not self.__eq__(other)

    def isvalid(self):
        """Returns True of the oifits object is both consistent (as
        determined by isconsistent()) and conforms to the OIFITS
        standard (according to Pauls et al., 2005, PASP, 117, 1255)."""

        warnings = []
        errors = []
        if not self.isconsistent():
            errors.append('oifits object is not consistent')
        if not self.target.size:
            errors.append('No OI_TARGET data')
        if not self.wavelength:
            errors.append('No OI_WAVELENGTH data')
        else:
            for wavelength in list(self.wavelength.values()):
                if len(wavelength.eff_wave) != len(wavelength.eff_band):
                    errors.append("eff_wave and eff_band are of different lengths for wavelength table '%s'"%key)
        if (self.vis.size + self.vis2.size + self.t3.size == 0):
            errors.append('Need to have atleast one measurement table (vis, vis2 or t3)')
        for vis in self.vis:
            nwave = len(vis.wavelength.eff_band)
            if (len(vis.visamp) != nwave) or (len(vis.visamperr) != nwave) or (len(vis.visphi) != nwave) or (len(vis.visphierr) != nwave) or (len(vis.flag) != nwave):
                errors.append("Data size mismatch for visibility measurement 0x%x (wavelength table has a length of %d)"%(id(vis), nwave))
        for vis2 in self.vis2:
            nwave = len(vis2.wavelength.eff_band)
            if (len(vis2.vis2data) != nwave) or (len(vis2.vis2err) != nwave) or (len(vis2.flag) != nwave):
                errors.append("Data size mismatch for visibility^2 measurement 0x%x (wavelength table has a length of %d)"%(id(vis), nwave))
        for t3 in self.t3:
            nwave = len(t3.wavelength.eff_band)
            if (len(t3.t3amp) != nwave) or (len(t3.t3amperr) != nwave) or (len(t3.t3phi) != nwave) or (len(t3.t3phierr) != nwave) or (len(t3.flag) != nwave):
                errors.append("Data size mismatch for visibility measurement 0x%x (wavelength table has a length of %d)"%(id(vis), nwave))

        if warnings:
            print("*** %d warning%s:"%(len(warnings), _plurals(len(warnings))))
            for warning in warnings:
                print('  ' + warning)
        if errors:
            print("*** %d ERROR%s:"%(len(errors), _plurals(len(errors)).upper()))
            for error in errors:
                print('  ' + error)

        return not (len(warnings) or len(errors))

    def isconsistent(self):
        """Returns True if the object is entirely self-contained,
        i.e. all cross-references to wavelength tables, arrays,
        stations etc. in the measurements refer to elements which are
        stored in the oifits object.  Note that an oifits object can
        be 'consistent' in this sense without being 'valid' as checked
        by isvalid()."""

        for vis in self.vis:
            if vis.array and (vis.array not in list(self.array.values())):
                print('A visibility measurement (0x%x) refers to an array which is not inside the main oifits object.'%id(vis))
                return False
            if ((vis.station[0] and (vis.station[0] not in vis.array.station)) or
                (vis.station[1] and (vis.station[1] not in vis.array.station))):
                print('A visibility measurement (0x%x) refers to a station which is not inside the main oifits object.'%id(vis))
                return False
            if vis.wavelength not in list(self.wavelength.values()):
                print('A visibility measurement (0x%x) refers to a wavelength table which is not inside the main oifits object.'%id(vis))
                return False
            if vis.target not in self.target:
                print('A visibility measurement (0x%x) refers to a target which is not inside the main oifits object.'%id(vis))
                return False

        for vis2 in self.vis2:
            if vis2.array and (vis2.array not in list(self.array.values())):
                print('A visibility^2 measurement (0x%x) refers to an array which is not inside the main oifits object.'%id(vis2))
                return False
            if ((vis2.station[0] and (vis2.station[0] not in vis2.array.station)) or
                (vis2.station[1] and (vis2.station[1] not in vis2.array.station))):
                print('A visibility^2 measurement (0x%x) refers to a station which is not inside the main oifits object.'%id(vis))
                return False
            if vis2.wavelength not in list(self.wavelength.values()):
                print('A visibility^2 measurement (0x%x) refers to a wavelength table which is not inside the main oifits object.'%id(vis2))
                return False
            if vis2.target not in self.target:
                print('A visibility^2 measurement (0x%x) refers to a target which is not inside the main oifits object.'%id(vis2))
                return False

        for t3 in self.t3:
            if t3.array and (t3.array not in list(self.array.values())):
                print('A closure phase measurement (0x%x) refers to an array which is not inside the main oifits object.'%id(t3))
                return False
            if ((t3.station[0] and (t3.station[0] not in t3.array.station)) or
                (t3.station[1] and (t3.station[1] not in t3.array.station)) or
                (t3.station[2] and (t3.station[2] not in t3.array.station))):
                print('A closure phase measurement (0x%x) refers to a station which is not inside the main oifits object.'%id(t3))
                return False
            if t3.wavelength not in list(self.wavelength.values()):
                print('A closure phase measurement (0x%x) refers to a wavelength table which is not inside the main oifits object.'%id(t3))
                return False
            if t3.target not in self.target:
                print('A closure phase measurement (0x%x) refers to a target which is not inside the main oifits object.'%id(t3))
                return False

        return True

    def info(self, recursive=True, verbose=0):
        """Print out a summary of the contents of the oifits object.
        Set recursive=True to obtain more specific information about
        each of the individual components, and verbose to an integer
        to increase the verbosity level."""

        if self.wavelength:
            wavelengths = 0
            if recursive:
                print("====================================================================")
                print("SUMMARY OF WAVELENGTH TABLES")
                print("====================================================================")
            for key in list(self.wavelength.keys()):
                wavelengths += len(self.wavelength[key].eff_wave)
                if recursive: print("'%s': %s"%(key, str(self.wavelength[key])))
            print("%d wavelength table%s with %d wavelength%s in total"%(len(self.wavelength), _plurals(len(self.wavelength)), wavelengths, _plurals(wavelengths)))
        if self.target.size:
            if recursive:
                print("====================================================================")
                print("SUMMARY OF TARGET TABLES")
                print("====================================================================")
                for target in self.target:
                    target.info()
            print("%d target%s"%(len(self.target), _plurals(len(self.target))))
        if self.array:
            stations = 0
            if recursive:
                print("====================================================================")
                print("SUMMARY OF ARRAY TABLES")
                print("====================================================================")
            for key in list(self.array.keys()):
                if recursive:
                    print(key + ':')
                    self.array[key].info(verbose=verbose)
                stations += len(self.array[key].station)
            print("%d array%s with %d station%s"%(len(self.array), _plurals(len(self.array)), stations, _plurals(stations)))
        if self.vis.size:
            if recursive:
                print("====================================================================")
                print("SUMMARY OF VISIBILITY MEASUREMENTS")
                print("====================================================================")
                for vis in self.vis:
                    vis.info()
            print("%d visibility measurement%s"%(len(self.vis), _plurals(len(self.vis))))
        if self.vis2.size:
            if recursive:
                print("====================================================================")
                print("SUMMARY OF VISIBILITY^2 MEASUREMENTS")
                print("====================================================================")
                for vis2 in self.vis2:
                    vis2.info()
            print("%d visibility^2 measurement%s"%(len(self.vis2), _plurals(len(self.vis2))))
        if self.t3.size:
            if recursive:
                print("====================================================================")
                print("SUMMARY OF T3 MEASUREMENTS")
                print("====================================================================")
                for t3 in self.t3:
                    t3.info()
            print("%d closure phase measurement%s"%(len(self.t3), _plurals(len(self.t3))))

    def save(self, filename):
        """Write the contents of the oifits object to a file in OIFITS
        format."""

        if not self.isconsistent():
            print('oifits object is not consistent, refusing to go further')
            return

        hdulist = pyfits.HDUList()
        hdu = pyfits.PrimaryHDU()
        hdu.header.update('DATE', datetime.datetime.now().strftime(format='%F'), comment='Creation date')
        hdu.header.add_comment('Written by OIFITS Python module version %s'%__version__)
        hdu.header.add_comment('http://www.mpia-hd.mpg.de/homes/boley/oifits/')

        wavelengthmap = {}
        hdulist.append(hdu)
        for insname, wavelength in self.wavelength.items():
            wavelengthmap[id(wavelength)] = insname
            hdu = pyfits.new_table(pyfits.ColDefs((
                pyfits.Column(name='EFF_WAVE', format='1E', unit='METERS', array=wavelength.eff_wave),
                pyfits.Column(name='EFF_BAND', format='1E', unit='METERS', array=wavelength.eff_band)
                )))
            hdu.header.update('EXTNAME', 'OI_WAVELENGTH')
            hdu.header.update('OI_REVN', 1, 'Revision number of the table definition')
            hdu.header.update('INSNAME', insname, 'Name of detector, for cross-referencing')
            hdulist.append(hdu)

        targetmap = {}
        if self.target.size:
            target_id = []
            target = []
            raep0 = []
            decep0 = []
            equinox = []
            ra_err = []
            dec_err = []
            sysvel = []
            veltyp = []
            veldef = []
            pmra = []
            pmdec = []
            pmra_err = []
            pmdec_err = []
            parallax = []
            para_err = []
            spectyp = []
            for i, targ in enumerate(self.target):
                key = i+1
                targetmap[id(targ)] = key
                target_id.append(key)
                target.append(targ.target)
                raep0.append(targ.raep0)
                decep0.append(targ.decep0)
                equinox.append(targ.equinox)
                ra_err.append(targ.ra_err)
                dec_err.append(targ.dec_err)
                sysvel.append(targ.sysvel)
                veltyp.append(targ.veltyp)
                veldef.append(targ.veldef)
                pmra.append(targ.pmra)
                pmdec.append(targ.pmdec)
                pmra_err.append(targ.pmra_err)
                pmdec_err.append(targ.pmdec_err)
                parallax.append(targ.parallax)
                para_err.append(targ.para_err)
                spectyp.append(targ.spectyp)

            hdu = pyfits.new_table(pyfits.ColDefs((
                pyfits.Column(name='TARGET_ID', format='1I', array=target_id),
                pyfits.Column(name='TARGET', format='16A', array=target),
                pyfits.Column(name='RAEP0', format='D1', unit='DEGREES', array=raep0),
                pyfits.Column(name='DECEP0', format='D1', unit='DEGREES', array=decep0),
                pyfits.Column(name='EQUINOX', format='E1', unit='YEARS', array=equinox),
                pyfits.Column(name='RA_ERR', format='D1', unit='DEGREES', array=ra_err),
                pyfits.Column(name='DEC_ERR', format='D1', unit='DEGREES', array=dec_err),
                pyfits.Column(name='SYSVEL', format='D1', unit='M/S', array=sysvel),
                pyfits.Column(name='VELTYP', format='A8', array=veltyp),
                pyfits.Column(name='VELDEF', format='A8', array=veldef),
                pyfits.Column(name='PMRA', format='D1', unit='DEG/YR', array=pmra),
                pyfits.Column(name='PMDEC', format='D1', unit='DEG/YR', array=pmdec),
                pyfits.Column(name='PMRA_ERR', format='D1', unit='DEG/YR', array=pmra_err),
                pyfits.Column(name='PMDEC_ERR', format='D1', unit='DEG/YR', array=pmdec_err),
                pyfits.Column(name='PARALLAX', format='E1', unit='DEGREES', array=parallax),
                pyfits.Column(name='PARA_ERR', format='E1', unit='DEGREES', array=para_err),
                pyfits.Column(name='SPECTYP', format='A16', array=spectyp)
                )))
            hdu.header.update('EXTNAME', 'OI_TARGET')
            hdu.header.update('OI_REVN', 1, 'Revision number of the table definition')
            hdulist.append(hdu)

        arraymap = {}
        stationmap = {}
        for arrname, array in self.array.items():
            arraymap[id(array)] = arrname
            tel_name = []
            sta_name = []
            sta_index = []
            diameter = []
            staxyz = []
            if array.station.size:
                for i, station in enumerate(array.station, 1):
                    stationmap[id(station)] = i
                    tel_name.append(station.tel_name)
                    sta_name.append(station.sta_name)
                    sta_index.append(i)
                    diameter.append(station.diameter)
                    staxyz.append(station.staxyz)
                hdu = pyfits.new_table(pyfits.ColDefs((
                    pyfits.Column(name='TEL_NAME', format='16A', array=tel_name),
                    pyfits.Column(name='STA_NAME', format='16A', array=sta_name),
                    pyfits.Column(name='STA_INDEX', format='1I', array=sta_index),
                    pyfits.Column(name='DIAMETER', unit='METERS', format='1E', array=diameter),
                    pyfits.Column(name='STAXYZ', unit='METERS', format='3D', array=staxyz)
                    )))
            hdu.header.update('EXTNAME', 'OI_ARRAY')
            hdu.header.update('OI_REVN', 1, 'Revision number of the table definition')
            hdu.header.update('ARRNAME', arrname, comment='Array name, for cross-referencing')
            hdu.header.update('FRAME', array.frame, comment='Coordinate frame')
            hdu.header.update('ARRAYX', array.arrxyz[0], comment='Array center x coordinate (m)')
            hdu.header.update('ARRAYY', array.arrxyz[1], comment='Array center y coordinate (m)')
            hdu.header.update('ARRAYZ', array.arrxyz[2], comment='Array center z coordinate (m)')
            hdulist.append(hdu)

        if self.vis.size:
            # The tables are grouped by ARRNAME and INSNAME -- all
            # observations which have the same ARRNAME and INSNAME are
            # put into a single FITS binary table.
            tables = {}
            for vis in self.vis:
                nwave = vis.wavelength.eff_wave.size
                if vis.array:
                    key = (arraymap[id(vis.array)], wavelengthmap[id(vis.wavelength)])
                else:
                    key = (None, wavelengthmap[id(vis.wavelength)])
                if key in list(tables.keys()):
                    data = tables[key]
                else:
                    data = tables[key] = {'target_id':[], 'time':[], 'mjd':[], 'int_time':[],
                                          'visamp':[], 'visamperr':[], 'visphi':[], 'visphierr':[],
                                          'cflux':[], 'cfluxerr':[], 'ucoord':[], 'vcoord':[],
                                          'sta_index':[], 'flag':[]}
                data['target_id'].append(targetmap[id(vis.target)])
                if vis.timeobs:
                    time = vis.timeobs - refdate
                    data['time'].append(time.days * 24.0 * 3600.0 + time.seconds)
                    mjd = (vis.timeobs - _mjdzero).days + (vis.timeobs - _mjdzero).seconds / 3600.0 / 24.0
                    data['mjd'].append(mjd)
                else:
                    data['time'].append(None)
                    data['mjd'].append(None)
                data['int_time'].append(vis.int_time)
                if nwave == 1:
                    data['visamp'].append(vis.visamp[0])
                    data['visamperr'].append(vis.visamperr[0])
                    data['visphi'].append(vis.visphi[0])
                    data['visphierr'].append(vis.visphierr[0])
                    data['flag'].append(vis.flag[0])
                    if vis.cflux != None:
                        data['cflux'].append(vis.cflux[0])
                    else:
                        data['cflux'].append(None)
                    if vis.cfluxerr != None:
                        data['cfluxerr'].append(vis.cfluxerr[0])
                    else:
                        data['cfluxerr'].append(None)
                else:
                    data['visamp'].append(vis.visamp)
                    data['visamperr'].append(vis.visamperr)
                    data['visphi'].append(vis.visphi)
                    data['visphierr'].append(vis.visphierr)
                    data['flag'].append(vis.flag)
                    if vis.cflux != None:
                        data['cflux'].append(vis.cflux)
                    else:
                        cflux=np.empty(nwave)
                        cflux[:]=None
                        data['cflux'].append(cflux)
                    if vis.cfluxerr != None:
                        data['cfluxerr'].append(vis.cfluxerr)
                    else:
                        cfluxerr=np.empty(nwave)
                        cfluxerr[:]=None
                        data['cfluxerr'].append(cfluxerr)
                data['ucoord'].append(vis.ucoord)
                data['vcoord'].append(vis.vcoord)
                if vis.station[0] and vis.station[1]:
                    data['sta_index'].append([stationmap[id(vis.station[0])], stationmap[id(vis.station[1])]])
                else:
                    data['sta_index'].append([-1, -1])
            for key in list(tables.keys()):
                data = tables[key]
                nwave = self.wavelength[key[1]].eff_wave.size

                hdu = pyfits.new_table(pyfits.ColDefs([
                    pyfits.Column(name='TARGET_ID', format='1I', array=data['target_id']),
                    pyfits.Column(name='TIME', format='1D', unit='SECONDS', array=data['time']),
                    pyfits.Column(name='MJD', unit='DAY', format='1D', array=data['mjd']),
                    pyfits.Column(name='INT_TIME', format='1D', unit='SECONDS', array=data['int_time']),
                    pyfits.Column(name='VISAMP', format='%dD'%nwave, array=data['visamp']),
                    pyfits.Column(name='VISAMPERR', format='%dD'%nwave, array=data['visamperr']),
                    pyfits.Column(name='VISPHI', unit='DEGREES', format='%dD'%nwave, array=data['visphi']),
                    pyfits.Column(name='VISPHIERR', unit='DEGREES', format='%dD'%nwave, array=data['visphierr']),
                    pyfits.Column(name='CFLUX', format='%dD'%nwave, array=data['cflux']),
                    pyfits.Column(name='CFLUXERR', format='%dD'%nwave, array=data['cfluxerr']),
                    pyfits.Column(name='UCOORD', format='1D', unit='METERS', array=data['ucoord']),
                    pyfits.Column(name='VCOORD', format='1D', unit='METERS', array=data['vcoord']),
                    pyfits.Column(name='STA_INDEX', format='2I', array=data['sta_index'], null=-1),
                    pyfits.Column(name='FLAG', format='%dL'%nwave)
                    ]))

                # Setting the data of logical field via the
                # pyfits.Column call above with length > 1 (eg
                # format='171L' above) seems to be broken, atleast as
                # of PyFITS 2.2.2
                hdu.data.field('FLAG').setfield(data['flag'], bool)
                hdu.header.update('EXTNAME', 'OI_VIS')
                hdu.header.update('OI_REVN', 1, 'Revision number of the table definition')
                hdu.header.update('DATE-OBS', refdate.strftime('%F'), comment='Zero-point for table (UTC)')
                if key[0]: hdu.header.update('ARRNAME', key[0], 'Identifies corresponding OI_ARRAY')
                hdu.header.update('INSNAME', key[1], 'Identifies corresponding OI_WAVELENGTH table')
                hdulist.append(hdu)

        if self.vis2.size:
            tables = {}
            for vis in self.vis2:
                nwave = vis.wavelength.eff_wave.size
                if vis.array:
                    key = (arraymap[id(vis.array)], wavelengthmap[id(vis.wavelength)])
                else:
                    key = (None, wavelengthmap[id(vis.wavelength)])
                if key in list(tables.keys()):
                    data = tables[key]
                else:
                    data = tables[key] = {'target_id':[], 'time':[], 'mjd':[], 'int_time':[],
                                          'vis2data':[], 'vis2err':[], 'ucoord':[], 'vcoord':[],
                                          'sta_index':[], 'flag':[]}
                data['target_id'].append(targetmap[id(vis.target)])
                if vis.timeobs:
                    time = vis.timeobs - refdate
                    data['time'].append(time.days * 24.0 * 3600.0 + time.seconds)
                    mjd = (vis.timeobs - _mjdzero).days + (vis.timeobs - _mjdzero).seconds / 3600.0 / 24.0
                    data['mjd'].append(mjd)
                else:
                    data['time'].append(None)
                    data['mjd'].append(None)
                data['int_time'].append(vis.int_time)
                if nwave == 1:
                    data['vis2data'].append(vis.vis2data[0])
                    data['vis2err'].append(vis.vis2err[0])
                    data['flag'].append(vis.flag[0])
                else:
                    data['vis2data'].append(vis.vis2data)
                    data['vis2err'].append(vis.vis2err)
                    data['flag'].append(vis.flag)
                data['ucoord'].append(vis.ucoord)
                data['vcoord'].append(vis.vcoord)
                if vis.station[0] and vis.station[1]:
                    data['sta_index'].append([stationmap[id(vis.station[0])], stationmap[id(vis.station[1])]])
                else:
                    data['sta_index'].append([-1, -1])
            for key in list(tables.keys()):
                data = tables[key]
                nwave = self.wavelength[key[1]].eff_wave.size

                hdu = pyfits.new_table(pyfits.ColDefs([
                    pyfits.Column(name='TARGET_ID', format='1I', array=data['target_id']),
                    pyfits.Column(name='TIME', format='1D', unit='SECONDS', array=data['time']),
                    pyfits.Column(name='MJD', format='1D', unit='DAY', array=data['mjd']),
                    pyfits.Column(name='INT_TIME', format='1D', unit='SECONDS', array=data['int_time']),
                    pyfits.Column(name='VIS2DATA', format='%dD'%nwave, array=data['vis2data']),
                    pyfits.Column(name='VIS2ERR', format='%dD'%nwave, array=data['vis2err']),
                    pyfits.Column(name='UCOORD', format='1D', unit='METERS', array=data['ucoord']),
                    pyfits.Column(name='VCOORD', format='1D', unit='METERS', array=data['vcoord']),
                    pyfits.Column(name='STA_INDEX', format='2I', array=data['sta_index'], null=-1),
                    pyfits.Column(name='FLAG', format='%dL'%nwave, array=data['flag'])
                    ]))
                # Setting the data of logical field via the
                # pyfits.Column call above with length > 1 (eg
                # format='171L' above) seems to be broken, atleast as
                # of PyFITS 2.2.2
                hdu.data.field('FLAG').setfield(data['flag'], bool)
                hdu.header.update('EXTNAME', 'OI_VIS2')
                hdu.header.update('OI_REVN', 1, 'Revision number of the table definition')
                hdu.header.update('DATE-OBS', refdate.strftime('%F'), comment='Zero-point for table (UTC)')
                if key[0]: hdu.header.update('ARRNAME', key[0], 'Identifies corresponding OI_ARRAY')
                hdu.header.update('INSNAME', key[1], 'Identifies corresponding OI_WAVELENGTH table')
                hdulist.append(hdu)

        if self.t3.size:
            tables = {}
            for t3 in self.t3:
                nwave = t3.wavelength.eff_wave.size
                if t3.array:
                    key = (arraymap[id(t3.array)], wavelengthmap[id(t3.wavelength)])
                else:
                    key = (None, wavelengthmap[id(t3.wavelength)])
                if key in list(tables.keys()):
                    data = tables[key]
                else:
                    data = tables[key] = {'target_id':[], 'time':[], 'mjd':[], 'int_time':[],
                                          't3amp':[], 't3amperr':[], 't3phi':[], 't3phierr':[],
                                          'u1coord':[], 'v1coord':[], 'u2coord':[], 'v2coord':[],
                                          'sta_index':[], 'flag':[]}
                data['target_id'].append(targetmap[id(t3.target)])
                if t3.timeobs:
                    time = t3.timeobs - refdate
                    data['time'].append(time.days * 24.0 * 3600.0 + time.seconds)
                    mjd = (t3.timeobs - _mjdzero).days + (t3.timeobs - _mjdzero).seconds / 3600.0 / 24.0
                    data['mjd'].append(mjd)
                else:
                    data['time'].append(None)
                    data['mjd'].append(None)
                data['int_time'].append(t3.int_time)
                if nwave == 1:
                    data['t3amp'].append(t3.t3amp[0])
                    data['t3amperr'].append(t3.t3amperr[0])
                    data['t3phi'].append(t3.t3phi[0])
                    data['t3phierr'].append(t3.t3phierr[0])
                    data['flag'].append(t3.flag[0])
                else:
                    data['t3amp'].append(t3.t3amp)
                    data['t3amperr'].append(t3.t3amperr)
                    data['t3phi'].append(t3.t3phi)
                    data['t3phierr'].append(t3.t3phierr)
                    data['flag'].append(t3.flag)
                data['u1coord'].append(t3.u1coord)
                data['v1coord'].append(t3.v1coord)
                data['u2coord'].append(t3.u2coord)
                data['v2coord'].append(t3.v2coord)
                if t3.station[0] and t3.station[1] and t3.station[2]:
                    data['sta_index'].append([stationmap[id(t3.station[0])], stationmap[id(t3.station[1])], stationmap[id(t3.station[2])]])
                else:
                    data['sta_index'].append([-1, -1, -1])
            for key in list(tables.keys()):
                data = tables[key]
                nwave = self.wavelength[key[1]].eff_wave.size

                hdu = pyfits.new_table(pyfits.ColDefs((
                    pyfits.Column(name='TARGET_ID', format='1I', array=data['target_id']),
                    pyfits.Column(name='TIME', format='1D', unit='SECONDS', array=data['time']),
                    pyfits.Column(name='MJD', format='1D', unit='DAY', array=data['mjd']),
                    pyfits.Column(name='INT_TIME', format='1D', unit='SECONDS', array=data['int_time']),
                    pyfits.Column(name='T3AMP', format='%dD'%nwave, array=data['t3amp']),
                    pyfits.Column(name='T3AMPERR', format='%dD'%nwave, array=data['t3amperr']),
                    pyfits.Column(name='T3PHI', format='%dD'%nwave, unit='DEGREES', array=data['t3phi']),
                    pyfits.Column(name='T3PHIERR', format='%dD'%nwave, unit='DEGREES', array=data['t3phierr']),
                    pyfits.Column(name='U1COORD', format='1D', unit='METERS', array=data['u1coord']),
                    pyfits.Column(name='V1COORD', format='1D', unit='METERS', array=data['v1coord']),
                    pyfits.Column(name='U2COORD', format='1D', unit='METERS', array=data['u2coord']),
                    pyfits.Column(name='V2COORD', format='1D', unit='METERS', array=data['v2coord']),
                    pyfits.Column(name='STA_INDEX', format='3I', array=data['sta_index'], null=-1),
                    pyfits.Column(name='FLAG', format='%dL'%nwave, array=data['flag'])
                    )))
                # Setting the data of logical field via the
                # pyfits.Column call above with length > 1 (eg
                # format='171L' above) seems to be broken, atleast as
                # of PyFITS 2.2.2
                hdu.data.field('FLAG').setfield(data['flag'], bool)
                hdu.header.update('EXTNAME', 'OI_T3')
                hdu.header.update('OI_REVN', 1, 'Revision number of the table definition')
                hdu.header.update('DATE-OBS', refdate.strftime('%F'), 'Zero-point for table (UTC)')
                if key[0]: hdu.header.update('ARRNAME', key[0], 'Identifies corresponding OI_ARRAY')
                hdu.header.update('INSNAME', key[1], 'Identifies corresponding OI_WAVELENGTH table')
                hdulist.append(hdu)

        hdulist.writeto(filename, clobber=True)



def open(filename, quiet=False):
    """Open an OIFITS file."""

    newobj = oifits()
    targetmap = {}
    sta_indices = {}

    if not quiet:
        print("Opening %s"%filename)
    hdulist = pyfits.open(filename)
    # First get all the OI_TARGET, OI_WAVELENGTH and OI_ARRAY tables
    for hdu in hdulist:
        header = hdu.header
        data = hdu.data
        if hdu.name == 'OI_WAVELENGTH':
            if newobj.wavelength == None: newobj.wavelength = {}
            insname = header['INSNAME']
            newobj.wavelength[insname] = OI_WAVELENGTH(data.field('EFF_WAVE'), data.field('EFF_BAND'))
        elif hdu.name == 'OI_TARGET':
            for row in data:
                target_id = row['TARGET_ID']
                target = OI_TARGET(target=row['TARGET'], raep0=row['RAEP0'], decep0=row['DECEP0'],
                                   equinox=row['EQUINOX'], ra_err=row['RA_ERR'], dec_err=row['DEC_ERR'],
                                   sysvel=row['SYSVEL'], veltyp=row['VELTYP'], veldef=row['VELDEF'],
                                   pmra=row['PMRA'], pmdec=row['PMDEC'], pmra_err=row['PMRA_ERR'],
                                   pmdec_err=row['PMDEC_ERR'], parallax=row['PARALLAX'],
                                   para_err=row['PARA_ERR'], spectyp=row['SPECTYP'])
                newobj.target = np.append(newobj.target, target)
                targetmap[target_id] = target
        elif hdu.name == 'OI_ARRAY':
            if newobj.array == None: newobj.array = {}
            arrname = header['ARRNAME']
            frame = header['FRAME']
            arrxyz = np.array([header['ARRAYX'], header['ARRAYY'], header['ARRAYZ']])
            newobj.array[arrname] = OI_ARRAY(frame, arrxyz, stations=data)
            # Save the sta_index for each array, as we will need it
            # later to match measurements to stations
            sta_indices[arrname] = data.field('sta_index')

    # Then get any science measurements
    for hdu in hdulist:
        header = hdu.header
        data = hdu.data
        if hdu.name in ('OI_VIS', 'OI_VIS2', 'OI_T3'):
            if 'ARRNAME' in list(header.keys()):
                arrname = header['ARRNAME']
            else:
                arrname = None
            if arrname and newobj.array:
                array = newobj.array[arrname]
            else:
                array = None
            wavelength = newobj.wavelength[header['INSNAME']]
        if hdu.name == 'OI_VIS':
            for row in data:
                date = header['DATE-OBS'].split('-')
                timeobs = datetime.datetime(int(date[0]), int(date[1]), int(date[2])) + datetime.timedelta(seconds=np.around(row.field('TIME'), 2))
                int_time = row.field('INT_TIME')
                visamp = np.reshape(row.field('VISAMP'), -1)
                visamperr = np.reshape(row.field('VISAMPERR'), -1)
                visphi = np.reshape(row.field('VISPHI'), -1)
                visphierr = np.reshape(row.field('VISPHIERR'), -1)
                if 'CFLUX' in row.array.names: cflux = np.reshape(row.field('CFLUX'), -1)
                else: cflux = None
                if 'CFLUXERR' in row.array.names: cfluxerr = np.reshape(row.field('CFLUXERR'), -1)
                else: cfluxerr = None
                flag = np.reshape(row.field('FLAG'), -1)
                ucoord = row.field('UCOORD')
                vcoord = row.field('VCOORD')
                target = targetmap[row.field('TARGET_ID')]
                if array:
                    sta_index = row.field('STA_INDEX')
                    s1 = array.station[sta_indices[arrname] == sta_index[0]][0]
                    s2 = array.station[sta_indices[arrname] == sta_index[1]][0]
                    station = [s1, s2]
                else:
                    station = [None, None]
                newobj.vis = np.append(newobj.vis, OI_VIS(timeobs=timeobs, int_time=int_time, visamp=visamp,
                                                          visamperr=visamperr, visphi=visphi, visphierr=visphierr,
                                                          flag=flag, ucoord=ucoord, vcoord=vcoord, wavelength=wavelength,
                                                          target=target, array=array, station=station, cflux=cflux,
                                                          cfluxerr=cfluxerr))
        elif hdu.name == 'OI_VIS2':
            for row in data:
                date = header['DATE-OBS'].split('-')
                timeobs = datetime.datetime(int(date[0]), int(date[1]), int(date[2])) + datetime.timedelta(seconds=np.around(row.field('TIME'), 2))
                int_time = row.field('INT_TIME')
                vis2data = np.reshape(row.field('VIS2DATA'), -1)
                vis2err = np.reshape(row.field('VIS2ERR'), -1)
                flag = np.reshape(row.field('FLAG'), -1)
                ucoord = row.field('UCOORD')
                vcoord = row.field('VCOORD')
                target = targetmap[row.field('TARGET_ID')]
                if array:
                    sta_index = row.field('STA_INDEX')
                    s1 = array.station[sta_indices[arrname] == sta_index[0]][0]
                    s2 = array.station[sta_indices[arrname] == sta_index[1]][0]
                    station = [s1, s2]
                else:
                    station = [None, None]
                newobj.vis2 = np.append(newobj.vis2, OI_VIS2(timeobs=timeobs, int_time=int_time, vis2data=vis2data,
                                                             vis2err=vis2err, flag=flag, ucoord=ucoord, vcoord=vcoord,
                                                             wavelength=wavelength, target=target, array=array,
                                                             station=station))
        elif hdu.name == 'OI_T3':
            for row in data:
                date = header['DATE-OBS'].split('-')
                timeobs = datetime.datetime(int(date[0]), int(date[1]), int(date[2])) + datetime.timedelta(seconds=np.around(row.field('TIME'), 2))
                int_time = row.field('INT_TIME')
                t3amp = np.reshape(row.field('T3AMP'), -1)
                t3amperr = np.reshape(row.field('T3AMPERR'), -1)
                t3phi = np.reshape(row.field('T3PHI'), -1)
                t3phierr = np.reshape(row.field('T3PHIERR'), -1)
                flag = np.reshape(row.field('FLAG'), -1)
                u1coord = row.field('U1COORD')
                v1coord = row.field('V1COORD')
                u2coord = row.field('U2COORD')
                v2coord = row.field('V2COORD')
                target = targetmap[row.field('TARGET_ID')]
                if array:
                    sta_index = row.field('STA_INDEX')
                    s1 = array.station[sta_indices[arrname] == sta_index[0]][0]
                    s2 = array.station[sta_indices[arrname] == sta_index[1]][0]
                    s3 = array.station[sta_indices[arrname] == sta_index[2]][0]
                    station = [s1, s2, s3]
                else:
                    station = [None, None, None]
                newobj.t3 = np.append(newobj.t3, OI_T3(timeobs=timeobs, int_time=int_time, t3amp=t3amp,
                                                       t3amperr=t3amperr, t3phi=t3phi, t3phierr=t3phierr,
                                                       flag=flag, u1coord=u1coord, v1coord=v1coord, u2coord=u2coord,
                                                       v2coord=v2coord, wavelength=wavelength, target=target,
                                                       array=array, station=station))

    hdulist.close()
    if not quiet:
        newobj.info(recursive=False)

    return newobj
