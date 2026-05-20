# array.py
# a interferometric telescope array class
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


import copy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

import ehtim.const_def as ehc
import ehtim.io.load
import ehtim.io.save
import ehtim.observing.obs_helpers as obsh
import ehtim.observing.obs_simulate as simobs
from ehtim.caltable import plot_tarr_dterms

###################################################################################################
# Array object
###################################################################################################

# Valid two-character feed_type strings for a single station (lowercase).
# Mixed-basis stations (e.g. Effelsberg R+X) are explicitly enumerated.
VALID_FEED_TYPES = frozenset({
    'rl', 'lr', 'xy', 'yx',
    'rx', 'ry', 'lx', 'ly',
    'xr', 'xl', 'yr', 'yl',
})

# Default symmetric SEFD when none is supplied (legacy convention).
_DEFAULT_SEFD = 10000.0


def _validate_feed_type(feed_type):
    """Raise ValueError if feed_type is not a recognised 2-char feed string."""
    if not isinstance(feed_type, str) or feed_type.lower() not in VALID_FEED_TYPES:
        raise ValueError(
            f"feed_type must be one of {sorted(VALID_FEED_TYPES)}; "
            f"got {feed_type!r}")


def _resolve_sefd_pair(sefd, sefd_p1, sefd_p2):
    """Resolve (sefd_p1, sefd_p2) from legacy + generic kwargs.

    Rules: pass exactly one of `sefd` (symmetric, both feeds) or the
    `sefd_p1`/`sefd_p2` pair (asymmetric, per-feed). Mixing the legacy
    `sefd` kwarg with either generic kwarg raises. The generic kwargs
    must be supplied as a pair. If none are given, both feeds default
    to _DEFAULT_SEFD.
    """
    any_generic = (sefd_p1 is not None) or (sefd_p2 is not None)
    if sefd is not None and any_generic:
        raise ValueError(
            "pass only one of sefd (legacy, symmetric) or "
            "sefd_p1/sefd_p2 (generic, per-feed), not both")
    if any_generic:
        if sefd_p1 is None or sefd_p2 is None:
            raise ValueError(
                "sefd_p1 and sefd_p2 must be supplied together")
        return float(sefd_p1), float(sefd_p2)
    val = float(sefd) if sefd is not None else _DEFAULT_SEFD
    return val, val


class TarrView:
    """Thin wrapper around an Array's underlying telescope-array recarray.

    Forwards array-like operations (indexing, iteration, length, dtype
    lookup, numpy interop, equality, pickling) to the wrapped recarray.
    The wrapper exists to guard column-form access of legacy feed-specific
    field names ('sefdr', 'sefdl', 'dr', 'dl') on arrays whose stations
    are not all 'rl'-feed: in that case the title-alias accessor would
    return values from the underlying generic slot (sefd_p1, d_p1, etc.),
    which on a non-RL station is the wrong-feed value. TarrView raises
    KeyError instead and points to the per-feed accessor.

    The recarray remains the storage truth; TarrView is created on the
    fly by Array.tarr's property getter and adds no persisted state.

    Note: this guard catches whole-column access (tarr['sefdr']). It does
    not catch row-form access (tarr[i]['sefdr']) — the row is a numpy
    void object owned by numpy. Phase 2+ code should prefer
    Array.sefd_for_feed / Array.dterm_for_feed for per-station lookups.
    """

    _LEGACY_FIELD_HINT = {
        'sefdr': 'Array.sefd_for_feed(site, feed)',
        'sefdl': 'Array.sefd_for_feed(site, feed)',
        'dr': 'Array.dterm_for_feed(site, feed)',
        'dl': 'Array.dterm_for_feed(site, feed)',
    }

    __slots__ = ('_tarr',)

    # Mark unhashable like the underlying ndarray.
    __hash__ = None

    def __init__(self, tarr):
        object.__setattr__(self, '_tarr', tarr)

    # ---- guard --------------------------------------------------------------
    def _is_homogeneous_rl(self):
        fts = self._tarr['feed_type']
        if len(fts) == 0:
            return True  # empty arrays default to legacy RL semantics
        s = set(str(ft).lower() for ft in fts)
        return s == {'rl'}

    def _guard(self, key):
        if isinstance(key, str) and key in self._LEGACY_FIELD_HINT \
                and not self._is_homogeneous_rl():
            fts = sorted({str(ft) for ft in self._tarr['feed_type']})
            hint = self._LEGACY_FIELD_HINT[key]
            raise KeyError(
                f"tarr[{key!r}] is undefined on a non-RL or mixed-feed array "
                f"(feed_types={fts}); use {hint} instead")

    # ---- core forwarding ----------------------------------------------------
    def __getitem__(self, key):
        self._guard(key)
        result = self._tarr[key]
        # If the result is itself a structured-dtype slice (boolean mask,
        # slice object, fancy index, ...), re-wrap so the guard stays in
        # effect. Single-row integer indexing produces a numpy void and
        # falls through unwrapped (documented limitation).
        if isinstance(result, np.ndarray) and result.dtype.names is not None \
                and 'feed_type' in result.dtype.names:
            return TarrView(result)
        return result

    def __setitem__(self, key, value):
        # Setting is delegated unguarded: callers writing whole columns
        # are responsible for their data. Read-side guards still apply.
        self._tarr[key] = value

    def __iter__(self):
        return iter(self._tarr)

    def __len__(self):
        return len(self._tarr)

    def __array__(self, dtype=None, copy=None):
        # Return the underlying ndarray directly; calling np.asarray() here
        # would strip the structured dtype (numpy quirk with title aliases).
        if dtype is None:
            return self._tarr.copy() if copy else self._tarr
        out = self._tarr.astype(dtype)
        return out.copy() if copy else out

    def __eq__(self, other):
        if isinstance(other, TarrView):
            other = other._tarr
        return self._tarr == other

    def __ne__(self, other):
        if isinstance(other, TarrView):
            other = other._tarr
        return self._tarr != other

    def __repr__(self):
        return f"TarrView({self._tarr!r})"

    # ---- attribute forwarding -----------------------------------------------
    def __getattr__(self, name):
        # __getattr__ only fires for misses; forward .dtype, .shape, .names,
        # .copy, .view, etc. to the underlying ndarray.
        #
        # Numpy's __array_interface__ / __array_struct__ on a structured
        # ndarray report 'typestr=V<nbytes>', which strips field names when
        # numpy uses them to construct a copy. Hide them so np.asarray()
        # falls back to our __array__ instead.
        if name in ('__array_interface__', '__array_struct__'):
            raise AttributeError(name)
        return getattr(self._tarr, name)

    # ---- pickle support -----------------------------------------------------
    def __reduce__(self):
        # Reconstruct via TarrView(recarray) on the other end.
        return (TarrView, (self._tarr,))


def _resolve_dterm_pair(dr, dl, d_p1, d_p2, feed_type):
    """Resolve (d_p1, d_p2) from legacy dr/dl + generic d_p1/d_p2 kwargs.

    `dr`/`dl` are only valid for feed_type='rl' and cannot be combined
    with their generic counterpart on the same feed. Missing values
    default to 0+0j (no leakage).
    """
    if feed_type.lower() != 'rl' and (dr is not None or dl is not None):
        raise ValueError(
            f"dr/dl kwargs are only valid for feed_type='rl' "
            f"(got feed_type={feed_type!r}); use d_p1/d_p2 instead")
    if dr is not None and d_p1 is not None:
        raise ValueError(
            "pass only one of dr (legacy) or d_p1 (generic), not both")
    if dl is not None and d_p2 is not None:
        raise ValueError(
            "pass only one of dl (legacy) or d_p2 (generic), not both")

    if d_p1 is not None:
        d_p1_val = complex(d_p1)
    elif dr is not None:
        d_p1_val = complex(dr)
    else:
        d_p1_val = 0 + 0j
    if d_p2 is not None:
        d_p2_val = complex(d_p2)
    elif dl is not None:
        d_p2_val = complex(dl)
    else:
        d_p2_val = 0 + 0j
    return d_p1_val, d_p2_val


class Array:

    """A VLBI array of telescopes with site locations, SEFDs, and other data.

       Attributes:
           tarr (numpy.recarray): The array of telescope data with datatype DTARR
           tkey (dict): A dictionary of rows in the tarr for each site name
           ephem (dict): A dictionary of TLEs for each space antenna,
                         Space antennas have x=y=z=0 in the tarr
    """

    def __init__(self, tarr, ephem={}):
        self.tarr = ehc.upgrade_tarr(tarr)
        self.ephem = ephem

        # check to see if ephemeris is correct
        for line in self.tarr:
            if np.any(np.isnan([line['x'], line['y'], line['z']])):
                sitename = str(line['site'])
                try:
                    elen = len(ephem[sitename])
                except NameError:
                    raise Exception(f'no ephemeris for site {sitename} !')
                if elen != 3:
                    raise Exception(f'wrong ephemeris format for site {sitename} !')

        # Dictionary of array indices for site names
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}

    def __setstate__(self, state):
        # Silently upgrade legacy pickles to the current mixedpol schema.
        if 'tarr' in state:
            state['tarr'] = ehc.upgrade_tarr(state['tarr'])
        if '_tarr' in state:
            state['_tarr'] = ehc.upgrade_tarr(state['_tarr'])
        self.__dict__.update(state)

    @property
    def tarr(self):
        return TarrView(self._tarr)

    @tarr.setter
    def tarr(self, tarr):
        if isinstance(tarr, TarrView):
            tarr = tarr._tarr
        self._tarr = tarr
        self.tkey = {tarr[i]['site']: i for i in range(len(tarr))}

    def is_homogeneous_feeds(self):
        """Return True if every station shares the same feed_type.

           Returns:
                bool : True for single-feed arrays (legacy 'rl', all-'xy', ...);
                       False for mixed-feed arrays.
        """
        return len(set(self._tarr['feed_type'])) <= 1

    def feed_types(self):
        """Return the set of distinct feed types present in the array.

           Returns:
                set[str] : e.g. {'rl'} for a legacy array, {'rl', 'xy'} for
                           a mixed circular+linear array.
        """
        return set(str(ft) for ft in self._tarr['feed_type'])

    def _row_and_feed_index(self, site, feed):
        """Resolve (row, slot_index) for site/feed lookup; raise on miss.

           slot_index is 0 if feed matches the station's first feed channel,
           1 if it matches the second. feed and feed_type are compared
           case-insensitively.
        """
        if site not in self.tkey:
            raise KeyError(
                f"site {site!r} not in array (have: {sorted(self.tkey)})")
        row = self._tarr[self.tkey[site]]
        ft = str(row['feed_type']).lower()
        feed_lc = str(feed).lower()
        if len(feed_lc) != 1 or len(ft) != 2:
            raise ValueError(
                f"feed must be a single character and feed_type a 2-char "
                f"string; got feed={feed!r}, feed_type={ft!r}")
        if feed_lc == ft[0]:
            return row, 0
        if feed_lc == ft[1]:
            return row, 1
        raise ValueError(
            f"feed {feed!r} not in feed_type {ft!r} of site {site!r}")

    def sefd_for_feed(self, site, feed):
        """Return the SEFD for a single feed channel of a station.

           Args:
                site (str) : station name (key in self.tkey)
                feed (str) : single feed character; must match one of the two
                             feed channels of the station's feed_type
                             (e.g. 'R'/'L' for 'rl', 'X'/'Y' for 'xy').
                             Case-insensitive.

           Returns:
                float : SEFD in Jy
        """
        row, slot = self._row_and_feed_index(site, feed)
        return float(row['sefd_p1' if slot == 0 else 'sefd_p2'])

    def dterm_for_feed(self, site, feed):
        """Return the leakage D-term for a single feed channel of a station.

           Args:
                site (str) : station name (key in self.tkey)
                feed (str) : single feed character; must match one of the two
                             feed channels of the station's feed_type.
                             Case-insensitive.

           Returns:
                complex : leakage D-term
        """
        row, slot = self._row_and_feed_index(site, feed)
        return complex(row['d_p1' if slot == 0 else 'd_p2'])

    def copy(self):
        """Copy the array object.

           Args:

           Returns:
               (Array): a copy of the Array object.
        """

        newarr = copy.deepcopy(self)
        return newarr


    def listbls(self):
        """List all baselines.

           Args:
           Returns:
                numpy.array : array of baselines
        """

        bls = []
        for i1 in sorted(self.tarr['site']):
            for i2 in sorted(self.tarr['site']):
                if [i1, i2] not in bls and [i2, i1] not in bls and i1 != i2:
                    bls.append([i1, i2])
        bls = np.array(bls)

        return bls

    def obsdata(self, ra, dec, rf, bw, tint, tadv, tstart, tstop,
                mjd=ehc.MJD_DEFAULT, timetype='UTC', polrep='stokes',
                elevmin=ehc.ELEV_LOW, elevmax=ehc.ELEV_HIGH,
                no_elevcut_space=False,
                tau=ehc.TAUDEF, fix_theta_GMST=False):
        """Generate u,v points and baseline uncertainties.

           Args:
               ra (float): the source right ascension in fractional hours
               dec (float): the source declination in fractional degrees
               tint (float): the scan integration time in seconds
               tadv (float): the uniform cadence between scans in seconds
               tstart (float): the start time of the observation in hours
               tstop (float): the end time of the observation in hours

               mjd (int): the mjd of the observation
               timetype (str): how to interpret tstart and tstop; either 'GMST' or 'UTC'
               polrep (str): polarization representation, one of
                             {'stokes', 'circ', 'lin', 'mixed'}. 'circ' requires
                             all-RL feeds; 'lin' requires all-XY feeds; 'mixed'
                             requires at least two distinct feed types. 'lin'
                             and 'mixed' are not yet wired through the
                             simulation backend (Phase 5).
               elevmin (float): station minimum elevation in degrees
               elevmax (float): station maximum elevation in degrees
               no_elevcut_space (bool): if True, do not apply elevation cut to orbiters
               tau (float): the base opacity at all sites, or a dict giving one opacity per site
               fix_theta_GMST (bool): if True, stops earth rotation to sample fixed u,v points

           Returns:
               Obsdata: an observation object with no data

        """
        valid_polreps = ('stokes', 'circ', 'lin', 'mixed')
        if polrep not in valid_polreps:
            raise ValueError(
                f"polrep must be one of {valid_polreps}; got {polrep!r}")

        feeds = self.feed_types()
        if polrep == 'circ' and feeds != {'rl'}:
            raise ValueError(
                f"polrep='circ' requires all stations to have feed_type='rl'; "
                f"array has feed_types={sorted(feeds)}")
        if polrep == 'lin' and feeds != {'xy'}:
            raise ValueError(
                f"polrep='lin' requires all stations to have feed_type='xy'; "
                f"array has feed_types={sorted(feeds)}")
        if polrep == 'mixed' and len(feeds) < 2:
            raise ValueError(
                f"polrep='mixed' requires at least two distinct feed types; "
                f"array has feed_types={sorted(feeds)}")
        if polrep in ('lin', 'mixed'):
            raise NotImplementedError(
                f"Array.obsdata(polrep={polrep!r}) is not yet supported by "
                f"the simulation backend (see Phase 5 of "
                f"obsdata_mixedpol_plan_v2.md)")

        obsarr = simobs.make_uvpoints(self, ra, dec, rf, bw,
                                      tint, tadv, tstart, tstop,
                                      mjd=mjd, polrep=polrep, tau=tau,
                                      elevmin=elevmin, elevmax=elevmax,
                                      no_elevcut_space=no_elevcut_space,
                                      timetype=timetype, fix_theta_GMST=fix_theta_GMST)

        uniquetimes = np.sort(np.unique(obsarr['time']))
        scans = np.array([[time - 0.5 * tadv, time + 0.5 * tadv] for time in uniquetimes])
        source = str(ra) + ":" + str(dec)
        obs = ehtim.obsdata.Obsdata(ra, dec, rf, bw, obsarr, self.tarr,
                                    source=source, mjd=mjd, timetype=timetype, polrep=polrep,
                                    ampcal=True, phasecal=True, opacitycal=True,
                                    dcal=True, frcal=True,
                                    scantable=scans)
        return obs

    def make_subarray(self, sites):
        """Make a subarray from the Array object array that only includes the sites listed.

           Args:
               sites (list) : list of sites in the subarray
           Returns:
               Array: an Array object with specified sites and metadata
        """
        all_sites = [t[0] for t in self.tarr]
        mask = np.array([t in sites for t in all_sites])
        subarr = Array(self.tarr[mask], ephem=self.ephem)
        return subarr

    def save_txt(self, fname):
        """Save the array data in a text file.

           Args:
               fname (str) : path to output array file
        """
        ehtim.io.save.save_array_txt(self, fname)
        return

    def plot_dterms(self, sites='all', label=None, legend=True, clist=ehc.SCOLORS,
                    rangex=False, rangey=False, markersize=2 * ehc.MARKERSIZE,
                    show=True, grid=True, export_pdf=""):
        """Make a plot of the D-terms.

           Args:
               sites (list) : list of sites to plot
               label (str) : title for plot
               legend (bool) : add telescope legend or not
               clist (list) : list of colors for different stations
               rangex (list) : lower and upper x-axis limits
               rangey (list) : lower and upper y-axis limits
               markersize (float) : marker size
               show (bool) : display the plot or not
               grid (bool) : add a grid to the plot or not
               export_pdf (str) : save a pdf file to this path

           Returns:
               matplotlib.axes
        """
        # sites
        if (isinstance(sites,str) and sites.lower() == 'all'):
            sites = list(self.tkey.keys())

        if isinstance(sites,str):
            sites = [sites]

        if len(sites)==0:
            sites = list(self.tkey.keys())

        keys = [self.tkey[site] for site in sites]

        axes = plot_tarr_dterms(self.tarr, keys=keys, label=label, legend=legend, clist=clist,
                                rangex=rangex, rangey=rangey, markersize=markersize,
                                show=show, grid=grid, export_pdf=export_pdf)

        return axes

    def add_site(self, site, coords, sefd=None,
                 fr_par=0, fr_elev=0, fr_off=0,
                 dr=None, dl=None,
                 feed_type='rl',
                 sefd_p1=None, sefd_p2=None,
                 d_p1=None, d_p2=None):

        """Add a ground station to the array

           Pass either the legacy symmetric kwargs (`sefd`, `dr`, `dl` — only
           valid for feed_type='rl') OR the generic per-feed kwargs
           (`sefd_p1`/`sefd_p2`, `d_p1`/`d_p2`). Mixing legacy and generic
           kwargs for the same field raises ValueError.

           Args:
               site (str): site name
               coords (tuple): (x, y, z) station coordinates in meters
               sefd (float): legacy symmetric SEFD applied to both feeds.
                             Defaults to 10000 Jy when none of sefd / sefd_p1
                             / sefd_p2 is supplied.
               fr_par, fr_elev, fr_off (float): field-rotation coefficients
               dr (complex): legacy D-term for the R feed; only valid when
                             feed_type='rl'.
               dl (complex): legacy D-term for the L feed; only valid when
                             feed_type='rl'.
               feed_type (str): two-character feed-type string, lowercase,
                                drawn from VALID_FEED_TYPES. Default 'rl'
                                preserves the legacy calling convention.
               sefd_p1, sefd_p2 (float): per-feed SEFDs. Must be supplied
                                         together.
               d_p1, d_p2 (complex): per-feed D-terms. Default 0+0j.

           Returns:
               Array: a new Array with the station appended.
        """
        _validate_feed_type(feed_type)
        sefd_p1_val, sefd_p2_val = _resolve_sefd_pair(sefd, sefd_p1, sefd_p2)
        d_p1_val, d_p2_val = _resolve_dterm_pair(dr, dl, d_p1, d_p2, feed_type)

        tarr_old = self.tarr.copy()
        ephem_old = self.ephem.copy()

        tarr_newline = np.array((str(site), float(coords[0]), float(coords[1]), float(coords[2]),
                                 sefd_p1_val, sefd_p2_val,
                                 d_p1_val, d_p2_val,
                                 float(fr_par), float(fr_elev), float(fr_off),
                                 feed_type.lower()), dtype=ehc.DTARR)
        tarr_new = np.append(tarr_old, tarr_newline)

        arr_out = Array(tarr_new, ephem_old)
        return arr_out

    def remove_site(self, site):
        """Remove a site from the array

        """
        tarr_old = self.tarr.copy()
        ephem_old = self.ephem.copy()
        ephem_new = ephem_old.copy()

        try:
            tarr_new = np.delete(tarr_old.copy(), self.tkey[site])
        except (KeyError, IndexError):
            raise Exception(f"could not find site {site} to delete from Array.tarr!")

        if site in ephem_old.keys():
            try:
                ephem_new.pop(site)
            except KeyError:
                raise Exception(f"could not find ephemeris for site {site} to delete from Array.ephem!")

        arr_out = Array(tarr_new, ephem_new)
        return arr_out

    def add_satellite_tle(self, tlearr, sefd=None,
                          feed_type='rl',
                          sefd_p1=None, sefd_p2=None,
                          d_p1=None, d_p2=None):

        """Add an earth-orbiting satellite to the array from a TLE

           Args:
             tlearr (str) : 3 element list with [name, tle line 1, tle line 2] as strings
             sefd (float) : legacy symmetric SEFD (sefdl = sefdr); defaults to 10000.
             feed_type (str): two-character feed_type string (default 'rl').
             sefd_p1, sefd_p2 (float): per-feed SEFDs; pass together as an
                                       alternative to `sefd`.
             d_p1, d_p2 (complex): per-feed D-terms.
        """
        _validate_feed_type(feed_type)
        sefd_p1_val, sefd_p2_val = _resolve_sefd_pair(sefd, sefd_p1, sefd_p2)
        d_p1_val, d_p2_val = _resolve_dterm_pair(None, None, d_p1, d_p2, feed_type)

        satname = tlearr[0]
        tarr_new = self.tarr.copy()
        ephem_new = self.ephem.copy()

        tarr_newline = np.array((str(satname), 0., 0., 0.,
                                 sefd_p1_val, sefd_p2_val,
                                 d_p1_val, d_p2_val,
                                 0., 0., 0., feed_type.lower()), dtype=ehc.DTARR)
        tarr_new = np.append(tarr_new, tarr_newline)
        ephem_new[satname] = tlearr
        arr_out = Array(tarr_new, ephem_new)

        return arr_out

    def add_satellite_elements(self, satname,
                               perigee_mjd=Time.now().mjd,
                               period_days=1., eccentricity=0.,
                               inclination=0., arg_perigee=0., long_ascending=0.,
                               sefd=None,
                               feed_type='rl',
                               sefd_p1=None, sefd_p2=None,
                               d_p1=None, d_p2=None):
        """Add an earth-orbiting satellite to the array from simple keplerian elements
           perfect keplerian orbit is assumed, no derivatives

           Args:
               perigee time given in mjd
               period given in days
               inclination, arg_perigee, long_ascending given in degrees
               sefd (float): legacy symmetric SEFD (defaults to 10000).
               feed_type (str): two-character feed_type string (default 'rl').
               sefd_p1, sefd_p2 (float): per-feed SEFDs; pass together as an
                                         alternative to `sefd`.
               d_p1, d_p2 (complex): per-feed D-terms.
        """
        _validate_feed_type(feed_type)
        sefd_p1_val, sefd_p2_val = _resolve_sefd_pair(sefd, sefd_p1, sefd_p2)
        d_p1_val, d_p2_val = _resolve_dterm_pair(None, None, d_p1, d_p2, feed_type)

        tarr_new = self.tarr.copy()
        ephem_new = self.ephem.copy()

        tarr_newline = np.array((str(satname), 0., 0., 0.,
                                 sefd_p1_val, sefd_p2_val,
                                 d_p1_val, d_p2_val,
                                 0., 0., 0., feed_type.lower()), dtype=ehc.DTARR)
        tarr_new = np.append(tarr_new, tarr_newline)

        ephem_new[satname] = [perigee_mjd, period_days, eccentricity, inclination, arg_perigee, long_ascending]
        arr_out = Array(tarr_new, ephem_new)

        return arr_out

    def plot_satellite_orbits(self, tstart_mjd=Time.now().mjd, tstop_mjd=Time.now().mjd+1, npoints=1000):
        earth_radius_polar = 6357. #km
        earth_radius_eq = 6378.

        fig = plt.figure(figsize=(18,6))
        gs = matplotlib.gridspec.GridSpec(1,3,width_ratios=[1,1,1])

        satellites = self.ephem.keys()
        for i,satellite in enumerate(satellites):

            if i==0:
                color='k'
            else:
                color=ehc.SCOLORS[i-1]

            # get skyfield satelllite object
            if len(self.ephem[satellite])==3: # TLE
                line1 = self.ephem[satellite][1]
                line2 = self.ephem[satellite][2]
                sat = obsh.sat_skyfield_from_tle(satellite, line1, line2)
            elif len(self.ephem[satellite])==6: #keplerian elements
                elements = self.ephem[satellite]
                sat = obsh.sat_skyfield_from_elements(satellite, tstart_mjd,
                                                      elements[0],elements[1],elements[2],elements[3],elements[4],elements[5])
            else:
                raise Exception(f"ephemeris format not recognized for {satellite}")

            # get GCRS positions
            fracmjds = np.linspace(tstart_mjd, tstop_mjd, npoints)
            positions = obsh.orbit_skyfield(sat, fracmjds, whichout='gcrs')
            positions *= 1.e-3 # convert to km
            distances = np.sqrt(positions[0]**2 + positions[1]**2 + positions[2]**2)
            maxdist = np.max(distances)

            ax1 = fig.add_subplot(gs[0])
            ax1.set_aspect(1)
            plt.plot(positions[0], positions[1], color=color, marker='.',ls='None')
            circle1 = matplotlib.patches.Circle((0, 0), earth_radius_eq, color='b')
            plt.gca().add_patch(circle1)
            plt.xlabel('x (km)')
            plt.ylabel('y (km)')
            plt.xlim(-1.1*maxdist, 1.1*maxdist)
            plt.ylim(-1.1*maxdist, 1.1*maxdist)
            plt.grid()

            ax2 = fig.add_subplot(gs[1])
            ax2.set_aspect(1)
            plt.plot(positions[1], positions[2], color=color, marker='.',ls='None')
            circle1 = matplotlib.patches.Ellipse((0, 0), 2*earth_radius_eq, 2*earth_radius_polar, color='b')
            plt.gca().add_patch(circle1)
            plt.xlabel('y (km)')
            plt.ylabel('z (km)')
            plt.xlim(-1.1*maxdist, 1.1*maxdist)
            plt.ylim(-1.1*maxdist, 1.1*maxdist)
            plt.grid()

            ax3 = fig.add_subplot(gs[2])
            ax3.set_aspect(1)
            plt.plot(positions[0], positions[2], color=color, marker='.',ls='None', label=satellite)
            circle1 = matplotlib.patches.Ellipse((0, 0), 2*earth_radius_eq, 2*earth_radius_polar, color='b')
            plt.gca().add_patch(circle1)
            plt.xlabel('x (km)')
            plt.ylabel('z (km)')
            plt.xlim(-1.1*maxdist, 1.1*maxdist)
            plt.ylim(-1.1*maxdist, 1.1*maxdist)
            plt.legend(frameon=False,loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid()

        plt.subplots_adjust(wspace=1)
        ehc.show_noblock()
        return

##########################################################################
# Array creation functions
##########################################################################


def load_txt(fname, ephemdir='ephemeris'):
    """Read an array from a text file.
       Sites with x=y=z=0 are spacecraft, TLE ephemerides read from ephemdir.

       Args:
           fname (str) : path to input array file
           ephemdir (str) : path to directory with TLE ephemerides for spacecraft
       Returns:
           Array: an Array object loaded from file
    """

    return ehtim.io.load.load_array_txt(fname, ephemdir=ephemdir)
