# caltable.py
# a calibration table class
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
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

import ehtim.const_def as ehc
import ehtim.io.load
import ehtim.io.save
import ehtim.observing.obs_helpers as obsh
from ehtim.warnings import MixedPolConventionWarning

##################################################################################################
# Caltable object
##################################################################################################

# NOTE: caltables loaded from old data carry no D-term tables; any D-term
# information stays in the tarr, assumed fixed over the observation.
# TODO: when the new calibration module lands, decide what happens when both
# the tarr and the caltable define D-terms (zero the tarr? warn and prefer
# the table?).

# Any optional time-dependent D-term tables are saved beside the gain files,
# opened with a version line. Gain files keep their headerless 5-column format.
DTERM_FILE_SUFFIX = '_dterms.txt'
DTERM_FILE_HEADER = '# ehtim caltable dterms format v1: time_mjd d1re d1im d2re d2im'


class Caltable:
    """A calibration table of time-dependent station gains and leakage (D-terms).

       Gains and D-terms live in separate per-site tables, each with its own
       time column; a single-row table represents a time-constant quantity.
       A caltable with no D-term table falls back to the D-terms in the tarr,
       assumed fixed over the observation.
       TODO: flag or resolve cases where both the tarr and the caltable
       define D-terms.

       Attributes:
           source (str): The source name
           ra (float): The source Right Ascension in fractional hours
           dec (float): The source declination in fractional degrees
           mjd (int): The integer MJD of the observation
           rf (float): The observation frequency in Hz
           bw (float): The observation bandwidth in Hz
           timetype (str): How to interpret tstart and tstop; either 'GMST' or 'UTC'

           tarr (numpy.recarray): The array of telescope data with datatype DTARR
           tkey (dict): A dictionary of rows in the tarr for each site name

           gains (dict): keys are sites in tarr, entries are gain tables of type DTCAL
           dterms (dict): keys are sites in tarr, entries are D-term tables of type
                          DTDTERM. Empty when the table carries no leakage solution.
           data (dict): backwards-compatible alias for ``gains``; it is the live
                        dict, so in-place mutation through it is visible on the table

    """

    def __init__(self, ra, dec, rf, bw, datadict, tarr, dtermdict=None,
                 source=ehc.SOURCE_DEFAULT, mjd=ehc.MJD_DEFAULT, timetype='UTC'):
        """A Calibration Table.

           Args:
               ra (float): The source Right Ascension in fractional hours
               dec (float): The source declination in fractional degrees
               rf (float): The observation frequency in Hz
               mjd (int): The integer MJD of the observation
               bw (float): The observation bandwidth in Hz

               datadict (dict):  keys are sites in tarr, entries are gain tables of type
                                 DTCAL. Older welded tables carrying D-term columns are
                                 split into the gain and D-term tables automatically.
               tarr (numpy.recarray): The array of telescope data with datatype DTARR
               dtermdict (dict): keys are sites in tarr, entries are D-term tables of
                                 type DTDTERM, on their own time grid. When given, it
                                 replaces anything split out of ``datadict``.

               source (str): The source name
               mjd (int): The integer MJD of the observation
               timetype (str): How to interpret tstart and tstop; either 'GMST' or 'UTC'

           Returns:
               (Caltable): an Caltable object
        """

        # Set the various parameters
        self.source = str(source)
        self.ra = float(ra)
        self.dec = float(dec)
        self.rf = float(rf)
        self.bw = float(bw)
        self.mjd = int(mjd)

        if timetype not in ['GMST', 'UTC']:
            raise Exception("timetype must by 'GMST' or 'UTC'")
        self.timetype = timetype

        # Dictionary of array indices for site names
        self.tarr = ehc.upgrade_tarr(tarr)
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}

        # Save the data, splitting out D-terms from any older welded tables
        self.dterms = {}
        if not isinstance(datadict, dict):
            raise TypeError("datadict must be a dict of per-site gain tables "
                            f"keyed by site name, got {type(datadict).__name__}")
        self.gains = {}
        for site, d in datadict.items():
            site_gains, site_dterms = ehc.upgrade_caltable(d)
            self.gains[site] = site_gains
            if site_dterms is not None:
                self.dterms[site] = site_dterms

        # An explicit D-term table replaces whatever the split produced
        if dtermdict is not None:
            if not isinstance(dtermdict, dict):
                raise TypeError("dtermdict must be a dict keyed by site name, "
                                f"got {type(dtermdict).__name__}")
            self.dterms = {}
            for site, table in dtermdict.items():
                site_dterms = ehc._normalize_recarray(table)
                if site_dterms is not None:
                    self.dterms[site] = site_dterms

    @property
    def data(self):
        """The gain table, under its legacy name.

        Returns the live ``gains`` dict, so callers that mutate through
        ``ct.data[site]`` or reassign ``ct.data[site] = ...`` still act on the
        table itself.
        """
        return self.gains

    @data.setter
    def data(self, datadict):
        # normalize like the constructor does, but refuse welded tables:
        # splitting here would move D-terms out of the caller's own dict
        if isinstance(datadict, dict):
            normalized = {}
            reshaped = False
            for site, table in datadict.items():
                site_gains, site_dterms = ehc.upgrade_caltable(table)
                if site_dterms is not None:
                    raise TypeError(
                        f"data is the gain table, but the table for {site} "
                        "carries D-term columns. Assign gains to .data and "
                        "leakage to .dterms, or hand the welded table to the "
                        "constructor, which splits it.")
                reshaped |= site_gains is not table
                normalized[site] = site_gains
            # keep the caller's own dict when nothing needed reshaping
            if reshaped:
                datadict = normalized
        self.gains = datadict

    def __setstate__(self, state):
        # upgrade old pickled caltables to the current schema, on a copy of
        # the state dict since a directly passed one belongs to the caller
        state = dict(state)

        if 'tarr' in state:
            state['tarr'] = ehc.upgrade_tarr(state['tarr'])

        # deal with pickles from before the old 'data' attribute was renamed
        # to 'gains' ('data' is a property now and would shadow the entry)
        if 'data' in state:
            legacy = state.pop('data')
            if 'gains' not in state:  # never overwrite new-style state
                if isinstance(legacy, dict):
                    gains = {}
                    dterms = dict(state.get('dterms', {}))
                    for site, d in legacy.items():
                        g, dt = ehc.upgrade_caltable(d)
                        gains[site] = g
                        if dt is not None:
                            dterms[site] = dt
                    state['gains'] = gains
                    state['dterms'] = dterms
                else:
                    # old __init__ stored non-dict datadicts verbatim
                    state['gains'] = legacy
        state.setdefault('dterms', {})
        self.__dict__.update(state)

    def copy(self):
        """Copy the observation object.

           Args:

           Returns:
               (Caltable): a deep copy of the Caltable object.
        """
        return copy.deepcopy(self)

    def plot_dterms(self, sites='all', label=None, legend=True, clist=ehc.SCOLORS,
                    rangex=False, rangey=False, markersize=2 * ehc.MARKERSIZE,
                    show=True, grid=True, export_pdf=""):
        """Make a plot of the D-terms.

           Plots the leakage recorded in the array table (tarr), one point per
           site. The caltable's own time-dependent D-term table (self.dterms)
           is not read here.

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
        if self.dterms:
            warnings.warn(
                "This plot shows the array table's D-terms. This cal table also "
                f"carries its own solved leakage for {', '.join(sorted(self.dterms))}, "
                "in self.dterms, which is not what is drawn here.",
                MixedPolConventionWarning, stacklevel=2)

        # sites: a site can carry leakage without gains, so take both dicts
        all_sites = list(self.gains.keys())
        all_sites += [site for site in self.dterms if site not in self.gains]

        if (isinstance(sites,str) and sites.lower() == 'all'):
            sites = all_sites

        if isinstance(sites,str):
            sites = [sites]

        if len(sites)==0:
            sites = all_sites

        keys = [self.tkey[site] for site in sites]

        axes = plot_tarr_dterms(self.tarr, keys=keys, label=label, legend=legend, clist=clist,
                                rangex=rangex, rangey=rangey, markersize=markersize,
                                show=show, grid=grid, export_pdf=export_pdf)

        return axes

    def plot_gains(self, sites, gain_type='amp', pol='R', label=None,
                   ang_unit='deg', timetype=False, yscale='log', legend=True,
                   clist=ehc.SCOLORS, rangex=False, rangey=False, markersize=[ehc.MARKERSIZE],
                   show=True, grid=False, axislabels=True, axis=False, export_pdf=""):
        """Plot gains on multiple sites vs time.
           Args:
               sites (list): a list of site names for which to plot gains. Empty list is all sites.
               gain_type (str): 'amp' or 'phase'
               pol str(str): 'R' or 'L'
               ang_unit (str): phase unit 'deg' or 'rad'
               timetype (str): 'GMST' or 'UTC'
               yscale (str): 'log' or 'lin',
               clist (list): list of colors for the plot
               label (str): base label for legend

               rangex (list): [xmin, xmax] x-axis (time) limits
               rangey (list): [ymin, ymax] y-axis (gain) limits

               legend (bool): Plot legend if True
               grid (bool): Plot gridlines if True
               axislabels (bool): Show axis labels if True
               show (bool): Display the plot if true
               axis (matplotlib.axes.Axes): add plot to this axis
               markersize (int): size of plot markers
               export_pdf (str): path to pdf file to save figure

           Returns:
               (matplotlib.axes.Axes): Axes object with the plot
        """

        colors = iter(clist)

        if timetype is False:
            timetype = self.timetype
        if timetype not in ['GMST', 'UTC', 'utc', 'gmst']:
            raise Exception("timetype should be 'GMST' or 'UTC'!")
        if gain_type not in ['amp', 'phase']:
            raise Exception("gain_type must be 'amp' or 'phase'  ")
        if pol not in ['R', 'L', 'both']:
            raise Exception("pol must be 'R' or 'L'")

        if ang_unit == 'deg':
            angle = ehc.DEGREE
        else:
            angle = 1.0

        # axis
        if axis:
            x = axis
        else:
            fig = plt.figure()
            x = fig.add_subplot(1, 1, 1)

        # sites
        if (isinstance(sites,str) and sites.lower() == 'all'):
            sites = sorted(list(self.gains.keys()))

        if isinstance(sites,str):
            sites = [sites]

        if len(sites)==0:
            sites = sorted(list(self.gains.keys()))

        if len(markersize) == 1:
            markersize = markersize * np.ones(len(sites))

        # plot gain on each site
        tmins = tmaxes = gmins = gmaxes = []
        for s in range(len(sites)):
            site = sites[s]
            times = self.gains[site]['time']
            if timetype in ['UTC', 'utc'] and self.timetype == 'GMST':
                times = obsh.gmst_to_utc(times, self.mjd)
            elif timetype in ['GMST', 'gmst'] and self.timetype == 'UTC':
                times = obsh.utc_to_gmst(times, self.mjd)
            if pol == 'R':
                gains = self.gains[site]['rscale']
            elif pol == 'L':
                gains = self.gains[site]['lscale']

            if gain_type == 'amp':
                gains = np.abs(gains)
                ylabel = r'$|G|$'

            if gain_type == 'phase':
                gains = np.angle(gains) / angle
                if ang_unit == 'deg':
                    ylabel = r'arg($|G|$) ($^\circ$)'
                else:
                    ylabel = r'arg($|G|$) (radian)'

            tmins.append(np.min(times))
            tmaxes.append(np.max(times))
            gmins.append(np.min(gains))
            gmaxes.append(np.max(gains))

            # Plot the data
            if label is None:
                bllabel = str(site)
            else:
                bllabel = label + ' ' + str(site)
            plt.plot(times, gains, color=next(colors), marker='o', markersize=markersize[s],
                     label=bllabel, linestyle='none')

        if not rangex:
            rangex = [np.min(tmins) - 0.2 * np.abs(np.min(tmins)),
                      np.max(tmaxes) + 0.2 * np.abs(np.max(tmaxes))]
            if np.any(np.isnan(np.array(rangex))):
                print("Warning: NaN in data x range: specifying rangex to default")
                rangex = [0, 24]
        if not rangey:
            rangey = [np.min(gmins) - 0.2 * np.abs(np.min(gmins)),
                      np.max(gmaxes) + 0.2 * np.abs(np.max(gmaxes))]
            if np.any(np.isnan(np.array(rangey))):
                print("Warning: NaN in data x range: specifying rangey to default")
                rangey = [1.e-2, 1.e2]

        plt.plot(np.linspace(rangex[0], rangex[1], 5), np.ones(5), 'k--')
        x.set_xlim(rangex)
        x.set_ylim(rangey)

        # labels
        if axislabels:
            x.set_xlabel(self.timetype + ' (hr)')
            x.set_ylabel(ylabel)
            plt.title(f'Caltable gains for {self.source} on day {self.mjd}')

        if legend:
            plt.legend()

        if yscale == 'log':
            x.set_yscale('log')
        if grid:
            x.grid()
        if export_pdf != "" and not axis:
            fig.savefig(export_pdf, bbox_inches='tight')
        if show:
            #plt.show(block=False)
            ehc.show_noblock()

        return x

    def enforce_positive(self, method='median', min_gain=0.9, sites=[], verbose=True):
        """Enforce that caltable gains are not low
           (e.g., that sites are not significantly more sensitive than estimated).
           By rescaling the entire gain curve to enforce a specified minimum site gain.

           Args:
               caltab (Caltable): Input Caltable with station gains
               method (str): 'median', 'mean', or 'min'
               min_gain (float): Site gains above this value are not modified.
               sites (list): List of sites to check and adjust. For sites=[], all sites are fixed.
               verbose (bool): If True, print corrections.

           Returns:
               (Caltable): Axes object with the plot
        """

        if len(sites) == 0:
            sites = self.gains.keys()

        caltab_pos = self.copy()
        for site in self.gains.keys():
            if site not in sites:
                continue
            if len(self.gains[site]['rscale']) == 0:
                continue

            if method == 'min':
                sitemin = np.min([np.abs(self.gains[site]['rscale']),
                                  np.abs(self.gains[site]['lscale'])])
            elif method == 'mean':
                sitemin = np.mean([np.abs(self.gains[site]['rscale']),
                                   np.abs(self.gains[site]['lscale'])])
            elif method == 'median':
                sitemin = np.median([np.abs(self.gains[site]['rscale']),
                                     np.abs(self.gains[site]['lscale'])])
            else:
                print('Method ' + method + ' not recognized!')
                return caltab_pos

            if sitemin < min_gain:
                if verbose:
                    print(method + ' gain for ' + site + ' is ' + str(sitemin) + '. Rescaling.')
                caltab_pos.gains[site]['rscale'] /= sitemin
                caltab_pos.gains[site]['lscale'] /= sitemin
            else:
                if verbose:
                    print(method + ' gain for ' + site + ' is ' + str(sitemin) + '. Not adjusting.')

        return caltab_pos

    # TODO default extrapolation?
    def pad_scans(self, maxdiff=60, padtype='median'):
        """Pad data points around scans.

           Args:
               maxdiff (float): "scan" separation length (seconds)
               padtype (str): padding type, 'endval' or 'median'

           Returns:
               (Caltable):  a padded caltable object
        """

        outdict = {}
        scopes = list(self.gains.keys())
        for scope in scopes:
            if np.any(self.gains[scope] is None) or len(self.gains[scope]) == 0:
                continue

            caldata = copy.deepcopy(self.gains[scope])

            # Gather data into "scans"
            # TODO we could use a scan table for this as well!
            gathered_data = []
            scandata = [caldata[0]]
            for i in range(1, len(caldata)):
                if (caldata[i]['time'] - caldata[i - 1]['time']) * 3600 > maxdiff:
                    scandata = np.array(scandata, dtype=caldata.dtype)
                    gathered_data.append(scandata)
                    scandata = [caldata[i]]
                else:
                    scandata.append(caldata[i])

            # This adds the last scan
            scandata = np.array(scandata, dtype=caldata.dtype)
            gathered_data.append(scandata)

            # Compute padding values and pad scans
            for i in range(len(gathered_data)):
                gg = gathered_data[i]

                # Generic feed names throughout, so a linear-feed table pads
                # as itself instead of being recast as circular.
                timepre = gg['time'][0] - maxdiff / 2. / 3600.
                timepost = gg['time'][-1] + maxdiff / 2. / 3600.

                if padtype == 'median':  # pad with median scan value
                    pre1 = post1 = np.median(gg['p1scale'])
                    pre2 = post2 = np.median(gg['p2scale'])
                elif padtype == 'endval':  # pad with endpoints
                    pre1 = gg['p1scale'][0]
                    post1 = gg['p1scale'][-1]
                    pre2 = gg['p2scale'][0]
                    post2 = gg['p2scale'][-1]
                else:  # pad with ones
                    pre1 = post1 = 1.
                    pre2 = post2 = 1.

                valspre = np.array([(timepre, pre1, pre2)], dtype=caldata.dtype)
                valspost = np.array([(timepost, post1, post2)], dtype=caldata.dtype)

                gg = np.insert(gg, 0, valspre)
                gg = np.append(gg, valspost)

                # output data table
                if i == 0:
                    caldata_out = gg
                else:
                    caldata_out = np.append(caldata_out, gg)

            try:
                caldata_out  # TODO: refractor to avoid using exception
            except NameError:
                print("No gathered_data")
            else:
                outdict[scope] = caldata_out

        # padding is a gain-table operation; leakage rides along untouched
        return Caltable(self.ra, self.dec, self.rf, self.bw, outdict, self.tarr,
                        source=self.source, mjd=self.mjd, timetype=self.timetype,
                        dtermdict=copy.deepcopy(self.dterms))

    def applycal(self, obs, interp='linear', extrapolate=None,
                 force_singlepol=False):
        """Apply the calibration table to an observation.

           Args:
               obs (Obsdata): The observation  with data to be calibrated
               interp (str): Interpolation method ('linear','nearest','cubic')
               extrapolate (bool): If True, points outside interpolation range will be extrapolated.
               force_singlepol (str): If 'L' or 'R', will set opposite polarization gains
                                      equal to chosen polarization

           Returns:
               (Obsdata): the calibrated Obsdata object
        """
        if not (self.tarr == obs.tarr).all():
            raise Exception("The telescope array in the Caltable is not the same as in the Obsdata")

        if self.dterms:
            warnings.warn(
                "This cal table carries time-dependent D-terms for "
                f"{', '.join(sorted(self.dterms))}, but applycal corrects gains only: "
                "the leakage is left in the data.",
                MixedPolConventionWarning, stacklevel=2)

        if extrapolate is True:  # extrapolate can be a tuple or numpy array
            fill_value = "extrapolate"
        else:
            fill_value = extrapolate

        orig_polrep = obs.polrep
        obs = obs.switch_polrep('circ')

        rinterp = {}
        linterp = {}
        skipsites = []
        for s in range(0, len(self.tarr)):
            site = self.tarr[s]['site']

            try:
                self.gains[site]
            except KeyError:
                skipsites.append(site)
                print(f"No Calibration  Data for {site} !")
                continue

            time_mjd = self.gains[site]['time'] / 24.0 + self.mjd
            rinterp[site] = relaxed_interp1d(time_mjd, self.gains[site]['rscale'],
                                             kind=interp, fill_value=fill_value, bounds_error=False)
            linterp[site] = relaxed_interp1d(time_mjd, self.gains[site]['lscale'],
                                             kind=interp, fill_value=fill_value, bounds_error=False)

        bllist = obs.bllist()
        datatable = []
        for bl_obs in bllist:
            t1 = bl_obs['t1'][0]
            t2 = bl_obs['t2'][0]
            time_mjd = bl_obs['time'] / 24.0 + obs.mjd

            if t1 in skipsites:
                rscale1 = lscale1 = np.array(1.)
            else:
                rscale1 = rinterp[t1](time_mjd)
                lscale1 = linterp[t1](time_mjd)
            if t2 in skipsites:
                rscale2 = lscale2 = np.array(1.)
            else:
                rscale2 = rinterp[t2](time_mjd)
                lscale2 = linterp[t2](time_mjd)

            if force_singlepol == 'R':
                lscale1 = rscale1
                lscale2 = rscale2

            if force_singlepol == 'L':
                rscale1 = lscale1
                rscale2 = lscale2

            rrscale = rscale1 * rscale2.conj()
            llscale = lscale1 * lscale2.conj()
            rlscale = rscale1 * lscale2.conj()
            lrscale = lscale1 * rscale2.conj()

            bl_obs['rrvis'] = (bl_obs['rrvis']) * rrscale
            bl_obs['llvis'] = (bl_obs['llvis']) * llscale
            bl_obs['rlvis'] = (bl_obs['rlvis']) * rlscale
            bl_obs['lrvis'] = (bl_obs['lrvis']) * lrscale

            bl_obs['rrsigma'] = bl_obs['rrsigma'] * np.abs(rrscale)
            bl_obs['llsigma'] = bl_obs['llsigma'] * np.abs(llscale)
            bl_obs['rlsigma'] = bl_obs['rlsigma'] * np.abs(rlscale)
            bl_obs['lrsigma'] = bl_obs['lrsigma'] * np.abs(lrscale)

            datatable.append(bl_obs)

        if len(datatable):
            datatable = np.hstack(datatable)

        calobs = ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw,
                                       np.array(datatable), obs.tarr,
                                       polrep=obs.polrep, scantable=obs.scans, source=obs.source,
                                       mjd=obs.mjd, ampcal=obs.ampcal, phasecal=obs.phasecal,
                                       opacitycal=obs.opacitycal, dcal=obs.dcal,
                                       frcal=obs.frcal, timetype=obs.timetype)
        calobs = calobs.switch_polrep(orig_polrep)

        return calobs

    def merge(self, caltablelist, interp='linear', extrapolate=1):
        """Merge the calibration table with a list of other calibration tables

           Args:
               caltablelist (list): The list of caltables to be merged
               interp (str): Interpolation method ('linear','nearest','cubic')
               extrapolate (bool): If True, points outside interpolation range will be extrapolated.

           Returns:
               (Caltable): the merged Caltable object
        """

        if extrapolate is True:  # extrapolate can be a tuple or numpy array
            fill_value = "extrapolate"
        else:
            fill_value = extrapolate

        if not hasattr(caltablelist, '__iter__'):
            caltablelist = [caltablelist]

        tarr1 = self.tarr.copy()
        tkey1 = self.tkey.copy()
        # copy the arrays, not just the dict, so the merged table never
        # writes back into this one
        data1 = {site: (table if table is None else table.copy())
                 for site, table in self.gains.items()}
        dterms1 = copy.deepcopy(self.dterms)
        for caltable in caltablelist:

            # TODO check metadata!

            # TODO CHECK ARE THEY ALL REFERENCED TO SAME MJD???
            tarr2 = caltable.tarr.copy()
            tkey2 = caltable.tkey.copy()

            # merging time-dependent D-term tables.
            # If both caltables carry D-terms for a site, raise for now:
            # composing D-terms requires full Jones treatment.
            for site, dterm_table in caltable.dterms.items():
                if site in dterms1:
                    raise NotImplementedError(
                        f"merge: both caltables carry D-terms for site {site}. "
                        "Composing two leakage solutions is deferred; it is a "
                        "Jones product, not a gain-style multiply.")
                # if D-terms are only in the merged-in caltable, we can add them.
                # Be sure to add the tarr row; it may be missed in the gain
                # merge below.
                if site not in tkey1 and site in tkey2:
                    tarr1 = np.append(tarr1, tarr2[tkey2[site]])
                    tkey1 = {tarr1[i]['site']: i for i in range(len(tarr1))}
                dterms1[site] = copy.deepcopy(dterm_table)
            data2 = caltable.gains.copy()
            sites2 = list(data2.keys())
            sites1 = list(data1.keys())
            for site in sites2:
                if site in sites1:  # if site in both tables

                    # merge the data by interpolating
                    time1 = data1[site]['time']
                    time2 = data2[site]['time']

                    rinterp1 = relaxed_interp1d(time1, data1[site]['rscale'],
                                                kind=interp, fill_value=fill_value,
                                                bounds_error=False)
                    linterp1 = relaxed_interp1d(time1, data1[site]['lscale'],
                                                kind=interp, fill_value=fill_value,
                                                bounds_error=False)
                    rinterp2 = relaxed_interp1d(time2, data2[site]['rscale'],
                                                kind=interp, fill_value=fill_value,
                                                bounds_error=False)
                    linterp2 = relaxed_interp1d(time2, data2[site]['lscale'],
                                                kind=interp, fill_value=fill_value,
                                                bounds_error=False)

                    times_merge = np.unique(np.hstack((time1, time2)))

                    rscale_merge = rinterp1(times_merge) * rinterp2(times_merge)
                    lscale_merge = linterp1(times_merge) * linterp2(times_merge)

                    # put the merged data back in data1
                    # TODO can we do this faster?
                    datatable = []
                    for i in range(len(times_merge)):
                        datatable.append(
                            np.array((times_merge[i], rscale_merge[i], lscale_merge[i]),
                                     dtype=ehc.DTCAL))
                    data1[site] = np.array(datatable)

                # sites not in both caltables
                else:
                    if site not in tkey1.keys():
                        tarr1 = np.append(tarr1, tarr2[tkey2[site]])
                    # copy (None-safe) so the merged table does not share
                    # the input's array
                    other = data2[site]
                    data1[site] = other if other is None else other.copy()

            # update tkeys every time
            tkey1 = {tarr1[i]['site']: i for i in range(len(tarr1))}

        new_caltable = Caltable(self.ra, self.dec, self.rf, self.bw, data1, tarr1,
                                source=self.source, mjd=self.mjd, timetype=self.timetype,
                                dtermdict=dterms1)

        return new_caltable

    def save_txt(self, obs, datadir='.', sqrt_gains=False, overwrite=False):
        """Saves a Caltable object to text files in the given directory

           Gains and D-terms go to separate per-site files; see save_caltable
           for the layout.

           Args:
               obs (Obsdata): The observation object associated with the Caltable
               datadir (str): directory to save caltable in
               sqrt_gains (bool): If True, we square gains before saving.
               overwrite (bool): If True, allow existing caltable files in datadir
                                 to be overwritten or deleted

           Returns:
        """

        return save_caltable(self, obs, datadir=datadir, sqrt_gains=sqrt_gains,
                             overwrite=overwrite)

    def scan_avg(self, obs, incoherent=True):
        """average the gains across scans.

           Args:
               obs (ehtim.Obsdata) : input observation
               incoherent (bool) : True to average gain amps, False to average amps+phase

           Returns:
               (Caltable): the averaged Caltable object
        """
        sites = list(self.gains.keys())
        ntele = len(sites)

        datatables = {}

        # iterate over each site
        for s in range(0, ntele):
            site = sites[s]

            # make a list of times that is the same value for all points in the same scan
            times = self.gains[site]['time']
            times_stable = times.copy()
            obs.add_scans()
            scans = obs.scans
            for j in range(len(times_stable)):
                for scan in scans:
                    if scan[0] <= times_stable[j] and scan[1] >= times_stable[j]:
                        times_stable[j] = scan[0]
                        break

            datatable = []
            for scan in scans:
                gains_l = self.gains[site]['lscale']
                gains_r = self.gains[site]['rscale']

                # if incoherent average then average the magnitude of gains
                if incoherent:
                    gains_l = np.abs(gains_l)
                    gains_r = np.abs(gains_r)

                # average the gains
                gains_l_avg = np.mean(gains_l[np.array(times_stable == scan[0])])
                gains_r_avg = np.mean(gains_r[np.array(times_stable == scan[0])])

                # add them to a new datatable
                datatable.append(np.array((scan[0], gains_r_avg, gains_l_avg), dtype=ehc.DTCAL))

            datatables[site] = np.array(datatable)

        if len(datatables) > 0:
            # the D-term table is carried over as-is
            # TODO: we may want to eventually have a function to average D-terms
            caltable = Caltable(obs.ra, obs.dec, obs.rf,
                                obs.bw, datatables, obs.tarr, source=obs.source,
                                mjd=obs.mjd, timetype=obs.timetype,
                                dtermdict=copy.deepcopy(self.dterms))
        else:
            caltable = False

        return caltable

    def invert_gains(self):
        """Invert the gain table in place, leaving any D-term table untouched.

           Returns:
               (Caltable): self, with every gain replaced by its reciprocal
        """

        sites = self.gains.keys()

        for site in sites:
            self.gains[site]['rscale'] = 1 / self.gains[site]['rscale']
            self.gains[site]['lscale'] = 1 / self.gains[site]['lscale']

        return self


def load_caltable(obs, datadir, sqrt_gains=False):
    """Load apriori Caltable object from text files in the given directory

       Reads the per-site gain files, and the per-site D-term files beside them
       when present -- a directory written before D-terms had an on-disk slot
       simply loads with no leakage. Gain files are what makes a directory a
       cal table: with none, this returns False even if D-term files are there.

       Args:
           obs (Obsdata): The observation object associated with the Caltable
           datadir (str): directory to save caltable in
           sqrt_gains (bool): If True, we take the sqrt of table gains before loading.

       Returns:
           (Caltable): a caltable object
    """

    tarr = obs.tarr
    array_filename = datadir + '/array.txt'
    if os.path.exists(array_filename):
        tarr = ehtim.io.load.load_array_txt(array_filename).tarr

    datatables = {}
    for s in range(0, len(tarr)):

        site = tarr[s]['site']
        filename = os.path.join(datadir, obs.source + '_' + site + '.txt')
        try:
            # ndmin=2 so a one-row file still iterates as rows, not characters
            data = np.loadtxt(filename, dtype=bytes, ndmin=2).astype(str)
        except OSError:
            try:
                filename = datadir + site + '.txt'
                data = np.loadtxt(filename, dtype=bytes, ndmin=2).astype(str)
            except OSError:
                continue


        datatable = []

        for row in data:

            time = (float(row[0]) - obs.mjd) * 24.0  # time is given in mjd

            if len(row) == 3:
                rscale = float(row[1])
                lscale = float(row[2])
            elif len(row) == 5:
                rscale = float(row[1]) + 1j * float(row[2])
                lscale = float(row[3]) + 1j * float(row[4])
            else:
                raise Exception("cannot load caltable -- format unknown!")
            if sqrt_gains:
                rscale = rscale**.5
                lscale = lscale**.5
            datatable.append(np.array((time, rscale, lscale), dtype=ehc.DTCAL))

        datatables[site] = np.array(datatable)

    # D-terms, if this directory has them. Written by save_caltable next to the
    # gain files; older directories simply have none.
    dterm_tables = {}
    for s in range(0, len(tarr)):
        site = tarr[s]['site']
        filename = os.path.join(datadir, obs.source + '_' + site + DTERM_FILE_SUFFIX)
        try:
            # skip the version line explicitly: letting loadtxt treat it as a
            # comment makes numpy warn about the skipped line on every load
            with open(filename) as dterm_file:
                skiprows = 1 if dterm_file.readline().lstrip().startswith('#') else 0
            data = np.loadtxt(filename, dtype=bytes, ndmin=2,
                              skiprows=skiprows, comments=None).astype(str)
        except OSError:
            continue

        dterm_table = []
        for row in data:
            if len(row) != 5:
                raise Exception("cannot load caltable D-terms -- format unknown!")
            time = (float(row[0]) - obs.mjd) * 24.0  # time is given in mjd
            d_p1 = float(row[1]) + 1j * float(row[2])
            d_p2 = float(row[3]) + 1j * float(row[4])
            dterm_table.append(np.array((time, d_p1, d_p2), dtype=ehc.DTDTERM))

        dterm_tables[site] = np.array(dterm_table)

    if len(datatables) > 0:
        caltable = Caltable(obs.ra, obs.dec, obs.rf, obs.bw, datatables, tarr,
                            source=obs.source, mjd=obs.mjd, timetype=obs.timetype,
                            dtermdict=dterm_tables)
    else:
        print(f"COULD NOT FIND CALTABLE IN DIRECTORY {datadir}")
        caltable = False
    return caltable


def _caltable_filenames(caltable, datadir):
    """Every file save_caltable writes to or removes in datadir."""
    src = caltable.source
    names = [datadir + '/array.txt']
    for site_info in caltable.tarr:
        site = site_info['site']
        names.append(datadir + '/' + src + '_' + site + '.txt')
        names.append(datadir + '/' + src + '_' + site + DTERM_FILE_SUFFIX)
    return names


def save_caltable(caltable, obs, datadir='.', sqrt_gains=False, overwrite=False):
    """Saves a Caltable object to text files in the given directory

       Writes array.txt, one '<source>_<site>.txt' per site of gains, and one
       '<source>_<site>_dterms.txt' per site that carries time-dependent
       leakage.

       Args:
           obs (Obsdata): The observation object associated with the Caltable
           datadir (str): directory to save caltable in
           sqrt_gains (bool): If True, we square gains before saving.
           overwrite (bool): If True, allow existing caltable files in datadir
                             to be overwritten or deleted

       Returns:
    """

    if not os.path.exists(datadir):
        os.makedirs(datadir)

    # don't clobber an existing caltable unless asked
    existing = [f for f in _caltable_filenames(caltable, datadir) if os.path.exists(f)]
    if existing and not overwrite:
        raise Exception(f"{len(existing)} caltable file(s) in {datadir} would be "
                        "overwritten or deleted; pass overwrite=True to allow it")

    # The table's tarr, not the observation's: load_caltable rebuilds tarr from
    # this file, so a site only the table knows about has to appear here.
    ehtim.io.save.save_array_txt(caltable.tarr, datadir + '/array.txt')

    datatables = caltable.gains
    src = caltable.source
    for site_info in caltable.tarr:
        site = site_info['site']
        filename = datadir + '/' + src + '_' + site + '.txt'

        if len(datatables.get(site, [])) == 0:
            # a stale file would be read back as this table's gains
            if os.path.exists(filename):
                os.remove(filename)
            continue

        outfile = open(filename, 'w')
        site_data = datatables[site]
        for entry in site_data:
            time = entry['time'] / 24.0 + obs.mjd

            if sqrt_gains:
                rscale = np.square(entry['rscale'])
                lscale = np.square(entry['lscale'])
            else:
                rscale = entry['rscale']
                lscale = entry['lscale']

            rreal = float(np.real(rscale))
            rimag = float(np.imag(rscale))
            lreal = float(np.real(lscale))
            limag = float(np.imag(lscale))
            outline = (str(float(time)) + ' ' +
                       str(float(rreal)) + ' ' + str(float(rimag)) + ' ' +
                       str(float(lreal)) + ' ' + str(float(limag)) + '\n')
            outfile.write(outline)
        outfile.close()

    # Time-dependent D-terms go in their own per-site files. Loop over tarr, not
    # the gain sites: a site can carry leakage with no gains.
    for site_info in caltable.tarr:
        site = site_info['site']
        filename = datadir + '/' + src + '_' + site + DTERM_FILE_SUFFIX

        if len(caltable.dterms.get(site, [])) == 0:
            # a stale file would be read back as this table's leakage
            if os.path.exists(filename):
                os.remove(filename)
            continue

        with open(filename, 'w') as outfile:
            outfile.write(DTERM_FILE_HEADER + '\n')
            for entry in caltable.dterms[site]:
                time = entry['time'] / 24.0 + obs.mjd
                outline = (str(float(time)) + ' ' +
                           str(float(np.real(entry['d_p1']))) + ' ' +
                           str(float(np.imag(entry['d_p1']))) + ' ' +
                           str(float(np.real(entry['d_p2']))) + ' ' +
                           str(float(np.imag(entry['d_p2']))) + '\n')
                outfile.write(outline)

    return


def make_caltable(obs, gains, sites, times):
    """Create a Caltable object for an observation
       Args:
           obs (Obsdata): The observation object associated with the Caltable
           gains (list): list of gains (?? format ??)
           sites (list): list of sites
           times (list): list of times

       Returns:
           (Caltable): a caltable object
    """
    ntele = len(sites)
    ntimes = len(times)

    datatables = {}
    for s in range(0, ntele):
        datatable = []
        for t in range(0, ntimes):
            gain = gains[s * ntimes + t]
            datatable.append(np.array((times[t], gain, gain), dtype=ehc.DTCAL))
        datatables[sites[s]] = np.array(datatable)
    if len(datatables) > 0:
        caltable = Caltable(obs.ra, obs.dec, obs.rf,
                            obs.bw, datatables, obs.tarr, source=obs.source,
                            mjd=obs.mjd, timetype=obs.timetype)
    else:
        caltable = False

    return caltable


def relaxed_interp1d(x, y, **kwargs):
    # TODO: the single-row fallback below invents a flat segment of half-width
    # 0.5 in whatever units x carries -- half an hour for merge, which
    # interpolates in raw hours, half a day for applycal, which interpolates in
    # MJD. Harmless while it only ever pads a stray one-row gain table, but a
    # time-constant D-term table is one row by construction, so the width stops
    # being incidental as soon as leakage is interpolated. Make it explicit then.
    try:
        len(x)
    except TypeError:
        x = np.asarray([x])
        y = np.asarray([y])  # allows to run on a single float number
    if len(x) == 1:
        x = np.array([-0.5, 0.5]) + x[0]
        y = np.array([1.0, 1.0]) * y[0]
    return scipy.interpolate.interp1d(x, y, **kwargs)


def plot_tarr_dterms(tarr, keys=None, label=None, legend=True, clist=ehc.SCOLORS,
                     rangex=False, rangey=False, markersize=2 * ehc.MARKERSIZE,
                     show=True, grid=True, export_pdf="", auto_order=True):

    if auto_order:
        # Ensure that the plot will put the stations in alphabetical order
        keys = np.argsort(tarr['site'])  # range(len(tarr))
    else:
        keys = range(len(tarr))

    colors = iter(clist)

    if export_pdf != "":
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(16, 8))
    else:
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)

    for key in keys:

        # get the label
        site = str(tarr[key]['site'])
        if label is None:
            bllabel = str(site)
        else:
            bllabel = label + ' ' + str(site)
        color = next(colors)

        axes[0].plot(np.real(tarr[key]['dr']), np.imag(tarr[key]['dr']),
                     color=color, marker='o', markersize=markersize,
                     label=bllabel, linestyle='none')
        axes[0].set_title("Right D-terms")
        axes[0].set_xlabel("Real")
        axes[0].set_ylabel("Imaginary")

        axes[1].plot(np.real(tarr[key]['dl']), np.imag(tarr[key]['dl']),
                     color=color, marker='o', markersize=markersize,
                     label=bllabel, linestyle='none')
        axes[1].set_title("Left D-terms")
        axes[1].set_xlabel("Real")
        axes[1].set_ylabel("Imaginary")

    axes[0].axhline(y=0, color='k')
    axes[0].axvline(x=0, color='k')
    axes[1].axhline(y=0, color='k')
    axes[1].axvline(x=0, color='k')

    if grid:
        axes[0].grid()
        axes[1].grid()

    if rangex:
        axes[0].set_xlim(rangex)
        axes[1].set_xlim(rangex)

    if rangey:
        axes[0].set_ylim(rangey)
        axes[1].set_ylim(rangey)

    if legend:
        axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if export_pdf != "":
        fig.savefig(export_pdf, bbox_inches='tight')

    return axes


def plot_compare_gains(caltab1, caltab2, obs, sites='all', pol='R', gain_type='amp', ang_unit='deg',
                       scan_avg=True, site_name_dict=None, fontsize=13, legend_fontsize=13,
                       yscale='log', legend=True, clist=ehc.SCOLORS,
                       rangex=False, rangey=False, scalefac=[0.9, 1.1],
                       markersize=[2 * ehc.MARKERSIZE], show=True, grid=False,
                       axislabels=True, remove_ticks=False, axis=False,
                       export_pdf=""):

    colors = iter(clist)

    if ang_unit == 'deg':
        angle = ehc.DEGREE
    else:
        angle = 1.0

    # axis
    if axis:
        x = axis
    else:
        fig = plt.figure()
        x = fig.add_subplot(1, 1, 1)

    if scan_avg:
        caltab1 = caltab1.scan_avg(obs, incoherent=True)
        caltab2 = caltab2.scan_avg(obs, incoherent=True)

    # sites
    if (isinstance(sites,str) and sites.lower() == 'all'):
        sites = list(set(caltab1.gains.keys()).intersection(caltab2.gains.keys()))

    if isinstance(sites,str):
        sites = [sites]

    if len(sites)==0:
        sites = list(set(caltab1.gains.keys()).intersection(caltab2.gains.keys()))

    if site_name_dict is None:
        site_name_dict = {}
        for site in sites:
            site_name_dict[site] = site

    if len(markersize) == 1:
        markersize = markersize * np.ones(len(sites))

    maxgain = 0.0
    mingain = 10000

    for s in range(len(sites)):

        site = sites[s]
        if pol == 'R':
            gains1 = caltab1.gains[site]['rscale']
            gains2 = caltab2.gains[site]['rscale']
        elif pol == 'L':
            gains1 = caltab1.gains[site]['lscale']
            gains2 = caltab2.gains[site]['lscale']

        if gain_type == 'amp':
            gains1 = np.abs(gains1)
            gains2 = np.abs(gains2)
            ylabel = 'Amplitudes'  # r'$|G|$'

        if gain_type == 'phase':
            gains1 = np.angle(gains1) / angle
            gains2 = np.angle(gains2) / angle
            if ang_unit == 'deg':
                ylabel = r'arg($|G|$) ($^\circ$)'
            else:
                ylabel = 'Phases (radian)'  # r'arg($|G|$) (radian)'

        # print a line
        maxgain = np.nanmax([maxgain, np.nanmax(gains1), np.nanmax(gains2)])
        mingain = np.nanmin([mingain, np.nanmin(gains1), np.nanmin(gains2)])

        # mark the gains on the plot
        plt.plot(gains1, gains2, marker='.', linestyle='None', color=next(
            colors), markersize=markersize[s], label=site_name_dict[site])

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.axes().set_aspect('equal')

    if rangex:
        x.set_xlim(rangex)
    else:
        x.set_xlim([mingain * scalefac[0], maxgain * scalefac[1]])
    if rangey:
        x.set_ylim(rangey)
    else:
        x.set_ylim([mingain * scalefac[0], maxgain * scalefac[1]])

    plt.plot([mingain * scalefac[0], maxgain * scalefac[1]],
             [mingain * scalefac[0], maxgain * scalefac[1]], 'grey', linewidth=1)

    # labels
    if axislabels:
        x.set_xlabel('Ground Truth Gain ' + ylabel, fontsize=fontsize)
        x.set_ylabel('Recovered Gain ' + ylabel, fontsize=fontsize)
    else:
        x.tick_params(axis="y", direction="in", pad=-30)
        x.tick_params(axis="x", direction="in", pad=-18)

    if remove_ticks:
        plt.setp(x.get_xticklabels(), visible=False)
        plt.setp(x.get_yticklabels(), visible=False)

    if legend:
        plt.legend(frameon=False, fontsize=legend_fontsize)

    if yscale == 'log':
        x.set_yscale('log')
        x.set_xscale('log')
    if grid:
        x.grid()
    if export_pdf != "" and not axis:
        fig.savefig(export_pdf, bbox_inches='tight')
    if show:
        #plt.show(block=False)
        ehc.show_noblock()

    return x
