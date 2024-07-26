# obsdata.py
# a interferometric observation class
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
from builtins import object

import string
import copy
import numpy as np
import numpy.lib.recfunctions as rec
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.spatial as spatial
import itertools as it
import sys

try:
    import pandas as pd
except ImportError:
    print("Warning: pandas not installed!")
    print("Please install pandas to use statistics package!")


import ehtim.image
import ehtim.io.save
import ehtim.io.load
import ehtim.const_def as ehc
import ehtim.observing.obs_helpers as obsh
import ehtim.statistics.dataframes as ehdf

import warnings
warnings.filterwarnings("ignore",
                        message="Casting complex values to real discards the imaginary part")

RAPOS = 0
DECPOS = 1
RFPOS = 2
BWPOS = 3
DATPOS = 4
TARRPOS = 5

##################################################################################################
# Obsdata object
##################################################################################################


class Obsdata(object):

    """A polarimetric VLBI observation of visibility amplitudes and phases (in Jy).

       Attributes:
           source (str): The source name
           ra (float): The source Right Ascension in fractional hours
           dec (float): The source declination in fractional degrees
           mjd (int): The integer MJD of the observation
           tstart (float): The start time of the observation in hours
           tstop (float): The end time of the observation in hours
           rf (float): The observation frequency in Hz
           bw (float): The observation bandwidth in Hz
           timetype (str): How to interpret tstart and tstop; either 'GMST' or 'UTC'
           polrep (str): polarization representation, either 'stokes' or 'circ'

           tarr (numpy.recarray): The array of telescope data with datatype DTARR
           tkey (dict): A dictionary of rows in the tarr for each site name
           data (numpy.recarray): the basic data with datatype DTPOL_STOKES or DTPOL_CIRC
           scantable (numpy.recarray): The array of scan information

           ampcal (bool): True if amplitudes calibrated
           phasecal (bool): True if phases calibrated
           opacitycal (bool): True if time-dependent opacities correctly accounted for in sigmas
           frcal (bool): True if feed rotation calibrated out of visibilities
           dcal (bool): True if D terms calibrated out of visibilities

           amp (numpy.recarray): An array of (averaged) visibility amplitudes
           bispec (numpy.recarray): An array of (averaged) bispectra
           cphase (numpy.recarray): An array of (averaged) closure phases
           cphase_diag (numpy.recarray): An array of (averaged) diagonalized closure phases
           camp (numpy.recarray): An array of (averaged) closure amplitudes
           logcamp (numpy.recarray): An array of (averaged) log closure amplitudes
           logcamp_diag (numpy.recarray): An array of (averaged) diagonalized log closure amps
    """

    def __init__(self, ra, dec, rf, bw, datatable, tarr, scantable=None,
                 polrep='stokes', source=ehc.SOURCE_DEFAULT, mjd=ehc.MJD_DEFAULT, timetype='UTC',
                 ampcal=True, phasecal=True, opacitycal=True, dcal=True, frcal=True,
                 trial_speedups=False):
        """A polarimetric VLBI observation of visibility amplitudes and phases (in Jy).

           Args:
               ra (float): The source Right Ascension in fractional hours
               dec (float): The source declination in fractional degrees
               rf (float): The observation frequency in Hz
               bw (float): The observation bandwidth in Hz

               datatable (numpy.recarray): the basic data with datatype DTPOL_STOKES or DTPOL_CIRC
               tarr (numpy.recarray): The array of telescope data with datatype DTARR
               scantable (numpy.recarray): The array of scan information

               polrep (str): polarization representation, either 'stokes' or 'circ'
               source (str): The source name
               mjd (int): The integer MJD of the observation
               timetype (str): How to interpret tstart and tstop; either 'GMST' or 'UTC'

               ampcal (bool): True if amplitudes calibrated
               phasecal (bool): True if phases calibrated
               opacitycal (bool): True if time-dependent opacities correctly accounted for in sigmas
               frcal (bool): True if feed rotation calibrated out of visibilities
               dcal (bool): True if D terms calibrated out of visibilities

           Returns:
               obsdata (Obsdata): an Obsdata object
        """

        if len(datatable) == 0:
            raise Exception("No data in input table!")
        if not (datatable.dtype in [ehc.DTPOL_STOKES, ehc.DTPOL_CIRC]):
            raise Exception("Data table dtype should be DTPOL_STOKES or DTPOL_CIRC")

        # Polarization Representation
        if polrep == 'stokes':
            self.polrep = 'stokes'
            self.poldict = ehc.POLDICT_STOKES
            self.poltype = ehc.DTPOL_STOKES
        elif polrep == 'circ':
            self.polrep = 'circ'
            self.poldict = ehc.POLDICT_CIRC
            self.poltype = ehc.DTPOL_CIRC
        else:
            raise Exception("only 'stokes' and 'circ' are supported polreps!")

        # Set the various observation parameters
        self.source = str(source)
        self.ra = float(ra)
        self.dec = float(dec)
        self.rf = float(rf)
        self.bw = float(bw)
        self.ampcal = bool(ampcal)
        self.phasecal = bool(phasecal)
        self.opacitycal = bool(opacitycal)
        self.dcal = bool(dcal)
        self.frcal = bool(frcal)

        if timetype not in ['GMST', 'UTC']:
            raise Exception("timetype must be 'GMST' or 'UTC'")
        self.timetype = timetype

        # Save the data
        self.data = datatable
        self.scans = scantable

        # Telescope array: default ordering is by sefd
        self.tarr = tarr
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}
        if np.any(self.tarr['sefdr'] != 0) or np.any(self.tarr['sefdl'] != 0):
            self.reorder_tarr_sefd(reorder_baselines=False)
            
        # reorder baselines to uvfits convention
        self.reorder_baselines(trial_speedups=trial_speedups)

        # Get tstart, mjd and tstop
        times = self.unpack(['time'])['time']
        self.tstart = times[0]
        self.mjd = int(mjd)
        self.tstop = times[-1]
        if self.tstop < self.tstart:
            self.tstop += 24.0

        # Saved closure quantity arrays
        self.amp = None
        self.bispec = None
        self.cphase = None
        self.cphase_diag = None
        self.camp = None
        self.logcamp = None
        self.logcamp_diag = None

    @property 
    def tarr(self):
        return self._tarr
        
    @tarr.setter 
    def tarr(self, tarr):
        self._tarr = tarr
        self.tkey = {tarr[i]['site']: i for i in range(len(tarr))}
        
    def obsdata_args(self):
        """"Copy arguments for making a new Obsdata into a list and dictonary
        """

        arglist = [self.ra, self.dec, self.rf, self.bw, self.data, self.tarr]
        argdict = {'scantable': self.scans, 'polrep': self.polrep, 'source': self.source,
                   'mjd': self.mjd, 'timetype': self.timetype,
                   'ampcal': self.ampcal, 'phasecal': self.phasecal, 'opacitycal': self.opacitycal,
                   'dcal': self.dcal, 'frcal': self.frcal}
        return (arglist, argdict)

    def copy(self):
        """Copy the observation object.

           Args:

           Returns:
               (Obsdata): a copy of the Obsdata object.
        """

        # TODO: Do we want to copy over e.g. closure tables?
        newobs = copy.deepcopy(self)

        return newobs

    def switch_timetype(self, timetype_out='UTC'):
        """Return a new observation with the time type switched

           Args:
               timetype (str): "UTC" or "GMST"

           Returns:
               (Obsdata): new Obsdata object with potentially different timetype
        """

        if timetype_out not in ['GMST', 'UTC']:
            raise Exception("timetype_out must be 'GMST' or 'UTC'")

        out = self.copy()
        if timetype_out == self.timetype:
            return out

        if timetype_out == 'UTC':
            out.data['time'] = obsh.gmst_to_utc(out.data['time'], out.mjd)
        if timetype_out == 'GMST':
            out.data['time'] = obsh.utc_to_gmst(out.data['time'], out.mjd)

        out.timetype = timetype_out
        return out

    def switch_polrep(self, polrep_out='stokes', allow_singlepol=True, singlepol_hand='R'):
        """Return a new observation with the polarization representation changed

           Args:
               polrep_out (str):  the polrep of the output data
               allow_singlepol (bool): If True, treat single-polarization data as Stokes I
                                       when converting from 'circ' polrep to 'stokes'
               singlepol_hand (str): 'R' or 'L'; determines which parallel-hand is assumed
                                       when converting 'stokes' to 'circ' if only I is present

           Returns:
               (Obsdata): new Obsdata object with potentially different polrep
        """

        if polrep_out not in ['stokes', 'circ']:
            raise Exception("polrep_out must be either 'stokes' or 'circ'")
        if polrep_out == self.polrep:
            return self.copy()
        elif polrep_out == 'stokes':  # circ -> stokes
            data = np.empty(len(self.data), dtype=ehc.DTPOL_STOKES)
            rrmask = np.isnan(self.data['rrvis'])
            llmask = np.isnan(self.data['llvis'])

            for f in ehc.DTPOL_STOKES:
                f = f[0]
                if f in ['time', 'tint', 't1', 't2', 'tau1', 'tau2', 'u', 'v']:
                    data[f] = self.data[f]
                elif f == 'vis':
                    data[f] = 0.5 * (self.data['rrvis'] + self.data['llvis'])
                elif f == 'qvis':
                    data[f] = 0.5 * (self.data['lrvis'] + self.data['rlvis'])
                elif f == 'uvis':
                    data[f] = 0.5j * (self.data['lrvis'] - self.data['rlvis'])
                elif f == 'vvis':
                    data[f] = 0.5 * (self.data['rrvis'] - self.data['llvis'])
                elif f in ['sigma', 'vsigma']:
                    data[f] = 0.5 * np.sqrt(self.data['rrsigma']**2 + self.data['llsigma']**2)
                elif f in ['qsigma', 'usigma']:
                    data[f] = 0.5 * np.sqrt(self.data['rlsigma']**2 + self.data['lrsigma']**2)

            if allow_singlepol:
                # In cases where only one polarization is present
                # use it as an estimator for Stokes I
                data['vis'][rrmask] = self.data['llvis'][rrmask]
                data['sigma'][rrmask] = self.data['llsigma'][rrmask]

                data['vis'][llmask] = self.data['rrvis'][llmask]
                data['sigma'][llmask] = self.data['rrsigma'][llmask]

        elif polrep_out == 'circ':  # stokes -> circ
            data = np.empty(len(self.data), dtype=ehc.DTPOL_CIRC)
            Vmask = np.isnan(self.data['vvis'])

            for f in ehc.DTPOL_CIRC:
                f = f[0]
                if f in ['time', 'tint', 't1', 't2', 'tau1', 'tau2', 'u', 'v']:
                    data[f] = self.data[f]
                elif f == 'rrvis':
                    data[f] = (self.data['vis'] + self.data['vvis'])
                elif f == 'llvis':
                    data[f] = (self.data['vis'] - self.data['vvis'])
                elif f == 'rlvis':
                    data[f] = (self.data['qvis'] + 1j * self.data['uvis'])
                elif f == 'lrvis':
                    data[f] = (self.data['qvis'] - 1j * self.data['uvis'])
                elif f in ['rrsigma', 'llsigma']:
                    data[f] = np.sqrt(self.data['sigma']**2 + self.data['vsigma']**2)
                elif f in ['rlsigma', 'lrsigma']:
                    data[f] = np.sqrt(self.data['qsigma']**2 + self.data['usigma']**2)

            if allow_singlepol:
                # In cases where only Stokes I is present, copy it to a specified parallel-hand
                prefix = singlepol_hand.lower() + singlepol_hand.lower()  # rr or ll
                if prefix not in ['rr', 'll']:
                    raise Exception('singlepol_hand must be R or L')

                data[prefix + 'vis'][Vmask] = self.data['vis'][Vmask]
                data[prefix + 'sigma'][Vmask] = self.data['sigma'][Vmask]

        arglist, argdict = self.obsdata_args()
        arglist[DATPOS] = data
        argdict['polrep'] = polrep_out
        newobs = Obsdata(*arglist, **argdict)

        return newobs

    def reorder_baselines(self, trial_speedups=False):
        """Reorder baselines to match uvfits convention, based on the telescope array ordering
        """
        if trial_speedups:
            self.reorder_baselines_trial_speedups()
        
        else: # original code
        
            # Time partition the datatable
            datatable = self.data.copy()
            datalist = []
            for key, group in it.groupby(datatable, lambda x: x['time']):
                #print(key*60*60,len(list(group)))
                datalist.append(np.array([obs for obs in group]))

            # loop through all data
            obsdata = []
            for tlist in datalist:
                blpairs = []
                for dat in tlist:
                    # Remove conjugate baselines
                    if not (set((dat['t1'], dat['t2']))) in blpairs:         

                        # Reverse the baseline in the right order for uvfits:
                        if(self.tkey[dat['t2']] < self.tkey[dat['t1']]):

                            (dat['t1'], dat['t2']) = (dat['t2'], dat['t1'])
                            (dat['tau1'], dat['tau2']) = (dat['tau2'], dat['tau1'])
                            dat['u'] = -dat['u']
                            dat['v'] = -dat['v']

                            if self.polrep == 'stokes':
                                dat['vis'] = np.conj(dat['vis'])
                                dat['qvis'] = np.conj(dat['qvis'])
                                dat['uvis'] = np.conj(dat['uvis'])
                                dat['vvis'] = np.conj(dat['vvis'])
                            elif self.polrep == 'circ':
                                dat['rrvis'] = np.conj(dat['rrvis'])
                                dat['llvis'] = np.conj(dat['llvis'])
                                # must switch l & r !!
                                rl = dat['rlvis'].copy()
                                lr = dat['lrvis'].copy()
                                dat['rlvis'] = np.conj(lr)
                                dat['lrvis'] = np.conj(rl)
                                
                                # You also have to switch the errors for the coherency!
                                rlerr = dat['rlsigma'].copy()
                                lrerr = dat['lrsigma'].copy()
                                dat["rlsigma"] = lrerr
                                dat["lrsigma"] = rlerr

                            else:
                                raise Exception("polrep must be either 'stokes' or 'circ'")

                        # Append the data point
                        blpairs.append(set((dat['t1'], dat['t2'])))
                        obsdata.append(dat)

            obsdata = np.array(obsdata, dtype=self.poltype)

            # Timesort data
            obsdata = obsdata[np.argsort(obsdata, order=['time', 't1'])]

            # Save the data
            self.data = obsdata

        return

    def reorder_baselines_trial_speedups(self):
        """Reorder baselines to match uvfits convention, based on the telescope array ordering
        """

        dat = self.data.copy()
              
        ############ Ensure correct baseline order
        # TODO can these be faster? 
        t1nums = np.fromiter([self.tkey[t] for t in dat['t1']],int) 
        t2nums = np.fromiter([self.tkey[t] for t in dat['t2']],int) 
        
        # which entries are in the wrong telescope order?
        ordermask = t2nums < t1nums
        
        # flip the order of these entries
        t1 = dat['t1'].copy()
        t2 = dat['t2'].copy()
        tau1 = dat['tau1'].copy()
        tau2 = dat['tau2'].copy()     
          
        dat['t1'][ordermask] = t2[ordermask]
        dat['t2'][ordermask] = t1[ordermask]
        dat['tau1'][ordermask] = tau2[ordermask]
        dat['tau2'][ordermask] = tau1[ordermask]
        dat['u'][ordermask] *= -1
        dat['v'][ordermask] *= -1
                    
        if self.polrep=='stokes':
            dat['vis'][ordermask]  = np.conj(dat['vis'][ordermask])
            dat['qvis'][ordermask] = np.conj(dat['qvis'][ordermask])
            dat['uvis'][ordermask] = np.conj(dat['uvis'][ordermask])
            dat['vvis'][ordermask] = np.conj(dat['vvis'][ordermask])                         

        elif self.polrep == 'circ':
            dat['rrvis'][ordermask] = np.conj(dat['rrvis'][ordermask])
            dat['llvis'][ordermask] = np.conj(dat['llvis'][ordermask])
            rl = dat['rlvis'].copy()
            lr = dat['lrvis'].copy()
            dat['rlvis'][ordermask] = np.conj(lr[ordermask])
            dat['lrvis'][ordermask] = np.conj(rl[ordermask])

            # Also need to switch error matrix
            rle = dat['rlsigma'].copy()
            lre = dat['lrsigma'].copy()
            dat['rlsigma'][ordermask] = lre[ordermask]
            dat['lrsigma'][ordermask] = rle[ordermask]

        else:
            raise Exception("polrep must be either 'stokes' or 'circ'")

        # Remove duplicate or conjugate entries at any timestep
        # Since telescope order has been sorted conjugates should appear as duplicates
        timeblcombos = np.vstack((dat['time'],t1nums,t2nums)).T        
        uniqdat, uniqdatinv = np.unique(timeblcombos,axis=0, return_inverse=True)
        
        if len(uniqdat) != len(dat): 
            print("WARNING: removing duplicate/conjuagte points in reorder_baselines!")
            deletemask = np.ones(len(dat)).astype(bool)

            for j in len(uniqdat):
                idxs = np.argwhere(uniqdatinv==j)[:,0]
                for idx in idxs[1:]: # delete all but first occurance
                    deletemask[idx] = False
            
            # remove duplicates   
            dat_unique = dat[deletemask]                 
                  
        # sort data
        dat = dat[np.argsort(dat, order=['time', 't1'])]

        # save the data
        self.data = dat

        return

    def reorder_tarr_sefd(self, reorder_baselines=True):
        """Reorder the telescope array by SEFD minimal to maximum.
        """

        sorted_list = sorted(self.tarr, key=lambda x: np.sqrt(x['sefdr']**2 + x['sefdl']**2))
        self.tarr = np.array(sorted_list, dtype=ehc.DTARR)
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}
        if reorder_baselines:
            self.reorder_baselines()

        return

    def reorder_tarr_snr(self, reorder_baselines=True):
        """Reorder the telescope array by median SNR maximal to minimal.
        """

        snr = self.unpack(['t1', 't2', 'snr'])
        snr_median = [np.median(snr[(snr['t1'] == site) + (snr['t2'] == site)]['snr'])
                      for site in self.tarr['site']]
        idx = np.argsort(snr_median)[::-1]
        self.tarr = self.tarr[idx]
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}
        if reorder_baselines:
            self.reorder_baselines()

        return

    def reorder_tarr_random(self, reorder_baselines=True):
        """Randomly reorder the telescope array.
        """

        idx = np.arange(len(self.tarr))
        np.random.shuffle(idx)
        self.tarr = self.tarr[idx]
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}
        if reorder_baselines:
            self.reorder_baselines()

        return

    def data_conj(self):
        """Make a data array including all conjugate baselines.

           Args:

           Returns:
               (numpy.recarray): a copy of the Obsdata.data table including all conjugate baselines.
        """

        data = np.empty(2 * len(self.data), dtype=self.poltype)

        # Add the conjugate baseline data
        for f in self.poltype:
            f = f[0]
            if f in ['t1', 't2', 'tau1', 'tau2']:
                if f[-1] == '1':
                    f2 = f[:-1] + '2'
                else:
                    f2 = f[:-1] + '1'
                data[f] = np.hstack((self.data[f], self.data[f2]))

            elif f in ['u', 'v']:
                data[f] = np.hstack((self.data[f], -self.data[f]))

            elif f in [self.poldict['vis1'], self.poldict['vis2'],
                       self.poldict['vis3'], self.poldict['vis4']]:
                if self.polrep == 'stokes':
                    data[f] = np.hstack((self.data[f], np.conj(self.data[f])))
                elif self.polrep == 'circ':
                    if f in ['rrvis', 'llvis']:
                        data[f] = np.hstack((self.data[f], np.conj(self.data[f])))
                    elif f == 'rlvis':
                        data[f] = np.hstack((self.data['rlvis'], np.conj(self.data['lrvis'])))
                    elif f == 'lrvis':
                        data[f] = np.hstack((self.data['lrvis'], np.conj(self.data['rlvis'])))

                    # ALSO SWITCH THE ERRORS!                    
                else:
                    raise Exception("polrep must be either 'stokes' or 'circ'")
            # The conjugate baselines need the transpose error terms.
            elif f == "rlsigma":
                data[f] = np.hstack((self.data["rlsigma"], self.data["lrsigma"]))
            elif f == "lrsigma":
                data[f] = np.hstack((self.data["lrsigma"], self.data["rlsigma"]))

            else:
                data[f] = np.hstack((self.data[f], self.data[f]))

        # Sort the data by time
        data = data[np.argsort(data['time'])]

        return data

    def tlist(self, conj=False, t_gather=0., scan_gather=False):
        """Group the data in a list of equal time observation datatables.

           Args:
                conj (bool): True if tlist_out includes conjugate baselines.
                t_gather (float): Grouping timescale (in seconds). 0.0 indicates no grouping.
                scan_gather (bool): If true, gather data into scans

           Returns:
                (list): a list of data tables containing time-partitioned data
        """

        if conj:
            data = self.data_conj()
        else:
            data = self.data

        # partition the data by time
        datalist = []

        if t_gather <= 0.0 and not scan_gather:
            # Only group measurements at the same time
            for key, group in it.groupby(data, lambda x: x['time']):
                datalist.append(np.array([obs for obs in group]))
        elif t_gather > 0.0 and not scan_gather:
            # Group measurements in time
            for key, group in it.groupby(data, lambda x: int(x['time'] / (t_gather / 3600.0))):
                datalist.append(np.array([obs for obs in group]))
        else:
            # Group measurements by scan
            if ((self.scans is None) or 
                 np.any([scan is None for scan in self.scans]) or
                 len(self.scans) == 0):
                print("No scan table in observation. Adding scan table before gathering...")
                self.add_scans()

            for key, group in it.groupby(
                    data, lambda x: np.searchsorted(self.scans[:, 0], x['time'])):
                datalist.append(np.array([obs for obs in group]))

        return np.array(datalist, dtype=object)


    def split_obs(self, t_gather=0., scan_gather=False):
        """Split single observation into multiple observation files, one per scan..

           Args:
                t_gather (float): Grouping timescale (in seconds). 0.0 indicates no grouping.
                scan_gather (bool): If true, gather data into scans

            Returns:
                (list): list of single-scan Obsdata objects
        """

        tlist = self.tlist(t_gather=t_gather, scan_gather=scan_gather)

        print("Splitting Observation File into " + str(len(tlist)) + " times")
        arglist, argdict = self.obsdata_args()

        # note that the tarr of the output includes all sites,
        # even those that don't participate in the scan
        splitlist = []
        for tdata in tlist:
            arglist[DATPOS] = tdata
            splitlist.append(Obsdata(*arglist, **argdict))

        return splitlist


    def getClosestScan(self, time, splitObs=None):
        """Split observation by scan and grab scan closest to timestamp

           Args:
                time (float): Time (GMST) you want to find the scan closest to
                splitObs (bool): a list of Obsdata objects, output from split_obs, to save time

            Returns:
                (Obsdata): Obsdata object composed of scan closest to time
        """

        ## check if splitObs has been passed in alread ##
        if splitObs is None:
            splitObs = self.split_obs()

        ## check for the scan with the closest start time to time arg ##
        ## TODO: allow user to choose start time, end time, or mid-time
        closest_index = 0
        delta_t = 1e22
        for s, s_obs in enumerate(splitObs):
            dt = abs(s_obs.tstart - time)
            if dt < delta_t:
                delta_t = dt 
                closest_index = s 

        print(f"Using scan with time {splitObs[closest_index].tstart}.")
        return splitObs[closest_index]


    def bllist(self, conj=False):
        """Group the data in a list of same baseline datatables.

           Args:
                conj (bool): True if tlist_out includes conjugate baselines.

           Returns:
                (list): a list of data tables containing baseline-partitioned data
        """

        if conj:
            data = self.data_conj()
        else:
            data = self.data

        # partition the data by baseline
        datalist = []
        idx = np.lexsort((data['t2'], data['t1']))
        for key, group in it.groupby(data[idx], lambda x: set((x['t1'], x['t2']))):
            datalist.append(np.array([obs for obs in group]))

        return np.array(datalist, dtype=object)
        
    def unpack_bl(self, site1, site2, fields, ang_unit='deg', debias=False, timetype=False):
        """Unpack the data over time on the selected baseline site1-site2.

           Args:
                site1 (str): First site name
                site2 (str): Second site name
                fields (list): list of unpacked quantities from available quantities in FIELDS
                ang_unit (str): 'deg' for degrees and 'rad' for radian phases
                debias (bool): True to debias visibility amplitudes
                timetype (str): 'GMST' or 'UTC' changes what is returned for 'time'

           Returns:
                (numpy.recarray): unpacked numpy array with data in fields requested
        """

        if timetype is False:
            timetype = self.timetype

        # If we only specify one field
        if timetype not in ['GMST', 'UTC', 'utc', 'gmst']:
            raise Exception("timetype should be 'GMST' or 'UTC'!")
        allfields = ['time']

        if not isinstance(fields, list):
            allfields.append(fields)
        else:
            for i in range(len(fields)):
                allfields.append(fields[i])

        # Get the data from data table on the selected baseline
        allout = []
        tlist = self.tlist(conj=True)
        for scan in tlist:
            for obs in scan:
                if (obs['t1'], obs['t2']) == (site1, site2):
                    obs = np.array([obs])
                    out = self.unpack_dat(obs, allfields, ang_unit=ang_unit,
                                          debias=debias, timetype=timetype)

                    allout.append(out)
        
        return np.array(allout)
        
    def unpack(self, fields, mode='all', ang_unit='deg', debias=False, conj=False, timetype=False):
        """Unpack the data for the whole observation .

           Args:
                fields (list): list of unpacked quantities from availalbe quantities in FIELDS
                mode (str): 'all' returns all data in single table,
                            'time' groups output by equal time, 'bl' groups by baseline
                ang_unit (str): 'deg' for degrees and 'rad' for radian phases
                debias (bool): True to debias visibility amplitudes
                conj (bool): True to include conjugate baselines
                timetype (str): 'GMST' or 'UTC' changes what is returned for 'time'

           Returns:
                (numpy.recarray): unpacked numpy array with data in fields requested

        """

        if mode not in ('time', 'all', 'bl'):
            raise Exception("possible options for mode are 'time', 'all' and 'bl'")

        # If we only specify one field
        if not isinstance(fields, list):
            fields = [fields]

        if mode == 'all':
            if conj:
                data = self.data_conj()
            else:
                data = self.data
            allout = self.unpack_dat(data, fields, ang_unit=ang_unit,
                                     debias=debias, timetype=timetype)

        elif mode == 'time':
            allout = []
            tlist = self.tlist(conj=True)
            for scan in tlist:
                out = self.unpack_dat(scan, fields, ang_unit=ang_unit,
                                      debias=debias, timetype=timetype)
                allout.append(out)

        elif mode == 'bl':
            allout = []
            bllist = self.bllist()
            for bl in bllist:
                out = self.unpack_dat(bl, fields, ang_unit=ang_unit,
                                      debias=debias, timetype=timetype)
                allout.append(out)

        return allout

    def unpack_dat(self, data, fields, conj=False, ang_unit='deg', debias=False, timetype=False):
        """Unpack the data in a passed data recarray.

           Args:
                data (numpy.recarray): data recarray of format DTPOL_STOKES or DTPOL_CIRC
                fields (list): list of unpacked quantities from availalbe quantities in FIELDS
                conj (bool): True to include conjugate baselines
                ang_unit (str): 'deg' for degrees and 'rad' for radian phases
                debias (bool): True to debias visibility amplitudes
                timetype (str): 'GMST' or 'UTC' changes what is returned for 'time'

           Returns:
                (numpy.recarray): unpacked numpy array with data in fields requested

        """

        if ang_unit == 'deg':
            angle = ehc.DEGREE
        else:
            angle = 1.0

        # If we only specify one field
        if isinstance(fields, str):
            fields = [fields]

        if not timetype:
            timetype = self.timetype
        if timetype not in ['GMST', 'UTC', 'gmst', 'utc']:
            raise Exception("timetype should be 'GMST' or 'UTC'!")

        # Get field data
        allout = []
        for field in fields:
            if field in ["time", "time_utc", "time_gmst"]:
                out = data['time']
                ty = 'f8'
            elif field in ["u", "v", "tint", "tau1", "tau2"]:
                out = data[field]
                ty = 'f8'
            elif field in ["uvdist"]:
                out = np.abs(data['u'] + 1j * data['v'])
                ty = 'f8'
            elif field in ["t1", "el1", "par_ang1", "hr_ang1"]:
                sites = data["t1"]
                keys = [self.tkey[site] for site in sites]
                tdata = self.tarr[keys]
                out = sites
                ty = 'U32'
            elif field in ["t2", "el2", "par_ang2", "hr_ang2"]:
                sites = data["t2"]
                keys = [self.tkey[site] for site in sites]
                tdata = self.tarr[keys]
                out = sites
                ty = 'U32'
            elif field in ['vis', 'amp', 'phase', 'snr', 'sigma', 'sigma_phase']:
                ty = 'c16'
                if self.polrep == 'stokes':
                    out = data['vis']
                    sig = data['sigma']
                elif self.polrep == 'circ':
                    out = 0.5 * (data['rrvis'] + data['llvis'])
                    sig = 0.5 * np.sqrt(data['rrsigma']**2 + data['llsigma']**2)
            elif field in ['qvis', 'qamp', 'qphase', 'qsnr', 'qsigma', 'qsigma_phase']:
                ty = 'c16'
                if self.polrep == 'stokes':
                    out = data['qvis']
                    sig = data['qsigma']
                elif self.polrep == 'circ':
                    out = 0.5 * (data['lrvis'] + data['rlvis'])
                    sig = 0.5 * np.sqrt(data['lrsigma']**2 + data['rlsigma']**2)
            elif field in ['uvis', 'uamp', 'uphase', 'usnr', 'usigma', 'usigma_phase']:
                ty = 'c16'
                if self.polrep == 'stokes':
                    out = data['uvis']
                    sig = data['usigma']
                elif self.polrep == 'circ':
                    out = 0.5j * (data['lrvis'] - data['rlvis'])
                    sig = 0.5 * np.sqrt(data['lrsigma']**2 + data['rlsigma']**2)
            elif field in ['vvis', 'vamp', 'vphase', 'vsnr', 'vsigma', 'vsigma_phase']:
                ty = 'c16'
                if self.polrep == 'stokes':
                    out = data['vvis']
                    sig = data['vsigma']
                elif self.polrep == 'circ':
                    out = 0.5 * (data['rrvis'] - data['llvis'])
                    sig = 0.5 * np.sqrt(data['rrsigma']**2 + data['llsigma']**2)
            elif field in ['pvis', 'pamp', 'pphase', 'psnr', 'psigma', 'psigma_phase']:
                ty = 'c16'
                if self.polrep == 'stokes':
                    out = data['qvis'] + 1j * data['uvis']
                    sig = np.sqrt(data['qsigma']**2 + data['usigma']**2)
                elif self.polrep == 'circ':
                    out = data['rlvis']
                    sig = data['rlsigma']
            elif field in ['m', 'mamp', 'mphase', 'msnr', 'msigma', 'msigma_phase']:
                ty = 'c16'
                if self.polrep == 'stokes':
                    out = (data['qvis'] + 1j * data['uvis']) / data['vis']
                    sig = obsh.merr(data['sigma'], data['qsigma'], data['usigma'], data['vis'], out)
                elif self.polrep == 'circ':
                    out = 2 * data['rlvis'] / (data['rrvis'] + data['llvis'])
                    sig = obsh.merr2(data['rlsigma'], data['rrsigma'], data['llsigma'],
                                     0.5 * (data['rrvis'] + data['llvis']), out)
            elif field in ['evis', 'eamp', 'ephase', 'esnr', 'esigma', 'esigma_phase']:
                ty = 'c16'
                ang = np.arctan2(data['u'], data['v'])  # TODO: correct convention EofN?
                if self.polrep == 'stokes':
                    q = data['qvis']
                    u = data['uvis']
                    qsig = data['qsigma']
                    usig = data['usigma']
                elif self.polrep == 'circ':
                    q = 0.5 * (data['lrvis'] + data['rlvis'])
                    u = 0.5j * (data['lrvis'] - data['rlvis'])
                    qsig = 0.5 * np.sqrt(data['lrsigma']**2 + data['rlsigma']**2)
                    usig = qsig
                out = (np.cos(2 * ang) * q + np.sin(2 * ang) * u)
                sig = np.sqrt(0.5 * ((np.cos(2 * ang) * qsig)**2 + (np.sin(2 * ang) * usig)**2))
            elif field in ['bvis', 'bamp', 'bphase', 'bsnr', 'bsigma', 'bsigma_phase']:
                ty = 'c16'
                ang = np.arctan2(data['u'], data['v'])  # TODO: correct convention EofN?
                if self.polrep == 'stokes':
                    q = data['qvis']
                    u = data['uvis']
                    qsig = data['qsigma']
                    usig = data['usigma']
                elif self.polrep == 'circ':
                    q = 0.5 * (data['lrvis'] + data['rlvis'])
                    u = 0.5j * (data['lrvis'] - data['rlvis'])
                    qsig = 0.5 * np.sqrt(data['lrsigma']**2 + data['rlsigma']**2)
                    usig = qsig
                out = (-np.sin(2 * ang) * q + np.cos(2 * ang) * u)
                sig = np.sqrt(0.5 * ((np.sin(2 * ang) * qsig)**2 + (np.cos(2 * ang) * usig)**2))
            elif field in ['rrvis', 'rramp', 'rrphase', 'rrsnr', 'rrsigma', 'rrsigma_phase']:
                ty = 'c16'
                if self.polrep == 'stokes':
                    out = data['vis'] + data['vvis']
                    sig = np.sqrt(data['sigma']**2 + data['vsigma']**2)
                elif self.polrep == 'circ':
                    out = data['rrvis']
                    sig = data['rrsigma']
            elif field in ['llvis', 'llamp', 'llphase', 'llsnr', 'llsigma', 'llsigma_phase']:
                ty = 'c16'
                if self.polrep == 'stokes':
                    out = data['vis'] - data['vvis']
                    sig = np.sqrt(data['sigma']**2 + data['vsigma']**2)
                elif self.polrep == 'circ':
                    out = data['llvis']
                    sig = data['llsigma']
            elif field in ['rlvis', 'rlamp', 'rlphase', 'rlsnr', 'rlsigma', 'rlsigma_phase']:
                ty = 'c16'
                if self.polrep == 'stokes':
                    out = data['qvis'] + 1j * data['uvis']
                    sig = np.sqrt(data['qsigma']**2 + data['usigma']**2)
                elif self.polrep == 'circ':
                    out = data['rlvis']
                    sig = data['rlsigma']
            elif field in ['lrvis', 'lramp', 'lrphase', 'lrsnr', 'lrsigma', 'lrsigma_phase']:
                ty = 'c16'
                if self.polrep == 'stokes':
                    out = data['qvis'] - 1j * data['uvis']
                    sig = np.sqrt(data['qsigma']**2 + data['usigma']**2)
                elif self.polrep == 'circ':
                    out = data['lrvis']
                    sig = data['lrsigma']
            elif field in ['rrllvis', 'rrllamp', 'rrllphase', 'rrllsnr',
                           'rrllsigma', 'rrllsigma_phase']:
                ty = 'c16'
                if self.polrep == 'stokes':
                    out = (data['vis'] + data['vvis']) / (data['vis'] - data['vvis'])
                    sig = (2.0**0.5 * (np.abs(data['vis'])**2 + np.abs(data['vvis'])**2)**0.5
                           / np.abs(data['vis'] - data['vvis'])**2
                           * (data['sigma']**2 + data['vsigma']**2)**0.5)
                elif self.polrep == 'circ':
                    out = data['rrvis'] / data['llvis']
                    sig = np.sqrt(np.abs(data['rrsigma'] / data['llvis'])**2
                                  + np.abs(data['llsigma'] * data['rrvis'] / data['llvis'])**2)

            else:
                raise Exception("%s is not a valid field \n" % field +
                                "valid field values are: " + ' '.join(ehc.FIELDS))

            if field in ["time_utc"] and self.timetype == 'GMST':
                out = obsh.gmst_to_utc(out, self.mjd)
            if field in ["time_gmst"] and self.timetype == 'UTC':
                out = obsh.utc_to_gmst(out, self.mjd)
            if field in ["time"] and self.timetype == 'GMST' and timetype == 'UTC':
                out = obsh.gmst_to_utc(out, self.mjd)
            if field in ["time"] and self.timetype == 'UTC' and timetype == 'GMST':
                out = obsh.utc_to_gmst(out, self.mjd)

            # Compute elevation and parallactic angles
            if field in ["el1", "el2", "hr_ang1", "hr_ang2", "par_ang1", "par_ang2"]:
                if self.timetype == 'GMST':
                    times_sid = data['time']
                else:
                    times_sid = obsh.utc_to_gmst(data['time'], self.mjd)

                thetas = np.mod((times_sid - self.ra) * ehc.HOUR, 2 * np.pi)
                coords = obsh.recarr_to_ndarr(tdata[['x', 'y', 'z']], 'f8')
                el_angle = obsh.elev(obsh.earthrot(coords, thetas), self.sourcevec())
                latlon = obsh.xyz_2_latlong(coords)
                hr_angles = obsh.hr_angle(times_sid * ehc.HOUR, latlon[:, 1], self.ra * ehc.HOUR)

                if field in ["el1", "el2"]:
                    out = el_angle / angle
                    ty = 'f8'
                if field in ["hr_ang1", "hr_ang2"]:
                    out = hr_angles / angle
                    ty = 'f8'
                if field in ["par_ang1", "par_ang2"]:
                    par_ang = obsh.par_angle(hr_angles, latlon[:, 0], self.dec * ehc.DEGREE)
                    out = par_ang / angle
                    ty = 'f8'

            # Get arg/amps/snr
            if field in ["amp", "qamp", "uamp", "vamp", "pamp", "mamp", "bamp", "eamp",
                         "rramp", "llamp", "rlamp", "lramp", "rrllamp"]:
                out = np.abs(out)
                if debias:
                    out = obsh.amp_debias(out, sig)
                ty = 'f8'
            elif field in ["sigma", "qsigma", "usigma", "vsigma",
                           "psigma", "msigma", "bsigma", "esigma",
                           "rrsigma", "llsigma", "rlsigma", "lrsigma", "rrllsigma"]:
                out = np.abs(sig)
                ty = 'f8'
            elif field in ["phase", "qphase", "uphase", "vphase", "pphase", "bphase", "ephase",
                           "mphase", "rrphase", "llphase", "lrphase", "rlphase", "rrllphase"]:
                out = np.angle(out) / angle
                ty = 'f8'
            elif field in ["sigma_phase", "qsigma_phase", "usigma_phase", "vsigma_phase",
                           "psigma_phase", "msigma_phase", "bsigma_phase", "esigma_phase",
                           "rrsigma_phase", "llsigma_phase", "rlsigma_phase", "lrsigma_phase",
                           "rrllsigma_phase"]:
                out = np.abs(sig) / np.abs(out) / angle
                ty = 'f8'
            elif field in ["snr", "qsnr", "usnr", "vsnr", "psnr", "bsnr", "esnr",
                           "msnr", "rrsnr", "llsnr", "rlsnr", "lrsnr", "rrllsnr"]:
                out = np.abs(out) / np.abs(sig)
                ty = 'f8'

            # Reshape and stack with other fields
            out = np.array(out, dtype=[(field, ty)])

            if len(allout) > 0:
                allout = rec.merge_arrays((allout, out), asrecarray=True, flatten=True)
            else:
                allout = out

        return allout

    def sourcevec(self):
        """Return the source position vector in geocentric coordinates at 0h GMST.

           Args:

           Returns:
                (numpy.array): normal vector pointing to source in geocentric coordinates (m)
        """

        sourcevec = np.array([np.cos(self.dec * ehc.DEGREE), 0, np.sin(self.dec * ehc.DEGREE)])

        return sourcevec

    def res(self):
        """Return the nominal resolution (1/longest baseline) of the observation in radians.

           Args:

           Returns:
                (float): normal array resolution in radians
        """

        res = 1.0 / np.max(self.unpack('uvdist')['uvdist'])

        return res


    def chisq(self, im_or_mov, dtype='vis', pol='I', ttype='nfft', mask=[], **kwargs):
        """Give the reduced chi^2 of the observation for the specified image and datatype.

           Args:
                im_or_mov (Image or Movie): image or movie object on which to test chi^2
                dtype (str): data type of chi^2 (e.g., 'vis', 'amp', 'bs', 'cphase')
                pol (str): polarization type ('I', 'Q', 'U', 'V', 'LL', 'RR', 'LR', or 'RL'
                mask (arr): mask of same dimension as im.imvec
                ttype (str): "fast" or "nfft" or "direct"
                fft_pad_factor (float): zero pad the image to (fft_pad_factor * image size) in FFT
                conv_func ('str'):  The convolving function for gridding; 'gaussian', 'pill','cubic'
                p_rad (int): The pixel radius for the convolving function
                order ('str'): Interpolation order for sampling the FFT

                systematic_noise (float): adds a fractional systematic noise tolerance to sigmas
                snrcut (float): a  snr cutoff for including data in the chi^2 sum
                debias (bool): if True then apply debiasing to amplitudes/closure amplitudes
                weighting (str): 'natural' or 'uniform'

                systematic_cphase_noise (float): a value in degrees to add to closure phase sigmas
                cp_uv_min (float): flag short baselines before forming closure quantities
                maxset (bool):  if True, use maximal set instead of minimal for closure quantities

           Returns:
                (float): image chi^2
        """
        if dtype not in ['vis', 'bs', 'amp', 'cphase',
                         'cphase_diag', 'camp', 'logcamp', 'logcamp_diag', 'm']:
            raise Exception("%s is not a supported dterms!" % dtype)

        # TODO -- should import this at top, but the circular dependencies create a mess...
        import ehtim.imaging.imager_utils as iu
        import ehtim.modeling.modeling_utils as mu

        # Movie -- weighted sum of all frame chi^2 values
        if hasattr(im_or_mov, 'get_image'):
            mov = im_or_mov
            obs_list = self.split_obs()

            chisq_list = []
            num_list = []
            for ii, obs in enumerate(obs_list):

                im = mov.get_image(obs.data[0]['time'])  # Get image at the observation time

                if pol not in im._imdict.keys():
                    raise Exception(pol + ' is not in the current image.' +
                                          ' Consider changing the polarization basis of the image.')

                try:
                    (data, sigma, A) = iu.chisqdata(obs, im, mask, dtype,
                                                    pol=pol, ttype=ttype, **kwargs)

                except IndexError:  # Not enough data to form closures!
                    continue

                imvec = im._imdict[pol]
                if len(mask) > 0 and np.any(np.invert(mask)):
                    imvec = imvec[mask]

                chisq_list.append(iu.chisq(imvec, A, data, sigma, dtype, ttype=ttype, mask=mask))
                num_list.append(len(data))

            chisq = np.sum(np.array(num_list) * np.array(chisq_list)) / np.sum(num_list)

        # Model -- single chi^2
        elif hasattr(im_or_mov,'N_models'):            
            (data, sigma, uv, jonesdict) = mu.chisqdata(self, dtype, pol, **kwargs)
            chisq = mu.chisq(im_or_mov, uv, data, sigma, dtype, jonesdict)

        # Image -- single chi^2
        else:
            im = im_or_mov
            if pol not in im._imdict.keys():
                raise Exception(pol + ' is not in the current image.' +
                                      ' Consider changing the polarization basis of the image.')

            (data, sigma, A) = iu.chisqdata(self, im, mask, dtype, pol=pol, ttype=ttype, **kwargs)

            imvec = im._imdict[pol]
            if len(mask) > 0 and np.any(np.invert(mask)):
                imvec = imvec[mask]

            chisq = iu.chisq(imvec, A, data, sigma, dtype, ttype=ttype, mask=mask)

        return chisq

    def polchisq(self, im, dtype='pvis', ttype='nfft', mask=[], **kwargs):
        """Give the reduced chi^2 for the specified image and polarimetric datatype.

           Args:
                im (Image): image to test polarimetric chi^2
                dtype (str): data type of polarimetric chi^2 ('pvis','m','pbs')
                pol (str): polarization type ('I', 'Q', 'U', 'V', 'LL', 'RR', 'LR', or 'RL'
                mask (arr): mask of same dimension as im.imvec
                ttype (str): if "fast" or "nfft" or "direct"

                fft_pad_factor (float): zero pad the image to (fft_pad_factor * image size) in FFT
                conv_func ('str'):  The convolving function for gridding; 'gaussian', 'pill','cubic'
                p_rad (int): The pixel radius for the convolving function
                order ('str'): Interpolation order for sampling the FFT

                systematic_noise (float): adds a fractional systematic noise tolerance to sigmas
                snrcut (float): a  snr cutoff for including data in the chi^2 sum
                debias (bool): if True then apply debiasing to amplitudes/closure amplitudes
                weighting (str): 'natural' or 'uniform'

                systematic_cphase_noise (float): value in degrees to add to closure phase sigmas
                cp_uv_min (float): flag short baselines before forming closure quantities
                maxset (bool):  if True, use maximal set instead of minimal for closure quantities

           Returns:
                (float): image chi^2
        """

        if dtype not in ['pvis', 'm', 'pbs','vvis']:
            raise Exception("Only supported polarimetric dterms are 'pvis','m, 'pbs','vvis'!")

        # TODO -- should import this at top, but the circular dependencies create a mess...
        import ehtim.imaging.pol_imager_utils as piu

        # Unpack the necessary polarimetric data
        (data, sigma, A) = piu.polchisqdata(self, im, mask, dtype, ttype=ttype, **kwargs)

        # Pack the comparison image in the proper format
        imstokes = im.switch_polrep(polrep_out='stokes', pol_prim_out='I')

        ivec = imstokes.imvec
        rhovec = np.sqrt(imstokes.qvec**2 + imstokes.uvec**2 + imstokes.vvec**2) / ivec
        phivec = np.angle(imstokes.qvec + 1j * imstokes.uvec) 
        psivec = np.arcsin(vvec/ivec)
        if len(mask) > 0 and np.any(np.invert(mask)):
            ivec = ivec[mask]
            rhovec = rhovec[mask]
            phivec = phivec[mask]
            psivec = psivec[mask]
        imarr = np.array((ivec, rhovec, phivec, psivec))


        # Calculate the chi^2
        chisq = piu.polchisq(imarr, A, data, sigma, dtype,
                             ttype=ttype, mask=mask)

        return chisq

    def recompute_uv(self):
        """Recompute u,v points using observation times and metadata

           Args:

           Returns:
                (Obsdata): New Obsdata object containing the same data with recomputed u,v points
        """

        times = self.data['time']
        site1 = self.data['t1']
        site2 = self.data['t2']
        arr = ehtim.array.Array(self.tarr)
        print("Recomputing U,V Points using MJD %d \n RA %e \n DEC %e \n RF %e GHz"
              % (self.mjd, self.ra, self.dec, self.rf / 1.e9))

        (timesout, uout, vout) = obsh.compute_uv_coordinates(arr, site1, site2, times,
                                                             self.mjd, self.ra, self.dec, self.rf,
                                                             timetype=self.timetype,
                                                             elevmin=0, elevmax=90, no_elevcut_space=False)

        if len(timesout) != len(times):
            raise Exception(
                "len(timesout) != len(times) in recompute_uv: check elevation  limits!!")

        datatable = self.data.copy()
        datatable['u'] = uout
        datatable['v'] = vout

        arglist, argdict = self.obsdata_args()
        arglist[DATPOS] = np.array(datatable)
        out = Obsdata(*arglist, **argdict)

        return out

    def avg_coherent(self, inttime, scan_avg=False, moving=False):
        """Coherently average data along u,v tracks in chunks of length inttime (sec)

           Args:
                inttime (float): coherent integration time in seconds
                scan_avg (bool): if True, average over scans in self.scans instead of intime
                moving (bool): averaging with moving window (boxcar width in seconds)
           Returns:
                (Obsdata): Obsdata object containing averaged data
        """

        if (scan_avg) and (getattr(self.scans, "shape", None) is None or len(self.scans) == 0):
            print('No scan data, ignoring scan_avg!')
            scan_avg = False

        if inttime <= 0.0 and scan_avg is False:
            print('No averaging done!')
            return self.copy()

        if moving:
            vis_avg = ehdf.coh_moving_avg_vis(self, dt=inttime, return_type='rec')
        else:
            vis_avg = ehdf.coh_avg_vis(self, dt=inttime, return_type='rec',
                                       err_type='predicted', scan_avg=scan_avg)

        arglist, argdict = self.obsdata_args()
        arglist[DATPOS] = vis_avg
        out = Obsdata(*arglist, **argdict)

        return out

    def avg_incoherent(self, inttime, scan_avg=False, debias=True, err_type='predicted'):
        """Incoherently average data along u,v tracks in chunks of length inttime (sec)

           Args:
                inttime (float): incoherent integration time in seconds
                scan_avg (bool): if True, average over scans in self.scans instead of intime
                debias (bool): if True, debias the averaged amplitudes
                err_type (str): 'predicted' or 'measured'

           Returns:
                (Obsdata): Obsdata object containing averaged data
        """

        print('Incoherently averaging data, putting phases to zero!')
        amp_rec = ehdf.incoh_avg_vis(self, dt=inttime, debias=debias, scan_avg=scan_avg,
                                     return_type='rec', rec_type='vis', err_type=err_type)
        arglist, argdict = self.obsdata_args()
        arglist[DATPOS] = amp_rec
        out = Obsdata(*arglist, **argdict)

        return out

    def add_amp(self, avg_time=0, scan_avg=False, debias=True, err_type='predicted',
                return_type='rec', round_s=0.1, snrcut=0.):
        """Adds attribute self.amp: aan amplitude table with incoherently averaged amplitudes

           Args:
               avg_time (float): incoherent integration time in seconds
               scan_avg (bool): if True, average over scans in self.scans instead of intime
               debias (bool): if True then apply debiasing
               err_type (str): 'predicted' or 'measured'
               return_type: data frame ('df') or recarray ('rec')
               round_s (float): accuracy of datetime object in seconds
               snrcut (float): flag amplitudes with snr lower than this

        """

        # Get the spacing between datapoints in seconds
        if len(set([x[0] for x in list(self.unpack('time'))])) > 1:
            tint0 = np.min(np.diff(np.asarray(sorted(list(set(
                           [x[0] for x in list(self.unpack('time'))])))))) * 3600.
        else:
            tint0 = 0.0

        if avg_time <= tint0:
            adf = ehdf.make_amp(self, debias=debias, round_s=round_s)
            if return_type == 'rec':
                adf = ehdf.df_to_rec(adf, 'amp')
            print("Updated self.amp: no averaging")
        else:
            adf = ehdf.incoh_avg_vis(self, dt=avg_time, debias=debias, scan_avg=scan_avg,
                                     return_type=return_type, rec_type='amp', err_type=err_type)

        # snr cut
        adf = adf[adf['amp'] / adf['sigma'] > snrcut]
        self.amp = adf
        print("Updated self.amp: avg_time %f s\n" % avg_time)

        return

    def add_bispec(self, avg_time=0, return_type='rec', count='max', snrcut=0.,
                   err_type='predicted', num_samples=1000, round_s=0.1, uv_min=False):
        """Adds attribute self.bispec: bispectra table with bispectra averaged for dt

           Args:
               avg_time (float): bispectrum averaging timescale
               return_type: data frame ('df') or recarray ('rec')
               count (str): If 'min', return minimal set of bispectra,
                            if 'max' return all bispectra up to reordering
               err_type (str): 'predicted' or 'measured'
               num_samples: number of bootstrap (re)samples if measuring error
               round_s (float): accuracy of datetime object in seconds
               snrcut (float): flag bispectra with snr lower than this

        """

        # Get spacing between datapoints in seconds
        if len(set([x[0] for x in list(self.unpack('time'))])) > 1:
            tint0 = np.min(np.diff(np.asarray(sorted(list(set(
                           [x[0] for x in list(self.unpack('time'))])))))) * 3600.
        else:
            tint0 = 0

        if avg_time > tint0:
            cdf = ehdf.make_bsp_df(self, mode='all', round_s=round_s, count=count,
                                   snrcut=0., uv_min=uv_min)
            cdf = ehdf.average_bispectra(cdf, avg_time, return_type=return_type,
                                         num_samples=num_samples, snrcut=snrcut)
        else:
            cdf = ehdf.make_bsp_df(self, mode='all', round_s=round_s, count=count,
                                   snrcut=snrcut, uv_min=uv_min)
            print("Updated self.bispec: no averaging")
            if return_type == 'rec':
                cdf = ehdf.df_to_rec(cdf, 'bispec')

        self.bispec = cdf
        print("Updated self.bispec: avg_time %f s\n" % avg_time)

        return

    def add_cphase(self, avg_time=0, return_type='rec', count='max', snrcut=0.,
                   err_type='predicted', num_samples=1000, round_s=0.1, uv_min=False):
        """Adds attribute self.cphase: cphase table averaged for dt

           Args:
               avg_time (float): closure phase averaging timescale
               return_type: data frame ('df') or recarray ('rec')
               count (str): If 'min', return minimal set of phases,
                            if 'max' return all closure phases up to reordering
               err_type (str): 'predicted' or 'measured'
               num_samples: number of bootstrap (re)samples if measuring error
               round_s (float): accuracy of datetime object in seconds
               snrcut (float): flag closure phases with snr lower than this

        """

        # Get spacing between datapoints in seconds
        if len(set([x[0] for x in list(self.unpack('time'))])) > 1:
            tint0 = np.min(np.diff(np.asarray(sorted(list(set(
                           [x[0] for x in list(self.unpack('time'))])))))) * 3600.
        else:
            tint0 = 0

        if avg_time > tint0:
            cdf = ehdf.make_cphase_df(self, mode='all', round_s=round_s, count=count,
                                      snrcut=0., uv_min=uv_min)
            cdf = ehdf.average_cphases(cdf, avg_time, return_type=return_type, err_type=err_type,
                                       num_samples=num_samples, snrcut=snrcut)
        else:
            cdf = ehdf.make_cphase_df(self, mode='all', round_s=round_s, count=count,
                                      snrcut=snrcut, uv_min=uv_min)
            if return_type == 'rec':
                cdf = ehdf.df_to_rec(cdf, 'cphase')
            print("Updated self.cphase: no averaging")

        self.cphase = cdf
        print("updated self.cphase: avg_time %f s\n" % avg_time)

        return

    def add_cphase_diag(self, avg_time=0, return_type='rec', vtype='vis', count='min', snrcut=0.,
                        err_type='predicted', num_samples=1000, round_s=0.1, uv_min=False):
        """Adds attribute self.cphase_diag: cphase_diag table averaged for dt

           Args:
               avg_time (float): closure phase averaging timescale
               return_type: data frame ('df') or recarray ('rec')
               vtype (str): Visibility type (e.g., 'vis', 'llvis', 'rrvis', etc.)
               count (str): If 'min', return minimal set of phases,
                            If 'max' return all closure phases up to reordering
               err_type (str): 'predicted' or 'measured'
               num_samples: number of bootstrap (re)samples if measuring error
               round_s (float): accuracy of datetime object in seconds
               snrcut (float): flag closure phases with snr lower than this

        """

        # Get spacing between datapoints in seconds
        if len(set([x[0] for x in list(self.unpack('time'))])) > 1:
            tint0 = np.min(np.diff(np.asarray(sorted(list(set(
                           [x[0] for x in list(self.unpack('time'))]))))))
            tint0 *= 3600
        else:
            tint0 = 0

        # Dom TODO: implement averaging during diagonal closure phase creation
        if avg_time > tint0:
            print("Averaging while creating diagonal closure phases is not yet implemented!")
            print("Proceeding for now without averaging.")
            cdf = ehdf.make_cphase_diag_df(self, vtype=vtype, round_s=round_s,
                                           count=count, snrcut=snrcut, uv_min=uv_min)
        else:
            cdf = ehdf.make_cphase_diag_df(self, vtype=vtype, round_s=round_s,
                                           count=count, snrcut=snrcut, uv_min=uv_min)
            if return_type == 'rec':
                cdf = ehdf.df_to_rec(cdf, 'cphase_diag')
            print("Updated self.cphase_diag: no averaging")

        self.cphase_diag = cdf
        print("updated self.cphase_diag: avg_time %f s\n" % avg_time)

        return

    def add_camp(self, avg_time=0, return_type='rec', ctype='camp',
                 count='max', debias=True, snrcut=0.,
                 err_type='predicted', num_samples=1000, round_s=0.1):
        """Adds attribute self.camp or self.logcamp: closure amplitudes table

           Args:
               avg_time (float): closure amplitude averaging timescale
               return_type: data frame ('df') or recarray ('rec')
               ctype (str): The closure amplitude type ('camp' or 'logcamp')
               debias (bool): If True, debias the closure amplitude
               count (str): If 'min', return minimal set of amplitudes,
                            if 'max' return all closure amplitudes up to inverses
               err_type (str): 'predicted' or 'measured'
               num_samples: number of bootstrap (re)samples if measuring error
               round_s (float): accuracy of datetime object in seconds
               snrcut (float): flag closure amplitudes with snr lower than this
        """

        # Get spacing between datapoints in seconds
        if len(set([x[0] for x in list(self.unpack('time'))])) > 1:
            tint0 = np.min(np.diff(np.asarray(sorted(list(set(
                           [x[0] for x in list(self.unpack('time'))]))))))
            tint0 *= 3600
        else:
            tint0 = 0

        if avg_time > tint0:
            foo = self.avg_incoherent(avg_time, debias=debias, err_type=err_type)
        else:
            foo = self
        cdf = ehdf.make_camp_df(foo, ctype=ctype, debias=False,
                                count=count, round_s=round_s, snrcut=snrcut)

        if ctype == 'logcamp':
            print("updated self.lcamp: no averaging")
        elif ctype == 'camp':
            print("updated self.camp: no averaging")
        if return_type == 'rec':
            cdf = ehdf.df_to_rec(cdf, 'camp')

        if ctype == 'logcamp':
            self.logcamp = cdf
            print("updated self.logcamp: avg_time %f s\n" % avg_time)
        elif ctype == 'camp':
            self.camp = cdf
            print("updated self.camp: avg_time %f s\n" % avg_time)

        return

    def add_logcamp(self, avg_time=0, return_type='rec', ctype='camp',
                    count='max', debias=True, snrcut=0.,
                    err_type='predicted', num_samples=1000, round_s=0.1):
        """Adds attribute self.logcamp: closure amplitudes table

           Args:
               avg_time (float): closure amplitude averaging timescale
               return_type: data frame ('df') or recarray ('rec')
               ctype (str): The closure amplitude type ('camp' or 'logcamp')
               debias (bool): If True, debias the closure amplitude
               count (str): If 'min', return minimal set of amplitudes,
                            if 'max' return all closure amplitudes up to inverses
               err_type (str): 'predicted' or 'measured'
               num_samples: number of bootstrap (re)samples if measuring error
               round_s (float): accuracy of datetime object in seconds
               snrcut (float): flag closure amplitudes with snr lower than this

        """

        self.add_camp(return_type=return_type, ctype='logcamp',
                      count=count, debias=debias, snrcut=snrcut,
                      avg_time=avg_time, err_type=err_type,
                      num_samples=num_samples, round_s=round_s)

        return

    def add_logcamp_diag(self, avg_time=0, return_type='rec', count='min', snrcut=0.,
                         debias=True, err_type='predicted', num_samples=1000, round_s=0.1):
        """Adds attribute self.logcamp_diag: logcamp_diag table averaged for dt

           Args:
               avg_time (float): diagonal log closure amplitude averaging timescale
               return_type: data frame ('df') or recarray ('rec')
               debias (bool): If True, debias the diagonal log closure amplitude
               count (str): If 'min', return minimal set of amplitudes,
                            If 'max' return all diagonal log closure amplitudes up to inverses
               err_type (str): 'predicted' or 'measured'
               num_samples: number of bootstrap (re)samples if measuring error
               round_s (float): accuracy of datetime object in seconds
               snrcut (float): flag diagonal log closure amplitudes with snr lower than this

        """

        # Get spacing between datapoints in seconds
        if len(set([x[0] for x in list(self.unpack('time'))])) > 1:
            tint0 = np.min(np.diff(np.asarray(sorted(list(set(
                           [x[0] for x in list(self.unpack('time'))]))))))
            tint0 *= 3600
        else:
            tint0 = 0

        if avg_time > tint0:
            foo = self.avg_incoherent(avg_time, debias=debias, err_type=err_type)
            cdf = ehdf.make_logcamp_diag_df(foo, debias='False', count=count,
                                            round_s=round_s, snrcut=snrcut)
        else:
            foo = self
            cdf = ehdf.make_logcamp_diag_df(foo, debias=debias, count=count,
                                            round_s=round_s, snrcut=snrcut)

        if return_type == 'rec':
            cdf = ehdf.df_to_rec(cdf, 'logcamp_diag')

        self.logcamp_diag = cdf
        print("updated self.logcamp_diag: avg_time %f s\n" % avg_time)

        return

    def add_all(self, avg_time=0, return_type='rec',
                count='max', debias=True, snrcut=0.,
                err_type='predicted', num_samples=1000, round_s=0.1):
        """Adds tables of all all averaged derived quantities
           self.amp,self.bispec,self.cphase,self.camp,self.logcamp

           Args:
               avg_time (float): closure amplitude averaging timescale
               return_type: data frame ('df') or recarray ('rec')
               debias (bool): If True, debias the closure amplitude
               count (str): If 'min', return minimal set of closure quantities,
                            if 'max' return all closure quantities
               err_type (str): 'predicted' or 'measured'
               num_samples: number of bootstrap (re)samples if measuring error
               round_s (float): accuracy of datetime object in seconds
               snrcut (float): flag closure amplitudes with snr lower than this

        """

        self.add_amp(return_type=return_type, avg_time=avg_time, debias=debias, err_type=err_type)
        self.add_bispec(return_type=return_type, count=count,
                        avg_time=avg_time, snrcut=snrcut, err_type=err_type,
                        num_samples=num_samples, round_s=round_s)
        self.add_cphase(return_type=return_type, count=count,
                        avg_time=avg_time, snrcut=snrcut, err_type=err_type,
                        num_samples=num_samples, round_s=round_s)
        self.add_cphase_diag(return_type=return_type, count='min',
                             avg_time=avg_time, snrcut=snrcut, err_type=err_type,
                             num_samples=num_samples, round_s=round_s)
        self.add_camp(return_type=return_type, ctype='camp',
                      count=count, debias=debias, snrcut=snrcut,
                      avg_time=avg_time, err_type=err_type,
                      num_samples=num_samples, round_s=round_s)
        self.add_camp(return_type=return_type, ctype='logcamp',
                      count=count, debias=debias, snrcut=snrcut,
                      avg_time=avg_time, err_type=err_type,
                      num_samples=num_samples, round_s=round_s)
        self.add_logcamp_diag(return_type=return_type, count='min',
                              debias=debias, avg_time=avg_time,
                              snrcut=snrcut, err_type=err_type,
                              num_samples=num_samples, round_s=round_s)

        return

    def add_scans(self, info='self', filepath='', dt=0.0165, margin=0.0001):
        """Compute scans and add self.scans to Obsdata object.

            Args:
                info (str): 'self' to infer from data, 'txt' for text file,
                            'vex' for vex schedule file
                filepath (str): path to txt/vex file with scans info
                dt (float): minimal time interval between scans in hours
                margin (float): padding scans by that time margin in hours

        """

        # infer scans directly from data
        if info == 'self':
            times_uni = np.asarray(sorted(list(set(self.data['time']))))
            scans = np.zeros_like(times_uni)
            scan_id = 0
            for cou in range(len(times_uni) - 1):
                scans[cou] = scan_id
                if (times_uni[cou + 1] - times_uni[cou] > dt):
                    scan_id += 1
            scans[-1] = scan_id
            scanlist = np.asarray([np.asarray([
                                   np.min(times_uni[scans == cou]) - margin,
                                   np.max(times_uni[scans == cou]) + margin])
                                   for cou in range(int(scans[-1]) + 1)])

        # read in scans from a text file
        elif info == 'txt':
            scanlist = np.loadtxt(filepath)

        # read in scans from a vex file
        elif info == 'vex':
            vex0 = ehtim.vex.Vex(filepath)
            t_min = [vex0.sched[x]['start_hr'] for x in range(len(vex0.sched))]
            duration = []
            for x in range(len(vex0.sched)):
                duration_foo = max([vex0.sched[x]['scan'][y]['scan_sec']
                                    for y in range(len(vex0.sched[x]['scan']))])
                duration.append(duration_foo)
            t_max = [tmin + dur / 3600. for (tmin, dur) in zip(t_min, duration)]
            scanlist = np.array([[tmin, tmax] for (tmin, tmax) in zip(t_min, t_max)])

        else:
            print("Parameter 'info' can only assume values 'self', 'txt' or 'vex'! ")
            scanlist = None

        self.scans = scanlist

        return

    def cleanbeam(self, npix, fov, pulse=ehc.PULSE_DEFAULT):
        """Make an image of the observation clean beam.

           Args:
               npix (int): The pixel size of the square output image.
               fov (float): The field of view of the square output image in radians.
               pulse (function): The function convolved with the pixel values for continuous image.

           Returns:
               (Image): an Image object with the clean beam.
        """

        im = ehtim.image.make_square(self, npix, fov, pulse=pulse)
        beamparams = self.fit_beam()
        im = im.add_gauss(1.0, beamparams)

        return im

    def fit_beam(self, weighting='uniform', units='rad'):
        """Fit a Gaussian to the dirty beam and return the parameters (fwhm_maj, fwhm_min, theta).

           Args:
               weighting (str): 'uniform' or 'natural'.
               units (string): 'rad' returns values in radians,
                               'natural' returns FWHMs in uas and theta in degrees

           Returns:
               (tuple): a tuple (fwhm_maj, fwhm_min, theta) of the dirty beam parameters in radians.
        """

        # Define the fit function that compares the quadratic expansion of the dirty image
        # with the quadratic expansion of an elliptical gaussian
        def fit_chisq(beamparams, db_coeff):

            (fwhm_maj2, fwhm_min2, theta) = beamparams
            a = 4 * np.log(2) * (np.cos(theta)**2 / fwhm_min2 + np.sin(theta)**2 / fwhm_maj2)
            b = 4 * np.log(2) * (np.cos(theta)**2 / fwhm_maj2 + np.sin(theta)**2 / fwhm_min2)
            c = 8 * np.log(2) * np.cos(theta) * np.sin(theta) * (1.0 / fwhm_maj2 - 1.0 / fwhm_min2)
            gauss_coeff = np.array((a, b, c))

            chisq = np.sum((np.array(db_coeff) - gauss_coeff)**2)

            return chisq

        # These are the coefficients (a,b,c) of a quadratic expansion of the dirty beam
        # For a point (x,y) in the image plane, the dirty beam expansion is 1-ax^2-by^2-cxy
        u = self.unpack('u')['u']
        v = self.unpack('v')['v']
        sigma = self.unpack('sigma')['sigma']

        weights = np.ones(u.shape)
        if weighting == 'natural':
            weights = 1. / sigma**2

        abc = np.array([np.sum(weights * u**2),
                        np.sum(weights * v**2),
                        2 * np.sum(weights * u * v)])
        abc *= (2. * np.pi**2 / np.sum(weights))
        abc *= 1e-20  # Decrease size of coefficients

        # Fit the beam
        guess = [(50)**2, (50)**2, 0.0]
        params = opt.minimize(fit_chisq, guess, args=(abc,), method='Powell')

        # Return parameters, adjusting fwhm_maj and fwhm_min if necessary
        if params.x[0] > params.x[1]:
            fwhm_maj = 1e-10 * np.sqrt(params.x[0])
            fwhm_min = 1e-10 * np.sqrt(params.x[1])
            theta = np.mod(params.x[2], np.pi)
        else:
            fwhm_maj = 1e-10 * np.sqrt(params.x[1])
            fwhm_min = 1e-10 * np.sqrt(params.x[0])
            theta = np.mod(params.x[2] + np.pi / 2.0, np.pi)

        gparams = np.array((fwhm_maj, fwhm_min, theta))

        if units == 'natural':
            gparams[0] /= ehc.RADPERUAS
            gparams[1] /= ehc.RADPERUAS
            gparams[2] *= 180. / np.pi

        return gparams

    def dirtybeam(self, npix, fov, pulse=ehc.PULSE_DEFAULT, weighting='uniform'):
        """Make an image of the observation dirty beam.

           Args:
               npix (int): The pixel size of the square output image.
               fov (float): The field of view of the square output image in radians.
               pulse (function): The function convolved with the pixel values for continuous image.
               weighting (str): 'uniform' or 'natural'
           Returns:
               (Image): an Image object with the dirty beam.
        """

        pdim = fov / npix
        sigma = self.unpack('sigma')['sigma']
        u = self.unpack('u')['u']
        v = self.unpack('v')['v']
        if weighting == 'natural':
            weights = 1. / sigma**2
        else:
            weights = np.ones(u.shape)

        xlist = np.arange(0, -npix, -1) * pdim + (pdim * npix) / 2.0 - pdim / 2.0

        # TODO -- use NFFT
        # TODO -- different beam weightings
        im = np.array([[np.mean(weights * np.cos(-2 * np.pi * (i * u + j * v)))
                        for i in xlist]
                        for j in xlist])

        im = im[0:npix, 0:npix]
        im = im / np.sum(im)  # Normalize to a total beam power of 1

        src = self.source + "_DB"
        outim = ehtim.image.Image(im, pdim, self.ra, self.dec,
                                  rf=self.rf, source=src, mjd=self.mjd, pulse=pulse)

        return outim

    def dirtyimage(self, npix, fov, pulse=ehc.PULSE_DEFAULT, weighting='uniform'):
        """Make the observation dirty image (direct Fourier transform).

           Args:
               npix (int): The pixel size of the square output image.
               fov (float): The field of view of the square output image in radians.
               pulse (function): The function convolved with the pixel values for continuous image.
               weighting (str): 'uniform' or 'natural'
           Returns:
               (Image): an Image object with dirty image.
        """

        pdim = fov / npix
        u = self.unpack('u')['u']
        v = self.unpack('v')['v']
        sigma = self.unpack('sigma')['sigma']
        xlist = np.arange(0, -npix, -1) * pdim + (pdim * npix) / 2.0 - pdim / 2.0
        if weighting == 'natural':
            weights = 1. / sigma**2
        else:
            weights = np.ones(u.shape)

        dim = np.array([[np.mean(weights * np.cos(-2 * np.pi * (i * u + j * v)))
                         for i in xlist]
                         for j in xlist])
        normfac = 1. / np.sum(dim)

        for label in ['vis1', 'vis2', 'vis3', 'vis4']:
            visname = self.poldict[label]

            vis = self.unpack(visname)[visname]

            # TODO -- use NFFT
            # TODO -- different beam weightings
            im = np.array([[np.mean(weights * (np.real(vis) * np.cos(-2 * np.pi * (i * u + j * v)) -
                                               np.imag(vis) * np.sin(-2 * np.pi * (i * u + j * v))))
                            for i in xlist]
                            for j in xlist])

            # Final normalization
            im = im * normfac
            im = im[0:npix, 0:npix]

            if label == 'vis1':
                out = ehtim.image.Image(im, pdim, self.ra, self.dec, polrep=self.polrep,
                                        rf=self.rf, source=self.source, mjd=self.mjd, pulse=pulse)
            else:
                pol = {ehc.vis_poldict[key]: key for key in ehc.vis_poldict.keys()}[visname]
                out.add_pol_image(im, pol)

        return out

    def rescale_zbl(self, totflux, uv_max, debias=True):
        """Rescale the short baselines to a new level of total flux.

           Args:
               totflux (float): new total flux to rescale to
               uv_max (float): maximum baseline length to rescale
               debias (bool): Debias amplitudes before computing original total flux from short bls

           Returns:
               (Obsdata): An Obsdata object with the inflated noise values.
        """

        # estimate the original total flux
        obs_zerobl = self.flag_uvdist(uv_max=uv_max)
        obs_zerobl.add_amp(debias=True)
        orig_totflux = np.sum(obs_zerobl.amp['amp'] * (1 / obs_zerobl.amp['sigma']**2))
        orig_totflux /= np.sum(1 / obs_zerobl.amp['sigma']**2)

        print('Rescaling zero baseline by ' + str(orig_totflux - totflux) + ' Jy' +
              ' to ' + str(totflux) + ' Jy')

        # Rescale short baselines to excise contributions from extended flux
        # Note: this does not do the proper thing for fractional polarization)
        obs = self.copy()
        for j in range(len(obs.data)):
            if (obs.data['u'][j]**2 + obs.data['v'][j]**2)**0.5 < uv_max:
                obs.data['vis'][j] *= totflux / orig_totflux
                obs.data['qvis'][j] *= totflux / orig_totflux
                obs.data['uvis'][j] *= totflux / orig_totflux
                obs.data['vvis'][j] *= totflux / orig_totflux
                obs.data['sigma'][j] *= totflux / orig_totflux
                obs.data['qsigma'][j] *= totflux / orig_totflux
                obs.data['usigma'][j] *= totflux / orig_totflux
                obs.data['vsigma'][j] *= totflux / orig_totflux

        return obs

    def add_leakage_noise(self, Dterm_amp=0.1, min_noise=0.01, debias=False):
        """Add estimated systematic noise from leakage at quadrature to thermal noise.
           Requires cross-hand visibilities.
           !! this operation is not currently tracked and should be applied with extreme caution!!

           Args:
               Dterm_amp (float): Estimated magnitude of leakage terms
               min_noise (float): Minimum fractional systematic noise to add
               debias (bool): Debias amplitudes before computing fractional noise

           Returns:
               (Obsdata): An Obsdata object with the inflated noise values.
        """

        # Extract visibility amplitudes
        # Switch to Stokes for graceful handling of circular basis products missing RR or LL
        amp = self.switch_polrep('stokes').unpack('amp', debias=debias)['amp']
        rlamp = np.nan_to_num(self.switch_polrep('circ').unpack('rlamp', debias=debias)['rlamp'])
        lramp = np.nan_to_num(self.switch_polrep('circ').unpack('lramp', debias=debias)['lramp'])

        frac_noise = (Dterm_amp * rlamp / amp)**2 + (Dterm_amp * lramp / amp)**2
        frac_noise = frac_noise * (frac_noise > min_noise) + min_noise * (frac_noise < min_noise)

        out = self.copy()
        for sigma in ['sigma1', 'sigma2', 'sigma3', 'sigma4']:
            try:
                field = self.poldict[sigma]
                out.data[field] = (self.data[field]**2 + np.abs(frac_noise * amp)**2)**0.5
            except KeyError:
                continue

        return out

    def add_fractional_noise(self, frac_noise, debias=False):
        """Add a constant fraction of amplitude at quadrature to thermal noise.
           Effectively imposes a maximal signal-to-noise ratio.
           !! this operation is not currently tracked and should be applied with extreme caution!!

           Args:
               frac_noise (float): The fraction of noise to add.
               debias (bool):      Whether or not to add frac_noise of debiased amplitudes.

           Returns:
               (Obsdata): An Obsdata object with the inflated noise values.
        """

        # Extract visibility amplitudes
        # Switch to Stokes for graceful handling of circular basis products missing RR or LL
        amp = self.switch_polrep('stokes').unpack('amp', debias=debias)['amp']

        out = self.copy()
        for sigma in ['sigma1', 'sigma2', 'sigma3', 'sigma4']:
            try:
                field = self.poldict[sigma]
                out.data[field] = (self.data[field]**2 + np.abs(frac_noise * amp)**2)**0.5
            except KeyError:
                continue

        return out

    def find_amt_fractional_noise(self, im, dtype='vis', target=1.0, debias=False,
                                  maxiter=200, ftol=1e-20, gtol=1e-20):
        """Returns the amount of fractional sys error necessary
           to make the image have a chisq close to the targeted value (1.0)
        """

        obs = self.copy()

        def objfunc(frac_noise):
            obs_tmp = obs.add_fractional_noise(frac_noise, debias=debias)
            chisq = obs_tmp.chisq(im, dtype=dtype)
            return np.abs(target - chisq)

        optdict = {'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
        res = opt.minimize(objfunc, 0.0, method='L-BFGS-B', options=optdict)

        return res.x

    def rescale_noise(self, noise_rescale_factor=1.0):
        """Rescale the thermal noise on all Stokes parameters by a constant factor.
           This is useful for AIPS data, which has a missing factor relating 'weights' to noise.

           Args:
               noise_rescale_factor (float): The number to multiple the existing sigmas by.

           Returns:
               (Obsdata): An Obsdata object with the rescaled noise values.
        """

        datatable = self.data.copy()

        for d in datatable:
            d[-4] = d[-4] * noise_rescale_factor
            d[-3] = d[-3] * noise_rescale_factor
            d[-2] = d[-2] * noise_rescale_factor
            d[-1] = d[-1] * noise_rescale_factor

        arglist, argdict = self.obsdata_args()
        arglist[DATPOS] = np.array(datatable)
        out = Obsdata(*arglist, **argdict)

        return out

    def estimate_noise_rescale_factor(self, max_diff_sec=0.0, min_num=10, median_snr_cut=0,
                                      count='max', vtype='vis', print_std=False):
        """Estimate a constant noise rescaling factor on all baselines, times, and polarizations.
           Uses pairwise differences of closure phases relative to the expected scatter.
           This is useful for AIPS data, which has a missing factor relating 'weights' to noise.

           Args:
               max_diff_sec (float): The maximum difference of adjacent closure phases (in seconds)
                                     If 0, auto-estimates to twice the median scan length.
               min_num (int): The minimum number of closure phase differences for a triangle
                              to be included in the set of estimators.
               median_snr_cut (float): Do not include a triangle if its median SNR is below this
               count (str): If 'min', use minimal set of phases,
                            if 'max' use all closure phases up to reordering
               vtype (str): Visibility type (e.g., 'vis', 'llvis', 'rrvis', etc.)
               print_std (bool): Whether or not to print the std dev. for each closure triangle.

           Returns:
               (float): The rescaling factor.
        """

        if max_diff_sec == 0.0:
            max_diff_sec = 5 * np.median(self.unpack('tint')['tint'])
            print("estimated max_diff_sec: ", max_diff_sec)

        # Now check the noise statistics on all closure phase triangles
        c_phases = self.c_phases(vtype=vtype, mode='time', count=count, ang_unit='')

        # First, just determine the set of closure phase triangles
        all_triangles = []
        for scan in c_phases:
            for cphase in scan:
                all_triangles.append((cphase[1], cphase[2], cphase[3]))
        std_list = []
        print("Estimating noise rescaling factor from %d triangles...\n" % len(set(all_triangles)))

        # Now determine the differences of adjacent samples on each triangle,
        # relative to the expected thermal noise
        i_count = 0
        for tri in set(all_triangles):
            i_count = i_count + 1
            if print_std:
                sys.stdout.write('\rGetting noise for triangles %i/%i ' %
                                 (i_count, len(set(all_triangles))))
                sys.stdout.flush()
            all_tri = np.array([[]])
            for scan in c_phases:
                for cphase in scan:
                    if (cphase[1] == tri[0] and cphase[2] == tri[1] and cphase[3] == tri[2] and
                            not np.isnan(cphase[-2]) and not np.isnan(cphase[-2])):

                        all_tri = np.append(all_tri, ((cphase[0], cphase[-2], cphase[-1])))

            all_tri = all_tri.reshape(int(len(all_tri) / 3), 3)

            # See whether the triangle has sufficient SNR
            if np.median(np.abs(all_tri[:, 1] / all_tri[:, 2])) < median_snr_cut:
                if print_std:
                    print(tri, 'median snr too low (%6.4f)' %
                          np.median(np.abs(all_tri[:, 1] / all_tri[:, 2])))
                continue

            # Now go through and find studentized differences of adjacent points
            s_list = np.array([])
            for j in range(len(all_tri) - 1):
                if (all_tri[j + 1, 0] - all_tri[j, 0]) * 3600.0 < max_diff_sec:
                    diff = (all_tri[j + 1, 1] - all_tri[j, 1]) % (2.0 * np.pi)
                    if diff > np.pi:
                        diff -= 2.0 * np.pi
                    s_list = np.append(
                        s_list, diff / (all_tri[j, 2]**2 + all_tri[j + 1, 2]**2)**0.5)

            if len(s_list) > min_num:
                std_list.append(np.std(s_list))
                if print_std:
                    print(tri, '%6.4f [%d differences]' % (np.std(s_list), len(s_list)))
            else:
                if print_std and len(all_tri) > 0:
                    print(tri, '%d cphases found [%d differences < min_num = %d]' %
                          (len(all_tri), len(s_list), min_num))

        if len(std_list) == 0:
            print("No suitable closure phase differences! Try using a larger max_diff_sec.")
            median = 1.0
        else:
            median = np.median(std_list)

        return median

    def flag_elev(self, elev_min=0.0, elev_max=90, output='kept'):
        """Flag visibilities for which either station is outside a stated elevation range

           Args:
               elev_min (float): Minimum elevation (deg)
               elev_max (float): Maximum elevation (deg)
               output (str): returns 'kept', 'flagged', or 'both' (a dictionary)

           Returns:
               (Obsdata): a observation object with flagged data points removed
        """

        el_pairs = self.unpack(['el1', 'el2'])
        mask = (np.min((el_pairs['el1'], el_pairs['el2']), axis=0) > elev_min)
        mask *= (np.max((el_pairs['el1'], el_pairs['el2']), axis=0) < elev_max)

        datatable_kept = self.data.copy()
        datatable_flagged = self.data.copy()

        datatable_kept = datatable_kept[mask]
        datatable_flagged = datatable_flagged[np.invert(mask)]
        print('Flagged %d/%d visibilities' % (len(datatable_flagged), len(self.data)))

        obs_kept = self.copy()
        obs_flagged = self.copy()
        obs_kept.data = datatable_kept
        obs_flagged.data = datatable_flagged

        if output == 'flagged':
            return obs_flagged
        elif output == 'both':
            return {'kept': obs_kept, 'flagged': obs_flagged}
        else:
            return obs_kept

    def flag_large_fractional_pol(self, max_fractional_pol=1.0, output='kept'):
        """Flag visibilities for which the fractional polarization is above a specified threshold

           Args:
               max_fractional_pol (float): Maximum fractional polarization
               output (str): returns 'kept', 'flagged', or 'both' (a dictionary)

           Returns:
               (Obsdata): a observation object with flagged data points removed
        """

        m = np.nan_to_num(self.unpack(['mamp'])['mamp'])
        mask = m < max_fractional_pol

        datatable_kept = self.data.copy()
        datatable_flagged = self.data.copy()

        datatable_kept = datatable_kept[mask]
        datatable_flagged = datatable_flagged[np.invert(mask)]
        print('Flagged %d/%d visibilities' % (len(datatable_flagged), len(self.data)))

        obs_kept = self.copy()
        obs_flagged = self.copy()
        obs_kept.data = datatable_kept
        obs_flagged.data = datatable_flagged

        if output == 'flagged':
            return obs_flagged
        elif output == 'both':
            return {'kept': obs_kept, 'flagged': obs_flagged}
        else:
            return obs_kept

    def flag_uvdist(self, uv_min=0.0, uv_max=1e12, output='kept'):
        """Flag data points outside a given uv range

           Args:
               uv_min (float): remove points with uvdist less than  this
               uv_max (float): remove points with uvdist greater than  this
               output (str): returns 'kept', 'flagged', or 'both' (a dictionary)

           Returns:
               (Obsdata): a observation object with flagged data points removed
        """

        uvdist_list = self.unpack('uvdist')['uvdist']
        mask = np.array([uv_min <= uvdist_list[j] <= uv_max for j in range(len(uvdist_list))])
        datatable_kept = self.data.copy()
        datatable_flagged = self.data.copy()

        datatable_kept = datatable_kept[mask]
        datatable_flagged = datatable_flagged[np.invert(mask)]
        print('U-V flagged %d/%d visibilities' % (len(datatable_flagged), len(self.data)))

        obs_kept = self.copy()
        obs_flagged = self.copy()
        obs_kept.data = datatable_kept
        obs_flagged.data = datatable_flagged

        if output == 'flagged':
            return obs_flagged
        elif output == 'both':
            return {'kept': obs_kept, 'flagged': obs_flagged}
        else:
            return obs_kept

    def flag_sites(self, sites, output='kept'):
        """Flag data points that include the specified sites

           Args:
               sites (list): list of sites to remove from the data
               output (str): returns 'kept', 'flagged', or 'both' (a dictionary)

           Returns:
               (Obsdata): a observation object with flagged data points removed
        """

        # This will remove all visibilities that include any of the specified sites

        t1_list = self.unpack('t1')['t1']
        t2_list = self.unpack('t2')['t2']
        mask = np.array([t1_list[j] not in sites and t2_list[j] not in sites
                         for j in range(len(t1_list))])

        datatable_kept = self.data.copy()
        datatable_flagged = self.data.copy()

        datatable_kept = datatable_kept[mask]
        datatable_flagged = datatable_flagged[np.invert(mask)]
        print('Flagged %d/%d visibilities' % (len(datatable_flagged), len(self.data)))

        obs_kept = self.copy()
        obs_flagged = self.copy()
        obs_kept.data = datatable_kept
        obs_flagged.data = datatable_flagged

        if output == 'flagged':  # return only the points flagged as anomalous
            return obs_flagged
        elif output == 'both':
            return {'kept': obs_kept, 'flagged': obs_flagged}
        else:
            return obs_kept

    def flag_bl(self, sites, output='kept'):
        """Flag data points that include the specified baseline

           Args:
               sites (list): baseline to remove from the data
               output (str): returns 'kept', 'flagged', or 'both' (a dictionary)

           Returns:
               (Obsdata): a observation object with flagged data points removed
        """

        # This will remove all visibilities that include any of the specified baseline
        obs_out = self.copy()
        t1_list = obs_out.unpack('t1')['t1']
        t2_list = obs_out.unpack('t2')['t2']
        mask = np.array([not(t1_list[j] in sites and t2_list[j] in sites)
                         for j in range(len(t1_list))])

        datatable_kept = self.data.copy()
        datatable_flagged = self.data.copy()

        datatable_kept = datatable_kept[mask]
        datatable_flagged = datatable_flagged[np.invert(mask)]
        print('Flagged %d/%d visibilities' % (len(datatable_flagged), len(self.data)))

        obs_kept = self.copy()
        obs_flagged = self.copy()
        obs_kept.data = datatable_kept
        obs_flagged.data = datatable_flagged

        if output == 'flagged':  # return only the points flagged as anomalous
            return obs_flagged
        elif output == 'both':
            return {'kept': obs_kept, 'flagged': obs_flagged}
        else:
            return obs_kept

    def flag_low_snr(self, snr_cut=3, output='kept'):
        """Flag low snr data points

           Args:
               snr_cut (float): remove points with snr lower than  this
               output (str): returns 'kept', 'flagged', or 'both' (a dictionary)

           Returns:
               (Obsdata): a observation object with flagged data points removed
        """

        mask = self.unpack('snr')['snr'] > snr_cut

        datatable_kept = self.data.copy()
        datatable_flagged = self.data.copy()

        datatable_kept = datatable_kept[mask]
        datatable_flagged = datatable_flagged[np.invert(mask)]
        print('snr flagged %d/%d visibilities' % (len(datatable_flagged), len(self.data)))

        obs_kept = self.copy()
        obs_flagged = self.copy()
        obs_kept.data = datatable_kept
        obs_flagged.data = datatable_flagged

        if output == 'flagged':  # return only the points flagged as anomalous
            return obs_flagged
        elif output == 'both':
            return {'kept': obs_kept, 'flagged': obs_flagged}
        else:
            return obs_kept

    def flag_high_sigma(self, sigma_cut=.005, sigma_type='sigma', output='kept'):
        """Flag high sigma (thermal noise on Stoke I) data points

           Args:
               sigma_cut (float): remove points with sigma higher than  this
               sigma_type (str): sigma type (sigma, rrsigma, llsigma, etc.)
               output (str): returns 'kept', 'flagged', or 'both' (a dictionary)

           Returns:
               (Obsdata): a observation object with flagged data points removed
        """

        mask = self.unpack(sigma_type)[sigma_type] < sigma_cut

        datatable_kept = self.data.copy()
        datatable_flagged = self.data.copy()

        datatable_kept = datatable_kept[mask]
        datatable_flagged = datatable_flagged[np.invert(mask)]
        print('sigma flagged %d/%d visibilities' % (len(datatable_flagged), len(self.data)))

        obs_kept = self.copy()
        obs_flagged = self.copy()
        obs_kept.data = datatable_kept
        obs_flagged.data = datatable_flagged

        if output == 'flagged':  # return only the points flagged as anomalous
            return obs_flagged
        elif output == 'both':
            return {'kept': obs_kept, 'flagged': obs_flagged}
        else:
            return obs_kept

    def flag_UT_range(self, UT_start_hour=0., UT_stop_hour=0.,
                      flag_type='all', flag_what='', output='kept'):
        """Flag data points within a certain UT range

           Args:
               UT_start_hour (float): start of  time window
               UT_stop_hour (float): end of time window
               flag_type (str): 'all', 'baseline', or  'station'
               flag_what (str): baseline or station to flag
               output (str): returns 'kept', 'flagged', or 'both' (a dictionary)

           Returns:
               (Obsdata): a observation object with flagged data points removed
        """

        # This drops (or only keeps) points within a specified UT range

        UT_mask = self.unpack('time')['time'] <= UT_start_hour
        UT_mask = UT_mask + (self.unpack('time')['time'] >= UT_stop_hour)
        if flag_type != 'all':
            t1_list = self.unpack('t1')['t1']
            t2_list = self.unpack('t2')['t2']
            if flag_type == 'station':
                station = flag_what
                what_mask = np.array([not (t1_list[j] == station or t2_list[j] == station)
                                      for j in range(len(t1_list))])
            elif flag_type == 'baseline':
                station1 = flag_what.split('-')[0]
                station2 = flag_what.split('-')[1]
                stations = [station1, station2]
                what_mask = np.array([not ((t1_list[j] in stations) and (t2_list[j] in stations))
                                      for j in range(len(t1_list))])
        else:
            what_mask = np.array([False for j in range(len(UT_mask))])
        mask = UT_mask | what_mask

        datatable_kept = self.data.copy()
        datatable_flagged = self.data.copy()

        datatable_kept = datatable_kept[mask]
        datatable_flagged = datatable_flagged[np.invert(mask)]
        print('time flagged %d/%d visibilities' % (len(datatable_flagged), len(self.data)))

        obs_kept = self.copy()
        obs_flagged = self.copy()
        obs_kept.data = datatable_kept
        obs_flagged.data = datatable_flagged

        if output == 'flagged':  # return only the points flagged as anomalous
            return obs_flagged
        elif output == 'both':
            return {'kept': obs_kept, 'flagged': obs_flagged}
        else:
            return obs_kept

    def flags_from_file(self, flagfile, flag_type='station'):
        """Flagging data based on csv file

            Args:
                flagfile (str): path to csv file with mjds of flagging start / stop time,
                                and optionally baseline / stations
               flag_type (str): 'all', 'baseline', or  'station'

            Returns:
                (Obsdata): a observation object with flagged data points removed
        """

        df = pd.read_csv(flagfile)
        mjd_start = list(df['mjd_start'])
        mjd_stop = list(df['mjd_stop'])
        if flag_type == 'station':
            whatL = list(df['station'])
        elif flag_type == 'baseline':
            whatL = list(df['baseline'])
        elif flag_type == 'all':
            whatL = ['' for cou in range(len(mjd_start))]
        obs = self.copy()
        for cou in range(len(mjd_start)):
            what = whatL[cou]
            starth = (mjd_start[cou] % 1) * 24.
            stoph = (mjd_stop[cou] % 1) * 24.
            obs = obs.flag_UT_range(UT_start_hour=starth, UT_stop_hour=stoph,
                                    flag_type=flag_type, flag_what=what, output='kept')

        return obs

    def flag_anomalous(self, field='snr', max_diff_seconds=100, robust_nsigma_cut=5, output='kept'):
        """Flag anomalous data points

           Args:
               field (str): The quantity to test for
               max_diff_seconds (float): The moving window size for testing outliers
               robust_nsigma_cut (float): Outliers further than this from the mean are removed
               output (str): returns 'kept', 'flagged', or 'both' (a dictionary)

           Returns:
               (Obsdata): a observation object with flagged data points removed
        """

        stats = dict()
        for t1 in set(self.data['t1']):
            for t2 in set(self.data['t2']):
                vals = self.unpack_bl(t1, t2, field)
                if len(vals) > 0:
                    # nans will all be dropped, which can be problematic for polarimetric values
                    vals[field] = np.nan_to_num(vals[field])
                for j in range(len(vals)):
                    near_vals_mask = np.abs(vals['time'] - vals['time']
                                            [j]) < max_diff_seconds / 3600.0
                    fields = vals[field][near_vals_mask]

                    # Here, we use median absolute deviation as a robust proxy for standard
                    # deviation
                    dfields = np.median(np.abs(fields - np.median(fields)))
                    # Avoid problems when the MAD is zero (e.g., a single sample)
                    if dfields == 0.0:
                        dfields = 1.0
                    stat = np.abs(vals[field][j] - np.median(fields)) / dfields
                    stats[(vals['time'][j][0], tuple(sorted((t1, t2))))] = stat

        mask = np.array([stats[(rec[0], tuple(sorted((rec[2], rec[3]))))][0] < robust_nsigma_cut
                         for rec in self.data])

        datatable_kept = self.data.copy()
        datatable_flagged = self.data.copy()

        datatable_kept = datatable_kept[mask]
        datatable_flagged = datatable_flagged[np.invert(mask)]
        print('anomalous %s flagged %d/%d visibilities' %
              (field, len(datatable_flagged), len(self.data)))

        # Make new observations with all data first to avoid problems with empty arrays
        obs_kept = self.copy()
        obs_flagged = self.copy()
        obs_kept.data = datatable_kept
        obs_flagged.data = datatable_flagged

        if output == 'flagged':  # return only the points flagged as anomalous
            return obs_flagged
        elif output == 'both':
            return {'kept': obs_kept, 'flagged': obs_flagged}
        else:
            return obs_kept

    def filter_subscan_dropouts(self, perc=0, return_type='rec'):
        """Filtration to drop data and ensure that we only average parts with same timestamp.
           Potentially this could reduce risk of non-closing errors.

           Args:
               perc (float): drop baseline from scan if it has less than this fraction
                              of median baseline observation time during the scan
               return_type (str): data frame ('df') or recarray ('rec')

           Returns:
               (Obsdata): a observation object with flagged data points removed
        """

        if not isinstance(self.scans, np.ndarray):
            print('List of scans in ndarray format required! Add it with add_scans')

        else:
            # make df and add scan_id to data
            df = ehdf.make_df(self)
            tot_points = np.shape(df)[0]
            bins, labs = ehdf.get_bins_labels(self.scans)
            df['scan_id'] = list(pd.cut(df.time, bins, labels=labs))

            # first flag baselines that are working for short part of scan
            df['count_samples'] = 1
            hm1 = df.groupby(['scan_id', 'baseline', 'polarization'])
            hm1 = hm1.agg({'count_samples': np.sum}).reset_index()
            hm1['count_baselines_before'] = 1
            hm2 = hm1.groupby(['scan_id', 'polarization'])
            hm2 = hm2.agg({'count_samples': lambda x: perc * np.median(x),
                           'count_baselines_before': np.sum}).reset_index()

            # dictionary with minimum acceptable number of samples per scan
            dict_elem_in_scan = dict(zip(hm2.scan_id, hm2.count_samples))

            # list of acceptable scans and baselines
            hm1 = hm1[list(map(lambda x: x[1] >= dict_elem_in_scan[x[0]],
                               list(zip(hm1.scan_id, hm1.count_samples))))]
            list_good_scans_baselines = list(zip(hm1.scan_id, hm1.baseline))

            # filter out data
            df_filtered = df[list(map(lambda x: x in list_good_scans_baselines,
                                      list(zip(df.scan_id, df.baseline))))]

            # how many baselines present during scan?
            df_filtered['count_samples'] = 1
            hm3 = df_filtered.groupby(['scan_id', 'baseline', 'polarization'])
            hm3 = hm3.agg({'count_samples': np.sum}).reset_index()
            hm3['count_baselines_after'] = 1
            hm4 = hm3.groupby(['scan_id', 'polarization'])
            hm4 = hm4.agg({'count_baselines_after': np.sum}).reset_index()
            dict_how_many_baselines = dict(zip(hm4.scan_id, hm4.count_baselines_after))

            # how many baselines present during each time?
            df_filtered['count_baselines_per_time'] = 1
            hm5 = df_filtered.groupby(['datetime', 'scan_id', 'polarization'])
            hm5 = hm5.agg({'count_baselines_per_time': np.sum}).reset_index()
            dict_datetime_num_baselines = dict(zip(hm5.datetime, hm5.count_baselines_per_time))

            # only keep times when all baselines available
            df_filtered2 = df_filtered[list(map(lambda x: dict_datetime_num_baselines[x[1]] == dict_how_many_baselines[x[0]], list(
                zip(df_filtered.scan_id, df_filtered.datetime))))]

            remaining_points = np.shape(df_filtered2)[0]
            print('Flagged out {} of {} datapoints'.format(
                tot_points - remaining_points, tot_points))
            if return_type == 'rec':
                out_vis = ehdf.df_to_rec(df_filtered2, 'vis')

            arglist, argdict = self.obsdata_args()
            arglist[DATPOS] = out_vis
            out = Obsdata(*arglist, **argdict)

            return out

    def reverse_taper(self, fwhm):
        """Reverse taper the observation with a circular Gaussian kernel

           Args:
               fwhm (float): real space fwhm size of convolution kernel in radian

           Returns:
               (Obsdata): a new reverse-tapered observation object
        """
        datatable = self.data.copy()
        vis1 = datatable[self.poldict['vis1']]
        vis2 = datatable[self.poldict['vis2']]
        vis3 = datatable[self.poldict['vis3']]
        vis4 = datatable[self.poldict['vis4']]
        sigma1 = datatable[self.poldict['sigma1']]
        sigma2 = datatable[self.poldict['sigma2']]
        sigma3 = datatable[self.poldict['sigma3']]
        sigma4 = datatable[self.poldict['sigma4']]
        u = datatable['u']
        v = datatable['v']

        fwhm_sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        ker = np.exp(-2 * np.pi**2 * fwhm_sigma**2 * (u**2 + v**2))

        datatable[self.poldict['vis1']] = vis1 / ker
        datatable[self.poldict['vis2']] = vis2 / ker
        datatable[self.poldict['vis3']] = vis3 / ker
        datatable[self.poldict['vis4']] = vis4 / ker
        datatable[self.poldict['sigma1']] = sigma1 / ker
        datatable[self.poldict['sigma2']] = sigma2 / ker
        datatable[self.poldict['sigma3']] = sigma3 / ker
        datatable[self.poldict['sigma4']] = sigma4 / ker

        arglist, argdict = self.obsdata_args()
        arglist[DATPOS] = datatable
        obstaper = Obsdata(*arglist, **argdict)

        return obstaper

    def taper(self, fwhm):
        """Taper the observation with a circular Gaussian kernel

           Args:
               fwhm (float): real space fwhm size of convolution kernel in radian

           Returns:
               (Obsdata): a new tapered observation object
        """
        datatable = self.data.copy()

        vis1 = datatable[self.poldict['vis1']]
        vis2 = datatable[self.poldict['vis2']]
        vis3 = datatable[self.poldict['vis3']]
        vis4 = datatable[self.poldict['vis4']]
        sigma1 = datatable[self.poldict['sigma1']]
        sigma2 = datatable[self.poldict['sigma2']]
        sigma3 = datatable[self.poldict['sigma3']]
        sigma4 = datatable[self.poldict['sigma4']]
        u = datatable['u']
        v = datatable['v']

        fwhm_sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        ker = np.exp(-2 * np.pi**2 * fwhm_sigma**2 * (u**2 + v**2))

        datatable[self.poldict['vis1']] = vis1 * ker
        datatable[self.poldict['vis2']] = vis2 * ker
        datatable[self.poldict['vis3']] = vis3 * ker
        datatable[self.poldict['vis4']] = vis4 * ker
        datatable[self.poldict['sigma1']] = sigma1 * ker
        datatable[self.poldict['sigma2']] = sigma2 * ker
        datatable[self.poldict['sigma3']] = sigma3 * ker
        datatable[self.poldict['sigma4']] = sigma4 * ker

        arglist, argdict = self.obsdata_args()
        arglist[DATPOS] = datatable
        obstaper = Obsdata(*arglist, **argdict)

        return obstaper

    def deblur(self):
        """Deblur the observation obs by dividing by the Sgr A* scattering kernel.

           Args:

           Returns:
               (Obsdata): a new deblurred observation object.
        """

        # make a copy of observation data
        datatable = self.data.copy()

        vis1 = datatable[self.poldict['vis1']]
        vis2 = datatable[self.poldict['vis2']]
        vis3 = datatable[self.poldict['vis3']]
        vis4 = datatable[self.poldict['vis4']]
        sigma1 = datatable[self.poldict['sigma1']]
        sigma2 = datatable[self.poldict['sigma2']]
        sigma3 = datatable[self.poldict['sigma3']]
        sigma4 = datatable[self.poldict['sigma4']]
        u = datatable['u']
        v = datatable['v']

        # divide visibilities by the scattering kernel
        for i in range(len(vis1)):
            ker = obsh.sgra_kernel_uv(self.rf, u[i], v[i])
            vis1[i] = vis1[i] / ker
            vis2[i] = vis2[i] / ker
            vis2[i] = vis3[i] / ker
            vis4[i] = vis4[i] / ker
            sigma1[i] = sigma1[i] / ker
            sigma2[i] = sigma2[i] / ker
            sigma3[i] = sigma3[i] / ker
            sigma4[i] = sigma4[i] / ker

        datatable[self.poldict['vis1']] = vis1
        datatable[self.poldict['vis2']] = vis2
        datatable[self.poldict['vis3']] = vis3
        datatable[self.poldict['vis4']] = vis4
        datatable[self.poldict['sigma1']] = sigma1
        datatable[self.poldict['sigma2']] = sigma2
        datatable[self.poldict['sigma3']] = sigma3
        datatable[self.poldict['sigma4']] = sigma4

        arglist, argdict = self.obsdata_args()
        arglist[DATPOS] = datatable
        obsdeblur = Obsdata(*arglist, **argdict)

        return obsdeblur

    def reweight(self, uv_radius, weightdist=1.0):
        """Reweight the sigmas based on the local  density of uv points

           Args:
               uv_radius (float): radius in uv-plane to look for nearby points
               weightdist (float): ??

           Returns:
               (Obsdata): a new reweighted observation object.
        """

        obs_new = self.copy()
        npts = len(obs_new.data)

        uvpoints = np.vstack((obs_new.data['u'], obs_new.data['v'])).transpose()
        uvpoints_tree1 = spatial.cKDTree(uvpoints)
        uvpoints_tree2 = spatial.cKDTree(-uvpoints)

        for i in range(npts):
            matches1 = uvpoints_tree1.query_ball_point(uvpoints[i, :], uv_radius)
            matches2 = uvpoints_tree2.query_ball_point(uvpoints[i, :], uv_radius)
            nmatches = len(matches1) + len(matches2)

            for sigma in ['sigma', 'qsigma', 'usigma', 'vsigma']:
                obs_new.data[sigma][i] = np.sqrt(nmatches)

        scale = np.mean(self.data['sigma']) / np.mean(obs_new.data['sigma'])
        for sigma in ['sigma', 'qsigma', 'usigma', 'vsigma']:
            obs_new.data[sigma] *= scale * weightdist

        if weightdist < 1.0:
            for i in range(npts):
                for sigma in ['sigma', 'qsigma', 'usigma', 'vsigma']:
                    obs_new.data[sigma][i] += (1 - weightdist) * self.data[sigma][i]

        return obs_new

    def fit_gauss(self, flux=1.0, fittype='amp', paramguess=(
            100 * ehc.RADPERUAS, 100 * ehc.RADPERUAS, 0.)):
        """Fit a gaussian to either Stokes I complex visibilities or Stokes I visibility amplitudes.

           Args:
                flux (float): total flux in the fitted gaussian
                fitttype (str): "amp" to fit to visibilty amplitudes
                paramguess (tuble): initial guess of fit Gaussian (fwhm_maj, fwhm_min, theta)

           Returns:
                (tuple) : (fwhm_maj, fwhm_min, theta) of the fit Gaussian parameters in radians.
        """

        # TODO this fit doesn't work very well!!
        vis = self.data['vis']
        u = self.data['u']
        v = self.data['v']
        sig = self.data['sigma']

        # error function
        if fittype == 'amp':
            def errfunc(p):
                vismodel = obsh.gauss_uv(u, v, flux, p, x=0., y=0.)
                err = np.sum((np.abs(vis) - np.abs(vismodel))**2 / sig**2)
                return err
        else:
            def errfunc(p):
                vismodel = obsh.gauss_uv(u, v, flux, p, x=0., y=0.)
                err = np.sum(np.abs(vis - vismodel)**2 / sig**2)
                return err

        optdict = {'maxiter': 5000}  # minimizer params
        res = opt.minimize(errfunc, paramguess, method='Powell', options=optdict)
        gparams = res.x

        return gparams

    def bispectra(self, vtype='vis', mode='all', count='min',
                  timetype=False, uv_min=False, snrcut=0.):
        """Return a recarray of the equal time bispectra.

           Args:
               vtype (str): The visibilty type from which to assemble bispectra
                            ('vis', 'qvis', 'uvis','vvis','rrvis','lrvis','rlvis','llvis')
               mode (str): If 'time', return phases in a list of equal time arrays,
                           if 'all', return all phases in a single array
               count (str): If 'min', return minimal set of bispectra,
                            if 'max' return all bispectra up to reordering
               timetype (str): 'GMST' or 'UTC'
               uv_min (float): flag baselines shorter than this before forming closure quantities
               snrcut (float): flag bispectra with snr lower than this

           Returns:
               (numpy.recarry): A recarray of the bispectra values with datatype DTBIS
        """

        if timetype is False:
            timetype = self.timetype
        if mode not in ('time', 'all'):
            raise Exception("possible options for mode are 'time' and 'all'")
        if count not in ('max', 'min', 'min-cut0bl'):
            raise Exception("possible options for count are 'max', 'min', or 'min-cut0bl'")
        if vtype not in ('vis', 'qvis', 'uvis', 'vvis', 'rrvis', 'lrvis', 'rlvis', 'llvis'):
            raise Exception("possible options for vtype are" +
                            " 'vis', 'qvis', 'uvis','vvis','rrvis','lrvis','rlvis','llvis'")
        if timetype not in ['GMST', 'UTC', 'gmst', 'utc']:
            raise Exception("timetype should be 'GMST' or 'UTC'!")

        # Flag zero baselines
        obsdata = self.copy()
        if uv_min:
            obsdata = obsdata.flag_uvdist(uv_min=uv_min)
            # get which sites were flagged
            obsdata_flagged = self.copy()
            obsdata_flagged = obsdata_flagged.flag_uvdist(uv_max=uv_min)

        # Generate the time-sorted data with conjugate baselines
        tlist = obsdata.tlist(conj=True)
        out = []
        bis = []
        tt = 1
        for tdata in tlist:

            # sys.stdout.write('\rGetting bispectra:: type %s, count %s, scan %i/%i ' %
            #                 (vtype, count, tt, len(tlist)))
            # sys.stdout.flush()

            tt += 1

            time = tdata[0]['time']
            if timetype in ['GMST', 'gmst'] and self.timetype == 'UTC':
                time = obsh.utc_to_gmst(time, self.mjd)
            if timetype in ['UTC', 'utc'] and self.timetype == 'GMST':
                time = obsh.gmst_to_utc(time, self.mjd)
            sites = list(set(np.hstack((tdata['t1'], tdata['t2']))))

            # Create a dictionary of baselines at the current time incl. conjugates;
            l_dict = {}
            for dat in tdata:
                l_dict[(dat['t1'], dat['t2'])] = dat

            # Determine the triangles in the time step
            # Minimal Set
            if count == 'min':
                tris = obsh.tri_minimal_set(sites, self.tarr, self.tkey)

            # Maximal  Set
            elif count == 'max':
                tris = np.sort(list(it.combinations(sites, 3)))

            elif count == 'min-cut0bl':
                tris = obsh.tri_minimal_set(sites, self.tarr, self.tkey)

                # if you cut the 0 baselines, add in triangles that now are not in the minimal set
                if uv_min:
                    # get the reference site
                    sites_ordered = [x for x in self.tarr['site'] if x in sites]
                    ref = sites_ordered[0]

                    # check if the reference site was in a zero baseline
                    zerobls = np.vstack([obsdata_flagged.data['t1'], obsdata_flagged.data['t2']])
                    if np.sum(zerobls == ref):

                        # determine which sites were cut out of the minimal set
                        cutsites = np.unique(np.hstack([zerobls[1][zerobls[0] == ref],
                                                        zerobls[0][zerobls[1] == ref]]))

                        # we can only handle if there was 1 connecting site that was cut
                        if len(cutsites) > 1:
                            raise Exception("Cannot have the root node be in a clique" +
                                            "with more than 2 sites sharing 0 baselines'")

                        # get the remaining sites
                        cutsite = cutsites[0]
                        sites_remaining = np.array(sites_ordered)[np.array(sites_ordered) != ref]
                        sites_remaining = sites_remaining[np.array(sites_remaining) != cutsite]
                        # get the next site in the list, ideally sorted by snr
                        second_ref = sites_remaining[0]

                        # add in additional triangles
                        for s2 in range(1, len(sites_remaining)):
                            tris.append((cutsite, second_ref, sites_remaining[s2]))

            # Generate bispectra for each triangle
            for tri in tris:

                # Select triangle entries in the data dictionary
                try:
                    l1 = l_dict[(tri[0], tri[1])]
                    l2 = l_dict[(tri[1], tri[2])]
                    l3 = l_dict[(tri[2], tri[0])]
                except KeyError:
                    continue

                (bi, bisig) = obsh.make_bispectrum(l1, l2, l3, vtype, polrep=self.polrep)

                # Cut out low snr points
                if np.abs(bi) / bisig < snrcut:
                    continue

                # Append to the equal-time list
                bis.append(np.array((time,
                                     tri[0], tri[1], tri[2],
                                     l1['u'], l1['v'],
                                     l2['u'], l2['v'],
                                     l3['u'], l3['v'],
                                     bi, bisig), dtype=ehc.DTBIS))

            # Append to outlist
            if mode == 'time' and len(bis) > 0:
                out.append(np.array(bis))
                bis = []

        if mode == 'all':
            out = np.array(bis)

        return out

    def c_phases(self, vtype='vis', mode='all', count='min', ang_unit='deg',
                 timetype=False, uv_min=False, snrcut=0.):
        """Return a recarray of the equal time closure phases.

           Args:
               vtype (str): The visibilty type from which to assemble closure phases
                            ('vis','qvis','uvis','vvis','pvis')
               mode (str): If 'time', return phases in a list of equal time arrays,
                           if 'all', return all phases in a single array
               count (str): If 'min', return minimal set of phases,
                            if 'max' return all closure phases up to reordering
               ang_unit (str): If 'deg', return closure phases in degrees, else return in radians
               timetype (str): 'UTC' or 'GMST'
               uv_min (float): flag baselines shorter than this before forming closure quantities
               snrcut (float): flag bispectra with snr lower than this

           Returns:
               (numpy.recarry): A recarray of the closure phases with datatype DTCPHASE
        """

        if timetype is False:
            timetype = self.timetype
        if mode not in ('time', 'all'):
            raise Exception("possible options for mode are 'time' and 'all'")
        if count not in ('max', 'min', 'min-cut0bl'):
            raise Exception("possible options for count are 'max', 'min', or 'min-cut0bl'")
        if vtype not in ('vis', 'qvis', 'uvis', 'vvis', 'rrvis', 'lrvis', 'rlvis', 'llvis'):
            raise Exception("possible options for vtype are" +
                            " 'vis', 'qvis', 'uvis','vvis','rrvis','lrvis','rlvis','llvis'")
        if timetype not in ['GMST', 'UTC', 'gmst', 'utc']:
            raise Exception("timetype should be 'GMST' or 'UTC'!")

        if ang_unit == 'deg':
            angle = ehc.DEGREE
        else:
            angle = 1.0

        # Get the bispectra data
        bispecs = self.bispectra(vtype=vtype, mode='time', count=count,
                                 timetype=timetype, uv_min=uv_min, snrcut=snrcut)

        # Reformat into a closure phase list/array
        out = []
        cps = []

        cpnames = ('time', 't1', 't2', 't3', 'u1', 'v1', 'u2',
                   'v2', 'u3', 'v3', 'cphase', 'sigmacp')
        for bis in bispecs:
            for bi in bis:
                if len(bi) == 0:
                    continue
                bi.dtype.names = cpnames
                bi['sigmacp'] = np.real(bi['sigmacp'] / np.abs(bi['cphase']) / angle)
                bi['cphase'] = np.real((np.angle(bi['cphase']) / angle))
                cps.append(bi.astype(np.dtype(ehc.DTCPHASE)))

            if mode == 'time' and len(cps) > 0:
                out.append(np.array(cps))
                cps = []

        if mode == 'all':
            out = np.array(cps)

        return out

    def c_phases_diag(self, vtype='vis', count='min', ang_unit='deg',
                      timetype=False, uv_min=False, snrcut=0.):
        """Return a recarray of the equal time diagonalized closure phases.

           Args:
               vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis')
                            from which to assemble closure phases
               count (str): If 'min', return minimal set of phases,
                            If 'min-cut0bl' return minimal set after flagging zero-baselines
               ang_unit (str): If 'deg', return closure phases in degrees, else return in radians
               timetype (str): 'UTC' or 'GMST'
               uv_min (float): flag baselines shorter than this before forming closure quantities
               snrcut (float): flag bispectra with snr lower than this

           Returns:
               (numpy.recarry): A recarray of diagonalized closure phases (datatype DTCPHASEDIAG),
                                along with associated triangles and transformation matrices
        """

        if timetype is False:
            timetype = self.timetype
        if count not in ('min', 'min-cut0bl'):
            raise Exception(
                "possible options for count are 'min' or 'min-cut0bl' for diagonal closure phases")
        if vtype not in ('vis', 'qvis', 'uvis', 'vvis', 'rrvis', 'lrvis', 'rlvis', 'llvis'):
            raise Exception(
                "possible options for vtype are 'vis', 'qvis', " +
                "'uvis','vvis','rrvis','lrvis','rlvis','llvis'")
        if timetype not in ['GMST', 'UTC', 'gmst', 'utc']:
            raise Exception("timetype should be 'GMST' or 'UTC'!")

        if ang_unit == 'deg':
            angle = ehc.DEGREE
        else:
            angle = 1.0

        # determine the appropriate sigmatype
        if vtype in ["vis", "qvis", "uvis", "vvis"]:
            if vtype == 'vis':
                sigmatype = 'sigma'
            if vtype == 'qvis':
                sigmatype = 'qsigma'
            if vtype == 'uvis':
                sigmatype = 'usigma'
            if vtype == 'vvis':
                sigmatype = 'vsigma'
        if vtype in ["rrvis", "llvis", "rlvis", "lrvis"]:
            if vtype == 'rrvis':
                sigmatype = 'rrsigma'
            if vtype == 'llvis':
                sigmatype = 'llsigma'
            if vtype == 'rlvis':
                sigmatype = 'rlsigma'
            if vtype == 'lrvis':
                sigmatype = 'lrsigma'

        # get the time-sorted visibility data including conjugate baselines
        viss = np.concatenate(self.tlist(conj=True))

        # get the closure phase data
        cps = self.c_phases(vtype=vtype, mode='all', count=count, ang_unit=ang_unit,
                            timetype=timetype, uv_min=uv_min, snrcut=snrcut)

        # get the unique timestamps for the closure phases
        T_cps = np.unique(cps['time'])

        # list of diagonalized closure phases and corresponding transformation matrices
        dcps = []
        dcp_errs = []
        tfmats = []

        tris = []
        us = []
        vs = []

        # loop over the timestamps
        for kk, t in enumerate(T_cps):

            sys.stdout.write('\rDiagonalizing closure phases:: type %s, count %s, scan %i/%i ' %
                             (vtype, count, kk + 1, len(T_cps)))
            sys.stdout.flush()

            # index masks for this timestamp
            mask_cp = (cps['time'] == t)
            mask_vis = (viss['time'] == t)

            # closure phases for this timestamp
            cps_here = cps[mask_cp]

            # visibilities for this timestamp
            viss_here = viss[mask_vis]

            # initialize the design matrix
            design_mat = np.zeros((mask_cp.sum(), mask_vis.sum()))

            # loop over the closure phases within this timestamp
            trilist = []
            ulist = []
            vlist = []
            for ic, cp in enumerate(cps_here):

                trilist.append((cp['t1'], cp['t2'], cp['t3']))
                ulist.append((cp['u1'], cp['u2'], cp['u3']))
                vlist.append((cp['v1'], cp['v2'], cp['v3']))

                # matrix entry for first leg of triangle
                ind1 = ((viss_here['t1'] == cp['t1']) & (viss_here['t2'] == cp['t2']))
                design_mat[ic, ind1] = 1.0

                # matrix entry for second leg of triangle
                ind2 = ((viss_here['t1'] == cp['t2']) & (viss_here['t2'] == cp['t3']))
                design_mat[ic, ind2] = 1.0

                # matrix entry for third leg of triangle
                ind3 = ((viss_here['t1'] == cp['t3']) & (viss_here['t2'] == cp['t1']))
                design_mat[ic, ind3] = 1.0

            # construct the covariance matrix
            visphase_err = viss_here[sigmatype] / np.abs(viss_here[vtype])
            sigma_mat = np.diag(visphase_err**2.0)
            covar_mat = np.matmul(np.matmul(design_mat, sigma_mat), np.transpose(design_mat))

            # diagonalize via eigendecomposition
            eigeninfo = np.linalg.eigh(covar_mat)
            S_matrix = np.copy(eigeninfo[1]).transpose()
            dcphase = np.matmul(S_matrix, cps_here['cphase'])
            if ang_unit != 'deg':
                dcphase *= angle
            dcphase_err = np.sqrt(np.copy(eigeninfo[0])) / angle

            dcps.append(dcphase)
            dcp_errs.append(dcphase_err)
            tfmats.append(S_matrix)
            tris.append(trilist)
            us.append(ulist)
            vs.append(vlist)

        # Reformat into a list
        out = []
        for kk, t in enumerate(T_cps):
            dcparr = []
            for idcp, dcp in enumerate(dcps[kk]):
                dcparr.append((t, dcp, dcp_errs[kk][idcp]))
            dcparr = np.array(dcparr, dtype=[('time', 'f8'), ('cphase', 'f8'), ('sigmacp', 'f8')])
            out.append((dcparr,
                        np.array(tris[kk]).astype(np.dtype([('trianges', 'U2')])),
                        np.array(us[kk]).astype(np.dtype([('u', 'f8')])),
                        np.array(vs[kk]).astype(np.dtype([('v', 'f8')])),
                        tfmats[kk].astype(np.dtype([('tform_matrix', 'f8')]))))
        print("\n")
        return out

    def bispectra_tri(self, site1, site2, site3, 
                      vtype='vis', timetype=False, snrcut=0., method='from_maxset', 
                      bs=[], force_recompute=False):

        """Return complex bispectrum  over time on a triangle (1-2-3).

           Args:
               site1 (str): station 1 name
               site2 (str): station 2 name
               site3 (str): station 3 name

               vtype (str): The visibilty type from which to assemble bispectra
                            ('vis','qvis','uvis','vvis','pvis')
               timetype (str): 'UTC' or 'GMST'
               snrcut (float): flag bispectra with snr lower than this

               method (str): 'from_maxset' (old, default), 'from_vis' (new, more robust)
               bs (list): optionally pass in the precomputed, time-sorted bispectra
               force_recompute (bool): if True, recompute bispectra instead of using saved data

           Returns:
               (numpy.recarry): A recarray of the bispectra on this triangle with datatype DTBIS
        """
        if timetype is False:
            timetype = self.timetype

        if method=='from_maxset' and (vtype in ['lrvis','pvis','rlvis']):
            print ("Warning! method='from_maxset' default in bispectra_tri() inconsistent with vtype=%s" % vtype)
            print ("Switching to method='from_vis'")
            method = 'from_vis'

        tri = (site1, site2, site3)
        outdata = []

        # get selected bispectra from the maximal set
        # TODO: verify consistency/performance of from_vis, and delete this method
        if method=='from_maxset':

            if ((len(bs) == 0) and not (self.bispec is None) and not (len(self.bispec) == 0) and
                    not force_recompute):
                bs = self.bispec
            elif (len(bs) == 0) or force_recompute:
                bs = self.bispectra(mode='all', count='max', vtype=vtype,
                                    timetype=timetype, snrcut=snrcut)

            # Get requested bispectra over time
            for obs in bs:
                obstri = (obs['t1'], obs['t2'], obs['t3'])
                if set(obstri) == set(tri):
                    t1 = copy.deepcopy(obs['t1'])
                    t2 = copy.deepcopy(obs['t2'])
                    t3 = copy.deepcopy(obs['t3'])
                    u1 = copy.deepcopy(obs['u1'])
                    u2 = copy.deepcopy(obs['u2'])
                    u3 = copy.deepcopy(obs['u3'])
                    v1 = copy.deepcopy(obs['v1'])
                    v2 = copy.deepcopy(obs['v2'])
                    v3 = copy.deepcopy(obs['v3'])

                    # Reorder baselines and flip the sign of the closure phase if necessary
                    if t1 == site1:
                        if t2 == site2:
                            pass
                        else:
                            obs['t2'] = t3
                            obs['t3'] = t2

                            obs['u1'] = -u3
                            obs['v1'] = -v3
                            obs['u2'] = -u2
                            obs['v2'] = -v2
                            obs['u3'] = -u1
                            obs['v3'] = -v1
                            obs['bispec'] = np.conjugate(obs['bispec'])

                    elif t1 == site2:
                        if t2 == site3:
                            obs['t1'] = t3
                            obs['t2'] = t1
                            obs['t3'] = t2

                            obs['u1'] = u3
                            obs['v1'] = v3
                            obs['u2'] = u1
                            obs['v2'] = v1
                            obs['u3'] = u2
                            obs['v3'] = v2

                        else:
                            obs['t1'] = t2
                            obs['t2'] = t1

                            obs['u1'] = -u1
                            obs['v1'] = -v1
                            obs['u2'] = -u3
                            obs['v2'] = -v3
                            obs['u3'] = -u2
                            obs['v3'] = -v2
                            obs['bispec'] = np.conjugate(obs['bispec'])

                    elif t1 == site3:
                        if t2 == site1:
                            obs['t1'] = t2
                            obs['t2'] = t3
                            obs['t3'] = t1

                            obs['u1'] = u2
                            obs['v1'] = v2
                            obs['u2'] = u3
                            obs['v2'] = v3
                            obs['u3'] = u1
                            obs['v3'] = v1

                        else:
                            obs['t1'] = t3
                            obs['t3'] = t1

                            obs['u1'] = -u2
                            obs['v1'] = -v2
                            obs['u2'] = -u1
                            obs['v2'] = -v1
                            obs['u3'] = -u3
                            obs['v3'] = -v3
                            obs['bispec'] = np.conjugate(obs['bispec'])

                    outdata.append(np.array(obs, dtype=ehc.DTBIS))
                    continue

        # get selected bispectra from the visibilities directly
        # taken from bispectra() method
        elif method=='from_vis':

            # get all equal-time data, and loop  over to construct bispectra
            tlist = self.tlist(conj=True)
            for tdata in tlist:

                time = tdata[0]['time']
                if timetype in ['GMST', 'gmst'] and self.timetype == 'UTC':
                    time = obsh.utc_to_gmst(time, self.mjd)
                if timetype in ['UTC', 'utc'] and self.timetype == 'GMST':
                    time = obsh.gmst_to_utc(time, self.mjd)

                # Create a dictionary of baselines at the current time incl. conjugates;
                l_dict = {}
                for dat in tdata:
                    l_dict[(dat['t1'], dat['t2'])] = dat

                # Select triangle entries in the data dictionary
                try:
                    l1 = l_dict[(tri[0], tri[1])]
                    l2 = l_dict[(tri[1], tri[2])]
                    l3 = l_dict[(tri[2], tri[0])]
                except KeyError:
                    continue

                (bi, bisig) = obsh.make_bispectrum(l1, l2, l3, vtype, polrep=self.polrep)

                # Cut out low snr points
                if np.abs(bi) / bisig < snrcut:
                    continue

                # Append to the equal-time list
                outdata.append(np.array((time,
                                         tri[0], tri[1], tri[2],
                                         l1['u'], l1['v'],
                                         l2['u'], l2['v'],
                                         l3['u'], l3['v'],
                                         bi, 
                                         bisig),
                               dtype=ehc.DTBIS))
        else:
            raise Exception("keyword 'method' in bispectra_tri() must be either 'from_cphase' or 'from_vis'")

        outdata = np.array(outdata)
        return outdata

    def cphase_tri(self, site1, site2, site3, vtype='vis', ang_unit='deg',
                   timetype=False, snrcut=0., method='from_maxset', 
                   cphases=[], force_recompute=False):
        """Return closure phase  over time on a triangle (1-2-3).

           Args:
               site1 (str): station 1 name
               site2 (str): station 2 name
               site3 (str): station 3 name

               vtype (str): The visibilty type from which to assemble closure phases
                            (e.g., 'vis','qvis','uvis','vvis','pvis')
               ang_unit (str): If 'deg', return closure phases in degrees, else return in radians
               timetype (str): 'GMST' or 'UTC'
               snrcut (float): flag bispectra with snr lower than this

               method (str): 'from_maxset' (old, default), 'from_vis' (new, more robust)
               cphases (list): optionally pass in the precomputed time-sorted cphases
               force_recompute (bool): if True, do not use save closure phase tables

           Returns:
               (numpy.recarry): A recarray of the closure phases with datatype DTCPHASE
        """

        if timetype is False:
            timetype = self.timetype

        if method=='from_maxset' and (vtype in ['lrvis','pvis','rlvis']):
            print ("Warning! method='from_maxset' default in cphase_tri() is inconsistent with vtype=%s" % vtype)
            print ("Switching to method='from_vis'")
            method = 'from_vis'

        tri = (site1, site2, site3)
        outdata = []

        # get selected closure phases from the maximal set
        # TODO: verify consistency/performance of from_vis, and delete this method
        if method=='from_maxset':
                
            # Get closure phases (maximal set)
            if ((len(cphases) == 0) and not (self.cphase is None) and not (len(self.cphase) == 0) and
                    not force_recompute):
                cphases = self.cphase

            elif (len(cphases) == 0) or force_recompute:
                cphases = self.c_phases(mode='all', count='max', vtype=vtype, ang_unit=ang_unit,
                                        timetype=timetype, snrcut=snrcut)

            # Get requested closure phases over time
            for obs in cphases:
                obstri = (obs['t1'], obs['t2'], obs['t3'])
                if set(obstri) == set(tri):
                    t1 = copy.deepcopy(obs['t1'])
                    t2 = copy.deepcopy(obs['t2'])
                    t3 = copy.deepcopy(obs['t3'])
                    u1 = copy.deepcopy(obs['u1'])
                    u2 = copy.deepcopy(obs['u2'])
                    u3 = copy.deepcopy(obs['u3'])
                    v1 = copy.deepcopy(obs['v1'])
                    v2 = copy.deepcopy(obs['v2'])
                    v3 = copy.deepcopy(obs['v3'])

                    # Reorder baselines and flip the sign of the closure phase if necessary
                    if t1 == site1:
                        if t2 == site2:
                            pass
                        else:
                            obs['t2'] = t3
                            obs['t3'] = t2

                            obs['u1'] = -u3
                            obs['v1'] = -v3
                            obs['u2'] = -u2
                            obs['v2'] = -v2
                            obs['u3'] = -u1
                            obs['v3'] = -v1
                            obs['cphase'] *= -1

                    elif t1 == site2:
                        if t2 == site3:
                            obs['t1'] = t3
                            obs['t2'] = t1
                            obs['t3'] = t2

                            obs['u1'] = u3
                            obs['v1'] = v3
                            obs['u2'] = u1
                            obs['v2'] = v1
                            obs['u3'] = u2
                            obs['v3'] = v2

                        else:
                            obs['t1'] = t2
                            obs['t2'] = t1

                            obs['u1'] = -u1
                            obs['v1'] = -v1
                            obs['u2'] = -u3
                            obs['v2'] = -v3
                            obs['u3'] = -u2
                            obs['v3'] = -v2
                            obs['cphase'] *= -1

                    elif t1 == site3:
                        if t2 == site1:
                            obs['t1'] = t2
                            obs['t2'] = t3
                            obs['t3'] = t1

                            obs['u1'] = u2
                            obs['v1'] = v2
                            obs['u2'] = u3
                            obs['v2'] = v3
                            obs['u3'] = u1
                            obs['v3'] = v1

                        else:
                            obs['t1'] = t3
                            obs['t3'] = t1

                            obs['u1'] = -u2
                            obs['v1'] = -v2
                            obs['u2'] = -u1
                            obs['v2'] = -v1
                            obs['u3'] = -u3
                            obs['v3'] = -v3
                            obs['cphase'] *= -1

                    outdata.append(np.array(obs, dtype=ehc.DTCPHASE))
                    continue

        # get selected closure phases from the visibilities directly
        # taken from bispectra() method
        elif method=='from_vis':
            if ang_unit == 'deg': angle = ehc.DEGREE
            else: angle = 1.0

            # get all equal-time data, and loop  over to construct closure phase
            tlist = self.tlist(conj=True)
            for tdata in tlist:

                time = tdata[0]['time']
                if timetype in ['GMST', 'gmst'] and self.timetype == 'UTC':
                    time = obsh.utc_to_gmst(time, self.mjd)
                if timetype in ['UTC', 'utc'] and self.timetype == 'GMST':
                    time = obsh.gmst_to_utc(time, self.mjd)

                # Create a dictionary of baselines at the current time incl. conjugates;
                l_dict = {}
                for dat in tdata:
                    l_dict[(dat['t1'], dat['t2'])] = dat

                # Select triangle entries in the data dictionary
                try:
                    l1 = l_dict[(tri[0], tri[1])]
                    l2 = l_dict[(tri[1], tri[2])]
                    l3 = l_dict[(tri[2], tri[0])]
                except KeyError:
                    continue

                (bi, bisig) = obsh.make_bispectrum(l1, l2, l3, vtype, polrep=self.polrep)

                # Cut out low snr points
                if np.abs(bi) / bisig < snrcut:
                    continue

                # Append to the equal-time list
                outdata.append(np.array((time,
                                         tri[0], tri[1], tri[2],
                                         l1['u'], l1['v'],
                                         l2['u'], l2['v'],
                                         l3['u'], l3['v'],
                                         np.real(np.angle(bi) / angle),
                                         np.real(bisig / np.abs(bi) / angle)),
                               dtype=ehc.DTCPHASE))
        else:
            raise Exception("keyword 'method' in cphase_tri() must be either 'from_cphase' or 'from_vis'")

        outdata = np.array(outdata)
        return outdata

    def c_amplitudes(self, vtype='vis', mode='all', count='min', ctype='camp', debias=True,
                     timetype=False, snrcut=0.):
        """Return a recarray of the equal time closure amplitudes.

           Args:
               vtype (str): The visibilty type from which to assemble closure amplitudes
                            ('vis','qvis','uvis','vvis','pvis')
               ctype (str): The closure amplitude type ('camp' or 'logcamp')
               mode (str): If 'time', return amplitudes in a list of equal time arrays,
                           if 'all', return all amplitudes in a single array
               count (str): If 'min', return minimal set of amplitudes,
                            if 'max' return all closure amplitudes up to inverses
               debias (bool): If True, debias the closure amplitude
               timetype (str): 'GMST' or 'UTC'
               snrcut (float): flag closure amplitudes with snr lower than this

           Returns:
               (numpy.recarry): A recarray of the closure amplitudes with datatype DTCAMP

        """

        if timetype is False:
            timetype = self.timetype
        if mode not in ('time', 'all'):
            raise Exception("possible options for mode are 'time' and 'all'")
        if count not in ('max', 'min'):
            raise Exception("possible options for count are 'max' and 'min'")
        if vtype not in ('vis', 'qvis', 'uvis', 'vvis', 'rrvis', 'lrvis', 'rlvis', 'llvis'):
            raise Exception("possible options for vtype are " +
                            "'vis', 'qvis', 'uvis','vvis','rrvis','lrvis','rlvis','llvis'")
        if not (ctype in ['camp', 'logcamp']):
            raise Exception("closure amplitude type must be 'camp' or 'logcamp'!")
        if timetype not in ['GMST', 'UTC', 'gmst', 'utc']:
            raise Exception("timetype should be 'GMST' or 'UTC'!")

        # Get data sorted by time
        tlist = self.tlist(conj=True)
        out = []
        cas = []
        tt = 1
        for tdata in tlist:

            # sys.stdout.write('\rGetting closure amps:: type %s %s , count %s, scan %i/%i' %
            #                 (vtype, ctype, count, tt, len(tlist)))
            # sys.stdout.flush()
            tt += 1

            time = tdata[0]['time']
            if timetype in ['GMST', 'gmst'] and self.timetype == 'UTC':
                time = obsh.utc_to_gmst(time, self.mjd)
            if timetype in ['UTC', 'utc'] and self.timetype == 'GMST':
                time = obsh.gmst_to_utc(time, self.mjd)

            sites = np.array(list(set(np.hstack((tdata['t1'], tdata['t2'])))))
            if len(sites) < 4:
                continue

            # Create a dictionary of baseline data at the current time including conjugates;
            l_dict = {}
            for dat in tdata:
                l_dict[(dat['t1'], dat['t2'])] = dat

            # Minimal set
            if count == 'min':
                quadsets = obsh.quad_minimal_set(sites, self.tarr, self.tkey)

            # Maximal Set
            elif count == 'max':
                # Find all quadrangles
                quadsets = np.sort(list(it.combinations(sites, 4)))
                # Include 3 closure amplitudes on each quadrangle
                quadsets = np.array([(q, [q[0], q[2], q[1], q[3]], [q[0], q[1], q[3], q[2]])
                                     for q in quadsets]).reshape((-1, 4))

            # Loop over all closure amplitudes
            for quad in quadsets:
                # Blue is numerator, red is denominator
                if (quad[0], quad[1]) not in l_dict.keys():
                    continue
                if (quad[2], quad[3]) not in l_dict.keys():
                    continue
                if (quad[1], quad[2]) not in l_dict.keys():
                    continue
                if (quad[0], quad[3]) not in l_dict.keys():
                    continue

                try:
                    blue1 = l_dict[quad[0], quad[1]]
                    blue2 = l_dict[quad[2], quad[3]]
                    red1 = l_dict[quad[0], quad[3]]
                    red2 = l_dict[quad[1], quad[2]]
                except KeyError:
                    continue

                # Compute the closure amplitude and the error
                (camp, camperr) = obsh.make_closure_amplitude(blue1, blue2, red1, red2, vtype,
                                                              polrep=self.polrep,
                                                              ctype=ctype, debias=debias)

                if ctype == 'camp' and camp / camperr < snrcut:
                    continue
                elif ctype == 'logcamp' and 1. / camperr < snrcut:
                    continue

                # Add the closure amplitudes to the equal-time list
                # Our site convention is (12)(34)/(14)(23)
                cas.append(np.array((time,
                                     quad[0], quad[1], quad[2], quad[3],
                                     blue1['u'], blue1['v'], blue2['u'], blue2['v'],
                                     red1['u'], red1['v'], red2['u'], red2['v'],
                                     camp, camperr),
                                    dtype=ehc.DTCAMP))

            # Append all equal time closure amps to outlist
            if mode == 'time':
                out.append(np.array(cas))
                cas = []

        if mode == 'all':
            out = np.array(cas)

        return out

    def c_log_amplitudes_diag(self, vtype='vis', mode='all', count='min',
                              debias=True, timetype=False, snrcut=0.):
        """Return a recarray of the equal time diagonalized log closure amplitudes.

           Args:
               vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis')
                            From which to assemble closure amplitudes
               ctype (str): The closure amplitude type ('camp' or 'logcamp')
               mode (str): If 'time', return amplitudes in a list of equal time arrays,
                           If 'all', return all amplitudes in a single array
               count (str): If 'min', return minimal set of amplitudes,
                            If 'max' return all closure amplitudes up to inverses
               debias (bool): If True, debias the closure amplitude -
                              The individual visibility amplitudes are always debiased.
               timetype (str): 'GMST' or 'UTC'
               snrcut (float): flag closure amplitudes with snr lower than this

           Returns:
               (numpy.recarry): A recarray of diagonalized closure amps with datatype DTLOGCAMPDIAG

        """

        if timetype is False:
            timetype = self.timetype
        if mode not in ('time', 'all'):
            raise Exception("possible options for mode are 'time' and 'all'")
        if count not in ('min'):
            raise Exception("count can only be 'min' for diagonal log closure amplitudes")
        if vtype not in ('vis', 'qvis', 'uvis', 'vvis', 'rrvis', 'lrvis', 'rlvis', 'llvis'):
            raise Exception(
                "possible options for vtype are 'vis', 'qvis', 'uvis', " +
                "'vvis','rrvis','lrvis','rlvis','llvis'")
        if timetype not in ['GMST', 'UTC', 'gmst', 'utc']:
            raise Exception("timetype should be 'GMST' or 'UTC'!")

        # determine the appropriate sigmatype
        if vtype in ["vis", "qvis", "uvis", "vvis"]:
            if vtype == 'vis':
                sigmatype = 'sigma'
            if vtype == 'qvis':
                sigmatype = 'qsigma'
            if vtype == 'uvis':
                sigmatype = 'usigma'
            if vtype == 'vvis':
                sigmatype = 'vsigma'
        if vtype in ["rrvis", "llvis", "rlvis", "lrvis"]:
            if vtype == 'rrvis':
                sigmatype = 'rrsigma'
            if vtype == 'llvis':
                sigmatype = 'llsigma'
            if vtype == 'rlvis':
                sigmatype = 'rlsigma'
            if vtype == 'lrvis':
                sigmatype = 'lrsigma'

        # get the time-sorted visibility data including conjugate baselines
        viss = np.concatenate(self.tlist(conj=True))

        # get the log closure amplitude data
        lcas = self.c_amplitudes(vtype=vtype, mode=mode, count=count,
                                 ctype='logcamp', debias=debias, timetype=timetype, snrcut=snrcut)

        # get the unique timestamps for the log closure amplitudes
        T_lcas = np.unique(lcas['time'])

        # list of diagonalized log closure camplitudes and corresponding transformation matrices
        dlcas = []
        dlca_errs = []
        tfmats = []

        quads = []
        us = []
        vs = []

        # loop over the timestamps
        for kk, t in enumerate(T_lcas):

            printstr = ('\rDiagonalizing log closure amplitudes:: type %s, count %s, scan %i/%i ' %
                        (vtype, count, kk + 1, len(T_lcas)))
            sys.stdout.write(printstr)
            sys.stdout.flush()

            # index masks for this timestamp
            mask_lca = (lcas['time'] == t)
            mask_vis = (viss['time'] == t)

            # log closure amplitudes for this timestamp
            lcas_here = lcas[mask_lca]

            # visibilities for this timestamp
            viss_here = viss[mask_vis]

            # initialize the design matrix
            design_mat = np.zeros((mask_lca.sum(), mask_vis.sum()))

            # loop over the log closure amplitudes within this timestamp
            quadlist = []
            ulist = []
            vlist = []
            for il, lca in enumerate(lcas_here):

                quadlist.append((lca['t1'], lca['t2'], lca['t3'], lca['t4']))
                ulist.append((lca['u1'], lca['u2'], lca['u3'], lca['u4']))
                vlist.append((lca['v1'], lca['v2'], lca['v3'], lca['v4']))

                # matrix entry for first leg of quadrangle
                ind1 = ((viss_here['t1'] == lca['t1']) & (viss_here['t2'] == lca['t2']))
                design_mat[il, ind1] = 1.0

                # matrix entry for second leg of quadrangle
                ind2 = ((viss_here['t1'] == lca['t3']) & (viss_here['t2'] == lca['t4']))
                design_mat[il, ind2] = 1.0

                # matrix entry for third leg of quadrangle
                ind3 = ((viss_here['t1'] == lca['t1']) & (viss_here['t2'] == lca['t4']))
                design_mat[il, ind3] = -1.0

                # matrix entry for fourth leg of quadrangle
                ind4 = ((viss_here['t1'] == lca['t2']) & (viss_here['t2'] == lca['t3']))
                design_mat[il, ind4] = -1.0

            # construct the covariance matrix
            logvisamp_err = viss_here[sigmatype] / np.abs(viss_here[vtype])
            sigma_mat = np.diag(logvisamp_err**2.0)
            covar_mat = np.matmul(np.matmul(design_mat, sigma_mat), np.transpose(design_mat))

            # diagonalize via eigendecomposition
            eigeninfo = np.linalg.eigh(covar_mat)
            T_matrix = np.copy(eigeninfo[1]).transpose()
            dlogcamp = np.matmul(T_matrix, lcas_here['camp'])
            dlogcamp_err = np.sqrt(np.copy(eigeninfo[0]))

            dlcas.append(dlogcamp)
            dlca_errs.append(dlogcamp_err)
            tfmats.append(T_matrix)
            quads.append(quadlist)
            us.append(ulist)
            vs.append(vlist)

        # Reformat into a list
        out = []
        for kk, t in enumerate(T_lcas):
            dlcaarr = []
            for idlca, dlca in enumerate(dlcas[kk]):
                dlcaarr.append((t, dlca, dlca_errs[kk][idlca]))
            dlcaarr = np.array(dlcaarr, dtype=[('time', 'f8'), ('camp', 'f8'), ('sigmaca', 'f8')])
            out.append((dlcaarr,
                        np.array(quads[kk]).astype(np.dtype([('quadrangles', 'U2')])),
                        np.array(us[kk]).astype(np.dtype([('u', 'f8')])),
                        np.array(vs[kk]).astype(np.dtype([('v', 'f8')])),
                        tfmats[kk].astype(np.dtype([('tform_matrix', 'f8')]))))
        print("\n")
        return out

    def camp_quad(self, site1, site2, site3, site4, 
                  vtype='vis', ctype='camp', debias=True, timetype=False, snrcut=0.,
                  method='from_maxset',
                  camps=[], force_recompute=False):
        """Return closure phase over time on a quadrange (1-2)(3-4)/(1-4)(2-3).

           Args:
               site1 (str): station 1 name
               site2 (str): station 2 name
               site3 (str): station 3 name
               site4 (str): station 4 name

               vtype (str): The visibilty type from which to assemble closure amplitudes
                            ('vis','qvis','uvis','vvis','pvis')
               ctype (str): The closure amplitude type ('camp' or 'logcamp')
               debias (bool): If True, debias the closure amplitude
               timetype (str): 'UTC' or 'GMST'
               snrcut (float): flag closure amplitudes with snr lower than this

               method (str): 'from_maxset' (old, default), 'from_vis' (new, more robust)
               camps (list): optionally pass in the time-sorted, precomputed camps
               force_recompute (bool): if True, do not use save closure amplitude data

           Returns:
               (numpy.recarry): A recarray of the closure amplitudes with datatype DTCAMP
        """

        if timetype is False:
            timetype = self.timetype


        if method=='from_maxset' and (vtype in ['lrvis','pvis','rlvis']):
            print ("Warning! method='from_maxset' default in camp_quad() is inconsistent with vtype=%s" % vtype)
            print ("Switching to method='from_vis'")
            method = 'from_vis'

        quad = (site1, site2, site3, site4)
        outdata = []

        # get selected closure amplitudes from the maximal set
        # TODO: verify consistency/performance of from_vis, and delete this method
        if method=='from_maxset':
            if (((ctype == 'camp') and (len(camps) == 0)) and not (self.camp is None) and
                    not (len(self.camp) == 0) and not force_recompute):
                camps = self.camp
            elif (((ctype == 'logcamp') and (len(camps) == 0)) and not (self.logcamp is None) and
                  not (len(self.logcamp) == 0) and not force_recompute):
                camps = self.logcamp
            elif (len(camps) == 0) or force_recompute:
                camps = self.c_amplitudes(mode='all', count='max', vtype=vtype, ctype=ctype,
                                          debias=debias, timetype=timetype, snrcut=snrcut)

            # blue bls in numerator, red in denominator
            b1 = set((site1, site2))
            b2 = set((site3, site4))
            r1 = set((site1, site4))
            r2 = set((site2, site3))

            for obs in camps: # camps does not contain inverses!

                num = [set((obs['t1'], obs['t2'])), set((obs['t3'], obs['t4']))]
                denom = [set((obs['t1'], obs['t4'])), set((obs['t2'], obs['t3']))]

                obsquad = (obs['t1'], obs['t2'], obs['t3'], obs['t4'])
                if set(quad) == set(obsquad):

                    # is this either  the closure amplitude or inverse?
                    rightup = (b1 in num) and (b2 in num) and (r1 in denom) and (r2 in denom)
                    wrongup = (b1 in denom) and (b2 in denom) and (r1 in num) and (r2 in num)
                    if not (rightup or wrongup):
                        continue

                    # flip the inverse closure amplitudes
                    if wrongup:
                        t1old = copy.deepcopy(obs['t1'])
                        u1old = copy.deepcopy(obs['u1'])
                        v1old = copy.deepcopy(obs['v1'])
                        t2old = copy.deepcopy(obs['t2'])
                        u2old = copy.deepcopy(obs['u2'])
                        v2old = copy.deepcopy(obs['v2'])
                        t3old = copy.deepcopy(obs['t3'])
                        u3old = copy.deepcopy(obs['u3'])
                        v3old = copy.deepcopy(obs['v3'])
                        t4old = copy.deepcopy(obs['t4'])
                        u4old = copy.deepcopy(obs['u4'])
                        v4old = copy.deepcopy(obs['v4'])
                        campold = copy.deepcopy(obs['camp'])
                        csigmaold = copy.deepcopy(obs['sigmaca'])

                        obs['t1'] = t1old
                        obs['t2'] = t4old
                        obs['t3'] = t3old
                        obs['t4'] = t2old

                        obs['u1'] = u3old
                        obs['v1'] = v3old

                        obs['u2'] = -u4old
                        obs['v2'] = -v4old

                        obs['u3'] = u1old
                        obs['v3'] = v1old

                        obs['u4'] = -u2old
                        obs['v4'] = -v2old

                        if ctype == 'logcamp':
                            obs['camp'] = -campold
                            obs['sigmaca'] = csigmaold
                        else:
                            obs['camp'] = 1. / campold
                            obs['sigmaca'] = csigmaold / (campold**2)

                    t1old = copy.deepcopy(obs['t1'])
                    u1old = copy.deepcopy(obs['u1'])
                    v1old = copy.deepcopy(obs['v1'])
                    t2old = copy.deepcopy(obs['t2'])
                    u2old = copy.deepcopy(obs['u2'])
                    v2old = copy.deepcopy(obs['v2'])
                    t3old = copy.deepcopy(obs['t3'])
                    u3old = copy.deepcopy(obs['u3'])
                    v3old = copy.deepcopy(obs['v3'])
                    t4old = copy.deepcopy(obs['t4'])
                    u4old = copy.deepcopy(obs['u4'])
                    v4old = copy.deepcopy(obs['v4'])

                    # this is all same closure amplitude, but the ordering of labels is different
                    # return the label ordering that the user requested!
                    if (obs['t2'], obs['t1'], obs['t4'], obs['t3']) == quad:
                        obs['t1'] = t2old
                        obs['t2'] = t1old
                        obs['t3'] = t4old
                        obs['t4'] = t3old

                        obs['u1'] = -u1old
                        obs['v1'] = -v1old

                        obs['u2'] = -u2old
                        obs['v2'] = -v2old

                        obs['u3'] = u4old
                        obs['v3'] = v4old

                        obs['u4'] = u3old
                        obs['v4'] = v3old

                    elif (obs['t3'], obs['t4'], obs['t1'], obs['t2']) == quad:
                        obs['t1'] = t3old
                        obs['t2'] = t4old
                        obs['t3'] = t1old
                        obs['t4'] = t2old

                        obs['u1'] = u2old
                        obs['v1'] = v2old

                        obs['u2'] = u1old
                        obs['v2'] = v1old

                        obs['u3'] = -u4old
                        obs['v3'] = -v4old

                        obs['u4'] = -u3old
                        obs['v4'] = -v3old

                    elif (obs['t4'], obs['t3'], obs['t2'], obs['t1']) == quad:
                        obs['t1'] = t4old
                        obs['t2'] = t3old
                        obs['t3'] = t2old
                        obs['t4'] = t1old

                        obs['u1'] = -u2old
                        obs['v1'] = -v2old

                        obs['u2'] = -u1old
                        obs['v2'] = -v1old

                        obs['u3'] = -u3old
                        obs['v3'] = -v3old

                        obs['u4'] = -u4old
                        obs['v4'] = -v4old

                    # append to output array
                    outdata.append(np.array(obs, dtype=ehc.DTCAMP))

        # get selected bispectra from the visibilities directly
        # taken from c_ampitudes() method
        elif method=='from_vis':

            # get all equal-time data, and loop  over to construct closure amplitudes
            tlist = self.tlist(conj=True)
            for tdata in tlist:

                time = tdata[0]['time']
                if timetype in ['GMST', 'gmst'] and self.timetype == 'UTC':
                    time = obsh.utc_to_gmst(time, self.mjd)
                if timetype in ['UTC', 'utc'] and self.timetype == 'GMST':
                    time = obsh.gmst_to_utc(time, self.mjd)
                sites = np.array(list(set(np.hstack((tdata['t1'], tdata['t2'])))))
                if len(sites) < 4:
                    continue

                # Create a dictionary of baselines at the current time incl. conjugates;
                l_dict = {}
                for dat in tdata:
                    l_dict[(dat['t1'], dat['t2'])] = dat

                # Select quadrangle entries in the data dictionary
                # Blue is numerator, red is denominator
                if (quad[0], quad[1]) not in l_dict.keys():
                    continue
                if (quad[2], quad[3]) not in l_dict.keys():
                    continue
                if (quad[1], quad[2]) not in l_dict.keys():
                    continue
                if (quad[0], quad[3]) not in l_dict.keys():
                    continue

                try:
                    blue1 = l_dict[quad[0], quad[1]]
                    blue2 = l_dict[quad[2], quad[3]]
                    red1 = l_dict[quad[0], quad[3]]
                    red2 = l_dict[quad[1], quad[2]]
                except KeyError:
                    continue

                # Compute the closure amplitude and the error
                (camp, camperr) = obsh.make_closure_amplitude(blue1, blue2, red1, red2, vtype,
                                                              polrep=self.polrep,
                                                              ctype=ctype, debias=debias)

                if ctype == 'camp' and camp / camperr < snrcut:
                    continue
                elif ctype == 'logcamp' and 1. / camperr < snrcut:
                    continue

                # Add the closure amplitudes to the equal-time list
                # Our site convention is (12)(34)/(14)(23)
                outdata.append(np.array((time,
                                         quad[0], quad[1], quad[2], quad[3],
                                         blue1['u'], blue1['v'], blue2['u'], blue2['v'],
                                         red1['u'], red1['v'], red2['u'], red2['v'],
                                         camp, camperr),
                                         dtype=ehc.DTCAMP))

        else:
            raise Exception("keyword 'method' in camp_quad() must be either 'from_cphase' or 'from_vis'")

        outdata = np.array(outdata)
        return outdata

    def plotall(self, field1, field2,
                conj=False, debias=False, tag_bl=False, ang_unit='deg', timetype=False,
                axis=False, rangex=False, rangey=False, xscale='linear', yscale='linear',
                color=ehc.SCOLORS[0], marker='o', markersize=ehc.MARKERSIZE, label=None,
                snrcut=0.,
                grid=True, ebar=True, axislabels=True, legend=False,
                show=True, export_pdf=""):
        """Plot two fields against each other.

           Args:
               field1 (str): x-axis field (from FIELDS)
               field2 (str): y-axis field (from FIELDS)

               conj (bool): Plot conjuage baseline data points if True
               debias (bool): If True, debias amplitudes.
               tag_bl (bool): if True, label each baseline
               ang_unit (str): phase unit 'deg' or 'rad'
               timetype (str): 'GMST' or 'UTC'

               axis (matplotlib.axes.Axes): add plot to this axis
               xscale (str): 'linear' or 'log' y-axis scale               
               yscale (str): 'linear' or 'log' y-axis scale
               
               rangex (list): [xmin, xmax] x-axis limits
               rangey (list): [ymin, ymax] y-axis limits

               color (str): color for scatterplot points
               marker (str): matplotlib plot marker
               markersize (int): size of plot markers
               label (str): plot legend label
               snrcut (float): flag closure amplitudes with snr lower than this

               grid (bool): Plot gridlines if True
               ebar (bool): Plot error bars if True
               axislabels (bool): Show axis labels if True
               legend (bool): Show legend if True

               show (bool): Display the plot if true
               export_pdf (str): path to pdf file to save figure

           Returns:
               (matplotlib.axes.Axes): Axes object with data plot
        """

        if timetype is False:
            timetype = self.timetype

        # Determine if fields are valid
        field1 = field1.lower()
        field2 = field2.lower()
        if (field1 not in ehc.FIELDS) and (field2 not in ehc.FIELDS):
            raise Exception("valid fields are " + ' '.join(ehc.FIELDS))

        if 'amp' in [field1, field2] and not (self.amp is None):
            print("Warning: plotall is not using amplitudes in Obsdata.amp array!")

        # Label individual baselines
        # ANDREW TODO this is way too slow, make  it faster??
        if tag_bl:
            clist = ehc.SCOLORS

            # make a color coding dictionary
            cdict = {}
            ii = 0
            baselines = np.sort(list(it.combinations(self.tarr['site'], 2)))
            for baseline in baselines:
                cdict[(baseline[0], baseline[1])] = clist[ii % len(clist)]
                cdict[(baseline[1], baseline[0])] = clist[ii % len(clist)]
                ii += 1

            # get unique baselines -- TODO easier way? separate function?
            alldata = []
            allsigx = []
            allsigy = []
            bllist = []
            colors = []
            bldata = self.bllist(conj=conj)
            for bl in bldata:
                t1 = bl['t1'][0]
                t2 = bl['t2'][0]

                bllist.append((t1, t2))
                colors.append(cdict[(t1, t2)])

                # Unpack data
                dat = self.unpack_dat(bl, [field1, field2],
                                      ang_unit=ang_unit, debias=debias, timetype=timetype)
                alldata.append(dat)

                # X error bars
                if obsh.sigtype(field1):
                    allsigx.append(self.unpack_dat(bl, [obsh.sigtype(field1)],
                                                   ang_unit=ang_unit)[obsh.sigtype(field1)])
                else:
                    allsigx.append(None)

                # Y error bars
                if obsh.sigtype(field2):
                    allsigy.append(self.unpack_dat(bl, [obsh.sigtype(field2)],
                                                   ang_unit=ang_unit)[obsh.sigtype(field2)])
                else:
                    allsigy.append(None)

        # Don't Label individual baselines
        else:
            bllist = [['All', 'All']]
            colors = [color]

            # unpack data
            alldata = [self.unpack([field1, field2],
                                   conj=conj, ang_unit=ang_unit, debias=debias, timetype=timetype)]

            # X error bars
            if obsh.sigtype(field1):
                allsigx = self.unpack(obsh.sigtype(field2), conj=conj, ang_unit=ang_unit)
                allsigx = [allsigx[obsh.sigtype(field1)]]
            else:
                allsigx = [None]

            # Y error bars
            if obsh.sigtype(field2):
                allsigy = self.unpack(obsh.sigtype(field2), conj=conj, ang_unit=ang_unit)
                allsigy = [allsigy[obsh.sigtype(field2)]]
            else:
                allsigy = [None]

        # make plot(s)
        if axis:
            x = axis
        else:
            fig = plt.figure()
            x = fig.add_subplot(1, 1, 1)

        xmins = []
        xmaxes = []
        ymins = []
        ymaxes = []
        for i in range(len(alldata)):
            data = alldata[i]
            sigy = allsigy[i]
            sigx = allsigx[i]
            color = colors[i]
            bl = bllist[i]

            # Flag out nans (to avoid problems determining plotting limits)
            mask = ~(np.isnan(data[field1]) + np.isnan(data[field2]))

            # Flag out due to snrcut
            if snrcut > 0.:
                sigs = [sigx, sigy]
                for jj, field in enumerate([field1, field2]):
                    if field in ehc.FIELDS_AMPS:
                        fmask = data[field] / sigs[jj] > snrcut
                    elif field in ehc.FIELDS_PHASE:
                        fmask = sigs[jj] < (180. / np.pi / snrcut)
                    elif field in ehc.FIELDS_SNRS:
                        fmask = data[field] > snrcut
                    else:
                        fmask = np.ones(mask.shape).astype(bool)
                    mask *= fmask

            data = data[mask]
            if sigy is not None:
                sigy = sigy[mask]
            if sigx is not None:
                sigx = sigx[mask]
            if len(data) == 0:
                continue

            xmins.append(np.min(data[field1]))
            xmaxes.append(np.max(data[field1]))
            ymins.append(np.min(data[field2]))
            ymaxes.append(np.max(data[field2]))

            # Plot the data
            tolerance = len(data[field2])

            if label is None:
                labelstr = "%s-%s" % ((str(bl[0]), str(bl[1])))

            else:
                labelstr = str(label)

            if ebar and (np.any(sigy) or np.any(sigx)):
                x.errorbar(data[field1], data[field2], xerr=sigx, yerr=sigy, label=labelstr,
                           fmt=marker, markersize=markersize, color=color, picker=tolerance)
            else:
                x.plot(data[field1], data[field2], marker, markersize=markersize, color=color,
                       label=labelstr, picker=tolerance)

        # axis scales
        x.set_xscale(xscale)
        x.set_yscale(yscale)
        
        # Data ranges
        if not rangex:
            rangex = [np.min(xmins) - 0.2 * np.abs(np.min(xmins)),
                      np.max(xmaxes) + 0.2 * np.abs(np.max(xmaxes))]
            if np.any(np.isnan(np.array(rangex))):
                print("Warning: NaN in data x range: specifying rangex to default")
                rangex = [-100, 100]

        if not rangey:
            rangey = [np.min(ymins) - 0.2 * np.abs(np.min(ymins)),
                      np.max(ymaxes) + 0.2 * np.abs(np.max(ymaxes))]
            if np.any(np.isnan(np.array(rangey))):
                print("Warning: NaN in data y range: specifying rangey to default")
                rangey = [-100, 100]

        x.set_xlim(rangex)
        x.set_ylim(rangey)


        # label and save
        if axislabels:
            try:
                x.set_xlabel(ehc.FIELD_LABELS[field1])
                x.set_ylabel(ehc.FIELD_LABELS[field2])
            except KeyError:
                x.set_xlabel(field1.capitalize())
                x.set_ylabel(field2.capitalize())
        if legend and tag_bl:
            plt.legend(ncol=2)
        elif legend:
            plt.legend()
        if grid:
            x.grid()
        if export_pdf != "" and not axis:
            fig.savefig(export_pdf, bbox_inches='tight')
        if export_pdf != "" and axis:
            fig = plt.gcf()
            fig.savefig(export_pdf, bbox_inches='tight')
        if show:
            #plt.show(block=False)
            ehc.show_noblock()

        return x

    def plot_bl(self, site1, site2, field,
                debias=False, ang_unit='deg', timetype=False,
                axis=False, rangex=False, rangey=False, snrcut=0.,
                color=ehc.SCOLORS[0], marker='o', markersize=ehc.MARKERSIZE, label=None,
                grid=True, ebar=True, axislabels=True, legend=False,
                show=True, export_pdf=""):
        """Plot a field over time on a baseline site1-site2.

           Args:
               site1 (str): station 1 name
               site2 (str): station 2 name
               field (str): y-axis field (from FIELDS)

               debias (bool): If True, debias amplitudes.
               ang_unit (str): phase unit 'deg' or 'rad'
               timetype (str): 'GMST' or 'UTC'

               axis (matplotlib.axes.Axes): add plot to this axis
               rangex (list): [xmin, xmax] x-axis limits
               rangey (list): [ymin, ymax] y-axis limits

               color (str): color for scatterplot points
               marker (str): matplotlib plot marker
               markersize (int): size of plot markers
               label (str): plot legend label

               grid (bool): Plot gridlines if True
               ebar (bool): Plot error bars if True
               axislabels (bool): Show axis labels if True
               legend (bool): Show legend if True
               show (bool): Display the plot if true
               export_pdf (str): path to pdf file to save figure

           Returns:
               (matplotlib.axes.Axes): Axes object with data plot
        """

        if timetype is False:
            timetype = self.timetype

        field = field.lower()
        if field == 'amp' and not (self.amp is None):
            print("Warning: plot_bl is not using amplitudes in Obsdata.amp array!")

        if label is None:
            label = str(self.source)
        else:
            label = str(label)

        # Determine if fields are valid
        if field not in ehc.FIELDS:
            raise Exception("valid fields are " + string.join(ehc.FIELDS))

        plotdata = self.unpack_bl(site1, site2, field, ang_unit=ang_unit,
                                  debias=debias, timetype=timetype)
        sigmatype = obsh.sigtype(field)
        if obsh.sigtype(field):
            errdata = self.unpack_bl(site1, site2, obsh.sigtype(field),
                                     ang_unit=ang_unit, debias=debias)
        else:
            errdata = None

        # Flag out nans (to avoid problems determining plotting limits)
        mask = ~np.isnan(plotdata[field][:, 0])

        # Flag out due to snrcut
        if snrcut > 0.:
            if field in ehc.FIELDS_AMPS:
                fmask = plotdata[field] / errdata[sigmatype] > snrcut
            elif field in ehc.FIELDS_PHASE:
                fmask = errdata[sigmatype] < (180. / np.pi / snrcut)
            elif field in ehc.FIELDS_SNRS:
                fmask = plotdata[field] > snrcut
            else:
                fmask = np.ones(mask.shape).astype(bool)
            fmask = fmask[:, 0]
            mask *= fmask

        plotdata = plotdata[mask]
        if errdata is not None:
            errdata = errdata[mask]

        if not rangex:
            rangex = [self.tstart, self.tstop]
            if np.any(np.isnan(np.array(rangex))):
                print("Warning: NaN in data x range: specifying rangex to default")
                rangex = [0, 24]
        if not rangey:
            rangey = [np.min(plotdata[field]) - 0.2 * np.abs(np.min(plotdata[field])),
                      np.max(plotdata[field]) + 0.2 * np.abs(np.max(plotdata[field]))]
            if np.any(np.isnan(np.array(rangey))):
                print("Warning: NaN in data y range: specifying rangex to default")
                rangey = [-100, 100]

        # Plot the data
        if axis:
            x = axis
        else:
            fig = plt.figure()
            x = fig.add_subplot(1, 1, 1)

        if ebar and obsh.sigtype(field) is not False:
            x.errorbar(plotdata['time'][:, 0], plotdata[field][:, 0],
                       yerr=errdata[obsh.sigtype(field)][:, 0],
                       fmt=marker, markersize=markersize, color=color,
                       linestyle='none', label=label)
        else:
            x.plot(plotdata['time'][:, 0], plotdata[field][:, 0], marker, markersize=markersize,
                   color=color, label=label, linestyle='none')

        x.set_xlim(rangex)
        x.set_ylim(rangey)

        if axislabels:
            x.set_xlabel(timetype + ' (hr)')
            try:
                x.set_ylabel(ehc.FIELD_LABELS[field])
            except KeyError:
                x.set_ylabel(field.capitalize())
            x.set_title('%s - %s' % (site1, site2))

        if grid:
            x.grid()
        if legend:
            plt.legend()
        if export_pdf != "" and not axis:
            fig.savefig(export_pdf, bbox_inches='tight')
        if export_pdf != "" and axis:
            fig = plt.gcf()
            fig.savefig(export_pdf, bbox_inches='tight')
        if show:
            #plt.show(block=False)
            ehc.show_noblock()
        return x

    def plot_cphase(self, site1, site2, site3,
                    vtype='vis', cphases=[], force_recompute=False,
                    ang_unit='deg', timetype=False, snrcut=0.,
                    axis=False, rangex=False, rangey=False,
                    color=ehc.SCOLORS[0], marker='o', markersize=ehc.MARKERSIZE, label=None,
                    grid=True, ebar=True, axislabels=True, legend=False,
                    show=True, export_pdf=""):
        """Plot a closure phase over time on a triangle site1-site2-site3.

           Args:
               site1 (str): station 1 name
               site2 (str): station 2 name
               site3 (str): station 3 name

               vtype (str): The visibilty type from which to assemble closure phases
                            ('vis','qvis','uvis','vvis','pvis')
               cphases (list): optionally pass in the prcomputed, time-sorted closure phases
               force_recompute (bool): if True, do not use stored closure phase able
               snrcut (float): flag closure amplitudes with snr lower than this

               ang_unit (str): phase unit 'deg' or 'rad'
               timetype (str): 'GMST' or 'UTC'
               axis (matplotlib.axes.Axes): add plot to this axis
               rangex (list): [xmin, xmax] x-axis limits
               rangey (list): [ymin, ymax] y-axis limits

               color (str): color for scatterplot points
               marker (str): matplotlib plot marker
               markersize (int): size of plot markers
               label (str): plot legend label

               grid (bool): Plot gridlines if True
               ebar (bool): Plot error bars if True
               axislabels (bool): Show axis labels if True
               legend (bool): Show legend if True

               show (bool): Display the plot if True
               export_pdf (str): path to pdf file to save figure

           Returns:
               (matplotlib.axes.Axes): Axes object with data plot
        """

        if timetype is False:
            timetype = self.timetype
        if ang_unit == 'deg':
            angle = 1.0
        else:
            angle = ehc.DEGREE

        if label is None:
            label = str(self.source)
        else:
            label = str(label)

        # Get closure phases (maximal set)
        if (len(cphases) == 0) and (self.cphase is not None) and not force_recompute:
            cphases = self.cphase

        cpdata = self.cphase_tri(site1, site2, site3, vtype=vtype, timetype=timetype,
                                 cphases=cphases, force_recompute=force_recompute, snrcut=snrcut)
        plotdata = np.array([[obs['time'], obs['cphase'] * angle, obs['sigmacp']]
                             for obs in cpdata])

        nan_mask = np.isnan(plotdata[:, 1])
        plotdata = plotdata[~nan_mask]

        if len(plotdata) == 0:
            print("%s %s %s : No closure phases on this triangle!" % (site1, site2, site3))
            return

        # Plot the data
        if axis:
            x = axis
        else:
            fig = plt.figure()
            x = fig.add_subplot(1, 1, 1)

        # Data ranges
        if not rangex:
            rangex = [self.tstart, self.tstop]
            if np.any(np.isnan(np.array(rangex))):
                print("Warning: NaN in data x range: specifying rangex to default")
                rangex = [0, 24]

        if not rangey:
            if ang_unit == 'deg':
                rangey = [-190, 190]
            else:
                rangey = [-1.1 * np.pi, 1.1 * np.pi]

        x.set_xlim(rangex)
        x.set_ylim(rangey)

        if ebar and np.any(plotdata[:, 2]):
            x.errorbar(plotdata[:, 0], plotdata[:, 1], yerr=plotdata[:, 2],
                       fmt=marker, markersize=markersize,
                       color=color, linestyle='none', label=label)
        else:
            x.plot(plotdata[:, 0], plotdata[:, 1], marker, markersize=markersize,
                   color=color, linestyle='none', label=label)

        if axislabels:
            x.set_xlabel(self.timetype + ' (h)')
            if ang_unit == 'deg':
                x.set_ylabel(r'Closure Phase $(^\circ)$')
            else:
                x.set_ylabel(r'Closure Phase (radian)')

        x.set_title('%s - %s - %s' % (site1, site2, site3))

        if grid:
            x.grid()
        if legend:
            plt.legend()
        if export_pdf != "" and not axis:
            fig.savefig(export_pdf, bbox_inches='tight')
        if export_pdf != "" and axis:
            fig = plt.gcf()
            fig.savefig(export_pdf, bbox_inches='tight')
        if show:
            #plt.show(block=False)
            ehc.show_noblock()

        return x

    def plot_camp(self, site1, site2, site3, site4,
                  vtype='vis', ctype='camp', camps=[], force_recompute=False,
                  debias=False, timetype=False, snrcut=0.,
                  axis=False, rangex=False, rangey=False,
                  color=ehc.SCOLORS[0], marker='o', markersize=ehc.MARKERSIZE, label=None,
                        grid=True, ebar=True, axislabels=True, legend=False,
                        show=True, export_pdf=""):
        """Plot closure amplitude over time on a quadrangle (1-2)(3-4)/(1-4)(2-3).

           Args:
               site1 (str): station 1 name
               site2 (str): station 2 name
               site3 (str): station 3 name
               site4 (str): station 4 name

               vtype (str): The visibilty type  from which to assemble closure amplitudes
                            ('vis','qvis','uvis','vvis','pvis')
               ctype (str): The closure amplitude type ('camp' or 'logcamp')
               camps (list): optionally pass in camps so they don't have to be recomputed
               force_recompute (bool): if True, recompute camps instead of using stored data
               snrcut (float): flag closure amplitudes with snr lower than this

               debias (bool): If True, debias the closure amplitude
               timetype (str): 'GMST' or 'UTC'

               axis (matplotlib.axes.Axes): amake_cdd plot to this axis
               rangex (list): [xmin, xmax] x-axis limits
               rangey (list): [ymin, ymax] y-axis limits
               color (str): color for scatterplot points
               marker (str): matplotlib plot marker
               markersize (int): size of plot markers
               label (str): plot legend label

               grid (bool): Plot gridlines if True
               ebar (bool): Plot error bars if True
               axislabels (bool): Show axis labels if True
               legend (bool): Show legend if True

               show (bool): Display the plot if True
               export_pdf (str): path to pdf file to save figure

           Returns:
               (matplotlib.axes.Axes): Axes object with data plot
        """

        if timetype is False:
            timetype = self.timetype
        if label is None:
            label = str(self.source)

        else:
            label = str(label)

        # Get closure amplitudes (maximal set)
        if ((ctype == 'camp') and (len(camps) == 0) and (self.camp is not None) and
                not (len(self.camp) == 0) and not force_recompute):
            camps = self.camp
        elif ((ctype == 'logcamp') and (len(camps) == 0) and (self.logcamp is not None) and
              not (len(self.logcamp) == 0) and not force_recompute):
            camps = self.logcamp

        # Get closure amplitudes (maximal set)
        cpdata = self.camp_quad(site1, site2, site3, site4,
                                vtype=vtype, ctype=ctype, snrcut=snrcut,
                                debias=debias, timetype=timetype,
                                camps=camps, force_recompute=force_recompute)

        if len(cpdata) == 0:
            print('No closure amplitudes on this triangle!')
            return

        plotdata = np.array([[obs['time'], obs['camp'], obs['sigmaca']] for obs in cpdata])
        plotdata = np.array(plotdata)

        nan_mask = np.isnan(plotdata[:, 1])
        plotdata = plotdata[~nan_mask]

        if len(plotdata) == 0:
            print("No closure amplitudes on this quadrangle!")
            return

        # Data ranges
        if not rangex:
            rangex = [self.tstart, self.tstop]
            if np.any(np.isnan(np.array(rangex))):
                print("Warning: NaN in data x range: specifying rangex to default")
                rangex = [0, 24]

        if not rangey:
            rangey = [np.min(plotdata[:, 1]) - 0.2 * np.abs(np.min(plotdata[:, 1])),
                      np.max(plotdata[:, 1]) + 0.2 * np.abs(np.max(plotdata[:, 1]))]
            if np.any(np.isnan(np.array(rangey))):
                print("Warning: NaN in data y range: specifying rangey to default")
                if ctype == 'camp':
                    rangey = [0, 100]
                if ctype == 'logcamp':
                    rangey = [-10, 10]

        # Plot the data
        if axis:
            x = axis
        else:
            fig = plt.figure()
            x = fig.add_subplot(1, 1, 1)

        if ebar and np.any(plotdata[:, 2]):
            x.errorbar(plotdata[:, 0], plotdata[:, 1], yerr=plotdata[:, 2],
                       fmt=marker, markersize=markersize,
                       color=color, linestyle='none', label=label)
        else:
            x.plot(plotdata[:, 0], plotdata[:, 1], marker, markersize=markersize,
                   color=color, linestyle='none', label=label)

        x.set_xlim(rangex)
        x.set_ylim(rangey)

        if axislabels:
            x.set_xlabel(self.timetype + ' (h)')
            if ctype == 'camp':
                x.set_ylabel('Closure Amplitude')
            elif ctype == 'logcamp':
                x.set_ylabel('Log Closure Amplitude')
            x.set_title('(%s - %s)(%s - %s)/(%s - %s)(%s - %s)' % (site1, site2, site3, site4,
                                                                   site1, site4, site2, site3))
        if grid:
            x.grid()
        if legend:
            plt.legend()
        if export_pdf != "" and not axis:
            fig.savefig(export_pdf, bbox_inches='tight')
        if export_pdf != "" and axis:
            fig = plt.gcf()
            fig.savefig(export_pdf, bbox_inches='tight')
        if show:
            #plt.show(block=False)
            ehc.show_noblock()
            return
        else:
            return x

    def save_txt(self, fname):
        """Save visibility data to a text file.

           Args:
                fname (str): path to output text file
        """

        ehtim.io.save.save_obs_txt(self, fname)

        return

    def save_uvfits(self, fname, force_singlepol=False, polrep_out='circ'):
        """Save visibility data to uvfits file.
           Args:
                fname (str): path to output uvfits file.
                force_singlepol (str): if 'R' or 'L', will interpret stokes I field as 'RR' or 'LL'
                polrep_out (str): 'circ' or 'stokes': how data should be stored in the uvfits file
        """

        if (force_singlepol is not False) and (self.polrep != 'stokes'):
            raise Exception(
                "force_singlepol is incompatible with polrep!='stokes'")

        output = ehtim.io.save.save_obs_uvfits(self, fname,
                                               force_singlepol=force_singlepol, polrep_out=polrep_out)

        return

    def make_hdulist(self, force_singlepol=False, polrep_out='circ'):
        """Returns an hdulist in the same format as in a saved .uvfits file.
           Args:
                force_singlepol (str): if 'R' or 'L', will interpret stokes I field as 'RR' or 'LL'
                polrep_out (str): 'circ' or 'stokes': how data should be stored in the uvfits file
           Returns:
                hdulist (astropy.io.fits.HDUList)
        """

        if (force_singlepol is not False) and (self.polrep != 'stokes'):
            raise Exception(
                "force_singlepol is incompatible with polrep!='stokes'")

        hdulist = ehtim.io.save.save_obs_uvfits(self, None,
                                                force_singlepol=force_singlepol, polrep_out=polrep_out)
        return hdulist


    def save_oifits(self, fname, flux=1.0):
        """ Save visibility data to oifits. Polarization data is NOT saved.

            Args:
                fname (str): path to output text file
                flux (float): normalization total flux
        """

        if self.polrep != 'stokes':
            raise Exception("save_oifits not yet implemented for polreps other than 'stokes'")

        # Antenna diameters are currently incorrect
        # the exact times are also not correct in the datetime object
        ehtim.io.save.save_obs_oifits(self, fname, flux=flux)

        return

##################################################################################################
# Observation creation functions
##################################################################################################


def merge_obs(obs_List, force_merge=False):
    """Merge a list of observations into a single observation file.

       Args:
           obs_List (list): list of split observation Obsdata objects.
           force_merge (bool): forces the observations to merge even if parameters are different

       Returns:
           mergeobs (Obsdata): merged Obsdata object containing all scans in input list
    """

    if (len(set([obs.polrep for obs in obs_List])) > 1):
        raise Exception("All observations must have the same polarization representation !")
        return

    if np.any([obs.timetype == 'GMST' for obs in obs_List]):
        raise Exception("merge_obs only works for observations with obs.timetype='UTC'!")
        return

    if not force_merge:
        if (len(set([obs.ra for obs in obs_List])) > 1 or
            len(set([obs.dec for obs in obs_List])) > 1 or
            len(set([obs.rf for obs in obs_List])) > 1 or
            len(set([obs.bw for obs in obs_List])) > 1 or
                len(set([obs.source for obs in obs_List])) > 1):
            # or  len(set([np.floor(obs.mjd) for obs in obs_List])) > 1):

            raise Exception("All observations must have the same parameters!")
            return

    # the reference observation is the one with the minimum mjd
    obs_idx = np.argmin([obs.mjd for obs in obs_List])
    obs_ref = obs_List[obs_idx]

    # re-reference times to new mjd
    # must be in UTC!
    mjd_ref = obs_ref.mjd
    for obs in obs_List:
        mjd_offset = obs.mjd - mjd_ref
        obs.data['time'] += mjd_offset * 24
        if not(obs.scans is None or len(obs.scans)==0):
            obs.scans += mjd_offset * 24

    # merge the data
    data_merge = np.hstack([obs.data for obs in obs_List])

    # merge the scan list
    scan_merge = None
    for obs in obs_List:
        if (obs.scans is None or len(obs.scans)==0):
            continue
        if (scan_merge is None or len(scan_merge)==0):
            scan_merge = [obs.scans]
        else:
            scan_merge.append(obs.scans)

    if not (scan_merge is None or len(scan_merge) == 0):
        scan_merge = np.vstack(scan_merge)
        _idxsort = np.argsort(scan_merge[:, 0])
        scan_merge = scan_merge[_idxsort]

    # merge the list of telescopes
    tarr_merge = np.unique(np.concatenate([obs.tarr for obs in obs_List]))

    arglist, argdict = obs_ref.obsdata_args()
    arglist[DATPOS] = data_merge
    arglist[TARRPOS] = tarr_merge
    argdict['scantable'] = scan_merge
    mergeobs = Obsdata(*arglist, **argdict)

    return mergeobs


def load_txt(fname, polrep='stokes'):
    """Read an observation from a text file.

       Args:
           fname (str): path to input text file
           polrep (str): load data as either 'stokes' or 'circ'

       Returns:
           obs (Obsdata): Obsdata object loaded from file
    """

    return ehtim.io.load.load_obs_txt(fname, polrep=polrep)


def load_uvfits(fname, flipbl=False, remove_nan=False, force_singlepol=None,
                channel=all, IF=all, polrep='stokes', allow_singlepol=True,
                ignore_pzero_date=True,
                trial_speedups=False):
    """Load observation data from a uvfits file.

       Args:
           fname (str or HDUList): path to input text file or HDUList object
           flipbl (bool): flip baseline phases if True.
           remove_nan (bool): True to remove nans from missing polarizations
           polrep (str): load data as either 'stokes' or 'circ'
           force_singlepol (str): 'R' or 'L' to load only 1 polarization
           channel (list): list of channels to average in the import. channel=all averages all
           IF (list): list of IFs to  average in  the import. IF=all averages all IFS
           remove_nan (bool): whether or not to remove entries with nan data
           
           ignore_pzero_date (bool): if True, ignore the offset parameters in DATE field 
                                     TODO: what is the correct behavior per AIPS memo 117?
       Returns:
           obs (Obsdata): Obsdata object loaded from file
    """

    return ehtim.io.load.load_obs_uvfits(fname, flipbl=flipbl, force_singlepol=force_singlepol,
                                         channel=channel, IF=IF, polrep=polrep,
                                         remove_nan=remove_nan, allow_singlepol=allow_singlepol,
                                         ignore_pzero_date=ignore_pzero_date,
                                         trial_speedups=trial_speedups)


def load_oifits(fname, flux=1.0):
    """Load data from an oifits file. Does NOT currently support polarization.

       Args:
           fname (str): path to input text file
           flux (float): normalization total flux

       Returns:
           obs (Obsdata): Obsdata object loaded from file
    """

    return ehtim.io.load.load_obs_oifits(fname, flux=flux)


def load_maps(arrfile, obsspec, ifile, qfile=0, ufile=0, vfile=0,
              src=ehc.SOURCE_DEFAULT, mjd=ehc.MJD_DEFAULT, ampcal=False, phasecal=False):
    """Read an observation from a maps text file and return an Obsdata object.

       Args:
           arrfile (str): path to input array file
           obsspec (str): path to input obs spec file
           ifile (str): path to input Stokes I data file
           qfile (str): path to input Stokes Q data file
           ufile (str): path to input Stokes U data file
           vfile (str): path to input Stokes V data file
           src (str): source name
           mjd (int): integer observation  MJD
           ampcal (bool): True if amplitude calibrated
           phasecal (bool): True if phase calibrated

       Returns:
           obs (Obsdata): Obsdata object loaded from file
    """

    return ehtim.io.load.load_obs_maps(arrfile, obsspec, ifile,
                                       qfile=qfile, ufile=ufile, vfile=vfile,
                                       src=src, mjd=mjd, ampcal=ampcal, phasecal=phasecal)

def load_obs(
                fname,                  
                polrep='stokes',        
                flipbl=False,               
                remove_nan=False, 
                force_singlepol=None, 
                channel=all, 
                IF=all, 
                allow_singlepol=True,
                flux=1.0,
                obsspec=None, 
                ifile=None, 
                qfile=None, 
                ufile=None, 
                vfile=None,
                src=ehc.SOURCE_DEFAULT, 
                mjd=ehc.MJD_DEFAULT, 
                ampcal=False, 
                phasecal=False
    ):
    """Smart obs read-in, detects file type and loads appropriately.

       Args:
           fname (str): path to input text file
           polrep (str): load data as either 'stokes' or 'circ'
           flipbl (bool): flip baseline phases if True.
           remove_nan (bool): True to remove nans from missing polarizations
           polrep (str): load data as either 'stokes' or 'circ'
           force_singlepol (str): 'R' or 'L' to load only 1 polarization
           channel (list): list of channels to average in the import. channel=all averages all
           IF (list): list of IFs to  average in  the import. IF=all averages all IFS
           flux (float): normalization total flux
           obsspec (str): path to input obs spec file
           ifile (str): path to input Stokes I data file
           qfile (str): path to input Stokes Q data file
           ufile (str): path to input Stokes U data file
           vfile (str): path to input Stokes V data file
           src (str): source name
           mjd (int): integer observation  MJD
           ampcal (bool): True if amplitude calibrated
           phasecal (bool): True if phase calibrated

       Returns:
           obs (Obsdata): Obsdata object loaded from file
    """


    ## grab file ending ##
    fname_extension = fname.split('.')[-1]
    print(f"Extension is {fname_extension}.")

    ## check extension ##
    if fname_extension.lower() == 'uvfits':
        return load_uvfits(fname, flipbl=flipbl, remove_nan=remove_nan, force_singlepol=force_singlepol, channel=channel, IF=IF, polrep=polrep, allow_singlepol=allow_singlepol)

    elif fname_extension.lower() in ['txt', 'text']:
        return load_txt(fname, polrep=polrep)

    elif fname_extension.lower() == 'oifits':
        return load_oifits(fname, flux=flux)


    else:
        if obsspec is not None and ifile is None:
            print("You have provided a value for <obsspec> but no value for <ifile>")
            return 
        elif obsspec is None and ifile is not None:
            print("You have provided a value for <ifile> but no value for <obsspec>")
            return 

        elif obsspec is not None and ifile is not None:
            return load_maps(fname, obsspec, ifile, qfile=qfile, ufile=ufile, vfile=vfile,
              src=src, mjd=mjd, ampcal=ampcal, phasecal=phasecal)

