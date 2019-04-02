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

import string, copy
import numpy as np
import numpy.lib.recfunctions as rec
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as opt
import itertools as it
import sys
import copy

import ehtim.image
import ehtim.io.save
import ehtim.io.load
import ehtim.observing.obs_simulate as simobs

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *
from ehtim.statistics.dataframes import *
from ehtim.statistics.stats import *
import scipy.spatial as spatial
import scipy.optimize as opt

import warnings
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

RAPOS = 0
DECPOS = 1
RFPOS = 2
BWPOS =  3
DATPOS =  4
TARRPOS =  5

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

           amp (numpy.recarray): An array of saved (averaged) visibility amplitudes
           bispec (numpy.recarray): An array of saved (averaged) bispectra
           cphase (numpy.recarray): An array of saved (averaged) closure phases
           camp (numpy.recarray): An array of saved (averaged) closure amplitudes
           logcamp (numpy.recarray): An array of saved (averaged) log closure amplitudes
    """

    def __init__(self, ra, dec, rf, bw, datatable, tarr, scantable=None,
                       polrep='stokes', source=SOURCE_DEFAULT, mjd=MJD_DEFAULT, timetype='UTC',
                       ampcal=True, phasecal=True, opacitycal=True, dcal=True, frcal=True):

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
        if not (datatable.dtype in [DTPOL_STOKES, DTPOL_CIRC]):
            raise Exception("Data table dtype should be DTPOL_STOKES or DTPOL_CIRC")

        # Polarization Representation
        if polrep=='stokes':
            self.polrep = 'stokes'
            self.poldict = POLDICT_STOKES
            self.poltype = DTPOL_STOKES
        elif polrep=='circ':
            self.polrep = 'circ'
            self.poldict = POLDICT_CIRC
            self.poltype = DTPOL_CIRC
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
            raise Exception("timetype must by 'GMST' or 'UTC'")
        self.timetype = timetype

        # Save the data
        self.data = datatable
        self.scans = scantable

        # Telescope array: default ordering is by sefd
        self.tarr =  tarr
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}
        if np.any(self.tarr['sefdr']!=0) or np.any(self.tarr['sefdl']!=0):
            self.reorder_tarr_sefd()
        self.reorder_baselines()

        # Get tstart, mjd and tstop
        times = self.unpack(['time'])['time']
        self.tstart = times[0]
        self.mjd = int(mjd)
        self.tstop = times[-1]
        if self.tstop < self.tstart:
            self.tstop += 24.0

        # Saved closure quantity arrays
        # TODO should we precompute these?
        self.amp=None
        self.bispec=None
        self.cphase=None
        self.camp=None
        self.logcamp=None

    def obsdata_args(self):
        """"Copy arguments for making a  new obsdata argument into a list and dictonary
        """

        arglist = [self.ra, self.dec, self.rf, self.bw,  self.data, self.tarr]
        argdict = {'scantable': self.scans, 'polrep': self.polrep,  'source':self.source, 
                   'mjd': self.mjd, 'timetype': self.timetype, 
                   'ampcal':self.ampcal, 'phasecal':self.phasecal, 'opacitycal':self.opacitycal, 
                   'dcal':self.dcal, 'frcal':self.frcal}
        return (arglist, argdict)


    def copy(self):

        """Copy the observation object.

           Args:

           Returns:
               (Obsdata): a copy of the Obsdata object.
        """

        newobs = copy.deepcopy(self)

        #arglist, argdict = self.obsdata_args()
        #newobs =  Obsdata(*arglist, **argdict)

        ## copy over any precomputed tables
        #newobs.amp = self.amp
        #newobs.bispec = self.bispec
        #newobs.cphase = self.cphase
        #newobs.camp = self.camp
        #newobs.logcamp = self.logcamp

        return newobs

    def switch_polrep(self, polrep_out='stokes', allow_singlepol=True, singlepol_hand='R'):

        """Return a new observation with the polarization representation changed

           Args:
               polrep_out (str):  the polrep of the output data
               allow_singlepol (bool): If True, treat single-polarization data as Stokes I when converting from 'circ' polrep to 'stokes'
               singlepol_hand (str): 'R' or 'L'; determines which parallel-hand is assumed when converting 'stokes' to 'circ' when only I is present

           Returns:
               (Obsdata): new Obsdata object with potentially different polrep
        """

        if polrep_out not in ['stokes','circ']:
            raise Exception("polrep_out must be either 'stokes' or 'circ'")
        if polrep_out==self.polrep:
            return self.copy()
        elif polrep_out=='stokes': #circ -> stokes
            data = np.empty(len(self.data), dtype=DTPOL_STOKES)
            rrmask = np.isnan(self.data['rrvis'])
            llmask = np.isnan(self.data['llvis'])

            for f in DTPOL_STOKES:
                f = f[0]
                if f in ['time','tint','t1', 't2', 'tau1', 'tau2','u','v']:
                    data[f] = self.data[f]
                elif f=='vis':
                    data[f] = 0.5*(self.data['rrvis'] + self.data['llvis'])
                elif f=='qvis':
                    data[f] = 0.5*(self.data['lrvis'] + self.data['rlvis'])
                elif f=='uvis':
                    data[f] = 0.5j*(self.data['lrvis'] - self.data['rlvis'])
                elif f=='vvis':
                    data[f] = 0.5*(self.data['rrvis'] - self.data['llvis'])
                elif f in ['sigma','vsigma']:
                    data[f] = 0.5*np.sqrt(self.data['rrsigma']**2 + self.data['llsigma']**2)
                elif f in ['qsigma','usigma']:
                    data[f] = 0.5*np.sqrt(self.data['rlsigma']**2 + self.data['lrsigma']**2)

            if allow_singlepol:
                # In cases where only one polarization is present, use it as an estimator for Stokes I
                data['vis'][rrmask]   = self.data['llvis'][rrmask]
                data['sigma'][rrmask] = self.data['llsigma'][rrmask]

                data['vis'][llmask]   = self.data['rrvis'][llmask]
                data['sigma'][llmask] = self.data['rrsigma'][llmask]

        elif polrep_out=='circ': #stokes -> circ
            data = np.empty(len(self.data), dtype=DTPOL_CIRC)
            Vmask = np.isnan(self.data['vvis'])

            for f in DTPOL_CIRC:
                f = f[0]
                if f in ['time','tint','t1', 't2', 'tau1', 'tau2','u','v']:
                    data[f] = self.data[f]
                elif f=='rrvis':
                    data[f] = (self.data['vis'] + self.data['vvis'])
                elif f=='llvis':
                    data[f] = (self.data['vis'] - self.data['vvis'])
                elif f=='rlvis':
                    data[f] = (self.data['qvis'] + 1j*self.data['uvis'])
                elif f=='lrvis':
                    data[f] = (self.data['qvis'] - 1j*self.data['uvis'])
                elif f in ['rrsigma','llsigma']:
                    data[f] = np.sqrt(self.data['sigma']**2 + self.data['vsigma']**2)
                elif f in ['rlsigma','lrsigma']:
                    data[f] = np.sqrt(self.data['qsigma']**2 + self.data['usigma']**2)

            if allow_singlepol:
                # In cases where only Stokes I is present, copy it to a specified parallel-hand
                prefix = singlepol_hand.lower() + singlepol_hand.lower() # rr or ll
                if prefix not in ['rr','ll']:
                    raise Exception('singlepol_hand must be R or L')

                data[prefix + 'vis'][Vmask]   = self.data['vis'][Vmask]
                data[prefix + 'sigma'][Vmask] = self.data['sigma'][Vmask]

        arglist, argdict = self.obsdata_args()
        arglist[DATPOS] = data
        argdict['polrep'] = polrep_out
        newobs =  Obsdata(*arglist, **argdict)

        return newobs

    def reorder_baselines(self):

        """Reorder baselines to match uvfits convention, based on the telescope array ordering
        """

        # Time partition the datatable
        datatable = self.data.copy()
        datalist = []
        for key, group in it.groupby(datatable, lambda x: x['time']):
            datalist.append(np.array([obs for obs in group]))

        # Remove conjugate baselines
        obsdata = []
        for tlist in datalist:
            blpairs = []
            for dat in tlist:
                if not (set((dat['t1'], dat['t2']))) in blpairs:

                     # Reverse the baseline in the right order for uvfits:
                     if(self.tkey[dat['t2']] < self.tkey[dat['t1']]):

                        (dat['t1'], dat['t2']) = (dat['t2'], dat['t1'])
                        (dat['tau1'], dat['tau2']) = (dat['tau2'], dat['tau1'])
                        dat['u'] = -dat['u']
                        dat['v'] = -dat['v']

                        if self.polrep=='stokes':
                            dat['vis'] = np.conj(dat['vis'])
                            dat['qvis'] = np.conj(dat['qvis'])
                            dat['uvis'] = np.conj(dat['uvis'])
                            dat['vvis'] = np.conj(dat['vvis'])
                        elif self.polrep=='circ':
                            dat['rrvis'] = np.conj(dat['rrvis'])
                            dat['llvis'] = np.conj(dat['llvis'])
                            #must switch l & r !!
                            rl = dat['rlvis'].copy()
                            lr = dat['lrvis'].copy()
                            dat['rlvis'] = np.conj(lr)
                            dat['lrvis'] = np.conj(rl)
                        else:
                            raise Exception("polrep must be either 'stokes' or 'circ'")

                     # Append the data point
                     blpairs.append(set((dat['t1'],dat['t2'])))
                     obsdata.append(dat)

        obsdata = np.array(obsdata, dtype=self.poltype)

        # Timesort data
        obsdata = obsdata[np.argsort(obsdata, order=['time','t1'])]

        # Save the data
        self.data = obsdata

        return

    def reorder_tarr_sefd(self):

        """Reorder the telescope array by SEFD minimal to maximum.
        """

        sorted_list = sorted(self.tarr, key=lambda x: np.sqrt(x['sefdr']**2 + x['sefdl']**2))
        self.tarr = np.array(sorted_list,dtype=DTARR)
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}
        self.reorder_baselines()

        return

    def reorder_tarr_snr(self):

        """Reorder the telescope array by median SNR maximal to minimal.
        """

        snr = self.unpack(['t1','t2','snr'])
        snr_median = [np.median(snr[(snr['t1']==site) + (snr['t2']==site)]['snr']) for site in self.tarr['site']]
        idx = np.argsort(snr_median)[::-1]
        self.tarr = self.tarr[idx]
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}
        self.reorder_baselines()

        return

    def reorder_tarr_random(self):

        """Randomly reorder the telescope array.
        """

        idx = np.arange(len(self.tarr))
        np.random.shuffle(idx)
        self.tarr = self.tarr[idx]
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}
        self.reorder_baselines()

        return

    def data_conj(self):

        """Make a data array including all conjugate baselines.

           Args:

           Returns:
               (numpy.recarray): a copy of the Obsdata.data table including all conjugate baselines.
        """

        data = np.empty(2*len(self.data), dtype=self.poltype)

        # Add the conjugate baseline data
        for f in self.poltype:
            f = f[0]
            if f in ['t1', 't2', 'tau1', 'tau2']:
                if f[-1]=='1': f2 = f[:-1]+'2'
                else: f2 = f[:-1]+'1'
                data[f] = np.hstack((self.data[f], self.data[f2]))

            elif f in ['u','v']:
                data[f] = np.hstack((self.data[f], -self.data[f]))

            elif f in [self.poldict['vis1'],self.poldict['vis2'],
                       self.poldict['vis3'],self.poldict['vis4']]:
                if self.polrep=='stokes':
                    data[f] = np.hstack((self.data[f], np.conj(self.data[f])))
                elif self.polrep=='circ':
                    if f in ['rrvis','llvis']:
                        data[f] = np.hstack((self.data[f], np.conj(self.data[f])))
                    elif f=='rlvis':
                        data[f] = np.hstack((self.data['rlvis'], np.conj(self.data['lrvis'])))
                    elif f=='lrvis':
                        data[f] = np.hstack((self.data['lrvis'], np.conj(self.data['rlvis'])))
                else:
                    raise Exception("polrep must be either 'stokes' or 'circ'")

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
            for key, group in it.groupby(data, lambda x: int(x['time']/(t_gather/3600.0))):
                datalist.append(np.array([obs for obs in group]))
        else:
            # Group measurements by scan
            if np.any(self.scans == None) or len(self.scans) == 0:
                print("No scan table in observation. Adding scan table before gathering...")
                self.add_scans()

            for key, group in it.groupby(data, lambda x: np.searchsorted(self.scans[:,0],x['time'])):
                datalist.append(np.array([obs for obs in group]))

        return np.array(datalist)

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

        return np.array(datalist)

    def unpack_bl(self, site1, site2, fields, ang_unit='deg', debias=False, timetype=False):

        """Unpack the data over time on the selected baseline site1-site2.

           Args:
                site1 (str): First site name
                site2 (str): Second site name
                fields (list): list of unpacked quantities from available quantities in FIELDS
                ang_unit (str): 'deg' for degrees and 'rad' for radian phases
                debias (bool): True to debias visibility amplitudes
                timetype (str): 'GMST' or 'UTC'

           Returns:
                (numpy.recarray): unpacked numpy array with data in fields requested
        """

        if timetype==False:
            timetype=self.timetype

        # If we only specify one field
        if timetype not  in ['GMST','UTC','utc','gmst']:
            raise Exception("timetype should be 'GMST' or 'UTC'!")
        allfields = ['time']

        if not isinstance(fields, list): allfields.append(fields)
        else:
            for i in range(len(fields)): allfields.append(fields[i])

        # Get the data from data table on the selected baseline
        allout = []
        tlist = self.tlist(conj=True)
        for scan in tlist:
            for obs in scan:
                if (obs['t1'], obs['t2']) == (site1, site2):
                    obs = np.array([obs])
                    out = self.unpack_dat(obs, allfields, ang_unit=ang_unit, debias=debias, timetype=timetype)

                    allout.append(out)

        #if len(allout)==0:
            #raise Exception("Baseline %s-%s has no data!"%(site1,site2))

        return np.array(allout)

    def unpack(self, fields, mode='all', ang_unit='deg',  debias=False, conj=False, timetype=False):

        """Unpack the data for the whole observation .

           Args:
                fields (list): list of unpacked quantities from availalbe quantities in FIELDS
                mode (str): 'all' returns all data in single table, 'time' groups output by equal time, 'bl' groups by baseline
                ang_unit (str): 'deg' for degrees and 'rad' for radian phases
                debias (bool): True to debias visibility amplitudes
                conj (bool): True to include conjugate baselines
                timetype (str): 'GMST' or 'UTC'

           Returns:
                (numpy.recarray): unpacked numpy array with data in fields requested

        """

        if not mode in ('time', 'all', 'bl'):
            raise Exception("possible options for mode are 'time', 'all' and 'bl'")

        # If we only specify one field
        if not isinstance(fields, list): fields = [fields]

        if mode=='all':
            if conj:
                data = self.data_conj()
            else:
                data = self.data
            allout=self.unpack_dat(data, fields, ang_unit=ang_unit, debias=debias,timetype=timetype)

        elif mode=='time':
            allout=[]
            tlist = self.tlist(conj=True)
            for scan in tlist:
                out=self.unpack_dat(scan, fields, ang_unit=ang_unit, debias=debias,timetype=timetype)
                allout.append(out)

        elif mode=='bl':
            allout = []
            bllist = self.bllist()
            for bl in bllist:
                out = self.unpack_dat(bl, fields, ang_unit=ang_unit, debias=debias,timetype=timetype)
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
                timetype (str): 'GMST' or 'UTC'

           Returns:
                (numpy.recarray): unpacked numpy array with data in fields requested

        """

        if ang_unit=='deg': angle=DEGREE
        else: angle = 1.0

        # If we only specify one field
        if type(fields) is str:
            fields = [fields]

        if not timetype:
            timetype=self.timetype
        if timetype not  in ['GMST','UTC','gmst','utc']:
            raise Exception("timetype should be 'GMST' or 'UTC'!")

        # Get field data
        allout = []
        for field in fields:
            if field in ["time","time_utc","time_gmst"]:
                out = data['time']
                ty='f8'
            elif field in ["u","v","tint","tau1","tau2"]:
                out = data[field]
                ty = 'f8'
            elif field in ["uvdist"]:
                out = np.abs(data['u'] + 1j * data['v'])
                ty = 'f8'
            elif field in ["t1","el1","par_ang1","hr_ang1"]:
                sites = data["t1"]
                keys = [self.tkey[site] for site in sites]
                tdata = self.tarr[keys]
                out = sites
                ty = 'U32'
            elif field in ["t2","el2","par_ang2","hr_ang2"]:
                sites = data["t2"]
                keys = [self.tkey[site] for site in sites]
                tdata = self.tarr[keys]
                out = sites
                ty = 'U32'
            elif field in ['vis','amp','phase','snr','sigma','sigma_phase']:
                ty = 'c16'
                if self.polrep=='stokes':
                    out = data['vis']
                    sig = data['sigma']
                elif self.polrep=='circ':
                    out = 0.5*(data['rrvis'] + data['llvis'])
                    sig = 0.5*np.sqrt(data['rrsigma']**2 + data['llsigma']**2)
            elif field in ['qvis','qamp','qphase','qsnr','qsigma','qsigma_phase']:
                ty = 'c16'
                if self.polrep=='stokes':
                    out = data['qvis']
                    sig = data['qsigma']
                elif self.polrep=='circ':
                    out = 0.5*(data['lrvis'] + data['rlvis'])
                    sig = 0.5*np.sqrt(data['lrsigma']**2 + data['rlsigma']**2)
            elif field in ['uvis','uamp','uphase','usnr','usigma','usigma_phase']:
                ty = 'c16'
                if self.polrep=='stokes':
                    out = data['uvis']
                    sig = data['usigma']
                elif self.polrep=='circ':
                    out = 0.5j*(data['lrvis'] - data['rlvis'])
                    sig = 0.5*np.sqrt(data['lrsigma']**2 + data['rlsigma']**2)
            elif field in ['vvis','vamp','vphase','vsnr','vsigma','vsigma_phase']:
                ty = 'c16'
                if self.polrep=='stokes':
                    out = data['vvis']
                    sig = data['vsigma']
                elif self.polrep=='circ':
                    out = 0.5*(data['rrvis'] - data['llvis'])
                    sig = 0.5*np.sqrt(data['rrsigma']**2 + data['llsigma']**2)
            elif field in ['pvis','pamp','pphase','psnr','psigma','psigma_phase']:
                ty = 'c16'
                if self.polrep=='stokes':
                    out = data['qvis'] + 1j * data['uvis']
                    sig = np.sqrt(data['qsigma']**2 + data['usigma']**2)
                elif self.polrep=='circ':
                    out = data['rlvis']
                    sig = data['rlsigma']
            elif field in ['m','mamp','mphase','msnr','msigma','msigma_phase']:
                ty = 'c16'
                if self.polrep=='stokes':
                    out = (data['qvis'] + 1j * data['uvis'])/data['vis']
                    sig = merr(data['sigma'], data['qsigma'], data['usigma'], data['vis'], out)
                elif self.polrep=='circ':
                    out = 2 * data['rlvis'] / (data['rrvis'] + data['llvis'])
                    sig = merr2(data['rlsigma'], data['rrsigma'], data['llsigma'], 0.5*(data['rrvis']+data['llvis']), out)
            elif field in ['rrvis', 'rramp', 'rrphase', 'rrsnr', 'rrsigma', 'rrsigma_phase']:
                ty = 'c16'
                if self.polrep=='stokes':
                    out = data['vis'] + data['vvis']
                    sig = np.sqrt(data['sigma']**2 + data['vsigma']**2)
                elif self.polrep=='circ':
                    out = data['rrvis']
                    sig = data['rrsigma']
            elif field in ['llvis', 'llamp', 'llphase', 'llsnr', 'llsigma', 'llsigma_phase']:
                ty = 'c16'
                if self.polrep=='stokes':
                    out = data['vis'] - data['vvis']
                    sig = np.sqrt(data['sigma']**2 + data['vsigma']**2)
                elif self.polrep=='circ':
                    out = data['llvis']
                    sig = data['llsigma']
            elif field in ['rlvis', 'rlamp', 'rlphase', 'rlsnr', 'rlsigma', 'rlsigma_phase']:
                ty = 'c16'
                if self.polrep=='stokes':
                    out = data['qvis'] + 1j*data['uvis']
                    sig = np.sqrt(data['qsigma']**2 + data['usigma']**2)
                elif self.polrep=='circ':
                    out = data['rlvis']
                    sig = data['rlsigma']
            elif field in ['lrvis', 'lramp', 'lrphase', 'lrsnr', 'lrsigma', 'lrsigma_phase']:
                ty = 'c16'
                if self.polrep=='stokes':
                    out = data['qvis'] - 1j*data['uvis']
                    sig = np.sqrt(data['qsigma']**2 + data['usigma']**2)
                elif self.polrep=='circ':
                    out = data['lrvis']
                    sig = data['lrsigma']

            else: raise Exception("%s is not a valid field \n" % field +
                                  "valid field values are: " + ' '.join(FIELDS))

            if field in ["time_utc"] and timetype=='GMST':
                out = gmst_to_utc(out, self.mjd)
            if field in ["time_gmst"] and timetype=='UTC':
                out = utc_to_gmst(out, self.mjd)

            # Compute elevation and parallactic angles
            if field in ["el1","el2","hr_ang1","hr_ang2","par_ang1","par_ang2"]:
                if self.timetype=='GMST':
                    times_sid = data['time']
                else:
                    times_sid = utc_to_gmst(data['time'], self.mjd)

                thetas = np.mod((times_sid - self.ra)*HOUR, 2*np.pi)
                coords = recarr_to_ndarr(tdata[['x','y','z']],'f8')
                el_angle = elev(earthrot(coords, thetas), self.sourcevec())
                latlon = xyz_2_latlong(coords)
                hr_angles = hr_angle(times_sid*HOUR, latlon[:,1], self.ra*HOUR)

                if field in ["el1","el2"]:
                    out = el_angle/angle
                    ty  = 'f8'
                if field in ["hr_ang1","hr_ang2"]:
                    out = hr_angles/angle
                    ty  = 'f8'
                if field in ["par_ang1","par_ang2"]:
                    par_ang = par_angle(hr_angles, latlon[:,0], self.dec*DEGREE)
                    out = par_ang/angle
                    ty  = 'f8'

            # Get arg/amps/snr
            if field in ["amp", "qamp", "uamp","vamp","pamp","mamp","rramp","llamp","rlamp","lramp"]:
                out = np.abs(out)
                if debias:
                    out = amp_debias(out, sig)
                ty = 'f8'
            elif field in ["sigma","qsigma","usigma","vsigma","psigma","msigma",
                           "rrsigma","llsigma","rlsigma","lrsigma"]:
                out = np.abs(sig)
                ty = 'f8'
            elif field in ["phase", "qphase", "uphase", "vphase","pphase",
                           "mphase","rrphase","llphase","lrphase","rlphase"]:
                out = np.angle(out)/angle
                ty = 'f8'
            elif field in ["sigma_phase","qsigma_phase","usigma_phase",
                           "vsigma_phase","psigma_phase","msigma_phase",
                           "rrsigma_phase","llsigma_phase","rlsigma_phase","lrsigma_phase"]:
                out = np.abs(sig)/np.abs(out)/angle
                ty = 'f8'
            elif field in ["snr", "qsnr", "usnr", "vsnr", "psnr", "msnr","rrsnr","llsnr","rlsnr","lrsnr"]:
                out = np.abs(out)/np.abs(sig)
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

        sourcevec = np.array([np.cos(self.dec*DEGREE), 0, np.sin(self.dec*DEGREE)])

        return sourcevec

    def res(self):

        """Return the nominal resolution (1/longest baseline) of the observation in radians.

           Args:

           Returns:
                (float): normal array resolution in radians
        """

        res = 1.0/np.max(self.unpack('uvdist')['uvdist'])

        return res

    def split_obs(self):

        """Split single observation into multiple observation files, one per scan.

           Args:

           Returns:
                (list): list of single-scan Obsdata objects
        """

        print("Splitting Observation File into " + str(len(self.tlist())) + " scans")
        arglist, argdict = self.obsdata_args()

        # note that the tarr of the output includes all sites, even those that don't participate in the scan
        splitlist = []
        for tdata in self.tlist():
            arglist[DATPOS] =  tdata
            splitlist.append(Obsdata(*arglist, **argdict))

        return splitlist

    def chisq(self, im, dtype='vis', pol='I', ttype='nfft', mask=[], **kwargs):
              #cp_uv_min=False,
              #debias=True, systematic_noise=0.0, systematic_cphase_noise=0.0, maxset=False,
              #ttype='nfft',fft_pad_factor=2):

        """Give the reduced chi^2 of the observation for the specified image and datatype.

           Args:
                im (Image): image to test chi^2
                dtype (str): data type of chi^2 (e.g., 'vis', 'amp', 'bs', 'cphase')
                pol (str): polarization type ('I', 'Q', 'U', 'V', 'LL', 'RR', 'LR', or 'RL'
                mask (arr): mask of same dimension as im.imvec to screen out pixels in chi^2 computation

                ttype (str): if "fast" or "nfft" use FFT to produce visibilities. Else "direct" for DTFT
                fft_pad_factor (float): zero pad the image to fft_pad_factor * image size in FFT
                conv_func ('str'):  The convolving function for gridding; options are 'gaussian', 'pill', and 'cubic'
                p_rad (int): The pixel radius for the convolving function in gridding for FFTs
                order ('str'): Interpolation order for sampling the FFT        
    
                systematic_noise (float): a fractional systematic noise tolerance to add to thermal sigmas
                snrcut (float): a  snr cutoff for including data in the chi^2 sum
                debias (bool): if True then apply debiasing to amplitudes/closure amplitudes
                weighting (str): 'natural' or 'uniform' 

                systematic_cphase_noise (float): a value in degrees to add to the closure phase sigmas
                cp_uv_min (float): flag baselines shorter than this before forming closure quantities
                maxset (bool):  if True, use maximal set instead of minimal for closure quantities

           Returns:
                (float): image chi^2
        """

        # TODO -- should import this at top, but the circular dependencies create a mess...
        import ehtim.imaging.imager_utils as iu
        if pol not in im._imdict.keys():
            raise Exception(pol + ' is not in the current image. Consider changing the polarization basis of the image.')

        (data, sigma, A) = iu.chisqdata(self, im, mask, dtype, pol=pol, ttype=ttype, **kwargs)
        chisq = iu.chisq(im._imdict[pol], A, data, sigma, dtype, ttype=ttype, mask=mask)

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
        print ("Recomputing U,V Points using MJD %d \n RA %e \n DEC %e \n RF %e GHz"
                                       % (self.mjd, self.ra, self.dec, self.rf/1.e9))

        (timesout,uout,vout) = compute_uv_coordinates(arr, site1, site2, times,
                                                      self.mjd, self.ra, self.dec,self.rf, 
                                                      timetype=self.timetype, elevmin=0, elevmax=90)

        if len(timesout) != len(times):
            raise Exception("len(timesout) != len(times) in recompute_uv: check elevation  limits!!")

        datatable = self.data.copy()
        datatable['u'] = uout
        datatable['v'] = vout

        arglist, argdict = self.obsdata_args()
        arglist[DATPOS] = np.array(datatable)
        out = Obsdata(*arglist, **argdict)

        return  out

    def avg_coherent(self, inttime, scan_avg=False):

        """Coherently average data along u,v tracks in chunks of length inttime (sec)

           Args:
                inttime (float): coherent integration time in seconds
                scan_avg (bool): if True, average over scans in self.scans instead of intime

           Returns:
                (Obsdata): Obsdata object containing averaged data
        """

        if (scan_avg==True)&(getattr(self.scans, "shape", None) is None or len(self.scans) == 0):
            print('No scan data, ignoring scan_avg!')
            scan_avg=False

        if inttime <= 0.0 and scan_avg == False:
            print('No averaging done!')
            return self.copy()

        vis_avg = coh_avg_vis(self,dt=inttime,return_type='rec',err_type='predicted',scan_avg=scan_avg)

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
        amp_rec = incoh_avg_vis(self,dt=inttime,debias=debias,scan_avg=scan_avg,return_type='rec',rec_type='vis',err_type=err_type)
        arglist, argdict = self.obsdata_args()
        arglist[DATPOS] = amp_rec
        out = Obsdata(*arglist, **argdict)

        return out

    def add_amp(self, avg_time=0, scan_avg=False, debias=True, err_type='predicted', return_type='rec', round_s=0.1, snrcut=0.):
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

        # Get spacing between datapoints in seconds
        if len(set([x[0] for x in list(self.unpack('time'))])) > 1:
            tint0 = np.min(np.diff(np.asarray(sorted(list(set([x[0] for x in list(self.unpack('time'))]))))))*3600.
        else:
            tint0 = 0.0

        if avg_time <= tint0:
            adf = make_amp(self, debias=debias, round_s=round_s)
            if return_type=='rec':
                adf = df_to_rec(adf,'amp')
            print("Updated self.amp: no averaging")
        else:
            adf = incoh_avg_vis(self,dt=avg_time,debias=debias,scan_avg=scan_avg,
                                return_type=return_type,rec_type='amp',err_type=err_type)

        # snr cut
        adf = adf[adf['amp']/adf['sigma'] > snrcut]
        self.amp = adf
        print("Updated self.amp: avg_time %f s\n"%avg_time)

        return

    def add_bispec(self, avg_time=0, return_type='rec', count='max', snrcut=0.,
                         err_type='predicted', num_samples=1000, round_s=0.1, uv_min=False):

        """Adds attribute self.bispec: bispectra table with bispectra averaged for dt

           Args:
               avg_time (float): bispectrum averaging timescale
               return_type: data frame ('df') or recarray ('rec')
               count (str): If 'min', return minimal set of bispectra, if 'max' return all bispectra up to reordering
               err_type (str): 'predicted' or 'measured'
               num_samples: number of bootstrap (re)samples if measuring error
               round_s (float): accuracy of datetime object in seconds
               snrcut (float): flag bispectra with snr lower than this               

        """

        # Get spacing between datapoints in seconds
        if len(set([x[0] for x in list(self.unpack('time'))])) > 1:
            tint0 = np.min(np.diff(np.asarray(sorted(list(set([x[0] for x in list(self.unpack('time'))]))))))*3600.
        else:
            tint0 = 0


        if avg_time>tint0:
            cdf = make_bsp_df(self, mode='all', round_s=round_s, count=count, snrcut=0., uv_min=uv_min)
            cdf = average_bispectra(cdf,avg_time,return_type=return_type,num_samples=num_samples, snrcut=snrcut)
        else:
            cdf = make_bsp_df(self, mode='all', round_s=round_s, count=count, snrcut=snrcut, uv_min=uv_min)
            print("Updated self.bispec: no averaging")
            if return_type=='rec':
                cdf = df_to_rec(cdf,'bispec')

        self.bispec = cdf
        print("Updated self.bispec: avg_time %f s\n"%avg_time)

        return

    def add_cphase(self,avg_time=0, return_type='rec',count='max', snrcut=0.,
                        err_type='predicted', num_samples=1000, round_s=0.1, uv_min=False):

        """Adds attribute self.cphase: cphase table averaged for dt

           Args:
               avg_time (float): closure phase averaging timescale
               return_type: data frame ('df') or recarray ('rec')
               count (str): If 'min', return minimal set of phases, if 'max' return all closure phases up to reordering
               err_type (str): 'predicted' or 'measured'
               num_samples: number of bootstrap (re)samples if measuring error
               round_s (float): accuracy of datetime object in seconds
               snrcut (float): flag closure phases with snr lower than this               

        """

        # Get spacing between datapoints in seconds
        if len(set([x[0] for x in list(self.unpack('time'))])) > 1:
            tint0 = np.min(np.diff(np.asarray(sorted(list(set([x[0] for x in list(self.unpack('time'))]))))))*3600.
        else:
            tint0 = 0

        if avg_time>tint0:
            cdf = make_cphase_df(self, mode='all', round_s=round_s, count=count, snrcut=0., uv_min=uv_min)
            cdf = average_cphases(cdf, avg_time, return_type=return_type, err_type=err_type, num_samples=num_samples, snrcut=snrcut)
        else:
            cdf = make_cphase_df(self, mode='all', round_s=round_s, count=count, snrcut=snrcut, uv_min=uv_min)
            if return_type=='rec':
                cdf = df_to_rec(cdf,'cphase')
            print("Updated self.cphase: no averaging")

        self.cphase = cdf
        print("updated self.cphase: avg_time %f s\n"%avg_time)

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
               count (str): If 'min', return minimal set of amplitudes, if 'max' return all closure amplitudes up to inverses
               err_type (str): 'predicted' or 'measured'
               num_samples: number of bootstrap (re)samples if measuring error
               round_s (float): accuracy of datetime object in seconds
               snrcut (float): flag closure amplitudes with snr lower than this               
        """

        # Get spacing between datapoints in seconds
        if len(set([x[0] for x in list(self.unpack('time'))])) > 1:
            tint0 = np.min(np.diff(np.asarray(sorted(list(set([x[0] for x in list(self.unpack('time'))]))))))*3600.
        else:
            tint0 = 0

        if avg_time>tint0:
            foo = self.avg_incoherent(avg_time,debias=debias,err_type=err_type)
        else:
            foo = self
        cdf = make_camp_df(foo,ctype=ctype,debias=False,count=count,round_s=round_s,snrcut=snrcut)

        if ctype=='logcamp':
            print("updated self.lcamp: no averaging")
        elif ctype=='camp':
            print("updated self.camp: no averaging")
        if return_type=='rec':
            cdf = df_to_rec(cdf,'camp')

        if ctype=='logcamp':
            self.logcamp = cdf
            print("updated self.logcamp: avg_time %f s\n" % avg_time)
        elif ctype=='camp':
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
               count (str): If 'min', return minimal set of amplitudes, if 'max' return all closure amplitudes up to inverses
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

    def add_all(self, avg_time=0, return_type='rec',
                      count='max', debias=True, snrcut=0.,
                      err_type='predicted', num_samples=1000, round_s=0.1):

        """Adds tables of all all averaged derived quantities self.amp,self.bispec,self.cphase,self.camp,self.logcamp

           Args:
               avg_time (float): closure amplitude averaging timescale
               return_type: data frame ('df') or recarray ('rec')
               debias (bool): If True, debias the closure amplitude
               count (str): If 'min', return minimal set of closure quantities, if 'max' return all closure quantities
               err_type (str): 'predicted' or 'measured'
               num_samples: number of bootstrap (re)samples if measuring error
               round_s (float): accuracy of datetime object in seconds
               snrcut (float): flag closure amplitudes with snr lower than this               

        """

        self.add_amp(return_type=return_type, avg_time=avg_time, debias=debias, err_type=err_type)
        self.add_bispec(return_type=return_type, count=count, avg_time=avg_time, snrcut=snrcut,
                        err_type=err_type, num_samples=num_samples, round_s=round_s)
        self.add_cphase(return_type=return_type, count=count, avg_time=avg_time, snrcut=snrcut,
                        err_type=err_type, num_samples=num_samples, round_s=round_s)
        self.add_camp(return_type=return_type, ctype='camp',
                     count=count, debias=debias, snrcut=snrcut,
                     avg_time=avg_time, err_type=err_type, num_samples=num_samples, round_s=round_s)
        self.add_camp(return_type=return_type, ctype='logcamp',
                     count=count, debias=debias, snrcut=snrcut,
                     avg_time=avg_time, err_type=err_type, num_samples=num_samples, round_s=round_s)

        return

    def add_scans(self, info='self', filepath='', dt=0.0165, margin=0.0001):

        """Compute scans and add self.scans to Obsdata object.

            Args:
                info (str): 'self' for scans inferred from data, 'txt' for text file, 'vex' for vex schedule file
                filepath (str): path to txt/vex file with scans info
                dt (float): minimal time interval between scans in hours
                margin (float): padding scans by that time margin in hours

        """

        if info=='self':
            times_uni = np.asarray(sorted(list(set(self.data['time']))))
            scans = np.zeros_like(times_uni)
            scan_id=0
            for cou in range(len(times_uni)-1):
                scans[cou] = scan_id
                if (times_uni[cou+1]-times_uni[cou] > dt):
                    scan_id+=1
            scans[-1]=scan_id
            scanlist = np.asarray([np.asarray([np.min(times_uni[scans==cou])-margin,np.max(times_uni[scans==cou])+margin]) 
                                   for cou in range(int(scans[-1])+1)])

        elif info=='txt':
             scanlist = np.loadtxt(filepath)

        elif info=='vex':
            vex0 = ehtim.vex.Vex(filepath)
            t_min = [vex0.sched[x]['start_hr'] for x in range(len(vex0.sched))]
            duration=[]
            for x in range(len(vex0.sched)):
                duration_foo =max([vex0.sched[x]['scan'][y]['scan_sec'] for y in range(len(vex0.sched[x]['scan']))])
                duration.append(duration_foo)
            t_max = [tmin + dur/3600. for (tmin,dur) in zip(t_min,duration)]
            scanlist = np.array([[tmin,tmax] for (tmin,tmax) in zip(t_min,t_max)])

        else:
            print("Parameter 'info' can only assume values 'self', 'txt' or 'vex'! ")
            scanlist=None

        self.scans = scanlist

        return

    def cleanbeam(self, npix, fov, pulse=PULSE_DEFAULT):

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
               units (string): 'rad' returns values in radians, 'natural' returns FWHMs in uas and theta in degrees

           Returns:
               (tuple): a tuple (fwhm_maj, fwhm_min, theta) of the dirty beam parameters in radians.
        """

        # Define the sum of squares function that compares the quadratic expansion of the dirty image
        # with the quadratic expansion of an elliptical gaussian
        def fit_chisq(beamparams, db_coeff):

            (fwhm_maj2, fwhm_min2, theta) = beamparams
            a = 4 * np.log(2) * (np.cos(theta)**2/fwhm_min2 + np.sin(theta)**2/fwhm_maj2)
            b = 4 * np.log(2) * (np.cos(theta)**2/fwhm_maj2 + np.sin(theta)**2/fwhm_min2)
            c = 8 * np.log(2) * np.cos(theta) * np.sin(theta) * (1.0/fwhm_maj2 - 1.0/fwhm_min2)
            gauss_coeff = np.array((a,b,c))

            chisq = np.sum((np.array(db_coeff) - gauss_coeff)**2)

            return chisq

        # These are the coefficients (a,b,c) of a quadratic expansion of the dirty beam
        # For a point (x,y) in the image plane, the dirty beam expansion is 1-ax^2-by^2-cxy
        u = self.unpack('u')['u']
        v = self.unpack('v')['v']
        sigma = self.unpack('sigma')['sigma']
        n = float(len(u))
        weights = np.ones(u.shape)
        if weighting == 'natural':
            weights = 1./sigma**2
            
        abc = (2.*np.pi**2/np.sum(weights)) * np.array([np.sum(weights*u**2), np.sum(weights*v**2), 2*np.sum(weights*u*v)])
        abc = 1e-20 * abc # Decrease size of coefficients

        # Fit the beam
        guess = [(50)**2, (50)**2, 0.0]
        params = opt.minimize(fit_chisq, guess, args=(abc,), method='Powell')

        # Return parameters, adjusting fwhm_maj and fwhm_min if necessary
        if params.x[0] > params.x[1]:
            fwhm_maj = 1e-10*np.sqrt(params.x[0])
            fwhm_min = 1e-10*np.sqrt(params.x[1])
            theta = np.mod(params.x[2], np.pi)
        else:
            fwhm_maj = 1e-10*np.sqrt(params.x[1])
            fwhm_min = 1e-10*np.sqrt(params.x[0])
            theta = np.mod(params.x[2] + np.pi/2.0, np.pi)

        gparams = np.array((fwhm_maj, fwhm_min, theta))

        if units == 'natural':
            gparams[0] /= RADPERUAS
            gparams[1] /= RADPERUAS
            gparams[2] *= 180./np.pi

        return gparams


    def dirtybeam(self, npix, fov, pulse=PULSE_DEFAULT, weighting='uniform'):

        """Make an image of the observation dirty beam.

           Args:
               npix (int): The pixel size of the square output image.
               fov (float): The field of view of the square output image in radians.
               pulse (function): The function convolved with the pixel values for continuous image.
               weighting (str): 'uniform' or 'natural'
           Returns:
               (Image): an Image object with the dirty beam.
        """

        pdim = fov/npix
        sigma = self.unpack('sigma')['sigma']
        u = self.unpack('u')['u']
        v = self.unpack('v')['v']
        if weighting=='natural':
            weights = 1./sigma**2
        else:
            weights = np.ones(u.shape)

        xlist = np.arange(0,-npix,-1)*pdim + (pdim*npix)/2.0 - pdim/2.0

        # TODO -- use NFFT
        # TODO -- more different beam weightings
        im = np.array([[np.mean(weights*np.cos(-2*np.pi*(i*u + j*v)))
                  for i in xlist]
                  for j in xlist])

        #weightim = np.mean(weights)
        #weighttotal = np.sum(weightim * np.ones(im.shape))

        im = im[0:npix, 0:npix]
        im = im / np.sum(im) # Normalize to a total beam power of 1
                             # TODO right normalization? 

        src = self.source + "_DB"
        outim = ehtim.image.Image(im, pdim, self.ra, self.dec, rf=self.rf, source=src, mjd=self.mjd, pulse=pulse)

        return outim

    def dirtyimage(self, npix, fov, pulse=PULSE_DEFAULT, weighting='uniform'):

        """Make the observation dirty image (direct Fourier transform).

           Args:
               npix (int): The pixel size of the square output image.
               fov (float): The field of view of the square output image in radians.
               pulse (function): The function convolved with the pixel values for continuous image.
               weighting (str): 'uniform' or 'natural'
           Returns:
               (Image): an Image object with dirty image.
        """

        pdim = fov/npix
        u = self.unpack('u')['u']
        v = self.unpack('v')['v']
        sigma = self.unpack('sigma')['sigma']
        xlist = np.arange(0,-npix,-1)*pdim + (pdim*npix)/2.0 - pdim/2.0
        if weighting=='natural':
            weights = 1./sigma**2
        else:
            weights = np.ones(u.shape)

        dim = np.array([[np.mean(weights*np.cos(-2*np.pi*(i*u + j*v)))
                  for i in xlist]
                  for j in xlist])
        normfac = 1./np.sum(dim)

        for label in ['vis1','vis2','vis3','vis4']:
            visname = self.poldict[label]
    
            vis = self.unpack(visname)[visname]

            # TODO -- use NFFT
            # TODO -- different beam weightings
            im  = np.array([[np.mean(weights * (np.real(vis)*np.cos(-2*np.pi*(i*u + j*v)) -
                                               np.imag(vis)*np.sin(-2*np.pi*(i*u + j*v))))
                              for i in xlist]
                              for j in xlist])

 
            # Final normalization
            im  = im * normfac
            im = im[0:npix, 0:npix]
            
            if label=='vis1':
                out = ehtim.image.Image(im, pdim, self.ra, self.dec, polrep=self.polrep,rf=self.rf, source=self.source, mjd=self.mjd, pulse=pulse)
            else:
                pol = {vis_poldict[key]: key for key in vis_poldict.keys()}[visname]
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
        orig_totflux = np.sum(obs_zerobl.amp['amp']*(1/obs_zerobl.amp['sigma']**2))/np.sum(1/obs_zerobl.amp['sigma']**2)

        print('Rescaling zero baseline by ' + str(orig_totflux - totflux) + ' Jy to ' + str(totflux) + ' Jy')

        # Rescale short baselines to excise contributions from extended flux
        # Note: this does not do the proper thing for fractional polarization)
        obs = self.copy()
        for j in range(len(obs.data)):
            if (obs.data['u'][j]**2 + obs.data['v'][j]**2)**0.5 < uv_max:
                obs.data['vis'][j] *= totflux / orig_totflux
                obs.data['qvis'][j] *= totflux / orig_totflux
                obs.data['uvis'][j] *= totflux / orig_totflux
                obs.data['vvis'][j] *= totflux / orig_totflux
                obs.data['vsigma'][j] *= totflux / orig_totflux
                obs.data['qsigma'][j] *= totflux / orig_totflux
                obs.data['usigma'][j] *= totflux / orig_totflux
                obs.data['vsigma'][j] *= totflux / orig_totflux

        return obs


    def add_leakage_noise(self, Dterm_amp=0.1, min_noise=0.01, debias=False):

        """Add estimated systematic noise from leakage at quadrature to thermal noise.
           Requires cross-hand visibilities.
           !!! this operation is not currently tracked in the data so should be applied with extreme caution!!!

           Args:
               Dterm_amp (float): Estimated magnitude of leakage terms
               min_noise (float): Minimum fractional systematic noise to add
               debias (bool): Debias amplitudes before computing fractional noise

           Returns:
               (Obsdata): An Obsdata object with the inflated noise values.
        """

        # Extract visibility amplitudes
        # Switch to Stokes for graceful handling of circular basis products missing RR or LL
        amp = self.switch_polrep('stokes').unpack('amp',debias=debias)['amp']
        rlamp = np.nan_to_num(self.switch_polrep('circ').unpack('rlamp',debias=debias)['rlamp'])
        lramp = np.nan_to_num(self.switch_polrep('circ').unpack('lramp',debias=debias)['lramp'])

        frac_noise = (Dterm_amp * rlamp/amp)**2 + (Dterm_amp * lramp/amp)**2
        frac_noise = frac_noise*(frac_noise > min_noise) + min_noise * (frac_noise < min_noise)

        out = self.copy()
        for sigma in ['sigma1','sigma2','sigma3','sigma4']:
            try:
                field = self.poldict[sigma]
                out.data[field] = (self.data[field]**2 + np.abs(frac_noise*amp)**2)**0.5
            except KeyError:
                continue

        return out


    def add_fractional_noise(self, frac_noise, debias=False):

        """Add a constant fraction of each visibility amplitude at quadrature to the corresponding thermal noise.
           Effectively imposes a maximal signal-to-noise ratio. 
           !!! this operation is not currently tracked in the data so should be applied with extreme caution!!!

           Args:
               frac_noise (float): The fraction of noise to add. frac_noise=0.05 imposes max SNR of 20.
               debias (bool):      Whether or not to add frac_noise of debiased amplitudes.

           Returns:
               (Obsdata): An Obsdata object with the inflated noise values.
        """

        # Extract visibility amplitudes
        # Switch to Stokes for graceful handling of circular basis products missing RR or LL
        amp = self.switch_polrep('stokes').unpack('amp',debias=debias)['amp']

        out = self.copy()
        for sigma in ['sigma1','sigma2','sigma3','sigma4']:
            try:
                field = self.poldict[sigma]
                out.data[field] = (self.data[field]**2 + np.abs(frac_noise*amp)**2)**0.5
            except KeyError:
                continue

        return out
    
    def find_amt_fractional_noise(self, im, dtype='vis', target=1.0, debias=False, maxiter=200, ftol=1e-20, gtol=1e-20):

        """Returns the amount of fractional sys error you need to add to an obs to make the image have a chisq close to the targeted value (1.0)
        """
            
        obs = self.copy()
        def objfunc(frac_noise):
            obs_tmp = obs.add_fractional_noise(frac_noise, debias=debias)
            chisq = obs_tmp.chisq(im, dtype=dtype)
            return np.abs(target - chisq)
        
        optdict = {'maxiter':maxiter, 'ftol':ftol, 'gtol':gtol}
        res = opt.minimize(objfunc, 0.0, method='L-BFGS-B',options=optdict)
        
        return res.x
    

    def rescale_noise(self, noise_rescale_factor=1.0):

        """Rescale the thermal noise on all Stokes parameters by a constant factor.
           This is useful for AIPS data, which has a missing factor relating 'weights' to thermal noise.

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

    def estimate_noise_rescale_factor(self, max_diff_sec=0.0, min_num=10, median_snr_cut=0, count='max', vtype='vis', print_std=False, ):

        """Estimate a constant rescaling factor for thermal noise across all baselines, times, and polarizations.
           Uses pairwise differences of closure phases relative to the expected scatter from the thermal noise.
           This is useful for AIPS data, which has a missing factor relating 'weights' to thermal noise.

           Args:
               max_diff_sec (float): The maximum difference of adjacent closure phases (in seconds) to be included in the estimate.
                                     If 0, auto-estimates this value to twice the median scan length.
               min_num (int): The minimum number of closure phase differences for a triangle to be included in the set of estimators.
               median_snr_cut (float): Do not include a triangle if its median SNR is below this number
               count (str): If 'min', use minimal set of phases, if 'max' use all closure phases up to reordering
               vtype (str): Visibility type (e.g., 'vis', 'llvis', 'rrvis', etc.)
               print_std (bool): Whether or not to print the normalized standard deviation for each closure triangle.

           Returns:
               (float): The rescaling factor. This can be applied to the data using the obsdata member function rescale_noise().
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
                all_triangles.append((cphase[1],cphase[2],cphase[3]))
        std_list = []
        print("Estimating noise rescaling factor from %d triangles...\n" % len(set(all_triangles)))

        # Now determine the differences of adjacent samples on each triangle, relative to the expected thermal noise
        i_count = 0
        for tri in set(all_triangles):
            i_count = i_count + 1
            if print_std:
                sys.stdout.write('\rGetting noise for triangles %i/%i ' % (i_count, len(set(all_triangles))))
                sys.stdout.flush()
            all_tri = np.array([[]])
            for scan in c_phases:
                for cphase in scan:
                    if cphase[1] == tri[0] and cphase[2] == tri[1] and cphase[3] == tri[2] and not np.isnan(cphase[-2]) and not np.isnan(cphase[-2]):
                        all_tri = np.append(all_tri, ((cphase[0], cphase[-2], cphase[-1])))
            all_tri = all_tri.reshape(int(len(all_tri)/3),3)

            # See whether the triangle has sufficient SNR
            if np.median(np.abs(all_tri[:,1]/all_tri[:,2])) < median_snr_cut:
                if print_std:
                    print(tri, 'median snr too low (%6.4f)' % np.median(np.abs(all_tri[:,1]/all_tri[:,2])))
                continue

            # Now go through and find studentized differences of adjacent points
            s_list = np.array([])
            for j in range(len(all_tri)-1):
                if (all_tri[j+1,0]-all_tri[j,0])*3600.0 < max_diff_sec:
                    diff = (all_tri[j+1,1]-all_tri[j,1]) % (2.0*np.pi)
                    if diff > np.pi: diff -= 2.0*np.pi
                    s_list = np.append(s_list, diff/(all_tri[j,2]**2 + all_tri[j+1,2]**2)**0.5)

            if len(s_list) > min_num:
                std_list.append(np.std(s_list))
                if print_std == True:
                    print(tri, '%6.4f [%d differences]' % (np.std(s_list),len(s_list)))
            else:
                if print_std == True and len(all_tri)>0:
                    print(tri, '%d cphases found [%d differences < min_num = %d]' % (len(all_tri),len(s_list),min_num))

        if len(std_list) == 0:
            print("No suitable closure phase differences identified! Try using a larger max_diff_sec.")
            median = 1.0
        else:
            median = np.median(std_list)

        return median

    def flag_elev(self, elev_min=0.0, elev_max=90, output='kept'):

        """Flag visibilities for which either station is outside a stated elevation range

           Args:
               elev_min (float): Minimum elevation (deg)
               elev_max (float): Maximum elevation (deg)
               output (str): return: 'kept' (data after flagging), 'flagged' (data that were flagged), or 'both' (a dictionary)

           Returns:
               (Obsdata): a observation object with flagged data points removed
        """

        el_pairs = self.unpack(['el1','el2'])
        mask = (np.min((el_pairs['el1'],el_pairs['el2']),axis=0) > elev_min) * (np.max((el_pairs['el1'],el_pairs['el2']),axis=0) < elev_max)

        datatable_kept    = self.data.copy()
        datatable_flagged = self.data.copy()

        datatable_kept    = datatable_kept[mask]
        datatable_flagged = datatable_flagged[np.invert(mask)]
        print('Flagged %d/%d visibilities' % (len(datatable_flagged), len(self.data)))

        obs_kept = self.copy()
        obs_flagged = self.copy()
        obs_kept.data    = datatable_kept
        obs_flagged.data = datatable_flagged

        if output == 'flagged':
            return obs_flagged
        elif output == 'both':
            return {'kept':obs_kept,'flagged':obs_flagged}
        else:
            return obs_kept

    def flag_large_fractional_pol(self, max_fractional_pol=1.0, output='kept'):

        """Flag visibilities for which the fractional polarization is above a specified threshold

           Args:
               max_fractional_pol (float): Maximum fractional polarization
               output (str): return: 'kept' (data after flagging), 'flagged' (data that were flagged), or 'both' (a dictionary)

           Returns:
               (Obsdata): a observation object with flagged data points removed
        """

        m = np.nan_to_num(self.unpack(['mamp'])['mamp'])
        mask = m < max_fractional_pol

        datatable_kept    = self.data.copy()
        datatable_flagged = self.data.copy()

        datatable_kept    = datatable_kept[mask]
        datatable_flagged = datatable_flagged[np.invert(mask)]
        print('Flagged %d/%d visibilities' % (len(datatable_flagged), len(self.data)))

        obs_kept = self.copy()
        obs_flagged = self.copy()
        obs_kept.data    = datatable_kept
        obs_flagged.data = datatable_flagged

        if output == 'flagged':
            return obs_flagged
        elif output == 'both':
            return {'kept':obs_kept,'flagged':obs_flagged}
        else:
            return obs_kept

    def flag_uvdist(self, uv_min=0.0, uv_max=1e12, output='kept'):

        """Flag data points outside a given uv range

           Args:
               uv_min (float): remove points with uvdist less than  this
               uv_max (float): remove points with uvdist greater than  this
               output (str): return: 'kept' (data after flagging), 'flagged' (data that were flagged), or 'both' (a dictionary)

           Returns:
               (Obsdata): a observation object with flagged data points removed
        """

        uvdist_list = self.unpack('uvdist')['uvdist']
        mask = np.array([uv_min <= uvdist_list[j] <= uv_max for j in range(len(uvdist_list))])
        datatable_kept    = self.data.copy()
        datatable_flagged = self.data.copy()

        datatable_kept    = datatable_kept[mask]
        datatable_flagged = datatable_flagged[np.invert(mask)]
        print('U-V flagged %d/%d visibilities' % (len(datatable_flagged), len(self.data)))

        obs_kept = self.copy()
        obs_flagged = self.copy()
        obs_kept.data    = datatable_kept
        obs_flagged.data = datatable_flagged

        if output == 'flagged':
            return obs_flagged
        elif output == 'both':
            return {'kept':obs_kept,'flagged':obs_flagged}
        else:
            return obs_kept

    def flag_sites(self, sites, output='kept'):

        """Flag data points that include the specified sites

           Args:
               sites (list): list of sites to remove from the data
               output (str): return: 'kept' (data after flagging), 'flagged' (data that were flagged), or 'both' (a dictionary)

           Returns:
               (Obsdata): a observation object with flagged data points removed
        """

        # This will remove all visibilities that include any of the specified sites
        datatable = self.data.copy()
        t1_list = self.unpack('t1')['t1']
        t2_list = self.unpack('t2')['t2']
        mask = np.array([t1_list[j] not in sites and t2_list[j] not in sites for j in range(len(t1_list))])

        datatable_kept    = self.data.copy()
        datatable_flagged = self.data.copy()

        datatable_kept    = datatable_kept[mask]
        datatable_flagged = datatable_flagged[np.invert(mask)]
        print('Flagged %d/%d visibilities' % (len(datatable_flagged), len(self.data)))

        obs_kept = self.copy()
        obs_flagged = self.copy()
        obs_kept.data    = datatable_kept
        obs_flagged.data = datatable_flagged

        if output == 'flagged': #return only the points flagged as anomalous
            return obs_flagged
        elif output == 'both':
            return {'kept':obs_kept,'flagged':obs_flagged}
        else:
            return obs_kept


    def flag_bl(self, sites, output='kept'):

        """Flag data points that include the specified baseline

           Args:
               sites (list): baseline to remove from the data
               output (str): return: 'kept' (data after flagging), 'flagged' (data that were flagged), or 'both' (a dictionary)

           Returns:
               (Obsdata): a observation object with flagged data points removed
        """

        # This will remove all visibilities that include any of the specified baseline
        obs_out = self.copy()
        t1_list = obs_out.unpack('t1')['t1']
        t2_list = obs_out.unpack('t2')['t2']
        mask = np.array([not(t1_list[j] in sites and t2_list[j] in sites) for j in range(len(t1_list))])

        datatable_kept    = self.data.copy()
        datatable_flagged = self.data.copy()

        datatable_kept    = datatable_kept[mask]
        datatable_flagged = datatable_flagged[np.invert(mask)]
        print('Flagged %d/%d visibilities' % (len(datatable_flagged), len(self.data)))

        obs_kept = self.copy()
        obs_flagged = self.copy()
        obs_kept.data    = datatable_kept
        obs_flagged.data = datatable_flagged

        if output == 'flagged': #return only the points flagged as anomalous
            return obs_flagged
        elif output == 'both':
            return {'kept':obs_kept,'flagged':obs_flagged}
        else:
            return obs_kept


    def flag_low_snr(self, snr_cut=3, output='kept'):

        """Flag low snr data points

           Args:
               snr_cut (float): remove points with snr lower than  this
               output (str): return: 'kept' (data after flagging), 'flagged' (data that were flagged), or 'both' (a dictionary)

           Returns:
               (Obsdata): a observation object with flagged data points removed
        """

        datatable = self.data.copy()
        mask = self.unpack('snr')['snr'] > snr_cut

        datatable_kept    = self.data.copy()
        datatable_flagged = self.data.copy()

        datatable_kept    = datatable_kept[mask]
        datatable_flagged = datatable_flagged[np.invert(mask)]
        print('snr flagged %d/%d visibilities' % (len(datatable_flagged), len(self.data)))

        obs_kept = self.copy()
        obs_flagged = self.copy()
        obs_kept.data    = datatable_kept
        obs_flagged.data = datatable_flagged

        if output == 'flagged': #return only the points flagged as anomalous
            return obs_flagged
        elif output == 'both':
            return {'kept':obs_kept,'flagged':obs_flagged}
        else:
            return obs_kept

    def flag_high_sigma(self, sigma_cut=.005, sigma_type='sigma', output='kept'):

        """Flag high sigma (thermal noise on Stoke I) data points

           Args:
               sigma_cut (float): remove points with sigma higher than  this
               sigma_type (str): sigma type (sigma, rrsigma, llsigma, etc.)
               output (str): return: 'kept' (data after flagging), 'flagged' (data that were flagged), or 'both' (a dictionary)

           Returns:
               (Obsdata): a observation object with flagged data points removed
        """

        datatable = self.data.copy()
        mask = self.unpack(sigma_type)[sigma_type] < sigma_cut

        datatable_kept    = self.data.copy()
        datatable_flagged = self.data.copy()

        datatable_kept    = datatable_kept[mask]
        datatable_flagged = datatable_flagged[np.invert(mask)]
        print('sigma flagged %d/%d visibilities' % (len(datatable_flagged), len(self.data)))

        obs_kept = self.copy()
        obs_flagged = self.copy()
        obs_kept.data    = datatable_kept
        obs_flagged.data = datatable_flagged

        if output == 'flagged': #return only the points flagged as anomalous
            return obs_flagged
        elif output == 'both':
            return {'kept':obs_kept,'flagged':obs_flagged}
        else:
            return obs_kept


    def flag_UT_range(self, UT_start_hour=0., UT_stop_hour=0.,flag_type='all',flag_what='',output='kept'):

        """Flag data points within a certain UT range

           Args:
               UT_start_hour (float): start of  time window
               UT_stop_hour (float): end of time window
               flag_type (str): 'all' for flagging everything, 'baseline' for flagginng given baseline, 'station' for flagging givern station
               flag_what (str): baseline or station to flag (order of stations in baseline doesn't matter)
               output (str): return: 'kept' (data after flagging), 'flagged' (data that were flagged), or 'both' (a dictionary)

           Returns:
               (Obsdata): a observation object with flagged data points removed
        """

        # This drops (or only keeps) points within a specified UT range
        datatable = self.data.copy()
        UT_mask = self.unpack('time')['time'] <= UT_start_hour
        UT_mask = UT_mask + (self.unpack('time')['time'] >= UT_stop_hour)
        if flag_type!='all':
            t1_list = self.unpack('t1')['t1']
            t2_list = self.unpack('t2')['t2']
            if flag_type=='station':
                station=flag_what
                what_mask = np.array([not (t1_list[j] == station or t2_list[j] == station) for j in range(len(t1_list))])
            elif flag_type=='baseline':
                station1=flag_what.split('-')[0]
                station2=flag_what.split('-')[1]
                stations=[station1,station2]
                what_mask = np.array([not ( (t1_list[j] in stations) and (t2_list[j] in stations) ) for j in range(len(t1_list))])
        else:
            what_mask = np.array([False for j in range(len(UT_mask))])
        mask = UT_mask|what_mask

        datatable_kept    = self.data.copy()
        datatable_flagged = self.data.copy()

        datatable_kept    = datatable_kept[mask]
        datatable_flagged = datatable_flagged[np.invert(mask)]
        print('time flagged %d/%d visibilities' % (len(datatable_flagged), len(self.data)))

        obs_kept = self.copy()
        obs_flagged = self.copy()
        obs_kept.data    = datatable_kept
        obs_flagged.data = datatable_flagged

        if output == 'flagged': #return only the points flagged as anomalous
            return obs_flagged
        elif output == 'both':
            return {'kept':obs_kept,'flagged':obs_flagged}
        else:
            return obs_kept

    def flags_from_file(self,flagfile, flag_type='station'):

        """Flagging data based on csv file

            Args:
                flagfile (str): path to csv file with mjd of flagging start / stop time, and optionally baseline / station to flag
                flag_type (str): 'all' for flagging everything, 'baseline' for flagginng given baseline, 'station' for flagging givern station

            Returns:
                (Obsdata): a observation object with flagged data points removed
        """
        df = pd.read_csv(flagfile)
        mjd_start = list(df['mjd_start'])
        mjd_stop = list(df['mjd_stop'])
        if flag_type=='station':
            whatL = list(df['station'])
        elif flag_type=='baseline':
            whatL = list(df['baseline'])
        elif flag_type=='all':
            whatL = ['' for cou in range(len(mjd_start))]
        obs = self.copy()
        for cou in range(len(mjd_start)):
            what = whatL[cou]
            starth = (mjd_start[cou] % 1)*24.
            stoph=(mjd_stop[cou] % 1)*24.
            obs = obs.flag_UT_range( UT_start_hour=starth, UT_stop_hour=stoph, flag_type=flag_type,flag_what=what,output='kept')

        return obs


    def flag_anomalous(self, field='snr', max_diff_seconds=100, robust_nsigma_cut=5, output='kept'):

        """Flag anomalous data points

           Args:
               field (str): The quantity to test for
               max_diff_seconds (float): The moving window size for testing outliers
               robust_nsigma_cut (float): Outliers further than this many sigmas from the mean are removed
               output (str): return: 'kept' (data after flagging), 'flagged' (data that were flagged), or 'both' (a dictionary)

           Returns:
               (Obsdata): a observation object with flagged data points removed
        """

        stats = dict()
        for t1 in set(self.data['t1']):
            for t2 in set(self.data['t2']):
                vals = self.unpack_bl(t1,t2,field)
                if len(vals) > 0:
                    vals[field] = np.nan_to_num(vals[field]) # nans will all be dropped, which can be problematic for polarimetric values
                for j in range(len(vals)):
                    near_vals_mask = np.abs(vals['time'] - vals['time'][j])<max_diff_seconds/3600.0
                    fields  = vals[field][np.abs(vals['time'] - vals['time'][j])<max_diff_seconds/3600.0]

                    # Here, we use median absolute deviation from the median as a robust proxy for standard deviation
                    dfields = np.median(np.abs(fields-np.median(fields)))
                    if dfields == 0.0: # Avoid problem when the MAD is zero (e.g., a single sample)
                        dfields = 1.0
                    stats[(vals['time'][j][0], tuple(sorted((t1,t2))))] = np.abs(vals[field][j]-np.median(fields)) / dfields

        mask = np.array([stats[(rec[0], tuple(sorted((rec[2], rec[3]))))][0] < robust_nsigma_cut for rec in self.data])

        datatable_kept    = self.data.copy()
        datatable_flagged = self.data.copy()

        datatable_kept    = datatable_kept[mask]
        datatable_flagged = datatable_flagged[np.invert(mask)]
        print('anomalous %s flagged %d/%d visibilities' % (field, len(datatable_flagged), len(self.data)))

        # Make new observations with all data first to avoid problems with empty arrays
        obs_kept = self.copy()
        obs_flagged = self.copy()
        obs_kept.data    = datatable_kept
        obs_flagged.data = datatable_flagged

        if output == 'flagged': #return only the points flagged as anomalous
            return obs_flagged
        elif output == 'both':
            return {'kept':obs_kept,'flagged':obs_flagged}
        else:
            return obs_kept


    def filter_subscan_dropouts(self,perc=0,return_type='rec'):
        '''Fancy filtration that drops some data to ensure that we only average parts with same timestamp.
            Potentially this could reduce risk of non-closing errors.

            Args:
                perc (float, reasonably in [0,1]) drop baseline from scan if it has less than this fraction of median 
                baseline observation time during the scan
                return_type (str): data frame ('df') or recarray ('rec')
        '''
        if type(self.scans)!=np.ndarray:
            print('List of scans in ndarray format required! Add it with add_scans')
        else:    
            #make df and add scan_id to data
            df = make_df(self)
            tot_points=np.shape(df)[0]
            bins, labs = get_bins_labels(self.scans)
            df['scan_id'] = list(pd.cut(df.time, bins,labels=labs))
        
            #first flag baselines that are working for short part of scan
            df['count_samples'] = 1
            hm1 = df.groupby(['scan_id','baseline','polarization']).agg({'count_samples':np.sum}).reset_index()
            hm1['count_baselines_before'] = 1
            hm2 = hm1.groupby(['scan_id','polarization']).agg({'count_samples': lambda x: perc*np.median(x),'count_baselines_before':np.sum}).reset_index()

            #dictionary with minimum acceptable number of samples per scan
            dict_elem_in_scan = dict(zip(hm2.scan_id,hm2.count_samples))
        
            #list of acceptable scans and baselines
            hm1=hm1[list(map(lambda x: x[1] >= dict_elem_in_scan[x[0]],list(zip(hm1.scan_id,hm1.count_samples))))]
            list_good_scans_baselines=list(zip(hm1.scan_id,hm1.baseline))
        
            #filter out data
            df_filtered=df[list(map(lambda x: x in list_good_scans_baselines,list(zip(df.scan_id,df.baseline))))]

            #how many baselines present during scan?
            df_filtered['count_samples'] = 1
            hm3 = df_filtered.groupby(['scan_id','baseline','polarization']).agg({'count_samples':np.sum}).reset_index()
            hm3['count_baselines_after'] = 1
            hm4 = hm3.groupby(['scan_id','polarization']).agg({'count_baselines_after': np.sum}).reset_index()
            dict_how_many_baselines = dict(zip(hm4.scan_id,hm4.count_baselines_after))

            #how many baselines present during each time?
            df_filtered['count_baselines_per_time'] = 1
            hm5=df_filtered.groupby(['datetime','scan_id','polarization']).agg({'count_baselines_per_time': np.sum}).reset_index()
            dict_datetime_num_baselines = dict(zip(hm5.datetime,hm5.count_baselines_per_time))

            #only keep times when all baselines available
            df_filtered2 = df_filtered[list(map(lambda x: dict_datetime_num_baselines[x[1]] == dict_how_many_baselines[x[0]],list(zip(df_filtered.scan_id,df_filtered.datetime))))]

            remaining_points=np.shape(df_filtered2)[0]
            print('Flagged out {} of {} datapoints'.format(tot_points - remaining_points,tot_points))
            if return_type=='rec':
                out_vis = df_to_rec(df_filtered2,'vis')

            out = Obsdata(self.ra, self.dec, self.rf, self.bw, out_vis, self.tarr, source=self.source, mjd=self.mjd,
                       ampcal=self.ampcal, phasecal=self.phasecal, opacitycal=self.opacitycal, dcal=self.dcal, frcal=self.frcal,
                       timetype=self.timetype, scantable=self.scans)
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

        fwhm_sigma = fwhm / (2*np.sqrt(2*np.log(2)))
        ker = np.exp(-2 * np.pi**2 * fwhm_sigma**2*(u**2+v**2))

        datatable[self.poldict['vis1']] = vis1/ker
        datatable[self.poldict['vis2']] = vis2/ker
        datatable[self.poldict['vis3']] = vis3/ker
        datatable[self.poldict['vis4']] = vis4/ker
        datatable[self.poldict['sigma1']] = sigma1/ker
        datatable[self.poldict['sigma2']] = sigma2/ker
        datatable[self.poldict['sigma3']] = sigma3/ker
        datatable[self.poldict['sigma4']] = sigma4/ker

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

        fwhm_sigma = fwhm / (2*np.sqrt(2*np.log(2)))
        ker = np.exp(-2 * np.pi**2 * fwhm_sigma**2*(u**2+v**2))

        datatable[self.poldict['vis1']] = vis1*ker
        datatable[self.poldict['vis2']] = vis2*ker
        datatable[self.poldict['vis3']] = vis3*ker
        datatable[self.poldict['vis4']] = vis4*ker
        datatable[self.poldict['sigma1']] = sigma1*ker
        datatable[self.poldict['sigma2']] = sigma2*ker
        datatable[self.poldict['sigma3']] = sigma3*ker
        datatable[self.poldict['sigma4']] = sigma4*ker

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
            ker = sgra_kernel_uv(self.rf, u[i], v[i])
            vis1[i] = vis1[i]/ker
            vis2[i] = vis2[i]/ker
            vis2[i] = vis3[i]/ker
            vis4[i] = vis4[i]/ker
            sigma1[i] = sigma1[i]/ker
            sigma2[i] = sigma2[i]/ker
            sigma3[i] = sigma3[i]/ker
            sigma4[i] = sigma4[i]/ker

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
            matches1 = uvpoints_tree1.query_ball_point(uvpoints[i,:], uv_radius)
            matches2 = uvpoints_tree2.query_ball_point(uvpoints[i,:], uv_radius)
            nmatches = len(matches1) + len(matches2)

            for sigma in ['sigma', 'qsigma', 'usigma', 'vsigma']:
                obs_new.data[sigma][i] = np.sqrt(nmatches)

        scale = np.mean( self.data['sigma'] ) / np.mean( obs_new.data['sigma'] )
        for sigma in ['sigma', 'qsigma', 'usigma', 'vsigma']:
            obs_new.data[sigma] *= scale*weightdist

        if weightdist < 1.0:
            for i in range(npts):
                for sigma in ['sigma', 'qsigma', 'usigma', 'vsigma']:
                    obs_new.data[sigma][i] += (1-weightdist)*self.data[sigma][i]

        return obs_new


    def fit_gauss(self, flux=1.0, fittype='amp', paramguess=(100*RADPERUAS, 100*RADPERUAS, 0.)):

        """Fit a gaussian to either Stokes I complex visibilities or Stokes I visibility amplitudes.

           Args:
                flux (float): total flux in the fitted gaussian
                fitttype (str): "amp" to fit to visibilty amplitudes
                paramguess (tuble): initial guess of fit Gaussian (fwhm_maj, fwhm_min, theta)

           Returns:
                (tuple) : a tuple (fwhm_maj, fwhm_min, theta) of the fit Gaussian parameters in radians.
        """

        #TODO this fit doesn't work very well!!
        vis = self.data['vis']
        u = self.data['u']
        v = self.data['v']
        sig = self.data['sigma']

        # error function
        if fittype=='amp':
            def errfunc(p):
                vismodel = gauss_uv(u,v, flux, p, x=0., y=0.)
                err = np.sum((np.abs(vis)-np.abs(vismodel))**2/sig**2)
                return err
        else:
            def errfunc(p):
                vismodel = gauss_uv(u,v, flux, p, x=0., y=0.)
                err = np.sum(np.abs(vis-vismodel)**2/sig**2)
                return err

        optdict = {'maxiter':5000} # minimizer params
        res = opt.minimize(errfunc, paramguess, method='Powell',options=optdict)
        gparams = res.x

        return gparams

    def bispectra(self, vtype='vis', mode='all', count='min',timetype=False, uv_min=False, snrcut=0.):

        """Return a recarray of the equal time bispectra.

           Args:
               vtype (str): The visibilty type ('vis', 'qvis', 'uvis','vvis','rrvis','lrvis','rlvis','llvis') from which to assemble bispectra
               mode (str): If 'time', return phases in a list of equal time arrays, if 'all', return all phases in a single array
               count (str): If 'min', return minimal set of bispectra, if 'max' return all bispectra up to reordering
               timetype (str): 'GMST' or 'UTC'
               uv_min (float): flag baselines shorter than this before forming closure quantities
               snrcut (float): flag bispectra with snr lower than this               

           Returns:
               (numpy.recarry): A recarray of the bispectra values with datatype DTBIS
        """

        if timetype==False:
            timetype=self.timetype
        if not mode in ('time', 'all'):
            raise Exception("possible options for mode are 'time' and 'all'")
        if not count in ('max', 'min', 'min-cut0bl'):
            raise Exception("possible options for count are 'max', 'min', or 'min-cut0bl'")
        if not vtype in ('vis', 'qvis', 'uvis','vvis','rrvis','lrvis','rlvis','llvis'):
            raise Exception("possible options for vtype are 'vis', 'qvis', 'uvis','vvis','rrvis','lrvis','rlvis','llvis'")
        if timetype not  in ['GMST','UTC','gmst','utc']:
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

            sys.stdout.write('\rGetting bispectra:: type %s, count %s, scan %i/%i ' % (vtype, count, tt, len(tlist)))
            sys.stdout.flush()

            tt += 1

            time = tdata[0]['time']
            if timetype in ['GMST','gmst'] and self.timetype=='UTC':
                time = utc_to_gmst(time, self.mjd)
            if timetype in ['UTC','utc'] and self.timetype=='GMST':
                time = gmst_to_utc(time, self.mjd)
            sites = list(set(np.hstack((tdata['t1'],tdata['t2']))))

            # Create a dictionary of baselines at the current time incl. conjugates;
            l_dict = {}
            for dat in tdata:
                l_dict[(dat['t1'], dat['t2'])] = dat

            # Determine the triangles in the time step
            # Minimal Set
            if count == 'min':
                tris = tri_minimal_set(sites, self.tarr, self.tkey)
            
            # Maximal  Set
            elif count == 'max':
                tris = list(it.combinations(sites,3))
            
            elif count == 'min-cut0bl':
                tris = tri_minimal_set(sites, self.tarr, self.tkey)
                
                # if you cut the 0 baselines then add in triangles that now are not in the minimal set
                if uv_min:
                    # get the reference site
                    sites_ordered = [x for x in self.tarr['site'] if x in sites]
                    ref = sites_ordered[0]
                    
                    # check if the reference site was in a zero baseline
                    zerobls = np.vstack([obsdata_flagged.data['t1'], obsdata_flagged.data['t2']])
                    if np.sum(zerobls==ref):
                    
                        # determine which sites were cut out of the minimal set
                        cutsites = np.unique(np.hstack([zerobls[1][zerobls[0] == ref], zerobls[0][zerobls[1] == ref]]))
                        
                        # we can only handle if there was 1 connecting site that was cut
                        if len(cutsites)>1:
                            raise Exception("Cannot have the root node be in a clique with more than 2 sites sharing 0 baselines'")

                        # get the remaining sites
                        cutsite = cutsites[0]
                        sites_remaining = np.array(sites_ordered)[ np.array(sites_ordered) != ref]
                        sites_remaining = sites_remaining[np.array(sites_remaining) != cutsite]
                        # get the next site in the list, ideally sorted by snr
                        second_ref = sites_remaining[0]
                        
                        # add in additional triangles
                        for s2 in range(1,len(sites_remaining)):
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

                (bi, bisig) = make_bispectrum(l1,l2,l3,vtype,polrep=self.polrep)
            
                # Cut out low snr points
                if np.abs(bi)/bisig < snrcut:
                    continue

                # Append to the equal-time list
                bis.append(np.array((time,
                                     tri[0], tri[1], tri[2],
                                     l1['u'], l1['v'],
                                     l2['u'], l2['v'],
                                     l3['u'], l3['v'],
                                     bi, bisig), dtype=DTBIS))

            # Append to outlist
            if mode=='time' and len(bis) > 0:
                out.append(np.array(bis))
                bis = []

        if mode=='all':
            out = np.array(bis)

        return out

    def c_phases(self, vtype='vis', mode='all', count='min', ang_unit='deg', timetype=False, uv_min=False, snrcut=0.):

        """Return a recarray of the equal time closure phases.

           Args:
               vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble closure phases
               mode (str): If 'time', return phases in a list of equal time arrays, if 'all', return all phases in a single array
               count (str): If 'min', return minimal set of phases, if 'max' return all closure phases up to reordering
               ang_unit (str): If 'deg', return closure phases in degrees, else return in radians
               timetype (str): 'UTC' or 'GMST'
               uv_min (float): flag baselines shorter than this before forming closure quantities
               snrcut (float): flag bispectra with snr lower than this               

           Returns:
               (numpy.recarry): A recarray of the closure phases with datatype DTCPHASE
        """

        if timetype==False:
            timetype=self.timetype
        if not mode in ('time', 'all'):
            raise Exception("possible options for mode are 'time' and 'all'")
        if not count in ('max', 'min', 'min-cut0bl'):
            raise Exception("possible options for count are 'max', 'min', or 'min-cut0bl'")
        if not vtype in ('vis', 'qvis', 'uvis','vvis','rrvis','lrvis','rlvis','llvis'):
            raise Exception("possible options for vtype are 'vis', 'qvis', 'uvis','vvis','rrvis','lrvis','rlvis','llvis'")
        if timetype not  in ['GMST','UTC','gmst','utc']:
            raise Exception("timetype should be 'GMST' or 'UTC'!")

        if ang_unit=='deg': angle=DEGREE
        else: angle = 1.0

        # Get the bispectra data
        bispecs = self.bispectra(vtype=vtype, mode='time', count=count, timetype=timetype, uv_min=uv_min, snrcut=snrcut)

        # Reformat into a closure phase list/array
        out = []
        cps = []

        for bis in bispecs:
            for bi in bis:
                if len(bi) == 0: continue
                bi.dtype.names = ('time','t1','t2','t3','u1','v1','u2','v2','u3','v3','cphase','sigmacp')
                bi['sigmacp'] = np.real(bi['sigmacp']/np.abs(bi['cphase'])/angle)
                bi['cphase'] = np.real((np.angle(bi['cphase'])/angle))
                cps.append(bi.astype(np.dtype(DTCPHASE)))

            if mode == 'time' and len(cps) > 0:
                out.append(np.array(cps))
                cps = []

        if mode == 'all':
            out = np.array(cps)

        print("\n")
        return out

    def bispectra_tri(self, site1, site2, site3, snrcut=0.,
                            vtype='vis', timetype=False, bs=[],force_recompute=False):

        """Return complex bispectrum  over time on a triangle (1-2-3).

           Args:
               site1 (str): station 1 name
               site2 (str): station 2 name
               site3 (str): station 3 name
               vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble closure phases
               timetype (str): 'UTC' or 'GMST'
               snrcut (float): flag bispectra with snr lower than this               

               bs (list): optionally pass in the time-sorted bispectra so they don't have to be recomputed
               force_recompute (bool): if True, recompute bispectra instead of using obs.cphase saved data

           Returns:
               (numpy.recarry): A recarray of the bispectra on this triangle with datatype DTBIS
        """
        if timetype==False:
            timetype=self.timetype

        # Get bispectra (maximal set)
        if (len(bs)==0) and not (self.bispec is None) and not (len(self.bispec)==0) and not force_recompute:
            bs=self.bispec
        elif (len(bs) == 0) or force_recompute:
            bs = self.bispectra(mode='all', count='max', vtype=vtype, timetype=timetype, snrcut=snrcut)

        # Get requested closure phases over time
        tri = (site1, site2, site3)
        outdata = []
        for obs in bs:
            obstri = (obs['t1'],obs['t2'],obs['t3'])
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
                if t1==site1:
                    if t2==site2:
                        pass
                    else:
                        obs['t2']=t3
                        obs['t3']=t2

                        obs['u1']=-u3
                        obs['v1']=-v3
                        obs['u2']=-u2
                        obs['v2']=-v2
                        obs['u3']=-u1
                        obs['v3']=-v1
                        obs['bispec'] = np.conjugate(obs['bispec'])

                elif t1==site2:
                    if t2==site3:
                        obs['t1']=t3
                        obs['t2']=t1
                        obs['t3']=t2

                        obs['u1']=u3
                        obs['v1']=v3
                        obs['u2']=u1
                        obs['v2']=v1
                        obs['u3']=u2
                        obs['v3']=v2

                    else:
                        obs['t1']=t2
                        obs['t2']=t1

                        obs['u1']=-u1
                        obs['v1']=-v1
                        obs['u2']=-u3
                        obs['v2']=-v3
                        obs['u3']=-u2
                        obs['v3']=-v2
                        obs['bispec'] = np.conjugate(obs['bispec'])

                elif t1==site3:
                    if t2==site1:
                        obs['t1']=t2
                        obs['t2']=t3
                        obs['t3']=t1

                        obs['u1']=u2
                        obs['v1']=v2
                        obs['u2']=u3
                        obs['v2']=v3
                        obs['u3']=u1
                        obs['v3']=v1

                    else:
                        obs['t1']=t3
                        obs['t3']=t1

                        obs['u1']=-u2
                        obs['v1']=-v2
                        obs['u2']=-u1
                        obs['v2']=-v1
                        obs['u3']=-u3
                        obs['v3']=-v3
                        obs['bispec'] = np.conjugate(obs['bispec'])

                outdata.append(np.array(obs, dtype=DTBIS))
                continue

        return np.array(outdata)


    def cphase_tri(self, site1, site2, site3, snrcut=0.,
                         vtype='vis', ang_unit='deg', timetype=False, cphases=[], force_recompute=False):

        """Return closure phase  over time on a triangle (1-2-3).

           Args:
               site1 (str): station 1 name
               site2 (str): station 2 name
               site3 (str): station 3 name
               vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble closure phases
               ang_unit (str): If 'deg', return closure phases in degrees, else return in radians
               timetype (str): 'GMST' or 'UTC'
               snrcut (float): flag bispectra with snr lower than this               

               cphases (list): optionally pass in the cphase so they are not recomputed if you are plotting multiple triangles
               force_recompute (bool): if True, recompute closure phases instead of using obs.cphase saved data

           Returns:
               (numpy.recarry): A recarray of the closure phases on this triangle with datatype DTCPHASE
        """

        if timetype==False:
            timetype=self.timetype

        # Get closure phases (maximal set)
        if (len(cphases)==0) and not (self.cphase is None) and not (len(self.cphase)==0) and not force_recompute:
            cphases=self.cphase

        elif (len(cphases) == 0) or force_recompute:
            cphases = self.c_phases(mode='all', count='max', vtype=vtype, ang_unit=ang_unit, timetype=timetype, snrcut=snrcut)

        # Get requested closure phases over time
        tri = (site1, site2, site3)
        outdata = []
        for obs in cphases:
            obstri = (obs['t1'],obs['t2'],obs['t3'])
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
                if t1==site1:
                    if t2==site2:
                        pass
                    else:
                        obs['t2']=t3
                        obs['t3']=t2

                        obs['u1']=-u3
                        obs['v1']=-v3
                        obs['u2']=-u2
                        obs['v2']=-v2
                        obs['u3']=-u1
                        obs['v3']=-v1
                        obs['cphase'] *= -1

                elif t1==site2:
                    if t2==site3:
                        obs['t1']=t3
                        obs['t2']=t1
                        obs['t3']=t2

                        obs['u1']=u3
                        obs['v1']=v3
                        obs['u2']=u1
                        obs['v2']=v1
                        obs['u3']=u2
                        obs['v3']=v2

                    else:
                        obs['t1']=t2
                        obs['t2']=t1

                        obs['u1']=-u1
                        obs['v1']=-v1
                        obs['u2']=-u3
                        obs['v2']=-v3
                        obs['u3']=-u2
                        obs['v3']=-v2
                        obs['cphase'] *= -1


                elif t1==site3:
                    if t2==site1:
                        obs['t1']=t2
                        obs['t2']=t3
                        obs['t3']=t1

                        obs['u1']=u2
                        obs['v1']=v2
                        obs['u2']=u3
                        obs['v2']=v3
                        obs['u3']=u1
                        obs['v3']=v1

                    else:
                        obs['t1']=t3
                        obs['t3']=t1

                        obs['u1']=-u2
                        obs['v1']=-v2
                        obs['u2']=-u1
                        obs['v2']=-v1
                        obs['u3']=-u3
                        obs['v3']=-v3
                        obs['cphase'] *= -1

                outdata.append(np.array(obs, dtype=DTCPHASE))
                continue

        return np.array(outdata)

    def c_amplitudes(self, vtype='vis', mode='all', count='min', ctype='camp', debias=True, timetype=False, snrcut=0.):

        """Return a recarray of the equal time closure amplitudes.

           Args:
               vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble closure amplitudes
               ctype (str): The closure amplitude type ('camp' or 'logcamp')
               mode (str): If 'time', return amplitudes in a list of equal time arrays, if 'all', return all amplitudes in a single array
               count (str): If 'min', return minimal set of amplitudes, if 'max' return all closure amplitudes up to inverses
               debias (bool): If True, debias the closure amplitude - the individual visibility amplitudes are always debiased.
               timetype (str): 'GMST' or 'UTC'
               snrcut (float): flag closure amplitudes with snr lower than this               

           Returns:
               (numpy.recarry): A recarray of the closure amplitudes with datatype DTCAMP

        """

        if timetype==False:
            timetype=self.timetype
        if not mode in ('time','all'):
            raise Exception("possible options for mode are 'time' and 'all'")
        if not count in ('max', 'min'):
            raise Exception("possible options for count are 'max' and 'min'")
        if not vtype in ('vis', 'qvis', 'uvis','vvis','rrvis','lrvis','rlvis','llvis'):
            raise Exception("possible options for vtype are 'vis', 'qvis', 'uvis','vvis','rrvis','lrvis','rlvis','llvis'")
        if not (ctype in ['camp', 'logcamp']):
            raise Exception("closure amplitude type must be 'camp' or 'logcamp'!")
        if timetype not  in ['GMST','UTC','gmst','utc']:
            raise Exception("timetype should be 'GMST' or 'UTC'!")

        # Get data sorted by time
        tlist = self.tlist(conj=True)
        out = []
        cas = []
        tt = 1
        for tdata in tlist:

            sys.stdout.write('\rGetting closure amps:: type %s %s , count %s, scan %i/%i' % (vtype, ctype, count, tt, len(tlist)))
            sys.stdout.flush()
            tt += 1

            time = tdata[0]['time']
            if timetype in ['GMST','gmst'] and self.timetype=='UTC':
                time = utc_to_gmst(time, self.mjd)
            if timetype in ['UTC','utc'] and self.timetype=='GMST':
                time = gmst_to_utc(time, self.mjd)

            sites = np.array(list(set(np.hstack((tdata['t1'], tdata['t2'])))))
            if len(sites) < 4:
                continue

            # Create a dictionary of baseline data at the current time including conjugates;
            l_dict = {}
            for dat in tdata:
                l_dict[(dat['t1'], dat['t2'])] = dat

            # Minimal set
            if count == 'min':
                quadsets = quad_minimal_set(sites, self.tarr, self.tkey)

            # Maximal Set
            elif count == 'max':
                # Find all quadrangles
                quadsets = list(it.combinations(sites,4))
                # Include 3 closure amplitudes on each quadrangle
                quadsets = np.array([(q, [q[0],q[2],q[1],q[3]], [q[0],q[1],q[3],q[2]]) for q in quadsets]).reshape((-1,4))

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
                (camp, camperr) = make_closure_amplitude(blue1, blue2, red1, red2, vtype,
                                                         polrep=self.polrep,
                                                         ctype=ctype, debias=debias)

                if ctype=='camp':
                    if camp/camperr < snrcut: continue
                elif ctype=='logcamp': #TODO -- check this!
                    if 1./camperr < snrcut: continue

                # Add the closure amplitudes to the equal-time list
                # Our site convention is (12)(34)/(14)(23)
                cas.append(np.array((time,
                                     quad[0], quad[1], quad[2], quad[3],
                                     blue1['u'], blue1['v'], blue2['u'], blue2['v'],
                                     red1['u'], red1['v'], red2['u'], red2['v'],
                                     camp, camperr),
                                     dtype=DTCAMP))

            # Append all equal time closure amps to outlist
            if mode=='time':
                out.append(np.array(cas))
                cas = []

        if mode=='all':
            out = np.array(cas)
        print("\n")
        return out

    def camp_quad(self, site1, site2, site3, site4, snrcut=0., 
                        vtype='vis', ctype='camp', debias=True, timetype=False,
                        camps=[], force_recompute=False):

        """Return closure phase over time on a quadrange (1-2)(3-4)/(1-4)(2-3).

           Args:
               site1 (str): station 1 name
               site2 (str): station 2 name
               site3 (str): station 3 name
               site4 (str): station 4 name
               vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble closure amplitudes
               ctype (str): The closure amplitude type ('camp' or 'logcamp')
               debias (bool): If True, debias the closure amplitude - the individual visibility amplitudes are always debiased.
               timetype (str): 'UTC' or 'GMST'
               camps (list): optionally pass in the time-sorted camps so they don't have to be recomputed
               snrcut (float): flag closure amplitudes with snr lower than this               

           Returns:
               (numpy.recarry): A recarray of the closure amplitudes with datatype DTCAMP
        """

        if timetype==False:
            timetype=self.timetype

        quad = (site1, site2, site3, site4)
        b1 = set((site1, site2))
        b2 = set((site3, site4))

        r1 = set((site1, site4))
        r2 = set((site2, site3))

        # Get the closure amplitudes
        outdata = []

        # Get closure amplitudes (maximal set)
        if ((ctype=='camp') and (len(camps)==0)) and not (self.camp is None) and not (len(self.camp)==0) and not force_recompute:
            camps=self.camp
        elif ((ctype=='logcamp') and (len(camps)==0)) and not (self.logcamp is None) and not (len(self.logcamp)==0) and not force_recompute:
            camps=self.logcamp
        elif (len(camps)==0) or force_recompute:
            camps = self.c_amplitudes(mode='all', count='max', vtype=vtype, ctype=ctype, debias=debias, timetype=timetype, snrcut=snrcut)

        # camps does not contain inverses
        for obs in camps:

            num   = [set((obs['t1'], obs['t2'])), set((obs['t3'], obs['t4']))]
            denom = [set((obs['t1'], obs['t4'])), set((obs['t2'], obs['t3']))]

            obsquad = (obs['t1'], obs['t2'], obs['t3'], obs['t4'])
            if set(quad) == set(obsquad):

                # is this either  the closure amplitude or inverse?
                rightup = (b1 in num) and (b2 in num) and (r1 in denom) and (r2 in denom)
                wrongup = (b1 in denom) and (b2 in denom) and (r1 in num) and (r2 in num)
                if not (rightup or wrongup): continue

                #flip the inverse closure amplitudes
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

                    if ctype=='logcamp':
                        obs['camp'] = -campold
                        obs['sigmaca'] = csigmaold
                    else:
                        obs['camp'] = 1./campold
                        obs['sigmaca'] = csigmaold/(campold**2)


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
                if (obs['t2'],obs['t1'],obs['t4'],obs['t3']) == quad:
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

                elif (obs['t3'],obs['t4'],obs['t1'],obs['t2']) == quad:
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

                elif (obs['t4'],obs['t3'],obs['t2'],obs['t1']) == quad:
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
                outdata.append(np.array(obs, dtype=DTCAMP))

        return np.array(outdata)

    def plotall(self, field1, field2,
                      conj=False, debias=True, tag_bl=False, ang_unit='deg', timetype=False,
                      axis=False, rangex=False, rangey=False,snrcut=0.,
                      color=SCOLORS[0], marker='o', markersize=MARKERSIZE, label=None,
                      grid=True, ebar=True, axislabels=True, legend=False,
                      show=True, export_pdf="", ):

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

        if timetype==False:
            timetype=self.timetype

        # Determine if fields are valid
        if (field1 not in FIELDS) and (field2 not in FIELDS):
            raise Exception("valid fields are " + ' '.join(FIELDS))

        if 'amp' in [field1, field2] and not (self.amp is None):
            print ("Warning: plotall is not using amplitudes in Obsdata.amp array!")

        # Label individual baselines
        # ANDREW TODO this is way too slow, make  it faster??
        if tag_bl:
            clist = SCOLORS

            # make a color coding dictionary
            cdict = {}
            ii = 0
            baselines = list(it.combinations(self.tarr['site'],2))
            for baseline in baselines:
                cdict[(baseline[0],baseline[1])] = clist[ii%len(clist)]
                cdict[(baseline[1],baseline[0])] = clist[ii%len(clist)]
                ii+=1

            # get unique baselines -- TODO easier way? separate function?
            alldata = []
            allsigx = []
            allsigy = []
            bllist = []
            colors = []
            bldata = self.bllist(conj=conj)
            for bl in bldata:
                t1 = bl['t1'][0]
                t2 = bl['t2'][1]
                bllist.append((t1,t2))
                colors.append(cdict[(t1,t2)])

                # Unpack data
                dat = self.unpack_dat(bl, [field1,field2], ang_unit=ang_unit, debias=debias, timetype=timetype)
                alldata.append(dat)

                # X error bars
                if sigtype(field1):
                    allsigx.append(self.unpack_dat(bl,[sigtype(field1)], ang_unit=ang_unit)[sigtype(field1)])
                else:
                    allsigx.append(None)

                # Y error bars
                if sigtype(field2):
                    allsigy.append(self.unpack_dat(bl,[sigtype(field2)], ang_unit=ang_unit)[sigtype(field2)])
                else:
                    allsigy.append(None)

        # Don't Label individual baselines
        else:
            bllist = [['All','All']]
            colors = [color]

            # unpack data
            alldata = [self.unpack([field1, field2], conj=conj, ang_unit=ang_unit, debias=debias)]

            # X error bars
            if sigtype(field1):
                allsigx = [self.unpack(sigtype(field2), conj=conj, ang_unit=ang_unit)[sigtype(field1)]]
            else:
                allsigx = [None]

            # Y error bars
            if sigtype(field2):
                allsigy = [self.unpack(sigtype(field2), conj=conj, ang_unit=ang_unit)[sigtype(field2)]]
            else:
                allsigy = [None]

        # make plot(s)
        if axis:
            x = axis
        else:
            fig=plt.figure()
            x = fig.add_subplot(1,1,1)

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
            if snrcut>0.:
                sigs = [sigx,sigy]
                for jj, field in enumerate([field1, field2]):
                    if field in FIELDS_AMPS:
                        fmask = data[field] / sigs[jj] > snrcut
                    elif field in FIELDS_PHASE:
                        fmask = sigs[jj] < (180./np.pi/snrcut)
                    elif field in FIELDS_SNRS:
                        fmask = data[field] > snrcut                
                    else:
                        fmask = np.ones(mask.shape).astype(bool)
                    mask *= fmask

            data = data[mask]
            if not sigy is None: sigy = sigy[mask]
            if not sigx is None: sigx = sigx[mask]
            if len(data) == 0:
                continue

            xmins.append(np.min(data[field1]))
            xmaxes.append(np.max(data[field1]))
            ymins.append(np.min(data[field2]))
            ymaxes.append(np.max(data[field2]))

            # Plot the data
            tolerance = len(data[field2])

            if label is None:
                labelstr="%s-%s"%((str(bl[0]),str(bl[1])))

            else:
                labelstr=str(label)

            if ebar and (np.any(sigy) or np.any(sigx)):
                x.errorbar(data[field1], data[field2], xerr=sigx, yerr=sigy, label=labelstr,
                           fmt=marker, markersize=markersize, color=color,picker=tolerance)
            else:
                x.plot(data[field1], data[field2], marker, markersize=markersize, color=color,
                       label=labelstr, picker=tolerance)

        # Data ranges
        if not rangex:
            rangex = [np.min(xmins) - 0.2 * np.abs(np.min(xmins)),
                      np.max(xmaxes) + 0.2 * np.abs(np.max(xmaxes))]
            if np.any(np.isnan(np.array(rangex))):
                print("Warning: NaN in data x range: specifying rangex to default")
                rangex = [-100,100]

        if not rangey:
            rangey = [np.min(ymins) - 0.2 * np.abs(np.min(ymins)),
                      np.max(ymaxes) + 0.2 * np.abs(np.max(ymaxes))]
            if np.any(np.isnan(np.array(rangey))):
                print("Warning: NaN in data y range: specifying rangey to default")
                rangey = [-100,100]

        x.set_xlim(rangex)
        x.set_ylim(rangey)

        # label and save
        if axislabels:
            try:
                x.set_xlabel(FIELD_LABELS[field1])
                x.set_ylabel(FIELD_LABELS[field2])
            except:
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
        if show:
            plt.show(block=False)

        return x

    def plot_bl(self, site1, site2, field,
                      debias=True, ang_unit='deg', timetype=False,
                      axis=False, rangex=False, rangey=False, snrcut=0.,
                      color=SCOLORS[0], marker='o', markersize=MARKERSIZE, label=None,
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

        if timetype==False:
            timetype=self.timetype
        if ang_unit=='deg': angle=DEGREE
        else: angle = 1.0

        if field=='amp' and not (self.amp is None):
            print ("Warning: plot_bl is not using amplitudes in Obsdata.amp array!")

        if label is None:
            label=str(self.source)
        else:
            label=str(label)

        # Determine if fields are valid
        if field not in FIELDS:
            raise Exception("valid fields are " + string.join(FIELDS))

        plotdata = self.unpack_bl(site1, site2, field, ang_unit=ang_unit, debias=debias, timetype=timetype)
        sigmatype = sigtype(field)
        if sigtype(field):
            errdata = self.unpack_bl(site1, site2, sigtype(field), ang_unit=ang_unit, debias=debias)
        else:
            errdata = None

        # Flag out nans (to avoid problems determining plotting limits)
        mask = ~np.isnan(plotdata[field][:,0])

        # Flag out due to snrcut
        if snrcut>0.:
            if field in FIELDS_AMPS:
                fmask = plotdata[field] / errdata[sigmatype] > snrcut
            elif field in FIELDS_PHASE:
                fmask = errdata[sigmatype] < (180./np.pi/snrcut)
            elif field in FIELDS_SNRS:
                fmask = plotdata[field] > snrcut                
            else:
                fmask = np.ones(mask.shape).astype(bool)
            fmask = fmask[:,0]
            mask *= fmask

        plotdata = plotdata[mask]
        if not errdata is None: 
            errdata = errdata[mask]

        if not rangex:
            rangex = [self.tstart,self.tstop]
            if np.any(np.isnan(np.array(rangex))):
                print("Warning: NaN in data x range: specifying rangex to default")
                rangex = [0,24]
        if not rangey:
            rangey = [np.min(plotdata[field]) - 0.2 * np.abs(np.min(plotdata[field])),
                      np.max(plotdata[field]) + 0.2 * np.abs(np.max(plotdata[field]))]
            if np.any(np.isnan(np.array(rangey))):
                print("Warning: NaN in data y range: specifying rangex to default")
                rangey = [-100,100]

        # Plot the data
        if axis:
            x = axis
        else:
            fig = plt.figure()
            x = fig.add_subplot(1,1,1)

        if ebar and sigtype(field)!=False:
            x.errorbar(plotdata['time'][:,0], plotdata[field][:,0],yerr=errdata[sigtype(field)][:,0],
                       fmt=marker, markersize=markersize, color=color, linestyle='none', label=label)
        else:
            x.plot(plotdata['time'][:,0], plotdata[field][:,0], marker, markersize=markersize,
                   color=color, label=label, linestyle='none')

        x.set_xlim(rangex)
        x.set_ylim(rangey)

        if axislabels:
            x.set_xlabel(self.timetype + ' (hr)')
            try:
                x.set_ylabel(FIELD_LABELS[field])
            except:
                x.set_ylabel(field.capitalize())
            x.set_title('%s - %s'%(site1,site2))

        if grid:
            x.grid()
        if legend:
            plt.legend()
        if export_pdf != "" and not axis:
            fig.savefig(export_pdf, bbox_inches='tight')
        if show:
            plt.show(block=False)

        return x

    def plot_cphase(self, site1, site2, site3,
                          vtype='vis', cphases=[], force_recompute=False, 
                          ang_unit='deg', timetype=False, snrcut=0.,
                          axis=False, rangex=False, rangey=False,
                          color=SCOLORS[0], marker='o', markersize=MARKERSIZE, label=None,
                          grid=True, ebar=True, axislabels=True, legend=False, 
                          show=True, export_pdf=""):

        """Plot a field over time on a baseline site1-site2.

           Args:
               site1 (str): station 1 name
               site2 (str): station 2 name
               field (str): y-axis field (from FIELDS)

               vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble bispectra
               cphases (list): optionally pass in the closure phases so they don't have to be recomputed
               force_recompute (bool): if True, recompute closure phases instead of using stored data
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

        if timetype==False:
            timetype=self.timetype
        if ang_unit=='deg': angle=1.0
        else: angle = DEGREE

        if label is None:
            label=str(self.source)
        else:
            label=str(label)

        # Get closure phases (maximal set)
        if (len(cphases)==0) and not (self.cphase is None) and not force_recompute:
            cphases=self.cphase

        cpdata = self.cphase_tri(site1, site2, site3, vtype=vtype, timetype=timetype, 
                                 cphases=cphases, force_recompute=force_recompute, snrcut=snrcut)
        plotdata = np.array([[obs['time'],obs['cphase']*angle,obs['sigmacp']] for obs in cpdata])

        nan_mask = np.isnan(plotdata[:,1])
        plotdata = plotdata[~nan_mask]

        if len(plotdata) == 0:
            print("%s %s %s : No closure phases on this triangle!" % (site1,site2,site3))
            return

        # Plot the data
        if axis:
            x = axis
        else:
            fig=plt.figure()
            x = fig.add_subplot(1,1,1)

        # Data ranges
        if not rangex:
            rangex = [self.tstart,self.tstop]
            if np.any(np.isnan(np.array(rangex))):
                print("Warning: NaN in data x range: specifying rangex to default")
                rangex = [0, 24]

        if not rangey:
            if ang_unit=='deg':rangey = [-190,190]
            else: rangey=[-1.1*np.pi,1.1*np.pi]

        x.set_xlim(rangex)
        x.set_ylim(rangey)


        if ebar and np.any(plotdata[:,2]):
            x.errorbar(plotdata[:,0], plotdata[:,1], yerr=plotdata[:,2], fmt=marker, markersize=markersize,
                       color=color, linestyle='none',label=label)
        else:
            x.plot(plotdata[:,0], plotdata[:,1], marker, markersize=markersize, color=color, linestyle='none', label=label)

        if axislabels:
            x.set_xlabel(self.timetype + ' (h)')
            if ang_unit=='deg':
                x.set_ylabel(r'Closure Phase $(^\circ)$')
            else:
                x.set_ylabel(r'Closure Phase (radian)')

        x.set_title('%s - %s - %s' % (site1,site2,site3))

        if grid:
            x.grid()
        if legend:
            plt.legend()
        if export_pdf != "" and not axis:
            fig.savefig(export_pdf, bbox_inches='tight')
        if show:
            plt.show(block=False)

        return x

    def plot_camp(self, site1, site2, site3, site4,
                        vtype='vis', ctype='camp', camps=[], force_recompute=False,
                        debias=False, timetype=False, snrcut=0.,
                        axis=False, rangex=False, rangey=False,
                        color=SCOLORS[0], marker='o', markersize=MARKERSIZE, label=None,
                        grid=True, ebar=True,axislabels=True, legend=False, 
                        show=True,export_pdf=""):

        """Plot closure phase over time on a quadrange (1-2)(3-4)/(1-4)(2-3).

           Args:
               site1 (str): station 1 name
               site2 (str): station 2 name
               site3 (str): station 3 name
               site4 (str): station 4 name

               vtype (str): The visibilty type ('vis','qvis','uvis','vvis','pvis') from which to assemble closure amplitudes
               ctype (str): The closure amplitude type ('camp' or 'logcamp')
               camps (list): optionally pass in camps so they don't have to be recomputed
               force_recompute (bool): if True, recompute closure amplitudes instead of using stored data
               snrcut (float): flag closure amplitudes with snr lower than this               

               debias (bool): If True, debias the closure amplitude - the individual visibility amplitudes are always debiased.
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

        if timetype==False:
            timetype=self.timetype
        if label is None:
            label=str(self.source)

        else:
            label=str(label)

        # Get closure amplitudes (maximal set)
        if ((ctype=='camp') and (len(camps)==0) and not (self.camp is None) and not (len(self.camp)==0) and not force_recompute):
            camps=self.camp
        elif ((ctype=='logcamp') and (len(camps)==0) and not (self.logcamp is None) and not (len(self.logcamp)==0) and not force_recompute):
            camps=self.logcamp

        # Get closure amplitudes (maximal set)
        cpdata = self.camp_quad(site1, site2, site3, site4,
                                vtype=vtype, ctype=ctype, snrcut=snrcut,
                                debias=debias, timetype=timetype,
                                camps=camps,force_recompute=force_recompute)

        if len(cpdata) == 0:
            print('No closure amplitudes on this triangle!')
            return

        plotdata = np.array([[obs['time'],obs['camp'],obs['sigmaca']] for obs in cpdata])
        plotdata = np.array(plotdata)

        nan_mask = np.isnan(plotdata[:,1])
        plotdata = plotdata[~nan_mask]

        if len(plotdata) == 0:
            print("No closure amplitudes on this quadrangle!")
            return

        # Data ranges
        if not rangex:
            rangex = [self.tstart,self.tstop]
            if np.any(np.isnan(np.array(rangex))):
                print("Warning: NaN in data x range: specifying rangex to default")
                rangex = [0, 24]

        if not rangey:
            rangey = [np.min(plotdata[:,1]) - 0.2 * np.abs(np.min(plotdata[:,1])),
                      np.max(plotdata[:,1]) + 0.2 * np.abs(np.max(plotdata[:,1]))]
            if np.any(np.isnan(np.array(rangey))):
                print("Warning: NaN in data y range: specifying rangey to default")
                if ctype=='camp': rangey = [0,100]
                if ctype=='logcamp': rangey = [-10,10]

        # Plot the data
        if axis:
            x = axis
        else:
            fig=plt.figure()
            x = fig.add_subplot(1,1,1)

        if ebar and np.any(plotdata[:,2]):
            x.errorbar(plotdata[:,0], plotdata[:,1], yerr=plotdata[:,2], fmt=marker, markersize=markersize,
                       color=color, linestyle='none',label=label)
        else:
            x.plot(plotdata[:,0], plotdata[:,1],marker, markersize=markersize, color=color, linestyle='none',label=label)

        x.set_xlim(rangex)
        x.set_ylim(rangey)

        if axislabels:
            x.set_xlabel(self.timetype + ' (h)')
            if ctype=='camp':
                x.set_ylabel('Closure Amplitude')
            elif ctype=='logcamp':
                x.set_ylabel('Log Closure Amplitude')
            x.set_title('(%s - %s)(%s - %s)/(%s - %s)(%s - %s)'%(site1,site2,site3,site4,
                                                                 site1,site4,site2,site3))
        if grid:
            x.grid()
        if legend:
            plt.legend()
        if export_pdf != "" and not axis:
            fig.savefig(export_pdf, bbox_inches='tight')
        if show:
            plt.show(block=False)
            return
        else:
            return x

    def save_txt(self, fname):

        """Save visibility data to a text file.

           Args:
                fname (str): path to output text file
        """

        ehtim.io.save.save_obs_txt(self,fname)

        return

    def save_uvfits(self, fname, force_singlepol=False, polrep_out='circ'):

        """Save visibility data to uvfits file.

           Args:
                fname (str): path to output text file
                force_singlepol (str): if 'R' or 'L', will interpret stokes I field as 'RR' or 'LL'
                polrep_out (str): 'circ' or 'stokes': how data should be stored in the uvfits file
        """

        if force_singlepol!=False and self.polrep!='stokes':
            raise Exception("force_singlepol is incompatible with polrep!='stokes'")

        ehtim.io.save.save_obs_uvfits(self,fname,force_singlepol=force_singlepol, polrep_out=polrep_out)

        return

    def save_oifits(self, fname, flux=1.0):

        """ Save visibility data to oifits. Polarization data is NOT saved.

            Args:
                fname (str): path to output text file
                flux (float): normalization total flux
        """

        if self.polrep!='stokes':
            raise Exception("save_oifits not yet implemented for polreps other than 'stokes'")

        #Antenna diameter currently incorrect and the exact times are not correct in the datetime object
        ehtim.io.save.save_obs_oifits(self, fname, flux=flux)

        return

##################################################################################################
# Observation creation functions
##################################################################################################
def merge_obs(obs_List):

    """Merge a list of observations into a single observation file.

       Args:
           obs_List (list): list of split observation Obsdata objects.

       Returns:
           mergeobs (Obsdata): merged Obsdata object containing all scans in input list
    """

    if (len(set([obs.polrep for obs in obs_List])) > 1):
        raise Exception("All observations must have the same polarization representaiton !")
        return

    if (len(set([obs.ra for obs in obs_List])) > 1 or
        len(set([obs.dec for obs in obs_List])) > 1 or
        len(set([obs.rf for obs in obs_List])) > 1 or
        len(set([obs.bw for obs in obs_List])) > 1 or
        len(set([obs.source for obs in obs_List])) > 1 or
        len(set([np.floor(obs.mjd) for obs in obs_List])) > 1):

        raise Exception("All observations must have the same parameters!")
        return

    # The important things to merge are the mjd, the data, and the list of telescopes
    data_merge = np.hstack([obs.data for obs in obs_List])

    scan_merge = []
    for obs in obs_List:
        if not (scan_merge is None):
            scan_merge.append(obs.scans)
    scan_merge = np.hstack(scan_merge)
    tarr_merge = np.unique(np.concatenate([obs.tarr for obs in obs_List]))

    arglist, argdict = obs_List[0].obsdata_args()
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
                channel=all, IF=all, polrep='stokes', allow_singlepol=True):

    """Load observation data from a uvfits file.
       Args:
           fname (str): path to input text file
           flipbl (bool): flip baseline phases if True.
           remove_nan (bool): True to remove nans from missing polarizations
           polrep (str): load data as either 'stokes' or 'circ'
           force_singlepol (str): 'R' or 'L' to load only 1 polarization
           channel (list): list of channels to average in the import. channel=all averages all
           IF (list): list of IFs to  average in  the import. IF=all averages all IFS

       Returns:
           obs (Obsdata): Obsdata object loaded from file
    """

    return ehtim.io.load.load_obs_uvfits(fname, flipbl=flipbl, force_singlepol=force_singlepol, 
                                         channel=channel, IF=IF, polrep=polrep, 
                                         remove_nan=remove_nan, allow_singlepol=allow_singlepol)

def load_oifits(fname, flux=1.0):

    """Load data from an oifits file. Does NOT currently support polarization.

       Args:
           fname (str): path to input text file
           flux (float): normalization total flux

       Returns:
           obs (Obsdata): Obsdata object loaded from file
    """

    return ehtim.io.load.load_obs_oifits(fname, flux=flux)

def load_maps(arrfile, obsspec, ifile, qfile=0, ufile=0, vfile=0, src=SOURCE_DEFAULT, mjd=MJD_DEFAULT, ampcal=False, phasecal=False):

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

    return ehtim.io.load.load_obs_maps(arrfile, obsspec, ifile, qfile=qfile, ufile=ufile, vfile=vfile,
                                       src=src, mjd=mjd, ampcal=ampcal, phasecal=phasecal)
