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

from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object

import string, copy
import numpy as np
import numpy.lib.recfunctions as rec
import matplotlib.pyplot as plt
import scipy.optimize as opt
import itertools as it
import sys, os

import ehtim.image
import ehtim.observing.obs_simulate
import ehtim.io.save
import ehtim.io.load

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

import scipy

##################################################################################################
# Caltable object
##################################################################################################
class Caltable(object):
    """
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

           data (dict): keys are sites in tarr, entries are calibration data tables of type DTCAL

    """

    def __init__(self, ra, dec, rf, bw, datadict, tarr, source=SOURCE_DEFAULT, mjd=MJD_DEFAULT, timetype='UTC'):

        """A Calibration Table.

           Args:
               ra (float): The source Right Ascension in fractional hours
               dec (float): The source declination in fractional degrees
               rf (float): The observation frequency in Hz
               mjd (int): The integer MJD of the observation
               bw (float): The observation bandwidth in Hz

               datadict (dict):  keys are sites in tarr, entries are calibration data tables of type DTCAL
               tarr (numpy.recarray): The array of telescope data with datatype DTARR

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
        self.tarr = tarr
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}

        # Save the data
        self.data = datadict

    def copy(self):
        """Copy the observation object.

           Args:

           Returns:
               (Caltable): a copy of the Caltable object.
        """
        new_caltable = Caltable(self.ra, self.dec, self.rf, self.bw, self.data, self.tarr, source=self.source, mjd=self.mjd, timetype=self.timetype)
        return new_caltable

    def plot_dterms(self, sites='all', label=None, legend=True, clist=SCOLORS,rangex=False,
                    rangey=False, markersize=2*MARKERSIZE, show=True, grid=True, export_pdf=""):

        # sites
        if sites in ['all' or 'All'] or sites==[]:
            sites = list(self.data.keys())

        if not type(sites) is list:
            sites = [sites]
        
        keys = [self.tkey[site] for site in sites]

        axes = plot_tarr_dterms(self.tarr, keys=keys, label=label, legend=legend, clist=clist,rangex=rangex,
                    rangey=rangey, markersize=markersize, show=show, grid=grid, export_pdf=export_pdf)
        
        return axes


    def plot_gains(self, sites, gain_type='amp', pol='R',label=None,
                   ang_unit='deg',timetype=False, yscale='log', legend=True,
                   clist=SCOLORS,rangex=False,rangey=False, markersize=[MARKERSIZE],
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

        if timetype==False:
            timetype=self.timetype
        if timetype not  in ['GMST','UTC','utc','gmst']:
            raise Exception("timetype should be 'GMST' or 'UTC'!")
        if gain_type not in ['amp','phase']:
            raise Exception("gain_type must be 'amp' or 'phase'  ")
        if pol not in ['R','L','both']:
            raise Exception("pol must be 'R' or 'L'")

        if ang_unit=='deg': angle=DEGREE
        else: angle = 1.0

        # axis
        if axis:
            x = axis
        else:
            fig = plt.figure()
            x = fig.add_subplot(1,1,1)

        # sites
        if sites in ['all' or 'All'] or sites==[]:
            sites = sorted(list(self.data.keys()))

        if not type(sites) is list:
            sites = [sites]
            
        if len(markersize)==1:
            markersize = markersize * np.ones(len(sites))


        # plot gain on each site
        tmins = tmaxes = gmins = gmaxes = []
        for s in range(len(sites)):
            site = sites[s]
            times = self.data[site]['time']
            if timetype in ['UTC','utc'] and self.timetype=='GMST':
                times = gmst_to_utc(times, self.mjd)
            elif timetype in ['GMST','gmst'] and self.timetype=='UTC':
                times = utc_to_gmst(times, self.mjd)
            if pol=='R':
                gains = self.data[site]['rscale']
            elif pol=='L':
                gains = self.data[site]['lscale']

            if gain_type=='amp':
                gains = np.abs(gains)
                ylabel = r'$|G|$'

            if gain_type=='phase':
                gains = np.angle(gains)/angle
                if ang_unit=='deg': ylabel = r'arg($|G|$) ($^\circ$)'
                else: ylabel = r'arg($|G|$) (radian)'

            tmins.append(np.min(times))
            tmaxes.append(np.max(times))
            gmins.append(np.min(gains))
            gmaxes.append(np.max(gains))

            # Plot the data
            if label is None:
                bllabel=str(site)
            else:
                bllabel = label + ' ' + str(site)
            plt.plot(times, gains, color=next(colors), marker='o', markersize=markersize[s], 
                     label=bllabel, linestyle='none')


        if not rangex:
            rangex = [np.min(tmins) - 0.2 * np.abs(np.min(tmins)),
                      np.max(tmaxes) + 0.2 * np.abs(np.max(tmaxes))]
            if np.any(np.isnan(np.array(rangex))):
                print("Warning: NaN in data x range: specifying rangex to default")
                rangex = [0,24]
        if not rangey:
            rangey = [np.min(gmins) - 0.2 * np.abs(np.min(gmins)),
                      np.max(gmaxes) + 0.2 * np.abs(np.max(gmaxes))]
            if np.any(np.isnan(np.array(rangey))):
                print("Warning: NaN in data x range: specifying rangey to default")
                rangey = [1.e-2,1.e2]

        plt.plot(np.linspace(rangex[0],rangex[1],5), np.ones(5),'k--')
        x.set_xlim(rangex)
        x.set_ylim(rangey)

        # labels
        if axislabels:
            x.set_xlabel(self.timetype + ' (hr)')
            x.set_ylabel(ylabel)
            plt.title('Caltable gains for %s on day %s' % (self.source, self.mjd))

        if legend:
            plt.legend()

        if yscale=='log':
            x.set_yscale('log')
        if grid:
            x.grid()
        if export_pdf != "" and not axis:
            fig.savefig(export_pdf, bbox_inches='tight')
        if show:
            plt.show(block=False)

        return x


    def enforce_positive(self, method='median', min_gain = 0.9, sites = [], verbose = True):
        """Enforce that caltable gains are not low (e.g., that sites are not significantly more sensitive than estimated).
           For site gains that are low, the entire gain curve is rescaled to enforce a specified minimum site gain.

           Args:
               caltab (Caltable): Input Caltable with station gains
               method (str): 'median', 'mean', or 'min'; determines how a representative gain from each site is computed.
               min_gain (float): Minimum acceptable gain. Site gains above this value will not be modified.
               sites (list): List of sites with gains to check and adjust. For sites=[], all sites will be examined.
               verbose (bool): If True, print corrections.

           Returns:
               (Caltable): Axes object with the plot
        """

        if len(sites) == 0:
            sites = self.data.keys()

        caltab_pos = self.copy()
        for site in self.data.keys():
            if not site in sites: continue
            if len(self.data[site]['rscale']) == 0: continue

            if method == 'min':
                sitemin = np.min([np.abs(self.data[site]['rscale']),np.abs(self.data[site]['lscale'])])
            elif method == 'mean':
                sitemin = np.mean([np.abs(self.data[site]['rscale']),np.abs(self.data[site]['lscale'])])
            elif method == 'median':
                sitemin = np.median([np.abs(self.data[site]['rscale']),np.abs(self.data[site]['lscale'])])
            else:
                print('Method ' + method + ' not recognized!')
                return caltab_pos

            if sitemin < min_gain:
                if verbose: print(method + ' gain for ' + site + ' is ' + str(sitemin) + '. Rescaling.')
                caltab_pos.data[site]['rscale'] /= sitemin
                caltab_pos.data[site]['lscale'] /= sitemin
            else:
                if verbose: print(method + ' gain for ' + site + ' is ' + str(sitemin) + '. Not adjusting.')

        return caltab_pos

    #TODO default extrapolation?
    def pad_scans(self, maxdiff=60, padtype='median'):
        
        """Pad data points around scans.

           Args:
               maxdiff (float): "scan" separation length (seconds)
               padtype (str): padding type, 'endval' or 'median'
           
           Returns:
               (Caltable):  a padded caltable object
        """
        import copy

        outdict = {}
        scopes = list(self.data.keys())
        for scope in scopes:
            if np.any(self.data[scope] is None) or len(self.data[scope]) == 0:
                continue

            caldata = copy.deepcopy(self.data[scope]) 

            # Gather data into "scans"
            # TODO we could use a scan table for this as well!
            gathered_data=[]
            scandata = [caldata[0]]
            for i in range(1, len(caldata)):
                if (caldata[i]['time']-caldata[i-1]['time'])*3600 > maxdiff:
                    scandata = np.array(scandata, dtype=DTCAL)
                    gathered_data.append(scandata)
                    scandata = [caldata[i]]
                else:
                    scandata.append(caldata[i])
                    
            # This adds the last scan        
            scandata = np.array(scandata)
            gathered_data.append(scandata)

            # Compute padding values and pad scans
            for i in range(len(gathered_data)):
                gg = gathered_data[i]

                medR = np.median(gg['rscale'])
                medL = np.median(gg['lscale'])

                timepre = gg['time'][0] - maxdiff/2./3600.
                timepost = gg['time'][-1] + maxdiff/2./3600.

                if padtype=='median': # pad with median scan value
                    medR = np.median(gg['rscale'])
                    medL = np.median(gg['lscale'])
                    preR = medR
                    postR = medR
                    preL = medL
                    postL = medL
                elif padtype=='endval': # pad with endpoints
                    preR = gg['rscale'][0]
                    postR = gg['rscale'][-1]
                    preL = gg['lscale'][0]
                    postL = gg['lscale'][-1]
                else:  # pad with ones
                    preR = 1.
                    postR = 1.
                    preL = 1.
                    postL = 1.

                valspre = np.array([(timepre,preR,preL)],dtype=DTCAL)
                valspost = np.array([(timepost,postR,postL)],dtype=DTCAL)

                gg = np.insert(gg,0,valspre)
                gg = np.append(gg,valspost)

                # output data table
                if i==0:
                    caldata_out = gg
                else:
                    caldata_out = np.append(caldata_out, gg)

            try:
                caldata_out # TODO: refractor to avoid using exception
            except NameError:
                print("No gathered_data")
            else:
                outdict[scope] = caldata_out

        return Caltable(self.ra, self.dec, self.rf, self.bw, outdict, self.tarr,
                        source=self.source, mjd=self.mjd, timetype=self.timetype)

    def applycal(self, obs, interp='linear', extrapolate=None, 
                       force_singlepol=False, copy_closure_tables=True):
        """Apply the calibration table to an observation.

           Args:
               obs (Obsdata): The observation  with data to be calibrated
               interp (str): Interpolation method ('linear','nearest','cubic')
               extrapolate (bool): If True, points outside interpolation range will be extrapolated.
               force_singlepol (str): If 'L' or 'R', will set opposite polarization gains equal to chosen polarization

           Returns:
               (Obsdata): the calibrated Obsdata object
        """
        if not (self.tarr == obs.tarr).all():
            raise Exception("The telescope array in the Caltable is not the same as in the Obsdata")

        if extrapolate is True: # extrapolate can be a tuple or numpy array
            fill_value = "extrapolate"
        else:
            fill_value = extrapolate

        obs_orig = obs.copy() # Need to do this before switch_polrep to keep tables
        orig_polrep = obs.polrep
        obs = obs.switch_polrep('circ')

        rinterp = {}
        linterp = {}
        skipsites = []
        for s in range(0, len(self.tarr)):
            site = self.tarr[s]['site']

            try:
                self.data[site]
            except KeyError:
                skipsites.append(site)
                print ("No Calibration  Data for %s !" % site)
                continue

            time_mjd = self.data[site]['time']/24.0 + self.mjd
            rinterp[site] = relaxed_interp1d(time_mjd, self.data[site]['rscale'],
                                             kind=interp, fill_value=fill_value, bounds_error=False)
            linterp[site] = relaxed_interp1d(time_mjd, self.data[site]['lscale'],
                                             kind=interp, fill_value=fill_value, bounds_error=False)

        bllist = obs.bllist()
        datatable = []
        for bl_obs in bllist:
            t1 = bl_obs['t1'][0]
            t2 = bl_obs['t2'][0]
            time_mjd = bl_obs['time']/24.0 + obs.mjd

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

            if len(datatable):
                datatable = np.hstack((datatable,bl_obs))
            else:
                datatable = bl_obs

        #calobs.data = np.array(datatable)
        calobs = ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, np.array(datatable), obs.tarr,
                                       polrep=obs.polrep, scantable=obs.scans, source=obs.source, mjd=obs.mjd,
                                       ampcal=obs.ampcal, phasecal=obs.phasecal, opacitycal=obs.opacitycal, 
                                       dcal=obs.dcal, frcal=obs.frcal,timetype=obs.timetype)
        calobs = calobs.switch_polrep(orig_polrep)

        if copy_closure_tables:
            calobs.camp = obs_orig.camp
            calobs.logcamp = obs_orig.logcamp
            calobs.cphase = obs_orig.cphase

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

        if extrapolate is True: # extrapolate can be a tuple or numpy array
            fill_value = "extrapolate"
        else:
            fill_value = extrapolate

        try:
            x=caltablelist.__iter__
        except AttributeError: caltablelist = [caltablelist]

        #self = ct
        #caltablelist = [ct2]

        tarr1 = self.tarr.copy()
        tkey1 = self.tkey.copy()
        data1 = self.data.copy()
        for caltable in caltablelist:

            #TODO check metadata!

            #TODO CHECK ARE THEY ALL REFERENCED TO SAME MJD???
            tarr2 = caltable.tarr.copy()
            tkey2 = caltable.tkey.copy()
            data2 = caltable.data.copy()
            sites2 = list(data2.keys())
            sites1 = list(data1.keys())
            for site in sites2:
                if site in sites1: # if site in both tables
                    #merge the data by interpolating
                    time1 = data1[site]['time']
                    time2 = data2[site]['time']


                    rinterp1 = relaxed_interp1d(time1, data1[site]['rscale'],
                                                kind=interp, fill_value=fill_value,bounds_error=False)
                    linterp1 = relaxed_interp1d(time1, data1[site]['lscale'],
                                                kind=interp, fill_value=fill_value,bounds_error=False)

                    rinterp2 = relaxed_interp1d(time2, data2[site]['rscale'],
                                                kind=interp, fill_value=fill_value,bounds_error=False)
                    linterp2 = relaxed_interp1d(time2, data2[site]['lscale'],
                                                kind=interp, fill_value=fill_value,bounds_error=False)

                    times_merge = np.unique(np.hstack((time1,time2)))

                    rscale_merge = rinterp1(times_merge) * rinterp2(times_merge)
                    lscale_merge = linterp1(times_merge) * linterp2(times_merge)


                    #put the merged data back in data1
                    #TODO can we do this faster?
                    datatable = []
                    for i in range(len(times_merge)):
                        datatable.append(np.array((times_merge[i], rscale_merge[i], lscale_merge[i]), dtype=DTCAL))
                    data1[site] = np.array(datatable)

                # sites not in both caltables
                else:
                    if site not in tkey1.keys():
                        tarr1 = np.append(tarr1,tarr2[tkey2[site]])
                    data1[site] =  data2[site]

            #update tkeys every time
            tkey1 =  {tarr1[i]['site']: i for i in range(len(tarr1))}

        new_caltable = Caltable(self.ra, self.dec, self.rf, self.bw, data1, tarr1, source=self.source, mjd=self.mjd, timetype=self.timetype)

        return new_caltable


    def save_txt(self, obs, datadir='.', sqrt_gains=False):
        """Saves a Caltable object to text files in the given directory
           Args:
               obs (Obsdata): The observation object associated with the Caltable
               datadir (str): directory to save caltable in
               sqrt_gains (bool): If True, we square gains before saving.

           Returns:
        """

        return save_caltable(self, obs, datadir=datadir, sqrt_gains=sqrt_gains)


    def scan_avg(self, obs, incoherent=True):
    
        sites = self.data.keys()
        ntele = len(sites)

        datatables = {}
        
        # iterate over each site
        for s in range(0,ntele):        
            site = sites[s]
            
            # make a list of times that is the same value for all points in the same scan
            times = self.data[site]['time']
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
                gains_l = self.data[site]['lscale']
                gains_r = self.data[site]['rscale']
                
                # if incoherent average then average the magnitude of gains
                if incoherent:
                    gains_l = np.abs(gains_l)
                    gains_r = np.abs(gains_r)
                
                # average the gains 
                gains_l_avg = np.mean(gains_l[np.array(times_stable==scan[0])])
                gains_r_avg = np.mean(gains_r[np.array(times_stable==scan[0])])
                
                # add them to a new datatable
                datatable.append(np.array((scan[0], gains_r_avg, gains_l_avg), dtype=DTCAL))
                
            datatables[site] = np.array(datatable)
            
        if len(datatables)>0:
            caltable = Caltable(obs.ra, obs.dec, obs.rf,
                            obs.bw, datatables, obs.tarr, source=obs.source,
                            mjd=obs.mjd, timetype=obs.timetype)
        else:
            caltable=False

        return caltable
        
    def invert_gains(self):
        
        sites = self.data.keys()
        ntele = len(sites)

        for s in range(0,ntele):
            site = sites[s]
            self.data[site]['rscale'] = 1/self.data[site]['rscale']
            self.data[site]['lscale'] = 1/self.data[site]['lscale']

        return self
        

def load_caltable(obs, datadir, sqrt_gains=False ):
    """Load apriori Caltable object from text files in the given directory
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
        filename = datadir + obs.source + '_' + site + '.txt'
        try:
            data = np.loadtxt(filename, dtype=bytes).astype(str)
        except IOError:
            continue

        #print ("filename)
        datatable = []


        for row in data:

            time = (float(row[0]) - obs.mjd) * 24.0 # time is given in mjd

            # Maciek's old convention had a square root
            #rscale = np.sqrt(float(row[1])) # r
            #lscale = np.sqrt(float(row[2])) # l

            if len(row) == 3:
                rscale = float(row[1])
                lscale = float(row[2])
            elif len(row) == 5:
                rscale = float(row[1]) + 1j*float(row[2])
                lscale = float(row[3]) + 1j*float(row[4])
            else:
                raise Exception("cannot load caltable -- format unknown!")
            if sqrt_gains:
                rscale = rscale**.5
                lscale = lscale**.5
            datatable.append(np.array((time, rscale, lscale), dtype=DTCAL))
            #if onerowonly:
            #    datatable.append(np.array((1.1*time, rscale, lscale), dtype=DTCAL))

        datatables[site] = np.array(datatable)
    if len(datatables)>0:
        caltable = Caltable(obs.ra, obs.dec, obs.rf, obs.bw, datatables, tarr, source=obs.source, mjd=obs.mjd, timetype=obs.timetype)
    else:
        print ("COULD NOT FIND CALTABLE IN DIRECTORY %s" % datadir)
        caltable=False
    return caltable

def save_caltable(caltable, obs, datadir='.', sqrt_gains=False):
    """Saves a Caltable object to text files in the given directory
       Args:
           obs (Obsdata): The observation object associated with the Caltable
           datadir (str): directory to save caltable in
           sqrt_gains (bool): If True, we square gains before saving.

       Returns:
    """

    if not os.path.exists(datadir):
        os.makedirs(datadir)
        
    ehtim.io.save.save_array_txt(obs.tarr, datadir + '/array.txt')

    datatables = caltable.data
    src = caltable.source
    for site_info in caltable.tarr:
        site = site_info['site']

        if len(datatables.get(site, [])) == 0:
            continue

        filename = datadir + '/' + src + '_' + site +'.txt'
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
            outline = str(float(time)) + ' ' + str(float(rreal)) + ' ' + str(float(rimag)) + ' ' + str(float(lreal)) + ' ' + str(float(limag)) + '\n'
            outfile.write(outline)
        outfile.close()

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
    for s in range(0,ntele):
        datatable = []
        for t in range(0,ntimes):
            gain = gains[s*ntele + t]
            datatable.append(np.array((times[t], gain, gain), dtype=DTCAL))
        datatables[sites[s]] = np.array(datatable)
    if len(datatables)>0:
        caltable = Caltable(obs.ra, obs.dec, obs.rf,
                        obs.bw, datatables, obs.tarr, source=obs.source,
                        mjd=obs.mjd, timetype=obs.timetype)
    else:
        caltable=False

    return caltable

def relaxed_interp1d(x, y, **kwargs):
    if len(x) == 1:
        x = np.array([-0.5, 0.5]) + x[0]
        y = np.array([ 1.0, 1.0]) * y[0]
    return scipy.interpolate.interp1d(x, y, **kwargs)
    

def plot_tarr_dterms(tarr, keys=None, label=None, legend=True, clist=SCOLORS,rangex=False,
                rangey=False, markersize=2*MARKERSIZE, show=True, grid=True, export_pdf="", auto_order=True):

    if auto_order:
        # Ensure that the plot will put the stations in alphabetical order
        keys = np.argsort(tarr['site']) #range(len(tarr))
    else:
        keys = range(len(tarr))

    colors = iter(clist)
    
    if export_pdf != "":
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(16,8))
    else: 
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)

    for key in keys:
        
        # get the label
        site = str(tarr[key]['site'])
        if label is None:
            bllabel=str(site)
        else:
            bllabel = label + ' ' + str(site)
        color = next(colors)

        axes[0].plot(np.real(tarr[key]['dr']), np.imag(tarr[key]['dr']), color=color, marker='o', markersize=markersize, 
                 label=bllabel, linestyle='none')
        axes[0].set_title("Right D-terms")
        axes[0].set_xlabel("Real")
        axes[0].set_ylabel("Imaginary");
              
        axes[1].plot(np.real(tarr[key]['dl']), np.imag(tarr[key]['dl']), color=color, marker='o', markersize=markersize, 
                 label=bllabel, linestyle='none')
        axes[1].set_title("Left D-terms")
        axes[1].set_xlabel("Real")
        axes[1].set_ylabel("Imaginary");
    
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
        #axes[0].legend()
        axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if export_pdf != "":
        fig.savefig(export_pdf, bbox_inches='tight')
    
    return axes
    
def plot_compare_gains(caltab1, caltab2, obs, sites='all', pol='R', gain_type='amp', ang_unit='deg',
                    scan_avg=True, site_name_dict=None, fontsize= 13, legend_fontsize = 13,
                    yscale='log', legend=True, clist=SCOLORS,rangex=False,rangey=False, scalefac=[0.9, 1.1],
                    markersize=[2*MARKERSIZE], show=True, grid=False, axislabels=True, remove_ticks=False, axis=False, 
                    export_pdf=""):

    colors = iter(clist)
    
    if ang_unit=='deg': angle=DEGREE
    else: angle = 1.0
    
    # axis
    if axis:
        x = axis
    else:
        fig = plt.figure()
        x = fig.add_subplot(1,1,1)
    
    if scan_avg: 
        caltab1 = caltab1.scan_avg(obs, incoherent=True)
        caltab2 = caltab2.scan_avg(obs, incoherent=True)
    
    # sites
    if sites in ['all' or 'All'] or sites==[]:
        sites = list(set(caltab1.data.keys()).intersection(caltab2.data.keys() ))

    if not type(sites) is list:
        sites = [sites]
    
    if site_name_dict == None:
        print('hi')
        site_name_dict = {}
        for site in sites:
            site_name_dict[site] = site
            
    if len(markersize)==1:
        markersize = markersize * np.ones(len(sites))

    maxgain = 0.0
    mingain = 10000
    
    for s in range(len(sites)):
    
        site = sites[s]
        if pol=='R':
            gains1 = caltab1.data[site]['rscale']
            gains2 = caltab2.data[site]['rscale']
        elif pol=='L':
            gains1 = caltab1.data[site]['lscale']
            gains2 = caltab2.data[site]['lscale']

        if gain_type=='amp':
            gains1 = np.abs(gains1)
            gains2 = np.abs(gains2)
            ylabel = 'Amplitudes' #r'$|G|$'

        if gain_type=='phase':
            gains1 = np.angle(gains1)/angle
            gains2 = np.angle(gains2)/angle
            if ang_unit=='deg': ylabel = r'arg($|G|$) ($^\circ$)'
            else: ylabel = 'Phases (radian)' #r'arg($|G|$) (radian)'

        
        # print a line
        maxgain = np.nanmax([maxgain, np.nanmax(gains1), np.nanmax(gains2)])
        mingain = np.nanmin([mingain, np.nanmin(gains1), np.nanmin(gains2)])
        
        # mark the gains on the plot
        plt.plot(gains1, gains2, marker='.', linestyle='None',color=next(colors), markersize=markersize[s], label=site_name_dict[site])
        
        
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    plt.axes().set_aspect('equal')
    
    if rangex: 
        x.set_xlim(rangex)
    else:
        x.set_xlim([mingain*scalefac[0], maxgain*scalefac[1]])
    if rangey: 
        x.set_ylim(rangey)
    else:
        x.set_ylim([mingain*scalefac[0], maxgain*scalefac[1]])
    
    plt.plot([mingain*scalefac[0], maxgain*scalefac[1]], [mingain*scalefac[0], maxgain*scalefac[1]], 'grey', linewidth=1) 

    # labels
    if axislabels:
        x.set_xlabel('Ground Truth Gain ' + ylabel, fontsize=fontsize)
        x.set_ylabel('Recovered Gain ' + ylabel, fontsize=fontsize)
    else:
        x.tick_params(axis="y",direction="in", pad=-30)
        x.tick_params(axis="x",direction="in", pad=-18)
    
    if remove_ticks:
        plt.setp(x.get_xticklabels(), visible=False)
        plt.setp(x.get_yticklabels(), visible=False)

    if legend:
        plt.legend(frameon=False, fontsize=legend_fontsize)

    if yscale=='log':
        x.set_yscale('log')
        x.set_xscale('log')
    if grid:
        x.grid()
    if export_pdf != "" and not axis:
        fig.savefig(export_pdf, bbox_inches='tight')
    if show:
        plt.show(block=False)    
               

    return x
        



