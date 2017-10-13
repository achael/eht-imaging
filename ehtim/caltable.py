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
import sys

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
    """

    def __init__(self, ra, dec, rf, bw, datatables, tarr, source=SOURCE_DEFAULT, mjd=MJD_DEFAULT, timetype='UTC'):
        """A polarimetric VLBI observation of visibility amplitudes and phases (in Jy).

           Args:


           Returns:
               caltable (Caltable): an Caltable object
        """

        if len(datatables) == 0:
            raise Exception("No data in input table!")

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
        self.tarr = tarr

        # Dictionary of array indices for site names
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}

        # Save the data
        self.data = datatables

    def copy(self):
        """Copy the observation object.

           Args:

           Returns:
               (Caltable): a copy of the Caltable object.
        """
        new_caltable = Caltable(self.ra, self.dec, self.rf, self.bw, self.data, self.tarr, source=self.source, mjd=self.mjd, timetype=self.timetype)
        return new_caltable

    def applycal(self, obs, interp='linear', extrapolate=None, force_singlepol = False):

        if not (self.tarr == obs.tarr).all():
            raise Exception("The telescope array in the Caltable is not the same as in the Obsdata")

        if extrapolate is True: # extrapolate can be a tuple or numpy array
            fill_value = "extrapolate"
        else:
            fill_value = extrapolate

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
            rinterp[site] = scipy.interpolate.interp1d(time_mjd, self.data[site]['rscale'],
                                                       kind=interp, fill_value=fill_value)
            linterp[site] = scipy.interpolate.interp1d(time_mjd, self.data[site]['lscale'],
                                                       kind=interp, fill_value=fill_value)

        bllist = obs.bllist()
        datatable = []
        for bl_obs in bllist:
            t1 = bl_obs['t1'][0]
            t2 = bl_obs['t2'][0]
            time_mjd = bl_obs['time']/24.0 + obs.mjd

            if t1 in skipsites:
                rscale1 = lscale1 = 1
            else:
                rscale1 = rinterp[t1](time_mjd)
                lscale1 = linterp[t1](time_mjd)
            if t2 in skipsites:
                rscale2 = lscale2 = 1
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

            rrvis = (bl_obs['vis']  +    bl_obs['vvis']) * rrscale
            llvis = (bl_obs['vis']  -    bl_obs['vvis']) * llscale
            rlvis = (bl_obs['qvis'] + 1j*bl_obs['uvis']) * rlscale
            lrvis = (bl_obs['qvis'] - 1j*bl_obs['uvis']) * lrscale

            bl_obs['vis']  = 0.5  * (rrvis + llvis)
            bl_obs['qvis'] = 0.5  * (rlvis + lrvis)
            bl_obs['uvis'] = 0.5j * (lrvis - rlvis)
            bl_obs['vvis'] = 0.5  * (rrvis - llvis)

            rrsigma = np.sqrt(bl_obs['sigma']**2 + bl_obs['vsigma']**2) * np.abs(rrscale)
            llsigma = np.sqrt(bl_obs['sigma']**2 + bl_obs['vsigma']**2) * np.abs(llscale)
            rlsigma = np.sqrt(bl_obs['qsigma']**2 + bl_obs['usigma']**2) * np.abs(rlscale)
            lrsigma = np.sqrt(bl_obs['qsigma']**2 + bl_obs['usigma']**2) * np.abs(lrscale)

            bl_obs['sigma']  = 0.5 * np.sqrt( rrsigma**2 + llsigma**2 )
            bl_obs['qsigma'] = 0.5 * np.sqrt( rlsigma**2 + lrsigma**2 )
            bl_obs['usigma'] = 0.5 * np.sqrt( lrsigma**2 + rlsigma**2 )
            bl_obs['vsigma'] = 0.5 * np.sqrt( rrsigma**2 + llsigma**2 )

            if len(datatable):
                datatable = np.hstack((datatable,bl_obs))
            else:
                datatable = bl_obs

        calobs = ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, np.array(datatable), obs.tarr, source=obs.source, mjd=obs.mjd)

        return calobs

    def save_txt(self, obs, datadir = '', sqrt_gains=False):
        """Saves a Caltable object to text files in the format src_site.txt given by Maciek's tables
        """

        save_caltable(self, obs, datadir = datadir, sqrt_gains=sqrt_gains)

def load_caltable(obs, datadir, sqrt_gains=False ):
    """Load Maciek's apriori cal tables
    """

    datatables = {}
    for s in range(0, len(obs.tarr)):

        site = obs.tarr[s]['site']
        filename = datadir + obs.source + '_' + site + '.txt'
        data = np.loadtxt(filename, dtype=bytes).astype(str)

        datatable = []
        for row in data:

            time = (float(row[0]) - obs.mjd) * 24.0 # time is given in mjd

             # Maciek's old convention had a square root
 #           rscale = np.sqrt(float(row[1])) # r
 #           lscale = np.sqrt(float(row[2])) # l
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

        datatables[site] = np.array(datatable)

    caltable = Caltable(obs.ra, obs.dec, obs.rf, obs.bw, datatables, obs.tarr, source=obs.source, mjd=obs.mjd,
                        timetype=obs.timetype)

    return caltable

def save_caltable(caltable, obs, datadir = '', sqrt_gains=False):
    """Saves a Caltable object to text files in the format src_site.txt given by Maciek's tables
    """
    import os 
    if not os.path.exists(datadir):
        os.makedirs(datadir)

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
#            outline = str(float(time)) + ' ' + str(float(rscale)) + ' ' + str(float(lscale)) + '\n'
            outline = str(float(time)) + ' ' + str(float(rreal)) + ' ' + str(float(rimag)) + ' ' + str(float(lreal)) + ' ' + str(float(limag)) + '\n'
            outfile.write(outline)
        outfile.close()
      
