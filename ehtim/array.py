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

from builtins import str
from builtins import range
from builtins import object

import numpy as np

import ehtim.observing.obs_simulate as simobs
import ehtim.io.save
import ehtim.io.load

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

###########################################################################################################################################
# Array object
###########################################################################################################################################
class Array(object):
    """A VLBI array of telescopes with site locations, SEFDs, and other data.

       Attributes:
           tarr (numpy.recarray): The array of telescope data with datatype DTARR
           tkey (dict): A dictionary of rows in the tarr for each site name
           ephem (dict): A dictionary of TLEs for each space antenna, Space antennas have x=y=z=0 in the tarr
    """

    def __init__(self, tarr, ephem={}):
        self.tarr = tarr
        self.ephem = ephem

        # check to see if ephemeris is correct
        for line in self.tarr:
            if np.any(np.isnan([line['x'],line['y'],line['z']])):
                sitename = str(line['site'])
                try:
                    elen = len(ephem[sitename])
                except NameError:
                    raise Exception ('no ephemeris for site %s !' % sitename)
                if elen != 3:

                    raise Exception ('wrong ephemeris format for site %s !' % sitename)

        # Dictionary of array indices for site names
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}

    def listbls(self):
        """List all baselines.
           
           Args:
           Returns: 
                numpy.array : array of baselines
        """

        bls = []
        for i1 in sorted(self.tarr['site']):
            for i2 in sorted(self.tarr['site']):
                if not ([i1,i2] in bls) and not ([i2,i1] in bls) and i1 != i2:
                    bls.append([i1,i2])

        return np.array(bls)

    def obsdata(self, ra, dec, rf, bw, tint, tadv, tstart, tstop, mjd=MJD_DEFAULT, timetype='UTC',
                      polrep='stokes', elevmin=ELEV_LOW, elevmax=ELEV_HIGH, 
                      tau=TAUDEF, fix_theta_GMST=False):

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
               polrep (str): polarization representation, either 'stokes' or 'circ'
               elevmin (float): station minimum elevation in degrees
               elevmax (float): station maximum elevation in degrees
               tau (float): the base opacity at all sites, or a dict giving one opacity per site

           Returns:
               Obsdata: an observation object with no data

        """

        obsarr = simobs.make_uvpoints(self, ra, dec, rf, bw,
                                            tint, tadv, tstart, tstop,
                                            mjd=mjd, polrep=polrep, tau=tau,
                                            elevmin=elevmin, elevmax=elevmax,
                                            timetype=timetype, fix_theta_GMST = fix_theta_GMST)

        uniquetimes = np.sort(np.unique(obsarr['time']))
        scans = np.array([[time-0.5*tadv, time+0.5*tadv] for time in uniquetimes]) 
        source=str(ra) + ":" + str(dec)
        obs = ehtim.obsdata.Obsdata(ra, dec, rf, bw, obsarr, self.tarr, 
                                    source=source, mjd=mjd, timetype=timetype, polrep=polrep,
                                    ampcal=True, phasecal=True, opacitycal=True, 
                                    dcal=True, frcal=True,
                                    scantable=scans)
        return obs

    def make_subarray(self, sites):
        """Make a subarray from the Array object array that only includes the sites listed in sites.

           Args:
               sites (list) : list of sites in the subarray
           Returns:
               Array: an Array object with specified sites and metadata
        """
        all_sites = [t[0] for t in self.tarr]
        mask = np.array([t in sites for t in all_sites])
        return Array(self.tarr[mask],ephem=self.ephem)

    def save_txt(self, fname):
        """Save the array data in a text file.

           Args:
               fname (str) : path to output array file
        """
        ehtim.io.save.save_array_txt(self,fname)
        return

###########################################################################################################################################
#Array creation functions
###########################################################################################################################################
def load_txt(fname, ephemdir='ephemeris'):
    """Read an array from a text file. 
       Sites with x=y=z=0 are spacecraft, TLE ephemerides read from ephemdir.

       Args:
           fname (str) : path to input array file
           ephemdir (str) : path to directroy with 2TLE ephemerides for spacecraft
       Returns:
           Array: an Array object loaded from file
    """

    return ehtim.io.load.load_array_txt(fname, ephemdir=ephemdir)
