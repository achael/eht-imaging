import numpy as np

import ehtim.observing.obs_simulate as simobs
import ehtim.io.save 
import ehtim.io.load

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

###########################################################################################################################################
#Array object
###########################################################################################################################################
class Array(object):
    """A VLBI array of telescopes with locations and SEFDs
    
        Attributes:
        tarr: The array of telescope data (name, x, y, z, sefdr,sefdl,dr,dl, fr_par_angle, fr_elev_angle, fr_offset)
        ephem: A dictionary of 2TLEs for each space antenna. Space antennas have x=y=z=0 in the tarr
        where x,y,z are geocentric coordinates.
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
        """List all baselines
        """
 
        bls = []
        for i1 in sorted(self.tarr['site']):
            for i2 in sorted(self.tarr['site']):
                if not ([i1,i2] in bls) and not ([i2,i1] in bls) and i1 != i2:
                    bls.append([i1,i2])
                    
        return np.array(bls)
            
    def obsdata(self, ra, dec, rf, bw, tint, tadv, tstart, tstop, mjd=MJD_DEFAULT, 
                      tau=TAUDEF, elevmin=ELEV_LOW, elevmax=ELEV_HIGH, timetype='UTC'):
        """Generate u,v points and baseline errors for the array.
           Return an Observation object with no visibilities.
           tstart and tstop are hrs in UTC
           tint and tadv are seconds.
           rf and bw are Hz
           ra is fractional hours
           dec is fractional degrees
           tau can be a single number or a dictionary giving one per site
        """
        obsarr = simobs.make_uvpoints(self, ra, dec, rf, bw, 
                                            tint, tadv, tstart, tstop, 
                                            mjd=MJD_DEFAULT, tau=TAUDEF, 
                                            elevmin=ELEV_LOW, elevmax=ELEV_HIGH, 
                                            timetype='UTC')

        obs = ehtim.obsdata.Obsdata(ra, dec, rf, bw, obsarr, self.tarr, 
                                    source=str(ra) + ":" + str(dec), 
                                    mjd=mjd, timetype=timetype)      
        return obs

    def make_subarray(self, sites):
        """Make a subarray from the Array object array that only includes the sites listed in sites
        """
        all_sites = [t[0] for t in self.tarr]   
        mask = np.array([t in sites for t in all_sites])
        return Array(self.tarr[mask])
     
    def save_txt(self, fname):
        """Save the array data in a text file
        """
        ehtim.io.save.save_array_txt(self,fname)
        return

###########################################################################################################################################
#Array creation functions
###########################################################################################################################################
def load_txt(fname, ephemdir='./ephemeris'):
    return ehtim.io.load.load_array_txt(fname, ephemdir=ephemdir)
