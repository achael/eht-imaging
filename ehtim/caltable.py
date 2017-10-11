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
# Obsdata object
##################################################################################################

DTCAL = [('time','f8'),('rscale','f8'),('lscale','f8')]
 

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
          
    def applycal(self, obs, interp='linear'):
    
        if not (self.tarr == obs.tarr).all():
            raise Exception("The telescope array in the Caltable is not the same as in the Obsdata")
         
        rinterp = {}
        linterp = {}
        for s in range(0, len(self.tarr)):
    
            site = self.tarr[s]['site']  
            time_mjd = self.data[site]['time']/24.0 + self.mjd
            rinterp[site] = scipy.interpolate.interp1d(time_mjd, self.data[site]['rscale'], kind=interp)
            linterp[site] = scipy.interpolate.interp1d(time_mjd, self.data[site]['lscale'], kind=interp)

            
        bllist = obs.bllist()
        datatable = []
        for bl_obs in bllist:
        
            t1 = bl_obs['t1'][0] 
            t2 = bl_obs['t2'][0]
            time_mjd = bl_obs['time']/24.0 + obs.mjd
            
            rrscale = np.sqrt(rinterp[t1](time_mjd) * rinterp[t2](time_mjd) )
            llscale = np.sqrt(linterp[t1](time_mjd) * linterp[t2](time_mjd) )
            rlscale = np.sqrt(rinterp[t1](time_mjd) * linterp[t2](time_mjd) )
            lrscale = np.sqrt(linterp[t1](time_mjd) * rinterp[t2](time_mjd) )
            
            rrvis = (bl_obs['vis']  +    bl_obs['vvis']) * rrscale
            llvis = (bl_obs['vis']  -    bl_obs['vvis']) * llscale
            rlvis = (bl_obs['qvis'] + 1j*bl_obs['uvis']) * rlscale
            lrvis = (bl_obs['qvis'] - 1j*bl_obs['uvis']) * lrscale
            
            bl_obs['vis'] =  0.5  * (rrvis + llvis) 
            bl_obs['qvis'] = 0.5  * (rlvis + lrvis)
            bl_obs['uvis'] = 0.5j * (lrvis - rlvis)
            bl_obs['vvis'] = 0.5  * (rrvis - llvis)
            
            
            rrsigma = np.sqrt(bl_obs['sigma']**2 + bl_obs['vsigma']**2) * rrscale
            llsigma = np.sqrt(bl_obs['sigma']**2 + bl_obs['vsigma']**2) * llscale
            rlsigma = np.sqrt(bl_obs['qsigma']**2 + bl_obs['usigma']**2) * rlscale
            lrsigma = np.sqrt(bl_obs['qsigma']**2 + bl_obs['usigma']**2) * lrscale
            
            bl_obs['sigma'] =  0.5 * np.sqrt( rrsigma**2 + llsigma**2 )
            bl_obs['qsigma'] = 0.5 * np.sqrt( rlsigma**2 + lrsigma**2 )
            bl_obs['usigma'] = 0.5 * np.sqrt( lrsigma**2 + rlsigma**2 ) 
            bl_obs['vsigma'] = 0.5 * np.sqrt( rrsigma**2 + llsigma**2 ) 
        
            if len(datatable):
                datatable = np.hstack((datatable,bl_obs))
            else:
                datatable = bl_obs
            
        calobs = eh.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, np.array(datatable), obs.tarr, source=obs.source, mjd=obs.mjd)
        
        return calobs

   
def load_caltable(obs, datapath, channel):

    datatables = {}
    
    for s in range(0, len(obs.tarr)):
    
        site = obs.tarr[s]['site']
        filename = datapath + obs.source + '_' + site + '_' + str(channel) + '.txt'

        data = np.loadtxt(filename, dtype=bytes).astype(str)   

        datatable = []
        for row in data:
            
            time = (float(row[0]) - obs.mjd) * 24.0 # time is given in mjd
            
            rscale = float(row[1]) # r
            lscale = float(row[2]) # l
                
            datatable.append(np.array((time, rscale, lscale), dtype=DTCAL))
        
        datatables[site] = np.array(datatable)

    caltable = Caltable(obs.ra, obs.dec, obs.rf, obs.bw, datatables, obs.tarr, source=obs.source, mjd=obs.mjd,
        timetype=obs.timetype)
        
    return caltable
    

