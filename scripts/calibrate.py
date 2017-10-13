#!/usr/bin/env python

import sys
import numpy as np
import ehtim as eh
import matplotlib.pyplot as plt

# Function definitions
def pick(obs, req_sites):
   tlists   = obs.tlist()
   mask     = [req_sites.issubset(set(tlist['t1']).union(tlist['t2']))
               for tlist in tlists]
   out      = obs.copy()
   out.data = np.concatenate(tlists[mask])
   return out

def multical(obs, sites, amp0=4.0, n=3):
    sites  = set(sites) # make sure that sites is a set
    common = {'sites':sites,
              'zbl_uvdist_max':10000000.0,
              'show_solution':False,
              'pad_amp':0.0,
              'gain_tol':10.0,
              'processes':64,
              'caltable':True}

    for i in range(n):
        # Self calibrate the amplitudes
        caltab = eh.self_cal.network_cal(pick(obs, sites), amp0, method='amp',
                                         **common)
        obs = caltab.applycal(obs, interp='nearest', extrapolate=True)

        # Self calibrate the phases
        caltab = eh.self_cal.network_cal(pick(obs, sites), amp0, method='phase',
                                         **common)
        obs = caltab.applycal(obs, interp='nearest', extrapolate=True)

    return obs

# Check arguments
if len(sys.argv) <= 2:
    print("Usage: calibrate <uvfits file> <SEFD directory>")
    exit(0)

# Load uvfits file
obs = eh.obsdata.load_uvfits(sys.argv[1], force_singlepol='R')

# A-priori calibrate by applying the caltable
caltab  = eh.caltable.load_caltable(obs, sys.argv[2])
obs_cal = caltab.applycal(obs, interp='nearest', extrapolate=True)

# Compute averages
obs_cal_avg = obs_cal.avg_coherent(10.0)
#obs_cal_avg.plotall('u','v',conj=True)
#obs_cal_avg.plot_cphase('AA','AP','LM')
#plt.show()

# Speed up testing
print("GET RID OF THIS!!!!!!!!!!!!!")
obs_cal_avg.data = np.concatenate(obs_cal_avg.tlist()[::10])

# First get the ALMA and APEX calibration right -- allow huge gain_tol
sites = {'AA','AP'}
obs_cal_avg = multical(obs_cal_avg, sites)

# Next get the SMA and JCMT calibration right -- allow modest gain_tol
sites = {'SM','JC'}
obs_cal_avg = multical(obs_cal_avg, sites)

# Recalibrate all redundant stations
sites = {'AA','AP','SM','JC'}
obs_cal_avg = multical(obs_cal_avg, sites)

# Make final plots
obs_cal_avg.plot_bl('AA','AP','amp')
obs_cal_avg.plot_bl('SM','JC','amp')
obs_cal_avg.plotall('uvdist', 'amp')

obs_cal_avg.plot_bl('AA','AP','phase')
obs_cal_avg.plot_bl('SM','JC','phase')
obs_cal_avg.plotall('uvdist', 'phase')

obs_cal_avg.plot_bl('AA','SM','amp')
obs_cal_avg.plot_bl('AA','LM','amp')

obs_cal_avg.plot_bl('AA','SM','phase')
obs_cal_avg.plot_bl('AA','LM','phase')

plt.show()
