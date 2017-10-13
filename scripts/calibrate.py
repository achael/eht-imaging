#!/usr/bin/env python

import sys
import numpy as np
import ehtim as eh
import matplotlib.pyplot as plt

# File locations
hops_uvfits_location = '../../EHT_Imaging_WG_Team1_LargeFiles'
caltable_location    = '../SEFD_tables/SEFDSv2'

# Function definitions
def pick(obs, req_sites):
   tlists   = obs.tlist()
   mask     = [req_sites.issubset(set(tlist['t1']).union(tlist['t2']))
               for tlist in tlists]
   out      = obs.copy()
   out.data = np.concatenate(tlists[mask])
   return out

def multical(obs, sites, amp0=8.0, n=3, gain_tol = 0.1):
    sites  = set(sites) # make sure that sites is a set
    common = {'sites':sites,
              'zbl_uvdist_max':10000000.0,
              'show_solution':False,
              'pad_amp':0.0,
              'gain_tol':gain_tol,
              'processes':0,
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
if len(sys.argv) <= 6:
    print("Usage: calibrate <src> <pol> <t_avg_sec> <expt> <band> [<processes> <pruning_factor>]")
    exit(0)

src = sys.argv[1]
pol = sys.argv[2]
t_avg = int(sys.argv[3]) # Averaging time (seconds)
expt = int(sys.argv[4]) #expt and track should be a single code
band = sys.argv[5] # 'lo' or 'hi'
if len(sys.argv) > 6:
    processes = int(sys.argv[6])
if len(sys.argv) > 7:
    pruning_factor = int(sys.argv[7])
else:
    pruning_factor = 1

# Determine the source zero-baseline flux density in Jy
if src == 'OJ287':
    zbl = 4.0 # The zero-baseline flux density (Jy); depends on the source
elif src == '3C279':
    zbl = 8.0

# Associate experiment numbers with track ids
if expt == 3597:
    track = 'D'
elif expt == 3598: 
    track = 'B'
elif expt == 3599: 
    track = 'C'
elif expt == 3600: 
    track = 'A'
elif expt == 3601: 
    track = 'E'

# Load uvfits file
obs = eh.obsdata.load_uvfits(hops_uvfits_location + '/er1-hops-' + band + '/6.uvfits/hops_' + str(expt) + '_' + src + '_' + band + '_closed.uvfits', force_singlepol=pol)

# A-priori calibrate by applying the caltable
caltab  = eh.caltable.load_caltable(obs, caltable_location + '/SEFD_' + band.upper() + '/' + track + '/')
obs_cal = caltab.applycal(obs, interp='nearest', extrapolate=True)

# Compute averages
obs_cal_avg = obs_cal.avg_coherent(t_avg)

# Speed up testing
if pruning_factor > 1:
    obs_cal_avg.data = np.concatenate(obs_cal_avg.tlist()[::pruning_factor])

# First get the ALMA and APEX calibration right -- allow huge gain_tol
sites = {'AA','AP'}
obs_cal_avg = multical(obs_cal_avg, sites, n=5, gain_tol = 10.0)

# Next get the SMA and JCMT calibration right -- allow modest gain_tol
sites = {'SM','JC'}
obs_cal_avg = multical(obs_cal_avg, sites, n=3, gain_tol = 0.3)

# Recalibrate all redundant stations
sites = {'AA','AP','SM','JC'}
obs_cal_avg = multical(obs_cal_avg, sites, n=2, gain_tol = 0.1)

# Make final plots
obs_cal_avg.plot_bl('AA','AP','amp')
obs_cal_avg.plot_bl('SM','JC','amp')
obs_cal_avg.plotall('uvdist', 'amp')
plt.show()
obs_cal_avg.save_uvfits('./Network_Cal/hops_' + str(expt) + '_' + src + '_' + band + '_' + pol + pol + '_' + str(int(t_avg)) + 's-avg_' + 'zbl=' + str(zbl) + '_network.uvfits')
