#!/usr/bin/env python

import sys
import os

import numpy as np
import ehtim as eh
import matplotlib.pyplot as plt

# Function definitions
def pick(obs, req_sites):
    """
    Pick out observations with only certain sites
    """
    tlists   = obs.tlist()
    mask     = [req_sites.issubset(set(tlist['t1']).union(tlist['t2']))
                for tlist in tlists]
    out      = obs.copy()
    out.data = np.concatenate(tlists[mask])
    return out

def multical(obs, sites, n=3, amp0=8.0, gain_tol=0.1):
    """
    Apply applycal() multiple times
    """
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
        caltab = eh.self_cal.network_cal(pick(obs,sites), amp0, method='amp',
                                         **common)
        obs = caltab.applycal(obs, interp='nearest', extrapolate=True)

        # Self calibrate the phases
        caltab = eh.self_cal.network_cal(pick(obs,sites), amp0, method='phase',
                                         **common)
        obs = caltab.applycal(obs, interp='nearest', extrapolate=True)

    return obs

# Tack to experiment number mapper (Unused, just for record)
expt = {'D':3597,
        'B':3598,
        'C':3599,
        'A':3600,
        'E':3601}

# Parameters
t_avg = 10.0
pol   = "R"

# Check arguments
if len(sys.argv) < 4:
    print("Usage: calibrate <input uvfits file> <SEFD directory> <amp_zbl> "+
          "[<pruning_factor>]")
    exit(0)

input_name     = sys.argv[1]
caltab_dir     = sys.argv[2]
amp_zbl        = float(sys.argv[3])
pruning_factor = int(sys.argv[4]) if len(sys.argv) > 4 else 1

out_prefix = os.path.basename(input_name[:-13])+pol+pol
print(input_name, caltab_dir, amp_zbl, out_prefix)

# Load uvfits file
obs = eh.obsdata.load_uvfits(input_name, force_singlepol=pol)

# A-priori calibrate by applying the caltable
caltab  = eh.caltable.load_caltable(obs, caltab_dir)
obs_cal = caltab.applycal(obs, interp='nearest', extrapolate=True, force_singlepol=pol)

# Compute averages
obs_cal_avg = obs_cal.avg_coherent(t_avg)

# Speed up testing
if pruning_factor > 1:
    obs_cal_avg.data = np.concatenate(obs_cal_avg.tlist()[::pruning_factor])

# First get the ALMA and APEX calibration right -- allow huge gain_tol
sites = {'AA','AP'}
obs_cal_avg = multical(obs_cal_avg, sites, n=5, amp0=amp_zbl, gain_tol=10.0)

# Next get the SMA and JCMT calibration right -- allow modest gain_tol
sites = {'SM','JC'}
obs_cal_avg = multical(obs_cal_avg, sites, n=3, amp0=amp_zbl, gain_tol=0.3)

# Recalibrate all redundant stations
sites = {'AA','AP','SM','JC'}
obs_cal_avg = multical(obs_cal_avg, sites, n=2, amp0=amp_zbl, gain_tol=0.1)

# Save output
obs_cal_avg.save_uvfits(out_prefix+'+netcal.uvfits')
