#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import argparse

import numpy as np
import ehtim as eh
import matplotlib.pyplot as plt

stepname='init'

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
        caltab.save_txt(obs, datadir='amp-caltab-{}-{}'.format(stepname, i))
        obs = caltab.applycal(obs, interp='nearest', extrapolate=True)

        # Self calibrate the phases
        caltab = eh.self_cal.network_cal(pick(obs,sites), amp0, method='phase',
                                         **common)
        caltab.save_txt(obs, datadir='phase-catlab-{}-{}'.format(stepname, i))
        obs = caltab.applycal(obs, interp='nearest', extrapolate=True)

    return obs

# Tack to experiment number mapper (Unused, just for record)
expt = {'D':3597,
        'B':3598,
        'C':3599,
        'A':3600,
        'E':3601}

# Argument parsing
parser = argparse.ArgumentParser(description="Perform network calibration")
parser.add_argument('input',                                    help="input uvfits file")
parser.add_argument('caltab',                                   help="caltable directory")
parser.add_argument('-o', '--output', default=None,             help="output file")
parser.add_argument('-P', '--prune',  default=1,    type=int,   help="pruning factor")
parser.add_argument('-z', '--ampzbl', default=7.0,  type=float, help="amplitude at zero-baseline")
parser.add_argument('-t', '--tavg',   default=10.0, type=float, help="averaging time")
parser.add_argument('-p', '--pol',    default="R",              help="polarization")
args = parser.parse_args()

if args.output is None:
    args.output = os.path.basename(args.input[:-13])+args.pol+args.pol+'+netcal.uvfits'
print("Parameters:")
print("    input: ", args.input)
print("    caltab:", args.caltab)
print("    output:", args.output)
print("    prune: ", args.prune)
print("    ampzbl:", args.ampzbl)
print("    tavg:  ", args.tavg)
print("    pol:   ", args.pol)

# Load uvfits file
obs = eh.obsdata.load_uvfits(args.input, force_singlepol=args.pol)

# A-priori calibrate by applying the caltable
caltab  = eh.caltable.load_caltable(obs, args.caltab)
obs_cal = caltab.applycal(obs, interp='nearest', extrapolate=True, force_singlepol=args.pol)

# Compute averages
obs_cal_avg = obs_cal.avg_coherent(args.tavg)

# Speed up testing
if args.prune > 1:
    obs_cal_avg.data = np.concatenate(obs_cal_avg.tlist()[::args.prune])

# First get the ALMA and APEX calibration right -- allow huge gain_tol
sites = {'AA','AP'}
obs_cal_avg = multical(obs_cal_avg, sites, n=5, amp0=args.ampzbl, gain_tol=10.0)

# Next get the SMA and JCMT calibration right -- allow modest gain_tol
sites = {'SM','JC'}
obs_cal_avg = multical(obs_cal_avg, sites, n=3, amp0=args.ampzbl, gain_tol=0.3)

# Recalibrate all redundant stations
sites = {'AA','AP','SM','JC'}
obs_cal_avg = multical(obs_cal_avg, sites, n=2, amp0=args.ampzbl, gain_tol=0.1)

# Save output
obs_cal_avg.save_uvfits(out_prefix+'+netcal.uvfits')
