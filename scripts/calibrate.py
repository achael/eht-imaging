import sys

import numpy as np
import ehtim as eh
import ehtim.plotting as pl

import datetime

import sys
import os

# Ugh, I can't figure out the correspondence here
# 3597 = day 094-095 = Track D
# 3598 = day 096     = Track B
# 3599 = day 097     = Track C
# 3600 = day 099-100 = Track A
# 3601 = day 100-101 = Track E

src = 'OJ287'
pol = 'R'
t_avg = 10. # Averaging time (seconds)
expt = 3600 #expt and track should be a single code
band = 'hi' # 'lo' or 'hi'
processes = 4

if src == 'OJ287':
    zbl = 4.0 # The zero-baseline flux density (Jy); depends on the source
elif src == '3C279':
    zbl = 8.0

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

obs = eh.obsdata.load_uvfits('../../EHT_Imaging_WG_Team1_LargeFiles/er1-hops-' + band + '/6.uvfits/hops_' + str(expt) + '_' + src + '_' + band + '_closed.uvfits', force_singlepol=pol)

#obs.plotall('u','v',conj=True)
#obs.plot_cphase('AA','AP','LM')

caltable = eh.caltable.load_caltable(obs, '../SEFD_tables/SEFDSv2/SEFD_' + band.upper() + '/' + track + '/')
obs_cal = caltable.applycal(obs, interp='nearest',  extrapolate = True)

obs_cal_avg = obs_cal.avg_coherent(t_avg)

#Just for testing
#obs_cal_avg.data = np.concatenate(obs_cal_avg.tlist()[::20])

# This function creates a new observation object that only includes the scans with all of the specified sites
def obs_sites_select(sites):
    sites = set(sites)
    obs_dense = obs_cal_avg.copy()
    all_scans = obs_dense.tlist()
    all_scans_sites = [set(np.concatenate([scan['t1'],scan['t2']])) for scan in all_scans]
    dense_mask = [sites.issubset(scan_sites) for scan_sites in all_scans_sites]
    obs_dense.data = np.concatenate( all_scans[dense_mask] )
    return obs_dense

# First get the ALMA and APEX calibration right -- allow huge gain_tol because ALMA has a large error in the a priori calibration. This only requires that ALMA and APEX are present and at least one other site
sites = ['AA','AP']
for repeat in range(5):
    # Self calibrate the amplitudes
    caltable = eh.self_cal.network_cal(obs_sites_select(sites), zbl, sites=sites, zbl_uvdist_max=10000000.0, method='amp', show_solution=False, pad_amp=0.0, gain_tol=10.0, processes=processes, caltable=True)
    obs_cal_avg = caltable.applycal(obs_cal_avg, interp='nearest', extrapolate = True)

    # Self calibrate the phases
    caltable = eh.self_cal.network_cal(obs_sites_select(sites), zbl, sites=sites, zbl_uvdist_max=10000000.0, method='phase', show_solution=False, pad_amp=0.0, gain_tol=10.0, processes=processes, caltable=True)
    obs_cal_avg = caltable.applycal(obs_cal_avg, interp='nearest', extrapolate = True)

# Next get the SMA and JCMT calibration right -- allow modest gain_tol. This only requires that SMA and JCMT are present plus at least one other size
sites = ['SM','JC']
for repeat in range(5):
    # Self calibrate the amplitudes
    caltable = eh.self_cal.network_cal(obs_sites_select(sites), zbl, sites=sites, zbl_uvdist_max=10000000.0, method='amp', show_solution=False, pad_amp=0.0, gain_tol=0.3, processes=processes, caltable=True)
    obs_cal_avg = caltable.applycal(obs_cal_avg, interp='nearest', extrapolate = True)

    # Self calibrate the phases
    caltable = eh.self_cal.network_cal(obs_sites_select(sites), zbl, sites=sites, zbl_uvdist_max=10000000.0, method='phase', show_solution=False, pad_amp=0.0, gain_tol=0.3, processes=processes, caltable=True)
    obs_cal_avg = caltable.applycal(obs_cal_avg, interp='nearest', extrapolate = True)

# Finally, tighten up the cases with all four of the sites; use small gain tol
sites = ['AA','AP','SM','JC']
for repeat in range(2):
    # Self calibrate the amplitudes
    caltable = eh.self_cal.network_cal(obs_sites_select(sites), 8.0, sites=sites, zbl_uvdist_max=10000000.0, method='amp', show_solution=False, pad_amp=0.0, gain_tol=0.1, processes=processes, caltable=True)
    obs_cal_avg = caltable.applycal(obs_cal_avg, interp='nearest', extrapolate = True)

    # Self calibrate the phases
    caltable = eh.self_cal.network_cal(obs_sites_select(sites), 8.0, sites=sites, zbl_uvdist_max=10000000.0, method='phase', show_solution=False, pad_amp=0.0, gain_tol=0.1, processes=processes, caltable=True)
    obs_cal_avg = caltable.applycal(obs_cal_avg, interp='nearest', extrapolate = True)

# Export the Results
obs_cal_avg.save_uvfits('./Network_Cal/hops_' + str(expt) + '_' + src + '_' + band + '_' + pol + pol + '_' + str(int(t_avg)) + 's-avg_' + 'zbl=' + str(zbl) + '_network.uvfits')
