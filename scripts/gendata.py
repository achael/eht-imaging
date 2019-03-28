import ehtim as eh
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs


# set as true if you want to use a model image, false will load the specified image
makeModel = True 
# path to a sample image
sampleimage = '../models/rowan_m87.txt'

# path to array file for loading SEFDs and site locations
array = '../arrays/EHT2017_m87.txt'

# parameters for model image
npix = 128
fov = 200*eh.RADPERUAS
source = 'M87'
ra = 19.414182210498385
dec = -29.24170032236311
zbl = 0.8
rf = 230000000000.0
mjd = 57854

ring_radius = 22*eh.RADPERUAS # the radius of the ring
ring_width = 10*eh.RADPERUAS # the width of the ring
nonun_frac = 0.5 # defines how much brighter the brighter location is on the ring
theta_nonun_rad = 270 # defines the angle of the brightest location
fracpol = 0.4 # fractional polarization on model image
corr = 5 * eh.RADPERUAS # the coherance length of the polarization

# parameters for simulated data
add_th_noise = True # if there are no sefds in obs_orig it will use the sigma for each data point
phasecal = False # if False then add random phases to simulate atmosphere
ampcal = False # if False then add random gain errors 
stabilize_scan_phase = True # if true then add a single phase error for each scan to act similar to adhoc phasing
stabilize_scan_amp = True # if true then add a single gain error at each scan
jones = True # apply jones matrix for including noise in the measurements (including leakage)
inv_jones = False # no not invert the jones matrix
frcal = True # True if you do not include effects of field rotation
dcal = False # True if you do not include the effects of leakage
dterm_offset = 0.05 # a random offset of the D terms is given at each site with this standard deviation away from 1

tint_sec = 10
tadv_sec = 600
tstart_hr = 0
tstop_hr = 24
bw_hz = 4e9
seed = 2
gain_offset = {'ALMA':0.15, 'APEX':0.15, 'SMT':0.15, 'LMT':0.6, 'PV':0.15, 'SMA':0.15, 'JCMT':0.15, 'SPT':0.15}  # the standard deviation of the absolute gain 
gainp = {'ALMA':0.05, 'APEX':0.05, 'SMT':0.05, 'LMT':0.5, 'PV':0.05, 'SMA':0.05, 'JCMT':0.05, 'SPT':0.05} # the standard deviation of gain differences 


# where to save results
savefolder = '../results/'
if not path.exists(savefolder):
    makedirs(savefolder)

# Load or make the image 

if makeModel: # model image
    # make an empty image
    simim = eh.image.make_empty(npix, fov, ra, dec, rf=rf, source=source, mjd=mjd)
    # add a non-uniform ring
    simim = simim.add_ring_m1(zbl, nonun_frac, ring_radius, theta_nonun_rad * np.pi/180., ring_width, x=0, y=0)
    simim.imvec *= zbl/simim.total_flux()
    # add random polarization field with a constant fractional polarization
    simim = simim.add_random_pol(fracpol, corr, seed=0)

else: # image from file
    # load a saved image
    simim = eh.image.load_txt(sampleimage)
    simim.mjd = mjd # replace the mjd of the image
    simim.ra = ra
    simim.dec = dec
    simim.imvec = zbl * simim.imvec / np.sum(simim.imvec) 

# display the image
simim.display()
#simim.display(pol='U')
simim.display(plotp=True)

# Generate an observation 
eht = eh.array.load_txt(array)

obs = simim.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, add_th_noise=add_th_noise, ampcal=ampcal, phasecal=phasecal, 
    stabilize_scan_phase=stabilize_scan_phase, stabilize_scan_amp=stabilize_scan_amp, gain_offset=gain_offset, 
    gainp=gainp, jones=jones,inv_jones=inv_jones,dcal=dcal, frcal=frcal, dterm_offset=dterm_offset, seed=seed)
                                 

# Display plots and save and save observation 
obs.plotall('u','v', conj=True, rangey=[-1e10, 1e10], rangex=[-1e10, 1e10]) # uv coverage
obs.plotall('uvdist','amp') # amplitude with baseline distance'

site1 = obs.tarr[0]['site']
site2 = obs.tarr[1]['site']
site3 = obs.tarr[2]['site']
obs.plot_bl(site1, site2,'phase', rangey=[-180, 180]) # visibility phase on a baseline over time
obs.plot_cphase(site1, site2, site3)


obs.save_uvfits(savefolder + 'simobs.uvfits')
simim.save_fits(savefolder + 'simim.fits')
simim.display(label_type='scale', has_title=False, cbar_unit=('$\mu$-Jy', '$\mu$as$^2$'), export_pdf=savefolder + 'simim.pdf')
















