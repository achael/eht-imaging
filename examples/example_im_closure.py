# example of imaging directly with closure quantities stored in obsdata object


# Note: must import ehtim outside the ehtim directory
# either in parent eht-imaging directory or after installing with setuptools

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import ehtim as eh
from   ehtim.calibrating import self_cal as sc
#from  ehtim.plotting import self_cal as sc

# Load the image and the array
im = eh.image.load_txt('../models/avery_sgra_eofn.txt')
eht = eh.array.load_txt('../arrays/EHT2017.txt')

# Look at the image
im.display()

# Observe the image
# tint_sec is the integration time in seconds, and tadv_sec is the advance time between scans
# tstart_hr is the GMST time of the start of the observation and tstop_hr is the GMST time of the end
# bw_hz is the  bandwidth in Hz
# sgrscat=True blurs the visibilities with the Sgr A* scattering kernel for the appropriate image frequency
# ampcal and phasecal determine if gain variations and phase errors are included
tint_sec = 5
tadv_sec = 10
tstart_hr = 0
tstop_hr = 24
bw_hz = 4e9
obs = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz,
                 sgrscat=False, ampcal=True, phasecal=False,ttype='nfft')

# Resolution
beamparams = obs.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
res = obs.res() # nominal array resolution, 1/longest baseline
print("Clean beam parameters: " , beamparams)
print("Nominal Resolution: " ,res)

# Export the visibility data to uvfits/text
#obs.save_txt('obs.txt') # exports a text file with the visibilities
#obs.save_uvfits('obs.uvp') # exports a UVFITS file modeled on template.UVP

# Generate an image prior
npix = 256
fov = 1*im.fovx()
zbl = im.total_flux() # total flux
prior_fwhm = 200*eh.RADPERUAS # Gaussian size in microarcssec
emptyprior = eh.image.make_square(obs, npix, fov)
flatprior = emptyprior.add_flat(zbl)
gaussprior = emptyprior.add_gauss(zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))

# Average the closure quantities and add them to the obsdata object

avg_time = 600
obs.add_bispec(avg_time=avg_time)
obs.add_amp(avg_time=avg_time)
obs.add_cphase(avg_time=avg_time)
obs.add_camp(avg_time=avg_time)
obs.add_logcamp(avg_time=avg_time)

# Image directly with these averaged data
flux = zbl
out  = eh.imager_func(obs, gaussprior, gaussprior, flux,
                      d1='bs', s1='simple',
                      alpha_s1=1, alpha_d1=100,
                      alpha_flux=100, alpha_cm=50,
                      maxit=100, ttype='nfft')

# Blur the image with a circular beam and image again to help convergance
out = out.blur_circ(res)
out = eh.imager_func(obs, out, out, flux,
                d1='amp', d2='cphase', s1='tv',
                alpha_s1=1, alpha_d1=50,alpha_d2=100,
                alpha_flux=100, alpha_cm=50,
                maxit=100,ttype='nfft')

out = out.blur_circ(res/2.0)
out = eh.imager_func(obs, out, out, flux,
                d1='amp', d2='cphase', s1='tv',
                alpha_s1=1, alpha_d1=10,alpha_d2=20,
                alpha_flux=100, alpha_cm=50,
                maxit=100,ttype='nfft')

out.display()
