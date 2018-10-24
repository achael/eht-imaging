# Note: this is an example sequence of commands to run in ipython
# The matplotlib windows may not open/close properly if you run this directly as a script

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

# Observe the image
tint_sec = 5
tadv_sec = 600
tstart_hr = 0
tstop_hr = 24
bw_hz = 400e9
obs = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz,
                 sgrscat=False, ampcal=True, phasecal=True)
# Resolution
beamparams = obs.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
res = obs.res() # nominal array resolution, 1/longest baseline
print("Clean beam parameters: " , beamparams)
print("Nominal Resolution: " ,res)

# Generate an image prior
npix = 128
fov = 1*im.fovx()
zbl = im.total_flux() # total flux
prior_fwhm = 200*eh.RADPERUAS # Gaussian size in microarcssec
emptyprior = eh.image.make_square(obs, npix, fov)
flatprior = emptyprior.add_flat(zbl)
gaussprior = emptyprior.add_gauss(zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))

# Image total flux with bispectrum
flux = zbl
imgr  = eh.imager.Imager(obs, gaussprior, gaussprior, flux,
                          data_term={'bs':100}, show_updates=False,
                          reg_term={'simple':1,'flux':100,'cm':50},
                          maxit=200, ttype='nfft')
imgr.make_image_I()

# Blur the image with a circular beam and image again to help convergance
out = imgr.out_last()
imgr.init_next = out.blur_circ(res)
imgr.prior_next = imgr.init_next
imgr.dat_term_next = {'amp':50, 'cphase':100}
imgr.reg_term_next = {'tv2':100, 'flux':1,'cm':1}
imgr.make_image_I()

out=imgr.out_last().threshold(0.01)

# Image polarization with the polarimetric ratio
imgr.init_next = out.blur_circ(0.25*res)
imgr.prior_next = imgr.init_next
imgr.transform_next = 'mcv'
imgr.dat_term_next = {'m':1}
imgr.reg_term_next = {'hw':1}
imgr.make_image_P()

# blur and image again with the polarimetric ratio
out=imgr.out_last()
imgr.init_next = out.blur_circ(0,.5*res)
imgr.prior_next = imgr.init_next
imgr.transform_next = 'mcv'
imgr.dat_term_next = {'m':5}
imgr.reg_term_next = {'hw':1,'ptv':1.e2}
imgr.make_image_P()

out = imgr.out_last()
out.display(plotp=True)
im.display(plotp=True)
