# Multifrequency imager example
# RIAF model with a constant spectral index

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import ehtim as eh
from   ehtim.calibrating import self_cal as sc
#from  ehtim.plotting import self_cal as sc

# Load the image and the array
im = eh.image.load_txt('./models/avery_sgra_eofn.txt')
eht = eh.array.load_txt('./arrays/EHT2017.txt')

# Add a spectral index to the image
alpha = 2.5
im.rf = 230.e9
im = im.add_const_mf(alpha)

im230 = im.get_image_mf(230.e9)
im345 = im.get_image_mf(345.e9)

# Observe the image at two different frequencies
tint_sec = 240
tadv_sec = 600
tstart_hr = 0
tstop_hr = 24
bw_hz = 2e9
obs230 = im230.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz,
                     sgrscat=False, ampcal=True, phasecal=True)
obs345 = im345.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz,
                        sgrscat=False, ampcal=True, phasecal=True)
obslist = [obs230,obs345]

# Resolution
beamparams230 = obs230.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
res230 = obs230.res() # nominal array resolution, 1/longest baseline
beamparams345 = obs345.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
res345 = obs345.res() # nominal array resolution, 1/longest baseline
print("Nominal Resolution: " ,res230,res345)

# Generate an image prior
npix = 32
fov = 1*im.fovx()
zbl = im.total_flux() # total flux
prior_fwhm = 200*eh.RADPERUAS # Gaussian size in microarcssec
emptyprior = eh.image.make_square(obs230, npix, fov)
flatprior = emptyprior.add_flat(zbl)
gaussprior = emptyprior.add_gauss(zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))
gaussprior = gaussprior.add_const_mf(2.5) #regularize to the right value for now

## Image both frequencies independently with complex visibilities
flux230 = zbl
imgr230  = eh.imager.Imager(obs230, gaussprior, gaussprior, flux230,
                            data_term={'vis':100}, show_updates=False,
                            reg_term={'simple':1,'flux':100,'cm':50},
                            maxit=200, ttype='nfft')
imgr230.make_image_I()
out230 = imgr230.out_last()


flux345 = im345.total_flux()
imgr345  = eh.imager.Imager(obs345, gaussprior, gaussprior, flux345,
                            data_term={'vis':100}, show_updates=False,
                            reg_term={'simple':1,'flux':100,'cm':50},
                            maxit=200, ttype='nfft')
imgr345.make_image_I()
out345 = imgr345.out_last()


# Image both frequencies together with multifrequency imaging
imgr  = eh.imager.Imager(obslist, gaussprior, gaussprior, zbl,
                            data_term={'vis':100}, show_updates=False,
                            reg_term={'simple':1,'flux':100,'cm':50,'l2_alpha':.1},
                            maxit=200, ttype='nfft')
imgr.make_image_I(mf=True)
out = imgr.out_last()

# look at results
out230_mf = out.get_image_mf(230.e9)
out345_mf = out.get_image_mf(345.e9)
out_specind = out.copy()
out_specind.imvec = out.specvec

out230_mf.display()
out345_mf.display()
out_specind.display(cfun='jet',cbar_lims=[2,3])

