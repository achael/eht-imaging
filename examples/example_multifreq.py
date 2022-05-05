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
im = eh.image.load_txt('../models/avery_sgra_eofn.txt')
eht = eh.array.load_txt('../arrays/EHT2017.txt')

# Add a spectral index to the image
alpha = 2.5
im.rf = 230.e9
im = im.add_const_mf(alpha)

im230 = im.get_image_mf(230.e9)
im345 = im.get_image_mf(345.e9)

# Observe the image at two different frequencies
tint_sec = 120
tadv_sec = 600
tstart_hr = 0
tstop_hr = 24
bw_hz = 2e9
obs230 = im230.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz,
                     sgrscat=False, ampcal=True, phasecal=True)
obs345 = im345.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz,
                        sgrscat=False, ampcal=True, phasecal=True)
obslist = [obs230,obs345]

#observe at 345 with no spectral index applied
im345_nospec = im230.copy()
im345_nospec.rf = 345e9
obs345_nospec = im345_nospec.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz,
                                     sgrscat=False, ampcal=True, phasecal=True)
obslist_nospec = [obs230,obs345_nospec]

# Resolution
beamparams230 = obs230.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
res230 = obs230.res() # nominal array resolution, 1/longest baseline
beamparams345 = obs345.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
res345 = obs345.res() # nominal array resolution, 1/longest baseline
print("Nominal Resolution: " ,res230,res345)

# Generate an image prior
npix = 64
fov = 1*im.fovx()
zbl = im.total_flux() # total flux
prior_fwhm = 200*eh.RADPERUAS # Gaussian size in microarcssec
emptyprior = eh.image.make_square(obs230, npix, fov)
flatprior = emptyprior.add_flat(zbl)
gaussprior = emptyprior.add_gauss(zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))


## Image both frequencies independently with complex visibilities
flux230 = zbl
imgr230  = eh.imager.Imager(obs230, gaussprior, gaussprior, flux230,
                            data_term={'vis':100},
                            reg_term={'tv2':1.e4,'l1':5},
                            maxit=200, ttype='nfft')
imgr230.make_image_I(show_updates=False)
out230 = imgr230.out_last()


flux345 = im345.total_flux()
imgr345  = eh.imager.Imager(obs345, gaussprior, gaussprior, flux345,
                            data_term={'vis':100},
                            reg_term={'tv2':1},
                            maxit=200, ttype='nfft')
imgr345.make_image_I(show_updates=False)
out345 = imgr345.out_last()

## Image both frequencies together without multifrequency imaging (no spectral index)
imgr  = eh.imager.Imager(obslist_nospec, gaussprior, gaussprior, zbl,
                            data_term={'vis':100}, 
                            reg_term={'tv2':1.e4,'l1':5},
                            maxit=200, ttype='nfft')
imgr.make_image_I(show_updates=False)
out_nospec = imgr.out_last()

# Image both frequencies together with spectral index
plt.close('all')
gaussprior = gaussprior.add_const_mf(3)
imgr  = eh.imager.Imager(obslist, gaussprior, gaussprior, zbl,
                            data_term={'vis':100},
                            reg_term={'tv2':1.e4,'l1':5},
                            maxit=100, ttype='nfft')
imgr.make_image_I(mf=True,show_updates=False)
out = imgr.out_last()

for i in range(3): # blur and reimage
    out = out.blur_circ(15*eh.RADPERUAS)
    imgr.maxit_next=500
    imgr.init_next = out
    imgr.make_image_I(mf=True,show_updates=False)
    out = imgr.out_last()

# look at results
out230_mf = out.get_image_mf(230.e9)
out345_mf = out.get_image_mf(345.e9)
out_specind = out.copy()
out_specind.imvec = out.specvec

out230_mf.display()
out345_mf.display()
out_specind.display(cfun='jet',cbar_lims=[2,3])

print(im230.total_flux(), out230_mf.total_flux())
print(im345.total_flux(), out345_mf.total_flux())
print(obs345.chisq(out345_mf), obs230.chisq(out230_mf))

