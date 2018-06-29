# This examples shows an image reconstruction using stochastic optics scattering mitigation at 86 GHz.
# Note: must import ehtim outside the ehtim directory
# either in parent eht-imaging directory or after installing with setuptools

import numpy as np
import ehtim as eh
import ehtim.scattering as so

# Create a sample unscattered image -- a ring
npix = 200 #Linear pixel dimension
fov = 500*eh.RADPERUAS #Field of view
total_flux = 3.4 # Jy
im = eh.image.Image(np.zeros((npix,npix)), fov/npix, eh.RA_DEFAULT, eh.DEC_DEFAULT, rf=86e9)
im = im.add_crescent(total_flux, 105*eh.RADPERUAS, 95*eh.RADPERUAS, 0, 0)
im = im.blur_circ(10*eh.RADPERUAS)
im.display()

# Scatter the image
ep = so.MakeEpsilonScreen(im.xdim,im.ydim,rngseed=34)
sm = so.ScatteringModel()
scatt = sm.Scatter(im,ep,DisplayImage=True)

# Observe the average image
eht = eh.array.load_txt('../arrays/VLBA_86GHz.txt') # Load the observing array
#observing parameters:
tint_sec = 60
tadv_sec = 600
tstart_hr = 0
tstop_hr = 24
bw_hz = 0.5e9
#create the observation
obs = scatt.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, sgrscat=False, ampcal=True, phasecal=True)

# Generate an image prior
npix = 55 #This must be odd
prior_fwhm = 300*eh.RADPERUAS # Gaussian size in microarcssec
gaussprior = eh.image.Image(np.zeros((npix,npix)), fov/npix, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd)
gaussprior = gaussprior.add_gauss(total_flux,(prior_fwhm, prior_fwhm, 0, 0, 0))

# Now try imaging
# First image with no scattering mitigation
imgr = eh.imager.Imager(obs, gaussprior, prior_im=gaussprior, maxit=200, flux=total_flux, clipfloor=-1.)
imgr.make_image_I()
imgr.out_last().display()

# Now try deblurring before imaging
imgr_deblur = eh.imager.Imager(sm.Deblur_obs(obs), gaussprior, prior_im=gaussprior, maxit=200, flux=total_flux, clipfloor=-1.)
imgr_deblur.make_image_I()
imgr_deblur.out_last().display()

# Now image using stochastic optics
imgr_so = eh.imager.Imager(obs, gaussprior, prior_im=gaussprior, maxit=200, flux=total_flux, clipfloor=-1.)
imgr_so.make_image_I_stochastic_optics()
# Now look at the unscattered image, the scattered image, and replicate the scattering using the solved screen
imgr_so.out_last().display()
imgr_so.out_scattered_last().display()
imgr_so.scattering_model.Scatter(imgr_so.out_last(), Epsilon_Screen=so.MakeEpsilonScreenFromList(imgr_so.out_epsilon_last(), npix), ea_ker = imgr_so._ea_ker, sqrtQ=imgr_so._sqrtQ, Linearized_Approximation=True, DisplayImage=True)

#Note that only the scattered image will fit the measured visibilities!
eh.comp_plots.plotall_obs_im_compare(obs, imgr_so.out_last(), 'uvdist', 'amp')
eh.comp_plots.plotall_obs_im_compare(obs, imgr_so.out_scattered_last(), 'uvdist', 'amp')

# Decrease the scattering regularization slightly and re-image (desired max |Epsilon| is ~2.5)
imgr_so.alpha_phi_next /= 2.0
imgr_so.init_next = imgr_so.out_last().blur_circ(obs.res())
imgr_so.epsilon_list_next = imgr_so.out_epsilon_last()
imgr_so.make_image_I_stochastic_optics()
imgr_so.out_last().display()
