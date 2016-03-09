# Note: this is an example sequence of commands I might run in ipython
# The matplotlib windows won't open/close properly if you run this as a script

import vlbi_imaging_utils as vb
import maxen as mx
import numpy as np

# Load the image and the array
#im = vb.load_im_fits('avery_sgra.fits') #for a fits image like the one attached
im = vb.load_im_txt('image.txt') #for a text file like the one attached
eht = vb.load_array('EHT2017.txt') #see the attached array text file

# Look at the image
im.display()

# Observe the image
# tint_sec is the integration time in seconds, and tadv_sec is the advance time between scans
# tstart_hr is the GMST time of the start of the observation and tstop_hr is the GMST time of the end
# bw_hz is the  bandwidth in Hz
# sgrscat=True blurs the visibilities with the Sgr A* scattering kernel for the appropriate image frequency
# ampcal and phasecal determine if gain variations and phase errors are included
tint_sec = 60
tadv_sec = 600
tstart_hr = 0
tstop_hr = 24
bw_hz = 4e9
obs = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, sgrscat=True, ampcal=False, phasecal=False)

# These are some simple plots you can check
obs.plotall('u','v') # uv coverage
obs.plotall('uvdist','amp') # amplitude with baseline distance'
obs.plot_bl('SMA','ALMA','phase') # visibility phase on a baseline over time
obs.plot_cphase('SMA', 'SMT', 'ALMA') # closure phase on a triangle over time

# You can check out the dirty image and dirty beam
npix = 64
fov = 1.5*im.xdim * im.psize # slightly enlarge the field of view
dim = obs.dirtyimage(npix, fov)
dbeam = obs.dirtybeam(npix, fov)
dim.display()
dbeam.display()

# Fit for the clean beam parameters (fwhm_maj, fwhm_min, theta) in radians
beamparams = obs.fit_beam()
res = 1 / np.max(obs.unpack('uvdist')['uvdist'])
print beamparams 
print res

# You can deblur the visibilities by dividing by the (hardcoded, frequency-dependent) scattering kernel
obs = vb.deblur(obs)

# Export the visibility data to uvfits/text
obs.save_txt('obs.txt') # exports a text file with the visibilities
obs.save_uvfits('obs.uvp') # exports a UVFITS file modeled on template.UVP

# Generate an image prior
npix = 64
fov = 1.5*im.xdim * im.psize # slightly enlarge the field of view
zbl = np.sum(im.imvec) # total flux
prior_fwhm = 100*vb.RADPERUAS # Gaussian size in microarcssec
emptyprior = mx.make_square_prior(obs, npix, fov)
flatprior = mx.add_flat(emptyprior, zbl)
gaussprior = mx.add_gauss(emptyprior, zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))

# Image total flux with the bispectrum
flux = np.sum(im.imvec)
out = mx.maxen_bs(obs, gaussprior, flux, maxit=50, alpha=50)
 
# Blur the image with a circular beam and image again to help convergance
out = mx.blur_circ(out, res/2)
out = mx.maxen_bs(obs, out, flux, maxit=250, alpha=70)
   
# Image Polarization
out = mx.maxen_m(obs, out, alpha=100, maxit=250, polentropy="hw")

# Blur the final image with 1/2 the clean beam
outblur = vb.blur_gauss(out, beamparams, 0.5, 0.5)

# Save the images
outname = "test"
out.save_txt(outname + '.txt')
out.save_fits(outname + '.fits')
outblur.save_txt(outname + '_blur.txt')
outblur.save_fits(outname + '_blur.fits')


