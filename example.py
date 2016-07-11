# Note: this is an example sequence of commands I might run in ipython
# The matplotlib windows may not open/close properly if you run this directly as a script

import vlbi_imaging_utils as vb
import maxen as mx
import numpy as np

# Load the image and the array
#im = vb.load_im_fits('./models/avery_sgra_eofn.fits') #for a fits image
im = vb.load_im_txt('./models/avery_sgra_eofn.txt') #for a text file
eht = vb.load_array('./arrays/EHT2017.txt') #see the attached array text file

# Look at the image
im.display(plotp=True)

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

# Or you can load an observation file directly from text or uvfits
# When loading from uvfits, you must provide an array text file with the stations
# listed in the same order as their labels in the uvfitsfile
#obs = vb.load_obs_txt('./data/testobs.txt')
#obs = vb.load_obs_uvfits('./data/testobs.UVP', './arrays/EHT2017.txt')


# These are some simple plots you can check
obs.plotall('u','v', conj=True) # uv coverage
obs.plotall('uvdist','amp') # amplitude with baseline distance'
obs.plot_bl('SMA','ALMA','phase') # visibility phase on a baseline over time
obs.plot_cphase('SMA', 'SMT', 'ALMA') # closure phase 1-2-3 on a over time
obs.plot_camp('ALMA','LMT','SMA','SPT') # closure amplitude (1-2)(3-4)/(1-4)(2-3) over time

# You can get lists of closure phases and amplitudes and save them to a file
#cphases = obs.c_phases(mode='all', count='max') # set count='min' to return a minimal set
#camps = obs.c_amplitudes(mode='all', count='max') # set count='min' to return a minimal set
#np.savetxt('./c_phases.txt',cphases)
#np.savetxt('./c_amplitudes.txt',camps)

# You can check out the dirty image, dirty beam, and clean beam
npix = 64
fov = 1.5*im.xdim * im.psize # slightly enlarge the field of view
dim = obs.dirtyimage(npix, fov)
dbeam = obs.dirtybeam(npix, fov)
cbeam = obs.cleanbeam(npix,fov)
dim.display()
dbeam.display()
cbeam.display()

# Resolution
beamparams = obs.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
res = obs.res() # nominal array resolution, 1/longest baseline
print(beamparams) 
print(res)

# You can deblur the visibilities by dividing by the scattering kernel
obs = vb.deblur(obs)

# Export the visibility data to uvfits/text
obs.save_txt('obs.txt') # exports a text file with the visibilities
obs.save_uvfits('obs.uvp') # exports a UVFITS file modeled on template.UVP

# Generate an image prior
npix = 100
fov = 1*im.xdim * im.psize
zbl = np.sum(im.imvec) # total flux
prior_fwhm = 150*vb.RADPERUAS # Gaussian size in microarcssec
emptyprior = vb.make_square(obs, npix, fov)
flatprior = vb.add_flat(emptyprior, zbl)
gaussprior = vb.add_gauss(emptyprior, zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))

# Image total flux with the bispectrum
flux = np.sum(im.imvec)
out = mx.maxen_bs(obs, gaussprior, flux, maxit=50, alpha=1e5)
 
# Blur the image with a circular beam and image again to help convergance
out = mx.blur_circ(out, res)
out = mx.maxen_bs(obs, out, flux, maxit=100, alpha=500)
out = mx.blur_circ(out, res/2)
out = mx.maxen_bs(obs, out, flux, maxit=250, alpha=100)

# Image Polarization
out = mx.maxen_m(obs, out, beta=100, maxit=250, polentropy="hw")

# Blur the final image with 1/2 the clean beam
outblur = vb.blur_gauss(out, beamparams, 0.5, 0.5)

# Save the images
outname = "test"
out.save_txt(outname + '.txt')
out.save_fits(outname + '.fits')
outblur.save_txt(outname + '_blur.txt')
outblur.save_fits(outname + '_blur.fits')


