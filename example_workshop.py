# Import numpy
import numpy as np

# Import vlbi_imaging_utils - contains classes for images, arrays, and observations as well as methods and functions to create display and manipulate data
import vlbi_imaging_utils as vb
import vlbi_plots as vbp

# Import maxen library - contains functions to produce images from data using different data products and regularizing functions
import maxen as mx

##########################################################
#SgrA Image - complex visibilities
##########################################################
obs = vb.load_obs_uvfits('sgraimage_uvdata/data/sgraimage.uvfits')

# Check some diagnostic plots
obs.plotall('u','v', conj=True) # uv coverage
obs.plotall('uvdist','amp') # amplitude with baseline distance'
obs.plotall('uvdist','phase') # phase with baseline distance'
obs.plot_bl('SMA','ALMA', 'phase') # visibility phase on a baseline over time
obs.plot_cphase('LMT', 'SPT', 'ALMA') # closure phase 1-2-3 on a over time

# Check out the dirty image, dirty beam, and clean beam
npix = 128
fov = 200*vb.RADPERUAS # slightly enlarge the field of view
dim = obs.dirtyimage(npix, fov)
dbeam = obs.dirtybeam(npix, fov)
cbeam = obs.cleanbeam(npix,fov)
dim.display()
dbeam.display()
cbeam.display()

# What is the array resolution?
beamparams = obs.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
res = obs.res() # nominal array resolution, 1/longest baseline
print beamparams 
print res/vb.RADPERUAS

# Generate an image prior
npix = 128
fov =  200*vb.RADPERUAS
zbl = 2.5
prior_fwhm = 100*vb.RADPERUAS # Prior Gaussian size
emptyprior = vb.make_square(obs, npix, fov) # Generate an empty prior
flatprior = vb.add_flat(emptyprior, zbl) # Generate a flat prior

gaussparams = (prior_fwhm, prior_fwhm, 0.0)
gaussprior = vb.add_gauss(emptyprior, zbl, gaussparams) # Generate a Gaussian prior

# Look at the gaussian prior
gaussprior.display()

# Image total flux with visibility amplitudes and phases
out = mx.maxen(obs, gaussprior, gaussprior, alpha=50, maxit=100, entropy="gs")

# blur with 1/2 telescope beam 
outblur = vb.blur_gauss(out, beamparams, 0.5)

# re-start imaging using TV prior for smoothness
out=outblur
out = mx.maxen(obs, out, out, alpha=10, maxit=150, entropy="gs")

# compare amps and phases
vbp.plotall_obs_im_compare(obs, out, "uvdist", "amp")
vbp.plotall_obs_im_compare(obs, out, "uvdist", "phase")

# display final images
outblur = vb.blur_gauss(out, beamparams, 0.5)
out.display()
outblur.display()

# save fits file of final blurred and unblurred image
out.save_fits('./sgraim.fits')
outblur.save_fits('./sgraim_blur.fits')

##########################################################
#M87 Image - bispectrum
##########################################################
obs = vb.load_obs_uvfits('m87image_uvdata/data/m87image.uvfits')

obs.plotall('u','v', conj=True) # uv coverage
obs.plotall('uvdist','amp') # amplitude with baseline distance'
obs.plotall('uvdist','phase') # phase with baseline distance'
obs.plot_bl('SMA','ALMA','phase') # visibility phase on a baseline over time
obs.plot_cphase('SMA', 'SMT', 'ALMA') # closure phase 1-2-3 on a over time
obs.plot_camp('ALMA','LMT','SMA','SPT') # closure amplitude (1-2)(3-4)/(1-4)(2-3) over time

# Check out the dirty image, dirty beam, and clean beam
npix = 128
fov = 150*vb.RADPERUAS # slightly enlarge the field of view
dim = obs.dirtyimage(npix, fov)
dbeam = obs.dirtybeam(npix, fov)
cbeam = obs.cleanbeam(npix,fov)
dim.display()
dbeam.display()
cbeam.display()

# What is the array resolution?
beamparams = obs.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
res = obs.res() # nominal array resolution, 1/longest baseline
print beamparams 
print res/vb.RADPERUAS

# Generate an image prior
npix = 128
fov =  150*vb.RADPERUAS
zbl = 1.0 
emptyprior = vb.make_square(obs, npix, fov) # Generate an empty prior
flatprior = vb.add_flat(emptyprior, zbl) # Generate a flat prior

gaussparams=(100*vb.RADPERUAS, 100*vb.RADPERUAS, 0.0)
gaussprior = vb.add_gauss(emptyprior, zbl, gaussparams) # Generate a Gaussian prior

# Image with amp and cphase
# first run imaging with gaussian prior
out = mx.maxen_amp_cphase(obs, gaussprior, gaussprior, zbl, maxit=100, alpha_visamp=100, alpha_clphase=50, entropy="gs")

# blur with 1/2 telescope beam 
outblur = vb.blur_gauss(out, beamparams, 0.5)

# re-start imaging using TV prior for smoothness x2
out=outblur
out = mx.maxen_amp_cphase(obs, out, out, zbl, maxit=150, alpha_visamp=50, alpha_clphase=25, entropy="tv")
outblur = vb.blur_gauss(out, beamparams, 0.5)
out=outblur
out = mx.maxen_amp_cphase(obs, out, out, zbl, maxit=150, alpha_visamp=50, alpha_clphase=25, entropy="gs")

# Compare the vis amps and some closure phases
vbp.plotall_obs_im_compare(obs, out, "uvdist", "amp")
vbp.plot_cphase_obs_im_compare(obs, out, "ALMA","SMA","LMT")

# display final images
outblur = vb.blur_gauss(out, beamparams, 0.25)
out.display()
outblur.display()

# save fits file of final blurred and unblurred image
out.save_fits('m87im.fits')
outblur.save_fits('m87im_blur.fits')
