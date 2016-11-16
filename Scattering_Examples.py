import vlbi_imaging_utils as vb
import maxen as mx
import numpy as np
import stochastic_optics as so
import vlbi_plots as vp 

#observing parameters
tint_sec = 1e5 
tadv_sec = 300
tstart_hr = 0.5
tstop_hr = 4.2
bw_hz = 0.5e9

# Load the image and the array
im = vb.load_im_txt('./models/avery_sgra_eofn.txt') #for a text file like the one attached
eht = vb.load_array('./arrays/EHT2017.txt') #see the attached array text file

# If the image has an odd number of pixels, make it even
if im.xdim%2 == 0:
	newim = im.imvec.reshape(im.ydim, im.xdim)
	newim = newim[:-1,:-1]
	im = vb.Image(newim, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd)

# Generate an image prior
obs = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, sgrscat=False, ampcal=True, phasecal=True)
npix = im.psize*(im.xdim-1.0)/im.xdim
npix = 101 #This must be odd
fov = 1.0*im.xdim * im.psize 
zbl = np.sum(im.imvec) # total flux
emptyprior = vb.make_square(obs, npix, fov)

#Here is the 2-Gaussian fit from Rusen's paper
Rusen_2gauss_image = vb.add_gauss(emptyprior, 0.77, (18.*vb.RADPERUAS, 18.*vb.RADPERUAS, 0, 0, 0))
Rusen_2gauss_image = vb.add_gauss(Rusen_2gauss_image, 2.37, (58.*vb.RADPERUAS, 58.*vb.RADPERUAS, 0, 0, 0),x=29.0*vb.RADPERUAS,y=36.0*vb.RADPERUAS)

#Here is the Crescent fit from Rusen's paper (im, flux, Rp, Rn, a, b, x=0, y=0):
Rusen_Crescent_image = vb.add_crescent(emptyprior, 3.14, 49.0*vb.RADPERUAS, 12.0*vb.RADPERUAS, -1.0*vb.RADPERUAS, 25.0*vb.RADPERUAS)

#Here is the Crescent fit from Michael's paper (im, flux, Rp, Rn, a, b, x=0, y=0):
Michael_Crescent_image = vb.add_crescent(emptyprior, 3.14, 47.9881*vb.RADPERUAS, 11.3345*vb.RADPERUAS, -6.716*vb.RADPERUAS, 33.9294*vb.RADPERUAS)

#Here is the Annulus fit from Michael's paper (im, flux, Rp, Rn, a, b, x=0, y=0):
Michael_Annulus_image = vb.add_crescent(emptyprior, 3.14, 97.0/2.0*vb.RADPERUAS, 21.0/2.0*vb.RADPERUAS, 0.0, 0.0)

#Here's how to add scattering (following https://arxiv.org/abs/1610.05326)
#ep is the "epsilon screen", which is the normalized scattering screen in Fourier space. Each component is an independent, normalized, complex Gaussian random number. To generate new screen realizations, just change the seed value, rngseed. If you want to see how the scattering looks for different polarizations or frequencies, just use the same ep and everything should work out fine. ep needs to have the same FOV and number of pixels as the unscattered image.

ep = so.MakeEpsilonScreen(Michael_Crescent_image.xdim,Michael_Crescent_image.ydim,rngseed=34)

Rusen_2gauss_image_scatt = so.Scatter(Rusen_2gauss_image,ep,DisplayImage=True)
Michael_Crescent_image_scatt = so.Scatter(Michael_Crescent_image,ep,DisplayImage=True)

#try at a different frequency (note that the scattering wraps around the FOV, so pick a sensible choice!)
Michael_Crescent_image_100GHz = vb.add_crescent(emptyprior, 3.14, 47.9881*vb.RADPERUAS, 11.3345*vb.RADPERUAS, -6.716*vb.RADPERUAS, 33.9294*vb.RADPERUAS)
Michael_Crescent_image_100GHz.rf = 100e9
Michael_Crescent_image_100GHz_scatt = so.Scatter(Michael_Crescent_image_100GHz,ep,DisplayImage=True)

#Compare the visibilities of the scattered and unscattered images
obs_unscatt = Michael_Crescent_image.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, sgrscat=False, ampcal=True, phasecal=True)
obs_scatt = Michael_Crescent_image_scatt.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, sgrscat=False, ampcal=True, phasecal=True)
vp.plotall_obs_compare([obs_unscatt, obs_scatt], 'uvdist','amp', show=True)
