# Note: must import ehtim outside the ehtim directory
# either in parent eht-imaging directory or after installing with setuptools

import numpy as np
import ehtim as eh
import ehtim.scattering as so
import ehtim.plotting as pl

# Load the image and the array
im = eh.image.load_txt('../models/avery_sgra_eofn.txt')
eht = eh.array.load_txt('../arrays/EHT2017.txt')

# If the image has an even number of pixels, make it odd
#if im.xdim%2 == 0:
#	newim = im.imvec.reshape(im.ydim, im.xdim)
#	newim = newim[:-1,:-1]
#	im = eh.image.Image(newim, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd)


# Create a ScatteringModel; this defaults to parameters appropriate for Sgr A*
sm = so.ScatteringModel()

# Scatter the image; display the result
im_scatt = sm.Scatter(im,DisplayImage=True)

# Compare visibilities of the unscattered, scattered, and deblurred images
obs_unscatt = im.observe(eht, tint=300., tadv = 300, tstart = 22., tstop = 28., bw = 16e9, add_th_noise=False, sgrscat=False, ampcal=True, phasecal=True, timetype = 'GMST')
obs_scatt = im_scatt.observe(eht, tint=300., tadv = 300, tstart = 22., tstop = 28., bw = 16e9, add_th_noise=False, sgrscat=False, ampcal=True, phasecal=True, timetype = 'GMST')
obs_scatt_deblur = sm.Deblur_obs(obs_scatt)
pl.comp_plots.plotall_obs_compare([obs_unscatt,obs_scatt,obs_scatt_deblur],'uvdist','amp')

# Now approximate the ensemble-average image by averaging 100 realizations of the scattering
im_scatt_List = [sm.Scatter(im,so.MakeEpsilonScreen(im.xdim,im.ydim,rngseed=j),DisplayImage=False) for j in range(100)]
im_scatt_ea = im.copy()
im_scatt_ea.imvec = np.mean([i.imvec for i in im_scatt_List],axis=0)
im_scatt_ea.display()

# Plot a time series of the total flux density of the scattered image over time. One point per hour for a week.
movie_frames = sm.Scatter_Movie(im, Vx_km_per_s=50.0, Vy_km_per_s=0.0, framedur_sec=3600.0, N_frames = 24*7, Return_Image_List=True)
flux_timeseries = [ np.sum(i.imvec) for i in movie_frames ]
import matplotlib.pyplot as plt
plt.close()
plt.clf
plt.plot(flux_timeseries)
plt.axhline(y=np.sum(im.imvec),c="red",linewidth=1,zorder=0)
plt.xlabel('Hour')
plt.ylabel('Total Flux Density (Jy)')
plt.show()

# Export a movie of the results
import ehtim.imaging.dynamical_imaging as di
di.export_movie(movie_frames,'./example_scattered.mp4')
