# Note: this is an example sequence of commands to run in ipython
# The matplotlib windows may not open/close properly if you run this directly as a script

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
plt.ion

import numpy as np
import ehtim as eh

# Load a sample array
eht = eh.array.load_txt('../arrays/EHT2019.txt')

### Make a simple model ###

# Start with an empty model object
mod = eh.model.Model()

# Add a ring model
mod = mod.add_ring(F0=1.5, d=40.*eh.RADPERUAS)

# View the model
mod.display()

# Add another model component
mod = mod.add_circ_gauss(1., 20.*eh.RADPERUAS, x0=-15.*eh.RADPERUAS, y0=20.*eh.RADPERUAS)

# View the model after blurring with a circular Gaussian
mod.blur_circ(20.*eh.RADPERUAS).display()

# Make an image of the model
im = mod.make_image(200.*eh.RADPERUAS, 1024)

# Observe the model
tint_sec = 5
tadv_sec = 3600
tstart_hr = 0
tstop_hr = 24
bw_hz = 1e9
obs = mod.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, ampcal=True, phasecal=False)

# Compare the observation to the model
eh.comp_plots.plotall_obs_im_compare(obs, mod, 'uvdist', 'amp')
eh.comp_plots.plotall_obs_im_compare(obs, mod, 'uvdist', 'phase')

### Next, we'll try some model fitting ###

# First, we need to define the model that we are using to fit an observation
# Some algorithms (e.g., gradient descent) also use this as the initialization 
mod_init  = eh.model.Model()
mod_init  = mod_init.add_ring(1.5, 40.*eh.RADPERUAS)
mod_init  = mod_init.add_circ_gauss(1., 20.*eh.RADPERUAS)

# Next, define the prior for each fitted parameter
# Because we don't have absolute phase information, we'll force the ring to be centered on the origin
mod_prior = mod_init.default_prior()
# Ring:
mod_prior[0]['F0'] = {'prior_type':'flat', 'min':0.0, 'max':2.0}
mod_prior[0]['d']  = {'prior_type':'flat', 'min':1.0*eh.RADPERUAS, 'max':50.0*eh.RADPERUAS}
mod_prior[0]['x0'] = {'prior_type':'fixed'}
mod_prior[0]['y0'] = {'prior_type':'fixed'}
# Gaussian
mod_prior[1]['F0'] = {'prior_type':'flat', 'min':0.0, 'max':2.0}
mod_prior[1]['FWHM'] = {'prior_type':'flat', 'min':1.0*eh.RADPERUAS, 'max':50.0*eh.RADPERUAS}
mod_prior[1]['x0'] = {'prior_type':'gauss','mean':0.0,'std':20.*eh.RADPERUAS}
mod_prior[1]['y0'] = {'prior_type':'gauss','mean':0.0,'std':20.*eh.RADPERUAS}

### Fit the model using dynamic nested sampling; this estimates the full posterior ###
# Fit using amplitudes and closure phases
mod_fit   = eh.modeler_func(obs, mod_init, mod_prior, d1='amp', d2='cphase', minimizer_func='dynesty_dynamic')

# View the fitted model (the MAP)
mod_fit['model'].display()
mod_fit['model'].blur_circ(20.*eh.RADPERUAS).display()

# View a few random samples from the posterior
eh.imaging.dynamical_imaging.plot_im_List([_.make_image(100.*eh.RADPERUAS, 128) for _ in mod_fit['posterior_models'][:10]])

# Compare the fitted model to the data
eh.comp_plots.plotall_obs_im_compare(obs,mod_fit['model'],'uvdist','amp') 

# Compare a sample of fitted models drawn from the posterior to the data
eh.comp_plots.plotall_obs_im_compare(obs,mod_fit['posterior_models'],'uvdist','amp') 

# Compare posteriors from with the true model values ('natural' uses natural units and rescalings)
from dynesty import plotting as dyplot
cfig, caxes = dyplot.cornerplot(mod_fit['res_natural'], labels=mod_fit['labels_natural'], truths=[1.5, 40, 1.0, 20, -15, 20])
cfig.set_size_inches((2.5*len(mod_fit['labels']),2.5*len(mod_fit['labels'])))
cfig.show()

# Save the MAP
mod_fit['model'].save_txt('sample_modelfit.txt')
