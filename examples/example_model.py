# Note: this is an example sequence of commands to run in ipython
# The matplotlib windows may not open/close properly if you run this directly as a script

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import ehtim as eh

# Load a sample array
eht = eh.array.load_txt('../arrays/EHT2019.txt')

# Make a ring model
mod = eh.model.Model()
mod = mod.add_mring(1.5, 40.*eh.RADPERUAS, beta_list=[0.1+0.2j])

# View the model
mod.display()

# View the model after blurring with a circular Gaussian
mod.blur_circ(5.*eh.RADPERUAS).display()

# Add another model component
mod = mod.add_circ_gauss(1., 20.*eh.RADPERUAS, x0=-15.*eh.RADPERUAS, y0=20.*eh.RADPERUAS)
mod.blur_circ(5.*eh.RADPERUAS).display()

# Make an image of the model
im = mod.make_image(200.*eh.RADPERUAS, 1024)

# Observe the model
tint_sec = 30
tadv_sec = 600
tstart_hr = 0
tstop_hr = 24
bw_hz = 4e9
obs = mod.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, ampcal=True, phasecal=False)

# Fit the observation using visibility amplitudes and closure phase
mod_init  = eh.model.Model()
mod_init  = mod_init.add_mring(1.5, 40.*eh.RADPERUAS, beta_list=[0.1+0.1j]).add_circ_gauss(1., 20.*eh.RADPERUAS)
mod_prior = mod_init.default_prior()
mod_prior[0]['x0'] = {'prior_type':'fixed'}
mod_prior[0]['y0'] = {'prior_type':'fixed'}
mod_prior[0]['F0'] = {'prior_type':'fixed'}
mod_prior[1]['F0'] = {'prior_type':'gauss','mean':1.0,'std':0.1}
mod_fit   = eh.modeler_func(obs, mod_init, mod_prior, d1='amp', d2='cphase')

eh.comp_plots.plotall_obs_im_compare(obs,mod_fit,'uvdist','amp') 
mod_fit.blur_circ(5.*eh.RADPERUAS).display()
