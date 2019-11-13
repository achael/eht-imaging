# Note: this is an example sequence of commands to run in ipython
# The matplotlib windows may not open/close properly if you run this directly as a script

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import ehtim as eh

import ehtim.modeling.modeling_utils as mu

mod = eh.model.Model()
mod.add_ring(1., 50.*eh.RADPERUAS)

# Create an observation
obs = eh.obsdata.load_uvfits('/home/michael/Dropbox/ER5_polarization_calibration/ER5/postproc-hops-lo/3.+netcal/3601/hops_3601_M87+netcal.uvfits')
obs.add_scans()
obs = obs.avg_coherent(0.,scan_avg=True)
obs = mod.observe_same(obs,ampcal=True,phasecal=True)

# Fit a model
mod_init = eh.model.Model()
mod_init.add_ring(1.1, 35.*eh.RADPERUAS)
mod_fit = eh.modeler_func(obs, mod_init, None, d1='vis',stop=1e-50)

# Fit a multi-component model
mod_init = eh.model.Model()
mod_init.add_ring(1.1, 35.*eh.RADPERUAS)
mod_init.add_ring(0.9, 45.*eh.RADPERUAS)
mod_fit2 = eh.modeler_func(obs, mod_init, None, d1='vis',stop=1e-50)
