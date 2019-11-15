# Note: this is an example sequence of commands to run in ipython
# The matplotlib windows may not open/close properly if you run this directly as a script

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import ehtim as eh

import ehtim.modeling.modeling_utils as mu

# Define the ground-truth model
mod = eh.model.Model()
mod.add_ring(1., 50.*eh.RADPERUAS)
mod.add_ring(0.5, 30.*eh.RADPERUAS, 5.*eh.RADPERUAS, 5.*eh.RADPERUAS) #For testing gradients
mod.make_image(100.*eh.RADPERUAS, 128).blur_circ(5.*eh.RADPERUAS).display()

# Create an observation
obs = eh.obsdata.load_uvfits('/home/michael/Dropbox/ER5_polarization_calibration/ER5/postproc-hops-lo/3.+netcal/3601/hops_3601_M87+netcal.uvfits')
obs.add_scans()
obs = obs.avg_coherent(0.,scan_avg=True)
obs = mod.observe_same(obs,ampcal=True,phasecal=True)

# Testing the chi^2
dtypes = ['vis','amp','bs','cphase','logcamp','camp','logamp', 'logcamp_diag', 'cphase_diag'] #, 'bs', 'amp', 'cphase', 'cphase_diag', 'camp', 'logcamp', 'logcamp_diag']
for dtype in dtypes:
    print('\nTesting chi^2 dtype:',dtype)
    chisqdata = eh.modeling.modeling_utils.chisqdata(obs, dtype)
    chisq = eh.modeling.modeling_utils.chisq(mod, chisqdata[2], chisqdata[0], chisqdata[1], dtype)
    print("chisq: %f" % chisq)
    print('\nTesting gradient')
    for x in [['F0',1e-10,0],['d',1e-4*eh.RADPERUAS,1],['x0',1e-4*eh.RADPERUAS,2],['y0',1e-4*eh.RADPERUAS,3]]:
        mod2 = mod.copy()
        dx = x[1]
        mod2.params[0][x[0]] += dx
        chisq2 = eh.modeling.modeling_utils.chisq(mod2, chisqdata[2], chisqdata[0], chisqdata[1], dtype)
        chisq_grad = eh.modeling.modeling_utils.chisqgrad(mod2, chisqdata[2], chisqdata[0], chisqdata[1], dtype, [True, True, True, True, False, False, False, False])
        if x[0] == 'F0':
            mult = 1
        else:
            mult = eh.RADPERUAS
        print(x[0],(chisq2-chisq)/dx*mult,chisq_grad[x[2]]*mult,(chisq2-chisq)/dx/chisq_grad[x[2]])

# Fit a model
mod_init = eh.model.Model()
mod_init.add_ring(1.5, 35.*eh.RADPERUAS)
mod_init.add_ring(0.5, 5.*eh.RADPERUAS)
prior = mod_init.default_prior()
mod_fit = eh.modeler_func(obs, mod_init, prior, d1='vis',alpha_d1=1,minimizer_method='Powell')
# Repeat the fit using a different minimizer
mod_fit = eh.modeler_func(obs, mod_fit, prior, d1='vis',alpha_d1=1,minimizer_method='L-BFGS-B')

# Fit holding some parameters fixed
mod_init = eh.model.Model()
mod_init.add_ring(1.5, 35.*eh.RADPERUAS)
mod_init.add_ring(0.5, 5.*eh.RADPERUAS)
prior = mod_init.default_prior()
prior[0]['x0'] = {'prior_type':'fixed'}
prior[0]['y0'] = {'prior_type':'fixed'}
mod_fit = eh.modeler_func(obs, mod_init, prior, d1='amp',d2='cphase',alpha_d1=1,stop=1e-50,minimizer_method='Powell')
mod_fit.make_image(100.*eh.RADPERUAS, 128).blur_circ(5.*eh.RADPERUAS).display()

# Fit a multi-component model
mod_init = eh.model.Model()
mod_init.add_circ_gauss(0.2, 5.*eh.RADPERUAS, 10.*eh.RADPERUAS, 0)
mod_init.add_circ_gauss(0.2, 5.*eh.RADPERUAS, 0., 10.*eh.RADPERUAS)
prior = mod_init.default_prior()
for j in range(len(prior)):
    prior[j]['F0']['prior_type'] = 'fixed'

mod_fit2 = eh.modeler_func(obs, mod_init, prior, d1='vis',alpha_d1=1,stop=1e-50,maxit=1000)
