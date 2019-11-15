# Note: this is an example sequence of commands to run in ipython
# The matplotlib windows may not open/close properly if you run this directly as a script

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import ehtim as eh

mod = eh.model.Model()
#mod.add_circ_gauss(1.5, 50.*eh.RADPERUAS, x0 = 10.*eh.RADPERUAS, y0 = 30.*eh.RADPERUAS)
#mod.add_mring(1.5, 50.*eh.RADPERUAS, beta_list=[0.5])
mod.add_thick_ring(1.5, 50.*eh.RADPERUAS, alpha=10.*eh.RADPERUAS, x0 = 10.*eh.RADPERUAS, y0 = 30.*eh.RADPERUAS)
#mod.add_ring(1.5, 25.*eh.RADPERUAS, x0 = 10.*eh.RADPERUAS, y0 = 30.*eh.RADPERUAS)
#mod.add_gauss(1.5, 20.*eh.RADPERUAS, 40.*eh.RADPERUAS, PA=20.*np.pi/180., x0 = 10.*eh.RADPERUAS, y0 = 30.*eh.RADPERUAS)
#mod.add_point(0.5, 5.02*eh.RADPERUAS, 25.04*eh.RADPERUAS)

# View the model as an image
mod_im = mod.make_image(100.*eh.RADPERUAS, 256)
mod_im.display()
#mod_im.blur_circ(10.*eh.RADPERUAS).display()

# Make an observation
obs = eh.obsdata.load_uvfits('/home/michael/Dropbox/ER5_polarization_calibration/ER5/postproc-hops-lo/3.+netcal/3601/hops_3601_M87+netcal.uvfits')
obs.add_scans()
obs = obs.avg_coherent(0.,scan_avg=True)
im.ra = obs.ra
im.dec = obs.dec
im.rf = obs.rf
eh.comp_plots.plotall_obs_im_compare(obs,[im,mod],'uvdist','amp') 

# Manual check of visibility consistency
im = im.regrid_image(im.fovx()*2, 1024)
mod_im = mod.image_same(im)
mod_im.display()
mod_im.blur_circ(10.*eh.RADPERUAS).display()
for u in np.arange(0, 4e9, 1e9):
    for v in np.arange(-4e9, 4e9, 1e9):
        print('%f %f %f %f %f %f' % (u/1e9,v/1e9,np.abs(mod.sample_uv(u,v)),np.abs(mod_im.sample_uv([[u,v]])[0][0]),np.angle(mod.sample_uv(u,v)),np.angle(mod_im.sample_uv([[u,v]])[0][0])))
