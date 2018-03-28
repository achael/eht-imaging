#example script for running data domain clean

import ehtim as eh
from ehtim.imaging.clean import *

#################################################################
# Data domain clean with complex visibilities
im = eh.image.load_txt('./models/avery_sgra_eofn.txt')
arr = eh.array.load_txt('./arrays/EHT2017.txt') 
#arr = eh.array.load_txt('./arrays/EHT2025.txt')
obs = im.observe(arr, 1000, 600, 0, 24., 4.e10, add_th_noise=False, phasecal=True)
prior = eh.image.make_square(obs, 128, 1.5*im.fovx())

# data domain clean with visibilities
outvis = dd_clean_vis(obs, prior, niter=100, loop_gain=0.1, method='min_chisq',weighting='uniform')

#################################################################
# Data domain clean directly with bispectrum
# trial image 2 -- 2 Gaussians
im2 = eh.image.make_square(obs, 256, 3*im.fovx())
im2 = im2.add_gauss(1., (1*eh.RADPERUAS, 1*eh.RADPERUAS, 0, 0, 0))
im2 = im2.add_gauss(.7, (1*eh.RADPERUAS, 1*eh.RADPERUAS, 0, -75*eh.RADPERUAS, 30*eh.RADPERUAS))
obs2 = im2.observe(arr, 600, 600, 0, 24., 4.e9, add_th_noise=False, phasecal=False)
prior = eh.image.make_square(obs, 50, 3*im.fovx())

outbs = dd_clean_amp_cphase(obs2, prior, niter=50, loop_gain=0.1, loop_gain_init=.01,phaseweight=2, weighting='uniform', bscount="min")


