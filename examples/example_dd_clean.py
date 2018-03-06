import ehtim as eh
#from clean import *

# trial image 1 -- Sgr A* image
im = eh.image.load_txt('./models/avery_sgra_eofn.txt')
#arr = eh.array.load_txt('./arrays/EHT2017_noRedundant.txt')
arr = eh.array.load_txt('./arrays/EHT2025.txt')
obs = im.observe(arr, 1000, 600, 0, 24., 4.e10, add_th_noise=False, phasecal=False)
prior = eh.image.make_square(obs, 128, 1.5*im.fovx())

# data domain clean with visibilities
outvis = dd_clean_vis(obs, prior, niter=100, loop_gain=0.1, method='min_chisq',weighting='uniform')

# data domain clean with bispectrum
# bispectrum dd clean doesn't seem to work well with the Sgr A* image !!
outbs = dd_clean_bispec_full(obs, prior, niter=2, loop_gain=0.1, order=1, weighting='uniform', bscount="min")


#outbs = dd_clean_bispec_imweight(obs, prior, niter=100, loop_gain=0.1,imweight=2, weighting='uniform', bscount="max")
outbs = dd_clean_amp_cphase(obs, prior, niter=1000, loop_gain=0.05, loop_gain_init=.05,phaseweight=3.e7, weighting='uniform', bscount="min")
obs_sc = eh.self_cal.self_cal(obs, outbs, method='phase')
obs_sc = eh.self_cal.self_cal(obs_sc, outbs, method='phase')

#################################################################
# trial image 2 -- 2 Gaussians
im2 = eh.image.make_square(obs, 256, 3*im.fovx())
im2 = im2.add_gauss(1., (1*eh.RADPERUAS, 1*eh.RADPERUAS, 0, 0, 0))
im2 = im2.add_gauss(.7, (1*eh.RADPERUAS, 1*eh.RADPERUAS, 0, -75*eh.RADPERUAS, 30*eh.RADPERUAS))
obs2 = im2.observe(arr, 600, 600, 0, 24., 4.e9, add_th_noise=False, phasecal=False)
prior = eh.image.make_square(obs, 50, 3*im.fovx())

# data domain clean with visibilities -- shouldn't work with uncalibrated phases!
out = dd_clean_vis(obs2, prior, niter=50, loop_gain=0.2, method='min_chisq',weighting='uniform')

# data domain clean with bispectrum
out = dd_clean_bispec(obs2, prior, niter=50, loop_gain=0.2, order=2, weighting='uniform', bscount="min")
out = dd_clean_bispec_full(obs2, prior, niter=5, loop_gain=0.2, weighting='uniform', bscount="min")

outbs = dd_clean_amp_cphase(obs2, prior, niter=50, loop_gain=0.1, loop_gain_init=1,phaseweight=2, weighting='uniform', bscount="min")
outbs = dd_clean_bispec_imweight(obs2, prior, niter=50, loop_gain=0.1,imweight=2, weighting='uniform', bscount="max")

#################################################################
# trial image 3 -- 4 Gaussians
prior = eh.image.make_square(obs, 64, 1.5*im.fovx())
im3 = prior.add_gauss(1, (5*eh.RADPERUAS, 5*eh.RADPERUAS, 0, 0, 0))
im3 = im3.add_gauss(.7, (5*eh.RADPERUAS, 5*eh.RADPERUAS, 0, -75*eh.RADPERUAS, -75*eh.RADPERUAS))
im3 = im3.add_gauss(.7, (5*eh.RADPERUAS, 5*eh.RADPERUAS, 0, 75*eh.RADPERUAS, 0*eh.RADPERUAS))
im3 = im3.add_gauss(.7, (5*eh.RADPERUAS, 5*eh.RADPERUAS, 0, -50*eh.RADPERUAS, 80*eh.RADPERUAS))
#im3 = im3.add_gauss(.5, (5*eh.RADPERUAS, 5*eh.RADPERUAS, 0, 0*eh.RADPERUAS, -80*eh.RADPERUAS))

obs3 = im3.observe(arr, 600, 600, 0, 24., 4.e9, add_th_noise=False)

# data domain clean with visibilities
out = dd_clean_vis(obs3, prior, niter=50, loop_gain=0.2, method='min_chisq',weighting='uniform')

# data domain clean with bispectrum
out = dd_clean_bispec(obs3, prior, niter=100, loop_gain=0.1, order=2, weighting='uniform', bscount="min")

#image oj287 data
out = dd_clean_bispec_full(obs_oj, gaussprior, niter=50, loop_gain=0.1, weighting='uniform', bscount="max")
