# Note: this is an example sequence of commands to run in ipython that generates a movie
# from a single night of observation. 
#
# Contact Katie Bouman (klbouman@mit.edu) for any questions 
#
# The methods/techniques used in this, referred to as StarWars, are described in 
# "Reconstructing Video from Interferometric Measurements of Time-Varying Sources" 
# by Katherine L. Bouman, Michael D. Johnson, Adrian V. Dalca, 
# Andrew Chael, Freek Roelofs, Sheperd S. Doeleman, and William T. Freeman

# Note: must import ehtim outside the ehtim directory
# either in parent eht-imaging directory or after installing with setuptools
from __future__ import division
from __future__ import print_function

import numpy as np
import ehtim as eh
from   ehtim.calibrating import self_cal as sc
from ehtim.imaging import patch_prior as pp
import ehtim.image as image
from ehtim.imaging import starwarps as sw
import matplotlib.pyplot as plt
import sys, os, copy
import scipy
import scipy.optimize as opt


############## parameters ##############

# data file
obsname = 'sample_movie.uvfits'

# image parameters
flux = 2.0
fwhm = 50 * eh.RADPERUAS
fov = 100 * eh.RADPERUAS
NPIX = 30
npixels = NPIX**2

# StarWarps optimization parameters
warp_method = 'phase'
measurement = {'vis':1 } # {'amp':1, 'cphase':1}
interiorPriors = True
reassign_apxImgs = False
numLinIters = 5
variance_img_diff = 1e-7

# parameters associated with EM 
nIters = 30
NHIST = 5000
stop=1e-10
maxit=4000

# directory where to save results
SAVE = True
dirname = '../results'


############## load data ##############

# load in the data
obs = eh.obsdata.load_uvfits(obsname)

# split the observations based upon the time
obs_List = sw.splitObs(obs)

############## reconstruct movie with no warp field ##############

# initialize the mean and the image covariance for the prior. 
# this can be a single image to be the same mean and covariance for each 
# time, or different for each time by appending an image/matrix for each timestep

# initialize mean
meanImg = []
emptyprior = eh.image.make_square(obs, NPIX, fov)
gaussprior = emptyprior.add_gauss(flux, (fwhm, fwhm, 0, 0, 0))
meanImg.append(gaussprior.copy())

# initialize covariance
imCov = []
imCov.append( sw.gaussImgCovariance_2(meanImg[0], powerDropoff=2.0, frac=1./2.) )

# make the covariance matrix that says how much variation there should be between frames in time 
noiseCov_img = np.eye(npixels)*variance_img_diff

# initialize the flowbasis and get the initTheta which says how to specify no motion for the specified flow basis
init_x, init_y, flowbasis_x, flowbasis_y, initTheta = sw.affineMotionBasis_noTranslation(meanImg[0])

# run StarWarps to find the distribution of the image at each timestep
expVal_t, expVal_t_t, expVal_tm1_t, loglikelihood, apxImgs = sw.computeSuffStatistics(
    meanImg, imCov, obs_List, noiseCov_img, initTheta, init_x, init_y, 
    flowbasis_x, flowbasis_y, initTheta, method=warp_method, measurement=measurement, 
    interiorPriors=interiorPriors, numLinIters=numLinIters, compute_expVal_tm1_t=False)


# save out results as a movie
if SAVE:
    # make the directory to save out the results
    try:
        os.stat(dirname)
    except:
        os.mkdir(dirname)
    
    # save out the movie that is generated assuming there is no motion
    sw.movie(expVal_t, out = dirname + '/movie_nomotion.mp4')


############## learn warp field and reconstruct movie using derived EM-algorithm ##############

# number of motion parameters
nbasis = flowbasis_x.shape[2] 

# set the bounds for the motion parameters
bnds = []
for b in range(0,nbasis):
    bnds.append( (-1.5,1.5) )


# initialize optimization parameters
newTheta = copy.deepcopy(initTheta)
feval = 0.0
optdict = {'maxiter':maxit, 'ftol':stop, 'maxcor':NHIST, 'disp':True} # minimizer params
negll = []
thetas = []
fevals = []

for iter in range(0, nIters+1):

    print('\rIteration %i of %i ...' % (iter, nIters+1) )
    
    # ========== E-step ========== #
    if iter==0 or reassign_apxImgs:
        apxImgs = False

    # solve for the sufficient statistics using the StarWarps approach with the previous value 
    # the warp parameters in newTheta
    expVal_t, expVal_t_t, expVal_tm1_t, loglikelihood, apxImgs = sw.computeSuffStatistics(
        meanImg, imCov, obs_List, noiseCov_img, newTheta, init_x, init_y, 
        flowbasis_x, flowbasis_y, initTheta, method=warp_method, measurement=measurement, 
        interiorPriors=interiorPriors, numLinIters=numLinIters, apxImgs=apxImgs)

    # save the negative log likelihood (nll), the value of the warp parameters (thetas) 
    # and the evaluation of the optimization function (feval)
    negll.append(-loglikelihood[2])
    thetas.append(newTheta)
    fevals.append(feval)
    
    # ========== visualize and save results ========== #
    
    if SAVE: 
    
        # make the directory to save out the results
        try:
            os.stat(dirname + '/' + str(iter))
        except:
            os.mkdir(dirname + '/' + str(iter))
    
        # save frames as the mean and the estimated standard deviation of each frame
        stdevImg = meanImg[0].copy()
        for i in range(0,len(obs_List)):
            stdevImg.imvec = np.sqrt(np.diag(expVal_t_t[i]))
            expVal_t[i].save_fits(dirname + '/' + str(iter) + '/mean_' + str(i) + '.fits')
            stdevImg.save_fits(dirname + '/' + str(iter) +  '/stdev_' + str(i) + '.fits')

        # compute the average image
        avgImg = meanImg[0].copy()
        avgImg.imvec = np.mean([im.imvec for im in expVal_t],axis=0)
    
        #save flow diagram
        plt.figure(), sw.plot_Flow(avgImg, thetas[iter], init_x, init_y, flowbasis_x, flowbasis_y, initTheta, step=1)
        plt.savefig(dirname + '/flow_' + str(iter) +  '.pdf')

        #save a movie
        sw.movie(expVal_t, dirname + '/movie_' + str(iter) + '.mp4')

        # save out mat file with the information
        scipy.io.savemat(dirname + '/info_' + str(iter) + '.mat', {'negll':negll, 'thetas':thetas, 'funeval':fevals})
    
    # ========== M-step ========== #
    if iter < nIters:
        result = opt.minimize(sw.expnegloglikelihood, newTheta, args=(expVal_t, expVal_t_t, expVal_tm1_t, meanImg, imCov, obs_List, noiseCov_img, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, warp_method), method='L-BFGS-B', jac=sw.deriv_expnegloglikelihood, bounds=bnds, options=optdict)
        newTheta = result.x
        feval = result.fun



