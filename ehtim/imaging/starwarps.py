# See example_starwarps.py for an example of how to use these methods
# Contact Katie Bouman (klbouman@caltech.edu) for any questions
#
# The methods/techniques used in this, referred to as StarWars, are described in 
# "Reconstructing Video from Interferometric Measurements of Time-Varying Sources" 
# by Katherine L. Bouman, Michael D. Johnson, Adrian V. Dalca, 
# Andrew Chael, Freek Roelofs, Sheperd S. Doeleman, and William T. Freeman

from __future__ import division
from __future__ import print_function

import numpy as np
#import ehtim as eh
import ehtim.image as image
import ehtim.observing.pulses
from ehtim.observing.obs_helpers import *
from ehtim.imaging.imager_utils import chisqdata

import scipy.stats as st
import scipy
import copy
import sys

import matplotlib.pyplot as plt

PROPERROR = True

##################################################################################################


def solve_singleImage(mu, Lambda_orig, obs, measurement={'vis':1}, numLinIters=5, mask=[], normalize=False):

    if len(mask):
        Lambda = Lambda_orig[mask[:,None] & mask[None,:]].reshape([np.sum(mask), -1])
    else:
        Lambda = Lambda_orig 
        mask = np.ones(mu.imvec.shape)>0  

    if list(measurement.keys())==1 and measurement.keys()[0]=='vis':
        numLinIters = 1
    
    z_List_t_t = mu.copy()
    z_List_lin = mu.copy()
        
    for k in range(0,numLinIters):
        meas, idealmeas, F, measCov, valid = getMeasurementTerms(obs, z_List_lin, measurement=measurement, mask=mask, normalize=normalize)
        if valid:
            z_List_t_t.imvec[mask], P_List_t_t = prodGaussiansLem2(F, measCov, meas, mu.imvec[mask], Lambda)
                
            if k < numLinIters-1:
                z_List_lin = z_List_t_t.copy()
        else:
            z_List_t_t = mu.copy()
            P_List_t_t = copy.deepcopy(Lambda)
        
    return (z_List_t_t, P_List_t_t, z_List_lin)


##################################################################################################

def forwardUpdates_apxImgs(mu, Lambda_orig, obs_List, A_orig, Q_orig, init_images, measurement={'vis':1}, lightcurve=None, numLinIters=5, interiorPriors=False, mask=[], normalize=False):
    '''
    Gaussian image prior:
    :param mu: (list - len(mu)=num_time_steps or 1): every element is an image object which contains the mean image 
        at given timestep. If list length is one mean image is duplicated for all time steps
    :param Lambda_orig: (list - len(Lambda_orig)=num_time_steps or 1): original unmasked covariance matrix.
        Every element is a 2D numpy array which contains the covariance at a given timestep. 
        If list length is one, the cov image is duplicated for all time steps.
        
    Observations: 
    :param obs_List: list of observations, for each time step
    
    Dynamical Evolution Model
    :param A_orig: original unmasked A matrix - time-invariant mean of warp field for dynamical evolution
    :param Q_orig: original unmasked Q matrix - time-invariant covariance matrix of dynamical evolution model,
        describing the amount of allowed intensity deviation
    
    Other Parameters:
    :param init_images: option to provide initialization for the forward updates.
        If none provided, then use the initialization from StarWarps paper
    :param measurement: data products used
    :param lightcurve: light curve time seres, needed if imposing a flux constraint
    :param numLinIters: number of linearized iterations. We have non-linear measurement function f_{t}(x_{t}), 
        and we linearize the solution around \tilde{x_{t}} by taking the first order Taylor series expansion of f.
        To improve the solution of the forward and backward terms, each step in the forward pass can be 
        iteratively re-solved and \tilde{x}_{t} can be updated at each iteration. 
        The values of \tilde{x}_{t} are fixed for the backward pass. 
        Note that if f is linear, only a single iteration will be enough to converge to the optimal solution. 
        Thus, if the only measurement is visibility, numLinIters = 1 should be set.
    :param interiorPriors: flag for whether to use interior priors
    :param mask: to select parts of the image to utilize. default is True for all pixels. 
        This is because getMeasurementTerms doesn't work with a mask yet.
    :param normalize: flag for whether to normalize sigma in getMeasurementTerms
    '''
    
    # linear case: measurement function is linear and problem is convex
    if list(measurement.keys())==1 and measurement.keys()[0]=='vis':
        numLinIters = 1
    
    # apply mask
    if len(mask):
        A = A_orig[mask[:,None] & mask[None,:]].reshape([np.sum(mask), -1])
        Q = Q_orig[mask[:,None] & mask[None,:]].reshape([np.sum(mask), -1])
        Lambda = []
        for t in range(0, len(Lambda_orig)):
            Lambda.append( Lambda_orig[t][mask[:,None] & mask[None,:]].reshape([np.sum(mask), -1]) )
    else:
        Lambda = Lambda_orig
        A = A_orig
        Q = Q_orig

    #if measurement=='bispectrum':
    #    print 'WARNING: check the loglikelihood for non-linear functions'

    # create an image of 0's
    zero_im = mu[0].copy()
    zero_im.imvec = 0.0*zero_im.imvec
    
    # intitilize the prediction and update mean and covariances
    z_List_t_t = [] # Mean of the hidden (state) image at t given data up to time t
    P_List_t_t = [] # Covariance of the hidden (state) image at t given data up to time t
    
    # initialize z and P lists with to be all zeros
    for t in range(0,len(obs_List)):
        z_List_t_t.append(zero_im.copy())
        P_List_t_t.append(np.zeros(Lambda[0].shape))
        
    # prediction 1 list: prediction at time t given all information up to t-1
    z_List_t_tm1 = copy.deepcopy(z_List_t_t)
    P_List_t_tm1 = copy.deepcopy(P_List_t_t)
    
    # prediction 2 list: deep copy of prediction 1 list (possibly intermediate state variable)
    z_star_List_t_tm1 = copy.deepcopy(z_List_t_t)
    P_star_List_t_tm1 = copy.deepcopy(P_List_t_t)
    # initialize linearization z's
    z_List_lin = copy.deepcopy(z_List_t_t)
    
    loglikelihood_prior = 0.0
    loglikelihood_data = 0.0
    
    # for each forward timestep... 
    for t in range(0,len(obs_List)):
        sys.stdout.write('\rForward timestep %i of %i total timesteps...' % (t,len(obs_List)))
        sys.stdout.flush()
        
        print('forward timestep: ' + str(t))
        
        # Duplicate mean and covariance if needed
        if len(mu) == 1:
            mu_t = mu[0]
            Lambda_t = Lambda[0]
        # get current image's mean and covariance
        else:
            mu_t = mu[t]
            Lambda_t = Lambda[t]
        
        # use lightcurve data if provided
        if lightcurve:
            tot = np.sum(mu_t.imvec)
            mu_t.imvec = lightcurve[t]/tot * mu_t.imvec
            Lambda_t = (lightcurve[t]/tot)**2 * Lambda_t

            
        # predict
        # Initialization of hidden state mean and covariance for t = 0
        if t==0:
            z_star_List_t_tm1[t].imvec = copy.deepcopy(mu_t.imvec)
            P_star_List_t_tm1[t] = copy.deepcopy(Lambda_t)
        # StarWarps initialization of hidden state mean and covariance
        else:
            z_List_t_tm1[t].imvec[mask] = np.dot( A, z_List_t_t[t-1].imvec[mask] ) 
            if PROPERROR:
                P_List_t_tm1[t] = Q + np.dot( np.dot( A, P_List_t_t[t-1] ), np.transpose(A) )
            else:
                print('no prop error')
                P_List_t_tm1[t] = Q 
            
            # main predict step, using Lemma 1 from StarWarps supplementary doc (also see eq 29-30), using interior priors
            if interiorPriors:
                z_star_List_t_tm1[t].imvec[mask], P_star_List_t_tm1[t] = prodGaussiansLem1( mu_t.imvec[mask], Lambda_t, z_List_t_tm1[t].imvec[mask], P_List_t_tm1[t] )
            else:
                z_star_List_t_tm1[t] = z_List_t_tm1[t].copy()
                P_star_List_t_tm1[t] = copy.deepcopy(P_List_t_tm1[t])
        
        
        # update
        # either go with user-provided initialization (if given) or take z_star_List_t_tm1 as an initialization
        if init_images is None:
            init_images_t = z_star_List_t_tm1[t].copy()
        elif len(init_images) == 1:
            init_images_t = init_images[0]
        else:
            init_images_t = init_images[t]
        z_List_lin[t] = init_images_t.copy()
        
        # Do the linearized iterations
        for k in range(0,numLinIters):
            # F is the derivative of the Forward model with respect to the unknown parameters
            if lightcurve:
                meas, idealmeas, F, measCov, valid = getMeasurementTerms(obs_List[t], z_List_lin[t], measurement=measurement, tot_flux=lightcurve[t], mask=mask, normalize=normalize)
            else:
                meas, idealmeas, F, measCov, valid = getMeasurementTerms(obs_List[t], z_List_lin[t], measurement=measurement, tot_flux=None, mask=mask, normalize=normalize)
            
            # main update step, using Lemma 2 from StarWarps supplementary doc (also see eq 30-31)
            if valid:
                z_List_t_t[t].imvec[mask], P_List_t_t[t] = prodGaussiansLem2(F, measCov, meas, z_star_List_t_tm1[t].imvec[mask], P_star_List_t_tm1[t])
                
                if k < numLinIters-1:
                    z_List_lin[t] = z_List_t_t[t].copy()
            else:
                z_List_t_t[t] = z_star_List_t_tm1[t].copy()
                P_List_t_t[t] = copy.deepcopy(P_star_List_t_tm1[t])
        
        # update the prior log likelihood, using interior priors
        if t>0 and interiorPriors:
            loglikelihood_prior = loglikelihood_prior + evaluateGaussianDist_log( z_List_t_tm1[t].imvec[mask], mu_t.imvec[mask], Lambda_t + P_List_t_tm1[t] )
            
        # update the data log likelihood
        if valid:
            loglikelihood_data = loglikelihood_data + evaluateGaussianDist_log( np.dot(F , z_star_List_t_tm1[t].imvec[mask]), meas, measCov + np.dot( F, np.dot(P_star_List_t_tm1[t], F.T)) )
            
            
    # compute the log likelihood (equation 27 in StarWarps paper)    
    loglikelihood = loglikelihood_prior + loglikelihood_data
    return ((loglikelihood_data, loglikelihood_prior, loglikelihood), z_List_t_tm1, P_List_t_tm1, z_List_t_t, P_List_t_t, z_List_lin)


###################################### EXTENDED MESSAGE PASSING ########################################

def backwardUpdates(mu, Lambda_orig, obs_List, A_orig, Q_orig, measurement={'vis':1}, lightcurve=None, apxImgs=False, mask=[], normalize=False):
    '''
    Gaussian image prior:
    :param mu: (list - len(mu)=num_time_steps or 1): every element is an image object which contains the mean image 
        at given timestep. If list length is one mean image is duplicated for all time steps
    :param Lambda_orig: (list - len(Lambda_orig)=num_time_steps or 1): original unmasked covariance matrix.
        Every element is a 2D numpy array which contains the covariance at a given timestep. 
        If list length is one, the cov image is duplicated for all time steps.
        
    Observations: 
    :param obs_List: list of observations, for each time step
    
    Dynamical Evolution Model
    :param A_orig: original unmasked A matrix - time-invariant mean of warp field for dynamical evolution
    :param Q_orig: original unmasked Q matrix - time-invariant covariance matrix of dynamical evolution model,
        describing the amount of allowed intensity deviation
        
    Other Parameters
    :param measurement: data products used
    :param lightcurve: light curve time seres, needed if imposing a flux constraint
    :param apxImgs: ???
    :param mask: to select parts of the image to utilize. default is True for all pixels.
        This is because getMeasurementTerms doesn't work with a mask yet.
    :param normalize: flag for whether to normalize sigma in getMeasurementTerms
    '''
    
    if len(mask):
        A = A_orig[mask[:,None] & mask[None,:]].reshape([np.sum(mask), -1])
        Q = Q_orig[mask[:,None] & mask[None,:]].reshape([np.sum(mask), -1])
        Lambda = []
        for t in range(0, len(Lambda_orig)):
            Lambda.append( Lambda_orig[t][mask[:,None] & mask[None,:]].reshape([np.sum(mask), -1]) )
    else:
        Lambda = Lambda_orig
        A = A_orig
        Q = Q_orig
        
    # create an image of 0's 
    zero_im = mu[0].copy()
    zero_im.imvec = 0.0*zero_im.imvec
    
    # intitilize the prediction and update mean and covariances
    z_t_t = []
    P_t_t = []
    # update list
    for t in range(0,len(obs_List)):   
        z_t_t.append(zero_im.copy())  
        P_t_t.append(np.zeros(Lambda[0].shape)) 
    # prediction 1 list
    z_star_t_tp1 = copy.deepcopy(z_t_t)   
    P_star_t_tp1 = copy.deepcopy(P_t_t) 
    
    
    lastidx = len(obs_List)-1
    for t in range(lastidx,-1,-1):
        sys.stdout.write('\rBackward timestep %i of %i total timesteps...' % (t,len(obs_List)))
        sys.stdout.flush()
    
        if len(mu) == 1:
            mu_t = mu[0]
            Lambda_t = Lambda[0]
        else:
            mu_t = mu[t]
            Lambda_t = Lambda[t]
        if lightcurve is not None:
            tot = np.sum(mu_t.imvec)
            mu_t.imvec = lightcurve[t]/tot * mu_t.imvec
            Lambda_t = (lightcurve[t]/tot)**2 * Lambda_t
        
        # predict
        if t==lastidx:
            z_star_t_tp1[t].imvec = copy.deepcopy(mu_t.imvec) 
            P_star_t_tp1[t] = copy.deepcopy(Lambda_t) 
        else:
            if PROPERROR:
                z_star_t_tp1[t].imvec[mask], P_star_t_tp1[t] = prodGaussiansLem2( A, Q + P_t_t[t+1], z_t_t[t+1].imvec[mask], mu_t.imvec[mask], Lambda_t)
            else:
                print('no prop error')
                z_star_t_tp1[t].imvec[mask], P_star_t_tp1[t] = prodGaussiansLem2( A, Q, z_t_t[t+1].imvec[mask], mu_t.imvec[mask], Lambda_t)

        # update
        if lightcurve:
            meas, idealmeas, F, measCov, valid = getMeasurementTerms(obs_List[t], apxImgs[t], measurement=measurement, tot_flux=lightcurve[t], mask=mask, normalize=normalize)
        else:
            meas, idealmeas, F, measCov, valid = getMeasurementTerms(obs_List[t], apxImgs[t], measurement=measurement, tot_flux=None, mask=mask, normalize=normalize)   

        if valid:
            z_t_t[t].imvec[mask], P_t_t[t] = prodGaussiansLem2(F, measCov, meas, z_star_t_tp1[t].imvec[mask], P_star_t_tp1[t])
            
        else:
            z_t_t[t] = z_star_t_tp1[t].copy()
            P_t_t[t] = copy.deepcopy(P_star_t_tp1[t])
    
    return (z_t_t, P_t_t)

    
def smoothingUpdates(z_t_t, P_t_t, z_t_tm1, P_t_tm1, A_orig, mask=[]):
    '''
    Smoothing
    '''
    z = copy.deepcopy(z_t_t)
    P = copy.deepcopy(P_t_t)
    backwardsA = copy.deepcopy(P_t_t)
    
    if len(mask):
        A = A_orig[mask[:,None] & mask[None,:]].reshape([np.sum(mask), -1])
    
    lastidx = len(z)-1
    for t in range(lastidx,-1,-1):
        
        if t < lastidx: 
            backwardsA[t] = np.dot( np.dot(P_t_t[t], A.T ), np.linalg.inv(P_t_tm1[t+1]) )
            z[t].imvec[mask] = z_t_t[t].imvec[mask] + np.dot( backwardsA[t], z[t+1].imvec[mask] - z_t_tm1[t+1].imvec[mask] )
            P[t] = np.dot( np.dot( backwardsA[t] , P[t+1] - P_t_tm1[t+1]), backwardsA[t].T ) + P_t_t[t]
    
    return (z, P, backwardsA)
    

    
def computeSuffStatistics(mu, Lambda, obs_List, Upsilon, theta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, init_images=None, method='phase', measurement={'vis':1}, lightcurve=None, interiorPriors=False, numLinIters=1, compute_expVal_tm1_t=True, mask=[], normalize=False):
    '''
    Gaussian image prior:
    :param mu: (list - len(mu)=num_time_steps or 1): every element is an image object which contains the mean image 
        at given timestep. If list length is one mean image is duplicated for all time steps
    :param Lambda_orig: (list - len(Lambda_orig)=num_time_steps or 1): original unmasked covariance matrix.
        Every element is a 2D numpy array which contains the covariance at a given timestep. 
        If list length is one, the cov image is duplicated for all time steps.
        
    Observations: 
    :param obs_List: list of observations, for each time step
    
    Dynamical Evolution Model
    :param A_orig: original unmasked A matrix - time-invariant mean of warp field for dynamical evolution
    :param Q_orig: original unmasked Q matrix - time-invariant covariance matrix of dynamical evolution model,
        describing the amount of allowed intensity deviation
    
    Other Parameters:
    :param init_images: option to provide initialization for the forward updates.
        If none provided, then use the initialization from StarWarps paper
    :param measurement: data products used
    :param lightcurve: light curve time seres, needed if imposing a flux constraint
    :param numLinIters: number of linearized iterations. We have non-linear measurement function f_{t}(x_{t}), 
        and we linearize the solution around \tilde{x_{t}} by taking the first order Taylor series expansion of f.
        To improve the solution of the forward and backward terms, each step in the forward pass can be 
        iteratively re-solved and \tilde{x}_{t} can be updated at each iteration. 
        The values of \tilde{x}_{t} are fixed for the backward pass. 
        Note that if f is linear, only a single iteration will be enough to converge to the optimal solution. 
        Thus, if the only measurement is visibility, numLinIters = 1 should be set.
    :param interiorPriors: Flag for whether to use interior priors.
    :param compute_expVal_tm1_t: flag for whether to compute the second sufficient statistic, E[x_{t-1}x_{t}^{T}].
    :param mask: to select parts of the image to utilize. default is True for all pixels. 
        This is because getMeasurementTerms doesn't work with a mask yet.
    :param normalize: flag for whether to normalize sigma in getMeasurementTerms
    '''
    
    # if mask not provided, create default mask
    if not len(mask):
        mask = np.ones(mu[0].imvec.shape)>0
    
    # check if first mean image is square
    if mu[0].xdim != mu[0].ydim:
        error('Error: This has only been checked thus far on square images!')

    # lightcurve and flux constraint go together
    if lightcurve == None  and 'flux' in measurement.keys(): #KATIE ADDED FEB 1 2021
        error('Error: if you are using a flux constraint you must specify a lightcurve')
    
    # if the visibility is the only measurement, 
    if list(measurement.keys())==1 and measurement.keys()[0]=='vis':
        numLinIters = 1
    
    # calculate matrix to represent warp field, from first mean image
    warpMtx = calcWarpMtx(mu[0], theta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method=method) 
    
    # Parameterize dynamical evolution model with Guassian
    # the time-invariant mean warp field
    A = warpMtx
    # time-invariant covariance matrix describing amount of allowed intensity variation in the warp field
    Q = Upsilon
    
    # Do forward passes
    loglikelihood, z_t_tm1, P_t_tm1, z_t_t, P_t_t, apxImgs = forwardUpdates_apxImgs(mu, Lambda, obs_List, A, Q, init_images=init_images, measurement=measurement, lightcurve=lightcurve, interiorPriors=interiorPriors, numLinIters=numLinIters, mask=mask, normalize=normalize)
 
    # Extended message passing with backward passes, using interior priors
    if interiorPriors:
        # Do backward passes
        z_backward_t_t, P_backward_t_t = backwardUpdates(mu, Lambda, obs_List, A, Q, measurement=measurement, lightcurve=lightcurve, apxImgs=apxImgs, mask=mask, normalize=normalize)
        
        z = copy.deepcopy(z_backward_t_t)
        P = copy.deepcopy(P_backward_t_t)
        for t in range(0,len(obs_List)):
            if t==0:
                z[t] = z_backward_t_t[t].copy()
                P[t] = copy.deepcopy(P_backward_t_t[t])
            else:
                z[t].imvec[mask], P[t] = prodGaussiansLem1(z_t_tm1[t].imvec[mask], P_t_tm1[t], z_backward_t_t[t].imvec[mask], P_backward_t_t[t])
        
    # Use smoothing updates instead
    else:
        z, P, backwardsA = smoothingUpdates(z_t_t, P_t_t, z_t_tm1, P_t_tm1, A, mask=mask)



    expVal_t = copy.deepcopy(z)    
    #initilize the lists
    expVal_t_t = copy.deepcopy(P)
    expVal_tm1_t = copy.deepcopy(P)
    for t in range(0,len(obs_List)):
        # expected value of xx^T for each x
        z_t_hvec = np.array([z[t].imvec[mask]])
        expVal_t_t[t] = np.dot(z_t_hvec.T, z_t_hvec) + P[t]

        # expected value of x_t x_t-1^T for each x except for the first one
        if t>0 and interiorPriors==False and compute_expVal_tm1_t:
            z_tm1_hvec = np.array([z[t-1].imvec[mask]])
            expVal_tm1_t[t] = np.dot(z_tm1_hvec.T, z_t_hvec) + np.dot(backwardsA[t-1], P[t])

    # expected value of x_t x_t-1^T, using interior priors
    if interiorPriors and compute_expVal_tm1_t:
        expVal_tm1_t = JointDist(z, z_t_t, P_t_t, z_backward_t_t, P_backward_t_t, A, Q)
    
    return (expVal_t, expVal_t_t, expVal_tm1_t, loglikelihood, apxImgs)
    



def JointDist(z, z_List_t_t_forward, P_List_t_t_forward, z_List_t_t_backward, P_List_t_t_backward, A, Q):
    '''
    Calculate the joint distribution p(x_{t},x_{t-1} | y_{1:N})
    See section 2.2 in StarWarps supplementary doc, starting from eq 60
    '''
    
    expVal_tm1_t = []
    expVal_tm1_t.append(0.0)
    
    # section 2.2
    for t in range(1, len(z_List_t_t_forward) ):
        
        Sigma = Q + P_List_t_t_backward[t]
        Sigma_inv = np.linalg.inv(Sigma)
        
        # eq 76
        M = np.dot(P_List_t_t_backward[t], np.dot(Sigma_inv, A) )
        
        # eq 77, 78
        (m, C) = prodGaussiansLem2(A, Sigma, z_List_t_t_backward[t].imvec, z_List_t_t_forward[t-1].imvec,  P_List_t_t_forward[t-1]) 
        
        # eq 79
        D_tmp1 = np.dot(M, np.dot(C, M.T))
        D_tmp2 = np.dot(Q, np.dot( Sigma_inv, P_List_t_t_backward[t] ) )        
        D = np.dot(C, np.dot(M.T, np.linalg.inv(D_tmp1 + D_tmp2) ) )
        
        # eq 81
        F = C - np.dot(D, np.dot(M, C))
        
        # E[x_{t}]
        z_t_hvec = np.array([z[t].imvec])
        # E[x_{t-1}]
        z_tm1_hvec = np.array([z[t-1].imvec])
        
        # eq 88
        expVal_tm1_t.append( np.dot(F, np.linalg.inv(D.T))  + np.dot(z_tm1_hvec.T, z_t_hvec) ) 
        
    return expVal_tm1_t
    
    
    
#################################


def maximizeWarpMtx(expVal_t_t, expVal_tm1_t, expVal_t=0, B=0):
    
    M1 = np.zeros(expVal_tm1_t[1].shape)
    M2 = np.zeros(expVal_t_t[1].shape)
    
    for t in range(1,len(expVal_t_t)):
        #M1 = M1 + 0.5*expVal_tm1_t[t] + 0.5*expVal_tm1_t[t].T
        M1 = M1 + expVal_tm1_t[t].T
        if B !=0:
            M1 = M1 + np.dot(B, expVal_t[t])
        M2 = M2 + expVal_t_t[t-1]

    warpMtx = np.dot( M1, np.linalg.inv(M2) )
    return warpMtx
    
def maximizeTheta_multiIter(expVal_t_t, expVal_tm1_t, dummy_im, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method='phase', nIter=10):
    
    newTheta = centerTheta
    for i in range(0, nIter):
        newTheta = maximizeTheta(expVal_t_t, expVal_tm1_t, dummy_im, newTheta, newTheta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method=method)
        
    return newTheta


def maximizeTheta(expVal_t_t, expVal_tm1_t, dummy_im, Q, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method='phase'):
    
    if method == 'phase' or method == 'approx_phase':
        dWarp_dTheta = calc_dWarp_dTheta(dummy_im, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method='phase')
    else:
        error('ERROR: WE ONLY HANDLE PHASE WARP MINIMIZATION RIGHT NOW')
        
    warpMtx = calcWarpMtx(dummy_im, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method=method)    
    invQ = np.linalg.inv(Q)
    
    nbasis = len(initTheta)
    thetaNew = np.zeros( initTheta.shape )
     
    G1 = np.zeros(expVal_tm1_t[1].shape)      
    for t in range(1,len(expVal_t_t)):
        #G1 = G1 +  0.5*expVal_tm1_t[t] + 0.5*expVal_tm1_t[t].T - np.dot( warpMtx,  expVal_t_t[t-1] )
        G1 = G1 +  expVal_tm1_t[t].T - np.dot( warpMtx,  expVal_t_t[t-1] )
        #G1 = G1 +  expVal_tm1_t[t] - np.dot( warpMtx,  expVal_t_t[t-1] )
        for b in range(0, nbasis):
            G1 = G1 + np.dot( dWarp_dTheta[b], expVal_t_t[t-1] )*centerTheta[b]
    G1 = np.dot(invQ, G1)

    G2 = [] 
    for b in range (0,nbasis):
        G2.append(np.zeros(expVal_t_t[1].shape))
        for t in range(1,len(expVal_t_t)): 
            G2[b] = G2[b] + np.dot( dWarp_dTheta[b], expVal_t_t[t-1] )
        G2[b] = np.dot(invQ, G2[b])

    D1 = np.zeros(initTheta.shape)
    for b1 in range(0, nbasis):         
        for p in range(0,dWarp_dTheta[b1].shape[0]):
            for q in range(0,dWarp_dTheta[b1].shape[1]):
                D1[b1] = D1[b1] + G1[p,q]*dWarp_dTheta[b1][p,q]

    D2 = np.zeros([nbasis, nbasis])
    for b1 in range(0, nbasis):
        for b2 in range(0, nbasis):
            for p in range(0,dWarp_dTheta[b1].shape[0]):
                for q in range(0,dWarp_dTheta[b1].shape[1]):
                    D2[b1,b2] = D2[b1,b2] + G2[b2][p,q]*dWarp_dTheta[b1][p,q]


    thetaNew = np.dot(np.linalg.inv(D2), D1)
    
    
    
    secondDeriv = np.zeros((nbasis, nbasis))
    for b in range(0, nbasis):
        thetaNew_tmp = copy.deepcopy(thetaNew)
        thetaNew_tmp[b] = 1.0
        secondDeriv[:,b] = D1 - np.dot(D2, thetaNew_tmp)
    eigvals,_ = np.linalg.eig(secondDeriv)   
    if all(eigvals>0):
        print('local min')
    elif all(eigvals<0):
        print('local max')
    elif any(eigvals==0.0):
        print('inconclusive')
    else:
        print('saddle point: ' + str(np.sum(eigvals<0)) + ' negative eigs of ' + str(len(eigvals)))
            
    
    return (thetaNew, secondDeriv, D1, D2)


def negloglikelihood(theta, mu, Lambda, obs_List, Upsilon, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method, measurement, interiorPriors, mask=[]):
    
    warpMtx = calcWarpMtx(mu, theta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method=method)   
    
    A = warpMtx
    B = np.zeros(mu.imvec.shape)
    Q = Upsilon
 
    loglike, z_t_tm1, P_t_tm1, z_t_t, P_t_t = forwardUpdates(mu, Lambda, obs_List, A, B, Q, measurement=measurement, interiorPriors=interiorPriors, mask=mask)
    
    return -loglike[2]
        
def expnegloglikelihood_full(theta, expectation_theta, mu, Lambda, obs_List, Q, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method, measurement, interiorPriors, numLinIters, apxImgs):
        
    expVal_t, expVal_t_t, expVal_tm1_t, _ = computeSuffStatistics(mu, Lambda, obs_List, Q, expectation_theta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method=method, measurement=measurement, interiorPriors=interiorPriors, numLinIters=numLinIters, apxImgs=apxImgs)
    neg_expll = expnegloglikelihood(theta, expVal_t, expVal_t_t, expVal_tm1_t, mu, Lambda, obs_List, Q, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method)
    print(neg_expll)

    
def expnegloglikelihood(theta, expVal_t, expVal_t_t, expVal_tm1_t, mu, Lambda, obs_List, Upsilon, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method):
    
    #if interiorPriors:
    #TODO:    print 'WARNING: not sure if this works with interior priors because of the derivation of the E[xMx] terms may be different'

    warpMtx = calcWarpMtx(mu[0], theta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method=method)
    A = warpMtx
    B = np.zeros(mu[0].imvec.shape)
    Q = Upsilon
    invQ = np.linalg.inv(Q)

    value = 0.0
    for t in range(1, len(expVal_t)):
        x_t   = np.array([expVal_t[t].imvec]).T
        x_tm1 = np.array([expVal_t[t-1].imvec]).T

        P_tm1_t = expVal_tm1_t[t] - np.dot(x_tm1, x_t.T)
        P_tm1_tm1 = expVal_t_t[t-1] - np.dot(x_tm1, x_tm1.T)
        
        term1 = exp_xtm1_M_xt(P_tm1_t.T, x_t, x_tm1, np.dot(invQ, A) )
        term2 = exp_xtm1_M_xt(P_tm1_t, x_tm1, x_t, np.dot(A.T, invQ) )
        term3 = exp_xtm1_M_xt(P_tm1_tm1, x_tm1, x_tm1, np.dot(A.T, np.dot(invQ, A) ) )
        term4 = np.dot(B.T, np.dot(invQ, np.dot(A, x_tm1))) 

        value = value - 0.5*( -term1 - term2 + term3 + term4 + term4.T  )
        
    value = -value
    return value
    
def exp_xtm1_M_xt(P, z1, z2, M):
    value = np.trace( np.dot(P, M.T) ) + np.dot(z1.T, np.dot(M, z2 ) )
    return value

    

def deriv_expnegloglikelihood_full(theta, expectation_theta, mu, Lambda, obs_List, Q, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method, measurement, interiorPriors, numLinIters, apxImgs):
    
    expVal_t, expVal_t_t, expVal_tm1_t, _ = computeSuffStatistics(mu, Lambda, obs_List, Q, expectation_theta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method=method, measurement=measurement, interiorPriors=interiorPriors, numLinIters=numLinIters, apxImgs=apxImgs)
    return deriv_expnegloglikelihood(theta, expVal_t, expVal_t_t, expVal_tm1_t, mu, Lambda, obs_List, Q, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method)

    
def deriv_expnegloglikelihood(theta, expVal_t, expVal_t_t, expVal_tm1_t, mu, Lambda, obs_List, Q, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method):

    if method == 'phase':
        dWarp_dTheta = calc_dWarp_dTheta(mu[0], theta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method=method)
    else:
        print('WARNING: WE ONLY HANDLE PHASE WARP MINIMIZATION RIGHT NOW')
    
    warpMtx = calcWarpMtx(mu[0], theta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method=method)
    
    invQ = np.linalg.inv(Q)    
    M1 = np.zeros(expVal_tm1_t[1].shape)
    for t in range(1,len(expVal_t_t)):
        #M1 = M1 + 0.5*expVal_tm1_t[t] + 0.5*expVal_tm1_t[t].T - np.dot( warpMtx,  expVal_t_t[t-1] ) 
        M1 = M1 + expVal_tm1_t[t].T - np.dot( warpMtx,  expVal_t_t[t-1] ) 
    M1 = np.dot( invQ , M1)
      
    deriv = np.zeros( initTheta.shape )
    for b in range(0,len(initTheta)):        
        for p in range(0,dWarp_dTheta[b].shape[0]):
            for q in range(0,dWarp_dTheta[b].shape[1]):
                deriv[b] = deriv[b] + M1[p,q]*dWarp_dTheta[b][p,q]    

    # the derivative computed is for the ll but we want the derivative of the neg ll
    deriv = -deriv
    return deriv
    
    
def maximizeBrightness(expVal_t_t, expVal_tm1_t, dummy_im, Q):
        
    dWarp_dTheta = np.eye(dummy_im.xdim*dummy_im.ydim) 
    invQ = np.linalg.inv(Q)
    
    M1 = np.zeros(expVal_tm1_t[1].shape)
    M2 = np.zeros(expVal_t_t[1].shape)
    
    top = 0.0
    bottom = 0.0
        
    for t in range(1,len(expVal_t_t)):
        #M1 = M1 + 0.5*expVal_tm1_t[t] + 0.5*expVal_tm1_t[t].T
        M1 = M1 + expVal_tm1_t[t].T
        M2 = M2 + expVal_t_t[t-1] 
    M1 = np.dot(invQ, M1)
    M2 = np.dot(invQ, M2)
    
    for p in range(0,dWarp_dTheta.shape[0]):
        for q in range(0,dWarp_dTheta.shape[1]):
            top = top + M1[p,q]*dWarp_dTheta[p,q]
            bottom = bottom + M2[p,q]*dWarp_dTheta[p,q]

    thetaNew = top/bottom
        
    return thetaNew
    
    
    
def evaluateGaussianDist_log(y, x, Sigma):
    
    n = len(x)
    if len(x) != np.prod(x.shape):
        raise AssertionError()
    
    diff = x - y
    (sign, logdet) = np.linalg.slogdet(Sigma)
    expval_log = - (n/2.0)*np.log( 2.0*np.pi ) - 0.5*(sign*logdet) -  0.5*np.dot( diff.T, np.dot( np.linalg.inv(Sigma), diff  ) ) 
    
    return expval_log
    
def evaluateGaussianDist(y, x, Sigma):
    
    expval_log = evaluateGaussianDist_log(y, x, Sigma)
    expval = np.exp( expval_log )
    return expval
        
def prodGaussiansLem1(m1, S1, m2, S2):
    
    K = np.linalg.inv(S1 + S2)
    
    covariance = np.dot( S1, np.dot( K, S2 ) )
    mean = np.dot(S1, np.dot(K, m2) ) + np.dot(S2, np.dot(K, m1) ) 
    
    return (mean, covariance)

def prodGaussiansLem2(A, Sigma, y, mu, Q):
    
    K1 = np.linalg.inv( Sigma + np.dot(A, np.dot(Q, np.transpose(A))) )
    K2 =  np.dot( Q,  np.dot( A.T, K1 ) )  
    
    covariance = Q - np.dot( K2, np.dot( A, Q ) )  
    mean = mu + np.dot( K2, y - np.dot(A, mu) )
    
    return (mean, covariance)


def getMeasurementTerms(obs, im, measurement={'vis': 1}, tot_flux=None, mask=[], normalize=False):
    if not np.sum(mask)==len(mask):
        raise ValueError('The code doenst currently work with a mask!')

    #initilize the concatenated data terms
    measdiff_all = []
    ideal_all = []
    F_all = []
    Cov_all = []
    data_all = []

    count = 0
    # loop through data products we want to constrain
    for dname in list(measurement.keys()):

        # ignore data terms that 0 weight
        if np.allclose(measurement[dname],0.0):
            continue

        # check to see if you have data in the current obs
        try:
            if dname=='flux':
                if tot_flux == None:
                    error('Error: if you are using a flux constraint you must specify a total flux (via the lightcurve)')
                data = np.array([tot_flux])
                sigma = np.array([1])
            else:
                data, sigma, A = chisqdata(obs, im, mask, dtype=dname, ttype='direct')
            count = count + 1
        except:
            continue

        #compute the derivative matrix and the ideal measurements if im was the true image
        if dname == 'vis':
            F = A
            ideal = np.dot(A,im.imvec)
        elif dname == 'bs':
            F = grad_bs(im.imvec, A)
            ideal =  bs(im.imvec,A)
        elif dname == 'cphase':
            F = grad_cphase(im.imvec, A)
            ideal = cphase(im.imvec,A)
        elif dname == 'amp':
            F = grad_amp(im.imvec, A)
            ideal = amp(im.imvec,A)
        elif dname == 'logcamp':
            F = grad_logcamp(im.imvec, A)
            ideal = logcamp(im.imvec,A)
        elif dname == 'flux':
            F = grad_flux(im.imvec)
            ideal = flux(im.imvec)

        #turn complex matrices to real
        if not np.allclose(data.imag,0):
            F = realimagStack(F)
            data = realimagStack(data)
            ideal = realimagStack(ideal)
            sigma = np.concatenate((sigma,sigma), axis=0)

        # change the error bars based upon which elements we want to constrain more
        weight = measurement[dname]
        if normalize:
            sigma = sigma / np.sqrt(np.sum(sigma ** 2))
        sigma  = sigma / np.sqrt(weight)
        Cov = np.diag(sigma ** 2)

        data_all = np.concatenate((data_all, data.reshape(-1)), axis=0).reshape(-1)
        ideal_all = np.concatenate((ideal_all, ideal.reshape(-1)), axis=0).reshape(-1)
        measdiff_all = np.concatenate(
            (measdiff_all, data.reshape(-1) + np.dot(F, im.imvec[mask]) - ideal.reshape(-1)), axis=0)
        F_all = np.concatenate((F_all, F), axis=0) if len(F_all) else F
        Cov_all = scipy.linalg.block_diag(Cov_all, Cov) if len(Cov_all) else Cov

    if len(data_all):
        return (measdiff_all, ideal_all, F_all, Cov_all, True)
    else:
        return (-1, -1, -1, -1, False)

def bs(imvec, Amatrices):
    """the bispectrum"""
    out = np.dot(Amatrices[0],imvec) * np.dot(Amatrices[1],imvec) * np.dot(Amatrices[2],imvec)
    return out

def grad_bs(imvec, Amatrices):
    """The gradient of the bispectrum"""
    pt1 = np.dot(Amatrices[1],imvec) * np.dot(Amatrices[2],imvec)
    pt2 = np.dot(Amatrices[0],imvec) * np.dot(Amatrices[2],imvec)
    pt3 = np.dot(Amatrices[0],imvec) * np.dot(Amatrices[1],imvec)
    out = pt1[:,None] * Amatrices[0] + pt2[:,None] * Amatrices[1] + pt3[:,None] * Amatrices[2]
    return out

def flux(imvec):
    return np.sum(imvec)

def grad_flux(imvec):
    return np.ones((1, len(imvec)))

def cphase(imvec, Amatrices):
    """the closure phase"""
    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    clphase_samples = np.angle(i1 * i2 * i3)
    out = np.exp(1j * clphase_samples)
    return out

def grad_cphase(imvec, Amatrices):
    """The gradient of the closure phase"""
    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    clphase_samples = np.angle(i1 * i2 * i3)
    pt1  = 1.0/i1
    pt2  = 1.0/i2
    pt3  = 1.0/i3
    dphi  = (pt1[:,None]*Amatrices[0]) + (pt2[:,None] * Amatrices[1]) + \
            (pt3[:,None]* Amatrices[2])
    out = 1j * np.imag(dphi) * np.exp(1j * clphase_samples[:,None])
    return out

def amp(imvec, A):
    """the amplitude"""
    i1 = np.dot(A, imvec)
    out = np.abs(i1)
    return out

def grad_amp(imvec, A):
    """The gradient of the amplitude """
    i1 = np.dot(A, imvec)
    pp = np.abs(i1) / i1
    out = np.real(pp[:,None] * A)
    return out

def camp(imvec, Amatrices):
    """the closure amplitude"""
    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    i4 = np.dot(Amatrices[3], imvec)
    out = np.abs((i1 * i2)/(i3 * i4))
    return out


def grad_camp(imvec, Amatrices):
    """The gradient of the closure amplitude """
    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    i4 = np.dot(Amatrices[3], imvec)
    clamp_samples = np.abs((i1 * i2)/(i3 * i4))

    pt1 =  clamp_samples/i1
    pt2 =  clamp_samples/i2
    pt3 = -clamp_samples/i3
    pt4 = -clamp_samples/i4
    out = (pt1[:,None]*Amatrices[0]) + (pt2[:,None]*Amatrices[1]) + (pt3[:,None]*Amatrices[2]) + (pt4[:,None]*Amatrices[3])
    return np.real(out)


def logcamp(imvec, Amatrices):
    """The Log closure amplitude """

    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    i4 = np.dot(Amatrices[3], imvec)
    out = np.log(np.abs(i1)) + np.log(np.abs(i2)) - np.log(np.abs(i3)) - np.log(np.abs(i4))
    return out

def grad_logcamp(imvec, Amatrices):
    """The gradient of the Log closure amplitude """

    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    i4 = np.dot(Amatrices[3], imvec)

    pt1 = 1/i1
    pt2 = 1/i2
    pt3 = -1/i3
    pt4 = -1/i4
    out = np.real(pt1[:,None] * Amatrices[0] + pt2[:,None] * Amatrices[1] + \
                  pt3[:,None] *  Amatrices[2] + pt4[:,None] * Amatrices[3])
    return out


def mergeObs(obs_List):
    
    obs = obs_List[0].copy()
    data = obs.data
    for t in range(1,len(obs_List)):
        data = np.concatenate((data, obs_List[t].data))
    obs.data = data
    return obs
    

def splitObs(obs):
    """Split single observation into multiple observation files, one per scan
    """

    print("Splitting Observation File into " + str(len(obs.tlist())) + " scans")

    #Note that the tarr of the output includes all sites, even those that don't participate in the scan
    obs_List = [ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, tdata, obs.tarr, source=obs.source,
                         mjd=obs.mjd, ampcal=obs.ampcal, phasecal=obs.phasecal) 
                 for tdata in obs.tlist() 
                ]
    return obs_List
    

def movie(im_List, out='movie.mp4', fps=10, dpi=120):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()
    frame = im_List[0].imvec #read_auto(filelist[len(filelist)/2])
    fov = im_List[0].psize*im_List[0].xdim
    extent = fov * np.array((1,-1,-1,1)) / 2.
    maxi = np.max(frame)
    im = plt.imshow( np.reshape(frame,[im_List[0].xdim, im_List[0].xdim]) , cmap='hot', extent=extent) #inferno
    plt.colorbar()
    im.set_clim([0,maxi])
    fig.set_size_inches([5,5])
    plt.tight_layout()

    def update_img(n):
        sys.stdout.write('\rprocessing image %i of %i ...' % (n,len(im_List)) )
        sys.stdout.flush()
        im.set_data(np.reshape(im_List[n].imvec, [im_List[n].xdim, im_List[n].xdim]) )
        return im

    ani = animation.FuncAnimation(fig,update_img,len(im_List),interval=1e3/fps)
    writer = animation.writers['ffmpeg'](fps=max(20, fps), bitrate=1e6)
    ani.save(out,writer=writer,dpi=dpi)

    
def dirtyImage(im, obs_List, init_x=[], init_y=[], flowbasis_x=[], flowbasis_y=[], initTheta=[]):
    
    if len(initTheta)==0:
        init_x, init_y, flowbasis_x, flowbasis_y, initTheta = affineMotionBasis(im)

    im_List = []; 
    for t in range(0,len(obs_List)):   
        im_List.append(im.copy())
    
    for t in range(0,len(obs_List)):
        A = genPhaseShiftMtx_obs(obs_List[t],init_x, init_y, flowbasis_x, flowbasis_y, initTheta, im.psize, pulse=ehtim.observing.pulses.deltaPulse2D)
        #im_List[t].imvec = np.real( np.dot( np.linalg.inv( np.dot(np.transpose(A),A) ),  np.dot( np.transpose(A), obs_List[t].data['vis'] ) ) )
        im_List[t].imvec = np.real( np.dot( np.transpose(np.conj(A) ), obs_List[t].data['vis']) )
        
    return im_List

def weinerFiltering(meanImg, covImg, obs_List, mask=[]):

    if type(obs_List) != list:
        obs_List = [obs_List]
    
    im_List = []
    cov_List = []
    exp_tm1_t = []

    for t in range(0,len(obs_List)):   
        im_List.append(meanImg.copy())
        cov_List.append(np.zeros(covImg.shape)) 
        exp_tm1_t.append(np.zeros(covImg.shape)) 
    
    for t in range(0,len(obs_List)):
        
        meas, idealmeas, A, measCov, valid = getMeasurementTerms(obs_List[t], meanImg, measurement={'vis':1}, mask=mask)
        
        if valid==False: 
            im_List[t] = meanImg.copy()
            cov_List[t] = copy.deepcopy(covImg)
        else:
            im_List[t].imvec, cov_List[t] = newDensity(meanImg.imvec, covImg, A, meas, idealmeas, measCov)   
        
        if t>0:
            exp_tm1_t[t] = np.dot( np.array([im_List[t-1].imvec]).T , np.array([im_List[t].imvec]) )
        
    return (im_List, cov_List, exp_tm1_t)

    

def newDensity(X, covX, A, Y, idealY, covY):

    measresidual = Y - idealY
    residualcov = np.dot(np.dot(A , covX ), np.transpose(A)) + covY
    G = np.dot( covX, np.dot( np.transpose(A) , np.linalg.inv ( residualcov ) ) ) 
    Xnew = X + np.dot( G, measresidual )
    covXnew = covX - np.dot(G, np.dot( A, covX ) )
        
    return (Xnew, covXnew)
    
    
def newDensity_linearize(X, covX, A, Y, idealY, covY, Xlin):
    
    measresidual = Y - idealY + np.dot(A,Xlin) - np.dot(A,X)
    residualcov = np.dot(np.dot(A , covX ), np.transpose(A)) + covY
    G = np.dot( covX, np.dot( np.transpose(A) , np.linalg.inv ( residualcov ) ) ) 
    Xnew = X + np.dot( G, measresidual )
    covXnew = covX - np.dot(G, np.dot( A, covX ) )
        
    return (Xnew, covXnew)
    

def newDensity3(X0, covX0, X1, covX1, X2, Y, idealY_X2, A_X2, covY):

    invCovX0 = np.linalg.inv(covX0)
    invCovX1 = np.linalg.inv(covX1)
    invCovY  = np.linalg.inv(covY)
    At_CovY = np.dot( np.transpose(A_X2), invCovY )
    At_CovY_A = np.dot( At_CovY , A_X2 )
    
    covXnew = np.linalg.inv( At_CovY_A + invCovX0 + invCovX1 )
    Xnew = np.dot( covXnew , ( np.dot(At_CovY, Y - idealY_X2) + np.dot(At_CovY_A, X2) + np.dot(invCovX0,X0) + np.dot(invCovX1, X1) ) )
        
    return (Xnew, covXnew)
    

def newDensity2(X, covX, A, Y, covY):

    measresidual = Y - np.dot( A, X )
    residualcov = np.dot(np.dot(A , covX ), np.transpose(A)) + covY
    G = np.dot( covX, np.dot( np.transpose(A) , np.linalg.inv ( residualcov ) ) ) 
    Xnew = X + np.dot( G, measresidual )
    covXnew = covX - np.dot(G, np.dot( A, covX ) )
        
    return (Xnew, covXnew)



def gaussImgCovariance_2(im, powerDropoff=1.0, frac=1.): 
    
    eps = 0.001  
    
    init_x, init_y, flowbasis_x, flowbasis_y, initTheta = affineMotionBasis(im)
    ufull, vfull = genFreqComp(im)
    shiftMtx = genPhaseShiftMtx(ufull, vfull, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, im.psize, im.pulse) 
    uvdist = np.reshape( np.sqrt(ufull**2 + vfull**2), (ufull.shape[0]) ) + eps
    uvdist = uvdist / np.min(uvdist)
    uvdist[0] = 'Inf'
    
    #shiftMtx = np.dot(shiftMtx, np.diag(im.imvec) )
    shiftMtx_exp = realimagStack(shiftMtx)    
    uvdist_exp = np.concatenate( (uvdist, uvdist), axis=0)
    
    imCov = np.dot( np.transpose(shiftMtx_exp) , np.dot( np.diag( 1/(uvdist_exp**powerDropoff) ), shiftMtx_exp ) )
    imCov = frac**2 * np.dot( np.diag(im.imvec).T, np.dot(imCov/imCov[0,0], np.diag(im.imvec) ) ); 
    return imCov
    
def gaussImgCovariance(im, pixelStdev=1.0, powerDropoff=1.0, filter='none', kernsig=3.0): 
    
    eps = 0.001  
    
    init_x, init_y, flowbasis_x, flowbasis_y, initTheta = affineMotionBasis(im)
    ufull, vfull = genFreqComp(im)
    shiftMtx = genPhaseShiftMtx(ufull, vfull, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, im.psize, im.pulse) 
    uvdist = np.reshape( np.sqrt(ufull**2 + vfull**2), (ufull.shape[0]) ) + eps
    uvdist = uvdist / np.min(uvdist)
    uvdist[0] = 'Inf'
    
    if filter == 'hamming':
        hammingwindow = np.dot(np.reshape(np.hamming(im.xdim), (im.xdim,1)), np.reshape(np.hamming(im.ydim) , (1, im.ydim)) )
        shiftMtx = np.dot(shiftMtx, np.diag(np.reshape(hammingwindow, (im.xdim*im.ydim))) ) 
    if filter == 'gaussian':
        gausswindow = gkern(kernlen=im.xdim, nsig=kernsig)
        shiftMtx = np.dot(shiftMtx, np.diag(np.reshape(gausswindow, (im.xdim*im.ydim))) )
    
    shiftMtx_exp = realimagStack(shiftMtx)    
    uvdist_exp = np.concatenate( (uvdist, uvdist), axis=0)
    
    imCov = np.dot( np.transpose(shiftMtx_exp) , np.dot( np.diag( 1/(uvdist_exp**powerDropoff) ), shiftMtx_exp ) )
    imCov = pixelStdev**2 * (imCov/imCov[0,0]); 
    return imCov

###################################### BASIS ########################################

def affineMotionBasis(im):
    
     xlist = np.arange(0,-im.xdim,-1)*im.psize + (im.psize*im.xdim)/2.0 - im.psize/2.0
     ylist = np.arange(0,-im.ydim,-1)*im.psize + (im.psize*im.ydim)/2.0 - im.psize/2.0
 
     init_x = np.array([[ [0] for i in xlist] for j in ylist])
     init_y = np.array([[ [0] for i in xlist] for j in ylist])  
    
     flowbasis_x = np.array([[ [i, j ,im.psize, 0, 0, 0] for i in xlist] for j in ylist])
     flowbasis_y = np.array([[ [0, 0, 0, i, j, im.psize] for i in xlist] for j in ylist])
     initTheta = np.array([1, 0, 0, 0, 1, 0])
     
     return (init_x, init_y, flowbasis_x, flowbasis_y, initTheta)
     
def affineMotionBasis_noTranslation(im):
    
     xlist = np.arange(0,-im.xdim,-1)*im.psize + (im.psize*im.xdim)/2.0 - im.psize/2.0
     ylist = np.arange(0,-im.ydim,-1)*im.psize + (im.psize*im.ydim)/2.0 - im.psize/2.0

     init_x = np.array([[ [0] for i in xlist] for j in ylist])
     init_y = np.array([[ [0] for i in xlist] for j in ylist])     

     flowbasis_x = np.array([[ [i, j, 0, 0] for i in xlist] for j in ylist])
     flowbasis_y = np.array([[ [0, 0, i, j] for i in xlist] for j in ylist])
     initTheta = np.array([1, 0, 0, 1])
     
     return (init_x, init_y, flowbasis_x, flowbasis_y, initTheta)
     
def translationBasis(im):
    
     xlist = np.arange(0,-im.xdim,-1)*im.psize + (im.psize*im.xdim)/2.0 - im.psize/2.0
     ylist = np.arange(0,-im.ydim,-1)*im.psize + (im.psize*im.ydim)/2.0 - im.psize/2.0
     
     init_x = np.array([[ [i] for i in xlist] for j in ylist])
     init_y = np.array([[ [j] for i in xlist] for j in ylist])

     flowbasis_x = np.array([[ [im.psize, 0.0] for i in xlist] for j in ylist])
     flowbasis_y = np.array([[ [0.0, im.psize] for i in xlist] for j in ylist])
     initTheta = np.array([0.0, 0.0])
 
     return (init_x, init_y, flowbasis_x, flowbasis_y, initTheta)
     
def xTranslationBasis(im):
    
     xlist = np.arange(0,-im.xdim,-1)*im.psize + (im.psize*im.xdim)/2.0 - im.psize/2.0
     ylist = np.arange(0,-im.ydim,-1)*im.psize + (im.psize*im.ydim)/2.0 - im.psize/2.0
     
     init_x = np.array([[ [i] for i in xlist] for j in ylist])
     init_y = np.array([[ [j] for i in xlist] for j in ylist])

     flowbasis_x = np.array([[ [im.psize] for i in xlist] for j in ylist])
     flowbasis_y = np.array([[ [0.0] for i in xlist] for j in ylist])
     initTheta = np.array([0.0])
 
     return (init_x, init_y, flowbasis_x, flowbasis_y, initTheta)
    
def dftMotionBasis(im):
    
    DFTBASIS_THRESH = 0.1
    print('WARNING SOMETHING ISNT RIGHT WITH THE DFT GEN')
    
    init_x, init_y, flowbasis_x, flowbasis_y, initTheta = affineMotionBasis(im)
    ufull, vfull = genFreqComp(im)
    shiftMtx = genPhaseShiftMtx(ufull, vfull, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, im.psize, im.pulse) 
    uvdist = np.reshape( np.sqrt(ufull**2 + vfull**2), (ufull.shape[0]) )
    uvdist_norm = uvdist / np.max(uvdist)

    halfbasis = shiftMtx[uvdist_norm < DFTBASIS_THRESH,:]
    fullbasis = np.concatenate( ( np.real( halfbasis[1:,:] ) , np.imag( halfbasis[1:,:] )  ), axis = 0)
    fullbasis = np.reshape(fullbasis.T, [im.xdim, im.ydim, fullbasis.shape[0]]) * im.psize
    
    flowbasis_x = np.concatenate( (fullbasis, np.zeros(fullbasis.shape)), axis=2 ) 
    flowbasis_y = np.concatenate( (np.zeros(fullbasis.shape), fullbasis), axis=2 )
    
    xlist = np.arange(0,-im.xdim,-1)*im.psize + (im.psize*im.xdim)/2.0 - im.psize/2.0
    ylist = np.arange(0,-im.ydim,-1)*im.psize + (im.psize*im.ydim)/2.0 - im.psize/2.0
     
    init_x = np.array([[ [i] for i in xlist] for j in ylist])
    init_y = np.array([[ [j] for i in xlist] for j in ylist])
    
    initTheta = np.zeros(flowbasis_x.shape[2])
    
    return (init_x, init_y, flowbasis_x, flowbasis_y, initTheta)
    
    
def applyMotionBasis(init_x, init_y, flowbasis_x, flowbasis_y, theta):
    
    imsize = flowbasis_x.shape[0:2]
    nbasis = theta.shape[0]
    npixels = np.prod(imsize)
    
    flow_x = init_x[:,:,0] + np.reshape( np.dot( np.reshape(flowbasis_x, (npixels, nbasis), order ='F'), theta ), imsize, order='F')
    flow_y = init_y[:,:,0] + np.reshape( np.dot( np.reshape(flowbasis_y, (npixels, nbasis), order ='F'), theta ), imsize, order='F')   

    return (flow_x, flow_y)
    

###################################### FULL WARPING ########################################

def applyWarp(im, theta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method='phase'):
    
    if method=='phase':
        outim = applyPhaseWarp(im, theta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta)
    else:
        outim = applyImageWarp(im, theta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta)
    return outim
    

def calcWarpMtx(im, theta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method='phase', normalize=False):
    
    npixels = im.xdim*im.ydim
    
    if method=='phase':
        ufull, vfull = genFreqComp(im)
        shiftMtx0 = genPhaseShiftMtx(ufull, vfull, init_x, init_y, flowbasis_x, flowbasis_y, theta, im.psize, im.pulse) 
        shiftMtx1 = genPhaseShiftMtx(ufull, vfull, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, im.psize, im.pulse) 
        
        #outMtx = np.real( np.dot(np.transpose(np.conj(shiftMtx1)), shiftMtx0 ) / (npixels) )
        outMtx = np.real( np.dot( np.linalg.inv(shiftMtx1), shiftMtx0 ) )
        
    elif method=='img':
        probeim = im.copy()
        outMtx = np.zeros((npixels, npixels))
        for i in range(0,npixels):
            probeim.imvec = np.zeros(probeim.imvec.shape)
            probeim.imvec[i] = 1.0
            outprobeim = applyImageWarp(probeim, theta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, pad=True)
            outMtx[:,i] = outprobeim.imvec 

        outMtx = np.nan_to_num(outMtx)
    
    if normalize:
        for i in range(0,npixels):
            outMtx[i,:] = outMtx[i,:]/np.sum(outMtx[i,:])
        
    return outMtx

def applyImageWarp(im, theta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, pad=False):
    
    flow_x_orig, flow_y_orig = applyMotionBasis(init_x, init_y, flowbasis_x, flowbasis_y, initTheta)
    flow_x_new, flow_y_new = applyMotionBasis(init_x, init_y, flowbasis_x, flowbasis_y, theta)
    
    from_pts = np.concatenate( ( reshapeFlowbasis(flow_x_new), reshapeFlowbasis(flow_y_new) ), axis=1)
    to_pts = np.concatenate( ( reshapeFlowbasis(flow_x_orig), reshapeFlowbasis(flow_y_orig) ), axis=1)
    im_pts = im.imvec
    
    # add padding of 0's around the warped image so that the griddata function works properly
    if pad:
        # add padding on the x axis
        for i in range(0,im.xdim):
            
            vec_x = flow_x_new[1,i] - flow_x_new[0,i]
            vec_y = flow_y_new[1,i] - flow_y_new[0,i]
            from_pts = np.row_stack( (from_pts, np.array( [ flow_x_new[0,i]-vec_x, flow_y_new[0,i]-vec_y ] ) ) )
            im_pts = np.concatenate( ( im_pts, np.array([0.0]) ), axis = 0 )
            
            vec_x = flow_x_new[im.ydim-2,i] - flow_x_new[im.ydim-1,i]
            vec_y = flow_y_new[im.ydim-2,i] - flow_y_new[im.ydim-1,i]
            from_pts = np.row_stack( (from_pts, np.array( [ flow_x_new[im.ydim-1,i]-vec_x, flow_y_new[im.ydim-1,i]-vec_y ] ) ) )
            im_pts = np.concatenate( ( im_pts, np.array([0.0]) ), axis = 0 )
        # add padding on the y axis
        for i in range(0,im.ydim):
            
            vec_x = flow_x_new[i,1] - flow_x_new[i,0]
            vec_y = flow_y_new[i,1] - flow_y_new[i,0]
            from_pts = np.row_stack( (from_pts, np.array( [ flow_x_new[i,0]-vec_x, flow_y_new[i,0]-vec_y ] ) ) )
            im_pts = np.concatenate( ( im_pts, np.array([0.0]) ), axis = 0 )
            
            vec_x = flow_x_new[i,im.xdim-2] - flow_x_new[i,im.xdim-1]
            vec_y = flow_y_new[i,im.xdim-2] - flow_y_new[i,im.xdim-1]
            from_pts = np.row_stack( (from_pts, np.array( [ flow_x_new[i,im.xdim-1]-vec_x, flow_y_new[i,im.xdim-1]-vec_y ] ) ) )
            im_pts = np.concatenate( ( im_pts, np.array([0.0]) ), axis = 0 )
            
    #npixels = flowbasis_x.shape[0]*flowbasis_x.shape[1]
    out = scipy.interpolate.griddata( from_pts , im_pts, to_pts , method='linear', fill_value=0.0 )
    #out = scipy.interpolate.griddata( np.concatenate( ( np.reshape(flow_x_new, (npixels, -1)), np.reshape(flow_y_new, (npixels, -1)) ), axis=1) , im.imvec, np.concatenate( ( np.reshape(flow_x_orig, (npixels, -1)), np.reshape(flow_y_orig, (npixels, -1)) ), axis=1), method='linear', fill_value=0.0 )    
    outim = image.Image(np.reshape(out, (im.ydim, im.xdim)), im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse)
    return outim

  
def applyPhaseWarp(im, theta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta):

    ufull, vfull = genFreqComp(im)
    
    shiftMtx0 = genPhaseShiftMtx(ufull, vfull, init_x, init_y, flowbasis_x, flowbasis_y, theta, im.psize, im.pulse) 
    shiftMtx1 = genPhaseShiftMtx(ufull, vfull, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, im.psize, im.pulse) 
     
    #out = np.real( np.dot(np.transpose(np.conj(shiftMtx1)), np.dot(shiftMtx0, im.imvec) ) ) / (im.xdim * im.ydim)
    out = np.real( np.dot( np.linalg.inv(shiftMtx1) , np.dot(shiftMtx0, im.imvec) ) )
    outim = image.Image(np.reshape(out, (im.ydim, im.xdim)), im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse)
    return outim
            
def genPhaseShiftMtx(ulist, vlist, init_x, init_y, flowbasis_x, flowbasis_y, theta, pdim, pulse=ehtim.observing.pulses.deltaPulse2D):
    
    flow_x, flow_y = applyMotionBasis(init_x, init_y, flowbasis_x, flowbasis_y, theta)

    imsize = flow_x.shape
    npixels = np.prod(imsize)
    
    flow_x_vec = np.reshape(flow_x, (npixels))
    flow_y_vec = np.reshape(flow_y, (npixels))

    shiftMtx_y = np.exp( [-1j * 2.0 * np.pi * flow_y_vec * v for v in vlist]  )
    shiftMtx_x = np.exp( [-1j * 2.0 * np.pi * flow_x_vec * u for u in ulist]  )

    uvlist = np.transpose(np.squeeze(np.array([ulist, vlist])))
    uvlist = np.reshape(uvlist, (vlist.shape[0], 2))
    pulseVec = [pulse(2*np.pi*uv[0], 2*np.pi*uv[1], pdim, dom="F") for uv in uvlist ]
    
    shiftMtx = np.dot( np.diag(pulseVec) , np.reshape( np.squeeze(shiftMtx_x * shiftMtx_y), (vlist.shape[0], npixels)  ) )
    return shiftMtx

################################### APPROXIMATE SHIFTING ###########################################

def calc_dWarp_dTheta(im, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method='phase'):
    
    if method == 'phase':
        
        ufull, vfull = genFreqComp(im)
        
        derivShiftMtx_x, derivShiftMtx_y = calcDerivShiftMtx_freq(ufull, vfull, im, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, includeImgFlow=False)

        shiftMtx1 = genPhaseShiftMtx(ufull, vfull, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, im.psize, im.pulse)
        invShiftMtx1 = np.linalg.inv(shiftMtx1)
        
        flowbasis = np.concatenate((reshapeFlowbasis(flowbasis_x), reshapeFlowbasis(flowbasis_y)), axis=0)
        
        reshape_flowbasis_x = reshapeFlowbasis(flowbasis_x)
        reshape_flowbasis_y = reshapeFlowbasis(flowbasis_y)
        
        
        dWarp_dTheta = []; 
        for b in range(0, flowbasis.shape[1]):
            K = np.dot( derivShiftMtx_x, np.diag( reshape_flowbasis_x[:,b] ) ) +  np.dot( derivShiftMtx_y, np.diag( reshape_flowbasis_y[:,b] ) )
            dWarp_dTheta.append(  np.real( np.dot( invShiftMtx1 , K ) )  ) 
        

    else: 
        print('WARNING: we do not handle this method yet')
            
    return dWarp_dTheta



def applyAppxWarp(im, theta, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method1='phase', method2='phase'):
    
    centerIm, dImg_dTheta = calAppxWarpTerms(im, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method1=method1, method2=method2)
    out = centerIm.imvec + np.dot(dImg_dTheta, theta - centerTheta)
    outim = image.Image(np.reshape(out, (im.ydim, im.xdim)), im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse)
    return outim
    
def calAppxWarpTerms(im, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method1='phase', method2='phase'):
    
    centerIm = applyWarp(im, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method=method1)
    dImg_dTheta = calc_dImage_dTheta(im, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method=method2)
    return (centerIm, dImg_dTheta)
     
def calc_dImage_dTheta(im, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method='phase'):
    
    if method == 'phase':
        ufull, vfull = genFreqComp(im)
        thetaDerivShiftMtx = calcDerivShiftMtx_freq(ufull, vfull, im, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, includeImgFlow=True)
        shiftMtx1 = genPhaseShiftMtx(ufull, vfull, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, im.psize, im.pulse)
        #dImg_dTheta = np.real( np.dot(   np.transpose(np.conj(shiftMtx1))  , thetaDerivShiftMtx) / (im.xdim * im.ydim) )
        dImg_dTheta = np.real( np.dot(   np.linalg.inv(shiftMtx1)  , thetaDerivShiftMtx)  )
    else:
        dImg_dTheta = calcDerivShiftMtx_image(im, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta)
        
    return dImg_dTheta

def calcDerivWarpMtx_noimg_noshift(im, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method='phase'):
    
    if method == 'phase':
        
        ufull, vfull = genFreqComp(im)
        #npixels = im.xdim*im.ydim

        freqShiftMtx_x, freqShiftMtx_y = calcDerivShiftMtx_freq(ufull, vfull, im, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, includeImgFlow=False)
        shiftMtx1 = genPhaseShiftMtx(ufull, vfull, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, im.psize, im.pulse) 
        
        #derivShiftMtx_y = np.real( np.dot(   np.transpose(np.conj(shiftMtx1))  , freqShiftMtx_y) / npixels )
        #derivShiftMtx_x = np.real( np.dot(   np.transpose(np.conj(shiftMtx1))  , freqShiftMtx_x) / npixels )
        invShiftMtx1 = np.linalg.inv(shiftMtx1)
        derivShiftMtx_y = np.real( np.dot(   invShiftMtx1  , freqShiftMtx_y) )
        derivShiftMtx_x = np.real( np.dot(   invShiftMtx1  , freqShiftMtx_x) )
    else:
        if (centerTheta != initTheta).any():
            raise ValueError('Can only take the optical flow derivative around no shift')
        
        derivShiftMtx_x = -gradMtx(im.xdim, im.ydim, dir='x')/im.psize
        derivShiftMtx_y = -gradMtx(im.xdim, im.ydim, dir='y')/im.psize
    
    return (derivShiftMtx_x, derivShiftMtx_y)


def calcDerivShiftMtx_freq(ulist, vlist, im, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, includeImgFlow=True):
    
    shiftMtx = genPhaseShiftMtx(ulist, vlist, init_x, init_y, flowbasis_x, flowbasis_y, centerTheta, im.psize, im.pulse)

    shiftVec_y = np.array( [-1j * 2.0 * np.pi * v * np.ones(im.xdim*im.ydim) for v in vlist]  )
    shiftVec_x = np.array( [-1j * 2.0 * np.pi * u * np.ones(im.xdim*im.ydim) for u in ulist]  )
    derivShiftMtx_x = shiftVec_x * shiftMtx
    derivShiftMtx_y = shiftVec_y * shiftMtx

    
    if includeImgFlow:
        # TODO: WARNING DOES THIS  HANDLE THE INIT_X INIT_Y
        print('WARNING: is this handling the init x and y??')
        
        flowbasis = np.concatenate((reshapeFlowbasis(flowbasis_x), reshapeFlowbasis(flowbasis_y)), axis=0)
        derivShiftMtx = np.concatenate( ( np.dot(derivShiftMtx_x, np.diag(im.imvec)) , np.dot(derivShiftMtx_y, np.diag(im.imvec)) ) , axis=1)
        thetaDerivShiftMtx = np.dot(derivShiftMtx, flowbasis)
        return thetaDerivShiftMtx
    else: 
        return (derivShiftMtx_x, derivShiftMtx_y)
   
def calcDerivShiftMtx_image(im, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta):
# out = im.imvec + np.dot(gradIm_x, theta - initTheta ) + np.dot(gradIm_y, theta - initTheta )

    centerImg = applyImageWarp(im, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta)

    Gx = -gradMtx(im.xdim, im.ydim, dir='x')/im.psize
    Gy = -gradMtx(im.xdim, im.ydim, dir='y')/im.psize

    gradIm_x = np.diag( np.dot(Gx, centerImg.imvec) )
    gradIm_y = np.diag( np.dot(Gy, centerImg.imvec) )
    
    # TODO: WARNING DOES THIS  HANDLE THE INIT_X INIT_Y
    print('WARNING: is this handling the init x and y??')
    
    flowbasis = np.concatenate((reshapeFlowbasis(flowbasis_x), reshapeFlowbasis(flowbasis_y)), axis=0)
    derivShiftMtx = np.concatenate( (gradIm_x, gradIm_y), axis=1)

    thetaDerivShiftMtx = np.dot(derivShiftMtx, flowbasis)
    return thetaDerivShiftMtx

def calcAppxWarpMtx_image(im, init_x, init_y, flowbasis_x, flowbasis_y, theta, initTheta):

    flow_x_orig, flow_y_orig = applyMotionBasis(init_x, init_y, flowbasis_x, flowbasis_y, initTheta)
    flow_x_new, flow_y_new = applyMotionBasis(init_x, init_y, flowbasis_x, flowbasis_y, theta)
    
    flow_x = flow_x_new - flow_x_orig
    flow_y = flow_y_new - flow_y_orig
    
    Gx = -gradMtx(im.xdim, im.ydim, dir='x')
    Gy = -gradMtx(im.xdim, im.ydim, dir='y')
    
    derivx = np.dot( np.diag( vec(flow_x) ), Gx/im.psize)
    derivy = np.dot( np.diag( vec(flow_y) ), Gy/im.psize)
    
    appxMtx = np.eye(im.xdim*im.ydim) + derivx + derivy
    return appxMtx


###################################### FREQUENCY SPACE ########################################
 
def shiftVisibilities(obs, shiftX, shiftY):
    obs.data['vis'] = obs.data['vis']*np.exp(-1j*2.0*np.pi*( obs.data['u']*shiftX + obs.data['v']*shiftY ))
    return obs

def genAppxShiftMtx(ulist, vlist, npixels, shiftMtx):

    derivShiftMtx_y = np.array( [-1j * 2.0 * np.pi * np.ones((npixels)) * v for v in vlist]  ) * shiftMtx
    derivShiftMtx_x = np.array( [-1j * 2.0 * np.pi * np.ones((npixels)) * u for u in ulist]  ) * shiftMtx
    
    return (derivShiftMtx_x, derivShiftMtx_y)
   
def genFreqComp(im):

    fN2 = int(np.floor(im.xdim/2)) #TODO: !!! THIS DOESNT WORK FOR ODD IMAGE SIZES
    fM2 = int(np.floor(im.ydim/2))
    
    ulist = (np.array([np.concatenate((np.linspace(0, fN2 - 1, fN2), np.linspace(-fN2, -1, fN2)), axis=0)])  / im.xdim ) / im.psize 
    vlist = (np.array([np.concatenate((np.linspace(0, fM2 - 1, fM2), np.linspace(-fM2, -1, fM2)), axis=0)])  / im.ydim ) / im.psize 
    
    ufull, vfull = np.meshgrid(ulist, vlist)   
    
    ufull = np.reshape(ufull, (im.xdim*im.ydim, -1), order='F')
    vfull = np.reshape(vfull, (im.xdim*im.ydim, -1), order='F')
    
    return (ufull, vfull)

def genPhaseShiftMtx_obs(obs, init_x, init_y, flowbasis_x, flowbasis_y, theta, pdim, pulse=ehtim.observing.pulses.deltaPulse2D):
    ulist = obs.unpack('u')['u']
    vlist = obs.unpack('v')['v']
    
    shiftMtx = genPhaseShiftMtx(ulist, vlist, init_x, init_y, flowbasis_x, flowbasis_y, theta, pdim, pulse)
    return shiftMtx
    
  
def calcDerivShiftMtx_obs(obs, im, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y):

    ulist = obs.unpack('u')['u']
    vlist = obs.unpack('v')['v']

    thetaDerivShiftMtx = calcDerivShiftMtx_freq(ulist, vlist, im, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y)
    return thetaDerivShiftMtx

def cmpFreqExtraction_phaseWarp(obs, im_true, im_canonical, theta, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta):
    
    data = obs.unpack(['u','v','vis','sigma'])
    uv = np.hstack((data['u'].reshape(-1,1), data['v'].reshape(-1,1)))
    A = ftmatrix(im_true.psize, im_true.xdim, im_true.ydim, uv, pulse=im_true.pulse)
    
    shiftMtx_true = genPhaseShiftMtx_obs(obs, init_x, init_y, flowbasis_x, flowbasis_y, theta, im_canonical.psize, im_canonical.pulse)
    
    shiftMtx_center = genPhaseShiftMtx_obs(obs, init_x, init_y, flowbasis_x, flowbasis_y, centerTheta, im_canonical.psize, im_canonical.pulse)
    thetaDerivShiftMtx = calcDerivShiftMtx_obs(obs, im_canonical, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y)
        
    centerim, dImg_dTheta = calAppxWarpTerms(im_canonical, centerTheta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, method1='phase', method2='phase')
    
    chiSq_shift = 0.5 * np.mean( np.abs( (obs.data['vis'] - np.dot(shiftMtx_true, im_canonical.imvec ) )/ obs.data['sigma'] )**2 )
    chiSq_appxshift = 0.5 * np.mean( np.abs( (obs.data['vis'] - np.dot(shiftMtx_center, im_canonical.imvec) - np.dot(thetaDerivShiftMtx, theta-centerTheta)  )  / obs.data['sigma'] )**2 )
    chiSq_true =  0.5 * np.mean( np.abs( (obs.data['vis'] - np.dot(A, im_true.imvec) ) / obs.data['sigma'] )**2 )

    return (chiSq_true, chiSq_shift, chiSq_appxshift)


################################# HELPER FUNCTIONS #############################################


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def gradMtx(w, h, dir='x'):
    
    G = np.eye(h*w)
    if dir=='x':
        G = G - np.diag(np.ones((h*w-1)), k=1)
        delrows = (np.linspace(w,h*w,h)-1).astype(int)
        G[delrows,:] = 0
    else:
        G = G  - np.diag(np.ones((h*w-w)), k=h)
        delrows = range(h*w-w,h*w)
        G[delrows,:] = 0
    return G


def realimagStack(mtx):
    stack = np.concatenate( ( np.real(mtx), np.imag(mtx) ), axis=0 ) 
    return stack
    
def reshapeFlowbasis(flowbasis):
    npixels = flowbasis.shape[0] * flowbasis.shape[1]
    #nbasis = flowbasis.shape[2]
    flowbasis = np.reshape(flowbasis, (npixels, -1) ) #, order ='F')
    return flowbasis
    
def vec(x, order='F'):
    x = np.reshape(x, (np.prod(x.shape)), order = order)
    return x

def listconcatenate(*lists):
    new_list = []
    for i in lists:
        new_list.extend(i)
    return new_list
    
def padNewFOV(im, fov_arcseconds):
    
    oldfov = im.psize * im.xdim
    newfov = fov_arcseconds * ehtim.RADPERUAS
    tnpixels = np.ceil(im.xdim * newfov/oldfov).astype('int')
    
    origimg = np.reshape(im.imvec, [im.xdim, im.xdim])
    padimg = np.pad(origimg, ((0,tnpixels-im.xdim), (0,tnpixels-im.xdim)), 'constant')
    padimg = np.roll(padimg, np.floor((tnpixels-im.xdim)/2.).astype('int'), axis=0)
    padimg = np.roll(padimg, np.floor((tnpixels-im.xdim)/2.).astype('int'), axis=1)
    
    return image.Image(padimg.reshape((tnpixels, tnpixels)), im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse)

    
def flipImg(im, flip_lr, flip_ud):
    
    img = np.reshape(im.imvec, [im.xdim, im.xdim])
    if flip_lr:
        img = np.fliplr(img)
    if flip_ud:
        img = np.flipud(img)
    im.imec = img.reshape((im.xdim*im.xdim))
    return image.Image(img, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse)

        
def rotateImg(im, k):
    
    img = np.reshape(im.imvec, [im.xdim, im.xdim])
    img = np.rot90(img, k=k)
    im.imec = img.reshape((im.xdim*im.xdim))
    return image.Image(img, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse)


####################################### MICHAELS STUFF ####################################### 

def plot_im_List(im_List, title_List=[], ipynb=False):

    plt.title("Test", fontsize=20)    
    plt.ion()
    plt.clf()
    
    Prior = im_List[0]

    for i in range(len(im_List)):
        plt.subplot(1, len(im_List), i+1)	
        plt.imshow(im_List[i].imvec.reshape(Prior.ydim,Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian')     
        plt.axis('off')
        xticks = ticks(Prior.xdim, Prior.psize/ehtim.RADPERAS/1e-6)
        yticks = ticks(Prior.ydim, Prior.psize/ehtim.RADPERAS/1e-6)
        plt.xticks(xticks[0], xticks[1])
        plt.yticks(yticks[0], yticks[1])
        if i == 0:
            plt.xlabel('Relative RA ($\mu$as)')
            plt.ylabel('Relative Dec ($\mu$as)')
        else:
            plt.xlabel('')
            plt.ylabel('')
            #plt.title('')
        if len(title_List)==len(im_List):
            plt.title(title_List[i], fontsize = 5)
	

        plt.draw()


def plot_Flow(Im, theta, init_x, init_y, flowbasis_x, flowbasis_y, initTheta, step=4, ipynb=False):
    
    # Get vectors and ratio from current image
    x = np.array([[i for i in range(Im.xdim)] for j in range(Im.ydim)])
    y = np.array([[j for i in range(Im.xdim)] for j in range(Im.ydim)])
    
    flow_x_new, flow_y_new = applyMotionBasis(init_x, init_y, flowbasis_x, flowbasis_y, theta)
    flow_x_orig, flow_y_orig = applyMotionBasis(init_x, init_y, flowbasis_x, flowbasis_y, initTheta)
    
    vx = -(flow_x_new - flow_x_orig)
    vy = -(flow_y_new - flow_y_orig)

    # Create figure and title
    plt.ion()
    plt.clf()

    # Stokes I plot
    plt.subplot(111)
    plt.imshow(Im.imvec.reshape(Im.ydim, Im.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian')
    plt.quiver(x[::step,::step], y[::step,::step], vx[::step,::step], vy[::step,::step],
               headaxislength=3, headwidth=7, headlength=5, minlength=0, minshaft=1,
               width=.005*Im.xdim/30., pivot='mid', color='w', angles='xy')

    xticks = ticks(Im.xdim, Im.psize/ehtim.RADPERAS/1e-6)
    yticks = ticks(Im.ydim, Im.psize/ehtim.RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title('Flow Map')
    #plt.ylim(plt.ylim()[::-1])
    # Display
    plt.draw()
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    

