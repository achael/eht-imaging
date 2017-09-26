# patch_prior.py
#
# Create a "prior" image for vlbi imaging by cleaning the input image.
# Image cleaning is done by breaking the image into patches and cleaning each one
# individually by assigning it a cluster in the input Gaussian mixture model and
# using a weiner filter to denoise each patch.
# These ideas are based on Expected Patch Log Likelihood patch prior work
#
# Code Author: Katie Bouman
# Date: June 1, 2016

from __future__ import division
from builtins import map
from builtins import range

from matplotlib import pyplot as plt
import ehtim.image as image
import scipy.io
import numpy as np

def patchPrior(im, beta, patchPriorFile='naturalPrior.mat', patchSize=8 ):

    # load data
    ldata = scipy.io.loadmat(patchPriorFile)

    # reassign and reshape data
    nmodels = ldata['nmodels'].ravel()
    nmodels = nmodels[0]
    mixweights = ldata['mixweights'].ravel()
    covs = np.array(ldata['covs'])
    means = np.array(ldata['means'])

    # reshape image
    img = np.reshape(im.imvec, (im.ydim, im.xdim) )

    I1, counts = cleanImage(img, beta, nmodels, covs, mixweights, means, patchSize)

    if not all(counts[0][0] == item for item in np.reshape(counts, (-1)) ):
         raise TypeError("The counts are not the same for every pixel in the image")

    I1 = I1/counts[0][0]
    out = image.Image(I1, im.psize, im.ra, im.dec, rf=im.rf, source=im.source, mjd=im.mjd, pulse=im.pulse)

    return (out, counts[0][0])

def cleanImage(img, beta, nmodels, covs, mixweights, means, patchSize=8):

    # pad images with 0's
    validRegion = np.lib.pad( np.ones(img.shape), (patchSize-1, patchSize-1), 'constant', constant_values=(0, 0) )
    cleanIPad = np.lib.pad( img , (patchSize-1, patchSize-1), 'constant', constant_values=(0, 0) )

    # adjust the dynamic range of image to be in the range 0 to 1
    minCleanI = min( np.reshape(cleanIPad, (-1)) )
    cleanIPad = cleanIPad - minCleanI
    maxCleanI = max( np.reshape(cleanIPad, (-1)) )
    cleanIPad = cleanIPad / maxCleanI;

    # extract all overlapping patches from the image
    Z = im2col(np.transpose(cleanIPad), patchSize)

    # clean each patch by weiner filtering
    meanZ =  np.mean(Z,0)
    Z = Z - np.tile( meanZ, [patchSize**2, 1] );
    cleanZ = cleanPatches( Z,patchSize,(beta)**(-0.5), nmodels, covs, mixweights, means);
    cleanZ = cleanZ + np.tile( meanZ, [patchSize**2, 1] );

    # join all patches together
    mm = validRegion.shape[0]
    nn = validRegion.shape[1]
    t = np.reshape(list(range(0,mm*nn,1)), (mm, nn) )
    temp = im2col(t, patchSize)
    I1 = np.transpose( np.bincount( np.array(list(map(int, np.reshape(temp, (-1)) ))), weights=np.reshape(cleanZ, (-1))) )
    counts =  np.transpose( np.bincount( np.array(list(map(int, np.reshape(temp, (-1)) ))), weights=np.reshape(np.ones(cleanZ.shape), (-1))) )

    # normalize and put back in the original scale
    I1 = I1/counts;
    I1 = (I1*maxCleanI) + minCleanI;
    I1 = I1*counts;

    # set all negative entries to 0 (hacky)
    I1[I1<0] = 0;

    # crop out the center valid region
    I1 = np.extract(np.reshape(validRegion, (-1)), I1)
    counts = np.extract(np.reshape(validRegion, (-1)), counts)

    # reshape
    I1 = np.transpose(np.reshape( I1, (img.shape[1], img.shape[0])));
    counts = np.transpose(np.reshape( counts, (img.shape[1], img.shape[0])));

    return I1, counts

def im2col(im, patchSize):

    # extract all overlapping patches from the image
    M,N = im.shape
    col_extent = N - patchSize + 1
    row_extent = M - patchSize + 1
    # Get Starting block indices
    start_idx = np.arange(patchSize)[:,None]*N + np.arange(patchSize)
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)
    # Get all actual indices & index into input array for final output
    Z = np.take (im,start_idx.ravel()[:,None] + offset_idx.ravel())
    return Z


def cleanPatches(Y, patchSize, noiseSD, nmodels, covs, mixweights, means):

    SigmaNoise = noiseSD**2 * np.eye(patchSize**2);

    #remove DC component
    meanY = np.mean(Y,0);
    Y = Y - np.tile(meanY, [Y.shape[0], 1] );

    #calculate assignment probabilities for each mixture component for all patches
    PYZ = np.zeros((nmodels,Y.shape[1]));
    for i in range (0,nmodels):
        PYZ[i,:] = np.log(mixweights[i]) + loggausspdf2(Y, covs[:,:,i] + SigmaNoise);


    #find the most likely component for each patch
    ks = PYZ.argmax(axis = 0)

    # and now perform weiner filtering
    Xhat = np.zeros(Y.shape);
    for i in range (0,nmodels):
        inds = np.array(np.where(ks==i)).ravel()
        Xhat[:,inds] = np.dot( covs[:,:,i], np.dot( np.linalg.inv( covs[:,:,i]+SigmaNoise ), Y[:,inds] ) ) + np.dot( SigmaNoise, np.dot( np.linalg.inv(covs[:,:,i]+SigmaNoise), np.transpose(np.tile( np.transpose(means[:,i]), [inds.shape[0], 1] )) ));


    Xhat = Xhat + np.tile(meanY, [Xhat.shape[0], 1] )
    return Xhat


def loggausspdf2(X, sigma):
#log pdf of Gaussian with zero mena
#Based on code written by Mo Chen (mochen@ie.cuhk.edu.hk). March 2009.
    d = X.shape[0]

    R = np.linalg.cholesky(sigma).T;
    # todo check that sigma is psd

    q = np.sum( ( np.dot( np.linalg.inv(np.transpose(R)) , X ) )**2 , 0);  # quadratic term (M distance)
    c = d*np.log(2*np.pi)+2*np.sum(np.log( np.diagonal(R) ), 0);   # normalization constant
    y = -(c+q)/2.0;

    return y


