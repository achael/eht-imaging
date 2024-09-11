from __future__ import division

import numpy as np
import ehtim.image as image

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *


def linearized_bi(x0, A3, bispec, sigs, nPixels, alpha=100, reg="patch"):

	Alin, blin = computeLinTerms_bi(x0, A3, bispec, sigs, nPixels, alpha=alpha, reg=reg)
	gradient = 2*np.dot(Alin.T, np.dot(Alin, x0) - blin)

	return -gradient

def linearizedSol_bs(Obsdata, currImage, Prior, alpha=100, beta=100, reg="patch"):
# note what beta is

	# normalize the prior
	# TODO: SHOULD THIS BE DONE??
    zbl = np.nanmax(np.abs(Obsdata.unpack(['vis'])['vis']))
    nprior = zbl * Prior.imvec / np.sum(Prior.imvec)

    if reg == "patch":
        linRegTerm, constRegTerm = spatchlingrad(currImage.imvec, nprior)

	# Get bispectra data
    biarr = Obsdata.bispectra(mode="all", count="max")
    
    bispec = biarr['bispec']
    sigs = biarr['sigmab']   
    
    nans = np.isnan(sigs)
    bispec = bispec[nans==False]
    sigs = sigs[nans==False]
    biarr = biarr[nans==False]
    
    
    uv1 = np.hstack((biarr['u1'].reshape(-1,1), biarr['v1'].reshape(-1,1)))
    uv2 = np.hstack((biarr['u2'].reshape(-1,1), biarr['v2'].reshape(-1,1)))
    uv3 = np.hstack((biarr['u3'].reshape(-1,1), biarr['v3'].reshape(-1,1)))
    

    # Compute the fourier matrices
    A3 = (ftmatrix(currImage.psize, currImage.xdim, currImage.ydim, uv1, pulse=currImage.pulse),
          ftmatrix(currImage.psize, currImage.xdim, currImage.ydim, uv2, pulse=currImage.pulse),
          ftmatrix(currImage.psize, currImage.xdim, currImage.ydim, uv3, pulse=currImage.pulse)
         )

    Alin, blin = computeLinTerms_bi(currImage.imvec, A3, bispec, sigs, currImage.xdim*currImage.ydim, alpha=alpha, reg=reg)

    out = np.linalg.solve(Alin + beta*linRegTerm, blin + beta*constRegTerm);
    
    return image.Image(out.reshape((currImage.ydim, currImage.xdim)), currImage.psize, currImage.ra, currImage.dec, rf=currImage.rf, source=currImage.source, mjd=currImage.mjd, pulse=currImage.pulse)


def computeLinTerms_bi(x0, A3, bispec, sigs, nPixels, alpha=100, reg="patch"):

    
    sigmaR = sigmaI = 1.0/(sigs**2)

    rA = np.real(A3);
    iA = np.imag(A3);

    rA1 = rA[0];
    rA2 = rA[1];
    rA3 = rA[2];
    iA1 = iA[0];
    iA2 = iA[1];
    iA3 = iA[2];

    rA1x0 = np.dot(rA1, x0);
    rA2x0 = np.dot(rA2, x0);
    rA3x0 = np.dot(rA3, x0);
    iA1x0 = np.dot(iA1, x0);
    iA2x0 = np.dot(iA2, x0);
    iA3x0 = np.dot(iA3, x0);

    fR = (rA1x0*rA2x0*rA3x0 - rA1x0*iA2x0*iA3x0 - iA1x0*rA2x0*iA3x0 - iA1x0*iA2x0*rA3x0);
    fI = (rA1x0*rA2x0*iA3x0 + rA1x0*iA2x0*rA3x0 + iA1x0*rA2x0*rA3x0 - iA1x0*iA2x0*iA3x0);

    yR = np.real(bispec);
    yI = np.imag(bispec);

    #f = np.sum( sigmaR*(fR - yR)**2 + sigmaI*(fI - yI)**2  + 2.0*sigmaRI*(fR - yR)*(fI - yI)) / 2.0;

    # size number of bispectrum x nPixels
    dxReal = (rA1 *np.tile(rA2x0*rA3x0, [nPixels, 1]).T + rA2*np.tile(rA1x0*rA3x0, [nPixels, 1]).T   + rA3*np.tile(rA2x0*rA1x0, [nPixels, 1]).T \
    -( rA1*np.tile(iA2x0*iA3x0, [nPixels, 1]).T + iA2*np.tile(rA1x0*iA3x0, [nPixels, 1]).T   + iA3*np.tile(iA2x0*rA1x0, [nPixels, 1]).T) \
    -( iA1*np.tile(rA2x0*iA3x0, [nPixels, 1]).T + rA2*np.tile(iA1x0*iA3x0, [nPixels, 1]).T   + iA3*np.tile(rA2x0*iA1x0, [nPixels, 1]).T ) \
    -( iA1*np.tile(iA2x0*rA3x0, [nPixels, 1]).T + iA2*np.tile(iA1x0*rA3x0, [nPixels, 1]).T   + rA3*np.tile(iA2x0*iA1x0, [nPixels, 1]).T ) );

    # size number of bispectrum x nPixels
    dxImag = (rA1*np.tile(rA2x0*iA3x0, [nPixels, 1]).T + rA2*np.tile(rA1x0*iA3x0, [nPixels, 1]).T   + iA3*np.tile(rA2x0*rA1x0, [nPixels, 1]).T  \
    +( rA1*np.tile(iA2x0*rA3x0, [nPixels, 1]).T + iA2*np.tile(rA1x0*rA3x0, [nPixels, 1]).T   + rA3*np.tile(iA2x0*rA1x0, [nPixels, 1]).T) \
    +( iA1*np.tile(rA2x0*rA3x0, [nPixels, 1]).T + rA2*np.tile(iA1x0*rA3x0, [nPixels, 1]).T   + rA3*np.tile(rA2x0*iA1x0, [nPixels, 1]).T ) \
    -( iA1*np.tile(iA2x0*iA3x0, [nPixels, 1]).T + iA2*np.tile(iA1x0*iA3x0, [nPixels, 1]).T   + iA3*np.tile(iA2x0*iA1x0, [nPixels, 1]).T ) );

    #size number of bixpectrum x 1
    betaR = fR - np.dot(dxReal,x0);
    betaI = fI - np.dot(dxImag,x0);

    blin =  np.dot( np.transpose( np.dot( np.diag( alpha*sigmaR ) , dxReal)) , (yR - betaR) ) +  np.dot( np.transpose( np.dot( np.diag( alpha*sigmaI ), dxImag )), (yI - betaI) ) ;
    Alin =  np.dot( np.transpose( np.dot( np.diag( alpha*sigmaR ) , dxReal)) , dxReal ) + np.dot( np.transpose( np.dot( np.diag( alpha*sigmaI ), dxImag )) , dxImag );

    return (Alin, blin)

def spatchlingrad(imvec, priorvec):
    return (np.diag(np.ones(len(priorvec))), priorvec)

