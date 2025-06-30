#!/usr/bin/env python

# This is a rough script to verify the consistency of Fourier transform types and to check gradients of the various image regularization options

from __future__ import division
from __future__ import print_function

import numpy as np
import ehtim as eh
from ehtim.imaging.imager_utils import chisqdata, chisq, chisqgrad

path = eh.__path__[0]
im = eh.image.load_txt(path + '/../models/jason_mad_eofn.txt')
eht = eh.array.load_txt(path + '/../arrays/EHT2017.txt')

tint_sec = 5
tadv_sec = 600
tstart_hr = 0
tstop_hr = 24
bw_hz = 4e9
obs_dft = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, sgrscat=False, ampcal=True, phasecal=True, ttype='direct', add_th_noise=False)
obs_nfft = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, sgrscat=False, ampcal=True, phasecal=True, ttype='nfft', add_th_noise=False)

prior = im.copy().regrid_image(im.fovx(),50)
im2 = prior.copy() # This is our test image

# Add random noise to the image
for j in range(len(im2.imvec)):
    im2.imvec[j] *= (1.0 + (np.random.rand()-0.5)/10.0)
    im2.imvec[j] += (1.0 + (np.random.rand()-0.5)/10.0) * np.mean(im2.imvec)

# mask
mask = (im2.imvec > 0.5*np.median(im2.imvec))

#mask=[]
test_imvec = im2.imvec

if len(mask) >0 and np.any(np.invert(mask)):
    print("unmasked size %i"%len(test_imvec))
    test_imvec = test_imvec[mask]
    print("masked size %i"%len(test_imvec))

# Testing the chi^2
for dtype in ['vis', 'bs', 'amp', 'cphase',  'camp', 'logcamp']:#'cphase_diag', 'logcamp_diag']:
    print('\nTesting chi^2 dtype:',dtype)
    chisqdata_dft = chisqdata(obs_dft, prior, mask, dtype, ttype='direct')
    chisqdata_nfft = chisqdata(obs_nfft, prior, mask, dtype, ttype='nfft')
    chisq_dft = chisq(test_imvec, chisqdata_dft[2], chisqdata_dft[0], chisqdata_dft[1], dtype, ttype='direct', mask=mask)
    chisq_nfft = chisq(test_imvec, chisqdata_nfft[2], chisqdata_nfft[0], chisqdata_nfft[1], dtype, ttype='nfft', mask=mask)
    print("chisq_dft: %f" % chisq_dft)
    print("chisq_nfft: %f" % chisq_nfft)

# Testing the gradient of chi^2
for dtype in ['vis', 'bs', 'amp', 'cphase',  'camp', 'logcamp']:#'cphase_diag', 'logcamp_diag']:
    print('\nTesting chi^2 gradient dtype:',dtype)
    chisqdata_dft = chisqdata(obs_dft, prior, mask, dtype, ttype='direct')
    chisqdata_nfft = chisqdata(obs_nfft, prior, mask, dtype, ttype='nfft')
    chisq_dft_grad = chisqgrad(test_imvec, chisqdata_dft[2], chisqdata_dft[0], chisqdata_dft[1], dtype, ttype='direct', mask=mask)
    chisq_nfft_grad = chisqgrad(test_imvec, chisqdata_nfft[2], chisqdata_nfft[0], chisqdata_nfft[1], dtype, ttype='nfft', mask=mask)
    compare_floor = 1.0
    print("Median Fractional Difference of DTFT/NFFT for chi^2 gradient of " + dtype, np.median(np.abs((chisq_dft_grad - chisq_nfft_grad)/(np.abs(chisq_dft_grad)+compare_floor))))

    dx = 1.e-12
   #y0 = chisq(test_imvec, chisqdata_dft[2], chisqdata_dft[0], chisqdata_dft[1], dtype, ttype='direct', mask=mask)
    y0 = chisq(test_imvec, chisqdata_nfft[2], chisqdata_nfft[0], chisqdata_nfft[1], dtype, ttype='nfft', mask=mask)
    grad_n = np.zeros(len(test_imvec))
    for j in range(len(test_imvec)):
        test_imvec2 = test_imvec.copy()
        test_imvec2[j] += dx
        #y1 = chisq(test_imvec2, chisqdata_dft[2], chisqdata_dft[0], chisqdata_dft[1], dtype, ttype='direct', mask=mask)
        y1 = chisq(test_imvec2, chisqdata_nfft[2], chisqdata_nfft[0], chisqdata_nfft[1], dtype, ttype='nfft', mask=mask)
        grad_n[j] = (y1-y0)/dx

    compare_floor = np.min(np.abs(chisq_nfft_grad))*1.e-20 + 1.e-100
    print("Median Fractional Gradient Difference %0.4f"% np.median(np.abs((grad_n - chisq_nfft_grad)/(np.abs(chisq_nfft_grad)+compare_floor))))    
    print("Maximum Fractional Gradient Difference %0.4f"% np.max(np.abs((grad_n - chisq_nfft_grad)/(np.abs(chisq_nfft_grad)+compare_floor))))    
    #print("\nMedian Fractional Gradient Difference for " + dtype + ":",np.median(np.abs((grad_n-chisq_dft_grad)/chisq_dft_grad)))
    #print("Maximal Fractional Gradient Difference for " + dtype + ":",np.max(np.abs((grad_n-chisq_dft_grad)/chisq_dft_grad)),'\n')       
    
# Testing the gradients of image regularization functions
import ehtim.imaging.imager_utils as iu
prior = test_imvec * 0.0 + 1.0
prior = prior * np.sum(test_imvec)/np.sum(prior)
#mask = [True,] * len(test_imvec)
for reg in ['simple', 'gs', 'l1', 'tv', 'tv2']:
    dx = 1.e-12
    y0 = iu.regularizer(test_imvec, prior, mask, 1.0, im2.xdim, im2.ydim, im2.psize, reg)
    grad_exact = iu.regularizergrad(test_imvec, prior, mask, 1.0, im2.xdim, im2.ydim, im2.psize, reg)
    grad = np.zeros(len(test_imvec))
    for j in range(len(test_imvec)):
        test_imvec2 = test_imvec.copy()
        test_imvec2[j] += dx
        y1 = iu.regularizer(test_imvec2, prior, mask, 1.0, im2.xdim, im2.ydim, im2.psize, reg)
        grad[j] = (y1-y0)/dx
    print("\nMedian Fractional Gradient Difference for " + reg + ":",np.median(np.abs((grad-grad_exact)/grad_exact)))
    print("Maximal Fractional Gradient Difference for " + reg + ":",np.max(np.abs((grad-grad_exact)/grad_exact)))       
