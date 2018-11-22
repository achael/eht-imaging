# This is a rough script to verify the consistency of Fourier transform types and to check gradients of the various image regularization options

from __future__ import division
from __future__ import print_function

import numpy as np
import ehtim as eh

im = eh.image.load_txt('../models/avery_sgra_eofn.txt')
eht = eh.array.load_txt('../arrays/EHT2017.txt')

tint_sec = 5
tadv_sec = 600
tstart_hr = 0
tstop_hr = 24
bw_hz = 4e9
obs_dft = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, sgrscat=False, ampcal=True, phasecal=True, ttype='direct', add_th_noise=False)
obs_nfft = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, sgrscat=False, ampcal=True, phasecal=True, ttype='nfft', add_th_noise=False)

prior = im.copy()
im2 = im.copy() # This is our test image

# Add random noise to the image
for j in range(len(im2.imvec)):
    im2.imvec[j] *= (1.0 + (np.random.rand()-0.5)/10.0)
    im2.imvec[j] += (1.0 + (np.random.rand()-0.5)/10.0) * np.mean(im2.imvec)

# mask
mask = im2.imvec > 0.1*np.mean(im2.imvec)

#mask=[]
test_imvec = im2.imvec

if len(mask) >0 and np.any(np.invert(mask)):
    print("unmasked size %i"%len(test_imvec))
    test_imvec = test_imvec[mask]
    print("masked size %i"%len(test_imvec))

# Testing the chi^2
for dtype in ['vis', 'bs', 'amp', 'cphase', 'camp', 'logcamp']:
    print('\nTesting chi^2 dtype:',dtype)
    chisqdata_dft = eh.imager.chisqdata(obs_dft, prior, mask, dtype, ttype='direct')
    chisqdata_nfft = eh.imager.chisqdata(obs_nfft, prior, mask, dtype, ttype='nfft')
    chisq_dft = eh.imager.chisq(test_imvec, chisqdata_dft[2], chisqdata_dft[0], chisqdata_dft[1], dtype, ttype='direct', mask=mask)
    chisq_nfft = eh.imager.chisq(test_imvec, chisqdata_nfft[2], chisqdata_nfft[0], chisqdata_nfft[1], dtype, ttype='nfft', mask=mask)
    print("chisq_dft: %f" % chisq_dft)
    print("chisq_nfft: %f" % chisq_nfft)

# Testing the gradient of chi^2
for dtype in ['vis', 'bs', 'amp', 'cphase', 'camp', 'logcamp']:
    print('\nTesting chi^2 gradient dtype:',dtype)
    chisqdata_dft = eh.imager.chisqdata(obs_dft, prior, mask, dtype, ttype='direct')
    chisqdata_nfft = eh.imager.chisqdata(obs_nfft, prior, mask, dtype, ttype='nfft')
    chisq_dft_grad = eh.imager.chisqgrad(test_imvec, chisqdata_dft[2], chisqdata_dft[0], chisqdata_dft[1], dtype, ttype='direct', mask=mask)
    chisq_nfft_grad = eh.imager.chisqgrad(test_imvec, chisqdata_nfft[2], chisqdata_nfft[0], chisqdata_nfft[1], dtype, ttype='nfft', mask=mask)
    compare_floor = 1.0
    print("Median Fractional Difference of DTFT/NFFT for chi^2 gradient of " + dtype, np.median(np.abs((chisq_dft_grad - chisq_nfft_grad)/(np.abs(chisq_dft_grad)+compare_floor))))

# Testing the gradients of image regularization functions
import ehtim.imaging.imager_utils as iu
prior = test_imvec * 0.0 + 1.0
prior = prior * np.sum(test_imvec)/np.sum(prior)
mask = [True,] * len(test_imvec)
for reg in ['simple', 'gs', 'l1', 'tv', 'tv2']:
    dx = 1.e-12
    y0 = iu.regularizer(test_imvec, prior, mask, 1.0, im.xdim, im.ydim, im.psize, reg)
    grad_exact = iu.regularizergrad(test_imvec, prior, mask, 1.0, im.xdim, im.ydim, im.psize, reg)
    grad = np.zeros(len(test_imvec))
    for j in range(len(test_imvec)):
        test_imvec2 = test_imvec.copy()
        test_imvec2[j] += dx
        y1 = iu.regularizer(test_imvec2, prior, mask, 1.0, im.xdim, im.ydim, im.psize, reg)
        grad[j] = (y1-y0)/dx
    print("Median Fractional Gradient Difference for " + reg + ":",np.median(np.abs((grad-grad_exact)/grad_exact)))
        
