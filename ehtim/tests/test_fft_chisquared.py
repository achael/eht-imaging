from __future__ import division
from __future__ import print_function

import numpy as np
import ehtim as eh
import time

im = eh.image.load_txt('models/avery_sgra_eofn.txt')
eht = eh.array.load_txt('arrays/EHT2019.txt')

PADFAC=10
PRAD_FFT = 12
PRAD_NFFT = 12

tint_sec = 5
tadv_sec = 600
tstart_hr = 0
tstop_hr = 24
bw_hz = 4e9

start = time.time()
obs_dft = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, sgrscat=False, ampcal=True, phasecal=False, ttype='direct', add_th_noise=False)
stop = time.time()
print ('direct',stop-start)

start = time.time()
obs_fft = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, sgrscat=False, ampcal=True, phasecal=False, ttype='fast', fft_pad_factor=PADFAC, add_th_noise=False)
stop = time.time()
print ('our fft',stop-start)

start = time.time()
obs_nfft = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, sgrscat=False, ampcal=True, phasecal=False, ttype='nfft', fft_pad_factor=PADFAC, add_th_noise=False)
stop = time.time()
print ('nfft',stop-start)


#prior = im.copy()
prior = eh.image.make_square(obs_dft, im.xdim , im.xdim*im.psize)
prior = prior.add_gauss(im.total_flux(), (50*eh.RADPERUAS, 50*eh.RADPERUAS, 0, 0, 0))

#im2 = im.copy() # This is our test image
im2 = prior.copy()

for j in range(len(im2.imvec)):
    im2.imvec[j] *= (1.0 + (np.random.rand()-0.5)/10.0)
    im2.imvec[j] += (np.random.rand()/10.)*im.imvec[j]

# mask
mask = im2.imvec > 0#0.1*np.mean(im2.imvec)

#mask=[]
test_imvec = im2.imvec

if len(mask) >0 and np.any(np.invert(mask)):
    print("unmasked size %i"%len(test_imvec))
    test_imvec = test_imvec[mask]
    print("masked size %i"%len(test_imvec))

# Testing the chi^2
for dtype in ['vis', 'bs', 'amp', 'cphase', 'camp', 'logcamp']:
    print('\nTesting',dtype)
    chisqdata_dft = eh.imager.chisqdata(obs_dft, prior, mask, dtype, ttype='direct')
    chisqdata_fft = eh.imager.chisqdata(obs_dft, prior, mask, dtype, ttype='fast', fft_pad_factor=PADFAC,p_rad=PRAD_FFT)
    chisqdata_nfft = eh.imager.chisqdata(obs_dft, prior, mask, dtype, ttype='nfft', fft_pad_factor=PADFAC,p_rad=PRAD_NFFT)
  
    chisq_dft = eh.imager.chisq(test_imvec, chisqdata_dft[2], chisqdata_dft[0], chisqdata_dft[1], dtype, ttype='direct', mask=mask)
    chisq_fft = eh.imager.chisq(test_imvec, chisqdata_fft[2], chisqdata_fft[0], chisqdata_fft[1], dtype, ttype='fast', mask=mask)
    chisq_nfft = eh.imager.chisq(test_imvec, chisqdata_nfft[2], chisqdata_nfft[0], chisqdata_nfft[1], dtype, ttype='nfft', mask=mask)
    print("\n")
    print("chisq_dft: %f" % chisq_dft)
    print("chisq_fft: %f" % chisq_fft)
    print("chisq_nfft: %f" % chisq_nfft)

    print("Fractional Difference direct-fast %0.4f"% np.abs((chisq_dft - chisq_fft)/(np.abs(chisq_dft))))
    print("Fractional Difference direct-nfft %0.4f"% np.abs((chisq_dft - chisq_nfft)/(np.abs(chisq_dft))))
    print("Fractional Difference nfft-fast %0.4f"% np.abs((chisq_nfft - chisq_fft)/(np.abs(chisq_nfft))))

# Testing the gradient of chi^2
for dtype in ['vis', 'bs', 'amp', 'cphase', 'camp', 'logcamp']:
    print('\nTesting',dtype)
    print('------------------------------')
    chisqdata_dft = eh.imager.chisqdata(obs_dft, prior, mask, dtype, ttype='direct')
    chisqdata_fft = eh.imager.chisqdata(obs_dft, prior, mask, dtype, ttype='fast', fft_pad_factor=PADFAC,p_rad=PRAD_FFT)
    chisqdata_nfft = eh.imager.chisqdata(obs_dft, prior, mask, dtype, ttype='nfft', fft_pad_factor=PADFAC,p_rad=PRAD_NFFT)
  
    chisq_dft_grad = eh.imager.chisqgrad(test_imvec, chisqdata_dft[2], chisqdata_dft[0], chisqdata_dft[1], dtype, ttype='direct', mask=mask)
    chisq_fft_grad = eh.imager.chisqgrad(test_imvec, chisqdata_fft[2], chisqdata_fft[0], chisqdata_fft[1], dtype, ttype='fast', mask=mask)
    chisq_nfft_grad = eh.imager.chisqgrad(test_imvec, chisqdata_nfft[2], chisqdata_nfft[0], chisqdata_nfft[1], dtype, ttype='nfft', mask=mask)
    #print("chisq_dft_grad:",chisq_dft_grad.reshape((im.ydim,im.xdim))[47:53,47:53])
    #print("chisq_fft_grad:",chisq_fft_grad.reshape((im.ydim,im.xdim))[47:53,47:53])
    compare_floor = np.min(np.abs(chisq_dft_grad))*1.e-20 + 1.e-100
    #chisq_dft_grad = chisq_dft_grad.reshape((im.ydim,im.xdim))[10:-10,10:-10]
    #chisq_fft_grad = chisq_fft_grad.reshape((im.ydim,im.xdim))[10:-10,10:-10]
    print("\n")
    print("Median Fractional Difference direct-fast %0.4f"% np.median(np.abs((chisq_dft_grad - chisq_fft_grad)/(np.abs(chisq_dft_grad)+compare_floor))))
    print("Median Fractional Difference direct-nfft %0.4f"% np.median(np.abs((chisq_dft_grad - chisq_nfft_grad)/(np.abs(chisq_dft_grad)+compare_floor))))
    print("Median Fractional Difference nfft-fast %0.4f"% np.median(np.abs((chisq_nfft_grad - chisq_fft_grad)/(np.abs(chisq_nfft_grad)+compare_floor))))

    print("Max Fractional Difference direct-fast %0.4f"% np.max(np.abs((chisq_dft_grad - chisq_fft_grad)/(np.abs(chisq_dft_grad)+compare_floor))))
    print("Max Fractional Difference direct-nfft %0.4f"% np.max(np.abs((chisq_dft_grad - chisq_nfft_grad)/(np.abs(chisq_dft_grad)+compare_floor))))
    print("Max Fractional Difference nfft-fast %0.4f"% np.max(np.abs((chisq_nfft_grad - chisq_fft_grad)/(np.abs(chisq_nfft_grad)+compare_floor))))



