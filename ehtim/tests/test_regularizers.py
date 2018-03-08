from __future__ import division
from __future__ import print_function

import numpy as np
import ehtim as eh
import time
import ehtim.imaging.imager_utils as iu

im = eh.image.load_txt('models/avery_sgra_eofn.txt')
im.pulse = eh.observing.pulses.deltaPulse2D
   
# mask
mask = im.imvec > 0

imvec = im.imvec
nprior= im.imvec*0.0+1.0
nprior= nprior/np.sum(nprior)
embed_mask = mask
flux = im.total_flux()*0.95 #not exactly equal to the total flux
alpha_A = 5000.0#7.5

# Test the normalization of the regularizers
blur_im = im.blur_circ(20.*eh.RADPERUAS)
for rtype in iu.REGULARIZERS:
    print(rtype,'norm_reg=False',iu.regularizer(blur_im.imvec, nprior, embed_mask, flux, im.xdim, im.ydim, im.psize, rtype, alpha_A = alpha_A, beam_size=20.*eh.RADPERUAS, norm_reg=False))
    print(rtype,'norm_reg=True',iu.regularizer(blur_im.imvec, nprior, embed_mask, flux, im.xdim, im.ydim, im.psize, rtype, alpha_A = alpha_A, beam_size=20.*eh.RADPERUAS, norm_reg=True))

for rtype in iu.REGULARIZERS:
    print('\nTesting the gradient of',rtype)
    def reg(imvec):
        return iu.regularizer(imvec, nprior, embed_mask, flux, im.xdim, im.ydim, im.psize, rtype, beam_size=20.*eh.RADPERUAS, alpha_A = alpha_A, norm_reg=True)

    def reggrad(imvec):
        return iu.regularizergrad(imvec, nprior, embed_mask, flux, im.xdim, im.ydim, im.psize, rtype, beam_size=20.*eh.RADPERUAS, alpha_A = alpha_A, norm_reg=True)

    def reggrad_numeric(imvec):
        dx = 1e-10
        reg1 = reg(imvec)
        grad = reggrad(imvec)
        for j in range(len(imvec)):
            imvec2 = imvec.copy()
            imvec2[j] += dx
            grad[j] = (reg(imvec2) - reg1)/dx
        return grad

    print("reg: %f" % reg(imvec))
    grad1 = reggrad(imvec)
    grad2 = reggrad_numeric(imvec)
    pad = np.median(np.abs(grad1))/1000.0
    print("reg_grad analytic: ", grad1)
    print("reg_grad numeric:  ", grad2)
    print("Fractional Difference %0.4f"% np.max(np.abs((grad1 - grad2))/(np.abs(grad1)+pad)))
