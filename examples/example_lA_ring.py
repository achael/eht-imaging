import numpy as np
import ehtim as eh
import matplotlib.pyplot as plt

# Load the image and the array
im = eh.image.load_txt('models/avery_sgra_eofn.txt')
eht = eh.array.load_txt('arrays/EHT2017.txt')

im.imvec *= 0.0
im = im.add_crescent(1.0, 25.*eh.RADPERUAS,20.*eh.RADPERUAS, 0, 0)
im = im.blur_circ(5.0*eh.RADPERUAS)

# Look at the image
im.display()

tint_sec = 5
tadv_sec = 600
tstart_hr = 0
tstop_hr = 24
bw_hz = 4e9
obs = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz,
                 sgrscat=False, ampcal=True, phasecal=True)

res = obs.res() # nominal array resolution, 1/longest baseline

# Generate an image prior
npix = 64
fov = 1*im.fovx()
zbl = im.total_flux() # total flux
prior_fwhm = 200*eh.RADPERUAS # Gaussian size in microarcssec
emptyprior = eh.image.make_square(obs, npix, fov)
flatprior = emptyprior.add_flat(zbl)
gaussprior = emptyprior.add_gauss(zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))

# Define the data products
d1 = 'vis'
d2 = False
alpha_d1 = 100.0
alpha_d2 = 0.0

# Image using MEM
flux = zbl
out  = eh.imager_func(obs, gaussprior, gaussprior, flux,
                      d1=d1, d2=d2, alpha_d1=alpha_d1, alpha_d2=alpha_d2, s1='simple', alpha_s1=10, ttype='nfft',
                      maxit=100, show_updates=False, beam_size=res, norm_reg=True)

for repeat in range(5):
    out  = eh.imager_func(obs, out.blur_circ(res), gaussprior, flux,
                      d1=d1, d2=d2, alpha_d1=alpha_d1, alpha_d2=alpha_d2, s1='simple', alpha_s1=10, ttype='nfft',
                      maxit=300, show_updates=False, beam_size=res, norm_reg=True)

out.display()
out.display(scale='log')

# Now with the tunable sparsity regularizer and tv2
alpha_A = 100.0
out2 = out.copy()
for repeat in range(5):
    alpha_A *= 2.0
    out2 = eh.imager_func(obs, out2.blur_circ(res/2.0), gaussprior, flux, d1=d1, d2=d2, alpha_d1=alpha_d1, alpha_d2=alpha_d2, s1='tv2', s2='lA', alpha_s1=10.0, alpha_s2=10.0, ttype='nfft', alpha_A = alpha_A, maxit=300, show_updates=False, beam_size=res, norm_reg=True)

for repeat in range(5):
    alpha_A *= 2.0
    out2 = eh.imager_func(obs, out2.blur_circ(res/2.0), gaussprior, flux, d1=d1, d2=d2, alpha_d1=alpha_d1, alpha_d2=alpha_d2, s1='tv2', s2='lA', alpha_s1=20.0, alpha_s2=20.0, ttype='nfft', alpha_A = alpha_A, maxit=500, show_updates=False, beam_size=res, norm_reg=True)

out2.display()
out2.display(scale='log')
