import vlbi_imaging_utils as vb
import maxen as mx
import numpy as np

im = vb.load_im_txt('./models/avery_sgra_eofn.txt')
arr = vb.load_array("./arrays/EHT2017_wKP_wRedundant.txt")

tint = 120
tadv = 600
bw = 4e9
obs = im.observe(arr, tint, tadv, 0, 24.0, bw, ampcal=False, phasecal=False, sgrscat=False)

# Generate an image prior
npix = 64
fov = 1.0 * im.xdim*im.psize #160.0*vb.RADPERUAS
zbl = 1.0 # total flux
prior_fwhm = 80*vb.RADPERUAS # Gaussian size in microarcssec
emptyprior = vb.make_square(obs, npix, fov)
flatprior = vb.add_flat(emptyprior, zbl)
gaussprior = vb.add_gauss(emptyprior, zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))

beamparams = obs.fit_beam()
res = 1 / np.max(obs.unpack('uvdist')['uvdist'])
print(beamparams) 
print(res)

out_cl = mx.maxen_onlyclosure(obs, gaussprior, flux = 1.0, maxit=50, alpha_clphase=10, alpha_clamp=10, gamma=500, delta=1e10, entropy="simple", stop=1e-15, grads=True)
out_cl = mx.maxen_onlyclosure(obs, mx.blur_circ(out_cl, 1e-10), flux = 1.0, maxit=100, alpha_clphase=10, alpha_clamp=10, gamma=500, delta=1e10, entropy="tv", stop=1e-10, grads=True)
out_cl = mx.maxen_onlyclosure(obs, mx.blur_circ(out_cl, 1e-10), flux = 1.0, maxit=100, alpha_clphase=5, alpha_clamp=5, gamma=500, delta=1e10, entropy="tv", stop=1e-10, grads=True)
out_cl = mx.maxen_onlyclosure(obs, mx.blur_circ(out_cl, 1e-10), flux = 1.0, maxit=100, alpha_clphase=5, alpha_clamp=5, gamma=500, delta=1e10, entropy="tv", stop=1e-10, grads=True)
out_cl = mx.maxen_onlyclosure(obs, mx.blur_circ(out_cl, 1e-10), flux = 1.0, maxit=100, alpha_clphase=5, alpha_clamp=5, gamma=500, delta=1e10, entropy="tv", stop=1e-10, grads=True)

mx.blur_circ(out_cl, 0.5e-10).display()

im.save_txt("Truth_MAD-Disk.txt")
vb.blur_gauss(im, beamparams, 0.5, frac_pol=0).save_txt("Truth_MAD-Disk_halfCLEAN.txt")
out_cl.save_txt("ClosureOnly_MAD-Disk_EHT2017wKP_wRedundant.txt")
vb.blur_gauss(out_cl, beamparams, 0.5, frac_pol=0).save_txt("ClosureOnly_MAD-Disk_EHT2017wKP_wRedundant_halfCLEAN.txt")




mx.blur_circ(out_cl, 0.5e-10).display()




out_cl = mx.maxen_onlyclosure(obs, mx.blur_circ(out_cl, 1e-10), flux = 1.0, maxit=100, alpha_clphase=10, alpha_clamp=10, gamma=500, delta=500, entropy="simple", stop=1e-10)




#Image with complex visibilities
out = mx.maxen(obs, gaussprior, maxit=250, alpha=100)
mx.blur_circ(out, 0.5e-10).display()


#Image with only closure quantities
out_cl = mx.maxen_onlyclosure(obs, out, flux = 1.0, maxit=1000, alpha_clphase=100, alpha_clamp=100, gamma=500, delta=500, entropy="simple", stop=1e-10)

out_cl = mx.maxen_onlyclosure(obs, mx.blur_circ(out, 2e-10), flux = 1.0, maxit=1000, alpha_clphase=100, alpha_clamp=100, gamma=500, delta=500, entropy="simple", stop=1e-10)



out_cl = mx.maxen_onlyclosure(obs, gaussprior, flux = 1.0, maxit=100, alpha_clphase=10, alpha_clamp=10, gamma=500, delta=500, entropy="simple", stop=1e-10)

out_cl = mx.maxen_onlyclosure(obs, mx.blur_circ(out_cl, 1e-10), flux = 1.0, maxit=100, alpha_clphase=10, alpha_clamp=10, gamma=500, delta=500, entropy="simple", stop=1e-10)


out_cl = mx.maxen_onlyclosure(obs, mx.blur_circ(out, 0.5e-10), flux = 1.0, maxit=1000, alpha_clphase=100, alpha_clamp=100, gamma=500, delta=500, entropy="simple", stop=1e-10)

#Image with only closure quantities
out = mx.maxen_onlyclosure(obs, gaussprior, flux = 1.0, maxit=1000, alpha_clphase=100, alpha_clamp=100, gamma=500, delta=500, entropy="simple", stop=1e-10)

#Image with bispectrum
out = mx.maxen_bs(obs, gaussprior, 2.0, maxit=50, alpha=100, stop=1e-10)

#Image with only closure quantities
out = mx.maxen_onlyclosure(obs, gaussprior, flux = 1.0, maxit=1000, alpha_clphase=100, alpha_clamp=100, gamma=500, delta=500, entropy="simple", stop=1e-10)
