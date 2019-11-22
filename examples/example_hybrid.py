import matplotlib.pyplot as plt
import numpy as np
import ehtim as eh
import ehtim.modeling.hybrid_utils as hu
import ehtim.imaging.dynamical_imaging as di

# Sample array
eht = eh.array.load_txt('../arrays/EHT2019.txt')

# Load an image
im = eh.image.load_txt('../models/jason_mad_eofn.txt')

# Create an observation
tint_sec = 60
tadv_sec = 600
tstart_hr = 0
tstop_hr = 24
bw_hz = 8e9
obs = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz,
                 sgrscat=False, ampcal=True, phasecal=True)

# Generate an image prior
npix = 64
fov = im.fovx()
im_flux = im.total_flux()*0.9 # total image flux
prior_fwhm = 60*eh.RADPERUAS # Gaussian size in microarcssec
emptyprior = eh.image.make_square(obs, npix, fov)
gaussprior = emptyprior.add_gauss(im_flux, (prior_fwhm, prior_fwhm, 0, 0, 0))

# Define the fitted model
mod_init = eh.model.Model()
mod_init.add_thick_mring(0.1, 50.*eh.RADPERUAS, 1.*eh.RADPERUAS, x0 = 0., y0 = 0., beta_list=[0.])
mod_prior = mod_init.default_prior()
mod_prior[0]['F0'] = {'prior_type': 'gauss', 'mean': im.total_flux()*0.1, 'std': im.total_flux()*0.05, 'transform': 'log'}
mod_prior[0]['d'] = {'prior_type': 'gauss', 'mean': 50.0*eh.RADPERUAS, 'std': 10.0*eh.RADPERUAS, 'transform': 'log'}
mod_prior[0]['alpha'] = {'prior_type': 'gauss', 'mean': 5.0, 'std': 2, 'transform': 'log'}
mod_prior[0]['beta1_re'] = {'prior_type': 'fixed'}
mod_prior[0]['beta1_im'] = {'prior_type': 'fixed'}

# Initial model fit using only long baselines
mod_fit = eh.modeler_func(obs.add_fractional_noise(0.05).flag_uvdist(uv_min=2e9), mod_init, mod_prior, d1='vis', alpha_d1=1, maxit=250)
eh.comp_plots.plotall_obs_im_compare(obs, mod_fit, 'uvdist', 'amp')

# Fit again, also allowing azimuthal brightness asymmetry
mod_prior[0]['beta1_re'] = {'prior_type': 'gauss', 'mean': 0.0, 'std': 0.25}
mod_prior[0]['beta1_im'] = {'prior_type': 'gauss', 'mean': 0.0, 'std': 0.25}    
mod_fit = eh.modeler_func(obs.add_fractional_noise(0.05).flag_uvdist(uv_min=2e9), mod_fit, mod_prior, d1='vis', alpha_d1=1, maxit=250)
eh.comp_plots.plotall_obs_im_compare(obs, mod_fit, 'uvdist', 'amp')

# Initial image fit using only short baselines
im_init = eh.imager_func(obs.flag_uvdist(uv_max=1e10), gaussprior, gaussprior, im_flux, 
                 s1='simple', alpha_s1=1, 
                 s2='tv2',    alpha_s2=1,  
                 d1='vis', alpha_d1=1, 
                 alpha_flux=0, 
                 norm_reg=True,maxit=250,show_updates=False)

# Perform the hybrid fit
hybrid_fit = eh.hybrid_func(obs, mod_fit, mod_prior, im_init.blur_circ(10.*eh.RADPERUAS), gaussprior, im_flux, 
                 s1='simple', alpha_s1=1, 
                 s2='tv2',    alpha_s2=1,  
                 d1='vis', alpha_d1=1, 
                 alpha_flux=0, alpha_cm=0, 
                 norm_reg=True,maxit=250)

# Repeat the fit, initializing to the blurred output image
for repeat in range(3):
    hybrid_fit = eh.hybrid_func(obs, hybrid_fit['model'], mod_prior, hybrid_fit['image'].blur_circ(10.*eh.RADPERUAS), gaussprior, im_flux, 
                 s1='simple', alpha_s1=1, 
                 s2='tv2',    alpha_s2=1,  
                 d1='vis', alpha_d1=5, 
                 alpha_flux=0, alpha_cm=0,  
                 norm_reg=True,maxit=250,show_updates=False)

# Display the results
hybrid_im = hybrid_fit['model'].image_same(im)
hybrid_im.imvec += hybrid_fit['image'].regrid_image(im.fovx(), im.xdim).imvec

di.plot_im_List([im, hybrid_fit['image'].regrid_image(im.fovx(), im.xdim), hybrid_fit['model'].image_same(im), hybrid_im])
eh.comp_plots.plotall_obs_im_compare(obs, [im_init, hybrid_fit['model'], hybrid_im], 'uvdist', 'amp',legendlabels=['Image','Model','Hybrid'])

res = obs.res()
di.plot_im_List([im.blur_circ(res/4), hybrid_fit['image'].regrid_image(im.fovx(), im.xdim).blur_circ(res/4), hybrid_fit['model'].image_same(im).blur_circ(res/4), hybrid_im.blur_circ(res/4)])
