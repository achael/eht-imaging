import vlbi_imaging_utils as vb
import imager_dft as imgr
import maxen as mx
import numpy as np
import vlbi_plots as vp
import cross_validation as cv
import imager_dft as imdft

im = vb.load_im_txt('./models/avery_sgra_eofn.txt') #for a text file
eht = vb.load_array('./arrays/EHT2017.txt') #see the attached array text file

# Look at the image
im.display()

# Observe the image
# tint_sec is the integration time in seconds, and tadv_sec is the advance time between scans
# tstart_hr is the GMST time of the start of the observation and tstop_hr is the GMST time of the end
# bw_hz is the  bandwidth in Hz
# sgrscat=True blurs the visibilities with the Sgr A* scattering kernel for the appropriate image frequency
# ampcal and phasecal determine if gain variations and phase errors are included
tint_sec = 60
tadv_sec = 600
tstart_hr = 0
tstop_hr = 24
bw_hz = 4e9
obs = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, sgrscat=False, ampcal=True, phasecal=True)

#Split the observation into a training and testing dataset and compare the u-v coverage
ret=cv.split_obs_training_test(obs, frac_test=0.1)
vp.plotall_obs_compare([ret['training'],ret['testing']],'u','v',conj=True)

#First, image everything
# Resolution
beamparams = obs.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
res = obs.res() # nominal array resolution, 1/longest baseline
print beamparams 
print res
# Generate an image prior
npix = 50
fov = 1*im.xdim * im.psize
zbl = np.sum(im.imvec) # total flux
prior_fwhm = 60*vb.RADPERUAS # Gaussian size in microarcssec
emptyprior = vb.make_square(obs, npix, fov)
flatprior = vb.add_flat(emptyprior, zbl)
gaussprior = vb.add_gauss(emptyprior, zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))

#Initial image
out = imdft.imager(obs, gaussprior, gaussprior, zbl, maxit=200, alpha_s1=1.0, s1='simple', show_updates=False);
for j in range(5):
    out = imdft.imager(obs, mx.blur_circ(out, res), gaussprior, zbl, maxit=200, alpha_s1=1.0, s1='simple', show_updates=False);
out.display()

#Now try one-parameter cross validation
num_iter = 2
frac_test = 0.1 #fraction of data to use as the CV test set
entropy = 'tv'
log_alpha_test_vals = [-4,-2,0,2,4]
cv_output_chi2 = np.zeros(len(log_alpha_test_vals))
cv_output_im = [im.copy() for j in range(len(log_alpha_test_vals))] #to show images of the cross-validation results

for i in range(num_iter):
    ret=cv.split_obs_training_test(obs, frac_test=frac_test)    
    for i_alpha in range(len(log_alpha_test_vals)):
        log_alpha_test = log_alpha_test_vals[i_alpha]
        out_cv = imdft.imager(ret['training'], mx.blur_circ(out, res), gaussprior, zbl, maxit=200, alpha_s1=10.**log_alpha_test, s1=entropy, show_updates=False);
        out_cv = imdft.imager(obs, mx.blur_circ(out_cv, res), gaussprior, zbl, maxit=200, alpha_s1=10.**log_alpha_test, s1=entropy, show_updates=False);
        cv_output_chi2[i_alpha] += cv.chisq(out_cv, ret['testing'])/float(num_iter)
        cv_output_im[i_alpha] = out_cv.copy()
        print "----------------------"
        print "alpha: ",(10.**log_alpha_test)
        print "cross-validation chi2: ",cv.chisq(out_cv, ret['testing'])    
        print "----------------------"

print "------------------"
print "Cross Validation Results (alpha, <chi^2> of test set"
for i_alpha in range(len(log_alpha_test_vals)):
    print '%f:\t%f' % (10.**log_alpha_test_vals[i_alpha],cv_output_chi2[i_alpha])
print "------------------"
cv.plot_im_List_Set([cv_output_im],[cv_output_chi2], [10.**x for x in log_alpha_test_vals]) #plot the image reconstructions with cross-validation


#Now try two-parameter cross validation
num_iter = 2
frac_test = 0.1 #fraction of data to use as the CV test set
entropy1 = 'tv'
entropy2 = 'l1'
log_alpha_test_vals1 = [-4,0,4]
log_alpha_test_vals2 = [-2,2]
cv_output_chi2 = np.zeros((len(log_alpha_test_vals1),len(log_alpha_test_vals2)))
cv_output_im = [[im.copy() for j2 in range(len(log_alpha_test_vals2))] for j1 in range(len(log_alpha_test_vals1))] #to show images of the cross-validation results

for i in range(num_iter):
    ret=cv.split_obs_training_test(obs, frac_test=frac_test)    
    for i1_alpha in range(len(log_alpha_test_vals1)):
        for i2_alpha in range(len(log_alpha_test_vals2)):
            log_alpha_test1 = log_alpha_test_vals1[i1_alpha]
            log_alpha_test2 = log_alpha_test_vals1[i2_alpha]
            out_cv = imdft.imager(ret['training'], mx.blur_circ(out, res), gaussprior, zbl, maxit=200, alpha_s1=10.**log_alpha_test1, alpha_s2=10.**log_alpha_test2, s1=entropy1, s2=entropy2, show_updates=False);
            out_cv = imdft.imager(obs, mx.blur_circ(out_cv, res), gaussprior, zbl, maxit=200, alpha_s1=10.**log_alpha_test1, alpha_s2=10.**log_alpha_test2, s1=entropy1, s2=entropy2, show_updates=False);
            cv_output_chi2[i1_alpha][i2_alpha] += cv.chisq(out_cv, ret['testing'])/float(num_iter)
            cv_output_im[i1_alpha][i2_alpha] = out_cv.copy()
            print "----------------------"
            print "alpha_s1: ",(10.**log_alpha_test1)
            print "alpha_s2: ",(10.**log_alpha_test2)
            print "cross-validation chi2: ",cv.chisq(out_cv, ret['testing'])    
            print "----------------------"

print "------------------"
print "Cross Validation Results (alpha, <chi^2> of test set"
for i1_alpha in range(len(log_alpha_test_vals1)):
    for i2_alpha in range(len(log_alpha_test_vals2)):
        print '%f:\t%f:\t%f' % (10.**log_alpha_test_vals1[i1_alpha],10.**log_alpha_test_vals2[i2_alpha],cv_output_chi2[i1_alpha][i2_alpha])
print "------------------"
cv.plot_im_List_Set(cv_output_im,cv_output_chi2,[[10.**x for x in log_alpha_test_vals1],[10.**x for x in log_alpha_test_vals2]]) #plot the image reconstructions with cross-validation



