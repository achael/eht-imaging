import os
import matplotlib.pyplot as plt
import numpy as np
import ehtim as eh

######################################################################################################################
# Imaging parameters
######################################################################################################################

image_pol = False
SCRIPTNAME='imagingtutorialscript_'


datafolder = '../results/'
name = 'simobs'
pathtooutput = '../results'
pathtoscripts = '.' 

source = 'M87'
zbl = 0.8                    # How much flux we believe lives in the compact component (Jy)
zbl_tot = 0.8                # True zero baseline values as measured by ALMA (Jy)
sys_noise = 0.01             # Systematic noise added to visibilities to account for (e.g.,) leakage
prior_fwhm = 50*eh.RADPERUAS # Gaussian prior size
fov = 120*eh.RADPERUAS       # field of view of the reconstructed image
npix = 64                    # number of pixels across the reconstructed image
tag = name + '_zbl' + str(zbl)

flag_zbl = False             # Option to flag zero baselines
fit_amps  = True             # Whether or not to include visibility amplitudes in the imaging
flag_amps = False            # Whether to flag anomalous amplitudes
snr_cut   = 0                # SNR cutoff
t_avg     = 0                # coherent averaging time (seconds)
LZ_gauss = 40*eh.RADPERUAS   # Gaussian FWHM for self-calibration of the LMT-SMT baseline (for M87)
LMT_rescale = 1.5            # Rescaling factor for LMT baselines (for non-M87)
systematic_noise = {'ALMA':0.01, 'APEX':0.01, 'SMT':0.05, 'LMT':0.15, 'PV':0.05, 'SMA':0.01, 'JCMT':0.01} # systematic noise on a priori amplitudes
reg_term = {'simple':100, 'tv':1, 'tv2':1} # Image regularization parameters

# set the data term weights used during imaging
if fit_amps:
    data_term={'amp':.2, 'cphase':1, 'logcamp':1}
else:
    data_term={'cphase':1, 'logcamp':1}

######################################################################################################################
# Load and prepare data
######################################################################################################################

# load the uvfits file
obs = eh.obsdata.load_uvfits(datafolder + name + '.uvfits')
obs_orig = obs.copy()
res = obs.res() # nominal array resolution, 1/longest baseline

# Rescale short baselines to excite contributions from extended flux. 
# setting zbl < zbl_tot assumes there is an extended constant flux component of zbl_tot-zbl Jy 
for j in range(len(obs.data)):
    if (obs.data['u'][j]**2 + obs.data['v'][j]**2)**0.5 < 0.1e9:
        for k in range(-8,0):
            obs.data[j][k] *= zbl/zbl_tot

# Do scan averaging
print("Coherently averaging the data...")
obs.add_scans() # this seperate the data into scans, if it isn't done so already with an NX table
obs = obs.avg_coherent(0.,scan_avg=True) # average each scan coherantly

# Drop low-snr points
if snr_cut > 0:
    print("\nFlagging low-snr points...")
    obs = obs.flag_low_snr(snr_cut)

# Flag problematic data again
if flag_amps:
    print("Flagging anomalous amplitudes...")
    obs = obs.flag_anomalous('amp',max_diff_seconds=1200.0)

# Flag zero baselines 
if flag_zbl:
    obs = obs.flag_uvdist(uv_min = 0.1e9)

# Order stations. This is to create a minimal set of closure quantities with the highest snr
obs.reorder_tarr_snr()

# From here on out, don't change obs. Use obs_sc to track gain changes.
obs_sc = obs.copy()

# Add systematic noise for leakage (reminder: this must be done *after* any averaging)
for d in obs_sc.data:
    d[-4] = (d[-4]**2 + np.abs(sys_noise*d[-8])**2)**0.5
    d[-3] = (d[-3]**2 + np.abs(sys_noise*d[-8])**2)**0.5
    d[-2] = (d[-2]**2 + np.abs(sys_noise*d[-8])**2)**0.5
    d[-1] = (d[-1]**2 + np.abs(sys_noise*d[-8])**2)**0.5

######################################################################################################################
# Set up imaging and first round of self-cal
######################################################################################################################

# Helper function to repeat imaging with and without blurring to assure good convergence
def converge(major=5):
    for repeat in range(major):
        imgr.init_next = imgr.out_last().blur_circ(res)
        imgr.make_image_I(show_updates=False)

obs_sc = obs.copy()
## Make a Gaussian prior
#gaussprior    = eh.image.make_square(obs, npix, fov).add_gauss(zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))
#gaussprior = gaussprior.add_gauss(zbl*1e-6, (prior_fwhm, prior_fwhm, 0, prior_fwhm, prior_fwhm))

## Self calibrate the LMT to a Gaussian model 
#print("Self-calibrating the LMT to a Gaussian model...")
#gausspriorLMT = eh.image.make_square(obs, npix, fov).add_gauss(zbl, (LZ_gauss, LZ_gauss, 0, 0, 0))
#obs_sc = obs_sc.switch_polrep('stokes')
#for repeat in range(3):
#    caltab = eh.selfcal(obs_sc.flag_uvdist(uv_max=2e9), gausspriorLMT, sites=['LM','LM'], method='vis', ttype='nfft', processes=4, caltable=True, gain_tol=1.0)
#    obs_sc = caltab.applycal(obs_sc, interp='nearest', extrapolate=True)

# Store the initial dataset
obs_sc_init = obs_sc.copy()

######################################################################################################################
# Alternate rounds of imaging and self cal
######################################################################################################################

# Make an image -- with closure quantities and visibility amplitudes. Initializing with a Gaussian Image
print("Round 1 of imaging...")
imgr = eh.imager.Imager(obs_sc, gaussprior, prior_im=gaussprior, data_term=data_term, maxit=150, clipfloor=-1., norm_reg=True, systematic_noise=systematic_noise, reg_term = reg_term, ttype='nfft')
imgr.make_image_I(show_updates=False) # perform the first round of imaging
converge() #repeat steps of blurring and re-imaging to avoid local minima
im1 = imgr.out_last().copy() # Store this image for later reference

# Self calibrate to the previous model (phase-only)
obs_sc = eh.selfcal(obs_sc, im1, method='phase', ttype='nfft')

# Make an image -- with closure quantities and complex visibilities. Initializing with a Gaussian Image
print("Round 2 of imaging...")
imgr = eh.imager.Imager(obs_sc, gaussprior, prior_im=gaussprior, data_term={'vis':imgr.dat_terms_last()['amp']*10, 'cphase':imgr.dat_terms_last()['cphase']*10, 'logcamp':imgr.dat_terms_last()['logcamp']*10}, maxit=100, clipfloor=-1., norm_reg=True, systematic_noise=systematic_noise, reg_term = reg_term, ttype='nfft')
imgr.make_image_I(show_updates=False)
converge()
im2 = imgr.out_last().copy() # Store this image for later reference

# Self calibrate to the previous model (phase-only)
obs_sc = eh.selfcal(obs_sc, im2, method='phase', ttype='nfft')

######################################################################################################################
# Save out the data
######################################################################################################################

# save the final image
im2.display(export_pdf=pathtooutput + SCRIPTNAME + tag + '.pdf')
im2.save_fits(pathtooutput + SCRIPTNAME + tag + '.fits')

# save the gain plots
ct = eh.selfcal(obs, im2, method='both', ttype='nfft', processes=0, caltable=True)
ct.plot_gains(list(np.sort(list(ct.data.keys()))), yscale='log', export_pdf= pathtooutput + SCRIPTNAME + tag + '_gains.pdf')
ct = ct.pad_scans()
ct.save_txt(obs_sc, datadir=pathtooutput + SCRIPTNAME + '_caltable')

# save the final datasets that have been self calibrated 
obs_sc_save = ct.applycal(obs,interp='nearest',extrapolate=True)
obs_sc_save.save_uvfits(pathtooutput+ SCRIPTNAME + tag + '.uvfits')
obs.save_uvfits(pathtooutput + SCRIPTNAME + tag + '_uncal.uvfits')

# save out a summary pdf
os.system('python ' + pathtoscripts + '/imgsum.py ' + pathtooutput + SCRIPTNAME + tag + '.fits ' + pathtooutput + SCRIPTNAME + tag + '.uvfits ' + pathtooutput + SCRIPTNAME + tag + '_uncal.uvfits ' + '--no_ebar --o ' + pathtooutput )

######################################################################################################################
# Image polarization! 
######################################################################################################################

# Image polarization with the polarimetric ratio
if image_pol:

    #remove JCMT because of current issue in eht-imaging
    obs_sc_pol = obs_sc.flag_sites(['JC'])
    
    if realdata:
        # FIX FOR ER5 DATA! rotate polarization because of error in ALMA ... do not have to do this normally
        datadict = {t['site']:np.array([(0.0, 0.0 + 1j*1.0, 1.0 + 1j*0.0)], dtype=eh.DTCAL) for t in obs_sc_pol.tarr}
        caltab = eh.caltable.Caltable(obs_sc_pol.ra,obs_sc_pol.dec,obs_sc_pol.rf,obs_sc_pol.bw,datadict,obs_sc_pol.tarr,obs_sc_pol.source,obs_sc_pol.mjd)
        obs_sc_pol = caltab.applycal(obs_sc_pol, interp='nearest',extrapolate=True)


    print("Round 3 of imaging with polarization...")
    imgr.obs_next = obs_sc_pol
    imgr.init_next = im2.blur_circ(0.25*res)
    imgr.prior_next = imgr.init_next
    imgr.transform_next = 'mcv'
    imgr.dat_term_next = {'m':10}
    imgr.reg_term_next = {'hw':1}
    imgr.make_image_P()
    im4 = imgr.out_last().copy() # Store this image for later reference


    # blur and image again with the polarimetric ratio
    print("Round 4 of imaging with polarization...")
    imgr.obs_next = obs_sc_pol
    imgr.init_next = im4.blur_circ(0,.5*res)
    imgr.prior_next = imgr.init_next
    imgr.transform_next = 'mcv'
    imgr.dat_term_next = {'m':100}
    imgr.reg_term_next = {'hw':1,'ptv':1.e2} # have more tv regularization
    imgr.make_image_P()
    im5 = imgr.out_last().copy() # Store this image for later reference
    
    im5.display(plotp=True)
