import vlbi_imaging_utils as vb
import maxen as mx
import numpy as np
import pulses
import patch_prior as pp
import linearize_energy as le


############# LOAD IMAGE AND OBSERVE ##############

# Load the image and the array
#im = vb.load_im_fits('./models/avery_sgra_eofn.fits') #for a fits image
im = vb.load_im_txt('./models/avery_sgra_eofn.txt') #for a text file
eht = vb.load_array('./arrays/EHT2017.txt') #see the attached array text file
im.pulse = pulses.trianglePulse2D

# Look at the image
#im.display(plotp=True)

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

obs = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, sgrscat=True, ampcal=False, phasecal=False)

############# IMAGING PARAMETERS ##############


# Generate an image prior
fov = im.xdim*im.psize #radians
flux = 2.0 # total flux

# define the full width half max of an initialization gaussian in radians
prior_fwhm = 150*vb.RADPERUAS 

# define the number of scales and the size of reconstructions
nScales = 10; 
sizeStart = 20; 
sizeFinal = 64; 


############ RUN OPTIMIZATION ###############

# determine the multiplier you use to increase the scale
if(nScales == 1):
    scaleFactor = 1
else:
    scaleFactor = np.exp((np.log(sizeFinal) - np.log(sizeStart))/(nScales-1));


# create the prior
emptyprior = vb.make_square(obs, sizeStart, fov)
emptyprior.pulse = pulses.trianglePulse2D
gaussprior = vb.add_gauss(emptyprior, flux, (prior_fwhm, prior_fwhm, 0, 0, 0))
gaussprior.pulse = pulses.trianglePulse2D

# initial image
cleanI = mx.maxen_bs(obs, gaussprior, gaussprior, flux, maxit=50, alpha=1e5, stop=1e-15)

# iterate through scales
for s in range(1,nScales+1):

    # resize the image
    sizeCurr = np.round(scaleFactor**(s-1)*sizeStart);
    cleanI = vb.resample_square(cleanI, sizeCurr)
    
    # solve for the image under different patch noise levels (beta) 
    for beta in (1.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0):
    
        print "scale: " + str(s) + " beta: " + str(beta)  
        
        patchprior, count = pp.patchPrior(cleanI, beta) #*1000.0
        #cleanI = mx.maxen_bs(obs, cleanI, patchprior, flux, maxit=50, alpha=10.0, gamma=500, delta=500, beta=beta*count, entropy="patch", datamin="lin")
        cleanI = le.linearizedSol_bs(obs, cleanI, patchprior, alpha=0.001, beta=beta*count, reg="patch") 

    cleanI.display()    
    

#########

# Save the images
outname = "test"
cleanI.save_txt(outname + '.txt')
cleanI.save_fits(outname + '.fits')
cleanI.save_txt(outname + '_blur.txt')
cleanI.save_fits(outname + '_blur.fits')


