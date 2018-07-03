import ehtim as eh

# Load the image and the array
im = eh.image.load_txt('../models/avery_sgra_eofn.txt')
eht = eh.array.load_txt('../arrays/EHT2017.txt')

# Observe the image
tint_sec = 30
tadv_sec = 1200*3
tstart_hr = 0
tstop_hr = 24
bw_hz = 4e9
obs = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz,
                 sgrscat=False, ampcal=False, phasecal=False)

# Test without multiprocessing
obs_nc = eh.calibrating.network_cal.network_cal(obs, im.total_flux(), processes=-1,msgtype='bh')
obs_sc = eh.calibrating.self_cal.self_cal(obs, im, processes=-1)

eh.comp_plots.plot_bl_obs_im_compare([obs,obs_nc,obs_sc],im,'SMA','ALMA','amp')
eh.comp_plots.plotall_obs_im_compare([obs,obs_nc,obs_sc],im,'uvdist','amp')
eh.comp_plots.plotall_obs_im_compare([obs,obs_nc,obs_sc],im,'uvdist','phase')

# Test with multiprocessing
obs_nc = eh.calibrating.network_cal.network_cal(obs, im.total_flux(), processes=0,msgtype='bh')
obs_sc = eh.calibrating.self_cal.self_cal(obs, im, processes=0)

eh.comp_plots.plot_bl_obs_im_compare([obs,obs_nc,obs_sc],im,'SMA','ALMA','amp')
eh.comp_plots.plotall_obs_im_compare([obs,obs_nc,obs_sc],im,'uvdist','amp')
eh.comp_plots.plotall_obs_im_compare([obs,obs_nc,obs_sc],im,'uvdist','phase')
