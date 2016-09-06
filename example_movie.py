#How to load and observe a movie
#Note that the time-related functions are all done by hand and are imprecise
#Will fix to use astropy time functions

import numpy as np
import vlbi_imaging_utils as vb
import movie_utils as mv

nframes = 4280 #number of fames
framedur_sec = 11.8 #seconds per frame
movielen_hr = framedur_sec * nframes / 3600.0 # hours in the movie

tstart_hr = 8.0
tstop_hr = tstart_hr + movielen_hr
tint_sec = 60.0
tadv_sec = 60.0
bw_hz = 4e9

# Load the movie - this will take a while! Not sure why yet!
mov = mv.load_movie_txt('./movie/a9_face/', nframes, framedur_sec)

# I frames are in U column in current movie text files !!!
mov.frames = mov.uframes

# Change the mjd of the movie start to correspond with the desired observation start
mov.mjd = np.floor(mov.mjd) + tstart_hr/24.0

# Load the array
eht = vb.load_array('./arrays/EHT2017.txt')

# Observe the movie
obs = mov.observe(eht, tint_sec, tadv_sec, tstart_hr, .999*tstop_hr, bw_hz) 

# You can then plot and save to file as normal
obs.plotall('u','v', conj=True) # uv coverage
obs.plotall('uvdist','amp') # amplitude with baseline distance'
obs.plot_bl('SMA','ALMA','phase') # visibility phase on a baseline over time
obs.plot_cphase('SMA', 'SMT', 'ALMA') # closure phase 1-2-3 on a over time
obs.plot_camp('ALMA','LMT','SMA','SPT') # closure amplitude (1-2)(3-4)/(1-4)(2-3) over time

# You can get lists of closure phases and amplitudes and save them to a file
#cphases = obs.c_phases(mode='all', count='max') # set count='min' to return a minimal set
#camps = obs.c_amplitudes(mode='all', count='max') # set count='min' to return a minimal set
#np.savetxt('./c_phases.txt',cphases)
#np.savetxt('./c_amplitudes.txt',camps)
