import sys
import glob
import subprocess
from itertools import cycle
import matplotlib.pyplot as plt

sys.path.append("../../../fork-eht-imaging/eht-imaging/ehtim/plotting/")
# import comparisions as comp
import os
import ehtim as eh
# from "../fork-eht-imaging/eht-imaging/" import plotting.comparisons as comp
import comparisons as comp


#### global dictionaries containing parameters and system/command line arguments
# command line arguments
args = {
    'folderpath':sys.argv[1],
    'directory':sys.argv[2]
}

# comparison parameters
params = {
    'cutoff':0.95,
    'fov':1.0,
    'zoom':0.1
}

# list of dictionaries
dictionaries = {
    'file_args':args,
    'parameters':params
}

OPTIONS_MESSAGE = 'Press f/F to edit field of view, press c/C to edit cutoff, and press r/R to generate a graph. Press p/P to print values from all dictionaries.\n>>>'

def generate_metricmtx(fpath_, directory_):
    path = fpath_
    dirname = directory_

    # get the image names
    filenames = [x for x in os.listdir(path + dirname + '/images/') if x.endswith('.fits') ]

    # load the obs file
    obs = eh.obsdata.load_uvfits(path + dirname + '/' + dirname + '.uvfits')
    beamparams = obs.fit_beam()

    # load the images
    imarr = []
    for i in range(len(filenames)):
        imarr.append(eh.image.load_fits(path + dirname + '/images/' + filenames[i]))

    # do the image comparisions 
    (metric_mtx, fracsteps) = comp.image_consistency(imarr, beamparams, metric='nxcorr', blursmall=True, beam_max=1.0, beam_steps=5, savepath=[])

    return (metric_mtx, fracsteps, beamparams, imarr)

def generate_graph(metric_mtx, fracsteps, beamparams, imarr):
    print "Using settings: "
    print_dicts()
    (cliques_fraclevels, im_cliques_fraclevels) = comp.image_agreements(imarr, beamparams, metric_mtx, fracsteps, cutoff=params['cutoff'])
    comp.generate_consistency_plot(cliques_fraclevels, im_cliques_fraclevels, zoom=params['zoom'], fov=params['fov'], show=True)

def print_dicts():
    global dictionaries
    for gk, gv in dictionaries.iteritems():
        for key, value in gv.iteritems() :
            print key, value

#### CLI function
# before any customization can be allowed, the metric_mtx must be generated.
# these must stay constants
print_dicts()
(METRIC_MTX, FRACSTEPS, BEAMPARAMS, IMARR) = generate_metricmtx(args['folderpath'], args['directory'])

while 1:
    user_input = raw_input(OPTIONS_MESSAGE).lower()
    if user_input in ['p', 'P']:
        print_dicts()

    if user_input in ['f', 'F']:
        new_fov = -1
        while new_fov < 0.0 or new_fov > 1.0:
            new_fov_message = "Please provide a new field of view (decimal form!) (between 0 and 1)\n>>> "
            new_fov = input(new_fov_message)
        params['fov'] = new_fov
        continue

    elif user_input in ['c', 'C']:
        new_cutoff = -1
        while new_cutoff < 0.0 or new_cutoff > 1.0:
            new_cutoff_message = "Please provide a new cutoff (decimal form!) (between 0 and 1)\n>>> "
            new_cutoff = input(new_cutoff_message)
        params['cutoff'] = new_cutoff
        continue

    elif user_input in ['z', 'Z']:
        new_zoom = -1
        while new_zoom < 0.0 or new_zoom > 1.0:
            new_zoom_message = "Please provide a new zoom (decimal form!) (between 0 and 1)\n>>> "
            new_zoom = input(new_zoom_message)
        params['zoom'] = new_zoom
        continue

    elif user_input in ['r', 'R']:
        print 'Generating the graph...'
        generate_graph(METRIC_MTX, FRACSTEPS, BEAMPARAMS, IMARR)



    
