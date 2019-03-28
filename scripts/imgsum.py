#!/usr/bin/env python

#imgsum.py
#Andrew Chael 07/12/2018
#produce an image summary plot for an image and uvfits file

from __future__ import print_function
import ehtim as eh
import numpy as np
import argparse
import os

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("inputim",help='/path/to/image.fits')
    parser.add_argument("inputobs",help='/path/to/uvfits.uvfits')
    parser.add_argument("inputobs_uncal",help='/path/to/uvfits_uncalibrated.uvfits')

    parser.add_argument('--c', '-c',type=str,default=" ", help="comments for top of sheet")
    parser.add_argument('--o','-o', type=str,help="path/to/output",default='.')
    parser.add_argument('--cp_uv_min', type=float,default=0, help="uv_min for closure phases")
    parser.add_argument('--aipscc', type=bool,default=False, help="load clean components from fits")
    parser.add_argument('--systematic_noise', type=float, default=0, help="systematic noise to add on amplitudes")
    parser.add_argument('--snrcut', type=float, default=0, help="snr cut")
    parser.add_argument('--systematic_cphase_noise', type=float, default=0,help="systematic noise to add on cphase")
    parser.add_argument('--fontsize', type=int, default=0,help="font size")
    parser.add_argument('--cfun', type=str, default='afmhot',help="image color function")
    parser.add_argument('--no_ebar', default=False,action='store_true',help="remove ebars from amp")
    parser.add_argument('--no_debias', default=False,action='store_true',help="don't debias amplitudes/closure amplitudes")
    parser.add_argument('--no_gains', default=False,action='store_true',help="remove gain plots")
    parser.add_argument('--no_cphase', default=False,action='store_true',help="remove closure phase plots")
    parser.add_argument('--no_camp', default=False,action='store_true',help="remove closure amp plots")
    parser.add_argument('--no_amp', default=False,action='store_true',help="remove closure amp plots")

    opt = parser.parse_args()
    print("Generating Image Summary PDF")
    print("===========================================")

    if opt.cp_uv_min==0: cp_uv_min=False
    else: cp_uv_min=opt.cp_uv_min
    if opt.aipscc: aipscc=True
    else: aipscc=False
    if opt.no_debias: debias=False
    else: debias=True
    if opt.no_ebar: ebar=False
    else: ebar=True
    if opt.no_gains: gainplots=False
    else: gainplots=True
    if opt.no_cphase: cphaseplots=False
    else: cphaseplots=True
    if opt.no_camp: campplots=False
    else: campplots=True
    if opt.no_amp: ampplots=False
    else: ampplots=True

    im = eh.image.load_fits(opt.inputim, aipscc=aipscc)
    obs = eh.obsdata.load_uvfits(opt.inputobs)
    obs_uncal = eh.obsdata.load_uvfits(opt.inputobs_uncal)

    basename = os.path.splitext(os.path.basename(opt.inputim))[0]
    outdir = str(opt.o)
    if outdir[-1] == '/': outname = outdir + basename + '.pdf'
    elif outdir[-3:] == 'pdf': outname = outdir
    else: outname = outdir +'/' + basename + '.pdf'

    args = [im, obs, obs_uncal, outname]
    kwargs = {'commentstr':opt.c, 'outdir':outdir,'ebar':ebar,'cfun':opt.cfun,'snrcut':snrcut,
              'sysnoise':opt.systematic_noise,'syscnoise':opt.systematic_cphase_noise,'fontsize':opt.fontsize,
              'gainplots':gainplots,'cphaseplots':cphaseplots,'campplots':campplots, 'ampplots':ampplots, 'debias':debias,  
              'cp_uv_min':cp_uv_min}

    eh.plotting.summary_plots.imgsum(*args, **kwargs)

