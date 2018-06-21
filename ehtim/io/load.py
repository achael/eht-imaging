# load.py
# functions to load observation & image data from files
#
#    Copyright (C) 2018 Andrew Chael
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range

import numpy as np
import string
import astropy.io.fits as fits
import astropy.time
import datetime
import os
import copy
import sys
import time as ttime

import ehtim.obsdata
import ehtim.image
import ehtim.array
import ehtim.movie
import ehtim.vex

import ehtim.io.oifits
from astropy.time import Time
from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")

##################################################################################################
# Vex IO
##################################################################################################
def load_vex(fname):
    """Read in .vex files. and function to observe them
       Assumes there is only 1 MODE in vex file
       Hotaka Shiokawa - 2017
    """
    print ("\nLoading vexfile: ", fname)
    return ehtim.vex.Vex(fname)


##################################################################################################
# Image IO
##################################################################################################
def load_im_txt(filename, pulse=PULSE_DEFAULT):
    """Read in an image from a text file and create an Image object
       Text file should have the same format as output from Image.save_txt()
       Make sure the header has exactly the same form!
    """

    print ("\nLoading text image: ", filename)

    # Read the header
    file = open(filename)
    src = ' '.join(file.readline().split()[2:])
    ra = file.readline().split()
    ra = float(ra[2]) + float(ra[4])/60.0 + float(ra[6])/3600.0
    dec = file.readline().split()
    dec = np.sign(float(dec[2])) *(abs(float(dec[2])) + float(dec[4])/60.0 + float(dec[6])/3600.0)
    mjd_float = float(file.readline().split()[2])
    mjd = int(mjd_float)
    time = (mjd_float - mjd) * 24
    rf = float(file.readline().split()[2]) * 1e9
    xdim = file.readline().split()
    xdim_p = int(xdim[2])
    psize_x = float(xdim[4])*RADPERAS/xdim_p
    ydim = file.readline().split()
    ydim_p = int(ydim[2])
    psize_y = float(ydim[4])*RADPERAS/ydim_p
    file.close()

    if psize_x != psize_y:
        raise Exception("Pixel dimensions in x and y are inconsistent!")

    # Load the data, convert to list format, make object
    datatable = np.loadtxt(filename, dtype=float)
    image = datatable[:,2].reshape(ydim_p, xdim_p)
    outim = ehtim.image.Image(image, psize_x, ra, dec, rf=rf, source=src, mjd=mjd, time=time, pulse=pulse)

    # Look for Stokes Q and U
    qimage = uimage = vimage = np.zeros(image.shape)
    if datatable.shape[1] == 6:
        qimage = datatable[:,3].reshape(ydim_p, xdim_p)
        uimage = datatable[:,4].reshape(ydim_p, xdim_p)
        vimage = datatable[:,5].reshape(ydim_p, xdim_p)
    elif datatable.shape[1] == 5:
        qimage = datatable[:,3].reshape(ydim_p, xdim_p)
        uimage = datatable[:,4].reshape(ydim_p, xdim_p)

    if np.any((qimage != 0) + (uimage != 0)) and np.any((vimage != 0)):
        #print('Loaded Stokes I, Q, U, and V Images')
        outim.add_qu(qimage, uimage)
        outim.add_v(vimage)
    elif np.any((vimage != 0)):
        #print('Loaded Stokes I and V Images')
        outim.add_v(vimage)
    elif np.any((qimage != 0) + (uimage != 0)):
        #print('Loaded Stokes I, Q, and U Images')
        outim.add_qu(qimage, uimage)
    else:
        pass
        #print('Loaded Stokes I Image Only')

    return outim

def load_im_fits(filename, punit="deg", pulse=PULSE_DEFAULT):
    """Read in an image from a FITS file and create an Image object
    """
    print ("\nLoading fits image: ", filename)

    # Radian or Degree?
    if punit=="deg":
        pscl = DEGREE
    elif punit=="rad":
        pscl = 1.0
    elif punit=="uas":
        pscl = RADPERUAS
    elif punit=="mas":
        pscl = RADPERUAS * 1000.0

    # Open the FITS file
    hdulist = fits.open(filename)

    # Assume stokes I is the primary hdu
    header = hdulist[0].header

    # Read some header values
    xdim_p = header['NAXIS1']
    psize_x = np.abs(header['CDELT1']) * pscl
    dim_p = header['NAXIS2']
    psize_y = np.abs(header['CDELT2']) * pscl

    if 'OBSRA' in list(header.keys()): ra = header['OBSRA']*12/180.
    elif 'CRVAL1' in list(header.keys()): ra = header['CRVAL1']*12/180.
    else: ra = 0.

    if 'OBSDEC' in list(header.keys()):  dec = header['OBSDEC']
    elif 'CRVAL2' in list(header.keys()):  dec = header['CRVAL2']
    else: dec = 0.

    if 'FREQ' in list(header.keys()): rf = header['FREQ']
    elif 'CRVAL3' in list(header.keys()): rf = header['CRVAL3']
    else: rf = 0.

    if 'MJD' in list(header.keys()): mjd_float = header['MJD']
    else: mjd_float = 0.
    mjd = int(mjd_float)
    time = (mjd_float - mjd) * 24

    if 'OBJECT' in list(header.keys()): src = header['OBJECT']
    else: src = ''

    # Get the image and create the object
    data = hdulist[0].data

    # Check for multiple stokes in top hdu
    stokes_in_hdu0 = False
    if len(data.shape) == 4:
        print("reading stokes images from top HDU -- assuming IQUV")
        stokesdata = data
        data = stokesdata[0,0]
        stokes_in_hdu0 = True

    data = data.reshape((data.shape[-2],data.shape[-1]))
    image = data[::-1,:] # flip y-axis!

    # normalize the flux
    normalizer = 1.0;
    if 'BUNIT' in list(header.keys()):
        if header['BUNIT'].lower() == 'JY/BEAM'.lower():
            print("converting Jy/Beam --> Jy/pixel")
            beamarea = (2.0*np.pi*header['BMAJ']*header['BMIN']/(8.0*np.log(2)))
            normalizer = (header['CDELT2'])**2/beamarea
    image *= normalizer

    # make image object
    outim = ehtim.image.Image(image, psize_x, ra, dec, rf=rf, source=src, mjd=mjd, time=time, pulse=pulse)

    # Look for Stokes Q and U
    qimage = uimage = vimage = np.array([])

    if stokes_in_hdu0: #stokes in top HDU
            try:
                qdata = stokesdata[1,0].reshape((data.shape[-2],data.shape[-1]))
                qimage = normalizer*qdata[::-1,:] # flip y-axis!
            except IndexError: pass
            try:
                udata = stokesdata[2,0].reshape((data.shape[-2],data.shape[-1]))
                uimage = normalizer*udata[::-1,:] # flip y-axis!
            except IndexError: pass
            try:
                vdata = stokesdata[3,0].reshape((data.shape[-2],data.shape[-1]))
                vimage = normalizer*vdata[::-1,:] # flip y-axis!
            except IndexError: pass

    else: #stokes in different HDUS
        for hdu in hdulist[1:]:
            header = hdu.header
            data = hdu.data
            try: data = data.reshape((data.shape[-2],data.shape[-1]))
            except IndexError: continue

            if 'STOKES' in list(header.keys()) and header['STOKES'] == 'Q':
                qimage = normalizer*data[::-1,:] # flip y-axis!
            if 'STOKES' in list(header.keys()) and header['STOKES'] == 'U':
                uimage = normalizer*data[::-1,:] # flip y-axis!
            if 'STOKES' in list(header.keys()) and header['STOKES'] == 'V':
                vimage = normalizer*data[::-1,:] # flip y-axis!

    if qimage.shape == uimage.shape == vimage.shape == image.shape:
        #print('Loaded Stokes I, Q, U, and V Images')
        outim.add_qu(qimage, uimage)
        outim.add_v(vimage)
    elif vimage.shape == image.shape:
        #print('Loaded Stokes I and V Images')
        outim.add_v(vimage)
    elif qimage.shape == uimage.shape == image.shape:
        #print('Loaded Stokes I, Q, and U Images')
        outim.add_qu(qimage, uimage)
    else:
        pass
        #print('Loaded Stokes I Image Only')

    return outim


##################################################################################################
# Movie IO
##################################################################################################

def load_movie_hdf5(file_name, framedur_sec=-1, psize=-1,
                    ra=17.761122472222223, dec=-28.992189444444445, rf=230e9, pulse=PULSE_DEFAULT):
    """Read in a movie from a hdf5 file and create a Movie object
       file_name should be the name of the hdf5 file
       thisdoes not use the header of the hdf5 file so you need to give it
       psize, framedur_sec, ra and dec
    """

    import h5py
    file    = h5py.File(file_name, 'r')
    name    = list(file.keys())[0]
    d       = file[str(name)]
    sim  = d[:]
    file.close()
    return Movie(sim, framedur_sec, psize, ra, dec, rf)


def load_movie_txt(basename, nframes, framedur=-1, pulse=PULSE_DEFAULT):
    """Read in a movie from text files and create a Movie object
       Text files should be filename + 00001, etc.
       Text files should have the same format as output from Image.save_txt()
       Make sure the header has exactly the same form!
    """

    imlist = []

    for i in range(nframes):
        filename = basename + "%05d" % i

        sys.stdout.write('\rReading Movie Image %i/%i...' % (i,nframes))
        sys.stdout.flush()

        im = load_im_txt(filename, pulse=pulse)
        imlist.append(im)

        hour = im.time
        if i == 0:
            hour0 = im.time
        else:
            pass

    if framedur == -1:
        framedur = ((hour - hour0)/float(nframes))*3600.0

    out_mov = ehtim.movie.merge_im_list(imlist, framedur=framedur)

    return out_mov


def load_movie_fits(basename, nframes, framedur=-1, pulse=PULSE_DEFAULT):
    """Read in a movie from fits files and create a Movie object
       Fits files should be filename + 00001, etc.
    """

    imlist = []

    for i in range(nframes):
        filename = basename + "%05d" % i

        sys.stdout.write('\rReading Movie Image %i/%i...' % (i,nframes))
        sys.stdout.flush()

        im = load_im_fits(filename, pulse=pulse)
        imlist.append(im)

        hour = im.time
        if i == 0:
            hour0 = im.time
        else:
            pass

    if framedur == -1:
        framedur = ((hour - hour0)/float(nframes))*3600.0

    out_mov = ehtim.movie.merge_im_list(imlist, framedur=framedur)

    return out_mov


##################################################################################################
# Array IO
##################################################################################################
def load_array_txt(filename, ephemdir='ephemeris'):
    """Read an array from a text file and return an Array object
       Sites with x=y=z=0 are spacecraft - 2TLE ephemeris loaded from ephemdir
    """

    tdata = np.loadtxt(filename,dtype=bytes,comments='#').astype(str)
    path = os.path.dirname(filename)

    tdataout = []
    if (tdata.shape[1] != 5 and tdata.shape[1] != 13):
        raise Exception("Array file should have format: "+
                        "(name, x, y, z, SEFDR, SEFDL "+
                        "FR_PAR_ANGLE FR_ELEV_ANGLE FR_OFFSET" +
                        "DR_RE   DR_IM   DL_RE    DL_IM )")

    elif tdata.shape[1] == 5:
    	tdataout = [np.array((x[0],float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[4]),
                              0.0, 0.0,
                              0.0, 0.0, 0.0),
                             dtype=DTARR) for x in tdata]
    elif tdata.shape[1] == 13:
    	tdataout = [np.array((x[0],float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5]),
                              float(x[9])+1j*float(x[10]), float(x[11])+1j*float(x[12]),
                              float(x[6]), float(x[7]), float(x[8])),
                             dtype=DTARR) for x in tdata]

    tdataout = np.array(tdataout)
    edata = {}
    for line in tdataout:
        if np.all(np.array([line['x'],line['y'],line['z']]) == (0.,0.,0.)):
            sitename = str(line['site'])
            ephempath = path  + '/' + ephemdir + '/' + sitename #TODO ephempath shouldn't always start with path
            try:
                edata[sitename] = np.loadtxt(ephempath, dtype=bytes, comments='#', delimiter='/').astype(str)
                print('loaded spacecraft ephemeris %s' % ephempath)
            except IOError:
                raise Exception ('no ephemeris file %s !' % ephempath)

    return ehtim.array.Array(tdataout, ephem=edata)

##################################################################################################
# Observation IO
##################################################################################################

def load_obs_txt(filename):
    """Read an observation from a text file and return an Obsdata object
       text file has the same format as output from Obsdata.savedata()
    """
    print ("\nLoading text observation: ", filename)

    # Read the header parameters
    file = open(filename)
    src = ' '.join(file.readline().split()[2:])
    ra = file.readline().split()
    ra = float(ra[2]) + float(ra[4])/60.0 + float(ra[6])/3600.0
    dec = file.readline().split()
    dec = np.sign(float(dec[2])) *(abs(float(dec[2])) + float(dec[4])/60.0 + float(dec[6])/3600.0)
    mjd = float(file.readline().split()[2])
    rf = float(file.readline().split()[2]) * 1e9
    bw = float(file.readline().split()[2]) * 1e9
    phasecal = bool(file.readline().split()[2])
    ampcal = bool(file.readline().split()[2])

    # New Header Parameters
    x = file.readline().split()
    if x[1] == 'OPACITYCAL:':
        opacitycal = bool(x[2])
        dcal = bool(file.readline().split()[2])
        frcal = bool(file.readline().split()[2])
        file.readline()
    else:
        opacitycal = True
        dcal = True
        frcal = True
    file.readline()

    # read the tarr
    line = file.readline().split()
    tarr = []
    while line[1][0] != "-":
        if len(line) == 6:
        	tarr.append(np.array((line[1], line[2], line[3], line[4], line[5], line[5], 0, 0, 0, 0, 0), dtype=DTARR))
        elif len(line) == 14:
        	tarr.append(np.array((line[1], line[2], line[3], line[4], line[5], line[6],
        	                      float(line[10])+1j*float(line[11]), float(line[12])+1j*float(line[13]),
        	                      line[7], line[8], line[9]), dtype=DTARR))
        else: raise Exception("Telescope header doesn't have the right number of fields!")
        line = file.readline().split()
    tarr = np.array(tarr, dtype=DTARR)
    file.close()

    # Load the data, convert to list format, return object
    datatable = np.loadtxt(filename, dtype=bytes).astype(str)
    datatable2 = []
    for row in datatable:
        time = float(row[0])
        tint = float(row[1])
        t1 = row[2]
        t2 = row[3]

        #Old datatable formats
        if datatable.shape[1] < 20:
            tau1 = float(row[6])
            tau2 = float(row[7])
            u = float(row[8])
            v = float(row[9])
            vis = float(row[10]) * np.exp(1j * float(row[11]) * DEGREE)
            if datatable.shape[1] == 19:
                qvis = float(row[12]) * np.exp(1j * float(row[13]) * DEGREE)
                uvis = float(row[14]) * np.exp(1j * float(row[15]) * DEGREE)
                vvis = float(row[16]) * np.exp(1j * float(row[17]) * DEGREE)
                sigma = qsigma = usigma = vsigma = float(row[18])
            elif datatable.shape[1] == 17:
                qvis = float(row[12]) * np.exp(1j * float(row[13]) * DEGREE)
                uvis = float(row[14]) * np.exp(1j * float(row[15]) * DEGREE)
                vvis = 0+0j
                sigma = qsigma = usigma = vsigma = float(row[16])
            elif datatable.shape[1] == 15:
                qvis = 0+0j
                uvis = 0+0j
                vvis = 0+0j
                sigma = qsigma = usigma = vsigma = float(row[12])
            else:
                raise Exception('Text file does not have the right number of fields!')

        # Current datatable format
        elif datatable.shape[1] == 20:
            tau1 = float(row[4])
            tau2 = float(row[5])
            u = float(row[6])
            v = float(row[7])
            vis = float(row[8]) * np.exp(1j * float(row[9]) * DEGREE)
            qvis = float(row[10]) * np.exp(1j * float(row[11]) * DEGREE)
            uvis = float(row[12]) * np.exp(1j * float(row[14]) * DEGREE)
            vvis = float(row[14]) * np.exp(1j * float(row[15]) * DEGREE)
            sigma = float(row[16])
            qsigma = float(row[17])
            usigma = float(row[18])
            vsigma = float(row[19])

        else:
            raise Exception('Text file does not have the right number of fields!')


        datatable2.append(np.array((time, tint, t1, t2, tau1, tau2,
                                    u, v, vis, qvis, uvis, vvis,
                                    sigma, qsigma, usigma, vsigma), dtype=DTPOL))

    # Return the data object
    datatable2 = np.array(datatable2)
    out =  ehtim.obsdata.Obsdata(ra, dec, rf, bw, datatable2, tarr, source=src, mjd=mjd,
                                 ampcal=ampcal, phasecal=phasecal, opacitycal=opacitycal, dcal=dcal, frcal=frcal)
    return out

def load_obs_maps(arrfile, obsspec, ifile, qfile=0, ufile=0, vfile=0, src=SOURCE_DEFAULT, mjd=MJD_DEFAULT, ampcal=False, phasecal=False):
    """Read an observation from a maps text file and return an Obsdata object
    """
    # Read telescope parameters from the array file
    tdata = np.loadtxt(arrfile, dtype=bytes).astype(str)
    tdata = [np.array((x[0],float(x[1]),float(x[2]),float(x[3]),float(x[-1]), 0., 0., 0., 0., 0.), dtype=DTARR) for x in tdata]
    tdata = np.array(tdata)

    # Read parameters from the obs_spec
    f = open(obsspec)
    stop = False
    while not stop:
        line = f.readline().split()
        if line==[] or line[0]=='\\':
            continue
        elif line[0] == 'FOV_center_RA':
            x = line[2].split(':')
            ra = float(x[0]) + float(x[1])/60.0 + float(x[2])/3600.0
        elif line[0] == 'FOV_center_Dec':
            x = line[2].split(':')
            dec = np.sign(float(x[0])) * (abs(float(x[0])) + float(x[1])/60.0 + float(x[2])/3600.0)
        elif line[0] == 'Corr_int_time':
            tint = float(line[2])
        elif line[0] == 'Corr_chan_bw':  #TODO what if multiple channels?
            bw = float(line[2]) * 1e6 #in MHz
        elif line[0] == 'Channel': #TODO what if multiple scans with different params?
            rf = float(line[2].split(':')[0]) * 1e6
        elif line[0] == 'Scan_start':
            x = line[2].split(':') #TODO properly compute MJD!
        elif line[0] == 'Endscan':
            stop=True
    f.close()

    # Load the data, convert to list format, return object
    datatable = []
    f = open(ifile)

    for line in f:
        line = line.split()
        if not (line[0] in ['UV', 'Scan','\n']):
            time = line[0].split(':')
            time = float(time[2]) + float(time[3])/60.0 + float(time[4])/3600.0
            u = float(line[1]) * 1000
            v = float(line[2]) * 1000
            bl = line[4].split('-')
            t1 = tdata[int(bl[0])-1]['site']
            t2 = tdata[int(bl[1])-1]['site']
            tau1 = 0.
            tau2 = 0.
            vis = float(line[7][:-1]) * np.exp(1j*float(line[8][:-1])*DEGREE)
            sigma = float(line[10])
            datatable.append(np.array((time, tint, t1, t2, tau1, tau2,
                                        u, v, vis, 0.0, 0.0, 0.0,
                                        sigma, 0.0, 0.0, 0.0), dtype=DTPOL))

    datatable = np.array(datatable)

    #TODO qfile ufile and vfile must have exactly the same format as ifile: add some consistency check
    if not qfile==0:
        f = open(qfile)
        i = 0
        for line in f:
            line = line.split()
            if not (line[0] in ['UV', 'Scan','\n']):
                datatable[i]['qvis'] = float(line[7][:-1]) * np.exp(1j*float(line[8][:-1])*DEGREE)
                datatable[i]['qsigma'] = float(line[10])
                i += 1

    if not ufile==0:
        f = open(ufile)
        i = 0
        for line in f:
            line = line.split()
            if not (line[0] in ['UV', 'Scan','\n']):
                datatable[i]['uvis'] = float(line[7][:-1]) * np.exp(1j*float(line[8][:-1])*DEGREE)
                datatable[i]['usigma'] = float(line[10])
                i += 1

    if not vfile==0:
        f = open(vfile)
        i = 0
        for line in f:
            line = line.split()
            if not (line[0] in ['UV', 'Scan','\n']):
                datatable[i]['vvis'] = float(line[7][:-1]) * np.exp(1j*float(line[8][:-1])*DEGREE)
                datatable[i]['vsigma'] = float(line[10])
                i += 1

    # Return the datatable
    return ehtim.obsdata.Obsdata(ra, dec, rf, bw, datatable, tdata, source=src, mjd=mjd)


#TODO can we save new telescope array terms and flags to uvfits and load them?
def load_obs_uvfits(filename, flipbl=False, force_singlepol=None, channel=all, IF=all):
    """Load uvfits data from a uvfits file.
       To read a single polarization (e.g., only RR) from a full polarization file, set force_singlepol='R', 'L', 'RL', or 'LR'
    """

    # Load the uvfits file
    print ("\nLoading uvfits: ", filename)
    hdulist = fits.open(filename)
    header = hdulist[0].header
    data = hdulist[0].data

    # Load the array data
    tnames = hdulist['AIPS AN'].data['ANNAME']
    tnums = hdulist['AIPS AN'].data['NOSTA'] - 1
    xyz = np.real(hdulist['AIPS AN'].data['STABXYZ'])
    try:
        sefdr = np.real(hdulist['AIPS AN'].data['SEFD'])
        sefdl = np.real(hdulist['AIPS AN'].data['SEFD']) #TODO add sefdl to uvfits?
    except KeyError:
        #print("Warning! no SEFD data in UVfits file")
        sefdr = np.zeros(len(tnames))
        sefdl = np.zeros(len(tnames))

    #TODO - get the *actual* values of these telescope parameters
    fr_par = np.zeros(len(tnames))
    fr_el = np.zeros(len(tnames))
    fr_off = np.zeros(len(tnames))
    dr = np.zeros(len(tnames)) + 1j*np.zeros(len(tnames))
    dl = np.zeros(len(tnames)) + 1j*np.zeros(len(tnames))

    tarr = [np.array((
            str(tnames[i]), xyz[i][0], xyz[i][1], xyz[i][2],
            sefdr[i], sefdl[i], dr[i], dl[i],
            fr_par[i], fr_el[i], fr_off[i]),
            dtype=DTARR) for i in range(len(tnames))]

    tarr = np.array(tarr)

    # Various header parameters
    try:
        ra = header['OBSRA'] * 12./180.
        dec = header['OBSDEC']
    except KeyError:
        if header['CTYPE6'] == 'RA':
            ra = header['CRVAL6'] * 12./180.
        else: raise Exception('Cannot find RA!')
        if header['CTYPE7'] == 'DEC':
            dec = header['CRVAL7']
        else: raise Exception('Cannot find DEC!')

    src = header['OBJECT']
    rf = hdulist['AIPS AN'].header['FREQ']

    if header['CTYPE4'] == 'FREQ':
        ch1_freq = header['CRVAL4']
        ch_bw = header['CDELT4']
        nchan = header['NAXIS4']

    else: raise Exception('Cannot find observing frequencies!')

    nif = 1
    try:
        if header['CTYPE5'] == 'IF':
            nif = header['NAXIS5']
    except KeyError:
        print ("no IF in uvfits header!")

    #determine the bandwidth
    bw = ch_bw * nchan * nif

    # Determine the number of correlation products in the data
    num_corr = data['DATA'].shape[5]
    print("Number of uvfits Correlation Products:",num_corr)
    if num_corr == 1 and force_singlepol != None:
        print("Cannot force single polarization when file is not full polarization.")
        force_singlepol = None

    # Mask to screen bad data
    # Reducing to single frequency

    # prepare the arrays of if and channels that will be extracted from the data.
    nvis = data['DATA'].shape[0]
    full_nchannels = data['DATA'].shape[4]
    full_nifs = data['DATA'].shape[3]
    if channel == all:
        channel = np.arange(0, full_nchannels, 1)
        nchannels = full_nchannels
    else:
        try:
            nchannels = len(np.array(channel))
            channel = np.array(channel).reshape(-1)
        except TypeError:
            channel = np.array([channel]).reshape(-1)
            nchannels = len(np.array(channel))

    if IF == all:
        IF = np.arange(0, full_nifs, 1)
        nifs =  full_nifs
    else:
        try:
            nifs =  len(IF)
            IF = np.array(IF).reshape(-1)
        except TypeError:
            IF = np.array([IF]).reshape(-1)
            nifs = len(np.array(IF))

    if (np.max(channel) >= full_nchannels) or (np.min(channel) < 0):
        raise Exception('The specified channel does not exist')
    if (np.max(IF) >= full_nifs) or (np.min(IF) < 0):
        raise Exception('The specified IF does not exist')


    #TODO CHECK THESE DECISIONS CAREFULLY!!!!
    rrweight = data['DATA'][:,0,0,IF,channel,0,2].reshape(nvis, nifs, nchannels)
    if num_corr >= 2:
        llweight = data['DATA'][:,0,0,IF,channel,1,2].reshape(nvis, nifs, nchannels)
    else:
        llweight = rrweight * 0.0
    if num_corr >= 3:
        rlweight = data['DATA'][:,0,0,IF,channel,2,2].reshape(nvis, nifs, nchannels)
    else:
        rlweight = rrweight * 0.0
    if num_corr >= 4:
        lrweight = data['DATA'][:,0,0,IF,channel,3,2].reshape(nvis, nifs, nchannels)
    else:
        lrweight = rrweight * 0.0

    # If necessary, enforce single polarization
    if force_singlepol == 'L':
        rrweight = rrweight * 0.0
        rlweight = rlweight * 0.0
        lrweight = lrweight * 0.0
    elif force_singlepol == 'R':
        llweight = llweight * 0.0
        rlweight = rlweight * 0.0
        lrweight = lrweight * 0.0
    elif force_singlepol == 'LR':
        print('WARNING: Putting LR data in Stokes I')
        rrweight = copy.deepcopy(lrweight)
        llweight = llweight * 0.0
        rlweight = rlweight * 0.0
        lrweight = lrweight * 0.0
    elif force_singlepol == 'RL':
        print('WARNING: Putting RL data in Stokes I')
        rrweight = copy.deepcopy(rlweight)
        llweight = llweight * 0.0
        rlweight = rlweight * 0.0
        lrweight = lrweight * 0.0

    # first, catch  nans
    rrnanmask_2d = (np.isnan(rrweight))
    llnanmask_2d = (np.isnan(llweight))
    rlnanmask_2d = (np.isnan(rlweight))
    lrnanmask_2d = (np.isnan(lrweight))

    rrweight[rrnanmask_2d] = 0.
    llweight[llnanmask_2d] = 0.
    rlweight[rlnanmask_2d] = 0.
    lrweight[lrnanmask_2d] = 0.

    #look for weights < 0 
    rrmask_2d = (rrweight > 0.)
    llmask_2d = (llweight > 0.)
    rlmask_2d = (rlweight > 0.)
    lrmask_2d = (lrweight > 0.)


    # if there is any unmasked data in the frequency column, use it
    rrmask = np.any(np.any(rrmask_2d, axis=2), axis=1)
    llmask = np.any(np.any(llmask_2d, axis=2), axis=1)
    rlmask = np.any(np.any(rlmask_2d, axis=2), axis=1)
    lrmask = np.any(np.any(rlmask_2d, axis=2), axis=1)

    # Total intensity mask
    # TODO or or and here? - what if we have only 1 of rr, ll?
    mask = rrmask + llmask

    if not np.any(mask):
        raise Exception("No unflagged RR or LL data in uvfits file!")
    if np.any(~(rrmask*llmask)):
        print("Warning: removing flagged data present!")

    # Obs Times
    jds = data['DATE'][mask].astype('d') + data['_DATE'][mask].astype('d')
    mjd = int(np.min(jds)-2400000.5)
    times = (jds - 2400000.5 - mjd) * 24.0

    try:
        scantable = []
        nxtable = hdulist['AIPS NX']
        for scan in nxtable.data:

            reftime = astropy.time.Time(hdulist['AIPS AN'].header['RDATE'], format='isot', scale='utc').jd
            #scantime = (scan['TIME'] + reftime  - mjd)
            scan_start = scan['TIME'] #in days since reference date
            scan_dur = scan['TIME INTERVAL']
            startvis = scan['START VIS'] - 1
            endvis = scan['END VIS'] - 1
            #scantable.append(np.array((scantime, scanint, startvis, endvis), dtype=DTSCANS))
            scantable.append([scan_start - 0.5*scan_dur,
                              scan_start + 0.5*scan_dur])

        scantable = np.array(scantable*24)

    except:
        print("No NX table in uvfits!")
        scantable = None

    # Integration times
    try:
        tints = data['INTTIM'][mask]
    except KeyError:
        tints = np.zeros(len(mask))
    # Sites - add names
    t1 = data['BASELINE'][mask].astype(int)//256
    t2 = data['BASELINE'][mask].astype(int) - t1*256
    t1 = t1 - 1
    t2 = t2 - 1
    scopes_num = np.sort(list(set(np.hstack((t1,t2)))))
    t1 = np.array([tarr[i]['site'] for i in t1])
    t2 = np.array([tarr[i]['site'] for i in t2])

    # Opacities (not in standard files)
    try:
        tau1 = data['TAU1'][mask]
        tau2 = data['TAU2'][mask]
    except KeyError:
        tau1 = tau2 = np.zeros(len(t1))

    # Convert uv in lightsec to lambda by multiplying by rf
    try:
        u = data['UU---SIN'][mask] * rf
        v = data['VV---SIN'][mask] * rf
    except KeyError:
        try:
            u = data['UU'][mask] * rf
            v = data['VV'][mask] * rf
        except KeyError:
            try:
                u = data['UU--'][mask] * rf
                v = data['VV--'][mask] * rf
            except KeyError:
                raise Exception("Cant figure out column label for UV coords")

    # Get and average visibility data
    # replace masked vis with nans so they don't mess up the average
    #TODO: coherent average ok?
    #TODO 2d or 1d mask
    rr_2d = data['DATA'][:,0,0,IF,channel,0,0] + 1j*data['DATA'][:,0,0,IF,channel,0,1]
    rr_2d = rr_2d.reshape(nvis, nifs, nchannels)
    if num_corr >= 2:
        ll_2d = data['DATA'][:,0,0,IF,channel,1,0] + 1j*data['DATA'][:,0,0,IF,channel,1,1]
        ll_2d = ll_2d.reshape(nvis, nifs, nchannels)
    else:
        ll_2d = rr_2d*0.0
    if num_corr >= 3:
        rl_2d = data['DATA'][:,0,0,IF,channel,2,0] + 1j*data['DATA'][:,0,0,IF,channel,2,1]
        rl_2d = rl_2d.reshape(nvis, nifs, nchannels)
    else:
        rl_2d = rr_2d*0.0
    if num_corr >= 4:
        lr_2d = data['DATA'][:,0,0,IF,channel,3,0] + 1j*data['DATA'][:,0,0,IF,channel,3,1]
        lr_2d = lr_2d.reshape(nvis, nifs, nchannels)
    else:
        lr_2d = rr_2d*0.0

    if force_singlepol == 'LR':
        rr_2d = copy.deepcopy(lr_2d)
    elif force_singlepol == 'RL':
        rr_2d = copy.deepcopy(rl_2d)

    rr_2d[~rrmask_2d] = np.nan
    ll_2d[~llmask_2d] = np.nan
    rl_2d[~rlmask_2d] = np.nan
    lr_2d[~lrmask_2d] = np.nan

    rr = np.nanmean(np.nanmean(rr_2d, axis=2), axis=1)[mask]
    ll = np.nanmean(np.nanmean(ll_2d, axis=2), axis=1)[mask]
    rl = np.nanmean(np.nanmean(rl_2d, axis=2), axis=1)[mask]
    lr = np.nanmean(np.nanmean(lr_2d, axis=2), axis=1)[mask]

    #rr = np.mean(data['DATA'][:,0,0,0,:,0,0][mask] + 1j*data['DATA'][:,0,0,0,:,0,1][mask], axis=1)
    #ll = np.mean(data['DATA'][:,0,0,0,:,1,0][mask] + 1j*data['DATA'][:,0,0,0,:,1,1][mask], axis=1)
    #rl = np.mean(data['DATA'][:,0,0,0,:,2,0][mask] + 1j*data['DATA'][:,0,0,0,:,2,1][mask], axis=1)
    #lr = np.mean(data['DATA'][:,0,0,0,:,3,0][mask] + 1j*data['DATA'][:,0,0,0,:,3,1][mask], axis=1)

    # average weights
    # variances are mean / N , or sum / N^2
    # replace masked weights with nans so they don't mess up the average
    rrweight[~rrmask_2d] = np.nan
    llweight[~llmask_2d] = np.nan
    rlweight[~rlmask_2d] = np.nan
    lrweight[~lrmask_2d] = np.nan

    nsig_rr = np.sum(np.sum(rrmask_2d, axis=2), axis=1).astype(float)
    nsig_rr[~rrmask] = np.nan
    rrsig = np.sqrt(np.nansum(np.nansum(1./rrweight, axis=2), axis=1)) / nsig_rr
    rrsig = rrsig[mask]

    nsig_ll = np.sum(np.sum(llmask_2d, axis=2), axis=1).astype(float)
    nsig_ll[~llmask] = np.nan
    llsig = np.sqrt(np.nansum(np.nansum(1./llweight, axis=2), axis=1)) / nsig_ll
    llsig = llsig[mask]

    nsig_rl = np.sum(np.sum(rlmask_2d, axis=2), axis=1).astype(float)
    nsig_rl[~rlmask] = np.nan
    rlsig = np.sqrt(np.nansum(np.nansum(1./rlweight, axis=2), axis=1)) / nsig_rl
    rlsig = rlsig[mask]

    nsig_lr = np.sum(np.sum(lrmask_2d, axis=2), axis=1).astype(float)
    nsig_lr[~lrmask] = np.nan
    lrsig = np.sqrt(np.nansum(np.nansum(1./lrweight, axis=2), axis=1)) / nsig_lr
    lrsig = lrsig[mask]

    # make sigmas from weights
    # zero out weights with zero error
    #rrweight[rrweight==0] = EP
    #llweight[llweight==0] = EP
    #rlweight[rlweight==0] = EP
    #lrweight[lrweight==0] = EP

    #rrsig = np.sqrt(rrweight)
    #llsig = 1/np.sqrt(llweight)
    #rlsig = 1/np.sqrt(rlweight)
    #lrsig = 1/np.sqrt(lrweight)

    # Form stokes parameters from data
    # look at these mask choices!!
    rrmask_dsize = rrmask[mask]
    llmask_dsize = llmask[mask]
    rlmask_dsize = rlmask[mask]
    lrmask_dsize = lrmask[mask]

    qumask_dsize = (rlmask_dsize * lrmask_dsize) # must have both RL & LR data to get Q, U
    vmask_dsize  = (rrmask_dsize * llmask_dsize) # must have both RR & LL data to get V

    # Stokes I
    ivis = 0.5 * (rr + ll)
    ivis[~llmask_dsize] = rr[~llmask_dsize] #if no RR, then say I is LL
    ivis[~rrmask_dsize] = ll[~rrmask_dsize] #if no LL, then say I is RR

    isigma = 0.5 * np.sqrt(rrsig**2 + llsig**2)
    isigma[~llmask_dsize] = rrsig[~llmask_dsize]
    isigma[~rrmask_dsize] = llsig[~rrmask_dsize]

    # TODO what should the polarization  sigmas be if no data?
    # Stokes V
    vvis = 0.5 * (rr - ll)
    vvis[~vmask_dsize] = 0.

    vsigma = copy.deepcopy(isigma) #ARGH POINTERS
    vsigma[~vmask_dsize] = isigma[~vmask_dsize]

    # Stokes Q,U
    qvis = 0.5 * (rl + lr)
    uvis = 0.5j * (lr - rl)
    qvis[~qumask_dsize] = 0.
    uvis[~qumask_dsize] = 0.

    qsigma = 0.5 * np.sqrt(rlsig**2 + lrsig**2)
    usigma = qsigma
    qsigma[~qumask_dsize] = isigma[~qumask_dsize]
    usigma[~qumask_dsize] = isigma[~qumask_dsize]

    # Reverse sign of baselines for correct imaging?
    if flipbl:
        u = -u
        v = -v

    # Make a datatable
    datatable = []
    for i in range(len(times)):
        datatable.append(np.array
                         ((
                           times[i], tints[i],
                           t1[i], t2[i], tau1[i], tau2[i],
                           u[i], v[i],
                           ivis[i], qvis[i], uvis[i], vvis[i],
                           isigma[i], qsigma[i], usigma[i], vsigma[i]
                           ), dtype=DTPOL
                         ))

    datatable = np.array(datatable)

    #TODO get calibration flags from uvfits?
    return ehtim.obsdata.Obsdata(ra, dec, rf, bw, datatable, tarr, source=src, mjd=mjd, scantable=scantable)


def load_obs_oifits(filename, flux=1.0):
    """Load data from an oifits file
       Does NOT currently support polarization
    """

    print('Warning: load_obs_oifits does NOT currently support polarimetric data!')

    # open oifits file and get visibilities
    oidata=ehtim.io.oifits.open(filename)
    vis_data = oidata.vis

    # get source info
    src = oidata.target[0].target
    ra = oidata.target[0].raep0.angle
    dec = oidata.target[0].decep0.angle

    # get annena info
    nAntennas = len(oidata.array[list(oidata.array.keys())[0]].station)
    sites = np.array([oidata.array[list(oidata.array.keys())[0]].station[i].sta_name for i in range(nAntennas)])
    arrayX = oidata.array[list(oidata.array.keys())[0]].arrxyz[0]
    arrayY = oidata.array[list(oidata.array.keys())[0]].arrxyz[1]
    arrayZ = oidata.array[list(oidata.array.keys())[0]].arrxyz[2]
    x = np.array([arrayX + oidata.array[list(oidata.array.keys())[0]].station[i].staxyz[0] for i in range(nAntennas)])
    y = np.array([arrayY + oidata.array[list(oidata.array.keys())[0]].station[i].staxyz[1] for i in range(nAntennas)])
    z = np.array([arrayZ + oidata.array[list(oidata.array.keys())[0]].station[i].staxyz[2] for i in range(nAntennas)])

    # get wavelength and corresponding frequencies
    wavelength = oidata.wavelength[list(oidata.wavelength.keys())[0]].eff_wave
    nWavelengths = wavelength.shape[0]
    bandpass = oidata.wavelength[list(oidata.wavelength.keys())[0]].eff_band
    frequency = C/wavelength

    #TODO: this result seems wrong...
    bw = np.mean(2*(np.sqrt( bandpass**2*frequency**2 + C**2) - C)/bandpass)
    rf = np.mean(frequency)

    # get the u-v point for each visibility
    u = np.array([vis_data[i].ucoord/wavelength for i in range(len(vis_data))])
    v = np.array([vis_data[i].vcoord/wavelength for i in range(len(vis_data))])

    # get visibility info - currently the phase error is not being used properly
    amp = np.array([vis_data[i]._visamp for i in range(len(vis_data))])
    phase = np.array([vis_data[i]._visphi for i in range(len(vis_data))])
    amperr = np.array([vis_data[i]._visamperr for i in range(len(vis_data))])
    visphierr = np.array([vis_data[i]._visphierr for i in range(len(vis_data))])
    timeobs = np.array([vis_data[i].timeobs for i in range(len(vis_data))]) #convert to single number

    #return timeobs
    time = np.transpose(np.tile(np.array([(ttime.mktime((timeobs[i] + datetime.timedelta(days=1)).timetuple()))/(60.0*60.0)
                                        for i in range(len(timeobs))]), [nWavelengths, 1]))

    # integration time
    tint = np.array([vis_data[i].int_time for i in range(len(vis_data))])
    #if not all(tint[0] == item for item in np.reshape(tint, (-1)) ):
        #raise TypeError("The time integrations for each visibility are different")
    tint = tint[0]
    tint = tint * np.ones( amp.shape )

    # get telescope names for each visibility
    t1 = np.transpose(np.tile( np.array([ vis_data[i].station[0].sta_name for i in range(len(vis_data))]), [nWavelengths,1]))
    t2 = np.transpose(np.tile( np.array([ vis_data[i].station[1].sta_name for i in range(len(vis_data))]), [nWavelengths,1]))

    # dummy variables
    tau1 = np.zeros(amp.shape)
    tau2 = np.zeros(amp.shape)
    qvis = np.zeros(amp.shape)
    uvis = np.zeros(amp.shape)
    vvis = np.zeros(amp.shape)
    sefdr = np.zeros(x.shape)
    sefdl = np.zeros(x.shape)
    fr_par = np.zeros(x.shape)
    fr_el = np.zeros(x.shape)
    fr_off = np.zeros(x.shape)
    dr = np.zeros(x.shape) + 1j*np.zeros(x.shape)
    dl = np.zeros(x.shape) + 1j*np.zeros(x.shape)

    # vectorize
    time = time.ravel()
    tint = tint.ravel()
    t1 = t1.ravel()
    t2 = t2.ravel()

    tau1 = tau1.ravel()
    tau2 = tau2.ravel()
    u = u.ravel()
    v = v.ravel()
    vis = amp.ravel() * np.exp ( -1j * phase.ravel() * np.pi/180.0 )
    qvis = qvis.ravel()
    uvis = uvis.ravel()
    vvis = vvis.ravel()
    amperr = amperr.ravel()

    #TODO - check that we are properly using the error from the amplitude and phase
    # create data tables
    datatable = np.array([(time[i], tint[i], t1[i], t2[i], tau1[i], tau2[i], u[i], v[i],
                           flux*vis[i], qvis[i], uvis[i], vvis[i],
                           flux*amperr[i], flux*amperr[i], flux*amperr[i], flux*amperr[i]
                          ) for i in range(len(vis))
                         ], dtype=DTPOL)

    tarr = np.array([(sites[i], x[i], y[i], z[i],
                      sefdr[i], sefdl[i], dr[i], dl[i],
                      fr_par[i], fr_el[i], fr_off[i],
                     ) for i in range(nAntennas)
                    ], dtype=DTARR)

    # return object

    return ehtim.obsdata.Obsdata(ra, dec, rf, bw, datatable, tarr, source=src, mjd=time[0])
