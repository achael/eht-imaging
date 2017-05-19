from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range
from past.utils import old_div
import numpy as np
import string
import astropy.io.fits as fits
import datetime
import os

import ehtim.obsdata
import ehtim.image 
import ehtim.array
import ehtim.movie
import ehtim.vex

import ehtim.io.oifits

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

##################################################################################################
# Vex IO
##################################################################################################  
def load_vex(fname):
    """Read in .vex files. and function to observe them
       Assumes there is only 1 MODE in vex file
       Hotaka Shiokawa - 2017
    """
    return ehtim.vex.Vex(fname)

##################################################################################################
# Movie IO
##################################################################################################  
#!AC TODO this needs a lot of work
#!AC TODO think about how to do the filenames
def load_movie_txt(basename, nframes, framedur=-1, pulse=PULSE_DEFAULT):
    """Read in a movie from text files and create a Movie object
       Text files should be filename + 00001, etc. 
       Text files should have the same format as output from Image.save_txt()
       Make sure the header has exactly the same form!
    """
    
    framelist = []
    qlist = []
    ulist = []

    for i in range(nframes):
        filename = basename + "%05d" % i
        
        # Read the header
        file = open(filename)
        src = string.join(file.readline().split()[2:])
        ra = file.readline().split()
        ra = float(ra[2]) + old_div(float(ra[4]),60.) + old_div(float(ra[6]),3600.)
        dec = file.readline().split()
        dec = np.sign(float(dec[2])) *(abs(float(dec[2])) + old_div(float(dec[4]),60.) + old_div(float(dec[6]),3600.))
        mjd_frac = float(file.readline().split()[2])
        mjd = np.floor(mjd_frac)
        hour = (mjd_frac - mjd)*24.0
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
        if i == 0:
            src0 = src
            ra0 = ra
            dec0 = dec
            mjd0 = mjd
            hour0 = hour
            rf0 = rf
            xdim0 = xdim
            ydim0 = ydim
            psize0 = psize_x
        else:
            pass
            #some checks on header consistency
            
        # Load the data, convert to list format, make object
        datatable = np.loadtxt(filename, dtype=float)
        image = datatable[:,2].reshape(ydim_p, xdim_p)
        framelist.append(image)
        
        # Look for Stokes Q and U
        qimage = uimage = np.zeros(image.shape)
        if datatable.shape[1] == 5:
            qimage = datatable[:,3].reshape(ydim_p, xdim_p)
            uimage = datatable[:,4].reshape(ydim_p, xdim_p)
            qlist.append(qimage)
            ulist.append(uimage)

    if frame_dur == -1:
        frame_dur = (hour - hour0)/float(nframes)*3600.0

    out_mov = ehtim.movie.Movie(framelist, framedur, psize0, ra0, dec0, rf=rf0, source=src0, mjd=mjd0, start_hr=hour0, pulse=pulse)
    
    if len(qlist):
        print('Loaded Stokes I, Q, and U movies')
        out_mov.add_qu(qlist, ulist)
    else:
        print('Loaded Stokes I movie only')
    
    return out_mov


##################################################################################################
# Image IO
##################################################################################################  
def load_im_txt(filename, pulse=PULSE_DEFAULT):
    """Read in an image from a text file and create an Image object
       Text file should have the same format as output from Image.save_txt()
       Make sure the header has exactly the same form!
    """
    
    # Read the header
    file = open(filename)
    src = string.join(file.readline().split()[2:])
    ra = file.readline().split()
    ra = float(ra[2]) + old_div(float(ra[4]),60.) + old_div(float(ra[6]),3600.)
    dec = file.readline().split()
    dec = np.sign(float(dec[2])) *(abs(float(dec[2])) + old_div(float(dec[4]),60.) + old_div(float(dec[6]),3600.))
    mjd = int(float(file.readline().split()[2]))
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
    outim = ehtim.image.Image(image, psize_x, ra, dec, rf=rf, source=src, mjd=mjd, pulse=pulse)
    
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
        print('Loaded Stokes I, Q, U, and V Images')
        outim.add_qu(qimage, uimage)
        outim.add_v(vimage)
    elif np.any((vimage != 0)):
        print('Loaded Stokes I and V Images')
        outim.add_v(vimage)
    elif np.any((qimage != 0) + (uimage != 0)):   
        print('Loaded Stokes I, Q, and U Images')
        outim.add_qu(qimage, uimage)     
    else:
        print('Loaded Stokes I Image Only')
    
    return outim

def load_im_fits(filename, punit="deg", pulse=PULSE_DEFAULT):
    """Read in an image from a FITS file and create an Image object
    """

    # Radian or Degree?
    if punit=="deg":
        pscl = DEGREE
    elif punit=="rad":
        pscl = 1.0
    elif punit=="uas":
        pscl = RADPERUAS
        
    # Open the FITS file
    hdulist = fits.open(filename)
    
    # Assume stokes I is the primary hdu
    header = hdulist[0].header
    
    # Read some header values
    ra = header['OBSRA']*12/180.
    dec = header['OBSDEC']
    xdim_p = header['NAXIS1']
    psize_x = np.abs(header['CDELT1']) * pscl
    dim_p = header['NAXIS2']
    psize_y = np.abs(header['CDELT2']) * pscl
    
    if 'MJD' in list(header.keys()): mjd = header['MJD']
    else: mjd = 0.0 
    
    if 'FREQ' in list(header.keys()): rf = header['FREQ']
    else: rf = 0.0
    
    if 'OBJECT' in list(header.keys()): src = header['OBJECT']
    else: src = ''
    
    # Get the image and create the object
    data = hdulist[0].data
    data = data.reshape((data.shape[-2],data.shape[-1]))
    image = data[::-1,:] # flip y-axis!
    
    # normalize the flux
    normalizer = 1.0;
    if 'BUNIT' in list(header.keys()):
        if header['BUNIT'] == 'JY/BEAM':
            beamarea = (2.0*np.pi*header['BMAJ']*header['BMIN']/(8.0*np.log(2)))
            normalizer = old_div((header['CDELT2'])**2, beamarea)
    image *= normalizer
            
    # make image object            
    outim = ehtim.image.Image(image, psize_x, ra, dec, rf=rf, source=src, mjd=mjd, pulse=pulse)
    
    # Look for Stokes Q and U
    qimage = uimage = vimage = np.array([])
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
        print('Loaded Stokes I, Q, U, and V Images')
        outim.add_qu(qimage, uimage)
        outim.add_v(vimage)
    elif vimage.shape == image.shape:
        print('Loaded Stokes I and V Images')
        outim.add_v(vimage)
    elif qimage.shape == uimage.shape == image.shape:   
        print('Loaded Stokes I, Q, and U Images')
        outim.add_qu(qimage, uimage)     
    else:
        print('Loaded Stokes I Image Only')
                            
    return outim


#!AC TODO - Michael...what exactly is this for? 
def load_im_manual_fits(filename, timesrot90=0, punit="deg", fov=-1, ra=RA_DEFAULT, dec=DEC_DEFAULT,
                        rf=RF_DEFAULT, src=SOURCE_DEFAULT, mjd=MJD_DEFAULT , pulse=PULSE_DEFAULT):

    # Radian or Degree?
    if punit=="deg":
        pscl = DEGREE
    elif punit=="rad":
        pscl = 1.0
    elif punit=="uas":
        pscl = RADPERUAS
    elif punit=="mas":
        pscl = RADPERUAS * 1000.0


    hdulist = fits.open(filename)
    header = hdulist[0].header
    data = hdulist[0].data

    if 'NAXIS1' in list(header.keys()): xdim_p = header['NAXIS1']
    else: xdim_p = data.shape[-2]

    if 'CDELT1' in list(header.keys()): 
        psize_x = np.abs(header['CDELT1']) * pscl
    else: 
        psize_x = (old_div(float(fov), data.shape[-2])) * pscl
        if fov==-1:
            print('WARNING: Must provide a field of view for the image')

    normalizer = 1.0; 
    if 'BUNIT' in list(header.keys()):
        if header['BUNIT'] == 'JY/BEAM':
            beamarea = (2.0*np.pi*header['BMAJ']*header['BMIN']/(8.0*np.log(2)))
            normalizer = old_div((header['CDELT2'])**2, beamarea)

    data = data.reshape((data.shape[-2],data.shape[-1]))

    image = data[::-1,:] # flip y-axis!
    image = np.rot90(image, k=timesrot90)
    outim = ehtim.image.Image(image*normalizer, psize_x, ra, dec, rf=rf, source=src, mjd=mjd, pulse=pulse)
    
    print('Loaded Stokes I image only')
    return outim
        
##################################################################################################
# Array IO
##################################################################################################  
def load_array_txt(filename, ephemdir='./ephemeris'):
    """Read an array from a text file and return an Array object
       Sites with x=y=z=0 are spacecraft - 2TLE ephemeris loaded from ephemdir
    """
    
    tdata = np.loadtxt(filename,dtype=str,comments='#')
    path = os.path.dirname(filename)

    tdataout = []
    if (tdata.shape[1] != 5 and tdata.shape[1] != 13):
        raise Exception("Array file should have format: "+ 
                        "(name, x, y, z, SEFDR, SEFDL "+
                        "FR_PAR_ANGLE FR_ELEV_ANGLE FR_OFFSET" +
                        "DR_RE   DR_IM   DL_RE    DL_IM )") 

    elif tdata.shape[1] == 5:                    
    	tdataout = [np.array((x[0],float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[4]),0.0, 0.0, 0.0, 0.0, 0.0),
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
            ephempath = path  + ephemdir + '/' + sitename #!AC TODO ephempath shouldn't always start with path
            try: 
                edata[sitename] = np.loadtxt(ephempath, dtype=str, comments='#', delimiter='/')
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
    
    # Read the header parameters
    file = open(filename)
    src = string.join(file.readline().split()[2:])
    ra = file.readline().split()
    ra = float(ra[2]) + old_div(float(ra[4]),60.) + old_div(float(ra[6]),3600.)
    dec = file.readline().split()
    dec = np.sign(float(dec[2])) *(abs(float(dec[2])) + old_div(float(dec[4]),60.) + old_div(float(dec[6]),3600.))
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
    datatable = np.loadtxt(filename, dtype=str)
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
    tdata = np.loadtxt(arrfile, dtype=str)
    tdata = [np.array((x[0],float(x[1]),float(x[2]),float(x[3]),float(x[-1])), dtype=DTARR) for x in tdata]
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
            ra = float(x[0]) + old_div(float(x[1]),60.) + old_div(float(x[2]),3600.)
        elif line[0] == 'FOV_center_Dec':
            x = line[2].split(':')
            dec = np.sign(float(x[0])) * (abs(float(x[0])) + old_div(float(x[1]),60.) + old_div(float(x[2]),3600.))
        elif line[0] == 'Corr_int_time':
            tint = float(line[2])
        elif line[0] == 'Corr_chan_bw':  #!AC TODO what if multiple channels?
            bw = float(line[2]) * 1e6 #in MHz
        elif line[0] == 'Channel': #!AC TODO what if multiple scans with different params?
            rf = float(line[2].split(':')[0]) * 1e6
        elif line[0] == 'Scan_start':
            x = line[2].split(':') #!AC properly compute MJD! 
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
            time = float(time[2]) + old_div(float(time[3]),60.) + old_div(float(time[4]),3600.)
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
    
    #!AC TODO qfile ufile and vfile must have exactly the same format as ifile: add some consistency check 
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


#!AC TODO can we save new telescope array terms and flags to uvfits and load them?
def load_obs_uvfits(filename, flipbl=False):
    """Load uvfits data from a uvfits file.
    """
        
    # Load the uvfits file
    hdulist = fits.open(filename)
    header = hdulist[0].header
    data = hdulist[0].data
    
    # Load the array data
    tnames = hdulist['AIPS AN'].data['ANNAME']
    tnums = hdulist['AIPS AN'].data['NOSTA'] - 1
    xyz = hdulist['AIPS AN'].data['STABXYZ']
    try:
        sefdr = np.real(hdulist['AIPS AN'].data['SEFD'])
        sefdl = np.real(hdulist['AIPS AN'].data['SEFD']) #!AC TODO add sefdl to uvfits?
    except KeyError:
        print("Warning! no SEFD data in UVfits file")
        sefdr = np.zeros(len(tnames))
        sefdl = np.zeros(len(tnames))
    
    #!AC TODO - get the *real* values of these telescope parameters
    fr_par = np.zeros(len(tnames))
    fr_el = np.zeros(len(tnames))
    fr_off = np.zeros(len(tnames))
    dr = np.zeros(len(tnames)) + 1j*np.zeros(len(tnames))
    dl = np.zeros(len(tnames)) + 1j*np.zeros(len(tnames))
        
    tarr = [np.array((tnames[i], xyz[i][0], xyz[i][1], xyz[i][2], 
            sefdr[i], sefdl[i], fr_par[i], fr_el[i], fr_off[i],
            dr[i], dl[i]),
            dtype=DTARR) for i in range(len(tnames))]
            
    tarr = np.array(tarr)
    
    # Various header parameters
    ra = header['OBSRA'] * 12./180.
    dec = header['OBSDEC']   
    src = header['OBJECT']
    if header['CTYPE4'] == 'FREQ':
        rf = header['CRVAL4']
        bw = header['CDELT4'] 
    else: raise Exception('Cannot find observing frequency!')
    
    
    # Mask to screen bad data
    rrweight = data['DATA'][:,0,0,0,0,0,2]
    llweight = data['DATA'][:,0,0,0,0,1,2]
    rlweight = data['DATA'][:,0,0,0,0,2,2]
    lrweight = data['DATA'][:,0,0,0,0,3,2]
    mask = (rrweight > 0) * (llweight > 0) * (rlweight > 0) * (lrweight > 0)
    
    # Obs Times
    jds = data['DATE'][mask].astype('d') + data['_DATE'][mask].astype('d')
    mjd = int(np.min(jds)-2400000.5)
    times = (jds - 2400000.5 - mjd) * 24.0

    # old time conversion
    #jds = data['DATE'][mask]
    #mjd = int(jdtomjd(np.min(jds)))
    #print len(set(data['DATE']))
    #if len(set(data['DATE'])) > 2:
    #    times = np.array([mjdtogmt(jdtomjd(jd)) for jd in jds])
    #else:
    #    times = data['_DATE'][mask] * 24.0
    
    # Integration times
    tints = data['INTTIM'][mask]
    
    # Sites - add names
    t1 = data['BASELINE'][mask].astype(int)//256
    t2 = data['BASELINE'][mask].astype(int) - t1*256
    t1 = t1 - 1
    t2 = t2 - 1
    scopes_num = np.sort(list(set(np.hstack((t1,t2)))))
    t1 = np.array([tarr[i]['site'] for i in t1])
    t2 = np.array([tarr[i]['site'] for i in t2])

    # Opacities (not in BU files)
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
                    
    # Get vis data
    rr = data['DATA'][:,0,0,0,0,0,0][mask] + 1j*data['DATA'][:,0,0,0,0,0,1][mask]
    ll = data['DATA'][:,0,0,0,0,1,0][mask] + 1j*data['DATA'][:,0,0,0,0,1,1][mask]
    rl = data['DATA'][:,0,0,0,0,2,0][mask] + 1j*data['DATA'][:,0,0,0,0,2,1][mask]
    lr = data['DATA'][:,0,0,0,0,3,0][mask] + 1j*data['DATA'][:,0,0,0,0,3,1][mask]
    rrsig = old_div(1,np.sqrt(rrweight[mask]))
    llsig = old_div(1,np.sqrt(llweight[mask]))
    rlsig = old_div(1,np.sqrt(rlweight[mask]))
    lrsig = old_div(1,np.sqrt(lrweight[mask]))
    
    # Form stokes parameters
    ivis = 0.5 * (rr + ll)
    qvis = 0.5 * (rl + lr)
    uvis = 0.5j * (lr - rl)
    vvis = 0.5 * (rr - ll)
    sigma = 0.5 * np.sqrt(rrsig**2 + llsig**2)
    qsigma = 0.5* np.sqrt(rlsig**2 + lrsig**2)
    usigma = qsigma
    vsigma = sigma
   
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
                           sigma[i], qsigma[i], usigma[i], vsigma[i],
                           ), dtype=DTPOL
                         ))
    datatable = np.array(datatable)
    
    #!AC TODO get calibration flags from uvfits?
    return ehtim.obsdata.Obsdata(ra, dec, rf, bw, datatable, tarr, source=src, mjd=mjd)


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
    frequency = old_div(C,wavelength)
    
    # !AC TODO: this result seems wrong...
    bw = np.mean(2*(np.sqrt( bandpass**2*frequency**2 + C**2) - C)/bandpass)
    rf = np.mean(frequency)
    
    # get the u-v point for each visibility
    u = np.array([old_div(vis_data[i].ucoord,wavelength) for i in range(len(vis_data))])
    v = np.array([old_div(vis_data[i].vcoord,wavelength) for i in range(len(vis_data))])
    
    # get visibility info - currently the phase error is not being used properly
    amp = np.array([vis_data[i]._visamp for i in range(len(vis_data))])
    phase = np.array([vis_data[i]._visphi for i in range(len(vis_data))])
    amperr = np.array([vis_data[i]._visamperr for i in range(len(vis_data))])
    visphierr = np.array([vis_data[i]._visphierr for i in range(len(vis_data))])
    timeobs = np.array([vis_data[i].timeobs for i in range(len(vis_data))]) #convert to single number
    
    #return timeobs
    time = np.transpose(np.tile(np.array([old_div((ttime.mktime((timeobs[i] + datetime.timedelta(days=1)).timetuple())),(60.0*60.0)) 
                                        for i in range(len(timeobs))]), [nWavelengths, 1]))
    
    # integration time
    tint = np.array([vis_data[i].int_time for i in range(len(vis_data))])
    if not all(tint[0] == item for item in np.reshape(tint, (-1)) ):
        raise TypeError("The time integrations for each visibility are different")
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

    #!AC TODO - check that we are properly using the error from the amplitude and phase
    # create data tables
    datatable = np.array([(time[i], tint[i], t1[i], t2[i], tau1[i], tau2[i], u[i], v[i], 
                           flux*vis[i], qvis[i], uvis[i], vvis[i], 
                           flux*amperr[i], flux*amperr[i], flux*amperr[i], flux*amperr[i]
                          ) for i in range(len(vis))
                         ], dtype=DTPOL)

    tarr = np.array([(sites[i], x[i], y[i], z[i], 
                       sefdr[i], sefdl[i], fr_par[i], fr_el[i], fr_off[i],
                       dr[i], dl[i]
                     ) for i in range(nAntennas)
                    ], dtype=DTARR)
    
    # return object

    return ehtim.obsdata.Obsdata(ra, dec, rf, bw, datatable, tarr, source=src, mjd=time[0])


