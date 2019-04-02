# save.py
# functions to save observation & image data from files
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
from builtins import range

import numpy as np
import string
import astropy.io.fits as fits
import datetime
import os

import ehtim.io.writeData
import ehtim.io.oifits
from astropy.time import Time
from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

##################################################################################################
# Image IO
##################################################################################################
def save_im_txt(im, fname, mjd=False, time=False):
    """Save image data to text file.

       Args:
            fname (str): path to output text file
            mjd (int): MJD of saved image
            time (float): UTC time of saved image

       Returns:
    """

    # Transform to Stokes parameters:
    if im.polrep!='stokes' or im.pol_prim!='I':
        im = im.switch_polrep(polrep_out='stokes', pol_prim_out=None)

    # Coordinate values
    pdimas = im.psize/RADPERAS
    xs = np.array([[j for j in range(im.xdim)] for i in range(im.ydim)]).reshape(im.xdim*im.ydim,1)
    xs = pdimas * (xs[::-1] - im.xdim/2.0)
    ys = np.array([[i for j in range(im.xdim)] for i in range(im.ydim)]).reshape(im.xdim*im.ydim,1)
    ys = pdimas * (ys[::-1] - im.xdim/2.0)

    # If V values but no Q/U values, make Q/U zero
    if len(im.vvec) and not len(im.qvec):
        im.qvec = 0*im.vvec
        im.uvec = 0*im.vvec

    # Format Data
    if len(im.qvec) and len(im.vvec):
        outdata = np.hstack((xs, ys, (im.imvec).reshape(im.xdim*im.ydim, 1),
                                     (im.qvec).reshape(im.xdim*im.ydim, 1),
                                     (im.uvec).reshape(im.xdim*im.ydim, 1),
                                     (im.vvec).reshape(im.xdim*im.ydim, 1)))
        hf = "x (as)     y (as)       I (Jy/pixel)  Q (Jy/pixel)  U (Jy/pixel)  V (Jy/pixel)"

        fmts = "%10.10f %10.10f %10.10f %10.10f %10.10f %10.10f"

    elif len(im.qvec):
        outdata = np.hstack((xs, ys, (im.imvec).reshape(im.xdim*im.ydim, 1),
                                     (im.qvec).reshape(im.xdim*im.ydim, 1),
                                     (im.uvec).reshape(im.xdim*im.ydim, 1)))
        hf = "x (as)     y (as)       I (Jy/pixel)  Q (Jy/pixel)  U (Jy/pixel)"

        fmts = "%10.10f %10.10f %10.10f %10.10f %10.10f"

    else:
        outdata = np.hstack((xs, ys, (im.imvec).reshape(im.xdim*im.ydim, 1)))
        hf = "x (as)     y (as)       I (Jy/pixel)"
        fmts = "%10.10f %10.10f %10.10f"

    # Header
    if not mjd: mjd = float(im.mjd)
    if not time: time = im.time
    mjd += (time/24.)

    head = ("SRC: %s \n" % im.source +
                "RA: " + rastring(im.ra) + "\n" + "DEC: " + decstring(im.dec) + "\n" +
                "MJD: %.6f \n" % (float(mjd)) +  
                "RF: %.4f GHz \n" % (im.rf/1e9) +
                "FOVX: %i pix %f as \n" % (im.xdim, pdimas * im.xdim) +
                "FOVY: %i pix %f as \n" % (im.ydim, pdimas * im.ydim) +
                "------------------------------------\n" + hf)

    # Save
    np.savetxt(fname, outdata, header=head, fmt=fmts)
    return

#TODO save image in circular basis? 
def save_im_fits(im, fname, mjd=False, time=False):
    """Save image data to a fits file.

       Args:
            fname (str): path to output fits file
            mjd (int): MJD of saved image
            time (float): UTC time of saved image

       Returns:
    """

    # Transform to Stokes parameters:
    if (im.polrep!='stokes') or (im.pol_prim!='I'):
        im = im.switch_polrep(polrep_out='stokes', pol_prim_out=None)

    # Create header and fill in some values
    header = fits.Header()
    header['OBJECT'] = im.source
    header['CTYPE1'] = 'RA---SIN'
    header['CTYPE2'] = 'DEC--SIN'
    header['CDELT1'] = -im.psize/DEGREE
    header['CDELT2'] =  im.psize/DEGREE
    header['OBSRA'] = im.ra * 180/12.
    header['OBSDEC'] = im.dec
    header['FREQ'] = im.rf

    #TODO these are the default values for centered images
    #TODO support for arbitrary CRPIX? 
    header['CRPIX1'] = im.xdim/2. + .5
    header['CRPIX2'] = im.ydim/2. + .5

    if not mjd: mjd = float(im.mjd)
    if not time: time = im.time
    mjd += (time/24.)

    header['MJD'] = float(mjd)
    header['TELESCOP'] = 'VLBI'
    header['BUNIT'] = 'JY/PIXEL'
    header['STOKES'] = 'I'

    # Create the fits image
    image = np.reshape(im.imvec,(im.ydim,im.xdim))[::-1,:] #flip y axis!
    hdu = fits.PrimaryHDU(image, header=header)
    hdulist = [hdu]
    if len(im.qvec):
        qimage = np.reshape(im.qvec,(im.xdim,im.ydim))[::-1,:]
        uimage = np.reshape(im.uvec,(im.xdim,im.ydim))[::-1,:]
        header['STOKES'] = 'Q'
        hduq = fits.ImageHDU(qimage, name='Q', header=header)
        header['STOKES'] = 'U'
        hduu = fits.ImageHDU(uimage, name='U', header=header)
        hdulist = [hdu, hduq, hduu]
    if len(im.vvec):
        vimage = np.reshape(im.vvec,(im.xdim,im.ydim))[::-1,:]
        header['STOKES'] = 'V'
        hduv = fits.ImageHDU(vimage, name='V', header=header)
        hdulist.append(hduv)

    hdulist = fits.HDUList(hdulist)

    # Save fits
    hdulist.writeto(fname, overwrite=True)

    return

##################################################################################################
# Movie IO
##################################################################################################

def save_mov_fits(mov, fname, mjd=False):
    """Save movie data to series of fits files.

       Args:
            fname (str): basename of output fits file
            mjd (int): MJD of saved movie

       Returns:
    """

    if mjd==False: mjd=mov.mjd

    for i in range(mov.nframes):
        time_frame = mov.start_hr + i*mov.framedur/3600.
        fname_frame = fname + "%05d" % i
        print ('saving file '+fname_frame)
        frame_im = mov.get_frame(i)
        save_im_fits(frame_im, fname_frame, mjd=mjd, time=time_frame)

    return

def save_mov_txt(mov, fname, mjd=False):
    """Save movie data to series of text files.

       Args:
            fname (str): basename of output text file
            mjd (int): MJD of saved movie

       Returns:
    """

    if mjd==False: mjd=mov.mjd

    for i in range(mov.nframes):
        time_frame = mov.start_hr + i*mov.framedur/3600.
        fname_frame = fname + "%05d" % i
        print ('saving file '+fname_frame)
        frame_im = mov.get_frame(i)
        save_im_txt(frame_im, fname_frame, mjd=mjd, time=time_frame)

    return


##################################################################################################
# Array IO
##################################################################################################

def save_array_txt(arr, fname):
    """Save the array data in a text file.
    """
    import ehtim.array as array

    if type(arr) == np.ndarray:
        tarr = arr
    else:
        try: 
            tarr = arr.tarr
        except:
            print("Array format not recognized!")

    out = ("#Site      X(m)             Y(m)             Z(m)           "+
                "SEFDR      SEFDL     FR_PAR   FR_EL   FR_OFF  "+
                "DR_RE    DR_IM    DL_RE    DL_IM   \n")
    for scope in range(len(tarr)):
        dat = (tarr[scope]['site'],
               tarr[scope]['x'], tarr[scope]['y'], tarr[scope]['z'],
               tarr[scope]['sefdr'], tarr[scope]['sefdl'],
               tarr[scope]['fr_par'], tarr[scope]['fr_elev'], tarr[scope]['fr_off'],
               tarr[scope]['dr'].real, tarr[scope]['dr'].imag, tarr[scope]['dl'].real, tarr[scope]['dl'].imag
              )
        out += "%-8s %15.5f  %15.5f  %15.5f  %8.2f   %8.2f  %5.2f   %5.2f   %5.2f  %8.4f %8.4f %8.4f %8.4f \n" % dat
    f = open(fname,'w')
    f.write(out)
    f.close()
    return


##################################################################################################
# Observation IO
##################################################################################################
def save_obs_txt(obs, fname):
    """Save the observation data in a text file.
    """

    # Get the necessary data and the header
    if obs.polrep=='stokes':
        outdata = obs.unpack(['time', 'tint', 't1', 't2','tau1','tau2',
                               'u', 'v', 'amp', 'phase', 'qamp', 'qphase', 'uamp', 'uphase', 'vamp', 'vphase',
                               'sigma', 'qsigma', 'usigma', 'vsigma'])
    elif obs.polrep=='circ':
        outdata = obs.unpack(['time', 'tint', 't1', 't2','tau1','tau2',
                               'u', 'v', 'rramp', 'rrphase', 'llamp', 'llphase', 'rlamp', 'rlphase', 'lramp', 'lrphase',
                               'rrsigma', 'llsigma', 'rlsigma', 'lrsigma'])

    else: raise Exception("obs.polrep not 'stokes' or 'circ'!")

    head = ("SRC: %s \n" % obs.source +
                "RA: " + rastring(obs.ra) + "\n" + "DEC: " + decstring(obs.dec) + "\n" +
                "MJD: %i \n" % obs.mjd +
                "RF: %.4f GHz \n" % (obs.rf/1e9) +
                "BW: %.4f GHz \n" % (obs.bw/1e9) +
                "PHASECAL: %i \n" % obs.phasecal +
                "AMPCAL: %i \n" % obs.ampcal +
                "OPACITYCAL: %i \n" % obs.opacitycal +
                "DCAL: %i \n" % obs.dcal +
                "FRCAL: %i \n" % obs.frcal +
                "----------------------------------------------------------------------"+
                "------------------------------------------------------------------\n" +
                "Site       X(m)             Y(m)             Z(m)           "+
                "SEFDR      SEFDL     FR_PAR   FR_EL   FR_OFF  "+
                "DR_RE    DR_IM    DL_RE    DL_IM   \n"
            )

    for i in range(len(obs.tarr)):
        head += ("%-8s %15.5f  %15.5f  %15.5f  %8.2f   %8.2f  %5.2f   %5.2f   %5.2f  %8.4f %8.4f %8.4f %8.4f \n" % (obs.tarr[i]['site'],
                                                                  obs.tarr[i]['x'], obs.tarr[i]['y'], obs.tarr[i]['z'],
                                                                  obs.tarr[i]['sefdr'], obs.tarr[i]['sefdl'],
                                                                  obs.tarr[i]['fr_par'], obs.tarr[i]['fr_elev'], obs.tarr[i]['fr_off'],
                                                                  (obs.tarr[i]['dr']).real, (obs.tarr[i]['dr']).imag,
                                                                  (obs.tarr[i]['dl']).real, (obs.tarr[i]['dl']).imag
                                                                 ))

    if obs.polrep=='stokes':
        head += (
                "----------------------------------------------------------------------"+
                "------------------------------------------------------------------\n" +
                "time (hr) tint    T1     T2    Tau1   Tau2   U (lambda)       V (lambda)         "+
                "Iamp (Jy)    Iphase(d)  Qamp (Jy)    Qphase(d)   Uamp (Jy)    Uphase(d)   Vamp (Jy)    Vphase(d)   "+
                "Isigma (Jy)   Qsigma (Jy)   Usigma (Jy)   Vsigma (Jy)"
                )
    elif obs.polrep=='circ':
        head += (
                "----------------------------------------------------------------------"+
                "------------------------------------------------------------------\n" +
                "time (hr) tint    T1     T2    Tau1   Tau2   U (lambda)       V (lambda)         "+
                "RRamp (Jy)   RRphase(d) LLamp (Jy)   LLphase(d)  RLamp (Jy)   RLphase(d)  LRamp (Jy)   LRphase(d)  "+
                "RRsigma (Jy)  LLsigma (Jy)  RLsigma (Jy)  LRsigma (Jy)"
                )


    # Format and save the data
    fmts = ("%011.8f %4.2f %6s %6s  %4.2f   %4.2f  %16.4f %16.4f    "+
           "%10.8f %10.4f   %10.8f %10.4f    %10.8f %10.4f    %10.8f %10.4f    "+
           "%10.8f    %10.8f    %10.8f    %10.8f")
    np.savetxt(fname, outdata, header=head, fmt=fmts)
    return


def save_obs_uvfits(obs, fname, force_singlepol=None, polrep_out='circ'):
    """Save observation data to uvfits.
       To save Stokes I as a single polarization (e.g., only RR) set force_singlepol='R' or 'L'
    """

    if polrep_out=='circ':
        obs = obs.switch_polrep('circ')
    elif polrep_out=='stokes':
        obs = obs.switch_polrep('stokes')
    else:
        raise Exception("'polrep_out' in 'save_obs_uvfits' must be 'circ' or 'stokes'!")

    # Open template UVFITS
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #hdulist = fits.open(dir_path+'/template.UVP')

    hdulist_new = fits.HDUList()
    hdulist_new.append(fits.GroupsHDU())

    ##################### AIPS Data TABLE #####################################################################################################
    # Data table
    # Data header (based on the BU format)
    #header = fits.Header()
    #header = hdulist[0].header
    MJD_0 = 2400000.5
    header = hdulist_new['PRIMARY'].header
    header['OBSRA'] = obs.ra * 180./12.
    header['OBSDEC'] = obs.dec
    header['OBJECT'] = obs.source
    header['MJD'] = float(obs.mjd)
    header['DATE-OBS'] = Time(obs.mjd + MJD_0, format='jd', scale='utc', out_subfmt='date').iso
    header['BSCALE'] = 1.0
    header['BZERO'] = 0.0  
    header['BUNIT'] = 'JY'
    header['VELREF'] = 3        # TODO ??
    header['EQUINOX'] = 'J2000'
    header['ALTRPIX'] = 1.e0 
    header['ALTRVAL'] = 0.e0 
    header['TELESCOP'] = 'VLBA' # TODO Can we change this field?
    header['INSTRUME'] = 'VLBA'
    header['OBSERVER'] = 'EHT'

    header['CTYPE2'] = 'COMPLEX'
    header['CRVAL2'] = 1.e0
    header['CDELT2'] = 1.e0
    header['CRPIX2'] = 1.e0
    header['CROTA2'] = 0.e0
    header['CTYPE3'] = 'STOKES'
    if polrep_out=='circ':
        header['CRVAL3'] = -1.e0
        header['CDELT3'] = -1.e0
    elif polrep_out=='stokes':
        header['CRVAL3'] = 1.e0
        header['CDELT3'] = 1.e0
    header['CRPIX3'] = 1.e0
    header['CROTA3'] = 0.e0
    header['CTYPE4'] = 'FREQ'
    header['CRVAL4'] = obs.rf
    header['CDELT4'] = obs.bw
    header['CRPIX4'] = 1.e0
    header['CROTA4'] = 0.e0
    header['CTYPE6'] = 'RA'
    header['CRVAL6'] = header['OBSRA']
    header['CDELT6'] = 1.e0
    header['CRPIX6'] = 1.e0
    header['CROTA6'] = 0.e0
    header['CTYPE7'] = 'DEC'
    header['CRVAL7'] = header['OBSDEC']
    header['CDELT7'] = 1.e0
    header['CRPIX7'] = 1.e0
    header['CROTA7'] = 0.e0
    header['PTYPE1'] = 'UU---SIN'
    header['PSCAL1'] = 1.0/obs.rf
    header['PZERO1'] = 0.e0
    header['PTYPE2'] = 'VV---SIN'
    header['PSCAL2'] = 1.0/obs.rf
    header['PZERO2'] = 0.e0
    header['PTYPE3'] = 'WW---SIN'
    header['PSCAL3'] = 1.0/obs.rf
    header['PZERO3'] = 0.e0
    header['PTYPE4'] = 'BASELINE'
    header['PSCAL4'] = 1.e0
    header['PZERO4'] = 0.e0
    header['PTYPE5'] = 'DATE'
    header['PSCAL5'] = 1.e0
    header['PZERO5'] = 0.e0
    header['PTYPE6'] = 'DATE'
    header['PSCAL6'] = 1.e0
    header['PZERO6'] = 0.0
    header['PTYPE7'] = 'INTTIM'
    header['PSCAL7'] = 1.e0
    header['PZERO7'] = 0.e0
    header['PTYPE8'] = 'TAU1'
    header['PSCAL8'] = 1.e0
    header['PZERO8'] = 0.e0
    header['PTYPE9'] = 'TAU2'
    header['PSCAL9'] = 1.e0
    header['PZERO9'] = 0.e0
    header['history'] = "AIPS SORT ORDER='TB'"

    # Get data

    if polrep_out=='circ':
        obsdata = obs.unpack(['time','tint','u','v','rrvis','llvis','rlvis','lrvis',
                              'rrsigma','llsigma','rlsigma','lrsigma','t1','t2','tau1','tau2'])
    elif polrep_out=='stokes':
        obsdata = obs.unpack(['time','tint','u','v','vis','qvis','uvis','vvis',
                              'sigma','qsigma','usigma','vsigma','t1','t2','tau1','tau2'])

    ndat = len(obsdata['time'])

    # times and tints
    #jds = (obs.mjd + 2400000.5) * np.ones(len(obsdata))
    #fractimes = (obsdata['time'] / 24.0)
    jds = (2400000.5 + obs.mjd) * np.ones(len(obsdata))
    fractimes = (obsdata['time']/24.0)
    #jds = jds + fractimes
    #fractimes = np.zeros(len(obsdata))
    tints = obsdata['tint']

    # Baselines
    t1 = [obs.tkey[scope] + 1 for scope in obsdata['t1']]
    t2 = [obs.tkey[scope] + 1 for scope in obsdata['t2']]
    bl = 256*np.array(t1) + np.array(t2)

    # opacities
    tau1 = obsdata['tau1']
    tau2 = obsdata['tau2']

    # uv are in lightseconds
    u = obsdata['u']
    v = obsdata['v']

    # rr, ll, lr, rl, weights

    if polrep_out=='circ':
        rr = obsdata['rrvis'] 
        ll = obsdata['llvis'] 
        rl = obsdata['rlvis']
        lr = obsdata['lrvis'] 
        weightrr = 1.0/(obsdata['rrsigma']**2)
        weightll = 1.0/(obsdata['llsigma']**2)
        weightrl = 1.0/(obsdata['rlsigma']**2)
        weightlr = 1.0/(obsdata['lrsigma']**2)

        # If necessary, enforce single polarization
        if force_singlepol == 'L':
            if obs.polrep=='stokes': 
                raise Exception("force_singlepol only works with obs.polrep=='stokes'!")
            print("force_singlepol='L': treating Stokes 'I' as LL and ignoring Q,U,V!!")
            ll = obsdata['vis']
            rr = rr * 0.0
            rl = rl * 0.0
            lr = lr * 0.0
            weightrr = weightrr * 0.0
            weightrl = weightrl * 0.0
            weightlr = weightlr * 0.0
        elif force_singlepol == 'R':
            if obs.polrep=='stokes': 
                raise Exception("force_singlepol only works with obs.polrep=='stokes'!")
            print("force_singlepol='R': treating Stokes 'I' as RR and ignoring Q,U,V!!")
            rr = obsdata['vis']
            ll = rr * 0.0
            rl = rl * 0.0
            lr = lr * 0.0
            weightll = weightll * 0.0
            weightrl = weightrl * 0.0
            weightlr = weightlr * 0.0

        dat1 = rr
        dat2 = ll
        dat3 = rl
        dat4 = lr
        weight1 = weightrr
        weight2 = weightll
        weight3 = weightrl
        weight4 = weightlr

    elif polrep_out=='stokes':
        dat1 = obsdata['vis']
        dat2 = obsdata['qvis']
        dat3 = obsdata['uvis']
        dat4 = obsdata['vvis']
        weight1 = 1.0/(obsdata['sigma']**2)
        weight2 = 1.0/(obsdata['qsigma']**2)
        weight3 = 1.0/(obsdata['usigma']**2)
        weight4 = 1.0/(obsdata['vsigma']**2)



    # Replace nans by zeros (including zero weights)
    dat1 = np.nan_to_num(dat1)
    dat2 = np.nan_to_num(dat2)
    dat3 = np.nan_to_num(dat3)
    dat4 = np.nan_to_num(dat4)
    weight1 = np.nan_to_num(weight1)
    weight2 = np.nan_to_num(weight2)
    weight3 = np.nan_to_num(weight3)
    weight4 = np.nan_to_num(weight4)

    # Data array
    outdat = np.zeros((ndat, 1, 1, 1, 1, 4, 3))
    outdat[:,0,0,0,0,0,0] = np.real(dat1)
    outdat[:,0,0,0,0,0,1] = np.imag(dat1)
    outdat[:,0,0,0,0,0,2] = weight1
    outdat[:,0,0,0,0,1,0] = np.real(dat2)
    outdat[:,0,0,0,0,1,1] = np.imag(dat2)
    outdat[:,0,0,0,0,1,2] = weight2
    outdat[:,0,0,0,0,2,0] = np.real(dat3)
    outdat[:,0,0,0,0,2,1] = np.imag(dat3)
    outdat[:,0,0,0,0,2,2] = weight3
    outdat[:,0,0,0,0,3,0] = np.real(dat4)
    outdat[:,0,0,0,0,3,1] = np.imag(dat4)
    outdat[:,0,0,0,0,3,2] = weight4

    # Save data
    pars = ['UU---SIN', 'VV---SIN', 'WW---SIN', 'BASELINE', 'DATE', 'DATE',
            'INTTIM', 'TAU1', 'TAU2']
    x = fits.GroupData(outdat, parnames=pars,
        pardata=[u, v, np.zeros(ndat), bl, jds, fractimes, tints,tau1,tau2],
        bitpix=-32)

    #hdulist[0].data = x
    #hdulist[0].header = header
    hdulist_new['PRIMARY'].data = x
    hdulist_new['PRIMARY'].header = header # TODO necessary, or is it a pointer?

    ##################### AIPS AN TABLE #####################################################################################################
    # Antenna table

    # Load the array data
    tarr = obs.tarr
    tnames = tarr['site']
    tnums = np.arange(1, len(tarr)+1)
    xyz = np.array([[tarr[i]['x'],tarr[i]['y'],tarr[i]['z']] for i in np.arange(len(tarr))])
    sefd = tarr['sefdr']

    nsta = len(tnames)
    col1 = fits.Column(name='ANNAME', format='8A', array=tnames)
    col2 = fits.Column(name='STABXYZ', format='3D', unit='METERS', array=xyz)
    col3 = fits.Column(name='NOSTA', format='1J', array=tnums)
    colfin = fits.Column(name='SEFD', format='1D', array=sefd)

    #TODO these antenna fields+header are questionable - look into them
    col4 = fits.Column(name='MNTSTA', format='1J', array=np.zeros(nsta))
    col5 = fits.Column(name='STAXOF', format='1E', unit='METERS', array=np.zeros(nsta))
    col6 = fits.Column(name='POLTYA', format='1A', array=np.array(['R' for i in range(nsta)], dtype='|S1'))
    col7 = fits.Column(name='POLAA', format='1E', unit='DEGREES', array=np.zeros(nsta))
    col8 = fits.Column(name='POLCALA', format='3E', array=np.zeros((nsta,3)))
    col9 = fits.Column(name='POLTYB', format='1A', array=np.array(['L' for i in range(nsta)], dtype='|S1'))
    col10 = fits.Column(name='POLAB', format='1E', unit='DEGREES', array=(90.*np.ones(nsta)))
    col11 = fits.Column(name='POLCALB', format='3E', array=np.zeros((nsta,3)))
    col25= fits.Column(name='ORBPARM', format='1E', array=np.zeros(0))

    #Antenna Header params - do I need to change more of these??
    #head = fits.Header()
    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs([col1,col2,col25,col3,col4,col5,col6,col7,col8,col9,col10,col11,colfin]), name='AIPS AN')
    hdulist_new.append(tbhdu)

    #head = hdulist['AIPS AN'].header
    head = hdulist_new['AIPS AN'].header

    head['EXTVER'] = 1
    head['ARRAYX'] = 0.e0
    head['ARRAYY'] = 0.e0
    head['ARRAYZ'] = 0.e0

    # TODO change the reference date
    #rdate_out = '2000-01-01T00:00:00.0'
    #rdate_gstiao_out = 114.38389781355
    #rdate_offset_out = 0.e0
    rdate_tt_new = Time(obs.mjd + MJD_0, format='jd', scale='utc', out_subfmt='date')
    rdate_out = rdate_tt_new.iso
    rdate_jd_out = rdate_tt_new.jd
    rdate_gstiao_out = rdate_tt_new.sidereal_time('apparent','greenwich').degree
    rdate_offset_out = (rdate_tt_new.ut1.datetime.second - rdate_tt_new.utc.datetime.second)
    rdate_offset_out += 1.e-6*(rdate_tt_new.ut1.datetime.microsecond - rdate_tt_new.utc.datetime.microsecond)

    head['RDATE'] = rdate_out
    head['GSTIA0'] = rdate_gstiao_out
    head['DEGPDY'] = 360.9856
    head['UT1UTC'] = rdate_offset_out   #difference between UT1 and UTC ?
    head['DATUTC'] = 0.e0
    head['TIMESYS'] = 'UTC'

    head['FREQ']= obs.rf
    head['POLARX'] = 0.e0
    head['POLARY'] = 0.e0

    head['ARRNAM'] = 'VLBA'  # TODO must be recognized by aips/casa
    head['XYZHAND'] = 'RIGHT'
    head['FRAME'] = '????'
    head['NUMORB'] = 0
    head['NO_IF'] = 1 #TODO nchan
    head['NOPCAL'] = 0  #TODO add pol cal information
    head['POLTYPE'] = 'VLBI'
    head['FREQID'] = 1

    hdulist_new['AIPS AN'].header = head # TODO necessary, or is it a pointer?

    ##################### AIPS FQ TABLE #####################################################################################################
    # Convert types & columns

    nif=1
    col1 = np.array(1, dtype=np.int32).reshape([nif]) #frqsel
    col2 = np.array(0.0, dtype=np.float64).reshape([nif]) #iffreq
    col3 = np.array([obs.bw], dtype=np.float32).reshape([nif]) #chwidth
    col4 = np.array([obs.bw], dtype=np.float32).reshape([nif]) #bw
    col5 = np.array([1], dtype=np.int32).reshape([nif]) #sideband

    col1 = fits.Column(name="FRQSEL", format="1J", array=col1)
    col2 = fits.Column(name="IF FREQ", format="%dD"%(nif), array=col2)
    col3 = fits.Column(name="CH WIDTH",format="%dE"%(nif),array=col3)
    col4 = fits.Column(name="TOTAL BANDWIDTH",format="%dE"%(nif),array=col4)
    col5 = fits.Column(name="SIDEBAND",format="%dJ"%(nif),array=col5)
    cols = fits.ColDefs([col1, col2,col3,col4,col5])

    # create table
    tbhdu = fits.BinTableHDU.from_columns(cols)

    # add header information
    tbhdu.header.append(("NO_IF", nif, "Number IFs"))
    tbhdu.header.append(("EXTNAME","AIPS FQ"))
    tbhdu.header.append(("EXTVER",1))
    hdulist_new.append(tbhdu)

    ##################### AIPS NX TABLE #####################################################################################################

    scan_times = []
    scan_time_ints = []
    start_vis = []
    stop_vis = []
    
    #TODO make sure jds AND scan_info MUST be time sorted!!
    jj = 0

    ROUND_SCAN_INT = 5
    comp_fac = 3600*24*100 # compare to 100th of a second
    scan_arr = obs.scans
    print ('Building NX table')
    if (scan_arr is None or len(scan_arr) == 0):
        print ("No NX table in saved uvfits")
    else:
        try: 
            scan_arr = scan_arr/24.
            for scan in  scan_arr:
                scan_start = round(scan[0], ROUND_SCAN_INT)
                scan_stop = round(scan[1], ROUND_SCAN_INT)
                scan_dur = (scan_stop - scan_start)

                if jj>=len(fractimes):
                    #print start_vis, stop_vis
                    break

                #print ("%.12f %.12f %.12f" % (fractimes[jj], scan_start, scan_stop)) 
                jd = round(fractimes[jj], ROUND_SCAN_INT)*comp_fac # ANDREW TODO precision??   

                if (np.floor(jd) >= np.floor(scan_start*comp_fac)) and (np.ceil(jd) <= np.ceil(comp_fac*scan_stop)):
                    start_vis.append(jj)

                    # TODO AIPS MEMO 117 says scan_times should be midpoint!, but AIPS data looks likes it's at the start? 
                    scan_times.append(scan_start + 0.5*scan_dur)# - rdate_jd_out)
                    scan_time_ints.append(scan_dur)
                    while (jj < len(fractimes) and np.floor(round(fractimes[jj],ROUND_SCAN_INT)*comp_fac) <= np.ceil(comp_fac*scan_stop)):
                        jj += 1
                    stop_vis.append(jj-1)
                else: 
                    continue

            if jj < len(fractimes):
                print (scan_arr[-1])
                print (round(scan_arr[-1][0],ROUND_SCAN_INT),round(scan_arr[-1][1],ROUND_SCAN_INT))
                print (jj, len(jds), round(jds[jj], ROUND_SCAN_INT))
                print("WARNING!!!: in save_uvfits NX table, didn't get to all entries when computing scan start/stop!")
                print (scan_times)
            time_nx = fits.Column(name="TIME", format="1D", unit='DAYS', array=np.array(scan_times))
            timeint_nx = fits.Column(name="TIME INTERVAL", format="1E", unit='DAYS', array=np.array(scan_time_ints))
            sourceid_nx = fits.Column(name="SOURCE ID",format="1J", unit='', array=np.ones(len(scan_times)))
            subarr_nx = fits.Column(name="SUBARRAY",format="1J", unit='', array=np.ones(len(scan_times)))
            freqid_nx = fits.Column(name="FREQ ID",format="1J", unit='', array=np.ones(len(scan_times)))
            startvis_nx = fits.Column(name="START VIS",format="1J", unit='', array=np.array(start_vis)+1)
            endvis_nx = fits.Column(name="END VIS",format="1J", unit='', array=np.array(stop_vis)+1)
            cols = fits.ColDefs([time_nx, timeint_nx, sourceid_nx, subarr_nx, freqid_nx, startvis_nx, endvis_nx])

            tbhdu = fits.BinTableHDU.from_columns(cols)
         
            # header information
            tbhdu.header.append(("EXTNAME","AIPS NX"))
            tbhdu.header.append(("EXTVER",1))

            hdulist_new.append(tbhdu) 
        except TypeError:
            print ("No NX table in saved uvfits")
    # Write final HDUList to file
    #hdulist.writeto(fname, overwrite=True)
    hdulist_new.writeto(fname, overwrite=True)

    return

def save_obs_oifits(obs, fname, flux=1.0):
    """ Save visibility data to oifits
        Polarization data is NOT saved
        Antenna diameter currently incorrect and the exact times are not correct in the datetime object
        Please contact Katie Bouman (klbouman@mit.edu) for any questions on this function
    """

    #TODO: Add polarization to oifits??
    print('Warning: save_oifits does NOT save polarimetric visibility data!')

    if (obs.polrep!='stokes'):
        raise Exception("save_obs_oifits only works with polrep 'stokes'!")

    # Normalizing by the total flux passed in - note this is changing the data inside the obs structure
    obs.data['vis'] /= flux
    obs.data['sigma'] /= flux

    data = obs.unpack(['u','v','amp','phase', 'sigma', 'time', 't1', 't2', 'tint'])
    biarr = obs.bispectra(mode="all", count="min")

    # extract the telescope names and parameters
    antennaNames = obs.tarr['site'] 
    sefd = obs.tarr['sefdr']
    antennaX = obs.tarr['x']
    antennaY = obs.tarr['y']
    antennaZ = obs.tarr['z']
    #antennaDiam = -np.ones(antennaX.shape) #todo: this is incorrect and there is just a dummy variable here
    antennaDiam = sefd # replace antennaDiam with SEFD for radio observtions

    # create dictionary
    union = {};
    union = ehtim.io.writeData.arrayUnion(antennaNames, union)

    # extract the integration time
    intTime = data['tint'][0]
    if not all(data['tint'][0] == item for item in np.reshape(data['tint'], (-1)) ):
        raise TypeError("The time integrations for each visibility are different")

    # get visibility information
    amp = data['amp']
    phase = data['phase']
    viserror = data['sigma']
    u = data['u']
    v = data['v']

    # convert antenna name strings to number identifiers
    ant1 = ehtim.io.writeData.convertStrings(data['t1'], union)
    ant2 = ehtim.io.writeData.convertStrings(data['t2'], union)

    # convert times to datetime objects
    time = data['time']
    dttime = np.array([datetime.datetime.utcfromtimestamp(x*60.0*60.0) for x in time]); #TODO: these do not correspond to the acutal times

    # get the bispectrum information
    bi = biarr['bispec']
    t3amp = np.abs(bi);
    t3phi = np.angle(bi, deg=1)
    t3amperr = biarr['sigmab']
    t3phierr = 180.0/np.pi * (1.0/t3amp) * t3amperr;
    uClosure = np.transpose(np.array([np.array(biarr['u1']), np.array(biarr['u2'])]));
    vClosure = np.transpose(np.array([np.array(biarr['v1']), np.array(biarr['v2'])]));

    # convert times to datetime objects
    timeClosure = biarr['time']
    dttimeClosure = np.array([datetime.datetime.utcfromtimestamp(x*60.0*60.0) for x in timeClosure]); #TODO: these do not correspond to the acutal times

    # convert antenna name strings to number identifiers
    biarr_ant1 = ehtim.io.writeData.convertStrings(biarr['t1'], union)
    biarr_ant2 = ehtim.io.writeData.convertStrings(biarr['t2'], union)
    biarr_ant3 = ehtim.io.writeData.convertStrings(biarr['t3'], union)
    antOrder = np.transpose(np.array([biarr_ant1, biarr_ant2, biarr_ant3]))

    # todo: check that putting the negatives on the phase and t3phi is correct
    ehtim.io.writeData.writeOIFITS(fname, obs.ra, obs.dec, obs.rf, obs.bw, intTime, amp, viserror, phase, viserror, u, v, ant1, ant2, dttime,
                          t3amp, t3amperr, t3phi, t3phierr, uClosure, vClosure, antOrder, dttimeClosure, antennaNames, antennaDiam, antennaX, antennaY, antennaZ)

    # Un-Normalizing by the total flux passed in - note this is changing the data inside the obs structure back to what it originally was
    obs.data['vis'] *= flux
    obs.data['sigma'] *= flux

    return
    
    

    
def save_dtype_txt(obs, fname, dtype='cphase'):
    """Save the dtype data in a text file.
    """

    head = ("SRC: %s \n" % obs.source +
                "RA: " + rastring(obs.ra) + "\n" + "DEC: " + decstring(obs.dec) + "\n" +
                "MJD: %i \n" % obs.mjd +
                "RF: %.4f GHz \n" % (obs.rf/1e9) +
                "BW: %.4f GHz \n" % (obs.bw/1e9) +
                "PHASECAL: %i \n" % obs.phasecal +
                "AMPCAL: %i \n" % obs.ampcal +
                "OPACITYCAL: %i \n" % obs.opacitycal +
                "DCAL: %i \n" % obs.dcal +
                "FRCAL: %i \n" % obs.frcal +
                "----------------------------------------------------------------------"+
                "------------------------------------------------------------------\n" +
                "Site       X(m)             Y(m)             Z(m)           "+
                "SEFDR      SEFDL     FR_PAR   FR_EL   FR_OFF  "+
                "DR_RE    DR_IM    DL_RE    DL_IM   \n"
            )

    for i in range(len(obs.tarr)):
        head += ("%-8s %15.5f  %15.5f  %15.5f  %8.2f   %8.2f  %5.2f   %5.2f   %5.2f  %8.4f %8.4f %8.4f %8.4f \n" % (obs.tarr[i]['site'],
                                                                  obs.tarr[i]['x'], obs.tarr[i]['y'], obs.tarr[i]['z'],
                                                                  obs.tarr[i]['sefdr'], obs.tarr[i]['sefdl'],
                                                                  obs.tarr[i]['fr_par'], obs.tarr[i]['fr_elev'], obs.tarr[i]['fr_off'],
                                                                  (obs.tarr[i]['dr']).real, (obs.tarr[i]['dr']).imag,
                                                                  (obs.tarr[i]['dl']).real, (obs.tarr[i]['dl']).imag
                                                                 ))

    if dtype=='cphase':
        outdata = obs.cphase
        head += (
                "----------------------------------------------------------------------"+
                "------------------------------------------------------------------\n" +
                "time (hr)     T1     T2      T3        U1 (lambda)     V1 (lambda)     U2 (lambda)     V2 (lambda)         U3 (lambda)     V3 (lambda)         Cphase (d) Sigmacp")
        fmts = ("%011.8f %6s %6s  %6s  %16.4f %16.4f  %16.4f  %16.4f  %16.4f  %16.4f  %10.4f  %10.8f")
        
    elif dtype=='logcamp':
        outdata = obs.logcamp
        head += (
                "----------------------------------------------------------------------"+
                "------------------------------------------------------------------\n" +
                "time (hr)     T1     T2      T3     T4     U1 (lambda)     V1 (lambda)      U2 (lambda)      V2 (lambda)         U3 (lambda)     V3 (lambda)       U4 (lambda)      V4 (lambda)           Logcamp     Sigmalogca")
        fmts = ("%011.8f %6s %6s  %6s %6s  %16.4f %16.4f  %16.4f  %16.4f  %16.4f %16.4f  %16.4f  %16.4f  %10.4f  %10.8f")

    elif dtype=='camp':
        outdata = obs.camp
        head += (
                "----------------------------------------------------------------------"+
                "------------------------------------------------------------------\n" +
                "time (hr)     T1     T2      T3     T4     U1 (lambda)     V1 (lambda)      U2 (lambda)      V2 (lambda)         U3 (lambda)     V3 (lambda)       U4 (lambda)      V4 (lambda)           Camp     Sigmaca")
        fmts = ("%011.8f %6s %6s  %6s %6s  %16.4f %16.4f  %16.4f  %16.4f  %16.4f %16.4f  %16.4f  %16.4f  %10.4f  %10.8f")

    elif dtype=='bs':
        outdata = obs.bispec
        head += (
                "----------------------------------------------------------------------"+
                "------------------------------------------------------------------\n" +
                "time (hr)     T1     T2      T3        U1 (lambda)     V1 (lambda)     U2 (lambda)     V2 (lambda)         U3 (lambda)     V3 (lambda)          Bispec   Sigmab")
        fmts = ("%011.8f %6s %6s  %6s  %16.4f %16.4f  %16.4f  %16.4f  %16.4f  %16.4f  %10.4f  %10.8f")

    elif dtype=='amp':
        outdata = obs.amp
        head += (
                "----------------------------------------------------------------------"+
                "------------------------------------------------------------------\n" +
                "time (hr) tint     T1     T2       U (lambda)     V (lambda)       Amp (Jy)     Ampsigma")
        fmts = ("%011.8f %4.2f %6s %6s  %16.4f %16.4f  %10.8f  %10.8f")

    else: 
        raise Exception(dtype + ' is not a possible data type!')

    np.savetxt(fname, outdata, header=head, fmt=fmts)
    return


