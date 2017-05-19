from __future__ import division
from __future__ import print_function
from builtins import range
from past.utils import old_div
import numpy as np
import string
import astropy.io.fits as fits
import datetime
import os

import ehtim.io.writeData
import ehtim.io.oifits

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

##################################################################################################
# Movie IO
##################################################################################################  

def save_mov_txt(mov, fname):
    """Save movie data to text files"""
    
    # Coordinate values
    pdimas = old_div(mov.psize,RADPERAS)
    xs = np.array([[j for j in range(mov.xdim)] for i in range(mov.ydim)]).reshape(mov.xdim*mov.ydim,1)
    xs = pdimas * (xs[::-1] - old_div(mov.xdim,2.0))
    ys = np.array([[i for j in range(mov.xdim)] for i in range(mov.ydim)]).reshape(mov.xdim*mov.ydim,1)
    ys = pdimas * (ys[::-1] - old_div(mov.xdim,2.0))
    
    for i in range(len(mov.frames)):
        fname_frame = fname + "%05d" % i
        # Data
        if len(mov.qframes):
            outdata = np.hstack((xs, ys, (mov.frames[i]).reshape(mov.xdim*mov.ydim, 1),
                                         (mov.qframes[i]).reshape(mov.xdim*mov.ydim, 1),
                                         (mov.uframes[i]).reshape(mov.xdim*mov.ydim, 1)))
            hf = "x (as)     y (as)       I (Jy/pixel)  Q (Jy/pixel)  U (Jy/pixel)"

            fmts = "%10.10f %10.10f %10.10f %10.10f %10.10f"
        else:
            outdata = np.hstack((xs, ys, (mov.frames[i]).reshape(mov.xdim*mov.ydim, 1)))
            hf = "x (as)     y (as)       I (Jy/pixel)"
            fmts = "%10.10f %10.10f %10.10f"
 
        # Header
        head = ("SRC: %s \n" % mov.source +
                    "RA: " + rastring(mov.ra) + "\n" + "DEC: " + decstring(mov.dec) + "\n" +
                    "MJD: %i \n" % (float(mov.mjd) + old_div(mov.start_hr,24.0) + i*mov.framedur/86400.0) + 
                    "RF: %.4f GHz \n" % (old_div(mov.rf,1e9)) + 
                    "FOVX: %i pix %f as \n" % (mov.xdim, pdimas * mov.xdim) +
                    "FOVY: %i pix %f as \n" % (mov.ydim, pdimas * mov.ydim) +
                    "------------------------------------\n" + hf)
         
        # Save
        np.savetxt(fname_frame, outdata, header=head, fmt=fmts)
        
        return


##################################################################################################
# Image IO
##################################################################################################  
def save_im_txt(im, fname):
    """Save image data to text file
    """

    # Coordinate values
    pdimas = old_div(im.psize,RADPERAS)
    xs = np.array([[j for j in range(im.xdim)] for i in range(im.ydim)]).reshape(im.xdim*im.ydim,1)
    xs = pdimas * (xs[::-1] - old_div(im.xdim,2.0))
    ys = np.array([[i for j in range(im.xdim)] for i in range(im.ydim)]).reshape(im.xdim*im.ydim,1)
    ys = pdimas * (ys[::-1] - old_div(im.xdim,2.0))
    
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
    head = ("SRC: %s \n" % im.source +
                "RA: " + rastring(im.ra) + "\n" + "DEC: " + decstring(im.dec) + "\n" +
                "MJD: %i \n" % im.mjd + 
                "RF: %.4f GHz \n" % (old_div(im.rf,1e9)) + 
                "FOVX: %i pix %f as \n" % (im.xdim, pdimas * im.xdim) +
                "FOVY: %i pix %f as \n" % (im.ydim, pdimas * im.ydim) +
                "------------------------------------\n" + hf)
     
    # Save
    np.savetxt(fname, outdata, header=head, fmt=fmts)    
    return

def save_im_fits(im, fname):
    """Save image data to FITS file
    """
            
    # Create header and fill in some values
    header = fits.Header()
    header['OBJECT'] = im.source
    header['CTYPE1'] = 'RA---SIN'
    header['CTYPE2'] = 'DEC--SIN'
    header['CDELT1'] = old_div(-im.psize,DEGREE)
    header['CDELT2'] = old_div(im.psize,DEGREE)
    header['OBSRA'] = im.ra * 180/12.
    header['OBSDEC'] = im.dec
    header['FREQ'] = im.rf
    header['MJD'] = float(im.mjd)
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
# Array IO
##################################################################################################  
  
def save_array_txt(arr, fname):
    """Save the array data in a text file
    """

    out = ("#Site      X(m)             Y(m)             Z(m)           "+
                "SEFDR      SEFDL     FR_PAR   FR_EL   FR_OFF  "+
                "DR_RE    DR_IM    DL_RE    DL_IM   \n")
    for scope in range(len(arr.tarr)):
        dat = (arr.tarr[scope]['site'], 
               arr.tarr[scope]['x'], arr.tarr[scope]['y'], arr.tarr[scope]['z'], 
               arr.tarr[scope]['sefdr'], arr.tarr[scope]['sefdl'],
               arr.tarr[scope]['fr_par'], arr.tarr[scope]['fr_elev'], arr.tarr[scope]['fr_off'],
               arr.tarr[scope]['dr'].real, arr.tarr[scope]['dr'].imag, arr.tarr[scope]['dl'].real, arr.tarr[scope]['dl'].imag
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

    # Get the necessary data and the header
    outdata = obs.unpack(['time', 'tint', 't1', 't2','tau1','tau2',
                           'u', 'v', 'amp', 'phase', 'qamp', 'qphase', 'uamp', 'uphase', 'vamp', 'vphase',
                           'sigma', 'qsigma', 'usigma', 'vsigma'])
    head = ("SRC: %s \n" % obs.source +
                "RA: " + rastring(obs.ra) + "\n" + "DEC: " + decstring(obs.dec) + "\n" +
                "MJD: %i \n" % obs.mjd + 
                "RF: %.4f GHz \n" % (old_div(obs.rf,1e9)) + 
                "BW: %.4f GHz \n" % (old_div(obs.bw,1e9)) +
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

    head += (
            "----------------------------------------------------------------------"+
            "------------------------------------------------------------------\n" +
            "time (hr) tint    T1     T2    Tau1   Tau2   U (lambda)       V (lambda)         "+
            "Iamp (Jy)    Iphase(d)  Qamp (Jy)    Qphase(d)   Uamp (Jy)    Uphase(d)   Vamp (Jy)    Vphase(d)   "+
            "Isigma (Jy)   Qsigma (Jy)   Usigma (Jy)   Vsigma (Jy)"
            )
      
    # Format and save the data
    fmts = ("%011.8f %4.2f %6s %6s  %4.2f   %4.2f  %16.4f %16.4f    "+
           "%10.8f %10.4f   %10.8f %10.4f    %10.8f %10.4f    %10.8f %10.4f    "+
           "%10.8f    %10.8f    %10.8f    %10.8f")
    np.savetxt(fname, outdata, header=head, fmt=fmts)
    return


def save_obs_uvfits(obs, fname):
    """Save visibility data to uvfits
       Needs template.UVP file
    """

    # Open template UVFITS
    dir_path = os.path.dirname(os.path.realpath(__file__))
    hdulist = fits.open(dir_path+'/template.UVP')

    ########################################################################
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
    
    #!AC TODO these antenna fields+header are questionable - look into them

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
    head = hdulist['AIPS AN'].header
    head['EXTNAME'] = 'AIPS AN'
    head['EXTVER'] = 1
    head['RDATE'] = '2000-01-01T00:00:00.0'
    head['GSTIA0'] = 114.38389781355     # !AC TODO ?? for jan 1 2000
    head['UT1UTC'] = 0.e0
    head['DATUTC'] = 0.e0
    head['TIMESYS'] = 'UTC'
    head['DEGPDY'] = 360.9856

    head['FREQ']= obs.rf
    head['FREQID'] = 1

    head['ARRNAM'] = 'ALMA'     #!AC TODO Can we change this field? 
    head['XYZHAND'] = 'RIGHT'
    head['ARRAYX'] = 0.e0
    head['ARRAYY'] = 0.e0
    head['ARRAYZ'] = 0.e0
    head['POLARX'] = 0.e0
    head['POLARY'] = 0.e0

    head['NUMORB'] = 0
    head['NO_IF'] = 1
    head['NOPCAL'] = 0            #!AC changed from 1
    head['POLTYPE'] = 'APPROX'

    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs([col1,col2,col25,col3,col4,col5,col6,col7,col8,col9,col10,col11,colfin]), name='AIPS AN', header=head)
    hdulist['AIPS AN'] = tbhdu

    ########################################################################
    # Data table
    # Data header (based on the BU format)
    #header = fits.Header()
    header = hdulist[0].header
    header['OBSRA'] = obs.ra * 180./12.
    header['OBSDEC'] = obs.dec
    header['OBJECT'] = obs.source
    header['MJD'] = float(obs.mjd)
    header['BUNIT'] = 'JY'
    header['VELREF'] = 3        # !AC TODO ??
    header['ALTRPIX'] = 1.e0
    header['TELESCOP'] = 'ALMA' # !AC TODO Can we change this field?  
    header['INSTRUME'] = 'ALMA'
    header['CTYPE2'] = 'COMPLEX'
    header['CRVAL2'] = 1.e0
    header['CDELT2'] = 1.e0
    header['CRPIX2'] = 1.e0
    header['CROTA2'] = 0.e0
    header['CTYPE3'] = 'STOKES'
    header['CRVAL3'] = -1.e0
    header['CRDELT3'] = -1.e0
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
    header['PSCAL1'] = old_div(1,obs.rf)
    header['PZERO1'] = 0.e0
    header['PTYPE2'] = 'VV---SIN'
    header['PSCAL2'] = old_div(1,obs.rf)
    header['PZERO2'] = 0.e0
    header['PTYPE3'] = 'WW---SIN'
    header['PSCAL3'] = old_div(1,obs.rf)
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
            
    # Get data
    obsdata = obs.unpack(['time','tint','u','v','vis','qvis','uvis','vvis','sigma','qsigma','usigma','vsigma','t1','t2','tau1','tau2'])
    ndat = len(obsdata['time'])
    
    # times and tints
    #jds = (obs.mjd + 2400000.5) * np.ones(len(obsdata))
    #fractimes = (obsdata['time'] / 24.0) 
    jds = (2400000.5 + obs.mjd) * np.ones(len(obsdata))
    fractimes = (old_div(obsdata['time'], 24.0)) 
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
    rr = obsdata['vis'] + obsdata['vvis']
    ll = obsdata['vis'] - obsdata['vvis']
    rl = obsdata['qvis'] + 1j*obsdata['uvis']
    lr = obsdata['qvis'] - 1j*obsdata['uvis']
    
    weightrr = old_div(1, (obsdata['sigma']**2 + obsdata['vsigma']**2))
    weightll = old_div(1, (obsdata['sigma']**2 + obsdata['vsigma']**2))
    weightrl = old_div(1, (obsdata['qsigma']**2 + obsdata['usigma']**2))
    weightlr = old_div(1, (obsdata['qsigma']**2 + obsdata['usigma']**2))
                            
    # Data array
    outdat = np.zeros((ndat, 1, 1, 1, 1, 4, 3))
    outdat[:,0,0,0,0,0,0] = np.real(rr)
    outdat[:,0,0,0,0,0,1] = np.imag(rr)
    outdat[:,0,0,0,0,0,2] = weightrr
    outdat[:,0,0,0,0,1,0] = np.real(ll)
    outdat[:,0,0,0,0,1,1] = np.imag(ll)
    outdat[:,0,0,0,0,1,2] = weightll
    outdat[:,0,0,0,0,2,0] = np.real(rl)
    outdat[:,0,0,0,0,2,1] = np.imag(rl)
    outdat[:,0,0,0,0,2,2] = weightrl
    outdat[:,0,0,0,0,3,0] = np.real(lr)
    outdat[:,0,0,0,0,3,1] = np.imag(lr)
    outdat[:,0,0,0,0,3,2] = weightlr
    
    # Save data
    
    pars = ['UU---SIN', 'VV---SIN', 'WW---SIN', 'BASELINE', 'DATE', 'DATE',
            'INTTIM', 'TAU1', 'TAU2']
    x = fits.GroupData(outdat, parnames=pars,
        pardata=[u, v, np.zeros(ndat), bl, jds, fractimes, tints,tau1,tau2],
        bitpix=-32)

            
    hdulist[0].data = x
    hdulist[0].header = header

    ##################################################################################
    # AIPS FQ TABLE -- Thanks to Kazu
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
    hdulist.append(tbhdu)
     
    # Write final HDUList to file
    hdulist.writeto(fname, overwrite=True)
            
    return


  

def save_obs_oifits(obs, fname, flux=1.0):
    """ Save visibility data to oifits
        Polarization data is NOT saved
        Antenna diameter currently incorrect and the exact times are not correct in the datetime object
        Please contact Katie Bouman (klbouman@mit.edu) for any questions on this function 
    """

    #TODO: Add polarization to oifits??
    print('Warning: save_oifits does NOT save polarimetric visibility data!')
    
    # Normalizing by the total flux passed in - note this is changing the data inside the obs structure
    obs.data['vis'] /= flux
    obs.data['sigma'] /= flux
    
    data = obs.unpack(['u','v','amp','phase', 'sigma', 'time', 't1', 't2', 'tint'])
    biarr = obs.bispectra(mode="all", count="min")

    # extract the telescope names and parameters
    antennaNames = obs.tarr['site'] #np.array(obs.tkey.keys())
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
    dttime = np.array([datetime.datetime.utcfromtimestamp(x*60*60) for x in time]); #TODO: these do not correspond to the acutal times
    
    # get the bispectrum information
    bi = biarr['bispec']
    t3amp = np.abs(bi);
    t3phi = np.angle(bi, deg=1)
    t3amperr = biarr['sigmab']
    t3phierr = 180.0/np.pi * (old_div(1,t3amp)) * t3amperr;
    uClosure = np.transpose(np.array([np.array(biarr['u1']), np.array(biarr['u2'])]));
    vClosure = np.transpose(np.array([np.array(biarr['v1']), np.array(biarr['v2'])]));
    
    # convert times to datetime objects
    timeClosure = biarr['time']
    dttimeClosure = np.array([datetime.datetime.utcfromtimestamp(x) for x in timeClosure]); #TODO: these do not correspond to the acutal times

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

