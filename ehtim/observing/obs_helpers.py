from __future__ import division
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div

import astropy.time as at
import numpy as np

from ehtim.const_def import *

##################################################################################################
# Other Functions
##################################################################################################
def gauss_uv(u, v, flux, beamparams, x=0., y=0.):
    """Return the value of the Gaussian FT with 
       beamparams is [FWHMmaj, FWHMmin, theta, x, y], all in radian
       theta is the orientation angle measured E of N
    """

    sigma_maj = old_div(beamparams[0], (2*np.sqrt(2*np.log(2)))) 
    sigma_min = old_div(beamparams[1], (2*np.sqrt(2*np.log(2))))
    theta = -beamparams[2] # theta needs to be negative in this convention!
    #try: 
    #	x=beamparams[3]
    #	y=beamparams[4] 
    #except IndexError:
    #	x=y=0.0
        
    # Covariance matrix
    a = (sigma_min * np.cos(theta))**2 + (sigma_maj*np.sin(theta))**2
    b = (sigma_maj * np.cos(theta))**2 + (sigma_min*np.sin(theta))**2
    c = (sigma_min**2 - sigma_maj**2) * np.cos(theta) * np.sin(theta)
    m = np.array([[a, c], [c, b]])
    
    uv = np.array([[u[i],v[i]] for i in range(len(u))])
    x2 = np.array([np.dot(uvi,np.dot(m,uvi)) for uvi in uv])   
    #x2 = np.dot(uv, np.dot(m, uv.T))
    g = np.exp(-2 * np.pi**2 * x2)
    p = np.exp(-2j * np.pi * (u*x + v*y))

    return flux * g * p
     
def sgra_kernel_uv(rf, u, v):
    """Return the value of the Sgr A* scattering kernel at a given u,v pt (in lambda), 
       at a given frequency rf (in Hz).
       Values from Bower et al.
    """
    
    lcm = (old_div(C,rf)) * 100 # in cm
    sigma_maj = FWHM_MAJ * (lcm**2) / (2*np.sqrt(2*np.log(2))) * RADPERUAS
    sigma_min = FWHM_MIN * (lcm**2) / (2*np.sqrt(2*np.log(2))) * RADPERUAS
    theta = -POS_ANG * DEGREE # theta needs to be negative in this convention!
    
    #bp = [fwhm_maj, fwhm_min, theta]
    #g = gauss_uv(u, v, 1., bp, x=0., y=0.)
    
    # Covariance matrix
    a = (sigma_min * np.cos(theta))**2 + (sigma_maj*np.sin(theta))**2
    b = (sigma_maj * np.cos(theta))**2 + (sigma_min*np.sin(theta))**2
    c = (sigma_min**2 - sigma_maj**2) * np.cos(theta) * np.sin(theta)
    m = np.array([[a, c], [c, b]])
    uv = np.array([u,v])
    
    x2 = np.dot(uv, np.dot(m, uv))
    g = np.exp(-2 * np.pi**2 * x2)
    
    return g

def sgra_kernel_params(rf):
    """Return elliptical gaussian parameters in radian for the Sgr A* scattering ellipse at a given frequency
       Values from Bower et al.
    """
    
    lcm = (old_div(C,rf)) * 100 # in cm
    fwhm_maj_rf = FWHM_MAJ * (lcm**2)  * RADPERUAS
    fwhm_min_rf = FWHM_MIN * (lcm**2)  * RADPERUAS
    theta = POS_ANG * DEGREE
    
    return np.array([fwhm_maj_rf, fwhm_min_rf, theta])


def blnoise(sefd1, sefd2, tint, bw):
    """Determine the standard deviation of Gaussian thermal noise on a baseline 
       This is the noise on the rr/ll/rl/lr correlation, not the stokes parameter
       2-bit quantization is responsible for the 0.88 factor
    """
    
    #!AC TODO Is the factor of sqrt(2) correct? 
    #noise = np.sqrt(sefd1*sefd2/(2*bw*tint))/0.88
    noise = old_div(np.sqrt(sefd1*sefd2/(bw*tint)),0.88)
    return noise

def merr(sigma, qsigma, usigma, I, m):
    """Return the error in mbreve real and imaginary parts"""

    err = np.sqrt(old_div((qsigma**2 + usigma**2 + (sigma*np.abs(m))**2), (np.abs(I) ** 2)))
    # old formula assumes all sigmas the same
    #err = sigma * np.sqrt((2 + np.abs(m)**2)/ (np.abs(I) ** 2))     
    return err

def cerror(sigma):
    """Return a complex number drawn from a circular complex Gaussian of zero mean
    """
    return np.random.normal(loc=0,scale=sigma) + 1j*np.random.normal(loc=0,scale=sigma)

def hashrandn(*args):
    """set the seed according to a collection of arguments and return random gaussian var
    """
    np.random.seed(hash(",".join(map(repr,args))) % 4294967295)
    return np.random.randn()

def hashrand(*args):
    """set the seed according to a collection of arguments and return random number in 0,1
    """
    np.random.seed(hash(",".join(map(repr,args))) % 4294967295)
    return np.random.rand()

def image_centroid(im):
    """Return the image centroid (in radians)
    """

    xlist = np.arange(0,-im.xdim,-1)*im.psize + old_div((im.psize*im.xdim),2.0) - old_div(im.psize,2.0)
    ylist = np.arange(0,-im.ydim,-1)*im.psize + old_div((im.psize*im.ydim),2.0) - old_div(im.psize,2.0)

    x0 = old_div(np.sum(np.outer(0.0*ylist+1.0, xlist).ravel()*im.imvec),np.sum(im.imvec))
    y0 = old_div(np.sum(np.outer(ylist, 0.0*xlist+1.0).ravel()*im.imvec),np.sum(im.imvec))

    return np.array([x0, y0])

def ftmatrix_centered(im, pdim, xdim, ydim, uvlist, pulse=PULSE_DEFAULT):
    """Return a DFT matrix for the xdim*ydim image with pixel width pdim
       that extracts spatial frequencies of the uv points in uvlist.
       in this version, it puts the image centroid at the origin
    """

    # !AC TODO : there is a residual value for the center being around 0, maybe we should chop this off to be exactly 0
    # Coordinate matrix for COM constraint
    xlist = np.arange(0,-xdim,-1)*pdim + old_div((pdim*xdim),2.0) - old_div(pdim,2.0)
    ylist = np.arange(0,-ydim,-1)*pdim + old_div((pdim*ydim),2.0) - old_div(pdim,2.0)
    x0 = old_div(np.sum(np.outer(0.0*ylist+1.0, xlist).ravel()*im),np.sum(im))
    y0 = old_div(np.sum(np.outer(ylist, 0.0*xlist+1.0).ravel()*im),np.sum(im))

    #Now shift the lists
    xlist = xlist - x0
    ylist = ylist - y0

    ftmatrices = [pulse(2*np.pi*uv[0], 2*np.pi*uv[1], pdim, dom="F") * np.outer(np.exp(-2j*np.pi*ylist*uv[1]), np.exp(-2j*np.pi*xlist*uv[0])) for uv in uvlist] #list of matrices at each freq
    ftmatrices = np.reshape(np.array(ftmatrices), (len(uvlist), xdim*ydim))
    return ftmatrices
      
def ftmatrix(pdim, xdim, ydim, uvlist, pulse=PULSE_DEFAULT, mask=[]):
    """Return a DFT matrix for the xdim*ydim image with pixel width pdim
       that extracts spatial frequencies of the uv points in uvlist.
    """

    xlist = np.arange(0,-xdim,-1)*pdim + old_div((pdim*xdim),2.0) - old_div(pdim,2.0)
    ylist = np.arange(0,-ydim,-1)*pdim + old_div((pdim*ydim),2.0) - old_div(pdim,2.0)

    # original sign convention
    #ftmatrices = [pulse(2*np.pi*uv[0], 2*np.pi*uv[1], pdim, dom="F") * np.outer(np.exp(-2j*np.pi*ylist*uv[1]), np.exp(-2j*np.pi*xlist*uv[0])) for uv in uvlist] #list of matrices at each freq
    
    # changed the sign convention to agree with BU data (Jan 2017)
    ftmatrices = [pulse(2*np.pi*uv[0], 2*np.pi*uv[1], pdim, dom="F") * np.outer(np.exp(2j*np.pi*ylist*uv[1]), np.exp(2j*np.pi*xlist*uv[0])) for uv in uvlist] #list of matrices at each freq
    
    ftmatrices = np.reshape(np.array(ftmatrices), (len(uvlist), xdim*ydim))

    if len(mask):
        ftmatrices = ftmatrices[:,mask]
        
    return ftmatrices

def ticks(axisdim, psize, nticks=8):
    """Return a list of ticklocs and ticklabels
       psize should be in desired units
    """
    
    axisdim = int(axisdim)
    nticks = int(nticks)
    if not axisdim % 2: axisdim += 1
    if nticks % 2: nticks -= 1
    tickspacing = old_div(float((axisdim-1)),nticks)
    ticklocs = np.arange(0, axisdim+1, tickspacing)
    ticklabels= np.around(psize * np.arange(old_div((axisdim-1),2.), old_div(-(axisdim),2.), -tickspacing), decimals=1)
    return (ticklocs, ticklabels)

def power_of_two(target):
    """Finds the next greatest power of two
    """
    cur = 1
    if target > 1:
        for i in range(0, int(target)):
            if (cur >= target):
                return cur
            else: cur *= 2
    else:
        return 1


def paritycompare(perm1, perm2):
    """Compare the parity of two permutations.
       Assume both lists are equal length and with same elements
       Copied from: http://stackoverflow.com/questions/1503072/how-to-check-if-permutations-have-equal-parity
    """
    
    perm2 = list(perm2)
    perm2_map = dict((v, i) for i,v in enumerate(perm2))
    transCount=0
    for loc, p1 in enumerate(perm1):
        p2 = perm2[loc]
        if p1 != p2:
            sloc = perm2_map[p1]
            perm2[loc], perm2[sloc] = p1, p2
            perm2_map[p1], perm2_map[p2] = sloc, loc
            transCount += 1
    
    if not (transCount % 2): return 1
    else: return  -1

def amp_debias(vis, sigma):
    """Return debiased visibility amplitudes
    """
    
    # !AC TODO: what to do if deb2 < 0? Currently we do nothing
    deb2 = np.abs(vis)**2 - np.abs(sigma)**2
    if type(deb2) == float or type(deb2)==np.float64:
        if deb2 < 0.0: return np.abs(vis)
        else: return np.sqrt(deb2)
    else:
        lowsnr = deb2 < 0.0
        deb2[lowsnr] = np.abs(vis[lowsnr])**2
        return np.sqrt(deb2)
        
def sigtype(datatype):
    """Return the type of noise corresponding to the data type
    """
    
    datatype = str(datatype)
    if datatype in ['vis', 'amp']: sigmatype='sigma'
    elif datatype in ['qvis', 'qamp']: sigmatype='qsigma'
    elif datatype in ['uvis', 'uamp']: sigmatype='usigma'
    elif datatype in ['vvis', 'vamp']: sigmatype='vsigma'
    elif datatype in ['pvis', 'pamp']: sigmatype='psigma'                
    elif datatype in ['pvis', 'pamp']: sigmatype='psigma'
    elif datatype in ['m', 'mamp']: sigmatype='msigma'
    elif datatype in ['phase']: sigmatype='sigma_phase'
    elif datatype in ['qphase']: sigmatype='qsigma_phase'
    elif datatype in ['uphase']: sigmatype='usigma_phase'
    elif datatype in ['vphase']: sigmatype='vsigma_phase'
    elif datatype in ['pphase']: sigmatype='psigma_phase'
    elif datatype in ['mphase']: sigmatype='msigma_phase'
    else: sigmatype = False
    
    return sigmatype                                    
    
           
def rastring(ra):
    """Convert a ra in fractional hours to formatted string
    """
    h = int(ra)
    m = int((ra-h)*60.)
    s = (ra-h-old_div(m,60.))*3600.
    out = "%2i h %2i m %2.4f s" % (h,m,s)
    return out 

def decstring(dec):
    """Convert a dec in fractional degrees to formatted string
    """
    
    deg = int(dec)
    m = int((abs(dec)-abs(deg))*60.)
    s = (abs(dec)-abs(deg)-old_div(m,60.))*3600.
    out = "%2i deg %2i m %2.4f s" % (deg,m,s)
    return out

def gmtstring(gmt):
    """Convert a gmt in fractional hours to formatted string
    """
    
    if gmt > 24.0: gmt = gmt-24.0
    h = int(gmt)
    m = int((gmt-h)*60.)
    s = (gmt-h-old_div(m,60.))*3600.
    out = "%02i:%02i:%2.4f" % (h,m,s)
    return out 

def utc_to_gmst(utc, mjd): 
    """Convert utc times in hours to gmst using astropy
    """

    mjd=int(mjd) #MJD should always be an integer, but was float in older versions of the code
    time_obj = at.Time(old_div(utc,24.0) + np.floor(mjd), format='mjd', scale='utc') 
    time_sidereal = time_obj.sidereal_time('mean','greenwich').hour
    return time_sidereal
    
def earthrot(vecs, thetas):
    """Rotate a vector / array of vectors about the z-direction by theta / array of thetas (radian)
    """

    if len(vecs.shape)==1: 
        vecs = np.array([vecs])
    if np.isscalar(thetas):
        thetas = np.array([thetas for i in range(len(vecs))])

    # equal numbers of sites and angles
    if len(thetas) == len(vecs):
        rotvec = np.array([np.dot(np.array(((np.cos(thetas[i]),-np.sin(thetas[i]),0),(np.sin(thetas[i]),np.cos(thetas[i]),0),(0,0,1))), vecs[i]) 
                       for i in range(len(vecs))])
    # only one rotation angle, many sites
    elif len(thetas) == 1:
        rotvec = np.array([np.dot(np.array(((np.cos(thetas[0]),-np.sin(thetas[0]),0),(np.sin(thetas[0]),np.cos(thetas[0]),0),(0,0,1))), vecs[i]) 
                       for i in range(len(vecs))])
    # only one site, many angles
    elif len(vecs) == 1:
        rotvec = np.array([np.dot(np.array(((np.cos(thetas[i]),-np.sin(thetas[i]),0),(np.sin(thetas[i]),np.cos(thetas[i]),0),(0,0,1))), vecs[0]) 
                       for i in range(len(thetas))]) 
    else:
        raise Exception("Unequal numbers of vectors and angles in earthrot(vecs, thetas)!")
                                            
    #if rotvec.shape[0]==1: rotvec = rotvec[0]
    return rotvec

def elev(obsvecs, sourcevec):
    """Return the elevation of a source with respect to an observer/observers in radians
       obsvec can be an array of vectors but sourcevec can ONLY be a single vector
    """
       
    if len(obsvecs.shape)==1:
        obsvecs=np.array([obsvecs])

    anglebtw = np.array([np.dot(obsvec,sourcevec)/np.linalg.norm(obsvec)/np.linalg.norm(sourcevec) for obsvec in obsvecs])
    el = 0.5*np.pi - np.arccos(anglebtw)

    return el
        
def elevcut(obsvecs, sourcevec, elevmin=ELEV_LOW, elevmax=ELEV_HIGH):
    """Return True if a source is observable by a telescope vector
    """
    
    angles = old_div(elev(obsvecs, sourcevec),DEGREE)
    
    return (angles > elevmin) * (angles < elevmax)

def hr_angle(gst, lon, ra):
    """Computes the hour angle for a source at RA, observer at longitude long, and GMST time gst
       gst in hours, ra & lon ALL in radian
       longitude positive east
    """

    hr_angle = np.mod(gst + lon - ra, 2*np.pi)
    return hr_angle
    
def par_angle(hr_angle, lat, dec):
    """Compute the parallactic angle for a source at hr_angle and dec for an observer with latitude lat. 
       All angles in radian
    """
       
    num = np.sin(hr_angle)*np.cos(lat)
    denom = np.sin(lat)*np.cos(dec) - np.cos(lat)*np.sin(dec)*np.cos(hr_angle)
    
    return np.arctan2(num, denom)

def xyz_2_latlong(obsvecs): 
    """Compute the (geocentric) latitude and longitude of a site at geocentric position x,y,z 
       The output is in radians
    """

    if len(obsvecs.shape)==1:
        obsvecs=np.array([obsvecs])        
    out = []
    for obsvec in obsvecs:
        x = obsvec[0]
        y = obsvec[1]
        z = obsvec[2]
        lon = np.array(np.arctan2(y,x))
        lat = np.array(np.arctan2(z, np.sqrt(x**2+y**2)))
        out.append([lat,lon])
        
    out = np.array(out)

    #if out.shape[0]==1: out = out[0]
    return out
