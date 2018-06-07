# const_def.py
# useful constants and definitions
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

from ehtim.observing.pulses import *

EP = 1.0e-10
C = 299792458.0
DEGREE = 3.141592653589/180.0
HOUR = 15.0*DEGREE
RADPERAS = DEGREE/3600.0
RADPERUAS = RADPERAS*1.e-6

# Default Parameters
SOURCE_DEFAULT = "SgrA"
RA_DEFAULT = 17.761122472222223
DEC_DEFAULT = -28.992189444444445
RF_DEFAULT = 230e9
MJD_DEFAULT = 51544
PULSE_DEFAULT = trianglePulse2D

# Telescope elevation cuts (degrees)
ELEV_LOW = 10.0
ELEV_HIGH = 85.0

TAUDEF = 0.1 # Default Optical Depth
GAINPDEF = 0.1 # Default rms of gain errors
DTERMPDEF = 0.1 # Default rms of D-term errors

# Sgr A* Kernel Values (Bower et al., in uas/cm^2)
FWHM_MAJ = 1.309 * 1000 # in uas
FWHM_MIN = 0.64 * 1000
POS_ANG = 78 # in degree, E of N

# Observation recarray datatypes
DTARR = [('site', 'U32'), ('x','f8'), ('y','f8'), ('z','f8'),
         ('sefdr','f8'),('sefdl','f8'),('dr','c16'),('dl','c16'),
         ('fr_par','f8'),('fr_elev','f8'),('fr_off','f8')]

DTPOL = [('time','f8'),('tint','f8'),
         ('t1','U32'),('t2','U32'),
         ('tau1','f8'),('tau2','f8'),
         ('u','f8'),('v','f8'),
         ('vis','c16'),('qvis','c16'),('uvis','c16'),('vvis','c16'),
         ('sigma','f8'),('qsigma','f8'),('usigma','f8'),('vsigma','f8')]

DTBIS = [('time','f8'),('t1','U32'),('t2','U32'),('t3','U32'),
         ('u1','f8'),('v1','f8'),('u2','f8'),('v2','f8'),('u3','f8'),('v3','f8'),
         ('bispec','c16'),('sigmab','f8')]

DTCPHASE = [('time','f8'),('t1','U32'),('t2','U32'),('t3','U32'),
            ('u1','f8'),('v1','f8'),('u2','f8'),('v2','f8'),('u3','f8'),('v3','f8'),
            ('cphase','f8'),('sigmacp','f8')]

DTCAMP = [('time','f8'),('t1','U32'),('t2','U32'),('t3','U32'),('t4','U32'),
          ('u1','f8'),('v1','f8'),('u2','f8'),('v2','f8'),
          ('u3','f8'),('v3','f8'),('u4','f8'),('v4','f8'),
          ('camp','f8'),('sigmaca','f8')]

DTCAL = [('time','f8'), ('rscale','c16'), ('lscale','c16')]

DTSCANS = [('time','f8'),('interval','f8'),('startvis','f8'),('endvis','f8')]


# Observation fields for plotting and retrieving data
FIELDS = ['time','time_utc','time_gmst',
          'tint','u','v','uvdist',
          't1','t2','tau1','tau2',
          'el1','el2','hr_ang1','hr_ang2','par_ang1','par_ang2',
          'vis','amp','phase','snr',
          'qvis','qamp','qphase','qsnr',
          'uvis','uamp','uphase','usnr',
          'vvis','vamp','vphase','vsnr',
          'sigma','qsigma','usigma','vsigma',
          'sigma_phase','qsigma_phase','usigma_phase','vsigma_phase',
          'psigma_phase','msigma_phase',
          'pvis','pamp','pphase','psnr',
          'm','mamp','mphase','msnr'
          'rrvis','rramp','rrphase','rrsnr','rrsigma','rrsigma_phase',
          'llvis','llamp','llphase','llsnr','llsigma','llsigma_phase'
          'rlvis','rlamp','rlphase','rlsnr','rlsigma','rlsigma_phase'
          'lrvis','lramp','lrphase','lrsnr','lrsigma','lrsigma_phase']

#miscellaneous functions

#TODO this makes a copy -- is there a faster robust way?
def recarr_to_ndarr(x,typ):
    """converts a record array x to a normal ndarray with all fields converted to datatype typ
    """

    fields = x.dtype.names
    shape = x.shape + (len(fields),)
    dt = [(name,typ) for name in fields]
    y = x.astype(dt).view(typ).reshape(shape)
    return y
