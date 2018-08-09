# cal_helpers.py
# helper functions for calibration
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

import numpy as np
import sys
import itertools as it
import copy
import os

import ehtim.obsdata
from ehtim.observing.obs_helpers import *

from multiprocessing import Process, Value, Lock

ZBLCUTOFF = 1.e7;

class Counter(object):
    """counter object for sharing among multiprocessing jobs"""

    def __init__(self,initval=0,maxval=0):
        self.val = Value('i',initval)
        self.maxval = maxval
        self.lock = Lock()
    def increment(self):
        with self.lock:
            self.val.value += 1
    def value(self):
        with self.lock:
            return self.val.value

def make_cluster_data(obs, zbl_uvdist_max=ZBLCUTOFF):
    """Cluster sites in an observation into groups with intra-group basline length not exceeding zbl_uvdist_max
    """

    clusters = []
    clustered_sites = []
    for i1 in range(len(obs.tarr)):
        t1 = obs.tarr[i1]

        if t1['site'] in clustered_sites:
            continue

        csites = [t1['site']]
        clustered_sites.append(t1['site'])
        for i2 in range(len(obs.tarr))[i1:]:
            t2 = obs.tarr[i2]
            if t2['site'] in clustered_sites:
                continue

            site1coord = np.array([t1['x'], t1['y'], t1['z']])
            site2coord = np.array([t2['x'], t2['y'], t2['z']])
            uvdist = np.sqrt(np.sum((site1coord-site2coord)**2)) / (C / obs.rf)

            if uvdist < zbl_uvdist_max:
                csites.append(t2['site'])
                clustered_sites.append(t2['site'])
        clusters.append(csites)

    clusterdict = {}
    for site in obs.tarr['site']:
        for k in range(len(clusters)):
            if site in  clusters[k]:
                clusterdict[site] = k

    clusterbls = [set(comb) for comb in it.combinations(range(len(clusterdict)),2)]

    cluster_data = (clusters, clusterdict, clusterbls)

    return cluster_data

#def norm_zbl(obs, flux=1.):
#    """Normalize scans to the zero baseline flux
#    """
#    # V = model visibility, V' = measured visibility, G_i = site gain
#    # G_i * conj(G_j) * V_ij = V'_ij

#    scans = obs.tlist()
#    n = len(scans)
#    data_norm = []
#    i = 0
#    for scan in scans:
#        i += 1
#        uvdist = np.sqrt(scan['u']**2 + scan['v']**2)
#        scan_zbl = np.abs(scan['vis'][np.argmin(uvdist)])

#        scan['vis'] = scan['vis']/scan_zbl
#        scan['sigma'] = scan['sigma']/scan_zbl
#        scan['qvis'] = scan['qvis']/scan_zbl
#        scan['qsigma'] = scan['qsigma']/scan_zbl
#        scan['uvis'] = scan['uvis']/scan_zbl
#        scan['usigma'] = scan['usigma']/scan_zbl
#        scan['vvis'] = scan['vvis']/scan_zbl
#        scan['vsigma'] = scan['vsigma']/scan_zbl

#        if len(data_norm):
#            data_norm = np.hstack((data_norm, scan))
#        else:
#            data_norm = scan

#    obs_cal = ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw, data_norm, obs.tarr,
#                                    source=obs.source, mjd=obs.mjd, timetype=obs.timetype,
#                                    ampcal=obs.ampcal, phasecal=obs.phasecal, 
#                                    dcal=obs.dcal, frcal=obs.frcal, scantable=obs.scans)
#    return obs_cal

