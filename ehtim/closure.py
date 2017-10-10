from __future__ import division
from __future__ import print_function

import numpy as np
import itertools as it

class Closure(object):

    def __init__(self, obs):

        # copy meta data from Obsdata
        self.source = obs.source
        self.ra = obs.ra
        self.dec = obs.dec
        self.rf = obs.rf
        self.bw = obs.bw
        self.ampcal = obs.ampcal
        self.phasecal = obs.phasecal
        self.opacitycal = obs.opacitycal
        self.dcal = obs.dcal
        self.frcal = obs.frcal
        self.timetype = obs.timetype
        self.tarr = obs.tarr
        self.tkey = obs.tkey

	# make list of all possible tri/quadr-angles
        #sites = [self.tarr[i][0] for i in range(len(self.tarr))] # List of telescope names
        sites = [self.tarr[i][0] for i in range(len(self.tarr))] # List of telescope names
        bl   = list(it.combinations(sites,2))
        tri  = list(it.combinations(sites,3))
        quad = list(it.combinations(sites,4))

        # closure phase/amp. time curves of all possible tri/quadr-angles ("None" if no data)
        cp = obs.get_cphase_curves(tri)
        ca = obs.get_camp_curves(quad)
#        va = obs.get_amp_curves(bl)

#        # remove bl/tri/quadr-angles that have no data
#        self.va=[]
#        self.bl=[]
#        for i in range(len(va)):
#            if va[i] is not None:
#                (self.va).append(va[i])
#                (self.bl).append(bl[i])

        self.cp=[]
        self.tri=[]
        for i in range(len(cp)):
            if cp[i] is not None:
                (self.cp).append(cp[i])
                (self.tri).append(tri[i])

        self.ca=[]
        self.quad=[]
        for i in range(len(ca)):
            if ca[i] is not None:
                (self.ca).append(ca[i])
                (self.quad).append(quad[i])


    def record_cp( self, tri_id ):
        cp = np.array(self.cp[tri_id])
        fname = "cp_%s-%s-%s"%(self.tri[tri_id][0],self.tri[tri_id][1],self.tri[tri_id][2])
        f = open(fname,"w")
        for i in range(len(cp[0])):
            f.write("%f %f %f\n"%(cp[0][i],cp[1][i],cp[2][i]))
        f.close()


    def record_ca( self, quad_id ):
        ca = np.array(self.ca[quad_id])
        fname = "ca_%s-%s-%s-%s"%(self.quad[quad_id][0],self.quad[quad_id][1],self.quad[quad_id][2],self.quad[quad_id][3])
        f = open(fname,"w")
        for i in range(len(ca[0])):
            f.write("%f %f %f\n"%(ca[0][i],ca[1][i],ca[2][i]))
        f.close()

