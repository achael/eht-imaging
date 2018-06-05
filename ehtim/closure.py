# closure.py
# a closure quantities data class
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
import itertools as it

##################################################################################################
# Closure object
##################################################################################################
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
        # sites = [self.tarr[i][0] for i in range(len(self.tarr))] # List of telescope names
        sites = [self.tarr[i][0] for i in range(len(self.tarr))] # List of telescope names
        bl   = list(it.combinations(sites,2))
        tris  = list(it.combinations(sites,3))
        quads = list(it.combinations(sites,4))

        # closure phase/amp. time curves of all possible tri/quadr-angles ("None" if no data)
        print("computing closure phases...")
        cp = []        
        for tri in tris: 
            cpdat = obs.cphase_tri(tri[0],tri[1],tri[2])
            time = cpdat['time']
            cphase  = cpdat['cphase']
            sigmacp =  cpdat['sigmacp']
            cpdat = np.array([time,cphase,sigmacp])
            cp.append(cpdat)

        print("computing closure amplitudes...")  
        ca = []      
        for quad in quads:
            cadat = obs.camp_quad(quad[0],quad[1],quad[3],quad[3])
            time = cadat['time']
            camp  = cadat['camp']
            sigmaca=  cadat['sigmaca']
            cadat = np.array([time,camp,sigmaca])
            ca.append(cadat)

        self.cp=[]
        self.tri=[]
        for i in range(len(cp)):
            if cp[i] is not None:
                (self.cp).append(cp[i])
                (self.tri).append(tris[i])

        self.ca=[]
        self.quad=[]
        for i in range(len(ca)):
            if ca[i] is not None:
                (self.ca).append(ca[i])
                (self.quad).append(quads[i])


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

