# comparisons.py
# Image Consistency Comparisons
#
#    Copyright (C) 2018 Katie Bouman
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

# test edit
from __future__ import print_function
from itertools import cycle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

import sys
import os
import argparse

import ehtim as eh
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
import glob
from itertools import cycle


def image_consistency(imarr, beamparams, metric='nxcorr', blursmall=True, beam_max=1.0, beam_steps=5, savepath=[]):

    # get the pixel sizes and fov to compare images at
    (min_psize, max_fov) = get_psize_fov(imarr)

    # initialize matrix matrix
    metric_mtx = np.zeros([len(imarr), len(imarr), beam_steps])

    # get the different fracsteps
    fracsteps = np.linspace(0,beam_max,beam_steps)

    # loop over the different beam sizes
    for fracidx in range(beam_steps):
        #print(fracidx)
        # look at every pair of images and compute their beam convolved metrics
        for i in range(len(imarr)):
            img1 = imarr[i]
            if fracsteps[fracidx]>0:
                img1 = img1.blur_gauss(beamparams, fracsteps[fracidx])

            for j in range(i+1, len(imarr)):
                img2 = imarr[j]
                if fracsteps[fracidx]>0:
                    img2 = img2.blur_gauss(beamparams, fracsteps[fracidx])

                #print(j, i, fracidx)

                # compute image comparision under a specified blur_frac
                (error, im1_pad, im2_shift) = img1.compare_images(img2, metric = [metric], psize = min_psize, target_fov = max_fov, blur_frac=0.0, beamparams=beamparams)

                # if specified save the shifted images used for comparision
                if savepath:
                    im1_pad.save_fits(savepath + '/' + str(i) + '_' + str(fracidx) + '.fits')
                    im2_shift.save_fits(savepath + '/' + str(j) +  '_' + str(fracidx) + '.fits')

                # save the metric value in a matrix
                metric_mtx[i,j,fracidx] = error[0]

    return (metric_mtx, fracsteps)



# look over an array of images and determine the min pixel size and max fov that can be used consistently across them
def get_psize_fov(imarr):
    min_psize = 100
    for i in range(0, len(imarr)):
        if i==0:
            max_fov = np.max([imarr[i].psize*imarr[i].xdim, imarr[i].psize*imarr[i].ydim])
            min_psize = imarr[i].psize
        else:
            max_fov = np.max([max_fov, imarr[i].psize*imarr[i].xdim, imarr[i].psize*imarr[i].ydim])
            min_psize = np.min([min_psize, imarr[i].psize])
    return (min_psize, max_fov)



def image_agreements(imarr, beamparams, metric_mtx, fracsteps, cutoff=0.95):

    (min_psize, max_fov) = get_psize_fov(imarr)

    im_cliques_fraclevels = []
    cliques_fraclevels = []
    for fracidx in range(len(fracsteps)):
        #print(fracidx)

        slice_metric_mtx = metric_mtx[:,:,fracidx]
        cuttoffidx = np.where( slice_metric_mtx >= cutoff)
        consistant = zip(*cuttoffidx)

        # make graph
        G=nx.Graph()
        for i in range(len(consistant)):
            G.add_edge(consistant[i][0], consistant[i][1])

        # find all cliques
        cliques = list(nx.find_cliques(G))
        #print(cliques)

        cliques_fraclevels.append(cliques)

        im_clique = []
        for c in range(len(cliques)):
            clique = cliques[c]
            im_avg = imarr[clique[0]].blur_gauss(beamparams,fracsteps[fracidx])

            for n in range(1,len(clique)):
                (error, im_avg, im2_shift) = im_avg.compare_images(imarr[clique[n]].blur_gauss(beamparams,fracsteps[fracidx]) , metric = ['xcorr'], psize = min_psize, target_fov = max_fov, blur_frac=0.0,
                         beamparams=beamparams)
                im_avg.imvec = (im_avg.imvec + im2_shift.imvec ) / 2.0


            im_clique.append(im_avg.copy())

        im_cliques_fraclevels.append(im_clique)

    return(cliques_fraclevels, im_cliques_fraclevels)


def change_cut_off(metric_mtx, fracsteps, imarr, beamparams, cutoff=0.95, zoom=0.1, fov=1):
    (cliques_fraclevels, im_cliques_fraclevels) = image_agreements(imarr, beamparams, metric_mtx, fracsteps, cutoff=cutoff)
    generate_consistency_plot(cliques_fraclevels, im_cliques_fraclevels, metric_mtx=metric_mtx, fracsteps=fracsteps, beamparams=beamparams, zoom=zoom, fov=fov)


def generate_consistency_plot(clique_fraclevels, im_clique_fraclevels, zoom=0.1, fov=1, show=True, framesize=(20,10), fracsteps=None, cutoff=None, r_offset = 1):

    fig, ax = plt.subplots(figsize=framesize)
    cycol = cycle('bgrcmk')

    x_loc = []

    for c, column in enumerate(clique_fraclevels):
        colorc = cycol.next()
        x_loc.append(((20./len(clique_fraclevels))*c))
        if len(column) == 0:
            continue

        for r, row in enumerate(column):

            # adding the images
            lenx = len(clique_fraclevels)
            leny = 0
            for li in clique_fraclevels:
                if len(li) > leny:
                    leny = len(li)
            sample_image = im_clique_fraclevels[c][r].regrid_image(fov*im_clique_fraclevels[c][r].fovx(), 512)
            arr_img = sample_image.imvec.reshape(sample_image.xdim, sample_image.ydim)
            imagebox = OffsetImage(arr_img, zoom=zoom, cmap='afmhot')

            imagebox.image.axes = ax


            ab = AnnotationBbox(imagebox, ((20./lenx)*c+r_offset,(20./leny)*r),
                                xycoords='data',
                                pad=0.0,
                                arrowprops=None)

            ax.add_artist(ab)

            # adding the arrows
            if c+1 != len(clique_fraclevels):
                for a, ro in enumerate(clique_fraclevels[c+1]):
                    if set(row).issubset(ro):
                        px = c+1
                        px = ((20./lenx)*px) + r_offset
                        py = a
                        py = (20./leny)*py
                        break

                xx = (20./lenx)*c + (8./lenx) + r_offset
                yy = (20./leny)*r
                ax.arrow(   xx, yy,
                            px - xx - (9./lenx), py- yy,
                            head_width=0.05,
                            head_length=0.1,
                            color=colorc
                        )
            row.sort()
            # adding the text
            txtstring = str(row)
            # print(ab.get_window_extent())
            # if len(row) == len(clique_fraclevels[-1][0]):
            #     txtstring = '[all]'

            # ax.text((20./lenx)*c - (0./lenx), (20./leny)*r  - (10./leny), txtstring, fontsize=6, horizontalalignment='center')
            ax.text((20./lenx)*c,(20./leny)*(r-0.5), txtstring, fontsize=10, horizontalalignment='center', color='black', zorder=1000)

    ax.set_xlim(0, 22)
    ax.set_ylim(-10, 22)

    ax.set_xticks(x_loc)
    ax.set_xticklabels(fracsteps)


    ax.set_yticks([])
    ax.set_yticklabels([])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_title('Blurred comparison of all images; cutoff={0}, fov (uas)={1}'.format(str(cutoff), str(im_clique_fraclevels[-1][-1].fovx()/eh.RADPERUAS)))

#     for item in [fig, ax]:
#         item.patch.set_visible(False)
#     fig.patch.set_visible(False)
#     ax.axis('off')
    if show == True:
        plt.show()
