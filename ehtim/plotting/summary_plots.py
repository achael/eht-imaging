# summary_plots.py
# Make data plots with multiple observations,images etc.
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

# TODO add systematic noise to individual closure quantities?

from __future__ import division
from __future__ import print_function

from builtins import str
from builtins import range
from builtins import object

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import datetime

from ehtim.plotting.comp_plots import plotall_obs_compare
from ehtim.plotting.comp_plots import plot_bl_obs_compare
from ehtim.plotting.comp_plots import plot_cphase_obs_compare
from ehtim.plotting.comp_plots import plot_camp_obs_compare
from ehtim.calibrating.self_cal import self_cal as selfcal
from ehtim.calibrating.pol_cal import leakage_cal, plot_leakage
import ehtim.const_def as ehc

FONTSIZE = 22
WSPACE = 0.8
HSPACE = 0.3
MARGINS = 0.5
PROCESSES = 4
MARKERSIZE = 5


def imgsum(im_or_mov, obs, obs_uncal, outname, outdir='.', title='imgsum', commentstr="",
           fontsize=FONTSIZE, cfun='afmhot', snrcut=0., maxset=False, ttype='nfft',
           gainplots=True, ampplots=True, cphaseplots=True, campplots=True, ebar=True,
           debias=True, cp_uv_min=False, force_extrapolate=True, processes=PROCESSES,
           sysnoise=0, syscnoise=0):
    """Produce an image summary plot for an image and uvfits file.

       Args:
           im_or_mov (Image or Movie): an Image object or Movie
           obs (Obsdata): the self-calibrated Obsdata object
           obs_uncal (Obsdata): the original Obsdata object
           outname (str): output pdf file name

           outdir (str): directory for output file
           title (str): the pdf file title
           commentstr (str): a comment for the top line of the pdf
           fontsize (float): the font size for text in the sheet
           cfun (float): matplotlib color function

           maxset (bool): True to use a maximal set of closure quantities

           gainplots (bool): include gain plots or not
           ampplots (bool): include amplitude consistency plots or not
           cphaseplots (bool): include closure phase consistency plots or not
           campplots (bool): include closure amplitude consistency plots or not
           ebar (bool): include error bars or not
           debias (bool): debias visibility amplitudes before computing chisq or not
           cp_uv_min (bool): minimum uv-distance cutoff for including a baseline in closure phase

           sysnoise (float): percent systematic noise added in quadrature
           syscnoise (float): closure phase systematic noise in degrees added in quadrature

           snrcut (dict): a dictionary of snrcut values for each quantity

           ttype (str): "fast" or "nfft" or "direct"
           force_extrapolate (bool): if True, always extrapolate movie start/stop frames
           processes (int): number of cores to use in multiprocessing
       Returns:

    """

    plt.close('all')  # close conflicting plots
    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)
    plt.rc('font', size=FONTSIZE)
    plt.rc('axes', titlesize=FONTSIZE)
    plt.rc('axes', labelsize=FONTSIZE)
    plt.rc('xtick', labelsize=FONTSIZE)
    plt.rc('ytick', labelsize=FONTSIZE)
    plt.rc('legend', fontsize=FONTSIZE)
    plt.rc('figure', titlesize=FONTSIZE)

    if fontsize == 0:
        fontsize = FONTSIZE

    if maxset:
        count = 'max'
    else:
        count = 'min'

    snrcut_dict = {key: 0. for key in ['vis', 'amp', 'cphase', 'logcamp', 'camp']}

    if type(snrcut) is dict:
        for key in snrcut.keys():
            snrcut_dict[key] = snrcut[key]
    else:
        for key in snrcut_dict.keys():
            snrcut_dict[key] = snrcut

    with PdfPages(outname) as pdf:
        titlestr = 'Summary Sheet for %s on MJD %s' % (im_or_mov.source, im_or_mov.mjd)

        # pdf metadata
        d = pdf.infodict()
        d['Title'] = title
        d['Author'] = u'EHT Team 1'
        d['Subject'] = titlestr
        d['CreationDate'] = datetime.datetime.today()
        d['ModDate'] = datetime.datetime.today()

        # define the figure
        fig = plt.figure(1, figsize=(18, 28), dpi=200)
        gs = gridspec.GridSpec(6, 4, wspace=WSPACE, hspace=HSPACE)

        # user comments
        if len(commentstr) > 1:
            titlestr = titlestr+'\n'+str(commentstr)

        plt.suptitle(titlestr, y=.9, va='center', fontsize=int(1.2*fontsize))

        ################################################################################
        print("===========================================")
        print("displaying the image")
        ax = plt.subplot(gs[0:2, 0:2])
        ax.set_title('Submitted Image')

        movie = hasattr(im_or_mov, 'get_image')
        if movie:
            im_display = im_or_mov.avg_frame()

            # TODO --- ok to always extrapolate?
            if force_extrapolate:
                im_or_mov.reset_interp(bounds_error=False)
        elif hasattr(im_or_mov, 'make_image'):      
            im_display = im_or_mov.make_image(obs.res() * 10., 512)
        else:
            im_display = im_or_mov.copy()

        ax = _display_img(im_display, axis=ax, show=False,
                          has_title=False, cfun=cfun, fontsize=fontsize)

        print("===========================================")
        print("displaying the blurred image")
        ax = plt.subplot(gs[0:2, 2:5])
        ax.set_title('Image blurred to nominal resolution')
        fwhm = obs.res()
        print("blur_FWHM: ", fwhm/ehc.RADPERUAS)
        beamparams = [fwhm, fwhm, 0]

        imblur = im_display.blur_gauss(beamparams, frac=1.0)
        ax = _display_img(imblur, beamparams=beamparams, axis=ax, show=False,
                          has_title=False, cfun=cfun, fontsize=fontsize)

        ################################################################################
        print("===========================================")
        print("calculating statistics")
        # display the overall chi2
        ax = plt.subplot(gs[2, 0:2])
        ax.set_title('Image statistics')
        # ax.axis('off')
        ax.set_yticks([])
        ax.set_xticks([])

        flux = im_display.total_flux()

        # SNR ordering
        # obs.reorder_tarr_snr()
        # obs_uncal.reorder_tarr_snr()

        maxset = False
        # compute chi^2
        chi2vis = obs.chisq(im_or_mov, dtype='vis', ttype=ttype,
                            systematic_noise=sysnoise, maxset=maxset, snrcut=snrcut_dict['vis'])
        chi2amp = obs.chisq(im_or_mov, dtype='amp', ttype=ttype,
                            systematic_noise=sysnoise, maxset=maxset, snrcut=snrcut_dict['amp'])
        chi2cphase = obs.chisq(im_or_mov, dtype='cphase', ttype=ttype, systematic_noise=sysnoise,
                               systematic_cphase_noise=syscnoise,
                               maxset=maxset, cp_uv_min=cp_uv_min, snrcut=snrcut_dict['cphase'])
        chi2logcamp = obs.chisq(im_or_mov, dtype='logcamp', ttype=ttype, systematic_noise=sysnoise,
                                maxset=maxset, snrcut=snrcut_dict['logcamp'])
        chi2camp = obs.chisq(im_or_mov, dtype='camp', ttype=ttype,
                             systematic_noise=sysnoise, maxset=maxset, snrcut=snrcut_dict['camp'])

        chi2vis_uncal = obs_uncal.chisq(im_or_mov, dtype='vis', ttype=ttype, systematic_noise=0,
                                        maxset=maxset, snrcut=snrcut_dict['vis'])
        chi2amp_uncal = obs_uncal.chisq(im_or_mov, dtype='amp', ttype=ttype, systematic_noise=0,
                                        maxset=maxset, snrcut=snrcut_dict['amp'])
        chi2cphase_uncal = obs_uncal.chisq(im_or_mov, dtype='cphase', ttype=ttype,
                                           systematic_noise=0,
                                           systematic_cphase_noise=0, maxset=maxset,
                                           cp_uv_min=cp_uv_min, snrcut=snrcut_dict['cphase'])
        chi2logcamp_uncal = obs_uncal.chisq(im_or_mov, dtype='logcamp', ttype=ttype,
                                            systematic_noise=0, maxset=maxset,
                                            snrcut=snrcut_dict['logcamp'])
        chi2camp_uncal = obs_uncal.chisq(im_or_mov, dtype='camp', ttype=ttype, systematic_noise=0,
                                         maxset=maxset, snrcut=snrcut_dict['camp'])

        print("chi^2 vis: %0.2f %0.2f" % (chi2vis, chi2vis_uncal))
        print("chi^2 amp: %0.2f %0.2f" % (chi2amp, chi2amp_uncal))
        print("chi^2 cphase: %0.2f %0.2f" % (chi2cphase, chi2cphase_uncal))
        print("chi^2 logcamp: %0.2f %0.2f" % (chi2logcamp, chi2logcamp_uncal))
        print("chi^2 camp: %0.2f %0.2f" % (chi2logcamp, chi2logcamp_uncal))

        fs = int(1*fontsize)
        fs2 = int(.8*fontsize)
        ax.text(.05, .9, "Source:", fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.05, .7, "MJD:", fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.05, .5, "FREQ:", fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.05, .3, "FOV:", fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.05, .1, "FLUX:", fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)

        ax.text(.23, .9, "%s" % im_or_mov.source, fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.23, .7, "%i" % im_or_mov.mjd, fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.23, .5, "%0.0f GHz" % (im_or_mov.rf/1.e9), fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.23, .3, "%0.1f $\mu$as" % (im_display.fovx()/ehc.RADPERUAS), fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.23, .1, "%0.2f Jy" % flux, fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)

        ax.text(.5, .9, "$\chi^2_{vis}$", fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.5, .7, "$\chi^2_{amp}$", fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.5, .5, "$\chi^2_{cphase}$", fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.5, .3, "$\chi^2_{log camp}$", fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.5, .1, "$\chi^2_{camp}$", fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)

        ax.text(.72, .9, "%0.2f" % chi2vis, fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.72, .7, "%0.2f" % chi2amp, fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.72, .5, "%0.2f" % chi2cphase, fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.72, .3, "%0.2f" % chi2logcamp, fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.72, .1, "%0.2f" % chi2camp, fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)

        ax.text(.85, .9, "(%0.2f)" % chi2vis_uncal, fontsize=fs2,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.85, .7, "(%0.2f)" % chi2amp_uncal, fontsize=fs2,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.85, .5, "(%0.2f)" % chi2cphase_uncal, fontsize=fs2,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.85, .3, "(%0.2f)" % chi2logcamp, fontsize=fs2,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.85, .1, "(%0.2f)" % chi2camp_uncal, fontsize=fs2,
                ha='left', va='center', transform=ax.transAxes)

        ################################################################################
        print("===========================================")
        print("calculating cphase statistics")
        # display the closure  phase chi2
        ax = plt.subplot(gs[3:6, 0:2])
        ax.set_title('Closure phase statistics')
        ax.set_yticks([])
        ax.set_xticks([])

        # get closure triangle combinations
        # ANDREW -- hacky, fix this!
        cp = obs.c_phases(mode="all", count=count, uv_min=cp_uv_min, snrcut=snrcut_dict['cphase'])
        n_cphase = len(cp)
        alltris = [(str(cpp['t1']), str(cpp['t2']), str(cpp['t3'])) for cpp in cp]
        uniqueclosure_tri = []
        for tri in alltris:
            if tri not in uniqueclosure_tri:
                uniqueclosure_tri.append(tri)

        # generate data
        obs_model = im_or_mov.observe_same(obs, add_th_noise=False, ttype=ttype)

        # TODO: check SNR cut
        cphases_obs = obs.c_phases(mode='all', count='max', vtype='vis',
                                   uv_min=cp_uv_min, snrcut=snrcut_dict['cphase'])
        if snrcut_dict['cphase'] > 0:
            cphases_obs_all = obs.c_phases(mode='all', count='max',
                                           vtype='vis', uv_min=cp_uv_min, snrcut=0.)
            cphases_model_all = obs_model.c_phases(
                mode='all', count='max', vtype='vis', uv_min=cp_uv_min, snrcut=0.)
            mask = [cphase in cphases_obs for cphase in cphases_obs_all]
            cphases_model = cphases_model_all[mask]
            print('cphase snr cut', snrcut_dict['cphase'], ' : kept', len(
                cphases_obs), '/', len(cphases_obs_all))
        else:
            cphases_model = obs_model.c_phases(
                mode='all', count='max', vtype='vis', uv_min=cp_uv_min, snrcut=0.)

        # generate chi^2 -- NO SYSTEMATIC NOISES
        cphase_chisq_data = []
        for c in range(0, len(uniqueclosure_tri)):
            cphases_obs_tri = obs.cphase_tri(uniqueclosure_tri[c][0],
                                             uniqueclosure_tri[c][1],
                                             uniqueclosure_tri[c][2],
                                             vtype='vis', ang_unit='deg', cphases=cphases_obs)

            if len(cphases_obs_tri) > 0:
                cphases_model_tri = obs_model.cphase_tri(uniqueclosure_tri[c][0],
                                                         uniqueclosure_tri[c][1],
                                                         uniqueclosure_tri[c][2],
                                                         vtype='vis', ang_unit='deg',
                                                         cphases=cphases_model)

                resids = (cphases_obs_tri['cphase'] - cphases_model_tri['cphase'])*ehc.DEGREE
                chisq_tri = 2*np.sum((1.0 - np.cos(resids)) /
                                     ((cphases_obs_tri['sigmacp']*ehc.DEGREE)**2))

                npts = len(cphases_obs_tri)
                data = [uniqueclosure_tri[c][0], uniqueclosure_tri[c]
                        [1], uniqueclosure_tri[c][2], npts, chisq_tri]
                cphase_chisq_data.append(data)

        # sort by decreasing chi^2
        idx = np.argsort([data[-1] for data in cphase_chisq_data])
        idx = list(reversed(idx))

        chisqtab = (r"\begin{tabular}{ l|l|l|l } \hline Triangle " +
                    r"& $N_{tri}$ & $\chi^2_{tri}/N_{tri}$ & $\chi^2_{tri}/N_{tot}$" +
                    r"\\ \hline \hline")
        first = True
        for i in range(len(cphase_chisq_data)):
            if i > 30:
                break
            data = cphase_chisq_data[idx[i]]
            tristr = r"%s-%s-%s" % (data[0], data[1], data[2])
            nstr = r"%i" % data[3]
            rchisqstr = r"%0.1f" % (float(data[4])/float(data[3]))
            rrchisqstr = r"%0.3f" % (float(data[4])/float(n_cphase))
            if first:
                chisqtab += r" " + tristr + " & " + nstr + " & " + rchisqstr + " & " + rrchisqstr
                first = False
            else:
                chisqtab += r" \\" + tristr + " & " + nstr + " & " + rchisqstr + " & " + rrchisqstr
        chisqtab += r" \end{tabular}"

        ax.text(0.5, .975, chisqtab, ha="center", va="top", transform=ax.transAxes, size=fontsize)

        ################################################################################
        print("===========================================")
        print("calculating camp statistics")
        # display the log closure amplitude chi2
        ax = plt.subplot(gs[2:6, 2::])
        ax.set_title('Log Closure amplitude statistics')
        # ax.axis('off')
        ax.set_yticks([])
        ax.set_xticks([])

        # get closure amplitude combinations
        # TODO -- hacky, fix this!
        cp = obs.c_amplitudes(mode="all", count=count, ctype='logcamp', debias=debias)
        n_camps = len(cp)
        allquads = [(str(cpp['t1']), str(cpp['t2']), str(cpp['t3']), str(cpp['t4'])) for cpp in cp]
        uniqueclosure_quad = []
        for quad in allquads:
            if quad not in uniqueclosure_quad:
                uniqueclosure_quad.append(quad)

        # generate data
        # TODO: check SNR cut
        camps_obs = obs.c_amplitudes(mode='all', count='max', ctype='logcamp',
                                     debias=debias, snrcut=snrcut_dict['logcamp'])
        if snrcut_dict['logcamp'] > 0:
            camps_obs_all = obs.c_amplitudes(
                mode='all', count='max', ctype='logcamp', debias=debias, snrcut=0.)
            camps_model_all = obs_model.c_amplitudes(
                mode='all', count='max', ctype='logcamp', debias=False, snrcut=0.)
            mask = [camp['camp'] in camps_obs['camp'] for camp in camps_obs_all]
            camps_model = camps_model_all[mask]
            print('closure amp snrcut', snrcut_dict['logcamp'],
                  ': kept', len(camps_obs), '/', len(camps_obs_all))
        else:
            camps_model = obs_model.c_amplitudes(
                mode='all', count='max', ctype='logcamp', debias=False, snrcut=0.)

        # generate chi2 -- NO SYSTEMATIC NOISES
        camp_chisq_data = []
        for c in range(0, len(uniqueclosure_quad)):
            camps_obs_quad = obs.camp_quad(uniqueclosure_quad[c][0], uniqueclosure_quad[c][1],
                                           uniqueclosure_quad[c][2],  uniqueclosure_quad[c][3],
                                           vtype='vis', camps=camps_obs, ctype='logcamp')

            if len(camps_obs_quad) > 0:
                camps_model_quad = obs.camp_quad(uniqueclosure_quad[c][0], uniqueclosure_quad[c][1],
                                                 uniqueclosure_quad[c][2], uniqueclosure_quad[c][3],
                                                 vtype='vis', camps=camps_model, ctype='logcamp')

                resids = camps_obs_quad['camp'] - camps_model_quad['camp']
                chisq_quad = np.sum(np.abs(resids/camps_obs_quad['sigmaca'])**2)
                npts = len(camps_obs_quad)

                data = (uniqueclosure_quad[c][0], uniqueclosure_quad[c][1],
                        uniqueclosure_quad[c][2], uniqueclosure_quad[c][3],
                        npts,
                        chisq_quad)
                camp_chisq_data.append(data)

        # sort by decreasing chi^2
        idx = np.argsort([data[-1] for data in camp_chisq_data])
        idx = list(reversed(idx))

        chisqtab = (r"\begin{tabular}{ l|l|l|l } \hline Quadrangle " +
                    r"& $N_{quad}$ & $\chi^2_{quad}/N_{quad}$ & $\chi^2_{quad}/N_{tot}$ " +
                    r"\\ \hline \hline")
        for i in range(len(camp_chisq_data)):
            if i > 45:
                break
            data = camp_chisq_data[idx[i]]
            tristr = r"%s-%s-%s-%s" % (data[0], data[1], data[2], data[3])
            nstr = r"%i" % data[4]
            rchisqstr = r"%0.1f" % (data[5]/float(data[4]))
            rrchisqstr = r"%0.3f" % (data[5]/float(n_camps))
            if i == 0:
                chisqtab += r" " + tristr + " & " + nstr + " & " + rchisqstr + " & " + rrchisqstr
            else:
                chisqtab += r" \\" + tristr + " & " + nstr + " & " + rchisqstr + " & " + rrchisqstr

        chisqtab += r" \end{tabular}"

        ax.text(0.5, .975, chisqtab, ha="center", va="top", transform=ax.transAxes, size=fontsize)

        # save the first page of the plot
        print('saving pdf page 1')
        pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
        plt.close()

        ################################################################################
        # plot the vis amps
        fig = plt.figure(2, figsize=(18, 28), dpi=200)
        gs = gridspec.GridSpec(6, 4, wspace=WSPACE, hspace=HSPACE)

        print("===========================================")
        print("plotting vis amps")
        ax = plt.subplot(gs[0:2, 0:2])
        obs_tmp = obs_model.copy()
        obs_tmp.data['sigma'] *= 0.
        ax = plotall_obs_compare([obs, obs_tmp],
                                 'uvdist', 'amp', axis=ax, legend=False,
                                 clist=['k', ehc.SCOLORS[1]],
                                 ttype=ttype, show=False, debias=debias,
                                 snrcut=snrcut_dict['amp'],
                                 ebar=ebar, markersize=MARKERSIZE)
        # modify the labels
        ax.set_title('Calibrated Visiblity Amplitudes')
        ax.set_xlabel('u-v distance (G$\lambda$)')
        ax.set_xlim([0, 1.e10])
        ax.set_xticks([0, 2.e9, 4.e9, 6.e9, 8.e9, 10.e9])
        ax.set_xticklabels(["0", "2", "4", "6", "8", "10"])
        ax.set_xticks([1.e9, 3.e9, 5.e9, 7.e9, 9.e9], minor=True)
        ax.set_xticklabels([], minor=True)

        ax.set_ylabel('Amplitude (Jy)')
        ax.set_ylim([0, 1.2*flux])
        yticks_maj = np.array([0, .2, .4, .6, .8, 1])*flux
        ax.set_yticks(yticks_maj)
        ax.set_yticklabels(["%0.2f" % fl for fl in yticks_maj])
        yticks_min = np.array([.1, .3, .5, .7, .9])*flux
        ax.set_yticks(yticks_min, minor=True)
        ax.set_yticklabels([], minor=True)

        # plot the caltable gains
        if gainplots:
            print("===========================================")
            print("plotting gains")
            ax2 = plt.subplot(gs[0:2, 2:6])
            obs_tmp = obs_uncal.copy()
            for i in range(1):
                ct = selfcal(obs_tmp, im_or_mov,
                             method='amp', ttype=ttype,
                             caltable=True, gain_tol=.2,
                             processes=processes)
                ct = ct.pad_scans()
                obs_tmp = ct.applycal(obs_tmp, interp='nearest', extrapolate=True)
                if np.any(np.isnan(obs_tmp.data['vis'])):
                    print("Warning: NaN in applycal vis table!")
                    break
                if i > 0:
                    ct_out = ct_out.merge([ct])
                else:
                    ct_out = ct

            ax2 = ct_out.plot_gains('all', rangey=[.1, 10],
                                    yscale='log', axis=ax2, legend=True, show=False)

            # median gains
            ax = plt.subplot(gs[3:6, 2:5])
            ax.set_title('Station gain statistics')
            ax.set_yticks([])
            ax.set_xticks([])

            gain_data = []
            for station in ct_out.tarr['site']:
                try:
                    gain = np.median(np.abs(ct_out.data[station]['lscale']))
                except:
                    continue
                pdiff = np.abs(gain-1)*100
                data = (station, gain, pdiff)
                gain_data.append(data)

            # sort by decreasing chi^2
            idx = np.argsort([data[-1] for data in gain_data])
            idx = list(reversed(idx))

            chisqtab = (r"\begin{tabular}{ l|l|l } \hline Site & " +
                        r"Median Gain & Percent diff. \\ \hline \hline")
            for i in range(len(gain_data)):
                if i > 45:
                    break
                data = gain_data[idx[i]]
                sitestr = r"%s" % (data[0])
                gstr = r"%0.2f" % data[1]
                pstr = r"%0.0f" % data[2]
                if i == 0:
                    chisqtab += r" " + sitestr + " & " + gstr + " & " + pstr
                else:
                    chisqtab += r" \\" + sitestr + " & " + gstr + " & " + pstr

            chisqtab += r" \end{tabular}"
            ax.text(0.5, .975, chisqtab, ha="center", va="top",
                    transform=ax.transAxes, size=fontsize)

        # baseline amplitude chi2
        print("===========================================")
        print("baseline vis amps chisq")
        ax = plt.subplot(gs[3:6, 0:2])
        ax.set_title('Visibility amplitude statistics')
        ax.set_yticks([])
        ax.set_xticks([])

        bl_unpk = obs.unpack(['t1', 't2'], debias=debias)
        n_bl = len(bl_unpk)
        allbl = [(str(bl['t1']), str(bl['t2'])) for bl in bl_unpk]
        uniquebl = []
        for bl in allbl:
            if bl not in uniquebl:
                uniquebl.append(bl)

        # generate chi2 -- NO SYSTEMATIC NOISES
        bl_chisq_data = []
        for ii in range(0, len(uniquebl)):
            bl = uniquebl[ii]

            amps_bl = obs.unpack_bl(bl[0], bl[1], ['amp', 'sigma'], debias=debias)
            if len(amps_bl) > 0:

                amps_bl_model = obs_model.unpack_bl(bl[0], bl[1], ['amp', 'sigma'], debias=False)

                if snrcut_dict['amp'] > 0:
                    amask = amps_bl['amp']/amps_bl['sigma'] > snrcut_dict['amp']
                    amps_bl = amps_bl[amask]
                    amps_bl_model = amps_bl_model[amask]

                chisq_bl = np.sum(
                    np.abs((amps_bl['amp'] - amps_bl_model['amp'])/amps_bl['sigma'])**2)
                npts = len(amps_bl_model)

                data = (bl[0], bl[1],
                        npts,
                        chisq_bl)
                bl_chisq_data.append(data)

        # sort by decreasing chi^2
        idx = np.argsort([data[-1] for data in bl_chisq_data])
        idx = list(reversed(idx))

        chisqtab = (r"\begin{tabular}{ l|l|l|l } \hline Baseline & " +
                    r"$N_{amp}$ & $\chi^2_{amp}/N_{amp}$ & $\chi^2_{amp}/N_{total}$ " +
                    r"\\ \hline \hline")
        for i in range(len(bl_chisq_data)):
            if i > 45:
                break
            data = bl_chisq_data[idx[i]]
            tristr = r"%s-%s" % (data[0], data[1])
            nstr = r"%i" % data[2]
            rchisqstr = r"%0.1f" % (data[3]/float(data[2]))
            rrchisqstr = r"%0.3f" % (data[3]/float(n_bl))
            if i == 0:
                chisqtab += r" " + tristr + " & " + nstr + " & " + rchisqstr + " & " + rrchisqstr
            else:
                chisqtab += r" \\" + tristr + " & " + nstr + " & " + rchisqstr + " & " + rrchisqstr

        chisqtab += r" \end{tabular}"

        ax.text(0.5, .975, chisqtab, ha="center", va="top", transform=ax.transAxes, size=fontsize)

        # save the first page of the plot
        print('saving pdf page 2')
        # plt.tight_layout()
        # plt.subplots_adjust(wspace=1,hspace=1)
        # plt.savefig(outname, pad_inches=MARGINS,bbox_inches='tight')
        pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
        plt.close()

        ################################################################################
        # plot the visibility amplitudes
        page = 3
        if ampplots:
            print("===========================================")
            print("plotting amplitudes")
            fig = plt.figure(3, figsize=(18, 28), dpi=200)
            plt.suptitle("Amplitude Plots", y=.9, va='center', fontsize=int(1.2*fontsize))
            gs = gridspec.GridSpec(6, 4, wspace=WSPACE, hspace=HSPACE)
            i = 0
            j = 0
            switch = False

            obs_model.data['sigma'] *= 0
            amax = 1.1*np.max(np.abs(np.abs(obs_model.data['vis'])))
            obs_all = [obs, obs_model]
            for bl in uniquebl:
                ax = plt.subplot(gs[2*i:2*(i+1), 2*j:2*(j+1)])
                ax = plot_bl_obs_compare(obs_all, bl[0], bl[1], 'amp', rangey=[0, amax],
                                         markersize=MARKERSIZE, debias=debias,
                                         snrcut=snrcut_dict['amp'],
                                         axis=ax, legend=False, clist=['k', ehc.SCOLORS[1]],
                                         ttype=ttype, show=False, ebar=ebar)
                if ax is None:
                    continue
                if switch:
                    i += 1
                    j = 0
                    switch = False
                else:
                    j = 1
                    switch = True

                ax.set_xlabel('')

                if i == 3:
                    print('saving pdf page %i' % page)
                    page += 1
                    pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
                    plt.close()
                    fig = plt.figure(3, figsize=(18, 28), dpi=200)
                    gs = gridspec.GridSpec(6, 4, wspace=WSPACE, hspace=HSPACE)
                    i = 0
                    j = 0
                    switch = False

            print('saving pdf page %i' % page)
            page += 1
            pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
            plt.close()

        ################################################################################
        # plot the closure phases
        if cphaseplots:
            print("===========================================")
            print("plotting closure phases")
            fig = plt.figure(3, figsize=(18, 28), dpi=200)
            plt.suptitle("Closure Phase Plots", y=.9, va='center', fontsize=int(1.2*fontsize))
            gs = gridspec.GridSpec(6, 4, wspace=WSPACE, hspace=HSPACE)
            i = 0
            j = 0
            switch = False
            obs_all = [obs, obs_model]
            cphases_model['sigmacp'] *= 0
            cphases_all = [cphases_obs, cphases_model]
            for tri in uniqueclosure_tri:

                ax = plt.subplot(gs[2*i:2*(i+1), 2*j:2*(j+1)])
                ax = plot_cphase_obs_compare(obs_all, tri[0], tri[1], tri[2], rangey=[-185, 185],
                                             cphases=cphases_all, markersize=MARKERSIZE,
                                             axis=ax, legend=False, clist=['k', ehc.SCOLORS[1]],
                                             ttype=ttype, show=False, ebar=ebar)
                if ax is None:
                    continue
                if switch:
                    i += 1
                    j = 0
                    switch = False
                else:
                    j = 1
                    switch = True

                ax.set_xlabel('')

                if i == 3:
                    print('saving pdf page %i' % page)
                    page += 1
                    pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
                    plt.close()
                    fig = plt.figure(3, figsize=(18, 28), dpi=200)
                    gs = gridspec.GridSpec(6, 4, wspace=WSPACE, hspace=HSPACE)
                    i = 0
                    j = 0
                    switch = False
            print('saving pdf page %i' % page)
            page += 1
            pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
            plt.close()

        ################################################################################
        # plot the log closure amps
        if campplots:
            print("===========================================")
            print("plotting closure amplitudes")
            fig = plt.figure(3, figsize=(18, 28), dpi=200)
            plt.suptitle("Closure Amplitude Plots", y=.9, va='center', fontsize=int(1.2*fontsize))
            gs = gridspec.GridSpec(6, 4, wspace=WSPACE, hspace=HSPACE)
            i = 0
            j = 0
            switch = False
            obs_all = [obs, obs_model]
            camps_model['sigmaca'] *= 0
            camps_all = [camps_obs, camps_model]
            cmax = 1.1*np.max(np.abs(camps_obs['camp']))
            for quad in uniqueclosure_quad:
                ax = plt.subplot(gs[2*i:2*(i+1), 2*j:2*(j+1)])
                ax = plot_camp_obs_compare(obs_all, quad[0], quad[1], quad[2], quad[3],
                                           markersize=MARKERSIZE,
                                           ctype='logcamp', rangey=[-cmax, cmax], camps=camps_all,
                                           axis=ax, legend=False, clist=['k', ehc.SCOLORS[1]],
                                           ttype=ttype, show=False, ebar=ebar)
                if ax is None:
                    continue
                if switch:
                    i += 1
                    j = 0
                    switch = False
                else:
                    j = 1
                    switch = True

                ax.set_xlabel('')

                if i == 3:
                    print('saving pdf page %i' % page)
                    page += 1
                    pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
                    plt.close()
                    fig = plt.figure(3, figsize=(18, 28), dpi=200)
                    gs = gridspec.GridSpec(6, 4, wspace=WSPACE, hspace=HSPACE)
                    i = 0
                    j = 0
                    switch = False
            print('saving pdf page %i' % page)
            page += 1
            pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
            plt.close()


def imgsum_pol(im, obs, obs_uncal, outname,
               leakage_arr=False, nvec=False,
               outdir='.', title='imgsum_pol', commentstr="",
               fontsize=FONTSIZE, cfun='afmhot', snrcut=0.,
               dtermplots=True, pplots=True, mplots=True, qplots=True, uplots=True, ebar=True,
               sysnoise=0):
    """Produce a polarimetric image summary plot for an image and uvfits file.

       Args:
           im (Image): an Image object
           obs (Obsdata): the calibrated Obsdata object
           obs_uncal (Obsdata): the original Obsdata object
           outname (str): output pdf file name

           leakage_arr (bool): array with calibrated d-terms
           nvec (int): number of polarimetric vectors to plot in  each direction

           outdir (str): directory for output file
           title (str): the pdf file title
           commentstr (str): a comment for the top line of the pdf
           fontsize (float): the font size for text in the sheet
           cfun (float): matplotlib color function
           snrcut (dict): a dictionary of snrcut values for each quantity

           dtermplots (bool): plot the d-terms or not
           mplots (bool): plot the fractional polarizations or not
           pplots (bool): plot the P=RL polarization or not
           mplots (bool): plot the Q data or not
           pplots (bool): plot the U data or not

           ebar (bool): include error bars or not
           sysnoise (float): percent systematic noise added in quadrature

       Returns:

    """

    # switch polreps and mask nan data
    im = im.switch_polrep(polrep_out='stokes')
    obs = obs.switch_polrep(polrep_out='stokes')
    obs_uncal = obs_uncal.switch_polrep(polrep_out='stokes')

    mask_nan = (np.isnan(obs_uncal.data['vis']) +
                np.isnan(obs_uncal.data['qvis']) +
                np.isnan(obs_uncal.data['uvis']) +
                np.isnan(obs_uncal.data['vvis']))
    obs_uncal.data = obs_uncal.data[~mask_nan]

    mask_nan = (np.isnan(obs.data['vis']) +
                np.isnan(obs.data['qvis']) +
                np.isnan(obs.data['uvis']) +
                np.isnan(obs.data['vvis']))
    obs.data = obs.data[~mask_nan]

    if len(im.qvec) == 0 or len(im.uvec) == 0:
        raise Exception("the image isn't polarized!")

    plt.close('all')  # close conflicting plots
    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)
    plt.rc('font', size=FONTSIZE)
    plt.rc('axes', titlesize=FONTSIZE)
    plt.rc('axes', labelsize=FONTSIZE)
    plt.rc('xtick', labelsize=FONTSIZE)
    plt.rc('ytick', labelsize=FONTSIZE)
    plt.rc('legend', fontsize=FONTSIZE)
    plt.rc('figure', titlesize=FONTSIZE)

    if fontsize == 0:
        fontsize = FONTSIZE

    snrcut_dict = {key: 0. for key in ['m', 'pvis', 'qvis', 'uvis']}

    if type(snrcut) is dict:
        for key in snrcut.keys():
            snrcut_dict[key] = snrcut[key]
    else:
        for key in snrcut_dict.keys():
            snrcut_dict[key] = snrcut

    # TODO -- ok? prevent errors in divisition
    if(np.any(im.ivec == 0)):
        im.ivec += 1.e-50*np.max(im.ivec)

    with PdfPages(outname) as pdf:
        titlestr = 'Summary Sheet for %s on MJD %s' % (im.source, im.mjd)

        # pdf metadata
        d = pdf.infodict()
        d['Title'] = title
        d['Author'] = u'EHT Team 1'
        d['Subject'] = titlestr
        d['CreationDate'] = datetime.datetime.today()
        d['ModDate'] = datetime.datetime.today()

        # define the figure
        fig = plt.figure(1, figsize=(18, 28), dpi=200)
        gs = gridspec.GridSpec(6, 4, wspace=WSPACE, hspace=HSPACE)

        # user comments
        if len(commentstr) > 1:
            titlestr = titlestr+'\n'+str(commentstr)

        plt.suptitle(titlestr, y=.9, va='center', fontsize=int(1.2*fontsize))

        ################################################################################
        print("===========================================")
        print("displaying the images")

        # unblurred image IQU
        ax = plt.subplot(gs[0:2, 0:2])
        ax.set_title('I')

        ax = _display_img_pol(im, axis=ax, show=False, has_title=False, cfun=cfun,
                              pol='I', polticks=True,
                              nvec=nvec, pcut=0.1,  mcut=0.01, contour=False,
                              fontsize=fontsize)

        ax = plt.subplot(gs[2:4, 0:2])
        ax.set_title('Q')
        ax = _display_img_pol(im, axis=ax, show=False, has_title=False, cfun=plt.get_cmap('bwr'),
                              pol='Q', polticks=False,
                              nvec=nvec, pcut=0.1,  mcut=0.01, contour=True,
                              fontsize=fontsize)

        ax = plt.subplot(gs[4:6, 0:2])
        ax.set_title('U')
        ax = _display_img_pol(im, axis=ax, show=False, has_title=False, cfun=plt.get_cmap('bwr'),
                              pol='U', polticks=False,
                              nvec=nvec, pcut=0.1,  mcut=0.01, contour=True,
                              fontsize=fontsize)

        # blurred image IQU
        ax = plt.subplot(gs[0:2, 2:5])
        beamparams = obs_uncal.fit_beam()
        fwhm = np.min((np.abs(beamparams[0]), np.abs(beamparams[1])))
        print("blur_FWHM: ", fwhm/ehc.RADPERUAS)

        imblur = im.blur_gauss(beamparams, frac=1.0, frac_pol=1.)

        ax = _display_img_pol(imblur, axis=ax, show=False, has_title=False, cfun=cfun,
                              pol='I', polticks=True, beamparams=beamparams,
                              nvec=nvec, pcut=0.1,  mcut=0.01, contour=False,
                              fontsize=fontsize)

        ax = plt.subplot(gs[2:4, 2:5])
        ax = _display_img_pol(imblur, axis=ax, show=False, has_title=False,
                              cfun=plt.get_cmap('bwr'),
                              pol='Q', polticks=False,
                              nvec=nvec, pcut=0.1,  mcut=0.01, contour=True,
                              fontsize=fontsize)

        ax = plt.subplot(gs[4:6, 2:5])
        ax = _display_img_pol(imblur, axis=ax, show=False, has_title=False,
                              cfun=plt.get_cmap('bwr'),
                              pol='U', polticks=False,
                              nvec=nvec, pcut=0.1,  mcut=0.01, contour=True,
                              fontsize=fontsize)

        print('saving pdf page 1')
        pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
        plt.close()

        # unblurred image m chi
        fig = plt.figure(2, figsize=(18, 28), dpi=200)
        gs = gridspec.GridSpec(6, 4, wspace=WSPACE, hspace=HSPACE)

        ax = plt.subplot(gs[0:2, 0:2])
        ax.set_title('m')
        ax = _display_img_pol(im, axis=ax, show=True, has_title=False,
                              cfun=plt.get_cmap('jet'),
                              pol='m', polticks=False,
                              nvec=nvec, pcut=0.1,  mcut=0.01, contour=False,
                              fontsize=fontsize)

        ax = plt.subplot(gs[2:4, 0:2])
        ax.set_title('chi')
        ax = _display_img_pol(im, axis=ax, show=False, has_title=False,
                              cfun=plt.get_cmap('jet'),
                              pol='chi', polticks=False,
                              nvec=nvec, pcut=0.1,  mcut=0.01, contour=False,
                              fontsize=fontsize)

        ax = plt.subplot(gs[0:2, 2:5])
        ax = _display_img_pol(imblur, axis=ax, show=False, has_title=False,
                              cfun=plt.get_cmap('jet'),
                              pol='m', polticks=False,
                              nvec=nvec, pcut=0.1,  mcut=0.01, contour=False,
                              fontsize=fontsize)

        ax = plt.subplot(gs[2:4, 2:5])
        ax = _display_img_pol(imblur, axis=ax, show=False, has_title=False,
                              cfun=plt.get_cmap('jet'),
                              pol='chi', polticks=False,
                              nvec=nvec, pcut=0.1,  mcut=0.01, contour=False,
                              fontsize=fontsize)

        ################################################################################
        print("===========================================")
        print("calculating statistics")
        # display the overall chi2
        ax = plt.subplot(gs[4, 0:2])
        ax.set_title('Image statistics')
        # ax.axis('off')
        ax.set_yticks([])
        ax.set_xticks([])

        flux = im.total_flux()

        # SNR ordering
        # obs.reorder_tarr_snr()
        # obs_uncal.reorder_tarr_snr()

        # compute chi^2
        chi2pvis = obs.polchisq(im, dtype='m', ttype='nfft',
                                systematic_noise=sysnoise, pol_prim='qu')
        chi2m = obs.polchisq(im, dtype='m', ttype='nfft',
                             systematic_noise=sysnoise, pol_prim='qu')
        chi2qvis = obs.chisq(im, dtype='vis', ttype='nfft',
                             systematic_noise=sysnoise, pol='Q')
        chi2uvis = obs.chisq(im, dtype='vis', ttype='nfft',
                             systematic_noise=sysnoise, pol='U')

        chi2pvis_uncal = obs_uncal.polchisq(im, dtype='m', ttype='nfft',
                                            systematic_noise=sysnoise, pol_prim='qu')
        chi2m_uncal = obs_uncal.polchisq(im, dtype='m', ttype='nfft',
                                         systematic_noise=sysnoise, pol_prim='qu')
        chi2qvis_uncal = obs_uncal.chisq(im, dtype='vis', ttype='nfft',
                                         systematic_noise=sysnoise, pol='Q')
        chi2uvis_uncal = obs_uncal.chisq(im, dtype='vis', ttype='nfft',
                                         systematic_noise=sysnoise, pol='U')

        print("chi^2 m: %0.2f %0.2f" % (chi2m, chi2m_uncal))
        print("chi^2 pvis: %0.2f %0.2f" % (chi2pvis, chi2pvis_uncal))
        print("chi^2 qvis: %0.2f %0.2f" % (chi2qvis, chi2qvis_uncal))
        print("chi^2 uvis: %0.2f %0.2f" % (chi2uvis, chi2uvis_uncal))

        fs = int(1*fontsize)
        fs2 = int(.8*fontsize)
        ax.text(.05, .9, "Source:", fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.05, .7, "MJD:", fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.05, .5, "FREQ:", fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.05, .3, "FOV:", fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.05, .1, "FLUX:", fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)

        ax.text(.23, .9, "%s" % im.source, fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.23, .7, "%i" % im.mjd, fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.23, .5, "%0.0f GHz" % (im.rf/1.e9), fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.23, .3, "%0.1f $\mu$as" % (im.fovx()/ehc.RADPERUAS), fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.23, .1, "%0.2f Jy" % flux, fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)

        ax.text(.5, .9, "$\chi^2_{m}$", fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.5, .7, "$\chi^2_{P}$", fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.5, .5, "$\chi^2_{Q}$", fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.5, .3, "$\chi^2_{U}$", fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)

        ax.text(.72, .9, "%0.2f" % chi2m, fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.72, .7, "%0.2f" % chi2pvis, fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.72, .5, "%0.2f" % chi2qvis, fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.72, .3, "%0.2f" % chi2uvis, fontsize=fs,
                ha='left', va='center', transform=ax.transAxes)

        ax.text(.85, .9, "(%0.2f)" % chi2m_uncal, fontsize=fs2,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.85, .7, "(%0.2f)" % chi2pvis_uncal, fontsize=fs2,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.85, .5, "(%0.2f)" % chi2qvis_uncal, fontsize=fs2,
                ha='left', va='center', transform=ax.transAxes)
        ax.text(.85, .3, "(%0.2f)" % chi2uvis_uncal, fontsize=fs2,
                ha='left', va='center', transform=ax.transAxes)

        ################################################################################
        # plot the D terms

        if dtermplots:
            print("===========================================")
            print("plotting d terms")
            ax = plt.subplot(gs[4:6, 2:5])

            if leakage_arr:
                obs_polcal = obs_uncal.copy()
                obs_polcal.tarr = leakage_arr.tarr
            else:
                obs_polcal = leakage_cal(obs_uncal, im, leakage_tol=1e6, ttype='nfft')

            ax = plot_leakage(obs_polcal, axis=ax, show=False,
                              rangex=[-20, 20], rangey=[-20, 20], markersize=5)

            print('saving pdf page 2')
            pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
            plt.close()

        # 3
        # baseline amplitude chi2
        fig = plt.figure(2, figsize=(18, 28), dpi=200)
        gs = gridspec.GridSpec(6, 4, wspace=WSPACE, hspace=HSPACE)

        print("===========================================")
        print("baseline m&p  chisq")

        bl_unpk = obs.unpack(['t1', 't2'])
        n_bl = len(bl_unpk)
        allbl = [(str(bl['t1']), str(bl['t2'])) for bl in bl_unpk]
        uniquebl = []
        for bl in allbl:
            if bl not in uniquebl:
                uniquebl.append(bl)

        # generate data
        obs_model = im.observe_same(obs, add_th_noise=False, ttype='nfft')

        # generate chi2 -- NO SYSTEMATIC NOISES
        bl_chisq_data_m = []
        bl_chisq_data_pvis = []
        bl_chisq_data_qvis = []
        bl_chisq_data_uvis = []

        for ii in range(0, len(uniquebl)):
            bl = uniquebl[ii]

            m_bl = obs.unpack_bl(bl[0], bl[1], ['m', 'msigma'], debias=False)
            pvis_bl = obs.unpack_bl(bl[0], bl[1], ['pvis', 'psigma'], debias=False)
            qvis_bl = obs.unpack_bl(bl[0], bl[1], ['qvis', 'qsigma'], debias=False)
            uvis_bl = obs.unpack_bl(bl[0], bl[1], ['uvis', 'usigma'], debias=False)

            if len(m_bl) > 0:

                m_bl_model = obs_model.unpack_bl(bl[0], bl[1], ['m', 'msigma'], debias=False)
                pvis_bl_model = obs_model.unpack_bl(bl[0], bl[1], ['pvis', 'psigma'], debias=False)
                qvis_bl_model = obs_model.unpack_bl(bl[0], bl[1], ['qvis', 'qsigma'], debias=False)
                uvis_bl_model = obs_model.unpack_bl(bl[0], bl[1], ['uvis', 'usigma'], debias=False)

                if snrcut_dict['m'] > 0:
                    amask = np.abs(m_bl['m'])/m_bl['msigma'] > snrcut_dict['m']
                    m_bl = m_bl[amask]
                    m_bl_model = m_bl_model[amask]
                if snrcut_dict['pvis'] > 0:
                    amask = np.abs(pvis_bl['pvis'])/pvis_bl['psigma'] > snrcut_dict['pvis']
                    pvis_bl = pvis_bl[amask]
                    pvis_bl_model = pvis_bl_model[amask]
                if snrcut_dict['qvis'] > 0:
                    amask = np.abs(qvis_bl['qvis'])/qvis_bl['qsigma'] > snrcut_dict['qvis']
                    qvis_bl = qvis_bl[amask]
                    qvis_bl_model = qvis_bl_model[amask]
                if snrcut_dict['uvis'] > 0:
                    amask = np.abs(uvis_bl['uvis'])/uvis_bl['usigma'] > snrcut_dict['uvis']
                    uvis_bl = uvis_bl[amask]
                    uvis_bl_model = uvis_bl_model[amask]

                chisq_m_bl = np.sum(np.abs((m_bl['m'] - m_bl_model['m'])/m_bl['msigma'])**2)
                npts_m = len(m_bl_model)
                data_m = (bl[0], bl[1], npts_m, chisq_m_bl)
                bl_chisq_data_m.append(data_m)

                chisq_pvis_bl = np.sum(
                    np.abs((pvis_bl['pvis'] - pvis_bl_model['pvis'])/pvis_bl['psigma'])**2)
                npts_pvis = len(pvis_bl_model)
                data_pvis = (bl[0], bl[1], npts_pvis, chisq_pvis_bl)
                bl_chisq_data_pvis.append(data_pvis)

                chisq_qvis_bl = np.sum(
                    np.abs((qvis_bl['qvis'] - qvis_bl_model['qvis'])/qvis_bl['qsigma'])**2)
                npts_qvis = len(qvis_bl_model)
                data_qvis = (bl[0], bl[1], npts_qvis, chisq_qvis_bl)
                bl_chisq_data_qvis.append(data_qvis)

                chisq_uvis_bl = np.sum(
                    np.abs((uvis_bl['uvis'] - uvis_bl_model['uvis'])/uvis_bl['usigma'])**2)
                npts_uvis = len(uvis_bl_model)
                data_uvis = (bl[0], bl[1], npts_uvis, chisq_uvis_bl)
                bl_chisq_data_uvis.append(data_uvis)

        # sort by decreasing chi^2
        idx_m = np.argsort([data[-1] for data in bl_chisq_data_m])
        idx_m = list(reversed(idx_m))
        idx_p = np.argsort([data[-1] for data in bl_chisq_data_pvis])
        idx_p = list(reversed(idx_p))
        idx_q = np.argsort([data[-1] for data in bl_chisq_data_qvis])
        idx_q = list(reversed(idx_q))
        idx_u = np.argsort([data[-1] for data in bl_chisq_data_uvis])
        idx_u = list(reversed(idx_u))

        chisqtab_m = (r"\begin{tabular}{ l|l|l|l } \hline Baseline & $N_{m}$ & " +
                      r"$\chi^2_{m}/N_{m}$ & $\chi^2_{m}/N_{total}$ \\ " +
                      r"\hline \hline")
        chisqtab_p = (r"\begin{tabular}{ l|l|l|l } \hline Baseline & $N_{p}$ & " +
                      r"$\chi^2_{p}/N_{p}$ & $\chi^2_{p}/N_{total}$ \\ " +
                      r"\hline \hline")
        chisqtab_q = (r"\begin{tabular}{ l|l|l|l } \hline Baseline & $N_{Q}$ & " +
                      r"$\chi^2_{Q}/N_{Q}$ & $\chi^2_{Q}/N_{total}$ \\ " +
                      r"\hline \hline")
        chisqtab_u = (r"\begin{tabular}{ l|l|l|l } \hline Baseline & $N_{U}$ & " +
                      r"$\chi^2_{U}/N_{U}$ & $\chi^2_{U}/N_{total}$ \\ " +
                      r"\hline \hline")

        for i in range(len(bl_chisq_data_m)):
            if i > 45:
                break
            data = bl_chisq_data_m[idx_m[i]]
            tristr = r"%s-%s" % (data[0], data[1])
            nstr = r"%i" % data[2]
            chisqstr = r"%0.1f" % data[3]
            rchisqstr = r"%0.1f" % (data[3]/float(data[2]))
            rrchisqstr = r"%0.3f" % (data[3]/float(n_bl))
            if i == 0:
                chisqtab_m += r" " + tristr + " & " + nstr + " & " + rchisqstr + " & " + rrchisqstr
            else:
                chisqtab_m += r" \\" + tristr + " & " + nstr + " & " + rchisqstr + " & " + rrchisqstr
        for i in range(len(bl_chisq_data_pvis)):
            if i > 45:
                break
            data = bl_chisq_data_pvis[idx_p[i]]
            tristr = r"%s-%s" % (data[0], data[1])
            nstr = r"%i" % data[2]
            rchisqstr = r"%0.1f" % (data[3]/float(data[2]))
            rrchisqstr = r"%0.3f" % (data[3]/float(n_bl))
            if i == 0:
                chisqtab_p += r" " + tristr + " & " + nstr + " & " + rchisqstr + " & " + rrchisqstr
            else:
                chisqtab_p += r" \\" + tristr + " & " + nstr + " & " + rchisqstr + " & " + rrchisqstr
        for i in range(len(bl_chisq_data_qvis)):
            if i > 45:
                break
            data = bl_chisq_data_qvis[idx_q[i]]
            tristr = r"%s-%s" % (data[0], data[1])
            nstr = r"%i" % data[2]
            rchisqstr = r"%0.1f" % (data[3]/float(data[2]))
            rrchisqstr = r"%0.3f" % (data[3]/float(n_bl))
            if i == 0:
                chisqtab_q += r" " + tristr + " & " + nstr + " & " + rchisqstr + " & " + rrchisqstr
            else:
                chisqtab_q += r" \\" + tristr + " & " + nstr + " & " + rchisqstr + " & " + rrchisqstr
        for i in range(len(bl_chisq_data_uvis)):
            if i > 45:
                break
            data = bl_chisq_data_qvis[idx_u[i]]
            tristr = r"%s-%s" % (data[0], data[1])
            nstr = r"%i" % data[2]
            rchisqstr = r"%0.1f" % (data[3]/float(data[2]))
            rrchisqstr = r"%0.3f" % (data[3]/float(n_bl))
            if i == 0:
                chisqtab_u += r" " + tristr + " & " + nstr + " & " + rchisqstr + " & " + rrchisqstr
            else:
                chisqtab_u += r" \\" + tristr + " & " + nstr + " & " + rchisqstr + " & " + rrchisqstr

        chisqtab_m += r" \end{tabular}"
        chisqtab_p += r" \end{tabular}"
        chisqtab_q += r" \end{tabular}"
        chisqtab_u += r" \end{tabular}"

        ax = plt.subplot(gs[0:3, 0:2])
        ax.set_title('baseline m statistics')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.text(0.5, .975, chisqtab_m, ha="center", va="top", transform=ax.transAxes, size=fontsize)

        ax = plt.subplot(gs[0:3, 2:5])
        ax.set_title('baseline P statistics')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.text(0.5, .975, chisqtab_p, ha="center", va="top", transform=ax.transAxes, size=fontsize)

        ax = plt.subplot(gs[3:6, 0:2])
        ax.set_title('baseline Q statistics')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.text(0.5, .975, chisqtab_q, ha="center", va="top", transform=ax.transAxes, size=fontsize)

        ax = plt.subplot(gs[3:6, 2:5])
        ax.set_title('baseline U statistics')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.text(0.5, .975, chisqtab_u, ha="center", va="top", transform=ax.transAxes, size=fontsize)

        # save the first page of the plot
        print('saving pdf page 3')
        pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
        plt.close()

        ################################################################################
        # plot the baseline pol amps and  phases
        page = 4
        if mplots:
            print("===========================================")
            print("plotting fractional polarizatons")
            fig = plt.figure(3, figsize=(18, 28), dpi=200)
            plt.suptitle("Fractional  Polarization Plots", y=.9,
                         va='center', fontsize=int(1.2*fontsize))
            gs = gridspec.GridSpec(6, 4, wspace=WSPACE, hspace=HSPACE)
            i = 0
            j = 0

            obs_model.data['sigma'] *= 0
            obs_model.data['qsigma'] *= 0
            obs_model.data['usigma'] *= 0
            obs_model.data['vsigma'] *= 0

            amax = 1.1*np.max(np.abs(np.abs(obs_model.unpack(['mamp'])['mamp'])))
            obs_all = [obs, obs_model]
            for nbl, bl in enumerate(uniquebl):
                j = 0
                ax = plt.subplot(gs[2*i:2*(i+1), 2*j:2*(j+1)])
                ax = plot_bl_obs_compare(obs_all, bl[0], bl[1], 'mamp', rangey=[0, amax],
                                         markersize=MARKERSIZE, debias=False,
                                         snrcut=snrcut_dict['m'],
                                         axis=ax, legend=False, clist=['k', ehc.SCOLORS[1]],
                                         ttype='nfft', show=False, ebar=ebar)
                ax.set_xlabel('')
                j = 1
                ax = plt.subplot(gs[2*i:2*(i+1), 2*j:2*(j+1)])
                ax = plot_bl_obs_compare(obs_all, bl[0], bl[1], 'mphase', rangey=[-180, 180],

                                         markersize=MARKERSIZE, debias=False,
                                         snrcut=snrcut_dict['m'],
                                         axis=ax, legend=False, clist=['k', ehc.SCOLORS[1]],
                                         ttype='nfft', show=False, ebar=ebar)
                i += 1
                ax.set_xlabel('')

                if ax is None:
                    continue

                if i == 3:
                    print('saving pdf page %i' % page)
                    page += 1
                    pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
                    plt.close()
                    fig = plt.figure(3, figsize=(18, 28), dpi=200)
                    gs = gridspec.GridSpec(6, 4, wspace=WSPACE, hspace=HSPACE)
                    i = 0
                    j = 0

                if nbl == len(uniquebl):
                    print('saving pdf page %i' % page)
                    page += 1
                    pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
                    plt.close()

        if pplots:
            print("===========================================")
            print("plotting total polarizaton")
            fig = plt.figure(3, figsize=(18, 28), dpi=200)
            plt.suptitle("Total  Polarization Plots", y=.9, va='center', fontsize=int(1.2*fontsize))
            gs = gridspec.GridSpec(6, 4, wspace=WSPACE, hspace=HSPACE)
            i = 0
            j = 0

            obs_model.data['sigma'] *= 0
            obs_model.data['qsigma'] *= 0
            obs_model.data['usigma'] *= 0
            obs_model.data['vsigma'] *= 0

            amax = 1.1*np.max(np.abs(np.abs(obs_model.unpack(['pamp'])['pamp'])))
            obs_all = [obs, obs_model]
            for nbl, bl in enumerate(uniquebl):
                j = 0
                ax = plt.subplot(gs[2*i:2*(i+1), 2*j:2*(j+1)])
                ax = plot_bl_obs_compare(obs_all, bl[0], bl[1], 'pamp', rangey=[0, amax],
                                         markersize=MARKERSIZE, debias=False,
                                         snrcut=snrcut_dict['pvis'],
                                         axis=ax, legend=False, clist=['k', ehc.SCOLORS[1]],
                                         ttype='nfft', show=False, ebar=ebar)
                ax.set_xlabel('')
                j = 1
                ax = plt.subplot(gs[2*i:2*(i+1), 2*j:2*(j+1)])
                ax = plot_bl_obs_compare(obs_all, bl[0], bl[1], 'pphase', rangey=[-180, 180],
                                         markersize=MARKERSIZE, debias=False,
                                         snrcut=snrcut_dict['pvis'],
                                         axis=ax, legend=False, clist=['k', ehc.SCOLORS[1]],
                                         ttype='nfft', show=False, ebar=ebar)
                i += 1
                ax.set_xlabel('')

                if ax is None:
                    continue

                if i == 3:
                    print('saving pdf page %i' % page)
                    page += 1
                    pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
                    plt.close()
                    fig = plt.figure(3, figsize=(18, 28), dpi=200)
                    gs = gridspec.GridSpec(6, 4, wspace=WSPACE, hspace=HSPACE)
                    i = 0
                    j = 0

                if nbl == len(uniquebl):
                    print('saving pdf page %i' % page)
                    page += 1
                    pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
                    plt.close()

        if qplots:
            print("===========================================")
            print("plotting Q fit")
            fig = plt.figure(3, figsize=(18, 28), dpi=200)
            plt.suptitle("Q Plots", y=.9, va='center', fontsize=int(1.2*fontsize))
            gs = gridspec.GridSpec(6, 4, wspace=WSPACE, hspace=HSPACE)
            i = 0
            j = 0

            obs_model.data['sigma'] *= 0
            obs_model.data['qsigma'] *= 0
            obs_model.data['usigma'] *= 0
            obs_model.data['vsigma'] *= 0

            amax = 1.1*np.max(np.abs(np.abs(obs_model.unpack(['qamp'])['qamp'])))
            obs_all = [obs, obs_model]
            for nbl, bl in enumerate(uniquebl):
                j = 0
                ax = plt.subplot(gs[2*i:2*(i+1), 2*j:2*(j+1)])
                ax = plot_bl_obs_compare(obs_all, bl[0], bl[1], 'qamp', rangey=[0, amax],
                                         markersize=MARKERSIZE, debias=False,
                                         snrcut=snrcut_dict['qvis'],
                                         axis=ax, legend=False, clist=['k', ehc.SCOLORS[1]],
                                         ttype='nfft', show=False, ebar=ebar)
                ax.set_xlabel('')
                j = 1
                ax = plt.subplot(gs[2*i:2*(i+1), 2*j:2*(j+1)])
                ax = plot_bl_obs_compare(obs_all, bl[0], bl[1], 'qphase', rangey=[-180, 180],
                                         markersize=MARKERSIZE, debias=False,
                                         snrcut=snrcut_dict['qvis'],
                                         axis=ax, legend=False, clist=['k', ehc.SCOLORS[1]],
                                         ttype='nfft', show=False, ebar=ebar)
                i += 1
                ax.set_xlabel('')

                if ax is None:
                    continue

                if i == 3:
                    print('saving pdf page %i' % page)
                    page += 1
                    pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
                    plt.close()
                    fig = plt.figure(3, figsize=(18, 28), dpi=200)
                    gs = gridspec.GridSpec(6, 4, wspace=WSPACE, hspace=HSPACE)
                    i = 0
                    j = 0

                if nbl == len(uniquebl):
                    print('saving pdf page %i' % page)
                    page += 1
                    pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
                    plt.close()

        if uplots:
            print("===========================================")
            print("plotting U fit")
            fig = plt.figure(3, figsize=(18, 28), dpi=200)
            plt.suptitle("U Plots", y=.9, va='center', fontsize=int(1.2*fontsize))
            gs = gridspec.GridSpec(6, 4, wspace=WSPACE, hspace=HSPACE)
            i = 0
            j = 0

            obs_model.data['sigma'] *= 0
            obs_model.data['qsigma'] *= 0
            obs_model.data['usigma'] *= 0
            obs_model.data['vsigma'] *= 0

            amax = 1.1*np.max(np.abs(np.abs(obs_model.unpack(['uamp'])['uamp'])))
            obs_all = [obs, obs_model]
            for nbl, bl in enumerate(uniquebl):
                j = 0
                ax = plt.subplot(gs[2*i:2*(i+1), 2*j:2*(j+1)])
                ax = plot_bl_obs_compare(obs_all, bl[0], bl[1], 'uamp', rangey=[0, amax],
                                         markersize=MARKERSIZE, debias=False,
                                         snrcut=snrcut_dict['uvis'],
                                         axis=ax, legend=False, clist=['k', ehc.SCOLORS[1]],
                                         ttype='nfft', show=False, ebar=ebar)
                ax.set_xlabel('')
                j = 1
                ax = plt.subplot(gs[2*i:2*(i+1), 2*j:2*(j+1)])
                ax = plot_bl_obs_compare(obs_all, bl[0], bl[1], 'uphase', rangey=[-180, 180],
                                         markersize=MARKERSIZE, debias=False,
                                         snrcut=snrcut_dict['uvis'],
                                         axis=ax, legend=False, clist=['k', ehc.SCOLORS[1]],
                                         ttype='nfft', show=False, ebar=ebar)
                i += 1
                ax.set_xlabel('')

                if ax is None:
                    continue

                if i == 3:
                    print('saving pdf page %i' % page)
                    page += 1
                    pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
                    plt.close()
                    fig = plt.figure(3, figsize=(18, 28), dpi=200)
                    gs = gridspec.GridSpec(6, 4, wspace=WSPACE, hspace=HSPACE)
                    i = 0
                    j = 0

                if nbl == len(uniquebl):
                    print('saving pdf page %i' % page)
                    page += 1
                    pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
                    plt.close()


def _display_img(im, beamparams=None, scale='linear', gamma=0.5, cbar_lims=False,
                 has_cbar=True, has_title=True, cfun='afmhot', dynamic_range=100,
                 axis=False, show=False, fontsize=FONTSIZE):
    """display the figure on a given axis
       cannot use im.display because  it makes a new figure
    """

    interp = 'gaussian'

    if axis:
        ax = axis
    else:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    imvec = np.array(im.imvec).reshape(-1)
    # flux unit is mJy/uas^2
    imvec = imvec * 1.e3
    fovfactor = im.xdim*im.psize*(1/ehc.RADPERUAS)
    factor = (1./fovfactor)**2 / (1./im.xdim)**2
    imvec = imvec * factor

    imarr = (imvec).reshape(im.ydim, im.xdim)
    unit = 'mJy/$\mu$ as$^2$'
    if scale == 'log':
        if (imarr < 0.0).any():
            print('clipping values less than 0')
            imarr[imarr < 0.0] = 0.0
        imarr = np.log(imarr + np.max(imarr)/dynamic_range)
        unit = 'log(' + unit + ')'

    if scale == 'gamma':
        if (imarr < 0.0).any():
            print('clipping values less than 0')
            imarr[imarr < 0.0] = 0.0
        imarr = (imarr + np.max(imarr)/dynamic_range)**(gamma)
        unit = '(' + unit + ')^gamma'

    if cbar_lims:
        imarr[imarr > cbar_lims[1]] = cbar_lims[1]
        imarr[imarr < cbar_lims[0]] = cbar_lims[0]

    if cbar_lims:
        ax = ax.imshow(imarr, cmap=plt.get_cmap(cfun), interpolation=interp,
                       vmin=cbar_lims[0], vmax=cbar_lims[1])
    else:
        ax = ax.imshow(imarr, cmap=plt.get_cmap(cfun), interpolation=interp)

    if has_cbar:
        cbar = plt.colorbar(ax, fraction=0.046, pad=0.04, format='%1.2g')
        cbar.set_label(unit, fontsize=fontsize)
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(labelsize=16)
        if cbar_lims:
            plt.clim(cbar_lims[0], cbar_lims[1])

    if not(beamparams is None):
        beamparams = [beamparams[0], beamparams[1], beamparams[2],
                      -.35*im.fovx(), -.35*im.fovy()]
        beamimage = im.copy()
        beamimage.imvec *= 0
        beamimage = beamimage.add_gauss(1, beamparams)
        halflevel = 0.5*np.max(beamimage.imvec)
        beamimarr = (beamimage.imvec).reshape(beamimage.ydim, beamimage.xdim)
        plt.contour(beamimarr, levels=[halflevel], colors='w', linewidths=3)
        ax = plt.gca()

    plt.axis('off')
    fov_uas = im.xdim * im.psize / ehc.RADPERUAS  # get the fov in uas
    roughfactor = 1./3.  # make the bar about 1/3 the fov
    fov_scale = 40
    start = im.xdim * roughfactor / 3.0  # select the start location
    end = start + fov_scale/fov_uas * im.xdim  # determine the end location
    plt.plot([start, end], [im.ydim-start, im.ydim-start], color="white", lw=1)  # plot line
    plt.text(x=(start+end)/2.0, y=im.ydim-start-im.ydim/20, s=str(fov_scale) + " $\mu$as",
             color="white", ha="center", va="center",
             fontsize=int(1.2*fontsize), fontweight='bold')

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if show:
        plt.show(block=False)

    return ax


def _display_img_pol(im, beamparams=None, scale='linear', gamma=0.5, cbar_lims=False,
                     has_cbar=True, has_title=True, cfun='afmhot', pol=None, polticks=False,
                     nvec=False, pcut=0.1,  mcut=0.01, contour=False, dynamic_range=100,
                     axis=False, show=False, fontsize=FONTSIZE):
    """display the polarimetric figure on a given axis
       cannot use im.display because  it makes a new figure
    """

    interp = 'gaussian'

    if axis:
        ax = axis
    else:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    if pol == 'm':
        imvec = im.mvec
        unit = r'$|\breve{m}|$'
        factor = 1
        cbar_lims = [0, 1]
    elif pol == 'chi':
        imvec = im.chivec / ehc.DEGREE
        unit = r'$\chi (^\circ)$'
        factor = 1
        cbar_lims = [0, 180]
    else:
        # flux unit is Tb
        factor = 3.254e13/(im.rf**2 * im.psize**2)
        unit = 'Tb (K)'
        try:
            imvec = np.array(im._imdict[pol]).reshape(-1)
        except KeyError:
            try:
                if im.polrep == 'stokes':
                    im2 = im.switch_polrep('circ')
                elif im.polrep == 'circ':
                    im2 = im.switch_polrep('stokes')
                imvec = np.array(im2._imdict[pol]).reshape(-1)
            except KeyError:
                raise Exception("Cannot make pol %s image in display()!" % pol)

    # flux unit is Tb
    imvec = imvec * factor
    imarr = (imvec).reshape(im.ydim, im.xdim)

    if scale == 'log':
        if (imarr < 0.0).any():
            print('clipping values less than 0')
            imarr[imarr < 0.0] = 0.0
        imarr = np.log(imarr + np.max(imarr)/dynamic_range)
        unit = 'log(' + unit + ')'

    if scale == 'gamma':
        if (imarr < 0.0).any():
            print('clipping values less than 0')
            imarr[imarr < 0.0] = 0.0
        imarr = (imarr + np.max(imarr)/dynamic_range)**(gamma)
        unit = '(' + unit + ')^gamma'

    if cbar_lims:
        imarr[imarr > cbar_lims[1]] = cbar_lims[1]
        imarr[imarr < cbar_lims[0]] = cbar_lims[0]

    if cbar_lims:
        ax = ax.imshow(imarr, cmap=plt.get_cmap(cfun), interpolation=interp,
                       vmin=cbar_lims[0], vmax=cbar_lims[1])
    else:
        ax = ax.imshow(imarr, cmap=plt.get_cmap(cfun), interpolation=interp)

    if contour:
        plt.contour(imarr, colors='k', linewidths=.25)

    if polticks:
        im_stokes = im.switch_polrep(polrep_out='stokes')
        ivec = np.array(im_stokes.imvec).reshape(-1)
        qvec = np.array(im_stokes.qvec).reshape(-1)
        uvec = np.array(im_stokes.uvec).reshape(-1)
        vvec = np.array(im_stokes.vvec).reshape(-1)

        if len(ivec) == 0:
            ivec = np.zeros(im_stokes.ydim*im_stokes.xdim)
        if len(qvec) == 0:
            qvec = np.zeros(im_stokes.ydim*im_stokes.xdim)
        if len(uvec) == 0:
            uvec = np.zeros(im_stokes.ydim*im_stokes.xdim)
        if len(vvec) == 0:
            vvec = np.zeros(im_stokes.ydim*im_stokes.xdim)

        if not nvec:
            nvec = im.xdim // 2

        thin = im.xdim//nvec
        maska = (ivec).reshape(im.ydim, im.xdim) > pcut * np.max(ivec)
        maskb = (np.abs(qvec + 1j*uvec)/ivec).reshape(im.ydim, im.xdim) > mcut
        mask = maska * maskb
        mask2 = mask[::thin, ::thin]
        x = (np.array([[i for i in range(im.xdim)] for j in range(im.ydim)])[::thin, ::thin])
        x = x[mask2]
        y = (np.array([[j for i in range(im.xdim)] for j in range(im.ydim)])[::thin, ::thin])
        y = y[mask2]
        a = (-np.sin(np.angle(qvec+1j*uvec)/2).reshape(im.ydim, im.xdim)[::thin, ::thin])
        a = a[mask2]
        b = (np.cos(np.angle(qvec+1j*uvec)/2).reshape(im.ydim, im.xdim)[::thin, ::thin])
        b = b[mask2]

        plt.quiver(x, y, a, b,
                   headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                   width=.01*im.xdim, units='x', pivot='mid', color='k', angles='uv',
                   scale=1.0/thin)
        plt.quiver(x, y, a, b,
                   headaxislength=20, headwidth=1, headlength=.01, minlength=0, minshaft=1,
                   width=.005*im.xdim, units='x', pivot='mid', color='w', angles='uv',
                   scale=1.1/thin)

    if has_cbar:
        cbar = plt.colorbar(ax, fraction=0.046, pad=0.04, format='%1.2g')
        cbar.set_label(unit, fontsize=fontsize)
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(labelsize=16)
        if cbar_lims:
            ax.set_clim(cbar_lims[0], cbar_lims[1])

    if not(beamparams is None):
        beamparams = [beamparams[0], beamparams[1], beamparams[2],
                      -.35*im.fovx(), -.35*im.fovy()]
        beamimage = im.copy()
        beamimage.imvec *= 0
        beamimage = beamimage.add_gauss(1, beamparams)
        halflevel = 0.5*np.max(beamimage.imvec)
        beamimarr = (beamimage.imvec).reshape(beamimage.ydim, beamimage.xdim)
        plt.contour(beamimarr, levels=[halflevel], colors='w', linewidths=3)
        ax = plt.gca()

    plt.axis('off')
    if has_cbar:
        fov_uas = im.xdim * im.psize / ehc.RADPERUAS  # get the fov in uas
        roughfactor = 1./3.  # make the bar about 1/3 the fov
        fov_scale = 40
        # round around 1/3 the fov to nearest 10
        start = im.xdim * roughfactor / 3.0  # select the start location
        end = start + fov_scale/fov_uas * im.xdim
        # determine the end location based on the size of the bar
        plt.plot([start, end], [im.ydim-start, im.ydim-start], color="white", lw=1)  # plot line
        plt.text(x=(start+end)/2.0, y=im.ydim-start-im.ydim/20, s=str(fov_scale) + " $\mu$as",
                 color="white", ha="center", va="center",
                 fontsize=int(1.2*fontsize), fontweight='bold')

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if show:
        plt.show(block=False)

    return ax
