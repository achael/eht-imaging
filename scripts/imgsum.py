#imgsum.py
#Andrew Chael 07/12/2018
#produce an image summary plot for an image and uvfits file

#TODO add systematic noise to individual closure quantities? 
from __future__ import print_function
import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import argparse
import math
import os
import datetime

#display parameters
SCOLORS = eh.SCOLORS
FONTSIZE = 22
WSPACE=0.8
HSPACE=0.3
MARGINS=0.5
PROCESSES=4
MARKERSIZE=5

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rc('font', size=FONTSIZE)      
plt.rc('axes', titlesize=FONTSIZE)
plt.rc('axes', labelsize=FONTSIZE) 
plt.rc('xtick', labelsize=FONTSIZE)
plt.rc('ytick', labelsize=FONTSIZE) 
plt.rc('legend', fontsize=FONTSIZE)    
plt.rc('figure', titlesize=FONTSIZE) 


################################################################################
################################################################################
#im = eh.image.load_fits('./out.fits')
#obs = eh.obsdata.load_uvfits('./obs.uvfits')
#sys = 0.1
def main(im, obs, obs_uncal, basename, outname, 
         commentstr="", outdir='.',ebar=True,cfun='afmhot',sysnoise=0,syscnoise=0,fontsize=FONTSIZE,
         gainplots=True,cphaseplots=True,campplots=True):

    with PdfPages(outname) as pdf:
        titlestr = 'Summary Sheet for %s on MJD %s' % (im.source, im.mjd)
        #pdf metadata 
        d = pdf.infodict()
        d['Title'] = basename
        d['Author'] = u'EHT Team 1'
        d['Subject'] = titlestr
        d['CreationDate'] = datetime.datetime.today()
        d['ModDate'] = datetime.datetime.today()

        #define the figure
        fig = plt.figure(1, figsize=(18, 28), dpi=200)
        gs = gridspec.GridSpec(6,4, wspace=WSPACE, hspace=HSPACE)

        #user comments
        if len(commentstr)>1:
            titlestr =titlestr+'\n'+str(commentstr)
        else:
            titlestr = titlestr

        plt.suptitle(titlestr,y=.9,va='center',fontsize=int(1.2*fontsize))

################################################################################
        print("===========================================")
        print("displaying the image")
        ax = plt.subplot(gs[0:2,0:2])
        ax.set_title('Submitted Image')
        ax = display_img(im,axis=ax, show=False,has_title=False,cfun=cfun,fontsize=fontsize)

        print("===========================================")
        print("displaying the blurred image")
        ax = plt.subplot(gs[0:2,2:5])
        ax.set_title('Image blurred to nominal resolution')
        #beamparams=obs_uncal.fit_gauss()
        #fwhm = np.min((np.abs(beamparams[0]),np.abs(beamparams[1])))
        fwhm = obs.res()
        print ("blur_FWHM: " , fwhm/eh.RADPERUAS)
        beamparams = [fwhm, fwhm, 0]

        res = obs.res()
        imblur = im.blur_gauss(beamparams, frac=1.0)
        #imblur = im.blur_circ(res)
        ax = display_img(imblur , beamparams=beamparams,axis=ax, show=False,has_title=False,cfun=cfun,fontsize=fontsize)

################################################################################
        print("===========================================")
        print("calculating statistics")
        #display the overall chi2
        ax = plt.subplot(gs[2,0:2])
        ax.set_title('Image statistics')
        #ax.axis('off')
        ax.set_yticks([])
        ax.set_xticks([])

        flux = im.total_flux()

        # SNR ordering
        #obs.reorder_tarr_snr()
        #obs_uncal.reorder_tarr_snr()
        maxset=False
        # compute chi^2
        chi2vis = obs.chisq(im, dtype='vis', ttype='nfft', systematic_noise=sysnoise, maxset=maxset)
        chi2amp = obs.chisq(im, dtype='amp', ttype='nfft', systematic_noise=sysnoise, maxset=maxset)
        chi2cphase = obs.chisq(im, dtype='cphase', ttype='nfft', systematic_noise=sysnoise, systematic_cphase_noise=syscnoise, maxset=maxset)
        chi2logcamp = obs.chisq(im, dtype='logcamp', ttype='nfft', systematic_noise=sysnoise, maxset=maxset)
        chi2camp = obs.chisq(im, dtype='camp', ttype='nfft', systematic_noise=sysnoise, maxset=maxset)

        maxset=True
#        chi2vis_uncal = obs_uncal.chisq(im, dtype='vis', ttype='nfft', systematic_noise=sysnoise, maxset=maxset)
#        chi2amp_uncal = obs_uncal.chisq(im, dtype='amp', ttype='nfft', systematic_noise=sysnoise, maxset=maxset)
#        chi2cphase_uncal = obs_uncal.chisq(im, dtype='cphase', ttype='nfft', systematic_noise=sysnoise, systematic_cphase_noise=syscnoise, maxset=maxset)
#        chi2logcamp_uncal = obs_uncal.chisq(im, dtype='logcamp', ttype='nfft', systematic_noise=sysnoise, maxset=maxset)
#        chi2camp_uncal = obs_uncal.chisq(im, dtype='camp', ttype='nfft', systematic_noise=sysnoise, maxset=maxset)
        chi2vis_uncal = obs.chisq(im, dtype='vis', ttype='nfft', systematic_noise=0, maxset=maxset)
        chi2amp_uncal = obs.chisq(im, dtype='amp', ttype='nfft', systematic_noise=0, maxset=maxset)
        chi2cphase_uncal = obs.chisq(im, dtype='cphase', ttype='nfft', systematic_noise=0, systematic_cphase_noise=0, maxset=maxset)
        chi2logcamp_uncal = obs.chisq(im, dtype='logcamp', ttype='nfft', systematic_noise=0, maxset=maxset)
        chi2camp_uncal = obs.chisq(im, dtype='camp', ttype='nfft', systematic_noise=0, maxset=maxset)

        print("chi^2 vis: ", chi2vis, chi2vis_uncal)
        print("chi^2 amp: ", chi2amp, chi2amp_uncal)
        print("chi^2 cphase: ", chi2cphase, chi2cphase_uncal)
        print("chi^2 logcamp: ", chi2logcamp, chi2logcamp_uncal)
        print("chi^2 camp: ", chi2logcamp, chi2logcamp_uncal)

        fs =int(1*fontsize)
        fs2 =int(.8*fontsize)
        ax.text(.05,.9,"Source:", fontsize=fs,
        ha='left',va='center',transform=ax.transAxes)
        ax.text(.05,.7,"MJD:" , fontsize=fs,
        ha='left',va='center',transform=ax.transAxes)
        ax.text(.05,.5,"FREQ:" , fontsize=fs,
        ha='left',va='center',transform=ax.transAxes)
        ax.text(.05,.3,"FOV:", fontsize=fs,
        ha='left',va='center',transform=ax.transAxes)
        ax.text(.05,.1,"FLUX:" , fontsize=fs,
        ha='left',va='center',transform=ax.transAxes)

        ax.text(.23,.9,"%s"%im.source, fontsize=fs,
        ha='left',va='center',transform=ax.transAxes)
        ax.text(.23,.7,"%i" % im.mjd, fontsize=fs,
        ha='left',va='center',transform=ax.transAxes)
        ax.text(.23,.5,"%0.0f GHz" % (im.rf/1.e9), fontsize=fs,
        ha='left',va='center',transform=ax.transAxes)
        ax.text(.23,.3,"%0.1f $\mu$as" % (im.fovx()/eh.RADPERUAS), fontsize=fs,
        ha='left',va='center',transform=ax.transAxes)
        ax.text(.23,.1,"%0.2f Jy" % flux, fontsize=fs,
        ha='left',va='center',transform=ax.transAxes)

        ax.text(.5,.9,"$\chi^2_{vis}$" , fontsize=fs,
        ha='left',va='center',transform=ax.transAxes)
        ax.text(.5,.7,"$\chi^2_{amp}$" , fontsize=fs,
        ha='left',va='center',transform=ax.transAxes)
        ax.text(.5,.5,"$\chi^2_{cphase}$" , fontsize=fs,
        ha='left',va='center',transform=ax.transAxes)
        ax.text(.5,.3,"$\chi^2_{log camp}$" , fontsize=fs,
        ha='left',va='center',transform=ax.transAxes)
        ax.text(.5,.1,"$\chi^2_{camp}$" , fontsize=fs,
        ha='left',va='center',transform=ax.transAxes)

        ax.text(.72,.9,"%0.2f" % chi2vis, fontsize=fs,
        ha='left',va='center',transform=ax.transAxes)
        ax.text(.72,.7,"%0.2f" % chi2amp, fontsize=fs,
        ha='left',va='center',transform=ax.transAxes)
        ax.text(.72,.5,"%0.2f" % chi2cphase, fontsize=fs,
        ha='left',va='center',transform=ax.transAxes)
        ax.text(.72,.3,"%0.2f" % chi2logcamp, fontsize=fs,
        ha='left',va='center',transform=ax.transAxes)
        ax.text(.72,.1,"%0.2f" % chi2camp, fontsize=fs,
        ha='left',va='center',transform=ax.transAxes)

        ax.text(.85,.9,"(%0.2f)" % chi2vis_uncal, fontsize=fs2,
        ha='left',va='center',transform=ax.transAxes)
        ax.text(.85,.7,"(%0.2f)" % chi2amp_uncal, fontsize=fs2,
        ha='left',va='center',transform=ax.transAxes)
        ax.text(.85,.5,"(%0.2f)" % chi2cphase_uncal, fontsize=fs2,
        ha='left',va='center',transform=ax.transAxes)
        ax.text(.85,.3,"(%0.2f)" % chi2logcamp, fontsize=fs2,
        ha='left',va='center',transform=ax.transAxes)
        ax.text(.85,.1,"(%0.2f)" % chi2camp_uncal, fontsize=fs2,
        ha='left',va='center',transform=ax.transAxes)

################################################################################
        print("===========================================")
        print("calculating cphase statistics")
        #display the closure  phase chi2
        ax = plt.subplot(gs[3:6,0:2])
        ax.set_title('Closure phase statistics')
        #ax.axis('off')
        ax.set_yticks([])
        ax.set_xticks([])

        # get closure triangle combinations
        # ANDREW -- hacky, fix this!
        cp = obs.c_phases(mode="all", count="min")
        n_cphase = len(cp)
        alltris = [(str(cpp['t1']),str(cpp['t2']),str(cpp['t3'])) for cpp in cp]
        uniqueclosure_tri = []
        for tri in alltris:
            if tri not in uniqueclosure_tri: uniqueclosure_tri.append(tri)
              
        # generate data
        obs_model = im.observe_same(obs, add_th_noise=False, ttype='nfft')
        cphases_obs = obs.c_phases(mode='all', count='max', vtype='vis')
        cphases_model = obs_model.c_phases(mode='all', count='max', vtype='vis')


        #generate chi^2 -- NO SYSTEMATIC NOISES
        ncphase = 0
        cphase_chisq_data=[]
        for c in range(0, len(uniqueclosure_tri)):
            cphases_obs_tri = obs.cphase_tri(uniqueclosure_tri[c][0], uniqueclosure_tri[c][1], uniqueclosure_tri[c][2],
                                             vtype='vis', ang_unit='deg', cphases=cphases_obs)

            if len(cphases_obs_tri)>0:
                cphases_model_tri = obs_model.cphase_tri(uniqueclosure_tri[c][0], uniqueclosure_tri[c][1], uniqueclosure_tri[c][2],
                                                         vtype='vis', ang_unit='deg', cphases=cphases_model)
                chisq_tri= 2*np.sum((1.0 - np.cos(cphases_obs_tri['cphase']*eh.DEGREE-cphases_model_tri['cphase']*eh.DEGREE))/
                                  ((cphases_obs_tri['sigmacp']*eh.DEGREE)**2))
                #chisq_tri /= len(cphases_obs_tri)
                npts = len(cphases_obs_tri)
                data =  [uniqueclosure_tri[c][0], uniqueclosure_tri[c][1], uniqueclosure_tri[c][2],npts,chisq_tri]
                cphase_chisq_data.append(data)

        #sort by decreasing chi^2
        idx = np.argsort([data[-1] for data in cphase_chisq_data])
        idx = list(reversed(idx))

        chisqtab=r"\begin{tabular}{ l|l|l|l } \hline Triangle & $N_{tri}$ & $\chi^2_{tri}/N_{tri}$ & $\chi^2_{tri}/N_{tot}$\\ \hline \hline"
        first=True
        for i in range(len(cphase_chisq_data)):
            if i>30:break
            data = cphase_chisq_data[idx[i]]
            tristr = r"%s-%s-%s" % (data[0],data[1],data[2])
            nstr = r"%i" % data[3]
            chisqstr = r"%0.1f" % data[4]
            rchisqstr = r"%0.1f" % (float(data[4])/float(data[3]))
            rrchisqstr = r"%0.3f" % (float(data[4])/float(n_cphase))
            if first:
                chisqtab += r" " + tristr + " & " + nstr + " & " + rchisqstr + " & " + rrchisqstr 
                first=False
            else:
                chisqtab += r" \\" + tristr + " & " + nstr + " & " + rchisqstr + " & " + rrchisqstr 
        chisqtab += r" \end{tabular}"

        ax.text(0.5,.975,chisqtab,ha="center",va="top",transform=ax.transAxes,size=fontsize)


################################################################################
        print("===========================================")
        print("calculating camp statistics")
        #display the log closure amplitude chi2
        ax = plt.subplot(gs[2:6,2::])
        ax.set_title('Log Closure amplitude statistics')
        #ax.axis('off')
        ax.set_yticks([])
        ax.set_xticks([])

        # get closure triangle combinations
        # ANDREW -- hacky, fix this!
        debias=True
        cp = obs.c_amplitudes(mode="all", count="min",ctype='logcamp',debias=debias)
        n_camps = len(cp)
        allquads = [(str(cpp['t1']),str(cpp['t2']),str(cpp['t3']),str(cpp['t4'])) for cpp in cp]
        uniqueclosure_quad = []
        for quad in allquads:
            if quad not in uniqueclosure_quad: 
                uniqueclosure_quad.append(quad)
              
        # generate data
        camps_obs = obs.c_amplitudes(mode='all', count='max', ctype='logcamp', debias=debias)
        camps_model = obs_model.c_amplitudes(mode='all', count='max', ctype='logcamp', debias=debias)


        #generate chi2 -- NO SYSTEMATIC NOISES
        ncamp = 0
        camp_chisq_data=[]
        for c in range(0, len(uniqueclosure_quad)):
            camps_obs_quad = obs.camp_quad(uniqueclosure_quad[c][0], uniqueclosure_quad[c][1], 
                                         uniqueclosure_quad[c][2],  uniqueclosure_quad[c][3],
                                         vtype='vis', camps=camps_obs, ctype='logcamp')

            if len(camps_obs_quad)>0:
                camps_model_quad = obs.camp_quad(uniqueclosure_quad[c][0], uniqueclosure_quad[c][1], 
                                             uniqueclosure_quad[c][2],  uniqueclosure_quad[c][3],
                                             vtype='vis', camps=camps_model, ctype='logcamp')
                chisq_quad = np.sum(np.abs((camps_obs_quad['camp'] - camps_model_quad['camp'])/camps_obs_quad['sigmaca'])**2)
                #chisq_quad /= (len(camps_obs_quad))
                npts = len(camps_obs_quad)

                data =  (uniqueclosure_quad[c][0], uniqueclosure_quad[c][1], uniqueclosure_quad[c][2],uniqueclosure_quad[c][3],
                         npts,
                         chisq_quad)
                camp_chisq_data.append(data)

        #sort by decreasing chi^2
        idx = np.argsort([data[-1] for data in camp_chisq_data])
        idx = list(reversed(idx))

        chisqtab=r"\begin{tabular}{ l|l|l|l } \hline Quadrangle & $N_{quad}$ & $\chi^2_{quad}/N_{quad}$ & $\chi^2_{quad}/N_{tot}$ \\ \hline \hline"
        for i in range(len(camp_chisq_data)):
            if i>45: break
            data = camp_chisq_data[idx[i]]
            tristr = r"%s-%s-%s-%s" % (data[0],data[1],data[2],data[3])
            nstr = r"%i" % data[4]
            chisqstr = r"%0.1f" % data[5]
            rchisqstr = r"%0.1f" % (data[5]/float(data[4]))
            rrchisqstr = r"%0.3f" % (data[5]/float(n_camps))
            if i==0:
                chisqtab += r" " + tristr + " & " + nstr + " & " + rchisqstr + " & " + rrchisqstr 
            else:
                chisqtab += r" \\" + tristr + " & " + nstr + " & " + rchisqstr + " & " + rrchisqstr 


        chisqtab += r" \end{tabular}"

        ax.text(0.5,.975,chisqtab,ha="center",va="top",transform=ax.transAxes,size=fontsize)


        #save the first page of the plot
        print('saving pdf page 1')
        #plt.tight_layout()
        #plt.subplots_adjust(wspace=1,hspace=1)
        #plt.savefig(outname, pad_inches=MARGINS,bbox_inches='tight')
        pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
        plt.close()

################################################################################
        #plot the vis amps
        fig = plt.figure(2, figsize=(18, 28), dpi=200)
        gs = gridspec.GridSpec(6,4, wspace=WSPACE, hspace=HSPACE)

        print("===========================================")
        print("plotting vis amps")
        ax = plt.subplot(gs[0:2,0:2])
        obs_tmp = obs_model.copy()
        obs_tmp.data['sigma']*=0.
        ax = eh.plotting.comp_plots.plotall_obs_compare([obs, obs_tmp], 
                                                   'uvdist','amp', axis=ax,legend=False, clist=['k',SCOLORS[1]], 
                                                    ttype='nfft',show=False, ebar=ebar,markersize=MARKERSIZE)
        #modify the labels
        #ax.xaxis.label.set_visible(False)
        #ax.yaxis.label.set_visible(False)
        ax.set_title('Calibrated Visiblity Amplitudes')
        ax.set_xlabel('u-v distance (G$\lambda$)')
        ax.set_xlim([0,1.e10])
        ax.set_xticks([0,2.e9,4.e9,6.e9,8.e9,10.e9])
        ax.set_xticklabels(["0","2","4","6","8","10"])
        ax.set_xticks([1.e9,3.e9,5.e9,7.e9,9.e9], minor=True)
        ax.set_xticklabels([], minor=True)

        ax.set_ylabel('Amplitude (Jy)')
        ax.set_ylim([0,1.2*flux])
        yticks_maj = np.array([0,.2,.4,.6,.8,1])*flux
        ax.set_yticks(yticks_maj)
        ax.set_yticklabels(["%0.2f"%fl for fl in yticks_maj])
        yticks_min = np.array([.1,.3,.5,.7,.9])*flux
        ax.set_yticks(yticks_min,minor=True)
        ax.set_yticklabels([], minor=True)

################################################################################3
        #plot the caltable gains
        if gainplots:
            print("===========================================")
            print("plotting gains")
            ax2 = plt.subplot(gs[0:2,2:6])
            obs_tmp = obs_uncal.copy()
            for i in range(1): 
                ct = eh.self_cal.self_cal(obs_tmp, im, 
                                          method='amp', ttype='nfft', 
                                          caltable=True, gain_tol=.2,
                                          processes=PROCESSES)
                ct = ct.pad_scans()
                obs_tmp =  ct.applycal(obs_tmp,interp='nearest',extrapolate=True) #apply caltable
                if np.any(np.isnan(obs_tmp.data['vis'])):
                    print ("Warning: NaN in applycal vis table!")
                    break
                if i>0:
                    ct_out = ct_out.merge([ct])
                else:
                    ct_out = ct

            ax2 = ct_out.plot_gains('all',rangey=[.1,10],yscale='log',axis=ax2,legend=True)

            #median gains
            ax = plt.subplot(gs[3:6,2:5])
            ax.set_title('Station gain statistics')
            #ax.axis('off')
            ax.set_yticks([])
            ax.set_xticks([])

            gain_data = []
            for station in ct_out.tarr['site']:
                try: 
                    gain = np.median(np.abs(ct_out.data[station]['lscale']))
                except: 
                    continue
                pdiff = np.abs(gain-1)*100
                data =  (station, gain, pdiff)
                gain_data.append(data)

            #sort by decreasing chi^2
            idx = np.argsort([data[-1] for data in gain_data])
            idx = list(reversed(idx))

            chisqtab=r"\begin{tabular}{ l|l|l } \hline Site & Median Gain & Percent diff. \\ \hline \hline"
            for i in range(len(gain_data)):
                if i>45: break
                data = gain_data[idx[i]]
                sitestr = r"%s" % (data[0])
                gstr = r"%0.2f" % data[1]
                pstr = r"%0.0f" % data[2]
                if i==0:
                    chisqtab += r" " + sitestr + " & " + gstr + " & " + pstr 
                else:
                    chisqtab += r" \\" + sitestr + " & " + gstr + " & " + pstr

            chisqtab += r" \end{tabular}"
            ax.text(0.5,.975,chisqtab,ha="center",va="top",transform=ax.transAxes,size=fontsize)

################################################################################3
        #baseline amplitude chi2
        print("===========================================")
        print("baseline vis amps chisq")
        ax = plt.subplot(gs[3:6,0:2])
        ax.set_title('Visibility amplitude statistics')
        #ax.axis('off')
        ax.set_yticks([])
        ax.set_xticks([])

        # get closure triangle combinations
        # ANDREW -- hacky, fix this!
        debias=True
        bl_unpk = obs.unpack(['t1','t2'],debias=debias)
        n_bl = len(bl_unpk)
        allbl = [(str(bl['t1']),str(bl['t2'])) for bl in bl_unpk]
        uniquebl = []
        for bl in allbl:
            if bl not in uniquebl: 
                uniquebl.append(bl)
              
        #generate chi2 -- NO SYSTEMATIC NOISES
        ncamp = 0
        bl_chisq_data=[]
        for ii in range(0, len(uniquebl)):
            bl = uniquebl[ii]
            amps_bl = obs.unpack_bl(bl[0],bl[1],['amp','sigma'], debias=False)

            if len(amps_bl)>0:
                amps_bl_model = obs_model.unpack_bl(bl[0],bl[1],['amp','sigma'], debias=False)
                chisq_bl = np.sum(np.abs((amps_bl['amp'] - amps_bl_model['amp'])/amps_bl['sigma'])**2)
                npts = len(amps_bl_model)

                data =  (bl[0],bl[1],
                         npts,
                         chisq_bl)
                bl_chisq_data.append(data)

        #sort by decreasing chi^2
        idx = np.argsort([data[-1] for data in bl_chisq_data])
        idx = list(reversed(idx))

        chisqtab=r"\begin{tabular}{ l|l|l|l } \hline Baseline & $N_{amp}$ & $\chi^2_{amp}/N_{amp}$ & $\chi^2_{amp}/N_{total}$ \\ \hline \hline"
        for i in range(len(bl_chisq_data)):
            if i>45: break
            data = bl_chisq_data[idx[i]]
            tristr = r"%s-%s" % (data[0],data[1])
            nstr = r"%i" % data[2]
            chisqstr = r"%0.1f" % data[3]
            rchisqstr = r"%0.1f" % (data[3]/float(data[2]))
            rrchisqstr = r"%0.3f" % (data[3]/float(n_bl))
            if i==0:
                chisqtab += r" " + tristr + " & " + nstr + " & " + rchisqstr + " & " + rrchisqstr 
            else:
                chisqtab += r" \\" + tristr + " & " + nstr + " & " + rchisqstr + " & " + rrchisqstr 


        chisqtab += r" \end{tabular}"

        ax.text(0.5,.975,chisqtab,ha="center",va="top",transform=ax.transAxes,size=fontsize)

        
        #save the first page of the plot
        print('saving pdf page 2')
        #plt.tight_layout()
        #plt.subplots_adjust(wspace=1,hspace=1)
        #plt.savefig(outname, pad_inches=MARGINS,bbox_inches='tight')
        pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
        plt.close()

################################################################################
        #plot the closure phases
        page =3 
        if cphaseplots:
            print("===========================================")
            print("plotting closure phases")
            fig = plt.figure(3, figsize=(18, 28), dpi=200)
            plt.suptitle("Closure Phase Plots",y=.9,va='center',fontsize=int(1.2*fontsize))
            gs = gridspec.GridSpec(6,4, wspace=WSPACE, hspace=HSPACE)
            i = 0
            j = 0
            switch=False
            obs_all = [obs, obs_model]
            cphases_model['sigmacp'] *= 0
            cphases_all = [cphases_obs, cphases_model]
            for tri in uniqueclosure_tri:

                ax = plt.subplot(gs[2*i:2*(i+1), 2*j:2*(j+1)])
                ax = eh.plotting.comp_plots.plot_cphase_obs_compare(obs_all,tri[0],tri[1],tri[2], rangey=[-185,185],
                                                                     cphases=cphases_all,markersize=MARKERSIZE,
                                                                     axis=ax,legend=False, clist=['k',SCOLORS[1]],
                                                                     ttype='nfft',show=False, ebar=ebar)
                if ax is None: continue
                if switch:
                    i += 1
                    j = 0
                    switch = False
                else:
                    j = 1
                    switch = True

                ax.set_xlabel('')
    #            ax.set_xlim([0,1.e10])
    #            ax.set_xticks([0,2.e9,4.e9,6.e9,8.e9,10.e9])
    #            ax.set_xticklabels(["0","2","4","6","8","10"],fontsize=fontsize)
    #            ax.set_xticks([1.e9,3.e9,5.e9,7.e9,9.e9], minor=True)
    #            ax.set_xticklabels([], minor=True)

    #            ax.set_ylabel('Amplitude (Jy)',fontsize=fontsize)
    #            ax.set_ylim([0,1.2*flux])
    #            yticks_maj = np.array([0,.2,.4,.6,.8,1])*flux
    #            ax.set_yticks(yticks_maj)
    #            ax.set_yticklabels(["%0.2f"%fl for fl in yticks_maj],fontsize=fontsize)
    #            yticks_min = np.array([.1,.3,.5,.7,.9])*flux
    #            ax.set_yticks(yticks_min,minor=True)
    #            ax.set_yticklabels([], minor=True)

                if i==3:
                    print('saving pdf page %i' % page)
                    page += 1
                    pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
                    plt.close()
                    fig = plt.figure(3, figsize=(18, 28), dpi=200)
                    gs = gridspec.GridSpec(6,4, wspace=WSPACE, hspace=HSPACE)
                    i = 0
                    j = 0
                    switch = False
            print('saving pdf page %i' % page)
            page += 1
            pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
            plt.close()


################################################################################
        #plot the log closure amps
        if campplots:
            print("===========================================")
            print("plotting closure amplitudes")
            fig = plt.figure(3, figsize=(18, 28), dpi=200)
            plt.suptitle("Closure Amplitude Plots",y=.9,va='center',fontsize=int(1.2*fontsize))
            gs = gridspec.GridSpec(6,4, wspace=WSPACE, hspace=HSPACE)
            i = 0
            j = 0
            switch=False
            obs_all = [obs, obs_model]
            camps_model['sigmaca'] *= 0
            camps_all = [camps_obs, camps_model]
            for quad in uniqueclosure_quad:

                ax = plt.subplot(gs[2*i:2*(i+1), 2*j:2*(j+1)])
                ax = eh.plotting.comp_plots.plot_camp_obs_compare(obs_all,quad[0],quad[1],quad[2],quad[3],markersize=MARKERSIZE,
                                                                 ctype='logcamp',rangey=[-3,3],camps=camps_all,
                                                                 axis=ax,legend=False, clist=['k',SCOLORS[1]],
                                                                 ttype='nfft',show=False, ebar=ebar)
                if ax is None: continue
                if switch:
                    i += 1
                    j = 0
                    switch = False
                else:
                    j = 1
                    switch = True


                ax.set_xlabel('')
    #            ax.set_xlim([0,1.e10])
    #            ax.set_xticks([0,2.e9,4.e9,6.e9,8.e9,10.e9])
    #            ax.set_xticklabels(["0","2","4","6","8","10"],fontsize=fontsize)
    #            ax.set_xticks([1.e9,3.e9,5.e9,7.e9,9.e9], minor=True)
    #            ax.set_xticklabels([], minor=True)

    #            ax.set_ylabel('Amplitude (Jy)',fontsize=fontsize)
    #            ax.set_ylim([0,1.2*flux])
    #            yticks_maj = np.array([0,.2,.4,.6,.8,1])*flux
    #            ax.set_yticks(yticks_maj)
    #            ax.set_yticklabels(["%0.2f"%fl for fl in yticks_maj],fontsize=fontsize)
    #            yticks_min = np.array([.1,.3,.5,.7,.9])*flux
    #            ax.set_yticks(yticks_min,minor=True)
    #            ax.set_yticklabels([], minor=True)

                if i==3:
                    print('saving pdf page %i' % page)
                    page += 1
                    pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
                    plt.close()
                    fig = plt.figure(3, figsize=(18, 28), dpi=200)
                    gs = gridspec.GridSpec(6,4, wspace=WSPACE, hspace=HSPACE)
                    i = 0
                    j = 0
                    switch = False
            print('saving pdf page %i' % page)
            page += 1
            pdf.savefig(pad_inches=MARGINS, bbox_inches='tight')
            plt.close()


def display_img(im, beamparams=None, scale='linear',gamma=0.5,cbar_lims=False,
                    has_cbar=True,has_title=True,cfun='afmhot',
                    axis=False,show=False,fontsize=FONTSIZE):
    """display the figure on a given axis
       cannot use im.display because  it makes a new figure
    """

    interp = 'gaussian'

    if axis:
        ax = axis
    else:
        fig=plt.figure()
        ax = fig.add_subplot(1,1,1)

    imvec = np.array(im.imvec).reshape(-1)
    #flux unit is mJy/uas^2
    imvec = imvec * 1.e3
    fovfactor = im.xdim*im.psize*(1/eh.RADPERUAS)
    factor = (1./fovfactor)**2 / (1./im.xdim)**2 
    imvec = imvec * factor
    
    imarr = (imvec).reshape(im.ydim, im.xdim)
    unit = 'mJy/$\mu$ as$^2$'
    if scale=='log':
        if (imarr < 0.0).any():
            print('clipping values less than 0')
            imarr[imarr<0.0] = 0.0
        imarr = np.log(imarr + np.max(imarr)/dynamic_range)
        unit = 'log(' + unit +')'

    if scale=='gamma':
        if (imarr < 0.0).any():
            print('clipping values less than 0')
            imarr[imarr<0.0] = 0.0
        imarr = (imarr + np.max(imarr)/dynamic_range)**(gamma)
        unit = '(' + unit + ')^gamma'
               
    if cbar_lims:
        imarr[imarr>cbar_lims[1]] = cbar_lims[1]
        imarr[imarr<cbar_lims[0]] = cbar_lims[0]
              
    
    if cbar_lims:
        ax = ax.imshow(imarr, cmap=plt.get_cmap(cfun), interpolation=interp, 
                        vmin=cbar_lims[0], vmax=cbar_lims[1])
    else:
        ax = ax.imshow(imarr, cmap=plt.get_cmap(cfun), interpolation=interp)
    
    if has_cbar: 
        cbar=plt.colorbar(ax, fraction=0.046, pad=0.04,format='%1.2g')
        cbar.set_label(unit,fontsize=fontsize) 
        cbar.ax.xaxis.set_label_position('top') 
        cbar.ax.tick_params(labelsize=16) 
        if cbar_lims:
            plt.clim(cbar_lims[0],cbar_lims[1])


    if not(beamparams is None):
        beamparams = [beamparams[0], beamparams[1], beamparams[2], 
                      -.35*im.fovx(), -.35*im.fovy()]
        beamimage = im.copy()
        beamimage.imvec *= 0
        beamimage = beamimage.add_gauss(1, beamparams)
        halflevel = 0.5*np.max(beamimage.imvec)
        beamimarr = (beamimage.imvec).reshape(beamimage.ydim,beamimage.xdim)   
        plt.contour(beamimarr, levels=[halflevel], colors='w', linewidths=3) 
        ax = plt.gca()

    
    plt.axis('off')
    fov_uas = im.xdim * im.psize / eh.RADPERUAS # get the fov in uas
    roughfactor = 1./3. # make the bar about 1/3 the fov
    fov_scale = 40
    #fov_scale = int( math.ceil(fov_uas * roughfactor / 10.0 ) ) * 10 # round around 1/3 the fov to nearest 10
    start = im.xdim * roughfactor / 3.0 # select the start location 
    end = start + fov_scale/fov_uas * im.xdim # determine the end location based on the size of the bar
    plt.plot([start, end], [im.ydim-start, im.ydim-start], color="white", lw=1) # plot line
    plt.text(x=(start+end)/2.0, y=im.ydim-start-im.ydim/20, s=str(fov_scale) + " $\mu$as",
                color="white", ha="center", va="center", fontsize=int(1.2*fontsize),fontweight='bold')

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if show:
        plt.show(block=False)

    return ax

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("inputim",help='/path/to/image.fits')
    parser.add_argument("inputobs",help='/path/to/uvfits.uvfits')
    parser.add_argument("inputobs_uncal",help='/path/to/uvfits_uncalibrated.uvfits')

    parser.add_argument('--c', '-c',type=str,default=" ", help="comments for top of sheet")
    parser.add_argument('--o','-o', type=str,help="path/to/output",default='.')
    parser.add_argument('--systematic_noise', type=float, default=0, help="systematic noise to add on amplitudes")
    parser.add_argument('--systematic_cphase_noise', type=float, default=0,help="systematic noise to add on cphase")
    parser.add_argument('--fontsize', type=int, default=FONTSIZE,help="font size")
    parser.add_argument('--cfun', type=str, default='afmhot',help="image color function")
    parser.add_argument('--no_ebar', default=False,action='store_true',help="remove ebars from amp")
    parser.add_argument('--no_gains', default=False,action='store_true',help="remove gain plots")
    parser.add_argument('--no_cphase', default=False,action='store_true',help="remove closure phase plots")
    parser.add_argument('--no_camp', default=False,action='store_true',help="remove closure amp plots")

    opt = parser.parse_args()
    print("Generating Image Summary PDF")
    print("===========================================")
    im = eh.image.load_fits(opt.inputim)
    obs = eh.obsdata.load_uvfits(opt.inputobs)
    obs_uncal = eh.obsdata.load_uvfits(opt.inputobs_uncal)

    basename = os.path.splitext(os.path.basename(opt.inputim))[0]
    outdir = str(opt.o)
    if outdir[-1] == '/': outname = outdir + basename + '.pdf'
    else: outname = outdir +'/' + basename + '.pdf'

    if opt.no_ebar: ebar=False
    else: ebar=True
    if opt.no_gains: gainplots=False
    else: gainplots=True
    if opt.no_cphase: cphaseplots=False
    else: cphaseplots=True
    if opt.no_camp: campplots=False
    else: campplots=True

    main(im, obs, obs_uncal, basename, outname,  commentstr=opt.c, outdir=outdir,ebar=ebar,cfun=opt.cfun,
         sysnoise=opt.systematic_noise,syscnoise=opt.systematic_cphase_noise,fontsize=opt.fontsize,
         gainplots=gainplots,cphaseplots=cphaseplots,campplots=campplots)

