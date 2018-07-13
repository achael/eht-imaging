#imgsum.py
#Andrew Chael 07/12/2018
#produce an image summary plot for an image and uvfits file

from __future__ import print_function
import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import math
import os

#display parameters
FONTSIZE = 20
WSPACE=0.8
HSPACE=0.3
MARGINS=0.5

################################################################################3
################################################################################3
#im = eh.image.load_fits('./out.fits')
#obs = eh.obsdata.load_uvfits('./obs.uvfits')
#sys = 0.1
def main(im, obs, basename, commentstr="", outdir='.',ebar=True,cfun='afmhot',sysnoise=0,fontsize=FONTSIZE):

    flux = im.total_flux()
    #define the figure
    fig = plt.figure(1, figsize=(18, 28), dpi=200)
    plt.rc('font', family='serif')
    gs = gridspec.GridSpec(6,4, wspace=WSPACE, hspace=HSPACE)

    ################################################################################3
    #user comments


    if len(commentstr)>1:
        titlestr = basename+'.pdf\n'+str(commentstr)
    else:
        titlestr = basename+'.pdf'
    plt.suptitle(titlestr,y=.9,va='center',fontsize=int(1.2*fontsize))
    plt.rc('text', usetex=True)
    ################################################################################3
    #plot the image
    ax = plt.subplot(gs[0:2,0:2])
    ax = display_img(im,axis=ax, show=False,has_title=False,cfun=cfun,fontsize=fontsize)

    ################################################################################3
    #plot the vis amps
    ax = plt.subplot(gs[0:2,2:5])

    ax = eh.plotting.comp_plots.plotall_obs_im_compare([obs], im, 
                                                       'uvdist','amp', axis=ax,legend=False, 
                                                        ttype='nfft',show=False,ebar=ebar)

    #modify the labels
    #ax.xaxis.label.set_visible(False)
    #ax.yaxis.label.set_visible(False)
    ax.set_xlabel('G$\lambda$',fontsize=fontsize)
    ax.set_xlim([0,1.e10])
    ax.set_xticks([0,2.e9,4.e9,6.e9,8.e9,10.e9])
    ax.set_xticklabels(["0","2","4","6","8","10"],fontsize=fontsize)
    ax.set_xticks([1.e9,3.e9,5.e9,7.e9,9.e9], minor=True)
    ax.set_xticklabels([], minor=True)

    ax.set_ylabel('Jy',fontsize=fontsize)
    ax.set_ylim([0,1.2*flux])
    yticks_maj = np.array([0,.2,.4,.6,.8,1])*flux
    ax.set_yticks(yticks_maj)
    ax.set_yticklabels(["%0.2f"%fl for fl in yticks_maj],fontsize=fontsize)
    yticks_min = np.array([.1,.3,.5,.7,.9])*flux
    ax.set_yticks(yticks_min,minor=True)
    ax.set_yticklabels([], minor=True)

    ################################################################################3
    #display the overall chi2
    ax = plt.subplot(gs[2,0:2])
    #ax.axis('off')
    ax.set_yticks([])
    ax.set_xticks([])

    chi2vis = obs.chisq(im, dtype='vis', ttype='nfft', systematic_noise=sysnoise)
    chi2amp = obs.chisq(im, dtype='amp', ttype='nfft', systematic_noise=sysnoise)
    chi2cphase = obs.chisq(im, dtype='cphase', ttype='nfft', systematic_noise=sysnoise)
    chi2logcamp = obs.chisq(im, dtype='logcamp', ttype='nfft', systematic_noise=sysnoise)
    chi2camp = obs.chisq(im, dtype='camp', ttype='nfft', systematic_noise=sysnoise)

    print("chi^2 vis: ", chi2vis)
    print("chi^2 amp: ", chi2amp)
    print("chi^2 cphase: ", chi2cphase)
    print("chi^2 logcamp: ", chi2logcamp)
    print("chi^2 camp: ", chi2logcamp)

    fs =int(1.2*fontsize)
    ax.text(.1,.9,"Source:", fontsize=fs,
    ha='left',va='center',transform=ax.transAxes)
    ax.text(.1,.7,"MJD:" , fontsize=fs,
    ha='left',va='center',transform=ax.transAxes)
    ax.text(.1,.5,"FREQ:" , fontsize=fs,
    ha='left',va='center',transform=ax.transAxes)
    ax.text(.1,.3,"FOV:", fontsize=fs,
    ha='left',va='center',transform=ax.transAxes)
    ax.text(.1,.1,"FLUX:" , fontsize=fs,
    ha='left',va='center',transform=ax.transAxes)

    ax.text(.28,.9,"%s"%im.source, fontsize=fs,
    ha='left',va='center',transform=ax.transAxes)
    ax.text(.28,.7,"%i" % im.mjd, fontsize=fs,
    ha='left',va='center',transform=ax.transAxes)
    ax.text(.28,.5,"%0.0f GHz" % (im.rf/1.e9), fontsize=fs,
    ha='left',va='center',transform=ax.transAxes)
    ax.text(.28,.3,"%0.1f $\mu$as" % (im.fovx()/eh.RADPERUAS), fontsize=fs,
    ha='left',va='center',transform=ax.transAxes)
    ax.text(.28,.1,"%0.2f Jy" % flux, fontsize=fs,
    ha='left',va='center',transform=ax.transAxes)

    ax.text(.6,.9,"$\chi^2_{vis}$" , fontsize=fs,
    ha='left',va='center',transform=ax.transAxes)
    ax.text(.6,.7,"$\chi^2_{amp}$" , fontsize=fs,
    ha='left',va='center',transform=ax.transAxes)
    ax.text(.6,.5,"$\chi^2_{cphase}$" , fontsize=fs,
    ha='left',va='center',transform=ax.transAxes)
    ax.text(.6,.3,"$\chi^2_{log camp}$" , fontsize=fs,
    ha='left',va='center',transform=ax.transAxes)
    ax.text(.6,.1,"$\chi^2_{camp}$" , fontsize=fs,
    ha='left',va='center',transform=ax.transAxes)

    ax.text(.78,.9,"%0.2f" % chi2vis, fontsize=fs,
    ha='left',va='center',transform=ax.transAxes)
    ax.text(.78,.7,"%0.2f" % chi2amp, fontsize=fs,
    ha='left',va='center',transform=ax.transAxes)
    ax.text(.78,.5,"%0.2f" % chi2cphase, fontsize=fs,
    ha='left',va='center',transform=ax.transAxes)
    ax.text(.78,.3,"%0.2f" % chi2logcamp, fontsize=fs,
    ha='left',va='center',transform=ax.transAxes)
    ax.text(.78,.1,"%0.2f" % chi2camp, fontsize=fs,
    ha='left',va='center',transform=ax.transAxes)


    ################################################################################
    #display the closure  phase chi2
    ax = plt.subplot(gs[3:6,0:2])
    #ax.axis('off')
    ax.set_yticks([])
    ax.set_xticks([])

    # get closure triangle combinations
    # ANDREW -- hacky, fix this!
    cp = obs.c_phases(mode="all", count="min")
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
            chisq_tri= np.sum((1.0 - np.cos(cphases_obs_tri['cphase']*eh.DEGREE-cphases_model_tri['cphase']*eh.DEGREE))/
                              ((cphases_obs_tri['sigmacp']*eh.DEGREE)**2))
            chisq_tri *= (2.0/len(cphases_obs_tri))
            npts = len(cphases_obs_tri)
            data =  [uniqueclosure_tri[c][0], uniqueclosure_tri[c][1], uniqueclosure_tri[c][2],npts,chisq_tri]
            cphase_chisq_data.append(data)

    #sort by decreasing chi^2
    idx = np.argsort([data[-1] for data in cphase_chisq_data])
    idx = list(reversed(idx))

    chisqtab=r"\begin{tabular}{ l|l|l } \hline Triangle & $N_{cphase}$ & $\chi^2_{tri}$ \\ \hline \hline"
    first=True
    for i in range(len(cphase_chisq_data)):
        if i>30:break
        data = cphase_chisq_data[idx[i]]
        tristr = r"%s-%s-%s" % (data[0],data[1],data[2])
        nstr = r"%i" % data[3]
        chisqstr = r"%0.1f" % data[4]
        if first:
            chisqtab += r" " + tristr + " & " + nstr + " & " + chisqstr 
            first=False
        else:
            chisqtab += r" \\" + tristr + " & " + nstr + " & " + chisqstr 
    chisqtab += r" \end{tabular}"

    ax.text(0.5,.975,chisqtab,ha="center",va="top",transform=ax.transAxes,size=fontsize)



    ################################################################################
    #display the log closure amplitude chi2
    ax = plt.subplot(gs[2:6,2::])
    #ax.axis('off')
    ax.set_yticks([])
    ax.set_xticks([])

    # get closure triangle combinations
    # ANDREW -- hacky, fix this!
    debias=True
    cp = obs.c_amplitudes(mode="all", count="min",ctype='logcamp',debias=debias)
    allquads = [(str(cpp['t1']),str(cpp['t2']),str(cpp['t3']),str(cpp['t4'])) for cpp in cp]
    uniqueclosure_quad = []
    for quad in allquads:
        if quad not in uniqueclosure_quad: 
            uniqueclosure_quad.append(quad)
          
    # generate data
    obs_model = im.observe_same(obs, add_th_noise=False, ttype='nfft')

    camps_obs = obs.c_amplitudes(mode='all', count='max', ctype='logcamp', debias=debias)

    camps_model = obs_model.c_amplitudes(mode='all', count='max', ctype='logcamp', debias=debias)


    #generate chi^2 -- NO SYSTEMATIC NOISES
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
            chisq_quad /= (len(camps_obs_quad))
            npts = len(camps_obs_quad)

            data =  (uniqueclosure_quad[c][0], uniqueclosure_quad[c][1], uniqueclosure_quad[c][2],uniqueclosure_quad[c][3],
                     npts,chisq_quad)
            camp_chisq_data.append(data)

    #sort by decreasing chi^2
    idx = np.argsort([data[-1] for data in camp_chisq_data])
    idx = list(reversed(idx))


    chisqtab=r"\begin{tabular}{ l|l|l } \hline Quadrangle & $N_{logcamp}$ & $\chi^2_{quad}$ \\ \hline \hline"
    for i in range(len(camp_chisq_data)):
        if i>45:break
        data = camp_chisq_data[idx[i]]
        tristr = r"%s-%s-%s-%s" % (data[0],data[1],data[2],data[3])
        nstr = r"%i" % data[4]
        chisqstr = r"%0.1f" % data[5]
        if i==0:
            chisqtab += r" " + tristr + " & " + nstr + " & " + chisqstr 
        else:
            chisqtab += r" \\" + tristr + " & " + nstr + " & " + chisqstr 

    chisqtab += r" \end{tabular}"

    ax.text(0.5,.975,chisqtab,ha="center",va="top",transform=ax.transAxes,size=fontsize)



    #save the plot
    print()
    #plt.tight_layout()
    #plt.subplots_adjust(wspace=1,hspace=1)
    plt.savefig(outname, pad_inches=MARGINS,bbox_inches='tight')

def display_img(im, scale='linear',gamma=0.5,cbar_lims=False,
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
    parser.add_argument("inputobs",help='/path/to/uvfits.fits')
    parser.add_argument('--c', '-c',type=str,help="comments",default=" ")
    parser.add_argument('--o','-o', type=str,help="path/to/output",default='.')
    parser.add_argument('--systematic_noise', type=float, default=0)
    parser.add_argument('--fontsize', type=int, default=FONTSIZE)
    parser.add_argument('--cfun', type=str, default='afmhot')
    parser.add_argument('--no_ebar', default=False,action='store_true',help="remove ebars from amp")

    opt = parser.parse_args()

    print()
    print("loading image: ",opt.inputim,"\n")
    im = eh.image.load_fits(opt.inputim)
    print("loading observation: ",opt.inputobs,"\n")
    obs = eh.obsdata.load_uvfits(opt.inputobs)

    basename = os.path.splitext(os.path.basename(opt.inputim))[0]
    outdir = str(opt.o)
    if outdir[-1] == '/': outname = outdir + basename + '.pdf'
    else: outname = outdir +'/' + basename + '.pdf'

    if opt.no_ebar: ebar=False
    else: ebar=True

    main(im, obs, basename, commentstr=opt.c, outdir=outdir,ebar=ebar,cfun=opt.cfun,sysnoise=opt.systematic_noise,fontsize=opt.fontsize)

