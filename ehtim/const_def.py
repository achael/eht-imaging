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

import sys
import matplotlib as mpl

mpl.rc('font',**{'family':'serif','size':12})

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
DTERMPDEF = 0.05 # Default rms of D-term errors

# Sgr A* Kernel Values (Bower et al., in uas/cm^2)
FWHM_MAJ = 1.309 * 1000 # in uas
FWHM_MIN = 0.64 * 1000
POS_ANG = 78 # in degree, E of N

# FFT & NFFT options
NFFT_KERSIZE_DEFAULT = 20
GRIDDER_P_RAD_DEFAULT = 2
GRIDDER_CONV_FUNC_DEFAULT = 'gaussian'
FFT_PAD_DEFAULT = 2
FFT_INTERP_DEFAULT = 3

# Observation recarray datatypes
DTARR = [('site', 'U32'), ('x','f8'), ('y','f8'), ('z','f8'),
         ('sefdr','f8'),('sefdl','f8'),('dr','c16'),('dl','c16'),
         ('fr_par','f8'),('fr_elev','f8'),('fr_off','f8')]

DTPOL_STOKES = [('time','f8'),('tint','f8'),
                 ('t1','U32'),('t2','U32'),
                 ('tau1','f8'),('tau2','f8'),
                 ('u','f8'),('v','f8'),
                 ('vis','c16'),('qvis','c16'),('uvis','c16'),('vvis','c16'),
                 ('sigma','f8'),('qsigma','f8'),('usigma','f8'),('vsigma','f8')]

DTPOL_CIRC = [('time','f8'),('tint','f8'),
                 ('t1','U32'),('t2','U32'),
                 ('tau1','f8'),('tau2','f8'),
                 ('u','f8'),('v','f8'),
                 ('rrvis','c16'),('llvis','c16'),('rlvis','c16'),('lrvis','c16'),
                 ('rrsigma','f8'),('llsigma','f8'),('rlsigma','f8'),('lrsigma','f8')]

DTAMP = [('time','f8'),('tint','f8'),
         ('t1','U32'),('t2','U32'),
         ('u','f8'),('v','f8'),
         ('amp','f8'),('sigma','f8')]

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

POLDICT_STOKES = {'vis1': 'vis', 'vis2': 'qvis', 'vis3': 'uvis', 'vis4': 'vvis', 
                  'sigma1': 'sigma', 'sigma2': 'qsigma', 'sigma3': 'usigma', 'sigma4': 'vsigma'} 
POLDICT_CIRC = {'vis1': 'rrvis', 'vis2': 'llvis', 'vis3': 'rlvis', 'vis4': 'lrvis', 
                  'sigma1': 'rrsigma', 'sigma2': 'llsigma', 'sigma3': 'rlsigma', 'sigma4': 'lrsigma'} 
vis_poldict={'I':'vis','Q':'qvis','U':'uvis','V':'vvis','RR':'rrvis','LL':'llvis','RL':'rlvis','LR':'lrvis'}
amp_poldict={'I':'amp','Q':'qamp','U':'uamp','V':'vamp','RR':'rramp','LL':'llamp','RL':'rlamp','LR':'lramp'}
sig_poldict={'I':'sigma','Q':'qsigma','U':'usigma','V':'vsigma','RR':'rrsigma','LL':'llsigma','RL':'rlsigma','LR':'lrsigma'}

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
FIELDS_AMPS = ["amp", "qamp", "uamp","vamp","pamp","mamp","rramp","llamp","rlamp","lramp"]
FIELDS_SIGS = ["sigma","qsigma","usigma","vsigma","psigma","msigma","rrsigma","llsigma","rlsigma","lrsigma"]
FIELDS_PHASE = ["phase", "qphase", "uphase", "vphase","pphase", "mphase","rrphase","llphase","lrphase","rlphase"]
FIELDS_SIGPHASE = ["sigma_phase","qsigma_phase","usigma_phase","vsigma_phase","psigma_phase","msigma_phase",
                   "rrsigma_phase","llsigma_phase","rlsigma_phase","lrsigma_phase"]
FIELDS_SNRS = ["snr", "qsnr", "usnr", "vsnr", "psnr", "msnr","rrsnr","llsnr","rlsnr","lrsnr"]

#plotting
MARKERSIZE = 3

FIELD_LABELS = {'time':'Time','time_utc':'Time (UTC)','time_gmst':'Time (GMST)',
          'tint':'Integration Time','u':r'$u$','v':r'$v$','uvdist':r'$u-v$ Distance',
          't1':'Site 1','t2':'Site 2','tau1':r'$\tau_1$','tau2':r'$\tau_2$',
          'el1':r'Elevation Angle$_1$','el2':r'Elevation Angle$_2$',
          'hr_ang1':r'Hour Angle$_1$','hr_ang2':r'Hour Angle$_2$', 
          'par_ang1':r'Parallactic Angle$_1$','par_ang2':r'Parallactic Angle$_2$',
          'vis':'Visibility','amp':'Amplitude','phase':'Phase','snr':'SNR',
          'qvis':'Q-Visibility','qamp':'Q-Amplitude','qphase':'Q-Phase','qsnr':'Q-SNR',
          'uvis':'U-Visibility','uamp':'U-Amplitude','uphase':'U-Phase','usnr':'U-SNR',
          'vvis':'V-Visibility','vamp':'V-Amplitude','vphase':'V-Phase','vsnr':'V-SNR',
          'sigma':r'$\sigma$','qsigma':r'$\sigma_{Q}$','usigma':r'$\sigma_{U}$',
          'vsigma':r'$\sigma_{V}$',
          'sigma_phase':r'$\sigma_{phase}$','qsigma_phase':r'$\sigma_{Q phase}$',
          'usigma_phase':r'$\sigma_{U phase}$','vsigma_phase':r'$\sigma_{V phase}$',
          'psigma_phase':r'$\sigma_{P phase}$','msigma_phase':r'$\sigma_{m phase}$',
          'pvis':r'P-Visibility','pamp':r'P-Amplitude','pphase':'P-Phase','psnr':'P-SNR',
          'mvis':r'm-Visibility','mamp':r'm-Amplitude','mphase':'m-Phase','msnr':'m-SNR',
          'rrvis':r'RR-Visibility','rramp':r'RR-Amplitude','rrphase':'RR-Phase','rrsnr':'RR-SNR',
          'rrsigma':r'$\sigma_{RR}$','rrsigma_phase':r'$\sigma_{RR phase}$',
          'llvis':r'LL-Visibility','llamp':r'LL-Amplitude','llphase':'LL-Phase','llsnr':'LL-SNR',
          'llsigma':r'$\sigma_{LL}$','llsigma_phase':r'$\sigma_{LL phase}$',
          'rlvis':r'RL-Visibility','rlamp':r'RL-Amplitude','rlphase':'RL-Phase','rlsnr':'RL-SNR',
          'rlsigma':r'$\sigma_{RL}$','rlsigma_phase':r'$\sigma_{RL phase}$',
          'lrvis':r'LR-Visibility','lramp':r'LR-Amplitude','lrphase':'LR-Phase','lrsnr':'LR-SNR',
          'lrsigma':r'$\sigma_{LR}$','lrsigma_phase':r'$\sigma_{LR phase}$'}

#Seaborn Colors from Maciek
#['dodgerblue','tomato','blueviolet','olivedrab','orange','saddlebrown','mediumblue','red','cyan','magenta','darkgreen','tan','k']
SCOLORS = [(0.11764705882352941, 0.5647058823529412, 1.0),
             (1.0, 0.38823529411764707, 0.2784313725490196),
             (0.5411764705882353, 0.16862745098039217, 0.8862745098039215),
             (0.4196078431372549, 0.5568627450980392, 0.13725490196078433),
             (1.0, 0.6470588235294118, 0.0),
             (0.5450980392156862, 0.27058823529411763, 0.07450980392156863),
             (0.0, 0.0, 0.803921568627451),
             (1.0, 0.0, 0.0),
             (0.0, 1.0, 1.0),
             (1.0, 0.0, 1.0),
             (0.0, 0.39215686274509803, 0.0),
             (0.8235294117647058, 0.7058823529411765, 0.5490196078431373),
             (0.0, 0.0, 0.0)]


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


def prog_msg(nscan, totscans, msgtype='bar',nscan_last=0):
    """print a progress method for calibration
    """
    complete_percent_last = int(100*float(nscan_last)/float(totscans))
    complete_percent = int(100*float(nscan)/float(totscans))
    ndigit = str(len(str(totscans)))

    if msgtype=='bar':
        bar_width = 30
        progress = int(bar_width * complete_percent/float(100))
        barparams = (nscan, totscans, ("-"*progress) + (" " * (bar_width-progress)),complete_percent)

        printstr = "\rScan %0"+ndigit+"i/%i : [%s]%i%%"
        sys.stdout.write(printstr % barparams)
        sys.stdout.flush()

    elif msgtype=='bar2':
        bar_width = 30
        progress = int(bar_width * complete_percent/float(100))
        barparams = (nscan, totscans, ("/"*progress) + (" " * (bar_width-progress)),complete_percent)

        printstr = "\rScan %0"+ndigit+"i/%i : [%s]%i%%"
        sys.stdout.write(printstr % barparams)
        sys.stdout.flush()

    elif msgtype=='casa':
        message_list = [".",".",".","10",".",".",".","20",".",".",".","30",".",".",".","40",
                        ".",".",".","50",".",".",".","60",".",".",".","70",".",".",".","80",
                        ".",".",".","90",".",".",".","DONE"]
        bar_width = len(message_list)
        progress = int(bar_width * complete_percent/float(100))
        message = ''.join(message_list[:progress])

        barparams = (nscan, totscans, message)
        printstr = "\rScan %0"+ndigit+"i/%i : %s"
        sys.stdout.write(printstr % barparams)
        sys.stdout.flush()

    elif msgtype=='itcrowd':
        message_list = ["0","1","1","8"," ","9","9","9"," ","8","8","1","9","9"," ",
                        "9","1","1","9"," ","7","2","5"," "," "," ","3"]
        bar_width = len(message_list)
        progress = int(bar_width * complete_percent/float(100))
        message = ''.join(message_list[:progress])
        if complete_percent<100:
            message += "." 
            message += " "*(bar_width-progress-1)

        barparams = (nscan, totscans, message)

        printstr= "\rScan %0"+ndigit+"i/%i : [%s]"
        sys.stdout.write(printstr % barparams)
        sys.stdout.flush()


    elif msgtype=='bh':
        message_all = BHIMAGE
        bar_width = len(message_all)
        progress = int(np.floor(bar_width * complete_percent/float(100)))-1
        progress_last = int(np.floor(bar_width * complete_percent_last/float(100)))-1
        if progress>progress_last:
            for i in range(progress_last+1,progress+1):
                message_line = ''.join(message_all[i])
                message_line = '%03i'%int(complete_percent) + message_line
                print(message_line)


    elif msgtype=='eht':
        message_all = EHTIMAGE
        bar_width = len(message_all)
        progress = int(np.floor(bar_width * complete_percent/float(100)))-1
        progress_last = int(np.floor(bar_width * complete_percent_last/float(100)))-1
        if progress>progress_last:
            for i in range(progress_last+1,progress+1):
                message_line = ''.join(message_all[i])
                message_line = '%03i'%int(complete_percent) + message_line
                print(message_line)


    elif msgtype=='gitstash':
        message_all = GITSTASHIMAGE
        bar_width = len(message_all)
        progress = int(np.floor(bar_width * complete_percent/float(100)))-1
        progress_last = int(np.floor(bar_width * complete_percent_last/float(100)))-1
        if progress>progress_last:
            for i in range(progress_last+1,progress+1):
                message_line = ''.join(message_all[i])
                message_line = '%03i'%int(complete_percent) + message_line
                print(message_line)

    elif msgtype=='dots':
        sys.stdout.write('.')
        sys.stdout.flush()

    else:# msgtype=='default':
        barparams = (nscan, totscans, complete_percent)
        printstr = "\rScan %0"+ndigit+"i/%i : %i%% done . . ."
        sys.stdout.write(printstr % barparams)
        sys.stdout.flush()


BHIMAGE = [
'                                                            ..                                                                                                 ',
 '                                                        .....   .                                                                                              ',
 '                                                     ...........         ....                                                                                  ',
 '                                                  ............      ........................                                                                   ',
 '                                                .........................,,******,,.....                                                                       ',
 '                                              .......,,,,..........,,**/(((/*,,...                                                                             ',
 '                                           .....,,,,,**,.......,**/(#%%#/,...                                                                                  ',
 '                                         .....,,,,,,**,.....,*/(#%&&%/,.                                                                                       ',
 '                                       ......,**,***/*,,,,*/(#%@@*.                                                                                            ',
 '                                     ..,....,****////***/((%&@@@#,                                                                                             ',
 '                                   ..,,,...****///////((#%&@@@%*.                                                                                              ',
 '                                  .,**,,.,*////((((((##%&@@@&/.                                                                                                ',
 '                                 .,//*,,,*/(((((((##%%&@@@@%,              ,,,,,.             .,..                                                             ',
 '                                 ,/(//***/(###((##%%%&@@@@#,      .   ,*/,.                         ...                                                        ',
 '                                .*(((////(#%%###%%&&@@@@@#,      ..*(*.                                 ..                                                     ',
 '                                ./((((//(#%&%%%%&&@@@@@@%*.    .*#/.                                       .                                                   ',
 '                                ,(#((#(((#&&&&&&@@@@@@@&(..  .(%*                                            ..                                                ',
 '                ...             ,(#####((#&&@@@@@@@@@@@%/..,##,                                                ..                                              ',
 '                ....            ,(###%%#(#%&@@@@@@@@@@@%/,#&*                                                    ..                                            ',
 '                .....           .(###%%%##%&@@@@@@@@@@@&%,                                                         .                                           ',
 '                ......          ./#%%%%%%##&@@@@@@@@@@@@@%,                                                         .                                          ',
 '                ........        .*%%%%%&%##%@@@@@@@@@@@@@(,.                                                         .                                         ',
 '                 ..,,,...  ..   .,#%%%%%&%%%&@@@@@@@@@@@&%/.                                                          .                                        ',
 '                 ..,,,,...  ... ../#%%%%&%%%&@@@@@@@@@@@@*.               @@   @     @        #                                                                ',
 '                  ..,,,,... ......*(##%%&&%%&&@@@@@@@@@@@&%(,.           @  @  @     @        #   ### ##               .                                       ',
 '                  ..,**,,.......,.,/###%%&%#%&@@@@@@@@@@@&%#/,..         @@@@  @@@  @@@  %%%  #   #  #  #              ..                                      ',
 '                   ..,**,,.......,,*(((#%&&%#%&@@@@@@@@&&&&%(*...        @     @ @   @        #   #  #  #              ..                                      ',
 '                    .,,*,,,,.......,/(((#%&&%##&@@&@@@&&%%%##(*..         @@   @ @   @@       ##  #  #  #               .                                      ',
 '                    ..,,,,,,,.......,/((((%&&%#(%&&&@@%%%###((/*..                                                      .                                      ',
 '                     ...,,.,,,....,,,*/((/(#%&(#%&@@%%%#(///**,..                  v 0.1 , 2019                         .                                      ',
 '                      ........,.....,,*////((%&&%###&@&%###(/*,,,,...                                                  ..                                      ',
 '                        .............,,**///((#%&&%%#@&%%#((((/*.......                                                ,.                                      ',
 '                         ............,,,**////((#%%&%@@#####((/***,....                                               .,.                                      ',
 '                                 ....,,*****////((#%%&@(((((((/**,,,.....                                             ,,                                       ',
 '                       .           ...,**/****/////(#%&@&%##((((((//**,.......          .....                        .*,                                       ',
 '                        ...          ..,*/////***////(#%&%%%###((((///****,,,,.......,,.......                      .,,.                                       ',
 '                         ....          .,*/((//****///((#%#######((((///*****,,,,,,...........                      ,*.                                        ',
 '                           .....        ..,/(((//****///((#(///((((((///******,,,,,...............                 .*,                                         ',
 '                              ...       ...,*/(#((///////((#(/*****///(((((////***,,,.........                    .*,                                          ',
 '                                        .....,*(#####(((((##%%##((((((////****,,,.........                       ,*,                                           ',
 '                                         .....,,*/(##%%#########(///*******,,,,...........                     .,*.                                            ',
 '                                            .......,,*/(((#########(//**,,,,,,,**,,,...                      .,*,.                                             ',
 '                                                 ..........,,,*****/********,,,....                        .,*,.                                               ',
 '                                                                  .   ....                              ..**.                                                  ',
 '                                                                               .      ..             ..,,,.                                                    ',
 '                                                                         .                      ........                                                       ',
 '                                                                               ...................                                                             ',
 '                                                                                                                                                        ',
 '']

EHTIMAGE = [
'                                   `..----..`                                                                                                                         ',     
'                           `-/oyhmNNNNMMMMNNNNmhs+:-`                                                                                                                 ',    
'                      `.+ymNMMMMMMMMMMMMMMMMMMMMMMMMNds/.                                                                                                             ',     
'                   `:ymNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNdo-`                                                                                                         ',     
'                `-yNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNmo.                                                                                                       ',     
'              `/dNMMMMMMMMMMMMMMmdysoo+++dMMMMMMMMMMMMMMMMMMMMNh:                      ://////                                                                        ',     
'            `+mMMMMMMMMMMMNmy+-```       sMMMMMNshNMMMMMMMMMMMMMNh-                    mNyoooo```     ``   ```     `  ```    `ds``                                    ',     
'           :dMMMMMMMMMMMho-`  .:-  :sd-  sMMMMNs.-yMMMMMMMMMMMMMMMNy.                  mN:     sd:   /ds -ydyhh+` .dyyhhdh: :dMmyy.                                   ',     
'         `yNMMMMMMMMMmo-` `/ymh:`-hNMM-  sMMMMMMmNMMMMMMNmMMMMMMMMMMm/                 mMmdddh -mm. .Nm.-Nm-..oMs -Mm:``:Mh``oMy``                                    ',     
'        .hMMMMMMMMMNo.  :ymMNo` oNMMMM-  sMMMMMMMMMMMMMm/`oNMMMMMMMMMNs`               mM/...`  /My`yM/ +Mmssssh+ -Md   .Md` +Ms                                      ',     
'       -mMMMMMMMMMs.  /hNMMN:  sMMMMMM-  sMMMMMMMMMNMMMNh+mMMMMMMMMMMMMy`              mM/....   sNsMo  -mm/```-` -Md   .Md` +Mh``                                    ',     
'      -mMMMMMMMMN/  -hNMMMN:  oMMMMMMM:  sMMMMMMMMy-sNMMMMMMMMMMMMMMMMMMy              ydhyyyy`  `yhy    .oyhhyy- .ds   .hy` `sdhy-                                   ',     
'     `mMMMMMMMMm-   -:::::-   -:::::::`  sMMMMMMMMmodMMMMMMMMMdMMMMMMMMMMo              ``````     `        ```    `     `     ```                                    ',     
'     oMMMMMMMMm-  -::::::.  `-::::::::`  sMNhNMMMMMMMMMMMMMMmo`/mMMMMMMMMM:                                                                                           ',     
'    -NMMMMMMMM:  +NMMMMMM:  /NMMMMMMMM-  sMd -+hNMMMMMMMMMMMMm+dNMMMMMMMMMh`                                                                                          ',     
'    yMMMMMMMMy  .NMMMMMMd`  hMMMMMMMMM-  sMNo-``.+hNMMMMMMMMMMMMMMMMMMMMMMN:           ..     `..                   `::                                               ',     
'   `mMMMMMMMN-  yMMMMMMMs   NMMMMMMMMM-  sMMMNho-` .+hNMMMMMMMMMMMMMMMMMMMMy           dm:    :mh                   `sy`                                              ',     
'   -NMMMMMMMm  .mMMMMMMM+  `NMMMMMMMMM:  sMMMMMMNds:` .+hNMMMNhsshMMMMMMMMMm           mN:    /Mm   -osss+-  .o::os:`os``ooooss/ `:osss/`  :s:+sss/`                  ',     
'   /MMMMMMMMh  `////////.   //////////`  sMm////////.    .+Mm:   `+MMMMMMMMN`          mMhsssshMm  +Nh/-/dm: -MNy+/..mM` --:oMd- hNs:-+Nd. oMm+-:hMo                  ',     
'   /MMMMMMMMh   --------`   ----------`  sMm--------.    `:Mm.    /NMMMMMMMN`          mMo////sMm `dM.   /Md -Mm`   .mM`  `oms. .Md    yM/ oMs   /My                  ',     
'   -NMMMMMMMm  .mNNNNNNN+  `mNNNNNNNNN-  sMMNNNNNdy/. `:sdMMMms++sNMMMMMMMMN           mN:    /Mm  yM/` `oMy -Mm`   .mM` -hm/   `Nm-  .hM: oMo   /My                  ',     
'   `mMMMMMMMN- `hMMMMMMMs   NMMMMMMMMM-  sMMMMmy/. `:sdMMMMMMMMMMMMMMMMMMMMh           dm:    /md  .sdyyyds` -md`   .dm`.mNhsss+ :hdyyhd+  +N+   /Ns                  ',     
'    hMMMMMMMMs  :MMMMMMMh`  dMMMMMMMMM-  sMNy/. `:smMMMMMMMMMMMMMMMMMMMMMMM/           ..`    `..    `---`    ..     ..  .......   .--.`   `.`   `.`                  ',     
'    :NMMMMMMMM-  oMMMMMMM:  /MMMMMMMMM:  sMd `:smMMMMMMMMMMMMNymMMMMMMMMMMd`                                                                                          ',     
'    `yMMMMMMMMd. `///////-  `/////////`  sMmsmMMMMMMMMMMMMMMm+ /mMMMMMMMMM/                                                                                           ',     
'     .NMMMMMMMMd.   .......   ........`  sMMMMMMMMNyNMMMMMMMMNyNMMMMMMMMMy`                             `--                                                           ',     
'      :NMMMMMMMMm-  :dNNNNm-  sNNNNNNN-  sMMMMMMMMs`+NMMMMMMMMMMMMMMMMMMd`           `yyyhhhyys`        .Nd                                                           ',     
'       /NMMMMMMMMNo` `omMMMm- `hMMMMMM-  sMMMMMMMMMmMMMMmsNMMMMMMMMMMMMd.             ...hMo... .-/:-`  .Nm   `-:/-.   .-//:-   `-:/:.  `-:/:.`   -..:/:.    .-//-.   ',     
'        :mMMMMMMMMMm/` `+dNMm/``yMMMMM-  sMMMMMMMMMMMMMd:`+NMMMMMMMMMMy`                 yM+  `omy++dm: .Nm  :dd++yNo -mN++oy` /mds+s+ :mms+ymh. .MmyoodN+  /mh  hm/  ',     
'         .hMMMMMMMMMMd+. `-odms.`/mMMM-  sMMMMMMNNMMMMMMNhMMMMMMMMMMNo`                  yM+  /MN+++sNh .Nm `mMo+++Nm..dmy+-` -mM`    `mM-   +Ms .Md`  .mM`.NNo++oNN  ',     
'          `+mMMMMMMMMMMNy:.  `:+: `+hm-  sMMMMMd-/mMMMMMMMMMMMMMMMMd:                    yM+  /Mm:::::- .Nm `mN/:::::`  ./sNm--NM`    `mM-   /Ms .Md`  .dM..NN/       ',     
'            .sNMMMMMMMMMMMNho:```    `   sMMMMMm/omMMMMMMMMMMMMMMN/`                     yM+  `sNy+//o: .Nd  /md+//++ -o///dN: oNd+/++ :mdo/omd. .MNy/+hNo  +Nh+//o:  ',     
'              .smMMMMMMMMMMMMMMmhyo+/:---hMMMMMMMMMMMMMMMMMMMMMd+`                       -/.    .:+++:` `/:   `:+++:. `:+++:.   `:++/.  `-++/-   .Mh-/+/.    `:+++:`  ',     
'               `+dNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNh:`                                                                                  .Mh                  ',     
'                  -odNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNh/.                                                                                     `+:                  ',     
'                     ./sdNMMMMMMMMMMMMMMMMMMMMMMMMMMNho:`                                                                                                             ',    
'                         `-/symNNMMMMMMMMMMMNNNdyo/.`                                                                                                                 ',   
'                                .----::::---.`                                                                                                                        ']   
                                                                                                                                                                          

                                                                                                                                                                               
                                                                                                                                                                               












































































GITSTASHIMAGE = [
'                                                                   (((((/(((((((####(((////////////((#%%%%%##                                                  ',
'                                                              ,#(((########((((((((((((((//////////((((((((((((,                                               ',
'                                                          /###((#(#####(((///////(((((((((///////////////////(/((#,                                            ',
'                                              .#&%%##(((((####((///////*********//////((((#######((((////////(((((////,.                                       ',
'                                          *##(###((((//(((/////*//////*********//////(((((((######((((/******/((/(((//***////                                  ',
'                                         ////(####(((/////*******///*****///**//(//((((((///(((((####(///*****////////*****///                                 ',
'                                        ///(##((////////////////////****//////((((((#((///**//(((###%##(////////****/******////*                               ',
'                                      ((((##(/***////*///((///////////////////////((((///****////((((#((///*/(((////(/*,***//////                              ',
'                                    /(#((((//*/////////((((///////////**/////////////********/**//((((///////(((((((((/*****/(((//.                            ',
'                                  ///((((////////**/(((((((((///((///*****/**//////**,,,,,*******/////////////((#(((((////////((((//,                          ',
'                                 *//////**/(///////(((#####((((/////****////*//**,,,.....,,**********//////////((((((((((((((///##((%#                         ',
'                                ////*****/((/////((#######((///****/******////***,,........,**,,***,,***///////(((((#####((((((((#(#%&&                        ',
'                               ((((*/////(((////((########((/*****************,,,....    ..,**,,,*,,,,,,******/((((((((##(((((((((((#%&&*                      ',
'                              (#((///////(((///(((#####((((/**,,,**********,,,,...       ...,,,,,,,,,,,,,,***,*//////(((###((((((((/((#&&/                     ',
'                             #%((//////**////(####%##((((/**,,,,,,,,,,,,,,,.....         ....,,..,,......,,,,,*********/((##((((((//*//(##                     ',
'                            *%#//////****//((###%%%#((////*,,,,,,,.....,,,.....            ...............,,,,,,,,,,,,,,*/(((###((//////(/#.                   ',
'                           #%#(///////////(###%%%##(/////**,,.................              ................,,,,,,,,,,.,,*/((###((//(//////#%                  ',
'                          %##(/((//((///((###%%%##/**,,,,,,..... ..........                 ....................,,.......,,*/(####((((##(/*(%&                 ',
'                         ###((((((((((((##%%####(/*,,.........                           ...................................,*/(###(##%%#(//(&*                ',
'                        %##((######(#####%%###((/*,..........                            .....................................,*(((####%##//(&&                ',
'                       *##(((###########%%##((//*,......                                 .................................... ..**(#%%##((/*/%%.               ',
'                      .%#(/(#%%####%%#####(((/**,........                                .........................................*/##(////*/(#*               ',
'                      &%(((#%%%%%%%%#######(//,,..........                                ....................................   ..,/((///*///(%               ',
'                      &%#(##%&&%%%%########(/*,,..........                               ...........................................,/////*///(#%              ',
'                     *%%###%&&&&&%%###%%%#(//*,,............    ...                        .................         ...............,*////////(##              ',
'                     &%#((#%&&&&&%%#%%%%%#(/**,,....................                     ................              ..............*//(((///(##              ',
'                    .&%###%%&&&&&%%#%%%%#((/**,,....................                    ................           ....   ...........*/((((///(##%             ',
'                   .&%####%%&&&&&%%#%%%%#((/**,,,,,.................            . .....................     ...........    ..........*/((((////(#%             ',
'                   /&%###%&&&&&&&&%%%%%#((///***,,,.................       ................................................  .....,..*/(((((///(##             ',
'                   *%%##%%&&&@&&&%%%%%%##(((//**,,,,...............      .........,,,,,........,,,........................... .......,/(((#(((((##             ',
'                   *%%##%%&&&&&&&%%%%%####((//**,,,,............      .........,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,..........  ......,/((###((((##             ',
'                   /&&%%%%&&&&&&&%%%%%#####((//**,,,........................,,,,**********,,,,,,,,,,,,,,,,*******,,,,,,.............,,*((###(((###             ',
'                   /&&&&&&&&&&&&&%%%%%#####(((//*,,....................,,**///((((((((///*****,,,,,,,,,,**/////((((////**,,,,........,*(##########             ',
'                   &&@@&&%%@&%%%%%%%%######(((//,,................,,,***//(((((((((((((((//****,,,,,,****///((((((((((((((((//*,,,,,,,*(########%%             ',
'                   &@@@&&&&&&%%%%%%%%%%%%#((((//,,...........,,,,,***////////////((((((((///***,,,,,,***///((((((///////////////****,,*((########%&            ',
'                   %@@@&&&&&&%%%%%%%%%%%%####(/*,.........,,,,,,***///////((((((((((((/((///***,,,,,,**///(((/(((((((/////////////***,*/(####((##%&            ',
'                    &@&&&%%&&%&%%%%%%%%%%###((/*,.......,,,*******/////(((((((((((((((((///***,,,,,,,**///((((((######((((((//////***,,/(###(####%&            ',
'                    .&&&&&&&&&&&&&%%%%%%%%###(/*,......,,,****////((((((####%##((((((((((/****,.....,,*/(((((((#########((((((((//***,,*/(##(####%%*           ',
'                     &&&&&&&&&&&&&&%%%%%%%###(*,.......,,***/////(((((#%%######(/(((((////***,.......,*/((((((/((######((/(####((//***,,*(#((####%%*           ',
'                     .&&&&&&&%%%%&&&%%%%%##((/*,.......,,*/////(((######%#%###(/*/(((//****,,,..   ..,*/////(///(((((##(///####(((//**,,*/((#####%&*           ',
'                      #&&&&&&&&%%&&&%%%%%%#((/,,.....,,,,**/////((#((((((((////***//*****,,,,...     .,**************///***/////((//**,,,*(####((#&*           ',
'                       @@@&&&&%#((((#%%%%%%#(/*,,,.....,,,,,*************************,,,,,.....      ..,,,**************************,,,,,*(###((/(%*           ',
'                       .@&&&&&%#(///(####%%#(/,,,,.........,,,,,,**********///****,,,,,,.......       .,,,,,,,,,**************,,,,,,,,,..,/(((((((#*           ',
'                        %@&&%###(//**/(###((/*,,..................,,,,*******,,,,,,....,......        ..,,,,,,,,,,,,,,,*****,,,,,,,......,*//((((/#*           ',
'                        .@@@#((###(/////(//**,,........         ...............................        .......,,,.........................****/((((*           ',
'                         %@&//((((/((//////*,,..........                  .....................         ......................         ...*****//((*           ',
'                         %@&//(///*/(/////**,,,.,,........                  ..................           ...,,,,.............       ......,*****///*           ',
'                         %&%*///****/////**,,,,,,..........                .........,,,,,......           ..,****,,,,........      .......,*/*****/,           ',
'                         .**/******////*,,,,,,,,,,,...........         .......,,,,,,,...........   .........,,****,,,...................,*///**,*/             ',
'                         .**/****/*/(//*,,,,,,,,,,,.........................,,****,,.................,..,,....,**/**,,,,,...............,*(//**,*(             ',
'                         .&%/**,**//(((//***,,,,,,,,,....................,,,,,***///*,.......,,,,,,,,,,,,,****,,**///**,,,,,,.............,*(((/***(           ',
'                         .@&/**,*///((//****,,,,,,,,................,,,,,*****////////*,,*//////////////////(((////////****,,,,,,,,,,,,....*(///*,*#           ',
'                         .@&(*,,,**//**********,,,,,.....,,,,,,,,,,,,,,*****///////////((##%%###(((((////((((((((///////********,,,,,,,,,,,*(/**,,/#           ',
'                          @&(*,,,*//**,,,******,,,,,...,,,,,,,,,,,,,,,*****/(###///***/((######((((((((((((((((((///(////************,,,,,,*/*,,,,/%           ',
'                          #@(*,,,,***,,,********,,,,,,,,,,,,,*************/(######,,,*######(//(((((((((#####/########(//////***********,,,,,,,***/%           ',
'                           &%/*,,..,,,,,********,,,,,,,,,**************//##@@#######@##@#######////(((####@@#######@@######///**********,,,*/**/#%             ',
'                           %&(***,,,...,,*********,,,,,,***********////(###%####@#####%##########////////(#######@###@%####////////*****,,,****(%#             ',
'                            &%(*****,...,**********,,,,,**********/////##################((/,,(((******///((###((####((((####(///////*****,,.,*/(#%            ',
'                              //*,,**,,,,**************************////######/*,,,,,,,,*******///////////////////********//////////*******,,,,/#(              ',
'                               *#*,..,,,..,//*****************************////////((((((((##((((((((((((((###((####(((((((/////**************(/                ',
'                                  **.,,,,,*/********************,******//((##%%####((//(/////*,*/////*////(((/((#%%&%%%##((///*************//                  ',
'                                    /*,**///********************,,,***///(##%%%%##(/*,,*,,,,,...,*,,,.,,*******/(%%&&%#((///***************#                   ',
'                                    *#///(##/////****************,,,**********////**,,,,.........,,....,,,,,,,,/(###(//******,,***********/                    ',
'                                          %%#/////***************,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,**,,,,*******************,,,,,,**********(                    ',
'                                           (#(/////***************,,,,,,,,,,,,,,..........,,,.,,,*,,,,,******,,*********,,,,,,******/***/(/                    ',
'                                           (%((//////**************,,,,,,,,...................,,,,,,,,,,,,,**,*********,,,,,,,**//**/***/(                     ',
'                                           (%((//(////*************,,,,,,..................,,,,,,,,,,,,,*************,,,,,,*****//****/((                      ',
'                                           (#(/////////************,,,,,,.............,,,,,,,,,,,,,*****************,,,,,,*****/****//(*                       ',
'                                           (#/**//((/////**********,,,,,,...........,,,,,,,,**********************,,,,,,,*******/***/(                         ',
'                                           /(/***//((/////*********,,,,,,..............,,,,**********************,,,,,,,,**********/#.                         ',
'                                           //*****/////////*******,,,,,,.................,,,,,,***************,,,,,,,,,,,******/**/%.                          ',
'                                           //*,,,***////////*******,,,,,,.....................,,,,,,,,,,,,,,,,,......,,,,*******//#                            ',
'                                           /*,,,,,***////////*******,,,,,........       ...........................,,,,,*******/(,                             ',
'                                           #*,,,,,,,****////////*******,,,,........         ......................,,,,,*******/#                               ',
'                                          .(,,,,,,,,,,****/////////*******,,,,,..................................,,,,,********/#                               ',
'                                          %(,,,,,,,,,,,****/////////////*****,,,,,,,,,,......,,,,,,,....,......,,,,**********,/#                               ',
'                                          %(,,,,,,,,,,,,****///////////////*****,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,*************,*#                               ',
'                                          %/.,,,,,,,,,,,,*****//////////////////***************,,,,,,,,,,,,******///*********,*(                               ',
'                                          (*,,,,..,,,,,,,,,*****/////////////////////////**////*************////////*******,,,*(.                              ',
'                       .#(/**/******/***//*,,,,,...,,,,,,,,,******/////////////////////////////////////////////////********,,,*((                              ',
'                      ,/,..,,,,,,***//////*,,,,....,,,,,,,,,*******///////////////////////////////////////////////*********,,,/((                              ',
'                     ,,.  ..*//*//////((((*,,.......,,,,,,,,,********///////////////////////////////////////////*************,*/#                              ',
'                     *....  .*/(((((((((((*,,.......,,,,,,,,,,*********///////////////////////////////////////**************,,*/#.                             ',
'                    *.. ..   .,//((((((///*,...,.....,,,,,,,,,,**********////////////////////////////////////**************,,,,,/(                             ',
'                    ,..  ..    ..*///******,,.........,,,,,,,,,************/////////////////////////////////***************,,,,,,,/(                           ',
'                   /...........  ..,*/(//*,,,..,,,..,,,,,,,,,,,****************///////////////////////////***************,,,,,,*,,,(                           ',
'                   ,.............. .,*//*,,,,........,,,,,,,,,,,,,****************////////////////////////*****,,********,,,,***,,,*#                          ',
'                   ..................,,,,,,,,.........,,,,,,,,,,,,,,*******************///////////////////*****,,********,,,*///*,,,(                          ',
'                   ............,,,,.  ...,,,....,,..,,,,,,,,,,,,,,,,,,,,,****************//////////////*******,,***********,*(((*,.,/                          ',
'                  ...................    ......,,....,,,,,,,,,,,,,,,,,,,,,************************************,,,,*********//*,,,,,*/                          ',
'                  . .. ...............     ..........,,,,,,,,,,,,,,,,,,,,,,,,********************************************///*,,,,,..*/                         ',
'                  . .. ..................        ........,,,,,,,,,,,,,,,,,,,,*****************************************///**,.,,.....,,,*/*                     ',
'                  ..................,,,,,... ....   .......,,,,,,,,,,,,,,,***********************************/******/////**,,,,,,,,..,,....,,                  ',
'                 ..............,,,,,,,,,,,,,,...................,,,,,,,,,,**********************************//*****//(//(((*,,,,,,...........,,,,,             ',
'            .     ............,,,,***,,,,,**,,,,..................,,,,,,,,*****************************////***/**///(////((/,,,,,,,,.,,..........,,*/.         ',
'   .. ,            ........,,....,,,********,****,,.....................,****************************//////**/////((((/**(#(*,,,,,,,,**,............  .,,      ',
 '']




