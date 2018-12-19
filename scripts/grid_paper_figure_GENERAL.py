''' 
code to combine many plot figures for the Imaging 4 paper 
written by Joseph farah

Input: reads in a configuration file with the following format:
 - comments above, delimited by /* */
 - first line: list of input files, arranged in wrapped order, with the word "blank" denoting a blank 
   slot in the figure, delimited by commas
 - second line: list of tuples representing figures to be highlighted and their color, delimited by commas (x, y, color) and individual tuples separated by a hashtag with no space
 - third line: "ROW" or "FIGURE", denoting whether the colorbar should be applied to the entire figure or by row
 - fourth line: the dimensions of the figure
 - fifth line: column labels, list of strings, delimited by commas
 - sixth line: row labels, list of strings, delimited by commas
 - seventh line: field of view, microarcseconds, integer
 - eighth line: load aipscc? boolean
 - ninth line: blurring size, radians
 - tenth line: color function
 - eleventh line: link to obs file
 - twelvth line: tuple (x,y) where you want the scale and beam to be shown, separated by hashtags



TODO: 
 - read in FITS
 - outline specific figures based on tuples, default no boxes
 - colorbar -- by row (side) or by whole figure (bottom)
 - flexibility in how we draw the colors
 - receive 

 - label shift
 - save ouptut
 - consistent field of view
 - load aipcc = true
 - blurring 
 - color function
 - blurring 


 - plot contours
 - arbitrary scale
 - 


'''

## imports ##
import sys
import itertools
import numpy as np
import ehtim as eh  
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pylab import *
rc('axes', linewidth=2)

## global variables ##
configurationObject = None


class Configuration(object):
    """
    creates a new Configuration object to keep track of configuration parameters

    """
    def __init__(self, _fp):
        """ 
        initializes instance variables and 
        reads in the configuration file

        """

        self._configFilePath        = _fp
        self.imageList              = []
        self.highlightedFiguresList = []
        self.colorbarPlacement      = 'FIGURE'
        self.shape                  = (0, 0)
        self.rowLabels              = []
        self.colLabels              = []
        self.fov                    = 150
        self.aipscc                 = False
        self.blurFactor             = 0
        self.cmap                   = 'afmhot'
        self.beamparams             = None
        self.beamOnList             = None

        self.readConfigFile()


    def readConfigFile(self):
        """
        read in the configuration file.

        """

        ## open the configuration file as a file object ##
        with open(self._configFilePath, "r") as configFileObject:
            configurationFileLines = configFileObject.readlines()

        ## ignore comments ##
        # isComment = False
        # for i, line in enumerate(configurationFileLines):
        #   print line
        #   if line.strip() == '/*':
        #       ## this is the first part of a comment. delete everything until the closing comment ##
        #       print "FIRST PART"
        #       isComment = True
        #   if line.strip() == '*/':
        #       ## this is the final line of a comment. stop deleting stuff! ##
        #       print "SECOND PART"
        #       del configurationFileLines[i]
        #       isComment = False
        #       break

        #   if isComment == True:
        #       del configurationFileLines[i]

        ## first line will be a list of image files ##
        self.imageList = map(str, configurationFileLines[0].split(','))
        print self.imageList

        ## second line will be a list of tuples wiht information about the highlighted figures ##
        self.highlightedFiguresList = configurationFileLines[1].split("#")

        ## third line will be "ROW" or "COLUMN" ##
        self.colorbarPlacement = configurationFileLines[2].strip()

        ## fourth line will be two numbers ##
        self.shape = (int(configurationFileLines[3].split(',')[0]), int(configurationFileLines[3].split(',')[1]))

        ## fifth line will be row labels ##
        self.rowLabels = configurationFileLines[4].split(',')

        ## sixth line will be column labels ##
        self.colLabels = configurationFileLines[5].split(',')

        ## seventh line will be field of view ##
        self.fov                    =  float(configurationFileLines[6].strip())*eh.RADPERUAS

        ## eighth line will be aipscc ##
        self.aipscc                 = str(configurationFileLines[7].strip()) == 'True'

        ## ninth line will be blur factor ##
        self.blurFactor             = float(configurationFileLines[8].strip())*eh.RADPERUAS

        ## tenth line will be colormap ##
        self.cmap                   = str(configurationFileLines[9].strip())

        ## eleventh line will be obs file ##
        pathToObsFile               = str(configurationFileLines[10].strip())
        obs                         = eh.obsdata.load_uvfits(pathToObsFile)
        self.beamparams             = obs.fit_beam()

        ## twelvth line will be figures with the beam params on ##
        self.beamOnList             = configurationFileLines[11].split("#")





def main():
    """
    main function flow

    """

    ## access global variables ##
    global configurationObject
    colorbarRange = [0, 0]

    ## read in configuration file ##
    _configFilePath = sys.argv[1]

    ## generate a new configuration object ##
    configurationObject = Configuration(_configFilePath)

    ## begin iterating over each figure ##
    fig, axs = plt.subplots(configurationObject.shape[0], configurationObject.shape[1], 
                        figsize=(configurationObject.shape[0]**2, configurationObject.shape[1]**1.999))

    ## list of indices where the labels will go ##
    labelIndicesRowRow = range(0, len(axs))                 # row index of row labels
    labelIndicesRowCol = [0 for i in range(len(axs))]       # col index of row labels
    labelIndicesColCol = range(0, len(axs))                 # col index of col labels
    labelIndicesColRow = [0 for i in range(len(axs))]       # row index of col labels

    subplotNum = 0
    for r, row in enumerate(axs):
        for c, col in enumerate(row):

            ## get the figure that should be plotted here and load it if it isn't blank ##
            if configurationObject.imageList[subplotNum].strip() == 'blank':
                print "blank"
                subplotNum += 1
                axs[r][c].tick_params(  
                    axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    left=False,         # ticks along the top edge are off
                    labelbottom=False,  # labels along the bottom edge are off
                    labelleft=False  # labels along the bottom edge are off
                )

                ## turn off the axes ##
                axs[r][c].spines['bottom'].set_color('#FFFFFF')
                axs[r][c].spines['top'].set_color('#FFFFFF') 
                axs[r][c].spines['right'].set_color('#FFFFFF')
                axs[r][c].spines['left'].set_color('#FFFFFF')

                ## push the label location down one/to the right one
                labelIndicesRowCol[r] +=  1
                labelIndicesColRow[c] +=  1

                continue

            if configurationObject.imageList[subplotNum] != 'blank':
                image = eh.image.load_image(configurationObject.imageList[subplotNum].strip(), aipscc=configurationObject.aipscc)
                print "Blurring"
                image = image.blur_circ(configurationObject.blurFactor)
                print "Regridding"
                image = image.regrid_image(configurationObject.fov, 128)
                colorbarRange = [min(colorbarRange[0], np.min(image.imvec)), 
                                 max(colorbarRange[1], np.max(image.imvec))]
                axsim = axs[r][c].imshow(image.imvec.reshape(image.xdim, image.ydim), interpolation='gaussian', cmap=configurationObject.cmap)

                for beamOnRequest in configurationObject.beamOnList:
                    print beamOnRequest
                    beamOnRequestList = beamOnRequest.strip().strip('(').strip(')').split(',')
                    hrRow = int(beamOnRequestList[0])
                    hrCol = int(beamOnRequestList[1])
                    if (hrRow, hrCol) == (r, c):
                        ## add beam image ##
                        beamparams = [configurationObject.beamparams[0], configurationObject.beamparams[1], configurationObject.beamparams[2],
                                      -.35*image.fovx(), -.35*image.fovy()]
                        beamimage = image.copy()
                        beamimage.imvec *= 0
                        beamimage = beamimage.add_gauss(1, beamparams)
                        halflevel = 0.5*np.max(beamimage.imvec)
                        beamimarr = (beamimage.imvec).reshape(beamimage.ydim,beamimage.xdim)
                        axs[r][c].contour(beamimarr, levels=[halflevel], colors='w', linewidths=1)

                        ## add scale ##
                        fov_uas = image.xdim * image.psize / eh.RADPERUAS # get the fov in uas
                        roughfactor = 1./3. # make the bar about 1/3 the fov
                        fov_scale = int( math.ceil(fov_uas * roughfactor / 10.0 ) ) * 10 # round around 1/3 the fov to nearest 10
                        start = image.xdim * roughfactor / 3.0 # select the start location
                        end = start + fov_scale/fov_uas * image.xdim # determine the end location based on the size of the bar
                        axs[r][c].plot([start, end], [image.ydim-start, image.ydim-start], color="white", lw=1) # plot line
                        axs[r][c].text(x=(start+end)/2.0, y=image.ydim-start+image.ydim/20, s= str(fov_scale) + " $\mu$as", color="white", ha="center", va="center", fontsize=12)

                ## add the colorbar if the ROW flag is enabled ##
                if configurationObject.colorbarPlacement == 'ROW' and c == len(axs[0]) - 1:
                    plt.subplots_adjust(wspace=0.02, hspace=0.08)
                    divider = make_axes_locatable(axs[r][c])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = plt.colorbar(axsim, ax=axs[r][c], ticks=[round(colorbarRange[0], 4), round((colorbarRange[1] - colorbarRange[1])/2.,4), round(colorbarRange[1], 4)], cax=cax)
                    cbar.set_label('Brightness Temp (K)')

                ## remove all ticks ##
                axs[r][c].tick_params(  
                    axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    left=False,         # ticks along the top edge are off
                    labelbottom=False,  # labels along the bottom edge are off
                    labelleft=False  # labels along the bottom edge are off
                )

                ## check if the r and c match up to one of the highlight requests ##
                isHighlighted = False
                for highlightRequest in configurationObject.highlightedFiguresList:
                    highlightRequestList = highlightRequest.strip().strip('(').strip(')').split(',')
                    hrRow = int(highlightRequestList[0])
                    hrCol = int(highlightRequestList[1])
                    if (hrRow, hrCol) == (r, c):
                        highlightColor = "#"+highlightRequestList[-1].strip()
                        axs[r][c].spines['bottom'].set_color(highlightColor)
                        axs[r][c].spines['top'].set_color(highlightColor) 
                        axs[r][c].spines['right'].set_color(highlightColor)
                        axs[r][c].spines['left'].set_color(highlightColor)
                        isHighlighted = True

                ## label placement ##
                ## check row labels first ##
                hasLabel = False
                for ri, ridx in enumerate(labelIndicesRowRow):
                    testRC = (ridx, labelIndicesRowCol[ri])
                    if testRC == (r, c):
                        axs[r][c].set_ylabel(configurationObject.rowLabels[r].strip())
                        hasLabel = True

                ## check column labels next ##
                for ri, ridx in enumerate(labelIndicesColRow):
                    testRC = (ridx, labelIndicesColCol[ri])
                    if testRC == (r, c):
                        axs[r][c].set_title(configurationObject.colLabels[c].strip())
                        hasLabel = True

                # if isHighlighted ==  False and hasLabel == False: axs[r][c].set_axis_off()
        subplotNum += 1

    print "Final colorbar range:", colorbarRange
    if configurationObject.colorbarPlacement == 'FIGURE':
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.14, 0.05, 0.65, 0.03])
        cbar = fig.colorbar(axsim, cax=cbar_ax, orientation='horizontal', ticks=[-1, 0, 1])
        plt.subplots_adjust(wspace=0.025, hspace=0.025)
        cbar.set_label('Brightness temperature (K)')
        cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])
        # [round(colorbarRange[0], 4), round((colorbarRange[1] - colorbarRange[1])/2.,4), round(colorbarRange[1], 4)]

    # gs1 = gridspec.GridSpec(len(axs), len(axs[0]))
    # gs1.update(wspace=0.0, hspace=0.0)
    

    plt.savefig("output.pdf")

if __name__ == '__main__':
    main()