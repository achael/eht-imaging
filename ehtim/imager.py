import numpy as np
import matplotlib.pyplot as plt
import time

import ehtim.observing.pulses 

from ehtim.imaging.imager_utils import *
from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

NHIST = 100 # number of steps to store for hessian approx
STOP = 1e-10 # convergence criterion

DATATERMS = ['vis', 'bs', 'amp', 'cphase', 'camp']
REGULARIZERS = ['gs', 'tv', 'l1', 'patch', 'simple', 'flux','cm']

class Imager(object):
    """A general interferometric imager.
    """
    
    def __init__(self, obsdata, init_im, prior_im=None,  flux=None, clipfloor=0., maxit=50, transform='log', data_term={'vis':100}, reg_term={'simple':1}):
        
        self.logstr = ""
        self._obs_list = []
        self._init_list = []
        self._prior_list = []
        self._out_list = []
        self._flux_list = {}
        self._reg_term_list = []
        self._dat_term_list = []
        self._clipfloor_list = []
        self._maxit_list = []
        self._flux_list = []
        self._transform_list = []

        # Parameters for the next imaging iteration
        self.reg_term_next = reg_term #e.g. [('simple',1), ('l1',10), ('flux',500), ('cm',500)]
        self.dat_term_next = data_term #e.g. [('amp', 1000), ('cphase',100)]
        
        self.obs_next = obsdata
        self.init_next = init_im

        if prior_im==None: 
            self.prior_next = self.init_next
        else:
            self.prior_next = prior_im

        if flux==None:
            self.flux_next = self.prior_next.total_flux()
        else:
            self.flux_next = flux

        self.clipfloor_next = clipfloor
        self.maxit_next = maxit
        self.transform_next = transform
        self._change_imgr_params = True
        self.nruns = 0 

        #set embedding matrices and prepare imager
        self.check_params()
        self.check_limits()
        self.init_imager_I()


    def set_embed(self):

        self._embed_mask = self.prior_next.imvec > self.clipfloor_next
        coord = np.array([[[x,y] for x in np.arange(self.prior_next.xdim/2,-self.prior_next.xdim/2,-1)]
                                 for y in np.arange(self.prior_next.ydim/2,-self.prior_next.ydim/2,-1)])
        coord = self.prior_next.psize * coord.reshape(self.prior_next.ydim * self.prior_next.xdim, 2)        
        self._coord_matrix = coord[self._embed_mask]

        return


    def check_params(self):
            
        dt_here = False
        dt_type = True
            
        for term in sorted(self.dat_term_next.keys()): 
            if (term != None) and (term != False): dt_here = True
            if not ((term in DATATERMS) or term==False): dt_type = False

        st_here = False
        st_type = True
        for term in sorted(self.reg_term_next.keys()): 
            if (term != None) and (term != False): st_here = True
            if not ((term in REGULARIZERS) or term == False): st_type = False   

        if not dt_here: 
            raise Exception("Must have at least one data term!")
        
        if not st_here: 
            raise Exception("Must have at least one regularizer term!")
        
        if not dt_type:
            raise Exception("Invalid data term: valid data terms are: " + string.join(DATATERMS))
        
        if not st_type:
            raise Exception("Invalid regularizer: valid regularizers are: " + string.join(REGULARIZERS))
        
        if ((self.prior_next.psize != self.init_next.psize) or 
            (self.prior_next.xdim != self.init_next.xdim) or 
            (self.prior_next.ydim != self.prior_next.ydim)):
            raise Exception("Initial image does not match dimensions of the prior image!")

        # determine if we need to change the saved imager parameters on the next imager run
        if self.nruns == 0:
            return

        if len(self.reg_term_next) != len(self.reg_term_last()): 
            self._change_imgr_params = True
            return

        if len(self.dat_term_next) != len(self.dat_term_last()): 
            self._change_imgr_params = True
            return

        for term in sorted(self.dat_term_next.keys()):
            if term not in self.dat_term_last().keys():
                self._change_imgr_params = True
                return

        for term in sorted(self.reg_term_next.keys()):
            if term not in self.reg_term_last().keys():
                self._change_imgr_params = True
                return
        
        if ((self.prior_next.psize != self.prior_last().psize) or 
            (self.prior_next.xdim != self.prior_last().xdim) or 
            (self.prior_next.ydim != self.prior_last().ydim)):
            self.change_imgr_params = True
            return
        
    def check_limits(self):

        imsize = np.max([self.prior_next.xdim, self.prior_next.ydim]) * self.prior_next.psize
        uvmax = 1./self.prior_next.psize
        uvmin = 1./imsize
        uvdists = self.obs_next.unpack('uvdist')['uvdist']
        maxbl = np.max(uvdists)
        minbl = np.max(uvdists[uvdists > 0])
        maxamp = np.max(np.abs(self.obs_next.unpack('amp')['amp']))

        if uvmax < maxbl:
            print "Warning! Pixel Spacing is larger than smallest spatial wavelength!"
        if uvmin > minbl:
            print "Warning! Field of View is smaller than largest nonzero spatial wavelength!" 
        if self.flux_next > 1.2*maxamp:
            print "Warning! Specified flux is > 120% of maximum visibility amplitude!"
        if self.flux_next < .8*maxamp:
            print "Warning! Specified flux is < 80% of maximum visibility amplitude!"

    def reg_terms_last(self):
        if self.nruns == 0:
            print "No imager runs yet!"
            return 
        return self._reg_term_list[-1]

    def dat_terms_last(self):
        if self.nruns == 0:
            print "No imager runs yet!"
            return 
        return self._dat_term_list[-1]

    def obs_last(self):
        if self.nruns == 0:
            print "No imager runs yet!"
            return 
        return self._obs_list[-1]

    def prior_last(self):
        if self.nruns == 0:
            print "No imager runs yet!"
            return 
        return self._prior_list[-1]
   
    def out_last(self):
        if self.nruns == 0:
            print "No imager runs yet!"
            return 
        return self._out_list[-1]

    def init_last(self):
        if self.nruns == 0:
            print "No imager runs yet!"
            return 
        return self._init_list[-1]
    
    def flux_last(self):
        if self.nruns == 0:
            print "No imager runs yet!"
            return 
        return self._flux_list[-1]

    def clipfloor_last(self):
        if self.nruns == 0:
            print "No imager runs yet!"
            return 
        return self._clipfloor_list[-1]

    def maxit_last(self):
        if self.nruns == 0:
            print "No imager runs yet!"
            return 
        return self._maxit_list[-1]

    def transform_last(self):
        if self.nruns == 0:
            print "No imager runs yet!"
            return 
        return self._transform_list[-1]

    def init_imager_I(self):

        # embedding, prior & initial image vectors
        self.set_embed()
        self._nprior_I = (self.flux_next * self.prior_next.imvec / np.sum((self.prior_next.imvec)[self._embed_mask]))[self._embed_mask]
        self._ninit_I = (self.flux_next * self.init_next.imvec / np.sum((self.init_next.imvec)[self._embed_mask]))[self._embed_mask]

        # data term tuples
        if self._change_imgr_params:
            self._data_tuples = {}
            for dname in self.dat_term_next.keys():
                tup = chisqdata(self.obs_next, self.prior_next, self._embed_mask, dname)
                self._data_tuples[dname] = tup
            self._change_imgr_params = False

        return

    def make_chisq_dict(self, imvec):
        chi2_dict = {}
        for dname in sorted(self.dat_term_next.keys()):
            data = self._data_tuples[dname][0]
            sigma = self._data_tuples[dname][1]
            A = self._data_tuples[dname][2]

            chi2 = chisq(imvec, A, data, sigma, dname)
            chi2_dict[dname] = chi2
        
        return chi2_dict
    
    def make_chisqgrad_dict(self, imvec):
        chi2grad_dict = {}
        for dname in sorted(self.dat_term_next.keys()):
            data = self._data_tuples[dname][0]
            sigma = self._data_tuples[dname][1]
            A = self._data_tuples[dname][2]

            chi2grad = chisqgrad(imvec, A, data, sigma, dname)
            chi2grad_dict[dname] = chi2grad 
        
        return chi2grad_dict

    def make_reg_dict(self, imvec):
        reg_dict = {}
        for regname in sorted(self.reg_term_next.keys()):
            # incorporate flux and cm into the generic "regularizer"  function!
            if regname == 'flux':
                #norm = flux**2
                norm = 1        
                reg = (np.sum(imvec) - self.flux_next)**2 
                reg /= norm

            elif regname == 'cm':
                #norm = flux**2 * Prior.psize**2
                norm = 1        
                reg = (np.sum(imvec*self._coord_matrix[:,0])**2 + 
                       np.sum(imvec*self._coord_matrix[:,1])**2)
                reg /= norm

            else:
                reg = regularizer(imvec, self._nprior_I, self._embed_mask, 
                                  self.flux_next, self.prior_next.xdim, 
                                  self.prior_next.ydim, self.prior_next.psize, 
                                  regname)
            reg_dict[regname] = reg

        return reg_dict
    
    def make_reggrad_dict(self, imvec):
        reggrad_dict = {}
        for regname in sorted(self.reg_term_next.keys()):
            # incorporate flux and cm into the generic "regularizer"  function!
            if regname == 'flux':
                #norm = flux**2
                norm = 1        
                reg = 2*(np.sum(imvec) - flux)
                reg /= norm

            elif regname == 'cm':
                #norm = flux**2 * Prior.psize**2
                norm = 1        
                reg = 2*(np.sum(imvec*self._coord_matrix[:,0])*self._coord_matrix[:,0] + 
                         np.sum(imvec*self._coord_matrix[:,1])*self._coord_matrix[:,1])
                reg /= norm

            else:
                reg = regularizergrad(imvec, self._nprior_I, self._embed_mask, 
                                      self.flux_next, self.prior_next.xdim, 
                                      self.prior_next.ydim, self.prior_next.psize, 
                                      regname)
            reggrad_dict[regname] = reg

        return reggrad_dict

    def objfunc(self, imvec):
        if self.transform_next == 'log': 
            imvec = np.exp(imvec)

        datterm = 0.
        chi2_term_dict = self.make_chisq_dict(imvec)            
        for dname in sorted(self.dat_term_next.keys()):
            datterm += self.dat_term_next[dname] * (chi2_term_dict[dname] - 1.)

        regterm = 0
        reg_term_dict = self.make_reg_dict(imvec)  
        for regname in sorted(self.reg_term_next.keys()):
            regterm += self.reg_term_next[regname] * reg_term_dict[regname]

        return datterm + regterm

    def objgrad(self, imvec):
        if self.transform_next == 'log': 
            imvec = np.exp(imvec)

        datterm = 0.
        chi2_term_dict = self.make_chisqgrad_dict(imvec)            
        for dname in sorted(self.dat_term_next.keys()):
            datterm += self.dat_term_next[dname] * (chi2_term_dict[dname] - 1.)

        regterm = 0
        reg_term_dict = self.make_reggrad_dict(imvec)    
        for regname in sorted(self.reg_term_next.keys()):
            regterm += self.reg_term_next[regname] * reg_term_dict[regname]

        grad = datterm + regterm

        # chain rule term for change of variables        
        if self.transform_next == 'log': 
            grad *= imvec

        return grad

    def plotcur(self, imvec):
        if self._show_updates:
            if self.transform_next == 'log': 
                imvec = np.exp(imvec)
            chi2_term_dict = self.make_chisq_dict(imvec)
            reg_term_dict = self.make_reg_dict(imvec)

            chi2_keys = sorted(chi2_term_dict.keys())
            chi2_1 = chi2_term_dict[chi2_keys[0]]
            chi2_2 = 0.
            if len(chi2_term_dict) > 1:
                chi2_2 = chi2_term_dict[chi2_keys[1]]

            outstr = "i: %d " % self._nit

            for dname in sorted(self.dat_term_next.keys()):
                outstr += "%s : %0.2f " % (dname, chi2_term_dict[dname])
            for regname in sorted(self.reg_term_next.keys()):
                outstr += "%s : %0.2f " % (regname, reg_term_dict[regname])

            if np.any(np.invert(self._embed_mask)): imvec = embed(imvec, self._embed_mask)
            plot_i(imvec, self.prior_next, self._nit, chi2_1, chi2_2, ipynb=False)
            
            print outstr
        self._nit += 1

    def make_image_I(self, grads=True, show_updates=True):
        
        # Checks and initialize
        self.check_params()
        self.check_limits()
        self.init_imager_I()

        # Generate and the initial image
        if self.transform_next == 'log': xinit = np.log(self._ninit_I)
        else: xinit = self._ninit_I       
        self._nit = 0

        # Print stats
        if show_updates: self._show_updates=True
        else: self._show_updates=False
        self.plotcur(xinit)
        
        # Minimize
        optdict = {'maxiter':self.maxit_next, 'ftol':STOP, 'maxcor':NHIST}
        tstart = time.time()
        if grads:
            res = opt.minimize(self.objfunc, xinit, method='L-BFGS-B', jac=self.objgrad, 
                               options=optdict, callback=self.plotcur)
        else:
            res = opt.minimize(self.objfunc, xinit, method='L-BFGS-B', 
                               options=optdict, callback=self.plotcur)
        tstop = time.time()

        # Format output
        out = res.x
        if self.transform_next == 'log': out = np.exp(res.x)
        if np.any(np.invert(self._embed_mask)): out = embed(out, self._embed_mask)
     
        outim = image.Image(out.reshape(self.prior_next.ydim, self.prior_next.xdim), 
                            self.prior_next.psize, self.prior_next.ra, self.prior_next.dec, 
                            rf=self.prior_next.rf, source=self.prior_next.source, 
                            mjd=self.prior_next.mjd, pulse=self.prior_next.pulse)
       
        # Preserving image complex polarization fractions
        if len(self.prior_next.qvec):
            qvec = self.prior_next.qvec * out / self.prior_next.imvec
            uvec = self.prior_next.uvec * out / self.prior_next.imvec
            outim.add_qu(qvec.reshape(self.prior_next.ydim, self.prior_next.xdim), 
                         uvec.reshape(self.prior_next.ydim, self.prior_next.xdim))
       
        # Print stats
        print "time: %f s" % (tstop - tstart)
        print "J: %f" % res.fun
        print res.message
        
        # Append to history
        logstr = str(self.nruns) + ": make_image_I()" #TODO - what should the log string be? 
        self._append_image_history(outim, logstr)
        self.nruns += 1

        # Return Image object
        return outim

    def _append_image_history(self, outim, logstr):
        self.logstr += (logstr + "\n")
        self._obs_list.append(self.obs_next)
        self._init_list.append(self.init_next)
        self._prior_list.append(self.prior_next)
        self._flux_list.append(self.flux_next)
        self._clipfloor_list.append(self.clipfloor_next)
        self._maxit_list.append(self.maxit_next)
        self._transform_list.append(self.transform_next)
        self._reg_term_list.append(self.reg_term_next)
        self._dat_term_list.append(self.dat_term_next)

        self._out_list.append(outim)
        return

    def make_image_P(self):
        return
    
    def make_image_scat(self):
        return  

   
    
    

    
