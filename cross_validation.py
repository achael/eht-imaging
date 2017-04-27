import vlbi_imaging_utils as vb
import numpy as np
import maxen as mx
import matplotlib.pyplot as plt
from IPython import display

def split_obs_training_test(obs, frac_test = 0.1):
    #Returns two observation objects: a training set and a testing set. The split is currently done based on scans. 
    split_obs = vb.split_obs(obs)
    n_training = int(np.floor((1.0-frac_test)*len(split_obs)))
    n_testing = len(split_obs) - n_training

    i_training = np.array(np.sort(np.random.choice(len(split_obs),n_training,replace=False)))
    i_testing  = np.array(np.sort(list(set(np.arange(len(split_obs)))-set(i_training))))

    obs_training = vb.merge_obs([split_obs[i] for i in i_training])
    obs_testing  = vb.merge_obs([split_obs[i] for i in i_testing])

    return {'training':obs_training, 'testing':obs_testing }


def chisq(im, obs): #need to generalize this to other data products
    data = obs.unpack(['u','v','vis','sigma'])
    uv = np.hstack((data['u'].reshape(-1,1), data['v'].reshape(-1,1)))
    A = vb.ftmatrix(im.psize, im.xdim, im.ydim, uv, pulse=im.pulse)
    return mx.chisq(im.imvec, A, data['vis'], data['sigma'])

def plot_im_List_Set(im_List_List, cv_chi2_matrix, alpha_matrix, plot_log_amplitude=False, ipynb=False):
    plt.ion()
    plt.clf()

    Prior = im_List_List[0][0]

    xnum = len(im_List_List[0])
    ynum = len(im_List_List)

    for i in range(xnum*ynum):
        plt.subplot(ynum, xnum, i+1)    
        im = im_List_List[(i-i%xnum)/xnum][i%xnum]
        if plot_log_amplitude == False:
            plt.imshow(im.imvec.reshape(im.ydim,im.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian')     
        else:
            plt.imshow(np.log(im.imvec.reshape(im.ydim,im.xdim)), cmap=plt.get_cmap('afmhot'), interpolation='gaussian') 
        plt.xticks([])
        plt.yticks([])        
        
        if ynum == 1:
            plt.title("Test chi^2: %0.3f" % (cv_chi2_matrix[0][i]))
            plt.xlabel("alpha: %0.3g" % (alpha_matrix[i]))
            plt.ylabel('')
        else:
            #plt.title("alpha1: %0.3f, alpha2: %0.3f, Test chi^2: %0.3f" % (alpha_matrix[0][(i-i%xnum)/xnum],alpha_matrix[1][i%xnum],cv_chi2_matrix[(i-i%xnum)/xnum][i%xnum]))
            plt.title("Test chi^2: %0.3f" % (cv_chi2_matrix[(i-i%xnum)/xnum][i%xnum]))
            if i%xnum == 0:
                plt.ylabel("alpha_s1: %0.3f" % (alpha_matrix[0][(i-i%xnum)/xnum]))
            else:
                plt.ylabel('')

            if (i-i%xnum)/xnum == ynum-1:
                plt.xlabel("alpha_s2: %0.3f" % (alpha_matrix[1][i%xnum]))
            else:
                plt.xlabel('')

    for j in range(5):
        plt.tight_layout()

    plt.show()

    if ipynb:
        display.clear_output()
        display.display(plt.gcf())

