import numpy as np
import matplotlib.pyplot as plt
import vlbi_imaging_utils as vb
import scipy.ndimage as nd
import scipy.interpolate
import vlbi_plots as vbp
import time
import scipy.special 

def main():
    im = vb.load_im_txt('./models/avery_m87_2_eofn.txt')
    eht = vb.load_array('./arrays/EHT2017_rusen.txt')
    obsd = eht.obsdata(im.ra, im.dec, 230.e9, 4.e9, 60., 600, 0., 24.)

    umin = np.min(obsd.unpack('uvdist')['uvdist'])
    umax = np.min(obsd.unpack('uvdist')['uvdist'])

    if im.psize < 1/(2*umax): 
        print "u_max safely < 0.5 maximum wavelength!"
    else:
        print "u_max approaching maximum wavelength!"

    npad = int(np.ceil(10./(im.psize*umin)))
    if npad % 2: npad += 1
    #npad = power_of_two(npad)
    padval = int(np.floor((npad - im.xdim)/2.)) # non square images?? # odd dimensions??

    print "setting padded image size to %i so that delta u <= 0.5 u_min" % npad

    imarr = im.imvec.reshape((im.xdim, im.ydim))
    imarr = np.pad(imarr, padval, 'constant', constant_values=0.0)
    print imarr.shape
    imarr = np.fft.ifftshift(imarr)
    #imarr=np.fliplr(imarr)
    #imarr=np.flipud(imarr)
    du = 1./(imarr.shape[0]*im.psize)

    #RFFT is faster, but only returns half the domain!
    t=time.time()
    vis_im = np.fft.rfft2(imarr)
    vis_im = np.fft.fftshift(vis_im, axes=0)

    # uv data points from obs
    ucoord = obsd.unpack(['u'])['u']
    vcoord = obsd.unpack(['v'])['v'] 
    uv= np.hstack((ucoord.reshape(-1,1), vcoord.reshape(-1,1))) #ordinary 2d array
    uv2= np.hstack((vcoord.reshape(-1,1), ucoord.reshape(-1,1)))

    #no negative u for rfft
    mask= ucoord < 0.
    uv2[:,1][mask] = -uv2[:,1][mask]
    uv2[:,0][mask] = -uv2[:,0][mask]
    uv2[:,1] = uv2[:,1]/ du
    uv2[:,0] = uv2[:,0]/ du + 0.5*npad
    uv2 = uv2.T

    visdatare = nd.map_coordinates(np.real(vis_im), uv2)
    visdataim = nd.map_coordinates(np.imag(vis_im), uv2)
    visdata = visdatare + 1j*visdataim
    visdata[mask] = visdata[mask].conj()

    #extra phase to match centroid convention
    phase = np.exp(-1j*np.pi*im.psize*(ucoord+vcoord))
    obs = obsd.copy()
    obs.data['vis'] = visdata*phase
    print "RFFT time:" , time.time()-t

    #Full FFT is slower but not confusing.
    t=time.time()
    vis_im = np.fft.fft2(imarr)
    vis_im = np.fft.fftshift(vis_im)

    # uv data points from obs
    ucoord = obsd.unpack(['u'])['u']
    vcoord = obsd.unpack(['v'])['v'] 
    uv= np.hstack((ucoord.reshape(-1,1), vcoord.reshape(-1,1))) #ordinary 2d array

    uv2= np.hstack((vcoord.reshape(-1,1), ucoord.reshape(-1,1)))
    uv2 = (uv2 / du + 0.5*npad).T

    visdatare = nd.map_coordinates(np.real(vis_im), uv2)
    visdataim = nd.map_coordinates(np.imag(vis_im), uv2)
    visdata = visdatare + 1j*visdataim

    #extra phase to match centroid convention -- right?
    phase = np.exp(-1j*np.pi*im.psize*(ucoord+vcoord))
    obs2 = obsd.copy()
    obs2.data['vis'] = visdata*phase
    print "FFT time:" , time.time()-t

    t=time.time()
    obs3 = im.observe_same_nonoise(obsd)
    print "DFT time:" , time.time() - t

    # griddata is SLOW! - too arbitrary
    # gridded fft data points -- right?? check cell centers
    #ulist = np.arange(0,-npad,-1)*du + (du*npad)/2.0
    #vlist = np.arange(0,npad,1)*du - (du*npad)/2.0
    #vlist = np.arange(-npad/2.+1,-npad,-1)*du + (du*npad)/2.0  #for case of half domain
    #vlist =  np.fft.fftshift(np.fft.fftfreq(npad,d=im.psize))
    #uvgrid = np.array([[(u,v) for u in vlist] for v in vlist])

    #vis_im = vis_im.flatten()
    #uvgrid = uvgrid.reshape((-1,2))
    #visdata = scipy.interpolate.griddata(uvgrid, vis_im, uv, method='linear') #change method - linear slow?

    #REGRID data -- puts nonzero vis everywhere.... want zero if not on tracks
    #t=time.time()
    #vlist =  np.fft.fftshift(np.fft.fftfreq(npad,d=im.psize))
    #uvgrid = np.array([[(u,v) for u in vlist] for v in vlist])
    #uvgrid = uvgrid.reshape((-1,2))

    ucoord = obsd.unpack(['u'])['u']
    vcoord = obsd.unpack(['v'])['v'] 
    vu2= np.hstack((vcoord.reshape(-1,1), ucoord.reshape(-1,1)))
    du = 1./(npad*im.psize)
    vu2 = (vu2 / du + 0.5*npad)

    visgrid = np.zeros((npad, npad)).astype('c16')
    pradius = 2
    for k in xrange(len(vlist)):
        point = vu2[k]
        vispoint = visdata[k]

        vumin = np.ceil(point - pradius).astype(int)
        vumax = np.floor(point + pradius).astype(int)

        #print vumin, vumax
        for i in np.arange(vumin[0], vumax[0]+1):
            for j in np.arange(vumin[1], vumax[1]+1):
                #visgrid[i,j] += conv_func_pill(j-point[1], i-point[0],1.) * vispoint
                #visgrid[i,j] += conv_func_gauss(j-point[1], i-point[0],pradius) * vispoint
                visgrid[i,j] += conv_func_sphere(j-point[1], i-point[0],pradius,0) * vispoint
     
        

def conv_func_pill(x,y): 
    if abs(x) < 0.5 and abs(y) < 0.5: 
        out = 1.
    else: 
        out = 0.
    return out

def conv_func_gauss(x,ys):
    return np.exp(-(x**2 + y**2))

def conv_func_sphere(x,y,p,m):
    etax = 2.*x/float(p)
    etay = 2.*x/float(p)
    psix =  abs(1-etax**2)**m * scipy.special.pro_rad1(m,0,0.5*np.pi*p,etax)[0] #BUG in scipy spheroidal function!!
    psiy = abs(1-etay**2)**m * scipy.special.pro_rad1(m,0,0.5*np.pi*p,etay)[0]
    
    return psix*psiy


def conv_func_pill(x,y): 
    if abs(x) < 0.5 and abs(y) < 0.5: 
        out = 1.
    else: 
        out = 0.
    return out

def conv_func_gauss(x,ys):
    return np.exp(-(x**2 + y**2))
def gridder(uv, data, npix, psize, conv_func="pillbox", p_rad=1.):
    """
    Grids data sampled at uv points in obs on a square npix x npix grid
    psize is the image domain pixel size of the corresponding real space image
    conv_func is the convolution function: current options are "pillbox" and "gaussian"
    p_rad is the radius inside wich the conv_func is nonzero (set to 1 for 
    """

    if len(uv) != len(data): 
        raise Exception("uv and data are not the same length!")
    if not (conv_func in ['pillbox','gaussian']):
        raise Exception("conv_func must be either 'pillbox' or 'gaussian'")

    vu2= np.hstack((uv[:,1].reshape(-1,1), uv[:,0].reshape(-1,1)))
    du = 1./(npix*psize)
    vu2 = (vu2 / du + 0.5*npix)

    datagrid = np.zeros((npad, npad)).astype('c16')
    for k in xrange(len(data)):
        point = vu2[k]
        vispoint = data[k]

        vumin = np.ceil(point - prad).astype(int)
        vumax = np.floor(point + prad).astype(int)

        #print vumin, vumax
        for i in np.arange(vumin[0], vumax[0]+1):
            for j in np.arange(vumin[1], vumax[1]+1):
                if conv_func == 'pillbox':
                    visgrid[i,j] += conv_func_pill(j-point[1], i-point[0]) * vispoint

                elif conv_func == 'gaussian':
                    visgrid[i,j] += conv_func_gauss(j-point[1], i-point[0]) * vispoint
    
    return datagrid

def power_of_two(target):
    cur = 1
    if target > 1:
        for i in xrange(0, int(target)):
            if (cur >= target):
                return cur
            else: cur *= 2
    else:
        return 1

#print "Regrid Data: " , time.time() - t





