import vlbi_imaging_utils_v3 as vb
import maxen_v2 as mx
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def vis_resid_analyze(obs, image):
    """Examines the residuals of image compared to the obs"""
    
    data = obs.unpack(['u','v','vis','phase','sigma'])
    vis = data['vis']
    phase = data['phase']
    sig = data['sigma']
    sigphase = sig / np.abs(vis)
    uv = data[['u','v']].view(('f8',2))
    
    # get model visibilities
    mat = vb.ftmatrix(image.psize, image.xdim, image.ydim, uv)
    vis_model = np.dot(mat, image.imvec)
    
    # residuals
    resid = (vis - vis_model)/sig
    residamp = (np.abs(vis) - np.abs(vis_model))/sig
    #residphase = (vis/np.abs(vis) - vis_model/np.abs(vis_model))/sigphase
    
    # chi - squared
    chisq = np.sum(np.abs(resid)**2)

    print  "Chi^2/2N: %f" % (chisq/(2*len(vis)))

    # plot the histograms
    plt.figure()
    plt.clf()
    plt.subplot(121)
    n,bins,patches = plt.hist(np.real(resid), normed=1, bins=50,range=(-5,5),  color='b', alpha=0.5)
    n,bins,patches = plt.hist(np.imag(resid), normed=1, bins=50,range=(-5,5),  color='r', alpha=0.5)
    y = scipy.stats.norm.pdf(bins)
    plt.plot(bins, y, 'b--', linewidth=3)
    plt.xlabel('Normalized Residual Re/Imag')
    plt.ylabel('p')
    
    plt.subplot(122)
    n,bins,patches = plt.hist(residamp, normed=1, bins=50,range=(-5,5),  color='b', alpha=0.5)
    y = scipy.stats.norm.pdf(bins)
    plt.plot(bins, y, 'b--', linewidth=3)
    plt.xlabel('Normalized Amplitude Residual')
    plt.ylabel('p')
    
#    plt.subplot(133)
#    n,bins,patches = plt.hist(np.real(residphase), normed=1, bins=50, color='b', alpha=0.5)
#    n,bins,patches = plt.hist(np.imag(residphase), normed=1, bins=50, color='r', alpha=0.5)
#    y = scipy.stats.norm.pdf(bins)
#    plt.plot(bins, y, 'b--', linewidth=3)
#    plt.xlabel('Normalized Phase Residual Re/Imag')
#    plt.ylabel('p')
    
    
#    plt.subplot(133)
#    n,bins,patches = plt.hist(np.abs(resid), normed=1, bins=50, color='g')
#    y = scipy.stats.rayleigh.pdf(bins)
#    plt.plot(bins, y, 'b--', linewidth=3)
#    plt.xlabel('Normalized Residual Amp.')
#    plt.ylabel('p')
    
    plt.show()
    
def bis_resid_analyze(obs, image):
    """Examines the residuals of image compared to the obs"""
    
    biarr = obs.bispectra(mode='all')
    uv1 = np.hstack((biarr['u1'].reshape(-1,1), biarr['v1'].reshape(-1,1)))
    uv2 = np.hstack((biarr['u2'].reshape(-1,1), biarr['v2'].reshape(-1,1)))
    uv3 = np.hstack((biarr['u3'].reshape(-1,1), biarr['v3'].reshape(-1,1)))
    bis = biarr['bispec']
    A3 = (vb.ftmatrix(image.psize, image.xdim, image.ydim, uv1),
          vb.ftmatrix(image.psize, image.xdim, image.ydim, uv2),
          vb.ftmatrix(image.psize, image.xdim, image.ydim, uv3)
         )
    sig = biarr['sigmab']
    
    # get model bispectra
    bis_model = np.dot(A3[0], image.imvec)*np.dot(A3[1], image.imvec)*np.dot(A3[2], image.imvec)
    
    # residuals
    resid = (bis - bis_model)/sig
    
    # chi - squared
    chisq = np.sum(np.abs(resid)**2)

    print  "Bispectrum Chi^2/2N: %f" % (chisq/(2*len(bis)))
    print "Bispectrum Chi^2/2(N-p): %f" % (chisq/(2*(len(bis)-len(image.imvec))))

    # plot the histograms
    plt.figure()
    plt.subplot(121)
    n,bins,patches = plt.hist(np.real(resid), normed=1, range=(-5,5), bins=50, color='b', alpha=0.5)
    n,bins,patches = plt.hist(np.imag(resid), normed=1, range=(-5,5), bins=50, color='r', alpha=0.5)
    y = scipy.stats.norm.pdf(bins)
    plt.plot(bins, y, 'b--', linewidth=3)
    plt.xlabel('Normalized Bispectrum Residual Re/Imag')
    plt.ylabel('p')
    
    plt.subplot(122)
    n,bins,patches = plt.hist(np.abs(resid), normed=1, range=(-5,5),  bins=50, color='g')
    y = scipy.stats.rayleigh.pdf(bins)
    plt.plot(bins, y, 'b--', linewidth=3)
    plt.xlabel('Normalized Bispectrum Residual Amp.')
    plt.ylabel('p')
    
#    plt.subplot(133)
#    n,bins,patches = plt.hist(np.angle(resid)/vb.DEGREE, normed=1, bins=100, color='g')
#    y = scipy.stats.vonmises.pdf(bins)
#    plt.plot(bins, y, 'k--', linewidth=1.5)
#    plt.xlabel('Normalized Residual Phase (deg)')
#    plt.ylabel('p')
    
    plt.show()
    
def pol_resid_analyze(obs, image):
    """Examines the residuals of image compared to the obs"""
    
    data = obs.unpack(['u','v','vis','qvis','uvis','pvis','m','sigma'])
    vis = data['vis']
    qvis = data['qvis']
    uvis = data['uvis']
    pvis = data['pvis']
    
    m = data['m']
    sig = data['sigma']
    psig = np.sqrt(2) * sig
    msig = vb.merr(sig, vis, m)

    uv = data[['u','v']].view(('f8',2))
        
    # get model visibilities
    mat = vb.ftmatrix(image.psize, image.xdim, image.ydim, uv)
    vis_model = np.dot(mat, image.imvec)
    qvis_model = np.dot(mat, image.qvec)
    uvis_model = np.dot(mat, image.uvec)
    pvis_model = np.dot(mat, image.qvec + 1j*image.uvec)
    m_model = pvis_model/vis_model
    
    # don't count measurements with SNR < 2:
    mask = (np.abs(qvis)/sig > 5)
    
    # debiased amps
    qvis_d = np.real(np.sqrt(np.abs(qvis)**2 - sig**2 + 0j))
    uvis_d = np.real(np.sqrt(np.abs(uvis)**2 - sig**2 + 0j))
    pvis_d = np.real(np.sqrt(np.abs(pvis)**2 - psig**2 + 0j))
    m_d = np.real(np.sqrt(np.abs(m)**2 - msig**2 + 0j))
    
    # residuals
    residq = (qvis[mask] - qvis_model[mask])/sig[mask]
    residu = (uvis - uvis_model)/sig
    residp = (pvis - pvis_model)/psig
    residm = (m - m_model)/msig
    
    residq_amp = (qvis_d[mask] - np.abs(qvis_model[mask]))/sig[mask]
    residu_amp = (uvis_d - np.abs(uvis_model))/sig
    residp_amp = (pvis_d - np.abs(pvis_model))/psig
    residm_amp = (m_d - np.abs(m_model))/msig
    
    # chi - squared
    qchisq = np.sum(np.abs(residq)**2)
    uchisq = np.sum(np.abs(residu)**2)
    pchisq = np.sum(np.abs(residp)**2)
    mchisq = np.sum(np.abs(residm)**2)
    
    print  "Q Chi^2/2N: %f" % (qchisq/(2*len(vis)))
    print  "U Chi^2/2N: %f" % (uchisq/(2*len(vis)))
    print  "P Chi^2/2N: %f" % (pchisq/(2*len(vis)))
    print  "m Chi^2/2N: %f" % (mchisq/(2*len(vis)))
    
    # plot the histograms
    plt.figure()
    plt.subplot(241)
    n,bins,patches = plt.hist(np.real(residq), normed=1, range=(-5,5), bins=50, color='b', alpha=0.5)
    n,bins,patches = plt.hist(np.imag(residq), normed=1, range=(-5,5), bins=50, color='r', alpha=0.5)
    y = scipy.stats.norm.pdf(bins)
    plt.plot(bins, y, 'b--', linewidth=3)
    plt.xlabel('Q Normalized Residual Re/Imag')
    plt.ylabel('p')
    
    plt.subplot(242)
    n,bins,patches = plt.hist(np.real(residu), normed=1, range=(-5,5), bins=50, color='b', alpha=0.5)
    n,bins,patches = plt.hist(np.imag(residu), normed=1, range=(-5,5), bins=50, color='r', alpha=0.5)
    y = scipy.stats.norm.pdf(bins)
    plt.plot(bins, y, 'b--', linewidth=3)
    plt.xlabel('U Normalized Residual Re/Imag')
    plt.ylabel('p')
    
    plt.subplot(243)
    n,bins,patches = plt.hist(np.real(residp), normed=1, range=(-5,5), bins=50, color='b', alpha=0.5)
    n,bins,patches = plt.hist(np.imag(residp), normed=1, range=(-5,5), bins=50, color='r', alpha=0.5)
    y = scipy.stats.norm.pdf(bins)
    plt.plot(bins, y, 'b--', linewidth=3)
    plt.xlabel('P Normalized Residual Re/Imag')
    plt.ylabel('p')
    
    plt.subplot(244)
    n,bins,patches = plt.hist(np.real(residm), normed=1, range=(-5,5), bins=50, color='b', alpha=0.5)
    n,bins,patches = plt.hist(np.imag(residm), normed=1, range=(-5,5), bins=50, color='r', alpha=0.5)
    y = scipy.stats.norm.pdf(bins)
    plt.plot(bins, y, 'b--', linewidth=3)
    plt.xlabel('m Normalized Residual Re/Imag')
    plt.ylabel('p')
    
    plt.subplot(245)
    n,bins,patches = plt.hist(residq_amp, normed=1,range=(-5,5),  bins=50, color='g')
    y = scipy.stats.norm.pdf(bins)
    plt.plot(bins, y, 'b--', linewidth=3)
    plt.xlabel('Q Normalized Amplitude Residual')
    plt.ylabel('p')
    
    plt.subplot(246)
    n,bins,patches = plt.hist(residu_amp, normed=1,range=(-5,5),  bins=50, color='g')
    y = scipy.stats.norm.pdf(bins)
    plt.plot(bins, y, 'b--', linewidth=3)
    plt.xlabel('U Normalized AmplitudeResidual')
    plt.ylabel('p')
    
    plt.subplot(247)
    n,bins,patches = plt.hist(residp_amp, normed=1,range=(-5,5),  bins=50, color='g')
    y = scipy.stats.norm.pdf(bins)
    plt.plot(bins, y, 'b--', linewidth=3)
    plt.xlabel('P Normalized Amplitude Residual')
    plt.ylabel('p')
    
    plt.subplot(248)
    n,bins,patches = plt.hist(residm_amp, normed=1,range=(-5,5),  bins=50, color='g')
    y = scipy.stats.norm.pdf(bins)
    plt.plot(bins, y, 'b--', linewidth=3)
    plt.xlabel('m Normalized Amplitude Residual')
    plt.ylabel('p')
    
#    plt.subplot(245)
#    n,bins,patches = plt.hist(np.abs(residq), normed=1, bins=50, color='g')
#    y = scipy.stats.rayleigh.pdf(bins)
#    plt.plot(bins, y, 'b--', linewidth=3)
#    plt.xlabel('Q Normalized Residual Amp.')
#    plt.ylabel('p')
#    
#    plt.subplot(246)
#    n,bins,patches = plt.hist(np.abs(residu), normed=1, bins=50, color='g')
#    y = scipy.stats.rayleigh.pdf(bins)
#    plt.plot(bins, y, 'b--', linewidth=3)
#    plt.xlabel('U Normalized Residual Amp.')
#    plt.ylabel('p')
#    
#    plt.subplot(247)
#    n,bins,patches = plt.hist(np.abs(residp), normed=1, bins=50, color='g')
#    y = scipy.stats.rayleigh.pdf(bins)
#    plt.plot(bins, y, 'b--', linewidth=3)
#    plt.xlabel('P Normalized Residual Amp.')
#    plt.ylabel('p')
#    
#    plt.subplot(248)
#    n,bins,patches = plt.hist(np.abs(residm), normed=1, bins=50, color='g')
#    y = scipy.stats.rayleigh.pdf(bins)
#    plt.plot(bins, y, 'b--', linewidth=3)
#    plt.xlabel('m Normalized Residual Amp.')
#    plt.ylabel('p')
     
    plt.show()
