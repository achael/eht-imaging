#!/usr/bin/env python2
#
# This EHT imaging pipeline generator is built on algorithms developed
# by Team 1 members of the EHT imaging working group during the 2nd
# EHT Imaging Workshop.

import ehtim as eh

# Implement individual processes in an imaging pipeline
def load(name):
    return eh.obsdata.load_uvfits(name)

def scale(obs,
          zbl=None,   uvdist=1.0e9,
          noise=None, max_diff_sec=100.0):
    if zbl is not None:
        d = obs.data
        b = d['u']**2 + d['v']**2 < uvdist**2
        for t in ["vis", "sigma"]:
            for p in ["", "q", "u", "v"]:
                d[p+t][b] *= zbl
        return obs
    if noise is not None:
        if noise == 'auto':
            noise = obs.estimate_noise_rescale_factor(max_diff_sec=max_diff_sec)
        return obs.rescale_noise(noise_rescale_factor=noise)

def average(obs, sec=300, old=False):
    if old:
        print("WARNING: using old coherent average method")
        return obs.avg_coherent_old(sec)
    else:
        return obs.avg_coherent(sec)

def flag(obs,
         anomalous=None, max_diff_sec=300,
         low_snr=None,
         uv_min=None):
    if anomalous is not None:
        return obs.flag_anomalous(field=anomalous, max_diff_sec=max_diff_sec)
    if low_snr is not None:
        return obs.flag_low_snr(low_snr)
    if uv_min is not None:
        return obs.flag_uvdist(uv_mind)

def merge(obss):
    for i in range(1, obss):
        obss[i].data['time'] += i * 1e-6
        obss[i].mjd = obss[0].mjd
        obss[i].rf  = obss[0].rf
        obss[i].ra  = obss[0].ra
        obss[i].dec = obss[0].dec
        obss[i].bw  = obss[0].bw
    return eh.obsdata.merge_obs(obss).copy()

def reorder(obs, according='snr'):
    if according == 'snr':
        obs.reorder_tarr_snr()
        return obs

def add_gaussprior(obs, fov=200, npix=64, prior_fwhm=100, zbl=1.0):
    fov        *= eh.RADPERUAS
    prior_fwhm *= eh.RADPERUAS
    emptyprior  = eh.image.make_square(obs, npix, fov)
    flatprior   = emptyprior.add_flat(zbl)
    gaussprior  = emptyprior.add_gauss(zbl,
                                       (prior_fwhm, prior_fwhm, -np.pi/4, 0, 0))
    return obs, gaussprior, gaussprior

def self_cal(obs, ref, **kwargs):
    return eh.self_cal.self_cal(obs, ref, processes=0, **kwargs)

def make_imager(obs, init, prior, npix=64, **kwargs):
    return eh.imager.Imager(obs, init, prior_im=prior, **kwargs)

def imaging(imgr, init, n=5, m=3, blursz=0.33, **kwargs):
    imgr.init_next = init
    for i in range(n):
        for j in range(m):
            imgr.make_image_I(**kwargs)
            imgr.init_next = imgr.out_last()
        imgr.init_next = imgr.out_liast().blur_circ(res * blursz)
    return imgr.out_last().copy()

# Factory method
def make_process(func_name, **kwargs):
    """A factory method that creates process in a pipeline"""
    def process(data): # closure
        return globals()[func_name](data, **kwargs)
    return process

# Main script
if __name__ == "__main__":
    pipeline = [make_process("load"),
                make_process("scale_zbl", scale=0.1),
                make_process("scale_noise"),
                make_process("flag",      anomalous="amp"),
                make_process("average",   old=True),
                make_process("average",   sec=600)]

    bundle = "M87/er4v2/data/lo/hops_3601_M87.LL+netcal.uvfits"
    for p in pipeline:
        bundle = p(*bundle if isinstance(bundle, tuple) else bundle)
