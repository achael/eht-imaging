#!/usr/bin/env python2
#
# This EHT imaging pipeline generator is built on algorithms developed
# by Team 1 members of the EHT imaging working group during the 2nd
# EHT Imaging Workshop.

import ehtim as eh
import yaml

class Pipeline:
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
            return obs.flag_anomalous(field=anomalous,
                                      max_diff_seconds=max_diff_sec)
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

# Factory method
def make_process(func_name, **kwargs):
    """A factory method that creates process in a pipeline"""
    def process(data): # closure
        print(func_name, data, kwargs)
        return globals()[func_name](data, **kwargs)
    return process

# Main script
if __name__ == "__main__":
    with open("pipeline.yaml", 'r') as f:
        dict = yaml.load(f)

    pipeline = []
    for p in dict['pipeline']:
        for k, v in p.items():
            pipeline += [make_process(k) if v is None else
                         make_process(k, **v)]

    obs = "M87/er4v2/data/lo/hops_3601_M87.LL+netcal.uvfits"
    for p in pipeline:
        obs = p(obs)
