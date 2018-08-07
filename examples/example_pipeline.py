#!/usr/bin/env python2

import ehtim as eh
import yaml

print("----------------------------------------------------------------")
print("Use the ehtim pipeline interface as a pipeline constructor")

with open("example_pipeline.yaml", 'r') as f:
    dict = yaml.load(f)

obs = eh.Pipeline(dict) \
    .apply(["M87/er4v2/data/lo/hops_3601_M87.LL+netcal.uvfits",
            "M87/er4v2/data/lo/hops_3601_M87.RR+netcal.uvfits"])

print("----------------------------------------------------------------")
print("Use the ehtim pipeline interface with method chaining notations")

obs = eh.Pipeline(["M87/er4v2/data/lo/hops_3601_M87.LL+netcal.uvfits",
                   "M87/er4v2/data/lo/hops_3601_M87.RR+netcal.uvfits"]) \
    .load() \
    .scale(zbl=0.1) \
    .scale(noise='auto') \
    .flag(anomalous='amp') \
    .average(sec=600, old=True) \
    .merge()
