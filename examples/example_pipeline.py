#!/usr/bin/env python2

import ehtim as eh
import ruamel.yaml as yaml

input = "M87/er4v2/data/lo/hops_3601_M87.LL+netcal.uvfits"

print("----------------------------------------------------------------")
print("Use the ehtim pipeline interface as a pipeline constructor")

with open("example_pipeline.yaml", 'r') as f:
    pipeline = eh.Pipeline(yaml.load(f))

obs = pipeline.apply(input).data

print("----------------------------------------------------------------")
print("Use the ehtim pipeline interface with method-chaining notation")

obs = eh.Pipeline(input) \
    .load() \
    .average(minlen=500, sec=300.0) \
    .flag(uv_min=0.1e9) \
    .flag(site='SR') \
    .reorder() \
    .data
