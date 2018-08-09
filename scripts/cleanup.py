#!/usr/bin/env python

import ehtim as eh
import ruamel.yaml as yaml

input  = "M87/er4v2/data/lo/hops_3601_M87.LL+netcal.uvfits"
config = "example_pipeline.yaml"
output = "cache.uvfits"

with open(config, 'r') as f:
    config = yaml.load(f)
    obs = eh.Pipeline(config).apply(input).data
    obs.save_uvfits(output)
