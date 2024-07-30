from __future__ import division
from __future__ import print_function

import ehtim as eh
from ehtim.io import load
path = eh.__path__[0]

def test_load_obs_uvfits():
    """Test if load_obs_uvfits() can successfully read a uvfits file
    """
    assert load.load_obs_uvfits(path + "/../data/sample.uvfits")
    # TODO: verify the result
