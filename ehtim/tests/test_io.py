from ..io import load

def test_load_obs_uvfits():
    """Test if load_obs_uvfits() can successfully read a uvfits file
    """
    assert load.load_obs_uvfits("../../data/sample.uvfits")
    # TODO: verify the result
