"""Smoke tests for ehtim.movie: image-list merge and HDF5 round-trip."""

import numpy as np

import ehtim as eh


def _gauss_movie():
    im = eh.image.make_empty(32, 200 * eh.RADPERUAS, 17.761, -29.0, rf=230e9)
    im = im.add_gauss(1.0, (40 * eh.RADPERUAS, 40 * eh.RADPERUAS, 0, 0, 0))
    return eh.movie.merge_im_list([im, im, im], framedur=60.0), im


def test_merge_im_list_builds_movie():
    mov, im = _gauss_movie()
    assert mov.nframes == 3
    assert mov.xdim == im.xdim
    assert mov.ydim == im.ydim


def test_movie_hdf5_roundtrip(tmp_path):
    mov, _ = _gauss_movie()
    fname = str(tmp_path / "mov.h5")
    mov.save_hdf5(fname)
    mov2 = eh.movie.load_hdf5(fname)
    assert mov2.nframes == mov.nframes
    np.testing.assert_allclose(mov2.frames[0], mov.frames[0], rtol=1e-6)
