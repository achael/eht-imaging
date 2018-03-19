ehtim (eht-imaging)
===================

Python modules for simulating and manipulating VLBI data and producing images with regularized maximum likelihood methods. This version is an early release so please submit a pull request or email achael@cfa.harvard.edu if you have trouble or need help for your application.

The package contains several primary classes for loading, simulating, and manipulating VLBI data. The main classes are the ``Image``, ``Array``, ``Obsdata``. ``Movie`` and ``Vex`` provide tools for producing time-variable simulated data and observing with real VLBI tracks from .vex files. ``imager`` is a generic Stokes I imaging module that can produce images from data sets using various data terms and regularizers.

Note that this is a pre-release of ehtim.  If you have a problem please submit a pull request on the git repository.

Installation
------------
Download the latest version from the `GitHub repository <https://github.com/achael/eht-imaging>`_, change to the main directory and run:

.. code-block:: bash

    pip install .

It should install the depended libraries `astropy <http://www.astropy.org/>`_, `ephem <http://pypi.python.org/pypi/pyephem/>`_, `future <http://pypi.python.org/pypi/future>`_, `matplotlib <http://www.matplotlib.org/>`_, `numpy <http://www.numpy.org/>`_, `scipy <http://www.scipy.org/>`_ automatically.

You may need to install `NFFT <https://github.com/NFFT/nfft>`_ and the `pynnft wrapper <https://github.com/ghisvail/pyNFFT/>`_ manually. First install NFFT following the instructions on the `github readme <https://github.com/NFFT/nfft>`_, making sure to use the --enable-openmp flag in compilation. Then install `pynnft <https://github.com/ghisvail/pyNFFT/>`_, with pip, following the readme instructions to link the installation to where you installed NFFT. 

Alternatively, you can use `conda <https://anaconda.org/conda-forge/pynfft/>`__ to to install both NFFT and the pynfft wrapper

Documentation
-------------
The `documentation <https://achael.github.io/eht-imaging>`_ is in progress, but here are some ways to learn to use the code:

- The file example.py has a series of sample commands to load an image and array, generate data, and produce an image.
- `Slides <https://www.dropbox.com/s/7533ucj8bt54yh7/Bouman_Chael.pdf?dl=0>`_ from the EHT2016 data generation and imaging workshop contain a tutorial on generating data with the vlbi imaging `website <http://vlbiimaging.csail.mit.edu>`_, loading into the library, and producing an image. Note that this presentation used a previous version of the code -- function names and prefixes will need to be updated.

PolMEM
------------
Produces polarimetric VLBI images using robust polarimetric ratio data products and entropy+total variation priors. (Andrew A. Chael et al 2016 ApJ 829 11)

ArXiv: https://arxiv.org/abs/1605.06156

CHIRP
------------

Python code to produce intensity images using patch-prior regularizers. These methods are described in detail in the following paper:

`CVPR Paper <http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Bouman_Computational_Imaging_for_CVPR_2016_paper.html>`_

Detailed derivations and additional results can be found in the accompanying `supplemental material <http://vlbiimaging.csail.mit.edu/static/papers/CHIRPsupp.pdf>`_

Please email klbouman@mit.edu if you have trouble or need help for your application.


Acknowledgements
----------------
The oifits_new code used for reading/writing .oifits files is a slightly modified version of Paul Boley's package at `<http://astro.ins.urfu.ru/pages/~pboley/oifits>`_. The oifits read/write functionality is still being tested and may not work with all versions of python or astropy.io.fits.

The jdcal.py module is from Prasanth Nair at `<http://github.com/phn/jdcal>`_.

The documentation is styled after `dfm's projects <https://github.com/dfm>`_ and the documentation for `scatterbrane <https://github.com/krosenfeld/scatterbrane>`_

License
-------
ehtim is licensed under GPLv3. See LICENSE.txt for more details.

