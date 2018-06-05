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

It should install the depended libraries `astropy <http://www.astropy.org/>`_, `ephem <http://pypi.python.org/pypi/pyephem/>`_, `future <http://pypi.python.org/pypi/future>`_, `matplotlib <http://www.matplotlib.org/>`_, `numpy <http://www.numpy.org/>`_, `scipy <http://www.scipy.org/>`_, `pandas <http://www.pandas.pydata.org/>`_ automatically.

You will need to install `NFFT <https://github.com/NFFT/nfft>`_ and the `pynnft wrapper <https://github.com/ghisvail/pyNFFT/>`_ . You can use `conda <https://anaconda.org/conda-forge/pynfft/>`__ to to install both NFFT and the pynfft wrapper Alternatively, first install NFFT following the instructions on the `github readme <https://github.com/NFFT/nfft>`_, making sure to use the --enable-openmp flag in compilation. Then install `pynnft <https://github.com/ghisvail/pyNFFT/>`_, with pip, following the readme instructions to link the installation to where you installed NFFT. 

Documentation
-------------
Documentation is  `here <https://achael.github.io/eht-imaging>`_ .

Here are some ways to learn to use the code:

- The file examples/example.py has a series of sample commands to load an image and array, generate data, and produce an image.
- `Slides <https://www.dropbox.com/s/7533ucj8bt54yh7/Bouman_Chael.pdf?dl=0>`_ from the EHT2016 data generation and imaging workshop contain a tutorial on generating data with the vlbi imaging `website <http://vlbiimaging.csail.mit.edu>`_, loading into the library, and producing an image. Note that this presentation used a previous version of the code -- function names and prefixes will need to be updated.

Publications that use ehtim
------------

`Chael et al. 2016 <https://arxiv.org/abs/1605.06156>`_ , `Bouman et al. 2016 <http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Bouman_Computational_Imaging_for_CVPR_2016_paper.html>`_ , `Johnson 2016 <https://arxiv.org/abs/1610.05326>`_ , `Roelofs et al. 2017 <https://arxiv.org/abs/1708.01056>`_ , `Bouman et al. 2017 <https://arxiv.org/abs/1711.01357>`_ , `Johnson et al. 2017 <https://arxiv.org/abs/1711.01286>`_ , `Chael et al. 2018 <https://arxiv.org/abs/1803.07088>`_


Acknowledgements
----------------
The oifits_new code used for reading/writing .oifits files is a slightly modified version of Paul Boley's package at `<http://astro.ins.urfu.ru/pages/~pboley/oifits>`_. The oifits read/write functionality is still being tested and may not work with all versions of python or astropy.

The jdcal.py module is from Prasanth Nair at `<http://github.com/phn/jdcal>`_.

The documentation is styled after `dfm's projects <https://github.com/dfm>`_ and the documentation for `scatterbrane <https://github.com/krosenfeld/scatterbrane>`_

License
-------
ehtim is licensed under GPLv3. See LICENSE.txt for more details.

