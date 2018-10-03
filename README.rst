ehtim (eht-imaging)
===================

Python modules for simulating and manipulating VLBI data and producing images with regularized maximum likelihood methods. This version is an early release so please submit a pull request or email achael@cfa.harvard.edu if you have trouble or need help for your application.

The package contains several primary classes for loading, simulating, and manipulating VLBI data. The main classes are the ``Image``, ``Array``, ``Obsdata``, ``Movie``  ``Caltable``  and ``Vex`` classes, which provide tools for loading images and data, producing simulated data from realistic u-v tracks, and calibrating, inspecting, and plotting data. ``Imager``  is a generic Stokes I imaging module that can produce images from data sets using various data terms and regularizers.

Note that this is a pre-release of ehtim.  If you have a problem please submit a pull request on the git repository.

Installation
------------
Download the latest version from the `GitHub repository <https://github.com/achael/eht-imaging>`_, change to the main directory and run:

.. code-block:: bash

    pip install .

It should install the necessary libraries `astropy <http://www.astropy.org/>`_, `ephem <http://pypi.python.org/pypi/pyephem/>`_, `future <http://pypi.python.org/pypi/future>`_, `matplotlib <http://www.matplotlib.org/>`_, `numpy <http://www.numpy.org/>`_, `scipy <http://www.scipy.org/>`_, `pandas <http://www.pandas.pydata.org/>`_ automatically.

If you want to use fast fourier transforms, you will also need to install `NFFT <https://github.com/NFFT/nfft>`_ and the `pynnft wrapper <https://github.com/ghisvail/pyNFFT/>`_ before installing ehtim. The simplest way is to use `conda <https://anaconda.org/conda-forge/pynfft/>`__ to to install both NFFT and the pynfft wrapper. 

.. code-block:: bash

    conda install -c conda-forge pynfft

Alternatively, first manually install NFFT following the instructions  `here <https://github.com/NFFT/nfft>`_, making sure to use the --enable-openmp flag in compilation. Then install `pynnft <https://github.com/ghisvail/pyNFFT/>`_ with pip, following the instructions to link the installation to where you installed NFFT. 

Documentation
-------------
Documentation is  `here <https://achael.github.io/eht-imaging>`_ .

Here are some ways to learn to use the code:

- Start with the script examples/example.py, which contains a series of sample commands to load an image and array, generate data, and produce an image with various imaging algorithms.
- `Slides <https://www.dropbox.com/s/7533ucj8bt54yh7/Bouman_Chael.pdf?dl=0>`_ from the EHT2016 data generation and imaging workshop contain a tutorial on generating data with the vlbi imaging `website <http://vlbiimaging.csail.mit.edu>`_, loading into the library, and producing an image. Note that this presentation used a previous version of the code -- some function names and prefixes may need to be updated.

Some publications that use ehtim
------------
If you use ehtim in your publication, please cite both  `Chael et al. 2016 <http://adsabs.harvard.edu/abs/2016ApJ...829...11C>`_  and  `Chael et al. 2018 <http://adsabs.harvard.edu/abs/2018ApJ...857...23C>`_

Let us know if you use ehtim in your publication and we'll list it here!

- High-Resolution Linear Polarimetric Imaging for the Event Horizon Telescope, `Chael et al. 2016 <https://arxiv.org/abs/1605.06156>`_ 

- Computational  Imaging for VLBI Image Reconstruction, `Bouman et al. 2016 <http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Bouman_Computational_Imaging_for_CVPR_2016_paper.html>`_ 

- Stochastic Optics: A Scattering Mitigation  Framework for Radio Interferometric Imaging, `Johnson 2016 <https://arxiv.org/abs/1610.05326>`_ 

- Quantifying Intrinsic Variability of  Sgr A* using Closure Phase Measurements of the Event Horizon Telescope, `Roelofs et al. 2017 <https://arxiv.org/abs/1708.01056>`_ 

- Reconstructing Video from Interferometric Measurements of Time-Varying Sources, `Bouman et al. 2017 <https://arxiv.org/abs/1711.01357>`_  

- Dynamical Imaging with Interferometry, `Johnson et al. 2017 <https://arxiv.org/abs/1711.01286>`_  

- Interferometric Imaging Directly with Closure Phases and Closure Amplitudes, `Chael et al. 2018 <https://arxiv.org/abs/1803.07088>`_

- A Model for Anisotropic Interstellar Scattering and its Application to Sgr A*, `Psaltis et al. 2018 <https://arxiv.org/abs/1805.01242>`_

- The Currrent Ability to Test Theories of Gravity with Black Hole Shadows, `Mizuno et al. 2018 <https://arxiv.org/abs/1804.05812>`_

- The Scattering and Intrinsic Structure of Sagittarius A* at Radio Wavelengths, `Johnson et al. 2018 <https://arxiv.org/abs/18008.08966>`_

Acknowledgements
----------------
The oifits_new code used for reading/writing .oifits files is a slightly modified version of Paul Boley's package at `<http://astro.ins.urfu.ru/pages/~pboley/oifits>`_. The oifits read/write functionality is still being tested and may not work with all versions of python or astropy.

The documentation is styled after `dfm's projects <https://github.com/dfm>`_ 

License
-------
ehtim is licensed under GPLv3. See LICENSE.txt for more details.


