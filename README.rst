ehtim (eht-imaging)
===================

Python modules for simulating and manipulating VLBI data and producing images with regularized maximum likelihood methods. This version is an early release so please raise an issue, submit a pull request, or email achael@princeton.edu if you have trouble or need help for your application.

The package contains several primary classes for loading, simulating, and manipulating VLBI data. The main classes are the ``Image``, ``Movie``, ``Array``, ``Obsdata``, ``Imager``, and ``Caltable`` classes, which provide tools for loading images and data, producing simulated data from realistic u-v tracks,  calibrating, inspecting, and  plotting data, and producing images from data sets in various polarizations using various data terms and regularizing functions.

Installation
------------

The latest stable version (`1.2.2 <https://github.com/achael/eht-imaging/releases/tag/v1.2.2>`_) is available on `PyPi <https://pypi.org/project/ehtim/>`_. Simply install pip and run

.. code-block:: bash

    pip install ehtim

Incremental updates are developed on the `dev branch <https://github.com/achael/eht-imaging/tree/dev>`_. To use the very latest (unstable) code, checkout the dev branch, change to the main eht-imaging directory, and run:

.. code-block:: bash

    pip install .

Installing with pip will update most of the required libraries automatically (`numpy <http://www.numpy.org/>`_, `scipy <http://www.scipy.org/>`_, `matplotlib <http://www.matplotlib.org/>`_, `astropy <http://www.astropy.org/>`_, `ephem <http://pypi.python.org/pypi/pyephem/>`_, `future <http://pypi.python.org/pypi/future>`_, `h5py <http://www.h5py.org/>`_, and `pandas <http://www.pandas.pydata.org/>`_).

**If you want to use fast fourier transforms, you will also need to separately install** `NFFT <https://github.com/NFFT/nfft>`_ **and its** `pynfft wrapper <https://github.com/ghisvail/pyNFFT/>`_. The simplest way is to use `conda <https://anaconda.org/conda-forge/pynfft/>`__ to install both:


.. code-block:: bash

    conda install -c conda-forge pynfft

Alternatively, first install NFFT manually following the instructions on the `readme <https://github.com/NFFT/nfft>`_, making sure to use the ``--enable-openmp`` flag in compilation. Then install `pynfft <https://github.com/ghisvail/pyNFFT/>`_, with pip, following the readme instructions to link the installation to where you installed NFFT. Finally, reinstall ehtim.

**Certain eht-imaging functions require other external packages that are not automatically installed.** In addition to pynfft, these include  `networkx <https://networkx.github.io/>`_ (for image comparison functions), `requests <http://docs.python-requests.org/en/master/>`_ (for dynamical imaging), and `scikit-image <https://scikit-image.org/>`_ (for Hough transforms). However, the vast majority of the code will work without these dependencies.

Documentation
-------------
Documentation is  `here <https://achael.github.io/eht-imaging>`_ .

A full tutorial is in progress, but here are some ways to learn to use the code:

- Start with the script examples/example.py, which contains a series of sample commands to load an image and array, generate data, and produce an image with various imaging algorithms.

- `Slides <https://www.dropbox.com/s/7533ucj8bt54yh7/Bouman_Chael.pdf?dl=0>`_ from the EHT2016 data generation and imaging workshop contain a tutorial on generating data with the VLBI imaging `website <http://vlbiimaging.csail.mit.edu>`_, loading into the library, and producing an image.

Some publications that use ehtim
--------------------------------
If you use ehtim in your publication, please cite `Chael+ 2018 <http://adsabs.harvard.edu/abs/2018ApJ...857...23C>`_

Let us know if you use ehtim in your publication and we'll list it here!

- High-Resolution Linear Polarimetric Imaging for the Event Horizon Telescope, `Chael et al. 2016 <https://arxiv.org/abs/1605.06156>`_ 

- Computational  Imaging for VLBI Image Reconstruction, `Bouman et al. 2016 <http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Bouman_Computational_Imaging_for_CVPR_2016_paper.html>`_ 

- Stochastic Optics: A Scattering Mitigation  Framework for Radio Interferometric Imaging, `Johnson 2016 <https://arxiv.org/abs/1610.05326>`_ 

- Reconstructing Video from Interferometric Measurements of Time-Varying Sources, `Bouman et al. 2017 <https://arxiv.org/abs/1711.01357>`_  

- Dynamical Imaging with Interferometry, `Johnson et al. 2017 <https://arxiv.org/abs/1711.01286>`_  

- Interferometric Imaging Directly with Closure Phases and Closure Amplitudes, `Chael et al. 2018 <https://arxiv.org/abs/1803.07088>`_

- A Model for Anisotropic Interstellar Scattering and its Application to Sgr A*, `Psaltis et al. 2018 <https://arxiv.org/abs/1805.01242>`_

- The Currrent Ability to Test Theories of Gravity with Black Hole Shadows, `Mizuno et al. 2018 <https://arxiv.org/abs/1804.05812>`_

- The Scattering and Intrinsic Structure of Sagittarius A* at Radio Wavelengths, `Johnson et al. 2018 <https://arxiv.org/abs/18008.08966>`_

- Testing General Relativity with the Black Hole Shadow Size and Asymmetry of Sagittarius A*: Limitations from Interstellar Scattering, `Zhu et al. 2018 <https://arxiv.org/abs/1811.02079>`_

- The Size, Shape, and Scattering of Sagittarius A* at 86 GHz: First VLBI with ALMA, `Issaoun et al. 2019a <https://arxiv.org/abs/1901.06226>`_

- First M87 Event Horizon Telescope Results IV: Imaging the Central Supermassive Black Hole, `The Event Horizon Telescope Collaboration 2019 <https://arxiv.org/abs/1906.11241>`_

- VLBI Imaging of black holes via second moment regularization, `Issaoun et al. 2019b <https://arxiv.org/pdf/1908.01296.pdf>`_

- Using evolutionary algorithms to model relativistic jets: Application to NGC 1052, `Fromm et al. 2019 <https://arxiv.org/pdf/1904.00106.pdf>`_

- EHT-HOPS Pipeline for Millimeter VLBI Data Reduction, `Blackburn et al. 2019 <https://arxiv.org/pdf/1903.08832>`_

- Multi-wavelength torus-jet model for Sagittarius A*, `Vincent et al. 2019 <https://arxiv.org/pdf/1902.01175>`_
