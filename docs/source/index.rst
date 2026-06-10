.. ehtim documentation master file, created by
   sphinx-quickstart on Tue May 16 13:37:05 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ehtim (eht-imaging)
===================
.. image:: https://zenodo.org/badge/42943499.svg
   :target: https://zenodo.org/badge/latestdoi/42943499


Python modules for simulating and manipulating VLBI data and producing images with regularized maximum likelihood methods. This version is an early release so please raise an issue, submit a pull request, or email achael@princeton.edu if you have trouble or need help for your application.

The package contains several primary classes for loading, simulating, and manipulating VLBI data. The main classes are the :class:`Image`, :class:`Array`, and :class:`Obsdata`, which provide tools for manipulating images, simulating interferometric data from images, and plotting and analyzing these data. :class:`Movie` and :class:`Vex` provide tools for producing time-variable simulated data and observing with real VLBI tracks from .vex files. :class:`imager` is a generic imager class that can produce images from data sets in various polarizations using various data terms and regularizers.  

.. note::

    This is a pre-release of ehtim. If you have a problem please submit a pull request on the git repository and/or email achael@outlook.com


Installation
------------

The latest stable version (|stable_release|) is available on `PyPi <https://pypi.org/project/ehtim/>`_. Simply install pip and run

.. code-block:: bash

    pip install ehtim

Incremental updates are developed on the `dev branch <https://github.com/achael/eht-imaging/tree/dev>`_. To use the very latest (unstable) code, checkout the dev branch, change to the main eht-imaging directory, and run:

.. code-block:: bash

    pip install .

Installing with pip will update most of the required libraries automatically (`numpy <http://www.numpy.org/>`_, `scipy <http://www.scipy.org/>`_, `matplotlib <http://www.matplotlib.org/>`_, `astropy <http://www.astropy.org/>`_, `ephem <http://pypi.python.org/pypi/pyephem/>`_, `future <http://pypi.python.org/pypi/future>`_, `h5py <http://www.h5py.org/>`_, and `pandas <http://www.pandas.pydata.org/>`_).


NFFT Installation
------------
**If you want to use fast fourier transforms, you will also need to separately install** `NFFT <https://github.com/NFFT/nfft>`_ **and its** `pyNFFT wrapper <https://github.com/ghisvail/pyNFFT/>`__. 

The simplest way is to use `conda <https://anaconda.org/conda-forge/pynfft/>`__ to install both NFFT and pyNFFT:


.. code-block:: bash

    conda install -c conda-forge pynfft

Alternatively, first install NFFT manually following the instructions on the `readme <https://github.com/NFFT/nfft>`__, making sure to use the ``--enable-openmp`` flag in compilation. Then install `pynfft <https://github.com/ghisvail/pyNFFT/>`__, with pip, following the readme instructions to link the installation to where you installed NFFT. Finally, reinstall ehtim.

**Note that, unfortunately, pyNFFT is only supported for python versions 3.11 or lower.** eht-imaging version 2.0 using the `finufft <https://github.com/flatironinstitute/finufft>`__ library is in active development.

**For M1/M2/M3/M4/M5 Macs (MacOS >= v12.0)**, install the updated Mac version of `pynfft <https://github.com/rohandahale/pyNFFT.git>`__ and follow the instructions on the `readme  <https://github.com/rohandahale/pyNFFT.git>`__ to manually install `fftw <http://www.fftw.org>`_, `nfft <https://github.com/NFFT/nfft>`__ and then `pynfft <https://github.com/rohandahale/pyNFFT.git>`__.

**Certain eht-imaging functions require other external packages that are not automatically installed.** In addition to pynfft, these include  `networkx <https://networkx.github.io/>`_ (for image comparison functions), `requests <http://docs.python-requests.org/en/master/>`_ (for dynamical imaging), and `scikit-image <https://scikit-image.org/>`_ (for a few image analysis functions). However, the vast majority of the code will work without these dependencies.

Documentation and Tutorials
---------------------------
Documentation is  `here <https://achael.github.io/eht-imaging>`_.

A tutorial jupyter notebook for imaging can be found in the repo at `tutorials/ehtim_tutorial.ipynb <https://github.com/achael/eht-imaging/blob/main/tutorials/ehtim_tutorial.ipynb>`__

`Slides <https://docs.google.com/presentation/d/1A0y9omYI2ueSUa6_t5reylBhw6eiLwjqDzw-HUOk8Ac/edit?usp=sharing>`__ for the included tutorial walk through the basic steps of reconstructing EHT images with the code

Scripts in the examples directory contain several older simple example workflows but have not been recently validated.  


Citation
--------------------------------
If you use ehtim in your publication, please cite `Chael+ 2018 <http://adsabs.harvard.edu/abs/2018ApJ...857...23C>`_.

The latest version is also available as a static doi on `Zenodo <https://zenodo.org/badge/latestdoi/42943499>`_.


Documentation
-------------
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   image
   array
   obsdata
   movie
   model
   imager
   calibration
   plotting
   scattering
   statistics
   survey
   vex



License
-------
ehtim is licensed under GPLv3. See LICENSE.txt for more details.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
