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

    This is a pre-release of ehtim. If you have a problem please submit a pull request on the git repository and/or email achael@princeton.edu.edu

The latest stable version (`1.2.6 <https://github.com/achael/eht-imaging/releases/tag/v1.2.6>`_) is available on `PyPi <https://pypi.org/project/ehtim/>`_. Simply install pip and run

.. code-block:: bash

    pip install ehtim

Incremental updates are developed on the `dev branch <https://github.com/achael/eht-imaging/tree/dev>`_. To use the very latest (unstable) code, checkout the dev branch, change to the main eht-imaging directory, and run:

.. code-block:: bash

    pip install .

Installing with pip will update most of the required libraries automatically (`numpy <http://www.numpy.org/>`_, `scipy <http://www.scipy.org/>`_, `matplotlib <http://www.matplotlib.org/>`_, `astropy <http://www.astropy.org/>`_, `ephem <http://pypi.python.org/pypi/pyephem/>`_, `future <http://pypi.python.org/pypi/future>`_, `h5py <http://www.h5py.org/>`_, and `pandas <http://www.pandas.pydata.org/>`_).

**If you want to use fast fourier transforms, you will also need to separately install** `NFFT <https://github.com/NFFT/nfft>`_ **and its** `pynfft wrapper <https://github.com/ghisvail/pyNFFT/>`__. The simplest way is to use `conda <https://anaconda.org/conda-forge/pynfft/>`__ to install both:


.. code-block:: bash

    conda install -c conda-forge pynfft

Alternatively, first install NFFT manually following the instructions on the `readme <https://github.com/NFFT/nfft>`__, making sure to use the ``--enable-openmp`` flag in compilation. Then install `pynfft <https://github.com/ghisvail/pyNFFT/>`__, with pip, following the readme instructions to link the installation to where you installed NFFT. Finally, reinstall ehtim.

**For M1 Macs (OS >= v12.0)**, install the M1 Mac version of `pynfft <https://github.com/rohandahale/pyNFFT.git>`__ and follow the instructions on the `readme  <https://github.com/rohandahale/pyNFFT.git>`__. It has the instructions to install `fftw <http://www.fftw.org>`_, `nfft <https://github.com/NFFT/nfft>`__ and then `pynfft <https://github.com/rohandahale/pyNFFT.git>`__.

**Certain eht-imaging functions require other external packages that are not automatically installed.** In addition to pynfft, these include  `networkx <https://networkx.github.io/>`_ (for image comparison functions), `requests <http://docs.python-requests.org/en/master/>`_ (for dynamical imaging), and `scikit-image <https://scikit-image.org/>`_ (for a few image analysis functions). However, the vast majority of the code will work without these dependencies.

Documentation and Tutorials
---------------------------
Documentation is  `here <https://achael.github.io/eht-imaging>`_.

A intro to imaging tutorial jupyter notebook can be found in the repo at `tutorials/ehtim_tutorial.ipynb <https://github.com/achael/eht-imaging/blob/main/tutorials/ehtim_tutorial.ipynb>`__

`Slides <https://docs.google.com/presentation/d/1A0y9omYI2ueSUa6_t5reylBhw6eiLwjqDzw-HUOk8Ac/edit?usp=sharing>`__ for the included tutorial walk through the basic steps of reconstructing EHT images with the code

Here are some other ways to learn to use the code:

- Start with the script examples/example.py, which contains a series of sample commands to load an image and array, generate data, and produce an image with various imaging algorithms.

- Older `Slides <https://www.dropbox.com/s/7533ucj8bt54yh7/Bouman_Chael.pdf?dl=0>`__ from the EHT2016 data generation and imaging workshop contain a tutorial on generating data with the VLBI imaging `website <http://vlbiimaging.csail.mit.edu>`_, loading into the library, and producing an image.

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


Acknowledgements
----------------
The :code:`oifits_new` code used for reading/writing .oifits files is a slightly modified version of Paul Boley's package at `<http://astro.ins.urfu.ru/pages/~pboley/oifits>`_. The oifits read/write functionality is still being tested and may not work with all versions of python.

This documentation is styled after `dfm's projects <https://github.com/dfm>`_ and the documentation for `scatterbrane <https://github.com/krosenfeld/scatterbrane>`_


License
-------
ehtim is licensed under GPLv3. See LICENSE.txt for more details.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
