.. ehtim documentation master file, created by
   sphinx-quickstart on Tue May 16 13:37:05 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ehtim (eht-imaging)
===================


Python modules for simulating and manipulating VLBI data and producing images with regularized maximum likelihood methods. This version is an early release so please submit a pull request or email achael@cfa.harvard.edu if you have trouble or need help for your application.

The package contains several primary classes for loading, simulating, and manipulating VLBI data. The main classes are the :class:`Image`, :class:`Array`, and :class:`Obsdata`, which provide tools for manipulating images, simulating interferometric data from images, and plotting and analyzing these data. :class:`Movie` and :class:`Vex` provide tools for producing time-variable simulated data and observing with real VLBI tracks from .vex files. :class:`imager` is a generic imager class that can produce images from data sets in various polarizationsusing various data terms and regularizers.  

.. note::

    This is a pre-release of ehtim. If you have a problem please submit a pull request on the git repository and/or email achael@cfa.harvard.edu

Installation
------------

Download the latest version from the `GitHub repository <https://github.com/achael/eht-imaging>`_, change to the main directory and run:

.. code-block:: bash

    pip install .


It should install most of the required libraries automatically (`astropy <http://www.astropy.org/>`_, `ephem <http://pypi.python.org/pypi/pyephem/>`_, `future <http://pypi.python.org/pypi/future>`_, `h5py <http://www.h5py.org/>`_ , `html <http://www.decalage.info/python/html>`_ , `networkx <https://networkx.github.io/>`_, `numpy <http://www.numpy.org/>`_, `pandas <http://www.pandas.pydata.org/>`_ , `matplotlib <http://www.matplotlib.org/>`_,  `requests <http://docs.python-requests.org/en/master/>`_, `scipy <http://www.scipy.org/>`_, `skimage <https://scikit-image.org/>`_).

**If you want to use fast fourier transforms, you will also need to separately install** `NFFT <https://github.com/NFFT/nfft>`_ **and its** `pynnft wrapper <https://github.com/ghisvail/pyNFFT/>`_. The simplest way is to use `conda <https://anaconda.org/conda-forge/pynfft/>`__ to to install both:

.. code-block:: bash

    conda install -c conda-forge pynfft

Alternatively, first install NFFT manually following the instructions on the `readme <https://github.com/NFFT/nfft>`_, making sure to use the :code:`--enable-openmp` flag in compilation. Then install `pynft <https://github.com/ghisvail/pyNFFT/>`_, with pip, following the readme instructions to link the installation to where you installed NFFT. Finally, reinstall ehtim.


Tutorials
-------------

Tutorials are in progress, but here are some ways to learn the code

- The script in `examples/example.py <https://github.com/achael/eht-imaging/blob/master/examples/example.py>`_ has a series of sample commands to load an image and array, generate data, and produce an image with regularized maximum likelihood on closure quantities.
- `Slides <https://www.dropbox.com/s/7533ucj8bt54yh7/Bouman_Chael.pdf?dl=0>`_ from the EHT 2016 conference data generation and imaging workshop contain a tutorial on generating data externally with the vlbi imaging `website <http://vlbiimaging.csail.mit.edu>`_, loading into the library, and producing an image.

Documentation
-------------
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   image
   array
   obsdata
   movie
   vex
   imager
   calibration
   plotting
   scattering
   statistics


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
