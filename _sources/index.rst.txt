.. ehtim documentation master file, created by
   sphinx-quickstart on Tue May 16 13:37:05 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ehtim (eht-imaging)
===================

Python modules for simulating and manipulating VLBI data and producing images with regularized gradient descent methods. This version is an early release so please submit a pull request or email achael@cfa.harvard.edu if you have trouble or need help for your application.

The package contains several primary classes for loading, simulating, and manipulating VLBI data. The main classes are the :class:`Image`, :class:`Array`, and :class:`Obsdata`. :class:`Movie` and :class:`Vex` provide tools for producing time-variable simulated data and observing with real VLBI tracks from .vex files. ``imager`` is a generic Stokes I imaging module that can produce images from data sets using various data terms and regularizers.  

.. note::

    This is a pre-release of ehtim.  If you have a problem please submit a pull request on the git repository.

Installation
------------
Download the latest version from the `GitHub repository <https://github.com/achael/eht-imaging>`_
, change to the main directory and run:

.. code-block:: bash

    python setup.py install

You will need working installations of `numpy <http://www.numpy.org/>`_, `matplotlib <http://www.matplotlib.org/>`_, `scipy <http://www.scipy.org/>`_, and `astropy <http://www.astropy.org/>`_. 

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


The documentation is in progress, but here are some other ways to learn to use the code: 

- The file examples/example.py has a series of sample commands to load an image and array, generate data, and produce an image. 
- `Slides <https://www.dropbox.com/s/7533ucj8bt54yh7/Bouman_Chael.pdf?dl=0>`_ from the EHT2016 data generation and imaging workshop contain a tutorial on generating data with the vlbi imaging `website <http://vlbiimaging.csail.mit.edu>`_, loading into the library, and producing an image.


Acknowledgements
----------------
The oifits_new code used for reading/writing .oifits files is a slightly modified version of Paul Boley's package at `<http://astro.ins.urfu.ru/pages/~pboley/oifits>`_. The oifits read/write functionality is still being tested and may not work with all versions of python or astropy.io.fits.

The jdcal.py module is from Prasanth Nair at `<http://github.com/phn/jdcal>`_.

This documentation is styled after `dfm's projects <https://github.com/dfm>`_ and the documentation for `scatterbrane <https://github.com/krosenfeld/scatterbrane>`_

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
