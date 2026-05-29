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

The latest stable version (`1.4 <https://github.com/achael/eht-imaging/releases/tag/v1.4>`_) is available on `PyPi <https://pypi.org/project/ehtim/>`_. Requires Python 3.11 or 3.12. Simply install pip and run

.. code-block:: bash

    pip install ehtim

Incremental updates are developed on the `dev branch <https://github.com/achael/eht-imaging/tree/dev>`_. To use the very latest (unstable) code, checkout the dev branch, change to the main eht-imaging directory, and run:

.. code-block:: bash

    pip install .

Installing with pip will install the required libraries automatically (`numpy <http://www.numpy.org/>`_, `scipy <http://www.scipy.org/>`_, `matplotlib <http://www.matplotlib.org/>`_, `astropy <http://www.astropy.org/>`_, `finufft <https://github.com/flatironinstitute/finufft>`_, `skyfield <https://rhodesmill.org/skyfield/>`_, `h5py <http://www.h5py.org/>`_, `networkx <https://networkx.github.io/>`_, `requests <http://docs.python-requests.org/en/master/>`_, and `future <http://pypi.python.org/pypi/future>`_).


Optional Dependencies
---------------------
Certain functions require packages not in the default install:

- `pandas <http://www.pandas.pydata.org/>`_ for the legacy ``ehtim.statistics.dataframes`` utilities and the flag-file CSV reader / scan-id binning helpers in ``Obsdata``
- `paramsurvey <https://github.com/wumpus/paramsurvey>`_ for the parameter-survey utilities in ``ehtim.survey``

Install individually as needed:

.. code-block:: bash

    pip install pandas paramsurvey

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
