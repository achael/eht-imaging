ehtim (eht-imaging)
===================
.. image:: https://zenodo.org/badge/42943499.svg
   :target: https://zenodo.org/badge/latestdoi/42943499

Python modules for simulating and manipulating VLBI data and producing images with regularized maximum likelihood methods. This version is an early release so please raise an issue, submit a pull request, or email achael@princeton.edu if you have trouble or need help for your application.

The package contains several primary classes for loading, simulating, and manipulating VLBI data. The main classes are the ``Image``, ``Movie``, ``Array``, ``Obsdata``, ``Imager``, and ``Caltable`` classes, which provide tools for loading images and data, producing simulated data from realistic u-v tracks,  calibrating, inspecting, and  plotting data, and producing images from data sets in various polarizations using various data terms and regularizing functions.

Installation
------------

The latest stable version (`1.2.8 <https://github.com/achael/eht-imaging/releases/tag/v1.2.7>`_) is available on `PyPi <https://pypi.org/project/ehtim/>`_. Simply install pip and run

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

Citation
--------------------------------
If you use ehtim in your publication, please cite `Chael+ 2018 <http://adsabs.harvard.edu/abs/2018ApJ...857...23C>`_.

The latest version is also available as a static doi on `Zenodo <https://zenodo.org/badge/latestdoi/42943499>`_.

Selected publications that use ehtim
------------------------------------

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

- Testing GR with the Black Hole Shadow Size and Asymmetry of Sagittarius A*: Limitations from Interstellar Scattering, `Zhu et al. 2018 <https://arxiv.org/abs/1811.02079>`_

- The Size, Shape, and Scattering of Sagittarius A* at 86 GHz: First VLBI with ALMA, `Issaoun et al. 2019a <https://arxiv.org/abs/1901.06226>`_

- First M87 Event Horizon Telescope Results IV: Imaging the Central Supermassive Black Hole, `EHTC et al. 2019 <https://arxiv.org/abs/1906.11241>`_

- EHT-HOPS Pipeline for Millimeter VLBI Data Reduction, `Blackburn et al. 2019 <https://arxiv.org/pdf/1903.08832>`_

- Discriminating Accretion States via Rotational Symmetry in Simulated Polarimetric Images of M87, `Palumbo et al. 2020 <https://arxiv.org/pdf/2004.01751.pdf>`_

- SYMBA: An end-to-end VLBI synthetic data generation pipeline, `Roelofs et al. 2020 <https://arxiv.org/pdf/2004.01161.pdf>`_

- Monitoring the Morphology of M87* in 2009-2017 with the Event Horizon Telescope, `Wielgus et al. 2020 <https://arxiv.org/pdf/2009.11842>`_

- EHT imaging of the archetypal blazar 3C 279 at extreme 20 microarcsecond resolution, `Kim et al. 2020 <https://www.aanda.org/articles/aa/pdf/2020/08/aa37493-20.pdf>`_

- Verification of Radiative Transfer Schemes for the EHT, `Gold et al. 2020 <https://iopscience.iop.org/article/10.3847/1538-4357/ab96c6/pdf>`_

- Closure Traces: Novel Calibration-insensitive Quantities for Radio Astronomy, `Broderick and Pesce. 2020 <https://iopscience.iop.org/article/10.3847/1538-4357/abbd9d/pdf>`_

- Evaluation of New Submillimeter VLBI Sites for the Event Horizon Telescope, `Raymond et al. 2021 <https://iopscience.iop.org/article/10.3847/1538-3881/abc3c3/pdf>`_

- Imaging VGOS Observations and Investigating Source Structure Effects, `Xu et al. 2021 <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020JB021238>`_

- A D-term Modeling Code (DMC) for Simultaneous Calibration and Full-Stokes Imaging of VLBI Data, `Pesce et al. 2021 <https://iopscience.iop.org/article/10.3847/1538-3881/abe3f8/pdf>`_

- Using space-VLBI to probe gravity around Sgr A*, `Fromm et al. 2021 <https://www.aanda.org/articles/aa/pdf/2021/05/aa37335-19.pdf>`_

- Persistent Non-Gaussian Structure in the Image of Sagittarius A* at 86 GHz, `Issaoun et al. 2021 <https://iopscience.iop.org/article/10.3847/1538-4357/ac00b0/pdf>`_

- First M87 Event Horizon Telescope Results. VII. Polarization of the Ring, `EHTC et al. 2021 <https://iopscience.iop.org/article/10.3847/2041-8213/abe71d/pdf>`_

- Event Horizon Telescope observations of the jet launching and collimation in Centaurus A, `Janssen et al. 2021 <https://www.nature.com/articles/s41550-021-01417-w.pdf>`_

- RadioAstron discovers a mini-cocoon around the restarted parsec-scale jet in 3C 84 `Savolainen et al. 2021 <https://arxiv.org/pdf/2111.04481.pdf>`_

- Unravelling the Innermost Jet Structure of OJ 287 with the First GMVA+ALMA Observations, `Zhao et al. 2022 <https://arxiv.org/pdf/2205.00554.pdf>`_

- First Sagittarius A* Event Horizon Telescope Results. III: Imaging of the Galactic Center Supermassive Black Hole, `EHTC et al. 2022 <https://arxiv.org/pdf/2311.09479.pdf>`_

- Resolving the Inner Parsec of the Blazar J1924-2914 with the Event Horizon Telescope, `Issaoun et al. 2022 <https://arxiv.org/pdf/2208.01662.pdf>`_

- The Event Horizon Telescope Image of the Quasar NRAO 530, `Jorstad et al. 2023 <https://arxiv.org/pdf/2302.04622.pdf>`_

- First M87 Event Horizon Telescope Results. IX. Detection of Near-horizon Circular Polarization, `EHTC et al. 2023 <https://arxiv.org/pdf/2311.10976.pdf>`_

- Filamentary structures as the origin of blazar jet radio variability, `Fuentes et al. 2023 <https://arxiv.org/pdf/2311.01861.pdf>`_

- The persistent shadow of the supermassive black hole of M 87. I. Observations, calibration, imaging, and analysis, `EHTC 2023 <https://www.aanda.org/articles/aa/pdf/2024/01/aa47932-23.pdf>`_

- Parsec-scale evolution of the gigahertz-peaked spectrum quasar PKS 0858-279, `Kosogorov et al. 2024 <https://arxiv.org/pdf/2401.03603.pdf>`_

oifits Documentation
--------------------

The oifits_new.py file used for reading/writing .oifits files is a slightly modified version of Paul Boley's package `oifits <http://astro.ins.urfu.ru/pages/~pboley/oifits/>`_.  
The oifits read/write functionality in ehtim is still being developed and may not work with all versions of python or astropy.

The documentation is styled after `dfm's projects <https://github.com/dfm>`_ 

License
-------
ehtim is licensed under GPLv3. See LICENSE.txt for more details.
