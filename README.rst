ehtim (eht-imaging)
===================
.. image:: https://zenodo.org/badge/42943499.svg
   :target: https://zenodo.org/badge/latestdoi/42943499

Python modules for simulating and manipulating VLBI data and producing images with regularized maximum likelihood methods. This version is an early release so please raise an issue, submit a pull request, or email achael@outlook.com if you have trouble or need help for your application.

The package contains several primary classes for loading, simulating, and manipulating VLBI data. The main classes are the ``Image``, ``Movie``, ``Array``, ``Obsdata``, ``Imager``, and ``Caltable`` classes, which provide tools for loading images and data, producing simulated data from realistic u-v tracks,  calibrating, inspecting, and  plotting data, and producing images from data sets in various polarizations using various data terms and regularizing functions.

Installation
------------

The latest development version (`1.4 <https://github.com/achael/eht-imaging/releases/tag/v1.4>`_) requires Python 3.11 or 3.12. Simply install pip and run

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
--------
If you use ehtim in your publication, please cite `Chael+ 2018 <http://adsabs.harvard.edu/abs/2018ApJ...857...23C>`_.

The latest version is also available as a static doi on `Zenodo <https://zenodo.org/badge/latestdoi/42943499>`_.


Selected publications that use ehtim
------------------------------------

Let us know if you use eht-imaging in your publication and we'll list it here!

- High-Resolution Linear Polarimetric Imaging for the Event Horizon Telescope, `Chael et al. 2016 <https://arxiv.org/abs/1605.06156>`_ 

- Computational Imaging for VLBI Image Reconstruction, `Bouman et al. 2016 <http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Bouman_Computational_Imaging_for_CVPR_2016_paper.html>`_ 

- Stochastic Optics: A Scattering Mitigation Framework for Radio Interferometric Imaging, `Johnson 2016 <https://arxiv.org/abs/1610.05326>`_ 

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

- Fundamental Physics Opportunities with the Next-Generation Event Horizon Telescope, `Ayzenberg et al. 2023 <https://arxiv.org/abs/2312.02130>`_

- Parsec-scale evolution of the gigahertz-peaked spectrum quasar PKS 0858-279, `Kosogorov et al. 2024 <https://arxiv.org/pdf/2401.03603.pdf>`_

- Atmospheric Limitations for High-frequency Ground-based Very Long Baseline Interferometry, `Pesce et a. 2024 <https://arxiv.org/abs/2404.01482>`_

- Evidence of a toroidal magnetic field in the core of 3C 84, `Paraschos et al. 2024 <https://arxiv.org/abs/2405.00097>`_

- First Sagittarius A* Event Horizon Telescope Results. VII. Polarization of the Ring, `EHTC et al. 2024 <https://iopscience.iop.org/article/10.3847/2041-8213/ad2df0>`_

- Towards an astronomical use of new-generation geodetic observations. I. From the correlator to full-polarization images, `Perez-Diez et al. 2024, <https://arxiv.org/abs/2406.12509>`_

- Discovery of Limb Brightening in the Parsec-scale Jet of NGC 315 through Global Very Long Baseline Interferometry Observations and Its Implications for Jet Models, `Park et al. 2024 <https://arxiv.org/abs/2408.09069>`_

- Kilogauss magnetic field and jet dynamics in the quasar NRAO 530, `Lisakov et al. 2024 <https://arxiv.org/abs/2411.03446>`_

- Evolution, speed, and precession of the parsec-scale jet in the 3C 84 radio galaxy, `Foschi et al. 2024 <https://arxiv.org/pdf/2412.09215>`_

- Origin of the ring ellipticity in the black hole images of M87*, `Dahale et al. 2025 <https://arxiv.org/pdf/2505.10333>`_

- Revealing a ribbon-like jet in OJ 287 with RadioAstron, `Traianou et al. 2025 <https://arxiv.org/pdf/2508.01747>`_

- Revisiting 3C 279 jet morphology with space VLBI at 26 microarcsecond resolution, `Toscano et al. 2025 <https://arxiv.org/pdf/2509.21987>`_

- Horizon-scale variability of M87* from 2017-2021 EHT obesrvations, `EHTC et al. 2025 <https://arxiv.org/pdf/2509.24593>`_

- Probing the disk-jet coupling in M 87, `Saiz-Perez et al. 2025 <https://arxiv.org/pdf/2511.15482>`_

- Probing jet base emission of M87* with the 2021 Event Horizon Telescope observations, `Saurabh et al. <https://arxiv.org/pdf/2512.08970>`_

- Radio properties of the quasi-periodic eruption source RXJ1301.9+2747 at parsec scales, `von Fellenberg et al. <https://arxiv.org/pdf/2511.14863>`_

- Spatially resolved polarization swings in the supermassive binary black hole candidate OJ 287 with first Event Horizon Telescope observations, `Gomez et al. 2026 <https://www.aanda.org/articles/aa/pdf/2026/01/aa55831-25.pdf>`_

- Locating the missing large-scale emission in the jet of M87* with short EHT baselines, `Georgiev et al. 2026 <https://arxiv.org/pdf/2601.13356>`_

- Evidence of a non-equipartition energy regime in 1803+784, `Perez-Diez et al. 2026 <https://arxiv.org/pdf/2602.20746>`_

- Where within the 3c 84 jet are gamma-rays produced? `Paraschos et al. 2026 <https://arxiv.org/pdf/2603.22403>`_

- Ring Asymmetry and Spin in M87*, `Bernshteyn et al. 2026 <2601.00394>`_

License
-------
ehtim is licensed under GPLv3. See LICENSE.txt for more details.
