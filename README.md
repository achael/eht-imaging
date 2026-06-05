# ehtim (eht-imaging)

[![PyPI version](https://img.shields.io/pypi/v/ehtim.svg)](https://pypi.org/project/ehtim/)
[![License: GPLv3](https://img.shields.io/github/license/achael/eht-imaging.svg)](https://github.com/achael/eht-imaging/blob/main/LICENSE.txt)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://achael.github.io/eht-imaging/)
[![DOI](https://zenodo.org/badge/42943499.svg)](https://doi.org/10.5281/zenodo.2614008)

[![CI](https://github.com/achael/eht-imaging/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/achael/eht-imaging/actions/workflows/ci.yml)
[![codecov](https://img.shields.io/codecov/c/github/achael/eht-imaging/dev)](https://codecov.io/gh/achael/eht-imaging/branch/dev)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Python modules for simulating and manipulating VLBI data and producing images with regularized maximum likelihood methods. This version is an early release so please raise an issue, submit a pull request, or email achael@outlook.com if you have trouble or need help for your application.

The package contains several primary classes for loading, simulating, and manipulating VLBI data. The main classes are the `Image`, `Movie`, `Array`, `Obsdata`, `Imager`, and `Caltable` classes, which provide tools for loading images and data, producing simulated data from realistic u-v tracks,  calibrating, inspecting, and  plotting data, and producing images from data sets in various polarizations using various data terms and regularizing functions.

## Installation

The latest development version ([1.4](https://github.com/achael/eht-imaging/releases/tag/v1.4)) requires Python 3.11 or 3.12. Simply install pip and run

```bash
pip install ehtim
```

Incremental updates are developed on the [dev branch](https://github.com/achael/eht-imaging/tree/dev). To use the very latest (unstable) code, checkout the dev branch, change to the main eht-imaging directory, and run:

```bash
pip install .
```

Installing with pip will install the required libraries automatically ([numpy](http://www.numpy.org/), [scipy](http://www.scipy.org/), [matplotlib](http://www.matplotlib.org/), [astropy](http://www.astropy.org/), [finufft](https://github.com/flatironinstitute/finufft), [skyfield](https://rhodesmill.org/skyfield/), [h5py](http://www.h5py.org/), [networkx](https://networkx.github.io/), [requests](http://docs.python-requests.org/en/master/), and [future](http://pypi.python.org/pypi/future)).

## Optional Dependencies
Certain functions require packages not in the default install:

- [pandas](http://www.pandas.pydata.org/) for the legacy `ehtim.statistics.dataframes` utilities and the flag-file CSV reader / scan-id binning helpers in `Obsdata`
- [paramsurvey](https://github.com/wumpus/paramsurvey) for the parameter-survey utilities in `ehtim.survey`

Install individually as needed:

```bash
pip install pandas paramsurvey
```

## Documentation and Tutorials
Documentation is  [here](https://achael.github.io/eht-imaging).

A tutorial jupyter notebook for imaging can be found in the repo at [tutorials/ehtim_tutorial.ipynb](https://github.com/achael/eht-imaging/blob/main/tutorials/ehtim_tutorial.ipynb)

[Slides](https://docs.google.com/presentation/d/1A0y9omYI2ueSUa6_t5reylBhw6eiLwjqDzw-HUOk8Ac/edit?usp=sharing) for the included tutorial walk through the basic steps of reconstructing EHT images with the code

Scripts in the examples directory contain several older simple example workflows but have not been recently validated.

## Citation
If you use ehtim in your publication, please cite [Chael+ 2018](http://adsabs.harvard.edu/abs/2018ApJ...857...23C).

The latest version is also available as a static doi on [Zenodo](https://zenodo.org/badge/latestdoi/42943499).

## Selected publications that use ehtim

Let us know if you use eht-imaging in your publication and we'll list it here!

- High-Resolution Linear Polarimetric Imaging for the Event Horizon Telescope, [Chael et al. 2016](https://arxiv.org/abs/1605.06156)

- Computational Imaging for VLBI Image Reconstruction, [Bouman et al. 2016](http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Bouman_Computational_Imaging_for_CVPR_2016_paper.html)

- Stochastic Optics: A Scattering Mitigation Framework for Radio Interferometric Imaging, [Johnson 2016](https://arxiv.org/abs/1610.05326)

- Reconstructing Video from Interferometric Measurements of Time-Varying Sources, [Bouman et al. 2017](https://arxiv.org/abs/1711.01357)

- Dynamical Imaging with Interferometry, [Johnson et al. 2017](https://arxiv.org/abs/1711.01286)

- Interferometric Imaging Directly with Closure Phases and Closure Amplitudes, [Chael et al. 2018](https://arxiv.org/abs/1803.07088)

- A Model for Anisotropic Interstellar Scattering and its Application to Sgr A*, [Psaltis et al. 2018](https://arxiv.org/abs/1805.01242)

- The Currrent Ability to Test Theories of Gravity with Black Hole Shadows, [Mizuno et al. 2018](https://arxiv.org/abs/1804.05812)

- The Scattering and Intrinsic Structure of Sagittarius A* at Radio Wavelengths, [Johnson et al. 2018](https://arxiv.org/abs/18008.08966)

- Testing GR with the Black Hole Shadow Size and Asymmetry of Sagittarius A*: Limitations from Interstellar Scattering, [Zhu et al. 2018](https://arxiv.org/abs/1811.02079)

- The Size, Shape, and Scattering of Sagittarius A* at 86 GHz: First VLBI with ALMA, [Issaoun et al. 2019a](https://arxiv.org/abs/1901.06226)

- First M87 Event Horizon Telescope Results IV: Imaging the Central Supermassive Black Hole, [EHTC et al. 2019](https://arxiv.org/abs/1906.11241)

- EHT-HOPS Pipeline for Millimeter VLBI Data Reduction, [Blackburn et al. 2019](https://arxiv.org/pdf/1903.08832)

- Discriminating Accretion States via Rotational Symmetry in Simulated Polarimetric Images of M87, [Palumbo et al. 2020](https://arxiv.org/pdf/2004.01751.pdf)

- SYMBA: An end-to-end VLBI synthetic data generation pipeline, [Roelofs et al. 2020](https://arxiv.org/pdf/2004.01161.pdf)

- Monitoring the Morphology of M87* in 2009-2017 with the Event Horizon Telescope, [Wielgus et al. 2020](https://arxiv.org/pdf/2009.11842)

- EHT imaging of the archetypal blazar 3C 279 at extreme 20 microarcsecond resolution, [Kim et al. 2020](https://www.aanda.org/articles/aa/pdf/2020/08/aa37493-20.pdf)

- Verification of Radiative Transfer Schemes for the EHT, [Gold et al. 2020](https://iopscience.iop.org/article/10.3847/1538-4357/ab96c6/pdf)

- Closure Traces: Novel Calibration-insensitive Quantities for Radio Astronomy, [Broderick and Pesce. 2020](https://iopscience.iop.org/article/10.3847/1538-4357/abbd9d/pdf)

- Evaluation of New Submillimeter VLBI Sites for the Event Horizon Telescope, [Raymond et al. 2021](https://iopscience.iop.org/article/10.3847/1538-3881/abc3c3/pdf)

- Imaging VGOS Observations and Investigating Source Structure Effects, [Xu et al. 2021](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020JB021238)

- A D-term Modeling Code (DMC) for Simultaneous Calibration and Full-Stokes Imaging of VLBI Data, [Pesce et al. 2021](https://iopscience.iop.org/article/10.3847/1538-3881/abe3f8/pdf)

- Using space-VLBI to probe gravity around Sgr A*, [Fromm et al. 2021](https://www.aanda.org/articles/aa/pdf/2021/05/aa37335-19.pdf)

- Persistent Non-Gaussian Structure in the Image of Sagittarius A* at 86 GHz, [Issaoun et al. 2021](https://iopscience.iop.org/article/10.3847/1538-4357/ac00b0/pdf)

- First M87 Event Horizon Telescope Results. VII. Polarization of the Ring, [EHTC et al. 2021](https://iopscience.iop.org/article/10.3847/2041-8213/abe71d/pdf)

- Event Horizon Telescope observations of the jet launching and collimation in Centaurus A, [Janssen et al. 2021](https://www.nature.com/articles/s41550-021-01417-w.pdf)

- RadioAstron discovers a mini-cocoon around the restarted parsec-scale jet in 3C 84 [Savolainen et al. 2021](https://arxiv.org/pdf/2111.04481.pdf)

- Unravelling the Innermost Jet Structure of OJ 287 with the First GMVA+ALMA Observations, [Zhao et al. 2022](https://arxiv.org/pdf/2205.00554.pdf)

- First Sagittarius A* Event Horizon Telescope Results. III: Imaging of the Galactic Center Supermassive Black Hole, [EHTC et al. 2022](https://arxiv.org/pdf/2311.09479.pdf)

- Resolving the Inner Parsec of the Blazar J1924-2914 with the Event Horizon Telescope, [Issaoun et al. 2022](https://arxiv.org/pdf/2208.01662.pdf)

- The Event Horizon Telescope Image of the Quasar NRAO 530, [Jorstad et al. 2023](https://arxiv.org/pdf/2302.04622.pdf)

- First M87 Event Horizon Telescope Results. IX. Detection of Near-horizon Circular Polarization, [EHTC et al. 2023](https://arxiv.org/pdf/2311.10976.pdf)

- Filamentary structures as the origin of blazar jet radio variability, [Fuentes et al. 2023](https://arxiv.org/pdf/2311.01861.pdf)

- The persistent shadow of the supermassive black hole of M 87. I. Observations, calibration, imaging, and analysis, [EHTC 2023](https://www.aanda.org/articles/aa/pdf/2024/01/aa47932-23.pdf)

- Fundamental Physics Opportunities with the Next-Generation Event Horizon Telescope, [Ayzenberg et al. 2023](https://arxiv.org/abs/2312.02130)

- Parsec-scale evolution of the gigahertz-peaked spectrum quasar PKS 0858-279, [Kosogorov et al. 2024](https://arxiv.org/pdf/2401.03603.pdf)

- Atmospheric Limitations for High-frequency Ground-based Very Long Baseline Interferometry, [Pesce et a. 2024](https://arxiv.org/abs/2404.01482)

- Evidence of a toroidal magnetic field in the core of 3C 84, [Paraschos et al. 2024](https://arxiv.org/abs/2405.00097)

- First Sagittarius A* Event Horizon Telescope Results. VII. Polarization of the Ring, [EHTC et al. 2024](https://iopscience.iop.org/article/10.3847/2041-8213/ad2df0)

- Towards an astronomical use of new-generation geodetic observations. I. From the correlator to full-polarization images, [Perez-Diez et al. 2024,](https://arxiv.org/abs/2406.12509)

- Discovery of Limb Brightening in the Parsec-scale Jet of NGC 315 through Global Very Long Baseline Interferometry Observations and Its Implications for Jet Models, [Park et al. 2024](https://arxiv.org/abs/2408.09069)

- Kilogauss magnetic field and jet dynamics in the quasar NRAO 530, [Lisakov et al. 2024](https://arxiv.org/abs/2411.03446)

- Evolution, speed, and precession of the parsec-scale jet in the 3C 84 radio galaxy, [Foschi et al. 2024](https://arxiv.org/pdf/2412.09215)

- Origin of the ring ellipticity in the black hole images of M87*, [Dahale et al. 2025](https://arxiv.org/pdf/2505.10333)

- Revealing a ribbon-like jet in OJ 287 with RadioAstron, [Traianou et al. 2025](https://arxiv.org/pdf/2508.01747)

- Revisiting 3C 279 jet morphology with space VLBI at 26 microarcsecond resolution, [Toscano et al. 2025](https://arxiv.org/pdf/2509.21987)

- Horizon-scale variability of M87* from 2017-2021 EHT obesrvations, [EHTC et al. 2025](https://arxiv.org/pdf/2509.24593)

- Probing the disk-jet coupling in M 87, [Saiz-Perez et al. 2025](https://arxiv.org/pdf/2511.15482)

- Probing jet base emission of M87* with the 2021 Event Horizon Telescope observations, [Saurabh et al.](https://arxiv.org/pdf/2512.08970)

- Radio properties of the quasi-periodic eruption source RXJ1301.9+2747 at parsec scales, [von Fellenberg et al.](https://arxiv.org/pdf/2511.14863)

- Spatially resolved polarization swings in the supermassive binary black hole candidate OJ 287 with first Event Horizon Telescope observations, [Gomez et al. 2026](https://www.aanda.org/articles/aa/pdf/2026/01/aa55831-25.pdf)

- Locating the missing large-scale emission in the jet of M87* with short EHT baselines, [Georgiev et al. 2026](https://arxiv.org/pdf/2601.13356)

- Evidence of a non-equipartition energy regime in 1803+784, [Perez-Diez et al. 2026](https://arxiv.org/pdf/2602.20746)

- Where within the 3c 84 jet are gamma-rays produced? [Paraschos et al. 2026](https://arxiv.org/pdf/2603.22403)

- Ring Asymmetry and Spin in M87*, [Bernshteyn et al. 2026](https://iopscience.iop.org/article/10.3847/1538-4357/ae34af)

## License
ehtim is licensed under GPLv3. See LICENSE.txt for more details.
