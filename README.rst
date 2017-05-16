eht-imaging
============

Python modules for simulating and manipulating VLBI data and producing images with Maximum Entropy methods. This version is an early release so please submit a pull request or email achael@cfa.harvard.edu if you have trouble or
need help for your application.

Installation
------------
To install the ehtim library run
pyton setup.py --install

Documentation
-------------

Documentation is in progress, but here are some ways to learn to use the code: 
  1. the file example.py has a series of sample commands to load an image and array, generate data, and produce an image. 
  2. Slides from the EHT2016 data generation and imaging workshop https://www.dropbox.com/s/7533ucj8bt54yh7/Bouman_Chael.pdf?dl=0 contain a tutorial on generating data with the vlbiimaging website (http://vlbiimaging.csail.mit.edu), loading into the library, and producing an image. Note that this presentation used
a previous version of the code -- function names and prefixes need to be updated

Below is a non-exhaustive list of the imaging applications included in eht-imaging:https://arxiv.org/abs/1605.06156

PolMEM
------------
Produces polarimetric VLBI images using robust polarimetric ratio data products and entropy+total variation priors. (Andrew A. Chael et al 2016 ApJ 829 11)

ArXiv: https://arxiv.org/abs/1605.06156

CHIRP
------------

Python code to produce intensity images using patch-prior regularizers. These methods are described in detail in the following paper: 

CVPR Paper: http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Bouman_Computational_Imaging_for_CVPR_2016_paper.html

Detailed derivations and additional results can be found in the accompanying supplemental material:

Supplemental Material: http://vlbiimaging.csail.mit.edu/static/papers/CHIRPsupp.pdf

Please email klbouman@mit.edu if you have trouble or need help for your application.

Acknowledgements
----------------
The oifits_new.py code used for writing/reading oifits files is a slightly modified version of Paul Boley's package at http://astro.ins.urfu.ru/pages/~pboley/oifits/ The oifits read/write functionality is still being tested and may not work with all versions of python or astropy.io.fits.

The jdcal.py module is from Prasanth Nair http://github.com/phn/jdcal. 

