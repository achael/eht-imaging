PolMEM
============

Python modules for simulating and manipulating VLBI data and producing polarimetric images with Maximum Entropy methods. This version represents a very early release so please submit a pull request or email achael@cfa.harvard.edu if you have trouble or
need help for your application.

Documentation is in progress - the file example.py has a series of sample commands to load an image and array, generate data, and produce an image. Slides from the EHT2016 data generation and imaging workshop here https://www.dropbox.com/s/7533ucj8bt54yh7/Bouman_Chael.pdf?dl=0 contains a tutorial on generating data with an independent simulator (http://vlbiimaging.csail.mit.edu), loading it into the library, and producing an image. 

The oifits_new.py code used for writing/reading oifits files is a slightly modified version of Paul Boley's package at http://astro.ins.urfu.ru/pages/~pboley/oifits/ The oifits read/write functionality is still being tested and may not work with all versions of python or astropy.io.fits.

CHIRP
============

Python code to produce intensity images using patch-prior regularizers. These methods are described in detail in the following paper: 

CVPR Paper: http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Bouman_Computational_Imaging_for_CVPR_2016_paper.html

Detailed derivations and additional results can be found in the accompanying supplemental material:

Supplemental Material: http://vlbiimaging.csail.mit.edu/static/papers/CHIRPsupp.pdf

Please email klbouman@mit.edu if you have trouble or need help for your application.
