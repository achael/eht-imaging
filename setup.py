import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if __name__ == "__main__":
    setup(name="ehtim",

          version = "1.2",

          author = "Andrew Chael",
          author_email = "achael@cfa.harvard.edu",
          description = ("Python code to load, simulate, and manipulate VLBI "+
                         "datasets and a collection of imaging functions in "+
                         "total intensity and polarization."),
          license = "GPLv3",
          keywords = "imaging astronomy EHT polarimetry",
          url = "https://github.com/achael/eht-imaging",
          packages = ["ehtim",
                      "scripts",
                      "ehtim.calibrating",
                      "ehtim.imaging",
                      "ehtim.io",
                      "ehtim.observing",
                      "ehtim.plotting",
                      "ehtim.features",
                      "ehtim.scattering",
                      "ehtim.statistics"],
          scripts=["scripts/calibrate.py",
                   "scripts/cleanup.py",
                   "scripts/cli_blur_comp.py",
                   "scripts/gendata.py",
                   "scripts/imaging.py",
                   "scripts/imgsum.py",
                   "scripts/verify_gradients.py"],
          long_description=read('README.rst'),
          install_requires=["numpy",
                            "scipy",
                            "astropy",
                            "matplotlib",
                            "ephem",
                            "h5py",
                            "pandas",
                          # "pynfft",   # optional (but highly recommended)
                          # "networkx", # optional, only needed if using image_agreements()
                          # "requests", # optional; only needed if using dynamical imaging
                          # "scikit-image", #optional; only needed for hough transforms  
                            "future"]
         )

