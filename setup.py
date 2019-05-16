import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if __name__ == "__main__":
    setup(name="ehtim",
          version = "1.1.1",
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
                      "ehtim.scattering",
                      "ehtim.statistics"],
          long_description=read('README.rst'),
          install_requires=["astropy",
                            "ephem",
                            "future",
                            "h5py",
                          # "html",     # optional; only needed if use dynamical imaging
                            "matplotlib",
                            "networkx",
                            "numpy",
                          # "pandas",   # optional; only needed if using statistics
                          # "pynfft",   # optional; only needed if using clean
                          # "requests", # optional; only needed if using dynamical imaging
                            "scipy",
                            "scikit-image"]
    )

