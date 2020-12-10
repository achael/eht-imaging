import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if __name__ == "__main__":
    setup(name="ehtim",

          version = "1.2.2",

          author = "Andrew Chael",
          author_email = "achael@princeton.edu",
          description = "Imaging, analysis, and simulation software for radio interferometry",
          license = "GPLv3",
          keywords = "imaging astronomy EHT polarimetry",
          url = "https://github.com/achael/eht-imaging",
          download_url = "https://github.com/achael/eht-imaging/archive/v1.2.1.tar.gz",
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
                            "future"],
          classifiers=[
            'Development Status :: 3 - Alpha',     
            'Intended Audience :: Developers',    
            'Topic :: Software Development :: Build Tools',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 2.7',
          ],

         )

