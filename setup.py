import os
from setuptools import setup
#read
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if __name__ == "__main__":
    setup(name="ehtim",

          version = "1.2.9",

          author = "Andrew Chael",
          author_email = "achael@princeton.edu",
          description = "Imaging, analysis, and simulation software for radio interferometry",
          long_description=read('README.rst'),
          license = "GPLv3",
          keywords = "imaging astronomy EHT polarimetry",
          url = "https://github.com/achael/eht-imaging",
          download_url = "https://github.com/achael/eht-imaging/archive/v1.2.9.tar.gz",
          packages = ["ehtim",
                      "scripts",
                      "ehtim.calibrating",
                      "ehtim.imaging",
                      "ehtim.io",
                      "ehtim.modeling",
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
                   "scripts/imgsum.py"],

          install_requires=["numpy>=1.24,<2.0",
                            "scipy>=1.9.3,<1.14",
                            "astropy>=5.0.4",
                            "matplotlib>=3.7.3",
                            "skyfield",
                            "h5py",
                            "pandas",
                            "requests",
                            "future",
                            "networkx", 
                            "pynfft; platform_system!='Darwin' or platform_machine!='arm64'",
                            "paramsurvey"
                           ],
          classifiers=[
            'Development Status :: 3 - Alpha',     
            'Intended Audience :: Developers',    
            'Topic :: Software Development :: Build Tools',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Programming Language :: Python :: 3.8',
          ],

         )

