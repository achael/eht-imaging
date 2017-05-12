import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if __name__ == "__main__":
    setup(
        name="ehtim",
        version = "0.0.1",

        author = "Andrew Chael",
        author_email = "achael@cfa.harvard.edu",
        description = ("Python code to load, simulate, and manipulate VLBI datasets" +
                        "and a collection of imaging functions in total intensity and polarization."),
        license = "MIT",
        keywords = "imaging astronomy EHT polarimetry",
        url = "https://github.com/achael/eht-imaging",
        packages = ["ehtim","ehtim.observing","ehtim.plotting","ehtim.io","ehtim.imaging","ehtim.calib"],
        long_description=read('README.md'),
    )
