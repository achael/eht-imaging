import os
from setuptools import setup

if __name__ == "__main__":
    import vlbi_imaging_utils
    import maxen
    setup(
        name="PolMEM",
        version = "0.0.1",
        packages = ["vlbi_imaging_utils","maxen"],
        author = "Andrew Chael",
        author_email = "achael@cfa.harvard.edu",
        description = ("Python classes to load, simulate, and manipulate VLBI images and datasets and a collection of Maximum Entropy imaging functions in total intensity and polarization."),
        license = "MIT",
        keywords = "imaging astronomy EHT polarimetry"
    )
