"""
Setup the mokapot package.
"""
import setuptools

with open("README.md", "r") as readme:
    LONG_DESC = readme.read()

DESC = ("Semi-supervised learning for peptide detection by pretrained models")

CATAGORIES = ["Programming Language :: Python :: 3",
              "License :: OSI Approved :: Apache Software License",
              "Operating System :: OS Independent",
              "Topic :: Scientific/Engineering :: Bio-Informatics"]

setuptools.setup(
    name="mokapot",
    author="William E. Fondrie",
    author_email="fondriew@gmail.com",
    description=DESC,
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    url="https://github.com/wfondrie/mokapot",
    packages=setuptools.find_packages(),
    license="Apache 2.0",
    entry_points={"console_scripts": ["mokapot = mokapot.mokapot:main"]},
    classifiers=CATAGORIES,
    install_requires=["numpy",
                      "pandas",
                      "scikit-learn",
                      "numba",
                      "triqler",
                      "matplotlib"],
    use_scm_version=True,
    setup_requires=["setuptools-scm"],
    extras_require={
        "docs":  ["numpydoc",
                  "sphinx-argparse"]
    }
)
