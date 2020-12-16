<img src="https://raw.githubusercontent.com/wfondrie/mokapot/master/static/mokapot_logo_dark.svg" width=300>  

---  
[![conda](https://img.shields.io/conda/vn/bioconda/mokapot?color=green)](http://bioconda.github.io/recipes/mokapot/README.html)
[![PyPI](https://img.shields.io/pypi/v/mokapot?color=green)](https://pypi.org/project/mokapot/)
[![tests](https://github.com/wfondrie/mokapot/workflows/tests/badge.svg)](https://github.com/wfondrie/mokapot/actions?query=workflow%3Atests)
[![docs](https://readthedocs.org/projects/mokapot/badge/?version=latest)](https://mokapot.readthedocs.io/en/latest/?badge=latest)



Fast and flexible semi-supervised learning for peptide detection.  

mokapot is fundamentally a Python implementation of the semi-supervised learning
algorithm first introduced by Percolator. We developed mokapot to add additional
flexibility to our analyses, whether to try something experimental---such as
swapping Percolator's linear support vector machine classifier for a non-linear,
gradient boosting classifier---or to train a joint model across experiments
while retaining valid, per-experiment confidence estimates. We designed mokapot
to be extensible and support the analysis of additional types of proteomics
data, such as cross-linked peptides from cross-linking mass spectrometry
experiments. mokapot offers basic functionality from the command line, but using
mokapot as a Python package unlocks maximum flexibility.

For more information, check out our
[documentation](https://mokapot.readthedocs.io).  

## Citing  
If you use mokapot in your work, please cite:  

> Fondrie, W. E. & Noble, W. S. mokapot: Fast and flexible semi-supervised
> learning for peptide detection. *bioRxiv* (2020)
> doi:10.1101/2020.12.01.407270.

## Installation  

mokapot requires Python 3.6+ and can be installed with pip or conda.  

Using conda:
```
$ conda install -c bioconda mokapot
```

Using pip:
```
$ pip3 install mokapot
```

Additionally, you can install the development version directly from GitHub:  

```
$ pip3 install git+git://github.com/wfondrie/mokapot
```

## Basic Usage  

Before you can use mokapot, you need PSMs assigned by a search engine available
in the [Percolator tab-delimited file
format](https://github.com/percolator/percolator/wiki/Interface#tab-delimited-file-format)
(often referred to as the Percolator input, or "PIN", file format). 

Simple mokapot analyses can be performed at the command line:

```Bash
$ mokapot psms.pin
```

Alternatively, the Python API can be used to perform analyses in the Python
interpreter and affords greater flexibility:

```Python
>>> import mokapot
>>> psms = mokapot.read_pin("psms.pin")
>>> results, models = mokapot.brew(psms)
>>> results.to_txt()
```

Check out our [documentation](https://mokapot.readthedocs.io) for more details
and examples of mokapot in action.


[![Built with Spacemacs](https://cdn.rawgit.com/syl20bnr/spacemacs/442d025779da2f62fc86c2082703697714db6514/assets/spacemacs-badge.svg)](http://spacemacs.org)  
