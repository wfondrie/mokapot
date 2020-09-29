.. image:: https://badge.fury.io/py/mokapot.svg
   :target: https://badge.fury.io/py/mokapot
   :alt: PyPI

.. image:: https://github.com/wfondrie/mokapot/workflows/tests/badge.svg
   :target: https://github.com/wfondrie/mokapot/actions?query=workflow%3Atests
   :alt: tests

.. image:: https://readthedocs.org/projects/mokapot/badge/?version=latest
   :target: https://mokapot.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://cdn.rawgit.com/syl20bnr/spacemacs/442d025779da2f62fc86c2082703697714db6514/assets/spacemacs-badge.svg
   :target: http://spacemacs.org
   :alt: Built with Spacemacs


Getting Started
---------------

**mokapot** uses a semi-supervised learning approach to enhance peptide
detection from bottom-up proteomics experiments. It takes features describing
putative peptide-spectrum matches (PSMs) from database search engines as input,
re-scores them, and yields statistical measures---confidence estimates, such as
q-values and posterior error probabilities---indicating their quality.


Introduction
------------

Nearly every analysis of a bottom-up proteomics begins by using a search engine
to assign putative peptides to the acquired mass spectra, yielding a collection
of peptide-spectrum matches (PSMs). However, post-processing tools such as   
`Percolator <http://percolator.ms>`_ and `PeptideProphet
<http://peptideprophet.sourceforge.net/>`_ have proven invaluable for improving
the sensitivity of peptide detection and providing consistent statistical
frameworks for interpreting these detections.

mokapot is fundamentally a Python implementation of the semi-supervised learning
algorithm introduced by Percolator. We developed mokapot to add additional
flexibility for our analyses, whether to try something experimental---such as
swapping Percolator's linear support vector machine classifier for a non-linear,
gradient boosting classifier---or to train a joint model across experiments
while retaining valid, per-experiment confidence estimates. We designed mokapot
to be extensible and support the analysis of additional types of proteomics
data, such as cross-linked peptides from cross-linking mass spectrometry
experiments. mokapot offers basic functionality from the command line, but using
mokapot as a Python package unlocks maximum flexibility. 

Ready to try mokapot for your analyses? See below for details on how to install
and use mokapot. Additionally, check out the :doc:`vignettes/index` for other
examples of mokapot in action.


Installation
------------

Before you can install and use mokapot, you'll need to have Python 3.6+
installed. If you think it may be installed, you can check with:

.. code-block:: bash

   $ python3 --version

If you need to install Python, we recommend using the `Anaconda Python
distribution <https://www.anaconda.com/products/individual>`_. This distribution
comes with most of the mokapot dependencies installed and provides the conda
package manager.  

mokapot also depends on several Python packages:

- `NumPy <https://numpy.org/>`_
- `pandas <https://pandas.pydata.org/>`_
- `Matplotlib <https://matplotlib.org/>`_
- `scikit-learn <https://scikit-learn.org/stable/>`_
- `Numba <http://numba.pydata.org/>`_
- `triqler <https://github.com/statisticalbiotechnology/triqler>`_


We recommend using `pip` to install mokapot. Missing dependencies will also
be installed automatically:

.. code-block:: bash

   $ pip3 install mokapot


Basic Usage
-----------

Before you can use mokapot, you need PSMs assigned by a search engine available
in the `Percolator tab-delimited file format
<https://github.com/percolator/percolator/wiki/Interface#tab-delimited-file-format>`_
(often referred to as the Percolator input, or "PIN", file format). These files can
be generated from various search engines, such as `Comet
<http://comet-ms.sourceforge.net/>`_ or `Tide
<http://crux.ms/commands/tide-search.html>`_ (which is part of the `Crux mass
spectrometry toolkit <http://crux.ms>`_).

If you need an example file to get started with, a selection of PSMs from
Hogrebe et al. [1]_ is available to download from the mokapot
repository, `phospho_rep1.pin
<https://github.com/wfondrie/mokapot/raw/master/tests/data/phoshpo_rep1.pin>`_. This
is the file we'll use in the examples below.


Run **mokapot** from the command line
#####################################

Simple mokapot analyses can be performed from the command line:

.. code-block:: bash

   $ mokapot phospho_rep1.pin

That's it. Your results will be saved in your working directory as two
tab-delimited files, `mokapot.psms.txt` and `mokapot.peptides.txt`. For a full
list of parameters, see the :doc:`Command Line Interface <cli>`.


Use **mokapot** as a Python package 
###################################

It is easy to run the above analysis from the Python interpreter as well. First
start up the Python interpreter:

.. code-block:: bash

   $ python3

Then conduct your mokapot analysis:

.. code-block:: Python

   >>> import mokapot
   >>> psms = mokapot.read_pin("phospho_rep1.pin")
   >>> results, models = mokapot.brew(psms)
   >>> results.to_txt()

This is great for when your upstream and/or downstream analyses are also
conducted in Python. Additionally, a great deal more flexibility is available
when using mokapot from the Python interpreter. For more details, see the
:doc:`Python API <api/index>`.


 .. [1] Hogrebe, Alexander, et al. "Benchmarking common quantification strategies
       for large-scale phosphoproteomics." Nature communications 9.1 (2018):
       1-13.

 
.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: mokapot
   :titlesonly:

   self
   vignettes/index.rst
   api/index.rst
   cli.rst
   FAQ <faq.rst>
