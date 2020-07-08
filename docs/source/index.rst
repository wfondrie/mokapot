Welcome
-------

**mokapot** uses a semi-supervised learning approach to enhance peptide
detection from bottom-up proteomics experiments. 


Introduction
------------

Nearly every analysis of a bottom-up proteomics begins by using a search engine
to assign putative peptides to the acquired mass spectra, yielding a collection
of peptide-spectrum matches (PSMs). However, post-processing tools such as   
`Percolator <http://percolator.ms>`_ and `PeptideProphet
<http://peptideprophet.sourceforge.net/>`_ have proven invaluable for improving
the sensitivity of peptide detection and providing consistent statistical
frameworks for interpreting these detections.

**mokapot** is fundamentally a Python implementation of the semi-supervised
learning algorithm introduced by Percolator. We developed **mokapot** to add
additional flexibility to our analyses, whether to try something
experimental---such as swapping Percolator's linear support vector machine model
for a non-linear, gradient boosting classifier---or to train a joint model
across experiments while retaining valid per-experiment confidence estimates.
Furthermore, we designed **mokapot** to be extensible and support the analysis
of additional types of proteomics data, such as cross-linked peptides from 
cross-linking mass spectrometry experiments.

Ready to try **mokapot** for your analyses? Head over to the
:doc:`getting_started` page to learn about installation and see a basic example.
Additionally, check out the :doc:`vignettes/index` for other examples of
**mokapot** in action.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: mokapot
   :titlesonly:

   Overview <self>
   getting_started.rst
   vignettes/index.rst
   api/index.rst
   cli.rst
