
Python API
==========

The Python API enables maximum flexibility when using mokapot. It also aids in
making analyses reproducible by easily integrating into Jupyter notebooks and
Python scripts.

Read PSMs using the :py:func:`~mokapot.read_pin()` or
:py:func:`~mokapot.read_pepxml()` functions for files in the Percolator
tab-delimited format or PepXML format, respectively. Once a collection of PSMs
has been read, the :py:func:`~mokapot.brew()` function will apply the mokapot
algorithm to learn models from the PSMs and assign confidence estimates based on
their new scores. Alternatively, the
:py:meth:`~mokapot.dataset.LinearPsmDataset.assign_confidence()` method will
assign confidence estimates to PSMs based on the best feature, which is often
the primary score from the database search engine.

Alternatively, PSMs that are already represented in a
:py:class:`pandas.DataFrame` can be directly used to create a
:py:class:`~mokapot.dataset.LinearPsmDataset`.

Finally, custom machine learning models can be created using the
:py:class:`mokapot.model.Model` class.

.. note::

   Set your NumPy random seed to ensure reproducibility:

   .. code-block::

      import numpy as np
      np.random.seed(42)

   In a future release, we will update mokapot to use the `new NumPy random
   sampling API
   <https://numpy.org/doc/stable/reference/random/index.html?highlight=random%20sampling%20numpy%20random#module-numpy.random>`_.


.. toctree::
   :maxdepth: 1
   :hidden:
   :titlesonly:

   Overview <self>
   functions.rst
   model.rst
   dataset.rst
   confidence.rst
   proteins.rst

Functions
---------
.. currentmodule:: mokapot

Primary Functions
*****************
.. autosummary::
   :nosignatures:

   read_pin
   read_pepxml
   read_fasta
   brew
   to_txt
   to_flashlfq

Utility Functions
*****************
.. autosummary::
   :nosignatures:

   save_model
   load_model
   read_percolator
   plot_qvalues
   make_decoys
   digest


Machine Learning Models
-----------------------

Use a model that emulates the Linear support vector machine used by Percolator
or create a custom model from anything with a Scikit-Learn interface.

.. currentmodule:: mokapot.model
.. autosummary::
   :nosignatures:

   PercolatorModel
   Model


Collections of PSMs
-------------------

PSMs can be parsed from Percolator tab-delimited files, PepXML files, or
directly from a :py:class:`pandas.DataFrame`.

.. currentmodule:: mokapot.dataset
.. autosummary::
   :nosignatures:

   LinearPsmDataset
   .. CrossLinkedPsmDataset


Confidence Estimates
--------------------

An analysis with mokapot yields two forms of confidence estimates---q-values and
posterior error probabilities (PEPs)---at various levels: PSMs, peptides, and
optionally, proteins.

.. currentmodule:: mokapot.confidence
.. autosummary::
   :nosignatures:

   LinearConfidence
   .. CrossLinkedConfidence


Protein Sequences
-----------------

To calculate protein-level confidence estimates, mokapot needs the original
protein sequences and digestion parameters used for the database search. These
are created using the :py:func:`mokapot.read_fasta()` function, which return a
:py:class:`Proteins` object. :py:class:`Proteins` objects store the mapping of
peptides to the proteins that may have generated them and the mapping of
target protein sequences to their corresponding decoys.

.. currentmodule:: mokapot.proteins
.. autosummary::
   :nosignatures:

   Proteins
