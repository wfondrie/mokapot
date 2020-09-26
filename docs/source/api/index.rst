Python API
==========

The Python API enables maximum flexibility using mokapot. It also aids in
making analyses reproducible by easily integrating into Jupyter notebooks and
Python scripts.

Basic analyses with mokapot can be conducted using only the :ref:`Primary
Functions`. PSMs not saved in the Percolator tab-delimited format can be loaded
from a :py:class:`~pandas.DataFrame` using the :ref:`Dataset` classes. Finally,
custom models can be created using the :ref:`Model` class.

Using :py:func:`mokapot.brew()` or the
:py:meth:`~mokapot.datasets.LinearPsmDataset.assign_confidence()` method return
objects that contain :ref:`Confidence Estimates`.

.. toctree::
   :maxdepth: 1
   :hidden:
   :titlesonly:

   Overview <self>
   functions.rst
   model.rst
   dataset.rst
   confidence.rst

Functions
---------
.. currentmodule:: mokapot

Primary Functions
*****************
.. autosummary::
   :nosignatures:

   read_pin
   brew

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


Model
-----
.. currentmodule:: mokapot.model
.. autosummary::
   :nosignatures:

   Model
   PercolatorModel

Dataset
-------
.. currentmodule:: mokapot.dataset
.. autosummary::
   :nosignatures:

   LinearPsmDataset
   .. CrossLinkedPsmDataset

Confidence Estimates
--------------------
.. currentmodule:: mokapot.confidence
.. autosummary::
   :nosignatures:

   LinearConfidence
   .. CrossLinkedConfidence
