Python API
==========

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


Model
-----
.. currentmodule:: mokapot.model
.. autosummary::
   :nosignatures:

   Model

Datasets
--------
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
