"""The :py:class:`LinearPsmDataset` class is used to define a collection
peptide-spectrum matches. The :py:class:`LinearPsmDataset` class is suitable
for most types of data-dependent acquisition proteomics experiments.

Although the class can be constructed from a :py:class:`pandas.DataFrame`, it
is often easier to load the PSMs directly from a file in the `Percolator
tab-delimited format
<https://github.com/percolator/percolator/wiki/Interface#tab-delimited-file-format>`_
(also known as the Percolator input format, or "PIN") using the
:py:func:`~mokapot.read_pin()` function or from a PepXML file using the
:py:func:`~mokapot.read_pepxml()` function. If protein-level confidence
estimates are desired, make sure to use the
:py:meth:`~LinearPsmDataset.add_proteins()` method.

One of more instance of this class are required to use the
:py:func:`~mokapot.brew()` function.

"""

from .base import PsmDataset, calibrate_scores, update_labels
from .linear_psm import LinearPsmDataset
from .on_disk import OnDiskPsmDataset

__all__ = [
    "OnDiskPsmDataset",
    "LinearPsmDataset",
    "PsmDataset",
    "calibrate_scores",
    "update_labels",
]
