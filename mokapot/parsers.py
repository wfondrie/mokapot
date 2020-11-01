"""
This module contains the parsers for reading in PSMs
"""
import gzip
import logging

import numpy as np
import pandas as pd

from . import utils
from .dataset import LinearPsmDataset


LOGGER = logging.getLogger(__name__)

# Functions -------------------------------------------------------------------
def read_pin(pin_files, group_column=None, to_df=False, copy_data=False):
    """
    Read Percolator input (PIN) tab-delimited files.

    Read PSMs from one or more Percolator input (PIN) tab-delmited
    files, aggregating them into a single
    :py:class:`~mokapot.dataset.LinearPsmDataset`. For
    more details about the PIN file format, see the
    `Percolator documentation <https://github.com/percolator/percolator/
    wiki/Interface#tab-delimited-file-format>`_.

    Specifically, mokapot requires specific columns in the
    tab-delmited files: `specid`, `scannr`, `peptide`, `proteins`, and
    `label`. Note that these column names are case insensitive. In
    addition to the required columns, mokapot will look for an `expmass`
    and `calcmass` columns, which are generated by
    `Crux <http://crux.ms>`_, but are not intended to be features.

    In addition to PIN tab-delimited files, the `pin_files` argument
    can a :py:class:`pandas.DataFrame` containing the above columns.

    Finally, mokapot does not currently support specifying a
    default direction or feature weights in the PIN file itself.

    Parameters
    ----------
    pin_files : str, tuple of str, or pandas.DataFrame
        One or more PIN files to read or a :py:class:`pandas.DataFrame`.
    group_column : str, optional
        A factor to by which to group PSMs for grouped confidence
        estimation.
    to_df : bool, optional
        Return a :py:class:`pandas.DataFrame` instead of a
        py:class:`~mokapot.dataset.LinearPsmDataset`.
    copy_data : bool, optional
        If true, a deep copy of the data is created. This
        uses more memory, but is safer because it prevents
        accidental modification of the underlying data. This
        argument only has an effect when `pin_files` is a
        :py:class:`pandas.DataFrame`

    Returns
    -------
    LinearPsmDataset
        A :py:class:`~mokapot.dataset.LinearPsmDataset` object
        containing the PSMs from all of the PIN files.
    """
    logging.info("Parsing PSMs...")

    if isinstance(pin_files, pd.DataFrame):
        pin_df = pin_files.copy(deep=copy_data)
    else:
        pin_df = pd.concat(
            [read_percolator(f) for f in utils.tuplize(pin_files)]
        )

    # Find all of the necessary columns, case-insensitive:
    specid = [c for c in pin_df.columns if c.lower() == "specid"]
    peptides = [c for c in pin_df.columns if c.lower() == "peptide"]
    proteins = [c for c in pin_df.columns if c.lower() == "proteins"]
    labels = [c for c in pin_df.columns if c.lower() == "label"]
    other = [c for c in pin_df.columns if c.lower() == "calcmass"]
    spectra = [c for c in pin_df.columns if c.lower() in ["scannr", "expmass"]]
    nonfeat = sum([specid, spectra, peptides, proteins, labels, other], [])
    if group_column is not None:
        nonfeat += [group_column]

    features = [c for c in pin_df.columns if c not in nonfeat]

    # Check for errors:
    col_names = ["Label", "Peptide", "Proteins"]
    for col, name in zip([labels, peptides, proteins], col_names):
        if len(col) > 1:
            raise ValueError(f"More than one '{name}' column found.")

    if not all([specid, peptides, proteins, labels, spectra]):
        raise ValueError(
            "This PIN format is incompatible with mokapot. Please"
            " verify that the required columns are present."
        )

    # Convert labels to the correct format.
    if any(pin_df[labels[0]] == -1):
        pin_df[labels[0]] = (pin_df[labels[0]] + 1) / 2

    if to_df:
        return pin_df

    return LinearPsmDataset(
        psms=pin_df,
        target_column=labels[0],
        spectrum_columns=spectra,
        peptide_column=peptides[0],
        protein_column=proteins[0],
        group_column=group_column,
        feature_columns=features,
        copy_data=False,
    )


# Utility Functions -----------------------------------------------------------
def read_percolator(perc_file):
    """
    Read a Percolator tab-delimited file.

    Percolator input format (PIN) files and the Percolator result files
    are tab-delimited, but also have a tab-delimited protein list as the
    final column. This function parses the file and returns a DataFrame.

    Parameters
    ----------
    perc_file : str
        The file to parse.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of the parsed data.
    """
    LOGGER.info("Reading %s...", perc_file)
    if perc_file.endswith(".gz"):
        fopen = gzip.open
    else:
        fopen = open

    with fopen(perc_file) as perc:
        cols = perc.readline().rstrip().split("\t")
        psms_gen = (l.rstrip().split("\t", len(cols) - 1) for l in perc)
        psms = pd.DataFrame.from_records(psms_gen, columns=cols)

    return psms.apply(pd.to_numeric, errors="ignore")
