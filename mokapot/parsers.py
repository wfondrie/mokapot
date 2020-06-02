"""
This module contains the parsers for reading in PSMs
"""
import gzip
from typing import Tuple, Union

import pandas as pd

from . import utils
from .dataset import LinearPsmDataset


# Functions -------------------------------------------------------------------
def read_pin(pin_files: Union[str, Tuple[str, ...]]) -> LinearPsmDataset:
    """
    Read Percolator Input (PIN) formatted files.

    Read PSMs from one or more Percolator Input (PIN), aggregating them into
    a single mokapot.LinearPsmDataset. If multiple pin files are provided, an
    additional `pin file` column is added to the data.

    Parameters
    ----------
    pin_files : str or tuple of str
        One or more pin files to read.

    Returns
    -------
    mokapot.LinearPsmDataset
        A mokapot dataset containing the PSMs from all of the pin files.
    """
    pin_df = pd.concat([_parse_pin(f) for f in utils.tuplize(pin_files)])

    # Find all of the necessary columns, case-insensitive:
    specid = tuple(c for c in pin_df.columns if c.lower() == "specid")
    peptides = tuple(c for c in pin_df.columns if c.lower() == "peptide")
    proteins = tuple(c for c in pin_df.columns if c.lower() == "proteins")
    labels = tuple(c for c in pin_df.columns if c.lower() == "label")
    other = tuple(c for c in pin_df.columns if c.lower() == "calcmass")
    spectra = tuple(c for c in pin_df.columns
                    if c.lower() in ["scannr", "expmass"])

    nonfeat: Tuple[str, ...] = sum([specid, spectra, peptides,
                                    proteins, labels, other],
                                   tuple())

    features = tuple(c for c in pin_df.columns if c not in nonfeat)

    # Check for errors:
    if len(labels) > 1:
        raise ValueError("More than one label column found in pin file.")

    if len(proteins) > 1:
        raise ValueError("More than one protein column found in pin file.")

    if not all([specid, peptides, proteins, labels, spectra]):
        raise ValueError("The PIN format was invalid. Please verify the "
                         "required columns are present.")

    # Convert labels to the correct format.
    pin_df[labels[0]] = (pin_df[labels[0]] + 1) / 2

    return LinearPsmDataset(psms=pin_df,
                            target_column=labels[0],
                            spectrum_columns=spectra,
                            peptide_columns=peptides,
                            protein_column=proteins[0],
                            experiment_columns=None,
                            feature_columns=features)


# Utility Functions -----------------------------------------------------------
def _parse_pin(pin_file):
    """Parse a Percolator INput formatted file."""
    if pin_file.endswith(".gz"):
        fopen = gzip.open
    else:
        fopen = open

    with fopen(pin_file, "r") as pin:
        header = pin.readline()
        header = header.replace("\n", "").split("\t")
        rows = [l.replace("\n", "").split("\t", len(header)-1) for l in pin]

    pin_df = pd.DataFrame(columns=header, data=rows)
    return pin_df.apply(pd.to_numeric, errors="ignore")
