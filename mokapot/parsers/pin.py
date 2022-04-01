"""
This module contains the parsers for reading in PSMs
"""
import gzip
import logging

import numpy as np
import pandas as pd

from .. import utils
from ..dataset import LinearPsmDataset


LOGGER = logging.getLogger(__name__)

# Functions -------------------------------------------------------------------
def read_pin(
    pin_files,
    group_column=None,
    filename_column=None,
    calcmass_column=None,
    expmass_column=None,
    rt_column=None,
    charge_column=None,
    to_df=False,
    copy_data=False,
):
    """Read Percolator input (PIN) tab-delimited files.

    Read PSMs from one or more Percolator input (PIN) tab-delmited files,
    aggregating them into a single
    :py:class:`~mokapot.dataset.LinearPsmDataset`. For more details about the
    PIN file format, see the `Percolator documentation
    <https://github.com/percolator/percolator/
    wiki/Interface#tab-delimited-file-format>`_.

    Specifically, mokapot requires specific columns in the tab-delmited files:
    `specid`, `scannr`, `peptide`, `proteins`, and `label`. Note that these
    column names are case insensitive. In addition to these special columns
    defined for the PIN format, mokapot also looks for additional columns that
    specify the MS data file names, theoretical monoisotopic peptide masses,
    the measured mass, retention times, and charge states, which are necessary
    to create specific output formats for downstream tools, such as FlashLFQ.

    In addition to PIN tab-delimited files, the `pin_files` argument can be a
    :py:class:`pandas.DataFrame` containing the above columns.

    Finally, mokapot does not currently support specifying a default direction
    or feature weights in the PIN file itself. If these are present, they
    will be ignored.

    Parameters
    ----------
    pin_files : str, tuple of str, or pandas.DataFrame
        One or more PIN files to read or a :py:class:`pandas.DataFrame`.
    group_column : str, optional
        A factor to by which to group PSMs for grouped confidence
        estimation.
    filename_column : str, optional
        The column specifying the MS data file. If :code:`None`, mokapot will
        look for a column called "filename" (case insensitive). This is
        required for some output formats, such as FlashLFQ.
    calcmass_column : str, optional
        The column specifying the theoretical monoisotopic mass of the peptide
        including modifications. If :code:`None`, mokapot will look for a
        column called "calcmass" (case insensitive). This is required for some
        output formats, such as FlashLFQ.
    expmass_column : str, optional
        The column specifying the measured neutral precursor mass. If
        :code:`None`, mokapot will look for a column call "expmass" (case
        insensitive). This is required for some output formats.
    rt_column : str, optional
        The column specifying the retention time in seconds. If :code:`None`,
        mokapot will look for a column called "ret_time" (case insensitive).
        This is required for some output formats, such as FlashLFQ.
    charge_column : str, optional
        The column specifying the charge state of each peptide. If
        :code:`None`, mokapot will look for a column called "charge" (case
        insensitive). This is required for some output formats, such as
        FlashLFQ.
    to_df : bool, optional
        Return a :py:class:`pandas.DataFrame` instead of a
        :py:class:`~mokapot.dataset.LinearPsmDataset`.
    copy_data : bool, optional
        If true, a deep copy of the data is created. This uses more memory, but
        is safer because it prevents accidental modification of the underlying
        data. This argument only has an effect when `pin_files` is a
        :py:class:`pandas.DataFrame`

    Returns
    -------
    LinearPsmDataset
        A :py:class:`~mokapot.dataset.LinearPsmDataset` object containing the
        PSMs from all of the PIN files.
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
    scan = [c for c in pin_df.columns if c.lower() == "scannr"][0]
    nonfeat = sum([specid, [scan], peptides, proteins, labels], [])

    # Optional columns
    filename = _check_column(filename_column, pin_df, "filename")
    calcmass = _check_column(calcmass_column, pin_df, "calcmass")
    expmass = _check_column(expmass_column, pin_df, "expmass")
    ret_time = _check_column(rt_column, pin_df, "ret_time")
    charge = _check_column(charge_column, pin_df, "charge_column")
    spectra = [c for c in [filename, scan, ret_time, expmass] if c is not None]

    # Only add charge to features if there aren't other charge columns:
    alt_charge = [c for c in pin_df.columns if c.lower().startswith("charge")]
    if charge is not None and len(alt_charge) > 1:
        nonfeat.append(charge)

    # Add the grouping column
    if group_column is not None:
        nonfeat += [group_column]
        if group_column not in pin_df.columns:
            raise ValueError(f"The '{group_column} column was not found.")

    for col in [filename, calcmass, expmass, ret_time]:
        if col is not None:
            nonfeat.append(col)

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
    pin_df[labels[0]] = pin_df[labels[0]].astype(int)
    if any(pin_df[labels[0]] == -1):
        pin_df[labels[0]] = ((pin_df[labels[0]] + 1) / 2).astype(bool)

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
        filename_column=filename,
        scan_column=scan,
        calcmass_column=calcmass,
        expmass_column=expmass,
        rt_column=ret_time,
        charge_column=charge,
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
    if str(perc_file).endswith(".gz"):
        fopen = gzip.open
    else:
        fopen = open

    with fopen(perc_file) as perc:
        cols = perc.readline().rstrip().split("\t")
        dir_line = perc.readline().rstrip().split("\t")[0]
        if dir_line.lower() != "defaultdirection":
            perc.seek(0)
            _ = perc.readline()

        psms = pd.concat((c for c in _parse_in_chunks(perc, cols)), copy=False)

    return psms


def _parse_in_chunks(file_obj, columns, chunk_size=int(1e8)):
    """
    Parse a file in chunks

    Parameters
    ----------
    file_obj : file object
        The file to read lines from.
    columns : list of str
        The columns for each DataFrame.
    chunk_size : int
        The chunk size in bytes.

    Returns
    -------
    pandas.DataFrame
        The chunk of PSMs
    """
    while True:
        psms = file_obj.readlines(chunk_size)
        if not psms:
            break

        psms = [l.rstrip().split("\t", len(columns) - 1) for l in psms]
        psms = pd.DataFrame.from_records(psms, columns=columns)
        yield psms.apply(pd.to_numeric, errors="ignore")


def _check_column(col, df, default):
    """Check that a column exists in the dataframe."""
    if col is None:
        try:
            return [c for c in df.columns if c.lower() == default][0]
        except IndexError:
            return None

    if col not in df.columns:
        raise ValueError(f"The '{col}' column was not found.")

    return col
