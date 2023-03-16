"""
This module contains the parsers for reading in PSMs
"""
import logging

import pandas as pd
import polars as pl

from .. import utils
from ..dataset import LinearPsmDataset

LOGGER = logging.getLogger(__name__)


def read_pin(
    pin_files: str | tuple[str] | pl.DataFrame | pd.DataFrame,
    group_column: str = None,
    filename_column: str = None,
    calcmass_column: str = None,
    expmass_column: str = None,
    rt_column: str = None,
    charge_column: str = None,
    to_df: bool = False,
) -> LinearPsmDataset | pl.LazyFrame:
    """Read Percolator input (PIN) tab-delimited files.

    Read PSMs from one or more Percolator input (PIN) tab-delmited files,
    aggregating them into a single
    :py:class:`~mokapot.dataset.LinearPsmDataset`. For more details about the
    PIN file format, see the `Percolator documentation
    <https://github.com/percolator/percolator/
    wiki/Interface#tab-delimited-file-format>`_.

    Specifically, mokapot requires specific columns in the tab-delmited files:
    `specid`, `scannr`, `peptide`, and `label`. Note that these
    column names are case insensitive. In addition to these special columns
    defined for the PIN format, mokapot also looks for additional columns that
    specify the MS data file names, theoretical monoisotopic peptide masses,
    the measured mass, retention times, and charge states, which are necessary
    to create specific output formats for downstream tools, such as FlashLFQ.

    In addition to PIN tab-delimited files, the `pin_files` argument can be a
    :py:class:`pl.DataFrame` :py:class:`pandas.DataFrame` containing the above
    columns.

    Finally, mokapot does not currently support specifying a default direction
    or feature weights in the PIN file itself. If these are present, they
    will be ignored.

    Parameters
    ----------
    pin_files : str, tuple of str, polars.DataFrame, or pandas.DataFrame
        One or more PIN files or data frames to be read. Multiple files can be
        specified using globs, such as ``"psms_*.pin"`` or separately.
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

    Returns
    -------
    LinearPsmDataset or polars.LazyFrame
        A :py:class:`~mokapot.dataset.LinearPsmDataset` object containing the
        PSMs from all of the PIN files.

    """
    logging.info("Parsing PSMs...")

    # Figure out the type of the input...
    if isinstance(pin_files, pd.DataFrame):
        pin_df = pl.from_pandas(pin_files).lazy()
    elif isinstance(pin_files, pl.DataFrame):
        pin_df = pin_files.lazy()
    elif isinstance(pin_files, pl.LazyFrame):
        pin_df = pin_files
    else:
        [pl.scan_csv(f, sep="\t") for f in utils.tuplize(pin_files)]
        pin_df = pl.concat(pin_files, how="diagonal")

    # Find all of the necessary columns, case-insensitive:
    specid = [c for c in pin_df.columns if c.lower() == "specid"]
    peptides = [c for c in pin_df.columns if c.lower() == "peptide"][0]
    labels = [c for c in pin_df.columns if c.lower() == "label"][0]
    scan = [c for c in pin_df.columns if c.lower() == "scannr"][0]

    # Optional columns
    filename = _check_column(filename_column, pin_df, "filename")
    calcmass = _check_column(calcmass_column, pin_df, "calcmass")
    expmass = _check_column(expmass_column, pin_df, "expmass")
    ret_time = _check_column(rt_column, pin_df, "ret_time")
    charge = _check_column(charge_column, pin_df, "charge_column")
    spectra = [c for c in [filename, scan, ret_time, expmass] if c is not None]

    try:
        proteins = [c for c in pin_df.columns if c.lower() == "proteins"][0]
    except IndexError:
        proteins = None

    nonfeat = [*specid, scan, peptides, proteins, labels]

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
    if not all([specid, peptides, labels, spectra]):
        raise ValueError(
            "This PIN format is incompatible with mokapot. Please"
            " verify that the required columns are present."
        )

    # Convert labels to the correct format.
    pin_df = pin_df.with_columns(
        pl.when(pl.col("Label") == 1)
        .then(True)
        .otherwise(False)
        .alias("Label")
    )

    if to_df:
        return pin_df

    return LinearPsmDataset(
        psms=pin_df,
        target_column=labels,
        spectrum_columns=spectra,
        peptide_column=peptides,
        protein_column=proteins,
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
