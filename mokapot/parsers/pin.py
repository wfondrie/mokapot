"""
This module contains the parsers for reading in PSMs
"""
import logging

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from ..utils import (
    open_file,
    tuplize,
    create_chunks,
    convert_targets_column,
    flatten,
)
from ..dataset import OnDiskPsmDataset, read_file
from ..constants import (
    CHUNK_SIZE_COLUMNS_FOR_DROP_COLUMNS,
    CHUNK_SIZE_ROWS_FOR_DROP_COLUMNS,
)

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
    folds :
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
    return [
        read_percolator(
            pin_file,
            group_column=group_column,
            filename_column=filename_column,
            calcmass_column=calcmass_column,
            expmass_column=expmass_column,
            rt_column=rt_column,
            charge_column=charge_column,
        )
        for pin_file in tuplize(pin_files)
    ]


def read_percolator(
    perc_file,
    group_column=None,
    filename_column=None,
    calcmass_column=None,
    expmass_column=None,
    rt_column=None,
    charge_column=None,
):
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
    columns = get_column_names_from_file(perc_file)

    # TODO: Refactor the generation of column variables with simpler implementation
    # Find all of the necessary columns, case-insensitive:
    specid = [c for c in columns if c.lower() == "specid"]
    peptides = [c for c in columns if c.lower() == "peptide"]
    proteins = [c for c in columns if c.lower() == "proteins"]
    labels = [c for c in columns if c.lower() == "label"]
    scan = [c for c in columns if c.lower() == "scannr"][0]
    nonfeat = sum([specid, [scan], peptides, proteins, labels], [])

    # Optional columns
    filename = _check_column(filename_column, columns, "filename")
    calcmass = _check_column(calcmass_column, columns, "calcmass")
    expmass = _check_column(expmass_column, columns, "expmass")
    ret_time = _check_column(rt_column, columns, "ret_time")
    charge = _check_column(charge_column, columns, "charge_column")
    spectra = [c for c in [filename, scan, ret_time, expmass] if c is not None]

    # Only add charge to features if there aren't other charge columns:
    alt_charge = [c for c in columns if c.lower().startswith("charge")]
    if charge is not None and len(alt_charge) > 1:
        nonfeat.append(charge)

    # Add the grouping column
    if group_column is not None:
        nonfeat += [group_column]
        if group_column not in columns:
            raise ValueError(f"The '{group_column} column was not found.")

    for col in [filename, calcmass, expmass, ret_time]:
        if col is not None:
            nonfeat.append(col)

    features = [c for c in columns if c not in nonfeat]

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

    # Check that features don't have missing values:
    feat_slices = create_chunks(
        data=features + spectra + labels,
        chunk_size=CHUNK_SIZE_COLUMNS_FOR_DROP_COLUMNS,
    )
    df_spectra = []
    features_to_drop = Parallel(n_jobs=-1, require="sharedmem")(
        delayed(drop_missing_values_and_fill_spectra_dataframe)(
            file=perc_file,
            column=c,
            spectra=spectra + labels,
            df_spectra=df_spectra,
        )
        for c in feat_slices
    )
    df_spectra = convert_targets_column(
        pd.concat(df_spectra).apply(pd.to_numeric, errors="ignore"),
        target_column=labels[0],
    )
    features_to_drop = [drop for drop in features_to_drop if drop]
    features_to_drop = flatten(features_to_drop)
    if len(features_to_drop) > 1:
        LOGGER.warning("Missing values detected in the following features:")
        for col in features_to_drop:
            LOGGER.warning("  - %s", col)

        LOGGER.warning("Dropping features with missing values...")
    _feature_columns = tuple(
        [feature for feature in features if feature not in features_to_drop]
    )

    LOGGER.info("Using %i features:", len(_feature_columns))
    for i, feat in enumerate(_feature_columns):
        LOGGER.debug("  (%i)\t%s", i + 1, feat)

    return OnDiskPsmDataset(
        filename=perc_file,
        columns=columns,
        target_column=labels[0],
        spectrum_columns=spectra,
        peptide_column=peptides[0],
        protein_column=proteins[0],
        group_column=group_column,
        feature_columns=_feature_columns,
        metadata_columns=nonfeat,
        filename_column=filename,
        scan_column=scan,
        specId_column=specid[0],
        calcmass_column=calcmass,
        expmass_column=expmass,
        rt_column=ret_time,
        charge_column=charge,
        spectra_dataframe=df_spectra,
    )


# Utility Functions -----------------------------------------------------------
def drop_missing_values_and_fill_spectra_dataframe(
    file, column, spectra, df_spectra
):
    na_mask = pd.DataFrame([], columns=list(set(column) - set(spectra)))
    with open_file(file) as f:
        reader = read_file_in_chunks(
            file=f,
            use_cols=column,
            chunk_size=CHUNK_SIZE_ROWS_FOR_DROP_COLUMNS,
        )
        for i, feature in enumerate(reader):
            if set(spectra) <= set(column):
                df_spectra.append(feature[spectra])
                feature.drop(spectra, axis=1, inplace=True)
            na_mask = pd.concat(
                [na_mask, pd.DataFrame([feature.isna().any(axis=0)])],
                ignore_index=True,
            )
        del reader
        na_mask = na_mask.any(axis=0)
        if na_mask.any():
            return list(na_mask[na_mask].index)


def read_file_in_chunks(file, chunk_size, use_cols):
    """
    when reading in chunks an open file object is required as input to iterate over the
    chunks
    """
    return pd.read_csv(
        file,
        sep="\t",
        chunksize=chunk_size,
        usecols=use_cols,
        index_col=False,
    )


def get_column_names_from_file(file):
    with open_file(file) as perc:
        return perc.readline().rstrip().split("\t")


def get_rows_from_dataframe(idx, chunk, train_psms, psms):
    """
    extract rows from a chunk of a dataframe

    Parameters
    ----------
    idx : list of list of indexes
        The indexes to select from dataframe.
    train_psms : list of list of dataframes
        Contains subsets of dataframes that are already extracted.
    chunk : dataframe
        Subset of a dataframe.

    Returns
    -------
    List
        list of list of dataframes
    """
    chunk = convert_targets_column(
        data=chunk.apply(pd.to_numeric, errors="ignore"),
        target_column=psms.target_column,
    )
    for k, train in enumerate(idx):
        idx_ = list(set(train) & set(chunk.index))
        train_psms[k].append(
            chunk.loc[idx_].apply(pd.to_numeric, errors="ignore")
        )


def parse_in_chunks(psms, train_idx, chunk_size):
    """
    Parse a file in chunks

    Parameters
    ----------
    psms : OnDiskPsmDataset
        A collection of PSMs.
    train_idx : list of list of indexes
        The indexes to select from data.
    chunk_size : int
        The chunk size in bytes.

    Returns
    -------
    List
        list of dataframes
    """

    train_psms = [[] for _ in range(len(train_idx))]
    for _psms, idx in zip(psms, zip(*train_idx)):
        reader = read_file_in_chunks(
            file=_psms.filename,
            chunk_size=chunk_size,
            use_cols=_psms.columns,
        )
        Parallel(n_jobs=-1, require="sharedmem")(
            delayed(get_rows_from_dataframe)(idx, chunk, train_psms, _psms)
            for chunk in reader
        )
    return Parallel(n_jobs=-1, require="sharedmem")(
        delayed(pd.concat)(df) for df in train_psms
    )


def _check_column(col, columns, default):
    """Check that a column exists in the dataframe."""
    if col is None:
        try:
            return [c for c in columns if c.lower() == default][0]
        except IndexError:
            return None

    if col not in columns:
        raise ValueError(f"The '{col}' column was not found.")

    return col


def read_data_for_rescale(psms, subset_max_rescale):
    data_sizes = [
        sum(1 for line in open(_psms.filename)) - 1 for _psms in psms
    ]
    skip_rows_per_file = [None for _ in psms]
    if subset_max_rescale and subset_max_rescale < sum(data_sizes):
        subset_max_rescale_per_file = [
            subset_max_rescale // len(data_sizes)
            for _ in range(len(data_sizes))
        ]
        subset_max_rescale_per_file[-1] += subset_max_rescale - sum(
            subset_max_rescale_per_file
        )
        skip_rows_per_file = [
            sorted(
                np.random.choice(
                    a=range(1, data_size + 1),
                    size=data_size - subset_max,
                    replace=False,
                )
            )
            for data_size, subset_max in zip(
                data_sizes, subset_max_rescale_per_file
            )
        ]
    return pd.concat(
        [
            read_file(
                _psms.filename,
                use_cols=_psms.feature_columns,
                target_column=_psms.target_column,
            )
            for _psms, skip_rows in zip(psms, skip_rows_per_file)
        ]
    ).reset_index(drop=True)
