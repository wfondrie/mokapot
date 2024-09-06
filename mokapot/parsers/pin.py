"""
This module contains the parsers for reading in PSMs
"""

import logging
import warnings
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from .helpers import find_optional_column, find_columns, find_required_column
from ..utils import (
    open_file,
    tuplize,
    create_chunks,
    convert_targets_column,
    flatten,
)
from ..dataset import OnDiskPsmDataset
from ..constants import (
    CHUNK_SIZE_COLUMNS_FOR_DROP_COLUMNS,
    CHUNK_SIZE_ROWS_FOR_DROP_COLUMNS,
)
from ..tabular_data import TabularDataReader
from typing import List

LOGGER = logging.getLogger(__name__)


# Functions -------------------------------------------------------------------
def read_pin(
    pin_files,
    max_workers,
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
    max_workers: int
        Maximum number of parallel processes to use.
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
            max_workers=max_workers,
            filename_column=filename_column,
            calcmass_column=calcmass_column,
            expmass_column=expmass_column,
            rt_column=rt_column,
            charge_column=charge_column,
        )
        for pin_file in tuplize(pin_files)
    ]


def create_chunks_with_identifier(data, identifier_column, chunk_size):
    """
    This function will split data into chunks but will make sure that
    identifier_columns is never split

    Parameters
    ----------
    data: the data you want to split in chunks (1d list)
    identifier_column: columns that should never be splitted.
        Must be of length 2.
    chunk_size: the chunk size

    Returns
    -------

    """
    if (len(data) + len(identifier_column)) % chunk_size != 1:
        data_copy = data + identifier_column
        return create_chunks(data_copy, chunk_size)
    else:
        return create_chunks(data, chunk_size) + [identifier_column]


def read_percolator(
    perc_file: Path,
    max_workers,
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
    perc_file : Path
        The file to parse.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of the parsed data.
    """

    LOGGER.info("Reading %s...", perc_file)
    reader = TabularDataReader.from_path(perc_file)
    columns = reader.get_column_names()
    col_types = reader.get_column_types()

    # Find all the necessary columns, case-insensitive:
    specid = find_required_column("specid", columns)
    peptides = find_required_column("peptide", columns)
    proteins = find_required_column("proteins", columns)
    labels = find_required_column("label", columns)
    scan = find_required_column("scannr", columns)
    nonfeat = [specid, scan, peptides, proteins, labels]

    # Columns for different rollup levels
    # Currently no proteins, since the protein rollup is probably quite
    # different from the other rollup levels IMHO
    modifiedpeptides = find_columns("modifiedpeptide", columns)
    precursors = find_columns("precursor", columns)
    peptidegroups = find_columns("peptidegroup", columns)
    level_columns = [peptides] + modifiedpeptides + precursors + peptidegroups
    nonfeat += modifiedpeptides + precursors + peptidegroups

    # Optional columns
    filename = find_optional_column(filename_column, columns, "filename")
    calcmass = find_optional_column(calcmass_column, columns, "calcmass")
    expmass = find_optional_column(expmass_column, columns, "expmass")
    ret_time = find_optional_column(rt_column, columns, "ret_time")
    charge = find_optional_column(charge_column, columns, "charge_column")
    spectra = [c for c in [filename, scan, ret_time, expmass] if c is not None]

    # Only add charge to features if there aren't other charge columns:
    alt_charge = [c for c in columns if c.lower().startswith("charge")]
    if charge is not None and len(alt_charge) > 1:
        nonfeat.append(charge)

    for col in [filename, calcmass, expmass, ret_time]:
        if col is not None:
            nonfeat.append(col)

    features = [c for c in columns if c not in nonfeat]
    nonfeat_types = [col_types[columns.index(col)] for col in nonfeat]

    # Check for errors:
    if not all(spectra):
        raise ValueError(
            "This PIN format is incompatible with mokapot. Please"
            " verify that the required columns are present."
        )

    # Check that features don't have missing values:
    feat_slices = create_chunks_with_identifier(
        data=features,
        identifier_column=spectra + [labels],
        chunk_size=CHUNK_SIZE_COLUMNS_FOR_DROP_COLUMNS,
    )
    df_spectra_list = []
    features_to_drop = Parallel(n_jobs=max_workers, require="sharedmem")(
        delayed(drop_missing_values_and_fill_spectra_dataframe)(
            reader=reader,
            column=c,
            spectra=spectra + [labels],
            df_spectra_list=df_spectra_list,
        )
        for c in feat_slices
    )
    df_spectra = convert_targets_column(
        pd.concat(df_spectra_list), target_column=labels
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
        target_column=labels,
        spectrum_columns=spectra,
        peptide_column=peptides,
        protein_column=proteins,
        feature_columns=_feature_columns,
        metadata_columns=nonfeat,
        metadata_column_types=nonfeat_types,
        level_columns=level_columns,
        filename_column=filename,
        scan_column=scan,
        specId_column=specid,
        calcmass_column=calcmass,
        expmass_column=expmass,
        rt_column=ret_time,
        charge_column=charge,
        spectra_dataframe=df_spectra,
    )


# Utility Functions -----------------------------------------------------------
def drop_missing_values_and_fill_spectra_dataframe(
    reader: TabularDataReader,
    column: List,
    spectra: List,
    df_spectra_list: List[pd.DataFrame],
):
    na_mask = pd.DataFrame([], columns=list(set(column) - set(spectra)))
    file_iterator = reader.get_chunked_data_iterator(
        chunk_size=CHUNK_SIZE_ROWS_FOR_DROP_COLUMNS, columns=column
    )
    for i, feature in enumerate(file_iterator):
        if set(spectra) <= set(column):
            df_spectra_list.append(feature[spectra])
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=pd.errors.SettingWithCopyWarning
                )
                feature.drop(spectra, axis=1, inplace=True)
        na_mask = pd.concat(
            [na_mask, pd.DataFrame([feature.isna().any(axis=0)])],
            ignore_index=True,
        )
    del file_iterator
    na_mask = na_mask.any(axis=0)
    if na_mask.any():
        return list(na_mask[na_mask].index)


def read_file_in_chunks(file, chunk_size, use_cols):
    """
    when reading in chunks an open file object is required as input to
    iterate over the chunks
    """
    for df in pd.read_csv(
        file,
        sep="\t",
        chunksize=chunk_size,
        usecols=use_cols,
        index_col=False,
    ):
        yield df[use_cols]


def get_column_names_from_file(file):
    with open_file(file) as perc:
        return perc.readline().rstrip().split("\t")


def get_rows_from_dataframe(idx, chunk, train_psms, psms, file_idx):
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
    psms : OnDiskPsmDataset
        A collection of PSMs.
    file_idx : the index of the file being searched

    Returns
    -------
    List
        list of list of dataframes
    """
    chunk = convert_targets_column(
        data=chunk,
        target_column=psms.target_column,
    )
    for k, train in enumerate(idx):
        idx_ = list(set(train) & set(chunk.index))
        train_psms[file_idx][k].append(chunk.loc[idx_])


def concat_and_reindex_chunks(df, orig_idx):
    return [
        pd.concat(df_fold).reindex(orig_idx_fold)
        for df_fold, orig_idx_fold in zip(df, orig_idx)
    ]


def parse_in_chunks(psms, train_idx, chunk_size, max_workers):
    """
    Parse a file in chunks

    Parameters
    ----------
    psms : OnDiskPsmDataset
        A collection of PSMs.
    train_idx : list of a list of a list of indexes (first level are training
        splits, second one is the number of input files, third level the
        actual idexes The indexes to select from data.
    chunk_size : int
        The chunk size in bytes.
    max_workers: int
            Number of workers for Parallel

    Returns
    -------
    List
        list of dataframes
    """

    train_psms = [
        [[] for _ in range(len(train_idx))] for _ in range(len(psms))
    ]
    for _psms, idx, file_idx in zip(psms, zip(*train_idx), range(len(psms))):
        reader = TabularDataReader.from_path(_psms.filename)
        file_iterator = reader.get_chunked_data_iterator(
            chunk_size=chunk_size, columns=_psms.columns
        )
        Parallel(n_jobs=max_workers, require="sharedmem")(
            delayed(get_rows_from_dataframe)(
                idx, chunk, train_psms, _psms, file_idx
            )
            for chunk in file_iterator
        )
    train_psms_reordered = Parallel(n_jobs=max_workers, require="sharedmem")(
        delayed(concat_and_reindex_chunks)(df=df, orig_idx=orig_idx)
        for df, orig_idx in zip(train_psms, zip(*train_idx))
    )
    return [pd.concat(df) for df in zip(*train_psms_reordered)]
