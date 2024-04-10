import pyarrow.parquet as pq
import pandas as pd
import logging

from joblib import Parallel, delayed
from ..utils import (
    flatten,
    convert_targets_column,
    tuplize,
)
from .pin import (
    create_chunks_with_identifier,
    get_rows_from_dataframe,
    concat_and_reindex_chunks,
)
from .helpers import find_optional_column, find_columns, find_required_column
from ..constants import (
    CHUNK_SIZE_COLUMNS_FOR_DROP_COLUMNS,
    CHUNK_SIZE_ROWS_FOR_DROP_COLUMNS,
)

from ..dataset import OnDiskPsmDataset


LOGGER = logging.getLogger(__name__)


def get_column_names_from_file(file):
    return pq.ParquetFile(file).schema.names


def read_file_in_chunks_parquet(file, chunk_size, use_cols):
    """
    when reading in chunks an open file object is required as input to iterate over the
    chunks
    """
    pf = pq.ParquetFile(file)
    for i, record_batch in enumerate(
        pf.iter_batches(chunk_size, columns=use_cols)
    ):
        df = record_batch.to_pandas()
        df.index = df.index + i * chunk_size
        yield df


def parse_in_chunks_parquet(psms, train_idx, chunk_size, max_workers):
    """
    Parse a file in chunks

    Parameters
    ----------
    psms : OnDiskPsmDataset
        A collection of PSMs.
    train_idx : list of a list of a list of indexes (first level are training splits,
        second one is the number of input files, third level the actual idexes
        The indexes to select from data.
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
        reader = read_file_in_chunks_parquet(
            file=_psms.filename,
            chunk_size=chunk_size,
            use_cols=_psms.columns,
        )
        Parallel(n_jobs=max_workers, require="sharedmem")(
            delayed(get_rows_from_dataframe)(
                idx, chunk, train_psms, _psms, file_idx
            )
            for chunk in reader
        )
    train_psms_reordered = Parallel(n_jobs=max_workers, require="sharedmem")(
        delayed(concat_and_reindex_chunks)(df=df, orig_idx=orig_idx)
        for df, orig_idx in zip(train_psms, zip(*train_idx))
    )
    return [pd.concat(df) for df in zip(*train_psms_reordered)]


def drop_missing_values_and_fill_spectra_dataframe_parquet(
    file, column, spectra, df_spectra_list
):
    na_mask = pd.DataFrame([], columns=list(set(column) - set(spectra)))
    reader = read_file_in_chunks_parquet(
        file=file,
        use_cols=column,
        chunk_size=CHUNK_SIZE_ROWS_FOR_DROP_COLUMNS,
    )
    for i, feature in enumerate(reader):
        if set(spectra) <= set(column):
            df_spectra_list.append(feature[spectra])
            feature.drop(spectra, axis=1, inplace=True)
        na_mask = pd.concat(
            [na_mask, pd.DataFrame([feature.isna().any(axis=0)])],
            ignore_index=True,
        )
    del reader
    na_mask = na_mask.any(axis=0)
    if na_mask.any():
        return list(na_mask[na_mask].index)


def read_parquet(
    perc_files,
    max_workers,
    group_column=None,
    filename_column=None,
    calcmass_column=None,
    expmass_column=None,
    rt_column=None,
    charge_column=None,
):
    logging.info("Parsing PSMs...")
    return [
        read_parquet_file(
            parquet_file,
            max_workers=max_workers,
            group_column=group_column,
            filename_column=filename_column,
            calcmass_column=calcmass_column,
            expmass_column=expmass_column,
            rt_column=rt_column,
            charge_column=charge_column,
        )
        for parquet_file in tuplize(perc_files)
    ]


def read_parquet_file(
    perc_file,
    max_workers,
    group_column=None,
    filename_column=None,
    calcmass_column=None,
    expmass_column=None,
    rt_column=None,
    charge_column=None,
):
    """
    Read a Percolator input file in parquet format.

    This function parses the percolator input file in parquet format
    and returns an OnDiskPsmDataset.

    Parameters
    ----------
    perc_file : str
        The file to parse.

    Returns
    -------
    OnDiskPsmDataset
    """

    LOGGER.info("Reading %s...", perc_file)
    columns = get_column_names_from_file(perc_file)

    # Find all of the necessary columns, case-insensitive:
    specid = find_required_column("specid", columns)
    peptides = find_required_column("peptide", columns)
    proteins = find_required_column("proteins", columns)
    labels = find_required_column("label", columns)
    scan = find_required_column("scannr", columns)
    nonfeat = [specid, scan, peptides, proteins, labels]

    # Columns for different rollup levels
    # Currently no proteins, since the protein rollup is probably quite different
    # from the other rollup levels IMHO
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
        delayed(drop_missing_values_and_fill_spectra_dataframe_parquet)(
            file=perc_file,
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
        group_column=group_column,
        feature_columns=_feature_columns,
        metadata_columns=nonfeat,
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
