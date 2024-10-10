"""Test that Confidence classes are working correctly"""

import contextlib
import copy

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas.testing import assert_frame_equal
from pytest import approx

import mokapot
from mokapot import OnDiskPsmDataset, assign_confidence


@contextlib.contextmanager
def run_with_chunk_size(chunk_size):
    old_chunk_size = mokapot.confidence.CONFIDENCE_CHUNK_SIZE
    try:
        # We need to fully qualified path name to modify the constants
        mokapot.confidence.CONFIDENCE_CHUNK_SIZE = chunk_size
        yield
    finally:
        mokapot.confidence.CONFIDENCE_CHUNK_SIZE = old_chunk_size


def test_chunked_assign_confidence(psm_df_1000, tmp_path):
    """Test that assign_confidence() works correctly with small chunks"""

    # After correcting the targets column stuff and
    # with that the updated labels
    # (see _update_labels) it turned out that there were no targets with fdr
    # below 0.01 anymore, and so as a remedy the eval_fdr was raised to 0.02.
    # NB: with the old bug there would *always* be targets labelled as 1
    # incorrectly (namely the last and before last)

    pin_file, df, _ = psm_df_1000
    columns = list(pd.read_csv(pin_file, sep="\t").columns)
    df_spectra = pd.read_csv(
        pin_file, sep="\t", usecols=["scannr", "expmass", "target"]
    )
    score = df["score"].values
    psms_disk = OnDiskPsmDataset(
        pin_file,
        target_column="target",
        spectrum_columns=["scannr", "expmass"],
        peptide_column="peptide",
        feature_columns=[],
        filename_column="filename",
        scan_column="scannr",
        calcmass_column="calcmass",
        expmass_column="expmass",
        rt_column="ret_time",
        charge_column="charge",
        columns=columns,
        protein_column="proteins",
        metadata_columns=[
            "specid",
            "scannr",
            "expmass",
            "peptide",
            "proteins",
            "target",
        ],
        metadata_column_types=[
            "int",
            "int",
            "float",
            "string",
            "string",
            "int",
        ],
        level_columns=["peptide"],
        specId_column="specid",
        spectra_dataframe=df_spectra,
    )
    with run_with_chunk_size(100):
        assign_confidence(
            [copy.copy(psms_disk)],
            scores_list=[score],
            prefixes=[None],
            dest_dir=tmp_path,
            max_workers=4,
            eval_fdr=0.02,
        )

    df_results_group = pd.read_csv(tmp_path / "targets.peptides.csv", sep="\t")
    # assert len(df_results_group) == 500
    assert len(df_results_group) > 400
    assert df_results_group.columns.tolist() == [
        "PSMId",
        "peptide",
        "score",
        "q-value",
        "posterior_error_prob",
        "proteinIds",
    ]
    df = df_results_group.head(3)
    assert df["PSMId"].tolist() == [136, 96, 164]
    assert df["peptide"].tolist() == ["EVSSK", "HDWCK", "SYQVK"]
    assert df["score"].tolist() == approx([5.767435, 5.572517, 5.531904])
    assert df["q-value"].tolist() == approx([
        0.0103092780336737,
        0.0103092780336737,
        0.0103092780336737,
    ])
    assert df["posterior_error_prob"].tolist() == approx([
        3.315389846699129e-05,
        5.558992546200682e-05,
        6.191049743361808e-05,
    ])


def test_assign_confidence_parquet(psm_df_1000_parquet, tmp_path):
    """Test that assign_confidence() works with parquet files."""

    parquet_file, df, _ = psm_df_1000_parquet
    columns = pq.ParquetFile(parquet_file).schema.names
    df_spectra = pq.read_table(
        parquet_file, columns=["scannr", "expmass", "target"]
    ).to_pandas()
    scores = [df["score"].values]
    psms_disk = OnDiskPsmDataset(
        parquet_file,
        target_column="target",
        spectrum_columns=["scannr", "expmass"],
        peptide_column="peptide",
        feature_columns=[],
        filename_column="filename",
        scan_column="scannr",
        calcmass_column="calcmass",
        expmass_column="expmass",
        rt_column="ret_time",
        charge_column="charge",
        columns=columns,
        protein_column="proteins",
        metadata_columns=[
            "specid",
            "scannr",
            "expmass",
            "peptide",
            "proteins",
            "target",
        ],
        metadata_column_types=[
            np.dtype("int64"),
            np.dtype("int64"),
            np.dtype("float64"),
            np.dtype("O"),
            np.dtype("O"),
            np.dtype("int64"),
        ],
        level_columns=["peptide"],
        specId_column="specid",
        spectra_dataframe=df_spectra,
    )
    with run_with_chunk_size(100):
        np.random.seed(42)
        assign_confidence(
            [copy.copy(psms_disk)],
            scores_list=scores,
            prefixes=[None],
            dest_dir=tmp_path,
            max_workers=4,
            eval_fdr=0.02,
        )
        df_results_group1 = pd.read_parquet(
            tmp_path / "targets.peptides.parquet"
        )

    with run_with_chunk_size(10000):
        np.random.seed(42)
        assign_confidence(
            [copy.copy(psms_disk)],
            scores_list=scores,
            prefixes=[None],
            dest_dir=tmp_path,
            max_workers=4,
            eval_fdr=0.02,
        )
        df_results_group2 = pd.read_parquet(
            tmp_path / "targets.peptides.parquet"
        )

    assert_frame_equal(df_results_group1, df_results_group2)
