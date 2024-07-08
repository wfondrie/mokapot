"""Test that Confidence classes are working correctly"""

import contextlib

import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import copy

from pandas.testing import assert_frame_equal
from pytest import approx
import pyarrow as pa

import mokapot
from mokapot import OnDiskPsmDataset, assign_confidence
from mokapot.confidence import get_unique_peptides_from_psms


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

    pin_file, _, _ = psm_df_1000
    columns = list(pd.read_csv(pin_file, sep="\t").columns)
    df_spectra = pd.read_csv(
        pin_file, sep="\t", usecols=["scannr", "expmass", "target"]
    )
    psms_disk = OnDiskPsmDataset(
        filename=pin_file,
        target_column="target",
        spectrum_columns=["scannr", "expmass"],
        peptide_column="peptide",
        feature_columns=["score"],
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
        np.random.seed(42)
        assign_confidence(
            [copy.copy(psms_disk)],
            prefixes=[None],
            descs=[True],
            dest_dir=tmp_path,
            max_workers=4,
            eval_fdr=0.02,
        )

    df_results_group = pd.read_csv(tmp_path / "targets.peptides", sep="\t")
    assert len(df_results_group) == 500
    assert df_results_group.columns.tolist() == [
        "PSMId",
        "peptide",
        "score",
        "q-value",
        "posterior_error_prob",
        "proteinIds",
    ]
    df = df_results_group.head(3)
    assert df["PSMId"].tolist() == [98, 187, 176]
    assert df["peptide"].tolist() == ["PELPK", "IYFCK", "CGQGK"]
    assert df["score"].tolist() == approx([5.857438, 5.703985, 5.337845])
    assert df["q-value"].tolist() == approx(
        [
            0.01020408,
            0.01020408,
            0.01020408,
        ]
    )
    assert df["posterior_error_prob"].tolist() == approx(
        [
            1.635110e-05,
            2.496682e-05,
            6.854064e-05,
        ]
    )


def test_assign_confidence_parquet(psm_df_1000_parquet, tmp_path):
    """Test that assign_confidence() works with parquet files."""

    parquet_file, _, _ = psm_df_1000_parquet
    columns = pq.ParquetFile(parquet_file).schema.names
    df_spectra = pq.read_table(
        parquet_file, columns=["scannr", "expmass", "target"]
    ).to_pandas()
    psms_disk = OnDiskPsmDataset(
        filename=parquet_file,
        target_column="target",
        spectrum_columns=["scannr", "expmass"],
        peptide_column="peptide",
        feature_columns=["score"],
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
            pa.int64(),
            pa.int64(),
            pa.int64(),
            pa.string(),
            pa.string(),
            pa.int64(),
        ],
        level_columns=["peptide"],
        specId_column="specid",
        spectra_dataframe=df_spectra,
    )
    with run_with_chunk_size(100):
        np.random.seed(42)
        assign_confidence(
            [copy.copy(psms_disk)],
            prefixes=[None],
            descs=[True],
            dest_dir=tmp_path,
            max_workers=4,
            eval_fdr=0.02,
        )
        df_results_group1 = pd.read_csv(
            tmp_path / "targets.peptides", sep="\t"
        )

    with run_with_chunk_size(10000):
        np.random.seed(42)
        assign_confidence(
            [copy.copy(psms_disk)],
            prefixes=[None],
            descs=[True],
            dest_dir=tmp_path,
            max_workers=4,
            eval_fdr=0.02,
        )
        df_results_group2 = pd.read_csv(
            tmp_path / "targets.peptides", sep="\t"
        )

    assert_frame_equal(df_results_group1, df_results_group2)


def test_get_unique_psms_and_peptides(peptide_csv_file, psms_iterator):
    psms_iterator = psms_iterator
    get_unique_peptides_from_psms(
        iterable=psms_iterator,
        peptide_col_name="Peptide",
        write_columns=["PSMId", "Label", "Peptide", "score", "proteinIds"],
        out_peptides=peptide_csv_file,
        sep="\t",
    )

    expected_output = pd.DataFrame(
        [
            [1, 1, "HLAQLLR", -5.75, "_.dummy._"],
            [3, 0, "NVPTSLLK", -5.83, "_.dummy._"],
            [4, 1, "QILVQLR", -5.92, "_.dummy._"],
            [7, 1, "SRTSVIPGPK", -6.12, "_.dummy._"],
        ],
        columns=["PSMId", "Label", "Peptide", "score", "proteinIds"],
    )

    output = pd.read_csv(peptide_csv_file, sep="\t")
    pd.testing.assert_frame_equal(expected_output, output)
