"""Test that Confidence classes are working correctly"""

import contextlib
import copy

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from pandas.testing import assert_frame_equal

import mokapot
from mokapot import LinearPsmDataset, OnDiskPsmDataset, assign_confidence
from mokapot.column_defs import STANDARD_COLUMN_NAME_MAP


@contextlib.contextmanager
def run_with_chunk_size(chunk_size):
    old_chunk_size = mokapot.confidence.CONFIDENCE_CHUNK_SIZE
    try:
        # We need to fully qualified path name to modify the constants
        mokapot.confidence.CONFIDENCE_CHUNK_SIZE = chunk_size
        yield
    finally:
        mokapot.confidence.CONFIDENCE_CHUNK_SIZE = old_chunk_size


@pytest.fixture
def inmem_psms_ds(psm_df_builder):
    """A small-ish PSM dataset"""
    data = psm_df_builder(1000, 1000, score_diffs=[5.0])
    psms = LinearPsmDataset(
        psms=data.df,
        target_column="target",
        spectrum_columns=["specid"],
        peptide_column="peptide",
        feature_columns=list(data.score_cols),
        copy_data=True,
    )
    return psms


@pytest.mark.parametrize("deduplication", [True, False])
def test_assign_unscored_confidence(inmem_psms_ds, tmp_path, deduplication):
    orig_df = inmem_psms_ds.data.copy()
    _foo = assign_confidence(
        [inmem_psms_ds],
        scores_list=None,
        eval_fdr=0.01,
        dest_dir=tmp_path,
        max_workers=4,
        deduplication=False,
    )
    out_specid_peps = (
        _foo[0]
        .out_writers["psms"][0]
        .read()[["specid", "peptide"]]
        .sort_values("specid")
        .reset_index(drop=True)
    )
    orig_specid_peps = orig_df.loc[:, ["specid", "peptide"]].sort_values(
        "specid"
    )
    orig_specid_peps = orig_specid_peps[
        orig_specid_peps["specid"].isin(out_specid_peps["specid"])
    ].reset_index(drop=True)

    # Here I am making sure the spec ids and peptides are not getting shuffled.
    pd.testing.assert_frame_equal(out_specid_peps, orig_specid_peps)

    # TODO actually add assertions here ...


@pytest.mark.parametrize("deduplication", [True, False])
def test_chunked_assign_confidence(psm_df_1000, tmp_path, deduplication):
    """Test that assign_confidence() works correctly with small chunks"""

    # After correcting the targets column stuff and
    # with that the updated labels
    # (see _update_labels) it turned out that there were no targets with fdr
    # below 0.01 anymore, and so as a remedy the eval_fdr was raised to 0.02.
    # NB: with the old bug there would *always* be targets labelled as 1
    # incorrectly (namely the last and before last)

    pin_file, df, _, score_cols = psm_df_1000
    df_spectra = pd.read_csv(
        pin_file,
        sep="\t",
        usecols=["scannr", "expmass", "target"],
    )
    score = df[score_cols[0]].values
    psms_disk = OnDiskPsmDataset(
        pin_file,
        target_column="target",
        spectrum_columns=["scannr", "specid", "expmass", "filename"],
        peptide_column="peptide",
        feature_columns=[],
        extra_confidence_level_columns=[],
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
            deduplication=deduplication,
        )

    df_results_group = pd.read_csv(tmp_path / "targets.peptides.tsv", sep="\t")
    # The number of results is between 490 and 510
    # to account for collisions between targets/decoys
    # due to the random generation of the identifiers.
    assert len(df_results_group) > 490
    assert len(df_results_group) < 510
    assert df_results_group.columns.tolist() == [
        *psms_disk.spectrum_columns,
        psms_disk.peptide_column,
        STANDARD_COLUMN_NAME_MAP["score"],
        STANDARD_COLUMN_NAME_MAP["q-value"],
        STANDARD_COLUMN_NAME_MAP["posterior_error_prob"],
        # psms_disk.protein_column,
        # Since no protein column is assigned, we dont
        # expect any on the output.
    ]
    df_head = df_results_group.head(3)
    df_tail = df_results_group.tail(3)

    # Test the sorting of the file and the values based on the distribution of
    # the scores (see the fixture definition)
    assert np.all(df_head["specid"] < 500), (
        "Psms with ID > 500 should not be present (decoys)"
        # The fixture that generates the data assigns spec ids < 500 to targets
        # and > 500 to decoys.
    )

    assert np.all(df_head[STANDARD_COLUMN_NAME_MAP["score"]] > 5.0), (
        "Good scores should be greater than 5.0"
    )
    assert np.all(df_tail[STANDARD_COLUMN_NAME_MAP["score"]] < 0.0), (
        "Bad scores should be greater than less than 0"
    )

    assert np.all(df_head[STANDARD_COLUMN_NAME_MAP["q-value"]] < 0.015), (
        "Good q-values should be lt 0.015"
    )
    assert np.all(df_tail[STANDARD_COLUMN_NAME_MAP["q-value"]] > 0.9), (
        "Bad q-values should be gt 0.9"
    )

    assert np.all(
        df_head[STANDARD_COLUMN_NAME_MAP["posterior_error_prob"]] < 0.001
    ), "Good PEPs should be lt 0.001"
    assert np.all(
        df_tail[STANDARD_COLUMN_NAME_MAP["posterior_error_prob"]] > 0.98
    ), "Bad PEPs should be gt 0.98"


@pytest.mark.parametrize("deduplication", [True, False])
def test_assign_confidence_parquet(
    psm_df_1000_parquet, tmp_path, deduplication
):
    """Test that assign_confidence() works with parquet files."""

    parquet_file, df, _ = psm_df_1000_parquet
    df_spectra = pq.read_table(
        parquet_file, columns=["scannr", "expmass", "target"]
    ).to_pandas()
    scores = [df["score0"].values]
    psms_disk = OnDiskPsmDataset(
        parquet_file,
        target_column="target",
        spectrum_columns=["specid", "scannr", "expmass"],
        peptide_column="peptide",
        feature_columns=[],
        extra_confidence_level_columns=[],
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
            deduplication=deduplication,
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
            deduplication=deduplication,
        )
        df_results_group2 = pd.read_parquet(
            tmp_path / "targets.peptides.parquet"
        )

    assert_frame_equal(df_results_group1, df_results_group2)
