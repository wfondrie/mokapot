"""
These tests verify that the CLI works as expected.

At least for now, they do not check the correctness of the
output, just that the expect outputs are created.
"""

import contextlib
from pathlib import Path

import pandas as pd
import pytest

import mokapot
from ..helpers.cli import run_mokapot_cli
from ..helpers.utils import file_exist, file_approx_len, count_lines


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
def percolator_extended_file_small():
    """Get the extended percolator tab file restricted to 1000 rows"""
    return Path("data", "percolator-noSplit-extended-1000.tab")


@pytest.fixture
def percolator_extended_file_big():
    """Get the extended percolator tab file restricted to 10000 rows"""
    return Path("data", "percolator-noSplit-extended-10000.tab")


def test_rollup_10000(
    tmp_path, percolator_extended_file_small, percolator_extended_file_big
):
    """Test that rollup for intermediate levels works."""
    path = tmp_path

    common_params = [
        ("--dest_dir", path),
        ("--max_workers", 8),
        ("--test_fdr", 0.10),
        ("--train_fdr", 0.9),
        ("--verbosity", 2),
        ("--subset_max_train", 4000),
        ("--max_iter", 10),
        "--ensemble",
        "--keep_decoys",
    ]

    use_proteins = False
    if use_proteins:
        fasta = Path("data", "human_sp_td.fasta")
        common_params += ["--proteins", fasta]

    params = [
        percolator_extended_file_small,
        *common_params,
        "--save_models",
    ]
    run_mokapot_cli(params)
    assert file_exist(path, "targets.psms.csv")
    assert file_exist(path, "targets.peptides.csv")
    assert file_exist(path, "targets.precursors.csv")
    assert file_exist(path, "targets.modifiedpeptides.csv")
    assert file_exist(path, "targets.peptidegroups.csv")


def test_extra_cols(tmp_path):
    """Test that two identical mokapot runs produce same results."""

    extended_file = Path("data", "percolator-noSplit-extended-10000.tab")
    non_extended_file = Path("data", "percolator-noSplit-non-extended-10000.tab")

    params = [
        ("--dest_dir", tmp_path),
        ("--subset_max_train", "2500"),
        ("--max_workers", "8"),
        ("--max_iter", "2"),
        "--keep_decoys",
        "--ensemble",
    ]
    run_mokapot_cli([extended_file] + params + ["--file_root", "run1"])
    run_mokapot_cli([non_extended_file] + params + ["--file_root", "run2"])

    assert file_exist(tmp_path, "run1.targets.peptides.csv")
    assert file_exist(tmp_path, "run1.decoys.peptides.csv")
    assert file_exist(tmp_path, "run1.targets.psms.csv")
    assert file_exist(tmp_path, "run1.decoys.psms.csv")

    assert file_exist(tmp_path, "run2.targets.peptides.csv")
    assert file_exist(tmp_path, "run2.decoys.peptides.csv")
    assert file_exist(tmp_path, "run2.targets.psms.csv")
    assert file_exist(tmp_path, "run2.decoys.psms.csv")

    df_run1_t_psms = pd.read_csv(tmp_path / "run1.targets.psms.csv", sep="\t")
    df_run2_t_psms = pd.read_csv(tmp_path / "run2.targets.psms.csv", sep="\t")
    pd.testing.assert_frame_equal(
        df_run1_t_psms[df_run2_t_psms.columns], df_run2_t_psms
    )

    df_run1_t_peptides = pd.read_csv(
        tmp_path / "run1.targets.peptides.csv", sep="\t"
    )
    df_run2_t_peptides = pd.read_csv(
        tmp_path / "run2.targets.peptides.csv", sep="\t"
    )
    pd.testing.assert_frame_equal(
        df_run1_t_peptides[df_run2_t_peptides.columns], df_run2_t_peptides
    )


def test_deduplication(tmp_path):
    """Test that deduplication of psms works."""
    path = tmp_path
    file = Path("data") / "scope2_FP97AA.pin"

    params = [
        file,
        ("--dest_dir", path),
        "--ensemble",
        "--keep_decoys",
        "--skip_rollup",
        ("--peps_algorithm", "hist_nnls"),
    ]

    dedup_params = params + ["--file_root", "dedup"]
    run_mokapot_cli(dedup_params)

    no_dedup_params = params + [
        "--file_root",
        "nodedup",
        "--skip_deduplication",
    ]
    run_mokapot_cli(no_dedup_params)

    assert file_exist(path, "dedup.targets.psms.csv")
    assert file_exist(path, "nodedup.targets.psms.csv")
    assert file_approx_len(path, "dedup.targets.psms.csv", 5549)
    assert file_approx_len(path, "nodedup.targets.psms.csv", 37814)

    # Check that we have really the right number of results:
    # Without deduplication, all original targets have to be in the
    # targets.psms output file. With deduplication, either the target or the
    # decoy may survive the deduplication process, and thus we can only check
    # for the sum of targets and decoys in both output files.
    df = pd.read_csv(file, sep="\t", usecols=["Label", "ScanNr", "ExpMass"])
    nodedup_count = len(df[df.Label == 1])
    dedup_count = len(df.drop_duplicates(subset=["ScanNr", "ExpMass"]))

    lines1a = count_lines(path, "dedup.targets.psms.csv") - 1
    lines1b = count_lines(path, "dedup.decoys.psms.csv") - 1
    lines2 = count_lines(path, "nodedup.targets.psms.csv") - 1

    assert lines1a + lines1b == dedup_count
    assert lines2 == nodedup_count

    assert lines1a < lines2


def test_streaming(tmp_path, percolator_extended_file_big):
    """Test that streaming of confidence assignments works."""
    path = tmp_path
    file = Path("data") / "scope2_FP97AA.pin"

    base_params = [
        file,
        ("--dest_dir", path),
        "--ensemble",
        "--keep_decoys",
        "--skip_rollup",
        "--skip_deduplication",
    ]

    # Check that correct peps algorithm is used (need to run in the same
    # process, so that we can catch the exception)
    params = base_params + ["--file_root", "stream", "--stream_confidence"]
    with pytest.raises(ValueError, match="hist_nnls"):
        run_mokapot_cli(params, run_in_subprocess=False)

    # todo: discuss: with deduplication there are sometimes differences in
    #  the results, which can happen when different psms with the same
    #  spectrum id and exp_mass happen to have the same score - often one
    #  target and the corresponding decoy. Then one or the  other may
    #  survive deduplication, depending on which chunk it landed in, leading
    #  to different psms lists.

    # Run mokapot with streaming
    with run_with_chunk_size(100000):
        base_params += ["--peps_algorithm", "hist_nnls"]
        params = base_params + [
            "--file_root",
            "stream1",
            "--stream_confidence",
        ]
        run_mokapot_cli(params)
        assert file_exist(path, "stream1.targets.psms.csv")

    # Set chunk size so low, that chunking really kicks in
    with run_with_chunk_size(123):
        base_params += ["--peps_algorithm", "hist_nnls"]
        params = base_params + [
            "--file_root",
            "stream2",
            "--stream_confidence",
        ]
        run_mokapot_cli(params)
        assert file_exist(path, "stream2.targets.psms.csv")

    # Run mokapot without streaming
    params = base_params + ["--file_root", "base"]
    run_mokapot_cli(params)
    assert file_exist(path, "base.targets.psms.csv")

    # compare results
    def read_tsv(filename):
        return pd.read_csv(path / filename, sep="\t", index_col=False)

    df_streamed = read_tsv("stream1.targets.psms.csv")
    df_streamed2 = read_tsv("stream2.targets.psms.csv")

    pd.testing.assert_frame_equal(df_streamed, df_streamed2)

    df_base = read_tsv("base.targets.psms.csv")

    qvc = "q-value"
    pvc = "posterior_error_prob"
    pd.testing.assert_frame_equal(
        df_streamed.drop(columns=[qvc, pvc]), df_base.drop(columns=[qvc, pvc])
    )

    pd.testing.assert_series_equal(df_streamed[qvc], df_base[qvc], atol=0.01)
    pd.testing.assert_series_equal(df_streamed[pvc], df_base[pvc], atol=0.06)
