"""
These tests verify that the CLI works as expected.

At least for now, they do not check the correctness of the
output, just that the expect outputs are created.
"""

from pathlib import Path

import pandas as pd
import pytest

from ..helpers.cli import run_mokapot_cli
from ..helpers.utils import file_approx_len, file_exist, file_missing


@pytest.fixture
def scope_files():
    """Get the scope-ms files"""
    return sorted(list(Path("data").glob("scope*.pin")))


@pytest.fixture
def phospho_files():
    """Get the phospho file and fasta"""
    pin = Path("data", "phospho_rep1.pin")
    fasta = Path("data", "human_sp_td.fasta")
    return pin, fasta


def test_basic_cli(tmp_path, scope_files):
    """Test that basic cli works."""
    params = [scope_files[0], "--dest_dir", tmp_path, "--verbosity", 3]
    run_mokapot_cli(params)
    assert file_approx_len(tmp_path, "targets.psms.csv", 5487)
    assert file_approx_len(tmp_path, "targets.peptides.csv", 5183)

    targets_psms_df = pd.read_csv(
        Path(tmp_path, "targets.psms.csv"), sep="\t", index_col=None
    )
    assert targets_psms_df.columns.values.tolist() == [
        "PSMId",
        "peptide",
        "score",
        "q-value",
        "posterior_error_prob",
        "proteinIds",
    ]
    assert len(targets_psms_df.index) >= 5000

    assert targets_psms_df.iloc[0, 0] == "target_0_11040_3_-1"
    assert targets_psms_df.iloc[0, 5] == "sp|P10809|CH60_HUMAN"


def test_cli_options(tmp_path, scope_files):
    """Test non-defaults"""
    params = [
        scope_files[0],
        scope_files[1],
        ("--dest_dir", tmp_path),
        ("--file_root", "blah"),
        ("--train_fdr", "0.2"),
        ("--test_fdr", "0.1"),
        ("--seed", "100"),
        ("--direction", "RefactoredXCorr"),
        ("--folds", "2"),
        ("-v", "1"),
        ("--max_iter", "1"),
        "--keep_decoys",
        ("--subset_max_train", "50000"),
        ("--max_workers", "3"),
    ]

    run_mokapot_cli(params)
    filebase = ["blah." + f.name.split(".")[0] for f in scope_files[0:2]]

    assert file_approx_len(tmp_path, f"{filebase[0]}.targets.psms.csv", 5490)
    assert file_approx_len(tmp_path, f"{filebase[0]}.targets.peptides.csv", 5194)
    assert file_approx_len(tmp_path, f"{filebase[1]}.targets.psms.csv", 4659)
    assert file_approx_len(tmp_path, f"{filebase[1]}.targets.peptides.csv", 4406)

    # Test keep_decoys:
    assert file_approx_len(tmp_path, f"{filebase[0]}.decoys.psms.csv", 2090)
    assert file_approx_len(tmp_path, f"{filebase[0]}.decoys.peptides.csv", 2037)
    assert file_approx_len(tmp_path, f"{filebase[1]}.decoys.psms.csv", 1806)
    assert file_approx_len(tmp_path, f"{filebase[1]}.decoys.peptides.csv", 1755)


def test_cli_aggregate(tmp_path, scope_files):
    """Test that aggregate results in one result file."""
    params = [
        scope_files[0],
        scope_files[1],
        ("--dest_dir", tmp_path),
        ("--file_root", "blah"),
        "--aggregate",
        ("--max_iter", "1"),
    ]

    run_mokapot_cli(params)

    # Line counts were determined by one (hopefully correct) test run
    assert file_approx_len(tmp_path, "blah.targets.psms.csv", 10256)
    assert file_approx_len(tmp_path, "blah.targets.peptides.csv", 9663)
    assert file_missing(tmp_path, "blah.decoys.psms.csv")
    assert file_missing(tmp_path, "blah.decoys.peptides.csv")

    # Test that decoys are also in the output when --keep_decoys is used
    params += ["--keep_decoys"]
    run_mokapot_cli(params)
    assert file_approx_len(tmp_path, "blah.targets.psms.csv", 10256)
    assert file_approx_len(tmp_path, "blah.targets.peptides.csv", 9663)
    assert file_approx_len(tmp_path, "blah.decoys.psms.csv", 3787)
    assert file_approx_len(tmp_path, "blah.decoys.peptides.csv", 3694)


def test_cli_fasta(tmp_path, phospho_files):
    """Test that proteins happen"""
    params = [
        phospho_files[0],
        ("--dest_dir", tmp_path),
        ("--proteins", phospho_files[1]),
        ("--max_iter", "1"),
    ]

    run_mokapot_cli(params)

    assert file_approx_len(tmp_path, "targets.psms.csv", 42331)
    assert file_approx_len(tmp_path, "targets.peptides.csv", 33538)
    assert file_approx_len(tmp_path, "targets.proteins.csv", 7827)


def test_cli_saved_models(tmp_path, phospho_files):
    """Test that saved_models works"""
    params = [
        phospho_files[0],
        ("--dest_dir", tmp_path),
        ("--test_fdr", "0.01"),
    ]

    run_mokapot_cli(params + ["--save_models"])

    params += ["--load_models", *list(Path(tmp_path).glob("*.pkl"))]
    run_mokapot_cli(params)
    assert file_approx_len(tmp_path, "targets.psms.csv", 42331)
    assert file_approx_len(tmp_path, "targets.peptides.csv", 33538)


def test_cli_skip_rollup(tmp_path, phospho_files):
    """Test that peptides file results is skipped when using skip_rollup"""
    params = [
        phospho_files[0],
        ("--dest_dir", tmp_path),
        ("--test_fdr", "0.01"),
        "--skip_rollup",
    ]

    run_mokapot_cli(params)

    assert file_approx_len(tmp_path, "targets.psms.csv", 42331)
    assert file_missing(tmp_path, "targets.peptides.csv")


def test_cli_ensemble(tmp_path, phospho_files):
    """Test ensemble flag"""
    params = [
        phospho_files[0],
        ("--dest_dir", tmp_path),
        ("--test_fdr", "0.01"),
        "--ensemble",
    ]

    run_mokapot_cli(params)
    assert file_approx_len(tmp_path, "targets.psms.csv", 42331)
    assert file_approx_len(tmp_path, "targets.peptides.csv", 33538)
    # todo: nice to have: we should also test the *contents* of the files


def test_cli_bad_input(tmp_path):
    """Test with problematic input files"""

    # The input file contains "integers" of the form `6d05`, which caused
    # problems with certain readers

    params = [
        Path("data") / "percolator-noSplit-extended-201-bad.tab",
        ("--dest_dir", tmp_path),
        ("--train_fdr", "0.05"),
        "--ensemble",
    ]

    run_mokapot_cli(params)
    assert file_exist(tmp_path, "targets.psms.csv")
    assert file_exist(tmp_path, "targets.peptides.csv")


def test_negative_features(tmp_path, psm_df_1000):
    """Test that best feature selection works."""

    def make_pin_file(filename, desc, seed=None):
        import numpy as np

        df = psm_df_1000[1].copy()
        if seed is not None:
            np.random.seed(seed)
        scores = df["score"]
        targets = df["target"]
        df.drop(columns=["score", "score2", "target"], inplace=True)
        df["Label"] = targets * 1
        df["feat"] = scores * (1 if desc else -1)
        df["scannr"] = np.random.randint(0, 1000, 1000)
        file = tmp_path / filename
        df.to_csv(file, sep="\t", index=False)
        return file, df

    file1bad, df1b = make_pin_file("test1bad.pin", desc=True, seed=123)
    file2bad, df2b = make_pin_file("test2bad.pin", desc=False, seed=123)
    file1, df1 = make_pin_file("test1.pin", desc=True, seed=126)
    file2, df2 = make_pin_file("test2.pin", desc=False, seed=126)

    def read_result(filename):
        df = pd.read_csv(tmp_path / filename, sep="\t", index_col=False)
        return df.sort_values(by="PSMId").reset_index(drop=True)

    def mean_scores(str):
        def mean_score(file):
            psms_df = read_result(file)
            return psms_df.score.values.mean()

        target_mean = mean_score(f"{str}.targets.psms.csv")
        decoy_mean = mean_score(f"{str}.decoys.psms.csv")
        return (target_mean, decoy_mean, target_mean > decoy_mean)

    common_params = [
        ("--dest_dir", tmp_path),
        ("--train_fdr", 0.05),
        ("--test_fdr", 0.05),
        ("--peps_algorithm", "hist_nnls"),
        "--keep_decoys",
    ]

    # Test with data where a "good" model can be trained. Once with the normal
    # feat column, once with the feat column negated.
    params = [file1, "--file_root", "test1"]
    run_mokapot_cli(params + common_params)

    params = [file2, "--file_root", "test2"]
    run_mokapot_cli(params + common_params)

    psms_df1 = read_result("test1.targets.psms.csv")
    psms_df2 = read_result("test2.targets.psms.csv")
    pd.testing.assert_frame_equal(psms_df1, psms_df2)

    # In the case below, the trained model performs worse than just using the
    # feat column, so the score is just the same as the feature.

    params = [file1bad, "--file_root", "test1b"]
    run_mokapot_cli(params + common_params)

    params = [file2bad, "--file_root", "test2b"]
    run_mokapot_cli(params + common_params)

    psms_df1b = read_result("test1b.targets.psms.csv")
    psms_df2b = read_result("test2b.targets.psms.csv")
    pd.testing.assert_frame_equal(psms_df1b, psms_df2b)

    # Let's check now that the score columns are indeed equal to the
    # normal/negated feature column

    feature_col1 = df1b[df1b.Label == 1].sort_values(by="specid").feat
    score_col1 = psms_df1b.sort_values(by="PSMId").score
    pd.testing.assert_series_equal(
        score_col1, feature_col1, check_index=False, check_names=False
    )

    feature_col2 = df2b[df2b.Label == 1].sort_values(by="specid").feat
    score_col2 = psms_df2b.sort_values(by="PSMId").score
    pd.testing.assert_series_equal(
        score_col2, -feature_col2, check_index=False, check_names=False
    )

    # Lastly, test that the targets have a higher mean score than the decoys
    assert mean_scores("test1")[2]
    assert mean_scores("test2")[2]
    assert mean_scores("test1b")[2]
    assert mean_scores("test2b")[2]  # This one is the most likely to fail


def test_cli_help():
    """Test that help works"""

    # Triggering help should raise a SystemExit
    with pytest.raises(SystemExit):
        run_mokapot_cli(["-h"], run_in_subprocess=False)

    # This is caught, when run in a subprocess, and we can verify
    # stdout (only some contents to make sure it makes sense)
    res = run_mokapot_cli(["--help"], run_in_subprocess=True, capture_output=True)
    stdout = res["stdout"]
    assert "usage: mokapot" in stdout
    assert "Written by" in stdout
    assert "--enzyme" in stdout
