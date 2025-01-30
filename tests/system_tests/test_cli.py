"""
These tests verify that the CLI works as expected.

At least for now, they do not check the correctness of the
output, just that the expect outputs are created.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ..helpers.cli import run_mokapot_cli
from ..helpers.utils import (
    ColumnValidator,
    TableValidator,
    file_approx_len,
    file_exist,
    file_missing,
)


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

    assert file_approx_len(tmp_path, "targets.peptides.tsv", 5183)
    assert file_approx_len(tmp_path, "targets.psms.tsv", 5487)

    targets_psms_df = pd.read_csv(
        Path(tmp_path, "targets.psms.tsv"), sep="\t", index_col=None
    )
    assert targets_psms_df.columns.values.tolist() == [
        "SpecId",
        "ScanNr",
        "ExpMass",
        "peptide",
        "score",
        "mokapot_qvalue",
        "posterior_error_prob",
        "proteinIds",
    ]
    assert len(targets_psms_df.index) >= 5000

    assert targets_psms_df.iloc[0, 0] == "target_0_11040_3_-1"
    # assert targets_psms_df.iloc[0, 5] == "sp|P10809|CH60_HUMAN"
    assert targets_psms_df["proteinIds"].iloc[0] == "sp|P10809|CH60_HUMAN"


@pytest.mark.slow
def test_cli_options(tmp_path, scope_files):
    """Test non-defaults"""
    files_use = scope_files[0:2]
    file_root = "blah"
    params = [
        [str(f) for f in files_use],
        ("--dest_dir", tmp_path),
        ("--file_root", file_root),
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
        ("--verbosity", "3"),
    ]

    run_mokapot_cli(params)
    filebase = [f"{file_root}." + f.name.split(".")[0] for f in files_use]

    assert file_approx_len(tmp_path, f"{filebase[0]}.targets.psms.tsv", 5490)
    assert file_approx_len(
        tmp_path, f"{filebase[0]}.targets.peptides.tsv", 5194
    )
    assert file_approx_len(tmp_path, f"{filebase[1]}.targets.psms.tsv", 4659)
    assert file_approx_len(
        tmp_path, f"{filebase[1]}.targets.peptides.tsv", 4406
    )

    # Test keep_decoys:
    assert file_approx_len(tmp_path, f"{filebase[0]}.decoys.psms.tsv", 2090)
    assert file_approx_len(
        tmp_path, f"{filebase[0]}.decoys.peptides.tsv", 2037
    )
    assert file_approx_len(tmp_path, f"{filebase[1]}.decoys.psms.tsv", 1806)
    assert file_approx_len(
        tmp_path, f"{filebase[1]}.decoys.peptides.tsv", 1755
    )


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
    assert file_approx_len(tmp_path, "blah.targets.psms.tsv", 10256)
    assert file_approx_len(tmp_path, "blah.targets.peptides.tsv", 9663)
    assert file_missing(tmp_path, "blah.decoys.psms.tsv")
    assert file_missing(tmp_path, "blah.decoys.peptides.tsv")

    # Test that decoys are also in the output when --keep_decoys is used
    params += ["--keep_decoys"]
    run_mokapot_cli(params)
    assert file_approx_len(tmp_path, "blah.targets.psms.tsv", 10256)
    assert file_approx_len(tmp_path, "blah.targets.peptides.tsv", 9663)
    assert file_approx_len(tmp_path, "blah.decoys.psms.tsv", 3787)
    assert file_approx_len(tmp_path, "blah.decoys.peptides.tsv", 3694)


def test_cli_fasta(tmp_path, phospho_files):
    """Test that proteins happen"""
    params = [
        phospho_files[0],
        ("--dest_dir", tmp_path),
        ("--proteins", phospho_files[1]),
        ("--max_iter", "1"),
    ]

    run_mokapot_cli(params)

    assert file_approx_len(tmp_path, "targets.psms.tsv", 42331)
    assert file_approx_len(tmp_path, "targets.peptides.tsv", 33538)
    assert file_approx_len(tmp_path, "targets.proteins.tsv", 7827)

    # Check the contents of the tables.
    psm_df = pd.read_csv(tmp_path / "targets.psms.tsv", sep="\t")
    psm_validator = TableValidator(
        columns=[
            ColumnValidator(
                name="SpecId",
                col_type="O",
                value_range=None,
                allow_missing=False,
            ),
            ColumnValidator(
                name="ScanNr",
                col_type=int,
                value_range=(1, 70_000),
                allow_missing=False,
            ),
            ColumnValidator(
                name="ExpMass",
                col_type=float,
                value_range=(10.0, 10_000.0),
                allow_missing=False,
            ),
            ColumnValidator(
                name="peptide",
                col_type="O",
                value_range=None,
                allow_missing=False,
            ),
            ColumnValidator(
                # Should this be renamed as "mokapot_score"
                name="score",
                col_type=float,
                value_range=(-100.0, 100.0),
                allow_missing=False,
            ),
            ColumnValidator(
                name="mokapot_qvalue",
                col_type=float,
                value_range=(1e-32, 1.0),
                allow_missing=False,
            ),
            ColumnValidator(
                name="posterior_error_prob",
                col_type=float,
                value_range=(1e-32, 1.0),
                allow_missing=False,
            ),
            ColumnValidator(
                name="proteinIds",
                col_type="O",
                value_range=None,
                allow_missing=False,
            ),
        ],
        allow_extra=False,
        row_range=(42330 - 100, 42330 + 100),
    )
    psm_validator.validate(psm_df)

    peptide_df = pd.read_csv(tmp_path / "targets.peptides.tsv", sep="\t")
    peptide_validator = TableValidator(
        columns=[
            ColumnValidator(
                name="SpecId",
                col_type="object",
                value_range=("target_0_10000_3_-1", "target_0_9999_4_-1"),
                allow_missing=False,
            ),
            ColumnValidator(
                name="ScanNr",
                col_type="int64",
                value_range=(1, 70_000),
                allow_missing=False,
            ),
            ColumnValidator(
                name="ExpMass",
                col_type="float64",
                value_range=(100.0, 7000.0),
                allow_missing=False,
            ),
            ColumnValidator(
                name="peptide",
                col_type="object",
                value_range=(
                    # Shoudl this be returning the flanking aminoacids?
                    "-.MAAAAPNAGGSAPET[79.97]AGSAEAPLQYSLLLQY[79.97]LVGDKRQPR.L",
                    "R.Y[79.97]YGGGSEGGRAPK.R",
                ),
                allow_missing=False,
            ),
            ColumnValidator(
                name="score",
                col_type="float64",
                value_range=(-30.0, 30.0),
                allow_missing=False,
            ),
            ColumnValidator(
                name="mokapot_qvalue",
                col_type="float64",
                value_range=(1e-32, 1.0),
                allow_missing=False,
            ),
            ColumnValidator(
                name="posterior_error_prob",
                col_type="float64",
                value_range=(1e-32, 1.0),
                allow_missing=False,
            ),
            ColumnValidator(
                name="proteinIds",
                col_type="object",
                value_range=(
                    "sp|A0A075B6I4|LVX54_HUMAN",
                    "sp|Q9Y6Z5|AFDDT_HUMAN:decoy_sp|Q92824|PCSK5_HUMAN",
                ),
                allow_missing=False,
            ),
        ],
        allow_extra=False,
        row_range=(33537 - 100, 33537 + 100),
    )
    peptide_validator.validate(peptide_df)

    # Check the contents of the tables.
    protein_df = pd.read_csv(tmp_path / "targets.proteins.tsv", sep="\t")
    protein_validator = TableValidator(
        columns=[
            ColumnValidator(
                name="mokapot protein group",
                col_type="object",
                value_range=(
                    "sp|A0A075B6I4|LVX54_HUMAN",
                    "sp|Q9Y6Y8|S23IP_HUMAN",
                ),
                allow_missing=False,
            ),
            ColumnValidator(
                name="best peptide",
                col_type="object",
                value_range=(
                    "-.MAAAAPNAGGSAPET[79.97]AGSAEAPLQYSLLLQY[79.97]LVGDKRQPR.L",
                    "R.Y[79.97]S[79.97]QLISHQS[79.97]IHIGVK.P",
                ),
                allow_missing=False,
            ),
            ColumnValidator(
                name="stripped sequence",
                col_type="object",
                value_range=(
                    "AAAAAAAAAAAAAAAGAGAGAK",
                    "YYPTAEEVYGPEVETIVQEEDTQPLTEPIIKPVK",
                ),
                allow_missing=False,
            ),
            ColumnValidator(
                name="score",
                col_type="float64",
                value_range=(-100.0, 100.0),
                allow_missing=False,
            ),
            ColumnValidator(
                name="mokapot_qvalue",
                col_type="float64",
                value_range=(1e-32, 1.0),
                allow_missing=False,
            ),
            ColumnValidator(
                name="posterior_error_prob",
                col_type="float64",
                value_range=(1e-32, 1.0),
                allow_missing=False,
            ),
        ],
        allow_extra=False,
        row_range=(7826 - 100, 7826 + 100),
    )
    protein_validator.validate(protein_df)


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
    assert file_approx_len(tmp_path, "targets.psms.tsv", 42331)
    assert file_approx_len(tmp_path, "targets.peptides.tsv", 33538)


def test_cli_skip_rollup(tmp_path, phospho_files):
    """Test that peptides file results is skipped when using skip_rollup"""
    params = [
        phospho_files[0],
        ("--dest_dir", tmp_path),
        ("--test_fdr", "0.01"),
        "--skip_rollup",
    ]

    run_mokapot_cli(params)

    assert file_approx_len(tmp_path, "targets.psms.tsv", 42331)
    assert file_missing(tmp_path, "targets.peptides.tsv")


def test_cli_ensemble(tmp_path, phospho_files):
    """Test ensemble flag"""
    params = [
        phospho_files[0],
        ("--dest_dir", tmp_path),
        ("--test_fdr", "0.01"),
        "--ensemble",
    ]

    run_mokapot_cli(params)
    assert len(list(tmp_path.rglob("*"))) == 2
    assert file_approx_len(tmp_path, "targets.psms.tsv", 42331)
    assert file_approx_len(tmp_path, "targets.peptides.tsv", 33538)

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
    assert file_exist(tmp_path, "targets.psms.tsv")
    assert file_exist(tmp_path, "targets.peptides.tsv")


def test_negative_features(tmp_path, psm_df_1000):
    """Test that best feature selection works."""

    def make_pin_file(filename, desc, seed=None):
        # TODO use the builder function to make this one.
        import numpy as np

        pin, df, fasta, score_cols = psm_df_1000
        df = df.copy()

        if seed is not None:
            np.random.seed(seed)
        scores = df[score_cols[0]]
        targets = df["target"]
        df.drop(columns=score_cols + ["target"], inplace=True)
        df["Label"] = targets * 1  # Q: what does the *1 do ?
        df["feat"] = scores * (1 if desc else -1)
        # Q: Why is this re-shuffled?
        df["scannr"] = np.random.randint(0, 1000, 1000)
        file = tmp_path / filename
        cols_keep = [
            "PSMId",
            "scannr",
            # 'spectrum', # these 3 are detected as features
            # 'charge',   # so we remove them
            # 'specid',
            "calcmass",
            "expmass",
            "peptide",
            "Label",
            "filename",
            "ret_time",
            "feat",
            "proteins",
        ]
        df = df[cols_keep]
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

        target_mean = mean_score(f"{str}.targets.psms.tsv")
        decoy_mean = mean_score(f"{str}.decoys.psms.tsv")
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

    psms_df1 = read_result("test1.targets.psms.tsv")
    psms_df2 = read_result("test2.targets.psms.tsv")
    pd.testing.assert_frame_equal(psms_df1, psms_df2)

    # In the case below, the trained model performs worse than just using the
    # feat column, so the score is just the same as the feature.

    params = [file1bad, "--file_root", "test1b"]
    run_mokapot_cli(params + common_params)

    params = [file2bad, "--file_root", "test2b"]
    run_mokapot_cli(params + common_params)

    psms_df1b = read_result("test1b.targets.psms.tsv")
    psms_df2b = read_result("test2b.targets.psms.tsv")
    pd.testing.assert_frame_equal(psms_df1b, psms_df2b)

    sorted_df1b = df1b[df1b.Label == 1].sort_values(by="scannr")
    feature_col1 = sorted_df1b.feat
    sorted_psms_df1b = psms_df1b.sort_values(by="scannr")
    score_col1 = sorted_psms_df1b.score

    np.testing.assert_equal(
        np.argsort(feature_col1.to_numpy(), stable=True),
        np.argsort(score_col1.to_numpy(), stable=True),
    )

    # Q: is this meant to be the behavior? shouldnt the score be a
    #    scaled/calibrated version of the feature? thus not equal
    #    to the feature?
    pd.testing.assert_series_equal(
        score_col1, feature_col1, check_index=False, check_names=False
    )

    feature_col2 = df2b[df2b.Label == 1].sort_values(by="PSMId").feat
    score_col2 = psms_df2b.sort_values(by="PSMId").score
    pd.testing.assert_series_equal(
        score_col2, -feature_col2, check_index=False, check_names=False
    )

    # Lastly, test that the targets have a higher mean score than the decoys
    assert mean_scores("test1")[2]
    assert mean_scores("test2")[2]
    assert mean_scores("test1b")[2]
    assert mean_scores("test2b")[2]  # This one is the most likely to fail
