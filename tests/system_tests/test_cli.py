"""
These tests verify that the CLI works as expected.

At least for now, they do not check the correctness of the
output, just that the expect outputs are created.
"""

from pathlib import Path

from ..helpers.cli import run_mokapot_cli

import pytest
import pandas as pd


@pytest.fixture
def scope_files():
    """Get the scope-ms files"""
    return sorted(list(Path("data").glob("scope*")))


@pytest.fixture
def phospho_files():
    """Get the phospho file and fasta"""
    pin = Path("data", "phospho_rep1.pin")
    fasta = Path("data", "human_sp_td.fasta")
    return pin, fasta


def count_lines(path: Path):
    """Count the number of lines in a file.

    Parameters
    ----------
    path : Path
        The path to the file.

    Returns
    -------
    int
        The number of lines in the file.
    """
    with open(path, "r") as file:
        lines = file.readlines()
    return len(lines)


def test_basic_cli(tmp_path, scope_files):
    """Test that basic cli works."""
    params = [scope_files[0], "--dest_dir", tmp_path]
    run_mokapot_cli(params)
    assert Path(tmp_path, "targets.psms").exists()
    assert Path(tmp_path, "targets.peptides").exists()

    targets_psms_df = pd.read_csv(
        Path(tmp_path, "targets.psms"), sep="\t", index_col=None
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
        "--dest_dir",
        tmp_path,
        "--file_root",
        "blah",
        "--train_fdr",
        "0.2",
        "--test_fdr",
        "0.1",
        "--seed",
        "100",
        "--direction",
        "RefactoredXCorr",
        "--folds",
        "2",
        "-v",
        "1",
        "--max_iter",
        "1",
        "--keep_decoys",
        "--subset_max_train",
        "50000",
        "--max_workers",
        "3",
    ]

    run_mokapot_cli(params)
    file_bases = [f.name.split(".")[0] for f in scope_files[0:2]]

    assert Path(tmp_path, f"blah.{file_bases[0]}.targets.psms").exists()
    assert Path(tmp_path, f"blah.{file_bases[0]}.targets.peptides").exists()
    assert Path(tmp_path, f"blah.{file_bases[1]}.targets.psms").exists()
    assert Path(tmp_path, f"blah.{file_bases[1]}.targets.peptides").exists()

    # Test keep_decoys:
    assert Path(tmp_path, f"blah.{file_bases[0]}.decoys.psms").exists()
    assert Path(tmp_path, f"blah.{file_bases[0]}.decoys.peptides").exists()
    assert Path(tmp_path, f"blah.{file_bases[1]}.decoys.psms").exists()
    assert Path(tmp_path, f"blah.{file_bases[1]}.decoys.peptides").exists()


def test_cli_aggregate(tmp_path, scope_files):
    """Test that aggregate results in one result file."""
    params = [
        scope_files[0],
        scope_files[1],
        "--dest_dir",
        tmp_path,
        "--file_root",
        "blah",
        "--aggregate",
        "--max_iter",
        "1",
    ]

    run_mokapot_cli(params)
    assert Path(tmp_path, "blah.targets.psms").exists()
    assert Path(tmp_path, "blah.targets.peptides").exists()
    assert not Path(tmp_path, "blah.targets.decoy.psms").exists()
    assert not Path(tmp_path, "blah.targets.decoy.peptides").exists()

    # Line counts were determined by one hopefully correct test run
    # GT counts: 10256, 9663
    assert count_lines(Path(tmp_path, "blah.targets.psms")) in range(
        10256 - 100, 10256 + 100
    )
    assert count_lines(Path(tmp_path, "blah.targets.peptides")) in range(
        9663 - 50, 9663 + 50
    )

    # Test that decoys are also in the output when --keep_decoys is used
    params += ["--keep_decoys"]
    run_mokapot_cli(params)
    assert Path(tmp_path, "blah.decoys.psms").exists()
    assert Path(tmp_path, "blah.decoys.peptides").exists()

    assert count_lines(Path(tmp_path, "blah.decoys.psms")) in range(
        3787 - 50, 3787 + 50
    )
    assert count_lines(Path(tmp_path, "blah.decoys.peptides")) in range(
        3694 - 50, 3694 + 50
    )


def test_cli_fasta(tmp_path, phospho_files):
    """Test that proteins happen"""
    params = [
        phospho_files[0],
        "--dest_dir",
        tmp_path,
        "--proteins",
        phospho_files[1],
        "--max_iter",
        "1",
    ]

    run_mokapot_cli(params)
    assert Path(tmp_path, "targets.psms").exists()
    assert Path(tmp_path, "targets.peptides").exists()
    assert Path(tmp_path, "targets.proteins").exists()


def test_cli_saved_models(tmp_path, phospho_files):
    """Test that saved_models works"""
    params = [
        phospho_files[0],
        "--dest_dir",
        tmp_path,
        "--test_fdr",
        "0.01",
    ]

    run_mokapot_cli(params + ["--save_models"])

    params += ["--load_models", *list(Path(tmp_path).glob("*.pkl"))]
    run_mokapot_cli(params)
    assert Path(tmp_path, "targets.psms").exists()
    assert Path(tmp_path, "targets.peptides").exists()


def test_cli_skip_rollup(tmp_path, phospho_files):
    """Test that peptides file results is skipped when using skip_rollup"""
    params = [
        phospho_files[0],
        "--dest_dir",
        tmp_path,
        "--test_fdr",
        "0.01",
        "--skip_rollup",
    ]

    run_mokapot_cli(params)

    assert Path(tmp_path, "targets.psms").exists()
    assert not Path(tmp_path, "targets.peptides").exists()


def test_cli_ensemble(tmp_path, phospho_files):
    """Test ensemble flag"""
    params = [
        phospho_files[0],
        "--dest_dir",
        tmp_path,
        "--test_fdr",
        "0.01",
        "--ensemble",
    ]

    run_mokapot_cli(params)
    assert Path(tmp_path, "targets.psms").exists()
    assert Path(tmp_path, "targets.peptides").exists()
    # fixme: should also test the *contents* of the files


def test_cli_bad_input(tmp_path):
    """Test ensemble flag"""
    params = [
        Path("data") / "percolator-noSplit-extended-201-bad.tab",
        "--dest_dir",
        tmp_path,
        "--train_fdr",
        "0.05",
        "--ensemble",
    ]

    run_mokapot_cli(params)
    assert Path(tmp_path, "targets.psms").exists()
    assert Path(tmp_path, "targets.peptides").exists()
    # fixme: should also test the *contents* of the files
