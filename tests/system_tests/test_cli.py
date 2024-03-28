"""
These tests verify that the CLI works as expected.

At least for now, they do not check the correctness of the
output, just that the expect outputs are created.
"""

from pathlib import Path

from ..helpers.cli import run_mokapot_cli

import pytest


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


def test_basic_cli(tmp_path, scope_files):
    """Test that basic cli works."""
    params = [scope_files[0], "--dest_dir", tmp_path]
    run_mokapot_cli(params)
    assert Path(tmp_path, "targets.psms").exists()
    assert Path(tmp_path, "targets.peptides").exists()


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

    # Test that decoys are also in the output when --keep_decoys is used
    params += ["--keep_decoys"]
    run_mokapot_cli(params)
    assert Path(tmp_path, "blah.decoys.psms").exists()
    assert Path(tmp_path, "blah.decoys.peptides").exists()


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


def test_cli_plugins(tmp_path, phospho_files):
    try:
        import mokapot_ctree
    except ImportError:
        mokapot_ctree = None

    if mokapot_ctree is None:
        pytest.skip("Testing plugins is not installed")

    params = [
        phospho_files[0],
        "--dest_dir",
        tmp_path,
        "--test_fdr",
        "0.01",
    ]

    res = run_mokapot_cli(params + ["--help"], capture_output=True)
    assert "--yell" in res['stderr']

    # Make sure it does not yell when the plugin is not loaded explicitly
    res = run_mokapot_cli(params, capture_output=True)
    assert "Yelling at the user" not in res['stderr']

    # Check that it does yell when the plugin is loaded and arg is requested
    params += ["--plugin", "mokapot_ctree", "--yell"]
    res = run_mokapot_cli(params, capture_output=True)
    assert "Yelling at the user" in res['stderr']


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


def test_cli_rescale(tmp_path, scope_files):
    """Test that rescale works"""
    params = [
        scope_files[1],
        "--dest_dir",
        tmp_path,
        "--test_fdr",
        "0.01",
        "--save_models"
    ]

    run_mokapot_cli(params)

    params = [
        scope_files[0],
        "--dest_dir",
        tmp_path,
        "--test_fdr",
        "0.01",
        "--load_models",
        *list(Path(tmp_path).glob("*.pkl")),
        "--rescale",
    ]
    run_mokapot_cli(params)
    assert Path(tmp_path, "targets.psms").exists()
    assert Path(tmp_path, "targets.peptides").exists()

    run_mokapot_cli(params + ["--subset_max_rescale", "5000"])
    assert Path(tmp_path, "targets.psms").exists()
    assert Path(tmp_path, "targets.peptides").exists()
