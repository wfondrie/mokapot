"""
These tests verify that the CLI works as expected.

At least for now, they do not check the correctness of the
output, just that the expect outputs are created.
"""
import subprocess
from pathlib import Path

import pytest

# Warnings are errors for these tests
pytestmark = pytest.mark.filterwarnings("error")


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
    cmd = ["mokapot", scope_files[0], "--dest_dir", tmp_path]
    subprocess.run(cmd, check=True)
    assert Path(tmp_path, "targets.psms").exists()
    assert Path(tmp_path, "targets.peptides").exists()


def test_cli_options(tmp_path, scope_files):
    """Test non-defaults"""
    cmd = [
        "mokapot",
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

    subprocess.run(cmd, check=True)
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
    cmd = [
        "mokapot",
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

    subprocess.run(cmd, check=True)
    assert Path(tmp_path, "blah.targets.psms").exists()
    assert Path(tmp_path, "blah.targets.peptides").exists()
    assert not Path(tmp_path, "blah.targets.decoy.psms").exists()
    assert not Path(tmp_path, "blah.targets.decoy.peptides").exists()

    # Test that decoys are also in the output when --keep_decoys is used
    cmd += ["--keep_decoys"]
    subprocess.run(cmd, check=True)
    assert Path(tmp_path, "blah.decoys.psms").exists()
    assert Path(tmp_path, "blah.decoys.peptides").exists()


def test_cli_fasta(tmp_path, phospho_files):
    """Test that proteins happen"""
    cmd = [
        "mokapot",
        phospho_files[0],
        "--dest_dir",
        tmp_path,
        "--proteins",
        phospho_files[1],
        "--max_iter",
        "1",
    ]

    subprocess.run(cmd, check=True)
    assert Path(tmp_path, "targets.psms").exists()
    assert Path(tmp_path, "targets.peptides").exists()
    assert Path(tmp_path, "targets.proteins").exists()


def test_cli_saved_models(tmp_path, phospho_files):
    """Test that saved_models works"""
    cmd = [
        "mokapot",
        phospho_files[0],
        "--dest_dir",
        tmp_path,
        "--test_fdr",
        "0.01",
    ]

    subprocess.run(cmd + ["--save_models"], check=True)

    cmd += ["--load_models", *list(Path(tmp_path).glob("*.pkl"))]
    subprocess.run(cmd, check=True)
    assert Path(tmp_path, "targets.psms").exists()
    assert Path(tmp_path, "targets.peptides").exists()


def test_cli_plugins(tmp_path, phospho_files):
    try:
        import mokapot_ctree
    except ImportError:
        mokapot_ctree = None

    if mokapot_ctree is None:
        pytest.skip("Testing plugins is not installed")

    cmd = [
        "mokapot",
        phospho_files[0],
        "--dest_dir",
        tmp_path,
        "--test_fdr",
        "0.01",
    ]

    res = subprocess.run(cmd + ["--help"], check=True, capture_output=True)
    assert "--yell" in res.stdout.decode()

    # Make sure it does not yell when the plugin is not loaded explicitly
    res = subprocess.run(cmd, check=True, capture_output=True)
    assert "Yelling at the user" not in res.stderr.decode()

    # Check that it does yell when the plugin is loaded and arg is requested
    cmd += ["--plugin", "mokapot_ctree", "--yell"]
    res = subprocess.run(cmd, check=True, capture_output=True)
    assert "Yelling at the user" in res.stderr.decode()


def test_cli_skip_deduplication(tmp_path, phospho_files):
    """Test that peptides file results is skipped when using skip_deduplication"""
    cmd = [
        "mokapot",
        phospho_files[0],
        "--dest_dir",
        tmp_path,
        "--test_fdr",
        "0.01",
        "--skip_deduplication",
    ]

    subprocess.run(cmd, check=True)

    assert Path(tmp_path, "targets.psms").exists()
    assert not Path(tmp_path, "targets.peptides").exists()


def test_cli_ensemble(tmp_path, phospho_files):
    """Test ensemble flag"""
    cmd = [
        "mokapot",
        phospho_files[0],
        "--dest_dir",
        tmp_path,
        "--test_fdr",
        "0.01",
        "--ensemble",
    ]

    subprocess.run(cmd, check=True)
    assert Path(tmp_path, "targets.psms").exists()
    assert Path(tmp_path, "targets.peptides").exists()


def test_cli_rescale(tmp_path, scope_files):
    """Test that rescale works"""
    cmd = [
        "mokapot",
        scope_files[1],
        "--dest_dir",
        tmp_path,
        "--test_fdr",
        "0.01",
    ]

    subprocess.run(cmd + ["--save_models"], check=True)

    cmd = [
        "mokapot",
        scope_files[0],
        "--dest_dir",
        tmp_path,
        "--test_fdr",
        "0.01",
        "--load_models",
        *list(Path(tmp_path).glob("*.pkl")),
        "--rescale",
    ]
    subprocess.run(cmd, check=True)
    assert Path(tmp_path, "targets.psms").exists()
    assert Path(tmp_path, "targets.peptides").exists()

    subprocess.run(cmd + ["--subset_max_rescale", "5000"], check=True)
    assert Path(tmp_path, "targets.psms").exists()
    assert Path(tmp_path, "targets.peptides").exists()
