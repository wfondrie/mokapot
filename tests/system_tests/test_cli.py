"""
These tests verify that the CLI works as expected.

At least for now, they do not check the correctness of the
output, just that the expect outputs are created.
"""
import subprocess
from pathlib import Path

import pandas as pd
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


@pytest.fixture
def pepxml_file():
    """Get the pepxml file"""
    pepxml = Path("data", "msfragger.pepXML")
    return pepxml


def test_basic_cli(tmp_path, scope_files):
    """Test that basic cli works."""
    cmd = ["mokapot", scope_files[0], "--dest_dir", tmp_path]
    subprocess.run(cmd, check=True)
    assert Path(tmp_path, "mokapot.psms.txt").exists()
    assert Path(tmp_path, "mokapot.peptides.txt").exists()


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

    assert Path(tmp_path, f"blah.{file_bases[0]}.mokapot.psms.txt").exists()
    assert Path(
        tmp_path, f"blah.{file_bases[0]}.mokapot.peptides.txt"
    ).exists()
    assert Path(tmp_path, f"blah.{file_bases[1]}.mokapot.psms.txt").exists()
    assert Path(
        tmp_path, f"blah.{file_bases[1]}.mokapot.peptides.txt"
    ).exists()

    # Test keep_decoys:
    assert Path(
        tmp_path, f"blah.{file_bases[0]}.mokapot.decoy.psms.txt"
    ).exists()
    assert Path(
        tmp_path, f"blah.{file_bases[0]}.mokapot.decoy.peptides.txt"
    ).exists()
    assert Path(
        tmp_path, f"blah.{file_bases[1]}.mokapot.decoy.psms.txt"
    ).exists()
    assert Path(
        tmp_path, f"blah.{file_bases[1]}.mokapot.decoy.peptides.txt"
    ).exists()


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
    assert Path(tmp_path, "blah.mokapot.psms.txt").exists()
    assert Path(tmp_path, "blah.mokapot.peptides.txt").exists()
    assert not Path(tmp_path, "blah.mokapot.decoy.psms.txt").exists()
    assert not Path(tmp_path, "blah.mokapot.decoy.peptides.txt").exists()

    # Test that decoys are also in the output when --keep_decoys is used
    cmd += ["--keep_decoys"]
    subprocess.run(cmd, check=True)
    assert Path(tmp_path, "blah.mokapot.decoy.psms.txt").exists()
    assert Path(tmp_path, "blah.mokapot.decoy.peptides.txt").exists()


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
    assert Path(tmp_path, "mokapot.psms.txt").exists()
    assert Path(tmp_path, "mokapot.peptides.txt").exists()
    assert Path(tmp_path, "mokapot.proteins.txt").exists()


def test_cli_pepxml(tmp_path, pepxml_file):
    """Test that finding the correct parser works"""
    cmd = [
        "mokapot",
        pepxml_file,
        "--dest_dir",
        tmp_path,
        "--max_iter",
        "1",
        "--decoy_prefix",
        "rev_",
    ]

    subprocess.run(cmd, check=True)
    unbinned_file = Path(tmp_path, "mokapot.peptides.txt")
    assert Path(tmp_path, "mokapot.psms.txt").exists()
    assert unbinned_file.exists()

    cmd += ["--open_modification_bin_size", "0.01", "--file_root", "binned"]
    subprocess.run(cmd, check=True)
    binned_file = Path(tmp_path, "binned.mokapot.peptides.txt")
    assert Path(tmp_path, "binned.mokapot.psms.txt").exists()
    assert binned_file.exists()

    # If binning was successful, there should be more distinct peptides:
    unbinned = pd.read_csv(unbinned_file, sep="\t")
    binned = pd.read_csv(binned_file, sep="\t")
    assert len(binned) > len(unbinned)


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
    assert Path(tmp_path, "mokapot.psms.txt").exists()
    assert Path(tmp_path, "mokapot.peptides.txt").exists()


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
