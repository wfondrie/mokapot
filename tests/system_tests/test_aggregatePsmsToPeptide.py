"""
These tests verify that the aggregatePsmsToPeptides executable works as expected.
"""
import subprocess
from pathlib import Path

import pytest

# Warnings are errors for these tests
pytestmark = pytest.mark.filterwarnings("error")


def test_basic_cli(tmp_path, targets_decoys_psms_scored):
    """Test that basic cli works."""
    cmd = [
        "python",
        "-m",
        "mokapot.aggregatePsmsToPeptides",
        "--targets_psms",
        targets_decoys_psms_scored[0],
        "--decoys_psms",
        targets_decoys_psms_scored[1],
        "--dest_dir",
        tmp_path,
    ]
    subprocess.run(cmd, check=True)
    assert Path(tmp_path, "targets.peptides").exists()


def test_cli_keep_decoys(tmp_path, targets_decoys_psms_scored):
    """Test that --keep_decoys works."""
    cmd = [
        "python",
        "-m",
        "mokapot.aggregatePsmsToPeptides",
        "--targets_psms",
        targets_decoys_psms_scored[0],
        "--decoys_psms",
        targets_decoys_psms_scored[1],
        "--dest_dir",
        tmp_path,
        "--keep_decoys",
    ]
    subprocess.run(cmd, check=True)
    assert Path(tmp_path, "targets.peptides").exists()
    assert Path(tmp_path, "decoys.peptides").exists()


def test_non_default_fdr(tmp_path, targets_decoys_psms_scored):
    """Test non-defaults"""
    cmd = [
        "python",
        "-m",
        "mokapot.aggregatePsmsToPeptides",
        "--targets_psms",
        targets_decoys_psms_scored[0],
        "--decoys_psms",
        targets_decoys_psms_scored[1],
        "--test_fdr",
        "0.1",
        "--dest_dir",
        tmp_path,
        "--keep_decoys",
    ]

    subprocess.run(cmd, check=True)
    assert Path(tmp_path, "targets.peptides").exists()
    assert Path(tmp_path, "decoys.peptides").exists()
