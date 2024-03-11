"""
These tests verify that the aggregatePsmsToPeptides executable works as expected.
"""

import subprocess
from pathlib import Path
from mokapot.aggregatePsmsToPeptides import main as aggregate_main

import pytest

# Warnings are errors for these tests
pytestmark = pytest.mark.filterwarnings("error")

def run_aggreate_cli(params, run_in_subprocess=True):
    """
    Run the aggregatePsmsToPeptides command either in a subprocess or as direct
    call to the main function.

    Parameters
    ----------
    params : list
        List of parameters to be passed to the `mokapot.aggregatePsmsToPeptides` command line tool.

    run_in_subprocess : bool, optional
        Indicates whether the command should be run in a separate subprocess. Default is `False`.
    """
    if run_in_subprocess:
        cmd = [
            "python",
            "-m",
            "mokapot.aggregatePsmsToPeptides"] + params
        subprocess.run(cmd, check=True)
    else:
        aggregate_main([str(c) for c in params])


def test_basic_cli(tmp_path, targets_decoys_psms_scored):
    """Test that basic cli works."""
    cmd = [
        "--targets_psms",
        targets_decoys_psms_scored[0],
        "--decoys_psms",
        targets_decoys_psms_scored[1],
        "--dest_dir",
        tmp_path,
    ]
    run_aggreate_cli(cmd)
    assert Path(tmp_path, "targets.peptides").exists()


def test_cli_keep_decoys(tmp_path, targets_decoys_psms_scored):
    """Test that --keep_decoys works."""
    cmd = [
        "--targets_psms",
        targets_decoys_psms_scored[0],
        "--decoys_psms",
        targets_decoys_psms_scored[1],
        "--dest_dir",
        tmp_path,
        "--keep_decoys",
    ]
    run_aggreate_cli(cmd)
    assert Path(tmp_path, "targets.peptides").exists()
    assert Path(tmp_path, "decoys.peptides").exists()


def test_non_default_fdr(tmp_path, targets_decoys_psms_scored):
    """Test non-defaults"""
    cmd = [
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

    run_aggreate_cli(cmd)
    assert Path(tmp_path, "targets.peptides").exists()
    assert Path(tmp_path, "decoys.peptides").exists()
