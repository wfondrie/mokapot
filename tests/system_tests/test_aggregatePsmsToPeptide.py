"""
These tests verify that the aggregatePsmsToPeptides executable works as expected
"""

from pathlib import Path

import pytest

from helpers.cli import run_aggregate_cli

# Warnings are errors for these tests
pytestmark = pytest.mark.filterwarnings("error")


def test_basic_cli(tmp_path, targets_decoys_psms_scored):
    """Test that basic cli works."""
    params = [
        "--targets_psms",
        targets_decoys_psms_scored[0],
        "--decoys_psms",
        targets_decoys_psms_scored[1],
        "--dest_dir",
        tmp_path,
    ]
    run_aggregate_cli(params)
    assert Path(tmp_path, "targets.peptides").exists()


def test_cli_keep_decoys(tmp_path, targets_decoys_psms_scored):
    """Test that --keep_decoys works."""
    params = [
        "--targets_psms",
        targets_decoys_psms_scored[0],
        "--decoys_psms",
        targets_decoys_psms_scored[1],
        "--dest_dir",
        tmp_path,
        "--keep_decoys",
    ]
    run_aggregate_cli(params)
    assert Path(tmp_path, "targets.peptides").exists()
    assert Path(tmp_path, "decoys.peptides").exists()


def test_non_default_fdr(tmp_path, targets_decoys_psms_scored):
    """Test non-defaults"""
    params = [
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

    run_aggregate_cli(params)
    assert Path(tmp_path, "targets.peptides").exists()
    assert Path(tmp_path, "decoys.peptides").exists()
