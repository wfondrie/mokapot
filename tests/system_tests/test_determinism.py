"""
These tests verify that the aggregatePsmsToPeptides executable works as expected.
"""  # noqa: E501

from pathlib import Path

from ..helpers.cli import run_mokapot_cli

import pandas as pd
import pytest

# Warnings are errors for these tests
pytestmark = pytest.mark.filterwarnings("error")


def test_determinism_same_file(tmp_path, psm_files_4000):
    """Test that two identical mokapot runs produce same results."""

    params = [
        "--dest_dir",
        tmp_path,
        "--subset_max_train",
        "2500",
        "--keep_decoys",
        "--max_workers",
        "8",
        "--ensemble",
        "--max_iter",
        "2",
    ]
    run_mokapot_cli(params + [psm_files_4000[0], "--file_root", "run1"])
    run_mokapot_cli(params + [psm_files_4000[0], "--file_root", "run2"])

    assert Path(tmp_path, "run1.targets.peptides").exists()
    assert Path(tmp_path, "run1.decoys.peptides").exists()
    assert Path(tmp_path, "run1.targets.psms").exists()
    assert Path(tmp_path, "run1.decoys.psms").exists()

    assert Path(tmp_path, "run2.targets.peptides").exists()
    assert Path(tmp_path, "run2.decoys.peptides").exists()
    assert Path(tmp_path, "run2.targets.psms").exists()
    assert Path(tmp_path, "run2.decoys.psms").exists()

    df_run1_t_psms = pd.read_csv(tmp_path / "run1.targets.psms", sep="\t")
    df_run2_t_psms = pd.read_csv(tmp_path / "run2.targets.psms", sep="\t")
    pd.testing.assert_frame_equal(df_run1_t_psms, df_run2_t_psms)

    df_run1_t_peptides = pd.read_csv(
        tmp_path / "run1.targets.peptides", sep="\t"
    )
    df_run2_t_peptides = pd.read_csv(
        tmp_path / "run2.targets.peptides", sep="\t"
    )
    pd.testing.assert_frame_equal(df_run1_t_peptides, df_run2_t_peptides)

    df_run1_d_psms = pd.read_csv(tmp_path / "run1.decoys.psms", sep="\t")
    df_run2_d_psms = pd.read_csv(tmp_path / "run2.decoys.psms", sep="\t")
    pd.testing.assert_frame_equal(df_run1_d_psms, df_run2_d_psms)

    df_run1_d_peptides = pd.read_csv(
        tmp_path / "run1.decoys.peptides", sep="\t"
    )
    df_run2_d_peptides = pd.read_csv(
        tmp_path / "run2.decoys.peptides", sep="\t"
    )
    pd.testing.assert_frame_equal(df_run1_d_peptides, df_run2_d_peptides)


def test_determinism_different_psmid(tmp_path, psm_files_4000):
    """
    Test that two mokapot runs with 2 files where only the SpecIds are changed
    produce same results.
    """

    cmd = [
        "--dest_dir",
        tmp_path,
        "--subset_max_train",
        "2500",
        "--keep_decoys",
        "--max_workers",
        "8",
        "--ensemble",
        "--max_iter",
        "2",
    ]

    run_mokapot_cli(cmd + [psm_files_4000[0], "--file_root", "run1"])
    run_mokapot_cli(cmd + [psm_files_4000[1], "--file_root", "run2"])

    assert Path(tmp_path, "run1.targets.peptides").exists()
    assert Path(tmp_path, "run1.decoys.peptides").exists()
    assert Path(tmp_path, "run1.targets.psms").exists()
    assert Path(tmp_path, "run1.decoys.psms").exists()

    assert Path(tmp_path, "run2.targets.peptides").exists()
    assert Path(tmp_path, "run2.decoys.peptides").exists()
    assert Path(tmp_path, "run2.targets.psms").exists()
    assert Path(tmp_path, "run2.decoys.psms").exists()

    df_run1_t_psms = pd.read_csv(
        tmp_path / "run1.targets.psms", sep="\t"
    ).drop("PSMId", axis=1)
    df_run2_t_psms = pd.read_csv(
        tmp_path / "run2.targets.psms", sep="\t"
    ).drop("PSMId", axis=1)
    pd.testing.assert_frame_equal(df_run1_t_psms, df_run2_t_psms)

    df_run1_t_peptides = pd.read_csv(
        tmp_path / "run1.targets.peptides", sep="\t"
    ).drop("PSMId", axis=1)
    df_run2_t_peptides = pd.read_csv(
        tmp_path / "run2.targets.peptides", sep="\t"
    ).drop("PSMId", axis=1)
    pd.testing.assert_frame_equal(df_run1_t_peptides, df_run2_t_peptides)

    df_run1_d_psms = pd.read_csv(tmp_path / "run1.decoys.psms", sep="\t").drop(
        "PSMId", axis=1
    )
    df_run2_d_psms = pd.read_csv(tmp_path / "run2.decoys.psms", sep="\t").drop(
        "PSMId", axis=1
    )
    pd.testing.assert_frame_equal(df_run1_d_psms, df_run2_d_psms)

    df_run1_d_peptides = pd.read_csv(
        tmp_path / "run1.decoys.peptides", sep="\t"
    ).drop("PSMId", axis=1)
    df_run2_d_peptides = pd.read_csv(
        tmp_path / "run2.decoys.peptides", sep="\t"
    ).drop("PSMId", axis=1)
    pd.testing.assert_frame_equal(df_run1_d_peptides, df_run2_d_peptides)
