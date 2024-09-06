"""
These tests verify that the CLI works as expected.

At least for now, they do not check the correctness of the
output, just that the expect outputs are created.
"""

from pathlib import Path
import pandas as pd

from ..helpers.cli import run_mokapot_cli

import pytest


@pytest.fixture
def percolator_extended_file_small():
    """Get the extended percolator tab file restricted to 1000 rows"""
    return Path("data", "percolator-noSplit-extended-1000.tab")


@pytest.fixture
def percolator_extended_file_big():
    """Get the extended percolator tab file restricted to 10000 rows"""
    return Path("data", "percolator-noSplit-extended-10000.tab")


# @pytest.fixture
# def percolator_extended_file_huge():
#     """Get the full extended percolator tab file"""
#     return Path("scratch", "percolator-noSplit-extended.tab")


def test_rollup_10000(
    tmp_path, percolator_extended_file_small, percolator_extended_file_big
):
    """Test that basic cli works."""
    path = tmp_path

    import shutil

    shutil.rmtree(path, ignore_errors=True)

    retrain = False

    common_params = [
        "--dest_dir",
        path,
        "--max_workers",
        8,
        "--test_fdr",
        0.10,
        "--train_fdr",
        0.9,
        "--verbosity",
        2,
        "--subset_max_train",
        4000,
        "--max_iter",
        10,
        "--ensemble",
        "--keep_decoys",
    ]

    use_proteins = False
    if use_proteins:
        fasta = Path("data", "human_sp_td.fasta")
        common_params += ["--proteins", fasta]

    # common_params += ["--skip_rollup"]

    if retrain or not Path.exists(path / "mokapot.model_fold-1.pkl"):
        # params = [percolator_extended_file_big,
        params = [
            percolator_extended_file_small,
            *common_params,
            "--save_models",
        ]
    else:
        params = [
            percolator_extended_file_small,
            *common_params,
            "--load_models",
            *path.glob("mokapot.model_fold-*.pkl"),
        ]

    run_mokapot_cli(params)
    assert Path(path, "targets.psms").exists()
    assert Path(path, "targets.peptides").exists()
    assert Path(path, "targets.precursors").exists()
    assert Path(path, "targets.modifiedpeptides").exists()
    assert Path(path, "targets.peptidegroups").exists()


def test_extra_cols(tmp_path):
    """Test that two identical mokapot runs produce same results."""

    extended_file = Path("data", "percolator-noSplit-extended-10000.tab")
    non_extended_file = Path(
        "data", "percolator-noSplit-non-extended-10000.tab"
    )

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
    run_mokapot_cli([extended_file] + params + ["--file_root", "run1"])
    run_mokapot_cli([non_extended_file] + params + ["--file_root", "run2"])

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
    pd.testing.assert_frame_equal(
        df_run1_t_psms[df_run2_t_psms.columns], df_run2_t_psms
    )

    df_run1_t_peptides = pd.read_csv(
        tmp_path / "run1.targets.peptides", sep="\t"
    )
    df_run2_t_peptides = pd.read_csv(
        tmp_path / "run2.targets.peptides", sep="\t"
    )
    pd.testing.assert_frame_equal(
        df_run1_t_peptides[df_run2_t_peptides.columns], df_run2_t_peptides
    )
