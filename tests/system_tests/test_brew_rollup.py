"""
These tests verify that the CLI works as expected.

At least for now, they do not check the correctness of the
output, just that the expect outputs are created.
"""

import shutil
from pathlib import Path
from typing import Any, List

import pytest
from filelock import FileLock
from pandas.testing import assert_series_equal

from mokapot.rollup import compute_rollup_levels
from mokapot.tabular_data import CSVFileReader, ParquetFileWriter, TabularDataReader
from ..helpers.cli import _run_cli, run_mokapot_cli
from ..helpers.math import estimate_abs_int


def run_brew_rollup(params: List[Any], run_in_subprocess=None, capture_output=False):
    from mokapot.brew_rollup import main

    return _run_cli(
        "mokapot.brew_rollup", main, params, run_in_subprocess, capture_output
    )


@pytest.fixture(scope="session")
def rollup_src_dirs(tmp_path_factory):
    dest_dir = Path("scratch", "testing_rollup")
    dest_dir.mkdir(parents=True, exist_ok=True)
    pq_dest_dir = dest_dir / "parquet"
    pq_dest_dir.mkdir(parents=True, exist_ok=True)

    retrain = False
    recompute = retrain or False

    common_params = [
        ("--dest_dir", dest_dir),
        ("--max_workers", 8),
        ("--test_fdr", 0.10),
        ("--train_fdr", 0.05),
        ("--verbosity", 2),
        ("--subset_max_train", 4000),
        ("--max_iter", 10),
        "--ensemble",
        "--keep_decoys",
    ]

    # In case we run the tests parallel with xdist, we may run into race
    with FileLock(dest_dir / "rollup.lock"):
        # Train mokapot on larger input file
        if retrain or not Path.exists(dest_dir / "mokapot.model_fold-1.pkl"):
            params = [
                Path("data", "percolator-noSplit-extended-10000.tab"),
                *common_params,
                "--save_models",
            ]
            run_mokapot_cli(params)

        parts = {
            "part-a": "percolator-noSplit-extended-1000.tab",
            "part-b": "percolator-noSplit-extended-1000b.tab",
            "part-c": "percolator-noSplit-extended-1000c.tab",
        }

        for root, input_file in parts.items():
            # Run mokapot for the smaller data files
            if recompute or not Path.exists(
                dest_dir / f"{root}.targets.precursors.csv"
            ):
                params = [
                    Path("data", input_file),
                    *common_params,
                    ("--load_models", *dest_dir.glob("mokapot.model*.pkl")),
                    ("--file_root", root),
                ]
                run_mokapot_cli(params)

            # Convert csv output to parquet
            for file in Path(dest_dir).glob(f"{root}.*.csv"):
                outfile = pq_dest_dir / file.with_suffix(".parquet").name
                if outfile.exists():
                    continue
                reader = CSVFileReader(file)
                data = reader.read()
                writer = ParquetFileWriter(
                    outfile,
                    reader.get_column_names(),
                    reader.get_column_types(),
                )
                writer.write(data)

    yield dest_dir, pq_dest_dir

    # Cleanup files here
    # Note: If you want to keep the files, create a file or directory name
    # "dont_remove_me" in the dest_dir e.g. by the command
    # mkdir -p scratch/testing/dont_remove_me
    if not Path.exists(dest_dir / "dont_remove_me"):
        shutil.rmtree(dest_dir)


@pytest.mark.parametrize(
    "suffix",
    [".csv", ".parquet"],
)
def test_rollup_10000(rollup_src_dirs, suffix, tmp_path):
    """Test that basic cli works."""
    # rollup_dest_dir = tmp_path / suffix
    rollup_src_dir, rollup_src_dir_parquet = rollup_src_dirs

    rollup_dest_dir = tmp_path / suffix
    rollup_dest_dir.mkdir(parents=True, exist_ok=True)

    if suffix == ".parquet":
        src_dir = rollup_src_dir_parquet
    else:
        src_dir = rollup_src_dir

    rollup_params = [
        ("--level", "precursor"),
        ("--src_dir", src_dir),
        ("--qvalue_algorithm", "from_counts"),
        ("--verbosity", 2),
    ]
    run_brew_rollup(
        rollup_params + ["--dest_dir", rollup_dest_dir / "rollup0"],
        capture_output=False,
    )
    run_brew_rollup(
        rollup_params
        + ["--dest_dir", rollup_dest_dir / "rollup1", "--stream_confidence"],
        capture_output=False,
    )

    assert rollup_dest_dir / "rollup0" / f"rollup.targets.peptides{suffix}"

    file0 = rollup_dest_dir / "rollup0" / f"rollup.targets.peptides{suffix}"
    file1 = rollup_dest_dir / "rollup1" / f"rollup.targets.peptides{suffix}"
    df0 = TabularDataReader.from_path(file0).read()
    df1 = TabularDataReader.from_path(file1).read()

    qval_column = "q-value"
    assert_series_equal(df0[qval_column], df1[qval_column], atol=0.02)
    assert estimate_abs_int(df0.score, df1[qval_column] - df0[qval_column]) < 0.002
    assert (
        estimate_abs_int(df0.score, df1.posterior_error_prob - df0.posterior_error_prob)
        < 0.03
    )


def test_compute_rollup_levels():
    assert sorted(compute_rollup_levels("psm")) == [
        "modified_peptide",
        "peptide",
        "peptide_group",
        "precursor",
        "psm",
    ]
    assert sorted(compute_rollup_levels("precursor")) == [
        "modified_peptide",
        "peptide",
        "peptide_group",
        "precursor",
    ]
    assert sorted(compute_rollup_levels("modified_peptide")) == [
        "modified_peptide",
        "peptide",
    ]
    assert sorted(compute_rollup_levels("peptide")) == ["peptide"]
    assert sorted(compute_rollup_levels("peptide_group")) == ["peptide_group"]
