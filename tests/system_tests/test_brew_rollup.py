"""
These tests verify that the CLI works as expected.

At least for now, they do not check the correctness of the
output, just that the expect outputs are created.
"""

from pathlib import Path
from typing import Any, List

import pytest
from filelock import FileLock
from numpy.testing import assert_allclose
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
    dest_dir = tmp_path_factory.mktemp("csv")
    pq_dest_dir = tmp_path_factory.mktemp("parquet")

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

    def run_brew_rollup2(subdir: str, extra_params: list = []):
        run_brew_rollup(
            rollup_params + ["--dest_dir", rollup_dest_dir / subdir] + extra_params,
            capture_output=False,
        )
        assert rollup_dest_dir / subdir / f"rollup.targets.peptides{suffix}"
        file = rollup_dest_dir / subdir / f"rollup.targets.peptides{suffix}"
        df = TabularDataReader.from_path(file).read()
        return df

    df0 = run_brew_rollup2("rollup0")
    qval_column = "q-value"

    # Note: this maximum difference is relatively large here (about 0.048),
    # because, the scores are very concentrated around several values (nearly
    # discrete) and the streaming (histogram) method is smoothing this out, what
    # the pure counting method does not do.
    # Todo: maybe it would be worthwile to have a much finer histogram with
    #   much more bins for q-value calculation then for peps calculation
    df1 = run_brew_rollup2("rollup1", ["--stream_confidence"])
    assert_series_equal(df0[qval_column], df1[qval_column], atol=0.05)
    assert estimate_abs_int(df0.score, df1[qval_column] - df0[qval_column]) < 0.05
    assert (
        estimate_abs_int(df0.score, df1.posterior_error_prob - df0.posterior_error_prob)
        < 0.03
    )

    df2 = run_brew_rollup2("rollup2", [("--qvalue_algorithm", "storey")])
    assert_series_equal(df0[qval_column], df2[qval_column], atol=0.001)
    assert_allclose(df0[qval_column], df2[qval_column], atol=0.001)
    assert estimate_abs_int(df0.score, df2[qval_column] - df0[qval_column]) < 0.001

    df3 = run_brew_rollup2(
        "rollup2",
        [("--qvalue_algorithm", "storey"), ("--pi0_algorithm", "storey_fixed")],
    )
    assert not all(df2[qval_column] == df3[qval_column])
    assert_allclose(
        df2[qval_column], df3[qval_column], atol=0.3
    )  # todo: check is this real?


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
