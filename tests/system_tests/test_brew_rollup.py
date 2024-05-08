"""
These tests verify that the CLI works as expected.

At least for now, they do not check the correctness of the
output, just that the expect outputs are created.
"""
import logging
from pathlib import Path
from typing import List, Any

from mokapot.brew_rollup import compute_rollup_levels, DEFAULT_PARENT_LEVELS
from ..helpers.cli import run_mokapot_cli, _run_cli

import pytest


def run_brew_rollup(params: List[Any], run_in_subprocess=None,
                    capture_output=False):
    from mokapot.brew_rollup import main
    return _run_cli("mokapot.brew_rollup", main, params,
                    run_in_subprocess, capture_output)


def test_rollup_10000(tmp_path):
    """Test that basic cli works."""
    path = tmp_path
    path = Path("scratch", "testing")

    retrain = False
    recompute = False

    common_params = [
        "--dest_dir",
        path,
        "--max_workers", 8,
        "--test_fdr", 0.10,
        "--train_fdr", 0.9,
        "--verbosity", 2,
        "--subset_max_train", 4000,
        "--max_iter", 10,
        "--ensemble", "--keep_decoys",
    ]

    if retrain or not Path.exists(path / "mokapot.model_fold-1.pkl"):
        params = [Path("data", "percolator-noSplit-extended-10000.tab"),
                  *common_params,
                  "--save_models",
                  ]
        run_mokapot_cli(params)

    if recompute or not Path.exists(path / "a.targets.precursors"):
        params = [Path("data", "percolator-noSplit-extended-1000.tab"),
                  *common_params,
                  "--load_models", *path.glob("mokapot.model_fold-*.pkl"),
                  "--file_root", "a"
                  ]
        run_mokapot_cli(params)

    if recompute or not Path.exists(path / "b.targets.precursors"):
        params = [Path("data", "percolator-noSplit-extended-1000b.tab"),
                  *common_params,
                  "--load_models", *path.glob("mokapot.model_fold-*.pkl"),
                  "--file_root", "b"
                  ]
        run_mokapot_cli(params)

    if recompute or not Path.exists(path / "c.targets.precursors"):
        params = [Path("data", "percolator-noSplit-extended-1000c.tab"),
                  *common_params,
                  "--load_models", *path.glob("mokapot.model_fold-*.pkl"),
                  "--file_root", "c"
                  ]
        run_mokapot_cli(params)

    rollup_params = [
        "--level", "precursor",
        "--dest_dir", path,
        "--max_workers", 8,
        "--verbosity", 3,
        "--keep_decoys",
    ]
    output = run_brew_rollup(rollup_params, capture_output=True)
    print(output and output["stderr"])


def test_compute_rollup_levels():
    assert (sorted(compute_rollup_levels('psm')) ==
            ['modified_peptide', 'peptide', 'peptide_group', 'precursor', 'psm'])
    assert (sorted(compute_rollup_levels('precursor')) ==
            ['modified_peptide', 'peptide', 'peptide_group', 'precursor'])
    assert (sorted(compute_rollup_levels('modified_peptide')) ==
            ['modified_peptide', 'peptide'])
    assert (sorted(compute_rollup_levels('peptide')) ==
            ['peptide'])
    assert (sorted(compute_rollup_levels('peptide_group')) ==
            ['peptide_group'])
