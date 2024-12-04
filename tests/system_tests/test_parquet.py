"""
These tests verify that the CLI works as expected.

At least for now, they do not check the correctness of the
output, just that the expect outputs are created.
"""

from pathlib import Path

import pandas as pd

from ..helpers.cli import run_mokapot_cli
from ..helpers.utils import file_exist


def test_parquet_output(tmp_path):
    """Test that parquet input/output works."""
    params = [Path("data") / "10k_psms_test.parquet", "--dest_dir", tmp_path]
    run_mokapot_cli(params)
    assert file_exist(tmp_path, "targets.psms.parquet")
    assert file_exist(tmp_path, "targets.peptides.parquet")

    targets_psms_df = pd.read_parquet(Path(tmp_path, "targets.psms.parquet"))
    assert targets_psms_df.columns.values.tolist() == [
        "PSMId",
        "peptide",
        "score",
        "q-value",
        "posterior_error_prob",
        "proteinIds",
    ]
    assert len(targets_psms_df.index) >= 5000

    assert targets_psms_df.iloc[0, 0] == 6991
    assert targets_psms_df.iloc[0, 5] == "_.dummy._"
