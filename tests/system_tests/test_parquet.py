"""
These tests verify that the CLI works as expected.

At least for now, they do not check the correctness of the
output, just that the expect outputs are created.
"""

from pathlib import Path

import pandas as pd

from mokapot.column_defs import STANDARD_COLUMN_NAME_MAP

from ..helpers.cli import run_mokapot_cli
from ..helpers.utils import file_exist


def test_parquet_output(tmp_path):
    """Test that parquet input/output works."""
    params = [Path("data") / "10k_psms_test.parquet", "--dest_dir", tmp_path]
    orig_cols = list(pd.read_parquet(params[0]).columns)
    assert orig_cols == [
        "SpecId",
        "Label",
        "ScanNr",
        "ExpMass",
        "Mass",
        "MS8_feature_5",
        "missedCleavages",
        "MS8_feature_7",
        "MS8_feature_13",
        "MS8_feature_20",
        "MS8_feature_21",
        "MS8_feature_22",
        "MS8_feature_24",
        "MS8_feature_29",
        "MS8_feature_30",
        "MS8_feature_32",
        "Peptide",
        "Proteins",
    ]

    run_mokapot_cli(params)
    assert file_exist(tmp_path, "targets.psms.parquet")
    assert file_exist(tmp_path, "targets.peptides.parquet")

    targets_psms_df = pd.read_parquet(Path(tmp_path, "targets.psms.parquet"))
    assert len(targets_psms_df.index) >= 5000

    assert targets_psms_df.iloc[0, 0] == 6991
    # Here 'Proteins' gets propagated from the input file name
    assert targets_psms_df["Proteins"].iloc[0] == "_.dummy._"

    expected_cols = [
        "SpecId",
        "ScanNr",
        "ExpMass",
        "Peptide",
        # Mokapot-prefixed cols are added, rest are
        # propagated from the input file.
        STANDARD_COLUMN_NAME_MAP["score"],
        STANDARD_COLUMN_NAME_MAP["q-value"],
        STANDARD_COLUMN_NAME_MAP["posterior_error_prob"],
        "Proteins",
    ]
    assert list(targets_psms_df.columns) == expected_cols
