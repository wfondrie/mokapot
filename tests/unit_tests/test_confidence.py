"""Test that Confidence classes are working correctly"""
import pytest
import numpy as np
import pandas as pd
from mokapot import LinearPsmDataset


def test_one_group(psm_df_1000):
    """Test that one group is equivalent to no group."""
    psm_data, _ = psm_df_1000
    psm_data["group"] = 0

    psms = LinearPsmDataset(
        psms=psm_data,
        target_column="target",
        spectrum_columns="spectrum",
        peptide_column="peptide",
        feature_columns="score",
        filename_column="filename",
        scan_column="spectrum",
        calcmass_column="calcmass",
        expmass_column="expmass",
        rt_column="ret_time",
        charge_column="charge",
        group_column="group",
        copy_data=True,
    )

    np.random.seed(42)
    grouped = psms.assign_confidence()
    scores1 = grouped.group_confidence_estimates[0].psms["mokapot score"]

    np.random.seed(42)
    psms._group_column = None
    ungrouped = psms.assign_confidence()
    scores2 = ungrouped.psms["mokapot score"]

    pd.testing.assert_series_equal(scores1, scores2)
