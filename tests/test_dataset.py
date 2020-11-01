"""
These tests verify that the dataset classes are functioning
as intended.
"""
import os
import mokapot
import numpy as np
import pandas as pd

EX_FILE = os.path.join("data", "scope2_FP97AA.pin")
DAT1 = pd.DataFrame(
    {
        "target": [True, True, True, False, False, False],
        "spectrum": [1, 2, 3, 4, 5, 1],
        "peptide": ["a", "b", "a", "c", "d", "e"],
        "protein": ["A", "B"] * 3,
        "feature_1": [4, 3, 2, 2, 1, 0],
        "feature_2": [2, 3, 4, 1, 2, 3],
    }
)


def test_linear_init():
    """Test that a LinearPsmDataset is initialized correctly"""
    dat = DAT1.copy()
    dset = mokapot.LinearPsmDataset(
        dat,
        target_column="target",
        spectrum_columns="spectrum",
        peptide_column="peptide",
        protein_column="protein",
        feature_columns=None,
        copy_data=True,
    )

    pd.testing.assert_frame_equal(dset.data, DAT1)

    # Verify that changing the original dataframe does not change
    # change the dataset object.
    dat["target"] = [1, 0] * 3
    pd.testing.assert_frame_equal(dset.data, DAT1)

    # check the attributes
    features = ["feature_1", "feature_2"]
    metadata = ["target", "spectrum", "peptide", "protein"]

    pd.testing.assert_frame_equal(dset.spectra, DAT1.loc[:, ["spectrum"]])
    pd.testing.assert_series_equal(dset.peptides, DAT1.loc[:, "peptide"])
    pd.testing.assert_frame_equal(dset.features, DAT1.loc[:, features])
    pd.testing.assert_frame_equal(dset.metadata, DAT1.loc[:, metadata])
    assert dset.columns == DAT1.columns.tolist()
    assert np.array_equal(dset.targets, DAT1["target"].values)


def test_assign_confidence():
    """Test that assign_confidence() methods run"""
    dset = mokapot.read_pin(EX_FILE)
    dset.assign_confidence()


def test_update_labels():
    """Test that the _update_labels() methods are working"""
    dset = mokapot.LinearPsmDataset(
        DAT1,
        target_column="target",
        spectrum_columns="spectrum",
        peptide_column="peptide",
        protein_column="protein",
        group_column=None,
        feature_columns=None,
        copy_data=True,
    )

    scores = np.array([6, 5, 3, 3, 2, 1])
    real_labs = np.array([1, 1, 0, -1, -1, -1])
    new_labs = dset._update_labels(scores, eval_fdr=0.5)
    assert np.array_equal(real_labs, new_labs)
