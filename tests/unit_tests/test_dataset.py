"""
These tests verify that the dataset classes are functioning properly.
"""
import numpy as np
import pandas as pd
from mokapot import LinearPsmDataset, CrosslinkPsmDataset

from ..fixtures import *


def test_linear_init(psm_df_6):
    """Test that a LinearPsm Dataset initializes correctly"""
    dat = psm_df_6.copy()
    dset = LinearPsmDataset(
        psms=dat,
        target_column="target",
        spectrum_columns="spectrum",
        peptide_column="peptide",
        protein_column="protein",
        group_column="group",
        feature_columns=None,
        copy_data=True,
    )

    pd.testing.assert_frame_equal(dset.data, psm_df_6)

    # Verify that changing the original dataframe does not change the
    # the LinearPsmDataset object.
    dat["target"] = [1, 0] * 3
    pd.testing.assert_frame_equal(dset.data, psm_df_6)

    # Check the public attributes
    features = ["feature_1", "feature_2"]
    metadata = ["target", "spectrum", "group", "peptide", "protein"]

    pd.testing.assert_frame_equal(dset.spectra, psm_df_6.loc[:, ["spectrum"]])
    pd.testing.assert_series_equal(dset.peptides, psm_df_6.loc[:, "peptide"])
    pd.testing.assert_frame_equal(dset.features, psm_df_6.loc[:, features])
    pd.testing.assert_frame_equal(dset.metadata, psm_df_6.loc[:, metadata])
    pd.testing.assert_series_equal(dset.groups, psm_df_6.loc[:, "group"])
    assert dset.columns == psm_df_6.columns.tolist()
    assert np.array_equal(dset.targets, psm_df_6["target"].values)
    assert not dset.has_proteins


def test_crosslink_init(csm_df_6):
    """Test that we can initialize a CrosslinkPsmDataset"""
    dat = csm_df_6.copy()
    dset = CrosslinkPsmDataset(
        csms=dat,
        target_columns=("alpha_target", "beta_target"),
        spectrum_columns="spectrum",
        peptide_columns=("alpha_peptide", "beta_peptide"),
        protein_columns=("alpha_protein", "beta_protein"),
        group_column="group",
        feature_columns=None,
        copy_data=True,
    )

    pd.testing.assert_frame_equal(dset.data, csm_df_6)

    # Verify that changing the original dataframe does not change the
    # the LinearPsmDataset object.
    dat["alpha_target"] = [1, 0] * 3
    pd.testing.assert_frame_equal(dset.data, csm_df_6)

    # Check the public attributes
    features = ["feature_1", "feature_2"]
    metadata = [
        "alpha_target",
        "beta_target",
        "spectrum",
        "group",
        "alpha_peptide",
        "beta_peptide",
        "alpha_protein",
        "beta_protein",
    ]

    pd.testing.assert_frame_equal(dset.spectra, csm_df_6.loc[:, ["spectrum"]])
    pd.testing.assert_frame_equal(
        dset.peptides, csm_df_6.loc[:, ["alpha_peptide", "beta_peptide"]]
    )
    pd.testing.assert_frame_equal(dset.features, csm_df_6.loc[:, features])
    pd.testing.assert_frame_equal(dset.metadata, csm_df_6.loc[:, metadata])
    pd.testing.assert_series_equal(dset.groups, csm_df_6.loc[:, "group"])
    assert dset.columns == csm_df_6.columns.tolist()
    assert np.array_equal(
        dset.targets, csm_df_6[["alpha_target", "beta_target"]].sum(axis=1)
    )
    assert not dset.has_proteins


def test_linear_assign_confidence(psm_df_1000):
    """Test that assign_confidence() methods run"""
    psms, fasta = psm_df_1000
    dset = LinearPsmDataset(
        psms=psms,
        target_column="target",
        spectrum_columns="spectrum",
        peptide_column="peptide",
        feature_columns="score",
        copy_data=True,
    )

    # also try adding proteins:
    assert not dset.has_proteins
    dset.add_proteins(fasta, missed_cleavages=0, min_length=5, max_length=5)
    assert dset.has_proteins

    dset.assign_confidence(eval_fdr=0.05)

    # Make sure it works when lower scores are better:
    data, _ = psm_df_1000
    data["score"] = -data["score"]
    dset = LinearPsmDataset(
        psms=data,
        target_column="target",
        spectrum_columns="spectrum",
        peptide_column="peptide",
        feature_columns="score",
        copy_data=True,
    )
    dset.assign_confidence(eval_fdr=0.05)

    # Verify that the groups yields 2 results:
    dset = LinearPsmDataset(
        psms=psms,
        target_column="target",
        spectrum_columns="spectrum",
        peptide_column="peptide",
        group_column="group",
        feature_columns="score",
        copy_data=True,
    )
    res = dset.assign_confidence(eval_fdr=0.05)
    assert len(res) == 2


def test_crosslink_assign_confidence(csm_df_1000):
    """Test that assign_confidence() methods run"""
    csms, fasta = csm_df_1000
    dset = CrosslinkPsmDataset(
        csms=csms,
        target_columns=("alpha_target", "beta_target"),
        spectrum_columns=("spectrum"),
        peptide_columns=("alpha_peptide", "beta_peptide"),
        feature_columns="score",
        copy_data=True,
    )
    assert not dset.has_proteins
    dset.add_proteins(fasta, missed_cleavages=0, min_length=5, max_length=5)
    assert dset.has_proteins
    dset.assign_confidence()

    csms, _ = csm_df_1000
    csms["score"] = -csms["score"]
    dset = CrosslinkPsmDataset(
        csms=csms,
        target_columns=("alpha_target", "beta_target"),
        spectrum_columns=("spectrum"),
        peptide_columns=("alpha_peptide", "beta_peptide"),
        feature_columns="score",
        copy_data=True,
    )
    dset.assign_confidence(eval_fdr=0.05)

    # Verify that the groups yields 2 results:
    dset = CrosslinkPsmDataset(
        csms=csms,
        target_columns=("alpha_target", "beta_target"),
        spectrum_columns="spectrum",
        peptide_columns=("alpha_peptide", "beta_peptide"),
        group_column="group",
        feature_columns="score",
        copy_data=True,
    )
    res = dset.assign_confidence(eval_fdr=0.05)
    assert len(res) == 2


def test_linear_update_labels(psm_df_6):
    """Test that the _update_labels() methods are working"""
    dset = LinearPsmDataset(
        psm_df_6,
        target_column="target",
        spectrum_columns="spectrum",
        peptide_column="peptide",
        protein_column="protein",
        group_column="group",
        feature_columns=None,
        copy_data=True,
    )

    scores = np.array([6, 5, 3, 3, 2, 1])
    real_labs = np.array([1, 1, 0, -1, -1, -1])
    new_labs = dset._update_labels(scores, eval_fdr=0.5)
    assert np.array_equal(real_labs, new_labs)


def test_crosslink_update_labels(csm_df_6):
    """Test that the _update_labels() methods ware working"""
    dset = CrosslinkPsmDataset(
        csm_df_6,
        target_columns=("alpha_target", "beta_target"),
        spectrum_columns="spectrum",
        peptide_columns=("alpha_peptide", "beta_peptide"),
        protein_columns=("alpha_protein", "beta_protein"),
        group_column="group",
        feature_columns=None,
        copy_data=True,
    )
    scores = np.array([6, 5, 4, 3, 2, 1])
    real_labs = np.array([1, 1, -1, -1, -1, -1])
    new_labs = dset._update_labels(scores, eval_fdr=0.5)
    assert np.array_equal(real_labs, new_labs)

    real_labs = np.array([0, 0, -1, -1, -1, -1])
    new_labs = dset._update_labels(-scores, eval_fdr=0.5)
    assert np.array_equal(real_labs, new_labs)
