"""Test that Confidence classes are working correctly"""
from pathlib import Path
import numpy as np
import pandas as pd
import copy
from mokapot import OnDiskPsmDataset, assign_confidence


def test_one_group(psm_df_1000, tmp_path):
    """Test that one group is equivalent to no group."""

    pin_file, _, _ = psm_df_1000
    columns = list(pd.read_csv(pin_file, sep="\t").columns)
    df_spectra = pd.read_csv(
        pin_file, sep="\t", usecols=["scannr", "expmass", "target"]
    )
    psms_disk = OnDiskPsmDataset(
        filename=pin_file,
        target_column="target",
        spectrum_columns=["scannr", "expmass"],
        peptide_column="peptide",
        feature_columns=["score"],
        filename_column="filename",
        scan_column="scannr",
        calcmass_column="calcmass",
        expmass_column="expmass",
        rt_column="ret_time",
        charge_column="charge",
        columns=columns,
        protein_column=None,
        group_column="group",
        metadata_columns=[
            "specid",
            "scannr",
            "expmass",
            "peptide",
            "proteins",
            "target",
        ],
        specId_column="spectrum",
        spectra_dataframe=df_spectra,
    )
    np.random.seed(42)
    assign_confidence(
        [copy.copy(psms_disk)],
        prefixes=[None],
        descs=[True],
        dest_dir=tmp_path,
        max_workers=4,
    )
    df_results_group = pd.read_csv(tmp_path / "0.targets.peptides", sep="\t")

    np.random.seed(42)
    psms_disk.group_column = None
    assign_confidence(
        [psms_disk], prefixes=[None], descs=[True], dest_dir=tmp_path, max_workers=4,
    )
    df_results_no_group = pd.read_csv(tmp_path / "targets.peptides", sep="\t")

    pd.testing.assert_frame_equal(df_results_group, df_results_no_group)


def test_multi_groups(psm_df_100, tmp_path):
    """Test that group results are saved"""
    pin_file = psm_df_100
    columns = list(pd.read_csv(pin_file, sep="\t").columns)
    df_spectra = pd.read_csv(
        pin_file, sep="\t", usecols=["scannr", "expmass", "target"]
    )
    psms_disk = OnDiskPsmDataset(
        filename=pin_file,
        target_column="target",
        spectrum_columns=["scannr", "expmass"],
        peptide_column="peptide",
        feature_columns=["score"],
        filename_column="filename",
        scan_column="scannr",
        calcmass_column="calcmass",
        expmass_column="expmass",
        rt_column="ret_time",
        charge_column="charge",
        columns=columns,
        protein_column=None,
        group_column="group",
        metadata_columns=[
            "specid",
            "scannr",
            "expmass",
            "peptide",
            "proteins",
            "target",
        ],
        specId_column="spectrum",
        spectra_dataframe=df_spectra,
    )
    assign_confidence(
        [psms_disk], prefixes=[None], descs=[True], dest_dir=tmp_path, max_workers=4,
    )
    assert Path(tmp_path, f"{0}.targets.psms").exists()
    assert Path(tmp_path, f"{1}.targets.psms").exists()
    assert Path(tmp_path, f"{0}.targets.peptides").exists()
    assert Path(tmp_path, f"{1}.targets.peptides").exists()
