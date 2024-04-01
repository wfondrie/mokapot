"""Test that Confidence classes are working correctly"""

from pathlib import Path
import numpy as np
import pandas as pd
import copy
from mokapot import OnDiskPsmDataset, assign_confidence
from mokapot.confidence import get_unique_peptides_from_psms


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
        protein_column="proteins",
        group_column="group",
        metadata_columns=[
            "specid",
            "scannr",
            "expmass",
            "peptide",
            "proteins",
            "target",
        ],
        level_columns=["peptide"],
        specId_column="specid",
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
        [psms_disk],
        prefixes=[None],
        descs=[True],
        dest_dir=tmp_path,
        max_workers=4,
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
        protein_column="proteins",
        group_column="group",
        metadata_columns=[
            "specid",
            "scannr",
            "expmass",
            "peptide",
            "proteins",
            "target",
        ],
        level_columns=["peptide"],
        specId_column="specid",
        spectra_dataframe=df_spectra,
    )
    assign_confidence(
        [psms_disk],
        prefixes=[None],
        descs=[True],
        dest_dir=tmp_path,
        max_workers=4,
    )
    assert Path(tmp_path, f"{0}.targets.psms").exists()
    assert Path(tmp_path, f"{1}.targets.psms").exists()
    assert Path(tmp_path, f"{0}.targets.peptides").exists()
    assert Path(tmp_path, f"{1}.targets.peptides").exists()


def test_get_unique_psms_and_peptides(peptide_csv_file, psms_iterator):
    psms_iterator = psms_iterator
    get_unique_peptides_from_psms(
        iterable=psms_iterator,
        peptide_col_index=2,
        out_peptides=peptide_csv_file,
        sep="\t",
    )

    expected_output = pd.DataFrame(
        [
            [1, 1, "HLAQLLR", -5.75, "_.dummy._"],
            [3, 0, "NVPTSLLK", -5.83, "_.dummy._"],
            [4, 1, "QILVQLR", -5.92, "_.dummy._"],
            [7, 1, "SRTSVIPGPK", -6.12, "_.dummy._"],
        ],
        columns=["PSMId", "Label", "Peptide", "score", "proteinIds"],
    )

    output = pd.read_csv(peptide_csv_file, sep="\t")
    pd.testing.assert_frame_equal(expected_output, output)
