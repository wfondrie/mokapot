"""Test that FlashLFQ export is working"""

import pytest
import mokapot
import pandas as pd
from mokapot.writers.flashlfq import _format_flashlfq
from mokapot import LinearPsmDataset
from mokapot import read_pin

EXPECTED_COLS = {
    "File Name",
    "Base Sequence",
    "Full Sequence",
    "Peptide Monoisotopic Mass",
    "Scan Retention Time",
    "Precursor Charge",
    "Protein Accession",
}


def is_flashlfq_df(df):
    """Check if the df is a valid FlashLFQ input.

    https://github.com/smith-chem-wisc/FlashLFQ/wiki/Identification-Input-Formats

    - File Name - With or without file extension (e.g. MyFile or MyFile.mzML)
    - Base Sequence - Should only contain an amino acid sequence
      (e.g., PEPTIDE and not PEPT[Phosphorylation]IDE
    - Full Sequence - Modified sequence. Can contain any characters
      (e.g., PEPT[Phosphorylation]IDE is fine), but must be consistent between
      the same peptidoform to get accurate results
    - Peptide Monoisotopic Mass - Theoretical monoisotopic mass,
      including modification mass
    - Scan Retention Time - MS/MS identification scan retention time in minutes
    - Precursor Charge - Charge of the ion selected for MS/MS resulting in the
      identification. Use the number only (e.g., "3" and not "+3")
    - Protein Accession - Protein accession(s) for the peptide.
      It is important to list all of the parent protein options
      if you want the "shared peptides" to be accurate.
      Use the semicolon (;) to delimit different proteins.
    """
    # File Name	Scan Retention Time	Precursor Charge	Base Sequence
    # Full Sequence	Peptide Monoisotopic Mass	Protein Accession
    EXPECTED_COLS = {
        "File Name": str,
        "Base Sequence": str,
        "Full Sequence": str,
        "Peptide Monoisotopic Mass": float,
        "Scan Retention Time": float,
        "Precursor Charge": int,
        "Protein Accession": str,
    }
    for col, coltype in EXPECTED_COLS.items():
        assert col in df.columns, f"Column {col} not found in input"
        assert isinstance(df[col].iloc[0], coltype), (
            f"Column {col} is not {coltype}"
        )

    # Check that the base sequence matches the pattern [A-Z]+
    assert df["Base Sequence"].str.match("[A-Z]+").all(), (
        "Base sequence must only contain amino acids"
    )

    return True


@pytest.fixture
def flashlfq_psms_ds(psm_df_builder):
    """A small-ish PSM dataset"""
    data = psm_df_builder(1000, 1000, score_diffs=[5.0])
    psms = LinearPsmDataset(
        psms=data.df,
        target_column="target",
        spectrum_columns="specid",
        peptide_column="peptide",
        feature_columns=list(data.score_cols),
        filename_column="filename",
        scan_column="specid",
        calcmass_column="calcmass",
        expmass_column="expmass",
        rt_column="ret_time",
        charge_column="charge",
        copy_data=True,
    )
    return psms


@pytest.fixture
def flashlfq_psms_ds_ondisk(psm_df_builder, tmp_path):
    """A small OnDiskPsmDataset"""
    data = psm_df_builder(1000, 1000, score_diffs=[5.0])
    pin = tmp_path / "test.pin"
    df = data.df
    df["label"] = df["target"]
    df.to_csv(pin, sep="\t", index=False)

    datasets = read_pin(
        [pin],
        max_workers=1,
        filename_column="filename",
        calcmass_column="calcmass",
        expmass_column="expmass",
        rt_column="ret_time",
        charge_column="charge",
    )
    assert len(datasets) == 1
    psms = datasets[0]
    return psms


@pytest.parametrize("deduplication", [True, False])
def test_internal_flashlfq_ondisk(flashlfq_psms_ds_ondisk, deduplication):
    if deduplication:
        pytest.skip("Deduplication is not working")

    mods, scores = mokapot.brew([flashlfq_psms_ds_ondisk], test_fdr=0.1)
    conf = mokapot.assign_confidence(
        [flashlfq_psms_ds_ondisk],
        scores_list=scores,
        eval_fdr=0.1,
        deduplication=deduplication,  # RN fails with deduplication = True
    )
    _tmp = _format_flashlfq(conf[0])
    for col in EXPECTED_COLS:
        assert col in _tmp.columns, f"Column {col} not found in output"


def test_internal_flashlfq(flashlfq_psms_ds):
    mods, scores = mokapot.brew([flashlfq_psms_ds], test_fdr=0.1)
    conf = mokapot.assign_confidence(
        [flashlfq_psms_ds],
        scores_list=scores,
        eval_fdr=0.1,
        deduplication=False,  # RN fails with deduplication = True
    )
    _tmp = _format_flashlfq(conf[0])


def test_sanity(flashlfq_psms_ds, tmp_path):
    """Run simple sanity checks"""

    mods, scores = mokapot.brew([flashlfq_psms_ds], test_fdr=0.1)
    conf = mokapot.assign_confidence(
        [flashlfq_psms_ds],
        scores_list=scores,
        eval_fdr=0.1,
        deduplication=False,  # RN fails with deduplication = True
    )
    test1 = conf[0].to_flashlfq(tmp_path / "test1.txt")
    mokapot.to_flashlfq(conf, tmp_path / "test2.txt")
    test3 = mokapot.to_flashlfq([conf[0], conf[0]], tmp_path / "test3.txt")
    with pytest.raises(ValueError):
        mokapot.to_flashlfq("blah", tmp_path / "test4.txt")

    df1 = pd.read_table(test1)
    df3 = pd.read_table(test3)
    assert 2 * len(df1) == len(df3)
    assert len(df1.columns) == 7


# def test_basic(mock_conf, tmp_path):
#     """Test that the basic output works"""
#     conf = mock_conf
#     df = pd.read_table(mokapot.to_flashlfq(conf, tmp_path / "test.txt"))
#     expected = pd.DataFrame({
#         "File Name": ["c.mzML"] * 2,
#         "Base Sequence": ["ABCDXYZ", "ABCDEFG"],
#         "Full Sequence": ["B.ABCD[+2.817]XYZ.A", "ABCDE(shcah8)FG"],
#         "Peptide Monoisotopic Mass": [1, 2],
#         "Scan Retention Time": [60, 120],
#         "Precursor Charge": [2, 3],
#         "Protein Accession": ["A|B|C; B|C|A", "A|B|C"],
#     })
#
#     pd.testing.assert_frame_equal(df, expected)
#
#
# def test_with_missing(mock_conf, tmp_path):
#     """Test that missing columns causes errors"""
#     conf = mock_conf
#     cols = conf._optional_columns.copy()
#     for col in ["filename", "calcmass", "rt", "charge"]:
#         new_cols = cols.copy()
#         new_cols[col] = None
#         conf._optional_columns = new_cols
#         with pytest.raises(ValueError):
#             mokapot.to_flashlfq(conf, tmp_path / "test.txt")
#
#
# def test_no_proteins(mock_conf, tmp_path):
#     """Test when no proteins are available"""
#     conf = mock_conf
#     conf._protein_column = None
#     df = pd.read_table(mokapot.to_flashlfq(conf, tmp_path / "test.txt"))
#     expected = pd.Series([np.nan, np.nan], name="Protein Accession")
#     pd.testing.assert_series_equal(df["Protein Accession"], expected)
#
#
# def test_fasta_proteins(mock_conf, mock_proteins, tmp_path):
#     """Test that using mokapot protein groups works"""
#     conf = mock_conf
#     conf._proteins = mock_proteins
#     conf._has_proteins = True
#     df = pd.read_table(mokapot.to_flashlfq(conf, tmp_path / "test.txt"))
#     expected = pd.Series(["X|Y|Z", "A|B|C; X|Y|Z"], name="Protein Accession")
#     pd.testing.assert_series_equal(df["Protein Accession"], expected)
#
#     conf._proteins.shared_peptides = {}
#     df = pd.read_table(mokapot.to_flashlfq(conf, tmp_path / "test.txt"))
#     expected = pd.Series(["X|Y|Z"], name="Protein Accession")
#     pd.testing.assert_series_equal(df["Protein Accession"], expected)
