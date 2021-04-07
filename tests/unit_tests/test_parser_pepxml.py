"""Test the pepxml parser"""
import pytest
import mokapot
import numpy as np
from lxml import etree


@pytest.fixture
def small_pepxml(tmp_path):
    """Create a small pepxml file. This one happens to be from MSFragger"""
    out_file = str(tmp_path / "test.pep.xml")
    with open(out_file, "w+") as out_ref:
        out_ref.write(PEPXML_EXAMPLE)

    return out_file


@pytest.fixture
def not_pepxml(tmp_path):
    """Create a file that is not a PepXML."""
    out_file = str(tmp_path / "test.tsv")
    with open(out_file, "w+") as out_ref:
        out_ref.write(r"Blah\tblah\blah\nblah\tblah\blah\n")

    return out_file


def test_pepxml_success(small_pepxml):
    """Test that no errors occur"""
    mokapot.read_pepxml(small_pepxml, decoy_prefix="rev_")
    mokapot.read_pepxml(
        small_pepxml, open_modification_bin_size=0.01, decoy_prefix="rev_"
    )


def test_pepxml2df(small_pepxml):
    """Test that we can create a dataframe"""
    single = mokapot.read_pepxml(small_pepxml, decoy_prefix="rev_", to_df=True)

    print(single)
    assert len(single) == 4
    assert len(single["scan"].unique()) == 2
    np.testing.assert_array_equal(single["charge_2"], np.array([1, 1, 1, 0]))
    np.testing.assert_array_equal(single["charge_3"], np.array([0, 0, 0, 1]))

    multiple = mokapot.read_pepxml(
        [small_pepxml, small_pepxml], decoy_prefix="rev_", to_df=True
    )

    assert len(multiple) == 8


def test_pepxml_oms_bin_size(small_pepxml):
    """Test that bins are working"""
    psms = mokapot.read_pepxml(
        small_pepxml,
        decoy_prefix="rev_",
        to_df=True,
        open_modification_bin_size=0.5,
    )
    mass = (
        psms["peptide"]
        .str.replace(r"^.*\[", "", regex=True)
        .str.replace(r"\]", "", regex=True)
        .astype(float)
    )
    assert ((mass - psms["mass_diff"]) <= 0.25).all()


def test_not_pepxml(not_pepxml):
    """Test that parsing fails gracefully"""
    with pytest.raises(ValueError):
        mokapot.read_pepxml(not_pepxml)


def test_xl_df(small_pepxml):
    """Test that we can parse crosslinked PSMs"""
    csms = mokapot.read_pepxml(
        small_pepxml, decoy_prefix="rev_", parse_csms=True, to_df=True
    )
    print(csms)
    assert len(csms) == 3
    assert len(csms["scan"].unique()) == 3
    np.testing.assert_array_equal(
        csms["alpha_label"], np.array([False, True, False])
    )
    np.testing.assert_array_equal(
        csms["beta_label"], np.array([True, True, False])
    )


def test_xl_dataset(small_pepxml):
    """Make sure that we can parse to a CrosslinkPsmDataset"""
    csms = mokapot.read_pepxml(
        small_pepxml, decoy_prefix="rev_", parse_csms=True
    )
    np.testing.assert_array_equal(csms.targets, np.array([1, 2, 0]))
    expected_features = [
        "alpha_score",
        "alpha_rank",
        "alpha_link",
        "alpha_e-value",
        "alpha_ion_match",
        "alpha_consecutive_ion_match",
        "beta_score",
        "beta_rank",
        "beta_link",
        "beta_e-value",
        "beta_ion_match",
        "beta_consecutive_ion_match",
        "kojak_score",
        "delta_score",
        "ppm_error",
        "e-value",
        "ion_match",
        "consecutive_ion_match",
        "mass_diff",
        "abs_mz_diff",
        "charge_3",
        "charge_4",
    ]
    assert list(csms.features.columns) == expected_features


PEPXML_EXAMPLE = r"""<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="pepXML_std.xsl"?>
<msms_pipeline_analysis date="2018-11-29T15:10:44" xmlns="http://regis-web.systemsbiology.net/pepXML" summary_xml="Z:\sabra\RAW\mzML\Experiment1\UM_F_50cm_2019_0420.pepXML" xsi:schemaLocation="http://sashimi.sourceforge.net/schema_revision/pepXML/pepXML_v118.xsd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<msms_run_summary base_name="UM_F_50cm_2019_0420" raw_data_type="raw" raw_data=".mzXML">
<sample_enzyme name="Trypsin">
<specificity cut="KR" no_cut="P" sense="C"/>
</sample_enzyme>
<search_summary base_name="UM_F_50cm_2019_0420" precursor_mass_type="monoisotopic" search_engine="X! Tandem" search_engine_version="MSFragger-20181110" fragment_mass_type="monoisotopic" search_id="1">
<search_database local_path="/Z:/sabra/RAW/mzML/Experiment1/2018-11-29-td-UP000005640.fas" type="AA"/>
<enzymatic_search_constraint enzyme="default" min_number_termini="2" max_num_internal_cleavages="2"/>
<aminoacid_modification aminoacid="C" massdiff="57.0215" mass="160.0307" variable="N"/>
<aminoacid_modification aminoacid="K" massdiff="229.1629" mass="357.2579" variable="N"/>
<aminoacid_modification aminoacid="M" massdiff="15.9949" mass="147.0354" variable="Y"/>
<terminal_modification massdiff="42.0106" protein_terminus="Y" mass="43.0184" terminus="N" variable="Y"/>
<terminal_modification massdiff="229.1629" protein_terminus="N" mass="230.1708" terminus="N" variable="Y"/>
</search_summary>
<spectrum_query start_scan="8" assumed_charge="2" spectrum="UM_F_50cm_2019_0420.8.8.2" end_scan="8" index="1" precursor_neutral_mass="989.6051" retention_time_sec="123.372">
<search_result>
<search_hit peptide="QATARSK" massdiff="0.0230" calc_neutral_pep_mass="989.5821" peptide_next_aa="S" num_missed_cleavages="1" num_tol_term="2" num_tot_proteins="1" tot_num_ions="12" hit_rank="1" num_matched_ions="7" protein="rev_sp|Q9HBJ0|PLAC1_HUMAN Placenta-specific protein 1 OS=Homo sapiens OX=9606 GN=PLAC1 PE=2 SV=1" peptide_prev_aa="K" is_rejected="0">
<modification_info>
<mod_aminoacid_mass mass="357.2579" position="7"/>
</modification_info>
<search_score name="hyperscore" value="14.534"/>
<search_score name="nextscore" value="14.114"/>
<search_score name="expect" value="1.768e+00"/>
</search_hit>
<search_hit peptide="TATGVQGK" massdiff="0.0342" calc_neutral_pep_mass="989.5709" peptide_next_aa="E" num_missed_cleavages="0" num_tol_term="2" num_tot_proteins="1" tot_num_ions="14" hit_rank="2" num_matched_ions="8" protein="sp|Q9Y3P9|RBGP1_HUMAN Rab GTPase-activating protein 1 OS=Homo sapiens OX=9606 GN=RABGAP1 PE=1 SV=3" peptide_prev_aa="K" is_rejected="0">
<modification_info>
<mod_aminoacid_mass mass="357.2579" position="8"/>
</modification_info>
<search_score name="hyperscore" value="14.114"/>
<search_score name="nextscore" value="13.605"/>
<search_score name="expect" value="2.264e+00"/>
</search_hit>
<search_hit peptide="TQSIQGK" massdiff="0.0342" calc_neutral_pep_mass="989.5709" peptide_next_aa="G" num_missed_cleavages="0" num_tol_term="2" num_tot_proteins="1" tot_num_ions="12" hit_rank="3" num_matched_ions="7" protein="sp|Q53HL2|BOREA_HUMAN Borealin OS=Homo sapiens OX=9606 GN=CDCA8 PE=1 SV=2" peptide_prev_aa="R" is_rejected="0">
<modification_info>
<mod_aminoacid_mass mass="357.2579" position="7"/>
</modification_info>
<search_score name="hyperscore" value="13.605"/>
<search_score name="nextscore" value="13.315"/>
<search_score name="expect" value="3.056e+00"/>
</search_hit>
</search_result>
</spectrum_query>
<spectrum_query start_scan="9" assumed_charge="3" spectrum="UM_F_50cm_2019_0420.9.9.2" end_scan="9" index="2" precursor_neutral_mass="822.5355" retention_time_sec="123.442">
<search_result>
<search_hit peptide="RPAPLLR" massdiff="1.0120" calc_neutral_pep_mass="821.5235" peptide_next_aa="V" num_missed_cleavages="0" num_tol_term="2" num_tot_proteins="1" tot_num_ions="12" hit_rank="1" num_matched_ions="5" protein="rev_sp|O00445|SYT5_HUMAN Synaptotagmin-5 OS=Homo sapiens OX=9606 GN=SYT5 PE=1 SV=2" peptide_prev_aa="K" is_rejected="0">
<search_score name="hyperscore" value="9.293"/>
<search_score name="nextscore" value="8.009"/>
<search_score name="expect" value="2.003e+00"/>
</search_hit>
</search_result>
</spectrum_query>

<!-- Some crosslinked PSMs -->
<spectrum_query spectrum="XLpeplib_Beveridge_QEx-HFX_DSS_R1.17647.17647.4" start_scan="17647" end_scan="17647" precursor_neutral_mass="2795.369395" assumed_charge="4" index="90" retention_time_sec="79.4">
<search_result>
<search_hit hit_rank="1" peptide="-" peptide_prev_aa="-" peptide_next_aa="-" protein="-" num_tot_proteins="1" calc_neutral_pep_mass="2795.383230" massdiff="0.013835" xlink_type="xl">
<xlink identifier="BS3" mass="138.068074">
<linked_peptide peptide="CALGSLVPQIAPVGR" peptide_prev_aa="-" peptide_next_aa="L" protein="rev_sp|CTRB_BOVIN|" peptide_start_pos="1" protein_link_pos_a="1" num_tot_proteins="1" calc_neutral_pep_mass="1536.844554" complement_mass="1258.524840" designation="alpha">
<modification_info>
<mod_aminoacid_mass position="1" mass="160.030640" static="57.021460"/>
</modification_info>
<xlink_score name="score" value="0.3400"/>
<xlink_score name="rank" value="1"/>
<xlink_score name="link" value="1"/>
<xlink_score name="e-value" value="5.666e+02"/>
<xlink_score name="ion_match" value="5"/>
<xlink_score name="consecutive_ion_match" value="4"/>
</linked_peptide>
<linked_peptide peptide="MAEEVEEER" peptide_prev_aa="-" peptide_next_aa="L" protein="sp|SRPP_HEVBR|" peptide_start_pos="1" protein_link_pos_a="1" num_tot_proteins="1" calc_neutral_pep_mass="1120.470602" complement_mass="1674.898793" designation="beta">
<xlink_score name="score" value="0.2350"/>
<xlink_score name="rank" value="2"/>
<xlink_score name="link" value="1"/>
<xlink_score name="e-value" value="8.546e+02"/>
<xlink_score name="ion_match" value="2"/>
<xlink_score name="consecutive_ion_match" value="1"/>
</linked_peptide>
</xlink>
<search_score name="kojak_score" value="0.5750"/>
<search_score name="delta_score" value="0.2000"/>
<search_score name="ppm_error" value="4.9493"/>
<search_score name="e-value" value="3.490e+02"/>
<search_score name="ion_match" value="7"/>
<search_score name="consecutive_ion_match" value="2"/>
</search_hit>
</search_result>
</spectrum_query>

<spectrum_query spectrum="XLpeplib_Beveridge_QEx-HFX_DSS_R1.17647.17647.4" start_scan="17648" end_scan="17648" precursor_neutral_mass="2795.369395" assumed_charge="3" index="90" retention_time_sec="79.4">
<search_result>
<search_hit hit_rank="1" peptide="-" peptide_prev_aa="-" peptide_next_aa="-" protein="-" num_tot_proteins="1" calc_neutral_pep_mass="2795.383230" massdiff="0.013835" xlink_type="xl">
<xlink identifier="BS3" mass="138.068074">
<linked_peptide peptide="CALGSLVPQIAPVGR" peptide_prev_aa="-" peptide_next_aa="L" protein="sp|CTRB_BOVIN|" peptide_start_pos="1" protein_link_pos_a="1" num_tot_proteins="1" calc_neutral_pep_mass="1536.844554" complement_mass="1258.524840" designation="alpha">
<modification_info>
<mod_aminoacid_mass position="1" mass="160.030640" static="57.021460"/>
</modification_info>
<xlink_score name="score" value="0.3400"/>
<xlink_score name="rank" value="1"/>
<xlink_score name="link" value="1"/>
<xlink_score name="e-value" value="5.666e+02"/>
<xlink_score name="ion_match" value="5"/>
<xlink_score name="consecutive_ion_match" value="4"/>
</linked_peptide>
<linked_peptide peptide="MAEEVEEER" peptide_prev_aa="-" peptide_next_aa="L" protein="sp|SRPP_HEVBR|" peptide_start_pos="1" protein_link_pos_a="1" num_tot_proteins="1" calc_neutral_pep_mass="1120.470602" complement_mass="1674.898793" designation="beta">
<xlink_score name="score" value="0.2350"/>
<xlink_score name="rank" value="2"/>
<xlink_score name="link" value="1"/>
<xlink_score name="e-value" value="8.546e+02"/>
<xlink_score name="ion_match" value="2"/>
<xlink_score name="consecutive_ion_match" value="1"/>
</linked_peptide>
</xlink>
<search_score name="kojak_score" value="0.5750"/>
<search_score name="delta_score" value="0.2000"/>
<search_score name="ppm_error" value="4.9493"/>
<search_score name="e-value" value="3.490e+02"/>
<search_score name="ion_match" value="7"/>
<search_score name="consecutive_ion_match" value="2"/>
</search_hit>
</search_result>
</spectrum_query>

<spectrum_query spectrum="XLpeplib_Beveridge_QEx-HFX_DSS_R1.17647.17647.4" start_scan="17649" end_scan="17649" precursor_neutral_mass="2795.369395" assumed_charge="4" index="90" retention_time_sec="79.4">
<search_result>
<search_hit hit_rank="1" peptide="-" peptide_prev_aa="-" peptide_next_aa="-" protein="-" num_tot_proteins="1" calc_neutral_pep_mass="2795.383230" massdiff="0.013835" xlink_type="xl">
<xlink identifier="BS3" mass="138.068074">
<linked_peptide peptide="CALGSLVPQIAPVGR" peptide_prev_aa="-" peptide_next_aa="L" protein="rev_sp|CTRB_BOVIN|" peptide_start_pos="1" protein_link_pos_a="1" num_tot_proteins="1" calc_neutral_pep_mass="1536.844554" complement_mass="1258.524840" designation="alpha">
<modification_info>
<mod_aminoacid_mass position="1" mass="160.030640" static="57.021460"/>
</modification_info>
<xlink_score name="score" value="0.3400"/>
<xlink_score name="rank" value="1"/>
<xlink_score name="link" value="1"/>
<xlink_score name="e-value" value="5.666e+02"/>
<xlink_score name="ion_match" value="5"/>
<xlink_score name="consecutive_ion_match" value="4"/>
</linked_peptide>
<linked_peptide peptide="MAEEVEEER" peptide_prev_aa="-" peptide_next_aa="L" protein="rev_sp|SRPP_HEVBR|" peptide_start_pos="1" protein_link_pos_a="1" num_tot_proteins="1" calc_neutral_pep_mass="1120.470602" complement_mass="1674.898793" designation="beta">
<xlink_score name="score" value="0.2350"/>
<xlink_score name="rank" value="2"/>
<xlink_score name="link" value="1"/>
<xlink_score name="e-value" value="8.546e+02"/>
<xlink_score name="ion_match" value="2"/>
<xlink_score name="consecutive_ion_match" value="1"/>
</linked_peptide>
</xlink>
<search_score name="kojak_score" value="0.5750"/>
<search_score name="delta_score" value="0.2000"/>
<search_score name="ppm_error" value="4.9493"/>
<search_score name="e-value" value="3.490e+02"/>
<search_score name="ion_match" value="7"/>
<search_score name="consecutive_ion_match" value="2"/>
</search_hit>
</search_result>
</spectrum_query>

</msms_run_summary>
</msms_pipeline_analysis>
"""
