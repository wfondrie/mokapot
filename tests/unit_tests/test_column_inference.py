from mokapot.column_defs import ColumnGroups, OptionalColumns

# From sage 0.15.0.alpha
SAGE_COLNAMES = [
    "SpecId",
    "Label",
    "ScanNr",
    "ExpMass",
    "CalcMass",
    "FileName",
    "retentiontime",
    "ion_mobility",
    "rank",
    "z=2",
    "z=3",
    "z=4",
    "z=5",
    "z=6",
    "z=other",
    "peptide_len",
    "missed_cleavages",
    "semi_enzymatic",
    "isotope_error",
    "ln(precursor_ppm)",
    "fragment_ppm",
    "ln(hyperscore)",
    "ln(delta_next)",
    "ln(delta_best)",
    "aligned_rt",
    "predicted_rt",
    "sqrt(delta_rt_model)",
    "predicted_mobility",
    "sqrt(delta_mobility)",
    "matched_peaks",
    "longest_b",
    "longest_y",
    "longest_y_pct",
    "ln(matched_intensity_pct)",
    "scored_candidates",
    "ln(-poisson)",
    "posterior_error",
    "Peptide",
    "Proteins",
]

COMET_COLNAMES = [
    "SpecID",
    "Label",
    "ScanNr",
    "ExpMass",
    "CalcMass",
    "lnrSp",
    "deltLCn",
    "deltCn",
    "lnExpect",
    "Xcorr",
    "Sp",
    "IonFrac",
    "Mass",
    "PepLen",
    "Charge1",
    "Charge2",
    "enzN",
    "enzC",
    "enzInt",
    "lnNumSp",
    "dM",
    "absdM",
    "Peptide",
    "Proteins",
]

# https://github.com/percolator/percolator/wiki/Interface
PERC_DOCS_SAMPLE = [
    "PSMId",
    "Label",
    "ScanNr",
    "feature1name",
    "featureNname",
    "Peptide",
    "Proteins",
]

MSAID_COLNAMES = [
    "SpecId",
    "Label",
    "ScanNr",
    "ExpMass",
    "Mass",
    "MS8_feature_5",
    "missedCleavages",
    "MS8_feature_7",
    # There are other couple of hundred features here
    # but this should be enough to test the parsing
    "MS8_feature_13",
    "MS8_feature_20",
    "MS8_feature_158",
    "Peptide",
    "Proteins",
]


def test_column_inference_sage():
    cg = ColumnGroups.infer_from_colnames(SAGE_COLNAMES)
    expected_cg = ColumnGroups(
        columns=tuple(SAGE_COLNAMES),
        target_column="Label",
        peptide_column="Peptide",
        spectrum_columns=("FileName", "ScanNr", "ExpMass"),
        feature_columns=(
            "retentiontime",
            "ion_mobility",
            "rank",
            "z=2",
            "z=3",
            "z=4",
            "z=5",
            "z=6",
            "z=other",
            "peptide_len",
            "missed_cleavages",
            "semi_enzymatic",
            "isotope_error",
            "ln(precursor_ppm)",
            "fragment_ppm",
            "ln(hyperscore)",
            "ln(delta_next)",
            "ln(delta_best)",
            "aligned_rt",
            "predicted_rt",
            "sqrt(delta_rt_model)",
            "predicted_mobility",
            "sqrt(delta_mobility)",
            "matched_peaks",
            "longest_b",
            "longest_y",
            "longest_y_pct",
            "ln(matched_intensity_pct)",
            "scored_candidates",
            "ln(-poisson)",
            "posterior_error",
        ),
        extra_confidence_level_columns=(),
        optional_columns=OptionalColumns(
            id="SpecId",
            filename="FileName",
            scan="ScanNr",
            calcmass="CalcMass",
            expmass="ExpMass",
            rt=None,
            charge=None,
            protein="Proteins",
        ),
    )

    assert cg == expected_cg


def test_column_inference_percolator():
    cg = ColumnGroups.infer_from_colnames(PERC_DOCS_SAMPLE)
    expected_cg = ColumnGroups(
        columns=(
            "PSMId",
            "Label",
            "ScanNr",
            "feature1name",
            "featureNname",
            "Peptide",
            "Proteins",
        ),
        target_column="Label",
        peptide_column="Peptide",
        spectrum_columns=("ScanNr",),
        feature_columns=("feature1name", "featureNname"),
        extra_confidence_level_columns=(),
        optional_columns=OptionalColumns(
            id="PSMId",
            filename=None,
            scan="ScanNr",
            calcmass=None,
            expmass=None,
            rt=None,
            charge=None,
            protein="Proteins",
        ),
    )

    assert cg == expected_cg


def test_column_inference_msaid():
    cg = ColumnGroups.infer_from_colnames(MSAID_COLNAMES)
    expected_out = ColumnGroups(
        columns=tuple(MSAID_COLNAMES),
        target_column="Label",
        peptide_column="Peptide",
        spectrum_columns=("ScanNr", "ExpMass"),
        feature_columns=(
            "Mass",
            "MS8_feature_5",
            "missedCleavages",
            "MS8_feature_7",
            "MS8_feature_13",
            "MS8_feature_20",
            "MS8_feature_158",
        ),
        extra_confidence_level_columns=(),
        optional_columns=OptionalColumns(
            id="SpecId",
            filename=None,
            scan="ScanNr",
            calcmass=None,
            expmass="ExpMass",
            rt=None,
            charge=None,
            protein="Proteins",
        ),
    )
    assert cg == expected_out
