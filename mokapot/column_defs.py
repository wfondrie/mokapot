from dataclasses import dataclass

Q_VALUE_COL_NAME = "mokapot_qvalue"
STANDARD_COLUMN_NAME_MAP = {
    "SpecId": "psm_id",
    "PSMId": "psm_id",
    "Precursor": "precursor",
    "pcm": "precursor",
    "PCM": "precursor",
    "Peptide": "peptide",
    "PeptideGroup": "peptide_group",
    "peptidegroup": "peptide_group",
    "ModifiedPeptide": "modified_peptide",
    "modifiedpeptide": "modified_peptide",
    # "q-value": "q_value",
    "q-value": Q_VALUE_COL_NAME,
}


def get_standard_column_name(name):
    return STANDARD_COLUMN_NAME_MAP.get(name, name)


@dataclass
class OptionalColumns:
    """Helper class meant to store the optional columns from a dataset.

    It is used internally to pass the columns to places like the flashlfq
    formatter, which needs columns that are not inherently associated with
    the scoring process.
    """

    filename: str | None
    scan: str | None
    calcmass: str | None
    expmass: str | None
    rt: str | None
    charge: str | None
    protein: str | None

    def as_dict(self):
        return {
            "filename": self.filename,
            "scan": self.scan,
            "calcmass": self.calcmass,
            "expmass": self.expmass,
            "rt": self.rt,
            "charge": self.charge,
            "protein": self.protein,
        }


@dataclass(frozen=True, slots=True)
class ColumnGroups:
    columns: tuple[str, ...]
    target_column: str
    peptide_column: str
    spectrum_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    extra_confidence_level_columns: tuple[str, ...]
    optional_columns: OptionalColumns

    def __post_init__(self):
        # Make sure that all columns are present
        all_cols = [self.target_column, self.peptide_column]
        all_cols += self.spectrum_columns
        all_cols += self.feature_columns
        all_strict_cols = tuple(all_cols)
        for v in self.optional_columns.as_dict().values():
            if v is not None:
                all_cols.append(v)

        missing_cols = [c for c in all_cols if c not in self.columns]
        if missing_cols:
            raise ValueError(
                "The following specified columns were not found: "
                f"{missing_cols}, available columns are: {self.columns}"
            )

        # make sure there are no duplicates in all cols
        seen = set()
        dupes = []
        for col in all_strict_cols:
            if col in seen:
                dupes.append(col)
            seen.add(col)

        if dupes:
            raise ValueError(
                f"Duplicate columns found in all columns: {dupes}, {self}"
            )
