from dataclasses import dataclass


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

    def as_dict(self):
        return {
            "filename": self.filename,
            "scan": self.scan,
            "calcmass": self.calcmass,
            "expmass": self.expmass,
            "rt": self.rt,
            "charge": self.charge,
        }


@dataclass
class ColumnGroups:
    columns: list[str]
    target_column: str
    peptide_column: str
    spectrum_columns: list[str]
    feature_columns: list[str]
    extra_confidence_level_columns: list[str]
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
