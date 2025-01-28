from __future__ import annotations

from dataclasses import dataclass

from .utils import tuplize

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

    id: str | None
    filename: str | None
    scan: str | None
    calcmass: str | None
    expmass: str | None
    rt: str | None
    charge: str | None
    protein: str | None

    def as_dict(self):
        return {
            "id": self.id,
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
    """Helper class to store the columns and their names.

    Parameters
    ----------
    columns : tuple[str, ...]
        The columns that are available in the dataset.
    target_column : str
        The column that contains the target values.
        This is usually a column that labels is a given entry
        in the dataset is a target or a decoy. Internally
        we will check if this column is also a feature column
        and raise an error if it is.
    peptide_column : str
        The column that contains the peptide sequences.
        In practice we also allow peptidoform/proforma-like
        sequences. I till be used as a factor to deduplicate
        the 'peptide' level of confidences.
    spectrum_columns : tuple[str, ...]
        The columns that contain the spectrum metadata.
        These are metadata columns that specify what elements will
        compete with each other.
        Common examples of these would be: 'FileName', 'ScanNr'
        Therefore if two entries ('rows') share the same pair, only
        the top one will be kept.
    feature_columns : tuple[str, ...]
        The columns that contain the feature values.
        These columns will be used to fit the model and calculate
        the confidences. In general it is recommended to use features
        that are monotonic in nature (absolute mass error for example,
        low is always better, in contrast with mass error, where
        "close to 0 is better").
    extra_confidence_level_columns : tuple[str, ...]
        The columns that contain the extra confidence levels.
        Examples of this could be 'precursor ion', ''


    """

    columns: tuple[str, ...]
    target_column: str
    peptide_column: str
    spectrum_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    extra_confidence_level_columns: tuple[str, ...]
    optional_columns: OptionalColumns

    def __post_init__(self):
        # Make sure that all columns are present
        if self.target_column not in self.columns:
            msg = f"Target column {self.target_column} "
            msg += f"not found in columns {self.columns}"
            raise ValueError(msg)

        if self.target_column in self.feature_columns:
            msg = (
                f"Target column {self.target_column} is also a feature column"
            )
            raise ValueError(msg)

        all_cols = [self.target_column, self.peptide_column]
        all_cols += self.spectrum_columns
        all_cols += self.feature_columns
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
        for col in self.feature_columns:
            if col in seen:
                dupes.append(col)
            seen.add(col)

        if dupes:
            raise ValueError(
                f"Duplicate columns found in feature columns: {dupes}, {self}"
            )

    def update(
        self,
        *,
        target_column: str | None = None,
        peptide_column: str | None = None,
        spectrum_columns: tuple[str, ...] | list[str] | None = None,
        feature_columns: tuple[str, ...] | list[str] | None = None,
        extra_confidence_level_columns: tuple[str, ...]
        | list[str]
        | None = None,
        id_column: str | None = None,
        filename_column: str | None = None,
        scan_column: str | None = None,
        calcmass_column: str | None = None,
        expmass_column: str | None = None,
        rt_column: str | None = None,
        charge_column: str | None = None,
        protein_column: str | None = None,
    ):
        if spectrum_columns is not None:
            spectrum_columns = tuplize(spectrum_columns)
        else:
            spectrum_columns = self.spectrum_columns

        if feature_columns is not None:
            feature_columns = tuplize(feature_columns)
        else:
            feature_columns = self.feature_columns

        if extra_confidence_level_columns is not None:
            extra_conf_level_columns = tuplize(extra_confidence_level_columns)
        else:
            extra_conf_level_columns = self.extra_confidence_level_columns

        return ColumnGroups(
            columns=self.columns,
            target_column=target_column or self.target_column,
            peptide_column=peptide_column or self.peptide_column,
            spectrum_columns=spectrum_columns,
            feature_columns=feature_columns,
            extra_confidence_level_columns=extra_conf_level_columns,
            optional_columns=OptionalColumns(
                id=id_column or self.optional_columns.id,
                filename=filename_column or self.optional_columns.filename,
                scan=scan_column or self.optional_columns.scan,
                calcmass=calcmass_column or self.optional_columns.calcmass,
                expmass=expmass_column or self.optional_columns.expmass,
                rt=rt_column or self.optional_columns.rt,
                charge=charge_column or self.optional_columns.charge,
                protein=protein_column or self.optional_columns.protein,
            ),
        )

    def update_with(self, other: ColumnGroups):
        return self.update(
            target_column=other.target_column,
            peptide_column=other.peptide_column,
            spectrum_columns=other.spectrum_columns,
            feature_columns=other.feature_columns,
            extra_confidence_level_columns=other.extra_confidence_level_columns,
            id_column=other.optional_columns.id,
            filename_column=other.optional_columns.filename,
            scan_column=other.optional_columns.scan,
            calcmass_column=other.optional_columns.calcmass,
            expmass_column=other.optional_columns.expmass,
            rt_column=other.optional_columns.rt,
            charge_column=other.optional_columns.charge,
            protein_column=other.optional_columns.protein,
        )
