from __future__ import annotations

from dataclasses import dataclass

from typeguard import typechecked

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
        Examples of this could be 'precursor ion', 'peptide_group'
        or 'modified_peptide'
    optional_columns : OptionalColumns
        These are columns that are not used during the scoring/training
        but might be used for specific output formats.
        For example the id column will be propagated to the output
        table and is used to identify the PSM. The filename column
        might be required for some output formats, such as FlashLFQ.

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
            msg = f"Target column '{self.target_column}' "
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

    def get_unused_columns(self) -> list[str]:
        # Get columns not assigned to any field
        out = set(self.columns)
        out -= set([self.target_column])
        out -= set([self.peptide_column])
        out -= set(self.spectrum_columns)
        out -= set(self.feature_columns)
        out -= set(self.extra_confidence_level_columns)
        out -= set(self.optional_columns.as_dict().values())
        return list(out)

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

    @classmethod
    def infer_from_colnames(
        cls,
        colnames: list[str],
        filename_column: str | None = None,
        calcmass_column: str | None = None,
        expmass_column: str | None = None,
        rt_column: str | None = None,
        charge_column: str | None = None,
    ) -> ColumnGroups:
        # Note: I want to delete these arguments in the future
        #       Since they are never used when calling the API
        #       and this class makes them un-necessary.
        return default_column_inference(
            colnames,
            filename_column=filename_column,
            calcmass_column=calcmass_column,
            expmass_column=expmass_column,
            rt_column=rt_column,
            charge_column=charge_column,
        )


def default_column_inference(
    columns: list[str],
    filename_column=None,
    calcmass_column=None,
    expmass_column=None,
    rt_column=None,
    charge_column=None,
) -> ColumnGroups:
    # Find all the necessary columns, case-insensitive:
    if "id" in columns[0].lower():
        # If the first column has 'id' it will be used as an identifier.
        # Since both PSMid (percolator docs) and SpecId (sage) are valid
        # options.
        specid = columns[0]
    else:
        specid = find_required_column("specid", columns)

    peptides = find_required_column("peptide", columns)
    proteins = find_required_column("proteins", columns)
    labels = find_required_column("label", columns)
    scan = find_required_column("scannr", columns)
    nonfeat = [specid, scan, peptides, proteins, labels]

    # Columns for different rollup levels
    # Currently no proteins, since the protein rollup is probably quite
    # different from the other rollup levels IMHO
    modifiedpeptides = find_columns("modifiedpeptide", columns)
    precursors = find_columns("precursor", columns)
    peptidegroups = find_columns("peptidegroup", columns)
    extra_confidence_level_columns = (
        modifiedpeptides + precursors + peptidegroups
    )
    nonfeat += modifiedpeptides + precursors + peptidegroups

    # Optional columns
    filename = find_optional_column(filename_column, columns, "filename")
    calcmass = find_optional_column(calcmass_column, columns, "calcmass")
    expmass = find_optional_column(expmass_column, columns, "expmass")
    ret_time = find_optional_column(rt_column, columns, "ret_time")
    charge = find_optional_column(charge_column, columns, "charge_column")
    # Q: Why isnt `specid` used here?
    # A: Bc the specid is more accurately a PSM id, since multiple can occur
    #    for a single scan, thus should not beused to separate the "elements"
    #    that compete with each other.
    spectra = [c for c in [filename, scan, ret_time, expmass] if c is not None]

    # Only add charge to features if there aren't other charge columns:
    alt_charge = [c for c in columns if c.lower().startswith("charge")]
    if charge is not None and len(alt_charge) > 1:
        nonfeat.append(charge)

    for col in [filename, calcmass, expmass, ret_time]:
        if col is not None:
            nonfeat.append(col)

    features = [c for c in columns if c not in nonfeat]

    # Check for errors:
    if not all(spectra):
        raise ValueError(
            "This PIN format is incompatible with mokapot. Please"
            " verify that the required columns are present."
        )

    column_groups = ColumnGroups(
        columns=tuplize(columns),
        target_column=labels,
        peptide_column=peptides,
        spectrum_columns=tuplize(spectra),
        feature_columns=tuplize(features),
        extra_confidence_level_columns=tuplize(extra_confidence_level_columns),
        optional_columns=OptionalColumns(
            id=specid,
            filename=filename,
            scan=scan,
            calcmass=calcmass,
            expmass=expmass,
            rt=ret_time,
            charge=charge,
            protein=proteins,
        ),
    )

    return column_groups


@typechecked
def find_column(
    col: str, columns: list[str], required=True, unique=True, ignore_case=False
) -> str | list[str] | None:
    """
    Parameters
    ----------
    col : str
        The column name to search for.

    columns : list[str]
        The list of column names to search within.

    required : bool, optional
        Specifies whether the column is required. If set to True (default),
        an error will be raised if the column is not found.

    unique : bool, optional
        Specifies whether the column should be unique.
        If set to True (default), an error will be raised if multiple
        columns with the same name are found.

    ignore_case : bool, optional
        Specifies whether case should be ignored when comparing column names.
        If set to True, the comparison will be case-insensitive.

    Returns
    -------
    str or list[str] or None
        Returns the matched column name(s) based on the search criteria.

    Raises
    ------
    ValueError
        If the column is required and not found, or if multiple columns
            are found but unique is set to True.
    """
    if ignore_case:

        def str_compare(str1, str2):
            return str1.lower() == str2.lower()

    else:

        def str_compare(str1, str2):
            return str1 == str2

    found_columns = [c for c in columns if str_compare(c, col)]

    if required and len(found_columns) == 0:
        raise ValueError(
            f"The column '{col}' was not found. (Options: {columns})"
        )

    if unique:
        if len(found_columns) > 1:
            raise ValueError(
                f"The column '{col}' should be unique. Found {found_columns}."
            )
        return found_columns[0] if len(found_columns) > 0 else None
    else:
        return found_columns


@typechecked
def find_columns(col: str, columns: list[str]) -> list[str]:
    """
    Parameters
    ----------
    col : str
        The column name to search for.

    columns : List[str]
        The list of columns to search within.

    Returns
    -------
    List[str]
        The list of columns that match the given column name, ignoring case
        sensitivity.
    """
    return find_column(
        col, columns, required=False, unique=False, ignore_case=True
    )


@typechecked
def find_required_column(col: str, columns: list[str]) -> str:
    """
    Parameters
    ----------
    col : str
        The column to search for (case-insensitive).

    columns : list[str]
        The list of columns to search within.

    Returns
    -------
    str
        The required column found with correct case.

    Raises
    ------
    ValueError
        If the column was not found or not unique.
    """
    return find_column(
        col, columns, required=True, unique=True, ignore_case=True
    )


@typechecked
def find_optional_column(
    col: str | None, columns: list[str], default: str
) -> str | None:
    """
    Parameters
    ----------
    col : Optional[str]
        The column to check. If None, the default column will be searched in
        `columns`.

    columns : List[str]
        The list of available columns to check against.

    default : str
        The default column to search for if `col` is None.

    Returns
    -------
    Optional[str]:
        The validated column. If `col` is None, it returns the first matching
        column in `columns` with case-insensitive comparison to `default`. If
        `col` is not None, it returns `col` after ensuring it is present in
        `columns`.

    Raises
    ------
    ValueError
        If `col` is not None and it is not found in `columns`.
    """
    return find_column(
        col or default,
        columns,
        required=col is not None,
        unique=True,
        ignore_case=col is None,
    )
