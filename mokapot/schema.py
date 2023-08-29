"""Define the columns in a dataset.

PsmColumns and PeptideColumns are used to tell mokapot the
meaning of columns in a dataset.
"""
from dataclasses import dataclass

import polars as pl

from . import utils


class _SchemaValidatorMixin:
    """Methods to validate columns for a dataset.

    Parameters
    ----------
    required : list of str
        The required parameters.
    single : list of str
        The parameters that are expected to be single value.
    variadic : list of str
        The parameters that may or may not have multiple valus.
    """

    def __init__(
        self, required: list[str], single: list[str], variadic: list[str]
    ) -> None:
        """Initialize the ColumnValidator"""
        self._required = required
        self._single = single
        self._variadic = variadic

        # Initial validation:
        self._validate_required()
        self._validate_single()
        self._listify_variadic()

    def _validate_required(self) -> None:
        """Validate required columns.

        Parameters
        ----------
        data : polars.LazyFrame
            The dataframe to validate with.
        """
        missing = [c for c in self._required if getattr(self, c) is None]
        if not missing:
            return

        raise ValueError(
            "The following parameters are required:" ", ".join(missing)
        )

    def _validate_single(self) -> None:
        """Validate single columns to only have one value."""
        failed = []
        for param in self._single:
            if param is None:
                continue

            if len(utils.listify(getattr(self, param))) > 1:
                failed.append(param)

        if not failed:
            return

        raise ValueError(
            "The following parameters must be single columns:"
            ", ".join(failed)
        )

    def _listify_variadic(self) -> None:
        """Listify the variadic parameters."""
        for param in self._variadic:
            if getattr(self, param) is None:
                continue

            setattr(self, param, utils.listify(getattr(self, param)))

    def _validate_all_present(self, data: pl.LazyFrame) -> None:
        """Validate that all of the columns are present.

        Parameters
        ----------
        data : polars.LazyFrame
            The data to validate on.
        """
        all_cols = [
            utils.listify(getattr(self, p))
            for p in (self._single + self._variadic)
        ]

        # all of the unique columns:
        all_cols = set(sum(all_cols, []))

        # Add features as needed:
        if self.features is None:
            self.features = [c for c in data.columns if c not in all_cols]

        # Verify they all specified columns are present in the data.
        missing = [
            c for c in all_cols if c not in data.columns and c is not None
        ]
        if not missing:
            return

        raise ValueError(
            "The following columns were not found in the data: "
            ", ".join(missing)
        )

    def _validate_labels(self, data: pl.LazyFrame) -> bool:
        """Validate that the labels are in the correct format.

        Parameters
        ----------
        data : polars.LazyFrame
            The data to validate on.

        Returns
        -------
        bool
            This will be :code:`True` if Percolator-style {1, -1}
            labels are used.
        """
        dtype = data.schema[self.target]
        if dtype == pl.Boolean:
            return False

        perc_style = pl.DataFrame({self.target: [-1, 1]})
        moka_style = pl.DataFrame({self.target: [0, 1]})
        vals = (
            data.select(self.target)
            .unique()
            .sort(self.target)
            .collect(streaming=True)
        )
        if vals.frame_equal(perc_style):
            return True

        if vals.frame_equal(moka_style):
            return False

        raise ValueError("The target column is not in a recognized format.")

    def validate(self, data: pl.LazyFrame) -> bool:
        """Validate columns against the data.

        Parameters
        ----------
        data : polars.LazyFrame
            The data to validate on.

        Returns
        -------
        bool
            This will be :code:`True` if Percolator-style {1, -1}
            labels are used.
        """
        self._validate_all_present(data)
        return self._validate_labels(data)


@dataclass
class PsmSchema(_SchemaValidatorMixin):
    """The columns for a dataset of PSMs.

    Parameters
    ----------
    target : str
        The column specifying whether each PSM is a target (:code:`True`) or a
        decoy (:code:`False`).
    spectrum : str or list of str
        The column(s) that collectively identify unique mass spectra. Multiple
        columns can be useful to avoid combining scans from multiple mass
        spectrometry runs.
    peptide : str
        The column that defines a unique peptide. Sequences with modifications
        should denoted with ProForma-style square brackets :code:`[]` or
        parentheses :code:`()`.
    group : str, optional
        A factor by which to group PSMs for grouped confidence estimation.
    file : str, optional
        The column specifying the mass spectrometry data file (e.g. mzML)
        containing each spectrum. This is required for FlashLFQ output.
    scan : str, optional
        The column specifying the scan identifier for each spectrum.
    calcmass : str, optional
        The column specifying the theoretical monoisotopic mass of each
        peptide. This is required for some output formats, such as mzTab and
        FlashLFQ.
    expmass : str, optional
        The column specifying the measured neutral precursor mass. This is
        required for the some ouput formats.
    ret_time : str, optional
        The column specifying the retention time of each spectrum, in seconds.
        This is required for FlashLFQ output.
    charge : str, optional
        The column specifying the charge state. This is required for FlashLFQ
        output.
    metadata : str or list of str, optional
        Columns specified here are merely passed through to the output tables.
    features : str or list of str, optional
        The column(s) specifying the feature(s) to learn from. If
        :code:`None`, these are assumed to be all of the columns that were not
        specified in the other parameters.
    score : str, optional
        The column to use as the default score for ranking PSMs.
    desc : bool, optional
        Indicates that higher scores in the score column are better. This is
        omly relevant if a score column is specified.
    """

    target: str
    spectrum: list[str]
    peptide: str
    group: str | list[str] | None = None
    file: str | None = None
    scan: str | None = None
    calcmass: str | None = None
    expmass: str | None = None
    ret_time: str | None = None
    charge: str | None = None
    metadata: str | list[str] | None = None
    features: str | list[str] | None = None
    score: str | None = None
    desc: bool = True

    def __post_init__(self):
        """Perform initial validation."""
        super().__init__(
            required=["target", "spectrum", "peptide"],
            single=[
                "target",
                "peptide",
                "file",
                "scan",
                "calcmass",
                "expmass",
                "ret_time",
                "charge",
                "score",
            ],
            variadic=["spectrum", "metadata", "group", "features"],
        )

        if self.metadata is None:
            self.metadata = []
