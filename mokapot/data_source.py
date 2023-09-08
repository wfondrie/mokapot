"""Manage data sources for mokapot."""
from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Any, Generator

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from pyarrow import ArrowInvalid

from . import utils
from .dataset import PeptideDataset, PsmDataset

LOGGER = logging.getLogger(__name__)


class BaseDataSource(ABC):
    """A base class for managing data from various sources.

    Parameters
    ----------
    source : Any
        The source of the data, such as a file or dataframe.
    rng : int or np.random.Generator
        The random number generator state.
    dataset_class : callable, optional
        The dataset class to initialize from this data source.
    required_columns : dict | None, optional
        The required columns to look for in the data, case insensitive.
    optional_columns : dict | None, optional
        The optional columns to look for in the data, case insensitive.
    **kwargs : dict
        Additional arguments to pass to the dataset initialization.
    """

    def __init__(
        self,
        source: Any,
        rng: int | np.random.Generator | None,
        dataset_class: callable = PsmDataset,
        required_columns: dict | None = None,
        default_columns: dict | None = None,
        **kwargs: dict,
    ) -> None:
        """Initialize a BaseDataSource."""
        self._source = source
        self._rng = np.random.default_rng(rng)
        self._dataset_class = dataset_class

        if required_columns is None:
            self._required_columns = {}
        else:
            self._required_columns = required_columns

        if default_columns is None:
            self._default_columns = {}
        else:
            self._default_columns = default_columns

        self._kwargs = kwargs

    @property
    def source(self) -> Any:
        """The source file/data."""
        return self._source

    @property
    def rng(self) -> np.random.Generator:
        """The random number generator."""
        return self._rng

    def to_dataset(self) -> PsmDataset | PeptideDataset:
        """Initialize a dataset object."""
        return self._dataset_class(self, **self._kwargs)

    def validate_columns(self) -> None:
        """Check for the required columns and update case."""
        columns = self.columns  # may or may not be relatively expensive.
        for col_dict in (self._default_columns, self._required_columns):
            inverted = {v.lower(): k for k, v in col_dict.items()}
            for col in columns:
                try:
                    # Update with actual case:
                    col_dict[inverted[col.lower()]] = col
                    # Record that we found it:
                    del inverted[col.lower()]
                except KeyError:
                    pass

        # `inverted` contains the remaining required columns.
        if inverted:
            raise ValueError(
                "The following columns were not in the data found: "
                ",".join(inverted.keys())
            )

    @abstractmethod
    @property
    def columns(self) -> list[str]:
        """The columns parsed from the data source."""

    @abstractmethod
    def read(self) -> None:
        """Read the data."""

    @abstractmethod
    def __len__(self) -> int:
        """The total number of examples in the data."""

    @abstractmethod
    def stream(self) -> Generator[pl.DataFrame]:
        """Stream all data from the data source in chunks.

        This may or not be true streaming for all data sources.

        Yields
        ------
        polars.DataFrame
            A chunk of the data.
        """


class PinSource(BaseDataSource):
    """Percolator input (PIN)-like files/dataframes.

    PIN files are a tabular format with specific required columns: `specid`,
    `scannr`, `peptide`, and `label` (case-insensitive). Traditionally, PIN
    files are tab-delimited text files---however this function can also accept
    `Apache Parquet <https://parquet.apache.org/>`_ files for faster reading
    and more efficient storage.

    This function reads peptide-spectrum matches (PSMs) from one or more PIN
    files, aggregating them into a single
    :py:class:`~mokapot.dataset.PsmDataset`. For more details about the PIN
    file format, see the `Percolator documentation
    <https://github.com/percolator/percolator/wiki/Interface#tab-delimited-file-format>`_.
    Additionally, either Pandas or Polars DataFrames can be provided in lieu of
    a file to be read.

    In addition to these special columns defined for the PIN format, mokapot
    also looks for additional columns that specify the MS data file names,
    theoretical monoisotopic peptide masses, the measured mass, retention
    times, and charge states, which are necessary to create specific output
    formats for downstream tools, such as FlashLFQ.

    Finally, mokapot does not currently support specifying a default direction
    or feature weights in the PIN file itself.

    .. tip::
       The last column of typical PIN file contains protein accessions that are
       also tab-delimited. By default, this function will only keep the first
       protein in this tab-delimited list. However, if you want to keep them
       all, first sanitize your PIN file using :py:func:`~mokapot.sanitize_pin()`.

       The proteins in the PIN file itself are not used for protein-level FDR
       estimation. instead. Those come from a FASTA file, specified with
       :py:func:`~mokapot.read_fasta()` and/or
       :py:func:`~mokapot.PsmDataset.add_proteins()`.

    Parameters
    ----------
    pin_files : str, tuple of str, polars.DataFrame, or pandas.DataFrame
        One or more PIN files or data frames to be read. Multiple files can be
        specified using globs, such as ``"psms_*.pin"`` or separately.
    group_column : str, optional
        A factor to by which to group PSMs for grouped confidence
        estimation.
    filename_column : str, optional
        The column specifying the MS data file. If ``None``, mokapot will
        look for a column called "filename" (case insensitive). This is
        required for some output formats, such as FlashLFQ.
    calcmass_column : str, optional
        The column specifying the theoretical monoisotopic mass of the peptide
        including modifications. If ``None``, mokapot will look for a
        column called "calcmass" (case insensitive). This is required for some
        output formats, such as FlashLFQ.
    expmass_column : str, optional
        The column specifying the measured neutral precursor mass. If
        ``None``, mokapot will look for a column call "expmass" (case
        insensitive). This is required for some output formats.
    rt_column : str, optional
        The column specifying the retention time in seconds. If ``None``,
        mokapot will look for a column called "ret_time" (case insensitive).
        This is required for some output formats, such as FlashLFQ.
    charge_column : str, optional
        The column specifying the charge state of each peptide. If
        ``None``, mokapot will look for a column called "charge" (case
        insensitive). This is required for some output formats, such as
        FlashLFQ.
    subset_max_train : int, optional
        Select a random subset of PSMs which can be used for model training
        within limited memory.
    rng : int or np.random.Generator, optional
        A seed or Generator used to generate random splits, or ``None`` to
        use the default random number generator state.
    """

    def __init__(
        self,
        source: PathLike | tuple[PathLike] | pl.DataFrame,
        subset_max_train: int = None,
        group: str = None,
        rng: int | np.random.Generator = None,
    ) -> None:
        """Initialize a PinFile."""
        super().__init__(source, rng)
        self.subset_max_train = subset_max_train

        # Cached values:
        self._len = None
        self._columns = None
        self._train_rows = None

        # Figure out the source type:
        try:
            # See if its a dataframe...
            self.source.columns
            try:
                self.source.lazy()
                self._source_type = "polars"
            except (TypeError, AttributeError):
                self._source_type = "pandas"
        except AttributeError:
            # Must be a file...
            self._source = tuple(Path(f) for f in utils.tuplize(self.source))
            try:
                pq.ParquetFile(self.source)
                self._source_type = "parquet"
            except ArrowInvalid:
                self._source_type = "tsv"

        # Specify the required columns:
        self._required_columns = {
            "peptide": "peptide",
            "target": "label",
        }

        if group is not None:
            self._required_columns["group"] = group

        self._default_columns = {
            "filename": "filename",
            "theoretical_mass": "calcmass",
            "measured_mass": "expmass",
            "retention_time": "ret_time",
            "ion_mobility": "ion_mobility",
            "charge": "charge",
        }

        if any(c.lower() == "seqid" for c in self.columns):
            self._dataset_class = PeptideDataset
            self._required_columns["pairing"] = "seqid"
        else:
            self._required_columns["spectrum"] = "specid"

    def __len__(self) -> int:
        """The total number of rows across all source files."""
        if self._len is not None:
            return self._len

        if self._source_type in ("polars", "pandas"):
            self._len = len(self.source)
        elif self._source_type == "parquet":
            self._len = 0
            for source_file in self.source:
                self._len += pq.ParquetFile(source_file).metadata.num_rows
        elif self._source_type == "tsv":
            for source_file in self.source:
                self._len = 0
                with source_file.open() as src:
                    for num, _ in enumerate(src):
                        pass

                self._len += num
        else:
            raise ValueError("Uncrecognized source type.")

        return self._len

    @property
    def columns(self) -> list[str]:
        """The columns parsed from the pin file."""
        if self._columns is not None:
            return self._columns

        if self._source_type in ("polars", "pandas"):
            self._columns = list(self.source.columns)
            return self._columns

        for idx, source_file in enumerate(self.source):
            if self._source_type == "parquet":
                cols = pq.ParquetFile(source_file).schema.names
            elif self._source_type == "tsv":
                with source_file.open() as src:
                    src.readline().rstrip().split("\t")

            if not idx:
                ref_cols = cols

            if not cols == ref_cols:
                raise ValueError("The PIN files have different columns.")

        self._columns = cols
        return self._columns

    def read(self) -> pl.DataFrame:
        """Read the PIN file(s)."""
        logging.info("Parsing PIN file(s)...")
        if self._source_type == "polars":
            return self.source

        if self._source_type == "pandas":
            return pl.from_pandas(self.source)

        if self._source_type == "parquet":
            read = functools.partial(pl.scan_parquet)
        elif self._source_type == "tsv":
            read = functools.partial(pl.scan_csv, separator="\t")
        else:
            raise ValueError("Unrecognized source type")

        pl.concat([read(f) for f in self.source], how="diagonal")
