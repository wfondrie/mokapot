"""Parsers for reading in PSMs."""
import gzip
import logging
from io import StringIO
from os import PathLike
from typing import IO, Iterable

import numpy as np
import polars as pl

from .. import utils
from ..dataset import PsmDataset
from ..proteins import Proteins
from ..schema import PsmSchema

LOGGER = logging.getLogger(__name__)


def read_pin(
    pin_files: PathLike | pl.DataFrame | Iterable[PathLike | pl.DataFrame],
    group: str = None,
    file: str = None,
    calcmass: str = None,
    expmass: str = None,
    ret_time: str = None,
    charge: str = None,
    proteins: Proteins | None = None,
    eval_fdr: float = 0.01,
    subset: int | None = None,
    rng: int | np.random.Generator | None = None,
    strict_parsing: bool = False,
) -> PsmDataset:
    """Read Percolator input (PIN) tab-delimited files.

    Read PSMs from one or more Percolator input (PIN) tab-delmited files,
    aggregating them into a single
    :py:class:`~mokapot.dataset.PsmDataset`. For more details about the
    PIN file format, see the `Percolator documentation
    <https://github.com/percolator/percolator/
    wiki/Interface#tab-delimited-file-format>`_.
    This function also works on Parquet versions of PIN files.

    Specifically, mokapot requires the following columns in the PIN files:
    `specid`, `scannr`, `peptide`, and `label`. Note that these column names
    are case-insensitive. In addition to these special columns that are defined
    for the PIN format, mokapot also looks for additional columns that specify
    the MS data file names, theoretical monoisotopic peptide masses, the
    measured mass, retention times, and charge states, which are necessary to
    create specific output formats for downstream tools, such as FlashLFQ.

    In addition to PIN tab-delimited files, the `pin_files` argument can be a
    :py:class:`pl.DataFrame` or :py:class:`pandas.DataFrame` containing the
    above columns.

    Finally, mokapot does not currently support specifying a default direction
    or feature weights in the PIN file itself. If these are present, they
    will be ignored.

    Warning
    -------
    The final column in the PIN format is a tab-delimited list of proteins. For
    efficiency and because mokapot does not use this column for protein
    inference, the default behavior is to truncate this column to the first
    protein in each row. If this is not desired, use the `strict_parsing`
    parameter. Note that parsing data in this manner will not allow for
    memory-efficient data streaming that is normally used.

    Parameters
    ----------
    pin_files : PathLike, polars.DataFrame, iterable of either
        One or more PIN files or data frames to be read. Multiple files can be
        specified using globs, such as ``"psms_*.pin"`` or separately.
    group : str, optional
        A factor to by which to group PSMs for grouped confidence
        estimation.
    file : str, optional
        The column specifying the MS data file. If :code:`None`, mokapot will
        look for a column called "filename" (case insensitive). This is
        required for some output formats, such as FlashLFQ.
    ret_time : str, optional
        The column specifying the retention time in seconds. If :code:`None`,
        mokapot will look for a column called "ret_time" (case insensitive).
        This is required for some output formats, such as FlashLFQ.
    charge : str, optional
        The column specifying the charge state of each peptide. If
        :code:`None`, mokapot will look for a column called "charge" (case
        insensitive). This is required for some output formats, such as
        FlashLFQ.
    proteins : mokapot.Proteins, optional
        The proteins to use for protein-level confidence estimation. This
        may be created with :py:func:`mokapot.read_fasta()`.
    eval_fdr : float, optional
        The false discovery rate threshold for choosing the best feature and
        creating positive labels during the trainging procedure.
    subset: int, optional
        The maximum number of examples to use for training.
    rng : int or np.random.Generator, optional
        A seed or generator used for cross-validation split creation and to
        break ties, or :code:`None` to use the default random number
        generator state.
    strict_parsing : bool, optional
        If `True`, the fast, memory-efficient parser is replaced by a slower
        less efficient parser that correctly captures the PIN file protein
        column. This is generally not recommended.

    Returns
    -------
    PsmDataset
        The PSMs from all of the PIN files.

    """
    logging.info("Parsing PSMs...")
    data = build_df(pin_files, strict_parsing=strict_parsing)
    prot_col = find_column(None, data, "proteins", False)

    schema = PsmSchema(
        target=find_column(None, data, "label", True),
        spectrum=[
            find_column(None, data, "specid", True),
            find_column(None, data, "scannr", True),
        ],
        peptide=find_column(None, data, "peptide", True),
        file=find_column(file, data, "filename", False),
        expmass=find_column(expmass, data, "expmass", False),
        calcmass=find_column(calcmass, data, "calcmass", False),
        ret_time=find_column(ret_time, data, "ret_time", False),
        charge=find_column(charge, data, "charge", False),
        group=find_column(group, data, None, False),
        metadata=prot_col,
    )

    if schema.expmass is not None:
        schema.spectrum.append(schema.expmass)

    return PsmDataset(
        data=data,
        schema=schema,
        eval_fdr=eval_fdr,
        proteins=proteins,
        subset=subset,
        rng=rng,
    )


def percolator_to_df(perc_file: PathLike) -> pl.DataFrame:
    """Read a Percolator tab-delimited file.

    Percolator input format (PIN) files and the Percolator result files
    are tab-delimited, but also have a tab-delimited protein list as the
    final column. This function parses the file and returns a DataFrame.

    Parameters
    ----------
    perc_file : PathLike
        The file to parse.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of the parsed data.
    """
    LOGGER.info("Reading %s...", perc_file)
    if str(perc_file).endswith(".gz"):
        fopen = gzip.open
    else:
        fopen = open

    with fopen(perc_file) as perc:
        cols = perc.readline().rstrip().split("\t")
        dir_line = perc.readline().rstrip().split("\t")[0]
        if dir_line.lower() != "defaultdirection":
            perc.seek(0)
            _ = perc.readline()

        psms = pl.concat(_parse_in_chunks(perc, cols), how="vertical")

    return psms


def _parse_in_chunks(
    file_obj: IO, columns: list[str], chunk_size: int = int(1e8)
) -> pl.DataFrame:
    """
    Parse a file in chunks.

    Parameters
    ----------
    file_obj : file object
        The file to read lines from.
    columns : list of str
        The columns for each DataFrame.
    chunk_size : int
        The chunk size in bytes.

    Returns
    -------
    pandas.DataFrame
        The chunk of PSMs
    """
    while True:
        psms = file_obj.readlines(chunk_size)
        if not psms:
            break

        psms = [
            ",".join(p.rstrip().split("\t", len(columns) - 1)) for p in psms
        ]

        psms = [",".join(columns)] + psms

        yield pl.read_csv(StringIO("\n".join(psms)))


def find_column(
    col: str | None,
    df: pl.LazyFrame,
    default: str | None,
    required: bool,
) -> str:
    """Check that a column exists in the dataframe.

    Parameters
    ----------
    col : str
        The specified column that should be int he dataframe.
    df : polars.DataFrame
        The dataframe.
    default : str
        A case-insensitive fall-back option for the column.
    required : bool
        Is this column required?
    """
    if col is None:
        try:
            return [c for c in df.columns if c.lower() == default][0]
        except (IndexError, TypeError):
            if not required:
                return None

    if col not in df.columns and required:
        raise ValueError(f"The '{col}' column was not found.")

    return col


def build_df(
    pin_files: str | pl.DataFrame | Iterable[str | pl.DataFrame],
    strict_parsing: bool,
) -> pl.LazyFrame:
    """Build the PIN DataFrame.

    Parameters
    ----------
    pin_files : str, polars.DataFrame, iterable of str or polars.DataFrame
        One or more PIN files or data frames to be read.
    strict_parsing : bool
        Use our custom parser instead of the fast one build into polars.

    Returns
    -------
    polars.LazyFrame
        The parsed data.
    """
    dfs = []
    for pin_file in utils.listify(pin_files):
        # Figure out the type of the input
        try:
            pin_file = pl.from_pandas(pin_file)
        except TypeError:
            pass

        try:
            dfs.append(pin_file.lazy())
        except AttributeError:
            try:
                dfs.append(pl.scan_parquet(pin_file))
            except pl.ComputeError:
                if not strict_parsing:
                    df = pl.scan_csv(
                        pin_file,
                        separator="\t",
                        truncate_ragged_lines=True,
                    )
                else:
                    df = percolator_to_df(pin_file).lazy()

                dfs.append(df)

    # Verify columns are identical:
    first_cols = set(dfs[0].columns)
    for df in dfs:
        if set(df.columns) != first_cols:
            raise ValueError("All pin_files must have the same columns")

    return pl.concat(dfs, how="vertical")
