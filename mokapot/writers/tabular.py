"""Writer to save results in a tab-delmited format."""
from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from ..confidence import Confidence

LOGGER = logging.getLogger(__name__)


def to_parquet(
    conf: Confidence,
    *,
    dest_dir: PathLike | None = None,
    stem: str | None = None,
    decoys: bool = False,
    ext: str = "parquet",
    **kwargs: dict,
) -> list[Path, ...]:
    """Save confidence estimates to Apache Parquet files.

    Write the confidence estimates for each of the available levels
    (i.e. PSMs, peptides, proteins) to a Parquet file. Apache Parquet
    is a popular and effecient columnar data format and core part of
    modern data infrastructure.

    If deciding between Parquet or a text format, we recommend Parquet.

    Parameters
    ----------
    conf : mokapot.Confidence
        The mokapot confidence estimates.
    dest_dir : PathLike or None, optional
        The directory in which to save the files. :code:`None` will use the
        current working directory.
    stem : str or None, optional
        An optional prefix for the confidence estimate files. The suffix will
        always be "mokapot.{level}.{ext}" where "{level}" indicates the level
        at which confidence estimation was performed (i.e. PSMs, peptides,
        proteins).
    decoys : bool, optional
        Save decoys confidence estimates as well?
    ext : str, optional
        The extention to use when saving the files.
    **kwargs : dict
        Keyword arguments passed to
        :py:method:`polars.LazyFrame.sink_parquet()`.

    Returns
    -------
    list of Path
        The paths to the saved files.

    """
    with pl.StringCache():
        return _to_tabular(
            conf=conf,
            dest_dir=dest_dir,
            stem=stem,
            decoys=decoys,
            ext=ext,
            write_fn="write_parquet",
            **kwargs,
        )


def to_txt(
    conf: Confidence,
    *,
    dest_dir: PathLike | None = None,
    stem: str | None = None,
    separator: str = "\t",
    decoys: bool = False,
    ext: str = "txt",
) -> list[Path, ...]:
    """Save confidence estimates to delimited text files.

    Write the confidence estimates for each of the available levels
    (i.e. PSMs, peptides, proteins) to separate flat text files using the
    specified delimiter.

    Parameters
    ----------
    conf : mokapot.Confidence
        The mokapot confidence estimates.
    dest_dir : PathLike or None, optional
        The directory in which to save the files. :code:`None` will use the
        current working directory.
    stem : str or None, optional
        An optional prefix for the confidence estimate files. The suffix will
        always be "mokapot.{level}.{ext}" where "{level}" indicates the level
        at which confidence estimation was performed (i.e. PSMs, peptides,
        proteins).
    separator : str, optional
        The delimiter to use.
    decoys : bool, optional
        Save decoys confidence estimates as well?
    ext : str, optional
        The extention to use when saving the files.

    Returns
    -------
    list of Path
        The paths to the saved files.

    """
    with pl.StringCache():
        return _to_tabular(
            conf=conf,
            dest_dir=dest_dir,
            stem=stem,
            decoys=decoys,
            ext=ext,
            write_fn="write_csv",
            separator=separator,
        )


def to_csv(
    conf: Confidence,
    *,
    dest_dir: PathLike | None = None,
    stem: str | None = None,
    decoys: bool = False,
    ext: str = "csv",
    **kwargs: dict,
) -> list[Path, ...]:
    """Save confidence estimates to comma-separated value files.

    Write the confidence estimates for each of the available levels
    (i.e. PSMs, peptides, proteins) to CSV files using the
    specified delimiter.

    Parameters
    ----------
    conf : mokapot.Confidence
        The mokapot confidence estimates.
    dest_dir : PathLike or None, optional
        The directory in which to save the files. :code:`None` will use the
        current working directory.
    stem : str or None, optional
        An optional prefix for the confidence estimate files. The suffix will
        always be "mokapot.{level}.csv" where "{level}" indicates the level at
        which confidence estimation was performed (i.e. PSMs, peptides,
        proteins).
    decoys : bool, optional
        Save decoys confidence estimates as well?
    ext : str, optional
        The extention to use when saving the files.
    **kwargs : dict
        Keyword arguments passed to :py:method:`polars.LazyFrame.sink_csv()`.

    Returns
    -------
    list of Path
        The paths to the saved files.
    """
    with pl.StringCache():
        return _to_tabular(
            conf=conf,
            dest_dir=dest_dir,
            stem=stem,
            decoys=decoys,
            ext=ext,
            write_fn="write_csv",
            **kwargs,
        )


def _to_tabular(
    conf: Confidence,
    dest_dir: PathLike | None,
    stem: str | None,
    decoys: bool,
    ext: str,
    write_fn: str,
    **kwargs: dict,
) -> list[Path, ...]:
    """Save confidence estimates to tabular files.

    Use this helper to save the tabular data to files using
    the specified method or function.

    Parameters
    ----------
    conf : mokapot.Confidence
        The mokapot confidence estimates.
    dest_dir : PathLike or None, optional
        The directory in which to save the files. :code:`None` will use the
        current working directory.
    stem : str or None, optional
        An optional prefix for the confidence estimate files. The suffix will
        always be "mokapot.{level}.{ext}" where "{level}" indicates the level
        at which confidence estimation was performed (i.e. PSMs, peptides,
        proteins).
    decoys : bool, optional
        Save decoys confidence estimates as well?
    ext : str, optional
        The extention to use when saving the files.
    write_fn : str
        The function to use for saving. This is looked up as
        a polars DataFrame method.
    **kwargs : dict
        Arguments to pass to the write_fn

    Returns
    -------
    list of Path
        The paths to the saved files.

    """
    logging.info("Writing confidence estimates...")
    msg = "  - Wrote %s"
    if stem is None or not stem:
        stem = []
    else:
        stem = [stem]

    dest_dir = "" if dest_dir is None else dest_dir
    out_files = []
    for level, table in conf.results:
        fname = Path(dest_dir, ".".join(stem + ["mokapot", level.name, ext]))
        getattr(table.collect(streaming=True), write_fn)(fname, **kwargs)
        LOGGER.info(msg, fname)
        out_files.append(fname)

    if decoys:
        for level, table in conf.decoy_results:
            fname = Path(
                dest_dir,
                ".".join(stem + ["mokapot", "decoy", level.name, ext]),
            )
            getattr(table.collect(streaming=True), write_fn)(fname, **kwargs)
            LOGGER.info(msg, fname)
            out_files.append(fname)

    return out_files
