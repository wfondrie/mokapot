"""Write data in the generic format for FlashLFQ.

Details about the format can be found here:
https://github.com/smith-chem-wisc/FlashLFQ/wiki/Identification-Input-Formats#generic
"""
from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import polars as pl

if TYPE_CHECKING:
    from ..confidence import Confidence

LOGGER = logging.getLogger(__name__)


def to_flashlfq(
    conf: Confidence | Iterable[Confidence],
    out_file: PathLike = "mokapot.flashlfq.txt",
) -> Path:
    """Save confidenct peptides for quantification with FlashLFQ.

    `FlashLFQ <https://github.com/smith-chem-wisc/FlashLFQ>`_ is an open-source
    tool for label-free quantification. For mokapot to save results in a
    compatible format, a few extra columns are required to be present, which
    specify the MS data file name, the theoretical peptide monoisotopic mass,
    the retention time, and the charge for each PSM. If these are not present,
    saving to the FlashLFQ format is disabled.

    Furthermore, protein-level analysis must have been performed in mokapot.

    Parameters
    ----------
    conf : Confidence or iterable of Confidence objects
        The mokapot confidence estimates.
    out_file : PathLike, optional
        The output file to write.

    Returns
    -------
    Path
        The path to the saved file.

    """
    try:
        assert not isinstance(conf, str)
        iter(conf)
    except TypeError:
        conf = [conf]
    except AssertionError:
        raise ValueError("'conf' should be a Confidence object, not a string.")

    with pl.StringCache():
        (
            pl.concat([_format_flashlfq(c) for c in conf])
            .collect(streaming=True)
            .write_csv(str(out_file), separator="\t")
        )
    return Path(out_file)


def _format_flashlfq(conf: Confidence) -> pl.LazyFrame:
    """Format peptides for quantification with FlashLFQ.

    Parameters
    ----------
    conf : a Confidence object
        The confidence estiamtes

    Returns
    -------
    polars.LazyFrame
        The peptides in FlashLFQ format.
    """
    # Do some error checking for the required columns:
    required = ["file", "calcmass", "ret_time", "charge"]
    missing = [c for c in required if getattr(conf.schema, c) is None]
    if missing:
        missing = ", ".join(list(missing))
        raise ValueError(
            "The following additional schema parameters must be specified for "
            "a collection of PSMs in order to save them in FlashLFQ format: "
            f"{', '.join(missing)}"
        )

    if conf.proteins is None:
        raise ValueError(
            "Protein-level confidence estimates are required for FlashLFQ "
            "export."
        )

    data = (
        conf.results.peptides.filter(
            pl.col("mokapot q-value") <= conf.eval_fdr
        )
        .with_columns(
            [
                # The filename:
                pl.col(conf.schema.file).map_elements(
                    lambda x: Path(x).name, pl.Utf8
                ),
                # The stripped sequence:
                pl.col(conf.schema.peptide)
                .str.replace(r"[\[\(].*?[\]\)]", "")
                .str.replace(r"^.*?\.", "")
                .str.replace(r"\..*?$", "")
                .alias("Base Sequence"),
                # RT in minutes:
                pl.col(conf.schema.ret_time) / 60,
            ]
        )
        .rename(
            {
                conf.schema.file: "File Name",
                conf.schema.peptide: "Full Sequence",
                conf.schema.calcmass: "Peptide Monoisotopic Mass",
                conf.schema.ret_time: "Scan Retention Time",
                conf.schema.charge: "Precursor Charge",
                "mokapot protein group": "Protein Accession",
            }
        )
        .select(
            [
                "File Name",
                "Base Sequence",
                "Full Sequence",
                "Peptide Monoisotopic Mass",
                "Scan Retention Time",
                "Precursor Charge",
                "Protein Accession",
            ]
        )
    )

    return data
