"""This module contains the code to write mzTab files.

It is based off of Wout Bittremieux"s code in ANN-SoLo:
https://github.com/bittremieux/ANN-SoLo/blob/master/src/ann_solo/writer.py
"""
import re
import logging
from pathlib import Path

from .. import __version__

LOGGER = logging.getLogger(__name__)


def write_mztab(conf, out_file):
    """Write the PSMs, peptides, and Proteins to an mzTab file.

    Parameters
    ----------
    conf : a LinearConfidence object
        The PSMs, peptides, and Proteins to be saved.
    out_file : str
        The output file name.

    Returns
    -------
    str
        The file name of the mzTab output file.
    """
    LOGGER.info("Writing results to %s", out_file)

    runs = {}
    fname = conf._optional_columns["filename"]
    for idx, run in enumerate(conf.psms[fname]):
        runs[f"ms_run[{idx}]"] = run

    # The metadata
    metadata = [
        ("mzTab-version", "1.0.0"),
        ("mzTab-mode", "Summary"),
        ("mzTab-type", "Identification"),
        ("description", "Spectrum identification results from mokapot"),
        ("psm_search_engine_score[1]", ""),
        ("psm_search_engine_score[2]", ""),
        ("peptide_search_engine_score[1]", ""),
        ("peptide_search_engine_score[2]", ""),
        ("fixed_mod", "null"),
        ("variable_mod", "null"),
    ]

    if conf._has_proteins:
        metadata += [
            ("protein_search_engine_score[1]", ""),  # q-value
            ("protein_search_engine_score[2]", ""),  # PEP
        ]


def _format_psms(conf, runs):
    """Return the formatted text for PSMs"""
    header = "\t".join(
        [
            "PSH",
            "sequence",
            "PSM_ID",
            "accession",
            "unique",
            "database",
            "database_version",
            "search_engine",
            "search_engine_score[1]",
            "search_engine_score[2]",
            "modifications",
            "retention_time",
            "charge",
            "exp_mass_to_charge",
            "calc_mass_to_charge",
            "spectra_ref",
            "pre",
            "post",
            "start",
            "end",
        ]
    )

    proton = 1.00727646677
    psms = []
    rt = conf._optional_columns["rt"]
    charge = conf._optional_columns["charge"]
    exp = conf._optional_columns["expmass"]
    calc = conf._optional_column["calcmass"]
    scan = conf._optional_column["scan"]
    fname = conf._optional_column["filename"]
    for idx, psm in conf.psms.iterrows():
        psms.append(
            "\t".join(
                [
                    "PSM",
                    psm[conf._peptide_column],
                    "_".join(psm[conf._psm_columns]),
                    "null",
                    "null",
                    "null",
                    "null",
                    "mokapot",  # mokapot
                    str(psm["mokapot q-value"]),
                    str(psm["mokapot PEP"]),
                    "null",
                    str(psm[rt]),
                    str(psm[charge]),
                    str(psm[exp] / psm[charge] + proton),
                    str(psm[calc] / psm[charge] + proton),
                    f"ms_run[{runs[psm[fname]]}]:scan={psm[scan]}",
                    "null",
                    "null",
                    "null",
                    "null",
                ]
            )
        )
