"""This module writes data in the generic format for FlashLFQ.

Details about the format can be found here:
https://github.com/smith-chem-wisc/FlashLFQ/wiki/Identification-Input-Formats#generic
"""
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def write_flashlfq(
    peptides,
    out_file,
    filename_column,
    peptide_column,
    mass_column,
    rt_column,
    charge_column,
    protein_column,
    eval_fdr,
):
    """Write results for quantification with FlashLFQ

    Parameters
    ----------
    peptides : pandas.DataFrame
        The peptides from a LinearConfidence object.
    out_file : str
        The output file.
    filename_column : str
        The column specifying the MS data file.
    peptide_column : str
        The column specifying the peptide sequence
    mass_column: str
        The column specifying the theoretical monoisotopic mass of the peptide
        including modifications.
    rt_column : str
        The column specifying the retention time in seconds.
    charge_column : str
        The column specifying the charge state of each peptide.
    protein_column : str
        The column specifying proteins for the peptide.
    eval_fdr : str
        The FDR threshold to apply to the data.

    Returns
    -------
    str
        The output filename.
    """
    passing = peptides["mokapot q-value"] <= eval_fdr

    out_df = pd.DataFrame()
    out_df["File Name"] = peptides.loc[passing, filename_column]

    seq = peptides.loc[passing, peptide_column]
    base_seq = (
        seq.str.replace(r"[\[\(].*?[\]\)]", "", regex=True)
        .str.replace(r"^.*?\.", "", regex=True)
        .str.replace(r"\..*?$", "", regex=True)
    )

    out_df["Base Sequence"] = base_seq
    out_df["Full Sequence"] = seq
    out_df["Peptide Monoisotopic Mass"] = peptides.loc[passing, mass_column]
    out_df["Scan Retention Time"] = peptides.loc[passing, rt_column] / 60
    out_df["Precursor Charge"] = peptides.loc[passing, charge_column]

    if protein_column is not None:
        proteins = (
            peptides.loc[passing, protein_column]
            .str.replace("|", "-", regex=False)
            .str.replace("\t", "|", regex=False)
        )
    else:
        proteins = ""

    out_df["Protein Accession"] = proteins
    out_df.to_csv(out_file, sep="\t", index=False)
    return out_file
