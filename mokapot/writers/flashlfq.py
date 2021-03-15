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
    proteins,
    out_file,
    filename_column,
    peptide_column,
    mass_column,
    rt_column,
    charge_column,
    eval_fdr,
):
    """Write results for quantification with FlashLFQ

    If proteins are provided, use the mokapot protein groups. Else,
    use the protein_column.

    Parameters
    ----------
    peptides : pandas.DataFrame
        The peptides from a LinearConfidence object.
    proteins : str or FastaProteins object, optional
        If a string, the column specifying the proteins for a peptide.
        If a FastaProteins object, the parsed protein groups will be used for
        the assignment.
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

    if isinstance(proteins, str) or proteins is None:
        # TODO: Add delimiter sniffing.
        print("STRING!")
        prots = peptides.loc[passing, proteins].str.replace(
            "\t", "; ", regex=False
        )
    elif proteins is None:
        proteins = ""
    else:
        prots = base_seq.map(proteins.peptide_map.get)
        shared = pd.isna(prots)
        prots.loc[shared] = base_seq[shared].map(proteins.shared_peptides.get)

    out_df["Protein Accession"] = prots
    missing = pd.isna(out_df["Protein Accession"])
    num_missing = missing.sum()
    if num_missing:
        LOGGER.warn(
            "- Discarding %i peptides that could not be mapped to protein "
            "groups",
            num_missing,
        )
        out_df = out_df.loc[~missing, :]

    out_df.to_csv(out_file, sep="\t", index=False)
    return out_file
