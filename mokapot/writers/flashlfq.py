"""This module writes data in the generic format for FlashLFQ.

Details about the format can be found here:
https://github.com/smith-chem-wisc/FlashLFQ/wiki/Identification-Input-Formats#generic
"""

from pathlib import Path
import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


def to_flashlfq(conf, out_file="mokapot.flashlfq.txt"):
    """Save confidenct peptides for quantification with FlashLFQ.

    `FlashLFQ <https://github.com/smith-chem-wisc/FlashLFQ>`_ is an open-source
    tool for label-free quantification. For mokapot to save results in a
    compatible format, a few extra columns are required to be present, which
    specify the MS data file name, the theoretical peptide monoisotopic mass,
    the retention time, and the charge for each PSM. If these are not present,
    saving to the FlashLFQ format is disabled.

    Note that protein grouping in the FlashLFQ results will be more accurate if
    proteins were added for analysis with mokapot.

    Parameters
    ----------
    conf : Confidence object or tuple of Confidence objects
        One or more :py:class:`~mokapot.confidence.LinearConfidence` objects.
    out_file : str, optional
        The output file to write.

    Returns
    -------
    str
        The path to the saved file.

    """
    try:
        assert not isinstance(conf, str)
        iter(conf)
    except TypeError:
        conf = [conf]
    except AssertionError:
        raise ValueError("'conf' should be a Confidence object, not a string.")

    flashlfq = pd.concat([_format_flashlfq(c) for c in conf])
    flashlfq.to_csv(str(out_file), sep="\t", index=False)
    return out_file


def _format_flashlfq(conf):
    """Format peptides for quantification with FlashLFQ

    If proteins are provided, use the mokapot protein groups. Else,
    use the protein_column.

    Parameters
    ----------
    conf : a LinearConfidence object
        A :py:class:`~mokapot.confidence.LinearConfidence` object.

    Returns
    -------
    pandas.DataFrame
        The peptides in FlashLFQ format.
    """
    # Do some error checking for the required columns:
    required = ["filename", "calcmass", "rt", "charge"]

    opt_cols = {
        k: v
        for k, v in conf.get_optional_columns().as_dict().items()
        if v is not None
    }
    missing = [c for c in required if opt_cols[c] is None]
    if missing:
        missing = ", ".join([c + "_column" for c in missing])
        raise ValueError(
            "The following parameters must be specified when loading a "
            "collection of PSMs in order to save them in FlashLFQ format: "
            f"{missing}"
        )

    # TODO: make this work again ...
    # if conf._has_proteins:
    #     proteins = conf._proteins
    # elif conf._protein_column is not None:
    #     proteins = conf._protein_column
    # else:
    #     proteins = None
    proteins = None

    # Get parameters
    #### Start
    # TODO make this streaming for the future.
    # Create FlashLFQ dataframe

    # OLD: passing = peptides["mokapot q-value"] <= eval_fdr
    eval_fdr = conf.eval_fdr
    passing = pd.read_csv(conf.out_writers["peptides"][0].file_name, sep="\t")
    passing = passing[passing["mokapot_qvalue"] <= eval_fdr]

    cols_pull = opt_cols
    cols_pull["PSMId"] = conf.dataset.specId_column
    bar = conf.dataset.read_data(columns=list(cols_pull.values()))

    # Rename the columns rn it should have:
    # ['filename', 'scan', 'calcmass', 'expmass', 'rt', 'charge', 'PSMId']
    bar = bar.rename(columns={v: k for k, v in cols_pull.items()})

    # Join on the PSMId + col
    passing = passing.merge(
        bar,
        on="PSMId",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_right"),
    )

    ## Build the output file
    out_df = pd.DataFrame()
    out_df["File Name"] = passing["filename"].apply(lambda x: Path(x).name)

    try:
        seq = passing["peptide"]
    except KeyError:
        breakpoint()

    base_seq = (
        seq.str.replace(r"[\[\(].*?[\]\)]", "", regex=True)
        .str.replace(r"^.*?\.", "", regex=True)
        .str.replace(r"\..*?$", "", regex=True)
    )

    out_df["Base Sequence"] = base_seq
    out_df["Full Sequence"] = seq
    out_df["Peptide Monoisotopic Mass"] = passing["calcmass"]
    out_df["Scan Retention Time"] = passing["rt"]
    out_df["Precursor Charge"] = passing["charge"]

    if isinstance(proteins, str):
        # TODO: Add delimiter sniffing.
        prots = passing["proteinIds"].str.replace("\t", "; ", regex=False)
    elif proteins is None:
        prots = ""
    else:
        prots = base_seq.map(proteins.peptide_map.get)
        shared = pd.isna(prots)
        prots.loc[shared] = base_seq[shared].map(proteins.shared_peptides.get)

    out_df["Protein Accession"] = prots
    missing = pd.isna(out_df["Protein Accession"])
    num_missing = missing.sum()
    if num_missing:
        # TODO: revisit this warning ... it makes little sense to say
        # they wete not mapped if we are not really mapping them here ...
        # We could build an inverted index with a fasta file to do a
        # real mapping. OR just mention that they did not have an associated
        # ID for proteins.
        LOGGER.warning(
            "- Discarding %i peptides that could not be"
            " mapped to protein groups",
            num_missing,
        )
        out_df = out_df.loc[~missing, :]

    return out_df
