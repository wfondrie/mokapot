"""
Match target peptides to plausible corresponding decoys
"""
import numpy as np
import pandas as pd


def match_decoy(decoys, targets):
    """Find a corresponding target for each decoy.

    Matches a decoy to a unique random target peptide that
    has the same amino acid composition, including modifications.
    If none can be found, an :code:`nan` is returned for that
    peptide.

    Parameters
    ----------
    decoys : pandas.Series
        A collection of decoy peptides
    targets : pandas.Series
        A collection of target peptides

    Returns
    -------
    pandas.Series
        The corresponding target peptide for each
        decoy peptide.
    """
    targets.name = "target"
    decoys.name = "decoy"

    # Note we need to maintain the order of decoys, but not
    # the order of targets.
    targets = targets.sample(frac=1).reset_index(drop=True)
    targets = residue_sort(targets)
    decoys = residue_sort(decoys)
    decoys = pd.merge(decoys, targets, how="left").set_index("decoy")
    return decoys["target"].to_dict()


def residue_sort(peptides):
    """Sort peptide sequences by amino acid

    This function also considers potential modifications

    Parameters
    ----------
    peptides : pandas.Series
        A collection of peptides

    Returns
    -------
    pandas.DataFrame
        A lexographically sorted sequence that respects
        modifications.
    """
    comp = peptides.str.split("(?=[A-Z])").apply(lambda x: "".join(sorted(x)))
    peptides = peptides.to_frame()
    peptides["comp"] = comp
    peptides["n"] = peptides.groupby("comp").transform(
        lambda x: np.arange(x.size)
    )
    return peptides
