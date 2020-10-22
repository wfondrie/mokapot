"""
Match target peptides to plausible corresponding decoys
"""
from collections import defaultdict

import numpy as np
import pandas as pd
import numba as nb


def match_decoy(decoys, targets, ignore_mods=True):
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
    ignore_mods : bool
        Ignore modifications. Run much faster if True.

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

    # Build a map of composition to lists of peptides:
    targ_comps = residue_sort(targets, ignore_mods)

    # Find the first target peptide that matches the decoy composition
    decoy_map = {}
    decoy_comps = decoys.str.split("(?=[A-Z])").to_list()
    for decoy, comp in zip(decoys.to_list(), decoy_comps):
        try:
            decoy_map[decoy] = targ_comps["".join(sorted(comp))].pop()
        except IndexError:
            continue

    return decoy_map


def residue_sort(peptides, ignore_mods):
    """Sort peptide sequences by amino acid

    This function also considers potential modifications

    Parameters
    ----------
    peptides : pandas.Series
        A collection of peptides
    ignore_mods : bool
        Ignore modifications for the sake of speed.

    Returns
    -------
    pandas.DataFrame
        A lexographically sorted sequence that respects
        modifications.
    """
    if ignore_mods:
        compositions = peptides.to_list()
    else:
        compositions = peptides.str.split("(?=[A-Z])").to_list()

    comp_map = defaultdict(list)
    for pep, comp in zip(peptides.to_list(), compositions):
        comp_map[_sort(comp)].append(pep)

    return comp_map


# @nb.njit
def _sort(peptide):
    """Sort the residues of a peptide"""
    return "".join(sorted(peptide))
