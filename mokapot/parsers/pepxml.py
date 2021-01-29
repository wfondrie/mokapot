"""
This module contains a parser for PepXML files.
"""
import logging
import itertools
from functools import partial

import numpy as np
import pandas as pd
from lxml import etree

from .. import utils
from ..dataset import LinearPsmDataset

LOGGER = logging.getLogger(__name__)


# Functions -------------------------------------------------------------------
def read_pepxml(
    pepxml_files, decoy_prefix="decoy_", to_df=False, exclude_features=None
):
    """Read PepXML files.

    Read peptide-spectrum matches (PSMs) from one or more pepxml files,
    aggregating them into a single
    :py:class:`~mokapot.dataset.LinearPsmDataset`.

    Specifically, mokapot will extract the search engine scores as a set of
    features (found under the :code:`search_scores` tag). Additionally, mokapot
    will add the peptide lengths, mass error, the number of enzymatic termini
    and the number of missed cleavages are added as features.

    Parameters
    ----------
    pepxml_files : str or tuple of str
        One or more PepXML files to read.
    decoy_prefix : str, optional
        The prefix used to indicate a decoy protein in the description lines of
        the FASTA file.
    exclude_features : str or tuple of str, optional
        One or more features to exclude from the dataset. This is useful in the
        case that a search engine score may be biased again decoy PSMs/CSMs.
    to_df : bool, optional
        Return a :py:class:`pandas.DataFrame` instead of a
        :py:class:`~mokapot.dataset.LinearPsmDataset`.

    Returns
    -------
    LinearPsmDataset or pandas.DataFrame
        A :py:class:`~mokapot.dataset.LinearPsmDataset` or
        :py:class:`pandas.DataFrame` containing the parsed PSMs.
    """
    proton = 1.00727646677
    pepxml_files = utils.tuplize(pepxml_files)
    psms = pd.concat([_parse_pepxml(f, decoy_prefix) for f in pepxml_files])

    # Calculate massdiff features
    psms["exp_mass"] = psms["exp_mass"] / psms["charge"] + proton
    psms["calc_mass"] = psms["calc_mass"] / psms["charge"] + proton
    psms["mass_diff"] = psms["exp_mass"] - psms["calc_mass"]
    psms["abs_mass_diff"] = psms["mass_diff"].abs()

    # Log number of candidates:
    psms["num_matched_peptides"] = np.log10(psms["num_matched_peptides"])

    # Create charge columns:
    psms = pd.concat(
        [psms, pd.get_dummies(psms["charge"], prefix="charge")], axis=1
    )

    psms = psms.drop("charge", axis=1)

    # -log10 p-values
    nonfeat_cols = [
        "spectrum_id",
        "label",
        "exp_mass",
        "calc_mass",
        "peptide",
        "proteins",
    ]

    if exclude_features is not None:
        exclude_features = list(utils.tuplize(exclude_features))
    else:
        exclude_features = []

    nonfeat_cols += exclude_features
    feat_cols = [c for c in psms.columns if c not in nonfeat_cols]
    psms = psms.apply(_log_pvalues, features=feat_cols)

    if to_df:
        return psms

    dset = LinearPsmDataset(
        psms=psms,
        target_column="label",
        spectrum_columns=("spectrum_id",),
        peptide_column="peptide",
        protein_column="proteins",
        feature_columns=feat_cols,
        copy_data=False,
    )

    return dset


def _parse_pepxml(pepxml_file, decoy_prefix):
    """Parse the PSMs of a PepXML into a DataFrame

    Parameters
    ----------
    pepxml_file : str
        The PepXML file to parse.
    decoy_prefix : str
        The prefix used to indicate a decoy protein in the description lines of
        the FASTA file.

    Returns
    -------
    pandas.DataFrame
        A :py:class:`pandas.DataFrame` containing the information about each
        PSM.
    """

    parser = etree.iterparse(str(pepxml_file), tag="{*}spectrum_query")
    parse_fun = partial(_parse_spectrum, decoy_prefix=decoy_prefix)
    psms = map(parse_fun, parser)
    return pd.DataFrame.from_records(itertools.chain.from_iterable(psms))


def _parse_spectrum(spectrum, decoy_prefix):
    """Parse the PSMs for a single mass spectrum

    Parameters
    ----------
    spectrum: tuple of anything, lxml.etree.Element
        The second element of the tuple should be the XML element for a single
        spectrum. The first is not used, but is necessary for compatibility with
        using :code:`map()`.
    decoy_prefix : str
        The prefix used to indicate a decoy protein in the description lines of
        the FASTA file.

    Yields
    ------
    dict
        A dictionary describing all of the PSMs for a spectrum.
    """
    spectrum = spectrum[1]
    spec_info = {
        "spectrum_id": spectrum.get("spectrum"),
        "charge": int(spectrum.get("assumed_charge")),
    }

    spec_info["exp_mass"] = float(spectrum.get("precursor_neutral_mass"))
    for psms in spectrum.iter("{*}search_result"):
        for psm in psms.iter("{*}search_hit"):
            yield _parse_psm(psm, spec_info, decoy_prefix=decoy_prefix)


def _parse_psm(psm_info, spec_info, decoy_prefix):
    """Parse a single PSM

    Parameters
    ----------
    psm_info : lxml.etree.Element
        The XML element containing information about the PSM.
    spec_info : dict
        The parsed spectrum data.
    decoy_prefix : str
        The prefix used to indicate a decoy protein in the description lines of
        the FASTA file.

    Returns
    -------
    dict
        A dictionary containing parsed data about the PSM.
    """
    psm = spec_info.copy()
    psm["calc_mass"] = float(psm_info.get("calc_neutral_pep_mass"))
    psm["peptide"] = psm_info.get("peptide")
    psm["proteins"] = [psm_info.get("protein")]
    psm["label"] = not psm["proteins"][0].startswith(decoy_prefix)

    # Begin features:
    psm["missed_cleavages"] = int(psm_info.get("num_missed_cleavages"))
    psm["ntt"] = int(psm_info.get("num_tol_term"))
    psm["num_matched_peptides"] = int(psm_info.get("num_matched_peptides"))

    queries = [
        "{*}modification_info",
        "{*}search_score",
        "{*}alternative_protein",
    ]
    for element in psm_info.iter(*queries):
        if "modification_info" in element.tag:
            psm["peptide"] = element.get("modified_peptide")
        elif "alternative_protein" in element.tag:
            psm["proteins"].append(element.get("protein"))
            if not psm["label"]:
                psm["label"] = not psm["proteins"][-1].startswith(decoy_prefix)
        else:
            psm[element.get("name")] = float(element.get("value"))

    psm["proteins"] = "\t".join(psm["proteins"])
    return psm


def _log_pvalues(col, features):
    """Log-transform columns that are p-values.

    This function tries to detect feature columns that are p-values using a
    simple heuristic. If the column is a p-value, then it returns the -log (base
    10) of the column.

    Parameters:
    -----------
    col : pandas.Series
        A column of the dataset.
    features: list of str
        The features of the dataset. Only feature columns will be considered
        for transformation.

    Returns
    -------
    pandas.Series
        The log-transformed values of the column if the feature was determined
        to be a p-value.
    """
    if col.name not in features:
        return col

    # A simple heuristic to find p-value features:
    # p-values are between 0 and 1
    if col.max() <= 1 and col.min() >= 0:
        # Make sure this isn't a binary column:
        if ((col < 1) & (col > 0)).any():
            # Only log if values span >3 orders of magnitude:
            col[col == 0] = np.finfo(col.dtype).tiny
            if col.max() / col.min() >= 100:
                return -np.log10(col)

    return col
