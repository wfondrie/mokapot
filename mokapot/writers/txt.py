"""Writer to save results in a tab-delmited format"""
from pathlib import Path
from collections import defaultdict

import pandas as pd


def to_txt(conf, dest_dir=None, file_root=None, sep="\t", decoys=False):
    """Save confidence estimates to delimited text files.

    Write the confidence estimates for each of the available levels
    (i.e. PSMs, peptides, proteins) to separate flat text files using the
    specified delimiter. If more than one collection of confidence estimates
    is provided, they will be combined, yielding a single file for each level
    specified by either dataset.

    Parameters
    ----------
    conf : Confidence object or tuple of Confidence objects
        One or more :py:class:`~mokapot.confidence.LinearConfidence` objects.
    dest_dir : str or None, optional
        The directory in which to save the files. :code:`None` will use the
        current working directory.
    file_root : str or None, optional
        An optional prefix for the confidence estimate files. The suffix will
        always be "mokapot.{level}.txt" where "{level}" indicates the level at
        which confidence estimation was performed (i.e. PSMs, peptides,
        proteins).
    sep : str, optional
        The delimiter to use.
    decoys : bool, optional
        Save decoys confidence estimates as well?

    Returns
    -------
    list of str
        The paths to the saved files.

    """
    try:
        assert not isinstance(conf, str)
        iter(conf)
    except TypeError:
        conf = [conf]
    except AssertionError:
        raise ValueError("'conf' should be a Confidence object, not a string.")

    file_base = "mokapot"
    if file_root is not None:
        file_base = file_root + "." + file_base
    if dest_dir is not None:
        file_base = Path(dest_dir, file_base)

    results = defaultdict(list)
    for res in conf:
        for level, qval_list in _get_level_data(res, decoys).items():
            results[level] += qval_list

    out_files = []
    for level, qval_list in results.items():
        out_file = str(file_base) + f".{level}.txt"
        pd.concat(qval_list).to_csv(out_file, sep=sep, index=False)
        out_files.append(out_file)

    return out_files


def _get_level_data(conf, decoys):
    """Return the dataframes for each level.

    Parameters
    ----------
    conf : a Confidence object
        A LinearConfidence object.
    decoys : bool
        Should decoys be included?

    Returns
    -------
    Dict
        Each entry contains a level, dataframe pair.
    """
    results = defaultdict(list)
    for level, qvals in conf.confidence_estimates.items():
        if qvals is None:
            continue

        results[level].append(qvals)

    if decoys:
        for level, qvals in conf.decoy_confidence_estimates.items():
            if qvals is None:
                continue

            results[f"decoy.{level}"].append(qvals)

    return results
