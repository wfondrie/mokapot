from dataclasses import dataclass

import numpy as np
import pandas as pd

from mokapot.column_defs import ColumnGroups, OptionalColumns
from mokapot.utils import tuplize


@dataclass
class MockPsmDataframe:
    df: pd.DataFrame
    columns: ColumnGroups
    rng: np.random.Generator
    score_cols: list[str]
    fasta_string: str


def _psm_df_rand(
    ntargets: int,
    ndecoys: int,
    pct_real: float = 0.4,
    score_diffs: list[float] = [3.0],
    share_ids: bool = False,
) -> MockPsmDataframe:
    """A DataFrame with 100 PSMs."""
    rng = np.random.Generator(np.random.PCG64(42))
    score_cols = ["score" + str(i) for i in range(len(score_diffs))]
    nreal = int(ntargets * pct_real)
    nonreal = ntargets - nreal
    max_scan = ntargets + ndecoys
    targets = {
        "PSMId": np.arange(ntargets),
        "specid": np.arange(ntargets),
        "target": [True] * ntargets,
        "scannr": rng.integers(0, max_scan, ntargets),
        "calcmass": rng.uniform(500, 2000, size=ntargets),
        "expmass": rng.uniform(500, 2000, size=ntargets),
        "peptide": [_random_peptide(5, rng) for _ in range(ntargets)],
        "proteins": ["_dummy" for _ in range(ntargets)],
        "filename": "test.mzML",
        "ret_time": rng.uniform(0, 60 * 120, size=ntargets),
        "charge": rng.choice([2, 3, 4], size=ntargets),
    }
    targets = pd.DataFrame(targets)
    for n, d in zip(score_cols, score_diffs):
        targets[n] = np.concatenate([
            rng.normal(d, size=nreal),
            rng.normal(size=nonreal),
        ])

    decoys = {
        "PSMId": np.arange(ntargets, ntargets + ndecoys),
        "specid": np.arange(ntargets, ntargets + ndecoys),
        "target": [False] * ndecoys,
        "scannr": rng.integers(0, max_scan, ndecoys),
        "calcmass": rng.uniform(500, 2000, size=ndecoys),
        "expmass": rng.uniform(500, 2000, size=ndecoys),
        "peptide": [_random_peptide(5, rng) for _ in range(ndecoys)],
        "proteins": ["_dummy" for _ in range(ndecoys)],
        "filename": "test.mzML",
        "ret_time": rng.uniform(0, 60 * 120, size=ndecoys),
        "charge": rng.choice([2, 3, 4], size=ndecoys),
    }
    decoys = pd.DataFrame(decoys)
    for n, d in zip(score_cols, score_diffs):
        decoys[n] = rng.normal(size=len(decoys))

    if share_ids:
        assert len(targets) == len(decoys), "Sharing ids requires equal number"
        # targets["specid"] = decoys["specid"] Spec ID has to be different now.
        targets["spectrum"] = np.arange(ntargets)
        decoys["spectrum"] = np.arange(ndecoys)
        decoys["scannr"] = targets["scannr"]
        spec_cols = ["spectrum", "scannr", "specid"]
    else:
        spec_cols = ["scannr", "specid"]

    df = pd.concat([targets, decoys])
    target_peptides = targets["peptide"].to_list()
    decoy_peptides = decoys["peptide"].to_list()
    fasta_data = "\n".join(
        _make_fasta(100, target_peptides, 10, rng)
        + _make_fasta(100, decoy_peptides, 10, rng, "decoy")
    )
    assert len(df) == (ntargets + ndecoys)
    columns = ColumnGroups(
        columns=tuplize(df.columns),
        target_column="target",
        peptide_column="peptide",
        spectrum_columns=tuplize(spec_cols),
        feature_columns=tuplize(score_cols),
        extra_confidence_level_columns=tuplize([]),
        optional_columns=OptionalColumns(
            id="PSMId",
            filename="filename",
            scan="scannr",
            calcmass="calcmass",
            expmass="expmass",
            rt="ret_time",
            charge="charge",
            protein="proteins",
        ),
    )
    return MockPsmDataframe(
        df=df,
        columns=columns,
        rng=rng,
        score_cols=score_cols,
        fasta_string=fasta_data,
    )


def _make_fasta(
    num_proteins, peptides, peptides_per_protein, random_state, prefix=""
):
    """Create a FASTA string from a set of peptides

    Parameters
    ----------
    num_proteins : int
        The number of proteins to generate.
    peptides : list of str
        A list of peptide sequences.
    peptides_per_protein: int
        The number of peptides per protein.
    random_state : numpy.random.Generator object
        The random state.
    prefix : str
        The prefix, if generating decoys

    Returns
    -------
    list of str
        A list of lines in a FASTA file.
    """
    lines = []
    for protein in range(num_proteins):
        lines.append(f">{prefix}sp|test|test_{protein}")
        lines.append(
            "".join(list(random_state.choice(peptides, peptides_per_protein)))
        )

    return lines


def _random_peptide(length, random_state):
    """Generate a random peptide"""
    return "".join(
        list(random_state.choice(list("ACDEFGHILMNPQSTVWY"), length - 1))
        + ["K"]
    )
