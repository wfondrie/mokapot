from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator, Iterable

import numpy as np
import pandas as pd
from typeguard import typechecked

from .tabular_data import TabularDataWriter, SqliteWriter


@typechecked
class ConfidenceSqliteWriter(SqliteWriter):
    def __init__(
        self,
        database: str | Path | sqlite3.Connection,
        columns: list[str],
        column_types: list | None = None,
        level: str = "psms",
        qvalue_column: str = "q_value",
        pep_column: str = "posterior_error_prob",
    ) -> None:
        super().__init__(database, columns, column_types)
        self.level_cols = {
            "precursors": ["PRECURSOR_VALIDATION", "PCM_ID", "Precursor"],
            "modifiedpeptides": [
                "MODIFIED_PEPTIDE_VALIDATION",
                "MODIFIED_PEPTIDE_ID",
                "ModifiedPeptide",
            ],
            "peptides": ["PEPTIDE_VALIDATION", "PEPTIDE_ID", "peptide"],
            "peptidegroups": [
                "PEPTIDE_GROUP_VALIDATION",
                "PEPTIDE_GROUP_ID",
                "PeptideGroup",
            ],
        }
        self.level = level
        self.qvalue_column = qvalue_column
        self.pep_column = pep_column

    def get_query(self, level, qvalue_column, pep_column):
        if level == "psms":
            query = f"UPDATE CANDIDATE SET PSM_FDR = :{qvalue_column}, SVM_SCORE = :score, POSTERIOR_ERROR_PROBABILITY = :{pep_column} WHERE CANDIDATE_ID = :PSMId;"  # noqa: E501
        else:
            table_name, table_id_col, mokapot_id_col = self.level_cols[level]
            query = f"INSERT INTO {table_name}({table_id_col},FDR,PEP,SVM_SCORE) VALUES(:{mokapot_id_col},:{qvalue_column},:{pep_column},:score)"  # noqa: E501
        return query

    def append_data(self, data):
        query = self.get_query(self.level, self.qvalue_column, self.pep_column)
        # todo: what about using connection.executemany()? Should be faster...
        data = data.to_dict("records")
        for row in data:
            self.connection.execute(query, row)


@typechecked
def write_confidences(
    data_iterator: Iterator[pd.DataFrame],
    q_value_iterator: Iterable[np.array],
    pep_iterator: Iterable[np.array],
    target_iterator: Iterable[np.array],
    out_paths: list[Path],
    decoys: bool,
    level: str,
    out_columns: list[str],
    qvalue_column: str = "q_value",
    pep_column: str = "posterior_error_prob",
) -> None:
    """
    Write confidences for given rollup level to output files.
    Note, that the iterators all need to yield the same number of chunks, each
    one having the same size/length as the others.

    Parameters
    ----------
    data_iterator : Iterator[pd.DataFrame]
        An iterator that yields chunks of data as pandas DataFrames.
    q_value_iterator : Iterable[np.array]
        A iterator that yields numpy arrays containing the q-values for each
        data chunk.
    pep_iterator : Iterable[np.array]
        A iterator that yields numpy arrays containing the posterior error
        probabilities for each data chunk.
    target_iterator : Iterable[np.array]
        A iterator that yields numpy arrays indicating whether each data point
        is a target or decoy for each data chunk.
    out_paths : list[Path]
        A list of output file paths where the confidence data will be written.
        The first element contains the path for the targets and the second
        those for the decoys.
    decoys : bool
        A boolean flag indicating whether to include decoy data in the output.
    level : str
        The rollup level (psms, percursors, peptides, etc.)
    out_columns : list[str]
        A list of column names to include in the output.
    qvalue_column : str, optional
        The name of the column to store the q-values. Default is 'q_value'.
    pep_column : str, optional
        The name of the column to store the posterior error probabilities.
        Default is 'posterior_error_prob'.

    Returns
    -------
    None

    """
    if not decoys and len(out_paths) > 1:
        out_paths.pop(1)
    is_sqlite = True if out_paths[0].suffix == ".db" else False

    # Create the writers
    if is_sqlite:

        def create_writer(path):
            return ConfidenceSqliteWriter(
                path,
                out_columns,
                level=level,
                qvalue_column=qvalue_column,
                pep_column=pep_column,
            )
    else:

        def create_writer(path):
            return TabularDataWriter.from_suffix(path, out_columns)

    writers = [create_writer(path) for path in out_paths]
    # for writer in writers:
    #     writer.initialize()

    # Now write the confidence data
    for data_chunk, qvals_chunk, peps_chunk, targets_chunk in zip(
        data_iterator, q_value_iterator, pep_iterator, target_iterator
    ):
        data_chunk[qvalue_column] = qvals_chunk
        data_chunk[pep_column] = peps_chunk
        data_out = []
        if not is_sqlite:
            data_out.append(data_chunk.loc[targets_chunk, out_columns])
            if decoys:
                data_out.append(data_chunk.loc[~targets_chunk, out_columns])
        else:
            if decoys:
                data_out.append(data_chunk)
            else:
                data_out.append(data_chunk.loc[targets_chunk, out_columns])

        for writer, data in zip(writers, data_out):
            writer.append_data(data)

    # Finalize writer (clear buffers etc.)
    for writer in writers:
        writer.finalize()
