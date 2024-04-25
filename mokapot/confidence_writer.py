import sqlite3
from pathlib import Path

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
        pep_column: str="posterior_error_prob",
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
            query = f"UPDATE CANDIDATE SET PSM_FDR = :{qvalue_column}, SVM_SCORE = :score, POSTERIOR_ERROR_PROBABILITY = :{pep_column} WHERE CANDIDATE_ID = :PSMId;"
        else:
            table_name, table_id_col, mokapot_id_col = self.level_cols[level]
            query = f"INSERT INTO {table_name}({table_id_col},FDR,PEP,SVM_SCORE) VALUES(:{mokapot_id_col},:{qvalue_column},:{pep_column},:score)"
        return query

    def append_data(self, data):
        query = self.get_query(self.level, self.qvalue_column, self.pep_column)
        data = data.apply(pd.to_numeric, errors="ignore")  # fixme: this should be
        data = data.to_dict("records")
        for row in data:
            self.connection.execute(query, row)


class ConfidenceWriter:
    def __init__(
        self,
        data_iterator,
        q_value_iterator,
        pep_iterator,
        target_iterator,
        out_paths,
        decoys,
        level,
        out_columns,
    ):
        self.data_iterator = data_iterator
        self.q_value_iterator = q_value_iterator
        self.pep_iterator = pep_iterator
        self.target_iterator = target_iterator
        self.out_paths = out_paths
        self.decoys = decoys
        self.level = level
        self.out_columns = out_columns
        self.qvalue_column = "q_value"
        self.pep_column = "posterior_error_prob"
        if not self.decoys and len(self.out_paths) > 1:
            self.out_paths.pop(1)
        self.is_sqlite = True if self.out_paths[0].suffix == ".db" else False
        if self.is_sqlite:
            create_writer = lambda path: ConfidenceSqliteWriter(path, self.out_columns, level=self.level, qvalue_column=self.qvalue_column, pep_column=self.pep_column)
        else:
            create_writer = lambda path: TabularDataWriter.from_suffix(path, self.out_columns)
        self.writers = [create_writer(path)for path in self.out_paths]

    def write(self):
        for data_chunk, qvals_chunk, peps_chunk, targets_chunk in zip(
            self.data_iterator,
            self.q_value_iterator,
            self.pep_iterator,
            self.target_iterator,
        ):
            data_chunk[self.qvalue_column] = qvals_chunk
            data_chunk[self.pep_column] = peps_chunk
            data_out = []
            if not self.is_sqlite:
                data_out.append(
                    data_chunk.loc[targets_chunk, self.out_columns]
                )
                if self.decoys:
                    data_out.append(
                        data_chunk.loc[~targets_chunk, self.out_columns]
                    )
            else:
                if self.decoys:
                    data_out.append(data_chunk)
                else:
                    data_out.append(
                        data_chunk.loc[targets_chunk, self.out_columns]
                    )
            for writer, data in zip(self.writers, data_out):
                writer.append_data(data)

    def commit_data(self):
        for writer in self.writers:
            writer.commit_data()
