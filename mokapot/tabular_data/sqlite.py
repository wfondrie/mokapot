import sqlite3
from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd
from typeguard import typechecked

from mokapot.tabular_data import TabularDataWriter


@typechecked
class SqliteWriter(TabularDataWriter, ABC):
    """
    SqliteWriter class for writing tabular data to SQLite database.
    """

    connection: sqlite3.Connection

    def __init__(
        self,
        database: str | Path | sqlite3.Connection,
        columns: list[str],
        column_types: list | None = None,
    ) -> None:
        super().__init__(columns, column_types)
        if isinstance(database, sqlite3.Connection):
            self.file_name = None
            self.connection = database
        else:
            self.file_name = database
            self.connection = sqlite3.connect(self.file_name)

    def __str__(self):
        return f"SqliteFileWriter({self.file_name=},{self.columns=})"

    def __repr__(self):
        return f"SqliteFileWriter({self.file_name=},{self.columns=})"

    def initialize(self):
        # Nothing to do here, we expect the table(s) to already exist
        pass

    def finalize(self):
        self.connection.commit()
        self.connection.close()

    def append_data(self, data: pd.DataFrame):
        # Must be implemented in derived class
        # todo: discuss: maybe we can supply also a default implementation in
        #  this class given the table name and sql column names.
        raise NotImplementedError

    def get_associated_reader(self):
        # Currently there is no SqliteReader and also no need for it
        raise NotImplementedError("SqliteWriter has no associated reader yet.")


@typechecked
class ConfidenceSqliteWriter(SqliteWriter):
    def __init__(
        self,
        database: str | Path | sqlite3.Connection,
        columns: list[str],
        column_types: list[np.dtype],
        level: str = "psms",
        qvalue_column: str = "q-value",
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

    def get_query(self, level):
        if level == "psms":
            query = "UPDATE CANDIDATE SET PSM_FDR = :q_value, SVM_SCORE = :score, POSTERIOR_ERROR_PROBABILITY = :posterior_error_prob WHERE CANDIDATE_ID = :PSMId;"  # noqa: E501
        else:
            table_name, table_id_col, mokapot_id_col = self.level_cols[level]
            query = f"INSERT INTO {table_name}({table_id_col},FDR,PEP,SVM_SCORE) VALUES(:{mokapot_id_col},:q_value,:posterior_error_prob,:score)"  # noqa: E501
        return query

    def append_data(self, data):
        query = self.get_query(self.level)
        data = data.to_dict("records")
        for row in data:
            row["q_value"] = row[self.qvalue_column]
            row["posterior_error_prob"] = row[self.pep_column]
        self.connection.executemany(query, data)

    def read(self, level: str = "psms"):
        table_name, table_id_col, mokapot_id_col = self.level_cols[level]
        return pd.read_sql_table(table_name, self.connection)
