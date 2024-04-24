import warnings
from typing import Generator

import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
import pyarrow.parquet as pq
from typeguard import typechecked
import sqlite3

CSV_SUFFIXES = [
    ".csv",
    ".pin",
    ".tab",
    ".peptides",
    ".psms",
    ".proteins",
    ".modifiedpeptides",
    ".peptidegroups",
    ".precursors",
]
PARQUET_SUFFIXES = [".parquet"]
SQLITE_SUFFIXES = [".db"]


@typechecked
class TabbedFileReader(ABC):
    @abstractmethod
    def get_column_names(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_column_types(self) -> list:
        raise NotImplementedError

    @abstractmethod
    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_chunked_data_iterator(
        self, chunk_size: int, columns: list[str] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        raise NotImplementedError

    @staticmethod
    def from_path(file_name: Path, **kwargs) -> "TabbedFileReader":
        # Currently, we look only at the suffix, however, in the future we could
        # also look into the file itself (is it ascii? does it have some "magic
        # bytes"? ...)
        suffix = file_name.suffix
        if suffix in CSV_SUFFIXES:
            return CSVFileReader(file_name, **kwargs)
        elif suffix in PARQUET_SUFFIXES:
            return ParquetFileReader(file_name)
        # Fallback
        warnings.warn(
            f"Suffix '{suffix}' not recognized in file name '{file_name}'. Falling back to CSV..."
        )
        return CSVFileReader(file_name, **kwargs)


@typechecked
class CSVFileReader(TabbedFileReader):
    def __init__(self, file_name: Path, sep: str = "\t"):
        self.file_name = file_name
        self.stdargs = {"sep": sep, "index_col": False}

    def __str__(self):
        return f"CSVFileReader({self.file_name=})"

    def __repr__(self):
        return f"CSVFileReader({self.file_name=},{self.stdargs=})"

    def get_column_names(self) -> list[str]:
        return pd.read_csv(
            self.file_name, **self.stdargs, nrows=0
        ).columns.tolist()

    def get_column_types(self) -> list:
        raise NotImplementedError

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        result = pd.read_csv(self.file_name, usecols=columns, **self.stdargs)
        return result if columns is None else result[columns]

    def get_chunked_data_iterator(
        self, chunk_size: int, columns: list[str] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        for chunk in pd.read_csv(
            self.file_name,
            usecols=columns,
            chunksize=chunk_size,
            **self.stdargs,
        ):
            yield chunk if columns is None else chunk[columns]


@typechecked
class ParquetFileReader(TabbedFileReader):
    def __init__(self, file_name: Path):
        self.file_name = file_name

    def __str__(self):
        return f"ParquetFileReader({self.file_name=})"

    def __repr__(self):
        return f"ParquetFileReader({self.file_name=})"

    def get_column_names(self) -> list[str]:
        return pq.ParquetFile(self.file_name).schema.names

    def get_column_types(self) -> list:
        raise NotImplementedError

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        result = (
            pq.read_table(self.file_name, columns=columns)
            .to_pandas()
            .apply(pd.to_numeric, errors="ignore")
        )
        return result

    def get_chunked_data_iterator(
        self, chunk_size: int, columns: list[str] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        pf = pq.ParquetFile(self.file_name)

        for i, record_batch in enumerate(
            pf.iter_batches(chunk_size, columns=columns)
        ):
            df = record_batch.to_pandas()
            df.index = df.index + i * chunk_size
            yield df


@typechecked
class TabbedFileWriter(ABC):
    @abstractmethod
    def get_column_names(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_column_types(self) -> list:
        raise NotImplementedError

    @abstractmethod
    def write_header(self):
        raise NotImplementedError

    @abstractmethod
    def append_data(self, data: pd.DataFrame):
        raise NotImplementedError

    def check_valid_data(self, data: pd.DataFrame):
        assert all(data.columns.to_numpy() == self.get_column_names())

    def write(self, data: pd.DataFrame):
        self.write_header()
        self.append_data(data)

    def commit_data(self):
        pass

    @staticmethod
    def from_suffix(
        file_name: Path, columns: list[str], **kwargs
    ) -> "TabbedFileWriter":
        suffix = file_name.suffix
        if suffix in CSV_SUFFIXES:
            return CSVFileWriter(file_name, columns, **kwargs)
        elif suffix in PARQUET_SUFFIXES:
            return ParquetFileWriter(file_name, columns)
        elif suffix in SQLITE_SUFFIXES:
            return SqliteWriter(file_name, columns)

        # Fallback
        warnings.warn(
            f"Suffix '{suffix}' not recognized in file name '{file_name}'. Falling back to CSV..."
        )
        return CSVFileWriter(file_name, columns, **kwargs)


@typechecked
class CSVFileWriter(TabbedFileWriter):
    def __init__(
        self,
        file_name: Path,
        columns: list[str],
        column_types: list | None = None,
        sep: str = "\t",
    ):
        super().__init__()
        self.file_name = file_name
        self.columns = columns
        self.column_types = column_types
        self.stdargs = {"sep": sep, "index": False}

    def __str__(self):
        return f"CSVFileWriter({self.file_name=},{self.columns=})"

    def __repr__(self):
        return (
            f"CSVFileWriter({self.file_name=},{self.columns=},{self.stdargs=})"
        )

    def get_column_names(self):
        return self.columns

    def get_column_types(self):
        return self.column_types

    def write_header(self):
        df = pd.DataFrame(columns=self.columns)
        df.to_csv(self.file_name, **self.stdargs)

    def append_data(self, data: pd.DataFrame):
        self.check_valid_data(data)
        data.to_csv(self.file_name, mode="a", header=False, **self.stdargs)


@typechecked
class ParquetFileWriter(TabbedFileWriter):
    def __init__(
        self,
        file_name: Path,
        columns: list[str],
        column_types: list | None = None,
    ):
        super().__init__()
        self.file_name = file_name
        self.columns = columns
        self.column_types = column_types

    def __str__(self):
        return f"ParquetFileWriter({self.file_name=},{self.columns=})"

    def __repr__(self):
        return f"ParquetFileWriter({self.file_name=},{self.columns=})"

    def get_column_names(self):
        return self.columns

    def get_column_types(self):
        return self.column_types

    def write_header(self):
        raise NotImplementedError

    def append_data(self, data: pd.DataFrame):
        raise NotImplementedError

    def write(self, data: pd.DataFrame):
        data.to_parquet(self.file_name, index=False)


class SqliteWriter(TabbedFileWriter):
    def __init__(
        self,
        file_name: Path,
        columns: list[str],
        column_types: list | None = None,
    ) -> None:
        self.file_name = file_name
        self.columns = columns
        self.column_types = column_types
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
        self.connection = sqlite3.connect(self.file_name)

    def __str__(self):
        return f"SqliteFileWriter({self.file_name=},{self.columns=})"

    def __repr__(self):
        return f"SqliteFileWriter({self.file_name=},{self.columns=})"

    def get_column_names(self):
        return self.columns

    def get_column_types(self):
        return self.column_types

    def write_header(self):
        raise NotImplementedError

    def get_query(self, level, qvalue_column, pep_column):
        if level == "psms":
            query = f"UPDATE CANDIDATE SET PSM_FDR = :{qvalue_column}, SVM_SCORE = :score, POSTERIOR_ERROR_PROBABILITY = :{pep_column} WHERE CANDIDATE_ID = :PSMId;"
        else:
            table_name, table_id_col, mokapot_id_col = self.level_cols[level]
            query = f"INSERT INTO {table_name}({table_id_col},FDR,PEP,SVM_SCORE) VALUES(:{mokapot_id_col},:{qvalue_column},:{pep_column},:score)"
        return query

    def append_data(self, data, level, qvalue_column, pep_column):
        query = self.get_query(level, qvalue_column, pep_column)
        data = data.apply(pd.to_numeric, errors="ignore")
        data = data.to_dict("records")
        for row in data:
            self.connection.execute(query, row)

    def commit_data(self):
        self.connection.commit()
        self.connection.close()


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
        self.is_sqlite = True if self.out_paths[0].suffix == ".db" else False
        if not self.decoys and len(self.out_paths) > 1:
            self.out_paths.pop(1)
        self.writers = [
            TabbedFileWriter.from_suffix(path, self.out_columns)
            for path in self.out_paths
        ]
        self.qvalue_column = "q_value"
        self.pep_column = "posterior_error_prob"

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
                if self.is_sqlite:
                    writer.append_data(
                        data,
                        level=self.level,
                        qvalue_column=self.qvalue_column,
                        pep_column=self.pep_column,
                    )
                else:
                    writer.append_data(data)

    def commit_data(self):
        for writer in self.writers:
            writer.commit_data()
