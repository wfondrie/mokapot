import sqlite3
import warnings
from typing import Generator

import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
import pyarrow.parquet as pq
from typeguard import typechecked
from numpy import dtype

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
class TabularDataReader(ABC):
    @abstractmethod
    def get_column_names(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_column_types(self) -> list[dtype]:
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
    def from_path(file_name: Path, **kwargs) -> "TabularDataReader":
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


def _types_from_dataframe(df: pd.DataFrame) -> list[dtype]:
    type_map = df.dtypes
    column_names = df.columns.tolist()
    return [type_map[column_name] for column_name in column_names]


@typechecked
class CSVFileReader(TabularDataReader):
    def __init__(self, file_name: Path, sep: str = "\t"):
        self.file_name = file_name
        self.stdargs = {"sep": sep, "index_col": False}

    def __str__(self):
        return f"CSVFileReader({self.file_name=})"

    def __repr__(self):
        return f"CSVFileReader({self.file_name=},{self.stdargs=})"

    def get_column_names(self) -> list[str]:
        df = pd.read_csv(self.file_name, **self.stdargs, nrows=0)
        return df.columns.tolist()

    def get_column_types(self) -> list[dtype]:
        df = pd.read_csv(self.file_name, **self.stdargs, nrows=2)
        return _types_from_dataframe(df)

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
class DataFrameReader(TabularDataReader):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __str__(self):
        return f"DataFrameReader({self.df.columns=})"

    def __repr__(self):
        return f"DataFrameReader({self.df=})"

    def get_column_names(self) -> list[str]:
        return self.df.columns.tolist()

    def get_column_types(self) -> list[dtype]:
        return _types_from_dataframe(self.df)

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        return self.df

    def get_chunked_data_iterator(
        self, chunk_size: int, columns: list[str] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        for pos in range(0, len(self.df), chunk_size):
            chunk = self.df.iloc[pos:pos + chunk_size]
            yield chunk if columns is None else chunk[columns]


@typechecked
class ParquetFileReader(TabularDataReader):
    def __init__(self, file_name: Path):
        self.file_name = file_name

    def __str__(self):
        return f"ParquetFileReader({self.file_name=})"

    def __repr__(self):
        return f"ParquetFileReader({self.file_name=})"

    def get_column_names(self) -> list[str]:
        return pq.ParquetFile(self.file_name).schema.names

    def get_column_types(self) -> list:
        iterator = self.get_chunked_data_iterator(chunk_size=2)
        first_rows = next(iterator)
        return _types_from_dataframe(first_rows)

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
class TabularDataWriter(ABC):
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
    ) -> "TabularDataWriter":
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
class CSVFileWriter(TabularDataWriter):
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
class ParquetFileWriter(TabularDataWriter):
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


@typechecked
class SqliteWriter(TabularDataWriter):
    def __init__(
        self,
        database: str | Path | sqlite3.Connection,
        columns: list[str],
        column_types: list | None = None,
    ) -> None:
        if isinstance(database, sqlite3.Connection):
            self.file_name = None
            self.connection = database
        else:
            self.file_name = database
            self.connection = sqlite3.connect(self.file_name)
        self.columns = columns
        self.column_types = column_types

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

    def commit_data(self):
        self.connection.commit()
        self.connection.close()
