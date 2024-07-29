"""
Classes for reading and writing data in tabular form.
"""

from __future__ import annotations

import sqlite3
import warnings
from contextlib import contextmanager
from enum import Enum
from typing import Generator

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
import pyarrow.parquet as pq
from typeguard import typechecked
import pyarrow as pa

CSV_SUFFIXES = [
    ".csv",
    ".pin",
    ".tab",
    ".peptides",
    ".psms",
    ".proteins",
    ".modifiedpeptides",
    ".peptidegroups",
    ".modified_peptides",
    ".peptide_groups",
    ".precursors",
]
PARQUET_SUFFIXES = [".parquet"]
SQLITE_SUFFIXES = [".db"]


class TableType(Enum):
    DataFrame = "DataFrame"
    Records = "Records"
    Dicts = "Dicts"


def get_score_column_type(suffix):
    if suffix in PARQUET_SUFFIXES:
        return pa.float64()
    elif suffix in CSV_SUFFIXES:
        return "float"
    else:
        raise ValueError(f"Suffix '{suffix}' does not match expected formats")


@typechecked
class TabularDataReader(ABC):
    """
    An abstract class that represents a source for tabular data that can be
    read in either completely or chunk-wise.

    Implementations can be classes that either read from files, from memory
    (e.g. data frames), combine or modify other readers or represent computed
    tabular results.
    """
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

    def _returned_dataframe_is_mutable(self):
        return True

    @staticmethod
    def from_path(
        file_name: Path, column_map: dict[str, str] | None = None, **kwargs
    ) -> "TabularDataReader":
        # Currently, we look only at the suffix, however, in the future we
        # could also look into the file itself (is it ascii? does it have
        # some "magic bytes"? ...)
        suffix = file_name.suffix
        if suffix in CSV_SUFFIXES:
            reader = CSVFileReader(file_name, **kwargs)
        elif suffix in PARQUET_SUFFIXES:
            reader = ParquetFileReader(file_name, **kwargs)
        else:
            # Fallback
            warnings.warn(
                f"Suffix '{suffix}' not recognized in file name '{file_name}'."
                " Falling back to CSV..."
            )
            reader = CSVFileReader(file_name, **kwargs)

        if column_map is not None:
            reader = ColumnMappedReader(reader, column_map)

        return reader


@typechecked
class ColumnMappedReader(TabularDataReader):
    """
    A tabular data reader that renames the columns of another tabular data
    reader to new names.

    Attributes:
    -----------
        reader : TabularDataReader
            The underlying reader for the original data.
        column_map : dict[str, str]
            A dictionary that maps the original column names to the new
            column names.
    """
    def __init__(self, reader: TabularDataReader, column_map: dict[str, str]):
        self.reader = reader
        self.column_map = column_map

    def get_column_names(self) -> list[str]:
        return [
            self.column_map.get(column, column)
            for column in self.reader.get_column_names()
        ]

    def get_column_types(self) -> list:
        return self.reader.get_column_types()

    def _get_orig_columns(self, columns: list[str] | None) -> list[str] | None:
        if columns is None:
            return None

        all_orig_columns = self.reader.get_column_names()
        all_columns = self.get_column_names()
        reverse_column_map = dict(zip(all_columns, all_orig_columns))
        orig_columns = [reverse_column_map[column] for column in columns]
        return orig_columns

    def _get_mapped_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # todo: enable this again... Modifying in-place would be more
        #       efficient than creating a copy, but this implementation
        #       creates errors. Once those are ironed out we can re-enable
        #       this code.
        #
        # if self._returned_dataframe_is_mutable():
        #     df.rename(columns=self.column_map, inplace=True, copy=False)
        # else:
        #     df = df.rename(
        #        columns=self.column_map, inplace=False, copy=False
        #     )
        df = df.rename(columns=self.column_map, inplace=False)
        return df

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        df = self.reader.read(columns=self._get_orig_columns(columns))
        return self._get_mapped_dataframe(df)

    def get_chunked_data_iterator(
        self, chunk_size: int, columns: list[str] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        orig_columns = self._get_orig_columns(columns)
        for chunk in self.reader.get_chunked_data_iterator(
            chunk_size, columns=orig_columns
        ):
            yield self._get_mapped_dataframe(chunk)


@typechecked
class CSVFileReader(TabularDataReader):
    """
    A tabular data reader for reading CSV files.

    Attributes:
    -----------
        file_name :  Path
            The path to the CSV file.
        stdargs : dict
            Arguments for reading CSV file passed on to the pandas
            `read_csv` function.
    """
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

    def get_column_types(self) -> list:
        df = pd.read_csv(self.file_name, **self.stdargs, nrows=2)
        return df.dtypes.tolist()

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
    """
    This class allows reading pandas DataFrames in the context of tabular data
    readers.

    Attributes:
    -----------
        df : pd.DataFrame
            The DataFrame being read from.
    """
    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __str__(self):
        return f"DataFrameReader({self.df.columns=})"

    def __repr__(self):
        return f"DataFrameReader({self.df=})"

    def get_column_names(self) -> list[str]:
        return self.df.columns.tolist()

    def get_column_types(self) -> list:
        return self.df.dtypes.tolist()

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        return self.df if columns is None else self.df[columns]

    def get_chunked_data_iterator(
        self, chunk_size: int, columns: list[str] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        for pos in range(0, len(self.df), chunk_size):
            chunk = self.df.iloc[pos : pos + chunk_size]
            yield chunk if columns is None else chunk[columns]

    def _returned_dataframe_is_mutable(self):
        return False

    @staticmethod
    def from_series(series: pd.Series, name=None) -> "DataFrameReader":
        if name is not None:
            return DataFrameReader(series.to_frame(name=name))
        else:
            return DataFrameReader(series.to_frame())

    @staticmethod
    def from_array(array: list | np.ndarray, name: str) -> "DataFrameReader":
        return DataFrameReader(pd.DataFrame({name: array}))


@typechecked
class ParquetFileReader(TabularDataReader):
    """
    A class for reading Parquet files and retrieving data in tabular format.

    Attributes:
    -----------
    file_name : Path
        The path to the Parquet file.
    """

    def __init__(self, file_name: Path):
        self.file_name = file_name

    def __str__(self):
        return f"ParquetFileReader({self.file_name=})"

    def __repr__(self):
        return f"ParquetFileReader({self.file_name=})"

    def get_column_names(self) -> list[str]:
        return pq.ParquetFile(self.file_name).schema.names

    def get_column_types(self) -> list:
        return pq.ParquetFile(self.file_name).schema.to_arrow_schema().types

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        result = pq.read_table(self.file_name, columns=columns).to_pandas()
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
    """
    Abstract base class for writing tabular data to different file formats.

    Attributes:
    -----------
        columns : list[str]
            List of column names
        column_types : list | None
            List of column types (optional)
    """
    def __init__(
        self,
        columns: list[str],
        column_types: list | None = None,
    ):
        self.columns = columns
        self.column_types = column_types
        # todo: I think the TDW should have a field/option that says whether
        #  data should be appended or whether the file should be cleared first
        #  if it already contains data

    def get_column_names(self) -> list[str]:
        return self.columns

    def get_column_types(self) -> list:
        return self.column_types

    @abstractmethod
    def append_data(self, data: pd.DataFrame):
        raise NotImplementedError

    def check_valid_data(self, data: pd.DataFrame):
        columns = data.columns.tolist()
        if not columns == self.get_column_names():
            raise ValueError(
                f"Column names {columns} do not "
                f"match {self.get_column_names()}"
            )

        if self.column_types is not None:
            pass
            # todo: Commented out for a while till we have a better type
            #  compatibility check, or agreed on some "super type" of numpy
            #  dtype and pyarrow types (and what not...)
            # column_types = data.dtypes.tolist()
            # if not column_types == self.get_column_types():
            #     raise ValueError(
            #         f"Column types {column_types} do "
            #         f"not match {self.get_column_types()}"
            #     )

    def write(self, data: pd.DataFrame):
        self.check_valid_data(data)
        self.initialize()
        self.append_data(data)
        self.finalize()

    def initialize(self):
        pass

    def finalize(self):
        pass

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.finalize()

    @abstractmethod
    def get_associated_reader(self):
        raise NotImplementedError

    @staticmethod
    def from_suffix(
        file_name: Path,
        columns: list[str],
        buffer_size: int = 0,
        buffer_type: TableType = TableType.DataFrame,
        **kwargs,
    ) -> "TabularDataWriter":
        suffix = file_name.suffix
        if suffix in CSV_SUFFIXES:
            writer = CSVFileWriter(file_name, columns, **kwargs)
        elif suffix in PARQUET_SUFFIXES:
            writer = ParquetFileWriter(file_name, columns, **kwargs)
        elif suffix in SQLITE_SUFFIXES:
            writer = SqliteWriter(file_name, columns, **kwargs)
        else:  # Fallback
            warnings.warn(
                f"Suffix '{suffix}' not recognized in file name '{file_name}'."
                " Falling back to CSV..."
            )
            writer = CSVFileWriter(file_name, columns, **kwargs)

        if buffer_size > 1:
            writer = BufferedWriter(writer, buffer_size, buffer_type)
        return writer


@contextmanager
# @typechecked
def auto_finalize(writers: list[TabularDataWriter]):
    # todo: this method should actually (to be really secure), check which
    #  writers were correctly initialized and if some initialization throws an
    #  error, finalize all that already have been initialized. Similar with
    #  errors during finalization.
    for writer in writers:
        writer.__enter__()
    try:
        yield None
    finally:
        for writer in writers:
            writer.__exit__(None, None, None)


@typechecked
class BufferedWriter(TabularDataWriter):
    """
    This class represents a buffered writer for tabular data. It allows
    writing data to a tabular data writer in batches, reducing the
    number of write operations.

    Attributes:
    -----------
    writer : TabularDataWriter
        The tabular data writer to which the data will be written.
    buffer_size : int
        The number of records to buffer before writing to the writer.
    buffer_type : TableType
        The type of buffer being used. Can be one of TableType.DataFrame,
        TableType.Dicts, or TableType.Records.
    buffer : pd.DataFrame or list of dictionaries or np.recarray or None
        The buffer containing the tabular data to be written.
        The buffer type depends on the buffer_type attribute.
    """
    writer: TabularDataWriter
    buffer_size: int
    buffer_type: TableType
    buffer: pd.DataFrame | list[dict] | np.recarray | None

    def __init__(
        self,
        writer: TabularDataWriter,
        buffer_size=1000,
        buffer_type=TableType.DataFrame,
    ):
        super().__init__(writer.columns, writer.column_types)
        self.writer = writer
        self.buffer_size = buffer_size
        self.buffer_type = buffer_type
        self.buffer = None

    def _buffer_slice(
        self,
        start: int = 0,
        end: int | None = None,
        as_dataframe: bool = False,
    ):
        if self.buffer_type == TableType.DataFrame:
            slice = self.buffer.iloc[start:end]
        else:
            slice = self.buffer[start:end]
        if as_dataframe and not isinstance(slice, pd.DataFrame):
            return pd.DataFrame(slice)
        else:
            return slice

    def _write_buffer(self, force=False):
        if self.buffer is None:
            return
        while len(self.buffer) >= self.buffer_size:
            self.writer.append_data(
                self._buffer_slice(end=self.buffer_size, as_dataframe=True)
            )
            self.buffer = self._buffer_slice(
                start=self.buffer_size,
            )
        if force and len(self.buffer) > 0:
            self.writer.append_data(self._buffer_slice(as_dataframe=True))
            self.buffer = None

    def append_data(self, data: pd.DataFrame | dict | list[dict] | np.record):
        if self.buffer_type == TableType.DataFrame:
            if not isinstance(data, pd.DataFrame):
                raise TypeError(
                    "Parameter data must be of type DataFrame,"
                    f" not {type(data)}"
                )

            if self.buffer is None:
                self.buffer = data.copy(deep=True)
            else:
                self.buffer = pd.concat(
                    [self.buffer, data], axis=0, ignore_index=True
                )
        elif self.buffer_type == TableType.Dicts:
            if isinstance(data, dict):
                data = [data]
            if self.buffer is None:
                self.buffer = []
            self.buffer += data
        elif self.buffer_type == TableType.Records:
            if self.buffer is None:
                self.buffer = np.recarray(shape=(0,), dtype=data.dtype)
            self.buffer = np.append(self.buffer, data)
        else:
            raise RuntimeError("Not yet done...")

        self._write_buffer()

    def check_valid_data(self, data: pd.DataFrame):
        return self.writer.check_valid_data(data)

    def write(self, data: pd.DataFrame):
        self.writer.write(data)

    def initialize(self):
        self.writer.initialize()

    def finalize(self):
        self._write_buffer(force=True)
        self.writer.finalize()

    def get_associated_reader(self):
        return self.writer.get_associated_reader()


@typechecked
class CSVFileWriter(TabularDataWriter):
    """
    CSVFileWriter class for writing tabular data to a CSV file.

    Attributes:
    -----------
    file_name : Path
        The file path where the CSV file will be written.

    sep : str, optional
        The separator string used to separate fields in the CSV file.
        Default is tab character ("\t").
    """
    file_name: Path

    def __init__(
        self,
        file_name: Path,
        columns: list[str],
        column_types: list | None = None,
        sep: str = "\t",
    ):
        super().__init__(columns, column_types)
        self.file_name = file_name
        self.stdargs = {"sep": sep, "index": False}

    def __str__(self):
        return f"CSVFileWriter({self.file_name=},{self.columns=})"

    def __repr__(self):
        return (
            f"CSVFileWriter({self.file_name=},{self.columns=},{self.stdargs=})"
        )

    def get_schema(self, as_dict: bool = True):
        schema = []
        for name, type in zip(self.columns, self.column_types):
            schema.append((name, type))
        return {name: str(type) for name, type in schema}

    def initialize(self):
        # Just write header information
        df = pd.DataFrame(columns=self.columns)
        df.to_csv(self.file_name, **self.stdargs)

    def finalize(self):
        # no need to do anything here
        pass

    def append_data(self, data: pd.DataFrame):
        self.check_valid_data(data)
        data.to_csv(self.file_name, mode="a", header=False, **self.stdargs)

    def get_associated_reader(self):
        return CSVFileReader(self.file_name, sep=self.stdargs["sep"])


@typechecked
class ParquetFileWriter(TabularDataWriter):
    """
    This class is responsible for writing tabular data into Parquet files.


    Attributes:
    -----------
    file_name : Path
        The path to the Parquet file being written.
    """
    file_name: Path

    def __init__(
        self,
        file_name: Path,
        columns: list[str],
        column_types: list | None = None,
    ):
        super().__init__(columns, column_types)
        self.file_name = file_name

    def __str__(self):
        return f"ParquetFileWriter({self.file_name=},{self.columns=})"

    def __repr__(self):
        return f"ParquetFileWriter({self.file_name=},{self.columns=})"

    def get_schema(self, as_dict: bool = False):
        schema = []
        for name, type in zip(self.columns, self.column_types):
            schema.append((name, type))
        if as_dict:
            return {name: str(type) for name, type in schema}
        return pa.schema(schema)

    def initialize(self):
        self.writer = pq.ParquetWriter(
            self.file_name, schema=self.get_schema()
        )

    def finalize(self):
        self.writer.close()

    def append_data(self, data: pd.DataFrame):
        table = pa.Table.from_pandas(
            data, preserve_index=False, schema=self.get_schema()
        )
        self.writer.write_table(table)

    def write(self, data: pd.DataFrame):
        data.to_parquet(self.file_name, index=False)

    def get_associated_reader(self):
        return ParquetFileReader(self.file_name)


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
        # todo: maybe we can supply also a default implementation in this class
        #       given the table name and sql column names
        raise NotImplementedError

    def get_associated_reader(self):
        # todo: need an sqlite reader first...
        raise NotImplementedError


@typechecked
def remove_columns(
    column_names: list[str],
    column_types: list,
    columns_to_remove: list[str],
) -> tuple[list[str], list]:
    temp_columns = [
        (column, type)
        for column, type in zip(column_names, column_types)
        if column not in columns_to_remove
    ]
    temp_column_names, temp_column_types = zip(*temp_columns)
    return (list(temp_column_names), list(temp_column_types))
