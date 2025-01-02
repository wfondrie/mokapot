"""
Helper classes and methods used for streaming of tabular data.
"""

from __future__ import annotations

import warnings
from pprint import pformat
from typing import Callable, Generator, Iterator

import numpy as np
import pandas as pd
from typeguard import typechecked

from mokapot.tabular_data import (
    BufferType,
    TabularDataReader,
    TabularDataWriter,
)


@typechecked
class JoinedTabularDataReader(TabularDataReader):
    """
    Handles data from multiple tabular data sources, joining them horizontally.

    Attributes:
    -----------
        readers : list[TabularDataReader]
            A list of 'TabularDataReader' objects representing the individual
            data sources.
    """

    readers: list[TabularDataReader]

    def __init__(self, readers: list[TabularDataReader]):
        self.readers = readers

    def get_column_names(self) -> list[str]:
        return sum([reader.get_column_names() for reader in self.readers], [])

    def get_column_types(self) -> list:
        return sum([reader.get_column_types() for reader in self.readers], [])

    def _subset_columns(
        self, column_names: list[str] | None
    ) -> list[list[str] | None]:
        if column_names is None:
            return [None for _ in self.readers]
        return [
            [
                column_name
                for column_name in reader.get_column_names()
                if column_name in column_names
            ]
            for reader in self.readers
        ]

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        subset_column_lists = self._subset_columns(columns)
        df = pd.concat(
            [
                reader.read(columns=subset_columns)
                for reader, subset_columns in zip(
                    self.readers, subset_column_lists
                )
            ],
            axis=1,
        )
        return df if columns is None else df[columns]

    def get_chunked_data_iterator(
        self, chunk_size: int, columns: list[str] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        subset_column_lists = self._subset_columns(columns)
        iterators = [
            reader.get_chunked_data_iterator(
                chunk_size=chunk_size, columns=subset_columns
            )
            for reader, subset_columns in zip(
                self.readers, subset_column_lists
            )
        ]

        while True:
            try:
                chunks = [next(iterator) for iterator in iterators]
            except StopIteration:
                break
            df = pd.concat(chunks, axis=1)
            yield df if columns is None else df[columns]


@typechecked
class ComputedTabularDataReader(TabularDataReader):
    """
    A subclass of TabularDataReader that allows the computation of a specific
    column that is joined horizontally to the columns of the reader.

    Attributes:
    -----------
        reader : TabularDataReader
            The underlying reader object.
        column : str
            The name of the column to compute.
        dtype : np.dtype | pa.DataType
            The data type of the computed column.
        func : Callable
            A function to apply to the existing columns of each chunk.
    """

    def __init__(
        self,
        reader: TabularDataReader,
        column: str,
        dtype: np.dtype,
        func: Callable,
    ):
        self.reader = reader
        self.dtype = dtype
        self.func = func
        self.column = column

    def get_column_names(self) -> list[str]:
        return self.reader.get_column_names() + [self.column]

    def get_column_types(self) -> list:
        return self.reader.get_column_types() + [self.dtype]

    def _reader_columns(self, columns: list[str] | None):
        # todo: performance: Currently, we need to read all columns, since we
        #  don't know what's needed in the computation. This could be made more
        #  efficient by letting the class know which columns those are.
        return None

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        df = self.reader.read(self._reader_columns(columns))
        # We need to compute the result column only in two cases:
        #   a) all columns are requested (columns = None)
        #   b) the computed column is requested explicitly
        if columns is None or self.column in columns:
            df[self.column] = self.func(df)
        return df if columns is None else df[columns]

    def get_chunked_data_iterator(
        self, chunk_size: int, columns: list[str] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        iterator = self.reader.get_chunked_data_iterator(
            chunk_size=chunk_size, columns=self._reader_columns(columns)
        )

        while True:
            try:
                df = next(iterator)
            except StopIteration:
                break
            # See comments in `read` for explanation
            if columns is None or self.column in columns:
                df[self.column] = self.func(df)
            yield df if columns is None else df[columns]


@typechecked
class MergedTabularDataReader(TabularDataReader):
    """
    Merges data from multiple tabular data sources vertically into a single
    data source, ordering the rows (one by one) by the value of a priority
    column. I.e. for each output row, the row of the input readers with the
    highest value of the priority column is picked.

    Attributes:
    -----------
        readers : list[TabularDataReader]
            List of data readers to merge.
        priority_column : str
            Name of the priority column used for merging (highest value
            determines which reader to pick next).
        descending : bool
            Flag indicating whether the merge should be in descending order
            (default: True).
        reader_chunk_size : int
            Chunk size used when iterating over data readers (default: 1000).
        column_names : list[str]
            List of column names for the merged data.
        column_types : list
            List of column types for the merged data.
    """

    def __init__(
        self,
        readers: list[TabularDataReader],
        priority_column: str,
        descending: bool = True,
        reader_chunk_size: int = 1000,
    ):
        self.readers = readers
        self.priority_column = priority_column
        self.descending = descending
        self.reader_chunk_size = reader_chunk_size

        if len(readers) == 0:
            raise ValueError("At least one data reader is required")

        self.column_names = readers[0].get_column_names()
        self.column_types = readers[0].get_column_types()

        for reader in readers:
            if not reader.get_column_names() == self.column_names:
                raise ValueError("Column names do not match")

            if not reader.get_column_types() == self.column_types:
                raise ValueError("Column types do not match")

            if priority_column not in self.column_names:
                raise ValueError("Priority column not found")

    def get_column_names(self) -> list[str]:
        return self.column_names

    def get_column_types(self) -> list:
        return self.column_types

    def get_row_iterator(
        self,
        columns: list[str] | None = None,
        row_type: BufferType = BufferType.DataFrame,
    ) -> Iterator[pd.DataFrame | dict | np.record]:
        def iterate_over_df(df: pd.DataFrame) -> Iterator:
            for i in range(len(df)):
                row = df.iloc[[i]]
                row.index = [0]
                yield row

        def get_value_df(row, col):
            return row[col].iloc[0]

        def iterate_over_dicts(df: pd.DataFrame) -> Iterator:
            dict = df.to_dict(orient="records")
            return iter(dict)

        def get_value_dict(row, col):
            return row[col]

        def iterate_over_records(df: pd.DataFrame) -> Iterator:
            records = df.to_records(index=False)
            return iter(records)

        if row_type == BufferType.DataFrame:
            iterate_over_chunk = iterate_over_df
            get_value = get_value_df
        elif row_type == BufferType.Dicts:
            iterate_over_chunk = iterate_over_dicts
            get_value = get_value_dict
        elif row_type == BufferType.Records:
            iterate_over_chunk = iterate_over_records
            get_value = get_value_dict
        else:
            raise ValueError(
                "ret_type must be 'dataframe', 'records' or 'dicts',"
                f" not {row_type}"
            )

        def row_iterator_from_chunked(chunked_iter: Iterator) -> Iterator:
            for chunk in chunked_iter:
                for row in iterate_over_chunk(chunk):
                    yield row

        row_iterators = [
            row_iterator_from_chunked(
                reader.get_chunked_data_iterator(
                    chunk_size=self.reader_chunk_size, columns=columns
                )
            )
            for reader in self.readers
        ]
        current_rows = [next(row_iterator) for row_iterator in row_iterators]

        values = [get_value(row, self.priority_column) for row in current_rows]
        while len(row_iterators):
            if self.descending:
                iterator_index = np.argmax(values)
            else:
                iterator_index = np.argmin(values)

            row = current_rows[iterator_index]
            yield row

            try:
                current_rows[iterator_index] = next(
                    row_iterators[iterator_index]
                )
                new_value = get_value(
                    current_rows[iterator_index], self.priority_column
                )
                if self.descending and new_value > values[iterator_index]:
                    raise ValueError(
                        f"Value {new_value} exceeds {self.priority_column}"
                        " but should be descending"
                    )
                if not self.descending and new_value < values[iterator_index]:
                    raise ValueError(
                        f"Value {new_value} lower than {self.priority_column}"
                        " but should be ascending"
                    )

                values[iterator_index] = new_value
            except StopIteration:
                del row_iterators[iterator_index]
                del current_rows[iterator_index]
                del values[iterator_index]

    def get_chunked_data_iterator(
        self, chunk_size: int, columns: list[str] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        row_iterator = self.get_row_iterator(columns=columns)
        finished = False
        rows = []
        while not finished:
            try:
                row = next(row_iterator)
                rows.append(row)
            except StopIteration:
                finished = True
            if (finished and len(rows) > 0) or len(rows) == chunk_size:
                df = pd.concat(rows)
                df.reset_index(drop=True, inplace=True)
                yield df
                rows = []

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        row_iterator = self.get_row_iterator(columns=columns)
        rows = [row for row in row_iterator]
        df = pd.concat(rows)
        df.reset_index(drop=True, inplace=True)
        return df


@typechecked
def join_readers(readers: list[TabularDataReader]):
    return JoinedTabularDataReader(readers)


@typechecked
def merge_readers(
    readers: list[TabularDataReader],
    priority_column: str,
    descending: bool = True,
    reader_chunk_size: int = 1000,
):
    reader = MergedTabularDataReader(
        readers,
        priority_column,
        descending,
        reader_chunk_size=reader_chunk_size,
    )
    iterator = reader.get_chunked_data_iterator(chunk_size=1)
    return iterator


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
    buffer_type: BufferType
    buffer: pd.DataFrame | list[dict] | np.recarray | None

    def __init__(
        self,
        writer: TabularDataWriter,
        buffer_size=1000,
        buffer_type=BufferType.DataFrame,
    ):
        super().__init__(writer.columns, writer.column_types)
        self.writer = writer
        self.buffer_size = buffer_size
        self.buffer_type = buffer_type
        self.buffer = None
        # For BufferedWriters it is extremely important that they are
        # correctly initialized and finalized, so we make sure
        self.finalized = False
        self.initialized = False

    def __repr__(self):
        IGNORE_KEYS = {"buffer"}
        dict_repr = pformat({
            k: v for k, v in self.__dict__.items() if k not in IGNORE_KEYS
        })
        return f"{self.__class__!s}({dict_repr})"

    def __del__(self):
        if self.initialized and not self.finalized:
            warnings.warn(
                f"BufferedWriter not finalized (buffering: {self.writer})"
            )

    def _buffer_slice(
        self,
        start: int = 0,
        end: int | None = None,
        as_dataframe: bool = False,
    ):
        if self.buffer_type == BufferType.DataFrame:
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
            slice = self._buffer_slice(as_dataframe=True)
            self.writer.append_data(slice)
            self.buffer = None

    def append_data(self, data: pd.DataFrame | dict | list[dict] | np.record):
        assert self.initialized and not self.finalized

        if self.buffer_type == BufferType.DataFrame:
            if not isinstance(data, pd.DataFrame):
                raise TypeError(
                    "Parameter `data` must be of type DataFrame,"
                    f" not {type(data)}"
                )

            if self.buffer is None:
                self.buffer = data.copy(deep=True)
            else:
                self.buffer = pd.concat(
                    [self.buffer, data], axis=0, ignore_index=True
                )
        elif self.buffer_type == BufferType.Dicts:
            if isinstance(data, dict):
                data = [data]
            if not (isinstance(data, list) and isinstance(data[0], dict)):
                raise TypeError(
                    "Parameter `data` must be of type dict or list[dict],"
                    f" not {type(data)}"
                )
            if self.buffer is None:
                self.buffer = []
            self.buffer += data
        elif self.buffer_type == BufferType.Records:
            if self.buffer is None:
                self.buffer = np.recarray(shape=(0,), dtype=data.dtype)
            self.buffer = np.append(self.buffer, data)
        else:
            raise ValueError(f"Unknown buffer type {self.buffer_type}")

        self._write_buffer()

    def check_valid_data(self, data: pd.DataFrame):
        return self.writer.check_valid_data(data)

    def write(self, data: pd.DataFrame):
        self.writer.write(data)

    def initialize(self):
        assert not self.initialized
        self.initialized = True
        self.writer.initialize()

    def finalize(self):
        assert self.initialized
        self.finalized = True  # Only for checking whether this got called
        self._write_buffer(force=True)
        self.writer.finalize()

    def get_associated_reader(self):
        return self.writer.get_associated_reader()
