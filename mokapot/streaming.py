from __future__ import annotations

from typing import Generator, Callable, Iterator

import pandas as pd
import numpy as np
from typeguard import typechecked
import pyarrow as pa

from mokapot.tabular_data import TabularDataReader, TableType


@typechecked
class JoinedTabularDataReader(TabularDataReader):
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
    def __init__(
        self,
        reader: TabularDataReader,
        column: str,
        dtype: np.dtype | pa.DataType,
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

    def _reader_columns(self, columns: list[str]):
        return [column for column in columns if column != self.column]

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        df = self.reader.read(self._reader_columns(columns))
        if columns is not None or self.column in columns:
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
            if columns is not None or self.column in columns:
                df[self.column] = self.func(df)
            yield df if columns is None else df[columns]


@typechecked
class MergedTabularDataReader(TabularDataReader):
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

        assert len(readers) > 0, "At least one data reader is required"
        self.column_names = readers[0].get_column_names()
        self.column_types = readers[0].get_column_types()

        # todo: all those asserts should raise an exception,
        #   could happen in production too
        for reader in readers:
            assert (
                reader.get_column_names() == self.column_names
            ), "Column names do not match"
            assert (
                reader.get_column_types() == self.column_types
            ), "Column types do not match"
        assert (
            priority_column in self.column_names
        ), "Priority column not found"

    def get_column_names(self) -> list[str]:
        return self.column_names

    def get_column_types(self) -> list:
        return self.column_types

    def get_row_iterator(
        self,
        columns: list[str] | None = None,
        row_type: TableType = TableType.DataFrame,
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

        if row_type == TableType.DataFrame:
            iterate_over_chunk = iterate_over_df
            get_value = get_value_df
        elif row_type == TableType.Dicts:
            iterate_over_chunk = iterate_over_dicts
            get_value = get_value_dict
        elif row_type == TableType.Records:
            iterate_over_chunk = iterate_over_records
            get_value = get_value_dict
        else:
            raise ValueError(
                "ret_type must be 'dataframe', 'records'"
                f" or 'dicts', not {row_type}"
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
