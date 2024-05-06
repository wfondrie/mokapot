from typing import Generator

import pandas as pd
import numpy as np
from numpy import dtype
from typeguard import typechecked

from mokapot.tabular_data import TabularDataReader


@typechecked
class JoinedTabularDataReader(TabularDataReader):
    readers: list[TabularDataReader]

    def __init__(self, readers: list[TabularDataReader]):
        self.readers = readers

    def get_column_names(self) -> list[str]:
        return sum([reader.get_column_names() for reader in self.readers], [])

    def get_column_types(self) -> list[dtype]:
        return sum([reader.get_column_types() for reader in self.readers], [])

    def _subset_columns(self, column_names: list[str] | None) -> list[
        list[str] | None]:
        if column_names is None:
            return [None for _ in self.readers]
        return [[column_name for column_name in reader.get_column_names() if
                 column_name in column_names] for reader in self.readers]

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        subset_column_lists = self._subset_columns(columns)
        df = pd.concat(
            [reader.read(columns=subset_columns) for reader, subset_columns in
             zip(self.readers, subset_column_lists)], axis=1)
        return df if columns is None else df[columns]

    def get_chunked_data_iterator(self, chunk_size: int,
                                  columns: list[str] | None = None) -> \
            Generator[pd.DataFrame, None, None]:
        subset_column_lists = self._subset_columns(columns)
        iterators = [reader.get_chunked_data_iterator(chunk_size=chunk_size,
                                                      columns=subset_columns)
                     for reader, subset_columns in
                     zip(self.readers, subset_column_lists)]

        while True:
            try:
                chunks = [next(iterator) for iterator in iterators]
            except StopIteration:
                break
            df = pd.concat(chunks, axis=1)
            yield df if columns is None else df[columns]


@typechecked
class MergedTabularDataReader(TabularDataReader):
    def __init__(self, readers: list[TabularDataReader], priority_column: str,
                 descending: bool = True, reader_chunk_size: int = 10):
        self.readers = readers
        self.priority_column = priority_column
        self.descending = descending
        self.reader_chunk_size = reader_chunk_size

        assert len(readers) > 0, "At least one data reader is required"
        self.column_names = readers[0].get_column_names()
        self.column_types = readers[0].get_column_types()

        for reader in readers:
            assert reader.get_column_names() == self.column_names, "Column names do not match"
            assert reader.get_column_types() == self.column_types, "Column types do not match"
        assert priority_column in self.column_names, "Priority column not found"

    def get_column_names(self) -> list[str]:
        return self.column_names

    def get_column_types(self) -> list:
        return self.column_types

    def get_row_iterator(self, columns: list[str] | None = None) -> Generator[
        pd.DataFrame, None, None]:
        chunk_iterators = [
            reader.get_chunked_data_iterator(chunk_size=self.reader_chunk_size,
                                             columns=columns)
            for reader in self.readers]
        chunk_dfs = [next(chunk_iterator) for chunk_iterator in chunk_iterators]

        chunk_lengths = [len(df) for df in chunk_dfs]
        chunk_row_indices = [0 for _ in chunk_dfs]
        values = [df[self.priority_column].iloc[0] for df in chunk_dfs]
        while len(chunk_iterators):
            if self.descending:
                chunk_index = np.argmax(values)
            else:
                chunk_index = np.argmin(values)
            row = chunk_dfs[chunk_index].iloc[[chunk_row_indices[chunk_index]]]
            yield row
            chunk_row_indices[chunk_index] += 1
            try:
                if chunk_row_indices[chunk_index] == chunk_lengths[chunk_index]:
                    chunk_dfs[chunk_index] = next(chunk_iterators[chunk_index])
                    chunk_lengths[chunk_index] = len(chunk_dfs[chunk_index])
                    chunk_row_indices[chunk_index] = 0

                new_value = chunk_dfs[chunk_index][self.priority_column].iloc[
                    chunk_row_indices[chunk_index]]
                if self.descending and new_value > values[chunk_index]:
                    raise ValueError(
                        f"Value {new_value} exceeds {self.priority_column} but should be descending")
                if not self.descending and new_value < values[chunk_index]:
                    raise ValueError(
                        f"Value {new_value} lower than {self.priority_column} but should be ascending")

                values[chunk_index] = new_value
            except StopIteration:
                del chunk_iterators[chunk_index]
                del chunk_dfs[chunk_index]
                del chunk_lengths[chunk_index]
                del chunk_row_indices[chunk_index]
                del values[chunk_index]

    def get_chunked_data_iterator(self, chunk_size: int,
                                  columns: list[str] | None = None) -> \
            Generator[pd.DataFrame, None, None]:

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
                yield pd.concat(rows)
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
def merge_readers(readers: list[TabularDataReader], priority_column: str,
                  descending: bool = True, reader_chunk_size: int = 10):
    reader = MergedTabularDataReader(readers, priority_column, descending,
                                     reader_chunk_size=reader_chunk_size)
    iterator = reader.get_chunked_data_iterator(chunk_size=1)
    return iterator
