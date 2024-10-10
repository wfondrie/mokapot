import warnings
from pathlib import Path

import numpy as np
from typeguard import typechecked

from mokapot.tabular_data import (
    BufferType,
    BufferedWriter,
    ColumnMappedReader,
    ColumnSelectReader,
    CSVFileReader,
    CSVFileWriter,
    ParquetFileReader,
    ParquetFileWriter,
    SqliteWriter,
    TabularDataReader,
    TabularDataWriter,
)

CSV_SUFFIXES = [".csv", ".pin", ".tab", ".csv"]
PARQUET_SUFFIXES = [".parquet"]
SQLITE_SUFFIXES = [".db"]


@typechecked
def reader_from_path(
    file_name: Path,
    column_map: dict[str, str] | None = None,
    only_columns: list[str] | None = None,
    **kwargs,
) -> TabularDataReader:
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

    if only_columns is not None:
        reader = ColumnSelectReader(reader, only_columns)

    if column_map is not None:
        reader = ColumnMappedReader(reader, column_map)

    return reader


def writer_from_suffix(
    file_name: Path,
    columns: list[str],
    column_types: list[np.dtype],
    buffer_size: int = 0,
    buffer_type: BufferType = BufferType.DataFrame,
    **kwargs,
) -> TabularDataWriter:
    suffix = file_name.suffix
    if suffix in CSV_SUFFIXES:
        writer = CSVFileWriter(file_name, columns, column_types, **kwargs)
    elif suffix in PARQUET_SUFFIXES:
        writer = ParquetFileWriter(file_name, columns, column_types, **kwargs)
    elif suffix in SQLITE_SUFFIXES:
        writer = SqliteWriter(file_name, columns, column_types, **kwargs)
    else:  # Fallback
        warnings.warn(
            f"Suffix '{suffix}' not recognized in file name '{file_name}'."
            " Falling back to CSV..."
        )
        writer = CSVFileWriter(file_name, columns, column_types, **kwargs)

    if buffer_size > 1:
        writer = BufferedWriter(writer, buffer_size, buffer_type)
    return writer
