from pathlib import Path


class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __str__(self):
        return f"[{self.start}, {self.end}]"

    def __repr__(self):
        return f"[{self.start}, {self.end}]"

    def __contains__(self, value):
        try:
            return self.start <= value <= self.end
        except RuntimeError:
            return False


def count_lines(path: Path, *args):
    """Count the number of lines in a file.

    Parameters
    ----------
    path : Path
        The path to the file.

    Returns
    -------
    int
        The number of lines in the file.
    """
    path = Path(path, *args)
    if not path.is_file():
        return None
    with open(path, "r") as file:
        lines = file.readlines()
    return len(lines)


class FileCheck:
    def __init__(self, dir_path: Path, ext_path, min, max):
        self.passed, self.message = FileCheck._check(
            dir_path, ext_path, min, max
        )

    @staticmethod
    def _check(dir_path, ext_path, min, max):
        """Check whether a file exists and has the correct length.

        Note: if `min` is set to zero or less, it is checked that the file
        does *not* exist.

        Parameters
        ----------
        dir_path : str
            The directory path where the file is located.
        ext_path : str
            The file extension path.
        min : int or None
            The minimum number of lines expected in the file. If set to None,
            there is no minimum limit. If smaller or equal to zero, the file
            is required to NOT exist.
        max : int or None
            The maximum number of lines expected in the file. If set to None,
            there is no maximum limit.

        Returns
        ----------
        tuple
            A tuple containing a boolean value and an error message string or
            success message string.
        """
        path = Path(dir_path, ext_path)
        if min is not None and min <= 0:
            if path.is_file():
                return False, f"File `{ext_path}` does not exist (but it did)"
            else:
                return True, f"File `{ext_path}` does not exist as it should"

        if not path.is_file():
            return False, f"File `{ext_path}` exists (but it didn't)"

        if min is not None:
            line_count = count_lines(path)
            if max is not None:
                if line_count < min or line_count > max:
                    return (
                        False,
                        f"Line count of `{ext_path}` in [{min}, {max}] (but was {line_count})",  # noqa: E501
                    )
            elif line_count < min:
                return (
                    False,
                    f"Line count of `{ext_path}` at least {min} (but was {line_count})",  # noqa: E501
                )

        return True, f"File `{ext_path}` exists and is ok."

    def __bool__(self):
        return self.passed

    def __repr__(self):
        return self.message

    def __str__(self):
        return self.message


def file_check(
    file_path, ext_path, expected_lines=None, min=None, max=None, diff=100
):
    if expected_lines is not None:
        min, max = expected_lines - diff, expected_lines + diff
        if min < 1 and expected_lines > 0:
            min = 1
    return FileCheck(file_path, ext_path, min, max)


def file_exist(file_path, ext_path):
    return file_check(file_path, ext_path)


def file_missing(file_path, ext_path):
    return file_check(file_path, ext_path, min=0)


def file_min_len(file_path, ext_path, length):
    return file_check(file_path, ext_path, min=length)


def file_exact_len(file_path, ext_path, length):
    return file_check(file_path, ext_path, min=length, max=length)


def file_approx_len(file_path, ext_path, length, diff=100):
    return file_check(file_path, ext_path, expected_lines=length, diff=diff)
