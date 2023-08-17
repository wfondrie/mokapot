"""Get the version information."""
from importlib.metadata import PackageNotFoundError, version


def _get_version() -> str:
    """Return the version information for deptcharge."""
    try:
        return version(__name__)
    except PackageNotFoundError:
        return None
