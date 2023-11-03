"""Get the version information."""
from importlib.metadata import version


def _get_version() -> str:
    """Return the version information for mokapot."""
    return version("mokapot")
