"""Initialize the mokapot package."""

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version(__name__)
    except PackageNotFoundError:
        pass

except ImportError:
    from pkg_resources import DistributionNotFound, get_distribution

    try:
        __version__ = get_distribution(__name__).version
    except DistributionNotFound:
        pass

# Here the dataset has to be imported first
from .dataset import LinearPsmDataset, OnDiskPsmDataset  # noqa: I001
from .brew import brew
from .confidence import Confidence, assign_confidence
from .model import Model, PercolatorModel, load_model, save_model
from .parsers.fasta import digest, make_decoys, read_fasta
from .parsers.pepxml import read_pepxml
from .parsers.pin import read_percolator, read_pin
from .writers.flashlfq import to_flashlfq

__all__ = [
    "brew",
    "Confidence",
    "assign_confidence",
    "LinearPsmDataset",
    "OnDiskPsmDataset",
    "Model",
    "PercolatorModel",
    "load_model",
    "save_model",
    "read_fasta",
    "read_pepxml",
    "read_percolator",
    "read_pin",
    "to_flashlfq",
    "digest",
    "make_decoys",
    "__version__",
]
