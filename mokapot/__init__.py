"""Initialize the mokapot package."""

try:
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version(__name__)
    except PackageNotFoundError:
        pass

except ImportError:
    from pkg_resources import get_distribution, DistributionNotFound

    try:
        __version__ = get_distribution(__name__).version
    except DistributionNotFound:
        pass

from .dataset import LinearPsmDataset, OnDiskPsmDataset
from .model import Model, PercolatorModel, save_model, load_model
from .brew import brew
from .parsers.pin import read_pin, read_percolator
from .parsers.pepxml import read_pepxml
from .parsers.fasta import read_fasta, make_decoys, digest
from .confidence import Confidence, assign_confidence
from .writers.flashlfq import to_flashlfq
