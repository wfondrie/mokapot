"""
Initialize the mokapot package.
"""
import sys

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

from .dataset import LinearPsmDataset
from .model import Model, PercolatorModel, save_model, load_model
from .brew import brew
from .parsers import read_pin, read_percolator, read_pepxml
from .writers import to_flashlfq
from .confidence import LinearConfidence, plot_qvalues
from .proteins import make_decoys, digest, FastaProteins
