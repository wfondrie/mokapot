"""
Initialize the mokapot package.
"""
from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass

from .dataset import LinearPsmDataset
from .model import Model, PercolatorModel, save_model, load_model
from .brew import brew
from .parsers import read_pin, read_percolator, read_pepxml
from .confidence import LinearConfidence, plot_qvalues
from .proteins import make_decoys, digest, FastaProteins
