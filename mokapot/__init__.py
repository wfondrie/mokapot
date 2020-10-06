"""
Initialize the mokapot package.
"""
from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass

from mokapot.dataset import LinearPsmDataset
from mokapot.model import Model, PercolatorModel, save_model, load_model
from mokapot.brew import brew
from mokapot.parsers import read_pin, read_percolator
from mokapot.confidence import LinearConfidence, plot_qvalues
from mokapot.proteins import make_decoys, digest, FastaProteins
