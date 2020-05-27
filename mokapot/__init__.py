"""
Initialize the mokapot package.
"""
from pkg_resources import get_distribution

__version__ = get_distribution(__name__).version

from mokapot.dataset import LinearPsmDataset
from mokapot.model import Model
from mokapot.brew import brew
