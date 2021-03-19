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

from .dataset import LinearPsmDataset
from .model import Model, PercolatorModel, save_model, load_model
from .brew import brew
from .parsers.pin import read_pin, read_percolator
from .parsers.pepxml import read_pepxml
from .parsers.fasta import read_fasta, make_decoys, digest
from .writers import to_flashlfq, to_txt
from .confidence import LinearConfidence, plot_qvalues
