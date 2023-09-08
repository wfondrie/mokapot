"""Initialize the mokapot package."""
from .confidence import PsmConfidence
from .dataset import PsmDataset
from .parsers.fasta import digest, make_decoys, read_fasta
from .schema import PsmSchema
from .version import _get_version

# from .model import Model, PercolatorModel, save_model, load_model
# from .brew import brew
# from .parsers.pepxml import read_pepxml
# from .writers import to_flashlfq, to_txt


__version__ = _get_version()
