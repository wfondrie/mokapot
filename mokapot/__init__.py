"""Initialize the mokapot package."""
from .confidence import PsmConfidence
from .dataset import PsmDataset
from .model import Model, PercolatorModel, load_model, save_model
from .parsers.fasta import digest, make_decoys, read_fasta
from .parsers.pin import percolator_to_df, read_pin
from .schema import PsmSchema
from .version import _get_version
from .writers import to_csv, to_flashlfq, to_parquet, to_txt

# from .brew import brew
# from .parsers.pepxml import read_pepxml


__version__ = _get_version()
