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


# from .model import Model, PercolatorModel, save_model, load_model
# from .brew import brew
# from .parsers.pepxml import read_pepxml
# from .parsers.fasta import read_fasta, make_decoys, digest
# from .writers import to_flashlfq, to_txt
# from .confidence import LinearConfidence, plot_qvalues
