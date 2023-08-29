"""Base classes used in mokapot"""
import numpy as np
import polars as pl

from . import utils
from .schema import PsmSchema
from .proteins import Proteins


class BaseData:
    """A base for mokapot classes handling data.

    This base class verifies that data is a polars LazyFrame,
    validates the schema, and defines attributes and properies
    used throughout the classes in mokapot.

    Parameters
    ----------
    data : polars.DataFrame, polars.LazyFrame, or pandas.DataFrame
        A collection of PSMs, where the rows are PSMs and the columns are
        features or metadata describing them.
    schema : mokapot.PsmSchema
        The meaning of the columns in the data.
    proteins : mokapot.Proteins
        The proteins to use for protein-level confidence estimation. This
        may be created with :py:func:`mokapot.read_fasta()`.
    eval_fdr : float
        The false discovery rate threshold for choosing the best feature and
        creating positive labels during the trainging procedure.
    rng : int or np.random.Generator
        A seed or generator used for cross-validation split creation and to
        break ties, or :code:`None` to use the default random number
        generator state.
    unit : str
        The unit to use in logging messages.

    Attributes
    ----------
    data : polars.LazyFrame
    columns : list of str
    targets : numpy.ndarray
    proteins : mokapot.Proteins
    rng : numpy.random.Generator

    """
    def __init__(
            self,
            data: pl.DataFrame | pl.LazyFrame | dict,
            schema: PsmSchema,
            proteins: Proteins | None,
            eval_fdr: float,
            rng: float | None,
            unit: str,
    ) -> None:
        """Initialize the DataBase"""
        self.rng = rng
        self.schema = schema
        self.eval_fdr = eval_fdr
        self.proteins = proteins

        # Private:
        self._unit = unit
        self._len = None  # We cache this for speed.

        # Try and read data.
        # This should work with pl.DataFrame, pl.LazyFrame,
        # pd.DataFrame, or dictionary.
        self._data = utils.make_lazy(data)
        self._perc_style_targets = schema.validate(self._data)


    @property
    def targets(self) -> np.ndarray:
        """A :py:class:`numpy.ndarray` indicating whether each row is a target."""
        if self._perc_style_targets:
            expr = pl.col(self.schema.target) + 1
        else:
            expr = pl.col(self.schema.target)

        return (
            self.data.select(expr.cast(pl.Boolean))
            .collect(streaming=True)
            .to_numpy()
            .flatten()
        )

    def __len__(self) -> int:
        """The number of examples"""
        if self._len is None:
            self._len = (
                self._data.select(pl.count()).collect(streaming=True).item()
            )

        return self._len

    @property
    def data(self) -> pl.LazyFrame:
        """The underyling data."""
        return self._data

    @property
    def columns(self) -> list[str]:
        """The columns in the underlying data."""
        return self.data.columns

    @property
    def rng(self) -> np.random.Generator:
        """The random number generator."""
        return self._rng

    @rng.setter
    def rng(self, rng: int | np.random.Generator) -> None:
        """Set the random number generator"""
        self._rng = np.random.default_rng(rng)

    @property
    def proteins(self) -> Proteins | None:
        """Has a FASTA file been added?"""
        return self._proteins

    @proteins.setter
    def proteins(self, proteins: Proteins | None) -> None:
        """Add protein information to the dataset."""
        if proteins is None or isinstance(proteins, Proteins):
            self._proteins = proteins
            return

        raise ValueError(
            "Proteins must be a 'Proteins' object, such "
            "as that returned by 'mokapot.read_fasta()'."
        )
