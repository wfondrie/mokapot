"""Mixin classes."""
import numpy as np


class RngMixin:
    """Add an Generator property to a class."""

    @property
    def rng(self) -> np.random.Generator:
        """The random number generator."""
        return self._rng

    @rng.setter
    def rng(self, rng: int | np.random.Generator) -> None:
        """Set the random number generator."""
        self._rng = np.random.default_rng(rng)
