from __future__ import annotations

import logging
from argparse import _ArgumentGroup
from importlib.metadata import entry_points
from typing import Any

import polars as pl

from ..config import Config
from ..dataset import PsmDataset
from ..model import Model

LOGGER = logging.getLogger(__name__)


def get_plugins() -> dict[str, Any]:
    """Return a dict of all installed Plugins as {name: EntryPoint}."""
    plugins = entry_points(group="mokapot.plugins")

    pluginmap = {}
    for plugin in plugins:
        LOGGER.debug("loading %s", plugin.name)
        pluginmap[plugin.name] = plugin.load()

    return pluginmap


class BasePlugin:
    """A class on which to base plugins."""

    def add_arguments(self, parser: _ArgumentGroup) -> None:
        """Add arguments to the parser for this plugin.

        Parameters
        ----------
        parser : argparse._ArgumentGroup
            The parser group assigned to this plugin by mokapot,
            use parser.add_argument() to add arguments. to add arguments.

        """
        pass

    def process_data(
        self, data: PsmDataset | pl.DataFrame, config: Config
    ) -> Model:
        """Process the data before using it to train a model."""
        return data

    def get_model(self, config: Config) -> Model:
        """Return a model to be trained.

        The model returned should be an instance of mokapot.model.Model.
        Please check that class for the requirements of the model.
        """
        pass
