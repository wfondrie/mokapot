from __future__ import annotations

import sys
import logging

import pandas as pd

try:
    from importlib.metadata import entry_points
except ImportError:
    from importlib_metadata import entry_points

from argparse import _ArgumentGroup
from typing import Any

from ..config import Config
from ..model import Model
from ..dataset import LinearPsmDataset

LOGGER = logging.getLogger(__name__)


def get_plugins() -> dict[str, Any]:
    """Return a dict of all installed Plugins as {name: EntryPoint}."""

    plugins = entry_points(group="mokapot.plugins")

    pluginmap = {}
    for plugin in plugins:
        pluginmap[plugin.name] = plugin

    for k, v in pluginmap.items():
        LOGGER.debug(f"loading {k}")
        pluginmap[k] = v.load()
    return pluginmap


class BasePlugin:
    def add_arguments(parser: _ArgumentGroup) -> None:
        """Add arguments to the parser for this plugin.

        Parameters
        ----------

        parser : argparse._ArgumentGroup
            The parser group assigned to this plugin by mokapot,
            use parser.add_argument() to add arguments. to add arguments.

        """
        pass

    def process_data(
        self, data: LinearPsmDataset | pd.DataFrame, config: Config
    ) -> Model:
        """Process the data before using it to train a model."""
        return data

    def get_model(self, config: Config) -> Model:
        """Return a model to be trained.

        The model returned should be an instance of mokapot.model.Model.
        Please check that class for the requirements of the model.
        """
        pass
