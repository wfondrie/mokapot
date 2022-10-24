from __future__ import annotations

import sys

if sys.version_info >= (3, 10):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points

from argparse import _ArgumentGroup
from typing import Any

from mokapot.config import Config
from mokapot.model import Model


def get_plugins() -> dict[str, Any]:
    """Return a dict of all installed Plugins as {name: EntryPoint}."""

    plugins = entry_points(group="mokapot.plugins")

    pluginmap = {}
    for plugin in plugins:
        pluginmap[plugin.name] = plugin

    for k, v in pluginmap.items():
        # print(f"loading {k}")
        pluginmap[k] = v.load()
    return pluginmap


class BasePlugin:
    def add_arguments(parser: _ArgumentGroup) -> None:
        pass

    def process_data(self, data, config: Config) -> Model:
        return data

    def get_model(self, config: Config) -> Model:
        pass
